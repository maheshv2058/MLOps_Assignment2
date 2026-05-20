"""
train.py
--------
Model loading, Weights & Biases initialisation, Trainer configuration,
and the training loop.

Usage
-----
    python train.py [options]

Required environment variables
-------------------------------
    WANDB_API_KEY  — from https://wandb.ai/settings
    HF_TOKEN       — from https://huggingface.co/settings/tokens

Options
-------
    --data_dir        Directory written by data.py  (default: ./data_cache)
    --output_dir      Where checkpoints are saved   (default: ./results)
    --model_name      HuggingFace model id           (default: distilbert-base-cased)
    --epochs          Number of training epochs      (default: 3)
    --batch_size      Per-device train batch size    (default: 16)
    --eval_batch      Per-device eval batch size     (default: 32)
    --lr              Learning rate                  (default: 3e-5)
    --warmup_steps    LR scheduler warm-up steps     (default: 100)
    --weight_decay    AdamW weight decay             (default: 0.01)
    --max_length      Max tokenisation length        (default: 256)
    --wandb_project   W&B project name               (default: distilbert-goodreads-genres)
    --wandb_entity    W&B entity/team name            (default: maheshvgv-mahesh)
    --wandb_run       W&B run name                   (default: distilbert-run-1)
    --seed            Random seed                    (default: 42)
    --no_fp16         Disable mixed-precision even on GPU
"""

from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch
import wandb
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from data import build_datasets, load_splits
from utils import NUM_LABELS, compute_metrics, id2label, label2id


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def load_model(
    model_name: str,
    device: torch.device,
) -> DistilBertForSequenceClassification:
    """
    Load DistilBERT (or any compatible model) with the correct classification head.

    Parameters
    ----------
    model_name : HuggingFace model identifier.
    device     : torch.device to move the model to.

    Returns
    -------
    Model ready for fine-tuning.
    """
    print(f"Loading model: {model_name}  |  num_labels={NUM_LABELS}")
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
        id2label=id2label,
        label2id=label2id,
    ).to(device)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def run_training(args: argparse.Namespace) -> Trainer:
    """
    Full training pipeline: seed -> data -> model -> W&B -> Trainer.train().

    Parameters
    ----------
    args : Parsed CLI arguments (see parse_args).

    Returns
    -------
    Trainer instance after training (holds the best checkpoint).
    """
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU:    {torch.cuda.get_device_name(0)}")

    # --- Data ---
    train_texts, train_labels, test_texts, test_labels = load_splits(args.data_dir)
    tokenizer = DistilBertTokenizerFast.from_pretrained(args.model_name)

    from utils import GoodreadsDataset
    train_dataset = GoodreadsDataset(train_texts, train_labels, tokenizer, args.max_length)
    test_dataset  = GoodreadsDataset(test_texts,  test_labels,  tokenizer, args.max_length)

    print(f"Train samples: {len(train_dataset):,}")
    print(f"Test samples:  {len(test_dataset):,}")

    # --- Model ---
    model = load_model(args.model_name, device)

    # --- W&B ---
    wandb.login(key=os.environ.get("WANDB_API_KEY", ""))
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run,
        config={
            "model":          args.model_name,
            "epochs":         args.epochs,
            "batch_size":     args.batch_size,
            "learning_rate":  args.lr,
            "max_length":     args.max_length,
            "warmup_steps":   args.warmup_steps,
            "weight_decay":   args.weight_decay,
            "dataset":        "UCSD Goodreads",
            "platform":       "local / Kaggle",
            "num_labels":     NUM_LABELS,
            "train_samples":  len(train_dataset),
            "test_samples":   len(test_dataset),
            "seed":           args.seed,
        },
        tags=["distilbert", "text-classification", "goodreads"],
    )
    print(f"W&B run: {run.url}")

    # --- Training Arguments ---
    use_fp16 = (not args.no_fp16) and torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=args.output_dir,

        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        learning_rate=args.lr,

        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=50,

        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,

        report_to="wandb",          # single line enables full W&B logging
        run_name=args.wandb_run,

        seed=args.seed,
        fp16=use_fp16,
    )

    print(f"Training Arguments:")
    print(f"  Epochs:      {training_args.num_train_epochs}")
    print(f"  LR:          {training_args.learning_rate}")
    print(f"  report_to:   {training_args.report_to}")
    print(f"  fp16:        {training_args.fp16}")

    # --- Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("\nStarting training...")
    train_result = trainer.train()
    print("\nTraining completed.")
    print(f"  Runtime:          {train_result.metrics['train_runtime']:.1f} s")
    print(f"  Samples/sec:      {train_result.metrics['train_samples_per_second']:.1f}")
    print(f"  Final train loss: {train_result.metrics['train_loss']:.4f}")

    # Save the best model weights and the tokenizer to the output directory root
    print(f"Saving final model and tokenizer to: {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    return trainer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune DistilBERT on Goodreads genre classification."
    )
    parser.add_argument("--data_dir",       type=str,   default="./data_cache")
    parser.add_argument("--output_dir",     type=str,   default="./results")
    parser.add_argument("--model_name",     type=str,   default="distilbert-base-cased")
    parser.add_argument("--epochs",         type=int,   default=3)
    parser.add_argument("--batch_size",     type=int,   default=16)
    parser.add_argument("--eval_batch",     type=int,   default=32)
    parser.add_argument("--lr",             type=float, default=3e-5)
    parser.add_argument("--warmup_steps",   type=int,   default=100)
    parser.add_argument("--weight_decay",   type=float, default=0.01)
    parser.add_argument("--max_length",     type=int,   default=256)
    parser.add_argument("--wandb_project",  type=str, default="distilbert-goodreads-genres")
    parser.add_argument("--wandb_entity",   type=str, default="maheshvgv-mahesh")
    parser.add_argument("--wandb_run",      type=str, default="distilbert-run-1")
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument(
        "--no_fp16",
        action="store_true",
        help="Disable mixed-precision training even if a GPU is available.",
    )
    return parser.parse_args()


def main() -> None:
    args   = parse_args()
    trainer = run_training(args)
    # Keep trainer available for eval.py when used as a module
    return trainer


if __name__ == "__main__":
    main()
    wandb.finish()
    print("train.py completed successfully.")
