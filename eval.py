"""
eval.py
-------
Final evaluation on the test set, metric logging to W&B, and saving the
classification report as a versioned W&B Artifact.

Usage
-----
    # Run after train.py has saved a checkpoint to --model_dir
    python eval.py [options]

Required environment variables
-------------------------------
    WANDB_API_KEY  — from https://wandb.ai/settings

Options
-------
    --model_dir     Path to the saved model/tokenizer  (default: ./results)
    --data_dir      Directory written by data.py        (default: ./data_cache)
    --output_dir    Where eval_report.json is saved     (default: ./results)
    --max_length    Max tokenisation length             (default: 256)
    --eval_batch    Per-device eval batch size          (default: 32)
    --wandb_project W&B project name                    (default: distilbert-goodreads-genres)
    --wandb_entity  W&B entity/team name                 (default: maheshvgv-mahesh)
    --wandb_run     W&B run name for the eval run       (default: distilbert-eval)
    --seed          Random seed                         (default: 42)
"""

from __future__ import annotations

import argparse
import json
import os

import torch
import wandb
from sklearn.metrics import classification_report
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)

from data import load_splits
from utils import GoodreadsDataset, NUM_LABELS, compute_metrics, id2label, label2id


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(args: argparse.Namespace) -> dict:
    """
    Load a fine-tuned model, run evaluation on the test set, log metrics to
    W&B, and upload the classification report as a W&B Artifact.

    Parameters
    ----------
    args : Parsed CLI arguments (see parse_args).

    Returns
    -------
    dict containing eval_loss, eval_accuracy, and eval_f1.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load model and tokenizer ---
    print(f"Loading model from: {args.model_dir}")
    tokenizer = DistilBertTokenizerFast.from_pretrained(args.model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(
        args.model_dir,
        num_labels=NUM_LABELS,
        id2label=id2label,
        label2id=label2id,
    ).to(device)

    # --- Load test data ---
    _, _, test_texts, test_labels = load_splits(args.data_dir)
    test_dataset = GoodreadsDataset(test_texts, test_labels, tokenizer, args.max_length)
    print(f"Test samples: {len(test_dataset):,}")

    # --- Minimal TrainingArguments for inference (no training) ---
    eval_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_eval_batch_size=args.eval_batch,
        report_to="none",    # W&B logging done manually below
        seed=args.seed,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # --- Run evaluation ---
    print("Running evaluation on test set...")
    eval_results = trainer.evaluate(eval_dataset=test_dataset)

    final_accuracy = eval_results.get("eval_accuracy", 0.0)
    final_f1       = eval_results.get("eval_f1",       0.0)
    final_loss     = eval_results.get("eval_loss",     0.0)

    print("=" * 42)
    print("     FINAL EVALUATION METRICS")
    print("=" * 42)
    print(f"  Accuracy:  {final_accuracy:.4f}  ({final_accuracy * 100:.2f}%)")
    print(f"  F1 Score:  {final_f1:.4f}")
    print(f"  Eval Loss: {final_loss:.4f}")
    print("=" * 42)

    # --- W&B: log final metrics explicitly ---
    wandb.login(key=os.environ.get("WANDB_API_KEY", ""))
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run,
        config={
            "model_dir":   args.model_dir,
            "max_length":  args.max_length,
            "num_labels":  NUM_LABELS,
        },
    )

    wandb.log({
        "final/loss":     final_loss,
        "final/accuracy": final_accuracy,
        "final/f1":       final_f1,
    })

    wandb.run.summary["best_accuracy"] = final_accuracy
    wandb.run.summary["best_f1"]       = final_f1
    wandb.run.summary["best_loss"]     = final_loss

    print("Final metrics logged to W&B.")

    # --- Per-class classification report ---
    pred_output = trainer.predict(test_dataset)
    pred_labels = pred_output.predictions.argmax(axis=-1)
    true_labels = [test_dataset[i]["labels"].item() for i in range(len(test_dataset))]

    print("\nPer-class Classification Report:")
    print(
        classification_report(
            true_labels,
            pred_labels,
            target_names=[id2label[i] for i in range(NUM_LABELS)],
            zero_division=0,
        )
    )

    report_dict = classification_report(
        true_labels,
        pred_labels,
        target_names=[id2label[i] for i in range(NUM_LABELS)],
        output_dict=True,
        zero_division=0,
    )

    # --- Save report as JSON ---
    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.join(args.output_dir, "eval_report.json")
    with open(report_path, "w") as f:
        json.dump(report_dict, f, indent=2)
    print(f"Classification report saved to: {report_path}")

    # --- Upload as W&B Artifact ---
    artifact = wandb.Artifact(
        name="eval-report",
        type="evaluation",
        description="Per-class classification report from final test set evaluation.",
        metadata={
            "accuracy": final_accuracy,
            "f1":       final_f1,
            "loss":     final_loss,
        },
    )
    artifact.add_file(report_path)
    wandb.log_artifact(artifact)
    print("Artifact uploaded to W&B.")

    wandb.finish()
    return eval_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned model and log results to W&B."
    )
    parser.add_argument("--model_dir",      type=str, default="./results")
    parser.add_argument("--data_dir",       type=str, default="./data_cache")
    parser.add_argument("--output_dir",     type=str, default="./results")
    parser.add_argument("--max_length",     type=int, default=256)
    parser.add_argument("--eval_batch",     type=int, default=32)
    parser.add_argument("--wandb_project",  type=str, default="distilbert-goodreads-genres")
    parser.add_argument("--wandb_entity",   type=str, default="maheshvgv-mahesh")
    parser.add_argument("--wandb_run",      type=str, default="distilbert-eval")
    parser.add_argument("--seed",           type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(args)
    print("eval.py completed successfully.")


if __name__ == "__main__":
    main()
