"""
utils.py
--------
Shared helpers used across data.py, train.py, and eval.py.

Contents
--------
- GENRES          : ordered list of genre class names
- label2id        : dict mapping genre name -> integer label
- id2label        : dict mapping integer label -> genre name
- NUM_LABELS      : total number of classes
- GoodreadsDataset: PyTorch Dataset wrapping tokenised reviews
- compute_metrics : evaluation function passed to HuggingFace Trainer
"""

from __future__ import annotations

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

# ---------------------------------------------------------------------------
# Label Maps
# ---------------------------------------------------------------------------
GENRES: list[str] = [
    "children",
    "comics_graphic",
    "fantasy_paranormal",
    "history_biography",
    "mystery_thriller_crime",
    "poetry",
    "romance",
    "young_adult",
]

label2id: dict[str, int] = {genre: idx for idx, genre in enumerate(GENRES)}
id2label: dict[int, str] = {idx: genre for genre, idx in label2id.items()}
NUM_LABELS: int = len(GENRES)


# ---------------------------------------------------------------------------
# Dataset Class
# ---------------------------------------------------------------------------
class GoodreadsDataset(Dataset):
    """
    PyTorch Dataset that tokenises Goodreads review texts on construction.

    Parameters
    ----------
    texts     : list[str]               Raw review strings.
    labels    : list[int]               Integer genre labels.
    tokenizer : PreTrainedTokenizerBase HuggingFace tokenizer instance.
    max_length: int                     Maximum token sequence length.
    """

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 256,
    ) -> None:
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(pred) -> dict[str, float]:
    """
    Compute accuracy and weighted F1 score from Trainer prediction output.

    Called automatically by the HuggingFace Trainer after each eval epoch.

    Parameters
    ----------
    pred : transformers.EvalPrediction
        Object with .label_ids (true labels) and .predictions (logits).

    Returns
    -------
    dict with keys 'accuracy' and 'f1'.
    """
    labels = pred.label_ids
    preds  = pred.predictions.argmax(axis=-1)
    return {
        "accuracy": round(accuracy_score(labels, preds), 4),
        "f1":       round(f1_score(labels, preds, average="weighted", zero_division=0), 4),
    }
