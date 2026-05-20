"""
data.py
-------
Data loading, balanced sampling, train/test split, and dataset encoding.

Usage
-----
    python data.py

Outputs (written to --output_dir):
    train_texts.json
    train_labels.json
    test_texts.json
    test_labels.json
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import random
import sys
import warnings

import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast

from utils import GENRES, GoodreadsDataset, label2id

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants (overridable via CLI)
# ---------------------------------------------------------------------------
DATASET_URL = (
    "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/"
    "goodreads_reviews_dedup.json.gz"
)
DEFAULT_SAMPLES_PER_GENRE = 500
DEFAULT_MAX_LENGTH        = 256
DEFAULT_TEST_SIZE         = 0.2
DEFAULT_SEED              = 42
DEFAULT_MODEL_NAME        = "distilbert-base-cased"
DEFAULT_OUTPUT_DIR        = "./data_cache"


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def download_dataset(url: str, max_records_per_genre: int = 2000) -> pd.DataFrame:
    """
    Download the UCSD Goodreads reviews dataset and return as a DataFrame.

    If the url contains 'dedup', we automatically fall back to downloading
    and streaming by-genre files from the official UCSD Book Graph repository,
    since the dedup file does not contain a 'genre' column.

    Parameters
    ----------
    url                   : Direct URL or fallback indicator.
    max_records_per_genre : Number of records to stream per genre to guarantee sampling pool.

    Returns
    -------
    pd.DataFrame with 'genre' column injected.
    """
    if "dedup" in url:
        print("[INFO] Dedup dataset URL detected. Since the dedup dataset does not contain 'genre' columns,")
        print("       we are automatically streaming the by-genre datasets from Mcauley Lab.")
        
        all_dfs = []
        for genre in GENRES:
            genre_url = f"https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_{genre}.json.gz"
            print(f"Streaming genre: {genre} from:\n  {genre_url}")
            
            try:
                response = requests.get(genre_url, stream=True, timeout=30)
                response.raise_for_status()
                
                records = []
                with gzip.GzipFile(fileobj=response.raw) as gz_file:
                    for line in gz_file:
                        rec = json.loads(line)
                        if rec.get("review_text") and len(rec["review_text"].strip()) > 10:
                            rec["genre"] = genre
                            records.append(rec)
                            if len(records) >= max_records_per_genre:
                                break
                
                genre_df = pd.DataFrame(records)
                print(f"  Loaded {len(genre_df):,} records for {genre}.")
                all_dfs.append(genre_df)
            except Exception as e:
                print(f"[ERROR] Failed to stream genre {genre}: {e}")
                
        if not all_dfs:
            raise ValueError("Failed to download any genre files.")
        df = pd.concat(all_dfs, ignore_index=True)
        print(f"Successfully loaded a total of {len(df):,} records across all genres.")
        return df
    else:
        print(f"Downloading dataset from:\n  {url}")
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()

        records: list[dict] = []
        with gzip.GzipFile(fileobj=response.raw) as gz_file:
            for line in gz_file:
                records.append(json.loads(line))
                if len(records) >= 100_000:
                    break

        df = pd.DataFrame(records)
        print(f"Downloaded {len(df):,} records. Columns: {df.columns.tolist()}")
        return df


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def sample_balanced(
    df: pd.DataFrame,
    samples_per_genre: int,
    text_col: str = "review_text",
    genre_col: str = "genre",
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """
    Return a balanced subset with at most `samples_per_genre` rows per genre.

    Parameters
    ----------
    df                : Raw DataFrame from download_dataset.
    samples_per_genre : Maximum rows per genre class.
    text_col          : Column name containing review text.
    genre_col         : Column name containing genre label strings.
    seed              : Random seed for reproducibility.

    Returns
    -------
    Shuffled DataFrame with columns ['text', 'label'].
    """
    frames: list[pd.DataFrame] = []
    for genre in GENRES:
        subset = df[df[genre_col] == genre]
        n = min(samples_per_genre, len(subset))
        if n == 0:
            print(f"[WARNING] No samples found for genre: {genre}")
            continue
        sample = subset.sample(n=n, random_state=seed)[[text_col, genre_col]].copy()
        sample.rename(columns={text_col: "text"}, inplace=True)
        sample["label"] = label2id[genre]
        frames.append(sample)

    if not frames:
        raise ValueError(
            "No samples were found for any genre. "
            "Check that the genre_col and GENRES list match the dataset."
        )

    combined = (
        pd.concat(frames, ignore_index=True)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )
    combined = combined[["text", "label"]].dropna()
    combined["text"] = combined["text"].astype(str).str.strip()
    combined = combined[combined["text"].str.len() > 10].reset_index(drop=True)

    print(f"Balanced dataset: {len(combined):,} rows")
    print("Class distribution:")
    from utils import id2label  # local import to avoid circular at module level
    print(combined["label"].map(id2label).value_counts().to_string())
    return combined


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------
def split_dataset(
    df: pd.DataFrame,
    test_size: float = DEFAULT_TEST_SIZE,
    seed: int = DEFAULT_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified train/test split.

    Parameters
    ----------
    df        : Balanced DataFrame with 'text' and 'label' columns.
    test_size : Fraction of data held out for evaluation.
    seed      : Random seed.

    Returns
    -------
    (train_df, test_df)
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["label"],
    )
    print(f"Train: {len(train_df):,}  |  Test: {len(test_df):,}")
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Encode
# ---------------------------------------------------------------------------
def build_datasets(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str = DEFAULT_MODEL_NAME,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> tuple[GoodreadsDataset, GoodreadsDataset, DistilBertTokenizerFast]:
    """
    Tokenise DataFrames and return PyTorch Dataset objects.

    Parameters
    ----------
    train_df   : Training DataFrame with 'text' and 'label' columns.
    test_df    : Test DataFrame.
    model_name : HuggingFace model identifier.
    max_length : Maximum token sequence length.

    Returns
    -------
    (train_dataset, test_dataset, tokenizer)
    """
    print(f"Loading tokenizer: {model_name}")
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

    print("Tokenising train set...")
    train_dataset = GoodreadsDataset(
        train_df["text"].tolist(),
        train_df["label"].tolist(),
        tokenizer,
        max_length,
    )

    print("Tokenising test set...")
    test_dataset = GoodreadsDataset(
        test_df["text"].tolist(),
        test_df["label"].tolist(),
        tokenizer,
        max_length,
    )

    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Test dataset:  {len(test_dataset)} samples")
    return train_dataset, test_dataset, tokenizer


# ---------------------------------------------------------------------------
# Serialise raw splits for re-use across runs (optional)
# ---------------------------------------------------------------------------
def save_splits(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str,
) -> None:
    """Save text/label lists as JSON so train.py can reload without re-downloading."""
    os.makedirs(output_dir, exist_ok=True)
    for name, df in (("train", train_df), ("test", test_df)):
        with open(os.path.join(output_dir, f"{name}_texts.json"), "w") as f:
            json.dump(df["text"].tolist(), f)
        with open(os.path.join(output_dir, f"{name}_labels.json"), "w") as f:
            json.dump(df["label"].tolist(), f)
    print(f"Splits saved to: {output_dir}")


def load_splits(output_dir: str) -> tuple[list[str], list[int], list[str], list[int]]:
    """Load previously saved text/label lists."""
    def _load(fname):
        with open(os.path.join(output_dir, fname)) as f:
            return json.load(f)

    return (
        _load("train_texts.json"),
        _load("train_labels.json"),
        _load("test_texts.json"),
        _load("test_labels.json"),
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download, sample, split, and encode the Goodreads dataset."
    )
    parser.add_argument("--samples_per_genre", type=int, default=DEFAULT_SAMPLES_PER_GENRE)
    parser.add_argument("--max_length",        type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--test_size",         type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--seed",              type=int, default=DEFAULT_SEED)
    parser.add_argument("--model_name",        type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output_dir",        type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset_url",       type=str, default=DATASET_URL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    max_rec = max(args.samples_per_genre * 3, 2000)
    df_raw  = download_dataset(args.dataset_url, max_records_per_genre=max_rec)
    df      = sample_balanced(df_raw, args.samples_per_genre, seed=args.seed)
    train_df, test_df = split_dataset(df, args.test_size, args.seed)
    save_splits(train_df, test_df, args.output_dir)

    # Quick tokenisation check
    build_datasets(train_df, test_df, args.model_name, args.max_length)
    print("data.py completed successfully.")


if __name__ == "__main__":
    main()
