"""
inference.py
------------
Interactive inference script to classify custom book reviews into one of the 
8 Goodreads genres using the fine-tuned DistilBERT model.

Usage
-----
    python inference.py --model_path ./results
    OR
    python inference.py --model_path your-hf-username/distilbert-goodreads-genres
"""

import argparse
import os
import sys

import torch
from transformers import pipeline

# Emojis and display names for genres
GENRE_EMOJIS = {
    "children": "👶 Children",
    "comics_graphic": "🦸 Comics & Graphic Novels",
    "fantasy_paranormal": "🧙 Fantasy & Paranormal",
    "history_biography": "📜 History & Biography",
    "mystery_thriller_crime": "🕵️ Mystery, Thriller & Crime",
    "poetry": "✒️ Poetry",
    "romance": "💖 Romance",
    "young_adult": "🎒 Young Adult",
}

# ANSI escape codes for professional styling
RED = "\033[91m"
BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
UNDERLINE = "\033[4m"
END = "\033[0m"


def print_header(title: str) -> None:
    print(f"\n{CYAN}{BOLD}{'=' * 60}{END}")
    print(f"{CYAN}{BOLD}  {title}{END}")
    print(f"{CYAN}{BOLD}{'=' * 60}{END}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference using the fine-tuned Goodreads Genre Classifier."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./results",
        help="Path to the local model directory or a Hugging Face Hub repository identifier.",
    )
    return parser.parse_args()


def main() -> None:
    # Reconfigure stdout/stderr to use UTF-8 encoding to prevent Windows cp1252 codec errors
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    # Enable ANSI escape characters on Windows PowerShell/CMD
    if sys.platform == "win32":
        os.system("color")

    args = parse_args()

    print_header("GOODREADS GENRE CLASSIFICATION — INFERENCE WORKFLOW")
    print(f"{BLUE}[INFO]{END} Loading inference pipeline from: {BOLD}{args.model_path}{END}")
    print(f"{BLUE}[INFO]{END} This might take a few seconds on the first run...")

    try:
        # Load classification pipeline
        # Hugging Face pipeline handles tokenisation, model forwarding, and softmax automatically
        classifier = pipeline(
            "text-classification",
            model=args.model_path,
            tokenizer=args.model_path,
            device=0 if torch.cuda.is_available() else -1,
            top_k=None,  # Return scores for all classes
        )
        print(f"{GREEN}[SUCCESS]{END} Inference pipeline loaded successfully!")
        if torch.cuda.is_available():
            print(f"{GREEN}[SUCCESS]{END} Running on GPU: {torch.cuda.get_device_name(0)}")
        else:
            print(f"{YELLOW}[WARNING]{END} Running on CPU fallback.")

    except Exception as e:
        print(f"\n{BOLD}{RED}[ERROR]{END} Failed to load the model from '{args.model_path}'.")
        print(f"Detail: {e}")
        print("\nSuggestions:")
        print("1. Run training locally first using 'python train.py' to generate a local checkpoint.")
        print("2. Or pass a public Hugging Face model repository identifier:")
        print("   e.g. python inference.py --model_path mahesh-pgdai-mlops/distilbert-goodreads-genres")
        sys.exit(1)

    print(f"\n{GREEN}{BOLD}Ready for custom inputs!{END}")
    print("Type your book review text below and hit Enter. To quit, type 'exit' or 'quit'.")

    while True:
        try:
            print(f"\n{BOLD}Enter a Book Review:{END}")
            text = input("> ").strip()

            if not text:
                continue

            if text.lower() in ("exit", "quit", "q"):
                print(f"\n{BLUE}[INFO]{END} Exiting inference workflow. Happy reading! 📚\n")
                break

            if len(text) < 15:
                print(f"{YELLOW}[WARNING]{END} Review is too short. Please provide a more descriptive review for better classification.")
                continue

            # Run inference
            print(f"{BLUE}[INFO]{END} Classifying review...")
            predictions = classifier(text)[0]

            # Sort predictions by score descending
            sorted_preds = sorted(predictions, key=lambda x: x["score"], reverse=True)

            print(f"\n{CYAN}{BOLD}--- PREDICTION RESULTS ---{END}")
            # Display sorted results with horizontal bar graphs
            for i, pred in enumerate(sorted_preds):
                label = pred["label"]
                score = pred["score"]
                percentage = score * 100
                
                # Format visual bar chart (max width 30 blocks)
                bar_len = int(score * 30)
                bar = "█" * bar_len + "░" * (30 - bar_len)
                
                emoji_name = GENRE_EMOJIS.get(label, label)
                
                # Style top prediction with green bold text
                if i == 0:
                    print(f"  {GREEN}{BOLD}*{emoji_name:<28}{END} | {GREEN}{bar}{END} | {GREEN}{BOLD}{percentage:6.2f}%{END}")
                else:
                    print(f"   {emoji_name:<28} | {bar} | {percentage:6.2f}%")
            print(f"{CYAN}{BOLD}{'-' * 60}{END}")

        except KeyboardInterrupt:
            print(f"\n\n{BLUE}[INFO]{END} Exiting inference workflow. Happy reading! 📚\n")
            break
        except EOFError:
            print(f"\n{BLUE}[INFO]{END} Standard input closed. Exiting inference workflow. Happy reading! 📚\n")
            break
        except Exception as e:
            print(f"{BOLD}{RED}[ERROR]{END} An error occurred during inference: {e}")


if __name__ == "__main__":
    main()
