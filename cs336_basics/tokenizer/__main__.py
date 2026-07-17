"""Command-line entry point for training a BPE tokenizer."""

import argparse
import json
import pickle
from collections.abc import Sequence

from . import train_bpe


def main(argv: Sequence[str] | None = None) -> int:
    """Train a tokenizer and serialize its vocabulary and merges."""
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer")
    parser.add_argument("input_path", help="Path to training corpus")
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--special-tokens", nargs="+", default=["<|endoftext|>"])
    parser.add_argument("--no-multiprocessing", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--out-vocab", default="vocab.json", help="Output path for vocabulary (JSON)")
    parser.add_argument("--out-merges", default="merges.pkl", help="Output path for merges (pickle)")
    args = parser.parse_args(argv)

    vocab, merges = train_bpe(
        args.input_path,
        args.vocab_size,
        args.special_tokens,
        multiprocessing=not args.no_multiprocessing,
        report_progress=not args.no_progress,
    )

    with open(args.out_vocab, "w", encoding="utf-8") as f:
        json.dump(
            {str(k): v.decode("utf-8", errors="replace") for k, v in vocab.items()}, f, ensure_ascii=False, indent=2
        )
    with open(args.out_merges, "wb") as f:
        pickle.dump(merges, f)
    print(f"Saved vocab ({len(vocab)} tokens) → {args.out_vocab}")
    print(f"Saved merges ({len(merges)}) → {args.out_merges}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
