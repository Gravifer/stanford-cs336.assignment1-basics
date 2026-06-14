#!/usr/bin/env python3
import os
import pickle
import random

import numpy as np

from cs336_basics.common import batched_line_feed
from cs336_basics.tokenizer import BPETokenizer

vocabs: list[dict[int, bytes]] = []
for out_vocab in ["vocab-tinystories.pkl", "vocab-owt.pkl"]:
    with open(out_vocab, "rb") as f:
        vocab: dict[int, bytes] = pickle.load(f)
    print(f"Loaded vocab from {out_vocab}, size: {len(vocab)}")
    vocabs.append(vocab)

merges: list[list[tuple[bytes, bytes]]] = []
for out_merges in ["merges-tinystories.pkl", "merges-owt.pkl"]:
    with open(out_merges, "rb") as f:
        merge: list[tuple[bytes, bytes]] = pickle.load(f)
    print(f"Loaded merges from {out_merges}, length: {len(merge)}")
    merges.append(merge)

# cross test the two tokenizers on 10 document samples from both datasets, and compute compression ratios

SEED = 114514

documents: list[list[str]] = []
with open("data/TinyStoriesV2-GPT4-valid.txt") as f:
    # split on special token
    docs = f.read().split("<|endoftext|>")
random.seed(SEED)
documents.append(random.sample(docs, 10))
with open("data/owt_valid.txt") as f:
    docs = f.read().split("<|endoftext|>")
random.seed(SEED)
documents.append(random.sample(docs, 10))

encoders = [BPETokenizer(vocab, merges_, special_tokens=["<|endoftext|>"]) for vocab, merges_ in zip(vocabs, merges)]

data_dir = "data/"
for encoder, files, name in zip(
    encoders,
    [[], ["owt_train.txt"]],
    # [["TinyStoriesV2-GPT4-valid.txt", "TinyStoriesV2-GPT4-train.txt"], ["owt_valid.txt", "owt_train.txt"]],
    ["tinystories", "owt"],
):
    for file in files:
        filepath: str = data_dir + file
        print(f"Encoding {file} with {name} encoder...")
        with open(filepath) as f:
            # get file size
            file_size: int = os.path.getsize(filepath)
            # use iterator encoding
            encoded: list[int] = list(
                encoder.encode_iterable(batched_line_feed(f), estimate_total=file_size)
            )
            # text = f.read()
        # fragments = text.split("(<|endoftext|>)")

        # cast to a numpy array
        encoded: np.ndarray = np.array(encoded, dtype=np.uint16)

        out_file: str = filepath.rsplit(".", 1)[0] + ".npy"
        np.save(out_file, encoded)
        print(f"Saved encoded data to {out_file}")
