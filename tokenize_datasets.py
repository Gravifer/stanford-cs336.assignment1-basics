#!/usr/bin/env python3
import multiprocessing
import pickle

import numpy as np

from cs336_basics.pretokenization_example import find_chunk_boundaries
from cs336_basics.tokenizer import BPETokenizer


def tokenize_file_segment_worker(
    file_path: str,
    start_byte: int,
    end_byte: int,
    vocab_path: str,
    merges_path: str,
    chunk_size_chars: int = 128 * 1024,
) -> list[int]:
    """Worker process that instantiates its own tokenizer copy and processes

    a strictly bounded byte-slice of a massive file on disk.
    """
    # Lazy-load the tokenizer inside the process worker to isolate memory domains
    from cs336_basics.tokenizer import BPETokenizer

    tokenizer: BPETokenizer = BPETokenizer.from_files(
        vocab_filepath=vocab_path, merges_filepath=merges_path, report_progress=False
    )

    local_tokens: list[int] = []

    # Open file independently inside the worker process
    with open(file_path, "rb") as f:
        f.seek(start_byte)
        bytes_to_read = end_byte - start_byte
        bytes_processed = 0

        remainder = ""

        while bytes_processed < bytes_to_read:
            # Calculate safe remaining read window
            to_read = min(chunk_size_chars, bytes_to_read - bytes_processed)
            block_bytes = f.read(to_read)
            if not block_bytes:
                break

            bytes_processed += len(block_bytes)

            # Safe unicode decoding with replacement hooks for edge protection
            text = remainder + block_bytes.decode("utf-8", errors="replace")

            # Find the absolute last newline in this string window
            last_newline = text.rfind("\n")
            if last_newline != -1:
                chunk_to_encode = text[: last_newline + 1]
                remainder = text[last_newline + 1 :]
            else:
                # If no newline in block, pass it forward or buffer it
                chunk_to_encode = text
                remainder = ""

            if chunk_to_encode:
                # Process via your high-performance cache/heap pipeline
                local_tokens.extend(tokenizer.encode(chunk_to_encode, report_progress=False))

        # Flush dangling text residues at chunk boundary
        if remainder:
            local_tokens.extend(tokenizer.encode(remainder, report_progress=False))

    return local_tokens


def parallel_tokenize_pipeline(
    file_path: str, vocab_path: str, merges_path: str, split_token_bytes: bytes = b"\n", num_workers: int | None = None
):
    """Orchestrates multi-core tokenization of a single massive file."""
    if num_workers is None:
        num_workers: int = multiprocessing.cpu_count()

    print(f"Analyzing file boundaries for {num_workers} processes...")
    with open(file_path, "rb") as f:
        boundaries: list[int] = find_chunk_boundaries(
            f, desired_num_chunks=num_workers, split_special_token=split_token_bytes
        )

    tasks: list[tuple] = []
    for i in range(len(boundaries) - 1):
        tasks.append((file_path, boundaries[i], boundaries[i + 1], vocab_path, merges_path))

    print(f"Spawning {len(tasks)} parallel workers...")
    with multiprocessing.Pool(processes=len(tasks)) as pool:
        # Map tasks asynchronously across cores
        results: list[list[int]] = pool.starmap(tokenize_file_segment_worker, tasks)

    # Reassemble tokens in physical structural sequence
    print("Reassembling tokens from worker pools...")
    final_token_stream: list[int] = []
    for local_tokens in results:
        final_token_stream.extend(local_tokens)

    return final_token_stream


if __name__ == "__main__":
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

    encoders: list[BPETokenizer] = [
        BPETokenizer(vocab, merges_, special_tokens=["<|endoftext|>"]) for vocab, merges_ in zip(vocabs, merges)
    ]

    data_dir = "data/"
    for encoder, files, name in zip(
        encoders,
        # [[], ["owt_train.txt"]],
        [["TinyStoriesV2-GPT4-valid.txt", "TinyStoriesV2-GPT4-train.txt"], ["owt_valid.txt", "owt_train.txt"]],
        ["tinystories", "owt"],
    ):
        for file in files:
            filepath: str = data_dir + file
            print(f"Encoding {file} with {name} encoder...")
            # with open(filepath) as f:
            #     # get file size
            #     file_size: int = os.path.getsize(filepath)
            #     # use iterator encoding
            #     encoded: list[int] = list(encoder.encode_iterable(batched_line_feed(f), estimate_total=file_size))
            #     # text = f.read()
            # # fragments = text.split("(<|endoftext|>)")

            # move to parallelized pipeline
            encoded: list[int] = parallel_tokenize_pipeline(
                file_path=filepath,
                vocab_path=f"vocab-{name}.pkl",
                merges_path=f"merges-{name}.pkl",
                split_token_bytes=b"\n",
                num_workers=None,
            )

            # cast to a numpy array
            encoded: np.ndarray = np.array(encoded, dtype=np.uint16)

            out_file: str = filepath.rsplit(".", 1)[0] + ".npy"
            np.save(out_file, encoded)
            print(f"Saved encoded data to {out_file}")
