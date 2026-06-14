#!/usr/bin/env python3
import multiprocessing as mp
import os
import pickle
from queue import Queue

import numpy as np

from cs336_basics.chunk_progress_monitor import ChunkedProgressBar
from cs336_basics.pretokenization_example import find_chunk_boundaries
from cs336_basics.tokenizer import BPETokenizer


def tokenize_file_segment_worker(
    file_path: str,
    start_byte: int,
    end_byte: int,
    vocab_path: str,
    merges_path: str,
    worker_idx: int,
    progress_queue: Queue | None = None,
    chunk_size_chars: int = 128 * 1024,
) -> list[int]:
    """Worker process that instantiates its own tokenizer copy and processes

    a strictly bounded byte-slice of a massive file on disk, reporting progress.
    """
    # Lazy-load the tokenizer inside the process worker to isolate memory domains

    tokenizer: BPETokenizer = BPETokenizer.from_files(
        vocab_filepath=vocab_path, merges_filepath=merges_path, report_progress=False
    )

    local_tokens: list[int] = []
    update_threshold_bytes = 1024 * 1024 * 5  # Sync tracking frame every 5MB
    bytes_since_last_report = 0

    with open(file_path, "rb") as f:
        f.seek(start_byte)
        bytes_to_read = end_byte - start_byte
        bytes_processed = 0

        remainder = ""

        while bytes_processed < bytes_to_read:
            to_read = min(chunk_size_chars, bytes_to_read - bytes_processed)
            block_bytes = f.read(to_read)
            if not block_bytes:
                break

            actual_read = len(block_bytes)
            bytes_processed += actual_read
            bytes_since_last_report += actual_read

            text = remainder + block_bytes.decode("utf-8", errors="replace")

            last_newline = text.rfind("\n")
            if last_newline != -1:
                chunk_to_encode = text[: last_newline + 1]
                remainder = text[last_newline + 1 :]
            else:
                chunk_to_encode = text
                remainder = ""

            if chunk_to_encode:
                local_tokens.extend(tokenizer.encode(chunk_to_encode, report_progress=False))

            # Push up increments periodically to minimize pipeline contention
            if progress_queue is not None and bytes_since_last_report >= update_threshold_bytes:
                progress_queue.put((worker_idx, bytes_processed))
                bytes_since_last_report = 0

        if remainder:
            local_tokens.extend(tokenizer.encode(remainder, report_progress=False))

    # Definitive final progress sync for this worker's assigned workspace
    if progress_queue is not None:
        progress_queue.put((worker_idx, bytes_to_read))

    return local_tokens


def parallel_tokenize_pipeline(
    file_path: str,
    vocab_path: str,
    merges_path: str,
    split_token_bytes: bytes = b"\n",
    num_workers: int | None = None,
    report_progress: bool = True,
):
    """Orchestrates multi-core tokenization of a single massive file

    using ChunkedProgressBar to track all parallel segments concurrently.
    """
    if num_workers is None:
        num_workers = mp.cpu_count()
    else:
        num_workers = max(1, min(num_workers, mp.cpu_count()))  # Ensure at least one worker

    filename_base = os.path.basename(file_path)
    print(f"Analyzing file boundaries for {num_workers} processes across {filename_base}...")
    with open(file_path, "rb") as f:
        boundaries: list[int] = find_chunk_boundaries(
            f, desired_num_chunks=num_workers, split_special_token=split_token_bytes
        )

    num_actual_workers = len(boundaries) - 1
    print(f"Spawning {num_actual_workers} parallel workers...")

    with mp.Manager() as manager:
        progress_queue = manager.Queue() if report_progress else None

        # Build tasks incorporating worker structural tracking indexes
        tasks: list[tuple] = []
        for i in range(num_actual_workers):
            tasks.append((file_path, boundaries[i], boundaries[i + 1], vocab_path, merges_path, i, progress_queue))

        if report_progress and progress_queue is not None:
            monitor = ChunkedProgressBar(
                boundaries, progress_queue, title=f"Tokenizing {filename_base} across {num_actual_workers} CPU cores..."
            )
            monitor.start()

        with mp.Pool(processes=num_actual_workers) as pool:
            # Trigger execution asynchronously so the main thread can drive the UI loop
            async_result = pool.starmap_async(tokenize_file_segment_worker, tasks)

            # Keep main thread processing progress queue updates until jobs complete
            while not async_result.ready():
                # Block briefly to wait for completion; updates occur inside ChunkedProgressBar thread
                async_result.wait(timeout=0.5)

            if report_progress and progress_queue is not None:
                monitor.stop()

            # Retrieve final result stream payloads
            results: list[list[int]] = async_result.get()

    print("Reassembling tokens from worker pools...")
    final_token_stream: list[int] = []
    for local_tokens in results:
        final_token_stream.extend(local_tokens)

    return final_token_stream


if __name__ == "__main__":
    vocabs: list[dict[int, bytes]] = []
    for out_vocab in ["vocab-tinystories.pkl", "vocab-owt.pkl"]:
        if os.path.exists(out_vocab):
            with open(out_vocab, "rb") as f:
                vocab: dict[int, bytes] = pickle.load(f)
            print(f"Loaded vocab from {out_vocab}, size: {len(vocab)}")
            vocabs.append(vocab)

    merges: list[list[tuple[bytes, bytes]]] = []
    for out_merges in ["merges-tinystories.pkl", "merges-owt.pkl"]:
        if os.path.exists(out_merges):
            with open(out_merges, "rb") as f:
                merge: list[tuple[bytes, bytes]] = pickle.load(f)
            print(f"Loaded merges from {out_merges}, length: {len(merge)}")
            merges.append(merge)

    data_dir = "data/"
    for name, files in zip(
        ["tinystories", "owt"],
        [["TinyStoriesV2-GPT4-valid.txt", "TinyStoriesV2-GPT4-train.txt"], ["owt_valid.txt", "owt_train.txt"]],
    ):
        # Verify paths match active asset parameters before initializing pipelines
        vocab_p = f"vocab-{name}.pkl"
        merges_p = f"merges-{name}.pkl"
        if not (os.path.exists(vocab_p) and os.path.exists(merges_p)):
            continue

        for file in files:
            filepath: str = os.path.join(data_dir, file)
            if not os.path.exists(filepath):
                print(f"Skipping missing dataset file: {filepath}")
                continue

            encoded: list[int] = parallel_tokenize_pipeline(
                file_path=filepath,
                vocab_path=vocab_p,
                merges_path=merges_p,
                split_token_bytes=b"\n",
                num_workers=32,
                report_progress=True,
            )

            encoded_arr: np.ndarray = np.array(encoded, dtype=np.uint16)
            out_file: str = filepath.rsplit(".", 1)[0] + ".npy"
            np.save(out_file, encoded_arr)
            print(f"Saved encoded data to {out_file}\n")
