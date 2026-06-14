#!/usr/bin/env python3
import multiprocessing as mp
import os
import pickle
from queue import Queue

import numpy as np

from cs336_basics.chunk_progress_monitor import ChunkedProgressBar
from cs336_basics.pretokenization_example import find_chunk_boundaries
from cs336_basics.tokenizer import BPETokenizer


def tokenize_file_segment_to_mmap_worker(
    file_path: str,
    start_byte: int,
    end_byte: int,
    vocab_path: str,
    merges_path: str,
    output_mmap_path: str,
    mmap_offset_tokens: int,
    worker_idx: int,
    progress_queue: Queue | None = None,
    chunk_size_chars: int = 128 * 1024,
) -> int:
    """Worker process that instantiates its own tokenizer copy, processes a strictly

    bounded byte-slice of a file, and writes results directly into a shared memory-mapped
    array on disk to maintain a completely flat RAM footprint.

    Returns:
        int: The exact number of tokens written by this worker.
    """
    # Lazy-load the tokenizer inside the process worker to isolate memory domains

    tokenizer: BPETokenizer = BPETokenizer.from_files(
        vocab_filepath=vocab_path, merges_filepath=merges_path, report_progress=False
    )

    # Estimate worst-case allocation window (1 token per byte maximum for raw text)
    max_chunk_tokens = end_byte - start_byte

    # Open the shared temporary file in r+ mode
    shared_mmap = np.memmap(
        output_mmap_path,
        dtype=np.uint16,
        mode="r+",
        shape=(max_chunk_tokens,),
        offset=mmap_offset_tokens * np.dtype(np.uint16).itemsize,
    )

    tokens_written = 0
    update_threshold_bytes = 1024 * 1024 * 5  # Sync progress frame every 5MB
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
                chunk_ids = tokenizer.encode(chunk_to_encode, report_progress=False)
                n_toks = len(chunk_ids)

                # Commit IDs directly to disk layout without IPC pipe transport overhead
                shared_mmap[tokens_written : tokens_written + n_toks] = chunk_ids
                tokens_written += n_toks

            # Push tracking increments back to parent process periodically
            if progress_queue is not None and bytes_since_last_report >= update_threshold_bytes:
                progress_queue.put((worker_idx, bytes_processed))
                bytes_since_last_report = 0

        if remainder:
            chunk_ids = tokenizer.encode(remainder, report_progress=False)
            n_toks = len(chunk_ids)
            shared_mmap[tokens_written : tokens_written + n_toks] = chunk_ids
            tokens_written += n_toks

    # Flush changes out of OS memory buffers down to the storage subsystem explicitly
    shared_mmap.flush()

    # Definitive final progress sync for this worker's segment domain
    if progress_queue is not None:
        progress_queue.put((worker_idx, bytes_to_read))

    return tokens_written


def parallel_tokenize_pipeline(
    file_path: str,
    vocab_path: str,
    merges_path: str,
    out_npy_path: str,
    split_token_bytes: bytes = b"\n",
    num_workers: int | None = None,
    report_progress: bool = True,
):
    """Orchestrates multi-core tokenization of a single massive file

    by writing directly to a memory-mapped output file, keeping RAM constant.
    """
    if num_workers is None:
        num_workers = mp.cpu_count()
    else:
        num_workers = max(1, min(num_workers, mp.cpu_count()))

    filename_base = os.path.basename(file_path)
    print(f"Analyzing file boundaries for {num_workers} processes across {filename_base}...")
    with open(file_path, "rb") as f:
        boundaries: list[int] = find_chunk_boundaries(
            f, desired_num_chunks=num_workers, split_special_token=split_token_bytes
        )

    num_actual_workers = len(boundaries) - 1
    print(f"Spawning {num_actual_workers} parallel workers...")

    # Set up memory-mapped workspace file template matching the exact source file size bound
    max_possible_tokens = boundaries[-1]
    tmp_mmap_path = out_npy_path + ".tmp"

    # Allocate physical empty workspace template file on disk
    shape_template = (max_possible_tokens,)
    shared_mmap = np.memmap(tmp_mmap_path, dtype=np.uint16, mode="w+", shape=shape_template)
    del shared_mmap  # Close reference to flush descriptor bindings

    with mp.Manager() as manager:
        progress_queue = manager.Queue() if report_progress else None

        # Build tasks calculating precise offsets for independent worker storage segments
        tasks: list[tuple] = []
        current_token_offset = 0

        for i in range(num_actual_workers):
            tasks.append(
                (
                    file_path,
                    boundaries[i],
                    boundaries[i + 1],
                    vocab_path,
                    merges_path,
                    tmp_mmap_path,
                    current_token_offset,
                    i,
                    progress_queue,
                )
            )
            # Increment offset template using maximum space bound of the byte slice
            current_token_offset += boundaries[i + 1] - boundaries[i]

        if report_progress and progress_queue is not None:
            monitor = ChunkedProgressBar(
                boundaries, progress_queue, title=f"Tokenizing {filename_base} across {num_actual_workers} CPU cores..."
            )
            monitor.start()

        with mp.Pool(processes=num_actual_workers) as pool:
            # Trigger parallel runs asynchronously to leave the main thread free to handle screen rendering
            async_result = pool.starmap_async(tokenize_file_segment_to_mmap_worker, tasks)

            while not async_result.ready():
                async_result.wait(timeout=0.5)

            if report_progress and progress_queue is not None:
                monitor.stop()

            # Results contain only scalar item counts written by each process worker
            tokens_written_per_worker: list[int] = async_result.get()

    print("Consolidating contiguous output array fragments...")

    # Compute actual packed size boundaries omitting unneeded over-allocated tail slices
    total_valid_tokens = 0
    read_mmap = np.memmap(tmp_mmap_path, dtype=np.uint16, mode="r", shape=shape_template)

    # Initialize true destination array
    final_arr = np.empty(sum(tokens_written_per_worker), dtype=np.uint16)

    # Pack worker results into a single contiguous block
    current_read_offset = 0
    current_write_offset = 0

    for i, tokens_count in enumerate(tokens_written_per_worker):
        chunk_max_bytes = boundaries[i + 1] - boundaries[i]
        if tokens_count > 0:
            final_arr[current_write_offset : current_write_offset + tokens_count] = read_mmap[
                current_read_offset : current_read_offset + tokens_count
            ]
            current_write_offset += tokens_count
        current_read_offset += chunk_max_bytes

    # Clean up file mapping descriptor hooks before running unlink calls
    del read_mmap

    # Safely save final clean array layout file format structure
    np.save(out_npy_path, final_arr)
    print(f"Saved cleanly packed data to {out_npy_path}")

    if os.path.exists(tmp_mmap_path):
        os.remove(tmp_mmap_path)


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
        vocab_p = f"vocab-{name}.pkl"
        merges_p = f"merges-{name}.pkl"
        if not (os.path.exists(vocab_p) and os.path.exists(merges_p)):
            continue

        for file in files:
            filepath: str = os.path.join(data_dir, file)
            if not os.path.exists(filepath):
                print(f"Skipping missing dataset file: {filepath}")
                continue

            out_file: str = filepath.rsplit(".", 1)[0] + ".npy"

            parallel_tokenize_pipeline(
                file_path=filepath,
                vocab_path=vocab_p,
                merges_path=merges_p,
                out_npy_path=out_file,
                split_token_bytes=b"\n",
                num_workers=32,
                report_progress=True,
            )
            print(f"Processing sequence execution finished for {file}\n")
