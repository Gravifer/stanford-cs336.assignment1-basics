import os
import sys
import threading

import psutil

from tests.common import gpt2_bytes_to_unicode

b2u: dict[int, str] = gpt2_bytes_to_unicode()


def token_to_readable(token_bytes: bytes) -> str:
    """Render a token's bytes as a readable string using GPT-2's bytes_to_unicode mapping."""
    return "".join(b2u[b] for b in token_bytes)


def prettyprint_vocab(vocab: dict[int, bytes], cols: int = 10, col_width: int = 12, skip_bytes: bool = True) -> None:
    """Pretty-print the vocabulary table.

    Args:
        vocab: token id -> bytes mapping
        cols: number of columns per row
        col_width: character width of each token column
        skip_bytes: whether to skip printing most of the single-byte tokens
    """
    id_width = 6
    sep = " │ "
    print(f"{'id':>{id_width}}", end=sep)
    for j in range(cols):
        print(f"{j:>{col_width}}", end=sep)
    print()
    print("─" * (id_width + len(sep) + (col_width + len(sep)) * cols))
    start = cols * (256 // cols) if skip_bytes else 0
    for i in range(start, len(vocab), cols):
        print(f"{i:>{id_width}}", end=sep)
        for j in range(cols):
            idx = i + j
            if idx < len(vocab):
                print(f"{token_to_readable(vocab[idx]):>{col_width}}", end=sep)
        print()


class PeakMemoryMonitor:
    """Polls RSS of the current process + all children at 0.5s intervals."""

    def __init__(self, verbose: bool = False):
        self._proc = psutil.Process(os.getpid())
        self._peak_mb = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._inuse = False
        self._verbose = verbose

    def _rss_mb(self) -> float:
        try:
            total: float = self._proc.memory_info().rss
            for child in self._proc.children(recursive=True):
                try:
                    total += child.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    if self._verbose:
                        sys.stderr.write(f"Warning: failed to get memory info for child process {child.pid}\n")
            return total / 1024**2
        except psutil.NoSuchProcess:
            return 0.0

    def _poll(self):
        while not self._stop.wait(0.5):
            self._peak_mb: float = max(self._peak_mb, self._rss_mb())

    def __enter__(self):  # TODO: add an assertion-based sanity test forbidding double-entry
        if self._inuse:
            raise RuntimeError("PeakMemoryMonitor is not reentrant")
        self._inuse = True
        self._peak_mb: float = self._rss_mb()  # baseline
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join()
        self._peak_mb: float = max(self._peak_mb, self._rss_mb())
        self._inuse = False

    @property
    def peak_mb(self) -> float:
        return self._peak_mb
