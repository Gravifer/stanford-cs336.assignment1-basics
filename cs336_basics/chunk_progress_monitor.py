#! /usr/bin/env python3

import shutil
import sys
import threading
from queue import Empty, Queue
from typing import TextIO

__all__: list[str] = ["ChunkedProgressBar"]


def _in_jupyter() -> bool:
    try:
        get_ipython  # type: ignore
        return True
    except NameError:
        return False


class ChunkedProgressBar:
    """
    A unified progress monitor that partitions the terminal width
    proportionally to chunk boundaries and fills them concurrently.
    """

    def __init__(
        self, boundaries: list[int], queue: Queue[tuple[int, int]], title: str = "", *, stream: TextIO | None = None
    ) -> None:
        self.divisions: list[int] = boundaries
        self.total_work: int = boundaries[-1]
        self._queue: Queue[tuple[int, int]] = queue
        self._stop_event = threading.Event()
        self._progress: dict[int, int] = {i: 0 for i in range(len(boundaries) - 1)}
        self._stream: TextIO = stream if stream is not None else sys.stdout
        self._thread = threading.Thread(target=self._render_loop, daemon=True)
        self.title: str = title

    def start(self) -> None:
        self._thread.start()
        self._stream.write(f"{self.title}\n")
        self._stream.flush()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join()
        # Drain any messages that arrived after the last render loop iteration

        while True:
            try:
                chunk_idx, bytes_processed = self._queue.get(block=False)
            except Empty:
                break
            self._progress[chunk_idx] = bytes_processed
        self._render_frame()  # Final flush
        self._stream.write("\n")
        self._stream.flush()

    def _render_loop(self) -> None:
        # Throttle updates to ~10Hz to prevent IPC contention and Jupyter output lag
        while not self._stop_event.wait(0.1):
            while True:
                try:
                    chunk_idx, bytes_processed = self._queue.get(block=False)
                except Empty:
                    break
                self._progress[chunk_idx] = bytes_processed
            self._render_frame()
        # Drain remaining messages before exiting, so stop() sees a consistent state

        while True:
            try:
                chunk_idx, bytes_processed = self._queue.get(block=False)
            except Empty:
                break
            self._progress[chunk_idx] = bytes_processed

    def _render_frame(self) -> None:
        # Leave buffer for brackets and potential trailing spaces
        term_width = shutil.get_terminal_size((80, 20)).columns - 4
        char_array = [" "] * term_width

        # 1. Draw segment boundaries
        for b in self.divisions[1:-1]:
            idx = int((b / self.total_work) * term_width)
            if idx < term_width:
                char_array[idx] = "│"

        # 2. Fill intervals
        for chunk_idx, (start, end) in enumerate(zip(self.divisions, self.divisions[1:])):
            chunk_len = end - start
            if chunk_len == 0:
                continue

            bytes_done = self._progress.get(chunk_idx, 0)
            fill_ratio = bytes_done / chunk_len

            chars_start = int((start / self.total_work) * term_width)
            chars_end = int((end / self.total_work) * term_width)
            chars_to_fill = int((chars_end - chars_start) * fill_ratio)

            for i in range(chars_start, min(chars_start + chars_to_fill, chars_end)):
                if char_array[i] != "│":
                    char_array[i] = "█"

        # Jupyter environment suppression: carriage return is sufficient if flushed reliably,
        # but avoid printing if nothing changed to prevent frontend buffer bloat.
        rendered = f"\r[{''.join(char_array)}]"
        self._stream.write(rendered)
        self._stream.flush()
