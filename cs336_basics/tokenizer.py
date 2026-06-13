#! /usr/bin/env python3
import functools
import heapq
import json
import multiprocessing as mp
import os
import pickle
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from collections.abc import Collection, Iterable, Iterator, Set
from queue import Queue
from typing import Literal, LiteralString, NamedTuple

import regex as re
from tqdm import tqdm

from .chunk_progress_monitor import ChunkedProgressBar
from .pretokenization_example import find_chunk_boundaries

__all__: list[str] = ["train_bpe", "BPETokenizer", "Tokenizer"]

_PAT: LiteralString = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""  # see <https://github.com/openai/tiktoken/pull/234/change>
_pattern: re.Pattern[str] = re.compile(_PAT)


class _BPECandidate[T]:  # python only has min-heaps before 3.14, so we use a wrapper class that inverts comparison
    __slots__ = ("pair", "count", "vocab")

    # @property
    # def count(self) -> int:
    #     return self.occurrences.total()

    def __init__(self, pair: tuple[T, T], count: int, vocab: dict[T, bytes] | None):
        self.pair = pair  # starts as byte digrams, but as merges happen, becomes longer
        self.count = count
        self.vocab = vocab

    def __lt__(self, other: "_BPECandidate") -> bool:
        if not isinstance(other, _BPECandidate):
            raise NotImplementedError(f"Cannot compare _BPECandidate with {type(other)}")
        if (s := self.count) != (o := other.count):
            return s > o
        assert self.vocab is other.vocab, "Comparison only makes sense within the same corpus"
        if self.vocab is None or other.vocab is None:
            return self.pair > other.pair
        s: tuple[bytes, bytes] = (self.vocab[self[0]], self.vocab[self[1]])
        o: tuple[bytes, bytes] = (other.vocab[other[0]], other.vocab[other[1]])
        return s > o  # lexicographical

    def __getitem__(self, key: Literal[0] | Literal[1]) -> T:
        if isinstance(key, slice):
            return self.pair[key]
        if key == 0 or key == 1:
            return self.pair[key]
        else:
            raise IndexError(f"_BPECandidate is a pair; got {key} as index when expecting 0 or 1")

    def __repr__(self) -> str:
        return f"BPECandidate(digram={self.pair}, count={self.count})"


def _pretokenize_chunk(
    input_path: str | os.PathLike,
    chunk: tuple[int, int],
    sep: re.Pattern[bytes],
    pattern: re.Pattern[str],
    worker_idx: int | None = None,
    progress_queue: Queue | None = None,
) -> Counter[str]:
    """subprocess worker to pretokenize a chunk of a file and return the pre-token counts;
    used for multiprocessing support in train_bpe
    """
    with open(input_path, "rb") as f:  # * we acquire a file handle for each process
        start, end = chunk
        f.seek(start)
        chunk_data: bytes = f.read(end - start)
    pretokens: Counter[str] = Counter()
    bytes_processed = 0
    bytes_increment = 0
    update_threshold = 1024 * 1024 * 5  # Report every 5MB to minimize lock contention
    # Iterating eagerly allows us to update the progress bar,
    # replacing the generator comprehension
    for doc_bytes in sep.split(chunk_data):
        doc: str = doc_bytes.decode("utf-8", errors="ignore")
        matches = pattern.finditer(doc)
        pretokens.update(match.group() for match in matches)

        bytes_increment += len(doc_bytes)
        if bytes_increment >= update_threshold and (progress_queue is not None and worker_idx is not None):
            progress_queue.put((worker_idx, bytes_processed))
            bytes_increment = 0

    # Final sync for this chunk
    if progress_queue is not None and worker_idx is not None:
        progress_queue.put((worker_idx, end - start))
    return pretokens


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str | bytes],
    *,
    pretokenization: re.Pattern[str] | bool = True,
    multiprocessing: int | bool = True,
    report_progress: bool = True,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer on the given input text file.

    Args:
        input_path: Path to a text file with BPE tokenizer training data.
        vocab_size: The maximum final vocabulary size (including the initial
                    byte vocabulary, vocabulary items produced from merging, and any special tokens).
        special_tokens: A list of special tokens to include in the vocabulary.
                    During training, treat them as hard boundaries
                    that prevent merges across their spans,
                    but do not include them when computing merge statistics.
        pretokenization: If True, use the default pre-tokenization regex pattern;
                    if a regex pattern, use that for pre-tokenization instead.
        multiprocessing: If True, use as many processes as there are CPU cores;
                    if an integer, use at most that many processes;
                    if False or None, do not use multiprocessing.
        report_progress: If True, display a progress bar during training.
                    Only applicable if multiprocessing is enabled.

    Returns:
        A tuple containing:
        - vocab: The tokenizer vocabulary, a dictionary mapping token IDs to token bytes.
        - merges: A list of BPE merges produced from training.
                    Each merge is a tuple of bytes (<token1>, <token2>),
                    representing that <token1> was merged with <token2>.
                    Sorted by order of creation.
    """
    if len(special_tokens) < 1:
        raise ValueError('special_tokens should at least contain b"<|endoftext|>"')
    special_tokens: list[bytes] = [
        token.encode() if isinstance(token, str) else token for token in special_tokens
    ]  # normalize special tokens to bytes
    sep: re.Pattern[bytes] = re.compile(b"|".join(map(re.escape, special_tokens)))

    pattern: re.Pattern[str] = (
        _pattern
        if pretokenization is True
        else pretokenization or re.compile(r".*")  # TODO: reuse the pretokenization_pattern method from BPETokenizer
    )  # if pretokenization is False, the pre-token is the entire document

    num_processes = os.cpu_count() or multiprocessing or 1  # if os.cpu_count() is None, fall back to no capping
    num_processes = max(
        1, min(num_processes if multiprocessing is True else multiprocessing or 1, num_processes)
    )  # clamp between 1 and os.cpu_count()

    # ! SPEC: special tokens goes at the beginning
    vocab: dict[int, bytes] = {i: t for i, t in enumerate(special_tokens)} | {
        len(special_tokens) + i: bytes([i]) for i in range(256)
    }  # initial byte vocabulary
    merges: list[tuple[bytes, bytes]] = []
    num_merges = vocab_size - len(vocab)

    with open(input_path, "rb") as f:
        boundaries = (
            find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
            if num_processes > 1
            else [0, os.path.getsize(input_path)]
        )
    num_processes = len(boundaries) - 1 or 1  # in case we got fewer boundaries than desired

    pretokens: Counter[str] = Counter()
    if num_processes <= 1:
        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries, boundaries[1:]):
            pretokens.update(_pretokenize_chunk(input_path, (start, end), sep, pattern))
    else:
        with mp.Manager() as manager:
            if report_progress:
                progress_queue: Queue[tuple[int, int]] = manager.Queue()
                monitor = ChunkedProgressBar(
                    boundaries, progress_queue, f"Training BPE with {num_processes} processes..."
                )
            try:
                if report_progress:
                    monitor.start()
                with mp.Pool(num_processes) as pool:
                    counters: list[Counter[str]] = pool.starmap(
                        _pretokenize_chunk,
                        [
                            (
                                input_path,
                                (start, end),
                                sep,
                                pattern,
                                worker_idx,
                                progress_queue if report_progress else None,
                            )
                            for worker_idx, (start, end) in enumerate(zip(boundaries, boundaries[1:]))
                        ],
                    )
            finally:
                if report_progress:
                    monitor.stop()
        # // pretokens.update(chain.from_iterable(counters))  # ! WRONG
        for counter in counters:
            pretokens.update(counter)  # collect counts from all chunks

    print(f"Computing merges among {len(pretokens)} pretokens...")

    Seq: type = list[int]  # we need mutability // tuple[bytes, ...]  # * the representation suggested by the handout
    SeqHandle: type = int
    SeqCnt: type = NamedTuple("SeqCnt", [("seq", Seq), ("count", int)])
    seqs: tuple[SeqCnt, ...] = tuple(  # // tuple(ch.encode() for ch in pt)
        SeqCnt([len(special_tokens) + c for c in pt.encode()], count) for pt, count in sorted(pretokens.items())
    )

    PC: type = _BPECandidate  # using a wrapper because python only has min-heap before 3.14
    Pair: type = tuple[int, int]
    Occurrence: type = SeqHandle  # we'll just do linear lookups
    pairs: dict[Pair, Counter[Occurrence]] = defaultdict(Counter)
    for seqno, (seq, count) in enumerate(seqs):
        for offset, digram in enumerate(zip(seq, seq[1:])):
            pairs[digram][seqno] += 1
    pairs_heap: list[PC[int]] = [  # will be used lazily
        PC(digram, sum(seqs[seqno].count * n for seqno, n in occurrences.items()), vocab)
        for digram, occurrences in pairs.items()
    ]
    pairs_cnts: defaultdict[Pair, int] = defaultdict(
        int, {p.pair: p.count for p in pairs_heap}
    )  # will be used as authoritative
    heapq.heapify(pairs_heap)

    # -------------------------------------------------------------
    # compute merges
    # -------------------------------------------------------------
    for _ in tqdm(range(num_merges), disable=not report_progress):
        while pairs_heap:  # Pop the highest frequency pair lazily
            top: PC[int] = heapq.heappop(pairs_heap)
            if top.count == (authoritative := pairs_cnts[top.pair]):  # Validate against our authoritative counter
                break  # breaks while -> goto token registration
            # if we got here, the record was stale
            elif top.count < authoritative:
                heapq.heappush(pairs_heap, PC(top.pair, authoritative, vocab))
        else:
            break  # No more pairs left to merge; breaks for -> goto return

        assert top.count == pairs_cnts[top.pair] and top.count > 0, "Heap invariant should guarantee this"

        # Register the new token
        tok0, tok1 = vocab[top[0]], vocab[top[1]]

        # # DEBUG: if this is (' c', 'om'), print the count of it and ('t', 'h')
        # if tok0 == b" c" and tok1 == b"om":
        #     print(
        #         f"""DEBUG: merging {top.pair} with count {top.count}, which is {" ".join(map(repr, (tok0, tok1)))};
        #         count of ('t', 'h') is {pairs_cnts[(1 + ord("t"), 1 + ord("h"))]}
        #         """
        #     )
        #     assert False, "Debug break"

        tokid = len(vocab)
        vocab[tokid] = tok0 + tok1
        merges.append((tok0, tok1))

        affected_pairs: set[Pair] = set()

        for seqno, n in pairs[top.pair].items():  # For each sequence the pair occurs in
            seq: list[int] = seqs[seqno].seq
            pos: int = 0
            while True:  # For each occurrence of the pair in the sequence
                try:
                    i: int = seq.index(top[0], pos)  # find the first token
                    if i + 1 < len(seq) and seq[i + 1] == top[1]:  # check if followed by the second token
                        # We found an occurrence at index i; we need to merge it
                        del seq[i]  # remove the first token
                        seq[i] = tokid  # replace the second token with the new merged token
                        pairs_cnts[top.pair] -= seqs[seqno].count

                        # Update pairs_cnts and pairs for the affected digrams around the merged token
                        if i > 0:  # there's a preceding token, so a new digram is formed with the merged token
                            prev_pair: tuple[int, int] = (seq[i - 1], top[0])
                            pairs_cnts[prev_pair] -= seqs[seqno].count
                            affected_pairs.add(prev_pair)
                            new_prev_pair: tuple[int, int] = (seq[i - 1], tokid)
                            pairs_cnts[new_prev_pair] += seqs[seqno].count
                            pairs[new_prev_pair][seqno] += 1
                            affected_pairs.add(new_prev_pair)

                        if i < len(seq) - 1:  # there's a following token, do the same
                            next_pair: tuple[int, int] = (top[1], seq[i + 1])
                            pairs_cnts[next_pair] -= seqs[seqno].count
                            affected_pairs.add(next_pair)
                            new_next_pair: tuple[int, int] = (tokid, seq[i + 1])
                            pairs_cnts[new_next_pair] += seqs[seqno].count
                            pairs[new_next_pair][seqno] += 1
                            affected_pairs.add(new_next_pair)
                    pos = i + 1  # Move past the position we just examined to avoid rescanning
                except ValueError:
                    break  # No more occurrences of the first token
            # The merged pair no longer occurs in this sequence, so we can set its count to zero for this sequence
        #     assert pairs[top.pair][seqno] == 0, (
        #         f"Expect the {seqno}-th sequence to be free of {top.pair}, got {pairs[top.pair][seqno]} left"
        #     )
        # Push final authoritative counts for all affected pairs in one shot
        for p in affected_pairs:
            heapq.heappush(pairs_heap, PC(p, pairs_cnts[p], vocab))

        assert pairs_cnts[top.pair] == 0, f"Expect the pair {top.pair} to be depleted, got {pairs_cnts[top.pair]} left"
        del pairs[top.pair]
        del pairs_cnts[top.pair]

    return vocab, merges


class TextTokenizer(ABC):
    @abstractmethod
    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    @abstractmethod
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        raise NotImplementedError

    @abstractmethod
    def decode(self, ids: list[int]) -> str:
        raise NotImplementedError


class BPETokenizer(TextTokenizer):
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,  # the user is not allowed to specify byte sequences invalid as utf-8
        *,
        report_progress: bool = True,
    ):
        self._vocab = vocab
        self._merges = merges
        special_tokens: list[str] = special_tokens or []
        assert all(tok for tok in special_tokens)

        nul_id = next((id for id, token in vocab.items() if token == b"\x00"), -1)
        self._assumed_special_tokens: dict[bytes, int] = {vocab[id]: id for id in range(nul_id)}
        self._user_defined_special_tokens: dict[str, int] = dict()

        # append the user-defined special tokens to the vocab, ensuring no conflicts with existing tokens
        vocab_initial_size = len(vocab)
        vocab_initial_highest_id = max(vocab.keys())
        next_id = vocab_initial_highest_id + 1
        for token in special_tokens:
            token_bytes: bytes = token.encode()
            if token_bytes in vocab.values():  # check if the token already exists in the vocab
                for existing_id, existing_token in vocab.items():  # ? this may be inefficient
                    if token_bytes == existing_token:
                        self._user_defined_special_tokens[token] = existing_id
                        break
                continue
            # ? what if it is a substring of an existing token?
            vocab[next_id] = token_bytes
            self._user_defined_special_tokens[token] = next_id
            next_id += 1

        if report_progress and next_id > vocab_initial_highest_id + 1:
            print(
                f"{next_id - vocab_initial_highest_id} additional special tokens appended at [{vocab_initial_highest_id + 1}:{next_id}]; "
                f"size {vocab_initial_size} → {len(vocab)}"
            )

        # O(1) bytes→id lookup instead of linear scan
        self._vocab_inv: dict[bytes, int] = {v: k for k, v in self._vocab.items()}
        # O(1) merge priority lookup
        try:
            self._merge_rank: dict[tuple[bytes, bytes], int] = {pair: rank for rank, pair in enumerate(self._merges)}
        except TypeError as e:
            print(f"DEBUG: merges: {self._merges}")
            raise e
        # per-instance pretoken cache
        self._pretoken_cache: dict[bytes | str, tuple[int, ...]] = {}
        for special_token, id in self._user_defined_special_tokens.items():
            self._pretoken_cache[special_token] = (id,)
            self._pretoken_cache[special_token.encode()] = (id,)
        for special_token, id in self._assumed_special_tokens.items():
            self._pretoken_cache[special_token] = (id,)

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None,
        *,
        report_progress: bool = True,
    ):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)  # ! we should let any exception propagate
            if report_progress:
                print(f"Loaded vocab from {vocab_filepath}, size: {len(vocab)}")
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
            if report_progress:
                print(f"Loaded merges from {merges_filepath}, size: {len(merges)}")
        return cls(vocab, merges, special_tokens, report_progress=report_progress)

    @staticmethod
    @functools.lru_cache(maxsize=128)
    def pretokenization_pattern(enable: LiteralString | re.Pattern[str] | bool = True) -> re.Pattern[str]:
        """Returns the regex pattern used for pre-tokenization if enabled,
        otherwise returns a pattern that treats the entire document as a single token."""
        return _pattern if enable is True else re.compile(enable) or re.compile(r".*")

    @functools.lru_cache(maxsize=128)
    def _encode_pretoken(self, pretoken: str) -> tuple[int, ...]:
        """Encode a single pretoken (as bytes) into a sequence of token IDs, using the vocabulary and merges."""
        if pretoken in self._pretoken_cache:
            return self._pretoken_cache[pretoken]

        if pretoken in self._user_defined_special_tokens:
            token_ids: tuple[int] = (self._user_defined_special_tokens[pretoken],)
            self._pretoken_cache[pretoken] = token_ids
            return token_ids

        pretoken: bytes = pretoken.encode()

        if pretoken in self._assumed_special_tokens:
            token_ids: tuple[int] = (self._assumed_special_tokens[pretoken],)
            self._pretoken_cache[pretoken] = token_ids
            return token_ids
        tokens: list[bytes] = [bytes([b]) for b in pretoken]  # start with byte-level tokens

        while len(tokens) > 1:  # keep merging, until we have a single token or run out of rules
            best_rank, best_i = None, -1
            for i in range(len(tokens) - 1):
                rank = self._merge_rank.get((tokens[i], tokens[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_i = rank, i
            if best_i == -1:
                break
            tokens[best_i] = tokens[best_i] + tokens[best_i + 1]
            del tokens[best_i + 1]

        # token_ids: list[int] = []
        # for token in tokens:
        #     if token in self._vocab_inv:
        #         token_ids.append(self._vocab_inv[token])
        #     else:
        #         # print(f"DEBUG: {self._vocab}")
        #         raise ValueError(
        #             f"Pretoken {pretoken!r} contains byte sequence {token!r} that cannot be encoded with the current vocabulary"
        #         )
        # token_ids: tuple[int, ...] = tuple(token_ids)

        try:
            token_ids: tuple[int, ...] = tuple(self._vocab_inv[token] for token in tokens)
        except KeyError as e:
            raise ValueError(
                f"Pretoken {pretoken!r} contains byte sequence {e.args[0]!r} that cannot be encoded with the current vocabulary"
            )
        self._pretoken_cache[pretoken] = token_ids
        return token_ids

    def encode(
        self,
        text: str,
        *,
        pretokenization: re.Pattern[str] | bool = True,
        allowed_special: Literal["all"] | Set[str] = "all",
        disallowed_special: Literal["all"] | Collection[str] = set(),
        report_progress: bool = True,
    ) -> list[int]:
        #
        if allowed_special == "all":  # even then we don't allow ones that are not supplied to __init__
            allowed_special: set[str] = set(self._user_defined_special_tokens.keys())
        if disallowed_special == "all":
            disallowed_special: set[bytes] = set(t.encode() for t in self._user_defined_special_tokens.keys()) | set(
                self._assumed_special_tokens.keys()
            ) - set(t.encode() for t in allowed_special)
        if disallowed_special:
            if match := re.search(
                b"|".join(map(re.escape, list(disallowed_special))),
                text.encode(),
            ):
                # print(f"DEBUG: disallowed tokens: {disallowed_special}")
                raise ValueError(
                    f"Input text at {match.start()}-{match.end()} contains disallowed special token {match.group()!r}"
                )
        # we need to incorporate the allowed_special tokens into the pretokenization pattern, ensuring they take precedence over the default pattern
        # print(f"DEBUG: allowed_special tokens: {allowed_special}")
        # ! SPEC: the tests require that if one allowed_special is a substring of another, the longer one should take precedence;
        # ! we ensure this by sorting by length in descending order before joining into the regex pattern
        specials: re.Pattern[str] = re.compile(
            "(" + "|".join(map(re.escape, sorted(allowed_special, key=str.__len__, reverse=True))) + ")"
            if allowed_special
            else r"(.+)"
        )
        pattern = self.pretokenization_pattern(pretokenization)
        out: list[int] = []
        fragments: list[str] = [fragment for fragment in specials.split(text) if fragment != ""]
        # short_frags = [(idx, f) for idx, f in enumerate(fragments) if len(f) <= 15]
        # print(f"DEBUG: short fragments (<=15 chars): {short_frags[:10]}...")
        if report_progress:
            print()
        for fragment in (p := tqdm(fragments, disable=not report_progress)):
            if fragment == "":
                continue
            p.set_postfix_str(f"{fragment[:8]!r:<8}{'...' if len(fragment) > 8 else '   '}")
            if fragment in self._user_defined_special_tokens:
                out.append(self._user_defined_special_tokens[fragment])
                continue
            elif fragment.encode() in self._assumed_special_tokens:
                out.append(self._assumed_special_tokens[fragment.encode()])
                continue
            matches = pattern.finditer(fragment)
            for match in matches:
                pretoken: str = match.group()
                # print(f"DEBUG: pretoken {pretoken!r} from {match.start()}-{match.end()}")
                out.extend(self._encode_pretoken(pretoken))
        return out

    def encode_iterable(self, iterable: Iterable[str], *, report_progress: bool = True) -> Iterator[int]:
        # call self.encode() on each
        if report_progress:
            print()
        for chunk in (p := tqdm(iterable, disable=not report_progress)):
            p.set_postfix_str(f"{chunk[:8]!r:<8}{'...' if len(chunk) > 8 else '   '}")
            yield from self.encode(chunk, report_progress=False)

    def decode(self, ids: list[int], errors: str = "replace") -> str:
        """
        look up each ID's corresponding entries in the vocabulary (a byte sequence),
        concatenate them together, and then decode the bytes to a Unicode string.

        Note that user input IDs are not guaranteed to map to valid Unicode strings;
        in that case, use the 'replace' error handler to automatically replace malformed data with the replacement marker.
        """
        for id in ids:
            if id not in self._vocab:
                # ? should we raise an error, or fallback to U+FFFD?
                raise ValueError(f"ID {id} not found in vocabulary")
        byte_seq = b"".join(self._vocab[id] for id in ids)
        return byte_seq.decode("utf-8", errors=errors)


Tokenizer: type = BPETokenizer

if __name__ == "__main__":
    import argparse
    import json
    import pickle

    parser = argparse.ArgumentParser(description="Train a BPE tokenizer")
    parser.add_argument("input_path", help="Path to training corpus")
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--special-tokens", nargs="+", default=["<|endoftext|>"])
    parser.add_argument("--no-multiprocessing", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--out-vocab", default="vocab.json", help="Output path for vocabulary (JSON)")
    parser.add_argument("--out-merges", default="merges.pkl", help="Output path for merges (pickle)")
    args = parser.parse_args()

    vocab, merges = train_bpe(
        args.input_path,
        args.vocab_size,
        args.special_tokens,
        multiprocessing=not args.no_multiprocessing,
        report_progress=not args.no_progress,
    )

    with open(args.out_vocab, "w") as f:
        json.dump(
            {str(k): v.decode("utf-8", errors="replace") for k, v in vocab.items()}, f, ensure_ascii=False, indent=2
        )
    with open(args.out_merges, "wb") as f:
        pickle.dump(merges, f)
    print(f"Saved vocab ({len(vocab)} tokens) → {args.out_vocab}")
    print(f"Saved merges ({len(merges)}) → {args.out_merges}")
