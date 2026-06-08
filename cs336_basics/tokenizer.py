#! /usr/bin/env python3
import heapq
import os
from collections import Counter, defaultdict
from collections.abc import Iterable
from itertools import chain
from typing import Literal, LiteralString, NamedTuple

import regex as re
from regex._main import Match

from .pretokenization_example import find_chunk_boundaries

_PAT: LiteralString = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""  # see <https://github.com/openai/tiktoken/pull/234/change>
_pattern: re.Pattern[str] = re.compile(_PAT)


class _BPECandidate[T]:
    # python only has min-heaps before 3.14, so we use a wrapper class that inverts comparison
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
            raise IndexError(f"_BPECandidate is a pair of '{T}'s; got {key} as index when expecting 0 or 1")

    def __repr__(self) -> str:
        return f"BPECandidate(digram={self.pair}, count={self.count})"


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[LiteralString | bytes],
    *,
    pretokenization: re.Pattern[str] | bool = True,
    multiprocessing: int | bool = False,
    repetitive_pretokens: bool = False,
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
        repetitive_pretokens: If True, Count byte pairs within each pre-token first, then multiply by the pre-token's count.
                    if False, iterate through each pre-token directly.

    Returns:
        A tuple containing:
        - vocab: The tokenizer vocabulary, a dictionary mapping token IDs to token bytes.
        - merges: A list of BPE merges produced from training.
                    Each merge is a tuple of bytes (<token1>, <token2>),
                    representing that <token1> was merged with <token2>.
                    Sorted by order of creation.
    """
    assert len(special_tokens) > 0, 'Should at least contain b"<|endoftext|>"'
    special_tokens: list[bytes] = [
        token.encode() if isinstance(token, str) else token for token in special_tokens
    ]  # normalize special tokens to bytes
    sep: re.Pattern[bytes] = re.compile(b"|".join(map(re.escape, special_tokens)))

    pattern: re.Pattern[str] = (
        _pattern if pretokenization is True else pretokenization or re.compile(r".*")
    )  # if pretokenization is False, the pre-token is the entire document

    num_processes = os.cpu_count() or multiprocessing or 1  # if os.cpu_count() is None, fall back to no capping
    num_processes = max(1, min(multiprocessing or 1, num_processes))  # clamp between 1 and os.cpu_count()

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

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries, boundaries[1:]):
            f.seek(start)
            chunk: bytes = f.read(end - start)
            # remove special tokens before pre-tokenization
            docs: Iterable[str] = (doc.decode("utf-8", errors="ignore") for doc in sep.split(chunk))
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            matches: Iterable[Match[str]] = chain.from_iterable(
                pattern.finditer(doc) for doc in docs
            )  # re.finditer(_PAT, doc)
            pretokens: Counter[str] = Counter(match.group() for match in matches)
            # TODO: for now the tally is complete; for multiprocessing, send the tally back to the main process
        # ? will we need to acquire a file handle for each process?
    # TODO: for multiprocessing, collect counts from all chunks

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
    for _ in range(num_merges):
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
                    pos += 1  # Move past the merged token to avoid overlapping merges
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

    # ! SPEC: do not put the special tokens in the vocab

    return vocab, merges
