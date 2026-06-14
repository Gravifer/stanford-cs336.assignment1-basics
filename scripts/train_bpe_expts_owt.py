#!/usr/bin/env python3
import json
import pickle
import resource
import time

from cs336_basics.common import PeakMemoryMonitor, prettyprint_vocab
from cs336_basics.tokenizer import train_bpe

# TODO: enforce memory limit in MB
MEMORY_LIMIT_MB = 16000

if __name__ != "__main__":
    print("This script should not be imported")
    quit(-1)

f = "data/owt_train.txt"
ru0 = resource.getrusage(resource.RUSAGE_SELF)
t0 = time.perf_counter()
with PeakMemoryMonitor() as mem:
    vocab, merges = train_bpe(f, 32000, ["<|endoftext|>"], multiprocessing=16)

wall_time = time.perf_counter() - t0
ru1 = resource.getrusage(resource.RUSAGE_SELF)
cpu_user = ru1.ru_utime - ru0.ru_utime
cpu_sys = ru1.ru_stime - ru0.ru_stime
print(f"CPU times: user {cpu_user:.2f}s, sys: {cpu_sys:.2f}s, total: {cpu_user + cpu_sys:.2f}s")
print(f"Wall time: {wall_time:.1f}s")
print(f"Peak RAM (main + workers): {mem.peak_mb:.0f} MB")
prettyprint_vocab(vocab, cols=10, col_width=13)
out_vocab = "vocab.json"
out_merges = "merges.pkl"
with open(out_vocab, "w", encoding="utf-8") as f:
    json.dump({str(k): v.decode("utf-8", errors="replace") for k, v in vocab.items()}, f, ensure_ascii=False, indent=2)
with open(out_merges, "wb") as f:
    pickle.dump(merges, f)
