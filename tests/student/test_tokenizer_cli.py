import json
import pickle
import subprocess
import sys


def test_tokenizer_package_exposes_module_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "cs336_basics.tokenizer", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Train a BPE tokenizer" in result.stdout
    assert "--no-multiprocessing" in result.stdout
    assert "--no-progress" in result.stdout


def test_tokenizer_package_trains_and_writes_requested_outputs(tmp_path) -> None:
    corpus = tmp_path / "corpus.txt"
    vocab_path = tmp_path / "vocab.json"
    merges_path = tmp_path / "merges.pkl"
    corpus.write_text("hello hello<|endoftext|>world", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cs336_basics.tokenizer",
            str(corpus),
            "--vocab-size",
            "258",
            "--no-multiprocessing",
            "--no-progress",
            "--out-vocab",
            str(vocab_path),
            "--out-merges",
            str(merges_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    with vocab_path.open(encoding="utf-8") as file:
        vocab = json.load(file)
    with merges_path.open("rb") as file:
        merges = pickle.load(file)
    assert len(vocab) == 258
    assert len(merges) == 1
