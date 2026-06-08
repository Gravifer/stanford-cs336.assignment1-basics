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
