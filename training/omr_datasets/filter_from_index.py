import glob
import os

from homr.simple_logging import eprint
from training.transformer.training_vocabulary import (
    VocabularyStats,
    check_token_lines,
    read_tokens,
)


def find_tokens() -> tuple[VocabularyStats, set[str]]:
    stats = VocabularyStats()
    errors: set[str] = set()
    exclude = "validation"
    files = [
        f
        for f in glob.glob(os.path.join("datasets", "**", "*.tokens"), recursive=True)
        if exclude not in f.split(os.sep)
    ]
    for i, file in enumerate(files):
        try:
            tokens = read_tokens(file)
            stats.add_lines(tokens)
            check_token_lines(tokens)
            if i % 1000 == 0:
                eprint(i, len(errors))
        except Exception as e:
            eprint("======", file, "======")
            eprint(e)
            errors.add(file)
    return stats, errors


def write_invalid_tokens(file_path: str) -> None:
    """Writes the token files which fail the vocabulary check, one per line."""
    _, errors = find_tokens()
    eprint(f"Skipping {len(errors)} token files with invalid tokens")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        f.writelines(os.path.normpath(e) + "\n" for e in sorted(errors))


def read_invalid_tokens(file_path: str) -> set[str]:
    with open(file_path) as f:
        return {line.strip() for line in f if line.strip()}


def without_invalid_tokens(index_lines: list[str], invalid: set[str]) -> list[str]:
    """Drops index lines which reference a file from invalid."""
    return [
        line
        for line in index_lines
        if not any(os.path.normpath(p) in invalid for p in line.strip().split(","))
    ]
