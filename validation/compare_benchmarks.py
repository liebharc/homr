"""
Compares two benchmark databases sample by sample and prints a Markdown report.

Usage:
    poetry run python -m validation.compare_benchmarks head.db --base base.db
"""

import argparse
import sqlite3
import sys

Row = tuple[str, float | None, str | None]


def _load(path: str) -> dict[str, Row]:
    with sqlite3.connect(path) as conn:
        rows = conn.execute("SELECT sample_id, actual_text, ned, error FROM samples").fetchall()
    return {sample_id: (text or "", ned, error) for sample_id, text, ned, error in rows}


def _mean_ned(samples: dict[str, Row]) -> float:
    values = [ned for _, ned, _ in samples.values() if ned is not None]
    return 100 * sum(values) / len(values) if values else float("nan")


def _failures(samples: dict[str, Row]) -> int:
    return sum(error is not None for _, _, error in samples.values())


def _format_ned(ned: float | None) -> str:
    return "failed" if ned is None else f"{100 * ned:.2f}%"


def summarize(head_path: str) -> str:
    head = _load(head_path)
    return "\n".join(
        [
            "| | Mean NED | Failures | Samples |",
            "|---|---|---|---|",
            f"| head | {_mean_ned(head):.2f}% | {_failures(head)} | {len(head)} |",
        ]
    )


def compare(base_path: str, head_path: str) -> str:
    base, head = _load(base_path), _load(head_path)
    missing = sorted(set(base) ^ set(head))
    common = sorted(set(base) & set(head), key=lambda s: (len(s), s))
    changed = [s for s in common if base[s] != head[s]]
    delta = _mean_ned(head) - _mean_ned(base)

    lines = [
        "| | Mean NED | Failures | Samples |",
        "|---|---|---|---|",
        f"| base | {_mean_ned(base):.2f}% | {_failures(base)} | {len(base)} |",
        f"| head | {_mean_ned(head):.2f}% | {_failures(head)} | {len(head)} |",
        "",
        f"Mean NED change: **{delta:+.2f}** points (lower is better).",
        f"{len(changed)} of {len(common)} samples have a different output.",
    ]
    if missing:
        lines.append(f"Samples only in one run: {', '.join(missing)}")
    if changed:
        lines += ["", "| Sample | base | head |", "|---|---|---|"]
        lines += [
            f"| {s} | {_format_ned(base[s][1])} | {_format_ned(head[s][1])} |" for s in changed
        ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two benchmark databases.")
    parser.add_argument("head", help="SQLite output of the run to check.")
    parser.add_argument("--base", help="SQLite output of the reference run.")
    args = parser.parse_args()
    report = compare(args.base, args.head) if args.base else summarize(args.head)
    sys.stdout.write(report + "\n")


if __name__ == "__main__":
    main()
