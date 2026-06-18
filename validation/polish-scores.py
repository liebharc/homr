"""
btrkeks/polish-scores benchmark: measures OMR quality on the Polish Scores dataset.

Data source: btrkeks/polish-scores (HuggingFace dataset).
Tool:        selectable via --tool (default: music21).
Check:       ned_benchmark.run_benchmark
"""

import argparse
import tempfile
from collections.abc import Iterator
from pathlib import Path

from validation.ned_benchmark import run_benchmark, update_ned_scores
from validation.tools import TOOLS


def _add_kern_header(kern: str) -> str:
    """Prepend **kern header lines when the dataset strips them."""
    for line in kern.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("!"):
            continue
        if not stripped.startswith("**"):
            n_spines = len(line.split("\t"))
            return "\t".join(["**kern"] * n_spines) + "\n" + kern
        break
    return kern


def iter_polish_scores(
    image_dir: Path | None = None,
) -> Iterator[tuple[str, str, Path | None]]:
    """Yield (sample_id, kern_text, image_path) for each sample in btrkeks/polish-scores."""
    from datasets import load_dataset  # type: ignore  # noqa: PLC0415

    ds = load_dataset("btrkeks/polish-scores")
    for i, sample in enumerate(ds["train"]):
        kern = _add_kern_header(sample["transcription_kern"])
        image_path: Path | None = None
        if image_dir is not None:
            img = sample.get("image")
            if img is not None and hasattr(img, "save"):
                candidate = image_dir / f"{i}.png"
                img.save(candidate)
                image_path = candidate
        yield str(i), kern, image_path


def main() -> None:
    parser = argparse.ArgumentParser(description="OMR-NED benchmark for btrkeks/polish-scores.")
    parser.add_argument("--limit", type=int, default=None, help="Only process N samples.")
    parser.add_argument("--verbose", action="store_true", help="Print traceback on failure.")
    parser.add_argument("--output", type=str, default=None, help="Path to SQLite output file.")
    parser.add_argument(
        "--update",
        action="store_true",
        help="Recompute NED from stored kern/output data without re-running the tool.",
    )
    parser.add_argument(
        "--continue",
        dest="continue_run",
        action="store_true",
        help="Skip samples already present in --output and append new results.",
    )
    parser.add_argument(
        "--tool",
        choices=list(TOOLS),
        default="music21",
        help="OMR tool to benchmark (default: music21).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Samples per batch for tools that support batch_run (default: 10).",
    )
    parser.add_argument(
        "--kern-parser",
        choices=["native", "music21"],
        default="native",
        dest="kern_parser",
        help=(
            "Parser for the kern ground-truth side of NED comparison. "
            "'native' (default) = built-in humdrum parser; 'music21' = music21-based parser."
        ),
    )
    parser.add_argument(
        "--xml-parser",
        choices=["native", "music21", "musicdiff", "musicdiff_detailed"],
        default="native",
        dest="xml_parser",
        help=(
            "Parser/method for the NED comparison. "
            "'native' (default) = built-in token pipeline; "
            "'music21' = music21-based token pipeline; "
            "'musicdiff' = musicdiff holistic OMR-NED (component NEDs not available)."
            "'musicdiff_detailed' = musicdiff holistic OMR-NED."
        ),
    )
    args = parser.parse_args()
    output_db = args.output or f"polish-scores_{args.tool}.db"

    if args.update:
        update_ned_scores(
            output_db,
            verbose=args.verbose,
            kern_parser=args.kern_parser,
            xml_parser=args.xml_parser,
            limit=args.limit,
        )
        return

    tool = TOOLS[args.tool]
    with tempfile.TemporaryDirectory() as image_dir:
        run_benchmark(
            iter_polish_scores(image_dir=Path(image_dir)),
            tool,
            limit=args.limit,
            verbose=args.verbose,
            output_db=output_db,
            continue_run=args.continue_run,
            batch_size=args.batch_size,
            kern_parser=args.kern_parser,
            xml_parser=args.xml_parser,
        )


if __name__ == "__main__":
    main()
