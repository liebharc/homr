"""
liebharc/sheet-music-photos benchmark: measures OMR quality on phone photos of real scans.

Data source: https://github.com/liebharc/sheet-music-photos, cloned to
             datasets/sheet-music-photos on first use.
Tool:        selectable via --tool (default: homr).
Check:       ned_benchmark.run_benchmark

Ground truth is page.musicxml (verified) if present, else reference.musicxml (homr's
reading of the clean scan). Results are mean NEDs against these transcriptions.
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from validation.ned_benchmark import Sample, run_benchmark, update_ned_scores
from validation.tools import TOOLS

_REPO_URL = "https://github.com/liebharc/sheet-music-photos.git"
_DEFAULT_DIR = Path("datasets/sheet-music-photos")
_SCAN_DPI = 200  # as for the references in the dataset


def _ensure_dataset(dataset: Path) -> None:
    if not (dataset / "scans").is_dir():
        subprocess.run(  # noqa: S603
            ["git", "clone", "--depth", "1", _REPO_URL, str(dataset)], check=True  # noqa: S607
        )


def _ground_truth(scan: Path, verified_only: bool) -> str | None:
    for name in ("page.musicxml",) if verified_only else ("page.musicxml", "reference.musicxml"):
        path = scan / name
        if path.exists():
            return path.read_text(encoding="utf-8")
    return None


def _rasterize(pdf: Path, out: Path) -> None:
    import pypdfium2 as pdfium  # noqa: PLC0415

    document = pdfium.PdfDocument(pdf)
    document[0].render(scale=_SCAN_DPI / 72, grayscale=True).to_pil().save(out)
    document.close()


def get_samples(
    dataset: Path, image_dir: Path, only: str | None, verified_only: bool
) -> list[Sample]:
    """One sample per clean scan and one per photo; only="scans"/"photos" keeps one kind."""
    result = []
    for scan in sorted(p for p in (dataset / "scans").iterdir() if p.is_dir()):
        expected = _ground_truth(scan, verified_only)
        if expected is None:
            continue
        if only != "photos":
            image = image_dir / f"{scan.name}__scan.png"
            _rasterize(scan / "page.pdf", image)
            result.append(Sample(image.stem, expected, image))
        photos_json = scan / "photos" / "photos.json"
        if only != "scans" and photos_json.exists():
            for photo in json.loads(photos_json.read_text(encoding="utf-8"))["photos"]:
                image = scan / "photos" / photo["file"]
                result.append(Sample(f"{scan.name}__{image.stem}", expected, image))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="OMR-NED benchmark for sheet-music-photos.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=_DEFAULT_DIR,
        help=f"Checkout of sheet-music-photos (default: {_DEFAULT_DIR}, cloned if missing).",
    )
    parser.add_argument(
        "--only", choices=["scans", "photos"], default=None, help="Only clean scans or photos."
    )
    parser.add_argument(
        "--verified",
        action="store_true",
        help="Only use pages with a verified page.musicxml.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only process N samples.")
    parser.add_argument("--workers", type=int, default=1, help="Number of threads to use.")
    parser.add_argument("--verbose", action="store_true", help="Print traceback on failure.")
    parser.add_argument("--output", type=str, default=None, help="Path to SQLite output file.")
    parser.add_argument(
        "--update",
        action="store_true",
        help="Recompute NED from stored ground truth/output without re-running the tool.",
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
        default="homr",
        help="OMR tool to benchmark (default: homr).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Samples per batch for tools that support batch_run (default: 10).",
    )
    parser.add_argument(
        "--xml-parser",
        choices=["native", "music21", "musicdiff", "musicdiff_detailed"],
        default="native",
        dest="xml_parser",
        help="Parser/method for the NED comparison, see validation.smb.",
    )
    args = parser.parse_args()
    output_db = args.output or f"sheet-music-photos_{args.tool}.db"

    if args.update:
        update_ned_scores(
            output_db, verbose=args.verbose, xml_parser=args.xml_parser, limit=args.limit
        )
        return

    _ensure_dataset(args.dataset)
    tool = TOOLS[args.tool]
    with tempfile.TemporaryDirectory() as image_dir:
        run_benchmark(
            get_samples(args.dataset, Path(image_dir), args.only, args.verified),
            tool,
            args.workers,
            limit=args.limit,
            verbose=args.verbose,
            output_db=output_db,
            continue_run=args.continue_run,
            batch_size=args.batch_size,
            xml_parser=args.xml_parser,
        )


if __name__ == "__main__":
    main()
