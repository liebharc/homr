"""
Parser comparison: native vs music21 on ground-truth data (no OMR tool involved).

Kern mode (--dataset {polish-scores,smb}):
  Runs both kern parsers on each sample's kern ground truth and reports the
  NED between their outputs.

XML mode (--dataset lieder):
  Runs both XML parsers on each .musicxml file from the Lieder dataset and
  reports the NED between their outputs.

Usage:
  poetry run python -m validation.compare_parsers --dataset polish-scores [--limit 20]
  poetry run python -m validation.compare_parsers --dataset smb [--limit 20]
  poetry run python -m validation.compare_parsers --dataset lieder [--limit 20]
"""

import argparse
import sys
import traceback
from collections.abc import Callable, Iterator
from pathlib import Path

from homr.circle_of_fifths import strip_naturals
from homr.transformer.vocabulary import EncodedSymbol, sort_token_chords
from validation.ned_benchmark import _print_stats
from validation.ned_score import (
    NedResult,
    _kern_parts,
    _ned_from_parts,
    _strip_position,
    _xml_parts_from_text,
)

_REPO_ROOT = Path(__file__).parent.parent


def _norm_kern(parts: list[list[EncodedSymbol]]) -> list[list[EncodedSymbol]]:
    return [_strip_position([t for chord in sort_token_chords(p) for t in chord]) for p in parts]


def _norm_xml(parts: list[list[EncodedSymbol]]) -> list[list[EncodedSymbol]]:
    return [_strip_position(strip_naturals(p)) for p in parts]


def _add_kern_header(kern: str) -> str:
    for line in kern.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("!"):
            continue
        if not stripped.startswith("**"):
            n_spines = len(line.split("\t"))
            return "\t".join(["**kern"] * n_spines) + "\n" + kern
        break
    return kern


def _iter_polish_scores(limit: int | None) -> Iterator[tuple[str, str]]:
    from datasets import load_dataset  # type: ignore  # noqa: PLC0415

    ds = load_dataset("btrkeks/polish-scores")
    for i, sample in enumerate(ds["train"]):
        if limit is not None and i >= limit:
            break
        yield str(i), _add_kern_header(sample["transcription_kern"])


def _iter_smb(limit: int | None) -> Iterator[tuple[str, str]]:
    from datasets import load_dataset  # type: ignore  # noqa: PLC0415

    ds = load_dataset("PRAIG/SMB")
    count = 0
    for i, sample in enumerate(ds["test"]):
        kern = ""
        page = sample.get("page", {})
        if isinstance(page, dict):
            kern = page.get("kern") or ""
        if not kern:
            kern = sample.get("kern") or ""
        if not kern:
            parts_list = [r.get("kern", "") for r in sample.get("regions", []) if r.get("kern")]
            kern = "\n".join(parts_list)
        if not kern:
            continue
        if limit is not None and count >= limit:
            break
        yield str(i), kern
        count += 1


def _iter_lieder(limit: int | None) -> Iterator[tuple[str, str]]:
    lieder_dir = _REPO_ROOT / "datasets" / "Lieder-main" / "flat"
    if not lieder_dir.exists():
        raise FileNotFoundError(
            f"Lieder dataset not found at {lieder_dir}. "
            "Run python -m training.omr_datasets.convert_lieder first."
        )
    files = sorted(lieder_dir.glob("*.musicxml"))
    for i, path in enumerate(files):
        if limit is not None and i >= limit:
            break
        yield path.stem, path.read_text(encoding="utf-8")


def _compare_kern(kern_text: str) -> tuple[NedResult, bool, bool]:
    native = _kern_parts(kern_text, "native")
    m21 = _kern_parts(kern_text, "music21")
    return _ned_from_parts(_norm_kern(native), _norm_kern(m21)), bool(native), bool(m21)


def _compare_xml(xml_text: str) -> tuple[NedResult, bool, bool]:
    native = _xml_parts_from_text(xml_text, "native")
    m21 = _xml_parts_from_text(xml_text, "music21")
    return _ned_from_parts(_norm_xml(native), _norm_xml(m21)), bool(native), bool(m21)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare native vs music21 parsers on ground-truth data."
    )
    parser.add_argument(
        "--dataset",
        choices=["polish-scores", "smb", "lieder"],
        required=True,
        help="Dataset to compare parsers on.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only process N samples.")
    parser.add_argument("--verbose", action="store_true", help="Print full traceback on error.")
    args = parser.parse_args()

    if args.dataset == "polish-scores":
        samples: Iterator[tuple[str, str]] = _iter_polish_scores(args.limit)
        compare_fn: Callable[[str], tuple[NedResult, bool, bool]] = _compare_kern
        mode_label = "kern parsers (native vs music21)"
    elif args.dataset == "smb":
        samples = _iter_smb(args.limit)
        compare_fn = _compare_kern
        mode_label = "kern parsers (native vs music21)"
    else:
        samples = _iter_lieder(args.limit)
        compare_fn = _compare_xml
        mode_label = "XML parsers (native vs music21)"

    print(f"Parser comparison — {args.dataset} — {mode_label}")  # noqa: T201

    results: list[NedResult] = []
    errors: list[tuple[str, str]] = []
    native_empty = 0
    m21_empty = 0

    for sample_id, text in samples:
        try:
            result, native_ok, m21_ok = compare_fn(text)
            if not native_ok:
                native_empty += 1
            if not m21_ok:
                m21_empty += 1
            results.append(result)
            print(  # noqa: T201
                f"[{sample_id}] NED={result.ned * 100:5.1f}%  "
                f"rhythm={result.rhythm_ned * 100:5.1f}%  "
                f"pitch={result.pitch_ned * 100:5.1f}%  "
                f"lift={result.lift_ned * 100:5.1f}%  "
                f"art={result.articulation_ned * 100:5.1f}%  "
                f"slur={result.slur_ned * 100:5.1f}%  "
                f"native={result.kern_len:4d}  music21={result.xml_len:4d}"
                + ("  [native=empty]" if not native_ok else "")
                + ("  [music21=empty]" if not m21_ok else "")
            )
        except Exception as e:  # noqa: BLE001
            errors.append((sample_id, str(e)))
            print(f"[{sample_id}] ERROR: {e}", file=sys.stderr)  # noqa: T201
            if args.verbose:
                traceback.print_exc()

    print()  # noqa: T201
    print(  # noqa: T201
        f"Processed {len(results) + len(errors)} samples: "
        f"{len(results)} compared, {len(errors)} errors."
    )
    if native_empty:
        print(f"  native parser returned empty output: {native_empty} sample(s)")  # noqa: T201
    if m21_empty:
        print(f"  music21 parser returned empty output: {m21_empty} sample(s)")  # noqa: T201

    if results:
        print("\nAggregate Statistics (NED between native and music21 outputs):")  # noqa: T201
        _print_stats("Overall NED ", [r.ned for r in results])
        _print_stats("Rhythm NED  ", [r.rhythm_ned for r in results])
        _print_stats("Pitch NED   ", [r.pitch_ned for r in results])
        _print_stats("Lift NED    ", [r.lift_ned for r in results])
        _print_stats("Artic NED   ", [r.articulation_ned for r in results])
        _print_stats("Slur NED    ", [r.slur_ned for r in results])

    if errors:
        print(f"\nErrors ({len(errors)}):")  # noqa: T201
        for fid, err in errors:
            print(f"  [{fid}] {err}")  # noqa: T201


if __name__ == "__main__":
    main()
