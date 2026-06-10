#!/usr/bin/env python3
"""
Load and validate Track C distillation training-manifest rows.

This module is the training-side bridge between the generated
training_manifest.jsonl and future student-model training code. It intentionally
does not render scores, run HOMR, run ONNX teacher inference, build teacher
outputs, or train a model.

It can also be run as a bounded CLI sanity check using --max-items. That keeps
small validation runs controlled by shell arguments instead of requiring a
separate smoke-only source file.

Expected primary manifest field:
    target_flat_tokens_with_staff_breaks

The implementation is deliberately strict about missing required data, but
flexible about image-path field names so it can tolerate small manifest schema
changes while the Track C pipeline is still being built.
"""

from __future__ import annotations

import argparse
import json
import statistics
import struct
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


SPECIAL_TOKENS = ("<PAD>", "<BOS>", "<EOS>", "<UNK>")
TOKEN_FIELD_CANDIDATES = (
    "target_flat_tokens_with_staff_breaks",
    "target_flat_tokens",
    "target_tokens",
    "tokens",
)
IMAGE_FIELD_CANDIDATES = (
    "rendered_image_path",
    "image_path",
    "page_image_path",
    "rendered_page_path",
    "rendered_path",
    "input_image_path",
    "source_image_path",
    "png_path",
)
NESTED_IMAGE_FIELD_CANDIDATES = (
    ("render", "image_path"),
    ("render", "rendered_image_path"),
    ("rendered", "image_path"),
    ("page", "image_path"),
    ("source", "image_path"),
)
NESTED_TOKEN_FIELD_CANDIDATES = (
    ("canonical", "target_flat_tokens_with_staff_breaks"),
    ("canonical", "target_flat_tokens"),
    ("canonical", "tokens"),
    ("target", "flat_tokens_with_staff_breaks"),
    ("target", "tokens"),
)


class ManifestError(RuntimeError):
    """Raised when the manifest cannot be used as a training dataset."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a Track C training_manifest.jsonl by reading rows, "
            "loading rendered PNGs, building a target vocabulary, and checking "
            "token encode/decode round-trips."
        )
    )
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="Path to distillation/batches/<batch_id>/training_manifest.jsonl.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=5,
        help="Number of manifest rows whose images and round-trips are reported.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help=(
            "Repository root used to resolve relative paths. Defaults to the "
            "current working directory."
        ),
    )
    parser.add_argument(
        "--require-all-images",
        action="store_true",
        help="Load every manifest image, not just the first --max-items rows.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise ManifestError(f"Manifest does not exist: {path}")
    if not path.is_file():
        raise ManifestError(f"Manifest path is not a file: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ManifestError(
                    f"Invalid JSON on line {line_number} of {path}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ManifestError(
                    f"Line {line_number} is {type(value).__name__}, expected object."
                )
            value["_manifest_line_number"] = line_number
            rows.append(value)

    if not rows:
        raise ManifestError(f"Manifest has no non-empty JSONL rows: {path}")
    return rows


def get_nested(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    current: Any = row
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def normalize_tokens(raw: Any, *, row_index: int, field_name: str) -> list[str]:
    if raw is None:
        raise ManifestError(f"Row {row_index} has no token field.")

    if isinstance(raw, list):
        tokens = [str(token) for token in raw]
    elif isinstance(raw, str):
        # Prefer JSON list strings when present; otherwise fall back to whitespace
        # tokenization. The primary manifest field should normally already be a list.
        stripped = raw.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                tokens = [str(token) for token in parsed]
            else:
                tokens = stripped.split()
        else:
            tokens = stripped.split()
    else:
        raise ManifestError(
            f"Row {row_index} token field {field_name!r} is "
            f"{type(raw).__name__}, expected list[str] or string."
        )

    if not tokens:
        raise ManifestError(f"Row {row_index} has an empty token sequence.")
    return tokens


def extract_tokens(row: dict[str, Any], *, row_index: int) -> tuple[list[str], str]:
    for field in TOKEN_FIELD_CANDIDATES:
        if field in row:
            return normalize_tokens(row[field], row_index=row_index, field_name=field), field

    for nested in NESTED_TOKEN_FIELD_CANDIDATES:
        raw = get_nested(row, nested)
        if raw is not None:
            field_name = ".".join(nested)
            return normalize_tokens(raw, row_index=row_index, field_name=field_name), field_name

    raise ManifestError(
        f"Row {row_index} is missing target tokens. Tried fields: "
        f"{', '.join(TOKEN_FIELD_CANDIDATES)} and nested canonical/target variants."
    )


def extract_image_path_value(row: dict[str, Any]) -> tuple[str | None, str | None]:
    for field in IMAGE_FIELD_CANDIDATES:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value, field

    for nested in NESTED_IMAGE_FIELD_CANDIDATES:
        value = get_nested(row, nested)
        if isinstance(value, str) and value.strip():
            return value, ".".join(nested)

    return None, None


def candidate_roots(manifest_path: Path, repo_root: Path | None) -> list[Path]:
    roots: list[Path] = []
    if repo_root is not None:
        roots.append(repo_root)
    roots.append(Path.cwd())
    roots.append(manifest_path.parent)

    # Include ancestors of the manifest path so paths relative to batch root,
    # distillation/, or repository root can all be resolved.
    for parent in manifest_path.resolve().parents:
        roots.append(parent)

    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        try:
            key = str(root.resolve())
        except OSError:
            key = str(root)
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique


def resolve_path(raw_path: str, manifest_path: Path, repo_root: Path | None) -> Path | None:
    path = Path(raw_path)
    if path.is_absolute():
        return path if path.exists() else None

    normalized = Path(raw_path.replace("\\", "/"))
    for root in candidate_roots(manifest_path, repo_root):
        candidate = root / normalized
        if candidate.exists():
            return candidate
    return None


def read_png_shape(path: Path) -> tuple[int, int, int]:
    with path.open("rb") as handle:
        signature = handle.read(8)
        if signature != b"\x89PNG\r\n\x1a\n":
            raise ManifestError(f"Not a PNG file: {path}")

        length_bytes = handle.read(4)
        chunk_type = handle.read(4)
        if len(length_bytes) != 4 or chunk_type != b"IHDR":
            raise ManifestError(f"PNG file has no IHDR chunk: {path}")

        length = struct.unpack(">I", length_bytes)[0]
        if length < 13:
            raise ManifestError(f"PNG IHDR chunk is too short in: {path}")

        ihdr = handle.read(13)
        if len(ihdr) != 13:
            raise ManifestError(f"Could not read PNG IHDR data from: {path}")

    width, height, bit_depth, color_type, _compression, _filter, _interlace = struct.unpack(
        ">IIBBBBB", ihdr
    )

    channels_by_color_type = {
        0: 1,  # grayscale
        2: 3,  # truecolor
        3: 1,  # indexed-color
        4: 2,  # grayscale + alpha
        6: 4,  # truecolor + alpha
    }
    channels = channels_by_color_type.get(color_type)
    if channels is None:
        raise ManifestError(f"Unsupported PNG color type {color_type} in: {path}")

    if bit_depth not in (1, 2, 4, 8, 16):
        raise ManifestError(f"Unsupported PNG bit depth {bit_depth} in: {path}")

    return int(height), int(width), int(channels)


def load_image_shape(path: Path) -> dict[str, Any]:
    # Pillow is helpful when available because it also verifies that the image is
    # decodable. The stdlib PNG reader keeps this script usable without adding a
    # dependency just for dataset validation.
    try:
        from PIL import Image  # type: ignore
    except Exception:
        height, width, channels = read_png_shape(path)
        return {
            "height": height,
            "width": width,
            "channels": channels,
            "loader": "stdlib_png_header",
        }

    with Image.open(path) as image:
        image.load()
        width, height = image.size
        mode = image.mode

    channels_by_mode = {
        "1": 1,
        "L": 1,
        "P": 1,
        "RGB": 3,
        "RGBA": 4,
        "CMYK": 4,
        "LA": 2,
        "I": 1,
        "F": 1,
    }
    channels = channels_by_mode.get(mode)
    return {
        "height": int(height),
        "width": int(width),
        "channels": int(channels) if channels is not None else None,
        "mode": mode,
        "loader": "PIL",
    }


def build_vocab(sequences: Iterable[list[str]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for sequence in sequences:
        counter.update(sequence)

    vocab: dict[str, int] = {}
    for token in SPECIAL_TOKENS:
        vocab[token] = len(vocab)

    # Deterministic ordering: frequent tokens first, then lexical order.
    for token, _count in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
        if token not in vocab:
            vocab[token] = len(vocab)

    return vocab


def encode(tokens: list[str], vocab: dict[str, int]) -> list[int]:
    unk_id = vocab["<UNK>"]
    return [vocab.get(token, unk_id) for token in tokens]


def decode(ids: list[int], id_to_token: dict[int, str]) -> list[str]:
    return [id_to_token[index] for index in ids]


def short_row_id(row: dict[str, Any], fallback_index: int) -> dict[str, Any]:
    keys = (
        "page_id",
        "score_id",
        "source_id",
        "page_number",
        "teacher_output_path",
        "source_path",
    )
    result = {key: row[key] for key in keys if key in row}
    result["row_index"] = fallback_index
    result["manifest_line_number"] = row.get("_manifest_line_number")
    return result


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.manifest
    repo_root = args.repo_root if args.repo_root is not None else Path.cwd()

    rows = read_jsonl(manifest_path)

    token_sequences: list[list[str]] = []
    token_fields: Counter[str] = Counter()
    for index, row in enumerate(rows):
        tokens, token_field = extract_tokens(row, row_index=index)
        token_sequences.append(tokens)
        token_fields[token_field] += 1

    vocab = build_vocab(token_sequences)
    id_to_token = {index: token for token, index in vocab.items()}

    max_items = max(0, args.max_items)
    sampled_rows = rows[:max_items]
    sampled_tokens = token_sequences[:max_items]
    rows_to_load = rows if args.require_all_images else sampled_rows

    image_field_counts: Counter[str] = Counter()
    image_reports: list[dict[str, Any]] = []
    missing_image_paths: list[dict[str, Any]] = []

    for index, row in enumerate(rows_to_load):
        raw_path, image_field = extract_image_path_value(row)
        if image_field is not None:
            image_field_counts[image_field] += 1

        report: dict[str, Any] = {
            "row": short_row_id(row, index),
            "image_field": image_field,
            "raw_image_path": raw_path,
        }

        if raw_path is None:
            report["status"] = "missing_image_path_field"
            missing_image_paths.append(report)
            image_reports.append(report)
            continue

        resolved = resolve_path(raw_path, manifest_path, repo_root)
        if resolved is None:
            report["status"] = "image_not_found"
            missing_image_paths.append(report)
            image_reports.append(report)
            continue

        shape = load_image_shape(resolved)
        report.update(
            {
                "status": "ok",
                "resolved_image_path": str(resolved),
                "shape_hwc": [shape["height"], shape["width"], shape["channels"]],
                "loader": shape["loader"],
            }
        )
        if "mode" in shape:
            report["mode"] = shape["mode"]
        image_reports.append(report)

    round_trips: list[dict[str, Any]] = []
    for index, (row, tokens) in enumerate(zip(sampled_rows, sampled_tokens)):
        encoded = encode(tokens, vocab)
        decoded = decode(encoded, id_to_token)
        round_trips.append(
            {
                "row": short_row_id(row, index),
                "token_count": len(tokens),
                "encoded_count": len(encoded),
                "round_trip_ok": decoded == tokens,
                "first_tokens": tokens[:20],
                "first_token_ids": encoded[:20],
            }
        )

    token_lengths = [len(tokens) for tokens in token_sequences]
    image_failures = [report for report in image_reports if report.get("status") != "ok"]
    round_trip_failures = [
        report for report in round_trips if not bool(report.get("round_trip_ok"))
    ]

    summary = {
        "status": "ok"
        if not image_failures and not round_trip_failures
        else "failed_checks",
        "manifest": str(manifest_path),
        "repo_root": str(repo_root),
        "rows_total": len(rows),
        "rows_sampled": len(sampled_rows),
        "images_checked": len(rows_to_load),
        "images_ok": len(image_reports) - len(image_failures),
        "images_failed": len(image_failures),
        "token_field_counts": dict(token_fields),
        "image_field_counts": dict(image_field_counts),
        "target_token_counts": {
            "total": sum(token_lengths),
            "min": min(token_lengths),
            "max": max(token_lengths),
            "mean": statistics.fmean(token_lengths),
            "median": statistics.median(token_lengths),
        },
        "vocab": {
            "size": len(vocab),
            "special_tokens": list(SPECIAL_TOKENS),
            "pad_id": vocab["<PAD>"],
            "bos_id": vocab["<BOS>"],
            "eos_id": vocab["<EOS>"],
            "unk_id": vocab["<UNK>"],
            "first_50_tokens_by_id": [
                token for token, _index in sorted(vocab.items(), key=lambda item: item[1])[:50]
            ],
        },
        "round_trips": round_trips,
        "image_reports": image_reports,
    }

    if image_failures:
        summary["image_failures"] = image_failures
    if round_trip_failures:
        summary["round_trip_failures"] = round_trip_failures

    return summary


def main() -> int:
    args = parse_args()

    try:
        summary = summarize(args)
    except ManifestError as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": exc.__class__.__name__,
                    "error": str(exc),
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2

    indent = 2 if args.pretty else None
    print(json.dumps(summary, indent=indent, sort_keys=True))

    return 0 if summary["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
