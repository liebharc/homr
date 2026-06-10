#!/usr/bin/env python3
"""
Create deterministic train/validation/test JSONL splits from a Track C
training_manifest.jsonl.

The split unit is score_id, not page_id. This prevents pages from the same
source score from leaking across train/validation/test sets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


class SplitError(RuntimeError):
    """Raised when manifest splitting cannot be completed safely."""


def emit_event(payload: dict[str, Any], *, quiet: bool = False, stderr: bool = False) -> None:
    if quiet:
        return
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), file=sys.stderr if stderr else sys.stdout, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create score-level train/validation/test splits from "
            "distillation/batches/<batch>/training_manifest.jsonl."
        )
    )
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="Input training_manifest.jsonl.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Defaults to the input manifest directory. "
            "Writes training_manifest.train.jsonl, .val.jsonl, .test.jsonl, "
            "and split_summary.json."
        ),
    )
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Deterministic seed used only for stable score_id tie-breaking.",
    )
    parser.add_argument(
        "--score-field",
        default="score_id",
        help="Manifest field that identifies the source score/group.",
    )
    parser.add_argument(
        "--allow-missing-score-id",
        action="store_true",
        help=(
            "Use page_id or manifest line number as a fallback group key when "
            "score_id is missing. Off by default because this can hide leakage."
        ),
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the JSON summary to stdout and split_summary.json.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress prints. Final summary is still printed.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SplitError(f"Manifest does not exist: {path}")
    if not path.is_file():
        raise SplitError(f"Manifest path is not a file: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise SplitError(
                    f"Invalid JSON on line {line_number} of {path}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise SplitError(
                    f"Line {line_number} is {type(row).__name__}, expected object."
                )
            row["_manifest_line_number"] = line_number
            rows.append(row)

    if not rows:
        raise SplitError(f"Manifest has no non-empty JSONL rows: {path}")
    return rows


def stable_hash_int(value: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()
    return int(digest, 16)


def get_group_key(
    row: dict[str, Any],
    *,
    score_field: str,
    allow_missing_score_id: bool,
) -> str:
    value = row.get(score_field)
    if isinstance(value, str) and value.strip():
        return value

    if not allow_missing_score_id:
        line = row.get("_manifest_line_number", "unknown")
        raise SplitError(
            f"Manifest row {line} is missing non-empty {score_field!r}. "
            "Refusing to split by page because that can leak pages from the "
            "same source score across splits."
        )

    page_id = row.get("page_id")
    if isinstance(page_id, str) and page_id.strip():
        return f"fallback_page_id:{page_id}"
    return f"fallback_line:{row.get('_manifest_line_number', 'unknown')}"


def group_rows(
    rows: list[dict[str, Any]],
    *,
    score_field: str,
    allow_missing_score_id: bool,
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = get_group_key(
            row,
            score_field=score_field,
            allow_missing_score_id=allow_missing_score_id,
        )
        groups[key].append(row)
    return dict(groups)


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    ratios = {
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
    }
    for name, value in ratios.items():
        if value < 0:
            raise SplitError(f"{name} must be non-negative, got {value}")
    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise SplitError("At least one split ratio must be positive.")
    if abs(total - 1.0) > 1e-9:
        raise SplitError(
            f"Split ratios must sum to 1.0, got {total:.12g} "
            f"from {ratios}."
        )


def target_counts(total_rows: int, ratios: dict[str, float]) -> dict[str, float]:
    return {name: total_rows * ratio for name, ratio in ratios.items()}


def choose_best_split(
    counts: dict[str, int],
    group_size: int,
    targets: dict[str, float],
    active_splits: list[str],
) -> str:
    """Choose the split that minimizes squared target error after assignment."""

    best_name: str | None = None
    best_score: tuple[float, int, str] | None = None

    for name in active_splits:
        trial = dict(counts)
        trial[name] += group_size
        error = sum((trial[split] - targets[split]) ** 2 for split in active_splits)
        # Tie-break toward the currently smaller split, then lexical stability.
        score = (error, counts[name], name)
        if best_score is None or score < best_score:
            best_score = score
            best_name = name

    if best_name is None:
        raise SplitError("No active output split exists. Check split ratios.")
    return best_name


def assign_groups(
    groups: dict[str, list[dict[str, Any]]],
    *,
    ratios: dict[str, float],
    seed: int,
) -> dict[str, list[str]]:
    active_splits = [name for name in ("train", "val", "test") if ratios[name] > 0]
    if not active_splits:
        raise SplitError("No active splits; all ratios were zero.")
    if len(groups) < len(active_splits):
        raise SplitError(
            f"Need at least {len(active_splits)} score groups for non-empty "
            f"active splits {active_splits}, but found {len(groups)}."
        )

    total_rows = sum(len(rows) for rows in groups.values())
    targets = target_counts(total_rows, ratios)

    # Assign the largest groups first, but do not force the largest groups into
    # validation/test. On small page-level corpora, a score can contain more rows
    # than the entire validation target, so forcing largest groups into every
    # split gives unnecessarily bad splits. The objective below naturally fills
    # train first until doing so becomes worse than filling val/test.
    ordered_group_ids = sorted(
        groups,
        key=lambda key: (-len(groups[key]), stable_hash_int(key, seed), key),
    )

    assignments: dict[str, list[str]] = {name: [] for name in ("train", "val", "test")}
    counts: dict[str, int] = {name: 0 for name in ("train", "val", "test")}

    for group_id in ordered_group_ids:
        group_size = len(groups[group_id])
        split_name = choose_best_split(counts, group_size, targets, active_splits)
        assignments[split_name].append(group_id)
        counts[split_name] += group_size

    # Safety repair: if a positive-ratio split ended empty, move the smallest
    # group from the active split with the most groups. This should be rare, but
    # it makes the non-empty split contract explicit for tiny smoke manifests.
    for empty_split in active_splits:
        if assignments[empty_split]:
            continue

        donor_candidates = [
            split for split in active_splits if len(assignments[split]) > 1
        ]
        if not donor_candidates:
            raise SplitError(
                f"Could not make split {empty_split!r} non-empty without "
                "emptying another active split."
            )

        donor = max(
            donor_candidates,
            key=lambda split: (counts[split], len(assignments[split]), split),
        )
        moved_group = min(
            assignments[donor],
            key=lambda key: (len(groups[key]), stable_hash_int(key, seed), key),
        )
        assignments[donor].remove(moved_group)
        assignments[empty_split].append(moved_group)
        counts[donor] -= len(groups[moved_group])
        counts[empty_split] += len(groups[moved_group])

    return assignments

def strip_internal_fields(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(strip_internal_fields(row), ensure_ascii=False, sort_keys=True))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def write_json_atomic(path: Path, payload: dict[str, Any], *, pretty: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2 if pretty else None, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def validate_no_leakage(split_rows: dict[str, list[dict[str, Any]]], score_field: str) -> None:
    score_to_splits: dict[str, set[str]] = defaultdict(set)
    for split_name, rows in split_rows.items():
        for row in rows:
            value = row.get(score_field)
            if isinstance(value, str) and value.strip():
                score_to_splits[value].add(split_name)

    leaked = {
        score_id: sorted(split_names)
        for score_id, split_names in score_to_splits.items()
        if len(split_names) > 1
    }
    if leaked:
        raise SplitError(f"Score-level leakage detected: {leaked}")


def summarize_split(rows: list[dict[str, Any]], score_field: str) -> dict[str, Any]:
    score_ids = [row.get(score_field) for row in rows]
    score_counts = Counter(score_ids)
    token_counts = [int(row.get("n_target_tokens", 0) or 0) for row in rows]
    return {
        "rows": len(rows),
        "scores": len(score_counts),
        "target_tokens": sum(token_counts),
        "score_ids": sorted(str(score_id) for score_id in score_counts),
        "largest_score_page_counts": [
            {"score_id": str(score_id), "pages": count}
            for score_id, count in score_counts.most_common(10)
        ],
    }


def make_splits(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    manifest_path = args.manifest
    out_dir = args.out_dir or manifest_path.parent
    ratios = {
        "train": args.train_ratio,
        "val": args.val_ratio,
        "test": args.test_ratio,
    }

    emit_event(
        {
            "event": "start",
            "manifest": str(manifest_path),
            "out_dir": str(out_dir),
            "ratios": ratios,
            "score_field": args.score_field,
        },
        quiet=args.quiet,
    )

    rows = read_jsonl(manifest_path)
    emit_event({"event": "loaded", "rows": len(rows), "manifest": str(manifest_path)}, quiet=args.quiet)
    groups = group_rows(
        rows,
        score_field=args.score_field,
        allow_missing_score_id=args.allow_missing_score_id,
    )
    emit_event({"event": "grouped", "score_groups": len(groups), "rows": len(rows)}, quiet=args.quiet)
    assignments = assign_groups(groups, ratios=ratios, seed=args.seed)

    split_rows: dict[str, list[dict[str, Any]]] = {}
    for split_name, group_ids in assignments.items():
        selected: list[dict[str, Any]] = []
        for group_id in group_ids:
            selected.extend(groups[group_id])
        selected.sort(key=lambda row: int(row.get("_manifest_line_number", 0)))
        split_rows[split_name] = selected

    validate_no_leakage(split_rows, args.score_field)

    output_paths = {
        "train": out_dir / "training_manifest.train.jsonl",
        "val": out_dir / "training_manifest.val.jsonl",
        "test": out_dir / "training_manifest.test.jsonl",
    }
    for split_name, path in output_paths.items():
        write_jsonl_atomic(path, split_rows[split_name])

    total_written = sum(len(rows_for_split) for rows_for_split in split_rows.values())
    if total_written != len(rows):
        raise SplitError(f"Wrote {total_written} rows, expected {len(rows)}")

    summary = {
        "status": "ok",
        "manifest": str(manifest_path),
        "out_dir": str(out_dir),
        "seed": args.seed,
        "score_field": args.score_field,
        "ratios": ratios,
        "rows_total": len(rows),
        "score_groups_total": len(groups),
        "outputs": {name: str(path) for name, path in output_paths.items()},
        "splits": {
            split_name: summarize_split(rows_for_split, args.score_field)
            for split_name, rows_for_split in split_rows.items()
        },
        "leakage_check": "passed",
        "seconds": time.time() - started,
    }

    summary_path = out_dir / "split_summary.json"
    write_json_atomic(summary_path, summary, pretty=args.pretty)
    summary["outputs"]["summary"] = str(summary_path)
    write_json_atomic(summary_path, summary, pretty=args.pretty)

    return summary


def main() -> int:
    args = parse_args()
    try:
        summary = make_splits(args)
    except SplitError as exc:
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

    print(json.dumps(summary, indent=2 if args.pretty else None, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
