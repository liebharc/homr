from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def format_duration(seconds: float) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "unknown"

    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    if hours > 0:
        return f"{hours}h {minutes}m {secs:04.1f}s"
    if minutes > 0:
        return f"{minutes}m {secs:04.1f}s"
    return f"{secs:.1f}s"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            text = line.strip()

            if not text:
                continue

            try:
                record = json.loads(text)
            except json.JSONDecodeError:
                print(
                    f"[warn] Ignoring unreadable JSONL line {line_number}: {path}",
                    file=sys.stderr,
                )
                continue

            if isinstance(record, dict):
                records.append(record)

    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_path(path: str | Path) -> str:
    return str(path).replace("\\", "/")


def load_source_pool(source_pool: Path) -> list[dict[str, Any]]:
    records = read_jsonl(source_pool)

    valid: list[dict[str, Any]] = []

    for record in records:
        source_path = record.get("source_path")
        score_id = record.get("score_id")

        if not isinstance(source_path, str) or not source_path:
            continue

        if not isinstance(score_id, str) or not score_id:
            score_id = Path(source_path).stem
            record["score_id"] = score_id

        valid.append(record)

    if not valid:
        raise ValueError(f"No valid source records found in {source_pool}")

    return valid


def discover_existing_batch_manifests(batch_root: Path) -> list[Path]:
    if not batch_root.exists():
        return []

    return sorted(batch_root.glob("*/batch_manifest.jsonl"))


def load_used_sources(
    *,
    batch_root: Path,
    current_batch_manifest: Path,
) -> tuple[set[str], set[str]]:
    """
    Return:
        used_source_paths, used_score_ids

    The current batch manifest is excluded so a partially written current batch
    can be resumed cleanly.
    """
    used_source_paths: set[str] = set()
    used_score_ids: set[str] = set()

    for manifest_path in discover_existing_batch_manifests(batch_root):
        if manifest_path.resolve() == current_batch_manifest.resolve():
            continue

        for record in read_jsonl(manifest_path):
            source_path = record.get("source_path")
            score_id = record.get("score_id")

            if isinstance(source_path, str) and source_path:
                used_source_paths.add(source_path)

            if isinstance(score_id, str) and score_id:
                used_score_ids.add(score_id)

    return used_source_paths, used_score_ids


def prepare_resume_state(
    *,
    batch_manifest: Path,
    overwrite: bool,
    valid_source_paths: set[str],
) -> tuple[list[dict[str, Any]], set[str], str | None]:
    """
    Return:
        kept_records, already_selected_source_paths, redo_source_path

    The manifest itself is the log. On resume, all valid records are kept except
    the last valid record, which is removed and recomputed.
    """
    if batch_manifest.exists() and overwrite:
        batch_manifest.unlink()
        return [], set(), None

    existing_records = read_jsonl(batch_manifest)

    if not existing_records:
        return [], set(), None

    valid_records = [
        record
        for record in existing_records
        if isinstance(record.get("source_path"), str)
        and record["source_path"] in valid_source_paths
    ]

    if not valid_records:
        write_jsonl(batch_manifest, [])
        return [], set(), None

    redo_source_path = valid_records[-1]["source_path"]
    kept_records = valid_records[:-1]

    write_jsonl(batch_manifest, kept_records)

    already_selected = {
        record["source_path"]
        for record in kept_records
        if isinstance(record.get("source_path"), str)
    }

    return kept_records, already_selected, redo_source_path


def make_batch_record(
    *,
    source_record: dict[str, Any],
    batch_id: str,
    batch_dir: Path,
    sequence_index: int,
) -> dict[str, Any]:
    score_id = str(source_record["score_id"])
    source_path = normalize_path(source_record["source_path"])

    rendered_image_path = batch_dir / "rendered_images" / f"{score_id}.png"
    homr_output_path = batch_dir / "homr_outputs" / f"{score_id}.musicxml"
    canonical_target_path = batch_dir / "canonical_targets" / f"{score_id}.json"

    return {
        "batch_id": batch_id,
        "batch_sequence_index": int(sequence_index),
        "score_id": score_id,
        "source_path": source_path,
        "source_relative_path": source_record.get("source_relative_path"),
        "source_ext": source_record.get("source_ext"),
        "source_size_bytes": source_record.get("source_size_bytes"),
        "rendered_image_path": normalize_path(rendered_image_path),
        "homr_output_path": normalize_path(homr_output_path),
        "canonical_target_path": normalize_path(canonical_target_path),
        "selected_at": now_iso(),
    }


def print_progress(
    *,
    done_count: int,
    target_count: int,
    processed_this_run: int,
    start_time: float,
    last_score_id: str,
) -> None:
    elapsed_sec = time.perf_counter() - start_time
    speed = processed_this_run / elapsed_sec if elapsed_sec > 0 else 0.0
    remaining = max(0, target_count - done_count)
    eta_sec = remaining / speed if speed > 0 else float("inf")

    print(
        f"[progress] {done_count}/{target_count} | "
        f"run_selected={processed_this_run} | "
        f"elapsed={format_duration(elapsed_sec)} | "
        f"ETA={format_duration(eta_sec)} | "
        f"speed={speed:.2f} records/s | "
        f"last={last_score_id}"
    )


def select_batch(
    *,
    source_pool: Path,
    batch_root: Path,
    batch_id: str,
    batch_size: int,
    overwrite: bool,
    allow_duplicate_score_ids: bool,
    progress_every: int,
) -> None:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    source_records = load_source_pool(source_pool)

    batch_dir = batch_root / batch_id
    batch_manifest = batch_dir / "batch_manifest.jsonl"

    valid_source_paths = {
        str(record["source_path"])
        for record in source_records
        if isinstance(record.get("source_path"), str)
    }

    used_source_paths, used_score_ids = load_used_sources(
        batch_root=batch_root,
        current_batch_manifest=batch_manifest,
    )

    kept_records, already_selected_current, redo_source_path = prepare_resume_state(
        batch_manifest=batch_manifest,
        overwrite=overwrite,
        valid_source_paths=valid_source_paths,
    )

    current_score_ids = {
        str(record["score_id"])
        for record in kept_records
        if isinstance(record.get("score_id"), str)
    }

    current_count = len(kept_records)
    needed = max(0, batch_size - current_count)

    print(f"[setup] source_pool:               {source_pool}")
    print(f"[setup] batch_root:                {batch_root}")
    print(f"[setup] batch_id:                  {batch_id}")
    print(f"[setup] batch_manifest:            {batch_manifest}")
    print(f"[setup] requested batch size:      {batch_size}")
    print(f"[setup] existing kept in batch:    {current_count}")
    print(f"[setup] still needed:              {needed}")
    print(f"[setup] redo first:                {redo_source_path if redo_source_path else 'none'}")
    print(f"[setup] previous used sources:     {len(used_source_paths)}")
    print(f"[setup] previous used score_ids:   {len(used_score_ids)}")
    print(f"[setup] allow duplicate score_ids: {allow_duplicate_score_ids}")

    if needed == 0:
        print("[done] batch already has requested number of records")
        print(f"[done] output: {batch_manifest}")
        return

    batch_dir.mkdir(parents=True, exist_ok=True)
    (batch_dir / "rendered_images").mkdir(parents=True, exist_ok=True)
    (batch_dir / "homr_outputs").mkdir(parents=True, exist_ok=True)
    (batch_dir / "canonical_targets").mkdir(parents=True, exist_ok=True)
    (batch_dir / "logs").mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()
    selected_this_run = 0
    skipped_used_source = 0
    skipped_duplicate_score_id = 0

    next_sequence_index = current_count

    for source_record in source_records:
        source_path = str(source_record["source_path"])
        score_id = str(source_record["score_id"])

        if source_path in used_source_paths or source_path in already_selected_current:
            skipped_used_source += 1
            continue

        if not allow_duplicate_score_ids:
            if score_id in used_score_ids or score_id in current_score_ids:
                skipped_duplicate_score_id += 1
                continue

        next_sequence_index += 1

        batch_record = make_batch_record(
            source_record=source_record,
            batch_id=batch_id,
            batch_dir=batch_dir,
            sequence_index=next_sequence_index,
        )

        append_jsonl(batch_manifest, batch_record)

        already_selected_current.add(source_path)
        current_score_ids.add(score_id)

        selected_this_run += 1
        done_count = current_count + selected_this_run

        if progress_every > 0 and (
            selected_this_run == 1
            or selected_this_run % progress_every == 0
            or done_count == batch_size
        ):
            print_progress(
                done_count=done_count,
                target_count=batch_size,
                processed_this_run=selected_this_run,
                start_time=start_time,
                last_score_id=score_id,
            )

        if done_count >= batch_size:
            break

    elapsed_sec = time.perf_counter() - start_time
    total_selected = current_count + selected_this_run

    print("[done]")
    print(f"[done] selected_this_run:           {selected_this_run}")
    print(f"[done] total_selected_in_manifest:  {total_selected}")
    print(f"[done] requested batch size:        {batch_size}")
    print(f"[done] skipped_used_source:         {skipped_used_source}")
    print(f"[done] skipped_duplicate_score_id:  {skipped_duplicate_score_id}")
    print(f"[done] elapsed:                     {format_duration(elapsed_sec)}")
    if elapsed_sec > 0:
        print(f"[done] speed:                       {selected_this_run / elapsed_sec:.2f} records/s")
    print(f"[done] output:                      {batch_manifest}")

    if total_selected < batch_size:
        print(
            "[warn] Could not fill requested batch size. "
            "The source pool may be exhausted under the current duplicate policy.",
            file=sys.stderr,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select a human-controlled batch for full-HOMR distillation."
    )

    parser.add_argument(
        "--source-pool",
        type=Path,
        default=Path("distillation/data/source_pool.jsonl"),
        help="Source pool JSONL produced by build_source_pool.py.",
    )
    parser.add_argument(
        "--batch-root",
        type=Path,
        default=Path("distillation/batches"),
        help="Directory where batch folders are stored.",
    )
    parser.add_argument(
        "--batch-id",
        type=str,
        required=True,
        help="Batch identifier, for example batch_000000_smoke or batch_000001.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        required=True,
        help="Number of source scores to select into this batch.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the current batch manifest and reselect from scratch.",
    )
    parser.add_argument(
        "--allow-duplicate-score-ids",
        action="store_true",
        help=(
            "Allow multiple files with the same score_id. "
            "Default is to select at most one source per score_id across all batches."
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Print progress every N newly selected records.",
    )

    args = parser.parse_args()

    select_batch(
        source_pool=args.source_pool,
        batch_root=args.batch_root,
        batch_id=args.batch_id,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
        allow_duplicate_score_ids=args.allow_duplicate_score_ids,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()