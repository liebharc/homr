from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


SUPPORTED_EXTENSIONS = {".mxl", ".xml"}


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def score_id_from_path(path: Path) -> str:
    return path.stem


def discover_score_files(mxl_root: Path) -> list[Path]:
    if not mxl_root.exists():
        raise FileNotFoundError(f"MXL root does not exist: {mxl_root}")

    if not mxl_root.is_dir():
        raise NotADirectoryError(f"MXL root is not a directory: {mxl_root}")

    paths: list[Path] = []

    for ext in sorted(SUPPORTED_EXTENSIONS):
        paths.extend(mxl_root.rglob(f"*{ext}"))

    paths = sorted(p for p in paths if p.is_file())

    return paths


def load_existing_records(output_jsonl: Path) -> list[dict[str, Any]]:
    if not output_jsonl.exists():
        return []

    records: list[dict[str, Any]] = []

    with output_jsonl.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            text = line.strip()

            if not text:
                continue

            try:
                record = json.loads(text)
            except json.JSONDecodeError:
                print(
                    f"[resume] Ignoring unreadable JSONL line {line_number}: {output_jsonl}",
                    file=sys.stderr,
                )
                continue

            if isinstance(record, dict) and isinstance(record.get("source_path"), str):
                records.append(record)

    return records


def rewrite_jsonl(output_jsonl: Path, records: list[dict[str, Any]]) -> None:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with output_jsonl.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_jsonl_record(output_jsonl: Path, record: dict[str, Any]) -> None:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with output_jsonl.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def prepare_resume_state(
    *,
    output_jsonl: Path,
    valid_source_paths: set[str],
    overwrite: bool,
) -> tuple[set[str], str | None]:
    """
    Return:
        completed_source_paths_to_skip, redo_source_path

    The output JSONL is the progress log. On resume, all previously written
    source paths are skipped except the last valid written source path, which is
    removed from the file and recomputed.
    """
    if output_jsonl.exists() and overwrite:
        output_jsonl.unlink()
        return set(), None

    existing_records = load_existing_records(output_jsonl)

    if not existing_records:
        return set(), None

    valid_records = [
        record for record in existing_records
        if record.get("source_path") in valid_source_paths
    ]

    if not valid_records:
        rewrite_jsonl(output_jsonl, [])
        return set(), None

    redo_source_path = valid_records[-1]["source_path"]
    kept_records = valid_records[:-1]

    rewrite_jsonl(output_jsonl, kept_records)

    completed_source_paths = {
        record["source_path"]
        for record in kept_records
    }

    return completed_source_paths, redo_source_path


def print_progress(
    *,
    done_count: int,
    total_count: int,
    processed_this_run: int,
    start_time: float,
    last_source_path: str,
) -> None:
    elapsed_sec = time.perf_counter() - start_time
    speed = processed_this_run / elapsed_sec if elapsed_sec > 0 else 0.0
    remaining = max(0, total_count - done_count)
    eta_sec = remaining / speed if speed > 0 else float("inf")

    print(
        f"[progress] {done_count}/{total_count} | "
        f"run_processed={processed_this_run} | "
        f"elapsed={format_duration(elapsed_sec)} | "
        f"ETA={format_duration(eta_sec)} | "
        f"speed={speed:.2f} files/s | "
        f"last={last_source_path}"
    )


def build_record(path: Path, mxl_root: Path, include_hash: bool) -> dict[str, Any]:
    relative_path = path.relative_to(mxl_root)
    source_path = str(path).replace("\\", "/")

    record: dict[str, Any] = {
        "score_id": score_id_from_path(path),
        "source_path": source_path,
        "source_relative_path": str(relative_path).replace("\\", "/"),
        "source_ext": path.suffix.lower(),
        "source_size_bytes": int(path.stat().st_size),
        "discovered_at": now_iso(),
    }

    if include_hash:
        record["source_sha256"] = sha256_file(path)

    return record


def build_source_pool(
    *,
    mxl_root: Path,
    output_jsonl: Path,
    max_files: int | None,
    overwrite: bool,
    include_hash: bool,
    progress_every: int,
) -> None:
    score_files = discover_score_files(mxl_root)

    if max_files is not None:
        score_files = score_files[: max(0, int(max_files))]

    if not score_files:
        raise FileNotFoundError(f"No .mxl or .xml files found under {mxl_root}")

    valid_source_paths = {
        str(path).replace("\\", "/")
        for path in score_files
    }

    completed_source_paths, redo_source_path = prepare_resume_state(
        output_jsonl=output_jsonl,
        valid_source_paths=valid_source_paths,
        overwrite=overwrite,
    )

    pending_files = [
        path for path in score_files
        if str(path).replace("\\", "/") not in completed_source_paths
    ]

    print(f"[setup] mxl_root:       {mxl_root}")
    print(f"[setup] output_jsonl:   {output_jsonl}")
    print(f"[setup] total files:    {len(score_files)}")
    print(f"[setup] already logged: {len(completed_source_paths)}")
    print(f"[setup] to process:     {len(pending_files)}")
    print(f"[setup] redo first:     {redo_source_path if redo_source_path else 'none'}")
    print(f"[setup] include_hash:   {include_hash}")

    if not pending_files:
        print("[done] nothing to process")
        print(f"[done] output: {output_jsonl}")
        return

    start_time = time.perf_counter()
    processed_this_run = 0

    for path in pending_files:
        record = build_record(
            path=path,
            mxl_root=mxl_root,
            include_hash=include_hash,
        )

        append_jsonl_record(output_jsonl, record)

        processed_this_run += 1
        done_count = len(completed_source_paths) + processed_this_run

        if progress_every > 0 and (
            processed_this_run == 1
            or processed_this_run % progress_every == 0
            or done_count == len(score_files)
        ):
            print_progress(
                done_count=done_count,
                total_count=len(score_files),
                processed_this_run=processed_this_run,
                start_time=start_time,
                last_source_path=record["source_path"],
            )

    elapsed_sec = time.perf_counter() - start_time

    print("[done]")
    print(f"[done] processed_this_run: {processed_this_run}")
    print(f"[done] total_logged:        {len(completed_source_paths) + processed_this_run}")
    print(f"[done] elapsed:             {format_duration(elapsed_sec)}")
    if elapsed_sec > 0:
        print(f"[done] speed:               {processed_this_run / elapsed_sec:.2f} files/s")
    print(f"[done] output:              {output_jsonl}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the full-HOMR distillation source pool from MXL/XML files."
    )

    parser.add_argument(
        "--mxl-root",
        type=Path,
        default=Path("dataset/mxl"),
        help="Root folder containing source .mxl/.xml files.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("distillation/data/source_pool.jsonl"),
        help="Output source-pool JSONL.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional maximum number of files to index.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output JSONL and rebuild from scratch.",
    )
    parser.add_argument(
        "--include-hash",
        action="store_true",
        help="Compute SHA256 for every source file. Slower but more reproducible.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Print progress every N newly processed files.",
    )

    args = parser.parse_args()

    build_source_pool(
        mxl_root=args.mxl_root,
        output_jsonl=args.output_jsonl,
        max_files=args.max_files,
        overwrite=args.overwrite,
        include_hash=args.include_hash,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()