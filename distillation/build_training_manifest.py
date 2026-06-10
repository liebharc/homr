#!/usr/bin/env python3
"""
Build a JSONL training manifest from ONNX-HOMR teacher outputs.

Input:
    distillation/batches/<batch>/teacher_outputs/*.json

Output:
    distillation/batches/<batch>/training_manifest.jsonl

Each row pairs:
    rendered page image -> canonical HOMR teacher target
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def emit_event(payload: dict[str, Any], *, quiet: bool = False, stderr: bool = False) -> None:
    if quiet:
        return
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), file=sys.stderr if stderr else sys.stdout, flush=True)


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(path.suffix + ".tmp")

    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            f.write("\n")

        f.flush()
        os.fsync(f.fileno())

    tmp.replace(path)


def collect_teacher_files(teacher_dir: Path) -> list[Path]:
    if not teacher_dir.exists():
        raise FileNotFoundError(f"Teacher output directory not found: {teacher_dir}")

    files = sorted(teacher_dir.glob("*.json"))

    if not files:
        raise FileNotFoundError(f"No teacher JSON files found in: {teacher_dir}")

    return files


def validate_teacher_payload(path: Path, payload: dict[str, Any]) -> tuple[bool, str]:
    if payload.get("status") != "ok":
        return False, "status_not_ok"

    image_path = payload.get("image_path")
    if not image_path:
        return False, "missing_image_path"

    if not Path(image_path).exists():
        return False, "image_path_missing_on_disk"

    canonical = payload.get("canonical")
    if not isinstance(canonical, dict):
        return False, "missing_canonical"

    staff_sequences = canonical.get("staff_token_sequences")
    flat_tokens = canonical.get("flat_tokens_with_staff_breaks")

    if not isinstance(staff_sequences, list):
        return False, "missing_staff_token_sequences"

    if not isinstance(flat_tokens, list):
        return False, "missing_flat_tokens"

    if len(staff_sequences) == 0:
        return False, "zero_staff_sequences"

    if len(flat_tokens) == 0:
        return False, "zero_flat_tokens"

    return True, "ok"


def build_manifest_row(
    *,
    teacher_path: Path,
    payload: dict[str, Any],
) -> dict[str, Any]:
    canonical = payload["canonical"]

    staff_sequences = canonical["staff_token_sequences"]
    flat_tokens = canonical["flat_tokens_with_staff_breaks"]

    token_count = sum(len(seq) for seq in staff_sequences if isinstance(seq, list))

    return {
        "schema": "homr_student_training_manifest_v1",

        "score_id": payload.get("score_id"),
        "page_id": payload.get("page_id"),
        "page_number": payload.get("page_number"),

        "image_path": payload.get("image_path"),
        "teacher_path": str(teacher_path),

        "target_schema": canonical.get("target_schema", "staff_token_sequences_v1"),
        "target_staff_token_sequences": staff_sequences,
        "target_flat_tokens_with_staff_breaks": flat_tokens,

        "n_staffs": int(payload.get("n_staffs", len(staff_sequences))),
        "n_target_tokens": int(token_count),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--teacher-dir",
        type=Path,
        required=True,
        help="Directory containing teacher_outputs/*.json.",
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output manifest JSONL. Default: <batch>/training_manifest.jsonl",
    )

    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Include rows with empty canonical token targets.",
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on the first invalid teacher output instead of skipping.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print JSON progress every N teacher files. Use 0 to disable progress.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress terminal progress prints except errors and final completion.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    teacher_dir = args.teacher_dir
    batch_dir = teacher_dir.parent
    out_path = args.out or batch_dir / "training_manifest.jsonl"

    teacher_files = collect_teacher_files(teacher_dir)
    started = time.time()
    emit_event(
        {
            "event": "start",
            "teacher_dir": str(teacher_dir),
            "manifest_path": str(out_path),
            "teacher_files": len(teacher_files),
        },
        quiet=args.quiet,
    )

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    for index, teacher_path in enumerate(teacher_files, start=1):
        payload = read_json(teacher_path)

        ok, reason = validate_teacher_payload(teacher_path, payload)

        if not ok:
            if reason in {"zero_staff_sequences", "zero_flat_tokens"} and args.allow_empty:
                pass
            else:
                item = {
                    "teacher_path": str(teacher_path),
                    "reason": reason,
                }

                skipped.append(item)
                if args.progress_every > 0 and (index == 1 or index % args.progress_every == 0 or index == len(teacher_files)):
                    emit_event(
                        {
                            "event": "progress",
                            "processed": index,
                            "total": len(teacher_files),
                            "rows": len(rows),
                            "skipped": len(skipped),
                            "last_teacher_path": str(teacher_path),
                            "last_status": "skipped",
                            "reason": reason,
                            "seconds": time.time() - started,
                        },
                        quiet=args.quiet,
                    )

                if args.strict:
                    raise RuntimeError(
                        f"Invalid teacher output: {teacher_path} reason={reason}"
                    )

                continue

        rows.append(
            build_manifest_row(
                teacher_path=teacher_path,
                payload=payload,
            )
        )
        if args.progress_every > 0 and (index == 1 or index % args.progress_every == 0 or index == len(teacher_files)):
            emit_event(
                {
                    "event": "progress",
                    "processed": index,
                    "total": len(teacher_files),
                    "rows": len(rows),
                    "skipped": len(skipped),
                    "last_teacher_path": str(teacher_path),
                    "last_status": "ok",
                    "seconds": time.time() - started,
                },
                quiet=args.quiet,
            )

    if not rows:
        raise RuntimeError(
            f"No valid manifest rows produced from {teacher_dir}. "
            f"Skipped {len(skipped)} teacher files."
        )

    write_jsonl_atomic(out_path, rows)

    summary = {
        "event": "done",
        "teacher_dir": str(teacher_dir),
        "manifest_path": str(out_path),
        "teacher_files": len(teacher_files),
        "rows": len(rows),
        "skipped": len(skipped),
        "total_target_tokens": sum(row["n_target_tokens"] for row in rows),
        "seconds": time.time() - started,
    }

    print(json.dumps(summary, sort_keys=True), flush=True)

    if skipped:
        skipped_path = out_path.with_suffix(".skipped.json")
        with skipped_path.open("w", encoding="utf-8") as f:
            json.dump(skipped, f, indent=2, sort_keys=True)

        print(
            json.dumps(
                {
                    "event": "skipped_written",
                    "path": str(skipped_path),
                    "count": len(skipped),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())