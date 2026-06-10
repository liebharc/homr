from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from progress import emit_pipeline_event


DEFAULT_RENDER_DPI = 300
DEFAULT_TRIM_MARGIN = 10
STAGE_NAME = "render_batch"


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


def normalize_path(path: str | Path) -> str:
    return str(path).replace("\\", "/")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


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


def resolve_musescore_path(explicit_path: Path | None = None) -> Path:
    if explicit_path is not None:
        candidate = explicit_path
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"MuseScore executable not found: {candidate}")

    env_path = os.environ.get("MUSESCORE_PATH")
    if env_path:
        candidate = Path(env_path)
        if candidate.exists():
            return candidate

    candidates = [
        Path(r"D:\Program Files\MuseScore 4\bin\MuseScore4.exe"),
        Path(r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe"),
        Path(r"C:\Program Files\MuseScore 3\bin\mscore3.exe"),
        Path(r"C:\Program Files (x86)\MuseScore 3\bin\mscore3.exe"),
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    for executable_name in ("MuseScore4.exe", "mscore3.exe", "MuseScore4", "mscore3"):
        found = shutil.which(executable_name)
        if found:
            return Path(found)

    raise FileNotFoundError(
        "MuseScore executable not found. Set MUSESCORE_PATH to the full path "
        "of MuseScore4.exe or pass --musescore-path."
    )


def validate_batch_records(records: list[dict[str, Any]], batch_manifest: Path) -> list[dict[str, Any]]:
    valid: list[dict[str, Any]] = []

    for i, record in enumerate(records, start=1):
        required = ["batch_id", "score_id", "source_path", "rendered_image_path"]

        missing = [
            key for key in required
            if not isinstance(record.get(key), str) or not record.get(key)
        ]

        if missing:
            print(
                f"[warn] Skipping invalid manifest record {i}; missing {missing}: {batch_manifest}",
                file=sys.stderr,
            )
            continue

        valid.append(record)

    if not valid:
        raise ValueError(f"No valid renderable records found in {batch_manifest}")

    return valid


def prepare_resume_state(
    *,
    render_log: Path,
    overwrite: bool,
    valid_score_ids: set[str],
) -> tuple[list[dict[str, Any]], set[str], str | None]:
    """
    Return:
        kept_records, completed_score_ids, redo_score_id

    The render log is the only progress state. On resume, all valid logged
    records are kept except the last valid record, which is removed and redone.
    """
    if render_log.exists() and overwrite:
        render_log.unlink()
        return [], set(), None

    existing_records = read_jsonl(render_log)

    if not existing_records:
        return [], set(), None

    valid_records = [
        record
        for record in existing_records
        if isinstance(record.get("score_id"), str)
        and record["score_id"] in valid_score_ids
    ]

    if not valid_records:
        write_jsonl(render_log, [])
        return [], set(), None

    redo_score_id = str(valid_records[-1]["score_id"])
    kept_records = valid_records[:-1]

    write_jsonl(render_log, kept_records)

    completed_score_ids = {
        str(record["score_id"])
        for record in kept_records
        if isinstance(record.get("score_id"), str)
    }

    return kept_records, completed_score_ids, redo_score_id


def page_index_from_path(path: Path, score_id: str) -> int:
    """
    MuseScore usually writes:
        score.png      for single direct output in some modes
        score-1.png    for first page
        score-2.png    for second page
    """
    stem = path.stem

    if stem == score_id:
        return 1

    match = re.fullmatch(re.escape(score_id) + r"-(\d+)", stem)
    if match:
        return int(match.group(1))

    return 10**9


def find_rendered_pages(output_dir: Path, score_id: str) -> list[Path]:
    candidates = list(output_dir.glob(f"{score_id}.png"))
    candidates.extend(output_dir.glob(f"{score_id}-*.png"))

    pages = [
        path for path in candidates
        if path.exists() and path.is_file() and path.stat().st_size > 0
    ]

    pages = sorted(
        pages,
        key=lambda p: (page_index_from_path(p, score_id), p.name),
    )

    return pages


def cleanup_previous_outputs(output_dir: Path, score_id: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in output_dir.glob(f"{score_id}.png"):
        path.unlink(missing_ok=True)

    for path in output_dir.glob(f"{score_id}-*.png"):
        path.unlink(missing_ok=True)


def run_musescore_render(
    *,
    musescore_path: Path,
    source_path: Path,
    output_base_path: Path,
    score_id: str,
    dpi: int,
    trim_margin: int,
    timeout_sec: int | None,
) -> tuple[bool, str, list[Path]]:
    output_dir = output_base_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    cleanup_previous_outputs(output_dir, score_id)

    cmd = [
        str(musescore_path),
        "-T",
        str(trim_margin),
        "-r",
        str(dpi),
        "-o",
        str(output_base_path),
        str(source_path),
    ]

    try:
        completed = subprocess.run(
            cmd,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=timeout_sec,
        )

        stderr_text = completed.stderr.decode(errors="replace").strip()

        pages = find_rendered_pages(output_dir, score_id)

        if completed.returncode != 0:
            return False, stderr_text, pages

        if not pages:
            return False, "MuseScore returned success but no page PNGs were found.", pages

        return True, "", pages

    except subprocess.TimeoutExpired:
        pages = find_rendered_pages(output_dir, score_id)
        return False, f"MuseScore render timed out after {timeout_sec} seconds.", pages

    except Exception as exc:
        pages = find_rendered_pages(output_dir, score_id)
        return False, f"{type(exc).__name__}: {exc}", pages


def build_render_record(
    *,
    batch_record: dict[str, Any],
    source_path: Path,
    output_base_path: Path,
    rendered_pages: list[Path],
    dpi: int,
    trim_margin: int,
    elapsed_sec: float,
    status: str,
    message: str | None,
    include_hash: bool,
) -> dict[str, Any]:
    pages_payload: list[dict[str, Any]] = []

    for page_number, page_path in enumerate(rendered_pages, start=1):
        page_record: dict[str, Any] = {
            "page_number": page_number,
            "image_path": normalize_path(page_path),
            "image_size_bytes": int(page_path.stat().st_size),
        }

        if include_hash:
            page_record["image_sha256"] = sha256_file(page_path)

        pages_payload.append(page_record)

    return {
        "batch_id": batch_record["batch_id"],
        "score_id": batch_record["score_id"],
        "batch_sequence_index": batch_record.get("batch_sequence_index"),
        "source_path": normalize_path(source_path),
        "rendered_image_base_path": normalize_path(output_base_path),
        "rendered_pages": pages_payload,
        "num_rendered_pages": len(pages_payload),
        "render_dpi": int(dpi),
        "trim_margin": int(trim_margin),
        "render_status": status,
        "elapsed_sec": float(elapsed_sec),
        "message": message,
        "created_at": now_iso(),
    }


def print_progress(
    *,
    done_count: int,
    total_count: int,
    processed_this_run: int,
    ok_total: int,
    error_total: int,
    pages_total: int,
    start_time: float,
    last_score_id: str,
    last_status: str,
) -> None:
    elapsed_sec = time.perf_counter() - start_time
    emit_pipeline_event(
        stage=STAGE_NAME,
        event="progress",
        status=last_status,
        unit="scores",
        done=done_count,
        total=total_count,
        elapsed_seconds=elapsed_sec,
        counts={
            "processed_this_run": processed_this_run,
            "ok": ok_total,
            "errors": error_total,
            "pages": pages_total,
        },
        ids={"last_score_id": last_score_id},
    )


def count_statuses(records: list[dict[str, Any]]) -> tuple[int, int, int]:
    ok = 0
    errors = 0
    pages = 0

    for record in records:
        if record.get("render_status") == "ok":
            ok += 1
        else:
            errors += 1

        try:
            pages += int(record.get("num_rendered_pages") or 0)
        except Exception:
            pass

    return ok, errors, pages


def render_batch(
    *,
    batch_manifest: Path,
    render_log: Path | None,
    overwrite: bool,
    musescore_path_arg: Path | None,
    dpi: int,
    trim_margin: int,
    timeout_sec: int | None,
    include_hash: bool,
    progress_every: int,
) -> None:
    batch_records = validate_batch_records(
        read_jsonl(batch_manifest),
        batch_manifest=batch_manifest,
    )

    batch_dir = batch_manifest.parent
    if render_log is None:
        render_log = batch_dir / "logs" / "render_log.jsonl"

    valid_score_ids = {
        str(record["score_id"])
        for record in batch_records
    }

    kept_records, completed_score_ids, redo_score_id = prepare_resume_state(
        render_log=render_log,
        overwrite=overwrite,
        valid_score_ids=valid_score_ids,
    )

    ok_total, error_total, pages_total = count_statuses(kept_records)

    pending_records = [
        record for record in batch_records
        if str(record["score_id"]) not in completed_score_ids
    ]

    musescore_path = resolve_musescore_path(musescore_path_arg)

    print(f"[setup] batch_manifest: {batch_manifest}")
    print(f"[setup] render_log:     {render_log}")
    print(f"[setup] batch_dir:      {batch_dir}")
    print(f"[setup] total records:  {len(batch_records)}")
    print(f"[setup] already logged: {len(completed_score_ids)}")
    print(f"[setup] to process:     {len(pending_records)}")
    print(f"[setup] redo first:     {redo_score_id if redo_score_id else 'none'}")
    print(f"[setup] MuseScore:      {musescore_path}")
    print(f"[setup] dpi:            {dpi}")
    print(f"[setup] trim_margin:    {trim_margin}")
    print(f"[setup] include_hash:   {include_hash}")

    if not pending_records:
        print("[done] nothing to process")
        print(f"[done] ok_total:     {ok_total}")
        print(f"[done] errors_total: {error_total}")
        print(f"[done] pages_total:  {pages_total}")
        print(f"[done] output:       {render_log}")
        return

    start_time = time.perf_counter()
    processed_this_run = 0

    for batch_record in pending_records:
        score_id = str(batch_record["score_id"])
        source_path = Path(str(batch_record["source_path"]))
        output_base_path = Path(str(batch_record["rendered_image_path"]))

        item_start = time.perf_counter()

        if not source_path.exists():
            elapsed_sec = time.perf_counter() - item_start
            rendered_pages: list[Path] = []
            render_record = build_render_record(
                batch_record=batch_record,
                source_path=source_path,
                output_base_path=output_base_path,
                rendered_pages=rendered_pages,
                dpi=dpi,
                trim_margin=trim_margin,
                elapsed_sec=elapsed_sec,
                status="error",
                message=f"Source file does not exist: {source_path}",
                include_hash=include_hash,
            )
            error_total += 1

        else:
            success, message, rendered_pages = run_musescore_render(
                musescore_path=musescore_path,
                source_path=source_path,
                output_base_path=output_base_path,
                score_id=score_id,
                dpi=dpi,
                trim_margin=trim_margin,
                timeout_sec=timeout_sec,
            )

            elapsed_sec = time.perf_counter() - item_start

            render_record = build_render_record(
                batch_record=batch_record,
                source_path=source_path,
                output_base_path=output_base_path,
                rendered_pages=rendered_pages,
                dpi=dpi,
                trim_margin=trim_margin,
                elapsed_sec=elapsed_sec,
                status="ok" if success else "error",
                message=None if success else message,
                include_hash=include_hash,
            )

            if success:
                ok_total += 1
            else:
                error_total += 1

            pages_total += len(rendered_pages)

        append_jsonl(render_log, render_record)

        processed_this_run += 1
        done_count = len(completed_score_ids) + processed_this_run

        if progress_every > 0 and (
            processed_this_run == 1
            or processed_this_run % progress_every == 0
            or done_count == len(batch_records)
        ):
            print_progress(
                done_count=done_count,
                total_count=len(batch_records),
                processed_this_run=processed_this_run,
                ok_total=ok_total,
                error_total=error_total,
                pages_total=pages_total,
                start_time=start_time,
                last_score_id=score_id,
                last_status=render_record["render_status"],
            )

    elapsed_sec = time.perf_counter() - start_time

    print("[done]")
    print(f"[done] processed_this_run: {processed_this_run}")
    print(f"[done] total_logged:        {len(completed_score_ids) + processed_this_run}")
    print(f"[done] ok_total:            {ok_total}")
    print(f"[done] errors_total:        {error_total}")
    print(f"[done] pages_total:         {pages_total}")
    print(f"[done] elapsed:             {format_duration(elapsed_sec)}")
    if elapsed_sec > 0:
        print(f"[done] speed:               {processed_this_run / elapsed_sec:.2f} scores/s")
    print(f"[done] output:              {render_log}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a selected full-HOMR distillation batch into page images."
    )

    parser.add_argument(
        "--batch-manifest",
        type=Path,
        required=True,
        help="Batch manifest JSONL produced by select_batch.py.",
    )
    parser.add_argument(
        "--render-log",
        type=Path,
        default=None,
        help="Optional render log path. Defaults to <batch>/logs/render_log.jsonl.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the render log and render the batch from scratch.",
    )
    parser.add_argument(
        "--musescore-path",
        type=Path,
        default=None,
        help="Optional explicit path to MuseScore executable.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_RENDER_DPI,
        help=f"Render DPI. Default: {DEFAULT_RENDER_DPI}.",
    )
    parser.add_argument(
        "--trim-margin",
        type=int,
        default=DEFAULT_TRIM_MARGIN,
        help=f"MuseScore trim margin passed with -T. Default: {DEFAULT_TRIM_MARGIN}.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=180,
        help="Per-score render timeout in seconds. Use 0 or negative for no timeout.",
    )
    parser.add_argument(
        "--include-hash",
        action="store_true",
        help="Compute SHA256 for rendered images. Slower but more reproducible.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=5,
        help="Print progress every N newly rendered score records.",
    )

    args = parser.parse_args()

    timeout_sec = args.timeout_sec
    if timeout_sec is not None and timeout_sec <= 0:
        timeout_sec = None

    render_batch(
        batch_manifest=args.batch_manifest,
        render_log=args.render_log,
        overwrite=args.overwrite,
        musescore_path_arg=args.musescore_path,
        dpi=args.dpi,
        trim_margin=args.trim_margin,
        timeout_sec=timeout_sec,
        include_hash=args.include_hash,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()