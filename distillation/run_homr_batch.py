from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_TIMEOUT_SEC = 300


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


def load_render_records(render_log: Path) -> list[dict[str, Any]]:
    records = read_jsonl(render_log)

    valid: list[dict[str, Any]] = []

    for i, record in enumerate(records, start=1):
        if record.get("render_status") != "ok":
            continue

        score_id = record.get("score_id")
        batch_id = record.get("batch_id")
        rendered_pages = record.get("rendered_pages")

        if not isinstance(score_id, str) or not score_id:
            print(f"[warn] Skipping render record {i}: missing score_id", file=sys.stderr)
            continue

        if not isinstance(batch_id, str) or not batch_id:
            print(f"[warn] Skipping render record {i}: missing batch_id", file=sys.stderr)
            continue

        if not isinstance(rendered_pages, list) or not rendered_pages:
            print(f"[warn] Skipping render record {i}: no rendered pages", file=sys.stderr)
            continue

        valid.append(record)

    if not valid:
        raise ValueError(f"No successful render records with pages found in {render_log}")

    return valid


def flatten_rendered_pages(render_records: list[dict[str, Any]], batch_dir: Path) -> list[dict[str, Any]]:
    page_jobs: list[dict[str, Any]] = []

    for record in render_records:
        batch_id = str(record["batch_id"])
        score_id = str(record["score_id"])
        source_path = str(record.get("source_path", ""))

        for page_item in record.get("rendered_pages", []):
            if not isinstance(page_item, dict):
                continue

            image_path = page_item.get("image_path")
            page_number = page_item.get("page_number")

            if not isinstance(image_path, str) or not image_path:
                continue

            try:
                page_number_int = int(page_number)
            except Exception:
                page_number_int = len(page_jobs) + 1

            page_id = f"{score_id}_page_{page_number_int:04d}"
            homr_output_path = batch_dir / "homr_outputs" / score_id / f"page_{page_number_int:04d}.musicxml"

            page_jobs.append(
                {
                    "batch_id": batch_id,
                    "score_id": score_id,
                    "page_id": page_id,
                    "page_number": page_number_int,
                    "source_path": source_path,
                    "rendered_image_path": image_path,
                    "homr_output_path": normalize_path(homr_output_path),
                }
            )

    if not page_jobs:
        raise ValueError("No rendered page jobs could be derived from render log.")

    return page_jobs


def prepare_resume_state(
    *,
    homr_log: Path,
    overwrite: bool,
    valid_page_ids: set[str],
) -> tuple[list[dict[str, Any]], set[str], str | None]:
    """
    Return:
        kept_records, completed_page_ids, redo_page_id

    The HOMR log is the only pipeline progress state. On resume, all valid
    logged records are kept except the last valid record, which is removed and
    recomputed.
    """
    if homr_log.exists() and overwrite:
        homr_log.unlink()
        return [], set(), None

    existing_records = read_jsonl(homr_log)

    if not existing_records:
        return [], set(), None

    valid_records = [
        record
        for record in existing_records
        if isinstance(record.get("page_id"), str)
        and record["page_id"] in valid_page_ids
    ]

    if not valid_records:
        write_jsonl(homr_log, [])
        return [], set(), None

    redo_page_id = str(valid_records[-1]["page_id"])
    kept_records = valid_records[:-1]

    write_jsonl(homr_log, kept_records)

    completed_page_ids = {
        str(record["page_id"])
        for record in kept_records
        if isinstance(record.get("page_id"), str)
    }

    return kept_records, completed_page_ids, redo_page_id


def fill_command_template(
    *,
    command_template: str,
    image_path: Path,
    output_path: Path,
) -> list[str]:
    """
    The template must contain:
        {image_path}
        {output_path}

    Example:
        python -m homr.main "{image_path}" --output "{output_path}"

    This avoids hard-coding HOMR CLI assumptions in the pipeline.
    """
    if "{image_path}" not in command_template or "{output_path}" not in command_template:
        raise ValueError(
            "HOMR command template must contain both {image_path} and {output_path}."
        )

    command_text = command_template.format(
        image_path=str(image_path),
        output_path=str(output_path),
    )

    return shlex.split(command_text, posix=False)


def run_homr_command(
    *,
    command_template: str,
    image_path: Path,
    output_path: Path,
    timeout_sec: int | None,
) -> tuple[bool, str]:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        output_path.unlink()

    command = fill_command_template(
        command_template=command_template,
        image_path=image_path,
        output_path=output_path,
    )

    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_sec,
        )

        stdout_text = completed.stdout.decode(errors="replace").strip()
        stderr_text = completed.stderr.decode(errors="replace").strip()
        message = "\n".join(x for x in [stdout_text, stderr_text] if x).strip()

        if completed.returncode != 0:
            return False, message or f"HOMR command failed with code {completed.returncode}"

        if not output_path.exists():
            return False, "HOMR returned success but output file was not created."

        if output_path.stat().st_size <= 0:
            return False, "HOMR output file is empty."

        return True, message

    except subprocess.TimeoutExpired:
        return False, f"HOMR command timed out after {timeout_sec} seconds."

    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def build_homr_record(
    *,
    page_job: dict[str, Any],
    image_path: Path,
    output_path: Path,
    elapsed_sec: float,
    status: str,
    message: str | None,
    include_hash: bool,
    command_template: str,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "batch_id": page_job["batch_id"],
        "score_id": page_job["score_id"],
        "page_id": page_job["page_id"],
        "page_number": page_job["page_number"],
        "source_path": page_job.get("source_path"),
        "rendered_image_path": normalize_path(image_path),
        "homr_output_path": normalize_path(output_path),
        "homr_status": status,
        "elapsed_sec": float(elapsed_sec),
        "message": message,
        "homr_command_template": command_template,
        "created_at": now_iso(),
    }

    if image_path.exists():
        record["image_size_bytes"] = int(image_path.stat().st_size)
        if include_hash:
            record["image_sha256"] = sha256_file(image_path)

    if status == "ok" and output_path.exists():
        record["homr_output_size_bytes"] = int(output_path.stat().st_size)
        if include_hash:
            record["homr_output_sha256"] = sha256_file(output_path)

    return record


def count_statuses(records: list[dict[str, Any]]) -> tuple[int, int]:
    ok = 0
    errors = 0

    for record in records:
        if record.get("homr_status") == "ok":
            ok += 1
        else:
            errors += 1

    return ok, errors


def print_progress(
    *,
    done_count: int,
    total_count: int,
    processed_this_run: int,
    ok_total: int,
    error_total: int,
    start_time: float,
    last_page_id: str,
    last_status: str,
) -> None:
    elapsed_sec = time.perf_counter() - start_time
    speed = processed_this_run / elapsed_sec if elapsed_sec > 0 else 0.0
    remaining = max(0, total_count - done_count)
    eta_sec = remaining / speed if speed > 0 else float("inf")

    print(
        f"[progress] {done_count}/{total_count} | "
        f"run_processed={processed_this_run} | "
        f"ok_total={ok_total} | errors_total={error_total} | "
        f"elapsed={format_duration(elapsed_sec)} | "
        f"ETA={format_duration(eta_sec)} | "
        f"speed={speed:.2f} pages/s | "
        f"last={last_page_id} | status={last_status}"
    )


def run_homr_batch(
    *,
    render_log: Path,
    homr_log: Path | None,
    overwrite: bool,
    command_template: str,
    timeout_sec: int | None,
    include_hash: bool,
    progress_every: int,
) -> None:
    batch_dir = render_log.parents[1]

    if homr_log is None:
        homr_log = batch_dir / "logs" / "homr_log.jsonl"

    render_records = load_render_records(render_log)
    page_jobs = flatten_rendered_pages(render_records, batch_dir=batch_dir)

    valid_page_ids = {
        str(job["page_id"])
        for job in page_jobs
    }

    kept_records, completed_page_ids, redo_page_id = prepare_resume_state(
        homr_log=homr_log,
        overwrite=overwrite,
        valid_page_ids=valid_page_ids,
    )

    ok_total, error_total = count_statuses(kept_records)

    pending_jobs = [
        job for job in page_jobs
        if str(job["page_id"]) not in completed_page_ids
    ]

    print(f"[setup] render_log:       {render_log}")
    print(f"[setup] homr_log:         {homr_log}")
    print(f"[setup] batch_dir:        {batch_dir}")
    print(f"[setup] total pages:      {len(page_jobs)}")
    print(f"[setup] already logged:   {len(completed_page_ids)}")
    print(f"[setup] to process:       {len(pending_jobs)}")
    print(f"[setup] redo first:       {redo_page_id if redo_page_id else 'none'}")
    print(f"[setup] include_hash:     {include_hash}")
    print(f"[setup] command template: {command_template}")

    if not pending_jobs:
        print("[done] nothing to process")
        print(f"[done] ok_total:     {ok_total}")
        print(f"[done] errors_total: {error_total}")
        print(f"[done] output:       {homr_log}")
        return

    start_time = time.perf_counter()
    processed_this_run = 0

    for page_job in pending_jobs:
        page_id = str(page_job["page_id"])
        image_path = Path(str(page_job["rendered_image_path"]))
        output_path = Path(str(page_job["homr_output_path"]))

        item_start = time.perf_counter()

        if not image_path.exists():
            elapsed_sec = time.perf_counter() - item_start

            record = build_homr_record(
                page_job=page_job,
                image_path=image_path,
                output_path=output_path,
                elapsed_sec=elapsed_sec,
                status="error",
                message=f"Rendered image does not exist: {image_path}",
                include_hash=include_hash,
                command_template=command_template,
            )
            error_total += 1

        else:
            success, message = run_homr_command(
                command_template=command_template,
                image_path=image_path,
                output_path=output_path,
                timeout_sec=timeout_sec,
            )

            elapsed_sec = time.perf_counter() - item_start

            record = build_homr_record(
                page_job=page_job,
                image_path=image_path,
                output_path=output_path,
                elapsed_sec=elapsed_sec,
                status="ok" if success else "error",
                message=message if message else None,
                include_hash=include_hash,
                command_template=command_template,
            )

            if success:
                ok_total += 1
            else:
                error_total += 1

        append_jsonl(homr_log, record)

        processed_this_run += 1
        done_count = len(completed_page_ids) + processed_this_run

        if progress_every > 0 and (
            processed_this_run == 1
            or processed_this_run % progress_every == 0
            or done_count == len(page_jobs)
        ):
            print_progress(
                done_count=done_count,
                total_count=len(page_jobs),
                processed_this_run=processed_this_run,
                ok_total=ok_total,
                error_total=error_total,
                start_time=start_time,
                last_page_id=page_id,
                last_status=record["homr_status"],
            )

    elapsed_sec = time.perf_counter() - start_time

    print("[done]")
    print(f"[done] processed_this_run: {processed_this_run}")
    print(f"[done] total_logged:        {len(completed_page_ids) + processed_this_run}")
    print(f"[done] ok_total:            {ok_total}")
    print(f"[done] errors_total:        {error_total}")
    print(f"[done] elapsed:             {format_duration(elapsed_sec)}")
    if elapsed_sec > 0:
        print(f"[done] speed:               {processed_this_run / elapsed_sec:.2f} pages/s")
    print(f"[done] output:              {homr_log}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full HOMR on rendered page images for one distillation batch."
    )

    parser.add_argument(
        "--render-log",
        type=Path,
        required=True,
        help="Render log JSONL produced by render_batch.py.",
    )
    parser.add_argument(
        "--homr-log",
        type=Path,
        default=None,
        help="Optional HOMR log path. Defaults to <batch>/logs/homr_log.jsonl.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the HOMR log and rerun from scratch.",
    )
    parser.add_argument(
        "--homr-command-template",
        type=str,
        required=True,
        help=(
            "Command template for running HOMR on one image. "
            "Must contain {image_path} and {output_path}. "
            "Example: python -m homr.main \"{image_path}\" --output \"{output_path}\""
        ),
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=DEFAULT_TIMEOUT_SEC,
        help=f"Per-page HOMR timeout in seconds. Default: {DEFAULT_TIMEOUT_SEC}. Use <=0 for no timeout.",
    )
    parser.add_argument(
        "--include-hash",
        action="store_true",
        help="Compute SHA256 for rendered images and HOMR outputs.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Print progress every N newly processed pages.",
    )

    args = parser.parse_args()

    timeout_sec = args.timeout_sec
    if timeout_sec is not None and timeout_sec <= 0:
        timeout_sec = None

    run_homr_batch(
        render_log=args.render_log,
        homr_log=args.homr_log,
        overwrite=args.overwrite,
        command_template=args.homr_command_template,
        timeout_sec=timeout_sec,
        include_hash=args.include_hash,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()