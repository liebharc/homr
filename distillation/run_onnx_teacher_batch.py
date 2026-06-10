#!/usr/bin/env python3
"""
Generate ONNX-direct HOMR teacher outputs for full-page images from a page log.

Pipeline per rendered page:

    page PNG
      -> ONNX SegNet
      -> deterministic HOMR staff/layout prep
      -> prepared staff .npy files
      -> ONNX TrOMR encoder + decoder
      -> teacher JSON with canonical target fields

This script intentionally does NOT call homr.main, Python SegNet inference,
Python TrOMR inference, requests, Staff2Score.predict, Encoder, get_decoder,
or parse_staff_tromr.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from attacks.src.homr_wrapper import HOMRBlackBoxWrapper, symbol_to_string


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc

    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(path.suffix + ".tmp")

    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())

    tmp.replace(path)


def emit_event(payload: dict[str, Any], *, quiet: bool = False, stderr: bool = False) -> None:
    if quiet:
        return
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), file=sys.stderr if stderr else sys.stdout, flush=True)


def first_present(row: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        value = row.get(key)

        if value not in (None, ""):
            return value

    return None


def resolve_path(raw: Any, *, base_dir: Path) -> Path:
    p = Path(str(raw))

    if p.is_absolute():
        return p

    candidate = base_dir / p

    if candidate.exists():
        return candidate

    return p


def page_number_from_path(path: Path, fallback: int) -> int:
    stem = path.stem

    for sep in ("_", "-"):
        tail = stem.rsplit(sep, 1)[-1]

        if tail.isdigit():
            return int(tail)

    return fallback


def collect_pages(page_log: Path) -> list[dict[str, Any]]:
    """
    Supports both:
      1. one JSONL row per score with rendered_pages: [...]
      2. one JSONL row per rendered page with image_path/page_path/png_path
    """
    rows = read_jsonl(page_log)
    batch_dir = page_log.parent.parent
    pages: list[dict[str, Any]] = []

    for row_index, row in enumerate(rows):
        status = str(row.get("status", "ok")).lower()

        if status not in {"ok", "success", "done", "completed", "rendered"}:
            continue

        score_id = str(
            first_present(row, ["score_id", "source_id", "id", "stem"])
            or f"score_{row_index:06d}"
        )

        source_path = first_present(row, ["source_path", "mxl_path", "musicxml_path", "xml_path"])

        rendered_pages = row.get("rendered_pages")

        if isinstance(rendered_pages, list):
            for local_index, item in enumerate(rendered_pages, start=1):
                if isinstance(item, dict):
                    raw_image = first_present(
                        item,
                        ["image_path", "page_path", "png_path", "rendered_image", "path"],
                    )
                    fallback_page_number = int(
                        first_present(item, ["page_number", "page", "page_index"])
                        or local_index
                    )
                else:
                    raw_image = item
                    fallback_page_number = local_index

                if raw_image is None:
                    continue

                image_path = resolve_path(raw_image, base_dir=batch_dir)
                page_number = page_number_from_path(image_path, fallback_page_number)
                page_id = image_path.stem

                pages.append(
                    {
                        "score_id": score_id,
                        "source_path": str(source_path) if source_path else None,
                        "page_id": page_id,
                        "page_number": page_number,
                        "image_path": image_path,
                    }
                )

            continue

        raw_image = first_present(
            row,
            ["image_path", "page_path", "png_path", "rendered_image", "output_png", "path"],
        )

        if raw_image is None:
            continue

        image_path = resolve_path(raw_image, base_dir=batch_dir)
        fallback_page_number = int(first_present(row, ["page_number", "page", "page_index"]) or 1)
        page_number = page_number_from_path(image_path, fallback_page_number)
        page_id = image_path.stem

        pages.append(
            {
                "score_id": score_id,
                "source_path": str(source_path) if source_path else None,
                "page_id": page_id,
                "page_number": page_number,
                "image_path": image_path,
            }
        )

    return pages


def load_completed_pages(log_path: Path) -> set[str]:
    if not log_path.exists():
        return set()

    completed: set[str] = set()

    for row in read_jsonl(log_path):
        if row.get("status") == "ok" and row.get("page_id"):
            completed.add(str(row["page_id"]))

    return completed


def symbol_to_payload(symbol: Any) -> dict[str, Any]:
    return {
        "rhythm": str(getattr(symbol, "rhythm", "")),
        "pitch": str(getattr(symbol, "pitch", "")),
        "lift": str(getattr(symbol, "lift", "")),
        "articulation": str(getattr(symbol, "articulation", "")),
        "position": str(getattr(symbol, "position", "")),
        "coordinates": getattr(symbol, "coordinates", None),
        "token": canonical_symbol_token(symbol),
        "raw_token": symbol_to_string(symbol),
        "string": str(symbol),
    }


def canonical_symbol_token(symbol: Any) -> str:
    """
    Convert a HOMR EncodedSymbol-like object into the canonical flat token used
    for Track C distillation targets.

    HOMR symbols sometimes carry stale pitch/lift/articulation/position values on
    structural rhythm tokens such as barline and chord. For student targets those
    structural symbols must be normalized, otherwise the manifest gets tokens like
    "barline C5 # _ upper" or "chord F4 # _ upper".

    Normal music-bearing symbols are preserved as:
        rhythm pitch lift articulation position
    """
    rhythm = str(getattr(symbol, "rhythm", ""))
    pitch = str(getattr(symbol, "pitch", "."))
    lift = str(getattr(symbol, "lift", "."))
    articulation = str(getattr(symbol, "articulation", "."))
    position = str(getattr(symbol, "position", "."))

    # Sequence/control symbols from HOMR should remain simple rhythm tokens if
    # they ever appear here. Track C adds <STAFF_BREAK> separately between staffs.
    if rhythm in {"PAD", "BOS", "EOS"}:
        return rhythm

    # Chord is a structural separator, not a note-bearing event.
    if rhythm == "chord":
        return "chord . . . ."

    # Barline-like and repeat/volta symbols are structural.
    if rhythm.startswith((
        "barline",
        "doublebarline",
        "bolddoublebarline",
        "repeat",
        "volta",
    )):
        return f"{rhythm} . . . ."

    # Key/time signatures are global structural symbols in this target format.
    if rhythm.startswith(("keySignature", "timeSignature")):
        return f"{rhythm} . . . ."

    return " ".join((rhythm, pitch, lift, articulation, position))


def canonicalize_tokens(symbols: list[Any]) -> list[str]:
    return [canonical_symbol_token(symbol) for symbol in symbols]


def read_staff_metadata(page_cache_dir: Path) -> dict[str, Any]:
    metadata_path = page_cache_dir / "metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Prepared-staff metadata missing: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_one_page(
    *,
    page: dict[str, Any],
    segnet: Any,
    cache_one_image_fn: Any,
    tromr: HOMRBlackBoxWrapper,
    prepared_cache_dir: Path,
    teacher_dir: Path,
    overwrite: bool,
) -> dict[str, Any]:
    started = time.time()

    image_path = Path(page["image_path"])

    if not image_path.exists():
        raise FileNotFoundError(f"Rendered page image not found: {image_path}")

    cache_summary = cache_one_image_fn(
        image_path=image_path,
        output_dir=prepared_cache_dir,
        segnet=segnet,
        enable_debug=False,
        overwrite=overwrite,
        strict_shape=True,
    )

    page_cache_dir = Path(cache_summary["output_dir"])
    metadata = read_staff_metadata(page_cache_dir)

    staffs_out: list[dict[str, Any]] = []
    canonical_staff_token_sequences: list[list[str]] = []

    for staff_meta in metadata.get("staffs", []):
        staff_index = int(staff_meta["index"])
        npy_path = page_cache_dir / staff_meta["filename_npy"]

        symbols = tromr.predict_file(npy_path)
        tokens = canonicalize_tokens(symbols)

        canonical_staff_token_sequences.append(tokens)

        staffs_out.append(
            {
                "staff_index": staff_index,
                "voice_index": int(staff_meta.get("voice_index", 0)),
                "staff_index_within_voice": int(staff_meta.get("staff_index_within_voice", 0)),
                "prepared_staff_npy": str(npy_path),
                "prepared_staff_png": str(page_cache_dir / staff_meta["filename_png"]),
                "shape": staff_meta.get("shape"),
                "is_grandstaff": bool(staff_meta.get("is_grandstaff", False)),
                "token_count": len(tokens),
                "tokens": tokens,
                "symbols": [symbol_to_payload(symbol) for symbol in symbols],
            }
        )

    canonical_flat_tokens: list[str] = []

    for staff_i, tokens in enumerate(canonical_staff_token_sequences):
        if staff_i > 0:
            canonical_flat_tokens.append("<STAFF_BREAK>")
        canonical_flat_tokens.extend(tokens)

    teacher_payload = {
        "status": "ok",
        "schema": "onnx_homr_teacher_v1",

        "score_id": page["score_id"],
        "source_path": page["source_path"],
        "page_id": page["page_id"],
        "page_number": page["page_number"],
        "image_path": str(image_path),

        "pipeline": {
            "segnet": "models/onnx/segnet.onnx via ONNX Runtime",
            "staff_preparation": "deterministic HOMR geometry + prepare_staff_image",
            "tromr_encoder": "models/onnx/tromr_encoder.onnx via ONNX Runtime",
            "tromr_decoder": "models/onnx/tromr_decoder.onnx via ONNX Runtime",
        },

        "prepared_cache_dir": str(page_cache_dir),
        "n_staffs": len(staffs_out),
        "n_voices": int(metadata.get("n_voices", 0)),
        "staffs": staffs_out,

        "canonical": {
            "target_schema": "staff_token_sequences_v1",
            "staff_token_sequences": canonical_staff_token_sequences,
            "flat_tokens_with_staff_breaks": canonical_flat_tokens,
        },

        "elapsed_seconds": time.time() - started,
    }

    out_path = teacher_dir / f"{page['page_id']}.json"
    write_json_atomic(out_path, teacher_payload)

    return {
        "teacher_path": str(out_path),
        "n_staffs": len(staffs_out),
        "elapsed_seconds": teacher_payload["elapsed_seconds"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--render-log", type=Path, help="Backward-compatible clean render log JSONL.")
    group.add_argument("--page-log", type=Path, help="Generalized page log JSONL; accepts clean render logs or augment_log.jsonl.")

    parser.add_argument(
        "--teacher-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--teacher-log",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--prepared-cache-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/onnx"),
    )

    parser.add_argument(
        "--segnet-onnx",
        type=Path,
        default=Path("models/onnx/segnet.onnx"),
    )

    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--max-pages", type=int, default=-1)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Print JSON progress every N page records. Use 0 to suppress periodic progress.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress terminal progress prints except errors and final completion.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    page_log = args.page_log or args.render_log
    batch_dir = page_log.parent.parent

    teacher_dir = args.teacher_dir or batch_dir / "teacher_outputs"
    teacher_log = args.teacher_log or batch_dir / "logs" / "teacher_log.jsonl"
    prepared_cache_dir = args.prepared_cache_dir or batch_dir / "prepared_staffs"

    teacher_dir.mkdir(parents=True, exist_ok=True)
    teacher_log.parent.mkdir(parents=True, exist_ok=True)
    prepared_cache_dir.mkdir(parents=True, exist_ok=True)

    pages = collect_pages(page_log)

    if args.max_pages > 0:
        pages = pages[: args.max_pages]

    if not pages:
        raise RuntimeError(f"No pages found in {page_log}")

    completed = set() if args.overwrite else load_completed_pages(teacher_log)

    # Important: construct the TrOMR ONNX wrapper before importing
    # dataset.cache_prepared_staffs. The staff-prep module imports HOMR staff parsing
    # utilities, which can load forbidden neural modules into sys.modules as side
    # effects. The wrapper's guard must run before that happens.
    tromr = HOMRBlackBoxWrapper(
        model_dir=args.model_dir,
        use_cuda=not args.cpu,
        strict_shape=True,
        strict_import_guard=True,
    )

    from dataset.cache_prepared_staffs import SegNetONNX, cache_one_image

    segnet = SegNetONNX(
        model_path=args.segnet_onnx,
        use_cuda=not args.cpu,
        batch_size=8,
        win_size=320,
        step_size=320,
)

    emit_event(
        {
            "event": "start",
            "page_log": str(page_log),
            "render_log": str(args.render_log) if args.render_log else None,
            "n_pages": len(pages),
            "teacher_dir": str(teacher_dir),
            "teacher_log": str(teacher_log),
            "prepared_cache_dir": str(prepared_cache_dir),
            "device": "cpu" if args.cpu else "cuda",
        },
        quiet=args.quiet,
    )

    ok = 0
    skipped = 0
    errors = 0
    started = time.time()

    for index, page in enumerate(pages, start=1):
        page_id = str(page["page_id"])

        if page_id in completed:
            skipped += 1
            if args.progress_every > 0 and (index == 1 or index % args.progress_every == 0 or index == len(pages)):
                emit_event(
                    {
                        "event": "progress",
                        "status": "skipped_existing",
                        "processed": index,
                        "total": len(pages),
                        "ok": ok,
                        "skipped": skipped,
                        "errors": errors,
                        "page_id": page_id,
                        "seconds": time.time() - started,
                    },
                    quiet=args.quiet,
                )
            continue

        if args.progress_every > 0 and (index == 1 or index % args.progress_every == 0):
            emit_event(
                {
                    "event": "page_start",
                    "processed": index - 1,
                    "total": len(pages),
                    "page_id": page_id,
                    "score_id": page["score_id"],
                    "image_path": str(page["image_path"]),
                },
                quiet=args.quiet,
            )

        try:
            result = run_one_page(
                page=page,
                segnet=segnet,
                cache_one_image_fn=cache_one_image,
                tromr=tromr,
                prepared_cache_dir=prepared_cache_dir,
                teacher_dir=teacher_dir,
                overwrite=args.overwrite,
            )

            log_row = {
                "status": "ok",
                "score_id": page["score_id"],
                "page_id": page_id,
                "page_number": page["page_number"],
                "image_path": str(page["image_path"]),
                "teacher_path": result["teacher_path"],
                "n_staffs": result["n_staffs"],
                "elapsed_seconds": result["elapsed_seconds"],
                "timestamp_unix": time.time(),
            }
            append_jsonl(teacher_log, log_row)

            ok += 1
            if args.progress_every > 0 and (index == 1 or index % args.progress_every == 0 or index == len(pages)):
                emit_event(
                    {
                        "event": "page_done",
                        "processed": index,
                        "total": len(pages),
                        "ok": ok,
                        "skipped": skipped,
                        "errors": errors,
                        "seconds": time.time() - started,
                        **log_row,
                    },
                    quiet=args.quiet,
                )

        except Exception as exc:
            errors += 1

            error_row = {
                "status": "error",
                "score_id": page["score_id"],
                "page_id": page_id,
                "page_number": page["page_number"],
                "image_path": str(page["image_path"]),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "timestamp_unix": time.time(),
            }

            append_jsonl(teacher_log, error_row)
            emit_event({"event": "error", **error_row}, quiet=False, stderr=True)

            if not args.continue_on_error:
                return 1

    print(
        json.dumps(
            {
                "event": "done",
                "total": len(pages),
                "ok": ok,
                "skipped": skipped,
                "errors": errors,
                "seconds": time.time() - started,
                "teacher_log": str(teacher_log),
                "teacher_dir": str(teacher_dir),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    return 0 if errors == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())