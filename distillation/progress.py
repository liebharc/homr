#!/usr/bin/env python3
"""Uniform JSON progress events for the adversarial-HOMR pipeline.

All terminal progress emitted by pipeline scripts should use this module so the
base schema is stable across rendering, teacher generation, manifest building,
splitting, target encoding, and student training. Domain-specific measurements
belong under ``metrics``; common progress/time/rate/count fields stay uniform.
"""

from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

SCHEMA = "adv_homr_pipeline_event_v1"

COUNT_KEYS = {
    "ok",
    "skipped",
    "errors",
    "failed",
    "rows",
    "scores",
    "pages",
    "staffs",
    "staff_count",
    "token_count",
    "teacher_files",
    "encoded_rows",
    "invalid_symbols",
    "n_staffs",
    "total_target_tokens",
}

PATH_KEY_MARKERS = ("path", "dir", "manifest", "log", "output", "latest", "metrics")
ID_KEY_MARKERS = ("id",)


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def format_duration(seconds: float | int | None) -> str | None:
    if seconds is None:
        return None
    try:
        value = float(seconds)
    except Exception:
        return None
    if not math.isfinite(value):
        return None
    value = max(0.0, value)
    hours = int(value // 3600)
    minutes = int((value % 3600) // 60)
    secs = value % 60
    if hours:
        return f"{hours}h {minutes}m {secs:04.1f}s"
    if minutes:
        return f"{minutes}m {secs:04.1f}s"
    return f"{secs:.1f}s"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except Exception:
        return None
    return result if math.isfinite(result) else None


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def build_event(
    *,
    stage: str,
    event: str,
    status: str | None = None,
    unit: str = "items",
    done: int | None = None,
    total: int | None = None,
    elapsed_seconds: float | None = None,
    start_time: float | None = None,
    counts: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    ids: dict[str, Any] | None = None,
    paths: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if elapsed_seconds is None and start_time is not None:
        elapsed_seconds = time.time() - start_time

    elapsed = _as_float(elapsed_seconds)
    done_i = _as_int(done)
    total_i = _as_int(total)
    remaining = None
    fraction = None
    percent = None
    eta = None
    rate_per_second = None
    rate_per_minute = None

    if done_i is not None and total_i is not None:
        remaining = max(0, total_i - done_i)
        if total_i > 0:
            fraction = max(0.0, min(1.0, done_i / total_i))
            percent = fraction * 100.0

    if elapsed is not None and elapsed > 0 and done_i is not None and done_i > 0:
        rate_per_second = done_i / elapsed
        rate_per_minute = rate_per_second * 60.0
        if remaining is not None and rate_per_second > 0:
            eta = remaining / rate_per_second

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "timestamp_iso": now_iso(),
        "timestamp_unix": time.time(),
        "stage": stage,
        "event": event,
        "status": status,
        "unit": unit,
        "progress": {
            "done": done_i,
            "total": total_i,
            "remaining": remaining,
            "fraction": fraction,
            "percent": percent,
        },
        "time": {
            "elapsed_seconds": elapsed,
            "elapsed_human": format_duration(elapsed),
            "eta_seconds": eta,
            "eta_human": format_duration(eta),
        },
        "rate": {
            "items_per_second": rate_per_second,
            "items_per_minute": rate_per_minute,
            "unit": unit,
        },
        "counts": _jsonable(counts or {}),
        "metrics": _jsonable(metrics or {}),
        "ids": _jsonable(ids or {}),
        "paths": _jsonable(paths or {}),
    }

    if extra:
        payload["extra"] = _jsonable(extra)

    # Drop None fields recursively enough to keep terminal output readable.
    return _strip_none(payload)


def _strip_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _strip_none(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_strip_none(v) for v in value]
    return value


def event_from_legacy_payload(stage: str, payload: dict[str, Any], *, default_unit: str = "items") -> dict[str, Any]:
    source = dict(payload)
    event = str(source.pop("event", "progress"))
    status = source.pop("status", None)

    done = None
    for key in ("processed", "done", "step", "rows"):
        if key in source:
            done = source.pop(key)
            break

    total = None
    for key in ("total", "total_steps", "n_pages", "teacher_files"):
        if key in source:
            total = source.pop(key)
            break

    elapsed = None
    for key in ("elapsed_seconds", "elapsed_sec", "seconds"):
        if key in source:
            elapsed = source.pop(key)
            break

    unit = str(source.pop("unit", default_unit))

    counts: dict[str, Any] = {}
    metrics: dict[str, Any] = {}
    ids: dict[str, Any] = {}
    paths: dict[str, Any] = {}
    extra: dict[str, Any] = {}

    for key, value in source.items():
        lower = key.lower()
        if key in COUNT_KEYS or lower.endswith("_count") or lower.endswith("_total"):
            counts[key] = value
        elif any(marker in lower for marker in PATH_KEY_MARKERS):
            paths[key] = value
        elif lower.endswith("_id") or lower in {"id"}:
            ids[key] = value
        elif lower.endswith("loss") or lower.endswith("_loss") or lower in {
            "loss", "lr", "learning_rate", "speed", "device", "batch_size", "epoch",
            "split", "last_status", "reason", "error", "error_type", "traceback",
            "best_loss", "vocab_sizes", "max_staffs", "max_seq_len",
        }:
            metrics[key] = value
        else:
            extra[key] = value

    return build_event(
        stage=stage,
        event=event,
        status=str(status) if status is not None else None,
        unit=unit,
        done=_as_int(done),
        total=_as_int(total),
        elapsed_seconds=_as_float(elapsed),
        counts=counts,
        metrics=metrics,
        ids=ids,
        paths=paths,
        extra=extra,
    )


def emit_pipeline_event(
    *,
    stage: str,
    payload: dict[str, Any] | None = None,
    quiet: bool = False,
    stderr: bool = False,
    default_unit: str = "items",
    **kwargs: Any,
) -> dict[str, Any]:
    if payload is not None and kwargs:
        raise ValueError("Pass either payload=... or keyword event fields, not both.")
    if payload is not None:
        event = event_from_legacy_payload(stage, payload, default_unit=default_unit)
    else:
        event = build_event(stage=stage, **kwargs)
    if not quiet:
        print(
            json.dumps(event, ensure_ascii=False, sort_keys=True),
            file=sys.stderr if stderr else sys.stdout,
            flush=True,
        )
    return event
