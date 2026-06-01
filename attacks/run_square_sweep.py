"""
Track B Square Attack sweep runner for cached HOMR-prepared staff images.

Features
--------
- Runs Track B Square Attack over cached prepared staff `.npy` files.
- Stores aggregate metrics JSON for paper tables and plots.
- Stores per-job JSONL records.
- Supports checkpoint/resume with --resume.
- Automatically skips completed staff-epsilon jobs.
- Stores raw SER/CER and capped SER/CER.
- Optionally generates plots after the sweep with --plot.

Important
---------
This script attacks cached HOMR-prepared staff images directly.

It must not rerun:
- full-page HOMR layout
- SegNet
- Staff2Score.predict
- parse_staff_tromr
- PyTorch/TensorFlow neural inference

Expected path:
    cached prepared staff image
      -> HomrWrapper.predict_prepared_staff(...)
      -> run_square_attack(...)
      -> HomrWrapper.predict_prepared_staff(x_adv)
      -> SER/CER metrics

Metric note
-----------
Raw SER/CER are Levenshtein edit distances normalized by the clean reference length.
They are not upper-bounded by 1 because insertion errors may exceed the reference length.

Capped SER/CER are also stored:
    ser_capped = min(ser, 1.0)
    cer_capped = min(cer, 1.0)

Use raw SER/CER as the scientific metric. Use capped SER/CER for bounded visual comparison.

Real run example
----------------
python attacks/run_square_sweep.py `
  --max-staffs 1000 `
  --epsilon 0.10 0.20 0.30 0.40 `
  --n-max 20 `
  --p-init 0.8 `
  --p-final 0.05 `
  --target-source clean_onnx_prediction `
  --progress-every 1 `
  --checkpoint-jsonl results/logs/TRACK_B_REAL_ATTACK_1000staff_20q_checkpoint.jsonl `
  --timing-jsonl results/logs/TRACK_B_REAL_ATTACK_1000staff_20q_checkpoint.jsonl `
  --output-json results/logs/TRACK_B_REAL_ATTACK_1000staff_20q_metrics.json `
  --resume `
  --plot `
  --plots-dir results/plots
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np


# ---------------------------------------------------------------------------
# Import project modules robustly
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from attacks.src.homr_wrapper import HomrWrapper
    from attacks.src.square_attack import run_square_attack
    from attacks.src.statistics_engine import (
        character_error_rate,
        symbol_error_rate,
    )
except Exception as exc:
    print("[fatal] Failed to import benchmark modules.", file=sys.stderr)
    print(f"[fatal] Repository root inserted into sys.path: {REPO_ROOT}", file=sys.stderr)
    print(f"[fatal] Original error: {exc}", file=sys.stderr)
    raise


# ---------------------------------------------------------------------------
# Timing/progress utilities
# ---------------------------------------------------------------------------

def format_duration(seconds: float) -> str:
    """Format seconds as h/m/s for progress logging."""
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


def now_iso() -> str:
    """Return local timestamp for JSON logs."""
    return datetime.now().isoformat(timespec="seconds")


def append_jsonl(path: str | Path | None, record: dict[str, Any]) -> None:
    """Append a JSON record to a JSONL file if logging is enabled."""
    if path is None:
        return

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


@dataclass
class ProgressSnapshot:
    completed_jobs: int
    total_jobs: int
    elapsed_sec: float
    eta_sec: float
    jobs_per_min: float
    queries_per_sec: float
    total_queries: int
    skipped_jobs: int


class ProgressMeter:
    """Tracks elapsed time, speed, total query count, ETA, and skipped jobs."""

    def __init__(
        self,
        total_jobs: int,
        already_completed: int = 0,
        already_queries: int = 0,
    ) -> None:
        self.total_jobs = max(1, int(total_jobs))
        self.completed_jobs = int(already_completed)
        self.skipped_jobs = int(already_completed)
        self.total_queries = int(already_queries)
        self.start_time = time.perf_counter()

    def update(self, queries: int) -> ProgressSnapshot:
        self.completed_jobs += 1
        self.total_queries += int(queries)

        elapsed_sec = time.perf_counter() - self.start_time

        newly_done = max(1, self.completed_jobs - self.skipped_jobs)
        jobs_per_sec = newly_done / elapsed_sec if elapsed_sec > 0 else 0.0
        jobs_per_min = jobs_per_sec * 60.0

        queries_per_sec = self.total_queries / elapsed_sec if elapsed_sec > 0 else 0.0

        remaining_jobs = max(0, self.total_jobs - self.completed_jobs)
        eta_sec = remaining_jobs / jobs_per_sec if jobs_per_sec > 0 else float("inf")

        return ProgressSnapshot(
            completed_jobs=self.completed_jobs,
            total_jobs=self.total_jobs,
            elapsed_sec=elapsed_sec,
            eta_sec=eta_sec,
            jobs_per_min=jobs_per_min,
            queries_per_sec=queries_per_sec,
            total_queries=self.total_queries,
            skipped_jobs=self.skipped_jobs,
        )

    @staticmethod
    def line(snapshot: ProgressSnapshot) -> str:
        return (
            f"[progress] {snapshot.completed_jobs}/{snapshot.total_jobs} jobs | "
            f"skipped={snapshot.skipped_jobs} | "
            f"elapsed={format_duration(snapshot.elapsed_sec)} | "
            f"ETA={format_duration(snapshot.eta_sec)} | "
            f"speed={snapshot.jobs_per_min:.2f} jobs/min | "
            f"query_speed={snapshot.queries_per_sec:.2f} q/s | "
            f"queries={snapshot.total_queries}"
        )


# ---------------------------------------------------------------------------
# Token and metric helpers
# ---------------------------------------------------------------------------

def token_to_string(token: Any) -> str:
    """Convert arbitrary decoded token / EncodedSymbol-like object to a stable string."""
    if token is None:
        return "<None>"

    if isinstance(token, str):
        return token

    if isinstance(token, (int, float, bool)):
        return str(token)

    if isinstance(token, dict):
        return json.dumps(token, sort_keys=True, ensure_ascii=False)

    for attr_name in (
        "token",
        "name",
        "symbol",
        "text",
        "value",
        "encoded",
        "rhythm",
        "pitch",
    ):
        if hasattr(token, attr_name):
            try:
                return str(getattr(token, attr_name))
            except Exception:
                pass

    return str(token)


def tokens_to_strings(tokens: Iterable[Any]) -> list[str]:
    return [token_to_string(t) for t in tokens]


def compute_metrics(pred_tokens: list[Any], target_tokens: list[Any]) -> dict[str, float]:
    """
    Compute raw and capped SER/CER.

    Raw SER/CER may exceed 1.0 because Levenshtein distance includes insertions.
    Capped metrics are for bounded visualization only.
    """
    pred_list = tokens_to_strings(pred_tokens)
    target_list = tokens_to_strings(target_tokens)

    pred_str = " ".join(pred_list)
    target_str = " ".join(target_list)

    ser = float(symbol_error_rate(pred_list, target_list))
    cer = float(character_error_rate(pred_str, target_str))

    return {
        "ser": ser,
        "cer": cer,
        "ser_capped": min(ser, 1.0),
        "cer_capped": min(cer, 1.0),
    }


# ---------------------------------------------------------------------------
# Cache discovery and metadata loading
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StaffEntry:
    staff_path: Path
    score_dir: Path
    score_stem: str
    staff_filename: str
    staff_index: int | None
    metadata_path: Path | None


def parse_staff_index(filename: str) -> int | None:
    """Extract index from staff_XXX.npy if possible."""
    stem = Path(filename).stem
    try:
        return int(stem.split("_")[-1])
    except Exception:
        return None


def discover_staff_files(cache_dir: Path, max_staffs: int | None) -> list[StaffEntry]:
    """Find cached prepared staff `.npy` files recursively."""
    if not cache_dir.exists():
        raise FileNotFoundError(f"Cache directory does not exist: {cache_dir}")

    staff_paths = sorted(cache_dir.glob("**/staff_*.npy"))

    if max_staffs is not None:
        staff_paths = staff_paths[: max(0, int(max_staffs))]

    entries: list[StaffEntry] = []

    for path in staff_paths:
        score_dir = path.parent
        metadata_path = score_dir / "metadata.json"
        if not metadata_path.exists():
            metadata_path = None

        entries.append(
            StaffEntry(
                staff_path=path,
                score_dir=score_dir,
                score_stem=score_dir.name,
                staff_filename=path.name,
                staff_index=parse_staff_index(path.name),
                metadata_path=metadata_path,
            )
        )

    return entries


def load_metadata_tokens(entry: StaffEntry) -> list[Any]:
    """
    Load gt_tokens from sibling metadata.json.

    Expected metadata shape:
        {
          "staffs": [
            {
              "index": 0,
              "filename_npy": "staff_000.npy",
              "gt_tokens": [...]
            }
          ]
        }
    """
    if entry.metadata_path is None:
        raise FileNotFoundError(
            f"No metadata.json found next to {entry.staff_path}. "
            "Use --target-source clean_onnx_prediction or regenerate cache metadata."
        )

    with entry.metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    staffs = metadata.get("staffs", [])
    if not isinstance(staffs, list):
        raise ValueError(f"Invalid metadata schema in {entry.metadata_path}: staffs is not a list")

    match = None

    for item in staffs:
        if not isinstance(item, dict):
            continue

        if item.get("filename_npy") == entry.staff_filename:
            match = item
            break

        if entry.staff_index is not None and item.get("index") == entry.staff_index:
            match = item
            break

    if match is None:
        raise ValueError(
            f"Could not find staff metadata for {entry.staff_filename} in {entry.metadata_path}"
        )

    gt_tokens = match.get("gt_tokens", [])
    if not gt_tokens:
        raise ValueError(
            f"metadata gt_tokens is empty for {entry.staff_path}. "
            "Use --target-source clean_onnx_prediction for current self-consistency sweeps."
        )

    return gt_tokens


def load_staff_image(path: Path) -> np.ndarray:
    """
    Load a cached prepared staff image and normalize shape/dtype.

    Expected file content:
        float32 [0, 1], shape [256, 1280] or [256, 1280, 1]
    """
    arr = np.load(path)
    arr = np.asarray(arr)

    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[:, :, 0]

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D staff image or HxWx1 array, got shape {arr.shape}: {path}")

    arr = arr.astype(np.float32, copy=False)

    if arr.size == 0:
        raise ValueError(f"Empty staff image: {path}")

    if np.isnan(arr).any() or np.isinf(arr).any():
        raise ValueError(f"Staff image contains NaN or Inf: {path}")

    if arr.max() > 1.0:
        arr = arr / 255.0

    return np.clip(arr, 0.0, 1.0).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Checkpoint/resume helpers
# ---------------------------------------------------------------------------

def checkpoint_key(
    staff_path: str | Path,
    epsilon: float,
    n_max: int,
    target_source: str,
) -> tuple[str, str, int, str]:
    """
    Stable key for one staff-epsilon-query-budget-target job.

    Epsilon is formatted to avoid tiny float representation differences.
    """
    return (
        str(Path(staff_path).as_posix()),
        f"{float(epsilon):.10g}",
        int(n_max),
        str(target_source),
    )


def record_key(record: dict[str, Any]) -> tuple[str, str, int, str] | None:
    """Extract checkpoint key from an existing JSONL record."""
    try:
        return checkpoint_key(
            staff_path=record["staff_path"],
            epsilon=float(record["epsilon"]),
            n_max=int(record["n_max"]),
            target_source=str(record["target_source"]),
        )
    except Exception:
        return None


def load_checkpoint_records(
    checkpoint_path: str | Path | None,
) -> dict[tuple[str, str, int, str], dict[str, Any]]:
    """
    Load successful completed jobs from checkpoint JSONL.

    If the same key appears multiple times, the latest successful record wins.
    """
    if checkpoint_path is None:
        return {}

    path = Path(checkpoint_path)
    if not path.exists():
        return {}

    completed: dict[tuple[str, str, int, str], dict[str, Any]] = {}

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(
                    f"[warning] Ignoring malformed checkpoint JSONL line {line_no}: {path}",
                    file=sys.stderr,
                )
                continue

            if record.get("status") != "ok":
                continue

            key = record_key(record)
            if key is None:
                continue

            completed[key] = record

    return completed


# ---------------------------------------------------------------------------
# Aggregation and output
# ---------------------------------------------------------------------------

def mean_or_nan(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def std_or_nan(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.std(np.asarray(values, dtype=np.float64)))



def _normalize_track_b_record(record: dict[str, Any]) -> dict[str, Any]:
    """Return a unified-schema copy of one Track B job/error record."""
    out = dict(record)
    out.setdefault("track", "B")

    eps = out.get("epsilon")
    if eps is not None:
        try:
            eps_f = float(eps)
            out.setdefault("epsilon", eps_f)
            out.setdefault("epsilon_math", eps_f)
            out.setdefault("condition_id", f"eps_{eps_f:.10g}")
        except Exception:
            out.setdefault("epsilon_math", eps)
            out.setdefault("condition_id", f"eps_{eps}")

    out.setdefault("epsilon_phys", None)
    out.setdefault("alpha", None)

    if out.get("status") == "ok":
        if "adv_ser" in out and "ser" not in out:
            out["ser"] = out["adv_ser"]
        if "adv_cer" in out and "cer" not in out:
            out["cer"] = out["adv_cer"]
        if "adv_ser_capped" in out and "ser_capped" not in out:
            out["ser_capped"] = out["adv_ser_capped"]
        if "adv_cer_capped" in out and "cer_capped" not in out:
            out["cer_capped"] = out["adv_cer_capped"]

        if "ser_delta" in out and "delta_ser" not in out:
            out["delta_ser"] = out["ser_delta"]
        if "cer_delta" in out and "delta_cer" not in out:
            out["delta_cer"] = out["cer_delta"]
        if "ser_capped_delta" in out and "delta_ser_capped" not in out:
            out["delta_ser_capped"] = out["ser_capped_delta"]
        if "cer_capped_delta" in out and "delta_cer_capped" not in out:
            out["delta_cer_capped"] = out["cer_capped_delta"]

        out.setdefault("success", bool(float(out.get("ser", 0.0)) > float(out.get("clean_ser", 0.0))))

    return out


def aggregate_by_epsilon(job_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate per-job records into one unified result row per epsilon."""
    normalized = [_normalize_track_b_record(r) for r in job_records]
    grouped: dict[float, list[dict[str, Any]]] = {}

    for row in normalized:
        if row.get("status") != "ok":
            continue
        eps = float(row["epsilon"])
        grouped.setdefault(eps, []).append(row)

    output: list[dict[str, Any]] = []

    for eps in sorted(grouped):
        rows = grouped[eps]

        ser_values = [float(r["ser"]) for r in rows]
        cer_values = [float(r["cer"]) for r in rows]
        ser_capped_values = [float(r.get("ser_capped", min(float(r["ser"]), 1.0))) for r in rows]
        cer_capped_values = [float(r.get("cer_capped", min(float(r["cer"]), 1.0))) for r in rows]

        clean_ser_values = [float(r["clean_ser"]) for r in rows]
        clean_cer_values = [float(r["clean_cer"]) for r in rows]
        clean_ser_capped_values = [
            float(r.get("clean_ser_capped", min(float(r["clean_ser"]), 1.0))) for r in rows
        ]
        clean_cer_capped_values = [
            float(r.get("clean_cer_capped", min(float(r["clean_cer"]), 1.0))) for r in rows
        ]

        delta_ser_values = [float(r.get("delta_ser", float(r["ser"]) - float(r["clean_ser"]))) for r in rows]
        delta_cer_values = [float(r.get("delta_cer", float(r["cer"]) - float(r["clean_cer"]))) for r in rows]
        delta_ser_capped_values = [
            float(r.get("delta_ser_capped", float(r.get("ser_capped", min(float(r["ser"]), 1.0))) - float(r.get("clean_ser_capped", min(float(r["clean_ser"]), 1.0)))))
            for r in rows
        ]
        delta_cer_capped_values = [
            float(r.get("delta_cer_capped", float(r.get("cer_capped", min(float(r["cer"]), 1.0))) - float(r.get("clean_cer_capped", min(float(r["clean_cer"]), 1.0)))))
            for r in rows
        ]

        query_values = [float(r["n_queries"]) for r in rows]
        success_values = [1.0 if bool(r.get("success")) else 0.0 for r in rows]
        elapsed_values = [float(r["job_elapsed_sec"]) for r in rows]
        qps_values = [float(r["attack_queries_per_sec"]) for r in rows]

        condition_elapsed = float(sum(elapsed_values))

        output.append(
            {
                "track": "B",
                "condition_id": f"eps_{eps:.10g}",
                "epsilon": eps,
                "epsilon_math": eps,
                "epsilon_phys": None,
                "alpha": None,
                "n_staffs": len(rows),
                "n_jobs_total": len(rows),
                "n_jobs_ok": len(rows),
                "n_jobs_error": 0,

                "mean_clean_ser": mean_or_nan(clean_ser_values),
                "std_clean_ser": std_or_nan(clean_ser_values),
                "mean_clean_cer": mean_or_nan(clean_cer_values),
                "std_clean_cer": std_or_nan(clean_cer_values),

                "mean_clean_ser_capped": mean_or_nan(clean_ser_capped_values),
                "std_clean_ser_capped": std_or_nan(clean_ser_capped_values),
                "mean_clean_cer_capped": mean_or_nan(clean_cer_capped_values),
                "std_clean_cer_capped": std_or_nan(clean_cer_capped_values),

                "mean_ser": mean_or_nan(ser_values),
                "std_ser": std_or_nan(ser_values),
                "mean_cer": mean_or_nan(cer_values),
                "std_cer": std_or_nan(cer_values),

                "mean_ser_capped": mean_or_nan(ser_capped_values),
                "std_ser_capped": std_or_nan(ser_capped_values),
                "mean_cer_capped": mean_or_nan(cer_capped_values),
                "std_cer_capped": std_or_nan(cer_capped_values),

                "mean_delta_ser": mean_or_nan(delta_ser_values),
                "std_delta_ser": std_or_nan(delta_ser_values),
                "mean_delta_cer": mean_or_nan(delta_cer_values),
                "std_delta_cer": std_or_nan(delta_cer_values),
                "mean_delta_ser_capped": mean_or_nan(delta_ser_capped_values),
                "std_delta_ser_capped": std_or_nan(delta_ser_capped_values),
                "mean_delta_cer_capped": mean_or_nan(delta_cer_capped_values),
                "std_delta_cer_capped": std_or_nan(delta_cer_capped_values),

                "success_rate": mean_or_nan(success_values),
                "avg_queries": mean_or_nan(query_values),
                "mean_job_elapsed_sec": mean_or_nan(elapsed_values),
                "std_job_elapsed_sec": std_or_nan(elapsed_values),
                "condition_elapsed_sec": condition_elapsed,
                "elapsed_seconds": condition_elapsed,
                "jobs_per_min": float((len(rows) / condition_elapsed) * 60.0) if condition_elapsed > 0 else 0.0,
                "mean_attack_queries_per_sec": mean_or_nan(qps_values),
                "total_queries": int(sum(query_values)),
            }
        )

    return output


def aggregate_track_b_image_level(job_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Aggregate Track B staff jobs back to source score/image level.

    This makes Track B closer to Track A's image_level_results for paired
    image-subset comparisons.
    """
    normalized = [_normalize_track_b_record(r) for r in job_records if r.get("status") == "ok"]
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = {}

    for row in normalized:
        score_stem = str(row.get("score_stem") or Path(str(row.get("staff_path", ""))).parent.name)
        eps = float(row["epsilon"])
        grouped.setdefault((score_stem, eps), []).append(row)

    output: list[dict[str, Any]] = []
    for (score_stem, eps), rows in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        ser_values = [float(r["ser"]) for r in rows]
        cer_values = [float(r["cer"]) for r in rows]
        delta_ser_values = [float(r.get("delta_ser", float(r["ser"]) - float(r["clean_ser"]))) for r in rows]
        delta_cer_values = [float(r.get("delta_cer", float(r["cer"]) - float(r["clean_cer"]))) for r in rows]
        elapsed_values = [float(r.get("job_elapsed_sec", 0.0)) for r in rows]
        output.append(
            {
                "track": "B",
                "status": "ok",
                "condition_id": f"eps_{eps:.10g}",
                "image_stem": score_stem,
                "score_stem": score_stem,
                "epsilon": eps,
                "epsilon_math": eps,
                "epsilon_phys": None,
                "alpha": None,
                "n_staffs": len(rows),
                "n_staff_pairs": len(rows),
                "mean_ser": mean_or_nan(ser_values),
                "mean_cer": mean_or_nan(cer_values),
                "mean_delta_ser": mean_or_nan(delta_ser_values),
                "mean_delta_cer": mean_or_nan(delta_cer_values),
                "success_rate": mean_or_nan([1.0 if bool(r.get("success")) else 0.0 for r in rows]),
                "job_elapsed_sec": float(sum(elapsed_values)),
                "mean_job_elapsed_sec": mean_or_nan(elapsed_values),
                "total_queries": int(sum(float(r.get("n_queries", 0)) for r in rows)),
            }
        )
    return output



def write_summary_json(
    output_path: Path,
    args: argparse.Namespace,
    staff_entries: list[StaffEntry],
    job_records: list[dict[str, Any]],
    run_start_iso: str,
    run_elapsed_sec: float,
    resumed_records_count: int,
) -> None:
    """Write final metrics JSON with unified Track A/B-compatible schema."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    normalized_records = [_normalize_track_b_record(r) for r in job_records]
    ok_records = [r for r in normalized_records if r.get("status") == "ok"]
    error_records = [r for r in normalized_records if r.get("status") != "ok"]
    aggregate_results = aggregate_by_epsilon(normalized_records)
    image_level_results = aggregate_track_b_image_level(ok_records)

    payload = {
        "experiment_metadata": {
            "run_start": run_start_iso,
            "run_end": now_iso(),
            "run_elapsed_sec": run_elapsed_sec,
            "run_elapsed": format_duration(run_elapsed_sec),
            "track": "B",
            "benchmark_mode": "cached_prepared_staff_tromr_onnx",
            "experiment": "square_attack",
            "execution_order": "staff_major_clean_once_epsilon_inner",
            "target_source": args.target_source,
            "cache_dir": str(Path(args.cache_dir)),
            "n_staff_files_discovered": len(staff_entries),
            "n_images_approximated_from_score_dirs": len({entry.score_stem for entry in staff_entries}),
            "n_jobs_total": len(normalized_records),
            "n_jobs_ok": len(ok_records),
            "n_jobs_error": len(error_records),
            "resumed_records_count": int(resumed_records_count),
            "epsilons": [float(x) for x in args.epsilon],
            "epsilon_math_grid": [float(x) for x in args.epsilon],
            "n_max": int(args.n_max),
            "p_init": float(args.p_init),
            "p_final": float(args.p_final),
            "max_staffs": args.max_staffs,
            "timing_jsonl": args.timing_jsonl,
            "checkpoint_jsonl": args.checkpoint_jsonl,
            "resume": bool(args.resume),
            "plot": bool(args.plot),
            "plots_dir": args.plots_dir,
            "metric_notes": [
                "Raw SER/CER are normalized Levenshtein edit distances and may exceed 1.0 due to insertions.",
                "Capped SER/CER are min(raw_metric, 1.0) and are included only for bounded visual comparison.",
                "Delta SER/CER are adversarial minus clean baseline and are the preferred cross-track comparison metrics.",
            ],
            "notes": [
                "Track B attacks cached HOMR-prepared staff images directly.",
                "This runner does not intentionally rerun full-page HOMR layout inside the attack loop.",
                "clean_onnx_prediction target source is self-consistency scoring, not final MusicXML ground truth scoring.",
                "Checkpoint key is (staff_path, epsilon, n_max, target_source).",
                "jobs/staff_level_results are staff-epsilon jobs; image_level_results aggregate staff jobs by score/image stem.",
            ],
        },
        # Unified schema preferred by plotters and downstream analysis.
        "aggregate_results": aggregate_results,
        "jobs": normalized_records,
        "staff_level_results": ok_records,
        "image_level_results": image_level_results,
        "errors": error_records,
        # Backward-compatible Track-B alias.
        "square_attack_results": aggregate_results,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"[done] Wrote summary JSON: {output_path}")



def try_generate_plots(args: argparse.Namespace, output_json: Path) -> None:
    """
    Generate plots after completed sweep.

    Fail-safe:
    - Metrics JSON is already written before this function runs.
    - If plotting fails, attack results are not lost.
    """
    if not args.plot:
        return

    print("[plot] Automatic plotting enabled.")

    try:
        import importlib.util

        candidates = [
            REPO_ROOT / "attacks" / "python scripts" / "plot_results.py",
            REPO_ROOT / "attacks" / "plot_results.py",
        ]

        plot_module = None
        for candidate in candidates:
            if not candidate.exists():
                continue
            spec = importlib.util.spec_from_file_location("homr_benchmark_plot_results", candidate)
            if spec is None or spec.loader is None:
                continue
            plot_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(plot_module)
            break

        if plot_module is None or not hasattr(plot_module, "plot_square_results"):
            raise ImportError(
                "Could not find plot_square_results in attacks/python scripts/plot_results.py "
                "or attacks/plot_results.py"
            )

        try:
            plot_module.plot_square_results(
                square_json=output_json,
                output_dir=Path(args.plots_dir),
                plot_set="core",
            )
        except TypeError:
            plot_module.plot_square_results(
                square_json=output_json,
                output_dir=Path(args.plots_dir),
            )

        print(f"[plot] Wrote plots to: {args.plots_dir}")

    except Exception as exc:
        print(
            f"[warning] Sweep finished, but automatic plotting failed: {exc}",
            file=sys.stderr,
        )
        print(
            "[warning] You can still generate plots manually after fixing the issue with:",
            file=sys.stderr,
        )
        print(
            f'python "attacks/python scripts/plot_results.py" --square-json {output_json} --output-dir {args.plots_dir}',
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Track B Square Attack sweep over cached HOMR-prepared staff images. "
            "Uses staff-major execution so each clean staff prediction is computed once "
            "and reused across all epsilon values."
        )
    )

    parser.add_argument(
        "--cache-dir",
        type=str,
        default="dataset/cached_prepared_staffs",
        help="Directory containing cached prepared staff folders.",
    )

    parser.add_argument(
        "--output-json",
        type=str,
        default="results/logs/square_sweep_metrics.json",
        help="Output JSON metrics path.",
    )

    parser.add_argument(
        "--timing-jsonl",
        type=str,
        default=None,
        help=(
            "Optional JSONL timing/checkpoint output path. Example: "
            "results/logs/square_sweep_timing.jsonl"
        ),
    )

    parser.add_argument(
        "--checkpoint-jsonl",
        type=str,
        default=None,
        help=(
            "JSONL checkpoint path. If omitted, --timing-jsonl is used. "
            "When --resume is enabled, successful records from this file are skipped."
        ),
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint JSONL and skip already completed jobs.",
    )

    parser.add_argument(
        "--max-staffs",
        type=int,
        default=None,
        help="Maximum number of staff .npy files to evaluate.",
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        nargs="+",
        default=[0.01, 0.02, 0.05, 0.10, 0.20],
        help="One or more epsilon_math values.",
    )

    parser.add_argument(
        "--n-max",
        type=int,
        default=100,
        help="Maximum Square Attack queries per staff-epsilon job.",
    )

    parser.add_argument(
        "--p-init",
        type=float,
        default=0.8,
        help="Initial square size fraction.",
    )

    parser.add_argument(
        "--p-final",
        type=float,
        default=0.05,
        help="Final square size fraction.",
    )

    parser.add_argument(
        "--target-source",
        type=str,
        default="clean_onnx_prediction",
        choices=["clean_onnx_prediction", "metadata_gt_tokens"],
        help=(
            "Target tokens for score_query. Use clean_onnx_prediction for current "
            "self-consistency sweeps when metadata gt_tokens are empty."
        ),
    )

    parser.add_argument(
        "--profile-timing",
        action="store_true",
        help=(
            "Print detailed per-section timings: clean prediction, attack, adversarial "
            "prediction, job time, and query speed."
        ),
    )

    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Print progress every N completed staff-epsilon jobs. Use 0 to disable.",
    )

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue sweep if an individual staff-epsilon job fails.",
    )

    parser.add_argument(
        "--save-adv-dir",
        type=str,
        default=None,
        help=(
            "Optional directory to save adversarial .npy files. Disabled by default "
            "to avoid generating many files."
        ),
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Automatically generate CSV table and HTML plots after the sweep finishes.",
    )

    parser.add_argument(
        "--plots-dir",
        type=str,
        default="results/plots",
        help="Directory for automatically generated plots when --plot is enabled.",
    )

    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.n_max <= 0:
        raise ValueError("--n-max must be positive")

    if args.p_init <= 0:
        raise ValueError("--p-init must be positive")

    if args.p_final <= 0:
        raise ValueError("--p-final must be positive")

    if args.max_staffs is not None and args.max_staffs <= 0:
        raise ValueError("--max-staffs must be positive if provided")

    if not args.epsilon:
        raise ValueError("--epsilon must contain at least one value")

    for eps in args.epsilon:
        if eps < 0:
            raise ValueError(f"epsilon must be non-negative, got {eps}")
        if eps > 1:
            raise ValueError(
                f"epsilon={eps} is > 1.0 in normalized image space; refusing to run."
            )

    if args.checkpoint_jsonl is None:
        args.checkpoint_jsonl = args.timing_jsonl

    if args.resume and args.checkpoint_jsonl is None:
        raise ValueError("--resume requires --checkpoint-jsonl or --timing-jsonl")

    if args.timing_jsonl is None and args.checkpoint_jsonl is not None:
        args.timing_jsonl = args.checkpoint_jsonl


# ---------------------------------------------------------------------------
# Optional adversarial output saving
# ---------------------------------------------------------------------------

def save_adv_image_if_requested(
    save_adv_dir: str | None,
    entry: StaffEntry,
    epsilon: float,
    x_adv: np.ndarray,
) -> str | None:
    """Optionally save adversarial staff image as .npy."""
    if save_adv_dir is None:
        return None

    out_dir = Path(save_adv_dir) / entry.score_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    eps_label = f"{epsilon:.4f}".replace(".", "p")
    out_path = out_dir / f"{Path(entry.staff_filename).stem}_eps_{eps_label}_adv.npy"

    np.save(out_path, x_adv.astype(np.float32, copy=False))

    return str(out_path)


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_sweep(args: argparse.Namespace) -> int:
    validate_args(args)

    run_start_iso = now_iso()
    run_start = time.perf_counter()

    cache_dir = Path(args.cache_dir)
    output_json = Path(args.output_json)

    print("[start] Track B Square Attack sweep")
    print(f"[start] cache_dir={cache_dir}")
    print(f"[start] epsilons={args.epsilon}")
    print(f"[start] n_max={args.n_max}, p_init={args.p_init}, p_final={args.p_final}")
    print(f"[start] target_source={args.target_source}")
    print(f"[start] resume={args.resume}")
    if args.checkpoint_jsonl:
        print(f"[start] checkpoint JSONL: {args.checkpoint_jsonl}")
    if args.timing_jsonl:
        print(f"[start] timing JSONL: {args.timing_jsonl}")
    if args.profile_timing:
        print("[start] detailed timing enabled via --profile-timing")
    if args.plot:
        print(f"[start] automatic plotting enabled: {args.plots_dir}")

    staff_entries = discover_staff_files(cache_dir, args.max_staffs)
    if not staff_entries:
        raise FileNotFoundError(
            f"No staff_*.npy files found under {cache_dir}. "
            "Run dataset/cache_prepared_staffs.py first."
        )

    print(f"[start] discovered_staffs={len(staff_entries)}")

    total_jobs = len(staff_entries) * len(args.epsilon)

    completed_records: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    if args.resume:
        completed_records = load_checkpoint_records(args.checkpoint_jsonl)
        print(f"[resume] loaded completed checkpoint records={len(completed_records)}")

    expected_keys: set[tuple[str, str, int, str]] = set()
    for entry in staff_entries:
        for eps in args.epsilon:
            expected_keys.add(
                checkpoint_key(
                    staff_path=entry.staff_path,
                    epsilon=float(eps),
                    n_max=int(args.n_max),
                    target_source=args.target_source,
                )
            )

    resumed_records = {
        key: record for key, record in completed_records.items()
        if key in expected_keys
    }

    job_records: list[dict[str, Any]] = list(resumed_records.values())
    resumed_records_count = len(resumed_records)

    already_queries = sum(int(r.get("n_queries", 0)) for r in resumed_records.values())
    progress = ProgressMeter(
        total_jobs=total_jobs,
        already_completed=resumed_records_count,
        already_queries=already_queries,
    )

    if resumed_records_count:
        print(
            f"[resume] matching completed jobs={resumed_records_count}/{total_jobs}; "
            f"already_queries={already_queries}"
        )

    # Important: wrapper is constructed once, not inside query loops.
    t0 = time.perf_counter()
    wrapper = HomrWrapper()
    wrapper_load_sec = time.perf_counter() - t0
    print(f"[timing] HomrWrapper load/init={wrapper_load_sec:.3f}s")

    for staff_counter, entry in enumerate(staff_entries, start=1):
        staff_outer_start = time.perf_counter()

        print(
            f"[staff] {staff_counter}/{len(staff_entries)} "
            f"{entry.score_stem}/{entry.staff_filename}"
        )

        pending_epsilons = []
        for epsilon in args.epsilon:
            key = checkpoint_key(
                staff_path=entry.staff_path,
                epsilon=float(epsilon),
                n_max=int(args.n_max),
                target_source=args.target_source,
            )
            if key not in resumed_records:
                pending_epsilons.append(float(epsilon))

        if not pending_epsilons:
            if args.profile_timing:
                print(f"[resume] all epsilons already completed for {entry.staff_path}")
            continue

        try:
            t0 = time.perf_counter()
            clean_image = load_staff_image(entry.staff_path)
            load_staff_sec = time.perf_counter() - t0

            t0 = time.perf_counter()
            clean_pred_tokens = wrapper.predict_prepared_staff(clean_image)
            clean_predict_sec = time.perf_counter() - t0

            if args.target_source == "clean_onnx_prediction":
                target_tokens = clean_pred_tokens
                target_load_sec = 0.0
            elif args.target_source == "metadata_gt_tokens":
                t0 = time.perf_counter()
                target_tokens = load_metadata_tokens(entry)
                target_load_sec = time.perf_counter() - t0
            else:
                raise ValueError(f"Unsupported target source: {args.target_source}")

            clean_metrics = compute_metrics(clean_pred_tokens, target_tokens)

            if args.profile_timing:
                print(
                    f"[timing] staff={staff_counter}/{len(staff_entries)} | "
                    f"load_staff={load_staff_sec:.3f}s | "
                    f"clean_predict={clean_predict_sec:.3f}s | "
                    f"target_load={target_load_sec:.3f}s | "
                    f"target_tokens={len(target_tokens)} | "
                    f"clean_ser={clean_metrics['ser']:.6f} | "
                    f"clean_cer={clean_metrics['cer']:.6f}"
                )

        except Exception as exc:
            error_record = {
                "timestamp": now_iso(),
                "status": "error",
                "phase": "staff_setup",
                "staff_counter": staff_counter,
                "staff_path": str(entry.staff_path),
                "score_stem": entry.score_stem,
                "staff_filename": entry.staff_filename,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
            job_records.append(error_record)
            append_jsonl(args.timing_jsonl, error_record)

            print(f"[error] Failed staff setup for {entry.staff_path}: {exc}", file=sys.stderr)

            if args.continue_on_error:
                for _ in pending_epsilons:
                    snapshot = progress.update(queries=0)
                    if args.progress_every > 0 and snapshot.completed_jobs % args.progress_every == 0:
                        print(ProgressMeter.line(snapshot))
                continue

            raise

        for epsilon in args.epsilon:
            key = checkpoint_key(
                staff_path=entry.staff_path,
                epsilon=float(epsilon),
                n_max=int(args.n_max),
                target_source=args.target_source,
            )

            if key in resumed_records:
                if args.profile_timing:
                    print(
                        f"[resume] skip completed staff={entry.staff_path} "
                        f"epsilon={float(epsilon):.4f}"
                    )
                continue

            job_start = time.perf_counter()

            try:
                t0 = time.perf_counter()
                attack_result = run_square_attack(
                    staff_image=clean_image,
                    target_tokens=target_tokens,
                    wrapper=wrapper,
                    epsilon=float(epsilon),
                    n_max=int(args.n_max),
                    p_init=float(args.p_init),
                    p_final=float(args.p_final),
                )
                attack_sec = time.perf_counter() - t0

                x_adv = attack_result["x_adv"]
                n_queries = int(attack_result.get("n_queries", args.n_max))
                l_best = float(attack_result.get("L_best", float("nan")))
                loss_trajectory = attack_result.get("loss_trajectory", [])

                t0 = time.perf_counter()
                adv_pred_tokens = wrapper.predict_prepared_staff(x_adv)
                adv_predict_sec = time.perf_counter() - t0

                adv_metrics = compute_metrics(adv_pred_tokens, target_tokens)

                linf = float(np.max(np.abs(x_adv.astype(np.float32) - clean_image.astype(np.float32))))
                attack_queries_per_sec = n_queries / attack_sec if attack_sec > 0 else 0.0

                adv_path = save_adv_image_if_requested(
                    args.save_adv_dir,
                    entry,
                    float(epsilon),
                    x_adv,
                )

                job_elapsed_sec = time.perf_counter() - job_start

                record = {
                    "timestamp": now_iso(),
                    "status": "ok",
                    "staff_counter": staff_counter,
                    "staff_path": str(entry.staff_path),
                    "score_stem": entry.score_stem,
                    "staff_filename": entry.staff_filename,
                    "staff_index": entry.staff_index,
                    "epsilon": float(epsilon),
                    "n_max": int(args.n_max),
                    "n_queries": n_queries,
                    "target_source": args.target_source,

                    "target_token_count": len(target_tokens),
                    "clean_pred_token_count": len(clean_pred_tokens),
                    "adv_pred_token_count": len(adv_pred_tokens),

                    "clean_ser": clean_metrics["ser"],
                    "clean_cer": clean_metrics["cer"],
                    "clean_ser_capped": clean_metrics["ser_capped"],
                    "clean_cer_capped": clean_metrics["cer_capped"],

                    "adv_ser": adv_metrics["ser"],
                    "adv_cer": adv_metrics["cer"],
                    "adv_ser_capped": adv_metrics["ser_capped"],
                    "adv_cer_capped": adv_metrics["cer_capped"],

                    "ser_delta": adv_metrics["ser"] - clean_metrics["ser"],
                    "cer_delta": adv_metrics["cer"] - clean_metrics["cer"],
                    "ser_capped_delta": adv_metrics["ser_capped"] - clean_metrics["ser_capped"],
                    "cer_capped_delta": adv_metrics["cer_capped"] - clean_metrics["cer_capped"],

                    "success": bool(adv_metrics["ser"] > clean_metrics["ser"]),
                    "L_best": l_best,
                    "loss_trajectory_len": len(loss_trajectory)
                    if isinstance(loss_trajectory, list)
                    else None,

                    "linf": linf,
                    "linf_ok": bool(linf <= float(epsilon) + 1e-5),

                    "load_staff_sec": load_staff_sec,
                    "clean_predict_sec": clean_predict_sec,
                    "target_load_sec": target_load_sec,
                    "attack_sec": attack_sec,
                    "adv_predict_sec": adv_predict_sec,
                    "job_elapsed_sec": job_elapsed_sec,
                    "attack_queries_per_sec": attack_queries_per_sec,

                    "adv_path": adv_path,
                    "checkpoint_key": {
                        "staff_path": Path(entry.staff_path).as_posix(),
                        "epsilon": f"{float(epsilon):.10g}",
                        "n_max": int(args.n_max),
                        "target_source": args.target_source,
                    },
                }

                job_records.append(record)
                append_jsonl(args.timing_jsonl, record)
                resumed_records[key] = record

                if args.profile_timing:
                    print(
                        f"[timing] staff={staff_counter}/{len(staff_entries)} "
                        f"epsilon={float(epsilon):.4f} | "
                        f"attack={attack_sec:.3f}s | "
                        f"adv_predict={adv_predict_sec:.3f}s | "
                        f"job={job_elapsed_sec:.3f}s | "
                        f"queries={n_queries} | "
                        f"q/s={attack_queries_per_sec:.2f} | "
                        f"clean_ser={clean_metrics['ser']:.6f} | "
                        f"adv_ser={adv_metrics['ser']:.6f} | "
                        f"adv_ser_capped={adv_metrics['ser_capped']:.6f} | "
                        f"linf={linf:.8f}"
                    )

                snapshot = progress.update(queries=n_queries)

                if args.progress_every > 0:
                    if snapshot.completed_jobs % args.progress_every == 0:
                        print(ProgressMeter.line(snapshot))

            except Exception as exc:
                job_elapsed_sec = time.perf_counter() - job_start

                error_record = {
                    "timestamp": now_iso(),
                    "status": "error",
                    "phase": "epsilon_job",
                    "staff_counter": staff_counter,
                    "staff_path": str(entry.staff_path),
                    "score_stem": entry.score_stem,
                    "staff_filename": entry.staff_filename,
                    "epsilon": float(epsilon),
                    "n_max": int(args.n_max),
                    "target_source": args.target_source,
                    "job_elapsed_sec": job_elapsed_sec,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }

                job_records.append(error_record)
                append_jsonl(args.timing_jsonl, error_record)

                print(
                    f"[error] Failed job staff={entry.staff_path} epsilon={epsilon}: {exc}",
                    file=sys.stderr,
                )

                snapshot = progress.update(queries=0)

                if args.progress_every > 0:
                    if snapshot.completed_jobs % args.progress_every == 0:
                        print(ProgressMeter.line(snapshot))

                if not args.continue_on_error:
                    raise

        staff_outer_sec = time.perf_counter() - staff_outer_start
        if args.profile_timing:
            print(
                f"[timing] completed staff={staff_counter}/{len(staff_entries)} "
                f"in {format_duration(staff_outer_sec)}"
            )

    run_elapsed_sec = time.perf_counter() - run_start

    write_summary_json(
        output_path=output_json,
        args=args,
        staff_entries=staff_entries,
        job_records=job_records,
        run_start_iso=run_start_iso,
        run_elapsed_sec=run_elapsed_sec,
        resumed_records_count=resumed_records_count,
    )

    try_generate_plots(args=args, output_json=output_json)

    ok_count = sum(1 for r in job_records if r.get("status") == "ok")
    err_count = len(job_records) - ok_count

    print(
        f"[done] jobs_ok={ok_count}, jobs_error={err_count}, "
        f"resumed={resumed_records_count}, "
        f"elapsed={format_duration(run_elapsed_sec)}"
    )

    if args.timing_jsonl:
        print(f"[done] timing/checkpoint JSONL: {args.timing_jsonl}")

    if args.plot:
        print(f"[done] plots dir: {args.plots_dir}")

    return 0 if err_count == 0 else 2


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    return run_sweep(args)


if __name__ == "__main__":
    raise SystemExit(main())