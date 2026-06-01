"""
Plot HOMR adversarial benchmark results.

Supports both:
- Track A spectral-noise metrics JSON produced by attacks/run_spectral_sweep.py
- Track B Square Attack metrics JSON produced by attacks/run_square_sweep.py

Default plot set is `core`, where delta SER/CER are the primary scientific
curves. Use `--plot-set full` for raw/capped diagnostic plots as well.

This file is safe to keep in either:
    attacks/python scripts/plot_results.py
or:
    attacks/plot_results.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _load_payload(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[metadata] wrote {path}")


def _write_table(df: pd.DataFrame, output_path: Path, label: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[table] wrote {output_path} ({label}, rows={len(df)})")


def _ensure_columns(df: pd.DataFrame, required: list[str], source: Path | str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {source}: {missing}")


def _coerce_numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _add_if_missing(df: pd.DataFrame, column: str, value: Any) -> None:
    if column not in df.columns:
        df[column] = value


def _safe_label(value: Any) -> str:
    if pd.isna(value):
        return "NA"
    try:
        return f"{float(value):g}"
    except Exception:
        return str(value)


def save_line_plot(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    title: str,
    y_label: str,
    output_path: Path,
    x_label: str,
    log_x: bool = False,
    color: str | None = None,
    line_dash: str | None = None,
) -> Path:
    labels = {x: x_label, y: y_label}
    if color is not None:
        labels[color] = color
    if line_dash is not None:
        labels[line_dash] = line_dash

    fig = px.line(
        df,
        x=x,
        y=y,
        color=color,
        line_dash=line_dash,
        markers=True,
        title=title,
        labels=labels,
    )

    if log_x:
        numeric_x = pd.to_numeric(df[x], errors="coerce")
        if numeric_x.notna().all() and (numeric_x > 0).all():
            fig.update_xaxes(type="log")

    fig.update_layout(template="plotly_white", hovermode="x unified")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path)
    print(f"[plot] wrote {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Track B / Square Attack
# ---------------------------------------------------------------------------

def load_square_results(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    payload = _load_payload(path)
    metadata = payload.get("experiment_metadata", {})
    rows = payload.get("aggregate_results") or payload.get("square_attack_results", [])
    if not rows:
        raise ValueError(f"No square_attack_results or aggregate_results found in {path}")

    df = pd.DataFrame(rows)
    if "epsilon_math" not in df.columns and "epsilon" in df.columns:
        df["epsilon_math"] = df["epsilon"]
    if "epsilon" not in df.columns and "epsilon_math" in df.columns:
        df["epsilon"] = df["epsilon_math"]

    df = df.sort_values("epsilon_math").reset_index(drop=True)

    # Backward compatibility and computed fields.
    if "mean_ser_capped" not in df.columns and "mean_ser" in df.columns:
        df["mean_ser_capped"] = df["mean_ser"].clip(upper=1.0)
    if "mean_cer_capped" not in df.columns and "mean_cer" in df.columns:
        df["mean_cer_capped"] = df["mean_cer"].clip(upper=1.0)
    for col in ("std_ser_capped", "std_cer_capped"):
        _add_if_missing(df, col, pd.NA)

    _add_if_missing(df, "mean_clean_ser", 0.0)
    _add_if_missing(df, "mean_clean_cer", 0.0)
    _add_if_missing(df, "mean_clean_ser_capped", 0.0)
    _add_if_missing(df, "mean_clean_cer_capped", 0.0)

    if "mean_delta_ser" not in df.columns:
        df["mean_delta_ser"] = df["mean_ser"] - df["mean_clean_ser"]
    if "mean_delta_cer" not in df.columns:
        df["mean_delta_cer"] = df["mean_cer"] - df["mean_clean_cer"]
    if "mean_delta_ser_capped" not in df.columns:
        df["mean_delta_ser_capped"] = df["mean_ser_capped"] - df["mean_clean_ser_capped"]
    if "mean_delta_cer_capped" not in df.columns:
        df["mean_delta_cer_capped"] = df["mean_cer_capped"] - df["mean_clean_cer_capped"]

    for fallback_col in (
        "success_rate",
        "avg_queries",
        "mean_job_elapsed_sec",
        "mean_attack_queries_per_sec",
        "jobs_per_min",
    ):
        _add_if_missing(df, fallback_col, pd.NA)

    required = [
        "epsilon_math",
        "mean_ser",
        "std_ser",
        "mean_cer",
        "std_cer",
        "mean_delta_ser",
        "mean_delta_cer",
    ]
    _ensure_columns(df, required, path)

    numeric_cols = [
        "epsilon",
        "epsilon_math",
        "n_staffs",
        "mean_ser",
        "std_ser",
        "mean_cer",
        "std_cer",
        "mean_ser_capped",
        "std_ser_capped",
        "mean_cer_capped",
        "std_cer_capped",
        "mean_clean_ser",
        "mean_clean_cer",
        "mean_delta_ser",
        "mean_delta_cer",
        "mean_delta_ser_capped",
        "mean_delta_cer_capped",
        "success_rate",
        "avg_queries",
        "mean_job_elapsed_sec",
        "mean_attack_queries_per_sec",
        "jobs_per_min",
    ]
    df = _coerce_numeric_columns(df, numeric_cols)
    return df, metadata


def write_square_jobs_table(square_json: Path, output_dir: Path) -> Path | None:
    payload = _load_payload(square_json)
    jobs = payload.get("jobs", [])
    if not jobs:
        print("[table] no square jobs found; skipping square_jobs_table.csv")
        return None
    df_jobs = pd.DataFrame(jobs)
    output_path = output_dir / "square_jobs_table.csv"
    _write_table(df_jobs, output_path, "Track B jobs")
    return output_path


def plot_square_results(square_json: Path, output_dir: Path, plot_set: str = "core") -> Path:
    df, metadata = load_square_results(square_json)
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_table(df, output_dir / "square_results_table.csv", "Track B aggregate")
    write_square_jobs_table(square_json, output_dir)

    outputs: dict[str, str] = {"table": "square_results_table.csv"}

    core_specs = [
        ("mean_delta_ser", "Track B Square Attack: ΔSER vs ε", "Mean ΔSER", "square_delta_ser_vs_epsilon.html"),
        ("mean_delta_cer", "Track B Square Attack: ΔCER vs ε", "Mean ΔCER", "square_delta_cer_vs_epsilon.html"),
        ("success_rate", "Track B Square Attack: Success Rate vs ε", "Success rate", "square_success_vs_epsilon.html"),
        ("mean_job_elapsed_sec", "Track B Square Attack: Runtime vs ε", "Mean job time, seconds", "square_runtime_vs_epsilon.html"),
    ]
    full_extra_specs = [
        ("mean_ser", "Track B Square Attack: Raw SER vs ε", "Mean raw SER", "square_ser_vs_epsilon.html"),
        ("mean_cer", "Track B Square Attack: Raw CER vs ε", "Mean raw CER", "square_cer_vs_epsilon.html"),
        ("mean_ser_capped", "Track B Square Attack: Capped SER vs ε", "Mean capped SER", "square_ser_capped_vs_epsilon.html"),
        ("mean_cer_capped", "Track B Square Attack: Capped CER vs ε", "Mean capped CER", "square_cer_capped_vs_epsilon.html"),
        ("mean_attack_queries_per_sec", "Track B Square Attack: Query Speed vs ε", "Mean attack queries per second", "square_query_speed_vs_epsilon.html"),
        ("avg_queries", "Track B Square Attack: Average Queries vs ε", "Average queries", "square_avg_queries_vs_epsilon.html"),
    ]
    specs = core_specs + (full_extra_specs if plot_set == "full" else [])

    for y, title, label, filename in specs:
        if y not in df.columns or not df[y].notna().any():
            continue
        outputs[y] = save_line_plot(
            df,
            x="epsilon_math",
            y=y,
            title=title,
            y_label=label,
            x_label="ε_math",
            output_path=output_dir / filename,
            log_x=True,
        ).name

    summary_path = output_dir / "square_plot_metadata.json"
    _write_json(
        summary_path,
        {
            "source_json": str(square_json),
            "metadata": metadata,
            "plot_set": plot_set,
            "n_rows": int(len(df)),
            "epsilons": [float(x) for x in df["epsilon_math"].dropna().tolist()],
            "outputs": outputs,
            "metric_note": (
                "Core plots use delta SER/CER: perturbed/adversarial error minus clean baseline error. "
                "Raw and capped metrics remain in CSV tables and are plotted only with --plot-set full."
            ),
        },
    )
    return summary_path


# ---------------------------------------------------------------------------
# Track A / Spectral Noise
# ---------------------------------------------------------------------------

def _load_spectral_results_from_payload(
    payload: dict[str, Any],
    source: Path | str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    metadata = payload.get("experiment_metadata", {})
    rows = payload.get("aggregate_results") or payload.get("spectral_noise_results", [])
    if not rows:
        raise ValueError(f"No aggregate_results or spectral_noise_results found in {source}")

    df = pd.DataFrame(rows)
    sort_cols = [col for col in ("alpha", "epsilon_phys") if col in df.columns]
    df = df.sort_values(sort_cols or ["epsilon_phys"]).reset_index(drop=True)

    if "epsilon_phys" not in df.columns and "epsilon" in df.columns:
        df["epsilon_phys"] = df["epsilon"]
    if "epsilon" not in df.columns and "epsilon_phys" in df.columns:
        df["epsilon"] = df["epsilon_phys"]

    if "mean_ser_capped" not in df.columns and "mean_ser" in df.columns:
        df["mean_ser_capped"] = df["mean_ser"].clip(upper=1.0)
    if "mean_cer_capped" not in df.columns and "mean_cer" in df.columns:
        df["mean_cer_capped"] = df["mean_cer"].clip(upper=1.0)
    for col in ("std_ser_capped", "std_cer_capped"):
        _add_if_missing(df, col, pd.NA)

    _add_if_missing(df, "mean_clean_ser", 0.0)
    _add_if_missing(df, "mean_clean_cer", 0.0)
    _add_if_missing(df, "mean_clean_ser_capped", 0.0)
    _add_if_missing(df, "mean_clean_cer_capped", 0.0)

    if "mean_delta_ser" not in df.columns and "mean_ser" in df.columns:
        df["mean_delta_ser"] = df["mean_ser"] - df["mean_clean_ser"]
    if "mean_delta_cer" not in df.columns and "mean_cer" in df.columns:
        df["mean_delta_cer"] = df["mean_cer"] - df["mean_clean_cer"]
    if "mean_delta_ser_capped" not in df.columns:
        df["mean_delta_ser_capped"] = df["mean_ser_capped"] - df["mean_clean_ser_capped"]
    if "mean_delta_cer_capped" not in df.columns:
        df["mean_delta_cer_capped"] = df["mean_cer_capped"] - df["mean_clean_cer_capped"]

    if "layout_success_rate" not in df.columns:
        if {"n_images_ok", "n_images_attempted"}.issubset(df.columns):
            denom = pd.to_numeric(df["n_images_attempted"], errors="coerce").replace(0, pd.NA)
            df["layout_success_rate"] = pd.to_numeric(df["n_images_ok"], errors="coerce") / denom
        else:
            df["layout_success_rate"] = pd.NA

    if "layout_match_rate" not in df.columns:
        if {"n_layout_mismatch_images", "n_images_ok"}.issubset(df.columns):
            denom = pd.to_numeric(df["n_images_ok"], errors="coerce").replace(0, pd.NA)
            df["layout_match_rate"] = 1.0 - (pd.to_numeric(df["n_layout_mismatch_images"], errors="coerce") / denom)
        else:
            df["layout_match_rate"] = pd.NA

    _ensure_columns(
        df,
        [
            "epsilon_phys",
            "mean_ser",
            "std_ser",
            "mean_cer",
            "std_cer",
            "mean_delta_ser",
            "mean_delta_cer",
            "n_staff_pairs",
            "n_images_attempted",
            "n_images_ok",
            "elapsed_seconds",
        ],
        source,
    )

    numeric_cols = [
        "epsilon",
        "epsilon_phys",
        "alpha",
        "n_images_attempted",
        "n_images_ok",
        "n_images_failed",
        "n_staff_pairs",
        "n_staff_pairs_expected",
        "n_layout_mismatch_images",
        "layout_success_rate",
        "layout_match_rate",
        "mean_staffs_clean",
        "mean_staffs_noisy",
        "mean_ser",
        "std_ser",
        "mean_cer",
        "std_cer",
        "mean_ser_capped",
        "std_ser_capped",
        "mean_cer_capped",
        "std_cer_capped",
        "mean_clean_ser",
        "mean_clean_cer",
        "mean_delta_ser",
        "std_delta_ser",
        "mean_delta_cer",
        "std_delta_cer",
        "mean_delta_ser_capped",
        "mean_delta_cer_capped",
        "condition_elapsed_sec",
        "elapsed_seconds",
        "mean_job_elapsed_sec",
        "jobs_per_min",
        "images_per_second",
        "staffs_per_second",
        "total_queries",
        "mean_noise_weight_white",
        "mean_noise_weight_pink",
        "mean_noise_weight_brown",
    ]
    df = _coerce_numeric_columns(df, numeric_cols)

    staff_df = pd.DataFrame(payload.get("staff_level_results", []))
    image_df = pd.DataFrame(payload.get("image_level_results", []))
    jobs_df = pd.DataFrame(payload.get("jobs", payload.get("image_level_results", [])))
    error_df = pd.DataFrame(payload.get("errors", []))
    return df, staff_df, image_df, jobs_df, error_df, metadata


def load_spectral_results(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    return _load_spectral_results_from_payload(_load_payload(path), source=path)


def plot_spectral_results_from_payload(
    payload: dict[str, Any],
    output_dir: Path,
    source_json: Path | None = None,
    plot_set: str = "core",
) -> Path:
    df, staff_df, image_df, jobs_df, error_df, metadata = _load_spectral_results_from_payload(
        payload,
        source=source_json or "<in-memory-payload>",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_table(df, output_dir / "spectral_results_table.csv", "Track A aggregate")
    if not staff_df.empty:
        _write_table(staff_df, output_dir / "spectral_staff_results_table.csv", "Track A staff rows")
    if not image_df.empty:
        _write_table(image_df, output_dir / "spectral_image_results_table.csv", "Track A image rows")
    if not jobs_df.empty:
        _write_table(jobs_df, output_dir / "spectral_jobs_table.csv", "Track A jobs")
    if not error_df.empty:
        _write_table(error_df, output_dir / "spectral_errors_table.csv", "Track A errors")

    outputs: dict[str, str] = {"table": "spectral_results_table.csv"}
    if not staff_df.empty:
        outputs["staff_table"] = "spectral_staff_results_table.csv"
    if not image_df.empty:
        outputs["image_table"] = "spectral_image_results_table.csv"
    if not jobs_df.empty:
        outputs["jobs_table"] = "spectral_jobs_table.csv"
    if not error_df.empty:
        outputs["errors_table"] = "spectral_errors_table.csv"

    color_col = "alpha" if "alpha" in df.columns and df["alpha"].nunique(dropna=True) > 1 else None

    core_specs = [
        ("mean_delta_ser", "Track A Spectral Noise: ΔSER vs ε", "Mean ΔSER", "spectral_delta_ser_vs_epsilon.html"),
        ("mean_delta_cer", "Track A Spectral Noise: ΔCER vs ε", "Mean ΔCER", "spectral_delta_cer_vs_epsilon.html"),
        ("layout_success_rate", "Track A Spectral Noise: Layout Success vs ε", "Layout success rate", "spectral_layout_success_vs_epsilon.html"),
        ("elapsed_seconds", "Track A Spectral Noise: Runtime vs ε", "Elapsed seconds", "spectral_runtime_vs_epsilon.html"),
    ]
    full_extra_specs = [
        ("mean_ser", "Track A Spectral Noise: Raw SER vs ε", "Mean raw SER", "spectral_ser_vs_epsilon.html"),
        ("mean_cer", "Track A Spectral Noise: Raw CER vs ε", "Mean raw CER", "spectral_cer_vs_epsilon.html"),
        ("mean_ser_capped", "Track A Spectral Noise: Capped SER vs ε", "Mean capped SER", "spectral_ser_capped_vs_epsilon.html"),
        ("mean_cer_capped", "Track A Spectral Noise: Capped CER vs ε", "Mean capped CER", "spectral_cer_capped_vs_epsilon.html"),
        ("layout_match_rate", "Track A Spectral Noise: Staff Count Match vs ε", "Staff count match rate", "spectral_layout_match_vs_epsilon.html"),
        ("mean_staffs_noisy", "Track A Spectral Noise: Mean Detected Staffs vs ε", "Mean detected staffs", "spectral_staff_count_vs_epsilon.html"),
        ("staffs_per_second", "Track A Spectral Noise: Staff Throughput vs ε", "Staffs per second", "spectral_staffs_per_second_vs_epsilon.html"),
    ]
    specs = core_specs + (full_extra_specs if plot_set == "full" else [])

    for y, title, y_label, filename in specs:
        if y not in df.columns or not df[y].notna().any():
            continue
        outputs[y] = save_line_plot(
            df,
            x="epsilon_phys",
            y=y,
            title=title,
            y_label=y_label,
            x_label="ε_phys",
            output_path=output_dir / filename,
            log_x=False,
            color=color_col,
        ).name

    summary_path = output_dir / "spectral_plot_metadata.json"
    _write_json(
        summary_path,
        {
            "source_json": str(source_json) if source_json is not None else None,
            "metadata": metadata,
            "plot_set": plot_set,
            "n_rows": int(len(df)),
            "epsilons": [float(x) for x in df["epsilon_phys"].dropna().tolist()],
            "outputs": outputs,
            "metric_note": (
                "Core plots use delta SER/CER: noisy error minus clean baseline error. "
                "Raw and capped metrics remain in CSV tables and are plotted only with --plot-set full."
            ),
            "target_note": (
                "If target_source is clean_onnx_prediction, Track A measures degradation drift "
                "relative to the clean ONNX output, not final MusicXML ground-truth accuracy."
            ),
        },
    )
    return summary_path


def plot_spectral_results(spectral_json: Path, output_dir: Path, plot_set: str = "core") -> Path:
    return plot_spectral_results_from_payload(
        payload=_load_payload(spectral_json),
        source_json=spectral_json,
        output_dir=output_dir,
        plot_set=plot_set,
    )


# ---------------------------------------------------------------------------
# Combined Track A + Track B plots
# ---------------------------------------------------------------------------

def _severity_index(series: pd.Series, max_value: float | None = None) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    denom = float(max_value) if max_value is not None and max_value > 0 else float(x.max(skipna=True) or 1.0)
    if denom <= 0:
        denom = 1.0
    return x / denom


def build_combined_results(
    spectral_json: Path,
    square_json: Path,
    *,
    spectral_max_epsilon: float | None = None,
    square_max_epsilon: float | None = None,
    average_track_a_alpha: bool = True,
) -> pd.DataFrame:
    spectral_df, *_ = load_spectral_results(spectral_json)
    square_df, _ = load_square_results(square_json)

    a = spectral_df.copy()
    a["track"] = "Track A spectral"
    a["epsilon_raw"] = a["epsilon_phys"]
    a["severity_index"] = _severity_index(a["epsilon_phys"], spectral_max_epsilon)
    if "alpha" not in a.columns:
        a["alpha"] = pd.NA

    if average_track_a_alpha and "alpha" in a.columns and a["alpha"].nunique(dropna=True) > 1:
        group_cols = ["severity_index", "epsilon_raw"]
        numeric_cols = [
            "mean_delta_ser",
            "mean_delta_cer",
            "mean_ser",
            "mean_cer",
            "layout_success_rate",
        ]
        present = [c for c in numeric_cols if c in a.columns]
        a = a.groupby(group_cols, as_index=False)[present].mean(numeric_only=True)
        a["track"] = "Track A spectral (mean over α)"
        a["alpha"] = "mean"

    b = square_df.copy()
    b["track"] = "Track B square"
    b["epsilon_raw"] = b["epsilon_math"]
    b["severity_index"] = _severity_index(b["epsilon_math"], square_max_epsilon)
    b["alpha"] = pd.NA

    keep_cols = [
        "track",
        "severity_index",
        "epsilon_raw",
        "alpha",
        "mean_delta_ser",
        "mean_delta_cer",
        "mean_ser",
        "mean_cer",
    ]
    for frame in (a, b):
        for col in keep_cols:
            if col not in frame.columns:
                frame[col] = pd.NA

    combined = pd.concat([a[keep_cols], b[keep_cols]], ignore_index=True)
    combined = _coerce_numeric_columns(
        combined,
        ["severity_index", "epsilon_raw", "mean_delta_ser", "mean_delta_cer", "mean_ser", "mean_cer"],
    )
    return combined.sort_values(["track", "severity_index"]).reset_index(drop=True)



def _curve_auc(df: pd.DataFrame, y: str) -> dict[str, float]:
    """Compute trapezoidal AUC by track on normalized severity."""
    output: dict[str, float] = {}
    for track, group in df.dropna(subset=["severity_index", y]).groupby("track"):
        g = group.sort_values("severity_index")
        x = g["severity_index"].astype(float).to_numpy()
        vals = g[y].astype(float).to_numpy()
        if len(x) >= 2:
            output[str(track)] = float(np.trapz(vals, x))
        elif len(x) == 1:
            output[str(track)] = float(vals[0])
    return output


def _paired_curve_difference_stats(df: pd.DataFrame, y: str, n_grid: int = 101) -> dict[str, Any]:
    """
    Descriptive curve-level comparison between Track A and Track B.

    This interpolates both curves onto a shared normalized-severity grid and
    reports B - A differences. The sign-flip p-value is a curve diagnostic, not a
    substitute for an independent sample-level test.
    """
    tracks = [str(x) for x in df["track"].dropna().unique().tolist()]
    a_track = next((t for t in tracks if t.startswith("Track A")), None)
    b_track = next((t for t in tracks if t.startswith("Track B")), None)

    if a_track is None or b_track is None:
        return {"metric": y, "available": False, "reason": "Need one Track A and one Track B curve."}

    a = df[(df["track"] == a_track) & df["severity_index"].notna() & df[y].notna()].sort_values("severity_index")
    b = df[(df["track"] == b_track) & df["severity_index"].notna() & df[y].notna()].sort_values("severity_index")

    if len(a) < 2 or len(b) < 2:
        return {"metric": y, "available": False, "reason": "Need at least two points per curve."}

    a_x = a["severity_index"].astype(float).to_numpy()
    a_y = a[y].astype(float).to_numpy()
    b_x = b["severity_index"].astype(float).to_numpy()
    b_y = b[y].astype(float).to_numpy()

    lo = max(float(np.min(a_x)), float(np.min(b_x)))
    hi = min(float(np.max(a_x)), float(np.max(b_x)))
    if hi <= lo:
        return {"metric": y, "available": False, "reason": "Track curves do not overlap in normalized severity."}

    grid = np.linspace(lo, hi, int(n_grid))
    a_interp = np.interp(grid, a_x, a_y)
    b_interp = np.interp(grid, b_x, b_y)
    diff = b_interp - a_interp

    # Deterministic sign-flip diagnostic for whether mean curve difference is
    # far from zero. Points are interpolated curve samples, not independent raw
    # images/staffs, so label this as descriptive.
    rng = np.random.default_rng(12345)
    observed = float(np.mean(diff))
    n_perm = 10000
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, len(diff)))
    perm_means = np.mean(signs * diff[np.newaxis, :], axis=1)
    p_value = float((np.sum(np.abs(perm_means) >= abs(observed)) + 1) / (n_perm + 1))

    return {
        "metric": y,
        "available": True,
        "track_a": a_track,
        "track_b": b_track,
        "severity_min": lo,
        "severity_max": hi,
        "n_grid": int(n_grid),
        "mean_track_b_minus_track_a": observed,
        "median_track_b_minus_track_a": float(np.median(diff)),
        "max_abs_track_b_minus_track_a": float(np.max(np.abs(diff))),
        "auc_track_a": float(np.trapz(a_interp, grid)),
        "auc_track_b": float(np.trapz(b_interp, grid)),
        "auc_track_b_minus_track_a": float(np.trapz(b_interp, grid) - np.trapz(a_interp, grid)),
        "sign_flip_p_value_descriptive": p_value,
        "note": (
            "This is a descriptive curve-level sign-flip diagnostic over interpolated severity points. "
            "For formal sample-level inference, use paired image-level rows from the same image subset."
        ),
    }


def compute_combined_statistics(combined_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute descriptive combined Track A/B statistics for delta SER/CER."""
    rows: list[dict[str, Any]] = []
    payload: dict[str, Any] = {
        "statistics_note": (
            "These statistics compare curves on normalized severity. They are descriptive unless the "
            "underlying runs used the same source-image subset and a sample-level paired test is performed."
        )
    }

    for metric in ("mean_delta_ser", "mean_delta_cer"):
        auc = _curve_auc(combined_df, metric)
        diff_stats = _paired_curve_difference_stats(combined_df, metric)
        row = {"metric": metric, **{f"auc_{k}": v for k, v in auc.items()}, **diff_stats}
        rows.append(row)
        payload[metric] = row

    return pd.DataFrame(rows), payload


def plot_combined_results(
    spectral_json: Path,
    square_json: Path,
    output_dir: Path,
    *,
    spectral_max_epsilon: float | None = None,
    square_max_epsilon: float | None = None,
    plot_set: str = "core",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_core = build_combined_results(
        spectral_json,
        square_json,
        spectral_max_epsilon=spectral_max_epsilon,
        square_max_epsilon=square_max_epsilon,
        average_track_a_alpha=True,
    )
    _write_table(combined_core, output_dir / "combined_results_table.csv", "Combined Track A/B")

    stats_df, stats_payload = compute_combined_statistics(combined_core)
    _write_table(stats_df, output_dir / "combined_statistics_table.csv", "Combined Track A/B statistics")
    _write_json(output_dir / "combined_statistics.json", stats_payload)

    outputs: dict[str, str] = {
        "table": "combined_results_table.csv",
        "statistics_table": "combined_statistics_table.csv",
        "statistics_json": "combined_statistics.json",
    }
    for y, title, label, filename in [
        ("mean_delta_ser", "Track A vs Track B: ΔSER vs Normalized Severity", "Mean ΔSER", "combined_delta_ser_vs_normalized_severity.html"),
        ("mean_delta_cer", "Track A vs Track B: ΔCER vs Normalized Severity", "Mean ΔCER", "combined_delta_cer_vs_normalized_severity.html"),
    ]:
        if y in combined_core.columns and combined_core[y].notna().any():
            outputs[y] = save_line_plot(
                combined_core,
                x="severity_index",
                y=y,
                title=title,
                y_label=label,
                x_label="Normalized severity ε / max(ε)",
                output_path=output_dir / filename,
                log_x=False,
                color="track",
            ).name

    if plot_set == "full":
        combined_full = build_combined_results(
            spectral_json,
            square_json,
            spectral_max_epsilon=spectral_max_epsilon,
            square_max_epsilon=square_max_epsilon,
            average_track_a_alpha=False,
        )
        combined_full["series"] = combined_full.apply(
            lambda r: f"Track A α={_safe_label(r['alpha'])}" if str(r["track"]).startswith("Track A") else str(r["track"]),
            axis=1,
        )
        _write_table(combined_full, output_dir / "combined_results_by_alpha_table.csv", "Combined Track A/B by alpha")
        outputs["by_alpha_table"] = "combined_results_by_alpha_table.csv"
        for y, title, label, filename in [
            ("mean_delta_ser", "Track A by α vs Track B: ΔSER vs Normalized Severity", "Mean ΔSER", "combined_delta_ser_by_alpha_vs_normalized_severity.html"),
            ("mean_delta_cer", "Track A by α vs Track B: ΔCER vs Normalized Severity", "Mean ΔCER", "combined_delta_cer_by_alpha_vs_normalized_severity.html"),
        ]:
            if y in combined_full.columns and combined_full[y].notna().any():
                outputs[y + "_by_alpha"] = save_line_plot(
                    combined_full,
                    x="severity_index",
                    y=y,
                    title=title,
                    y_label=label,
                    x_label="Normalized severity ε / max(ε)",
                    output_path=output_dir / filename,
                    log_x=False,
                    color="series",
                ).name

    summary_path = output_dir / "combined_plot_metadata.json"
    _write_json(
        summary_path,
        {
            "spectral_json": str(spectral_json),
            "square_json": str(square_json),
            "plot_set": plot_set,
            "spectral_max_epsilon": spectral_max_epsilon,
            "square_max_epsilon": square_max_epsilon,
            "outputs": outputs,
            "normalization_note": (
                "Normalized severity is epsilon divided by the maximum epsilon for that track. "
                "This is dimensionless and compares relative stress within each track; it does not claim physical equivalence between epsilon_phys and epsilon_math."
            ),
            "metric_note": "Combined plots use delta SER/CER as the primary cross-track metric.",
        },
    )
    return summary_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot HOMR adversarial benchmark result JSON files.")
    parser.add_argument("--square-json", type=str, default=None, help="Path to Track B square sweep metrics JSON.")
    parser.add_argument("--spectral-json", type=str, default=None, help="Path to Track A spectral sweep metrics JSON.")
    parser.add_argument("--output-dir", type=str, default="results/plots", help="Output directory for HTML plots, CSV tables, and plot metadata.")
    parser.add_argument("--plot-set", choices=["core", "full"], default="core", help="core = delta-first primary plots; full = include raw/capped/debug plots.")
    parser.add_argument("--combined", action="store_true", help="When both --spectral-json and --square-json are supplied, also generate combined normalized-severity plots.")
    parser.add_argument("--spectral-max-epsilon", type=float, default=None, help="Denominator for Track A normalized severity. Defaults to max epsilon_phys in the supplied JSON.")
    parser.add_argument("--square-max-epsilon", type=float, default=None, help="Denominator for Track B normalized severity. Defaults to max epsilon_math in the supplied JSON.")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.square_json is None and args.spectral_json is None:
        parser.error("Provide at least one of --square-json or --spectral-json.")

    output_dir = Path(args.output_dir)

    if args.square_json is not None:
        plot_square_results(Path(args.square_json), output_dir, plot_set=args.plot_set)

    if args.spectral_json is not None:
        plot_spectral_results(Path(args.spectral_json), output_dir, plot_set=args.plot_set)

    if args.combined:
        if args.square_json is None or args.spectral_json is None:
            parser.error("--combined requires both --spectral-json and --square-json.")
        plot_combined_results(
            Path(args.spectral_json),
            Path(args.square_json),
            output_dir,
            spectral_max_epsilon=args.spectral_max_epsilon,
            square_max_epsilon=args.square_max_epsilon,
            plot_set=args.plot_set,
        )

    print("[done] Plot generation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
