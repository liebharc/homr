"""
Track A spectral-noise sweep runner for the adversarial HOMR benchmark.

This script evaluates natural out-of-distribution degradation on full-page sheet
music images. It injects colored spectral noise into each full-page image in
memory, runs the ONNX SegNet + deterministic HOMR layout pipeline to recover
prepared staff images, runs ONNX TrOMR recognition on each prepared staff, and
writes aggregate SER/CER metrics plus optional plots.

Benchmark boundary
------------------
Neural inference is restricted to ONNX Runtime:
    - SegNet: dataset.cache_prepared_staffs.SegNetONNX
    - TrOMR encoder/decoder: attacks.src.homr_wrapper.HomrWrapper

HOMR Python is used only for deterministic layout/preprocessing through the
helpers in dataset/cache_prepared_staffs.py. This script does not call
Staff2Score.predict(...), parse_staff_tromr(...), PyTorch, or TensorFlow.

Target-source caveat
--------------------
The current cached staff metadata produced by dataset/cache_prepared_staffs.py
stores empty gt_tokens. Until validated MusicXML/MXL token alignment is added,
the useful Track A mode is self-consistency drift:

    target_source = clean_onnx_prediction

That means each noisy prepared staff prediction is compared against the clean
ONNX prediction for the same image/staff index. These results measure robustness
under degradation, but they are not final MusicXML ground-truth accuracy.

Examples
--------
Help check:
    python attacks/run_spectral_sweep.py --help

Tiny smoke run:
    python attacks/run_spectral_sweep.py --max-images 1 --epsilon 0.0 0.05 \
        --alpha 1.0 --output-json results/logs/TRACK_A_SMOKE_metrics.json \
        --plot --plots-dir results/plots
"""

from __future__ import annotations

import argparse
import json
import importlib.util
import math
import sys
import tempfile
import time
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from attacks.src.homr_wrapper import HomrWrapper, symbols_to_strings
from attacks.src.spectral_noise import (
    apply_endpoint_interpolation,
    find_image_paths,
    generate_multiscale_colored_noise,
    save_float_image,
)
from attacks.src.statistics_engine import batch_metrics, character_error_rate, symbol_error_rate


# IMPORTANT: dataset.cache_prepared_staffs imports HOMR staff parsing helpers. In
# this HOMR version, those deterministic helper imports also load modules named
# staff2score / staff_parsing_tromr as import side effects. HomrWrapper performs
# its ONNX-boundary guard during __init__, so the wrapper must be constructed
# before importing the layout/cache helper module.
_LAYOUT_HELPERS: dict[str, Any] | None = None


def _load_layout_helpers() -> dict[str, Any]:
    global _LAYOUT_HELPERS

    if _LAYOUT_HELPERS is None:
        from dataset import cache_prepared_staffs as cps

        _LAYOUT_HELPERS = {
            "SegNetONNX": cps.SegNetONNX,
            "StaffRegions": cps.StaffRegions,
            "_ensure_same_number_of_staffs": cps._ensure_same_number_of_staffs,
            "_get_number_of_voices": cps._get_number_of_voices,
            "detect_staffs_in_image_onnx": cps.detect_staffs_in_image_onnx,
            "ensure_uint8_grayscale": cps.ensure_uint8_grayscale,
            "prepare_staff_image": cps.prepare_staff_image,
            "tr_omr_max_height": cps.tr_omr_max_height,
            "tr_omr_max_width": cps.tr_omr_max_width,
            "uint8_to_float01": cps.uint8_to_float01,
            "validate_prepared_shape": cps.validate_prepared_shape,
        }

    return _LAYOUT_HELPERS


@dataclass(frozen=True)
class PreparedStaffItem:
    image_stem: str
    source_image: str
    index: int
    voice_index: int
    staff_index_within_voice: int
    image: np.ndarray  # float32 [0, 1], shape [256, 1280]


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")
    return data


def _config_get(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = config
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _as_float_list(values: Any, default: list[float]) -> list[float]:
    if values is None:
        return default
    if isinstance(values, (int, float, str)):
        return [float(values)]
    return [float(v) for v in values]


def _read_bgr_float01(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return (image.astype(np.float32) / 255.0).astype(np.float32)


def _write_temp_bgr_png(image_float01: np.ndarray, directory: Path, stem: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{stem}.png"
    out = np.clip(image_float01 * 255.0, 0.0, 255.0).round().astype(np.uint8)
    ok = cv2.imwrite(str(path), out)
    if not ok:
        raise OSError(f"Failed to write temporary image: {path}")
    return path


def extract_prepared_staffs_from_image_path(
    *,
    image_path: Path,
    segnet: Any,
    enable_debug: bool,
    strict_shape: bool,
) -> list[PreparedStaffItem]:
    """
    Run the ONNX SegNet + deterministic HOMR layout path and return prepared staffs.

    This mirrors the core of dataset/cache_prepared_staffs.cache_one_image(...),
    but returns arrays in memory instead of writing a cache directory.
    """
    helpers = _load_layout_helpers()
    detect_staffs_in_image_onnx = helpers["detect_staffs_in_image_onnx"]
    _ensure_same_number_of_staffs = helpers["_ensure_same_number_of_staffs"]
    _get_number_of_voices = helpers["_get_number_of_voices"]
    StaffRegions = helpers["StaffRegions"]
    prepare_staff_image = helpers["prepare_staff_image"]
    ensure_uint8_grayscale = helpers["ensure_uint8_grayscale"]
    validate_prepared_shape = helpers["validate_prepared_shape"]
    uint8_to_float01 = helpers["uint8_to_float01"]
    tr_omr_max_height = helpers["tr_omr_max_height"]
    tr_omr_max_width = helpers["tr_omr_max_width"]

    multi_staffs, preprocessed_image, _debug = detect_staffs_in_image_onnx(
        image_path=image_path,
        segnet=segnet,
        enable_debug=enable_debug,
    )

    multi_staffs = _ensure_same_number_of_staffs(multi_staffs, preprocessed_image)
    regions = StaffRegions(multi_staffs)
    number_of_voices = _get_number_of_voices(multi_staffs)

    items: list[PreparedStaffItem] = []
    cache_index = 0

    for voice_index in range(number_of_voices):
        staffs_for_voice = [multi_staff.staffs[voice_index] for multi_staff in multi_staffs]

        for staff_index_within_voice, staff in enumerate(staffs_for_voice):
            staff_image_raw, _transformed_staff = prepare_staff_image(
                debug=_debug,
                index=cache_index,
                staff=staff,
                staff_image=preprocessed_image,
                regions=regions,
            )

            staff_uint8 = ensure_uint8_grayscale(staff_image_raw)

            if strict_shape:
                validate_prepared_shape(
                    staff_uint8,
                    source=f"{image_path.name} staff {cache_index}",
                )

            if tuple(staff_uint8.shape) != (int(tr_omr_max_height), int(tr_omr_max_width)):
                raise ValueError(
                    f"Prepared staff shape mismatch for {image_path.name} staff {cache_index}: "
                    f"got {staff_uint8.shape}, expected {(tr_omr_max_height, tr_omr_max_width)}"
                )

            items.append(
                PreparedStaffItem(
                    image_stem=image_path.stem,
                    source_image=str(image_path),
                    index=int(cache_index),
                    voice_index=int(voice_index),
                    staff_index_within_voice=int(staff_index_within_voice),
                    image=uint8_to_float01(staff_uint8),
                )
            )
            cache_index += 1

    return items


def _tokens_to_cer_string(tokens: list[str]) -> str:
    return " ".join(tokens)


def _predict_staff_tokens(wrapper: HomrWrapper, staff_image: np.ndarray) -> list[str]:
    symbols = wrapper.predict_prepared_staff(staff_image)
    return symbols_to_strings(symbols)


def _compute_pair_metrics(pred_tokens: list[str], target_tokens: list[str]) -> dict[str, float]:
    pred_str = _tokens_to_cer_string(pred_tokens)
    target_str = _tokens_to_cer_string(target_tokens)
    return {
        "ser": float(symbol_error_rate(pred_tokens, target_tokens)),
        "cer": float(character_error_rate(pred_str, target_str)),
        "ser_capped": float(min(symbol_error_rate(pred_tokens, target_tokens), 1.0)),
        "cer_capped": float(min(character_error_rate(pred_str, target_str), 1.0)),
    }


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def _pstdev(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0 if values else float("nan")
    m = _mean(values)
    return float(math.sqrt(sum((v - m) ** 2 for v in values) / len(values)))


def _metric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        try:
            value_f = float(value)
        except Exception:
            continue
        if math.isfinite(value_f):
            values.append(value_f)
    return values


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    """
    Aggregate staff-level metric rows.

    Raw SER/CER may exceed 1.0 because Levenshtein distance includes insertions.
    Capped SER/CER are bounded at 1.0 for plots only.
    """
    output: dict[str, float] = {}

    metric_pairs = {
        "ser": ("mean_ser", "std_ser"),
        "cer": ("mean_cer", "std_cer"),
        "ser_capped": ("mean_ser_capped", "std_ser_capped"),
        "cer_capped": ("mean_cer_capped", "std_cer_capped"),
        "clean_ser": ("mean_clean_ser", "std_clean_ser"),
        "clean_cer": ("mean_clean_cer", "std_clean_cer"),
        "clean_ser_capped": ("mean_clean_ser_capped", "std_clean_ser_capped"),
        "clean_cer_capped": ("mean_clean_cer_capped", "std_clean_cer_capped"),
        "delta_ser": ("mean_delta_ser", "std_delta_ser"),
        "delta_cer": ("mean_delta_cer", "std_delta_cer"),
        "delta_ser_capped": ("mean_delta_ser_capped", "std_delta_ser_capped"),
        "delta_cer_capped": ("mean_delta_cer_capped", "std_delta_cer_capped"),
    }

    for source_key, (mean_key, std_key) in metric_pairs.items():
        values = _metric_values(rows, source_key)
        output[mean_key] = _mean(values) if values else 0.0
        output[std_key] = _pstdev(values) if values else 0.0

    output["n"] = float(len(rows))
    return output


def _load_plot_module(script_path: Path | None) -> Any | None:
    """
    Load plot_results.py from the repo's current location.

    Your repo currently stores it under:
        attacks/python scripts/plot_results.py

    Older code used:
        attacks/plot_results.py

    This loader supports both without requiring the folder-with-space to be a
    Python package.
    """
    candidates: list[Path] = []

    if script_path is not None:
        candidates.append(Path(script_path))

    candidates.extend(
        [
            ROOT_DIR / "attacks" / "python scripts" / "plot_results.py",
            ROOT_DIR / "attacks" / "plot_results.py",
        ]
    )

    for candidate in candidates:
        if not candidate.exists():
            continue

        spec = importlib.util.spec_from_file_location("homr_benchmark_plot_results", candidate)
        if spec is None or spec.loader is None:
            continue

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return None


def _safe_plot(metrics: dict[str, Any], plots_dir: Path, script_path: Path | None = None, plot_set: str = "core") -> Path | None:
    """
    Plot Track A results.

    Preferred path: call plot_results.plot_spectral_results(...), which writes the
    complete CSV/HTML plot suite. Fallback: write a single compact HTML curve so
    sweeps never fail merely because plotting is unavailable.
    """
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_module = _load_plot_module(script_path)
    if plot_module is not None and hasattr(plot_module, "plot_spectral_results"):
        try:
            try:
                return plot_module.plot_spectral_results_from_payload(
                    payload=metrics,
                    source_json=None,
                    output_dir=plots_dir,
                    plot_set=plot_set,
                )
            except TypeError:
                # Backward compatibility with older plot_results.py versions.
                return plot_module.plot_spectral_results_from_payload(
                    payload=metrics,
                    source_json=None,
                    output_dir=plots_dir,
                )
        except Exception as exc:
            print(f"[plot] full spectral plot suite failed; falling back to compact plot: {type(exc).__name__}: {exc}")

    try:
        import plotly.graph_objects as go
    except Exception as exc:  # pragma: no cover - optional plotting dependency
        print(f"[plot] skipped: could not import plotly ({type(exc).__name__}: {exc})")
        return None

    results = metrics.get("spectral_noise_results", [])
    if not results:
        print("[plot] skipped: no spectral_noise_results")
        return None

    x = [row["epsilon_phys"] for row in results]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=[row.get("mean_delta_ser", 0.0) for row in results], mode="lines+markers", name="mean ΔSER"))
    fig.add_trace(go.Scatter(x=x, y=[row.get("mean_delta_cer", 0.0) for row in results], mode="lines+markers", name="mean ΔCER"))

    fig.update_layout(
        title="Track A spectral-noise degradation: delta metrics",
        xaxis_title="epsilon_phys",
        yaxis_title="delta error rate",
        template="plotly_white",
    )

    out = plots_dir / "spectral_ser_cer.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    return out



def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def format_duration(seconds: float) -> str:
    if seconds is None or not math.isfinite(float(seconds)):
        return "unknown"
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hours:
        return f"{hours}h {minutes}m {secs:04.1f}s"
    if minutes:
        return f"{minutes}m {secs:04.1f}s"
    return f"{secs:.1f}s"


def append_jsonl(path: Path | None, record: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        value_f = float(value)
    except Exception:
        return default
    return value_f if math.isfinite(value_f) else default


def _condition_id(alpha: float, epsilon_phys: float) -> str:
    return f"alpha={float(alpha):.10g}|epsilon_phys={float(epsilon_phys):.10g}"


def _job_key(image_path: Path, alpha: float, epsilon_phys: float, target_source: str) -> tuple[str, str, str, str]:
    return (
        str(Path(image_path).as_posix()),
        f"{float(alpha):.10g}",
        f"{float(epsilon_phys):.10g}",
        str(target_source),
    )


def _record_key(record: dict[str, Any]) -> tuple[str, str, str, str] | None:
    try:
        return (
            str(Path(record["image_path"]).as_posix()),
            f"{float(record['alpha']):.10g}",
            f"{float(record['epsilon_phys']):.10g}",
            str(record.get("target_source", "clean_onnx_prediction")),
        )
    except Exception:
        return None


def _load_checkpoint_records(path: Path | None) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    records: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"[warning] ignoring malformed checkpoint line {line_no}: {path}", file=sys.stderr)
                continue
            if row.get("status") != "ok":
                continue
            key = _record_key(row)
            if key is not None:
                records[key] = row
    return records


def _aggregate_image_jobs(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate image-level job rows using image means. Staff-weighted science metrics come from staff rows."""
    out: dict[str, float] = {}
    pairs = {
        "clean_ser": ("mean_clean_ser", "std_clean_ser"),
        "clean_cer": ("mean_clean_cer", "std_clean_cer"),
        "clean_ser_capped": ("mean_clean_ser_capped", "std_clean_ser_capped"),
        "clean_cer_capped": ("mean_clean_cer_capped", "std_clean_cer_capped"),
        "ser": ("mean_image_ser", "std_image_ser"),
        "cer": ("mean_image_cer", "std_image_cer"),
        "ser_capped": ("mean_image_ser_capped", "std_image_ser_capped"),
        "cer_capped": ("mean_image_cer_capped", "std_image_cer_capped"),
        "delta_ser": ("mean_image_delta_ser", "std_image_delta_ser"),
        "delta_cer": ("mean_image_delta_cer", "std_image_delta_cer"),
        "job_elapsed_sec": ("mean_job_elapsed_sec", "std_job_elapsed_sec"),
        "noise_generation_sec": ("mean_noise_generation_sec", "std_noise_generation_sec"),
        "layout_elapsed_sec": ("mean_layout_elapsed_sec", "std_layout_elapsed_sec"),
        "recognition_elapsed_sec": ("mean_recognition_elapsed_sec", "std_recognition_elapsed_sec"),
    }
    for source, (mean_key, std_key) in pairs.items():
        vals = _metric_values(rows, source)
        out[mean_key] = _mean(vals) if vals else 0.0
        out[std_key] = _pstdev(vals) if vals else 0.0
    return out


def _image_job_from_staff_rows(image_record: dict[str, Any], staff_rows: list[dict[str, Any]]) -> dict[str, Any]:
    agg = _aggregate_rows(staff_rows)
    out = dict(image_record)
    # Mirror Track B job naming where the primary attacked/degraded metrics are ser/cer.
    out.update(
        {
            "ser": float(agg.get("mean_ser", 0.0)),
            "cer": float(agg.get("mean_cer", 0.0)),
            "ser_capped": float(agg.get("mean_ser_capped", 0.0)),
            "cer_capped": float(agg.get("mean_cer_capped", 0.0)),
            "clean_ser": float(agg.get("mean_clean_ser", 0.0)),
            "clean_cer": float(agg.get("mean_clean_cer", 0.0)),
            "clean_ser_capped": float(agg.get("mean_clean_ser_capped", 0.0)),
            "clean_cer_capped": float(agg.get("mean_clean_cer_capped", 0.0)),
            "delta_ser": float(agg.get("mean_delta_ser", 0.0)),
            "delta_cer": float(agg.get("mean_delta_cer", 0.0)),
            "delta_ser_capped": float(agg.get("mean_delta_ser_capped", 0.0)),
            "delta_cer_capped": float(agg.get("mean_delta_cer_capped", 0.0)),
        }
    )
    out.update({f"staff_{k}": v for k, v in agg.items()})
    return out


def run_spectral_sweep(args: argparse.Namespace) -> dict[str, Any]:
    """
    Image-major Track A runner.

    Efficiency rule:
        - Clean layout/recognition is computed once per image.
        - For each image and alpha, the multiscale Fourier noise field is generated once.
        - epsilon=0 jobs reuse clean predictions and do not rerun layout.
        - Nonzero epsilon jobs rerun full-page noisy layout, preserving Track A semantics.
    """
    config = _load_yaml(args.config)

    images_dir = args.images_dir or Path(_config_get(config, "dataset", "images_dir", default="dataset/images"))
    model_dir = args.model_dir or Path("models/onnx")
    segnet_model = args.segnet_model or (model_dir / "segnet.onnx")
    output_json = args.output_json or (Path(_config_get(config, "output", "logs_dir", default="results/logs")) / _config_get(config, "output", "log_filename", default="sweep_metrics.json"))
    plots_dir = args.plots_dir or Path(_config_get(config, "output", "plots_dir", default="results/plots"))

    default_eps = _as_float_list(
        _config_get(config, "spectral_noise", "epsilon_phys_grid", default=None),
        [0.0, 0.05, 0.10, 0.20, 0.30, 0.50],
    )
    epsilon_grid = [float(v) for v in (args.epsilon if args.epsilon is not None else default_eps)]

    default_alpha_grid = _as_float_list(
        _config_get(config, "spectral_noise", "alpha_grid", default=None),
        None,
    )
    if default_alpha_grid is None:
        default_alpha_grid = [float(_config_get(config, "spectral_noise", "alpha", default=1.0))]
    alpha_grid = [float(v) for v in (args.alpha if args.alpha is not None else default_alpha_grid)]

    max_images = args.max_images
    if max_images is None:
        configured_n = _config_get(config, "dataset", "n_images", default=None)
        max_images = int(configured_n) if configured_n is not None else None

    image_paths = find_image_paths(images_dir, recursive=args.recursive)
    if max_images is not None and max_images > 0:
        image_paths = image_paths[:max_images]
    if not image_paths:
        raise FileNotFoundError(f"No input images found under {images_dir}")

    if args.checkpoint_jsonl is None:
        args.checkpoint_jsonl = args.timing_jsonl
    if args.timing_jsonl is None and args.checkpoint_jsonl is not None:
        args.timing_jsonl = args.checkpoint_jsonl
    if args.resume and args.checkpoint_jsonl is None:
        raise ValueError("--resume requires --checkpoint-jsonl or --timing-jsonl")

    run_started_perf = time.perf_counter()
    run_start_iso = now_iso()

    total_jobs = len(image_paths) * len(alpha_grid) * len(epsilon_grid)
    completed_records = _load_checkpoint_records(args.checkpoint_jsonl) if args.resume else {}
    expected_keys = {
        _job_key(image_path, alpha, eps, args.target_source)
        for image_path in image_paths
        for alpha in alpha_grid
        for eps in epsilon_grid
    }
    resumed_records = {k: v for k, v in completed_records.items() if k in expected_keys}

    print(f"[Track A] images={len(image_paths)} epsilons={epsilon_grid} alphas={alpha_grid}")
    print("[Track A] execution_order=image-major")
    print(f"[Track A] target_source={args.target_source}")
    print(f"[Track A] output_json={output_json}")
    print(f"[Track A] total_jobs={total_jobs} resumed={len(resumed_records)}")
    if args.timing_jsonl:
        print(f"[Track A] timing/checkpoint JSONL: {args.timing_jsonl}")

    # Construct the TrOMR ONNX wrapper before importing HOMR layout helpers.
    wrapper = HomrWrapper(
        model_dir=model_dir,
        use_cuda=not args.cpu,
        max_decode_len=args.max_decode_len,
        strict_import_guard=False,
    )

    helpers = _load_layout_helpers()
    SegNetONNX = helpers["SegNetONNX"]
    segnet = SegNetONNX(
        model_path=segnet_model,
        use_cuda=not args.cpu,
        batch_size=args.batch_size,
        win_size=args.win_size,
        step_size=args.step_size,
    )

    staff_level_results: list[dict[str, Any]] = []
    image_level_results: list[dict[str, Any]] = list(resumed_records.values())
    jobs: list[dict[str, Any]] = list(resumed_records.values())
    error_rows: list[dict[str, Any]] = []

    clean_targets: dict[str, list[list[str]]] = {}
    clean_staff_counts: dict[str, int] = {}
    clean_extract_elapsed: dict[str, float] = {}
    clean_predict_elapsed: dict[str, float] = {}
    clean_query_count = 0

    completed_jobs = len(resumed_records)
    started_nonresumed = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="track_a_spectral_") as tmp_name:
        tmp_dir = Path(tmp_name)

        for image_i, image_path in enumerate(image_paths, start=1):
            image_run_started = time.perf_counter()
            image_stem = image_path.stem
            print(f"[image {image_i}/{len(image_paths)}] {image_path.name}")

            # Clean baseline is needed if any job for this image is not resumed.
            image_keys = [_job_key(image_path, alpha, eps, args.target_source) for alpha in alpha_grid for eps in epsilon_grid]
            if all(key in resumed_records for key in image_keys):
                print("  [skip] all image-alpha-epsilon jobs already completed")
                continue

            try:
                clean_started = time.perf_counter()
                extract_started = time.perf_counter()
                clean_staffs = extract_prepared_staffs_from_image_path(
                    image_path=image_path,
                    segnet=segnet,
                    enable_debug=args.debug,
                    strict_shape=not args.no_strict_shape,
                )
                clean_extract_elapsed[image_stem] = time.perf_counter() - extract_started

                predict_started = time.perf_counter()
                clean_tokens = [_predict_staff_tokens(wrapper, item.image) for item in clean_staffs]
                clean_predict_elapsed[image_stem] = time.perf_counter() - predict_started
                clean_staff_counts[image_stem] = len(clean_staffs)
                clean_targets[image_stem] = clean_tokens
                clean_query_count += len(clean_tokens)

                print(
                    f"  [clean] staffs={len(clean_staffs)} "
                    f"extract={clean_extract_elapsed[image_stem]:.1f}s "
                    f"recognition={clean_predict_elapsed[image_stem]:.1f}s"
                )
            except Exception as exc:
                record = {
                    "track": "A",
                    "status": "error",
                    "phase": "clean_target",
                    "image": image_path.name,
                    "image_path": str(image_path),
                    "image_stem": image_stem,
                    "image_index": int(image_i),
                    "job_elapsed_sec": float(time.perf_counter() - image_run_started),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "error": str(exc),
                }
                error_rows.append(record)
                append_jsonl(args.timing_jsonl, record)
                if not args.continue_on_error:
                    raise
                continue

            clean_bgr = _read_bgr_float01(image_path)

            for alpha_i, alpha in enumerate(alpha_grid, start=1):
                alpha_started = time.perf_counter()
                noise_started = time.perf_counter()
                noise_field, noise_metadata = generate_multiscale_colored_noise(
                    height=clean_bgr.shape[0],
                    width=clean_bgr.shape[1],
                    alpha=alpha,
                    seed=int(args.seed) + image_i * 1009 + alpha_i * 5711,
                    return_metadata=True,
                )
                noise_generation_sec = time.perf_counter() - noise_started
                sampled_weights = dict(noise_metadata.get("sampled_weights", {}))
                expected_weights = dict(noise_metadata.get("expected_weights", {}))

                print(
                    f"  [alpha {alpha_i}/{len(alpha_grid)}] alpha={alpha:g} "
                    f"weights=(w={sampled_weights.get('white', 0.0):.3f}, "
                    f"p={sampled_weights.get('pink', 0.0):.3f}, "
                    f"b={sampled_weights.get('brown', 0.0):.3f}) "
                    f"noise={noise_generation_sec:.2f}s"
                )

                for eps_i, epsilon_phys in enumerate(epsilon_grid, start=1):
                    key = _job_key(image_path, alpha, epsilon_phys, args.target_source)
                    condition_id = _condition_id(alpha, epsilon_phys)
                    if key in resumed_records:
                        continue

                    job_started = time.perf_counter()
                    staff_rows_for_job: list[dict[str, Any]] = []
                    clean_count = int(clean_staff_counts.get(image_stem, 0))
                    n_pairs = 0
                    n_noisy_queries = 0
                    layout_elapsed = 0.0
                    recognition_elapsed = 0.0
                    noisy_count = clean_count
                    layout_match = True
                    layout_success = clean_count > 0
                    status = "ok"

                    image_record: dict[str, Any] = {
                        "track": "A",
                        "status": "ok",
                        "condition_id": condition_id,
                        "image": image_path.name,
                        "image_path": str(image_path),
                        "image_stem": image_stem,
                        "image_index": int(image_i),
                        "alpha": float(alpha),
                        "epsilon": float(epsilon_phys),
                        "epsilon_phys": float(epsilon_phys),
                        "epsilon_math": None,
                        "stage": args.stage,
                        "onnx_mode": True,
                        "target_source": args.target_source,
                        "noise_model": noise_metadata.get("model"),
                        "noise_weight_white": float(sampled_weights.get("white", 0.0)),
                        "noise_weight_pink": float(sampled_weights.get("pink", 0.0)),
                        "noise_weight_brown": float(sampled_weights.get("brown", 0.0)),
                        "noise_expected_weight_white": float(expected_weights.get("white", 0.0)),
                        "noise_expected_weight_pink": float(expected_weights.get("pink", 0.0)),
                        "noise_expected_weight_brown": float(expected_weights.get("brown", 0.0)),
                        "noise_metadata": noise_metadata,
                        "clean_n_staffs": clean_count,
                        "noisy_n_staffs": None,
                        "n_staff_pairs": 0,
                        "layout_match": False,
                        "layout_success": False,
                        "noise_generation_sec": float(noise_generation_sec),
                        "layout_elapsed_sec": None,
                        "recognition_elapsed_sec": None,
                        "job_elapsed_sec": None,
                    }

                    try:
                        if float(epsilon_phys) == 0.0:
                            # Identity job. Avoid rerunning full-page layout for every alpha.
                            noisy_staffs = clean_staffs
                            noisy_tokens_by_staff = clean_targets[image_stem]
                            noisy_count = clean_count
                            n_pairs = clean_count
                            layout_match = True
                            layout_success = n_pairs > 0
                            layout_elapsed = 0.0
                            recognition_elapsed = 0.0
                        else:
                            noisy_bgr = apply_endpoint_interpolation(
                                clean_bgr,
                                noise_field,
                                epsilon_phys=float(epsilon_phys),
                            )

                            if args.save_debug_images and image_i <= args.debug_image_limit:
                                debug_path = args.debug_dir / f"{image_path.stem}_eps_{epsilon_phys:g}_alpha_{alpha:g}.png"
                                save_float_image(debug_path, noisy_bgr)

                            noisy_path = _write_temp_bgr_png(
                                noisy_bgr,
                                directory=tmp_dir,
                                stem=f"{image_path.stem}_alpha_{alpha:g}_eps_{epsilon_phys:g}",
                            )

                            layout_started = time.perf_counter()
                            noisy_staffs = extract_prepared_staffs_from_image_path(
                                image_path=noisy_path,
                                segnet=segnet,
                                enable_debug=args.debug,
                                strict_shape=not args.no_strict_shape,
                            )
                            layout_elapsed = time.perf_counter() - layout_started

                            target_staff_tokens = clean_targets[image_stem]
                            noisy_count = len(noisy_staffs)
                            layout_match = noisy_count == len(target_staff_tokens)
                            n_pairs = min(noisy_count, len(target_staff_tokens))
                            layout_success = n_pairs > 0

                            recognition_started = time.perf_counter()
                            noisy_tokens_by_staff = [
                                _predict_staff_tokens(wrapper, noisy_staffs[staff_j].image)
                                for staff_j in range(n_pairs)
                            ]
                            recognition_elapsed = time.perf_counter() - recognition_started
                            n_noisy_queries = n_pairs

                        target_staff_tokens = clean_targets[image_stem]

                        for staff_j in range(n_pairs):
                            pred_tokens = noisy_tokens_by_staff[staff_j]
                            target_tokens = target_staff_tokens[staff_j]
                            noisy_metrics = _compute_pair_metrics(pred_tokens, target_tokens)
                            clean_metrics = {
                                "clean_ser": 0.0,
                                "clean_cer": 0.0,
                                "clean_ser_capped": 0.0,
                                "clean_cer_capped": 0.0,
                            }
                            row = {
                                "track": "A",
                                "status": "ok",
                                "condition_id": condition_id,
                                "image": image_path.name,
                                "image_path": str(image_path),
                                "image_stem": image_stem,
                                "image_index": int(image_i),
                                "staff_filename": None,
                                "staff_index": int(staff_j),
                                "alpha": float(alpha),
                                "epsilon": float(epsilon_phys),
                                "epsilon_phys": float(epsilon_phys),
                                "epsilon_math": None,
                                "target_source": args.target_source,
                                "stage": args.stage,
                                "onnx_mode": True,
                                "clean_n_staffs": clean_count,
                                "noisy_n_staffs": int(noisy_count),
                                "layout_match": bool(layout_match),
                                "layout_success": bool(layout_success),
                                "n_pred_tokens": int(len(pred_tokens)),
                                "n_target_tokens": int(len(target_tokens)),
                                "pred_tokens_preview": pred_tokens[: min(16, len(pred_tokens))],
                                "target_tokens_preview": target_tokens[: min(16, len(target_tokens))],
                                "noise_weight_white": float(sampled_weights.get("white", 0.0)),
                                "noise_weight_pink": float(sampled_weights.get("pink", 0.0)),
                                "noise_weight_brown": float(sampled_weights.get("brown", 0.0)),
                                **clean_metrics,
                                **noisy_metrics,
                            }
                            row["delta_ser"] = float(row["ser"] - row["clean_ser"])
                            row["delta_cer"] = float(row["cer"] - row["clean_cer"])
                            row["delta_ser_capped"] = float(row["ser_capped"] - row["clean_ser_capped"])
                            row["delta_cer_capped"] = float(row["cer_capped"] - row["clean_cer_capped"])
                            staff_rows_for_job.append(row)
                            staff_level_results.append(row)

                        image_record.update(
                            {
                                "status": status,
                                "noisy_n_staffs": int(noisy_count),
                                "n_staff_pairs": int(n_pairs),
                                "layout_match": bool(layout_match),
                                "layout_success": bool(layout_success),
                                "layout_elapsed_sec": float(layout_elapsed),
                                "recognition_elapsed_sec": float(recognition_elapsed),
                                "queries": int(n_noisy_queries),
                                "job_elapsed_sec": float(time.perf_counter() - job_started),
                            }
                        )
                        image_record = _image_job_from_staff_rows(image_record, staff_rows_for_job)

                        image_level_results.append(image_record)
                        jobs.append(image_record)
                        append_jsonl(args.timing_jsonl, image_record)
                        completed_jobs += 1

                        if args.progress_every and completed_jobs % int(args.progress_every) == 0:
                            elapsed_progress = time.perf_counter() - started_nonresumed
                            done_new = max(1, completed_jobs - len(resumed_records))
                            jobs_per_min = (done_new / elapsed_progress) * 60.0 if elapsed_progress > 0 else 0.0
                            remaining = max(0, total_jobs - completed_jobs)
                            eta = remaining / (jobs_per_min / 60.0) if jobs_per_min > 0 else float("inf")
                            print(
                                f"    [progress] {completed_jobs}/{total_jobs} jobs | "
                                f"elapsed={format_duration(time.perf_counter() - run_started_perf)} | "
                                f"ETA={format_duration(eta)} | speed={jobs_per_min:.2f} jobs/min"
                            )

                    except Exception as exc:
                        image_record.update(
                            {
                                "track": "A",
                                "status": "error",
                                "phase": "epsilon_sweep",
                                "noisy_n_staffs": int(noisy_count) if noisy_count is not None else None,
                                "n_staff_pairs": int(n_pairs),
                                "layout_match": bool(layout_match),
                                "layout_success": bool(layout_success),
                                "layout_elapsed_sec": float(layout_elapsed),
                                "recognition_elapsed_sec": float(recognition_elapsed),
                                "job_elapsed_sec": float(time.perf_counter() - job_started),
                                "error_type": type(exc).__name__,
                                "error_message": str(exc),
                                "error": str(exc),
                            }
                        )
                        image_level_results.append(image_record)
                        jobs.append(image_record)
                        error_rows.append(image_record)
                        append_jsonl(args.timing_jsonl, image_record)
                        completed_jobs += 1
                        if not args.continue_on_error:
                            raise

                print(f"  [alpha done] alpha={alpha:g} elapsed={format_duration(time.perf_counter() - alpha_started)}")

    # Aggregate completed staff rows by condition.
    aggregate_results: list[dict[str, Any]] = []
    for alpha in alpha_grid:
        for epsilon_phys in epsilon_grid:
            condition_id = _condition_id(alpha, epsilon_phys)
            staff_rows = [r for r in staff_level_results if r.get("condition_id") == condition_id and r.get("status") == "ok"]
            job_rows = [r for r in image_level_results if r.get("condition_id") == condition_id]
            ok_jobs = [r for r in job_rows if r.get("status") == "ok"]
            err_jobs = [r for r in job_rows if r.get("status") != "ok"]

            agg_staff = _aggregate_rows(staff_rows)
            agg_jobs = _aggregate_image_jobs(ok_jobs)

            elapsed_values = _metric_values(ok_jobs + err_jobs, "job_elapsed_sec")
            condition_elapsed = float(sum(elapsed_values)) if elapsed_values else 0.0
            layout_success_values = [1.0 if r.get("layout_success") else 0.0 for r in ok_jobs]
            layout_match_values = [1.0 if r.get("layout_match") else 0.0 for r in ok_jobs]
            clean_staff_values = [_safe_float(r.get("clean_n_staffs")) for r in ok_jobs]
            noisy_staff_values = [_safe_float(r.get("noisy_n_staffs")) for r in ok_jobs]
            weight_white_values = [_safe_float(r.get("noise_weight_white")) for r in ok_jobs]
            weight_pink_values = [_safe_float(r.get("noise_weight_pink")) for r in ok_jobs]
            weight_brown_values = [_safe_float(r.get("noise_weight_brown")) for r in ok_jobs]
            success_values = [1.0 if _safe_float(r.get("ser")) > _safe_float(r.get("clean_ser")) else 0.0 for r in ok_jobs]

            row = {
                "track": "A",
                "condition_id": condition_id,
                "alpha": float(alpha),
                "epsilon": float(epsilon_phys),
                "epsilon_phys": float(epsilon_phys),
                "epsilon_math": None,
                "stage": args.stage,
                "onnx_mode": True,
                "target_source": args.target_source,
                "n_images": int(len(ok_jobs)),
                "n_images_attempted": int(len(job_rows)),
                "n_images_ok": int(len(ok_jobs)),
                "n_images_failed": int(len(err_jobs)),
                "n_jobs_total": int(len(job_rows)),
                "n_jobs_ok": int(len(ok_jobs)),
                "n_jobs_error": int(len(err_jobs)),
                "n_staffs": int(len(staff_rows)),
                "n_staff_pairs": int(len(staff_rows)),
                "n_staff_pairs_expected": int(sum(clean_staff_values)),
                "n_staff_pairs_evaluated": int(len(staff_rows)),
                "n_layout_mismatch_images": int(sum(1 for r in ok_jobs if not r.get("layout_match"))),
                "layout_success_rate": _mean(layout_success_values) if layout_success_values else 0.0,
                "layout_match_rate": _mean(layout_match_values) if layout_match_values else 0.0,
                "mean_staffs_clean": _mean(clean_staff_values) if clean_staff_values else 0.0,
                "mean_staffs_noisy": _mean(noisy_staff_values) if noisy_staff_values else 0.0,
                "success_rate": _mean(success_values) if success_values else 0.0,
                "condition_elapsed_sec": condition_elapsed,
                "elapsed_seconds": condition_elapsed,
                "mean_job_elapsed_sec": agg_jobs.get("mean_job_elapsed_sec", 0.0),
                "std_job_elapsed_sec": agg_jobs.get("std_job_elapsed_sec", 0.0),
                "jobs_per_min": float((len(ok_jobs) / condition_elapsed) * 60.0) if condition_elapsed > 0 else 0.0,
                "images_per_second": float(len(ok_jobs) / condition_elapsed) if condition_elapsed > 0 else 0.0,
                "staffs_per_second": float(len(staff_rows) / condition_elapsed) if condition_elapsed > 0 else 0.0,
                "mean_noise_generation_sec": agg_jobs.get("mean_noise_generation_sec", 0.0),
                "mean_layout_elapsed_sec": agg_jobs.get("mean_layout_elapsed_sec", 0.0),
                "mean_recognition_elapsed_sec": agg_jobs.get("mean_recognition_elapsed_sec", 0.0),
                "mean_noise_weight_white": _mean(weight_white_values) if weight_white_values else 0.0,
                "mean_noise_weight_pink": _mean(weight_pink_values) if weight_pink_values else 0.0,
                "mean_noise_weight_brown": _mean(weight_brown_values) if weight_brown_values else 0.0,
                "clean_queries": int(clean_query_count),
                "noisy_queries": int(sum(_safe_float(r.get("queries")) for r in ok_jobs)),
                "total_queries": int(clean_query_count + sum(_safe_float(r.get("queries")) for r in ok_jobs)),
                **agg_staff,
            }
            aggregate_results.append(row)
            print(
                f"[aggregate] alpha={alpha:g} eps={epsilon_phys:g} "
                f"mean_ser={row['mean_ser']:.6f} mean_cer={row['mean_cer']:.6f} "
                f"staffs={row['n_staffs']} layout_match={row['layout_match_rate']:.3f} "
                f"elapsed={condition_elapsed:.1f}s"
            )

    total_elapsed = time.perf_counter() - run_started_perf
    payload = {
        "experiment_metadata": {
            "run_start": run_start_iso,
            "run_end": now_iso(),
            "run_elapsed_sec": float(total_elapsed),
            "run_elapsed": format_duration(total_elapsed),
            "track": "A",
            "benchmark_mode": "full_page_multiscale_spectral_noise_onnx_self_consistency",
            "experiment": "spectral_noise_degradation",
            "execution_order": "image_major_clean_once_alpha_noise_once_epsilon_inner",
            "stage": args.stage,
            "onnx_mode": True,
            "target_source": args.target_source,
            "target_source_warning": (
                "clean_onnx_prediction measures degradation drift relative to clean ONNX output; "
                "it is not final MusicXML ground-truth accuracy."
            ),
            "images_dir": str(images_dir),
            "model_dir": str(model_dir),
            "segnet_model": str(segnet_model),
            "n_images_requested": int(len(image_paths)),
            "n_clean_targets_ok": int(len(clean_targets)),
            "n_clean_targets_error": int(sum(1 for r in error_rows if r.get("phase") == "clean_target")),
            "n_jobs_total": int(total_jobs),
            "n_jobs_ok": int(sum(1 for r in jobs if r.get("status") == "ok")),
            "n_jobs_error": int(sum(1 for r in jobs if r.get("status") != "ok")),
            "resumed_records_count": int(len(resumed_records)),
            "spectral_alpha_grid": alpha_grid,
            "epsilon_phys_grid": epsilon_grid,
            "noise_model": "alpha_conditioned_multiscale_fourier_endpoint",
            "noise_model_notes": [
                "Each image-alpha pair samples white, pink, and brown Fourier components with alpha-conditioned Dirichlet weights.",
                "The same sampled noise field is reused across all epsilon values for that image-alpha pair.",
                "The mixed field is normalized before endpoint interpolation, so epsilon_phys is the single global severity parameter.",
                "Negative field values move pixels toward black; positive values move pixels toward white.",
            ],
            "providers": wrapper.encoder_sess.get_providers(),
            "segnet_providers": segnet.session.get_providers(),
            "timing_jsonl": str(args.timing_jsonl) if args.timing_jsonl else None,
            "checkpoint_jsonl": str(args.checkpoint_jsonl) if args.checkpoint_jsonl else None,
            "resume": bool(args.resume),
            "plot": bool(args.plot),
            "plots_dir": str(plots_dir),
            "plot_script": str(args.plot_script) if args.plot_script else None,
            "plot_set": str(args.plot_set),
            "metric_notes": [
                "Raw SER/CER are normalized Levenshtein edit distances and may exceed 1.0 due to insertions.",
                "Capped SER/CER are min(raw_metric, 1.0) and are included only for bounded visual comparison.",
                "For clean_onnx_prediction target_source, clean_* metrics are self-consistency baselines and are exactly 0 by construction.",
                "Track A jobs are image-alpha-epsilon evaluations; Track B jobs are staff-epsilon attacks.",
            ],
            "notes": [
                "Track A injects spectral noise into full-page images.",
                "SegNet and TrOMR neural inference are ONNX Runtime calls.",
                "HOMR Python is used for deterministic layout, geometry, dewarping, and prepare_staff_image only.",
                "epsilon=0 jobs reuse clean predictions and skip redundant noisy layout.",
            ],
        },
        # Unified schema preferred by future code.
        "aggregate_results": aggregate_results,
        "jobs": jobs,
        "image_level_results": image_level_results,
        "staff_level_results": staff_level_results,
        "errors": error_rows,
        # Backward-compatible Track-A alias expected by existing plotters/scripts.
        "spectral_noise_results": aggregate_results,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output_json.with_suffix(output_json.suffix + ".tmp")
    with tmp_output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    tmp_output.replace(output_json)
    print(f"[Track A] wrote {output_json}")

    if args.plot:
        plot_path = _safe_plot(payload, plots_dir, script_path=args.plot_script, plot_set=args.plot_set)
        if plot_path is not None:
            print(f"[Track A] wrote plot output under {plot_path.parent}")

    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Track A spectral-noise degradation sweep with ONNX SegNet + ONNX TrOMR."
    )

    parser.add_argument("--config", type=Path, default=Path("attacks/config/sweep_parameters.yaml"))
    parser.add_argument("--images-dir", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--segnet-model", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--plots-dir", type=Path, default=None)
    parser.add_argument("--plot-script", type=Path, default=None, help="Optional explicit path to plot_results.py. Defaults to attacks/python scripts/plot_results.py if present.")
    parser.add_argument("--timing-jsonl", type=Path, default=None, help="Optional JSONL file for per-job timing/checkpoint rows.")
    parser.add_argument("--checkpoint-jsonl", type=Path, default=None, help="Optional checkpoint JSONL. If omitted, --timing-jsonl is used.")
    parser.add_argument("--resume", action="store_true", help="Skip completed image-alpha-epsilon jobs present in checkpoint JSONL.")

    parser.add_argument("--epsilon", type=float, nargs="+", default=None, help="Override epsilon_phys grid.")
    parser.add_argument("--alpha", type=float, nargs="+", default=None, help="Override alpha grid. Alpha biases the white/pink/brown mixture distribution; e.g. --alpha 0 1 2.")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--recursive", action="store_true")

    parser.add_argument("--cpu", action="store_true", help="Force CPUExecutionProvider.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--win-size", type=int, default=320)
    parser.add_argument("--step-size", type=int, default=320)
    parser.add_argument("--max-decode-len", type=int, default=None)

    parser.add_argument(
        "--target-source",
        choices=["clean_onnx_prediction"],
        default="clean_onnx_prediction",
        help="Currently only clean ONNX self-consistency targets are validated.",
    )
    parser.add_argument(
        "--stage",
        default="A2_SELF_CONSISTENCY",
        help="Result label. Keep A2_SELF_CONSISTENCY until validated MusicXML gt-token scoring exists.",
    )

    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-set", choices=["core", "full"], default="core", help="core = delta-first primary plots; full = include raw/capped/debug plots.")
    parser.add_argument("--debug", action="store_true", help="Enable HOMR Debug image output where underlying helpers support it.")
    parser.add_argument("--no-strict-shape", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1, help="Print progress every N completed image-alpha-epsilon jobs. Use 0 to disable.")

    parser.add_argument("--save-debug-images", action="store_true")
    parser.add_argument("--debug-dir", type=Path, default=Path("results/debug/spectral_noise_examples"))
    parser.add_argument("--debug-image-limit", type=int, default=3)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_spectral_sweep(args)


if __name__ == "__main__":
    main()
