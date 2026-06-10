#!/usr/bin/env python3
"""
Sequentially degrade rendered full-page score images for HOMR student distillation.

Input:
    distillation/batches/<batch>/logs/render_log.jsonl

Output:
    <out-dir>/augmented_pages/*.png
    <out-dir>/logs/augment_log.jsonl
    <out-dir>/augmentation_summary.json

The output augment_log.jsonl is intentionally page-log compatible with
run_onnx_teacher_batch.py --page-log. HOMR must label the final transformed image,
not the original clean render.
"""

from __future__ import annotations

import argparse
import colorsys
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


PROFILE_CONFIGS: dict[str, dict[str, float]] = {
    "camera_light_v1": {
        "severity_min": 0.18,
        "severity_max": 0.34,
        "rotation_deg": 2.2,
        "translation_frac": 0.014,
        "scale_frac": 0.020,
        "skew_frac": 0.016,
        "perspective_frac": 0.028,
        "local_warp_px": 4.0,
        "artifact_strength": 0.22,
        "artifact_events": 4,
        "artifact_color_strength": 0.22,
        "artifact_warp_px": 1.4,
        "artifact_ink_interaction": 0.30,
        "blur_sigma": 0.50,
        "jpeg_quality_min": 80,
        "jpeg_quality_max": 95,
        "shadow_strength": 0.16,
        "contrast_drop": 0.14,
        "brightness_shift": 0.10,
        "ink_strength": 0.20,
        "paper_texture": 0.045,
    },

    "camera_visible_v1": {
        "severity_min": 0.44,
        "severity_max": 0.72,
        "rotation_deg": 5.8,
        "translation_frac": 0.055,
        "scale_frac": 0.065,
        "skew_frac": 0.060,
        "perspective_frac": 0.100,
        "local_warp_px": 18.0,
        "artifact_strength": 0.72,
        "artifact_events": 8,
        "artifact_color_strength": 0.58,
        "artifact_warp_px": 6.0,
        "artifact_ink_interaction": 0.88,
        "blur_sigma": 1.05,
        "jpeg_quality_min": 58,
        "jpeg_quality_max": 86,
        "shadow_strength": 0.42,
        "contrast_drop": 0.30,
        "brightness_shift": 0.24,
        "ink_strength": 0.50,
        "paper_texture": 0.105,
    },

    "camera_medium_v1": {
        "severity_min": 0.54,
        "severity_max": 0.84,
        "rotation_deg": 7.0,
        "translation_frac": 0.070,
        "scale_frac": 0.080,
        "skew_frac": 0.078,
        "perspective_frac": 0.135,
        "local_warp_px": 26.0,
        "artifact_strength": 0.95,
        "artifact_events": 12,
        "artifact_color_strength": 0.76,
        "artifact_warp_px": 9.0,
        "artifact_ink_interaction": 1.10,
        "blur_sigma": 1.35,
        "jpeg_quality_min": 48,
        "jpeg_quality_max": 80,
        "shadow_strength": 0.58,
        "contrast_drop": 0.42,
        "brightness_shift": 0.32,
        "ink_strength": 0.70,
        "paper_texture": 0.16,
    },

    "camera_hard_v1": {
        "severity_min": 0.72,
        "severity_max": 1.00,
        "rotation_deg": 9.0,
        "translation_frac": 0.090,
        "scale_frac": 0.105,
        "skew_frac": 0.105,
        "perspective_frac": 0.175,
        "local_warp_px": 42.0,
        "artifact_strength": 1.35,
        "artifact_events": 18,
        "artifact_color_strength": 1.05,
        "artifact_warp_px": 16.0,
        "artifact_ink_interaction": 1.35,
        "blur_sigma": 1.90,
        "jpeg_quality_min": 30,
        "jpeg_quality_max": 70,
        "shadow_strength": 0.82,
        "contrast_drop": 0.58,
        "brightness_shift": 0.46,
        "ink_strength": 0.98,
        "paper_texture": 0.28,
    },
}


FAMILIES = ("geometry", "paper_ink", "lighting", "artifact_field", "blur", "compression")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            rows.append(row)
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
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
    """Collect pages from either score-level render logs or page-level logs."""
    rows = read_jsonl(page_log)
    batch_dir = page_log.parent.parent
    pages: list[dict[str, Any]] = []

    for row_index, row in enumerate(rows):
        status = str(first_present(row, ["status", "render_status"]) or "ok").lower()
        if status not in {"ok", "success", "done", "completed", "rendered"}:
            continue

        score_id = str(first_present(row, ["score_id", "source_id", "id", "stem"]) or f"score_{row_index:06d}")
        source_path = first_present(row, ["source_path", "mxl_path", "musicxml_path", "xml_path"])
        rendered_pages = row.get("rendered_pages")

        if isinstance(rendered_pages, list):
            for local_index, item in enumerate(rendered_pages, start=1):
                if isinstance(item, dict):
                    raw_image = first_present(item, ["image_path", "page_path", "png_path", "rendered_image", "path"])
                    fallback_page_number = int(first_present(item, ["page_number", "page", "page_index"]) or local_index)
                else:
                    raw_image = item
                    fallback_page_number = local_index
                if raw_image is None:
                    continue
                image_path = resolve_path(raw_image, base_dir=batch_dir)
                page_number = page_number_from_path(image_path, fallback_page_number)
                pages.append({
                    "score_id": score_id,
                    "source_path": str(source_path) if source_path else None,
                    "page_id": image_path.stem,
                    "page_number": page_number,
                    "image_path": image_path,
                    "source_log_row": row_index,
                })
            continue

        raw_image = first_present(row, ["image_path", "page_path", "png_path", "rendered_image", "output_png", "path"])
        if raw_image is None:
            continue
        image_path = resolve_path(raw_image, base_dir=batch_dir)
        fallback_page_number = int(first_present(row, ["page_number", "page", "page_index"]) or 1)
        page_number = page_number_from_path(image_path, fallback_page_number)
        pages.append({
            "score_id": score_id,
            "source_path": str(source_path) if source_path else None,
            "page_id": str(first_present(row, ["page_id"]) or image_path.stem),
            "page_number": page_number,
            "image_path": image_path,
            "source_log_row": row_index,
        })

    return pages


def coefficients_from_points(src: list[tuple[float, float]], dst: list[tuple[float, float]]) -> list[float]:
    """Return PIL perspective coefficients mapping output dst to input src."""
    matrix: list[list[float]] = []
    vector: list[float] = []
    for (x_src, y_src), (x_dst, y_dst) in zip(src, dst):
        matrix.append([x_dst, y_dst, 1, 0, 0, 0, -x_src * x_dst, -x_src * y_dst])
        matrix.append([0, 0, 0, x_dst, y_dst, 1, -y_src * x_dst, -y_src * y_dst])
        vector.extend([x_src, y_src])
    coeffs = np.linalg.solve(np.asarray(matrix, dtype=np.float64), np.asarray(vector, dtype=np.float64))
    return coeffs.tolist()


def bilinear_sample(arr: np.ndarray, sample_x: np.ndarray, sample_y: np.ndarray, *, fill: float = 1.0) -> np.ndarray:
    """Vectorized bilinear sampler for [H,W] or [H,W,C] float arrays in [0,1]."""
    src = np.asarray(arr, dtype=np.float32)
    h, w = src.shape[:2]

    x = np.asarray(sample_x, dtype=np.float32)
    y = np.asarray(sample_y, dtype=np.float32)

    valid = (x >= 0.0) & (x <= w - 1) & (y >= 0.0) & (y <= h - 1)

    x0 = np.floor(np.clip(x, 0, w - 1)).astype(np.int32)
    y0 = np.floor(np.clip(y, 0, h - 1)).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)

    wx = (x - x0).astype(np.float32)
    wy = (y - y0).astype(np.float32)

    if src.ndim == 2:
        ia = src[y0, x0]
        ib = src[y0, x1]
        ic = src[y1, x0]
        id_ = src[y1, x1]
        out = (1.0 - wx) * (1.0 - wy) * ia + wx * (1.0 - wy) * ib + (1.0 - wx) * wy * ic + wx * wy * id_
        return np.where(valid, out, np.float32(fill)).astype(np.float32)

    ia = src[y0, x0, :]
    ib = src[y0, x1, :]
    ic = src[y1, x0, :]
    id_ = src[y1, x1, :]
    out = (
        (1.0 - wx)[:, :, None] * (1.0 - wy)[:, :, None] * ia
        + wx[:, :, None] * (1.0 - wy)[:, :, None] * ib
        + (1.0 - wx)[:, :, None] * wy[:, :, None] * ic
        + wx[:, :, None] * wy[:, :, None] * id_
    )
    out[~valid] = np.float32(fill)
    return out.astype(np.float32)


def coordinate_grid(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    grid_x = np.arange(w, dtype=np.float32)[None, :].repeat(h, axis=0)
    grid_y = np.arange(h, dtype=np.float32)[:, None].repeat(w, axis=1)
    return grid_x, grid_y


def warp_image_with_displacement(image: Image.Image, disp_x: np.ndarray, disp_y: np.ndarray, *, fill: int = 255) -> Image.Image:
    arr = np.asarray(image, dtype=np.float32) / 255.0
    h, w = arr.shape[:2]
    grid_x, grid_y = coordinate_grid((h, w))
    out = bilinear_sample(arr, grid_x + disp_x, grid_y + disp_y, fill=fill / 255.0)
    return Image.fromarray(np.clip(out * 255.0, 0.0, 255.0).astype(np.uint8), mode=image.mode)


def sample_multiscale_signed_field(
    shape: tuple[int, int],
    rng_np: np.random.Generator,
    *,
    scales: list[int],
    weights: list[float] | None = None,
    blur_factor: float = 0.35,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """
    Stack resized random fields at specified feature scales.

    For page images this is much cheaper than repeated FFT noise and gives direct
    control over micro/fine/medium/coarse/macro structure.
    """
    h, w = shape
    total = np.zeros((h, w), dtype=np.float32)
    metadata: list[dict[str, float]] = []
    if weights is None:
        weights = [1.0] * len(scales)

    for scale, base_weight in zip(scales, weights):
        scale = max(2, int(scale))
        small_h = max(2, math.ceil(h / scale))
        small_w = max(2, math.ceil(w / scale))
        base = rng_np.normal(0.0, 1.0, size=(small_h, small_w)).astype(np.float32)
        field = resize_field(base, (h, w), blur_radius=max(0.0, scale * blur_factor / 12.0))
        weight = float(base_weight) * float(rng_np.uniform(0.75, 1.25))
        total += weight * field
        metadata.append({"scale_px": float(scale), "weight": float(weight)})

    return signed_unit(total), metadata


def sample_vector_color_field(
    shape: tuple[int, int],
    rng_np: np.random.Generator,
    *,
    scales: list[int],
    strength: float,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """
    Vector-valued color noise: the noise itself changes color across space.

    This is not one monochrome field with a global tint. Each RGB channel gets a
    correlated but distinct multiscale field, then we mix in a shared luminance
    component so it still behaves like physical document variation.
    """
    shared, meta = sample_multiscale_signed_field(
        shape,
        rng_np,
        scales=scales,
        weights=[1.0 for _ in scales],
        blur_factor=0.32,
    )
    channels = []
    channel_weights = []
    for _ in range(3):
        channel, _ = sample_multiscale_signed_field(
            shape,
            rng_np,
            scales=scales,
            weights=[float(rng_np.uniform(0.65, 1.35)) for _ in scales],
            blur_factor=0.28,
        )
        mix = signed_unit(0.55 * shared + 0.45 * channel)
        channels.append(mix)
        channel_weights.append(float(rng_np.uniform(0.75, 1.25)))
    field = np.stack(channels, axis=2).astype(np.float32)
    field *= np.asarray(channel_weights, dtype=np.float32)[None, None, :]
    field = np.clip(field * float(strength), -1.0, 1.0).astype(np.float32)
    return field, meta


def sample_multiscale_warp(
    shape: tuple[int, int],
    rng: random.Random,
    rng_np: np.random.Generator,
    amplitude_px: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Multi-scale local page warping.

    Fine:   20-80 px wavelength, roughly 0.2-1 px displacement.
    Medium: 80-250 px wavelength, roughly 1-4 px displacement.
    Coarse: 250-900 px wavelength, roughly 4-15 px displacement.

    All displacement components are summed, then the image is resampled once.
    """
    amplitude_px = float(max(0.0, amplitude_px))
    if amplitude_px <= 0.0:
        h, w = shape
        return np.zeros((h, w), dtype=np.float32), np.zeros((h, w), dtype=np.float32), {"amplitude_px": 0.0}

    fine_x, fine_x_meta = sample_multiscale_signed_field(shape, rng_np, scales=[20, 40, 80], weights=[1.00, 0.80, 0.55], blur_factor=0.45)
    fine_y, fine_y_meta = sample_multiscale_signed_field(shape, rng_np, scales=[24, 48, 78], weights=[1.00, 0.78, 0.52], blur_factor=0.45)

    med_x, med_x_meta = sample_multiscale_signed_field(shape, rng_np, scales=[90, 150, 240], weights=[1.00, 0.85, 0.70], blur_factor=0.55)
    med_y, med_y_meta = sample_multiscale_signed_field(shape, rng_np, scales=[80, 160, 250], weights=[1.00, 0.90, 0.72], blur_factor=0.55)

    coarse_x, coarse_x_meta = sample_multiscale_signed_field(shape, rng_np, scales=[280, 460, 820], weights=[1.00, 0.80, 0.62], blur_factor=0.65)
    coarse_y, coarse_y_meta = sample_multiscale_signed_field(shape, rng_np, scales=[260, 520, 900], weights=[1.00, 0.82, 0.64], blur_factor=0.65)

    # Ripple/buckle component, cheap and visibly page-like.
    h, w = shape
    x = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
    y = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
    angle = rng.uniform(0.0, 2.0 * math.pi)
    axis = math.cos(angle) * x + math.sin(angle) * y
    orth = -math.sin(angle) * x + math.cos(angle) * y
    ripple = np.sin(rng.uniform(1.2, 3.8) * math.pi * axis + rng.uniform(0.0, 2.0 * math.pi))
    band = np.exp(-((orth - rng.uniform(-0.45, 0.45)) ** 2) / (2.0 * rng.uniform(0.12, 0.35) ** 2)).astype(np.float32)
    ripple = signed_unit(ripple * band)

    # Amplitude split. These ratios approximate the scale meanings above.
    fine_amp = min(1.0, amplitude_px * 0.12)
    medium_amp = min(4.0, amplitude_px * 0.36)
    coarse_amp = amplitude_px * 0.68
    ripple_amp = amplitude_px * 0.28

    disp_x = fine_amp * fine_x + medium_amp * med_x + coarse_amp * coarse_x + ripple_amp * ripple
    disp_y = fine_amp * fine_y + medium_amp * med_y + coarse_amp * coarse_y + ripple_amp * ripple

    return disp_x.astype(np.float32), disp_y.astype(np.float32), {
        "amplitude_px": float(amplitude_px),
        "fine": {"amplitude_px": float(fine_amp), "x_scales": fine_x_meta, "y_scales": fine_y_meta},
        "medium": {"amplitude_px": float(medium_amp), "x_scales": med_x_meta, "y_scales": med_y_meta},
        "coarse": {"amplitude_px": float(coarse_amp), "x_scales": coarse_x_meta, "y_scales": coarse_y_meta},
        "ripple": {"amplitude_px": float(ripple_amp), "angle_radians": float(angle)},
        "resampling": "single bilinear pass after summing all local warp fields",
    }



def smooth_vertex_noise(rows: int, cols: int, rng_np: np.random.Generator, passes: int = 2) -> np.ndarray:
    """Small helper for fast grid/mesh displacement fields."""
    field = rng_np.normal(0.0, 1.0, size=(rows, cols)).astype(np.float32)
    for _ in range(max(0, int(passes))):
        field = (
            field
            + np.roll(field, 1, axis=0)
            + np.roll(field, -1, axis=0)
            + np.roll(field, 1, axis=1)
            + np.roll(field, -1, axis=1)
        ) / 5.0
    field -= float(field.mean())
    field /= float(field.std() + 1e-6)
    return np.clip(field / 2.5, -1.0, 1.0).astype(np.float32)


def multiscale_mesh_offsets(
    rows: int,
    cols: int,
    amplitude_px: float,
    rng: random.Random,
    rng_np: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Fast multi-scale warp at mesh vertices, not full-resolution remap.

    Fine:   small vertex jitter, approx 20-80 px local bends.
    Medium: grid-scale bending, approx 80-250 px page buckling.
    Coarse: smooth global-ish bending, approx 250-900 px curl.
    """
    rr = np.linspace(-1.0, 1.0, rows, dtype=np.float32)[:, None]
    cc = np.linspace(-1.0, 1.0, cols, dtype=np.float32)[None, :]

    fine_x = smooth_vertex_noise(rows, cols, rng_np, passes=0)
    fine_y = smooth_vertex_noise(rows, cols, rng_np, passes=0)
    medium_x = smooth_vertex_noise(rows, cols, rng_np, passes=2)
    medium_y = smooth_vertex_noise(rows, cols, rng_np, passes=2)
    coarse_x = smooth_vertex_noise(rows, cols, rng_np, passes=5)
    coarse_y = smooth_vertex_noise(rows, cols, rng_np, passes=5)

    angle = rng.uniform(0.0, 2.0 * math.pi)
    axis = math.cos(angle) * cc + math.sin(angle) * rr
    orth = -math.sin(angle) * cc + math.cos(angle) * rr
    ripple = np.sin(rng.uniform(1.2, 3.5) * math.pi * axis + rng.uniform(0, 2 * math.pi))
    band = np.exp(-((orth - rng.uniform(-0.35, 0.35)) ** 2) / (2.0 * rng.uniform(0.18, 0.45) ** 2)).astype(np.float32)
    ripple = ripple * band

    amp = float(amplitude_px)
    # v9: fine/local warp was too weak in v8. Increase it without changing the
    # broad page curl. Fine warp remains bounded so staff lines wobble instead of
    # tearing.
    fine_amp = min(4.0, 0.30 * amp)
    medium_amp = min(7.5, 0.42 * amp)
    coarse_amp = 0.66 * amp
    ripple_amp = 0.34 * amp

    dx = fine_amp * fine_x + medium_amp * medium_x + coarse_amp * coarse_x + ripple_amp * ripple
    dy = fine_amp * fine_y + medium_amp * medium_y + coarse_amp * coarse_y + ripple_amp * ripple

    # Keep page edges less explosive but not fixed; real camera/page curl still moves edges.
    edge_damp_y = np.minimum(np.minimum(np.linspace(0.45, 1.0, rows), np.linspace(1.0, 0.45, rows)), 1.0)[:, None]
    edge_damp_x = np.minimum(np.minimum(np.linspace(0.45, 1.0, cols), np.linspace(1.0, 0.45, cols)), 1.0)[None, :]
    damp = np.minimum(edge_damp_x, edge_damp_y).astype(np.float32)
    dx *= damp
    dy *= damp

    return dx.astype(np.float32), dy.astype(np.float32), {
        "amplitude_px": amp,
        "fine_amplitude_px": float(fine_amp),
        "medium_amplitude_px": float(medium_amp),
        "coarse_amplitude_px": float(coarse_amp),
        "ripple_amplitude_px": float(ripple_amp),
        "mesh_rows": int(rows),
        "mesh_cols": int(cols),
        "implementation": "PIL Image.Transform.MESH single-pass local warp",
    }


def apply_fast_mesh_warp(
    image: Image.Image,
    rng: random.Random,
    rng_np: np.random.Generator,
    amplitude_px: float,
    *,
    cell_px: int = 96,
    fill: int | tuple[int, int, int] = 255,
) -> tuple[Image.Image, dict[str, Any]]:
    """
    Fast local warping using PIL's C-level mesh transform.

    This is much faster than full-resolution Python/NumPy bilinear remapping.
    """
    w, h = image.size
    amplitude_px = float(amplitude_px)
    if amplitude_px <= 0.05:
        return image, {"amplitude_px": 0.0, "skipped": True}

    xs = list(range(0, w, int(cell_px)))
    ys = list(range(0, h, int(cell_px)))
    if xs[-1] != w:
        xs.append(w)
    if ys[-1] != h:
        ys.append(h)

    rows = len(ys)
    cols = len(xs)
    dx, dy, meta = multiscale_mesh_offsets(rows, cols, amplitude_px, rng, rng_np)

    mesh = []
    for j in range(rows - 1):
        for i in range(cols - 1):
            x0, x1 = xs[i], xs[i + 1]
            y0, y1 = ys[j], ys[j + 1]
            bbox = (x0, y0, x1, y1)
            # Quad is source coordinates for the destination rectangle corners.
            # PIL MESH/QUAD source order is:
            # upper-left, lower-left, lower-right, upper-right.
            # The previous v7 order used upper-left, upper-right, lower-right,
            # lower-left, which twists each cell and creates the broken sideways
            # grid artifacts you observed.
            quad = (
                float(x0 + dx[j, i]), float(y0 + dy[j, i]),
                float(x0 + dx[j + 1, i]), float(y1 + dy[j + 1, i]),
                float(x1 + dx[j + 1, i + 1]), float(y1 + dy[j + 1, i + 1]),
                float(x1 + dx[j, i + 1]), float(y0 + dy[j, i + 1]),
            )
            mesh.append((bbox, quad))

    out = image.transform(
        image.size,
        Image.Transform.MESH,
        mesh,
        resample=Image.Resampling.BICUBIC,
        fillcolor=fill,
    )
    meta["cell_px"] = int(cell_px)
    return out, meta



def estimate_local_paper_fill(image: Image.Image, band_frac: float = 0.025) -> int | tuple[int, int, int]:
    """
    Estimate the local page/background color for geometric transform fill.

    Uses a border band and prefers bright/background-like pixels so staff/notation
    near the border does not bias the fill toward black. For RGB images, returns
    an RGB tuple. For grayscale images, returns an integer.

    This is used for pixels sampled outside the transformed source image.
    """
    if image.mode in {"RGB", "RGBA"}:
        arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
        h, w = arr.shape[:2]
        band = max(1, int(round(min(h, w) * band_frac)))
        border = np.concatenate([
            arr[:band, :, :].reshape(-1, 3),
            arr[-band:, :, :].reshape(-1, 3),
            arr[:, :band, :].reshape(-1, 3),
            arr[:, -band:, :].reshape(-1, 3),
        ], axis=0)

        # Prefer paper/background pixels, not notation.
        luma = (
            0.2126 * border[:, 0].astype(np.float32)
            + 0.7152 * border[:, 1].astype(np.float32)
            + 0.0722 * border[:, 2].astype(np.float32)
        )
        cutoff = max(120.0, float(np.percentile(luma, 55)))
        bright = border[luma >= cutoff]
        sample = bright if len(bright) >= max(32, len(border) // 20) else border
        fill = np.median(sample, axis=0)
        return tuple(int(np.clip(round(x), 0, 255)) for x in fill)

    arr_l = np.asarray(image.convert("L"), dtype=np.uint8)
    h, w = arr_l.shape
    band = max(1, int(round(min(h, w) * band_frac)))
    border = np.concatenate([
        arr_l[:band, :].reshape(-1),
        arr_l[-band:, :].reshape(-1),
        arr_l[:, :band].reshape(-1),
        arr_l[:, -band:].reshape(-1),
    ], axis=0)
    cutoff = max(120.0, float(np.percentile(border.astype(np.float32), 55)))
    bright = border[border.astype(np.float32) >= cutoff]
    sample = bright if len(bright) >= max(32, len(border) // 20) else border
    return int(np.clip(round(float(np.median(sample))), 0, 255))


def apply_geometry(image: Image.Image, rng: random.Random, cfg: dict[str, float], strength: float) -> tuple[Image.Image, dict[str, Any]]:
    w, h = image.size
    fill_color = estimate_local_paper_fill(image)
    rotation = rng.uniform(-cfg["rotation_deg"], cfg["rotation_deg"]) * strength
    translate_x = rng.uniform(-cfg["translation_frac"], cfg["translation_frac"]) * strength * w
    translate_y = rng.uniform(-cfg["translation_frac"], cfg["translation_frac"]) * strength * h
    scale = 1.0 + rng.uniform(-cfg["scale_frac"], cfg["scale_frac"]) * strength
    shear_x = rng.uniform(-cfg["skew_frac"], cfg["skew_frac"]) * strength
    shear_y = rng.uniform(-cfg["skew_frac"], cfg["skew_frac"]) * strength

    out = image.rotate(rotation, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=fill_color)
    out = out.transform(out.size, Image.Transform.AFFINE, (scale, shear_x, translate_x, shear_y, scale, translate_y), resample=Image.Resampling.BICUBIC, fillcolor=fill_color)

    max_dx = cfg["perspective_frac"] * strength * w
    max_dy = cfg["perspective_frac"] * strength * h
    src_pts = [(0, 0), (w, 0), (w, h), (0, h)]
    dst_pts = [
        (rng.uniform(-max_dx, max_dx), rng.uniform(-max_dy, max_dy)),
        (w + rng.uniform(-max_dx, max_dx), rng.uniform(-max_dy, max_dy)),
        (w + rng.uniform(-max_dx, max_dx), h + rng.uniform(-max_dy, max_dy)),
        (rng.uniform(-max_dx, max_dx), h + rng.uniform(-max_dy, max_dy)),
    ]
    coeffs = coefficients_from_points(src_pts, dst_pts)
    out = out.transform(out.size, Image.Transform.PERSPECTIVE, coeffs, resample=Image.Resampling.BICUBIC, fillcolor=fill_color)

    local_warp_px = float(cfg.get("local_warp_px", 0.0)) * float(strength)
    local_warp_meta: dict[str, Any] | None = None
    if local_warp_px > 0.05:
        rng_np = np.random.default_rng(rng.randint(0, 2**31 - 1))
        out, local_warp_meta = apply_fast_mesh_warp(
            out,
            rng,
            rng_np,
            local_warp_px,
            # v9: smaller mesh cells make the fine/local warp scale visible.
            # This is still PIL C-level MESH, much faster than full-resolution
            # Python/NumPy remapping.
            cell_px=56 if max(w, h) >= 1400 else 44,
            fill=fill_color,
        )

    params = {
        "rotation_degrees": rotation,
        "translation_px": [translate_x, translate_y],
        "scale": scale,
        "shear": [shear_x, shear_y],
        "perspective_dst_corners": dst_pts,
        "local_warp": local_warp_meta,
        "outside_fill_color": fill_color,
        "outside_fill_policy": "median bright border paper/background color",
    }
    return out, params


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0, 255).astype(np.uint8)


def low_frequency_noise(shape: tuple[int, int], rng_np: np.random.Generator, scale: int = 16) -> np.ndarray:
    h, w = shape
    small_h = max(2, math.ceil(h / scale))
    small_w = max(2, math.ceil(w / scale))
    small = rng_np.normal(0.0, 1.0, size=(small_h, small_w)).astype(np.float32)
    img = Image.fromarray(((small - small.min()) / (np.ptp(small) + 1e-6) * 255).astype(np.uint8), mode="L")
    img = img.resize((w, h), Image.Resampling.BICUBIC).filter(ImageFilter.GaussianBlur(radius=max(1, scale / 4)))
    noise = np.asarray(img, dtype=np.float32) / 255.0
    return noise * 2.0 - 1.0


def apply_paper_ink(image: Image.Image, rng: random.Random, rng_np: np.random.Generator, cfg: dict[str, float], strength: float) -> tuple[Image.Image, dict[str, Any]]:
    arr = np.asarray(image.convert("L"), dtype=np.float32)
    h, w = arr.shape
    texture = low_frequency_noise((h, w), rng_np, scale=24)
    paper_texture_strength = cfg["paper_texture"] * strength
    arr = arr + 255.0 * paper_texture_strength * texture

    # Light ink thickening/thinning via min/max filters on dark foreground.
    ink_mode = rng.choice(["thicken", "thin", "none"])
    radius = 3 if strength > 0.5 else 1
    img = Image.fromarray(normalize_to_uint8(arr), mode="L")
    if ink_mode == "thicken" and rng.random() < cfg["ink_strength"] * strength:
        img = img.filter(ImageFilter.MinFilter(radius))
    elif ink_mode == "thin" and rng.random() < cfg["ink_strength"] * strength:
        img = img.filter(ImageFilter.MaxFilter(radius))

    contrast = 1.0 - rng.uniform(0.0, cfg["contrast_drop"]) * strength
    brightness = 1.0 + rng.uniform(-cfg["brightness_shift"], cfg["brightness_shift"]) * strength
    img = ImageEnhance.Contrast(img).enhance(max(0.1, contrast))
    img = ImageEnhance.Brightness(img).enhance(max(0.1, brightness))
    return img, {"paper_texture_strength": paper_texture_strength, "ink_mode": ink_mode, "contrast": contrast, "brightness": brightness}


def apply_lighting(image: Image.Image, rng: random.Random, cfg: dict[str, float], strength: float) -> tuple[Image.Image, dict[str, Any]]:
    """
    Apply lighting/shadow without destroying color.

    Earlier versions converted to L here, which erased all page coloring after the
    artifact stage. This preserves RGB by multiplying each channel by the same
    illumination field.
    """
    rgb_input = image.mode in {"RGB", "RGBA"}
    if rgb_input:
        arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        h, w = arr.shape[:2]
    else:
        arr = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        h, w = arr.shape

    x = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
    y = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
    angle = rng.uniform(0.0, 2.0 * math.pi)
    gradient = math.cos(angle) * x + math.sin(angle) * y
    gradient = (gradient - gradient.min()) / (np.ptp(gradient) + 1e-6)
    grad_strength = rng.uniform(0.0, cfg["shadow_strength"]) * strength
    field = 1.0 - grad_strength * gradient

    # Optional soft shadow band.
    band_center = rng.uniform(-0.7, 0.7)
    band_width = rng.uniform(0.18, 0.45)
    coord = math.cos(angle + math.pi / 2.0) * x + math.sin(angle + math.pi / 2.0) * y
    band = np.exp(-((coord - band_center) ** 2) / (2.0 * band_width**2))
    band_strength = rng.uniform(0.0, cfg["shadow_strength"]) * strength
    field *= 1.0 - band_strength * band

    # Mild colored illumination variation. Keep it subtle; page color should come
    # mostly from the artifact/page-coloring stage.
    if rgb_input:
        illuminant = np.ones(3, dtype=np.float32)
        if rng.random() < 0.45:
            warm_cool = rng.uniform(-0.06, 0.06) * strength
            illuminant = np.array([1.0 + warm_cool, 1.0, 1.0 - warm_cool], dtype=np.float32)
        out = np.clip(arr * field[:, :, None] * illuminant[None, None, :], 0.0, 1.0)
        return Image.fromarray((out * 255.0).round().astype(np.uint8), mode="RGB"), {
            "gradient_strength": grad_strength,
            "shadow_band_strength": band_strength,
            "angle_radians": angle,
            "preserved_rgb": True,
            "illuminant_rgb": [float(x) for x in illuminant],
        }

    out = normalize_to_uint8(arr * field * 255.0)
    return Image.fromarray(out, mode="L"), {
        "gradient_strength": grad_strength,
        "shadow_band_strength": band_strength,
        "angle_radians": angle,
        "preserved_rgb": False,
    }



def resize_field(field: np.ndarray, shape: tuple[int, int], *, blur_radius: float = 0.0) -> np.ndarray:
    """Resize a small procedural field to [H, W] and normalize it to [-1, 1]."""
    h, w = shape
    arr = np.asarray(field, dtype=np.float32)
    lo = float(arr.min())
    hi = float(arr.max())
    if hi - lo < 1e-6:
        arr = np.zeros_like(arr, dtype=np.float32)
    else:
        arr = (arr - lo) / (hi - lo)
    img = Image.fromarray((arr * 255.0).astype(np.uint8), mode="L")
    img = img.resize((w, h), Image.Resampling.BICUBIC)
    if blur_radius > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    out = np.asarray(img, dtype=np.float32) / 255.0
    return out * 2.0 - 1.0


def fbm_field(shape: tuple[int, int], rng_np: np.random.Generator, *, octaves: int = 5) -> np.ndarray:
    """
    Cheap fractal field made from several resized Gaussian fields.

    This replaces the old single Fourier texture with a broader, nonstationary
    procedural basis that can support paper-like, stain-like, and shadow-like
    samples depending on later sampled masks and colors.
    """
    h, w = shape
    total = np.zeros((h, w), dtype=np.float32)
    amplitude = 1.0
    amplitude_sum = 0.0
    for octave in range(octaves):
        scale = int(max(4, min(h, w) / (2 ** (octave + 2))))
        small_h = max(2, math.ceil(h / scale))
        small_w = max(2, math.ceil(w / scale))
        small = rng_np.normal(0.0, 1.0, size=(small_h, small_w)).astype(np.float32)
        total += amplitude * resize_field(small, (h, w), blur_radius=max(0.5, scale / 6.0))
        amplitude_sum += amplitude
        amplitude *= 0.55
    total /= max(amplitude_sum, 1e-6)
    total -= float(total.mean())
    total /= float(total.std() + 1e-6)
    return np.clip(total / 3.0, -1.0, 1.0).astype(np.float32)



def signed_unit(field: np.ndarray) -> np.ndarray:
    arr = np.asarray(field, dtype=np.float32)
    arr = arr - float(arr.mean())
    arr = arr / float(arr.std() + 1e-6)
    return np.clip(arr / 3.0, -1.0, 1.0).astype(np.float32)


def fine_grain_field(shape: tuple[int, int], rng_np: np.random.Generator, *, blur_radius: float = 0.0) -> np.ndarray:
    """High-frequency signed field for visible paper/sensor/ink grain."""
    h, w = shape
    field = rng_np.normal(0.0, 1.0, size=(h, w)).astype(np.float32)
    if blur_radius > 0:
        img = Image.fromarray(((signed_unit(field) + 1.0) * 127.5).astype(np.uint8), mode="L")
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        field = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    return signed_unit(field)


def sparse_salt_pepper_field(shape: tuple[int, int], rng_np: np.random.Generator, density: float, *, blur_radius: float = 0.45) -> np.ndarray:
    """Sparse signed specks; unlike a broad field, this creates visible dirt/scan grit."""
    h, w = shape
    density = max(0.0, float(density))
    mask = rng_np.random((h, w), dtype=np.float32) < density
    signs = rng_np.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(h, w))
    field = np.zeros((h, w), dtype=np.float32)
    field[mask] = signs[mask]
    if blur_radius > 0:
        img = Image.fromarray(((field + 1.0) * 127.5).astype(np.uint8), mode="L")
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        field = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    return np.clip(field, -1.0, 1.0).astype(np.float32)


def endpoint_texture_rgb(rgb: np.ndarray, field: np.ndarray, epsilon: float, color_bias: np.ndarray | None = None) -> np.ndarray:
    """Original spectral-noise idea generalized to RGB and stackable texture fields."""
    z = np.clip(np.asarray(field, dtype=np.float32), -1.0, 1.0)
    eps = float(max(0.0, epsilon))
    pos = np.maximum(z, 0.0)[:, :, None]
    neg = np.maximum(-z, 0.0)[:, :, None]
    if color_bias is None:
        color_bias = np.ones(3, dtype=np.float32)
    bias = np.asarray(color_bias, dtype=np.float32)[None, None, :]
    toward_white = (1.0 - rgb) * bias
    toward_black = rgb * bias
    out = rgb + eps * pos * toward_white - eps * neg * toward_black
    return np.clip(out, 0.0, 1.0).astype(np.float32)

def rotated_coordinates(shape: tuple[int, int], angle: float) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    y = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
    x = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
    ca = math.cos(angle)
    sa = math.sin(angle)
    xr = ca * x + sa * y
    yr = -sa * x + ca * y
    return xr, yr


def gaussian_blob_mask(shape: tuple[int, int], rng: random.Random) -> tuple[np.ndarray, dict[str, Any]]:
    h, w = shape
    angle = rng.uniform(0.0, 2.0 * math.pi)
    xr, yr = rotated_coordinates(shape, angle)
    cx = rng.uniform(-0.75, 0.75)
    cy = rng.uniform(-0.75, 0.75)
    sx = rng.uniform(0.08, 0.55)
    sy = rng.uniform(0.04, 0.35)
    exponent = ((xr - cx) / sx) ** 2 + ((yr - cy) / sy) ** 2
    mask = np.exp(-0.5 * exponent).astype(np.float32)
    softness = rng.uniform(0.55, 1.25)
    mask = np.power(mask, softness).astype(np.float32)
    return mask, {"center": [cx, cy], "sigma": [sx, sy], "angle_radians": angle, "softness": softness}


def ribbon_mask(shape: tuple[int, int], rng: random.Random, rng_np: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
    angle = rng.uniform(0.0, 2.0 * math.pi)
    xr, yr = rotated_coordinates(shape, angle)
    center = rng.uniform(-0.75, 0.75)
    width = rng.uniform(0.025, 0.18)
    waviness = rng.uniform(0.0, 0.12)
    frequency = rng.uniform(1.0, 4.0)
    phase = rng.uniform(0.0, 2.0 * math.pi)
    curve = center + waviness * np.sin(frequency * math.pi * xr + phase)
    mask = np.exp(-((yr - curve) ** 2) / (2.0 * width**2)).astype(np.float32)
    texture = 0.75 + 0.25 * fbm_field(shape, rng_np, octaves=3)
    mask = np.clip(mask * texture, 0.0, 1.0).astype(np.float32)
    return mask, {"center": center, "width": width, "angle_radians": angle, "waviness": waviness, "frequency": frequency}


def speckle_mask(shape: tuple[int, int], rng: random.Random, rng_np: np.random.Generator, strength: float) -> tuple[np.ndarray, dict[str, Any]]:
    h, w = shape
    density = rng.uniform(0.00003, 0.00030) * (0.5 + strength)
    n = max(1, int(h * w * density))
    mask = np.zeros((h, w), dtype=np.float32)
    ys = rng_np.integers(0, h, size=n)
    xs = rng_np.integers(0, w, size=n)
    vals = rng_np.uniform(0.35, 1.0, size=n).astype(np.float32)
    mask[ys, xs] = vals
    radius = rng.uniform(0.4, 2.2) * (0.7 + strength)
    img = Image.fromarray((mask * 255.0).astype(np.uint8), mode="L").filter(ImageFilter.GaussianBlur(radius=radius))
    mask = np.asarray(img, dtype=np.float32) / 255.0
    if mask.max() > 0:
        mask = mask / float(mask.max())
    return mask.astype(np.float32), {"count": n, "blur_radius": radius}


def random_artifact_color(rng: random.Random, mode: str, strength: float) -> np.ndarray:
    """Return a strong RGB artifact shift with genuinely broad hue support."""
    if mode == "shadow":
        h = rng.random()
        s = rng.uniform(0.18, 0.70)
        v = rng.uniform(0.12, 0.40)
        rgb = np.asarray(colorsys.hsv_to_rgb(h, s, v), dtype=np.float32)
        base = -rgb
    elif mode == "highlight":
        h = rng.random()
        s = rng.uniform(0.10, 0.55)
        v = rng.uniform(0.70, 1.00)
        base = np.asarray(colorsys.hsv_to_rgb(h, s, v), dtype=np.float32) - 0.5
    else:
        h = rng.random()
        s = rng.uniform(0.20, 0.90)
        v = rng.uniform(0.35, 0.95)
        rgb = np.asarray(colorsys.hsv_to_rgb(h, s, v), dtype=np.float32)
        if mode in {"stain", "paper"} and rng.random() < 0.70:
            rgb = 0.55 * rgb + 0.45 * np.asarray(colorsys.hsv_to_rgb(rng.uniform(0.06, 0.16), rng.uniform(0.25, 0.65), rng.uniform(0.55, 0.95)), dtype=np.float32)
        sign = -1.0 if mode in {"smudge", "dirt", "crease", "speckle"} and rng.random() < 0.45 else 1.0
        base = sign * (rgb - 0.5) * 2.0
    return np.clip(base * float(strength), -1.0, 1.0).astype(np.float32)


def apply_unified_artifact_field(
    image: Image.Image,
    rng: random.Random,
    rng_np: np.random.Generator,
    cfg: dict[str, float],
    strength: float,
) -> tuple[Image.Image, dict[str, Any]]:
    """
    v14: direct-lightening subtractive/additive semantics.

    1. Subtractive noise is applied first on the black/white page:
       normal noise is filtered by encountered blackness. Where it encounters
       black ink, it can lighten toward white, including full white.

    2. Page coloring is visible again, but with moderate saturation.

    3. Additive pigment is softer except for black pigment, which can still
       reach full opacity/full black.
    """
    gray0 = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    h, w = gray0.shape
    ink0 = np.clip(1.0 - gray0, 0.0, 1.0)

    global_strength = float(cfg["artifact_strength"] * strength)
    color_strength = float(cfg["artifact_color_strength"] * strength)
    ink_strength = float(cfg["artifact_ink_interaction"] * strength)
    n_events = max(3, int(round(float(cfg["artifact_events"]) * (0.38 + 0.22 * strength))))

    # Shared fields. Keep these limited for speed.
    micro = signed_unit(rng_np.normal(0.0, 1.0, size=(h, w)).astype(np.float32))
    fine, fine_meta = sample_multiscale_signed_field((h, w), rng_np, scales=[2, 5, 10], weights=[1.0, 0.85, 0.65], blur_factor=0.08)
    mid, mid_meta = sample_multiscale_signed_field((h, w), rng_np, scales=[18, 38, 76], weights=[1.0, 0.85, 0.70], blur_factor=0.25)
    large, large_meta = sample_multiscale_signed_field((h, w), rng_np, scales=[110, 230, 460], weights=[1.0, 0.78, 0.58], blur_factor=0.50)

    color_mid, color_mid_meta = sample_vector_color_field((h, w), rng_np, scales=[16, 36, 72], strength=1.0)
    color_large, color_large_meta = sample_vector_color_field((h, w), rng_np, scales=[120, 260, 520], strength=1.0)

    # ------------------------------------------------------------------
    # 1) Subtractive deterioration on the original black/white page.
    # ------------------------------------------------------------------
    # This is now exactly "noise filtered by what it encounters":
    #   noise exists everywhere, but only acts strongly where the source page is
    #   black/dark. If the source pixel is white, the subtractive effect is near zero.
    subtractive_noise = signed_unit(0.42 * micro + 0.72 * fine + 0.52 * mid + 0.12 * large)

    # encountered_black is a continuous mask: 0 on white page, 1 on black ink.
    encountered_black = np.clip((0.88 - gray0) / 0.88, 0.0, 1.0).astype(np.float32)

    # Positive lobes of the noise remove ink. Negative lobes leave it alone here.
    subtractive_lobe = np.maximum(subtractive_noise, 0.0)

    # v13: direct subtractive noise, not a rare branch.
    #
    # This is the exact intended model:
    #   1. Generate normal multiscale noise.
    #   2. Check what source color it encounters.
    #   3. If it encounters black/dark ink, compute how much to lighten it.
    #   4. If it encounters white paper, do essentially nothing.
    #
    # This guarantees that the subtractive branch exists wherever original ink exists.
    # v14: subtractive means direct lightening of the grayscale image.
    # No layer/opacity interpretation here:
    #   gray_new = gray_original + lightening_delta
    # where lightening_delta is zero on white paper and positive only where the
    # original grayscale page is dark/black.
    subtractive_continuous_noise = np.clip((subtractive_noise + 1.0) * 0.5, 0.0, 1.0).astype(np.float32)

    # Fine black-filtered lightening: breaks stems/staff edges/noise-scale notation.
    fine_lighten_noise = signed_unit(0.85 * fine + 0.55 * micro + 0.25 * mid)
    fine_lighten_threshold = rng.uniform(-0.18, 0.18)
    fine_lighten = np.clip(
        (fine_lighten_noise - fine_lighten_threshold) / max(0.08, 1.0 - fine_lighten_threshold),
        0.0,
        1.0,
    ).astype(np.float32)

    # Patch black-filtered lightening: larger note/staff fade regions.
    patch_lighten_noise = signed_unit(0.55 * mid + 0.35 * large + 0.25 * fine)
    patch_lighten_threshold = rng.uniform(-0.14, 0.24)
    patch_lighten = np.clip(
        (patch_lighten_noise - patch_lighten_threshold) / max(0.10, 1.0 - patch_lighten_threshold),
        0.0,
        1.0,
    ).astype(np.float32)

    # Max-white direct-lightening mask, also black-filtered.
    max_white_source = signed_unit(0.65 * fine + 0.45 * mid + 0.20 * micro)
    max_white_threshold = rng.uniform(0.12, 0.42)
    max_white = np.clip(
        (max_white_source - max_white_threshold) / max(0.06, 1.0 - max_white_threshold),
        0.0,
        1.0,
    ).astype(np.float32)

    # This is an intensity delta, not opacity.
    # encountered_black gates it so white paper is not lightened/blotched.
    lighten_delta = encountered_black * (
        ink_strength * (
            0.20
            + 0.48 * subtractive_continuous_noise
            + 0.72 * fine_lighten
            + 0.55 * patch_lighten
            + 0.90 * max_white
        )
    )

    max_white_forced = False
    if rng.random() < min(0.98, 0.70 + 0.35 * strength):
        max_white_forced = True
        forced_lighten = encountered_black * np.maximum(max_white, 0.65 * fine_lighten)
        lighten_delta = np.maximum(
            lighten_delta,
            np.clip(forced_lighten * rng.uniform(0.95, 1.45) * max(0.70, strength), 0.0, 1.0),
        )

    # Direct grayscale lightening. Full white is achieved by clipping at 1.
    gray = np.clip(gray0 + lighten_delta, 0.0, 1.0)

    # Track where original ink was actually lightened. This is only a diagnostic
    # and a guard for later black pigment, not a layer opacity.
    erased_ink_mask = np.clip((gray - gray0) * encountered_black, 0.0, 1.0).astype(np.float32)

    # Compatibility metadata names.
    subtractive_alpha = lighten_delta
    subtractive_base_alpha = encountered_black * ink_strength * (0.20 + 0.48 * subtractive_continuous_noise)
    fine_erasure_mask = encountered_black * fine_lighten
    patch_erasure_mask = encountered_black * patch_lighten
    max_white_mask = encountered_black * max_white
    gray = np.clip(gray, 0.0, 1.0)
    ink = np.clip(1.0 - gray, 0.0, 1.0)

    # ------------------------------------------------------------------
    # 2) Visible, moderately saturated page coloring.
    # ------------------------------------------------------------------
    if rng.random() < 0.66:
        paper_h = rng.uniform(0.06, 0.17)      # warm paper
    elif rng.random() < 0.70:
        paper_h = rng.uniform(0.48, 0.63)      # cool scanner/camera cast
    else:
        paper_h = rng.random()                 # full hue possible
    # v11: v10 was too invisible; raise saturation modestly, but keep it below v9/v8.
    paper_s = rng.uniform(0.045, 0.190)
    paper_v = rng.uniform(0.88, 1.00)
    paper_color = np.asarray(colorsys.hsv_to_rgb(paper_h, paper_s, paper_v), dtype=np.float32)

    # Vector-valued page color noise; visible but not "nebula".
    paper_variation = color_strength * (0.050 * color_mid + 0.070 * color_large)
    paper_rgb = np.clip(paper_color[None, None, :] + paper_variation, 0.0, 1.0)

    ink_color = np.asarray([rng.uniform(0.0, 0.018), rng.uniform(0.0, 0.018), rng.uniform(0.0, 0.022)], dtype=np.float32)
    rgb = paper_rgb * gray[:, :, None] + ink_color[None, None, :] * ink[:, :, None]

    # Fine page texture after coloring.
    rgb = endpoint_texture_rgb(rgb, micro, epsilon=0.014 * global_strength)
    rgb = endpoint_texture_rgb(rgb, fine, epsilon=0.024 * global_strength)
    rgb = endpoint_texture_rgb(rgb, mid, epsilon=0.036 * global_strength)
    rgb = endpoint_texture_rgb(rgb, large, epsilon=0.046 * global_strength)

    full_page_color_mode = rng.random() < (0.14 + 0.14 * strength)
    full_page_color_strength = 0.0
    if full_page_color_mode:
        full_page_color_strength = rng.uniform(0.08, 0.22) * color_strength
        rgb += full_page_color_strength * color_large

    # ------------------------------------------------------------------
    # 3) Additive pigment/noise after page coloring.
    # ------------------------------------------------------------------
    # Additive = deposited material. Black pigment can be opaque; colored pigment
    # is kept relatively transparent.
    specks = sparse_salt_pepper_field(
        (h, w),
        rng_np,
        density=rng.uniform(0.0010, 0.0050) * (0.55 + strength),
        blur_radius=rng.uniform(0.0, 0.50),
    )

    additive_noise = signed_unit(0.38 * micro + 0.62 * fine + 0.56 * mid + 0.28 * specks)
    additive_support = np.clip((additive_noise - rng.uniform(-0.18, 0.30)) / 0.62, 0.0, 1.0).astype(np.float32)

    # Black pigment: can match/merge with black ink and reach max opacity.
    black_pigment_alpha = np.clip(
        ink_strength
        * (
            0.42 * np.maximum(-additive_noise, 0.0) * np.clip(0.25 + ink, 0.0, 1.0)
            + 0.85 * additive_support * rng.uniform(0.25, 0.70)
            + 0.65 * np.clip(specks, 0.0, 1.0)
        ),
        0.0,
        1.0,
    )

    true_black_source = signed_unit(0.55 * fine + 0.55 * mid + 0.42 * specks + 0.18 * micro)
    true_black_threshold = rng.uniform(0.38, 0.70)
    true_black_mask = np.clip((true_black_source - true_black_threshold) / max(0.07, 1.0 - true_black_threshold), 0.0, 1.0).astype(np.float32)
    true_black_mask = np.maximum(true_black_mask, np.clip(specks, 0.0, 1.0) * rng.uniform(0.50, 0.90))

    true_black_forced = False
    if rng.random() < min(0.90, 0.38 + 0.52 * strength):
        true_black_forced = True
        black_pigment_alpha = np.maximum(
            black_pigment_alpha,
            np.clip(true_black_mask * rng.uniform(0.85, 1.25), 0.0, 1.0),
        )

    # Do not immediately refill deliberately erased ink pixels with black pigment.
    black_pigment_alpha = black_pigment_alpha * (1.0 - 0.88 * erased_ink_mask)

    # Apply black pigment by absorption to black.
    rgb = rgb * (1.0 - black_pigment_alpha[:, :, None])

    # Colored additive pigment: much softer than v10. Color varies spatially.
    colored_support = np.clip(additive_support[:, :, None] * rng.uniform(0.10, 0.36) * color_strength, 0.0, 0.38)
    pigment_color_field = np.clip(0.5 + 0.5 * signed_unit(0.60 * color_mid + 0.55 * color_large), 0.0, 1.0)
    rgb = rgb * (1.0 - 0.22 * colored_support) + pigment_color_field * (0.22 * colored_support)

    # Lighting/shadow after pigment.
    tone = signed_unit(0.20 * fine + 0.55 * mid + 0.75 * large)
    rgb *= np.clip(1.0 + 0.30 * global_strength * tone[:, :, None], 0.36, 1.40)

    # Cheap local events. Softer unless black-pigment mode.
    event_metadata: list[dict[str, Any]] = []
    for event_i in range(n_events):
        basis = rng.choice(["blob", "field", "speckle"])
        mode = rng.choice(["black_pigment", "subtractive_patch", "colored_pigment", "shadow", "highlight"])

        if basis == "blob":
            mask, mask_meta = gaussian_blob_mask((h, w), rng)
            mask = np.clip(mask * (0.95 + 0.32 * mid + 0.15 * fine), 0.0, 1.0).astype(np.float32)
        elif basis == "speckle":
            mask = np.clip(specks, 0.0, 1.0)
            mask_meta = {"source": "shared_speckles"}
        else:
            threshold = rng.uniform(-0.35, 0.25)
            source = signed_unit(0.45 * mid + 0.65 * large)
            mask = np.clip((source - threshold) / 0.75, 0.0, 1.0).astype(np.float32)
            mask_meta = {"source": "shared_field", "threshold": float(threshold)}

        alpha = np.clip(mask[:, :, None] * rng.uniform(0.12, 0.58) * global_strength, 0.0, 0.75)
        local_color = np.clip(0.5 + 0.5 * signed_unit(rng.uniform(0.5, 1.0) * color_mid + rng.uniform(0.3, 1.0) * color_large), 0.0, 1.0)

        if mode == "black_pigment":
            # Only black pigment gets high opacity.
            rgb *= np.clip(1.0 - 0.95 * alpha, 0.0, 1.0)
        elif mode == "subtractive_patch":
            # Post-color deterioration only acts where ink remains.
            ink_event = ink[:, :, None]
            rgb = rgb * (1.0 - 0.62 * alpha * ink_event) + np.clip(paper_rgb + 0.14, 0.0, 1.0) * (0.62 * alpha * ink_event)
        elif mode == "colored_pigment":
            # Soft transparent pigment, not aggressive overlay.
            rgb = rgb * (1.0 - 0.18 * alpha) + local_color * (0.18 * alpha)
        elif mode == "shadow":
            rgb *= np.clip(1.0 - 0.42 * alpha, 0.15, 1.0)
        else:
            rgb = rgb * (1.0 - 0.13 * alpha) + np.clip(rgb + 0.24 * local_color, 0.0, 1.0) * (0.13 * alpha)

        event_metadata.append({"event_index": event_i, "basis": basis, "mode": mode, "mask": mask_meta})

    # Show-through.
    showthrough_alpha = rng.uniform(0.010, 0.085) * strength
    if showthrough_alpha > 0.010:
        ghost = image.convert("L").transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        ghost = ghost.rotate(rng.uniform(-3.0, 3.0), resample=Image.Resampling.BICUBIC, fillcolor=255)
        ghost = ghost.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.8, 2.2)))
        ghost_arr = np.asarray(ghost, dtype=np.float32) / 255.0
        ghost_ink = 1.0 - ghost_arr
        rgb -= showthrough_alpha * ghost_ink[:, :, None] * np.clip(paper_color * rng.uniform(0.35, 0.80), 0.0, 1.0)[None, None, :]

    out = np.clip(rgb, 0.0, 1.0)
    return Image.fromarray((out * 255.0).round().astype(np.uint8), mode="RGB"), {
        "model": "unified_nonstationary_artifact_field_v16_local_paper_fill",
        "global_strength": global_strength,
        "color_strength": color_strength,
        "ink_interaction_strength": ink_strength,
        "paper_color_rgb": [float(x) for x in paper_color],
        "ink_color_rgb": [float(x) for x in ink_color],
        "stage_order": [
            "direct_grayscale_lightening_noise_filtered_by_encountered_black",
            "moderate_saturation_page_coloring",
            "soft_additive_colored_pigment_and_high_opacity_black_pigment",
            "geometry_applied_after_artifacts_by_degrade_image",
        ],
        "subtractive_alpha_max": float(np.max(subtractive_alpha)),
        "subtractive_base_alpha_max": float(np.max(subtractive_base_alpha)),
        "erased_ink_mask_max": float(np.max(erased_ink_mask)),
        "fine_erasure_mask_max": float(np.max(fine_erasure_mask)),
        "patch_erasure_mask_max": float(np.max(patch_erasure_mask)),
        "max_white_forced": bool(max_white_forced),
        "max_white_mask_max": float(np.max(max_white_mask)),
        "black_pigment_alpha_max": float(np.max(black_pigment_alpha)),
        "true_black_forced": bool(true_black_forced),
        "true_black_threshold": float(true_black_threshold),
        "true_black_mask_max": float(np.max(true_black_mask)),
        "full_page_color_mode": bool(full_page_color_mode),
        "full_page_color_strength": float(full_page_color_strength),
        "multiscale_appearance": {
            "micro_px": "1-4",
            "fine_px": "4-12",
            "medium_px": "12-80",
            "large_px": "110-520",
            "scalar_fields": {"fine": fine_meta, "medium": mid_meta, "large": large_meta},
            "vector_color_fields": {"medium": color_mid_meta, "large": color_large_meta},
        },
        "events": event_metadata,
        "showthrough_alpha": float(showthrough_alpha),
        "output_mode": "RGB",
        "bugfix_notes": [
            "v14 subtractive stage is direct grayscale lightening, not layer opacity/removal",
            "lightening_delta is normal multiscale noise filtered by encountered blackness",
            "subtractive stage directly increases grayscale intensity and can clip to full white before page coloring",
            "page coloring saturation raised relative to v10 but kept moderate",
            "colored additive pigment is softer; black pigment alone can reach high opacity",
        ],
    }

def apply_blur(image: Image.Image, rng: random.Random, cfg: dict[str, float], strength: float) -> tuple[Image.Image, dict[str, Any]]:
    sigma = rng.uniform(0.0, cfg["blur_sigma"]) * strength
    if sigma > 0:
        image = image.filter(ImageFilter.GaussianBlur(radius=sigma))
    return image, {"gaussian_sigma": sigma}


def apply_compression(image: Image.Image, rng: random.Random, cfg: dict[str, float], strength: float) -> tuple[Image.Image, dict[str, Any]]:
    q_min = int(cfg["jpeg_quality_min"])
    q_max = int(cfg["jpeg_quality_max"])
    # Higher strength biases toward lower quality.
    floor = int(round(q_max - (q_max - q_min) * strength))
    quality = rng.randint(max(q_min, min(floor, q_max)), q_max)
    import io
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=False)
    buf.seek(0)
    out = Image.open(buf).convert("RGB")
    return out, {"jpeg_quality": quality}


def sample_family_strengths(rng_np: np.random.Generator, global_severity: float) -> dict[str, float]:
    weights = rng_np.dirichlet(np.ones(len(FAMILIES), dtype=np.float32)).astype(float)
    # Keep every family present, while the Dirichlet residual creates mixtures.
    floor = 0.65
    strengths: dict[str, float] = {}
    for family, weight in zip(FAMILIES, weights):
        strengths[family] = float(global_severity * (floor + (1.0 - floor) * weight))
    return strengths


def degrade_image(image: Image.Image, *, seed: int, profile: str) -> tuple[Image.Image, dict[str, Any]]:
    cfg = PROFILE_CONFIGS[profile]
    rng = random.Random(seed)
    rng_np = np.random.default_rng(seed)
    global_severity = rng.uniform(cfg["severity_min"], cfg["severity_max"])
    strengths = sample_family_strengths(rng_np, global_severity)

    out = image.convert("L")
    params: dict[str, Any] = {
        "profile": profile,
        "seed": seed,
        "global_severity": global_severity,
        "family_strengths": strengths,
        "stage_order": [
            "artifact_field: subtractive on white page, page coloring, additive pigment",
            "geometry: warp/translate/skew/perspective/rotate using local paper fill outside source",
            "lighting",
            "blur",
            "compression",
        ],
        "families": {},
    }

    # v10 order requested:
    #   1. noising/subtractive deterioration while page is still white
    #   2. page coloring
    #   3. additive pigment/noise
    #   4. physical transformations
    out, params["families"]["artifact_field"] = apply_unified_artifact_field(out, rng, rng_np, cfg, strengths["artifact_field"])

    params["families"]["paper_ink"] = {
        "skipped": True,
        "reason": "Replaced by ordered artifact_field: subtractive deterioration before page coloring, then additive pigment after coloring.",
    }

    out, params["families"]["geometry"] = apply_geometry(out, rng, cfg, strengths["geometry"])
    out, params["families"]["lighting"] = apply_lighting(out, rng, cfg, strengths["lighting"])
    out, params["families"]["blur"] = apply_blur(out, rng, cfg, strengths["blur"])
    out, params["families"]["compression"] = apply_compression(out, rng, cfg, strengths["compression"])
    return out, params


def output_page_id(source_page_id: str, variant_index: int, variants_per_page: int) -> str:
    if variants_per_page == 1:
        return f"{source_page_id}__aug"
    return f"{source_page_id}__aug{variant_index:02d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create sequential camera/degradation variants of rendered score pages.")
    parser.add_argument("--render-log", type=Path, required=True, help="Input clean render/page log JSONL.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output batch directory for augmented pages and logs.")
    parser.add_argument("--profile", choices=sorted(PROFILE_CONFIGS), default="camera_light_v1")
    parser.add_argument("--variants-per-page", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-pages", type=int, default=-1)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.variants_per_page <= 0:
        raise ValueError("--variants-per-page must be positive")

    pages = collect_pages(args.render_log)
    if args.max_pages > 0:
        pages = pages[: args.max_pages]
    if not pages:
        raise RuntimeError(f"No pages found in {args.render_log}")

    out_dir: Path = args.out_dir
    pages_dir = out_dir / "augmented_pages"
    logs_dir = out_dir / "logs"
    augment_log = logs_dir / "augment_log.jsonl"
    summary_path = out_dir / "augmentation_summary.json"
    pages_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    if args.overwrite and augment_log.exists():
        augment_log.unlink()

    emit_event({
        "event": "start",
        "render_log": str(args.render_log),
        "out_dir": str(out_dir),
        "augment_log": str(augment_log),
        "profile": args.profile,
        "source_pages": len(pages),
        "variants_per_page": args.variants_per_page,
        "seed": args.seed,
    }, quiet=args.quiet)

    started = time.time()
    ok = 0
    skipped = 0
    errors = 0
    source_missing = 0

    for page_index, page in enumerate(pages, start=1):
        source_image = Path(page["image_path"])
        if not source_image.exists():
            source_missing += 1
            errors += 1
            row = {
                "status": "error",
                "error_type": "FileNotFoundError",
                "error": f"Source image not found: {source_image}",
                "score_id": page["score_id"],
                "source_page_id": page["page_id"],
                "source_image_path": str(source_image),
                "timestamp_unix": time.time(),
            }
            append_jsonl(augment_log, row)
            emit_event({"event": "error", **row}, quiet=False, stderr=True)
            continue

        try:
            clean = Image.open(source_image).convert("L")
        except Exception as exc:
            errors += 1
            row = {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "score_id": page["score_id"],
                "source_page_id": page["page_id"],
                "source_image_path": str(source_image),
                "timestamp_unix": time.time(),
            }
            append_jsonl(augment_log, row)
            emit_event({"event": "error", **row}, quiet=False, stderr=True)
            continue

        for variant_index in range(args.variants_per_page):
            variant_seed = args.seed + page_index * 1000003 + variant_index * 9176
            page_id = output_page_id(str(page["page_id"]), variant_index, args.variants_per_page)
            out_path = pages_dir / f"{page_id}.png"

            if out_path.exists() and not args.overwrite:
                skipped += 1
                continue

            try:
                degraded, params = degrade_image(clean, seed=variant_seed, profile=args.profile)
                degraded.save(out_path, format="PNG", optimize=True)

                row = {
                    "status": "ok",
                    "schema": "homr_augmented_page_log_v1",
                    "score_id": page["score_id"],
                    "source_path": page["source_path"],
                    "source_page_id": page["page_id"],
                    "source_image_path": str(source_image),
                    "page_id": page_id,
                    "page_number": page["page_number"],
                    "variant_index": variant_index,
                    "image_path": str(out_path),
                    "image_size_bytes": out_path.stat().st_size,
                    "augmentation": params,
                    "timestamp_unix": time.time(),
                }
                append_jsonl(augment_log, row)
                ok += 1
            except Exception as exc:
                errors += 1
                row = {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "score_id": page["score_id"],
                    "source_page_id": page["page_id"],
                    "page_id": page_id,
                    "source_image_path": str(source_image),
                    "image_path": str(out_path),
                    "timestamp_unix": time.time(),
                }
                append_jsonl(augment_log, row)
                emit_event({"event": "error", **row}, quiet=False, stderr=True)

        if args.progress_every > 0 and (page_index == 1 or page_index % args.progress_every == 0 or page_index == len(pages)):
            emit_event({
                "event": "progress",
                "processed_source_pages": page_index,
                "total_source_pages": len(pages),
                "ok": ok,
                "skipped": skipped,
                "errors": errors,
                "seconds": time.time() - started,
            }, quiet=args.quiet)

    summary = {
        "schema": "homr_augmentation_summary_v1",
        "status": "ok" if errors == 0 else "completed_with_errors",
        "render_log": str(args.render_log),
        "out_dir": str(out_dir),
        "augment_log": str(augment_log),
        "augmented_pages_dir": str(pages_dir),
        "profile": args.profile,
        "seed": args.seed,
        "source_pages": len(pages),
        "variants_per_page": args.variants_per_page,
        "ok": ok,
        "skipped": skipped,
        "errors": errors,
        "source_missing": source_missing,
        "seconds": time.time() - started,
    }
    write_json_atomic(summary_path, summary)
    print(json.dumps({"event": "done", **summary}, sort_keys=True), flush=True)
    return 0 if errors == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
