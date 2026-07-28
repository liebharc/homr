"""Augraphy document-degradation augmentation for training."""

import os
import tempfile
from typing import Any

import numpy as np

from homr.type_definitions import NDArray

_worker_tmpdir: str | None = None


def _get_worker_tmpdir() -> str:
    # Each DataLoader worker is a separate process; give it its own directory so
    # augraphy's ./augraphy_cache/ writes don't collide across workers.
    global _worker_tmpdir  # noqa: PLW0603
    if _worker_tmpdir is None:
        _worker_tmpdir = tempfile.mkdtemp(prefix=f"augraphy_{os.getpid()}_")
    return _worker_tmpdir


def apply_augraphy(image: NDArray) -> NDArray:
    """Apply Augraphy document degradation. 5% no-op; returns input on any failure."""
    if np.random.random() < 0.05:
        return image

    try:
        pipeline = _build_pipeline()
    except ImportError:
        return image

    saved_cwd = os.getcwd()
    os.chdir(_get_worker_tmpdir())
    try:
        result = pipeline(image)
        augmented: NDArray = result["output"] if isinstance(result, dict) else result
        if augmented is None or augmented.shape[:2] != image.shape[:2]:
            return image
        if augmented.dtype != np.uint8:
            augmented = np.clip(augmented, 0, 255).astype(np.uint8)
        return augmented
    except Exception:
        return image
    finally:
        os.chdir(saved_cwd)


def _build_pipeline(rng: np.random.Generator | None = None) -> Any:
    try:
        import augraphy  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("augraphy is required: pip install augraphy") from exc

    r = rng.random if rng is not None else np.random.random

    ink_phase = []

    if r() < 0.20:
        ink_phase.append(augraphy.InkBleed(intensity_range=(0.2, 0.5), p=1.0))

    if r() < 0.50:
        ink_phase.append(augraphy.LowInkRandomLines(count_range=(5, 15), p=1.0))

    paper_phase: list[Any] = [augraphy.Brightness(brightness_range=(0.5, 1.5), p=1.0)]

    if r() < 0.10:
        try:
            paper_phase.append(augraphy.BleedThrough(intensity_range=(0.5, 1.0), alpha=0.2, p=1.0))
        except (AttributeError, TypeError):
            pass

    if r() < 0.25:
        try:
            paper_phase.append(
                augraphy.LightingGradient(
                    light_position=None,
                    direction=None,
                    max_brightness=255,
                    min_brightness=0,
                    mode="gaussian",
                    transparency=0.92,
                    p=1.0,
                )
            )
        except (AttributeError, TypeError):
            pass

    post_phase = []

    if r() < 0.90:
        try:
            post_phase.append(augraphy.DirtyScreen(clumsiness_range=(0.1, 1.0), p=1.0))
        except (AttributeError, TypeError):
            try:
                post_phase.append(augraphy.DirtyScreen(p=1.0))
            except AttributeError:
                pass

    if r() < 0.60:
        try:
            post_phase.append(augraphy.NoiseTexturize(sigma_range=(1, 2), p=1.0))
        except AttributeError:
            pass

    if r() < 0.33:
        try:
            post_phase.append(augraphy.SubtleNoise(subtle_range=8, p=1.0))
        except AttributeError:
            pass

    if r() < 0.50:
        try:
            post_phase.append(augraphy.Jpeg(quality_range=(60, 95), p=1.0))
        except AttributeError:
            pass

    if r() < 0.20:
        try:
            if r() < 0.5:
                post_phase.append(augraphy.DirtyRollers(p=1.0))
            else:
                post_phase.append(augraphy.DirtyDrum(p=1.0))
        except AttributeError:
            pass

    if r() < 0.15:
        try:
            post_phase.append(augraphy.ShadowCast(p=1.0))
        except AttributeError:
            pass

    if r() < 0.05:
        post_phase.append(augraphy.LowInkRandomLines(count_range=(1, 3), p=1.0))

    return augraphy.AugraphyPipeline(
        ink_phase=ink_phase,
        paper_phase=paper_phase,
        post_phase=post_phase,
        log=False,
    )
