"""Augraphy document-degradation augmentation for training."""

import os
import random
import tempfile
from typing import Any

import augraphy.base.augmentationpipeline as augraphy_pipeline_module
import cv2
import numpy as np
from augraphy import BleedThrough, DirtyScreen, LightingGradient, NoiseTexturize
from augraphy.utilities.overlaybuilder import OverlayBuilder

from homr.type_definitions import NDArray

_worker_tmpdir: str | None = None

# Augraphy writes every input as PNG into ./augraphy_cache/ (~4 ms per image), only so that
# BleedThrough can read one of the last 30 back. We keep them in memory instead.
_recent_images: list[NDArray] = []
_max_recent_images = 30


class _Cv2WithoutImwrite:
    def __getattr__(self, name: str) -> Any:
        return getattr(cv2, name)

    @staticmethod
    def imwrite(*_args: Any, **_kwargs: Any) -> bool:
        return True


augraphy_pipeline_module.cv2 = _Cv2WithoutImwrite()
augraphy_pipeline_module.glob = lambda *_args, **_kwargs: []


def _remember_image(image: NDArray) -> None:
    if len(_recent_images) >= _max_recent_images:
        _recent_images.pop(0)
    _recent_images.append(image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))


def _get_worker_tmpdir() -> str:
    # Each DataLoader worker is a separate process; give it its own directory so
    # augraphy's ./augraphy_cache/ writes don't collide across workers.
    global _worker_tmpdir  # noqa: PLW0603
    if _worker_tmpdir is None:
        _worker_tmpdir = tempfile.mkdtemp(prefix=f"augraphy_{os.getpid()}_")
    return _worker_tmpdir


class FastLightingGradient(LightingGradient):
    """
    LightingGradient that computes the gaussian light mask directly at image size.
    Augraphy draws it on a canvas several times larger and rotates it, ~100 ms per image.
    """

    def generate_parallel_light_mask(  # noqa: PLR0913
        self,
        mask_size: tuple[int, int],
        position: tuple[int, int] | None = None,
        direction: int | None = None,
        max_brightness: int = 255,
        min_brightness: int = 0,
        mode: str = "gaussian",
        linear_decay_rate: float | None = None,
    ) -> NDArray:
        if mode != "gaussian":
            return super().generate_parallel_light_mask(
                mask_size,
                position,
                direction,
                max_brightness,
                min_brightness,
                mode,
                linear_decay_rate,
            )
        width, height = mask_size
        # Same random draws as augraphy
        if position is None:
            pos_x = random.randint(0, width)
            pos_y = random.randint(0, height)
        else:
            pos_x, pos_y = position
        if direction is None:
            direction = random.randint(0, 360)

        # Distance from the light strip after cv2's rotation by direction around the light
        angle = np.deg2rad(direction)
        dx = np.arange(width, dtype=np.float32) - pos_x
        dy = np.arange(height, dtype=np.float32) - pos_y
        distance = np.float32(np.sin(angle)) * dx[None, :] + np.float32(np.cos(angle)) * dy[:, None]
        radius = np.float32(height / 3)
        brightness = np.exp(np.float32(-0.5) * np.square(distance / radius))
        # Skips augraphy's median blur, which changes a smooth gradient by a few gray levels
        mask = (brightness * (max_brightness - min_brightness) + min_brightness).astype(np.uint8)
        return 255 - mask


class FastBleedThrough(BleedThrough):
    """BleedThrough that takes the foreground from the images in memory, see _recent_images."""

    def create_bleedthrough_foreground(self, image: NDArray) -> NDArray:
        if _recent_images:
            foreground = _recent_images[random.randint(0, len(_recent_images) - 1)]
            foreground = cv2.resize(
                foreground, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA
            )
        else:
            foreground = image
        # flip left-right only, flip top-bottom get inverted text, which is not realistic
        return cv2.flip(foreground, 1)


class FastDirtyScreen(DirtyScreen):
    """
    DirtyScreen with the same random draws, but it only touches the generated points.
    Augraphy builds ten full-size noise images per call, ~30 ms per image.
    """

    def __call__(  # noqa: PLR0913
        self,
        image: NDArray,
        layer: Any = None,
        mask: Any = None,
        keypoints: Any = None,
        bounding_boxes: Any = None,
        force: bool = False,
    ) -> Any:
        if not (force or self.should_run()):
            return None
        image = image.copy()
        has_alpha = image.ndim > 2 and image.shape[2] == 4
        if has_alpha:
            image, image_alpha = image[:, :, :3], image[:, :, 3]
        is_gray = image.ndim == 2
        if is_gray:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        ysize, xsize = image.shape[:2]
        max_box_size = max(ysize, xsize)

        merged = np.full(ysize * xsize, 255, dtype=np.float64)
        for _ in range(10):
            n_samples = [
                random.randint(self.n_samples[0], self.n_samples[1])
                for _ in range(random.randint(self.n_clusters[0], self.n_clusters[1]))
            ]
            std = random.randint(self.std_range[0], self.std_range[1])
            # sklearn's make_blobs: uniform centers, gaussian points around them
            centers = np.random.uniform(0, max_box_size - 1, size=(len(n_samples), 2))
            points = np.repeat(centers, n_samples, axis=0)
            points = (points + np.random.normal(0, std, size=points.shape)).astype(int)
            inside = (
                (points[:, 0] >= 0)
                & (points[:, 1] >= 0)
                & (points[:, 0] <= xsize - 1)
                & (points[:, 1] <= ysize - 1)
            )
            pixels = np.unique(points[inside, 1] * xsize + points[inside, 0])
            values = np.random.randint(self.value_range[0], self.value_range[1], size=len(pixels))
            # cv2.multiply(merged, noise, scale=1 / 255) with noise = 255 elsewhere
            merged[pixels] = np.rint(merged[pixels] * values / 255)

        merged_2d = merged.astype(np.uint8).reshape(ysize, xsize)
        dirty = merged_2d != 255
        image_output = image.copy()
        image_output[dirty] = merged_2d[dirty][:, None]

        ob = OverlayBuilder("ink_to_paper", image_output, image, 1, (1, 1), "center", 0, 0.5)
        image_output = ob.build_overlay()

        if is_gray:
            image_output = cv2.cvtColor(image_output, cv2.COLOR_BGR2GRAY)
        if has_alpha:
            image_output = np.dstack((image_output, image_alpha))
        if mask is not None or keypoints is not None or bounding_boxes is not None:
            return [image_output, mask, keypoints, bounding_boxes]
        return image_output


class FastNoiseTexturize(NoiseTexturize):
    """
    NoiseTexturize that sums the noise in one float32 channel, drawn with numpy's faster
    Generator. Augraphy stacks every noise layer into three float64 channels.
    """

    def noise(self, width: int, height: int, channel: int, ratio: int, sigma: float) -> NDArray:
        ysize = random.randint(self.texture_height_range[0], self.texture_height_range[1])
        xsize = random.randint(self.texture_width_range[0], self.texture_width_range[1])
        rng = np.random.default_rng(np.random.randint(2**31))
        result = rng.standard_normal((ysize, xsize), dtype=np.float32) * np.float32(sigma)
        return cv2.resize(result, dsize=(width, height), interpolation=cv2.INTER_LINEAR)

    def __call__(  # noqa: PLR0913
        self,
        image: NDArray,
        layer: Any = None,
        mask: Any = None,
        keypoints: Any = None,
        bounding_boxes: Any = None,
        force: bool = False,
    ) -> Any:
        if not (force or self.should_run()):
            return None
        image = image.copy()
        has_alpha = image.ndim > 2 and image.shape[2] == 4
        if has_alpha:
            image, image_alpha = image[:, :, :3], image[:, :, 3]

        sigma = random.randint(self.sigma_range[0], self.sigma_range[1])
        turbulence = random.randint(self.turbulence_range[0], self.turbulence_range[1])
        rows, cols = image.shape[:2]
        noise_sum = np.zeros((rows, cols), dtype=np.float32)
        ratio = cols
        while ratio != 1:
            noise_sum += self.noise(cols, rows, 0, ratio, sigma=sigma)
            ratio = (ratio // turbulence) or 1
        if image.ndim > 2:
            noise_sum = noise_sum[:, :, None]
        cut = np.clip(image.astype(np.float32) + noise_sum, 0, 255).astype(np.uint8)

        if has_alpha:
            cut = np.dstack((cut, image_alpha))
        if mask is not None or keypoints is not None or bounding_boxes is not None:
            return [cut, mask, keypoints, bounding_boxes]
        return cut


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
        _remember_image(image)
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
            paper_phase.append(FastBleedThrough(intensity_range=(0.5, 1.0), alpha=0.2, p=1.0))
        except (AttributeError, TypeError):
            pass

    if r() < 0.25:
        try:
            paper_phase.append(
                FastLightingGradient(
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
            post_phase.append(FastDirtyScreen(clumsiness_range=(0.1, 1.0), p=1.0))
        except (AttributeError, TypeError):
            try:
                post_phase.append(FastDirtyScreen(p=1.0))
            except AttributeError:
                pass

    if r() < 0.60:
        try:
            post_phase.append(FastNoiseTexturize(sigma_range=(1, 2), p=1.0))
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
