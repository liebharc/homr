"""
What a camera did to a printed page, as measured from a photo of a known print (see
photo_measure.py), and the one function that draws ink the way that camera would.

Everything is in photo pixels (the photo brought to PHOTO_SIZE). The page is cut into spots STEP
pixels apart; per spot there is a blur and an ink strength, and they change smoothly between spots.
Per photo there is a faint skirt (a small share of the ink spread far out: light scattered in the
paper and the lens) and a darkness curve (thin, faint strokes come out darker than a straight
model gives, solid black lighter). Ink is darkness: the share of the paper's light it takes away.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

STEP = 96  # spot spacing, photo pixels at PHOTO_SIZE
SUPER = 2  # the page is drawn at twice the photo's size, then shrunk: what a camera does
NO_CURVE = (1.0, 0.0)
MIN_SIGMA = 0.3  # a sharper blur than this is a single pixel anyway
DEFAULT_BLUR = (1.0, 1.0, 0.0)  # for a photo with no measured spot at all


@dataclass
class Look:
    """blur: per spot, the wide and narrow sigma of a stretched Gaussian (photo pixels) and the
    angle of its wide axis, the direction (cos, sin) in (x, y) pixel coordinates. strength: per
    spot, the darkness of fully inked paper before the curve. skirt: (share, sigma), that share of
    the ink spread with a Gaussian of that sigma. curve: (gamma, toe), see darkness_curve."""

    blur: NDArray
    strength: NDArray
    skirt: tuple[float, float]
    curve: tuple[float, float]
    step: int = STEP


def spots(shape: tuple[int, int], step: int = STEP) -> tuple[int, int]:
    """The rows and columns of spots covering a photo of this shape. Spot (i, j) is centred on
    pixel (i * step + step / 2, j * step + step / 2)."""
    return math.ceil(shape[0] / step), math.ceil(shape[1] / step)


def darkness_curve(d: NDArray, curve: tuple[float, float]) -> NDArray:
    """A darkness (0..1) through the photo's darkness curve. toe keeps the faintest darkness from
    being raised without limit; toe 0 is a plain gamma, NO_CURVE is straight."""
    gamma, toe = curve
    d = np.clip(d, 0.0, 1.0)
    return ((d + toe) ** gamma - toe**gamma) / ((1 + toe) ** gamma - toe**gamma)


def filled(grid: NDArray, default: float | tuple[float, ...]) -> NDArray:
    """The grid with every spot lacking a value (NaN) given its nearest spot's value."""
    grid = np.asarray(grid, np.float32)
    missing = np.isnan(grid).any(axis=-1) if grid.ndim == 3 else np.isnan(grid)  # noqa: PLR2004
    if not missing.any():
        return grid
    if missing.all():
        return np.broadcast_to(np.asarray(default, np.float32), grid.shape).copy()
    _, nearest = ndimage.distance_transform_edt(missing, return_indices=True)
    return grid[tuple(nearest)]


def field(grid: NDArray, shape: tuple[int, int], step: int = STEP) -> NDArray:
    """Spot values to every pixel of a photo of this shape: missing spots take their nearest
    spot's value, then the values change linearly between spot centres."""
    rows, cols = grid.shape[:2]
    values = filled(grid, 0.0)
    smooth = cv2.resize(values, (cols * step, rows * step), interpolation=cv2.INTER_LINEAR)
    return np.asarray(smooth[: shape[0], : shape[1]], np.float32)


def known_print(flat: NDArray, maps: tuple[NDArray, NDArray]) -> NDArray:
    """The flat page's darkness (0..1) at every photo pixel, sharp: drawn at SUPER times the photo's
    size through the maps, then shrunk. 0 off the page."""
    map_x, map_y = maps
    h, w = map_x.shape
    big = (w * SUPER, h * SUPER)
    big_x = cv2.resize(map_x, big, interpolation=cv2.INTER_LINEAR)
    big_y = cv2.resize(map_y, big, interpolation=cv2.INTER_LINEAR)
    lit = cv2.remap(flat, big_x, big_y, cv2.INTER_LINEAR, borderValue=255)
    small = cv2.resize(lit, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
    return np.asarray(1 - small / 255, np.float32)


def stretched_kernel(wide: float, narrow: float, angle: float) -> NDArray:
    """A normalised stretched Gaussian, rows y and columns x."""
    wide, narrow = max(wide, MIN_SIGMA), max(narrow, MIN_SIGMA)
    r = math.ceil(3 * max(wide, narrow))
    ys, xs = np.mgrid[-r : r + 1, -r : r + 1].astype(np.float32)
    c, s = math.cos(angle), math.sin(angle)
    along, across = c * xs + s * ys, -s * xs + c * ys
    kernel = np.exp(-0.5 * ((along / wide) ** 2 + (across / narrow) ** 2))
    return np.asarray(kernel / kernel.sum(), np.float32)


def blur_by_spot(sharp: NDArray, blur: NDArray, step: int = STEP) -> NDArray:
    """sharp blurred with each spot's stretched Gaussian, blended linearly between spot centres."""
    h, w = sharp.shape
    blur = filled(blur, DEFAULT_BLUR)
    kernels = [
        [stretched_kernel(*map(float, blur[i, j])) for j in range(blur.shape[1])]
        for i in range(blur.shape[0])
    ]
    reach = max(k.shape[0] for row in kernels for k in row) // 2
    padded = cv2.copyMakeBorder(sharp, reach, reach, reach, reach, cv2.BORDER_CONSTANT, value=0)
    out = np.zeros((h, w), np.float32)
    weight = np.zeros((h, w), np.float32)
    for i in range(blur.shape[0]):
        cy = i * step + step / 2
        y0, y1 = max(0, math.floor(cy - step)), min(h, math.ceil(cy + step))
        wy = np.clip(1 - np.abs(np.arange(y0, y1) + 0.5 - cy) / step, 0, None)
        for j in range(blur.shape[1]):
            cx = j * step + step / 2
            x0, x1 = max(0, math.floor(cx - step)), min(w, math.ceil(cx + step))
            wx = np.clip(1 - np.abs(np.arange(x0, x1) + 0.5 - cx) / step, 0, None)
            tent = (wy[:, None] * wx[None, :]).astype(np.float32)
            weight[y0:y1, x0:x1] += tent
            region = padded[y0 : y1 + 2 * reach, x0 : x1 + 2 * reach]
            if not region.any():
                continue
            done = cv2.filter2D(region, -1, kernels[i][j], borderType=cv2.BORDER_CONSTANT)
            out[y0:y1, x0:x1] += tent * done[reach : reach + y1 - y0, reach : reach + x1 - x0]
    return np.asarray(out / np.maximum(weight, 1e-6), np.float32)


def draw_ink(flat: NDArray, maps: tuple[NDArray, NDArray], look: Look) -> NDArray:
    """The darkness the camera would show at every photo pixel for the flat page's ink: the sharp
    print blurred per spot, a share of it spread into the skirt, through the darkness curve, times
    the ink strength. 0 off the page."""
    sharp = known_print(flat, maps)
    blurred = blur_by_spot(sharp, look.blur, look.step)
    share, skirt_sigma = look.skirt
    if share > 0:
        wide = cv2.GaussianBlur(sharp, (0, 0), max(skirt_sigma, MIN_SIGMA))
        blurred = (1 - share) * blurred + share * wide
    strength = field(filled(look.strength, 1.0), sharp.shape, look.step)
    drawn = strength * darkness_curve(blurred, look.curve)
    drawn[maps[0] < 0] = 0
    return np.asarray(drawn, np.float32)


def save_look(path: Path, look: Look) -> None:
    np.savez_compressed(
        path,
        blur=look.blur,
        strength=look.strength,
        skirt=np.array(look.skirt),
        curve=np.array(look.curve),
        step=np.array(look.step),
    )


def load_look(path: Path) -> Look:
    saved = np.load(path)
    return Look(
        saved["blur"],
        saved["strength"],
        (float(saved["skirt"][0]), float(saved["skirt"][1])),
        (float(saved["curve"][0]), float(saved["curve"][1])),
        int(saved["step"]),
    )
