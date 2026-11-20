"""
Measuring what the camera did to a photo's print (photo_look.Look), from the template page that
was printed: the known print is drawn into the photo through the page map, lined up with the photo
spot by spot, and the blur, ink strength, skirt and darkness curve are fitted so that drawing the
known print with them gives back the photo. The same drawing (photo_look.draw_ink) then pastes new
music, so the check of the old print is a check of what gets pasted.
"""

import cv2
import numpy as np
from numpy.typing import NDArray

from training.omr_datasets.photo_look import STEP, field, spots
from training.omr_datasets.photo_maps import FH, FLAT_MM, FW, PhotoConditions

WINDOW = 256  # photo pixels around a spot for the first, coarse line-up: the map can be 20+ px off
FINE = 128  # photo pixels around a spot for the fine line-up
MIN_INK = 0.02  # a spot with less print than this (mean darkness) isn't measured
MIN_EXPLAINED = 0.5  # a spot whose lined-up print explains less of its darkness is dropped
MAX_JUMP = 2.0  # a spot whose shift is further than this from its neighbours' is dropped
EDGE_MM = 6  # spots this close to the page edge aren't measured: the map is least sure there


def seen_darkness(cond: PhotoConditions) -> NDArray:
    """How much darker each photo pixel is than the paper there (the erased photo), 0 = paper."""
    gray = cv2.cvtColor(cond.photo, cv2.COLOR_BGR2GRAY).astype(np.float32)
    paper = cv2.cvtColor(np.asarray(cond.paper, np.float32), cv2.COLOR_BGR2GRAY)
    return np.asarray(1 - gray / np.maximum(paper, 1), np.float32)


def usable(cond: PhotoConditions) -> NDArray:
    """The photo pixels surely on the page: EDGE_MM in from its edges."""
    edge = EDGE_MM * FLAT_MM
    x, y = cond.map_x, cond.map_y
    return np.asarray(
        cond.on_page & (x >= edge) & (x <= FW - 1 - edge) & (y >= edge) & (y <= FH - 1 - edge), bool
    )


def _moved(image: NDArray, d: tuple[float, float]) -> NDArray:
    """image(x - d): its content moved by d."""
    m = np.array([[1, 0, d[0]], [0, 1, d[1]]], np.float32)
    return cv2.warpAffine(
        image, m, image.shape[::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )


def _window(image: NDArray, cy: int, cx: int, size: int) -> tuple[slice, slice]:
    h, w = image.shape
    y0 = min(max(cy - size // 2, 0), h - size)
    x0 = min(max(cx - size // 2, 0), w - size)
    return slice(y0, y0 + size), slice(x0, x0 + size)


def _spot_shift(
    known: NDArray, seen: NDArray, cy: int, cx: int, step: int
) -> tuple[float, float, float]:
    """The shift (dx, dy) moving the known print onto the photo at a spot, and the share of the
    spot's darkness the moved print explains."""
    hanning = cv2.createHanningWindow((WINDOW, WINDOW), cv2.CV_64F)
    ys, xs = _window(known, cy, cx, WINDOW)
    (dx, dy), _ = cv2.phaseCorrelate(
        known[ys, xs].astype(np.float64), seen[ys, xs].astype(np.float64), hanning
    )
    # fine: ECC (Evangelidis and Psarakis 2008) fits a small affine warp W to the pixel values,
    # sub-pixel, with known(W(x)) ~ seen(x); the map's error changes across the window, so the
    # shift is read at the spot's centre c: shift = c - W(c)
    fy, fx = _window(known, cy, cx, FINE)
    warp = np.array([[1, 0, -dx], [0, 1, -dy]], np.float32)
    everywhere = np.ones((FINE, FINE), np.uint8)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)
    try:
        _, fitted = cv2.findTransformECC(
            seen[fy, fx], known[fy, fx], warp, cv2.MOTION_AFFINE, criteria, everywhere, 1
        )
        c = np.array([cx - fx.start, cy - fy.start, 1.0])
        wx, wy = np.asarray(fitted) @ c
        dx, dy = float(c[0] - wx), float(c[1] - wy)
    except cv2.error:  # no convergence: keep the coarse shift, the explained share decides
        pass
    half = step // 2
    moved = _moved(known[ys, xs], (dx, dy))
    oy, ox = cy - half - ys.start, cx - half - xs.start
    m = moved[oy : oy + step, ox : ox + step]
    t = seen[cy - half : cy + half, cx - half : cx + half]
    s = max(float((m * t).sum() / max(float((m * m).sum()), 1e-6)), 0.0)
    explained = 1 - float(((t - s * m) ** 2).sum()) / max(float((t * t).sum()), 1e-6)
    return float(dx), float(dy), explained


def line_up(
    known: NDArray, seen: NDArray, usable: NDArray, step: int = STEP
) -> tuple[NDArray, NDArray]:
    """Per spot, the shift (dx, dy) in photo pixels that moves the drawn print onto the photo:
    the photo shows at x what the drawing shows at x - shift. NaN where the spot has too little
    print, isn't wholly on the page, isn't explained by the print, or jumps from its neighbours
    (a spot of staff lines alone can lock one line off)."""
    h, w = known.shape
    rows, cols = spots((h, w), step)
    dx = np.full((rows, cols), np.nan, np.float32)
    dy = np.full((rows, cols), np.nan, np.float32)
    known = cv2.GaussianBlur(known, (0, 0), 1.0)  # closer to the photo's blur: a cleaner peak
    half = step // 2
    for i in range(rows):
        for j in range(cols):
            cy, cx = i * step + half, j * step + half
            if cy + half > h or cx + half > w:
                continue
            spot = (slice(cy - half, cy + half), slice(cx - half, cx + half))
            if not usable[spot].all() or float(known[spot].mean()) < MIN_INK:
                continue
            sx, sy, explained = _spot_shift(known, seen, cy, cx, step)
            if explained >= MIN_EXPLAINED:
                dx[i, j], dy[i, j] = sx, sy
    keep_x, keep_y = dx.copy(), dy.copy()
    for i in range(rows):
        for j in range(cols):
            if np.isnan(dx[i, j]):
                continue
            near = [
                (dx[a, b], dy[a, b])
                for a in range(max(0, i - 1), min(rows, i + 2))
                for b in range(max(0, j - 1), min(cols, j + 2))
                if (a, b) != (i, j) and not np.isnan(dx[a, b])
            ]
            if len(near) < 2:  # noqa: PLR2004
                continue
            mx, my = np.median(np.array(near), axis=0)
            if np.hypot(dx[i, j] - mx, dy[i, j] - my) > MAX_JUMP:
                keep_x[i, j] = keep_y[i, j] = np.nan
    return keep_x, keep_y


def refine_maps(
    maps: tuple[NDArray, NDArray], shift: tuple[NDArray, NDArray], step: int = STEP
) -> tuple[NDArray, NDArray]:
    """The page maps with the measured shift folded in: the new map at x is the old map at
    x - shift(x). Off the page (-1) stays off; pixels whose source touches it go off too."""
    map_x, map_y = maps
    h, w = map_x.shape
    if np.isnan(shift[0]).all():
        return map_x, map_y
    dx, dy = field(shift[0], (h, w), step), field(shift[1], (h, w), step)
    ys, xs = np.mgrid[:h, :w].astype(np.float32)
    sx, sy = xs - dx, ys - dy
    new_x = cv2.remap(map_x, sx, sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    new_y = cv2.remap(map_y, sx, sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    on = cv2.remap((map_x >= 0).astype(np.float32), sx, sy, cv2.INTER_LINEAR, borderValue=0)
    off = on < 0.999  # noqa: PLR2004
    new_x[off], new_y[off] = -1, -1
    return np.asarray(new_x, np.float32), np.asarray(new_y, np.float32)
