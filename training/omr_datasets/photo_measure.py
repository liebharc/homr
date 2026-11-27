"""
Measuring what the camera did to a photo's print (photo_look.Look), from the template page that
was printed: the known print is drawn into the photo through the page map, lined up with the photo
spot by spot, and the blur, ink strength, skirt and darkness curve are fitted so that drawing the
known print with them gives back the photo. The same drawing (photo_look.draw_ink) then pastes new
music, so the check of the old print is a check of what gets pasted.
"""

import math
from dataclasses import dataclass, replace

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq, least_squares

from training.omr_datasets.photo_look import (
    STEP,
    Look,
    darkness_curve,
    draw_ink,
    field,
    filled,
    finished_ink,
    ink_mix,
    known_print,
    sharpened,
    spots,
    stretched_kernel,
)
from training.omr_datasets.photo_maps import (
    ERASE_MARGIN_MM,
    FH,
    FLAT_MM,
    FW,
    PhotoConditions,
)

WINDOW = 256  # photo pixels around a spot for the first, coarse line-up: the map can be 20+ px off
FINE = 128  # photo pixels around a spot for the fine line-up
MIN_INK = 0.02  # a spot with less print than this (mean darkness) isn't measured
MIN_EXPLAINED = 0.5  # a spot whose lined-up print explains less of its darkness is dropped
MAX_JUMP = 2.0  # a spot whose shift is further than this from its neighbours' is dropped
# a photo whose page map was off by more than this is left out: the old print is erased only
# ERASE_MARGIN_MM around its box, so some could be left on the paper
MAP_OFF_MAX_MM = 0.75 * ERASE_MARGIN_MM
# the second, finer line-up: after the first, a real photo's print is still off by about 0.5 px,
# changing from one 48 px area to the next, which a fit would take for blur
FINE_STEP = 48
FINE_WINDOW = 64
MIN_CORRELATION = 0.6  # a finer window whose print matches the photo less well isn't used
EDGE_MM = 6  # spots this close to the page edge aren't measured: the map is least sure there
MIN_SPOTS = 12  # with fewer measured spots a photo is left out: too little print to go on
ROUNDS = 3  # the spot fits and the photo's skirt and curve, fitted in turn
REACH = 24  # photo pixels of print around a spot that its blur can bring in
START_SKIRT = (0.05, 6.0)
# the phone's sharpening: its amount is chosen last, from these, as the one whose drawing
# passes the check best, with the blur and skirt held (fitted along with them, it traded off
# against them and drifted); its radius is held (with a tiny radius any amount does next to
# nothing, and radius 2 and 4 fitted real photos alike)
SHARPEN_RADIUS = 2.0
START_SHARPEN = (0.0, SHARPEN_RADIUS)  # (amount, radius)
SHARPEN_TRIES = tuple(round(0.05 * k, 2) for k in range(9))  # 0 to 0.4
START_CURVE = (1.0, 0.02)  # (gamma, TOE)
# the curve's toe is held here, only its bend (gamma) is fitted: the two trade off almost exactly
# over the darkness that matters, so fitting both lets the curve soak up the blur
TOE = 0.02
# the skirt is light scattered within a few millimetres; wider, the fit was soaking up the erased
# paper's own slow errors (a real photo ran to the old limit of 20)
MAX_SKIRT = 12.0
GAMMAS = (0.4, 1.6)  # the darkness curve's bend is looked for in here
MAX_LEFTOVER = 3.0  # photo pixels a spot's print may still be off after lining up
# the blur is drawn in single precision: slopes are estimated with steps this big (relative), or
# the fit sees no slope and stops where it started
DIFF_STEP = 1e-3
# the per-photo check: drawn / real within this, for line width, line darkness, ink per line and
# solid black; the skirt (darkness 3-9 px out from the ink) within this share of the paper
TOLERANCE = 0.05
SKIRT_TOLERANCE = 0.003
PROFILE = (96, 24)  # height, width of the strips staff lines are measured in


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
    return _dropped_jumps(dx, dy)


def _dropped_jumps(dx: NDArray, dy: NDArray) -> tuple[NDArray, NDArray]:
    """The shifts with those more than MAX_JUMP from their neighbours' median made NaN."""
    rows, cols = dx.shape
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


def line_up_fine(
    known: NDArray, seen: NDArray, usable: NDArray, step: int = FINE_STEP
) -> tuple[NDArray, NDArray]:
    """line_up's second pass, on a print already lined up: per FINE_STEP spot, the shift left
    (translation only, by ECC over FINE_WINDOW px). NaN where unsure."""
    h, w = known.shape
    rows, cols = spots((h, w), step)
    dx = np.full((rows, cols), np.nan, np.float32)
    dy = np.full((rows, cols), np.nan, np.float32)
    known = cv2.GaussianBlur(known, (0, 0), 1.0)
    everywhere = np.ones((FINE_WINDOW, FINE_WINDOW), np.uint8)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)
    half = FINE_WINDOW // 2
    for i in range(rows):
        for j in range(cols):
            cy, cx = i * step + step // 2, j * step + step // 2
            if cy - half < 0 or cx - half < 0 or cy + half > h or cx + half > w:
                continue
            win = (slice(cy - half, cy + half), slice(cx - half, cx + half))
            if not usable[win].all() or float(known[win].mean()) < MIN_INK:
                continue
            warp = np.eye(2, 3, dtype=np.float32)
            try:
                match, fitted = cv2.findTransformECC(
                    seen[win], known[win], warp, cv2.MOTION_TRANSLATION, criteria, everywhere, 1
                )
            except cv2.error:
                continue
            if match >= MIN_CORRELATION:
                found = np.asarray(fitted)
                dx[i, j], dy[i, j] = -float(found[0, 2]), -float(found[1, 2])
    return _dropped_jumps(dx, dy)


def line_up_maps(
    flat: NDArray, maps: tuple[NDArray, NDArray], seen: NDArray, usable: NDArray
) -> tuple[NDArray, NDArray]:
    """The page maps lined up with the photo: line_up per STEP spot, then line_up_fine per
    FINE_STEP spot on the result."""
    first = refine_maps(maps, line_up(known_print(flat, maps), seen, usable))
    return refine_maps(first, line_up_fine(known_print(flat, first), seen, usable), FINE_STEP)


def map_off_mm(shift: tuple[NDArray, NDArray], maps: tuple[NDArray, NDArray]) -> float:
    """The most the page map was off where it was lined up (line_up's shift), in millimetres of
    the flat page."""
    dx, dy = shift
    found = ~np.isnan(dx) & ~np.isnan(dy)
    if not found.any():
        return 0.0
    map_x = maps[0]
    both = (map_x[:, 1:] >= 0) & (map_x[:, :-1] >= 0)
    flat_per_photo = float(np.median(np.abs(np.diff(map_x, axis=1))[both])) if both.any() else 1.0
    return float(np.hypot(dx[found], dy[found]).max()) * flat_per_photo / FLAT_MM


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


def measured_spots(known: NDArray, usable: NDArray, step: int = STEP) -> list[tuple[int, int]]:
    """The spots wholly on the usable page, with REACH around them, and with enough print."""
    h, w = known.shape
    rows, cols = spots((h, w), step)
    found = []
    for i in range(rows):
        for j in range(cols):
            y0, x0 = i * step, j * step
            if y0 - REACH < 0 or x0 - REACH < 0 or y0 + step + REACH > h or x0 + step + REACH > w:
                continue
            spot = (slice(y0, y0 + step), slice(x0, x0 + step))
            if usable[spot].all() and float(known[spot].mean()) >= MIN_INK:
                found.append((i, j))
    return found


class _Spot:
    """One spot's print (with REACH around it) and the photo's darkness there."""

    def __init__(self, known: NDArray, seen: NDArray, i: int, j: int, step: int) -> None:
        y0, x0 = i * step, j * step
        self.region = known[y0 - REACH : y0 + step + REACH, x0 - REACH : x0 + step + REACH]
        self.seen = seen[y0 : y0 + step, x0 : x0 + step].astype(np.float64)
        self.at = (slice(y0, y0 + step), slice(x0, x0 + step))
        self.pad = (slice(y0 - REACH, y0 + step + REACH), slice(x0 - REACH, x0 + step + REACH))
        # log wide, log narrow, angle, and the print's leftover shift (dx, dy): lining up leaves a
        # pixel or two on a bent page, and a misplaced line looks like a blurrier one
        self.params = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        self.fitted = False
        self.blurred = self.blur(self.params)

    def blur(self, params: NDArray) -> NDArray:
        region = self.region
        if params[3] or params[4]:
            region = _moved(region, (float(params[3]), float(params[4])))
        kernel = stretched_kernel(math.exp(params[0]), math.exp(params[1]), float(params[2]))
        done = cv2.filter2D(region, -1, kernel, borderType=cv2.BORDER_CONSTANT)
        return np.asarray(done, np.float64)  # with REACH around the spot: _shape crops it


def _shape(
    blurred: NDArray,
    skirted: NDArray,
    skirt: tuple[float, float],
    curve: tuple[float, float],
    sharpen: tuple[float, float],
) -> NDArray:
    """A spot's drawn darkness from its blurred print and skirt (both with REACH around it, which
    the sharpening needs), cropped to the spot."""
    ink = sharpened((1 - skirt[0]) * blurred + skirt[0] * skirted, sharpen)
    return np.asarray(darkness_curve(ink[REACH:-REACH, REACH:-REACH], curve), np.float64)


def _strength(m: NDArray, t: NDArray) -> float:
    return max(float((m * t).sum() / max(float((m * m).sum()), 1e-9)), 0.0)


def _steady_strength(m: NDArray, t: NDArray) -> float:
    """The ink strength from the pixels a sub-pixel misplacement barely changes: line middles and
    the inside of thick ink, where the drawn darkness is high and flat."""
    gradient = (
        np.hypot(cv2.Sobel(m, cv2.CV_64F, 1, 0, ksize=3), cv2.Sobel(m, cv2.CV_64F, 0, 1, ksize=3))
        / 8
    )
    steady = (m > 0.3 * m.max()) & (gradient < 0.03)  # noqa: PLR2004
    if steady.sum() < 20:  # noqa: PLR2004
        return _strength(m, t)
    return _strength(m[steady], t[steady])


def fit_look(
    flat: NDArray, maps: tuple[NDArray, NDArray], seen: NDArray, usable: NDArray, step: int = STEP
) -> Look | None:
    """The look that, drawn through these (lined-up) maps, best gives the photo's darkness: per
    spot a stretched blur and an ink strength, per photo the skirt and the darkness curve. In
    turn: the spots' blurs, the skirt (both on pixels), then the curve and strength (calibrate).
    Spots without print take their nearest spot's values. None with fewer than MIN_SPOTS spots."""
    known = known_print(flat, maps)
    places = measured_spots(known, usable, step)
    if len(places) < MIN_SPOTS:
        return None
    fits = [_Spot(known, seen, i, j, step) for i, j in places]
    skirt, curve, sharpen = START_SKIRT, START_CURVE, START_SHARPEN
    low, high = math.log(0.3), math.log(6.0)
    for _ in range(ROUNDS):
        skirted = cv2.GaussianBlur(known, (0, 0), skirt[1]).astype(np.float64)
        for spot in fits:
            far = skirted[spot.pad]

            def residual(
                p: NDArray,
                spot: _Spot = spot,
                far: NDArray = far,
                skirt: tuple[float, float] = skirt,
                curve: tuple[float, float] = curve,
                sharpen: tuple[float, float] = sharpen,
            ) -> NDArray:
                m = _shape(spot.blur(p), far, skirt, curve, sharpen)
                return np.asarray((spot.seen - _strength(m, spot.seen) * m).ravel())

            # from a round blur the fit has no pull towards any direction: the first round starts
            # from stretched guesses at four angles and keeps the best
            starts = (
                [spot.params]
                if spot.fitted
                else [
                    np.array([math.log(1.3), math.log(0.9), a, 0.0, 0.0])
                    for a in (0, math.pi / 4, math.pi / 2, 3 * math.pi / 4)
                ]
            )
            found = min(
                (
                    least_squares(
                        residual,
                        s,
                        bounds=(
                            [low, low, -np.inf, -MAX_LEFTOVER, -MAX_LEFTOVER],
                            [high, high, np.inf, MAX_LEFTOVER, MAX_LEFTOVER],
                        ),
                        x_scale=[0.2, 0.2, 0.3, 0.3, 0.3],
                        diff_step=DIFF_STEP,
                    )
                    for s in starts
                ),
                key=lambda r: r.cost,
            )
            spot.params = found.x
            spot.fitted = True
            spot.blurred = spot.blur(found.x)

        skirt = _fitted_skirt(fits, known, skirt, curve, sharpen)
        look = calibrate(
            _assembled(places, fits, known, skirt, curve, sharpen, step), flat, maps, seen, usable
        )
        curve = look.curve
    # the phone's sharpening last, with the blur and skirt held: fed back into their fits, they
    # drift together (a wider blur or a bigger skirt, sharpened more, gives the same lines)
    mix, checker = ink_mix(flat, maps, look), Checker(seen, known, usable)
    best: tuple[float, Look] | None = None
    for amount in SHARPEN_TRIES:
        trial = calibrate(
            replace(look, sharpen=(amount, SHARPEN_RADIUS)),
            flat,
            maps,
            seen,
            usable,
            mix=mix,
            checker=checker,
        )
        score = worst(checker.check(finished_ink(mix, maps, trial)))
        if best is None or score < best[0]:
            best = (score, trial)
    assert best is not None  # noqa: S101
    return best[1]


def worst(check: "LookCheck") -> float:
    """How far the check's worst measure is off, in its own tolerances (1 = at the limit)."""
    ratios = (check.line_width, check.line_darkness, check.line_ink, check.solid)
    off = [abs(r - 1) / TOLERANCE for r in ratios] + [abs(check.skirt) / SKIRT_TOLERANCE]
    return max(math.inf if math.isnan(o) else o for o in off)


def _fitted_skirt(
    fits: list[_Spot],
    known: NDArray,
    skirt: tuple[float, float],
    curve: tuple[float, float],
    sharpen: tuple[float, float],
) -> tuple[float, float]:
    """The skirt (share, sigma) that, with the spots' blurs, best gives the photo's pixels."""

    def residual(q: NDArray) -> NDArray:
        sk = (float(q[0]), math.exp(q[1]))
        spread = cv2.GaussianBlur(known, (0, 0), sk[1]).astype(np.float64)
        parts = []
        for spot in fits:
            m = _shape(spot.blurred, spread[spot.pad], sk, curve, sharpen)
            parts.append((spot.seen - _strength(m, spot.seen) * m).ravel())
        return np.concatenate(parts)

    found = least_squares(
        residual,
        [skirt[0], math.log(skirt[1])],
        bounds=([0, math.log(2)], [0.4, math.log(MAX_SKIRT)]),
        x_scale=[0.05, 0.2],
        diff_step=DIFF_STEP,
    )
    return float(found.x[0]), math.exp(found.x[1])


def _assembled(
    places: list[tuple[int, int]],
    fits: list[_Spot],
    known: NDArray,
    skirt: tuple[float, float],
    curve: tuple[float, float],
    sharpen: tuple[float, float],
    step: int,
) -> Look:
    """The spots' fits as a look over the whole page: smoothed, and filled where unmeasured."""
    rows, cols = spots(known.shape, step)
    blur = np.full((rows, cols, 3), np.nan, np.float32)
    strength = np.full((rows, cols), np.nan, np.float32)
    skirted = cv2.GaussianBlur(known, (0, 0), skirt[1]).astype(np.float64)
    for (i, j), spot in zip(places, fits, strict=True):
        big, small, turn = (
            math.exp(spot.params[0]),
            math.exp(spot.params[1]),
            float(spot.params[2]),
        )
        if small > big:
            big, small, turn = small, big, turn + math.pi / 2
        blur[i, j] = (big, small, turn % math.pi)
        strength[i, j] = _steady_strength(
            _shape(spot.blurred, skirted[spot.pad], skirt, curve, sharpen), spot.seen
        )
    return Look(
        filled(_smoothed_blur(blur), (1.0, 1.0, 0.0)),
        filled(_smoothed(strength), 1.0),
        skirt,
        curve,
        step,
        sharpen,
    )


def _smoothed(grid: NDArray) -> NDArray:
    """Each measured spot as the median of the measured spots around it (3 x 3): one spot's
    print may hold too little to measure it well, and the camera's blur changes smoothly."""
    rows, cols = grid.shape[:2]
    out = grid.copy()
    for i in range(rows):
        for j in range(cols):
            if np.isnan(grid[i, j]).any():
                continue
            near = grid[max(0, i - 1) : i + 2, max(0, j - 1) : j + 2].reshape(-1, *grid.shape[2:])
            near = near[~np.isnan(near).reshape(len(near), -1).any(axis=1)]
            out[i, j] = np.median(near, axis=0)
    return out


def _smoothed_blur(blur: NDArray) -> NDArray:
    """_smoothed on the blur's covariance (xx, yy, xy), where directions average properly."""
    wide, narrow, angle = blur[..., 0], blur[..., 1], blur[..., 2]
    c, s = np.cos(angle), np.sin(angle)
    cov = np.stack(
        [
            wide**2 * c**2 + narrow**2 * s**2,
            wide**2 * s**2 + narrow**2 * c**2,
            (wide**2 - narrow**2) * c * s,
        ],
        axis=-1,
    )
    xx, yy, xy = np.moveaxis(_smoothed(cov), -1, 0)
    mean, half = (xx + yy) / 2, np.sqrt(((xx - yy) / 2) ** 2 + xy**2)
    out = np.stack(
        [
            np.sqrt(mean + half),
            np.sqrt(np.maximum(mean - half, 1e-6)),
            (0.5 * np.arctan2(2 * xy, xx - yy)) % np.pi,
        ],
        axis=-1,
    )
    return np.asarray(out, np.float32)


@dataclass
class LookCheck:
    """How the drawn print compares with the photo, on measures a fraction-of-a-pixel
    misplacement can't fake: per staff line (each found where it sits in its own image) its
    width, peak darkness and total ink, drawn / real; the middle of thick ink, drawn / real; and
    the darkness 3-9 px out from the ink, drawn minus real (trimmed means). Medians over the page.
    """

    line_width: float
    line_darkness: float
    line_ink: float
    solid: float
    skirt: float


def _lines(profile: NDArray) -> list[tuple[float, float, float, float]]:
    """(centre, spread, peak, total ink) of every staff line across a darkness profile."""
    found = []
    for i in range(4, len(profile) - 4):
        v = profile[i]
        if (
            v > 0.25  # noqa: PLR2004
            and v == profile[i - 3 : i + 4].max()
            and profile[i - 4] < 0.5 * v
            and profile[i + 4] < 0.5 * v
        ):
            win = np.clip(profile[i - 4 : i + 5], 0, None)
            x = np.arange(-4, 5)
            c = float((win * x).sum() / win.sum())
            found.append(
                (
                    i + c,
                    float(np.sqrt((win * (x - c) ** 2).sum() / win.sum())),
                    float(v),
                    float(win.sum()),
                )
            )
    return found


def _trimmed(values: NDArray, share: float = 0.1) -> float:
    """The mean of the values between the share-th and (1 - share)-th quantiles."""
    low, high = np.quantile(values, [share, 1 - share])
    kept = values[(values >= low) & (values <= high)]
    return float(kept.mean()) if kept.size else float(np.median(values))


class Checker:
    """check_look for one photo, ready for many drawings: the photo's own staff lines and the
    places to measure are found once."""

    def __init__(self, seen: NDArray, known: NDArray, usable: NDArray) -> None:
        self.windows: list[tuple[int, int, list[tuple[float, float, float, float]]]] = []
        h, w = seen.shape
        tall, wide = PROFILE
        for y0 in range(0, h - tall, tall // 2):
            for x0 in range(0, w - wide, wide):
                if usable[y0 : y0 + tall, x0 : x0 + wide].all():
                    found = _lines(seen[y0 : y0 + tall, x0 : x0 + wide].mean(axis=1))
                    if found:
                        self.windows.append((y0, x0, found))
        ink = (known > 0.5).astype(np.uint8)  # noqa: PLR2004
        inside = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
        outside = cv2.distanceTransform(1 - ink, cv2.DIST_L2, 3)
        self.core = usable & (inside > 2.5)  # noqa: PLR2004
        self.skirt = usable & (ink == 0) & (outside >= 3) & (outside < 9)  # noqa: PLR2004
        self.seen_core = float(np.median(seen[self.core])) if self.core.any() else math.nan
        # a trimmed mean: out there the photo's darkness is a few grey levels, and a median of
        # whole grey levels snaps to one of them (off by up to half a level); trimmed, stray ink
        # the template doesn't have (dust, a pencil dot) doesn't count
        self.seen_skirt = _trimmed(seen[self.skirt]) if self.skirt.any() else math.nan

    def check(self, drawn: NDArray) -> LookCheck:
        ratios: list[tuple[float, float, float]] = []
        tall, wide = PROFILE
        for y0, x0, real in self.windows:
            got = _lines(drawn[y0 : y0 + tall, x0 : x0 + wide].mean(axis=1))
            for c, spread, peak, total in real:
                near = [g for g in got if abs(g[0] - c) < 2]  # noqa: PLR2004
                if len(near) == 1:
                    ratios.append((near[0][1] / spread, near[0][2] / peak, near[0][3] / total))
        width, darkness, line_ink = (
            np.median(np.array(ratios), axis=0) if ratios else (math.nan,) * 3
        )
        # no darkness where the print is solid black: a map so far off that it measures paper
        solid = (
            float(np.median(drawn[self.core])) / self.seen_core if self.seen_core > 0 else math.nan
        )
        far = _trimmed(drawn[self.skirt]) - self.seen_skirt if self.skirt.any() else math.nan
        return LookCheck(float(width), float(darkness), float(line_ink), solid, far)


def check_look(seen: NDArray, drawn: NDArray, known: NDArray, usable: NDArray) -> LookCheck:
    """The check of a look: the photo's darkness (seen) against the known print drawn with it
    (drawn), known being the sharp print lined up with the photo."""
    return Checker(seen, known, usable).check(drawn)


def failed_measure(check: LookCheck) -> str | None:
    """Every measure outside its tolerance, as text ('line width 0.88x, skirt -0.0100'), or None
    when the look passes."""
    wrong = []
    for name, value in (
        ("line width", check.line_width),
        ("line darkness", check.line_darkness),
        ("line ink", check.line_ink),
        ("solid black", check.solid),
    ):
        if math.isnan(value):
            wrong.append(f"{name}: nothing to measure")
        elif abs(value - 1) > TOLERANCE:
            wrong.append(f"{name} {value:.2f}x")
    if math.isnan(check.skirt):
        wrong.append("skirt: nothing to measure")
    elif abs(check.skirt) > SKIRT_TOLERANCE:
        wrong.append(f"skirt {check.skirt:+.4f}")
    return ", ".join(wrong) or None


def calibrate(
    look: Look,
    flat: NDArray,
    maps: tuple[NDArray, NDArray],
    seen: NDArray,
    usable: NDArray,
    *,
    mix: NDArray | None = None,
    checker: Checker | None = None,
) -> Look:
    """The look with its darkness curve and ink strength set on the measures sub-pixel
    misplacement can't fake (a pixel fit is ruled by the ink edges, where leftover misplacement
    lives): the curve's bend so that staff lines and solid black come out equally right, then
    the strength scaled so that both match the photo. mix (ink_mix) and checker can be given when
    they are already at hand. Unchanged without staff lines or solid black to measure."""
    checker = checker or Checker(seen, known_print(flat, maps), usable)
    mix = ink_mix(flat, maps, look) if mix is None else mix  # the slow part

    def measured(gamma: float) -> LookCheck:
        return checker.check(finished_ink(mix, maps, replace(look, curve=(gamma, TOE))))

    def apart(gamma: float) -> float:
        check = measured(gamma)
        return math.log(check.line_darkness / check.solid)

    gamma = look.curve[0]
    try:
        low, high = apart(GAMMAS[0]), apart(GAMMAS[1])
        if low * high < 0:
            gamma = float(brentq(apart, *GAMMAS, xtol=0.005))
        elif math.isfinite(low) and math.isfinite(high):
            gamma = GAMMAS[0] if abs(low) < abs(high) else GAMMAS[1]
    except (ValueError, ZeroDivisionError, RuntimeError):  # nothing to measure, or no root
        return look
    check = measured(gamma)
    scale = 1 / math.sqrt(check.line_darkness * check.solid)
    if not math.isfinite(scale):
        return look
    return replace(look, strength=look.strength * np.float32(scale), curve=(gamma, TOE))


def measure_and_check(
    cond: PhotoConditions, flat: NDArray
) -> tuple[Look, tuple[NDArray, NDArray]] | str:
    """A photo's look and its finer page maps, measured from the template page that was
    printed (flat), or why the photo is left out."""
    seen = seen_darkness(cond)
    ok = usable(cond)
    shift = line_up(known_print(flat, (cond.map_x, cond.map_y)), seen, ok)
    if int((~np.isnan(shift[0])).sum()) < MIN_SPOTS:
        return "too little print lined up with its template"
    off = map_off_mm(shift, (cond.map_x, cond.map_y))
    if off > MAP_OFF_MAX_MM:
        return f"page map off by {off:.0f} mm, more than erasing allows"
    maps = refine_maps((cond.map_x, cond.map_y), shift)
    maps = refine_maps(maps, line_up_fine(known_print(flat, maps), seen, ok), FINE_STEP)
    look = fit_look(flat, maps, seen, ok)
    if look is None:
        return "too little print to measure"
    reason = failed_measure(
        check_look(seen, draw_ink(flat, maps, look), known_print(flat, maps), ok)
    )
    return reason if reason is not None else (look, maps)
