import unittest

import cv2
import numpy as np
from numpy.typing import NDArray

from training.omr_datasets.photo_look import STEP, known_print, spots
from training.omr_datasets.photo_maps import FH, FW
from training.omr_datasets.photo_measure import line_up, refine_maps

SHAPE = (1000, 707)  # a photo of the whole page, about as fine as a real one per staff
SPACING = 42  # flat pixels between staff lines: 12 photo pixels
STAFF_ONLY = (1500, 2200)  # flat x range of the first staff kept free of notes
STAFF_TOPS = [400, 900, 1400, 1900, 2400, 2900]


def synthetic_print(seed: int = 0) -> NDArray:
    """A flat page of music-like print: staves, filled note heads with stems, slanted beams and
    arcs, placed at random, except a stretch of the first staff with only its lines."""
    rng = np.random.default_rng(seed)
    page = np.full((FH, FW), 255, np.uint8)
    for top in STAFF_TOPS:
        for k in range(5):
            y = top + k * SPACING
            cv2.line(page, (300, y), (2180, y), 0, 5)
        x = 340
        while x < 2140:
            if top == STAFF_TOPS[0] and STAFF_ONLY[0] - 80 < x < STAFF_ONLY[1]:
                x = STAFF_ONLY[1]
                continue
            y = top + int(rng.integers(-2, 11)) * SPACING // 2
            cv2.ellipse(page, (x, y), (24, 18), -20, 0, 360, 0, -1)
            up = rng.random() < 0.5
            stem = (x + 22, y - 140) if up else (x - 22, y + 140)
            cv2.line(page, (x + 22 if up else x - 22, y), stem, 0, 5)
            if rng.random() < 0.4:
                end = (stem[0] + 90, stem[1] + int(rng.integers(-30, 31)))
                cv2.line(page, stem, end, 0, 16)
            if rng.random() < 0.3:
                cv2.ellipse(page, (x + 60, y + 80), (90, 30), 0, 200, 340, 0, 5)
            x += int(rng.integers(90, 200))
    return page


def page_maps(shape: tuple[int, int] = SHAPE) -> tuple[NDArray, NDArray]:
    h, w = shape
    qx, qy = np.meshgrid(np.linspace(0, FW - 1, w), np.linspace(0, FH - 1, h))
    return qx.astype(np.float32), qy.astype(np.float32)


def smooth_shift(shape: tuple[int, int] = SHAPE) -> tuple[NDArray, NDArray]:
    """A shift of up to 12 photo pixels that changes slowly over the photo."""
    h, w = shape
    ys, xs = np.mgrid[:h, :w].astype(np.float32)
    dx = 8 * np.sin(xs / w * 2.5 + 0.5) + 4 * np.cos(ys / h * 3.0)
    dy = 7 * np.cos(xs / w * 2.0) - 5 * np.sin(ys / h * 2.2 + 1.0)
    return dx.astype(np.float32), dy.astype(np.float32)


def shifted_maps(
    maps: tuple[NDArray, NDArray], shift: tuple[NDArray, NDArray]
) -> tuple[NDArray, NDArray]:
    """The maps of a photo whose print sits `shift` further on: pixel x shows what the plain
    maps put at x - shift(x)."""
    h, w = maps[0].shape
    ys, xs = np.mgrid[:h, :w].astype(np.float32)
    sx, sy = xs - shift[0], ys - shift[1]
    return (
        cv2.remap(maps[0], sx, sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE),
        cv2.remap(maps[1], sx, sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE),
    )


def photo_darkness(flat: NDArray, maps: tuple[NDArray, NDArray], blur: float = 1.0) -> NDArray:
    return np.asarray(cv2.GaussianBlur(known_print(flat, maps), (0, 0), blur), np.float32)


def at_spots(field: NDArray, shape: tuple[int, int] = SHAPE) -> NDArray:
    rows, cols = spots(shape)
    ys = np.minimum(np.arange(rows) * STEP + STEP // 2, shape[0] - 1)
    xs = np.minimum(np.arange(cols) * STEP + STEP // 2, shape[1] - 1)
    return field[np.ix_(ys, xs)]


class TestLineUp(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.flat = synthetic_print()
        cls.maps = page_maps()
        cls.truth = smooth_shift()
        cls.true_maps = shifted_maps(cls.maps, cls.truth)
        cls.seen = photo_darkness(cls.flat, cls.true_maps)
        cls.known = known_print(cls.flat, cls.maps)
        cls.usable = np.ones(SHAPE, bool)
        cls.shift = line_up(cls.known, cls.seen, cls.usable)

    def test_line_up_finds_a_smooth_shift(self) -> None:
        dx, dy = self.shift
        found = ~np.isnan(dx)
        self.assertGreater(int(found.sum()), 20)
        err = np.hypot(dx - at_spots(self.truth[0]), dy - at_spots(self.truth[1]))[found]
        self.assertGreaterEqual(float((err < 0.3).mean()), 0.9, np.round(np.sort(err)[-8:], 2))

    def test_refined_maps_line_the_print_up(self) -> None:
        sharp = known_print(self.flat, self.true_maps)
        before = float(np.abs(known_print(self.flat, self.maps) - sharp).mean())
        after = float(
            np.abs(known_print(self.flat, refine_maps(self.maps, self.shift)) - sharp).mean()
        )
        self.assertLess(after, before / 4, (before, after))

    def test_a_spot_locked_one_line_off_is_dropped(self) -> None:
        # the photo's staff-only stretch of the first staff moved one more line down
        bump = (np.zeros(SHAPE, np.float32), np.zeros(SHAPE, np.float32))
        h_scale, w_scale = SHAPE[0] / FH, SHAPE[1] / FW
        y0, y1 = int((STAFF_TOPS[0] - 150) * h_scale), int((STAFF_TOPS[0] + 320) * h_scale)
        x0, x1 = int(STAFF_ONLY[0] * w_scale), int(STAFF_ONLY[1] * w_scale)
        bump[1][y0:y1, x0:x1] = SPACING * h_scale
        shift = (self.truth[0] + bump[0], self.truth[1] + bump[1])
        seen = photo_darkness(self.flat, shifted_maps(self.maps, shift))
        dx, dy = line_up(self.known, seen, self.usable)
        rows, cols = dx.shape
        i = int((STAFF_TOPS[0] + 2 * SPACING) * h_scale) // STEP
        j = int((STAFF_ONLY[0] + STAFF_ONLY[1]) / 2 * w_scale) // STEP
        if np.isnan(dy[i, j]):
            return
        near = [
            dy[a, b]
            for a in range(max(0, i - 1), min(rows, i + 2))
            for b in range(max(0, j - 1), min(cols, j + 2))
            if (a, b) != (i, j) and not np.isnan(dy[a, b])
        ]
        self.assertLess(abs(float(dy[i, j]) - float(np.median(near))), 1.0)
