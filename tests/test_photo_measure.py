import math
import time
import unittest

import cv2
import numpy as np
from numpy.typing import NDArray

from training.omr_datasets.photo_look import STEP, Look, draw_ink, known_print, spots
from training.omr_datasets.photo_maps import FH, FW, PhotoConditions
from training.omr_datasets.photo_measure import (
    MIN_INK,
    MIN_SPOTS,
    TOLERANCE,
    calibrate,
    check_look,
    failed_measure,
    fit_look,
    line_up,
    line_up_maps,
    measure_and_check,
    refine_maps,
)

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


def true_look(shape: tuple[int, int] = SHAPE) -> Look:
    """Blur from (1.0, 0.8, 0) at the top to (2.2, 1.2, pi/4) at the bottom, strength 0.85 to
    0.95, a skirt of 10% at sigma 6 and a darkness curve (0.8, 0.02)."""
    rows, cols = spots(shape)
    t = np.repeat((np.arange(rows) / (rows - 1))[:, None], cols, axis=1).astype(np.float32)
    blur = np.stack([1.0 + 1.2 * t, 0.8 + 0.4 * t, t * math.pi / 4], axis=-1)
    return Look(
        blur.astype(np.float32), (0.85 + 0.1 * t).astype(np.float32), (0.1, 6.0), (0.8, 0.02)
    )


def photo_seen(flat: NDArray, maps: tuple[NDArray, NDArray], look: Look, paper: float) -> NDArray:
    """The darkness a photo with this paper level shows: grey levels rounded as a camera would,
    with a little grain."""
    drawn = draw_ink(flat, maps, look)
    grain = np.random.default_rng(3).normal(0, 1.0, drawn.shape)
    grey = np.clip(np.round(paper * (1 - drawn) + grain), 0, 255)
    return np.asarray(1 - grey / paper, np.float32)


def with_print(flat: NDArray, maps: tuple[NDArray, NDArray]) -> NDArray:
    """The spots (row, column) that carry print, as fit_look measures them."""
    known = known_print(flat, maps)
    rows, cols = spots(known.shape)
    return np.array(
        [
            [
                (i + 1) * STEP <= known.shape[0]
                and (j + 1) * STEP <= known.shape[1]
                and known[i * STEP : (i + 1) * STEP, j * STEP : (j + 1) * STEP].mean() >= MIN_INK
                for j in range(cols)
            ]
            for i in range(rows)
        ]
    )


def angle_apart(a: NDArray, b: NDArray) -> NDArray:
    d = np.abs(a - b) % math.pi
    return np.minimum(d, math.pi - d)


class TestFit(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.flat = synthetic_print()
        cls.maps = page_maps()
        cls.truth = true_look()
        cls.usable = np.ones(SHAPE, bool)
        cls.bright = fit_look(
            cls.flat, cls.maps, photo_seen(cls.flat, cls.maps, cls.truth, 230), cls.usable
        )
        cls.dim = fit_look(
            cls.flat, cls.maps, photo_seen(cls.flat, cls.maps, cls.truth, 120), cls.usable
        )
        cls.spots = with_print(cls.flat, cls.maps)

    def test_fit_recovers_a_known_look(self) -> None:
        got = self.bright
        assert got is not None  # noqa: S101
        s, t = got.blur[self.spots], self.truth.blur[self.spots]
        wide_ok = np.abs(s[:, 0] / t[:, 0] - 1) <= 0.1
        self.assertGreaterEqual(float(wide_ok.mean()), 0.8, np.round(s[:, 0] / t[:, 0], 2))
        stretched = t[:, 0] / t[:, 1] > 1.3
        apart = np.degrees(angle_apart(s[stretched, 2], t[stretched, 2]))
        self.assertGreaterEqual(float((apart <= 15).mean()), 0.8, np.round(apart))
        strength = got.strength[self.spots] / self.truth.strength[self.spots]
        self.assertGreaterEqual(
            float((np.abs(strength - 1) <= 0.05).mean()), 0.8, np.round(strength, 2)
        )
        self.assertAlmostEqual(got.skirt[0], 0.1, delta=0.03)
        self.assertAlmostEqual(got.skirt[1], 6.0, delta=1.5)
        self.assertAlmostEqual(got.curve[0], 0.8, delta=0.05)

    def test_a_sharpened_photo_is_matched(self) -> None:
        # a slightly narrower blur can stand in for some sharpening: what counts is the photo
        sharpened = Look(
            self.truth.blur,
            self.truth.strength,
            self.truth.skirt,
            self.truth.curve,
            sharpen=(0.25, 2.0),
        )
        seen = photo_seen(self.flat, self.maps, sharpened, 230)
        got = fit_look(self.flat, self.maps, seen, self.usable)
        assert got is not None  # noqa: S101
        known = known_print(self.flat, self.maps)
        check = check_look(seen, draw_ink(self.flat, self.maps, got), known, self.usable)
        for name in ("line_width", "line_darkness", "line_ink", "solid"):
            self.assertAlmostEqual(getattr(check, name), 1.0, delta=TOLERANCE, msg=name)
        # sharpening and the skirt both change the paper beside the ink, and can't be told apart
        # fully: there the drawn photo comes within 0.005 (the check asks 0.003, so a strongly
        # sharpened photo can be left out: on the safe side)
        self.assertLessEqual(abs(check.skirt), 0.005)
        self.assertLess(self.bright.sharpen[0], 0.08)  # the unsharpened photo: next to none

    def test_dim_photo_gives_the_same_look(self) -> None:
        assert self.bright is not None and self.dim is not None  # noqa: S101
        for a, b in (
            (self.bright.blur[self.spots, 0], self.dim.blur[self.spots, 0]),
            (self.bright.strength[self.spots], self.dim.strength[self.spots]),
        ):
            self.assertLessEqual(float(np.median(np.abs(b / a - 1))), 0.05)

    def test_small_leftover_shifts_are_not_taken_for_blur(self) -> None:
        # a bent page: shifts of a few pixels that change faster than the 96 px spots can follow
        # (real photos keep 0.5 px typically, 2-3 px at worst, after the first line-up)
        h, w = SHAPE
        ys, xs = np.mgrid[:h, :w].astype(np.float32)
        jitter = (
            2.5 * np.sin(xs / 37.0) * np.cos(ys / 53.0),
            2.5 * np.cos(xs / 47.0) * np.sin(ys / 41.0),
        )
        seen = photo_seen(self.flat, shifted_maps(self.maps, jitter), self.truth, 230)
        got = fit_look(
            self.flat, line_up_maps(self.flat, self.maps, seen, self.usable), seen, self.usable
        )
        assert got is not None  # noqa: S101
        ratio = got.blur[self.spots, 0] / self.truth.blur[self.spots, 0]
        self.assertGreaterEqual(float((np.abs(ratio - 1) <= 0.1).mean()), 0.8, np.round(ratio, 2))

    def test_sparse_print_still_covers_the_page(self) -> None:
        sparse = self.flat.copy()
        sparse[:, FW // 2 :] = 255
        sparse[FH // 2 :] = 255
        got = fit_look(
            sparse, self.maps, photo_seen(sparse, self.maps, self.truth, 230), self.usable
        )
        assert got is not None  # noqa: S101
        self.assertTrue(np.all(np.isfinite(got.blur)) and np.all(np.isfinite(got.strength)))
        tiny = np.full((FH, FW), 255, np.uint8)
        tiny[400:520, 300:900] = sparse[400:520, 300:900]
        seen = photo_seen(tiny, self.maps, self.truth, 230)
        self.assertLess(int(with_print(tiny, self.maps).sum()), MIN_SPOTS)
        self.assertIsNone(fit_look(tiny, self.maps, seen, self.usable))


def conditions_of(
    seen_flat: NDArray,
    maps: tuple[NDArray, NDArray],
    look: Look,
    true_maps: tuple[NDArray, NDArray],
) -> PhotoConditions:
    """A photo of `seen_flat` drawn with `look` through `true_maps`, on paper with a gentle shadow,
    whose page map (`maps`) is a few pixels off, as UVDoc's and the marks' are."""
    h, w = maps[0].shape
    ys = np.mgrid[:h, :w][0].astype(np.float32)
    paper_grey = 225 - 25 * ys / h
    drawn = draw_ink(seen_flat, true_maps, look)
    grain = np.random.default_rng(5).normal(0, 1.0, drawn.shape)
    grey = np.clip(np.round(paper_grey * (1 - drawn) + grain), 0, 255).astype(np.uint8)
    photo = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
    paper = np.repeat(paper_grey[..., None], 3, axis=2).astype(np.float32)
    return PhotoConditions(photo, maps[0], maps[1], np.ones((h, w), bool), paper)


class TestCheck(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.flat = synthetic_print()
        cls.maps = page_maps()
        cls.look = true_look()
        cls.seen = photo_seen(cls.flat, cls.maps, cls.look, 230)
        cls.known = known_print(cls.flat, cls.maps)
        cls.usable = np.ones(SHAPE, bool)

    def check_of(self, look: Look) -> object:
        return check_look(self.seen, draw_ink(self.flat, self.maps, look), self.known, self.usable)

    def test_check_passes_the_model_that_drew_the_photo(self) -> None:
        check = self.check_of(self.look)
        for name in ("line_width", "line_darkness", "line_ink", "solid"):
            self.assertAlmostEqual(getattr(check, name), 1.0, delta=0.02, msg=name)
        self.assertLessEqual(abs(check.skirt), 0.001)
        self.assertIsNone(failed_measure(check))

    def test_skirt_measure_ignores_stray_ink(self) -> None:
        # specks of ink the template doesn't have (dust, a pencil dot) near the print
        stray = self.seen.copy()
        rng = np.random.default_rng(7)
        for y, x in zip(
            rng.integers(50, SHAPE[0] - 50, 400), rng.integers(50, SHAPE[1] - 50, 400), strict=True
        ):
            stray[y : y + 2, x : x + 2] = 0.6
        check = check_look(
            stray, draw_ink(self.flat, self.maps, self.look), self.known, self.usable
        )
        self.assertLessEqual(abs(check.skirt), 0.001)

    def test_check_names_what_is_wrong(self) -> None:
        blurrier = Look(
            self.look.blur * np.array([2, 2, 1], np.float32),
            self.look.strength,
            self.look.skirt,
            self.look.curve,
        )
        reason = failed_measure(self.check_of(blurrier))
        assert reason is not None  # noqa: S101
        self.assertTrue("line width" in reason or "line darkness" in reason, reason)
        no_skirt = Look(self.look.blur, self.look.strength, (0.0, 1.0), self.look.curve)
        reason = failed_measure(self.check_of(no_skirt))
        assert reason is not None  # noqa: S101
        self.assertIn("skirt", reason)

    def test_calibrate_matches_line_darkness_and_solid_black(self) -> None:
        off = Look(self.look.blur, self.look.strength * 0.9, self.look.skirt, (1.0, 0.02))
        before = self.check_of(off)
        self.assertGreater(abs(before.line_darkness - 1), 0.05)
        fixed = calibrate(off, self.flat, self.maps, self.seen, self.usable)
        after = self.check_of(fixed)
        self.assertAlmostEqual(after.line_darkness, 1.0, delta=0.02)
        self.assertAlmostEqual(after.solid, 1.0, delta=0.02)
        self.assertAlmostEqual(fixed.curve[0], 0.8, delta=0.1)

    def test_measure_and_check_end_to_end(self) -> None:
        true_maps = shifted_maps(self.maps, smooth_shift())
        cond = conditions_of(self.flat, self.maps, self.look, true_maps)
        start = time.time()
        result = measure_and_check(cond, self.flat)
        self.assertLess(time.time() - start, 120)
        self.assertNotIsInstance(result, str, result)
        other = measure_and_check(cond, synthetic_print(seed=1))
        self.assertIsInstance(other, str)
