import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from training.omr_datasets.photo_look import (
    NO_CURVE,
    NO_SHARPEN,
    STEP,
    Look,
    darkness_curve,
    draw_ink,
    load_look,
    save_look,
    spots,
)
from training.omr_datasets.photo_maps import FH, FW

SHAPE = (700, 495)  # a small photo of the whole page


def page_maps(shape: tuple[int, int] = SHAPE) -> tuple[NDArray, NDArray]:
    """Maps laying the flat page over the whole photo."""
    h, w = shape
    qx, qy = np.meshgrid(np.linspace(0, FW - 1, w), np.linspace(0, FH - 1, h))
    return qx.astype(np.float32), qy.astype(np.float32)


def flat_with_dot() -> NDArray:
    flat = np.full((FH, FW), 255, np.uint8)
    flat[FH // 2 - 1 : FH // 2 + 2, FW // 2 - 1 : FW // 2 + 2] = 0
    return flat


def even_look(
    wide: float, narrow: float, angle: float, skirt: tuple[float, float] = (0.0, 1.0)
) -> Look:
    rows, cols = spots(SHAPE)
    blur = np.zeros((rows, cols, 3), np.float32)
    blur[...] = (wide, narrow, angle)
    return Look(blur, np.ones((rows, cols), np.float32), skirt, NO_CURVE)


def spread(d: NDArray) -> tuple[float, float, float]:
    """Wide and narrow standard deviation of a darkness blob, and the angle of its wide axis
    (in x, y pixel coordinates)."""
    ys, xs = np.mgrid[: d.shape[0], : d.shape[1]].astype(np.float64)
    total = d.sum()
    mx, my = (d * xs).sum() / total, (d * ys).sum() / total
    cov = (
        np.array(
            [
                [(d * (xs - mx) ** 2).sum(), (d * (xs - mx) * (ys - my)).sum()],
                [(d * (xs - mx) * (ys - my)).sum(), (d * (ys - my) ** 2).sum()],
            ]
        )
        / total
    )
    values, vectors = np.linalg.eigh(cov)
    wide = vectors[:, 1]
    return math.sqrt(values[1]), math.sqrt(values[0]), math.atan2(wide[1], wide[0]) % math.pi


class TestPhotoLook(unittest.TestCase):
    def test_curve_is_straight_at_no_curve_and_monotone(self) -> None:
        x = np.linspace(0, 1, 101)
        self.assertTrue(np.allclose(darkness_curve(x, NO_CURVE), x))
        y = darkness_curve(x, (0.8, 0.02))
        self.assertTrue(np.all(np.diff(y) > 0))
        self.assertAlmostEqual(float(y[0]), 0.0, places=6)
        self.assertAlmostEqual(float(y[-1]), 1.0, places=6)
        self.assertGreater(float(darkness_curve(np.array([0.3]), (0.8, 0.02))[0]), 0.3)

    def test_draw_ink_blurs_a_dot_with_the_stretched_gaussian(self) -> None:
        drawn = draw_ink(flat_with_dot(), page_maps(), even_look(3.0, 1.0, math.pi / 6))
        cy, cx = SHAPE[0] // 2, SHAPE[1] // 2
        wide, narrow, angle = spread(drawn[cy - 20 : cy + 21, cx - 20 : cx + 21])
        dot = 3 * SHAPE[1] / FW / math.sqrt(12)  # the dot's own spread, in photo pixels
        self.assertAlmostEqual(math.sqrt(wide**2 - dot**2), 3.0, delta=0.3)
        self.assertAlmostEqual(math.sqrt(max(narrow**2 - dot**2, 0)), 1.0, delta=0.2)
        self.assertAlmostEqual(math.degrees(angle), 30, delta=5)

    def test_skirt_puts_its_share_far_out(self) -> None:
        plain = draw_ink(flat_with_dot(), page_maps(), even_look(1.0, 1.0, 0.0))
        skirted = draw_ink(flat_with_dot(), page_maps(), even_look(1.0, 1.0, 0.0, (0.1, 6.0)))
        self.assertAlmostEqual(float(skirted.sum() / plain.sum()), 1.0, delta=0.02)
        ys, xs = np.mgrid[: SHAPE[0], : SHAPE[1]]
        r = np.hypot(ys - SHAPE[0] // 2, xs - SHAPE[1] // 2)
        ring = (r >= 6) & (r <= 12)
        self.assertGreater(float(skirted[ring].sum()), 5 * float(plain[ring].sum()))

    def test_sharpening_narrows_a_line_and_keeps_its_ink(self) -> None:
        flat = np.full((FH, FW), 255, np.uint8)
        flat[FH // 2 - 6 : FH // 2 + 6, 400:2000] = 0  # a staff line across the page
        plain = draw_ink(flat, page_maps(), even_look(1.5, 1.5, 0.0))
        look = even_look(1.5, 1.5, 0.0)
        look.sharpen = (0.3, 2.0)
        sharp = draw_ink(flat, page_maps(), look)
        column = SHAPE[1] // 2
        a, b = plain[:, column], sharp[:, column]
        ys = np.arange(SHAPE[0], dtype=np.float64)

        def width(p: NDArray) -> float:
            centre = (p * ys).sum() / p.sum()
            return float(np.sqrt((p * (ys - centre) ** 2).sum() / p.sum()))

        self.assertLess(width(b), 0.95 * width(a))
        self.assertGreater(float(b.max()), float(a.max()))
        self.assertAlmostEqual(float(b.sum() / a.sum()), 1.0, delta=0.03)
        self.assertEqual(even_look(1.0, 1.0, 0.0).sharpen, NO_SHARPEN)

    def test_draw_ink_is_finite_at_the_page_edge(self) -> None:
        map_x, map_y = page_maps()
        map_x[:40], map_y[:40] = -1, -1  # the top of the photo is off the page
        map_x[:, :30], map_y[:, :30] = -1, -1
        flat = np.full((FH, FW), 255, np.uint8)
        flat[:, :400] = 0  # ink right up to the page's left edge
        look = even_look(1.5, 1.0, 0.0)
        look.blur[0, 0] = (0.0, 0.0, 0.0)
        look.blur[1, 1] = np.nan
        drawn = draw_ink(flat, (map_x, map_y), look)
        self.assertTrue(np.all(np.isfinite(drawn)))
        self.assertEqual(float(np.abs(drawn[:40]).max()), 0.0)
        self.assertEqual(float(np.abs(drawn[:, :30]).max()), 0.0)
        self.assertGreater(float(drawn[300, 40]), 0.5)

    def test_look_round_trip(self) -> None:
        look = even_look(2.0, 1.5, 0.3, (0.1, 6.0))
        look.strength[2, 3] = 0.8
        look.sharpen = (0.25, 1.5)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "photo.look.npz"
            save_look(path, look)
            back = load_look(path)
        self.assertTrue(np.array_equal(back.blur, look.blur))
        self.assertTrue(np.array_equal(back.strength, look.strength))
        self.assertEqual((back.skirt, back.curve, back.step), (look.skirt, look.curve, STEP))
        self.assertEqual(back.sharpen, look.sharpen)
