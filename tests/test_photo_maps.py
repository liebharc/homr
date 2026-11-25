import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from training.omr_datasets.photo_maps import (
    ARUCO,
    apply_fit,
    find_marks,
    mark_fit,
    prepare_photo,
    upright,
)


def page_with_marks(
    number: int, size: tuple[int, int] = (1000, 700), mark: int = 80, marks: int = 8
) -> NDArray:
    """A white page with the first `marks` of the eight marks of template `number`, at its corners
    and edge middles."""
    h, w = size
    page = np.full((h, w, 3), 255, np.uint8)
    spots = [
        (20, 20),
        (w // 2 - mark // 2, 20),
        (w - 20 - mark, 20),
        (w - 20 - mark, h // 2 - mark // 2),
        (w - 20 - mark, h - 20 - mark),
        (w // 2 - mark // 2, h - 20 - mark),
        (20, h - 20 - mark),
        (20, h // 2 - mark // 2),
    ]
    for spot, (x, y) in enumerate(spots[:marks]):
        marker = cv2.aruco.generateImageMarker(ARUCO, 8 * number + spot, mark)
        page[y : y + mark, x : x + mark] = marker[..., None]
    return page


TEMPLATE_T23 = {
    "name": "T23",
    "music_box": [173, 246, 1960, 2770],
    "marks": {
        "ids": list(range(184, 192)),
        "corners": [[[59, 59], [177, 59], [177, 177], [59, 177]]] * 8,
    },
}


class TestPhotoMaps(unittest.TestCase):
    def test_mark_fit_recovers_a_known_homography(self) -> None:
        truth = (
            np.random.default_rng(0).uniform([100, 100], [2300, 3300], (20, 2)).astype(np.float32)
        )
        m = np.array([[1.02, 0.01, 5], [-0.02, 0.98, -3], [1e-6, 2e-6, 1]])
        gx, gy = apply_fit(np.linalg.inv(m), truth[:, 0], truth[:, 1])
        guess = np.stack([gx, gy], axis=1)
        x, y = apply_fit(mark_fit(guess, truth), guess[:, 0], guess[:, 1])
        self.assertTrue(np.allclose(np.stack([x, y], axis=1), truth, atol=0.5))

    def test_two_marks_use_a_shift_turn_and_scale(self) -> None:
        square = np.array([[100, 100], [200, 100], [200, 200], [100, 200]], np.float32)
        truth = np.concatenate([square, square + np.array([1500, 2000], np.float32)])
        guess = truth * np.array([1.05, 0.95]) + np.array([10, -20])  # stretched unevenly
        m = mark_fit(guess.astype(np.float32), truth)
        a, b = m[:2, 0], m[:2, 1]
        self.assertAlmostEqual(float(np.linalg.norm(a)), float(np.linalg.norm(b)), places=6)
        self.assertAlmostEqual(float(a @ b), 0.0, places=6)

    def test_a_turned_photo_is_set_upright(self) -> None:
        page = page_with_marks(number=23)
        for k in (1, 2, 3):
            turned = np.ascontiguousarray(np.rot90(page, k))
            self.assertTrue(np.array_equal(upright(turned, find_marks(turned)), page))

    def test_an_upside_down_page_is_turned_although_its_angles_wrap_round(self) -> None:
        photo = np.arange(24, dtype=np.uint8).reshape(4, 6)
        marks = {}
        for n, angle in enumerate((174.7, -176.0)):
            a = np.radians(angle)
            c0 = np.array([100.0 * n, 0.0], np.float32)
            c1 = c0 + 10 * np.array([np.cos(a), np.sin(a)], np.float32)
            marks[184 + n] = np.stack([c0, c1, c0, c0])
        self.assertTrue(np.array_equal(upright(photo, marks), np.rot90(photo, 2)))

    def test_photo_without_marks_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            blank = Path(folder) / "blank.jpg"
            cv2.imwrite(str(blank), np.full((800, 600, 3), 230, np.uint8))
            self.assertIsNone(prepare_photo(blank, TEMPLATE_T23, uvdoc=None))
