import unittest

import numpy as np

from training.transformer.image_utils import _find_ink


def _page(paper: int, ink: int) -> np.ndarray:
    image = np.full((100, 200), paper, np.uint8)
    image[50:52, 20:180] = ink
    return image


class TestFindInk(unittest.TestCase):
    def test_white_paper_uses_the_fixed_threshold(self) -> None:
        image = _page(255, 127)
        image[20:22, 20:180] = 128
        mask = _find_ink(image)
        self.assertTrue(mask[50, 100])
        self.assertFalse(mask[20, 100])

    def test_dark_paper_is_not_ink(self) -> None:
        mask = _find_ink(_page(100, 20))
        self.assertTrue(mask[50, 100])
        self.assertEqual(mask.sum(), 2 * 160)

    def test_shadow_on_the_page_is_not_ink(self) -> None:
        image = _page(220, 20)
        image[:, 100:] = 90
        image[50:52, 100:180] = 10
        mask = _find_ink(image)
        self.assertTrue(mask[50, 150])
        self.assertFalse(mask[10, 150])
