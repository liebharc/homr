import unittest

import numpy as np

from homr.autocrop import autocrop_with_offset


def make_page() -> np.ndarray:
    page = np.full((600, 400, 3), 245, dtype=np.uint8)
    page[100:500:20, 50:350] = 0  # Staff like lines
    return page


class TestAutocrop(unittest.TestCase):
    def test_crops_page_on_desk(self) -> None:
        # The paper has to cover most of the photo
        desk = np.zeros((700, 500, 3), dtype=np.uint8)
        desk[:] = (50, 90, 140)  # Brown, BGR
        desk[50:650, 50:450] = make_page()

        cropped, (x, y) = autocrop_with_offset(desk)

        self.assertLess(cropped.shape[0] * cropped.shape[1], 0.8 * 700 * 500)
        self.assertLess(abs(x - 50) + abs(y - 50), 30)

    def test_keeps_paper_in_shadow(self) -> None:
        page = make_page()
        page[:200] = (page[:200] * 0.5).astype(np.uint8)

        cropped, offset = autocrop_with_offset(page)

        self.assertEqual(page.shape, cropped.shape)
        self.assertEqual((0, 0), offset)

    def test_crops_desk_but_keeps_paper_in_shadow(self) -> None:
        photo = np.zeros((700, 500, 3), dtype=np.uint8)
        photo[:] = (50, 90, 140)  # Brown desk, BGR
        page = make_page()
        page[450:] = (page[450:] * 0.5).astype(np.uint8)  # Shadow at the bottom
        photo[50:650, 0:400] = page[:, :400]
        photo[650:, 0:400] = 120  # The page continues in the shadow

        cropped, (x, y) = autocrop_with_offset(photo)

        self.assertEqual(700, y + cropped.shape[0])  # Bottom kept
        self.assertLess(cropped.shape[1], 450)  # Desk on the right removed

    def test_keeps_ink_in_shadow_beside_desk(self) -> None:
        rng = np.random.default_rng(0)
        photo = np.zeros((700, 500, 3), dtype=np.float32)
        photo[:] = (50, 90, 140)  # Brown desk, BGR
        paper = np.zeros((700, 420, 3), dtype=np.float32)
        paper[:] = (200, 225, 240)
        paper[60:690:20, 30:390] = 30  # Staff like lines
        paper[480:] *= 0.5  # Shadow at the bottom
        photo[:, :420] = paper
        photo = np.clip(photo + rng.normal(0, 6, photo.shape), 0, 255).astype(np.uint8)

        cropped, (x, y) = autocrop_with_offset(photo)

        self.assertEqual(700, y + cropped.shape[0])  # Bottom kept
        line_in_shadow = cropped[600 - y, 40 - x : 380 - x].mean(axis=1)
        self.assertTrue((line_in_shadow < 80).all())
