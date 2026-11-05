import unittest

import numpy as np

from training.omr_datasets.musescore_svg import (
    SvgRectangle,
    SvgStaff,
    _extend_staffs_with_dynamics,
)
from training.transformer.image_utils import random_staff_crop


class TestStaffCrop(unittest.TestCase):
    def test_staffs_are_extended_to_the_dynamics_next_to_them(self) -> None:
        staffs = [SvgStaff(0, 100, 500, 40), SvgStaff(0, 300, 500, 40)]
        below_first = SvgRectangle(50, 170, 20, 15)
        far_away = SvgRectangle(50, 900, 20, 15)
        _extend_staffs_with_dynamics(staffs, [below_first, far_away])
        self.assertEqual((staffs[0].y, staffs[0].y + staffs[0].height), (100, 185))
        self.assertEqual((staffs[1].y, staffs[1].y + staffs[1].height), (300, 340))

    def test_crop_from_the_lines_keeps_the_dynamics(self) -> None:
        image = np.zeros((400, 100, 3), dtype=np.uint8)
        crop_info = {"stems": [100.0, 330.0], "lines": [150.0, 190.0], "staff_space": 10.0}
        for _ in range(50):
            self.assertGreaterEqual(len(random_staff_crop(image, crop_info)), 230)
