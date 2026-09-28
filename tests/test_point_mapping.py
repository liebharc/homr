import unittest

import numpy as np

from homr.point_mapping import chain, undo_crop, undo_resize
from homr.staff_dewarping import PiecewiseAffineTransform


class TestPointMapping(unittest.TestCase):
    def test_undo_crop_and_resize(self) -> None:
        # Crop at (100, 50), then resize from 400x200 to 800x100 (width x height)
        to_input = chain(undo_resize((200, 400), (100, 800)), undo_crop(100, 50))

        self.assertEqual(to_input((80.0, 30.0)), (140.0, 110.0))

    def test_inverse_transform_point_undoes_transform_point(self) -> None:
        src = np.array(
            [[0, 0], [100, 0], [200, 0], [0, 100], [100, 100], [200, 100], [0, 200], [200, 200]]
        )
        dst = src.copy()
        dst[4] = [110, 120]  # bend the middle point
        tform = PiecewiseAffineTransform()
        tform.estimate(src, dst)

        for point in [(50.0, 50.0), (130.0, 90.0), (20.0, 170.0), (150.0, 150.0)]:
            forward = tform.transform_point(point)
            back = tform.inverse_transform_point(forward)
            self.assertAlmostEqual(back[0], point[0], places=3)
            self.assertAlmostEqual(back[1], point[1], places=3)

    def test_inverse_transform_point_outside_of_mesh_is_unchanged(self) -> None:
        src = np.array([[0, 0], [100, 0], [0, 100], [100, 100]])
        tform = PiecewiseAffineTransform()
        tform.estimate(src, src + 5)

        self.assertEqual(tform.inverse_transform_point((500.0, 500.0)), (500.0, 500.0))
