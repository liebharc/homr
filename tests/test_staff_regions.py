import unittest

from homr.model import MultiStaff, Staff, StaffPoint
from homr.staff_regions import StaffRegions


def build_staff(y_min: int, y_max: int) -> Staff:
    step = (y_max - y_min) // 4
    p1 = StaffPoint(0, list(range(y_min, y_max + 1, step)), 0)
    p2 = StaffPoint(100, list(range(y_min, y_max + 1, step)), 0)
    return Staff([p1, p2])


class TestStaffRegions(unittest.TestCase):
    def test_staff_above(self) -> None:
        system1 = MultiStaff(
            [
                build_staff(100, 200),
                build_staff(300, 400),
            ],
            [],
        )
        system2 = MultiStaff(
            [
                build_staff(500, 600),
                build_staff(700, 800),
            ],
            [],
        )
        regions = StaffRegions([system1, system2])
        self.assertEqual(0, regions.get_start_of_closest_staff_above(100))
        self.assertEqual(200, regions.get_start_of_closest_staff_above(201))
        self.assertEqual(400, regions.get_start_of_closest_staff_above(500))
        self.assertEqual(600, regions.get_start_of_closest_staff_above(633))
        self.assertEqual(800, regions.get_start_of_closest_staff_above(1000))

    def test_staff_below(self) -> None:
        system1 = MultiStaff(
            [
                build_staff(100, 200),
                build_staff(300, 400),
            ],
            [],
        )
        system2 = MultiStaff(
            [
                build_staff(500, 600),
                build_staff(700, 800),
            ],
            [],
        )
        regions = StaffRegions([system1, system2])
        self.assertEqual(100, regions.get_start_of_closest_staff_below(0))
        self.assertEqual(100, regions.get_start_of_closest_staff_below(99))
        self.assertEqual(700, regions.get_start_of_closest_staff_below(600))
        self.assertEqual(1e12, regions.get_start_of_closest_staff_below(800))
