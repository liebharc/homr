import unittest

from homr.model import MultiStaff, Staff, StaffPoint
from homr.staff_parsing import _ensure_same_number_of_staffs, _get_number_of_voices


def make_staff(number: int, is_grandstaff: bool = False) -> Staff:
    y_points = [10 * i + 100 * float(number) for i in range(5)]
    staff = Staff([StaffPoint(0.0, y_points, 0)])
    staff.is_grandstaff = is_grandstaff
    return staff


def make_row(number: int, is_grandstaff: bool = False) -> MultiStaff:
    return MultiStaff([make_staff(number, is_grandstaff)], [])


def _layouts(result: list[MultiStaff]) -> list[list[bool]]:
    return [[s.is_grandstaff for s in row.staffs] for row in result]


class TestStaffParsing(unittest.TestCase):
    def test_ensure_same_number_of_staffs_already_uniform(self) -> None:
        staffs = [make_row(i, is_grandstaff=True) for i in range(4)]

        result = _ensure_same_number_of_staffs(staffs)

        self.assertEqual([[True]] * 4, _layouts(result))
        self.assertEqual(1, _get_number_of_voices(result))

    def test_ensure_same_number_of_staffs_merges_repeating_pattern(self) -> None:
        # A solo staff followed by a piano grand staff, repeated for every system,
        # as in a vocal score with piano accompaniment.
        staffs = []
        for i in range(0, 12, 3):
            staffs.append(make_row(i))
            staffs.append(make_row(i + 1, is_grandstaff=True))

        result = _ensure_same_number_of_staffs(staffs)

        self.assertEqual([[False, True]] * 4, _layouts(result))
        self.assertEqual(2, _get_number_of_voices(result))

    def test_ensure_same_number_of_staffs_merges_repeating_pattern_grand_staff_first(self) -> None:
        # Same pattern, but the piano grand staff comes first in every system
        # instead of the solo staff, e.g. an instrumental introduction before
        # the voice enters for the rest of the piece.
        staffs = []
        for i in range(0, 12, 3):
            staffs.append(make_row(i, is_grandstaff=True))
            staffs.append(make_row(i + 1))

        result = _ensure_same_number_of_staffs(staffs)

        self.assertEqual([[True, False]] * 4, _layouts(result))

    def test_ensure_same_number_of_staffs_absorbs_inconsistently_pre_merged_row(self) -> None:
        # A solo-staff-plus-grand-staff pair that an earlier detection step
        # (staffs sharing a bar line) happened to merge into one row, while
        # the same kind of pair elsewhere on the page stayed as two separate
        # rows -- purely because of how cleanly a bar line lined up, not
        # because the underlying layout differs. The pre-merged row must be
        # recognized as fitting the repeating pattern, not trimmed away as an
        # outlier just because its row shape looks different from the rest.
        pre_merged = MultiStaff([make_staff(0), make_staff(1, is_grandstaff=True)], [])
        staffs = [pre_merged]
        for i in range(2, 8, 2):
            staffs.append(make_row(i))
            staffs.append(make_row(i + 1, is_grandstaff=True))

        result = _ensure_same_number_of_staffs(staffs)

        self.assertEqual([[False, True]] * 4, _layouts(result))

    def test_ensure_same_number_of_staffs_drops_leading_outlier(self) -> None:
        # A single system at the very top that doesn't match the otherwise
        # fully uniform rest of the page (e.g. the worst-detected staff).
        staffs = [make_row(0)] + [make_row(i, is_grandstaff=True) for i in range(1, 5)]

        result = _ensure_same_number_of_staffs(staffs)

        self.assertEqual([[True]] * 4, _layouts(result))

    def test_ensure_same_number_of_staffs_drops_trailing_outlier(self) -> None:
        staffs = [make_row(i, is_grandstaff=True) for i in range(4)] + [make_row(4)]

        result = _ensure_same_number_of_staffs(staffs)

        self.assertEqual([[True]] * 4, _layouts(result))

    def test_ensure_same_number_of_staffs_falls_back_to_break_apart(self) -> None:
        # No consistent per-system layout and no clean repeating pattern at
        # any period/trim: the only safe option left is to treat every raw
        # staff as its own system, including exploding rows that an earlier
        # step had bundled together.
        staffs = [
            MultiStaff([make_staff(0), make_staff(1, is_grandstaff=True)], []),
            MultiStaff([make_staff(2, is_grandstaff=True)], []),
            MultiStaff([make_staff(3), make_staff(4)], []),
            MultiStaff([make_staff(5)], []),
            MultiStaff([make_staff(6, is_grandstaff=True)], []),
        ]

        result = _ensure_same_number_of_staffs(staffs)

        self.assertEqual(
            [[False], [True], [True], [False], [False], [False], [True]], _layouts(result)
        )
        for row in result:
            self.assertEqual(1, len(row.staffs))
