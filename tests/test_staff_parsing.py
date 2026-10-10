import unittest

from homr.model import MultiStaff, Staff, StaffPoint
from homr.staff_parsing import _assign_parts, _measure_rests_like
from homr.system_repair import (
    _common_layout,
    _ensure_same_number_of_staffs,
    _unify_grand_staffs,
)
from homr.transformer.vocabulary import EncodedSymbol


def make_staff(number: float, is_grandstaff: bool = False) -> Staff:
    """A staff 40 high, at 100 * number. Staffs at n and n + 0.5 are close, like a grand staff."""
    y_points = [10 * i + 100 * number for i in range(5)]
    staff = Staff([StaffPoint(0.0, y_points, 0)])
    staff.is_grandstaff = is_grandstaff
    return staff


def make_row(number: float, is_grandstaff: bool = False) -> MultiStaff:
    return MultiStaff([make_staff(number, is_grandstaff)], [])


def make_grandstaff_row(number: int) -> MultiStaff:
    """A grand staff of the staffs number and number + 1."""
    return MultiStaff([make_staff(number).merge(make_staff(number + 1))], [])


def _layouts(result: list[MultiStaff]) -> list[list[bool]]:
    return [[s.is_grandstaff for s in row.staffs] for row in result]


class TestStaffParsing(unittest.TestCase):
    def test_ensure_same_number_of_staffs_already_uniform(self) -> None:
        staffs = [make_row(i, is_grandstaff=True) for i in range(4)]

        result = _ensure_same_number_of_staffs(staffs)

        self.assertEqual([[True]] * 4, _layouts(result))

    def test_ensure_same_number_of_staffs_merges_repeating_pattern(self) -> None:
        # A voice and a piano grand staff in every system
        staffs = []
        for i in range(0, 12, 3):
            staffs.append(make_row(i))
            staffs.append(make_row(i + 1, is_grandstaff=True))

        result = _ensure_same_number_of_staffs(staffs)

        self.assertEqual([[False, True]] * 4, _layouts(result))

    def test_ensure_same_number_of_staffs_merges_repeating_pattern_grand_staff_first(self) -> None:
        # The piano grand staff above the solo staff in every system
        staffs = []
        for i in range(0, 12, 3):
            staffs.append(make_row(i, is_grandstaff=True))
            staffs.append(make_row(i + 1))

        result = _ensure_same_number_of_staffs(staffs)

        self.assertEqual([[True, False]] * 4, _layouts(result))

    def test_ensure_same_number_of_staffs_absorbs_inconsistently_pre_merged_row(self) -> None:
        # Only the first system was connected by a bar line
        pre_merged = MultiStaff([make_staff(0), make_staff(1, is_grandstaff=True)], [])
        staffs = [pre_merged]
        for i in range(2, 8, 2):
            staffs.append(make_row(i))
            staffs.append(make_row(i + 1, is_grandstaff=True))

        result = _ensure_same_number_of_staffs(staffs)

        self.assertEqual([[False, True]] * 4, _layouts(result))

    def test_ensure_same_number_of_staffs_keeps_leading_system_with_other_layout(self) -> None:
        staffs = [make_row(0, is_grandstaff=True)]
        staffs += [MultiStaff([make_staff(i), make_staff(i + 1, True)], []) for i in (1, 3, 5)]

        result = _ensure_same_number_of_staffs(staffs)

        self.assertEqual([[True]] + [[False, True]] * 3, _layouts(result))

    def test_ensure_same_number_of_staffs_drops_single_staff_below_last_system(self) -> None:
        staffs = [make_row(i, is_grandstaff=True) for i in range(4)] + [make_row(4)]

        result = _ensure_same_number_of_staffs(staffs)

        self.assertEqual([[True]] * 4, _layouts(result))

    def test_ensure_same_number_of_staffs_splits_pairs_on_page_of_single_staffs(self) -> None:
        staffs = [make_row(0), make_row(1), MultiStaff([make_staff(2), make_staff(3)], [])]
        staffs += [make_row(4), MultiStaff([make_staff(5), make_staff(6)], []), make_row(7)]

        result = _ensure_same_number_of_staffs(staffs)

        # Connected pairs on a page of single staffs are most likely false connections
        self.assertEqual([[False]] * 8, _layouts(result))

    def test_ensure_same_number_of_staffs_keeps_layout_change(self) -> None:
        # A system where the voice pauses, between systems with voice and piano
        staffs = [
            MultiStaff([make_staff(3 * i), make_staff(3 * i + 1, True)], []) for i in range(4)
        ]
        staffs.insert(2, make_row(12, is_grandstaff=True))

        result = _ensure_same_number_of_staffs(staffs)

        self.assertEqual([[False, True]] * 2 + [[True]] + [[False, True]] * 2, _layouts(result))

    def test_ensure_same_number_of_staffs_merges_neighbors_into_common_layout(self) -> None:
        staffs = [
            MultiStaff([make_staff(3 * i), make_staff(3 * i + 1, True)], []) for i in range(3)
        ]
        staffs += [make_row(9), make_row(10, is_grandstaff=True)]

        result = _ensure_same_number_of_staffs(staffs)

        self.assertEqual([[False, True]] * 4, _layouts(result))

    def test_ensure_same_number_of_staffs_merges_single_staffs_with_missed_brace(self) -> None:
        staffs = [make_row(i, is_grandstaff=True) for i in (0, 1, 4, 5)]
        staffs[2:2] = [make_row(2), make_row(2.5)]

        result = _ensure_same_number_of_staffs(staffs)

        self.assertEqual([[True]] * 5, _layouts(result))

    def test_ensure_same_number_of_staffs_keeps_evenly_spaced_single_staffs(self) -> None:
        # Verses for the voice alone after systems of a piano
        staffs = [make_row(i, is_grandstaff=True) for i in range(3)]
        staffs += [make_row(i) for i in range(3, 9)]

        result = _ensure_same_number_of_staffs(staffs)

        self.assertEqual([[True]] * 3 + [[False]] * 6, _layouts(result))

    def test_ensure_same_number_of_staffs_splits_wrongly_connected_systems(self) -> None:
        # A crease connected the staffs of several systems
        connected = [make_staff(i, is_grandstaff=i % 2 == 1) for i in range(7)]
        staffs = [MultiStaff(connected, []), make_row(7, is_grandstaff=True)]

        result = _ensure_same_number_of_staffs(staffs)

        self.assertEqual([[False, True]] * 4, _layouts(result))

    def test_ensure_same_number_of_staffs_merges_only_staffs_which_overlap(self) -> None:
        staffs = [make_row(i, is_grandstaff=True) for i in range(4)]
        right = Staff([StaffPoint(500.0, [10 * i + 1100.0 for i in range(5)], 0)])
        staffs[2:2] = [make_row(10), MultiStaff([right], [])]

        result = _ensure_same_number_of_staffs(staffs)

        self.assertEqual([[True]] * 2 + [[False]] * 2 + [[True]] * 2, _layouts(result))

    def test_ensure_same_number_of_staffs_keeps_systems_without_common_layout(self) -> None:
        staffs = [
            MultiStaff([make_staff(0), make_staff(1, is_grandstaff=True)], []),
            MultiStaff([make_staff(2, is_grandstaff=True), make_staff(3)], []),
        ]

        result = _ensure_same_number_of_staffs(staffs)

        self.assertEqual(staffs, result)

    def test_ensure_same_number_of_staffs_finds_grand_staff_found_as_single_staffs(self) -> None:
        # A grand staff was found as two connected single staffs
        staffs = [make_row(0), make_row(1, is_grandstaff=True), make_row(2)]
        staffs += [MultiStaff([make_staff(3), make_staff(3.5)], [])]
        staffs += [make_row(5), make_row(6, is_grandstaff=True)]

        result = _ensure_same_number_of_staffs(staffs)

        self.assertEqual([[False, True]] * 3, _layouts(result))

    def test_ensure_same_number_of_staffs_merges_system_with_missed_brace(self) -> None:
        staffs = [make_row(i, is_grandstaff=True) for i in range(4)]
        staffs.append(MultiStaff([make_staff(4), make_staff(5)], []))

        result = _ensure_same_number_of_staffs(staffs)

        self.assertEqual([[True]] * 5, _layouts(result))

    def test_ensure_same_number_of_staffs_splits_wrongly_found_grand_staff(self) -> None:
        staffs = [MultiStaff([make_staff(2 * i), make_staff(2 * i + 1)], []) for i in range(3)]
        staffs.append(MultiStaff([make_staff(6).merge(make_staff(7))], []))

        result = _ensure_same_number_of_staffs(staffs)

        self.assertEqual([[False, False]] * 4, _layouts(result))

    def test_common_layout_prefers_mix_of_single_and_grand_staffs(self) -> None:
        grand = [False, True, False, True, False, False, False, True]
        staffs = [make_row(i, is_grandstaff) for i, is_grandstaff in enumerate(grand)]

        self.assertEqual((False, True), _common_layout(staffs))

    def test_ensure_same_number_of_staffs_merges_single_staffs_as_close_as_grand_staffs(
        self,
    ) -> None:
        # Two grand staffs whose braces were missed
        staffs = [make_row(0), make_row(0.5), make_row(2), make_row(2.5)]
        staffs += [make_grandstaff_row(4), make_grandstaff_row(6)]

        result = _ensure_same_number_of_staffs(staffs)

        self.assertEqual([[True]] * 4, _layouts(result))

    def test_ensure_same_number_of_staffs_joins_voice_and_piano_of_single_system(self) -> None:
        staffs = [make_row(0), make_row(1, is_grandstaff=True)]

        result = _ensure_same_number_of_staffs(staffs)

        self.assertEqual([[False, True]], _layouts(result))

    def test_ensure_same_number_of_staffs_keeps_two_grand_staffs_per_system(self) -> None:
        # Piano four hands, the last system has one piano only
        staffs = [MultiStaff([make_staff(i, True), make_staff(i + 1, True)], []) for i in (0, 2)]
        staffs.append(make_row(4, is_grandstaff=True))

        result = _ensure_same_number_of_staffs(staffs)

        self.assertEqual([[True, True]] * 2 + [[True]], _layouts(result))

    def test_ensure_same_number_of_staffs_merges_only_staffs_as_close_as_grand_staffs(
        self,
    ) -> None:
        staffs = [make_grandstaff_row(0), make_grandstaff_row(2)]
        staffs += [make_row(4), make_row(6)]  # Twice the gap of a grand staff
        staffs += [make_grandstaff_row(8), make_grandstaff_row(10)]

        result = _ensure_same_number_of_staffs(staffs)

        self.assertEqual([[True]] * 2 + [[False]] * 2 + [[True]] * 2, _layouts(result))

    def test_unify_grand_staffs_keeps_page_if_staffs_dont_overlap(self) -> None:
        right = Staff([StaffPoint(500.0, [10 * i + 600.0 for i in range(5)], 0)])
        staffs = [MultiStaff([make_staff(0), make_staff(1, True)], []) for _ in range(2)]
        staffs.append(MultiStaff([make_staff(4), make_staff(5), right], []))

        self.assertEqual(staffs, _unify_grand_staffs(staffs))

    def test_ensure_same_number_of_staffs_keeps_systems_of_several_grand_staffs(self) -> None:
        staffs = [
            MultiStaff([make_staff(0, True), make_staff(1, True)], []),
            MultiStaff([make_staff(i, True) for i in (2, 3, 4)], []),
        ]

        result = _ensure_same_number_of_staffs(staffs)

        self.assertEqual(_layouts(staffs), _layouts(result))

    def test_unify_grand_staffs_keeps_page_without_most_common_layout(self) -> None:
        staffs = [
            MultiStaff([make_staff(0).merge(make_staff(1))], []),
            MultiStaff([make_staff(2), make_staff(3)], []),
        ]
        self.assertEqual(staffs, _unify_grand_staffs(staffs))

    def test_unify_grand_staffs_keeps_page_if_grand_staff_cant_be_split(self) -> None:
        # Detected as grand staff, but without the lines of two staffs
        staffs = [MultiStaff([make_staff(2 * i), make_staff(2 * i + 1)], []) for i in range(3)]
        staffs.append(MultiStaff([make_staff(6, is_grandstaff=True)], []))
        self.assertEqual(staffs, _unify_grand_staffs(staffs))

    def test_assign_parts_with_hidden_voice(self) -> None:
        # Voice and piano, in the second system the voice is hidden
        voice_and_piano = MultiStaff([make_staff(0), make_staff(1, is_grandstaff=True)], [])
        piano = make_row(3, is_grandstaff=True)

        self.assertEqual([(0, 1), (1,)], _assign_parts([voice_and_piano, piano]))

    def test_assign_parts_ambiguous(self) -> None:
        # Two voices, a system with one of them doesn't tell which one
        two_voices = MultiStaff([make_staff(0), make_staff(1)], [])

        self.assertIsNone(_assign_parts([two_voices, make_row(3)]))

    def test_measure_rests_like(self) -> None:
        symbols = [
            EncodedSymbol("clef_G2"),
            EncodedSymbol("note_8", "C4"),
            EncodedSymbol("note_8", "D4"),
            EncodedSymbol("barline"),
            EncodedSymbol("note_4", "E4"),
            EncodedSymbol("repeatEnd"),
        ]

        result = _measure_rests_like(symbols, is_grandstaff=True)

        self.assertEqual(
            ["measureRest", "chord", "measureRest", "barline"]
            + ["measureRest", "chord", "measureRest", "repeatEnd"],
            [s.rhythm for s in result],
        )
        self.assertEqual("lower", result[2].position)
