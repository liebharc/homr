import unittest

import numpy as np

from validation.degradations import VARIANTS
from validation.staff_structure import (
    classify,
    expected_staffs_per_system,
    resolve_variants,
    staffs_per_system,
)


class TestStaffStructure(unittest.TestCase):
    def test_expected_staffs_per_system_sums_staffs_of_parts(self) -> None:
        musicxml = (
            '<part id="P1"><measure><attributes><divisions>1</divisions></attributes>'
            "</measure></part>"
            '<part id="P2"><measure><attributes><staves>2</staves></attributes>'
            "</measure></part>"
        )
        self.assertEqual(3, expected_staffs_per_system(musicxml))

    def test_staffs_per_system_joins_staffs_by_the_start_bar_line(self) -> None:
        def staff(top: int) -> str:
            return "".join(
                f'<polyline class="StaffLines" points="100,{top + 10 * i} 900,{top + 10 * i}"/>'
                for i in range(5)
            )

        def bar_line(top: int, bottom: int) -> str:
            return f'<polyline class="BarLine" points="100,{top} 100,{bottom}"/>'

        # A system of two staffs, then a system with a single staff
        svg = staff(0) + staff(100) + staff(300) + bar_line(0, 100) + bar_line(100, 140)
        self.assertEqual([2, 1], staffs_per_system(svg))
        self.assertEqual([], staffs_per_system('<polyline class="StaffLines" points="0,0 1,0"/>'))

    def test_classify(self) -> None:
        def result(systems: list[int], detected: int | None = None) -> dict[str, object]:
            return {
                "systems": systems,
                "detected_staffs": sum(systems) if detected is None else detected,
            }

        self.assertEqual("correct", classify(result([2, 3, 3]), [2, 3, 3]))
        self.assertEqual("staffs dropped", classify(result([2, 2], detected=5), [2, 2]))
        self.assertEqual("missed staffs", classify(result([3, 3]), [3, 3, 3]))
        self.assertEqual("extra staffs", classify(result([3, 4, 3]), [3, 3, 3]))
        self.assertEqual("systems split", classify(result([1, 1, 1, 1]), [2, 2]))
        self.assertEqual("systems merged", classify(result([4]), [2, 2]))
        self.assertEqual("some systems wrong", classify(result([3, 3, 2]), [2, 3, 3]))
        self.assertEqual("no staffs", classify(result([]), [2]))
        self.assertEqual("error", classify({"error": "No staffs found"}, [2]))

    def test_resolve_variants(self) -> None:
        self.assertEqual(["clean", "glare", "rotation"], resolve_variants(["glare", "rotation"]))
        self.assertIn("low_light", resolve_variants(["light"]))
        self.assertEqual(list(VARIANTS), resolve_variants(None))
        with self.assertRaises(ValueError):
            resolve_variants(["unknown"])

    def test_variants_are_deterministic(self) -> None:
        # A white page with staff like lines, small to keep the test fast
        image = np.full((400, 300, 3), 255, dtype=np.uint8)
        image[100:300:10, 20:280] = 0
        for name, degrade in VARIANTS.items():
            first = degrade(image, np.random.default_rng(1))
            second = degrade(image, np.random.default_rng(1))
            self.assertTrue(np.array_equal(first, second), name)
            self.assertEqual(3, first.ndim, name)
            self.assertEqual(np.uint8, first.dtype, name)
