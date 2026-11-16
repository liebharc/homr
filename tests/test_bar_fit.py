import unittest

from homr.transformer.bar_fit import StaffBars, bar_fit, has_unfit_bar
from homr.transformer.vocabulary import EncodedSymbol


def _bar(*durations: str) -> list[EncodedSymbol]:
    return [EncodedSymbol(f"note_{d}", "C4", position="upper") for d in durations] + [
        EncodedSymbol("barline")
    ]


class TestBarFit(unittest.TestCase):
    def test_all_bars_fit(self) -> None:
        symbols = _bar("4", "4", "2") * 3
        self.assertEqual(bar_fit(symbols), 3)
        self.assertFalse(has_unfit_bar(symbols))

    def test_a_long_bar_does_not_fit(self) -> None:
        symbols = _bar("2", "2") + _bar("2", "2", "4") + _bar("2", "2")
        self.assertEqual(bar_fit(symbols), 2)
        bars = StaffBars(symbols)
        self.assertEqual(bars.first_unfit(0), 1)
        self.assertTrue(has_unfit_bar(symbols))

    def test_a_short_first_bar_fits(self) -> None:
        symbols = _bar("4") + _bar("2", "2") * 3
        self.assertIsNone(StaffBars(symbols).first_unfit(0))

    def test_a_short_bar_later_does_not_fit(self) -> None:
        symbols = _bar("2", "2") + _bar("4") + _bar("2", "2") * 2
        self.assertEqual(StaffBars(symbols).first_unfit(0), 1)

    def test_too_few_bars_are_not_checked(self) -> None:
        symbols = _bar("2", "2") + _bar("4")
        self.assertFalse(has_unfit_bar(symbols))
