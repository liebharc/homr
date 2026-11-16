"""
Bar lengths of a decoded staff, used to choose between readings and to find bars to
decode again. The typical bar is the median duration of the staff's closed bars (upper
staff). A bar fits if it has the typical length and, in a grand staff, the lower staff
lasts as long as the upper one. A shorter bar is fine at the start of a staff (pickup,
end of a volta).
"""

from fractions import Fraction

from homr.transformer.vocabulary import (
    EncodedSymbol,
    _get_duration_of_measure,
    sort_token_chords,
)


def is_bar_end(rhythm: str) -> bool:
    return "barline" in rhythm or "repeat" in rhythm


def bar_durations(symbols: list[EncodedSymbol]) -> tuple[Fraction, Fraction]:
    """(upper, lower) duration of one bar, given without its closing barline."""
    chords = sort_token_chords(symbols)

    def duration(position: str) -> Fraction:
        return _get_duration_of_measure([[s for s in c if s.position == position] for c in chords])

    return duration("upper"), duration("lower")


def split_bars(symbols: list[EncodedSymbol]) -> list[tuple[int, int]]:
    """(start, end) indices of the closed bars, end is the index of the closing barline."""
    bars, start = [], 0
    for i, symbol in enumerate(symbols):
        if is_bar_end(symbol.rhythm):
            if i > start:
                bars.append((start, i))
            start = i + 1
    return bars


def header(symbols: list[EncodedSymbol]) -> list[str]:
    """Clefs, keys and time signatures of a bar (with their staff), a repair keeps them."""
    return sorted(
        s.rhythm + s.position
        for s in symbols
        if s.rhythm.startswith(("clef", "keySignature", "timeSignature"))
    )


def fits(upper: Fraction, lower: Fraction, typical: Fraction, has_lower: bool, first: bool) -> bool:
    if has_lower and upper != lower:
        return False
    if upper > typical:
        return False
    return not (upper < typical and not first)


class StaffBars:
    """The closed bars of a decoded staff and their durations."""

    def __init__(self, symbols: list[EncodedSymbol]) -> None:
        self.bars = split_bars(symbols)
        self.durations = [bar_durations(symbols[s:e]) for s, e in self.bars]
        uppers = sorted(u for u, _ in self.durations)
        self.typical = uppers[len(uppers) // 2] if uppers else Fraction(0)
        self.has_lower = any(lower > 0 for _, lower in self.durations)

    def checkable(self) -> bool:
        return len(self.bars) >= 3 and self.typical != 0  # noqa: PLR2004

    def fits(self, index: int) -> bool:
        return fits(*self.durations[index], self.typical, self.has_lower, index == 0)

    def first_unfit(self, start: int) -> int | None:
        return next((i for i in range(start, len(self.bars)) if not self.fits(i)), None)


def bar_fit(symbols: list[EncodedSymbol]) -> int:
    """
    Number of closed bars with the typical length whose upper and lower staff agree, used
    to choose between readings of a staff.
    """
    bars: list[list[list[EncodedSymbol]]] = [[]]
    for chord in sort_token_chords(symbols):
        if chord and is_bar_end(chord[0].rhythm):
            bars.append([])
        else:
            bars[-1].append(chord)
    closed = [b for b in bars[:-1] if b]
    if not closed:
        return 0

    def duration(bar: list[list[EncodedSymbol]], position: str) -> Fraction:
        return _get_duration_of_measure([[s for s in c if s.position == position] for c in bar])

    uppers = [duration(b, "upper") for b in closed]
    lowers = [duration(b, "lower") for b in closed]
    has_lower = any(d > 0 for d in lowers)
    typical = sorted(uppers)[len(uppers) // 2]
    return sum(
        1
        for upper, lower in zip(uppers, lowers, strict=True)
        if upper == typical and (not has_lower or lower == upper)
    )


def has_unfit_bar(symbols: list[EncodedSymbol]) -> bool:
    bars = StaffBars(symbols)
    return bars.checkable() and bars.first_unfit(0) is not None
