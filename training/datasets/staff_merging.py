from collections import defaultdict

from homr.transformer.vocabulary import (
    EncodedSymbol,
    has_rhythm_symbol_a_position,
    nonote,
)


class EncodedSymbolWithPos:
    """
    Encoded symbol annotated with its time position in a source voice.
    """

    def __init__(self, position: int, symbol: EncodedSymbol, insert_before: bool = False) -> None:
        """
        Store a symbol and its merge ordering metadata.

        Args:
            position: Integer time position from the source voice.
            symbol: Encoded symbol at that position.
            insert_before: Whether the symbol should sort before symbols at the
                same position.
        """
        self.position = position
        self.symbol = symbol
        self.rhythm = symbol.rhythm
        self.insert_before = insert_before

    def sort_order(self) -> int:
        """
        Return the ordering key used while merging voices.
        """
        return self.position * 2 - (1 if self.insert_before else 0)

    def __str__(self) -> str:
        """
        Format the positioned symbol for debugging.
        """
        return str(self.position) + " " + str(self.symbol)

    def __repr__(self) -> str:
        """
        Return the debug representation of the positioned symbol.
        """
        return str(self)


def merge_upper_and_lower_staff(voices: list[list[EncodedSymbolWithPos]]) -> list[EncodedSymbol]:
    """
    Merge upper and lower staff voices into one flat token sequence.

    Args:
        voices: Positioned symbol lists, with the first voice treated as upper staff
            and later voices as lower staff.

    Returns:
        Flat encoded sequence with chord markers inserted where needed.
    """
    voices = [voice for voice in voices if len(voice) > 0]
    positions: defaultdict[int, list[EncodedSymbol]] = defaultdict(list)
    for voice_no, voice in enumerate(voices):
        position = "upper" if voice_no == 0 else "lower"
        for symbol in voice:
            if (
                has_rhythm_symbol_a_position(symbol.symbol.rhythm)
                and symbol.symbol.position == nonote
            ):
                symbol.symbol.position = position
            positions[symbol.sort_order()].append(symbol.symbol)

    result: list[EncodedSymbol] = []
    for key in sorted(positions):
        result.extend(create_chord_over_two_staffs(positions[key]))

    if (
        len(result) > 0
        and "barline" not in result[-1].rhythm
        and not result[-1].rhythm.startswith("repeat")
    ):
        result.append(EncodedSymbol("barline"))
    if len(result) > 0 and result[-1].rhythm == "repeatEndStart":
        result.pop()
        result.append(EncodedSymbol("repeatEnd"))
    return result


def create_chord_over_two_staffs(symbols: list[EncodedSymbol]) -> list[EncodedSymbol]:
    """
    Serialize simultaneous upper/lower staff symbols into a chord-aware sequence.

    Args:
        symbols: Symbols sharing the same merge position.

    Returns:
        Flat token fragment containing barlines, clefs, signatures and notes/rests
        with explicit ``chord`` separators where needed.
    """
    barlines = []
    key = []
    time = []
    clef = []
    notes_or_rests = []
    for symbol in symbols:
        rhythm = symbol.rhythm
        if "barline" in rhythm or "repeat" in rhythm:
            if symbol not in barlines:
                barlines.append(symbol)
        elif rhythm.startswith("keySignature"):
            if symbol not in key:
                key.append(symbol)
        elif rhythm.startswith("timeSignature"):
            if symbol not in time:
                time.append(symbol)
        elif rhythm.startswith("clef"):
            clef.append(symbol)
        else:
            notes_or_rests.append(symbol)
    result = []
    result.extend(barlines)
    for i, symbol in enumerate(clef):
        is_first = i == 0
        if not is_first:
            result.append(EncodedSymbol("chord"))
        result.append(symbol)
    result.extend(key)
    result.extend(time)

    for i, symbol in enumerate(notes_or_rests):
        is_first = i == 0
        if not is_first:
            result.append(EncodedSymbol("chord"))
        result.append(symbol)
    return result
