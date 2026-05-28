from collections import defaultdict

from homr.transformer.vocabulary import (
    EncodedSymbol,
    has_rhythm_symbol_a_position,
    nonote,
)


class EncodedSymbolWithPos:
    def __init__(self, position: int, symbol: EncodedSymbol, insert_before: bool = False) -> None:
        self.position = position
        self.symbol = symbol
        self.rhythm = symbol.rhythm
        self.insert_before = insert_before

    def sort_order(self) -> int:
        return self.position * 2 - (1 if self.insert_before else 0)

    def __str__(self) -> str:
        return str(self.position) + " " + str(self.symbol)

    def __repr__(self) -> str:
        return str(self)


def merge_upper_and_lower_staff(voices: list[list[EncodedSymbolWithPos]]) -> list[EncodedSymbol]:
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
