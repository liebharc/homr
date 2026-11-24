"""
The tokens of a part as groups of symbols which start together, each with its
start in the measure, and the division and measure lengths found from them.
"""

import math
from collections import defaultdict
from fractions import Fraction

import numpy as np

from homr.transformer.vocabulary import EncodedSymbol, sort_token_chords

TimeSignature = tuple[str, str]


def is_timed(symbol: EncodedSymbol) -> bool:
    return symbol.rhythm.startswith(("note", "rest"))


def is_grace(symbol: EncodedSymbol) -> bool:
    return "G" in symbol.rhythm


def length(symbol: EncodedSymbol) -> Fraction:
    """In whole notes, 0 for grace notes."""
    return symbol.get_duration().fraction


class SymbolChord:
    def __init__(self, symbols: list[EncodedSymbol], start: Fraction = Fraction(0)) -> None:
        self.symbols = symbols
        self.start = start  # in whole notes from the start of the measure

    def __repr__(self) -> str:
        return str.join("&", [str(s) for s in self.symbols])

    @property
    def rhythm(self) -> str:
        return self.symbols[0].rhythm if self.symbols else ""

    def is_barline(self) -> bool:
        return "barline" in self.rhythm or "repeat" in self.rhythm

    def is_notes(self) -> bool:
        return bool(self.symbols) and is_timed(self.symbols[0]) and not self.is_multi_rest()

    def is_multi_rest(self) -> bool:
        return (
            len(self.symbols) == 1 and self.rhythm.startswith("rest") and self.rhythm.endswith("m")
        )

    def timed(self) -> list[EncodedSymbol]:
        """Notes and rests which take time, grace notes don't."""
        return [s for s in self.symbols if is_timed(s) and length(s) > 0]

    def into_positions(self) -> list["SymbolChord"]:
        buckets: dict[str, list[EncodedSymbol]] = defaultdict(list)
        for symbol in self.symbols:
            buckets[symbol.position].append(symbol)
        chords = [SymbolChord(s, self.start) for s in buckets.values()]
        # Voices of only rests go first
        chords.sort(key=lambda chord: not all(s.rhythm.startswith("rest") for s in chord.symbols))
        return chords


def group_into_chords(voice: list[EncodedSymbol]) -> list[SymbolChord]:
    """Groups with their start times."""
    groups = split_mixed_chords([SymbolChord(s) for s in sort_token_chords(voice)])
    _set_start_times(groups)
    return groups


def _set_start_times(groups: list[SymbolChord]) -> None:
    """
    Tokens say which notes start together, not when. A group starts when the earliest
    still-sounding note ends (that is how training data is grouped), but not before the
    earlier notes of its own positions (voices) have ended.
    """
    time = Fraction(0)
    sounding: list[Fraction] = []
    voice_free: dict[str, Fraction] = {}
    for group in groups:
        if group.is_barline():
            group.start = time
            time, sounding, voice_free = Fraction(0), [], {}
            continue
        timed = group.timed() if group.is_notes() else []
        free = max((voice_free.get(s.position, Fraction(0)) for s in timed), default=Fraction(0))
        time = max(time, free)
        sounding = [end for end in sounding if end > time]
        group.start = time
        if not timed:
            continue
        # A position can hold notes of different lengths, the shortest one frees it
        for symbol in sorted(timed, key=length, reverse=True):
            voice_free[symbol.position] = time + length(symbol)
        sounding.extend(time + length(s) for s in timed)
        time = min(sounding)
        sounding = [end for end in sounding if end > time]


_BEFORE_NOTES_ORDER = ["clef", "keySignature", "timeSignature"]


def _split_mixed_chord(symbols: list[EncodedSymbol]) -> list[list[EncodedSymbol]]:
    """
    The transformer sometimes joins a clef, key, time signature or barline to a chord
    of notes, e.g. a clef change in one staff on the beat where the other staff plays.
    Split it: clefs, keys and time signatures go before the notes (in the order they
    have at the start of a line), barlines and repeats after them.
    """
    notes = [s for s in symbols if is_timed(s)]
    if not notes or len(notes) == len(symbols):
        return [symbols]
    others = [s for s in symbols if not is_timed(s)]
    after = [s for s in others if "barline" in s.rhythm or "repeat" in s.rhythm]
    before: dict[str, list[EncodedSymbol]] = defaultdict(list)
    for s in others:
        if s not in after:
            before[s.rhythm.split("_")[0].split("/")[0]].append(s)
    rank = {kind: i for i, kind in enumerate(_BEFORE_NOTES_ORDER)}
    ordered = sorted(before, key=lambda kind: rank.get(kind, len(rank)))
    return [*(before[kind] for kind in ordered), notes, *([s] for s in after)]


def split_mixed_chords(groups: list[SymbolChord]) -> list[SymbolChord]:
    result: list[SymbolChord] = []
    split_barline = False
    for group in groups:
        parts = _split_mixed_chord(group.symbols)
        if split_barline and group.is_barline() and len(parts) == 1:
            # The barline split off the previous chord and this one are the same barline:
            # keep one, preferring a repeat or double barline over a plain barline.
            if group.rhythm == "barline" and result[-1].rhythm != "barline":
                split_barline = False
                continue
            result.pop()
        result.extend(SymbolChord(part) for part in parts)
        split_barline = len(parts) > 1 and result[-1].is_barline()
    return result


def division_and_measure_lengths(groups: list[SymbolChord]) -> tuple[int, list[Fraction]]:
    """
    The division and one measure length per time signature, estimated from the measures
    up to the next time signature. Measures before the first one count towards it.
    """
    lengths = [Fraction(1, 4)] + [length(s) for g in groups if g.is_notes() for s in g.timed()]
    division = math.lcm(*(d.denominator for d in lengths))
    segments: list[list[Fraction]] = [[]]
    has_time_signature = False
    end = Fraction(0)
    for group in groups:
        if group.rhythm.startswith("timeSignature"):
            if end > 0:
                segments[-1].append(end)
                end = Fraction(0)
            if has_time_signature:
                segments.append([])
            has_time_signature = True
        elif group.is_barline():
            if end > 0:
                segments[-1].append(end)
            end = Fraction(0)
        elif group.is_notes():
            end = max([end, *(group.start + length(s) for s in group.timed())])
    if end > 0:
        segments[-1].append(end)
    all_measures = [d for segment in segments for d in segment]
    if not all_measures:
        return division, [Fraction(1)] * len(segments)
    # A time signature without measures after it, e.g. a courtesy one at the end of a
    # system, belongs to the measures of the next one
    result: list[Fraction] = []
    following = Fraction(np.median(all_measures))  # type: ignore[arg-type]
    for segment in reversed(segments):
        if segment:
            following = Fraction(np.median(segment))  # type: ignore[arg-type]
        result.append(following)
    return division, result[::-1]


def starts_with_time_signature(groups: list[SymbolChord]) -> bool:
    """Whether a time signature comes before the first note, rest or barline."""
    for group in groups:
        if group.rhythm.startswith("timeSignature"):
            return True
        if group.rhythm.startswith(("note", "rest", "measureRest")) or group.is_barline():
            return False
    return False


def time_signature_length(time: TimeSignature | None) -> Fraction | None:
    if time is None:
        return None
    try:
        result = Fraction(int(time[0]), int(time[1]))
    except (ValueError, ZeroDivisionError):
        return None
    return result if result > 0 else None
