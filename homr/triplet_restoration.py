"""
Restores triplets which are printed without their "3".

Many editions print the 3 only on the first triplets of a passage, or not at all. The
transformer reads such a triplet as three plain notes, e.g. three eighths which take a
beat and a half instead of one beat.

The symbols of a measure tell which notes start together, and the MusicXML generator
derives the timing from that: a chord starts when the earliest note which is still
sounding ends. This makes it possible to check a measure without knowing the answer:
in music a voice carries on where its previous note ended. If three triplet eighths
are read as plain eighths while the other hand plays a quarter, then the quarter ends
before the notes it should lead to have started.

So for every measure which does not make sense in this way, three notes in a row with
the same plain note value are taken as a triplet if that leaves fewer places in the
measure where a voice does not carry on.
"""

import copy
import re
from collections import Counter
from collections.abc import Iterator
from fractions import Fraction

from homr.transformer.vocabulary import EncodedSymbol, sort_token_chords

# A note or rest of a measure: its voice, its duration and its index in the list of symbols
Note = tuple[str, Fraction, int]
Measure = list[list[Note]]
# The indices of the notes of three chords in a row, and if the three start off the beat
Three = tuple[frozenset[int], bool]
# Threes are known by the places of their chords in the measure and their note value
ThreesByPlace = dict[tuple[tuple[int, ...], str], Three]

# Six eighths against a half note only meet it if both threes are triplets, so threes
# which follow each other directly are also tried together, up to this many
_MAX_THREES_IN_A_ROW = 4

# One or two measures which make more sense with a triplet are usually misread in some
# other way: music which has triplets has several. A page (one voice of it) which shows
# triplets of a note value in fewer places than this shows none. Threes which were taken
# together count as one place.
_MIN_TRIPLETS_ON_A_PAGE = 3

# A "measure" which is longer than this many whole notes is music without barlines, or
# with barlines which were not recognized. There is no measure to make sense of.
_MAX_MEASURE_LENGTH = 2

_PLAIN = re.compile(r"(note|rest)_(4|8|16|32|64)$")


def _as_triplet(symbol: EncodedSymbol) -> EncodedSymbol:
    match = _PLAIN.match(symbol.rhythm)
    if match is None:
        return symbol
    result = copy.copy(symbol)
    result.rhythm = f"{match[1]}_{int(match[2]) * 3 // 2}"
    result._duration = None
    return result


def _split_into_measures(symbols: list[EncodedSymbol]) -> list[Measure]:
    index_of = {id(symbol): i for i, symbol in enumerate(symbols)}
    measures: list[Measure] = [[]]
    for chord in sort_token_chords(symbols):
        rhythm = chord[0].rhythm
        if "barline" in rhythm or "repeat" in rhythm:
            measures.append([])
        elif rhythm.startswith(("note", "rest")) and not rhythm.endswith("m"):
            measures[-1].append(
                [
                    (symbol.position, symbol.get_duration().fraction, index_of[id(symbol)])
                    for symbol in chord
                    if symbol.rhythm.startswith(("note", "rest"))
                ]
            )
    return measures


def _play(
    measure: Measure, triplets: frozenset[int] = frozenset()
) -> Iterator[tuple[Fraction, list[tuple[str, Fraction]]]]:
    """
    Times the measure the way the MusicXML generator does it: a chord starts when the
    earliest note which is still sounding ends. The notes in triplets are played as
    members of a triplet.

    Yields the start of every chord together with the voice and the end of each of its notes.
    """
    clock = Fraction(0)
    sounding: list[Fraction] = []
    for chord in measure:
        notes = []
        for voice, duration, index in chord:
            played = duration * 2 / 3 if index in triplets else duration
            if played > 0:
                notes.append((voice, clock + played))
        yield clock, notes
        if notes:
            sounding.extend(end for _, end in notes)
            clock = min(end for end in sounding if end > clock)
            sounding = [end for end in sounding if end > clock]


def _count_breaks(measure: Measure, triplets: frozenset[int] = frozenset()) -> tuple[int, Fraction]:
    """
    Counts the places where a voice does not carry on exactly where its previous note
    ended, and the voices which end before the measure does.

    Returns the number of such places and their total size.
    """
    end_of_voice: dict[str, Fraction] = {}
    breaks = 0
    size = Fraction(0)
    for start, notes in _play(measure, triplets):
        ends: dict[str, Fraction] = {}
        for voice, end in notes:
            ends[voice] = max(ends.get(voice, end), end)
        for voice in ends:
            if voice in end_of_voice and end_of_voice[voice] != start:
                breaks += 1
                size += abs(start - end_of_voice[voice])
        end_of_voice.update(ends)
    if end_of_voice:
        end_of_measure = max(end_of_voice.values())
        for end in end_of_voice.values():
            if end != end_of_measure:
                breaks += 1
                size += end_of_measure - end
    return breaks, size


def _get_length(measure: Measure) -> Fraction:
    return max((end for _, notes in _play(measure) for _, end in notes), default=Fraction(0))


def _find_single_threes(measure: Measure, symbols: list[EncodedSymbol]) -> ThreesByPlace:
    """
    Finds three chords in a row in one voice which all have the same plain note value.
    Voices which go through the same three chords with the same value form one three.

    A three is off the beat if it does not start a whole number of threes into its run
    of equal notes. Taking any three notes of a run as a triplet closes the same gap, so
    the caller needs this to tell them apart.
    """
    found: dict[tuple[tuple[int, ...], str], tuple[set[int], bool]] = {}
    for voice in sorted({voice for chord in measure for voice, _, _ in chord}):
        chords = [
            (place, [index for v, _, index in chord if v == voice])
            for place, chord in enumerate(measure)
        ]
        chords = [(place, notes) for place, notes in chords if notes]
        values = []
        for _, notes in chords:
            matches = [_PLAIN.match(symbols[index].rhythm) for index in notes]
            plain = {match[2] for match in matches if match is not None}
            values.append(plain.pop() if all(matches) and len(plain) == 1 else None)
        start_of_run = 0
        for i, value in enumerate(values):
            if i > 0 and (value is None or value != values[i - 1]):
                start_of_run = i
            if value is None or values[i : i + 3] != [value] * 3:
                continue
            in_voice = [index for _, notes in chords[i : i + 3] for index in notes]
            if not any(symbols[index].rhythm.startswith("note") for index in in_voice):
                continue
            key = (tuple(place for place, _ in chords[i : i + 3]), value)
            in_all_voices, off_beat = found.get(key, (set(), False))
            in_all_voices.update(in_voice)
            found[key] = (in_all_voices, off_beat or (i - start_of_run) % 3 != 0)
    return {key: (frozenset(indices), off_beat) for key, (indices, off_beat) in found.items()}


def _find_threes(measure: Measure, symbols: list[EncodedSymbol]) -> list[Three]:
    """
    Finds the threes of a measure, and the threes which follow each other directly as one.
    """
    single = _find_single_threes(measure, symbols)
    threes = dict(single.values())
    in_a_row = single
    for _ in range(_MAX_THREES_IN_A_ROW - 1):
        in_a_row = _add_following_three(in_a_row, single, measure, symbols)
        for indices, off_beat in in_a_row.values():
            threes.setdefault(indices, off_beat)
    return sorted(threes.items(), key=lambda three: sorted(three[0]))


def _add_following_three(
    in_a_row: ThreesByPlace, single: ThreesByPlace, measure: Measure, symbols: list[EncodedSymbol]
) -> ThreesByPlace:
    """
    Extends each run of threes by the three which follows it directly in the same voices.
    """
    longer: ThreesByPlace = {}
    for (places, value), (indices, off_beat) in in_a_row.items():
        voices = {symbols[index].position for index in indices}
        next_place = next(
            (
                place
                for place in range(places[-1] + 1, len(measure))
                if any(voice in voices for voice, _, _ in measure[place])
            ),
            None,
        )
        for (next_places, next_value), (next_indices, _) in single.items():
            if (
                next_value == value
                and next_places[0] == next_place
                and not indices & next_indices
                and {symbols[index].position for index in next_indices} == voices
            ):
                longer[(places + next_places, value)] = (indices | next_indices, off_beat)
    return longer


def _choose_triplets(measure: Measure, symbols: list[EncodedSymbol]) -> list[frozenset[int]]:
    """
    Takes threes as triplets, the most helpful first, as long as each of them
    leaves fewer places in the measure where a voice does not carry on.
    """
    chosen: list[frozenset[int]] = []
    triplets: frozenset[int] = frozenset()
    breaks = _count_breaks(measure)
    if breaks[0] == 0 or _get_length(measure) > _MAX_MEASURE_LENGTH:
        return []
    threes = _find_threes(measure, symbols)
    while True:
        trials = [
            (_count_breaks(measure, triplets | indices), off_beat, sorted(indices), indices)
            for indices, off_beat in threes
            if not indices & triplets
        ]
        if not trials:
            break
        best_breaks, _, _, best = min(trials, key=lambda trial: trial[:3])
        if best_breaks[0] >= breaks[0]:
            break
        chosen.append(best)
        triplets, breaks = triplets | best, best_breaks
    return chosen


def _note_value(symbols: list[EncodedSymbol], indices: frozenset[int]) -> str:
    return symbols[min(indices)].rhythm.split("_")[1]


def _count_by_note_value(
    symbols: list[EncodedSymbol], chosen: list[list[frozenset[int]]]
) -> Counter[str]:
    return Counter(_note_value(symbols, indices) for in_measure in chosen for indices in in_measure)


def restore_unmarked_triplets(symbols: list[EncodedSymbol]) -> list[EncodedSymbol]:
    """
    Takes the symbols of one voice of a page and returns them with the triplets
    which were read as plain notes restored.
    """
    measures = _split_into_measures(symbols)
    chosen = [_choose_triplets(measure, symbols) for measure in measures]
    shown = _count_by_note_value(symbols, chosen)
    triplets = {
        index
        for in_measure in chosen
        for indices in in_measure
        if shown[_note_value(symbols, indices)] >= _MIN_TRIPLETS_ON_A_PAGE
        for index in indices
    }
    return [_as_triplet(symbol) if i in triplets else symbol for i, symbol in enumerate(symbols)]
