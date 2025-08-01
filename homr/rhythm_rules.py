from fractions import Fraction

from homr import constants
from homr.results import (
    DurationModifier,
    ResultChord,
    ResultMeasure,
    ResultNote,
    ResultStaff,
    ResultSymbol,
    ResultTimeSignature,
)
from homr.simple_logging import eprint


def correct_rhythm(staffs: list[ResultStaff]) -> list[ResultStaff]:
    return [correct_staff(staff) for staff in staffs]


def correct_staff(staff: ResultStaff) -> ResultStaff:
    time_sig: ResultTimeSignature | None = None
    measures: list[ResultMeasure] = []
    for i, measure in enumerate(staff.measures):
        likely_to_have_pickups = i == 0 or i == len(staff.measures) - 1  # noqa: PLR1714
        (time_sig, corrected) = correct_measure(measure, time_sig, i + 1)
        if likely_to_have_pickups:
            measures.append(measure)
        else:
            measures.append(corrected)
    return ResultStaff(measures)


def correct_measure(  # noqa: C901
    measure: ResultMeasure, time_sig: ResultTimeSignature | None, measure_number: int
) -> tuple[ResultTimeSignature | None, ResultMeasure]:
    has_chords = False
    has_chord_before_time_sig = False
    for symbol in measure.symbols:
        if isinstance(symbol, ResultTimeSignature):
            time_sig = symbol
            if has_chords:
                has_chord_before_time_sig = True
        if isinstance(symbol, ResultChord):
            has_chords = True

    if time_sig is None or has_chord_before_time_sig:
        return (time_sig, measure)

    beat_duration = 4 / time_sig.denominator * constants.duration_of_quarter
    expected_duration = time_sig.numerator * beat_duration

    (actual_duration, initial_confidence) = sum_up_duration(measure.symbols)

    difference = abs(expected_duration - actual_duration)

    if difference == 0 and initial_confidence > 0:
        return (time_sig, measure)

    options = create_all_alternatives(measure.symbols)
    if len(options) <= 1:
        return (time_sig, measure)
    best_option: list[ResultSymbol] = []
    best_confidence = 0
    for option in options:
        (duration, confidence) = sum_up_duration(option)
        if duration == expected_duration and confidence > best_confidence:
            best_option = option

    if len(best_option):
        eprint(
            "Measure",
            measure_number,
            "duration mismatch",
            actual_duration,
            "vs",
            expected_duration,
            "replaced",
            measure.symbols,
            "with",
            best_option,
        )
        return (time_sig, ResultMeasure(best_option))
    else:
        eprint(
            "Measure",
            measure_number,
            "duration mismatch",
            actual_duration,
            "vs",
            expected_duration,
            "but found no alternative to",
            measure.symbols,
        )

    return (time_sig, measure)


def sum_up_duration(symbols: list[ResultSymbol]) -> tuple[int, float]:
    duration_sum: Fraction = Fraction(0)
    confidence = 1.0
    # if triplets don't come in a group of 3 then we know that this measure is incorrect
    triplet_exp = 3
    triplet_count = 0
    for symbol in symbols:
        if isinstance(symbol, ResultChord):
            duration_sum += symbol.duration.duration
            if symbol.duration.modifier == DurationModifier.TRIPLET:
                triplet_count += 1
            else:
                if triplet_count % triplet_exp != 0:
                    confidence = 0
                triplet_count = 0
            confidence *= symbol.duration.confidence
        else:
            if triplet_count % triplet_exp != 0:
                confidence = 0
            triplet_count = 0

    if triplet_count % triplet_exp != 0:
        confidence = 0

    return (int(duration_sum), confidence)


def create_all_alternatives_limited(
    symbols: list[ResultSymbol], max_mutations: int = 2
) -> list[list[ResultSymbol]]:
    result: list[tuple[list[ResultSymbol | None], int]] = [([], 0)]  # (sequence, mutation_count)

    for symbol in symbols:
        alternatives: list[ResultSymbol | None] = []
        if isinstance(symbol, ResultChord):
            alternatives += create_alternatives(symbol)
        else:
            alternatives = [symbol]

        new_result = []
        for seq, mutation_count in result:
            for alt in alternatives:
                is_mutation = alt != symbol
                new_mutation_count = mutation_count + (1 if is_mutation else 0)
                if new_mutation_count <= max_mutations:
                    new_result.append((seq + [alt], new_mutation_count))
        result = new_result

    return filter_none_values([seq for seq, _ in result])


def create_all_alternatives(symbols: list[ResultSymbol]) -> list[list[ResultSymbol]]:
    result: list[list[ResultSymbol | None]] = [[]]
    for symbol in symbols:
        alternatives: list[ResultSymbol | None] = []
        if isinstance(symbol, ResultChord):
            alternatives += create_alternatives(symbol)
        else:
            alternatives = [symbol]

        new_result = []
        for seq in result:
            for alt in alternatives:
                new_result.append(seq + [alt])
        result = new_result

    return filter_none_values(result)


def create_alternatives(chord: ResultChord) -> list[ResultChord | None]:
    # Adding "None" to the alternatives would check if removing the item fixes the rhythm
    alternatives: list[ResultChord | None] = [chord]
    if len(chord.notes) == 1 and chord.notes[0].duration.modifier == DurationModifier.TRIPLET:
        note = chord.notes[0]
        if note.alternative_duration:
            alternatives.append(
                ResultChord(
                    note.alternative_duration,
                    [ResultNote(note.pitch, note.alternative_duration, note.duration)],
                    chord.duration,
                )
            )
    return alternatives


def filter_none_values(result: list[list[ResultSymbol | None]]) -> list[list[ResultSymbol]]:
    return [[symbol for symbol in inner if symbol is not None] for inner in result]
