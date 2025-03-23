from homr import constants
from homr.results import DurationModifier, ResultChord, ResultClef, ResultStaff


def notes_to_tonal_notation(staffs: list[ResultStaff]) -> list[str]:
    """Converts the notes and rests in the staff to tonal notation.
    Tonal notation is useful if you need something short and easy to read to understand
    quickly what has been detected.

    The tonal notation has been extended with clef information and markings for measures and chords.
    """
    results: list[str] = []
    for staff in staffs:
        measures: list[str] = []
        for measure in staff.measures:
            measure_results: list[str] = []
            for symbol in measure.symbols:
                if isinstance(symbol, ResultClef):
                    measure_results.append("clef" + str(symbol.clef_type))
                if isinstance(symbol, ResultChord):
                    chord: list[str] = []
                    for note in symbol.notes:
                        duration_modifier = ""
                        if note.duration.modifier == DurationModifier.DOT:
                            duration_modifier = "."
                        chord.append(
                            str(note.pitch)
                            + "-"
                            + str(note.duration.base_duration / constants.duration_of_quarter / 4)
                            + duration_modifier
                        )
                    measure_results.append(str.join("&", chord))
            measures.append(str.join("+", measure_results))
        results.append(str.join("|", measures))

    return results
