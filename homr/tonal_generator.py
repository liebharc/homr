from homr.results import DurationModifier, ResultChord, ResultClef, ResultStaff


def notes_to_tonal_notation(staffs: list[ResultStaff]) -> list[str]:
    """Converts the notes and rests in the staff to tonal notation.
    Tonal notation is useful if you need something short and easy to read to understand
    quickly what has been detected.

    The tonal notation has been extended with clef information and markings for measures and chords.
    """
    duration_dict = {
        "whole": "w",
        "half": "h",
        "quarter": "q",
        "eighth": "e",
        "16th": "s",
        "32nd": "t",
        "64th": "sf",
    }
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
                    if symbol.is_rest:
                        chord.append("r")
                    for note in symbol.notes:
                        chord.append(
                            f"{note.pitch.step}{note.pitch.alter_str()}{note.pitch.octave}"
                        )
                    duration_modifier = ""
                    if symbol.duration.modifier == DurationModifier.DOT:
                        duration_modifier = "."
                    duration = (
                        duration_dict.get(symbol.duration.duration_name, "q") + duration_modifier
                    )
                    pitches = str.join("&", chord)
                    measure_results.append(f"{pitches}-{duration}")
            measures.append(str.join("+", measure_results))
        results.append(str.join("|", measures))

    return results
