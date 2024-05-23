from homr.debug import AttentionDebug
from homr.model import Note, NoteGroup, Staff
from homr.results import ResultClef, ResultNote, ResultNoteGroup, ResultStaff
from homr.simple_logging import eprint
from homr.tr_omr_parser import parse_tr_omr_output
from homr.transformer.configs import default_config
from homr.transformer.staff2score import Staff2Score


def _get_notes_with_clef_information(
    staff: ResultStaff,
) -> list[list[tuple[ResultClef | None, ResultNote | ResultNoteGroup]]]:
    clef: ResultClef | None = None
    result = []
    for measure in staff.measures:
        result_measure = []
        for symbol in measure.symbols:
            if isinstance(symbol, ResultClef):
                clef = symbol
            elif isinstance(symbol, ResultNote | ResultNoteGroup):
                result_measure.append((clef, symbol))
        if len(result_measure) > 0:
            result.append(result_measure)
    return result


def _override_note_pitches(staff: Staff, result: ResultStaff) -> None:  # noqa: C901
    """
    Adjust the note and octave of the notes as detected by TrOMR with results we obtained by
    traditional homer methods. That's because homer is more accurate than TrOMR.

    This method never touches accidentals, as TrOMR is more accurate in that regard.
    """
    homer_measures = staff.get_measures()
    result_measures = _get_notes_with_clef_information(result)
    total_notes = 0
    total_notes_checked = 0
    total_notes_changed = 0
    if len(homer_measures) != len(result_measures):
        single_homer_measure = [note for measure in homer_measures for note in measure]
        single_result_measure = [note for measure in result_measures for note in measure]
        if len(single_homer_measure) != len(single_result_measure):
            eprint(
                "Skipping pitch override because measures do not match "
                + str(len(homer_measures))
                + " vs. "
                + str(len(result_measures))
            )
            return
        else:
            homer_measures = [single_homer_measure]
            result_measures = [single_result_measure]
            eprint(
                "Pitch override based on notes, as number of measures do not match, "
                + "but number of notes do"
            )
    for k in range(min(len(homer_measures), len(result_measures))):
        homer_notes = homer_measures[k]
        result_clef_and_notes = result_measures[k]
        total_notes += len(result_clef_and_notes)
        if len(homer_notes) != len(result_clef_and_notes):
            continue
        for i in range(min(len(homer_notes), len(result_clef_and_notes))):
            result_clef = result_clef_and_notes[i][0]
            result_note = result_clef_and_notes[i][1]
            if result_clef is None:
                continue
            clef_type = result_clef.clef_type
            circle_of_fifth = result_clef.circle_of_fifth
            home_note_symbol = homer_notes[i]
            if isinstance(result_note, ResultNote) and isinstance(home_note_symbol, Note):
                homer_pitch = home_note_symbol.get_pitch(clef_type, circle_of_fifth).to_result()
                if (
                    homer_pitch.step != result_note.pitch.step
                    or homer_pitch.octave != result_note.pitch.octave
                ):
                    total_notes_changed += 1
                total_notes_checked += 1
                result_note.pitch.step = homer_pitch.step
                result_note.pitch.octave = homer_pitch.octave
            elif isinstance(result_note, ResultNoteGroup) and isinstance(
                home_note_symbol, NoteGroup
            ):
                homer_pitches = [
                    p.get_pitch(clef_type, circle_of_fifth).to_result()
                    for p in home_note_symbol.notes
                ]
                for j in range(min(len(homer_pitches), len(result_note.notes))):
                    if (
                        homer_pitches[j].step != result_note.notes[j].pitch.step
                        or homer_pitches[j].octave != result_note.notes[j].pitch.octave
                    ):
                        total_notes_changed += 1
                    total_notes_checked += 1
                    result_note.notes[j].pitch.step = homer_pitches[j].step
                    result_note.notes[j].pitch.octave = homer_pitches[j].octave
    eprint(
        "Pitch override",
        total_notes,
        "total,",
        total_notes_checked,
        "checked,",
        total_notes_changed,
        "changed",
    )


inference: Staff2Score | None = None


def parse_staff_tromr(staff: Staff, staff_file: str, debug: AttentionDebug | None) -> ResultStaff:
    global inference  # noqa: PLW0603
    print("Running TrOmr inference on", staff_file)
    if inference is None:
        inference = Staff2Score(default_config)
    output = str.join("", inference.predict(staff_file, debug=debug, staff=staff))
    result = parse_tr_omr_output(output)
    # _override_note_pitches(staff, result)
    return result
