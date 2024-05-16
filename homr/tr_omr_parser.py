from homr import constants
from homr.results import (
    ClefType,
    ResultClef,
    ResultDuration,
    ResultMeasure,
    ResultNote,
    ResultNoteGroup,
    ResultPitch,
    ResultRest,
    ResultStaff,
    ResultTimeSignature,
)
from homr.simple_logging import eprint


def parse_clef(clef: str) -> ResultClef:
    parts = clef.split("-")
    clef_type = parts[1]
    return ResultClef(ClefType.TREBLE if clef_type.startswith("G") else ClefType.BASS, 0)


def parse_key_signature(key_signature: str, clef: ResultClef) -> None:
    key_signature_mapping = {
        "CbM": -7,
        "GbM": -6,
        "DbM": -5,
        "AbM": -4,
        "EbM": -3,
        "BbM": -2,
        "FM": -1,
        "CM": 0,
        "GM": 1,
        "DM": 2,
        "AM": 3,
        "EM": 4,
        "BM": 5,
        "F#M": 6,
        "C#M": 7,
    }
    signature_name = key_signature.split("-")[1]
    if signature_name in key_signature_mapping:
        clef.circle_of_fifth = key_signature_mapping[signature_name]
    else:
        eprint("WARNING: Unrecognized key signature: " + signature_name)


def parse_time_signature(clef: str) -> ResultTimeSignature:
    parts = clef.split("-")
    clef_type = parts[1]
    return ResultTimeSignature(clef_type)


def parse_duration_name(duration_name: str) -> int:
    duration_mapping = {
        "whole": constants.duration_of_quarter * 4,
        "half": constants.duration_of_quarter * 2,
        "quarter": constants.duration_of_quarter,
        "eighth": constants.duration_of_quarter // 2,
        "sixteenth": constants.duration_of_quarter // 4,
        "thirty_second": constants.duration_of_quarter // 8,
    }
    return duration_mapping.get(duration_name, constants.duration_of_quarter // 16)


def _adjust_duration_with_dot(duration: int, has_dot: bool) -> int:
    if has_dot:
        return duration * 3 // 2
    else:
        return duration


def parse_note(note: str) -> ResultNote:
    try:
        note_details = note.split("-")[1]
        pitch_and_duration = note_details.split("_")
        pitch = pitch_and_duration[0]
        duration = pitch_and_duration[1]
        has_dot = duration.endswith(".")
        duration_name = duration[:-1] if has_dot else duration
        note_name = pitch[0]
        octave = int(pitch[1])
        alter = None
        len_with_accidental = 2
        if len(pitch) > len_with_accidental:
            accidental = pitch[2]
            if accidental == "b":
                alter = -1
            elif accidental == "#":
                alter = 1
            else:
                alter = 0

        return ResultNote(
            ResultPitch(note_name, octave, alter),
            ResultDuration(
                _adjust_duration_with_dot(parse_duration_name(duration_name), has_dot), has_dot
            ),
        )
    except Exception:
        eprint("Failed to parse note: " + note)
        return ResultNote(
            ResultPitch("C", 4, 0), ResultDuration(constants.duration_of_quarter, False)
        )


def parse_notes(notes: str) -> ResultNote | ResultNoteGroup:
    note_parts = notes.split("|")
    note_parts = [note_part for note_part in note_parts if note_part.startswith("note")]
    if len(note_parts) == 1:
        return parse_note(note_parts[0])
    else:
        return ResultNoteGroup([parse_note(note_part) for note_part in note_parts])


def parse_rest(rest: str) -> ResultRest:
    rest = rest.split("|")[0]
    duration = rest.split("-")[1]
    has_dot = duration.endswith(".")
    duration_name = duration[:-1] if has_dot else duration
    return ResultRest(
        ResultDuration(
            _adjust_duration_with_dot(parse_duration_name(duration_name), has_dot), has_dot
        )
    )


def parse_tr_omr_output(output: str) -> ResultStaff:
    eprint("TrOMR output: " + output)
    parts = output.split("+")
    measures = []
    current_measure = ResultMeasure([])

    parse_functions = {
        "clef": parse_clef,
        "timeSignature": parse_time_signature,
        "note": parse_notes,
        "rest": parse_rest,
    }

    for part in parts:
        if part == "barline":
            measures.append(current_measure)
            current_measure = ResultMeasure([])
        elif part.startswith("keySignature"):
            if len(current_measure.symbols) > 0 and isinstance(
                current_measure.symbols[-1], ResultClef
            ):
                parse_key_signature(part, current_measure.symbols[-1])
        elif part.startswith("multirest"):
            eprint("Skipping over multirest")
        else:
            for prefix, parse_function in parse_functions.items():
                if part.startswith(prefix):
                    current_measure.symbols.append(parse_function(part))
                    break

    if len(current_measure.symbols) > 0:
        measures.append(current_measure)
    return ResultStaff(measures)
