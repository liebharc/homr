import re

from homr.circle_of_fifths import (
    get_circle_of_fifth_notes,
    key_signature_to_circle_of_fifth,
)


def filter_for_kern(lines: list[str]) -> list[str]:
    filtered = []
    kern_indices = []
    interpretation_found = False

    for raw_line in lines:
        line = raw_line.rstrip("\n")

        # Before we hit the exclusive interpretation line, keep everything
        if not interpretation_found:
            if line.startswith("**"):
                fields = line.split("\t")
                kern_indices = [i for i, field in enumerate(fields) if field == "**kern"]
                filtered_fields = [fields[i] for i in kern_indices]
                filtered.append("\t".join(filtered_fields))
                interpretation_found = True
            else:
                filtered.append(line)
            continue

        # After interpretation line: filter based on kern indices
        fields = line.split("\t")
        filtered_fields = []

        for i in kern_indices:
            # Use "." if line is short (e.g., missing columns)
            filtered_fields.append(fields[i] if i < len(fields) else ".")

        filtered.append("\t".join(filtered_fields))

    return filtered


def merge_kerns(voices: list[str]) -> str:
    split = [v.splitlines() for v in voices]
    merged = zip(*split, strict=True)
    return str.join("\n", [str.join("\t", e) for e in merged])


def get_symbols_from_file(file: str) -> list[str]:
    with open(file) as f:
        return get_symbols(filter_for_kern(f.readlines()))


def _merge_multiple_voices(lines: list[str]) -> list[str]:  # noqa: C901, PLR0912
    """
    This method tries to merge multiple voices for the same staff together.
    """
    merges = []
    result = []
    for raw_line in lines:
        line = raw_line.strip()
        if "*clef" in line:
            clefs = line.split()
            if len(clefs) == 3:  # noqa: PLR2004
                if clefs[0] == clefs[1]:
                    merges = [[0, 1], [2]]
                    result.append(str.join("\t", [clefs[0], clefs[2]]))
                else:
                    merges = [[0], [1, 2]]
                    result.append(str.join("\t", [clefs[0], clefs[1]]))
            elif len(clefs) == 4:  # noqa: PLR2004
                if clefs[0] == clefs[1] and clefs[2] == clefs[3]:
                    merges = [[0, 1], [2, 3]]
                    result.append(str.join("\t", [clefs[0], clefs[2]]))
                else:
                    return lines
            else:
                return lines
        elif len(merges) == 0:
            result.append(line)
        else:
            cells = line.split("\t")
            if len(cells) <= 2:  # noqa: PLR2004
                result.append(line)
            else:
                new_cells = []
                for merge in merges:
                    current_cell = []
                    for index in merge:
                        if index < len(cells):
                            current_cell.append(cells[index])
                    new_cells.append(str.join(" ", current_cell))
                result.append(str.join("\t", new_cells))
    return result


def get_symbols(lines: list[str]) -> list[str]:  # noqa: C901, PLR0912
    lines = _merge_multiple_voices(lines)
    standalone_dot = "."
    ignore_symbols = [standalone_dot]
    symbols = []
    for line in lines:
        norm_line = line.strip()
        if norm_line.startswith("!") or not norm_line:
            continue
        is_key_or_time = norm_line.startswith(("*M", "*k", "*clef")) and not norm_line.startswith(
            "*MM"
        )
        if norm_line.startswith("*") and not is_key_or_time:
            continue
        fields = norm_line.split("\t")
        symbols_in_this_field = []
        for i, field in enumerate(fields):
            for symbol in field.split():
                if not symbol:
                    continue
                if symbol in ignore_symbols:
                    continue
                if symbol.startswith("="):
                    # With this mapping of "=" we ignore information about measures
                    symbols_in_this_field.append("=")
                    continue

                # By stripping these symbols we ignore encoding of phrases, slurs and ties
                if not symbol.startswith("*"):
                    phrases_slurs_ties = "()[]{}_;"
                    stem_symbols = "/\\"
                    articulation_symbols = "^'\"~`"
                    other_symbols = "@$<>"
                    for ignored_symbol in (
                        stem_symbols + phrases_slurs_ties + articulation_symbols + other_symbols
                    ):
                        symbol = symbol.replace(ignored_symbol, "")  # noqa: PLW2901
                if symbol == "*":
                    continue
                symbols_in_this_field.append(symbol)
            if i < len(fields) - 1:
                symbols_in_this_field.append("<TAB>")
            symbols.extend(_sort_notes(symbols_in_this_field))
            symbols_in_this_field = []
        while len(symbols) > 0 and symbols[-1] == "<TAB>":
            # To use tokens more efficiently we ignore all tabs which are immediately followed
            # by a newline
            del symbols[-1]
        symbols.append("<NL>")
    return symbols


def _kern_pitch_value(note: str) -> int:
    """
    Convert a **kern note to a numeric pitch value for comparison.
    This handles octave by letter repetition and case (upper = lower octave).
    """
    match = re.match(r"(\d*)([a-gA-G]+)", note)
    if not match:
        return -1  # fallback for unrecognized

    _, pitch = match.groups()

    base_pitches = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}
    letter = pitch[0].lower()
    octave_shift = len(pitch) - 1  # 'cc' -> +1 octave, 'ccc' -> +2

    if pitch[0].islower():
        # Octave 5 + shifts (middle C = c)
        base_octave = 5 + octave_shift
    else:
        # Octave 3 - shifts (C = octave 4, B = 3)
        base_octave = 4 - octave_shift

    midi = base_octave * 12 + base_pitches[letter]
    return -midi


def _is_note(symbol: str) -> bool:
    return re.match(r"\d*[a-gA-G]+", symbol) is not None


def _sort_notes(symbols: list[str]) -> list[str]:
    """
    Sorts only the notes from lowest to highest pitch, leaving non-note symbols in place.
    """
    # Extract and sort notes
    notes = [(i, s) for i, s in enumerate(symbols) if _is_note(s)]
    sorted_notes = sorted([s for _, s in notes], key=_kern_pitch_value, reverse=True)

    # Reinsert sorted notes
    result = symbols[:]
    j = 0
    for i, s in enumerate(symbols):
        if _is_note(s):
            result[i] = sorted_notes[j]
            j += 1
    return result


def split_symbol_into_token(symbol: str) -> tuple[str, str, str, str]:
    # Splits a token into a token for each decoder: is_note, rhythm, pitch, lift
    # Refer to https://www.humdrum.org/rep/kern/ for a description of
    # the different symbols in kern notation

    match = re.match("^([0-9q]+[\\.q]*)?([a-gA-G]+|r|R|RR)([#n-]*)?(.*)?$", symbol)
    if match:
        # By ignroing group 4, we ignore information about beams
        rhythm = match[1]  # r=rest, R=unpitched note, RR=semi unpitched note
        pitch = match[2]
        lift = match[3]
        if not rhythm:
            rhythm = "q"  # grace note
        if not lift:
            lift = ""
        if rhythm.startswith("0"):
            rhythm = "0"
        rhythm = rhythm.replace("qq", "q")
        rhythm = rhythm.replace("..", ".")
        return ("note", rhythm, pitch, lift)
    if symbol.startswith("*k"):
        # The loss function between the decoders is configured such that
        # pitch and lift must return nonote if the is_note decoder return nonote
        # as a consequence: Every time we want to return a symbol different from
        # nonte in lift and pitch then the is_note decoder must return "note"
        return ("note", "*k", "*symbol", symbol)
    if symbol == "=":
        return ("note", symbol, "*symbol", symbol)
    if symbol.startswith("*clef"):
        return ("note", "*clef", symbol, symbol)
    return ("nonote", symbol, "nonote", "nonote")


def split_kern_file_into_measures(kern_file: str) -> tuple[int, str, list[str]]:
    # Return: Number of staffs, key and time sig, measures
    measures = []
    number_of_staffs = 0
    current_measure: list[str] = []
    before_first_measure = ""

    with open(kern_file) as kern:
        lines = kern.readlines()
        lines = filter_for_kern(lines)
        for line in lines:
            if line.startswith("*staff"):
                number_of_staffs = len(line.split())

            if line.startswith("="):
                if before_first_measure == "":
                    before_first_measure = str.join("\n", current_measure) + "\n"
                else:
                    measures.append(str.join("\n", current_measure))
                current_measure = [line]
            else:
                current_measure.append(line)

    return (number_of_staffs, before_first_measure, measures)


def split_kern_measures_into_voices(  # noqa: C901
    number_of_staffs: int, before_first_measure: str, measures: list[str]
) -> tuple[list[str], list[list[str]]]:
    prelude_lines = before_first_measure.split("\n")
    prelude_per_voice: list[list[str]] = [[] for _ in range(number_of_staffs)]
    for line in prelude_lines:
        cells = line.strip().split("\t")
        if len(cells) == 1:
            for voice in prelude_per_voice:
                voice.append(cells[0])
        else:
            for i in range(number_of_staffs):
                prelude_per_voice[i].append(cells[i])

    measures_per_voice: list[list[str]] = [[] for _ in range(number_of_staffs)]
    for measure in measures:
        lines = measure.split("\n")
        lines_per_voice: list[list[str]] = [[] for _ in range(number_of_staffs)]
        for line in lines:
            cells = line.strip().split("\t")
            if len(cells) == 1:
                for voice in lines_per_voice:
                    voice.append(cells[0])
            else:
                for i in range(number_of_staffs):
                    lines_per_voice[i].append(cells[i])

        for i in range(number_of_staffs):
            measures_per_voice[i].append(str.join("\n", lines_per_voice[i]))

    preludes = [str.join("\n", v) + "\n" for v in prelude_per_voice]
    return (preludes, measures_per_voice)


def load_and_sanitize_kern_file(filename: str) -> str:
    symbols = get_symbols_from_file(filename)
    tokens = str.join(" ", symbols)
    tokens = tokens.replace("<NL>", "\n")
    tokens = tokens.replace("<TAB>", "\t")
    return tokens


def semantic_to_kern(semantic_path: str) -> str:
    with open(semantic_path) as f:
        first_line = f.readline()
        return semantic_to_kern_notation(first_line)


def semantic_to_kern_notation(semantic: str) -> str:
    translations = {"clef-G2": "*clefG2", "clef-F2": "*clefF2", "barline": "="}
    result = ["**kern"]
    parts = re.split(r"[\s\+]", semantic)
    for part in parts:
        if part in translations:
            result.append(translations[part])
        if "note" in part or "rest" in part:
            chord = part.split("|")
            new_chord = []
            for note in chord:
                new_chord.append(translate_note_or_rest(note))
            result.append(str.join(" ", new_chord))
        if part.startswith("keySignature-"):
            result.append(translate_key(part))
        if part.startswith("timeSignature-"):
            result.append(translate_time(part))
    return str.join("\n", result)


def translate_duration(duration: str, is_grace: bool) -> str:
    duration_map = {
        # From the humdrum definition: The number zero (0) is reserved for the breve duration
        "quadruple_whole": "0",
        "double_whole": "0",
        "whole": "1",
        "half": "2",
        "quarter": "4",
        "eighth": "8",
        "sixteenth": "16",
        "thirty_second": "32",
        "sixty_fourth": "64",
        "hundred_twenty_eighth": "128",
    }

    has_dot = "." in duration
    suffix = "." if has_dot else ""
    if is_grace:
        suffix = "q"
    duration = duration.replace(".", "")

    return duration_map[duration] + suffix


def translate_note_or_rest(note: str) -> str:

    accidental_map = {"N": "", "b": "-", "#": "#"}
    is_grace = note.startswith("grace")
    note = note.replace("grace", "").replace("_fermata", "").replace("³", "")

    if note.startswith("multirest-"):
        return "1r"

    if note.startswith("rest-"):
        duration = note.split("-")[1]
        return f"{translate_duration(duration, is_grace)}r"

    elif note.startswith("note-"):
        _, pitch_dur = note.split("note-")
        parts = pitch_dur.split("_")
        pitch_part = parts[0]
        duration = str.join("_", parts[1:])

        # Extract pitch name, accidental, and octave
        if pitch_part[-1] in ["b", "#", "N"]:
            accidental = accidental_map[pitch_part[-1]]
            pitch_letter = pitch_part[0].lower()
            octave = int(pitch_part[1:-1])
        elif pitch_part[-2] in ["b", "#", "N"]:
            accidental = accidental_map[pitch_part[-2]]
            pitch_letter = pitch_part[0].lower()
            octave = int(pitch_part[2:])
        else:
            accidental = ""
            pitch_letter = pitch_part[0].lower()
            octave = int(pitch_part[1:])

        # Determine case and repetitions for octave
        kern_base_octave = 4
        if octave < kern_base_octave:
            kern_pitch = pitch_part[0].upper() * (kern_base_octave - octave)
        elif octave == kern_base_octave:
            kern_pitch = pitch_letter
        else:
            kern_pitch = pitch_letter * (octave - 3)

        return f"{translate_duration(duration, is_grace)}{kern_pitch}{accidental}"

    else:
        raise ValueError(f"Invalid input format: {note}")


def translate_key(key: str) -> str:
    circle = key_signature_to_circle_of_fifth(key.split("-")[1])
    notes = get_circle_of_fifth_notes(circle)
    sym = "#" if circle > 0 else "-"
    return "*k[" + str.join("", [n.lower() + sym for n in notes]) + "]"


def translate_time(time: str) -> str:
    if time == "timeSignature-C":
        return "*M4/4"
    if time == "timeSignature-C/":
        return "*M2/2"
    return time.replace("timeSignature-", "*M")
