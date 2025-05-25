import re

from homr import constants
from homr.circle_of_fifths import (
    AbstractKeyTransformation,
    KeyTransformation,
    NoKeyTransformation,
    key_signature_to_circle_of_fifth,
)


def convert_kern_to_semantic(lines: list[str]) -> list[str]:
    staffs = _merge_multiple_voices_on_the_same_staff(lines)
    return [_convert_single_staff(staff) for staff in reversed(staffs)]


def _merge_multiple_voices_on_the_same_staff(  # noqa: C901, PLR0912
    lines: list[str],
) -> list[list[str]]:
    """
    Merges voices into staffs.

    Humdrum kern uses special symbols: *^ and *v to split and merge voices
    """
    staff_lines: list[list[str]] = []
    spine_to_staff: list[int] = []

    for raw_line in lines:
        line = raw_line.rstrip("\n")
        if not line.strip():
            for staff in staff_lines:
                staff.append("")
            continue

        tokens = line.split("\t")

        # Initialize spines
        if all(tok.startswith("**") for tok in tokens):
            spine_to_staff = list(range(len(tokens)))
            staff_lines = [[] for _ in range(max(spine_to_staff) + 1)]
            for i in range(len(staff_lines)):
                staff_lines[i].append("**kern")
            continue

        # Split
        if "*^" in tokens:
            new_map = []
            for i, tok in enumerate(tokens):
                if tok == "*^":
                    new_map.extend([spine_to_staff[i], spine_to_staff[i]])
                else:
                    new_map.append(spine_to_staff[i])
            spine_to_staff = new_map

        # Join
        elif "*v" in tokens:
            new_map = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and tokens[i] == "*v" and tokens[i + 1] == "*v":
                    new_map.append(spine_to_staff[i])
                    i += 2
                else:
                    new_map.append(spine_to_staff[i])
                    i += 1
            spine_to_staff = new_map
            continue

        # Data line
        grouped: dict[int, list[str]] = {}
        for i, tok in enumerate(tokens):
            if i >= len(spine_to_staff):
                continue  # skip extra unexpected tokens
            s = spine_to_staff[i]
            grouped.setdefault(s, []).append(tok)
        for s, items in grouped.items():
            while len(staff_lines) <= s:
                staff_lines.append([])
            staff_lines[s].append(" ".join(items))

    return staff_lines


def _convert_single_staff(lines: list[str]) -> str:
    converter = HumdrumKernConverter()
    return converter.convert_humdrum_kern(lines)


class HumdrumKernConverter:

    def __init__(self) -> None:
        self.key: AbstractKeyTransformation = NoKeyTransformation()

    def parse_clef(self, clef: str) -> str:
        return clef.split()[0].replace("*clef", "clef-")

    def parse_key_signature(self, key_signature: str) -> str:
        key_signature_mapping = {
            "*k[b-e-a-d-g-c-f-]": "CbM",
            "*k[b-e-a-d-g-c-]": "GbM",
            "*k[b-e-a-d-g-]": "DbM",
            "*k[b-e-a-d-]": "AbM",
            "*k[b-e-a-]": "EbM",
            "*k[b-e-]": "BbM",
            "*k[b-]": "FM",
            "*k[]": "CM",
            "*k[f#]": "GM",
            "*k[f#c#]": "DM",
            "*k[f#c#g#]": "AM",
            "*k[f#c#g#d#]": "EM",
            "*k[f#c#g#d#a#]": "BM",
            "*k[f#c#g#d#a#e#]": "F#M",
            "*k[f#c#g#d#a#e#b#]": "C#M",
        }
        key = key_signature_mapping[key_signature.split()[0]]
        circle = key_signature_to_circle_of_fifth(key)
        self.key = KeyTransformation(circle)
        return "keySignature-" + key

    def parse_time_signature(self, time_signature: str) -> str:
        return time_signature.split()[0].replace("*M", "timeSignature-")

    def parse_duration(self, duration: str) -> str:
        if not duration:
            # TODO: Proper grace note handling
            return "hundred_twenty_eighth"
        has_dot = "." in duration
        suffix = "." if has_dot else ""
        duration_value = int(duration.replace(".", ""))
        standard_durations = {
            1: "whole",
            2: "half",
            4: "quarter",
            8: "eighth",
            16: "sixteenth",
            32: "thirty_second",
            64: "sixty_fourth",
            128: "hundred_twenty_eighth",
        }

        if duration_value in standard_durations:
            return standard_durations[duration_value] + suffix

        if duration_value % 3 == 0:
            base = 2 * duration_value // 3
            return standard_durations[base] + constants.triplet_symbol

        if duration_value % 5 == 0:
            base = 4 * duration_value // 5
            return standard_durations[base]  # We have no symbol for quintuplets

        if duration_value % 7 == 0:
            base = 6 * duration_value // 7
            return standard_durations[base]  # We also have no symbol for this case

        raise Exception("Unknown duration " + str(duration))

    def kern_note_to_scientific(self, kern_note: str, alter: str) -> str:
        if kern_note == "r":
            return "rest"

        letter = kern_note[0].upper()
        count = len(kern_note)

        if kern_note[0].islower():
            octave = 3 + count  # c = C4
        else:
            octave = 4 - count  # C = C3

        alter = self.key.add_accidental(letter + str(octave), alter)

        return "note-" + letter + str(octave) + alter

    def parse_note_or_rest(self, note: str) -> str:
        match = re.match("(\\d*[q\\.]*)([a-gA-Gr]+)(-|--|n|#|##)?(q?)", note)
        if not match:
            raise Exception("Invalid note " + note)
        duration = match[1]
        alter = ""
        accidental = match[3]
        is_grace = match[4]
        grace_suffix = "Q" if is_grace else ""
        if accidental:
            if accidental.startswith("-"):
                alter = "b"
            elif accidental.startswith("#"):
                alter = "#"
            elif accidental.startswith("n"):
                alter = "N"

        note_name = self.kern_note_to_scientific(match[2], alter)
        if note_name == "rest":
            return note_name + "-" + self.parse_duration(duration) + grace_suffix
        return note_name + "_" + self.parse_duration(duration) + grace_suffix

    def parse_notes_and_rests(self, notes: str) -> str:
        note_parts = notes.split()
        note_parts = [n for n in note_parts if n != "."]
        result_notes = [self.parse_note_or_rest(note_part) for note_part in note_parts]
        return str.join("|", result_notes)

    def convert_humdrum_kern(self, lines: list[str]) -> str:  # noqa: C901
        symbols = []
        any_notes_in_bar = False

        parse_functions = {"*clef": self.parse_clef, "*M": self.parse_time_signature}
        grace_note = ""

        for line in lines:
            if line.startswith("="):
                if any_notes_in_bar:
                    symbols.append("barline")
                    self.key = self.key.reset_at_end_of_measure()
            elif line.startswith("*k"):
                symbols.append(self.parse_key_signature(line))
            elif line.startswith("*"):
                for prefix, parse_function in parse_functions.items():
                    if line.startswith(prefix):
                        symbols.append(parse_function(line))
                        break
            else:
                note_result = self.parse_notes_and_rests(line)
                if note_result != "":
                    all_graces = all(n.endswith("Q") for n in note_result.split("|"))
                    if all_graces:
                        # Encode grace notes as a chord on the following note
                        grace_note = note_result
                    else:
                        if grace_note != "":
                            chord_notes = note_result.split("|")
                            chord_notes.insert(0, grace_note)
                            note_result = str.join("|", chord_notes)
                            grace_note = ""
                        symbols.append(note_result.replace("Q", ""))
                    any_notes_in_bar = True
        return str.join(" ", symbols)
