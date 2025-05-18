import re

from homr import constants
from homr.results import (
    ClefType,
    DurationModifier,
    ResultChord,
    ResultClef,
    ResultDuration,
    ResultMeasure,
    ResultNote,
    ResultPitch,
    ResultStaff,
    ResultTimeSignature,
    get_min_duration,
)
from homr.simple_logging import eprint


class TrOMRParserKern:
    def __init__(self) -> None:
        self._key_signatures: list[int] = []
        self._time_signatures: list[str] = []
        self._clefs: list[ClefType] = []
        self._number_of_accidentals: int = 0

    def number_of_clefs(self) -> int:
        return len(self._clefs)

    def number_of_key_signatures(self) -> int:
        return len(self._key_signatures)

    def number_of_time_signatures(self) -> int:
        return len(self._time_signatures)

    def number_of_accidentals(self) -> int:
        """
        Returns the number of accidentals including the key signatures.
        """
        return self._number_of_accidentals + sum(
            [abs(key_signature) for key_signature in self._key_signatures]
        )

    def parse_clef(self, clef: str) -> ResultClef:
        clef_type_str = clef.replace("*clef", "")
        clef_type = ClefType(clef_type_str[0], int(clef_type_str[1]))
        self._clefs.append(clef_type)
        return ResultClef(clef_type, 0)

    def parse_key_signature(self, key_signature: str, clef: ResultClef) -> None:
        key_signature_mapping = {
            "*k[b-e-a-d-g-c-f-]": -7,
            "*k[b-e-a-d-g-c-]": -6,
            "*k[b-e-a-d-g-]": -5,
            "*k[b-e-a-d-]": -4,
            "*k[b-e-a-]": -3,
            "*k[b-e-]": -2,
            "*k[b-]": -1,
            "*k[]": 0,
            "*k[f#]": 1,
            "*k[f#c#]": 2,
            "*k[f#c#g#]": 3,
            "*k[f#c#g#d#]": 4,
            "*k[f#c#g#d#a#]": 5,
            "*k[f#c#g#d#a#e#]": 6,
            "*k[f#c#g#d#a#e#b#]": 7,
        }
        if key_signature in key_signature_mapping:
            clef.circle_of_fifth = key_signature_mapping[key_signature]
            self._key_signatures.append(clef.circle_of_fifth)
        else:
            eprint("WARNING: Unrecognized key signature: " + key_signature)

    def parse_time_signature(self, time_signature: str) -> ResultTimeSignature:
        parts = time_signature.replace("*M", "").split("/")
        return ResultTimeSignature(int(parts[0]), int(parts[1]))

    def parse_duration_name(self, duration_name: str) -> int:
        if "q" in duration_name:
            return 0
        duration = int(duration_name.replace(".", "").replace("q", ""))
        whole = 4 * constants.duration_of_quarter
        if duration == 0:
            return whole
        return whole / duration

    def parse_duration(self, duration: str) -> ResultDuration:
        has_dot = duration.endswith(".")
        # is_grace = duration.endswith("q")

        modifier = DurationModifier.NONE
        if has_dot:
            duration = duration[:-1]
            modifier = DurationModifier.DOT
        return ResultDuration(
            self.parse_duration_name(duration),
            modifier,
        )

    def kern_note_to_scientific(self, kern_note: str) -> tuple[str, int]:
        if kern_note == "r":
            return ("", -1)

        letter = kern_note[0].upper()
        count = len(kern_note)

        if kern_note[0].islower():
            octave = 3 + count  # c = C4
        else:
            octave = 4 - count  # C = C3

        return (letter, octave)

    def parse_note_or_rest(self, note: str) -> ResultNote:
        try:
            match = re.match("(\\d+[q\\.]*)([a-gA-G]+)(-|--|n|#|##)?", note)
            duration = match[1]
            alter = None
            accidental = match[3]
            if accidental:
                self._number_of_accidentals += 1
                if accidental.startswith("-"):
                    alter = -1
                elif accidental.startswith("#"):
                    alter = 1
                else:
                    alter = 0

            note_name, octave = self.kern_note_to_scientific(match[2])

            return ResultNote(ResultPitch(note_name, octave, alter), self.parse_duration(duration))
        except Exception:
            eprint("Failed to parse note: " + note)
            return ResultNote(ResultPitch("C", 4, 0), ResultDuration(constants.duration_of_quarter))

    def parse_notes_and_rests(self, notes: str) -> ResultChord | None:
        note_parts = notes.split()
        result_notes = [self.parse_note_or_rest(note_part) for note_part in note_parts]
        # remove rests as we encode them as empty chord
        no_rests = [r for r in result_notes if r.pitch.octave >= 0]
        return ResultChord(get_min_duration(result_notes), no_rests)

    def parse_tr_omr_output(self, output: str) -> ResultStaff:  # noqa: C901
        parts = output.splitlines()
        measures = []
        current_measure = ResultMeasure([])

        parse_functions = {"*clef": self.parse_clef, "*M": self.parse_time_signature}

        for part_raw in parts:
            part = part_raw.strip()
            if part == "=":
                measures.append(current_measure)
                current_measure = ResultMeasure([])
            elif part.startswith("*k"):
                if len(current_measure.symbols) > 0 and isinstance(
                    current_measure.symbols[-1], ResultClef
                ):
                    self.parse_key_signature(part, current_measure.symbols[-1])
            elif part.startswith("*"):
                for prefix, parse_function in parse_functions.items():
                    if part.startswith(prefix):
                        current_measure.symbols.append(parse_function(part))
                        break
            else:
                note_result = self.parse_notes_and_rests(part)
                if note_result is not None:
                    current_measure.symbols.append(note_result)

        if len(current_measure.symbols) > 0:
            measures.append(current_measure)
        return ResultStaff(measures)
