import re

from homr.circle_of_fifths import strip_naturals
from homr.transformer.vocabulary import EncodedSymbol, empty, nonote
from training.datasets.staff_merging import (
    EncodedSymbolWithPos,
    merge_upper_and_lower_staff,
)
from training.transformer.training_vocabulary import VocabularyStats, check_token_lines


def convert_kern_to_tokens(lines: list[str]) -> list[EncodedSymbol]:
    staffs = _merge_multiple_voices_on_the_same_staff(lines)
    merged = merge_upper_and_lower_staff(
        [_convert_single_staff(staff_no, staff) for staff_no, staff in enumerate(reversed(staffs))]
    )
    merged = _remove_redundant_key_changes(merged)
    merged = _fix_final_repeat_start(merged)
    merged = strip_naturals(merged)
    return merged


def _merge_multiple_voices_on_the_same_staff(
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
            continue

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


def _remove_redundant_key_changes(symbols: list[EncodedSymbol]) -> list[EncodedSymbol]:
    last_symbol = EncodedSymbol("")
    result = []
    for symbol in symbols:
        # Key signature was already added, this happend e.g. in
        # datasets/grandstaff/scarlatti-d/keyboard-sonatas/L348K244/min3_up_m-89-93.tokens
        # as there is a clef change for one staff and a key change for both, but the
        # key change doesn't happen in one line then
        if symbol.rhythm.startswith("keySignature") and symbol.rhythm == last_symbol.rhythm:
            continue
        result.append(symbol)
        last_symbol = symbol
    return result


def _fix_final_repeat_start(symbols: list[EncodedSymbol]) -> list[EncodedSymbol]:
    """
    If a measure ends with a repeat start then in the actual image you only see
    a barline rendered.
    """
    if len(symbols) == 0:
        return symbols
    if symbols[-1].rhythm == "repeatEndStart":
        symbols[-1].rhythm = "repeatEnd"
    if symbols[-1].rhythm == "repeatStart":
        symbols[-1].rhythm = "barline"
    return symbols


def _convert_single_staff(staff_no: int, lines: list[str]) -> list[EncodedSymbolWithPos]:
    converter = HumdrumKernConverter()
    return converter.convert_humdrum_kern(staff_no, lines)


class HumdrumKernConverter:
    def __init__(self) -> None:
        # Grandstaff definitions: https://link.springer.com/article/10.1007/s10032-023-00432-z#Tab1
        self.ignore_beams = ("L", "J", "K", "k")
        self.ignore_alteration_displays = ("x", "X", "i", "I", "j", "Z", "y", "Y")
        self.ignore_tie_continue = "_"
        # According to the grandstaff paper angleBracketOpen & Close stands for tieStart and tieEnd
        # but there is no tie visible
        self.angled_brackets = ("<", ">")

    def _accidental_to_lift(self, accidental: str) -> str:
        return {"-": "b", "--": "bb", "#": "#", "##": "##", "n": "N"}.get(accidental, empty)

    def _articulation_from_suffix(self, suffix: str) -> str:
        for symbol in self.ignore_beams:
            suffix = suffix.replace(symbol, "")
        for symbol in self.ignore_alteration_displays:
            suffix = suffix.replace(symbol, "")
        for symbol in self.angled_brackets:
            suffix = suffix.replace(symbol, "")
        suffix = suffix.replace(self.ignore_tie_continue, "")

        if not suffix:
            return empty

        mapping = {":": "arpeggiate", "[": "tieStart", "]": "tieStop"}
        articulations = []
        for char in suffix:
            articulations.append(mapping[char])
        return str.join("_", articulations)

    def parse_clef(self, clef: str) -> EncodedSymbol:
        clef_name = clef.split()[0].replace("*clef", "clef_")
        return EncodedSymbol(clef_name, empty, empty, empty)

    def parse_key_signature(self, key_signature: str) -> EncodedSymbol:
        mapping = {
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
        circle = mapping[key_signature.split()[0]]
        return EncodedSymbol(f"keySignature_{circle}")

    def parse_time_signature(self, ts: str) -> EncodedSymbol:
        ts_val = ts.split()[0].replace("*M", "")
        parts = ts_val.split("/")
        return EncodedSymbol(f"timeSignature/{parts[1]}")

    def parse_duration(self, dur: str, is_rest: bool = False, is_grace: bool = False) -> str:
        if not dur:
            raise ValueError("Missing duration " + dur)
        has_dot = dur.endswith(".")
        dur_val = int(dur.replace(".", ""))
        grace = "G" if is_grace else ""
        base = "rest" if is_rest else "note"
        return f"{base}_{dur_val}{grace}{'.' if has_dot else ''}"

    def kern_note_to_pitch(self, kern_note: str) -> str:
        letter = kern_note[0].upper()
        count = len(kern_note)
        return f"{letter}{3 + count}" if kern_note[0].islower() else f"{letter}{4 - count}"

    def parse_note_or_rest(self, token: str) -> EncodedSymbol:
        match = re.match(r"(\d*\.*)([a-grA-GR]+)(--|-|n|##|#)?([^#]*)", token)
        if not match:
            raise Exception(f"Invalid note {token}")

        dur, pitch, accidental, suffix = match[1], match[2], match[3], match[4]
        is_rest = pitch == "r"
        is_grace = "q" in suffix
        suffix = suffix.replace("q", "")

        rhythm_key = self.parse_duration(dur or "4", is_rest=is_rest, is_grace=is_grace)
        if is_rest:
            return EncodedSymbol(rhythm_key, empty, empty, empty)

        lift_val = self._accidental_to_lift(accidental)
        pitch_val = self.kern_note_to_pitch(pitch)
        articulation_val = self._articulation_from_suffix(suffix)
        return EncodedSymbol(rhythm_key, pitch_val, lift_val, articulation_val)

    def parse_barline(self, line: str) -> list[EncodedSymbol]:
        symbol = line.split(" ")[0]
        mapping = {
            "=:|!|:": ["repeatEndStart"],
            "=": ["barline"],
            "=-": [],  # barline after clef, key and time sig
            "==:|!": ["repeatEnd"],
            "==": ["bolddoublebarline"],
            "=:|!": ["repeatEnd"],
            "=!|:": ["repeatStart"],
            "=||": ["doublebarline"],
            "=|!": ["barline"],
        }
        return [EncodedSymbol(s) for s in mapping[symbol]]

    def _get_default_clef(self, staff_no: int) -> EncodedSymbol:
        if staff_no == 0:
            return EncodedSymbol("clef_G2", empty, empty, empty, "upper")
        return EncodedSymbol("clef_F4", empty, empty, empty, "lower")

    def _add_line_numbers(self, lines: list[str]) -> list[tuple[int, str]]:
        """
        Control chars seem to have no specific order and must be treated
        as if we would be on the same line.
        """
        line_no = 0
        result: list[tuple[int, str]] = []
        in_control_group = True
        for line in lines:
            control_line = line.startswith("*")
            if control_line and in_control_group:
                result.append((line_no, line))
            else:
                line_no += 1
                result.append((line_no, line))
                in_control_group = control_line
        return result

    def convert_humdrum_kern(
        self, staff_no: int, lines: list[str]
    ) -> list[EncodedSymbolWithPos]:  # noqa: C901
        result: list[EncodedSymbolWithPos] = []

        clef = EncodedSymbolWithPos(-10, self._get_default_clef(staff_no))
        keySignature = EncodedSymbolWithPos(-9, EncodedSymbol("keySignature_0"))
        timeSignature = EncodedSymbolWithPos(-8, EncodedSymbol("timeSignature/4"))
        initial_signature_was_added = False
        for line_no, line in self._add_line_numbers(lines):
            if line.startswith("="):
                if initial_signature_was_added:
                    parsed = self.parse_barline(line)
                    result.extend([EncodedSymbolWithPos(line_no, p) for p in parsed])
            elif line.startswith("*k"):
                keySignature = EncodedSymbolWithPos(-9, self.parse_key_signature(line))
                if initial_signature_was_added:
                    result.append(EncodedSymbolWithPos(line_no, keySignature.symbol))
            elif line.startswith("*M"):
                timeSignature = EncodedSymbolWithPos(-8, self.parse_time_signature(line))
                if initial_signature_was_added:
                    result.append(EncodedSymbolWithPos(line_no, timeSignature.symbol))
            elif line.startswith("*clef"):
                clef = EncodedSymbolWithPos(-10, self.parse_clef(line))
                if initial_signature_was_added:
                    result.append(EncodedSymbolWithPos(line_no, clef.symbol))
            elif line.startswith("*"):
                # All other control instructions can be ignored
                pass
            else:
                if not initial_signature_was_added:
                    # Symbols can be appear in various order
                    # and duplicated (in which the latest wins)
                    result.append(clef)
                    result.append(keySignature)
                    result.append(timeSignature)
                    initial_signature_was_added = True
                symbols = line.split()
                for token in symbols:
                    if token != nonote:
                        result.append(EncodedSymbolWithPos(line_no, self.parse_note_or_rest(token)))

        return result


if __name__ == "__main__":
    import glob
    import os

    from homr.simple_logging import eprint

    stats = VocabularyStats()
    files = glob.glob(os.path.join("datasets", "grandstaff", "**", "**.krn"), recursive=True)
    for file in files:
        with open(file, encoding="utf-8", errors="ignore") as f:
            tokens = convert_kern_to_tokens(f.readlines())
            check_token_lines(tokens)
            stats.add_lines(tokens)
    eprint("Stats", stats)
