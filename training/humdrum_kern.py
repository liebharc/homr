import re

from homr.transformer.vocabulary import SplitSymbol, empty
from training.transformer.training_vocabulary import VocabularyStats, check_token_lines


def convert_kern_to_tokens(lines: list[str]) -> list[list[SplitSymbol]]:
    staffs = _merge_multiple_voices_on_the_same_staff(lines)
    return [_convert_single_staff(staff) for staff in reversed(staffs)]


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


def _convert_single_staff(lines: list[str]) -> list[SplitSymbol]:
    converter = HumdrumKernConverter()
    return converter.convert_humdrum_kern(lines)


class HumdrumKernConverter:
    def __init__(self) -> None:
        # Grandstaff definitions: https://link.springer.com/article/10.1007/s10032-023-00432-z#Tab1
        self.ignore_beams = ("L", "J", "K", "k")
        self.ignore_alteration_displays = ("x", "X", "i", "I", "j", "Z", "y", "Y")
        self.ignore_tie_continue = "_"
        # According to the grandstaff paper angleBracketOpen & Close stands for tieStart and tieEnd
        # but there is no tie visible
        self.angled_brackets = ("<", ">")

        self.slur_level = 0

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

        mapping = {":": "arpeggiate"}
        return mapping[suffix]

    def parse_clef(self, clef: str) -> SplitSymbol:
        clef_name = clef.split()[0].replace("*clef", "clef_")
        return SplitSymbol(clef_name)

    def parse_key_signature(self, key_signature: str) -> SplitSymbol:
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
        return SplitSymbol(f"keySignature_{circle}")

    def parse_time_signature(self, ts: str) -> SplitSymbol:
        ts_val = ts.split()[0].replace("*M", "")
        parts = ts_val.split("/")
        return SplitSymbol(f"timeSignature/{parts[1]}")

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

    def parse_note_or_rest(self, token: str) -> SplitSymbol:
        match = re.match(r"(\d*\.*)([a-grA-GR]+)(--|-|n|##|#)?([^#]*)", token)
        if not match:
            raise Exception(f"Invalid note {token}")

        dur, pitch, accidental, suffix = match[1], match[2], match[3], match[4]
        is_rest = pitch == "r"
        is_grace = "q" in suffix
        suffix = suffix.replace("q", "")

        rhythm_key = self.parse_duration(dur or "4", is_rest=is_rest, is_grace=is_grace)
        if is_rest:
            return SplitSymbol(rhythm_key, empty, empty, empty)

        lift_val = self._accidental_to_lift(accidental)
        pitch_val = self.kern_note_to_pitch(pitch)
        if "[" in suffix:
            self.slur_level += 1
            suffix = suffix.replace("[", "")
        if "]" in suffix:
            self.slur_level -= 1
            suffix = suffix.replace("]", "")
        articulation_val = self._articulation_from_suffix(suffix)
        return SplitSymbol(rhythm_key, pitch_val, lift_val, articulation_val)

    def parse_barline(self, line: str) -> list[SplitSymbol]:
        symbol = line.split(" ")[0]
        mapping = {
            "=:|!|:": ["repeatEnd", "barline", "repeatStart"],
            "=": ["barline"],
            "=-": [],
            "==:|!": ["repeatEnd", "barline"],
            "==": ["bolddoublebarline"],
            "=:|!": ["repeatEnd", "barline"],
            "=!|:": ["repeatEnd", "barline"],
            "=||": ["doublebarline"],
            "=|!": ["barline"],
        }
        return [SplitSymbol(s) for s in mapping[symbol]]

    def interleave_chord_symbol(self, notes: list[SplitSymbol]) -> list[SplitSymbol]:
        result = []
        for i, note in enumerate(notes):
            last_note = i == len(notes) - 1
            result.append(note)
            if not last_note:
                result.append(SplitSymbol("chord"))
        return result

    def _swap_with_previous(self, results: list[SplitSymbol], swap: tuple[str, ...]) -> None:
        if len(results) < 2:
            return

        item = results[-1]
        previous = results[-2]
        if previous.rhythm.startswith(swap):
            results.pop()
            results.pop()
            results.append(item)
            results.append(previous)

    def convert_humdrum_kern(self, lines: list[str]) -> list[SplitSymbol]:  # noqa: C901
        result: list[SplitSymbol] = []

        clef = SplitSymbol("clef_G2")
        keySignature = SplitSymbol("keySignature_0")
        timeSignature = SplitSymbol("timeSignature/4")
        initial_signature_was_added = False

        for line in lines:
            if line.startswith("="):
                if initial_signature_was_added:
                    parsed = self.parse_barline(line)
                    result.extend(parsed)
            elif line.startswith("*k"):
                keySignature = self.parse_key_signature(line)
                if initial_signature_was_added:
                    result.append(keySignature)
                self._swap_with_previous(result, tuple("timeSignature"))
            elif line.startswith("*M"):
                timeSignature = self.parse_time_signature(line)
                if initial_signature_was_added:
                    result.append(timeSignature)
            elif line.startswith("*clef"):
                clef = self.parse_clef(line)
                if initial_signature_was_added:
                    result.append(clef)
                self._swap_with_previous(result, ("timeSignature", "keySignature"))
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
                chord = []
                for token in symbols:
                    if token != ".":
                        chord.append(self.parse_note_or_rest(token))

                result.extend(self.interleave_chord_symbol(chord))

                if self.slur_level > 0 and len(chord) > 0:
                    result.append(SplitSymbol("tieSlur"))

        if result[-1].rhythm != "barline":
            result.append(SplitSymbol("barline"))

        return result


if __name__ == "__main__":
    import glob
    import os

    from homr.simple_logging import eprint

    stats = VocabularyStats()
    for file in glob.glob(os.path.join("datasets", "grandstaff", "**", "**.krn"), recursive=True):
        with open(file, encoding="utf-8", errors="ignore") as f:
            eprint(file, file.replace(".krn", ".jpg"))
            staffs = convert_kern_to_tokens(f.readlines())
            for tokens in staffs:
                check_token_lines(tokens)
                stats.add_lines(tokens)
    eprint("Stats", stats)
