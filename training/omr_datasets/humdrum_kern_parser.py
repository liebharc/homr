import re
from enum import Enum

from homr.circle_of_fifths import strip_naturals
from homr.transformer.vocabulary import EncodedSymbol, empty, nonote
from training.omr_datasets.staff_merging import (
    EncodedSymbolWithPos,
    merge_upper_and_lower_staff,
)
from training.transformer.training_vocabulary import VocabularyStats, check_token_lines


class StaffPosition(Enum):
    UPPER = "upper"
    LOWER = "lower"


# How much a barline token says about the line it describes, see _merge_barlines.
_BARLINE_RANK = {
    "barline": 0,
    "doublebarline": 1,
    "bolddoublebarline": 2,
    "repeatStart": 3,
    "repeatEnd": 3,
    "repeatEndStart": 4,
}


def _merge_barlines(kept: EncodedSymbol, dropped: EncodedSymbol) -> EncodedSymbol:
    """Combine two barlines which the image draws as a single one."""
    if {kept.rhythm, dropped.rhythm} == {"repeatEnd", "repeatStart"}:
        return EncodedSymbol("repeatEndStart")
    if _BARLINE_RANK.get(dropped.rhythm, -1) > _BARLINE_RANK[kept.rhythm]:
        return dropped
    return kept


class _StaffState:
    def __init__(self, position: StaffPosition) -> None:
        clef_name = "clef_G2" if position is StaffPosition.UPPER else "clef_F4"
        self.clef = EncodedSymbolWithPos(
            -10, EncodedSymbol(clef_name, empty, empty, empty, empty, position.value)
        )
        self.key = EncodedSymbolWithPos(-9, EncodedSymbol("keySignature_0"))
        self.time = EncodedSymbolWithPos(-8, EncodedSymbol("timeSignature/4"))
        self.explicit_clef_seen = False

        self.position = position
        self.result: list[EncodedSymbolWithPos] = []
        self.opening_added = False
        self.data_seen = False
        self.line_no = 0
        self.in_control_group = True

    def advance_line_no(self, line: str) -> None:
        # Control chars seem to have no specific order and must be treated
        # as if we would be on the same line.
        control_line = line.startswith("*")
        if control_line and self.in_control_group:
            return
        self.line_no += 1
        self.in_control_group = control_line

    def add_barline(self, symbols: list[EncodedSymbol]) -> None:
        """Add a barline, or merge it into the one before if the image draws them as one.

        Kern can write two barlines without anything in between, a repeat end followed
        by a repeat start for example, or the barline a staff system ends with followed
        by the same barline again at the start of the next system. A data line of rests
        or even placeholders (see nonote) means the staff has moved on and whatever
        barline comes next is a line of its own.
        """
        if not self.data_seen:
            if symbols and self.result and self.result[-1].rhythm in _BARLINE_RANK:
                self.result[-1] = EncodedSymbolWithPos(
                    self.result[-1].position, _merge_barlines(self.result[-1].symbol, symbols[0])
                )
            return
        if symbols:
            self.data_seen = False
        self.result.extend(EncodedSymbolWithPos(self.line_no, sym) for sym in symbols)

    def set_key(self, new_key: EncodedSymbol) -> None:
        if self.opening_added and new_key != self.key.symbol:
            self.result.append(EncodedSymbolWithPos(self.line_no, new_key))
        self.key = EncodedSymbolWithPos(-9, new_key)

    def set_time(self, new_time: EncodedSymbol) -> None:
        if self.opening_added and new_time != self.time.symbol:
            self.result.append(EncodedSymbolWithPos(self.line_no, new_time))
        self.time = EncodedSymbolWithPos(-8, new_time)

    def set_clef(self, new_clef: EncodedSymbol) -> None:
        if self.opening_added and (not self.explicit_clef_seen or new_clef != self.clef.symbol):
            self.result.append(EncodedSymbolWithPos(self.line_no, new_clef))
        self.explicit_clef_seen = True
        self.clef = EncodedSymbolWithPos(-10, new_clef)

    def add_note(self, symbol: EncodedSymbol) -> None:
        if not self.opening_added:
            self.result.extend([self.clef, self.key, self.time])
            self.opening_added = True
        self.result.append(EncodedSymbolWithPos(self.line_no, symbol))


def convert_kern_to_tokens(lines: list[str]) -> list[EncodedSymbol]:
    staffs, warnings = _convert_into_staffs(lines)
    if warnings:
        raise ValueError(" ".join(warnings))
    if len(staffs) != 2:
        raise ValueError("Skip scores with only 1 staff")
    staffs = list(reversed(staffs))  # reverse to get treble first
    return _post_process(merge_upper_and_lower_staff(staffs))


def convert_kern_to_parts(lines: list[str]) -> list[list[EncodedSymbol]]:
    staffs, _ = _convert_into_staffs(lines)
    # NED ignores Position, so no need to reverse the staffs
    return [_post_process(merge_upper_and_lower_staff([staff])) for staff in staffs]


def _post_process(symbols: list[EncodedSymbol]) -> list[EncodedSymbol]:
    symbols = _remove_redundant_key_changes(symbols)
    symbols = _fix_final_repeat_start(symbols)
    return strip_naturals(symbols)


def _convert_into_staffs(lines: list[str]) -> tuple[list[list[EncodedSymbolWithPos]], list[str]]:
    def _is_exordium(tokens: list[str]) -> bool:
        return all(tok.startswith("**") for tok in tokens)

    # Count staffs in a first pass before actual parsing. Staff count is the number of
    # spines, but a later document may open extra spines, so take the minimum.
    num_of_staffs = 999
    for line in lines:
        tokens = line.rstrip("\n").split("\t")
        if _is_exordium(tokens):
            num_of_staffs = min(num_of_staffs, len(tokens))
    assert num_of_staffs != 999, "No exordium found"  # noqa: S101

    # Typically we expect num_of_staffs == 2, which the training dataset
    # `grandstaff`` mostly follows.
    # Kern lists the bass staff first and the treble staff second, so the code here should work.
    #
    # However, in the validation dataset `smb`, kern scores concatenate several documents,
    # so num_of_staffs != 2. Then we cannot tell the StaffPosition and assign one arbitrarily.
    # NED ignores Position, so the assignment does not affect NED correctness.
    staffs = [
        HumdrumKernConverter(StaffPosition.LOWER if i == 0 else StaffPosition.UPPER)
        for i in range(num_of_staffs)
    ]
    # Indexed by spine, several spines can point to the same staff (see Split)
    spine_to_staff: list[HumdrumKernConverter] = []
    warnings: list[str] = []

    for raw_line in lines:
        line = raw_line.rstrip("\n")

        # empty
        if not line.strip():
            for staff in staffs:
                staff.feed([""])
            continue

        # comment
        if line.startswith("!"):
            continue

        # real processing
        tokens = line.split("\t")
        num_of_spine = len(tokens)

        # Exordium. Spines beyond the staff count belong to a staff which carries two
        # voices without being split with "*^" (see _count_staffs). We assume they are the
        # bass staff, which is where they usually are.
        if _is_exordium(tokens):
            if num_of_spine == num_of_staffs:
                spine_to_staff = staffs
            else:
                # special case for validation dataset `smb`:
                # one staff carries two voices without being split with "*^",
                # so num_of_spine != num_of_staffs. Here we assume the extra spines
                # belong to the bass staff, which is where they usually are.
                spine_to_staff = [staffs[0]] * (num_of_spine - num_of_staffs + 1) + staffs[1:]
            continue

        assert num_of_spine == len(spine_to_staff), "Number of spines does not match"  # noqa: S101

        # Spine operations: "*^" splits a spine into two voices, "*v" joins them
        # back into one. Both can appear on the same line.
        if "*^" in tokens or "*v" in tokens:
            new_map = []
            i = 0
            while i < num_of_spine:
                staff = spine_to_staff[i]
                if tokens[i] == "*^":
                    new_map.extend([staff, staff])
                    i += 1
                    continue
                new_map.append(staff)
                if tokens[i] != "*v":
                    i += 1
                    continue
                end = i + 1
                # count how many *v there are.
                while end < num_of_spine and tokens[end] == "*v":
                    end += 1
                for _staff in spine_to_staff[i + 1 : end]:
                    if _staff is not staff:
                        warnings.append("voices to join should point to the same staff")
                        break
                i = end
            spine_to_staff = new_map
            continue

        # Data line
        grouped: dict[HumdrumKernConverter, list[str]] = {}
        for staff, tok in zip(spine_to_staff, tokens, strict=True):
            grouped.setdefault(staff, []).append(tok)
        for staff, voices in grouped.items():
            staff.feed(voices)

    return [staff.state.result for staff in staffs], warnings


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


class HumdrumKernConverter:
    def __init__(self, position: StaffPosition) -> None:
        # Grandstaff definitions: https://link.springer.com/article/10.1007/s10032-023-00432-z#Tab1
        self.ignore_beams = ("L", "J", "K", "k")
        self.ignore_alteration_displays = ("x", "X", "i", "I", "j", "Z", "y", "Y")
        self.ignore_tie_continue = "_"
        # According to the grandstaff paper angleBracketOpen & Close stands for tieStart and tieEnd
        # but there is no tie visible
        self.angled_brackets = ("<", ">")

        self.state = _StaffState(position)

    def _accidental_to_lift(self, accidental: str) -> str:
        return {"-": "b", "--": "bb", "#": "#", "##": "##", "n": "N"}.get(accidental, empty)

    def _articulation_from_suffix(self, suffix: str) -> tuple[str, str]:
        for symbol in self.ignore_beams:
            suffix = suffix.replace(symbol, "")
        for symbol in self.ignore_alteration_displays:
            suffix = suffix.replace(symbol, "")
        for symbol in self.angled_brackets:
            suffix = suffix.replace(symbol, "")
        suffix = suffix.replace(self.ignore_tie_continue, "")

        if not suffix:
            return empty, empty

        slur_mapping = {
            "[": "slurStart",
            "]": "slurStop",
            "(": "slurStart",
            ")": "slurStop",
        }
        articulation_mapping = {
            ":": "arpeggiate",
            "'": "staccato",
            "`": "staccatissimo",
            "t": "trill",
            "T": "trill",
            "m": "mordent",
            "M": "trill",  # invertedMordent maps to trill in our XML parser
            "S": "turn",
            "$": "turn",
            "^": "accent",
            ";": "fermata",
        }
        articulations = []
        slurs = []
        for char in suffix:
            if char in slur_mapping:
                slurs.append(slur_mapping[char])
            elif char in articulation_mapping:
                articulations.append(articulation_mapping[char])

        if slurs and articulations:
            return str.join("_", articulations), str.join("_", slurs)
        elif slurs:
            return empty, str.join("_", slurs)
        elif articulations:
            return str.join("_", articulations), empty
        else:
            # For ruff, this case should be excluded with
            # return empty, empty in line 148
            return empty, empty

    def parse_clef(self, clef: str, position: str) -> EncodedSymbol:
        clef_name = clef.split(maxsplit=1)[0].replace("*clef", "clef_")
        defaults = {"clef_F": "clef_F4", "clef_G": "clef_G2", "clef_C": "clef_C3"}
        clef_name = defaults.get(clef_name, clef_name)
        return EncodedSymbol(clef_name, empty, empty, empty, empty, position)

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
            "*kcancel": 0,
            "*kcancek": 0,  # typo for *kcancel in smb sample 267
        }
        circle = mapping[key_signature.split(maxsplit=1)[0]]
        return EncodedSymbol(f"keySignature_{circle}")

    def parse_time_signature(self, ts: str) -> EncodedSymbol:
        ts_val = ts.split(maxsplit=1)[0].replace("*M", "")
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

    _DUR_RE = re.compile(r"[()[\]<>&/\\^~yYxXiIjZN]*(\d+\.?)")

    def _extract_dur(self, token: str) -> str | None:
        """Return the raw duration string from a kern token, or None if absent."""
        m = self._DUR_RE.match(token)
        return m.group(1) if m else None

    def parse_note_or_rest(
        self, token: str, position: str, default_dur: str = "4"
    ) -> EncodedSymbol:
        # Prefix: slur/tie/accent/stem/roll/alteration markers before the duration.
        # Between duration and pitch: grace note q, sforzando ^^, tuplet % ratios, etc.
        # Non-capturing group for "between" keeps group indices identical to before.
        match = re.match(
            r"[()[\]<>&/\\^~yYxXiIjZN]*(\d*\.*)(?:[^a-grA-GR#]*)([a-grA-GR]+)(--|-|n|##|#)?([^#]*)",
            token,
        )
        if not match:
            raise Exception(f"Invalid note {token}")

        dur, pitch, accidental, suffix = match[1], match[2], match[3], match[4]
        is_rest = pitch == "r"
        is_grace = "q" in token
        suffix = suffix.replace("q", "")

        rhythm_key = self.parse_duration(dur or default_dur, is_rest=is_rest, is_grace=is_grace)
        if is_rest:
            return EncodedSymbol(rhythm_key, empty, empty, empty, empty, position)

        lift_val = self._accidental_to_lift(accidental)
        pitch_val = self.kern_note_to_pitch(pitch)
        articulation_val, slur_val = self._articulation_from_suffix(suffix)
        return EncodedSymbol(rhythm_key, pitch_val, lift_val, articulation_val, slur_val, position)

    def parse_barline(self, line: str) -> list[EncodedSymbol]:
        symbol = line.split(" ", maxsplit=1)[0]
        # A fermata over the barline is not a barline shape of its own.
        symbol = symbol.rstrip(";")
        mapping = {
            "=:|!|:": ["repeatEndStart"],
            "=": ["barline"],
            "=-": [],  # barline after clef, key and time sig
            "==:|!": ["repeatEnd"],
            "==": ["bolddoublebarline"],
            "==!!": ["bolddoublebarline"],
            "=:|!": ["repeatEnd"],
            "=!|:": ["repeatStart"],
            "=:!!:": ["repeatEndStart"],
            "=||": ["doublebarline"],
            "=|!": ["barline"],
        }
        return [EncodedSymbol(s) for s in mapping[symbol]]

    def feed(self, voices: list[str]) -> None:
        """Process one kern line of this staff, with one entry in voices per voice."""
        s = self.state
        line = " ".join(voices)
        s.advance_line_no(line)
        if line.startswith("="):
            s.add_barline(self.parse_barline(line))
        elif line.startswith("*k"):
            s.set_key(self.parse_key_signature(line))
        elif line.startswith("*M"):
            s.set_time(self.parse_time_signature(line))
        elif line.startswith("*clef"):
            s.set_clef(self.parse_clef(line, s.position.value))
        elif line.startswith("*"):
            # All other control instructions can be ignored
            pass
        else:
            if line.strip():  # blank lines are formatting, not data
                s.data_seen = True
            for i, voice in enumerate(voices):
                # We accept at most 2 voices per staff.
                # Everything beyond 2nd is folded into 2nd.
                position = s.position.value
                if i >= 1:
                    position = s.position.value + "2"
                chord_dur = "4"
                first = True
                for token in voice.split():
                    if token == nonote:
                        continue
                    if first:
                        extracted = self._extract_dur(token)
                        if extracted:
                            chord_dur = extracted
                        first = False
                    s.add_note(self.parse_note_or_rest(token, position, chord_dur))


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
