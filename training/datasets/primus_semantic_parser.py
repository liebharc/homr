import re

from homr.circle_of_fifths import key_signature_to_circle_of_fifth
from homr.transformer.vocabulary import (
    EncodedSymbol,
    empty,
    has_rhythm_symbol_a_position,
)
from training.transformer.training_vocabulary import VocabularyStats, check_token_lines

# --- Mapping helpers ---
duration_map = {
    "quadruple_whole": "1",  # TODO, should be 4m (4 measures)
    "double_whole": "1",  # TODO, should be 2m
    "whole": "1",
    "half": "2",
    "quarter": "4",
    "eighth": "8",
    "sixteenth": "16",
    "thirty_second": "32",
    "sixty_fourth": "64",
    "hundred_twenty_eighth": "128",
}


class PrimusConverter:
    @staticmethod
    def split_pitch_accidental(pitch_str: str) -> tuple[str, str]:
        """Split pitch string into base pitch and accidental."""
        m = re.match(r"([A-G])([b#N]*)(\d)", pitch_str)
        if not m:
            raise ValueError("Failed to parse " + pitch_str)
        note, accidental, octave = m.groups()
        base_pitch = f"{note}{octave}"
        lift = accidental if accidental else empty
        return base_pitch, lift

    @staticmethod
    def parse_duration(duration: str, is_grace_note: bool = False) -> str:
        dot_count = duration.count(".")
        dur_clean = duration.replace(".", "")
        grace = "G" if is_grace_note else ""
        if dur_clean not in duration_map:
            raise ValueError("Unknown duration " + dur_clean)
        return duration_map[dur_clean] + grace + "." * dot_count

    @staticmethod
    def get_articulations(duration: str) -> tuple[str, str]:
        if duration.endswith("fermata"):
            return duration.replace("_fermata", ""), "fermata"
        return duration, empty

    @staticmethod
    def parse_note(symbol: str) -> EncodedSymbol:
        base = symbol.split("note-")[-1]
        parts = base.split("_")
        pitch_part = parts[0]
        duration = str.join("_", parts[1:])
        duration, articulation = PrimusConverter.get_articulations(duration)
        base_pitch, lift = PrimusConverter.split_pitch_accidental(pitch_part)
        is_grace_note = symbol.startswith("grace")
        rhythm_key = PrimusConverter.parse_duration(duration, is_grace_note)
        rhythm_val = "note_" + rhythm_key

        return EncodedSymbol(rhythm_val, base_pitch, lift, articulation)

    @staticmethod
    def parse_rest(symbol: str) -> EncodedSymbol:
        duration = symbol.split("rest-")[1]
        duration, articulation = PrimusConverter.get_articulations(duration)
        rhythm_key = PrimusConverter.parse_duration(duration)
        rhythm_val = "rest_" + rhythm_key
        return EncodedSymbol(rhythm_val, empty, empty, articulation)

    @staticmethod
    def parse_multirest(symbol: str) -> EncodedSymbol:
        _, measures = symbol.split("-")
        num = min(int(measures), 10)
        if num == 1:
            # Return a whole rest instead
            return EncodedSymbol("rest_0", empty, empty, empty)
        return EncodedSymbol(f"rest_{num}m", empty, empty, empty)

    @staticmethod
    def parse_clef(symbol: str) -> EncodedSymbol:
        return EncodedSymbol(symbol.replace("-", "_"), empty, empty, empty)

    @staticmethod
    def parse_key_signature(symbol: str) -> EncodedSymbol:
        key = symbol.split("-")[1]
        circle = key_signature_to_circle_of_fifth(key)
        return EncodedSymbol(f"keySignature_{circle}")

    @staticmethod
    def parse_time_signature(symbol: str) -> EncodedSymbol:
        _, fraction = symbol.split("-")
        if fraction == "C":
            denom = "4"
        elif fraction == "C/":
            denom = "2"
        else:
            num, denom = fraction.split("/")
        return EncodedSymbol(f"timeSignature/{denom}")

    @staticmethod
    def parse_barline(symbol: str) -> EncodedSymbol:
        return EncodedSymbol("barline")

    @staticmethod
    def parse_tie(symbol: str) -> EncodedSymbol:
        return EncodedSymbol("tieSlur")

    @classmethod
    def convert_symbol(cls, symbol: str) -> EncodedSymbol:  # noqa: PLR0911
        if symbol.startswith(("note-", "gracenote-")):
            return cls.parse_note(symbol)
        elif symbol.startswith("rest-"):
            return cls.parse_rest(symbol)
        elif symbol.startswith("multirest-"):
            return cls.parse_multirest(symbol)
        elif symbol.startswith("clef-"):
            return cls.parse_clef(symbol)
        elif symbol.startswith("keySignature-"):
            return cls.parse_key_signature(symbol)
        elif symbol.startswith("timeSignature-"):
            return cls.parse_time_signature(symbol)
        elif symbol == "barline":
            return cls.parse_barline(symbol)
        elif symbol == "tie":
            return cls.parse_tie(symbol)
        else:
            return EncodedSymbol(symbol)


def convert_primus_semantic_to_tokens(semantic: str) -> list[EncodedSymbol]:
    symbols = re.split("\\s+", semantic.strip())
    tokens = [PrimusConverter.convert_symbol(sym) for sym in symbols]
    if tokens[-1].rhythm != "barline":
        tokens.append(EncodedSymbol("barline"))
    for symbol in tokens:
        if has_rhythm_symbol_a_position(symbol.rhythm):
            symbol.position = "upper"
    return tokens


if __name__ == "__main__":
    import glob
    import os

    from homr.simple_logging import eprint

    stats = VocabularyStats()
    for file in glob.glob(os.path.join("datasets", "Corpus", "**", "**.semantic"), recursive=True):
        with open(file, encoding="utf-8", errors="ignore") as f:
            first_line = f.readline().strip()
            tokens = convert_primus_semantic_to_tokens(first_line)
            check_token_lines(tokens)
            stats.add_lines(tokens)
    eprint("Stats", stats)
