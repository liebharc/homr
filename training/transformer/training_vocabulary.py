import json
import re
from collections import defaultdict

import torch

from homr.transformer.configs import default_config
from homr.transformer.vocabulary import (
    EncodedSymbol,
    Vocabulary,
    empty,
    has_rhythm_symbol_a_position,
    nonote,
    sort_token_chords,
)

vocab = Vocabulary()

NOTE_TO_DIATONIC = {
    "C": 0,
    "D": 1,
    "E": 2,
    "F": 3,
    "G": 4,
    "A": 5,
    "B": 6,
}

CLEF_ANCHORS = {
    "clef_G1": "D5",
    "clef_G2": "B4",
    "clef_F3": "F3",
    "clef_F4": "D3",
    "clef_F5": "B2",
    "clef_C1": "G4",
    "clef_C2": "E4",
    "clef_C3": "C4",
    "clef_C4": "A3",
    "clef_C5": "F3",
}


def check_token_line(line: EncodedSymbol) -> None:
    if (
        line.rhythm not in vocab.rhythm
        or line.lift not in vocab.lift
        or line.articulation not in vocab.articulation
        or line.pitch not in vocab.pitch
        or line.position not in vocab.position
        or line.slur not in vocab.slur
    ):
        raise ValueError("Invalid symbol " + str(line))

    if not line.is_valid():
        raise ValueError("Invalid combination " + str(line))


def check_token_lines(lines: list[EncodedSymbol]) -> None:
    for line in lines:
        check_token_line(line)


def _symbol_to_sortable(symbol: EncodedSymbol) -> int:
    position = 10000000 if symbol.position == "lower" else 0
    if "note" in symbol.rhythm:
        return (
            vocab.pitch[symbol.pitch] * len(vocab.rhythm) + vocab.rhythm[symbol.rhythm] + position
        )
    if "rest" in symbol.rhythm:
        return 100000 + vocab.rhythm[symbol.rhythm] + position
    return 1000000 + position



def _chord_to_str(chord: list[EncodedSymbol]) -> str:
    sorted_chord = sorted(chord, key=_symbol_to_sortable)
    upper_slurs = set()
    lower_slurs = set()
    upper_artics = set()
    lower_artics = set()

    annotation_resorted: list[EncodedSymbol] = []
    for symbol in sorted_chord:
        artic_stripped, symbol_stripped = symbol.strip_articulations([], remove_all=True)
        slur_stripped, symbol_stripped = symbol_stripped.strip_slurs([], remove_all=True)
        for articulation in artic_stripped:
            if symbol.position == "lower":
                lower_artics.add(articulation)
            else:
                upper_artics.add(articulation)
        for slur in slur_stripped:
            if symbol.position == "lower":
                lower_slurs.add(slur)
            else:
                upper_slurs.add(slur)

        annotation_resorted.append(symbol_stripped)
    
    def _remove_item_helper(s: set, input_item: str):
        if len(s) > 1:
            s.discard(input_item)
        return s

    upper_artics = _remove_item_helper(upper_artics, ".")
    lower_artics =  _remove_item_helper(lower_artics, ".")
    upper_slurs = _remove_item_helper(upper_slurs,".")
    lower_slurs = _remove_item_helper(lower_slurs, ".")

    if len(upper_slurs) > 0:
        first_upper = next(
            (idx for idx, s in enumerate(annotation_resorted) if s.position != "lower"), None
        )
        if first_upper is not None:
            annotation_resorted[first_upper] = annotation_resorted[first_upper].add_slurs(
                list(upper_slurs)
            )
    if len(lower_slurs) > 0:
        first_lower = next(
            (idx for idx, s in enumerate(annotation_resorted) if s.position == "lower"), None
        )
        if first_lower is not None:
            annotation_resorted[first_lower] = annotation_resorted[first_lower].add_slurs(
                list(lower_slurs)
            )

    if len(upper_artics) > 0:
        first_upper = next(
            (idx for idx, s in enumerate(annotation_resorted) if s.position != "lower"), None
        )
        if first_upper is not None:
            annotation_resorted[first_upper] = annotation_resorted[first_upper].add_articulations(
                list(upper_artics)
            )
    if len(lower_artics) > 0:
        first_lower = next(
            (idx for idx, s in enumerate(annotation_resorted) if s.position == "lower"), None
        )
        if first_lower is not None:
            annotation_resorted[first_lower] = annotation_resorted[first_lower].add_articulations(
                list(lower_artics)
            )
    return str.join("&", [str(c) for c in annotation_resorted])


def calc_ratio_of_tuplets(symbols: list[EncodedSymbol]) -> float:
    tuplets = [s for s in symbols if s.is_tuplet()]
    return float(len(tuplets)) / len(symbols)


def token_lines_to_str(symbols: list[EncodedSymbol]) -> str:
    chords = sort_token_chords(symbols)
    chord_strings = [_chord_to_str(c) for c in chords]
    return str.join("\n", chord_strings)


def _pitch_to_diatonic(pitch: str) -> int | None:
    match = re.match(r"^([A-G])[#bN]*([0-9])$", pitch)
    if not match:
        return None
    note, octave = match.groups()
    return int(octave) * 7 + NOTE_TO_DIATONIC[note]


def _clef_anchor(clef: str) -> int | None:
    if clef not in CLEF_ANCHORS:
        return None
    return _pitch_to_diatonic(CLEF_ANCHORS[clef])


def max_ledger_lines(tokens: list[EncodedSymbol]) -> int:
    anchors = {
        "upper": _clef_anchor("clef_G2"),
        "lower": _clef_anchor("clef_F4"),
    }
    max_ledger_lines = 0
    for symbol in tokens:
        if symbol.rhythm.startswith("clef"):
            anchor = _clef_anchor(symbol.rhythm)
            if anchor is None:
                continue
            if symbol.position in anchors:
                anchors[symbol.position] = anchor
            else:
                anchors["upper"] = anchor
                anchors["lower"] = anchor
            continue

        if not symbol.rhythm.startswith("note"):
            continue

        if symbol.pitch in (nonote, empty):
            continue

        absolute_val = _pitch_to_diatonic(symbol.pitch)
        if absolute_val is None:
            continue

        anchor = anchors.get(symbol.position, anchors["upper"])
        if anchor is None:
            raise ValueError("Failed to get anchor")
        offset = absolute_val - anchor
        ledger_lines = max(0, (abs(offset) - 4 + 1) // 2)
        max_ledger_lines = max(max_ledger_lines, ledger_lines)

    return max_ledger_lines


def read_token_lines(lines: list[str]) -> list[EncodedSymbol]:
    """
    Parse lines from a ``.tokens`` file into a flat encoded-symbol sequence.

    Each input line represents one musical time position. Chord members are separated
    by ``&``; all members after the first are preceded in the returned sequence by an
    explicit ``EncodedSymbol("chord")`` marker. Legacy four-field entries default to
    the upper staff position, while five-field entries include their position. Entries
    containing ``tieSlur`` are ignored.

    Args:
        lines: Raw lines read from a token file.

    Returns:
        Flat sequence of encoded symbols ready for validation or tensor conversion.
    """
    result = []
    for line in lines:
        entries = line.split("&")
        for i, entry in enumerate(entries):
            if "tieSlur" in entry:
                continue
            parts = entry.strip().split()
            if len(parts) == 6:
                rhythm, pitch, lift, articulation, slur, position = parts
            else:
                rhythm, pitch, lift, articulation, slur = parts
                position = "upper"
            symbol = EncodedSymbol(rhythm, pitch, lift, articulation, slur, position)
            is_first = i == 0
            if not is_first:
                result.append(EncodedSymbol("chord"))

            result.append(symbol)

    return result


def read_tokens(filepath: str) -> list[EncodedSymbol]:
    with open(filepath, encoding="utf-8") as f:
        lines = f.readlines()
        return read_token_lines(lines)


class DecoderBranches:
    def __init__(
        self,
        rhythms: torch.Tensor,
        pitchs: torch.Tensor,
        lifts: torch.Tensor,
        articulations: torch.Tensor,
        positions: torch.Tensor,
        slurs: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        self.rhythms = rhythms
        self.pitchs = pitchs
        self.lifts = lifts
        self.articulations = articulations
        self.positions = positions
        self.slurs = slurs
        self.mask = mask


def to_decoder_branches(symbols: list[EncodedSymbol]) -> DecoderBranches:
    nonote_token = default_config.nonote_token
    begin_of_seq = vocab.rhythm["BOS"]
    end_of_seq = vocab.rhythm["EOS"]
    rhythms = [begin_of_seq]
    pitchs = [nonote_token]
    lifts = [nonote_token]
    articulations = [nonote_token]
    slurs = [nonote_token]
    position = [nonote_token]
    mask = [True]
    for symbol in symbols:
        rhythms.append(vocab.rhythm[symbol.rhythm])
        pitchs.append(vocab.pitch[symbol.pitch])
        lifts.append(vocab.lift[symbol.lift])
        articulations.append(vocab.articulation[symbol.articulation])
        position.append(vocab.position[symbol.position])
        slurs.append(vocab.slur[symbol.slur])
        mask.append(True)

    rhythms.append(end_of_seq)
    pitchs.append(nonote_token)
    lifts.append(nonote_token)
    articulations.append(nonote_token)
    position.append(nonote_token)
    slurs.append(nonote_token)
    mask.append(True)

    while len(rhythms) < default_config.max_seq_len:
        rhythms.append(default_config.pad_token)
        pitchs.append(nonote_token)
        lifts.append(nonote_token)
        articulations.append(nonote_token)
        position.append(nonote_token)
        slurs.append(nonote_token)
        mask.append(False)

    return DecoderBranches(
        rhythms=torch.tensor(rhythms),
        lifts=torch.tensor(lifts),
        articulations=torch.tensor(articulations),
        pitchs=torch.tensor(pitchs),
        positions=torch.tensor(position),
        slurs=torch.tensor(slurs),
        mask=torch.tensor(mask),
    )


class VocabularyStats:
    def __init__(self) -> None:
        self.rhythm: dict[str, int] = defaultdict(int)
        self.lift: dict[str, int] = defaultdict(int)
        self.articulation: dict[str, int] = defaultdict(int)
        self.pitch: dict[str, int] = defaultdict(int)
        self.slur: dict[str, int] = defaultdict(int)
        self.max_seq_len = 0

    def add_lines(self, lines: list[EncodedSymbol]) -> None:
        for line in lines:
            self.rhythm[line.rhythm] += 1
            self.lift[line.lift] += 1
            self.articulation[line.articulation] += 1
            self.slur[line.slur] += 1
            self.pitch[line.pitch] += 1
        self.max_seq_len = max(self.max_seq_len, len(lines))

    def __str__(self) -> str:
        lines = [
            json.dumps(self.rhythm, indent=2, sort_keys=True),
            json.dumps(self.lift, indent=2, sort_keys=True),
            json.dumps(self.articulation, indent=2, sort_keys=True),
            json.dumps(self.slur, indent=2, sort_keys=True),
            json.dumps(self.pitch, indent=2, sort_keys=True),
            f"max_seq_len={self.max_seq_len}",
        ]
        return str.join("\n", lines)

    def __repr__(self) -> str:
        return str(self)


def map_rhythm_symbol_with_position() -> torch.Tensor:
    rhythm_vocab = vocab.rhythm
    rhythm_map = torch.zeros(len(rhythm_vocab), dtype=torch.bool)
    for rhythm, idx in rhythm_vocab.items():
        if has_rhythm_symbol_a_position(rhythm):
            rhythm_map[idx] = True
    return rhythm_map


if __name__ == "__main__":
    import glob
    import os
    import sys

    from homr.simple_logging import eprint

    stats = VocabularyStats()
    errors: set[str] = set()
    i = 0
    if len(sys.argv) > 1:
        files = []
        for index_path in sys.argv[1:]:
            index_file = open(index_path)
            index_lines = index_file.readlines()
            index_file.close()
            files.extend([line.strip().split(",")[1] for line in index_lines])
    else:
        files = glob.glob(os.path.join("datasets", "**", "**.tokens"), recursive=True)

    for file in files:
        try:
            tokens = read_tokens(file)
            stats.add_lines(tokens)
            check_token_lines(tokens)
            i += 1
            if i % 1000 == 0:
                eprint(i, len(errors))
        except Exception as e:
            eprint("======", file, "======")
            eprint(e)
            errors.add(file)

    eprint("Stats", stats)
    if len(errors) > 0:
        eprint("errors", errors)
