import json
from collections import defaultdict

import torch

from homr.transformer.configs import default_config
from homr.transformer.vocabulary import (
    EncodedSymbol,
    Vocabulary,
)

vocab = Vocabulary()


def check_token_line(line: EncodedSymbol) -> None:
    if (
        line.rhythm not in vocab.rhythm
        or line.lift not in vocab.lift
        or line.articulation not in vocab.articulation
        or line.pitch not in vocab.pitch
        or line.position not in vocab.position
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
    return str.join("&", [str(c) for c in sorted_chord])


def sort_token_chords(
    symbols: list[EncodedSymbol], keep_chord_symbol: bool = False
) -> list[list[EncodedSymbol]]:
    chords: list[list[EncodedSymbol]] = []
    is_in_chord = False
    for symbol in symbols:
        if symbol.rhythm == "chord":
            is_in_chord = True
        elif is_in_chord and len(chords) > 0:
            if keep_chord_symbol:
                chords[-1].append(EncodedSymbol("chord"))
            chords[-1].append(symbol)
            is_in_chord = False
        else:
            chords.append([symbol])
    return chords


def calc_ratio_of_tuplets(symbols: list[EncodedSymbol]) -> float:
    tuplets = [s for s in symbols if s.is_tuplet()]
    return float(len(tuplets)) / len(symbols)


def token_lines_to_str(symbols: list[EncodedSymbol]) -> str:
    chords = sort_token_chords(symbols)

    chord_strings = [_chord_to_str(c) for c in chords]
    return str.join("\n", chord_strings)


def read_token_lines(lines: list[str]) -> list[EncodedSymbol]:
    result = []
    for line in lines:
        entries = line.split("&")
        for i, entry in enumerate(entries):
            if "tieSlur" in entry:
                continue
            parts = entry.strip().split()
            if len(parts) == 5:
                rhythm, pitch, lift, articulation, position = parts
            else:
                rhythm, pitch, lift, articulation = parts
                position = "upper"
            symbol = EncodedSymbol(rhythm, pitch, lift, articulation, position)
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
        states: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        self.rhythms = rhythms
        self.pitchs = pitchs
        self.lifts = lifts
        self.articulations = articulations
        self.positions = positions
        self.states = states
        self.mask = mask


def to_decoder_branches(symbols: list[EncodedSymbol]) -> DecoderBranches:
    nonote_token = default_config.nonote_token
    begin_of_seq = vocab.rhythm["BOS"]
    end_of_seq = vocab.rhythm["EOS"]
    rhythms = [begin_of_seq]
    pitchs = [nonote_token]
    lifts = [nonote_token]
    articulations = [nonote_token]
    position = [nonote_token]
    mask = [True]
    key = "keySignature_0"
    clef_upper = "clef_G2"
    clef_lower = "clef_F4"
    states = [vocab.state[f"{key}+{clef_upper}+{clef_lower}"]]
    for symbol in symbols:
        rhythms.append(vocab.rhythm[symbol.rhythm])
        pitchs.append(vocab.pitch[symbol.pitch])
        lifts.append(vocab.lift[symbol.lift])
        articulations.append(vocab.articulation[symbol.articulation])
        position.append(vocab.position[symbol.position])
        states.append(vocab.state[f"{key}+{clef_upper}+{clef_lower}"])
        mask.append(True)
        if symbol.rhythm.startswith("keySignature"):
            key = symbol.rhythm
        elif symbol.rhythm.startswith("clef"):
            if symbol.position == "upper":
                clef_upper = symbol.rhythm
            else:
                clef_lower = symbol.rhythm

    rhythms.append(end_of_seq)
    pitchs.append(nonote_token)
    lifts.append(nonote_token)
    articulations.append(nonote_token)
    position.append(nonote_token)
    states.append(vocab.state[f"{key}+{clef_upper}+{clef_lower}"])
    mask.append(True)

    while len(rhythms) < default_config.max_seq_len:
        rhythms.append(default_config.pad_token)
        pitchs.append(nonote_token)
        lifts.append(nonote_token)
        articulations.append(nonote_token)
        position.append(nonote_token)
        states.append(nonote_token)
        mask.append(False)

    return DecoderBranches(
        rhythms=torch.tensor(rhythms),
        lifts=torch.tensor(lifts),
        articulations=torch.tensor(articulations),
        pitchs=torch.tensor(pitchs),
        positions=torch.tensor(position),
        states=torch.tensor(states),
        mask=torch.tensor(mask),
    )


class VocabularyStats:
    def __init__(self) -> None:
        self.rhythm: dict[str, int] = defaultdict(int)
        self.lift: dict[str, int] = defaultdict(int)
        self.articulation: dict[str, int] = defaultdict(int)
        self.pitch: dict[str, int] = defaultdict(int)
        self.max_seq_len = 0

    def add_lines(self, lines: list[EncodedSymbol]) -> None:
        for line in lines:
            self.rhythm[line.rhythm] += 1
            self.lift[line.lift] += 1
            self.articulation[line.articulation] += 1
            self.pitch[line.pitch] += 1
        self.max_seq_len = max(self.max_seq_len, len(lines))

    def __str__(self) -> str:
        lines = [
            json.dumps(self.rhythm, indent=2, sort_keys=True),
            json.dumps(self.lift, indent=2, sort_keys=True),
            json.dumps(self.articulation, indent=2, sort_keys=True),
            json.dumps(self.pitch, indent=2, sort_keys=True),
            f"max_seq_len={self.max_seq_len}",
        ]
        return str.join("\n", lines)

    def __repr__(self) -> str:
        return str(self)


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
            check_token_lines(tokens)
            stats.add_lines(tokens)
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
