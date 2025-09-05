import json
from collections import defaultdict

import torch

from homr.transformer.configs import default_config
from homr.transformer.vocabulary import (
    SplitSymbol,
    Vocabulary,
    is_valid_combination,
    rhythm_to_category,
)

vocab = Vocabulary()


def check_token_line(line: SplitSymbol) -> None:
    if (
        line.rhythm not in vocab.rhythm
        or line.lift not in vocab.lift
        or line.articulation not in vocab.articulation
        or line.pitch not in vocab.pitch
    ):
        raise ValueError("Invalid symbol " + str(line))

    note_branch = rhythm_to_category(line.rhythm)
    if not is_valid_combination(line.rhythm, line.lift, line.articulation, line.pitch, note_branch):
        raise ValueError("Invalid combination " + str(line))


def check_token_lines(lines: list[SplitSymbol]) -> None:
    for line in lines:
        check_token_line(line)


def _split_symbol_to_sortable(symbol: SplitSymbol) -> int:
    if "note" in symbol.rhythm:
        return vocab.pitch[symbol.pitch] * len(vocab.rhythm) + vocab.rhythm[symbol.rhythm]
    if "rest" in symbol.rhythm:
        return 100000 + vocab.rhythm[symbol.rhythm]
    return 1000000


def _chord_to_str(chord: list[SplitSymbol]) -> str:
    sorted_chord = sorted(chord, key=_split_symbol_to_sortable)
    return str.join("&", [str(c) for c in sorted_chord])


def sort_token_chords(
    symbols: list[SplitSymbol], keep_chord_symbol: bool = False
) -> list[list[SplitSymbol]]:
    chords: list[list[SplitSymbol]] = []
    is_in_chord = False
    for symbol in symbols:
        if symbol.rhythm == "chord":
            is_in_chord = True
        elif is_in_chord:
            if keep_chord_symbol:
                chords[1].append(SplitSymbol("chord"))
            chords[-1].append(symbol)
            is_in_chord = False
        else:
            chords.append([symbol])
    return chords


def token_lines_to_str(symbols: list[SplitSymbol]) -> str:
    # TODO convert lifts, sort symbols in chords, but also e.g. slur vs cres start
    chords = sort_token_chords(symbols)

    chord_strings = [_chord_to_str(c) for c in chords]
    return str.join("\n", chord_strings)


def read_tokens(filepath: str) -> list[SplitSymbol]:
    result = []
    with open(filepath, encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            entries = line.split("&")
            for i, entry in enumerate(entries):
                if "tieSlur" in entry:
                    continue
                rhythm, pitch, lift, articulation = entry.split()
                symbol = SplitSymbol(rhythm, pitch, lift, articulation)
                is_first = i == 0
                if not is_first:
                    result.append(SplitSymbol("chord"))

                result.append(symbol)

    return result


class DecoderBranches:
    def __init__(
        self,
        rhythms: torch.Tensor,
        pitchs: torch.Tensor,
        lifts: torch.Tensor,
        articulations: torch.Tensor,
        notes: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        self.rhythms = rhythms
        self.pitchs = pitchs
        self.lifts = lifts
        self.articulations = articulations
        self.notes = notes
        self.mask = mask


def to_decoder_branches(symbols: list[SplitSymbol]) -> DecoderBranches:
    nonote_token = default_config.nonote_token
    begin_of_seq = vocab.rhythm["BOS"]
    end_of_seq = vocab.rhythm["EOS"]
    rhythms = [begin_of_seq]
    pitchs = [nonote_token]
    lifts = [nonote_token]
    articulations = [nonote_token]
    notes = [nonote_token]
    mask = [True]
    for symbol in symbols:
        rhythms.append(vocab.rhythm[symbol.rhythm])
        pitchs.append(vocab.pitch[symbol.pitch])
        lifts.append(vocab.lift[symbol.lift])
        articulations.append(vocab.articulation[symbol.articulation])
        notes.append(vocab.note[rhythm_to_category(symbol.rhythm)])
        mask.append(True)

    rhythms.append(end_of_seq)
    pitchs.append(nonote_token)
    lifts.append(nonote_token)
    articulations.append(nonote_token)
    notes.append(nonote_token)
    mask.append(True)

    while len(rhythms) < default_config.max_seq_len:
        rhythms.append(default_config.pad_token)
        pitchs.append(nonote_token)
        lifts.append(nonote_token)
        articulations.append(nonote_token)
        notes.append(nonote_token)
        mask.append(False)

    return DecoderBranches(
        rhythms=torch.tensor(rhythms),
        lifts=torch.tensor(lifts),
        articulations=torch.tensor(articulations),
        pitchs=torch.tensor(pitchs),
        notes=torch.tensor(notes),
        mask=torch.tensor(mask),
    )


class VocabularyStats:
    def __init__(self) -> None:
        self.rhythm: dict[str, int] = defaultdict(int)
        self.lift: dict[str, int] = defaultdict(int)
        self.articulation: dict[str, int] = defaultdict(int)
        self.pitch: dict[str, int] = defaultdict(int)
        self.max_seq_len = 0

    def add_lines(self, lines: list[SplitSymbol]) -> None:
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
        index_file = open(sys.argv[1])
        index_lines = index_file.readlines()
        index_file.close()
        files = [line.strip().split(",")[1] for line in index_lines]
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
