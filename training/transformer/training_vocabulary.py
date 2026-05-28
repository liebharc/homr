import json
from collections import defaultdict

import torch

from homr.transformer.configs import default_config
from homr.transformer.vocabulary import EncodedSymbol, Vocabulary, sort_token_chords

vocab = Vocabulary()


def check_token_line(line: EncodedSymbol) -> None:
    """
    Validate that a single encoded symbol can be represented by the training vocabulary.

    The symbol is first checked against every vocabulary branch used by the decoder
    (rhythm, lift, articulation, pitch and staff position). It is then checked for a
    valid branch combination, for example notes/rests/clefs must carry note-specific
    fields while non-note symbols must use the non-note placeholders.

    Raises:
        ValueError: If any branch token is unknown, or if the token combination is
            not valid for the symbol rhythm.
    """
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
    """
    Validate every encoded symbol in a token sequence.

    Args:
        lines: Flat sequence of encoded symbols, including explicit ``chord`` marker
            symbols when present.

    Raises:
        ValueError: Propagated from ``check_token_line`` for the first invalid symbol.
    """
    for line in lines:
        check_token_line(line)


def _symbol_to_sortable(symbol: EncodedSymbol) -> int:
    """
    Build a stable numeric sort key for symbols inside one chord.

    Notes are ordered by pitch and rhythm, rests are placed after notes, and other
    symbols are placed after rests. Lower-staff symbols receive a large offset so
    that upper-staff and lower-staff material stays grouped consistently.

    Args:
        symbol: The symbol to rank within a chord.

    Returns:
        Integer sort key used by ``_chord_to_str``.
    """
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
    """
    Calculate how much of a token sequence uses tuplet-derived durations.

    Args:
        symbols: Flat sequence of encoded symbols to inspect.

    Returns:
        Number of tuplet symbols divided by the total number of symbols.
    """
    tuplets = [s for s in symbols if s.is_tuplet()]
    return float(len(tuplets)) / len(symbols)


def token_lines_to_str(symbols: list[EncodedSymbol]) -> str:
    """
    Serialize a flat token sequence into the line-based ``.tokens`` format.

    Explicit ``chord`` marker symbols are first grouped by ``sort_token_chords``.
    Each resulting chord or single symbol is written on its own line, with chord
    members joined by ``&``.

    Args:
        symbols: Flat sequence of encoded symbols.

    Returns:
        Text suitable for writing to a ``.tokens`` file.
    """
    chords = sort_token_chords(symbols)
    print(f"Chords: {chords}")
    chord_strings = [_chord_to_str(c) for c in chords]
    print(f"Result: {chord_strings}")
    return str.join("\n", chord_strings)


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
    """
    Read and parse a ``.tokens`` file.

    Args:
        filepath: Path to the token file.

    Returns:
        Flat sequence of encoded symbols parsed by ``read_token_lines``.
    """
    with open(filepath, encoding="utf-8") as f:
        lines = f.readlines()
        return read_token_lines(lines)


class DecoderBranches:
    """
    Container for the decoder target tensors used by the training data loader.

    The transformer predicts several categorical branches for each output step.
    This object keeps the padded rhythm, pitch, lift, articulation and position
    tensors together with a boolean mask identifying real sequence positions.
    """

    def __init__(
        self,
        rhythms: torch.Tensor,
        pitchs: torch.Tensor,
        lifts: torch.Tensor,
        articulations: torch.Tensor,
        positions: torch.Tensor,
        slur: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        """
        Store precomputed decoder branch tensors.

        Args:
            rhythms: Rhythm token ids, including BOS/EOS/PAD positions.
            pitchs: Pitch token ids aligned with ``rhythms``.
            lifts: Accidental/lift token ids aligned with ``rhythms``.
            articulations: Articulation token ids aligned with ``rhythms``.
            positions: Staff-position token ids aligned with ``rhythms``.
            mask: Boolean tensor that is true for BOS, real tokens and EOS, and
                false for padding.
        """
        self.rhythms = rhythms
        self.pitchs = pitchs
        self.lifts = lifts
        self.articulations = articulations
        self.positions = positions
        self.slur = slur
        self.mask = mask


def to_decoder_branches(symbols: list[EncodedSymbol]) -> DecoderBranches:
    """
    Convert encoded symbols into fixed-length tensors for decoder supervision.

    The returned tensors start with a BOS rhythm token and end with an EOS rhythm
    token. Non-rhythm branches use the configured non-note token at BOS, EOS and
    padding positions. All branches are padded to ``default_config.max_seq_len``;
    the mask marks BOS, real symbols and EOS as valid and padding as invalid.

    Args:
        symbols: Flat sequence of encoded symbols to encode.

    Returns:
        ``DecoderBranches`` containing one tensor per decoder branch plus the mask.
    """
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
    """
    Accumulate simple corpus statistics for encoded training tokens.

    Counts are tracked independently for the decoder branches used in this module,
    and ``max_seq_len`` records the longest flat token sequence seen.
    """

    def __init__(self) -> None:
        """
        Initialize empty counters for all tracked token branches.
        """
        self.rhythm: dict[str, int] = defaultdict(int)
        self.lift: dict[str, int] = defaultdict(int)
        self.articulation: dict[str, int] = defaultdict(int)
        self.pitch: dict[str, int] = defaultdict(int)
        self.slur: dict[str, int] = defaultdict(int)
        self.max_seq_len = 0

    def add_lines(self, lines: list[EncodedSymbol]) -> None:
        """
        Add one parsed token sequence to the accumulated statistics.

        Args:
            lines: Flat sequence of encoded symbols from one training example.
        """
        for line in lines:
            self.rhythm[line.rhythm] += 1
            self.lift[line.lift] += 1
            self.articulation[line.articulation] += 1
            self.slur[line.slur] += 1
            self.pitch[line.pitch] += 1
        self.max_seq_len = max(self.max_seq_len, len(lines))

    def __str__(self) -> str:
        """
        Format the collected statistics as sorted, human-readable JSON blocks.

        Returns:
            Multi-line string containing branch counts and the maximum sequence
            length observed so far.
        """
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
        """
        Return the same representation as ``__str__`` for interactive inspection.
        """
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
