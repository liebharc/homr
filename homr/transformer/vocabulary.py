import copy
import itertools
import random
import re
from fractions import Fraction
from typing import Iterable

from homr.simple_logging import eprint

nonote = "."
empty = "_"  # used for decorations on note, if there is no decoration


def build_dict(tokens: Iterable[str]) -> dict[str, int]:
    result = {}
    for i, t in enumerate(tokens):
        if t in result:
            raise ValueError("Duplicated entry for " + t)
        if not t.strip():
            raise ValueError("Tokens must not be a whitespace, this makes parsing easier")
        if t.strip() != t:
            raise ValueError("Tokens must not contain a whitespace, this makes parsing easier")
        if "&" in t:
            raise ValueError("& is reserved as alternatives for chords")
        result[t] = i
    return result


def build_rhythm() -> dict[str, int]:
    rhythm = []

    # sequence symbols
    rhythm.extend(["PAD", "BOS", "EOS"])
    rhythm.append("chord")

    # bar lines
    rhythm.extend(["barline", "doublebarline", "bolddoublebarline"])
    rhythm.extend(
        ["repeatStart", "repeatEnd", "repeatEndStart"]
    )  # , "daCapo", "daSegno", "segno", "coda"
    rhythm.extend(["voltaStart", "voltaStop", "voltaDiscontinue"])

    # clefs
    rhythm.extend([f"clef_F{c}" for c in range(3, 6)])
    rhythm.extend([f"clef_C{c}" for c in range(1, 6)])
    rhythm.extend([f"clef_G{c}" for c in range(1, 3)])

    # signatures
    rhythm.extend([f"keySignature_{c}" for c in range(-7, 8)])
    rhythm.extend([f"timeSignature/{c}" for c in [1, 2, 3, 4, 6, 8, 12, 16, 32, 48]])

    # rhythm, kern durations are based on https://www.humdrum.org/rep/kern/
    rhythm.extend([f"rest_{c}m" for c in range(2, 11)])  # multirests
    kern_base_durations = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 32, 64, 128]  # 128, 256, 512
    dots = ["", ".", ".."]
    grace = ["", "G"]
    kern_values = [
        f"{d}{g}{dot}" for d, g, dot in itertools.product(kern_base_durations, grace, dots)
    ]

    # Durations coming from tuplets
    irregular_durations = [7, 11, 13, 18, 20, 21, 22, 24, 26, 28, 30, 34, 36, 40, 48, 56, 96]

    rhythm.extend([f"note_{d}" for d in kern_values])
    rhythm.extend([f"note_{d}" for d in irregular_durations])
    rhythm.extend([f"rest_{d}" for d in kern_values])
    rhythm.extend([f"rest_{d}" for d in irregular_durations])

    # Dynamics
    # rhythm.extend(
    #    [f"dynamic_{d}" for d in ["ppp", "pp", "p", "mp", "mf", "f", "ff", "fff", "sfz", "fp"]]
    # )
    # rhythm.extend(
    #    [
    #        "crescendoStart",
    #        "crescendoEnd",
    #        "diminuendoStart",
    #        "diminuendoEnd",
    #    ]
    # )

    return build_dict(rhythm)


def build_lift() -> dict[str, int]:
    lifts = [nonote, empty, "#", "##", "N", "b", "bb"]
    return build_dict(lifts)


def build_position() -> dict[str, int]:
    """
    The staff position, applies to notes, rests and clefs
    """
    positions = [nonote, "upper", "lower"]
    return build_dict(positions)


def build_articulation() -> dict[str, int]:
    articulation = [nonote, empty]

    # The lieder dataset has the superset of articulations
    # so we use it to define this list
    articulations_lieder = [
        "accent",
        "accent_arpeggiate",
        "accent_arpeggiate_fermata",
        "accent_arpeggiate_staccato",
        "accent_arpeggiate_tenuto",
        "accent_breathMark",
        "accent_breathMark_fermata",
        "accent_fermata",
        "accent_fermata_staccato",
        "accent_staccatissimo",
        "accent_staccato",
        "accent_staccato_tenuto",
        "accent_tenuto",
        "accent_tremolo",
        "accent_trill",
        "accent_fermata_trill",
        "arpeggiate",
        "arpeggiate_breathMark_fermata",
        "arpeggiate_fermata",
        "arpeggiate_fermata_staccato",
        "arpeggiate_staccatissimo",
        "arpeggiate_staccato",
        "arpeggiate_staccato_tenuto",
        "arpeggiate_tenuto",
        "arpeggiate_tremolo",
        "arpeggiate_trill",
        "breathMark",
        "breathMark_fermata",
        "breathMark_fermata_tenuto",
        "breathMark_staccato",
        "breathMark_tenuto",
        "breathMark_trill",
        "breathMark_tremolo",
        "breathMark_staccato_tenuto",
        "fermata",
        "fermata_staccato",
        "fermata_staccato_tenuto",
        "fermata_tenuto",
        "fermata_tremolo",
        "fermata_trill",
        "fermata_turn",
        "spiccato",
        "staccatissimo",
        "staccato",
        "staccato_tenuto",
        "staccato_tremolo",
        "staccato_trill",
        "staccato_turn",
        "tenuto",
        "tremolo",
        "trill",
        "turn",
    ]

    articulation.extend(articulations_lieder)

    return build_dict(articulation)


def build_slur() -> dict[str, int]:
    slur = [nonote, empty]
    slur.extend(["slurStart_slurStop", "slurStart", "slurStop"])
    return build_dict(slur)


def build_pitch() -> dict[str, int]:
    pitch = [nonote, empty]
    note_names = ["C", "D", "E", "F", "G", "A", "B"]
    octave = range(10)
    pitch.extend(reversed([f"{n}{octave}" for octave, n in itertools.product(octave, note_names)]))
    return build_dict(pitch)


def has_rhythm_symbol_a_position(rhythm: str) -> bool:
    return rhythm.startswith(("note", "rest", "clef"))


class Vocabulary:
    def __init__(self) -> None:
        self.rhythm = build_rhythm()
        self.lift = build_lift()
        self.articulation = build_articulation()
        self.pitch = build_pitch()
        self.slur = build_slur()
        self.position = build_position()


class SymbolDuration:
    def __init__(
        self, base_duration: Fraction, dots: int, actual_notes: int, normal_notes: int, kern: int
    ) -> None:
        self.base_duration = base_duration
        self.dots = dots
        action_normal = Fraction(actual_notes, normal_notes)
        self.actual_notes = action_normal.numerator
        self.normal_notes = action_normal.denominator
        self.fraction = self._to_fraction()
        self.kern = kern

    def _to_fraction(self) -> Fraction:
        """
        Convert to a final Fraction duration (relative to whole note).
        """
        dur = self.base_duration

        # Apply dots
        add = dur / 2
        for _ in range(self.dots):
            dur += add
            add /= 2

        # Apply tuplet scaling
        if self.actual_notes != self.normal_notes:
            dur *= Fraction(self.normal_notes, self.actual_notes)

        return dur


def prior_power_of_two(n: int) -> int:
    """Return the next power of two <= n."""
    if n < 1:
        # This produces wrong rhythms, but at least it produces something
        return 1
    return 1 << (n.bit_length() - 1)


def kern_to_symbol_duration(kern: str) -> SymbolDuration:
    """
    Parse a Humdrum **kern duration string into a SymbolDuration object.
    """
    if kern.endswith("m"):
        # Multirest
        SymbolDuration(Fraction(1), 0, 1, 1, 4)

    # Extract numeric prefix (can be > 1 digit)
    i = 0
    while i < len(kern) and kern[i].isdigit():
        i += 1
    base_str = kern[:i]
    rest = kern[i:]

    base = int(base_str) if base_str else 4  # default quarter
    dots = rest.count(".")

    if "G" in kern:
        # Grace note
        return SymbolDuration(Fraction(0), dots, 1, 1, base)
    if base == 0:
        # Special: Whoe measure rest
        return SymbolDuration(Fraction(1), dots, 1, 1, base)

    # If base is a power of two, it's a normal note
    if base & (base - 1) == 0:
        base_duration = Fraction(1, base)
        actual_notes, normal_notes = 1, 1
        return SymbolDuration(base_duration, dots, actual_notes, normal_notes, base)
    else:
        # Tuplet case: find next higher power of two
        normal_notes = prior_power_of_two(base)
        base_duration = Fraction(1, normal_notes)
        actual_notes = base
        return SymbolDuration(base_duration, dots, actual_notes, normal_notes, normal_notes)


class EncodedSymbol:
    """
    A musical symbol split into the different decoder branches.
    """

    def __init__(
        self,
        rhythm: str,
        pitch: str = nonote,
        lift: str = nonote,
        articulation: str = nonote,
        slur: str = nonote,
        position: str = nonote,
        coordinates: tuple[float, float] | None = None,
    ) -> None:
        self.rhythm = rhythm
        self.pitch = pitch
        self.lift = lift
        self.articulation = articulation
        self.slur = slur
        self.position = position

        # These coordinates are derived from transformer attention and are inherently imprecise,
        # since the model is optimized for predictive accuracy rather than spatial localization.
        # Because patch tokens are processed in raster order (top-to-bottom, left-to-right),
        # this ordering can be used to reject cases where attention-based coordinates
        # violate monotonic scan constraints and are therefore unreliable.
        self.coordinates = coordinates
        self._duration: SymbolDuration | None = None

    def is_control_symbol(self) -> bool:
        return self.rhythm in ("BOS", "EOS", "PAD")

    def is_tuplet(self) -> bool:
        no_tuplet = self.remove_tuplet()
        return no_tuplet.rhythm != self.rhythm

    def remove_tuplet(self) -> "EncodedSymbol":
        match = re.match(r"(note|rest)_(\d+)(.*)", self.rhythm)
        if not match:
            return self
        duration = int(match[2])
        if duration % 3 == 0:
            duration = duration // 3 * 2
        elif duration % 5 == 0:
            duration = duration // 5 * 4
        elif duration % 7 == 0:
            duration = duration // 7 * 4
        else:
            return self

        result = copy.copy(self)
        result.rhythm = match[1] + "_" + str(duration) + match[3]
        result._duration = None
        return result

    def change_lift(self, lift: str) -> "EncodedSymbol":
        result = copy.copy(self)
        result.lift = lift
        return result

    def to_upper_position(self) -> "EncodedSymbol":
        if self.position != "lower":
            return self
        result = copy.copy(self)
        result.position = "upper"
        return result

    def is_valid(self) -> bool:
        has_position = has_rhythm_symbol_a_position(self.rhythm)
        is_note = [
            s != nonote
            for s in [self.lift, self.articulation, self.pitch, self.slur, self.position]
        ]
        return all(item == has_position for item in is_note)

    def add_articulations(self, articulations: list[str]) -> "EncodedSymbol":
        all_articulations = []
        all_articulations.extend(articulations)
        all_articulations.extend([a for a in self.articulation.split("_") if a])
        result = copy.copy(self)
        result.articulation = str.join("_", sorted(all_articulations))
        return result

    def add_slurs(self, slurs: list[str]) -> "EncodedSymbol":
        all_slurs = []
        all_slurs.extend(slurs)
        all_slurs.extend([a for a in self.slur.split("_") if a])
        result = copy.copy(self)
        result.slur = str.join("_", sorted(all_slurs))
        return result

    def strip_articulations(
        self, to_be_removed: list[str], remove_all: bool = False
    ) -> tuple[list[str], "EncodedSymbol"]:
        if self.articulation == nonote and not has_rhythm_symbol_a_position(self.rhythm):
            return [], copy.copy(self)
        stripped = []
        remaining = []
        for articulation in self.articulation.split("_"):
            if not articulation:
                continue
            if remove_all or articulation in to_be_removed:
                stripped.append(articulation)
            else:
                remaining.append(articulation)
        result = copy.copy(self)
        if remaining:
            result.articulation = str.join("_", remaining)
        else:
            result.articulation = empty

        return stripped, result

    def strip_slurs(
        self, to_be_removed: list[str], remove_all: bool = False
    ) -> tuple[list[str], "EncodedSymbol"]:
        if self.slur == nonote and not has_rhythm_symbol_a_position(self.rhythm):
            return [], copy.copy(self)
        stripped = []
        remaining = []
        for slur in self.slur.split("_"):
            if not slur:
                continue
            if remove_all or slur in to_be_removed:
                stripped.append(slur)
            else:
                remaining.append(slur)
        result = copy.copy(self)
        if remaining:
            result.slur = str.join("_", remaining)
        else:
            result.slur = empty

        return stripped, result

    def get_duration(self) -> SymbolDuration:
        """
        Convert a Humdrum **kern duration into a Fraction (relative to a whole note).
        """
        if self._duration is not None:
            return self._duration

        rhythm = self.rhythm
        if not rhythm.startswith(("note", "rest")):
            eprint("Warning, invalid symbol in group: Only notes and rests have durations")
            return SymbolDuration(Fraction(0), 0, 1, 1, 1)
        kern = rhythm.split("_")[1]

        duration = kern_to_symbol_duration(kern)
        self._duration = duration
        return duration

    def __str__(self) -> str:
        return str.join(
            " ", (self.rhythm, self.pitch, self.lift, self.articulation, self.slur, self.position)
        )

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, EncodedSymbol):
            return (
                self.rhythm == __value.rhythm
                and self.pitch == __value.pitch
                and self.lift == __value.lift
                and self.articulation == __value.articulation
                and self.slur == __value.slur
                and self.position == __value.position
            )
        else:
            return False

    def __hash__(self) -> int:
        return hash(
            (self.rhythm, self.pitch, self.lift, self.articulation, self.slur, self.position)
        )

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, EncodedSymbol):
            return NotImplemented
        return str(self) > str(other)


def _remove_redudant_clefs_keys_and_time_signatures(
    chords: list[list[EncodedSymbol]],
) -> list[list[EncodedSymbol]]:
    clef_upper = ""
    clef_lower = ""
    key = ""
    time = ""
    result_chords = []
    for chord in chords:
        result = []
        for symbol in chord:
            if symbol.rhythm.startswith("clef"):
                if symbol.position == "upper":
                    if symbol.rhythm != clef_upper:
                        clef_upper = symbol.rhythm
                        result.append(symbol)
                elif symbol.rhythm != clef_lower:
                    clef_lower = symbol.rhythm
                    result.append(symbol)
            elif symbol.rhythm.startswith("keySignature"):
                if symbol.rhythm != key:
                    key = symbol.rhythm
                    result.append(symbol)
            elif symbol.rhythm.startswith("timeSignature"):
                if symbol.rhythm != time:
                    time = symbol.rhythm
                    result.append(symbol)
            else:
                result.append(symbol)
        result_chords.append(result)
    return result_chords


def _remove_duplicated_piches(chord: list[EncodedSymbol]) -> list[EncodedSymbol]:
    if len(chord) <= 1 or not chord[0].rhythm.startswith(("note", "rest")):
        return chord
    by_pitch: dict[str, EncodedSymbol] = {}
    order_of_appearance = []
    for symbol in chord:
        key = symbol.pitch + " " + symbol.position
        if key in by_pitch:
            if symbol.get_duration().fraction > by_pitch[key].get_duration().fraction:
                by_pitch[symbol.pitch] = symbol
        else:
            by_pitch[key] = symbol
            order_of_appearance.append(key)

    return [by_pitch[s] for s in order_of_appearance]


def _group_into_chords(symbols: list[EncodedSymbol]) -> list[list[EncodedSymbol]]:
    chords: list[list[EncodedSymbol]] = []
    is_in_chord = False
    for symbol in symbols:
        if symbol.rhythm == "chord":
            is_in_chord = True
        elif is_in_chord and len(chords) > 0:
            chords[-1].append(symbol)
            is_in_chord = False
        else:
            chords.append([symbol])
    return chords


def _flatten_chords(chords: list[list[EncodedSymbol]]) -> list[EncodedSymbol]:
    result = []
    for chord in chords:
        if len(chords) == 0:
            continue
        is_in_chord = False
        for symbol in chord:
            if is_in_chord:
                result.append(EncodedSymbol("chord"))
            result.append(symbol)
            is_in_chord = True

    return result


def _group_into_measures(chords: list[list[EncodedSymbol]]) -> list[list[list[EncodedSymbol]]]:
    measures = []
    current_measure = []
    for chord in chords:
        current_measure.append(chord)
        if len(chord) > 0 and ("barline" in chord[0].rhythm or "repeat" in chord[0].rhythm):
            measures.append(current_measure)
            current_measure = []
    if len(current_measure) > 0:
        measures.append(current_measure)
    return measures


def _flatten_measures(measures: list[list[list[EncodedSymbol]]]) -> list[list[EncodedSymbol]]:
    return [chord for measure in measures for chord in measure]


def _get_duration_of_measure(measure: list[list[EncodedSymbol]]) -> Fraction:
    total_duration = Fraction(0)
    for chord in measure:
        duration = Fraction(0)
        for symbol in chord:
            if symbol.rhythm.startswith(("note", "rest")):
                fraction = symbol.get_duration().fraction
                if fraction > Fraction(0) and (fraction < duration or duration == Fraction(0)):
                    duration = fraction
        total_duration += duration
    return total_duration


def _get_typical_duration_of_measures(measures: list[list[list[EncodedSymbol]]]) -> Fraction:
    durations = [_get_duration_of_measure(m) for m in measures]
    if len(durations) == 0:
        return Fraction(0)
    return sorted(durations)[len(durations) // 2]


def _remove_tuplets(measure: list[list[EncodedSymbol]]) -> list[list[EncodedSymbol]]:
    return [[symbol.remove_tuplet() for symbol in chord] for chord in measure]


def _fix_over_eager_tuplets(chords: list[list[EncodedSymbol]]) -> list[list[EncodedSymbol]]:
    """
    The transformer tends to add too many tuplets, so we remove them
    based on the length of a measurement.
    """
    measures = _group_into_measures(chords)
    mean = _get_typical_duration_of_measures(measures)
    result = []
    for i, measure in enumerate(measures):
        if _get_duration_of_measure(measure) < mean:
            eprint("Removing tuplets from measure #", i + 1)
            result.append(_remove_tuplets(measure))
        else:
            result.append(measure)
    return _flatten_measures(result)


def _only_keep_lower_staff_if_there_is_a_clef(
    chords: list[list[EncodedSymbol]],
) -> list[list[EncodedSymbol]]:
    has_lower_clef = False
    all_results = []
    for i, chord in enumerate(chords):
        result = []
        for symbol in chord:
            if has_lower_clef:
                result.append(symbol)
            elif i < 5 and symbol.rhythm.startswith("clef") and symbol.position == "lower":
                has_lower_clef = True
                result.append(symbol)
            else:
                result.append(symbol.to_upper_position())
        all_results.append(result)
    delta = len(chords) - len(all_results)
    if delta > 0:
        eprint("Removed", delta, "results as there was no matching clef")
    return all_results


def remove_duplicated_symbols(
    symbols: list[EncodedSymbol], cleanup_tuplets: bool = True
) -> list[EncodedSymbol]:
    chords = _group_into_chords(symbols)
    if cleanup_tuplets:
        chords = _fix_over_eager_tuplets(chords)
        chords = _only_keep_lower_staff_if_there_is_a_clef(chords)
    chords = [_remove_duplicated_piches(c) for c in chords]
    chords = _remove_redudant_clefs_keys_and_time_signatures(chords)
    return _flatten_chords(chords)


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

    return [sorted(chord) for chord in chords]


if __name__ == "__main__":
    import json

    from homr.simple_logging import eprint

    vocab = Vocabulary()

    eprint("Rhythm=", json.dumps(vocab.rhythm, indent=2))
    eprint("Lift=", json.dumps(vocab.lift, indent=2))
    eprint("Articulation=", json.dumps(vocab.articulation, indent=2))
    eprint("Pitch=", json.dumps(vocab.pitch, indent=2))
    eprint("Slurs=", json.dumps(vocab.slur, indent=2))
    eprint("Positions=", json.dumps(vocab.position, indent=2))

    valid_combinations = []

    for r, li, a, p, sl, pos in itertools.product(
        vocab.rhythm, vocab.lift, vocab.articulation, vocab.pitch, vocab.slur, vocab.position
    ):
        symbol = EncodedSymbol(r, li, a, p, sl, pos)
        is_valid = symbol.is_valid()
        if not is_valid:
            continue
        valid_combinations.append(symbol)

    eprint(
        "Number of combinations",
        len(valid_combinations),
        "- some examples:",
    )
    for symbol in random.sample(valid_combinations, 10):
        eprint(symbol)
