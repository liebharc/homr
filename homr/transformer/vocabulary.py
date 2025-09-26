import itertools
import random
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
    # rhythm.extend(["voltaStart", "voltaEnd"])

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

    # Note relations (https://en.wikipedia.org/wiki/List_of_musical_symbols)
    rhythm.extend(["tieSlur"])  #  "gliss"

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


def build_state() -> dict[str, int]:
    """
    States keeps track of the current clef (for upper and lower staff)
    and the current key signature.
    """
    keys = [f"keySignature_{c}" for c in range(-7, 8)]
    clefs = []
    clefs.extend([f"clef_F{c}" for c in range(3, 6)])
    clefs.extend([f"clef_C{c}" for c in range(1, 6)])
    clefs.extend([f"clef_G{c}" for c in range(1, 3)])
    states = [nonote]
    for key, upper_clef, lower_clef in itertools.product(keys, clefs, clefs):
        states.append(f"{key}+{upper_clef}+{lower_clef}")
    return build_dict(states)


def build_articulation() -> dict[str, int]:
    articulation = [nonote, empty]

    # The lieder dataset has the superset of articulations
    # so we use it to define this list
    articulations_lieder = [
        "accent",
        "accent_arpeggiate",
        "accent_arpeggiate_fermata",
        "accent_arpeggiate_slurStart",
        "accent_arpeggiate_slurStart_staccato",
        "accent_arpeggiate_slurStart_tieStop",
        "accent_arpeggiate_slurStop",
        "accent_arpeggiate_slurStop_staccato",
        "accent_arpeggiate_staccato",
        "accent_arpeggiate_tieStart",
        "accent_breathMark",
        "accent_breathMark_fermata",
        "accent_breathMark_slurStop",
        "accent_fermata",
        "accent_fermata_slurStart",
        "accent_fermata_slurStop",
        "accent_fermata_tieStart",
        "accent_slurStart",
        "accent_slurStart_slurStop",
        "accent_slurStart_slurStop_tenuto",
        "accent_slurStart_staccato",
        "accent_slurStart_tenuto",
        "accent_slurStart_tieStart",
        "accent_slurStop",
        "accent_slurStop_staccato",
        "accent_slurStop_tieStart",
        "accent_slurStop_tieStart_trill",
        "accent_slurStop_tieStop",
        "accent_slurStop_trill",
        "accent_staccatissimo",
        "accent_staccato",
        "accent_staccato_tieStart",
        "accent_tenuto",
        "accent_tieStart",
        "accent_tieStop",
        "accent_tremolo",
        "accent_trill",
        "arpeggiate",
        "arpeggiate_fermata",
        "arpeggiate_fermata_slurStart",
        "arpeggiate_fermata_slurStop",
        "arpeggiate_fermata_tieStart",
        "arpeggiate_fermata_tieStop",
        "arpeggiate_slurStart",
        "arpeggiate_slurStart_slurStop",
        "arpeggiate_slurStart_slurStop_tieStart",
        "arpeggiate_slurStart_staccato",
        "arpeggiate_slurStart_staccato_tenuto",
        "arpeggiate_slurStart_tenuto",
        "arpeggiate_slurStart_tieStart",
        "arpeggiate_slurStart_tieStop",
        "arpeggiate_slurStop",
        "arpeggiate_slurStop_staccatissimo",
        "arpeggiate_slurStop_staccato",
        "arpeggiate_slurStop_tieStart",
        "arpeggiate_staccatissimo",
        "arpeggiate_staccato",
        "arpeggiate_tenuto",
        "arpeggiate_tieStart",
        "arpeggiate_tieStart_tieStop",
        "arpeggiate_tieStop",
        "arpeggiate_trill",
        "breathMark",
        "breathMark_fermata",
        "breathMark_fermata_slurStop",
        "breathMark_fermata_tenuto",
        "breathMark_slurStart",
        "breathMark_slurStop",
        "breathMark_slurStop_staccato",
        "breathMark_slurStop_tieStop",
        "breathMark_staccato",
        "breathMark_staccato_tenuto",
        "breathMark_tenuto",
        "breathMark_tieStop",
        "breathMark_tieStop_trill",
        "fermata",
        "fermata_slurStart",
        "fermata_slurStart_slurStop",
        "fermata_slurStart_tieStart",
        "fermata_slurStart_tieStop",
        "fermata_slurStart_trill",
        "fermata_slurStop",
        "fermata_slurStop_tenuto",
        "fermata_slurStop_tieStart",
        "fermata_slurStop_tieStop",
        "fermata_staccato",
        "fermata_tenuto",
        "fermata_tieStart",
        "fermata_tieStart_tieStop",
        "fermata_tieStop",
        "fermata_tremolo",
        "fermata_trill",
        "slurStart",
        "slurStart_slurStop",
        "slurStart_slurStop_staccato",
        "slurStart_slurStop_staccato_tenuto",
        "slurStart_slurStop_staccato_tieStop",
        "slurStart_slurStop_tenuto",
        "slurStart_slurStop_tieStart",
        "slurStart_slurStop_tieStart_tieStop",
        "slurStart_slurStop_tieStop",
        "slurStart_slurStop_trill",
        "slurStart_staccatissimo",
        "slurStart_staccato",
        "slurStart_staccato_tenuto",
        "slurStart_staccato_tieStart",
        "slurStart_staccato_tieStop",
        "slurStart_tenuto",
        "slurStart_tenuto_tieStart",
        "slurStart_tieStart",
        "slurStart_tieStart_tieStop",
        "slurStart_tieStop",
        "slurStart_tieStop_turn",
        "slurStart_tremolo",
        "slurStart_trill",
        "slurStart_turn",
        "slurStop",
        "slurStop_staccatissimo",
        "slurStop_staccato",
        "slurStop_staccato_tenuto",
        "slurStop_staccato_tieStart",
        "slurStop_staccato_tieStop",
        "slurStop_tenuto",
        "slurStop_tenuto_tieStart",
        "slurStop_tieStart",
        "slurStop_tieStart_tieStop",
        "slurStop_tieStart_trill",
        "slurStop_tieStop",
        "slurStop_tremolo",
        "slurStop_trill",
        "slurStop_turn",
        "staccatissimo",
        "staccato",
        "staccato_tenuto",
        "staccato_tenuto_tieStart",
        "staccato_tieStart",
        "staccato_tieStart_tieStop",
        "staccato_tieStop",
        "staccato_trill",
        "tenuto",
        "tenuto_tieStart",
        "tenuto_tieStop",
        "tieStart",
        "tieStart_tieStop",
        "tieStart_tremolo",
        "tieStart_trill",
        "tieStop",
        "tieStop_trill",
        "tremolo",
        "trill",
        "turn",
    ]

    articulation.extend(articulations_lieder)

    return build_dict(articulation)


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
        self.position = build_position()
        self.state = build_state()


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


def next_power_of_two(n: int) -> int:
    """Return the next power of two >= n."""
    return 1 << (n - 1).bit_length()


def kern_to_symbol_duration(kern: str) -> SymbolDuration:
    """
    Parse a Humdrum **kern duration string into a SymbolDuration object.
    """
    if kern.endswith("m"):
        # Multirest
        SymbolDuration(Fraction(0), 0, 1, 1, 4)

    # Extract numeric prefix (can be > 1 digit)
    i = 0
    while i < len(kern) and kern[i].isdigit():
        i += 1
    base_str = kern[:i]
    rest = kern[i:]

    base = int(base_str) if base_str else 4  # default quarter
    dots = rest.count(".")

    if base == 0:
        # Special: grace note (duration = 0)
        return SymbolDuration(Fraction(0), dots, 1, 1, base)

    # If base is a power of two, it's a normal note
    if base & (base - 1) == 0:
        base_duration = Fraction(1, base)
        actual_notes, normal_notes = 1, 1
        return SymbolDuration(base_duration, dots, actual_notes, normal_notes, base)
    else:
        # Tuplet case: find next higher power of two
        normal_notes = next_power_of_two(base)
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
        position: str = nonote,
    ) -> None:
        self.rhythm = rhythm
        self.pitch = pitch
        self.lift = lift
        self.articulation = articulation
        self.position = position
        self._duration: SymbolDuration | None = None

    def is_control_symbol(self) -> bool:
        return self.rhythm in ("BOS", "EOS", "PAD")

    def is_valid(self) -> bool:
        has_position = has_rhythm_symbol_a_position(self.rhythm)
        is_note = [s != nonote for s in [self.lift, self.articulation, self.pitch, self.position]]
        return all(item == has_position for item in is_note)

    def get_duration(self) -> SymbolDuration:
        """
        Convert a Humdrum **kern duration into a Fraction (relative to a whole note).
        Assumes 4 = quarter note = 1/4.
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
        return str.join(" ", (self.rhythm, self.pitch, self.lift, self.articulation, self.position))

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, EncodedSymbol):
            return (
                self.rhythm == __value.rhythm
                and self.pitch == __value.pitch
                and self.lift == __value.lift
                and self.articulation == __value.articulation
                and self.position == __value.position
            )
        else:
            return False

    def __hash__(self) -> int:
        return hash((self.rhythm, self.pitch, self.lift, self.articulation, self.position))


if __name__ == "__main__":
    import json

    from homr.simple_logging import eprint

    vocab = Vocabulary()

    eprint("Rhythm=", json.dumps(vocab.rhythm, indent=2))
    eprint("Lift=", json.dumps(vocab.lift, indent=2))
    eprint("Articulation=", json.dumps(vocab.articulation, indent=2))
    eprint("Pitch=", json.dumps(vocab.pitch, indent=2))
    eprint("Positions=", json.dumps(vocab.position, indent=2))
    eprint("States=", json.dumps(vocab.state, indent=2))

    valid_combinations = []

    for r, li, a, p, pos in itertools.product(
        vocab.rhythm, vocab.lift, vocab.articulation, vocab.pitch, vocab.position
    ):
        symbol = EncodedSymbol(r, li, a, p, pos)
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
