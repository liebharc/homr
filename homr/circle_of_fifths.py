import abc
from abc import ABC

from homr.simple_logging import eprint
from homr.transformer.vocabulary import EncodedSymbol, empty, nonote

definition = {
    -7: "CbM",
    -6: "GbM",
    -5: "DbM",
    -4: "AbM",
    -3: "EbM",
    -2: "BbM",
    -1: "FM",
    0: "CM",
    1: "GM",
    2: "DM",
    3: "AM",
    4: "EM",
    5: "BM",
    6: "F#M",
    7: "C#M",
}

inv_definition = {v: k for k, v in definition.items()}

circle_of_fifth_notes_positive = ["F", "C", "G", "D", "A", "E", "B"]
circle_of_fifth_notes_negative = list(reversed(circle_of_fifth_notes_positive))


def key_signature_to_circle_of_fifth(key_signature: str) -> int:
    if key_signature not in inv_definition:
        eprint("Warning: Unknown key signature", key_signature)
        return 0
    return inv_definition[key_signature]


def repeat_note_for_all_octaves(notes: list[str]) -> list[str]:
    """
    Takes a list of notes and returns a list of notes that includes all octaves.
    """

    result = []

    for note in notes:
        for octave in range(11):
            result.append(note + str(octave))
    return result


class AbstractKeyTransformation(ABC):

    @abc.abstractmethod
    def add_accidental(self, note: str, accidental: str) -> str:
        pass

    @abc.abstractmethod
    def reset_at_end_of_measure(self) -> "AbstractKeyTransformation":
        pass


class NoKeyTransformation(AbstractKeyTransformation):

    def __init__(self) -> None:
        self.current_accidentals: dict[str, str] = {}

    def add_accidental(self, note: str, accidental: str) -> str:
        if accidental != "" and (
            note not in self.current_accidentals or self.current_accidentals[note] != accidental
        ):
            self.current_accidentals[note] = accidental
            return accidental
        else:
            return ""

    def reset_at_end_of_measure(self) -> "NoKeyTransformation":
        return NoKeyTransformation()


class KeyTransformation(AbstractKeyTransformation):

    def __init__(self, circle_of_fifth: int):
        self.circle_of_fifth = circle_of_fifth
        self.sharps: set[str] = set()
        self.flats: set[str] = set()
        if circle_of_fifth > 0:
            self.sharps = set(
                repeat_note_for_all_octaves(circle_of_fifth_notes_positive[0:circle_of_fifth])
            )
        elif circle_of_fifth < 0:
            self.flats = set(
                repeat_note_for_all_octaves(
                    circle_of_fifth_notes_negative[0 : abs(circle_of_fifth)]
                )
            )

    def add_accidental(self, note: str, accidental: str | None) -> str:
        """
        Returns the accidental if it wasn't placed before.
        """

        if accidental in ["#", "b", "N"]:
            previous_accidental = "N"
            if note in self.sharps:
                self.sharps.remove(note)
                previous_accidental = "#"
            if note in self.flats:
                self.flats.remove(note)
                previous_accidental = "b"
            if accidental == "#":
                self.sharps.add(note)
            elif accidental == "b":
                self.flats.add(note)
            return accidental if accidental != previous_accidental else ""
        else:
            if note in self.sharps:
                self.sharps.remove(note)
                return "N"

            if note in self.flats:
                self.flats.remove(note)
                return "N"
            return ""

    def reset_at_end_of_measure(self) -> "KeyTransformation":
        return KeyTransformation(self.circle_of_fifth)


def convert_sounding_to_engraving_representation(
    symbols: list[EncodedSymbol],
) -> list[EncodedSymbol]:
    """
    The distinction in encoding accidentals is typically described as:

    1. Notation-based encoding (visual/engraving representation):
    - Stores accidentals exactly as they appear in the score, following notation rules.
    - Example: In a measure with two F#s, only the first carries a #; the second is implied.
    - Used in some engraving formats and MuseScore's internal format.

    2. Pitch-based encoding (sounding representation):
    - Stores the actual sounding pitch of each note, ignoring visual notation conventions.
    - Example: Both F#s in the measure are explicitly encoded.
    - Used in standard MusicXML <pitch> representation and MIDI.
    """
    results = []
    key = KeyTransformation(0)
    for symbol in symbols:
        if "barline" in symbol.rhythm:
            key = key.reset_at_end_of_measure()
            results.append(symbol)
        elif symbol.rhythm.startswith("keySignature_"):
            key = KeyTransformation(int(symbol.rhythm.split("_")[1]))
            results.append(symbol)
        elif symbol.lift != nonote:
            lift = symbol.lift if symbol.lift != empty else None
            note = symbol.pitch[0]
            accidental = key.add_accidental(note, lift)
            results.append(symbol.change_lift(accidental if accidental else empty))
        else:
            results.append(symbol)

    return results


def convert_engraving_to_sounding_representation(
    symbols: list[EncodedSymbol],
) -> list[EncodedSymbol]:
    """
    Converts notation-based (engraving) representation into pitch-based (sounding) representation.
    All accidentals are explicitly stored according to actual sounding pitch.
    """
    results = []
    key = KeyTransformation(0)

    for symbol in symbols:
        if "barline" in symbol.rhythm:
            key = key.reset_at_end_of_measure()
            results.append(symbol)
        elif symbol.rhythm.startswith("keySignature_"):
            key = KeyTransformation(int(symbol.rhythm.split("_")[1]))
            results.append(symbol)
        elif symbol.lift != nonote:
            note = symbol.pitch[0]
            # In engraving, the lift may be empty (implied by key signature or previous accidental)
            # In sounding, we need the actual pitch: remove any previous
            # accidental tracking to force the current one
            lift = symbol.lift if symbol.lift != empty else None
            actual_accidental = None

            if lift in ["#", "b", "N"]:
                actual_accidental = lift
                # Update key state to reflect that this accidental has been
                # applied for future notes in the measure
                key.add_accidental(note, lift)
            elif note in key.sharps:  # Determine if note is sharp/flat by key signature
                actual_accidental = "#"
            elif note in key.flats:
                actual_accidental = "b"
            else:
                actual_accidental = empty

            results.append(symbol.change_lift(actual_accidental if actual_accidental else empty))
        else:
            results.append(symbol)

    return results
