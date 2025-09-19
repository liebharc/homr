import abc
from abc import ABC

from homr.simple_logging import eprint
from homr.transformer.vocabulary import EncodedSymbol, empty

circle_of_fifth_notes_positive = ["F", "C", "G", "D", "A", "E", "B"]
circle_of_fifth_notes_negative = list(reversed(circle_of_fifth_notes_positive))


def get_circle_of_fifth_notes(circle_of_fifth: int) -> list[str]:
    if circle_of_fifth >= 0:
        return circle_of_fifth_notes_positive[0:circle_of_fifth]
    else:
        return circle_of_fifth_notes_negative[0 : abs(circle_of_fifth)]


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
    def add_note(self, note: EncodedSymbol) -> EncodedSymbol:
        pass

    @abc.abstractmethod
    def reset_at_end_of_measure(self) -> "AbstractKeyTransformation":
        pass


class NoKeyTransformation(AbstractKeyTransformation):

    def __init__(self) -> None:
        self.current_accidentals: dict[str, str] = {}

    def add_note(self, note: EncodedSymbol) -> EncodedSymbol:
        if note.lift != empty and (
            note.pitch not in self.current_accidentals
            or self.current_accidentals[note.pitch] != note.lift
        ):
            self.current_accidentals[note.pitch] = note.lift
            return note
        else:
            note.lift = empty
            return note

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

    def add_note(self, note: EncodedSymbol) -> EncodedSymbol:
        note.lift = self._get_lift(note)
        return note

    def _get_lift(self, note: EncodedSymbol) -> str:
        """
        Returns the accidental if it wasn't placed before.
        """
        if note.lift != empty:
            previous_accidental = "N"
            if note.pitch in self.sharps:
                self.sharps.remove(note.pitch)
                previous_accidental = "#"
            if note.pitch in self.flats:
                self.flats.remove(note.pitch)
                previous_accidental = "b"

            if note.lift == "#":
                self.sharps.add(note.pitch)
            elif note.lift == "b":
                self.flats.add(note.pitch)

            return note.lift if note.lift != previous_accidental else empty
        else:
            if note.pitch in self.sharps:
                self.sharps.remove(note.pitch)
                return "N"
            if note.pitch in self.flats:
                self.flats.remove(note.pitch)
                return "N"
            return empty

    def reset_at_end_of_measure(self) -> "KeyTransformation":
        return KeyTransformation(self.circle_of_fifth)


def semantic_to_agnostic_accidentals(symbols: list[EncodedSymbol]) -> list[EncodedSymbol]:
    key: AbstractKeyTransformation = NoKeyTransformation()

    result = []
    for symbol in symbols:
        if "barline" in symbol.rhythm or "repeat" in symbol.rhythm:
            key = key.reset_at_end_of_measure()
            result.append(symbol)
        elif symbol.rhythm.startswith("keySignature"):
            circle_of_fifth = int(symbol.rhythm.split("_")[1])
            key = KeyTransformation(circle_of_fifth)
            result.append(symbol)
        elif symbol.rhythm.startswith("note"):
            result.append(key.add_note(symbol))
        else:
            result.append(symbol)

    return result


def agnostic_to_semantic_accidentals(symbols: list[EncodedSymbol]) -> list[EncodedSymbol]:
    return symbols
