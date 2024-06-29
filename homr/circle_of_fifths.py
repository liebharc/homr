import abc
from abc import ABC

from homr.simple_logging import eprint

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


def circle_of_fifth_to_key_signature(circle: int) -> str:
    return definition[circle]


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
