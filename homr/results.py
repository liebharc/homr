from enum import Enum

import numpy as np

from homr import constants
from homr.simple_logging import eprint


class ResultSymbol:
    def __init__(self) -> None:
        pass


class ClefType:
    @staticmethod
    def treble() -> "ClefType":
        return ClefType(sign="G", line=2)

    @staticmethod
    def bass() -> "ClefType":
        return ClefType(sign="F", line=4)

    def __init__(self, sign: str, line: int) -> None:
        """
        Why we don't support more clef types, e.g. the other examples given in
        https://www.w3.org/2021/06/musicxml40/musicxml-reference/elements/clef/:

        Since the other clef types share the same symbol with one of the ones we support,
        we have to expect that the there is a lot of misdetections and this would degrade the
        performance.

        E.g. if the treble and french violin (https://en.wikipedia.org/wiki/Clef)
        are easily confused. If support french violin then we will have cases where #
        the treble clef is detected as french violin and then the pitch will be wrong.

        If we get more training data and a reliable detecton of the rarer clef types,
        we can add them here.
        """
        self.sign = sign.upper()

        if self.sign not in ["G", "F", "C"]:
            raise Exception("Unknown clef sign " + sign)

        # Extend get_reference_pitch if you add more clef types
        treble_clef_line = 2
        bass_clef_line = 4
        alto_clef_line = 3
        if sign == "G" and line != treble_clef_line:
            eprint("Unsupported treble clef line", line)
            self.line = treble_clef_line
        elif sign == "F" and line != bass_clef_line:
            eprint("Unsupported bass clef line", line)
            self.line = bass_clef_line
        elif sign == "C" and line != alto_clef_line:
            eprint("Unsupported alto clef line", line)
            self.line = alto_clef_line
        else:
            self.line = line

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, ClefType):
            return self.sign == __value.sign and self.line == __value.line
        else:
            return False

    def __hash__(self) -> int:
        return hash((self.sign, self.line))

    def __str__(self) -> str:
        return f"{self.sign}{self.line}"

    def __repr__(self) -> str:
        return str(self)

    def get_reference_pitch(self) -> "ResultPitch":
        if self.sign == "G":
            g2 = ResultPitch("C", 4, None)
            return g2.move_by(2 * (self.line - 2), None)
        elif self.sign == "F":
            e2 = ResultPitch("E", 2, None)
            return e2.move_by(2 * (self.line - 4), None)
        elif self.sign == "C":
            c3 = ResultPitch("C", 3, None)
            return c3.move_by(2 * (self.line - 3), None)
        raise ValueError("Unknown clef sign " + str(self))


class ResultTimeSignature(ResultSymbol):
    def __init__(self, numerator: int, denominator: int) -> None:
        self.numerator = numerator
        self.denominator = denominator

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, ResultTimeSignature):
            return self.numerator == __value.numerator and self.denominator == __value.denominator
        else:
            return False

    def __hash__(self) -> int:
        return hash((self.numerator, self.denominator))

    def __str__(self) -> str:
        return f"{self.numerator}/{self.denominator}"

    def __repr__(self) -> str:
        return str(self)


note_names = ["C", "D", "E", "F", "G", "A", "B"]


class ResultPitch:
    def __init__(self, step: str, octave: int, alter: int | None) -> None:
        self.step = step
        self.octave = octave
        self.alter = alter

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, ResultPitch):
            return (
                self.step == __value.step
                and self.octave == __value.octave
                and self.alter == __value.alter
            )
        else:
            return False

    def name_and_octave(self) -> str:
        return self.step + str(self.octave)

    def __hash__(self) -> int:
        return hash((self.step, self.octave, self.alter))

    def __str__(self) -> str:
        alter = ""
        if self.alter == 1:
            alter = "#"
        elif self.alter == -1:
            alter = "b"
        elif self.alter == 0:
            alter = "♮"
        return f"{self.step}{self.octave}{alter}"

    def __repr__(self) -> str:
        return str(self)

    def get_relative_position(self, other: "ResultPitch") -> int:
        return (
            (self.octave - other.octave) * 7
            + note_names.index(self.step)
            - note_names.index(other.step)
        )

    def move_by(self, steps: int, alter: int | None) -> "ResultPitch":
        step_index = (note_names.index(self.step) + steps) % 7
        step = note_names[step_index]
        octave = self.octave + abs(steps - step_index) // 6 * np.sign(steps)
        return ResultPitch(step, octave, alter)


def get_pitch_from_relative_position(
    reference_pitch: ResultPitch, relative_position: int, alter: int | None
) -> ResultPitch:
    step_index = (note_names.index(reference_pitch.step) + relative_position) % 7
    step = note_names[step_index]
    # abs & sign give us integer division with rounding towards 0
    octave = reference_pitch.octave + abs(relative_position - step_index) // 6 * np.sign(
        relative_position
    )
    return ResultPitch(step, int(octave), alter)


class ResultClef(ResultSymbol):
    def __init__(self, clef_type: ClefType, circle_of_fifth: int) -> None:
        self.clef_type = clef_type
        self.circle_of_fifth = circle_of_fifth

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, ResultClef):
            return (
                self.clef_type == __value.clef_type
                and self.circle_of_fifth == __value.circle_of_fifth
            )
        else:
            return False

    def __hash__(self) -> int:
        return hash((self.clef_type, self.circle_of_fifth))

    def __str__(self) -> str:
        return f"{self.clef_type}/{self.circle_of_fifth}"

    def __repr__(self) -> str:
        return str(self)

    def get_reference_pitch(self) -> ResultPitch:
        return self.clef_type.get_reference_pitch()


def move_pitch_to_clef(
    pitch: ResultPitch, current: ResultClef | None, new: ResultClef
) -> ResultPitch:
    """
    Moves the pitch from the current clef to the new clef under the assumption that the clef
    was incorrectly identified, but the pitch position is correct.
    """
    if current is None or new is None or current.clef_type == new.clef_type:
        return pitch
    current_reference_pitch = current.get_reference_pitch()
    new_reference_pitch = new.get_reference_pitch()
    relative_position = pitch.get_relative_position(current_reference_pitch)
    return get_pitch_from_relative_position(
        new_reference_pitch, relative_position, alter=pitch.alter
    )


def _get_duration_name(duration: int) -> str:
    duration_dict = {
        4 * constants.duration_of_quarter: "whole",
        2 * constants.duration_of_quarter: "half",
        constants.duration_of_quarter: "quarter",
        constants.duration_of_quarter / 2: "eighth",
        constants.duration_of_quarter / 4: "16th",
        constants.duration_of_quarter / 8: "32nd",
        constants.duration_of_quarter / 16: "64th",
    }
    result = duration_dict.get(duration, None)
    if result is None:
        eprint("Unknown duration", duration)
        return "quarter"
    return result


class DurationModifier(Enum):
    NONE = 0
    DOT = 1
    TRIPLET = 2

    def __init__(self, duration: int) -> None:
        self.duration = duration

    def __str__(self) -> str:
        if self == DurationModifier.NONE:
            return ""
        elif self == DurationModifier.DOT:
            return "."
        elif self == DurationModifier.TRIPLET:
            return constants.triplet_symbol
        else:
            return "Invalid duration"


def _adjust_duration(duration: int, modifier: DurationModifier) -> int:
    if modifier == DurationModifier.DOT:
        return duration * 3 // 2
    elif modifier == DurationModifier.TRIPLET:
        return duration * 2 // 3
    else:
        return duration


class ResultDuration:
    def __init__(self, base_duration: int, modifier: DurationModifier = DurationModifier.NONE):
        self.duration = _adjust_duration(base_duration, modifier)
        self.modifier = modifier
        self.duration_name = _get_duration_name(base_duration)

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, ResultDuration):
            return self.duration == __value.duration and self.modifier == __value.modifier
        else:
            return False

    def __hash__(self) -> int:
        return hash((self.duration, self.modifier))

    def __str__(self) -> str:
        return f"{self.duration_name}{str(self.modifier)}"

    def __repr__(self) -> str:
        return str(self)


class ResultNote:
    def __init__(self, pitch: ResultPitch, duration: ResultDuration):
        self.pitch = pitch
        self.duration = duration

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, ResultNote):
            return self.pitch == __value.pitch and self.duration == __value.duration
        else:
            return False

    def __hash__(self) -> int:
        return hash((self.pitch, self.duration))

    def __str__(self) -> str:
        return f"{self.pitch}_{self.duration}"

    def __repr__(self) -> str:
        return str(self)


def get_min_duration(notes: list[ResultNote]) -> ResultDuration:
    if len(notes) == 0:
        return ResultDuration(constants.duration_of_quarter)
    return min([note.duration for note in notes], key=lambda x: x.duration)


class ResultChord(ResultSymbol):
    """
    A chord which contains 0 to many pitches. 0 pitches indicates that this is a rest.

    The duration of the chord is the distance to the next chord. The individual pitches
    my have a different duration.
    """

    def __init__(self, duration: ResultDuration, notes: list[ResultNote]):
        self.notes = notes
        self.duration = duration

    @property
    def is_rest(self) -> bool:
        return len(self.notes) == 0

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, ResultChord):
            return self.duration == __value.duration and self.notes == __value.notes
        else:
            return False

    def __hash__(self) -> int:
        return hash((self.notes, self.duration))

    def __str__(self) -> str:
        return f"{'&'.join(map(str, self.notes))}"

    def __repr__(self) -> str:
        return str(self)


class ResultMeasure:
    def __init__(self, symbols: list[ResultSymbol]):
        self.symbols = symbols
        self.is_new_line = False

    def is_empty(self) -> bool:
        return len(self.symbols) == 0

    def remove_symbol(self, symbol: ResultSymbol) -> None:
        len_before = len(self.symbols)
        self.symbols = [s for s in self.symbols if s is not symbol]
        if len_before == len(self.symbols):
            raise Exception("Could not remove symbol")

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, ResultMeasure):
            if len(self.symbols) != len(__value.symbols):
                return False
            for i in range(len(self.symbols)):
                if self.symbols[i] != __value.symbols[i]:
                    return False
            return True
        else:
            return False

    def __hash__(self) -> int:
        return hash(tuple(self.symbols))

    def __str__(self) -> str:
        return f"{' '.join(map(str, self.symbols))}" + "|"

    def __repr__(self) -> str:
        return str(self)

    def length_in_quarters(self) -> float:
        return sum(
            symbol.duration.duration for symbol in self.symbols if isinstance(symbol, ResultChord)
        )


class ResultStaff:
    def __init__(self, measures: list[ResultMeasure]):
        self.measures = measures

    def merge(self, other: "ResultStaff") -> "ResultStaff":
        return ResultStaff(self.measures + other.measures)

    def get_symbols(self) -> list[ResultSymbol]:
        symbols = []
        for measure in self.measures:
            symbols.extend(measure.symbols)
        return symbols

    def number_of_new_lines(self) -> int:
        return sum(1 for measure in self.measures if measure.is_new_line)

    def replace_symbol(self, old_symbol: ResultSymbol, new_symbol: ResultSymbol) -> None:
        for measure in self.measures:
            measure.symbols = [new_symbol if s is old_symbol else s for s in measure.symbols]

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, ResultStaff):
            if len(self.measures) != len(__value.measures):
                return False
            for i in range(len(self.measures)):
                if self.measures[i] != __value.measures[i]:
                    return False
            return True
        else:
            return False

    def __hash__(self) -> int:
        return hash(tuple(self.measures))

    def __str__(self) -> str:
        return "Staff(" + f"{' '.join(map(str, self.measures))}" + ")"

    def __repr__(self) -> str:
        return str(self)

    def is_empty(self) -> bool:
        return len(self.measures) == 0
