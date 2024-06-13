from enum import Enum

import numpy as np

from homr import constants
from homr.simple_logging import eprint


class ResultSymbol:
    def __init__(self) -> None:
        pass


class ClefType(Enum):
    TREBLE = 1
    BASS = 2

    def __str__(self) -> str:
        if self == ClefType.TREBLE:
            return "G"
        elif self == ClefType.BASS:
            return "F"
        else:
            raise Exception("Unknown ClefType")

    def __repr__(self) -> str:
        return str(self)


class ResultTimeSignature(ResultSymbol):
    def __init__(self, time_signature: str) -> None:
        self.time_signature = time_signature

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, ResultTimeSignature):
            return self.time_signature == __value.time_signature
        else:
            return False

    def __hash__(self) -> int:
        return hash(self.time_signature)

    def __str__(self) -> str:
        return self.time_signature

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
        return f"{self.clef_type}{self.circle_of_fifth}"

    def __repr__(self) -> str:
        return str(self)

    def get_reference_pitch(self) -> ResultPitch:
        if self.clef_type == ClefType.TREBLE:
            return ResultPitch("D", 4, None)
        elif self.clef_type == ClefType.BASS:
            return ResultPitch("F", 2, None)
        else:
            raise Exception("Unknown ClefType " + str(self.clef_type))


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


def get_duration_name(duration: int) -> str:
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


class ResultDuration:
    def __init__(self, duration: int, has_dot: bool):
        self.duration = duration
        self.has_dot = has_dot
        self.duration_name = get_duration_name(self.base_duration())

    def base_duration(self) -> int:
        return self.duration * 2 // 3 if self.has_dot else self.duration

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, ResultDuration):
            return self.duration == __value.duration and self.has_dot == __value.has_dot
        else:
            return False

    def __hash__(self) -> int:
        return hash((self.duration, self.has_dot))

    def __str__(self) -> str:
        return f"{self.duration_name}{'.' if self.has_dot else ''}"

    def __repr__(self) -> str:
        return str(self)


class ResultRest(ResultSymbol):
    def __init__(self, duration: ResultDuration):
        self.duration = duration

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, ResultRest):
            return self.duration == __value.duration
        else:
            return False

    def __hash__(self) -> int:
        return hash(self.duration)

    def __str__(self) -> str:
        return f"R_{self.duration}"

    def __repr__(self) -> str:
        return str(self)


class ResultNote(ResultSymbol):
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


class ResultNoteGroup(ResultSymbol):
    def __init__(self, notes: list[ResultNote]):
        self.notes = notes
        self.duration = notes[0].duration if len(notes) > 0 else ResultDuration(0, False)

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, ResultNoteGroup):
            if len(self.notes) != len(__value.notes):
                return False
            for i in range(len(self.notes)):
                if self.notes[i] != __value.notes[i]:
                    return False
            return True
        else:
            return False

    def __hash__(self) -> int:
        return hash(tuple(self.notes))

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
            symbol.duration.base_duration()
            for symbol in self.symbols
            if isinstance(symbol, ResultNote | ResultNoteGroup | ResultRest)
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
