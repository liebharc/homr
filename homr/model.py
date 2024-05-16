from abc import abstractmethod
from enum import Enum
from typing import Any

import cv2
import numpy as np

from homr import constants
from homr.bounding_boxes import (
    AngledBoundingBox,
    BoundingBox,
    BoundingEllipse,
    DebugDrawable,
    RotatedBoundingBox,
)
from homr.circle_of_fifths import get_circle_of_fifth_notes
from homr.results import (
    ClefType,
    ResultClef,
    ResultDuration,
    ResultNote,
    ResultNoteGroup,
    ResultPitch,
    ResultRest,
    ResultSymbol,
)
from homr.type_definitions import NDArray


class Prediction:
    def __init__(self, result: dict[Any, Any], best: Any) -> None:
        self.result = result
        self.best = best

    def __str__(self) -> str:
        return str(self.result)


class InputPredictions:
    def __init__(
        self,
        original: NDArray,
        preprocessed: NDArray,
        notehead: NDArray,
        symbols: NDArray,
        staff: NDArray,
        clefs_keys: NDArray,
        stems_rest: NDArray,
    ) -> None:
        self.original = original
        self.preprocessed = preprocessed
        self.notehead = notehead
        self.symbols = symbols
        self.staff = staff
        self.stems_rest = stems_rest
        self.clefs_keys = clefs_keys


class SymbolOnStaff(DebugDrawable):
    def __init__(self, center: tuple[float, float]) -> None:
        self.center = center

    @abstractmethod
    def to_result(self) -> ResultSymbol | None:
        pass


class AccidentalType(Enum):
    SHARP = 1
    FLAT = 2
    NATURAL = 3

    def __str__(self) -> str:
        if self == AccidentalType.SHARP:
            return "#"
        elif self == AccidentalType.FLAT:
            return "b"
        elif self == AccidentalType.NATURAL:
            return "n"
        else:
            raise Exception("Unknown AccidentalType")

    def to_alter(self) -> int:
        if self == AccidentalType.SHARP:
            return 1
        elif self == AccidentalType.FLAT:
            return -1
        elif self == AccidentalType.NATURAL:
            return 0
        else:
            raise Exception("Unknown AccidentalType")


class Accidental(SymbolOnStaff):
    def __init__(self, box: BoundingBox, prediction: Prediction, position: int) -> None:
        super().__init__(box.center)
        self.box = box
        self.prediction = prediction
        self.position = position

    def get_accidental_type(self) -> AccidentalType:
        if self.prediction.best == "sharp":
            return AccidentalType.SHARP
        elif self.prediction.best == "flat":
            return AccidentalType.FLAT
        elif self.prediction.best == "natural":
            return AccidentalType.NATURAL
        else:
            raise Exception("Unknown accidental type " + self.prediction.best)

    def draw_onto_image(self, img: NDArray, color: tuple[int, int, int] = (255, 0, 0)) -> None:
        self.box.draw_onto_image(img, color)
        cv2.putText(
            img,
            str(self.prediction.best) + "-" + str(self.position),
            (int(self.box.box[0]), int(self.box.box[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
            cv2.LINE_AA,
        )

    def to_result(self) -> None:
        return None


class Rest(SymbolOnStaff):
    def __init__(self, box: BoundingBox, prediction: Prediction) -> None:
        super().__init__(box.center)
        self.box = box
        self.prediction = prediction
        self.has_dot = False

    def draw_onto_image(self, img: NDArray, color: tuple[int, int, int] = (255, 0, 0)) -> None:
        self.box.draw_onto_image(img, color)
        cv2.putText(
            img,
            str(self.prediction.best),
            (int(self.box.box[0]), int(self.box.box[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
            cv2.LINE_AA,
        )

    def get_duration(self) -> int:
        text = self.prediction.best
        duration_dict = {
            "rest_whole": 4 * constants.duration_of_quarter,
            "rest_half": 2 * constants.duration_of_quarter,
            "rest_quarter": constants.duration_of_quarter,
            "rest_8th": constants.duration_of_quarter // 2,
            "rest_16th": constants.duration_of_quarter // 4,
            "rest_32nd": constants.duration_of_quarter // 8,
            "rest_64th": constants.duration_of_quarter // 16,
        }
        return duration_dict.get(text, 0)  # returns 0 if text is not found in the dictionary

    def to_result(self) -> ResultRest:
        return ResultRest(ResultDuration(self.get_duration(), self.has_dot))


class StemDirection(Enum):
    UP = 1
    DOWN = 2


class NoteHeadType(Enum):
    HOLLOW = 1
    SOLID = 2

    def __str__(self) -> str:
        if self == NoteHeadType.HOLLOW:
            return "O"
        elif self == NoteHeadType.SOLID:
            return "*"
        else:
            raise Exception("Unknown NoteHeadType")


note_names = ["C", "D", "E", "F", "G", "A", "B"]


class Pitch:
    def __init__(self, step: str, alter: int | None, octave: int):
        self.step = step
        self.alter: int | None
        if alter is not None:
            self.alter = int(alter)
        else:
            self.alter = None
        self.octave = int(octave)

    def move_by_position(
        self, position: int, accidental: Accidental | None, circle_of_fifth: int
    ) -> "Pitch":
        # Find the current position of the note in the scale
        current_position = note_names.index(self.step)

        # Calculate the new position
        new_position = (current_position + position) % len(note_names)

        # Calculate the new octave
        new_octave = self.octave + ((current_position + position) // len(note_names))

        # Get the new step
        new_step = note_names[new_position]

        alter = None
        if new_step in get_circle_of_fifth_notes(circle_of_fifth):
            if circle_of_fifth < 0:
                alter = -1
            else:
                alter = 1

        if accidental is not None:
            accidental_type = accidental.get_accidental_type()
            if accidental_type == AccidentalType.SHARP:
                alter = 1
            elif accidental_type == AccidentalType.FLAT:
                alter = -1
            elif accidental_type == AccidentalType.NATURAL:
                alter = 0

        return Pitch(new_step, alter, new_octave)

    def to_result(self) -> ResultPitch:
        return ResultPitch(self.step, self.octave, self.alter)


reference_pitch_f_clef = Pitch("F", 0, 2)
reference_pitch_g_clef = Pitch("D", 0, 4)


class Note(SymbolOnStaff):
    def __init__(
        self,
        box: BoundingEllipse,
        position: int,
        notehead_type: Prediction,
        stem: RotatedBoundingBox | None,
        stem_direction: StemDirection | None,
    ):
        super().__init__(box.center)
        self.box = box
        self.position = position
        self.has_dot = False
        self.beam_count = 0
        self.notehead_type = notehead_type
        self.stem = stem
        self.clef_type = ClefType.TREBLE
        self.circle_of_fifth = 0
        self.accidental: Accidental | None = None
        self.stem_direction = stem_direction
        self.beams: list[RotatedBoundingBox] = []
        self.flags: list[RotatedBoundingBox] = []

    def draw_onto_image(self, img: NDArray, color: tuple[int, int, int] = (255, 0, 0)) -> None:
        self.box.draw_onto_image(img, color)
        dot_string = "." if self.has_dot else ""
        cv2.putText(
            img,
            str(self.notehead_type.best) + dot_string + str(self.position),
            (int(self.box.center[0]), int(self.box.center[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
            cv2.LINE_AA,
        )
        if self.stem is not None:
            self.stem.draw_onto_image(img, color)
        for beam in self.beams:
            beam.draw_onto_image(img, color)
        for flag in self.flags:
            flag.draw_onto_image(img, color)

    def get_pitch(
        self, clef_type: ClefType | None = None, circle_of_fifth: int | None = None
    ) -> Pitch:
        clef_type = self.clef_type if clef_type is None else clef_type
        circle_of_fifth = self.circle_of_fifth if circle_of_fifth is None else circle_of_fifth
        reference = (
            reference_pitch_g_clef if clef_type == ClefType.TREBLE else reference_pitch_f_clef
        )
        return reference.move_by_position(self.position, self.accidental, circle_of_fifth)

    def _adjust_duration_with_dot(self, duration: int) -> int:
        if self.has_dot:
            return duration * 3 // 2
        else:
            return duration

    def _get_base_duration(self) -> int:
        if self.notehead_type.best == NoteHeadType.HOLLOW:
            if self.stem is None:
                return 4 * constants.duration_of_quarter
            else:
                return 2 * constants.duration_of_quarter
        if self.notehead_type.best == NoteHeadType.SOLID:
            if self.stem is None:
                # TODO that would be an odd result, return a quarter note
                return constants.duration_of_quarter
            elif self.beam_count == 0:
                return constants.duration_of_quarter
            elif self.beam_count > 0:
                # TODO take the note count into account
                return constants.duration_of_quarter // 2
        raise Exception(
            "Unknown notehead type "
            + str(self.notehead_type.best)
            + " "
            + str(self.stem)
            + " "
            + str(self.beam_count)
        )

    def get_duration(self) -> int:
        return self._adjust_duration_with_dot(self._get_base_duration())

    def to_result(self) -> ResultNote:
        return ResultNote(
            self.get_pitch().to_result(), ResultDuration(self.get_duration(), self.has_dot)
        )


class NoteGroup(SymbolOnStaff):
    def __init__(self, notes: list[Note]) -> None:
        super().__init__(notes[0].center)
        self.notes = notes

    def to_result(self) -> ResultNoteGroup:
        return ResultNoteGroup([note.to_result() for note in self.notes])

    def draw_onto_image(self, img: NDArray, color: tuple[int, int, int] = (255, 0, 0)) -> None:
        for note in self.notes:
            note.draw_onto_image(img, color)


class BarLine(SymbolOnStaff):
    def __init__(self, box: RotatedBoundingBox):
        super().__init__(box.center)
        self.box = box

    def draw_onto_image(self, img: NDArray, color: tuple[int, int, int] = (255, 0, 0)) -> None:
        self.box.draw_onto_image(img, color)

    def to_result(self) -> None:
        return None


class Clef(SymbolOnStaff):
    def __init__(self, box: BoundingBox, prediction: Prediction):
        super().__init__(box.center)
        self.box = box
        self.prediction = prediction
        self.accidentals: list[Accidental] = []

    def draw_onto_image(self, img: NDArray, color: tuple[int, int, int] = (255, 0, 0)) -> None:
        self.box.draw_onto_image(img, color)
        cv2.putText(
            img,
            str(self.prediction.best),
            (self.box.box[0], self.box.box[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
            cv2.LINE_AA,
        )

    def get_clef_type(self) -> ClefType:
        if self.prediction.best == "gclef":
            return ClefType.TREBLE
        elif self.prediction.best == "fclef":
            return ClefType.BASS
        else:
            raise Exception("Unknown clef type " + self.prediction.best)

    def get_circle_of_fifth(self) -> int:
        if len(self.accidentals) == 0:
            return 0
        accidental_types = [accidental.get_accidental_type() for accidental in self.accidentals]
        max_accidental = max(accidental_types, key=accidental_types.count)
        if max_accidental == AccidentalType.SHARP:
            return len(self.accidentals)
        elif max_accidental == AccidentalType.FLAT:
            return -len(self.accidentals)
        else:
            # TODO we could likely work with this as it implies that there
            # are wrong detections, might be that naturals more look like sharps
            return 0

    def __str__(self) -> str:
        return str(self.get_clef_type()) + " " + str(self.get_circle_of_fifth())

    def to_result(self) -> ResultClef:
        return ResultClef(self.get_clef_type(), self.get_circle_of_fifth())


class StaffPoint:
    def __init__(self, x: float, y: list[float], angle: float):
        if len(y) != constants.number_of_lines_on_a_staff:
            raise Exception("A staff must consist of exactly 5 lines")
        self.x = x
        self.y = y
        self.angle = angle
        self.average_unit_size = np.mean(np.diff(y))

    def find_position_in_unit_sizes(self, box: AngledBoundingBox) -> int:
        center = box.center
        idx_of_closest_y = int(np.argmin(np.abs([y_value - center[1] for y_value in self.y])))
        distance = self.y[idx_of_closest_y] - center[1]
        distance_in_unit_sizes = round(2 * distance / self.average_unit_size)
        position = (
            2 * (constants.number_of_lines_on_a_staff - idx_of_closest_y)
            + distance_in_unit_sizes
            - 1
        )
        return position  # type: ignore

    def to_bounding_box(self) -> BoundingBox:
        return BoundingBox(
            [int(self.x), int(self.y[0]), int(self.x), int(self.y[-1])], np.array([]), -2
        )


class Staff(DebugDrawable):
    def __init__(self, grid: list[StaffPoint]):
        self.grid = grid
        self.min_x = grid[0].x
        self.max_x = grid[-1].x
        self.min_y = min([min(p.y) for p in grid])
        self.max_y = max([max(p.y) for p in grid])
        self.average_unit_size = np.mean([p.average_unit_size for p in grid])
        self.ledger_lines: list[RotatedBoundingBox] = []
        self.symbols: list[SymbolOnStaff] = []
        self._y_tolerance = constants.max_number_of_ledger_lines * self.average_unit_size

    def is_on_staff_zone(self, item: AngledBoundingBox) -> bool:
        point = self.get_at(item.center[0])
        if point is None:
            return False
        if (
            item.center[1] > point.y[-1] + self._y_tolerance
            or item.center[1] < point.y[0] - self._y_tolerance
        ):
            return False
        return True

    def add_symbol(self, symbol: SymbolOnStaff) -> None:
        self.symbols.append(symbol)

    def add_symbols(self, symbols: list[SymbolOnStaff]) -> None:
        self.symbols.extend(symbols)

    def get_measures(self) -> list[list[Note | NoteGroup]]:
        measures: list[list[Note | NoteGroup]] = []
        current_measure: list[Note | NoteGroup] = []
        symbols_on_measure = self.get_notes() + self.get_note_groups() + self.get_bar_lines()
        for symbol in sorted(symbols_on_measure, key=lambda s: s.center[0]):
            if isinstance(symbol, BarLine):
                measures.append(current_measure)
                current_measure = []
            else:
                current_measure.append(symbol)

        # Add the last measure
        measures.append(current_measure)

        # Remove empty measures
        measures = [measure for measure in measures if len(measure) > 0]
        return measures

    def get_at(self, x: float) -> StaffPoint | None:
        closest_point = min(self.grid, key=lambda p: abs(p.x - x))
        if abs(closest_point.x - x) > constants.staff_position_tolerance:
            return None
        return closest_point

    def y_distance_to(self, point: tuple[float, float]) -> float:
        staff_point = self.get_at(point[0])
        if staff_point is None:
            return 1e10  # Something large to mimic infinity
        return min([abs(y - point[1]) for y in staff_point.y])

    def draw_onto_image(self, img: NDArray, color: tuple[int, int, int] = (255, 0, 0)) -> None:
        for i in range(constants.number_of_lines_on_a_staff):
            for j in range(len(self.grid) - 1):
                p1 = self.grid[j]
                p2 = self.grid[j + 1]
                cv2.line(
                    img, (int(p1.x), int(p1.y[i])), (int(p2.x), int(p2.y[i])), color, thickness=2
                )

    def get_bar_lines(self) -> list[BarLine]:
        result = []
        for symbol in self.symbols:
            if isinstance(symbol, BarLine):
                result.append(symbol)
        return result

    def get_clefs(self) -> list[Clef]:
        result = []
        for symbol in self.symbols:
            if isinstance(symbol, Clef):
                result.append(symbol)
        return result

    def get_notes(self) -> list[Note]:
        result = []
        for symbol in self.symbols:
            if isinstance(symbol, Note):
                result.append(symbol)
        return result

    def get_note_groups(self) -> list[NoteGroup]:
        result = []
        for symbol in self.symbols:
            if isinstance(symbol, NoteGroup):
                result.append(symbol)
        return result

    def get_all_except_notes(self) -> list[SymbolOnStaff]:
        result = []
        for symbol in self.symbols:
            if not isinstance(symbol, Note):
                result.append(symbol)
        return result


class MultiStaff(DebugDrawable):
    """
    A grand staff or a staff with multiple voices.
    """

    def __init__(self, staffs: list[Staff], connections: list[RotatedBoundingBox]) -> None:
        self.staffs = sorted(staffs, key=lambda s: s.min_y)
        self.connections = connections

    def merge(self, other: "MultiStaff") -> "MultiStaff":
        unique_staffs = []
        unique_connections = []
        for staff in self.staffs + other.staffs:
            if staff not in unique_staffs:
                unique_staffs.append(staff)
        for connection in self.connections + other.connections:
            if connection not in unique_connections:
                unique_connections.append(connection)
        return MultiStaff(unique_staffs, unique_connections)

    def break_apart(self) -> list["MultiStaff"]:
        return [MultiStaff([staff], []) for staff in self.staffs]

    def draw_onto_image(self, img: NDArray, color: tuple[int, int, int] = (255, 0, 0)) -> None:
        for staff in self.staffs:
            staff.draw_onto_image(img, color)
        for connection in self.connections:
            connection.draw_onto_image(img, color)
