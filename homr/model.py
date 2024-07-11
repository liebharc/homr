from abc import abstractmethod
from collections.abc import Callable
from enum import Enum

import cv2
import numpy as np
from typing_extensions import Self

from homr import constants
from homr.bounding_boxes import (
    AngledBoundingBox,
    BoundingBox,
    BoundingEllipse,
    DebugDrawable,
    RotatedBoundingBox,
)
from homr.circle_of_fifths import get_circle_of_fifth_notes
from homr.results import ClefType, ResultPitch
from homr.type_definitions import NDArray


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
    def copy(self) -> Self:
        pass

    def transform_coordinates(
        self, transformation: Callable[[tuple[float, float]], tuple[float, float]]
    ) -> Self:
        copy = self.copy()
        copy.center = transformation(self.center)
        return copy

    def calc_distance_to(self, point: tuple[float, float]) -> float:
        return float(np.linalg.norm(np.array(self.center) - np.array(point)))

    def calc_x_distance_to(self, point: tuple[float, float]) -> float:
        return abs(self.center[0] - point[0])


class Accidental(SymbolOnStaff):
    def __init__(self, box: BoundingBox, position: int) -> None:
        super().__init__(box.center)
        self.box = box
        self.position = position

    def draw_onto_image(self, img: NDArray, color: tuple[int, int, int] = (255, 0, 0)) -> None:
        self.box.draw_onto_image(img, color)
        cv2.putText(
            img,
            "accidental-" + str(self.position),
            (int(self.box.box[0]), int(self.box.box[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
            cv2.LINE_AA,
        )

    def __str__(self) -> str:
        return "Accidental(" + str(self.center) + ")"

    def __repr__(self) -> str:
        return str(self)

    def copy(self) -> "Accidental":
        return Accidental(self.box, self.position)


class Rest(SymbolOnStaff):
    def __init__(self, box: BoundingBox) -> None:
        super().__init__(box.center)
        self.box = box
        self.has_dot = False

    def draw_onto_image(self, img: NDArray, color: tuple[int, int, int] = (255, 0, 0)) -> None:
        self.box.draw_onto_image(img, color)
        cv2.putText(
            img,
            "rest",
            (int(self.box.box[0]), int(self.box.box[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
            cv2.LINE_AA,
        )

    def __str__(self) -> str:
        return "Rest(" + str(self.center) + ")"

    def __repr__(self) -> str:
        return str(self)

    def copy(self) -> "Rest":
        return Rest(self.box)


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

    def move_by_position(self, position: int, circle_of_fifth: int) -> "Pitch":
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

        return Pitch(new_step, alter, new_octave)

    def to_result(self) -> ResultPitch:
        return ResultPitch(self.step, self.octave, self.alter)

    def copy(self) -> "Pitch":
        return Pitch(self.step, self.alter, self.octave)


class Note(SymbolOnStaff):
    def __init__(
        self,
        box: BoundingEllipse,
        position: int,
        stem: RotatedBoundingBox | None,
        stem_direction: StemDirection | None,
    ):
        super().__init__(box.center)
        self.box = box
        self.position = position
        self.has_dot = False
        self.beam_count = 0
        self.stem = stem
        self.clef_type = ClefType.treble()
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
            "note" + dot_string + str(self.position),
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
        reference = clef_type.get_reference_pitch()
        reference_pitch = Pitch(reference.step, reference.alter, reference.octave)
        # Position + 1 as the model uses a higher reference point on the staff
        return reference_pitch.move_by_position(self.position + 1, circle_of_fifth)

    def to_tr_omr_note(self, clef_type: ClefType) -> str:
        pitch = self.get_pitch(clef_type=clef_type).to_result()

        # We have no information about the duration here and default to quarter
        return "note-" + str(pitch) + "_quarter"

    def __str__(self) -> str:
        return "Note(" + str(self.center) + ", " + str(self.position) + ")"

    def __repr__(self) -> str:
        return str(self)

    def copy(self) -> "Note":
        return Note(self.box, self.position, self.stem, self.stem_direction)


class NoteGroup(SymbolOnStaff):
    def __init__(self, notes: list[Note]) -> None:
        average_center = np.mean([note.center for note in notes], axis=0)
        super().__init__(average_center)
        # sort notes by pitch, highest position first
        self.notes = sorted(notes, key=lambda note: note.position, reverse=True)

    def draw_onto_image(self, img: NDArray, color: tuple[int, int, int] = (255, 0, 0)) -> None:
        for note in self.notes:
            note.draw_onto_image(img, color)

    def to_tr_omr_note(self, clef_type: ClefType) -> str:
        return "|".join([note.to_tr_omr_note(clef_type) for note in self.notes])

    def __str__(self) -> str:
        return "NoteGroup(" + str.join(",", [str(note) for note in self.notes]) + ")"

    def __repr__(self) -> str:
        return str(self)

    def copy(self) -> "NoteGroup":
        return NoteGroup([note.copy() for note in self.notes])

    def transform_coordinates(
        self, transformation: Callable[[tuple[float, float]], tuple[float, float]]
    ) -> "NoteGroup":
        return NoteGroup([note.transform_coordinates(transformation) for note in self.notes])


class BarLine(SymbolOnStaff):
    def __init__(self, box: RotatedBoundingBox):
        super().__init__(box.center)
        self.box = box

    def draw_onto_image(self, img: NDArray, color: tuple[int, int, int] = (255, 0, 0)) -> None:
        self.box.draw_onto_image(img, color)

    def __str__(self) -> str:
        return "BarLine(" + str(self.center) + ")"

    def __repr__(self) -> str:
        return str(self)

    def copy(self) -> "BarLine":
        return BarLine(self.box)


class Clef(SymbolOnStaff):
    def __init__(self, box: BoundingBox):
        super().__init__(box.center)
        self.box = box
        self.accidentals: list[Accidental] = []

    def draw_onto_image(self, img: NDArray, color: tuple[int, int, int] = (255, 0, 0)) -> None:
        self.box.draw_onto_image(img, color)
        cv2.putText(
            img,
            "clef",
            (self.box.box[0], self.box.box[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
            cv2.LINE_AA,
        )

    def __str__(self) -> str:
        return "Clef(" + str(self.center) + ")"

    def __repr__(self) -> str:
        return str(self)

    def copy(self) -> "Clef":
        return Clef(self.box)


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

    def transform_coordinates(
        self, transformation: Callable[[tuple[float, float]], tuple[float, float]]
    ) -> "StaffPoint":
        xy = [transformation((self.x, y_value)) for y_value in self.y]
        average_x = np.mean([x for x, _ in xy])
        return StaffPoint(float(average_x), [y for _, y in xy], self.angle)

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
        self.average_unit_size = np.median([p.average_unit_size for p in grid])
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

    def get_accidentals(self) -> list[Accidental]:
        result = []
        for symbol in self.symbols:
            if isinstance(symbol, Accidental):
                result.append(symbol)
        return result

    def get_note_groups(self) -> list[NoteGroup]:
        result = []
        for symbol in self.symbols:
            if isinstance(symbol, NoteGroup):
                result.append(symbol)
        return result

    def get_notes_and_groups(self) -> list[Note | NoteGroup]:
        result = []
        for symbol in self.symbols:
            if isinstance(symbol, Note | NoteGroup):
                result.append(symbol)
        return result

    def get_all_except_notes(self) -> list[SymbolOnStaff]:
        result = []
        for symbol in self.symbols:
            if not isinstance(symbol, Note):
                result.append(symbol)
        return result

    def __str__(self) -> str:
        return "Staff(" + str.join(", ", [str(s) for s in self.symbols]) + ")"

    def __repr__(self) -> str:
        return str(self)

    def copy(self) -> "Staff":
        return Staff(self.grid)

    def transform_coordinates(
        self, transformation: Callable[[tuple[float, float]], tuple[float, float]]
    ) -> "Staff":
        copy = Staff([point.transform_coordinates(transformation) for point in self.grid])
        copy.symbols = [symbol.transform_coordinates(transformation) for symbol in self.symbols]
        return copy


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
