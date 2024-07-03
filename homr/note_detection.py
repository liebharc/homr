import cv2.typing as cvt
import numpy as np

from homr import constants
from homr.bounding_boxes import BoundingEllipse, DebugDrawable, RotatedBoundingBox
from homr.model import Note, NoteGroup, Staff, StemDirection, SymbolOnStaff
from homr.simple_logging import eprint
from homr.type_definitions import NDArray


class NoteheadWithStem(DebugDrawable):
    def __init__(
        self,
        notehead: BoundingEllipse,
        stem: RotatedBoundingBox | None,
        stem_direction: StemDirection | None = None,
    ):
        self.notehead = notehead
        self.stem = stem
        self.stem_direction = stem_direction

    def draw_onto_image(self, img: NDArray, color: tuple[int, int, int] = (255, 0, 0)) -> None:
        self.notehead.draw_onto_image(img, color)
        if self.stem is not None:
            self.stem.draw_onto_image(img, color)


def adjust_bbox(bbox: cvt.Rect, noteheads: NDArray) -> cvt.Rect:
    region = noteheads[bbox[1] : bbox[3], bbox[0] : bbox[2]]
    ys, _ = np.where(region > 0)
    if len(ys) == 0:
        # Invalid note. Will be eliminated with zero height.
        return bbox
    top = np.min(ys) + bbox[1] - 1
    bottom = np.max(ys) + bbox[1] + 1
    return (bbox[0], int(top), bbox[2], int(bottom))


def get_center(bbox: cvt.Rect) -> tuple[int, int]:
    cen_y = int(round((bbox[1] + bbox[3]) / 2))
    cen_x = int(round((bbox[0] + bbox[2]) / 2))
    return cen_x, cen_y


def check_bbox_size(bbox: cvt.Rect, noteheads: NDArray, unit_size: float) -> list[cvt.Rect]:
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cen_x, _ = get_center(bbox)
    note_w = constants.NOTEHEAD_SIZE_RATIO * unit_size
    note_h = unit_size

    new_bbox: list[cvt.Rect] = []
    if abs(w - note_w) > abs(w - note_w * 2):
        # Contains at least two notes, one left and one right.
        left_box: cvt.Rect = (bbox[0], bbox[1], cen_x, bbox[3])
        right_box: cvt.Rect = (cen_x, bbox[1], bbox[2], bbox[3])

        # Upper and lower bounds could have changed
        left_box = adjust_bbox(left_box, noteheads)
        right_box = adjust_bbox(right_box, noteheads)

        # Check recursively
        if left_box is not None:
            new_bbox.extend(check_bbox_size(left_box, noteheads, unit_size))
        if right_box is not None:
            new_bbox.extend(check_bbox_size(right_box, noteheads, unit_size))

    # Check height
    if len(new_bbox) > 0:
        tmp_new = []
        for box in new_bbox:
            tmp_new.extend(check_bbox_size(box, noteheads, unit_size))
        new_bbox = tmp_new
    else:
        num_notes = int(round(h / note_h))
        if num_notes > 0:
            sub_h = h // num_notes
            for i in range(num_notes):
                sub_box = (
                    bbox[0],
                    round(bbox[1] + i * sub_h),
                    bbox[2],
                    round(bbox[1] + (i + 1) * sub_h),
                )
                new_bbox.append(sub_box)

    return new_bbox


def split_clumps_of_noteheads(
    notehead: NoteheadWithStem, noteheads: NDArray, staff: Staff
) -> list[NoteheadWithStem]:
    """
    Note heads might be clumped together by the notehead detection algorithm.
    """
    bbox = [
        int(notehead.notehead.top_left[0]),
        int(notehead.notehead.top_left[1]),
        int(notehead.notehead.bottom_right[0]),
        int(notehead.notehead.bottom_right[1]),
    ]
    split_boxes = check_bbox_size(bbox, noteheads, staff.average_unit_size)
    if len(split_boxes) <= 1:
        return [notehead]
    result = []
    for box in split_boxes:
        center = get_center(box)
        size = (box[2] - box[0], box[3] - box[1])
        notehead = NoteheadWithStem(
            BoundingEllipse(
                (center, size, 0), notehead.notehead.contours, notehead.notehead.debug_id
            ),
            notehead.stem,
            notehead.stem_direction,
        )
        result.append(notehead)
    return result


def combine_noteheads_with_stems(
    noteheads: list[BoundingEllipse], stems: list[RotatedBoundingBox]
) -> tuple[list[NoteheadWithStem], list[RotatedBoundingBox]]:
    """
    Combines noteheads with their stems as this tells us
    what vertical lines are stems and which are bar lines.
    """
    result = []
    noteheads = sorted(noteheads, key=lambda notehead: notehead.box[0][1])
    used_stems = set()
    for notehead in noteheads:
        thickened_notehead = notehead.make_box_thicker(15)
        found_stem = False
        for stem in stems:
            if stem.is_overlapping(thickened_notehead):
                is_stem_above = stem.center[1] < notehead.center[1]
                if is_stem_above:
                    direction = StemDirection.UP
                else:
                    direction = StemDirection.DOWN
                result.append(NoteheadWithStem(notehead, stem, direction))
                used_stems.add(stem)
                found_stem = True
                break
        if not found_stem:
            result.append(NoteheadWithStem(notehead, None, None))

    unaccounted_stems_or_bars = [stem for stem in stems if stem not in used_stems]
    return result, unaccounted_stems_or_bars


def _are_notes_likely_a_chord(note1: Note, note2: Note, tolerance: float) -> bool:
    if note1.stem is None or note2.stem is None:
        return abs(note1.center[0] - note2.center[0]) < tolerance
    return abs(note1.stem.center[0] - note2.stem.center[0]) < tolerance


def _create_note_group(notes: list[Note]) -> Note | NoteGroup:
    if len(notes) == 1:
        return notes[0]
    result = NoteGroup(notes)
    return result


def _group_notes_on_staff(staff: Staff) -> None:
    notes = staff.get_notes()
    groups: list[list[Note]] = []
    for note in notes:
        group_found = False
        for group in groups:
            for grouped_note in group:
                if _are_notes_likely_a_chord(
                    note, grouped_note, constants.tolerance_note_grouping(staff.average_unit_size)
                ):
                    group_found = True
                    group.append(note)
                    break
            if group_found:
                break
        if not group_found:
            groups.append([note])
    note_groups: list[SymbolOnStaff] = [_create_note_group(group) for group in groups]
    note_groups.extend(staff.get_all_except_notes())
    sorted_by_x = sorted(note_groups, key=lambda group: group.center[0])
    staff.symbols = sorted_by_x


def add_notes_to_staffs(
    staffs: list[Staff], noteheads: list[NoteheadWithStem], symbols: NDArray, notehead_pred: NDArray
) -> list[Note]:
    result = []
    for staff in staffs:
        for notehead_chunk in noteheads:
            if not staff.is_on_staff_zone(notehead_chunk.notehead):
                continue
            center = notehead_chunk.notehead.center
            point = staff.get_at(center[0])
            if point is None:
                continue
            if (
                notehead_chunk.notehead.size[0] < 0.5 * point.average_unit_size
                or notehead_chunk.notehead.size[1] < 0.5 * point.average_unit_size
            ):
                continue
            for notehead in split_clumps_of_noteheads(notehead_chunk, notehead_pred, staff):
                point = staff.get_at(center[0])
                if point is None:
                    continue
                if (
                    notehead.notehead.size[0] < 0.5 * point.average_unit_size
                    or notehead.notehead.size[0] > 3 * point.average_unit_size
                    or notehead.notehead.size[1] < 0.5 * point.average_unit_size
                    or notehead.notehead.size[1] > 2 * point.average_unit_size
                ):
                    continue
                position = point.find_position_in_unit_sizes(notehead.notehead)
                note = Note(notehead.notehead, position, notehead.stem, notehead.stem_direction)
                result.append(note)
                staff.add_symbol(note)
    number_of_notes = 0
    number_of_note_groups = 0
    for staff in staffs:
        _group_notes_on_staff(staff)
        number_of_notes += len(staff.get_notes())
        number_of_note_groups += len(staff.get_note_groups())
    eprint(
        "After grouping there are",
        number_of_notes,
        "notes and",
        number_of_note_groups,
        "note groups",
    )
    return result
