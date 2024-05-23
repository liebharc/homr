import cv2
import numpy as np

from homr import constants
from homr.debug import Debug
from homr.image_utils import crop_image_and_return_new_top
from homr.model import Clef, InputPredictions, MultiStaff, NoteGroup, Staff
from homr.results import (
    ResultClef,
    ResultMeasure,
    ResultNote,
    ResultNoteGroup,
    ResultStaff,
    move_pitch_to_clef,
)
from homr.simple_logging import eprint
from homr.staff_dewarping import StaffDewarping, dewarp_staff_image
from homr.staff_parsing_tromr import parse_staff_tromr
from homr.type_definitions import NDArray


def _ensure_same_number_of_staffs(staffs: list[MultiStaff]) -> list[MultiStaff]:
    have_all_the_same_number_of_staffs = True
    for staff in staffs:
        if len(staff.staffs) != len(staffs[0].staffs):
            have_all_the_same_number_of_staffs = False
            break

    if have_all_the_same_number_of_staffs:
        return staffs
    result: list[MultiStaff] = []
    for staff in staffs:
        result.extend(staff.break_apart())
    return sorted(result, key=lambda s: s.staffs[0].min_y)


def _get_number_of_voices(staffs: list[MultiStaff]) -> int:
    return len(staffs[0].staffs)


tr_omr_max_height = 128
tr_omr_max_width = 1280


def get_tr_omr_canvas_size(
    image_shape: NDArray, margin_top: int = 0, margin_bottom: int = 0
) -> NDArray:
    tr_omr_max_height_with_margin = tr_omr_max_height - margin_top - margin_bottom
    tr_omr_ratio = float(tr_omr_max_height_with_margin) / tr_omr_max_width
    height, width = image_shape[:2]

    # Calculate the new size such that it fits exactly into the
    # tr_omr_max_height and tr_omr_max_width
    # while maintaining the aspect ratio of height and width.

    if height / width > tr_omr_ratio:
        # The height is the limiting factor.
        new_shape = [
            int(width / height * tr_omr_max_height_with_margin),
            tr_omr_max_height_with_margin,
        ]
    else:
        # The width is the limiting factor.
        new_shape = [tr_omr_max_width, int(height / width * tr_omr_max_width)]
    return np.array(new_shape)


def center_image_on_canvas(
    image: NDArray, canvas_size: tuple[int, int], margin_top: int = 0, margin_bottom: int = 0
) -> NDArray:

    resized = cv2.resize(image, canvas_size)

    new_image = np.zeros((tr_omr_max_height, tr_omr_max_width, 3), np.uint8)
    new_image[:, :] = (255, 255, 255)

    # Copy the resized image into the center of the new image.
    x_offset = 0
    tr_omr_max_height_with_margin = tr_omr_max_height - margin_top - margin_bottom
    y_offset = (tr_omr_max_height_with_margin - resized.shape[0]) // 2 + margin_top
    new_image[y_offset : y_offset + resized.shape[0], x_offset : x_offset + resized.shape[1]] = (
        resized
    )

    return new_image


def add_image_into_tr_omr_canvas(
    image: NDArray, margin_top: int = 0, margin_bottom: int = 0
) -> NDArray:
    new_shape = get_tr_omr_canvas_size(image.shape, margin_top, margin_bottom)
    new_image = center_image_on_canvas(image, new_shape, margin_top, margin_bottom)
    return new_image


def copy_image_in_center_of_double_the_height_and_white_background(image: NDArray) -> NDArray:
    height, width = image.shape[:2]
    new_image = np.zeros((height * 2, width, 3), np.uint8)
    new_image[:, :] = (255, 255, 255)
    new_image[height // 2 : height // 2 + height, :] = image
    return new_image


def remove_black_contours_at_edges_of_image(bgr: NDArray, unit_size: float) -> NDArray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 97, 255, cv2.THRESH_BINARY)
    thresh = 255 - thresh  # type: ignore
    contours, _hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    threshold = constants.black_spot_removal_threshold(unit_size)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < threshold or h < threshold:
            continue
        is_at_edge_of_image = x == 0 or y == 0 or x + w == bgr.shape[1] or y + h == bgr.shape[0]
        if not is_at_edge_of_image:
            continue
        average_gray_intensity = 127
        is_mostly_dark = np.mean(thresh[y : y + h, x : x + w]) < average_gray_intensity  # type: ignore
        if is_mostly_dark:
            continue
        bgr[y : y + h, x : x + w] = (255, 255, 255)
    return bgr


def prepare_staff_image(  # noqa: PLR0915
    debug: Debug,
    index: int,
    ranges: list[float],
    staff: Staff,
    predictions: InputPredictions,
    perform_dewarp: bool = True,
) -> tuple[str, Staff]:
    centers = [s.center for s in staff.symbols]
    x_values = np.array([c[0] for c in centers])
    y_values = np.array([c[1] for c in centers])

    x_min = min(*x_values, staff.min_x) - staff.average_unit_size
    x_max = max(*x_values, staff.max_x) + staff.average_unit_size
    y_min = min(
        *(y_values - 0.5 * staff.average_unit_size), staff.min_y - 2.5 * staff.average_unit_size
    )
    y_max = max(
        *(y_values + 0.5 * staff.average_unit_size), staff.max_y + 2.5 * staff.average_unit_size
    )
    region = np.array([int(x_min), int(y_min), int(x_max), int(y_max)])
    staff_center = (staff.max_y + staff.min_y) // 2
    y_offsets = []
    staff_above = max([r for r in ranges if r < staff_center], default=-1)
    if staff_above >= 0:
        y_offsets.append(staff.max_y - staff_above)
    staff_below = min([r for r in ranges if r > staff_center], default=-1)
    if staff_below >= 0:
        y_offsets.append(staff_below - staff.min_y)
    if len(y_offsets) > 0:
        min_y_offset = min(y_offsets)
        if (
            min_y_offset > 3 * staff.average_unit_size
            and min_y_offset < 8 * staff.average_unit_size
        ):
            region[1] = int(staff.min_y - min_y_offset)
            region[3] = int(staff.max_y + min_y_offset)
    staff_image = predictions.preprocessed
    image_dimensions = get_tr_omr_canvas_size([region[3] - region[1], region[2] - region[0]])
    scaling_factor = image_dimensions[1] / (region[3] - region[1])
    staff_image = cv2.resize(
        staff_image,
        (int(staff_image.shape[1] * scaling_factor), int(staff_image.shape[0] * scaling_factor)),
    )
    region = np.round(region * scaling_factor)
    if perform_dewarp:
        eprint("Dewarping staff", index)
        region_step1 = np.array(region) + np.array([-10, -50, 10, 50])
        staff_image, top_left = crop_image_and_return_new_top(staff_image, *region_step1)
        region_step2 = np.array(region) - np.array([*top_left, *top_left])
        top_left = top_left / scaling_factor
        staff = _dewarp_staff(staff, None, top_left, scaling_factor)
        dewarp = dewarp_staff_image(staff_image, staff, index, debug)
        staff_image = (255 * dewarp.dewarp(staff_image)).astype(np.uint8)
        staff_image, top_left = crop_image_and_return_new_top(staff_image, *region_step2)
        scaling_factor = 1

        eprint("Dewarping staff", index, "done")
    else:
        staff_image, top_left = crop_image_and_return_new_top(staff_image, *region)

    staff_image = remove_black_contours_at_edges_of_image(staff_image, staff.average_unit_size)
    staff_image = center_image_on_canvas(staff_image, image_dimensions)
    staff_file = debug.write_model_input_image(f"_staff-{index}_input.jpg", staff_image)
    transformed_staff = _dewarp_staff(staff, dewarp, top_left, scaling_factor)
    if debug.debug:
        for symbol in transformed_staff.symbols:
            center = symbol.center
            cv2.circle(staff_image, (int(center[0]), int(center[1])), 5, (0, 0, 255))
            if isinstance(symbol, NoteGroup):
                for note in symbol.notes:
                    cv2.circle(
                        staff_image, (int(note.center[0]), int(note.center[1])), 3, (255, 255, 0)
                    )
            cv2.putText(
                staff_image,
                type(symbol).__name__,
                (int(center[0]), int(center[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (0, 0, 255),
                1,
            )
        debug.write_model_input_image(f"_staff-{index}_debug_annotated.jpg", staff_image)
    return staff_file, transformed_staff


def _dewarp_staff(
    staff: Staff, dewarp: StaffDewarping | None, region: tuple[int, int], scaling: float
) -> Staff:
    """
    Applies the same transformation on the staff coordinates as we did on the image.
    """

    def transform_coordinates(point: tuple[float, float]) -> tuple[float, float]:
        x, y = point
        x -= region[0]
        y -= region[1]
        if dewarp is not None:
            x, y = dewarp.dewarp_point((x, y))
        x = x * scaling
        y = y * scaling
        return x, y

    return staff.transform_coordinates(transform_coordinates)


def move_key_information(staff: Staff, destination: ResultStaff) -> None:
    source = [clef for clef in staff.symbols if isinstance(clef, Clef)]
    dest = [clef for clef in destination.get_symbols() if isinstance(clef, ResultClef)]
    for i in range(min(len(source), len(dest))):
        dest[i].circle_of_fifth = source[i].get_circle_of_fifth()


def parse_staff_image(
    debug: Debug, ranges: list[float], index: int, staff: Staff, predictions: InputPredictions
) -> ResultStaff:
    staff_file, transformed_staff = prepare_staff_image(
        debug, index, ranges, staff, predictions, perform_dewarp=True
    )
    attention_debug = debug.build_attention_debug(staff_file)
    result = parse_staff_tromr(
        staff_file=staff_file,
        staff=transformed_staff,
        debug=attention_debug,
    )
    if attention_debug is not None:
        attention_debug.write()
    return result


def _pick_most_dominant_clef(staff: ResultStaff) -> ResultStaff:  # noqa: C901, PLR0912
    clefs = [clef for clef in staff.get_symbols() if isinstance(clef, ResultClef)]
    clef_types = [clef.clef_type for clef in clefs]
    if len(clef_types) == 0:
        return staff
    most_frequent_clef_type = max(set(clef_types), key=clef_types.count)
    if most_frequent_clef_type is None:
        return staff
    if clef_types.count(most_frequent_clef_type) == 1:
        return staff
    circle_of_fifth = 0  # doesn't matter if we only look at the clef type
    most_frequent_clef = ResultClef(most_frequent_clef_type, circle_of_fifth)
    last_clef_was_originally = None
    for symbol in staff.get_symbols():
        if isinstance(symbol, ResultClef):
            last_clef_was_originally = ResultClef(symbol.clef_type, 0)
            symbol.clef_type = most_frequent_clef_type
        elif isinstance(symbol, ResultNote):
            symbol.pitch = move_pitch_to_clef(
                symbol.pitch, last_clef_was_originally, most_frequent_clef
            )
        elif isinstance(symbol, ResultNoteGroup):
            for note in symbol.notes:
                note.pitch = move_pitch_to_clef(
                    note.pitch, last_clef_was_originally, most_frequent_clef
                )
        elif isinstance(symbol, ResultMeasure):
            for measure_symbol in symbol.symbols:
                if isinstance(symbol, ResultClef):
                    last_clef_was_originally = ResultClef(symbol.clef_type, 0)
                    symbol.clef_type = most_frequent_clef_type
                elif isinstance(measure_symbol, ResultNote):
                    measure_symbol.pitch = move_pitch_to_clef(
                        measure_symbol.pitch, last_clef_was_originally, most_frequent_clef
                    )
                elif isinstance(measure_symbol, ResultNoteGroup):
                    for note in measure_symbol.notes:
                        note.pitch = move_pitch_to_clef(
                            note.pitch, last_clef_was_originally, most_frequent_clef
                        )

    return staff


def _pick_most_dominant_key_signature(staff: ResultStaff) -> ResultStaff:
    clefs = [clef for clef in staff.get_symbols() if isinstance(clef, ResultClef)]
    key_signatures = [clef.circle_of_fifth for clef in clefs]
    if len(key_signatures) == 0:
        return staff
    most_frequent_key = max(set(key_signatures), key=key_signatures.count)
    if most_frequent_key is None:
        return staff
    if key_signatures.count(most_frequent_key) == 1:
        return staff
    for clef in clefs:
        clef.circle_of_fifth = most_frequent_key
    return staff


def _remove_redundant_clefs(measures: list[ResultMeasure]) -> None:
    last_clef = None
    for measure in measures:
        for symbol in measure.symbols:
            if isinstance(symbol, ResultClef):
                if last_clef is not None and last_clef == symbol:
                    measure.remove_symbol(symbol)
                else:
                    last_clef = symbol


def merge_and_clean(staffs: list[ResultStaff]) -> ResultStaff:
    """
    Merge all staffs of a voice into a single staff.
    """
    result = ResultStaff([])
    for staff in staffs:
        result = result.merge(staff)
    _pick_most_dominant_clef(result)
    _pick_most_dominant_key_signature(result)
    _remove_redundant_clefs(result.measures)
    result.measures = [measure for measure in result.measures if not measure.is_empty()]
    return result


def determine_ranges(staffs: list[MultiStaff]) -> list[float]:
    staff_centers = []
    for voice in staffs:
        for staff in voice.staffs:
            staff_centers.append((staff.max_y + staff.min_y) // 2)
    staff_centers = sorted(staff_centers)
    return staff_centers


def remember_new_line(measures: list[ResultMeasure]) -> None:
    if len(measures) > 0:
        measures[0].is_new_line = True


def parse_staffs(
    debug: Debug, staffs: list[MultiStaff], predictions: InputPredictions
) -> list[ResultStaff]:
    """
    Dewarps each staff and then runs it through an algorithm which extracts
    the rhythm and pitch information.
    """
    staffs = _ensure_same_number_of_staffs(staffs)
    # For simplicity we call every staff in a multi staff a voice,
    # even if it's part of a grand staff.
    number_of_voices = _get_number_of_voices(staffs)
    i = 0
    ranges = determine_ranges(staffs)
    voices = []
    for voice in range(number_of_voices):
        staffs_for_voice = [staff.staffs[voice] for staff in staffs]
        result_for_voice = []
        for staff in staffs_for_voice:
            if len(staff.symbols) == 0:
                continue
            result_staff = parse_staff_image(debug, ranges, i, staff, predictions)
            if result_staff.is_empty():
                eprint("Skipping empty staff", i)
                i += 1
                continue
            remember_new_line(result_staff.measures)
            result_for_voice.append(result_staff)
            i += 1
        voices.append(merge_and_clean(result_for_voice))
    return voices
