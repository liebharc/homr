import itertools
import math

import cv2
import numpy as np

from homr import constants
from homr.debug import Debug
from homr.image_utils import crop_image_and_return_new_top
from homr.model import MultiStaff, Staff
from homr.point_mapping import PointMapping, chain, identity, undo_crop, undo_resize
from homr.simple_logging import eprint
from homr.staff_dewarping import StaffDewarping, dewarp_staff_image
from homr.staff_parsing_tromr import parse_staff_tromr
from homr.staff_regions import StaffRegions
from homr.system_repair import _layout
from homr.transformer.configs import Config, default_config
from homr.transformer.vocabulary import EncodedSymbol, remove_duplicated_symbols
from homr.type_definitions import NDArray

tr_omr_max_height = default_config.max_height
tr_omr_max_width = default_config.max_width


def get_tr_omr_canvas_size(
    image_shape: tuple[int, ...], margin_top: int = 0, margin_bottom: int = 0
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


def _canvas_y_offset(resized_height: int, margin_top: int = 0, margin_bottom: int = 0) -> int:
    tr_omr_max_height_with_margin = tr_omr_max_height - margin_top - margin_bottom
    return (tr_omr_max_height_with_margin - resized_height) // 2 + margin_top


def center_image_on_canvas(
    image: NDArray, canvas_size: NDArray, margin_top: int = 0, margin_bottom: int = 0
) -> NDArray:
    is_grayscale = image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1)

    resized = cv2.resize(image, canvas_size)  # type: ignore

    if is_grayscale:
        new_image = np.full(
            (tr_omr_max_height, tr_omr_max_width),
            255,
            dtype=np.uint8,
        )
    else:
        new_image = np.full(
            (tr_omr_max_height, tr_omr_max_width, 3),
            255,
            dtype=np.uint8,
        )

    x_offset = 0
    y_offset = _canvas_y_offset(resized.shape[0], margin_top, margin_bottom)

    new_image[
        y_offset : y_offset + resized.shape[0],
        x_offset : x_offset + resized.shape[1],
    ] = resized

    return new_image


def add_image_into_tr_omr_canvas(image: NDArray) -> NDArray:
    new_shape = get_tr_omr_canvas_size(image.shape)
    new_image = center_image_on_canvas(image, new_shape)
    return new_image


def remove_black_contours_at_edges_of_image(gray: NDArray, unit_size: float) -> NDArray:
    _, thresh = cv2.threshold(gray, 97, 255, cv2.THRESH_BINARY)
    thresh = 255 - thresh
    contours, _hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    threshold = constants.black_spot_removal_threshold(unit_size)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < threshold or h < threshold:
            continue
        is_at_edge_of_image = x == 0 or y == 0 or x + w == gray.shape[1] or y + h == gray.shape[0]
        if not is_at_edge_of_image:
            continue
        average_gray_intensity = 127
        is_mostly_dark = np.mean(thresh[y : y + h, x : x + w]) < average_gray_intensity
        if is_mostly_dark:
            continue
        gray[y : y + h, x : x + w] = 255
    return gray


def _calculate_region(staff: Staff, regions: StaffRegions) -> NDArray:
    x_min = staff.min_x - 2 * staff.average_unit_size
    x_max = staff.max_x + 2 * staff.average_unit_size
    y_min = max(
        staff.min_y - 4 * staff.average_unit_size,
        regions.get_start_of_closest_staff_above(staff.min_y),
    )
    y_max = min(
        staff.max_y + 4 * staff.average_unit_size,
        regions.get_start_of_closest_staff_below(staff.max_y),
    )
    return np.array([int(x_min), int(y_min), int(x_max), int(y_max)])


def prepare_staff_image(
    debug: Debug, index: int, staff: Staff, staff_image: NDArray, regions: StaffRegions
) -> tuple[NDArray, Staff, PointMapping]:
    """
    Returns the staff image for the transformer, the staff in the coordinates of that image
    and a mapping from the coordinates of that image back to the coordinates of the page.
    """
    region = _calculate_region(staff, regions)
    image_dimensions = get_tr_omr_canvas_size(
        (int(region[3] - region[1]), int(region[2] - region[0]))
    )
    scaling_factor = image_dimensions[1] / (region[3] - region[1])
    page_shape = staff_image.shape
    staff_image = cv2.resize(
        staff_image,
        (int(staff_image.shape[1] * scaling_factor), int(staff_image.shape[0] * scaling_factor)),
    )
    undo_scaling = undo_resize(page_shape, staff_image.shape)
    region = np.round(region * scaling_factor)
    eprint("Dewarping staff", index)
    region_step1 = np.array(region) + np.array([-10, -50, 10, 50])
    staff_image, top_left = crop_image_and_return_new_top(staff_image, *region_step1)
    undo_crop1 = undo_crop(*top_left)
    region_step2 = np.array(region) - np.array([*top_left, *top_left])
    top_left = top_left / scaling_factor
    staff = _dewarp_staff(staff, None, top_left, scaling_factor)
    dewarp = dewarp_staff_image(staff_image, staff, index, debug)
    staff_image = dewarp.dewarp(staff_image)
    staff_image, top_left = crop_image_and_return_new_top(staff_image, *region_step2)
    undo_crop2 = undo_crop(*top_left)
    scaling_factor = 1

    eprint("Dewarping staff", index, "done")

    staff_image = remove_black_contours_at_edges_of_image(staff_image, staff.average_unit_size)
    before_canvas_shape = staff_image.shape
    staff_image = center_image_on_canvas(staff_image, image_dimensions)
    canvas_content_shape = (int(image_dimensions[1]), int(image_dimensions[0]))
    to_page = chain(
        undo_crop(0, -_canvas_y_offset(canvas_content_shape[0])),
        undo_resize(before_canvas_shape, canvas_content_shape),
        undo_crop2,
        dewarp.undewarp_point,
        undo_crop1,
        undo_scaling,
    )
    debug.write_image_with_fixed_suffix(f"_staff-{index}_input.jpg", staff_image)
    if debug.debug:
        transformed_staff = _dewarp_staff(staff, dewarp, top_left, scaling_factor)
        transformed_staff_image = staff_image.copy()
        for symbol in transformed_staff.symbols:
            center = symbol.center
            cv2.circle(transformed_staff_image, (int(center[0]), int(center[1])), 5, (0, 0, 255))
            cv2.putText(
                transformed_staff_image,
                type(symbol).__name__,
                (int(center[0]), int(center[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (0, 0, 255),
                1,
            )
        debug.write_image_with_fixed_suffix(
            f"_staff-{index}_debug_annotated.jpg", transformed_staff_image
        )
    return staff_image, staff, to_page


def _dewarp_staff(
    staff: Staff, dewarp: StaffDewarping | None, region: NDArray, scaling: float
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


def _get_symbol_center(symbol: EncodedSymbol) -> tuple[float, float] | None:
    if symbol.coordinates is None or symbol.rhythm.startswith("chord"):
        return None
    center = np.asarray(symbol.coordinates, dtype=np.float64).reshape(-1)
    if len(center) < 2 or math.isnan(center[0]) or math.isnan(center[1]):  # noqa: PLR2004
        return None
    return float(center[0]), float(center[1])


def parse_staff_image(
    debug: Debug,
    index: int,
    staff: Staff,
    image: NDArray,
    regions: StaffRegions,
    config: Config,
    page_to_input_image: PointMapping = identity,
) -> list[EncodedSymbol]:
    staff_image, transformed_staff, staff_to_page = prepare_staff_image(
        debug, index, staff, image, regions=regions
    )
    eprint("Running TrOmr inference on staff image", index)
    result = parse_staff_tromr(staff_image=staff_image, staff=transformed_staff, config=config)
    for symbol in result:
        center = _get_symbol_center(symbol)
        if center is not None:
            symbol.image_coordinates = page_to_input_image(staff_to_page(center))
    if debug.debug:
        result_image = staff_image.copy()
        for i, symbol in enumerate(result):
            center = _get_symbol_center(symbol)
            if center is None:
                continue
            center_int = (int(center[0]), int(center[1]))
            cv2.circle(result_image, center_int, 5, color=(0, 0, 255), thickness=2)
            cv2.putText(
                result_image,
                str(i) + ": " + symbol.rhythm,
                (center_int[0], center_int[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (0, 0, 255),
                1,
            )

        debug.write_image_with_fixed_suffix(f"_staff-{index}_output.jpg", result_image)
    return result


def _slots_in_layout(row: tuple[bool, ...], layout: tuple[bool, ...]) -> tuple[int, ...] | None:
    """
    The staffs of the layout which the staffs of a system are, None if that isn't
    unique, e.g. a system shows only the piano as the voice is hidden.
    """
    matches = [
        slots
        for slots in itertools.combinations(range(len(layout)), len(row))
        if all(
            layout[slot] == is_grandstaff for slot, is_grandstaff in zip(slots, row, strict=True)
        )
    ]
    return matches[0] if len(matches) == 1 else None


def _assign_parts(staffs: list[MultiStaff]) -> list[tuple[int, ...]] | None:
    """
    For every system the parts which its staffs belong to, the parts are the staffs
    of the largest system. None if a system doesn't fit into it unambiguously.
    """
    layouts = {_layout(row) for row in staffs}
    largest = max(layouts, key=len)
    if any(len(layout) == len(largest) and layout != largest for layout in layouts):
        return None
    result = []
    for row in staffs:
        slots = _slots_in_layout(_layout(row), largest)
        if slots is None:
            return None
        result.append(slots)
    return result


def _measure_rests_like(symbols: list[EncodedSymbol], is_grandstaff: bool) -> list[EncodedSymbol]:
    """
    Rests for a staff which is hidden in a system: one per measure of the symbols
    another staff of the system has, with the same barlines.
    """
    positions = ["upper", "lower"] if is_grandstaff else ["upper"]
    result: list[EncodedSymbol] = []
    in_measure = False
    for symbol in symbols:
        if symbol.rhythm.startswith(("note", "rest")) and not in_measure:
            for i, position in enumerate(positions):
                if i > 0:
                    result.append(EncodedSymbol("chord"))
                result.append(EncodedSymbol("measureRest", position=position))
            in_measure = True
        elif "barline" in symbol.rhythm or "repeat" in symbol.rhythm:
            result.append(EncodedSymbol(symbol.rhythm))
            in_measure = False
    return result


def parse_staffs(
    debug: Debug,
    staffs: list[MultiStaff],
    image: NDArray,
    config: Config,
    selected_staff: int = -1,
    page_to_input_image: PointMapping = identity,
) -> list[list[EncodedSymbol]]:
    """
    Dewarps each staff and then runs it through an algorithm which extracts
    the rhythm and pitch information.

    page_to_input_image maps the coordinates of image back to the image the user provided,
    it's used to fill EncodedSymbol.image_coordinates.
    """
    parts = [tuple(range(len(row.staffs))) for row in staffs]
    if len({len(staff.staffs) for staff in staffs}) > 1:
        # The voices are read across systems, a system with fewer staffs hides some
        assigned = _assign_parts(staffs)
        if assigned is None:
            eprint("The layout changes on the page, reading every staff on its own")
            staffs = [single for staff in staffs for single in staff.break_apart()]
            parts = [(0,)] * len(staffs)
        else:
            eprint("Staffs are hidden in some systems, parts of the systems:", assigned)
            parts = assigned
    # For simplicity we call every staff in a multi staff a voice,
    # even if it's part of a grand staff.
    number_of_voices = max(len(staff.staffs) for staff in staffs)
    layout = next(_layout(row) for row in staffs if len(row.staffs) == number_of_voices)
    i = 0
    results: dict[tuple[int, int], list[EncodedSymbol]] = {}
    regions = StaffRegions(staffs)
    for voice in range(number_of_voices):
        for staff_index, row in enumerate(staffs):
            if voice not in parts[staff_index]:
                continue
            if selected_staff >= 0 and staff_index != selected_staff:
                eprint("Ignoring staff due to selected_staff argument", i)
                i += 1
                continue
            staff = row.staffs[parts[staff_index].index(voice)]
            result_staff = parse_staff_image(
                debug, i, staff, image, regions, config, page_to_input_image
            )
            i += 1
            if len(result_staff) == 0:
                eprint("Skipping empty staff", i - 1)
                continue
            results[(staff_index, voice)] = result_staff

    voices = []
    for voice in range(number_of_voices):
        result_for_voice = []
        for staff_index, slots in enumerate(parts):
            symbols = results.get((staff_index, voice))
            shown = [results[(staff_index, v)] for v in slots if (staff_index, v) in results]
            if voice not in slots and len(shown) > 0:
                symbols = _measure_rests_like(shown[0], layout[voice])
            if symbols is None:
                continue
            result_for_voice.extend(symbols)
            result_for_voice.append(EncodedSymbol("newline"))

        voices.append(remove_duplicated_symbols(result_for_voice))
    return voices
