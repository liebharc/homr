import math

import cv2
import numpy as np

from homr import constants
from homr.debug import Debug
from homr.image_utils import crop_image_and_return_new_top
from homr.model import MultiStaff, Staff
from homr.simple_logging import eprint
from homr.staff_dewarping import StaffDewarping, dewarp_staff_image
from homr.staff_parsing_tromr import parse_staff_tromr
from homr.staff_regions import StaffRegions
from homr.transformer.configs import Config, default_config
from homr.transformer.vocabulary import EncodedSymbol, remove_duplicated_symbols
from homr.type_definitions import NDArray


def _have_all_the_same_number_of_staffs(staffs: list[MultiStaff]) -> bool:
    for staff in staffs:
        if len(staff.staffs) != len(staffs[0].staffs):
            return False
    return True


def _is_close_to_image_top_or_bottom(staff: MultiStaff, image: NDArray) -> bool:
    tolerance = 50.0
    closest_distance_to_top_or_bottom: list[float] = [
        min(s.min_x, image.shape[0] - s.max_x) for s in staff.staffs
    ]
    return min(closest_distance_to_top_or_bottom) < tolerance


def _ensure_same_number_of_staffs(staffs: list[MultiStaff], image: NDArray) -> list[MultiStaff]:
    if _have_all_the_same_number_of_staffs(staffs):
        return staffs
    if len(staffs) > 2:
        if _is_close_to_image_top_or_bottom(
            staffs[0], image
        ) and _have_all_the_same_number_of_staffs(staffs[1:]):
            eprint("Removing first system from all voices, as it has a different number of staffs")
            return staffs[1:]
        if _is_close_to_image_top_or_bottom(
            staffs[-1], image
        ) and _have_all_the_same_number_of_staffs(staffs[:-1]):
            eprint("Removing last system from all voices, as it has a different number of staffs")
            return staffs[:-1]
    result: list[MultiStaff] = []
    for staff in staffs:
        result.extend(staff.break_apart())
    return sorted(result, key=lambda s: s.staffs[0].min_y)


def _get_number_of_voices(staffs: list[MultiStaff]) -> int:
    return len(staffs[0].staffs)


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
        new_image = np.full(  # type: ignore
            (tr_omr_max_height, tr_omr_max_width, 3),
            255,
            dtype=np.uint8,
        )

    x_offset = 0
    tr_omr_max_height_with_margin = tr_omr_max_height - margin_top - margin_bottom
    y_offset = (tr_omr_max_height_with_margin - resized.shape[0]) // 2 + margin_top

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
        is_mostly_dark = np.mean(thresh[y : y + h, x : x + w]) < average_gray_intensity  # type: ignore
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


def apply_clahe(gray_image: NDArray, clip_limit: float = 1.0, kernel_size: int = 8) -> NDArray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(kernel_size, kernel_size))
    gray_image = clahe.apply(gray_image)

    return cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)


def prepare_staff_image(
    debug: Debug, index: int, staff: Staff, staff_image: NDArray, regions: StaffRegions
) -> tuple[NDArray, Staff]:
    region = _calculate_region(staff, regions)
    image_dimensions = get_tr_omr_canvas_size(
        (int(region[3] - region[1]), int(region[2] - region[0]))
    )
    scaling_factor = image_dimensions[1] / (region[3] - region[1])
    staff_image = cv2.resize(
        staff_image,
        (int(staff_image.shape[1] * scaling_factor), int(staff_image.shape[0] * scaling_factor)),
    )
    region = np.round(region * scaling_factor)
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

    staff_image = remove_black_contours_at_edges_of_image(staff_image, staff.average_unit_size)
    staff_image = center_image_on_canvas(staff_image, image_dimensions)
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
    return staff_image, staff


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


def parse_staff_image(
    debug: Debug, index: int, staff: Staff, image: NDArray, regions: StaffRegions, config: Config
) -> list[EncodedSymbol]:
    staff_image, transformed_staff = prepare_staff_image(
        debug, index, staff, image, regions=regions
    )
    eprint("Running TrOmr inference on staff image", index)
    result = parse_staff_tromr(staff_image=staff_image, staff=transformed_staff, config=config)
    if debug.debug:
        result_image = staff_image.copy()
        for i, symbol in enumerate(result):
            center = symbol.coordinates
            if center is None or symbol.rhythm.startswith("chord"):
                continue
            if math.isnan(center[0]) or math.isnan(center[1]):
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


def parse_staffs(
    debug: Debug, staffs: list[MultiStaff], image: NDArray, config: Config, selected_staff: int = -1
) -> list[list[EncodedSymbol]]:
    """
    Dewarps each staff and then runs it through an algorithm which extracts
    the rhythm and pitch information.
    """
    staffs = _ensure_same_number_of_staffs(staffs, image)
    # For simplicity we call every staff in a multi staff a voice,
    # even if it's part of a grand staff.
    number_of_voices = _get_number_of_voices(staffs)
    i = 0
    voices = []
    regions = StaffRegions(staffs)
    for voice in range(number_of_voices):
        staffs_for_voice = [staff.staffs[voice] for staff in staffs]
        result_for_voice = []
        for staff_index, staff in enumerate(staffs_for_voice):
            if selected_staff >= 0 and staff_index != selected_staff:
                eprint("Ignoring staff due to selected_staff argument", i)
                i += 1
                continue
            result_staff = parse_staff_image(debug, i, staff, image, regions, config)
            if len(result_staff) == 0:
                eprint("Skipping empty staff", i)
                i += 1
                continue
            result_staff.append(EncodedSymbol("newline"))
            result_for_voice.extend(result_staff)
            i += 1

        voices.append(remove_duplicated_symbols(result_for_voice))
    return voices
