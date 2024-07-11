import cv2
import numpy as np

from homr import constants
from homr.bounding_boxes import RotatedBoundingBox
from homr.debug import Debug
from homr.model import MultiStaff, Staff
from homr.type_definitions import NDArray


def prepare_brace_dot_image(
    symbols: NDArray, staff: NDArray, all_other: NDArray, unit_size: float
) -> NDArray:
    brace_dot = cv2.subtract(symbols, staff)
    """
    Remove horizontal lines.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
    out = cv2.erode(brace_dot.astype(np.uint8), kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
    return cv2.dilate(out, kernel)


def _filter_for_tall_elements(
    brace_dot: list[RotatedBoundingBox], staffs: list[Staff]
) -> list[RotatedBoundingBox]:
    """
    We filter elements in two steps:
    1. Use a rough unit size estimate to reduce the data size
    2. Find the closest staff and take its unit size to take warping into account
    """
    rough_unit_size = staffs[0].average_unit_size
    symbols_larger_than_rough_estimate = [
        symbol
        for symbol in brace_dot
        if symbol.size[1] > constants.min_height_for_brace_rough(rough_unit_size)
        and symbol.size[0] < constants.max_width_for_brace_rough(rough_unit_size)
    ]
    result = []
    for symbol in symbols_larger_than_rough_estimate:
        closest_staff = min(staffs, key=lambda staff: staff.y_distance_to(symbol.center))
        unit_size = closest_staff.average_unit_size
        if symbol.size[1] > constants.min_height_for_brace(unit_size):
            result.append(symbol)
    return result


def _get_connections_between_staffs_at_bar_lines(
    staff1: Staff, staff2: Staff, brace_dot: list[RotatedBoundingBox]
) -> list[RotatedBoundingBox]:
    bar_lines1 = staff1.get_bar_lines()
    bar_lines2 = staff2.get_bar_lines()
    result: list[RotatedBoundingBox] = []
    for symbol in brace_dot:
        symbol_thicker = symbol.make_box_thicker(30)
        first_overlapping_staff1 = [
            line for line in bar_lines1 if symbol_thicker.is_overlapping(line.box)
        ]
        first_overlapping_staff2 = [
            line for line in bar_lines2 if symbol_thicker.is_overlapping(line.box)
        ]
        if len(first_overlapping_staff1) >= 1 and len(first_overlapping_staff2) >= 1:
            result.append(symbol)
    return result


def _get_connections_between_staffs_at_clefs(
    staff1: Staff, staff2: Staff, brace_dot: list[RotatedBoundingBox]
) -> list[RotatedBoundingBox]:
    clefs1 = staff1.get_clefs()
    clefs2 = staff2.get_clefs()
    result: list[RotatedBoundingBox] = []
    for symbol in brace_dot:
        symbol_thicker = symbol.make_box_thicker(
            constants.tolerance_for_staff_at_any_point(staff1.average_unit_size)
        )
        first_overlapping_staff1 = [
            clef for clef in clefs1 if symbol_thicker.is_overlapping(clef.box)
        ]
        first_overlapping_staff2 = [
            clef for clef in clefs2 if symbol_thicker.is_overlapping(clef.box)
        ]
        if len(first_overlapping_staff1) >= 1 and len(first_overlapping_staff2) >= 1:
            result.append(symbol)
    return result


def _get_connections_between_staffs_at_lines(
    staff1: Staff, staff2: Staff, brace_dot: list[RotatedBoundingBox]
) -> list[RotatedBoundingBox]:
    result: list[RotatedBoundingBox] = []
    for symbol in brace_dot:
        symbol_thicker = symbol.make_box_thicker(
            constants.tolerance_for_touching_clefs(staff1.average_unit_size)
        )
        point1 = staff1.get_at(symbol.center[0])
        point2 = staff2.get_at(symbol.center[0])
        if point1 is None or point2 is None:
            continue
        if symbol_thicker.is_overlapping(
            point1.to_bounding_box()
        ) and symbol_thicker.is_overlapping(point2.to_bounding_box()):
            result.append(symbol)

    return result


def _get_connections_between_staffs(
    staff1: Staff, staff2: Staff, brace_dot: list[RotatedBoundingBox]
) -> list[RotatedBoundingBox]:
    result = []
    result.extend(_get_connections_between_staffs_at_bar_lines(staff1, staff2, brace_dot))
    result.extend(_get_connections_between_staffs_at_clefs(staff1, staff2, brace_dot))
    result.extend(_get_connections_between_staffs_at_lines(staff1, staff2, brace_dot))
    return result


def _merge_multi_staff_if_they_share_a_staff(staffs: list[MultiStaff]) -> list[MultiStaff]:
    """
    If two MultiStaff objects share a staff, merge them into one MultiStaff object.
    """
    result: list[MultiStaff] = []
    for staff in staffs:
        any_merged = False
        for existing in result:
            if len(set(staff.staffs).intersection(set(existing.staffs))) > 0:
                result.remove(existing)
                result.append(existing.merge(staff))
                any_merged = True
                break
        if not any_merged:
            result.append(staff)
    return result


def find_braces_brackets_and_grand_staff_lines(
    debug: Debug, staffs: list[Staff], brace_dot: list[RotatedBoundingBox]
) -> list[MultiStaff]:
    """
    Connect staffs from multiple voices or grand staffs by searching for brackets and grand staffs.
    """
    brace_dot = _filter_for_tall_elements(brace_dot, staffs)
    result = []
    for i, staff in enumerate(staffs):
        neighbors: list[Staff] = []
        if i > 0:
            neighbors.append(staffs[i - 1])
        if i < len(staffs) - 1:
            neighbors.append(staffs[i + 1])
        any_connected_neighbor = False
        for neighbor in neighbors:
            connections = _get_connections_between_staffs(staff, neighbor, brace_dot)
            if len(connections) >= constants.minimum_connections_to_form_combined_staff:
                result.append(MultiStaff([staff, neighbor], connections))
                any_connected_neighbor = True
        if not any_connected_neighbor:
            result.append(MultiStaff([staff], []))

    return _merge_multi_staff_if_they_share_a_staff(result)


def _is_tiny_square(symbol: RotatedBoundingBox, unit_size: float) -> bool:
    return symbol.size[0] < 0.5 * unit_size and symbol.size[1] < 0.5 * unit_size


def find_dots(
    staffs: list[Staff], brace_dot: list[RotatedBoundingBox], unit_size: float
) -> list[RotatedBoundingBox]:
    brace_dot = [symbol for symbol in brace_dot if _is_tiny_square(symbol, unit_size)]
    result = []
    for staff in staffs:
        for symbol in brace_dot:
            if not staff.is_on_staff_zone(symbol):
                continue
            point = staff.get_at(symbol.center[0])
            if point is None:
                continue
            position = point.find_position_in_unit_sizes(symbol)
            is_even_position = position % 2 == 0
            # Dots are never on staff lines which would be indicated by an odd position
            if not is_even_position:
                continue
            result.append(symbol)

    return result
