import cv2
import numpy as np

from homr import constants
from homr.bounding_boxes import RotatedBoundingBox
from homr.debug import Debug
from homr.model import MultiStaff, Staff
from homr.type_definitions import NDArray


def prepare_brace_dot_image(symbols: NDArray, staff: NDArray) -> NDArray:
    brace_dot = cv2.subtract(symbols, staff)
    """
    Remove horizontal lines and Make elements larger.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
    out = cv2.erode(brace_dot.astype(np.uint8), kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 35))
    return cv2.dilate(out, kernel)


def _trim_symbol_to_core_span(symbol: RotatedBoundingBox) -> RotatedBoundingBox:
    """
    Recover a brace/bracket candidate's true vertical span even if it merged
    with unrelated ink during preprocessing. A real brace is wide (it
    bulges), while ink it can accidentally fuse with, such as a neighboring
    staff's clef or a chain of note stems, is comparatively thin. We keep
    only the rows that are close to the blob's own widest row, using a
    threshold relative to that blob (not an absolute pixel count), so this
    works regardless of scan resolution or how much extra ink got attached.
    """
    x, y, w, h = cv2.boundingRect(symbol.contours)
    if h <= 0 or w <= 0:
        return symbol
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [symbol.contours - (x, y)], -1, 255, thickness=cv2.FILLED)
    row_widths = (mask > 0).sum(axis=1)
    max_width = row_widths.max()
    if max_width == 0:
        return symbol
    core_rows = np.where(row_widths >= max_width * constants.brace_core_width_ratio)[0]
    if len(core_rows) == 0:
        return symbol
    core_min_y = y + int(core_rows.min())
    core_max_y = y + int(core_rows.max()) + 1
    core_height = core_max_y - core_min_y
    if core_height <= 0:
        return symbol
    new_box = (
        (symbol.center[0], (core_min_y + core_max_y) / 2),
        (symbol.size[0], core_height),
        symbol.angle,
    )
    return RotatedBoundingBox(new_box, symbol.contours, symbol.debug_id)


def _filter_for_tall_elements(
    brace_dot: list[RotatedBoundingBox], staffs: list[Staff]
) -> list[RotatedBoundingBox]:
    """
    We filter elements in two steps:
    1. Use a rough unit size estimate to reduce the data size
    2. Find the closest staff and take its unit size to take warping into account

    Also drops candidates narrower than min_width_for_brace_dot_candidate here (see its
    definition in homr/constants.py) - a fixed pixel width tied to the dilation kernel that
    built these candidates, not to unit size, so it belongs in this rough/absolute pass
    rather than the per-staff unit-size-relative pass below.
    """
    rough_unit_size = staffs[0].average_unit_size
    symbols_larger_than_rough_estimate = [
        symbol
        for symbol in brace_dot
        if symbol.size[1] > constants.min_height_for_brace_rough(rough_unit_size)
        and symbol.size[0] < constants.max_width_for_brace_rough(rough_unit_size)
        and symbol.size[0] >= constants.min_width_for_brace_dot_candidate
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


def _create_grandstaffs(
    staffs: list[MultiStaff], brace_dot: list[RotatedBoundingBox]
) -> list[MultiStaff]:
    if len(staffs) == 0:
        return staffs
    return [s.create_grandstaffs(brace_dot) for s in staffs]


def find_braces_brackets_and_grand_staff_lines(
    debug: Debug, staffs: list[Staff], brace_dot: list[RotatedBoundingBox]
) -> list[MultiStaff]:
    """
    Connect staffs from multiple voices or grand staffs by searching for brackets and grand staffs.
    """
    brace_dot = [_trim_symbol_to_core_span(symbol) for symbol in brace_dot]
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

    return _create_grandstaffs(_merge_multi_staff_if_they_share_a_staff(result), brace_dot)
