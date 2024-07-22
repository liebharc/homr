from collections.abc import Generator, Iterable

import cv2
import cv2.typing as cvt
import numpy as np
from scipy import signal  # type: ignore

from homr import constants
from homr.bounding_boxes import (
    DebugDrawable,
    RotatedBoundingBox,
    create_rotated_bounding_box,
)
from homr.debug import Debug
from homr.model import Staff, StaffPoint
from homr.simple_logging import eprint
from homr.type_definitions import NDArray


def prepare_staff_image(img: NDArray) -> NDArray:
    """
    Remove small details.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 3))
    out = cv2.erode(img.astype(np.uint8), kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 3))
    return cv2.dilate(out, kernel)


def make_lines_stronger(img: NDArray, kernel_size: tuple[int, int]) -> NDArray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    img = cv2.dilate(img.astype(np.uint8), kernel)
    img = cv2.threshold(img, 0.1, 1, cv2.THRESH_BINARY)[1].astype(np.uint8)
    return img


class StaffLineSegment(DebugDrawable):
    def __init__(self, debug_id: int, staff_fragments: list[RotatedBoundingBox]):
        self.debug_id = debug_id
        self.staff_fragments = sorted(staff_fragments, key=lambda box: box.box[0][0])
        self.min_x = min([line.center[0] - line.size[0] / 2 for line in staff_fragments])
        self.max_x = max([line.center[0] + line.size[0] / 2 for line in staff_fragments])
        self.min_y = min([line.center[1] - line.size[1] / 2 for line in staff_fragments])
        self.max_y = max([line.center[1] + line.size[1] / 2 for line in staff_fragments])

    def merge(self, other: "StaffLineSegment") -> "StaffLineSegment":
        staff_lines = self.staff_fragments.copy()
        for fragment in other.staff_fragments:
            if fragment not in staff_lines:
                staff_lines.append(fragment)
        return StaffLineSegment(self.debug_id, staff_lines)

    def get_at(self, x: float) -> RotatedBoundingBox | None:
        tolerance = constants.staff_line_segment_x_tolerance
        for fragment in self.staff_fragments:
            if (
                x >= fragment.center[0] - fragment.size[0] / 2 - tolerance
                and x <= fragment.center[0] + fragment.size[0] / 2 + tolerance
            ):
                return fragment
        return None

    def is_overlapping(self, other: "StaffLineSegment") -> bool:
        for line in self.staff_fragments:
            for other_line in other.staff_fragments:
                if line.is_overlapping(other_line):
                    return True
        return False

    def draw_onto_image(self, img: NDArray, color: tuple[int, int, int] = (255, 0, 0)) -> None:
        for line in self.staff_fragments:
            line.draw_onto_image(img, color)
        cv2.putText(
            img,
            str(self.debug_id),
            (int(self.staff_fragments[0].box[0][0]), int(self.staff_fragments[0].box[0][1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
            cv2.LINE_AA,
        )


class StaffAnchor(DebugDrawable):
    """
    An anchor is what we call a reliable staff line. That is five parlallel bar lines
    which by their relation to other symbols make it likely that they belong to a staff.
    This is a crucial step as it allows us to then build the complete staff.
    """

    def __init__(self, staff_lines: list[StaffLineSegment], symbol: RotatedBoundingBox):
        self.staff_lines = staff_lines
        y_positions = sorted(
            [
                line.staff_fragments[0].get_center_extrapolated(symbol.center[0])
                for line in staff_lines
            ]
        )
        y_deltas = [abs(y_positions[i] - y_positions[i - 1]) for i in range(1, len(y_positions))]
        self.unit_sizes = y_deltas
        if len(y_deltas) == 0:
            self.average_unit_size = 0.0
        else:
            self.average_unit_size = float(np.mean(y_deltas))
        self.symbol = symbol
        self.max_y = max([line.max_y for line in staff_lines])
        self.min_y = min([line.min_y for line in staff_lines])
        max_number_of_ledger_lines = 5
        self.y_range = range(int(min(y_positions)), int(max(y_positions)))
        self.zone = range(
            int(self.min_y - max_number_of_ledger_lines * self.average_unit_size),
            int(self.max_y + max_number_of_ledger_lines * self.average_unit_size),
        )

    def draw_onto_image(self, img: NDArray, color: tuple[int, int, int] = (0, 255, 0)) -> None:
        for staff in self.staff_lines:
            staff.draw_onto_image(img, color)
        self.symbol.draw_onto_image(img, color)
        x = int(self.symbol.center[0])
        cv2.line(img, [x - 50, self.zone.start], [x + 50, self.zone.start], color, 2)
        cv2.line(img, [x - 50, self.zone.stop], [x + 50, self.zone.stop], color, 2)


def _get_all_contours(lines: list[StaffLineSegment]) -> list[cvt.MatLike]:
    all_fragments: list[RotatedBoundingBox] = []
    for line in lines:
        all_fragments.extend(line.staff_fragments)
    result: list[cvt.MatLike] = []
    for fragment in all_fragments:
        result.extend(fragment.contours)
    return result


class RawStaff(RotatedBoundingBox):
    """
    A raw staff is made of parts which we found on the image. It has gaps and segments start and
    end differently on every staff line.
    """

    def __init__(self, staff_id: int, lines: list[StaffLineSegment], anchors: list[StaffAnchor]):
        contours = _get_all_contours(lines)
        box = cv2.minAreaRect(np.array(contours))
        super().__init__(box, np.concatenate(contours), staff_id)
        self.staff_id = staff_id
        self.lines = lines
        self.anchors = anchors
        self.min_x = self.center[0] - self.size[0] / 2
        self.max_x = self.center[0] + self.size[0] / 2
        self.min_y = self.center[1] - self.size[1] / 2
        self.max_y = self.center[1] + self.size[1] / 2

    def merge(self, other: "RawStaff") -> "RawStaff":
        lines = []
        for i, line in enumerate(self.lines):
            lines.append(other.lines[i].merge(line))
        return RawStaff(self.staff_id, lines, self.anchors + other.anchors)

    def draw_onto_image(self, img: NDArray, color: tuple[int, int, int] = (255, 0, 0)) -> None:
        for line in self.lines:
            line.draw_onto_image(img, color)


def get_staff_for_anchor(anchor: StaffAnchor, staffs: list[RawStaff]) -> RawStaff | None:
    for staff in staffs:
        for i, anchor_line in enumerate(anchor.staff_lines):
            line_requirement = set(anchor_line.staff_fragments)
            if line_requirement.issubset(set(staff.lines[i].staff_fragments)):
                return staff
    return None


def find_raw_staffs_by_connecting_line_fragments(
    anchors: list[StaffAnchor], staff_fragments: list[RotatedBoundingBox]
) -> list[RawStaff]:
    """
    First we build a list of all lines by combining fragments. Then we identify the lines
    which go through the anchors and build a staff from them.
    """
    staffs: list[RawStaff] = []
    staff_id = 0
    for anchor in anchors:
        existing_staff = get_staff_for_anchor(anchor, staffs)
        fragments = [
            fragment
            for fragment in staff_fragments
            if fragment.center[1] >= anchor.zone.start and fragment.center[1] <= anchor.zone.stop
        ]
        connected = connect_staff_lines(fragments, anchor.average_unit_size)
        staff_lines: list[StaffLineSegment] = []
        for anchor_line in anchor.staff_lines:
            line_requirement = set(anchor_line.staff_fragments)
            matching_anchor = [
                line for line in connected if line_requirement.issubset(set(line.staff_fragments))
            ]
            if len(matching_anchor) == 1:
                staff_lines.extend(matching_anchor)
            else:
                staff_lines.append(anchor_line)
        if existing_staff:
            staffs.remove(existing_staff)
            staffs.append(existing_staff.merge(RawStaff(staff_id, staff_lines, [anchor])))
        else:
            staffs.append(RawStaff(staff_id, staff_lines, [anchor]))
        staff_id += 1
    return staffs


def remove_duplicate_staffs(staffs: list[RawStaff]) -> list[RawStaff]:
    """
    Sometimes we find the same staff twice, but fail to connect them.
    This function removes the duplicates.
    """
    result: list[RawStaff] = []
    for staff in staffs:
        overlapping = [other for other in result if staff.is_overlapping(other)]
        if len(overlapping) == 0:
            result.append(staff)
            continue
        staff_duplicates = 2
        if len(overlapping) >= staff_duplicates:
            # Think this through again, for the moment we just take the existing ones
            continue
        if len(overlapping[0].anchors) < len(staff.anchors):
            # The staff with the most anchors is the most reliable one
            result = [s for s in result if s != overlapping[0]]
            result.append(staff)
    return result


def connect_staff_lines(
    staff_lines: list[RotatedBoundingBox], unit_size: float
) -> list[StaffLineSegment]:
    """
    Checks which fragments connect to each other (extrapolation is used to fill gaps)
    and builds a list of StaffLineSegments
    where segments have an increased likelyhood to belong to a staff.
    """
    # With the pop below we are going through the elements from left to right
    sorted_by_right_to_left = sorted(staff_lines, key=lambda box: box.box[0][0], reverse=True)
    result: list[list[RotatedBoundingBox]] = []
    while len(sorted_by_right_to_left) > 0:
        current_staff_line: RotatedBoundingBox = sorted_by_right_to_left.pop()
        is_short_line = current_staff_line.box[1][0] < constants.is_short_line(unit_size)
        if is_short_line:
            continue
        connected = False
        for staff_lines in result:
            if staff_lines[-1].is_overlapping_extrapolated(current_staff_line, unit_size):
                staff_lines.append(current_staff_line)
                connected = True
        if not connected:
            result.append([current_staff_line])
    result_top_to_bottom = sorted(result, key=lambda lines: lines[0].box[0][1])
    connected_lines = [
        StaffLineSegment(i, staff_lines) for i, staff_lines in enumerate(result_top_to_bottom)
    ]
    return connected_lines


def are_lines_crossing(lines: list[StaffLineSegment]) -> bool:
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            if lines[i].is_overlapping(lines[j]):
                return True
    return False


def are_lines_parallel(lines: list[StaffLineSegment], unit_size: float) -> bool:
    all_angles = []
    all_fragments: list[RotatedBoundingBox] = []
    for line in lines:
        for fragment in line.staff_fragments:
            all_angles.append(fragment.angle)
            all_fragments.append(fragment)
    if len(all_angles) == 0:
        return False
    average_angle = np.mean(all_angles)
    for fragment in all_fragments:
        if abs(
            fragment.angle - average_angle
        ) > constants.max_angle_for_lines_to_be_parallel and fragment.size[
            0
        ] > constants.is_short_connected_line(
            unit_size
        ):
            return False
    return True


def begins_or_ends_on_one_staff_line(
    line: RotatedBoundingBox, staff_lines: list[StaffLineSegment], unit_size: float
) -> bool:
    for staff_line in staff_lines:
        fragment = staff_line.get_at(line.center[0])
        if fragment is None:
            continue
        staff_y = fragment.get_center_extrapolated(line.center[0])
        if abs(staff_y - line.center[1]) < unit_size:
            return True
    return False


def find_staff_anchors(
    staff_lines: list[RotatedBoundingBox],
    anchor_symbols: list[RotatedBoundingBox],
    are_clefs: bool = False,
) -> list[StaffAnchor]:
    """
    Finds staff anchors by looking for five parallel bar lines which go
    over or interrupt symbols which are always on staffs
    (and never above or beyond them like notes can be).
    """
    result: list[StaffAnchor] = []

    for center_symbol in anchor_symbols:
        # As the symbol disconnects the staff lines it's the hardest to detect them at the center.
        # Therefore we try to detect them at the left and right side of the symbol as well.
        if are_clefs:
            adjacent = [
                center_symbol,
                center_symbol.move_to_x_horizontal_by(50),
                center_symbol,
                center_symbol.move_to_x_horizontal_by(100),
                center_symbol,
                center_symbol.move_to_x_horizontal_by(150),
            ]
        else:
            adjacent = [
                center_symbol.move_to_x_horizontal_by(-10),
                center_symbol.move_to_x_horizontal_by(-5),
                center_symbol,
                center_symbol.move_to_x_horizontal_by(5),
                center_symbol.move_to_x_horizontal_by(10),
            ]
        for symbol in adjacent:
            estimated_unit_size = round(symbol.size[1] / (constants.number_of_lines_on_a_staff - 1))
            thickened_bar_line = symbol.make_box_taller(estimated_unit_size)
            overlapping_staff_lines = [
                line for line in staff_lines if line.is_intersecting(thickened_bar_line)
            ]
            connected_lines = connect_staff_lines(overlapping_staff_lines, estimated_unit_size)
            if len(connected_lines) > constants.number_of_lines_on_a_staff:
                connected_lines = [
                    line
                    for line in connected_lines
                    if (line.max_x - line.min_x)
                    > constants.is_short_connected_line(estimated_unit_size)
                ]
            if are_lines_crossing(connected_lines) or not are_lines_parallel(
                connected_lines, estimated_unit_size
            ):
                continue
            if not are_clefs and not begins_or_ends_on_one_staff_line(
                symbol, connected_lines, estimated_unit_size
            ):
                continue
            if not len(connected_lines) == constants.number_of_lines_on_a_staff:
                continue

            result.append(StaffAnchor(connected_lines, symbol))
    return result


def resample_staff_segment(  # noqa: C901
    anchor: StaffAnchor, staff: RawStaff, axis_range: Iterable[int]
) -> Generator[StaffPoint, None, None]:
    x = anchor.symbol.center[0]
    line_fragments = [line.staff_fragments[0] for line in anchor.staff_lines]
    centers: list[float] = [line.get_center_extrapolated(x) for line in line_fragments]
    previous_point = StaffPoint(
        x, centers, float(np.mean([line.angle for line in line_fragments]))
    )  # Dummy point at the anchor points
    for x in axis_range:
        lines = [line.get_at(x) for line in staff.lines]
        axis_center = [
            line.get_center_extrapolated(x) if line is not None else None for line in lines
        ]
        center_values = [center for center in axis_center if center is not None]
        incomplete = all(center is None for center in axis_center)
        if incomplete:
            continue
        deltas = np.diff(center_values)
        non_parallel = [delta < 0.5 * anchor.average_unit_size for delta in deltas]
        for i, invalid in enumerate(non_parallel):
            if invalid:
                axis_center[i] = None
                axis_center[i + 1] = None

        for i, previous_y in enumerate(previous_point.y):
            center_value = axis_center[i]
            if (
                center_value is not None
                and abs(center_value - previous_y) > 0.5 * anchor.average_unit_size
            ):
                axis_center[i] = None

        prev_center = -1
        for i in list(range(len(axis_center))) + list(reversed(list(range(len(axis_center))))):
            if axis_center[i] is not None:
                prev_center = i
            elif prev_center >= 0:
                center_value = axis_center[prev_center]
                if center_value is not None:
                    axis_center[i] = center_value + anchor.average_unit_size * (i - prev_center)
        incomplete = any(center is None for center in axis_center)
        if incomplete:
            continue
        angle = float(np.mean([line.angle for line in lines if line is not None]))
        previous_point = StaffPoint(x, [c for c in axis_center if c is not None], angle)
        yield previous_point


def resample_staff(staff: RawStaff) -> Staff:
    anchors_left_to_right = sorted(staff.anchors, key=lambda a: a.symbol.center[0])
    staff_density = 10
    start = (staff.min_x // staff_density) * staff_density
    stop = (staff.max_x // staff_density + 1) * staff_density
    current_anchor = 0
    anchor = anchors_left_to_right[current_anchor]

    grid: list[StaffPoint] = []
    x = start
    for i, anchor in enumerate(anchors_left_to_right):
        to_left = range(int(x), int(anchor.symbol.center[0]), staff_density)
        if i < len(anchors_left_to_right) - 1:
            to_right = range(
                int(anchor.symbol.center[0]),
                int((anchor.symbol.center[0] + anchors_left_to_right[i + 1].symbol.center[0]) / 2),
                staff_density,
            )
        else:
            to_right = range(int(anchor.symbol.center[0]), int(stop), staff_density)
        x = to_right.stop
        grid.extend(reversed(list(resample_staff_segment(anchor, staff, reversed(to_left)))))
        grid.extend(resample_staff_segment(anchor, staff, to_right))

    return Staff(grid)


def resample_staffs(staffs: list[RawStaff]) -> list[Staff]:
    """
    The RawStaffs might have gaps and segments start and end differently on every staff line.
    This function resamples the staffs so for every point of the staff we know the y positions
    of all staff lines. In the end this makes the staffs easier to use in the rest of
    the analysis.
    """
    result = []
    for staff in staffs:
        result.append(resample_staff(staff))
    return result


def range_intersect(r1: range, r2: range) -> range | None:
    return range(max(r1.start, r2.start), min(r1.stop, r2.stop)) or None


def filter_edge_of_vision(staffs: list[Staff], image_shape: tuple[int, ...]) -> list[Staff]:
    """
    Removes staffs which begin in at the right edge or at the lower edge of the image,
    as this are very likely incomplete staffs.
    """
    result = []
    for staff in staffs:
        starts_at_right_edge = staff.min_x > 0.90 * image_shape[1]
        starts_at_bottom_edge = staff.min_y > 0.95 * image_shape[0]
        ends_at_left_edge = staff.max_x < 0.20 * image_shape[1]
        if any([starts_at_right_edge, starts_at_bottom_edge, ends_at_left_edge]):
            continue
        result.append(staff)
    return result


def sort_staffs_top_to_bottom(staffs: list[Staff]) -> list[Staff]:
    return sorted(staffs, key=lambda staff: staff.min_y)


def filter_unusual_anchors(anchors: list[StaffAnchor]) -> list[StaffAnchor]:
    unit_sizes = [anchor.average_unit_size for anchor in anchors]
    average_unit_size = np.mean(unit_sizes)
    unit_size_deviation = np.std(unit_sizes)
    result = []
    for anchor in anchors:
        if abs(anchor.average_unit_size - average_unit_size) > 2 * unit_size_deviation:
            continue
        result.append(anchor)
    return result


def init_zone(clef_anchors: list[StaffAnchor], image_shape: tuple[int, ...]) -> list[range]:
    def make_range(start: float, stop: float) -> range:
        return range(max(int(start), 0), min(int(stop), image_shape[1]))

    # We increase the range only right of the clef as it's the only place
    # where we expect to find staff lines
    margin_right = 10
    ranges = [
        make_range(c.symbol.bottom_left[0], c.symbol.top_right[0] + margin_right)
        for c in clef_anchors
    ]
    ranges = sorted(ranges, key=lambda r: r.start)
    result = []
    for i, r in enumerate(ranges):
        if i == 0:
            result.append(r)
        else:
            overlaps_with_the_last = r.start < result[-1].stop
            if overlaps_with_the_last:
                result[-1] = range(result[-1].start, r.stop)
            else:
                result.append(r)
    return result


def filter_line_peaks(
    peaks: NDArray, norm: NDArray, max_gap_ratio: float = 1.5
) -> tuple[NDArray, list[int]]:
    valid_peaks = np.array([True for _ in range(len(peaks))])

    # Filter by height
    for idx, p in enumerate(peaks):
        max_peak_height = 15
        if norm[p] > max_peak_height:
            valid_peaks[idx] = False

    # Filter by x-axis
    gaps = peaks[1:] - peaks[:-1]
    count = max(5, round(len(peaks) * 0.2))
    approx_unit = np.mean(np.sort(gaps)[:count])
    max_gap = approx_unit * max_gap_ratio

    ext_peaks = [peaks[0] - max_gap - 1] + list(
        peaks
    )  # Prepend an invalid peak for better handling edge case
    groups = []
    group = -1
    for i in range(1, len(ext_peaks)):
        if ext_peaks[i] - ext_peaks[i - 1] > max_gap:
            group += 1
        groups.append(group)

    groups.append(groups[-1] + 1)  # Append an invalid group for better handling edge case
    cur_g = groups[0]
    count = 1
    for idx in range(1, len(groups)):
        group = groups[idx]
        if group == cur_g:
            count += 1
            continue

        if count < constants.number_of_lines_on_a_staff:
            # Incomplete peaks. Also eliminates the top and bottom incomplete staff lines.
            valid_peaks[idx - count : idx] = False
        elif count > constants.number_of_lines_on_a_staff:
            cand_peaks = peaks[idx - count : idx]
            head_part = cand_peaks[: constants.number_of_lines_on_a_staff]
            tail_part = cand_peaks[-constants.number_of_lines_on_a_staff :]
            if sum(norm[head_part]) > sum(norm[tail_part]):
                valid_peaks[idx - count + constants.number_of_lines_on_a_staff : idx] = False
            else:
                valid_peaks[idx - count : idx - constants.number_of_lines_on_a_staff] = False

        cur_g = group
        count = 1
    return valid_peaks, groups[:-1]


def find_horizontal_lines(
    image: NDArray, unit_size: float, line_threshold: float = 0.0
) -> list[list[int]]:
    # Split into zones horizontally and detects staff lines separately.
    count = np.zeros(len(image), dtype=np.uint16)
    sub_ys, _sub_xs = np.where(image > 0)
    for y in sub_ys:
        count[y] += 1

    count = np.insert(count, [0, len(count)], [0, 0])  # Prepend / append
    norm = (count - np.mean(count)) / np.std(count)
    centers, _ = signal.find_peaks(norm, height=line_threshold, distance=unit_size, prominence=1)
    centers -= 1
    norm = norm[1:-1]  # Remove prepend / append
    _valid_centers, groups = filter_line_peaks(centers, norm)
    grouped_centers: dict[int, list[int]] = {}
    for i, center in enumerate(centers):
        group_number = groups[i]
        if group_number not in grouped_centers:
            grouped_centers[group_number] = []
        grouped_centers[group_number].append(center)
    complete_groups = []
    for key in grouped_centers.keys():
        if len(grouped_centers[key]) == constants.number_of_lines_on_a_staff:
            complete_groups.append(sorted(grouped_centers[key]))
    return complete_groups


def predict_other_anchors_from_clefs(
    clef_anchors: list[StaffAnchor], image: NDArray
) -> list[RotatedBoundingBox]:
    if len(clef_anchors) == 0:
        return []
    average_unit_size = float(np.mean([anchor.average_unit_size for anchor in clef_anchors]))
    anchor_symbols = [anchor.symbol for anchor in clef_anchors]
    clef_zones = init_zone(clef_anchors, image.shape)
    result: list[RotatedBoundingBox] = []
    for zone in clef_zones:
        vertical_slice = image[:, zone]
        lines_groups = find_horizontal_lines(vertical_slice, average_unit_size)
        for group in lines_groups:
            min_y = min(group)
            max_y = max(group)
            center_y = (min_y + max_y) / 2
            center_x = zone.start + (zone.stop - zone.start) / 2
            box = ((int(center_x), int(center_y)), (zone.stop - zone.start, int(max_y - min_y)), 0)
            result.append(RotatedBoundingBox(box, np.array([]), 0))
    return [r for r in result if not r.is_overlapping_with_any(anchor_symbols)]


def break_wide_fragments(
    fragments: list[RotatedBoundingBox], limit: int = 100
) -> list[RotatedBoundingBox]:
    """
    Wide fragments (large x dimension) which are curved tend to be filtered by later steps.
    We instead split them into smaller parts, so that the parts better approximate the different
    angles of the curve.
    """
    result = []
    for fragment in fragments:
        remaining_fragment = fragment
        while remaining_fragment.size[0] > limit:
            min_x = min([c[0][0] for c in remaining_fragment.contours])
            contours_left = [c for c in remaining_fragment.contours if c[0][0] < min_x + limit]
            contours_right = [c for c in remaining_fragment.contours if c[0][0] >= min_x + limit]
            # sort by x
            contours_left = sorted(contours_left, key=lambda c: c[0][0])
            contours_right = sorted(contours_right, key=lambda c: c[0][0])
            if len(contours_left) == 0 or len(contours_right) == 0:
                break
            # Make sure that the contours remain connected by adding
            # the first point of the right side to the left side and vice versa
            contours_left.append(contours_right[0])
            contours_right.append(contours_left[-1])
            result.append(
                create_rotated_bounding_box(np.array(contours_left), remaining_fragment.debug_id)
            )
            remaining_fragment = create_rotated_bounding_box(
                np.array(contours_right), remaining_fragment.debug_id
            )
        result.append(remaining_fragment)
    return result


def detect_staff(
    debug: Debug,
    image: NDArray,
    staff_fragments: list[RotatedBoundingBox],
    clefs_keys: list[RotatedBoundingBox],
    likely_bar_or_rests_lines: list[RotatedBoundingBox],
) -> list[Staff]:
    """
    Detect staffs on the image. Staffs can be warped, have gaps and can be interrupted by symbols.
    """
    staff_anchors = find_staff_anchors(staff_fragments, clefs_keys, are_clefs=True)
    eprint("Found " + str(len(staff_anchors)) + " clefs")

    possible_other_clefs = predict_other_anchors_from_clefs(staff_anchors, image)
    eprint("Found " + str(len(possible_other_clefs)) + " possible other clefs")
    staff_anchors.extend(find_staff_anchors(staff_fragments, possible_other_clefs, are_clefs=True))

    staff_anchors.extend(
        find_staff_anchors(staff_fragments, likely_bar_or_rests_lines, are_clefs=False)
    )

    staff_anchors = filter_unusual_anchors(staff_anchors)
    eprint("Found " + str(len(staff_anchors)) + " staff anchors")
    debug.write_bounding_boxes_alternating_colors("staff_anchors", staff_anchors)

    raw_staffs_with_possible_duplicates = find_raw_staffs_by_connecting_line_fragments(
        staff_anchors, staff_fragments
    )
    eprint("Found " + str(len(raw_staffs_with_possible_duplicates)) + " staffs")
    raw_staffs = remove_duplicate_staffs(raw_staffs_with_possible_duplicates)
    if len(raw_staffs_with_possible_duplicates) != len(raw_staffs):
        eprint(
            "Removed "
            + str(len(raw_staffs_with_possible_duplicates) - len(raw_staffs))
            + " duplicate staffs"
        )
    debug.write_bounding_boxes_alternating_colors(
        "raw_staffs", raw_staffs + likely_bar_or_rests_lines + clefs_keys
    )

    staffs = resample_staffs(raw_staffs)

    staffs = filter_edge_of_vision(staffs, image.shape)

    staffs = sort_staffs_top_to_bottom(staffs)

    return staffs
