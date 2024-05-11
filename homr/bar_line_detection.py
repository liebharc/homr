from . import constants
from .bounding_boxes import RotatedBoundingBox
from .model import BarLine, Staff


def detect_bar_lines(
    bar_lines: list[RotatedBoundingBox], unit_size: float
) -> list[RotatedBoundingBox]:
    """
    Filters the bar line candidates based on their size.
    """
    result = []
    for bar_line in bar_lines:
        if bar_line.size[1] < constants.bar_line_min_height(unit_size):
            continue
        if bar_line.size[0] > constants.bar_line_max_width(unit_size):
            continue
        result.append(bar_line)
    return result


def add_bar_lines_to_staffs(
    staffs: list[Staff], bar_lines: list[RotatedBoundingBox]
) -> list[BarLine]:
    result = []
    for staff in staffs:
        for bar_line in bar_lines:
            if not staff.is_on_staff_zone(bar_line):
                continue
            point = staff.get_at(bar_line.center[0])
            if point is None:
                continue

            if abs(bar_line.top_left[1] - point.y[0]) > constants.bar_line_to_staff_tolerance(
                point.average_unit_size
            ):
                continue

            if abs(bar_line.bottom_left[1] - point.y[-1]) > constants.bar_line_to_staff_tolerance(
                point.average_unit_size
            ):
                continue

            bar_line_symbol = BarLine(bar_line)
            staff.add_symbol(bar_line_symbol)
            result.append(bar_line_symbol)
    return result
