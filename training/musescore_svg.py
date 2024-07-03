import glob
import math
from xml.dom import minidom

from homr import constants
from homr.simple_logging import eprint


class SvgValidationError(Exception):
    pass


class SvgRectangle:
    def __init__(self, x: int, y: int, width: int, height: int):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def intersects(self, rect2: "SvgRectangle") -> bool:
        # Unpack the rectangles
        x1, y1, width1, height1 = [self.x, self.y, self.width, self.height]
        x2, y2, width2, height2 = [rect2.x, rect2.y, rect2.width, rect2.height]

        if x1 + width1 < x2 or x2 + width2 < x1:
            return False

        if y1 + height1 < y2 or y2 + height2 < y1:
            return False

        return True

    def merge(self, other: "SvgRectangle") -> "SvgRectangle":
        x = min(self.x, other.x)
        y = min(self.y, other.y)
        width = max(self.x + self.width, other.x + other.width) - x
        height = max(self.y + self.height, other.y + other.height) - y
        return SvgRectangle(x, y, width, height)

    def __str__(self) -> str:
        return f"({self.x}, {self.y}, {self.width}, {self.height})"

    def __repr__(self) -> str:
        return self.__str__()


class SvgStaff(SvgRectangle):
    def __init__(self, x: int, y: int, width: int, height: int):
        super().__init__(x, y, width, height)
        self.bar_line_x_positions = set()

        # Add the starting and ending barline
        self.bar_line_x_positions.add(self.x)
        self.bar_line_x_positions.add(self.x + self.width)
        self.min_measure_width = 100

    def add_bar_line(self, bar_line: SvgRectangle) -> None:
        already_present = any(
            abs(bar_line.x - x) < self.min_measure_width for x in self.bar_line_x_positions
        )
        if not already_present:
            self.bar_line_x_positions.add(bar_line.x)

    @property
    def number_of_measures(self) -> int:
        return len(self.bar_line_x_positions) - 1

    def __str__(self) -> str:
        return f"({self.x}, {self.y}, {self.width}, {self.height}): {self.number_of_measures}"

    def __repr__(self) -> str:
        return self.__str__()


class SvgMusicFile:
    def __init__(self, filename: str, width: float, height: float, staffs: list[SvgStaff]):
        self.filename = filename
        self.width = width
        self.height = height
        self.staffs = staffs


def get_position_from_multiple_svg_files(musicxml_file: str) -> list[SvgMusicFile]:
    pattern = musicxml_file.replace(".musicxml", "*.svg")
    svgs = glob.glob(pattern)
    sorted_by_id = sorted(svgs, key=lambda x: int(x.split("-")[-1].split(".")[0]))
    result: list[SvgMusicFile] = []
    for svg in sorted_by_id:
        result.append(get_position_information_from_svg(svg))
    return result


def _parse_paths(points: str) -> SvgRectangle:
    [start, end] = points.split()
    [x1, y1] = start.split(",")
    [x2, y2] = end.split(",")
    return SvgRectangle(
        math.floor(float(x1)),
        math.floor(float(y1)),
        math.ceil(float(x2) - float(x1)),
        math.ceil(float(y2) - float(y1)),
    )


def _combine_staff_lines_and_bar_lines(
    staff_lines: list[SvgRectangle], bar_lines: list[SvgRectangle]
) -> list[SvgStaff]:
    if len(staff_lines) % constants.number_of_lines_on_a_staff != 0:
        eprint("Warning: Staff lines are not a multiple of 5, but is ", len(staff_lines))
        return []
    groups: list[list[SvgRectangle]] = []
    staffs_sorted_by_y = sorted(staff_lines, key=lambda s: s.y)
    for i, staff_line in enumerate(staffs_sorted_by_y):
        if i % constants.number_of_lines_on_a_staff == 0:
            groups.append([])
        groups[-1].append(staff_line)

    merged_groups: list[SvgRectangle] = []
    for group in groups:
        merged_group = group[0]
        for line in group[1:]:
            merged_group = merged_group.merge(line)
        merged_groups.append(merged_group)
    staffs = [SvgStaff(staff.x, staff.y, staff.width, staff.height) for staff in merged_groups]

    for bar_line in bar_lines:
        for staff in staffs:
            if staff.intersects(bar_line):
                staff.add_bar_line(bar_line)

    return staffs


def get_position_information_from_svg(svg_file: str) -> SvgMusicFile:
    doc = minidom.parse(svg_file)  # noqa: S318
    svg_element = doc.getElementsByTagName("svg")[0]
    width = float(svg_element.getAttribute("width").replace("px", ""))
    height = float(svg_element.getAttribute("height").replace("px", ""))
    lines = doc.getElementsByTagName("polyline")
    staff_lines: list[SvgRectangle] = []
    bar_lines: list[SvgRectangle] = []
    for line in lines:
        class_name = line.getAttribute("class")
        if class_name == "StaffLines":
            staff_lines.append(_parse_paths(line.getAttribute("points")))
        if class_name == "BarLine":
            bar_lines.append(_parse_paths(line.getAttribute("points")))
    paths = doc.getElementsByTagName("path")

    number_of_clefs = 0
    for path in paths:
        class_name = path.getAttribute("class")
        if class_name == "Clef":
            number_of_clefs += 1
    combined = _combine_staff_lines_and_bar_lines(staff_lines, bar_lines)
    if len(combined) != number_of_clefs:
        raise SvgValidationError(
            f"Number of clefs {number_of_clefs} does not match the number of staffs {len(combined)}"
        )
    return SvgMusicFile(svg_file, width, height, combined)
