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
        self.min_measure_width = 50

    def add_bar_line(self, bar_line: SvgRectangle) -> None:
        already_present = any(
            abs(bar_line.x - x) < self.min_measure_width for x in self.bar_line_x_positions
        )
        if not already_present:
            self.bar_line_x_positions.add(bar_line.x)

    def merge_staff(self, other: "SvgStaff") -> "SvgStaff":
        if self.number_of_measures != other.number_of_measures:
            raise ValueError("Can't merge staffs with a different number of measures")
        x_min = min(self.x, other.x)
        y_min = min(self.y, other.y)
        x_max = max(self.x + self.width, other.x + other.width)
        y_max = max(self.y + self.height, other.y + other.height)
        result = SvgStaff(x_min, y_min, x_max - x_min, y_max - y_min)
        for pos in self.bar_line_x_positions:
            result.bar_line_x_positions.add(pos)
        return result

    def extend_y_range(self, point: int) -> None:
        """Extend the staff's vertical range to include the given point."""
        top = self.y
        bottom = self.y + self.height
        if point < top:
            self.y = point
            self.height = bottom - point
        elif point > bottom:
            self.height = point - top

    def contains_x_position(self, x: int) -> bool:
        """Check if an x-coordinate falls within this staff's horizontal range."""
        return self.x <= x <= self.x + self.width

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
        self.number_of_measures = sum([staff.number_of_measures for staff in staffs])

    def merge_voice_with_next_one(self, voice: int, number_of_voices: int) -> None:
        """Merge a voice with the next voice within each group.

        Args:
            voice: Index of the voice to merge (0-based within each group)
            number_of_voices: Number of voices per group

        For example, if number_of_voices=3 and voice=1:
        - Merges: staff[1] with staff[2], staff[4] with staff[5], etc.
        - Result: [0, merged(1,2)], [3, merged(4,5)], ...
        """
        if voice < 0 or voice >= number_of_voices - 1:
            raise ValueError(
                f"Voice {voice} cannot be merged with the next voice. "
                f"Valid range is [0, {number_of_voices - 2}] for {number_of_voices} voices."
            )

        new_staffs: list[SvgStaff] = []
        i = 0

        while i < len(self.staffs):
            position_in_group = i % number_of_voices

            if position_in_group == voice:
                # Merge this staff with the next one
                if i + 1 < len(self.staffs):
                    merged_staff = self.staffs[i].merge_staff(self.staffs[i + 1])
                    new_staffs.append(merged_staff)
                    i += 2  # Skip both staffs
                else:
                    eprint(f"Cannot merge staff at index {i}: no next staff available")
                    self.number_of_measures = 0
                    self.staffs = []
                    return
            else:
                # Just append this staff
                new_staffs.append(self.staffs[i])
                i += 1

        self.staffs = new_staffs

        # Recalculate number of measures
        self.number_of_measures = sum(staff.number_of_measures for staff in self.staffs)


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


def _extend_staffs_with_stems(staffs: list[SvgStaff], stems: list[SvgRectangle]) -> None:
    """Efficiently extend staff vertical ranges to include all stems.

    This function finds the closest staff for each stem and extends that staff's
    vertical range to include the stem's top and bottom points.

    Args:
        staffs: List of staffs (assumed to be sorted by y-coordinate)
        stems: List of stems (will be sorted by y-coordinate internally)
    """
    if not staffs or not stems:
        return

    stems_sorted_by_y = sorted(stems, key=lambda s: s.y)

    # Process each stem and find its closest staff
    for stem in stems_sorted_by_y:
        # Find the closest staff by checking which staff's y-range is closest
        # We use the stem's x-coordinate to determine which staff it belongs to
        stem_center_x = stem.x + stem.width // 2
        stem_center_y = stem.y + stem.height // 2

        # Find the best matching staff (one that contains the stem's x position
        # and is closest in y)
        best_staff = None
        best_distance = float("inf")

        for staff in staffs:
            # Check if stem is within the horizontal range of this staff
            if staff.contains_x_position(stem_center_x):
                # Calculate vertical distance from stem center to staff center
                staff_center_y = staff.y + staff.height // 2
                distance = abs(stem_center_y - staff_center_y)

                if distance < best_distance:
                    best_distance = distance
                    best_staff = staff

        # If no staff contains the stem's x position, find the closest one by y
        if best_staff is None:
            for staff in staffs:
                staff_center_y = staff.y + staff.height // 2
                distance = abs(stem_center_y - staff_center_y)

                if distance < best_distance:
                    best_distance = distance
                    best_staff = staff

        # Extend the staff to include both top and bottom of the stem
        if best_staff is not None:
            best_staff.extend_y_range(stem.y)
            best_staff.extend_y_range(stem.y + stem.height)


def get_position_information_from_svg(svg_file: str) -> SvgMusicFile:
    doc = minidom.parse(svg_file)  # noqa: S318
    svg_element = doc.getElementsByTagName("svg")[0]
    width = float(svg_element.getAttribute("width").replace("px", ""))
    height = float(svg_element.getAttribute("height").replace("px", ""))
    lines = doc.getElementsByTagName("polyline")
    staff_lines: list[SvgRectangle] = []
    bar_lines: list[SvgRectangle] = []
    stems: list[SvgRectangle] = []
    for line in lines:
        class_name = line.getAttribute("class")
        if class_name == "StaffLines":
            staff_lines.append(_parse_paths(line.getAttribute("points")))
        if class_name == "BarLine":
            bar_lines.append(_parse_paths(line.getAttribute("points")))
        if class_name == "Stem":
            stems.append(_parse_paths(line.getAttribute("points")))

    combined = _combine_staff_lines_and_bar_lines(staff_lines, bar_lines)

    # Extend staffs using stem information
    _extend_staffs_with_stems(combined, stems)

    return SvgMusicFile(svg_file, width, height, combined)
