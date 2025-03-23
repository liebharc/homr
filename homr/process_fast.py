import time

import cv2
import numpy as np

from homr import color_adjust
from homr.accidental_rules import maintain_accidentals
from homr.autocrop import autocrop
from homr.bar_line_detection import prepare_bar_line_image
from homr.bounding_boxes import (
    BoundingBox,
    RotatedBoundingBox,
    create_rotated_bounding_boxes,
)
from homr.brace_dot_detection import find_braces_brackets_and_grand_staff_lines
from homr.debug import Debug
from homr.resize import resize_image
from homr.results import ResultStaff
from homr.segmentation import staff_detection
from homr.simple_logging import eprint
from homr.staff_detection import break_wide_fragments, detect_staff
from homr.staff_parsing import parse_staffs
from homr.tonal_generator import notes_to_tonal_notation
from homr.transformer.configs import default_config
from homr.transformer.staff2score import Staff2Score
from homr.type_definitions import NDArray
from homr.xml_generator import XmlGeneratorArguments, generate_xml

inference: Staff2Score | None = None


def process_fast(  # noqa: PLR0915
    image: NDArray, enable_debug: bool, target_area: BoundingBox | None = None
) -> list[ResultStaff]:
    image_original_colors = resize_image(image)
    image_original_colors = autocrop(image_original_colors)
    image, _background = color_adjust.color_adjust(image_original_colors, 40)
    debug = Debug(image, "homr_input.png", enable_debug)
    eprint("Running segmentation")
    start = time.time()
    [staff_mask, bracket_mask, staff_lines] = staff_detection.inference(image)
    debug.write_threshold_image("staff_mask", staff_mask)
    debug.write_threshold_image("bracket_mask", bracket_mask)
    debug.write_threshold_image("staff_lines", staff_lines)
    eprint("Running segmentation - Done in [s]:", time.time() - start)

    eprint("Gettings staffs")
    staff_areas = create_rotated_bounding_boxes(staff_mask, skip_merging=False, min_size=(200, 50))
    staff_areas = sorted(staff_areas, key=lambda staff: staff.top_left[1])
    eprint("Found", len(staff_areas), "staff areas")
    debug.write_bounding_boxes("staff_areas", staff_areas)
    if target_area:
        remaining_area = filter_areas(staff_areas, target_area)
        staff_areas = [remaining_area]
        retain_area = remaining_area.to_bounding_box().increase_size_in_each_dimension(
            50, image.shape
        )
        eprint("Retaining only elements inside", retain_area.box)
        x1, y1, x2, y2 = retain_area.box
        retain_mask = np.zeros_like(staff_mask)
        retain_mask[y1:y2, x1:x2] = 1
        debug.write_threshold_image("retain_mask", retain_mask)
        staff_mask = cv2.bitwise_and(staff_mask, staff_mask, mask=retain_mask)
        bracket_mask = cv2.bitwise_and(bracket_mask, bracket_mask, mask=retain_mask)
        staff_lines = cv2.bitwise_and(staff_lines, staff_lines, mask=retain_mask)
        debug.write_threshold_image("staff_mask", staff_mask)
        debug.write_threshold_image("bracket_mask", bracket_mask)
        debug.write_threshold_image("staff_lines", staff_lines)

    eprint("Creating bounds for staff_fragments")
    staff_fragments = create_rotated_bounding_boxes(
        staff_lines, skip_merging=True, min_size=(2, 1), max_size=(10000, 100)
    )

    staff_fragments = break_wide_fragments(staff_fragments)
    debug.write_bounding_boxes("staff_fragments", staff_fragments)
    eprint("Found " + str(len(staff_fragments)) + " staff line fragments")

    eprint("Creating bounds for bar lines and brackets")
    bracket_mask = prepare_bar_line_image(bracket_mask)
    bar_lines_and_brackets = create_rotated_bounding_boxes(
        bracket_mask, skip_merging=True, min_size=(1, 5)
    )
    debug.write_bounding_boxes("barlines", bar_lines_and_brackets)

    # TODO filter bar lines by height
    staffs = detect_staff(debug, staff_lines, staff_fragments, [], bar_lines_and_brackets)
    debug.write_bounding_boxes_alternating_colors("staffs", staffs)
    eprint("Found", len(staffs), "staffs")

    multi_staffs = find_braces_brackets_and_grand_staff_lines(debug, staffs, bar_lines_and_brackets)
    eprint(
        "Found",
        len(multi_staffs),
        "connected staffs (after merging grand staffs, multiple voices): ",
        [len(staff.staffs) for staff in multi_staffs],
    )

    result_staffs = parse_staffs(debug, multi_staffs, image)

    merged_staffs = maintain_accidentals(result_staffs)
    debug.clean_debug_files_from_previous_runs()
    return merged_staffs


def filter_areas(
    staff_areas: list[RotatedBoundingBox], target_area: BoundingBox
) -> RotatedBoundingBox:
    if target_area is not None:
        overlapping_staffs = [staff for staff in staff_areas if staff.is_overlapping(target_area)]
        eprint(
            "There remain",
            len(overlapping_staffs),
            "staffs after filtering for target area",
            target_area.box,
        )

        if overlapping_staffs:
            staff_with_largest_overlap = max(
                overlapping_staffs,
                key=lambda staff: staff.to_bounding_box().get_overlapping_area_size(target_area),
            )
            eprint("Using staff with the largest target area")
            return staff_with_largest_overlap
        else:
            eprint("Using first staff in the image")
            return staff_areas[0]


def write_to_xml(
    staffs: list[ResultStaff],
    xml_generator_args: XmlGeneratorArguments,
) -> None:

    eprint("Writing XML")
    xml = generate_xml(xml_generator_args, staffs, "Score")
    xml.write("homr_result.musicxml")


def determine_ranges(staffs: list[RotatedBoundingBox]) -> list[float]:
    staff_centers = []
    for staff in staffs:
        staff_centers.append((staff.top_left[1] + staff.bottom_right[1]) // 2)
    staff_centers = sorted(staff_centers)
    return staff_centers


def get_score(image: NDArray) -> list[str]:
    global inference  # noqa: PLW0603
    if inference is None:
        inference = Staff2Score(default_config)
    return inference.predict(image)


if __name__ == "__main__":
    import sys

    image_path = sys.argv[1]
    image = cv2.imread(image_path)

    staffs = process_fast(image, True, BoundingBox((0, 0, 0, 0), np.array([])))
    print(notes_to_tonal_notation(staffs))  # noqa: T201
