import time

import cv2
import numpy as np

from homr import color_adjust, constants
from homr.accidental_rules import maintain_accidentals
from homr.autocrop import autocrop
from homr.bounding_boxes import (
    BoundingBox,
    RotatedBoundingBox,
    create_rotated_bounding_boxes,
)
from homr.debug import Debug
from homr.image_utils import crop_image_and_return_new_top
from homr.resize import resize_image
from homr.results import DurationModifier, ResultChord, ResultClef, ResultStaff
from homr.segmentation import staff_detection
from homr.simple_logging import eprint
from homr.staff_dewarping import dewarp_staff_image
from homr.staff_extraction_fast import construct_staff_from_lines
from homr.staff_parsing import (
    center_image_on_canvas,
    get_tr_omr_canvas_size,
    merge_and_clean,
    remove_black_contours_at_edges_of_image,
)
from homr.tr_omr_parser import TrOMRParser
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
    staffs = create_rotated_bounding_boxes(staff_mask, skip_merging=False, min_size=(200, 50))
    staffs = sorted(staffs, key=lambda staff: staff.top_left[0], reverse=True)
    eprint("Found", len(staffs), "staffs")
    debug.write_bounding_boxes("staffs", staffs)

    if len(staffs) == 0:
        raise ValueError("No staffs found")

    if target_area is not None:
        overlapping_staffs = [staff for staff in staffs if staff.is_overlapping(target_area)]
        eprint(
            "Remaining ",
            len(overlapping_staffs),
            "staffs after filtering for target area",
            target_area.box,
        )

        if overlapping_staffs:
            staff_with_largest_overlap = max(
                overlapping_staffs,
                key=lambda staff: staff.to_bounding_box().get_overlapping_area_size(target_area),
            )
            staffs = [staff_with_largest_overlap]
            eprint("Using staff with the largest target area")
        else:
            staffs = [staffs[0]]
            eprint("Using first staff in the image")

    result_staffs = []
    parser = TrOMRParser()
    for i, staff in enumerate(staffs):
        eprint("Processing staff", (i + 1))
        staff_img = prepare_staff_image(i + 1, image, staff_lines, staff, debug)
        start = time.time()
        staff_result = str.join("+", get_score(staff_img))
        eprint("Processing staff", (i + 1), "done in [s]:", time.time() - start)
        eprint(staff_result)
        staff_parsed = parser.parse_tr_omr_output(staff_result)
        result_staffs.append(staff_parsed)

    merged_staffs = [merge_and_clean(result_staffs, True)]

    merged_staffs = maintain_accidentals(merged_staffs)
    return merged_staffs


def notes_to_tonal_notation(staffs: list[ResultStaff]) -> list[str]:
    """Converts the notes and rests in the staff to tonal notation.
    Tonal notation is useful if you need something short and easy to read to understand
    quickly what has been detected.

    The tonal notation has been extended with clef information and markings for measures and chords.
    """
    results: list[str] = []
    for staff in staffs:
        measures: list[str] = []
        for measure in staff.measures:
            measure_results: list[str] = []
            for symbol in measure.symbols:
                if isinstance(symbol, ResultClef):
                    measure_results.append("clef" + str(symbol.clef_type))
                if isinstance(symbol, ResultChord):
                    chord: list[str] = []
                    for note in symbol.notes:
                        duration_modifier = ""
                        if note.duration.modifier == DurationModifier.DOT:
                            duration_modifier = "."
                        chord.append(
                            str(note.pitch)
                            + "-"
                            + str(note.duration.base_duration / constants.duration_of_quarter / 4)
                            + duration_modifier
                        )
                    measure_results.append(str.join("&", chord))
            measures.append(str.join("+", measure_results))
        results.append(str.join("|", measures))

    return results


def write_to_xml(
    staffs: list[ResultStaff],
    xml_generator_args: XmlGeneratorArguments,
) -> None:

    eprint("Writing XML")
    xml = generate_xml(xml_generator_args, staffs, "Score")
    xml.write("homr_result.musicxml")


def prepare_staff_image(
    index: int,
    staff_image: NDArray,
    staff_lines: NDArray,
    staff: RotatedBoundingBox,
    debug: Debug,
) -> NDArray:
    height = staff.bottom_right[1] - staff.top_left[1]
    region = [
        staff.top_left[0],
        staff.top_left[1] - height / 4,
        staff.bottom_right[0],
        staff.bottom_right[1] + height / 4,
    ]
    staff_image, _ignored = crop_image_and_return_new_top(
        staff_image, region[0], region[1], region[2], region[3]
    )
    staff_lines, _ignored = crop_image_and_return_new_top(
        staff_lines, region[0], region[1], region[2], region[3]
    )
    if debug.debug:
        debug.write_image_with_fixed_suffix(f"_staff-{index}_cropped.jpg", staff_image)
    image_dimensions = get_tr_omr_canvas_size(
        (int(region[3] - region[1]), int(region[2] - region[0]))
    )
    scaling_factor = image_dimensions[1] / (region[3] - region[1])
    staff_image = cv2.resize(
        staff_image,
        (int(staff_image.shape[1] * scaling_factor), int(staff_image.shape[0] * scaling_factor)),
    )
    staff_lines = cv2.resize(
        staff_lines,
        (int(staff_image.shape[1] * scaling_factor), int(staff_image.shape[0] * scaling_factor)),
    )

    staff_image = rotate_image(staff_image, staff.angle)
    staff_lines = rotate_image(staff_lines, staff.angle)
    staff_from_lines = construct_staff_from_lines(staff_lines)
    dewarp = dewarp_staff_image(staff_image, staff_from_lines, index, debug)
    staff_image = (255 * dewarp.dewarp(staff_image)).astype(np.uint8)

    staff_image = remove_black_contours_at_edges_of_image(staff_image, 8)
    staff_image = center_image_on_canvas(staff_image, image_dimensions)

    if debug.debug:
        debug.write_image_with_fixed_suffix(f"_staff-{index}_input.jpg", staff_image)
    return staff_image


def rotate_image(image: NDArray, angle: float) -> NDArray:
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def get_score(image: NDArray) -> list[str]:
    global inference  # noqa: PLW0603
    if inference is None:
        inference = Staff2Score(default_config)
    return inference.predict(image)


if __name__ == "__main__":
    import sys

    image_path = sys.argv[1]
    image = cv2.imread(image_path)

    staffs = process_fast(image, True)
    print(notes_to_tonal_notation(staffs))  # noqa: T201
