import argparse
import glob
import os
import sys

import cv2
import numpy as np

from homr import color_adjust, download_utils
from homr.accidental_detection import add_accidentals_to_staffs
from homr.accidental_rules import maintain_accidentals
from homr.autocrop import autocrop
from homr.bar_line_detection import (
    add_bar_lines_to_staffs,
    detect_bar_lines,
    prepare_bar_line_image,
)
from homr.bounding_boxes import (
    BoundingEllipse,
    RotatedBoundingBox,
    create_bounding_ellipses,
    create_rotated_bounding_boxes,
)
from homr.brace_dot_detection import (
    find_braces_brackets_and_grand_staff_lines,
    prepare_brace_dot_image,
)
from homr.debug import Debug
from homr.model import InputPredictions
from homr.noise_filtering import filter_predictions
from homr.note_detection import add_notes_to_staffs, combine_noteheads_with_stems
from homr.resize import resize_image
from homr.rest_detection import add_rests_to_staffs
from homr.segmentation.config import segnet_path, unet_path
from homr.segmentation.segmentation import segmentation
from homr.simple_logging import eprint
from homr.staff_detection import break_wide_fragments, detect_staff, make_lines_stronger
from homr.staff_parsing import parse_staffs
from homr.title_detection import detect_title
from homr.transformer.configs import default_config
from homr.type_definitions import NDArray
from homr.xml_generator import XmlGeneratorArguments, generate_xml

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class PredictedSymbols:
    def __init__(
        self,
        noteheads: list[BoundingEllipse],
        staff_fragments: list[RotatedBoundingBox],
        clefs_keys: list[RotatedBoundingBox],
        accidentals: list[RotatedBoundingBox],
        stems_rest: list[RotatedBoundingBox],
        bar_lines: list[RotatedBoundingBox],
    ) -> None:
        self.noteheads = noteheads
        self.staff_fragments = staff_fragments
        self.clefs_keys = clefs_keys
        self.accidentals = accidentals
        self.stems_rest = stems_rest
        self.bar_lines = bar_lines


def get_predictions(
    original: NDArray, preprocessed: NDArray, img_path: str, save_cache: bool
) -> InputPredictions:
    result = segmentation(preprocessed, img_path, use_cache=save_cache)
    original_image = cv2.resize(original, (result.staff.shape[1], result.staff.shape[0]))
    preprocessed_image = cv2.resize(preprocessed, (result.staff.shape[1], result.staff.shape[0]))
    return InputPredictions(
        original=original_image,
        preprocessed=preprocessed_image,
        notehead=result.notehead.astype(np.uint8),
        symbols=result.symbols.astype(np.uint8),
        staff=result.staff.astype(np.uint8),
        clefs_keys=result.clefs_keys.astype(np.uint8),
        stems_rest=result.stems_rests.astype(np.uint8),
    )


def replace_extension(path: str, new_extension: str) -> str:
    return os.path.splitext(path)[0] + new_extension


def load_and_preprocess_predictions(
    image_path: str, enable_debug: bool, enable_cache: bool
) -> tuple[InputPredictions, Debug]:
    image = cv2.imread(image_path)
    image = autocrop(image)
    image = resize_image(image)
    preprocessed, _background = color_adjust.color_adjust(image, 40)
    predictions = get_predictions(image, preprocessed, image_path, enable_cache)
    debug = Debug(predictions.original, image_path, enable_debug)
    debug.write_image("color_adjust", preprocessed)

    predictions = filter_predictions(predictions, debug)

    predictions.staff = make_lines_stronger(predictions.staff, (1, 2))
    debug.write_threshold_image("staff", predictions.staff)
    debug.write_threshold_image("symbols", predictions.symbols)
    debug.write_threshold_image("stems_rest", predictions.stems_rest)
    debug.write_threshold_image("notehead", predictions.notehead)
    debug.write_threshold_image("clefs_keys", predictions.clefs_keys)
    return predictions, debug


def predict_symbols(debug: Debug, predictions: InputPredictions) -> PredictedSymbols:
    eprint("Creating bounds for noteheads")
    noteheads = create_bounding_ellipses(predictions.notehead, min_size=(4, 4))
    eprint("Creating bounds for staff_fragments")
    staff_fragments = create_rotated_bounding_boxes(
        predictions.staff, skip_merging=True, min_size=(5, 1), max_size=(10000, 100)
    )

    eprint("Creating bounds for clefs_keys")
    clefs_keys = create_rotated_bounding_boxes(
        predictions.clefs_keys, min_size=(20, 40), max_size=(1000, 1000)
    )
    eprint("Creating bounds for accidentals")
    accidentals = create_rotated_bounding_boxes(
        predictions.clefs_keys, min_size=(5, 5), max_size=(100, 100)
    )
    eprint("Creating bounds for stems_rest")
    stems_rest = create_rotated_bounding_boxes(predictions.stems_rest)
    eprint("Creating bounds for bar_lines")
    bar_line_img = prepare_bar_line_image(predictions.stems_rest)
    debug.write_threshold_image("bar_line_img", bar_line_img)
    bar_lines = create_rotated_bounding_boxes(bar_line_img, skip_merging=True, min_size=(1, 5))

    return PredictedSymbols(
        noteheads, staff_fragments, clefs_keys, accidentals, stems_rest, bar_lines
    )


def process_image(  # noqa: PLR0915
    image_path: str,
    enable_debug: bool,
    enable_cache: bool,
    xml_generator_args: XmlGeneratorArguments,
) -> tuple[str, str, str]:
    eprint("Processing " + image_path)
    predictions, debug = load_and_preprocess_predictions(image_path, enable_debug, enable_cache)
    xml_file = replace_extension(image_path, ".musicxml")
    try:
        eprint("Loaded segmentation")
        symbols = predict_symbols(debug, predictions)
        eprint("Predicted symbols")

        symbols.staff_fragments = break_wide_fragments(symbols.staff_fragments)
        debug.write_bounding_boxes("staff_fragments", symbols.staff_fragments)
        eprint("Found " + str(len(symbols.staff_fragments)) + " staff line fragments")

        noteheads_with_stems, likely_bar_or_rests_lines = combine_noteheads_with_stems(
            symbols.noteheads, symbols.stems_rest
        )
        debug.write_bounding_boxes_alternating_colors("notehead_with_stems", noteheads_with_stems)
        eprint("Found " + str(len(noteheads_with_stems)) + " noteheads")
        if len(noteheads_with_stems) == 0:
            raise Exception("No noteheads found")

        average_note_head_height = float(
            np.median([notehead.notehead.size[1] for notehead in noteheads_with_stems])
        )
        eprint("Average note head height: " + str(average_note_head_height))

        all_noteheads = [notehead.notehead for notehead in noteheads_with_stems]
        all_stems = [note.stem for note in noteheads_with_stems if note.stem is not None]
        bar_lines_or_rests = [
            line
            for line in symbols.bar_lines
            if not line.is_overlapping_with_any(all_noteheads)
            and not line.is_overlapping_with_any(all_stems)
        ]
        bar_line_boxes = detect_bar_lines(bar_lines_or_rests, average_note_head_height)
        debug.write_bounding_boxes_alternating_colors("bar_lines", bar_line_boxes)
        eprint("Found " + str(len(bar_line_boxes)) + " bar lines")

        debug.write_bounding_boxes(
            "anchor_input", symbols.staff_fragments + bar_line_boxes + symbols.clefs_keys
        )
        staffs = detect_staff(
            debug, predictions.staff, symbols.staff_fragments, symbols.clefs_keys, bar_line_boxes
        )
        if len(staffs) == 0:
            raise Exception("No staffs found")
        debug.write_bounding_boxes_alternating_colors("staffs", staffs)

        global_unit_size = np.mean([staff.average_unit_size for staff in staffs])

        bar_lines_found = add_bar_lines_to_staffs(staffs, bar_line_boxes)
        eprint("Found " + str(len(bar_lines_found)) + " bar lines")

        possible_rests = [
            rest for rest in bar_lines_or_rests if not rest.is_overlapping_with_any(bar_line_boxes)
        ]
        rests = add_rests_to_staffs(staffs, possible_rests)
        eprint("Found", len(rests), "rests")

        all_classified = predictions.notehead + predictions.clefs_keys + predictions.stems_rest
        brace_dot_img = prepare_brace_dot_image(
            predictions.symbols, predictions.staff, all_classified, global_unit_size
        )
        debug.write_threshold_image("brace_dot", brace_dot_img)
        brace_dot = create_rotated_bounding_boxes(
            brace_dot_img, skip_merging=True, max_size=(100, -1)
        )

        notes = add_notes_to_staffs(
            staffs, noteheads_with_stems, predictions.symbols, predictions.notehead
        )
        accidentals = add_accidentals_to_staffs(staffs, symbols.accidentals)
        eprint("Found", len(accidentals), "accidentals")

        multi_staffs = find_braces_brackets_and_grand_staff_lines(debug, staffs, brace_dot)
        eprint(
            "Found",
            len(multi_staffs),
            "connected staffs (after merging grand staffs, multiple voices): ",
            [len(staff.staffs) for staff in multi_staffs],
        )

        debug.write_all_bounding_boxes_alternating_colors(
            "notes", multi_staffs, notes, rests, accidentals
        )

        title = detect_title(debug, staffs[0])
        eprint("Found title: " + title)

        result_staffs = parse_staffs(debug, multi_staffs, predictions)

        result_staffs = maintain_accidentals(result_staffs)

        eprint("Writing XML")
        xml = generate_xml(xml_generator_args, result_staffs, title)
        xml.write(xml_file)

        eprint(
            "Finished parsing "
            + str(len(result_staffs))
            + " voices over "
            + str(sum(staff.number_of_new_lines() for staff in result_staffs))
            + " staves"
        )
        teaser_file = replace_extension(image_path, "_teaser.png")
        debug.write_teaser(teaser_file, staffs)
        debug.clean_debug_files_from_previous_runs()

        eprint("Result was written to", xml_file)

        return xml_file, title, teaser_file
    except:
        if os.path.exists(xml_file):
            os.remove(xml_file)
        raise
    finally:
        debug.clean_debug_files_from_previous_runs()


def get_all_image_files_in_folder(folder: str) -> list[str]:
    image_files = []
    for ext in ["png", "jpg", "jpeg"]:
        image_files.extend(glob.glob(os.path.join(folder, "**", f"*.{ext}"), recursive=True))
    without_teasers = [
        img
        for img in image_files
        if "_teaser" not in img
        and "_debug" not in img
        and "_staff" not in img
        and "_tesseract" not in img
    ]
    return sorted(without_teasers)


def download_weights() -> None:
    base_url = "https://github.com/liebharc/homr/releases/download/checkpoints/"
    models = [segnet_path, unet_path, default_config.filepaths.checkpoint]
    missing_models = [model for model in models if not os.path.exists(model)]
    if len(missing_models) == 0:
        return

    eprint("Downloading", len(missing_models), "models - this is only required once")
    for model in missing_models:
        if not os.path.exists(model) or True:
            base_name = os.path.basename(model).split(".")[0]
            eprint(f"Downloading {base_name}")
            try:
                zip_name = base_name + ".zip"
                download_url = base_url + zip_name
                downloaded_zip = os.path.join(os.path.dirname(model), zip_name)
                download_utils.download_file(download_url, downloaded_zip)

                destination_dir = os.path.dirname(model)
                download_utils.unzip_file(downloaded_zip, destination_dir)
            finally:
                if os.path.exists(downloaded_zip):
                    os.remove(downloaded_zip)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="homer", description="An optical music recognition (OMR) system"
    )
    parser.add_argument("image", type=str, nargs="?", help="Path to the image to process")
    parser.add_argument(
        "--init",
        action="store_true",
        help="Downloads the models if they are missing and then exits. "
        + "You don't have to call init before processing images, "
        + "it's only useful if you want to prepare for example a Docker image.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument(
        "--cache", action="store_true", help="Read an existing cache file or create a new one"
    )
    parser.add_argument(
        "--output-large-page",
        action="store_true",
        help="Adds instructions to the musicxml so that it gets rendered on larger pages",
    )
    parser.add_argument(
        "--output-metronome", type=int, help="Adds a metronome to the musicxml with the given bpm"
    )
    parser.add_argument(
        "--output-tempo", type=int, help="Adds a tempo to the musicxml with the given bpm"
    )
    args = parser.parse_args()

    download_weights()
    if args.init:
        eprint("Init finished")
        return

    xml_generator_args = XmlGeneratorArguments(
        args.output_large_page, args.output_metronome, args.output_tempo
    )

    if not args.image:
        eprint("No image provided")
        parser.print_help()
        sys.exit(1)
    elif os.path.isfile(args.image):
        process_image(args.image, args.debug, args.cache, xml_generator_args)
    elif os.path.isdir(args.image):
        image_files = get_all_image_files_in_folder(args.image)
        eprint("Processing", len(image_files), "files:", image_files)
        error_files = []
        for image_file in image_files:
            eprint("=========================================")
            try:
                process_image(image_file, args.debug, args.cache, xml_generator_args)
                eprint("Finished", image_file)
            except Exception as e:
                eprint(f"An error occurred while processing {image_file}: {e}")
                error_files.append(image_file)
        if len(error_files) > 0:
            eprint("Errors occurred while processing the following files:", error_files)
    else:
        raise ValueError(f"{args.image} is not a valid file or directory")


if __name__ == "__main__":
    main()
