import argparse
import os
from pathlib import Path

import cv2
import editdistance  # type: ignore

from homr import download_utils, notation_conversions
from homr.simple_logging import eprint
from homr.staff_parsing import add_image_into_tr_omr_canvas
from homr.transformer.configs import Config
from homr.transformer.staff2score import Staff2Score
from training.musescore_svg import get_position_from_multiple_svg_files
from training.transformer.kern_tokens import (
    semantic_to_kern,
    split_kern_file_into_measures,
    split_kern_measures_into_voices,
)


def calc_symbol_error_rate_for_list(dataset: list[str], config: Config) -> None:
    model = Staff2Score(config)
    checkpoint_file = Path(config.filepaths.checkpoint).resolve()
    result_file = str(checkpoint_file).split(".")[0] + "_ser.txt"
    all_sers = []
    i = 0
    total = len(dataset)
    interesting_results: list[tuple[str, str]] = []
    for sample in dataset:
        img_path, semantic_path = sample.strip().split(",")
        expected_str = semantic_to_kern(semantic_path)
        image = cv2.imread(img_path)
        actual_str = model.predict(image)
        print(img_path)
        print(actual_str)
        actual = actual_str.split("\n")
        expected = expected_str.split("\n")
        # actual = sort_chords(actual)
        # expected = sort_chords(expected)
        distance = editdistance.eval(expected, actual)
        ser = distance / len(expected)
        all_sers.append(ser)
        ser = round(100 * ser)
        ser_avg = round(100 * sum(all_sers) / len(all_sers))
        i += 1
        is_staff_with_accidentals = "Polyphonic_tude_No" in img_path and "staff-3" in img_path
        if is_staff_with_accidentals:
            interesting_results.append((str.join(" ", expected), str.join(" ", actual)))
        percentage = round(i / total * 100)
        eprint(f"Progress: {percentage}%, SER: {ser}%, SER avg: {ser_avg}%")

    for result in interesting_results:
        eprint("Expected:", result[0])
        eprint("Actual  :", result[1])

    ser_avg = round(100 * sum(all_sers) / len(all_sers))
    eprint(f"Done, SER avg: {ser_avg}%")

    with open(result_file, "w") as f:
        f.write(f"SER avg: {ser_avg}%\n")


def sort_chords(symbols: list[str]) -> list[str]:
    result = []
    for symbol in symbols:
        result.append(str.join("|", sorted(symbol.split("|"))))
    return result


def index_folder(folder: str, index_file: str) -> None:
    with open(index_file, "w") as index:
        for subfolder in reversed(os.listdir(folder)):
            full_name = os.path.abspath(os.path.join(folder, subfolder))
            if not os.path.isdir(full_name):
                continue
            file = os.path.join(full_name, "music.musicxml")
            kern_file = notation_conversions.musicxml_to_kern(file)
            number_of_voices, time_sig, measures = split_kern_file_into_measures(kern_file)
            time_sig_per_voice, measures_per_voice = split_kern_measures_into_voices(
                number_of_voices, time_sig, measures
            )
            time_sig_per_voice = list(reversed(time_sig_per_voice))
            measures_per_voice = list(reversed(measures_per_voice))
            svg_files = get_position_from_multiple_svg_files(file)
            measures_in_svg = [sum(s.number_of_measures for s in file.staffs) for file in svg_files]
            sum_of_measures_in_xml = number_of_voices * len(measures)
            if sum(measures_in_svg) != sum_of_measures_in_xml:
                eprint(
                    file,
                    "INFO: Number of measures in SVG files",
                    sum(measures_in_svg),
                    "does not match number of measures in XML",
                    sum_of_measures_in_xml,
                )
                continue
            voice = 0
            total_staffs_in_previous_files = 0
            for svg_file in svg_files:
                for staff_idx, staff in enumerate(svg_file.staffs):
                    selected_measures: list[str] = []
                    staffs_per_voice = len(svg_file.staffs) // number_of_voices
                    for _ in range(staff.number_of_measures):
                        selected_measures.append(measures_per_voice[voice].pop(0))

                    expected_file_content = (
                        time_sig_per_voice[voice] + str.join("\n", selected_measures) + "\n"
                    )

                    file_number = (
                        total_staffs_in_previous_files
                        + voice * staffs_per_voice
                        + staff_idx // number_of_voices
                    )

                    file_name = f"staff-{file_number}.jpg"
                    staff_image = os.path.join(full_name, file_name)
                    with open(os.path.join(full_name, f"staff-{file_number}.krn"), "w") as f:
                        f.write(expected_file_content)
                    image = cv2.imread(staff_image)
                    if image is None:
                        continue
                    preprocessed_file_name = os.path.join(full_name, f"staff-pre-{file_number}.jpg")
                    preprocessed = add_image_into_tr_omr_canvas(image)
                    cv2.imwrite(preprocessed_file_name, preprocessed)
                    voice = (voice + 1) % number_of_voices
                    if os.path.exists(staff_image):
                        index.write(
                            preprocessed_file_name
                            + ","
                            + os.path.join(full_name, f"staff-{file_number}.krn")
                            + "\n"
                        )
                total_staffs_in_previous_files += len(svg_file.staffs)


if __name__ == "__main__":
    # ruff: noqa: T201
    parser = argparse.ArgumentParser(description="Calculate symbol error rate.")
    parser.add_argument("checkpoint_file", type=str, help="Path to the checkpoint file.")
    args = parser.parse_args()

    script_location = os.path.dirname(os.path.realpath(__file__))
    data_set_location = os.path.join(script_location, "..", "datasets")
    validation_data_set_location = os.path.join(data_set_location, "validation")
    download_path = os.path.join(data_set_location, "validation.zip")
    download_url = "https://github.com/liebharc/homr/releases/download/datasets/validation.zip"
    if not os.path.exists(validation_data_set_location):
        try:
            eprint("Downloading validation data set")
            download_utils.download_file(download_url, download_path)
            download_utils.unzip_file(download_path, data_set_location)
        finally:
            if os.path.exists(download_path):
                os.remove(download_path)

    index_file = os.path.join(validation_data_set_location, "index.txt")
    if not os.path.exists(index_file):
        index_folder(validation_data_set_location, index_file)

    with open(index_file) as f:
        index = f.readlines()
    config = Config()
    is_dir = os.path.isdir(args.checkpoint_file)
    if is_dir:
        # glob recursive for all model.safetensors file in the directory
        checkpoint_files = list(Path(args.checkpoint_file).rglob("model.safetensors"))
    else:
        checkpoint_files = [Path(args.checkpoint_file)]

    for checkpoint_file in checkpoint_files:
        config.filepaths.checkpoint = str(checkpoint_file)
        calc_symbol_error_rate_for_list(index, config)
