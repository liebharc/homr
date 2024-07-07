import argparse
import os
from pathlib import Path

import cv2
import editdistance  # type: ignore

from homr import download_utils
from homr.simple_logging import eprint
from homr.transformer.configs import Config
from homr.transformer.staff2score import Staff2Score
from training.musescore_svg import get_position_from_multiple_svg_files
from training.music_xml import group_in_measures, music_xml_to_semantic


def calc_symbol_error_rate_for_list(dataset: list[str], config: Config) -> None:
    model = Staff2Score(config, keep_all_symbols_in_chord=True)
    checkpoint_file = Path(config.filepaths.checkpoint).resolve()
    result_file = str(checkpoint_file).split(".")[0] + "_ser.txt"
    all_sers = []
    i = 0
    total = len(dataset)
    interesting_results: list[tuple[str, str]] = []
    for sample in dataset:
        img_path, semantic_path = sample.strip().split(",")
        expected_str = _load_semantic_file(semantic_path)[0].strip()
        image = cv2.imread(img_path)
        actual = model.predict(image)[0].split("+")
        actual = [
            symbol for symbol in actual if not symbol.startswith("timeSignature")
        ]  # reference data has no time signature
        expected = expected_str.split("+")
        actual = sort_chords(actual)
        expected = sort_chords(expected)
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


def _load_semantic_file(semantic_path: str) -> list[str]:
    with open(semantic_path) as f:
        return f.readlines()


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
            semantic = music_xml_to_semantic(file)
            measures = [group_in_measures(voice) for voice in semantic]
            svg_files = get_position_from_multiple_svg_files(file)
            number_of_voices = len(semantic)
            total_number_of_measures = semantic[0].count("barline")
            measures_in_svg = [sum(s.number_of_measures for s in file.staffs) for file in svg_files]
            sum_of_measures_in_xml = total_number_of_measures * number_of_voices
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
                        selected_measures.append(str.join("+", measures[voice][1].pop(0)))

                    prelude = measures[voice][0]
                    semantic_content = str.join("+", selected_measures) + "\n"

                    if not semantic_content.startswith("clef"):
                        semantic_content = prelude + semantic_content

                    file_number = (
                        total_staffs_in_previous_files
                        + voice * staffs_per_voice
                        + staff_idx // number_of_voices
                    )

                    file_name = f"staff-{file_number}.jpg"
                    staff_image = os.path.join(full_name, file_name)
                    with open(os.path.join(full_name, f"staff-{file_number}.semantic"), "w") as f:
                        f.write(semantic_content)
                    voice = (voice + 1) % number_of_voices
                    if os.path.exists(staff_image):
                        index.write(
                            staff_image
                            + ","
                            + os.path.join(full_name, f"staff-{file_number}.semantic")
                            + "\n"
                        )
                total_staffs_in_previous_files += len(svg_file.staffs)


if __name__ == "__main__":
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
