import argparse
import os
from pathlib import Path

import editdistance  # type: ignore

from homr import download_utils
from homr.transformer.configs import Config
from homr.transformer.staff2score import Staff2Score
from training.transformer.split_merge_symbols import convert_alter_to_accidentals


def calc_symbol_error_rate_for_list(dataset: list[str], config: Config) -> None:
    model = Staff2Score(config)
    checkpoint_file = Path(config.filepaths.checkpoint).resolve()
    result_file = str(checkpoint_file).split(".")[0] + "_ser.txt"
    all_sers = []
    i = 0
    total = len(dataset)
    for sample in dataset:
        img_path, semantic_path = sample.strip().split(",")
        expected_str = convert_alter_to_accidentals(_load_semantic_file(semantic_path))[0].strip()
        actual = model.predict(img_path)[0].split("+")
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
        if i % 10 == 0:
            print("Expected:", expected)
            print("Actual:", actual)
        percentage = round(i / total * 100)
        print(f"Progress: {percentage}%, SER: {ser}%, SER avg: {ser_avg}%")
    ser_avg = round(100 * sum(all_sers) / len(all_sers))
    print(f"Done, SER avg: {ser_avg}%")

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
            print("Downloading validation data set")
            download_utils.download_file(download_url, download_path)
            download_utils.unzip_file(download_path, data_set_location)
        finally:
            if os.path.exists(download_path):
                os.remove(download_path)
    staff_files = Path(validation_data_set_location).rglob("staff-*.jpg")
    index = []
    for staff_file in staff_files:
        semantic_file = staff_file.with_suffix(".semantic")
        index.append(str(staff_file) + "," + str(semantic_file).strip())
    config = Config()
    config.filepaths.checkpoint = args.checkpoint_file
    calc_symbol_error_rate_for_list(index, config)
