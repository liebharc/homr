import argparse
import os
from pathlib import Path
from typing import Any

import cv2
import editdistance

from homr import download_utils
from homr.simple_logging import eprint
from homr.transformer.configs import Config as ConfigTorch
from homr.transformer.vocabulary import SplitSymbol
from training.convert_lieder import convert_xml_and_svg_file
from training.transformer.training_vocabulary import (
    read_tokens,
    sort_token_chords,
    token_lines_to_str,
)


def calc_symbol_error_rate_for_list(
    dataset: list[str], config: ConfigTorch | None, onnx: bool
) -> None:
    model: Any
    if onnx:
        from homr.transformer.staff2score import Staff2Score as Staff2ScoreOnnx

        model = Staff2ScoreOnnx(False)
        result_file = "onnx_ser.txt"

    else:
        from training.architecture.transformer.staff2score import (
            Staff2Score as Staff2ScoreTorch,
        )

        if config is None:
            raise ValueError("Config must not be None")

        model = Staff2ScoreTorch(config)
        checkpoint_file = Path(config.filepaths.checkpoint).resolve()
        result_file = str(checkpoint_file).split(".")[0] + "_ser.txt"

    all_sers = []
    i = 0
    total = len(dataset)
    interesting_results: list[tuple[str, str]] = []
    for sample in dataset:
        img_path, token_path = sample.strip().split(",")
        expected = read_tokens(token_path)
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError("Failed to read " + img_path)
        actual: list[SplitSymbol] = model.predict(image)
        # Calculate the SER only based on notes and rests
        relevant_symbols = ("note", "rest", "keySignature")
        actual = [
            _ignore_articulation(t)
            for chord in sort_token_chords(actual)
            for t in chord
            if t.rhythm.startswith(relevant_symbols)
        ]
        expected = [
            _ignore_articulation(t)
            for chord in sort_token_chords(expected)
            for t in chord
            if t.rhythm.startswith(relevant_symbols)
        ]
        distance = editdistance.eval(expected, actual)
        ser = distance / len(expected)
        all_sers.append(ser)
        ser = round(100 * ser)
        ser_avg = round(100 * sum(all_sers) / len(all_sers))
        i += 1
        has_usually_high_ser = (
            "Playing_With_Fire_BlackPink" in img_path and "staff-1.jpg" in img_path
        )
        if has_usually_high_ser:
            interesting_results.append((token_lines_to_str(expected), token_lines_to_str(actual)))
        percentage = round(i / total * 100)
        img_path_rel = os.path.relpath(img_path)
        eprint(f"Progress: {percentage}%, SER: {ser}%, SER avg: {ser_avg}% ({img_path_rel})")

    for result in interesting_results:
        eprint("Expected:", result[0])
        eprint("Actual  :", result[1])

    ser_avg = round(100 * sum(all_sers) / len(all_sers))
    eprint(f"Done, SER avg: {ser_avg}%")

    with open(result_file, "w") as f:
        f.write(f"SER avg: {ser_avg}%\n")


def _ignore_articulation(symbol: SplitSymbol) -> SplitSymbol:
    """
    We ignore articulations for now to get results which are compareable
    to previous versions of the model without articulations.
    """
    return SplitSymbol(symbol.rhythm, symbol.pitch, symbol.lift)


def index_folder(folder: str, index_file: str) -> None:
    with open(index_file, "w") as index:
        for subfolder in reversed(os.listdir(folder)):
            full_name = os.path.abspath(os.path.join(folder, subfolder))
            if not os.path.isdir(full_name):
                continue
            file = os.path.join(full_name, "music.musicxml")
            dirname = os.path.dirname(file)
            lines = convert_xml_and_svg_file(
                Path(file), just_token_files=True, fail_if_image_is_missing=False
            )
            for i, line in enumerate(lines):
                token_file = line.split(",")[1].strip()
                staff_image = os.path.join(dirname, f"staff-{i}.jpg")
                if not os.path.exists(staff_image):
                    continue
                index.write(staff_image)
                index.write(",")
                index.write(token_file)
                index.write("\n")


def main() -> None:
    """
    Validates the transformer of homr. It uses a model specified by the user
    with the original inference code located in homr/transformer.
    """
    parser = argparse.ArgumentParser(description="Calculate symbol error rate.")
    # optional: if no path is given it uses the onnx backend with
    # the model located in homr/transformer
    parser.add_argument(
        "checkpoint_file", type=str, default=None, nargs="?", help="Path to the checkpoint file."
    )
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

    if args.checkpoint_file is None:
        # use onnx backend
        eprint("Running with onnx backend")
        calc_symbol_error_rate_for_list(index, None, onnx=True)

    else:
        eprint("Running with torch backend")
        config = ConfigTorch()
        is_dir = os.path.isdir(args.checkpoint_file)
        if is_dir:
            # glob recursive for all model.safetensors file in the directory
            checkpoint_files = list(Path(args.checkpoint_file).rglob("model.safetensors"))
        else:
            checkpoint_files = [Path(args.checkpoint_file)]

        for checkpoint_file in checkpoint_files:
            config.filepaths.checkpoint = str(checkpoint_file)
            calc_symbol_error_rate_for_list(index, config, onnx=False)


if __name__ == "__main__":
    main()
