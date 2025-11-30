import argparse
import os
from pathlib import Path
from typing import Any

import cv2
import editdistance

from homr import download_utils
from homr.simple_logging import eprint
from homr.staff_parsing import add_image_into_tr_omr_canvas
from homr.transformer.configs import Config as ConfigTorch
from homr.transformer.vocabulary import EncodedSymbol
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
        actual: list[EncodedSymbol] = model.predict(image)
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
        if ser > 0.5:
            eprint("Expected:", token_lines_to_str(expected))
            eprint("Actual  :", token_lines_to_str(actual))

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


def _ignore_articulation(symbol: EncodedSymbol) -> EncodedSymbol:
    """
    We ignore articulations for now to get results which are compareable
    to previous versions of the model without articulations.
    """
    return EncodedSymbol(symbol.rhythm, symbol.pitch, symbol.lift)


def index_folder(folder: str, index_file: str) -> None:
    result = []
    with open(os.path.join(folder, "index.txt")) as f:
        lines = f.readlines()

    for line in lines:
        img_file, token_file = line.split(",")
        staff_image = cv2.imread(img_file)
        if staff_image is None:
            raise ValueError("Failed to load " + img_file)
        prepared = add_image_into_tr_omr_canvas(staff_image, False, 0, 0)
        processed_path = img_file.replace(".jpg", "-pre.jpg")
        cv2.imwrite(processed_path, prepared)
        result.append(str.join(",", [processed_path, token_file]))

    with open(index_file, "w") as f:
        f.writelines(result)


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
    data_set_location = os.path.abspath(os.path.join(script_location, "..", "datasets"))
    validation_data_set_location = os.path.join(data_set_location, "validation")
    download_path = os.path.join(data_set_location, "validation.zip")
    download_url = (
        "https://github.com/liebharc/homr/releases/download/datasets/validation_tokens.zip"
    )
    if not os.path.exists(validation_data_set_location):
        try:
            eprint("Downloading validation data set")
            download_utils.download_file(download_url, download_path)
            download_utils.unzip_file(download_path, validation_data_set_location)
        finally:
            if os.path.exists(download_path):
                os.remove(download_path)

    index_file = os.path.join(validation_data_set_location, "index_tokens.txt")
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
            checkpoint_files = sorted(
                Path(args.checkpoint_file).rglob("model.safetensors"),
                key=lambda f: f.stat().st_mtime,
                reverse=True,
            )
        else:
            checkpoint_files = [Path(args.checkpoint_file)]

        for checkpoint_file in checkpoint_files:
            config.filepaths.checkpoint = str(checkpoint_file)
            calc_symbol_error_rate_for_list(index, config, onnx=False)


if __name__ == "__main__":
    main()
