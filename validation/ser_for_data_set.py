import argparse
import os
import re

import cv2
import editdistance  # type: ignore

from homr.simple_logging import eprint
from homr.transformer.configs import Config
from homr.transformer.staff2score import Staff2Score
from training.transformer.data_set_filters import contains_supported_clef


def calc_symbol_error_rate_for_list(dataset: list[str], result_file: str, config: Config) -> None:
    model = Staff2Score(config, keep_all_symbols_in_chord=True)
    i = 0
    total = len(dataset)

    with open(result_file, "w") as result:
        for sample in dataset:
            img_path, semantic_path = sample.strip().split(",")
            expected_str = _load_semantic_file(semantic_path)[0].strip()
            if not contains_supported_clef(expected_str):
                continue

            image = cv2.imread(img_path)
            actual = model.predict(image)[0].split("+")
            expected = re.split(r"\+|\s+", expected_str)
            if "timeSignature" not in expected:
                # reference data has no time signature
                actual = [symbol for symbol in actual if not symbol.startswith("timeSignature")]
            actual = sort_chords(actual)
            expected = sort_chords(expected)
            distance = editdistance.eval(expected, actual)
            ser = distance / len(expected)
            ser = round(100 * ser)
            i += 1
            percentage = round(i / total * 100)
            len_expected = len(expected)
            len_actual = len(actual)
            added_symbols = "+".join(set(actual) - set(expected))
            missing_symbols = "+".join(set(expected) - set(actual))
            result.write(
                f"{img_path},{semantic_path},{ser},{len_expected},{len_actual},{added_symbols},{missing_symbols},{'+'.join(expected)},{'+'.join(actual)}\n"
            )
            eprint(f"Progress: {percentage}%, SER: {ser}%")


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
    parser.add_argument("index_file", type=str, help="Path to the index file.")
    args = parser.parse_args()

    index_file = args.index_file
    if not os.path.exists(index_file):
        raise FileNotFoundError(f"Index file {index_file} does not exist.")

    with open(index_file) as f:
        index = f.readlines()

    config = Config()
    checkpoint_file = str(args.checkpoint_file)
    config.filepaths.checkpoint = str(checkpoint_file)
    result_file = index_file.split(".")[0] + "_ser.txt"
    calc_symbol_error_rate_for_list(index, result_file, config)
    eprint(result_file)
