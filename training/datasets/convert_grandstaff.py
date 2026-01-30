import multiprocessing
import os
import random
from pathlib import Path

import cv2
import numpy as np

from homr.download_utils import download_file, untar_file
from homr.simple_logging import eprint
from homr.staff_parsing import add_image_into_tr_omr_canvas
from homr.type_definitions import NDArray
from training.datasets.humdrum_kern_parser import convert_kern_to_tokens
from training.datasets.musescore_svg import SvgValidationError
from training.transformer.image_utils import add_margin
from training.transformer.training_vocabulary import (
    calc_ratio_of_tuplets,
    token_lines_to_str,
)

script_location = os.path.dirname(os.path.realpath(__file__))
git_root = Path(script_location).parent.parent.absolute()
dataset_root = os.path.join(git_root, "datasets")
grandstaff_root = os.path.join(dataset_root, "grandstaff")
grandstaff_train_index = os.path.join(grandstaff_root, "index.txt")


def _get_dark_pixels_per_row(image: NDArray) -> NDArray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dark_pixels_per_row = np.zeros(gray.shape[0])
    dark_threshold = 200
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            if gray[i, j] < dark_threshold:
                dark_pixels_per_row[i] += 1
    return dark_pixels_per_row


def _create_staff_image(path: str, basename: str) -> str:
    """
    This algorithm is taken from `oemer` staffline extraction algorithm. In this simplified version
    it only works with images which have no distortions.
    """
    image = cv2.imread(path)
    if image is None:
        raise ValueError("Failed to read " + path)
    dark_pixels_per_row = _get_dark_pixels_per_row(image)
    upper_bound, lower_bound = _get_image_bounds(dark_pixels_per_row)
    image = image[upper_bound:-lower_bound]
    pre_processed_image = _prepare_image(image)
    cv2.imwrite(basename + "-pre.jpg", pre_processed_image)
    return basename + "-pre.jpg"


def _prepare_image(image: NDArray) -> NDArray:
    margin_left = random.randint(0, 10)
    margin_right = random.randint(0, 10)
    margin_top = random.randint(10, 30)
    margin_bottom = random.randint(10, 30)
    image = add_margin(image, margin_top, margin_bottom, margin_left, margin_right)
    result = add_image_into_tr_omr_canvas(image)
    return result


def _get_image_bounds(dark_pixels_per_row: NDArray) -> tuple[int, int]:
    white_upper_area_size = 0
    for i in range(dark_pixels_per_row.shape[0]):
        if dark_pixels_per_row[i] > 0:
            break
        white_upper_area_size += 1
    white_lower_area_size = 1
    for i in range(dark_pixels_per_row.shape[0] - 1, 0, -1):
        if dark_pixels_per_row[i] > 0:
            break
        white_lower_area_size += 1
    return white_upper_area_size, white_lower_area_size


def _check_staff_image(path: str, basename: str) -> str:
    """
    This method helps with reprocessing a folder more quickly by skipping
    the image splitting.
    """
    if not os.path.exists(basename + "-pre.jpg"):
        raise ValueError("Image is missing " + path)
    return basename + "-pre.jpg"


def _kern_to_tokens(path: str, basename: str) -> str:
    with open(path) as text_file:
        result = convert_kern_to_tokens(text_file.readlines())

    if calc_ratio_of_tuplets(result) > 0.2:
        return ""

    with open(basename + ".tokens", "w") as f:
        f.write(token_lines_to_str(result))
    return basename + ".tokens"


def _convert_file(path: Path, ony_recreate_token_files: bool = False) -> str:  # noqa: PLR0911
    try:
        basename = str(path).replace(".krn", "")
        image_file = str(path).replace(".krn", ".jpg")
        tokens = _kern_to_tokens(str(path), basename)
        if not tokens:
            return ""
        if ony_recreate_token_files:
            image = _check_staff_image(image_file, basename)
        else:
            image = _create_staff_image(image_file, basename)
        return (
            str(Path(image).relative_to(git_root)) + "," + str(Path(tokens).relative_to(git_root))
        )

    except SvgValidationError:
        return ""
    except Exception as e:
        eprint("Failed to convert ", path, e)
        return ""


def _convert_file_only_tokens(path: Path) -> tuple[Path, str]:
    return path, _convert_file(path, True)


def _convert_tokens_and_image(path: Path) -> tuple[Path, str]:
    return path, _convert_file(path, False)


def _filter_out_known_bad_ones(path: Path) -> bool:
    # These images contain to meaningful data or have
    # artifacts like large black areas
    bad_files = {
        "scarlatti-d/keyboard-sonatas/L342K220/min3_up_m-5-9.krn",
        "mozart/piano-sonatas/sonata10-3/maj3_down_m-157-160.krn",
        "mozart/piano-sonatas/sonata10-3/original_m-157-160.krn",
        "mozart/piano-sonatas/sonata10-3/min3_down_m-157-160.krn",
        "mozart/piano-sonatas/sonata10-3/maj2_down_m-157-161.krn",
        "beethoven/piano-sonatas/sonata11-2/maj3_down_m-1-6.krn",
        "beethoven/piano-sonatas/sonata11-2/min3_down_m-1-6.krn",
        "beethoven/piano-sonatas/sonata11-2/original_m-1-6.krn",
        "chopin/preludes/prelude28-17/maj2_down_m-88-91.krn",
    }

    # normalize to forward slashes relative path string
    normalized = path.as_posix()
    return normalized not in bad_files


def convert_grandstaff(only_recreate_token_files: bool = False) -> None:
    if not os.path.exists(grandstaff_root):
        eprint(
            "Downloading grandstaff from https://sites.google.com/view/multiscore-project/datasets"
        )
        grandstaff_archive = os.path.join(dataset_root, "grandstaff.tgz")
        download_file("https://grfia.dlsi.ua.es/musicdocs/grandstaff.tgz", grandstaff_archive)
        untar_file(grandstaff_archive, grandstaff_root)
        eprint("Adding musicxml files to grandstaff dataset")

    index_file = grandstaff_train_index
    if only_recreate_token_files:
        index_file = os.path.join(grandstaff_root, "index_tmp.txt")

    eprint("Indexing Grandstaff dataset, this can up to several hours.")
    krn_files = list(Path(grandstaff_root).rglob("*.krn"))
    krn_files = [file for file in krn_files if _filter_out_known_bad_ones(file)]
    skipped: set[Path] = set()
    with open(index_file, "w") as f:
        file_number = 0
        with multiprocessing.Pool() as p:
            for file, result in p.imap_unordered(
                (
                    _convert_file_only_tokens
                    if only_recreate_token_files
                    else _convert_tokens_and_image
                ),
                krn_files,
            ):
                if result != "":
                    f.write(result + "\n")
                    file_number += 1
                    if file_number % 1000 == 0:
                        eprint(
                            f"Processed {file_number}/{len(krn_files)} files,",
                            f"skipped: {len(skipped)}",
                        )
                else:
                    skipped.add(file)
    eprint("Done indexing")


if __name__ == "__main__":
    import sys

    multiprocessing.set_start_method("spawn")
    only_recreate_token_files = False
    if "--only-tokens" in sys.argv:
        only_recreate_token_files = True
    convert_grandstaff(only_recreate_token_files)
