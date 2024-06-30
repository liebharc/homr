import multiprocessing
import os
import random
from collections.abc import Generator
from pathlib import Path

import cv2

from homr.download_utils import download_file, untar_file
from homr.simple_logging import eprint
from homr.staff_parsing import add_image_into_tr_omr_canvas
from training.convert_grandstaff import distort_image

script_location = os.path.dirname(os.path.realpath(__file__))
git_root = Path(script_location).parent.absolute()
dataset_root = os.path.join(git_root, "datasets")
primus = os.path.join(dataset_root, "Corpus")
primus_train_index = os.path.join(primus, "index.txt")
primus_distorted_train_index = os.path.join(primus, "distored_index.txt")

if not os.path.exists(primus):
    eprint("Downloading Camera-PrIMuS from https://grfia.dlsi.ua.es/primus/")
    primus_archive = os.path.join(dataset_root, "CameraPrIMuS.tgz")
    download_file("https://grfia.dlsi.ua.es/primus/packages/CameraPrIMuS.tgz", primus_archive)
    untar_file(primus_archive, dataset_root)  # the archive contains already a Corpus folder


def _replace_suffix(path: Path, suffix: str) -> Path | None:
    suffixes = [".jpg", ".jpeg", ".png"]
    if suffix == ".semantic":
        suffixes.insert(0, "_distorted.jpg")
    for s in suffixes:
        if s in str(path):
            return Path(str(path).replace(s, suffix))
    return None


def _find_semantic_file(path: Path) -> Path | None:
    semantic_file = _replace_suffix(path, ".semantic")
    if semantic_file is not None and semantic_file.exists():
        return semantic_file
    return None


def _convert_file(path: Path, distort: bool = False) -> list[str]:
    if "-pre.jpg" in str(path):
        return []
    if "," in str(path):
        return []
    image = cv2.imread(str(path))
    if image is None:
        eprint("Warning: Could not read image", path)
        return []
    margin_top = random.randint(0, 10)
    margin_bottom = random.randint(0, 10)
    preprocessed = add_image_into_tr_omr_canvas(image, margin_top, margin_bottom)
    preprocessed_path = _replace_suffix(path, "-pre.jpg")
    if preprocessed_path is None:
        eprint("Warning: Unknown extension", path)
        return []
    cv2.imwrite(str(preprocessed_path.absolute()), preprocessed)
    if distort:
        distort_image(str(preprocessed_path.absolute()))
    semantic_file = _find_semantic_file(path)
    if semantic_file is None:
        eprint("Warning: No semantic file found for", path)
        return []
    return [
        str(preprocessed_path.relative_to(git_root))
        + ","
        + str(semantic_file.relative_to(git_root))
        + "\n"
    ]


def _convert_file_without_distortions(path: Path) -> list[str]:
    return _convert_file(path)


def _convert_and_distort_file(path: Path) -> list[str]:
    return _convert_file(path, True)


def _convert_dataset(
    glob_result: Generator[Path, None, None], index_file: str, distort: bool = False
) -> None:
    with open(index_file, "w") as f:
        file_number = 0
        with multiprocessing.Pool(8) as p:
            for result in p.imap_unordered(
                _convert_and_distort_file if distort else _convert_file_without_distortions,
                glob_result,
            ):
                f.writelines(result)
                file_number += 1
                if file_number % 1000 == 0:
                    eprint(f"Processed {file_number} files")


def convert_primus_dataset() -> None:
    eprint("Indexing PrIMuS dataset")
    _convert_dataset(Path(primus).rglob("*.png"), primus_train_index, distort=True)
    eprint("Indexing PrIMuS Distorted dataset")
    _convert_dataset(Path(primus).rglob("*_distorted.jpg"), primus_distorted_train_index)
    eprint("Done indexing")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    convert_primus_dataset()
