import multiprocessing
import os
import random
from pathlib import Path

import cv2

from homr.download_utils import download_file, untar_file
from homr.simple_logging import eprint
from homr.staff_parsing import add_image_into_tr_omr_canvas
from training.convert_grandstaff import distort_image
from training.primus_semantic_parser import convert_primus_semantic_to_tokens
from training.transformer.training_vocabulary import token_lines_to_str

script_location = os.path.dirname(os.path.realpath(__file__))
git_root = Path(script_location).parent.absolute()
dataset_root = os.path.join(git_root, "datasets")
primus = os.path.join(dataset_root, "Corpus")
primus_train_index = os.path.join(primus, "index.txt")

if not os.path.exists(primus):
    eprint("Downloading Camera-PrIMuS from https://grfia.dlsi.ua.es/primus/")
    primus_archive = os.path.join(dataset_root, "CameraPrIMuS.tgz")
    download_file("https://grfia.dlsi.ua.es/primus/packages/CameraPrIMuS.tgz", primus_archive)
    untar_file(primus_archive, dataset_root)  # the archive contains already a Corpus folder


def _replace_suffix(path: Path, suffix: str) -> Path:
    suffixes = [".jpg", ".jpeg", ".png"]
    if suffix == ".semantic":
        suffixes.insert(0, "_distorted.jpg")
    for s in suffixes:
        if s in str(path):
            return Path(str(path).replace(s, suffix))
    raise ValueError("Unknown extension " + str(path))


def _find_semantic_file(path: Path) -> Path | None:
    semantic_file = _replace_suffix(path, ".semantic")
    if semantic_file.exists():
        return semantic_file
    return None


def _convert_file(path: Path, only_recreate_token_files: bool) -> list[str]:
    if "-pre.jpg" in str(path):
        return []
    if "," in str(path):
        return []
    preprocessed_path = _replace_suffix(path, "-pre.jpg")
    if not only_recreate_token_files:
        image = cv2.imread(str(path))
        if image is None:
            eprint("Warning: Could not read image", path)
            return []
        margin_top = random.randint(5, 25)
        margin_bottom = random.randint(5, 25)
        preprocessed = add_image_into_tr_omr_canvas(image, margin_top, margin_bottom)
        cv2.imwrite(str(preprocessed_path.absolute()), preprocessed)
        distort_image(str(preprocessed_path.absolute()))
    semantic_file = _find_semantic_file(path)
    if semantic_file is None:
        eprint("Warning: No semantic file found for", path)
        return []
    tokens = convert_primus_semantic_to_tokens(semantic_file.read_text())
    token_file = _replace_suffix(path, ".tokens")
    token_file.write_text(token_lines_to_str(tokens))
    return [
        str(preprocessed_path.relative_to(git_root))
        + ","
        + str(token_file.relative_to(git_root))
        + "\n"
    ]


def _convert_file_only_tokens(path: Path) -> list[str]:
    return _convert_file(path, True)


def _convert_semantic_and_image(path: Path) -> list[str]:
    return _convert_file(path, False)


def convert_primus_dataset(only_recreate_token_files: bool = False) -> None:
    eprint("Indexing PrIMuS dataset")
    with open(primus_train_index, "w") as f:
        file_number = 0
        with multiprocessing.Pool(8) as p:
            for result in p.imap_unordered(
                (
                    _convert_file_only_tokens
                    if only_recreate_token_files
                    else _convert_semantic_and_image
                ),
                Path(primus).rglob("*.png"),
            ):
                f.writelines(result)
                file_number += 1
                if file_number % 1000 == 0:
                    eprint(f"Processed {file_number} files")
    eprint("Done indexing")


if __name__ == "__main__":
    import sys

    multiprocessing.set_start_method("spawn")
    only_recreate_token_files = False
    if "--only-tokens" in sys.argv:
        only_recreate_token_files = True
    convert_primus_dataset(only_recreate_token_files)
