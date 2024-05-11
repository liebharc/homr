import multiprocessing
import os
import random
import sys
from collections.abc import Generator
from pathlib import Path

import cv2

from homr.staff_parsing import add_image_into_tr_omr_canvas

script_location = os.path.dirname(os.path.realpath(__file__))
git_root = Path(script_location).parent.absolute()
dataset_root = os.path.join(git_root, "datasets")

diw_root = os.path.join(dataset_root, "diw")
diw_img_root = os.path.join(diw_root, "5k", "img")
diw_train_index = os.path.join(diw_root, "index.txt")

if not os.path.exists(diw_root):
    print(
        "You need to download Documents In the Wild from https://github.com/cvlab-stonybrook/PaperEdge?tab=readme-ov-files"
    )
    print("and unpack it to at " + diw_root)
    print("Unfortunately we can't do that in a script as it's stored on Google drive")
    sys.exit(1)


def _convert_file(file: Path) -> list[str]:
    if "-pre.png" in str(file):
        return []
    image = cv2.imread(str(file))

    image_width = image.shape[1]
    max_width = 1280

    width = min(max_width, image_width)
    height = random.randint(32, 128)

    top = (image.shape[0] - height) // 2
    bottom = top + height
    left = (image.shape[1] - width) // 2
    right = left + width
    cropped = image[top:bottom, left:right, :]

    target_path = str(file).replace(".png", "-pre.png")
    converted, _ignored = add_image_into_tr_omr_canvas(cropped)
    cv2.imwrite(target_path, converted)
    return [f"{target_path},nosymbols\n"]


def _convert_dataset(glob_result: Generator[Path, None, None], index_file: str) -> None:
    with open(index_file, "w") as f:
        file_number = 0
        with multiprocessing.Pool(8) as p:
            for result in p.imap_unordered(_convert_file, glob_result):
                f.writelines(result)
                file_number += 1
                if file_number % 1000 == 0:
                    print(f"Processed {file_number} files")


def convert_diw_dataset() -> None:
    print("Indexing documents in the wild dataset")
    _convert_dataset(Path(diw_img_root).rglob("*[!-pre].png"), diw_train_index)
    print("Done indexing")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    convert_diw_dataset()
