import multiprocessing
import os
import random
from pathlib import Path

import cv2
import numpy as np
import PIL
import PIL.Image
from torchvision import transforms as tr  # type: ignore
from torchvision.transforms import Compose  # type: ignore

from homr.download_utils import download_file, untar_file, unzip_file
from homr.simple_logging import eprint
from homr.staff_dewarping import warp_image_randomly
from homr.staff_parsing import add_image_into_tr_omr_canvas
from training.musescore_svg import SvgValidationError
from training.music_xml import MusicXmlValidationError

script_location = os.path.dirname(os.path.realpath(__file__))
git_root = Path(script_location).parent.absolute()
dataset_root = os.path.join(git_root, "datasets")
grandstaff_root = os.path.join(dataset_root, "grandstaff")
grandstaff_train_index = os.path.join(grandstaff_root, "index.txt")


if not os.path.exists(grandstaff_root):
    eprint("Downloading grandstaff from https://sites.google.com/view/multiscore-project/datasets")
    grandstaff_archive = os.path.join(dataset_root, "grandstaff.tgz")
    download_file("https://grfia.dlsi.ua.es/musicdocs/grandstaff.tgz", grandstaff_archive)
    untar_file(grandstaff_archive, grandstaff_root)
    eprint("Adding musicxml files to grandstaff dataset")
    music_xml_download = os.path.join(dataset_root, "grandstaff_musicxml.zip")
    download_file(
        "https://github.com/liebharc/grandstaff_musicxml/archive/refs/heads/main.zip",
        music_xml_download,
    )
    unzip_file(music_xml_download, grandstaff_root, flatten_root_entry=True)


def distort_image(path: str) -> str:
    image = PIL.Image.open(path)
    image = _add_random_gray_tone(image)
    pipeline = Compose(
        [
            tr.RandomRotation(degrees=1),
            tr.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            tr.RandomAdjustSharpness(2),
        ]
    )

    augmented_image = pipeline(img=image)
    augmented_image = warp_image_randomly(augmented_image)
    augmented_image.save(path)
    return path


def _add_random_gray_tone(image: PIL.Image.Image) -> PIL.Image.Image:
    image_arr = np.array(image)
    random_gray_value = 255 - np.random.randint(0, 50)

    mask = np.all(image_arr > random_gray_value, axis=-1)

    jitter = np.random.randint(-5, 5, size=mask.shape)

    gray = np.clip(random_gray_value + jitter, 0, 255)

    image_arr[mask] = gray[mask, None]

    return PIL.Image.fromarray(image_arr)


def _convert_file(path: Path) -> list[str]:
    try:
        basename = str(path).replace(".krn", "")
        kern_file = str(path)
        image_file = str(path).replace(".krn", ".jpg")
        image = cv2.imread(str(image_file))
        if image is None:
            eprint("Warning: Could not read image", path)
            return []

        margin_top = random.randint(0, 10)
        margin_bottom = random.randint(0, 10)
        preprocessed = add_image_into_tr_omr_canvas(image, False, margin_top, margin_bottom)
        preprocessed_path = Path(basename + "-pre.jpg")
        if preprocessed_path is None:
            eprint("Warning: Unknown extension", path)
            return []
        cv2.imwrite(str(preprocessed_path.absolute()), preprocessed)
        distort_image(str(preprocessed_path.absolute()))

        return [
            str(preprocessed_path.relative_to(git_root))
            + ","
            + str(Path(kern_file).relative_to(git_root))
        ]
    except (SvgValidationError, MusicXmlValidationError):
        return []
    except Exception as e:
        eprint("Failed to convert ", path, e)
        return []


def convert_grandstaff() -> None:
    index_file = grandstaff_train_index

    eprint("Indexing Grandstaff dataset, this can up to several hours.")
    krn_files = list(Path(grandstaff_root).rglob("*.krn"))
    with open(index_file, "w") as f:
        file_number = 0
        skipped_files = 0
        with multiprocessing.Pool() as p:
            for result in p.imap_unordered(
                (_convert_file),
                krn_files,
            ):
                if len(result) > 0:
                    for line in result:
                        f.write(line + "\n")
                else:
                    skipped_files += 1
                file_number += 1
                if file_number % 1000 == 0:
                    eprint(
                        f"Processed {file_number}/{len(krn_files)} files,",
                        f"skipped {skipped_files} files",
                    )
    eprint("Done indexing")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    convert_grandstaff()
