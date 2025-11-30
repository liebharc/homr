import multiprocessing
import os
from pathlib import Path

import cv2
import numpy as np
import PIL
import PIL.Image
from torchvision import transforms as tr
from torchvision.transforms import Compose

from homr.download_utils import download_file, untar_file
from homr.simple_logging import eprint
from homr.staff_dewarping import warp_image_randomly
from homr.staff_parsing import add_image_into_tr_omr_canvas
from homr.type_definitions import NDArray
from training.datasets.humdrum_kern_parser import convert_kern_to_tokens
from training.datasets.musescore_svg import SvgValidationError
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


def _distort_staff_image(path: str, basename: str) -> str:
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
    distorted_image = _prepare_image(image)
    distorted_image = distort_image(distorted_image)
    cv2.imwrite(basename + "-pre.jpg", distorted_image)
    return basename + "-pre.jpg"


def _prepare_image(image: NDArray) -> NDArray:
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


def add_margin(image: NDArray, top: int, bottom: int, left: int, right: int) -> NDArray:
    # image is 2D grayscale
    flat = image.reshape(-1)
    bg = np.bincount(flat).argmax()

    h, w = image.shape

    top_pad = np.full((top, w), bg, dtype=image.dtype)
    bottom_pad = np.full((bottom, w), bg, dtype=image.dtype)

    img_tb = np.vstack([top_pad, image, bottom_pad])

    new_h = img_tb.shape[0]
    left_pad = np.full((new_h, left), bg, dtype=image.dtype)
    right_pad = np.full((new_h, right), bg, dtype=image.dtype)

    return np.hstack([left_pad, img_tb, right_pad])


def distort_image(image: NDArray) -> NDArray:
    image, _background_value = _add_random_gray_tone(image)
    pil_image = PIL.Image.fromarray(image)
    pipeline = Compose(
        [
            tr.RandomRotation(degrees=2),
            tr.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.5),
            tr.RandomAdjustSharpness(0),
            tr.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 0.5)),
        ]
    )

    augmented_image = pipeline(img=pil_image)
    augmented_image = warp_image_randomly(augmented_image)

    # add paper texture overlay
    paper_texture = np.random.normal(loc=235, scale=10, size=image.shape).astype(np.uint8)  # type: ignore
    augmented_array = np.array(augmented_image)
    alpha = 0.2
    augmented_array = cv2.addWeighted(augmented_array, 1 - alpha, paper_texture, alpha, 0)

    rows, cols = augmented_array.shape[:2]

    # Randomized Gaussian parameters
    sigma_x = np.random.uniform(cols / 4, cols)
    sigma_y = np.random.uniform(rows / 4, rows)
    center_x = np.random.uniform(0, cols)
    center_y = np.random.uniform(0, rows)

    # Generate coordinate grids
    x = np.arange(cols) - center_x
    y = np.arange(rows) - center_y
    x_grid, y_grid = np.meshgrid(x, y)

    # Optional rotation
    theta = np.random.uniform(0, 2 * np.pi)
    x_rot = x_grid * np.cos(theta) + y_grid * np.sin(theta)
    y_rot = -x_grid * np.sin(theta) + y_grid * np.cos(theta)

    # Create asymmetric Gaussian vignette
    mask = np.exp(-0.5 * ((x_rot**2 / sigma_x**2) + (y_rot**2 / sigma_y**2)))
    mask = mask / mask.max()

    # Scale mask to subtle effect (0.9–1.0)
    mask = 0.9 + 0.1 * mask

    # Optional channel variation
    if augmented_array.shape[2] == 3:
        mask = np.stack([mask * np.random.uniform(0.9, 1.0) for _ in range(3)], axis=-1)

    augmented_array = (augmented_array * mask).astype(np.uint8)

    return augmented_array


def _add_random_gray_tone(image_arr: NDArray) -> tuple[NDArray, int]:
    """
    Adds a gray background. While doing so it ensures
    that all symbols are still visible on the image.
    """
    gray = cv2.cvtColor(image_arr, cv2.COLOR_BGR2GRAY)

    lightest_pixel_value = _find_lighest_non_white_pixel(gray)

    minimum_contrast = 70
    pure_white = 255

    if lightest_pixel_value >= pure_white - minimum_contrast:
        return image_arr, pure_white

    strongest_possible_gray = lightest_pixel_value + minimum_contrast

    random_gray_value = np.random.randint(strongest_possible_gray, pure_white)
    if random_gray_value >= pure_white:
        return image_arr, random_gray_value

    mask = np.all(image_arr > random_gray_value, axis=-1)

    jitter = np.random.randint(-5, 5, size=mask.shape)
    gray = np.clip(random_gray_value + jitter, 0, pure_white)

    image_arr[mask] = np.stack([gray[mask]] * 3, axis=-1)

    return image_arr, random_gray_value


def _find_lighest_non_white_pixel(gray: NDArray) -> int:
    pure_white = 255
    valid_pixels = gray[gray < pure_white]

    if valid_pixels.size > 0:
        return valid_pixels.max()
    else:
        return pure_white


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
            image = _distort_staff_image(image_file, basename)
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
