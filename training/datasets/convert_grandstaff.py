import multiprocessing
import os
from pathlib import Path

import cv2
import numpy as np
import albumentations as A

from homr.download_utils import download_file, untar_file
from homr.simple_logging import eprint
from homr.staff_dewarping import warp_image_array_fast, warp_image_randomly
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



def _dilate(img: np.ndarray, **kwargs) -> np.ndarray:
    kernel = np.ones((2, 2), np.uint8)
    return cv2.dilate(img, kernel, iterations=1)


def _erode(img: np.ndarray, **kwargs) -> np.ndarray:
    kernel = np.ones((2, 2), np.uint8)
    return cv2.erode(img, kernel, iterations=1)

def _add_random_black_edges(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    img = img.copy()

    max_thickness = int(0.06 * min(h, w))
    thickness = np.random.randint(1, max_thickness + 1)

    sides = ["top", "bottom", "left", "right"]
    np.random.shuffle(sides)
    sides = sides[: np.random.randint(1, 3)]  # 1 or 2 sides

    for side in sides:
        if side == "top":
            img[:thickness, :] = 0
        elif side == "bottom":
            img[-thickness:, :] = 0
        elif side == "left":
            img[:, :thickness] = 0
        elif side == "right":
            img[:, -thickness:] = 0

    return img

def distort_image(image: NDArray) -> NDArray:
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    h, w = image.shape[:2]
    pad = int(0.08 * min(h, w))  # safe margin

    class ShowThrough(A.ImageOnlyTransform):
        def apply(self, img, **params):
            flipped = cv2.flip(img, 1)
            back = cv2.GaussianBlur(flipped, (11, 11), 0)
            return cv2.addWeighted(img, 0.92, back, 0.08, 0)

    pipeline = A.Compose(
        [
            
            A.Perspective(scale=(0.00, 0.01), keep_size=True, p=0.5),
            A.SafeRotate(limit=2, border_mode=cv2.BORDER_CONSTANT, p=0.7, fill=255),
            BookWarp(p=0.35),  # your custom piecewise/book warp

            # 4. Crop back to original size
            A.CenterCrop(height=h, width=w),

            # Crop back to original size
            A.CenterCrop(height=h, width=w),

            # Lighting
            A.RandomShadow(
                shadow_dimension=4,
                p=0.3,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.4,
            ),

            # Paper / scan
            ShowThrough(p=0.15),

            A.GaussNoise(
                std_range=(0.01, 0.05),
                p=0.3,
            ),

            # Ink quality (topology-safe)
            A.OneOf(
                [
                    A.Lambda(image=_dilate),
                    A.Lambda(image=_erode),
                ],
                p=0.25,
            ),

            # Digital compression
            A.ImageCompression(p=0.25),
        ],
        p=1.0,
    )

    augmented = pipeline(image=image)["image"]

    # Apply gray gradient background last
    augmented = _add_random_gray_tone(augmented)

    # Random black scanner / binder edges
    if np.random.rand() < 0.3:
        augmented = _add_random_black_edges(augmented)

    return augmented

def book_page_warp(
    img: np.ndarray,
    curvature: float = 0.25,
    vertical_bow: float = 0.04,
    spine_pos: float | None = None,
    focal_mult: float = 1.4,
) -> np.ndarray:
    """
    Physically-inspired book page warp using a 3D surface + camera projection.

    curvature     : strength of page bend (0.15–0.35 realistic)
    vertical_bow  : secondary vertical tension (0–0.06)
    spine_pos     : x-position of spine in [0,1]; None = random
    focal_mult    : camera focal length multiplier
    """

    h, w = img.shape[:2]

    if spine_pos is None:
        spine_pos = np.random.uniform(0.25, 0.45)

    # Coordinate grids (page space)
    yy, xx = np.meshgrid(
        np.linspace(0, 1, h),
        np.linspace(0, 1, w),
        indexing="ij",
    )

    # Signed distance from spine
    dx = xx - spine_pos

    # Depth model (book curvature)
    z = curvature * (1.0 - (dx / np.max(np.abs(dx))) ** 2)
    z = np.clip(z, 0, None)

    # Vertical paper tension
    z += vertical_bow * np.cos(np.pi * yy)

    # Camera model
    f = focal_mult * w
    cx, cy = w / 2, h / 2

    # 3D surface coordinates
    X = (xx - 0.5) * w
    Y = (yy - 0.5) * h
    Z = z * w  # depth in pixel scale

    # Project to image plane
    denom = f + Z
    map_x = (f * X / denom + cx).astype(np.float32)
    map_y = (f * Y / denom + cy).astype(np.float32)

    # Valid remap
    warped = cv2.remap(
        img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    return warped


class BookWarp(A.ImageOnlyTransform):
    def __init__(
        self,
        curvature=(0.18, 0.35),
        vertical_bow=(0.02, 0.06),
        p=0.3,
    ):
        super().__init__(p=p)
        self.curvature = curvature
        self.vertical_bow = vertical_bow

    def apply(self, img, **params):
        return warp_image_array_fast(
            img,
        )

def _add_random_gray_tone(image_arr: NDArray) -> NDArray:
    """
    Adds a gray background. While doing so it ensures
    that all symbols are still visible on the image.
    """
    gray = cv2.cvtColor(image_arr, cv2.COLOR_BGR2GRAY)

    lightest_pixel_value = _find_lighest_non_white_pixel(gray)

    minimum_contrast = 70
    pure_white = 255

    if lightest_pixel_value >= pure_white - minimum_contrast:
        return image_arr

    strongest_possible_gray = lightest_pixel_value + minimum_contrast

    random_gray_value = np.random.randint(strongest_possible_gray, pure_white)
    if random_gray_value >= pure_white:
        return image_arr

    mask = np.all(image_arr > random_gray_value, axis=-1)

    jitter = np.random.randint(-5, 5, size=mask.shape)
    gray = np.clip(random_gray_value + jitter, 0, pure_white)

    image_arr[mask] = np.stack([gray[mask]] * 3, axis=-1)

    return image_arr


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
