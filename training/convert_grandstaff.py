import multiprocessing
import os
import platform
import stat
import sys
from pathlib import Path

import cv2
import numpy as np
import PIL
import PIL.Image
from scipy.signal import find_peaks  # type: ignore
from torchvision import transforms as tr  # type: ignore
from torchvision.transforms import Compose  # type: ignore

from homr.download_utils import download_file, untar_file
from homr.simple_logging import eprint
from homr.staff_dewarping import warp_image_randomly
from homr.staff_parsing import add_image_into_tr_omr_canvas
from homr.type_definitions import NDArray
from training.musescore_svg import SvgValidationError
from training.music_xml import MusicXmlValidationError, music_xml_to_semantic

script_location = os.path.dirname(os.path.realpath(__file__))
git_root = Path(script_location).parent.absolute()
dataset_root = os.path.join(git_root, "datasets")
grandstaff_root = os.path.join(dataset_root, "grandstaff")
grandstaff_train_index = os.path.join(grandstaff_root, "index.txt")

hum2xml = os.path.join(dataset_root, "hum2xml")
if platform.system() == "Windows":
    eprint("Transformer training is only implemented for Linux")
    eprint("Feel free to submit a PR to support Windows")
    eprint("The main work should be to download hum2xml.exe and change the calls")
    eprint("to use the exe-file instead of the linux binary.")
    sys.exit(1)
if not os.path.exists(hum2xml):
    eprint("Downloading hum2xml from https://extras.humdrum.org/man/hum2xml/")
    download_file("http://extras.humdrum.org/bin/linux/hum2xml", hum2xml)
    os.chmod(hum2xml, stat.S_IXUSR)


if not os.path.exists(grandstaff_root):
    eprint("Downloading grandstaff from https://sites.google.com/view/multiscore-project/datasets")
    grandstaff_archive = os.path.join(dataset_root, "grandstaff.tgz")
    download_file("https://grfia.dlsi.ua.es/musicdocs/grandstaff.tgz", grandstaff_archive)
    untar_file(grandstaff_archive, grandstaff_root)


def _get_dark_pixels_per_row(image: NDArray) -> NDArray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dark_pixels_per_row = np.zeros(gray.shape[0])
    dark_threshold = 200
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            if gray[i, j] < dark_threshold:
                dark_pixels_per_row[i] += 1
    return dark_pixels_per_row


def _find_central_valleys(image: NDArray, dark_pixels_per_row: NDArray) -> np.int32 | None:
    conv_len = image.shape[0] // 4 + 1
    blurred = np.convolve(dark_pixels_per_row, np.ones(conv_len) / conv_len, mode="same")

    # Find the central valley
    peaks, _ = find_peaks(-blurred, distance=10, prominence=1)
    if len(peaks) == 1:
        peaks = [peaks[len(peaks) // 2]]
        middle = peaks[0]
        return np.int32(middle)
    return None


def _split_staff_image(path: str, basename: str) -> tuple[str | None, str | None]:
    """
    This algorithm is taken from `oemer` staffline extraction algorithm. In this simplified version
    it only works with images which have no distortions.
    """
    image = cv2.imread(path)
    dark_pixels_per_row = _get_dark_pixels_per_row(image)
    upper_bound, lower_bound = _get_image_bounds(dark_pixels_per_row)
    image = image[upper_bound:-lower_bound]
    dark_pixels_per_row = dark_pixels_per_row[upper_bound:-lower_bound]
    norm = (dark_pixels_per_row - np.mean(dark_pixels_per_row)) / np.std(dark_pixels_per_row)
    centers, _ = find_peaks(norm, height=1.4, distance=3, prominence=1)
    lines_per_staff = 5
    if len(centers) == lines_per_staff:
        upper = _prepare_image(image)
        predistorted_path = basename + "_distorted.jpg"
        if os.path.exists(predistorted_path):
            predistorted_image = cv2.imread(predistorted_path)
            single_image = _prepare_image(predistorted_image)
            cv2.imwrite(basename + "_single-pre.jpg", single_image)
            return distort_image(basename + "_single-pre.jpg"), None
        eprint(f"INFO: Couldn't find pre-distorted image {path}, using custom distortions")
        cv2.imwrite(basename + "_upper-pre.jpg", upper)
        return distort_image(basename + "_upper-pre.jpg"), None
    elif len(centers) == 2 * lines_per_staff:
        middle = np.int32(np.round((centers[4] + centers[5]) / 2))
    else:
        central_valley = _find_central_valleys(image, dark_pixels_per_row)
        if central_valley is None:
            return None, None
        middle = central_valley

    overlap = 10
    if middle < overlap or middle > image.shape[0] - overlap:
        eprint(f"INFO: Failed to split {path}, middle is at {middle}")
        return None, None

    upper = _prepare_image(image[: middle + overlap])
    lower = _prepare_image(image[middle - overlap :])
    cv2.imwrite(basename + "_upper-pre.jpg", upper)
    cv2.imwrite(basename + "_lower-pre.jpg", lower)
    return distort_image(basename + "_upper-pre.jpg"), distort_image(basename + "_lower-pre.jpg")


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


def _check_staff_image(path: str, basename: str) -> tuple[str | None, str | None]:
    """
    This method helps with reprocessing a folder more quickly by skipping
    the image splitting.
    """
    if not os.path.exists(basename + "_upper-pre.jpg"):
        return None, None
    return basename + "_upper-pre.jpg", basename + "_lower-pre.jpg"


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


def contains_max_one_clef(semantic: str) -> bool:
    """
    hum2xml sometimes generates invalid musicxml which
    we can detect by checking for multiple clefs, e.g.

    scarlatti-d/keyboard-sonatas/L481K025/min3_up_m-79-82.krn

    The issue here is likely that it uses two G2 clefs, and
    overlays them on top of each other to indicate
    multiple notes at the same time.
    """
    return semantic.count("clef-") <= 1


def _music_xml_to_semantic(path: str, basename: str) -> tuple[str | None, str | None]:
    result = music_xml_to_semantic(path)
    staffs_in_grandstaff = 2
    if len(result) != staffs_in_grandstaff:
        return None, None
    lines = [" ".join(staff) for staff in result]
    if not all(contains_max_one_clef(line) for line in lines):
        return None, None

    with open(basename + "_upper.semantic", "w") as f:
        f.write(lines[0])
    with open(basename + "_lower.semantic", "w") as f:
        f.write(lines[1])
    return basename + "_upper.semantic", basename + "_lower.semantic"


def _convert_file(  # noqa: PLR0911
    path: Path, ony_recreate_semantic_files: bool = False
) -> list[str]:
    try:
        basename = str(path).replace(".krn", "")
        image_file = str(path).replace(".krn", ".jpg")
        musicxml = str(path).replace(".krn", ".musicxml")
        result = os.system(f"{hum2xml} {path} > {musicxml}")  # noqa: S605
        if result != 0:
            eprint(f"Failed to convert {path}")
            return []
        upper_semantic, lower_semantic = _music_xml_to_semantic(musicxml, basename)
        if upper_semantic is None or lower_semantic is None:
            eprint(f"Failed to convert {musicxml}")
            return []
        if ony_recreate_semantic_files:
            upper, lower = _check_staff_image(image_file, basename)
        else:
            upper, lower = _split_staff_image(image_file, basename)
        if upper is None:
            return []
        if lower is None:
            return [
                str(Path(upper).relative_to(git_root))
                + ","
                + str(Path(upper_semantic).relative_to(git_root)),
            ]
        return [
            str(Path(upper).relative_to(git_root))
            + ","
            + str(Path(upper_semantic).relative_to(git_root)),
            str(Path(lower).relative_to(git_root))
            + ","
            + str(Path(lower_semantic).relative_to(git_root)),
        ]
    except (SvgValidationError, MusicXmlValidationError):
        return []
    except Exception as e:
        eprint("Failed to convert ", path, e)
        return []


def _convert_file_only_semantic(path: Path) -> list[str]:
    return _convert_file(path, True)


def _convert_semantic_and_image(path: Path) -> list[str]:
    return _convert_file(path, False)


def convert_grandstaff(only_recreate_semantic_files: bool = False) -> None:
    index_file = grandstaff_train_index
    if only_recreate_semantic_files:
        index_file = os.path.join(grandstaff_root, "index_tmp.txt")

    eprint("Indexing Grandstaff dataset, this can up to several hours.")
    krn_files = list(Path(grandstaff_root).rglob("*.krn"))
    with open(index_file, "w") as f:
        file_number = 0
        skipped_files = 0
        with multiprocessing.Pool() as p:
            for result in p.imap_unordered(
                (
                    _convert_file_only_semantic
                    if only_recreate_semantic_files
                    else _convert_semantic_and_image
                ),
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
    only_recreate_semantic_files = False
    if "--only-semantic" in sys.argv:
        only_recreate_semantic_files = True
    convert_grandstaff(only_recreate_semantic_files)
