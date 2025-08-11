from math import floor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from training.architecture.segmentation import config
from training.architecture.segmentation.inference import inference
from homr.type_definitions import NDArray
from homr.color_adjust import color_adjust


def generate_pred(image: NDArray, ouput_path, num_classes) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
    sep = inference(config.segnet_path, image)

    if ouput_path is not None:
        # multiplying sep so we have values ranging from 0 to 255
        output = sep * floor(255 / num_classes)
        out_img = Image.fromarray(output.astype(np.uint8))
        out_img.save(ouput_path)


    stems_layer = 1
    stems_rests = np.where(sep == stems_layer, 1, 0)
    notehead_layer = 2
    notehead = np.where(sep == notehead_layer, 1, 0)
    clefs_keys_layer = 3
    clefs_keys = np.where(sep == clefs_keys_layer, 1, 0)
    staff_layer = 4
    staff = np.where(sep == staff_layer, 1, 0)
    symbol_layer = 5
    symbols = np.where(sep == symbol_layer, 1, 0)

    return staff, symbols, stems_rests, notehead, clefs_keys


class ExtractResult:
    def __init__(
        self,
        filename: Path,
        original: NDArray,
        staff: NDArray,
        symbols: NDArray,
        stems_rests: NDArray,
        notehead: NDArray,
        clefs_keys: NDArray,
    ):
        self.filename = filename
        self.original = original
        self.staff = staff
        self.symbols = symbols
        self.stems_rests = stems_rests
        self.notehead = notehead
        self.clefs_keys = clefs_keys


def extract(original_image: NDArray, img_path_str: str, ouput_path, num_classes) -> ExtractResult:
    img_path = Path(img_path_str)

    staff, symbols, stems_rests, notehead, clefs_keys = generate_pred(original_image, ouput_path, num_classes)

    original_image = cv2.resize(original_image, (staff.shape[1], staff.shape[0]))

    return ExtractResult(
        img_path, original_image, staff, symbols, stems_rests, notehead, clefs_keys
    )


def segmentation(image: NDArray, img_path: str, use_cache: bool = False) -> ExtractResult:
    return extract(image, img_path, use_cache=use_cache)


def test_segnet(image_path, num_classes=None, output_path=None):
    """
    image_path(str): path to the input image
    num_classes(int): number of output classes of the segnet
    output_path(str): path to save the results of the segnet
    """
    img = cv2.imread(image_path)
    preprocessed, _background = color_adjust(img, 40)
    if preprocessed.shape[0] >= 320 or preprocessed.shape[1] >= 320: # 320 is win_size
        raise ValueError(f"Input image is too small ({preprocessed.shape[0]} x {preprocessed.shape[1]}); minimum size is 320 by 320")

    return extract(preprocessed, image_path,  output_path, num_classes)
