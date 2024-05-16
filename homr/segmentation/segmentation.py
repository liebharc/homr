import hashlib
import lzma
import os
from pathlib import Path

import cv2
import numpy as np

from homr.segmentation import config
from homr.segmentation.inference import inference
from homr.simple_logging import eprint
from homr.type_definitions import NDArray


def generate_pred(image: NDArray) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
    if config.unet_path == config.segnet_path:
        raise ValueError("unet_path and segnet_path should be different")
    eprint("Extracting staffline and symbols")
    staff_symbols_map, _ = inference(
        config.unet_path,
        image,
    )
    staff_layer = 1
    staff = np.where(staff_symbols_map == staff_layer, 1, 0)
    symbol_layer = 2
    symbols = np.where(staff_symbols_map == symbol_layer, 1, 0)

    eprint("Extracting layers of different symbols")
    sep, _ = inference(
        config.segnet_path,
        image,
        manual_th=None,
    )
    stems_layer = 1
    stems_rests = np.where(sep == stems_layer, 1, 0)
    notehead_layer = 2
    notehead = np.where(sep == notehead_layer, 1, 0)
    clefs_keys_layer = 3
    clefs_keys = np.where(sep == clefs_keys_layer, 1, 0)

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


def extract(original_image: NDArray, img_path_str: str, use_cache: bool = False) -> ExtractResult:
    img_path = Path(img_path_str)
    f_name = os.path.splitext(img_path.name)[0]
    npy_path = img_path.parent / f"{f_name}.npy"
    loaded_from_cache = False
    if npy_path.exists() and use_cache:
        eprint("Found a cache")
        file_hash = hashlib.sha256(original_image).hexdigest()  # type: ignore
        with lzma.open(npy_path, "rb") as f:
            staff = np.load(f)
            notehead = np.load(f)
            symbols = np.load(f)
            stems_rests = np.load(f)
            clefs_keys = np.load(f)
            cached_file_hash = f.readline().decode().strip()
            model_name = f.readline().decode().strip()
            if cached_file_hash == "" or model_name == "":
                eprint("Cache is missing meta information, skipping cache")
            elif file_hash != cached_file_hash:
                eprint("File hash mismatch, skipping cache")
            elif model_name != config.segmentation_version:
                eprint("Models have been updated, skipping cache")
            else:
                loaded_from_cache = True
                eprint("Loading from cache")
    if not loaded_from_cache:
        ori_inf_type = os.environ.get("INFERENCE_WITH_TF", None)
        os.environ["INFERENCE_WITH_TF"] = "true"
        staff, symbols, stems_rests, notehead, clefs_keys = generate_pred(original_image)
        if ori_inf_type is not None:
            os.environ["INFERENCE_WITH_TF"] = ori_inf_type
        else:
            del os.environ["INFERENCE_WITH_TF"]

        if use_cache:
            eprint("Saving cache")
            file_hash = hashlib.sha256(original_image).hexdigest()  # type: ignore
            with lzma.open(npy_path, "wb") as f:
                np.save(f, staff)
                np.save(f, notehead)
                np.save(f, symbols)
                np.save(f, stems_rests)
                np.save(f, clefs_keys)
                f.write((file_hash + "\n").encode())
                f.write((config.segmentation_version + "\n").encode())
    original_image = cv2.resize(original_image, (staff.shape[1], staff.shape[0]))

    return ExtractResult(
        img_path, original_image, staff, symbols, stems_rests, notehead, clefs_keys
    )


def segmentation(image: NDArray, img_path: str, use_cache: bool = False) -> ExtractResult:
    return extract(image, img_path, use_cache=use_cache)
