from pathlib import Path
from time import perf_counter

import numpy as np
import onnxruntime as ort

from homr.simple_logging import eprint
from homr.type_definitions import NDArray
from homr.segmentation.config import segnet_path_onnx, segmentation_version

import os 
import lzma
import hashlib
import cv2

class Segnet:
    def __init__(self, model_path, use_gpu):
        if use_gpu:
            try:
                self.model = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
            except Exception as e:
                eprint("Error while trying to load model using CUDA. You probably don't have a compatible gpu") # noqa: E501
                eprint(e)
                self.model = ort.InferenceSession(model_path)
        else:
            self.model = ort.InferenceSession(model_path)
        self.input_name = self.model.get_inputs()[0].name # size: [batch_size, 3, 320, 320]
        self.output_name = self.model.get_outputs()[0].name


    def run(self, input_data):
        out = self.model.run([self.output_name], {self.input_name: input_data})[0]
        return out

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


def merge_patches(patches, image_shape: list[int], win_size: int, step_size: int = -1):
    reconstructed = np.zeros(image_shape, dtype=np.float32)
    weight = np.zeros(image_shape, dtype=np.float32)

    idx = 0
    for iy in range(0, image_shape[0], step_size):
        if iy + win_size > image_shape[0]:
            y = image_shape[0] - win_size
        else:
            y = iy
        for ix in range(0, image_shape[1], step_size):
            if ix + win_size > image_shape[1]:
                x = image_shape[1] - win_size
            else:
                x = ix

            reconstructed[y : y + win_size, x : x + win_size] += patches[idx]
            weight[y : y + win_size, x : x + win_size] += 1
            idx += 1

    # Avoid division by zero
    weight[weight == 0] = 1
    reconstructed /= weight

    return reconstructed.astype(patches[0].dtype)


def inference(image_org: np.ndarray,
              batch_size: int,
              step_size: int,
              use_gpu: bool,
              win_size: int
              ):
    """
    Inference function for the segementation model.
    Args:
        image_org(np.ndarray): Array of the input image
        batch_size(int): Mainly for speeding up GPU performance. Minimal impact on CPU speed. 
        step_size(int): How far the window moves between to input images. 
        use_gpu(bool): Use gpu for inference. Only for debugging purposes (uses try-except to check if gpu is available). 
        win_size(int): Debug only.

    Returns:
        ExtractResult class.
    """
    eprint('Starting Inference.')
    t0 = perf_counter()
    if step_size < 0:
        step_size = win_size // 2

    model = Segnet(segnet_path_onnx, use_gpu)
    data = []
    batch = []
    image = np.transpose(image_org, (2, 0, 1)).astype(np.float32)
    for y in range(0, image.shape[1], step_size):
        if y + win_size > image.shape[1]:
            y = image.shape[1] - win_size
        for x in range(0, image.shape[2], step_size):
            if x + win_size > image.shape[2]:
                x = image.shape[2] - win_size
            hop = image[:, y : y + win_size, x : x + win_size]
            batch.append(hop)

            # When there
            if batch_size == len(batch):
                #run model
                batch_out = model.run(np.stack(batch, axis=0))
                for out in batch_out:
                    out_filtered = np.argmax(out, axis=0)
                    data.append(out_filtered)
                # reset the batch list so it is not full anymore
                batch = []

    # There might still be something in the batch list
    # So we run inference one more time
    if batch:
        batch_out = model.run(np.stack(batch, axis=0))
        for out in batch_out:
            out = np.argmax(out, axis=0)
            data.append(out)

    eprint(f"Segnet Inference time: {perf_counter()- t0}; batch_size of {batch_size}")

    data = merge_patches(data, (image_org.shape[0], image_org.shape[1]), win_size, step_size)
    stems_layer = 1
    stems_rests = np.where(data == stems_layer, 1, 0)
    notehead_layer = 2
    notehead = np.where(data == notehead_layer, 1, 0)
    clefs_keys_layer = 3
    clefs_keys = np.where(data == clefs_keys_layer, 1, 0)
    staff_layer = 4
    staff = np.where(data == staff_layer, 1, 0)
    symbol_layer = 5
    symbols = np.where(data == symbol_layer, 1, 0)

    return staff, symbols, stems_rests, notehead, clefs_keys

def extract(original_image: NDArray, 
            img_path_str: str, 
            use_cache: bool = False,
            batch_size: int = 8,
            step_size: int = -1,
            use_gpu: bool = True,
            win_size: int = 320
            ):
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
            elif model_name != segmentation_version:
                eprint("Models have been updated, skipping cache")
            else:
                loaded_from_cache = True
                eprint("Loading from cache")

    if not loaded_from_cache:
        staff, symbols, stems_rests, notehead, clefs_keys = inference(original_image, 
                                                                      img_path_str,
                                                                      batch_size=batch_size,
                                                                      step_size=step_size,
                                                                      use_gpu=use_gpu,
                                                                      win_size=win_size
                                                                      )
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
                f.write((segmentation_version + "\n").encode())

    original_image = cv2.resize(original_image, (staff.shape[1], staff.shape[0]))

    return ExtractResult(
        img_path, original_image, staff, symbols, stems_rests, notehead, clefs_keys
    )