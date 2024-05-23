import json
import os
from typing import Any

import numpy as np
import tensorflow as tf
from PIL import Image

from homr.simple_logging import eprint
from homr.type_definitions import NDArray


class InferenceModel:
    def __init__(self, model_path: str) -> None:
        model, metadata = _load_model(model_path)
        self.model = model
        self.input_shape = metadata["input_shape"]
        self.output_shape = metadata["output_shape"]

    def inference(  # noqa: C901, PLR0912
        self,
        image: NDArray,
        step_size: int = 128,
        batch_size: int = 16,
        manual_th: Any | None = None,
    ) -> tuple[NDArray, NDArray]:

        # Collect data
        # Tricky workaround to avoid random mistery transpose when loading with 'Image'.
        image_rgb = Image.fromarray(image).convert("RGB")
        image = np.array(image_rgb)
        win_size = self.input_shape[1]
        data = []
        for y in range(0, image.shape[0], step_size):
            if y + win_size > image.shape[0]:
                y = image.shape[0] - win_size  # noqa: PLW2901
            for x in range(0, image.shape[1], step_size):
                if x + win_size > image.shape[1]:
                    x = image.shape[1] - win_size  # noqa: PLW2901
                hop = image[y : y + win_size, x : x + win_size]
                data.append(hop)

        # Predict
        pred = []
        for idx in range(0, len(data), batch_size):
            eprint(f"{idx+1}/{len(data)} (step: {batch_size})", end="\r")
            batch = np.array(data[idx : idx + batch_size])
            out = self.model.serve(batch)
            pred.append(out)
        eprint(f"{len(data)}/{len(data)} (step: {batch_size})")  # Add newline after progress

        # Merge prediction patches
        output_shape = image.shape[:2] + (self.output_shape[-1],)
        out = np.zeros(output_shape, dtype=np.float32)
        mask = np.zeros(output_shape, dtype=np.float32)
        hop_idx = 0
        for y in range(0, image.shape[0], step_size):
            if y + win_size > image.shape[0]:
                y = image.shape[0] - win_size  # noqa: PLW2901
            for x in range(0, image.shape[1], step_size):
                if x + win_size > image.shape[1]:
                    x = image.shape[1] - win_size  # noqa: PLW2901
                batch_idx = hop_idx // batch_size
                remainder = hop_idx % batch_size
                hop = pred[batch_idx][remainder]
                out[y : y + win_size, x : x + win_size] += hop
                mask[y : y + win_size, x : x + win_size] += 1
                hop_idx += 1

        out /= mask
        if manual_th is None:
            class_map = np.argmax(out, axis=-1)
        else:
            if len(manual_th) != output_shape[-1] - 1:
                raise ValueError(f"{manual_th}, {output_shape[-1]}")
            class_map = np.zeros(out.shape[:2] + (len(manual_th),))
            for idx, th in enumerate(manual_th):
                class_map[..., idx] = np.where(out[..., idx + 1] > th, 1, 0)

        return class_map, out


cached_segmentation: dict[str, Any] = {}


def inference(
    model_path: str,
    image: NDArray,
    step_size: int = 128,
    batch_size: int = 16,
    manual_th: Any | None = None,
) -> tuple[NDArray, NDArray]:
    if model_path not in cached_segmentation:
        model = InferenceModel(model_path)
        cached_segmentation[model_path] = model
    else:
        model = cached_segmentation[model_path]
    return model.inference(image, step_size, batch_size, manual_th)


def _load_model(model_path: str) -> tuple[Any, dict[str, Any]]:
    """Load model and metadata"""

    model = tf.saved_model.load(model_path)
    with open(os.path.join(model_path, "meta.json")) as f:
        metadata = json.loads(f.read())
    return model, metadata
