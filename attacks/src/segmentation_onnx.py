from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort


CLASS_NAMES = {
    0: "background",
    1: "stems_rests",
    2: "notehead",
    3: "clefs_keys",
    4: "staff",
    5: "symbols",
}


def build_onnx_providers(use_cuda: bool = True):
    cuda_options = {
        "device_id": 0,
        "cudnn_conv_algo_search": "HEURISTIC",
        "arena_extend_strategy": "kNextPowerOfTwo",
    }

    if use_cuda and "CUDAExecutionProvider" in ort.get_available_providers():
        return [
            ("CUDAExecutionProvider", cuda_options),
            "CPUExecutionProvider",
        ]

    return ["CPUExecutionProvider"]


def extract_patch_chw(image_chw: np.ndarray, y: int, x: int, win_size: int) -> np.ndarray:
    c, h, w = image_chw.shape

    patch = np.full(
        (c, win_size, win_size),
        255,
        dtype=image_chw.dtype,
    )

    y0 = max(y, 0)
    x0 = max(x, 0)
    y1 = min(y + win_size, h)
    x1 = min(x + win_size, w)

    py0 = 0
    px0 = 0
    py1 = py0 + (y1 - y0)
    px1 = px0 + (x1 - x0)

    patch[:, py0:py1, px0:px1] = image_chw[:, y0:y1, x0:x1]

    return patch


def merge_patches(
    patches: list[np.ndarray],
    image_shape: tuple[int, int],
    win_size: int,
    step_size: int,
) -> np.ndarray:
    reconstructed = np.zeros(image_shape, dtype=np.float32)
    weight = np.zeros(image_shape, dtype=np.float32)

    idx = 0

    for iy in range(0, image_shape[0], step_size):
        y = min(iy, image_shape[0] - win_size)
        y0 = max(y, 0)
        y1 = min(y + win_size, image_shape[0])

        for ix in range(0, image_shape[1], step_size):
            x = min(ix, image_shape[1] - win_size)
            x0 = max(x, 0)
            x1 = min(x + win_size, image_shape[1])

            patch = patches[idx]

            ph = y1 - y0
            pw = x1 - x0

            reconstructed[y0:y1, x0:x1] += patch[:ph, :pw]
            weight[y0:y1, x0:x1] += 1

            idx += 1

    weight[weight == 0] = 1
    reconstructed /= weight

    return reconstructed.astype(patches[0].dtype)


class SegNetONNX:
    def __init__(
        self,
        model_path: str | Path = "models/onnx/segnet.onnx",
        use_cuda: bool = True,
        batch_size: int = 8,
        win_size: int = 320,
        step_size: int = -1,
    ):
        self.model_path = Path(model_path)
        self.batch_size = int(batch_size)
        self.win_size = int(win_size)
        self.step_size = self.win_size // 2 if step_size < 0 else int(step_size)

        if not self.model_path.exists():
            raise FileNotFoundError(f"SegNet ONNX model not found: {self.model_path}")

        self.providers = build_onnx_providers(use_cuda=use_cuda)

        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=self.providers,
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict_class_map(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        HOMR-style sliding-window SegNet inference.

        Input:
            image_rgb: [H, W, 3], uint8 or float

        Output:
            class_map: [H, W], uint8, values 0..5
        """

        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(f"Expected RGB image [H, W, 3], got {image_rgb.shape}")

        image_chw = np.transpose(image_rgb, (2, 0, 1)).astype(np.float32)

        _, h, w = image_chw.shape

        class_patches = []
        batch = []

        for y_loop in range(0, max(h, self.win_size), self.step_size):
            y = min(y_loop, h - self.win_size)

            for x_loop in range(0, max(w, self.win_size), self.step_size):
                x = min(x_loop, w - self.win_size)

                patch = extract_patch_chw(image_chw, y, x, self.win_size)
                batch.append(patch)

                if len(batch) == self.batch_size:
                    self._flush_batch(batch, class_patches)
                    batch.clear()

        if batch:
            self._flush_batch(batch, class_patches)
            batch.clear()

        class_map = merge_patches(
            class_patches,
            image_shape=(h, w),
            win_size=self.win_size,
            step_size=self.step_size,
        )

        return class_map.astype(np.uint8)

    def _flush_batch(self, batch: list[np.ndarray], class_patches: list[np.ndarray]) -> None:
        batch_input = np.stack(batch, axis=0).astype(np.float32)

        output = self.session.run(
            [self.output_name],
            {self.input_name: batch_input},
        )[0]

        for patch_output in output:
            class_patch = np.argmax(patch_output, axis=0).astype(np.uint8)
            class_patches.append(class_patch)

    @staticmethod
    def masks_from_class_map(class_map: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "background": class_map == 0,
            "stems_rests": class_map == 1,
            "notehead": class_map == 2,
            "clefs_keys": class_map == 3,
            "staff": class_map == 4,
            "symbols": class_map == 5,
            "notation": class_map > 0,
        }


def build_staff_band_mask(
    staff_mask: np.ndarray,
    vertical_kernel_height: int = 45,
    horizontal_kernel_width: int = 35,
) -> np.ndarray:
    mask = staff_mask.astype(np.uint8) * 255

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, vertical_kernel_height),
    )
    band = cv2.dilate(mask, vertical_kernel, iterations=1)

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (horizontal_kernel_width, 1),
    )
    band = cv2.morphologyEx(band, cv2.MORPH_CLOSE, horizontal_kernel, iterations=1)

    return band > 0


def extract_vertical_regions_from_band_mask(
    mask: np.ndarray,
    min_height: int = 25,
    min_gap: int = 20,
) -> list[tuple[int, int]]:
    row_sums = mask.astype(np.uint8).sum(axis=1)
    active = row_sums > 0

    raw_regions = []

    in_region = False
    start = None

    for y, val in enumerate(active):
        if val and not in_region:
            start = y
            in_region = True
        elif not val and in_region:
            raw_regions.append((start, y))
            in_region = False

    if in_region:
        raw_regions.append((start, len(active) - 1))

    merged = []

    for region in raw_regions:
        if not merged:
            merged.append(region)
            continue

        prev_start, prev_end = merged[-1]
        cur_start, cur_end = region

        if cur_start - prev_end <= min_gap:
            merged[-1] = (prev_start, cur_end)
        else:
            merged.append(region)

    return [
        (y0, y1)
        for y0, y1 in merged
        if y1 - y0 >= min_height
    ]


def extract_staff_crops_from_class_map(
    image_rgb: np.ndarray,
    class_map: np.ndarray,
    pad_ratio: float = 0.8,
    min_pad: int = 30,
) -> list[dict]:
    masks = SegNetONNX.masks_from_class_map(class_map)

    staff_band_mask = build_staff_band_mask(masks["staff"])

    regions = extract_vertical_regions_from_band_mask(staff_band_mask)

    crops = []

    for index, (y0, y1) in enumerate(regions):
        pad = max(min_pad, int((y1 - y0) * pad_ratio))

        yy0 = max(0, y0 - pad)
        yy1 = min(image_rgb.shape[0], y1 + pad)

        crop = image_rgb[yy0:yy1, :]

        crops.append(
            {
                "index": index,
                "y0": int(yy0),
                "y1": int(yy1),
                "raw_region": (int(y0), int(y1)),
                "crop": crop,
            }
        )

    return crops