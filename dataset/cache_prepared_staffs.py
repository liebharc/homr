"""
Cache HOMR-prepared TrOMR staff images using ONNX SegNet + deterministic HOMR geometry.

This file intentionally avoids importing homr.main because homr.main imports download_utils,
which imports requests and download-related code we do not need for the benchmark.

Pipeline:

    full-page image
        -> autocrop / resize / CLAHE              [HOMR deterministic Python]
        -> segnet.onnx via ONNX Runtime           [neural inference]
        -> mask postprocessing                    [HOMR deterministic Python]
        -> staff / note / barline geometry        [HOMR deterministic Python]
        -> homr.staff_parsing.prepare_staff_image [HOMR deterministic Python]
        -> save .npy float32 [0, 1], shape [256, 1280]
        -> STOP

This file does NOT call:
    - homr.main
    - homr.main.download_weights
    - homr.segmentation.inference_segnet.extract
    - homr.staff_parsing.parse_staffs
    - homr.staff_parsing.parse_staff_image
    - homr.staff_parsing_tromr.parse_staff_tromr
    - homr.transformer.staff2score.Staff2Score.predict

The saved .npy files are the correct Track B inputs for Square Attack.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort


ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from homr import color_adjust
from homr.autocrop import autocrop
from homr.bar_line_detection import detect_bar_lines, prepare_bar_line_image
from homr.bounding_boxes import (
    BoundingEllipse,
    RotatedBoundingBox,
    create_bounding_ellipses,
    create_rotated_bounding_boxes,
)
from homr.brace_dot_detection import (
    find_braces_brackets_and_grand_staff_lines,
    prepare_brace_dot_image,
)
from homr.debug import Debug
from homr.model import InputPredictions, MultiStaff
from homr.noise_filtering import filter_predictions
from homr.note_detection import add_notes_to_staffs, combine_noteheads_with_stems
from homr.resize import resize_image
from homr.simple_logging import eprint
from homr.staff_detection import break_wide_fragments, detect_staff, make_lines_stronger
from homr.staff_parsing import (
    _ensure_same_number_of_staffs,
    _get_number_of_voices,
    prepare_staff_image,
    tr_omr_max_height,
    tr_omr_max_width,
)
from homr.staff_regions import StaffRegions
from homr.type_definitions import NDArray


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")


@dataclass
class PredictedSymbols:
    noteheads: list[BoundingEllipse]
    staff_fragments: list[RotatedBoundingBox]
    clefs_keys: list[RotatedBoundingBox]
    stems_rest: list[RotatedBoundingBox]
    bar_lines: list[RotatedBoundingBox]


@dataclass
class CacheConfig:
    enable_debug: bool
    use_cuda: bool
    batch_size: int
    win_size: int
    step_size: int


def build_onnx_providers(use_cuda: bool) -> list[Any]:
    """
    Build ONNX Runtime providers.

    TensorRT is deliberately not used here. CUDAExecutionProvider is preferred,
    CPUExecutionProvider is fallback.
    """
    cuda_options = {
        "device_id": 0,
        "cudnn_conv_algo_search": "HEURISTIC",
        "arena_extend_strategy": "kNextPowerOfTwo",
    }

    available = ort.get_available_providers()

    if use_cuda and "CUDAExecutionProvider" in available:
        return [
            ("CUDAExecutionProvider", cuda_options),
            "CPUExecutionProvider",
        ]

    return ["CPUExecutionProvider"]


def build_patch_origins(length: int, win_size: int, step_size: int) -> list[int]:
    """
    Build patch origins for one image axis.

    Ensures the final patch covers the end of the image.
    """
    if length <= 0:
        raise ValueError(f"Invalid image axis length: {length}")

    if win_size <= 0:
        raise ValueError(f"Invalid window size: {win_size}")

    if step_size <= 0:
        raise ValueError(f"Invalid step size: {step_size}")

    if length <= win_size:
        return [0]

    origins = list(range(0, length - win_size + 1, step_size))
    last_origin = length - win_size

    if origins[-1] != last_origin:
        origins.append(last_origin)

    return origins


def extract_patch_chw(
    image_chw: np.ndarray,
    y: int,
    x: int,
    win_size: int,
    pad_value: float = 255.0,
) -> np.ndarray:
    """
    Extract one CHW patch, padding with white where needed.
    """
    channels, height, width = image_chw.shape

    patch = np.full(
        (channels, win_size, win_size),
        pad_value,
        dtype=np.float32,
    )

    y0 = max(y, 0)
    x0 = max(x, 0)
    y1 = min(y + win_size, height)
    x1 = min(x + win_size, width)

    patch_y0 = y0 - y
    patch_x0 = x0 - x
    patch_y1 = patch_y0 + (y1 - y0)
    patch_x1 = patch_x0 + (x1 - x0)

    patch[:, patch_y0:patch_y1, patch_x0:patch_x1] = image_chw[:, y0:y1, x0:x1]

    return patch


class SegNetONNX:
    """
    ONNX Runtime wrapper for SegNet.

    Important: this class merges six-channel SegNet score maps first and applies argmax only
    after reconstructing the full-page score map. It does NOT average categorical class IDs.
    """

    def __init__(
        self,
        model_path: Path,
        use_cuda: bool = True,
        batch_size: int = 8,
        win_size: int = 320,
        step_size: int = 320,
    ) -> None:
        self.model_path = Path(model_path)
        self.batch_size = int(batch_size)
        self.win_size = int(win_size)
        self.step_size = int(step_size)

        if not self.model_path.exists():
            raise FileNotFoundError(f"SegNet ONNX model not found: {self.model_path}")

        self.providers = build_onnx_providers(use_cuda=use_cuda)

        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=self.providers,
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_shape = self.session.get_outputs()[0].shape
    
    @staticmethod
    def ensure_three_channel_image(image: np.ndarray) -> np.ndarray:
        """
        Ensure image is [H, W, 3] before feeding SegNet ONNX.

        HOMR's CLAHE preprocessing may return grayscale [H, W], while
        segnet.onnx expects 3 channels. For grayscale input, duplicate the
        same channel into BGR-style 3-channel form.

        This is still ONNX inference. No HOMR neural Python model is used.
        """
        if image is None:
            raise ValueError("Input image is None.")

        arr = np.asarray(image)

        if arr.ndim == 2:
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

        if arr.ndim == 3 and arr.shape[2] == 1:
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

        if arr.ndim == 3 and arr.shape[2] == 3:
            return arr

        if arr.ndim == 3 and arr.shape[2] == 4:
            return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)

        raise ValueError(f"Unsupported image shape for SegNet: {arr.shape}")

    def predict_score_map(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Run sliding-window SegNet ONNX inference.

        Input:
            image_bgr:
                Either grayscale [H, W] or 3-channel image [H, W, 3].
                Grayscale input is expanded to 3 channels because segnet.onnx
                expects [B, 3, 320, 320].

        Output:
            score_map: shape [6, H, W], float32.
        """
        image_bgr = self.ensure_three_channel_image(image_bgr)

        image_chw = np.transpose(image_bgr, (2, 0, 1)).astype(np.float32)
        _, height, width = image_chw.shape

        y_origins = build_patch_origins(height, self.win_size, self.step_size)
        x_origins = build_patch_origins(width, self.win_size, self.step_size)

        score_sum: np.ndarray | None = None
        weight = np.zeros((height, width), dtype=np.float32)

        batch_patches: list[np.ndarray] = []
        batch_coords: list[tuple[int, int]] = []

        def flush_batch() -> None:
            nonlocal score_sum

            if not batch_patches:
                return

            batch_input = np.stack(batch_patches, axis=0).astype(np.float32)

            output = self.session.run(
                [self.output_name],
                {self.input_name: batch_input},
            )[0]

            if output.ndim != 4:
                raise ValueError(
                    f"Expected SegNet output [B, C, H, W], got shape {output.shape}"
                )

            batch_size_out, n_classes, out_h, out_w = output.shape

            if out_h != self.win_size or out_w != self.win_size:
                raise ValueError(
                    f"Expected SegNet patch output spatial size "
                    f"[{self.win_size}, {self.win_size}], got [{out_h}, {out_w}]"
                )

            if score_sum is None:
                score_sum = np.zeros((n_classes, height, width), dtype=np.float32)

            if batch_size_out != len(batch_coords):
                raise ValueError(
                    f"SegNet returned {batch_size_out} outputs for "
                    f"{len(batch_coords)} input patches."
                )

            for patch_scores, (y, x) in zip(output, batch_coords, strict=True):
                y0 = max(y, 0)
                x0 = max(x, 0)
                y1 = min(y + self.win_size, height)
                x1 = min(x + self.win_size, width)

                patch_y0 = y0 - y
                patch_x0 = x0 - x
                patch_y1 = patch_y0 + (y1 - y0)
                patch_x1 = patch_x0 + (x1 - x0)

                score_sum[:, y0:y1, x0:x1] += patch_scores[
                    :, patch_y0:patch_y1, patch_x0:patch_x1
                ]
                weight[y0:y1, x0:x1] += 1.0

            batch_patches.clear()
            batch_coords.clear()

        for y in y_origins:
            for x in x_origins:
                patch = extract_patch_chw(
                    image_chw=image_chw,
                    y=y,
                    x=x,
                    win_size=self.win_size,
                    pad_value=255.0,
                )

                batch_patches.append(patch)
                batch_coords.append((y, x))

                if len(batch_patches) >= self.batch_size:
                    flush_batch()

        flush_batch()

        if score_sum is None:
            raise RuntimeError("No SegNet patches were processed.")

        weight[weight == 0.0] = 1.0
        score_map = score_sum / weight[np.newaxis, :, :]

        return score_map.astype(np.float32)

    def predict_class_map(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Return integer semantic class map [H, W], values 0..5.
        """
        score_map = self.predict_score_map(image_bgr)
        class_map = np.argmax(score_map, axis=0).astype(np.uint8)
        return class_map


def find_image_paths(images_dir: Path, recursive: bool) -> list[Path]:
    """
    Find source page images without duplicate paths.

    Important on Windows:
        Path.glob("*.png") may also match files ending in ".PNG", so iterating
        over both lower- and upper-case extensions can return the same file
        twice. Deduplicate by resolved, case-normalized path before applying
        --limit. Without this, --limit 100 can become only 50 unique pages.
    """
    if recursive:
        paths: list[Path] = []
        for ext in IMAGE_EXTENSIONS:
            paths.extend(images_dir.rglob(f"*{ext}"))
    else:
        paths = []
        for ext in IMAGE_EXTENSIONS:
            paths.extend(images_dir.glob(f"*{ext}"))

    unique: dict[str, Path] = {}
    for path in paths:
        if (
            "_teaser" in path.name
            or "_debug" in path.name
            or "_staff" in path.name
            or "_tesseract" in path.name
        ):
            continue

        try:
            key = str(path.resolve()).casefold()
        except Exception:
            key = str(path.absolute()).casefold()

        unique[key] = path

    return sorted(unique.values(), key=lambda p: str(p).casefold())


def build_input_predictions_from_segnet(
    *,
    original_bgr: np.ndarray,
    preprocessed_bgr: np.ndarray,
    class_map: np.ndarray,
) -> InputPredictions:
    """
    Convert SegNet class map into HOMR InputPredictions.

    Class mapping:
        0 = background
        1 = stems / rests
        2 = noteheads
        3 = clefs / key signatures
        4 = staff lines
        5 = symbols
    """
    if class_map.ndim != 2:
        raise ValueError(f"Expected class_map [H, W], got {class_map.shape}")

    height, width = class_map.shape

    original_resized = cv2.resize(original_bgr, (width, height))
    preprocessed_resized = cv2.resize(preprocessed_bgr, (width, height))

    return InputPredictions(
        original=original_resized,
        preprocessed=preprocessed_resized,
        notehead=(class_map == 2).astype(np.uint8),
        symbols=(class_map == 5).astype(np.uint8),
        staff=(class_map == 4).astype(np.uint8),
        clefs_keys=(class_map == 3).astype(np.uint8),
        stems_rest=(class_map == 1).astype(np.uint8),
    )


def load_and_preprocess_predictions_onnx(
    *,
    image_path: Path,
    segnet: SegNetONNX,
    enable_debug: bool,
) -> tuple[InputPredictions, Debug]:
    """
    HOMR preprocessing + ONNX SegNet segmentation.

    This replaces homr.main.load_and_preprocess_predictions(...) without importing homr.main
    and without calling homr.segmentation.inference_segnet.extract(...).
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    image = autocrop(image)
    image = resize_image(image)
    preprocessed = color_adjust.apply_clahe(image)

    class_map = segnet.predict_class_map(preprocessed)

    predictions = build_input_predictions_from_segnet(
        original_bgr=image,
        preprocessed_bgr=preprocessed,
        class_map=class_map,
    )

    debug = Debug(predictions.original, str(image_path), enable_debug)
    debug.write_image("color_adjust", predictions.preprocessed)

    predictions = filter_predictions(predictions, debug)

    predictions.staff = make_lines_stronger(predictions.staff, (1, 2))

    debug.write_threshold_image("staff", predictions.staff)
    debug.write_threshold_image("symbols", predictions.symbols)
    debug.write_threshold_image("stems_rest", predictions.stems_rest)
    debug.write_threshold_image("notehead", predictions.notehead)
    debug.write_threshold_image("clefs_keys", predictions.clefs_keys)

    return predictions, debug


def predict_symbols(debug: Debug, predictions: InputPredictions) -> PredictedSymbols:
    """
    Copy of HOMR's deterministic symbol-bound construction, without importing homr.main.
    """
    eprint("Creating bounds for noteheads")
    noteheads = create_bounding_ellipses(predictions.notehead, min_size=(4, 4))

    eprint("Creating bounds for staff_fragments")
    staff_fragments = create_rotated_bounding_boxes(
        predictions.staff,
        skip_merging=True,
        min_size=(5, 1),
        max_size=(10000, 100),
    )

    eprint("Creating bounds for clefs_keys")
    clefs_keys = create_rotated_bounding_boxes(
        predictions.clefs_keys,
        min_size=(20, 40),
        max_size=(1000, 1000),
    )

    eprint("Creating bounds for stems_rest")
    stems_rest = create_rotated_bounding_boxes(predictions.stems_rest)

    eprint("Creating bounds for bar_lines")
    bar_line_img = prepare_bar_line_image(predictions.stems_rest)
    debug.write_threshold_image("bar_line_img", bar_line_img)
    bar_lines = create_rotated_bounding_boxes(
        bar_line_img,
        skip_merging=True,
        min_size=(1, 5),
    )

    return PredictedSymbols(
        noteheads=noteheads,
        staff_fragments=staff_fragments,
        clefs_keys=clefs_keys,
        stems_rest=stems_rest,
        bar_lines=bar_lines,
    )


def detect_staffs_in_image_onnx(
    *,
    image_path: Path,
    segnet: SegNetONNX,
    enable_debug: bool,
) -> tuple[list[MultiStaff], NDArray, Debug]:
    """
    Deterministic HOMR layout pipeline using ONNX SegNet masks.

    This mirrors the useful part of homr.main.detect_staffs_in_image(...), but deliberately
    skips title detection and does not import homr.main.
    """
    predictions, debug = load_and_preprocess_predictions_onnx(
        image_path=image_path,
        segnet=segnet,
        enable_debug=enable_debug,
    )

    symbols = predict_symbols(debug, predictions)

    symbols.staff_fragments = break_wide_fragments(symbols.staff_fragments)
    debug.write_bounding_boxes("staff_fragments", symbols.staff_fragments)
    eprint("Found " + str(len(symbols.staff_fragments)) + " staff line fragments")

    noteheads_with_stems = combine_noteheads_with_stems(
        symbols.noteheads,
        symbols.stems_rest,
    )
    debug.write_bounding_boxes_alternating_colors(
        "notehead_with_stems",
        noteheads_with_stems,
    )
    eprint("Found " + str(len(noteheads_with_stems)) + " noteheads")

    if len(noteheads_with_stems) == 0:
        raise RuntimeError("No noteheads found")

    average_note_head_height = float(
        np.median([notehead.notehead.size[1] for notehead in noteheads_with_stems])
    )
    eprint("Average note head height: " + str(average_note_head_height))

    all_noteheads = [notehead.notehead for notehead in noteheads_with_stems]
    all_stems = [note.stem for note in noteheads_with_stems if note.stem is not None]

    bar_lines_or_rests = [
        line
        for line in symbols.bar_lines
        if not line.is_overlapping_with_any(all_noteheads)
        and not line.is_overlapping_with_any(all_stems)
    ]

    bar_line_boxes = detect_bar_lines(
        bar_lines_or_rests,
        average_note_head_height,
    )
    debug.write_bounding_boxes_alternating_colors("bar_lines", bar_line_boxes)
    eprint("Found " + str(len(bar_line_boxes)) + " bar lines")

    debug.write_bounding_boxes(
        "anchor_input",
        symbols.staff_fragments + bar_line_boxes + symbols.clefs_keys,
    )

    staffs = detect_staff(
        debug,
        predictions.staff,
        symbols.staff_fragments,
        symbols.clefs_keys,
        bar_line_boxes,
    )

    if len(staffs) == 0:
        raise RuntimeError("No staffs found")

    debug.write_bounding_boxes_alternating_colors("staffs", staffs)

    brace_dot_img = prepare_brace_dot_image(
        predictions.symbols,
        predictions.staff,
    )
    debug.write_threshold_image("brace_dot", brace_dot_img)

    brace_dot = create_rotated_bounding_boxes(
        brace_dot_img,
        skip_merging=True,
        max_size=(100, -1),
    )

    notes = add_notes_to_staffs(
        staffs,
        noteheads_with_stems,
        predictions.symbols,
        predictions.notehead,
    )

    multi_staffs = find_braces_brackets_and_grand_staff_lines(
        debug,
        staffs,
        brace_dot,
    )

    eprint(
        "Found",
        len(multi_staffs),
        "connected staffs (after merging grand staffs, multiple voices): ",
        [len(staff.staffs) for staff in multi_staffs],
    )

    debug.write_all_bounding_boxes_alternating_colors(
        "notes",
        multi_staffs,
        notes,
    )

    return multi_staffs, predictions.preprocessed, debug


def ensure_uint8_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert prepared staff image to uint8 grayscale [H, W].
    """
    arr = np.asarray(image)

    if arr.ndim == 3:
        if arr.shape[2] == 1:
            arr = arr[:, :, 0]
        elif arr.shape[2] == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        elif arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2GRAY)
        else:
            raise ValueError(f"Unsupported image channel count: {arr.shape}")
    elif arr.ndim != 2:
        raise ValueError(f"Expected 2D or 3D image, got shape {arr.shape}")

    if arr.dtype == np.uint8:
        return arr

    arr = arr.astype(np.float32)

    if arr.max(initial=0.0) <= 1.5:
        arr = arr * 255.0

    return np.clip(arr, 0.0, 255.0).round().astype(np.uint8)


def uint8_to_float01(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image, got {image.dtype}")

    return (image.astype(np.float32) / 255.0).astype(np.float32)


def validate_prepared_shape(image: np.ndarray, source: str) -> None:
    expected_shape = (tr_omr_max_height, tr_omr_max_width)

    if tuple(image.shape) != expected_shape:
        raise ValueError(
            f"{source} produced prepared staff shape {image.shape}, "
            f"expected {expected_shape}."
        )


def cache_one_image(
    *,
    image_path: Path,
    output_dir: Path,
    segnet: SegNetONNX,
    enable_debug: bool,
    overwrite: bool,
    strict_shape: bool,
) -> dict[str, Any]:
    image_output_dir = output_dir / image_path.stem
    metadata_path = image_output_dir / "metadata.json"

    if metadata_path.exists() and not overwrite:
        with metadata_path.open("r", encoding="utf-8") as f:
            existing = json.load(f)

        return {
            "image": image_path.name,
            "status": "skipped_existing",
            "n_staffs": int(existing.get("n_staffs", 0)),
            "output_dir": str(image_output_dir),
        }

    image_output_dir.mkdir(parents=True, exist_ok=True)

    debug: Debug | None = None

    try:
        multi_staffs, preprocessed_image, debug = detect_staffs_in_image_onnx(
            image_path=image_path,
            segnet=segnet,
            enable_debug=enable_debug,
        )

        multi_staffs = _ensure_same_number_of_staffs(
            multi_staffs,
            preprocessed_image,
        )

        regions = StaffRegions(multi_staffs)
        number_of_voices = _get_number_of_voices(multi_staffs)

        metadata: dict[str, Any] = {
            "image_stem": image_path.stem,
            "source_image": str(image_path),
            "n_staffs": 0,
            "n_voices": int(number_of_voices),
            "canvas_shape": [int(tr_omr_max_height), int(tr_omr_max_width)],
            "cache_boundary": "homr.staff_parsing.prepare_staff_image",
            "neural_inference_used": {
                "segnet": "ONNX Runtime",
                "tromr_encoder": "not run",
                "tromr_decoder": "not run",
            },
            "staffs": [],
        }

        cache_index = 0

        for voice_index in range(number_of_voices):
            staffs_for_voice = [
                multi_staff.staffs[voice_index]
                for multi_staff in multi_staffs
            ]

            for staff_index_within_voice, staff in enumerate(staffs_for_voice):
                staff_image_raw, transformed_staff = prepare_staff_image(
                    debug=debug,
                    index=cache_index,
                    staff=staff,
                    staff_image=preprocessed_image,
                    regions=regions,
                )

                staff_image_uint8 = ensure_uint8_grayscale(staff_image_raw)

                if strict_shape:
                    validate_prepared_shape(
                        staff_image_uint8,
                        source=f"{image_path.name} staff {cache_index}",
                    )

                staff_image_float = uint8_to_float01(staff_image_uint8)

                npy_name = f"staff_{cache_index:03d}.npy"
                png_name = f"staff_{cache_index:03d}.png"

                npy_path = image_output_dir / npy_name
                png_path = image_output_dir / png_name

                np.save(npy_path, staff_image_float)

                ok = cv2.imwrite(str(png_path), staff_image_uint8)

                if not ok:
                    raise OSError(f"Failed to write {png_path}")

                metadata["staffs"].append(
                    {
                        "index": int(cache_index),
                        "voice_index": int(voice_index),
                        "staff_index_within_voice": int(staff_index_within_voice),
                        "filename_npy": npy_name,
                        "filename_png": png_name,
                        "shape": [int(x) for x in staff_image_float.shape],
                        "dtype": "float32",
                        "range": [0.0, 1.0],
                        "is_grandstaff": bool(
                            getattr(transformed_staff, "is_grandstaff", False)
                        ),
                        "gt_tokens": [],
                    }
                )

                cache_index += 1

        metadata["n_staffs"] = int(cache_index)

        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return {
            "image": image_path.name,
            "status": "processed",
            "n_staffs": int(cache_index),
            "output_dir": str(image_output_dir),
        }

    except Exception as exc:
        error_payload = {
            "image": image_path.name,
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

        with (image_output_dir / "error.json").open("w", encoding="utf-8") as f:
            json.dump(error_payload, f, indent=2)

        raise

    finally:
        if debug is not None:
            try:
                debug.clean_debug_files_from_previous_runs()
            except Exception:
                pass


def cache_prepared_staffs(
    *,
    images_dir: Path,
    output_dir: Path,
    model_path: Path,
    limit: int,
    recursive: bool,
    use_cuda: bool,
    batch_size: int,
    win_size: int,
    step_size: int,
    enable_debug: bool,
    overwrite: bool,
    strict_shape: bool,
    continue_on_error: bool,
) -> None:
    images_dir = images_dir.resolve()
    output_dir = output_dir.resolve()
    model_path = model_path.resolve()

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    if not model_path.exists():
        raise FileNotFoundError(f"SegNet ONNX model not found: {model_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = find_image_paths(images_dir, recursive=recursive)

    if limit > 0:
        image_paths = image_paths[:limit]

    if not image_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")

    segnet = SegNetONNX(
        model_path=model_path,
        use_cuda=use_cuda,
        batch_size=batch_size,
        win_size=win_size,
        step_size=step_size,
    )

    print("SegNet providers:", segnet.session.get_providers())
    print("SegNet input:", segnet.input_name, segnet.input_shape)
    print("SegNet output:", segnet.output_name, segnet.output_shape)

    summary: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for image_index, image_path in enumerate(image_paths, start=1):
        print(f"\n[{image_index}/{len(image_paths)}] {image_path.name}")

        try:
            item = cache_one_image(
                image_path=image_path,
                output_dir=output_dir,
                segnet=segnet,
                enable_debug=enable_debug,
                overwrite=overwrite,
                strict_shape=strict_shape,
            )

            summary.append(item)
            print(f"  {item['status']}: {item['n_staffs']} prepared staff image(s)")

        except Exception as exc:
            item = {
                "image": image_path.name,
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

            summary.append(item)
            errors.append(item)

            print(f"  ERROR: {type(exc).__name__}: {exc}")

            if not continue_on_error:
                break

    payload = {
        "images_dir": str(images_dir),
        "output_dir": str(output_dir),
        "model_path": str(model_path),
        "n_images_selected": len(image_paths),
        "n_processed": sum(1 for item in summary if item["status"] == "processed"),
        "n_skipped": sum(1 for item in summary if item["status"] == "skipped_existing"),
        "n_errors": len(errors),
        "total_staffs_cached": sum(
            int(item.get("n_staffs", 0))
            for item in summary
            if item["status"] in {"processed", "skipped_existing"}
        ),
        "items": summary,
        "errors": errors,
    }

    summary_path = output_dir / "summary.json"

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSummary written to {summary_path}")

    if errors and not continue_on_error:
        raise RuntimeError(
            "Stopped after first error. Use --continue-on-error to process remaining images."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cache HOMR-prepared TrOMR staff images using ONNX SegNet "
            "and deterministic HOMR layout code."
        )
    )

    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("dataset/images"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/cached_prepared_staffs"),
    )

    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/onnx/segnet.onnx"),
    )

    parser.add_argument(
        "-N",
        "--limit",
        type=int,
        default=-1,
        help="Number of source images to process. Use -1 for all.",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPUExecutionProvider for SegNet ONNX.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--win-size",
        type=int,
        default=320,
    )

    parser.add_argument(
        "--step-size",
        type=int,
        default=320,
        help="SegNet sliding-window step size. HOMR main uses 320.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    parser.add_argument(
        "--no-strict-shape",
        action="store_true",
        help="Do not require prepared staff images to be exactly [256, 1280].",
    )

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cache_prepared_staffs(
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        limit=args.limit,
        recursive=args.recursive,
        use_cuda=not args.cpu,
        batch_size=args.batch_size,
        win_size=args.win_size,
        step_size=args.step_size,
        enable_debug=args.debug,
        overwrite=args.overwrite,
        strict_shape=not args.no_strict_shape,
        continue_on_error=args.continue_on_error,
    )


if __name__ == "__main__":
    main()