import glob
import os
from collections.abc import Sequence
from itertools import chain

import cv2
import numpy as np

from homr.bounding_boxes import DebugDrawable
from homr.type_definitions import NDArray


class AttentionDebug:
    def __init__(self, filename: str, image: NDArray, parent: "Debug") -> None:
        self.image = image
        self.destname = filename
        self.attentions: list[NDArray] = []
        self.parent = parent

    def add_attention(self, attention: NDArray, center: tuple[float, float]) -> None:
        attention_resized = cv2.resize(attention, (self.image.shape[1], self.image.shape[0]))
        # Apply a colormap to the attention weights
        attention_colormap = cv2.applyColorMap(  # type: ignore
            np.uint8(255.0 * attention_resized / attention_resized.max()), cv2.COLORMAP_JET  # type: ignore
        )
        overlay = cv2.addWeighted(self.image, 0.6, attention_colormap, 0.4, 0)

        # Draw the center of attention
        center_coordinates = (int(center[1]), int(center[0]))
        radius = 20
        color = (0, 255, 0)
        thickness = 2
        cv2.circle(overlay, center_coordinates, radius, color, thickness)

        self.attentions.append(overlay)

    def reset(self) -> None:
        self.attentions = []

    def write(self) -> None:
        if not self.attentions:
            return
        attention = cv2.vconcat(self.attentions)
        self.parent._remember_file_name(self.destname)
        cv2.imwrite(self.destname, attention)
        self.attentions = []


class Debug:
    def __init__(self, original_image: NDArray, filename: str, debug: bool):
        self.filename = filename
        self.original_image = original_image
        filename = filename.replace("\\", "/")
        self.dir_name = os.path.dirname(filename)
        self.base_filename = os.path.join(self.dir_name, filename.split("/")[-1].split(".")[0])
        self.debug = debug
        self.colors = [
            (0, 255, 0),
            (0, 0, 255),
            (255, 0, 0),
            (255, 255, 0),
            (0, 255, 255),
            (255, 0, 255),
            (255, 165, 0),
            (255, 182, 193),
            (128, 0, 128),
            (64, 224, 208),
        ]
        self.debug_output_counter = 0
        self.written_files: list[str] = []

    def clean_debug_files_from_previous_runs(self) -> None:
        prefixes = (
            self.base_filename + "_debug_",
            self.base_filename + "_tesseract_input",
            self.base_filename + "_staff-",
        )

        for file in glob.glob(self.base_filename + "*"):
            if file.startswith(prefixes) and file not in self.written_files:
                os.remove(file)

    def _debug_file_name(self, suffix: str) -> str:
        self.debug_output_counter += 1
        return f"{self.base_filename}_debug_{str(self.debug_output_counter)}_{suffix}.png"

    def write_threshold_image(self, suffix: str, image: NDArray) -> None:
        if not self.debug:
            return
        filename = self._debug_file_name(suffix)
        self._remember_file_name(filename)
        cv2.imwrite(filename, 255 * image)

    def _remember_file_name(self, filename: str) -> None:
        self.written_files.append(filename)

    def write_bounding_boxes(self, suffix: str, bounding_boxes: Sequence[DebugDrawable]) -> None:
        if not self.debug:
            return
        img = self.original_image.copy()
        for box in bounding_boxes:
            box.draw_onto_image(img)
        filename = self._debug_file_name(suffix)
        self._remember_file_name(filename)
        cv2.imwrite(filename, img)

    def write_image(self, suffix: str, image: NDArray) -> None:
        if not self.debug:
            return
        filename = self._debug_file_name(suffix)
        self._remember_file_name(filename)
        cv2.imwrite(filename, image)

    def write_image_with_fixed_suffix(self, suffix: str, image: NDArray) -> None:
        if not self.debug:
            return
        filename = self.base_filename + suffix
        self._remember_file_name(filename)
        cv2.imwrite(filename, image)

    def write_all_bounding_boxes_alternating_colors(
        self, suffix: str, *boxes: Sequence[DebugDrawable]
    ) -> None:
        self.write_bounding_boxes_alternating_colors(suffix, list(chain.from_iterable(boxes)))

    def write_bounding_boxes_alternating_colors(
        self, suffix: str, bounding_boxes: Sequence[DebugDrawable]
    ) -> None:
        if not self.debug:
            return
        self.write_teaser(self._debug_file_name(suffix), bounding_boxes)

    def write_teaser(self, filename: str, bounding_boxes: Sequence[DebugDrawable]) -> None:
        img = self.original_image.copy()
        for i, box in enumerate(bounding_boxes):
            color = self.colors[i % len(self.colors)]
            box.draw_onto_image(img, color)
        self._remember_file_name(filename)
        cv2.imwrite(filename, img)

    def write_model_input_image(self, suffix: str, staff_image: NDArray) -> str:
        """
        These files aren't really debug files, but it's convenient to handle them here
        so that they are cleaned up together with the debug files.

        Model input images are the input to the transformer or OCR images.
        """
        filename = self.base_filename + suffix
        if self.debug:
            self._remember_file_name(filename)
        cv2.imwrite(filename, staff_image)
        return filename

    def build_attention_debug(self, image: NDArray, suffix: str) -> AttentionDebug | None:
        if not self.debug:
            return None
        filename = self.base_filename + suffix
        self._remember_file_name(filename)
        return AttentionDebug(filename, image, self)
