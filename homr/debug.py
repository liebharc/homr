import os
from collections.abc import Sequence
from itertools import chain

import cv2

from homr.bounding_boxes import DebugDrawable
from homr.type_definitions import NDArray


def show_image_on_screen(img: NDArray, name: str = "image", wait: bool = True) -> None:
    cv2.imshow(name, img)
    if wait:
        cv2.waitKey(0)


def show_image_and_boxes_on_screen(
    img: NDArray,
    boxes: Sequence[DebugDrawable],
    name: str = "image",
    wait: bool = True,
) -> None:
    img = cv2.cvtColor(255 * img, cv2.COLOR_GRAY2BGR)
    for box in boxes:
        box.draw_onto_image(img)
    cv2.imshow(name, img)
    if wait:
        cv2.waitKey(0)


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
        if not self.debug:
            return

        prefixes = (
            self.base_filename + "_debug_",
            self.base_filename + "_tesseract_input",
            self.base_filename + "_staff-",
        )

        for file in os.listdir("."):
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
