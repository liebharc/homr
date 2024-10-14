import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, TypeVar, cast

import cv2
import cv2.typing as cvt
import numpy as np
from scipy import ndimage  # type: ignore

from homr import constants
from homr.image_utils import crop_image
from homr.simple_logging import eprint
from homr.type_definitions import NDArray

TBounds = TypeVar("TBounds", bound="RotatedBoundingBox | BoundingBox | BoundingEllipse")


def rotate_point_around_center(
    point: tuple[float, float], center: tuple[float, float], angle: float
) -> tuple[float, float]:
    return (
        point[0] * np.cos(angle) - point[1] * np.sin(angle) + center[0],
        point[0] * np.sin(angle) + point[1] * np.cos(angle) + center[1],
    )


def calculate_edges_of_rotated_rectangle(
    box: cvt.RotatedRect,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]:
    half_size = np.array([box[1][0] / 2, box[1][1] / 2])
    center = box[0]
    top_left = center - half_size
    bottom_left = center + np.array([-half_size[0], half_size[1]])
    top_right = center + np.array([half_size[0], -half_size[1]])
    bottom_right = center + half_size
    return (
        (top_left[0], top_left[1]),
        (bottom_left[0], bottom_left[1]),
        (top_right[0], top_right[1]),
        (bottom_right[0], bottom_right[1]),
    )


def do_polygons_overlap(poly1: cvt.MatLike, poly2: cvt.MatLike) -> bool:
    # Check if any point of one ellipse is inside the other ellipse
    for point in poly1:
        if cv2.pointPolygonTest(poly2, (float(point[0]), float(point[1])), False) >= 0:
            return True
    for point in poly2:
        if cv2.pointPolygonTest(poly1, (float(point[0]), float(point[1])), False) >= 0:
            return True

    return False


class DebugDrawable(ABC):
    @abstractmethod
    def draw_onto_image(self, img: NDArray, color: tuple[int, int, int] = (0, 0, 255)) -> None:
        pass


class AnyPolygon(DebugDrawable):
    def __init__(self, polygon: Any):
        self.polygon = polygon


class BoundingBox(AnyPolygon):
    """
    A bounding box in the format of (x1, y1, x2, y2)
    """

    def __init__(self, box: cvt.Rect, contours: cvt.MatLike, debug_id: int = 0):
        self.debug_id = debug_id
        self.contours = contours
        self.box = box
        self.center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        self.size = (box[2] - box[0], box[3] - box[1])
        self.rotated_box = (self.center, self.size, 0)
        self.size = (box[2] - box[0], box[3] - box[1])
        super().__init__(cv2.boxPoints(self.rotated_box).astype(np.int64))

    def draw_onto_image(self, img: NDArray, color: tuple[int, int, int] = (0, 0, 255)) -> None:
        cv2.rectangle(
            img,
            (int(self.box[0]), int(self.box[1])),
            (int(self.box[2]), int(self.box[3])),
            color,
            2,
        )

    def rotate_and_extract(self, img: NDArray, angle: float) -> NDArray:
        rotated = ndimage.rotate(
            crop_image(img, self.box[0], self.box[1], self.box[2] + 1, self.box[3] + 1), angle
        )
        return cast(NDArray, rotated)

    def extract(self, img: NDArray) -> NDArray:
        return crop_image(img, self.box[0], self.box[1], self.box[2] + 1, self.box[3] + 1)

    def increase_height(self, y_top: int, y_bottom: int) -> "BoundingBox":
        return BoundingBox(
            (
                self.box[0],
                min(self.box[1], int(y_top)),
                self.box[2],
                max(self.box[1], int(y_bottom)),
            ),
            self.contours,
            self.debug_id,
        )

    def increase_width(self, x_left: int, x_right: int) -> "BoundingBox":
        return BoundingBox(
            (
                min(self.box[0], int(x_left)),
                self.box[1],
                max(self.box[0], int(x_right)),
                self.box[3],
            ),
            self.contours,
            self.debug_id,
        )

    def split_into_quadrants(self) -> list["BoundingBox"]:
        """
        Splits the bounding box into four equally sized quadrants.
        It returns them in the order: top left, top right, bottom left, bottom right
        """
        x_center = int(self.box[0] + self.size[0] / 2)
        y_center = int(self.box[1] + self.size[1] / 2)
        return [
            BoundingBox(
                (self.box[0], self.box[1], x_center, y_center), self.contours, self.debug_id
            ),
            BoundingBox(
                (x_center, self.box[1], self.box[2], y_center), self.contours, self.debug_id
            ),
            BoundingBox(
                (self.box[0], y_center, x_center, self.box[3]), self.contours, self.debug_id
            ),
            BoundingBox(
                (x_center, y_center, self.box[2], self.box[3]), self.contours, self.debug_id
            ),
        ]


class AngledBoundingBox(AnyPolygon):
    def __init__(
        self, box: cvt.RotatedRect, contours: cvt.MatLike, polygon: Any, debug_id: int = 0
    ):
        super().__init__(polygon)
        self.debug_id = debug_id
        self.contours = contours
        angle = box[2]
        self.box: cvt.RotatedRect
        if angle > 135:  # noqa: PLR2004
            angle = angle - 180
            self.box = ((box[0][0], box[0][1]), (box[1][0], box[1][1]), angle)
        elif angle < -135:  # noqa: PLR2004
            angle = angle + 180
            self.box = ((box[0][0], box[0][1]), (box[1][0], box[1][1]), angle)
        elif angle > 45:  # noqa: PLR2004
            angle = angle - 90
            self.box = ((box[0][0], box[0][1]), (box[1][1], box[1][0]), angle)
        elif angle < -45:  # noqa: PLR2004
            angle = angle + 90
            self.box = ((box[0][0], box[0][1]), (box[1][1], box[1][0]), angle)
        else:
            self.box = ((box[0][0], box[0][1]), (box[1][0], box[1][1]), angle)
        self.center = self.box[0]
        self.size = self.box[1]
        self.angle = self.box[2]
        self.top_left, self.bottom_left, self.top_right, self.bottom_right = (
            calculate_edges_of_rotated_rectangle(self.box)
        )
        self.polygon = polygon

    def is_overlapping(self, other: AnyPolygon) -> bool:
        if not self._can_shapes_possibly_touch(other):
            return False
        return do_polygons_overlap(self.polygon, other.polygon)

    def is_overlapping_with_any(self, others: Sequence["AngledBoundingBox"]) -> bool:
        for other in others:
            if self.is_overlapping(other):
                return True
        return False

    def _can_shapes_possibly_touch(self, other: "AnyPolygon") -> bool:
        """
        A fast check if the two shapes can possibly touch. If this returns False,
        the two shapes do not touch.
        If this returns True, the two shapes might touch and further checks are necessary.
        """

        # Get the centers and major axes of the rectangles
        center1, axes1, _ = self.box
        center2: Sequence[float]
        axes2: Sequence[float]
        if isinstance(other, BoundingBox):
            center2, axes2, _ = (
                other.rotated_box
            )  # (variable) rotated_box: tuple[tuple[float, float], tuple[int, int], Literal[0]]
        elif isinstance(other, AngledBoundingBox):
            center2, axes2, _ = (
                other.box
            )  # (variable) box: tuple[tuple[float, float], tuple[int, int], float]
        else:
            raise ValueError(f"Unknown type {type(other)}")
        major_axis1 = max(axes1)
        major_axis2 = max(axes2)

        # Calculate the distance between the centers
        distance = ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5

        # If the distance is greater than the sum of the major axes, the rectangles do not overlap
        if distance > major_axis1 + major_axis2:
            return False
        return True

    @abstractmethod
    def draw_onto_image(self, img: NDArray, color: tuple[int, int, int] = (0, 0, 255)) -> None:
        pass

    @abstractmethod
    def extract_point_sequence_from_image(self, img: NDArray) -> NDArray:
        pass

    def get_color_ratio(self, img: NDArray) -> float:
        """
        Gets the ratio of white to total pixels for this bounding box in the image.
        """
        colors = self.extract_point_sequence_from_image(img)
        white = len([color for color in colors if color == 1])
        total = len(colors)
        ratio = white / total
        return ratio

    def crop_rect_from_image(self, img: NDArray) -> NDArray:
        return crop_image(
            img,
            self.top_left[0],
            self.top_left[1],
            self.bottom_right[0] + 1,
            self.bottom_right[1] + 1,
        )


class RotatedBoundingBox(AngledBoundingBox):
    def __init__(self, box: cvt.RotatedRect, contours: cvt.MatLike, debug_id: int = 0):
        super().__init__(box, contours, cv2.boxPoints(box).astype(np.int64), debug_id)

    def draw_onto_image(self, img: NDArray, color: tuple[int, int, int] = (0, 0, 255)) -> None:
        box = cv2.boxPoints(self.box).astype(np.int64)
        cv2.drawContours(img, [box], 0, color, 2)

    def is_intersecting(self, other: "RotatedBoundingBox") -> bool:
        # TODO: How is this different from is_overlapping?
        return cv2.rotatedRectangleIntersection(self.box, other.box)[0] != cv2.INTERSECT_NONE

    def is_overlapping_extrapolated(self, other: "RotatedBoundingBox", unit_size: float) -> bool:
        return self._get_intersection_point_extrapolated(other, unit_size) is not None

    def make_box_thicker(self, thickness: int) -> "RotatedBoundingBox":
        if thickness <= 0:
            return self
        return RotatedBoundingBox(
            (
                (self.box[0][0], self.box[0][1]),
                (self.box[1][0] + thickness, self.box[1][1] + thickness),
                self.box[2],
            ),
            self.contours,
            self.debug_id,
        )

    def move_to_x_horizontal_by(self, x_delta: int) -> "RotatedBoundingBox":
        new_x = self.center[0] + x_delta
        return RotatedBoundingBox(
            ((new_x, self.center[1]), self.box[1], self.box[2]), self.contours, self.debug_id
        )

    def make_box_taller(self, thickness: int) -> "RotatedBoundingBox":
        return RotatedBoundingBox(
            (
                (self.box[0][0], self.box[0][1]),
                (self.box[1][0], self.box[1][1] + thickness),
                self.box[2],
            ),
            self.contours,
            self.debug_id,
        )

    def get_center_extrapolated(self, x: float) -> float:
        return (x - self.box[0][0]) * np.tan(self.box[2] / 180 * np.pi) + self.box[0][1]  # type: ignore

    def _get_intersection_point_extrapolated(
        self, other: "RotatedBoundingBox", unit_size: float
    ) -> tuple[float, float] | None:
        if self.box[0][0] > other.box[0][0]:
            left, right = other, self
        else:
            left, right = self, other
        center: float = float(np.mean([left.center[0], right.center[0]]))

        tolerance = constants.tolerance_for_staff_line_detection(unit_size)
        max_gap = constants.max_line_gap_size(unit_size)
        distance_between_left_and_center_considering_size = (
            center - left.center[0] - left.size[0] // 2
        )
        distance_between_right_and_center_considering_size = (
            right.center[0] - center - right.size[0] // 2
        )
        if (
            distance_between_left_and_center_considering_size > max_gap
            or distance_between_right_and_center_considering_size > max_gap
        ):
            return None
        left_at_center = left.get_center_extrapolated(center)
        right_at_center = right.get_center_extrapolated(center)
        if abs(left_at_center - right_at_center) > tolerance:
            return None
        return (center, (left_at_center + right_at_center) / 2)

    def extract_point_sequence_from_image(self, img: NDArray) -> NDArray:
        rectangle = self.box
        poly = cv2.boxPoints(rectangle).astype(np.int64)

        # Create an empty mask
        mask = np.zeros_like(img)

        # Fill the polygon in the mask
        cv2.fillPoly(mask, [poly], 1)  # type: ignore

        # Use the mask to index the image
        points = img[mask == 1]

        return points  # type: ignore

    def to_bounding_box(self) -> BoundingBox:
        return BoundingBox(
            (
                int(self.top_left[0]),
                int(self.top_left[1]),
                int(self.bottom_right[0]),
                int(self.bottom_right[1]),
            ),
            self.contours,
            self.debug_id,
        )


class BoundingEllipse(AngledBoundingBox):
    def __init__(self, box: cvt.RotatedRect, contours: cvt.MatLike, debug_id: int = 0):
        super().__init__(
            box,
            contours,
            cv2.ellipse2Poly(
                (int(box[0][0]), int(box[0][1])),
                (int(box[1][0] / 2), int(box[1][1] / 2)),
                int(box[2]),
                0,
                360,
                1,
            ),
            debug_id,
        )

    def draw_onto_image(self, img: NDArray, color: tuple[int, int, int] = (0, 0, 255)) -> None:
        cv2.ellipse(img, self.box, color=color, thickness=2)

    def make_box_thicker(self, thickness: int) -> "BoundingEllipse":
        return BoundingEllipse(
            (
                (self.box[0][0], self.box[0][1]),
                (self.box[1][0] + thickness, self.box[1][1] + thickness),
                self.box[2],
            ),
            self.contours,
            self.debug_id,
        )

    def make_box_taller(self, thickness: int) -> "RotatedBoundingBox":
        return RotatedBoundingBox(
            (
                (self.box[0][0], self.box[0][1]),
                (self.box[1][0], self.box[1][1] + thickness),
                self.box[2],
            ),
            self.contours,
            self.debug_id,
        )

    def extract_point_sequence_from_image(self, img: NDArray) -> NDArray:
        # Create an empty mask
        mask = np.zeros_like(img)

        # Fill the polygon in the mask
        cv2.fillPoly(mask, [self.polygon], 1)  # type: ignore

        # Use the mask to index the image
        points = img[mask == 1]

        return points  # type: ignore


def create_bounding_boxes(img: NDArray) -> list[BoundingBox]:
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for i, countour in enumerate(contours):
        boxes.append(create_bounding_box(countour, debug_id=i))
    return boxes


def create_bounding_box(contour: cvt.MatLike, debug_id: int) -> BoundingBox:
    x, y, w, h = cv2.boundingRect(contour)
    box = (x, y, x + w, y + h)
    return BoundingBox(box, contour, debug_id=debug_id)


def _has_box_valid_size(box: cvt.RotatedRect) -> bool:
    return (
        not math.isnan(box[1][0]) and not math.isnan(box[1][1]) and box[1][0] > 0 and box[1][1] > 0
    )


def create_rotated_bounding_boxes(
    img: NDArray,
    skip_merging: bool = False,
    min_size: tuple[int, int] | None = None,
    max_size: tuple[int, int] | None = None,
    thicken_boxes: int | None = None,
) -> list[RotatedBoundingBox]:
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[RotatedBoundingBox] = []
    for i, countour in enumerate(contours):
        fitBox = cv2.minAreaRect(countour)
        if not _has_box_valid_size(fitBox):
            continue
        box = RotatedBoundingBox(fitBox, countour, debug_id=i)
        if min_size is not None and (box.size[0] < min_size[0] or box.size[1] < min_size[1]):
            continue
        if max_size is not None:
            if max_size[0] > 0 and box.size[0] > max_size[0]:
                continue
            if max_size[1] > 0 and box.size[1] > max_size[1]:
                continue
        boxes.append(box)
    if skip_merging:
        return boxes
    if thicken_boxes is not None:
        boxes = [box.make_box_thicker(thicken_boxes) for box in boxes]
    return _get_box_for_whole_group(merge_overlaying_bounding_boxes(boxes))


def create_rotated_bounding_box(contour: cvt.MatLike, debug_id: int) -> RotatedBoundingBox:
    box = cv2.minAreaRect(contour)
    return RotatedBoundingBox(box, contour, debug_id=debug_id)


def create_lines(
    img: NDArray,
    threshold: int = 100,
    min_line_length: int = 100,
    max_line_gap: int = 10,
    skip_merging: bool = False,
) -> list[RotatedBoundingBox]:
    lines = cv2.HoughLinesP(
        img, 1, np.pi / 180, threshold, minLineLength=min_line_length, maxLineGap=max_line_gap
    )
    boxes = []
    for i, line in enumerate(lines):
        x1, y1, x2, y2 = line[0]
        contour = np.array([[x1, y1], [x2, y2]])
        box = cv2.minAreaRect(contour)
        if box[1][0] > box[1][1]:
            boxes.append(RotatedBoundingBox(box, contour, debug_id=i))
    if skip_merging:
        return boxes
    return _get_box_for_whole_group(merge_overlaying_bounding_boxes(boxes))


def create_bounding_ellipses(
    img: NDArray,
    skip_merging: bool = False,
    min_size: tuple[int, int] | None = None,
    max_size: tuple[int, int] | None = None,
) -> list[BoundingEllipse]:
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for i, countour in enumerate(contours):
        min_length_to_fit_ellipse = 5  # this is a requirement by opencv
        if len(countour) < min_length_to_fit_ellipse:
            continue
        fitBox = cv2.fitEllipse(countour)
        if not _has_box_valid_size(fitBox):
            continue
        box = BoundingEllipse(fitBox, countour, debug_id=i)
        if min_size is not None and (box.size[0] < min_size[0] or box.size[1] < min_size[1]):
            continue
        if max_size is not None and (box.size[0] > max_size[0] or box.size[1] > max_size[1]):
            continue
        boxes.append(box)
    if skip_merging:
        return boxes
    return _get_ellipse_for_whole_group(merge_overlaying_bounding_boxes(boxes))


def move_overlaying_bounding_boxes(
    src: list[AngledBoundingBox], dest: list[AngledBoundingBox], dest_img: NDArray
) -> tuple[list[AngledBoundingBox], NDArray]:
    """
    Every item in src which overlaps with one in dest will be transferred to dest_img
    """
    result_img = dest_img.copy()
    result_src = src.copy()
    for dest_box in dest:
        for src_box in src:
            if src_box.is_overlapping(dest_box):
                result_img[src_box.contours] = 1
                if src_box in result_src:
                    result_src.remove(src_box)
    return result_src, result_img


def _do_groups_overlap(group1: list[AngledBoundingBox], group2: list[AngledBoundingBox]) -> bool:
    for box1 in group1:
        for box2 in group2:
            if box1.is_overlapping(box2):
                return True
    return False


def _merge_groups_recursive(
    groups: list[list[AngledBoundingBox]], step: int
) -> list[list[AngledBoundingBox]]:
    step_limit = 10
    if step > step_limit:
        eprint("Too many steps in _merge_groups_recursive, giving back current results")
        return groups
    number_of_changes = 0
    merged: list[list[AngledBoundingBox]] = []
    used_groups = set()
    for i, group in enumerate(groups):
        match_found = False
        if i in used_groups:
            continue
        for j in range(i + 1, len(groups)):
            if j in used_groups:
                continue
            other_group = groups[j]
            if _do_groups_overlap(group, other_group):
                merged.append(group + other_group)
                number_of_changes += 1
                used_groups.add(j)
                match_found = True
                break
        if not match_found:
            merged.append(group)

    if number_of_changes == 0:
        return merged
    else:
        return _merge_groups_recursive(merged, step + 1)


def get_largest_of_every_group(groups: list[list[AngledBoundingBox]]) -> list[AngledBoundingBox]:
    result = []
    for group in groups:
        largest = max(group, key=lambda box: box.size[0] * box.size[1])
        result.append(largest)
    return result


def _get_ellipse_for_whole_group(groups: list[list[AngledBoundingBox]]) -> list[BoundingEllipse]:
    result = []
    for group in groups:
        complete_contour = np.concatenate([box.contours for box in group])
        box = cv2.minAreaRect(complete_contour)
        result.append(BoundingEllipse(box, complete_contour))
    return result


def _get_box_for_whole_group(groups: list[list[AngledBoundingBox]]) -> list[RotatedBoundingBox]:
    result = []
    for group in groups:
        complete_contour = np.concatenate([box.contours for box in group])
        box = cv2.minAreaRect(complete_contour)
        result.append(RotatedBoundingBox(box, complete_contour))
    return result


def merge_overlaying_bounding_boxes(
    boxes: Sequence[AngledBoundingBox],
) -> list[list[AngledBoundingBox]]:
    initial_groups: list[list[AngledBoundingBox]] = []
    for box in boxes:
        initial_groups.append([box])
    return _merge_groups_recursive(initial_groups, 0)
