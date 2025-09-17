import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import cv2
import cv2.typing as cvt
import numpy as np

from homr import constants
from homr.image_utils import crop_image
from homr.simple_logging import eprint
from homr.type_definitions import NDArray


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
        if cv2.pointPolygonTest(poly2, (float(point[0]), float(point[1])), False) >= 0:  # type: ignore
            return True
    for point in poly2:
        if cv2.pointPolygonTest(poly1, (float(point[0]), float(point[1])), False) >= 0:  # type: ignore
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

    def extract(self, img: NDArray) -> NDArray:
        return crop_image(img, self.box[0], self.box[1], self.box[2] + 1, self.box[3] + 1)

    def blank_everything_outside_of_box(self, img: NDArray) -> NDArray:
        x1, y1, x2, y2 = self.box
        img = img.copy()
        white_background = np.full_like(img, 255, dtype=np.uint8)
        white_background[y1:y2, x1:x2] = img[y1:y2, x1:x2]
        return white_background

    def increase_size_in_each_dimension(
        self, increase: int, image_size: tuple[int, ...]
    ) -> "BoundingBox":
        return BoundingBox(
            (
                max(self.box[0] - increase, 0),
                max(self.box[1] - increase, 0),
                min(self.box[2] + increase, image_size[1]),
                min(self.box[3] + increase, image_size[0]),
            ),
            self.contours,
            self.debug_id,
        )


class AngledBoundingBox(AnyPolygon):
    def __init__(
        self, box: cvt.RotatedRect, contours: cvt.MatLike, polygon: Any, debug_id: int = 0
    ):
        super().__init__(polygon)
        self.debug_id = debug_id
        self.contours = contours
        angle = box[2]
        self.box: cvt.RotatedRect
        if angle > 135:
            angle = angle - 180
            self.box = ((box[0][0], box[0][1]), (box[1][0], box[1][1]), angle)
        elif angle < -135:
            angle = angle + 180
            self.box = ((box[0][0], box[0][1]), (box[1][0], box[1][1]), angle)
        elif angle > 45:
            angle = angle - 90
            self.box = ((box[0][0], box[0][1]), (box[1][1], box[1][0]), angle)
        elif angle < -45:
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

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, AngledBoundingBox):
            return self.box == __value.box
        else:
            return False

    def __hash__(self) -> int:
        return hash(self.box)

    def __str__(self) -> str:
        return f"{self.box}"

    def __repr__(self) -> str:
        return str(self)

    @abstractmethod
    def draw_onto_image(self, img: NDArray, color: tuple[int, int, int] = (0, 0, 255)) -> None:
        pass


class RotatedBoundingBox(AngledBoundingBox):
    def __init__(self, box: cvt.RotatedRect, contours: cvt.MatLike, debug_id: int = 0):
        super().__init__(box, contours, cv2.boxPoints(box).astype(np.int64), debug_id)

    def draw_onto_image(self, img: NDArray, color: tuple[int, int, int] = (0, 0, 255)) -> None:
        box = cv2.boxPoints(self.box).astype(np.int64)
        cv2.drawContours(img, [box], 0, color, 2)

    def is_intersecting(self, other: "RotatedBoundingBox") -> bool:
        if not self._can_shapes_possibly_touch(other):
            return False
        # TODO: How is this different from is_overlapping?
        return cv2.rotatedRectangleIntersection(self.box, other.box)[0] != cv2.INTERSECT_NONE

    def ensure_min_dimension(self, min_width: int, min_height: int) -> "RotatedBoundingBox":
        return RotatedBoundingBox(
            (
                (self.box[0][0], self.box[0][1]),
                (max(self.box[1][0], min_width), max(self.box[1][1], min_height)),
                self.box[2],
            ),
            self.contours,
            self.debug_id,
        )

    def make_box_thicker(self, thickness: int) -> "RotatedBoundingBox":
        if thickness <= 0:
            return self
        # We tried to move the center by int(thickness / 2), however this gave much worse results
        # for some examples
        # That's possibly a case of an ill defined function, but downstream code depends on the
        # behavior which we have today
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

    def make_box_taller_keep_center(self, thickness: int) -> "RotatedBoundingBox":
        return RotatedBoundingBox(
            (
                (self.box[0][0], self.box[0][1] - thickness // 2),
                (self.box[1][0], self.box[1][1] + thickness),
                self.box[2],
            ),
            self.contours,
            self.debug_id,
        )

    def get_center_extrapolated(self, x: float) -> float:
        return (x - self.box[0][0]) * np.tan(self.box[2] / 180 * np.pi) + self.box[0][1]

    def is_overlapping_extrapolated(self, other: "RotatedBoundingBox", unit_size: float) -> bool:
        # Pick left and right by x-coordinate of the first box corner
        if self.box[0][0] > other.box[0][0]:
            left, right = other, self
        else:
            left, right = self, other

        center_x = (left.center[0] + right.center[0]) * 0.5

        tolerance = constants.tolerance_for_staff_line_detection(unit_size)
        max_gap = constants.max_line_gap_size(unit_size)

        if (
            center_x - left.center[0] - (left.size[0] // 2) > max_gap
            or right.center[0] - center_x - (right.size[0] // 2) > max_gap
        ):
            return False

        # Compute extrapolated y-values, same as get_center_extrapolated but inlined
        left_angle = math.tan(left.box[2] * math.pi / 180.0)
        right_angle = math.tan(right.box[2] * math.pi / 180.0)

        left_y = (center_x - left.box[0][0]) * left_angle + left.box[0][1]
        right_y = (center_x - right.box[0][0]) * right_angle + right.box[0][1]

        return abs(left_y - right_y) <= tolerance

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
        x1, y1, x2, y2 = line[0]  # type: ignore
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


class UnionFind:
    def __init__(self, n: int):
        self.parent: list[int] = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> None:
        rootX = self.find(x)
        rootY = self.find(y)

        if rootX != rootY:
            # Union by rank to keep the tree flat
            if self.rank[rootX] > self.rank[rootY]:
                self.parent[rootY] = rootX
            elif self.rank[rootX] < self.rank[rootY]:
                self.parent[rootX] = rootY
            else:
                self.parent[rootY] = rootX
                self.rank[rootX] += 1


def _merge_groups_optimized(groups: list[list[AngledBoundingBox]]) -> list[list[AngledBoundingBox]]:
    n = len(groups)
    uf = UnionFind(n)

    # Try to find overlaps and union groups that overlap
    for i in range(n):
        for j in range(i + 1, n):
            if _do_groups_overlap(groups[i], groups[j]):
                uf.union(i, j)

    # Create merged groups based on the union-find results
    merged_groups: dict[int, list[AngledBoundingBox]] = {}
    for i in range(n):
        root = uf.find(i)
        if root not in merged_groups:
            merged_groups[root] = []
        merged_groups[root].extend(groups[i])

    return list(merged_groups.values())


def merge_overlaying_bounding_boxes(
    boxes: Sequence[AngledBoundingBox],
) -> list[list[AngledBoundingBox]]:
    initial_groups: list[list[AngledBoundingBox]] = []
    for box in boxes:
        initial_groups.append([box])
    return _merge_groups_optimized(initial_groups)
