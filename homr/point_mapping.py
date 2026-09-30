"""
Maps points back through the image transformations of the pipeline.

Every step which moves pixels (autocrop, resizing, staff extraction, dewarping, placing
the staff on the transformer canvas) contributes one inverse mapping. Chained together
they take a point from the transformer input back to the image the user provided.
"""

from collections.abc import Callable

Point = tuple[float, float]
PointMapping = Callable[[Point], Point]


def identity(point: Point) -> Point:
    return point


def undo_crop(x: float, y: float) -> PointMapping:
    """
    Inverse of cropping an image at the top left corner (x, y).
    Pasting an image at (x, y) is a crop at (-x, -y).
    """

    def mapping(point: Point) -> Point:
        return point[0] + x, point[1] + y

    return mapping


def undo_resize(shape_before: tuple[int, ...], shape_after: tuple[int, ...]) -> PointMapping:
    """
    Inverse of resizing an image, the shapes are numpy shapes (height, width, ...).
    """
    scale_x = shape_after[1] / shape_before[1]
    scale_y = shape_after[0] / shape_before[0]

    def mapping(point: Point) -> Point:
        return point[0] / scale_x, point[1] / scale_y

    return mapping


def chain(*mappings: PointMapping) -> PointMapping:
    """
    Applies the mappings in the given order, so list the inverse of the last
    transformation first.
    """

    def mapping(point: Point) -> Point:
        for m in mappings:
            point = m(point)
        return point

    return mapping
