import cv2
import numpy as np
import PIL.Image
from skimage import transform

from homr.debug import Debug
from homr.model import Staff
from homr.simple_logging import eprint
from homr.type_definitions import NDArray


class StaffDewarping:
    def __init__(self, tform: transform.PiecewiseAffineTransform | None):
        self.tform = tform

    def dewarp(self, image: NDArray, fill_color: int = 1, order: int = 1) -> NDArray:
        if self.tform is None:
            return image
        return transform.warp(  # type: ignore
            image,
            self.tform.inverse,
            output_shape=image.shape,
            mode="constant",
            order=order,
            cval=fill_color,
        )

    def dewarp_point(self, point: tuple[float, float]) -> tuple[float, float]:
        if self.tform is None:
            return point
        return self.tform(point)  # type: ignore


def is_point_on_image(pts: tuple[int, int], image: NDArray) -> bool:
    height, width = image.shape[:2]
    margin = 10
    if pts[0] < margin or pts[0] > width - margin or pts[1] < margin or pts[1] > height - margin:
        return False
    return True


def calculate_span_and_optimal_points(
    staff: Staff, image: NDArray
) -> tuple[list[list[tuple[int, int]]], list[list[tuple[int, int]]]]:
    span_points: list[list[tuple[int, int]]] = []
    optimal_points: list[list[tuple[int, int]]] = []
    first_y_offset = None
    number_of_y_intervals = 6
    if int(image.shape[0] / number_of_y_intervals) == 0:
        return span_points, optimal_points
    for y in range(2, image.shape[0] - 2, int(image.shape[0] / number_of_y_intervals)):
        line_points: list[tuple[int, int]] = []
        for x in range(2, image.shape[1], 80):
            y_values = staff.get_at(x)
            if y_values is not None:
                y_offset = y_values.y[2]
                if not first_y_offset:
                    first_y_offset = y_offset
                    y_delta = 0
                else:
                    y_delta = y_offset - first_y_offset
                point = (x, y + y_delta)
                if is_point_on_image(point, image):
                    line_points.append(point)

        minimum_number_of_points = 2
        if len(line_points) > minimum_number_of_points:
            average_y = sum([p[1] for p in line_points]) / len(line_points)
            span_points.append(line_points)
            optimal_points.append([(p[0], int(average_y)) for p in line_points])
    return span_points, optimal_points


class FastPiecewiseAffineTransform(transform.PiecewiseAffineTransform):
    """
    From https://github.com/scikit-image/scikit-image/pull/6963/files
    """

    def __call__(self, coords):  # type: ignore
        coords = np.asarray(coords)

        simplex = self._tesselation.find_simplex(coords)

        affines = np.stack([affine.params for affine in self.affines])[simplex]

        points = np.c_[coords, np.ones((coords.shape[0], 1))]

        result = np.einsum("ikj,ij->ik", affines, points)
        result[simplex == -1, :] = -1
        result = result[:, :2]

        return result


def calculate_dewarp_transformation(
    image: NDArray,
    source: list[list[tuple[int, int]]],
    destination: list[list[tuple[int, int]]],
    fast: bool = False,
) -> StaffDewarping:
    def add_image_edges_to_lines(
        lines: list[list[tuple[int, int]]],
    ) -> list[list[tuple[int, int]]]:
        lines.insert(0, [(0, 0), (0, image.shape[1])])
        lines.append([(image.shape[0], 0), (image.shape[0], image.shape[1])])
        return lines

    def add_first_and_last_point_to_every_line(
        lines: list[list[tuple[int, int]]],
    ) -> list[list[tuple[int, int]]]:
        for line in lines:
            line.insert(0, (0, line[0][1]))
            line.append((image.shape[1], line[-1][1]))
        return lines

    source = add_image_edges_to_lines(add_first_and_last_point_to_every_line(source))
    destination = add_image_edges_to_lines(add_first_and_last_point_to_every_line(destination))

    # Convert your points to numpy arrays
    source_conc = np.concatenate(source)
    destination_conc = np.concatenate(destination)

    tform = FastPiecewiseAffineTransform() if fast else transform.PiecewiseAffineTransform()  # type: ignore
    tform.estimate(source_conc, destination_conc)  # type: ignore
    return StaffDewarping(tform)


def dewarp_staff_image(image: NDArray, staff: Staff, index: int, debug: Debug) -> StaffDewarping:
    try:
        span_points, optimal_points = calculate_span_and_optimal_points(staff, image)
        if debug.debug:
            debug_img = image.copy()
            for line in span_points:
                for point in line:
                    cv2.circle(debug_img, [int(point[0]), int(point[1])], 5, (0, 0, 255), -1)
            for line in optimal_points:
                for point in line:
                    cv2.circle(debug_img, [int(point[0]), int(point[1])], 5, (255, 0, 0), -1)

            debug.write_image_with_fixed_suffix(f"_staff-{index}_debug_span_points.png", debug_img)
        return calculate_dewarp_transformation(image, span_points, optimal_points)
    except Exception as e:
        eprint("Dewarping failed for staff", index, "with error", e)
    return StaffDewarping(None)


def warp_image_randomly(image: PIL.Image.Image) -> PIL.Image.Image:
    array = np.array(image)
    result = warp_image_array_randomly(array)
    return PIL.Image.fromarray(result)


def warp_image_array_randomly(image: NDArray) -> NDArray:
    center = (image.shape[1] // 2, image.shape[0] // 2)
    num_points = 3
    upper = [(i * image.shape[1] // num_points, 0) for i in range(num_points)]
    source = [(i * image.shape[1] // num_points, center[1]) for i in range(num_points)]
    lower = [(i * image.shape[1] // num_points, image.shape[0]) for i in range(num_points)]
    max_random_offset = 10
    destination = [
        (
            i * image.shape[1] // num_points,
            center[1] + np.random.randint(-max_random_offset, max_random_offset),
        )
        for i in range(num_points)
    ]
    result = calculate_dewarp_transformation(
        image, [upper, source, lower], [upper, destination, lower], fast=True
    ).dewarp(image, order=3)
    return (255 * result).astype(np.uint8)


if __name__ == "__main__":
    import sys

    image = cv2.imread(sys.argv[1])
    cv2.imwrite(sys.argv[2], warp_image_array_randomly(image))
