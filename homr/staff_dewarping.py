import cv2
import numpy as np
import PIL.Image
from skimage import transform

from homr.types import NDArray


class StaffDewarping:
    def __init__(self, tform: transform.PiecewiseAffineTransform):
        self.tform = tform

    def dewarp(self, image: NDArray, fill_color: int = 1, order: int = 1) -> NDArray:
        return transform.warp(  # type: ignore
            image,
            self.tform.inverse,
            output_shape=image.shape,
            mode="constant",
            order=order,
            cval=fill_color,
        )


def calculate_dewarp_transformation(
    image: NDArray,
    source: list[list[tuple[float, float]]],
    destination: list[list[tuple[float, float]]],
) -> StaffDewarping:
    def add_image_edges_to_lines(
        lines: list[list[tuple[float, float]]],
    ) -> list[list[tuple[float, float]]]:
        lines.insert(0, [(0.0, 0.0), (0.0, float(image.shape[1]))])
        lines.append([(float(image.shape[0]), 0.0), (float(image.shape[0]), float(image.shape[1]))])
        return lines

    def add_first_and_last_point_to_every_line(
        lines: list[list[tuple[float, float]]],
    ) -> list[list[tuple[float, float]]]:
        for line in lines:
            line.insert(0, (0.0, float(line[0][1])))
            line.append((float(image.shape[1]), float(line[-1][1])))
        return lines

    source = add_image_edges_to_lines(add_first_and_last_point_to_every_line(source))
    destination = add_image_edges_to_lines(add_first_and_last_point_to_every_line(destination))

    # Convert your points to numpy arrays
    source_conc = np.concatenate(source)
    destination_conc = np.concatenate(destination)

    tform = transform.PiecewiseAffineTransform()  # type: ignore
    tform.estimate(source_conc, destination_conc)  # type: ignore
    return StaffDewarping(tform)


def warp_image_randomly(image: PIL.Image.Image) -> PIL.Image.Image:
    array = np.array(image)
    result = warp_image_array_randomly(array)
    return PIL.Image.fromarray(result)


def warp_image_array_randomly(image: NDArray) -> NDArray:
    center = (image.shape[1] / 2, image.shape[0] / 2)
    num_points = 3
    upper = [(i * image.shape[1] / num_points, 0.0) for i in range(num_points)]
    source = [(i * image.shape[1] / num_points, float(center[1])) for i in range(num_points)]
    lower = [(i * image.shape[1] / num_points, float(image.shape[0])) for i in range(num_points)]
    max_random_offset = 10
    destination = [
        (
            i * image.shape[1] / num_points,
            center[1] + np.random.randint(-max_random_offset, max_random_offset),
        )
        for i in range(num_points)
    ]
    result = calculate_dewarp_transformation(
        image, [upper, source, lower], [upper, destination, lower]
    ).dewarp(image, order=3)
    return (255 * result).astype(np.uint8)


if __name__ == "__main__":
    import sys

    image = cv2.imread(sys.argv[1])
    cv2.imwrite(sys.argv[2], warp_image_array_randomly(image))
