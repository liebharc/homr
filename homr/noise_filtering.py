import cv2
import numpy as np

from homr import constants
from homr.debug import Debug
from homr.model import InputPredictions
from homr.simple_logging import eprint
from homr.type_definitions import NDArray


def estimate_noise(gray: NDArray) -> int:
    H, W = gray.shape
    M = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]])
    sigma = np.sum(np.sum(np.absolute(cv2.filter2D(gray, cv2.CV_64F, M)))) / (H * W)
    return sigma  # type: ignore


def create_noise_grid(gray: NDArray, debug: Debug) -> NDArray | None:  # noqa: C901, PLR0912
    imgheight, imgwidth = gray.shape
    M, N = imgheight // 20, imgwidth // 20

    debug_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    mask = np.zeros(gray.shape, dtype=np.uint8)

    grid = create_grid(gray, M, N)
    filtered_cells, total_cells = apply_noise_filter(grid, mask, debug_image, M, N)

    if debug.debug:
        debug.write_image("noise_crop", debug_image)

    return handle_filter_results(filtered_cells, total_cells, mask)


def create_grid(gray: NDArray, M: int, N: int) -> NDArray:
    imgheight, imgwidth = gray.shape
    grid = np.zeros([int(np.ceil(imgheight / M)), int(np.ceil(imgwidth / N))], dtype=np.uint8)

    for i, y1 in enumerate(range(0, imgheight, M)):
        for j, x1 in enumerate(range(0, imgwidth, N)):
            y2, x2 = y1 + M, x1 + N
            tile = gray[y1:y2, x1:x2]
            noise = estimate_noise(tile)
            grid[i, j] = noise

    return grid


def apply_noise_filter(
    grid: NDArray, mask: NDArray, debug_image: NDArray, M: int, N: int
) -> tuple[int, int]:
    imgheight, imgwidth = mask.shape
    filtered_cells, total_cells = 0, 0

    for i, y1 in enumerate(range(0, imgheight, M)):
        for j, x1 in enumerate(range(0, imgwidth, N)):
            y2, x2 = y1 + M, x1 + N
            noise = grid[i, j]
            neighbors = get_neighbors(grid, i, j)
            any_neighbor_above_limit = np.any(np.array(neighbors) > constants.image_noise_limit)

            if noise > constants.image_noise_limit and any_neighbor_above_limit:
                cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 255))
                filtered_cells += 1
            else:
                mask[y1:y2, x1:x2] = 255
                cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0))

            cv2.putText(
                debug_image,
                f"{noise:.2f}",
                (x1 + N // 2, y1 + M // 2),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (0, 0, 255),
            )
            total_cells += 1

    return filtered_cells, total_cells


def get_neighbors(grid: NDArray, i: int, j: int) -> list[int]:
    neighbors = []
    if i > 0:
        neighbors.append(grid[i - 1, j])
    if j > 0:
        neighbors.append(grid[i, j - 1])
    if i < grid.shape[0] - 1:
        neighbors.append(grid[i + 1, j])
    if j < grid.shape[1] - 1:
        neighbors.append(grid[i, j + 1])
    return neighbors


def handle_filter_results(filtered_cells: int, total_cells: int, mask: NDArray) -> NDArray | None:
    half = 0.5
    if filtered_cells / total_cells > half:
        eprint(
            f"Would filter more than 50% of the image with {filtered_cells} of {total_cells} "
            + "cells, skipping noise filtering"
        )
        return None
    elif filtered_cells > 0:
        eprint(f"Filtered {filtered_cells} of {total_cells} cells")
        return mask
    return None


def filter_predictions(prediction: InputPredictions, debug: Debug) -> InputPredictions:
    mask = create_noise_grid(255 * prediction.staff, debug)
    if mask is None:
        return prediction
    return InputPredictions(
        original=cv2.bitwise_and(prediction.original, prediction.original, mask=mask),
        preprocessed=cv2.bitwise_and(prediction.preprocessed, prediction.preprocessed, mask=mask),
        notehead=cv2.bitwise_and(prediction.notehead, prediction.notehead, mask=mask),
        symbols=cv2.bitwise_and(prediction.symbols, prediction.symbols, mask=mask),
        staff=cv2.bitwise_and(prediction.staff, prediction.staff, mask=mask),
        clefs_keys=cv2.bitwise_and(prediction.clefs_keys, prediction.clefs_keys, mask=mask),
        stems_rest=cv2.bitwise_and(prediction.stems_rest, prediction.stems_rest, mask=mask),
    )
