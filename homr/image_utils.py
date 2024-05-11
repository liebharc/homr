from homr.types import NDArray


def _limit_x(image: NDArray, x: float) -> int:
    return max(0, min(image.shape[1] - 1, int(round(x))))


def _limit_y(image: NDArray, y: float) -> int:
    return max(0, min(image.shape[0] - 1, int(round(y))))


def crop_image(image: NDArray, x1: float, y1: float, x2: float, y2: float) -> NDArray:
    x_min = min(x1, x2)
    x_max = max(x1, x2)
    y_min = min(y1, y2)
    y_max = max(y1, y2)
    x1_limited = _limit_x(image, x_min)
    y1_limited = _limit_y(image, y_min)
    x2_limited = _limit_x(image, x_max)
    y2_limited = _limit_y(image, y_max)
    return image[y1_limited:y2_limited, x1_limited:x2_limited]
