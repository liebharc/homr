import cv2
import numpy as np
import PIL.Image
import torch
from torchvision import transforms as tr
from torchvision.transforms import Compose

from homr.staff_dewarping import warp_image_randomly
from homr.type_definitions import NDArray


def read_image_to_ndarray(path: str) -> NDArray:
    img: NDArray = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # type: ignore
    if img is None:
        raise ValueError("Failed to read image from " + path)

    if len(img.shape) == 3 and img.shape[-1] == 4:
        img = 255 - img[:, :, 3]
    return img


def prepare_for_tensor(img: NDArray) -> NDArray:
    if len(img.shape) == 3 and img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def ndarray_to_tensor(img: NDArray, mean: float = 0.7931, std: float = 0.1738) -> torch.Tensor:
    img_array = img.astype(np.float32) / 255.0
    tensor = torch.tensor(img_array)
    mean_t = torch.tensor([mean]).view(1, 1, 1)
    std_t = torch.tensor([std]).view(1, 1, 1)
    return (tensor - mean_t) / std_t


def rotate_and_unsqueeze(tensor: torch.Tensor) -> torch.Tensor:
    if len(tensor.shape) == 2:
        tensor = torch.rot90(tensor, k=-1, dims=(0, 1))
        tensor = tensor.unsqueeze(0)
    else:
        tensor = torch.rot90(tensor, k=-1, dims=(1, 2))
    return tensor


def add_margin(image: NDArray, top: int, bottom: int, left: int, right: int) -> NDArray:
    # image is 2D grayscale
    flat = image.reshape(-1)
    bg = np.bincount(flat).argmax()

    h, w = image.shape

    top_pad = np.full((top, w), bg, dtype=image.dtype)
    bottom_pad = np.full((bottom, w), bg, dtype=image.dtype)

    img_tb = np.vstack([top_pad, image, bottom_pad])

    new_h = img_tb.shape[0]
    left_pad = np.full((new_h, left), bg, dtype=image.dtype)
    right_pad = np.full((new_h, right), bg, dtype=image.dtype)

    return np.hstack([left_pad, img_tb, right_pad])


def distort_image(image: NDArray) -> NDArray:
    image, _background_value = _add_random_gray_tone(image)
    pil_image = PIL.Image.fromarray(image)

    pipeline = Compose(
        [
            tr.RandomRotation(degrees=2),
            tr.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.5),
            tr.RandomAdjustSharpness(0),
            tr.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 0.5)),
        ]
    )

    augmented_image = pipeline(img=pil_image)
    augmented_image = warp_image_randomly(augmented_image)

    # add paper texture overlay
    paper_texture = np.random.normal(loc=235, scale=10, size=image.shape).astype(np.uint8)  # type: ignore
    augmented_array = np.array(augmented_image)
    alpha = 0.2
    augmented_array = cv2.addWeighted(augmented_array, 1 - alpha, paper_texture, alpha, 0)

    rows, cols = augmented_array.shape[:2]

    # Randomized Gaussian parameters
    sigma_x = np.random.uniform(cols / 4, cols)
    sigma_y = np.random.uniform(rows / 4, rows)
    center_x = np.random.uniform(0, cols)
    center_y = np.random.uniform(0, rows)

    # Generate coordinate grids
    x = np.arange(cols) - center_x
    y = np.arange(rows) - center_y
    x_grid, y_grid = np.meshgrid(x, y)

    # Optional rotation
    theta = np.random.uniform(0, 2 * np.pi)
    x_rot = x_grid * np.cos(theta) + y_grid * np.sin(theta)
    y_rot = -x_grid * np.sin(theta) + y_grid * np.cos(theta)

    # Create asymmetric Gaussian vignette
    mask = np.exp(-0.5 * ((x_rot**2 / sigma_x**2) + (y_rot**2 / sigma_y**2)))
    mask = mask / mask.max()

    # Scale mask to subtle effect (0.9–1.0)
    mask = 0.9 + 0.1 * mask

    # Optional channel variation
    if len(augmented_array.shape) == 3 and augmented_array.shape[2] == 3:
        mask = np.stack([mask * np.random.uniform(0.9, 1.0) for _ in range(3)], axis=-1)

    augmented_array = (augmented_array * mask).astype(np.uint8)

    return augmented_array


def _add_random_gray_tone(image_arr: NDArray) -> tuple[NDArray, int]:
    """
    Adds a gray background. While doing so it ensures
    that all symbols are still visible on the image.
    """
    if len(image_arr.shape) == 2:
        gray = image_arr
    else:
        gray = cv2.cvtColor(image_arr, cv2.COLOR_BGR2GRAY)

    lightest_pixel_value = _find_lighest_non_white_pixel(gray)

    minimum_contrast = 70
    pure_white = 255

    if lightest_pixel_value >= pure_white - minimum_contrast:
        return image_arr, pure_white

    strongest_possible_gray = lightest_pixel_value + minimum_contrast

    random_gray_value = np.random.randint(strongest_possible_gray, pure_white)
    if random_gray_value >= pure_white:
        return image_arr, random_gray_value

    if len(image_arr.shape) == 3:
        mask = np.all(image_arr > random_gray_value, axis=-1)
    else:
        mask = image_arr > random_gray_value

    jitter = np.random.randint(-5, 5, size=mask.shape)
    gray_bg = np.clip(random_gray_value + jitter, 0, pure_white)

    if len(image_arr.shape) == 3:
        image_arr[mask] = np.stack([gray_bg[mask]] * 3, axis=-1)
    else:
        image_arr[mask] = gray_bg[mask]

    return image_arr, random_gray_value


def _find_lighest_non_white_pixel(gray: NDArray) -> int:
    pure_white = 255
    valid_pixels = gray[gray < pure_white]

    if valid_pixels.size > 0:
        return valid_pixels.max()
    else:
        return pure_white
