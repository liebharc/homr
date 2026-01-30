import albumentations as A
import cv2
import numpy as np
import torch
from PIL import Image

from homr.staff_dewarping import warp_image_randomly
from homr.type_definitions import NDArray


def read_image_to_ndarray(path: str) -> NDArray:
    img: NDArray = cv2.imread(path, cv2.IMREAD_COLOR_BGR)  # type: ignore
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
    # Works for (H, W) and (H, W, C) by treating channels uniformly

    if image.ndim == 2:
        image = image[..., None]  # promote to (H, W, 1)
        squeeze = True
    elif image.ndim == 3:
        squeeze = False
    else:
        raise ValueError("Unsupported image shape")

    h, w, c = image.shape

    # per-channel background via mode
    bg = np.array(
        [np.bincount(image[..., ch].ravel()).argmax() for ch in range(c)],
        dtype=image.dtype,
    )

    top_pad = np.full((top, w, c), bg, dtype=image.dtype)
    bottom_pad = np.full((bottom, w, c), bg, dtype=image.dtype)
    img_tb = np.vstack([top_pad, image, bottom_pad])

    new_h = img_tb.shape[0]
    left_pad = np.full((new_h, left, c), bg, dtype=image.dtype)
    right_pad = np.full((new_h, right, c), bg, dtype=image.dtype)

    out = np.hstack([left_pad, img_tb, right_pad])

    return out[..., 0] if squeeze else out


def distort_image(image: NDArray, allow_occlusions: bool = False) -> NDArray:
    """
    Apply data augmentation to an image.

    Args:
        image: Input image array
        allow_occlusions: If True, applies occlusive augmentations like annotations,
                         dropout, and shadows that may hide musical elements.
                         Set to False for validation to preserve all information.
                         Defaults to False for safety.
    """
    image, background_value = _add_random_gray_tone(image)

    # Build transform list conditionally
    transforms_list = [
        # Geometric distortions (reduced from p=1.0)
        A.Rotate(limit=2, border_mode=cv2.BORDER_CONSTANT, fill=background_value, p=0.3),
        # Perspective distortion (simulates photographed sheet music)
        A.Perspective(scale=(0.02, 0.05), fit_output=True, p=0.2),
        # Stroke width variations (different pens, printing quality, scan artifacts)
        A.OneOf(
            [
                A.Morphological(scale=(2, 3), operation="erosion", p=1.0),  # thins strokes
                A.Morphological(scale=(2, 3), operation="dilation", p=1.0),  # thickens strokes
            ],
            p=0.2,
        ),  # Apply either erosion OR dilation 20% of the time
        # Aging and fading effects (reduced from p=0.3)
        A.OneOf(
            [
                A.RandomBrightnessContrast(
                    brightness_limit=0.3, contrast_limit=(-0.3, -0.1), p=1.0
                ),
                A.RandomToneCurve(scale=0.3, p=1.0),
                A.MultiplicativeNoise(multiplier=(0.8, 1.0), per_channel=True, p=1.0),
            ],
            p=0.2,
        ),
        # Standard color/brightness variations (reduced from p=0.8)
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.5, p=0.5),
        # Blur and sharpness (reduced from p=0.8)
        A.OneOf(
            [
                A.GaussianBlur(blur_limit=(3, 5), sigma_limit=(0.1, 0.5), p=1.0),
                A.Sharpen(alpha=(0.0, 0.1), lightness=(0.9, 1.0), p=1.0),
            ],
            p=0.5,
        ),
        # Paper texture and noise (kept at p=0.5)
        A.OneOf(
            [
                A.GaussNoise(
                    std_range=(0.02, 0.06),
                    mean_range=(0.0, 0.0),
                    per_channel=False,
                    noise_scale_factor=0.25,
                    p=1.0,
                ),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3), p=1.0),
            ],
            p=0.5,
        ),
    ]

    # Only add occlusive/destructive augmentations for training
    if allow_occlusions:
        transforms_list.extend(
            [
                # Uneven exposure/scanning artifacts
                A.RandomShadow(
                    shadow_roi=(0, 0, 1, 1),
                    num_shadows_limit=(1, 3),
                    shadow_dimension=5,
                    p=0.2,
                ),
                # Annotations, markings, and occlusions
                A.CoarseDropout(
                    num_holes_range=(2, 8),
                    hole_height_range=(10, 30),
                    hole_width_range=(20, 60),
                    fill=0,
                    p=0.4,
                ),
                # Moisture damage, age spots, localized fading
                A.RandomFog(
                    fog_coef_range=(0.1, 0.3),
                    alpha_coef=0.1,
                    p=0.2,
                ),
            ]
        )

    transform = A.Compose(transforms_list)

    # Apply albumentations
    transformed = transform(image=image)
    augmented_image = transformed["image"]

    # Apply custom staff dewarping
    pil_image = Image.fromarray(augmented_image)
    augmented_image = warp_image_randomly(pil_image)
    augmented_array = np.array(augmented_image)

    # Paper texture overlay (non-destructive)
    paper_texture = np.random.normal(loc=235, scale=10, size=augmented_array.shape).astype(np.uint8)
    alpha = 0.2
    augmented_array = cv2.addWeighted(augmented_array, 1 - alpha, paper_texture, alpha, 0)

    # Vignette effect (non-destructive)
    augmented_array = _apply_vignette(augmented_array)

    return augmented_array


def _apply_vignette(image: NDArray) -> NDArray:
    """Apply random asymmetric vignette effect to simulate scanner/camera artifacts."""
    rows, cols = image.shape[:2]

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

    # Optional channel variation for color images
    if len(image.shape) == 3 and image.shape[2] == 3:
        mask = np.stack([mask * np.random.uniform(0.9, 1.0) for _ in range(3)], axis=-1)

    return (image * mask).astype(np.uint8)


def _add_random_gray_tone(image_arr: NDArray) -> tuple[NDArray, int]:
    """
    Add a random gray background tone while ensuring minimum contrast is maintained.
    This simulates different paper colors and scanning conditions.
    """
    if len(image_arr.shape) == 2:
        gray = image_arr
    else:
        gray = cv2.cvtColor(image_arr, cv2.COLOR_BGR2GRAY)

    lightest_pixel_value = _find_lighest_non_white_pixel(gray)

    # Occasionally use lower minimum contrast for more challenging cases (20% of time)
    if np.random.random() < 0.2:
        minimum_contrast = np.random.randint(50, 70)
    else:
        minimum_contrast = 70

    pure_white = 255
    if lightest_pixel_value >= pure_white - minimum_contrast:
        return image_arr, pure_white

    strongest_possible_gray = lightest_pixel_value + minimum_contrast
    random_gray_value = np.random.randint(strongest_possible_gray, pure_white)

    if random_gray_value >= pure_white:
        return image_arr, random_gray_value

    # Create mask for background pixels
    if len(image_arr.shape) == 3:
        mask = np.all(image_arr > random_gray_value, axis=-1)
    else:
        mask = image_arr > random_gray_value

    # Add jitter to background for texture
    jitter = np.random.randint(-5, 5, size=mask.shape)
    gray_bg = np.clip(random_gray_value + jitter, 0, pure_white)

    # Apply background
    if len(image_arr.shape) == 3:
        image_arr[mask] = np.stack([gray_bg[mask]] * 3, axis=-1)
    else:
        image_arr[mask] = gray_bg[mask]

    return image_arr, random_gray_value


def _find_lighest_non_white_pixel(gray: NDArray) -> int:
    """Find the lightest pixel value that isn't pure white (background)."""
    pure_white = 255
    valid_pixels = gray[gray < pure_white]
    if valid_pixels.size > 0:
        return valid_pixels.max()
    else:
        return pure_white
