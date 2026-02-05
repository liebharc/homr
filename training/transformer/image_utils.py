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


def prepare_white_background(image: NDArray) -> NDArray:
    """
    Ensure image starts with a white background by replacing pure white pixels
    and very light pixels with pure white (255). This simulates realistic scanning
    where content is placed on white paper.
    """
    if image.ndim == 2:
        # Grayscale: replace pixels > 250 with pure white
        mask = image > 250
        image = image.copy()
        image[mask] = 255
    elif image.ndim == 3:
        # RGB: replace pixels where all channels > 250
        mask = np.all(image > 250, axis=-1)
        image = image.copy()
        image[mask] = 255
    return image


def add_random_margins(
    image: NDArray,
    min_margin: int = 10,
    max_margin: int = 100,
    fill_white: bool = True,
) -> NDArray:
    """
    Add random margins on all four sides of the image.

    Args:
        image: Input image (H, W) or (H, W, C)
        min_margin: Minimum margin size in pixels
        max_margin: Maximum margin size in pixels
        fill_white: If True, use white (255) fill; otherwise use detected background

    Returns:
        Image with random margins added
    """
    top = np.random.randint(min_margin, max_margin + 1)
    bottom = np.random.randint(min_margin, max_margin + 1)
    left = np.random.randint(min_margin, max_margin + 1)
    right = np.random.randint(min_margin, max_margin + 1)

    if fill_white:
        # Use pure white fill
        if image.ndim == 2:
            padded = np.pad(
                image, ((top, bottom), (left, right)), mode="constant", constant_values=255
            )
        else:
            padded = np.pad(
                image, ((top, bottom), (left, right), (0, 0)), mode="constant", constant_values=255
            )
        return padded
    else:
        # Use existing add_margin with background detection
        return add_margin(image, top, bottom, left, right)


def apply_ink_fade(image: NDArray, p: float = 0.15) -> NDArray:
    """
    Apply gradient-based ink fading to simulate aged or faded ink.
    This is more realistic than uniform brightness changes.

    Args:
        image: Input image
        p: Probability of applying this augmentation

    Returns:
        Image with optional ink fade applied
    """
    if np.random.random() > p:
        return image

    h, w = image.shape[:2]

    # Choose gradient type
    gradient_type = np.random.choice(["linear", "radial", "noise"])

    if gradient_type == "linear":
        # Linear gradient in random direction
        angle = np.random.uniform(0, 2 * np.pi)
        x = np.arange(w)
        y = np.arange(h)
        xx, yy = np.meshgrid(x, y)
        gradient = xx * np.cos(angle) + yy * np.sin(angle)
        gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min())
    elif gradient_type == "radial":
        # Radial gradient from random center
        center_x = np.random.uniform(0, w)
        center_y = np.random.uniform(0, h)
        x = np.arange(w) - center_x
        y = np.arange(h) - center_y
        xx, yy = np.meshgrid(x, y)
        gradient = np.sqrt(xx**2 + yy**2)
        gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min())
    else:  # noise-based
        # Smooth noise gradient
        gradient = np.random.randn(h // 10, w // 10)
        gradient = cv2.resize(gradient, (w, h), interpolation=cv2.INTER_LINEAR)
        gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min())

    # Randomly flip gradient direction
    if np.random.random() < 0.5:
        gradient = 1 - gradient

    # Apply subtle fade (max 30-40 brightness units to preserve legibility)
    fade_strength = np.random.uniform(20, 40)
    fade_mask = gradient * fade_strength

    if image.ndim == 2:
        faded = image.astype(np.float32) + fade_mask
    else:
        fade_mask = np.stack([fade_mask] * image.shape[2], axis=-1)
        faded = image.astype(np.float32) + fade_mask

    return np.clip(faded, 0, 255).astype(np.uint8)


def apply_diverse_noise(image: NDArray, p: float = 0.5) -> NDArray:
    """
    Apply diverse noise patterns with varying shapes, sizes, and intensities.
    This simulates real-world scanning artifacts and paper texture.

    Args:
        image: Input image
        p: Probability of applying noise

    Returns:
        Image with diverse noise applied
    """
    if np.random.random() > p:
        return image

    result = image.copy().astype(np.float32)
    h, w = image.shape[:2]

    # Apply 1-3 different noise types
    num_noise_types = np.random.randint(1, 4)

    for _ in range(num_noise_types):
        noise_type = np.random.choice(["patch", "gaussian", "salt_pepper"])

        if noise_type == "patch":
            # Patch-based noise with varying shapes and sizes
            num_patches = np.random.randint(5, 20)
            for _ in range(num_patches):
                # Random patch size
                patch_h = np.random.randint(5, 50)
                patch_w = np.random.randint(5, 50)

                # Random position
                y = np.random.randint(0, max(1, h - patch_h))
                x = np.random.randint(0, max(1, w - patch_w))

                # Random noise variance
                noise_std = np.random.uniform(5, 25)

                # Create patch noise
                patch_noise = np.random.normal(0, noise_std, (patch_h, patch_w))

                # Random shape (rectangular or circular)
                if np.random.random() < 0.3:
                    # Circular mask
                    cy, cx = patch_h // 2, patch_w // 2
                    y_grid, x_grid = np.ogrid[:patch_h, :patch_w]
                    mask = (y_grid - cy) ** 2 + (x_grid - cx) ** 2 <= (
                        min(patch_h, patch_w) // 2
                    ) ** 2
                    patch_noise = patch_noise * mask

                # Apply patch noise
                if image.ndim == 2:
                    result[y : y + patch_h, x : x + patch_w] += patch_noise
                else:
                    for c in range(image.shape[2]):
                        result[y : y + patch_h, x : x + patch_w, c] += patch_noise

        elif noise_type == "gaussian":
            # Global Gaussian noise with random intensity
            noise_std = np.random.uniform(3, 12)
            if image.ndim == 2:
                noise = np.random.normal(0, noise_std, (h, w))
                result += noise
            else:
                for c in range(image.shape[2]):
                    noise = np.random.normal(0, noise_std, (h, w))
                    result[:, :, c] += noise

        else:  # salt_pepper
            # Salt and pepper noise
            amount = np.random.uniform(0.001, 0.005)
            num_salt = int(amount * h * w * 0.5)
            num_pepper = int(amount * h * w * 0.5)

            # Salt (white pixels)
            coords_y = np.random.randint(0, h, num_salt)
            coords_x = np.random.randint(0, w, num_salt)
            if image.ndim == 2:
                result[coords_y, coords_x] = 255
            else:
                result[coords_y, coords_x, :] = 255

            # Pepper (black pixels)
            coords_y = np.random.randint(0, h, num_pepper)
            coords_x = np.random.randint(0, w, num_pepper)
            if image.ndim == 2:
                result[coords_y, coords_x] = 0
            else:
                result[coords_y, coords_x, :] = 0

    return np.clip(result, 0, 255).astype(np.uint8)


def apply_clahe(image: NDArray, p: float = 0.1) -> NDArray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to amplify
    existing noise and texture. This should be applied AFTER noise addition.

    Args:
        image: Input image
        p: Probability of applying CLAHE (default 0.1 = 10% of images)

    Returns:
        Image with optional CLAHE applied
    """
    if np.random.random() > p:
        return image

    # Conservative CLAHE parameters to avoid artifacts
    clip_limit = np.random.uniform(2.0, 3.0)
    tile_grid_size = (8, 8)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if image.ndim == 2:
        return clahe.apply(image)
    else:
        # Apply to each channel
        result = np.zeros_like(image)
        for c in range(image.shape[2]):
            result[:, :, c] = clahe.apply(image[:, :, c])
        return result


def distort_image(image: NDArray, allow_occlusions: bool = False) -> NDArray:
    """
    Apply data augmentation to an image.

    Args:
        image: Input image array
        allow_occlusions: If True, applies occlusive augmentations like annotations,
                         dropout, and shadows that may hide musical elements.
                         Set to False for validation to preserve all information.
                         Defaults to False for safety.

    New pipeline order:
        1. Prepare white background
        2. Add pre-canvas random margins (white fill)
        3. Apply geometric transforms (with white fill)
        4. Apply morphological operations
        5. Apply aging/brightness effects
        6. Apply optional ink fade with gradients
        7. Apply color jitter
        8. Apply blur/sharpen
        9. Apply diverse noise
        10. Apply optional CLAHE
        11. Apply minimal occlusions (if training mode)
        12. Apply warp_image_randomly
        13. Apply paper texture
        14. Apply vignette
        15. Add post-canvas random margins (white fill)
    """
    # Step 1: Prepare white background
    image = prepare_white_background(image)

    # Step 2: Add pre-canvas random margins (white fill)
    image = add_random_margins(image, min_margin=0, max_margin=20, fill_white=True)

    # Determine background value for transforms that need it
    image, background_value = _add_random_gray_tone(image)

    # Build transform list for albumentations
    transforms_list = [
        # Step 3: Geometric distortions with WHITE fill (255, 255, 255)
        A.Rotate(
            limit=2,
            border_mode=cv2.BORDER_CONSTANT,
            fill=(255, 255, 255),  # White fill for all channels
            p=0.3,
        ),
        # Perspective distortion (simulates photographed sheet music)
        A.Perspective(
            scale=(0.005, 0.015),
            fit_output=True,
            border_mode=cv2.BORDER_CONSTANT,
            fill=(255, 255, 255),  # White fill for all channels
            p=0.6,
        ),
        # Step 4: Stroke width variations (different pens, printing quality, scan artifacts)
        A.Morphological(scale=(2, 3), operation="erosion", p=0.2),  # thins strokes
        # Step 5: Aging and fading effects (reduced severity)
        A.OneOf(
            [
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=(-0.2, -0.1), p=1.0
                ),
                A.RandomToneCurve(scale=0.2, p=1.0),
                A.MultiplicativeNoise(multiplier=(0.85, 1.0), per_channel=True, p=1.0),
            ],
            p=0.15,
        ),
        # Step 7: Standard color/brightness variations (reduced)
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.03, p=0.4),
        # Step 8: Blur and sharpness (reduced)
        A.OneOf(
            [
                A.GaussianBlur(blur_limit=(3, 5), sigma_limit=(0.1, 0.4), p=1.0),
                A.Sharpen(alpha=(0.0, 0.1), lightness=(0.9, 1.0), p=1.0),
            ],
            p=0.3,
        ),
    ]

    # Step 11: Only add MINIMAL occlusive/destructive augmentations for training
    # These are HEAVILY reduced to prevent the model from learning to hallucinate
    if allow_occlusions:
        transforms_list.extend(
            [
                # Uneven exposure/scanning artifacts - REDUCED severity
                A.RandomShadow(
                    shadow_roi=(0, 0, 1, 1),
                    num_shadows_limit=(1, 2),  # Reduced from (1, 3)
                    shadow_dimension=3,  # Reduced from 5
                    p=0.05,  # HEAVILY reduced from 0.2
                ),
                # Annotations, markings, and occlusions - REDUCED
                # Small holes only, much less destructive
                A.CoarseDropout(
                    num_holes_range=(1, 3),  # Reduced from (2, 8)
                    hole_height_range=(5, 15),  # Reduced from (10, 30)
                    hole_width_range=(10, 30),  # Reduced from (20, 60)
                    fill=255,  # White fill instead of black
                    p=0.1,  # HEAVILY reduced from 0.4
                ),
                # Moisture damage, age spots - REDUCED
                A.RandomFog(
                    fog_coef_range=(0.05, 0.15),  # Reduced from (0.1, 0.3)
                    alpha_coef=0.05,  # Reduced from 0.1
                    p=0.05,  # HEAVILY reduced from 0.2
                ),
            ]
        )

    transform = A.Compose(transforms_list)

    # Apply albumentations
    transformed = transform(image=image)
    augmented_image = transformed["image"]

    # Step 6: Apply optional ink fade with gradients (NEW)
    augmented_image = apply_ink_fade(augmented_image, p=0.15)

    # Step 9: Apply diverse noise (NEW - replaces old noise augmentation)
    augmented_image = apply_diverse_noise(augmented_image, p=0.5)

    # Step 10: Apply optional CLAHE to amplify existing noise (NEW)
    augmented_image = apply_clahe(augmented_image, p=0.1)

    # Step 12: Apply custom staff dewarping
    pil_image = Image.fromarray(augmented_image)
    augmented_image = warp_image_randomly(pil_image)
    augmented_array = np.array(augmented_image)

    # Step 13: Paper texture overlay (non-destructive, reduced intensity)
    paper_texture = np.random.normal(loc=240, scale=8, size=augmented_array.shape).astype(np.uint8)
    alpha = 0.15  # Reduced from 0.2 for subtlety
    augmented_array = cv2.addWeighted(augmented_array, 1 - alpha, paper_texture, alpha, 0)

    # Step 14: Vignette effect (non-destructive)
    augmented_array = _apply_vignette(augmented_array)

    # Step 15: Add post-canvas random margins (white fill)
    augmented_array = add_random_margins(
        augmented_array, min_margin=0, max_margin=20, fill_white=True
    )

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
