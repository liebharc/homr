import albumentations as alb  # type: ignore
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2  # type: ignore

from homr.transformer.configs import Config
from homr.types import NDArray


def normalize(image: NDArray) -> NDArray:
    return (255.0 - image) / 255.0


def resize(image: NDArray, height: int) -> NDArray:
    width = int(float(height * image.shape[1]) / image.shape[0])
    sample_img = cv2.resize(image, (width, height))
    return sample_img


def add_image_into_tr_omr_canvas(
    image: NDArray, margin_top: int = 0, margin_bottom: int = 0
) -> NDArray:
    tr_omr_max_height = 128
    tr_omr_max_width = 1280
    tr_omr_max_height_with_margin = tr_omr_max_height - margin_top - margin_bottom
    tr_omr_ratio = float(tr_omr_max_height_with_margin) / tr_omr_max_width
    height, width = image.shape[:2]

    # Calculate the new size such that it fits exactly into
    # the area of tr_omr_max_height and tr_omr_max_width
    # while maintaining the aspect ratio of height and width.

    if height / width > tr_omr_ratio:
        # The height is the limiting factor.
        new_shape = (
            int(width / height * tr_omr_max_height_with_margin),
            tr_omr_max_height_with_margin,
        )
    else:
        # The width is the limiting factor.
        new_shape = (tr_omr_max_width, int(height / width * tr_omr_max_width))

    resized = cv2.resize(image, new_shape)

    new_image = 255 * np.ones((tr_omr_max_height, tr_omr_max_width, 3), np.uint8)

    # Copy the resized image into the center of the new image.
    x_offset = 0
    y_offset = (tr_omr_max_height_with_margin - resized.shape[0]) // 2 + margin_top
    new_image[y_offset : y_offset + resized.shape[0], x_offset : x_offset + resized.shape[1]] = (
        resized
    )

    return new_image


_transform = alb.Compose(
    [
        alb.ToGray(always_apply=True),
        alb.Normalize((0.7931, 0.7931, 0.7931), (0.1738, 0.1738, 0.1738)),
        ToTensorV2(),
    ]
)


def readimg(config: Config, path: str) -> NDArray:
    img: NDArray = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if img.shape[-1] == 4:  # noqa: PLR2004
        img = 255 - img[:, :, 3]
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[-1] == 3:  # noqa: PLR2004
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif len(img.shape) == 2:  # noqa: PLR2004
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        raise RuntimeError("Unsupport image type!")

    h, w, c = img.shape
    size_h = config.max_height
    new_h = size_h
    new_w = int(size_h / h * w)
    new_w = new_w // config.patch_size * config.patch_size
    img = cv2.resize(img, (new_w, new_h))
    img = _transform(image=img)["image"][:1]
    return img
