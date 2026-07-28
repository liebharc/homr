import functools
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from homr.staff_dewarping import warp_image_randomly
from homr.type_definitions import NDArray
from training.transformer.augraphy_augment import apply_augraphy

_texture_config: dict = {"dir": None, "paths": None}
cv2.setNumThreads(0)


def set_texture_dir(path: str) -> None:
    """Configure a directory of paper texture images for background augmentation."""
    _texture_config["dir"] = path
    _texture_config["paths"] = None


def _get_texture_paths() -> list[str]:
    if _texture_config["paths"] is None:
        d = _texture_config["dir"]
        if d is None:
            _texture_config["paths"] = []
        else:
            p = Path(d)
            _texture_config["paths"] = (
                [str(f) for f in p.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
                if p.is_dir()
                else []
            )
    paths: list[str] = _texture_config["paths"]
    return paths


_font_cache: dict = {"paths": None}
_FONT_KEYWORDS: frozenset[str] = frozenset(
    {"times", "liberation", "freeserif", "palatino", "georgia", "garamond", "edwin"}
)


def _get_annotation_font_paths() -> list[str]:
    if _font_cache["paths"] is not None:
        paths: list[str] = _font_cache["paths"]
        return paths
    search_dirs = [
        Path("/usr/share/fonts"),
        Path("/usr/local/share/fonts"),
        Path.home() / ".fonts",
        Path.home() / ".local/share/fonts",
    ]
    found: list[str] = []
    for d in search_dirs:
        if not d.is_dir():
            continue
        for pattern in ("*.ttf", "*.otf"):
            for f in d.rglob(pattern):
                if any(kw in f.stem.lower() for kw in _FONT_KEYWORDS):
                    found.append(str(f))
    _font_cache["paths"] = found
    return found


def apply_text_injection(image: NDArray) -> NDArray:
    """Inject random alphanumeric fragments to simulate score annotations."""
    h, w = image.shape[:2]
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).convert("RGBA")
    font_paths = _get_annotation_font_paths()
    chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    n_fragments = int(np.random.randint(1, 4))
    for _ in range(n_fragments):
        if np.random.random() < 0.4:
            text = str(int(np.random.randint(0, 101)))
        else:
            length = int(np.random.randint(1, 5))
            indices = np.random.randint(0, len(chars), size=length)
            text = "".join(chars[int(i)] for i in indices)

        font_size = int(np.random.uniform(0.08, 0.14) * h)
        font_size = max(8, min(font_size, 36))

        font: ImageFont.FreeTypeFont | ImageFont.ImageFont
        if font_paths:
            try:
                font = ImageFont.truetype(str(np.random.choice(font_paths)), font_size)
            except Exception:
                font = ImageFont.load_default(size=font_size)
        else:
            font = ImageFont.load_default(size=font_size)

        bbox = font.getbbox(text)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = int(np.random.uniform(0, max(1, w - tw)))
        y = int(np.random.uniform(0, max(1, h - th)))

        gray_val = int(np.random.uniform(20, 80))
        opacity = int(np.random.uniform(0.6, 0.9) * 255)

        text_layer = Image.new("RGBA", pil_img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(text_layer)
        draw.text((x, y), text, fill=(gray_val, gray_val, gray_val, opacity), font=font)
        pil_img = Image.alpha_composite(pil_img, text_layer)

    result_bgr: NDArray = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    return result_bgr


def read_image_to_ndarray(path: str) -> NDArray:
    img: NDArray = cv2.imread(path, cv2.IMREAD_COLOR_BGR)  # type: ignore
    if img is None:
        raise ValueError("Failed to read image from " + path)
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


def pad_to_3_dims(tensor: torch.Tensor) -> torch.Tensor:
    if len(tensor.shape) == 2:
        tensor = tensor.unsqueeze(0)
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


def apply_clahe(image: NDArray, p: float = 0.1) -> NDArray:
    if np.random.random() > p:
        return image

    clip_limit = np.random.uniform(2.0, 3.0)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))

    if image.ndim == 2:
        return clahe.apply(image)
    else:
        result = np.zeros_like(image)
        for c in range(image.shape[2]):
            result[:, :, c] = clahe.apply(image[:, :, c])
        return result


def apply_ink_color_shift(image: NDArray) -> NDArray:
    """Lighten dark ink pixels by 0-40 luma units to simulate non-pure-black ink."""
    shift = int(np.random.randint(0, 41))
    if shift == 0:
        return image
    luma = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    dark = luma < 80
    result = image.copy()
    result[dark] = np.clip(result[dark].astype(np.int32) + shift, 0, 255).astype(np.uint8)
    return result


def apply_background_augmentation(image: NDArray) -> NDArray:
    """Paper texture blend, gradient shading, and per-channel color cast."""
    h, w = image.shape[:2]
    img_f = image.astype(np.float32)

    luma = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    ink_mask = (255.0 - luma.astype(np.float32)) / 255.0
    if image.ndim == 3:
        ink_mask = ink_mask[:, :, np.newaxis]

    # Paper texture blend (70% when available)
    textures = _get_texture_paths()
    if textures and np.random.random() < 0.70:
        tex_path = str(np.random.choice(textures))
        tex = cv2.imread(tex_path)
        if tex is not None:
            if tex.ndim == 2:
                tex = cv2.cvtColor(tex, cv2.COLOR_GRAY2BGR)
            th, tw = tex.shape[:2]
            tiled = np.tile(tex, (-(-h // th), -(-w // tw), 1))[:h, :w].astype(np.float32)
            img_f = img_f * ink_mask + tiled * (1.0 - ink_mask)

    # Gradient shading: linear + radial, ±10 and ±6 luma units
    xg, yg = _gradient_grids(h, w)
    gx = float(np.random.uniform(-10, 10))
    gy = float(np.random.uniform(-10, 10))
    radial = float(np.random.uniform(-6, 6)) * np.exp(-(xg**2 + yg**2) / 0.5)
    shading = (gx * xg + gy * yg + radial).astype(np.float32)
    img_f += shading[:, :, np.newaxis] if image.ndim == 3 else shading

    # Per-channel color cast (BGR order: B ±6, G ±3, R ±4)
    if image.ndim == 3:
        cast = np.array(
            [
                float(np.random.uniform(-6.0, 6.0)),
                float(np.random.uniform(-3.0, 3.0)),
                float(np.random.uniform(-4.0, 4.0)),
            ],
            dtype=np.float32,
        )
        img_f += cast[np.newaxis, np.newaxis, :]

    return np.clip(img_f, 0, 255).astype(np.uint8)


@functools.lru_cache(maxsize=16)
def _gradient_grids(h: int, w: int) -> tuple[NDArray, NDArray]:
    xs = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, h, dtype=np.float32)
    xg, yg = np.meshgrid(xs, ys)
    return xg, yg


_MIN_STAFF_CONTRAST = 60


def _enforce_min_staff_contrast(image: NDArray, ink_mask: NDArray) -> NDArray:
    """Darken ink pixels that became too bright relative to background after augmentation."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    bg_pixels = gray[~ink_mask] if (~ink_mask).any() else gray.ravel()
    bg_level = int(np.percentile(bg_pixels, 75))
    ceiling = bg_level - _MIN_STAFF_CONTRAST
    if ceiling <= 0:
        return image
    too_light = ink_mask & (gray > ceiling)
    if not too_light.any():
        return image
    result = image.copy()
    result[too_light] = np.clip(result[too_light].astype(np.int32), 0, ceiling).astype(np.uint8)
    return result


def _make_geometric_transforms(allow_occlusions: bool) -> A.Compose:
    transforms_list: list[A.BasicTransform] = [
        A.Rotate(limit=2, border_mode=cv2.BORDER_CONSTANT, fill=255, p=0.5),
        A.Perspective(scale=(0.005, 0.015), fill=255, p=0.3),
        A.OneOf(
            [
                A.GaussianBlur(blur_limit=(3, 5), sigma_limit=(0.1, 0.5), p=1.0),
                A.Sharpen(alpha=(0.0, 0.1), lightness=(0.9, 1.0), p=1.0),
            ],
            p=0.3,
        ),
    ]
    if allow_occlusions:
        transforms_list.append(
            A.CoarseDropout(
                num_holes_range=(2, 8),
                hole_height_range=(10, 30),
                hole_width_range=(20, 60),
                fill=255,
                p=0.4,
            ),
        )
    return A.Compose(transforms_list)


_geometric_transforms: dict[bool, A.Compose] = {}


def distort_image(image: NDArray, allow_occlusions: bool = False) -> NDArray:
    # Staff dewarping
    image = np.array(warp_image_randomly(Image.fromarray(image)))

    # Geometric
    if allow_occlusions not in _geometric_transforms:
        _geometric_transforms[allow_occlusions] = _make_geometric_transforms(allow_occlusions)
    image = _geometric_transforms[allow_occlusions](image=image)["image"]

    # Capture ink locations after geometry, before brightness augmentation
    _pre_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    ink_mask = _pre_gray < 128

    # Ink and background
    image = apply_ink_color_shift(image)
    image = apply_background_augmentation(image)

    # Annotation injection
    if np.random.random() < 0.35:
        image = apply_text_injection(image)

    # Document degradation
    image = apply_augraphy(image)

    # Local contrast
    image = apply_clahe(image, p=0.1)

    # Ensure staff lines remain visible despite brightness/degradation augmentation
    image = _enforce_min_staff_contrast(image, ink_mask)

    return image
