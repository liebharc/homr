"""
Degradations which make clean pages look like phone photos. Each is deterministic for a
given random generator and keeps every staff readable, so the ground truth stays valid.
"""

import os
import random
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from homr.type_definitions import NDArray

Degradation = Callable[[NDArray, np.random.Generator], NDArray]

# Used instead of the generated surroundings if set
_background_photos: list[Path] = []


def set_background_photos(directory: Path | None) -> None:
    """Must be called before the worker processes are forked."""
    _background_photos.clear()
    if directory is not None:
        extensions = {".jpg", ".jpeg", ".png"}
        _background_photos.extend(
            sorted(p for p in directory.iterdir() if p.suffix.lower() in extensions)
        )


def _to_uint8(image: NDArray) -> NDArray:
    return np.clip(image, 0, 255).astype(np.uint8)


def _sigmoid_mask(shape: tuple[int, int], sharpness: float, rng: np.random.Generator) -> NDArray:
    """A soft edge at a random position and angle, 0 on one side and 1 on the other."""
    height, width = shape
    angle = rng.uniform(0, 2 * np.pi)
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    distance = (x - width / 2) * np.cos(angle) + (y - height / 2) * np.sin(angle)
    offset = rng.uniform(-0.25, 0.25) * max(width, height)
    return 1 / (1 + np.exp(-(distance - offset) / (sharpness * max(width, height))))


def _content_box(image: NDArray) -> tuple[int, int, int, int]:
    """Bounding box (x0, y0, x1, y1) of the ink on the page."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(gray < 128)
    if len(xs) == 0:
        return 0, 0, image.shape[1], image.shape[0]
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _with_augraphy(augmentation: Any, image: NDArray, rng: np.random.Generator) -> NDArray:
    # Augraphy uses the global random generators and writes to ./augraphy_cache
    seed = int(rng.integers(0, 2**31))
    random.seed(seed)
    np.random.seed(seed)
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            result = augmentation(image.copy())
        finally:
            os.chdir(cwd)
    if result is None:
        return image
    if result.ndim == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    return _to_uint8(result[..., :3])


# Light and exposure


def gray_background(image: NDArray, rng: np.random.Generator) -> NDArray:
    """Underexposed photo: the paper is gray, not white."""
    paper, ink = rng.uniform(135, 155), rng.uniform(25, 40)
    return _to_uint8(ink + image.astype(np.float32) * (paper - ink) / 255)


def uneven_light(image: NDArray, rng: np.random.Generator) -> NDArray:
    """One side of the sheet is much darker than the other."""
    mask = _sigmoid_mask(image.shape[:2], sharpness=0.25, rng=rng)
    darkest = rng.uniform(0.35, 0.5)
    light = darkest + (1 - darkest) * mask
    return _to_uint8(image.astype(np.float32) * light[..., np.newaxis])


def hard_shadow(image: NDArray, rng: np.random.Generator) -> NDArray:
    """The shadow of the phone or a hand across a part of the sheet."""
    mask = _sigmoid_mask(image.shape[:2], sharpness=0.01, rng=rng)
    strength = rng.uniform(0.45, 0.6)
    light = 1 - strength * (1 - mask)
    return _to_uint8(image.astype(np.float32) * light[..., np.newaxis])


def glare(image: NDArray, rng: np.random.Generator) -> NDArray:
    """A lamp reflected on glossy paper, washing out the ink."""
    height, width = image.shape[:2]
    cx, cy = rng.uniform(0.2, 0.8) * width, rng.uniform(0.2, 0.8) * height
    sigma = rng.uniform(0.08, 0.14) * width
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    spot = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2)) * rng.uniform(0.7, 0.85)
    result = image.astype(np.float32)
    return _to_uint8(result + (255 - result) * spot[..., np.newaxis])


def color_cast(image: NDArray, rng: np.random.Generator) -> NDArray:
    """Warm light from a light bulb or cold light from a window."""
    warm = np.array([0.62, 0.85, 1.0], dtype=np.float32)  # BGR
    cold = np.array([1.0, 0.93, 0.8], dtype=np.float32)
    tint = warm if rng.random() < 0.5 else cold
    return _to_uint8(image.astype(np.float32) * tint * rng.uniform(0.85, 0.95))


def _dim(image: NDArray, rng: np.random.Generator, paper: float, noise: float) -> NDArray:
    ink = rng.uniform(20, 35)
    result = ink + image.astype(np.float32) * (paper - ink) / 255
    luminance_noise = np.asarray(rng.normal(0, noise, image.shape[:2]), dtype=np.float32)
    color_noise = np.asarray(rng.normal(0, noise / 2, image.shape), dtype=np.float32)
    return _to_uint8(result + luminance_noise[..., np.newaxis] + color_noise)


def low_light(image: NDArray, rng: np.random.Generator) -> NDArray:
    """A photo in a dim room: dark, low contrast and noisy."""
    return _dim(image, rng, paper=rng.uniform(95, 115), noise=9)


def slightly_dim(image: NDArray, rng: np.random.Generator) -> NDArray:
    return _dim(image, rng, paper=rng.uniform(130, 150), noise=5)


# Ink and paper


def faded_ink(image: NDArray, rng: np.random.Generator) -> NDArray:
    remaining_ink = rng.uniform(0.4, 0.5)
    return _to_uint8(255 - (255 - image.astype(np.float32)) * remaining_ink)


def yellowed_paper(image: NDArray, rng: np.random.Generator) -> NDArray:
    result = image.astype(np.float32) * np.array([0.72, 0.88, 0.97], dtype=np.float32)
    texture = cv2.resize(
        rng.normal(0, 1, (40, 40)).astype(np.float32),
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_CUBIC,
    )
    return _to_uint8(result * (1 + 0.05 * texture[..., np.newaxis]))


def bleed_through(image: NDArray, rng: np.random.Generator) -> NDArray:
    """The print of the back of the page shines through."""
    import augraphy  # noqa: PLC0415

    return _with_augraphy(
        augraphy.BleedThrough(intensity_range=(0.3, 0.5), alpha=0.3, p=1), image, rng
    )


def pencil_marks(image: NDArray, rng: np.random.Generator) -> NDArray:
    """Pencil fingerings, circles and lines."""
    result = image.copy()
    height, width = image.shape[:2]
    for _ in range(int(rng.integers(15, 30))):
        color = (int(rng.integers(90, 140)),) * 3
        thickness = max(1, round(width / 900))
        x, y = int(rng.uniform(0.05, 0.95) * width), int(rng.uniform(0.05, 0.95) * height)
        size = int(rng.uniform(0.01, 0.03) * width)
        kind = rng.integers(0, 3)
        if kind == 0:
            cv2.ellipse(result, (x, y), (size, int(size * 0.7)), 0, 0, 360, color, thickness)
        elif kind == 1:
            points = np.cumsum(rng.normal(0, size / 3, (6, 2)), axis=0) + (x, y)
            cv2.polylines(result, [points.astype(np.int32)], False, color, thickness)
        else:
            cv2.putText(
                result,
                str(rng.integers(1, 6)),
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                size / 40,
                color,
                thickness,
            )
    return result


def stains(image: NDArray, rng: np.random.Generator) -> NDArray:
    """A few brownish blots, e.g. from coffee or water."""
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.float32)
    for _ in range(int(rng.integers(3, 7))):
        center = (int(rng.uniform(0, width)), int(rng.uniform(0, height)))
        axes = (int(rng.uniform(0.03, 0.1) * width), int(rng.uniform(0.03, 0.1) * width))
        cv2.ellipse(mask, center, axes, rng.uniform(0, 180), 0, 360, rng.uniform(0.3, 0.6), -1)
    mask = cv2.GaussianBlur(mask, (0, 0), 0.01 * width)
    stain_color = np.array([0.55, 0.7, 0.85], dtype=np.float32)  # BGR
    factor = 1 - mask[..., np.newaxis] * (1 - stain_color)
    return _to_uint8(image.astype(np.float32) * factor)


# Geometry


def _random_background(shape: tuple[int, int], rng: np.random.Generator) -> NDArray:
    height, width = shape
    if _background_photos:
        photo = cv2.imread(str(_background_photos[rng.integers(0, len(_background_photos))]))
        if photo is not None:
            return cv2.resize(photo, (width, height), interpolation=cv2.INTER_AREA)
    return desk_texture(shape, rng)


def _warp(image: NDArray, warp: Callable[[NDArray], NDArray], rng: np.random.Generator) -> NDArray:
    """Warps the sheet and fills the uncovered area with surroundings."""
    background = _random_background(image.shape[:2], rng)
    mask = warp(np.full(image.shape[:2], 255, np.uint8))
    return np.where(mask[..., np.newaxis] > 0, warp(image), background)


def perspective(image: NDArray, rng: np.random.Generator) -> NDArray:
    """Photo taken at an angle."""
    height, width = image.shape[:2]
    corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    # Shrink the sheet first so that the corners stay in the image
    target = corners * 0.8 + np.array([width, height], dtype=np.float32) * 0.1
    shift = rng.uniform(-0.09, 0.09, (4, 2)).astype(np.float32) * (width, height)
    matrix = cv2.getPerspectiveTransform(corners, (target + shift).astype(np.float32))
    size = (width, height)
    return _warp(image, lambda img: cv2.warpPerspective(img, matrix, size), rng)


def rotation(image: NDArray, rng: np.random.Generator) -> NDArray:
    height, width = image.shape[:2]
    angle = rng.uniform(4, 8) * rng.choice([-1, 1])
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 0.85)
    return _warp(image, lambda img: cv2.warpAffine(img, matrix, (width, height)), rng)


def page_curl(image: NDArray, rng: np.random.Generator) -> NDArray:
    """A book page bending towards the spine: lines converge and darken near it."""
    height, width = image.shape[:2]
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    distance = x / width if rng.random() < 0.5 else 1 - x / width
    curl_width = rng.uniform(0.3, 0.45)
    curl = np.clip(1 - distance / curl_width, 0, 1) ** 2
    strength = rng.uniform(0.12, 0.2)
    map_y = height / 2 + (y - height / 2) * (1 + strength * curl)
    curled = cv2.remap(image, x, map_y, cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    shading = 1 - 0.35 * curl
    curled = _to_uint8(curled.astype(np.float32) * shading[..., np.newaxis])
    page = np.full((height, width), 255, np.uint8)
    mask = cv2.remap(page, x, map_y, cv2.INTER_NEAREST, borderValue=0)
    background = _random_background((height, width), rng)
    return np.where(mask[..., np.newaxis] > 0, curled, background)


def wrinkles(image: NDArray, rng: np.random.Generator) -> NDArray:
    """A crumpled sheet: small displacements and dark creases."""
    height, width = image.shape[:2]

    def smooth_noise(cells: int) -> NDArray:
        noise = rng.normal(0, 1, (cells, cells)).astype(np.float32)
        return cv2.resize(noise, (width, height), interpolation=cv2.INTER_CUBIC)

    amplitude = 0.004 * width
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    map_x = x + amplitude * smooth_noise(12)
    map_y = y + amplitude * smooth_noise(12)
    warped = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    # Creases are thin, dark lines where a smooth noise crosses zero
    creases = 1 - 0.2 * np.exp(-((smooth_noise(8) / 0.025) ** 2))
    return _to_uint8(warped.astype(np.float32) * creases[..., np.newaxis])


# Optics


def defocus(image: NDArray, rng: np.random.Generator) -> NDArray:
    sigma = rng.uniform(0.0012, 0.0016) * image.shape[1]
    return cv2.GaussianBlur(image, (0, 0), sigma)


def _motion_blur(image: NDArray, rng: np.random.Generator, length: int) -> NDArray:
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1
    angle = rng.uniform(0, 180)
    rotation_matrix = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle, 1)
    rotated = cv2.warpAffine(kernel, rotation_matrix, (length, length))
    return cv2.filter2D(image, -1, rotated / rotated.sum())


def motion_blur(image: NDArray, rng: np.random.Generator) -> NDArray:
    """A shaking hand while the photo is taken."""
    return _motion_blur(image, rng, max(3, int(rng.uniform(0.006, 0.009) * image.shape[1])))


def slight_motion_blur(image: NDArray, rng: np.random.Generator) -> NDArray:
    return _motion_blur(image, rng, max(3, int(rng.uniform(0.002, 0.003) * image.shape[1])))


def low_resolution(image: NDArray, rng: np.random.Generator) -> NDArray:
    """The sheet only covers a small part of the photo."""
    width = int(rng.uniform(650, 800))
    if image.shape[1] <= width:
        return image
    height = round(image.shape[0] * width / image.shape[1])
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def _jpeg(image: NDArray, quality: int) -> NDArray:
    _, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("Failed to decode JPEG")
    return decoded


def jpeg(image: NDArray, rng: np.random.Generator) -> NDArray:
    """Strong compression, e.g. after sending the photo with a messenger."""
    return _jpeg(image, int(rng.integers(15, 25)))


def phone_jpeg(image: NDArray, rng: np.random.Generator) -> NDArray:
    return _jpeg(image, int(rng.integers(70, 85)))


# Surroundings


def desk_texture(shape: tuple[int, int], rng: np.random.Generator) -> NDArray:
    """A wooden desk: brown with a grain along one axis."""
    height, width = shape
    grain = cv2.resize(
        rng.normal(0, 1, (max(2, height // 6), 6)).astype(np.float32),
        (width, height),
        interpolation=cv2.INTER_CUBIC,
    )
    base = np.array([rng.uniform(40, 70), rng.uniform(80, 110), rng.uniform(120, 160)])
    desk = base * (1 + 0.15 * grain[..., np.newaxis])
    return _to_uint8(desk)


def room_texture(shape: tuple[int, int], rng: np.random.Generator) -> NDArray:
    """A blurry room with furniture behind a music stand."""
    height, width = shape
    room = np.full((height, width, 3), rng.uniform(40, 80), dtype=np.float32)
    for _ in range(int(rng.integers(6, 12))):
        x0, y0 = int(rng.uniform(0, width)), int(rng.uniform(0, height))
        x1, y1 = x0 + int(rng.uniform(0.1, 0.5) * width), y0 + int(rng.uniform(0.1, 0.5) * height)
        color = rng.uniform(20, 230, 3).tolist()
        cv2.rectangle(room, (x0, y0), (x1, y1), color, -1)
    return _to_uint8(cv2.GaussianBlur(room, (0, 0), 0.01 * width))


def _place_on(
    image: NDArray, background: NDArray, rng: np.random.Generator
) -> tuple[NDArray, tuple[int, int]]:
    """Places the sheet at a random position on a background, returns the top left corner."""
    height, width = image.shape[:2]
    canvas_height, canvas_width = background.shape[:2]
    x = int(rng.uniform(0, canvas_width - width))
    y = int(rng.uniform(0, canvas_height - height))
    canvas = background.copy()
    canvas[y : y + height, x : x + width] = image
    return canvas, (x, y)


def _canvas_shape(image: NDArray, page_share: float) -> tuple[int, int]:
    height, width = image.shape[:2]
    return int(height / page_share), int(width / page_share)


def on_desk(image: NDArray, rng: np.random.Generator) -> NDArray:
    shape = _canvas_shape(image, rng.uniform(0.75, 0.85))
    background = _random_background(shape, rng)
    return _place_on(image, background, rng)[0]


def on_music_stand(image: NDArray, rng: np.random.Generator) -> NDArray:
    shape = _canvas_shape(image, rng.uniform(0.7, 0.8))
    room = room_texture(shape, rng)
    canvas, (x, y) = _place_on(image, room, rng)
    # The ledge of the stand below the sheet
    ledge_top = min(shape[0] - 1, y + image.shape[0])
    ledge_height = int(0.03 * shape[0])
    cv2.rectangle(canvas, (0, ledge_top), (shape[1], ledge_top + ledge_height), (25, 25, 25), -1)
    return canvas


def book_spread(image: NDArray, rng: np.random.Generator) -> NDArray:
    """A part of the opposite page of a book is visible, with its own staffs."""
    height, width = image.shape[:2]
    visible = int(rng.uniform(0.2, 0.35) * width)
    opposite = image[:, width - visible :] if rng.random() < 0.5 else image[:, :visible]
    # The opposite page has its own layout: its staffs don't line up with this page
    shift = int(rng.uniform(0.03, 0.08) * height)
    opposite = np.vstack([np.full((shift, visible, 3), 255, np.uint8), opposite[: height - shift]])
    gutter = np.linspace(0.55, 1, int(0.04 * width), dtype=np.float32)
    opposite_on_the_left = rng.random() < 0.5
    if opposite_on_the_left:
        spread, seam = np.hstack([opposite, image]), visible
    else:
        spread, seam = np.hstack([image, opposite]), width
    shading = np.ones(spread.shape[1], dtype=np.float32)
    shading[seam - len(gutter) : seam] = gutter[::-1]
    shading[seam : seam + len(gutter)] = gutter
    shaded = spread.astype(np.float32) * shading[np.newaxis, :, np.newaxis]
    return on_desk(_to_uint8(shaded), rng)


def objects_around(image: NDArray, rng: np.random.Generator) -> NDArray:
    """Blobs in skin, pencil and fur colors beside the music."""
    shape = _canvas_shape(image, rng.uniform(0.7, 0.8))
    canvas, (x, y) = _place_on(image, _random_background(shape, rng), rng)
    x0, y0, x1, y1 = _content_box(image)
    music = (x + x0, y + y0, x + x1, y + y1)
    height, width = shape
    objects = [
        ((70, 100, 160), (0.05, 0.08)),  # Hand
        ((30, 180, 220), (0.012, 0.15)),  # Pencil
        ((40, 45, 50), (0.15, 0.12)),  # Cat
    ]
    for color, (rx, ry) in objects:
        for _ in range(20):
            cx, cy = int(rng.uniform(0, width)), int(rng.uniform(0, height))
            ax, ay = int(rx * width), int(ry * width)
            radius = max(ax, ay)
            if (
                cx + radius < music[0]
                or cx - radius > music[2]
                or cy + radius < music[1]
                or cy - radius > music[3]
            ):
                cv2.ellipse(canvas, (cx, cy), (ax, ay), rng.uniform(0, 180), 0, 360, color, -1)
                break
    return canvas


def compose(*degradations: Degradation) -> Degradation:
    def composed(image: NDArray, rng: np.random.Generator) -> NDArray:
        for degradation in degradations:
            image = degradation(image, rng)
        return image

    return composed


GROUPS: dict[str, dict[str, Degradation]] = {
    "clean": {"clean": lambda image, rng: image},
    "light": {
        "gray_background": gray_background,
        "uneven_light": uneven_light,
        "hard_shadow": hard_shadow,
        "glare": glare,
        "color_cast": color_cast,
        "low_light": low_light,
    },
    "paper": {
        "faded_ink": faded_ink,
        "yellowed_paper": yellowed_paper,
        "bleed_through": bleed_through,
        "pencil_marks": pencil_marks,
        "stains": stains,
    },
    "geometry": {
        "perspective": perspective,
        "rotation": rotation,
        "page_curl": page_curl,
        "wrinkles": wrinkles,
    },
    "optics": {
        "defocus": defocus,
        "motion_blur": motion_blur,
        "low_resolution": low_resolution,
        "jpeg": jpeg,
    },
    "surroundings": {
        "on_desk": on_desk,
        "on_music_stand": on_music_stand,
        "book_spread": book_spread,
        "objects_around": objects_around,
    },
    # Realistic combinations
    "photos": {
        # perspective already puts the sheet on a desk, light affects both
        "phone_on_desk": compose(perspective, color_cast, uneven_light, defocus, phone_jpeg),
        "phone_music_stand_dim": compose(
            on_music_stand, rotation, slightly_dim, slight_motion_blur, phone_jpeg
        ),
        "old_print_photo": compose(
            yellowed_paper,
            faded_ink,
            bleed_through,
            stains,
            uneven_light,
            low_resolution,
            phone_jpeg,
        ),
    },
}

VARIANTS: dict[str, Degradation] = {
    name: degradation for group in GROUPS.values() for name, degradation in group.items()
}
