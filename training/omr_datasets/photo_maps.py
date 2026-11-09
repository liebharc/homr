"""
Where each pixel of a phone photo of a printed template page lies on the flat page.

Template pages carry eight ArUco marks; page k carries the ids 8k..8k+7. UVDoc guesses the
bending of the page from the photo alone, and the marks correct that guess. The photo's printed
music is then painted over, so other music can be drawn onto the same paper (photo_paste.py).
"""

import importlib.util
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import griddata

from homr.simple_logging import eprint

ARUCO = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
DETECTOR = cv2.aruco.ArucoDetector(ARUCO)
FW, FH = 2480, 3508  # the flat template page: A4 at 300 dpi
FLAT_MM = 300 / 25.4  # flat page pixels per mm
PHOTO_SIZE = 2000  # long side every photo is brought to
# With fewer marks they all sat on one side of the page, and the map was guessed for the rest:
# the new music landed off the paper's bends and the old print showed through
MIN_MARKS = 4
UVDOC_INPUT = (488, 712)


class PageMapper(Protocol):
    def page_maps(self, photo: NDArray) -> tuple[NDArray, NDArray]: ...


class Uvdoc:
    """UVDoc (github.com/tanguymagne/UVDoc, MIT): model.py and model/best_model.pkl in uvdoc_dir."""

    def __init__(self, uvdoc_dir: Path) -> None:
        import torch  # noqa: PLC0415  (only training needs torch)

        spec = importlib.util.spec_from_file_location("uvdoc_model", uvdoc_dir / "model.py")
        if spec is None or spec.loader is None:
            raise FileNotFoundError(uvdoc_dir / "model.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.model = module.UVDocnet(num_filter=32, kernel_size=5)
        weights = torch.load(
            uvdoc_dir / "model" / "best_model.pkl", map_location="cpu", weights_only=True
        )
        self.model.load_state_dict(weights["model_state"])
        self.model.eval()

    def page_maps(self, photo: NDArray) -> tuple[NDArray, NDArray]:
        """UVDoc's guess of where each photo pixel lies on the flat page (NaN off the page)."""
        import torch  # noqa: PLC0415

        ph, pw = photo.shape[:2]
        rgb = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        with torch.no_grad():
            image = torch.from_numpy(cv2.resize(rgb, UVDOC_INPUT).transpose(2, 0, 1)).unsqueeze(0)
            grid, _ = self.model(image)
        grid = grid[0].numpy()
        rows, cols = grid.shape[1:]
        points = np.stack(
            [((grid[0] + 1) / 2 * (pw - 1)).ravel(), ((grid[1] + 1) / 2 * (ph - 1)).ravel()], axis=1
        )
        u, v = np.meshgrid(np.linspace(0, 1, cols), np.linspace(0, 1, rows))
        qx, qy = np.meshgrid(np.arange(pw), np.arange(ph))
        map_u = griddata(points, u.ravel(), (qx, qy), method="linear")
        map_v = griddata(points, v.ravel(), (qx, qy), method="linear")
        return (map_u * (FW - 1)).astype(np.float32), (map_v * (FH - 1)).astype(np.float32)


@dataclass
class PhotoConditions:
    """What a photo does to a page: the upright photo, its map to the flat page, the paper with
    the printed music painted over, how dark its ink is and how sharp the ink edges are."""

    photo: NDArray
    map_x: NDArray
    map_y: NDArray
    on_page: NDArray
    paper: NDArray
    ink_level: float
    sharp: float


def find_marks(photo: NDArray) -> dict[int, NDArray]:
    """{mark id: its 4 corners}: top left, top right, bottom right, bottom left of the mark."""
    gray = photo if photo.ndim == 2 else cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = DETECTOR.detectMarkers(np.ascontiguousarray(gray))
    if ids is None:
        return {}
    return {int(i): c[0] for i, c in zip(ids.ravel(), corners, strict=True)}


def page_number(ids: Iterable[int]) -> int | None:
    """The template most of the marks belong to, or None without marks."""
    pages = Counter(i // 8 for i in ids)
    return pages.most_common(1)[0][0] if pages else None


def upright(photo: NDArray, marks: dict[int, NDArray]) -> NDArray:
    """The photo turned by quarter turns so the page's top is up. UVDoc takes the photo's top as
    the page's top. A mark's first edge runs along its top, from left to right."""
    votes = Counter(  # one vote per mark: the angles wrap round at 180 degrees
        round(float(np.degrees(np.arctan2(c[1][1] - c[0][1], c[1][0] - c[0][0]))) / 90) % 4
        for c in marks.values()
    )
    return np.ascontiguousarray(np.rot90(photo, votes.most_common(1)[0][0]))


def mark_fit(guess: NDArray, truth: NDArray) -> NDArray:
    """The 3x3 transform moving guessed flat positions of mark corners onto their true positions:
    shift, turn and scale for one or two marks, affine for three, a homography for four or more."""
    guess, truth = np.asarray(guess, np.float32), np.asarray(truth, np.float32)
    if len(guess) >= 16:  # three marks often sit in a row, where a homography swings wildly
        return cv2.findHomography(guess, truth, 0)[0]
    if len(guess) >= 12:
        fit = cv2.estimateAffine2D(guess, truth, method=cv2.LMEDS)[0]
    else:  # two marks can't hold a shear
        fit = cv2.estimateAffinePartial2D(guess, truth, method=cv2.LMEDS)[0]
    return np.vstack([fit, [0, 0, 1]])


def apply_fit(m: NDArray, x: NDArray, y: NDArray) -> tuple[NDArray, NDArray]:
    w = m[2, 0] * x + m[2, 1] * y + m[2, 2]
    return (m[0, 0] * x + m[0, 1] * y + m[0, 2]) / w, (m[1, 0] * x + m[1, 1] * y + m[1, 2]) / w


def extended(map_x: NDArray, map_y: NDArray) -> tuple[NDArray, NDArray]:
    """The maps carried past the page UVDoc saw, by a homography fitted to its map. In a close-up
    UVDoc's page stops short of the frame, so marks near the edge fell off it."""
    gone = np.isnan(map_x)
    if not gone.any() or gone.all():
        return map_x, map_y
    ys, xs = np.nonzero(~gone[::20, ::20])
    src = np.stack([xs * 20, ys * 20], axis=1).astype(np.float32)
    dst = np.stack([map_x[ys * 20, xs * 20], map_y[ys * 20, xs * 20]], axis=1)
    m = cv2.findHomography(src, dst, 0)[0]
    qy, qx = np.nonzero(gone)
    fx, fy = apply_fit(m, qx.astype(np.float32), qy.astype(np.float32))
    map_x, map_y = map_x.copy(), map_y.copy()
    map_x[qy, qx], map_y[qy, qx] = fx, fy
    return map_x, map_y


def at(maps: tuple[NDArray, NDArray], points: NDArray) -> NDArray:
    """The maps' values at photo points (x, y)."""
    map_x, map_y = maps
    xi = np.clip(np.round(points[:, 0]).astype(int), 0, map_x.shape[1] - 1)
    yi = np.clip(np.round(points[:, 1]).astype(int), 0, map_x.shape[0] - 1)
    return np.stack([map_x[yi, xi], map_y[yi, xi]], axis=1)


def edge_sharpness(n: NDArray, page: NDArray) -> float:
    """Steepness of the ink edges in an image divided by its paper (1 = paper): 90th percentile
    of the gradient."""
    any_ink = (n < 0.82) & page
    edges = cv2.dilate(any_ink.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    edges &= page & (n >= 0.6)
    if not edges.any():
        return 0.0
    gradient = np.hypot(cv2.Sobel(n, cv2.CV_32F, 1, 0), cv2.Sobel(n, cv2.CV_32F, 0, 1))
    return float(np.percentile(gradient[edges], 90))


def erased(
    photo: NDArray,
    gray: NDArray,
    around: NDArray,
    *,
    inside: NDArray,
    where: NDArray,
    solid: NDArray,
    marks: NDArray,
) -> NDArray:
    """The photo with its music painted over by the paper around it. The ink is taken generously,
    its soft grey edges too, and every hole is filled with the paper level of the blank pixels
    near it plus fresh grain as fine and as strong as the photo's.
    where: the pixels the old print can be on. solid: pixels painted over whatever they show.
    inside: the pixels surely on the paper, the only ones the paper level is taken from.
    marks: the corners of the marks found in the photo, painted over with a margin."""
    ink = (((gray < 0.96 * around) & where) | solid).astype(np.uint8)
    for quad in marks.reshape(-1, 4, 2):  # a mark's grey halo darkened the paper put there
        middle = quad.mean(axis=0)
        cv2.fillConvexPoly(ink, np.round(middle + 1.5 * (quad - middle)).astype(np.int32), 1)
    ink = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    f = photo.astype(np.float32)
    clear = ((ink == 0) & inside).astype(np.float32)[..., None]
    level: NDArray | None = None
    for sigma in (120, 40, 12):  # a wide hole takes its paper from further away
        weight = cv2.GaussianBlur(clear, (0, 0), sigma)[..., None]
        guess = cv2.GaussianBlur(f * clear, (0, 0), sigma) / np.maximum(weight, 1e-6)
        near = np.clip((weight - 0.1) / 0.2, 0, 1)  # blended, so no edge where one reach takes over
        level = guess if level is None else near * guess + (1 - near) * level
    assert level is not None  # noqa: S101
    fine = f - cv2.GaussianBlur(f, (0, 0), 2)
    grain = np.random.default_rng(0).normal(0, 1, f.shape).astype(np.float32)
    grain -= cv2.GaussianBlur(grain, (0, 0), 2)
    grain *= fine[(ink == 0) & inside].std(axis=0) / grain.reshape(-1, 3).std(axis=0)
    w = cv2.GaussianBlur(ink.astype(np.float32), (0, 0), 1.5)[..., None]  # no seam at the edge
    return f * (1 - w) + (level + grain) * w


def old_print(template: dict[str, Any]) -> list[tuple[float, float, float, float]]:
    """Where a template's print lies on the flat page: the music with 8 mm to spare (the map is
    off by about 4 mm), and the small label at the bottom with 3 mm."""
    mm = FLAT_MM
    x, y, w, h = template["music_box"]
    label_x, label_y = FW / 2 + 12 * mm, FH - 7 * mm
    return [
        (x - 8 * mm, y - 8 * mm, x + w + 8 * mm, y + h + 8 * mm),
        (label_x - 3 * mm, label_y - 6 * mm, FW - 16 * mm, label_y + 4 * mm),
    ]


def mark_boxes(template: dict[str, Any]) -> list[tuple[float, float, float, float]]:
    """Every mark of a template on the flat page, at 1.6 times its size."""
    boxes = []
    for corners in template["marks"]["corners"]:
        c = np.float32(corners)
        middle, half = c.mean(axis=0), 0.8 * (c.max(axis=0) - c.min(axis=0))
        boxes.append((*(middle - half), *(middle + half)))
    return boxes


def conditions(
    photo: NDArray,
    raw: tuple[NDArray, NDArray],
    known: tuple[NDArray, NDArray],
    template: dict[str, Any],
) -> PhotoConditions | None:
    """The photo's conditions from UVDoc's maps (raw) and the marks found in it (known: photo
    points, flat points). None when fewer than 4 mark corners land on UVDoc's page."""
    map_x, map_y = extended(*raw)
    guess = at((map_x, map_y), known[0])
    ok = ~np.isnan(guess).any(axis=1)
    if ok.sum() < 4:
        return None
    map_x, map_y = apply_fit(mark_fit(guess[ok], known[1][ok]), map_x, map_y)
    with np.errstate(invalid="ignore"):
        on_page = ~np.isnan(map_x) & (map_x >= 0) & (map_x <= FW - 1)
        on_page &= (map_y >= 0) & (map_y <= FH - 1)
    map_x = np.where(on_page, map_x, -1).astype(np.float32)
    map_y = np.where(on_page, map_y, -1).astype(np.float32)
    gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    around = cv2.medianBlur(closed, 21)
    where = np.zeros_like(on_page)
    for left, top, right, bottom in old_print(template):
        where |= on_page & (map_x >= left) & (map_x <= right) & (map_y >= top) & (map_y <= bottom)
    solid = np.zeros_like(on_page)
    for left, top, right, bottom in mark_boxes(template):
        solid |= on_page & (map_x >= left) & (map_x <= right) & (map_y >= top) & (map_y <= bottom)
    edge = 6 * FLAT_MM  # 6 mm: further in than the map is off
    inside = on_page & (map_x >= edge) & (map_x <= FW - 1 - edge)
    inside &= (map_y >= edge) & (map_y <= FH - 1 - edge)
    paper = erased(photo, gray, around, inside=inside, where=where, solid=solid, marks=known[0])
    n = gray.astype(np.float32) / np.maximum(around, 1)
    ink_level = float(np.median(n[(n < 0.6) & on_page]))  # the ink's core, as a share of the paper
    sharp = edge_sharpness(n, on_page)
    return PhotoConditions(photo, map_x, map_y, on_page, paper, ink_level, sharp)


def prepare_photo(
    path: Path, template: dict[str, Any], uvdoc: PageMapper | None
) -> PhotoConditions | None:
    """The conditions of a photo of `template`, upright at PHOTO_SIZE. None, with the reason
    printed, when fewer than MIN_MARKS of the template's marks are found, or too few land on the
    page UVDoc sees."""
    photo = cv2.imread(str(path))
    if photo is None:
        eprint(f"{path.name}: not an image, skipped")
        return None
    scale = PHOTO_SIZE / max(photo.shape[:2])
    photo = cv2.resize(photo, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    number = int(template["name"][1:])
    found = {i: c for i, c in find_marks(photo).items() if i // 8 == number}
    if not found:
        eprint(f"{path.name}: no marks of {template['name']} found, skipped")
        return None
    photo = upright(photo, found)
    found = {i: c for i, c in find_marks(photo).items() if i // 8 == number}
    if len(found) < MIN_MARKS:
        eprint(f"{path.name}: {len(found)} marks found, {MIN_MARKS} needed for a map, skipped")
        return None
    truth = dict(zip(template["marks"]["ids"], template["marks"]["corners"], strict=True))
    if uvdoc is None:
        eprint(f"{path.name}: no page map, skipped")
        return None
    known = (
        np.concatenate(list(found.values())),
        np.concatenate([np.float32(truth[i]) for i in found]),
    )
    result = conditions(photo, uvdoc.page_maps(photo), known, template)
    if result is None:
        eprint(f"{path.name}: fewer than 4 mark corners on the page UVDoc sees, skipped")
    return result
