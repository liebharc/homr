"""
Training staves from phone photos of printed template pages
(github.com/slobaspeed/homr-photo-pages): the printed music is erased and Lieder songs are
pasted onto the same real paper, then cut into staves the way homr reads them, each with its
exact answer.

Downloads the photos and UVDoc (github.com/tanguymagne/UVDoc, MIT), and uses the Lieder songs
rendered by convert_lieder.py. Only photos of pages T23-T42 are used: T01-T22 carry the music of
homr's photo test set, and photos of T43-T50 are a test set.

For development, three environment variables point at local copies instead:
PHOTO_DATASET_DIR (the photo repo), UVDOC_DIR (UVDoc) and PHOTO_SONGS_DIR (MusicXML or .mxl songs).
"""

import json
import os
import random
import re
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from homr.download_utils import download_file, unzip_file
from homr.simple_logging import eprint
from training.omr_datasets.convert_lieder import convert_lieder, git_root, lieder
from training.omr_datasets.photo_maps import (
    PHOTO_SIZE,
    PageMapper,
    PhotoConditions,
    Uvdoc,
    find_marks,
    page_number,
    prepare_photo,
)
from training.omr_datasets.photo_paste import (
    FONTS,
    SIZES,
    engrave_on_template,
    put_on_photo,
)
from training.omr_datasets.photo_staves import cut_page

dataset_root = git_root / "datasets"
photos_root = dataset_root / "photo_pages"
photos_train_index = str(photos_root / "index.txt")
PHOTOS_URL = "https://github.com/slobaspeed/homr-photo-pages/archive/refs/heads/main.zip"
UVDOC_URL = "https://github.com/tanguymagne/UVDoc/archive/refs/heads/main.zip"
FIRST_TRAINING_PAGE, LAST_TRAINING_PAGE = 23, 42
SONGS_PER_PHOTO = 100
SEED = 31

Cut = Callable[..., tuple[list[str], Counter[str]]]


def training_page(number: int | None) -> bool:
    """Only pages T23-T42 train: T01-T22 carry homr's test music, T43-T50 are a test set."""
    return number is not None and FIRST_TRAINING_PAGE <= number <= LAST_TRAINING_PAGE


def template_song_ids(templates: Iterable[dict[str, Any]]) -> set[str]:
    """The Lieder ids (lc...) of the songs printed on template pages."""
    return {m for t in templates for m in re.findall(r"lc\d+", t["source"])}


def _read_song(path: Path) -> str:
    if path.suffix == ".mxl":
        with zipfile.ZipFile(path) as z:
            name = next(
                n
                for n in z.namelist()
                if n.endswith((".xml", ".musicxml")) and not n.startswith("META-INF")
            )
            return z.read(name).decode("utf-8")
    return path.read_text(encoding="utf-8")


def songs(song_dir: Path, excluded: set[str]) -> Iterator[tuple[str, str]]:
    """(name, MusicXML) of every song in song_dir in a fixed shuffled order, without the songs on
    template pages, songs with tremolos (homr's tokens have none) and songs of more than 2 parts."""
    paths = sorted(p for p in song_dir.rglob("*") if p.suffix in (".musicxml", ".mxl"))
    random.Random(SEED).shuffle(paths)
    for path in paths:
        if set(re.findall(r"lc\d+", path.stem)) & excluded:
            continue
        xml = _read_song(path)
        parts = ET.fromstring(xml.encode("utf-8")).findall("part")  # noqa: S314
        if "<tremolo" in xml or len(parts) > 2:
            continue
        yield path.stem, xml


class CachedMaps:
    """UVDoc's maps of a photo, read from `cache` when there, else made by mapper() (made once,
    on first need) and saved there."""

    def __init__(self, mapper: Callable[[], PageMapper]) -> None:
        self.mapper = mapper
        self.made: PageMapper | None = None
        self.cache = Path()

    def page_maps(self, photo: NDArray) -> tuple[NDArray, NDArray]:
        if self.cache.is_file():
            maps = np.load(self.cache)
            return maps["map_x"], maps["map_y"]
        if self.made is None:
            self.made = self.mapper()
        map_x, map_y = self.made.page_maps(photo)
        np.savez_compressed(self.cache, map_x=map_x, map_y=map_y)
        return map_x, map_y


def photo_conditions(
    photo_dir: Path,
    templates: dict[str, dict[str, Any]],
    root: Path,
    mapper: Callable[[], PageMapper],
) -> dict[str, PhotoConditions]:
    """The conditions of every training photo. UVDoc's maps are cached in root/maps, so a second
    run doesn't need UVDoc; mapper() is only made when a map is missing."""
    (root / "maps").mkdir(parents=True, exist_ok=True)
    cached = CachedMaps(mapper)
    found = {}
    for path in sorted(photo_dir.glob("*.jpg")):
        photo = cv2.imread(str(path))
        if photo is None:
            eprint(f"{path.name}: not an image, left out")
            continue
        scale = PHOTO_SIZE / max(photo.shape[:2])
        small = cv2.resize(photo, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        number = page_number(find_marks(small))
        if not training_page(number):
            eprint(f"{path.name}: page {number} is not a training page, left out")
            continue
        cached.cache = root / "maps" / f"{path.stem}.npz"
        cond = prepare_photo(path, templates[f"T{number:02d}"], cached)
        if cond is not None:
            found[path.stem] = cond
    return found


def make_pages(
    conds: dict[str, PhotoConditions],
    song_list: Iterable[tuple[str, str]],
    pages_dir: Path,
    songs_per_photo: int,
) -> None:
    """songs_per_photo songs onto every photo, round the photos in turn: <photo>__<n>.jpg, its
    answer (.musicxml) and where its staves are (.page.json). Skipped when already done."""
    done = pages_dir / "done"
    if done.exists():
        return
    pages_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    names = sorted(conds)
    want, made = songs_per_photo * len(names), 0
    for song, xml in song_list:
        if made == want:
            break
        page = engrave_on_template(xml, rng.choice(FONTS), rng.choice(SIZES))
        if page is None:
            continue
        name = names[made % len(names)]
        cond = conds[name]
        stem = pages_dir / f"{name}__{made // len(names):03d}"
        image = put_on_photo(cond.photo, cond, page.flat)
        cv2.imwrite(str(stem.with_suffix(".jpg")), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        stem.with_suffix(".musicxml").write_text(page.musicxml, encoding="utf-8")
        info = {"photo": name, "song": song, "systems": page.systems}
        stem.with_suffix(".page.json").write_text(json.dumps(info), encoding="utf-8")
        made += 1
    eprint(f"{made} photo pages made")
    done.write_text(str(made))


def cut_pages(
    conds: dict[str, PhotoConditions],
    pages_dir: Path,
    staves_dir: Path,
    cut: Cut,
    config: Any,
) -> list[str]:
    """Every page cut into staves; the index lines of all of them."""
    staves_dir.mkdir(parents=True, exist_ok=True)
    lines, dropped = [], Counter[str]()
    for image_path in sorted(pages_dir.glob("*.jpg")):
        page = json.loads(image_path.with_suffix(".page.json").read_text(encoding="utf-8"))
        cond = conds[page["photo"]]
        found, why = cut(image_path, (cond.map_x, cond.map_y), page, config, staves_dir)
        lines += found
        dropped += why
    eprint(f"{len(lines)} staves kept; dropped: {dict(dropped)}")
    return lines


def run(
    photo_dir: Path,
    templates: dict[str, dict[str, Any]],
    song_list: Iterable[tuple[str, str]],
    root: Path,
    *,
    mapper: Callable[[], PageMapper],
    cut: Cut = cut_page,
    songs_per_photo: int = SONGS_PER_PHOTO,
    config: Any = None,
) -> int:
    """Maps, then pages, then staves, each finished before the next starts (UVDoc runs on the CPU,
    homr's segmentation on the GPU). Writes root/index.txt afresh; returns its number of lines."""
    conds = photo_conditions(photo_dir, templates, root, mapper)
    make_pages(conds, song_list, root / "pages", songs_per_photo)
    lines = cut_pages(conds, root / "pages", root / "staves", cut, config)
    (root / "index.txt").write_text("".join(f"{line}\n" for line in lines))
    return len(lines)


def _downloaded(url: str, folder: str, variable: str) -> Path:
    if os.environ.get(variable):
        return Path(os.environ[variable])
    path = dataset_root / folder
    if not path.exists():
        eprint(f"Downloading {url}")
        archive = dataset_root / f"{folder}.zip"
        download_file(url, str(archive))
        unzip_file(str(archive), str(dataset_root))
    return path


def convert_photos() -> None:
    from homr.main import ProcessingConfig  # noqa: PLC0415  (the reader, OCR included)

    dataset = _downloaded(PHOTOS_URL, "homr-photo-pages-main", "PHOTO_DATASET_DIR")
    uvdoc_dir = _downloaded(UVDOC_URL, "UVDoc-main", "UVDOC_DIR")
    song_dir = Path(os.environ.get("PHOTO_SONGS_DIR") or Path(lieder) / "rendered_scores")
    if not song_dir.exists():
        convert_lieder()
    data = json.loads((dataset / "templates" / "templates.json").read_text(encoding="utf-8"))
    templates = {t["name"]: t for t in data["templates"]}
    config = ProcessingConfig(
        enable_debug=False,
        enable_cache=False,
        write_staff_positions=False,
        read_staff_positions=False,
        selected_staff=-1,
        transformer_use_gpu=False,
        segnet_use_gpu=True,
        coreml_encoder=False,
        title_detection=False,
    )
    kept = run(
        dataset / "photos" / "train",
        templates,
        songs(song_dir, template_song_ids(templates.values())),
        photos_root,
        mapper=lambda: Uvdoc(uvdoc_dir),
        songs_per_photo=SONGS_PER_PHOTO,
        config=config,
    )
    eprint(f"Photo pages: {kept} staves in {photos_train_index}")


if __name__ == "__main__":
    convert_photos()
