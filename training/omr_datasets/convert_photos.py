"""
Training staves from phone photos of printed template pages
(github.com/slobaspeed/homr-photo-pages): the printed music is erased and Lieder songs are
pasted onto the same real paper, then cut into staves the way homr reads them, each with its
exact answer.

Downloads the photos and UVDoc (github.com/tanguymagne/UVDoc, MIT), and uses the Lieder songs
rendered by convert_lieder.py. Photos of pages T23 and up are used, except T43-T50, which are a
test set; T01-T22 carry the music of homr's photo test set. A photo showing any mark of those
pages is left out.

For development, three environment variables point at local copies instead:
PHOTO_DATASET_DIR (the photo repo), UVDOC_DIR (UVDoc) and PHOTO_SONGS_DIR (MusicXML or .mxl songs).
"""

import json
import multiprocessing
import os
import random
import re
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from homr.download_utils import download_file, unzip_file
from homr.simple_logging import eprint
from training.omr_datasets.convert_lieder import convert_lieder, git_root, lieder
from training.omr_datasets.photo_index import photos_root, photos_train_index
from training.omr_datasets.photo_maps import (
    PHOTO_SIZE,
    PageMapper,
    Uvdoc,
    find_marks,
    page_number,
    prepare_photo,
)
from training.omr_datasets.photo_paste import (
    FONTS,
    SIZES,
    bar_count,
    engrave_on_template,
    put_on_photo,
)
from training.omr_datasets.photo_staves import cut_page

dataset_root = git_root / "datasets"
PHOTOS_URL = "https://github.com/slobaspeed/homr-photo-pages/archive/refs/heads/main.zip"
UVDOC_COMMIT = "4c9b82b537057aff2526e6dd118a847cdd072e82"  # pinned: its model.py is run
UVDOC_URL = f"https://github.com/tanguymagne/UVDoc/archive/{UVDOC_COMMIT}.zip"
FIRST_TRAINING_PAGE, LAST_PAGE = 23, 124  # 8 * 124 + 7 = 999, the last id of the marks
FORBIDDEN_PAGES = [*range(1, 23), *range(43, 51)]  # homr's test music; the test photos
SONGS_PER_PHOTO = 100
WORKERS = min(8, os.cpu_count() or 1)  # making pages, on the CPU
CUT_WORKERS = 2  # cutting staves: each runs homr's segmentation on the GPU
SEED = 31

Cut = Callable[..., tuple[list[str], Counter[str]]]


def training_page(number: int | None) -> bool:
    """Pages T23 and up train, except T43-T50: T01-T22 carry homr's test music, T43-T50 are a
    test set."""
    return (
        number is not None
        and FIRST_TRAINING_PAGE <= number <= LAST_PAGE
        and number not in FORBIDDEN_PAGES
    )


def training_photo(ids: Iterable[int]) -> int | None:
    """The training page a photo's marks show, or None. A single mark of a forbidden page leaves
    the photo out, so neither a stray detection nor two pages in one shot let it in."""
    ids = list(ids)
    if any(i // 8 in FORBIDDEN_PAGES for i in ids):
        return None
    number = page_number(ids)
    return number if training_page(number) else None


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


def songs(song_dir: Path, excluded: set[str]) -> list[tuple[str, Path]]:
    """(name, path) of every song in song_dir in a fixed shuffled order, without the songs on
    template pages."""
    paths = sorted(p for p in song_dir.rglob("*") if p.suffix in (".musicxml", ".mxl"))
    random.Random(SEED).shuffle(paths)
    return [(p.stem, p) for p in paths if not set(re.findall(r"lc\d+", p.stem)) & excluded]


def usable_song(source: str | Path) -> str | None:
    """The song's MusicXML (source: the text, or a file), or None for a song that can't be read,
    has tremolos (homr's tokens have none) or more than 2 parts."""
    try:
        xml = _read_song(source) if isinstance(source, Path) else source
        parts = ET.fromstring(xml.encode("utf-8")).findall("part")  # noqa: S314
    except (OSError, UnicodeDecodeError, ET.ParseError, StopIteration, zipfile.BadZipFile):
        return None
    if "<tremolo" in xml or len(parts) > 2:
        return None
    return xml


class CachedMaps:
    """UVDoc's maps of a photo, read from `cache` when there, else made by mapper() (made once,
    on first need) and saved there."""

    def __init__(self, mapper: Callable[[], PageMapper] | None) -> None:
        self.mapper = mapper
        self.made: PageMapper | None = None
        self.cache = Path()

    def page_maps(self, photo: NDArray) -> tuple[NDArray, NDArray]:
        if self.cache.is_file():
            maps = np.load(self.cache)
            return maps["map_x"], maps["map_y"]
        if self.mapper is None:
            raise FileNotFoundError(self.cache)
        if self.made is None:
            self.made = self.mapper()
        map_x, map_y = self.made.page_maps(photo)
        np.savez_compressed(self.cache, map_x=map_x, map_y=map_y)
        return map_x, map_y


@dataclass
class Photo:
    """A usable training photo: where it is, its template, UVDoc's cached maps of it and the
    finished map to the flat page (both in root/maps)."""

    path: Path
    template: dict[str, Any]
    uvdoc_maps: Path
    page_maps: Path


def find_photos(
    photo_dir: Path,
    templates: dict[str, dict[str, Any]],
    root: Path,
    mapper: Callable[[], PageMapper],
) -> dict[str, Photo]:
    """Every usable training photo by name. UVDoc's maps are cached in root/maps, so a second run
    doesn't need UVDoc; mapper() is only made when a map is missing."""
    (root / "maps").mkdir(parents=True, exist_ok=True)
    cached = CachedMaps(mapper)
    found = {}
    paths = sorted(p for p in photo_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg"))
    for path in paths:
        photo = cv2.imread(str(path))
        if photo is None:
            eprint(f"{path.name}: not an image, left out")
            continue
        scale = PHOTO_SIZE / max(photo.shape[:2])
        small = cv2.resize(photo, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        ids = list(find_marks(small))
        number = training_photo(ids)
        if number is None or f"T{number:02d}" not in templates:
            pages = sorted({f"T{i // 8:02d}" for i in ids}) or ["no marks"]
            eprint(f"{path.name}: {', '.join(pages)}, not a training page, left out")
            continue
        template = templates[f"T{number:02d}"]
        cached.cache = root / "maps" / f"{path.stem}.npz"
        try:
            cond = prepare_photo(path, template, cached)
        except Exception as e:  # e.g. a degenerate fit of the marks
            eprint(f"{path.name}: no map ({e}), left out")
            continue
        if cond is not None:
            page_maps = root / "maps" / f"{path.stem}.page.npz"
            np.savez_compressed(page_maps, map_x=cond.map_x, map_y=cond.map_y)
            found[path.stem] = Photo(path, template, cached.cache, page_maps)
    return found


def _map_jobs(job: Callable[[Any], Any], jobs: list[Any], workers: int) -> list[Any]:
    """job on every one of jobs, in order; on `workers` processes when more than one."""
    if workers <= 1 or len(jobs) <= 1:
        return [job(j) for j in jobs]
    # the default start: on Linux a copy of this process, so a calling script (training/train.py)
    # isn't run again in every worker; elsewhere such a script needs `if __name__ == "__main__"`
    with multiprocessing.Pool(min(workers, len(jobs))) as pool:
        return pool.map(job, jobs)


def _photo_pages(
    job: tuple[int, int, str, Photo, Sequence[tuple[str, str | Path]], int, Path],
) -> int:
    """The pages of one photo: the songs whose place in the list falls to it (every photos-th,
    from its own place), until it has songs_per_photo pages. Each song starts at a random bar."""
    k, photos, name, photo, song_list, songs_per_photo, pages_dir = job
    cv2.setNumThreads(1)  # one thread per worker; OpenCV's thread pool and forked workers clash
    cached = CachedMaps(None)
    cached.cache = photo.uvdoc_maps
    cond = prepare_photo(photo.path, photo.template, cached)
    if cond is None:
        return 0
    made, broken = 0, 0
    for i in range(k, len(song_list), photos):
        if made == songs_per_photo:
            break
        song, source = song_list[i]
        xml = usable_song(source)
        if xml is None:
            continue
        rng = random.Random(f"{SEED}:{i}")  # the same choices whatever the number of workers
        start = rng.randrange(max(1, bar_count(xml) - 3))  # at least 4 bars left from there
        try:
            page = engrave_on_template(xml, rng.choice(FONTS), rng.choice(SIZES), start)
        except Exception:  # a song Verovio or the layout can't take
            broken += 1
            continue
        if page is None:
            continue
        stem = f"{name}__{made:03d}"
        image = put_on_photo(cond.photo, cond, page.flat)
        cv2.imwrite(str(pages_dir / f"{stem}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        (pages_dir / f"{stem}.musicxml").write_text(page.musicxml, encoding="utf-8")
        info = {"photo": name, "song": song, "start": start, "systems": page.systems}
        (pages_dir / f"{stem}.page.json").write_text(json.dumps(info), encoding="utf-8")
        made += 1
    eprint(f"{name}: {made} pages made, {broken} songs couldn't be engraved")
    return made


def make_pages(
    photos: dict[str, Photo],
    song_list: Sequence[tuple[str, str | Path]],
    pages_dir: Path,
    songs_per_photo: int,
    *,
    workers: int = 1,
) -> None:
    """songs_per_photo songs onto every photo: <photo>__<n>.jpg, its answer (.musicxml) and where
    its staves are (.page.json). Skipped when done for the same photos and number of songs; made
    afresh when photos were added or taken away."""
    names = sorted(photos)
    done = pages_dir / "done.json"
    wanted = {"photos": names, "songs_per_photo": songs_per_photo}
    if done.exists() and json.loads(done.read_text(encoding="utf-8")) == wanted:
        return
    pages_dir.mkdir(parents=True, exist_ok=True)
    for old in pages_dir.iterdir():
        old.unlink()
    jobs = [
        (k, len(names), name, photos[name], song_list, songs_per_photo, pages_dir)
        for k, name in enumerate(names)
    ]
    made = sum(_map_jobs(_photo_pages, jobs, workers))
    eprint(f"{made} photo pages made")
    if made:
        done.write_text(json.dumps(wanted), encoding="utf-8")


_loaded_maps: dict[Path, tuple[NDArray, NDArray]] = {}  # per process: each photo's map once


def _cut_one(job: tuple[Path, Path, Any, Path, Cut]) -> tuple[list[str], Counter[str]]:
    image_path, page_maps, config, staves_dir, cut = job
    cv2.setNumThreads(1)
    if page_maps not in _loaded_maps:
        maps = np.load(page_maps)
        _loaded_maps[page_maps] = (maps["map_x"], maps["map_y"])
    page = json.loads(image_path.with_suffix(".page.json").read_text(encoding="utf-8"))
    return cut(image_path, _loaded_maps[page_maps], page, config, staves_dir)


def cut_pages(
    photos: dict[str, Photo],
    pages_dir: Path,
    staves_dir: Path,
    cut: Cut,
    config: Any,
    *,
    workers: int = 1,
) -> list[str]:
    """Every page cut into staves; the index lines of all of them, in page order."""
    staves_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for image_path in sorted(pages_dir.glob("*.jpg")):
        page = json.loads(image_path.with_suffix(".page.json").read_text(encoding="utf-8"))
        photo = photos.get(page["photo"])
        if photo is not None:  # else its photo is gone
            jobs.append((image_path, photo.page_maps, config, staves_dir, cut))
    lines, dropped = [], Counter[str]()
    for found, why in _map_jobs(_cut_one, jobs, workers):
        lines += found
        dropped += why
    eprint(f"{len(lines)} staves kept; dropped: {dict(dropped)}")
    return lines


def run(
    photo_dir: Path,
    templates: dict[str, dict[str, Any]],
    song_list: Sequence[tuple[str, str | Path]],
    root: Path,
    *,
    mapper: Callable[[], PageMapper],
    cut: Cut = cut_page,
    songs_per_photo: int = SONGS_PER_PHOTO,
    config: Any = None,
    workers: int = 1,
    cut_workers: int = 1,
) -> int:
    """Maps, then pages, then staves, each finished before the next starts (UVDoc runs on the CPU,
    homr's segmentation on the GPU). Pages are made on `workers` processes, staves cut on
    `cut_workers`; the result doesn't depend on either. Writes root/index.txt afresh; returns its
    number of lines. Raises when no staff is kept, so training doesn't quietly run without the
    photos. song_list: (name, MusicXML text or file)."""
    photos = find_photos(photo_dir, templates, root, mapper)
    make_pages(photos, song_list, root / "pages", songs_per_photo, workers=workers)
    lines = cut_pages(photos, root / "pages", root / "staves", cut, config, workers=cut_workers)
    if not lines:
        raise RuntimeError(f"no photo staves kept from {photo_dir}: see the reasons above")
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
    from homr.main import (  # noqa: PLC0415  (the reader, OCR included)
        ProcessingConfig,
        download_weights,
    )

    download_weights(segnet_use_gpu=True, transformer_use_gpu=True, coreml_encoder=False)

    dataset = _downloaded(PHOTOS_URL, "homr-photo-pages-main", "PHOTO_DATASET_DIR")
    uvdoc_dir = _downloaded(UVDOC_URL, f"UVDoc-{UVDOC_COMMIT}", "UVDOC_DIR")
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
        workers=WORKERS,
        cut_workers=CUT_WORKERS,
    )
    eprint(f"Photo pages: {kept} staves in {photos_train_index}")


if __name__ == "__main__":
    convert_photos()
