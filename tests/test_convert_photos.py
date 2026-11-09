import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from tests.test_photo_maps import page_with_marks
from tests.test_photo_paste import piano_score
from training.omr_datasets.convert_photos import (
    run,
    songs,
    template_song_ids,
    training_page,
    training_photo,
)
from training.omr_datasets.photo_maps import FH, FW, prepare_photo


def template(number: int) -> dict[str, Any]:
    """The entry of a template whose marks sit where page_with_marks draws them on a page as
    big as the flat page."""
    h, w, mark = FH, FW, 280
    spots = [
        (20, 20),
        (w // 2 - mark // 2, 20),
        (w - 20 - mark, 20),
        (w - 20 - mark, h // 2 - mark // 2),
        (w - 20 - mark, h - 20 - mark),
        (w // 2 - mark // 2, h - 20 - mark),
        (20, h - 20 - mark),
        (20, h // 2 - mark // 2),
    ]
    corners = [[[x, y], [x + mark, y], [x + mark, y + mark], [x, y + mark]] for x, y in spots]
    return {
        "name": f"T{number}",
        "source": "",
        "music_box": [300, 400, 1880, 2700],
        "marks": {"ids": list(range(8 * number, 8 * number + 8)), "corners": corners},
    }


class CountingMapper:
    """Maps a photo of a flat page straight onto the page, counting its calls."""

    calls = 0

    def page_maps(self, photo: NDArray) -> tuple[NDArray, NDArray]:
        CountingMapper.calls += 1
        h, w = photo.shape[:2]
        qx, qy = np.meshgrid(np.linspace(0, FW - 1, w), np.linspace(0, FH - 1, h))
        return qx.astype(np.float32), qy.astype(np.float32)


def fake_cut(image_path: Path, *_: Any) -> tuple[list[str], Counter[str]]:
    return [f"{image_path.stem}.png,{image_path.stem}.tokens"], Counter()


class TestConvertPhotos(unittest.TestCase):
    def test_training_pages_are_t23_up_without_the_test_pages(self) -> None:
        numbers = (None, 1, 22, 23, 42, 43, 50, 51, 124, 125)
        self.assertEqual([n for n in numbers if training_page(n)], [23, 42, 51, 124])

    def test_one_mark_of_a_forbidden_page_refuses_the_photo(self) -> None:
        self.assertEqual(training_photo([184, 185, 186]), 23)
        self.assertIsNone(training_photo([184, 185, 186, 8 * 45]))  # a mark of test page T45
        self.assertIsNone(training_photo([184, 8 * 12]))  # a mark of T12, homr's test music
        self.assertIsNone(training_photo([]))

    def test_the_index_and_its_import_need_no_engraving_libraries(self) -> None:
        code = (
            "import sys; import training.omr_datasets.photo_index as p; "
            "assert p.photos_train_index.endswith('index.txt'); "
            "assert not {'cairosvg', 'verovio', 'scipy'} & set(sys.modules), sys.modules.keys()"
        )
        result = subprocess.run(  # noqa: S603
            [sys.executable, "-c", code], capture_output=True, text=True, check=False
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_template_songs_are_left_out(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            for name in ("a_lc6474604", "b_lc111"):
                (Path(folder) / f"{name}.musicxml").write_text(piano_score(4), encoding="utf-8")
            used = [{"source": "10_Canzonets_10_Recollection_Hob_XXVIa26_lc6474604_mxl"}]
            ids = template_song_ids(used)
            self.assertEqual([name for name, _ in songs(Path(folder), ids)], ["b_lc111"])

    def test_a_second_run_reuses_maps_and_writes_no_duplicate_lines(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root, photos = Path(folder) / "out", Path(folder) / "photos"
            photos.mkdir()
            templates = {t["name"]: t for t in (template(23), template(24), template(45))}
            for number in (23, 24, 45):  # T45 is a test page: never used
                page = page_with_marks(number, size=(FH, FW), mark=280)
                cv2.imwrite(str(photos / f"T{number}.jpg"), page)
            song_list = [(f"song{i}", piano_score(8)) for i in range(4)]
            CountingMapper.calls = 0
            first = run(
                photos,
                templates,
                song_list,
                root,
                mapper=CountingMapper,
                cut=fake_cut,
                songs_per_photo=2,
            )
            self.assertEqual(CountingMapper.calls, 2)
            second = run(
                photos,
                templates,
                song_list,
                root,
                mapper=CountingMapper,
                cut=fake_cut,
                songs_per_photo=2,
            )
            self.assertEqual(CountingMapper.calls, 2)
            lines = (root / "index.txt").read_text().splitlines()
            self.assertEqual(first, second)
            self.assertEqual(len(lines), 4)
            self.assertEqual(len(set(lines)), 4)
            self.assertFalse(any(line.startswith("T45") for line in lines))


def no_cut(image_path: Path, *_: Any) -> tuple[list[str], Counter[str]]:
    return [], Counter({"homr found no staves": 1})


class TestConvertPhotosRuns(unittest.TestCase):
    def setUp(self) -> None:
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name) / "out"
        self.photos = Path(self.folder.name) / "photos"
        self.photos.mkdir()
        self.templates = {t["name"]: t for t in (template(23), template(24))}
        self.songs = [(f"song{i}", piano_score(8)) for i in range(6)]

    def photo(self, number: int, name: str) -> None:
        page = page_with_marks(number, size=(FH, FW), mark=280)
        cv2.imwrite(str(self.photos / name), page)

    def run_it(self, cut: Any = fake_cut) -> int:
        return run(
            self.photos,
            self.templates,
            self.songs,
            self.root,
            mapper=CountingMapper,
            cut=cut,
            songs_per_photo=1,
        )

    def test_a_photo_added_later_is_used(self) -> None:
        self.photo(23, "a.jpg")
        self.assertEqual(self.run_it(), 1)
        self.photo(24, "b.jpg")
        self.assertEqual(self.run_it(), 2)

    def test_a_photo_taken_away_later_is_no_error(self) -> None:
        self.photo(23, "a.jpg")
        self.photo(24, "b.jpg")
        self.assertEqual(self.run_it(), 2)
        (self.photos / "b.jpg").unlink()
        self.assertEqual(self.run_it(), 1)

    def test_names_with_dots_and_capital_extensions_keep_their_pages(self) -> None:
        self.photo(23, "PXL_20261101_083726.MP.JPG")
        self.photo(24, "b.jpeg")
        self.assertEqual(self.run_it(), 2)
        pages = sorted(p.name for p in (self.root / "pages").glob("*.jpg"))
        self.assertEqual(pages, ["PXL_20261101_083726.MP__000.jpg", "b__000.jpg"])

    def test_no_staves_at_all_is_an_error_not_an_empty_index(self) -> None:
        self.photo(23, "a.jpg")
        with self.assertRaises(RuntimeError):
            self.run_it(cut=no_cut)
        self.assertFalse((self.root / "index.txt").exists())

    def test_a_broken_photo_is_skipped(self) -> None:
        self.photo(23, "a.jpg")
        self.photo(24, "b.jpg")

        class Failing(CountingMapper):
            def page_maps(self, photo: NDArray) -> tuple[NDArray, NDArray]:
                if CountingMapper.calls == 0:
                    CountingMapper.calls += 1
                    raise ValueError("a degenerate fit")
                return super().page_maps(photo)

        CountingMapper.calls = 0
        kept = run(
            self.photos,
            self.templates,
            self.songs,
            self.root,
            mapper=Failing,
            cut=fake_cut,
            songs_per_photo=1,
        )
        self.assertEqual(kept, 1)

    def test_a_photo_with_fewer_than_4_marks_is_left_out(self) -> None:
        for marks, usable in ((3, False), (4, True)):
            page = page_with_marks(23, size=(FH, FW), mark=280, marks=marks)
            cv2.imwrite(str(self.photos / f"m{marks}.jpg"), page)
            cond = prepare_photo(self.photos / f"m{marks}.jpg", template(23), CountingMapper())
            self.assertEqual(cond is not None, usable)

    def test_pages_start_at_random_bars(self) -> None:
        self.photo(23, "a.jpg")
        self.songs = [(f"song{i}", piano_score(40)) for i in range(6)]
        run(
            self.photos,
            self.templates,
            self.songs,
            self.root,
            mapper=CountingMapper,
            cut=fake_cut,
            songs_per_photo=6,
        )
        firsts = set()
        for answer in (self.root / "pages").glob("*.musicxml"):
            text = answer.read_text(encoding="utf-8")
            firsts.add(text.split('<measure number="')[1].split('"')[0])
        self.assertGreater(len(firsts), 1)

    def test_more_workers_make_the_same_pages_and_staves(self) -> None:
        self.photo(23, "a.jpg")
        self.photo(24, "b.jpg")
        found = []
        for workers in (1, 2):
            root = Path(self.folder.name) / f"w{workers}"
            run(
                self.photos,
                self.templates,
                self.songs,
                root,
                mapper=CountingMapper,
                cut=fake_cut,
                songs_per_photo=2,
                workers=workers,
            )
            pages = {p.name: p.read_text(encoding="utf-8") for p in (root / "pages").glob("*.json")}
            found.append((pages, (root / "index.txt").read_text()))
        self.assertEqual(found[0], found[1])

    def test_a_song_that_breaks_the_engraver_is_skipped(self) -> None:
        self.photo(23, "a.jpg")
        broken = [("broken", "<score-partwise><part id='P1'><measure/>")]
        self.songs = broken + self.songs
        self.assertEqual(self.run_it(), 1)
