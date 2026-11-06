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
)
from training.omr_datasets.photo_maps import FH, FW


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
    def test_only_pages_23_to_42_train(self) -> None:
        numbers = (None, 1, 22, 23, 42, 43, 50, 51)
        self.assertEqual([n for n in numbers if training_page(n)], [23, 42])

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
