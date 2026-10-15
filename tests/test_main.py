import fnmatch
import os
import tempfile
import unittest
from unittest import mock

from homr.main import get_all_image_files_in_folder, main


class TestGetAllImageFilesInFolder(unittest.TestCase):
    def test_case_insensitive_file_system_returns_each_file_once(self) -> None:
        folder = "folder"
        files = [os.path.join(folder, name) for name in ["a.png", "b.PNG", "c.jpg"]]

        def case_insensitive_glob(pattern: str, recursive: bool = False) -> list[str]:
            extension = pattern.rsplit(".", 1)[-1].lower()
            return [f for f in files if fnmatch.fnmatch(f.lower(), f"*.{extension}")]

        with mock.patch("homr.main.glob.glob", side_effect=case_insensitive_glob):
            self.assertEqual(get_all_image_files_in_folder(folder), files)


class TestExitCode(unittest.TestCase):
    def test_failed_image_exits_with_error(self) -> None:
        with (
            tempfile.NamedTemporaryFile(suffix=".png") as image,
            mock.patch("sys.argv", ["homr", "--no-title", image.name]),
            mock.patch("homr.main.download_weights"),
            mock.patch("homr.main.process_image", side_effect=Exception("No noteheads found")),
            self.assertRaises(SystemExit) as exit_info,
        ):
            main()
        self.assertEqual(exit_info.exception.code, 1)
