import fnmatch
import os
import unittest
from unittest import mock

from homr.main import get_all_image_files_in_folder


class TestGetAllImageFilesInFolder(unittest.TestCase):
    def test_case_insensitive_file_system_returns_each_file_once(self) -> None:
        folder = "folder"
        files = [os.path.join(folder, name) for name in ["a.png", "b.PNG", "c.jpg"]]

        def case_insensitive_glob(pattern: str, recursive: bool = False) -> list[str]:
            extension = pattern.rsplit(".", 1)[-1].lower()
            return [f for f in files if fnmatch.fnmatch(f.lower(), f"*.{extension}")]

        with mock.patch("homr.main.glob.glob", side_effect=case_insensitive_glob):
            self.assertEqual(get_all_image_files_in_folder(folder), files)
