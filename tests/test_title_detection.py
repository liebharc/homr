import unittest

from homr.title_detection import is_tempo_marking


class TestTitleDetection(unittest.TestCase):
    def test_is_tempo_marking(self) -> None:
        self.assertTrue(is_tempo_marking(""))
        self.assertTrue(is_tempo_marking("60 BPM"))
        self.assertTrue(is_tempo_marking("= 60"))
        self.assertTrue(is_tempo_marking("= 60 J"))
        self.assertFalse(is_tempo_marking("Tabi"))
        self.assertFalse(is_tempo_marking("Arranged byAnimenz"))
