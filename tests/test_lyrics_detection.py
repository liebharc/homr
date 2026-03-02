import unittest
from typing import Any

import musicxml.xmlelement.xmlelement as mxl
import numpy as np

from homr import lyrics_detection
from homr.lyrics_detection import (
    LyricCandidate,
    _filter_lyrics_between_staves,
    _group_lyrics_into_lines,
    _normalize_ocr_results,
    _split_joined_token,
    _tokenize_ocr_result,
    assign_lyrics_to_symbols,
    lyrics_by_measure,
    lyrics_by_measure_per_verse,
)
from homr.music_xml_generator import XmlGeneratorArguments, generate_xml
from homr.transformer.vocabulary import EncodedSymbol, empty


class TestLyricDetection(unittest.TestCase):
    def test_assigns_lyrics_to_closest_note_in_measure(self) -> None:
        symbols = [
            EncodedSymbol(
                "note_4",
                pitch="C4",
                lift=empty,
                articulation=empty,
                position="upper",
                coordinates=(20, 60),
            ),
            EncodedSymbol(
                "note_4",
                pitch="D4",
                lift=empty,
                articulation=empty,
                position="upper",
                coordinates=(60, 60),
            ),
            EncodedSymbol("barline", coordinates=(80, 60)),
            EncodedSymbol(
                "note_4",
                pitch="E4",
                lift=empty,
                articulation=empty,
                position="upper",
                coordinates=(120, 60),
            ),
            EncodedSymbol(
                "note_4",
                pitch="F4",
                lift=empty,
                articulation=empty,
                position="upper",
                coordinates=(160, 60),
            ),
        ]
        lyrics = [
            LyricCandidate(text="hello", x=58, y=110),
            LyricCandidate(text="world", x=123, y=110),
        ]

        assign_lyrics_to_symbols(symbols, lyrics)

        self.assertIsNone(symbols[0].lyric)
        self.assertEqual(symbols[1].lyric, "hello")
        self.assertEqual(symbols[3].lyric, "world")
        self.assertIsNone(symbols[4].lyric)

    def test_ignores_chord_followup_note_for_lyric_anchor(self) -> None:
        symbols = [
            EncodedSymbol(
                "note_4",
                pitch="C4",
                lift=empty,
                articulation=empty,
                position="upper",
                coordinates=(40, 50),
            ),
            EncodedSymbol("chord"),
            EncodedSymbol(
                "note_4",
                pitch="E4",
                lift=empty,
                articulation=empty,
                position="upper",
                coordinates=(43, 45),
            ),
        ]
        lyrics = [LyricCandidate(text="la", x=43, y=90)]

        assign_lyrics_to_symbols(symbols, lyrics)

        self.assertEqual(symbols[0].lyric, "la")
        self.assertIsNone(symbols[2].lyric)

    def test_writes_lyrics_to_musicxml(self) -> None:
        symbols = [
            EncodedSymbol("clef_G2", position="upper"),
            EncodedSymbol(
                "note_4",
                pitch="C4",
                lift=empty,
                articulation=empty,
                position="upper",
                coordinates=(40, 50),
            ),
            EncodedSymbol("barline"),
        ]
        symbols[1].lyric = "hello"
        xml, _ = generate_xml(XmlGeneratorArguments(), [symbols], "")

        found_lyrics: list[str] = []

        def walk(node: Any) -> None:
            if isinstance(node, list):
                for child in node:
                    walk(child)
                return
            if isinstance(node, mxl.XMLLyric):
                text = node.get_children_of_type(mxl.XMLText)
                if len(text) > 0:
                    found_lyrics.append(str(text[0].value_))
                return
            if hasattr(node, "get_children"):
                for child in node.get_children():
                    walk(child)

        walk(xml)
        self.assertEqual(found_lyrics, ["hello"])

    def test_groups_lyrics_by_measure(self) -> None:
        symbols = [
            EncodedSymbol("note_4", position="upper", lyric="he"),
            EncodedSymbol("rest_4", position="upper", lyric="llo"),
            EncodedSymbol("barline"),
            EncodedSymbol("note_4", position="upper", lyric="world"),
        ]

        grouped = lyrics_by_measure(symbols)
        self.assertEqual(grouped, [["he", "llo"], ["world"]])

    def test_groups_lyrics_by_measure_per_verse_keeps_measure_alignment(self) -> None:
        first_measure = EncodedSymbol("note_4", position="upper", lyric="play")
        second_measure = EncodedSymbol("note_4", position="upper", lyric="song")
        second_measure.lyric_verses = {1: "song", 2: "strum"}  # type: ignore[attr-defined]
        symbols = [
            first_measure,
            EncodedSymbol("barline"),
            second_measure,
        ]

        grouped = lyrics_by_measure_per_verse(symbols)
        self.assertEqual(grouped[1], [["play"], ["song"]])
        self.assertEqual(grouped[2], [[], ["strum"]])

    def test_assigns_lyric_to_rest_when_closest(self) -> None:
        symbols = [
            EncodedSymbol(
                "rest_4",
                pitch=empty,
                lift=empty,
                articulation=empty,
                position="upper",
                coordinates=(40, 70),
            ),
            EncodedSymbol(
                "note_4",
                pitch="C4",
                lift=empty,
                articulation=empty,
                position="upper",
                coordinates=(100, 70),
            ),
            EncodedSymbol("barline"),
        ]
        lyrics = [LyricCandidate(text="hmm", x=44, y=110)]

        assign_lyrics_to_symbols(symbols, lyrics)
        self.assertEqual(symbols[0].lyric, "hmm")
        self.assertIsNone(symbols[1].lyric)

    def test_allows_multiple_candidates_on_same_note(self) -> None:
        symbols = [
            EncodedSymbol(
                "note_4",
                pitch="C4",
                lift=empty,
                articulation=empty,
                position="upper",
                coordinates=(60, 70),
            ),
            EncodedSymbol(
                "note_4",
                pitch="D4",
                lift=empty,
                articulation=empty,
                position="upper",
                coordinates=(130, 70),
            ),
            EncodedSymbol("barline"),
        ]
        lyrics = [
            LyricCandidate(text="was", x=58, y=120),
            LyricCandidate(text="not", x=64, y=120),
        ]

        assign_lyrics_to_symbols(symbols, lyrics)

        self.assertEqual(symbols[0].lyric, "was not")
        self.assertIsNone(symbols[1].lyric)

    def test_between_staff_lyrics_are_forced_to_upper_staff(self) -> None:
        symbols = [
            EncodedSymbol(
                "note_4",
                pitch="C4",
                lift=empty,
                articulation=empty,
                position="upper",
                coordinates=(130, 80),
            ),
            EncodedSymbol(
                "note_4",
                pitch="C3",
                lift=empty,
                articulation=empty,
                position="lower",
                coordinates=(120, 170),
            ),
            EncodedSymbol("barline"),
        ]
        lyrics = [LyricCandidate(text="between", x=121, y=125)]

        assign_lyrics_to_symbols(symbols, lyrics)
        self.assertEqual(symbols[0].lyric, "between")
        self.assertIsNone(symbols[1].lyric)

    def test_lyrics_below_lower_notes_are_assigned_to_lower_staff(self) -> None:
        symbols = [
            EncodedSymbol(
                "note_4",
                pitch="C4",
                lift=empty,
                articulation=empty,
                position="upper",
                coordinates=(130, 80),
            ),
            EncodedSymbol(
                "note_4",
                pitch="C3",
                lift=empty,
                articulation=empty,
                position="lower",
                coordinates=(120, 170),
            ),
            EncodedSymbol("barline"),
        ]
        lyrics = [LyricCandidate(text="bass", x=121, y=188)]

        assign_lyrics_to_symbols(symbols, lyrics)
        self.assertIsNone(symbols[0].lyric)
        self.assertEqual(symbols[1].lyric, "bass")

    def test_assigns_two_lyric_lines_as_two_verses(self) -> None:
        symbols = [
            EncodedSymbol(
                "note_4",
                pitch="C4",
                lift=empty,
                articulation=empty,
                position="upper",
                coordinates=(50, 70),
            ),
            EncodedSymbol(
                "note_4",
                pitch="D4",
                lift=empty,
                articulation=empty,
                position="upper",
                coordinates=(100, 70),
            ),
            EncodedSymbol("barline"),
        ]
        lyrics = [
            LyricCandidate(text="Hey", x=48, y=110),
            LyricCandidate(text="Mister", x=102, y=110),
            LyricCandidate(text="Hey", x=49, y=126),
            LyricCandidate(text="Guitar", x=101, y=126),
        ]

        assign_lyrics_to_symbols(symbols, lyrics)

        first_verses = symbols[0].lyric_verses  # type: ignore[attr-defined]
        second_verses = symbols[1].lyric_verses  # type: ignore[attr-defined]
        self.assertEqual(first_verses[1], "Hey")
        self.assertEqual(first_verses[2], "Hey")
        self.assertEqual(second_verses[1], "Mister")
        self.assertEqual(second_verses[2], "Guitar")

    def test_writes_rest_lyrics_to_musicxml(self) -> None:
        symbols = [
            EncodedSymbol("clef_G2", position="upper"),
            EncodedSymbol(
                "rest_4",
                pitch=empty,
                lift=empty,
                articulation=empty,
                position="upper",
                coordinates=(40, 50),
                lyric="restword",
            ),
            EncodedSymbol("barline"),
        ]
        xml, _ = generate_xml(XmlGeneratorArguments(), [symbols], "")

        found_lyrics: list[str] = []

        def walk(node: Any) -> None:
            if isinstance(node, list):
                for child in node:
                    walk(child)
                return
            if isinstance(node, mxl.XMLLyric):
                text = node.get_children_of_type(mxl.XMLText)
                if len(text) > 0:
                    found_lyrics.append(str(text[0].value_))
                return
            if hasattr(node, "get_children"):
                for child in node.get_children():
                    walk(child)

        walk(xml)
        self.assertEqual(found_lyrics, ["restword"])

    def test_writes_multiple_lyric_verses_to_musicxml(self) -> None:
        symbols = [
            EncodedSymbol("clef_G2", position="upper"),
            EncodedSymbol(
                "note_4",
                pitch="C4",
                lift=empty,
                articulation=empty,
                position="upper",
                coordinates=(40, 50),
                lyric="Hey",
            ),
            EncodedSymbol("barline"),
        ]
        symbols[1].lyric_verses = {1: "Hey", 2: "Mister"}  # type: ignore[attr-defined]
        xml, _ = generate_xml(XmlGeneratorArguments(), [symbols], "")

        found_lyrics: list[tuple[str, str]] = []

        def walk(node: Any) -> None:
            if isinstance(node, list):
                for child in node:
                    walk(child)
                return
            if isinstance(node, mxl.XMLLyric):
                text = node.get_children_of_type(mxl.XMLText)
                if len(text) > 0:
                    found_lyrics.append((str(node.number), str(text[0].value_)))
                return
            if hasattr(node, "get_children"):
                for child in node.get_children():
                    walk(child)

        walk(xml)
        self.assertEqual(found_lyrics, [("1", "Hey"), ("2", "Mister")])

    def test_splits_joined_words_from_ocr(self) -> None:
        self.assertEqual(_split_joined_token("Ifound"), ["I", "found"])
        self.assertEqual(_split_joined_token("Darlingjust"), ["Darling", "just"])

    def test_normalizes_paddleocr_results(self) -> None:
        raw = [
            [
                [
                    [[1.0, 2.0], [11.0, 2.0], [11.0, 8.0], [1.0, 8.0]],
                    ("hello", 0.95),
                ]
            ]
        ]
        normalized = _normalize_ocr_results(raw)
        self.assertEqual(len(normalized), 1)
        self.assertEqual(normalized[0][1], "hello")
        self.assertAlmostEqual(normalized[0][2], 0.95)

    def test_ignores_non_paddleocr_result_shapes(self) -> None:
        raw = [{"dt_polys": [], "rec_texts": [], "rec_scores": []}]
        normalized = _normalize_ocr_results(raw)
        self.assertEqual(normalized, [])

    def test_normalizes_paddleocr_mapping_with_numpy_polys(self) -> None:
        raw = [
            {
                "dt_polys": [
                    np.array([[1, 2], [11, 2], [11, 8], [1, 8]], dtype=np.int16),
                ],
                "rec_texts": ["hello"],
                "rec_scores": [0.95],
            }
        ]
        normalized = _normalize_ocr_results(raw)
        self.assertEqual(len(normalized), 1)
        self.assertEqual(normalized[0][1], "hello")
        self.assertAlmostEqual(normalized[0][2], 0.95)

    def test_tokenize_ocr_result_splits_multiline_box(self) -> None:
        bbox = [[0.0, 0.0], [100.0, 0.0], [100.0, 20.0], [0.0, 20.0]]
        tokens = _tokenize_ocr_result(bbox, "Hey Mister\nHey Guitar", 0.91, 100)

        self.assertEqual([token.text for token in tokens], ["Hey", "Mister", "Hey", "Guitar"])
        first_line_y = {round(token.y, 1) for token in tokens[:2]}
        second_line_y = {round(token.y, 1) for token in tokens[2:]}
        self.assertEqual(len(first_line_y), 1)
        self.assertEqual(len(second_line_y), 1)
        self.assertLess(next(iter(first_line_y)), next(iter(second_line_y)))

    def test_group_lyrics_into_lines_splits_by_large_vertical_gap(self) -> None:
        lyrics = [
            LyricCandidate("Hey", 10, 151.0, line_height=8.0),
            LyricCandidate("Mis", 30, 152.5, line_height=8.0),
            LyricCandidate("ter", 50, 154.0, line_height=8.0),
            LyricCandidate("Hey", 12, 171.5, line_height=8.0),
            LyricCandidate("Gui", 35, 172.5, line_height=8.0),
            LyricCandidate("tar", 55, 173.5, line_height=8.0),
        ]

        groups = _group_lyrics_into_lines(lyrics)
        self.assertEqual(len(groups), 2)
        self.assertEqual([token.text for token in groups[0]], ["Hey", "Mis", "ter"])
        self.assertEqual([token.text for token in groups[1]], ["Hey", "Gui", "tar"])

    def test_filter_lyrics_between_staves_discards_outside_tokens(self) -> None:
        symbols = [
            EncodedSymbol(
                "note_4",
                pitch="C4",
                lift=empty,
                articulation=empty,
                position="upper",
                coordinates=(60, 80),
            ),
            EncodedSymbol(
                "note_4",
                pitch="E4",
                lift=empty,
                articulation=empty,
                position="upper",
                coordinates=(100, 88),
            ),
            EncodedSymbol(
                "note_4",
                pitch="C3",
                lift=empty,
                articulation=empty,
                position="lower",
                coordinates=(60, 170),
            ),
            EncodedSymbol(
                "note_4",
                pitch="E3",
                lift=empty,
                articulation=empty,
                position="lower",
                coordinates=(100, 178),
            ),
        ]
        lyrics = [
            LyricCandidate(text="chord", x=40, y=55),
            LyricCandidate(text="in", x=45, y=125),
            LyricCandidate(text="gap", x=65, y=132),
            LyricCandidate(text="bass", x=80, y=198),
        ]

        filtered = _filter_lyrics_between_staves(symbols, lyrics)
        self.assertEqual([token.text for token in filtered], ["in", "gap"])

    def test_filter_lyrics_between_staves_fallback_when_lower_staff_missing(self) -> None:
        symbols = [
            EncodedSymbol(
                "note_4",
                pitch="C4",
                lift=empty,
                articulation=empty,
                position="upper",
                coordinates=(60, 80),
            ),
            EncodedSymbol(
                "note_4",
                pitch="E4",
                lift=empty,
                articulation=empty,
                position="upper",
                coordinates=(100, 120),
            ),
            EncodedSymbol(
                "note_4",
                pitch="G4",
                lift=empty,
                articulation=empty,
                position="upper",
                coordinates=(140, 128),
            ),
        ]
        lyrics = [
            LyricCandidate(text="chord", x=40, y=56),
            LyricCandidate(text="Hey", x=45, y=160),
            LyricCandidate(text="Guitar", x=65, y=182),
            LyricCandidate(text="footer", x=80, y=235),
        ]

        filtered = _filter_lyrics_between_staves(symbols, lyrics)
        self.assertEqual([token.text for token in filtered], ["Hey", "Guitar"])

    def test_run_ocr_converts_grayscale_to_three_channels_for_paddle_predict(self) -> None:
        class _DummyReader:
            def __init__(self) -> None:
                self.received: Any | None = None

            def predict(self, image: Any) -> Any:
                self.received = image
                return [
                    [
                        [
                            [[1.0, 2.0], [11.0, 2.0], [11.0, 8.0], [1.0, 8.0]],
                            ("hello", 0.95),
                        ]
                    ]
                ]

        dummy = _DummyReader()
        original_reader = lyrics_detection._reader
        original_backend = lyrics_detection._reader_backend
        try:
            lyrics_detection._reader = dummy
            lyrics_detection._reader_backend = "paddle"

            gray = np.array([[0, 255], [128, 64]], dtype=np.uint8)
            normalized = lyrics_detection._run_ocr(gray)

            self.assertIsNotNone(dummy.received)
            self.assertEqual(dummy.received.shape, (2, 2, 3))
            self.assertEqual(normalized[0][1], "hello")
        finally:
            lyrics_detection._reader = original_reader
            lyrics_detection._reader_backend = original_backend
