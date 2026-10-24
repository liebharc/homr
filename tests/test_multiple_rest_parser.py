import unittest

from training.omr_datasets.music_xml_parser import music_xml_string_to_tokens


class TestMultipleRestParser(unittest.TestCase):
    def test_multiple_rest_collapses_hidden_measures(self) -> None:
        xml = (
            '<score-partwise version="4.0"><part id="P1">'
            '<measure number="1"><attributes><divisions>4</divisions>'
            "<time><beats>4</beats><beat-type>4</beat-type></time>"
            "<clef><sign>G</sign><line>2</line></clef>"
            "<measure-style><multiple-rest>3</multiple-rest></measure-style>"
            '</attributes><note><rest measure="yes"/><duration>16</duration></note></measure>'
            '<measure number="2"><note><rest measure="yes"/>'
            "<duration>16</duration></note></measure>"
            '<measure number="3"><note><rest measure="yes"/><duration>16</duration></note>'
            '<barline location="right"><bar-style>light-heavy</bar-style></barline></measure>'
            '<measure number="4"><note><rest measure="yes"/>'
            "<duration>16</duration></note></measure>"
            "</part></score-partwise>"
        )

        measures = music_xml_string_to_tokens(xml)[0]

        self.assertEqual(len(measures), 2)
        self.assertEqual(measures[0][2].rhythm, "rest_3m")
        self.assertEqual(measures[0][3].rhythm, "barline")
        self.assertEqual(measures[1][0].rhythm, "rest_1")
