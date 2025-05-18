import unittest

from homr.tr_omr_parser_kern import TrOMRParserKern


class TestTrOmrParserKern(unittest.TestCase):

    unit_size = 3

    def test_parsing(self) -> None:
        data = "*clefG2\n*k[b-] \n *M4/4 \n 2A- \n 8e- \n 8 \n 8 \n = \n 2.e- \n 4r- \n = \n 2d- \n 8.d- \n 8e- \n = \n 4. \n 8e- \n 2"  # noqa: E501

        parser = TrOMRParserKern()
        actual = parser.parse_tr_omr_output(data)
        self.assertTrue(len(actual.measures) > 0)
