import unittest

from training.transformer.kern_tokens import semantic_to_kern_notation


class TestSemanticToKern(unittest.TestCase):

    def test_semantic_to_kern(self) -> None:
        semantic = "clef-G2	keySignature-AM	timeSignature-3/8	multirest-25	barline	note-A4_eighth	note-E5_quarter	barline	note-C#5_eighth	note-B4_sixteenth	note-C#5_sixteenth	note-A4_eighth	barline	note-B4_eighth	note-E4_quarter	barline	note-A4_sixteenth.	note-B4_thirty_second	note-C#5_eighth	note-B4_eighth	barline	"  # noqa: E501
        actual = semantic_to_kern_notation(semantic)

        expected_content = """**kern
*clefG2
*k[f#c#g#]
*M3/8
1r
=
8a
4ee
=
8cc#
16b
16cc#
8a
=
8b
4e
=
16.a
32b
8cc#
8b
="""

        self.assertEqual(actual, expected_content)
