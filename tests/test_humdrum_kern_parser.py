import unittest

from training.datasets.humdrum_kern_parser import convert_kern_to_tokens
from training.transformer.training_vocabulary import token_lines_to_str


class TestHumdrumKern(unittest.TestCase):

    def test_humdrum_to_semantic_note_duration(self) -> None:
        """
        This file is the reason why we implemented an own conversion logic.
        hum2xml converts the 32th notes incorrectly to 8th notes.

        datasets/grandstaff/scarlatti-d/keyboard-sonatas/L481K025/maj3_up_m-86-91.krn
        """
        kern = """**kern\t**kern
*clefF4\t*clefG2
*k[f#c#g#d#a#e#b#]\t*k[f#c#g#d#a#e#b#]
*M2/4\t*M2/4
=-\t=-
8BB# 8B#L\t32dd#LLL
.\t32cc#J
.\t16b#JJ
8AA# 8A#J\t32cc#LLL
.\t32b#J
.\t16a#JJ
8EE# 8E#L\t8a# 8cc#L
8EE# 8E#J\t8g## 8b#J
=\t=
2AA#\t2a#
==:|!\t==:|!
*-\t*-
"""
        tokens = token_lines_to_str(convert_kern_to_tokens(kern.splitlines()))

        expected = """clef_G2 _ _ _ upper&clef_F4 _ _ _ lower
keySignature_7 . . . .
timeSignature/4 . . . .
note_32 D5 # _ upper&note_8 B3 # _ lower&note_8 B2 # _ lower
note_32 C5 # _ upper
note_16 B4 # _ upper
note_32 C5 # _ upper&note_8 A3 # _ lower&note_8 A2 # _ lower
note_32 B4 # _ upper
note_16 A4 # _ upper
note_8 C5 # _ upper&note_8 A4 # _ upper&note_8 E3 # _ lower&note_8 E2 # _ lower
note_8 B4 # _ upper&note_8 G4 ## _ upper&note_8 E3 # _ lower&note_8 E2 # _ lower
barline . . . .
note_2 A4 # _ upper&note_2 A2 # _ lower
repeatEnd . . . ."""

        self.maxDiff = None
        self.assertEqual(tokens, expected)

    def test_multiple_voices_on_same_staff(self) -> None:
        """
        datasets/grandstaff/scarlatti-d/keyboard-sonatas/L338K450/original_m-41-45.krn
        """
        kern = """**kern\t**kern
*\t*^
*clefF4\t*clefG2\t*clefG2
*k[b-e-]\t*k[b-e-]\t*k[b-e-]
*M4/4\t*M4/4\t*M4/4
*met(c)\t*met(c)\t*met(c)
=-\t=-\t=-
*^\t*\t*
4r\t8GG 8GL\t8gL\t4r
.\t8BB- 8GJ\t16ddL\t.
.\t.\t16dJJ\t.
4r\t8C 8GL\t8e-L\t4r
.\t8AA 8AJ\t16ccL\t.
.\t.\t16f#JJ\t.
4G\t8BB-L\t4.g\t8gL
.\t8CJ\t.\t16e-L
.\t.\t.\t16cJJ
8DL\t4r\t.\t8B-L
8DDJ\t.\t8f#\t8A 8c 8dJ
*v\t*v\t*\t*
*\t*v\t*v
=\t=
1GG\t1B- 1d
==:|!\t==:|!
*-\t*-
"""
        tokens = token_lines_to_str(convert_kern_to_tokens(kern.splitlines()))

        expected = """clef_G2 _ _ _ upper&clef_F4 _ _ _ lower
keySignature_-2 . . . .
timeSignature/4 . . . .
note_8 G4 _ _ upper&rest_4 _ _ _ upper&note_8 G3 _ _ lower&note_8 G2 _ _ lower&rest_4 _ _ _ lower
note_16 D5 _ _ upper&note_8 G3 _ _ lower&note_8 B2 b _ lower
note_16 D4 _ _ upper
note_8 E4 b _ upper&rest_4 _ _ _ upper&note_8 G3 _ _ lower&note_8 C3 _ _ lower&rest_4 _ _ _ lower
note_16 C5 _ _ upper&note_8 A3 _ _ lower&note_8 A2 _ _ lower
note_16 F4 # _ upper
note_4. G4 _ _ upper&note_8 G4 _ _ upper&note_4 G3 _ _ lower&note_8 B2 b _ lower
note_16 E4 b _ upper&note_8 C3 _ _ lower
note_16 C4 _ _ upper
note_8 B3 b _ upper&note_8 D3 _ _ lower&rest_4 _ _ _ lower
note_8 F4 # _ upper&note_8 D4 _ _ upper&note_8 C4 _ _ upper&note_8 A3 _ _ upper&note_8 D2 _ _ lower
barline . . . .
note_1 D4 _ _ upper&note_1 B3 b _ upper&note_1 G2 _ _ lower
repeatEnd . . . ."""

        self.maxDiff = None
        self.assertEqual(tokens, expected)

    def test_triplets(self) -> None:
        """
        datasets/grandstaff/scarlatti-d/keyboard-sonatas/L052K165/maj2_up_m-13-17.krn
        """
        kern = """**kern\t**kern
*clefF4\t*clefG2
*k[f#c#]\t*k[f#c#]
*M3/4\t*M3/4
=-\t=-
4E\t12ggL
.\t12ff#
.\t12eeJ
8r\t4ee
8BL\t.
8G\t4r
8EJ\t.
=\t=
4EE\t4r
4r\t4ee
4g\t4ee
=\t=
4f#\t12eeL
.\t12ff#
.\t12cc#J
4f#\t4dd
4f#\t4dd
=\t=
4e\t12ddL
.\t12cc#
.\t12ddJ
4e\t12eeL
.\t12ff#
.\t12ggJ
4A\t12ggL
.\t12ff#
.\t12eeJ
=\t=
4D\t12ff#L
.\t12ee
.\t12ddJ
8r\t4dd
8AL\t.
8F#\t4r
8DJ\t.
=\t=
*-\t*-
"""
        tokens = token_lines_to_str(convert_kern_to_tokens(kern.splitlines()))

        expected = """clef_G2 _ _ _ upper&clef_F4 _ _ _ lower
keySignature_2 . . . .
timeSignature/4 . . . .
note_12 G5 _ _ upper&note_4 E3 _ _ lower
note_12 F5 # _ upper
note_12 E5 _ _ upper
note_4 E5 _ _ upper&rest_8 _ _ _ lower
note_8 B3 _ _ lower
rest_4 _ _ _ upper&note_8 G3 _ _ lower
note_8 E3 _ _ lower
barline . . . .
rest_4 _ _ _ upper&note_4 E2 _ _ lower
note_4 E5 _ _ upper&rest_4 _ _ _ lower
note_4 E5 _ _ upper&note_4 G4 _ _ lower
barline . . . .
note_12 E5 _ _ upper&note_4 F4 # _ lower
note_12 F5 # _ upper
note_12 C5 # _ upper
note_4 D5 _ _ upper&note_4 F4 # _ lower
note_4 D5 _ _ upper&note_4 F4 # _ lower
barline . . . .
note_12 D5 _ _ upper&note_4 E4 _ _ lower
note_12 C5 # _ upper
note_12 D5 _ _ upper
note_12 E5 _ _ upper&note_4 E4 _ _ lower
note_12 F5 # _ upper
note_12 G5 _ _ upper
note_12 G5 _ _ upper&note_4 A3 _ _ lower
note_12 F5 # _ upper
note_12 E5 _ _ upper
barline . . . .
note_12 F5 # _ upper&note_4 D3 _ _ lower
note_12 E5 _ _ upper
note_12 D5 _ _ upper
note_4 D5 _ _ upper&rest_8 _ _ _ lower
note_8 A3 _ _ lower
rest_4 _ _ _ upper&note_8 F3 # _ lower
note_8 D3 _ _ lower
barline . . . ."""

        self.maxDiff = None
        self.assertEqual(tokens, expected)

    def test_simple(self) -> None:
        """
        datasets/grandstaff/scarlatti-d/keyboard-sonatas/L052K165/maj2_down_m-65-69.krn
        """
        kern = """**kern\t**kern
*clefF4\t*clefG2
*k[b-e-]\t*k[b-e-]
*M3/4\t*M3/4
=-\t=-
4BB-\t16ddLL
.\t16b-
.\t16ee-
.\t16ccJJ
4r\t16ffLL
.\t16dd
.\t16gg
.\t16ee-JJ
4D 4B-\t8ffL
.\t8bb-J[
=\t=
4E- 4B-\t8bb-L]
.\t8ccJ
4F 4B-\t4.cc
4F 4A\t.
.\t16b-LL
.\t16ccJJ
=\t=
4BB- 4B-\t16ddLL
.\t16b-
.\t16ee-
.\t16ccJJ
4r\t16ffLL
.\t16dd
.\t16gg
.\t16ee-JJ
4D\t8ffL
.\t8bb-J[
=\t=
4E-\t8bb-L]
.\t8ccJ
4F\t4.cc
4FF\t.
.\t16b-LL
.\t16aJJ
=\t=
4BBB-\t8b-L
.\t8fJ
4r\t8B-L
.\t8c
4r\t8d
.\t8eJ
=:|!\t=:|!
*-\t*-
"""
        tokens = token_lines_to_str(convert_kern_to_tokens(kern.splitlines()))

        expected = """clef_G2 _ _ _ upper&clef_F4 _ _ _ lower
keySignature_-2 . . . .
timeSignature/4 . . . .
note_16 D5 _ _ upper&note_4 B2 b _ lower
note_16 B4 b _ upper
note_16 E5 b _ upper
note_16 C5 _ _ upper
note_16 F5 _ _ upper&rest_4 _ _ _ lower
note_16 D5 _ _ upper
note_16 G5 _ _ upper
note_16 E5 b _ upper
note_8 F5 _ _ upper&note_4 B3 b _ lower&note_4 D3 _ _ lower
note_8 B5 b tieStart upper
barline . . . .
note_8 B5 b tieStop upper&note_4 B3 b _ lower&note_4 E3 b _ lower
note_8 C5 _ _ upper
note_4. C5 _ _ upper&note_4 B3 b _ lower&note_4 F3 _ _ lower
note_4 A3 _ _ lower&note_4 F3 _ _ lower
note_16 B4 b _ upper
note_16 C5 _ _ upper
barline . . . .
note_16 D5 _ _ upper&note_4 B3 b _ lower&note_4 B2 b _ lower
note_16 B4 b _ upper
note_16 E5 b _ upper
note_16 C5 _ _ upper
note_16 F5 _ _ upper&rest_4 _ _ _ lower
note_16 D5 _ _ upper
note_16 G5 _ _ upper
note_16 E5 b _ upper
note_8 F5 _ _ upper&note_4 D3 _ _ lower
note_8 B5 b tieStart upper
barline . . . .
note_8 B5 b tieStop upper&note_4 E3 b _ lower
note_8 C5 _ _ upper
note_4. C5 _ _ upper&note_4 F3 _ _ lower
note_4 F2 _ _ lower
note_16 B4 b _ upper
note_16 A4 _ _ upper
barline . . . .
note_8 B4 b _ upper&note_4 B1 b _ lower
note_8 F4 _ _ upper
note_8 B3 b _ upper&rest_4 _ _ _ lower
note_8 C4 _ _ upper
note_8 D4 _ _ upper&rest_4 _ _ _ lower
note_8 E4 _ _ upper
repeatEnd . . . ."""

        self.maxDiff = None
        self.assertEqual(tokens, expected)

    def test_accidentals(self) -> None:
        """
        datasets/grandstaff/scarlatti-d/keyboard-sonatas/L335K055/maj3_down_m-21-26.krn
        """
        kern = """**kern	**kern
*clefF4	*clefG2
*k[b-e-a-]	*k[b-e-a-]
*M3/8	*M3/8
=-	=-
4.d-	16dd-LL
.	16ee
.	16ff
.	16ee
.	16ee
.	16ffJJ
=	=
4.d-	16r
.	16ffLL
.	16f
.	16g
.	16a
.	16b-JJ
=	=
4.c	16ccLL
.	16dd
.	16ee-
.	16dd
.	16dd
.	16ee-JJ
=	=
4c 4e-	16r
.	16ee-LL
.	16a
.	16b-
8B-	16e-
.	16g-JJ
=	=
4A	16fLL
.	16ee-
.	16dd-
.	16cc
8B-	16ff
.	16b-JJ
=	=
4c 4e-	16r
.	16ee-LL
.	16a
.	16b-
8B-	16e-
.	16g-JJ
=	=
*-	*-
"""
        tokens = token_lines_to_str(convert_kern_to_tokens(kern.splitlines()))

        expected = """clef_G2 _ _ _ upper&clef_F4 _ _ _ lower
keySignature_-3 . . . .
timeSignature/8 . . . .
note_16 D5 b _ upper&note_4. D4 b _ lower
note_16 E5 _ _ upper
note_16 F5 _ _ upper
note_16 E5 _ _ upper
note_16 E5 _ _ upper
note_16 F5 _ _ upper
barline . . . .
rest_16 _ _ _ upper&note_4. D4 b _ lower
note_16 F5 _ _ upper
note_16 F4 _ _ upper
note_16 G4 _ _ upper
note_16 A4 _ _ upper
note_16 B4 b _ upper
barline . . . .
note_16 C5 _ _ upper&note_4. C4 _ _ lower
note_16 D5 _ _ upper
note_16 E5 b _ upper
note_16 D5 _ _ upper
note_16 D5 _ _ upper
note_16 E5 b _ upper
barline . . . .
rest_16 _ _ _ upper&note_4 E4 b _ lower&note_4 C4 _ _ lower
note_16 E5 b _ upper
note_16 A4 _ _ upper
note_16 B4 b _ upper
note_16 E4 b _ upper&note_8 B3 b _ lower
note_16 G4 b _ upper
barline . . . .
note_16 F4 _ _ upper&note_4 A3 _ _ lower
note_16 E5 b _ upper
note_16 D5 b _ upper
note_16 C5 _ _ upper
note_16 F5 _ _ upper&note_8 B3 b _ lower
note_16 B4 b _ upper
barline . . . .
rest_16 _ _ _ upper&note_4 E4 b _ lower&note_4 C4 _ _ lower
note_16 E5 b _ upper
note_16 A4 _ _ upper
note_16 B4 b _ upper
note_16 E4 b _ upper&note_8 B3 b _ lower
note_16 G4 b _ upper
barline . . . ."""

        self.maxDiff = None
        self.assertEqual(tokens, expected)

    def test_grace_note(self) -> None:
        """
        datasets/grandstaff/scarlatti-d/keyboard-sonatas/L344K114/original_m-154-157.krn
        """
        kern = """**kern\t**kern
*clefF4\t*clefG2
*k[f#c#g#]\t*k[f#c#g#]
*M3/8\t*M3/8
=-\t=-
4.EE\t4.e
=\t=
.\teeq
16r\t4.ee[
16eLL\t.
16a\t.
16g#\t.
16f#\t.
16eJJ\t.
=\t=
16dLL\t4.ee_
16c#\t.
16B\t.
16A\t.
16G#\t.
16F#JJ\t.
=\t=
16ELL\t4.ee_
16E\t.
16A\t.
16G#\t.
16F#\t.
16EJJ\t.
=\t=
*-\t*-
"""
        tokens = token_lines_to_str(convert_kern_to_tokens(kern.splitlines()))

        expected = """clef_G2 _ _ _ upper&clef_F4 _ _ _ lower
keySignature_3 . . . .
timeSignature/8 . . . .
note_4. E4 _ _ upper&note_4. E2 _ _ lower
barline . . . .
note_4G E5 _ _ upper
note_4. E5 _ tieStart upper&rest_16 _ _ _ lower
note_16 E4 _ _ lower
note_16 A4 _ _ lower
note_16 G4 # _ lower
note_16 F4 # _ lower
note_16 E4 _ _ lower
barline . . . .
note_4. E5 _ _ upper&note_16 D4 _ _ lower
note_16 C4 # _ lower
note_16 B3 _ _ lower
note_16 A3 _ _ lower
note_16 G3 # _ lower
note_16 F3 # _ lower
barline . . . .
note_4. E5 _ _ upper&note_16 E3 _ _ lower
note_16 E3 _ _ lower
note_16 A3 _ _ lower
note_16 G3 # _ lower
note_16 F3 # _ lower
note_16 E3 _ _ lower
barline . . . ."""

        self.maxDiff = None
        self.assertEqual(tokens, expected)

    def test_staff_asignment(self) -> None:
        kern = """**kern	**kern
*	*^
*clefF4	*clefG2	*clefG2
*k[]	*k[]	*k[]
*M2/4	*M2/4	*M2/4
=-	=-	=-
8EE	8r	16bLL
.	.	16g#
8r	8eeL	16e
.	.	16g#JJ
4r	8ff	16bLL
.	.	16g#
.	8eeJ	16e
.	.	16g#JJ
*	*v	*v
=	="""
        tokens = token_lines_to_str(convert_kern_to_tokens(kern.splitlines()))
        expected = """clef_G2 _ _ _ upper&clef_F4 _ _ _ lower
keySignature_0 . . . .
timeSignature/4 . . . .
note_16 B4 _ _ upper&rest_8 _ _ _ upper&note_8 E2 _ _ lower
note_16 G4 # _ upper
note_8 E5 _ _ upper&note_16 E4 _ _ upper&rest_8 _ _ _ lower
note_16 G4 # _ upper
note_8 F5 _ _ upper&note_16 B4 _ _ upper&rest_4 _ _ _ lower
note_16 G4 # _ upper
note_8 E5 _ _ upper&note_16 E4 _ _ upper
note_16 G4 # _ upper
barline . . . ."""
        self.maxDiff = None
        self.assertEqual(tokens, expected)
