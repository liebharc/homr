import unittest

from training.humdrum_kern import convert_kern_to_semantic


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
        semantic = convert_kern_to_semantic(kern.splitlines())

        upper = "clef-G2 keySignature-C#M timeSignature-2/4 note-D5_thirty_second note-C5_thirty_second note-B4_sixteenth note-C5_thirty_second note-B4_thirty_second note-A4_sixteenth note-A4_eighth|note-C5_eighth note-G4_eighth|note-B4_eighth barline note-A4_half barline"  # noqa: E501
        lower = "clef-F4 keySignature-C#M timeSignature-2/4 note-B2_eighth|note-B3_eighth note-A2_eighth|note-A3_eighth note-E2_eighth|note-E3_eighth note-E2_eighth|note-E3_eighth barline note-A2_half barline"  # noqa: E501

        self.maxDiff = None
        self.assertEqual(semantic, [upper, lower])

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
        semantic = convert_kern_to_semantic(kern.splitlines())

        upper = "clef-G2 keySignature-BbM timeSignature-4/4 note-G4_eighth|rest-quarter note-D5_sixteenth note-D4_sixteenth note-E4_eighth|rest-quarter note-C5_sixteenth note-F4#_sixteenth note-G4_quarter.|note-G4_eighth note-E4_sixteenth note-C4_sixteenth note-B3_eighth note-F4_eighth|note-A3_eighth|note-C4_eighth|note-D4_eighth barline note-B3_whole|note-D4_whole barline"  # noqa: E501
        lower = "clef-F4 keySignature-BbM timeSignature-4/4 rest-quarter|note-G2_eighth|note-G3_eighth note-B2_eighth|note-G3_eighth rest-quarter|note-C3_eighth|note-G3_eighth note-A2_eighth|note-A3_eighth note-G3_quarter|note-B2_eighth note-C3_eighth note-D3_eighth|rest-quarter note-D2_eighth barline note-G2_whole barline"  # noqa: E501

        self.maxDiff = None
        self.assertEqual(semantic, [upper, lower])

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
        semantic = convert_kern_to_semantic(kern.splitlines())

        upper = "clef-G2 keySignature-DM timeSignature-3/4 note-G5_eighth³ note-F5_eighth³ note-E5_eighth³ note-E5_quarter rest-quarter barline rest-quarter note-E5_quarter note-E5_quarter barline note-E5_eighth³ note-F5_eighth³ note-C5_eighth³ note-D5_quarter note-D5_quarter barline note-D5_eighth³ note-C5_eighth³ note-D5_eighth³ note-E5_eighth³ note-F5_eighth³ note-G5_eighth³ note-G5_eighth³ note-F5_eighth³ note-E5_eighth³ barline note-F5_eighth³ note-E5_eighth³ note-D5_eighth³ note-D5_quarter rest-quarter barline"  # noqa: E501

        self.maxDiff = None
        self.assertEqual(semantic[0], upper)

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
        semantic = convert_kern_to_semantic(kern.splitlines())

        upper = "clef-G2 keySignature-BbM timeSignature-3/4 note-D5_sixteenth note-B4_sixteenth note-E5_sixteenth note-C5_sixteenth note-F5_sixteenth note-D5_sixteenth note-G5_sixteenth note-E5_sixteenth note-F5_eighth note-B5_eighth barline note-B5_eighth note-C5_eighth note-C5_quarter. note-B4_sixteenth note-C5_sixteenth barline note-D5_sixteenth note-B4_sixteenth note-E5_sixteenth note-C5_sixteenth note-F5_sixteenth note-D5_sixteenth note-G5_sixteenth note-E5_sixteenth note-F5_eighth note-B5_eighth barline note-B5_eighth note-C5_eighth note-C5_quarter. note-B4_sixteenth note-A4_sixteenth barline note-B4_eighth note-F4_eighth note-B3_eighth note-C4_eighth note-D4_eighth note-E4N_eighth barline"  # noqa: E501

        self.maxDiff = None
        self.assertEqual(semantic[0], upper)

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
        semantic = convert_kern_to_semantic(kern.splitlines())

        upper = "clef-G2 keySignature-EbM timeSignature-3/8 note-D5b_sixteenth note-E5N_sixteenth note-F5_sixteenth note-E5_sixteenth note-E5_sixteenth note-F5_sixteenth barline rest-sixteenth note-F5_sixteenth note-F4_sixteenth note-G4_sixteenth note-A4N_sixteenth note-B4_sixteenth barline note-C5_sixteenth note-D5_sixteenth note-E5_sixteenth note-D5_sixteenth note-D5_sixteenth note-E5_sixteenth barline rest-sixteenth note-E5_sixteenth note-A4N_sixteenth note-B4_sixteenth note-E4_sixteenth note-G4b_sixteenth barline note-F4_sixteenth note-E5_sixteenth note-D5b_sixteenth note-C5_sixteenth note-F5_sixteenth note-B4_sixteenth barline rest-sixteenth note-E5_sixteenth note-A4N_sixteenth note-B4_sixteenth note-E4_sixteenth note-G4b_sixteenth barline"  # noqa: E501

        self.maxDiff = None
        self.assertEqual(semantic[0], upper)

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
        semantic = convert_kern_to_semantic(kern.splitlines())

        upper = "clef-G2 keySignature-AM timeSignature-3/8 note-E4_quarter. barline note-E5_hundred_twenty_eighth|note-E5_quarter. barline note-E5_quarter. barline note-E5_quarter. barline"  # noqa: E501

        self.maxDiff = None
        self.assertEqual(semantic[0], upper)
