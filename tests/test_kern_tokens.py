import unittest

from homr.transformer.configs import default_config
from homr.transformer.split_merge_symbols import merge_kern_tokens
from training.transformer.kern_tokens import (
    filter_for_kern,
    get_symbols,
    split_symbol_into_token,
)


class TestKernTokens(unittest.TestCase):

    def test_filter_for_kern(self) -> None:
        file_content = """!!!COM: Johannes Brahms
!!!OTL: 7 Lieder, Op. 48
!!!OMD: Liebesklage des Mädchens
!!!OMV: 3
!!!YEC: OpenScore (CC0)
**kern	**kern	**dynam	**kern	**text
*part2	*part2	*part2	*part1	*part1
*staff3	*staff2	*staff2/3	*staff1	*staff1
*I"Pianoforte	*	*	*I"Singstimme Voice	*
*I'Pno	*	*	*I'Ob	*
*clefF4	*clefG2	*	*clefG2	*
*k[f#c#g#d#a#]	*k[f#c#g#d#a#]	*	*k[f#c#g#d#a#]	*
!!LO:TX:omd:t
*M6/4	*M6/4	*	*M6/4	*
=1	=1	=1	=1	=1
!	!	!	!LO:TX:B:t=Etwas langsam	!
!LO:TX:b:i:t=col Ped.	!	!	!	!
!	!	!LO:DY:a	!	!
!	!	!LO:DY:n=1:a	!	!
(4Gn 4A# 4c# 4e	(8gn/L	p other-dynamics	2.r	.
.	8ee/	.	.	.
4F# 4A# 4d#	8f#/	.	.	.
.	8dd#/	.	.	.
4E 4F# 4A# 4c#	8f#/	.	.	.
.	8cc#/J	.	.	.
4D# 4F# 4B	8f#/L	.	2r	.
.	8b/	.	.	.
[4F# 4A# 4c#	8f#/	.	.	.
.	8cc#/	.	.	.
4F#] 4B 4d#)	8f#/	.	4b\\	Wer
.	8dd#/J)	.	.	."""
        actual = str.join("\n", filter_for_kern(file_content.splitlines()))

        expected_content = """!!!COM: Johannes Brahms
!!!OTL: 7 Lieder, Op. 48
!!!OMD: Liebesklage des Mädchens
!!!OMV: 3
!!!YEC: OpenScore (CC0)
**kern	**kern	**kern
*part2	*part2	*part1
*staff3	*staff2	*staff1
*I"Pianoforte	*	*I"Singstimme Voice
*I\'Pno	*	*I\'Ob
*clefF4	*clefG2	*clefG2
*k[f#c#g#d#a#]	*k[f#c#g#d#a#]	*k[f#c#g#d#a#]
!!LO:TX:omd:t	.	.
*M6/4	*M6/4	*M6/4
=1	=1	=1
!	!	!LO:TX:B:t=Etwas langsam
!LO:TX:b:i:t=col Ped.	!	!
!	!	!
!	!	!
(4Gn 4A# 4c# 4e	(8gn/L	2.r
.	8ee/	.
4F# 4A# 4d#	8f#/	.
.	8dd#/	.
4E 4F# 4A# 4c#	8f#/	.
.	8cc#/J	.
4D# 4F# 4B	8f#/L	2r
.	8b/	.
[4F# 4A# 4c#	8f#/	.
.	8cc#/	.
4F#] 4B 4d#)	8f#/	4b\\
.	8dd#/J)	."""

        self.assertEqual(actual, expected_content)

    def test_parsing_of_kern_file(self) -> None:
        file_content = """**kern	**kern
*part1	*part1
*staff2	*staff1
*I"Piano	*
*I'Pno.	*
*clefF4	*clefG2
*k[]	*k[]
*M4/4	*M4/4
=1	=1
2C	4c [4g 4cc 4ee [4gg
.	4r
2G	4g] 4dd 4gg]
.	4r
=2	=2
2C	4c 4g [4cc 4ee 4gg
.	4r
2F	4f 4cc] 4ff
.	4r
=3	=3
2C	4c 4g 4cc [4ee 4gg 4bb-X
.	4r
2A	4a 4ee] 4aa
.	4r
=4	=4
2C	4c 4g 4cc [4ee 4gg
.	4r
2E	4e 4b 4ee] 4gg#X
.	4r
=5	=5
2C	4c 4g 4cc 4ee [4gg
.	4r
2E-X	4e-X 4b-X 4ee-X 4gg]
.	4r
=6	=6
2C	(4c 4g 4cc 4ee 4gg 4bb-X 4eee
.	4r
2A-X	4a-X) 4ee-X 4aa-X 4ccc
.	4r
==	==
*-	*-
!!!system-decoration: {(s1,s2)}
"""
        symbols = get_symbols(file_content.splitlines())
        self.assertEqual(
            set(symbols),
            {
                "*clefF4",
                "*clefG2",
                "2G",
                "2A-X",
                "4b",
                "*M4/4",
                "4aa-X",
                "2C",
                "<TAB>",
                "4aa",
                "*k[]",
                "4e-X",
                "4a-X",
                "<NL>",
                "=",
                "4b-X",
                "2F",
                "4cc",
                "4gg#X",
                "4bb-X",
                "4ccc",
                "4ff",
                "4e",
                "4ee",
                "4g",
                "4f",
                "4a",
                "2A",
                "4ee-X",
                "4eee",
                "4c",
                "4r",
                "2E",
                "4gg",
                "4dd",
                "2E-X",
            },
        )

    def test_token_efficiency(self) -> None:
        file_content = """**kern	**kern
    *clefF4	*clefG2
    *k[f#]	*k[f#]
    *M6/8	*M6/8
    =-	=-
    32GLLL	8bL
    32d	.
    32B	.
    32dJJJ	.
    32GLLL	8.b
    32d	.
    32B	.
    32dJJJ	.
    32GLLL	.
    32d	.
    32A	16ccJk
    32dJJJ	.
    32BLLL	16eeLL
    32d	.
    32G	16ddJJ
    32dJJJ	.
    32BLLL	16r
    32d	.
    32G	16dd
    32dJJJ	.
    32BLLL	32eeLLL
    32d	32dd
    32G	32cc
    32dJJJ	32bJJJ
    =	=
    32F#LLL	16bLL
    32d	.
    32c#	16aJJ
    32dJJJ	.
    32F#LLL	8.aL
    32d	.
    32c#	.
    32dJJJ	.
    32F#LLL	.
    32d	.
    32G	16bJk
    32dJJJ	.
    32G#LLL	8ddL
    32d	.
    32c#X	.
    32dJJJ	.
    32ALLL	16ccJk
    32d	.
    32c#	32r
    32dJJJ	32cc
    32F#LLL	32ddLLL
    32d	32cc
    32D	32b
    32dJJJ	32aJJJ
    =	=
    32ELLL	16aLL
    32d	.
    32c#	16gJJ
    32dJJJ	.
    8E	32r
    .	32gLLL
    .	32f#
    .	32gJJJ
    8r	32aLLL>
    .	32g
    .	32cc
    .	32bJJJ
    32F#LLL	16g#LL
    32d	.
    32c#X	16aJJ
    32dJJJ	.
    8F#	32r
    .	32aLLL<
    .	32g#
    .	32aJJJ
    8r	32bLLL
    .	32a
    .	32dd
    .	32ccJJJ
    =	=
    8G 8dL	32ccLLL
    .	32b
    .	32a#
    .	32bJJJ
    8E 8G	32ccLLL
    .	32b
    .	32a#
    .	32bJJJ
    8C 8AJ	32ddLLL
    .	32cc
    .	32b
    .	32ccJJJ
    8D 8F#	32bLLL>
    .	32a
    .	32g#
    .	32aJJJ
    8r	32gLLL
    .	32f#
    .	32e#
    .	32f#JJJ
    8r	32eLLL
    .	32d
    .	32c#
    .	32dJJJ
    =	=
    32GGLLL	16g 16bLK<
    32G	.
    32G	8g 8bJ
    32GJJJ	.
    32GLLL	.
    32G	.
    32G	16g# 16bLL
    32GJJJ	.
    32GLLL	16a 16cc
    32G	.
    32G	16a# 16cc#JJ
    32GJJJ	.
    8G	32b 32ddLLL
    .	32g
    .	32a
    .	32bJJJ
    8r	32ccLLL
    .	32dd
    .	32ee
    .	32ff#JJJ
    8r	32ggLLL
    .	32aa
    .	32bbJJJ
    .	32r
    =	=
    32DDLLL	32g 32bLLL
    32D	32f# 32aJJ
    32D	8f# 8aJ
    32DJJJ	.
    32DLLL	.
    32D	.
    32D	16f# 16aLL<
    32DJJJ	.
    32DLLL	16g 16b-
    32D	.
    32D	16g# 16bJJ
    32DJJJ	.
    8D	32a 32ccLLL
    .	32a
    .	32b
    .	32ccJJJ
    8r	32ddLLL
    .	32ee
    .	32ff#
    .	32ggJJJ
    8r	32aaLLL
    .	32bb
    .	32cccJJJ
    .	32r
    =	=
    *-	*-
    """
        symbols = get_symbols(file_content.splitlines())
        self.assertEqual(len(symbols), 501)

    def test_key_change(self) -> None:
        file_content = """**kern	**kern
*clefF4	*clefG2
*k[]	*k[]
*M3/8	*M3/8
=-	=-
4.BB 4.G 4.B	8gL
.	8g
.	8gggJ[
*clefG2	*
=	=
16ddLL	8ff# 8aaL
16dJJ	.
*clefF4	*
16gLL	8gg 8bb
    """
        symbols = get_symbols(file_content.splitlines())
        self.assertEqual(symbols.count("*clefF4"), 2)
        self.assertEqual(symbols.count("*clefG2"), 2)

    def test_multiple_voices_for_g_staff(self) -> None:
        file_content = """**kern	**kern
*	*^
*clefF4	*clefG2	*clefG2
*k[b-e-a-d-g-]	*k[b-e-a-d-g-]	*k[b-e-a-d-g-]
*M3/4	*M3/4	*M3/4
=-	=-	=-
4AA- 4E- 4A-	4ee-	8ccL
.	.	16b-L
.	.	16ccJJ
4GG 4E-	4ee-	8dd-L
.	.	16ccL
.	.	16b-JJ
4AA- 4E-	4ee-	8ccL
.	.	16b-L
.	.	16a-JJ
=	=	=
4EE- 4E-	4ee-	8gL
.	.	16fL
.	.	16e-JJ
4C 4E- 4A-	4ee-	8a-L
.	.	16gL
.	.	16a-JJ
4BB- 4D- 4G	4ee-	8b-L
.	.	16a-L
.	.	16b-JJ
    """
        symbols = get_symbols(file_content.splitlines())
        self.assertEqual(symbols.count("*clefF4"), 1)
        self.assertEqual(symbols.count("*clefG2"), 1)
        self._assert_no_multiple_tabs_per_line(symbols)

    def test_multiple_voices_for_f_staff(self) -> None:
        file_content = """**kern	**kern
*^	*
*clefF4	*clefF4	*clefG2
*k[b-e-a-d-]	*k[b-e-a-d-]	*k[b-e-a-d-]
*M3/8	*M3/8	*M3/8
=-	=-	=-
8FFL	4.FF	4ee
8C	.	.
8FJ	.	8ff
    """
        symbols = get_symbols(file_content.splitlines())
        self.assertEqual(symbols.count("*clefF4"), 1)
        self.assertEqual(symbols.count("*clefG2"), 1)
        self._assert_no_multiple_tabs_per_line(symbols)

    def test_multiple_voices_for_each_staff(self) -> None:
        file_content = """**kern	**kern
*^	*^
*clefF4	*clefF4	*clefG2	*clefG2
*k[]	*k[]	*k[]	*k[]
*M3/4	*M3/4	*M3/4	*M3/4
=-	=-	=-	=-
2A#	8.C#L	8.a#L	2.e#
.	16BB#Jk	16a#Jk	.
.	4C#	4f##	.
4r	4r	4ff##	.
*v	*v	*	*
*	*v	*v
=	=
2.r	8ee#L>
.	8cc#J
.	4a#
.	4f##
=	=
2.r	8e#<L>
.	8c#<J
.	4A#<
.	4E#<
=	=
2.EE#< 2.BB#<	2.G#<
==	==
*-	*-
    """
        symbols = get_symbols(file_content.splitlines())
        self.assertEqual(symbols.count("*clefF4"), 1)
        self.assertEqual(symbols.count("*clefG2"), 1)
        self._assert_no_multiple_tabs_per_line(symbols)
        note_symbols = 0
        for symbol in symbols:
            note, rhythm, pitch, lift = split_symbol_into_token(symbol)
            rhythm_token = default_config.rhythm_vocab[rhythm]
            is_note_rhythm = rhythm_token in default_config.noteindexes
            self.assertEqual(note == "nonote", not is_note_rhythm)
            self.assertEqual(note == "nonote", pitch == "nonote")
            self.assertEqual(note == "nonote", lift == "nonote")
            if is_note_rhythm:
                note_symbols += 1
        # Around half of the symbols should be classified as note so that the consist loss is meaningful
        self.assertTrue(abs(100 * note_symbols / len(symbols) - 50) < 10)

    def test_sort_pitches(self) -> None:
        symbols = get_symbols(["4c	2c 2r 2cc 4gg"])
        self.assertEqual(symbols, ["4c", "<TAB>", "2c", "2r", "2cc", "4gg", "<NL>"])

    def test_split_merge_note(self) -> None:
        tokens = split_symbol_into_token("4CC#")
        self.assertEqual(tokens, ("note", "4", "CC", "#"))
        symbol = merge_kern_tokens(tokens[1], tokens[2], tokens[3])
        self.assertEqual(symbol, "4CC#")

    def test_split_merge_rest(self) -> None:
        tokens = split_symbol_into_token("4r")
        self.assertEqual(tokens, ("note", "4", "r", ""))
        symbol = merge_kern_tokens(tokens[1], tokens[2], tokens[3])
        self.assertEqual(symbol, "4r")
        self.assertEqual(symbol, "4cc#")

    def test_split_merge_tab(self) -> None:
        tokens = split_symbol_into_token("<TAB>")
        self.assertEqual(tokens, ("nonote", "<TAB>", "nonote", "nonote"))
        symbol = merge_kern_tokens(tokens[1], tokens[2], tokens[3])
        self.assertEqual(symbol, "<TAB>")

    def test_split_merge_new_line(self) -> None:
        tokens = split_symbol_into_token("<NL>")
        self.assertEqual(tokens, ("nonote", "<NL>", "nonote", "nonote"))
        symbol = merge_kern_tokens(tokens[1], tokens[2], tokens[3])
        self.assertEqual(symbol, "<NL>")

    def test_split_merge_clef(self) -> None:
        tokens = split_symbol_into_token("*clefF4")
        self.assertEqual(tokens, ("note", "*clef", "*clefF4", "*clefF4"))
        symbol = merge_kern_tokens(tokens[1], tokens[2], tokens[3])
        self.assertEqual(symbol, "*clefF4")

    def test_split_merge_key(self) -> None:
        tokens = split_symbol_into_token("*k[]")
        self.assertEqual(tokens, ("note", "*k", "*symbol", "*k[]"))
        symbol = merge_kern_tokens(tokens[1], tokens[2], tokens[3])
        self.assertEqual(symbol, "*k[]")

    def test_split_merge_barline(self) -> None:
        tokens = split_symbol_into_token("=")
        self.assertEqual(tokens, ("note", "=", "*symbol", "="))
        symbol = merge_kern_tokens(tokens[1], tokens[2], tokens[3])
        self.assertEqual(symbol, "=")

    def _assert_no_multiple_tabs_per_line(self, symbols: list[str]) -> None:
        number_of_tabs = 0
        for symbol in symbols:
            if symbol == "<TAB>":
                number_of_tabs += 1
                self.assertTrue(number_of_tabs <= 1)
            elif symbol == "<NL>":
                number_of_tabs = 0
