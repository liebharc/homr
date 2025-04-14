import unittest

from training.transformer.kern_tokens import filter_for_kern, get_symbols


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
