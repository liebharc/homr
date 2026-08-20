# ruff: noqa: E501, S101

import unittest
import xml.etree.ElementTree as ET

from homr.music_xml_generator import (
    SymbolChord,
    XmlGeneratorArguments,
    convert_ties,
    generate_xml,
    rebalance_measure_voices,
)
from homr.transformer.vocabulary import EncodedSymbol
from training.transformer.training_vocabulary import (
    read_token_lines,
)


def _notes(measure: ET.Element) -> list[ET.Element]:
    return [c for c in measure if c.tag == "note"]


def _pitch(note: ET.Element) -> str:
    p = note.find("pitch")
    if p is None:
        return "rest"
    step = p.findtext("step", "")
    return step


def _duration(note: ET.Element) -> int:
    d = note.findtext("duration")
    return int(d) if d is not None else 0


def _voice(note: ET.Element) -> str:
    return note.findtext("voice", "")


def _staff(note: ET.Element) -> str:
    return note.findtext("staff", "")


def _backups(measure: ET.Element) -> list[int]:
    return [int(c.findtext("duration", "0")) for c in measure if c.tag == "backup"]


def _ties(xml: ET.Element) -> list[str]:
    return [t.get("type", "") for t in xml.iter("tie")]


def _tieds(xml: ET.Element) -> list[str]:
    return [t.get("type", "") for t in xml.iter("tied")]


def _slurs(xml: ET.Element) -> list[str]:
    return [s.get("type", "") for s in xml.iter("slur")]


def _first_measure(xml: ET.Element) -> ET.Element:
    part = xml.find("part")
    assert part is not None
    m = part.find("measure")
    assert m is not None
    return m


class TestMusicXmlGenerator(unittest.TestCase):
    """
    MusicXML testing is mostly covered by training/validate_music_xml_conversion.py
    This script requires that the data sets are downloaded and converted and uses
    the data sets to check that back and forth conversion works.
    """

    def test_chord_with_different_duratons(self) -> None:
        tabi_measure_18_upper = """clef_G2 . . . . upper
keySignature_4 . . . . .
timeSignature/8 . . . . .
note_4. G3 # _ _ upper &note_4. C4 # _ _ upper&note_16 E4 # _ _ upper
note_16 F4 # _ _ upper
note_4 E4 # _ _ upper
note_8 E4 # _ _ upper
note_8 C4 # _ _ upper
note_8 D4 # _ _ upper
barline . . . . ."""
        tokens = read_token_lines(tabi_measure_18_upper.splitlines())
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "")
        measure = _first_measure(xml)
        notes = _notes(measure)
        backups = _backups(measure)

        # Pitches in order after rebalancing
        pitches = [_pitch(n) for n in notes]
        self.assertIn("E", pitches)
        self.assertIn("G", pitches)
        self.assertIn("F", pitches)
        self.assertIn("D", pitches)

        # There must be backups due to chord with different durations
        self.assertGreater(len(backups), 0)

        # All notes have a voice and staff assigned
        for note in notes:
            self.assertNotEqual(_voice(note), "")
            self.assertEqual(_staff(note), "1")

    def test_grand_staff_generation(self) -> None:
        grandstaff = """clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
keySignature_1 . . . . .
timeSignature/4 . . . . .
note_1 G4 _ _ _ upper&note_1 A3 # _ _ upper&rest_2 _ _ _ _ upper&note_4 G3 _ _ _ lower
rest_4 _ _ _ _ lower
note_2 E4 _ _ _ upper&note_2 C2 _ _ _ lower
barline . . . . ."""
        tokens = read_token_lines(grandstaff.splitlines())
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "")
        measure = _first_measure(xml)
        notes = _notes(measure)

        # Both staves must be present
        staves = {_staff(n) for n in notes}
        self.assertIn("1", staves)
        self.assertIn("2", staves)

        # Upper staff notes: G4, A3, rest, E4; lower: G3, rest, C2
        pitches_upper = [_pitch(n) for n in notes if _staff(n) == "1"]
        pitches_lower = [_pitch(n) for n in notes if _staff(n) == "2"]
        self.assertIn("G", pitches_upper)
        self.assertIn("E", pitches_upper)
        self.assertIn("G", pitches_lower)
        self.assertIn("C", pitches_lower)

        # Upper voices are 1-4, lower voices are 5-8
        for note in notes:
            v = int(_voice(note))
            s = int(_staff(note))
            if s == 1:
                self.assertLessEqual(v, 4)
            else:
                self.assertGreaterEqual(v, 5)

    def test_two_voices_on_the_same_staff(self) -> None:
        two_voices = """clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
keySignature_0 . . . . .
timeSignature/4 . . . . .
note_2 G4 _ _ _ upper&note_4 E4 _ _ _ upper2&note_1 C3 _ _ _ lower
note_4 D4 _ _ _ upper2
barline . . . . ."""
        tokens = read_token_lines(two_voices.splitlines())
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "")
        by_pitch = {_pitch(n): n for n in _notes(_first_measure(xml))}

        # The second voice belongs to the upper staff, but is a voice of its own
        # instead of being a chord note of the first voice.
        self.assertEqual(_staff(by_pitch["E"]), "1")
        self.assertIsNone(by_pitch["E"].find("chord"))
        self.assertNotEqual(_voice(by_pitch["E"]), _voice(by_pitch["G"]))
        self.assertEqual(_voice(by_pitch["E"]), _voice(by_pitch["D"]))

    def test_begin_chord_with_standalone_rests(self) -> None:
        """
        If the lower position consists of a standalone rest then start the
        chord with this. That fixes an issue where the upper position
        consists of tuplets because in that case backups must not be used.

        See tabi.jpg measure 9 for an example.
        """
        chord = SymbolChord(
            [
                EncodedSymbol("note_12", position="upper"),
                EncodedSymbol("note_12", position="upper"),
                EncodedSymbol("rest_8", position="lower"),
            ]
        )
        first, second = chord.into_positions()

        self.assertEqual(first.symbols, [EncodedSymbol("rest_8", position="lower")])
        self.assertEqual(
            second.symbols,
            [
                EncodedSymbol("note_12", position="upper"),
                EncodedSymbol("note_12", position="upper"),
            ],
        )

    def test_rebalance_measure_voices_assigns_stable_voices_per_staff(self) -> None:
        measure = ET.Element("measure")

        note1 = self._build_test_note(duration=4, staff=1, voice=1)
        measure.append(note1)
        measure.append(self._build_test_backup(duration=4))

        note2 = self._build_test_note(duration=2, staff=1, voice=1)
        measure.append(note2)

        note3 = self._build_test_note(duration=2, staff=1, voice=1)
        measure.append(note3)

        note4 = self._build_test_note(duration=2, staff=1, voice=1, is_chord=True)
        measure.append(note4)

        measure.append(self._build_test_backup(duration=4))
        note5 = self._build_test_note(duration=4, staff=2, voice=1)
        measure.append(note5)
        measure.append(self._build_test_backup(duration=4))

        note6 = self._build_test_note(duration=2, staff=2, voice=1)
        measure.append(note6)

        rebalance_measure_voices(measure)

        self.assertEqual(self._read_note_voice(note1), "2")
        self.assertEqual(self._read_note_voice(note2), "1")
        self.assertEqual(self._read_note_voice(note3), "1")
        self.assertEqual(self._read_note_voice(note4), "1")
        self.assertEqual(self._read_note_voice(note5), "6")
        self.assertEqual(self._read_note_voice(note6), "5")

    def _build_test_note(
        self, duration: int, staff: int, voice: int, is_chord: bool = False
    ) -> ET.Element:
        note = ET.Element("note")
        if is_chord:
            ET.SubElement(note, "chord")
        ET.SubElement(note, "duration").text = str(duration)
        ET.SubElement(note, "staff").text = str(staff)
        ET.SubElement(note, "voice").text = str(voice)
        return note

    def _build_test_backup(self, duration: int) -> ET.Element:
        backup = ET.Element("backup")
        ET.SubElement(backup, "duration").text = str(duration)
        return backup

    def _read_note_voice(self, note: ET.Element) -> str:
        v = note.findtext("voice")
        self.assertIsNotNone(v)
        return str(v)

    def test_slur_between_same_pitches_becomes_a_tie(self) -> None:
        """A slur joining two identical pitches is a tie, and needs both elements."""
        tied = """clef_G2 . . . . upper
timeSignature/4 . . . . .
note_4 C4 _ _ slurStart upper
note_4 C4 _ _ slurStop upper
note_2 D4 _ _ _ upper
barline . . . . ."""
        tokens = read_token_lines(tied.splitlines())
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "")

        # <tie> is what sounds, <tied> is what is drawn: both are required, or
        # the notes are drawn joined and still played separately.
        self.assertEqual(_ties(xml), ["start", "stop"])
        self.assertEqual(_tieds(xml), ["start", "stop"])
        # the slur it came from is gone
        self.assertEqual(_slurs(xml), [])

    def test_tie_between_two_chords(self) -> None:
        """Every notehead of a chord ties to its own pitch in the next one."""
        chords = """clef_G2 . . . . upper
timeSignature/4 . . . . .
note_4 C4 _ _ slurStart upper&note_4 E4 _ _ slurStart upper
note_4 C4 _ _ slurStop upper&note_4 E4 _ _ slurStop upper
barline . . . . ."""
        tokens = read_token_lines(chords.splitlines())
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "")

        self.assertEqual(sorted(_ties(xml)), ["start", "start", "stop", "stop"])
        self.assertEqual(_slurs(xml), [])

    def test_tie_on_one_notehead_of_a_chord(self) -> None:
        """The model marks the notehead the curve touches, not the whole chord.

        On real output almost every slurred chord carries the slur on some of
        its members only, so a tie has to be found for that pitch alone and
        leave the rest of the chord as it is.
        """
        chords = """clef_G2 . . . . upper
timeSignature/4 . . . . .
note_4 C4 _ _ _ upper&note_4 E4 _ _ slurStart upper
note_4 C4 _ _ _ upper&note_4 E4 _ _ slurStop upper
barline . . . . ."""
        tokens = read_token_lines(chords.splitlines())
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "")

        self.assertEqual(_ties(xml), ["start", "stop"])
        self.assertEqual(_slurs(xml), [])
        tied = [n for n in xml.iter("note") if n.find("tie") is not None]
        self.assertEqual([_pitch(n) for n in tied], ["E", "E"])

    def test_slur_from_a_chord_to_a_different_pitch_stays_a_slur(self) -> None:
        chords = """clef_G2 . . . . upper
timeSignature/4 . . . . .
note_4 C4 _ _ _ upper&note_4 E4 _ _ slurStart upper
note_4 C4 _ _ _ upper&note_4 G4 _ _ slurStop upper
barline . . . . ."""
        tokens = read_token_lines(chords.splitlines())
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "")

        self.assertEqual(_slurs(xml), ["start", "stop"])
        self.assertEqual(_ties(xml), [])

    def test_phrase_slur_over_three_events_stays_a_slur(self) -> None:
        """Adjacency is what separates a tie from a phrase mark.

        A curve that leaves a pitch and comes back to it later is a phrase,
        however identical its two ends are, so only the immediately following
        event may close a tie.
        """
        phrase = """clef_G2 . . . . upper
timeSignature/4 . . . . .
note_4 C4 _ _ _ upper&note_4 G4 _ _ slurStart upper
note_4 A4 _ _ _ upper
note_4 C4 _ _ _ upper&note_4 G4 _ _ slurStop upper
barline . . . . ."""
        tokens = read_token_lines(phrase.splitlines())
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "")

        self.assertEqual(_slurs(xml), ["start", "stop"])
        self.assertEqual(_ties(xml), [])

    def test_shared_pitch_without_a_slur_of_its_own_stays_untied(self) -> None:
        """A pitch in both chords is not enough; the curve has to be on it.

        Here C4 is in both chords and the curve runs from G4 to C4, so nothing
        ties: G4 has no stop to reach, and C4 has no start behind it.
        """
        crossing = """clef_G2 . . . . upper
timeSignature/4 . . . . .
note_4 C4 _ _ _ upper&note_4 G4 _ _ slurStart upper
note_4 C4 _ _ slurStop upper&note_4 A4 _ _ _ upper
barline . . . . ."""
        tokens = read_token_lines(crossing.splitlines())
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "")

        self.assertEqual(_slurs(xml), ["start", "stop"])
        self.assertEqual(_ties(xml), [])

    def test_slur_between_different_pitches_stays_a_slur(self) -> None:
        phrase = """clef_G2 . . . . upper
timeSignature/4 . . . . .
note_4 C4 _ _ slurStart upper
note_4 E4 _ _ slurStop upper
note_2 D4 _ _ _ upper
barline . . . . ."""
        tokens = read_token_lines(phrase.splitlines())
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "")

        self.assertEqual(_slurs(xml), ["start", "stop"])
        self.assertEqual(_ties(xml), [])

    def test_tie_is_recognised_across_a_barline(self) -> None:
        """Ties cross barlines constantly, which is why the pass runs per part."""
        across = """clef_G2 . . . . upper
timeSignature/4 . . . . .
note_1 C4 _ _ slurStart upper
barline . . . . .
note_1 C4 _ _ slurStop upper
barline . . . . ."""
        tokens = read_token_lines(across.splitlines())
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "")

        self.assertEqual(_ties(xml), ["start", "stop"])
        self.assertEqual(_slurs(xml), [])

    def test_same_pitch_but_not_adjacent_stays_a_slur(self) -> None:
        """Same pitch is not enough: a tie joins a note to its immediate successor."""
        apart = """clef_G2 . . . . upper
timeSignature/4 . . . . .
note_4 C4 _ _ slurStart upper
note_4 E4 _ _ _ upper
note_4 C4 _ _ slurStop upper
note_4 D4 _ _ _ upper
barline . . . . ."""
        tokens = read_token_lines(apart.splitlines())
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "")

        self.assertEqual(_ties(xml), [])
        self.assertEqual(_slurs(xml), ["start", "stop"])

    def test_convert_ties_leaves_a_part_without_slurs_alone(self) -> None:
        part = ET.Element("part", id="P1")
        measure = ET.SubElement(part, "measure", number="1")
        note = ET.SubElement(measure, "note")
        pitch = ET.SubElement(note, "pitch")
        ET.SubElement(pitch, "step").text = "C"
        ET.SubElement(pitch, "octave").text = "4"
        ET.SubElement(note, "duration").text = "4"
        ET.SubElement(note, "voice").text = "1"
        ET.SubElement(note, "staff").text = "1"

        convert_ties(part)
        self.assertEqual(_ties(part), [])

    def test_tie_found_while_another_slur_is_open_on_the_same_staff(self) -> None:
        """Several slurs share a staff, and so share a slur number.

        The tie here opens and closes inside a longer slur. Nothing can tell
        the two apart by number, which is why ties are detected from a note
        and its successor rather than by pairing starts with stops.
        """
        nested = """clef_G2 . . . . upper
timeSignature/4 . . . . .
note_4 E4 _ _ slurStart upper
note_4 C4 _ _ slurStart upper
note_4 C4 _ _ slurStop upper
note_4 G4 _ _ slurStop upper
barline . . . . ."""
        tokens = read_token_lines(nested.splitlines())
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "")

        # the inner pair became a tie
        self.assertEqual(_ties(xml), ["start", "stop"])
        self.assertEqual(_tieds(xml), ["start", "stop"])
        # the outer slur is untouched
        self.assertEqual(_slurs(xml), ["start", "stop"])
