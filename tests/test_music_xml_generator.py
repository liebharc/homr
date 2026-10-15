# ruff: noqa: E501, S101

import unittest
import xml.etree.ElementTree as ET

from homr.music_xml_generator import (
    SymbolChord,
    XmlGeneratorArguments,
    _split_mixed_chord,
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


def _times(xml: ET.Element) -> list[tuple[str, str]]:
    return [(t.findtext("beats", ""), t.findtext("beat-type", "")) for t in xml.iter("time")]


def _first_measure(xml: ET.Element) -> ET.Element:
    part = xml.find("part")
    assert part is not None
    m = part.find("measure")
    assert m is not None
    return m


def _onsets(xml: ET.Element) -> list[tuple[str, str, float]]:
    """(staff, pitch with octave, onset in quarter notes) for every note of the first measure."""
    part = xml.find("part")
    assert part is not None
    divisions = int(part.findtext("measure/attributes/divisions", "1"))
    time, onset, result = 0, 0, []
    for element in _first_measure(xml):
        if element.tag == "backup":
            time -= int(element.findtext("duration", "0"))
        elif element.tag == "forward":
            time += int(element.findtext("duration", "0"))
        elif element.tag == "note":
            if element.find("chord") is None:
                onset = time
                time += _duration(element)
            pitch = element.find("pitch")
            if pitch is not None:
                name = pitch.findtext("step", "") + pitch.findtext("octave", "")
                result.append((_staff(element), name, onset / divisions))
    return sorted(result)


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

    def test_recognized_time_signature_is_the_only_one(self) -> None:
        """
        The clef opens a second attributes element in the first measure. The computed
        fallback time signature must not be added next to a recognized one (issue #161).
        """
        six_eight = """clef_G2 . . . . upper
keySignature_0 . . . . .
timeSignature/8 . . . . .
note_4. C4 _ _ _ upper
note_4. D4 _ _ _ upper
barline . . . . .
note_4. C4 _ _ _ upper
note_4. D4 _ _ _ upper
barline . . . . ."""
        tokens = read_token_lines(six_eight.splitlines())
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "")

        self.assertEqual(_times(xml), [("6", "8")])

    def test_time_signature_is_computed_if_none_was_recognized(self) -> None:
        no_time_signature = """clef_G2 . . . . upper
keySignature_0 . . . . .
note_4. C4 _ _ _ upper
note_4. D4 _ _ _ upper
barline . . . . .
note_4. C4 _ _ _ upper
note_4. D4 _ _ _ upper
barline . . . . ."""
        tokens = read_token_lines(no_time_signature.splitlines())
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "")

        self.assertEqual(_times(xml), [("3", "4")])

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

    def test_next_group_starts_when_the_earliest_sounding_note_ends(self) -> None:
        """
        Tokens tell which notes start together, not when each group starts. Here the left
        hand's quarter note E3 starts under the right hand's quarter note C5, so the right
        hand's D5 starts when C5 ends (beat 1), while E3 is still sounding. Advancing by the
        shortest note of the group just written would put D5 at beat 1.5 instead.
        """
        hands_moving_independently = """clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
keySignature_0 . . . . .
timeSignature/4 . . . . .
note_4 C5 _ _ _ upper&note_8 C3 _ _ _ lower
note_4 E3 _ _ _ lower
note_4 D5 _ _ _ upper
note_8 G3 _ _ _ lower
note_2 E5 _ _ _ upper&note_2 C3 _ _ _ lower
barline . . . . ."""
        tokens = read_token_lines(hands_moving_independently.splitlines())
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "")
        self.assertEqual(
            _onsets(xml),
            sorted(
                [
                    ("1", "C5", 0.0),
                    ("1", "D5", 1.0),
                    ("1", "E5", 2.0),
                    ("2", "C3", 0.0),
                    ("2", "E3", 0.5),
                    ("2", "G3", 1.5),
                    ("2", "C3", 2.0),
                ]
            ),
        )

    def test_independent_hands_in_elijah_measure(self) -> None:
        """
        Measure 6 of the Elijah page from issue #142, as the transformer reads it: the
        right hand plays a quarter chord and dotted rhythms while the left hand holds A2
        and moves in eighths and quarters. Every note must start on its beat, so that
        no staff runs past the end of the 4/4 measure.
        """
        elijah_measure_6 = """clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
keySignature_-4 . . . . .
timeSignature/4 . . . . .
note_4 A4 b _ _ upper&note_4 E4 b _ _ upper&note_4 C4 _ _ _ upper&note_2 A2 b _ _ lower2&note_8 A2 b _ _ lower
note_4 A3 b _ _ lower
note_8. A4 b _ _ upper&note_8. E4 b _ _ upper&note_8. C4 _ _ _ upper
note_8 A3 b _ _ lower
note_16 C5 _ _ _ upper&note_16 A4 b _ _ upper&note_16 E4 b _ _ upper
note_8. C5 _ _ _ upper&note_8. A4 b _ _ upper&note_8. G4 b _ _ upper&note_8. E4 b _ _ upper&note_8 A2 b _ _ lower
note_4 C4 _ _ _ lower&note_4 A3 b _ _ lower
note_16 E5 b _ _ upper&note_16 A4 b _ _ upper&note_16 G4 b _ _ upper
note_8. E5 b _ _ upper&note_8. A4 b _ _ upper&note_8. G4 b _ _ upper
note_8 E4 b _ _ lower&note_8 C4 _ _ _ lower&note_8 A3 b _ _ lower
note_16 E5 b _ _ upper&note_16 A4 b _ _ upper&note_16 G4 b _ _ upper
barline . . . . ."""
        tokens = read_token_lines(elijah_measure_6.splitlines())
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "")
        onsets = _onsets(xml)
        upper = sorted({onset for staff, _, onset in onsets if staff == "1"})
        lower = sorted({onset for staff, _, onset in onsets if staff == "2"})
        self.assertEqual(upper, [0.0, 1.0, 1.75, 2.0, 2.75, 3.0, 3.75])
        self.assertEqual(lower, [0.0, 0.5, 1.5, 2.0, 2.5, 3.5])

    def test_grace_notes_take_no_time(self) -> None:
        """A group of only grace notes must not move later notes (the next note stays on beat 1)."""
        with_grace_note = """clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
keySignature_0 . . . . .
timeSignature/4 . . . . .
note_4 C5 _ _ _ upper&note_2 C3 _ _ _ lower
note_8G D5 _ _ _ upper
note_4 E5 _ _ _ upper
note_2 F5 _ _ _ upper&note_2 G2 _ _ _ lower
barline . . . . ."""
        tokens = read_token_lines(with_grace_note.splitlines())
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "")
        by_pitch = {name: onset for _, name, onset in _onsets(xml)}
        self.assertEqual(by_pitch["E5"], 1.0)
        self.assertEqual(by_pitch["F5"], 2.0)
        self.assertEqual(by_pitch["G2"], 2.0)

    def test_grace_note_without_pitch_is_skipped(self) -> None:
        tokens = read_token_lines("""clef_G2 _ _ _ _ upper
timeSignature/4 . . . . .
note_8G . _ _ _ upper
note_2 C5 _ _ _ upper
barline . . . . .""".splitlines())
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "")
        self.assertEqual([name for _, name, _ in _onsets(xml)], ["C5"])

    def test_measure_rest_of_a_hidden_grand_staff(self) -> None:
        tokens = read_token_lines("""clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
timeSignature/4 . . . . .
note_2 C5 _ _ _ upper&note_2 C3 _ _ _ lower
barline . . . . .""".splitlines())
        tokens += [
            EncodedSymbol("newline"),
            EncodedSymbol("measureRest", position="upper"),
            EncodedSymbol("chord"),
            EncodedSymbol("measureRest", position="lower"),
            EncodedSymbol("barline"),
        ]
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "")

        measures = xml.findall("part/measure")
        self.assertEqual(2, len(measures))
        rests = _notes(measures[1])
        self.assertEqual(["1", "2"], [n.findtext("staff") for n in rests])
        self.assertEqual("yes", rests[0].find("rest").get("measure"))  # type: ignore[union-attr]
        self.assertEqual(_duration(_notes(measures[0])[0]), _duration(rests[0]))
        self.assertEqual(["1", "5"], [_voice(n) for n in rests])

    def test_image_position_is_written_as_comment(self) -> None:
        tokens = read_token_lines("""clef_G2 . . . . upper
note_4 E4 _ _ _ upper
note_4 C4 _ _ _ upper
barline . . . . .""".splitlines())
        tokens[1].image_coordinates = (45.4, 230.6)
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "")

        notes = list(xml.iter("note"))
        comments = [[c.text for c in n if c.tag is ET.Comment] for n in notes]  # type: ignore[comparison-overlap]
        self.assertEqual(comments, [[" imgpos: 45, 231 "], []])
        self.assertIn("<!-- imgpos: 45, 231 -->", ET.tostring(xml, encoding="unicode"))

    def test_clef_change_joined_to_a_chord_of_notes(self) -> None:
        """
        The transformer sometimes joins a clef change in one staff to a chord of notes in the
        other (a real clef change at that beat). The clef must be written, before the notes
        that follow it, and must not be treated as a rest (that used to fail an assertion
        and lose the whole page).
        """
        clef_in_chord = """clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
keySignature_0 . . . . .
timeSignature/4 . . . . .
note_4 C5 _ _ _ upper&note_4 C3 _ _ _ lower
note_4 A4 _ _ _ upper&clef_G2 _ _ _ _ lower
note_4 F4 _ _ _ lower
note_2 G4 _ _ _ upper&note_2 E4 _ _ _ lower
barline . . . . ."""
        tokens = read_token_lines(clef_in_chord.splitlines())
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "")
        measure = _first_measure(xml)
        pitches = [name for _, name, _ in _onsets(xml)]
        self.assertEqual(sorted(pitches), ["A4", "C3", "C5", "E4", "F4", "G4"])
        order = [
            ("clef", c.get("number"), c.findtext("sign"))
            for element in measure
            for c in ([element] if element.tag == "note" else element.iter("clef"))
        ]
        lower_clefs = [entry for entry in order if entry[0] == "clef" and entry[1] == "2"]
        self.assertEqual([sign for _, _, sign in lower_clefs], ["F", "G"])
        notes_and_clefs = [
            "clef" if element.tag != "note" else _pitch(element) + _staff(element)
            for element in measure
            if element.tag == "note" or element.find("clef") is not None
        ]
        self.assertLess(notes_and_clefs.index("clef", 1), notes_and_clefs.index("F2"))

    def test_barline_joined_to_a_chord_of_notes(self) -> None:
        """A barline the transformer joined to the last chord of a measure still ends the measure."""
        barline_in_chord = """clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
keySignature_0 . . . . .
timeSignature/4 . . . . .
note_4 C5 _ _ _ upper&note_4 C3 _ _ _ lower
note_4 D5 _ _ _ upper&note_4 D3 _ _ _ lower&barline _ _ _ _ .
note_2 E5 _ _ _ upper&note_2 E3 _ _ _ lower
barline . . . . ."""
        tokens = read_token_lines(barline_in_chord.splitlines())
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "")
        measures = xml.findall("part/measure")
        self.assertEqual(len(measures), 2)
        self.assertEqual(sorted(_pitch(n) for n in _notes(measures[0])), ["C", "C", "D", "D"])
        self.assertEqual(sorted(_pitch(n) for n in _notes(measures[1])), ["E", "E"])

    def test_clef_with_a_pitch_is_not_written_as_a_note(self) -> None:
        """A clef token that came with a pitch, joined to a note, must not become an extra note."""
        clef_with_pitch = """clef_G2 _ _ _ _ upper&clef_G2 _ _ _ _ lower
keySignature_0 . . . . .
timeSignature/4 . . . . .
note_2 F4 _ _ _ upper&clef_F4 F5 _ _ _ lower
note_2 G4 _ _ _ upper&note_2 C3 _ _ _ lower
barline . . . . ."""
        tokens = read_token_lines(clef_with_pitch.splitlines())
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "")
        pitches = sorted(name for _, name, _ in _onsets(xml))
        self.assertEqual(pitches, ["C3", "F4", "G4"])

    def test_barline_joined_to_a_chord_and_repeated_after_it_counts_once(self) -> None:
        """A barline joined to a chord and then written again on its own ends only one measure."""
        barline_twice = """clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
keySignature_0 . . . . .
timeSignature/4 . . . . .
note_4 C5 _ _ _ upper&note_4 C3 _ _ _ lower
note_4 D5 _ _ _ upper&note_4 D3 _ _ _ lower&barline F5 _ _ _ upper
barline . . . . .
note_2 E5 _ _ _ upper&note_2 E3 _ _ _ lower
barline . . . . ."""
        tokens = read_token_lines(barline_twice.splitlines())
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "")
        measures = xml.findall("part/measure")
        self.assertEqual(len(measures), 2)
        self.assertEqual(sorted(_pitch(n) for n in _notes(measures[0])), ["C", "C", "D", "D"])

    def test_mixed_chord_is_split_into_before_notes_after(self) -> None:
        """
        Clefs, keys and time signatures go before the notes of the chord, in the order
        they have at the start of a line, barlines after.
        """
        chord = [
            EncodedSymbol("barline"),
            EncodedSymbol("timeSignature/4"),
            EncodedSymbol("note_4", "C4", position="upper"),
            EncodedSymbol("keySignature_1"),
            EncodedSymbol("clef_G2", position="lower"),
            EncodedSymbol("rest_4", position="lower"),
        ]
        parts = _split_mixed_chord(chord)
        rhythms = [[s.rhythm for s in part] for part in parts]
        self.assertEqual(
            rhythms,
            [
                ["clef_G2"],
                ["keySignature_1"],
                ["timeSignature/4"],
                ["note_4", "rest_4"],
                ["barline"],
            ],
        )

    def test_chords_without_notes_or_only_notes_are_not_split(self) -> None:
        clefs = [
            EncodedSymbol("clef_G2", position="upper"),
            EncodedSymbol("clef_F4", position="lower"),
        ]
        notes = [
            EncodedSymbol("note_4", "C4", position="upper"),
            EncodedSymbol("rest_4", position="lower"),
        ]
        self.assertEqual(_split_mixed_chord(clefs), [clefs])
        self.assertEqual(_split_mixed_chord(notes), [notes])

    def test_key_signature_joined_to_a_chord_of_notes(self) -> None:
        key_in_chord = """clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
keySignature_0 . . . . .
timeSignature/4 . . . . .
note_2 C5 _ _ _ upper&note_2 C3 _ _ _ lower
barline . . . . .
note_2 D5 _ _ _ upper&note_2 D3 _ _ _ lower&keySignature_2 . . . . .
barline . . . . ."""
        tokens = read_token_lines(key_in_chord.splitlines())
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "")
        measures = xml.findall("part/measure")
        self.assertEqual(len(measures), 2)
        self.assertEqual(measures[1].findtext("attributes/key/fifths"), "2")
        self.assertEqual(sorted(_pitch(n) for n in _notes(measures[1])), ["D", "D"])

    def test_time_signature_joined_to_a_chord_of_notes(self) -> None:
        time_in_chord = """clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
keySignature_0 . . . . .
note_4 C5 _ _ _ upper&note_4 C3 _ _ _ lower&timeSignature/4 . . . . .
note_4 D5 _ _ _ upper&note_4 D3 _ _ _ lower
barline . . . . ."""
        tokens = read_token_lines(time_in_chord.splitlines())
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "")
        self.assertIsNotNone(_first_measure(xml).find("attributes/time"))
        self.assertEqual(sorted(name for _, name, _ in _onsets(xml)), ["C3", "C5", "D3", "D5"])

    def test_clef_joined_to_a_chord_of_rests(self) -> None:
        clef_with_rests = """clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
keySignature_0 . . . . .
timeSignature/4 . . . . .
rest_4 _ _ _ _ upper&rest_4 _ _ _ _ lower&clef_G2 _ _ _ _ lower
note_4 C5 _ _ _ upper&note_4 E4 _ _ _ lower
barline . . . . ."""
        tokens = read_token_lines(clef_with_rests.splitlines())
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "")
        lower_clefs = [
            c.findtext("sign") for c in _first_measure(xml).iter("clef") if c.get("number") == "2"
        ]
        self.assertEqual(lower_clefs, ["F", "G"])
        self.assertEqual(sorted(name for _, name, _ in _onsets(xml)), ["C5", "E4"])

    def test_repeat_joined_to_a_chord_of_notes_is_kept(self) -> None:
        """
        A repeat sorts before the notes of its chord, so the chord used to be written as a
        repeat only and its notes were dropped without an error.
        """
        repeat_in_chord = """clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
keySignature_0 . . . . .
timeSignature/4 . . . . .
note_2 C5 _ _ _ upper&note_2 C3 _ _ _ lower&repeatEnd . . . . .
note_2 D5 _ _ _ upper&note_2 D3 _ _ _ lower
barline . . . . ."""
        tokens = read_token_lines(repeat_in_chord.splitlines())
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "")
        measures = xml.findall("part/measure")
        self.assertEqual(len(measures), 2)
        self.assertEqual(sorted(_pitch(n) for n in _notes(measures[0])), ["C", "C"])
        repeat = measures[0].find("barline/repeat")
        self.assertIsNotNone(repeat)
        self.assertEqual(repeat.get("direction"), "backward")  # type: ignore[union-attr]

    def test_repeat_split_off_a_chord_wins_over_a_following_plain_barline(self) -> None:
        repeat_then_barline = """clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
keySignature_0 . . . . .
timeSignature/4 . . . . .
note_2 C5 _ _ _ upper&note_2 C3 _ _ _ lower&repeatEnd . . . . .
barline . . . . .
note_2 D5 _ _ _ upper&note_2 D3 _ _ _ lower
barline . . . . ."""
        tokens = read_token_lines(repeat_then_barline.splitlines())
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "")
        measures = xml.findall("part/measure")
        self.assertEqual(len(measures), 2)
        repeat = measures[0].find("barline/repeat")
        self.assertIsNotNone(repeat)
        self.assertEqual(repeat.get("direction"), "backward")  # type: ignore[union-attr]

    def test_clef_joined_to_the_middle_of_a_triplet(self) -> None:
        """Splitting the chord must not break the triplet around it."""
        clef_in_triplet = """clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
keySignature_0 . . . . .
timeSignature/4 . . . . .
note_12 C5 _ _ _ upper&note_4 C3 _ _ _ lower
note_12 D5 _ _ _ upper&clef_G2 _ _ _ _ lower
note_12 E5 _ _ _ upper
note_4 F5 _ _ _ upper&note_4 F4 _ _ _ lower
barline . . . . ."""
        tokens = read_token_lines(clef_in_triplet.splitlines())
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "")
        triplet = [
            n for n in _notes(_first_measure(xml)) if n.find("time-modification") is not None
        ]
        self.assertEqual([_pitch(n) for n in triplet], ["C", "D", "E"])
        marks = [[t.get("type") for t in n.iter("tuplet")] for n in triplet]
        self.assertEqual(marks, [["start"], [], ["stop"]])
        by_pitch = {name: onset for _, name, onset in _onsets(xml)}
        self.assertEqual(by_pitch["F5"], 1.0)

    def test_real_page_clef_change_in_a_chord(self) -> None:
        """First measure of a system of Clair de lune (Mutopia edition, page 4) as read by the transformer."""
        clair_de_lune = """clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
keySignature_-5 . . . . .
rest_8 _ _ _ _ upper&note_8 D2 b _ _ lower&note_8 A2 b _ _ lower
note_4 F3 _ _ _ upper2&note_4 A3 b _ _ upper&clef_G2 _ _ _ _ lower
note_8 F4 _ _ _ lower&note_8 A4 b _ slurStart lower
note_4. F5 _ _ _ upper&note_4. A5 b _ _ upper&note_2. F4 _ _ _ lower&note_2. C4 b _ _ lower2&note_2. A4 b _ slurStop lower
note_4. F5 _ _ slurStart_slurStop upper&note_4. D5 b _ _ upper
barline . . . . ."""
        tokens = read_token_lines(clair_de_lune.splitlines())
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "")
        lower_clefs = [
            c.findtext("sign") for c in _first_measure(xml).iter("clef") if c.get("number") == "2"
        ]
        self.assertEqual(lower_clefs, ["F", "G"])
        pitches = [_pitch(n) + n.findtext("pitch/octave", "") for n in _notes(_first_measure(xml))]
        self.assertEqual(
            pitches,
            ["rest", "D2", "A2", "F3", "A3", "F4", "A4", "F5", "A5", "F4", "A4", "C4", "F5", "D5"],
        )

    def test_real_page_barline_in_a_chord(self) -> None:
        """First two measures of polish-scores train_050 as read by the transformer."""
        train_050 = """clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
keySignature_-2 . . . . .
note_8 F5 _ _ _ upper&note_8 F4 _ _ _ upper&note_8 F2 _ _ _ lower&note_8 D5 _ _ _ upper&note_8 B4 b _ _ upper
rest_8 _ _ _ _ upper&rest_8 _ _ _ _ lower
clef_G2 _ _ _ _ lower&clef_F4 _ _ _ _ upper
rest_8 _ _ _ _ upper&note_2 F4 _ _ _ lower&note_2 D4 _ _ _ lower&note_2 B4 b accent _ lower
note_12 C3 # _ slurStart upper
note_12 E3 b _ _ upper
note_12 D3 _ _ _ upper
note_12 F3 _ _ _ upper
note_12 B3 b _ _ upper
note_8 F2 _ _ _ lower&barline _ _ _ _ .
rest_8 _ _ _ _ upper&rest_8 _ _ _ _ lower
rest_8 _ _ _ _ upper&note_2 E4 _ _ _ lower&note_2 C4 # _ _ lower&note_2 A4 _ _ _ lower
note_8 G3 # _ slurStart upper
note_8 B3 b _ _ upper
note_8 A3 _ _ _ upper
note_8 E4 _ _ _ upper
note_8 G4 _ _ _ upper
barline . . . . ."""
        tokens = read_token_lines(train_050.splitlines())
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "")
        measures = xml.findall("part/measure")
        self.assertEqual(len(measures), 2)
        self.assertEqual([_pitch(n) for n in _notes(measures[0])][-1], "F")
        self.assertIn("G", [_pitch(n) for n in _notes(measures[1])])

    def test_barline_split_off_a_chord_and_a_repeat_joined_to_the_next_chord(self) -> None:
        """Only a plain barline after a split-off barline is a duplicate, not a chord with a repeat."""
        both = """clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
keySignature_0 . . . . .
timeSignature/4 . . . . .
note_2 C5 _ _ _ upper&note_2 C3 _ _ _ lower&barline _ _ _ _ .
note_2 D5 _ _ _ upper&note_2 D3 _ _ _ lower&repeatEnd . . . . .
note_2 E5 _ _ _ upper&note_2 E3 _ _ _ lower
barline . . . . ."""
        tokens = read_token_lines(both.splitlines())
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "")
        measures = xml.findall("part/measure")
        self.assertEqual(len(measures), 3)
        self.assertEqual(
            [sorted(_pitch(n) for n in _notes(m)) for m in measures],
            [["C", "C"], ["D", "D"], ["E", "E"]],
        )

    def test_symbols_split_off_a_chord_are_written_as_if_they_came_separately(self) -> None:
        """A key and time signature joined to a chord give the same MusicXML as on their own lines."""
        joined = """clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
note_4 C5 _ _ _ upper&note_4 C3 _ _ _ lower&keySignature_2 . . . . .&timeSignature/4 . . . . .
note_4 D5 _ _ _ upper&note_4 D3 _ _ _ lower
barline . . . . ."""
        separate = """clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
keySignature_2 . . . . .
timeSignature/4 . . . . .
note_4 C5 _ _ _ upper&note_4 C3 _ _ _ lower
note_4 D5 _ _ _ upper&note_4 D3 _ _ _ lower
barline . . . . ."""
        written = [
            ET.tostring(
                generate_xml(XmlGeneratorArguments(), [read_token_lines(t.splitlines())], ""),
                encoding="unicode",
            )
            for t in (joined, separate)
        ]
        self.assertEqual(written[0], written[1])
