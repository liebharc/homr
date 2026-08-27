# ruff: noqa: E501, S101

import unittest
import xml.etree.ElementTree as ET
from fractions import Fraction

from homr.music_xml_generator import (
    SymbolChord,
    XmlGeneratorArguments,
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

    def test_multi_voice_measure_retiming(self) -> None:
        """
        Measure 6 of the accompaniment in liebharc/homr#142: the model
        serializes the two left-hand voices as sequential events, so the
        shared-cursor interpretation overflows the measure (27/4 quarters
        in 4/4). Re-timing must replay each staff on its own cursor and
        shrink the held half note's advance to its engraved slot, so that
        every staff closes exactly on the measure duration.
        """
        arpeggio_measure = """note_8 A4 b _ _ upper&note_8 E4 b _ _ upper&note_8 C4 _ _ _ upper&note_2 A2 b _ _ lower&note_2 A1 b _ _ lower
note_4 E4 b _ _ upper&note_4 C4 _ _ _ upper&note_4 A3 b _ _ upper
note_8 E4 b _ _ upper&note_8 C4 _ _ _ upper&note_8 A3 b _ _ upper
note_8 A4 b _ _ upper&note_8 E4 b _ _ upper&note_8 C4 _ _ _ upper&note_2 A2 b _ _ lower&note_2 A1 b _ _ lower
note_4 E4 b _ _ upper&note_4 C4 _ _ _ upper&note_4 A3 b _ _ upper
note_8 E4 b _ _ upper&note_8 C4 _ _ _ upper&note_8 A3 b _ _ upper
barline . . . . ."""
        two_voice_measure = """note_4 A4 b _ _ upper&note_4 E4 b _ _ upper&note_4 C4 _ _ _ upper&note_2 A2 b _ _ lower
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
        header = """clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
keySignature_-4 . . . . .
timeSignature/4 . . . . ."""
        tokens = read_token_lines(
            str.join(
                "\n", [header, arpeggio_measure, two_voice_measure, arpeggio_measure]
            ).splitlines()
        )
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "")
        part = xml.find("part")
        assert part is not None
        divisions_text = part.findtext("measure/attributes/divisions")
        assert divisions_text is not None
        divisions = int(divisions_text)

        for measure in part.findall("measure"):
            attacks = self._read_note_attacks(measure)
            measure_end = max(attack + duration for attack, duration, _, _, _ in attacks)
            self.assertEqual(measure_end, 4 * divisions, measure.get("number"))

        retimed = part.findall("measure")[1]
        lower = [
            (attack, duration, pitch)
            for attack, duration, pitch, staff, _ in self._read_note_attacks(retimed)
            if staff == "2"
        ]
        in_beats = [(Fraction(a, divisions), Fraction(d, divisions), p) for a, d, p in lower]
        self.assertEqual(
            in_beats,
            [
                # The held half note keeps its duration but only advances the
                # moving voice's first (unwritten) eighth
                (Fraction(0), Fraction(2), "A2"),
                (Fraction(1, 2), Fraction(1), "A3"),
                (Fraction(3, 2), Fraction(1, 2), "A3"),
                (Fraction(2), Fraction(1, 2), "A2"),
                (Fraction(5, 2), Fraction(1), "C4"),
                (Fraction(7, 2), Fraction(1, 2), "E4"),
            ],
        )

        # Voice separation follows the plan, not overlap packing: the held
        # half note is its own voice; the moving eighth-quarter-eighth
        # ostinato stays in one voice even after the half has ended.
        lower_voices = [
            voice for _, _, _, staff, voice in self._read_note_attacks(retimed) if staff == "2"
        ]
        self.assertEqual(lower_voices, ["6", "5", "5", "5", "5", "5"])

    def test_competing_retimings_are_left_alone(self) -> None:
        """
        An overflowing measure can admit more than one equally simple
        re-timing: here either lower-staff half note can act as the held
        voice and absorb the extra quarter, and the two readings put the
        E3 attack on different beats. The arithmetic alone cannot decide,
        so the measure must be left as it was instead of guessing.
        """
        clean_measure = """note_1 C5 _ _ _ upper&note_1 C3 _ _ _ lower
barline . . . . ."""
        ambiguous_measure = """note_4 C5 _ _ _ upper&note_2 C3 _ _ _ lower
note_4 D5 _ _ _ upper
note_2 E3 _ _ _ lower
note_4 E5 _ _ _ upper
note_4 F5 _ _ _ upper&note_4 G3 _ _ _ lower
barline . . . . ."""
        header = """clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
keySignature_1 . . . . .
timeSignature/4 . . . . ."""
        tokens = read_token_lines(
            str.join("\n", [header, clean_measure, ambiguous_measure, clean_measure]).splitlines()
        )
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "")
        part = xml.find("part")
        assert part is not None
        divisions_text = part.findtext("measure/attributes/divisions")
        assert divisions_text is not None
        divisions = int(divisions_text)

        ambiguous = part.findall("measure")[1]
        attacks = self._read_note_attacks(ambiguous)
        measure_end = max(attack + duration for attack, duration, _, _, _ in attacks)
        self.assertGreater(measure_end, 4 * divisions)

    def test_retimed_chord_with_moving_and_held_note_splits_voices(self) -> None:
        """
        A single token chord can carry both lines of a staff: a quarter that
        moves with the cursor and a half that keeps sounding across the next
        attack. The held note must go to its own voice per note, not per
        part — sharing the part's voice would make one voice carry two
        overlapping notes.
        """
        clean_measure = """note_1 C5 _ _ _ upper&note_1 C2 _ _ _ lower
barline . . . . ."""
        mixed_measure = """note_4 C5 _ _ _ upper&note_4 E3 _ _ _ lower&note_2 C3 _ _ _ lower
note_4 D5 _ _ _ upper
note_2 G3 _ _ _ lower
note_4 E5 _ _ _ upper
note_4 F5 _ _ _ upper&note_4 B3 _ _ _ lower
barline . . . . ."""
        header = """clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
keySignature_1 . . . . .
timeSignature/4 . . . . ."""
        tokens = read_token_lines(
            str.join("\n", [header, clean_measure, mixed_measure, clean_measure]).splitlines()
        )
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "")
        part = xml.find("part")
        assert part is not None
        divisions_text = part.findtext("measure/attributes/divisions")
        assert divisions_text is not None
        divisions = int(divisions_text)

        retimed = part.findall("measure")[1]
        attacks = self._read_note_attacks(retimed)
        measure_end = max(attack + duration for attack, duration, _, _, _ in attacks)
        self.assertEqual(measure_end, 4 * divisions)

        lower = [event for event in attacks if event[3] == "2"]
        self.assertEqual(
            lower,
            [
                (0, divisions, "E3", "2", "5"),
                (0, 2 * divisions, "C3", "2", "6"),
                (divisions, 2 * divisions, "G3", "2", "5"),
                (3 * divisions, divisions, "B3", "2", "5"),
            ],
        )

    def _read_note_attacks(self, measure: ET.Element) -> list[tuple[int, int, str, str, str]]:
        """Non-chord-tone note events as (attack, duration, pitch, staff, voice)."""
        cursor = 0
        events = []
        for el in measure:
            if el.tag == "note":
                if el.find("chord") is not None:
                    continue
                duration = int(el.findtext("duration", "0"))
                pitch = el.find("pitch")
                name = (
                    pitch.findtext("step", "") + pitch.findtext("octave", "")
                    if pitch is not None
                    else "rest"
                )
                events.append(
                    (cursor, duration, name, el.findtext("staff", "1"), el.findtext("voice", ""))
                )
                cursor += duration
            elif el.tag == "backup":
                cursor -= int(el.findtext("duration", "0"))
            elif el.tag == "forward":
                cursor += int(el.findtext("duration", "0"))
        return events

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
