"""Model -> MusicXML, in the order of the MusicXML 4.0 schema."""

import xml.etree.ElementTree as ET
from itertools import groupby

from homr.music_xml import model as m


def _sub(parent: ET.Element, tag: str, text: object = None, **attrib: str) -> ET.Element:
    element = ET.SubElement(parent, tag, attrib)
    if text is not None:
        element.text = str(text)
    return element


def write_part(part: m.Part) -> ET.Element:
    xml = ET.Element("part", id=part.id)
    for number, measure in enumerate(part.measures):
        writer = _MeasureWriter(_sub(xml, "measure", number=str(measure.number)))
        if number == 0:
            writer.header = (part.divisions, part.has_two_staves)
        writer.write(measure)
    return xml


class _MeasureWriter:
    """
    Writes one voice after the other, as MusicXML readers expect: each voice in the
    order of time, with a backup to the start of the measure before the next one and a
    forward over gaps. Attributes go into the first voice at their time, directions into
    the first voice of their staff.
    """

    def __init__(self, xml: ET.Element) -> None:
        self.xml = xml
        self.cursor = 0
        self.header: tuple[int, bool] | None = None  # divisions and two staves

    def write(self, measure: m.Measure) -> None:
        for item in measure.items:
            if isinstance(item, m.NewSystem | m.NewPage):
                kind = "new-system" if isinstance(item, m.NewSystem) else "new-page"
                _sub(self.xml, "print", **{kind: "yes"})
        barlines = [i for i in measure.items if isinstance(i, m.Barline)]
        for barline in (b for b in barlines if b.location == "left"):
            _barline(self.xml, barline)
        for voice in _voices(measure):
            self.move_to(0)
            # Attributes at the same time are one <attributes>
            for _, group in groupby(
                voice, key=lambda i: (i.time, "") if isinstance(i, m.Attribute) else id(i)
            ):
                self.write_items(list(group))
        right = [b for b in barlines if b.location != "left"]
        if right:
            self.move_to(max(self.cursor, measure.end()))  # a barline ends the measure
        for barline in right:
            _barline(self.xml, barline)

    def write_items(self, items: list[m.Item]) -> None:
        item = items[0]
        if isinstance(item, m.Note):
            if not item.is_chord:
                self.move_to(item.start)
                self.cursor = item.end
            _note(self.xml, item)
            return
        self.move_to(item.time)
        if isinstance(item, m.Dynamics):
            _dynamics(self.xml, item)
        elif isinstance(item, m.OctaveShift):
            _octave_shift(self.xml, item)
        elif isinstance(item, m.Metronome):
            _metronome(self.xml, item)
        else:
            _attributes(self.xml, items, self.header)  # type: ignore[arg-type]
            self.header = None

    def move_to(self, time: int) -> None:
        if time < self.cursor:
            _sub(_sub(self.xml, "backup"), "duration", self.cursor - time)
        elif time > self.cursor:
            _sub(_sub(self.xml, "forward"), "duration", time - self.cursor)
        self.cursor = time


def _voices(measure: m.Measure) -> list[list[m.Item]]:
    """
    The notes of each voice, and the other items, in the order to write them: by time,
    what isn't a note before the notes at its time. Voice numbers belong to the part,
    a voice of a cross-staff beam has notes on both staves.
    """
    by_voice: dict[int, list[m.Item]] = {}
    for note in sorted(measure.notes(), key=lambda n: n.voice):
        by_voice.setdefault(note.voice, []).append(note)
    voices = list(by_voice.values()) or [[]]
    first_on_staff: dict[int, int] = {}
    for index, voice in enumerate(voices):
        for item in voice:
            if isinstance(item, m.Note):
                first_on_staff.setdefault(item.staff, index)
    for item in measure.items:
        if isinstance(item, m.Dynamics | m.OctaveShift):
            voices[first_on_staff.get(item.staff, 0)].append(item)
        elif isinstance(item, m.Attribute | m.Metronome):
            voices[0].append(item)
    for voice in voices:
        # Stable: graces stay before their note, chord notes after their first note
        voice.sort(key=_order)
    return voices


def _order(item: m.Item) -> tuple[int, int]:
    return (item.start, 1) if isinstance(item, m.Note) else (item.time, 0)


def _note(parent: ET.Element, note: m.Note) -> None:  # noqa: C901
    xml = _sub(parent, "note")
    if note.is_grace:
        _sub(xml, "grace")
    if note.is_chord:
        _sub(xml, "chord")
    if note.pitch is None:
        _sub(xml, "rest", **({"measure": "yes"} if note.is_measure_rest else {}))
    else:
        pitch = _sub(xml, "pitch")
        _sub(pitch, "step", note.pitch.step)
        if note.pitch.alter is not None:
            _sub(pitch, "alter", note.pitch.alter)
        _sub(pitch, "octave", note.pitch.octave)
    if note.duration is not None:
        _sub(xml, "duration", note.duration)
    for tie in note.ties:
        _sub(xml, "tie", type=tie)
    _sub(xml, "voice", note.voice)
    if note.type is not None:
        _sub(xml, "type", note.type)
    for _ in range(note.dots):
        _sub(xml, "dot")
    if note.accidental is not None:
        _sub(xml, "accidental", note.accidental)
    if note.time_modification is not None:
        time_modification = _sub(xml, "time-modification")
        _sub(time_modification, "actual-notes", note.time_modification[0])
        _sub(time_modification, "normal-notes", note.time_modification[1])
    if note.stem is not None:
        _sub(xml, "stem", note.stem)
    _sub(xml, "staff", note.staff)
    for number, beam in note.beams:
        _sub(xml, "beam", beam, number=str(number))
    _notations(xml, note)
    if note.image_position is not None:
        x, y = note.image_position
        xml.append(ET.Comment(f" imgpos: {x}, {y} "))


def _notations(note_xml: ET.Element, note: m.Note) -> None:
    parts = (note.ties, note.slurs, note.tuplets, note.ornaments, note.articulations, note.direct)
    if not any(parts) and not note.has_tremolo:
        return
    xml = _sub(note_xml, "notations")
    for tie in note.ties:
        _sub(xml, "tied", type=tie)
    for slur in note.slurs:
        _sub(xml, "slur", type=slur.type, **({"number": str(slur.number)} if slur.number else {}))
    for tuplet in note.tuplets:
        _sub(xml, "tuplet", type=tuplet)
    if note.ornaments or note.has_tremolo:
        ornaments = _sub(xml, "ornaments")
        for tag, attrib, text in note.ornaments:
            _sub(ornaments, tag, text, **attrib)
        if note.has_tremolo:
            _sub(ornaments, "tremolo", 3, type="single")
    if note.articulations:
        articulations = _sub(xml, "articulations")
        for tag in note.articulations:
            _sub(articulations, tag)
    for tag in note.direct:
        _sub(xml, tag)


def _attributes(
    parent: ET.Element, items: list[m.Attribute], header: tuple[int, bool] | None
) -> None:
    xml = _sub(parent, "attributes")
    if header is not None:
        _sub(xml, "divisions", header[0])
    for key in (i for i in items if isinstance(i, m.Key)):
        number = {"number": str(key.staff)} if key.staff else {}
        _sub(_sub(xml, "key", **number), "fifths", key.fifths)
    for time in (i for i in items if isinstance(i, m.Time)):
        time_xml = _sub(xml, "time")
        _sub(time_xml, "beats", time.beats)
        _sub(time_xml, "beat-type", time.beat_type)
    if header is not None and header[1]:
        _sub(xml, "staves", 2)
        _sub(xml, "part-symbol", "brace")
    for clef in (i for i in items if isinstance(i, m.Clef)):
        clef_xml = _sub(xml, "clef", number=str(clef.staff))
        _sub(clef_xml, "sign", clef.sign)
        _sub(clef_xml, "line", clef.line)
        if clef.octave_change != 0:
            _sub(clef_xml, "clef-octave-change", clef.octave_change)
    for rest in (i for i in items if isinstance(i, m.MultipleRest)):
        _sub(_sub(xml, "measure-style"), "multiple-rest", rest.measures)


def _dynamics(parent: ET.Element, dynamics: m.Dynamics) -> None:
    direction = _sub(parent, "direction", placement="below")
    xml = _sub(_sub(direction, "direction-type"), "dynamics")
    for tag, text in dynamics.marks:
        _sub(xml, tag, text)
    _sub(direction, "staff", dynamics.staff)


def _octave_shift(parent: ET.Element, shift: m.OctaveShift) -> None:
    direction = _sub(parent, "direction", placement=shift.placement)
    direction_type = _sub(direction, "direction-type")
    _sub(direction_type, "octave-shift", type=shift.type, size="8", number=str(shift.staff))
    _sub(direction, "staff", shift.staff)


def _metronome(parent: ET.Element, metronome: m.Metronome) -> None:
    direction = _sub(parent, "direction")
    xml = _sub(_sub(direction, "direction-type"), "metronome")
    _sub(xml, "beat-unit", "quarter")
    _sub(xml, "per-minute", metronome.per_minute)
    _sub(direction, "sound", tempo=str(metronome.tempo))


def _barline(parent: ET.Element, barline: m.Barline) -> None:
    """At most one style, ending and repeat each, the last one set."""
    xml = _sub(parent, "barline", location=barline.location)
    by_kind = {type(child): child for child in barline.children}
    if (style := by_kind.get(m.BarStyle)) is not None:
        _sub(xml, "bar-style", style.style)  # type: ignore[union-attr]
    if (ending := by_kind.get(m.Ending)) is not None:
        _sub(xml, "ending", type=ending.type, number=str(ending.number))  # type: ignore[union-attr]
    if (repeat := by_kind.get(m.Repeat)) is not None:
        _sub(xml, "repeat", direction=repeat.direction)  # type: ignore[union-attr]
