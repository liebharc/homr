"""
MusicXML -> model, the inverse of writer.py. The items of a measure keep the order
of the XML. Notes get their start from backup and forward, everything else the time
at which it stands. Notations which aren't printed are left out, except ties, slurs
and arpeggios.
"""

import xml.etree.ElementTree as ET

from homr.music_xml import model as m


def read_parts(root: ET.Element) -> list[m.Part]:
    return [read_part(part) for part in root.findall("part")]


def read_part(xml: ET.Element, strict: bool = True) -> m.Part:
    """
    strict: a backup beyond the start of its measure is an error, otherwise it goes
    back to the start.
    """
    part = m.Part(xml.get("id", ""), False)
    for xml_measure in xml.findall("measure"):
        measure = _MeasureReader(xml_measure, strict).read()
        if part.divisions == 0 and measure.divisions:
            part.divisions = measure.divisions
        part.has_two_staves |= any(n.staff == 2 for n in measure.notes())
        part.measures.append(measure)
    return part


def _text(element: ET.Element | None, default: str = "") -> str:
    if element is None or element.text is None:
        return default
    return element.text.strip()


def _int(element: ET.Element | None, default: int = 0) -> int:
    text = _text(element)
    return int(text) if text else default


def _visible(element: ET.Element) -> bool:
    return element.get("print-object") != "no"


class _MeasureReader:
    def __init__(self, xml: ET.Element, strict: bool) -> None:
        self.xml = xml
        self.strict = strict
        self.measure = m.Measure(_measure_number(xml))
        self.time = 0
        self.last_start = 0  # of the last note which isn't a chord note

    def read(self) -> m.Measure:
        for child in self.xml:
            if child.tag == "note":
                self.note(child)
            elif child.tag in ("backup", "forward"):
                duration = _int(child.find("duration"))
                if child.tag == "backup" and duration > self.time and self.strict:
                    raise ValueError(f"Backup duration is too long {self.time} {-duration}")
                self.time = max(0, self.time + (-duration if child.tag == "backup" else duration))
            elif child.tag == "attributes":
                self.attributes(child)
            elif child.tag == "barline":
                self.barline(child)
            elif child.tag == "direction":
                self.direction(child)
            elif child.tag == "print":
                if child.get("new-page") == "yes":
                    self.measure.items.append(m.NewPage(self.time))
                if child.get("new-system") == "yes":
                    self.measure.items.append(m.NewSystem(self.time))
        return self.measure

    def note(self, xml: ET.Element) -> None:
        note = m.Note(staff=_int(xml.find("staff"), 1), voice=_int(xml.find("voice"), 0))
        note.is_chord = xml.find("chord") is not None
        note.is_grace = xml.find("grace") is not None
        note.is_hidden = not _visible(xml) or any(
            _text(head) == "none" for head in xml.findall("notehead")
        )
        pitch, rest = xml.find("pitch"), xml.find("rest")
        if pitch is not None:
            alter = _text(pitch.find("alter"))
            note.pitch = m.Pitch(
                _text(pitch.find("step")), _text(pitch.find("octave")), _number(alter)
            )
        elif rest is not None:
            note.is_measure_rest = rest.get("measure") == "yes"
        else:
            note.is_unpitched = True
        if xml.find("duration") is not None:
            note.duration = _int(xml.find("duration"))
        note.type = xml.findtext("type")
        note.dots = len(xml.findall("dot"))
        note.accidental = xml.findtext("accidental")
        note.stem = xml.findtext("stem")
        actual = xml.find("time-modification/actual-notes")
        normal = xml.find("time-modification/normal-notes")
        if actual is not None and normal is not None:
            note.time_modification = (_int(actual), _int(normal))
        note.beams = [(int(b.get("number", "1")), _text(b)) for b in xml.findall("beam")]
        note.ties = [tie.get("type", "") for tie in xml.findall("tie")]
        _read_notations(note, xml.find("notations"))
        if not note.is_chord:
            self.last_start = self.time
            if not note.is_grace:
                self.time += note.duration or 0
        note.start = self.last_start
        self.measure.items.append(note)

    def attributes(self, xml: ET.Element) -> None:
        if xml.find("divisions") is not None:
            self.measure.divisions = _int(xml.find("divisions"), 1) or 1
        for clef in xml.findall("clef"):
            octave_change = _int(clef.find("clef-octave-change"))
            item = m.Clef(
                int(clef.get("number", "1")),
                _text(clef.find("sign")),
                _text(clef.find("line")),
                octave_change,
                self.time,
                clef.get("after-barline") == "yes",
            )
            self.measure.items.append(item)
        for key in xml.findall("key"):
            number = key.get("number")
            fifths = int(_text(key.find("fifths"), "0"))
            self.measure.items.append(m.Key(fifths, int(number) if number else None, self.time))
        for time in xml.findall("time"):
            beats, beat_type = _text(time.find("beats")), _text(time.find("beat-type"))
            self.measure.items.append(m.Time(beats, beat_type, self.time))
        for rest in xml.findall("measure-style/multiple-rest"):
            self.measure.items.append(m.MultipleRest(_int(rest), self.time))

    def barline(self, xml: ET.Element) -> None:
        barline = m.Barline(xml.get("location", "right"), time=self.time)
        for child in xml:
            if child.tag == "bar-style":
                barline.children.append(m.BarStyle(_text(child)))
            elif child.tag == "ending":
                number = "".join(c for c in child.get("number", "1").split(",")[0] if c.isdigit())
                barline.children.append(m.Ending(child.get("type", ""), int(number or 1)))
            elif child.tag == "repeat":
                barline.children.append(m.Repeat(child.get("direction", "")))
        self.measure.items.append(barline)

    def direction(self, xml: ET.Element) -> None:
        staff = _int(xml.find("staff"), 1)
        for direction_type in xml.findall("direction-type"):
            for shift in direction_type.findall("octave-shift"):
                size = int(shift.get("size", "8"))
                placement = xml.get("placement", "")
                item = m.OctaveShift(staff, shift.get("type", ""), placement, self.time, size)
                self.measure.items.append(item)
            for dynamics in direction_type.findall("dynamics"):
                if _visible(dynamics):
                    marks = [(child.tag, child.text) for child in dynamics]
                    self.measure.items.append(m.Dynamics(staff, marks, self.time))


def _measure_number(xml: ET.Element) -> int:
    digits = "".join(c for c in xml.get("number", "0") if c.isdigit())
    return int(digits or 0)


def _number(text: str) -> int | float | None:
    if not text:
        return None
    value = float(text)
    return int(value) if value.is_integer() else value


def _read_notations(note: m.Note, xml: ET.Element | None) -> None:
    if xml is None:
        return
    # A tuplet counts whether or not it's printed
    note.tuplets = [t.get("type", "") for t in xml.findall("tuplet")]
    if not _visible(xml):
        return
    note.has_notations = True
    for child in xml:
        if child.tag == "tied":
            note.tied.append(child.get("type", ""))
        elif child.tag == "slur":
            number = child.get("number")
            note.slurs.append(m.Slur(child.get("type", ""), int(number) if number else None))
        elif child.tag == "arpeggiate":
            note.direct.append("arpeggiate")
        elif not _visible(child):
            continue
        elif child.tag == "fermata":
            note.direct.append("fermata")
        elif child.tag == "articulations":
            note.articulations.extend(articulation.tag for articulation in child)
        elif child.tag == "ornaments":
            for ornament in child:
                note.ornaments.append((ornament.tag, dict(ornament.attrib), ornament.text))
