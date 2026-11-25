"""
Tokens -> MusicXML. homr/music_xml/builder.py lays the tokens of each part out
into the model of homr/music_xml/model.py, the passes complete it (voices, ties,
accidentals, stems, beams), and homr/music_xml/writer.py writes the XML.
"""

import xml.etree.ElementTree as ET

from homr.music_xml import model as m
from homr.music_xml.builder import PartBuilder
from homr.music_xml.passes import PASSES, shorten_measure_rests_in_pickup
from homr.music_xml.tokens import TimeSignature
from homr.music_xml.writer import write_part
from homr.transformer.vocabulary import EncodedSymbol, is_lower_position

__all__ = ["TimeSignature", "XmlGeneratorArguments", "generate_xml", "last_time_signatures"]


class XmlGeneratorArguments:
    def __init__(
        self, large_page: bool | None = None, metronome: int | None = None, tempo: int | None = None
    ):
        self.large_page = large_page
        self.metronome = metronome
        self.tempo = tempo


def generate_xml(
    args: XmlGeneratorArguments,
    staffs: list[list[EncodedSymbol]],
    title: str,
    inherited_times: list[TimeSignature | None] | None = None,
) -> ET.Element:
    """
    inherited_times: the time signature in force at the end of the previous page, per
    part, for a page that continues a score. A part that doesn't start with a time
    signature continues in it, instead of one inferred from its measure lengths.
    """
    root = ET.Element("score-partwise", version="4.0")
    root.append(build_work(title))
    root.append(build_identification())
    root.append(build_defaults(args))
    parts = [
        m.Part(get_part_id(index), any(is_lower_position(s.position) for s in staff))
        for index, staff in enumerate(staffs)
    ]
    root.append(build_part_list([part.has_two_staves for part in parts]))
    metronome = None
    if args.metronome:
        metronome = m.Metronome(args.metronome, args.tempo if args.tempo else args.metronome)
    for index, (part, staff) in enumerate(zip(parts, staffs, strict=True)):
        inherited = None
        if inherited_times and index < len(inherited_times):
            inherited = inherited_times[index]
        PartBuilder(staff, part, metronome if index == 0 else None, inherited).build()
        for complete in PASSES:
            complete(part)
    shorten_measure_rests_in_pickup(parts)
    root.extend(write_part(part) for part in parts)
    return root


def last_time_signatures(root: ET.Element) -> list[TimeSignature | None]:
    """The last time signature of each part, e.g. to continue a score on the next page."""
    result: list[TimeSignature | None] = []
    for part in root.iter("part"):
        times = list(part.iter("time"))
        last = times[-1] if times else None
        result.append(
            (last.findtext("beats", ""), last.findtext("beat-type", ""))
            if last is not None
            else None
        )
    return result


def build_identification() -> ET.Element:
    ident = ET.Element("identification")
    enc = ET.SubElement(ident, "encoding")
    ET.SubElement(enc, "software").text = "homr"
    return ident


def build_work(title_text: str) -> ET.Element:
    work = ET.Element("work")
    ET.SubElement(work, "work-title").text = title_text
    return work


def build_defaults(args: XmlGeneratorArguments) -> ET.Element:
    defaults = ET.Element("defaults")
    if args.large_page:
        page_layout = ET.SubElement(defaults, "page-layout")
        ET.SubElement(page_layout, "page-height").text = "300"
        ET.SubElement(page_layout, "page-width").text = "110"
    return defaults


def get_part_id(index: int) -> str:
    return "P" + str(index + 1)


def _part_metadata(has_two_staves: bool) -> tuple[str, str, str, int]:
    """Return (part_name, instrument_name, instrument_sound, midi_program) for a part.

    We classify by staff layout only:
    - single-staff parts -> Voice
    - two-staff parts -> Piano
    midi_program is 1-based (MusicXML: 1-128; 1 = Acoustic Grand Piano, 54 = Voice Oohs).
    """
    if has_two_staves:
        return ("Piano", "Piano", "keyboard.piano", 1)
    return ("Voice", "Voice", "voice", 54)


def build_part_list(has_two_staves_by_part: list[bool]) -> ET.Element:
    part_list = ET.Element("part-list")
    for part, has_two_staves in enumerate(has_two_staves_by_part):
        part_id = get_part_id(part)
        part_name_str, instrument_name_str, instrument_sound_str, midi_program = _part_metadata(
            has_two_staves
        )
        score_part = ET.SubElement(part_list, "score-part", id=part_id)
        ET.SubElement(score_part, "part-name").text = part_name_str
        score_instrument = ET.SubElement(score_part, "score-instrument", id=part_id + "-I1")
        ET.SubElement(score_instrument, "instrument-name").text = instrument_name_str
        ET.SubElement(score_instrument, "instrument-sound").text = instrument_sound_str
        midi_instrument = ET.SubElement(score_part, "midi-instrument", id=part_id + "-I1")
        ET.SubElement(midi_instrument, "midi-channel").text = str(part + 1)
        ET.SubElement(midi_instrument, "midi-program").text = str(midi_program)
        ET.SubElement(midi_instrument, "volume").text = "100"
        ET.SubElement(midi_instrument, "pan").text = "0"
    return part_list


if __name__ == "__main__":
    import sys

    from training.transformer.training_vocabulary import read_tokens

    file = "tabi_measure.tokens"
    if len(sys.argv) > 1:
        file = sys.argv[1]
    tokens = read_tokens(file)
    xml = generate_xml(XmlGeneratorArguments(True), [tokens], "")
    ET.ElementTree(xml).write(
        file.replace(".tokens", ".musicxml"), encoding="unicode", xml_declaration=True
    )
