"""Runs the passes of homr/music_xml on small hand-written MusicXML."""

import xml.etree.ElementTree as ET
from collections.abc import Callable

from homr.music_xml import model as m
from homr.music_xml.reader import read_part
from homr.music_xml.writer import write_part

__all__ = ["apply", "read_part"]


def apply(complete: Callable[[m.Part], None], xml: ET.Element) -> ET.Element:
    """Runs a pass on hand-written MusicXML and returns the XML of the result."""
    part = read_part(xml, strict=False)
    complete(part)
    return write_part(part)
