"""
Tokens <-> MusicXML, shared by the generator and the parser. Each table maps a
token to what the generator writes; the parser reads it back with the inverse
table and the extra spellings in READ_ALIASES.
"""

from typing import NamedTuple


class Notation(NamedTuple):
    container: str  # "" for a direct child of <notations>
    tag: str


ARTICULATIONS: dict[str, Notation] = {
    "fermata": Notation("", "fermata"),
    "arpeggiate": Notation("", "arpeggiate"),
    "accent": Notation("articulations", "accent"),
    "accentStrong": Notation("articulations", "strong-accent"),
    "mordent": Notation("articulations", "mordent"),
    "staccato": Notation("articulations", "staccato"),
    "staccatissimo": Notation("articulations", "staccatissimo"),
    "tenuto": Notation("articulations", "tenuto"),
    "breathMark": Notation("articulations", "breath-mark"),
    "caesura": Notation("articulations", "caesura"),
    "doit": Notation("articulations", "doit"),
    "trill": Notation("ornaments", "trill-mark"),
    "turn": Notation("ornaments", "inverted-turn"),
    "tremolo": Notation("ornaments", "tremolo"),
}

# camelCase MusicXML name -> token, "" drops it
READ_ALIASES: dict[str, str] = {
    "strongAccent": "accentStrong",
    "softAccent": "accent",
    "invertedTurn": "turn",
    "trillMark": "trill",
    # https://www.w3.org/2021/06/musicxml40/musicxml-reference/elements/wavy-line/
    "wavyLine": "trill",
    "invertedMordent": "trill",
    "spiccato": "staccatissimo",
    "accidentalMark": "",
    "scoop": "",
    "doit": "",
    "falloff": "",
    "schleifer": "",
    "caesura": "",
    "otherOrnament": "",
    "otherArticulation": "",
}

BEAMS = {"beamStart": "begin", "beamContinue": "continue", "beamStop": "end"}
BEAMS_READ = {xml: token for token, xml in BEAMS.items()}
BEAMS_PER_TYPE = {"eighth": 1, "16th": 2, "32nd": 3, "64th": 4, "128th": 5}

DURATION_NAMES = {
    0: "breve",
    1: "whole",
    2: "half",
    4: "quarter",
    8: "eighth",
    16: "16th",
    32: "32nd",
    64: "64th",
    128: "128th",
}
DURATION_NUMBER = {name: number for number, name in DURATION_NAMES.items()}

LIFT_TO_ALTER = {"N": 0, "#": 1, "##": 2, "b": -1, "bb": -2}

# <accidental> -> lift
ACCIDENTAL_TO_LIFT = {
    "sharp": "#",
    "flat": "b",
    "natural": "N",
    "double-sharp": "##",
    "flat-flat": "bb",
    # The printed sign is the second one
    "sharp-sharp": "##",
    "natural-sharp": "#",
    "natural-flat": "b",
}
# The generator writes the first spelling of each alter
ALTER_TO_ACCIDENTAL = {
    LIFT_TO_ALTER[lift]: name for name, lift in reversed(ACCIDENTAL_TO_LIFT.items())
}

# The parser reads any "heavy" bar style as bold, any "light" one as double
BAR_STYLES = {"bolddoublebarline": "heavy-heavy", "doublebarline": "light-light"}
REPEATS = {"repeatStart": "forward", "repeatEnd": "backward"}
REPEATS_READ = {xml: token for token, xml in REPEATS.items()}
ENDINGS = {"voltaStart": "start", "voltaStop": "stop", "voltaDiscontinue": "discontinue"}
ENDINGS_READ = {xml: token for token, xml in ENDINGS.items()}


def read_bar_style(style: str) -> str | None:
    if "heavy" in style:
        return "bolddoublebarline"
    if "light" in style:
        return "doublebarline"
    return None


def alter_to_lift(alter: int) -> str:
    if alter == 0:
        return "N"
    if alter > 0:
        return "#" if alter == 1 else "##"
    return "b" if alter == -1 else "bb"


def read_articulation(tag: str) -> str:
    """The token of an articulation or ornament tag, "" if it has none."""
    first, *rest = tag.split("-")
    name = first + "".join(part[:1].upper() + part[1:] for part in rest if part)
    name = name[0].lower() + name[1:]
    return READ_ALIASES.get(name, name)
