"""
Clef and key carried over from the previous staff of a voice.

Scores often print clef and key only on the first system. The model then still reads a
clef at the start of the following staffs, but unsure. HeaderGate replaces such an unsure
clef by the one in effect from the previous staff, and on such a staff also the key: a
staff without a printed clef usually has no printed key either. A clearly printed clef
means a printed staff start, so its key is read from the image.
"""

from dataclasses import dataclass, field

from homr.transformer.vocabulary import EncodedSymbol

GATE_PROBABILITY = 0.85


@dataclass
class StaffContext:
    clefs: dict[str, str] = field(default_factory=dict)  # "upper"/"lower" -> clef
    key: str | None = None


def _clef_position(position: str) -> str:
    return "lower" if position.startswith("lower") else "upper"


def update_context(context: StaffContext | None, symbols: list[EncodedSymbol]) -> StaffContext:
    """The clefs and key in effect after a staff."""
    result = StaffContext(dict(context.clefs), context.key) if context else StaffContext()
    for symbol in symbols:
        if symbol.rhythm.startswith("clef"):
            result.clefs[_clef_position(symbol.position)] = symbol.rhythm
        elif symbol.rhythm.startswith("keySignature"):
            result.key = symbol.rhythm
    return result


class HeaderGate:
    """
    Decides on the clefs and keys of a staff start (all symbols before the first one which
    isn't a clef, key or chord) while the staff is decoded.
    """

    def __init__(self, context: StaffContext | None) -> None:
        self.context = context
        self.in_header = True
        self.unsure_clef = False

    def replacement(self, rhythm: str, position: str, probability: float) -> str | None:
        """The rhythm token to use instead of the model's, None to keep the model's."""
        is_clef = rhythm.startswith("clef")
        is_key = rhythm.startswith("keySignature")
        if self.in_header and not (is_clef or is_key or rhythm == "chord"):
            self.in_header = False
        if not self.in_header or self.context is None:
            return None
        if is_clef:
            if probability >= GATE_PROBABILITY:
                return None
            self.unsure_clef = True
            wanted = self.context.clefs.get(_clef_position(position))
            if wanted is not None and wanted != rhythm:
                return wanted
        elif is_key and self.unsure_clef:
            # without a printed key the model is sure there is none, so its confidence
            # doesn't tell anything here
            wanted_key = self.context.key
            if wanted_key is not None and wanted_key != rhythm:
                return wanted_key
        return None
