import re

from homr.transformer.vocabulary import EncodedSymbol, empty, nonote

# Middle line (L3) anchor notes for each clef
CLEF_ANCHORS = {
    "clef_G1": "D5",
    "clef_G2": "B4",
    "clef_F3": "F3",
    "clef_F4": "D3",
    "clef_F5": "B2",
    "clef_C1": "G4",
    "clef_C2": "E4",
    "clef_C3": "C4",
    "clef_C4": "A3",
    "clef_C5": "F3",
}

# Diatonic step mapping
NOTE_VALS = {"C": 0, "D": 1, "E": 2, "F": 3, "G": 4, "A": 5, "B": 6}
INV_NOTE_VALS = {v: k for k, v in NOTE_VALS.items()}

# Offset to Agnostic Token Mapping (0 = L3)
OFFSET_TO_AGNOSTIC = {
    0: "L3",
    -1: "S2", -2: "L2", -3: "S1", -4: "L1",
    1: "S3", 2: "L4", 3: "S4", 4: "L5",
}
# Fill in ledger lines below (LL1 to LL14)
for i in range(1, 15):
    OFFSET_TO_AGNOSTIC[-4 - i] = f"LL{i}"
# Fill in ledger lines above (LH1 to LH14)
for i in range(1, 15):
    OFFSET_TO_AGNOSTIC[4 + i] = f"LH{i}"

AGNOSTIC_TO_OFFSET = {v: k for k, v in OFFSET_TO_AGNOSTIC.items()}


def _pitch_to_diatonic(pitch: str) -> int:
    if pitch in (nonote, empty):
        return 0
    # Pitch can be "C4" or "C#4" or "Cb4". We only care about base and octave.
    match = re.match(r"([A-G])(?:[#bN]*\d)?", pitch)
    if not match:
        # Check for just base and octave like "C4"
        match = re.match(r"([A-G])(\d)", pitch)
        if not match:
            return 0
    
    # Actually, EncodedSymbol.pitch usually contains the semantic pitch like "C4" or "C#4"
    # But wait, in semantic format, pitch is "C4" and lift is "#".
    # Let's re-verify EncodedSymbol structure.
    # It has .pitch (base+octave) and .lift (accidental).
    
    match = re.match(r"([A-G])(\d)", pitch)
    if not match:
        return 0
    name, octave = match.groups()
    return int(octave) * 7 + NOTE_VALS[name]


def _diatonic_to_pitch(val: int) -> str:
    octave = val // 7
    name = INV_NOTE_VALS[val % 7]
    return f"{name}{octave}"


def to_agnostic_pitch(symbols: list[EncodedSymbol]) -> list[EncodedSymbol]:
    """
    Converts absolute pitches to staff positions relative to the current clef.
    """
    results = []
    # Default anchors
    anchors = {
        "upper": _pitch_to_diatonic("B4"), # G2
        "lower": _pitch_to_diatonic("D3"), # F4
        nonote: _pitch_to_diatonic("B4"),
    }

    for symbol in symbols:
        if symbol.rhythm.startswith("clef_"):
            clef = symbol.rhythm
            if clef in CLEF_ANCHORS:
                anchor = _pitch_to_diatonic(CLEF_ANCHORS[clef])
                if symbol.position in anchors:
                    anchors[symbol.position] = anchor
                    if symbol.position == "upper":
                        anchors[nonote] = anchor
                else:
                    for k in anchors:
                        anchors[k] = anchor
            results.append(symbol)
        elif symbol.pitch not in (nonote, empty) and not symbol.pitch.startswith(("LL", "L1", "S1", "LH")):
            current_anchor = anchors.get(symbol.position, anchors[nonote])
            absolute_val = _pitch_to_diatonic(symbol.pitch)
            relative_val = absolute_val - current_anchor
            
            # Clamp to supported range
            relative_val = max(-18, min(18, relative_val))
            pitch_token = OFFSET_TO_AGNOSTIC.get(relative_val, "L3")
            
            results.append(
                EncodedSymbol(
                    rhythm=symbol.rhythm,
                    pitch=pitch_token,
                    lift=symbol.lift,
                    articulation=symbol.articulation,
                    position=symbol.position,
                    coordinates=symbol.coordinates,
                )
            )
        else:
            results.append(symbol)
    return results


def from_agnostic_pitch(symbols: list[EncodedSymbol]) -> list[EncodedSymbol]:
    """
    Converts staff positions back to absolute pitches based on the current clef.
    """
    results = []
    anchors = {
        "upper": _pitch_to_diatonic("B4"),
        "lower": _pitch_to_diatonic("D3"),
        nonote: _pitch_to_diatonic("B4"),
    }

    for symbol in symbols:
        if symbol.rhythm.startswith("clef_"):
            clef = symbol.rhythm
            if clef in CLEF_ANCHORS:
                anchor = _pitch_to_diatonic(CLEF_ANCHORS[clef])
                if symbol.position in anchors:
                    anchors[symbol.position] = anchor
                    if symbol.position == "upper":
                        anchors[nonote] = anchor
                else:
                    for k in anchors:
                        anchors[k] = anchor
            results.append(symbol)
        elif symbol.pitch in AGNOSTIC_TO_OFFSET:
            current_anchor = anchors.get(symbol.position, anchors[nonote])
            offset = AGNOSTIC_TO_OFFSET[symbol.pitch]
            absolute_val = current_anchor + offset
            pitch_token = _diatonic_to_pitch(absolute_val)
            results.append(
                EncodedSymbol(
                    rhythm=symbol.rhythm,
                    pitch=pitch_token,
                    lift=symbol.lift,
                    articulation=symbol.articulation,
                    position=symbol.position,
                    coordinates=symbol.coordinates,
                )
            )
        else:
            results.append(symbol)
    return results
