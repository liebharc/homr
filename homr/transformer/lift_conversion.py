import re
from homr.transformer.vocabulary import EncodedSymbol, empty, nonote
from homr.circle_of_fifths import KeyTransformation

SEMANTIC_TO_AGNOSTIC = {
    "#": "sharp",
    "b": "flat",
    "N": "natural",
    "##": "double_sharp",
    "bb": "double_flat",
    empty: empty,
    nonote: nonote,
}

AGNOSTIC_TO_SEMANTIC = {
    "sharp": "#",
    "flat": "b",
    "natural": "N",
    "double_sharp": "##",
    "double_flat": "bb",
    empty: empty,
    nonote: nonote,
}


def to_agnostic_lift(symbols: list[EncodedSymbol]) -> list[EncodedSymbol]:
    """
    Converts semantic lifts to agnostic lifts (visually printed tokens).
    Hides accidentals that match the current key signature or measure state.
    Note: Semantic 'empty' (_) means no accidental sign (natural).
    """
    results = []
    key_transform = KeyTransformation(0)

    for symbol in symbols:
        if symbol.rhythm.startswith("keySignature_"):
            try:
                current_key_sig = int(symbol.rhythm.split("_")[1])
                key_transform = KeyTransformation(current_key_sig)
            except (ValueError, IndexError):
                pass
            results.append(symbol)
            continue

        if "barline" in symbol.rhythm or "repeat" in symbol.rhythm:
            key_transform = key_transform.reset_at_end_of_measure()
            results.append(symbol)
            continue

        if symbol.lift == nonote or symbol.pitch in (nonote, empty):
            results.append(symbol)
            continue

        # Check if it's a semantic pitch
        if not re.match(r"[A-G]\d", symbol.pitch):
            # Not a semantic pitch, fallback to simple mapping
            if symbol.lift in SEMANTIC_TO_AGNOSTIC:
                results.append(symbol.change_lift(SEMANTIC_TO_AGNOSTIC[symbol.lift]))
            else:
                results.append(symbol)
            continue

        # In this semantic format, 'empty' means natural
        semantic_accid = "N" if symbol.lift == empty else symbol.lift

        visible = key_transform.add_accidental(symbol.pitch, semantic_accid)
        
        if semantic_accid in ["##", "bb"]:
            # Double sharps/flats are always visible in some styles, 
            # but we use what add_accidental says for consistency if possible.
            # However, SEMANTIC_TO_AGNOSTIC.get("", empty) would hide them.
            # So we force visibility for double accidentals.
            agnostic_lift = SEMANTIC_TO_AGNOSTIC.get(semantic_accid, semantic_accid)
        else:
            agnostic_lift = SEMANTIC_TO_AGNOSTIC.get(visible, empty)

        results.append(symbol.change_lift(agnostic_lift))

    return results


def from_agnostic_lift(symbols: list[EncodedSymbol]) -> list[EncodedSymbol]:
    """
    Converts agnostic lifts back to semantic lifts.
    Restores hidden accidentals from the key signature and measure state.
    Note: Semantic 'empty' (_) means no accidental sign (natural).
    """
    results = []
    key_transform = KeyTransformation(0)

    for symbol in symbols:
        if symbol.rhythm.startswith("keySignature_"):
            try:
                current_key_sig = int(symbol.rhythm.split("_")[1])
                key_transform = KeyTransformation(current_key_sig)
            except (ValueError, IndexError):
                pass
            results.append(symbol)
            continue

        if "barline" in symbol.rhythm or "repeat" in symbol.rhythm:
            key_transform = key_transform.reset_at_end_of_measure()
            results.append(symbol)
            continue

        if symbol.lift == nonote or symbol.pitch in (nonote, empty):
            results.append(symbol)
            continue

        # Check if it's a semantic pitch
        if not re.match(r"[A-G]\d", symbol.pitch):
            # Not a semantic pitch, fallback to simple mapping
            if symbol.lift in AGNOSTIC_TO_SEMANTIC:
                results.append(symbol.change_lift(AGNOSTIC_TO_SEMANTIC[symbol.lift]))
            else:
                results.append(symbol)
            continue

        if symbol.lift == empty:
            state = key_transform.get_accidental(symbol.pitch)
            # State 'N' maps back to semantic 'empty'
            semantic_lift = empty if state == "N" else state
        else:
            semantic_accid = AGNOSTIC_TO_SEMANTIC.get(symbol.lift, symbol.lift)
            key_transform.add_accidental(symbol.pitch, semantic_accid)
            # Again, 'N' maps to 'empty'
            semantic_lift = empty if semantic_accid == "N" else semantic_accid

        results.append(symbol.change_lift(semantic_lift))

    return results

