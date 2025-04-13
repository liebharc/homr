import re


def get_symbols(file: str) -> list[str]:
    standalone_dot = "."
    ignore_symbols = [standalone_dot]
    with open(file) as f:
        symbols = []
        for line in f.readlines():
            if line.startswith("!"):
                continue
            is_key_or_time = line.startswith(("*M", "*k"))
            if line.startswith("*") and not is_key_or_time:
                continue
            for symbol in line.split():
                if not symbol:
                    continue
                if symbol in ignore_symbols:
                    continue
                if symbol.startswith("="):
                    # With this mapping of "=" we ignore information about measures
                    symbols.append("=")
                    continue
                symbols.append(symbol)
            symbols.append("<NL>")
        return symbols


def split_symbol_into_token(symbol: str) -> tuple[str, str, str, str]:
    # Splits a token into a token for each decoder: is_note, rhythm, pitch, lift
    # Refert to https://www.humdrum.org/rep/kern/ for a description of
    # the different symbols in kern notation

    # By stripping these symbols we ignore encoding of phrases, slurs and ties
    if not symbol.startswith("*"):
        phrases_slurs_ties = "()[]{}_;"
        stem_symbols = "/\\"
        articulation_symbols = "^'\"~`"
        other_symbols = "@$"
        for ignored_symbol in (
            stem_symbols + phrases_slurs_ties + articulation_symbols + other_symbols
        ):
            symbol = symbol.replace(ignored_symbol, "")

    match = re.match("^([0-9q]+[\\.q]*)?([a-gA-G]+|r|R|RR)([#n-]*)?(.*)?$", symbol)
    if match:
        # By ignroing group 4, we ignore information about beams
        rhythm = match[1]  # r=rest, R=unpitched note, RR=semi unpitched note
        pitch = match[2]
        lift = match[3]
        if not rhythm:
            rhythm = "q"  # grace note
        if not lift:
            lift = "nonote"
        return ("note", rhythm, pitch, lift)
    return ("nonote", symbol, "nonote", "nonote")


if __name__ == "__main__":
    # ruff: noqa: T201
    import sys
    from pathlib import Path

    src_files = Path(sys.argv[1])
    if src_files.is_dir():
        tokens = []
        for filename in src_files.rglob("*.krn"):
            try:
                symbols = get_symbols(str(filename))
                tokens += [split_symbol_into_token(sym) for sym in symbols]
            except Exception as e:
                print("Failed to parse", filename, e)
    else:
        symbols = get_symbols(sys.argv[1])
        tokens = [split_symbol_into_token(sym) for sym in symbols]
    note_tokens = set()
    rhythm_tokens = set()
    pitch_tokens = set()
    lift_tokens = set()

    for note, rhythm, pitch, lift in tokens:
        note_tokens.add(note)
        rhythm_tokens.add(rhythm)
        pitch_tokens.add(pitch)
        lift_tokens.add(lift)

    print("note", sorted(note_tokens))
    print("rhythm", sorted(rhythm_tokens))
    print("pitch", sorted(pitch_tokens))
    print("lift", sorted(lift_tokens))
