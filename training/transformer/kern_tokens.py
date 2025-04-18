import re


def filter_for_kern(lines: list[str]) -> list[str]:
    filtered = []
    kern_indices = []
    interpretation_found = False

    for raw_line in lines:
        line = raw_line.rstrip("\n")

        # Before we hit the exclusive interpretation line, keep everything
        if not interpretation_found:
            if line.startswith("**"):
                fields = line.split("\t")
                kern_indices = [i for i, field in enumerate(fields) if field == "**kern"]
                filtered_fields = [fields[i] for i in kern_indices]
                filtered.append("\t".join(filtered_fields))
                interpretation_found = True
            else:
                filtered.append(line)
            continue

        # After interpretation line: filter based on kern indices
        fields = line.split("\t")
        filtered_fields = []

        for i in kern_indices:
            # Use "." if line is short (e.g., missing columns)
            filtered_fields.append(fields[i] if i < len(fields) else ".")

        filtered.append("\t".join(filtered_fields))

    return filtered


def get_symbols_from_file(file: str) -> list[str]:
    with open(file) as f:
        return get_symbols(filter_for_kern(f.readlines()))


def get_symbols(lines: list[str]) -> list[str]:  # noqa: C901
    standalone_dot = "."
    ignore_symbols = [standalone_dot]
    symbols = []
    for line in lines:
        if line.startswith("!"):
            continue
        is_key_or_time = line.startswith(("*M", "*k"))
        if line.startswith("*") and not is_key_or_time:
            continue
        fields = line.split("\t")
        for i, field in enumerate(fields):
            for symbol in field.split():
                if not symbol:
                    continue
                if symbol in ignore_symbols:
                    continue
                if symbol.startswith("="):
                    # With this mapping of "=" we ignore information about measures
                    symbols.append("=")
                    continue

                # By stripping these symbols we ignore encoding of phrases, slurs and ties
                if not symbol.startswith("*"):
                    phrases_slurs_ties = "()[]{}_;"
                    stem_symbols = "/\\"
                    articulation_symbols = "^'\"~`"
                    other_symbols = "@$<>"
                    for ignored_symbol in (
                        stem_symbols + phrases_slurs_ties + articulation_symbols + other_symbols
                    ):
                        symbol = symbol.replace(ignored_symbol, "")  # noqa: PLW2901
                if symbol == "*":
                    continue
                symbols.append(symbol)
            if i < len(fields) - 1:
                symbols.append("<TAB>")
        symbols.append("<NL>")
    return symbols


def split_symbol_into_token(symbol: str) -> tuple[str, str, str, str]:
    # Splits a token into a token for each decoder: is_note, rhythm, pitch, lift
    # Refert to https://www.humdrum.org/rep/kern/ for a description of
    # the different symbols in kern notation

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
        if rhythm.startswith("0"):
            rhythm = "0"
        rhythm = rhythm.replace("qq", "q")
        rhythm = rhythm.replace("..", ".")
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
                symbols = get_symbols_from_file(str(filename))
                tokens += [split_symbol_into_token(sym) for sym in symbols]
            except Exception as e:
                print("Failed to parse", filename, e)
    else:
        symbols = get_symbols_from_file(sys.argv[1])
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
