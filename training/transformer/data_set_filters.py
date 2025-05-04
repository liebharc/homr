from training.transformer.kern_tokens import (
    get_symbols_from_file,
    split_symbol_into_token,
)


def contains_supported_clef(kern_line: str) -> bool:
    contains_any_clef = "*clef" in kern_line
    if not contains_any_clef:
        return True
    clefs = kern_line.split()

    return all(item in {"*clefF4", "*clefG2"} for item in clefs if "*clef" in item)


def contains_supported_number_of_kerns(kern_line: str) -> bool:
    contains_kern_def = "**kern" in kern_line
    if not contains_kern_def:
        return True
    kerns = kern_line.strip().split()
    kerns = [kern for kern in kerns if kern == "**kern"]
    max_number_of_staffs = (
        2  # We limit the transformer to max two staffs in order to keep its size small
    )
    return len(kerns) <= max_number_of_staffs


def contains_only_supported_clefs(kern_file: str) -> bool:
    with open(kern_file) as f:
        lines = f.readlines()
        return all(
            contains_supported_clef(line) and contains_supported_number_of_kerns(line)
            for line in lines
        )


if __name__ == "__main__":
    # ruff: noqa: T201
    import sys
    from pathlib import Path

    src_files = Path(sys.argv[1])
    invalid_clefs = 0
    number_of_files = 0
    max_symbols = 0
    if src_files.is_dir():
        tokens = []
        for filename in src_files.rglob("*.krn"):
            try:
                if not contains_only_supported_clefs(str(filename)):
                    print(filename)
                    invalid_clefs += 1
                number_of_files += 1
                symbols = get_symbols_from_file(str(filename))
                max_symbols = max(len(symbols), max_symbols)
                tokens += [split_symbol_into_token(sym) for sym in symbols]
            except Exception as e:
                print("Failed to parse", filename, e)
    else:
        if not contains_only_supported_clefs(sys.argv[1]):
            invalid_clefs += 1
        number_of_files += 1
        symbols = get_symbols_from_file(sys.argv[1])
        max_symbols = max(len(symbols), max_symbols)
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
    print("max_symbols", max_symbols)
    print("invalid_clefs", invalid_clefs, "of", number_of_files, "files")
