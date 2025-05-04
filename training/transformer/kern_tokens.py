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


def merge_kerns(voices: list[str]) -> str:
    split = [v.splitlines() for v in voices]
    merged = zip(*split, strict=True)
    return str.join("\n", [str.join("\t", e) for e in merged])


def get_symbols_from_file(file: str) -> list[str]:
    with open(file) as f:
        return get_symbols(filter_for_kern(f.readlines()))


def get_symbols(lines: list[str]) -> list[str]:  # noqa: C901, PLR0912
    standalone_dot = "."
    ignore_symbols = [standalone_dot]
    symbols = []
    for line in lines:
        norm_line = line.strip()
        if norm_line.startswith("!") or not norm_line:
            continue
        is_key_or_time = norm_line.startswith(
            ("*M", "*k", "*clef", "**kern")
        ) and not norm_line.startswith("*MM")
        if norm_line.startswith("*") and not is_key_or_time:
            continue
        fields = norm_line.split("\t")
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
        while len(symbols) > 0 and symbols[-1] == "<TAB>":
            # To use tokens more efficiently we ignore all tabs which are immediately followed
            # by a newline
            del symbols[-1]
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


def split_kern_file_into_measures(kern_file: str) -> tuple[int, str, list[str]]:
    # Return: Number of staffs, key and time sig, measures
    measures = []
    number_of_staffs = 0
    current_measure: list[str] = []
    before_first_measure = ""

    with open(kern_file) as kern:
        lines = kern.readlines()
        lines = filter_for_kern(lines)
        for line in lines:
            if line.startswith("*staff"):
                number_of_staffs = len(line.split())

            if line.startswith("="):
                if before_first_measure == "":
                    before_first_measure = str.join("\n", current_measure) + "\n"
                else:
                    measures.append(str.join("\n", current_measure))
                current_measure = [line]
            else:
                current_measure.append(line)

    return (number_of_staffs, before_first_measure, measures)


def split_kern_measures_into_voices(  # noqa: C901
    number_of_staffs: int, before_first_measure: str, measures: list[str]
) -> tuple[list[str], list[list[str]]]:
    prelude_lines = before_first_measure.split("\n")
    prelude_per_voice: list[list[str]] = [[] for _ in range(number_of_staffs)]
    for line in prelude_lines:
        cells = line.strip().split("\t")
        if len(cells) == 1:
            for voice in prelude_per_voice:
                voice.append(cells[0])
        else:
            for i in range(number_of_staffs):
                prelude_per_voice[i].append(cells[i])

    measures_per_voice: list[list[str]] = [[] for _ in range(number_of_staffs)]
    for measure in measures:
        lines = measure.split("\n")
        lines_per_voice: list[list[str]] = [[] for _ in range(number_of_staffs)]
        for line in lines:
            cells = line.strip().split("\t")
            if len(cells) == 1:
                for voice in lines_per_voice:
                    voice.append(cells[0])
            else:
                for i in range(number_of_staffs):
                    lines_per_voice[i].append(cells[i])

        for i in range(number_of_staffs):
            measures_per_voice[i].append(str.join("\n", lines_per_voice[i]))

    preludes = [str.join("\n", v) + "\n" for v in prelude_per_voice]
    return (preludes, measures_per_voice)


def load_and_sanitize_kern_file(filename: str) -> str:
    symbols = get_symbols_from_file(filename)
    tokens = str.join(" ", symbols)
    tokens = tokens.replace("<NL>", "\n")
    tokens = tokens.replace("<TAB>", "\t")
    return tokens
