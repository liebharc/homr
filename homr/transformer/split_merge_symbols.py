import re

from homr import constants
from homr.circle_of_fifths import (
    KeyTransformation,
    NoKeyTransformation,
    key_signature_to_circle_of_fifth,
)
from homr.simple_logging import eprint
from homr.transformer.configs import default_config


class SymbolMerger:
    def __init__(self, keep_all_symbols_in_chord: bool) -> None:
        self._keep_all_symbols_in_chord = keep_all_symbols_in_chord
        self.merge: list[list[str]] = []
        self.next_symbol_is_chord: bool = False
        self.last_clef: str = ""

    def _append_symbol(self, symbol: str) -> None:
        if self.next_symbol_is_chord:
            if len(self.merge) == 0:
                eprint("Warning: Unexpected chord symbol")
                return
            self.merge[-1].append(symbol)
            self.next_symbol_is_chord = False
        else:
            self.merge.append([symbol])

    def add_symbol(self, predrhythm: str, predpitch: str, predlift: str) -> bool:
        """
        Adds a symbol to the merge list. Returns True if the symbol should be retried.

        If you done with adding symbols, call complete() to get the merged string.
        """
        if predrhythm == "|":
            self.next_symbol_is_chord = True
            if len(self.merge) == 0:
                eprint("Warning: Unexpected chord symbol")
                return True
            return False
        elif "note" in predrhythm:
            lift = ""
            if predlift in (
                "lift_##",
                "lift_#",
                "lift_bb",
                "lift_b",
                "lift_N",
            ):
                lift = predlift.split("_")[-1]
            self._append_symbol(predpitch + lift + "_" + predrhythm.split("note-")[-1])
            return False
        elif "clef" in predrhythm:
            # Two clefs in a the same staff are very unlikely
            if self.last_clef and self.last_clef != predrhythm:
                eprint("Warning: Two clefs in a staff")
                return True
            self.last_clef = predrhythm
            self._append_symbol(predrhythm)
            return False
        else:
            self._append_symbol(predrhythm)
            return False

    def _clean_and_sort_chord(self, chord: list[str]) -> list[str]:
        if len(chord) == 1:
            return chord
        if not self._keep_all_symbols_in_chord:
            chord = [symbol for symbol in chord if symbol.startswith("note")]
        chord = sorted(chord, key=pitch_name_to_sortable)
        return chord

    def complete(self) -> str:
        merged = [str.join("|", self._clean_and_sort_chord(symbols)) for symbols in self.merge]
        return str.join("+", merged)


def merge_single_line(
    predrhythm: list[str],
    predpitch: list[str],
    predlift: list[str],
    keep_all_symbols_in_chord: bool,
) -> str:
    merger = SymbolMerger(keep_all_symbols_in_chord=keep_all_symbols_in_chord)

    for j in range(len(predrhythm)):
        merger.add_symbol(predrhythm[j], predpitch[j], predlift[j])

    return merger.complete()


def merge_symbols(
    predrhythms: list[list[str]],
    predpitchs: list[list[str]],
    predlifts: list[list[str]],
    keep_all_symbols_in_chord: bool = False,
) -> list[str]:
    merges = []
    for i in range(len(predrhythms)):
        predrhythm = predrhythms[i]
        predlift = predlifts[i]
        predpitch = predpitchs[i]
        merge = merge_single_line(
            predrhythm, predpitch, predlift, keep_all_symbols_in_chord=keep_all_symbols_in_chord
        )
        merges.append(merge)
    return merges


def _get_alter(symbol: str) -> str | None:
    if symbol.startswith(("note", "gracenote")):
        note = symbol.split("_")[0]
        # We have no support for double accidentals right now
        alterations = {"##": "#", "#": "#", "bb": "b", "b": "b", "N": "N"}
        for alteration, return_value in alterations.items():
            if alteration in note:
                return return_value
        return ""
    return None


def _alter_to_lift(symbol: str) -> str:
    if symbol == "#":
        return "lift_#"
    elif symbol == "b":
        return "lift_b"
    elif symbol == "N":
        return "lift_N"
    else:
        return "lift_null"


def _replace_accidentals(notename: str) -> str:
    notename = notename.replace("#", "")
    notename = notename.replace("b", "")
    notename = notename.replace("N", "")
    return notename


def _symbol_to_pitch(symbol: str) -> str:
    if symbol.startswith(("note", "gracenote")):
        without_duration = symbol.split("_")[0]
        notename = without_duration.split("-")[1]
        notename = _replace_accidentals(notename)
        notename = "note-" + notename
        return notename
    return "nonote"


def _add_duration_modifier(duration: str) -> str:
    # TrOMR only allows one dot
    if "." in duration:
        return "."
    if constants.triplet_symbol in duration:
        # Ignore triplets for now
        # return constants.triplet_symbol
        return ""
    return ""


def _translate_duration(duration: str) -> str:
    duration = duration.replace("second", "breve")
    duration = duration.replace("double", "breve")
    duration = duration.replace("quadruple", "breve")
    duration = duration.replace("thirty", "thirty_second")
    duration = duration.replace("sixty", "sixty_fourth")
    duration = duration.replace("hundred", "hundred_twenty_eighth")

    # We add duration modifiers later again
    duration = duration.replace(".", "")
    duration = duration.replace(constants.triplet_symbol, "")
    return duration


def _symbol_to_rhythm(symbol: str) -> str:
    if symbol.startswith(("note", "gracenote")):
        note = "note-" + _translate_duration(symbol.split("_")[1])
        return note + _add_duration_modifier(symbol)
    symbol = symbol.replace("rest-double_whole", "multirest-2")
    symbol = symbol.replace("rest-quadruple_whole", "multirest-2")
    symbol = symbol.replace("_fermata", "")

    # We add duration modifiers later again
    symbol = symbol.replace(".", "")
    symbol = symbol.replace(constants.triplet_symbol, "")
    multirest_match = re.match(r"(rest-whole|multirest-)(\d+)", symbol)
    if multirest_match:
        rest_length = int(multirest_match[2])
        # Some multirests don't exist in the rhtythm tokenizer,
        # for now it's good enough to just recognize them as any multirest
        if rest_length <= 1:
            return "rest-whole"
        max_supported_multi_rest = 10
        if rest_length > max_supported_multi_rest:
            return "multirest-" + str(max_supported_multi_rest)
        symbol = "multirest-" + str(rest_length)
    timesignature_match = re.match(r"timeSignature-(\d+)/(\d+)", symbol)
    if timesignature_match:
        return "timeSignature-/" + timesignature_match[2]
    return symbol + _add_duration_modifier(symbol)


def _symbol_to_note(symbol: str) -> str:
    if symbol.startswith(("note", "gracenote")):
        return "note"
    return "nonote"


def _note_name_and_octave_to_sortable(note_name_with_octave: str) -> int:
    if note_name_with_octave not in default_config.pitch_vocab:
        eprint(
            "Warning: unknown note in _note_name_and_octave_to_sortable: ", note_name_with_octave
        )
        return 1000
    # minus to get the right order
    return -int(default_config.pitch_vocab[note_name_with_octave])


def pitch_name_to_sortable(pitch_or_rest_name: str) -> int:
    if pitch_or_rest_name.startswith("rest"):
        pitch_or_rest_name = pitch_or_rest_name.replace("rest_", "rest-")
        if pitch_or_rest_name in default_config.rhythm_vocab:
            return 1000 + int(default_config.rhythm_vocab[pitch_or_rest_name])
        else:
            eprint("Warning: rest not in rhythm_vocab", pitch_or_rest_name)
            return 1000
    note_name = pitch_or_rest_name.split("_")[0]
    note_name = _replace_accidentals(note_name)
    return _note_name_and_octave_to_sortable(note_name)


def _sort_by_pitch(
    lifts: list[str], pitches: list[str], rhythms: list[str], notes: list[str]
) -> tuple[list[str], list[str], list[str], list[str]]:
    lifts = lifts.copy()
    pitches = pitches.copy()
    rhythms = rhythms.copy()
    notes = notes.copy()

    def swap(i: int, j: int) -> None:
        lifts[i], lifts[j] = lifts[j], lifts[i]
        pitches[i], pitches[j] = pitches[j], pitches[i]
        rhythms[i], rhythms[j] = rhythms[j], rhythms[i]
        notes[i], notes[j] = notes[j], notes[i]

    for i in range(len(pitches)):
        if not rhythms[i].startswith("note") and not rhythms[i].startswith("rest"):
            continue
        expect_chord = True
        for j in range(i + 1, len(pitches)):
            is_chord = rhythms[j] == "|"
            if is_chord != expect_chord:
                break
            if is_chord:
                expect_chord = False
                continue
            if not rhythms[j].startswith("note") and not rhythms[j].startswith("rest"):
                break
            symbol_at_i = pitches[i] if pitches[i] != "nonote" else rhythms[i]
            symbol_at_j = pitches[j] if pitches[j] != "nonote" else rhythms[j]
            if pitch_name_to_sortable(symbol_at_i) > pitch_name_to_sortable(symbol_at_j):
                swap(i, j)
            expect_chord = True
    return lifts, pitches, rhythms, notes


def convert_alter_to_accidentals(merged: list[str]) -> list[str]:
    """
    Moves alter information into accidentals.
    For example:
    """
    all_results = []
    for line in range(len(merged)):
        key = KeyTransformation(0)
        line_result = []
        for symbols in re.split("\\s+", merged[line].replace("+", " ")):
            symbol_result = []
            for symbol in re.split("(\\|)", symbols):
                if symbol.startswith("keySignature"):
                    key = KeyTransformation(key_signature_to_circle_of_fifth(symbol.split("-")[-1]))
                    symbol_result.append(symbol)
                elif symbol == "barline":
                    key = key.reset_at_end_of_measure()
                    symbol_result.append(symbol)
                elif symbol.startswith(("note", "gracenote")):
                    pitch = _symbol_to_pitch(symbol)
                    alter = _get_alter(symbol)
                    note_name = pitch[5:7]
                    accidental = key.add_accidental(note_name, alter)
                    parts = symbol.split("_")
                    transformed_symbol = (
                        parts[0].replace("N", "").replace("#", "").replace("b", "")
                        + accidental
                        + "_"
                        + parts[1]
                    )
                    symbol_result.append(transformed_symbol)
                elif symbol != "|":
                    symbol_result.append(symbol)

            if len(symbol_result) > 0:
                line_result.append(str.join("|", symbol_result))
        all_results.append(str.join("+", line_result))
    return all_results


def split_semantic_file(
    file_path: str,
) -> tuple[list[list[str]], list[list[str]], list[list[str]], list[list[str]]]:
    is_primus = "Corpus" in file_path
    with open(file_path) as f:
        return split_symbols(f.readlines(), convert_to_modified_semantic=is_primus)


def split_symbols(  # noqa: C901
    merged: list[str], convert_to_modified_semantic: bool = True
) -> tuple[list[list[str]], list[list[str]], list[list[str]], list[list[str]]]:
    """
    modified_semantic: Semantic format but with accidentals depending on how they are placed.

    E.g. the semantic format is Key D Major, Note C#, Note Cb, Note Cb
    so the TrOMR will be: Key D Major, Note C, Note Cb, Note C because
    the flat is the only visible accidental in the image.
    """
    predlifts = []
    predpitchs = []
    predrhythms = []
    prednotes = []
    for line in range(len(merged)):
        predlift = []
        predpitch = []
        predrhythm = []
        prednote = []
        key = KeyTransformation(0) if convert_to_modified_semantic else NoKeyTransformation()
        for symbols in re.split("\\s+|\\+", merged[line].strip()):
            symbollift = []
            symbolpitch = []
            symbolrhythm = []
            symbolnote = []
            for symbol in re.split("(\\|)", symbols):
                if symbol.startswith("keySignature"):
                    if convert_to_modified_semantic:
                        key = KeyTransformation(
                            key_signature_to_circle_of_fifth(symbol.split("-")[-1])
                        )
                if symbol == "barline":
                    key = key.reset_at_end_of_measure()

                if symbol == "tie":
                    continue
                elif symbol == "|":
                    symbolrhythm.append("|")
                    symbolpitch.append("nonote")
                    symbollift.append("nonote")
                    symbolnote.append("nonote")
                else:
                    pitch = _symbol_to_pitch(symbol)
                    symbolpitch.append(pitch)
                    symbolrhythm.append(_symbol_to_rhythm(symbol))
                    symbolnote.append(_symbol_to_note(symbol))
                    alter = _get_alter(symbol)
                    if alter is not None:
                        note_name = pitch[5:7]
                        alter = key.add_accidental(note_name, alter)
                        symbollift.append(_alter_to_lift(alter))
                    else:
                        symbollift.append("nonote")
            if len(symbolpitch) > 0:
                symbollift, symbolpitch, symbolrhythm, symbolnote = _sort_by_pitch(
                    symbollift, symbolpitch, symbolrhythm, symbolnote
                )
                predpitch += symbolpitch
                predrhythm += symbolrhythm
                prednote += symbolnote
                predlift += symbollift
        predlifts.append(predlift)
        predpitchs.append(predpitch)
        predrhythms.append(predrhythm)
        prednotes.append(prednote)
    return predlifts, predpitchs, predrhythms, prednotes
