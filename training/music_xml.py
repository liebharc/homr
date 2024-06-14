from typing import Any

import xmltodict

from homr.circle_of_fifths import KeyTransformation, circle_of_fifth_to_key_signature


class SemanticPart:
    def __init__(self) -> None:
        self.staffs: list[list[str]] = []
        self.chords: list[list[str]] = []

    def append_clefs(self, clefs: list[str]) -> None:
        if len(self.staffs) > 0:
            return
        for clef in clefs:
            self.staffs.append([clef])
            self.chords.append([])

    def append_symbol(self, symbol: str) -> None:
        if len(self.staffs) == 0:
            raise ValueError("Expected to get clefs as first symbol")
        if symbol.startswith("note"):
            raise ValueError("Call append_note for notes")
        else:
            for staff in self.staffs:
                staff.append(symbol)

    def append_note(self, staff: int, symbol: str) -> None:
        if len(self.staffs) == 0:
            raise ValueError("Expected to get clefs as first symbol")
        self.staffs[staff].append(symbol)

    def append_chord(self, staff: int, symbol: str) -> None:
        if len(self.staffs) == 0:
            raise ValueError("Expected to get clefs as first symbol")
        self.chords[staff].append(symbol)

    def flush_chord(self, staff: int) -> None:
        if len(self.staffs) == 0:
            raise ValueError("Expected to get clefs as first symbol")
        if len(self.chords[staff]) == 0:
            return
        self.staffs[staff].append("|".join(self.chords[staff]))
        self.chords[staff].clear()

    def flush_chords(self) -> None:
        for staff in range(len(self.staffs)):
            self.flush_chord(staff)

    def get_staffs(self) -> list[list[str]]:
        return self.staffs


def _translate_duration(duration: str | dict[Any, Any]) -> str:
    duration_text = ""
    if isinstance(duration, dict):
        duration_text = duration["#text"]
    else:
        duration_text = duration
    definition = {
        "breve": "double_whole",
        "whole": "whole",
        "half": "half",
        "quarter": "quarter",
        "eighth": "eighth",
        "16th": "sixteenth",
        "32nd": "thirty_second",
        "64th": "sixty_fourth",
    }
    return definition[duration_text]


def _get_alter(note: dict[str, str]) -> str:
    if "alter" not in note:
        return ""
    if note["alter"] == "1":
        return "#"
    if note["alter"] == "-1":
        return "b"
    if note["alter"] == "0":
        return "0"
    return ""


def _ensure_list(obj: Any) -> list[Any]:
    if isinstance(obj, list):
        return obj
    return [obj]


def _count_dots(note: Any) -> str:
    if "dot" not in note:
        return ""
    return "." * len(_ensure_list(note["dot"]))


def _process_attributes(
    semantic: SemanticPart, attribute: dict[str, dict[str, str]], key: KeyTransformation
) -> KeyTransformation:
    if "clef" in attribute:
        clefs = _ensure_list(attribute["clef"])
        clefs = sorted(clefs, key=lambda c: int(c.get("number", "0")))
        semantic.append_clefs(["clef-" + clef["sign"] + clef["line"] for clef in clefs])
    if "key" in attribute:
        semantic.append_symbol(
            "keySignature-" + circle_of_fifth_to_key_signature(int(attribute["key"]["fifths"]))
        )
        key = KeyTransformation(int(attribute["key"]["fifths"]))
    if "time" in attribute:
        semantic.append_symbol(
            "timeSignature-" + attribute["time"]["beats"] + "/" + attribute["time"]["beat-type"]
        )
    return key


def _process_note(semantic: SemanticPart, note: Any, key: KeyTransformation) -> KeyTransformation:
    staff = int(note.get("staff", "1")) - 1
    if "chord" not in note:
        # Flush the previous chord
        semantic.flush_chord(staff)
    if "rest" in note:
        dot = _count_dots(note)
        if note["rest"] and "@measure" in note["rest"]:
            semantic.append_note(staff, "rest-whole" + dot)
        else:
            semantic.append_note(staff, "rest-" + _translate_duration(note["type"]) + dot)
    if "pitch" in note:
        alter = _get_alter(note["pitch"])
        key.add_accidental(
            note["pitch"]["step"] + alter + note["pitch"]["octave"],
            _get_alter(note["pitch"]),
        )
        semantic.append_chord(
            staff,
            "note-"
            + note["pitch"]["step"]
            + alter
            + note["pitch"]["octave"]
            + "_"
            + _translate_duration(note["type"])
            + _count_dots(note),
        )
    return key


def _music_part_to_semantic(part: Any) -> list[list[str]]:
    semantic = SemanticPart()
    for measure in _ensure_list(part["measure"]):
        key = KeyTransformation(0)
        if "backup" in measure:
            raise ValueError("Backup not supported")
        if "attributes" in measure:
            for attribute in _ensure_list(measure["attributes"]):
                _process_attributes(semantic, attribute, key)
        if "note" in measure:
            for note in _ensure_list(measure["note"]):
                _process_note(semantic, note, key)

        # Flush the last chord
        semantic.flush_chords()
        semantic.append_symbol("barline")
        key = key.reset_at_end_of_measure()
    return semantic.get_staffs()


def music_xml_to_semantic(path: str) -> list[list[str]]:
    result = []
    with open(path) as f:
        musicxml = xmltodict.parse(f.read())
        parts = _ensure_list(musicxml["score-partwise"]["part"])
        for part in parts:
            semantic = _music_part_to_semantic(part)
            result.extend(semantic)
    return result
