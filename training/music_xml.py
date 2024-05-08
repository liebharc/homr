from typing import Any

import xmltodict

from homr.circle_of_fifths import KeyTransformation, circle_of_fifth_to_key_signature


def _translate_duration(duration: str) -> str:
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
    return definition[duration]


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
    semantic: list[str], attribute: dict[str, dict[str, str]], key: KeyTransformation
) -> KeyTransformation:
    if "clef" in attribute:
        clef = _ensure_list(attribute["clef"])[0]
        semantic.append("clef-" + clef["sign"] + clef["line"])
    if "key" in attribute:
        semantic.append(
            "keySignature-" + circle_of_fifth_to_key_signature(int(attribute["key"]["fifths"]))
        )
        key = KeyTransformation(int(attribute["key"]["fifths"]))
    if "time" in attribute:
        semantic.append(
            "timeSignature-" + attribute["time"]["beats"] + "/" + attribute["time"]["beat-type"]
        )
    return key


def _process_note(
    semantic: list[str], note: Any, chord: Any, key: KeyTransformation
) -> tuple[list[str], KeyTransformation]:
    if "chord" not in note:
        if len(chord) > 0:
            # Flush the previous chord
            semantic.append("|".join(chord))
            chord = []
    if "rest" in note:
        dot = _count_dots(note)
        if note["rest"] and "@measure" in note["rest"]:
            semantic.append("rest-whole" + dot)
        else:
            semantic.append("rest-" + _translate_duration(note["type"]) + dot)
    if "pitch" in note:
        alter = _get_alter(note["pitch"])
        key.add_accidental(
            note["pitch"]["step"] + alter + note["pitch"]["octave"],
            _get_alter(note["pitch"]),
        )
        chord.append(
            "note-"
            + note["pitch"]["step"]
            + alter
            + note["pitch"]["octave"]
            + "_"
            + _translate_duration(note["type"])
            + _count_dots(note)
        )
    return chord, key


def _music_part_to_semantic(part: Any) -> list[str]:
    try:
        semantic: list[str] = []
        for measure in _ensure_list(part["measure"]):
            chord: list[str] = []
            key = KeyTransformation(0)
            if "attributes" in measure:
                for attribute in _ensure_list(measure["attributes"]):
                    key = _process_attributes(semantic, attribute, key)
            if "note" in measure:
                for note in _ensure_list(measure["note"]):
                    chord, key = _process_note(semantic, note, chord, key)

            if len(chord) > 0:
                # Flush the last chord
                semantic.append("|".join(chord))
            semantic.append("barline")
            key = key.reset_at_end_of_measure()
        return semantic
    except Exception as e:
        print("Failure at ", part)
        raise e


def music_xml_to_semantic(path: str) -> list[list[str]]:
    result = []
    with open(path) as f:
        musicxml = xmltodict.parse(f.read())
        parts = _ensure_list(musicxml["score-partwise"]["part"])
        for part in parts:
            semantic = _music_part_to_semantic(part)
            result.append(semantic)
    return result
