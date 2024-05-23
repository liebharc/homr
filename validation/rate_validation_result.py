# mypy: disable-error-code="no-any-return, no-any-unimported"
import argparse
import glob
import os
import xml.etree.ElementTree as ET

import editdistance  # type: ignore
import musicxml.xmlelement.xmlelement as mxl  # type: ignore
from musicxml.parser.parser import _parse_node  # type: ignore

from homr.simple_logging import eprint


class Note:
    def __init__(self, note: mxl.XMLNote) -> None:  # type: ignore
        self.note = note
        self.is_chord = get_child_of_type(note, mxl.XMLChord) is not None
        self.pitch = get_child_of_type(note, mxl.XMLPitch)
        if self.pitch:
            self.step = get_child_of_type(self.pitch, mxl.XMLStep)._value
            self.alter = get_child_of_type(self.pitch, mxl.XMLAlter)._value
            self.octave = get_child_of_type(self.pitch, mxl.XMLOctave)._value
        else:
            # Rest
            self.step = None
            self.alter = None
            self.octave = None
        self.duration = get_child_of_type(note, mxl.XMLDuration)._value

    def __str__(self) -> str:
        return f"{self.step}-{self.octave}-{self.alter}: {self.duration}"

    def __repr__(self) -> str:
        return f"{self.step}-{self.octave}-{self.alter}: {self.duration}"

    def __hash__(self) -> int:
        return hash((self.step, self.octave, self.duration, self.alter))

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, Note):
            return False
        return (
            self.step == __value.step
            and self.octave == __value.octave
            and self.duration == __value.duration
            and self.alter == __value.alter
        )

    def __lt__(self, other: "Note") -> bool:
        if self.step != other.step:
            return self.step < other.step
        if self.octave != other.octave:
            return self.octave < other.octave
        if self.alter != other.alter:
            return self.alter < other.alter
        return self.duration < other.duration


class MusicFile:
    def __init__(self, filename: str, keys: list[int], notes: list[Note]) -> None:
        self.filename = filename
        self.keys = keys.copy()
        self.notes = notes.copy()
        self.notestr = [str(note) for note in notes]
        self.is_reference = "reference" in filename

    def diff(self, other: "MusicFile") -> int:
        notedist = editdistance.eval(self.notestr, other.notestr)
        keydist = editdistance.eval(self.keys, other.keys)
        keydiff_rating = 10  # Rate keydiff higher than notediff
        return keydiff_rating * keydist + notedist

    def __str__(self) -> str:
        return str.join(" ", self.notestr)

    def __repr__(self) -> str:
        return str.join(" ", self.notestr)


def all_files_in_folder(foldername: str) -> list[str]:
    return glob.glob(os.path.join(foldername, "*"))


def get_all_direct_subfolders(foldername: str) -> list[str]:
    return sorted(
        [
            os.path.join(foldername, f)
            for f in os.listdir(foldername)
            if os.path.isdir(os.path.join(foldername, f))
        ]
    )


def is_file_is_empty(filename: str) -> bool:
    return os.stat(filename).st_size == 0


def find_minimal_diff_against_all_other_files(
    file: MusicFile, files: list[MusicFile]
) -> tuple[int | None, MusicFile | None]:
    minimal_diff = None
    minimal_diff_file = None
    for other_file in files:
        if other_file != file:
            diff = diff_against_reference(file, other_file)
            if minimal_diff is None or diff < minimal_diff:
                minimal_diff = diff
                minimal_diff_file = other_file
    return minimal_diff, minimal_diff_file


def diff_against_reference(file: MusicFile, reference: MusicFile) -> int:
    return file.diff(reference)


def remove_node_recursively(doc: mxl.XMLScorePartwise, node_names: list[str]) -> None:
    for parent in doc.iter():
        for node_name in node_names:
            for child in parent.findall(node_name):
                parent.remove(child)


def parse_musicxml(filename: str) -> mxl.XMLScorePartwise:
    tree = ET.ElementTree()
    tree.parse(filename)
    root = tree.getroot()
    remove_node_recursively(root, ["miscellaneous", "sound"])
    node: mxl.XMLScorePartwise = _parse_node(root)
    return node


def get_child_of_type(node: mxl.XMLElement, xml_type: type) -> mxl.XMLElement:
    children = [child for child in node.get_children() if isinstance(child, xml_type)]
    if len(children) == 0:
        return None
    return children[0]


def get_all_measures(node: mxl.XMLScorePartwise) -> list[mxl.XMLMeasure]:
    parts = [part for part in node.get_leaves() if isinstance(part, mxl.XMLPart)]
    measures = [
        measure
        for part in parts
        for measure in part.get_children()
        if isinstance(measure, mxl.XMLMeasure)
    ]
    return measures


def get_all_keys_from_measures(measures: list[mxl.XMLMeasure]) -> list[int]:
    def get_fifth(key: mxl.XMLKey) -> int:
        fifths = get_child_of_type(key, mxl.XMLFifths)
        if fifths is None:
            return 0
        return fifths._value

    keys = [
        key
        for measure in measures
        for attribute in measure.get_children()
        if isinstance(attribute, mxl.XMLAttributes)
        for key in attribute.get_children()
        if isinstance(key, mxl.XMLKey)
    ]
    return [get_fifth(key) for key in keys]


def get_all_notes_from_measure(measure: list[mxl.XMLMeasure]) -> list[Note]:
    notes = [note for note in measure.get_children() if isinstance(note, mxl.XMLNote)]  # type: ignore

    return sort_notes_in_chords([Note(note) for note in notes])


def sort_notes_in_chords(notes: list[Note]) -> list[Note]:
    """
    Notes in a chord are not sorted in music XML. In order to compare them, we need to sort them.
    We use the pitch as sort criteria.
    """
    chords: list[list[Note]] = []
    for note in notes:
        if note.is_chord:
            chords[-1].append(note)
        else:
            chords.append([note])
    sorted_chords = [sorted(chord) for chord in chords]
    flattened_chords = [note for chord in sorted_chords for note in chord]
    return flattened_chords


def get_all_notes_from_measures(measures: list[mxl.XMLMeasure]) -> list[Note]:
    return [note for measure in measures for note in get_all_notes_from_measure(measure)]


def get_keys_and_notes_from_filename(filename: str) -> MusicFile:
    xml = parse_musicxml(filename)
    measures = get_all_measures(xml)
    keys = get_all_keys_from_measures(measures)
    notes = get_all_notes_from_measures(measures)
    file = MusicFile(filename, keys, notes)
    return file


def is_xml_or_musicxml(filename: str) -> bool:
    return filename.endswith((".xml", ".musicxml"))


def rate_folder(foldername: str) -> tuple[float | None, int]:
    files = all_files_in_folder(foldername)
    all_diffs = []
    sum_of_failures = 0
    xmls = []
    for file in files:
        if not is_xml_or_musicxml(file):
            continue
        if is_file_is_empty(file):
            eprint(">>> Found empty file, that means that the run failed", os.path.basename(file))
            sum_of_failures += 1
            continue
        xmls.append(get_keys_and_notes_from_filename(file))

    if len(xmls) <= 1:
        eprint("Not enough files found to compare", foldername)
        sum_of_failures += len(xmls)
        return None, sum_of_failures

    reference = [xml for xml in xmls if xml.is_reference]
    folder_base_name = os.path.basename(foldername.rstrip(os.path.sep))
    if len(reference) != 1:
        for xml in xmls:
            minimal_diff, minimal_diff_file = find_minimal_diff_against_all_other_files(xml, xmls)
            if minimal_diff is None or minimal_diff_file is None:
                eprint("No minimal diff found for", xml.filename)
                sum_of_failures += 1
                continue
            all_diffs.append(minimal_diff)
    else:
        for xml in xmls:
            if xml.is_reference:
                continue
            diff = diff_against_reference(xml, reference[0])
            all_diffs.append(diff)

    average_diff = sum(all_diffs) / len(all_diffs)
    eprint("In folder", folder_base_name, ": Average diff is", average_diff)
    return average_diff, sum_of_failures


def write_validation_result_for_folder(
    foldername: str, diffs: float, failures: int, lines: list[str]
) -> None:
    with open(os.path.join(foldername, "validation_result.txt"), "w") as f:
        for line in lines:
            f.write(line + "\n")
        f.write("Diffs: " + str(diffs) + "\n")
        f.write("Failures: " + str(failures) + "\n")


def rate_all_folders(foldername: str) -> bool:
    folders = get_all_direct_subfolders(foldername)
    if len(folders) == 0:
        return False
    all_diffs = []
    sum_of_failures = 0
    lines = []
    for folder in folders:
        diffs, failures = rate_folder(folder)
        if diffs is not None:
            all_diffs.append(diffs)
        folder_base_name = os.path.basename(folder)
        lines.append(folder_base_name + ": " + str(diffs) + ", " + str(failures))
        sum_of_failures += failures
    if len(all_diffs) == 0:
        eprint("Everything failed")
        return True
    average_diff = sum(all_diffs) / len(all_diffs)
    write_validation_result_for_folder(foldername, average_diff, sum_of_failures, lines)
    eprint()
    for line in lines:
        eprint(line)
    eprint("Average diff:", average_diff)
    eprint("Sum of failures:", sum_of_failures)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rate validation results.")
    parser.add_argument(
        "folder", type=str, help="The folder to rate. If 'latest', the newest folder will be rated."
    )
    args = parser.parse_args()

    if not rate_all_folders(args.folder):
        rate_folder(args.folder)
