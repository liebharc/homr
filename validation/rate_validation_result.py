import argparse
import glob
import os
from itertools import chain

import editdistance

from homr.simple_logging import eprint
from homr.staff_parsing import remove_duplicated_symbols  # type: ignore[attr-defined]
from homr.transformer.vocabulary import EncodedSymbol, sort_token_chords
from training.datasets.music_xml_parser import music_xml_file_to_tokens


def _ignore_articulation(symbol: EncodedSymbol) -> EncodedSymbol:
    """
    We ignore articulations for now to get results which are compareable
    to previous versions of the model without articulations.
    """
    return EncodedSymbol(symbol.rhythm, symbol.pitch, symbol.lift)


class MusicFile:
    def __init__(self, filename: str, voices: list[list[list[EncodedSymbol]]]) -> None:
        self.filename = filename
        self.voices = voices
        voices_in_deq = list(chain.from_iterable(voices))
        symbols = [symbol for measure in voices_in_deq for symbol in measure]
        sorted_chords = sort_token_chords(symbols, keep_chord_symbol=True)
        self.symbols = remove_duplicated_symbols(
            [symbol for chord in sorted_chords for symbol in chord], cleanup_tuplets=False
        )
        self.keys = [str(s) for s in self.symbols if s.rhythm.startswith("keySignature")]
        self.notestr = [
            str(_ignore_articulation(s))
            for s in self.symbols
            if s.rhythm.startswith(("note", "rest"))
        ]
        self.symbolstr = [str(s) for s in self.symbols]
        self.is_reference = "reference" in filename

    def diff(self, other: "MusicFile", compare_all: bool) -> int:
        if compare_all:
            return editdistance.eval(self.symbolstr, other.symbolstr)
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
    file: MusicFile, files: list[MusicFile], compare_all: bool
) -> tuple[int | None, MusicFile | None]:
    minimal_diff = None
    minimal_diff_file = None
    for other_file in files:
        if other_file != file:
            diff = diff_against_reference(file, other_file, compare_all)
            if minimal_diff is None or diff < minimal_diff:
                minimal_diff = diff
                minimal_diff_file = other_file
    return minimal_diff, minimal_diff_file


def diff_against_reference(file: MusicFile, reference: MusicFile, compare_all: bool) -> int:
    return file.diff(reference, compare_all)


def get_tokens_from_filename(filename: str) -> MusicFile:
    voices = music_xml_file_to_tokens(filename)
    file = MusicFile(filename, voices)
    return file


def is_xml_or_musicxml(filename: str) -> bool:
    return filename.endswith((".xml", ".musicxml"))


def rate_folder(foldername: str, compare_all: bool) -> tuple[float | None, int]:
    files = all_files_in_folder(foldername)
    all_diffs = []
    sum_of_failures = 0
    xmls: list[MusicFile] = []
    for file in files:
        if not is_xml_or_musicxml(file):
            continue
        if is_file_is_empty(file):
            eprint(">>> Found empty file, that means that the run failed", os.path.basename(file))
            sum_of_failures += 1
            continue
        xmls.append(get_tokens_from_filename(file))

    if len(xmls) <= 1:
        eprint("Not enough files found to compare", foldername)
        sum_of_failures += len(xmls)
        return None, sum_of_failures

    reference = [xml for xml in xmls if xml.is_reference]
    folder_base_name = os.path.basename(foldername.rstrip(os.path.sep))
    if len(reference) != 1:
        for xml in xmls:
            minimal_diff, minimal_diff_file = find_minimal_diff_against_all_other_files(
                xml, xmls, compare_all
            )
            if minimal_diff is None or minimal_diff_file is None:
                eprint("No minimal diff found for", xml.filename)
                sum_of_failures += 1
                continue
            all_diffs.append(minimal_diff)
    else:
        for xml in xmls:
            if xml.is_reference:
                continue
            diff = diff_against_reference(xml, reference[0], compare_all)
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


def rate_all_folders(foldername: str, compare_all: bool) -> bool:
    folders = get_all_direct_subfolders(foldername)
    if len(folders) == 0:
        return False
    all_diffs = []
    sum_of_failures = 0
    lines = []
    for folder in folders:
        diffs, failures = rate_folder(folder, compare_all)
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
    parser.add_argument(
        "--all",
        action="store_true",
        help="Considers all symbols, without this only keys, notes and rests are compared",
    )
    args = parser.parse_args()

    rate_all_folders(args.folder, args.all)
