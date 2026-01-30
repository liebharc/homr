import argparse
import glob
import os
from dataclasses import dataclass
from itertools import chain

import editdistance

from homr.simple_logging import eprint
from homr.staff_parsing import remove_duplicated_symbols  # type: ignore[attr-defined]
from homr.transformer.vocabulary import EncodedSymbol, sort_token_chords
from training.datasets.music_xml_parser import Measure, music_xml_file_to_tokens


@dataclass
class ValidationMetrics:
    diff: float
    ser: float
    distance: int = 0
    expected_length: int = 0

    @property
    def total_ser(self) -> float:
        return self.distance / self.expected_length if self.expected_length > 0 else 0.0

    def __str__(self) -> str:
        s = f"{self.diff:.1f} diffs"
        if self.expected_length > 0:
            s += f", SER: {100 * self.total_ser:.1f}%"
        return s


def _ignore_articulation(symbol: EncodedSymbol) -> EncodedSymbol:
    """
    We ignore articulations for now to get results which are compareable
    to previous versions of the model without articulations.
    """
    return EncodedSymbol(symbol.rhythm, symbol.pitch, symbol.lift)


class MusicFile:
    def __init__(self, filename: str, voices: list[list[Measure]]) -> None:
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

    def calculate_metrics(self, other: "MusicFile", compare_all: bool) -> ValidationMetrics:
        diff = self.diff(other, compare_all)
        if compare_all:
            expected = other.symbolstr
            actual = self.symbolstr
        else:
            expected = other.keys + other.notestr
            actual = self.keys + self.notestr

        distance = editdistance.eval(expected, actual)
        expected_length = len(expected)
        ser = distance / expected_length if expected_length > 0 else 0.0
        return ValidationMetrics(float(diff), ser, distance, expected_length)

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
) -> tuple[ValidationMetrics | None, MusicFile | None]:
    minimal_metrics = None
    minimal_diff_file = None
    for other_file in files:
        if other_file != file:
            metrics = file.calculate_metrics(other_file, compare_all)
            if minimal_metrics is None or metrics.diff < minimal_metrics.diff:
                minimal_metrics = metrics
                minimal_diff_file = other_file
    return minimal_metrics, minimal_diff_file


def get_tokens_from_filename(filename: str) -> MusicFile:
    voices = music_xml_file_to_tokens(filename)
    file = MusicFile(filename, voices)
    return file


def is_xml_or_musicxml(filename: str) -> bool:
    return filename.endswith((".xml", ".musicxml"))


def rate_folder(foldername: str, compare_all: bool) -> tuple[ValidationMetrics | None, int]:
    files = all_files_in_folder(foldername)
    all_metrics: list[ValidationMetrics] = []
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
            metrics, minimal_diff_file = find_minimal_diff_against_all_other_files(
                xml, xmls, compare_all
            )
            if metrics is None or minimal_diff_file is None:
                eprint("No minimal diff found for", xml.filename)
                sum_of_failures += 1
                continue
            all_metrics.append(metrics)
    else:
        for xml in xmls:
            if xml.is_reference:
                continue
            metrics = xml.calculate_metrics(reference[0], compare_all)
            all_metrics.append(metrics)

    average_diff = sum(m.diff for m in all_metrics) / len(all_metrics)
    average_ser = sum(m.ser for m in all_metrics) / len(all_metrics)
    total_distance = sum(m.distance for m in all_metrics)
    total_expected_length = sum(m.expected_length for m in all_metrics)
    average_metrics = ValidationMetrics(
        average_diff, average_ser, total_distance, total_expected_length
    )

    eprint("In folder", folder_base_name, ":", average_metrics)
    return average_metrics, sum_of_failures


def write_validation_result_for_folder(
    foldername: str, metrics: ValidationMetrics, failures: int, lines: list[str]
) -> None:
    with open(os.path.join(foldername, "validation_result.txt"), "w") as f:
        for line in lines:
            f.write(line + "\n")
        f.write("Diffs: " + str(metrics.diff) + "\n")
        f.write("SER: " + str(metrics.ser) + "\n")
        if metrics.expected_length > 0:
            f.write("Total SER: " + str(metrics.total_ser) + "\n")
        f.write("Failures: " + str(failures) + "\n")


def rate_all_folders(foldername: str, compare_all: bool) -> bool:
    folders = get_all_direct_subfolders(foldername)
    if len(folders) == 0:
        return False
    all_metrics: list[ValidationMetrics] = []
    sum_of_failures = 0
    lines = []
    for folder in folders:
        metrics, failures = rate_folder(folder, compare_all)
        if metrics is not None:
            all_metrics.append(metrics)
        folder_base_name = os.path.basename(folder)
        lines.append(folder_base_name + ": " + str(metrics) + ", " + str(failures))
        sum_of_failures += failures
    if len(all_metrics) == 0:
        eprint("Everything failed")
        return True

    average_diff = sum(m.diff for m in all_metrics) / len(all_metrics)
    average_ser = sum(m.ser for m in all_metrics) / len(all_metrics)
    total_distance = sum(m.distance for m in all_metrics)
    total_expected_length = sum(m.expected_length for m in all_metrics)
    average_metrics = ValidationMetrics(
        average_diff, average_ser, total_distance, total_expected_length
    )

    write_validation_result_for_folder(foldername, average_metrics, sum_of_failures, lines)
    eprint()
    for line in lines:
        eprint(line)
    eprint(average_metrics)
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
