"""
This code is based on relieur: https://github.com/papoteur-mga/relieur/tree/master
which is licensed under Apache-2.0.
We thank Papoteur (https://github.com/papoteur-mga) and contributors for their work!
"""

import os
import xml.etree.ElementTree as ET
from glob import glob
from pathlib import Path

from homr.simple_logging import eprint


def get_children_of_type(element, tag_):
    return element.findall(tag_)


def clef_attributes(clef) -> dict[str]:  # type ignore
    # take an ElementTree <clef> element
    attributes = {}
    for a in get_children_of_type(clef, "sign"):
        attributes["Sign"] = a.text
    for a in get_children_of_type(clef, "sign"):
        attributes["Line"] = a.text
    for a in get_children_of_type(clef, "clef-octave-change"):
        attributes["ClefOctaveChange"] = a.text
    return attributes


def time_attributes(time) -> dict[str]:  # type ignore
    # take an ElementTree <time> element
    attributes = {}
    for a in get_children_of_type(time, "beats"):
        attributes["Beats"] = a.text
    for a in get_children_of_type(time, "beat-type"):
        attributes["BeatType"] = a.text
    for a in get_children_of_type(time, "interchangeable"):
        attributes["Interchangeable"] = a.text
    for a in get_children_of_type(time, "senza-misura"):
        attributes["SenzaMisura"] = a.text
    return attributes


def key_attributes(key) -> dict[str]:  # type ignore
    # take an ElementTree <key> element
    attributes = {}
    for a in get_children_of_type(key, "fifths"):
        attributes["Fifths"] = a.text
    for a in get_children_of_type(key, "key-alter"):
        attributes["KeyAlter"] = a.text
    for a in get_children_of_type(key, "mode"):
        attributes["Mode"] = a.text
    return attributes


def update_attributes_in_force(part_attributes: dict, measure) -> None:  # type: ignore
    for attrib in get_children_of_type(measure, "attributes"):
        for div in get_children_of_type(attrib, "divisions"):
            part_attributes["Divisions"] = div.text
        for key in get_children_of_type(attrib, "key"):
            part_attributes["Key"] = key_attributes(key)
        for xtime in get_children_of_type(attrib, "time"):
            part_attributes["Time"] = time_attributes(xtime)
        for clef in get_children_of_type(attrib, "clef"):
            part_attributes["Clef"] = clef_attributes(clef)


def process_concat(
    concat: list[str],
    debug=False,
) -> tuple:
    sorted_list = get_file_list(concat, debug=debug)
    eprint(sorted_list)
    if not sorted_list:
        return None, 0, 0
    # Main file is the first of the list
    main_file = sorted_list[0]
    eprint(f"Starting with {main_file}")
    m = ET.parse(main_file).getroot()  # noqa: S314
    # look for the last key and last divisions
    last_parts_attributes = []
    part1 = None  # Initialize part1 to avoid UnboundLocalError
    for part1 in get_children_of_type(m, "part"):
        part_attributes: dict = {}
        for measure in get_children_of_type(part1, "measure"):
            update_attributes_in_force(part_attributes, measure)
        last_parts_attributes.append(part_attributes)
    for f in sorted_list[1:]:
        # new file to add
        if debug:
            eprint(f"Processing {f}")
        b = ET.parse(f).getroot()  # noqa: S314
        ip = 0
        for part1 in get_children_of_type(m, "part"):
            # each part from the main score
            current_len = len(get_children_of_type(part1, "measure"))
            if debug:
                eprint(f"Main part has {current_len} measures")
            ib = 0
            for part in get_children_of_type(b, "part"):
                if ib == ip:
                    # we add the part of the new file having the same order
                    # as the part from the main score, else we pass
                    for measure in get_children_of_type(part, "measure"):
                        new_number = str(int(measure.get("number")) + current_len)
                        if int(measure.get("number")) == 1:
                            in_force = last_parts_attributes[ib]
                            for attrib in get_children_of_type(measure, "attributes"):
                                for div in get_children_of_type(attrib, "divisions"):
                                    if in_force.get("Divisions") == div.text:
                                        attrib.remove(div)
                                        if debug:
                                            eprint(
                                                f"Remove division at measure {new_number}, part {ib + 1}"  # noqa: E501
                                            )
                                for key in get_children_of_type(attrib, "key"):
                                    if in_force.get("Key") == key_attributes(key):
                                        if debug:
                                            eprint(
                                                f"Remove key at measure {new_number} part {ib + 1}"
                                            )
                                        attrib.remove(key)
                                for xtime in get_children_of_type(attrib, "time"):
                                    if in_force.get("Time") == time_attributes(xtime):
                                        attrib.remove(xtime)
                                        if debug:
                                            eprint(
                                                f"Remove time at measure {new_number} part {ib + 1}"
                                            )
                                for clef in get_children_of_type(attrib, "clef"):
                                    if in_force.get("Clef") == clef_attributes(clef):
                                        attrib.remove(clef)
                                        if debug:
                                            eprint(
                                                f"Remove clef at measure {new_number} part {ib + 1}"
                                            )
                        measure.set("number", new_number)
                        part1.append(measure)
                        # Track what is in force after this measure, so the next file is
                        # compared with the previous page and not with the first one.
                        update_attributes_in_force(last_parts_attributes[ib], measure)
                        if debug:
                            eprint(f"Added measure {new_number}, part {ib + 1}")
                    current_len = len(get_children_of_type(part1, "measure"))
                ib += 1
            ip += 1
    return (
        m,
        len(sorted_list),
        len(get_children_of_type(part1, "measure")) if part1 is not None else 0,
    )


def get_file_list(
    concat: tuple[str],
    debug=False,
) -> list[str]:
    # get the list of files
    sorted_list = []

    for pattern in concat:
        if not Path(pattern).suffix:
            pattern += "*.musicxml"  # noqa: PLW2901

        matched_files = list(glob(pattern))

        if len(matched_files) == 0 and debug:
            eprint(f"No file found for {pattern}")

        for fichier in matched_files:
            if not os.path.exists(fichier):
                if debug:
                    eprint(f"The file {fichier} does not exist.")
                return None
            if os.path.isdir(fichier):
                if debug:
                    eprint(f"{fichier} is a directory.")
                return None

            sorted_list.append(fichier)
    if len(sorted_list) == 0:
        eprint(f"No files found for {concat}")
        return None
    return sorted_list
