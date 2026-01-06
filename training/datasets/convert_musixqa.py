import json
import multiprocessing
import os
import re
from fractions import Fraction
from pathlib import Path

import cv2

from homr.download_utils import download_file, untar_file
from homr.simple_logging import eprint
from homr.staff_parsing import add_image_into_tr_omr_canvas
from homr.transformer.vocabulary import EncodedSymbol, Vocabulary
from training.datasets.convert_grandstaff import distort_image
from training.datasets.staff_detection_ideal import main as detect_staff

script_location = os.path.dirname(os.path.realpath(__file__))
git_root = Path(script_location).parent.parent.absolute()
dataset_root = os.path.join(git_root, "datasets")
musixqa_root = os.path.join(dataset_root, "musixqa")
musixqa_images = os.path.join(musixqa_root, "images")
musixqa_meta_data = os.path.join(musixqa_root, "metadata.json")
musixqa_homr_root = os.path.join(dataset_root, "musixqa_homr")
musixqa_index = os.path.join(musixqa_homr_root, "index.txt")

# Mapping of major keys to circle-of-fifths values
KEY_TO_FIFTHS = {
    "C Major": 0,
    "G Major": 1,
    "D Major": 2,
    "A Major": 3,
    "E Major": 4,
    "B Major": 5,
    "F# Major": 6,
    "C# Major": 7,
    "F Major": -1,
    "Bb Major": -2,
    "Eb Major": -3,
    "Ab Major": -4,
    "Db Major": -5,
    "Gb Major": -6,
    "Cb Major": -7,
}


def parse_pitch(pitch_str: str) -> tuple[str, str]:
    if not pitch_str or pitch_str == "rest":
        return ".", "."
    match = re.match(r"([A-G])([#|b]*)([0-9])", pitch_str)
    if not match:
        return pitch_str, "."
    pitch_name = match.group(1) + match.group(3)
    lift = match.group(2) or "_"
    return pitch_name, lift


def convert_duration(duration_str: str) -> str:
    frac = Fraction(duration_str)
    for base in [1, 2, 4, 8, 16, 32, 64]:
        base_frac = Fraction(1, base)
        if frac == base_frac:
            return str(base)
        if frac == base_frac * Fraction(3, 2):
            return f"{base}."
        if frac == base_frac * Fraction(7, 4):
            return f"{base}.."
    raise ValueError(f"Unsupported duration: {duration_str}")


def interleave_staves(bars: list[dict], is_last_system: bool = False) -> list[list[EncodedSymbol]]:
    all_symbols: list = []
    # Track which pitches have an active tie outgoing (the next note should have tieStop)
    active_ties: dict = {}  # (staff_name, pitch) -> bool

    for bar_idx, bar_data in enumerate(bars):
        staves = bar_data["staves"]
        timed_events = []
        staff_keys = sorted(staves.keys(), reverse=True)

        # Add repeatStart if present
        if bar_data.get("repeat") == "start":
            # If the previous symbol was a standard barline,
            # remove it because repeatStart implies a barline
            if all_symbols and str(all_symbols[-1][0]).startswith("barline"):
                all_symbols.pop()
            all_symbols.append([EncodedSymbol("repeatStart")])

        for staff_name in staff_keys:
            if staff_name not in staves:
                continue
            offset = Fraction(0)
            position = "upper"

            for event in staves[staff_name]:
                dur_str = event["duration"]
                dur_kern = convert_duration(dur_str)
                articulations = []

                if "pitch" in event:
                    pitch_name, lift = parse_pitch(event["pitch"])

                    tie_key = (staff_name, event["pitch"])
                    has_active_tie = active_ties.get(tie_key, False)
                    has_outgoing_tie = event.get("tie", False)

                    if has_active_tie and has_outgoing_tie:
                        articulations.append("tieStart")
                        articulations.append("tieStop")
                    elif has_active_tie:
                        articulations.append("tieStop")
                    elif has_outgoing_tie:
                        articulations.append("tieStart")

                    active_ties[tie_key] = has_outgoing_tie

                    art_str = "_".join(sorted(articulations)) if articulations else "_"
                    sym = EncodedSymbol(f"note_{dur_kern}", pitch_name, lift, art_str, position)
                else:
                    sym = EncodedSymbol(f"rest_{dur_kern}", ".", ".", ".", position)

                timed_events.append((offset, sym))
                offset += Fraction(dur_str)

        events_by_offset: dict = {}
        for offset, sym in timed_events:
            events_by_offset.setdefault(offset, []).append(sym)

        sorted_offsets = sorted(events_by_offset.keys())
        for offset in sorted_offsets:
            all_symbols.append(events_by_offset[offset])

        # Add barline or repeatEnd
        # The final bar line of an image is a bolddoublebarline
        # Meaning: Only if it is the LAST bar of the LAST system (End of Piece/Image).
        is_last_in_system = bar_idx == len(bars) - 1

        if bar_data.get("repeat") == "end":
            all_symbols.append([EncodedSymbol("repeatEnd")])
        elif is_last_in_system and is_last_system:
            all_symbols.append([EncodedSymbol("bolddoublebarline")])
        else:
            all_symbols.append([EncodedSymbol("barline")])

    return all_symbols

    return all_symbols


vocab = Vocabulary()


def validate_tokens(token_str: str) -> None:
    tokens = [
        t.strip()
        for t in token_str.replace(",", " ").split(" ")
        if t.strip() and t.strip() != "&" and t.strip() != "."
    ]
    for t in tokens:
        # Check rhythm, pitch, lift, articulation, position
        # This is a bit simplified check as Homr tokens are multi-branch
        # But we can at least check if the rhythm part is in vocab.rhythm
        parts = t.split(" ")
        rhythm = parts[0]
        if rhythm not in vocab.rhythm:
            # Try cleaning it if it's like note_4.
            if rhythm not in vocab.rhythm:
                raise ValueError(f"Invalid rhythm token: {rhythm}")


def convert_piece_to_homr(
    piece_data: dict,
    bars_subset: list[dict],
    is_first_system: bool = False,
    is_last_system: bool = False,
) -> str:
    tokens = []
    has_bass = any("bass" in b["staves"] for b in bars_subset)
    tokens.append(str(EncodedSymbol("clef_F4" if has_bass else "clef_G2", "_", "_", "_", "upper")))

    fifths = KEY_TO_FIFTHS.get(piece_data["key"], 0)
    tokens.append(str(EncodedSymbol(f"keySignature_{fifths}")))

    # timeSignatures only appear on the first system
    if is_first_system:
        time_sig = piece_data["time_signature"]
        denominator = time_sig.split("/")[-1]
        tokens.append(str(EncodedSymbol(f"timeSignature/{denominator}")))

    chords = interleave_staves(bars_subset, is_last_system=is_last_system)
    for chord in chords:
        if len(chord) > 1:
            tokens.append("&".join(str(s) for s in chord))
        else:
            tokens.append(str(chord[0]))

    result = "\n".join(tokens)
    # validate_tokens(result) # Optional: can be slow
    return result


def convert_musixqa(only_recreate_token_files: bool = False) -> None:
    if not os.path.exists(musixqa_root):
        eprint("Downloading MusiXQA...")
        os.makedirs(musixqa_root, exist_ok=True)
        download_file(
            "https://huggingface.co/datasets/puar-playground/MusiXQA/resolve/main/images.tar?download=true",
            os.path.join(musixqa_root, "images.tar"),
        )
        untar_file(os.path.join(musixqa_root, "images.tar"), musixqa_root)
        download_file(
            "https://huggingface.co/datasets/puar-playground/MusiXQA/resolve/main/metadata.json?download=true",
            musixqa_meta_data,
        )

    if not os.path.exists(musixqa_homr_root):
        os.makedirs(musixqa_homr_root)

    with open(musixqa_meta_data, "r") as f:
        metadata = json.load(f)

    image_files = sorted([f for f in os.listdir(musixqa_images) if f.endswith(".png")])
    total_count = len(image_files)
    processed_count = 0
    success_count = 0

    with open(musixqa_index, "w") as f_index:
        for image_name in image_files:
            piece_id = image_name.replace(".png", "")
            if piece_id not in metadata:
                continue

            piece_data = metadata[piece_id]

            # Grandstaff check (Global): Ignore grandstaff pieces silently
            is_grandstaff_piece = any(
                "treble" in b["staves"] and "bass" in b["staves"] for b in piece_data["bars"]
            )
            if is_grandstaff_piece:
                continue

            img_path = os.path.join(musixqa_images, image_name)

            try:
                staff_groups, group_barlines = detect_staff(img_path)
                total_bars_detected = sum(len(bls) - 1 for bls in group_barlines)
                total_bars_meta = len(piece_data["bars"])

                # Validation: measure count match
                if total_bars_detected != total_bars_meta:
                    eprint(
                        f"Warning: Measure count mismatch for {piece_id}: "
                        f"detected {total_bars_detected}, meta {total_bars_meta}"
                    )
                    continue

                img = cv2.imread(img_path)
                if img is None:
                    eprint(f"Error reading image: {img_path}")
                    continue

                current_bar_idx = 0
                for i, (group, barlines) in enumerate(
                    zip(staff_groups, group_barlines, strict=True)
                ):
                    t, b, left, right, lt, lb = group
                    num_bars = len(barlines) - 1
                    if num_bars <= 0:
                        continue

                    bars_subset = piece_data["bars"][current_bar_idx : current_bar_idx + num_bars]
                    current_bar_idx += num_bars

                    # Grandstaff check (System): Should be redundant if global check works
                    # but keep for safety
                    is_grandstaff = any(
                        "treble" in b["staves"] and "bass" in b["staves"] for b in bars_subset
                    )
                    if is_grandstaff:
                        # The layout of the grand staff examples in this dataset is atypical.
                        # In standard music engraving practice, events that are vertically
                        # aligned represent simultaneity: noteheads, rests, and chords
                        # sounding at the same rhythmic position are placed in the same
                        # vertical column across staves. In these examples, that convention
                        # is not followed. Notes that are rhythmically simultaneous are not
                        # vertically aligned, which disrupts the visual representation of
                        # metric and rhythmic structure. This suggests that the notation
                        # was likely generated algorithmically rather than engraved according
                        # to established conventions, and that the data may be entirely
                        # synthetic rather than derived from human-edited scores.
                        continue

                    is_last_system = i == len(staff_groups) - 1
                    system_tokens = convert_piece_to_homr(
                        piece_data,
                        bars_subset,
                        is_first_system=(i == 0),
                        is_last_system=is_last_system,
                    )

                    # Save cropped image and tokens
                    system_id = f"{piece_id}_{i}"
                    crop = img[
                        max(0, t - 10) : min(img.shape[0], b + 10),
                        max(0, left - 10) : min(img.shape[1], right + 10),
                    ]
                    # Homr expects 128px height or similar, and distorted?
                    # add_image_into_tr_omr_canvas and distort_image from convert_grandstaff
                    crop_ready = add_image_into_tr_omr_canvas(crop)
                    crop_distorted = distort_image(crop_ready)

                    out_img_path = os.path.join(musixqa_homr_root, f"{system_id}.jpg")
                    out_tokens_path = os.path.join(musixqa_homr_root, f"{system_id}.tokens")

                    cv2.imwrite(out_img_path, crop_distorted)
                    with open(out_tokens_path, "w") as f_out:
                        f_out.write(system_tokens)

                    # Write to index
                    rel_img_path = os.path.relpath(out_img_path, git_root)
                    rel_tokens_path = os.path.relpath(out_tokens_path, git_root)
                    f_index.write(f"{rel_img_path},{rel_tokens_path}\n")

                success_count += 1
            except Exception as e:
                eprint(f"Error processing {piece_id}: {e}")

            processed_count += 1
            if processed_count % 50 == 0:
                eprint(f"Processed {processed_count}/{total_count}, Success: {success_count}")

    eprint(f"Finished. Success rate: {success_count/processed_count:.1%}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    convert_musixqa()
