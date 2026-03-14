import os
import re
import sys
import tempfile
import threading
import types
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from homr.simple_logging import eprint
from homr.transformer.vocabulary import EncodedSymbol, empty, nonote
from homr.type_definitions import NDArray

# PaddleOCR 3.x imports `modelscope` via `paddlex` at import time.
# In this project that path can pull in torch and fail due to CUDA/NCCL ABI
# conflicts unrelated to OCR. We provide a minimal stub so PaddleOCR can load
# and continue using its other model hosters.
if "modelscope" not in sys.modules:
    modelscope_stub = types.ModuleType("modelscope")

    def _modelscope_snapshot_download(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("ModelScope download is disabled in this runtime.")

    modelscope_stub.snapshot_download = _modelscope_snapshot_download
    sys.modules["modelscope"] = modelscope_stub

from paddleocr import PaddleOCR

@dataclass(frozen=True)
class LyricCandidate:
    text: str
    x: float
    y: float
    confidence: float = 1.0
    line_height: float = 0.0


@dataclass(frozen=True)
class _NoteCandidate:
    symbol_index: int
    measure: int
    x: float
    y: float
    position: str


@dataclass(frozen=True)
class LyricAssignment:
    lyric: LyricCandidate
    symbol_index: int
    verse: int


_reader: Any | None = None
_reader_backend = "uninitialized"
_backend_preference = "auto"
_reader_lock = threading.Lock()

# Lyrics OCR Paddle configuration.
_PADDLE_OCR_VERSIONS = ["PP-OCRv5", "PP-OCRv4", "PP-OCRv3"]
_PADDLE_OCR_LANG = "en"
_PADDLE_OCR_DET_LIMIT_SIDE_LEN = 768


def set_lyrics_ocr_backend(backend: str) -> None:
    normalized = backend.lower().strip()
    if normalized not in ("auto", "paddle", "rapid"):
        raise ValueError("Unsupported lyrics OCR backend: " + backend)
    if normalized == "rapid":
        raise ValueError("RapidOCR backend is disabled for lyrics OCR. Use 'paddle' or 'auto'.")

    global _backend_preference  # noqa: PLW0603
    global _reader  # noqa: PLW0603
    global _reader_backend  # noqa: PLW0603

    if normalized == _backend_preference:
        return

    _backend_preference = normalized
    _reader = None
    _reader_backend = "uninitialized"


@lru_cache(maxsize=1)
def _english_word_set() -> set[str]:
    candidates = [Path("/usr/share/dict/words")]
    words: set[str] = set()
    for path in candidates:
        if not path.exists():
            continue
        try:
            with path.open(encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    word = line.strip().lower()
                    if word:
                        words.add(word)
        except Exception as ex:
            eprint("Skipping dictionary file", str(path), ex)
    return words


def _initialize_reader() -> None:
    global _reader  # noqa: PLW0603
    global _reader_backend  # noqa: PLW0603
    # if _reader is not None:
    #     return
    # with _reader_lock:
    #     if _reader is None:
    #         errors: list[str] = []
    #         if _backend_preference in ("auto", "paddle"):
    #             try:
    #                 paddleocr_home = os.path.join(tempfile.gettempdir(), "homr-paddleocr-cache")
    #                 os.environ["PADDLEOCR_HOME"] = paddleocr_home
    #                 os.environ["PADDLE_OCR_BASE_DIR"] = paddleocr_home
    #                 os.makedirs(paddleocr_home, exist_ok=True)
    #                 paddlex_home = os.path.join(tempfile.gettempdir(), "homr-paddlex-cache")
    #                 os.environ["PADDLE_PDX_CACHE_HOME"] = paddlex_home
    #                 os.makedirs(paddlex_home, exist_ok=True)
    #                 os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

            # from paddleocr import PaddleOCR  # type: ignore[import-not-found]

    #                 paddle_errors: list[str] = []
    #                 for version in _PADDLE_OCR_VERSIONS:
    #                     try:
    paddle_errors: list[str] = []
    for version in _PADDLE_OCR_VERSIONS:
        paddle_kwargs: dict[str, Any] = {
            "lang": _PADDLE_OCR_LANG,
            "ocr_version": version,
            "use_textline_orientation": True,
            "use_gpu": True,
            "text_det_limit_side_len": _PADDLE_OCR_DET_LIMIT_SIDE_LEN,
        }
        try:
            _reader = PaddleOCR(**paddle_kwargs)
            break
        except (TypeError, AssertionError) as ex:
            paddle_errors.append(f"{version}: {ex}")
    if _reader is None:
        # PaddleOCR 2.x compatibility: this version does not support
        # the newer constructor arguments used in PaddleOCR 3.x.
        legacy_kwargs: dict[str, Any] = {
            "lang": _PADDLE_OCR_LANG,
            "use_angle_cls": True,
            "use_gpu": True,
            "det_limit_side_len": _PADDLE_OCR_DET_LIMIT_SIDE_LEN,
        }
        try:
            _reader = PaddleOCR(**legacy_kwargs)
        except Exception as ex:
            paddle_errors.append("legacy: " + str(ex))
            raise RuntimeError("Failed to initialize PaddleOCR: " + " | ".join(paddle_errors)) from ex
    _reader_backend = "paddle"
            #                 eprint(f"Lyrics OCR backend: paddle ({version})")
            #                 return
            #             except TypeError:
            #                 # PaddleOCR 2.x compatibility path.
            #                 legacy_kwargs: dict[str, Any] = {
            #                     "lang": _PADDLE_OCR_LANG,
            #                     "ocr_version": version,
            #                     "use_angle_cls": True,
            #                     "det_limit_side_len": _PADDLE_OCR_DET_LIMIT_SIDE_LEN,
            #                 }
            #                 _reader = PaddleOCR(**legacy_kwargs)
            #                 _reader_backend = "paddle"
            #                 eprint(f"Lyrics OCR backend: paddle ({version})")
            #                 return
            #             except Exception as ex:
            #                 paddle_errors.append(f"{version}: {ex}")
            #         raise RuntimeError(" | ".join(paddle_errors))
            #     except Exception as ex:
            #         errors.append("paddle: " + str(ex))

            # if _backend_preference == "rapid":
            #     errors.append("rapid: disabled by configuration")

            # raise RuntimeError("Failed to initialize lyric OCR backend: " + "; ".join(errors))


def _is_bbox(candidate: Any) -> bool:
    if isinstance(candidate, np.ndarray):
        if candidate.ndim != 2 or candidate.shape[0] < 4 or candidate.shape[1] < 2:
            return False
        return np.issubdtype(candidate.dtype, np.number)
    if not isinstance(candidate, list | tuple):
        return False
    if len(candidate) < 4:
        return False
    return all(
        isinstance(point, list | tuple)
        and len(point) >= 2
        and isinstance(point[0], int | float)
        and isinstance(point[1], int | float)
        for point in candidate
    )


def _bbox_to_list(bbox: Any) -> list[list[float]]:
    if isinstance(bbox, np.ndarray):
        return np.asarray(bbox, dtype=np.float32).tolist()
    return [[float(point[0]), float(point[1])] for point in bbox]


def _extract_ocr_entries(node: Any, output: list[tuple[list[list[float]], str, float]]) -> None:
    if isinstance(node, tuple):
        node = list(node)

    if isinstance(node, dict):
        _extract_ocr_entries_from_mapping(node, output)
        for value in node.values():
            _extract_ocr_entries(value, output)
        return

    if not isinstance(node, list):
        if hasattr(node, "to_dict") and callable(node.to_dict):
            try:
                _extract_ocr_entries(node.to_dict(), output)
                return
            except Exception as ex:
                eprint("Failed to normalize OCR result object:", ex)
        if hasattr(node, "__dict__"):
            _extract_ocr_entries(vars(node), output)
        return

    if (
        len(node) == 3
        and _is_bbox(node[0])
        and isinstance(node[1], str)
        and isinstance(node[2], int | float)
    ):
        output.append((_bbox_to_list(node[0]), node[1], float(node[2])))
        return

    if (
        len(node) == 2
        and _is_bbox(node[0])
        and isinstance(node[1], list | tuple)
        and len(node[1]) >= 2
        and isinstance(node[1][0], str)
        and isinstance(node[1][1], int | float)
    ):
        output.append((_bbox_to_list(node[0]), node[1][0], float(node[1][1])))
        return

    for child in node:
        _extract_ocr_entries(child, output)


def _extract_ocr_entries_from_mapping(
    node: dict[Any, Any], output: list[tuple[list[list[float]], str, float]]
) -> None:
    boxes = node.get("dt_polys")
    texts = node.get("rec_texts", node.get("rec_text"))
    scores = node.get("rec_scores", node.get("rec_score"))
    if boxes is None or texts is None or scores is None:
        return

    if isinstance(boxes, np.ndarray):
        boxes = boxes.tolist()
    if isinstance(texts, np.ndarray):
        texts = texts.tolist()
    if isinstance(scores, np.ndarray):
        scores = scores.tolist()

    if not isinstance(boxes, list | tuple):
        return
    if not isinstance(texts, list | tuple):
        texts = [texts]
    if not isinstance(scores, list | tuple):
        scores = [scores]

    count = min(len(boxes), len(texts), len(scores))
    for i in range(count):
        bbox = boxes[i]
        text = texts[i]
        score = scores[i]
        if isinstance(text, np.ndarray):
            text = text.tolist()
        if isinstance(score, np.ndarray):
            score = score.tolist()
        if isinstance(text, list | tuple):
            text = text[0] if len(text) > 0 else ""
        if isinstance(score, list | tuple):
            score = score[0] if len(score) > 0 else 0.0
        if _is_bbox(bbox) and isinstance(text, str) and isinstance(score, int | float):
            output.append((_bbox_to_list(bbox), text, float(score)))


def _normalize_ocr_results(raw: Any) -> list[tuple[list[list[float]], str, float]]:
    output: list[tuple[list[list[float]], str, float]] = []
    _extract_ocr_entries(raw, output)
    return output


def _run_ocr(image: NDArray) -> list[tuple[list[list[float]], str, float]]:
    _initialize_reader()
    if _reader is None:
        return []

    if _reader_backend != "paddle":
        raise RuntimeError("Unexpected lyrics OCR backend: " + str(_reader_backend))

    image_for_ocr = _normalize_ocr_image(image)

    if hasattr(_reader, "predict"):
        raw = _reader.predict(image_for_ocr)
    else:
        raw = _reader.ocr(image_for_ocr, cls=True)

    return _normalize_ocr_results(raw)


def _normalize_ocr_image(image: NDArray) -> NDArray:
    output = np.asarray(image)
    if output.ndim == 2:
        output = np.stack([output, output, output], axis=-1)
    elif output.ndim == 3 and output.shape[2] == 1:
        output = np.repeat(output, 3, axis=2)
    elif output.ndim == 3 and output.shape[2] > 3:
        output = output[:, :, :3]

    if output.dtype != np.uint8:
        if np.issubdtype(output.dtype, np.floating):
            if float(output.max(initial=0.0)) <= 1.0:
                output = np.clip(output * 255.0, 0, 255)
            else:
                output = np.clip(output, 0, 255)
        else:
            output = np.clip(output, 0, 255)
        output = output.astype(np.uint8)

    return np.ascontiguousarray(output)


def _to_coordinates(symbol: EncodedSymbol) -> tuple[float, float] | None:
    if symbol.coordinates is None:
        return None
    try:
        values = np.asarray(symbol.coordinates, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if values.size < 2:
        return None
    x, y = float(values[0]), float(values[1])
    if not np.isfinite(x) or not np.isfinite(y):
        return None
    return x, y


def _is_measure_boundary(rhythm: str) -> bool:
    return "barline" in rhythm or rhythm in ("repeatStart", "repeatEnd", "repeatEndStart")


def _is_primary_note(symbol: EncodedSymbol, previous_rhythm: str) -> bool:
    if previous_rhythm == "chord":
        return False
    if symbol.rhythm.startswith("note"):
        return symbol.pitch not in (nonote, empty)
    if symbol.rhythm.startswith("rest"):
        return True
    return False


def _collect_note_candidates(symbols: list[EncodedSymbol]) -> list[_NoteCandidate]:
    current_measure = 0
    previous_rhythm = ""
    notes: list[_NoteCandidate] = []
    for i, symbol in enumerate(symbols):
        center = _to_coordinates(symbol)
        if center is not None and _is_primary_note(symbol, previous_rhythm):
            notes.append(
                _NoteCandidate(
                    symbol_index=i,
                    measure=current_measure,
                    x=center[0],
                    y=center[1],
                    position=symbol.position,
                )
            )
        if _is_measure_boundary(symbol.rhythm):
            current_measure += 1
        previous_rhythm = symbol.rhythm
    return notes


def _determine_measure(lyric: LyricCandidate, note_candidates: list[_NoteCandidate]) -> int | None:
    note_ranges: dict[int, tuple[float, float]] = {}
    for note in note_candidates:
        if note.measure in note_ranges:
            current = note_ranges[note.measure]
            note_ranges[note.measure] = (min(current[0], note.x), max(current[1], note.x))
        else:
            note_ranges[note.measure] = (note.x, note.x)

    if not note_ranges:
        return None

    closest_measure: int | None = None
    closest_distance = float("inf")
    for measure, (x_min, x_max) in note_ranges.items():
        if x_min <= lyric.x <= x_max:
            distance = 0.0
        elif lyric.x < x_min:
            distance = x_min - lyric.x
        else:
            distance = lyric.x - x_max
        if distance < closest_distance:
            closest_distance = distance
            closest_measure = measure

    return closest_measure


def assign_lyrics_to_symbols(
    symbols: list[EncodedSymbol], lyrics: list[LyricCandidate]
) -> list[LyricAssignment]:
    assignments: list[LyricAssignment] = []
    if len(symbols) == 0 or len(lyrics) == 0:
        return assignments

    note_candidates = _collect_note_candidates(symbols)
    if len(note_candidates) == 0:
        return assignments

    by_measure: dict[int, list[_NoteCandidate]] = {}
    for note in note_candidates:
        by_measure.setdefault(note.measure, []).append(note)

    for verse, lyric_line in enumerate(_group_lyrics_into_lines(lyrics), start=1):
        for lyric_candidate in sorted(lyric_line, key=lambda candidate: candidate.x):
            measure = _determine_measure(lyric_candidate, note_candidates)
            if measure is None:
                continue
            candidates = by_measure.get(measure, note_candidates)
            candidates = _prefer_staff_for_lyrics(candidates, lyric_candidate.y)

            best_note = min(
                candidates,
                key=lambda note: _distance_to_note(
                    symbols, note, lyric_candidate.x, lyric_candidate.y, verse
                ),
            )
            symbol = symbols[best_note.symbol_index]
            _set_symbol_lyric(symbol, verse, lyric_candidate.text)
            assignments.append(
                LyricAssignment(
                    lyric=lyric_candidate,
                    symbol_index=best_note.symbol_index,
                    verse=verse,
                )
            )

    return assignments


def _distance_to_note(
    symbols: list[EncodedSymbol], note: _NoteCandidate, lyric_x: float, lyric_y: float, verse: int
) -> float:
    score = (note.x - lyric_x) ** 2 + 0.15 * (note.y - lyric_y) ** 2
    return score


def _group_lyrics_into_lines(lyrics: list[LyricCandidate]) -> list[list[LyricCandidate]]:
    if len(lyrics) == 0:
        return []
    if len(lyrics) == 1:
        return [lyrics]

    ordered = sorted(lyrics, key=lambda lyric: lyric.y)
    ys = np.array([lyric.y for lyric in ordered], dtype=np.float32)
    diffs = np.diff(ys)
    if len(diffs) == 0:
        return [sorted(ordered, key=lambda lyric: lyric.x)]

    positive_diffs = diffs[diffs > 0]
    median_gap = float(np.median(positive_diffs)) if len(positive_diffs) > 0 else 0.0
    split_threshold = 8.0 if len(ordered) < 8 else max(8.0, median_gap * 6.0)
    split_indices = [index + 1 for index, gap in enumerate(diffs) if float(gap) >= split_threshold]

    if len(split_indices) == 0:
        return [sorted(ordered, key=lambda lyric: lyric.x)]

    groups: list[list[LyricCandidate]] = []
    start = 0
    for split in split_indices:
        groups.append(sorted(ordered[start:split], key=lambda lyric: lyric.x))
        start = split
    groups.append(sorted(ordered[start:], key=lambda lyric: lyric.x))

    # Ignore isolated outliers that would create a one-token "verse".
    merged: list[list[LyricCandidate]] = []
    for group in groups:
        if len(group) == 1 and len(merged) > 0:
            merged[-1].extend(group)
            merged[-1].sort(key=lambda lyric: lyric.x)
        else:
            merged.append(group)
    if len(merged) > 1 and len(merged[-1]) == 1:
        merged[-2].extend(merged[-1])
        merged[-2].sort(key=lambda lyric: lyric.x)
        merged = merged[:-1]

    return merged


def _get_symbol_lyric_verses(symbol: EncodedSymbol) -> dict[int, str]:
    verses = {}
    dynamic = getattr(symbol, "lyric_verses", None)
    if isinstance(dynamic, dict):
        for verse, text in dynamic.items():
            if isinstance(verse, int) and isinstance(text, str) and text.strip():
                verses[verse] = text.strip()
    if symbol.lyric:
        verses.setdefault(1, symbol.lyric.strip())
    return verses


def _get_symbol_lyric(symbol: EncodedSymbol, verse: int) -> str | None:
    return _get_symbol_lyric_verses(symbol).get(verse)


def _set_symbol_lyric(symbol: EncodedSymbol, verse: int, text: str) -> None:
    verses = _get_symbol_lyric_verses(symbol)
    if verse in verses:
        verses[verse] = f"{verses[verse]} {text}"
    else:
        verses[verse] = text
    symbol.lyric_verses = verses  # type: ignore[attr-defined]
    symbol.lyric = verses.get(1)


def _prefer_staff_for_lyrics(
    candidates: list[_NoteCandidate], lyric_y: float
) -> list[_NoteCandidate]:
    upper = [candidate for candidate in candidates if candidate.position == "upper"]
    lower = [candidate for candidate in candidates if candidate.position == "lower"]
    if len(upper) == 0 or len(lower) == 0:
        return candidates

    lower_ys = np.array([candidate.y for candidate in lower], dtype=np.float32)
    lower_min = float(np.percentile(lower_ys, 15))
    lower_mid = float(np.percentile(lower_ys, 50))
    tolerance = 2.0

    # Default behavior for grand staff lyrics: attach to upper staff.
    # Only allow lower-staff lyrics if text is below lower-note positions.
    if lyric_y <= lower_mid + tolerance:
        return upper
    if lyric_y > lower_min + tolerance:
        return lower
    return upper


def _between_staff_y_bounds(symbols: list[EncodedSymbol]) -> tuple[float, float] | None:
    note_candidates = _collect_note_candidates(symbols)
    upper_ys = np.array(
        [candidate.y for candidate in note_candidates if candidate.position == "upper"],
        dtype=np.float32,
    )
    lower_ys = np.array(
        [candidate.y for candidate in note_candidates if candidate.position == "lower"],
        dtype=np.float32,
    )
    if len(upper_ys) == 0 or len(lower_ys) == 0:
        return None

    upper_edge = float(np.percentile(upper_ys, 85))
    lower_edge = float(np.percentile(lower_ys, 15))
    if upper_edge >= lower_edge:
        upper_edge = float(np.percentile(upper_ys, 50))
        lower_edge = float(np.percentile(lower_ys, 50))
        if upper_edge >= lower_edge:
            return None
    return upper_edge, lower_edge


def _filter_lyrics_between_staves(
    symbols: list[EncodedSymbol], lyrics: list[LyricCandidate]
) -> list[LyricCandidate]:
    bounds = _between_staff_y_bounds(symbols)
    if bounds is not None:
        upper_edge, lower_edge = bounds
        margin = 2.0
        return [
            lyric
            for lyric in lyrics
            if upper_edge - margin <= lyric.y <= lower_edge + margin
        ]

    # Fallback: when TrOMR misses one staff label (e.g. no `lower` positions),
    # keep only a narrow band below the detected note band.
    note_candidates = _collect_note_candidates(symbols)
    if len(note_candidates) == 0:
        return []

    upper_ys = np.array(
        [candidate.y for candidate in note_candidates if candidate.position == "upper"],
        dtype=np.float32,
    )
    lower_ys = np.array(
        [candidate.y for candidate in note_candidates if candidate.position == "lower"],
        dtype=np.float32,
    )

    if len(upper_ys) > 0 and len(lower_ys) == 0:
        anchor = float(np.percentile(upper_ys, 85))
        return [lyric for lyric in lyrics if anchor + 8.0 <= lyric.y <= anchor + 90.0]

    if len(lower_ys) > 0 and len(upper_ys) == 0:
        anchor = float(np.percentile(lower_ys, 15))
        return [lyric for lyric in lyrics if anchor - 90.0 <= lyric.y <= anchor - 8.0]

    return []


def _split_joined_token(token: str) -> list[str]:
    result = [token]
    lower = token.lower()
    if len(lower) >= 6 and lower.isalpha():
        dictionary = _english_word_set()
        if len(dictionary) > 0 and lower not in dictionary:
            n = len(lower)
            best_score = [float("-inf")] * (n + 1)
            best_parts: list[list[str] | None] = [None] * (n + 1)
            best_score[0] = 0.0
            best_parts[0] = []

            max_word_len = 16
            for start in range(n):
                if best_parts[start] is None:
                    continue
                for end in range(start + 1, min(n, start + max_word_len) + 1):
                    piece = lower[start:end]
                    if piece not in dictionary:
                        continue
                    score = best_score[start] + len(piece) * len(piece)
                    if score > best_score[end]:
                        best_score[end] = score
                        best_parts[end] = [*best_parts[start], piece]

            pieces = best_parts[n]
            is_valid_split = pieces is not None and len(pieces) > 1
            if is_valid_split and pieces is not None:
                is_valid_split = len(pieces) <= 3 and not any(
                    len(part) == 1 and part not in ("a", "i") for part in pieces
                )
            if is_valid_split:
                split_result: list[str] = []
                offset = 0
                for index, piece in enumerate(pieces):
                    original = token[offset : offset + len(piece)]
                    if index == 0 and token[:1].isupper():
                        original = piece.capitalize()
                    else:
                        original = original.lower()
                    split_result.append(original)
                    offset += len(piece)
                result = split_result

    return result


def _note_y_values(symbols: list[EncodedSymbol]) -> list[float]:
    y_values = []
    for symbol in symbols:
        if symbol.rhythm.startswith("note"):
            center = _to_coordinates(symbol)
            if center is not None:
                y_values.append(center[1])
    return y_values


def _estimate_lyric_region(staff_image: NDArray, symbols: list[EncodedSymbol]) -> tuple[int, int]:
    y_values = _note_y_values(symbols)
    height = staff_image.shape[0]

    if len(y_values) == 0:
        top = int(height * 0.55)
        return top, min(height, top + int(height * 0.2))

    sorted_ys = np.sort(np.array(y_values))
    if len(sorted_ys) >= 8:
        gaps = np.diff(sorted_ys)
        largest_gap_index = int(np.argmax(gaps))
        largest_gap = float(gaps[largest_gap_index])
        # In grand staff images, the lyrics are typically between upper and lower staff.
        if largest_gap >= 24.0:
            upper_end = float(sorted_ys[largest_gap_index])
            lower_start = float(sorted_ys[largest_gap_index + 1])
            top = max(int(upper_end + 2), 0)
            bottom = min(int(lower_start - 2), height)
            if bottom - top < 12:
                midpoint = int((upper_end + lower_start) / 2)
                half_height = 10
                top = max(midpoint - half_height, 0)
                bottom = min(midpoint + half_height, height)
            return top, bottom

    top = min(max(int(np.percentile(sorted_ys, 75) + 12), 0), height - 1)
    max_height = int(height * 0.22)
    return top, min(height, top + max_height)


def _tokenize_ocr_result(
    bbox: list[list[float]], text: str, confidence: float, y_offset: int
) -> list[LyricCandidate]:
    x_values = [point[0] for point in bbox]
    y_values = [point[1] for point in bbox]
    if len(x_values) == 0 or len(y_values) == 0:
        return []

    x_min = float(min(x_values))
    x_max = float(max(x_values))
    candidates = []
    raw_lines = [line.strip() for line in re.split(r"\r?\n+", text) if line.strip()]
    if len(raw_lines) == 0:
        raw_lines = [re.sub(r"\s+", " ", text).strip()]
    raw_lines = [line for line in raw_lines if line]
    if len(raw_lines) == 0:
        return []

    y_min = float(min(y_values))
    y_max = float(max(y_values))
    total_height = max(y_max - y_min, 1.0)
    line_band_height = total_height / len(raw_lines)

    for line_index, raw_line in enumerate(raw_lines):
        normalized_line = re.sub(r"\s+", " ", raw_line).strip()
        if normalized_line == "":
            continue

        cleaned_tokens = []
        for token in normalized_line.split(" "):
            cleaned = token.strip()
            if cleaned == "":
                continue
            if not any(char.isalpha() for char in cleaned):
                continue
            cleaned_tokens.extend(_split_joined_token(cleaned))

        if len(cleaned_tokens) == 0:
            continue

        line_y_center = y_min + (line_index + 0.5) * line_band_height + y_offset
        line_height = max(line_band_height, 1.0)

        if len(cleaned_tokens) == 1:
            candidates.append(
                LyricCandidate(
                    text=cleaned_tokens[0],
                    x=(x_min + x_max) / 2.0,
                    y=line_y_center,
                    confidence=confidence,
                    line_height=line_height,
                )
            )
            continue

        span = max(x_max - x_min, 1.0)
        step = span / len(cleaned_tokens)
        for token_index, token in enumerate(cleaned_tokens):
            candidates.append(
                LyricCandidate(
                    text=token,
                    x=x_min + (token_index + 0.5) * step,
                    y=line_y_center,
                    confidence=confidence,
                    line_height=line_height,
                )
            )
    return candidates


def _lyrics_from_ocr_results(
    symbols: list[EncodedSymbol],
    ocr_results: list[tuple[list[list[float]], str, float]],
    y_offset: int,
) -> list[LyricCandidate]:
    if len(ocr_results) == 0:
        return []

    min_confidence = 0.0
    lyrics: list[LyricCandidate] = []
    for bbox, text, confidence in ocr_results:
        if confidence < min_confidence:
            continue
        lyrics.extend(_tokenize_ocr_result(bbox, text, confidence, y_offset))
    return _filter_lyrics_between_staves(symbols, lyrics)


def detect_lyric_candidates(
    staff_image: NDArray, symbols: list[EncodedSymbol]
) -> list[LyricCandidate]:
    # Run OCR on the full dewarped staff image to avoid dropping lyrics in
    # narrow lyric crops.
    lyric_top = 0
    lyric_image = staff_image
    if lyric_image.size == 0:
        return []

    ocr_results = _run_ocr(lyric_image)
    return _lyrics_from_ocr_results(symbols, ocr_results, lyric_top)


def detect_and_assign_lyrics(
    staff_image: NDArray, symbols: list[EncodedSymbol]
) -> list[LyricAssignment]:
    lyrics = detect_lyric_candidates(staff_image, symbols)
    return assign_lyrics_to_symbols(symbols, lyrics)


def lyrics_by_measure_per_verse(symbols: list[EncodedSymbol]) -> dict[int, list[list[str]]]:
    verse_numbers: set[int] = set()
    for symbol in symbols:
        if symbol.rhythm.startswith(("note", "rest")):
            verse_numbers.update(_get_symbol_lyric_verses(symbol).keys())
    if len(verse_numbers) == 0:
        return {}

    verses = sorted(verse_numbers)
    grouped: dict[int, list[list[str]]] = {verse: [] for verse in verses}
    current_measure: dict[int, list[str]] = {verse: [] for verse in verses}

    for symbol in symbols:
        if symbol.rhythm.startswith(("note", "rest")):
            symbol_verses = _get_symbol_lyric_verses(symbol)
            for verse, measure_words in current_measure.items():
                text = symbol_verses.get(verse)
                if text:
                    measure_words.append(text)
        if _is_measure_boundary(symbol.rhythm):
            for verse, measure_words in current_measure.items():
                grouped[verse].append(list(measure_words))
                measure_words.clear()

    has_tail = any(len(measure_words) > 0 for measure_words in current_measure.values())
    has_no_measures = all(len(measures) == 0 for measures in grouped.values())
    if has_tail or has_no_measures:
        for verse, measure_words in current_measure.items():
            grouped[verse].append(list(measure_words))

    return grouped


def lyrics_by_measure(symbols: list[EncodedSymbol]) -> list[list[str]]:
    return lyrics_by_measure_per_verse(symbols).get(1, [])
