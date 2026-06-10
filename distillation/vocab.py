#!/usr/bin/env python3
"""
Build HOMR-factorized page targets for Track C student training.

This module uses HOMR's own token system as the source of truth for musical
symbols:

    homr.transformer.vocabulary.Vocabulary
    homr.transformer.vocabulary.EncodedSymbol

It does not build a separate flat vocabulary. It does not assign custom IDs to
complete symbol strings.

The training-manifest input should come from build_training_manifest.py and is
expected to contain:

    target_staff_token_sequences

Each staff sequence contains HOMR-style five-component symbol strings:

    rhythm pitch lift articulation position

Each staff sequence is encoded with HOMR's factorized component dictionaries:

    rhythm_ids
    pitch_ids
    lift_ids
    articulation_ids
    position_ids
    mask

For page-level training, staff boundaries are represented structurally in JSON:

    homr_target_staffs: [staff_0, staff_1, ...]

HOMR's page-level postprocessing uses EncodedSymbol("newline") between parsed
staff/system outputs before MusicXML generation. That separator is recorded in
metadata as a HOMR page postprocess symbol, but it is not encoded as a TrOMR
vocabulary token because it is not part of HOMR's transformer Vocabulary.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))

try:
    from homr.transformer.vocabulary import EncodedSymbol, Vocabulary
except Exception as exc:  # pragma: no cover - environment/setup error.
    raise RuntimeError(
        "Could not import HOMR vocabulary. Run this from the repository root "
        "with the HOMR package available on PYTHONPATH."
    ) from exc


TOKEN_FIELD_STAFF = "target_staff_token_sequences"
TOKEN_FIELD_FLAT = "target_flat_tokens"
TOKEN_FIELD_FLAT_WITH_LEGACY_BREAKS = "target_flat_tokens_with_staff_breaks"
HOMR_PAGE_NEWLINE_RHYTHM = "newline"
HOMR_PAGE_NEWLINE_TOKEN = "newline . . . ."


class TargetEncodingError(RuntimeError):
    """Raised when target manifests cannot be encoded using HOMR Vocabulary."""


def emit_event(payload: dict[str, Any], *, quiet: bool = False, stderr: bool = False) -> None:
    if quiet:
        return
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), file=sys.stderr if stderr else sys.stdout, flush=True)


@dataclass
class InvalidSymbol:
    manifest: str
    row_index: int
    line_number: int | None
    page_id: str | None
    score_id: str | None
    staff_index: int
    token_index: int
    token: str
    issues: list[str]

    def to_json(self) -> dict[str, Any]:
        return {
            "manifest": self.manifest,
            "row_index": self.row_index,
            "line_number": self.line_number,
            "score_id": self.score_id,
            "page_id": self.page_id,
            "staff_index": self.staff_index,
            "token_index": self.token_index,
            "token": self.token,
            "issues": self.issues,
        }


@dataclass
class ManifestStats:
    manifest: str
    rows: int = 0
    encoded_rows: int = 0
    staffs: int = 0
    page_newline_separators: int = 0
    symbols_without_bos_eos: int = 0
    symbols_with_bos_eos: int = 0
    max_staff_len_without_bos_eos: int = 0
    max_staff_len_with_bos_eos: int = 0
    token_source_counts: Counter[str] = field(default_factory=Counter)
    rhythm_counts: Counter[str] = field(default_factory=Counter)
    invalid_symbols: list[InvalidSymbol] = field(default_factory=list)

    def to_json(self, *, max_invalid_examples: int) -> dict[str, Any]:
        return {
            "manifest": self.manifest,
            "rows": self.rows,
            "encoded_rows": self.encoded_rows,
            "staffs": self.staffs,
            "page_newline_separators": self.page_newline_separators,
            "symbols_without_bos_eos": self.symbols_without_bos_eos,
            "symbols_with_bos_eos": self.symbols_with_bos_eos,
            "max_staff_len_without_bos_eos": self.max_staff_len_without_bos_eos,
            "max_staff_len_with_bos_eos": self.max_staff_len_with_bos_eos,
            "token_source_counts": dict(sorted(self.token_source_counts.items())),
            "unique_rhythms": len(self.rhythm_counts),
            "top_rhythms": self.rhythm_counts.most_common(25),
            "invalid_symbol_count": len(self.invalid_symbols),
            "invalid_symbol_examples": [
                item.to_json() for item in self.invalid_symbols[:max_invalid_examples]
            ],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Encode Track C page targets using HOMR's factorized "
            "Vocabulary/EncodedSymbol system."
        )
    )
    parser.add_argument(
        "--train-manifest",
        required=True,
        type=Path,
        help="Training split manifest JSONL, usually training_manifest.train.jsonl.",
    )
    parser.add_argument(
        "--validate-manifest",
        action="append",
        default=[],
        type=Path,
        help="Additional manifest to encode/validate. May be passed multiple times.",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help=(
            "Output summary JSON path. This records HOMR vocabulary sizes and "
            "target encoding/validation statistics."
        ),
    )
    parser.add_argument(
        "--encoded-out-dir",
        type=Path,
        default=None,
        help="Directory for *.encoded.jsonl outputs. Defaults to each manifest directory.",
    )
    parser.add_argument(
        "--token-source",
        choices=("staff", "flat_with_newlines", "auto"),
        default="staff",
        help=(
            "Target source to encode. 'staff' uses target_staff_token_sequences. "
            "'flat_with_newlines' splits flat targets on HOMR newline tokens. "
            "'auto' prefers staff and falls back to flat_with_newlines."
        ),
    )
    parser.add_argument(
        "--allow-invalid",
        action="store_true",
        help="Write summaries even if invalid HOMR symbols are found. Default is to fail.",
    )
    parser.add_argument(
        "--max-invalid-examples",
        type=int,
        default=50,
        help="Maximum invalid-symbol examples included in the summary JSON/stdout.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output and summary file.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print JSON progress every N manifest rows. Use 0 to disable progress.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress terminal progress prints except errors and final summary.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise TargetEncodingError(f"Manifest does not exist: {path}")
    if not path.is_file():
        raise TargetEncodingError(f"Manifest path is not a file: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise TargetEncodingError(
                    f"Invalid JSON on line {line_number} of {path}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise TargetEncodingError(
                    f"Line {line_number} in {path} is {type(row).__name__}, expected object."
                )
            row["_manifest_line_number"] = line_number
            rows.append(row)

    if not rows:
        raise TargetEncodingError(f"Manifest has no non-empty JSONL rows: {path}")
    return rows


def write_json_atomic(path: Path, payload: dict[str, Any], *, pretty: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2 if pretty else None,
            sort_keys=pretty,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def encoded_manifest_path(manifest_path: Path, encoded_out_dir: Path | None) -> Path:
    out_dir = encoded_out_dir or manifest_path.parent
    if manifest_path.name.endswith(".jsonl"):
        name = manifest_path.name[:-6] + ".encoded.jsonl"
    else:
        name = manifest_path.name + ".encoded.jsonl"
    return out_dir / name


def ensure_token_list(value: Any, *, row_index: int, field_name: str) -> list[str]:
    if isinstance(value, list):
        tokens = [str(token) for token in value]
    elif isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                tokens = [str(token) for token in parsed]
            else:
                tokens = stripped.split()
        else:
            tokens = stripped.splitlines() if "\n" in stripped else stripped.split()
    else:
        raise TargetEncodingError(
            f"Row {row_index} field {field_name!r} is {type(value).__name__}, "
            "expected list[str] or string."
        )

    for token in tokens:
        if not token.strip():
            raise TargetEncodingError(f"Row {row_index} has a blank token in {field_name}.")
        if token != token.strip():
            raise TargetEncodingError(
                f"Row {row_index} token has surrounding whitespace in {field_name}: {token!r}"
            )
    return tokens


def is_homr_page_newline_token(token: str) -> bool:
    return token == HOMR_PAGE_NEWLINE_RHYTHM or token == HOMR_PAGE_NEWLINE_TOKEN


def staff_sequences_from_row(
    row: dict[str, Any],
    *,
    row_index: int,
    token_source: str,
) -> tuple[list[list[str]], str]:
    if token_source in {"staff", "auto"} and TOKEN_FIELD_STAFF in row:
        raw_staffs = row[TOKEN_FIELD_STAFF]
        if not isinstance(raw_staffs, list):
            raise TargetEncodingError(
                f"Row {row_index} field {TOKEN_FIELD_STAFF!r} is "
                f"{type(raw_staffs).__name__}, expected list[list[str]]."
            )
        staffs: list[list[str]] = []
        for staff_index, raw_staff in enumerate(raw_staffs):
            tokens = ensure_token_list(
                raw_staff,
                row_index=row_index,
                field_name=f"{TOKEN_FIELD_STAFF}[{staff_index}]",
            )
            tokens = [token for token in tokens if not is_homr_page_newline_token(token)]
            if tokens:
                staffs.append(tokens)
        if not staffs:
            raise TargetEncodingError(f"Row {row_index} has no non-empty staff token sequences.")
        return staffs, TOKEN_FIELD_STAFF

    if token_source == "staff":
        raise TargetEncodingError(
            f"Row {row_index} is missing {TOKEN_FIELD_STAFF!r}; this script is configured "
            "to use staff-level target lists inside each page target."
        )

    raw_flat = row.get(TOKEN_FIELD_FLAT, row.get(TOKEN_FIELD_FLAT_WITH_LEGACY_BREAKS))
    if raw_flat is None:
        raise TargetEncodingError(
            f"Row {row_index} is missing {TOKEN_FIELD_STAFF!r} and flat fallback fields."
        )

    flat_tokens = ensure_token_list(raw_flat, row_index=row_index, field_name=TOKEN_FIELD_FLAT)

    staffs = []
    current: list[str] = []
    for token in flat_tokens:
        if is_homr_page_newline_token(token):
            if current:
                staffs.append(current)
                current = []
            continue
        current.append(token)
    if current:
        staffs.append(current)

    if not staffs:
        raise TargetEncodingError(f"Row {row_index} has no target tokens after newline splitting.")
    return staffs, TOKEN_FIELD_FLAT


def parse_symbol_token(token: str) -> tuple[EncodedSymbol | None, list[str]]:
    parts = token.split(" ")
    if len(parts) != 5:
        return None, [f"expected 5 space-separated HOMR fields, got {len(parts)}"]
    rhythm, pitch, lift, articulation, position = parts
    if rhythm == HOMR_PAGE_NEWLINE_RHYTHM:
        return None, ["HOMR page newline must be represented as page structure, not encoded by transformer Vocabulary"]
    return EncodedSymbol(rhythm, pitch, lift, articulation, position), []


def validate_symbol(symbol: EncodedSymbol, vocab: Vocabulary) -> list[str]:
    issues: list[str] = []
    checks = (
        ("rhythm", symbol.rhythm, vocab.rhythm),
        ("pitch", symbol.pitch, vocab.pitch),
        ("lift", symbol.lift, vocab.lift),
        ("articulation", symbol.articulation, vocab.articulation),
        ("position", symbol.position, vocab.position),
    )
    for name, value, mapping in checks:
        if value not in mapping:
            issues.append(f"{name} not in HOMR vocabulary: {value!r}")

    if not issues and not symbol.is_valid():
        issues.append("invalid HOMR component combination according to EncodedSymbol.is_valid()")

    return issues


def encode_symbol(symbol: EncodedSymbol, vocab: Vocabulary) -> dict[str, int]:
    return {
        "rhythm": int(vocab.rhythm[symbol.rhythm]),
        "pitch": int(vocab.pitch[symbol.pitch]),
        "lift": int(vocab.lift[symbol.lift]),
        "articulation": int(vocab.articulation[symbol.articulation]),
        "position": int(vocab.position[symbol.position]),
    }


def branch_nonote_ids(vocab: Vocabulary) -> dict[str, int]:
    return {
        "pitch": int(vocab.pitch["."]),
        "lift": int(vocab.lift["."]),
        "articulation": int(vocab.articulation["."]),
        "position": int(vocab.position["."]),
    }


def encode_staff(
    tokens: list[str],
    *,
    vocab: Vocabulary,
    manifest: Path,
    row: dict[str, Any],
    row_index: int,
    staff_index: int,
    stats: ManifestStats,
) -> dict[str, Any] | None:
    symbols: list[EncodedSymbol] = []
    row_invalids: list[InvalidSymbol] = []

    for token_index, token in enumerate(tokens):
        symbol, parse_issues = parse_symbol_token(token)
        issues = list(parse_issues)
        if symbol is not None:
            issues.extend(validate_symbol(symbol, vocab))

        if issues:
            row_invalids.append(
                InvalidSymbol(
                    manifest=str(manifest),
                    row_index=row_index,
                    line_number=row.get("_manifest_line_number"),
                    page_id=row.get("page_id") if isinstance(row.get("page_id"), str) else None,
                    score_id=row.get("score_id") if isinstance(row.get("score_id"), str) else None,
                    staff_index=staff_index,
                    token_index=token_index,
                    token=token,
                    issues=issues,
                )
            )
            continue

        assert symbol is not None
        symbols.append(symbol)

    if row_invalids:
        stats.invalid_symbols.extend(row_invalids)
        return None

    nonote = branch_nonote_ids(vocab)

    rhythm_ids = [int(vocab.rhythm["BOS"])]
    pitch_ids = [nonote["pitch"]]
    lift_ids = [nonote["lift"]]
    articulation_ids = [nonote["articulation"]]
    position_ids = [nonote["position"]]
    mask = [True]

    for symbol in symbols:
        encoded = encode_symbol(symbol, vocab)
        rhythm_ids.append(encoded["rhythm"])
        pitch_ids.append(encoded["pitch"])
        lift_ids.append(encoded["lift"])
        articulation_ids.append(encoded["articulation"])
        position_ids.append(encoded["position"])
        mask.append(True)
        stats.rhythm_counts[symbol.rhythm] += 1

    rhythm_ids.append(int(vocab.rhythm["EOS"]))
    pitch_ids.append(nonote["pitch"])
    lift_ids.append(nonote["lift"])
    articulation_ids.append(nonote["articulation"])
    position_ids.append(nonote["position"])
    mask.append(True)

    return {
        "staff_index": staff_index,
        "length_without_bos_eos": len(symbols),
        "length_with_bos_eos": len(rhythm_ids),
        "rhythm_ids": rhythm_ids,
        "pitch_ids": pitch_ids,
        "lift_ids": lift_ids,
        "articulation_ids": articulation_ids,
        "position_ids": position_ids,
        "mask": mask,
    }


def page_structure_payload(encoded_staffs: list[dict[str, Any]]) -> dict[str, Any]:
    staff_indices = [int(staff["staff_index"]) for staff in encoded_staffs]
    return {
        "schema": "homr_page_structure_v1",
        "representation": "ordered_staff_sequences",
        "staff_order": staff_indices,
        "n_staffs": len(staff_indices),
        "postprocess_separator": {
            "rhythm": HOMR_PAGE_NEWLINE_RHYTHM,
            "encoded_symbol": HOMR_PAGE_NEWLINE_RHYTHM,
            "insert_after_each_staff_except_last": True,
            "encoded_in_transformer_vocabulary": False,
            "reason": (
                "HOMR page assembly uses EncodedSymbol('newline') for system/page layout, "
                "while HOMR transformer Vocabulary remains the source of truth for musical staff symbols."
            ),
        },
    }


def encode_manifest(
    manifest: Path,
    *,
    vocab: Vocabulary,
    encoded_out_dir: Path | None,
    token_source: str,
    progress_every: int = 0,
    quiet: bool = False,
) -> tuple[ManifestStats, Path, list[dict[str, Any]]]:
    rows = read_jsonl(manifest)
    started = time.time()
    stats = ManifestStats(manifest=str(manifest), rows=len(rows))
    encoded_rows: list[dict[str, Any]] = []
    emit_event(
        {
            "event": "start",
            "manifest": str(manifest),
            "rows": len(rows),
            "token_source": token_source,
        },
        quiet=quiet,
    )

    for row_index, row in enumerate(rows):
        staffs, source_name = staff_sequences_from_row(row, row_index=row_index, token_source=token_source)
        stats.token_source_counts[source_name] += 1

        encoded_staffs: list[dict[str, Any]] = []
        had_invalid = False
        for staff_index, staff_tokens in enumerate(staffs):
            encoded_staff = encode_staff(
                staff_tokens,
                vocab=vocab,
                manifest=manifest,
                row=row,
                row_index=row_index,
                staff_index=staff_index,
                stats=stats,
            )
            if encoded_staff is None:
                had_invalid = True
                continue

            stats.staffs += 1
            stats.symbols_without_bos_eos += int(encoded_staff["length_without_bos_eos"])
            stats.symbols_with_bos_eos += int(encoded_staff["length_with_bos_eos"])
            stats.max_staff_len_without_bos_eos = max(
                stats.max_staff_len_without_bos_eos,
                int(encoded_staff["length_without_bos_eos"]),
            )
            stats.max_staff_len_with_bos_eos = max(
                stats.max_staff_len_with_bos_eos,
                int(encoded_staff["length_with_bos_eos"]),
            )
            encoded_staffs.append(encoded_staff)

        if had_invalid:
            continue

        stats.page_newline_separators += max(0, len(encoded_staffs) - 1)
        encoded_row = {
            "schema": "homr_factorized_page_training_manifest_v1",
            "source_manifest": str(manifest),
            "source_manifest_line_number": row.get("_manifest_line_number"),
            "score_id": row.get("score_id"),
            "page_id": row.get("page_id"),
            "page_number": row.get("page_number"),
            "image_path": row.get("image_path") or row.get("rendered_image_path"),
            "teacher_path": row.get("teacher_path"),
            "target_source_field": source_name,
            "n_staffs": len(encoded_staffs),
            "homr_page_structure": page_structure_payload(encoded_staffs),
            "homr_target_staffs": encoded_staffs,
        }
        encoded_rows.append(encoded_row)

        processed = row_index + 1
        if progress_every > 0 and (processed == 1 or processed % progress_every == 0 or processed == len(rows)):
            emit_event(
                {
                    "event": "progress",
                    "manifest": str(manifest),
                    "processed": processed,
                    "total": len(rows),
                    "encoded_rows": len(encoded_rows),
                    "invalid_symbols": len(stats.invalid_symbols),
                    "staffs": stats.staffs,
                    "symbols_with_bos_eos": stats.symbols_with_bos_eos,
                    "seconds": time.time() - started,
                    "last_page_id": row.get("page_id"),
                },
                quiet=quiet,
            )

    stats.encoded_rows = len(encoded_rows)
    out_path = encoded_manifest_path(manifest, encoded_out_dir)
    return stats, out_path, encoded_rows


def homr_vocab_summary(vocab: Vocabulary) -> dict[str, Any]:
    return {
        "source": "homr.transformer.vocabulary.Vocabulary",
        "encoding": "homr_factorized",
        "sizes": {
            "rhythm": len(vocab.rhythm),
            "pitch": len(vocab.pitch),
            "lift": len(vocab.lift),
            "articulation": len(vocab.articulation),
            "position": len(vocab.position),
        },
        "special": {
            "rhythm_pad": "PAD",
            "rhythm_bos": "BOS",
            "rhythm_eos": "EOS",
            "nonote": ".",
            "empty": "_",
            "page_newline": HOMR_PAGE_NEWLINE_RHYTHM,
        },
        "special_ids": {
            "rhythm_pad": int(vocab.rhythm["PAD"]),
            "rhythm_bos": int(vocab.rhythm["BOS"]),
            "rhythm_eos": int(vocab.rhythm["EOS"]),
            "pitch_nonote": int(vocab.pitch["."]),
            "lift_nonote": int(vocab.lift["."]),
            "articulation_nonote": int(vocab.articulation["."]),
            "position_nonote": int(vocab.position["."]),
            "page_newline": None,
        },
        "page_newline_note": (
            "HOMR page assembly uses EncodedSymbol('newline') outside the transformer Vocabulary; "
            "it is represented as page structure metadata, not a custom vocabulary ID."
        ),
    }


def main() -> int:
    args = parse_args()
    vocab = Vocabulary()

    manifests = [args.train_manifest, *args.validate_manifest]
    manifest_payloads: list[dict[str, Any]] = []
    total_invalid = 0
    encoded_outputs: list[dict[str, str]] = []

    for manifest in manifests:
        stats, encoded_path, encoded_rows = encode_manifest(
            manifest,
            vocab=vocab,
            encoded_out_dir=args.encoded_out_dir,
            token_source=args.token_source,
            progress_every=args.progress_every,
            quiet=args.quiet,
        )
        total_invalid += len(stats.invalid_symbols)
        write_jsonl_atomic(encoded_path, encoded_rows)
        emit_event(
            {
                "event": "encoded_written",
                "manifest": str(manifest),
                "encoded_manifest": str(encoded_path),
                "encoded_rows": len(encoded_rows),
                "invalid_symbols": len(stats.invalid_symbols),
            },
            quiet=args.quiet,
        )
        encoded_outputs.append({"manifest": str(manifest), "encoded_manifest": str(encoded_path)})
        manifest_payloads.append(stats.to_json(max_invalid_examples=args.max_invalid_examples))

    status = "ok" if total_invalid == 0 else "invalid_targets"
    payload = {
        "schema": "homr_factorized_page_target_encoding_summary_v1",
        "status": status,
        "homr_vocabulary": homr_vocab_summary(vocab),
        "token_source": args.token_source,
        "page_target_representation": "ordered_staff_sequences_with_homr_newline_postprocess_metadata",
        "encoded_outputs": encoded_outputs,
        "manifests": manifest_payloads,
        "total_invalid_symbol_count": total_invalid,
    }

    write_json_atomic(args.out, payload, pretty=args.pretty)
    print(json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None, sort_keys=args.pretty), flush=True)

    if total_invalid and not args.allow_invalid:
        raise TargetEncodingError(
            f"Found {total_invalid} invalid HOMR target symbols. Fix teacher targets or pass "
            "--allow-invalid only for debugging."
        )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except TargetEncodingError as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(2)
