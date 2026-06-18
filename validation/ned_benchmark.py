"""
OMR-NED benchmark framework.

Three parts are kept distinct and swappable:
  - Data source  (smb.py, polish-scores.py, ...) - yields (sample_id, kern_text, image)
  - Tool         (tools.py)                      - (kern_text, image) -> output_text
  - Check        (ned_score.py)                  - compute_ned + NedResult

Tools may return MusicXML or **kern; the format is auto-detected and parsed directly
- no intermediate conversion.  The raw tool output is stored as-is in the database.

Detailed report
---------------
Pass ``output_db`` to ``run_benchmark`` to write a SQLite database with:
  - ``samples``     - one row per sample: raw kern + raw tool output + NED metrics
  - ``token_events``- one row per edit-distance alignment op (match/delete/insert/substitute)
                      with all component fields (rhythm, pitch, lift, articulation, slur)

Example queries:
  -- How often was a clef missing?
  SELECT COUNT(*) FROM token_events WHERE event_type='delete' AND exp_rhythm LIKE '%clef%';

  -- How often was pitch A6 mistaken with B8?
  SELECT COUNT(*) FROM token_events
  WHERE event_type='substitute' AND exp_pitch='A6' AND act_pitch='B8';

  -- Most error-prone expected tokens overall (by rhythm):
  SELECT exp_rhythm, COUNT(*) AS errors
  FROM token_events WHERE event_type!='match' AND exp_rhythm IS NOT NULL
  GROUP BY exp_rhythm ORDER BY errors DESC LIMIT 20;

  -- How often does '#' get hallucinated?
  SELECT COUNT(*) FROM token_events
  WHERE (event_type='insert' AND act_lift='#')
     OR (event_type='substitute' AND act_lift='#' AND (exp_lift IS NULL OR exp_lift!='#'));
"""

import contextlib
import os
import signal
import statistics
import sys
import traceback
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

from homr.transformer.vocabulary import EncodedSymbol
from validation.benchmark_db import (
    _DELETE_EVENTS,
    _INSERT_EVENT,
    _INSERT_SAMPLE,
    BenchmarkDB,
)
from validation.ned_score import (
    NedResult,
    TokenEvent,
    _events_for_parts,
    _musicdiff_detailed_ned_for_sample,
    _musicdiff_ned_for_sample,
    _musicdiff_register_once,
    _ned_from_parts,
    _parse_output,
)

_SAMPLE_TIMEOUT_S = 120  # per-sample wall-clock limit; guards against infinite loops


@contextlib.contextmanager
def _sample_timeout() -> Iterator[None]:
    """Raise TimeoutError if the block exceeds _SAMPLE_TIMEOUT_S seconds."""

    def _handler(signum: int, frame: object) -> None:
        raise TimeoutError(f"sample timed out after {_SAMPLE_TIMEOUT_S}s")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(_SAMPLE_TIMEOUT_S)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def _print_stats(label: str, values: list[float]) -> None:
    print(f"  {label}:")  # noqa: T201
    print(f"    Mean:     {statistics.mean(values) * 100:6.2f}%")  # noqa: T201
    print(f"    Median:   {statistics.median(values) * 100:6.2f}%")  # noqa: T201
    print(f"    Min:      {min(values) * 100:6.2f}%")  # noqa: T201
    print(f"    Max:      {max(values) * 100:6.2f}%")  # noqa: T201
    if len(values) > 1:
        pct_vals = [v * 100 for v in values]
        print(f"    Variance: {statistics.variance(pct_vals):8.4f}%²")  # noqa: T201


def _record_success(
    sample_id: str,
    kern_text: str,
    raw_output: str,
    results: list[NedResult],
    db: BenchmarkDB | None,
    kern_parser: str = "native",
    xml_parser: str = "native",
) -> None:
    kern_parts: list[list[EncodedSymbol]] = []
    xml_parts: list[list[EncodedSymbol]] = []
    if xml_parser == "musicdiff":
        result = _musicdiff_ned_for_sample(kern_text, raw_output)
    elif xml_parser == "musicdiff_detailed":
        result = _musicdiff_detailed_ned_for_sample(kern_text, raw_output)
    else:
        kern_parts, xml_parts = _parse_output(kern_text, raw_output, kern_parser, xml_parser)
        result = _ned_from_parts(kern_parts, xml_parts)
    results.append(result)
    print(  # noqa: T201
        f"[{sample_id}] NED={result.ned * 100:5.1f}%  "
        f"rhythm={result.rhythm_ned * 100:5.1f}%  "
        f"pitch={result.pitch_ned * 100:5.1f}%  "
        f"lift={result.lift_ned * 100:5.1f}%  "
        f"art={result.articulation_ned * 100:5.1f}%  "
        f"slur={result.slur_ned * 100:5.1f}%  "
        f"dist={result.distance:4d}  ref={result.kern_len:4d}  pred={result.xml_len:4d}"
    )
    if db is not None:
        events: list[TokenEvent] = (
            []
            if xml_parser in ("musicdiff", "musicdiff_detailed")
            else _events_for_parts(kern_parts, xml_parts)
        )
        db.write_success(sample_id, kern_text, raw_output, result, events)


def _record_failure(
    sample_id: str,
    kern_text: str,
    error: str,
    failures: list[tuple[str, str]],
    db: BenchmarkDB | None,
    verbose: bool,
    exc: BaseException | None = None,
    actual_text: str | None = None,
) -> None:
    failures.append((sample_id, error))
    print(f"[{sample_id}] ERROR: {error}", file=sys.stderr)  # noqa: T201
    if verbose and exc is not None:
        traceback.print_exception(type(exc), exc, exc.__traceback__)
    if db is not None:
        db.write_failure(sample_id, kern_text, error, actual_text)


def update_ned_scores(
    output_db: str,
    verbose: bool = False,
    kern_parser: str = "native",
    xml_parser: str = "native",
    limit: int | None = None,
) -> None:
    """Recompute NED for all samples with stored actual_text using the current parsers.

    Samples whose actual_text is NULL (tool failures or hard parse errors) are skipped,
    since there is no output to re-parse.
    limit: if set, process at most this many samples.
    """
    db = BenchmarkDB(output_db)
    query = (
        "SELECT sample_id, kern_text, actual_text FROM samples"
        " WHERE actual_text IS NOT NULL ORDER BY rowid"
    )
    if limit is not None:
        query += f" LIMIT {limit}"
    rows = db._conn.execute(query).fetchall()

    print(f"Recomputing NED for {len(rows)} samples …")  # noqa: T201

    if xml_parser in ("musicdiff", "musicdiff_detailed"):
        _musicdiff_register_once()

    results: list[NedResult] = []
    failures: list[tuple[str, str]] = []

    for sample_id, kern_text, actual_text in rows:
        try:
            if xml_parser == "musicdiff":
                result = _musicdiff_ned_for_sample(kern_text, actual_text)
                events: list[TokenEvent] = []
            elif xml_parser == "musicdiff_detailed":
                result = _musicdiff_detailed_ned_for_sample(kern_text, actual_text)
                events = []
            else:
                kern_parts, xml_parts = _parse_output(
                    kern_text, actual_text, kern_parser, xml_parser
                )
                result = _ned_from_parts(kern_parts, xml_parts)
                events = _events_for_parts(kern_parts, xml_parts)

            db._conn.execute(_DELETE_EVENTS, (sample_id,))
            db._conn.execute(
                _INSERT_SAMPLE,
                (
                    sample_id,
                    kern_text,
                    actual_text,
                    result.ned,
                    result.distance,
                    result.kern_len,
                    result.xml_len,
                    result.rhythm_ned,
                    result.pitch_ned,
                    result.lift_ned,
                    result.articulation_ned,
                    result.slur_ned,
                    None,
                ),
            )
            db._conn.executemany(_INSERT_EVENT, [{"sample_id": sample_id, **e} for e in events])
            db._conn.commit()

            results.append(result)
            print(  # noqa: T201
                f"[{sample_id}] NED={result.ned * 100:5.1f}%  "
                f"rhythm={result.rhythm_ned * 100:5.1f}%  "
                f"pitch={result.pitch_ned * 100:5.1f}%"
            )
        except Exception as e:  # noqa: BLE001
            failures.append((sample_id, str(e)))
            print(f"[{sample_id}] ERROR: {e}", file=sys.stderr)  # noqa: T201
            if verbose:
                traceback.print_exception(type(e), e, e.__traceback__)

    db.close()

    print()  # noqa: T201
    summary = f"Updated {len(results)} samples"
    if failures:
        summary += f", {len(failures)} newly failed"
    print(summary)  # noqa: T201
    if results:
        _print_stats("Overall NED ", [r.ned for r in results])
        _print_stats("Rhythm NED  ", [r.rhythm_ned for r in results])
        _print_stats("Pitch NED   ", [r.pitch_ned for r in results])


def run_benchmark(
    samples: Iterable[tuple[str, str, Path | None]],
    tool: Callable[[str, Path | None], str],
    limit: int | None = None,
    verbose: bool = False,
    output_db: str | None = None,
    continue_run: bool = False,
    batch_size: int = 10,
    kern_parser: str = "native",
    xml_parser: str = "native",
) -> None:
    """
    Wire data source, tool, and check together.

    samples:      iterable of (sample_id, kern_text, image_path) from any data source
    tool:         (kern_text, image_path) -> output_text (MusicXML or **kern)
    kern_parser:  "native" (default) or "music21" - kern ground-truth parser
    xml_parser:   "native" (default), "music21", "musicdiff", or "musicdiff_detailed"
                  If the tool has a batch_run() method, samples are processed in chunks of
                  batch_size (useful for homr which loads model weights once per chunk).
    output_db:    path to a SQLite file; written fresh by default, or appended with continue_run
    continue_run: if True, skip sample_ids already present in output_db
    batch_size:   number of samples per batch_run() call (default 10)
    """
    db: BenchmarkDB | None = None
    skip_ids: set[str] = set()

    if output_db is not None:
        if not continue_run and os.path.exists(output_db):
            os.remove(output_db)
        db = BenchmarkDB(output_db)
        if continue_run:
            skip_ids = db.already_processed_ids()
            if skip_ids:
                print(  # noqa: T201
                    f"Continuing run — skipping {len(skip_ids)} already-processed samples."
                )

    results: list[NedResult] = []
    failures: list[tuple[str, str]] = []

    # Collect from iterator (applying limit and skips) so we can optionally batch.
    all_from_iter: list[tuple[str, str, Path | None]] = []
    for count, triple in enumerate(samples):
        if limit is not None and count >= limit:
            break
        all_from_iter.append(triple)

    skipped = sum(1 for s, _, _ in all_from_iter if s in skip_ids)
    active = [(s, k, i) for s, k, i in all_from_iter if s not in skip_ids]

    if xml_parser in ("musicdiff", "musicdiff_detailed"):
        _musicdiff_register_once()

    if hasattr(tool, "batch_run") and active:
        # Batch mode: split into chunks so progress is visible after each batch.
        chunks = [active[i : i + batch_size] for i in range(0, len(active), batch_size)]
        n_chunks = len(chunks)
        done = 0
        for chunk_idx, chunk in enumerate(chunks, start=1):
            print(  # noqa: T201
                f"\n--- Batch {chunk_idx}/{n_chunks} "
                f"({done + 1}–{done + len(chunk)} of {len(active)}) ---"
            )
            batch_results: list[tuple[str | None, str | None]] = tool.batch_run(chunk)
            for (sample_id, kern_text, _image), (raw_output, error) in zip(
                chunk, batch_results, strict=False
            ):
                if error is not None or raw_output is None:
                    _record_failure(
                        sample_id, kern_text, error or "unknown error", failures, db, verbose
                    )
                else:
                    try:
                        with _sample_timeout():
                            _record_success(
                                sample_id,
                                kern_text,
                                raw_output,
                                results,
                                db,
                                kern_parser,
                                xml_parser,
                            )
                    except Exception as e:  # noqa: BLE001
                        _record_failure(
                            sample_id,
                            kern_text,
                            str(e),
                            failures,
                            db,
                            verbose,
                            e,
                            actual_text=raw_output,
                        )
            done += len(chunk)
    else:
        # Single-sample mode.
        for sample_id, kern_text, image in active:
            raw_output = None
            try:
                with _sample_timeout():
                    raw_output = tool(kern_text, image)
                    _record_success(
                        sample_id, kern_text, raw_output, results, db, kern_parser, xml_parser
                    )
            except Exception as e:  # noqa: BLE001
                _record_failure(
                    sample_id,
                    kern_text,
                    str(e),
                    failures,
                    db,
                    verbose,
                    e,
                    actual_text=raw_output,
                )

    if db is not None:
        db.close()

    print()  # noqa: T201
    summary = (
        f"Processed {len(results) + len(failures)} samples: "
        f"{len(results)} succeeded, {len(failures)} failed."
    )
    if skipped:
        summary += f" ({skipped} skipped, already in DB)"
    print(summary)  # noqa: T201

    if results:
        print("\nAggregate Statistics:")  # noqa: T201
        _print_stats("Overall NED ", [r.ned for r in results])
        _print_stats("Rhythm NED  ", [r.rhythm_ned for r in results])
        _print_stats("Pitch NED   ", [r.pitch_ned for r in results])
        _print_stats("Lift NED    ", [r.lift_ned for r in results])
        _print_stats("Artic NED   ", [r.articulation_ned for r in results])
        _print_stats("Slur NED    ", [r.slur_ned for r in results])

    if failures:
        print(f"\nFailures ({len(failures)}):")  # noqa: T201
        for fid, err in failures:
            print(f"  [{fid}] {err}")  # noqa: T201
