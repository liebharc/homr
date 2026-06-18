"""
Show a side-by-side token diff for one sample from a benchmark database.

Usage:
    poetry run python validation/show_diff.py <db_file> [--sample ID] [--output DIR]

  db_file       Path to a SQLite file produced by a ned_benchmark run.
  --sample      Sample ID (string) or numeric index (default: 0).
  --list        List all sample IDs with NED scores and exit.
  --output      Directory to write expected.krn, actual.{musicxml,krn}, and score.png.
  --dataset     HuggingFace dataset name to fetch the score image from.
                Auto-detected from the DB filename when not provided.
"""

# ruff: noqa: T201

import argparse
import sqlite3
import sys
from pathlib import Path

_COL = 47  # width of one token column in the side-by-side display


def _fmt(
    rhythm: str | None,
    pitch: str | None,
    lift: str | None,
    articulation: str | None,
    slur: str | None,
) -> str:
    return (
        f"{rhythm or '.':22s} {pitch or '.':6s} {lift or '.':4s}"
        f" {articulation or '.':12s} {slur or '.'}"
    )


def _print_diff(conn: sqlite3.Connection, sample_id: str) -> tuple[int, int, int, int]:
    """Print side-by-side diff from token_events. Returns (match, delete, insert, substitute)."""
    events = conn.execute(
        """
        SELECT staff, event_type,
               exp_rhythm, exp_pitch, exp_lift, exp_articulation, exp_slur,
               act_rhythm, act_pitch, act_lift, act_articulation, act_slur
        FROM token_events WHERE sample_id = ?
        ORDER BY id
        """,
        (sample_id,),
    ).fetchall()

    print(f"{'EXPECTED':^{_COL}}    {'ACTUAL':^{_COL}}")  # noqa: T201
    print("=" * ((_COL + 2) * 2))  # noqa: T201

    n_match = n_delete = n_insert = n_sub = 0
    cur_staff: str | None = None
    for staff, evt, er, ep, el, ea, es, ar, ap, al, aa, as_ in events:
        if staff != cur_staff:
            print(f"\n--- {staff} ---")  # noqa: T201
            cur_staff = staff
        exp = _fmt(er, ep, el, ea, es)
        act = _fmt(ar, ap, al, aa, as_)
        if evt == "match":
            print(f"  {exp:<{_COL}}  {act}")  # noqa: T201
            n_match += 1
        elif evt == "delete":
            print(f"< {exp:<{_COL}}")  # noqa: T201
            n_delete += 1
        elif evt == "insert":
            print(f"  {'':>{_COL}}> {act}")  # noqa: T201
            n_insert += 1
        elif evt == "substitute":
            print(f"! {exp:<{_COL}}! {act}")  # noqa: T201
            n_sub += 1
    return n_match, n_delete, n_insert, n_sub


def _resolve_sample_id(conn: sqlite3.Connection, sample_arg: str | None) -> str:
    if sample_arg is None or sample_arg.isdigit():
        idx = int(sample_arg or "0")
        row = conn.execute(
            "SELECT sample_id FROM samples ORDER BY rowid LIMIT 1 OFFSET ?", (idx,)
        ).fetchone()
        if row is None:
            print(f"No sample at index {idx}.", file=sys.stderr)  # noqa: T201
            sys.exit(1)
        return str(row[0])
    return sample_arg


def _detect_dataset(db_path: Path) -> str | None:
    name = db_path.stem.lower()
    if "smb" in name:
        return "PRAIG/SMB"
    if "polish" in name:
        return "btrkeks/polish-scores"
    return None


def _save_score_image(sample_id: str, dataset: str, dest: Path) -> bool:
    try:
        from datasets import load_dataset  # type: ignore  # noqa: PLC0415
    except ImportError:
        return False
    try:
        if dataset == "PRAIG/SMB":
            i = int(sample_id.split("_", 1)[0])
            ds = load_dataset(dataset)
            img = ds["test"][i].get("image")
            if img is not None and hasattr(img, "save"):
                img.save(dest)
                return True
        elif dataset == "btrkeks/polish-scores":
            i = int(sample_id)
            ds = load_dataset(dataset)
            img = ds["train"][i].get("image")
            if img is not None and hasattr(img, "save"):
                img.save(dest)
                return True
    except Exception as e:  # noqa: BLE001
        print(f"Could not fetch image: {e}", file=sys.stderr)  # noqa: T201
    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show token diff for one sample from a benchmark DB."
    )
    parser.add_argument("db", help="Path to the SQLite benchmark database.")
    parser.add_argument(
        "-s",
        "--sample",
        default=None,
        dest="sample_id",
        help="Sample ID (string) or numeric index (default: 0).",
    )
    parser.add_argument(
        "-l",
        "--list",
        action="store_true",
        help="List all sample IDs with NED scores and exit.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Directory to write expected.krn, actual.{musicxml,krn}, and score.png.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="HuggingFace dataset name for image fetching (auto-detected from DB name).",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Database not found: {db_path}", file=sys.stderr)  # noqa: T201
        sys.exit(1)

    conn = sqlite3.connect(str(db_path))

    if args.list:
        rows = conn.execute("SELECT sample_id, ned, error FROM samples ORDER BY rowid").fetchall()
        for i, (sid, ned, err) in enumerate(rows):
            if err:
                print(f"  {i:4d}  {sid}  ERROR: {err}")  # noqa: T201
            else:
                print(f"  {i:4d}  {sid}  NED={ned * 100:.1f}%")  # noqa: T201
        conn.close()
        return

    sample_id = _resolve_sample_id(conn, args.sample_id)

    meta = conn.execute(
        "SELECT ned, distance, kern_len, xml_len, error FROM samples WHERE sample_id = ?",
        (sample_id,),
    ).fetchone()
    if meta is None:
        print(f"Sample '{sample_id}' not found in DB.", file=sys.stderr)  # noqa: T201
        sys.exit(1)

    ned, distance, kern_len, xml_len, error = meta
    if error:
        print(f"Sample '{sample_id}' FAILED: {error}", file=sys.stderr)  # noqa: T201
        sys.exit(1)

    print(  # noqa: T201
        f"Sample: {sample_id}  "
        f"NED={ned * 100:.1f}%  dist={distance}  kern={kern_len}  xml={xml_len}"
    )
    print()  # noqa: T201

    n_match, n_delete, n_insert, n_sub = _print_diff(conn, sample_id)

    print()  # noqa: T201
    print(  # noqa: T201
        f"Tokens: {n_match} match  {n_delete} delete  {n_insert} insert  {n_sub} substitute"
    )

    if args.output is not None:
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        row = conn.execute(
            "SELECT kern_text, actual_text FROM samples WHERE sample_id = ?",
            (sample_id,),
        ).fetchone()
        if row is not None:
            kern_text, actual_text = row
            if kern_text:
                dest = out / "expected.krn"
                dest.write_text(kern_text, encoding="utf-8")
                print(f"Wrote {dest}")  # noqa: T201
            if actual_text:
                stripped = actual_text.lstrip()
                is_kern = stripped.startswith("**") or "\t**kern" in stripped[:200]
                dest = out / ("actual.krn" if is_kern else "actual.musicxml")
                dest.write_text(actual_text, encoding="utf-8")
                print(f"Wrote {dest}")  # noqa: T201

        dataset = args.dataset or _detect_dataset(db_path)
        if dataset:
            dest = out / "score.png"
            if _save_score_image(sample_id, dataset, dest):
                print(f"Wrote {dest}")  # noqa: T201
            else:
                print(
                    f"Could not write score image (dataset={dataset})", file=sys.stderr
                )  # noqa: T201
        else:
            print(  # noqa: T201
                "No dataset detected for image; pass --dataset to fetch score.png.",
                file=sys.stderr,
            )

    conn.close()


if __name__ == "__main__":
    main()
