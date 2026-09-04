"""Run the check over one case, the sample, or everything -- and always report.

    python -m fixturecheck one laulun-aika-s2      ~20s
    python -m fixturecheck ten                     ~4 min
    python -m fixturecheck all                     ~35 min

The pytest gate is left alone and answers a different question: it says yes or
no, in CI, in seconds. This says how much, and shows the music.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fixturecheck import cases, report  # noqa: E402
from fixturecheck.compare import compare_output  # noqa: E402

PARSES = cases.CACHE / "parses"


def parse(case: cases.Case, fingerprint: str) -> Path | None:
    """Read the case's picture with this homr, reusing the last read if it stands.

    Keyed on the code: a parse is only stale when homr changes, so editing the
    report or adding a case re-runs nothing. Nothing here shares a filename with
    a reference -- homr writes `<image>.musicxml`, which is exactly where the
    reference lives, and that silently destroyed ninety-three of them once.
    """
    PARSES.mkdir(parents=True, exist_ok=True)
    out = PARSES / f"{case.name}@{fingerprint}.musicxml"
    if out.exists():
        return out
    with tempfile.TemporaryDirectory(prefix="parse-") as tmp:
        copy = Path(tmp) / f"{case.name}.png"
        shutil.copy(case.image, copy)
        run = subprocess.run(
            [sys.executable, "-c", "from homr.main import main; main()",
             str(copy), "--gpu", "no"],
            capture_output=True, text=True, timeout=900,
            cwd=cases.ROOT)
        produced = copy.with_suffix(".musicxml")
        if run.returncode != 0 or not produced.exists():
            return None
        shutil.copy(produced, out)
    return out


def code_fingerprint() -> str:
    """What homr is right now: the commit, plus whatever is uncommitted."""
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=cases.ROOT,
                          capture_output=True, text=True).stdout.strip() or "nogit"
    dirty = subprocess.run(["git", "diff", "--stat", "HEAD", "--", "homr"], cwd=cases.ROOT,
                           capture_output=True, text=True).stdout
    if dirty.strip():
        import hashlib
        head += "+" + hashlib.sha256(dirty.encode()).hexdigest()[:6]
    return head


def main() -> None:
    tier = sys.argv[1] if len(sys.argv) > 1 else "ten"
    if tier == "one":
        names = sys.argv[2:]
    elif tier == "ten":
        names = cases.sample()
    elif tier == "all":
        names = cases.every()
    else:
        names = [tier] + sys.argv[2:]
        tier = "one"
    if not names:
        print("nothing to check")
        return

    fingerprint = code_fingerprint()
    previous = report.load_previous()
    print(f"homr {fingerprint}: {len(names)} case(s)")
    entries = []
    for name in names:
        found = cases.resolve([name])
        if not found:
            print(f"  {name}: could not be built")
            continue
        case = found[0]
        parsed = parse(case, fingerprint)
        if parsed is None:
            print(f"  {case.name}: homr could not read it")
            continue
        result = compare_output(case.reference, parsed)
        before = previous.get(case.name)
        page = report.case_page(case, parsed, result, before)
        entries.append({"name": case.name, "page": page, "score": result.score,
                        "agree": result.agree, "voice": result.voice,
                        "pitch": result.pitch, "size": result.size,
                        "timing": result.timing, "structure": result.structure,
                        "staves_page": result.staves_page,
                        "staves_homr": result.staves_homr,
                        "unison": result.unison, "before": before})
        moved = ""
        if before:
            change = (result.voice + result.pitch) - (before["voice"] + before["pitch"])
            moved = "  (no change)" if change == 0 else f"  ({change:+d} faults)"
        staves = (f", staves {result.staves_page} vs {result.staves_homr}"
                  if result.structure else "")
        print(f"  {case.name}: {result.agree} agree, {result.voice} voice, "
              f"{result.pitch} pitch, {result.size} count, "
              f"{result.timing} beat{staves}{moved}")

    if entries:
        report.save_results(entries)
        print("\n" + str(report.index_page(entries, tier)))


if __name__ == "__main__":
    main()
