"""The report, which is the output rather than a thing you ask for afterwards.

Every run writes it. There is no flag for "just the numbers": a count says a
system agrees on staves, bars and noteheads and cannot say whether the parse is
the music, and asking for the pictures separately is how a whole afternoon went
on hand-built pages.

Each case page opens with the same three images in the same order -- the printed
band, homr's output engraved, the reference engraved -- so two cases can be read
against each other without working out what is being shown.
"""

from __future__ import annotations

import html
import json
import os
import shutil
import subprocess
from pathlib import Path

from fixturecheck.compare import Result

OUT = Path(__file__).resolve().parent.parent / "check-report"

STYLE = """
body { font: 14px/1.55 system-ui, sans-serif; margin: 0 auto; max-width: 1150px;
       padding: 24px; color: #1a1a1a; background: #fafafa; }
h1 { font-size: 22px; margin-bottom: 2px; }
h2 { font-size: 17px; margin: 30px 0 6px; }
h3 { font-size: 13px; margin: 18px 0 2px; color: #444; text-transform: uppercase;
     letter-spacing: .06em; }
p.lead { color: #555; margin-top: 0; }
a { color: #1b3a7a; }
img { display: block; width: 100%; border: 1px solid #e2e2e2; border-radius: 6px;
      background: #fff; margin: 4px 0 12px; }
table { border-collapse: collapse; width: 100%; background: #fff;
        border: 1px solid #e2e2e2; border-radius: 6px; overflow: hidden;
        margin-bottom: 10px; }
th, td { padding: 5px 9px; text-align: left; border-bottom: 1px solid #f0f0f0;
         font-variant-numeric: tabular-nums; }
th { background: #f4f4f4; font-size: 12px; text-transform: uppercase;
     letter-spacing: .04em; color: #555; }
tr.voice td { background: #fdecec; }
tr.pitch td { background: #fde8d8; }
tr.size  td { background: #fff8e1; }
tr.timing td { background: #eef4fb; }
tr.structure td { background: #f3e8fb; font-weight: 600; }
tr.unison td { background: #f4f4f7; }
td.ok { color: #1c5c2c; } td.no { color: #8a1f1f; font-weight: 600; }
.sum { display: flex; gap: 22px; flex-wrap: wrap; margin: 8px 0 16px; }
.sum b { font-size: 20px; display: block; }
.up { color: #1c5c2c; font-weight: 600; } .down { color: #8a1f1f; font-weight: 600; }
.same { color: #888; }
p.warn { background: #f3e8fb; border: 1px solid #ddc7ee; border-radius: 6px;
         padding: 9px 12px; margin: 0 0 14px; }
"""


def _engrave(source: Path, into: Path, dpi: int = 220) -> str:
    """One score as a picture, trimmed to the music rather than an empty A4."""
    cli = (os.environ.get("MUSESCORE_CLI_PATH") or "musescore3").strip().strip('"')
    run = subprocess.run([cli, "-T", "10", "-r", str(dpi), str(source), "-o", str(into)],
                         capture_output=True, text=True, timeout=600)
    numbered = into.with_name(f"{into.stem}-1.png")
    if numbered.exists():
        numbered.replace(into)
    return into.name if run.returncode == 0 and into.exists() else ""


def case_page(case, parsed: Path, result: Result, before: dict | None) -> str:
    """Write one case's page and return its filename."""
    OUT.mkdir(parents=True, exist_ok=True)
    page = OUT / f"{case.name}-page.png"
    shutil.copy(case.image, page)
    output = _engrave(parsed, OUT / f"{case.name}-homr.png")
    reference = _engrave(case.reference, OUT / f"{case.name}-ref.png")

    def moved(field: str) -> str:
        if not before or field not in before:
            return ""
        was = before[field]
        now = getattr(result, field)
        if was == now:
            return "<span class='same'>no change</span>"
        better = now < was if field != "agree" else now > was
        return (f"<span class='{'up' if better else 'down'}'>"
                f"{now - was:+d} since the last run</span>")

    structure = ""
    if result.structure:
        structure = (
            f"<p class='warn'>The page prints <b>{result.staves_page}</b> staves and "
            f"homr wrote <b>{result.staves_homr}</b>. Every note is matched on its "
            f"staff, so from the first staff that diverges the rows below are "
            f"comparing different music &mdash; read them as one wrong answer about "
            f"the staves, not as many wrong notes.</p>")

    rows = "".join(
        f"<tr class='{row.kind if row.kind != 'agree' else ''}'>"
        f"<td>{html.escape(row.where)}</td><td>{html.escape(row.page)}</td>"
        f"<td>{html.escape(row.homr)}</td>"
        f"<td class='{'ok' if row.kind in ('agree', 'unison') else 'no'}'>"
        f"{html.escape(row.verdict)}</td></tr>"
        for row in result.rows)

    body = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(case.name)}</title><style>{STYLE}</style></head><body>
<p><a href="index.html">&larr; every case</a></p>
<h1>{html.escape(case.name)}</h1>
<p class="lead">{html.escape(case.origin)}
{'&mdash; committed fixture' if case.committed else ''}</p>
<div class="sum">
  <div><b>{result.agree}</b>agree {moved('agree')}</div>
  <div><b>{result.voice}</b>wrong voice {moved('voice')}</div>
  <div><b>{result.pitch}</b>wrong pitch {moved('pitch')}</div>
  <div><b>{result.size}</b>different number of notes {moved('size')}</div>
  <div><b>{result.timing}</b>beat shifted {moved('timing')}</div>
  <div><b>{result.unison}</b>unisons</div>
</div>
{structure}

<h3>Input &mdash; the printed page</h3>
<img src="{page.name}" alt="the printed system">
<h3>Output &mdash; what homr writes</h3>
{f'<img src="{output}" alt="homr">' if output else '<p class="lead">(could not engrave)</p>'}
<h3>Fixture &mdash; the reference</h3>
{f'<img src="{reference}" alt="reference">' if reference else '<p class="lead">(could not engrave)</p>'}

<h2>Every note, the page against homr's output</h2>
<p class="lead">Compared on staff position, so a score written an octave above
where it sounds is not counted wrong, and on which of the staff's voices a note
is in rather than the voice's number. This is homr's <b>output</b> &mdash; not
the noteheads it detected, which is a different layer and can disagree in either
direction.</p>
<table><tr><th>where</th><th>the page</th><th>homr</th><th></th></tr>
{rows}</table>
</body></html>"""
    target = OUT / f"{case.name}.html"
    target.write_text(body)
    return target.name


def index_page(entries: list[dict], tier: str) -> Path:
    """The table of every case in this run, with what moved since the last one."""
    OUT.mkdir(parents=True, exist_ok=True)
    total = {k: sum(e[k] for e in entries)
             for k in ("agree", "voice", "pitch", "size", "timing", "structure", "unison")}
    judged = total["agree"] + total["voice"] + total["pitch"]

    def cell(entry: dict, field: str) -> str:
        now = entry[field]
        was = (entry.get("before") or {}).get(field)
        if was is None or was == now:
            return str(now)
        better = now < was if field != "agree" else now > was
        return f"{now} <span class='{'up' if better else 'down'}'>({now - was:+d})</span>"

    rows = "".join(
        f"<tr><td><a href='{html.escape(e['page'])}'>{html.escape(e['name'])}</a></td>"
        f"<td>{cell(e, 'agree')}</td><td>{cell(e, 'voice')}</td>"
        f"<td>{cell(e, 'pitch')}</td><td>{cell(e, 'size')}</td>"
        f"<td>{cell(e, 'timing')}</td>"
        f"<td>{'%d vs %d' % (e['staves_page'], e['staves_homr']) if e['structure'] else ''}</td>"
        f"<td>{e['score']:.1f}%</td></tr>"
        for e in sorted(entries, key=lambda e: (-e["voice"] - e["pitch"], e["name"])))

    body = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>fixture check &mdash; {html.escape(tier)}</title><style>{STYLE}</style></head><body>
<h1>Fixture check &mdash; {html.escape(tier)}</h1>
<p class="lead">{len(entries)} case(s), each one printed system judged against a
reference built from its song's cleaned score. Sorted worst first. A number in
green or red is what changed since the last run of the same case.</p>
<div class="sum">
  <div><b>{total['agree']}</b>agree</div>
  <div><b>{total['voice']}</b>wrong voice</div>
  <div><b>{total['pitch']}</b>wrong pitch</div>
  <div><b>{total['size']}</b>different number of notes</div>
  <div><b>{total['timing']}</b>beat shifted</div>
  <div><b>{total['structure']}</b>case(s) with the wrong staves</div>
  <div><b>{100.0 * total['agree'] / judged if judged else 0:.1f}%</b>of judged notes right</div>
</div>
<p class="lead">A case whose staff count disagrees is one wrong answer about the
structure, and the note rows under it are then comparing different music &mdash;
read its counts as a consequence of that, not as many wrong notes.</p>
<table><tr><th>case</th><th>agree</th><th>wrong voice</th><th>wrong pitch</th>
<th>note count</th><th>beat shifted</th><th>staves</th><th>score</th></tr>
{rows}</table>
</body></html>"""
    target = OUT / "index.html"
    target.write_text(body)
    return target


def load_previous() -> dict:
    path = OUT / "results.json"
    return json.loads(path.read_text()) if path.exists() else {}


def save_results(entries: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(
        {e["name"]: {k: e[k] for k in ("agree", "voice", "pitch", "size", "timing",
                                       "structure", "unison")}
         for e in entries}, indent=1))
