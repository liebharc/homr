"""Generate a self-contained HTML benchmark report from a SQLite NED benchmark database.

Usage:
    poetry run python -m validation.report benchmark.db
    poetry run python -m validation.report benchmark.db --output report.html
"""

import argparse
import json
import sqlite3
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------


def _fetch_summary(conn: sqlite3.Connection) -> dict[str, Any]:
    total, succeeded, failed = conn.execute(
        "SELECT COUNT(*), "
        "COUNT(CASE WHEN error IS NULL THEN 1 END), "
        "COUNT(CASE WHEN error IS NOT NULL THEN 1 END) "
        "FROM samples"
    ).fetchone()

    rows = conn.execute(
        "SELECT ned, rhythm_ned, pitch_ned, lift_ned, articulation_ned, slur_ned "
        "FROM samples WHERE error IS NULL"
    ).fetchall()

    component_stats: dict[str, dict[str, float]] = {}
    if rows:
        names = ["ned", "rhythm_ned", "pitch_ned", "lift_ned", "articulation_ned", "slur_ned"]
        for i, name in enumerate(names):
            vals = sorted(r[i] for r in rows)
            component_stats[name] = {
                "mean": statistics.mean(vals),
                "median": statistics.median(vals),
                "min": vals[0],
                "max": vals[-1],
            }

    return {
        "total": total,
        "succeeded": succeeded,
        "failed": failed,
        "component_stats": component_stats,
    }


def _fetch_ned_distribution(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute("SELECT ned FROM samples WHERE error IS NULL").fetchall()
    buckets = [0] * 11
    for (ned,) in rows:
        idx = min(int(ned * 10), 10)
        buckets[idx] += 1
    labels = [f"{i * 10}–{i * 10 + 10}%" for i in range(10)] + [">100%"]
    return [{"label": lb, "count": c} for lb, c in zip(labels, buckets, strict=False)]


def _fetch_event_breakdown(conn: sqlite3.Connection) -> dict[str, int]:
    rows = conn.execute(
        "SELECT event_type, COUNT(*) FROM token_events GROUP BY event_type"
    ).fetchall()
    return {r[0]: r[1] for r in rows}


def _fetch_top_events(
    conn: sqlite3.Connection,
    event_type: str,
    field: str,
    limit: int = 20,
) -> list[dict[str, Any]]:
    # For deletes the interesting side is the expected token (what was missing).
    # For inserts it's the actual token (what was hallucinated).
    # Column names are built from a fixed internal set - no injection risk.
    col = f"exp_{field}" if event_type == "delete" else f"act_{field}"
    rows = conn.execute(
        f"SELECT {col}, COUNT(*) AS cnt FROM token_events "  # noqa: S608
        f"WHERE event_type = ? AND {col} IS NOT NULL "
        f"GROUP BY {col} ORDER BY cnt DESC LIMIT ?",
        (event_type, limit),
    ).fetchall()
    return [{"token": r[0], "count": r[1]} for r in rows]


def _fetch_top_substitutions(
    conn: sqlite3.Connection,
    field: str,
    limit: int = 20,
) -> list[dict[str, Any]]:
    exp_col = f"exp_{field}"
    act_col = f"act_{field}"
    rows = conn.execute(
        f"SELECT {exp_col}, {act_col}, COUNT(*) AS cnt FROM token_events "  # noqa: S608
        f"WHERE event_type = 'substitute' "
        f"AND {exp_col} IS NOT NULL AND {act_col} IS NOT NULL "
        f"AND {exp_col} != {act_col} "
        f"GROUP BY {exp_col}, {act_col} ORDER BY cnt DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [{"expected": r[0], "actual": r[1], "count": r[2]} for r in rows]


def _fetch_error_rates(
    conn: sqlite3.Connection,
    field: str,
    min_count: int = 20,
    limit: int = 20,
) -> list[dict[str, Any]]:
    exp_col = f"exp_{field}"
    rows = conn.execute(
        f"SELECT {exp_col}, COUNT(*) AS total, "  # noqa: S608
        "SUM(CASE WHEN event_type != 'match' THEN 1 ELSE 0 END) AS errors "
        f"FROM token_events WHERE {exp_col} IS NOT NULL "
        f"GROUP BY {exp_col} HAVING total >= ? "
        "ORDER BY CAST(errors AS REAL) / total DESC LIMIT ?",
        (min_count, limit),
    ).fetchall()
    return [
        {
            "token": r[0],
            "total": r[1],
            "errors": r[2],
            "rate": round(r[2] / r[1] * 100, 1),
        }
        for r in rows
    ]


def _build_report_data(conn: sqlite3.Connection, db_path: Path) -> dict[str, Any]:
    return {
        "db_name": db_path.name,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),  # noqa: UP017
        "summary": _fetch_summary(conn),
        "ned_distribution": _fetch_ned_distribution(conn),
        "event_breakdown": _fetch_event_breakdown(conn),
        "missing": {
            "rhythm": _fetch_top_events(conn, "delete", "rhythm"),
            "pitch": _fetch_top_events(conn, "delete", "pitch"),
            "lift": _fetch_top_events(conn, "delete", "lift"),
        },
        "hallucinated": {
            "rhythm": _fetch_top_events(conn, "insert", "rhythm"),
            "pitch": _fetch_top_events(conn, "insert", "pitch"),
            "lift": _fetch_top_events(conn, "insert", "lift"),
        },
        "substitutions": {
            "rhythm": _fetch_top_substitutions(conn, "rhythm"),
            "pitch": _fetch_top_substitutions(conn, "pitch"),
        },
        "error_rates": {
            "rhythm": _fetch_error_rates(conn, "rhythm"),
            "pitch": _fetch_error_rates(conn, "pitch"),
        },
    }


# ---------------------------------------------------------------------------
# HTML template  (placeholder: __DATA_JSON__)
# ---------------------------------------------------------------------------

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title id="page-title">OMR Benchmark Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
<style>
* {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}
body {
  font-family: system-ui, Segoe UI, Helvetica, Arial, sans-serif;
  background: #f1f3f5;
  color: #212529;
  font-size: 14px;
}
h2 {
  font-size: 1rem;
  font-weight: 600;
  margin-bottom: .75rem;
  color: #343a40;
}
h3 {
  font-size: .85rem;
  font-weight: 600;
  margin-bottom: .5rem;
  color: #495057;
}
header {
  background: #1a1a2e;
  color: #fff;
  padding: 1.25rem 2rem;
  display: flex;
  align-items: baseline;
  gap: 1.5rem;
}
header h1 {
  font-size: 1.25rem;
  font-weight: 700;
  letter-spacing: -.01em;
}
header .meta {
  font-size: .8rem;
  opacity: .65;
}
main {
  max-width: 1440px;
  margin: 0 auto;
  padding: 1.5rem 2rem;
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}
.cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 1rem;
}
.card {
  background: #fff;
  border-radius: 8px;
  padding: 1rem 1.25rem;
  box-shadow: 0 1px 3px rgba(0,0,0,.08);
}
.card .label {
  font-size: .75rem;
  text-transform: uppercase;
  letter-spacing: .05em;
  color: #868e96;
  margin-bottom: .25rem;
}
.card .value {
  font-size: 1.6rem;
  font-weight: 700;
  line-height: 1.1;
}
.card .sub {
  font-size: .75rem;
  color: #868e96;
  margin-top: .2rem;
}
.card.good .value { color: #2b8a3e; }
.card.warn .value { color: #e67700; }
.card.bad  .value { color: #c92a2a; }
.chart-grid {
  display: grid;
  grid-template-columns: 2fr 1.5fr 1fr;
  gap: 1rem;
}
.chart-box {
  background: #fff;
  border-radius: 8px;
  padding: 1.25rem;
  box-shadow: 0 1px 3px rgba(0,0,0,.08);
}
.chart-box canvas { max-height: 220px; }
.panel {
  background: #fff;
  border-radius: 8px;
  padding: 1.25rem;
  box-shadow: 0 1px 3px rgba(0,0,0,.08);
}
.panel h2 .sub { font-weight: 400; color: #868e96; }
.tabs {
  display: flex;
  gap: .375rem;
  margin-bottom: .75rem;
  flex-wrap: wrap;
}
.tab-btn {
  background: #f1f3f5;
  border: 1px solid #dee2e6;
  border-radius: 6px;
  padding: .3rem .75rem;
  font-size: .8rem;
  cursor: pointer;
  color: #495057;
  transition: background .15s, color .15s;
}
.tab-btn.active {
  background: #4361ee;
  color: #fff;
  border-color: #4361ee;
}
.tab-pane { display: none; }
.tab-pane.active { display: block; }
.bar-table {
  width: 100%;
  border-collapse: collapse;
  font-size: .8rem;
}
.bar-table tr:hover { background: #f8f9fa; }
.bar-table td { padding: .25rem .5rem; vertical-align: middle; }
.token-cell {
  font-family: ui-monospace, monospace;
  max-width: 200px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: #1971c2;
}
.arrow-cell {
  color: #868e96;
  padding: 0 .25rem;
  text-align: center;
  width: 1.5rem;
}
.bar-cell { width: 40%; padding: .25rem .5rem; }
.count-cell { text-align: right; white-space: nowrap; color: #495057; width: 4rem; }
.dim { color: #adb5bd; }
.bar {
  height: 12px;
  border-radius: 3px;
  background: #4361ee;
  min-width: 2px;
}
.bar.red    { background: #e03131; }
.bar.green  { background: #2f9e44; }
.bar.purple { background: #7048e8; }
.bar.orange { background: #e67700; }
.empty { color: #adb5bd; font-size: .8rem; padding: .5rem 0; }
.two-col {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
}
@media (max-width: 900px) {
  .chart-grid { grid-template-columns: 1fr 1fr; }
  .two-col    { grid-template-columns: 1fr; }
}
@media (max-width: 600px) {
  .chart-grid { grid-template-columns: 1fr; }
  .cards      { grid-template-columns: 1fr 1fr; }
}
</style>
</head>
<body>
<header>
  <h1>OMR Benchmark Report</h1>
  <span class="meta" id="hdr-meta"></span>
</header>
<main>

<!-- SUMMARY CARDS -->
<div class="cards" id="cards"></div>

<!-- OVERVIEW CHARTS -->
<div class="chart-grid">
  <div class="chart-box">
    <h2>NED by Component (mean &amp; median)</h2>
    <canvas id="nedCompChart"></canvas>
  </div>
  <div class="chart-box">
    <h2>NED Distribution</h2>
    <canvas id="nedDistChart"></canvas>
  </div>
  <div class="chart-box">
    <h2>Token Event Types</h2>
    <canvas id="eventChart"></canvas>
  </div>
</div>

<!-- MISSING TOKENS -->
<div class="panel">
  <h2>Top Missing Tokens
    <span class="sub">(deleted from expected)</span>
  </h2>
  <div class="tabs" id="miss-tabs">
    <button class="tab-btn active"
      onclick="switchTab('miss','rhythm')">Rhythm</button>
    <button class="tab-btn"
      onclick="switchTab('miss','pitch')">Pitch</button>
    <button class="tab-btn"
      onclick="switchTab('miss','lift')">Lift / Accidental</button>
  </div>
  <div class="tab-pane active" id="miss-pane-rhythm"></div>
  <div class="tab-pane" id="miss-pane-pitch"></div>
  <div class="tab-pane" id="miss-pane-lift"></div>
</div>

<!-- HALLUCINATED TOKENS -->
<div class="panel">
  <h2>Top Hallucinated Tokens
    <span class="sub">(inserted into actual)</span>
  </h2>
  <div class="tabs" id="hall-tabs">
    <button class="tab-btn active"
      onclick="switchTab('hall','rhythm')">Rhythm</button>
    <button class="tab-btn"
      onclick="switchTab('hall','pitch')">Pitch</button>
    <button class="tab-btn"
      onclick="switchTab('hall','lift')">Lift / Accidental</button>
  </div>
  <div class="tab-pane active" id="hall-pane-rhythm"></div>
  <div class="tab-pane" id="hall-pane-pitch"></div>
  <div class="tab-pane" id="hall-pane-lift"></div>
</div>

<!-- SUBSTITUTIONS -->
<div class="panel">
  <h2>Top Confusions
    <span class="sub">(expected &rarr; actual)</span>
  </h2>
  <div class="tabs" id="sub-tabs">
    <button class="tab-btn active"
      onclick="switchTab('sub','rhythm')">Rhythm</button>
    <button class="tab-btn"
      onclick="switchTab('sub','pitch')">Pitch</button>
  </div>
  <div class="tab-pane active" id="sub-pane-rhythm"></div>
  <div class="tab-pane" id="sub-pane-pitch"></div>
</div>

<!-- ERROR RATES -->
<div class="panel">
  <h2>Most Error-Prone Tokens
    <span class="sub">(min 20 occurrences)</span>
  </h2>
  <div class="two-col">
    <div>
      <h3>By Rhythm</h3>
      <div id="rate-pane-rhythm"></div>
    </div>
    <div>
      <h3>By Pitch</h3>
      <div id="rate-pane-pitch"></div>
    </div>
  </div>
</div>

</main>
<script>
const DATA = __DATA_JSON__;

// ---- utilities ----

function pct(v) {
  return v == null ? 'n/a' : (v * 100).toFixed(1) + '%';
}

function esc(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function switchTab(group, tab) {
  document.querySelectorAll('#' + group + '-tabs .tab-btn')
    .forEach(b => b.classList.remove('active'));
  document.querySelectorAll('[id^="' + group + '-pane-"]')
    .forEach(p => p.classList.remove('active'));
  const btn = document.querySelector(
    '#' + group + '-tabs .tab-btn[onclick*="\'' + tab + '\'"]'
  );
  if (btn) btn.classList.add('active');
  const pane = document.getElementById(group + '-pane-' + tab);
  if (pane) pane.classList.add('active');
}

// ---- header ----

(function () {
  const s = DATA.summary;
  document.getElementById('hdr-meta').textContent =
    DATA.db_name + '  ·  ' + DATA.generated_at +
    '  ·  ' + s.total + ' samples';
  document.title = 'OMR Benchmark — ' + DATA.db_name;
})();

// ---- summary cards ----

(function () {
  const s = DATA.summary;
  const cs = s.component_stats || {};
  const ned = cs.ned || {};
  const rate = s.total ? (s.succeeded / s.total * 100).toFixed(0) : 0;
  const cards = [
    {
      label: 'Succeeded',
      value: s.succeeded,
      sub: rate + '% of ' + s.total,
      cls: s.failed === 0 ? 'good' : (s.succeeded === 0 ? 'bad' : ''),
    },
    {
      label: 'Failed',
      value: s.failed,
      sub: 'samples with errors',
      cls: s.failed > 0 ? 'warn' : 'good',
    },
    {
      label: 'Mean NED',
      value: pct(ned.mean),
      sub: 'median ' + pct(ned.median),
      cls: '',
    },
    {
      label: 'Best NED',
      value: pct(ned.min),
      sub: 'lowest per sample',
      cls: 'good',
    },
    {
      label: 'Worst NED',
      value: pct(ned.max),
      sub: 'highest per sample',
      cls: 'bad',
    },
  ];
  document.getElementById('cards').innerHTML = cards.map(function (c) {
    return '<div class="card ' + c.cls + '">' +
      '<div class="label">' + esc(c.label) + '</div>' +
      '<div class="value">' + esc(String(c.value)) + '</div>' +
      '<div class="sub">' + esc(c.sub) + '</div>' +
      '</div>';
  }).join('');
})();

// ---- NED component bar chart ----

(function () {
  const cs = DATA.summary.component_stats || {};
  const fields = [
    'ned', 'rhythm_ned', 'pitch_ned',
    'lift_ned', 'articulation_ned', 'slur_ned',
  ];
  const labels = ['Overall', 'Rhythm', 'Pitch', 'Lift', 'Articulation', 'Slur'];
  function toP(f, key) {
    return cs[f] ? +(cs[f][key] * 100).toFixed(2) : null;
  }
  new Chart(document.getElementById('nedCompChart'), {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'Mean',
          data: fields.map(f => toP(f, 'mean')),
          backgroundColor: '#4361ee',
        },
        {
          label: 'Median',
          data: fields.map(f => toP(f, 'median')),
          backgroundColor: '#74c0fc',
        },
      ],
    },
    options: {
      responsive: true,
      animation: false,
      scales: {
        y: { beginAtZero: true, ticks: { callback: v => v + '%' } },
      },
      plugins: {
        legend: { position: 'bottom', labels: { boxWidth: 12 } },
      },
    },
  });
})();

// ---- NED distribution histogram ----

(function () {
  const dist = DATA.ned_distribution;
  new Chart(document.getElementById('nedDistChart'), {
    type: 'bar',
    data: {
      labels: dist.map(d => d.label),
      datasets: [{
        label: 'Samples',
        data: dist.map(d => d.count),
        backgroundColor: '#4361ee',
      }],
    },
    options: {
      responsive: true,
      animation: false,
      scales: { y: { beginAtZero: true } },
      plugins: { legend: { display: false } },
    },
  });
})();

// ---- event breakdown donut ----

(function () {
  const ev = DATA.event_breakdown;
  const order = ['match', 'delete', 'insert', 'substitute'];
  const colors = {
    match: '#2f9e44',
    delete: '#e03131',
    insert: '#e67700',
    substitute: '#7048e8',
  };
  const keys = order.filter(k => k in ev);
  new Chart(document.getElementById('eventChart'), {
    type: 'doughnut',
    data: {
      labels: keys.map(k => k.charAt(0).toUpperCase() + k.slice(1)),
      datasets: [{
        data: keys.map(k => ev[k]),
        backgroundColor: keys.map(k => colors[k]),
      }],
    },
    options: {
      responsive: true,
      animation: false,
      plugins: {
        legend: {
          position: 'bottom',
          labels: { boxWidth: 12, font: { size: 11 } },
        },
      },
    },
  });
})();

// ---- bar table helpers ----

function renderBarTable(id, items, barCls) {
  const el = document.getElementById(id);
  if (!el) return;
  if (!items || !items.length) {
    el.innerHTML = '<p class="empty">No data</p>';
    return;
  }
  const max = items[0].count;
  el.innerHTML = '<table class="bar-table"><tbody>' +
    items.map(function (item) {
      const w = max ? Math.round(item.count / max * 100) : 0;
      return '<tr>' +
        '<td class="token-cell">' + esc(item.token) + '</td>' +
        '<td class="bar-cell">' +
        '<div class="bar ' + barCls + '" style="width:' + w + '%"></div>' +
        '</td>' +
        '<td class="count-cell">' + item.count + '</td>' +
        '</tr>';
    }).join('') + '</tbody></table>';
}

function renderSubTable(id, items, barCls) {
  const el = document.getElementById(id);
  if (!el) return;
  if (!items || !items.length) {
    el.innerHTML = '<p class="empty">No data</p>';
    return;
  }
  const max = items[0].count;
  el.innerHTML = '<table class="bar-table"><tbody>' +
    items.map(function (item) {
      const w = max ? Math.round(item.count / max * 100) : 0;
      return '<tr>' +
        '<td class="token-cell">' + esc(item.expected) + '</td>' +
        '<td class="arrow-cell">&rarr;</td>' +
        '<td class="token-cell">' + esc(item.actual) + '</td>' +
        '<td class="bar-cell">' +
        '<div class="bar ' + barCls + '" style="width:' + w + '%"></div>' +
        '</td>' +
        '<td class="count-cell">' + item.count + '</td>' +
        '</tr>';
    }).join('') + '</tbody></table>';
}

function renderRateTable(id, items) {
  const el = document.getElementById(id);
  if (!el) return;
  if (!items || !items.length) {
    el.innerHTML = '<p class="empty">No data (need ≥20 occurrences)</p>';
    return;
  }
  el.innerHTML = '<table class="bar-table"><tbody>' +
    items.map(function (item) {
      return '<tr>' +
        '<td class="token-cell">' + esc(item.token) + '</td>' +
        '<td class="bar-cell">' +
        '<div class="bar orange" style="width:' + item.rate + '%"></div>' +
        '</td>' +
        '<td class="count-cell">' + item.rate + '%</td>' +
        '<td class="count-cell dim">' + item.errors + '/' + item.total + '</td>' +
        '</tr>';
    }).join('') + '</tbody></table>';
}

// ---- populate all tables ----

renderBarTable('miss-pane-rhythm',  DATA.missing.rhythm,       'red');
renderBarTable('miss-pane-pitch',   DATA.missing.pitch,        'red');
renderBarTable('miss-pane-lift',    DATA.missing.lift,         'red');

renderBarTable('hall-pane-rhythm',  DATA.hallucinated.rhythm,  'green');
renderBarTable('hall-pane-pitch',   DATA.hallucinated.pitch,   'green');
renderBarTable('hall-pane-lift',    DATA.hallucinated.lift,    'green');

renderSubTable('sub-pane-rhythm',   DATA.substitutions.rhythm, 'purple');
renderSubTable('sub-pane-pitch',    DATA.substitutions.pitch,  'purple');

renderRateTable('rate-pane-rhythm', DATA.error_rates.rhythm);
renderRateTable('rate-pane-pitch',  DATA.error_rates.pitch);
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_report(db_path: Path, output_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    try:
        data = _build_report_data(conn, db_path)
    finally:
        conn.close()

    json_str = json.dumps(data, ensure_ascii=False)
    html = _HTML.replace("__DATA_JSON__", json_str)
    output_path.write_text(html, encoding="utf-8")
    print(f"Report written to {output_path}")  # noqa: T201


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an HTML report from a NED benchmark SQLite database."
    )
    parser.add_argument("db", help="Path to the SQLite benchmark database.")
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output HTML file path (default: <db>.html).",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        parser.error(f"Database not found: {db_path}")

    output_path = Path(args.output) if args.output else db_path.with_suffix(".html")
    generate_report(db_path, output_path)


if __name__ == "__main__":
    main()
