"""Parses every experiments/results/*.csv into proposal Table 1 (selective
RMSE by abstention rate p) and renders it as a self-contained HTML page -
the same table shown in the "Selective RMSE Sweep" artifact, but regenerable
locally any time the CSVs change (e.g. once ml-25m or another baseline gets
filled in), with no dependency on that artifact still existing.

  python generate_table1.py                      # reads experiments/results/*.csv, writes table1.html next to them
  python generate_table1.py --results-dir X --out Y.html
  python generate_table1.py --open                # also open the result in your default browser

Every run_*.py script in this folder writes its own CSV (method, dataset, p,
selective_rmse, ...) - this script doesn't run anything itself, it only
reads whatever's already there, so it's safe to run at any point mid-sweep
and will just show fewer filled cells.
"""

from __future__ import annotations

import argparse
import csv
import json
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

P_RATES = (0.0, 0.05, 0.10, 0.20)

DATASETS = [
    {"key": "ml-100k", "label": "ML-100K", "n": "100K"},
    {"key": "ml-1m", "label": "ML-1M", "n": "1M"},
    {"key": "ml-25m", "label": "ML-25M", "n": "25M"},
    {"key": "douban", "label": "Douban", "n": ""},
    {"key": "amazon-games", "label": "Am. Games", "n": ""},
]
DATASET_KEYS = {d["key"] for d in DATASETS}

# Table row order + display metadata, matching proposal Table 1's layout
# (the two own-uncertainty rows first, then rand/supp baseline pairs). A
# method found in the CSVs but not listed here still gets a row - it's just
# appended at the end with a generic label - so this script never silently
# drops data it doesn't recognise.
ROW_SPECS = [
    {"method": "Ours (U_hat)", "name": "Ours (Û)", "tag": "joint uncertainty MF, own abstention", "hero": True},
    {"method": "UA-IMC (U_hat)", "name": "UA-IMC (Û)", "tag": "GNN + joint uncertainty loss", "note": "†"},
    {"method": "IGMC (rand)", "name": "IGMC (rand)", "tag": "random abstention", "note": "‡", "section": True},
    {"method": "IGMC (supp)", "name": "IGMC (supp)", "tag": "low-support abstention", "note": "‡"},
    {"method": "SoftImpute (rand)", "name": "SoftImpute (rand)", "tag": "random abstention", "section": True},
    {"method": "SoftImpute (supp)", "name": "SoftImpute (supp)", "tag": "low-support abstention"},
    {"method": "AutoRec (rand)", "name": "AutoRec (rand)", "tag": "random abstention", "section": True},
    {"method": "AutoRec (supp)", "name": "AutoRec (supp)", "tag": "low-support abstention"},
    {"method": "Plain MF (rand)", "name": "Plain MF (rand)", "tag": "random abstention", "section": True},
    {"method": "Plain MF (supp)", "name": "Plain MF (supp)", "tag": "low-support abstention"},
]
KNOWN_METHODS = {r["method"] for r in ROW_SPECS}


def load_results(results_dir: Path) -> dict[tuple[str, str], dict[float, float]]:
    """(method, dataset) -> {p: selective_rmse}, merged across every CSV in
    results_dir. Later files win on an exact duplicate (method,dataset,p)
    row, which only matters if a results file was manually edited."""
    data: dict[tuple[str, str], dict[float, float]] = {}
    for csv_path in sorted(results_dir.glob("*.csv")):
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    method = row["method"]
                    dataset = row["dataset"]
                    p = round(float(row["p"]), 4)
                    rmse = float(row["selective_rmse"])
                except (KeyError, ValueError):
                    continue  # not a Table 1-shaped row (e.g. a differently-formatted CSV dropped in the folder)
                data.setdefault((method, dataset), {})[p] = rmse
    return data


def build_rows(data: dict[tuple[str, str], dict[float, float]]) -> list[dict]:
    """ROW_SPECS plus a generic row for any method present in the data that
    ROW_SPECS doesn't already know about, so a new baseline's CSV shows up
    here without needing this script edited first."""
    rows = list(ROW_SPECS)
    extra_methods = sorted({m for (m, _ds) in data if m not in KNOWN_METHODS})
    for m in extra_methods:
        rows.append({"method": m, "name": m, "tag": ""})
    return rows


def series_for(data, method: str, dataset: str) -> list[float | None]:
    by_p = data.get((method, dataset))
    if not by_p:
        return None
    return [by_p.get(p) for p in P_RATES]


def best_per_column(data, rows) -> dict[tuple[str, int], float]:
    best: dict[tuple[str, int], float] = {}
    for ds in DATASETS:
        for pi in range(len(P_RATES)):
            values = []
            for r in rows:
                series = series_for(data, r["method"], ds["key"])
                if series and series[pi] is not None:
                    values.append(series[pi])
            if values:
                best[(ds["key"], pi)] = min(values)
    return best


def render(rows, data) -> str:
    best = best_per_column(data, rows)

    table_rows = []
    for r in rows:
        cells = []
        for ds in DATASETS:
            series = series_for(data, r["method"], ds["key"]) or [None] * len(P_RATES)
            cells.append(series)
        table_rows.append({
            "name": r["name"],
            "tag": r.get("tag", ""),
            "note": r.get("note", ""),
            "hero": bool(r.get("hero")),
            "section": bool(r.get("section")),
            "cells": cells,
        })

    payload = {
        "p_rates": list(P_RATES),
        "datasets": DATASETS,
        "rows": table_rows,
        "best": {f"{k[0]}|{k[1]}": v for k, v in best.items()},
    }
    return HTML_TEMPLATE.replace("__PAYLOAD__", json.dumps(payload))


HTML_TEMPLATE = r"""<title>Selective RMSE Sweep</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Serif:wght@500;600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
  :root {
    --bg: #F3F6F8; --surface: #FFFFFF; --surface-2: #EBF0F3;
    --ink: #1B2430; --ink-muted: #5B6672; --ink-faint: #97A1AB;
    --border: #DAE1E7; --accent: #2C6E8C; --accent-ink: #164559;
    --accent-soft: #DCEAF2; --accent-soft-2: #EEF5F9;
    --best: #C8641E; --best-soft: #FBEADD; --warn-ink: #8A5A22;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --bg: #14181D; --surface: #1B2027; --surface-2: #232A32;
      --ink: #E7ECF0; --ink-muted: #9BA7B2; --ink-faint: #66707B;
      --border: #313A43; --accent: #6FB4D6; --accent-ink: #BFE1F0;
      --accent-soft: #23404D; --accent-soft-2: #1D323C;
      --best: #E2934F; --best-soft: #3A2A18; --warn-ink: #E0B27C;
    }
  }
  :root[data-theme="dark"] {
    --bg: #14181D; --surface: #1B2027; --surface-2: #232A32;
    --ink: #E7ECF0; --ink-muted: #9BA7B2; --ink-faint: #66707B;
    --border: #313A43; --accent: #6FB4D6; --accent-ink: #BFE1F0;
    --accent-soft: #23404D; --accent-soft-2: #1D323C;
    --best: #E2934F; --best-soft: #3A2A18; --warn-ink: #E0B27C;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--ink); font-family: "IBM Plex Sans", system-ui, sans-serif; padding: 40px 32px 64px; }
  .page { max-width: 1180px; margin: 0 auto; }
  header { margin-bottom: 28px; }
  .eyebrow { font-family: "IBM Plex Mono", monospace; font-size: 12px; letter-spacing: .08em; text-transform: uppercase; color: var(--accent); margin: 0 0 10px; }
  h1 { font-family: "IBM Plex Serif", Georgia, serif; font-weight: 600; font-size: 30px; line-height: 1.15; margin: 0 0 8px; text-wrap: balance; }
  .subhead { color: var(--ink-muted); font-size: 14.5px; line-height: 1.55; max-width: 66ch; margin: 0; }
  .subhead code { font-family: "IBM Plex Mono", monospace; font-size: .92em; background: var(--surface-2); padding: 1px 5px; border-radius: 4px; }
  .table-wrap { margin-top: 26px; background: var(--surface); border: 1px solid var(--border); border-radius: 12px; overflow-x: auto; }
  table { border-collapse: collapse; width: 100%; min-width: 1240px; font-family: "IBM Plex Mono", monospace; font-variant-numeric: tabular-nums; }
  thead th { position: sticky; top: 0; background: var(--surface); z-index: 2; }
  .ds-head th { font-family: "IBM Plex Sans", sans-serif; font-weight: 600; font-size: 12.5px; color: var(--ink); text-align: center; padding: 14px 8px 6px; border-bottom: 1px solid var(--border); }
  .ds-head th.blank { border-bottom: 1px solid transparent; }
  .ds-head .n { display: block; font-family: "IBM Plex Mono", monospace; font-weight: 400; font-size: 10.5px; color: var(--ink-faint); margin-top: 2px; letter-spacing: .02em; }
  .p-head th { font-size: 11px; font-weight: 500; color: var(--ink-muted); text-align: center; padding: 4px 8px 10px; border-bottom: 1px solid var(--border); }
  .p-head th.blank { border-bottom: 1px solid var(--border); }
  th.rowlabel, td.rowlabel { position: sticky; left: 0; background: var(--surface); z-index: 1; text-align: left; font-family: "IBM Plex Sans", sans-serif; padding: 10px 16px 10px 18px; border-right: 1px solid var(--border); white-space: nowrap; }
  .method-name { font-weight: 600; font-size: 13.5px; color: var(--ink); }
  .method-tag { font-family: "IBM Plex Mono", monospace; font-size: 10.5px; color: var(--ink-faint); margin-left: 6px; }
  .method-note { display: inline-block; margin-left: 6px; font-family: "IBM Plex Mono", monospace; font-size: 10px; color: var(--warn-ink); vertical-align: super; }
  tbody tr.group-hero { background: var(--accent-soft-2); }
  tbody tr.group-hero td.rowlabel { background: var(--accent-soft-2); }
  tbody tr.group-hero .method-name { color: var(--accent-ink); }
  tbody tr:not(:last-child) td { border-bottom: 1px solid var(--border); }
  tbody tr.section-start td { border-top: 1px solid var(--border); }
  td.cell { text-align: center; padding: 10px 8px; font-size: 12.5px; color: var(--ink); }
  td.cell.best { background: var(--best-soft); color: var(--warn-ink); font-weight: 600; border-radius: 6px; }
  td.cell.na { color: var(--ink-faint); font-size: 11px; }
  .col-sep-left { border-left: 1px solid var(--border); }
  footer { margin-top: 22px; display: grid; gap: 10px; font-size: 12.5px; color: var(--ink-muted); line-height: 1.6; max-width: 90ch; }
  footer .note { display: flex; gap: 8px; }
  footer .mark { font-family: "IBM Plex Mono", monospace; color: var(--warn-ink); flex: none; }
  footer .best-key { display: inline-flex; align-items: center; gap: 6px; margin-top: 4px; }
  footer .best-key .swatch { width: 12px; height: 12px; border-radius: 3px; background: var(--best-soft); border: 1px solid var(--best); }
</style>
<div class="page">
  <header>
    <p class="eyebrow">IS470 &middot; Table 1 &middot; Selective RMSE by abstention rate p</p>
    <h1>Explicit recommendation, uncertainty &amp; abstention</h1>
    <p class="subhead">
      Generated by <code>experiments/generate_table1.py</code> from <code>experiments/results/*.csv</code> &mdash;
      every method-row scored at <code>p&nbsp;=&nbsp;0,&nbsp;.05,&nbsp;.10,&nbsp;.20</code> per the proposal's eq. (2)&ndash;(3).
      Lower is better; the shaded cell in each column is the best selective RMSE for that dataset&times;abstention-rate
      condition across every method run so far.
    </p>
  </header>
  <div class="table-wrap"><table id="tbl"></table></div>
  <footer id="foot"></footer>
</div>
<script>
const DATA = __PAYLOAD__;

function fmt(v) { return v === null || v === undefined ? null : v.toFixed(4); }

let thead = "<thead><tr class='ds-head'><th class='blank'></th>";
DATA.datasets.forEach(ds => {
  thead += `<th colspan="${DATA.p_rates.length}" class="col-sep-left">${ds.label}${ds.n ? `<span class="n">${ds.n} ratings</span>` : "<span class='n'>&nbsp;</span>"}</th>`;
});
thead += "</tr><tr class='p-head'><th class='blank'></th>";
DATA.datasets.forEach(() => {
  DATA.p_rates.forEach((p, i) => {
    thead += `<th${i === 0 ? " class='col-sep-left'" : ""}>p=${p === 0 ? "0" : p.toFixed(2)}</th>`;
  });
});
thead += "</tr></thead>";

let tbody = "<tbody>";
DATA.rows.forEach(r => {
  const cls = [r.hero ? "group-hero" : "", r.section ? "section-start" : ""].filter(Boolean).join(" ");
  tbody += `<tr class="${cls}"><td class="rowlabel"><span class="method-name">${r.name}</span>` +
    `${r.tag ? `<span class="method-tag">${r.tag}</span>` : ""}` +
    `${r.note ? `<span class="method-note">${r.note}</span>` : ""}</td>`;
  r.cells.forEach((series, di) => {
    const ds = DATA.datasets[di];
    series.forEach((v, pi) => {
      const sep = pi === 0 ? " col-sep-left" : "";
      if (v === null || v === undefined) {
        tbody += `<td class="cell na${sep}">&mdash;</td>`;
      } else {
        const b = DATA.best[ds.key + "|" + pi];
        const isBest = b !== undefined && Math.abs(v - b) < 1e-9;
        tbody += `<td class="cell${isBest ? " best" : ""}${sep}">${fmt(v)}</td>`;
      }
    });
  });
  tbody += "</tr>";
});
tbody += "</tbody>";

document.getElementById("tbl").innerHTML = thead + tbody;
document.getElementById("foot").innerHTML = `
  <div class="best-key"><span class="swatch"></span> shaded = best selective RMSE in that column, across all methods run so far</div>
  <div class="note"><span class="mark">&mdash;</span> not yet run for that method/dataset pair (see each run_*.py script's own docstring for why - a dense-matrix size limit, a deferred dataset, etc.)</div>
  <div class="note"><span class="mark">&dagger;</span> UA-IMC here is an approximation, not a faithful reproduction of Kasalicky/Ledent/Alves (RecSys'23) - see run_uaimc.py's module docstring.</div>
  <div class="note"><span class="mark">&Dagger;</span> IGMC uses a 4-way structural node role in place of DRNL, exact at 1-hop subgraphs (used here) - see run_igmc.py / week2/IGMC.ipynb.</div>
`;
</script>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "experiments" / "results")
    parser.add_argument("--out", type=Path, default=None, help="default: <results-dir>/table1.html")
    parser.add_argument("--open", action="store_true", help="open the generated page in your default browser")
    args = parser.parse_args()

    out_path = args.out or (args.results_dir / "table1.html")

    data = load_results(args.results_dir)
    rows = build_rows(data)
    html = render(rows, data)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")

    n_filled = sum(1 for series in data.values() for v in series.values() if v is not None)
    print(f"parsed {len(data)} (method, dataset) pairs, {n_filled} cells, from {args.results_dir}")
    print(f"wrote {out_path}")

    if args.open:
        webbrowser.open(out_path.resolve().as_uri())


if __name__ == "__main__":
    main()
