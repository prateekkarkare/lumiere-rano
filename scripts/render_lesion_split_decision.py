"""
Render the lesion-splitting decision memo (connectivity + volume floor) as standalone HTML.

Reads ``output/connectivity/connectivity_sweep.json`` (produced by ``sweep_connectivity.py``)
and writes ``docs/lesion_split_decision.html``. Inline SVG, no plotting dependency -- same
constraints and palette as ``render_volume_audit.py``.

Usage:
    .venv/bin/python scripts/render_lesion_split_decision.py
"""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SWEEP = ROOT / "output" / "connectivity" / "connectivity_sweep.json"
OUT = ROOT / "docs" / "lesion_split_decision.html"

CHOSEN_FLOOR = "20"
ACCENT = "#b5533a"
INK3 = "#6b6b64"


def esc(s) -> str:
    return html.escape(str(s))


def axes(w, h, pad_l=44, pad_b=26, pad_t=10, pad_r=10):
    return pad_l, w - pad_r, pad_t, h - pad_b


def chart_tradeoff(d) -> str:
    """Agreement % and timepoints-zeroed against the floor -- the actual trade."""
    live = [r for r in d["records"] if r["enhancing_mm3"] > 0]
    floors = [str(f) for f in d["floors_mm3"]]
    agree = [100 * sum(1 for r in live
                       if r["counts_by_floor_26"][f] == r["counts_by_floor_6"][f]) / len(live)
             for f in floors]
    zeroed = [sum(1 for r in live if r["counts_by_floor_26"][f] == 0) for f in floors]

    w, h = 500, 200
    x0, x1, y0, y1 = axes(w, h)
    n = len(floors)
    xs = [x0 + (x1 - x0) * i / (n - 1) for i in range(n)]
    sy_a = lambda v: y1 - (y1 - y0) * v / 100          # noqa: E731
    # both series share the 0-100% axis: a second hidden scale would let a reader
    # mistake "14 of 83 timepoints emptied" for "all of them".
    zpct = [100 * z / len(live) for z in zeroed]

    p = []
    for gy in range(0, 101, 25):
        p.append(f'<line x1="{x0}" y1="{sy_a(gy):.1f}" x2="{x1}" y2="{sy_a(gy):.1f}" '
                 f'stroke="var(--grid)" stroke-width="1"/>')
        p.append(f'<text x="{x0-7}" y="{sy_a(gy)+4:.1f}" text-anchor="end" font-size="10" '
                 f'fill="var(--ink-4)">{gy}%</text>')
    ci = floors.index(CHOSEN_FLOOR)
    p.append(f'<rect x="{xs[ci]-13:.1f}" y="{y0}" width="26" height="{y1-y0}" '
             f'fill="{ACCENT}" opacity=".07"/>')
    p.append('<polyline fill="none" stroke="var(--ink-2)" stroke-width="2" points="'
             + " ".join(f"{x:.1f},{sy_a(v):.1f}" for x, v in zip(xs, agree)) + '"/>')
    p.append(f'<polyline fill="none" stroke="{ACCENT}" stroke-width="2" stroke-dasharray="4 3" '
             'points="' + " ".join(f"{x:.1f},{sy_a(v):.1f}" for x, v in zip(xs, zpct)) + '"/>')
    for x, a_, z_, zc in zip(xs, agree, zeroed, zpct):
        p.append(f'<circle cx="{x:.1f}" cy="{sy_a(a_):.1f}" r="2.6" fill="var(--ink-2)"/>')
        p.append(f'<circle cx="{x:.1f}" cy="{sy_a(zc):.1f}" r="2.6" fill="{ACCENT}"/>')
        if z_:
            p.append(f'<text x="{x:.1f}" y="{sy_a(zc)-7:.1f}" text-anchor="middle" '
                     f'font-size="9" fill="{ACCENT}">{z_}</text>')
    for x, f in zip(xs, floors):
        p.append(f'<text x="{x:.1f}" y="{y1+15}" text-anchor="middle" font-size="10" '
                 f'fill="var(--ink-4)">{f}</text>')
    p.append(f'<text x="{(x0+x1)/2:.0f}" y="{h-1}" text-anchor="middle" font-size="10.5" '
             f'fill="var(--ink-3)">volume floor (mm³)</text>')
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" '
            f'aria-label="connectivity agreement and timepoints zeroed versus volume floor">'
            + "".join(p) + "</svg>")


def chart_counts(d) -> str:
    """Lesion-count distribution before and after the floor."""
    live = [r for r in d["records"] if r["enhancing_mm3"] > 0]
    def dist(f):
        c = Counter(min(r["counts_by_floor_26"][f], 8) for r in live)
        return [c.get(i, 0) for i in range(0, 9)]
    a, b = dist("0"), dist(CHOSEN_FLOOR)
    w, h = 500, 190
    x0, x1, y0, y1 = axes(w, h, pad_l=34)
    mx = max(max(a), max(b))
    bw = (x1 - x0) / 9
    p = []
    for i in range(9):
        for k, (vals, col, off) in enumerate(
                ((a, "var(--ink-4)", 0.08), (b, ACCENT, 0.52))):
            v = vals[i]
            bh = (y1 - y0) * v / mx
            p.append(f'<rect x="{x0+bw*i+bw*off:.1f}" y="{y1-bh:.1f}" width="{bw*0.40:.1f}" '
                     f'height="{bh:.1f}" fill="{col}" rx="1.5"/>')
            if v:
                p.append(f'<text x="{x0+bw*i+bw*(off+0.20):.1f}" y="{y1-bh-3:.1f}" '
                         f'text-anchor="middle" font-size="9" fill="var(--ink-3)">{v}</text>')
        lab = "8+" if i == 8 else str(i)
        p.append(f'<text x="{x0+bw*i+bw*0.5:.1f}" y="{y1+15}" text-anchor="middle" '
                 f'font-size="10" fill="var(--ink-4)">{lab}</text>')
    p.append(f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="var(--line)"/>')
    p.append(f'<text x="{(x0+x1)/2:.0f}" y="{h-1}" text-anchor="middle" font-size="10.5" '
             f'fill="var(--ink-3)">lesions in the timepoint (26-connectivity)</text>')
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" '
            f'aria-label="lesion count distribution before and after the floor">'
            + "".join(p) + "</svg>")


def chart_dominance(d) -> str:
    """Sorted share of enhancing volume held by the largest lesion, with the 90% line."""
    live = [r for r in d["records"] if r["enhancing_mm3"] > 0]
    fr = sorted(r["largest_volume_fraction"] for r in live)
    w, h = 500, 190
    x0, x1, y0, y1 = axes(w, h, pad_l=40)
    n = len(fr)
    sx = lambda i: x0 + (x1 - x0) * i / (n - 1)   # noqa: E731
    sy = lambda v: y1 - (y1 - y0) * v             # noqa: E731
    p = []
    for gy in (0, .25, .5, .75, 1.0):
        p.append(f'<line x1="{x0}" y1="{sy(gy):.1f}" x2="{x1}" y2="{sy(gy):.1f}" '
                 f'stroke="var(--grid)"/>')
        p.append(f'<text x="{x0-7}" y="{sy(gy)+4:.1f}" text-anchor="end" font-size="10" '
                 f'fill="var(--ink-4)">{gy:.0%}</text>')
    below = sum(1 for v in fr if v < .90)
    p.append(f'<rect x="{x0}" y="{y0}" width="{sx(below)-x0:.1f}" height="{y1-y0}" '
             f'fill="{ACCENT}" opacity=".07"/>')
    p.append('<polyline fill="none" stroke="var(--ink-2)" stroke-width="2" points="'
             + " ".join(f"{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(fr)) + '"/>')
    p.append(f'<line x1="{x0}" y1="{sy(.9):.1f}" x2="{x1}" y2="{sy(.9):.1f}" '
             f'stroke="{ACCENT}" stroke-width="1.4" stroke-dasharray="5 3"/>')
    p.append(f'<text x="{x1-4}" y="{sy(.9)-6:.1f}" text-anchor="end" font-size="10" '
             f'fill="{ACCENT}">90% of enhancing volume in one lesion</text>')
    # sits low inside the shaded band: the top of the panel is taken by the 100% gridline
    # label and the 90% annotation, and text stacked there reads as one collided line.
    p.append(f'<text x="{(x0+sx(below))/2:.0f}" y="{y1-9}" text-anchor="middle" font-size="10.5" '
             f'fill="{ACCENT}">{below} timepoints where one target is not enough</text>')
    p.append(f'<text x="{(x0+x1)/2:.0f}" y="{h-1}" text-anchor="middle" font-size="10.5" '
             f'fill="var(--ink-3)">the {n} timepoints with enhancing disease, sorted</text>')
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" '
            f'aria-label="share of enhancing volume in the largest lesion, sorted">'
            + "".join(p) + "</svg>")


CSS = """:root{--bg:#f7f7f5;--panel:#fff;--strip:#f2f2ef;--grid:#e6e6e1;--ink-1:#1a1a18;
--ink-2:#3d3d39;--ink-3:#6b6b64;--ink-4:#9a9a92;--line:#e2e2dc;--accent:#b5533a;
--warn-bg:#fdf4ec;--warn-br:#e8b58a;--ok-bg:#f0f4ee;--ok-br:#b9c9ac;}
@media (prefers-color-scheme:dark){:root{--bg:#14140f;--panel:#1c1c18;--strip:#232320;
--grid:#2f2f2a;--ink-1:#f0efe9;--ink-2:#d2d1c9;--ink-3:#9a998f;--ink-4:#6e6d64;--line:#2f2f2a;
--accent:#e0876a;--warn-bg:#2a1f16;--warn-br:#6b4a2e;--ok-bg:#1b2318;--ok-br:#3f5433;}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink-1);
font:15px/1.6 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif;}
.wrap{max-width:1080px;margin:0 auto;padding:44px 22px 90px}
h1{font-size:27px;letter-spacing:-.02em;margin:0 0 6px}
h2{font-size:18px;letter-spacing:-.01em;margin:46px 0 6px;padding-top:20px;
border-top:1px solid var(--line)}
h3{font-size:13px;text-transform:uppercase;letter-spacing:.07em;color:var(--ink-3);margin:26px 0 8px}
p{color:var(--ink-2);margin:9px 0;max-width:74ch}
.sub{color:var(--ink-3);font-size:14px;margin-bottom:26px;max-width:74ch}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:18px 20px;margin:16px 0}
.fig{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:20px;margin:14px 0;overflow-x:auto}
.fig .cap{font-size:12px;color:var(--ink-3);margin-top:10px;max-width:74ch}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(168px,1fr));gap:12px;margin:22px 0}
.kpi{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.kpi .n{font-size:23px;font-weight:700;letter-spacing:-.02em}
.kpi .l{font-size:11.5px;color:var(--ink-3);margin-top:3px;line-height:1.35}
.warn{background:var(--warn-bg);border:1px solid var(--warn-br);border-radius:10px;padding:16px 20px;margin:20px 0}
.warn b{color:var(--accent)}
.ok{background:var(--ok-bg);border:1px solid var(--ok-br);border-radius:10px;padding:16px 20px;margin:20px 0}
.decision{background:var(--panel);border:2px solid var(--accent);border-radius:10px;padding:20px 22px;margin:22px 0}
.decision .n{font-size:20px;font-weight:700;letter-spacing:-.01em;margin-bottom:6px}
table{border-collapse:collapse;width:100%;font-size:13px;margin:10px 0}
th,td{text-align:right;padding:6px 10px;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}
th{color:var(--ink-3);font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.05em}
td.mono,th.mono{font-variant-numeric:tabular-nums}
tr.hi td{background:var(--strip);font-weight:600}
.legend{display:flex;flex-wrap:wrap;gap:16px;margin:4px 0 14px;font-size:12px;color:var(--ink-3)}
.legend i{display:inline-block;width:14px;height:3px;border-radius:2px;margin-right:6px;vertical-align:3px}
code{background:var(--strip);padding:1px 5px;border-radius:4px;font-size:12.5px}
ul{color:var(--ink-2);max-width:74ch}li{margin:5px 0}
.foot{color:var(--ink-4);font-size:12px;margin-top:40px;border-top:1px solid var(--line);padding-top:16px}"""


def build(d: dict) -> str:
    recs = d["records"]
    live = [r for r in recs if r["enhancing_mm3"] > 0]
    floors = [str(f) for f in d["floors_mm3"]]
    F = CHOSEN_FLOOR
    total_vol = sum(r["enhancing_mm3"] for r in recs)

    def agree(f):
        return sum(1 for r in live if r["counts_by_floor_26"][f] == r["counts_by_floor_6"][f])

    disc = sum(r["volume_discarded_by_floor_26"][F] for r in recs)
    zeroed = [r for r in live if r["counts_by_floor_26"][F] == 0]
    nd = [r for r in live if r["largest_volume_fraction"] < .90]
    tot_sl = sum(r["largest_n_slices"] for r in live)
    tot_fr = sum(r["largest_slices_fragmented"] for r in live)
    # cases where 6-connectivity ends up with FEWER surviving lesions than 26
    fewer6 = [r for r in live if r["counts_by_floor_26"][F] > r["counts_by_floor_6"][F]]

    rows = []
    for f in floors:
        a = agree(f)
        dv = sum(r["volume_discarded_by_floor_26"][f] for r in recs)
        z = sum(1 for r in live if r["counts_by_floor_26"][f] == 0)
        med = sorted(r["counts_by_floor_26"][f] for r in live)[len(live) // 2]
        cls = ' class="hi"' if f == F else ""
        rows.append(
            f"<tr{cls}><td class=mono>{f}</td>"
            f"<td class=mono>{a}/{len(live)} &nbsp;{100*a/len(live):.1f}%</td>"
            f"<td class=mono>{dv:,}</td><td class=mono>{100*dv/total_vol:.3f}%</td>"
            f"<td class=mono>{z}</td><td class=mono>{med}</td></tr>")

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LUMIERE — Lesion Splitting Decision</title><style>{CSS}</style></head><body><div class="wrap">

<h1>Splitting the enhancing mask into lesions: 26-connectivity, floor 20&nbsp;mm³</h1>
<div class="sub">The enhancing compartment answers "is this voxel tumour?" but not "how many lesions
are there." Imposing lesion identity needs two choices — a connectivity rule and a volume floor.
Both are settled here against all {len(recs)} practice-arm timepoints, not against intuition.</div>

<div class="decision">
<div class="n">Decision</div>
<p style="margin-top:2px"><b>26-connectivity</b> (face, edge or corner contact joins two voxels),
followed by a <b>{F}&nbsp;mm³ volume floor</b> applied to the lesion inventory only.</p>
</div>

<div class="kpis">
<div class="kpi"><div class="n">{len(live)}</div><div class="l">timepoints with enhancing
disease, of {len(recs)} in the practice arm</div></div>
<div class="kpi"><div class="n">{100*agree(F)/len(live):.0f}%</div><div class="l">of those, where
6- and 26-connectivity agree on lesion count at this floor</div></div>
<div class="kpi"><div class="n">{100*disc/total_vol:.3f}%</div><div class="l">of all enhancing
volume discarded by the floor ({disc:,} of {total_vol:,} mm³)</div></div>
<div class="kpi"><div class="n">{len(nd)}</div><div class="l">timepoints where the largest lesion
holds under 90% of enhancing volume</div></div>
</div>

<h2>Why 26 and not 6</h2>
<p>The a-priori argument is that enhancing tumour presents as thin rims, one or two voxels thick
after resampling, and 6-connectivity shatters a continuous rim into fragments. The sweep found a
sharper argument that does not depend on that intuition.</p>
<p>At a {F}&nbsp;mm³ floor there are <b>{len(fewer6)} timepoints where 6-connectivity ends up with
FEWER surviving lesions than 26</b>, not more. It cannot merge anything — connectivity only ever
splits — so the only way this happens is that 6-connectivity fragments a real lesion into pieces
that individually fall below the floor and are then discarded.</p>
<div class="warn"><b>The failure this avoids.</b> 6-connectivity combined with a floor does not
merely over-count debris; it can delete genuine lesions outright. That is the over-merge failure
mode — a lesion that silently stops existing — arriving through a side door. 26-connectivity
cannot produce it, because any component it forms is a superset of the 6-connected one.</div>

<h2>Choosing the floor</h2>
<p>The floor is a hygiene filter on the lesion inventory: at 1&nbsp;mm atlas resolution, resampled
from acquisitions with ~5&nbsp;mm slices, isolated few-voxel components are partial-volume and
interpolation debris rather than tumour. Its cost is not volume — even at 100&nbsp;mm³ the floor
discards well under 1% of enhancing volume. Its cost is <em>erasure</em>.</p>

<div class="fig">{chart_tradeoff(d)}
<div class="legend"><span><i style="background:var(--ink-2)"></i>connectivity agreement (%)</span>
<span><i style="background:var(--accent)"></i>timepoints emptied by the floor (% of 83, count labelled)</span></div>
<div class="cap">Raising the floor buys agreement between the two connectivities and pays for it by
emptying more timepoints. The shaded band is the chosen floor.</div></div>

<table><thead><tr><th>Floor mm³</th><th>26 vs 6 agree</th><th>Volume discarded</th>
<th>% of arm</th><th>Timepoints emptied</th><th>Median lesions</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table>

<p>{F}&nbsp;mm³ sits where the two curves cross usefully. It is anchored, not tuned:</p>
<ul>
<li><b>Far below measurability.</b> A lesion satisfying RANO's 10&nbsp;×&nbsp;10&nbsp;mm bar on at
least two slices occupies on the order of 100–200&nbsp;mm³. A {F}&nbsp;mm³ floor is 5–10× below
anything that could ever be a target lesion, so it cannot remove one.</li>
<li><b>At the scale of the acquisition.</b> A single native voxel on a 0.9&nbsp;×&nbsp;0.9&nbsp;×&nbsp;5&nbsp;mm
grid is about 4&nbsp;mm³; a 3&nbsp;×&nbsp;3 in-plane footprint on one native slice is roughly
36&nbsp;mm³. Components below {F}&nbsp;mm³ are smaller than anything independently observable in
the data the segmentation was computed from.</li>
<li><b>Still sensitive to new lesions.</b> The floor must stay well under measurability precisely
because a new, non-measurable enhancing focus is still evidence of progression. {F}&nbsp;mm³ keeps
those.</li>
</ul>

<div class="warn"><b>The constraint this creates.</b> At {F}&nbsp;mm³, {len(zeroed)} timepoints lose
their entire lesion inventory (enhancing volumes {", ".join(str(r['enhancing_mm3']) for r in sorted(zeroed, key=lambda r: r['enhancing_mm3']))}&nbsp;mm³).
Those timepoints still <em>have</em> enhancing disease. An empty inventory therefore means
<b>"no measurable target lesion"</b> and must never be read as "no disease" or allowed to produce
a CR. Volumetry never applies the floor — it sums every voxel — and the response call reads
volume, not inventory size. This is the same distinction <code>measurement.py</code> already
enforces between <code>None</code> and <code>False</code>.</div>

<h2>What the floor actually removes</h2>
<div class="fig">{chart_counts(d)}
<div class="legend"><span><i style="background:var(--ink-4)"></i>no floor</span>
<span><i style="background:{ACCENT}"></i>floor {F} mm³</span></div>
<div class="cap">Lesion counts per timepoint, before and after. Unfloored, the median timepoint
carries {sorted(r['counts_by_floor_26']['0'] for r in live)[len(live)//2]} components and the worst
carries {max(r['counts_by_floor_26']['0'] for r in live)}. After the floor the median is
{sorted(r['counts_by_floor_26'][F] for r in live)[len(live)//2]}. Genuine multifocality survives:
{sum(1 for r in live if r['counts_by_floor_26'][F] >= 3)} timepoints still carry three or more
lesions.</div></div>

<h2>Two constraints this puts on the measurement step</h2>
<h3>One target lesion is not enough</h3>
<div class="fig">{chart_dominance(d)}
<div class="cap">Share of enhancing volume held by the largest lesion, one point per timepoint,
sorted. Median {sorted(r['largest_volume_fraction'] for r in live)[len(live)//2]:.1%}, but the
minimum is {min(r['largest_volume_fraction'] for r in live):.1%}.</div></div>
<p>In <b>{len(nd)} of {len(live)}</b> timepoints the largest lesion holds under 90% of the
enhancing volume. Selecting a single biggest target and measuring only that would discard the
majority of the disease in the worst cases. RANO's sum of products across several target lesions
is not a formality — this arm needs it.</p>

<h3>Cross-sections are routinely in several pieces</h3>
<p>Across every axial slice occupied by a largest lesion ({tot_sl:,} slices in total),
<b>{100*tot_fr/tot_sl:.1f}%</b> show that single lesion as more than one disjoint 2-D piece — the
lesion is connected through adjacent slices, not within the slice. Only
{sum(1 for r in live if r['largest_n_slices'] and r['largest_slices_fragmented'] == r['largest_n_slices'])}
timepoint is fragmented on every one of its slices, and
{sum(1 for r in live if r['largest_n_slices'] and r['largest_slices_fragmented'] == 0)} are never
fragmented. Any per-slice diameter measurement must handle a multi-piece cross-section as the
normal case, and must never draw a caliper across the gap between two pieces.</p>

<div class="ok"><b>Reproduce.</b>
<code>.venv/bin/python scripts/sweep_connectivity.py</code> writes
<code>output/connectivity/connectivity_sweep.json</code>;
<code>scripts/summarize_connectivity_sweep.py</code> prints the tables and
<code>scripts/render_lesion_split_decision.py</code> regenerates this page. The sweep's component
labelling was cross-checked against an independent breadth-first implementation on
Patient-067 week-109 (9 vs 9 components under 26-connectivity, 16 vs 16 under 6).</div>

<div class="foot">LUMIERE · lesion-splitting decision · practice arm only ({len(recs)} timepoints,
7 patients) · atlas MNI 1&nbsp;mm isotropic, 1 voxel = 1&nbsp;mm³ · enhancing compartment
(DeepBraTumIA label 1) · generated by <code>scripts/render_lesion_split_decision.py</code></div>
</div></body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", default=str(SWEEP))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    d = json.loads(Path(a.json).read_text())
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build(d))
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
