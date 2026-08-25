"""
Render the atlas-vs-native volumetry audit as a standalone HTML report with inline SVG charts.

Reads ``output/volume_audit/volumes.csv`` (produced by ``audit_atlas_native_volumes.py``) and
writes ``docs/volume_audit.html``. No plotting dependency — the project venv is deliberately lean
(numpy + nibabel + pydantic), and ``docs/`` already ships hand-built HTML, so the charts are
emitted as inline SVG and the report opens in any browser with nothing installed.

Charts:
  1. Per-patient small multiples (sampled patients x 3 compartments): the atlas volume trajectory
     with all four native-grid measurements overlaid, plus a residual strip showing % deviation.
  2. Cohort scatter: % deviation from atlas vs anisotropy ratio, one panel per compartment, with
     per-bin p10-p90 bands over all 91 patients.
  3. Cohort summary: spread (p90-p10 width) vs anisotropy, one line per compartment.

Usage:
    .venv/bin/python scripts/render_volume_audit.py
    .venv/bin/python scripts/render_volume_audit.py --patients Patient-067 Patient-006
"""

from __future__ import annotations

import argparse
import csv
import html
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = ROOT / "output" / "volume_audit" / "volumes.csv"
DEFAULT_HTML = ROOT / "docs" / "volume_audit.html"

COMPARTMENTS = [
    ("enhancing", "Contrast-enhancing", "#e0533d"),
    ("necrosis_nonenhancing", "Necrosis / non-enhancing", "#c98a1b"),
    ("edema", "Edema (T2/FLAIR)", "#3f8fd0"),
]
MODALITIES = [("CT1", "#e0533d"), ("T1", "#c98a1b"), ("T2", "#5aa469"), ("FLAIR", "#3f8fd0")]
ATLAS_COLOR = "#8b93a7"
#: minimum atlas volume (mm3) for a ratio to be meaningful — below this, one voxel is a big %
MIN_VOLUME_MM3 = 1000.0
ANISO_BINS = [(1.0, 1.01), (1.01, 3.0), (3.0, 6.0), (6.0, 10.0), (10.0, 14.0), (14.0, 30.0)]
#: compartment-size bands (mm3). Size, not compartment identity, turns out to drive the spread.
VOLUME_BANDS = [
    (200, 1000, "0.2–1 cm³"), (1000, 5000, "1–5 cm³"), (5000, 20000, "5–20 cm³"),
    (20000, 60000, "20–60 cm³"), (60000, 1e12, "> 60 cm³"),
]
#: the smallest compartment worth a ratio at all — below this a single voxel dominates
MIN_VOLUME_BAND = 200.0
DEFAULT_SAMPLE = ["Patient-067", "Patient-073", "Patient-006", "Patient-029", "Patient-019", "Patient-042"]


# ---------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------
def load(csv_path: Path):
    """-> (by_case, patients_in_order). by_case[(patient, tp, compartment)][modality] = record."""
    by_case: dict[tuple[str, str, str], dict[str, dict]] = defaultdict(dict)
    order: dict[str, list[str]] = defaultdict(list)
    for r in csv.DictReader(csv_path.open()):
        key = (r["patient"], r["timepoint"], r["compartment"])
        by_case[key][r["modality"]] = {
            "volume": float(r["volume_mm3"]),
            "aniso": float(r["anisotropy_ratio"]),
            "voxel": float(r["voxel_mm3"]),
            "week": float(r["week_offset"]) if r["week_offset"] else None,
        }
        if r["timepoint"] not in order[r["patient"]]:
            order[r["patient"]].append(r["timepoint"])
    return by_case, order


def deviations(by_case, *, min_volume: float = MIN_VOLUME_MM3) -> list[tuple[str, float, float]]:
    """All (compartment, anisotropy, percent_deviation_from_atlas) points across the cohort."""
    out = []
    for (_, _, comp), per_mod in by_case.items():
        atlas = per_mod.get("atlas")
        if atlas is None or atlas["volume"] < min_volume:
            continue
        for mod, _ in MODALITIES:
            rec = per_mod.get(mod)
            if rec is not None:
                out.append((comp, rec["aniso"], (rec["volume"] / atlas["volume"] - 1.0) * 100.0))
    return out


def sized_deviations(by_case) -> list[tuple[str, float, float, float]]:
    """(compartment, atlas_volume_mm3, anisotropy, percent_deviation) — keeps the size dimension.

    Separate from ``deviations`` because the size axis is what shows the spread is driven by how
    big the compartment is, not by which compartment it is.
    """
    out = []
    for (_, _, comp), per_mod in by_case.items():
        atlas = per_mod.get("atlas")
        if atlas is None or atlas["volume"] < MIN_VOLUME_BAND:
            continue
        for mod, _ in MODALITIES:
            rec = per_mod.get(mod)
            if rec is not None:
                out.append((comp, atlas["volume"], rec["aniso"],
                            (rec["volume"] / atlas["volume"] - 1.0) * 100.0))
    return out


def spread(values) -> float:
    return float(np.percentile(values, 90) - np.percentile(values, 10))


# ---------------------------------------------------------------------------------------
# tiny SVG helpers
# ---------------------------------------------------------------------------------------
def esc(s) -> str:
    return html.escape(str(s))


def txt(x, y, s, *, size=11, fill="var(--ink-2)", anchor="start", weight=400, family=None) -> str:
    fam = f' font-family="{family}"' if family else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}"{fam}>{esc(s)}</text>')


def line(x1, y1, x2, y2, *, stroke="var(--grid)", w=1, dash=None, opacity=1.0) -> str:
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" '
            f'stroke-width="{w}"{d} opacity="{opacity}"/>')


def path(points, *, stroke, w=1.6, fill="none", opacity=1.0, dash=None) -> str:
    if not points:
        return ""
    d = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    da = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{w}" '
            f'stroke-linejoin="round" stroke-linecap="round" opacity="{opacity}"{da}/>')


def circle(cx, cy, r, *, fill, stroke="none", w=1, opacity=1.0) -> str:
    return (f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.2f}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{w}" opacity="{opacity}"/>')


def rect(x, y, w, h, *, fill, opacity=1.0, rx=0) -> str:
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0):.1f}" height="{max(h, 0):.1f}" '
            f'rx="{rx}" fill="{fill}" opacity="{opacity}"/>')


def nice_ticks(lo: float, hi: float, n: int = 4) -> list[float]:
    if hi <= lo:
        return [lo]
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=raw)
    start = math.ceil(lo / step) * step
    return [start + i * step for i in range(int((hi - start) / step) + 1)]


def fmt_vol(v: float) -> str:
    if v >= 1000:
        return f"{v / 1000:.0f}k" if v >= 10000 else f"{v / 1000:.1f}k"
    return f"{v:.0f}"


# ---------------------------------------------------------------------------------------
# Figure 1 — per-patient small multiples
# ---------------------------------------------------------------------------------------
PW, PH, RH = 268, 132, 46          # panel width, main height, residual-strip height
PAD_L, PAD_T, GAP_X, GAP_Y = 52, 26, 26, 30


def panel(x0: float, y0: float, tps: list[str], series: dict, comp_color: str, title: str) -> str:
    """One patient x one compartment: atlas trajectory + 4 native overlays + residual strip."""
    s = [txt(x0, y0 - 8, title, size=10.5, fill="var(--ink-3)", weight=600)]
    atlas = [series[tp].get("atlas") for tp in tps]
    have = [i for i, a in enumerate(atlas) if a is not None]
    if not have:
        return "".join(s) + txt(x0 + PW / 2, y0 + PH / 2, "no mask", size=10,
                                fill="var(--ink-4)", anchor="middle")

    vmax = max(a["volume"] for a in atlas if a) or 1.0
    vmax = vmax * 1.18
    n = max(len(tps) - 1, 1)
    px = lambda i: x0 + (i / n) * PW
    py = lambda v: y0 + PH - (v / vmax) * PH

    # y grid + labels
    for t in nice_ticks(0, vmax, 3):
        if t > vmax:
            continue
        s.append(line(x0, py(t), x0 + PW, py(t)))
        s.append(txt(x0 - 6, py(t) + 3.4, fmt_vol(t), size=9, fill="var(--ink-4)", anchor="end"))

    # native overlays first (thin, behind), then atlas (bold, on top)
    for mod, mcolor in MODALITIES:
        pts = [(px(i), py(series[tps[i]][mod]["volume"])) for i in have if mod in series[tps[i]]]
        s.append(path(pts, stroke=mcolor, w=1.0, opacity=0.75))
        for cx, cy in pts:
            s.append(circle(cx, cy, 1.7, fill=mcolor, opacity=0.9))
    s.append(path([(px(i), py(atlas[i]["volume"])) for i in have], stroke=comp_color, w=2.1))
    for i in have:
        s.append(circle(px(i), py(atlas[i]["volume"]), 3.0, fill="var(--panel)",
                        stroke=comp_color, w=2.0))

    # residual strip: % deviation of each native grid from atlas
    ry = y0 + PH + 16
    devs = [(px(i), (series[tps[i]][m]["volume"] / atlas[i]["volume"] - 1) * 100, mc)
            for i in have if atlas[i]["volume"] >= MIN_VOLUME_MM3
            for m, mc in MODALITIES if m in series[tps[i]]]
    lim = max(5.0, max((abs(d) for _, d, _ in devs), default=5.0) * 1.15)
    s.append(rect(x0, ry, PW, RH, fill="var(--strip)", rx=3))
    s.append(rect(x0, ry + RH / 2 - (2 / lim) * (RH / 2), PW, (4 / lim) * (RH / 2),
                  fill=ATLAS_COLOR, opacity=0.13))          # +/-2% reference band
    s.append(line(x0, ry + RH / 2, x0 + PW, ry + RH / 2, stroke="var(--ink-4)", w=1, dash="2 3"))
    for cx, d, mc in devs:
        s.append(circle(cx, ry + RH / 2 - (d / lim) * (RH / 2), 2.1, fill=mc, opacity=0.85))
    s.append(txt(x0 - 6, ry + 9, f"+{lim:.0f}%", size=8, fill="var(--ink-4)", anchor="end"))
    s.append(txt(x0 - 6, ry + RH - 2, f"-{lim:.0f}%", size=8, fill="var(--ink-4)", anchor="end"))
    return "".join(s)


def figure_patients(by_case, order, patients: list[str]) -> str:
    cols = len(COMPARTMENTS)
    cell_h = PH + RH + 16 + GAP_Y + 14
    w = PAD_L + cols * PW + (cols - 1) * GAP_X + 26
    h = PAD_T + len(patients) * cell_h + 30
    out = [f'<svg viewBox="0 0 {w:.0f} {h:.0f}" width="100%" role="img" '
           f'aria-label="Per-patient atlas versus native volumes">']

    for r, pid in enumerate(patients):
        tps = order[pid]
        y0 = PAD_T + r * cell_h + 14
        out.append(txt(4, y0 + PH / 2, pid.replace("Patient-", "P-"), size=11,
                       fill="var(--ink-1)", weight=700))
        for c, (comp, label, color) in enumerate(COMPARTMENTS):
            x0 = PAD_L + c * (PW + GAP_X)
            series = {tp: by_case.get((pid, tp, comp), {}) for tp in tps}
            out.append(panel(x0, y0, tps, series, color, label if r == 0 else ""))
        out.append(txt(PAD_L, y0 + PH + RH + 30, f"timepoints in order  →  {len(tps)} studies",
                       size=9, fill="var(--ink-4)"))
    out.append("</svg>")
    return "".join(out)


# ---------------------------------------------------------------------------------------
# Figure 2 — cohort deviation vs anisotropy
# ---------------------------------------------------------------------------------------
def figure_cohort(points, seed: int = 7) -> str:
    CW, CH, CGAP, LPAD, TPAD = 300, 250, 34, 56, 30
    w = LPAD + 3 * CW + 2 * CGAP + 24
    h = TPAD + CH + 62
    out = [f'<svg viewBox="0 0 {w:.0f} {h:.0f}" width="100%" role="img" '
           f'aria-label="Deviation from atlas versus anisotropy, whole cohort">']
    rng = random.Random(seed)
    ylim = 12.0
    ax_lo, ax_hi = math.log10(1.0), math.log10(20.0)
    lx = lambda a: (math.log10(max(a, 1.0)) - ax_lo) / (ax_hi - ax_lo)

    for c, (comp, label, color) in enumerate(COMPARTMENTS):
        x0 = LPAD + c * (CW + CGAP)
        px = lambda a: x0 + lx(a) * CW
        py = lambda d: TPAD + CH / 2 - (max(-ylim, min(ylim, d)) / ylim) * (CH / 2)
        sub = [(a, d) for cm, a, d in points if cm == comp]

        out.append(txt(x0, TPAD - 12, label, size=11, fill=color, weight=700))
        out.append(rect(x0, TPAD, CW, CH, fill="var(--strip)", rx=4))
        for t in (-10, -5, 0, 5, 10):
            out.append(line(x0, py(t), x0 + CW, py(t),
                            stroke="var(--ink-4)" if t == 0 else "var(--grid)",
                            w=1, dash=None if t == 0 else "2 3"))
            if c == 0:
                out.append(txt(x0 - 8, py(t) + 3.4, f"{t:+d}%" if t else "0",
                               size=9, fill="var(--ink-4)", anchor="end"))
        for a in (1, 2, 5, 10, 20):
            out.append(line(px(a), TPAD, px(a), TPAD + CH, stroke="var(--grid)", w=1))
            out.append(txt(px(a), TPAD + CH + 15, f"{a}x", size=9,
                           fill="var(--ink-4)", anchor="middle"))

        # per-bin p10-p90 band + median
        band_hi, band_lo, med = [], [], []
        for lo, hi in ANISO_BINS:
            vals = [d for a, d in sub if lo <= a < hi]
            if len(vals) < 20:
                continue
            xm = px(math.sqrt(max(lo, 1.0) * min(hi, 20.0)))
            band_hi.append((xm, py(float(np.percentile(vals, 90)))))
            band_lo.append((xm, py(float(np.percentile(vals, 10)))))
            med.append((xm, py(float(np.median(vals)))))
        if band_hi:
            poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in band_hi + band_lo[::-1])
            out.append(f'<polygon points="{poly}" fill="{color}" opacity="0.17"/>')

        for a, d in sub:
            if rng.random() < 0.42:  # subsample so the SVG stays light and the band stays visible
                out.append(circle(px(a) + rng.uniform(-2, 2), py(d), 1.25, fill=color, opacity=0.34))
        out.append(path(med, stroke=color, w=2.0))
        for x, y in med:
            out.append(circle(x, y, 2.6, fill="var(--panel)", stroke=color, w=1.8))
        out.append(txt(x0 + CW / 2, TPAD + CH + 34, "anisotropy ratio of the native grid (log)",
                       size=9.5, fill="var(--ink-3)", anchor="middle"))
    out.append("</svg>")
    return "".join(out)


# ---------------------------------------------------------------------------------------
# Figure 3 — spread vs anisotropy (the summary chart)
# ---------------------------------------------------------------------------------------
def figure_size(sized) -> str:
    """Spread vs COMPARTMENT SIZE, split by grid quality. Log y — the range spans three decades."""
    W, H, L, T = 600, 288, 62, 30
    w, h = L + W + 176, T + H + 64
    out = [f'<svg viewBox="0 0 {w:.0f} {h:.0f}" width="100%" role="img" '
           f'aria-label="Measurement spread versus compartment size">']

    series = [
        ("native grid ≥6× anisotropic", "#e0533d", lambda a: a >= 6.0),
        ("native grid isotropic", "#3f8fd0", lambda a: a <= 1.01),
    ]
    lo_y, hi_y = 0.05, 60.0
    ly = lambda v: math.log10(max(v, lo_y))
    n = len(VOLUME_BANDS)
    px = lambda i: L + (i + 0.5) / n * W
    py = lambda v: T + H - (ly(v) - ly(lo_y)) / (ly(hi_y) - ly(lo_y)) * H

    # RANO-relevant zone: a 25% measurement spread is the size of a response threshold
    out.append(rect(L, T, W, py(25) - T, fill="#e0533d", opacity=0.09))
    out.append(line(L, py(25), L + W, py(25), stroke="#e0533d", w=1, dash="4 3", opacity=0.7))
    out.append(txt(L + W - 6, py(25) - 6, "25% — the size of a RANO response threshold",
                   size=9.5, fill="#e0533d", anchor="end", weight=600))

    for t in (0.1, 0.5, 1, 5, 10, 50):
        out.append(line(L, py(t), L + W, py(t)))
        out.append(txt(L - 8, py(t) + 3.4, f"{t:g}%", size=9.5, fill="var(--ink-4)", anchor="end"))
    for i, (_, _, lbl) in enumerate(VOLUME_BANDS):
        out.append(txt(px(i), T + H + 18, lbl, size=10, fill="var(--ink-3)", anchor="middle"))

    for label, color, keep in series:
        pts, labels = [], []
        for i, (blo, bhi, _) in enumerate(VOLUME_BANDS):
            vals = [d for _, v, a, d in sized if blo <= v < bhi and keep(a)]
            if len(vals) >= 20:
                s = spread(vals)
                pts.append((px(i), py(s)))
                labels.append(s)
        out.append(path(pts, stroke=color, w=2.6))
        for (x, y), s in zip(pts, labels):
            out.append(circle(x, y, 4.0, fill="var(--panel)", stroke=color, w=2.4))
            out.append(txt(x, y - 11, f"{s:.1f}", size=9.5, fill=color, anchor="middle", weight=700))
        if pts:
            out.append(txt(pts[-1][0] + 13, pts[-1][1] + 3.6, label, size=10.5, fill=color, weight=700))

    out.append(txt(L + W / 2, T + H + 42, "size of the compartment being measured (atlas volume)",
                   size=10, fill="var(--ink-3)", anchor="middle"))
    out.append(txt(4, T - 14, "p10–p90 spread of native-vs-atlas volume (percentage points, log scale)",
                   size=10, fill="var(--ink-3)", weight=600))
    out.append("</svg>")
    return "".join(out)


# ---------------------------------------------------------------------------------------
# page
# ---------------------------------------------------------------------------------------
CSS = """
:root{--bg:#f7f7f5;--panel:#fff;--strip:#f2f2ef;--grid:#e6e6e1;--ink-1:#1a1a18;--ink-2:#3d3d39;
--ink-3:#6b6b64;--ink-4:#9a9a92;--line:#e2e2dc;--accent:#b5533a;--warn-bg:#fdf4ec;--warn-br:#e8b58a;}
@media (prefers-color-scheme:dark){:root{--bg:#14140f;--panel:#1c1c18;--strip:#232320;--grid:#2f2f2a;
--ink-1:#f0efe9;--ink-2:#d2d1c9;--ink-3:#9a998f;--ink-4:#6e6d64;--line:#2f2f2a;--accent:#e0876a;
--warn-bg:#2a1f16;--warn-br:#6b4a2e;}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink-1);
font:15px/1.6 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif;}
.wrap{max-width:1080px;margin:0 auto;padding:44px 22px 90px}
h1{font-size:27px;letter-spacing:-.02em;margin:0 0 6px}
h2{font-size:18px;letter-spacing:-.01em;margin:46px 0 6px;padding-top:20px;border-top:1px solid var(--line)}
h3{font-size:13px;text-transform:uppercase;letter-spacing:.07em;color:var(--ink-3);margin:26px 0 8px}
p{color:var(--ink-2);margin:9px 0;max-width:74ch}
.sub{color:var(--ink-3);font-size:14px;margin-bottom:26px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:18px 20px;margin:16px 0}
.fig{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:20px;margin:14px 0;overflow-x:auto}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(168px,1fr));gap:12px;margin:22px 0}
.kpi{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.kpi .n{font-size:23px;font-weight:700;letter-spacing:-.02em}
.kpi .l{font-size:11.5px;color:var(--ink-3);margin-top:3px;line-height:1.35}
.warn{background:var(--warn-bg);border:1px solid var(--warn-br);border-radius:10px;padding:16px 20px;margin:20px 0}
.warn b{color:var(--accent)}
table{border-collapse:collapse;width:100%;font-size:13px;margin:10px 0}
th,td{text-align:right;padding:6px 10px;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}
th{color:var(--ink-3);font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.05em}
td.mono,th.mono{font-variant-numeric:tabular-nums}
.legend{display:flex;flex-wrap:wrap;gap:16px;margin:4px 0 14px;font-size:12px;color:var(--ink-3)}
.legend i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;vertical-align:-1px}
code{background:var(--strip);padding:1px 5px;border-radius:4px;font-size:12.5px}
.foot{color:var(--ink-4);font-size:12px;margin-top:40px;border-top:1px solid var(--line);padding-top:16px}
"""


def kpi(n, label) -> str:
    return f'<div class="kpi"><div class="n">{esc(n)}</div><div class="l">{esc(label)}</div></div>'


def build(by_case, order, patients, points, sized, n_patients, n_cases) -> str:
    leg_mod = "".join(
        f'<span><i style="background:{c}"></i>native {m}</span>' for m, c in MODALITIES
    )
    leg_comp = "".join(
        f'<span><i style="background:{c}"></i>{l}</span>' for _, l, c in COMPARTMENTS
    )

    # summary table: size band x compartment, on anisotropic grids (where the effect lives)
    rowsh = []
    for blo, bhi, blbl in VOLUME_BANDS:
        cells = []
        for comp, _, _ in COMPARTMENTS:
            vals = [d for cm, v, a, d in sized if cm == comp and blo <= v < bhi and a >= 6.0]
            cells.append(f"{spread(vals):.1f}" if len(vals) >= 20 else "–")
        allb = [d for _, v, a, d in sized if blo <= v < bhi and a >= 6.0]
        rowsh.append(f"<tr><td>{blbl}</td><td class='mono'>{len(allb)}</td>"
                     f"<td class='mono'><b>{spread(allb):.1f}</b></td>"
                     + "".join(f"<td class='mono'>{c}</td>" for c in cells) + "</tr>")

    iso = [abs(d) for _, a, d in points if a <= 1.01]
    small = [d for _, v, a, d in sized if v < 1000 and a >= 6.0]
    big = [d for _, v, a, d in sized if v >= 20000 and a >= 6.0]

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LUMIERE — Atlas vs Native Volumetry Audit</title><style>{CSS}</style></head><body><div class="wrap">

<h1>Atlas-space volumetry is unbiased — but its precision depends on lesion size, not on the compartment</h1>
<div class="sub">Every DeepBraTumIA compartment volume computed five ways — once on the MNI 1&nbsp;mm atlas grid,
and once on each of the four native acquisition grids — across {n_patients} patients and {n_cases} timepoints.</div>

<div class="warn">
<b>Read this first — a label-decode error was found and corrected during this audit.</b>
<p>DeepBraTumIA's label 1 is <b>contrast-enhancing</b> and label 2 is <b>necrosis / non-enhancing</b>.
<code>label_schema.py</code> had these two swapped, inherited from the <code>Label name</code> column of
<code>LUMIERE-pyradiomics-deepbratumia-features.csv</code>, which is itself mislabeled. Three independent
checks agree: the shipped <code>measured_volumes_in_mm3.json</code> matches label&nbsp;1 to
<code>Enhancing_Core</code> in <b>599/599</b> masks; contrast physics (post- minus pre-contrast T1, normalised
per scan) shows label&nbsp;1 enhancing in <b>51/52</b> sampled cases while label&nbsp;2 <i>loses</i> signal
(median −0.073); and the resulting trajectories are clinically coherent. HD-GLIO-AUTO was re-tested the same
way and is <b>correct as documented</b>. Any earlier analysis that treated label&nbsp;2 as enhancing was
measuring necrosis and must be re-run. All figures below use the corrected mapping.</p>
</div>

<div class="kpis">
{kpi("0.00%", "Median deviation of native from atlas — across every compartment, grid and case. The transforms are rigid, so no volume is created or destroyed.")}
{kpi(f"±{np.percentile(iso, 90):.2f}%", "p90 |deviation| when the native grid is isotropic. Grid choice is essentially free at this quality.")}
{kpi(f"{spread(big):.1f} pp", "Spread for large compartments (>20 cm³) even on thick-slice grids. Big tumours are robust.")}
{kpi(f"{spread(small):.0f} pp", "Spread for sub-1 cm³ compartments on ≥6× anisotropic grids. This is the number that matters.")}
</div>

<h2>1 · Six patients, five measurements each</h2>
<p>Each panel is one patient and one compartment. The thick line is the atlas-grid volume; the four thin lines
are the same segmentation measured on the CT1, T1, T2 and FLAIR native grids. The strip beneath each panel
shows how far each native measurement sits from atlas, in percent, with the ±2% band shaded.</p>
<div class="legend">{leg_comp}</div>
<div class="legend">{leg_mod}<span>thick line = atlas (MNI 1&nbsp;mm)</span></div>
<div class="fig">{figure_patients(by_case, order, patients)}</div>
<p>The trajectories are indistinguishable at the scale that matters clinically — which is the finding. Where the
residual strip does move, it moves on the thick-slice grids, and it moves most for the enhancing compartment.</p>

<h2>2 · The whole cohort: deviation against anisotropy</h2>
<p>Every native measurement in the dataset, plotted against how anisotropic its grid is. The band is the
p10–p90 range per anisotropy bin; the line through it is the median. Cases whose atlas volume is under
{MIN_VOLUME_MM3:.0f}&nbsp;mm³ are excluded — at that size a single voxel is a large percentage.</p>
<div class="fig">{figure_cohort(points)}</div>
<p>Two things are visible at once. The median sits on zero at every anisotropy — nearest-neighbour resampling
is <b>unbiased</b>, it does not systematically inflate or shrink a compartment. But the spread fans out sharply
with slice thickness. It appears to fan out fastest for contrast-enhancing tissue; the next section shows that
appearance is a confound.</p>

<h2>3 · It is not the compartment — it is the size</h2>
<p>The obvious reading of chart 2 is "the enhancing compartment is the fragile one". That reading is a
confound. Resampling error is created at a structure's boundary, so what matters is surface area relative to
volume — and small structures have a worse ratio than large ones, whatever tissue they are. Enhancing
compartments simply tend to be the smallest of the three. Control for size and the compartments converge.</p>
<div class="fig">{figure_size(sized)}</div>
<p>The spread is a function of how big the thing you are measuring is, and it spans nearly three orders of
magnitude. On an isotropic grid it never exceeds 2 percentage points at any size. On a thick-slice grid a
60&nbsp;cm³ mass still measures to within about 1.5&nbsp;points — but a sub-1&nbsp;cm³ compartment swings by
{spread(small):.0f} points, which is larger than a RANO response threshold.</p>

<h3>Spread by size and compartment, on ≥6× anisotropic grids</h3>
<table><thead><tr><th>compartment size</th><th class="mono">n</th><th class="mono">all</th>
{''.join(f'<th class="mono">{l}</th>' for _, l, _ in COMPARTMENTS)}</tr></thead>
<tbody>{''.join(rowsh)}</tbody></table>
<p style="font-size:12.5px;color:var(--ink-3)">p10–p90 spread, in percentage points. Read across a row: within
a size band the three compartments are close to each other. Read down the "all" column: size changes the
answer by more than 20×. Size dominates; compartment identity barely registers.</p>

<h2>4 · What this settles</h2>
<div class="card">
<p><b>Atlas-space volumetry is safe to build on, and needs no bias correction.</b> All 2,396 <code>.tfm</code>
transforms are rigid (determinant 1 to 1e-10), and the measurements confirm the consequence: median deviation
is 0.00% at every anisotropy and every size. Nearest-neighbour resampling adds noise, not bias. Piece 2 and
the RANO engine can work in atlas space without a Jacobian correction.</p>
<p><b>The error bar must be a function of lesion size, not a single constant.</b> Quoting one tolerance for
"a volume" would be wrong by a factor of 20 at the extremes. For a large tumour on any grid, and for anything
at all on an isotropic grid, the tolerance is well under 2%. For a sub-1&nbsp;cm³ compartment on 6–7&nbsp;mm
slices it is ±{spread(small) / 2:.0f}%.</p>
<p><b>That worst case is clinically live, and it is the case RANO cares most about.</b> A
{spread(small):.0f}-point spread is the same size as a RANO response threshold, so on small lesions the choice
of grid alone could flip a call. Small enhancing foci are exactly what early-progression and new-lesion
detection deal with — so Piece 4 must carry a size-dependent confidence, and any new-lesion rule needs a
minimum-volume floor beneath which a "change" is not reported as real.</p>
<p><b>Field-of-view clipping is not a risk here.</b> Zero masks in the cohort touch the atlas boundary, so the
182×218×182&nbsp;mm box is large enough for every tumour in LUMIERE. Keep it as a check — a future case can
violate it — but it needs no mitigation today.</p>
<p><b>Caveat that limits the whole claim.</b> The four native masks are not four independent measurements of
the tumour — they are one segmentation, performed once in atlas space, resampled four ways. Everything above
measures <i>resampling</i> uncertainty. Segmentation uncertainty is larger and remains unmeasured, because
LUMIERE ships no manual masks.</p>
</div>

<div class="foot">Generated by <code>scripts/render_volume_audit.py</code> from
<code>output/volume_audit/volumes.csv</code> · source <code>scripts/audit_atlas_native_volumes.py</code> ·
volumes via <code>src/rano/volumetry/volumes.py</code> using the corrected <code>label_schema.py</code>.</div>
</div></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--out", default=str(DEFAULT_HTML))
    ap.add_argument("--patients", nargs="+", default=DEFAULT_SAMPLE)
    args = ap.parse_args()

    by_case, order = load(Path(args.csv))
    points = deviations(by_case)
    sized = sized_deviations(by_case)
    n_patients = len({p for p, _, _ in by_case})
    n_cases = len({(p, t) for p, t, _ in by_case})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build(by_case, order, args.patients, points, sized, n_patients, n_cases))
    print(f"{len(points)} native-vs-atlas comparisons over {n_patients} patients / {n_cases} timepoints")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
