"""
Draw every synthetic test shape with the lines the ruler actually measured on it.

WHY A PICTURE
-------------
The perpendicular measurement was wrong twice before these shapes existed, and both times the
numbers it produced on real lesions looked perfectly reasonable. Numbers cannot show you that a
line cut across empty space; a drawing can. Each panel shows the tissue, the two measured lines,
and -- where they differ -- the unconstrained caliper as a dashed line, so the constraint doing
its job is visible rather than asserted.

Writes docs/ruler_validation.html. Inline SVG, no plotting dependency, same palette as the other
reports in docs/.

Usage:
    .venv/bin/python scripts/render_ruler_validation.py
"""

from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rano.measurement import fill_enclosed_holes, measure_slice, pieces_2d  # noqa: E402
from rano.measurement.shapes import SHAPES  # noqa: E402

OUT = ROOT / "docs" / "ruler_validation.html"
PANEL = 240
ACCENT = "#b5533a"
LONG_C = "#534AB7"
PERP_C = "#1D9E75"


def runs(mask: np.ndarray, s: float) -> str:
    """The mask as run-length rectangles: one rect per horizontal run, exact and compact."""
    out = []
    for j in range(mask.shape[1]):
        col = mask[:, j]
        d = np.diff(np.concatenate(([0], col.view(np.int8), [0])))
        for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            out.append(f'<rect x="{a*s:.2f}" y="{j*s:.2f}" '
                       f'width="{(b-a)*s:.2f}" height="{s:.2f}"/>')
    return "".join(out)


def line(p, q, s, colour, width=2.2, dash=None) -> str:
    (x1, y1), (x2, y2) = p, q
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1*s:.2f}" y1="{y1*s:.2f}" x2="{x2*s:.2f}" y2="{y2*s:.2f}" '
            f'stroke="{colour}" stroke-width="{width}" stroke-linecap="round"{d}/>')


def panel(name: str, mask, want_long, want_perp, note, fill_holes=True) -> str:
    m, piece_idx, n_pieces = measure_slice(mask, fill_holes=fill_holes)
    n = max(mask.shape)
    s = PANEL / n

    # what filling added, for the measured piece only
    ps = pieces_2d(mask)
    added = np.zeros_like(mask)
    if ps:
        pts = ps[piece_idx]
        lo = pts.min(axis=0)
        occ = np.zeros(tuple(pts.max(axis=0) - lo + 1), bool)
        occ[pts[:, 0] - lo[0], pts[:, 1] - lo[1]] = True
        diff = (fill_enclosed_holes(occ) & ~occ) if fill_holes else np.zeros_like(occ)
        for a, b in np.argwhere(diff):
            added[a + lo[0], b + lo[1]] = True

    svg = [f'<svg viewBox="0 0 {PANEL} {PANEL}" width="100%" role="img">',
           f'<title>{html.escape(name)}</title>',
           f'<g fill="var(--tissue)">{runs(mask, s)}</g>']
    if added.any():
        svg.append(f'<g fill="var(--filled)">{runs(added, s)}</g>')
    if m.caliper_mm - m.long_mm > 0.5:
        svg.append(line(*m.caliper_endpoints, s, ACCENT, 1.6, "5 4"))
    svg.append(line(*m.long_endpoints, s, LONG_C))
    svg.append(line(*m.perp_endpoints, s, PERP_C))
    svg.append("</svg>")

    def row(label, got, want):
        if want is None:
            return f'<tr><td>{label}</td><td class=mono>{got:.2f}</td><td class=sub>—</td></tr>'
        err = got - want
        cls = "ok" if abs(err) <= (0.5 if name.startswith("square") else 1.5) else "bad"
        return (f'<tr><td>{label}</td><td class=mono>{got:.2f}</td>'
                f'<td class="mono {cls}">{want:.2f} &nbsp; {err:+.2f}</td></tr>')

    extra = ""
    if m.caliper_mm - m.long_mm > 0.5:
        extra = (f'<div class="flag">unconstrained caliper would read '
                 f'<b>{m.caliper_mm:.2f}</b> — {m.caliper_mm - m.long_mm:+.2f} mm, '
                 f'a line that leaves the tissue</div>')
    if n_pieces > 1:
        extra += f'<div class="flag">{n_pieces} separate pieces; measured the largest only</div>'
    if added.any():
        extra += f'<div class="flag">{int(added.sum())} mm² of enclosed hole filled first</div>'
    if not fill_holes:
        extra += '<div class="flag">hole filling switched off for this panel</div>'

    return f"""<div class="card">
<div class="hd">{html.escape(name)}</div>
<div class="note">{html.escape(note)}</div>
{''.join(svg)}
<table><thead><tr><th>line</th><th>measured</th><th>expected / error</th></tr></thead><tbody>
{row('longest', m.long_mm, want_long)}
{row('perpendicular', m.perp_mm, want_perp)}
<tr><td>product</td><td class=mono>{m.product_mm2:.0f} mm²</td><td class=sub>—</td></tr>
</tbody></table>{extra}</div>"""


CSS = """:root{--bg:#f7f7f5;--panel:#fff;--grid:#e6e6e1;--ink-1:#1a1a18;--ink-2:#3d3d39;
--ink-3:#6b6b64;--ink-4:#9a9a92;--line:#e2e2dc;--accent:#b5533a;--ok:#3B6D11;--bad:#A32D2D;
--tissue:#d9d6cc;--filled:#f0c9b8;}
@media (prefers-color-scheme:dark){:root{--bg:#14140f;--panel:#1c1c18;--grid:#2f2f2a;
--ink-1:#f0efe9;--ink-2:#d2d1c9;--ink-3:#9a998f;--ink-4:#6e6d64;--line:#2f2f2a;--accent:#e0876a;
--ok:#97C459;--bad:#F09595;--tissue:#4a4a43;--filled:#6b4a2e;}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink-1);
font:15px/1.6 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:44px 22px 90px}
h1{font-size:27px;letter-spacing:-.02em;margin:0 0 6px}
h2{font-size:18px;margin:44px 0 6px;padding-top:20px;border-top:1px solid var(--line)}
p{color:var(--ink-2);margin:9px 0;max-width:74ch}
.sub{color:var(--ink-3)}
.legend{display:flex;flex-wrap:wrap;gap:20px;margin:18px 0 4px;font-size:13px;color:var(--ink-2)}
.legend i{display:inline-block;width:20px;height:3px;border-radius:2px;margin-right:7px;
vertical-align:3px}
.legend .sw{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:7px;
vertical-align:-1px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:16px;margin-top:18px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.hd{font-weight:600;font-size:14px}
.note{font-size:12px;color:var(--ink-3);margin:2px 0 8px;min-height:2.6em}
svg{display:block;background:transparent}
table{border-collapse:collapse;width:100%;font-size:12.5px;margin-top:8px}
th,td{text-align:right;padding:3px 6px;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}
th{color:var(--ink-3);font-weight:600;font-size:10.5px;text-transform:uppercase;letter-spacing:.05em}
.mono{font-variant-numeric:tabular-nums}
.ok{color:var(--ok)}.bad{color:var(--bad);font-weight:600}
.flag{font-size:11.5px;color:var(--accent);margin-top:7px;line-height:1.5}
.foot{color:var(--ink-4);font-size:12px;margin-top:44px;border-top:1px solid var(--line);padding-top:16px}
code{background:var(--grid);padding:1px 5px;border-radius:4px;font-size:12.5px}"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    cards = "".join(panel(name, *spec) for name, spec in SHAPES.items())
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LUMIERE — Ruler Validation</title><style>{CSS}</style></head><body><div class="wrap">

<h1>What the ruler actually measures</h1>
<p class="sub">Every shape below has an answer known before it was measured. The lines drawn on
each one are the lines the code chose, not an illustration of what it ought to have chosen.</p>

<div class="legend">
<span><i style="background:{LONG_C}"></i>longest line across the tissue</span>
<span><i style="background:{PERP_C}"></i>longest line at right angles to it</span>
<span><i style="background:{ACCENT}"></i>unconstrained caliper, where it differs — leaves the tissue</span>
<span><span class="sw" style="background:var(--tissue)"></span>tissue</span>
<span><span class="sw" style="background:var(--filled)"></span>enclosed hole, filled first</span>
</div>

<div class="grid">{cards}</div>

<h2>Why curved shapes read about a millimetre high</h2>
<p>A voxel is a 1&nbsp;mm square, not a point, so a shape is measured as the union of its squares.
For anything aligned to the grid — the square — that is exact. For a circle it is not: the voxels
near the diagonal stick out past the smooth boundary the circle came from, and measuring the
squares honestly measures that overhang too. The result is a consistent <b>one voxel</b> of
overshoot on curved shapes and none on grid-aligned ones.</p>
<p>That is a property of the grid rather than slack in the ruler, and it is the same overshoot to
expect when comparing against a radiologist's calipers, who is measuring a smooth image rather
than a stack of squares. It does not upset the 10&nbsp;mm measurability gate, because the gate
needs <em>both</em> lines to clear 10&nbsp;mm and the perpendicular is the less inflated of the
two: a disc of true diameter 10&nbsp;mm passes, one of 9.5&nbsp;mm does not.</p>

<h2>What each drawing is for</h2>
<p>The <b>rotated ellipses</b> catch a ruler that only searches along the voxel axes — such a
ruler reports the bounding box, and its answer would swell and shrink as the shape turns. The
<b>L-shape</b> and <b>crescent</b> are the reason the constraint exists: on both, the widest
caliper the shape allows runs clean across empty space, and you can see the dashed line doing it.
The <b>annulus</b> checks that a ring is measured through its middle, and the shaded centre shows
the fill that made that possible. <b>Two blobs</b> checks the one rule with no exceptions — a
ruler may never span two separate pieces of tissue.</p>

<div class="foot">LUMIERE · ruler validation · shapes from
<code>rano.measurement.shapes</code>, measurements from <code>rano.measurement.ruler</code> ·
regenerate with <code>scripts/render_ruler_validation.py</code> · asserted in
<code>tests/test_measurement_ruler.py</code></div>
</div></body></html>"""
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(page)
    print(f"wrote {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
