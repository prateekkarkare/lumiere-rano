#!/usr/bin/env python3
"""
Render the per-patient case tables as a self-contained, filterable HTML page.

The terminal view (``run_rano_calls.py --cases``) is for reading one patient at a time. This is for
the other question: across 376 scored scans, where does the rule disagree with the reader, and does
the reader's own stated rationale explain it?

It does. The reader records a rationale on every one of their 238 PD calls, and sorting those calls
by whether the evidence cited was inside this ruleset's field of view splits them cleanly:

    cited only findings the rule can see    42/44   95% agreement
    cited a mix                             71/80   89%
    cited only findings it cannot see      70/114   61%

Same rule, same thresholds. The page leads with that because it is the difference between "the rule
is mediocre" and "the rule is accurate within its field of view and blind outside it", and only one
of those is true. Note the buckets in the detailed table below overlap by construction -- one
rationale can cite a target lesion AND a new lesion -- so those rows do not sum to 238.

Usage:
    .venv/bin/python scripts/render_case_tables.py
    .venv/bin/python scripts/render_case_tables.py --profile rano_classic_ported
"""

from __future__ import annotations

import argparse
import csv
import html
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CALLS = ROOT / "output" / "rano_calls" / "calls.csv"
DEFAULT_OUT = ROOT / "output" / "rano_calls" / "case_tables.html"

SCORABLE = ("CR", "PR", "SD", "PD")

#: What the reader cited, normalised into buckets. A rationale can fall in several.
#: The ordering matters only for display; membership is independent.
BUCKETS: list[tuple[str, str, bool]] = [
    # (label, regex, is_visible_to_a_volume_rule)
    ("measured target lesion", r"(?<!non-)(?<!non )target l", True),
    ("non-target lesion", r"non-?\s?target", False),
    ("new lesion", r"\bnew\b", False),
    ("T2/FLAIR progression", r"t2 progression|t2-progr", False),
    ("non-measurable lesion", r"non-?measurable", False),
    ("in irradiated field", r"irradiated", False),
    ("resection status", r"resection of the enhancing", True),
]


def bucket(rationale: str) -> list[str]:
    r = rationale.lower()
    hits = [label for label, pattern, _ in BUCKETS if re.search(pattern, r)]
    return hits or ["none recorded" if not r.strip() else "other"]


VISIBLE = {label for label, _, vis in BUCKETS if vis}


def load(path: Path, profile: str) -> list[dict]:
    rows = [
        r
        for r in csv.DictReader(path.open())
        if r["profile"] == profile and r["expert"] in SCORABLE and r["calc"] in SCORABLE
    ]
    for r in rows:
        r["week_f"] = float(r["week"]) if r["week"] else 0.0
        r["agree_b"] = r["calc"] == r["expert"]
        r["buckets"] = bucket(r["expert_rationale"])
    return rows


def pct(n: int, d: int) -> str:
    return f"{n / d:.0%}" if d else "—"


def pct1(n: int, d: int) -> str:
    return f"{n / d:.1%}" if d else "—"


def fmt_ratio(v: str) -> str:
    if v in ("", "None"):
        return "—"
    return f"{float(v):+.0%}"


def esc(s: str) -> str:
    return html.escape(s or "")


def as_ascii(page: str) -> str:
    """Every non-ASCII character as a numeric reference.

    The page body cannot declare its own charset -- the publishing host owns the <head> -- and a
    plain static server sends text/html with no charset parameter, at which point the browser
    guesses and 'mm3' renders as mojibake. Character references sidestep the question entirely.
    """
    return page.encode("ascii", "xmlcharrefreplace").decode("ascii")


# --------------------------------------------------------------------------------------
# page
# --------------------------------------------------------------------------------------

def build(rows: list[dict], profile: str) -> str:
    total, agree = len(rows), sum(1 for r in rows if r["agree_b"])

    # --- rationale cross-tab, expert PD only (the only class with full rationale coverage)
    pd_rows = [r for r in rows if r["expert"] == "PD"]
    tab: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in pd_rows:
        for b in r["buckets"]:
            tab[b][1] += 1
            if r["calc"] == "PD":
                tab[b][0] += 1
    ordered = sorted(tab.items(), key=lambda kv: -kv[1][1])

    # The sharpest cut: split the reader's PD calls by whether the evidence they cited was inside
    # this ruleset's field of view at all. Three-way, because many rationales cite both.
    def split(pred):
        sub = [r for r in pd_rows if pred(set(r["buckets"]))]
        return sum(1 for r in sub if r["calc"] == "PD"), len(sub)

    tiers = [
        ("only findings the rule can see", *split(lambda b: b <= VISIBLE)),
        ("a mix of both", *split(lambda b: bool(b & VISIBLE) and not b <= VISIBLE)),
        ("only findings it cannot see", *split(lambda b: not (b & VISIBLE))),
    ]
    tier_html = "".join(
        f'<div class="tier"><span class="tn">{pct(ok, n)}</span>'
        f'<span class="tl">expert cited {esc(label)}</span>'
        f'<span class="td">{ok} of {n} progression calls matched</span></div>'
        for label, ok, n in tiers
        if n
    )

    xt_rows = "".join(
        f'<tr class="{"vis" if label in VISIBLE else "blind"}">'
        f'<td>{esc(label)}<span class="tag">{"in view" if label in VISIBLE else "not encoded"}</span></td>'
        f'<td class="num">{n}</td><td class="num">{ok}</td>'
        f'<td class="num rate"><span class="bar" style="--w:{ok / n * 100:.1f}%"></span>{pct(ok, n)}</td></tr>'
        for label, (ok, n) in ordered
        if n
    )

    # --- per-patient blocks
    by_patient: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_patient[r["patient"]].append(r)
    for v in by_patient.values():
        v.sort(key=lambda r: r["week_f"])

    patients = sorted(
        by_patient.items(),
        key=lambda kv: (
            sum(1 for r in kv[1] if r["agree_b"]) / len(kv[1]),
            -len(kv[1]),
            kv[0],
        ),
    )

    blocks = []
    for name, rs in patients:
        ok = sum(1 for r in rs if r["agree_b"])
        strip = "".join(
            f'<i class="s {"y" if r["agree_b"] else "n"}" title="{esc(r["timepoint"])}: '
            f'expert {r["expert"]}, calc {r["calc"]}"></i>'
            for r in rs
        )
        body = []
        for r in rs:
            unk = (
                f'<span class="unk">blind to: {esc(r["unknowns"].replace("|", ", "))}</span>'
                if r["unknowns"]
                else ""
            )
            prov = (
                f'<span class="unk">first scored {esc(r["provisional"])}, then revised</span>'
                if r["provisional"]
                else ""
            )
            pseudo = (
                '<span class="unk">inside the post-radiotherapy window</span>'
                if r["pseudoprogression_risk"] == "1"
                else ""
            )
            body.append(
                f'<tr class="row {"y" if r["agree_b"] else "n"}" '
                f'data-expert="{r["expert"]}" data-agree="{int(r["agree_b"])}">'
                f'<td class="scan">{esc(r["timepoint"].replace("week-", "wk "))}</td>'
                f'<td><span class="chip {r["expert"].lower()}">{r["expert"]}</span>'
                f'<div class="why">{esc(r["expert_rationale"]) or "<i>no rationale recorded</i>"}</div></td>'
                f'<td><span class="chip {r["calc"].lower()}">{r["calc"]}</span>'
                f'<div class="why">{esc(r["reason"])}{prov}{pseudo}{unk}</div></td>'
                f'<td class="num meas">{float(r["enhancing_mm3"]):,.0f}<span>mm³</span><br>'
                f'<b>{fmt_ratio(r["change_vs_nadir"])}</b><span>nadir</span><br>'
                f'{fmt_ratio(r["change_vs_baseline"])}<span>base</span></td>'
                f"</tr>"
            )
        blocks.append(
            f'<section class="pt" data-patient="{esc(name)}" data-rate="{ok / len(rs):.4f}">'
            f'<h3><span>{esc(name)}</span><span class="strip">{strip}</span>'
            f'<span class="score">{ok}/{len(rs)}<b>{pct(ok, len(rs))}</b></span></h3>'
            f'<table class="cases"><thead><tr><th>scan</th><th>expert reader</th>'
            f'<th>our ruleset</th><th class="num">enhancing</th></tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></section>'
        )

    return TEMPLATE.format(
        profile=esc(profile),
        total=total,
        agree=agree,
        agree_pct=pct1(agree, total),
        n_patients=len(patients),
        n_disagree=total - agree,
        xt_rows=xt_rows,
        tier_html=tier_html,
        n_pd=len(pd_rows),
        blocks="\n".join(blocks),
    )


TEMPLATE = """<title>Call by Call</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap">
<style>
:root {{
  --paper:#F4F6F8; --surface:#FFFFFF; --ink:#16202B; --muted:#5D6B79;
  --rule:#D6DDE4; --rule-soft:#E7ECF1;
  --cr:#0F7B6C; --pr:#55913F; --sd:#A9782A; --pd:#B93F36;
  --cr-bg:#E2F1EE; --pr-bg:#E9F1E2; --sd-bg:#F6EEDC; --pd-bg:#F8E5E3;
  --ok:#0F7B6C; --no:#B93F36; --accent:#2E6F9E;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --paper:#0E141A; --surface:#161F28; --ink:#E2E9EF; --muted:#8FA0AE;
    --rule:#2A3540; --rule-soft:#212B35;
    --cr:#3EBBA5; --pr:#8CC96F; --sd:#DCA84D; --pd:#E86B5F;
    --cr-bg:#123632; --pr-bg:#1D3320; --sd-bg:#382C15; --pd-bg:#3B1D1B;
    --ok:#3EBBA5; --no:#E86B5F; --accent:#62A8DC;
  }}
}}
:root[data-theme="dark"] {{
  --paper:#0E141A; --surface:#161F28; --ink:#E2E9EF; --muted:#8FA0AE;
  --rule:#2A3540; --rule-soft:#212B35;
  --cr:#3EBBA5; --pr:#8CC96F; --sd:#DCA84D; --pd:#E86B5F;
  --cr-bg:#123632; --pr-bg:#1D3320; --sd-bg:#382C15; --pd-bg:#3B1D1B;
  --ok:#3EBBA5; --no:#E86B5F; --accent:#62A8DC;
}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--paper);color:var(--ink);
  font-family:"Source Serif 4",Georgia,serif;font-size:16px;line-height:1.6}}
.wrap{{max-width:1080px;margin:0 auto;padding:clamp(1.8rem,1rem+3vw,4rem) clamp(1rem,.4rem+2vw,2.4rem) 6rem}}
h1,h2,h3,th,.chip,.eyebrow,.tag,.score,button{{font-family:Archivo,Helvetica,Arial,sans-serif}}
.mono,.num,.scan,.why,code{{font-family:"IBM Plex Mono",ui-monospace,Menlo,monospace}}

header{{border-bottom:2px solid var(--ink);padding-bottom:1.4rem;margin-bottom:2.4rem}}
.eyebrow{{font-size:.68rem;font-weight:600;letter-spacing:.16em;text-transform:uppercase;
  color:var(--muted);margin:0 0 .8rem}}
h1{{font-size:clamp(1.9rem,1.2rem+3vw,3.1rem);font-weight:700;letter-spacing:-.025em;
  line-height:1.03;margin:0 0 .8rem;text-wrap:balance}}
.stand{{color:var(--muted);max-width:62ch;margin:0;font-size:1.04rem}}
.kpis{{display:flex;flex-wrap:wrap;gap:.3rem 2.2rem;margin-top:1.4rem;
  font-family:"IBM Plex Mono",monospace;font-size:.76rem;color:var(--muted)}}
.kpis b{{color:var(--ink);font-weight:600}}

h2{{font-size:1.32rem;font-weight:700;letter-spacing:-.015em;margin:2.8rem 0 .7rem}}
p{{margin:0 0 1rem;max-width:66ch}}
.lead{{color:var(--muted)}}

table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}
th{{text-align:left;font-size:.64rem;font-weight:600;letter-spacing:.1em;text-transform:uppercase;
  color:var(--muted);padding:0 .6rem .5rem;border-bottom:1px solid var(--ink);white-space:nowrap}}
td{{padding:.55rem .6rem;border-bottom:1px solid var(--rule-soft);vertical-align:top;font-size:.88rem}}
.num,th.num{{text-align:right;font-family:"IBM Plex Mono",monospace}}

/* cross-tab */
.xt{{background:var(--surface);border:1px solid var(--rule);border-radius:6px;padding:1.1rem 1.2rem;margin:1.4rem 0}}
.xt td:first-child{{font-size:.9rem}}
.tag{{display:inline-block;margin-left:.6rem;font-size:.6rem;font-weight:600;letter-spacing:.08em;
  text-transform:uppercase;padding:.1em .45em;border-radius:3px;vertical-align:.1em}}
tr.vis .tag{{color:var(--cr);background:var(--cr-bg)}}
tr.blind .tag{{color:var(--pd);background:var(--pd-bg)}}
.tiers{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:1px;
  background:var(--rule);border:1px solid var(--rule);border-radius:6px;overflow:hidden;margin:1.4rem 0}}
.tier{{background:var(--surface);padding:1rem 1.1rem 1.1rem;display:flex;flex-direction:column;gap:.15rem}}
.tn{{font-family:Archivo,sans-serif;font-size:2rem;font-weight:700;letter-spacing:-.03em;line-height:1}}
.tier:first-child .tn{{color:var(--cr)}}
.tier:nth-child(2) .tn{{color:var(--sd)}}
.tier:last-child .tn{{color:var(--pd)}}
.tl{{font-size:.88rem;margin-top:.35rem}}
.td{{font-family:"IBM Plex Mono",monospace;font-size:.72rem;color:var(--muted)}}
td.rate{{white-space:nowrap}}
.bar{{display:inline-block;height:.42rem;width:var(--w);max-width:82px;background:var(--accent);
  border-radius:2px;margin-right:.5rem;vertical-align:.06em;opacity:.75}}

/* controls */
.controls{{position:sticky;top:0;z-index:5;background:var(--paper);
  border-bottom:1px solid var(--rule);padding:.85rem 0;margin:2rem 0 1.4rem;
  display:flex;flex-wrap:wrap;gap:.5rem .7rem;align-items:center}}
input[type=search]{{font:inherit;font-family:"IBM Plex Mono",monospace;font-size:.82rem;
  padding:.4rem .65rem;border:1px solid var(--rule);border-radius:4px;
  background:var(--surface);color:var(--ink);min-width:9rem}}
button{{font-size:.72rem;font-weight:600;letter-spacing:.05em;padding:.42rem .7rem;
  border:1px solid var(--rule);border-radius:4px;background:var(--surface);
  color:var(--muted);cursor:pointer}}
button[aria-pressed=true]{{background:var(--ink);color:var(--paper);border-color:var(--ink)}}
button:focus-visible,input:focus-visible{{outline:2px solid var(--accent);outline-offset:2px}}
.count{{margin-left:auto;font-family:"IBM Plex Mono",monospace;font-size:.74rem;color:var(--muted)}}

/* patient blocks */
.pt{{background:var(--surface);border:1px solid var(--rule);border-radius:6px;
  padding:.9rem 1.1rem 1.1rem;margin-bottom:1rem}}
.pt h3{{display:flex;align-items:center;gap:.9rem;margin:0 0 .6rem;font-size:1rem;font-weight:700}}
.strip{{display:flex;gap:2px;flex-wrap:wrap}}
.s{{width:9px;height:16px;border-radius:2px;display:block}}
.s.y{{background:var(--ok);opacity:.85}} .s.n{{background:var(--no);opacity:.85}}
.score{{margin-left:auto;font-family:"IBM Plex Mono",monospace;font-size:.78rem;
  font-weight:400;color:var(--muted);white-space:nowrap}}
.score b{{margin-left:.5rem;color:var(--ink)}}
.cases{{min-width:640px}}
.tablewrap{{overflow-x:auto}}
.scan{{font-size:.78rem;color:var(--muted);white-space:nowrap;padding-top:.7rem}}
tr.row{{border-left:3px solid transparent}}
tr.row.n td:first-child{{box-shadow:inset 3px 0 0 var(--no)}}
tr.row.y td:first-child{{box-shadow:inset 3px 0 0 var(--ok)}}
.chip{{display:inline-block;font-size:.66rem;font-weight:700;letter-spacing:.07em;
  padding:.15em .5em;border-radius:3px}}
.chip.cr{{color:var(--cr);background:var(--cr-bg)}}
.chip.pr{{color:var(--pr);background:var(--pr-bg)}}
.chip.sd{{color:var(--sd);background:var(--sd-bg)}}
.chip.pd{{color:var(--pd);background:var(--pd-bg)}}
.why{{font-size:.74rem;line-height:1.45;color:var(--muted);margin-top:.3rem;max-width:34ch}}
.why i{{opacity:.6}}
.unk{{display:block;color:var(--accent);opacity:.9;margin-top:.15rem}}
.meas{{font-size:.74rem;line-height:1.5;color:var(--muted);white-space:nowrap;padding-top:.55rem}}
.meas b{{color:var(--ink);font-weight:600}}
.meas span{{opacity:.55;margin-left:.3em}}
.empty{{display:none;color:var(--muted);font-style:italic;padding:2rem 0}}
footer{{margin-top:3rem;padding-top:1.3rem;border-top:2px solid var(--ink);
  font-size:.76rem;color:var(--muted);max-width:66ch}}
@media (max-width:640px){{
  .pt h3{{flex-wrap:wrap}} .score{{margin-left:0}} .controls{{position:static}}
}}
</style>

<div class="wrap">
<header>
  <p class="eyebrow">rano.criteria &middot; profile {profile}</p>
  <h1>Call by Call</h1>
  <p class="stand">Every scored scan, the expert reader's call beside the ruleset's, each with the reasoning that produced it. Where they disagree, the reader's own rationale usually says why.</p>
  <div class="kpis">
    <span><b>{total}</b> scored scans</span>
    <span><b>{n_patients}</b> patients</span>
    <span><b>{agree}</b> agree ({agree_pct})</span>
    <span><b>{n_disagree}</b> disagree</span>
  </div>
</header>

<h2>The disagreements are not spread evenly</h2>
<p class="lead">The reader recorded a rationale on all {n_pd} of their progression calls. Bucketing those rationales separates the scans judged on a measured enhancing lesion &mdash; the one criterion this ruleset implements &mdash; from the scans judged on evidence it has no access to.</p>

<div class="tiers">{tier_html}</div>

<p class="lead">Same rule, same thresholds, three very different accuracies &mdash; sorted only by what kind of evidence the reader was looking at. Below, the same PD calls broken out by each finding they cited.</p>

<div class="xt">
  <table>
    <thead><tr><th>expert cited, on a PD call</th><th class="num">scans</th><th class="num">we agreed</th><th class="num">rate</th></tr></thead>
    <tbody>{xt_rows}</tbody>
  </table>
</div>
<p class="lead">A rationale can cite several findings, so the rows overlap and do not sum to {n_pd}. The pattern is the point: agreement is highest where the reader measured the same thing the rule measures, and drops wherever the decisive evidence was a new lesion, a non-target lesion, or T2/FLAIR spread.</p>

<div class="controls">
  <input type="search" id="q" placeholder="patient…" aria-label="Filter by patient">
  <button id="f-all" aria-pressed="true">All scans</button>
  <button id="f-dis" aria-pressed="false">Disagreements only</button>
  <button class="cls" data-cls="CR" aria-pressed="false">CR</button>
  <button class="cls" data-cls="PR" aria-pressed="false">PR</button>
  <button class="cls" data-cls="SD" aria-pressed="false">SD</button>
  <button class="cls" data-cls="PD" aria-pressed="false">PD</button>
  <span class="count" id="count"></span>
</div>

<div class="tablewrap" id="list">
{blocks}
</div>
<p class="empty" id="empty">No scans match those filters.</p>

<footer>
<p>Patients ordered worst agreement first, longest trajectory first within that. The strip beside each name is one mark per scan in chronological order. Class buttons filter on the <em>expert's</em> call. Volumes are DeepBraTumIA's shipped <code>Enhancing_Core</code>, read by key. Expert abbreviations are expanded from the rating file's own header; <span class="mono">Target L.</span> entries keep their measurements verbatim.</p>
</footer>
</div>

<script>
(function () {{
  const q = document.getElementById('q');
  const bAll = document.getElementById('f-all'), bDis = document.getElementById('f-dis');
  const clsBtns = [...document.querySelectorAll('.cls')];
  const blocks = [...document.querySelectorAll('.pt')];
  const count = document.getElementById('count'), empty = document.getElementById('empty');
  let disOnly = false;

  function apply() {{
    const term = q.value.trim().toLowerCase();
    const classes = clsBtns.filter(b => b.getAttribute('aria-pressed') === 'true')
                           .map(b => b.dataset.cls);
    let shownRows = 0, shownPts = 0;
    for (const block of blocks) {{
      const nameHit = !term || block.dataset.patient.toLowerCase().includes(term);
      let any = 0;
      for (const row of block.querySelectorAll('tr.row')) {{
        const ok = nameHit
          && (!disOnly || row.dataset.agree === '0')
          && (!classes.length || classes.includes(row.dataset.expert));
        row.style.display = ok ? '' : 'none';
        if (ok) any++;
      }}
      block.style.display = any ? '' : 'none';
      shownRows += any;
      if (any) shownPts++;
    }}
    count.textContent = shownRows + ' scans \\u00b7 ' + shownPts +
                        (shownPts === 1 ? ' patient' : ' patients');
    empty.style.display = shownRows ? 'none' : 'block';
  }}

  q.addEventListener('input', apply);
  bAll.addEventListener('click', () => {{
    disOnly = false;
    bAll.setAttribute('aria-pressed', 'true');
    bDis.setAttribute('aria-pressed', 'false');
    apply();
  }});
  bDis.addEventListener('click', () => {{
    disOnly = true;
    bDis.setAttribute('aria-pressed', 'true');
    bAll.setAttribute('aria-pressed', 'false');
    apply();
  }});
  clsBtns.forEach(b => b.addEventListener('click', () => {{
    b.setAttribute('aria-pressed', b.getAttribute('aria-pressed') === 'true' ? 'false' : 'true');
    apply();
  }}));
  apply();
}})();
</script>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calls", type=Path, default=DEFAULT_CALLS)
    ap.add_argument("--profile", default="mrano_volumetric")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    rows = load(args.calls, args.profile)
    if not rows:
        print(f"no scored rows for profile {args.profile!r} in {args.calls}")
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(as_ascii(build(rows, args.profile)))
    agree = sum(1 for r in rows if r["agree_b"])
    print(f"{len(rows)} scored scans, {agree} agree ({agree / len(rows):.1%})")
    print(f"wrote {args.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
