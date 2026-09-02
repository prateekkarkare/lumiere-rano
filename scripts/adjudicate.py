#!/usr/bin/env python3
"""
Build the adjudication worksheet: every disagreement, bucketed, with the evidence needed to
RULE on it -- and no verdict of its own.

WHY THIS EXISTS
    The evaluation harness (run_rano_calls.py) tells you THAT the rule and the expert disagree.
    It cannot tell you WHO IS WRONG, and neither can this script. That is a human judgement, and
    it is the whole deliverable. What a machine can do is put the evidence for each candidate
    explanation on the same line as the disagreement, so the ruling takes seconds instead of a
    manual dig through three files.

    Every column below is one of the "one-line checks" from the W9 hypothesis list. Nothing here
    is a score, a threshold, or a recommendation. The ``verdict`` and ``ruling`` columns ship
    EMPTY, on purpose.

THE THREE BUCKETS (precedence order, so every disagreement lands in exactly one)
    1. the segmentation reported enhancing == 0.0 -- a measurement-layer problem, not a rule one
    2. the expert's own rationale names a NEW LESION or T2 progression -- components the rule
       structurally cannot see (new_lesion is None on 100% of timepoints)
    3. everything else: a genuine rule / threshold / reference disagreement -- the only bucket
       where a ruling is meaningful

THE EVIDENCE COLUMNS, and which hypothesis each one settles
    nadir_from / postop_between  H3  a Post-Op row between the nadir and this scan = void comparison
    baseline_tp / baseline_week  H4  a baseline week < RT_END_WEEK is a PRE-radiotherapy reference
    noise_band_pp / margin_pp    H7  is the change bigger than our own measured uncertainty?
    h6_signature                 H6  down a long way from baseline, up a long way from nadir
    necrosis_mm3                 H2  large necrosis beside enhancing == 0 = the segmenter mis-binned
    rationale_blank              --  the expert left no reason: candidate UNKNOWABLE
    less_than_3_months           H8  the dataset's own pseudoprogression flag, which we never read

Usage:
    .venv/bin/python scripts/adjudicate.py                 # bucket 3 worksheet, all patients
    .venv/bin/python scripts/adjudicate.py --bucket all    # every disagreement
    .venv/bin/python scripts/adjudicate.py --cohort practice --limit 10
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rano.adapters.lumiere import weeks  # noqa: E402
from rano.contract.data_contract import uncertainty_pp  # noqa: E402
from rano.criteria import PROFILES, Response, assess_trajectory  # noqa: E402

import run_rano_calls as H  # the loading + trajectory code, reused rather than re-derived  # noqa: E402

#: The expert's own words for the two components the rule cannot see. Matched on the RAW
#: rationale, before the harness expands its abbreviations.
_NEW_LESION = re.compile(r"\bnew\b", re.I)
_T2_PROGR = re.compile(r"T2[- ]?Progr", re.I)


def bucket_of(enhancing_mm3, rationale: str) -> int:
    """1, 2 or 3 -- in precedence order, so the buckets partition the disagreements."""
    if enhancing_mm3 is not None and enhancing_mm3 == 0.0:
        return 1
    if _NEW_LESION.search(rationale) or _T2_PROGR.search(rationale):
        return 2
    return 3


def replay_references(reference, measurements, criteria):
    """Re-derive, per assessed timepoint, WHERE its nadir came from.

    ``ReferenceState`` carries the nadir VALUE but not its provenance, and provenance is exactly
    what H3 needs. This walks the trajectory the same way ``assess_trajectory`` does -- seed from
    the reference scan, fold each measurement in only AFTER it has been assessed -- so the nadir
    reported here is the one the call was actually made against.
    """
    out: dict[str, dict] = {}
    nadir_v = reference.enhancing_mm3 if reference is not None else None
    nadir_tp = reference.timepoint if reference is not None else None
    for m in measurements:
        out[m.timepoint] = {"nadir_mm3": nadir_v, "nadir_from": nadir_tp}
        if m.enhancing_mm3 is not None and (nadir_v is None or m.enhancing_mm3 < nadir_v):
            nadir_v, nadir_tp = m.enhancing_mm3, m.timepoint
    return out



def _pct(x) -> str:
    return "--" if x in (None, "") else f"{float(x):+.0%}"


def _flag(v) -> bool:
    """Truthy for both paths into render(): in-memory rows carry ints, rows read back from the
    worksheet CSV carry strings. Comparing against '1' silently reports 'no' for every in-memory
    row -- exactly the kind of quiet wrong answer this project exists to avoid."""
    return str(v) in ("1", "True")


def render(rows) -> str:
    """One readable card per disagreement: the numbers, both sides' reasoning, then the evidence
    lines for each hypothesis that could explain it. No verdict -- that line is left blank."""
    out: list[str] = []
    for i, r in enumerate(rows, 1):
        out.append("=" * 78)
        out.append(f"[{i:>2}]  {r['patient']}  {r['timepoint']}   "
                   f"expert {r['expert']}  ->  calc {r['calc']}      (arm: {r['arm']})")
        out.append("-" * 78)
        enh = r["enhancing_mm3"]
        out.append(f"  enhancing {float(enh):,.0f} mm3" if enh not in (None, "") else "  enhancing --")
        out.append(f"    vs nadir     {_pct(r['change_vs_nadir'])}"
                   f"   (nadir {float(r['nadir_mm3']):,.0f} mm3 from {r['nadir_from']})"
                   if r["nadir_mm3"] not in (None, "") else
                   f"    vs nadir     {_pct(r['change_vs_nadir'])}")
        out.append(f"    vs baseline  {_pct(r['change_vs_baseline'])}"
                   f"   (baseline {r['baseline_tp']}, week {r['baseline_week']})")
        nec, ede = r["necrosis_mm3"], r["edema_mm3"]
        out.append(f"    necrosis {float(nec):,.0f} mm3   edema {float(ede):,.0f} mm3"
                   if nec not in (None, "") and ede not in (None, "") else "")
        out.append("")
        out.append(f"  EXPERT SAYS  {r['expert']}   {r['expert_rationale'] or '(no rationale recorded)'}")
        out.append(f"  RULE SAYS    {r['calc']}   {r['calc_reason']}")
        if r["unknowns"]:
            out.append(f"               unavailable to the rule: {r['unknowns'].replace('|', ', ')}")
        out.append("")
        out.append("  EVIDENCE")
        bw = r["baseline_week"]
        out.append(f"    H4 pre-RT baseline    {'YES' if bw not in (None,'') and float(bw) < 10 else 'no '}"
                   f"  baseline sits at week {bw} (RT ends ~week 10)")
        out.append(f"    H3 nadir voided       {'YES' if r['postop_between'] else 'no '}"
                   f"  {('surgery at ' + r['postop_between'].replace('|', ', ') + ' sits between the nadir scan and this one') if r['postop_between'] else 'no Post-Op row between the nadir scan and this one'}")
        m = r["margin_pp"]
        out.append(f"    H7 inside our noise   {'YES' if m not in (None,'') and float(m) < 0 else 'no '}"
                   f"  change {abs(float(r['change_vs_nadir']))*100:,.0f} pp vs combined uncertainty band {r['noise_band_pp']} pp"
                   if r["margin_pp"] not in (None, "") and r["change_vs_nadir"] not in (None, "")
                   else "    H7 inside our noise   --   no usable ratio")
        out.append(f"    H6 rebound divergence {'YES' if _flag(r['h6_signature']) else 'no '}"
                   f"  (<= -50% vs baseline AND >= +40% vs nadir)")
        out.append(f"    H5 confirmation drove {'YES' if r['driver'] == 'confirmation' else 'no '}"
                   f"  {('provisional was ' + r['provisional']) if r['provisional'] else ''}")
        out.append(f"    -- rationale blank    {'YES' if _flag(r['rationale_blank']) else 'no '}"
                   f"  {'the expert recorded no reason; the data may not settle it' if _flag(r['rationale_blank']) else ''}")
        out.append(f"    H8 <3 months flag     {'YES' if _flag(r['less_than_3_months']) else 'no '}"
                   f"  (LUMIERE's own pseudoprogression column)")
        out.append("")
        out.append("  VERDICT  ______________________   (BUG / DIVERGENCE / UNKNOWABLE)")
        out.append("  BECAUSE  ______________________________________________________________")
        out.append("")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", type=Path, default=H.DEFAULT_ZIP)
    ap.add_argument("--ratings", type=Path, default=H.DEFAULT_RATINGS)
    ap.add_argument("--lock", type=Path, default=H.DEFAULT_LOCK)
    ap.add_argument("--out", type=Path, default=ROOT / "output" / "adjudication")
    ap.add_argument("--profile", default="mrano_volumetric")
    ap.add_argument("--cohort", choices=("all", "practice", "held_out"), default="all")
    ap.add_argument("--bucket", default="3", choices=("1", "2", "3", "all"))
    ap.add_argument("--rt-end-week", type=float, default=H.RT_END_WEEK)
    ap.add_argument("--limit", type=int, default=0, help="print at most N rows (0 = all)")
    ap.add_argument("--exclude-arm", action="append", default=[],
                    help="repeatable; e.g. --exclude-arm held_out to protect the locked arm")
    ap.add_argument("--render", action="store_true", help="human-readable cards instead of JSON")
    args = ap.parse_args()

    criteria = PROFILES[args.profile]
    volumes = H.load_shipped_volumes(args.zip)
    expert = H.load_expert(args.ratings)
    arms = H.load_arms(args.lock)

    # the dataset's own pseudoprogression flag -- H8 notes we never read it
    l3m = {
        (r["Patient"].strip(), r["Date"].strip()): r["LessThan3Months"].strip().lower() == "x"
        for r in csv.DictReader(args.ratings.open())
    }

    if args.cohort != "all":
        keep = {p for p, a in arms.items() if a == args.cohort}
        volumes = {k: v for k, v in volumes.items() if k[0] in keep}

    trajectories = H.build_trajectories(volumes, expert, args.rt_end_week)

    rows: list[dict] = []
    for patient in sorted(trajectories):
        reference, measurements = trajectories[patient]
        result = assess_trajectory(patient, measurements, criteria, reference=reference)
        refs = replay_references(reference, measurements, criteria)

        # every Post-Op week for this patient -- H3's "is there a surgery in between"
        postop_weeks = sorted(
            w
            for (p, tp), row in expert.items()
            if p == patient and row.rating == H.POST_OP and (w := weeks.week_offset(tp)) is not None
        )

        for a in result.assessments:
            row = expert.get((patient, a.timepoint))
            rating = row.rating if row is not None else ""
            if rating not in H.SCORABLE_RATINGS or not a.scorable or a.call.value == rating:
                continue  # agreements and unscorable rows are not adjudicable

            rationale = row.rationale.strip() if row is not None else ""
            vols = volumes.get((patient, a.timepoint), {})
            ref = refs.get(a.timepoint, {})
            week = weeks.week_offset(a.timepoint)
            nadir_from_week = weeks.week_offset(ref.get("nadir_from") or "")

            # H3: a surgery strictly between the nadir scan and this one voids the comparison
            postop_between = (
                [w for w in postop_weeks if nadir_from_week < w < week]
                if nadir_from_week is not None and week is not None
                else []
            )

            # H7: our own measured uncertainty on both sides of the ratio. Conservative: the
            # bands add, because a ratio's relative error is the sum of its terms' relative errors.
            u_now = uncertainty_pp(a.enhancing_mm3) if a.enhancing_mm3 is not None else None
            u_nad = uncertainty_pp(ref["nadir_mm3"]) if ref.get("nadir_mm3") is not None else None
            band = (u_now + u_nad) if (u_now is not None and u_nad is not None) else None
            change_pp = abs(a.change_vs_nadir) * 100 if a.change_vs_nadir is not None else None
            margin = (change_pp - band) if (band is not None and change_pp is not None) else None

            rows.append(
                {
                    "bucket": bucket_of(a.enhancing_mm3, rationale),
                    "patient": patient,
                    "timepoint": a.timepoint,
                    "week": week,
                    "arm": arms.get(patient, "unassigned"),
                    "expert": rating,
                    "calc": a.call.value,
                    "shape": f"{rating}->{a.call.value}",
                    "enhancing_mm3": a.enhancing_mm3,
                    "necrosis_mm3": vols.get("necrosis_nonenhancing"),
                    "edema_mm3": vols.get("edema"),
                    "change_vs_nadir": a.change_vs_nadir,
                    "change_vs_baseline": a.change_vs_baseline,
                    "nadir_mm3": ref.get("nadir_mm3"),
                    "nadir_from": ref.get("nadir_from"),
                    "postop_between": "|".join(f"week-{w:03.0f}" for w in postop_between),
                    "baseline_tp": result.baseline_timepoint,
                    "baseline_week": weeks.week_offset(result.baseline_timepoint or ""),
                    "noise_band_pp": round(band, 1) if band is not None else None,
                    "margin_pp": round(margin, 1) if margin is not None else None,
                    "h6_signature": int(
                        a.change_vs_baseline is not None
                        and a.change_vs_nadir is not None
                        and a.change_vs_baseline <= -0.50
                        and a.change_vs_nadir >= 0.40
                    ),
                    "rationale_blank": int(not rationale or rationale.lower() == "none"),
                    "less_than_3_months": int(bool(l3m.get((patient, a.timepoint)))),
                    "driver": a.driver,
                    "provisional": a.provisional_call.value if a.provisional_call else "",
                    "unknowns": "|".join(a.unknowns),
                    "calc_reason": a.reason,
                    "expert_rationale": rationale,
                    "verdict": "",   # BUG | DIVERGENCE | UNKNOWABLE   <- yours to fill
                    "ruling": "",    # one line of evidence            <- yours to fill
                }
            )

    rows.sort(key=lambda r: (r["bucket"], r["patient"], weeks.sort_key(r["timepoint"])))

    counts = {b: sum(1 for r in rows if r["bucket"] == b) for b in (1, 2, 3)}
    print(f"profile {criteria.name}   cohort {args.cohort}   disagreements {len(rows)}")
    print(f"  bucket 1  enhancing == 0.0                    {counts[1]:>4}")
    print(f"  bucket 2  rationale names new lesion / T2     {counts[2]:>4}")
    print(f"  bucket 3  genuine rule disagreement           {counts[3]:>4}")

    sel = rows if args.bucket == "all" else [r for r in rows if r["bucket"] == int(args.bucket)]
    if args.exclude_arm:
        sel = [r for r in sel if r["arm"] not in args.exclude_arm]
        print(f"\n  excluded arms: {', '.join(args.exclude_arm)}  ->  {len(sel)} rows remain")
    shapes: dict[str, int] = {}
    for r in sel:
        shapes[r["shape"]] = shapes.get(r["shape"], 0) + 1
    print("\n  shape (expert -> calc):  " + "   ".join(
        f"{k} {v}" for k, v in sorted(shapes.items(), key=lambda kv: -kv[1])))

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"worksheet_bucket{args.bucket}.csv"
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(sel)
    print(f"\nwrote {path.relative_to(ROOT)} ({len(sel)} rows, verdict column EMPTY)")

    if args.render:
        print()
        print(render(sel[: args.limit] if args.limit else sel))
    elif args.limit:
        print()
        for r in sel[: args.limit]:
            print(json.dumps(r, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
