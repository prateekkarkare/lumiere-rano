"""Old baseline rule vs new, side by side. Permitted patients only (held-out arm never opened).

  old: the first study the expert labelled Post-Op, matched by timepoint label
  new: the first scan far enough past THIS patient's radiotherapy (rano.criteria.timeline)

Usage: .venv/bin/python scripts/experiments/baseline_rule_ab.py [Patient-012 ...]
"""
import sys
from collections import Counter
from pathlib import Path

sys.path[:0] = ["src", "scripts"]
import run_rano_calls as H
from rano.adapters.lumiere.treatment import read_surgical_labels, treatment_from_labels
from rano.criteria import DEFAULT_PROFILE as C, Response, assess_trajectory
from rano.criteria.timeline import place_scan

ROOT = Path(".")
SCORABLE = {"CR", "PR", "SD", "PD"}
volumes = H.load_shipped_volumes(ROOT / "Imaging-v202211.zip")
expert = H.load_expert(ROOT / "LUMIERE-ExpertRating-v202211.csv")
arms = H.load_arms(ROOT / "output/cohort/cohort_lock.json")
surgical = read_surgical_labels(str(ROOT / "LUMIERE-ExpertRating-v202211.csv"))
permitted = {p for p, _ in volumes if arms.get(p, "unassigned") != "held_out"}   # filter FIRST

old = H.build_trajectories(volumes, expert, H.RT_END_WEEK)
new = H.build_trajectories(volumes, expert, H.RT_END_WEEK, surgical=surgical)


def calls(traj, patient):
    ref, ms = traj[patient]
    if ref is None and not ms:
        return {}, None
    res = assess_trajectory(patient, ms, C, reference=ref)
    return {a.timepoint: a for a in res.assessments}, (ref.timepoint if ref else None)


moved, changed, dropped, gained, agree = Counter(), [], [], [], Counter()
for p in sorted(permitted):
    a, ref_old = calls(old, p)
    b, ref_new = calls(new, p)
    moved["same reference" if ref_old == ref_new else "reference moved"] += 1
    for tp in sorted(set(a) | set(b), key=lambda t: (t not in a, t)):
        rating = expert.get((p, tp))
        if rating is None or rating.rating not in SCORABLE:
            continue
        ca, cb = (a.get(tp), b.get(tp))
        sa = ca.call.value if ca and ca.scorable else None
        sb = cb.call.value if cb and cb.scorable else None
        for tag, call in (("old", sa), ("new", sb)):
            if call:
                agree[(tag, call == rating.rating)] += 1
        if sa and not sb:
            dropped.append((p, tp, sa, rating.rating))
        elif sb and not sa:
            gained.append((p, tp, sb, rating.rating))
        elif sa != sb:
            changed.append((p, tp, sa, sb, rating.rating))

print(f"permitted patients: {len(permitted)}   {dict(moved)}")
for tag in ("old", "new"):
    ok, n = agree[(tag, True)], agree[(tag, True)] + agree[(tag, False)]
    print(f"  {tag}: {ok}/{n} scored scans agree with the expert ({ok / n:.1%})")
print(f"\nstopped being scored: {len(dropped)}   newly scored: {len(gained)}   call changed: {len(changed)}")
print("  dropped, by the call they used to get:", dict(Counter(f"{c} (expert {e})" for _, _, c, e in dropped)))
for p, tp, sa, sb, e in changed:
    print(f"  changed  {p} {tp:<11} {sa} -> {sb}   expert {e}")

for patient in sys.argv[1:] or ["Patient-012"]:
    print("\n" + "=" * 96)
    record, problem = treatment_from_labels(surgical.get(patient), [tp for pp, tp in volumes if pp == patient] and
                                            [H.weeks.week_offset(tp) for pp, tp in volumes if pp == patient])
    print(f"{patient}   surgeries {[s.week for s in record.surgeries or ()]}   "
          f"radiotherapy {[(c.start.week, c.end.week) for c in record.radiotherapy or ()]}"
          + (f"   PROBLEM: {problem}" if problem else ""))
    a, ref_old = calls(old, patient)
    b, ref_new = calls(new, patient)
    print(f"  reference: old {ref_old}   new {ref_new}")
    print(f"  {'scan':<11}{'enh mm3':>9}  {'expert':<8}{'old':<20}{'new':<20}{'where this scan sits'}")
    for tp in sorted(set(a) | set(b), key=lambda t: H.weeks.sort_key(t)):
        row = expert.get((patient, tp))
        week = H.weeks.week_offset(tp)
        ca, cb = a.get(tp), b.get(tp)
        enh = volumes[(patient, tp)].get("enhancing")
        stage = place_scan(week, record).stage.value if week is not None else "undated"
        fmt = lambda c: "" if c is None else (c.call.value + (f" ({c.driver})" if c.call in
                                              (Response.CR, Response.PR, Response.PD, Response.SD) else ""))
        print(f"  {tp:<11}{enh:>9,.0f}  {(row.rating if row else '-'):<8}{fmt(ca):<20}{fmt(cb):<20}{stage}")
