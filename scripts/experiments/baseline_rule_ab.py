"""Old baseline rule vs new, scored on the SAME scans. Written up in docs/BASELINE_2026-09-16.md, Part 3.

  old: the first study the expert labelled Post-Op                  (run_rano_calls.py default)
  new: the first scan >= 3 weeks past THIS patient's radiotherapy   (--baseline post-rt, RANO 2.0)

Both arms are built exactly as run_rano_calls.py builds them -- treatment record, surgery reset,
measurable-baseline gate -- so the HEADLINES must reproduce its two reports before anything below them
is read. The new rule never scores a scan the old one does not, so like-for-like means the old rule
restricted to the new rule's scans.

Held-out patients enter only as aggregate counts; per-patient detail is refused for them.

Usage: .venv/bin/python scripts/experiments/baseline_rule_ab.py [Patient-012 ...]
"""
import random
import sys
from collections import Counter
from math import comb
from pathlib import Path

sys.path[:0] = ["src", "scripts"]
import run_rano_calls as H
from rano.adapters.lumiere.treatment import read_surgical_labels, treatment_from_labels
from rano.criteria import DEFAULT_PROFILE as C, compare_calls
from rano.criteria.timeline import place_scan

ROOT = Path(".")
ZIP = ROOT / "Imaging-v202211.zip"
CACHE = ROOT / "output/measurement/measurable_references.csv"
volumes = H.load_shipped_volumes(ZIP)
expert = H.load_expert(ROOT / "LUMIERE-ExpertRating-v202211.csv")
arms = H.load_arms(ROOT / "output/cohort/cohort_lock.json")
surgical = read_surgical_labels(str(ROOT / "LUMIERE-ExpertRating-v202211.csv"))


def arm(rule):
    traj = H.build_trajectories(volumes, expert, H.RT_END_WEEK, surgical=surgical, baseline_rule=rule)
    traj = H.apply_measurability(traj, H.measurable_flags(H.reference_scans(traj), ZIP, CACHE))
    pairs, rows = H.run_profile(traj, expert, arms, C)
    return traj, {(p.patient, p.timepoint): p for p in pairs}, rows


old_traj, old, old_rows = arm("post-op-label")
new_traj, new, new_rows = arm("post-rt")
common = set(old) & set(new)


def held_out(patient):
    return arms.get(patient, "unassigned") == "held_out"


def rating(patient, tp):
    return expert[(patient, tp)].rating if (patient, tp) in expert else "unrated"


def line(label, pairs):
    r = compare_calls(pairs)
    per = "  ".join(f"{c} {r.per_class[c].n_correct}/{r.per_class[c].support}" for c in ("CR", "PR", "SD", "PD"))
    return (f"  {label:<28} n={r.n_compared:<4} agree {r.agreement:6.1%} (always-PD {r.majority_baseline:5.1%})"
            f"  balanced {r.balanced_accuracy:6.1%}   {per}")


def agreement(keys, side):
    return compare_calls([side[k] for k in keys]).agreement


def balanced(keys, side):
    return compare_calls([side[k] for k in keys]).balanced_accuracy


def sign_test(a, b):
    """Exact two-sided sign test on the scans only one of the two rules gets right."""
    n, k = a + b, max(a, b)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2**n) if n else 1.0


def mechanism(a, b):
    """What differs between the two reasons behind a changed call. Approximate: it reads reason text."""
    for label, marker in (
        ("pseudoprogression window moved", "weeks of radiotherapy"),
        ("measurable gate on a different reference", "held no lesion big enough"),
        ("confirmation outcome changed", "not confirmed"),
    ):
        if (marker in a.reason) != (marker in b.reason):
            return label
    return "baseline/nadir moved"


def why_dropped(patient, tp):
    ref = new_traj[patient][0]
    if ref is None:
        return "patient never gets a post-RT baseline"
    if tp == ref.timepoint:
        return "becomes the new baseline"
    return "before the new baseline" if H.weeks.week_offset(tp) < ref.week else "other"


print("HEADLINES -- must reproduce run_rano_calls.py without and with --baseline post-rt")
print(line("old rule, all its scans", list(old.values())))
print(line("new rule, all its scans", list(new.values())))
print(f"  scored only by the new rule: {len(set(new) - set(old))}")

dropped = [k for k in old if k not in common]
print(f"\nSCANS THE NEW RULE STOPS SCORING: {len(dropped)}  {dict(Counter(why_dropped(*k) for k in dropped))}")
print(f"  expert calls on them: {dict(Counter(old[k].expert for k in dropped))}")

baselines = [(p, t[0]) for p, t in new_traj.items() if t[0] is not None]
late = sum(r.weeks_since_rt is not None and r.weeks_since_rt > 5 for _, r in baselines)
print(f"\nEXPERT CALL ON THE {len(baselines)} SCANS THE NEW RULE MAKES THE BASELINE: "
      f"{dict(Counter(rating(p, r.timepoint) for p, r in baselines))}")
print(f"  a PD/PR/CR there was measured against an EARLIER scan;  {late} are > 5 weeks after the assumed RT end")

for label, keep in (("ALL 91 PATIENTS", lambda p: True), ("OUTSIDE THE HELD-OUT ARM", lambda p: not held_out(p))):
    mine = [k for k in old if keep(k[0])]
    same = sorted(k for k in common if keep(k[0]))
    print(f"\n{label}")
    print(line("old rule, all its scans", [old[k] for k in mine]))
    print(line("old rule, the dropped scans", [old[k] for k in mine if k not in common]))
    print(line("OLD rule, same scans", [old[k] for k in same]))
    print(line("NEW rule, same scans", [new[k] for k in same]))
    for name, score in (("agreement", agreement), ("balanced", balanced)):
        a, b, c = score(mine, old), score(same, old), score(same, new)
        print(f"  {name:<9}  headline {100 * (c - a):+.1f} pp  =  scan set {100 * (b - a):+.1f}"
              f"  +  rule {100 * (c - b):+.1f}")

    only_old = sum(old[k].agree and not new[k].agree for k in same)
    only_new = sum(new[k].agree and not old[k].agree for k in same)
    changed = [k for k in same if old[k].predicted != new[k].predicted]
    print(f"  calls changed {len(changed)}/{len(same)}: only old agrees {only_old}, only new agrees {only_new}, "
          f"neither {len(changed) - only_old - only_new}   sign test p={sign_test(only_old, only_new):.3f}")

    # one patient's scans share a reference, so resample patients rather than scans
    patients = sorted({p for p, _ in same})
    by_patient = {p: [k for k in same if k[0] == p] for p in patients}
    rng, deltas = random.Random(0), []
    for _ in range(2000):
        draw = [k for p in (rng.choice(patients) for _ in patients) for k in by_patient[p]]
        deltas.append(balanced(draw, new) - balanced(draw, old))
    deltas.sort()
    print(f"  rule effect on balanced accuracy, patient bootstrap 95%: "
          f"{100 * deltas[50]:+.1f} to {100 * deltas[1949]:+.1f} pp")
    transitions = Counter(f"{old[k].predicted}->{new[k].predicted} ({old[k].expert})" for k in changed)
    print(f"  changed, old->new (expert): {dict(transitions.most_common())}")
    print(f"  why, from the reason text: {dict(Counter(mechanism(old[k], new[k]) for k in changed).most_common())}")

for patient in sys.argv[1:] or ["Patient-012"]:
    print("\n" + "=" * 96)
    if patient not in old_traj or held_out(patient):
        print(f"{patient}: not shown -- unknown, or in the held-out arm")
        continue
    record, problem = treatment_from_labels(
        surgical.get(patient), [H.weeks.week_offset(tp) for pp, tp in volumes if pp == patient]
    )
    print(f"{patient}   surgeries {[s.week for s in record.surgeries or ()]}   radiotherapy (assumed) "
          f"{[(c.start.week, c.end.week) for c in record.radiotherapy or ()]}"
          + (f"   PROBLEM: {problem}" if problem else ""))
    ref_old, ref_new = old_traj[patient][0], new_traj[patient][0]
    print(f"  reference: old {ref_old.timepoint if ref_old else None}   new {ref_new.timepoint if ref_new else None}")
    a = {r["timepoint"]: r for r in old_rows if r["patient"] == patient}
    b = {r["timepoint"]: r for r in new_rows if r["patient"] == patient}
    fmt = lambda r: "" if r is None else f"{r['calc']} ({r['driver']})"
    print(f"  {'scan':<11}{'enh mm3':>9}  {'expert':<8}{'old':<22}{'new':<22}where this scan sits")
    for tp in sorted(set(a) | set(b), key=H.weeks.sort_key):
        week = H.weeks.week_offset(tp)
        stage = place_scan(week, record).stage.value if week is not None else "undated"
        print(f"  {tp:<11}{volumes[(patient, tp)].get('enhancing', 0):>9,.0f}  {rating(patient, tp):<8}"
              f"{fmt(a.get(tp)):<22}{fmt(b.get(tp)):<22}{stage}")
