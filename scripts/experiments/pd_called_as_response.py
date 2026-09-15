"""The doctor called progression, the pipeline called a response. Why, case by case?

Numbers behind docs/FINDINGS_2026-09-14.md, sections 1-4.

COHORT DISCIPLINE: counts are aggregated over all 91 patients (numbers only). Every per-case
detail, measurement and rationale is read ONLY after the held-out arm has been filtered out.
"""
import re, sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path[:0] = ["src", "scripts"]
import run_rano_calls as H
from run_measurement import ENHANCING, components
from rano.adapters.lumiere import paths
from rano.adapters.lumiere.zip_ref import ZipSource
from rano.criteria import DEFAULT_PROFILE as C, PROFILES, Response, assess_trajectory
from rano.criteria.rano import ReferenceState
from rano.measurement import measure_lesion

ROOT = Path(".")
volumes = H.load_shipped_volumes(ROOT / "Imaging-v202211.zip")
expert = H.load_expert(ROOT / "LUMIERE-ExpertRating-v202211.csv")
arms = H.load_arms(ROOT / "output/cohort/cohort_lock.json")
traj = H.build_trajectories(volumes, expert, H.RT_END_WEEK)
RESPONSE = (Response.CR, Response.PR)


def run(crit, patients):
    for p in sorted(patients):
        ref, ms = traj[p]
        yield p, ref, ms, assess_trajectory(p, ms, crit, reference=ref)


def hits(crit, patients):
    for p, ref, ms, res in run(crit, patients):
        for i, a in enumerate(res.assessments):
            r = expert.get((p, a.timepoint))
            if r and r.rating == "PD" and a.call in RESPONSE:
                yield p, ref, ms, res, i


# ---- 1. how many, where (aggregate) -----------------------------------------------------
everyone = list(traj)
h = list(hits(C, everyone))
print(f"expert PD -> pipeline CR/PR ({C.name}): {len(h)}")
print("  by arm:", dict(Counter((arms.get(p, 'unassigned'), res.assessments[i].call.value) for p, _, _, res, i in h)))
unlimited = C.variant("unlimited_confirmation", confirmation_max_weeks=None)
print(f"  with confirmation_max_weeks=None: {len(list(hits(unlimited, everyone)))}")

# ---- 2. per case, held-out arm filtered FIRST -------------------------------------------
permitted = [p for p in traj if arms.get(p, "unassigned") != "held_out"]
src = ZipSource(str(ROOT / "Imaging-v202211.zip"))
with_t2 = {(p, a.timepoint): a for p, _, _, res in run(PROFILES["mrano_with_t2"], permitted) for a in res.assessments}


def ruler(p, tp):
    enh = np.asarray(src.open_nifti(paths.dbt_mask(p, tp)).dataobj) == ENHANCING
    out = []
    for sel in components(enh):
        m = np.zeros(enh.shape, bool); m[tuple(sel.T)] = True
        r = measure_lesion(m).measurement
        out.append(f"{len(sel)} mm3 {r.long_mm:.1f}x{r.perp_mm:.1f}{'' if r.measurable else ' (not measurable)'}")
    return "; ".join(out[:3]) or "no lesion >= 20 mm3"


for p, ref, ms, res, i in hits(C, permitted):
    a, m = res.assessments[i], ms[i]
    st = ReferenceState(ref.enhancing_mm3, ref.enhancing_mm3, ref.t2_flair_mm3)
    for x in ms[:i]:
        st = st.advanced_by(x)
    surgeries = [x.timepoint for x in ms[:i] if expert.get((p, x.timepoint)) and expert[(p, x.timepoint)].rating == "Post-Op"]
    v = volumes[(p, m.timepoint)]
    print(f"\n{p} {m.timepoint}  expert PD -> {a.call}")
    print(f"  branch     {a.reason}")
    print(f"  enhancing  {m.enhancing_mm3:,.0f}  nadir {st.nadir_mm3:,.0f}  baseline {st.baseline_mm3:,.0f} ({ref.timepoint})"
          f"  necrosis {v['necrosis_nonenhancing']:,.0f}  edema {v['edema']:,.0f}")
    print(f"  weeks since RT end {m.weeks_since_rt:g} (in 12-wk window: {m.weeks_since_rt < C.pseudoprogression_weeks})"
          f"  pseudoprogression flag {int(a.pseudoprogression_risk)}")
    print(f"  surgeries since reference: {surgeries or 'none'}   with T2 on: {with_t2[(p, m.timepoint)].call}")
    print(f"  rationale: {expert[(p, m.timepoint)].expert_rationale_display() or '(none)'}")
    for tp in [m.timepoint, ref.timepoint, *surgeries[-1:]]:
        print(f"  ruler {tp:<11} {ruler(p, tp)}")

# ---- 3. T2 component vs the doctor's call (aggregate) -----------------------------------
print("\nT2/FLAIR (edema) >= +40% vs its own nadir, by the doctor's call:")
t2 = {c: [] for c in ("CR", "PR", "SD", "PD")}
for p, _, _, res in run(C, everyone):
    for a in res.assessments:
        r = expert.get((p, a.timepoint))
        if r and r.rating in t2 and a.scorable and a.t2_change_vs_nadir is not None:
            t2[r.rating].append(a.t2_change_vs_nadir >= 0.40)
for c, s in t2.items():
    print(f"  {c}: {sum(s)}/{len(s)} ({sum(s)/len(s):.0%})")

# ---- 4. scans where the segmenter reports zero enhancing tissue ------------------------
zero = [(p, a.timepoint) for p, _, _, res in run(C, everyone) for a in res.assessments
        if a.scorable and a.enhancing_mm3 == 0 and expert.get((p, a.timepoint)) and expert[(p, a.timepoint)].rating in t2]
print(f"\nzero-enhancing scored scans: {len(zero)}  doctor said {dict(Counter(expert[k].rating for k in zero))}")
print("  by arm:", dict(Counter(arms.get(p, 'unassigned') for p, _ in zero)))
ok = [k for k in zero if arms.get(k[0], "unassigned") != "held_out"]
print(f"  permitted {len(ok)}: doctor said {dict(Counter(expert[k].rating for k in ok))}")
