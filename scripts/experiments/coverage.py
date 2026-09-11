"""Rotation, or did the scan simply not cover the whole head?

A rotation displaces most at the extremes but the brain still ENDS in the same place.
Truncated coverage means one visit's brain literally stops earlier -- and no registration
can invent data that was never acquired.
"""
import json, re, sys
sys.path.insert(0,'src')
import numpy as np
from rano.adapters.lumiere.zip_ref import ZipSource
from rano.adapters.lumiere import paths
src = ZipSource('Imaging-v202211.zip')
practice = [p["patient_id"] for p in json.load(open('output/cohort/cohort_lock.json'))['practice']['patients']]

dtop, dbot, spans = [], [], []
worst = []
for pat in practice:
    tps = sorted({m.group(1) for m in (re.match(rf'Imaging/{pat}/([^/]+)/DeepBraTumIA', n)
                                       for n in src.names) if m})
    tps = [t for t in tps if src.exists(paths.dbt_brain_mask(pat, t))]
    prev = None
    for tp in tps:
        cur = np.asarray(src.open_nifti(paths.dbt_brain_mask(pat, tp)).dataobj) > 0
        zs = np.flatnonzero(cur.any(axis=(0,1)))
        lo, hi = int(zs.min()), int(zs.max())
        spans.append((pat, tp, lo, hi, hi-lo+1))
        if prev is not None:
            dbot.append(abs(lo - prev[0])); dtop.append(abs(hi - prev[1]))
            worst.append((abs(hi-prev[1]) + abs(lo-prev[0]), pat, ptp, tp,
                          (prev[0], prev[1]), (lo, hi)))
        prev, ptp = (lo, hi), tp

dt, db = np.array(dtop), np.array(dbot)
print(f"{len(dt)} consecutive visit pairs — where the brain STARTS and STOPS in z\n")
print(f"  top of brain moves between visits:     median {np.median(dt):.1f} mm   "
      f"p90 {np.percentile(dt,90):.1f}   worst {dt.max()} mm")
print(f"  bottom of brain moves between visits:  median {np.median(db):.1f} mm   "
      f"p90 {np.percentile(db,90):.1f}   worst {db.max()} mm")
allspan = np.array([s[4] for s in spans])
print(f"\n  total brain height per visit: median {np.median(allspan):.0f} mm, "
      f"range {allspan.min()}–{allspan.max()} mm")
print(f"  pairs where the brain height differs by more than 10 mm: "
      f"{sum(1 for w in worst if abs((w[5][1]-w[5][0])-(w[4][1]-w[4][0]))>10)}/{len(worst)}")
print("\n  biggest disagreements (z range of brain, visit A -> visit B):")
for _, pat, a, b, ra, rb in sorted(worst, reverse=True)[:6]:
    print(f"    {pat:<13} {a:<11} {ra}  ->  {b:<11} {rb}   height {ra[1]-ra[0]+1} -> {rb[1]-rb[0]+1} mm")
