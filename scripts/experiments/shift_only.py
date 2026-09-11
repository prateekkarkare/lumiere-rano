"""How much of the misalignment is a plain shift, with no rotation?

Cheapest possible fix: move one visit's brain so its centre sits on the other's, then nudge
a few voxels to find the best fit. If that recovers most of the lost overlap, we do not need
a rotation solver -- and we do not need a new dependency.
"""
import json, re, sys
sys.path.insert(0,'src')
import numpy as np
from rano.adapters.lumiere.zip_ref import ZipSource
from rano.adapters.lumiere import paths
src = ZipSource('Imaging-v202211.zip')
practice = [p["patient_id"] for p in json.load(open('output/cohort/cohort_lock.json'))['practice']['patients']]

def jac(a, b):
    return 100*int((a & b).sum())/max(int((a | b).sum()), 1)

def shifted(m, d):
    out = m
    for ax, k in enumerate(d):
        if k: out = np.roll(out, int(k), ax)
    return out

before, after_c, after_r, dists = [], [], [], []
for pat in practice:
    tps = sorted({m.group(1) for m in (re.match(rf'Imaging/{pat}/([^/]+)/DeepBraTumIA', n)
                                       for n in src.names) if m})
    tps = [t for t in tps if src.exists(paths.dbt_brain_mask(pat, t))]
    prev = None
    for tp in tps:
        cur = np.asarray(src.open_nifti(paths.dbt_brain_mask(pat, tp)).dataobj) > 0
        if prev is not None:
            b = jac(prev, cur); before.append(b)
            d0 = np.rint(np.argwhere(prev).mean(0) - np.argwhere(cur).mean(0)).astype(int)
            dists.append(float(np.linalg.norm(d0)))
            c = jac(prev, shifted(cur, d0)); after_c.append(c)
            best, bd = c, d0                       # nudge +-2 voxels around the centroid guess
            for dx in (-2,-1,0,1,2):
                for dy in (-2,-1,0,1,2):
                    for dz in (-2,-1,0,1,2):
                        v = jac(prev, shifted(cur, d0 + np.array([dx,dy,dz])))
                        if v > best: best, bd = v, d0 + np.array([dx,dy,dz])
            after_r.append(best)
        prev = cur

b, c, r = np.array(before), np.array(after_c), np.array(after_r)
print(f"{len(b)} consecutive visit pairs, brain-mask overlap:\n")
print(f"  {'':<34}{'median':>9}{'p10':>8}{'worst':>8}")
print(f"  {'as shipped (atlas space)':<34}{np.median(b):>8.1f}%{np.percentile(b,10):>7.1f}%{b.min():>7.1f}%")
print(f"  {'after centring the two brains':<34}{np.median(c):>8.1f}%{np.percentile(c,10):>7.1f}%{c.min():>7.1f}%")
print(f"  {'after nudging +-2 voxels more':<34}{np.median(r):>8.1f}%{np.percentile(r,10):>7.1f}%{r.min():>7.1f}%")
gap_closed = (r - b) / np.maximum(100 - b, 1e-9) * 100
print(f"\n  share of the missing overlap that a pure shift recovers: "
      f"median {np.median(gap_closed):.0f}%")
print(f"  typical shift applied: {np.median(dists):.1f} voxels (= mm)")
