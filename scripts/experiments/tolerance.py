"""If the visits are misaligned by a few mm, does allowing a few mm of slack rescue tracking?"""
import json, re, sys
sys.path.insert(0,'src')
import numpy as np
from rano.adapters.lumiere.zip_ref import ZipSource
from rano.adapters.lumiere import paths
src = ZipSource('Imaging-v202211.zip')
practice = [p["patient_id"] for p in json.load(open('output/cohort/cohort_lock.json'))['practice']['patients']]

def grow(m, r):
    out = m.copy()
    for _ in range(r):
        g = out.copy()
        for ax in (0,1,2): g |= np.roll(out,1,ax) | np.roll(out,-1,ax)
        out = g
    return out

TOL = (0,1,2,3,5)
acc = {t: [] for t in TOL}
sizes = []
for pat in practice:
    tps = sorted({m.group(1) for m in (re.match(rf'Imaging/{pat}/([^/]+)/DeepBraTumIA', n)
                                       for n in src.names) if m})
    tps = [t for t in tps if src.exists(paths.dbt_mask(pat, t))]
    prev = None
    for tp in tps:
        cur = np.asarray(src.open_nifti(paths.dbt_mask(pat, tp)).dataobj) == 1
        if prev is not None and prev.any() and cur.any():
            sizes.append(min(int(prev.sum()), int(cur.sum())))
            for t in TOL:
                near = grow(prev, t) if t else prev
                acc[t].append(100*int((cur & near).sum())/int(cur.sum()))
        prev = cur

sizes = np.array(sizes)
print(f"{len(sizes)} visit pairs where both visits have enhancing disease.\n")
print("how much of the NEW visit's lesion sits within N mm of the OLD visit's lesion:\n")
print(f"  {'slack':>6} {'median':>8} {'p25':>7} {'pairs over 50%':>16}")
for t in TOL:
    a = np.array(acc[t])
    print(f"  {t:>4}mm {np.median(a):>7.0f}% {np.percentile(a,25):>6.0f}% "
          f"{(a>50).sum():>10}/{len(a)}")

print("\nsplit by lesion size (using 3 mm of slack):")
a3 = np.array(acc[3])
for lo, hi, name in [(0,500,'under 500 mm3'), (500,3000,'500 - 3,000'), (3000,10**9,'over 3,000')]:
    m = (sizes>=lo)&(sizes<hi)
    if m.sum(): print(f"  {name:<16} n={m.sum():>3}   median overlap {np.median(a3[m]):>5.0f}%")
