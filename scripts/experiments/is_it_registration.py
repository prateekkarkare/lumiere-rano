"""Is the brain-mask mismatch really misalignment? Three rivals, three tests.

  registration error   -> masks near-identical in SIZE, mismatch a thin shell all round
  skull-strip variation-> masks differ in SIZE; mismatch patchy at the edge
  field-of-view differs-> mismatch piled at the top or bottom slices
  real anatomy change  -> mismatch concentrated near the lesion
"""
import json, re, sys
sys.path.insert(0,'src')
import numpy as np
from rano.adapters.lumiere.zip_ref import ZipSource
from rano.adapters.lumiere import paths
src = ZipSource('Imaging-v202211.zip')
practice = [p["patient_id"] for p in json.load(open('output/cohort/cohort_lock.json'))['practice']['patients']]

vol_ratio, shell_frac, zend_frac, near_les = [], [], [], []
for pat in practice:
    tps = sorted({m.group(1) for m in (re.match(rf'Imaging/{pat}/([^/]+)/DeepBraTumIA', n)
                                       for n in src.names) if m})
    tps = [t for t in tps if src.exists(paths.dbt_brain_mask(pat, t))]
    prev = prevles = None
    for tp in tps:
        cur = np.asarray(src.open_nifti(paths.dbt_brain_mask(pat, tp)).dataobj) > 0
        les = np.asarray(src.open_nifti(paths.dbt_mask(pat, tp)).dataobj) > 0
        if prev is not None:
            a, b = int(prev.sum()), int(cur.sum())
            vol_ratio.append(max(a,b)/max(min(a,b),1))
            mism = prev ^ cur
            n = int(mism.sum())
            if n:
                # 1. is the mismatch a thin shell? -> how far is it from the shared brain edge
                both = prev & cur
                inner = both.copy()
                for ax in (0,1,2):
                    inner &= np.roll(both,1,ax) & np.roll(both,-1,ax)
                edge = both & ~inner
                # crude: fraction of mismatch voxels lying outside the shared core
                shell_frac.append(100*int((mism & ~both).sum())/n)
                # 2. is it piled at the top/bottom of the stack?
                zs = np.argwhere(mism)[:,2]
                zlo, zhi = np.percentile(np.argwhere(both)[:,2], [5,95])
                zend_frac.append(100*int(((zs < zlo) | (zs > zhi)).sum())/n)
                # 3. is it near the lesion?
                if les.any() or (prevles is not None and prevles.any()):
                    seed = les | (prevles if prevles is not None else les)
                    grown = seed.copy()
                    for _ in range(10):
                        g = grown.copy()
                        for ax in (0,1,2): g |= np.roll(grown,1,ax) | np.roll(grown,-1,ax)
                        grown = g
                    near_les.append(100*int((mism & grown).sum())/n)
        prev, prevles = cur, les

print(f"{len(vol_ratio)} consecutive visit pairs\n")
print(f"  brain-mask VOLUME ratio (larger/smaller):  median {np.median(vol_ratio):.3f}   "
      f"worst {max(vol_ratio):.3f}")
print(f"     -> if skull-stripping varied a lot, this would be well above 1.0")
print(f"\n  mismatch lying at the top/bottom 5% of the stack: median {np.median(zend_frac):.1f}%")
print(f"     -> if the scans covered different amounts of head, this would be large")
print(f"\n  mismatch lying within 10 mm of the lesion: median {np.median(near_les):.1f}%")
print(f"     -> if it were real anatomical change from the tumour, this would dominate")
