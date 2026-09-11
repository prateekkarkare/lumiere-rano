"""THE test for Piece 2: does registration rescue lesion matching for SMALL lesions?

Before registration, measured across the practice arm with 3mm of slack:
    under 500 mm3 ->  1% of the new visit's lesion lands on the old one
    500-3,000     -> 45%
    over 3,000    -> 71%

The first row is the one that matters. A NEW lesion is small -- that is what makes it new -- so
1% is the number that was killing new-lesion detection. If registration does not move it, the
registration bought us nothing, however good the brain overlap looks.
"""
import json, re, sys
sys.path.insert(0,'src')
import numpy as np
from rano.adapters.lumiere.zip_ref import ZipSource
from rano.adapters.lumiere import paths
from rano.registration import RigidFit, dilate, resample_like_fixed

src = ZipSource('Imaging-v202211.zip')
reg = json.loads(open('output/registration/transforms.json').read())

def enhancing(pat, tp):
    return np.asarray(src.open_nifti(paths.dbt_mask(pat, tp)).dataobj) == 1

raw, done, sizes = [], [], []
for pat, rec in reg["patients"].items():
    tps = list(rec["visits"].keys())
    prev_raw = prev_reg = None
    for tp in tps:
        cur = enhancing(pat, tp)
        fit = RigidFit.from_dict(rec["visits"][tp])
        cur_reg = resample_like_fixed(cur.astype(np.uint8), fit, labels=True) > 0
        if prev_raw is not None and prev_raw.any() and cur.any():
            sizes.append(min(int(prev_raw.sum()), int(cur.sum())))
            raw.append(100*int((cur & dilate(prev_raw, 3)).sum())/int(cur.sum()))
            if cur_reg.any() and prev_reg.any():
                done.append(100*int((cur_reg & dilate(prev_reg, 3)).sum())/int(cur_reg.sum()))
            else:
                done.append(0.0)
        prev_raw, prev_reg = cur, cur_reg

sizes = np.array(sizes); raw = np.array(raw); done = np.array(done)
print(f"{len(sizes)} consecutive visit pairs, 3 mm of slack\n")
print(f"  {'lesion size':<18} {'n':>4} {'before':>9} {'after':>9} {'change':>9}")
for lo, hi, name in [(0,500,'under 500 mm3'), (500,3000,'500 - 3,000'), (3000,10**9,'over 3,000')]:
    m = (sizes>=lo)&(sizes<hi)
    if m.sum():
        b, a = np.median(raw[m]), np.median(done[m])
        print(f"  {name:<18} {m.sum():>4} {b:>8.0f}% {a:>8.0f}% {a-b:>+8.0f}pp")
print(f"  {'ALL':<18} {len(sizes):>4} {np.median(raw):>8.0f}% {np.median(done):>8.0f}% "
      f"{np.median(done)-np.median(raw):>+8.0f}pp")
