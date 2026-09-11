"""Control: the BRAIN should sit in exactly the same place at every visit.

Every scan was registered to the MNI atlas independently, so 'same grid' is not the same
as 'same anatomy in the same voxels'. The brain mask is the control -- a patient's brain
does not change shape between visits, so any mismatch is registration error, not disease.
"""
import json, re, sys
sys.path.insert(0,'src')
import numpy as np
from rano.adapters.lumiere.zip_ref import ZipSource
from rano.adapters.lumiere import paths

src = ZipSource('Imaging-v202211.zip')
practice = [p["patient_id"] for p in json.load(open('output/cohort/cohort_lock.json'))['practice']['patients']]

print(f"  {'patient':<13} {'from':<12} {'to':<12} {'brain overlap':>14} {'centre shift mm':>16}")
allj, alld = [], []
for pat in practice:
    tps = sorted({m.group(1) for m in (re.match(rf'Imaging/{pat}/([^/]+)/DeepBraTumIA', n)
                                       for n in src.names) if m})
    tps = [t for t in tps if src.exists(paths.dbt_brain_mask(pat, t))]
    prev = None
    for tp in tps:
        cur = np.asarray(src.open_nifti(paths.dbt_brain_mask(pat, tp)).dataobj) > 0
        if prev is not None:
            inter = int((prev & cur).sum()); union = int((prev | cur).sum())
            j = 100*inter/max(union,1)
            d = float(np.linalg.norm(np.argwhere(cur).mean(0) - np.argwhere(prev).mean(0)))
            allj.append(j); alld.append(d)
            if len(allj) <= 6 or j < 90:
                print(f"  {pat:<13} {ptp:<12} {tp:<12} {j:>13.1f}% {d:>15.1f}")
        prev, ptp = cur, tp

j = np.array(allj); d = np.array(alld)
print(f"\nacross {len(j)} consecutive visit pairs:")
print(f"  brain-mask overlap:  median {np.median(j):.1f}%   worst {j.min():.1f}%   best {j.max():.1f}%")
print(f"  brain centre moves:  median {np.median(d):.1f} mm   worst {d.max():.1f} mm")
print(f"\nIf the brain itself does not line up, a small lesion cannot be expected to.")
