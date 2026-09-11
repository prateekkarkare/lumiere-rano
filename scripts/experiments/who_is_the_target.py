"""Patient-072 is stable at 13x11 for weeks. Which object in OUR mask is that?"""
import sys; sys.path.insert(0,'src')
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from reverse_engineer import components, caliper_mm, constrained_mm, src, paths

for tp, exp in [('week-078','14 x 13'), ('week-090','13 x 11'), ('week-105','13 x 11')]:
    a = np.asarray(src.open_nifti(paths.dbt_mask('Patient-072', tp)).dataobj)
    enh = a == 1
    print(f"\nPatient-072 {tp}   expert wrote: {exp}")
    print(f"  {'les':>4} {'vol mm3':>8} {'caliper':>8} {'necrosis touching':>18}")
    for k, sel in enumerate(components(enh)):
        sl_best = 0.0
        for z in np.unique(sel[:,2]):
            m = np.zeros(enh.shape[:2], bool); pts = sel[sel[:,2]==z]
            m[pts[:,0], pts[:,1]] = True
            sl_best = max(sl_best, caliper_mm(m))
        # how much label-2 sits immediately around this lesion?
        box = np.zeros(enh.shape, bool); box[sel[:,0], sel[:,1], sel[:,2]] = True
        gx = box.copy()
        for ax in (0,1,2):
            gx |= np.roll(box, 1, ax) | np.roll(box, -1, ax)
        touching = int(((a == 2) & gx).sum())
        print(f"  {'L'+str(k+1):>4} {len(sel):>8,} {sl_best:>8.1f} {touching:>18,}")
