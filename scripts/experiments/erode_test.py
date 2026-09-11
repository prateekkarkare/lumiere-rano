"""Is Patient-072's single big blob really one lump, or several joined by thin necks?"""
import sys; sys.path.insert(0,'src')
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from track import label          # reuses the labeller
from rano.measurement import measure_lesion as _measure_lesion


def measure_lesion(mask):
    """The scratch ruler this experiment was written against returned a bare tuple. It became
    rano.measurement, which also fills enclosed holes, so piece measurements can differ slightly
    from the original run. The erosion split counts -- the point of the experiment -- do not."""
    r = _measure_lesion(mask)
    if r is None:
        return 0.0, 0.0, 0.0, -1
    return r.measurement.long_mm, r.measurement.perp_mm, r.product_mm2, r.slice_index
from rano.adapters.lumiere.zip_ref import ZipSource
from rano.adapters.lumiere import paths
src = ZipSource('Imaging-v202211.zip')

def erode(m, r=1):
    out = m.copy()
    for _ in range(r):
        e = out.copy()
        for ax in (0,1,2):
            e &= np.roll(out,1,ax) & np.roll(out,-1,ax)
        out = e
    return out

for TP, exp in [('week-073','13 x 11.5'), ('week-090','13 x 11'), ('week-105','13 x 11')]:
    enh = np.asarray(src.open_nifti(paths.dbt_mask('Patient-072', TP)).dataobj) == 1
    lab, sizes = label(enh)
    big = lab == 1
    print(f"\nPatient-072 {TP} — biggest lesion {int(big.sum())} mm3, expert measured {exp}")
    for r in (0, 1, 2):
        e = erode(big, r) if r else big
        el, es = label(e)
        print(f"   erode {r} mm -> {len(es)} piece(s), sizes {es[:6]}")
        if r and len(es) > 1:
            for k in range(1, min(len(es), 4)+1):
                D, P, prod, z = measure_lesion(el == k)
                print(f"        piece {k}: {es[k-1]:>6} mm3   {D:>5.1f} x {P:>5.1f} mm")
