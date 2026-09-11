"""Newly-enhancing tissue: a separate focus that merely touches the old tumour, or a rind on it?

If a new focus really does bud off some distance away and only joins up through intervening
tissue, the GAINED voxels will contain a compact lump whose centre sits well clear of the old
lesion. If instead the gained voxels are a thin skin all over the old lesion, then nothing new
appeared at all -- and the radiologist is seeing something this segmentation never marked.
"""
import csv, json, sys
sys.path.insert(0,'src')
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from track_registered import label
from rano.adapters.lumiere.zip_ref import ZipSource
from rano.adapters.lumiere import paths
from rano.registration import RigidFit, resample_like_fixed

PAT = 'Patient-072'
src = ZipSource('Imaging-v202211.zip')
reg = json.loads(open('output/registration/transforms.json').read())['patients'][PAT]
rows = list(csv.DictReader(open('LUMIERE-ExpertRating-v202211.csv')))
rr = [k for k in rows[0] if k.startswith('Rating rationale')][0]
note = {r['Date']: r[rr].strip() for r in rows if r['Patient']==PAT}
tps = list(reg['visits'].keys())

def aligned(tp):
    m = np.asarray(src.open_nifti(paths.dbt_mask(PAT, tp)).dataobj) == 1
    return resample_like_fixed(m.astype(np.uint8), RigidFit.from_dict(reg['visits'][tp]),
                               labels=True) > 0

print(f"{PAT} — where newly-enhancing tissue lands, relative to the previous visit's tumour\n")
print(f"  {'visit':<12} {'gained':>8} {'lumps>=20':>10}  {'biggest lump: size / distance from old':<40} radiologist")
prev = None
for tp in tps:
    cur = aligned(tp)
    if prev is not None and cur.any():
        gained = cur & ~prev
        lab, sizes = label(gained)
        old_pts = np.argwhere(prev)
        desc = "—"
        if sizes:
            big = np.argwhere(lab == 1)
            c = big.mean(0)
            # distance from this lump's centre to the nearest voxel of the OLD tumour
            d = float(np.sqrt(((old_pts - c) ** 2).sum(1).min())) if len(old_pts) else float('nan')
            desc = f"{sizes[0]:>6,} mm3   {d:>5.1f} mm away"
        txt = note.get(tp, '')
        said = 'NEW' if 'new' in txt.lower() else '   '
        print(f"  {tp:<12} {int(gained.sum()):>8,} {len(sizes):>10}  {desc:<40} {said} {txt[:34]}")
    prev = cur
