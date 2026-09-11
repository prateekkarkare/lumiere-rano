"""A better rule: a new lesion need not be DISCONNECTED, only NEW.

The topological rule ("overlaps nothing from last visit") cannot see a nodule that buds off the
edge of existing tumour -- and on two patients that is every single new lesion the radiologist
recorded. So ask a different question: of the tissue that is enhancing NOW and was not before,
is any of it a compact lump rather than a thin layer added to a growing edge?

A lesion that merely grew gains a SHELL: thin, wrapped around the old one. A genuinely new focus
gains a LUMP: it has an interior. Eroding tells them apart -- a shell disappears, a lump survives.
"""
import csv, json, re, sys
sys.path.insert(0,'src')
import numpy as np
from rano.adapters.lumiere.zip_ref import ZipSource
from rano.adapters.lumiere import paths
from rano.registration import RigidFit, dilate, resample_like_fixed
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from new_lesion_check import components, kind, note, reg, src   # reuse, no re-run of the table

SLACK, CORE_MM = 3, 1

def erode(m, r=1):
    out = m.copy()
    for _ in range(r):
        e = out.copy()
        for ax in (0,1,2): e &= np.roll(out,1,ax) & np.roll(out,-1,ax)
        out = e
    return out

rows = []
for pat, rec in reg["patients"].items():
    prev = None
    for tp, v in rec["visits"].items():
        enh = np.asarray(src.open_nifti(paths.dbt_mask(pat, tp)).dataobj) == 1
        cur = resample_like_fixed(enh.astype(np.uint8), RigidFit.from_dict(v), labels=True) > 0
        if prev is not None:
            gained = cur & ~dilate(prev, SLACK)
            # a lump is gained tissue with an interior; a growth shell has none
            lumps = []
            for c in components(gained, floor=20):
                blob = np.zeros_like(cur); blob[c[:,0], c[:,1], c[:,2]] = True
                if erode(blob, CORE_MM).any():
                    lumps.append(len(c))
            rating, txt = note.get((pat, tp), ('',''))
            rows.append((pat, tp, rating, kind(txt), sorted(lumps, reverse=True)))
        prev = cur

real = [r for r in rows if r[2] not in ('Post-Op','Pre-Op','')]
for MIN in (20, 50, 100, 200):
    hit  = sum(1 for r in real if r[3] and any(s >= MIN for s in r[4]))
    miss = sum(1 for r in real if r[3] and not any(s >= MIN for s in r[4]))
    extra= sum(1 for r in real if not r[3] and any(s >= MIN for s in r[4]))
    quiet= sum(1 for r in real if not r[3] and not any(s >= MIN for s in r[4]))
    print(f"  lump at least {MIN:>4} mm3:  caught {hit:>2}/{hit+miss}  "
          f"({100*hit/max(hit+miss,1):>3.0f}%)   extras {extra:>2}/{extra+quiet}")

print("\nper patient, at a 100 mm3 lump threshold:")
for pat in reg["patients"]:
    sub = [r for r in real if r[0]==pat and r[3]]
    if sub:
        h = sum(1 for r in sub if any(s>=100 for s in r[4]))
        print(f"  {pat:<14} caught {h}/{len(sub)}")
