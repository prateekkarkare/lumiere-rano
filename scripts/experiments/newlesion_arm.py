"""Across all seven practice patients: when we say "new lesion", does the radiologist?

Two questions, one pass:
  1. Agreement. Our rule is "this object overlaps nothing from last visit". Theirs is written in
     the rationale column. How often do they coincide?
  2. Distance. When the radiologist reports a new lesion, how far from the previous tumour does
     the newly-enhancing tissue actually sit? On Patient-072 it was 0.2-6.7mm -- touching -- which
     an overlap rule can never see. The question is whether that is typical or peculiar to them.

Everything runs on registered masks, so misalignment is not a confound.
"""
import csv, json, sys
sys.path.insert(0, 'src')
import numpy as np
from rano.adapters.lumiere.zip_ref import ZipSource
from rano.adapters.lumiere import paths
from rano.registration import RigidFit, dilate, resample_like_fixed

FLOOR, SLACK = 20, 2
FWD26 = tuple((dx,dy,dz) for dx in (0,1) for dy in (-1,0,1) for dz in (-1,0,1) if (dx,dy,dz)>(0,0,0))
def _sl(d): return (slice(None),slice(None)) if d==0 else ((slice(0,-d),slice(d,None)) if d>0 else (slice(-d,None),slice(0,d)))

def label(mask):
    n = int(mask.sum()); out = np.zeros(mask.shape, np.int32)
    if n == 0: return out, []
    idx = np.full(mask.shape, -1, np.int64); idx[mask] = np.arange(n); par = list(range(n))
    def find(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    for off in FWD26:
        s = [_sl(d) for d in off]
        a, b = idx[tuple(x[0] for x in s)], idx[tuple(x[1] for x in s)]
        m = (a >= 0) & (b >= 0)
        for u, v in zip(a[m].tolist(), b[m].tolist()):
            ru, rv = find(u), find(v)
            if ru != rv: par[max(ru,rv)] = min(ru,rv)
    roots = np.fromiter((find(i) for i in range(n)), np.int64, n)
    coords = np.argwhere(mask)
    groups = sorted(((int((roots==r).sum()), coords[roots==r]) for r in np.unique(roots)),
                    key=lambda g: -g[0])
    sizes = []
    for k, (sz, sel) in enumerate((g for g in groups if g[0] >= FLOOR), start=1):
        out[sel[:,0], sel[:,1], sel[:,2]] = k; sizes.append(sz)
    return out, sizes

src = ZipSource('Imaging-v202211.zip')
reg = json.loads(open('output/registration/transforms.json').read())['patients']
rows = list(csv.DictReader(open('LUMIERE-ExpertRating-v202211.csv')))
rr = [k for k in rows[0] if k.startswith('Rating rationale')][0]
rk = [k for k in rows[0] if k.startswith('Rating (')][0]
notes = {(r['Patient'], r['Date']): (r[rk].strip(), r[rr].strip()) for r in rows}

out = []
for pat, rec in reg.items():
    prev = None
    for tp in rec['visits'].keys():
        m = np.asarray(src.open_nifti(paths.dbt_mask(pat, tp)).dataobj) == 1
        cur = resample_like_fixed(m.astype(np.uint8),
                                  RigidFit.from_dict(rec['visits'][tp]), labels=True) > 0
        if prev is not None:
            lab, sizes = label(cur)
            near = dilate(prev, SLACK)
            ours = [sizes[k-1] for k in range(1, len(sizes)+1) if not (lab == k)[near].any()]
            gained = cur & ~prev
            glab, gsizes = label(gained)
            dist = float('nan'); big = 0
            if gsizes and prev.any():
                c = np.argwhere(glab == 1).mean(0)
                dist = float(np.sqrt(((np.argwhere(prev) - c) ** 2).sum(1).min()))
                big = gsizes[0]
            rating, txt = notes.get((pat, tp), ('', ''))
            out.append({
                "patient": pat, "tp": tp, "rating": rating,
                "ours": ours, "we_say_new": bool(ours),
                "they_say_new": 'new' in txt.lower(),
                "kind": ('target' if 'new target' in txt.lower() else
                         'non-target' if 'new non-target' in txt.lower() else
                         'non-measurable' if 'new non-meas' in txt.lower() else
                         'other' if 'new' in txt.lower() else ''),
                "gained_mm3": int(gained.sum()), "biggest_new_lump_mm3": big,
                "lump_distance_mm": dist, "text": txt,
            })
        prev = cur

_dest = sys.argv[1] if len(sys.argv) > 1 else 'output/experiments/newlesion_arm.json'
import os; os.makedirs(os.path.dirname(_dest) or '.', exist_ok=True)
json.dump(out, open(_dest, 'w'), indent=1)
print(f"{len(out)} visit transitions across {len(reg)} patients -- ALL of them, Post-Op scans included. The CORRECTED scoreboard is at the bottom.")

tt = sum(1 for r in out if r['we_say_new'] and r['they_say_new'])
tf = sum(1 for r in out if r['we_say_new'] and not r['they_say_new'])
ft = sum(1 for r in out if not r['we_say_new'] and r['they_say_new'])
ff = sum(1 for r in out if not r['we_say_new'] and not r['they_say_new'])
print(f"\n                        radiologist says NEW    says nothing new")
print(f"  we find a new lesion        {tt:>10}            {tf:>10}")
print(f"  we find none                {ft:>10}            {ff:>10}")
n_they = tt + ft
print(f"\n  of the {n_they} visits where the radiologist reports a new lesion, we catch {tt}"
      f"  ({100*tt/max(n_they,1):.0f}%)")
print(f"  of the {tt+tf} visits where we report one, {tt} are confirmed"
      f"  ({100*tt/max(tt+tf,1):.0f}%)")

d_new = np.array([r['lump_distance_mm'] for r in out
                  if r['they_say_new'] and not np.isnan(r['lump_distance_mm'])])
d_not = np.array([r['lump_distance_mm'] for r in out
                  if not r['they_say_new'] and not np.isnan(r['lump_distance_mm'])])
print(f"\ndistance from the previous tumour to the biggest lump of NEW enhancing tissue:")
print(f"  visits the radiologist calls NEW      n={len(d_new):>3}  median {np.median(d_new):>5.1f} mm"
      f"   p90 {np.percentile(d_new,90):>6.1f} mm   max {d_new.max():>6.1f} mm")
print(f"  visits they do not                    n={len(d_not):>3}  median {np.median(d_not):>5.1f} mm"
      f"   p90 {np.percentile(d_not,90):>6.1f} mm   max {d_not.max():>6.1f} mm")
print(f"  NEW-visits where that lump is within 10 mm (touching-ish): "
      f"{(d_new<10).sum()}/{len(d_new)}")

from collections import Counter
print(f"\nwhat kind of new lesion they report (and whether we caught it):")
for k, n in Counter(r['kind'] for r in out if r['they_say_new']).most_common():
    got = sum(1 for r in out if r['they_say_new'] and r['kind'] == k and r['we_say_new'])
    print(f"  new {k:<16} {n:>3} visits, we caught {got}")


# ---------------------------------------------------------------------------------------------
# CORRECTED. Everything above scores all 86 transitions, which is how this was first run -- and
# it was wrong. 15 of them are Post-Op scans (the surgeon has just changed the anatomy, so
# comparing lesions across them is meaningless) and one has no expert rating at all. The cohort
# lock already defines an assessable timepoint as rated CR/PR/SD/PD. These are the numbers
# docs/progress.html reports.
# ---------------------------------------------------------------------------------------------
SCORABLE = {'CR', 'PR', 'SD', 'PD'}
d = [r for r in out if r['rating'] in SCORABLE]
print(f"\n\n=== CORRECTED: assessable timepoints only ({len(d)} of {len(out)}) ===")
tt = sum(1 for r in d if r['we_say_new'] and r['they_say_new'])
tf = sum(1 for r in d if r['we_say_new'] and not r['they_say_new'])
ft = sum(1 for r in d if not r['we_say_new'] and r['they_say_new'])
ff = sum(1 for r in d if not r['we_say_new'] and not r['they_say_new'])
print("                        radiologist: NEW     nothing new")
print(f"  we find a new lesion       {tt:>10}      {tf:>10}")
print(f"  we find none               {ft:>10}      {ff:>10}")
print(f"  recall {100*tt/max(tt+ft,1):.0f}%   precision {100*tt/max(tt+tf,1):.0f}%")
dn = np.array([r['lump_distance_mm'] for r in d if r['they_say_new'] and not np.isnan(r['lump_distance_mm'])])
dx = np.array([r['lump_distance_mm'] for r in d if not r['they_say_new'] and not np.isnan(r['lump_distance_mm'])])
print(f"\n  distance to the newest enhancing tissue: NEW median {np.median(dn):.1f} mm, "
      f"ordinary growth median {np.median(dx):.1f} mm")
n_they = tt + ft
print(f"\n  {'size threshold':>14} {'report':>7} {'confirmed':>10} {'precision':>10} {'recall':>7}")
for t in (20, 50, 100, 200, 500):
    rep = [r for r in d if r['ours'] and max(r['ours']) >= t]
    conf = sum(1 for r in rep if r['they_say_new'])
    print(f"  {t:>11} mm3 {len(rep):>7} {conf:>10} {100*conf/max(len(rep),1):>9.0f}% "
          f"{100*conf/max(n_they,1):>6.0f}%")
print("\n  where we report a new lesion and they do not:")
for r in sorted((r for r in d if r['we_say_new'] and not r['they_say_new']), key=lambda r: -max(r['ours'])):
    print(f"    {r['patient']:<13} {r['tp']:<10} {str(sorted(r['ours'], reverse=True)[:2]):<16} "
          f"{r['rating']:<4} {r['text'][:44]}")
