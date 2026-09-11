"""Approximate the radiation field, and ask whether new enhancement falls inside or outside it.

The radiologist's actual question is not "is this in a new place" but "is this even tumour" --
because irradiated tissue enhances like tumour does. They resolve it by asking whether the new
enhancement lies inside the treated field. We do not have a dose map, but glioblastoma radiotherapy
targets the resection cavity plus residual tumour plus a margin, so the post-operative baseline
mask dilated by that margin approximates it.

Label 2 (necrosis/non-enhancing) was shown earlier to contain the surgical cavity, which is exactly
what the plan is drawn around -- so the seed is labels 1+2 at the most recent post-operative scan.
Re-operations reset the zone, so the zone in force is the latest Post-Op scan BEFORE each visit.
"""
import csv, json, sys
sys.path.insert(0, 'src')
import numpy as np
from rano.adapters.lumiere.zip_ref import ZipSource
from rano.adapters.lumiere import paths
from rano.registration import RigidFit, dilate, resample_like_fixed

MARGINS = (0, 5, 10, 15, 20, 25)
SCORABLE = {'CR', 'PR', 'SD', 'PD'}
FLOOR = 20
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

def aligned(pat, tp, labels):
    a = np.asarray(src.open_nifti(paths.dbt_mask(pat, tp)).dataobj)
    m = np.isin(a, list(labels))
    return resample_like_fixed(m.astype(np.uint8), RigidFit.from_dict(reg[pat]['visits'][tp]),
                               labels=True) > 0

out = []
for pat, rec in reg.items():
    tps = list(rec['visits'].keys())
    zone_seed = None          # labels 1+2 at the most recent post-op scan
    prev_enh = None
    for tp in tps:
        rating, txt = notes.get((pat, tp), ('', ''))
        enh = aligned(pat, tp, {1})
        if rating == 'Post-Op':
            zone_seed = aligned(pat, tp, {1, 2})     # a re-operation resets the field
        elif rating in SCORABLE and prev_enh is not None and zone_seed is not None:
            gained = enh & ~prev_enh
            row = {"patient": pat, "tp": tp, "rating": rating, "text": txt,
                   "they_say_new": 'new' in txt.lower(),
                   "gained_mm3": int(gained.sum()), "by_margin": {}}
            for mg in MARGINS:
                zone = dilate(zone_seed, mg)
                outside = gained & ~zone
                olab, osizes = label(outside)
                row["by_margin"][str(mg)] = {
                    "outside_mm3": int(outside.sum()),
                    "biggest_lump_outside_mm3": osizes[0] if osizes else 0,
                    "n_lumps_outside": len(osizes),
                }
            out.append(row)
        prev_enh = enh
    print(f"  {pat}: {sum(1 for r in out if r['patient']==pat)} scored visits", flush=True)

_dest = sys.argv[1] if len(sys.argv) > 1 else 'output/experiments/treated_zone.json'
import os; os.makedirs(os.path.dirname(_dest) or '.', exist_ok=True)
json.dump(out, open(_dest, 'w'), indent=1)
print(f"\n{len(out)} visits with a treated zone in force")


# ---------------------------------------------------------------------------------------------
# The verdict, as summarised in docs/progress.html: does new tissue OUTSIDE the approximated
# field separate "new lesion" visits from "ordinary growth" visits? No -- at every margin the two
# groups move together, and the one promising cell (early visits) holds two cases.
# ---------------------------------------------------------------------------------------------
import re as _re
new = [r for r in out if r['they_say_new']]; grow = [r for r in out if not r['they_say_new']]
print(f"\n{len(new)} visits the radiologist calls NEW, {len(grow)} not")
print(f"  {'margin':>7}   NEW with a lump >=20mm3 outside   growth with one")
for mg in [str(m) for m in MARGINS]:
    a = sum(1 for r in new if r['by_margin'][mg]['biggest_lump_outside_mm3'] >= 20)
    b = sum(1 for r in grow if r['by_margin'][mg]['biggest_lump_outside_mm3'] >= 20)
    print(f"  {mg:>5}mm   {a:>3}/{len(new):<30}{b:>3}/{len(grow)}")
fl = [r for r in out if r['by_margin']['0']['biggest_lump_outside_mm3'] >= 20]
cf = sum(1 for r in fl if r['they_say_new'])
print(f"\n  best rule: margin 0 mm, lump >= 20 mm3 -> precision {100*cf/max(len(fl),1):.0f}% "
      f"against a {100*len(new)/max(len(out),1):.0f}% base rate")
def _week(tp):
    m = _re.match(r'week-(\d+)', tp)
    return int(m.group(1)) if m else -1
print("\n  split by time since the first scan (the field should only matter early):")
for lo, hi, name in [(0, 16, 'within ~3 months'), (16, 40, '4-9 months'), (40, 10**9, 'later')]:
    sub = [r for r in out if lo <= _week(r['tp']) < hi]
    if sub:
        hit = [r for r in sub if r['by_margin']['10']['biggest_lump_outside_mm3'] >= 20]
        print(f"    {name:<18} {len(sub):>3} visits, {sum(r['they_say_new'] for r in sub)} NEW; "
              f"margin 10 mm flags {len(hit)}, {sum(r['they_say_new'] for r in hit)} confirmed")
