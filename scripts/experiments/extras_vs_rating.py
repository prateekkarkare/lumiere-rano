"""Are our 'extra' new-lesion calls real disease the notes did not enumerate, or noise?

The notes are terse and not an exhaustive inventory, so silence is not proof of absence. But
the RANO call is independent evidence: a genuinely new lesion forces PD. If the extras pile up
on visits the radiologist already called PD, they are probably real. If they scatter across CR
and SD, they are noise.
"""
import csv, json, sys
sys.path.insert(0,'src')
import numpy as np
from collections import Counter
from rano.adapters.lumiere.zip_ref import ZipSource
from rano.adapters.lumiere import paths
from rano.registration import RigidFit, dilate, resample_like_fixed

SLACK, FLOOR, CORE_MM, MIN_LUMP = 3, 20, 1, 20
src = ZipSource('Imaging-v202211.zip')
reg = json.loads(open('output/registration/transforms.json').read())
rows_csv = list(csv.DictReader(open('LUMIERE-ExpertRating-v202211.csv')))
RK = [k for k in rows_csv[0] if k.startswith('Rating (')][0]
RR = [k for k in rows_csv[0] if k.startswith('Rating rationale')][0]
note = {(r['Patient'], r['Date']): (r[RK].strip(), r[RR].strip()) for r in rows_csv}

FWD26 = tuple((dx,dy,dz) for dx in (0,1) for dy in (-1,0,1) for dz in (-1,0,1) if (dx,dy,dz)>(0,0,0))
def _sl(d): return (slice(None),slice(None)) if d==0 else ((slice(0,-d),slice(d,None)) if d>0 else (slice(-d,None),slice(0,d)))
def components(mask, floor=FLOOR):
    n=int(mask.sum())
    if n==0: return []
    idx=np.full(mask.shape,-1,np.int64); idx[mask]=np.arange(n); par=list(range(n))
    def find(x):
        while par[x]!=x: par[x]=par[par[x]]; x=par[x]
        return x
    for off in FWD26:
        s=[_sl(d) for d in off]; a,b=idx[tuple(x[0] for x in s)],idx[tuple(x[1] for x in s)]
        m=(a>=0)&(b>=0)
        for u,v in zip(a[m].tolist(),b[m].tolist()):
            ru,rv=find(u),find(v)
            if ru!=rv: par[max(ru,rv)]=min(ru,rv)
    roots=np.fromiter((find(i) for i in range(n)),np.int64,n); coords=np.argwhere(mask)
    out=[coords[roots==r] for r in np.unique(roots) if (roots==r).sum()>=floor]
    out.sort(key=len,reverse=True); return out
def erode(m, r=1):
    out=m.copy()
    for _ in range(r):
        e=out.copy()
        for ax in (0,1,2): e &= np.roll(out,1,ax)&np.roll(out,-1,ax)
        out=e
    return out

res = []
for pat, rec in reg["patients"].items():
    prev = None
    for tp, v in rec["visits"].items():
        enh = np.asarray(src.open_nifti(paths.dbt_mask(pat, tp)).dataobj) == 1
        cur = resample_like_fixed(enh.astype(np.uint8), RigidFit.from_dict(v), labels=True) > 0
        if prev is not None:
            gained = cur & ~dilate(prev, SLACK)
            lumps = []
            for c in components(gained):
                blob = np.zeros_like(cur); blob[c[:,0],c[:,1],c[:,2]] = True
                if erode(blob, CORE_MM).any(): lumps.append(len(c))
            rating, txt = note.get((pat, tp), ('',''))
            said = 'new ' in txt.lower() and 'l.' in txt.lower()
            fired = any(s >= MIN_LUMP for s in lumps)
            res.append((pat, tp, rating, said, fired, sorted(lumps, reverse=True)))
        prev = cur

real = [r for r in res if r[2] in ('CR','PR','SD','PD')]
silent = [r for r in real if not r[3]]
print(f"{len(real)} scorable transitions; {len(silent)} where the notes mention no new lesion.\n")
print(f"  {'RANO call':<10} {'silent visits':>14} {'we fired':>10} {'rate':>8}")
base = Counter(r[2] for r in silent)
fired = Counter(r[2] for r in silent if r[4])
for k in ('CR','PR','SD','PD'):
    if base[k]:
        print(f"  {k:<10} {base[k]:>14} {fired[k]:>10} {100*fired[k]/base[k]:>7.0f}%")
print(f"  {'ALL':<10} {len(silent):>14} {sum(fired.values()):>10} "
      f"{100*sum(fired.values())/len(silent):>7.0f}%")

pd_rate = 100*fired['PD']/max(base['PD'],1)
non_pd_n = sum(base[k] for k in ('CR','PR','SD'))
non_pd_f = sum(fired[k] for k in ('CR','PR','SD'))
print(f"\n  on PD visits we fire {pd_rate:.0f}% of the time; "
      f"on CR/PR/SD visits {100*non_pd_f/max(non_pd_n,1):.0f}%")
print("\n  the extras that landed on a NON-PD visit (the ones that would worry me):")
for pat, tp, rating, said, f, lumps in silent:
    if f and rating != 'PD':
        print(f"    {pat:<14} {tp:<12} {rating:<4} lumps {lumps[:3]}   {note[(pat,tp)][1][:46]}")
