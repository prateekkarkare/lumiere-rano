"""Do we call a new lesion where the radiologist called one?

A lesion at visit t is NEW if none of its voxels land within `slack` mm of any lesion at t-1,
once both visits have been put in the same frame by Piece 2.

The radiologist's notes distinguish three kinds of new lesion, and they are not equally fair
tests of an enhancing-mask detector:
  "new target L."        a new MEASURABLE enhancing lesion -- we should certainly see this
  "new non-target L."    new, but not measured; may be small, may be non-enhancing
  "new non-measurable L." new and explicitly too small to measure
"""
import csv, json, re, sys
sys.path.insert(0,'src')
import numpy as np
from rano.adapters.lumiere.zip_ref import ZipSource
from rano.adapters.lumiere import paths
from rano.registration import RigidFit, dilate, resample_like_fixed

FLOOR = 20
src = ZipSource('Imaging-v202211.zip')
reg = json.loads(open('output/registration/transforms.json').read())

rows = list(csv.DictReader(open('LUMIERE-ExpertRating-v202211.csv')))
RK = [k for k in rows[0] if k.startswith('Rating (')][0]
RR = [k for k in rows[0] if k.startswith('Rating rationale')][0]
note = {(r['Patient'], r['Date']): (r[RK].strip(), r[RR].strip()) for r in rows}

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

def kind(txt):
    t = txt.lower()
    if 'new target l' in t: return 'new target'
    if 'new non-measurable l' in t: return 'new non-measurable'
    if 'new non-target l' in t: return 'new non-target'
    if re.search(r'new .*l\.', t): return 'new (other)'
    return ''

def detect(slack):
    """-> list of (patient, visit, rating, expert_kind, our_new_sizes)"""
    out = []
    for pat, rec in reg["patients"].items():
        prev = None
        for tp, v in rec["visits"].items():
            fit = RigidFit.from_dict(v)
            enh = np.asarray(src.open_nifti(paths.dbt_mask(pat, tp)).dataobj) == 1
            cur = resample_like_fixed(enh.astype(np.uint8), fit, labels=True) > 0
            if prev is not None:
                near = dilate(prev, slack) if slack else prev
                fresh = [len(c) for c in components(cur)
                         if not near[c[:,0], c[:,1], c[:,2]].any()]
                rating, txt = note.get((pat, tp), ('',''))
                out.append((pat, tp, rating, kind(txt), sorted(fresh, reverse=True), txt))
            prev = cur
    return out


# Everything below is the experiment itself. Guarded so other experiments can import
# the helpers above without re-running this whole analysis as a side effect.
if __name__ == "__main__":
    for slack in (3,):
        res = detect(slack)
        print(f"slack {slack} mm, floor {FLOOR} mm3 — {len(res)} visit transitions\n")
        print(f"  {'patient':<13} {'visit':<12} {'rating':<8} {'expert':<19} {'we found':<22} ")
        print("  " + "-"*86)
        for pat, tp, rating, k, fresh, txt in res:
            ours = ', '.join(f'{s}' for s in fresh[:4]) if fresh else '-'
            mark = ''
            if k and not fresh: mark = '   MISS'
            if not k and fresh: mark = '   extra'
            print(f"  {pat:<13} {tp:<12} {rating:<8} {k or '-':<19} {ours:<22}{mark}")

        # ---- summary ----
        from collections import Counter
        print()
        real = [r for r in res if r[2] not in ('Post-Op', 'Pre-Op', '')]
        print(f"Excluding Pre-Op/Post-Op visits (surgery changes everything, and RANO does not "
              f"score them): {len(real)} of {len(res)} transitions.\n")

        def tab(rowset, label):
            hit  = sum(1 for r in rowset if r[3] and r[4])
            miss = sum(1 for r in rowset if r[3] and not r[4])
            extra= sum(1 for r in rowset if not r[3] and r[4])
            quiet= sum(1 for r in rowset if not r[3] and not r[4])
            n_exp = hit + miss
            print(f"  {label}")
            print(f"    expert said NEW, we found one     {hit:>3}")
            print(f"    expert said NEW, we found nothing {miss:>3}   <- misses")
            print(f"    expert silent, we found one       {extra:>3}   <- extras")
            print(f"    expert silent, we found nothing   {quiet:>3}")
            if n_exp:
                print(f"    caught {hit}/{n_exp} = {100*hit/n_exp:.0f}% of the lesions they flagged")
            print()
        tab(real, "ALL scorable visits")
        for k in ('new target', 'new non-target', 'new non-measurable', 'new (other)'):
            sub = [r for r in real if r[3] == k]
            if sub:
                hit = sum(1 for r in sub if r[4])
                print(f"  expert wrote '{k}': caught {hit}/{len(sub)}")
        print()
        per = Counter()
        for r in real:
            if r[3]: per[(r[0], 'flagged')] += 1
            if r[3] and r[4]: per[(r[0], 'caught')] += 1
        print("  by patient (of the lesions the expert flagged):")
        for pat in reg["patients"]:
            f = per[(pat,'flagged')]
            if f: print(f"    {pat:<14} caught {per[(pat,'caught')]}/{f}")
