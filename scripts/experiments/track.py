"""Follow lesions from visit to visit by asking one question: do they overlap?"""
import csv, re, sys
sys.path.insert(0,'src')
import numpy as np
from rano.adapters.lumiere.zip_ref import ZipSource
from rano.adapters.lumiere import paths

FLOOR = 20
FWD26 = tuple((dx,dy,dz) for dx in (0,1) for dy in (-1,0,1) for dz in (-1,0,1) if (dx,dy,dz)>(0,0,0))
def _sl(d): return (slice(None),slice(None)) if d==0 else ((slice(0,-d),slice(d,None)) if d>0 else (slice(-d,None),slice(0,d)))

def label(mask):
    """Returns an int array: 0 = background, 1..n = lesions, biggest first."""
    n = int(mask.sum())
    out = np.zeros(mask.shape, np.int32)
    if n == 0: return out, []
    idx = np.full(mask.shape, -1, np.int64); idx[mask] = np.arange(n); par = list(range(n))
    def find(x):
        while par[x]!=x: par[x]=par[par[x]]; x=par[x]
        return x
    for off in FWD26:
        s=[_sl(d) for d in off]; a,b = idx[tuple(x[0] for x in s)], idx[tuple(x[1] for x in s)]
        m=(a>=0)&(b>=0)
        for u,v in zip(a[m].tolist(), b[m].tolist()):
            ru,rv = find(u), find(v)
            if ru!=rv: par[max(ru,rv)] = min(ru,rv)
    roots = np.fromiter((find(i) for i in range(n)), np.int64, n)
    coords = np.argwhere(mask)
    groups = [(int((roots==r).sum()), coords[roots==r]) for r in np.unique(roots)]
    groups = [g for g in groups if g[0] >= FLOOR]
    groups.sort(key=lambda g: -g[0])
    sizes = []
    for k,(sz,sel) in enumerate(groups, start=1):
        out[sel[:,0], sel[:,1], sel[:,2]] = k
        sizes.append(sz)
    return out, sizes


# Everything below is the experiment itself. Guarded so other experiments can import
# the helpers above without re-running this whole analysis as a side effect.
if __name__ == "__main__":
    PAT = sys.argv[1] if len(sys.argv)>1 else 'Patient-072'
    src = ZipSource('Imaging-v202211.zip')
    tps = sorted({m.group(1) for m in (re.match(rf'Imaging/{PAT}/([^/]+)/DeepBraTumIA', n) for n in src.names) if m})
    tps = [t for t in tps if src.exists(paths.dbt_mask(PAT, t))]

    rows = list(csv.DictReader(open('LUMIERE-ExpertRating-v202211.csv')))
    rr = [k for k in rows[0] if k.startswith('Rating rationale')][0]
    rk = [k for k in rows[0] if k.startswith('Rating (')][0]
    note = {r['Date']: (r[rk].strip(), r[rr].strip()) for r in rows if r['Patient']==PAT}

    prev_lab, prev_ids = None, {}
    next_id = 1
    history = []
    for tp in tps:
        lab, sizes = label(np.asarray(src.open_nifti(paths.dbt_mask(PAT, tp)).dataobj) == 1)
        n = len(sizes)

        # how much does each lesion here overlap each lesion from last visit?
        ov = {}
        if prev_lab is not None:
            for k in range(1, n+1):
                vals, cnts = np.unique(prev_lab[lab == k], return_counts=True)
                for v, c in zip(vals.tolist(), cnts.tolist()):
                    if v > 0: ov[(k, v)] = c

        # each lesion's best parent, and each parent's best heir
        best_parent = {}
        for k in range(1, n+1):
            cands = [(c, v) for (kk, v), c in ov.items() if kk == k]
            if cands: best_parent[k] = max(cands)[1]
        heir = {}
        for v in set(best_parent.values()):
            heir[v] = max((ov[(k, v)], k) for k in best_parent if best_parent[k] == v)[1]

        ids, tag = {}, {}
        for k in range(1, n+1):
            par = best_parent.get(k)
            if par is not None and heir[par] == k:
                ids[k] = prev_ids[par]; tag[k] = ''
            else:
                ids[k] = next_id
                tag[k] = 'S' if par is not None else 'N'   # S = split off something, N = genuinely new
                next_id += 1
        history.append((tp, {ids[k]: sizes[k-1] for k in range(1, n+1)},
                            {ids[k]: tag[k] for k in range(1, n+1)}))
        prev_lab, prev_ids = lab, ids

    print(f"{PAT}: every lesion the tracker calls NEW (overlaps nothing at the previous visit)\n")
    print(f"  {'visit':<12} {'new lesions':>12} {'their sizes mm3':>26}   expert note")
    for tp, vols, tags in history:
        nw = sorted((v for i2,v in vols.items() if tags.get(i2)=='N'), reverse=True)
        sp = sum(1 for i2 in vols if tags.get(i2)=='S')
        rating, txt = note.get(tp, ('',''))
        said = 'new' in txt.lower()
        print(f"  {tp:<12} {len(nw):>12} {str(nw[:6]):>26}   {'[expert says NEW] ' if said else ''}{txt[:44]}")
    print()

    ever = {}
    for _, vols, _ in history:
        for i2, v in vols.items(): ever[i2] = max(ever.get(i2,0), v)
    show = [i2 for i2,v in sorted(ever.items(), key=lambda kv: -kv[1]) if v >= 300][:7]

    print(f"{PAT}: each column is one tracked lesion, cells are volume in mm3")
    print("N = genuinely new (overlaps nothing before)   S = split off an existing lesion\n")
    hdr = "  " + f"{'visit':<12}" + "".join(f"{'#'+str(i2):>10}" for i2 in show) + "   expert"
    print(hdr); print("  " + "-"*(len(hdr)+30))
    for tp, vols, tags in history:
        cells = ""
        for i2 in show:
            v = vols.get(i2)
            cells += f"{(str(v)+tags.get(i2,'')) if v else chr(183):>10}"
        rating, txt = note.get(tp, ('',''))
        print(f"  {tp:<12}{cells}   {rating:<8} {txt[:52]}")
