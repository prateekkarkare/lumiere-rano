"""Track one patient's lesions with and without registration, against the radiologist's notes.

Before Piece 2, this tracker found ZERO new lesions at all eight Patient-072 visits where the
radiologist wrote "new lesion". That was the whole reason for building registration. This runs
the same tracker on the same data twice -- raw atlas space, and after applying each visit's
rigid fit -- so the difference is attributable to the registration and nothing else.
"""
import csv, json, re, sys
sys.path.insert(0, 'src')
from collections import deque
import numpy as np
from rano.adapters.lumiere.zip_ref import ZipSource
from rano.adapters.lumiere import paths
from rano.registration import RigidFit, dilate, resample_like_fixed

FLOOR = 20
FWD26 = tuple((dx,dy,dz) for dx in (0,1) for dy in (-1,0,1) for dz in (-1,0,1) if (dx,dy,dz)>(0,0,0))
def _sl(d): return (slice(None),slice(None)) if d==0 else ((slice(0,-d),slice(d,None)) if d>0 else (slice(-d,None),slice(0,d)))

def label(mask):
    """0 = background, 1..n = lesions above the floor, biggest first."""
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

def track(masks, slack_mm=0):
    """Walk the visits; a lesion overlapping nothing at the previous visit is NEW."""
    hist, prev_lab, prev_ids, nxt = [], None, {}, 1
    for tp, m in masks:
        lab, sizes = label(m)
        n = len(sizes)
        near = dilate(prev_lab > 0, slack_mm) if (prev_lab is not None and slack_mm) else None
        ov = {}
        if prev_lab is not None:
            for k in range(1, n+1):
                here = lab == k
                if near is not None:
                    here = here & near
                    # a slack match still has to name WHICH old lesion, so look it up nearby
                    src_lab = prev_lab
                    for r in range(slack_mm):
                        pass
                vals, cnts = np.unique(prev_lab[(lab == k)], return_counts=True)
                for v, c in zip(vals.tolist(), cnts.tolist()):
                    if v > 0: ov[(k, v)] = c
                if near is not None and not any(kk == k for kk, _ in ov):
                    # nothing overlapped exactly -- allow a near miss within the slack
                    vals, cnts = np.unique(prev_lab[dilate(lab == k, slack_mm)], return_counts=True)
                    for v, c in zip(vals.tolist(), cnts.tolist()):
                        if v > 0: ov[(k, v)] = c
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
                ids[k], tag[k] = prev_ids[par], ''
            else:
                ids[k], tag[k] = nxt, ('S' if par is not None else 'N'); nxt += 1
        hist.append((tp, [(sizes[k-1], tag[k]) for k in range(1, n+1)]))
        prev_lab, prev_ids = lab, ids
    return hist


# Everything below is the experiment itself. Guarded so other experiments can import
# the helpers above without re-running this whole analysis as a side effect.
if __name__ == "__main__":
    PAT = sys.argv[1] if len(sys.argv) > 1 else 'Patient-072'
    SLACK = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    src = ZipSource('Imaging-v202211.zip')
    reg = json.loads(open('output/registration/transforms.json').read())['patients'][PAT]

    rows = list(csv.DictReader(open('LUMIERE-ExpertRating-v202211.csv')))
    rr = [k for k in rows[0] if k.startswith('Rating rationale')][0]
    rk = [k for k in rows[0] if k.startswith('Rating (')][0]
    note = {r['Date']: (r[rk].strip(), r[rr].strip()) for r in rows if r['Patient'] == PAT}

    tps = list(reg['visits'].keys())
    raw, aligned = [], []
    for tp in tps:
        m = np.asarray(src.open_nifti(paths.dbt_mask(PAT, tp)).dataobj) == 1
        raw.append((tp, m))
        fit = RigidFit.from_dict(reg['visits'][tp])
        aligned.append((tp, resample_like_fixed(m.astype(np.uint8), fit, labels=True) > 0))

    h_raw = track(raw, slack_mm=0)
    h_reg = track(aligned, slack_mm=SLACK)

    print(f"{PAT} — new lesions found, before and after registration "
          f"(26-conn, floor {FLOOR} mm3, {SLACK} mm slack after)\n")
    print(f"  {'visit':<12} {'raw':>22} {'registered':>22}   radiologist")
    agree = disagree_missed = disagree_extra = 0
    for (tp, a), (_, b) in zip(h_raw, h_reg):
        na = [s for s, t in a if t == 'N']
        nb = [s for s, t in b if t == 'N']
        rating, txt = note.get(tp, ('', ''))
        said_new = 'new' in txt.lower()
        if tp == tps[0]:
            mark = '(baseline)'
        else:
            if said_new and nb: agree += 1; mark = 'BOTH say new'
            elif said_new and not nb: disagree_missed += 1; mark = 'we MISS it'
            elif not said_new and nb: disagree_extra += 1; mark = 'we invent one'
            else: agree += 1; mark = 'both say none'
        fa = f"{len(na)}  {sorted(na, reverse=True)[:3]}" if na else "0"
        fb = f"{len(nb)}  {sorted(nb, reverse=True)[:3]}" if nb else "0"
        flag = 'NEW' if said_new else '   '
        print(f"  {tp:<12} {fa:>22} {fb:>22}   {flag}  {mark}")
    print(f"\n  agree {agree}   we miss a new lesion {disagree_missed}   "
          f"we invent one {disagree_extra}   (of {len(tps)-1} scored visits)")
