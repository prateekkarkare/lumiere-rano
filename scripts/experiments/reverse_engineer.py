"""Compare expert mm x mm measurements against what the mask can support, ring cases first."""
import csv, json, re, sys
sys.path.insert(0, 'src')
from collections import deque
import numpy as np
from rano.adapters.lumiere.zip_ref import ZipSource
from rano.adapters.lumiere import paths

FLOOR = 20
MEAS = re.compile(r'(\d+(?:\.\d+)?)\s*(?:mm)?\s*[x×X]\s*(\d+(?:\.\d+)?)\s*(?:mm)?')

# ---------- geometry ----------
def hull(points):
    """Monotone-chain convex hull of integer points."""
    pts = sorted(set(map(tuple, points)))
    if len(pts) <= 2: return pts
    def half(ps):
        out = []
        for p in ps:
            while len(out) >= 2:
                (ax,ay),(bx,by) = out[-2], out[-1]
                if (bx-ax)*(p[1]-ay) - (by-ay)*(p[0]-ax) <= 0: out.pop()
                else: break
            out.append(p)
        return out
    return half(pts)[:-1] + half(pts[::-1])[:-1]

def caliper_mm(mask2d):
    """Longest straight line the shape can support, outer edge to outer edge.
    Voxels are unit squares, so we hull their CORNERS, not their centres."""
    ij = np.argwhere(mask2d)
    if len(ij) == 0: return 0.0
    corners = np.concatenate([ij, ij + [1,0], ij + [0,1], ij + [1,1]])
    h = np.array(hull(corners), dtype=float)
    if len(h) < 2: return 0.0
    d = np.sum((h[:,None,:] - h[None,:,:])**2, axis=2)
    return float(np.sqrt(d.max()))

def boundary(mask2d):
    m = mask2d
    b = np.zeros_like(m)
    b[:-1,:] |= m[:-1,:] & ~m[1:,:];  b[1:,:]  |= m[1:,:]  & ~m[:-1,:]
    b[:,:-1] |= m[:,:-1] & ~m[:,1:];  b[:,1:]  |= m[:,1:]  & ~m[:,:-1]
    b[0,:] |= m[0,:]; b[-1,:] |= m[-1,:]; b[:,0] |= m[:,0]; b[:,-1] |= m[:,-1]
    return np.argwhere(b)

def inside(mask2d, p, q):
    """Does the straight segment p->q stay on the shape the whole way?"""
    n = int(max(abs(q[0]-p[0]), abs(q[1]-p[1])) * 2) + 1
    t = np.linspace(0, 1, n)
    xs = np.rint(p[0] + t*(q[0]-p[0])).astype(int)
    ys = np.rint(p[1] + t*(q[1]-p[1])).astype(int)
    return bool(mask2d[xs, ys].all())

def constrained_mm(mask2d):
    """Longest chord that never leaves the shape. Voxel centres, so ~1mm short of
    the edge-to-edge caliper -- fine, we are looking for big differences here."""
    b = boundary(mask2d)
    if len(b) < 2: return 0.0
    d2 = np.sum((b[:,None,:] - b[None,:,:])**2, axis=2)
    iu = np.triu_indices(len(b), 1)
    order = np.argsort(d2[iu])[::-1]
    for k in order:                      # longest first; stop at the first legal one
        i, j = iu[0][k], iu[1][k]
        if inside(mask2d, b[i], b[j]):
            return float(np.sqrt(d2[i,j]))
    return 0.0

def fill_holes(mask2d):
    """Flood the background inward from the border; anything it can't reach is enclosed."""
    H, W = mask2d.shape
    free = ~mask2d
    seen = np.zeros_like(free)
    q = deque()
    for i in range(H):
        for j in (0, W-1):
            if free[i,j] and not seen[i,j]: seen[i,j]=True; q.append((i,j))
    for j in range(W):
        for i in (0, H-1):
            if free[i,j] and not seen[i,j]: seen[i,j]=True; q.append((i,j))
    while q:
        i,j = q.popleft()
        for di,dj in ((1,0),(-1,0),(0,1),(0,-1)):
            a,b = i+di, j+dj
            if 0<=a<H and 0<=b<W and free[a,b] and not seen[a,b]:
                seen[a,b]=True; q.append((a,b))
    holes = free & ~seen
    return mask2d | holes, int(holes.sum())

# ---------- 3D components ----------
FWD26 = tuple((dx,dy,dz) for dx in (0,1) for dy in (-1,0,1) for dz in (-1,0,1) if (dx,dy,dz)>(0,0,0))
def _sl(d): return (slice(None),slice(None)) if d==0 else ((slice(0,-d),slice(d,None)) if d>0 else (slice(-d,None),slice(0,d)))
def components(mask):
    n = int(mask.sum())
    if n == 0: return []
    idx = np.full(mask.shape, -1, np.int64); idx[mask] = np.arange(n)
    par = list(range(n))
    def find(x):
        while par[x]!=x: par[x]=par[par[x]]; x=par[x]
        return x
    for off in FWD26:
        s = [_sl(d) for d in off]
        a, b = idx[tuple(x[0] for x in s)], idx[tuple(x[1] for x in s)]
        m = (a>=0)&(b>=0)
        for u,v in zip(a[m].tolist(), b[m].tolist()):
            ru,rv = find(u),find(v)
            if ru!=rv: par[max(ru,rv)] = min(ru,rv)
    roots = np.fromiter((find(i) for i in range(n)), np.int64, n)
    coords = np.argwhere(mask)
    out = []
    for r in np.unique(roots):
        sel = coords[roots==r]
        if len(sel) >= FLOOR: out.append(sel)
    out.sort(key=len, reverse=True)
    return out

# ---------- run ----------
practice = [p["patient_id"] for p in json.load(open('output/cohort/cohort_lock.json'))['practice']['patients']]
rows = list(csv.DictReader(open('LUMIERE-ExpertRating-v202211.csv')))
rr = [k for k in rows[0] if k.startswith('Rating rationale')][0]
src = ZipSource('Imaging-v202211.zip')


# Everything below is the experiment itself. Guarded so other experiments can import
# the helpers above without re-running this whole analysis as a side effect.
if __name__ == "__main__":
    print(f"{'patient':<13} {'timept':<11} {'expert':>12} | {'les':>4} {'vol':>7} "
          f"{'caliper':>8} {'inside':>7} {'in-fill':>8} {'hole%':>6}")
    print("-"*95)
    for r in rows:
        if r['Patient'] not in practice: continue
        ms = MEAS.findall(r[rr] or '')
        if not ms: continue
        p, tp = r['Patient'], r['Date']
        if not src.exists(paths.dbt_mask(p, tp)):
            print(f"{p:<13} {tp:<11} {' '.join(f'{a}x{b}' for a,b in ms):>12} |   (no segmentation in the archive)\n")
            continue
        enh = np.asarray(src.open_nifti(paths.dbt_mask(p, tp)).dataobj) == 1
        comps = components(enh)
        exp = " ".join(f"{a}x{b}" for a,b in ms)
        for k, sel in enumerate(comps[:3]):
            vol = len(sel)
            cal = con = conf = 0.0; hole_vox = 0; tot_vox = 0
            for z in np.unique(sel[:,2]):
                sl = np.zeros(enh.shape[:2], bool)
                pts = sel[sel[:,2]==z]
                sl[pts[:,0], pts[:,1]] = True
                filled, nh = fill_holes(sl)
                hole_vox += nh; tot_vox += int(sl.sum())
                cal  = max(cal,  caliper_mm(sl))
                con  = max(con,  constrained_mm(sl))
                conf = max(conf, constrained_mm(filled))
            hp = 100*hole_vox/max(tot_vox,1)
            lab = exp if k==0 else ""
            pn  = p if k==0 else ""
            tn  = tp if k==0 else ""
            print(f"{pn:<13} {tn:<11} {lab:>12} | {'L'+str(k+1):>4} {vol:>7} "
                  f"{cal:>8.1f} {con:>7.1f} {conf:>8.1f} {hp:>5.1f}%")
        print()
