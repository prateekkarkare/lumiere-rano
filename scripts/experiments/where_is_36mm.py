"""Patient-072 week-090: the expert says 13x11. We say 36mm. Where does 36mm come from?"""
import sys; sys.path.insert(0,'src')
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from reverse_engineer import components, caliper_mm, constrained_mm, src, paths

P, TP = 'Patient-072', 'week-090'
a = np.asarray(src.open_nifti(paths.dbt_mask(P, TP)).dataobj)
enh = a == 1
comps = components(enh)
sel = comps[0]
big = np.zeros(enh.shape, bool); big[sel[:,0], sel[:,1], sel[:,2]] = True
print(f"{P} {TP}: largest enhancing lesion = {len(sel)} mm3")
print(f"bounding box (mm): x {np.ptp(sel[:,0])+1}, y {np.ptp(sel[:,1])+1}, z {np.ptp(sel[:,2])+1}")

print(f"\nper-slice: how wide is it on each slice it occupies?")
print(f"  {'z':>4} {'voxels':>7} {'caliper mm':>11} {'pieces':>7}")
NB8 = [(dx,dy) for dx in (-1,0,1) for dy in (-1,0,1) if (dx,dy)!=(0,0)]
def pieces2d(m):
    from collections import deque
    seen = np.zeros_like(m); n=0
    for p0 in map(tuple, np.argwhere(m)):
        if seen[p0]: continue
        n+=1; q=deque([p0]); seen[p0]=True
        while q:
            i,j=q.popleft()
            for di,dj in NB8:
                x,y=i+di,j+dj
                if 0<=x<m.shape[0] and 0<=y<m.shape[1] and m[x,y] and not seen[x,y]:
                    seen[x,y]=True; q.append((x,y))
    return n
for z in np.unique(sel[:,2]):
    m = big[:,:,z]
    print(f"  {z:>4} {int(m.sum()):>7} {caliper_mm(m):>11.1f} {pieces2d(m):>7}")

# does a 1mm erosion split it? (the thin-bridge test)
def erode(mask):
    out = mask.copy()
    for ax in (0,1,2):
        out &= np.roll(mask,1,ax) & np.roll(mask,-1,ax)
    return out
er = erode(big)
print(f"\nthin-bridge test: erode by 1 mm -> {int(er.sum())} voxels remain")
sub = components(er)
print(f"  pieces of 20mm3 or more after eroding: {len(sub)}  sizes {[len(s) for s in sub][:8]}")
