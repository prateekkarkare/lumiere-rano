"""Throwaway diagnostic: are the slice-84 patches one 3D lesion or several?
Not the real implementation -- a scratch BFS to answer one question with data."""
import sys; sys.path.insert(0, 'src')
from collections import deque
import numpy as np
from rano.adapters.lumiere.zip_ref import ZipSource
from rano.adapters.lumiere import paths

P, TP, Z = 'Patient-067', 'week-109', 84
src = ZipSource('Imaging-v202211.zip')
m = np.asarray(src.open_nifti(paths.dbt_mask(P, TP)).dataobj).astype(np.uint8)
enh = (m == 1)
print(f"{P} {TP}: enhancing voxels = {enh.sum()}")

# --- 26-connected components in 3D, BFS over the ~900 foreground voxels ---
NB26 = [(dx,dy,dz) for dx in (-1,0,1) for dy in (-1,0,1) for dz in (-1,0,1)
        if (dx,dy,dz) != (0,0,0)]
def label3d(mask, nbrs):
    coords = {tuple(c): -1 for c in np.argwhere(mask)}
    lab, sizes = 0, []
    for c0 in list(coords):
        if coords[c0] != -1: continue
        lab += 1; n = 0; q = deque([c0]); coords[c0] = lab
        while q:
            x,y,z = q.popleft(); n += 1
            for dx,dy,dz in nbrs:
                c = (x+dx, y+dy, z+dz)
                if coords.get(c, 0) == -1:
                    coords[c] = lab; q.append(c)
        sizes.append(n)
    return coords, sizes

coords, sizes = label3d(enh, NB26)
order = np.argsort(sizes)[::-1]
print(f"\n26-connectivity, 3D: {len(sizes)} components")
print("  sizes (mm^3, desc):", [sizes[i] for i in order][:15],
      "..." if len(sizes) > 15 else "")

# --- what does slice Z look like on its own? ---
NB8 = [(dx,dy,0) for dx in (-1,0,1) for dy in (-1,0,1) if (dx,dy) != (0,0)]
sl = np.zeros_like(enh); sl[:,:,Z] = enh[:,:,Z]
_, sl_sizes = label3d(sl, NB8)
print(f"\nslice z={Z} alone: {len(sl_sizes)} separate 2D pieces, sizes {sorted(sl_sizes, reverse=True)}")

# which 3D component does each 2D piece belong to?
sl_coords, _ = label3d(sl, NB8)
byslpiece = {}
for c, p2 in sl_coords.items():
    byslpiece.setdefault(p2, set()).add(coords[c])
print("  2D piece -> 3D component id(s):")
for p2 in sorted(byslpiece, key=lambda k: -sum(1 for c,v in sl_coords.items() if v==k)):
    n = sum(1 for c,v in sl_coords.items() if v == p2)
    print(f"    piece {p2} ({n:>3} vox) -> 3D component {sorted(byslpiece[p2])}")

# --- how common is this? largest component, pieces per slice ---
big = order[0] + 1
bigmask = np.zeros_like(enh)
for c, l in coords.items():
    if l == big: bigmask[c] = True
zs = np.flatnonzero(bigmask.any(axis=(0,1)))
print(f"\nlargest 3D component: {sizes[order[0]]} mm^3, spans z={zs.min()}..{zs.max()} ({len(zs)} slices)")
multi = []
for z in zs:
    s = np.zeros_like(enh); s[:,:,z] = bigmask[:,:,z]
    _, ss = label3d(s, NB8)
    if len(ss) > 1: multi.append((int(z), len(ss)))
print(f"  slices where this ONE lesion looks like >1 piece: {len(multi)} of {len(zs)}")
print("  ", multi)

# --- 6- vs 26-connectivity, same volume ---
NB6 = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
_, s6 = label3d(enh, NB6)
print(f"\nsame mask under 6-connectivity: {len(s6)} components (vs {len(sizes)} under 26)")
print("  6-conn sizes (desc):", sorted(s6, reverse=True)[:15])
