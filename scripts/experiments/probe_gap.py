"""Are the gaps between the enhancing pieces necrosis? i.e. is this one ring-lesion?"""
import sys; sys.path.insert(0,'src')
from collections import deque
import numpy as np
from rano.adapters.lumiere.zip_ref import ZipSource
from rano.adapters.lumiere import paths

P, TP, Z = 'Patient-067','week-109',84
m = np.asarray(ZipSource('Imaging-v202211.zip').open_nifti(paths.dbt_mask(P,TP)).dataobj).astype(np.uint8)
NB8=[(dx,dy,0) for dx in(-1,0,1) for dy in(-1,0,1) if (dx,dy)!=(0,0)]
NB26=[(dx,dy,dz) for dx in(-1,0,1) for dy in(-1,0,1) for dz in(-1,0,1) if (dx,dy,dz)!=(0,0,0)]

def label(mask, nbrs):
    coords={tuple(c):-1 for c in np.argwhere(mask)}; sizes=[]; lab=0
    for c0 in list(coords):
        if coords[c0]!=-1: continue
        lab+=1; n=0; q=deque([c0]); coords[c0]=lab
        while q:
            x,y,z=q.popleft(); n+=1
            for dx,dy,dz in nbrs:
                c=(x+dx,y+dy,z+dz)
                if coords.get(c,0)==-1: coords[c]=lab; q.append(c)
        sizes.append(n)
    return coords,sizes

for name, sel in [("enhancing only", m==1), ("enhancing + necrosis", (m==1)|(m==2))]:
    s=np.zeros_like(m,bool); s[:,:,Z]=sel[:,:,Z]
    _,sz=label(s,NB8)
    print(f"slice {Z}, {name:<22}: {len(sz):>2} 2D piece(s), sizes {sorted(sz,reverse=True)[:8]}")
    _,sz3=label(sel,NB26)
    print(f"   {'':<22}   3D 26-conn: {len(sz3)} components, top {sorted(sz3,reverse=True)[:6]}")

# how much of the "gap" is necrosis vs plain brain, on this slice
sl_e, sl_n = m[:,:,Z]==1, m[:,:,Z]==2
print(f"\nslice {Z}: enhancing {sl_e.sum()} vox, necrosis {sl_n.sum()} vox "
      f"-> the enhancing pieces are the rim of a {sl_e.sum()+sl_n.sum()}-vox lesion cross-section")

# effect of a volume floor on component COUNT, both connectivities
NB6=[(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
_,s26=label(m==1,NB26); _,s6=label(m==1,NB6)
print("\nsurviving component count vs volume floor (enhancing, whole volume):")
print(f"  {'floor mm3':>9} | {'26-conn':>7} | {'6-conn':>6}")
for f in (0,2,5,10,20,50,100):
    print(f"  {f:>9} | {sum(1 for v in s26 if v>=f):>7} | {sum(1 for v in s6 if v>=f):>6}")
