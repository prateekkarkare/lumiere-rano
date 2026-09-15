"""Two pictures behind docs/FINDINGS_2026-09-14.md, sections 4 and 5. Writes docs/figures/*.png.

Patient-070: a lesion both segmenters report as 0 mm3. Patient-090: what T2 progression looks like.
Both are unassigned patients (not in the locked cohort). Visits are rigidly registered onto one
anchor visit with rano.registration, and each fit is outcome-checked (brain overlap printed).
"""
import sys; sys.path[:0]=["src"]
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rano.adapters.lumiere import paths
from rano.adapters.lumiere.zip_ref import ZipSource
from rano.contract.case import Modality as M
from rano.registration.rigid import fit_rigid, registration_mask, resample_like_fixed
OUT="docs/figures"; src=ZipSource("Imaging-v202211.zip")
F=ImageFont.load_default(size=22); Fs=ImageFont.load_default(size=17)
COL={1:(255,50,50),2:(255,215,0),3:(0,170,255)}
ss=lambda P,tp,mod: np.asarray(src.open_nifti(paths.dbt_skull_strip(P,tp,mod)).dataobj).astype(np.float32)
seg=lambda P,tp: np.asarray(src.open_nifti(paths.dbt_mask(P,tp)).dataobj)
brain=lambda P,tp: np.asarray(src.open_nifti(paths.dbt_brain_mask(P,tp)).dataobj)>0
dice=lambda a,b: 2*(a&b).sum()/(a.sum()+b.sum())

def aligned(P, anchor, tp):
    """Everything for visit tp, rigidly moved onto the anchor visit. QC printed: did brains overlap better?"""
    if tp==anchor:
        return {M.CT1:ss(P,tp,M.CT1), M.FLAIR:ss(P,tp,M.FLAIR), "seg":seg(P,tp)}
    fb,mb=brain(P,anchor),brain(P,tp)
    fit=fit_rigid(ss(P,anchor,M.CT1), ss(P,tp,M.CT1), registration_mask(fb,seg(P,anchor)>0), registration_mask(mb,seg(P,tp)>0))
    after=resample_like_fixed(mb,fit)>0
    print(f"  {P} {tp} -> {anchor}: brain overlap {dice(fb,mb):.3f} -> {dice(fb,after):.3f}  "
          f"shift {tuple(round(v,1) for v in fit.translation_mm)} mm  rot {tuple(round(v,1) for v in fit.rotation_degrees)} deg")
    return {M.CT1:resample_like_fixed(ss(P,tp,M.CT1),fit,labels=False), M.FLAIR:resample_like_fixed(ss(P,tp,M.FLAIR),fit,labels=False),
            "seg":resample_like_fixed(seg(P,tp),fit)}

def panel(a,m,sl,K,labels=(1,2,3)):
    lo,hi=np.percentile(a[a>0],[1,99.7]); g=np.rot90(np.clip((a[sl]-lo)/(hi-lo),0,1)*255).astype(np.uint8)
    rgb=np.repeat(np.kron(g,np.ones((K,K),np.uint8))[...,None],3,2)
    if m is not None:
        mm=np.kron(np.rot90(m[sl]),np.ones((K,K),int))
        for lab in labels:
            b=mm==lab; e=b&~(np.roll(b,1,0)&np.roll(b,-1,0)&np.roll(b,1,1)&np.roll(b,-1,1)); e=e|np.roll(e,1,0)|np.roll(e,1,1); rgb[e]=COL[lab]
    return Image.fromarray(rgb)

def grid(cells,header,caps,footer,out):
    w,h=cells[0][0].size; top=56; capH=60; foot=26*len(footer.split("\n"))+20
    cv=Image.new("RGB",(w*len(cells[0]),top+(h+capH)*len(cells)+foot),(12,12,12)); d=ImageDraw.Draw(cv)
    d.text((12,16),header,font=F,fill="white")
    for r,row in enumerate(cells):
        y=top+r*(h+capH)
        for c,im in enumerate(row):
            cv.paste(im,(c*w,y+capH))
            for i,line in enumerate(caps[r][c].split("\n")): d.text((c*w+10,y+6+i*23),line,font=Fs,fill=(235,235,235))
    for i,line in enumerate(footer.split("\n")): d.text((12,cv.height-foot+10+i*24),line,font=Fs,fill=(200,200,200))
    cv.save(out); print(out,cv.size)

# ---------------- Patient-070
P="Patient-070"; A="week-044"
V={tp:aligned(P,A,tp) for tp in ["week-019","week-044","week-061"]}
x,y,z=110,100,68; R=26; K=7; cells=[]; caps=[]
for view,sl in [("axial",(slice(x-R,x+R),slice(y-R,y+R),z)),("coronal",(slice(x-R,x+R),y,slice(z-R,z+R)))]:
    row=[];cr=[]
    for tp,cap in [("week-019","week 19\nnothing here"),("week-044","week 44: bright nodule\nmask: 0 mm3 (both tools)"),("week-061","week 61: 28 x 22 mm ring\nmask finds it (red)")]:
        im=panel(V[tp][M.CT1],V[tp]["seg"],sl,K,labels=(1,2)); ImageDraw.Draw(im).text((8,8),view,font=Fs,fill=(255,255,0)); row.append(im); cr.append(cap)
    cells.append(row); caps.append(cr)
grid(cells,"Patient-070  |  contrast T1  |  the same spot, three visits (registered)",caps,
     "red = mask 'enhancing', yellow = mask 'necrosis'. At week 44 the doctor wrote 'target L. 12mm x 10mm'.\nDeepBraTumIA and HD-GLIO-AUTO both report 0 mm3 of enhancing tissue on that scan.",
     f"{OUT}/patient070_missed_lesion.png")

# ---------------- Patient-090
P="Patient-090"; A="week-000-2"
V={tp:aligned(P,A,tp) for tp in [A,"week-029"]}
cells=[];caps=[]; K=3
for name,mod,zz,draw in [("contrast T1",M.CT1,60,False),("FLAIR",M.FLAIR,60,True),("FLAIR, higher up",M.FLAIR,84,True)]:
    row=[];cr=[]
    for tp,wk in [(A,"week 0 (reference)"),("week-029","week 29")]:
        row.append(panel(V[tp][mod],V[tp]["seg"] if draw else None,(slice(None),slice(None),zz),K)); cr.append(f"{wk}  |  {name}")
    cells.append(row); caps.append(cr)
grid(cells,"Patient-090  |  the ring fades, the FLAIR-bright region spreads (registered)",caps,
     "blue = mask 'edema' (all FLAIR-bright tissue)   yellow = 'necrosis / non-enhancing'   red = 'enhancing'\nenhancing 2,728 -> 84 mm3    FLAIR-bright 54,229 -> 92,557 mm3    necrosis/non-enhancing 278 -> 6,113 mm3",
     f"{OUT}/patient090_t2_progression.png")

# ---------------- Patient-070: what each segmenter reports, visit by visit
P = "Patient-070"
for tp in ["week-019", "week-044", "week-061"]:
    m = seg(P, tp); h = src.open_nifti(paths.hdglio_mask(P, tp))
    hv = float(np.prod(h.header.get_zooms()[:3])); ha = np.asarray(h.dataobj)
    print(f"{P} {tp}: DeepBraTumIA enhancing {(m == 1).sum():>6} mm3 | HD-GLIO-AUTO enhancing {int((ha == 2).sum() * hv):>6} mm3")
