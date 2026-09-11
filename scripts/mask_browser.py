"""
Eyeball browser for LUMIERE DeepBraTumIA masks — a local, read-only slice viewer.

WHY THIS EXISTS
---------------
Before writing lesion-splitting code you should know what the masks actually look like: how
fragmented the enhancing compartment is, whether rims are continuous or broken, how often a
timepoint is multifocal. This serves that look. It computes nothing the pipeline depends on.

COHORT DISCIPLINE
-----------------
Serves the 7 PRACTICE patients only. Held-out patient ids are rejected at the request layer,
not merely hidden in the UI -- a browser is exactly the kind of tool that erodes a cohort lock
by accident. The practice list is read from output/cohort/cohort_lock.json at startup.

GEOMETRY
--------
Atlas space, MNI 1mm isotropic, orientation LAS (verified from the affine at startup):
    axis 0 -> x, index increases toward patient LEFT
    axis 1 -> y, index increases toward ANTERIOR
    axis 2 -> z, index increases toward SUPERIOR  <- axial slice index
An axial slice is ``vol[:, :, z]``. It is displayed transposed and flipped so anterior is at the
top of the image; columns run in +x order, which puts patient-left on the RIGHT of the screen --
NEUROLOGICAL convention. Upscaling is NEAREST-NEIGHBOUR on purpose: it preserves voxel edges, so
staircase artefacts from resampling stay visible instead of being smoothed into plausibility.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import threading
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rano.adapters.lumiere import paths  # noqa: E402
from rano.adapters.lumiere.zip_ref import ZipSource  # noqa: E402
from rano.contract.case import Modality  # noqa: E402
from rano.labels import LABEL_SCHEMA  # noqa: E402

DEFAULT_ZIP = ROOT / "Imaging-v202211.zip"
COHORT_LOCK = ROOT / "output" / "cohort" / "cohort_lock.json"

#: display colours, keyed by DeepBraTumIA integer label
LABEL_COLOURS: dict[int, tuple[int, int, int]] = {
    1: (255, 59, 48),    # enhancing            -- red
    2: (10, 132, 255),   # necrosis/nonenhancing -- blue
    3: (255, 214, 10),   # edema                -- yellow
}

_MOD_BY_NAME = {m.value.lower(): m for m in Modality}
_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")


_FWD26 = tuple((dx, dy, dz) for dx in (0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
               if (dx, dy, dz) > (0, 0, 0))


def _shift(d: int) -> tuple[slice, slice]:
    if d == 0:
        return slice(None), slice(None)
    return (slice(0, -d), slice(d, None)) if d > 0 else (slice(-d, None), slice(0, d))


def _label_components(mask: np.ndarray, floor: int = 20) -> np.ndarray:
    """26-connected components, numbered largest-first. Anything under ``floor`` becomes 0.

    Same rule as the locked lesion-splitting decision (docs/lesion_split_decision.html), so what
    you see coloured here is what the pipeline counts as separate lesions.
    """
    n = int(mask.sum())
    out = np.zeros(mask.shape, np.int32)
    if n == 0:
        return out
    idx = np.full(mask.shape, -1, np.int64)
    idx[mask] = np.arange(n)
    par = list(range(n))

    def find(x: int) -> int:
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    for off in _FWD26:
        sl = [_shift(d) for d in off]
        a = idx[tuple(s_[0] for s_ in sl)]
        b = idx[tuple(s_[1] for s_ in sl)]
        m = (a >= 0) & (b >= 0)
        for u, v in zip(a[m].tolist(), b[m].tolist()):
            ru, rv = find(u), find(v)
            if ru != rv:
                par[max(ru, rv)] = min(ru, rv)

    roots = np.fromiter((find(i) for i in range(n)), np.int64, n)
    coords = np.argwhere(mask)
    groups = [(int((roots == r).sum()), coords[roots == r]) for r in np.unique(roots)]
    groups = [g for g in groups if g[0] >= floor]
    groups.sort(key=lambda g: -g[0])
    for k, (_, sel) in enumerate(groups, start=1):
        out[sel[:, 0], sel[:, 1], sel[:, 2]] = k
    return out


def _growth_counts(cur: np.ndarray, old: np.ndarray) -> dict:
    return {
        "kept": int((cur & old).sum()),
        "gained": int((cur & ~old).sum()),
        "lost": int((old & ~cur).sum()),
    }


def _surface(vol: np.ndarray) -> np.ndarray:
    """Voxels with at least one face exposed to background."""
    if not vol.any():
        return vol
    inner = vol.copy()
    for ax in (0, 1, 2):
        inner &= np.roll(vol, 1, ax) & np.roll(vol, -1, ax)
    return vol & ~inner


class Store:
    """Zip-backed volume cache. ``ZipFile`` is not thread-safe, so every read holds ``_lock``."""

    def __init__(self, zip_path: Path, practice: dict[str, list[str]]) -> None:
        self.src = ZipSource(str(zip_path))
        self.practice = practice
        self._lock = threading.Lock()
        self._masks: OrderedDict[tuple[str, str], np.ndarray] = OrderedDict()
        self._bgs: OrderedDict[tuple[str, str, str], np.ndarray] = OrderedDict()
        self._stats: dict[tuple[str, str], dict] = {}

    def _check(self, patient: str, tp: str) -> None:
        if patient not in self.practice:
            raise PermissionError(
                f"{patient} is not in the practice arm. The cohort lock puts the held-out "
                f"patients off limits; this viewer will not open them."
            )
        if tp not in self.practice[patient]:
            raise KeyError(f"{patient} has no timepoint {tp}")

    @staticmethod
    def _evict(cache: OrderedDict, limit: int) -> None:
        while len(cache) > limit:
            cache.popitem(last=False)

    def mask(self, patient: str, tp: str) -> np.ndarray:
        self._check(patient, tp)
        key = (patient, tp)
        with self._lock:
            if key in self._masks:
                self._masks.move_to_end(key)
                return self._masks[key]
            img = self.src.open_nifti(paths.dbt_mask(patient, tp))
            arr = np.asarray(img.dataobj).astype(np.uint8)
            self._masks[key] = arr
            self._evict(self._masks, 6)
        return arr

    def background(self, patient: str, tp: str, mod_name: str) -> np.ndarray | None:
        """8-bit windowed anatomical background, or None when that sequence is absent."""
        self._check(patient, tp)
        key = (patient, tp, mod_name)
        with self._lock:
            if key in self._bgs:
                self._bgs.move_to_end(key)
                return self._bgs[key]
        mod = _MOD_BY_NAME.get(mod_name)
        if mod is None:
            return None
        member = paths.dbt_skull_strip(patient, tp, mod)
        with self._lock:
            if not self.src.exists(member):
                return None
            vol = np.asarray(self.src.open_nifti(member).dataobj, dtype=np.float32)
        # window on non-zero voxels only: the skull-stripped background is mostly exact zeros,
        # and including them drags the low percentile to 0 and washes the brain out.
        brain = vol[vol > 0]
        if brain.size == 0:
            lo, hi = 0.0, 1.0
        else:
            lo, hi = np.percentile(brain, [1.0, 99.0])
            if hi <= lo:
                lo, hi = float(brain.min()), float(brain.max()) or 1.0
        scaled = np.clip((vol - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        out = (scaled * 255).astype(np.uint8)
        with self._lock:
            self._bgs[key] = out
            self._evict(self._bgs, 6)
        return out

    def stats(self, patient: str, tp: str) -> dict:
        """Per-slice voxel counts per label, plus totals. Volumes are counts x 1mm^3."""
        self._check(patient, tp)
        key = (patient, tp)
        if key in self._stats:
            return self._stats[key]
        m = self.mask(patient, tp)
        per_slice = {
            str(lab): np.count_nonzero(m == lab, axis=(0, 1)).astype(int).tolist()
            for lab in LABEL_COLOURS
        }
        totals = {lab: int(sum(v)) for lab, v in per_slice.items()}
        enh = per_slice["1"]
        out = {
            "shape": list(m.shape),
            "n_slices": int(m.shape[2]),
            "per_slice": per_slice,
            "totals_mm3": totals,
            "peak_enhancing_slice": int(np.argmax(enh)) if max(enh) > 0 else int(m.shape[2] // 2),
            "available_bg": [
                n for n, mod in _MOD_BY_NAME.items()
                if self.src.exists(paths.dbt_skull_strip(patient, tp, mod))
            ],
        }
        self._stats[key] = out
        return out

    def voxels(self, patient: str, tp: str, label: int, prev: str | None) -> dict:
        """Surface voxels of one compartment, tagged for colouring.

        Only surface voxels are sent: the inside of a solid lump is never visible, and dropping
        it cuts the payload by roughly an order of magnitude. ``tag`` is the lesion number when
        ``prev`` is absent; with ``prev`` it is 0 kept / 1 gained / 2 lost since that visit, which
        is what answers "where did this thing grow".
        """
        cur = self.mask(patient, tp) == label
        if prev:
            old = self.mask(patient, prev) == label
            gained, lost = cur & ~old, old & ~cur
            keep = cur & old
            parts = [(keep, 0), (gained, 1), (lost, 2)]
            comps = None
        else:
            parts = [(cur, 0)]
            comps = _label_components(cur)

        xs: list[int] = []; ys: list[int] = []; zs: list[int] = []; tg: list[int] = []
        for vol, tag in parts:
            surf = _surface(vol)
            ii = np.argwhere(surf)
            if ii.size == 0:
                continue
            xs += ii[:, 0].tolist(); ys += ii[:, 1].tolist(); zs += ii[:, 2].tolist()
            if comps is None:
                tg += [tag] * len(ii)
            else:
                tg += comps[ii[:, 0], ii[:, 1], ii[:, 2]].tolist()

        sizes: list[int] = []
        if comps is not None and comps.max() > 0:
            sizes = np.bincount(comps.ravel())[1:].tolist()
        return {
            "x": xs, "y": ys, "z": zs, "tag": tg,
            "mode": "growth" if prev else "lesions",
            "n_surface": len(xs),
            "n_total": int(cur.sum()),
            "lesion_sizes": sizes,
            "counts": _growth_counts(cur, self.mask(patient, prev) == label) if prev else {},
        }

    def render(
        self, patient: str, tp: str, z: int, bg: str, labels: set[int], alpha: float, zoom: int
    ) -> bytes:
        m = self.mask(patient, tp)
        z = int(np.clip(z, 0, m.shape[2] - 1))

        def orient(a: np.ndarray) -> np.ndarray:
            # (x, y) -> (y, x) with anterior at the top; see module docstring.
            return np.flipud(a.T)

        msl = orient(m[:, :, z])
        bgvol = self.background(patient, tp, bg) if bg != "none" else None
        if bgvol is not None:
            g = orient(bgvol[:, :, z])
        else:
            g = np.zeros(msl.shape, dtype=np.uint8)
        rgb = np.repeat(g[:, :, None], 3, axis=2).astype(np.float32)

        for lab in sorted(labels):
            colour = LABEL_COLOURS.get(lab)
            if colour is None:
                continue
            sel = msl == lab
            if not sel.any():
                continue
            rgb[sel] = (1.0 - alpha) * rgb[sel] + alpha * np.array(colour, dtype=np.float32)

        im = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")
        if zoom > 1:
            im = im.resize((im.width * zoom, im.height * zoom), Image.NEAREST)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()


def load_practice(lock_path: Path, src: ZipSource) -> dict[str, list[str]]:
    """Practice patients -> every timepoint whose atlas mask is actually present in the zip.

    Timepoints come from the archive, not from the lock's ``assessable_timepoints``: browsing is
    exploration, and the non-assessable timepoints are part of what there is to look at. The
    PATIENT list comes from the lock and nowhere else.
    """
    lock = json.loads(lock_path.read_text())
    ids = [p["patient_id"] for p in lock["practice"]["patients"]]
    pat = re.compile(r"^Imaging/([^/]+)/([^/]+)/DeepBraTumIA-segmentation/atlas/"
                     r"segmentation/seg_mask\.nii\.gz$")
    found: dict[str, list[str]] = {p: [] for p in ids}
    for name in src.names:
        m = pat.match(name)
        if m and m.group(1) in found:
            found[m.group(1)].append(m.group(2))
    return {p: sorted(tps) for p, tps in found.items() if tps}


PAGE = """<!doctype html><html><head><meta charset=utf-8>
<title>LUMIERE mask browser</title><style>
*{box-sizing:border-box}
body{margin:0;background:#0d0f13;color:#d7dbe3;font:13px/1.45 ui-sans-serif,system-ui,-apple-system,sans-serif}
header{padding:10px 16px;border-bottom:1px solid #232833;display:flex;gap:18px;align-items:baseline;flex-wrap:wrap}
h1{font-size:13px;margin:0;font-weight:650;letter-spacing:.02em}
.sub{font-size:11px;color:#6f7887}
main{display:flex;gap:0;align-items:flex-start;height:calc(100vh - 43px)}
#side{width:260px;flex:none;border-right:1px solid #232833;height:100%;overflow-y:auto;padding:12px}
#stage{flex:1;display:flex;flex-direction:column;padding:14px;min-width:0;height:100%}
#imgwrap{flex:1;display:flex;align-items:center;justify-content:center;overflow:auto;min-height:0}
#foot{flex:none;display:flex;flex-direction:column;align-items:center;padding-top:8px}
.grp{margin-bottom:16px}
.lbl{font-size:10px;text-transform:uppercase;letter-spacing:.09em;color:#6f7887;margin-bottom:6px;font-weight:600}
select,button{background:#161a21;color:#d7dbe3;border:1px solid #2b3140;border-radius:5px;padding:5px 7px;font:inherit;font-size:12px}
select{width:100%}
button{cursor:pointer}button:hover{background:#1e242e}
button.on{background:#2b3853;border-color:#3f5180}
.row{display:flex;gap:5px;flex-wrap:wrap}
.chip{display:flex;align-items:center;gap:6px;padding:4px 7px;border:1px solid #2b3140;border-radius:5px;cursor:pointer;font-size:12px;user-select:none}
.chip.off{opacity:.32}
.sw{width:10px;height:10px;border-radius:2px;flex:none}
img{image-rendering:pixelated;border-radius:4px;background:#000}
#imgwrap::-webkit-scrollbar,#side::-webkit-scrollbar{width:7px;height:7px}
#imgwrap::-webkit-scrollbar-thumb,#side::-webkit-scrollbar-thumb{background:#2b3140;border-radius:4px}
#imgwrap::-webkit-scrollbar-track{background:transparent}
#meta{font-size:11px;color:#8b94a4;margin-top:7px;text-align:center;font-variant-numeric:tabular-nums;line-height:1.7}
#meta b{color:#d7dbe3;font-weight:600}
input[type=range]{width:100%}
#spark{display:block;margin-top:10px;background:#11141a;border-radius:4px;cursor:pointer}
kbd{background:#1b2029;border:1px solid #2b3140;border-radius:3px;padding:1px 4px;font-size:10px;font-family:ui-monospace,monospace}
.hint{font-size:10.5px;color:#5d6675;line-height:1.7;margin-top:4px}
</style></head><body>
<header><h1>LUMIERE mask browser</h1>
<span class=sub><a href="/3d" style="color:#7fa6d8">3D view</a> &middot; DeepBraTumIA &middot; atlas MNI 1&nbsp;mm &middot; axial &middot; neurological (patient-left on screen-right) &middot; practice arm only</span>
</header>
<main>
<div id=side>
  <div class=grp><div class=lbl>Patient</div><select id=pat></select></div>
  <div class=grp><div class=lbl>Timepoint <span id=tpn class=sub></span></div><select id=tp></select>
    <div class=row style="margin-top:6px"><button id=tprev>&larr; prev</button><button id=tnext>next &rarr;</button></div></div>
  <div class=grp><div class=lbl>Compartments</div><div class=row id=labs></div></div>
  <div class=grp><div class=lbl>Background</div><select id=bg></select></div>
  <div class=grp><div class=lbl>Overlay opacity <span id=av class=sub></span></div>
    <input type=range id=alpha min=0 max=100 value=45></div>
  <div class=grp><div class=lbl>Zoom</div><div class=row>
    <button data-z=2>2x</button><button data-z=3>3x</button><button data-z=4>4x</button></div></div>
  <div class=grp><div class=lbl>Enhancing voxels per slice</div>
    <canvas id=spark width=236 height=54></canvas>
    <div class=hint>Click to jump. The marker is the current slice; the tallest bar is the peak-area slice.</div></div>
  <div class=grp><div class=lbl>Keys</div><div class=hint>
    <kbd>&uarr;</kbd><kbd>&darr;</kbd> slice &middot; <kbd>&larr;</kbd><kbd>&rarr;</kbd> timepoint<br>
    <kbd>1</kbd><kbd>2</kbd><kbd>3</kbd> toggle compartment &middot; <kbd>p</kbd> peak slice</div></div>
</div>
<div id=stage>
  <div id=imgwrap><img id=view alt="axial slice"></div>
  <div id=foot>
    <input type=range id=zsl min=0 max=181 value=90 style="width:min(560px,92%)">
    <div id=meta></div>
  </div>
</div>
</main>
<script>
const S={idx:null,pat:null,tp:null,z:90,bg:'ct1',labs:new Set([1,2,3]),alpha:.45,zoom:2,st:null};
const NAMES={1:'enhancing',2:'necrosis / non-enh',3:'edema'};
const COL={1:'#ff3b30',2:'#0a84ff',3:'#ffd60a'};
const $=id=>document.getElementById(id);

function chips(){
  $('labs').innerHTML='';
  for(const k of [1,2,3]){
    const d=document.createElement('div');
    d.className='chip'+(S.labs.has(k)?'':' off');
    d.innerHTML=`<span class=sw style="background:${COL[k]}"></span>${k} ${NAMES[k]}`;
    d.onclick=()=>{S.labs.has(k)?S.labs.delete(k):S.labs.add(k);chips();draw();};
    $('labs').appendChild(d);
  }
}

async function loadIndex(){
  S.idx=await (await fetch('/api/index')).json();
  $('pat').innerHTML=Object.keys(S.idx.patients).map(p=>`<option>${p}</option>`).join('');
  S.pat=Object.keys(S.idx.patients)[0];
  await loadPatient();
}
async function loadPatient(){
  const tps=S.idx.patients[S.pat];
  $('tp').innerHTML=tps.map(t=>`<option>${t}</option>`).join('');
  $('tpn').textContent=`(${tps.length})`;
  S.tp=tps[0];
  await loadTp();
}
async function loadTp(){
  S.st=await (await fetch(`/api/tp?patient=${S.pat}&tp=${S.tp}`)).json();
  const avail=S.st.available_bg;
  $('bg').innerHTML=avail.map(b=>`<option${b===S.bg?' selected':''}>${b}</option>`).join('')+'<option>none</option>';
  if(!avail.includes(S.bg)&&avail.length) S.bg=avail[0];
  $('bg').value=S.bg;
  $('zsl').max=S.st.n_slices-1;
  S.z=S.st.peak_enhancing_slice;
  draw();
}
function draw(){
  $('zsl').value=S.z;
  const labs=[...S.labs].sort().join(',')||'none';
  $('view').src=`/png?patient=${S.pat}&tp=${S.tp}&z=${S.z}&bg=${S.bg}`
    +`&labels=${labs}&alpha=${S.alpha}&zoom=${S.zoom}`;
  const ps=S.st.per_slice, T=S.st.totals_mm3;
  $('meta').innerHTML=
    `slice z=<b>${S.z}</b> / ${S.st.n_slices-1} &nbsp;&middot;&nbsp; on this slice: `+
    [1,2,3].map(k=>`<span style="color:${COL[k]}">&#9632;</span> <b>${ps[k][S.z]}</b>`).join(' &nbsp; ')+
    `<br>whole timepoint (mm&sup3;): `+
    [1,2,3].map(k=>`${NAMES[k]} <b>${T[k].toLocaleString()}</b>`).join(' &nbsp;&middot;&nbsp; ');
  spark();
}
function spark(){
  const c=$('spark'),x=c.getContext('2d'),n=S.st.n_slices,v=S.st.per_slice[1],mx=Math.max(...v,1);
  x.clearRect(0,0,c.width,c.height);
  x.fillStyle='#ff3b30';
  for(let i=0;i<n;i++){
    const h=Math.round(v[i]/mx*(c.height-2)), px=Math.floor(i/n*c.width);
    if(h>0) x.fillRect(px,c.height-h,Math.max(1,Math.ceil(c.width/n)),h);
  }
  x.fillStyle='#eaeef6';
  x.fillRect(Math.floor(S.z/n*c.width),0,1,c.height);
}
$('spark').onclick=e=>{
  const r=e.target.getBoundingClientRect();
  S.z=Math.round((e.clientX-r.left)/r.width*(S.st.n_slices-1));draw();
};
$('pat').onchange=e=>{S.pat=e.target.value;loadPatient();};
$('tp').onchange=e=>{S.tp=e.target.value;loadTp();};
$('bg').onchange=e=>{S.bg=e.target.value;draw();};
$('zsl').oninput=e=>{S.z=+e.target.value;draw();};
$('alpha').oninput=e=>{S.alpha=+e.target.value/100;$('av').textContent=e.target.value+'%';draw();};
document.querySelectorAll('[data-z]').forEach(b=>b.onclick=()=>{S.zoom=+b.dataset.z;draw();});
function stepTp(d){
  const tps=S.idx.patients[S.pat],i=tps.indexOf(S.tp)+d;
  if(i>=0&&i<tps.length){S.tp=tps[i];$('tp').value=S.tp;loadTp();}
}
$('tprev').onclick=()=>stepTp(-1);$('tnext').onclick=()=>stepTp(1);
addEventListener('keydown',e=>{
  if(e.target.tagName==='SELECT')return;
  const k=e.key;
  if(k==='ArrowUp'){S.z=Math.min(S.z+1,S.st.n_slices-1);draw();e.preventDefault();}
  else if(k==='ArrowDown'){S.z=Math.max(S.z-1,0);draw();e.preventDefault();}
  else if(k==='ArrowRight'){stepTp(1);e.preventDefault();}
  else if(k==='ArrowLeft'){stepTp(-1);e.preventDefault();}
  else if(k==='p'){S.z=S.st.peak_enhancing_slice;draw();}
  else if('123'.includes(k)){const n=+k;S.labs.has(n)?S.labs.delete(n):S.labs.add(n);chips();draw();}
});
$('av').textContent='45%';chips();loadIndex();
</script></body></html>"""


PAGE3D = """<!doctype html><html><head><meta charset=utf-8>
<title>LUMIERE lesions in 3D</title><style>
*{box-sizing:border-box}
body{margin:0;background:#0d0f13;color:#d7dbe3;font:13px/1.45 ui-sans-serif,system-ui,-apple-system,sans-serif;overflow:hidden}
header{padding:9px 16px;border-bottom:1px solid #232833;display:flex;gap:16px;align-items:baseline;flex-wrap:wrap}
h1{font-size:13px;margin:0;font-weight:650}
.sub{font-size:11px;color:#6f7887}
a{color:#7fa6d8;text-decoration:none}a:hover{text-decoration:underline}
main{display:flex;height:calc(100vh - 40px)}
#side{width:250px;flex:none;border-right:1px solid #232833;padding:12px;overflow-y:auto}
#cv{flex:1;position:relative}
canvas{display:block}
.grp{margin-bottom:15px}
.lbl{font-size:10px;text-transform:uppercase;letter-spacing:.09em;color:#6f7887;margin-bottom:5px;font-weight:600}
select{width:100%;background:#161a21;color:#d7dbe3;border:1px solid #2b3140;border-radius:5px;padding:5px 7px;font:inherit;font-size:12px}
.hint{font-size:10.5px;color:#5d6675;line-height:1.6;margin-top:5px}
#legend{margin-top:4px;font-size:11.5px;line-height:1.9}
#legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:7px;vertical-align:-1px}
#stat{font-size:11px;color:#8b94a4;line-height:1.75;font-variant-numeric:tabular-nums}
#stat b{color:#d7dbe3}
#busy{position:absolute;top:12px;left:14px;font-size:11px;color:#6f7887}
</style></head><body>
<header><h1>LUMIERE lesions in 3D</h1>
<span class=sub>surface voxels &middot; atlas MNI 1&nbsp;mm &middot; drag to turn, scroll to zoom &middot;
<a href="/">back to slices</a></span></header>
<main>
<div id=side>
  <div class=grp><div class=lbl>Patient</div><select id=pat></select></div>
  <div class=grp><div class=lbl>Visit</div><select id=tp></select></div>
  <div class=grp><div class=lbl>Compartment</div><select id=lab>
    <option value=1>1 enhancing</option><option value=2>2 necrosis / non-enh</option>
    <option value=3>3 edema</option></select></div>
  <div class=grp><div class=lbl>Colour by</div><select id=mode>
    <option value=lesions>separate lesions</option>
    <option value=growth>change since previous visit</option></select></div>
  <div class=grp><div class=lbl>Legend</div><div id=legend></div></div>
  <div class=grp><div class=lbl>Numbers</div><div id=stat></div></div>
  <div class=grp><div class=hint>One colour = one 26-connected lesion, floor 20&nbsp;mm&sup3; &mdash;
    the same rule the pipeline uses. If the specks share a colour, they are one tumour joined
    through slices you cannot see edge-on.</div></div>
</div>
<div id=cv><div id=busy>loading…</div></div>
</main>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const PAL=[0xff3b30,0x0a84ff,0xffd60a,0x30d158,0xbf5af2,0xff9f0a,0x64d2ff,0xff6482];
const GROWTH={0:0x6e7681,1:0xff3b30,2:0x0a84ff};
const GNAME={0:'unchanged',1:'gained since last visit',2:'lost since last visit'};
const $=id=>document.getElementById(id);
let S={idx:null,pat:null,tp:null,lab:1,mode:'lesions'};
let renderer,scene,cam,group,mesh;

function initGL(){
  const host=$('cv');
  renderer=new THREE.WebGLRenderer({antialias:true});
  renderer.setPixelRatio(devicePixelRatio);
  scene=new THREE.Scene(); scene.background=new THREE.Color(0x0d0f13);
  cam=new THREE.PerspectiveCamera(42,1,1,4000);
  group=new THREE.Group(); scene.add(group);
  scene.add(new THREE.AmbientLight(0xffffff,.62));
  const d=new THREE.DirectionalLight(0xffffff,.85); d.position.set(1,1.2,1); scene.add(d);
  const d2=new THREE.DirectionalLight(0xffffff,.3); d2.position.set(-1,-.6,-.8); scene.add(d2);
  host.appendChild(renderer.domElement);
  resize(); addEventListener('resize',resize);
  let drag=false,px=0,py=0;
  const el=renderer.domElement;
  el.addEventListener('mousedown',e=>{drag=true;px=e.clientX;py=e.clientY});
  addEventListener('mouseup',()=>drag=false);
  addEventListener('mousemove',e=>{
    if(!drag)return;
    group.rotation.y+=(e.clientX-px)*.01; group.rotation.x+=(e.clientY-py)*.01;
    px=e.clientX; py=e.clientY;
  });
  el.addEventListener('wheel',e=>{e.preventDefault();cam.position.z*=e.deltaY>0?1.1:0.9;},{passive:false});
  (function loop(){requestAnimationFrame(loop);renderer.render(scene,cam);})();
}
function resize(){
  const host=$('cv'), w=host.clientWidth, h=host.clientHeight;
  renderer.setSize(w,h); cam.aspect=w/h; cam.updateProjectionMatrix();
}

async function load(){
  $('busy').textContent='loading…';
  const prev=S.mode==='growth'?prevTp():'';
  const r=await fetch(`/api/voxels?patient=${S.pat}&tp=${S.tp}&label=${S.lab}`+
                      (prev?`&prev=${prev}`:''));
  const d=await r.json();
  if(d.error){$('busy').textContent=d.error;return;}
  draw(d); $('busy').textContent='';
}
function prevTp(){
  const t=S.idx.patients[S.pat], i=t.indexOf(S.tp);
  return i>0?t[i-1]:'';
}
function draw(d){
  if(mesh){group.remove(mesh);mesh.geometry.dispose();mesh.material.dispose();mesh=null;}
  const n=d.x.length;
  if(!n){$('legend').innerHTML='<span class=sub>nothing in this compartment</span>';
         $('stat').innerHTML='0 voxels';return;}
  const geo=new THREE.BoxGeometry(1,1,1);
  // three.js multiplies the per-instance colour by the geometry's own colour attribute.
  // BoxGeometry has none, so without this every cube renders black.
  geo.setAttribute('color', new THREE.BufferAttribute(
      new Float32Array(geo.attributes.position.count*3).fill(1), 3));
  const mat=new THREE.MeshLambertMaterial({vertexColors:true});
  mesh=new THREE.InstancedMesh(geo,mat,n);
  const m=new THREE.Matrix4(), col=new THREE.Color();
  let cx=0,cy=0,cz=0;
  for(let i=0;i<n;i++){cx+=d.x[i];cy+=d.y[i];cz+=d.z[i];}
  cx/=n;cy/=n;cz/=n;
  for(let i=0;i<n;i++){
    m.setPosition(d.x[i]-cx, d.z[i]-cz, d.y[i]-cy);   // z is up: anatomical superior
    mesh.setMatrixAt(i,m);
    const t=d.tag[i];
    col.setHex(d.mode==='growth'?GROWTH[t]:(t>0?PAL[(t-1)%PAL.length]:0x555a63));
    mesh.setColorAt(i,col);
  }
  mesh.instanceMatrix.needsUpdate=true; mesh.instanceColor.needsUpdate=true;
  group.add(mesh);
  let rad=0;
  for(let i=0;i<n;i++){
    const a=d.x[i]-cx,b=d.y[i]-cy,c=d.z[i]-cz; rad=Math.max(rad,Math.hypot(a,b,c));
  }
  cam.position.set(0,0,Math.max(rad*3.1,60)); cam.lookAt(0,0,0);
  legend(d); stats(d);
}
function legend(d){
  if(d.mode==='growth'){
    $('legend').innerHTML=[0,1,2].map(k=>
      `<div><i style="background:#${GROWTH[k].toString(16).padStart(6,'0')}"></i>${GNAME[k]}</div>`).join('');
  }else{
    const s=d.lesion_sizes||[];
    $('legend').innerHTML=s.length
      ? s.slice(0,8).map((v,i)=>`<div><i style="background:#${PAL[i%PAL.length].toString(16).padStart(6,'0')}"></i>lesion ${i+1} &mdash; ${v.toLocaleString()} mm&sup3;</div>`).join('')
        +(s.length>8?`<div class=sub>+${s.length-8} more</div>`:'')
      : '<span class=sub>no lesion above the floor</span>';
  }
}
function stats(d){
  const c=d.counts||{};
  $('stat').innerHTML=`total <b>${d.n_total.toLocaleString()}</b> mm&sup3;<br>`+
    `surface voxels drawn <b>${d.n_surface.toLocaleString()}</b>`+
    (d.mode==='growth'&&c.gained!==undefined
      ? `<br>unchanged <b>${c.kept.toLocaleString()}</b><br>gained <b>${c.gained.toLocaleString()}</b>`+
        `<br>lost <b>${c.lost.toLocaleString()}</b>` : '')+
    (d.lesion_sizes&&d.lesion_sizes.length?`<br>lesions <b>${d.lesion_sizes.length}</b>`:'');
}

(async function(){
  initGL();
  S.idx=await (await fetch('/api/index')).json();
  const ps=Object.keys(S.idx.patients);
  $('pat').innerHTML=ps.map(p=>`<option>${p}</option>`).join(''); S.pat=ps[0];
  fillTp(); load();
})();
function fillTp(){
  const t=S.idx.patients[S.pat];
  $('tp').innerHTML=t.map(x=>`<option>${x}</option>`).join(''); S.tp=t[0];
}
$('pat').onchange=e=>{S.pat=e.target.value;fillTp();load();};
$('tp').onchange=e=>{S.tp=e.target.value;load();};
$('lab').onchange=e=>{S.lab=+e.target.value;load();};
$('mode').onchange=e=>{S.mode=e.target.value;load();};
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    store: Store  # set on the class in main()

    def log_message(self, fmt, *args):  # quieter than the stdlib default
        pass

    def _send(self, code: int, body: bytes, ctype: str, cache: bool = False) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _case(self, q: dict) -> tuple[str, str]:
        patient = (q.get("patient") or [""])[0]
        tp = (q.get("tp") or [""])[0]
        if not _SAFE_ID.match(patient or "") or not _SAFE_ID.match(tp or ""):
            raise ValueError("patient and tp are required and must be plain identifiers")
        return patient, tp

    def do_GET(self) -> None:  # noqa: N802
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path == "/":
                self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            elif u.path == "/api/index":
                self._json({
                    "patients": self.store.practice,
                    "labels": {str(k): LABEL_SCHEMA["DeepBraTumIA"][k] for k in LABEL_COLOURS},
                })
            elif u.path == "/3d":
                self._send(200, PAGE3D.encode(), "text/html; charset=utf-8")
            elif u.path == "/api/voxels":
                patient, tp = self._case(q)
                prev = (q.get("prev") or [""])[0] or None
                if prev and not _SAFE_ID.match(prev):
                    raise ValueError("prev must be a plain identifier")
                self._json(self.store.voxels(
                    patient, tp,
                    label=int((q.get("label") or [1])[0]),
                    prev=prev,
                ))
            elif u.path == "/api/tp":
                self._json(self.store.stats(*self._case(q)))
            elif u.path == "/png":
                patient, tp = self._case(q)
                labels = {
                    int(v) for v in (q.get("labels") or ["1,2,3"])[0].split(",")
                    if v.strip().isdigit()
                }
                png = self.store.render(
                    patient, tp,
                    z=int((q.get("z") or [90])[0]),
                    bg=(q.get("bg") or ["ct1"])[0],
                    labels=labels,
                    alpha=float(np.clip(float((q.get("alpha") or [0.45])[0]), 0.0, 1.0)),
                    zoom=int(np.clip(int((q.get("zoom") or [3])[0]), 1, 6)),
                )
                self._send(200, png, "image/png", cache=True)
            else:
                self._json({"error": "not found"}, 404)
        except PermissionError as e:
            self._json({"error": str(e)}, 403)
        except (KeyError, ValueError) as e:
            self._json({"error": str(e)}, 400)
        except Exception as e:  # surfaced in the UI rather than dying silently
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zip", default=str(DEFAULT_ZIP))
    ap.add_argument("--lock", default=str(COHORT_LOCK))
    ap.add_argument("--port", type=int, default=8760)
    a = ap.parse_args()

    src = ZipSource(a.zip)
    practice = load_practice(Path(a.lock), src)
    if not practice:
        print("no practice-arm masks found in the archive", file=sys.stderr)
        return 1

    Handler.store = Store(Path(a.zip), practice)
    n_tp = sum(len(v) for v in practice.values())
    print(f"practice arm: {len(practice)} patients, {n_tp} timepoints with an atlas mask")
    for p, tps in practice.items():
        print(f"  {p:<14} {len(tps):>3} timepoints  {tps[0]} .. {tps[-1]}")
    print(f"\nmask browser on http://localhost:{a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
