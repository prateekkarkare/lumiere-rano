"""Between consecutive visits: how much stays put, and does the disease MOVE?

'stable %' = voxels present at both visits, as a share of the larger visit.
'shift mm' = distance between the centre of the GAINED tissue and the centre of the LOST
tissue. Near zero means growth and shrinkage happen in the same place -- a lesion breathing
in and out. Large means disease vanished from one place and appeared in another.
"""
import csv, json, re, sys
sys.path.insert(0,'src')
import numpy as np
from rano.adapters.lumiere.zip_ref import ZipSource
from rano.adapters.lumiere import paths

src = ZipSource('Imaging-v202211.zip')
practice = [p["patient_id"] for p in json.load(open('output/cohort/cohort_lock.json'))['practice']['patients']]
rows = list(csv.DictReader(open('LUMIERE-ExpertRating-v202211.csv')))
rr = [k for k in rows[0] if k.startswith('Rating rationale')][0]
note = {(r['Patient'], r['Date']): r[rr].strip() for r in rows}

allrows = []
for pat in practice:
    tps = sorted({m.group(1) for m in (re.match(rf'Imaging/{pat}/([^/]+)/DeepBraTumIA', n)
                                       for n in src.names) if m})
    tps = [t for t in tps if src.exists(paths.dbt_mask(pat, t))]
    print(f"\n{pat}")
    print(f"  {'from':<12} {'to':<12} {'prev':>7} {'cur':>7} {'kept':>7} {'gain':>7} {'lost':>7}"
          f" {'stable':>7} {'shift':>7}   expert said")
    prev = None
    for tp in tps:
        cur = np.asarray(src.open_nifti(paths.dbt_mask(pat, tp)).dataobj) == 1
        if prev is not None and (prev.any() or cur.any()):
            kept = prev & cur; gain = cur & ~prev; lost = prev & ~cur
            nk, ng, nl = int(kept.sum()), int(gain.sum()), int(lost.sum())
            denom = max(int(prev.sum()), int(cur.sum()), 1)
            stable = 100*nk/denom
            if ng and nl:
                cg = np.argwhere(gain).mean(axis=0); cl = np.argwhere(lost).mean(axis=0)
                shift = float(np.linalg.norm(cg-cl))
            else:
                shift = float('nan')
            said = note.get((pat, tp), '')
            flag = 'NEW' if 'new' in said.lower() else ''
            print(f"  {ptp:<12} {tp:<12} {int(prev.sum()):>7} {int(cur.sum()):>7} {nk:>7} "
                  f"{ng:>7} {nl:>7} {stable:>6.0f}% {shift:>7.1f}   {flag}")
            allrows.append((pat, ptp, tp, stable, shift, nk, ng, nl))
        prev, ptp = cur, tp

st = np.array([r[3] for r in allrows]); sh = np.array([r[4] for r in allrows])
sh = sh[~np.isnan(sh)]
print(f"\n\nacross {len(allrows)} consecutive visit pairs in the practice arm:")
print(f"  share of the lesion that stays put:  median {np.median(st):.0f}%   "
      f"p10 {np.percentile(st,10):.0f}%   p90 {np.percentile(st,90):.0f}%")
print(f"  distance between new tissue and lost tissue: median {np.median(sh):.1f} mm   "
      f"p10 {np.percentile(sh,10):.1f}   p90 {np.percentile(sh,90):.1f} mm")
print(f"  pairs where that distance exceeds 15 mm: {(sh>15).sum()}/{len(sh)}")
