"""The radiologist says "new lesion" and we find none in the enhancing mask.
Is the new disease in a different compartment?"""
import csv, json, sys
sys.path.insert(0,'src')
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from track_registered import label, track, FLOOR
from rano.adapters.lumiere.zip_ref import ZipSource
from rano.adapters.lumiere import paths
from rano.registration import RigidFit, resample_like_fixed

PAT = 'Patient-072'
src = ZipSource('Imaging-v202211.zip')
reg = json.loads(open('output/registration/transforms.json').read())['patients'][PAT]
rows = list(csv.DictReader(open('LUMIERE-ExpertRating-v202211.csv')))
rr = [k for k in rows[0] if k.startswith('Rating rationale')][0]
note = {r['Date']: r[rr].strip() for r in rows if r['Patient']==PAT}
tps = list(reg['visits'].keys())

COMPARTMENTS = {'enhancing (1)': {1}, 'necrosis (2)': {2}, 'edema (3)': {3},
                'enh + nec (1,2)': {1,2}, 'anything (1,2,3)': {1,2,3}}
results = {}
for name, labels in COMPARTMENTS.items():
    series = []
    for tp in tps:
        a = np.asarray(src.open_nifti(paths.dbt_mask(PAT, tp)).dataobj)
        m = np.isin(a, list(labels))
        fit = RigidFit.from_dict(reg['visits'][tp])
        series.append((tp, resample_like_fixed(m.astype(np.uint8), fit, labels=True) > 0))
    results[name] = track(series, slack_mm=2)

print(f"{PAT} — new objects per visit, by compartment (registered, floor {FLOOR} mm3, 2 mm slack)\n")
hdr = f"  {'visit':<12}" + "".join(f"{n.split(' (')[0][:9]:>11}" for n in COMPARTMENTS) + "   radiologist"
print(hdr); print("  " + "-"*(len(hdr)+18))
for i, tp in enumerate(tps):
    cells = ""
    for name in COMPARTMENTS:
        nb = [s for s, t in results[name][i][1] if t == 'N']
        cells += f"{(str(len(nb)) if nb else '·'):>11}"
    txt = note.get(tp, '')
    said = 'NEW' if 'new' in txt.lower() else '   '
    print(f"  {tp:<12}{cells}   {said}  {txt[:46]}")
