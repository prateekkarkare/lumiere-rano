"""
Does the necrosis label contain surgical cavities?

The question behind rule 3 of the ruler: fill only holes the tissue ENCLOSES, and never simply add
the necrosis label back. A dead tumour core is wrapped in living tumour, and measuring through it
is right. A surgical cavity is an empty pocket where tumour was cut out, and measuring across it
is wrong. If DeepBraTumIA labels cavities as necrosis, adding label 2 back would measure an empty
hole as tumour.

The dataset supplies the test. A scan rated Post-Op with rationale CRET means the surgeon removed
ALL the enhancing tumour. Any large "necrosis" on such a scan cannot be a dead core -- there is no
tumour left for it to be the core of.

Practice arm only. Run from the repository root.
"""
import csv
import json
import sys

sys.path.insert(0, 'src')
import numpy as np  # noqa: E402

from rano.adapters.lumiere import paths  # noqa: E402
from rano.adapters.lumiere.zip_ref import ZipSource  # noqa: E402

practice = {p["patient_id"] for p in
            json.load(open('output/cohort/cohort_lock.json'))['practice']['patients']}
rows = list(csv.DictReader(open('LUMIERE-ExpertRating-v202211.csv')))
rk = next(k for k in rows[0] if k.startswith('Rating ('))
rr = next(k for k in rows[0] if k.startswith('Rating rationale'))
src = ZipSource('Imaging-v202211.zip')


def volumes(patient: str, tp: str):
    member = paths.dbt_mask(patient, tp)
    if not src.exists(member):
        return None
    a = np.asarray(src.open_nifti(member).dataobj)
    return int((a == 1).sum()), int((a == 2).sum()), int((a == 3).sum())


def collect(keep) -> list[tuple]:
    out = []
    for r in rows:
        if r['Patient'] in practice and keep(r):
            v = volumes(r['Patient'], r['Date'])
            if v:
                out.append((r['Patient'], r['Date'], *v))
    return out


cret = collect(lambda r: r[rk].strip() == 'Post-Op' and 'CRET' in r[rr])
growing = collect(lambda r: r[rk].strip() == 'PD')

print("Scans where the surgeon removed ALL enhancing tumour (Post-Op, CRET):\n")
print(f"  {'patient':<13} {'visit':<12} {'enhancing':>10} {'necrosis':>10} {'edema':>10}")
for p, tp, e, n, o in cret:
    print(f"  {p:<13} {tp:<12} {e:>10,} {n:>10,} {o:>10,}")


def med(rs: list[tuple], i: int) -> int:
    return int(np.median([x[i] for x in rs]))


print(f"\n  median over {len(cret)} CRET scans:   enhancing {med(cret, 2):>6,}   "
      f"necrosis {med(cret, 3):>6,}   edema {med(cret, 4):>6,} mm3")
print(f"  median over {len(growing)} PD scans:     enhancing {med(growing, 2):>6,}   "
      f"necrosis {med(growing, 3):>6,}   edema {med(growing, 4):>6,} mm3")
print("\nMore 'necrosis' where the tumour was removed than where it is growing is backwards for a dead")
print("core -- and exactly what cavities labelled as necrosis would produce. So the ruler fills only")
print("holes the tissue completely encloses, and never adds label 2 back wholesale.")
