"""
Piece 2 — register every practice-arm visit onto its patient's first visit.

WHY
---
Atlas space gives every visit the same grid, not the same anatomy in the same voxels. Each visit
was registered to the MNI template independently and those registrations disagree: consecutive
visits' brain masks overlap ~92% and the brain's centre wanders ~2.7mm. Volumetry never cared --
a voxel count is blind to position -- but lesion tracking does, and a 3mm slip destroys matching
for small lesions, which is exactly the size a NEW lesion is.

WHAT IT WRITES
--------------
``output/registration/transforms.json``: six numbers plus a rotation centre per visit, and the
QC that says whether the fit helped. Transforms only -- no resampled volumes. A label mask loses
up to half a voxel of boundary every time it is resampled, and it has already been resampled once
(native -> atlas), so anything downstream should transform COORDINATES using these numbers rather
than move whole volumes again.

ANCHOR
------
Each patient's chronologically first visit, deterministically. Everything is registered to it, so
any two visits of that patient are then in a common frame. The anchor's own transform is the
identity: it is already where it needs to be.

Usage:
    .venv/bin/python scripts/run_registration.py
    .venv/bin/python scripts/run_registration.py --patients Patient-067 Patient-072
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rano.adapters.lumiere import paths  # noqa: E402
from rano.adapters.lumiere.zip_ref import ZipSource  # noqa: E402
from rano.contract.case import Modality  # noqa: E402
from rano.registration import RigidFit, fit_rigid, registration_mask, resample_like_fixed  # noqa: E402

DEFAULT_ZIP = ROOT / "Imaging-v202211.zip"
COHORT_LOCK = ROOT / "output" / "cohort" / "cohort_lock.json"
OUT = ROOT / "output" / "registration" / "transforms.json"

#: registration runs on the post-contrast T1 -- present at every practice timepoint, and the
#: sequence the segmentation itself was driven from, so mask and image agree by construction.
REG_MODALITY = Modality.CT1

#: millimetres of slack around the abnormal tissue excluded from the fit
LESION_MARGIN_MM = 5


def overlap_pct(a: np.ndarray, b: np.ndarray) -> float:
    union = int((a | b).sum())
    return 100.0 * int((a & b).sum()) / union if union else float("nan")


def centroid_distance_mm(a: np.ndarray, b: np.ndarray) -> float:
    if not a.any() or not b.any():
        return float("nan")
    return float(np.linalg.norm(np.argwhere(a).mean(0) - np.argwhere(b).mean(0)))


def timepoints_for(src: ZipSource, patient: str) -> list[str]:
    pat = re.compile(rf"^Imaging/{re.escape(patient)}/([^/]+)/DeepBraTumIA-segmentation/")
    seen = {m.group(1) for m in (pat.match(n) for n in src.names) if m}
    usable = [
        tp for tp in sorted(seen)
        if src.exists(paths.dbt_mask(patient, tp))
        and src.exists(paths.dbt_brain_mask(patient, tp))
        and src.exists(paths.dbt_skull_strip(patient, tp, REG_MODALITY))
    ]
    return usable


def load_visit(src: ZipSource, patient: str, tp: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Image to align on, the brain, and everything abnormal (tumour, cavity, oedema)."""
    img = np.asarray(
        src.open_nifti(paths.dbt_skull_strip(patient, tp, REG_MODALITY)).dataobj, dtype=np.float32
    )
    brain = np.asarray(src.open_nifti(paths.dbt_brain_mask(patient, tp)).dataobj) > 0
    abnormal = np.asarray(src.open_nifti(paths.dbt_mask(patient, tp)).dataobj) > 0
    return img, brain, abnormal


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zip", default=str(DEFAULT_ZIP))
    ap.add_argument("--lock", default=str(COHORT_LOCK))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--patients", nargs="*", help="restrict to these patient ids")
    a = ap.parse_args()

    src = ZipSource(a.zip)
    lock = json.loads(Path(a.lock).read_text())
    practice = [p["patient_id"] for p in lock["practice"]["patients"]]
    if a.patients:
        unknown = set(a.patients) - set(practice)
        if unknown:  # the cohort lock is not a suggestion
            print(f"refusing: {sorted(unknown)} are not in the practice arm", file=sys.stderr)
            return 2
        practice = [p for p in practice if p in a.patients]

    t0 = time.time()
    out: dict = {"registration_modality": REG_MODALITY.value,
                 "lesion_margin_mm": LESION_MARGIN_MM,
                 "anchor_rule": "each patient's chronologically first usable visit",
                 "patients": {}}

    for patient in practice:
        tps = timepoints_for(src, patient)
        if not tps:
            continue
        anchor = tps[0]
        a_img, a_brain, a_abn = load_visit(src, patient, anchor)
        a_mask = registration_mask(a_brain, a_abn, LESION_MARGIN_MM)

        print(f"\n{patient}   anchor {anchor}   {len(tps)} visits")
        print(f"  {'visit':<12} {'shift':>7} {'rot deg':>8} {'overlap before':>15} "
              f"{'after':>7} {'centre before':>14} {'after':>7}")

        rec: dict = {"anchor": anchor, "visits": {}}
        prev_before = prev_after = None
        for tp in tps:
            centre = tuple(float(c) / 2.0 for c in a_brain.shape)
            status = "registered"
            if tp == anchor:
                fit = RigidFit.identity(center=centre)  # type: ignore[arg-type]
                brain_after, m_brain = a_brain, a_brain
                status = "anchor"
            else:
                m_img, m_brain, m_abn = load_visit(src, patient, tp)
                m_mask = registration_mask(m_brain, m_abn, LESION_MARGIN_MM)
                fit = fit_rigid(a_img, m_img, a_mask, m_mask)
                brain_after = resample_like_fixed(m_brain.astype(np.uint8), fit, labels=True) > 0

                # Trust nothing: judge the fit by whether the brains actually line up better,
                # not by the optimiser reporting convergence. A rigid fit can converge happily
                # onto a wrong minimum, and it looks identical from the inside.
                if overlap_pct(a_brain, brain_after) < overlap_pct(a_brain, m_brain):
                    slide = np.argwhere(m_brain).mean(0) - np.argwhere(a_brain).mean(0)
                    cand = RigidFit.from_translation(tuple(float(v) for v in slide), centre)  # type: ignore[arg-type]
                    cand_brain = resample_like_fixed(m_brain.astype(np.uint8), cand, labels=True) > 0
                    if overlap_pct(a_brain, cand_brain) > overlap_pct(a_brain, m_brain):
                        fit, brain_after, status = cand, cand_brain, "fell back to translation"
                    else:
                        fit = RigidFit.identity(center=centre)  # type: ignore[arg-type]
                        brain_after, status = m_brain, "REJECTED - left unregistered"

            ob, oa = overlap_pct(a_brain, m_brain), overlap_pct(a_brain, brain_after)
            cb, ca = centroid_distance_mm(a_brain, m_brain), centroid_distance_mm(a_brain, brain_after)
            rot = max(abs(d) for d in fit.rotation_degrees)
            rec["visits"][tp] = {
                **fit.to_dict(),
                "status": status,
                "qc": {"brain_overlap_before_pct": ob, "brain_overlap_after_pct": oa,
                       "brain_centre_before_mm": cb, "brain_centre_after_mm": ca},
            }
            note = "" if status in ("registered", "anchor") else f"  <- {status}"
            print(f"  {tp:<12} {fit.shift_magnitude_mm:>6.1f}m {rot:>8.2f} "
                  f"{ob:>14.1f}% {oa:>6.1f}% {cb:>13.1f} {ca:>6.1f}{note}", flush=True)
        out["patients"][patient] = rec

    p = Path(a.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))

    rows = [v["qc"] for r in out["patients"].values() for v in r["visits"].values()]
    ob = np.array([r["brain_overlap_before_pct"] for r in rows])
    oa = np.array([r["brain_overlap_after_pct"] for r in rows])
    print(f"\n{'='*70}\n{len(rows)} visits registered in {time.time()-t0:.0f}s")
    print(f"  brain overlap with the anchor:  before {np.median(ob):.1f}%  ->  after {np.median(oa):.1f}%  (median)")
    print(f"  worst case:                     before {ob.min():.1f}%  ->  after {oa.min():.1f}%")
    print(f"  visits made worse by the fit:   {(oa < ob - 0.5).sum()}/{len(rows)}")
    from collections import Counter
    st = Counter(v["status"] for r in out["patients"].values() for v in r["visits"].values())
    for k, n in st.most_common():
        print(f"    {k:<32} {n}")
    print(f"\nwrote {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
