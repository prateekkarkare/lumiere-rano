"""
Emit ``data_contract.json`` — Piece 1's deliverable artifact.

Walks the LUMIERE adapter, runs the checks, computes atlas-space compartment volumes with the
corrected label schema, attaches the size-dependent uncertainty and the expert RANO rating, and
writes one JSON document covering the requested patients.

Only the mask array is materialized per timepoint (volumes need it); modality voxels are never
read, so this stays fast over the cohort. Geometry is header-only throughout.

Usage:
    .venv/bin/python scripts/emit_data_contract.py --n 3
    .venv/bin/python scripts/emit_data_contract.py --all --out output/contract/data_contract.json
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

from rano.adapters.lumiere.adapter import LumiereAdapter
from rano.contract.case import ImageRef, MaskSource, Modality, Patient, Timepoint
from rano.contract.data_contract import (
    CheckEntry,
    DataContract,
    ImageEntry,
    PatientContract,
    TimepointContract,
    VolumeEntry,
    uncertainty_pp,
)
from rano.fingerprint.schema import GeometryFingerprint, ImageFingerprint, TimepointFingerprint
from rano.validate.checks.mask_grid_alignment import check_mask_grid_alignment
from rano.volumetry.volumes import compartment_volumes, region_volumes, voxel_volume_mm3

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ZIP = ROOT / "Imaging-v202211.zip"
DEFAULT_MANIFEST = ROOT / "LUMIERE-datacompleteness.csv"
DEFAULT_RATINGS = ROOT / "LUMIERE-ExpertRating-v202211.csv"
DEFAULT_OUT = ROOT / "output" / "contract" / "data_contract.json"

#: the configurable modality contract (default per the Piece 1 brief)
REQUIRED_MODALITIES = (Modality.CT1, Modality.FLAIR)
RESPONSE_LABELS = {"CR", "PR", "SD", "PD"}


def load_ratings(path: Path) -> dict[tuple[str, str], str]:
    rows = list(csv.DictReader(path.open()))
    key = next(k for k in rows[0] if k.startswith("Rating ("))
    return {(r["Patient"].strip(), r["Date"].strip()): r[key].strip() for r in rows}


def _image_entry(ref: ImageRef) -> ImageEntry:
    geo = ref.geometry
    return ImageEntry(
        source=ref.source,
        space=str(ref.space),
        shape=geo.shape,
        spacing_mm=geo.spacing,
        anisotropy_ratio=round(geo.anisotropy_ratio, 4),
        voxel_mm3=round(voxel_volume_mm3(geo.affine), 6),
    )


def _geo_fp(ref: ImageRef) -> ImageFingerprint:
    """A geometry-only fingerprint — enough for the alignment check, no voxels read."""
    geo = ref.geometry
    return ImageFingerprint(
        role="mask",
        source=ref.source,
        space=str(ref.space),
        geometry=GeometryFingerprint(
            shape=geo.shape, spacing=geo.spacing, orientation=geo.orientation,
            dtype=geo.dtype, anisotropy_ratio=geo.anisotropy_ratio,
            affine=tuple(tuple(float(x) for x in row) for row in geo.affine),
        ),
    )


def build_timepoint(tp: Timepoint, rating: str | None) -> TimepointContract:
    checks: list[CheckEntry] = []

    modalities = {m.value: _image_entry(ref) for m, ref in tp.modalities.items()}
    missing = [m.value for m in REQUIRED_MODALITIES if m not in tp.modalities]
    checks.append(CheckEntry(
        code="modalities_present",
        status="pass" if not missing else "fail",
        detail="all required modalities present" if not missing
               else f"missing required: {', '.join(missing)}",
    ))

    if tp.mask is None:
        checks.append(CheckEntry(code="mask_present", status="fail",
                                 detail="no canonical mask resolved for this timepoint"))
        return TimepointContract(
            id=tp.id, label=tp.label, week_offset=tp.week_offset, modalities=modalities,
            expert_rating=rating, checks=tuple(checks), readiness="unusable",
        )
    checks.append(CheckEntry(code="mask_present", status="pass",
                             detail=f"{tp.mask_source.value if tp.mask_source else '?'} mask resolved"))

    # grid alignment — geometry only, from the fingerprint record shape the check expects
    align = check_mask_grid_alignment(TimepointFingerprint(
        id=tp.id, label=tp.label, week_offset=tp.week_offset, modalities={},
        mask=_geo_fp(tp.mask),
        mask_reference=_geo_fp(tp.mask_reference) if tp.mask_reference is not None else None,
    ))
    checks.append(CheckEntry(code=align.code, status=align.status, detail=align.detail))

    # volumes — the one place voxels are read
    loaded = tp.mask.load()
    labels = np.asarray(loaded.data)
    source = tp.mask_source or MaskSource.DEEPBRATUMIA
    try:
        comps = compartment_volumes(labels, loaded.affine, source)
        regions = region_volumes(labels, loaded.affine, source)
        checks.append(CheckEntry(code="label_schema_valid", status="pass",
                                 detail="all mask integers are documented in label_schema.py"))
    except ValueError as exc:
        checks.append(CheckEntry(code="label_schema_valid", status="fail", detail=str(exc)))
        return TimepointContract(
            id=tp.id, label=tp.label, week_offset=tp.week_offset, modalities=modalities,
            mask=_image_entry(tp.mask),
            mask_source=source.value, expert_rating=rating,
            checks=tuple(checks), readiness="unusable",
        )

    checks.append(CheckEntry(
        code="expert_rating_present",
        status="pass" if rating in RESPONSE_LABELS else "warn",
        detail=f"expert RANO rating: {rating}" if rating else "no expert rating for this timepoint",
    ))

    if any(c.status == "fail" for c in checks if c.code != "modalities_present"):
        readiness = "unusable"
    elif any(c.status in ("fail", "warn") for c in checks):
        readiness = "needs_attention"
    else:
        readiness = "usable"

    to_entry = lambda v: VolumeEntry(volume_mm3=round(v, 2), uncertainty_pp=uncertainty_pp(v))
    return TimepointContract(
        id=tp.id, label=tp.label, week_offset=tp.week_offset, modalities=modalities,
        mask=_image_entry(tp.mask), mask_source=source.value,
        compartment_volumes={k: to_entry(v) for k, v in comps.items()},
        region_volumes={k: to_entry(v) for k, v in regions.items()},
        expert_rating=rating, checks=tuple(checks), readiness=readiness,
    )


def build_patient(patient: Patient, ratings) -> PatientContract:
    tps = tuple(build_timepoint(tp, ratings.get((patient.id, tp.label))) for tp in patient)
    n_usable = sum(1 for t in tps if t.readiness != "unusable")
    n_assessable = sum(1 for t in tps if t.is_assessable)
    if n_assessable >= 2:
        readiness = "usable"
    elif n_usable:
        readiness = "needs_attention"
    else:
        readiness = "unusable"
    return PatientContract(
        patient_id=patient.id, n_timepoints=len(tps), n_usable=n_usable,
        n_assessable=n_assessable, is_longitudinal=patient.is_longitudinal,
        timepoints=tps, readiness=readiness,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", default=str(DEFAULT_ZIP))
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--ratings", default=str(DEFAULT_RATINGS))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--n", type=int, default=3, help="first N patients (default 3)")
    ap.add_argument("--all", action="store_true", help="every patient in the manifest")
    ap.add_argument("--patients", nargs="+")
    args = ap.parse_args()

    adapter = LumiereAdapter(args.zip, args.manifest)
    ratings = load_ratings(Path(args.ratings))
    ids = args.patients or (adapter.patient_ids() if args.all else adapter.patient_ids()[: args.n])

    t0 = time.time()
    patients = []
    for i, pid in enumerate(ids, 1):
        patients.append(build_patient(adapter.load_patient(pid), ratings))
        print(f"[{i:>3}/{len(ids)}] {pid}", file=sys.stderr)

    contract = DataContract(
        provenance=DataContract.provenance_now(
            source_archive=Path(args.zip).name, manifest=Path(args.manifest).name,
            adapter=adapter.name, mask_source=MaskSource.DEEPBRATUMIA.value,
        ),
        patients=tuple(patients),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(contract.model_dump_json(indent=2))

    tot = sum(p.n_timepoints for p in patients)
    print(f"\n{len(patients)} patients / {tot} timepoints in {time.time() - t0:.1f}s")
    print(f"  usable timepoints    : {sum(p.n_usable for p in patients)}")
    print(f"  assessable (seg+RANO): {sum(p.n_assessable for p in patients)}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
