"""
Atlas-vs-native volumetry audit — compute every compartment volume TWICE (or five times).

For each timepoint that has a DeepBraTumIA mask, this computes compartment volumes from:
  * the ATLAS mask   (MNI 1mm isotropic, one grid for the whole cohort), and
  * the FOUR NATIVE masks (the same segmentation back-transformed onto each sequence's own
    acquisition grid).

The transforms are rigid (verified: all 2,396 .tfm have det == 1.0 to 1e-10 and are orthonormal
to 1.5e-15), so registration CANNOT rescale volume. Any atlas-vs-native difference is therefore
attributable to nearest-neighbour resampling of a categorical mask between grids of different
resolution — plus, in principle, clipping at the atlas field of view, which is measured here as
``atlas_boundary_voxels``.

Writes one tidy CSV row per (patient, timepoint, space, compartment) to
``output/volume_audit/volumes.csv``. Rendering lives in ``render_volume_audit.py`` — this script
only measures.

Usage:
    .venv/bin/python scripts/audit_atlas_native_volumes.py                 # all patients
    .venv/bin/python scripts/audit_atlas_native_volumes.py --n 8           # first 8 patients
    .venv/bin/python scripts/audit_atlas_native_volumes.py --patients Patient-001 Patient-002
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

from rano.adapters.lumiere.adapter import LumiereAdapter
from rano.adapters.lumiere.paths import dbt_native_mask
from rano.adapters.lumiere.zip_ref import ZipNiftiRef
from rano.contract.case import MaskSource, Modality, Space
from rano.volumetry.volumes import compartment_volumes, voxel_volume_mm3

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ZIP = ROOT / "Imaging-v202211.zip"
DEFAULT_MANIFEST = ROOT / "LUMIERE-datacompleteness.csv"
DEFAULT_OUT = ROOT / "output" / "volume_audit"

FIELDS = [
    "patient", "timepoint", "week_offset", "space", "modality",
    "shape", "spacing_mm", "voxel_mm3", "anisotropy_ratio",
    "compartment", "volume_mm3", "n_voxels", "atlas_boundary_voxels",
]


def _anisotropy(affine: np.ndarray) -> tuple[tuple[float, float, float], float]:
    spacing = tuple(float(np.linalg.norm(affine[:3, i])) for i in range(3))
    lo = min(spacing)
    return spacing, (max(spacing) / lo if lo > 0 else float("inf"))


def _boundary_voxels(labels: np.ndarray) -> int:
    """Non-zero mask voxels touching the volume's outer face — the FOV-clipping tell.

    A tumour mask should never reach the edge of the atlas box. If it does, volume was lost to
    the field of view and every number for that case is a lower bound.
    """
    faces = 0
    for axis in range(labels.ndim):
        for index in (0, -1):
            faces += int(np.count_nonzero(np.take(labels, index, axis=axis)))
    return faces


def _rows_for_mask(
    ref, *, patient: str, tp_label: str, week, space: str, modality: str, is_atlas: bool
) -> list[dict]:
    loaded = ref.load()
    labels = np.asarray(loaded.data)
    affine = np.asarray(loaded.affine, dtype=float)

    spacing, aniso = _anisotropy(affine)
    vox = voxel_volume_mm3(affine)
    volumes = compartment_volumes(labels, affine, MaskSource.DEEPBRATUMIA)
    boundary = _boundary_voxels(labels) if is_atlas else ""

    return [
        {
            "patient": patient,
            "timepoint": tp_label,
            "week_offset": week if week is not None else "",
            "space": space,
            "modality": modality,
            "shape": "x".join(str(s) for s in labels.shape),
            "spacing_mm": "/".join(f"{s:.4f}" for s in spacing),
            "voxel_mm3": f"{vox:.6f}",
            "anisotropy_ratio": f"{aniso:.4f}",
            "compartment": name,
            "volume_mm3": f"{volume:.3f}",
            "n_voxels": int(round(volume / vox)) if vox > 0 else 0,
            "atlas_boundary_voxels": boundary,
        }
        for name, volume in volumes.items()
    ]


def audit_patient(adapter: LumiereAdapter, patient_id: str) -> tuple[list[dict], list[str]]:
    """Every atlas + native compartment volume for one patient. Returns (rows, warnings)."""
    patient = adapter.load_patient(patient_id)
    src = adapter._src  # the adapter owns the ZipSource; native masks aren't on the contract yet
    rows: list[dict] = []
    warnings: list[str] = []

    for tp in patient:
        if tp.mask is None:
            warnings.append(f"{patient_id}/{tp.label}: no atlas mask; skipped")
            continue
        try:
            rows += _rows_for_mask(
                tp.mask, patient=patient_id, tp_label=tp.label, week=tp.week_offset,
                space="atlas", modality="atlas", is_atlas=True,
            )
        except ValueError as exc:  # off-schema label -> record, never silently drop
            warnings.append(f"{patient_id}/{tp.label} atlas: {exc}")
            continue

        for modality in (Modality.CT1, Modality.T1, Modality.T2, Modality.FLAIR):
            member = dbt_native_mask(patient_id, tp.label, modality)
            if not src.exists(member):
                warnings.append(f"{patient_id}/{tp.label}: native {modality.value} mask absent")
                continue
            ref = ZipNiftiRef(src, member, Space.native(member))
            try:
                rows += _rows_for_mask(
                    ref, patient=patient_id, tp_label=tp.label, week=tp.week_offset,
                    space="native", modality=modality.value, is_atlas=False,
                )
            except ValueError as exc:
                warnings.append(f"{patient_id}/{tp.label} native {modality.value}: {exc}")

    return rows, warnings


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--zip", default=str(DEFAULT_ZIP))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--n", type=int, help="audit only the first N patients")
    parser.add_argument("--patients", nargs="+", help="explicit patient IDs, overrides --n")
    args = parser.parse_args()

    adapter = LumiereAdapter(args.zip, args.manifest)
    ids = args.patients or adapter.patient_ids()
    if args.patients is None and args.n:
        ids = ids[: args.n]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "volumes.csv"

    all_warnings: list[str] = []
    t0 = time.time()
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for i, pid in enumerate(ids, 1):
            rows, warnings = audit_patient(adapter, pid)
            writer.writerows(rows)
            all_warnings += warnings
            print(f"[{i:>3}/{len(ids)}] {pid:<14} {len(rows):>4} rows"
                  f"{'  (' + str(len(warnings)) + ' warnings)' if warnings else ''}",
                  file=sys.stderr)

    if all_warnings:
        (out_dir / "warnings.txt").write_text("\n".join(all_warnings) + "\n")

    print(f"\n{len(ids)} patients in {time.time() - t0:.1f}s -> {csv_path}")
    print(f"{len(all_warnings)} warnings -> {out_dir / 'warnings.txt'}" if all_warnings else "no warnings")


if __name__ == "__main__":
    main()
