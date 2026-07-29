"""
Shared fixtures: a small synthetic LUMIERE-shaped zip + manifest, so tests never touch the real
32 GB archive. Fixture NIfTI bytes are built with ``nib.Nifti1Image.to_bytes()`` + ``gzip.compress``,
verified (see session notes) to round-trip identically through ``ZipNiftiRef``'s decode path
(``gzip.GzipFile`` + ``nib.FileHolder`` + ``from_file_map``).

Cohort shape, deliberately mirroring real-archive quirks we've already found in LUMIERE:
  * Patient-001: two timepoints; week-010 is missing T2 and the mask (real cohorts have gaps).
  * Patient-002: a single timepoint, sparse (only CT1) — exercises N==1 / non-longitudinal.
  * Patient-003: manifest lists week-020 before week-005 — tests chronological re-sorting.
  * Patient-999: manifest claims a fully-present timepoint that has ZERO members in the zip —
    mirrors the real Patient-025 manifest-vs-archive label mismatch (see project memory).
"""

from __future__ import annotations

import gzip
import zipfile

import nibabel as nib
import numpy as np
import pytest

from rano.adapters.lumiere import paths
from rano.contract.case import Modality

NATIVE_SHAPE = (16, 16, 4)
NATIVE_AFFINE = np.diag([0.9, 0.9, 5.0, 1.0])  # anisotropic, like real native LUMIERE grids

ATLAS_SHAPE = (8, 10, 8)
ATLAS_AFFINE = np.eye(4)  # 1mm isotropic-like grid standing in for the DeepBraTumIA atlas space


def _gz_nifti(shape, affine, dtype, data: np.ndarray | None = None) -> bytes:
    arr = np.zeros(shape, dtype=dtype) if data is None else data
    return gzip.compress(nib.Nifti1Image(arr, affine).to_bytes())


def _mask_array(shape) -> np.ndarray:
    """A tiny mask with a few voxels of each DeepBraTumIA label {1,2,3}, rest background."""
    arr = np.zeros(shape, dtype=np.uint8)
    arr[0, 0, 0] = 1
    arr[0, 0, 1] = 2
    arr[0, 0, 2] = 3
    return arr


@pytest.fixture
def lumiere_fixture(tmp_path) -> tuple[str, str]:
    """Returns (zip_path, manifest_csv_path) for a small synthetic LUMIERE-shaped cohort."""
    zip_path = tmp_path / "imaging.zip"
    manifest_path = tmp_path / "manifest.csv"
    members: dict[str, bytes] = {}

    # --- Patient-001 / week-000: full modalities + mask + reference + brain mask ---
    for mod in (Modality.CT1, Modality.T1, Modality.T2, Modality.FLAIR):
        members[paths.raw_image("Patient-001", "week-000", mod)] = _gz_nifti(
            NATIVE_SHAPE, NATIVE_AFFINE, np.int16
        )
    members[paths.dbt_mask("Patient-001", "week-000")] = _gz_nifti(
        ATLAS_SHAPE, ATLAS_AFFINE, np.uint8, _mask_array(ATLAS_SHAPE)
    )
    members[paths.dbt_skull_strip("Patient-001", "week-000", Modality.CT1)] = _gz_nifti(
        ATLAS_SHAPE, ATLAS_AFFINE, np.int16
    )
    members[paths.dbt_brain_mask("Patient-001", "week-000")] = _gz_nifti(
        ATLAS_SHAPE, ATLAS_AFFINE, np.uint8
    )

    # --- Patient-001 / week-010: follow-up, missing T2 and the mask ---
    for mod in (Modality.CT1, Modality.T1, Modality.FLAIR):
        members[paths.raw_image("Patient-001", "week-010", mod)] = _gz_nifti(
            NATIVE_SHAPE, NATIVE_AFFINE, np.int16
        )

    # --- Patient-002 / week-000: single timepoint, sparse (only CT1) ---
    members[paths.raw_image("Patient-002", "week-000", Modality.CT1)] = _gz_nifti(
        NATIVE_SHAPE, NATIVE_AFFINE, np.int16
    )

    # --- Patient-003: two timepoints, only CT1, to test chronological re-sorting ---
    for tp in ("week-020", "week-005"):
        members[paths.raw_image("Patient-003", tp, Modality.CT1)] = _gz_nifti(
            NATIVE_SHAPE, NATIVE_AFFINE, np.int16
        )

    # --- Patient-999 / week-005: manifest claims everything; zip has NOTHING (P-025-style) ---
    # (deliberately: no members written for this patient/timepoint)

    with zipfile.ZipFile(zip_path, "w") as zf:
        for member, content in members.items():
            zf.writestr(member, content)

    rows = [
        "Patient,Timepoint,CT1,T1,T2,FLAIR,DeepBraTumIA,HD-GLIO-AUTO",
        "Patient-001,week-000,x,x,x,x,x,",
        "Patient-001,week-010,x,x,,x,,",
        "Patient-002,week-000,x,,,,,",
        "Patient-003,week-020,x,,,,,",
        "Patient-003,week-005,x,,,,,",
        "Patient-999,week-005,x,x,x,x,x,",
    ]
    manifest_path.write_text("\n".join(rows) + "\n")

    return str(zip_path), str(manifest_path)
