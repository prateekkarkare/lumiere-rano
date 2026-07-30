"""
Integration: reproduce DeepBraTumIA's own shipped volumes from the real archive.

This is the end-to-end proof that the label decode AND the arithmetic are both right. The atlas
voxel is exactly 1.000 mm3, so ``measured_volumes_in_mm3.json`` is literally a voxel count —
which makes it an exact, not approximate, target. A label swap or a voxel-volume error would
both show up here immediately.

Skipped automatically when the 32 GB archive isn't present, so the suite still runs anywhere.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rano.adapters.lumiere import paths
from rano.adapters.lumiere.adapter import LumiereAdapter
from rano.adapters.lumiere.zip_ref import ZipSource
from rano.contract.case import MaskSource, Modality
from rano.volumetry.volumes import compartment_volumes, voxel_volume_mm3

ROOT = Path(__file__).resolve().parent.parent
ZIP = ROOT / "Imaging-v202211.zip"
MANIFEST = ROOT / "LUMIERE-datacompleteness.csv"

pytestmark = pytest.mark.skipif(
    not (ZIP.exists() and MANIFEST.exists()), reason="real LUMIERE archive not present"
)

#: JSON key -> the compartment name our schema gives the same tissue
JSON_TO_COMPARTMENT = {
    "Enhancing_Core": "enhancing",
    "Necrotic_NonEnhancing": "necrosis_nonenhancing",
    "Edema_Compartment": "edema",
}

CASES = [("Patient-001", "week-000-1"), ("Patient-002", "week-000"), ("Patient-003", "week-000-2")]


@pytest.fixture(scope="module")
def src() -> ZipSource:
    return ZipSource(str(ZIP))


@pytest.mark.parametrize("patient,tp", CASES)
def test_reproduces_shipped_volumes_exactly(src, patient, tp):
    """Our atlas-space volumes must equal the shipped JSON to the millimetre."""
    img = src.open_nifti(paths.dbt_mask(patient, tp))
    labels = np.asanyarray(img.dataobj)
    shipped = json.loads(src.read(paths.dbt_volumes_json(patient, tp)))

    ours = compartment_volumes(labels, img.affine, MaskSource.DEEPBRATUMIA)
    for json_key, compartment in JSON_TO_COMPARTMENT.items():
        assert ours[compartment] == pytest.approx(shipped[json_key], abs=1e-6), (
            f"{patient}/{tp}: {compartment} disagrees with shipped {json_key}"
        )


def test_atlas_voxel_is_exactly_one_cubic_mm(src):
    """The reason the check above can be exact rather than approximate."""
    img = src.open_nifti(paths.dbt_mask(*CASES[0]))
    assert voxel_volume_mm3(img.affine) == pytest.approx(1.0, abs=1e-9)


def test_native_masks_agree_with_atlas_for_a_large_compartment(src):
    """The audit's core claim, as a test: rigid transforms preserve volume.

    Restricted to a LARGE compartment, where resampling noise is small (see docs/volume_audit.html:
    spread is driven by compartment size). 5% is a deliberately loose bound — this asserts "no
    gross disagreement", not the tolerance itself, which the audit report quantifies properly.
    """
    patient, tp = "Patient-001", "week-000-1"
    atlas_img = src.open_nifti(paths.dbt_mask(patient, tp))
    atlas = compartment_volumes(
        np.asanyarray(atlas_img.dataobj), atlas_img.affine, MaskSource.DEEPBRATUMIA
    )["edema"]
    assert atlas > 20000  # sanity: this really is a large compartment

    for modality in (Modality.CT1, Modality.T1, Modality.T2, Modality.FLAIR):
        member = paths.dbt_native_mask(patient, tp, modality)
        assert src.exists(member), f"native mask missing for {modality.value}"
        img = src.open_nifti(member)
        native = compartment_volumes(
            np.asanyarray(img.dataobj), img.affine, MaskSource.DEEPBRATUMIA
        )["edema"]
        assert native == pytest.approx(atlas, rel=0.05), (
            f"{modality.value} native edema {native:.0f} vs atlas {atlas:.0f} mm3"
        )


def test_every_native_grid_is_wired_and_readable():
    """The adapter's path builders resolve for a real case, all four sequences."""
    adapter = LumiereAdapter(str(ZIP), str(MANIFEST))
    patient = adapter.load_patient("Patient-001")
    tp = patient.timepoints[0]
    for modality in (Modality.CT1, Modality.T1, Modality.T2, Modality.FLAIR):
        assert adapter._src.exists(paths.dbt_native_mask(patient.id, tp.label, modality))
        assert adapter._src.exists(paths.dbt_native_transform(patient.id, tp.label, modality))
