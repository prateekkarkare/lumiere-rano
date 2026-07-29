"""
LUMIERE archive member-path builders — the single source of truth for the zip layout.

Every path here was verified against the real ``Imaging-v202211.zip`` central directory
(Patient-001/week-000-1). If the archive layout ever changes, this is the ONE file to touch.

Layout, per ``Imaging/<patient>/<timepoint>/``:
    <SEQ>.nii.gz                                              raw skull-stripped, NATIVE  (SEQ upper)
    DeepBraTumIA-segmentation/atlas/segmentation/
        seg_mask.nii.gz                                      canonical mask, MNI 1mm
        measured_volumes_in_mm3.json                         shipped volumes (volumetry cross-check)
    DeepBraTumIA-segmentation/atlas/skull_strip/
        <seq>_skull_strip.nii.gz                             image in mask space, MNI  (seq lower)
        brain_mask.nii.gz                                    brain mask, MNI
    HD-GLIO-AUTO-segmentation/registered/segmentation.nii.gz second-opinion mask (QC only)
"""

from __future__ import annotations

from rano.contract.case import Modality

#: manifest column / raw filename uses UPPER-case sequence names
RAW_SEQ: dict[Modality, str] = {
    Modality.CT1: "CT1",
    Modality.T1: "T1",
    Modality.T2: "T2",
    Modality.FLAIR: "FLAIR",
}

#: DeepBraTumIA skull-strip filenames use lower-case sequence names
SS_SEQ: dict[Modality, str] = {
    Modality.CT1: "ct1",
    Modality.T1: "t1",
    Modality.T2: "t2",
    Modality.FLAIR: "flair",
}

_TP = "Imaging/{p}/{tp}"
_DBT = _TP + "/DeepBraTumIA-segmentation/atlas"


def raw_image(patient: str, tp: str, modality: Modality) -> str:
    return f"{_TP.format(p=patient, tp=tp)}/{RAW_SEQ[modality]}.nii.gz"


def dbt_mask(patient: str, tp: str) -> str:
    return f"{_DBT.format(p=patient, tp=tp)}/segmentation/seg_mask.nii.gz"


def dbt_volumes_json(patient: str, tp: str) -> str:
    return f"{_DBT.format(p=patient, tp=tp)}/segmentation/measured_volumes_in_mm3.json"


def dbt_skull_strip(patient: str, tp: str, modality: Modality) -> str:
    return f"{_DBT.format(p=patient, tp=tp)}/skull_strip/{SS_SEQ[modality]}_skull_strip.nii.gz"


def dbt_brain_mask(patient: str, tp: str) -> str:
    return f"{_DBT.format(p=patient, tp=tp)}/skull_strip/brain_mask.nii.gz"


def hdglio_mask(patient: str, tp: str) -> str:
    return f"{_TP.format(p=patient, tp=tp)}/HD-GLIO-AUTO-segmentation/registered/segmentation.nii.gz"
