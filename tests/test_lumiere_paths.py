"""
Tests for the LUMIERE zip member-path builders.

Expected strings are literal, copied from the real ``Imaging-v202211.zip`` central-directory
listing for Patient-001/week-000-1 (verified this session) — NOT derived from ``paths.py``
itself, so a bug that silently drifts the builders from the real archive layout is caught here.
"""

from __future__ import annotations

from rano.adapters.lumiere import paths
from rano.contract.case import Modality


def test_raw_image_paths():
    assert paths.raw_image("Patient-001", "week-000-1", Modality.CT1) == (
        "Imaging/Patient-001/week-000-1/CT1.nii.gz"
    )
    assert paths.raw_image("Patient-001", "week-000-1", Modality.FLAIR) == (
        "Imaging/Patient-001/week-000-1/FLAIR.nii.gz"
    )


def test_dbt_mask_and_volumes_json_paths():
    assert paths.dbt_mask("Patient-001", "week-000-1") == (
        "Imaging/Patient-001/week-000-1/DeepBraTumIA-segmentation/atlas/segmentation/seg_mask.nii.gz"
    )
    assert paths.dbt_volumes_json("Patient-001", "week-000-1") == (
        "Imaging/Patient-001/week-000-1/DeepBraTumIA-segmentation/atlas/segmentation/"
        "measured_volumes_in_mm3.json"
    )


def test_dbt_skull_strip_uses_lowercase_sequence_name():
    assert paths.dbt_skull_strip("Patient-001", "week-000-1", Modality.CT1) == (
        "Imaging/Patient-001/week-000-1/DeepBraTumIA-segmentation/atlas/skull_strip/"
        "ct1_skull_strip.nii.gz"
    )
    assert paths.dbt_skull_strip("Patient-001", "week-000-1", Modality.FLAIR) == (
        "Imaging/Patient-001/week-000-1/DeepBraTumIA-segmentation/atlas/skull_strip/"
        "flair_skull_strip.nii.gz"
    )


def test_dbt_brain_mask_path():
    assert paths.dbt_brain_mask("Patient-001", "week-000-1") == (
        "Imaging/Patient-001/week-000-1/DeepBraTumIA-segmentation/atlas/skull_strip/"
        "brain_mask.nii.gz"
    )


def test_hdglio_mask_path():
    assert paths.hdglio_mask("Patient-001", "week-000-1") == (
        "Imaging/Patient-001/week-000-1/HD-GLIO-AUTO-segmentation/registered/segmentation.nii.gz"
    )
