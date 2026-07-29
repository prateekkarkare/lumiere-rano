"""Tests for ZipSource / ZipNiftiRef — streaming NIfTI decode straight out of a zip."""

from __future__ import annotations

import numpy as np
import pytest

from rano.adapters.lumiere.zip_ref import ZipNiftiRef, ZipSource
from rano.contract.case import Space


def test_zipsource_indexes_members_and_reports_existence(lumiere_fixture):
    zip_path, _ = lumiere_fixture
    src = ZipSource(zip_path)
    assert src.exists("Imaging/Patient-001/week-000/CT1.nii.gz")
    assert not src.exists("Imaging/Patient-001/week-000/NOPE.nii.gz")
    assert len(src.names) > 0


def test_zip_niftiref_geometry_and_load_roundtrip(lumiere_fixture):
    zip_path, _ = lumiere_fixture
    src = ZipSource(zip_path)
    member = "Imaging/Patient-001/week-000/CT1.nii.gz"
    ref = ZipNiftiRef(src, member, Space.native("p1-ct1"))

    g = ref.geometry
    assert g.shape == (16, 16, 4)
    assert g.spacing == pytest.approx((0.9, 0.9, 5.0))
    assert not g.is_isotropic

    loaded = ref.load()
    assert loaded.data.shape == (16, 16, 4)
    assert np.allclose(loaded.affine, np.diag([0.9, 0.9, 5.0, 1.0]))


def test_zip_niftiref_mask_labels_survive_the_gzip_roundtrip(lumiere_fixture):
    zip_path, _ = lumiere_fixture
    src = ZipSource(zip_path)
    member = "Imaging/Patient-001/week-000/DeepBraTumIA-segmentation/atlas/segmentation/seg_mask.nii.gz"
    ref = ZipNiftiRef(src, member, Space.mni152_1mm())
    data = ref.load().data
    assert set(np.unique(data).tolist()) == {0, 1, 2, 3}


def test_zip_niftiref_geometry_is_cached_per_ref(lumiere_fixture):
    zip_path, _ = lumiere_fixture
    src = ZipSource(zip_path)
    ref = ZipNiftiRef(src, "Imaging/Patient-001/week-000/CT1.nii.gz", Space.native("p1-ct1"))
    g1 = ref.geometry
    g2 = ref.geometry
    assert g1 is g2, "geometry should be cached after the first header read"
