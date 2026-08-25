"""Unit tests for fingerprint/extractor.py against controlled in-memory arrays."""

from __future__ import annotations

import math

import numpy as np
import pytest

from rano.contract.case import Geometry, ImageRef, LoadedImage, MaskSource, Modality, Patient, Space, Timepoint
from rano.fingerprint.extractor import fingerprint_patient, fingerprint_timepoint


class _FakeRef(ImageRef):
    """Minimal in-memory ImageRef, standing in for a real adapter's concrete ref."""

    def __init__(self, arr: np.ndarray, affine: np.ndarray, space: Space, source: str = "fake"):
        self._arr, self._affine, self._space, self._source = arr, affine, space, source

    @property
    def source(self) -> str:
        return self._source

    @property
    def space(self) -> Space:
        return self._space

    @property
    def geometry(self) -> Geometry:
        return Geometry.from_header(self._arr.shape, self._affine, self._arr.dtype)

    def load(self) -> LoadedImage:
        return LoadedImage(self._arr, self._affine, header=None)


def _ref(arr: np.ndarray, space: Space, affine: np.ndarray | None = None) -> _FakeRef:
    return _FakeRef(arr, np.eye(4) if affine is None else affine, space)


# ---------------------------------------------------------------------- native modality basis
def test_native_modality_intensity_is_nonzero_basis_only():
    arr = np.zeros((4, 4, 4), dtype=np.int16)
    arr[0, 0, 0] = 10
    arr[0, 0, 1] = 20
    tp = Timepoint(
        id="p/w0", label="week-000", week_offset=0.0,
        modalities={Modality.CT1: _ref(arr, Space.native("p-ct1"))},
    )
    fp = fingerprint_timepoint(tp)
    ct1 = fp.modalities["CT1"]
    assert ct1.role == "modality:CT1"
    assert ct1.intensity.n_voxels == 2, "the 62 zero (skull-strip background) voxels must be excluded"
    assert ct1.intensity.mean == pytest.approx(15.0)
    assert ct1.intensity.min == pytest.approx(10.0)
    assert ct1.intensity.max == pytest.approx(20.0)
    assert ct1.label_histogram is None
    assert ct1.compartment_intensity is None


def test_all_zero_native_image_yields_nan_stats_not_a_crash():
    arr = np.zeros((3, 3, 3), dtype=np.int16)
    tp = Timepoint(
        id="p/w0", label="week-000", week_offset=0.0,
        modalities={Modality.FLAIR: _ref(arr, Space.native("p-flair"))},
    )
    fp = fingerprint_timepoint(tp)
    stats = fp.modalities["FLAIR"].intensity
    assert stats.n_voxels == 0
    assert math.isnan(stats.mean)


# ---------------------------------------------------------------------- mask: label histogram
def test_mask_label_histogram_counts_every_integer():
    arr = np.zeros((2, 2, 2), dtype=np.uint8)  # 8 voxels
    arr[0, 0, 0] = 1
    arr[0, 0, 1] = 2
    arr[0, 1, 0] = 2
    tp = Timepoint(
        id="p/w0", label="week-000", week_offset=0.0, modalities={},
        mask=_ref(arr, Space.mni152_1mm()), mask_source=MaskSource.DEEPBRATUMIA,
    )
    fp = fingerprint_timepoint(tp)
    counts = {lc.label: lc.n_voxels for lc in fp.mask.label_histogram}
    assert counts == {0: 5, 1: 1, 2: 2}
    assert fp.mask.intensity is None, "mask is categorical, not an intensity image"
    assert fp.mask_source == "DeepBraTumIA"


# ---------------------------------------------------------------------- mask_reference basis
def test_mask_reference_intensity_restricted_to_brain_mask():
    ref_arr = np.array([[[5, 5], [100, 100]], [[5, 5], [100, 100]]], dtype=np.int16)
    brain = np.array([[[1, 1], [0, 0]], [[1, 1], [0, 0]]], dtype=np.uint8)  # keep the "5" voxels only
    mni = Space.mni152_1mm()
    tp = Timepoint(
        id="p/w0", label="week-000", week_offset=0.0, modalities={},
        mask=None, mask_reference=_ref(ref_arr, mni), brain_mask=_ref(brain, mni),
    )
    fp = fingerprint_timepoint(tp)
    stats = fp.mask_reference.intensity
    assert stats.n_voxels == 4
    assert stats.mean == pytest.approx(5.0)
    assert fp.mask_reference.compartment_intensity is None, "no mask present, nothing to break out by"


def test_mask_reference_falls_back_to_nonzero_basis_without_brain_mask():
    ref_arr = np.zeros((2, 2, 2), dtype=np.int16)
    ref_arr[0, 0, 0] = 7
    mni = Space.mni152_1mm()
    tp = Timepoint(
        id="p/w0", label="week-000", week_offset=0.0, modalities={},
        mask_reference=_ref(ref_arr, mni),
    )
    fp = fingerprint_timepoint(tp)
    assert fp.mask_reference.intensity.n_voxels == 1
    assert fp.mask_reference.intensity.mean == pytest.approx(7.0)


def test_mask_reference_compartment_intensity_per_label_including_background():
    ref_arr = np.array([[[1, 1], [2, 2]], [[1, 1], [2, 2]]], dtype=np.int16)
    mask_arr = np.array([[[0, 0], [3, 3]], [[0, 0], [3, 3]]], dtype=np.uint8)
    mni = Space.mni152_1mm()
    tp = Timepoint(
        id="p/w0", label="week-000", week_offset=0.0, modalities={},
        mask=_ref(mask_arr, mni), mask_source=MaskSource.DEEPBRATUMIA,
        mask_reference=_ref(ref_arr, mni),
    )
    fp = fingerprint_timepoint(tp)
    by_label = {ci.label: ci.stats for ci in fp.mask_reference.compartment_intensity}
    assert set(by_label) == {0, 3}
    assert by_label[0].mean == pytest.approx(1.0)
    assert by_label[3].mean == pytest.approx(2.0)


def test_mask_reference_raises_on_shape_mismatch_with_mask():
    """A mask/mask_reference grid mismatch is an extraction failure, not silently swallowed."""
    ref_arr = np.zeros((2, 2, 2), dtype=np.int16)
    mask_arr = np.zeros((3, 3, 3), dtype=np.uint8)
    mni = Space.mni152_1mm()
    tp = Timepoint(
        id="p/w0", label="week-000", week_offset=0.0, modalities={},
        mask=_ref(mask_arr, mni), mask_source=MaskSource.DEEPBRATUMIA,
        mask_reference=_ref(ref_arr, mni),
    )
    with pytest.raises((ValueError, IndexError)):
        fingerprint_timepoint(tp)


# ---------------------------------------------------------------------- brain_mask
def test_brain_mask_gets_geometry_only():
    arr = np.ones((2, 2, 2), dtype=np.uint8)
    tp = Timepoint(
        id="p/w0", label="week-000", week_offset=0.0, modalities={},
        brain_mask=_ref(arr, Space.mni152_1mm()),
    )
    fp = fingerprint_timepoint(tp)
    assert fp.brain_mask.role == "brain_mask"
    assert fp.brain_mask.intensity is None
    assert fp.brain_mask.label_histogram is None


# ---------------------------------------------------------------------- absent slots
def test_missing_mask_and_mask_reference_stay_none():
    tp = Timepoint(id="p/w0", label="week-000", week_offset=0.0, modalities={})
    fp = fingerprint_timepoint(tp)
    assert fp.mask is None
    assert fp.mask_reference is None
    assert fp.brain_mask is None
    assert fp.mask_source is None
    assert fp.modalities == {}


# ---------------------------------------------------------------------- patient nesting
def test_fingerprint_patient_nests_every_timepoint():
    tp0 = Timepoint(
        id="P/w0", label="week-000", week_offset=0.0,
        modalities={Modality.CT1: _ref(np.ones((2, 2, 2), np.int16), Space.native("p-ct1"))},
    )
    tp1 = Timepoint(id="P/w10", label="week-010", week_offset=10.0, modalities={})
    patient = Patient(id="P", timepoints=(tp0, tp1))

    record = fingerprint_patient(patient)
    assert record.patient_id == "P"
    assert record.n_timepoints == 2
    assert [tp.label for tp in record.timepoints] == ["week-000", "week-010"]
    assert record.timepoints[0].modalities["CT1"].intensity.n_voxels == 8
