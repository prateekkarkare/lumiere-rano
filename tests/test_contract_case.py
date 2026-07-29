"""Tests for the internal case contract: Space, Geometry, ImageRef, Timepoint, Patient."""

from __future__ import annotations

import numpy as np
import pytest

from rano.contract.case import (
    Geometry,
    ImageRef,
    LoadedImage,
    MaskSource,
    Modality,
    Patient,
    Space,
    SpaceTag,
    Timepoint,
)


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


# ---------------------------------------------------------------------------- Space
def test_space_equality_by_tag_and_key():
    assert Space.mni152_1mm() == Space.mni152_1mm()
    assert Space.native("A") != Space.native("B"), "two NATIVE spaces must not be presumed aligned"
    assert Space.native("A") != Space.mni152_1mm()


# ---------------------------------------------------------------------------- Geometry
def test_geometry_from_header_isotropic():
    affine = np.eye(4)
    g = Geometry.from_header((10, 12, 10), affine, np.dtype("uint8"))
    assert g.shape == (10, 12, 10)
    assert g.spacing == (1.0, 1.0, 1.0)
    assert g.orientation == "RAS"
    assert g.is_isotropic
    assert g.voxel_volume_mm3 == 1.0
    assert g.n_voxels == 10 * 12 * 10


def test_geometry_from_header_anisotropic():
    affine = np.diag([0.9, 0.9, 6.0, 1.0])
    g = Geometry.from_header((256, 256, 30), affine, np.dtype("float32"))
    assert not g.is_isotropic
    assert g.anisotropy_ratio == pytest.approx(6.0 / 0.9)
    assert g.voxel_volume_mm3 == pytest.approx(0.9 * 0.9 * 6.0)


def test_geometry_hashable_despite_affine_array():
    """affine is excluded from eq/hash (compare=False) so Geometry can live in a set."""
    g1 = Geometry.from_header((4, 4, 4), np.eye(4), np.dtype("uint8"))
    g2 = Geometry.from_header((4, 4, 4), np.eye(4) * 1.0, np.dtype("uint8"))  # same fields
    g3 = Geometry.from_header((8, 8, 8), np.eye(4), np.dtype("uint8"))
    assert g1 == g2
    assert len({g1, g2, g3}) == 2


# ---------------------------------------------------------------------------- ImageRef
def test_imageref_is_abstract():
    with pytest.raises(TypeError):
        ImageRef()  # type: ignore[abstract]


def test_fake_ref_geometry_is_header_only_and_load_returns_data():
    arr = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    ref = _FakeRef(arr, affine, Space.native("p1"))
    g = ref.geometry
    assert g.shape == (2, 3, 4)
    assert g.spacing == (2.0, 2.0, 2.0)
    loaded = ref.load()
    assert np.array_equal(loaded.data, arr)


# ---------------------------------------------------------------------------- Timepoint
def _make_ref(space: Space) -> _FakeRef:
    return _FakeRef(np.zeros((4, 4, 4), np.uint8), np.eye(4), space)


def test_timepoint_modality_accessors():
    flair = _make_ref(Space.native("p1-flair"))
    tp = Timepoint(
        id="p1/w0", label="week-000", week_offset=0.0,
        modalities={Modality.FLAIR: flair},
    )
    assert tp.has_modality(Modality.FLAIR)
    assert not tp.has_modality(Modality.T1)
    assert tp.available_modalities == frozenset({Modality.FLAIR})
    assert tp.get(Modality.FLAIR) is flair
    assert tp.get(Modality.T1) is None
    assert not tp.has_mask


def test_timepoint_mask_and_reference_are_independent_spaces():
    mask = _make_ref(Space.mni152_1mm())
    ref = _make_ref(Space.mni152_1mm())
    native_flair = _make_ref(Space.native("p1-flair"))
    tp = Timepoint(
        id="p1/w0", label="week-000", week_offset=0.0,
        modalities={Modality.FLAIR: native_flair},
        mask=mask, mask_source=MaskSource.DEEPBRATUMIA, mask_reference=ref,
    )
    assert tp.has_mask
    assert tp.mask_reference.space == tp.mask.space
    assert tp.get(Modality.FLAIR).space != tp.mask.space, (
        "native FLAIR and the MNI mask must never be treated as the same grid"
    )


# ---------------------------------------------------------------------------- Patient
def _tp(label: str, week_offset: float | None) -> Timepoint:
    return Timepoint(id=f"p/{label}", label=label, week_offset=week_offset, modalities={})


def test_patient_accepts_chronological_order():
    p = Patient("P001", (_tp("week-000", 0.0), _tp("week-044", 44.0)))
    assert p.n_timepoints == 2
    assert p.baseline.label == "week-000"
    assert list(p) == list(p.timepoints)


def test_patient_rejects_out_of_order_timepoints():
    with pytest.raises(ValueError, match="not in chronological order"):
        Patient("P002", (_tp("week-044", 44.0), _tp("week-000", 0.0)))


def test_patient_allows_unparseable_timepoints_to_trail():
    # None week_offsets are excluded from the ordering check entirely (not equality-compared).
    p = Patient("P003", (_tp("week-000", 0.0), _tp("weird-label", None)))
    assert p.n_timepoints == 2


def test_patient_longitudinal_flag():
    single = Patient("P004", (_tp("week-000", 0.0),))
    multi = Patient("P005", (_tp("week-000", 0.0), _tp("week-010", 10.0)))
    assert not single.is_longitudinal
    assert multi.is_longitudinal


def test_patient_empty_has_no_baseline():
    p = Patient("P006", ())
    assert p.baseline is None
    assert p.n_timepoints == 0
