"""
Unit tests for ``rano.volumetry.volumes`` — synthetic masks with counts known by construction.

The two tests that earn their keep here are ``test_voxel_volume_uses_determinant_not_spacing``
(the failure it guards is a silent over-count on a sheared grid) and
``test_enhancing_is_label_1`` (the failure it guards is the label swap this module's schema
carried until 2026-07-30 — a swap is invisible to any check that only inspects the label SET).
"""

from __future__ import annotations

import numpy as np
import pytest

from rano.contract.case import MaskSource
from rano.volumetry.volumes import (
    compartment_volumes,
    label_voxel_counts,
    region_volumes,
    voxel_volume_mm3,
)


def _mask(counts: dict[int, int], shape=(20, 20, 20)) -> np.ndarray:
    """A flat mask containing exactly ``counts[label]`` voxels of each label, rest background."""
    arr = np.zeros(int(np.prod(shape)), dtype=np.uint8)
    i = 0
    for label, n in counts.items():
        arr[i : i + n] = label
        i += n
    return arr.reshape(shape)


# --------------------------------------------------------------------------- voxel volume
def test_voxel_volume_isotropic():
    assert voxel_volume_mm3(np.eye(4)) == pytest.approx(1.0)


def test_voxel_volume_anisotropic():
    affine = np.diag([0.3594, 4.4, 0.3594, 1.0])
    assert voxel_volume_mm3(affine) == pytest.approx(0.3594 * 4.4 * 0.3594)


def test_voxel_volume_is_rotation_invariant():
    """A rigid rotation must not change the physical size of a voxel."""
    theta = 0.7
    rot = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    affine = np.eye(4)
    affine[:3, :3] = rot @ np.diag([0.5, 0.5, 3.0])
    assert voxel_volume_mm3(affine) == pytest.approx(0.5 * 0.5 * 3.0)


def test_voxel_volume_uses_determinant_not_spacing():
    """On a SHEARED grid, prod(column norms) over-counts; the determinant is correct.

    This is the whole reason the implementation doesn't take the ``prod(spacing)`` shortcut.
    """
    affine = np.eye(4)
    affine[:3, :3] = np.array([[1.0, 0.6, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    spacings = [float(np.linalg.norm(affine[:3, i])) for i in range(3)]
    assert np.prod(spacings) == pytest.approx(np.sqrt(1 + 0.36))  # the wrong answer
    assert voxel_volume_mm3(affine) == pytest.approx(1.0)         # the right one


# --------------------------------------------------------------------------- label handling
def test_off_schema_label_raises():
    """An undocumented integer must stop the run, not be silently skipped."""
    with pytest.raises(ValueError, match=r"\[7\]"):
        label_voxel_counts(_mask({1: 5, 7: 3}), MaskSource.DEEPBRATUMIA)


def test_absent_compartment_is_zero_not_missing():
    vols = compartment_volumes(_mask({1: 10}), np.eye(4), MaskSource.DEEPBRATUMIA)
    assert vols["enhancing"] == pytest.approx(10.0)
    assert vols["necrosis_nonenhancing"] == 0.0
    assert vols["edema"] == 0.0


def test_background_excluded_by_default():
    vols = compartment_volumes(_mask({1: 4}), np.eye(4), MaskSource.DEEPBRATUMIA)
    assert "background" not in vols
    assert "background" in compartment_volumes(
        _mask({1: 4}), np.eye(4), MaskSource.DEEPBRATUMIA, include_background=True
    )


def test_enhancing_is_label_1_not_label_2():
    """Regression guard for the 2026-07-30 correction. See label_schema.py's CORRECTION note.

    Verified against the archive three ways: the shipped measured_volumes_in_mm3.json matches
    label 1 to Enhancing_Core in 599/599 masks; label 1 takes up contrast and label 2 does not;
    and the resulting trajectories are clinically coherent.
    """
    vols = compartment_volumes(_mask({1: 7, 2: 11, 3: 13}), np.eye(4), MaskSource.DEEPBRATUMIA)
    assert vols["enhancing"] == pytest.approx(7.0)
    assert vols["necrosis_nonenhancing"] == pytest.approx(11.0)
    assert vols["edema"] == pytest.approx(13.0)


# --------------------------------------------------------------------------- scaling & regions
def test_volumes_scale_with_voxel_size():
    labels = _mask({1: 100})
    fine = compartment_volumes(labels, np.eye(4), MaskSource.DEEPBRATUMIA)
    coarse = compartment_volumes(labels, np.diag([1.0, 1.0, 6.0, 1.0]), MaskSource.DEEPBRATUMIA)
    assert coarse["enhancing"] == pytest.approx(fine["enhancing"] * 6.0)


def test_region_volumes_composites():
    regions = region_volumes(_mask({1: 7, 2: 11, 3: 13}), np.eye(4), MaskSource.DEEPBRATUMIA)
    assert regions["ET"] == pytest.approx(7.0)          # enhancing only == label 1
    assert regions["TC"] == pytest.approx(7.0 + 11.0)   # union, unaffected by the label swap
    assert regions["WT"] == pytest.approx(7.0 + 11.0 + 13.0)


def test_hdglio_schema_still_maps_enhancing_to_label_2():
    """HD-GLIO was re-verified independently and is NOT affected by the DeepBraTumIA swap."""
    vols = compartment_volumes(_mask({1: 5, 2: 9}), np.eye(4), MaskSource.HDGLIO_AUTO)
    assert vols["enhancing"] == pytest.approx(9.0)
    assert vols["t2_flair_nonenhancing"] == pytest.approx(5.0)


def test_unknown_mask_source_raises():
    with pytest.raises(KeyError):
        compartment_volumes(_mask({1: 1}), np.eye(4), "NotATool")
