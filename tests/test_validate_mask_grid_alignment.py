"""
Unit tests for validate/checks/mask_grid_alignment.py against controlled fingerprint records.

This check consumes ``TimepointFingerprint`` (the fingerprinter's already-extracted record),
not the live contract ``Timepoint``/``ImageRef`` -- so these fixtures build plain fingerprint
schema objects directly. There's no ``.load()`` to guard against here: a fingerprint record is
just data, it has no way to touch voxels even if this check wanted it to.
"""

from __future__ import annotations

import pytest

from rano.fingerprint.schema import GeometryFingerprint, ImageFingerprint, TimepointFingerprint
from rano.validate.checks.mask_grid_alignment import (
    DIRECTION_COSINE_ATOL,
    SPACING_ATOL_MM,
    TRANSLATION_ATOL_MM,
    check_mask_grid_alignment,
)

SHAPE = (10, 12, 10)
BASE_AFFINE = (
    (1.0, 0.0, 0.0, 90.0),
    (0.0, 1.0, 0.0, -126.0),
    (0.0, 0.0, 1.0, -72.0),
    (0.0, 0.0, 0.0, 1.0),
)


def _geo(affine=BASE_AFFINE, shape=SHAPE) -> GeometryFingerprint:
    spacing = tuple(sum(row[i] ** 2 for row in affine[:3]) ** 0.5 for i in range(3))
    return GeometryFingerprint(
        shape=shape, spacing=spacing, orientation="RAS", dtype="int16",
        anisotropy_ratio=max(spacing) / min(spacing), affine=affine,
    )


def _image(affine=BASE_AFFINE, shape=SHAPE, role="mask") -> ImageFingerprint:
    return ImageFingerprint(role=role, source="fake", space="mni152_1mm:mni152_1mm", geometry=_geo(affine, shape))


def _tp(mask: ImageFingerprint | None, mask_reference: ImageFingerprint | None) -> TimepointFingerprint:
    return TimepointFingerprint(
        id="p/w0", label="week-000", week_offset=0.0, modalities={},
        mask=mask, mask_reference=mask_reference,
    )


def _perturbed_affine(deltas: dict) -> tuple:
    """BASE_AFFINE with a few {(row, col): delta} overrides applied."""
    rows = [list(row) for row in BASE_AFFINE]
    for (r, c), delta in deltas.items():
        rows[r][c] += delta
    return tuple(tuple(row) for row in rows)


def test_identical_grids_pass():
    tp = _tp(_image(role="mask"), _image(role="mask_reference"))
    result = check_mask_grid_alignment(tp)
    assert result.status == "pass"
    assert result.code == "mask_grid_alignment.ok"


def test_missing_mask_or_reference_is_pass_with_distinct_code():
    assert check_mask_grid_alignment(_tp(None, _image())).code == "mask_grid_alignment.no_pair"
    assert check_mask_grid_alignment(_tp(_image(), None)).code == "mask_grid_alignment.no_pair"
    assert check_mask_grid_alignment(_tp(None, None)).status == "pass"


def test_shape_mismatch_fails_without_needing_affine_comparison():
    tp = _tp(_image(shape=SHAPE), _image(shape=(20, 20, 20)))
    result = check_mask_grid_alignment(tp)
    assert result.status == "fail"
    assert result.code == "mask_grid_alignment.shape_mismatch"
    assert str(SHAPE) in result.detail


def test_translation_within_tolerance_passes():
    ref_affine = _perturbed_affine({(0, 3): TRANSLATION_ATOL_MM / 2, (1, 3): TRANSLATION_ATOL_MM / 2})
    tp = _tp(_image(), _image(affine=ref_affine))
    assert check_mask_grid_alignment(tp).status == "pass"


def test_translation_beyond_tolerance_fails_with_reason():
    ref_affine = _perturbed_affine({(0, 3): TRANSLATION_ATOL_MM * 50})
    tp = _tp(_image(), _image(affine=ref_affine))
    result = check_mask_grid_alignment(tp)
    assert result.status == "fail"
    assert result.code == "mask_grid_alignment.drift"
    assert "translation" in result.detail
    assert result.evidence["translation_delta_mm"] == pytest.approx(TRANSLATION_ATOL_MM * 50)


def test_spacing_drift_beyond_its_tighter_tolerance_fails():
    ref_affine = _perturbed_affine({(0, 0): SPACING_ATOL_MM * 10})
    tp = _tp(_image(), _image(affine=ref_affine))
    result = check_mask_grid_alignment(tp)
    assert result.status == "fail"
    assert "voxel spacing" in result.detail


def test_direction_cosine_drift_beyond_tolerance_fails():
    # small enough that column norms (spacing) barely move, isolating this to direction drift
    tilt = DIRECTION_COSINE_ATOL * 5
    ref_affine = _perturbed_affine({(1, 0): tilt, (0, 1): -tilt})
    tp = _tp(_image(), _image(affine=ref_affine))
    result = check_mask_grid_alignment(tp)
    assert result.status == "fail"
    assert "direction cosines" in result.detail
    assert "voxel spacing" not in result.detail
