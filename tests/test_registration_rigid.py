"""
Tests for rigid registration.

The two that earn their keep are ``test_recovers_a_known_shift`` and
``test_resampling_puts_the_moving_volume_back_on_the_fixed_one``. ITK's transform maps points
from the FIXED frame into the MOVING frame, which reads backwards, and getting it inverted
produces a registration that is confidently wrong in the exact amount it was supposed to fix --
a silent failure that no amount of eyeballing a metric value would catch. These tests use a
phantom shifted by a known number of voxels, so the right answer is known in advance.
"""

from __future__ import annotations

import numpy as np
import pytest

from rano.registration import RigidFit, dilate, fit_rigid, registration_mask, resample_like_fixed


def phantom(shape=(56, 56, 56), seed=0) -> np.ndarray:
    """An asymmetric blobby volume. Mutual information needs structure to lock onto -- a single
    centred sphere is rotationally symmetric and would let any rotation score equally well."""
    rng = np.random.default_rng(seed)
    x, y, z = np.indices(shape)
    vol = np.zeros(shape, np.float32)
    for cx, cy, cz, r, val in [
        (22, 26, 24, 11, 1.0), (36, 20, 30, 7, 0.65),
        (26, 38, 34, 5, 0.85), (32, 32, 18, 4, 0.45),
    ]:
        vol[(x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 < r * r] = val
    return vol + rng.normal(0, 0.01, shape).astype(np.float32)


def test_identical_volumes_give_a_near_identity_fit():
    v = phantom()
    fit = fit_rigid(v, v)
    assert fit.shift_magnitude_mm < 0.5
    assert max(abs(d) for d in fit.rotation_degrees) < 0.5


@pytest.mark.parametrize("axis,delta", [(0, 4), (1, -3), (2, 5)])
def test_recovers_a_known_shift(axis, delta):
    """np.roll(v, delta, axis) puts the content delta voxels further along that axis. The fit maps
    fixed -> moving, so it must report a translation of +delta on that axis and ~0 elsewhere."""
    fixed = phantom()
    moving = np.roll(fixed, delta, axis=axis)
    fit = fit_rigid(fixed, moving)
    recovered = fit.translation_mm
    assert recovered[axis] == pytest.approx(delta, abs=0.75), f"axis {axis}: {recovered}"
    for other in (0, 1, 2):
        if other != axis:
            assert abs(recovered[other]) < 0.75, f"leaked onto axis {other}: {recovered}"


def test_resampling_puts_the_moving_volume_back_on_the_fixed_one():
    """The end-to-end contract: fit, resample, and the two volumes should now sit on each other."""
    fixed = phantom()
    moving = np.roll(fixed, 5, axis=2)
    fb, mb = fixed > 0.2, moving > 0.2

    def overlap(a, b):
        return (a & b).sum() / (a | b).sum()

    before = overlap(fb, mb)
    back = resample_like_fixed(mb.astype(np.uint8), fit_rigid(fixed, moving), labels=True) > 0
    after = overlap(fb, back)
    assert before < 0.75, "phantom shift was too small to be a real test"
    assert after > 0.95, f"registration did not close the gap: {before:.2f} -> {after:.2f}"
    assert after > before


def test_recovers_a_known_rotation():
    """Build the moving volume by turning the phantom a known 6 degrees, then check we find it."""
    import SimpleITK as sitk

    fixed = phantom()
    turn = sitk.Euler3DTransform()
    turn.SetCenter((28.0, 28.0, 28.0))
    turn.SetRotation(0.0, 0.0, np.radians(6.0))
    spun = RigidFit(
        parameters=tuple(float(v) for v in turn.GetParameters()),  # type: ignore[arg-type]
        center=(28.0, 28.0, 28.0), metric=0.0, iterations=0, stop_condition="synthetic",
    )
    moving = resample_like_fixed(fixed, spun, labels=False)

    fit = fit_rigid(fixed, moving)
    rz = fit.rotation_degrees[2]
    assert abs(abs(rz) - 6.0) < 1.5, f"expected about 6 degrees about z, got {fit.rotation_degrees}"

    def overlap(a, b):
        return (a & b).sum() / (a | b).sum()

    fb, mb = fixed > 0.2, moving > 0.2
    back = resample_like_fixed(mb.astype(np.uint8), fit, labels=True) > 0
    assert overlap(fb, back) > overlap(fb, mb)


def test_a_fit_survives_a_round_trip_through_json():
    fit = fit_rigid(phantom(), np.roll(phantom(), 3, axis=1))
    again = RigidFit.from_dict(fit.to_dict())
    assert again.parameters == pytest.approx(fit.parameters)
    assert again.center == pytest.approx(fit.center)


def test_identity_fit_moves_nothing():
    v = (phantom() > 0.2).astype(np.uint8)
    same = resample_like_fixed(v, RigidFit.identity(center=(28.0, 28.0, 28.0)), labels=True)
    assert np.array_equal(same, v)


def test_dilate_grows_by_the_requested_margin():
    m = np.zeros((21, 21, 21), bool)
    m[10, 10, 10] = True
    assert dilate(m, 0).sum() == 1
    assert dilate(m, 1).sum() == 27          # 3x3x3 cube around the seed
    assert dilate(m, 2).sum() == 125         # 5x5x5


def test_registration_mask_excludes_the_lesion_and_a_margin():
    brain = np.zeros((30, 30, 30), bool); brain[5:25, 5:25, 5:25] = True
    lesion = np.zeros((30, 30, 30), bool); lesion[14:17, 14:17, 14:17] = True
    keep = registration_mask(brain, lesion, margin_mm=2)
    assert not keep[15, 15, 15], "lesion itself must be excluded"
    assert not keep[13, 15, 15], "the margin around it must be excluded too"
    assert keep[6, 6, 6], "brain far from the lesion must be kept"
    assert not keep[0, 0, 0], "outside the brain must never be included"
    assert keep.sum() < brain.sum()
