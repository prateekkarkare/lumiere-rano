"""
Rigid registration between two volumes that already share a grid.

WHY THIS EXISTS
---------------
Every LUMIERE image is resampled into MNI atlas space, so all of a patient's visits sit on an
identical 182x218x182 1mm grid. That is not the same as being aligned. Each visit was registered
to the atlas INDEPENDENTLY, and those registrations disagree with one another: measured across
the practice arm, consecutive visits' brain masks overlap only ~92% and the brain's centre of
mass moves ~2.7mm (worst 10.2mm). Same graph paper, map drawn in a slightly different place.

Volumetry never noticed, because a voxel count does not care where the voxels are. The first
operation that does care is asking whether a lesion at one visit is the same lesion as at the
next -- and there a 3mm slip is fatal for small lesions, which is exactly where new lesions live.

WHAT THIS DOES
--------------
Finds the six numbers (three rotations, three translations) that put one visit's brain on top of
another's. Rigid only: it is the same skull a few months apart, so the brain is allowed to move
but not to stretch. Rigid cannot express surgery, mass effect or atrophy, so brain overlap will
never reach 100% and chasing that would mean forcing a bad fit.

DIRECTION CONVENTION -- read this before using the result
---------------------------------------------------------
Following ITK, the returned transform maps points **from the fixed frame into the moving frame**.
That reads backwards, and it is the single most common way to get registration wrong. It is the
right way round for resampling: to place ``moving`` into ``fixed``'s frame you walk each voxel of
``fixed``, push it through the transform to find where it lands in ``moving``, and sample there.
``resample_like_fixed`` does exactly that, so prefer it over hand-rolling the arithmetic.

COORDINATE FRAME
----------------
Arrays are indexed ``(x, y, z)`` as nibabel presents them, and are handed to ITK on a plain 1mm
identity frame rather than their real affine. That is sound *only* because every atlas image in
this cohort shares one affine -- which ``fingerprint.atlas_consistency`` already verifies -- so
relative alignment is untouched and one millimetre equals one voxel. Do not reuse this module on
images that do not share a grid without revisiting that assumption.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk

__all__ = ["RigidFit", "fit_rigid", "resample_like_fixed", "dilate", "registration_mask"]


@dataclass(frozen=True, slots=True)
class RigidFit:
    """Six numbers and the point they turn about, plus how the search went.

    ``parameters`` is ITK's Euler3D ordering: three rotations in RADIANS about x, y and z,
    then three translations in millimetres. ``center`` is the rotation centre; a rotation is
    meaningless without one, so the pair must always be stored and restored together.
    """

    parameters: tuple[float, float, float, float, float, float]
    center: tuple[float, float, float]
    metric: float
    iterations: int
    stop_condition: str

    @property
    def rotation_degrees(self) -> tuple[float, float, float]:
        return tuple(float(np.degrees(v)) for v in self.parameters[:3])  # type: ignore[return-value]

    @property
    def translation_mm(self) -> tuple[float, float, float]:
        return tuple(float(v) for v in self.parameters[3:])  # type: ignore[return-value]

    @property
    def shift_magnitude_mm(self) -> float:
        return float(np.linalg.norm(self.parameters[3:]))

    def as_transform(self) -> sitk.Euler3DTransform:
        t = sitk.Euler3DTransform()
        t.SetCenter(self.center)
        t.SetParameters(self.parameters)
        return t

    def to_dict(self) -> dict:
        return {
            "parameters": list(self.parameters),
            "center": list(self.center),
            "rotation_degrees": list(self.rotation_degrees),
            "translation_mm": list(self.translation_mm),
            "shift_magnitude_mm": self.shift_magnitude_mm,
            "metric": self.metric,
            "iterations": self.iterations,
            "stop_condition": self.stop_condition,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RigidFit":
        return cls(
            parameters=tuple(d["parameters"]),  # type: ignore[arg-type]
            center=tuple(d["center"]),  # type: ignore[arg-type]
            metric=d["metric"],
            iterations=d["iterations"],
            stop_condition=d["stop_condition"],
        )

    @classmethod
    def from_translation(cls, t: tuple[float, float, float],
                         center: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> "RigidFit":
        """A pure slide, no turn. Used as a safe fallback when the full fit misbehaves."""
        return cls((0.0, 0.0, 0.0, float(t[0]), float(t[1]), float(t[2])),
                   center, metric=float("nan"), iterations=0,
                   stop_condition="translation only (fallback)")

    @classmethod
    def identity(cls, center: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> "RigidFit":
        """The anchor visit's own transform: it is already where it needs to be."""
        return cls((0.0,) * 6, center, metric=float("nan"), iterations=0,
                   stop_condition="identity (anchor visit)")


def _to_sitk(a: np.ndarray, dtype) -> sitk.Image:
    """(x, y, z) numpy -> ITK image on a 1mm identity frame. ITK reads arrays as (z, y, x)."""
    im = sitk.GetImageFromArray(np.ascontiguousarray(a.transpose(2, 1, 0)))
    im.SetSpacing((1.0, 1.0, 1.0))
    im.SetOrigin((0.0, 0.0, 0.0))
    im.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    return sitk.Cast(im, dtype)


def _from_sitk(im: sitk.Image) -> np.ndarray:
    return sitk.GetArrayFromImage(im).transpose(2, 1, 0)


def dilate(mask: np.ndarray, mm: int) -> np.ndarray:
    """Grow a boolean mask by ``mm`` voxels in every direction (26-neighbour steps)."""
    out = mask.astype(bool)
    for _ in range(int(mm)):
        grown = out.copy()
        for ax in (0, 1, 2):
            # roll the accumulating result, not the original: composing the three axes this way
            # grows a full 3x3x3 cube per step. Rolling the original gives only the 6 faces.
            grown |= np.roll(grown, 1, ax) | np.roll(grown, -1, ax)
        out = grown
    return out


def registration_mask(brain: np.ndarray, abnormal: np.ndarray, margin_mm: int = 5) -> np.ndarray:
    """Brain minus everything that changes -- the region the fit is allowed to look at.

    The tumour, its necrotic core or surgical cavity, and the surrounding oedema are precisely
    the things that differ between visits. Letting them into the metric asks the optimiser to
    make two different objects match, and the only way it can is by pulling everything else out
    of true. Excluding them, plus a margin for the fuzzy boundary, leaves the parts that really
    should be identical -- and lets the tumour land wherever it lands, which is the point.
    """
    return brain.astype(bool) & ~dilate(abnormal, margin_mm)


def fit_rigid(
    fixed: np.ndarray,
    moving: np.ndarray,
    fixed_mask: np.ndarray | None = None,
    moving_mask: np.ndarray | None = None,
    *,
    seed: int = 1234,
    iterations: int = 300,
) -> RigidFit:
    """Find the rigid motion taking ``fixed``'s frame into ``moving``'s (see DIRECTION above).

    Mattes mutual information rather than a difference-of-intensities metric: the two visits are
    the same sequence but not the same scan, and contrast, gain and scanner drift all vary. MI
    scores whether structures correspond, not whether numbers match.

    Coarse-to-fine over three resolutions, so a centimetre-scale offset is caught while the image
    is still blurred and cheap, before the fine level refines it. Fitting at full resolution alone
    is how rigid registration settles into a confident wrong answer.
    """
    f = _to_sitk(np.asarray(fixed, dtype=np.float32), sitk.sitkFloat32)
    m = _to_sitk(np.asarray(moving, dtype=np.float32), sitk.sitkFloat32)

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=48)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.15, seed=seed)  # seeded: the same inputs must give the same fit
    if fixed_mask is not None:
        R.SetMetricFixedMask(_to_sitk(fixed_mask.astype(np.uint8), sitk.sitkUInt8))
    if moving_mask is not None:
        R.SetMetricMovingMask(_to_sitk(moving_mask.astype(np.uint8), sitk.sitkUInt8))
    R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0, minStep=1e-4, numberOfIterations=iterations,
        gradientMagnitudeTolerance=1e-8,
    )
    R.SetOptimizerScalesFromPhysicalShift()  # otherwise a radian and a millimetre get equal weight
    R.SetShrinkFactorsPerLevel([4, 2, 1])
    R.SetSmoothingSigmasPerLevel([2, 1, 0])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    # optimised in place: with inPlace=False, Execute returns a CompositeTransform, which has
    # no GetCenter -- and a rotation without its centre is unusable.
    initial = sitk.Euler3DTransform(
        sitk.CenteredTransformInitializer(
            f, m, sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )
    )
    # Start from the two masks' centres of mass rather than the grid centre. Both volumes sit on
    # the same grid, so a geometry-based start is a no-op -- it hands the optimiser a blank sheet
    # and lets it hunt. On visits whose scan covers more of the neck, or whose intensities differ,
    # that hunt has been observed to run 80mm off and settle there, confidently wrong. Seeding
    # with the offset we can measure exactly leaves only a few millimetres to refine.
    if fixed_mask is not None and moving_mask is not None:
        fm, mm_ = np.asarray(fixed_mask, bool), np.asarray(moving_mask, bool)
        if fm.any() and mm_.any():
            offset = np.argwhere(mm_).mean(0) - np.argwhere(fm).mean(0)
            initial.SetTranslation(tuple(float(v) for v in offset))
    R.SetInitialTransform(initial, inPlace=True)
    R.Execute(f, m)

    return RigidFit(
        parameters=tuple(float(v) for v in initial.GetParameters()),  # type: ignore[arg-type]
        center=tuple(float(v) for v in initial.GetCenter()),  # type: ignore[arg-type]
        metric=float(R.GetMetricValue()),
        iterations=int(R.GetOptimizerIteration()),
        stop_condition=str(R.GetOptimizerStopConditionDescription()),
    )


def resample_like_fixed(
    moving: np.ndarray, fit: RigidFit, *, labels: bool = True
) -> np.ndarray:
    """Bring ``moving`` into the fixed visit's frame.

    ``labels=True`` uses nearest-neighbour, which is mandatory for a segmentation: linear
    interpolation of the integers 1, 2, 3 invents a 1.5 that means nothing. It also means every
    resampling nudges the boundary by up to half a voxel, so resample once and no more -- for
    comparing lesion positions, prefer transforming coordinates over moving whole volumes.
    """
    dtype = sitk.sitkUInt8 if labels else sitk.sitkFloat32
    m = _to_sitk(moving.astype(np.uint8 if labels else np.float32), dtype)
    ref = sitk.Image(m.GetSize(), dtype)
    ref.SetSpacing(m.GetSpacing()); ref.SetOrigin(m.GetOrigin()); ref.SetDirection(m.GetDirection())
    interp = sitk.sitkNearestNeighbor if labels else sitk.sitkLinear
    return _from_sitk(sitk.Resample(m, ref, fit.as_transform(), interp, 0, dtype))
