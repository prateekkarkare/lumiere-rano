"""
mask_grid_alignment — does this timepoint's mask actually share a grid with its own
mask_reference (its atlas-space stand-in), or only claim to via a matching Space tag?

This is a VALIDATOR check: per the pipeline (FINGERPRINTER -> VALIDATOR -> ROUTER), it consumes
``fingerprint.schema.TimepointFingerprint`` -- the record the fingerprinter already extracted --
never the live ``contract.case.Timepoint``/``ImageRef``. The validator has no business reaching
back past the fingerprinter into the adapter's zip handles; everything it needs (shape, spacing,
affine) was already read once, header-only, and written down. Re-reading it here would mean two
stages doing the same IO and would couple the validator to whichever adapter produced the case.

Why the affine is split into three pieces instead of one blanket delta: a NIfTI affine's 3x3
linear part is direction-cosines (unit rotation vectors) multiplied by voxel spacing — two
physically different quantities. Diffing raw matrix entries with one tolerance conflates them.
Here each piece gets its own ``np.allclose`` and its own tolerance:

  * translation (mm)         -- atol ~1e-3mm: a physical position offset.
  * direction cosines (unit) -- atol ~1e-3: which way the axes point, unitless.
  * voxel spacing (mm)       -- atol ~1e-4mm, tighter than translation: a spacing error doesn't
    just nudge the grid, it compounds across every voxel along that axis, so it costs more per
    unit of drift than the same-sized translation error.

Tolerance, not equality: today mask and mask_reference are produced together in one DeepBraTumIA
run and are bit-identical. The moment Piece 2 starts resampling, re-derived grids pick up float
roundoff, and an exact-equality gate would flag a genuinely-fine case as misaligned. This check
is built to survive that from day one rather than needing a rewrite when it happens.
"""

from __future__ import annotations

import numpy as np

from rano.fingerprint.schema import TimepointFingerprint
from rano.validate.core import CheckResult

#: default absolute tolerances -- mm for translation/spacing, unitless for direction cosines
TRANSLATION_ATOL_MM = 1e-3
DIRECTION_COSINE_ATOL = 1e-3
SPACING_ATOL_MM = 1e-4


def _direction_cosines(affine: np.ndarray) -> np.ndarray:
    """The affine's 3x3 linear part, each column normalized to a unit vector."""
    linear = affine[:3, :3]
    norms = np.linalg.norm(linear, axis=0)
    norms = np.where(norms == 0, 1.0, norms)  # guard a degenerate all-zero column
    return linear / norms


def check_mask_grid_alignment(
    tp: TimepointFingerprint,
    *,
    translation_atol: float = TRANSLATION_ATOL_MM,
    direction_atol: float = DIRECTION_COSINE_ATOL,
    spacing_atol: float = SPACING_ATOL_MM,
) -> CheckResult:
    """PASS/FAIL: does ``tp.mask`` share a grid with ``tp.mask_reference``, within tolerance?

    ``tp`` is a fingerprint record (``TimepointFingerprint``), not a live contract ``Timepoint``
    -- this check reads geometry the fingerprinter already extracted, nothing more.

    A missing mask or mask_reference is reported "pass" with a distinct code -- there's nothing
    to contradict, and a missing mask is a different deficit for a different check to report.
    """
    if tp.mask is None or tp.mask_reference is None:
        return CheckResult(
            status="pass",
            code="mask_grid_alignment.no_pair",
            detail="no mask and/or mask_reference present; nothing to check",
            evidence={},
        )

    mask_geo = tp.mask.geometry
    ref_geo = tp.mask_reference.geometry

    if mask_geo.shape != ref_geo.shape:
        return CheckResult(
            status="fail",
            code="mask_grid_alignment.shape_mismatch",
            detail=f"shape mismatch: mask {mask_geo.shape} vs mask_reference {ref_geo.shape}",
            evidence={"mask_shape": mask_geo.shape, "mask_reference_shape": ref_geo.shape},
        )

    mask_affine = np.array(mask_geo.affine)
    ref_affine = np.array(ref_geo.affine)

    mask_translation = mask_affine[:3, 3]
    ref_translation = ref_affine[:3, 3]
    translation_delta = float(np.max(np.abs(mask_translation - ref_translation)))
    translation_ok = bool(np.allclose(mask_translation, ref_translation, atol=translation_atol, rtol=0.0))

    mask_dircos = _direction_cosines(mask_affine)
    ref_dircos = _direction_cosines(ref_affine)
    direction_delta = float(np.max(np.abs(mask_dircos - ref_dircos)))
    direction_ok = bool(np.allclose(mask_dircos, ref_dircos, atol=direction_atol, rtol=0.0))

    mask_spacing = np.array(mask_geo.spacing)
    ref_spacing = np.array(ref_geo.spacing)
    spacing_delta = float(np.max(np.abs(mask_spacing - ref_spacing)))
    spacing_ok = bool(np.allclose(mask_spacing, ref_spacing, atol=spacing_atol, rtol=0.0))

    evidence = {
        "translation_delta_mm": translation_delta,
        "translation_atol_mm": translation_atol,
        "direction_cosine_delta": direction_delta,
        "direction_cosine_atol": direction_atol,
        "spacing_delta_mm": spacing_delta,
        "spacing_atol_mm": spacing_atol,
    }

    if translation_ok and direction_ok and spacing_ok:
        return CheckResult(
            status="pass",
            code="mask_grid_alignment.ok",
            detail="mask and mask_reference share one grid within tolerance",
            evidence=evidence,
        )

    reasons = []
    if not translation_ok:
        reasons.append(f"translation off by {translation_delta:.5f}mm (tolerance {translation_atol}mm)")
    if not direction_ok:
        reasons.append(f"direction cosines off by {direction_delta:.5f} (tolerance {direction_atol})")
    if not spacing_ok:
        reasons.append(f"voxel spacing off by {spacing_delta:.5f}mm (tolerance {spacing_atol}mm)")

    return CheckResult(
        status="fail",
        code="mask_grid_alignment.drift",
        detail="mask/mask_reference grid drift exceeds tolerance: " + "; ".join(reasons),
        evidence=evidence,
    )


__all__ = [
    "check_mask_grid_alignment",
    "TRANSLATION_ATOL_MM",
    "DIRECTION_COSINE_ATOL",
    "SPACING_ATOL_MM",
]
