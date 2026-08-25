"""
Volumetry — voxel counts to millimetres cubed. The number the whole pipeline exists to produce.

Two rules, both of which exist because the failure modes here are SILENT (a wrong answer, never
an exception):

1. **Voxel volume is ``|det(affine[:3, :3])|``, not ``prod(spacing)``.**
   A voxel is the unit cube pushed through the affine's 3x3 block, so its volume is that block's
   determinant. ``prod(spacing)`` — the product of the column norms — equals it only when the
   direction cosines are orthonormal. Every affine in LUMIERE satisfies that (verified: deviation
   from orthonormality <= 3e-16 across the archive), so the two agree here; a gantry-tilted or
   sheared grid from some future source would not, and would be quietly over-counted. The
   determinant costs nothing and is simply the definition.

   Corollary: nothing in this module reads ``header.get_zooms()`` or assumes an axis order.
   LUMIERE's own FLAIR masks are shape ``(640, 40, 640)`` — the thick axis sits in the MIDDLE —
   so any positional assumption about "the slice axis" is wrong on real data.

2. **An off-schema label is a hard error.**
   ``label_schema.py`` is the locked decode. If a mask carries an integer the schema doesn't
   document, we do not know what tissue it is, and every downstream volume is suspect. Counting
   only the labels we recognise would return a plausible, confident, wrong number — so this
   module raises instead. Labels that are *documented but absent* are 0.0 mm3, which is a normal
   finding (a compartment can genuinely not exist), never an error.

Volumes are returned by COMPARTMENT NAME (from the schema) and by COMPOSITE REGION (ET/TC/WT,
from ``COMPOSITE_REGIONS``) — never by raw integer. Integers stop at this boundary.
"""

from __future__ import annotations

import numpy as np

from rano.contract.case import MaskSource
from rano.labels import COMPOSITE_REGIONS, LABEL_SCHEMA


def voxel_volume_mm3(affine: np.ndarray) -> float:
    """Physical volume of one voxel, in mm3, for the grid described by ``affine``.

    ``|det|`` of the 3x3 linear block — see this module's docstring for why not ``prod(spacing)``.
    """
    linear = np.asarray(affine, dtype=float)[:3, :3]
    return float(abs(np.linalg.det(linear)))


def _schema_for(mask_source: MaskSource | str) -> dict[int, str]:
    key = mask_source.value if isinstance(mask_source, MaskSource) else str(mask_source)
    try:
        return LABEL_SCHEMA[key]
    except KeyError:
        raise KeyError(
            f"no label schema for mask source {key!r}; known: {sorted(LABEL_SCHEMA)}"
        ) from None


def label_voxel_counts(labels: np.ndarray, mask_source: MaskSource | str) -> dict[int, int]:
    """Voxel count per integer label present, after checking every integer is documented.

    Raises ``ValueError`` on any integer the schema doesn't know — see rule 2 above.
    """
    schema = _schema_for(mask_source)
    values, counts = np.unique(np.asarray(labels), return_counts=True)
    found = {int(v): int(c) for v, c in zip(values, counts)}

    off_schema = sorted(set(found) - set(schema))
    if off_schema:
        key = mask_source.value if isinstance(mask_source, MaskSource) else mask_source
        raise ValueError(
            f"mask carries label(s) {off_schema} not documented in the {key!r} schema "
            f"{sorted(schema)}; refusing to compute volumes from an undecodable mask"
        )
    return found


def compartment_volumes(
    labels: np.ndarray,
    affine: np.ndarray,
    mask_source: MaskSource | str = MaskSource.DEEPBRATUMIA,
    *,
    include_background: bool = False,
) -> dict[str, float]:
    """Volume in mm3 of every compartment the schema documents, keyed by compartment NAME.

    Compartments documented by the schema but absent from this mask are reported as ``0.0`` —
    an absent compartment is a finding, not a gap. ``background`` is excluded by default.
    """
    schema = _schema_for(mask_source)
    counts = label_voxel_counts(labels, mask_source)
    vox = voxel_volume_mm3(affine)
    return {
        name: float(counts.get(value, 0)) * vox
        for value, name in schema.items()
        if include_background or name != "background"
    }


def region_volumes(
    labels: np.ndarray,
    affine: np.ndarray,
    mask_source: MaskSource | str = MaskSource.DEEPBRATUMIA,
) -> dict[str, float]:
    """Volume in mm3 of each composite BraTS-style region (ET / TC / WT) defined for this source.

    Regions are unions of labels, so they are summed over voxel counts *before* multiplying by the
    voxel volume — no double counting, and no float error from adding pre-scaled volumes.
    """
    key = mask_source.value if isinstance(mask_source, MaskSource) else str(mask_source)
    try:
        regions = COMPOSITE_REGIONS[key]
    except KeyError:
        raise KeyError(
            f"no composite regions for mask source {key!r}; known: {sorted(COMPOSITE_REGIONS)}"
        ) from None

    counts = label_voxel_counts(labels, mask_source)
    vox = voxel_volume_mm3(affine)
    return {
        region: float(sum(counts.get(lbl, 0) for lbl in members)) * vox
        for region, members in regions.items()
    }


__all__ = [
    "voxel_volume_mm3",
    "label_voxel_counts",
    "compartment_volumes",
    "region_volumes",
]
