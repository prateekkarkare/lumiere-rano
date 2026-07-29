"""
Cohort-wide atlas-space consistency check — the first slice of the fingerprinter.

Every DeepBraTumIA-produced image (mask, mask_reference, brain_mask) is tagged
``Space.mni152_1mm()`` by the LUMIERE adapter, on the strength of DeepBraTumIA's own
documentation. That external label was never independently verified against a published
MNI152 template — and it doesn't need to be: nothing downstream in this pipeline computes
against an external reference file. What DOES matter functionally is that every image
carrying that tag, across the WHOLE cohort, actually lives on the same grid (identical
affine + shape). If it doesn't, cross-timepoint operations (Piece 2 registration,
new-lesion detection) would silently misalign under a label that claims otherwise — the
exact silent-failure class this project guards against.

This module checks exactly that, and only that: a pure, read-only aggregation over
``Geometry`` objects (header-only — no voxel arrays touched). It takes ``Patient``
objects, not zip members, so it is adapter-agnostic — the same check will apply to a
future DICOM-derived cohort registered into a shared space.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from rano.contract.case import ImageRef, Patient, SpaceTag, Timepoint


@dataclass(frozen=True)
class AtlasOutlier:
    """One image whose grid doesn't match the reference established by the first image seen."""

    patient: str
    timepoint: str
    item: str  # "mask" | "mask_reference" | "brain_mask"
    shape: tuple[int, ...]
    shape_mismatch: bool
    affine_mismatch: bool
    max_affine_delta: float


@dataclass(frozen=True)
class AtlasConsistencyReport:
    space_tag: SpaceTag
    reference_shape: tuple[int, ...]
    reference_affine: np.ndarray = field(repr=False)
    n_checked: int
    outliers: tuple[AtlasOutlier, ...]

    @property
    def is_consistent(self) -> bool:
        return not self.outliers

    @property
    def n_consistent(self) -> int:
        return self.n_checked - len(self.outliers)


def _tagged_items(tp: Timepoint, space_tag: SpaceTag) -> list[tuple[str, ImageRef]]:
    """The timepoint's images that carry the given space tag, by name."""
    candidates = (("mask", tp.mask), ("mask_reference", tp.mask_reference), ("brain_mask", tp.brain_mask))
    return [(name, ref) for name, ref in candidates if ref is not None and ref.space.tag is space_tag]


def check_atlas_consistency(
    patients: Iterable[Patient],
    *,
    space_tag: SpaceTag = SpaceTag.MNI152_1MM,
    atol: float = 1e-4,
) -> AtlasConsistencyReport:
    """Verify every image tagged ``space_tag`` shares one grid (affine + shape).

    The reference grid is simply the first such image encountered; every later one is
    diffed against it. Reads headers only (``ImageRef.geometry``) — no voxel data.
    """
    ref_shape: tuple[int, ...] | None = None
    ref_affine: np.ndarray | None = None
    outliers: list[AtlasOutlier] = []
    n = 0

    for patient in patients:
        for tp in patient:
            for item, ref in _tagged_items(tp, space_tag):
                geo = ref.geometry
                n += 1
                if ref_shape is None:
                    ref_shape, ref_affine = geo.shape, geo.affine
                    continue
                shape_mismatch = geo.shape != ref_shape
                delta = float(np.max(np.abs(geo.affine - ref_affine)))
                affine_mismatch = delta > atol
                if shape_mismatch or affine_mismatch:
                    outliers.append(
                        AtlasOutlier(
                            patient=patient.id,
                            timepoint=tp.label,
                            item=item,
                            shape=geo.shape,
                            shape_mismatch=shape_mismatch,
                            affine_mismatch=affine_mismatch,
                            max_affine_delta=delta,
                        )
                    )

    if ref_shape is None or ref_affine is None:
        raise ValueError(f"no images tagged {space_tag!r} were found in the given patients")

    return AtlasConsistencyReport(
        space_tag=space_tag,
        reference_shape=ref_shape,
        reference_affine=ref_affine,
        n_checked=n,
        outliers=tuple(outliers),
    )


__all__ = ["AtlasOutlier", "AtlasConsistencyReport", "check_atlas_consistency"]
