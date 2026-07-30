"""
Fingerprint record schema — the pydantic edge that carries per-patient stats out of Piece 1.

Per ``contract/case.py``'s design choice, pydantic lives at the edges (config in, QC report
out); this is the first "QC report out" edge. The records here are plain, serializable data —
no ``ImageRef`` handles, no open zip members — so they can be written to disk, diffed across
runs, and consumed by the validator without re-touching the source archive.

Labels are kept as raw integers (``LabelCount.label``, ``CompartmentIntensity.label``), never
resolved to compartment names. Per ``contract/case.py``'s own rule, label semantics belong to
the validator and volumetry stage, which import the locked ``label_schema.py`` — this module
doesn't, deliberately, so it never has an opinion on whether a given integer is expected.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class GeometryFingerprint(BaseModel):
    """Serializable mirror of ``contract.case.Geometry`` — header-only grid metadata."""

    model_config = ConfigDict(frozen=True)

    shape: tuple[int, ...]
    spacing: tuple[float, float, float]
    orientation: str
    dtype: str
    anisotropy_ratio: float
    affine: tuple[tuple[float, float, float, float], ...]


class IntensityStats(BaseModel):
    """Summary of voxel intensities over some basis (nonzero voxels, a mask, one label, ...).

    ``n_voxels`` is the count of voxels the basis selected, not the image's total voxel count.
    """

    model_config = ConfigDict(frozen=True)

    n_voxels: int
    mean: float
    std: float
    min: float
    max: float
    median: float
    p0_5: float
    p99_5: float


class LabelCount(BaseModel):
    """One integer label found in a mask, and how many voxels carry it."""

    model_config = ConfigDict(frozen=True)

    label: int
    n_voxels: int


class CompartmentIntensity(BaseModel):
    """Intensity stats on ``mask_reference``, restricted to one label's voxels in the mask."""

    model_config = ConfigDict(frozen=True)

    label: int
    stats: IntensityStats


class ImageFingerprint(BaseModel):
    """Everything extracted from one ``ImageRef``.

    ``intensity`` is the brain-extent stat (nonzero voxels for native modalities, ``brain_mask``-
    restricted for ``mask_reference``); ``label_histogram`` is populated only for ``mask``;
    ``compartment_intensity`` only for ``mask_reference``. Categorical/binary images (``mask``,
    ``brain_mask``) carry no ``intensity``.
    """

    model_config = ConfigDict(frozen=True)

    role: str  # "modality:CT1" | "mask" | "mask_reference" | "brain_mask"
    source: str
    space: str
    geometry: GeometryFingerprint
    intensity: IntensityStats | None = None
    label_histogram: tuple[LabelCount, ...] | None = None
    compartment_intensity: tuple[CompartmentIntensity, ...] | None = None


class TimepointFingerprint(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    label: str
    week_offset: float | None
    modalities: dict[str, ImageFingerprint]
    mask: ImageFingerprint | None = None
    mask_source: str | None = None
    mask_reference: ImageFingerprint | None = None
    brain_mask: ImageFingerprint | None = None


class PatientFingerprint(BaseModel):
    model_config = ConfigDict(frozen=True)

    patient_id: str
    n_timepoints: int
    timepoints: tuple[TimepointFingerprint, ...]


__all__ = [
    "GeometryFingerprint",
    "IntensityStats",
    "LabelCount",
    "CompartmentIntensity",
    "ImageFingerprint",
    "TimepointFingerprint",
    "PatientFingerprint",
]
