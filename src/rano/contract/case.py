"""
Internal case contract — the single in-memory representation every front door reduces to.

This is *the seam* of Piece 1 (see the C4 L4 diagram). Two rules shape every type here:

  * The contract is LAZY. A ``Timepoint`` holds ``ImageRef`` handles, never voxel arrays.
    ``Geometry`` is derived from the NIfTI header alone (shape + affine + dtype), so
    fingerprinting and validation can inspect a whole cohort without loading pixels.

  * Space lives per-image, not per-timepoint. Each ``ImageRef`` carries its own ``Space``.
    In LUMIERE the raw modalities sit in their own native spaces while the DeepBraTumIA
    mask sits in MNI atlas space — so "are these two grids aligned?" is only a meaningful
    question *within a single space*. This is what stops the mask-alignment check from
    comparing an MNI mask against native FLAIR and crying "misaligned" on every case.

Nothing here imports ``label_schema`` or hard-codes label integers: label semantics belong
to the validator and the volumetry stage, which import the locked ``label_schema.py``.
These types only name *which tool* produced a mask (``MaskSource``), never what its
integers mean.

Design choice: frozen dataclasses (not pydantic) for the core. pydantic lives at the edges
(config in, QC report out) where schema validation earns its keep; it has no place on the
array-adjacent hot path.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from math import prod
from typing import Iterator, Mapping

import nibabel as nib
import numpy as np


# --------------------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------------------
class Modality(str, Enum):
    """MRI sequences the pipeline understands. ``CT1`` is contrast-enhanced T1 (a.k.a. T1c/T1Gd)."""

    CT1 = "CT1"
    T1 = "T1"
    T2 = "T2"
    FLAIR = "FLAIR"


class MaskSource(str, Enum):
    """Which segmentation tool produced a mask.

    Values MUST match the keys of ``label_schema.LABEL_SCHEMA`` so downstream code can do
    ``LABEL_SCHEMA[mask_source.value]``. Naming a *tool* is fine here; encoding its *integers*
    is not — that stays in the locked ``label_schema.py``.
    """

    DEEPBRATUMIA = "DeepBraTumIA"
    HDGLIO_AUTO = "HD-GLIO-AUTO"


class SpaceTag(str, Enum):
    """Coarse coordinate-system family."""

    NATIVE = "native"          # a scanner/native grid; NOT assumed co-registered to any other
    MNI152_1MM = "mni152_1mm"  # DeepBraTumIA atlas space, 1mm isotropic


@dataclass(frozen=True)
class Space:
    """A concrete coordinate space.

    ``tag`` is the family; ``key`` distinguishes distinct spaces *within* a family. Two images
    live in the same space iff their ``Space`` compares equal. All MNI-atlas images share one
    key; each native grid gets its own key, so two ``NATIVE`` images are NOT presumed aligned
    until a registration stage declares them so.
    """

    tag: SpaceTag
    key: str

    @classmethod
    def mni152_1mm(cls) -> "Space":
        return cls(SpaceTag.MNI152_1MM, "mni152_1mm")

    @classmethod
    def native(cls, key: str) -> "Space":
        return cls(SpaceTag.NATIVE, key)

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.tag.value}:{self.key}"


# --------------------------------------------------------------------------------------
# Geometry — everything readable from a NIfTI header, no voxels loaded
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Geometry:
    """Grid metadata of a single image, derived from its header alone.

    ``affine`` is excluded from equality/hash (``compare=False``) so the dataclass stays
    hashable despite carrying a numpy array — equality rests on the scalar grid descriptors.
    """

    shape: tuple[int, ...]
    spacing: tuple[float, float, float]   # mm per voxel along the 3 spatial axes
    orientation: str                      # e.g. "RAS", from nibabel.aff2axcodes
    dtype: str
    anisotropy_ratio: float               # max(spacing) / min(spacing); 1.0 == isotropic
    affine: np.ndarray = field(compare=False, repr=False)

    @classmethod
    def from_header(cls, shape, affine, dtype) -> "Geometry":
        """Build from the three things any NIfTI header hands over cheaply."""
        affine = np.asarray(affine, dtype=float)
        spacing = tuple(float(np.linalg.norm(affine[:3, i])) for i in range(3))
        smin = min(spacing)
        anisotropy = max(spacing) / smin if smin > 0 else float("inf")
        return cls(
            shape=tuple(int(s) for s in shape),
            spacing=spacing,  # type: ignore[arg-type]
            orientation="".join(nib.aff2axcodes(affine)),
            dtype=str(dtype),
            anisotropy_ratio=float(anisotropy),
            affine=affine,
        )

    @property
    def voxel_volume_mm3(self) -> float:
        """Volume of one voxel — the multiplier the volumetry stage needs."""
        return float(prod(self.spacing))

    @property
    def n_voxels(self) -> int:
        return int(prod(self.shape))

    @property
    def is_isotropic(self) -> bool:
        return self.anisotropy_ratio <= 1.01  # tolerate float noise


# --------------------------------------------------------------------------------------
# ImageRef — a lazy handle to one image, not the array
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class LoadedImage:
    """The materialized result of ``ImageRef.load()``. Holds arrays, so equality is by identity."""

    data: np.ndarray
    affine: np.ndarray
    header: object  # nibabel header; kept opaque so the contract stays format-agnostic

    __hash__ = None  # type: ignore[assignment]

    def __eq__(self, other: object) -> bool:  # pragma: no cover - identity semantics
        return self is other


class ImageRef(ABC):
    """A lazy reference to one image volume.

    Concrete adapters subclass this — a LUMIERE ref streams a member out of the zip, a future
    DICOM ref assembles a series from disk. The contract never knows which. Implementations are
    responsible for their own caching (e.g. a bounded LRU on ``load``); ``geometry`` must be
    cheap (header-only) and must NOT trigger a full-array read.
    """

    @property
    @abstractmethod
    def source(self) -> str:
        """Human-readable provenance, e.g. a zip member path or a DICOM series UID."""

    @property
    @abstractmethod
    def space(self) -> Space:
        ...

    @property
    @abstractmethod
    def geometry(self) -> Geometry:
        """Grid metadata, read from the header without materializing voxels."""

    @abstractmethod
    def load(self) -> LoadedImage:
        """Materialize the voxel array (+ affine + header). May be cached by the implementation."""


# --------------------------------------------------------------------------------------
# Timepoint & Patient
# --------------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class Timepoint:
    """One study for one patient: the available modalities plus an optional canonical mask.

    Identity semantics (``eq=False``) because it carries a mapping and is a container, not a value.
    ``week_offset`` is the numeric sort key parsed from ``label`` (e.g. "week-044" -> 44.0);
    ``None`` means unparseable, and such timepoints sort last.
    """

    id: str
    label: str                              # original manifest label, e.g. "week-000-1"
    week_offset: float | None
    modalities: Mapping[Modality, ImageRef]
    mask: ImageRef | None = None
    mask_source: MaskSource | None = None
    # An image in the MASK's OWN space (e.g. the DeepBraTumIA atlas skull-strip), so the
    # mask-alignment check compares grids that are meant to match — never mask-vs-native-FLAIR.
    mask_reference: ImageRef | None = None
    # Brain mask in the mask's space, for the skull-strip sanity check. May be absent.
    brain_mask: ImageRef | None = None

    def has_modality(self, m: Modality) -> bool:
        return m in self.modalities

    @property
    def available_modalities(self) -> frozenset[Modality]:
        return frozenset(self.modalities)

    def get(self, m: Modality) -> ImageRef | None:
        return self.modalities.get(m)

    @property
    def has_mask(self) -> bool:
        return self.mask is not None


@dataclass(frozen=True, eq=False)
class Patient:
    """A patient as an ordered sequence of timepoints.

    The adapter guarantees chronological order; ``__post_init__`` asserts it so a mis-sorting
    adapter fails loudly here rather than silently corrupting longitudinal logic downstream.
    Timepoints with ``week_offset is None`` are permitted and expected to trail the ordered ones.
    """

    id: str
    timepoints: tuple[Timepoint, ...]

    def __post_init__(self) -> None:
        known = [tp.week_offset for tp in self.timepoints if tp.week_offset is not None]
        if known != sorted(known):
            raise ValueError(
                f"Patient {self.id!r}: timepoints not in chronological order: "
                f"{[tp.label for tp in self.timepoints]}"
            )

    @property
    def n_timepoints(self) -> int:
        return len(self.timepoints)

    @property
    def is_longitudinal(self) -> bool:
        """True when RANO assessment (not just segment-and-store) is in scope."""
        return len(self.timepoints) > 1

    @property
    def baseline(self) -> Timepoint | None:
        """The earliest (post-op) timepoint, or None if the patient has none."""
        return self.timepoints[0] if self.timepoints else None

    def __iter__(self) -> Iterator[Timepoint]:
        return iter(self.timepoints)


__all__ = [
    "Modality",
    "MaskSource",
    "SpaceTag",
    "Space",
    "Geometry",
    "LoadedImage",
    "ImageRef",
    "Timepoint",
    "Patient",
]
