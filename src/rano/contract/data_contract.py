"""
The data contract — the machine-readable record Piece 1 emits for one patient.

This is the "QC report out" edge in its final form: everything a downstream stage needs to decide
whether it can work with a case, WITHOUT reopening the source archive. It is the union of three
things Piece 1 established:

  * what the case physically IS      -- geometry, spaces, which modalities and masks resolved;
  * what the numbers ARE            -- per-compartment volumes in mm3, computed in atlas space;
  * how far those numbers can be TRUSTED -- a size-dependent uncertainty, plus per-check verdicts.

Two rulings from the 2026-07-30 audit are baked in here as constants rather than left implicit,
because a number without its provenance is exactly the silent failure this project guards against:

  VOLUMETRY_SPACE -- atlas. The patient->atlas transforms are rigid (all 2,396 .tfm have
  determinant 1 to 1e-10), so there is no Jacobian scaling and no per-patient bias to correct.
  Native-space back-transformed masks are NOT used for reported numbers: they are the same
  segmentation resampled onto coarser grids, so they are noisier, not truer.

  VOLUME_UNCERTAINTY_PP -- the honest error bar, which depends on how BIG the compartment is,
  not on which compartment it is. A single tolerance would be wrong by ~20x across the range.

pydantic, per the contract/case.py design note: this is an edge, not the array-adjacent core.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict

SCHEMA_VERSION = "1.0"

#: Space in which every reported volume is computed. See the module docstring for why.
VOLUMETRY_SPACE = "atlas:mni152_1mm"

#: (min_mm3, max_mm3, p10-p90 spread in percentage points). Measured over 5,776 native-vs-atlas
#: comparisons on >=6x anisotropic grids -- the conservative end. On isotropic grids the same
#: bands are 1.9 / 0.9 / 0.4 / 0.2 / 0.1 pp. See docs/volume_audit.html.
VOLUME_UNCERTAINTY_PP: tuple[tuple[float, float, float], ...] = (
    (0.0, 1_000.0, 36.6),
    (1_000.0, 5_000.0, 14.8),
    (5_000.0, 20_000.0, 5.3),
    (20_000.0, 60_000.0, 2.8),
    (60_000.0, float("inf"), 1.5),
)

Readiness = Literal["usable", "needs_attention", "unusable"]


def uncertainty_pp(volume_mm3: float) -> float:
    """The spread, in percentage points, to attach to a reported volume of this size."""
    for lo, hi, pp in VOLUME_UNCERTAINTY_PP:
        if lo <= volume_mm3 < hi:
            return pp
    return VOLUME_UNCERTAINTY_PP[-1][2]


# --------------------------------------------------------------------------------------
class ImageEntry(BaseModel):
    """One resolved image: where it came from, what grid it is on."""

    model_config = ConfigDict(frozen=True)

    source: str
    space: str
    shape: tuple[int, ...]
    spacing_mm: tuple[float, float, float]
    anisotropy_ratio: float
    voxel_mm3: float


class CheckEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: str
    status: Literal["pass", "warn", "fail"]
    detail: str


class VolumeEntry(BaseModel):
    """A reported volume, inseparable from its uncertainty and the space it was measured in."""

    model_config = ConfigDict(frozen=True)

    volume_mm3: float
    uncertainty_pp: float
    space: str = VOLUMETRY_SPACE


class TimepointContract(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    label: str
    week_offset: float | None
    modalities: dict[str, ImageEntry]
    mask: ImageEntry | None = None
    mask_source: str | None = None
    compartment_volumes: dict[str, VolumeEntry] = {}
    region_volumes: dict[str, VolumeEntry] = {}
    expert_rating: str | None = None
    checks: tuple[CheckEntry, ...] = ()
    readiness: Readiness = "unusable"

    @property
    def is_assessable(self) -> bool:
        """Usable segmentation AND an expert response label — the cohort-selection predicate."""
        return self.readiness != "unusable" and self.expert_rating in {"CR", "PR", "SD", "PD"}


class PatientContract(BaseModel):
    model_config = ConfigDict(frozen=True)

    patient_id: str
    n_timepoints: int
    n_usable: int
    n_assessable: int
    is_longitudinal: bool
    timepoints: tuple[TimepointContract, ...]
    readiness: Readiness


class Provenance(BaseModel):
    model_config = ConfigDict(frozen=True)

    generated_utc: str
    schema_version: str = SCHEMA_VERSION
    source_archive: str
    manifest: str
    adapter: str
    mask_source: str
    label_schema_note: str
    volumetry_space: str = VOLUMETRY_SPACE
    volumetry_ruling: str
    hdglio_role: str


class DataContract(BaseModel):
    """The top-level artifact: rulings + provenance + one entry per patient."""

    model_config = ConfigDict(frozen=True)

    provenance: Provenance
    patients: tuple[PatientContract, ...]

    @staticmethod
    def provenance_now(source_archive: str, manifest: str, adapter: str, mask_source: str) -> Provenance:
        return Provenance(
            generated_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            source_archive=source_archive,
            manifest=manifest,
            adapter=adapter,
            mask_source=mask_source,
            label_schema_note=(
                "DeepBraTumIA label 1 = contrast-enhancing, 2 = necrosis/non-enhancing. Corrected "
                "2026-07-30; the pyradiomics CSV 'Label name' column has these swapped. Verified "
                "599/599 against shipped measured_volumes_in_mm3.json and by contrast uptake."
            ),
            volumetry_ruling=(
                "Volumes are computed in atlas space (MNI 1mm). The patient->atlas transforms are "
                "RIGID (2,396/2,396 .tfm with |det| = 1 to 1e-10), so there is no Jacobian scaling "
                "and no per-patient volume bias. Measured median native-vs-atlas deviation is 0.00% "
                "at every anisotropy and every lesion size. Native back-transformed masks are NOT "
                "used for reported numbers -- being resampled onto coarser grids they are noisier, "
                "not truer. Precision depends on lesion SIZE, not compartment; see uncertainty_pp."
            ),
            hdglio_role=(
                "HD-GLIO-AUTO is an independent second opinion on the ENHANCING compartment ONLY. "
                "Its label 1 merges edema, non-enhancing tumour AND necrosis into one class, so it "
                "is not comparable to any single DeepBraTumIA compartment and cannot form TC. Only "
                "ET-vs-ET agreement is meaningful. Its label mapping (2 = enhancing) was "
                "independently re-verified 2026-07-30 and is correct as documented."
            ),
        )


__all__ = [
    "SCHEMA_VERSION",
    "VOLUMETRY_SPACE",
    "VOLUME_UNCERTAINTY_PP",
    "uncertainty_pp",
    "ImageEntry",
    "CheckEntry",
    "VolumeEntry",
    "TimepointContract",
    "PatientContract",
    "Provenance",
    "DataContract",
]
