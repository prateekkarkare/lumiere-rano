"""Internal case contract: the lazy, space-aware representation every adapter targets."""

from rano.contract.case import (
    Geometry,
    ImageRef,
    LoadedImage,
    MaskSource,
    Modality,
    Patient,
    Space,
    SpaceTag,
    Timepoint,
)
from rano.contract.treatment import RadiotherapyCourse, TreatmentRecord, TreatmentWeek

__all__ = [
    "Geometry",
    "ImageRef",
    "LoadedImage",
    "MaskSource",
    "Modality",
    "Patient",
    "Space",
    "SpaceTag",
    "Timepoint",
    "RadiotherapyCourse",
    "TreatmentRecord",
    "TreatmentWeek",
]
