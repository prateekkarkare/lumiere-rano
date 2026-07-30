"""Read-only stats over the case contract: per-patient extraction plus cohort-wide checks."""

from rano.fingerprint.atlas_consistency import (
    AtlasConsistencyReport,
    AtlasOutlier,
    check_atlas_consistency,
)
from rano.fingerprint.extractor import fingerprint_patient, fingerprint_timepoint
from rano.fingerprint.io import write_patient_fingerprint
from rano.fingerprint.schema import (
    CompartmentIntensity,
    GeometryFingerprint,
    ImageFingerprint,
    IntensityStats,
    LabelCount,
    PatientFingerprint,
    TimepointFingerprint,
)

__all__ = [
    "AtlasConsistencyReport",
    "AtlasOutlier",
    "check_atlas_consistency",
    "fingerprint_patient",
    "fingerprint_timepoint",
    "write_patient_fingerprint",
    "CompartmentIntensity",
    "GeometryFingerprint",
    "ImageFingerprint",
    "IntensityStats",
    "LabelCount",
    "PatientFingerprint",
    "TimepointFingerprint",
]
