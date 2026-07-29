"""Read-only cohort-wide statistics over the case contract. Never transforms pixels."""

from rano.fingerprint.atlas_consistency import (
    AtlasConsistencyReport,
    AtlasOutlier,
    check_atlas_consistency,
)

__all__ = ["AtlasConsistencyReport", "AtlasOutlier", "check_atlas_consistency"]
