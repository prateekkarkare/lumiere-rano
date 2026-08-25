"""Response criteria: the RANO rule, its threshold profiles, and agreement reporting.

Nothing in this package knows about LUMIERE, zips, or NIfTI. It takes volumes in and produces
calls out, so the same rule serves the offline evaluation and the online pipeline.
"""

from rano.criteria.compare import (
    AgreementReport,
    CallPair,
    ClassStats,
    compare_calls,
    format_case_table,
    format_confusion,
    format_summary_line,
    format_timeline,
    split_by_group,
)
from rano.criteria.measurement import (
    SCORABLE,
    Response,
    ResponseAssessment,
    TimepointMeasurement,
    TrajectoryResult,
)
from rano.criteria.profiles import (
    DEFAULT_PROFILE,
    ENHANCING_ONLY,
    MRANO_VOLUMETRIC,
    MRANO_WITH_T2,
    PROFILES,
    RANO_CLASSIC_PORTED,
    ResponseCriteria,
)
from rano.criteria.rano import ReferenceState, assess_timepoint, assess_trajectory

__all__ = [
    "AgreementReport",
    "CallPair",
    "ClassStats",
    "DEFAULT_PROFILE",
    "ENHANCING_ONLY",
    "MRANO_VOLUMETRIC",
    "MRANO_WITH_T2",
    "PROFILES",
    "RANO_CLASSIC_PORTED",
    "Response",
    "ReferenceState",
    "ResponseAssessment",
    "ResponseCriteria",
    "SCORABLE",
    "TimepointMeasurement",
    "TrajectoryResult",
    "assess_timepoint",
    "assess_trajectory",
    "compare_calls",
    "format_case_table",
    "format_confusion",
    "format_summary_line",
    "format_timeline",
    "split_by_group",
]
