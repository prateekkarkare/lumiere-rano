"""
The inputs and outputs of a response assessment — deliberately wider than any one dataset.

DESIGN RULE: **an unavailable signal is ``None``, never ``False``.**

RANO is a multi-component rule. Most datasets can supply only some of the components, and the
tempting shortcut is to let a missing one default to "no" -- no new lesion, no clinical
deterioration, no steroid increase. That shortcut turns "we did not look" into "we looked and
found nothing", which is the difference between an honest SD and a missed PD. Every optional
field below is therefore tri-state, and every assessment reports which components it could not
evaluate (``unknowns``) alongside the call it made anyway.

A caller who genuinely knows a component is negative passes ``False`` and gets a complete
assessment. A caller who cannot know passes ``None`` and gets a call plus an explicit admission.
Both are supported; silently conflating them is not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Response(StrEnum):
    """RANO response categories, plus the two states that are not responses."""

    CR = "CR"
    """Complete response -- no enhancing disease."""
    PR = "PR"
    """Partial response -- enhancing volume down by at least the profile's threshold."""
    SD = "SD"
    """Stable disease -- neither response nor progression."""
    PD = "PD"
    """Progressive disease."""

    BASELINE = "BASELINE"
    """The reference scan itself. Not a response; never scored against an expert call."""
    NE = "NE"
    """Not evaluable -- the enhancing measurement required to make any call is missing."""


#: The four calls that can be compared against an expert RANO rating.
SCORABLE: frozenset[Response] = frozenset({Response.CR, Response.PR, Response.SD, Response.PD})


@dataclass(frozen=True, slots=True)
class TimepointMeasurement:
    """Everything the criteria need about one scan. Optional fields are tri-state (see module doc).

    ``enhancing_mm3`` is the only field the rule cannot proceed without; absent, the timepoint is
    ``NE``. Volumes are millimetres cubed in a consistent space across the trajectory -- the rule
    compares a patient to their own earlier scans, so a space change mid-trajectory silently
    corrupts every ratio. Use one space (atlas throughout, or native throughout) and never mix.
    """

    timepoint: str
    """Opaque label, echoed back on the assessment. Ordering comes from ``week``, not from this."""

    week: float | None = None
    """Weeks since the patient's first scan. Required for confirmation windows; ``None`` disables
    them for this timepoint (reported as an unknown rather than assumed satisfied)."""

    enhancing_mm3: float | None = None
    """Contrast-enhancing tumour volume. The primary RANO signal."""

    t2_flair_mm3: float | None = None
    """Non-enhancing T2/FLAIR hyperintensity volume. RANO component 2."""

    new_lesion: bool | None = None
    """A lesion absent from all prior scans. RANO component 3. Needs connected-component
    tracking across timepoints -- a volume table alone can never supply it, so ``None`` here is
    the normal, honest state for volume-only pipelines."""

    weeks_since_rt: float | None = None
    """Weeks since completion of radiotherapy. Drives the pseudoprogression window."""

    non_measurable_only: bool | None = None
    """True when the only disease present is below the measurability floor. RANO forbids calling
    PR on non-measurable disease -- unequivocal PD and SD remain available."""

    clinical_deterioration: bool | None = None
    """Definite clinical worsening not attributable to other causes. A RANO PD criterion in its
    own right, and one no imaging pipeline can supply."""

    steroids_increased: bool | None = None
    """Steroid dose increased since the reference scan. Blocks CR and (in strict readings) PR."""


@dataclass(frozen=True, slots=True)
class ResponseAssessment:
    """One call, plus everything needed to defend or debug it.

    The ratios are stored because "why did this say PD?" is the question that gets asked about
    every disagreement, and recomputing them from the trajectory afterwards is both tedious and a
    chance to recompute them differently.
    """

    timepoint: str
    call: Response
    reason: str
    """One line, human-readable, naming the component that decided the call."""

    driver: str
    """Machine-readable component that decided it: one of ``enhancing``, ``t2_flair``,
    ``new_lesion``, ``clinical``, ``confirmation``, ``reference``, ``missing``."""

    enhancing_mm3: float | None = None
    change_vs_nadir: float | None = None
    """(enhancing - nadir) / nadir. ``None`` when nadir is 0 or either side is missing."""
    change_vs_baseline: float | None = None
    """(enhancing - baseline) / baseline. ``None`` when baseline is 0 or either side is missing."""
    t2_change_vs_nadir: float | None = None

    unknowns: tuple[str, ...] = ()
    """Components that could not be evaluated. Empty tuple == every component was available."""

    pseudoprogression_risk: bool = False
    """PD called inside the post-radiotherapy window where pseudoprogression is expected. Under
    RANO such a call is not reliable without tissue or an out-of-field lesion."""

    provisional_call: Response | None = None
    """What the timepoint scored before confirmation logic; set only when confirmation changed it."""

    @property
    def complete(self) -> bool:
        """True when no RANO component had to be skipped for want of data."""
        return not self.unknowns

    @property
    def scorable(self) -> bool:
        """True when this call can be compared against an expert rating."""
        return self.call in SCORABLE


@dataclass(frozen=True, slots=True)
class TrajectoryResult:
    """One patient's assessments plus the reference state they were made against."""

    patient: str
    assessments: tuple[ResponseAssessment, ...]
    baseline_timepoint: str | None
    baseline_mm3: float | None
    baseline_scored: bool
    """False when the baseline had to be taken from the first assessable scan, which therefore
    could not itself be scored. True when a dedicated reference scan (e.g. post-op) was supplied."""

    def scorable(self) -> tuple[ResponseAssessment, ...]:
        return tuple(a for a in self.assessments if a.scorable)
