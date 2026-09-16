"""
Where does a scan sit in the patient's treatment, and may it be the starting point?

The pipeline measures every later scan against ONE reference scan. Which scan that is, is not a
property of the image: it depends on when the scan was taken relative to surgery and radiotherapy.
This module answers that, and nothing else.

    before surgery              tumour still untreated                     not a starting point
    after surgery               operated, radiotherapy not started yet     not a starting point
    during radiotherapy         treatment still running                    not a starting point
    too soon after radiotherapy treatment effects still settling           not a starting point
    after radiotherapy          the first clear picture                    STARTING POINT
    unknown                     treatment dates missing                    nothing is scored

WHERE THE WINDOW EDGES COME FROM
    RANO 2.0 puts the baseline at the first scan about 4 weeks after radiotherapy ends. Read as a
    narrow 3-5 week window it would disqualify most real follow-up: on LUMIERE, 31 of 62 patients
    have their first rated scan more than 15 weeks after the first scan. So the rule here is a
    FLOOR, not a window -- at least ``min_weeks_after_rt`` past the end of radiotherapy -- and a
    scan further out than ``late_after_weeks`` is still the starting point, only flagged ``late``.
    Disqualifying it would leave those patients with no reference at all, which is worse than a
    reference the report says arrived late.

RE-OPERATIONS
    A scan taken after a LATER operation reads as "after surgery" again, not as a follow-up. What
    that means for the reference (RANO resets it at every surgery) is a separate change; this
    module only reports the placement.

ASSUMED DATES
    A placement made from an assumed date carries that assumption forward in ``assumptions`` so a
    call built on it can say so. See ``rano.contract.treatment``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from rano.contract.treatment import TreatmentRecord


class Stage(StrEnum):
    """Where a scan sits in the treatment. Only ``AFTER_RADIOTHERAPY`` may be a starting point."""

    BEFORE_SURGERY = "before surgery"
    AFTER_SURGERY = "after surgery, before radiotherapy"
    DURING_RADIOTHERAPY = "during radiotherapy"
    TOO_SOON_AFTER_RADIOTHERAPY = "too soon after radiotherapy"
    AFTER_RADIOTHERAPY = "after radiotherapy"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class BaselinePolicy:
    """How long after radiotherapy a scan must wait before it can be the starting point."""

    min_weeks_after_rt: float = 3.0
    """RANO 2.0's post-radiotherapy baseline is ~4 weeks out; 3 admits a scan booked a little early."""
    late_after_weeks: float = 5.0
    """Beyond this the scan is still the starting point, but flagged as late."""


@dataclass(frozen=True, slots=True)
class Placement:
    """One scan, placed."""

    stage: Stage
    reason: str
    weeks_since_rt_end: float | None = None
    late: bool = False
    assumptions: tuple[str, ...] = ()

    @property
    def can_be_starting_point(self) -> bool:
        return self.stage is Stage.AFTER_RADIOTHERAPY


def place_scan(
    week: float, treatment: TreatmentRecord, policy: BaselinePolicy = BaselinePolicy()
) -> Placement:
    """Place one scan on the treatment timeline. ``week`` is on the patient's own week axis."""
    surgeries = treatment.surgeries
    courses = treatment.radiotherapy
    assumptions = treatment.assumptions

    def placed(stage: Stage, reason: str, **extra) -> Placement:
        return Placement(stage, reason, assumptions=assumptions, **extra)

    if surgeries is None and not courses:
        return Placement(Stage.UNKNOWN, "no treatment record was delivered")

    if surgeries:
        if week < surgeries[0].week:
            return placed(Stage.BEFORE_SURGERY, f"before the first operation (week {surgeries[0].week:g})")
        last = max(s.week for s in surgeries if s.week <= week)
    else:
        last = None

    if not courses:
        return placed(Stage.UNKNOWN, "no radiotherapy dates, so the scan cannot be placed against them")
    course = courses[0]
    if course.end is None or course.start is None:
        return placed(Stage.UNKNOWN, "radiotherapy start or end is missing")

    if last is not None and last > course.end.week:
        return placed(Stage.AFTER_SURGERY, f"follows the operation at week {last:g}")
    if week < course.start.week:
        return placed(Stage.AFTER_SURGERY, f"before radiotherapy starts (week {course.start.week:g})")
    if week <= course.end.week:
        return placed(Stage.DURING_RADIOTHERAPY, f"radiotherapy runs to week {course.end.week:g}")

    gap = week - course.end.week
    if gap < policy.min_weeks_after_rt:
        return placed(
            Stage.TOO_SOON_AFTER_RADIOTHERAPY,
            f"only {gap:g} wk after radiotherapy (needs {policy.min_weeks_after_rt:g})",
            weeks_since_rt_end=gap,
        )
    late = gap > policy.late_after_weeks
    return placed(
        Stage.AFTER_RADIOTHERAPY,
        f"{gap:g} wk after radiotherapy" + (" -- later than a scheduled baseline" if late else ""),
        weeks_since_rt_end=gap,
        late=late,
    )


@dataclass(frozen=True, slots=True)
class BaselineChoice:
    """Which scan becomes the starting point, and why — including when none can."""

    index: int | None
    placement: Placement | None
    reason: str

    @property
    def found(self) -> bool:
        return self.index is not None


def choose_baseline(
    weeks: list[float | None], treatment: TreatmentRecord, policy: BaselinePolicy = BaselinePolicy()
) -> BaselineChoice:
    """The FIRST scan that may be a starting point. Chronological order is the caller's job.

    First, not best: picking a later scan because it looks like a friendlier reference would make
    every subsequent response measurement flattering.
    """
    for i, week in enumerate(weeks):
        if week is None:
            continue
        placement = place_scan(week, treatment, policy)
        if placement.can_be_starting_point:
            return BaselineChoice(i, placement, f"week {week:g}: {placement.reason}")
    if not weeks:
        return BaselineChoice(None, None, "the patient has no scans")
    last = place_scan(max(w for w in weeks if w is not None), treatment, policy) if any(
        w is not None for w in weeks
    ) else None
    return BaselineChoice(
        None, last, "no scan qualifies as a starting point" + (f" ({last.reason})" if last else "")
    )


__all__ = ["Stage", "BaselinePolicy", "Placement", "place_scan", "BaselineChoice", "choose_baseline"]
