"""
Treatment history — the dates a scan cannot tell you.

Whether a scan may serve as a RANO baseline depends on WHEN it was taken relative to surgery and
radiotherapy, and neither event is written into an MR image: the DICOM header carries the scan
date, while surgery lives in the surgical record and radiotherapy in the radiotherapy system. So
treatment history is a required INPUT, delivered alongside the scans — never inferred from pixels.

TIME AXIS
    Every week here sits on the same axis as ``Timepoint.week_offset``: weeks since the patient's
    first scan. An adapter that receives calendar dates converts them onto this axis; the core
    never sees a date. Weeks may be NEGATIVE — a patient whose first scan reaches us after an
    operation elsewhere had that operation before week 0.

THREE STATES, NOT TWO
    Each date is recorded, assumed, or missing:

        recorded   delivered by a clinical system                    TreatmentWeek(12.0)
        assumed    not delivered; the DATA SOURCE declares a stand-in TreatmentWeek(10.0, assumption="...")
        missing    not delivered, nothing declared                   None

    An assumed date cannot be built without its reason, so an assumption is always an explicit,
    readable declaration by whoever supplied the data — never a silent default. Anything decided
    from an assumed date must carry that reason forward; ``TreatmentRecord.assumptions`` collects
    them for exactly that purpose.

    For the lists, ``None`` and ``()`` are different facts: ``surgeries=None`` means nobody told us,
    ``surgeries=()`` means we were told there was no surgery.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class TreatmentWeek:
    """When a treatment event happened, and whether that is a record or a declared assumption."""

    week: float
    assumption: str | None = None
    """``None`` for a recorded date. For an assumed one: WHY this number, in words a reviewer can
    check (e.g. "radiotherapy dates not shipped; standard schedule: surgery + 10 weeks")."""

    def __post_init__(self) -> None:
        if not math.isfinite(self.week):
            raise ValueError(f"treatment week must be a finite number, got {self.week!r}")
        if self.assumption is not None and not self.assumption.strip():
            raise ValueError("an assumed treatment week must state its assumption; got a blank reason")

    @property
    def is_assumed(self) -> bool:
        return self.assumption is not None


@dataclass(frozen=True, slots=True)
class RadiotherapyCourse:
    """One course of radiotherapy. Either end may be missing on its own."""

    start: TreatmentWeek | None = None
    end: TreatmentWeek | None = None

    def __post_init__(self) -> None:
        if self.start is not None and self.end is not None and self.end.week < self.start.week:
            raise ValueError(
                f"radiotherapy ends (week {self.end.week:g}) before it starts (week {self.start.week:g})"
            )


@dataclass(frozen=True, slots=True)
class TreatmentRecord:
    """Everything delivered about a patient's treatment. The default is 'nothing delivered'."""

    surgeries: tuple[TreatmentWeek, ...] | None = None
    radiotherapy: tuple[RadiotherapyCourse, ...] | None = None

    def __post_init__(self) -> None:
        if self.surgeries is not None:
            weeks = [s.week for s in self.surgeries]
            if weeks != sorted(weeks):
                raise ValueError(f"surgeries must be in chronological order, got weeks {weeks}")

    @property
    def assumptions(self) -> tuple[str, ...]:
        """Every declared assumption in this record, once each, in the order they appear."""
        dates: list[TreatmentWeek | None] = list(self.surgeries or ())
        for course in self.radiotherapy or ():
            dates += [course.start, course.end]
        return tuple(dict.fromkeys(d.assumption for d in dates if d is not None and d.is_assumed))


__all__ = ["TreatmentWeek", "RadiotherapyCourse", "TreatmentRecord"]
