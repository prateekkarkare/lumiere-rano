"""
LUMIERE -> TreatmentRecord. The dataset ships no treatment dates, so they are rebuilt from the
only treatment facts it does carry, and every rebuilt date is DECLARED as an assumption.

WHAT IS READ
    From the expert rating file: each row's week, and whether its rating is one of the two
    surgical states, "Pre-Op" or "Post-Op". Nothing else. That file is also the ANSWER KEY
    (CR/PR/SD/PD and the rationale); a treatment record built from those values would leak the
    doctor's call into the pipeline's input, so ``read_surgical_labels`` never keeps them.

SURGERIES
    An operation is placed at the week of the first Post-Op scan after it. Post-Op scans with no
    other scan between them are ONE operation imaged more than once, not several operations
    (Patient-010 is labelled Post-Op at weeks 1, 1 and 2; Patient-085 at weeks 0 and 1).

RADIOTHERAPY
    One course, anchored on the patient's FIRST operation: starts RECOVERY_WEEKS after it, ends
    RECOVERY_WEEKS + CHEMORT_WEEKS after it (standard of care: ~4 weeks recovery, then 6 weeks of
    concurrent chemoradiotherapy). Later operations get no course -- whether a recurrence was
    re-irradiated is not known, and what is not known is not declared.

WHEN THE LABELS CONTRADICT THE SCANS
    A Pre-Op scan followed by an ordinary scan means an operation the labels never dated
    (Patient-079: Pre-Op at week 0, follow-ups from week 1, first Post-Op label at week 53), or a
    label on the wrong week (Patient-013: Pre-Op week 0, scan week 1, Post-Op label week 2). The
    record is then left MISSING and the reason returned, never patched: a guess stacked on a
    contradiction is not a declared assumption.

Everything works on week NUMBERS, never on folder labels. The rating file and the image folders
name the same scan differently in places ("week-000" vs "week-000-2"); matching by label is how the
scoring harness silently failed to find the post-operative scan for five patients.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from rano.adapters.lumiere import weeks
from rano.contract.treatment import RadiotherapyCourse, TreatmentRecord, TreatmentWeek

PRE_OP, POST_OP = "Pre-Op", "Post-Op"

RECOVERY_WEEKS = 4.0
CHEMORT_WEEKS = 6.0

SURGERY_ASSUMPTION = (
    "LUMIERE ships no surgery dates; each operation is placed at the week of the first "
    "Post-Op scan after it"
)
RADIOTHERAPY_ASSUMPTION = (
    "LUMIERE ships no radiotherapy dates; standard schedule assumed from the first operation: "
    f"starts +{RECOVERY_WEEKS:g} weeks, ends +{RECOVERY_WEEKS + CHEMORT_WEEKS:g} weeks"
)


@dataclass(frozen=True, slots=True)
class SurgicalLabels:
    """What the rating file says about one patient's operations -- and deliberately nothing more."""

    pre_op_weeks: frozenset[float]
    post_op_weeks: frozenset[float]
    other_scan_weeks: frozenset[float]
    """Weeks of rated scans that are neither Pre-Op nor Post-Op. Only their existence is kept."""


def read_surgical_labels(ratings_csv: str) -> dict[str, SurgicalLabels]:
    """patient -> surgical labels, from the expert rating file. Response ratings are not kept."""
    pre, post, other = defaultdict(set), defaultdict(set), defaultdict(set)
    with open(ratings_csv, newline="") as fh:
        reader = csv.DictReader(fh)
        rating_col = next(k for k in reader.fieldnames or () if k.startswith("Rating ("))
        for row in reader:
            patient, week = row["Patient"].strip(), weeks.week_offset(row["Date"].strip())
            if week is None:
                continue
            state = row[rating_col].strip()
            (pre if state == PRE_OP else post if state == POST_OP else other)[patient].add(week)
    return {
        p: SurgicalLabels(frozenset(pre[p]), frozenset(post[p]), frozenset(other[p]))
        for p in set(pre) | set(post) | set(other)
    }


def treatment_from_labels(
    labels: SurgicalLabels | None, scan_weeks: Iterable[float] = ()
) -> tuple[TreatmentRecord, str | None]:
    """Build one patient's record. Returns ``(record, problem)``; ``problem`` explains a record left
    missing because the labels contradict the scans, and is ``None`` otherwise.

    ``scan_weeks`` are the weeks of the scans actually delivered (the manifest), which may include
    scans the rating file never mentions.
    """
    if labels is None:
        return TreatmentRecord(), None

    # every scan that is not itself a Post-Op label: these are what separate two operations
    between = {w for w in scan_weeks if w is not None} | labels.other_scan_weeks | labels.pre_op_weeks

    for pre in sorted(labels.pre_op_weeks):
        later = sorted(w for w in between if w > pre)
        if later and not any(pre <= q <= later[0] for q in labels.post_op_weeks):
            return TreatmentRecord(), (
                f"Pre-Op scan at week {pre:g} is followed by a scan at week {later[0]:g} with no "
                f"Post-Op label in between (Post-Op labels at weeks {sorted(labels.post_op_weeks)})"
            )

    surgeries: list[float] = []
    previous: float | None = None
    for q in sorted(labels.post_op_weeks):
        if previous is None or any(previous < w < q for w in between):
            surgeries.append(q)
        previous = q

    if not surgeries:
        return TreatmentRecord(surgeries=()), None

    first = surgeries[0]
    course = RadiotherapyCourse(
        start=TreatmentWeek(first + RECOVERY_WEEKS, RADIOTHERAPY_ASSUMPTION),
        end=TreatmentWeek(first + RECOVERY_WEEKS + CHEMORT_WEEKS, RADIOTHERAPY_ASSUMPTION),
    )
    return (
        TreatmentRecord(
            surgeries=tuple(TreatmentWeek(q, SURGERY_ASSUMPTION) for q in surgeries),
            radiotherapy=(course,),
        ),
        None,
    )


__all__ = [
    "SurgicalLabels",
    "read_surgical_labels",
    "treatment_from_labels",
    "SURGERY_ASSUMPTION",
    "RADIOTHERAPY_ASSUMPTION",
    "RECOVERY_WEEKS",
    "CHEMORT_WEEKS",
]
