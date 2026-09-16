"""Placing a scan on the treatment timeline, and picking the starting point."""

from __future__ import annotations

import pytest

from rano.contract.treatment import RadiotherapyCourse, TreatmentRecord, TreatmentWeek
from rano.criteria.timeline import BaselinePolicy, Stage, choose_baseline, place_scan

WHY = "standard schedule assumed"


def record(surgeries=(0.0,), rt=(4.0, 10.0), assumed=True) -> TreatmentRecord:
    why = WHY if assumed else None
    return TreatmentRecord(
        surgeries=tuple(TreatmentWeek(s, why) for s in surgeries),
        radiotherapy=(RadiotherapyCourse(TreatmentWeek(rt[0], why), TreatmentWeek(rt[1], why)),) if rt else None,
    )


# ---------------------------------------------------------------------------- placing one scan
@pytest.mark.parametrize(
    "week, stage",
    [
        (-2.0, Stage.BEFORE_SURGERY),
        (0.0, Stage.AFTER_SURGERY),           # operated, radiotherapy has not started
        (3.9, Stage.AFTER_SURGERY),
        (4.0, Stage.DURING_RADIOTHERAPY),
        (10.0, Stage.DURING_RADIOTHERAPY),    # the last day of treatment is still treatment
        (12.9, Stage.TOO_SOON_AFTER_RADIOTHERAPY),
        (13.0, Stage.AFTER_RADIOTHERAPY),
    ],
)
def test_each_stage(week, stage):
    assert place_scan(week, record()).stage is stage


def test_only_after_radiotherapy_can_be_a_starting_point():
    assert place_scan(13.0, record()).can_be_starting_point
    assert not place_scan(12.0, record()).can_be_starting_point


def test_a_late_scan_still_qualifies_but_is_flagged():
    on_time, late = place_scan(14.0, record()), place_scan(30.0, record())
    assert on_time.can_be_starting_point and not on_time.late
    assert late.can_be_starting_point and late.late and late.weeks_since_rt_end == 20.0


def test_scan_after_a_second_operation_reads_as_after_surgery():
    # A re-operation at week 41 puts week 45 back into a post-surgical state, not a follow-up.
    assert place_scan(45.0, record(surgeries=(0.0, 41.0))).stage is Stage.AFTER_SURGERY


def test_window_edges_are_configurable():
    strict = BaselinePolicy(min_weeks_after_rt=6.0, late_after_weeks=8.0)
    assert place_scan(13.0, record(), strict).stage is Stage.TOO_SOON_AFTER_RADIOTHERAPY
    assert place_scan(16.0, record(), strict).can_be_starting_point


# ---------------------------------------------------------------------------- missing dates
@pytest.mark.parametrize(
    "treatment, fragment",
    [
        (TreatmentRecord(), "no treatment record"),
        (TreatmentRecord(surgeries=(TreatmentWeek(0.0),)), "no radiotherapy dates"),
        (record(rt=None), "no radiotherapy dates"),
    ],
)
def test_missing_dates_leave_the_scan_unplaced(treatment, fragment):
    placed = place_scan(20.0, treatment)
    assert placed.stage is Stage.UNKNOWN and fragment in placed.reason
    assert not placed.can_be_starting_point


# ---------------------------------------------------------------------------- assumptions travel
def test_a_placement_carries_the_assumptions_it_rests_on():
    assert place_scan(20.0, record()).assumptions == (WHY,)
    assert place_scan(20.0, record(assumed=False)).assumptions == ()


# ---------------------------------------------------------------------------- choosing
def test_the_first_qualifying_scan_wins_not_the_best_looking_one():
    choice = choose_baseline([0.0, 6.0, 14.0, 30.0], record())
    assert choice.index == 2 and choice.found


def test_undated_scans_are_skipped():
    assert choose_baseline([None, 14.0], record()).index == 1


def test_no_qualifying_scan_is_reported_with_its_reason():
    choice = choose_baseline([0.0, 6.0, 11.0], record())
    assert not choice.found and "no scan qualifies" in choice.reason


def test_no_scans_at_all():
    assert choose_baseline([], record()).reason == "the patient has no scans"
