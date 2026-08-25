"""
Unit tests for the RANO rule.

Every test below targets a way this rule can be wrong while still producing plausible output.
That is the whole risk profile here: a response classifier never crashes, it just quietly returns
the wrong category forever. Tests that only check "returns one of CR/PR/SD/PD" would pass on a
rule that always says SD.
"""

from __future__ import annotations

import pytest

from rano.criteria import (
    ENHANCING_ONLY,
    MRANO_VOLUMETRIC,
    ReferenceState,
    Response,
    TimepointMeasurement,
    assess_timepoint,
    assess_trajectory,
)

BARE = ENHANCING_ONLY  # no confirmation, no extra components -- isolates the volume arithmetic


def tp(name, enh, **kw):
    return TimepointMeasurement(timepoint=name, enhancing_mm3=enh, **kw)


# --------------------------------------------------------------------------------------
# the two reference points
# --------------------------------------------------------------------------------------

def test_progression_is_measured_against_nadir_not_baseline():
    """A tumour well below baseline can still progress. Using baseline for PD misses it."""
    ref = ReferenceState(baseline_mm3=10000.0, nadir_mm3=1000.0)
    a = assess_timepoint(tp("t", 1500.0), ref, BARE)
    assert a.call is Response.PD
    assert a.change_vs_nadir == pytest.approx(0.5)
    assert a.change_vs_baseline == pytest.approx(-0.85)


def test_response_is_measured_against_baseline_not_nadir():
    """Shrinking to 30% of baseline is PR even though it is 20% above the nadir."""
    ref = ReferenceState(baseline_mm3=10000.0, nadir_mm3=2500.0)
    a = assess_timepoint(tp("t", 3000.0), ref, BARE)
    assert a.call is Response.PR


def test_progression_takes_precedence_over_response():
    """Both thresholds can fire at once. A growing lesion is not a partial response."""
    ref = ReferenceState(baseline_mm3=10000.0, nadir_mm3=1000.0)
    a = assess_timepoint(tp("t", 2000.0), ref, BARE)  # -80% vs baseline, +100% vs nadir
    assert a.call is Response.PD


def test_nadir_never_includes_the_timepoint_being_assessed():
    """If the current volume could set its own nadir, the ratio is >= 0 always and PD is
    unreachable -- a bug that produces plausible output forever."""
    traj = [tp("t0", 1000.0), tp("t1", 500.0), tp("t2", 900.0)]
    result = assess_trajectory("p", traj, BARE)
    calls = {a.timepoint: a.call for a in result.assessments}
    assert calls["t1"] is Response.PR or calls["t1"] is Response.SD
    assert calls["t2"] is Response.PD  # +80% over the t1 nadir


def test_nadir_tracks_the_minimum_not_the_previous_scan():
    traj = [tp("t0", 1000.0), tp("t1", 200.0), tp("t2", 260.0), tp("t3", 300.0)]
    calls = {a.timepoint: a.call for a in assess_trajectory("p", traj, BARE).assessments}
    # t2 is +30% over the nadir of 200 -- under the +40% PD bar -- and still -74% against the
    # baseline of 1000, so the correct answer is a continuing partial response, not progression.
    assert calls["t2"] is Response.PR
    # +50% over the NADIR of 200. Against the previous scan (260) it is only +15%, so a rule that
    # tracked the previous scan instead of the minimum would call this stable and miss it.
    assert calls["t3"] is Response.PD


# --------------------------------------------------------------------------------------
# complete response and re-emergence
# --------------------------------------------------------------------------------------

def test_zero_enhancing_is_complete_response():
    a = assess_timepoint(tp("t", 0.0), ReferenceState(10000.0, 5000.0), BARE)
    assert a.call is Response.CR


def test_reemergence_after_complete_response_is_progression():
    """Any enhancing disease after a nadir of zero is PD. A ratio cannot express this -- the
    denominator is zero -- so it needs its own branch, and that branch is easy to omit."""
    a = assess_timepoint(tp("t", 50.0), ReferenceState(10000.0, 0.0), BARE)
    assert a.call is Response.PD


def test_zero_after_zero_is_still_complete_response():
    a = assess_timepoint(tp("t", 0.0), ReferenceState(10000.0, 0.0), BARE)
    assert a.call is Response.CR


def test_missing_enhancing_volume_is_not_evaluable_not_stable():
    a = assess_timepoint(tp("t", None), ReferenceState(10000.0, 5000.0), BARE)
    assert a.call is Response.NE
    assert not a.scorable


# --------------------------------------------------------------------------------------
# unknown inputs must not read as negative
# --------------------------------------------------------------------------------------

def test_unknown_new_lesion_is_recorded_on_a_non_progression_call():
    a = assess_timepoint(tp("t", 5000.0, new_lesion=None), ReferenceState(10000.0, 5000.0), MRANO_VOLUMETRIC)
    assert a.call is Response.SD
    assert "new_lesion" in a.unknowns
    assert not a.complete


def test_known_negative_new_lesion_yields_a_complete_call():
    m = tp("t", 5000.0, new_lesion=False, clinical_deterioration=False)
    a = assess_timepoint(m, ReferenceState(10000.0, 5000.0), MRANO_VOLUMETRIC)
    assert a.call is Response.SD
    assert a.complete


def test_unknowns_are_not_reported_when_they_could_not_change_the_call():
    """A missing new-lesion flag cannot un-progress an already-PD scan. Reporting it there would
    mark nearly every assessment incomplete and the flag would stop meaning anything."""
    a = assess_timepoint(tp("t", 9000.0, new_lesion=None), ReferenceState(10000.0, 5000.0), MRANO_VOLUMETRIC)
    assert a.call is Response.PD
    assert a.unknowns == ()


def test_new_lesion_alone_causes_progression():
    a = assess_timepoint(tp("t", 100.0, new_lesion=True), ReferenceState(10000.0, 5000.0), MRANO_VOLUMETRIC)
    assert a.call is Response.PD
    assert a.driver == "new_lesion"


def test_clinical_deterioration_alone_causes_progression():
    m = tp("t", 100.0, new_lesion=False, clinical_deterioration=True)
    a = assess_timepoint(m, ReferenceState(10000.0, 5000.0), MRANO_VOLUMETRIC)
    assert a.call is Response.PD
    assert a.driver == "clinical"


# --------------------------------------------------------------------------------------
# blocks
# --------------------------------------------------------------------------------------

def test_non_measurable_disease_blocks_partial_response():
    m = tp("t", 3000.0, non_measurable_only=True)
    a = assess_timepoint(m, ReferenceState(10000.0, 10000.0), MRANO_VOLUMETRIC)
    assert a.call is Response.SD
    assert a.provisional_call is Response.PR


def test_steroid_increase_blocks_complete_response():
    m = tp("t", 0.0, steroids_increased=True)
    a = assess_timepoint(m, ReferenceState(10000.0, 5000.0), MRANO_VOLUMETRIC)
    assert a.call is Response.PR


def test_absolute_guard_for_response_is_measured_against_the_baseline():
    """Regression: one guard computed against the nadir was applied to the PR branch, whose ratio
    is against the baseline. Here the change vs baseline is 9700 mm3 -- far past the guard -- while
    the change vs nadir is only 100 mm3, so a nadir-measured guard would wrongly suppress the PR."""
    criteria = ENHANCING_ONLY.variant("guarded", min_absolute_change_mm3=500.0)
    a = assess_timepoint(tp("t", 300.0), ReferenceState(baseline_mm3=10000.0, nadir_mm3=400.0), criteria)
    assert a.call is Response.PR


def test_absolute_guard_suppresses_tiny_ratio_swings():
    """A 0.4 cm3 lesion can swing tens of percent on slice thickness alone.

    Baseline is set to 400 so the response branch cannot fire (-25% vs baseline), isolating the
    progression guard: +50% over the nadir, but only 100 mm3 of it.
    """
    criteria = BARE.variant("guarded", min_absolute_change_mm3=500.0)
    a = assess_timepoint(tp("t", 300.0), ReferenceState(baseline_mm3=400.0, nadir_mm3=200.0), criteria)
    assert a.call is Response.SD


# --------------------------------------------------------------------------------------
# confirmation
# --------------------------------------------------------------------------------------

def test_unconfirmed_response_becomes_stable_disease():
    traj = [
        tp("t0", 10000.0, week=0),
        tp("t1", 2000.0, week=8),   # provisional PR: -80% vs baseline
        tp("t2", 4000.0, week=16),  # +100% vs nadir -> PD, so it cannot confirm t1
    ]
    calls = {a.timepoint: a.call for a in assess_trajectory("p", traj, MRANO_VOLUMETRIC).assessments}
    assert calls["t1"] is Response.SD


def test_confirmed_response_survives():
    traj = [
        tp("t0", 10000.0, week=0),
        tp("t1", 2000.0, week=8),
        tp("t2", 1900.0, week=16),
    ]
    result = assess_trajectory("p", traj, MRANO_VOLUMETRIC)
    calls = {a.timepoint: a.call for a in result.assessments}
    assert calls["t1"] is Response.PR


def test_partial_response_does_not_confirm_a_complete_response():
    """CR needs CR. Accepting PR as confirmation would over-call complete responses."""
    traj = [tp("t0", 10000.0, week=0), tp("t1", 0.0, week=8), tp("t2", 500.0, week=16)]
    calls = {a.timepoint: a.call for a in assess_trajectory("p", traj, MRANO_VOLUMETRIC).assessments}
    assert calls["t1"] is not Response.CR


def test_scan_too_soon_does_not_confirm():
    """The confirming scan must be at least confirmation_weeks out; a 1-week repeat is not one.

    t2 would confirm t1 on volume alone -- it is only the 1-week interval that disqualifies it, so
    the later PD at t3 becomes the confirming scan and t1 falls back to SD.
    """
    traj = [
        tp("t0", 10000.0, week=0),
        tp("t1", 2000.0, week=8),
        tp("t2", 1900.0, week=9),   # a response, but far too soon to confirm anything
        tp("t3", 9000.0, week=20),  # the first eligible scan, and it is progression
    ]
    calls = {a.timepoint: a.call for a in assess_trajectory("p", traj, MRANO_VOLUMETRIC).assessments}
    assert calls["t1"] is Response.SD


def test_response_at_end_of_followup_is_kept_and_flagged():
    """Downgrading every trajectory's last scan penalises follow-up length, not biology."""
    traj = [tp("t0", 10000.0, week=0), tp("t1", 2000.0, week=8)]
    a = assess_trajectory("p", traj, MRANO_VOLUMETRIC).assessments[-1]
    assert a.call is Response.PR
    assert "confirmation_followup" in a.unknowns


def test_end_of_followup_downgrade_when_configured():
    criteria = MRANO_VOLUMETRIC.variant("strict", confirm_at_end_of_followup=False)
    traj = [tp("t0", 10000.0, week=0), tp("t1", 2000.0, week=8)]
    a = assess_trajectory("p", traj, criteria).assessments[-1]
    assert a.call is Response.SD
    assert a.provisional_call is Response.PR


# --------------------------------------------------------------------------------------
# pseudoprogression
# --------------------------------------------------------------------------------------

def test_pseudoprogression_window_flags_but_keeps_the_call_by_default():
    m = tp("t", 9000.0, weeks_since_rt=4.0)
    a = assess_timepoint(m, ReferenceState(10000.0, 5000.0), MRANO_VOLUMETRIC)
    assert a.call is Response.PD
    assert a.pseudoprogression_risk


def test_pseudoprogression_downgrade_policy():
    criteria = MRANO_VOLUMETRIC.variant("strict", pseudoprogression_policy="downgrade")
    a = assess_timepoint(tp("t", 9000.0, weeks_since_rt=4.0), ReferenceState(10000.0, 5000.0), criteria)
    assert a.call is Response.SD
    assert a.provisional_call is Response.PD


def test_outside_the_window_is_not_flagged():
    a = assess_timepoint(tp("t", 9000.0, weeks_since_rt=30.0), ReferenceState(10000.0, 5000.0), MRANO_VOLUMETRIC)
    assert a.call is Response.PD
    assert not a.pseudoprogression_risk


# --------------------------------------------------------------------------------------
# trajectory plumbing
# --------------------------------------------------------------------------------------

def test_dedicated_reference_scores_every_measurement():
    ref = tp("post-op", 8000.0, week=0)
    traj = [tp("t1", 8000.0, week=8), tp("t2", 8200.0, week=16)]
    result = assess_trajectory("p", traj, BARE, reference=ref)
    assert result.baseline_scored
    assert result.baseline_timepoint == "post-op"
    assert [a.timepoint for a in result.assessments] == ["t1", "t2"]
    assert all(a.scorable for a in result.assessments)


def test_without_a_reference_the_first_scan_is_consumed_as_baseline():
    traj = [tp("t0", 8000.0, week=0), tp("t1", 8000.0, week=8)]
    result = assess_trajectory("p", traj, BARE)
    assert not result.baseline_scored
    assert result.assessments[0].call is Response.BASELINE
    assert not result.assessments[0].scorable


def test_empty_trajectory_is_not_an_error():
    result = assess_trajectory("p", [], BARE)
    assert result.assessments == ()
    assert result.baseline_timepoint is None


def test_profile_variant_does_not_mutate_the_shared_profile():
    before = MRANO_VOLUMETRIC.pd_increase
    other = MRANO_VOLUMETRIC.variant("other", pd_increase=0.99)
    assert other.pd_increase == 0.99
    assert MRANO_VOLUMETRIC.pd_increase == before
    assert other.name == "other"
