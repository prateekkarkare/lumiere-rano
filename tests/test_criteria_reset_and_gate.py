"""
Two rules that decide what the reference means: it restarts at every operation, and a response
cannot be called against a reference that held nothing measurable.

Both fail silently when wrong -- the rule keeps emitting plausible categories -- so each test
targets the specific wrong answer, not just "returns something".
"""

from __future__ import annotations

import pytest

from rano.contract.treatment import RadiotherapyCourse, TreatmentRecord, TreatmentWeek
from rano.criteria import (
    MRANO_VOLUMETRIC,
    ReferenceState,
    Response,
    TimepointMeasurement,
    assess_timepoint,
    assess_trajectory,
)

#: no confirmation step in the way, so these tests see the raw call
RULE = MRANO_VOLUMETRIC.variant("test_rule", require_confirmation=False, pseudoprogression_weeks=None)


def tp(name, enh, week=None, **kw):
    return TimepointMeasurement(timepoint=name, enhancing_mm3=enh, week=week, **kw)


def surgeries(*weeks) -> TreatmentRecord:
    return TreatmentRecord(
        surgeries=tuple(TreatmentWeek(w) for w in weeks),
        radiotherapy=(RadiotherapyCourse(TreatmentWeek(4.0), TreatmentWeek(10.0)),),
    )


def calls(result):
    return {a.timepoint: a.call for a in result.assessments}


# --------------------------------------------------------------------------------------
# the reference restarts at an operation
# --------------------------------------------------------------------------------------

def test_the_scan_after_an_operation_becomes_the_new_reference_and_is_not_scored():
    ms = [tp("wk20", 4000.0, 20.0), tp("wk41", 500.0, 41.0), tp("wk57", 600.0, 57.0)]
    result = assess_trajectory("P", ms, RULE, reference=tp("wk13", 3000.0, 13.0), treatment=surgeries(38.0))
    assert calls(result)["wk41"] is Response.BASELINE
    assert "operation at week 38" in next(a.reason for a in result.assessments if a.timepoint == "wk41")


def test_progression_is_not_called_against_a_nadir_from_before_the_operation():
    """Patient-004's shape: 175 mm3 thirty weeks and one operation ago is not a valid nadir."""
    ms = [tp("wk20", 4031.0, 20.0), tp("wk41", 3690.0, 41.0), tp("wk71", 634.0, 71.0)]
    ref = tp("wk13", 175.0, 13.0)
    without = assess_trajectory("P", ms, RULE, reference=ref)
    withreset = assess_trajectory("P", ms, RULE, reference=ref, treatment=surgeries(40.0))
    assert calls(without)["wk71"] is Response.PD          # +262% above a nadir that was cut out
    assert calls(withreset)["wk71"] is Response.PR        # -83% against the post-operative scan


def test_without_a_treatment_record_nothing_resets():
    ms = [tp("wk20", 4031.0, 20.0), tp("wk41", 3690.0, 41.0), tp("wk71", 634.0, 71.0)]
    result = assess_trajectory("P", ms, RULE, reference=tp("wk13", 175.0, 13.0), treatment=None)
    assert calls(result)["wk41"] is not Response.BASELINE


def test_the_switch_turns_the_reset_off():
    ms = [tp("wk41", 500.0, 41.0)]
    off = RULE.variant("no_reset", reset_reference_at_surgery=False)
    result = assess_trajectory("P", ms, off, reference=tp("wk13", 3000.0, 13.0), treatment=surgeries(38.0))
    assert calls(result)["wk41"] is not Response.BASELINE


def test_an_operation_before_the_reference_scan_does_not_reset_anything():
    """The trajectory's own baseline already sits after that operation."""
    ms = [tp("wk20", 4000.0, 20.0)]
    result = assess_trajectory("P", ms, RULE, reference=tp("wk13", 3000.0, 13.0), treatment=surgeries(0.0))
    assert calls(result)["wk20"] is Response.SD


def test_every_operation_resets_not_only_the_first():
    ms = [tp("wk20", 4000.0, 20.0), tp("wk41", 500.0, 41.0), tp("wk75", 800.0, 75.0)]
    result = assess_trajectory("P", ms, RULE, reference=tp("wk13", 3000.0, 13.0), treatment=surgeries(38.0, 70.0))
    assert calls(result)["wk41"] is Response.BASELINE and calls(result)["wk75"] is Response.BASELINE


# --------------------------------------------------------------------------------------
# a response needs a measurable reference
# --------------------------------------------------------------------------------------

def test_no_measurable_disease_at_the_reference_means_no_partial_response():
    """Patient-012's shape: 110 mm3 of specks at baseline, 20 mm3 now. -82% is not a response."""
    ref = ReferenceState(baseline_mm3=110.0, nadir_mm3=110.0, baseline_measurable=False)
    a = assess_timepoint(tp("wk48", 20.0), ref, RULE)
    assert a.call is Response.SD and a.provisional_call is Response.PR
    assert "no lesion big enough to measure" in a.reason


def test_a_measurable_reference_still_allows_the_response():
    ref = ReferenceState(baseline_mm3=2725.0, nadir_mm3=2725.0, baseline_measurable=True)
    assert assess_timepoint(tp("wk29", 84.0), ref, RULE).call is Response.PR


def test_the_follow_up_shrinking_below_measurable_is_a_response_not_a_block():
    """The gate is on the reference. Testing the follow-up would refuse every real response."""
    ref = ReferenceState(baseline_mm3=2725.0, nadir_mm3=2725.0, baseline_measurable=True)
    a = assess_timepoint(tp("wk29", 84.0, measurable_disease=False), ref, RULE)
    assert a.call is Response.PR


def test_unmeasured_reference_keeps_the_response_and_admits_the_gap():
    ref = ReferenceState(baseline_mm3=2725.0, nadir_mm3=2725.0, baseline_measurable=None)
    a = assess_timepoint(tp("wk29", 84.0), ref, RULE)
    assert a.call is Response.PR and "baseline_measurable" in a.unknowns


def test_the_gate_never_blocks_progression_or_complete_response():
    ref = ReferenceState(baseline_mm3=110.0, nadir_mm3=110.0, baseline_measurable=False)
    assert assess_timepoint(tp("t", 400.0), ref, RULE).call is Response.PD
    assert assess_timepoint(tp("t", 0.0), ref, RULE).call is Response.CR


def test_the_reset_carries_the_new_reference_measurability():
    """After an operation the gate must follow the NEW reference, not the old one."""
    ms = [tp("wk41", 100.0, 41.0, measurable_disease=False), tp("wk57", 20.0, 57.0)]
    result = assess_trajectory(
        "P", ms, RULE, reference=tp("wk13", 3000.0, 13.0, measurable_disease=True), treatment=surgeries(38.0)
    )
    assert calls(result)["wk57"] is Response.SD    # -80% vs the new reference, but it was specks


def test_the_switch_turns_the_gate_off():
    off = RULE.variant("no_gate", require_measurable_baseline=False)
    ref = ReferenceState(baseline_mm3=110.0, nadir_mm3=110.0, baseline_measurable=False)
    assert assess_timepoint(tp("wk48", 20.0), ref, off).call is Response.PR
