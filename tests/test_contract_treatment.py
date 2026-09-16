"""Treatment record: recorded / assumed / missing, and the rules that keep them apart."""

import math

import pytest

from rano.contract import Patient, RadiotherapyCourse, TreatmentRecord, TreatmentWeek


# ---------------------------------------------------------------------------- one date
def test_recorded_week_is_not_an_assumption():
    w = TreatmentWeek(12.0)
    assert not w.is_assumed and w.assumption is None


def test_assumed_week_carries_its_reason():
    w = TreatmentWeek(10.0, assumption="radiotherapy dates not shipped")
    assert w.is_assumed and w.assumption == "radiotherapy dates not shipped"


@pytest.mark.parametrize("blank", ["", "   "])
def test_assumption_without_a_reason_is_refused(blank):
    # "Only when explicitly declared": an assumption you cannot read is not a declaration.
    with pytest.raises(ValueError, match="must state its assumption"):
        TreatmentWeek(10.0, assumption=blank)


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_week_must_be_finite(bad):
    with pytest.raises(ValueError, match="finite"):
        TreatmentWeek(bad)


def test_negative_week_is_allowed():
    # First scan received after an operation elsewhere: the operation predates week 0.
    assert TreatmentWeek(-3.0).week == -3.0


# ---------------------------------------------------------------------------- radiotherapy
def test_radiotherapy_cannot_end_before_it_starts():
    with pytest.raises(ValueError, match="ends"):
        RadiotherapyCourse(start=TreatmentWeek(10.0), end=TreatmentWeek(4.0))


def test_radiotherapy_ends_may_be_missing_independently():
    assert RadiotherapyCourse(start=TreatmentWeek(4.0)).end is None
    assert RadiotherapyCourse(end=TreatmentWeek(10.0)).start is None


# ---------------------------------------------------------------------------- the record
def test_default_record_means_nothing_was_delivered():
    r = TreatmentRecord()
    assert r.surgeries is None and r.radiotherapy is None and r.assumptions == ()


def test_none_and_empty_are_different_facts():
    unknown, none_happened = TreatmentRecord(), TreatmentRecord(surgeries=(), radiotherapy=())
    assert unknown.surgeries is None and none_happened.surgeries == ()
    assert unknown != none_happened


def test_surgeries_must_be_chronological():
    with pytest.raises(ValueError, match="chronological"):
        TreatmentRecord(surgeries=(TreatmentWeek(41.0), TreatmentWeek(0.0)))


def test_assumptions_are_collected_once_each_in_order():
    rt = "radiotherapy dates not shipped"
    r = TreatmentRecord(
        surgeries=(TreatmentWeek(0.0), TreatmentWeek(41.0, assumption="surgery date approximated")),
        radiotherapy=(RadiotherapyCourse(TreatmentWeek(4.0, assumption=rt), TreatmentWeek(10.0, assumption=rt)),),
    )
    assert r.assumptions == ("surgery date approximated", rt)


def test_fully_recorded_record_has_no_assumptions():
    r = TreatmentRecord(
        surgeries=(TreatmentWeek(0.0),),
        radiotherapy=(RadiotherapyCourse(TreatmentWeek(4.0), TreatmentWeek(10.0)),),
    )
    assert r.assumptions == ()


# ---------------------------------------------------------------------------- on the patient
def test_patient_without_a_record_gets_nothing_delivered():
    assert Patient("P001", ()).treatment == TreatmentRecord()


def test_patient_carries_the_delivered_record():
    r = TreatmentRecord(surgeries=(TreatmentWeek(0.0),))
    assert Patient("P002", (), treatment=r).treatment is r
