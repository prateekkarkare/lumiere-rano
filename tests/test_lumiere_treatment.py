"""LUMIERE treatment rebuild: surgical labels -> TreatmentRecord, with every date declared."""

from __future__ import annotations

import pytest

from rano.adapters.lumiere.adapter import LumiereAdapter
from rano.adapters.lumiere.treatment import (
    RADIOTHERAPY_ASSUMPTION,
    SURGERY_ASSUMPTION,
    SurgicalLabels,
    read_surgical_labels,
    treatment_from_labels,
)
from rano.contract.treatment import TreatmentRecord


def labels(pre=(), post=(), other=()) -> SurgicalLabels:
    return SurgicalLabels(frozenset(pre), frozenset(post), frozenset(other))


def surgery_weeks(record: TreatmentRecord) -> list[float]:
    return [s.week for s in record.surgeries]


# ---------------------------------------------------------------------------- surgeries
def test_standard_patient_one_operation_every_date_declared():
    record, problem = treatment_from_labels(labels(pre=[0], post=[0], other=[14, 27]), scan_weeks=[0, 0, 14, 27])
    assert problem is None
    assert surgery_weeks(record) == [0]
    assert record.surgeries[0].assumption == SURGERY_ASSUMPTION
    assert set(record.assumptions) == {SURGERY_ASSUMPTION, RADIOTHERAPY_ASSUMPTION}


def test_repeat_post_op_scans_are_one_operation():
    # Patient-010 shape: Post-Op at weeks 1 and 2, nothing scanned in between.
    record, _ = treatment_from_labels(labels(pre=[0], post=[1, 2], other=[15]), scan_weeks=[0, 1, 15])
    assert surgery_weeks(record) == [1]


def test_reoperation_is_a_second_surgery():
    # Patient-004 shape: follow-ups between the two Post-Op scans.
    record, _ = treatment_from_labels(labels(pre=[0], post=[0, 41], other=[20, 38, 57]))
    assert surgery_weeks(record) == [0, 41]


def test_pre_op_scan_between_two_post_ops_separates_them():
    record, _ = treatment_from_labels(labels(pre=[0, 40], post=[0, 41]))
    assert surgery_weeks(record) == [0, 41]


def test_labels_but_no_operation_means_told_none_happened():
    record, problem = treatment_from_labels(labels(other=[5, 17]))
    assert problem is None and record.surgeries == () and record.radiotherapy is None


def test_no_labels_at_all_means_nothing_delivered():
    assert treatment_from_labels(None) == (TreatmentRecord(), None)


# ---------------------------------------------------------------------------- radiotherapy
@pytest.mark.parametrize("first_surgery", [0.0, 3.0])
def test_radiotherapy_follows_each_patients_own_first_operation(first_surgery):
    record, _ = treatment_from_labels(labels(pre=[0], post=[first_surgery, 41], other=[20]))
    (course,) = record.radiotherapy
    assert (course.start.week, course.end.week) == (first_surgery + 4, first_surgery + 10)
    assert course.end.assumption == RADIOTHERAPY_ASSUMPTION


# ---------------------------------------------------------------------------- contradictions
def test_pre_op_followed_by_a_plain_scan_leaves_the_record_missing():
    # Patient-079 shape: an operation the labels never dated.
    record, problem = treatment_from_labels(labels(pre=[0], post=[53], other=[1, 14]), scan_weeks=[0, 1, 14, 53])
    assert record == TreatmentRecord()
    assert "Pre-Op scan at week 0" in problem and "week 1" in problem


def test_post_op_label_on_the_wrong_week_is_a_contradiction_too():
    # Patient-013 shape: the scan right after surgery is week 1, the label says week 2.
    record, problem = treatment_from_labels(labels(pre=[0], post=[2]), scan_weeks=[0, 1, 17])
    assert record == TreatmentRecord() and problem is not None


def test_pre_op_as_the_last_scan_is_not_a_contradiction():
    record, problem = treatment_from_labels(labels(pre=[0]), scan_weeks=[0])
    assert problem is None and record.surgeries == ()


# ---------------------------------------------------------------------------- reading the file
RATINGS_HEADER = "Patient,Date,LessThan3Months,NonMeasurableLesions,Rating (according to RANO),Rating rationale"


def test_reader_keeps_surgical_states_and_never_the_response_call(tmp_path):
    csv_path = tmp_path / "ratings.csv"
    csv_path.write_text("\n".join([
        RATINGS_HEADER,
        "Patient-001,week-000-1,,,Pre-Op,",
        "Patient-001,week-000-2,,,Post-Op,CRET",
        "Patient-001,week-014,,,PD,new lesion",
        "Patient-001,week-014,,,PD,new lesion",       # duplicated rows exist in the real file
        "Patient-001,not-a-week,,,SD,",
    ]) + "\n")
    got = read_surgical_labels(str(csv_path))["Patient-001"]
    assert got == SurgicalLabels(frozenset({0.0}), frozenset({0.0}), frozenset({14.0}))
    assert set(SurgicalLabels.__slots__) == {"pre_op_weeks", "post_op_weeks", "other_scan_weeks"}


# ---------------------------------------------------------------------------- through the adapter
@pytest.fixture
def ratings_csv(tmp_path) -> str:
    path = tmp_path / "ratings.csv"
    path.write_text("\n".join([
        RATINGS_HEADER,
        "Patient-001,week-000,,,Post-Op,",
        "Patient-001,week-010,,,SD,",
        "Patient-003,week-005,,,Pre-Op,",           # then week-020 with no Post-Op: contradiction
        "Patient-003,week-020,,,SD,",
    ]) + "\n")
    return str(path)


def test_adapter_without_ratings_delivers_nothing(lumiere_fixture):
    assert LumiereAdapter(*lumiere_fixture).load_patient("Patient-001").treatment == TreatmentRecord()


def test_adapter_attaches_the_rebuilt_record(lumiere_fixture, ratings_csv):
    patient = LumiereAdapter(*lumiere_fixture, ratings_csv=ratings_csv).load_patient("Patient-001")
    assert surgery_weeks(patient.treatment) == [0]
    assert patient.treatment.radiotherapy[0].end.week == 10


def test_adapter_audit_reports_contradictions_and_nothing_else(lumiere_fixture, ratings_csv):
    audit = LumiereAdapter(*lumiere_fixture, ratings_csv=ratings_csv).audit_treatment()
    assert [row["patient"] for row in audit] == ["Patient-003"]
