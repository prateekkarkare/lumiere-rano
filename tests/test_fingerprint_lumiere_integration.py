"""
End-to-end: LumiereAdapter -> fingerprint_patient, against the synthetic lumiere_fixture cohort.

Verifies the extractor against real adapter output (per the project's own rule that some build
steps must check against the data, not just assert against hand-built fakes) — including the
deliberately degenerate case where the fixture's brain_mask is all-zero, so mask_reference's
brain-extent basis is empty and must come back as NaN stats rather than crash.
"""

from __future__ import annotations

import math

import pytest

from rano.adapters.lumiere.adapter import LumiereAdapter
from rano.fingerprint.extractor import fingerprint_patient


@pytest.fixture
def adapter(lumiere_fixture) -> LumiereAdapter:
    zip_path, manifest_path = lumiere_fixture
    return LumiereAdapter(zip_path, manifest_path)


def test_patient_001_full_timepoint_mask_histogram_matches_fixture(adapter):
    record = fingerprint_patient(adapter.load_patient("Patient-001"))
    assert record.n_timepoints == 2

    tp0 = record.timepoints[0]
    assert tp0.label == "week-000"
    counts = {lc.label: lc.n_voxels for lc in tp0.mask.label_histogram}
    assert counts == {0: 637, 1: 1, 2: 1, 3: 1}, "matches conftest._mask_array's 3 labeled voxels on an 8x10x8 grid"


def test_patient_001_mask_reference_empty_brain_basis_is_nan_not_a_crash(adapter):
    """The fixture's brain_mask is all-zero, so brain_mask==1 selects nothing."""
    record = fingerprint_patient(adapter.load_patient("Patient-001"))
    stats = record.timepoints[0].mask_reference.intensity
    assert stats.n_voxels == 0
    assert math.isnan(stats.mean)


def test_patient_001_mask_reference_compartment_intensity_covers_every_mask_label(adapter):
    record = fingerprint_patient(adapter.load_patient("Patient-001"))
    by_label = {ci.label: ci.stats for ci in record.timepoints[0].mask_reference.compartment_intensity}
    assert set(by_label) == {0, 1, 2, 3}
    assert by_label[1].n_voxels == 1
    assert by_label[0].n_voxels == 637


def test_patient_001_partial_timepoint_has_no_mask_slots(adapter):
    record = fingerprint_patient(adapter.load_patient("Patient-001"))
    tp1 = record.timepoints[1]
    assert tp1.label == "week-010"
    assert set(tp1.modalities) == {"CT1", "T1", "FLAIR"}
    assert tp1.mask is None
    assert tp1.mask_reference is None
    assert tp1.brain_mask is None


def test_patient_002_single_sparse_timepoint(adapter):
    record = fingerprint_patient(adapter.load_patient("Patient-002"))
    assert record.n_timepoints == 1
    assert set(record.timepoints[0].modalities) == {"CT1"}


def test_patient_999_fully_missing_timepoint_does_not_crash(adapter):
    """Mirrors the real Patient-025 case: manifest claims data the zip doesn't have."""
    record = fingerprint_patient(adapter.load_patient("Patient-999"))
    tp = record.timepoints[0]
    assert tp.modalities == {}
    assert tp.mask is None
