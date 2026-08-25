"""
End-to-end: LumiereAdapter -> fingerprint_timepoint -> check_mask_grid_alignment.

Exercises the real pipeline order (FINGERPRINTER -> VALIDATOR): the check never sees a live
Timepoint/ImageRef, only the TimepointFingerprint the fingerprinter already produced.
"""

from __future__ import annotations

import pytest

from rano.adapters.lumiere.adapter import LumiereAdapter
from rano.fingerprint.extractor import fingerprint_timepoint
from rano.validate.checks.mask_grid_alignment import check_mask_grid_alignment


@pytest.fixture
def adapter(lumiere_fixture) -> LumiereAdapter:
    zip_path, manifest_path = lumiere_fixture
    return LumiereAdapter(zip_path, manifest_path)


def test_patient_001_full_timepoint_passes(adapter):
    p1 = adapter.load_patient("Patient-001")
    fp = fingerprint_timepoint(p1.timepoints[0])
    result = check_mask_grid_alignment(fp)
    assert result.status == "pass"
    assert result.code == "mask_grid_alignment.ok"


def test_patient_001_partial_timepoint_has_no_mask_to_check(adapter):
    p1 = adapter.load_patient("Patient-001")
    fp = fingerprint_timepoint(p1.timepoints[1])  # week-010: no mask in this fixture
    assert check_mask_grid_alignment(fp).code == "mask_grid_alignment.no_pair"


def test_patient_002_sparse_timepoint_has_no_mask_to_check(adapter):
    p2 = adapter.load_patient("Patient-002")
    fp = fingerprint_timepoint(p2.timepoints[0])
    assert check_mask_grid_alignment(fp).code == "mask_grid_alignment.no_pair"


def test_patient_999_fully_missing_timepoint_does_not_crash(adapter):
    p999 = adapter.load_patient("Patient-999")
    fp = fingerprint_timepoint(p999.timepoints[0])
    assert check_mask_grid_alignment(fp).status == "pass"
