"""Tests for fingerprint/io.py — one JSON file per patient."""

from __future__ import annotations

from rano.fingerprint.io import write_patient_fingerprint
from rano.fingerprint.schema import PatientFingerprint


def test_write_patient_fingerprint_round_trips(tmp_path):
    record = PatientFingerprint(patient_id="Patient-001", n_timepoints=0, timepoints=())
    path = write_patient_fingerprint(record, tmp_path / "fingerprints")

    assert path == tmp_path / "fingerprints" / "Patient-001.json"
    assert path.exists()
    reloaded = PatientFingerprint.model_validate_json(path.read_text())
    assert reloaded == record


def test_write_patient_fingerprint_creates_output_dir(tmp_path):
    record = PatientFingerprint(patient_id="Patient-002", n_timepoints=0, timepoints=())
    out_dir = tmp_path / "does" / "not" / "exist" / "yet"
    write_patient_fingerprint(record, out_dir)
    assert (out_dir / "Patient-002.json").exists()
