"""
Integration tests for LumiereAdapter against the synthetic ``lumiere_fixture`` cohort.

Coverage mirrors what we verified by hand against the real 32GB archive (91/638 reconciliation,
Patient-025-style manifest/zip divergence, ordering guarantees) but on a fixture small enough to
run in CI in milliseconds.
"""

from __future__ import annotations

import pytest

from rano.adapters.lumiere.adapter import LumiereAdapter
from rano.contract.case import MaskSource, Modality, Space


@pytest.fixture
def adapter(lumiere_fixture) -> LumiereAdapter:
    zip_path, manifest_path = lumiere_fixture
    return LumiereAdapter(zip_path, manifest_path)


def test_patient_ids_follow_manifest_order(adapter):
    assert adapter.patient_ids() == ["Patient-001", "Patient-002", "Patient-003", "Patient-999"]


def test_load_unknown_patient_raises_keyerror(adapter):
    with pytest.raises(KeyError):
        adapter.load_patient("Patient-does-not-exist")


def test_full_timepoint_attaches_all_modalities_and_mask(adapter):
    p1 = adapter.load_patient("Patient-001")
    tp0 = p1.timepoints[0]
    assert tp0.label == "week-000"
    assert tp0.available_modalities == frozenset(
        {Modality.CT1, Modality.T1, Modality.T2, Modality.FLAIR}
    )
    assert tp0.has_mask
    assert tp0.mask_source == MaskSource.DEEPBRATUMIA
    assert tp0.mask.space == Space.mni152_1mm()
    assert tp0.mask_reference is not None
    assert tp0.mask_reference.space == tp0.mask.space, "reference must share the mask's own space"
    assert tp0.brain_mask is not None


def test_partial_timepoint_omits_missing_modality_and_mask(adapter):
    p1 = adapter.load_patient("Patient-001")
    tp1 = p1.timepoints[1]
    assert tp1.label == "week-010"
    assert tp1.available_modalities == frozenset({Modality.CT1, Modality.T1, Modality.FLAIR})
    assert not tp1.has_modality(Modality.T2)
    assert not tp1.has_mask
    assert tp1.mask_reference is None


def test_single_timepoint_patient_is_not_longitudinal(adapter):
    p2 = adapter.load_patient("Patient-002")
    assert p2.n_timepoints == 1
    assert not p2.is_longitudinal
    assert p2.timepoints[0].available_modalities == frozenset({Modality.CT1})


def test_manifest_row_order_does_not_dictate_chronological_order(adapter):
    """Patient-003's manifest lists week-020 before week-005; the adapter must re-sort."""
    p3 = adapter.load_patient("Patient-003")
    assert [tp.label for tp in p3] == ["week-005", "week-020"]


def test_fully_missing_timepoint_yields_empty_timepoint_not_a_crash(adapter):
    """Mirrors the real Patient-025 case: manifest claims data the zip does not have."""
    p999 = adapter.load_patient("Patient-999")
    tp = p999.timepoints[0]
    assert tp.available_modalities == frozenset()
    assert not tp.has_mask


def test_audit_flags_the_fully_missing_timepoint_in_both_directions(adapter):
    disc = adapter.audit_manifest_vs_zip()
    p999_items = {d["item"] for d in disc if d["patient"] == "Patient-999"}
    assert p999_items == {"CT1", "T1", "T2", "FLAIR", "DeepBraTumIA"}
    assert all(d["manifest"] is True and d["zip"] is False for d in disc if d["patient"] == "Patient-999")


def test_audit_is_clean_for_patients_whose_manifest_matches_the_zip(adapter):
    disc = adapter.audit_manifest_vs_zip()
    assert not any(d["patient"] in {"Patient-001", "Patient-002", "Patient-003"} for d in disc)


def test_only_deepbratumia_mask_source_is_supported_for_now(lumiere_fixture):
    zip_path, manifest_path = lumiere_fixture
    with pytest.raises(NotImplementedError):
        LumiereAdapter(zip_path, manifest_path, mask_source=MaskSource.HDGLIO_AUTO)


def test_iter_patients_yields_every_patient_lazily(adapter):
    patients = list(adapter.iter_patients())
    assert {p.id for p in patients} == set(adapter.patient_ids())
