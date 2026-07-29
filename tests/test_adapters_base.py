"""Tests for the Adapter ABC and the DICOM seam stub."""

from __future__ import annotations

import pytest

from rano.adapters.base import Adapter
from rano.adapters.dicom import DicomAdapter


def test_adapter_is_abstract():
    with pytest.raises(TypeError):
        Adapter()  # type: ignore[abstract]


def test_dicom_adapter_satisfies_the_abc():
    """The seam is provably satisfiable by something other than LUMIERE."""
    assert issubclass(DicomAdapter, Adapter)
    DicomAdapter("/some/root")  # construction must not raise


def test_dicom_adapter_methods_are_stubbed_not_silently_wrong():
    ad = DicomAdapter("/some/root")
    with pytest.raises(NotImplementedError):
        ad.patient_ids()
    with pytest.raises(NotImplementedError):
        ad.load_patient("anyone")
