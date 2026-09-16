"""
Integration: the treatment rebuild on real LUMIERE patients whose quirks shaped the rules.

Every patient named here is OUTSIDE the locked cohort (unassigned), so reading their surgical
labels breaks no discipline. Skipped when the real files are absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rano.adapters.lumiere.adapter import LumiereAdapter
from rano.contract.treatment import TreatmentRecord

ROOT = Path(__file__).resolve().parent.parent
ZIP, MANIFEST = ROOT / "Imaging-v202211.zip", ROOT / "LUMIERE-datacompleteness.csv"
RATINGS = ROOT / "LUMIERE-ExpertRating-v202211.csv"

pytestmark = pytest.mark.skipif(
    not (ZIP.exists() and MANIFEST.exists() and RATINGS.exists()), reason="real LUMIERE files not present"
)


@pytest.fixture(scope="module")
def adapter() -> LumiereAdapter:
    return LumiereAdapter(str(ZIP), str(MANIFEST), ratings_csv=str(RATINGS))


@pytest.mark.parametrize(
    "patient, surgeries, rt_end",
    [
        ("Patient-010", [1.0], 11.0),        # Post-Op at weeks 1, 1, 2 -> one operation
        ("Patient-085", [0.0], 10.0),        # Post-Op at weeks 0 and 1 -> one operation
        ("Patient-004", [0.0, 41.0], 10.0),  # re-operation; radiotherapy follows the first only
    ],
)
def test_known_shapes(adapter, patient, surgeries, rt_end):
    record = adapter.load_patient(patient).treatment
    assert [s.week for s in record.surgeries] == surgeries
    assert record.radiotherapy[0].end.week == rt_end


@pytest.mark.parametrize("patient", ["Patient-079", "Patient-013"])
def test_contradictions_are_left_missing(adapter, patient):
    assert adapter.load_patient(patient).treatment == TreatmentRecord()
