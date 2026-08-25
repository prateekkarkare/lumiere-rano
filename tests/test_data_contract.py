"""
Unit tests for ``rano.contract.data_contract`` — the emitted-artifact edge.

The rulings from the volumetry audit are encoded here as constants, so these tests are what stops
them drifting silently: a volume must never be emitted without its uncertainty, and the
uncertainty must follow lesion SIZE (the audit's actual finding) rather than compartment identity.
"""

from __future__ import annotations

import pytest

from rano.contract.data_contract import (
    VOLUME_UNCERTAINTY_PP,
    VOLUMETRY_SPACE,
    DataContract,
    TimepointContract,
    VolumeEntry,
    uncertainty_pp,
)


def test_uncertainty_bands_are_contiguous_and_cover_everything():
    """No gap and no overlap — every possible volume must land in exactly one band."""
    assert VOLUME_UNCERTAINTY_PP[0][0] == 0.0
    assert VOLUME_UNCERTAINTY_PP[-1][1] == float("inf")
    for (_, hi_prev, _), (lo_next, _, _) in zip(VOLUME_UNCERTAINTY_PP, VOLUME_UNCERTAINTY_PP[1:]):
        assert hi_prev == lo_next


def test_uncertainty_decreases_with_size():
    """The audit's core finding: bigger compartments measure more reproducibly."""
    spreads = [uncertainty_pp((lo + min(hi, 1e6)) / 2) for lo, hi, _ in VOLUME_UNCERTAINTY_PP]
    assert spreads == sorted(spreads, reverse=True)


@pytest.mark.parametrize("volume,expected", [(500.0, 36.6), (2_000.0, 14.8), (10_000.0, 5.3),
                                             (30_000.0, 2.8), (100_000.0, 1.5)])
def test_uncertainty_lookup(volume, expected):
    assert uncertainty_pp(volume) == pytest.approx(expected)


def test_uncertainty_at_exact_band_edges():
    """Edges are half-open [lo, hi) — 1000 belongs to the 1–5 cm3 band, not the one below."""
    assert uncertainty_pp(1_000.0) == pytest.approx(14.8)
    assert uncertainty_pp(999.99) == pytest.approx(36.6)


def test_zero_volume_gets_the_widest_band():
    """An absent compartment is 0 mm3 — a legitimate value that must still resolve a band."""
    assert uncertainty_pp(0.0) == pytest.approx(36.6)


def test_volume_entry_carries_space_by_default():
    """A volume is never a bare number: it names the space it was measured in."""
    assert VolumeEntry(volume_mm3=1.0, uncertainty_pp=5.0).space == VOLUMETRY_SPACE
    assert "mni" in VOLUMETRY_SPACE.lower()


def test_assessable_requires_both_usable_and_a_response_label():
    def tp(readiness, rating):
        return TimepointContract(id="p/w", label="w", week_offset=0.0, modalities={},
                                 readiness=readiness, expert_rating=rating)

    assert tp("usable", "PD").is_assessable
    assert tp("needs_attention", "CR").is_assessable
    assert not tp("unusable", "PD").is_assessable      # no usable segmentation
    assert not tp("usable", "Post-Op").is_assessable   # surgical state, not a response
    assert not tp("usable", None).is_assessable        # unrated


def test_provenance_records_both_rulings():
    p = DataContract.provenance_now("a.zip", "m.csv", "lumiere-zip", "DeepBraTumIA")
    assert "rigid" in p.volumetry_ruling.lower()
    assert "label 1 = contrast-enhancing" in p.label_schema_note
    assert "enhancing" in p.hdglio_role.lower() and "only" in p.hdglio_role.lower()
