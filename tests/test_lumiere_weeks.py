"""Tests for LUMIERE timepoint label parsing and chronological sort order."""

from __future__ import annotations

from rano.adapters.lumiere import weeks


def test_parse_plain_week():
    k = weeks.parse("week-044")
    assert (k.week, k.sub) == (44, 0)


def test_parse_week_with_sub_index():
    k = weeks.parse("week-000-2")
    assert (k.week, k.sub) == (0, 2)


def test_parse_rejects_garbage():
    assert weeks.parse("not-a-week") is None
    assert weeks.parse("") is None


def test_week_offset_matches_week_number_or_none():
    assert weeks.week_offset("week-044") == 44.0
    assert weeks.week_offset("week-000-2") == 0.0
    assert weeks.week_offset("garbage") is None


def test_sort_key_orders_week_then_sub_index():
    labels = ["week-044", "week-000-2", "week-000-1", "week-003"]
    ordered = sorted(labels, key=weeks.sort_key)
    assert ordered == ["week-000-1", "week-000-2", "week-003", "week-044"]


def test_sort_key_places_unparseable_labels_last():
    labels = ["week-010", "mystery-label", "week-000"]
    ordered = sorted(labels, key=weeks.sort_key)
    assert ordered == ["week-000", "week-010", "mystery-label"]
