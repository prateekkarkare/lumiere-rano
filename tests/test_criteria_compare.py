"""
Unit tests for agreement reporting.

The metric that matters here is the one that survives class imbalance. LUMIERE's expert ratings
are ~63% PD, so raw agreement flatters any rule that leans on progression; several tests below
pin exactly that, because a reporting bug in this direction does not look like a bug -- it looks
like a good result.
"""

from __future__ import annotations

import pytest

from rano.criteria import (
    CallPair,
    compare_calls,
    format_case_table,
    format_confusion,
    format_timeline,
    split_by_group,
)


def pair(patient, tp, predicted, expert, **kw):
    return CallPair(patient=patient, timepoint=tp, predicted=predicted, expert=expert, **kw)


def test_perfect_agreement():
    r = compare_calls([pair("p", "t1", "PD", "PD"), pair("p", "t2", "SD", "SD")])
    assert r.agreement == 1.0
    assert r.balanced_accuracy == 1.0


def test_agreement_counts_only_exact_matches():
    r = compare_calls([pair("p", "t1", "PD", "SD"), pair("p", "t2", "SD", "SD")])
    assert r.n_agree == 1
    assert r.agreement == 0.5


def test_majority_baseline_exposes_a_rule_that_only_guesses_the_common_class():
    """Nine PD, one SD; a rule that always says PD scores 90% and must not look impressive."""
    pairs = [pair("p", f"t{i}", "PD", "PD") for i in range(9)] + [pair("p", "t9", "PD", "SD")]
    r = compare_calls(pairs)
    assert r.agreement == pytest.approx(0.9)
    assert r.majority_baseline == pytest.approx(0.9)
    assert r.balanced_accuracy == pytest.approx(0.5)  # PD recall 100%, SD recall 0%


def test_balanced_accuracy_ignores_classes_with_no_support():
    """A class the expert never used must not drag the mean down as a zero."""
    r = compare_calls([pair("p", "t1", "PD", "PD"), pair("p", "t2", "SD", "SD")])
    assert r.per_class["CR"].support == 0
    assert r.balanced_accuracy == 1.0


def test_recall_and_precision_are_not_the_same_number():
    # expert: 2 PD, 1 SD.  computed: 3 PD.  PD recall 100%, PD precision 67%.
    pairs = [pair("p", "t1", "PD", "PD"), pair("p", "t2", "PD", "PD"), pair("p", "t3", "PD", "SD")]
    r = compare_calls(pairs)
    assert r.per_class["PD"].recall == pytest.approx(1.0)
    assert r.per_class["PD"].precision == pytest.approx(2 / 3)
    assert r.per_class["SD"].recall == 0.0


def test_confusion_orientation_is_expert_rows_by_computed_columns():
    """Transposing this is silent and inverts every conclusion about the rule's bias."""
    r = compare_calls([pair("p", "t1", "PD", "SD")])
    assert r.confusion["SD"]["PD"] == 1
    assert r.confusion["PD"]["SD"] == 0


def test_unexpected_classes_are_kept_not_dropped():
    r = compare_calls([pair("p", "t1", "NE", "PD")])
    assert r.confusion["PD"]["NE"] == 1
    assert r.n_compared == 1


def test_incomplete_calls_are_counted():
    pairs = [pair("p", "t1", "SD", "SD", complete=False), pair("p", "t2", "SD", "SD")]
    assert compare_calls(pairs).n_incomplete == 1


def test_empty_report_does_not_divide_by_zero():
    r = compare_calls([])
    assert r.agreement == 0.0
    assert r.majority_baseline == 0.0
    assert r.balanced_accuracy == 0.0


def test_split_by_group():
    pairs = [pair("a", "t1", "PD", "PD", group="practice"), pair("b", "t1", "SD", "PD", group="held_out")]
    groups = split_by_group(pairs)
    assert set(groups) == {"practice", "held_out"}
    assert compare_calls(groups["practice"]).agreement == 1.0


def test_confusion_renders_the_headline_numbers():
    r = compare_calls([pair("p", "t1", "PD", "PD"), pair("p", "t2", "SD", "PD")], "demo")
    text = format_confusion(r)
    assert "agreement 1/2" in text
    assert "balanced accuracy" in text


def test_timeline_orders_worst_first_then_longest():
    """A one-scan patient at 0% must not outrank a ten-scan patient at 0% -- that fills the head
    of the report with the least informative rows."""
    pairs = [pair("short", "week-001", "PD", "SD")]
    pairs += [pair("long", f"week-{i:03d}", "PD", "SD", week=float(i)) for i in range(10)]
    pairs += [pair("good", "week-001", "PD", "PD")]
    text = format_timeline(compare_calls(pairs))
    assert text.index("long") < text.index("short") < text.index("good")


def test_timeline_wraps_long_trajectories():
    pairs = [pair("p", f"week-{i:03d}", "PD", "PD", week=float(i)) for i in range(20)]
    text = format_timeline(compare_calls(pairs), per_row=6)
    assert text.count("    scan  ") == 4  # 20 scans over blocks of 6


def test_timeline_sorts_scans_by_week_not_by_label():
    pairs = [
        pair("p", "week-100", "PD", "PD", week=100.0),
        pair("p", "week-009", "SD", "SD", week=9.0),
    ]
    text = format_timeline(compare_calls(pairs))
    assert text.index("w009") < text.index("w100")


# --------------------------------------------------------------------------------------
# case tables -- both sides' reasoning
# --------------------------------------------------------------------------------------

def _case(**kw):
    base = dict(
        patient="p", timepoint="week-010", predicted="SD", expert="PD", week=10.0,
        reason="neither response nor progression (-36% vs nadir)",
        expert_reason="Target L.: 13mm x 32mm", detail="3,430 mm3   -36% vs nadir",
    )
    return CallPair(**{**base, **kw})


def test_case_table_shows_both_reasonings():
    text = format_case_table(compare_calls([_case()]))
    assert "Target L.: 13mm x 32mm" in text
    assert "neither response nor progression" in text
    assert "expert PD" in text and "calc   SD" in text


def test_case_table_names_the_missing_components():
    """'incomplete' is not actionable; the component names are."""
    text = format_case_table(compare_calls([_case(unknowns=("new_lesion", "clinical"), complete=False)]))
    assert "new_lesion" in text and "clinical" in text


def test_case_table_marks_a_missing_expert_rationale_rather_than_leaving_a_blank():
    """A blank column reads as 'we lost it'; the expert genuinely recorded none for many scans."""
    text = format_case_table(compare_calls([_case(expert_reason="")]))
    assert "no rationale recorded" in text


def test_case_table_disagreements_only_drops_agreeing_rows_but_keeps_the_patient_total():
    pairs = [_case(timepoint="week-010", predicted="PD"), _case(timepoint="week-020", predicted="SD")]
    text = format_case_table(compare_calls(pairs), disagreements_only=True)
    assert "week-020" in text
    assert "week-010" not in text
    assert "1/2 = 50%" in text  # the header still reports the whole trajectory


def test_case_table_omits_a_patient_with_no_disagreements():
    pairs = [_case(patient="clean", predicted="PD")]
    assert format_case_table(compare_calls(pairs), disagreements_only=True).strip() == ""
