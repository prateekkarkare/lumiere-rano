"""
Comparing computed calls against reference (expert) calls, and rendering the comparison.

Agreement percentage on its own is close to useless here, for a reason specific to this problem:
the class balance is brutal. LUMIERE's expert ratings are ~55% PD, so a rule that says "PD"
unconditionally scores in the fifties and looks respectable. Every report this module produces
therefore leads with the confusion matrix and per-class recall, and the headline number is
reported next to the majority-class baseline it has to beat.

Two views, because they answer different questions:

  * the CONFUSION MATRIX answers "what kind of mistake does this rule make" -- is it late on
    progression, or does it invent it;
  * the TIMELINE answers "where in a patient's course does it break" -- systematically at the
    start, at the nadir, or only after a response. A confusion matrix cannot show that a rule is
    right about every timepoint except that it calls progression one scan too late, which is both
    the most likely failure of a threshold rule and the most clinically important.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from rano.criteria.measurement import Response

#: Reporting order. Response first, progression last -- reads as a severity axis.
CLASS_ORDER: tuple[str, ...] = ("CR", "PR", "SD", "PD")

AGREE, DISAGREE = "✓", "✗"  # check, ballot-x


@dataclass(frozen=True, slots=True)
class CallPair:
    """One computed call set beside the reference call for the same scan."""

    patient: str
    timepoint: str
    predicted: str
    expert: str
    week: float | None = None
    group: str = ""
    """Free-form stratum (cohort arm, site, scanner...). Reports can be split on it."""
    reason: str = ""
    """Why the rule made this call, in one line."""
    expert_reason: str = ""
    """Why the reference reader made theirs, verbatim. Empty when the source records none."""
    detail: str = ""
    """Compact numeric context for the row (volume and the ratios it was judged on)."""
    unknowns: tuple[str, ...] = ()
    """RANO components that could have changed this call but were unavailable."""
    complete: bool = True
    """False when the rule had to skip a component for want of data -- see ``unknowns``."""

    @property
    def agree(self) -> bool:
        return self.predicted == self.expert


@dataclass(frozen=True, slots=True)
class ClassStats:
    support: int
    """How many scans the expert put in this class."""
    n_correct: int
    predicted_as: int
    """How many scans the rule put in this class."""

    @property
    def recall(self) -> float:
        return self.n_correct / self.support if self.support else 0.0

    @property
    def precision(self) -> float:
        return self.n_correct / self.predicted_as if self.predicted_as else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


@dataclass(frozen=True, slots=True)
class AgreementReport:
    label: str
    pairs: tuple[CallPair, ...]
    confusion: dict[str, dict[str, int]]
    per_class: dict[str, ClassStats]

    @property
    def n_compared(self) -> int:
        return len(self.pairs)

    @property
    def n_agree(self) -> int:
        return sum(1 for p in self.pairs if p.agree)

    @property
    def agreement(self) -> float:
        return self.n_agree / self.n_compared if self.n_compared else 0.0

    @property
    def majority_baseline(self) -> float:
        """What "always guess the commonest expert class" would score. The bar to clear."""
        if not self.pairs:
            return 0.0
        counts = defaultdict(int)
        for p in self.pairs:
            counts[p.expert] += 1
        return max(counts.values()) / len(self.pairs)

    @property
    def balanced_accuracy(self) -> float:
        """Mean per-class recall -- the class-imbalance-proof headline."""
        present = [s for s in self.per_class.values() if s.support]
        return sum(s.recall for s in present) / len(present) if present else 0.0

    @property
    def n_incomplete(self) -> int:
        return sum(1 for p in self.pairs if not p.complete)


def compare_calls(pairs: list[CallPair], label: str = "") -> AgreementReport:
    """Build the report. Classes with zero support are still emitted, so two reports over
    different subsets always have the same shape and can be printed side by side."""
    classes = list(CLASS_ORDER)
    for p in pairs:  # tolerate anything unexpected rather than dropping it silently
        for c in (p.expert, p.predicted):
            if c not in classes:
                classes.append(c)

    confusion = {e: {p: 0 for p in classes} for e in classes}
    for p in pairs:
        confusion[p.expert][p.predicted] += 1

    per_class = {}
    for c in classes:
        support = sum(confusion[c].values())
        predicted_as = sum(confusion[e][c] for e in classes)
        per_class[c] = ClassStats(support, confusion[c][c], predicted_as)

    return AgreementReport(label=label, pairs=tuple(pairs), confusion=confusion, per_class=per_class)


def split_by_group(pairs: list[CallPair]) -> dict[str, list[CallPair]]:
    out: dict[str, list[CallPair]] = defaultdict(list)
    for p in pairs:
        out[p.group].append(p)
    return dict(out)


# --------------------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------------------

def format_confusion(report: AgreementReport) -> str:
    """Expert (rows) x computed (columns), with per-class recall and precision."""
    classes = list(report.confusion)
    w = max(6, max((len(c) for c in classes), default=6) + 1)
    head = f"{'expert\\calc':>13} " + "".join(f"{c:>{w}}" for c in classes) + f"{'n':>7}{'recall':>9}"
    lines = [head, "-" * len(head)]

    for e in classes:
        stats = report.per_class[e]
        if not stats.support and not stats.predicted_as:
            continue
        cells = "".join(
            f"{report.confusion[e][p] or '.':>{w}}" if e != p else f"{report.confusion[e][p]:>{w}}"
            for p in classes
        )
        lines.append(f"{e:>13} {cells}{stats.support:>7}{stats.recall:>8.0%} ")

    prec = "".join(f"{report.per_class[c].precision:>{w}.0%}" for c in classes)
    lines.append("-" * len(head))
    lines.append(f"{'precision':>13} {prec}")
    lines.append("")
    lines.append(
        f"  agreement {report.n_agree}/{report.n_compared} = {report.agreement:.1%}"
        f"   (majority-class baseline {report.majority_baseline:.1%})"
    )
    lines.append(f"  balanced accuracy (mean per-class recall) {report.balanced_accuracy:.1%}")
    if report.n_incomplete:
        lines.append(
            f"  {report.n_incomplete}/{report.n_compared} calls made with at least one RANO "
            "component unavailable"
        )
    return "\n".join(lines)


def format_timeline(report: AgreementReport, per_row: int = 12, max_patients: int | None = None) -> str:
    """Per-patient trajectories: expert over computed, with agreement marks underneath.

    Long trajectories wrap into blocks of ``per_row`` scans so nothing depends on terminal width.
    Patients are ordered worst-agreement first -- the interesting ones should not be at the bottom.
    """
    by_patient: dict[str, list[CallPair]] = defaultdict(list)
    for p in report.pairs:
        by_patient[p.patient].append(p)
    for v in by_patient.values():
        v.sort(key=lambda p: (p.week if p.week is not None else 1e9, p.timepoint))

    # Worst agreement first, then LONGEST trajectory first. Without the second key the head of
    # the list fills with patients who have a single scan and therefore score 0% or 100% by
    # accident -- the least informative rows crowding out the most.
    ordered = sorted(
        by_patient.items(),
        key=lambda kv: (sum(1 for p in kv[1] if p.agree) / len(kv[1]), -len(kv[1]), kv[0]),
    )
    if max_patients is not None:
        ordered = ordered[:max_patients]

    out: list[str] = []
    for patient, ps in ordered:
        n_ok = sum(1 for p in ps if p.agree)
        group = f"  [{ps[0].group}]" if ps[0].group else ""
        out.append(f"{patient}{group}   {n_ok}/{len(ps)} = {n_ok / len(ps):.0%}")
        for start in range(0, len(ps), per_row):
            block = ps[start : start + per_row]
            labels = [p.timepoint.replace("week-", "w") for p in block]
            w = max(6, max(len(x) for x in labels) + 1)
            out.append("    scan  " + "".join(f"{x:>{w}}" for x in labels))
            out.append("    expert" + "".join(f"{p.expert:>{w}}" for p in block))
            out.append("    calc  " + "".join(f"{p.predicted:>{w}}" for p in block))
            out.append(
                "          " + "".join(f"{AGREE if p.agree else DISAGREE:>{w}}" for p in block)
            )
        out.append("")
    return "\n".join(out)


def format_case_table(
    report: AgreementReport,
    max_patients: int | None = None,
    disagreements_only: bool = False,
) -> str:
    """Per-patient case tables: both calls side by side, each with the reasoning behind it.

    The timeline view compresses a trajectory to two letters per scan, which is right for spotting
    *where* a rule breaks and useless for asking *why*. This view is the other half: one block per
    scan, the reference reader's stated rationale above our own, so a disagreement can be read as a
    disagreement about evidence rather than a mismatched pair of labels.

    Patients are ordered worst agreement first, longest trajectory first within that.
    """
    by_patient: dict[str, list[CallPair]] = defaultdict(list)
    for p in report.pairs:
        by_patient[p.patient].append(p)
    for v in by_patient.values():
        v.sort(key=lambda p: (p.week if p.week is not None else 1e9, p.timepoint))

    ordered = sorted(
        by_patient.items(),
        key=lambda kv: (sum(1 for p in kv[1] if p.agree) / len(kv[1]), -len(kv[1]), kv[0]),
    )
    if max_patients is not None:
        ordered = ordered[:max_patients]

    out: list[str] = []
    for patient, ps in ordered:
        shown = [p for p in ps if not p.agree] if disagreements_only else ps
        if not shown:
            continue
        n_ok = sum(1 for p in ps if p.agree)
        head = f"{patient}   {n_ok}/{len(ps)} = {n_ok / len(ps):.0%}"
        out.append(head)
        out.append("-" * max(len(head), 76))
        for p in shown:
            mark = AGREE if p.agree else DISAGREE
            out.append(f"  {p.timepoint:<12} {mark}   {p.detail}")
            out.append(f"  {'':<12} expert {p.expert:<3} {p.expert_reason or '(no rationale recorded)'}")
            out.append(f"  {'':<12} calc   {p.predicted:<3} {p.reason}")
            if p.unknowns:
                out.append(f"  {'':<12} {'':<10} not available to the rule: {', '.join(p.unknowns)}")
            out.append("")
        out.append("")
    return "\n".join(out)


def format_summary_line(report: AgreementReport) -> str:
    return (
        f"{report.label:<24} n={report.n_compared:<5} "
        f"agree {report.agreement:>6.1%}   balanced {report.balanced_accuracy:>6.1%}   "
        + "  ".join(
            f"{c} {report.per_class[c].recall:.0%}"
            for c in CLASS_ORDER
            if report.per_class.get(c) and report.per_class[c].support
        )
    )
