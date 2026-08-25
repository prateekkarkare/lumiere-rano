"""
The RANO rule itself. Pure functions, no dataset knowledge, no I/O.

WHAT IS AND IS NOT ENCODED
--------------------------
Encoded: the four imaging-side components and the reference logic --

  1. enhancing volume vs NADIR (progression) and vs BASELINE (response)
  2. non-enhancing T2/FLAIR volume vs its own nadir (progression)
  3. new lesion (progression)
  4. reference selection, nadir tracking, CR/PR confirmation, the post-radiotherapy
     pseudoprogression window, the non-measurable-disease block on PR, and the steroid block on CR

Not encoded, because no imaging pipeline can supply it: the clinical arm of RANO is accepted as an
INPUT (``clinical_deterioration``) but never inferred. Same for steroid dose. If you do not have
them, pass ``None`` and read ``assessment.unknowns``.

TWO REFERENCE POINTS, AND WHY THEY DIFFER
-----------------------------------------
Progression is measured against the NADIR -- the smallest enhancing volume seen so far -- because
a tumour that shrinks to a quarter of baseline and then doubles has progressed, even though it is
still below baseline. Response is measured against the BASELINE, because that is what "responded"
means. Using one reference for both is the single most common way to get this wrong; they are
tracked separately throughout.

The nadir NEVER includes the timepoint being assessed. Including it makes the ratio non-negative
by construction and PD unreachable -- a bug that produces plausible output forever.

PRECEDENCE
----------
Progression is checked before response. A lesion that is 70% below baseline but 50% above its
nadir is progressing; calling it PR because the baseline comparison also fires would be wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from rano.criteria.measurement import (
    Response,
    ResponseAssessment,
    TimepointMeasurement,
    TrajectoryResult,
)
from rano.criteria.profiles import DEFAULT_PROFILE, ResponseCriteria


@dataclass(frozen=True, slots=True)
class ReferenceState:
    """What the rule remembers about a patient's history at the moment of one assessment."""

    baseline_mm3: float | None = None
    nadir_mm3: float | None = None
    t2_nadir_mm3: float | None = None

    def advanced_by(self, m: TimepointMeasurement) -> "ReferenceState":
        """The state after folding ``m`` in. Called AFTER ``m`` has been assessed."""
        nadir = self.nadir_mm3
        if m.enhancing_mm3 is not None:
            nadir = m.enhancing_mm3 if nadir is None else min(nadir, m.enhancing_mm3)
        t2 = self.t2_nadir_mm3
        if m.t2_flair_mm3 is not None:
            t2 = m.t2_flair_mm3 if t2 is None else min(t2, m.t2_flair_mm3)
        return ReferenceState(self.baseline_mm3, nadir, t2)


def _ratio(value: float | None, reference: float | None) -> float | None:
    """Fractional change vs ``reference``. ``None`` when undefined -- including a zero reference,
    where the ratio is infinite and the caller must handle the case explicitly instead."""
    if value is None or reference is None or reference <= 0:
        return None
    return (value - reference) / reference


def _unknowns(
    call: Response, m: TimepointMeasurement, criteria: ResponseCriteria, ref: ReferenceState
) -> tuple[str, ...]:
    """Components whose unknown value could have changed THIS call.

    Not "every component we lack" -- that would mark almost every real-world assessment
    incomplete and the flag would stop meaning anything. A missing new-lesion flag cannot change
    a call that is already PD, so it is not reported there; it very much can change an SD, so it
    is reported there. ``complete`` therefore reads: *no missing input could have altered this*.
    """
    out: list[str] = []
    if call is not Response.PD:
        # Anything that could only ever escalate to PD matters exactly when we did not call PD.
        if criteria.use_new_lesion and m.new_lesion is None:
            out.append("new_lesion")
        if criteria.use_clinical and m.clinical_deterioration is None:
            out.append("clinical")
        if criteria.use_t2_progression and (m.t2_flair_mm3 is None or ref.t2_nadir_mm3 is None):
            out.append("t2_flair")
    if call is Response.CR and criteria.block_cr_when_steroids_increased and m.steroids_increased is None:
        out.append("steroids")
    if call is Response.PR and criteria.block_pr_when_non_measurable and m.non_measurable_only is None:
        out.append("non_measurable")
    return tuple(out)


def assess_timepoint(
    m: TimepointMeasurement,
    ref: ReferenceState,
    criteria: ResponseCriteria = DEFAULT_PROFILE,
) -> ResponseAssessment:
    """Apply the criteria to one scan against a reference state. No confirmation logic here --
    confirmation needs the future, so it lives in :func:`assess_trajectory`."""
    vs_nadir = _ratio(m.enhancing_mm3, ref.nadir_mm3)
    vs_baseline = _ratio(m.enhancing_mm3, ref.baseline_mm3)
    t2_vs_nadir = _ratio(m.t2_flair_mm3, ref.t2_nadir_mm3)

    def build(call: Response, reason: str, driver: str, **extra) -> ResponseAssessment:
        return ResponseAssessment(
            timepoint=m.timepoint,
            call=call,
            reason=reason,
            driver=driver,
            enhancing_mm3=m.enhancing_mm3,
            change_vs_nadir=vs_nadir,
            change_vs_baseline=vs_baseline,
            t2_change_vs_nadir=t2_vs_nadir,
            unknowns=_unknowns(call, m, criteria, ref),
            **extra,
        )

    if m.enhancing_mm3 is None:
        return build(Response.NE, "no enhancing volume measured", "missing")

    # The guard must be measured against the SAME reference as the ratio it is guarding, or it
    # silently gates one comparison on the size of a different one. Progression is judged against
    # the nadir, response against the baseline, so there are two guards, not one.
    def _big_enough(reference: float | None) -> bool:
        return abs(m.enhancing_mm3 - (reference or 0.0)) >= criteria.min_absolute_change_mm3

    pd_big_enough = _big_enough(ref.nadir_mm3)
    pr_big_enough = _big_enough(ref.baseline_mm3)

    # --- progression, in RANO's own precedence order ------------------------------------
    if criteria.use_new_lesion and m.new_lesion is True:
        pd = build(Response.PD, "new lesion", "new_lesion")
    elif criteria.use_clinical and m.clinical_deterioration is True:
        pd = build(Response.PD, "definite clinical deterioration", "clinical")
    elif vs_nadir is not None and vs_nadir >= criteria.pd_increase and pd_big_enough:
        pd = build(
            Response.PD,
            f"enhancing volume {vs_nadir:+.0%} vs nadir (>= {criteria.pd_increase:+.0%})",
            "enhancing",
        )
    elif ref.nadir_mm3 == 0 and m.enhancing_mm3 > 0 and pd_big_enough:
        pd = build(Response.PD, "enhancing disease re-emerged after a complete response", "enhancing")
    elif (
        criteria.use_t2_progression
        and t2_vs_nadir is not None
        and t2_vs_nadir >= criteria.t2_pd_increase
    ):
        pd = build(
            Response.PD,
            f"T2/FLAIR volume {t2_vs_nadir:+.0%} vs nadir (>= {criteria.t2_pd_increase:+.0%})",
            "t2_flair",
        )
    else:
        pd = None

    if pd is not None:
        in_window = (
            criteria.pseudoprogression_weeks is not None
            and m.weeks_since_rt is not None
            and m.weeks_since_rt < criteria.pseudoprogression_weeks
        )
        if in_window and criteria.pseudoprogression_policy == "downgrade":
            return build(
                Response.SD,
                f"{pd.reason}, but within {criteria.pseudoprogression_weeks:g} weeks of "
                "radiotherapy -- not callable as PD without histology or an out-of-field lesion",
                "confirmation",
                provisional_call=Response.PD,
                pseudoprogression_risk=True,
            )
        return replace(pd, pseudoprogression_risk=in_window)

    # --- response -----------------------------------------------------------------------
    if m.enhancing_mm3 == 0:
        if criteria.block_cr_when_steroids_increased and m.steroids_increased is True:
            return build(Response.PR, "no enhancing disease, but steroids increased", "enhancing")
        return build(Response.CR, "no enhancing disease", "enhancing")

    if vs_baseline is not None and vs_baseline <= criteria.pr_decrease and pr_big_enough:
        if criteria.block_pr_when_non_measurable and m.non_measurable_only is True:
            return build(
                Response.SD,
                "response threshold met but disease is non-measurable; PR is not available",
                "reference",
                provisional_call=Response.PR,
            )
        return build(
            Response.PR,
            f"enhancing volume {vs_baseline:+.0%} vs baseline (<= {criteria.pr_decrease:+.0%})",
            "enhancing",
        )

    detail = f"{vs_nadir:+.0%} vs nadir" if vs_nadir is not None else "no usable nadir ratio"
    return build(Response.SD, f"neither response nor progression ({detail})", "enhancing")


def _confirm(
    assessments: list[ResponseAssessment],
    measurements: list[TimepointMeasurement],
    criteria: ResponseCriteria,
) -> list[ResponseAssessment]:
    """Second pass: RANO requires CR and PR to hold on a repeat scan at least
    ``confirmation_weeks`` later. An unconfirmed response is SD.

    The confirming scan is the FIRST one far enough out, not the best one -- picking the most
    favourable later scan would let a response be confirmed by a rebound.
    """
    if not criteria.require_confirmation:
        return assessments

    out = list(assessments)
    for i, a in enumerate(assessments):
        if a.call not in (Response.CR, Response.PR):
            continue
        week = measurements[i].week
        if week is None:
            out[i] = replace(a, unknowns=tuple(sorted({*a.unknowns, "confirmation_week"})))
            continue

        j = next(
            (
                k
                for k in range(i + 1, len(measurements))
                if measurements[k].week is not None
                and measurements[k].week - week >= criteria.confirmation_weeks
            ),
            None,
        )
        if j is None:
            if criteria.confirm_at_end_of_followup:
                out[i] = replace(a, unknowns=tuple(sorted({*a.unknowns, "confirmation_followup"})))
            else:
                out[i] = replace(
                    a,
                    call=Response.SD,
                    provisional_call=a.call,
                    reason=f"{a.call} unconfirmed -- no scan >= {criteria.confirmation_weeks:g} weeks later",
                    driver="confirmation",
                )
            continue

        allowed = (Response.CR,) if a.call is Response.CR else (Response.CR, Response.PR)
        if assessments[j].call not in allowed:
            out[i] = replace(
                a,
                call=Response.SD,
                provisional_call=a.call,
                reason=(
                    f"{a.call} not confirmed at {measurements[j].timepoint} "
                    f"(+{measurements[j].week - week:g} wk, scored {assessments[j].call})"
                ),
                driver="confirmation",
            )
    return out


def assess_trajectory(
    patient: str,
    measurements: list[TimepointMeasurement],
    criteria: ResponseCriteria = DEFAULT_PROFILE,
    *,
    reference: TimepointMeasurement | None = None,
) -> TrajectoryResult:
    """Score one patient's whole trajectory, in the order given.

    ``measurements`` MUST already be chronological and MUST already exclude scans that are not
    response assessments (pre-operative studies above all -- an untreated tumour used as the
    baseline makes the post-operative drop look like a partial response for the rest of the
    patient's life).

    ``reference`` is the dedicated reference scan -- for post-surgical glioma follow-up, the
    immediate post-operative study. Supply it and every entry in ``measurements`` gets scored.
    Omit it and ``measurements[0]`` is consumed as the reference, emitted as ``BASELINE`` and
    left unscored; ``TrajectoryResult.baseline_scored`` records which happened.
    """
    seed = reference if reference is not None else (measurements[0] if measurements else None)
    if seed is None:
        return TrajectoryResult(patient, (), None, None, True)

    ref = ReferenceState(
        baseline_mm3=seed.enhancing_mm3,
        nadir_mm3=seed.enhancing_mm3,
        t2_nadir_mm3=seed.t2_flair_mm3,
    )

    scored = list(measurements) if reference is not None else list(measurements[1:])
    prefix: list[ResponseAssessment] = []
    if reference is None and measurements:
        prefix.append(
            ResponseAssessment(
                timepoint=measurements[0].timepoint,
                call=Response.BASELINE,
                reason="reference scan for this trajectory",
                driver="reference",
                enhancing_mm3=measurements[0].enhancing_mm3,
            )
        )

    assessments: list[ResponseAssessment] = []
    for m in scored:
        assessments.append(assess_timepoint(m, ref, criteria))
        ref = ref.advanced_by(m)

    assessments = _confirm(assessments, scored, criteria)

    return TrajectoryResult(
        patient=patient,
        assessments=tuple(prefix + assessments),
        baseline_timepoint=seed.timepoint,
        baseline_mm3=seed.enhancing_mm3,
        baseline_scored=reference is not None,
    )
