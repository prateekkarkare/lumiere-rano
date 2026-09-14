"""
Threshold sets, as named profiles.

RANO is not one rule. The 2010/2017 criteria were written for BIDIMENSIONAL products of
perpendicular diameters (+25% -> PD, -50% -> PR); the volumetric adaptations re-derived the
cutoffs so that a sphere growing by the 2D criterion maps onto the volumetric one (+40% -> PD,
-65% -> PR). Applying the 2D numbers to a volume is a real, defensible choice -- it is closer to
what a human reader following RANO would have done -- but it is a DIFFERENT rule, and quoting a
single "RANO agreement" figure without saying which one is meaningless.

So the thresholds live in a frozen object, the object has names, and the evaluation runs all of
them. The threshold choice becomes a measured result rather than a buried constant.

HONEST GAPS -- these defaults are not all equally well founded:
  * ``pd_increase`` / ``pr_decrease``: published, cited below.
  * ``t2_pd_increase``: NOT standardized anywhere. RANO says "significant increase in T2/FLAIR"
    and leaves it to the reader. MEASURED ON LUMIERE (2026-08-22, all 91 patients, 376 rated
    timepoints) the component as implementable from DeepBraTumIA's edema compartment does not
    discriminate at ANY threshold or against ANY reference:

        edema vs nadir, +40%   fires on 84% of expert-CR and 76% of expert-PD  (PD-CR = -8 pts)
        edema vs baseline      best separation +15 pts, at +200%
        edema vs prior scan    best separation  +8 pts, at  +40%

    The reason is mechanical: the reference scan is the immediate post-operative study, when
    oedema is suppressed by surgery and steroids. It then rises in everyone, so "T2 vs nadir"
    measures weeks-since-surgery, not progression. It is therefore OFF in the shipped profiles
    and kept alive as ``MRANO_WITH_T2`` so the finding stays measured rather than remembered.
    Re-enable it when the T2 signal is non-enhancing TUMOUR rather than the whole oedema
    compartment, and re-measure before trusting it.
  * ``min_absolute_change_mm3``: a guard of our own, not RANO. A 0.4 cm3 lesion can swing tens of
    percent on slice-thickness alone, so a pure ratio makes noise look like progression. SWEPT ON
    LUMIERE (2026-09-02, 74 patients / 221 timepoints, held-out arm excluded; scripts/
    sweep_guards.py) over 0 - 1000 mm3:

        0 - 40 mm3     IDENTICAL output. Not one scored call in the cohort turns on a change
                       this small, so the "one voxel scored as progression" failure mode is real
                       in principle and has zero incidence here.
        45 - 150 mm3   agreement flat, balanced accuracy +4 pp -- but the entire gain is TWO
                       timepoints in ONE patient (059), and it is bought by turning a correct PD
                       into a wrong SD in another. One patient is not a calibration.
        >= 200 mm3     four correct PD calls destroyed. Clearly too high.

    Default 10.0 -- roughly 2-10 ACQUIRED voxels (native voxels on this cohort run 1-5 mm3; the
    1 mm atlas grid is resampled, not acquired). Chosen as the largest value that provably changes
    nothing, so the guard states the intent without buying agreement it cannot defend.

    ``min_absolute_change_mm3`` is NOT the measurability gate and cannot become one: finding F3 of
    the 2026-08-31 adjudication shows measurability is a SHAPE test (>=10 mm in two perpendicular
    axes) that no volume threshold can express -- a 30x30x3 mm pancake is 1,414 mm3 and not
    measurable, a 12x11x10 mm blob is 691 mm3 and is. This is a noise floor and nothing more.

References:
  Wen et al. (2010), J Clin Oncol 28:1963 -- RANO high-grade glioma criteria (bidimensional).
  Ellingson et al. (2017), Neuro-Oncology 19:89 -- volumetric thresholds for glioma response.
  Wen et al. (2023), J Clin Oncol 41:5187 -- RANO 2.0.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

PseudoprogressionPolicy = Literal["flag", "downgrade"]


@dataclass(frozen=True, slots=True)
class ResponseCriteria:
    """A complete, named parameterisation of the response rule."""

    name: str
    description: str

    # --- component 1: enhancing volume -------------------------------------------------
    pd_increase: float
    """Fractional increase vs NADIR that calls progression. +0.40 == a 40% increase."""
    pr_decrease: float
    """Fractional change vs BASELINE that calls partial response. Negative, e.g. -0.65."""

    # --- component 2: non-enhancing T2/FLAIR -------------------------------------------
    use_t2_progression: bool = True
    """When False the T2/FLAIR component is skipped entirely (not treated as negative -- it is
    recorded as an unknown), which is how you measure what the component is worth."""
    t2_pd_increase: float = 0.40
    """Fractional increase in T2/FLAIR volume vs its own nadir that calls progression."""

    # --- components 3 & 4: new lesions, clinical ---------------------------------------
    use_new_lesion: bool = True
    use_clinical: bool = True
    """Both are honoured when supplied and reported as unknown when not. Turning them off says
    'this rule does not consider them', which is different from 'nobody told me'."""

    # --- reference & confirmation ------------------------------------------------------
    require_confirmation: bool = True
    """RANO requires CR and PR to be confirmed by a repeat scan. Unconfirmed responses become SD."""
    confirmation_weeks: float = 4.0
    """Minimum interval to the confirming scan."""
    confirmation_max_weeks: float | None = 16.0
    """Maximum interval to the confirming scan. RANO writes only a MINIMUM ("at least 4 weeks"),
    which read literally makes the next scan a confirmation however far away it is. MEASURED ON
    LUMIERE (2026-08-31 adjudication, finding F2): of 45 confirmation-driven demotions the gap to
    the confirming scan ran 4, 4, 5, ... 20, 27, 42, 50 weeks. A scan 42 weeks later is not
    confirming a response, it is the next chapter of the disease -- and the demotion it caused was
    wrong in 30 of the 45.

    Default 16.0. Glioma follow-up on this cohort runs every 8-12 weeks, so 16 admits the next
    scheduled scan plus a slipped one and rejects a scan a whole extra interval away. SWEPT
    (2026-09-02, same 221 timepoints as above): 12 wk and below loses SD recall, 16-18 wk is a flat
    optimum, 19 wk and above decays back to the unlimited behaviour. The swept optimum and the
    scheduling argument agree, which is the only reason to trust either -- the measured margin over
    unlimited is +1.2 pp balanced accuracy off a net of ONE timepoint, which on its own would be
    noise. Take this fix for correctness, not for the number.

    When the first scan at least ``confirmation_weeks`` out falls beyond this limit, nothing in the
    record can confirm OR refute the response. The call is then KEPT and flagged
    ``confirmation_too_late``, never demoted -- absence of evidence is recorded as absence of
    evidence. ``None`` restores the unlimited pre-2026-09 behaviour."""
    confirm_at_end_of_followup: bool = True
    """What to do with a response at the last scan, where confirmation is impossible. True keeps
    the call and flags it; False downgrades it to SD. Keeping it avoids systematically penalising
    every trajectory's final timepoint, which is an artefact of follow-up length, not biology."""

    pseudoprogression_weeks: float | None = 12.0
    """Post-radiotherapy window in which a new or growing enhancing lesion is as likely to be
    treatment effect as tumour. ``None`` disables the concept."""
    pseudoprogression_policy: PseudoprogressionPolicy = "downgrade"
    """``flag`` records the risk and keeps the PD call. ``downgrade`` turns such a PD into SD,
    which is the strict RANO reading (PD in-window needs histology or an out-of-field lesion)."""

    # --- guards ------------------------------------------------------------------------
    min_absolute_change_mm3: float = 10.0
    """Changes smaller than this in absolute terms never trigger PD or PR, whatever the ratio."""
    block_pr_when_non_measurable: bool = True
    """RANO forbids PR on non-measurable disease; PD and SD stay available."""
    block_cr_when_steroids_increased: bool = True

    def variant(self, name: str, **changes) -> "ResponseCriteria":
        """A copy with fields overridden -- for ablations, without mutating a shared profile."""
        return replace(self, name=name, **changes)


MRANO_VOLUMETRIC = ResponseCriteria(
    name="mrano_volumetric",
    description=(
        "Volumetric RANO with Ellingson-style thresholds: PD at +40% vs nadir, PR at -65% vs "
        "baseline. The cutoffs are the volume-space equivalents of the bidimensional ones."
    ),
    pd_increase=0.40,
    pr_decrease=-0.65,
    use_t2_progression=False,  # measured non-discriminative on LUMIERE -- see module docstring
)

RANO_CLASSIC_PORTED = ResponseCriteria(
    name="rano_classic_ported",
    description=(
        "The bidimensional RANO cutoffs (+25% PD, -50% PR) applied directly to volume. Closer to "
        "what the human readers followed, but the numbers were never derived for volumes -- a "
        "sphere that grows 25% by 2D product grows ~40% by volume, so this rule is STRICTLY more "
        "trigger-happy on progression than the criteria it is named after."
    ),
    pd_increase=0.25,
    pr_decrease=-0.50,
    use_t2_progression=False,
)

MRANO_WITH_T2 = MRANO_VOLUMETRIC.variant(
    "mrano_with_t2",
    description=(
        "MRANO_VOLUMETRIC plus the T2/FLAIR progression component at +40% vs its own nadir. "
        "Kept as a standing ablation: it is what a naive full-RANO reading of this dataset "
        "produces, and it is markedly WORSE than leaving the component out. Do not use it as a "
        "rule; use it as the evidence for why the component is off by default."
    ),
    use_t2_progression=True,
)

ENHANCING_ONLY = ResponseCriteria(
    name="enhancing_only",
    description=(
        "Ablation: enhancing volume alone, no T2/FLAIR, no new-lesion, no clinical, no "
        "confirmation. Reproduces the single-signal prototype and exists to quantify what every "
        "other component adds."
    ),
    pd_increase=0.40,
    pr_decrease=-0.65,
    use_t2_progression=False,
    use_new_lesion=False,
    use_clinical=False,
    require_confirmation=False,
    pseudoprogression_weeks=None,
    block_pr_when_non_measurable=False,
    block_cr_when_steroids_increased=False,
)

#: Every profile the evaluation runs by default, in reporting order.
PROFILES: dict[str, ResponseCriteria] = {
    p.name: p for p in (MRANO_VOLUMETRIC, RANO_CLASSIC_PORTED, ENHANCING_ONLY, MRANO_WITH_T2)
}

DEFAULT_PROFILE = MRANO_VOLUMETRIC
