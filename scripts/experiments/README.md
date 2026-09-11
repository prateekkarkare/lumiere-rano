# Experiments

The investigations behind `docs/progress.html`. The **decisions** they led to live in `src/` and
`scripts/`; these are the *questions* — kept so every number in the doc can be reproduced, and
so that six months from now you can re-run the reasoning rather than trust a summary of it.

They are scratch-quality on purpose. Each one answered a single question and then stopped.
Read them as a lab notebook, not a library.

**Run from the repository root:** `.venv/bin/python scripts/experiments/<name>.py`

Anything that aligns visits needs `output/registration/transforms.json` — make it with
`.venv/bin/python scripts/run_registration.py` (74 s). Every experiment reads the **practice
arm only**; the 17 held-out patients are never opened.

---

## 1. Is one lesion really one lesion?

| Script | Question | What it found |
|---|---|---|
| `probe_cc.py` | Patient-067 week-109 shows three separate patches on slice 84. Three lesions, or one? | One. All three belong to the same 3D object. That lesion is split into several pieces on **all 16** of its slices — never a single blob anywhere. |
| `probe_gap.py` | What fills the gaps between those patches? | Necrosis. It is a **broken ring**: adding the dead centre back turns three arcs (53, 37, 9 voxels) into one region of 371. |

## 2. Measuring against the radiologists — the first attempt

| Script | Question | What it found |
|---|---|---|
| `reverse_engineer.py` | Does filling holes change the long diameter? How do our numbers compare to what the radiologists wrote? | Filling rarely matters (identical on 12 of 17). Our numbers were too big — often 3×. |
| `where_is_36mm.py` | Patient-072: the radiologist says 13 mm, we say 36 mm. Where does 36 come from? | The ruler was measuring **across the gap between separate pieces** on one slice — straight through healthy brain. This became rule 2 of the ruler. |
| `who_is_the_target.py` | Which of our objects is the one they measured? | A small second lesion, not our biggest. The first sign that target selection is its own problem. |
| `erode_test.py` | Is Patient-072's big object several lumps joined by thin necks? | No. Eroding 1 mm leaves it in one piece. It is genuinely one solid mass. |

## 3. Tracking lesions from visit to visit

| Script | Question | What it found |
|---|---|---|
| `track.py` | Follow Patient-072's lesions by overlap. | The lesion is followed cleanly from before surgery to week-120 — but **zero** new lesions are found at the eight visits where the radiologist reports one. |
| `growth_sweep.py` | Between visits, how much of a lesion stays put? | A median of **2%**. That number is what first suggested something was wrong with alignment, not with tumours. |
| `tolerance.py` | Does a few mm of slack rescue matching? | Only for big lesions. Uses the original mask-growing code, which grew a diamond rather than a cube — so it reproduces the *pre-fix* 1% figure. The like-for-like baseline is 4% (see `tracking_after_reg.py`). |

## 4. Are the scans actually lined up?

| Script | Question | What it found |
|---|---|---|
| `registration_check.py` | A brain does not move between visits. Do the brain masks line up? | No — about **92%** overlap, and the brain's centre moves **2.7 mm** (worst 10 mm). |
| `is_it_registration.py` | Could the mismatch be skull-stripping, or the tumour reshaping the brain? | Neither. Brain volumes agree within 1%; only 4% of the mismatch is near the lesion. |
| `coverage.py` | Could the scans just cover different amounts of head? | No. Brain height agrees in 85 of 86 pairs. Same brain, sitting in a different place. |
| `shift_only.py` | Does sliding the brains together fix it? | Only 22% of it. The head is also **tilted** — which is why real rigid registration was needed. |
| `tracking_after_reg.py` | After registration, do small lesions match? | Lesions under 500 mm³: **4% → 30%**. Large lesions: 90% → 98%. |

## 5. New-lesion detection — the open problem

| Script | Question | What it found |
|---|---|---|
| `track_registered.py` | Did registration fix Patient-072's missing new lesions? | No — still zero at all nine visits. It did remove two false alarms caused purely by misalignment. |
| `which_compartment.py` | Are the new lesions hiding in necrosis or oedema instead? | No. No compartment shows new separate objects at those visits. |
| `where_is_the_new_tissue.py` | Where does the newly-enhancing tissue actually appear? | **0.2 to 6.7 mm** from the old tumour — touching it. When disease appeared 87.6 mm away, we caught it. |
| `new_lesion_check.py` | The overlap detector against the radiologist, split by *kind* of new lesion. | The first arm-wide scoreboard. An earlier version of `newlesion_arm.py`. |
| `newlesion_arm.py` | Arm-wide: when we say "new lesion", do they? | Overlap rule, assessable visits only: **recall 36%, precision 50%**. Distance is no help: 3.5 mm vs 3.6 mm. Above 500 mm³: **5 of 5** confirmed. The script prints the uncorrected all-transitions table first, then the corrected one — the correction is kept visible on purpose. |
| `new_focus.py` | Is the new tissue a **lump** with an interior, or a thin **shell** on a growing edge? | **The best rule found.** Catches 16 of 22 new lesions (73%) at a 20 mm³ lump, against 8 of 22 for overlap. |
| `extras_vs_rating.py` | The lump rule also fires where the notes mention no new lesion. Real disease, or noise? | Probably real: it fires on **62%** of those visits already called PD, and on **0%** of CR and PR visits. |
| `treated_zone.py` | Radiologists ask whether new enhancement falls inside the radiation field. Can we approximate that? | No. Both groups move together at every margin; 41% precision against a 31% base rate. |
| `rationale_vocabulary.py` | How often do the notes actually reason about the field or the 3-month clock? | About **1 in 10** notes. A real consideration, but a minority one — which is why the field feature could not carry the problem. |

---

## Not here, and why

The scratch **ruler** grew up into `src/rano/measurement/`, with the tests in
`tests/test_measurement_ruler.py` and the real comparison in `scripts/run_measurement.py`. The
scratch **registration prototype** grew up into `src/rano/registration/`. Their scratch versions
were dropped rather than kept alongside the real thing.
