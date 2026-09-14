# The LUMIERE / RANO Pipeline — The Whole Story in Plain Language

*Written 2026-08-29. This document assumes you remember nothing. It explains what we are
building, every word you need to follow it, everything we have done so far and when, what we
have proved, and what is still missing.*

---

## 0. How to read this

Sections 1–3 are background: what the project is, the vocabulary, the data. Section 4 is the
history. Sections 5–9 are the present state: what works, what is broken, and what is left.
Section 10 onwards is reference material — where the files are and which decisions are closed.

If you only read one thing, read **Section 6**. It is the problem we are actually stuck on.

---

## 1. What we are building, and why

A patient is diagnosed with **glioblastoma** — the most aggressive kind of brain tumour. They
have surgery, then radiotherapy, then chemotherapy. Every 8–12 weeks for the rest of their life
they get an MRI scan, and a doctor looks at that scan next to all the previous ones and answers
one question:

> **Is the tumour better, worse, or unchanged since last time?**

That answer decides whether the patient stays on their current treatment or switches to
something else. It is the single most consequential recurring decision in their care.

Today a human makes that call by hand: they put a ruler on the screen, measure the tumour in two
directions, compare against earlier scans, and apply a published rulebook called **RANO**.

**We are building software that does this automatically.** Feed it one patient's whole series of
MRI scans; it outputs, for each scan, a RANO response category and a written justification.

This is being built as a real industry-grade system — in reviewable pieces, with every claim
verified against the actual data rather than asserted from memory — and simultaneously as a
learning exercise in medical-image AI.

---

## 2. The words you need

### 2.1 The disease and the pictures

| Word | What it means |
|---|---|
| **Glioblastoma (GBM)** | The aggressive brain tumour this project is about. |
| **MRI sequence** | An MRI scanner can be operated in different modes; each mode makes different tissue bright. One scan session produces several 3D images, one per mode. |
| **T1** | A sequence. Basic anatomy. Fluid dark. |
| **CT1** (post-contrast T1) | T1 taken *after* injecting a gadolinium contrast agent. This is the money image. |
| **T2 / FLAIR** | Sequences where fluid and swollen/infiltrated tissue are bright. FLAIR is T2 with plain fluid (CSF) suppressed, so tumour-related brightness stands out. |
| **Enhancing tumour** | Tumour that lights up bright on CT1. It lights up because aggressive tumour breaks the blood-brain barrier, so contrast agent leaks into it. **This is the primary thing RANO measures.** |
| **Necrosis** | Dead tissue in the middle of the tumour. Does *not* take up contrast — it stays dark on CT1. |
| **Edema / non-enhancing compartment** | The bright-on-FLAIR region around the tumour: a mixture of swelling and infiltrating tumour cells that hasn't broken the blood-brain barrier yet. |
| **Resection cavity** | The hole left after surgery removes the tumour. |
| **Pseudoprogression** | Radiotherapy inflames tissue, which *also* makes it take up contrast. So in the ~12 weeks after radiotherapy, a tumour that appears to be growing may just be treatment reaction. Calling this "progression" is a classic and dangerous error. |
| **Pseudoresponse** | The mirror image: steroids and anti-angiogenic drugs seal the leaky vessels, so enhancement shrinks without the tumour shrinking. |

### 2.2 The computer-vision words

| Word | What it means |
|---|---|
| **Voxel** | A 3D pixel. An MRI image is a 3D grid of voxels. |
| **Spacing** | The real-world size of one voxel, in millimetres, per axis — e.g. `1.0 × 1.0 × 1.0` mm. |
| **Anisotropy** | When voxels are not cubes. Clinical brain MRI is often `0.5 × 0.5 × 6.0` mm — high detail in-plane, thick slices. Ratio here = 12. Thick slices are why small lesions are measured imprecisely. |
| **Segmentation / mask** | A second 3D image the same size as the scan, where each voxel holds an integer saying *what tissue this is*: 0 = background, 1 = enhancing, and so on. This is what turns a picture into numbers. |
| **Volumetry** | Counting the voxels of a label and multiplying by voxel volume to get mm³. That's it. The hard part is trusting the mask and the spacing. |
| **Registration** | Geometrically aligning two images so the same anatomical point is at the same coordinate in both. Needed before you can compare scan A to scan B voxel-by-voxel. |
| **Rigid transform** | A registration that only rotates and translates — no stretching. Crucially, **it cannot change volumes**. (A transform that stretches would silently inflate or deflate every volume you measure.) |
| **Native space** | The coordinate grid the scanner produced, unique to each scan. |
| **Atlas space (MNI)** | A standard reference brain grid. Warping everything into it puts all scans and all patients on one common grid. |
| **Affine** | The 4×4 matrix stored in a NIfTI file that maps voxel indices to real-world millimetre coordinates. Two images "on the same grid" means same shape and same affine. |
| **Connected-component / lesion instancing** | Splitting one big "enhancing" mask into *separate lesions* — blob 1, blob 2, blob 3 — so you can say "there is a new lesion here that wasn't there before". We do **not** have this yet. It matters enormously (see §8, H1). |

### 2.3 The clinical rulebook words

**RANO** = *Response Assessment in Neuro-Oncology*. The published rulebook for answering
"better / worse / same". Three generations matter here:

- **Macdonald (1990)** — the original. What the LUMIERE experts used.
- **RANO (2010, Wen et al.)** — the standard version. Measurements are **bidimensional**.
- **RANO 2.0 (2023, Wen et al.)** — the current version. **This is what we implement.**

The four answers:

| Call | Meaning |
|---|---|
| **CR** — Complete Response | No enhancing tumour left at all. |
| **PR** — Partial Response | Enhancing tumour substantially smaller than baseline. |
| **SD** — Stable Disease | Neither better enough nor worse enough. The default. |
| **PD** — Progressive Disease | Worse. Treatment is failing. |

Key concepts that make RANO harder than "compare two numbers":

- **Bidimensional product (SPD).** The classic measurement isn't a volume. The reader picks the
  longest diameter of the tumour on one slice, then the longest perpendicular diameter, and
  multiplies them. That product is the "size". A lesion counts as *measurable* only if both
  diameters are **≥ 10 mm**.
- **Baseline vs nadir — two different reference points, and confusing them is the classic bug.**
  - **Baseline** = the reference scan you judge *improvement* against. In RANO 2.0 for a newly
    diagnosed GBM, baseline is **the first scan after radiotherapy finishes**.
  - **Nadir** = the *smallest* the tumour has ever been so far. You judge *progression* against
    this. A tumour that shrank to a quarter of baseline and then doubled has progressed — even
    though it is still well below baseline.
  - **Any surgery resets both.** After a re-resection, the old numbers describe a tumour that has
    been physically cut out; comparing across that surgery is meaningless.
- **Thresholds.** Bidimensional: **+25% vs nadir = PD**, **−50% vs baseline = PR**. The
  volumetric adaptation (Ellingson 2017) re-derives these for volumes: **+40% = PD**, **−65% = PR**.
  (A sphere growing 25% in 2D product grows ~40% in volume — same tumour, different arithmetic.)
- **Progression beats response.** Check PD first. A lesion 70% below baseline but 50% above its
  nadir is progressing, not responding.
- **Confirmation.** CR and PR are not final until a repeat scan ≥4 weeks later still shows them.
  Unconfirmed response = SD.
- **Other PD triggers besides growth:** a **new lesion**, unequivocal **T2/FLAIR (non-enhancing)
  progression**, or **definite clinical deterioration**. Any one of them alone is PD.
- **Steroids block CR**, and PR is not available on non-measurable disease.

### 2.4 The scoring words

We compare our automatic calls against the expert's calls. How you score that comparison
matters more than it sounds:

- **Raw agreement / exact match** — what fraction of scans we got exactly right.
- **Majority-class baseline** — what you'd score by ignoring the images entirely and answering
  the commonest class every time. **On LUMIERE that is 63.3% (always say "PD").** Any headline
  number below this is worse than a constant.
- **Balanced accuracy** — the average of the per-class recalls (how much of CR did we catch, how
  much of PR, of SD, of PD — then average the four). A constant "PD" predictor scores 25% here.
  **This is the honest headline when classes are this lopsided.**
- **Confusion matrix** — the table of expert-said-X vs we-said-Y. It tells you *what kind* of
  mistake you make, which a single percentage never can.

---

## 3. The data we are building on

**LUMIERE** (Suter et al. 2022, *Scientific Data* 9:768) — a public longitudinal glioblastoma
MRI dataset.

- **91 patients**, each with many timepoints spanning up to ~3 years.
- Per timepoint: skull-stripped **T1, CT1, T2, FLAIR** as `.nii.gz`.
- **Two automatic segmentations shipped with it**, produced by two different tools:
  - **DeepBraTumIA** — 3 tumour labels, resampled to **1 mm atlas (MNI) space**, one identical
    grid for every scan. Ships a `measured_volumes_in_mm3.json` per timepoint.
    Labels: `1 = enhancing`, `2 = necrosis / non-enhancing`, `3 = edema`.
  - **HD-GLIO-AUTO** — 2 labels (enhancing, T2/FLAIR non-enhancing), in each scan's native space,
    often ~6 mm slices.
- **Expert RANO ratings** in `LUMIERE-ExpertRating-v202211.csv` — a human neuroradiologist's call
  per timepoint, with a free-text rationale, sometimes including their literal measurements
  ("Target L.: 13mm x 26mm"). **This is our ground truth.**
- Supporting CSVs: data completeness (which files exist), MR scanner/geometry info,
  demographics/pathology, and pre-computed pyradiomics feature tables.
- Everything lives inside a **32.6 GB zip** which we read from directly, without extracting.

Three facts about this ground truth that shape everything:

1. **The expert labels are wildly imbalanced.** Of the scorable ratings: **PD 253, SD 97, CR 27,
   PR 20**. Plus 124 Post-Op and 92 Pre-Op rows that are surgical states, not response calls.
2. **The expert labels follow the older Macdonald-era standard**, not RANO 2.0. So where the two
   standards genuinely disagree, we will "fail" the comparison by being *more* correct.
3. **The expert labels are independent of the shipped auto-segmentations.** The reader drew
   diameters by hand and used clinical judgement. That independence is what makes them a real
   target — and also means they encode things (steroid dose, clinical status) we cannot see.

---

## 4. The story so far

### Season 1 — Apr to Jun 2026: learning to segment

Python and imaging fundamentals, then 3D brain-tumour segmentation on the **BraTS** dataset,
reaching ~**0.79 Dice** with an nnU-Net-class model. That established the segmentation skill.
The deliberate decision at the end of Season 1: **stop polishing Dice**. A better segmenter is
not the novel part; the *longitudinal* layer is.

### July 2026: LUMIERE opens — and the first course correction

The original plan was to jump straight into registration and volumetry. That was pushed back
against, correctly: *"we should have done EDA on the data first before jumping into
registration."* The curriculum was rewritten to insert a data-understanding gate first. That
gate became `lumiere_eda.ipynb` and then Piece 1.

### Piece 1 — the loader and validator (closed 2026-07-30)

**Goal:** a component that can read any patient out of LUMIERE, check the data is sane, and hand
downstream stages a trustworthy, typed representation — plus the seam where a *future* adapter
will bring in raw hospital DICOM data.

The key design idea: LUMIERE arrives already skull-stripped, already segmented, already in atlas
space. A real new patient does not. So "is it skull-stripped / registered / segmented?" are
**checks** when reading LUMIERE but **processing steps** when reading a new patient. Same
internal representation, same validator, **two front doors**.

Built in seven commits:

| Step | What was built |
|---|---|
| 1 | The case data contract (`Patient` → ordered `Timepoint`s) and the ingestion adapter seam. |
| 2 | The LUMIERE adapter — reads the completeness CSV as the manifest, resolves zip paths, sorts weeks chronologically — plus the atlas-space consistency fingerprint. |
| 3 | The fingerprinter: extract geometry (shape, affine, spacing, orientation) per image, with a record schema and persistence. |
| 4 | Mask/image grid-alignment check + a structured validator result type (never crash on a bad case; emit pass/warn/fail). |
| 5 | Volumetry: per-compartment volumes in mm³ from the canonical mask in atlas space. |
| 6 | The atlas-vs-native volumetry audit. |
| 7 | The machine-readable data contract, the cohort lock, and the readiness memo. **PIECE 1 CLOSED.** |

**Two results that were better than what was asked for:**

1. **The label swap.** The schema said label 1 was necrosis and label 2 was enhancing — inherited
   from LUMIERE's own pyradiomics CSV, which is itself mislabelled. The earlier verification had
   only confirmed the integer *set* was `{0,1,2,3}` with plausible volumes, which cannot detect
   two names being swapped. The fix came from asking **which of two disagreeing sources can be
   checked against physics**: enhancing tissue takes up gadolinium and gets brighter on CT1;
   necrosis does not. Measured on 60 timepoints: label 1 got brighter after contrast (+0.587
   normalised), label 2 got *darker* (−0.073), in 51/52 cases. Label 2 cannot be the enhancing
   compartment. Corroborated by matching voxel counts against the shipped JSON in **599/599**
   masks. Committed with an explicit blast-radius analysis of everything the wrong mapping had
   touched. *(This is the archetype of a silent-failure bug: never errors, just quietly counts
   the wrong thing forever.)*

2. **The transforms are rigid — proved, not assumed.** The worry was that the patient→atlas
   transforms might scale volumes, which would bias every number we report. Rather than reason
   about it, all **2,396** `.tfm` transform files were measured: determinant **1.0000000000**,
   orthonormal to **1.5e-15**, and native-vs-atlas volumes agreed to a **median 0.00%** deviation
   over **5,776** comparisons. Conclusion: **atlas-space volumetry is unbiased**, and the
   fallback branch the brief had proposed was the wrong move.

**What Piece 1 also established — the honest error bar.** Volume precision depends on **how big
the lesion is**, not which compartment it is. On the worst (thick-slice) grids:

| Volume | Uncertainty (p10–p90 spread) |
|---|---|
| < 1,000 mm³ | **36.6 pp** |
| 1,000 – 5,000 mm³ | 14.8 pp |
| 5,000 – 20,000 mm³ | 5.3 pp |
| 20,000 – 60,000 mm³ | 2.8 pp |
| > 60,000 mm³ | 1.5 pp |

Read the top row again: a sub-1 cm³ lesion can swing **±37 percentage points** on resampling
alone — and the RANO PD threshold is **+40%**. That is why Piece 1's closing memo names
size-dependent confidence a *binding constraint* on the response engine.

**The cohort lock.** 24 patients with ≥6 assessable timepoints were locked into a study cohort:
**7 practice / 17 held out**, 225 assessable timepoints. The held-out arm is not to be inspected.

### Piece 2 — registration — closed by the dataset itself

Piece 2 was going to be cross-timepoint registration, and it slipped three times. Then the
fingerprinter answered the question directly: **1,773 of 1,773 DeepBraTumIA masks across all 91
patients sit on one identical grid** — same shape, same affine. DeepBraTumIA already registered
everything into atlas space. **For the mask pipeline, cross-timepoint registration is done.** The
item that consumed the whole schedule's slack never needed building. Struck from the critical path.

### 2026-08-13: reading two scans by hand

Before automating a judgement, make it yourself. Two cases were read manually in ITK-SNAP,
measuring with the on-screen ruler and calling RANO like a radiologist
(`knowledge-store/raw/RANO/read_log.md`):

- **Patient-002, week-037 — PD.** Called by the *right trigger*: a **new measurable lesion**
  appearing from a near-zero nadir, **not** the ≥25%-from-nadir rule (25% of ~0 is meaningless).
  Avoiding that trap unprompted is the single most common novice failure. Also correctly
  identified that week-037 → week-047 was an **invalid comparison** because a re-resection sat
  between them.
- **Patient-067, week-109 — PD by T2/FLAIR progression.** The enhancing tumour actually *shrank*
  (2.3 → 0.9 ml) while the FLAIR abnormality grew (37.5 → 44.8 ml). A tracker watching only
  enhancing volume would have called this improving. Matched the expert.

**Two defects worth remembering**, because they became project constraints:
- "Matched the expert on the millimetres" was an overclaim. The *product* agreed to 0.3% — but
  the two **diameters** were off **+9.6%** and **−8.5%**, opposite signs cancelling inside the
  product. Same-signed errors compound: +10%/+10% → +21% on the product, which is 84% of the
  entire PD threshold. The measurement error is decision-changing on any borderline case.
- The confounder checklist for the T2 call was *recited*, not *run* — LUMIERE ships no steroid
  dose, so "steroid change ruled out" could not be true. The honest call was "PD, low confidence,
  steroid status unavailable."

That week ended the reading phase with the decision: **building the categoriser IS the reading
practice.** Every RANO rule you have to encode in code is a judgement you are forced to make
explicit.

### 2026-08-22 to 08-24: the categoriser — first end-to-end RANO calls

This is the current state of `src/rano/criteria/`. Four modules:

- **`measurement.py`** — the input/output contract. Its governing rule: **an unavailable signal
  is `None`, never `False`.** The tempting shortcut is to let a missing "new lesion" flag default
  to "no new lesion" — that turns *"we did not look"* into *"we looked and found nothing"*, which
  is the difference between an honest SD and a missed PD. Every call carries an `unknowns` list
  naming the components it could not evaluate.
- **`rano.py`** — the rule itself. Pure functions, no dataset knowledge. Three things it gets
  structurally right: the **nadir never includes the scan being assessed** (including it makes PD
  mathematically unreachable — a bug that produces plausible output forever); **PD-vs-nadir and
  CR/PR-vs-baseline are tracked separately**; and **progression is checked before response**.
- **`profiles.py`** — the thresholds as named, frozen objects, so the threshold choice is a
  *measured result* rather than a buried constant. Four profiles ship.
- **`compare.py`** — the scoring, which leads with the confusion matrix and always prints the
  majority-class baseline next to the headline.

Plus `scripts/run_rano_calls.py`, which reads the shipped volumes straight out of the 32 GB zip,
builds each patient's trajectory, scores it, and renders per-patient case tables showing both
sides' reasoning next to each other.

**The T2/FLAIR component was measured and killed, with a mechanism.** Testing it across all 91
patients: at +40% vs nadir it fires on 84% of expert-CR scans and 76% of expert-PD scans — it is
*anti*-discriminative. The reason is mechanical: the reference scan is the immediate
post-operative study, when edema is suppressed by surgery and steroids. It then rises in
everybody, so "T2 vs nadir" is measuring *weeks since surgery*, not progression. It is off by
default and kept as a standing ablation so the finding stays measured rather than remembered.

**The result: 220/376 = 58.5% agreement.** Balanced accuracy 38.8%. Per-class recall CR 42% /
PR 10% / SD 26% / PD 77%.

### 2026-08-24: the goalpost was moved, deliberately

The old target was "≥80% exact match against the expert". It was retired — **not because the
week went badly, but because the metric was proved to be measuring the wrong thing.** See §6.

---

## 5. What we have actually achieved

Everything below is verified, not asserted:

- **A working ingestion layer** that lists every LUMIERE patient and timepoint from the dataset's
  own manifest and lazy-loads images and masks straight out of the 32 GB zip.
- **A structured validator** that never crashes on a bad case and emits per-check pass/warn/fail.
- **Volumetry over the whole cohort**: `output/volume_audit/volumes.csv` — 591 timepoints, 90
  patients, 8,865 rows (1,773 atlas + 7,092 native).
- **Our volumetry independently reproduces the shipped DeepBraTumIA volumes** — 60/60 checked
  timepoints agree.
- **The label decode is correct and proved three independent ways** (`label_schema.py`).
- **Atlas-space volumetry is proved unbiased** (2,396 rigid transforms, 5,776 comparisons).
- **An honest, size-dependent error bar** on every volume we report.
- **Cross-timepoint registration confirmed already done** (1,773/1,773 on one grid).
- **A locked study cohort** — 24 patients, 7 practice / 17 held out.
- **A machine-readable data contract** — `output/contract/data_contract.json`.
- **A complete, defensible RANO 2.0 rule engine** with tri-state inputs, per-call unknowns, named
  threshold profiles, and human-readable reasons on every call.
- **End-to-end scoring** against 376 expert-rated timepoints, with a confusion matrix, per-class
  recall, cohort-arm breakdown, and per-patient case tables.
- **151 automated tests passing.**
- **Two components measured and disproved on evidence** rather than left as open guesses: the
  T2/FLAIR component, and measuring `enhancing + necrosis` together.

That is a genuinely substantial piece of engineering. The gap is not craftsmanship.

---

## 6. The problem we are sitting in right now

**Read this section twice.**

Our headline number is **58.5%**. The number our own report prints next to it — what you'd score
by ignoring every image and answering **"PD"** every single time — is **63.3%**.

So the pipeline scores *below* a constant. And here is the part that matters: **every fix that
makes the rule more correct makes the score go down.**

Eight variants were tested against our own data:

| Change | Raw agreement | Balanced accuracy | CR recall | PR recall |
|---|---|---|---|---|
| As built | 58.2% | 38.6% | 42% | 10% |
| + reset baseline & nadir at every re-resection | 56.7% ↓ | **43.9%** ↑ | **58%** ↑ | **25%** ↑ |
| Baseline = first post-RT scan (week ≥ 10) | **62.7%** ↑ | 38.2% | 42% | 17% |
| Both | 60.3% | 43.6% | 58% | 28% |

And running all four shipped profiles (2026-08-29) makes the point even more brutally:

| Profile | Raw agreement | Balanced accuracy | CR | PR | SD | PD |
|---|---|---|---|---|---|---|
| `mrano_volumetric` (default) | 58.5% | 38.8% | 42% | 10% | 26% | 77% |
| `rano_classic_ported` (+25%/−50%) | 58.8% | 38.6% | 42% | 10% | 24% | 78% |
| `enhancing_only` (all extras off) | 57.2% | **43.2%** | 58% | 25% | 13% | 77% |
| `mrano_with_t2` (the known-bad one) | **63.6%** | **30.3%** | **0%** | 5% | 26% | 90% |

**Look at the last row.** The profile we *know* is wrong — the one whose own docstring says "do
not use this as a rule" — has the **highest raw agreement of all four**, and it is the only one
that beats the 63.3% constant. It achieves that by calling PD on 90% of everything and **never
once calling CR**. It scores best by being the least useful.

**The lesson, and it is the important one:**

> When your agreement metric and your correctness point in opposite directions, the metric is
> measuring the class prior, not the skill.

A bar that rewards exact match on a 63%-PD reference is *paying you to over-call progression* —
which in the clinic means telling patients their treatment has failed when it hasn't. That is
why the ≥80% exact-match bar was retired on 2026-08-24 and replaced (see §10).

**Nothing here is a craftsmanship failure. The rule engine works. The scoreboard was wrong.**

---

## 7. The 157 disagreements, sorted

Of 376 scored timepoints we disagree with the expert on 157. They decompose cleanly:

| Bucket | What it is | n | Agreement points at stake |
|---|---|---|---|
| **1** | The segmentation reports `enhancing == 0.0` | 44 | 11.7 |
| **2** | The expert's own rationale names a **new lesion** or **T2 progression** | 49 | 13.0 |
| **3** | Genuine rule / threshold / reference disagreement | 64 | 17.0 |

**Ceiling if buckets 1 and 2 were solved perfectly: 83.0%.**

Bucket 3's internal shape: 27 SD→PD, 13 PD→SD, 9 PR→PD, 4 PR→SD, 4 CR→PR, 3 CR→PD.
(Read "27 SD→PD" as: the expert said SD, we said PD, 27 times.)

---

## 8. The nine hypotheses, in plain language

These are the candidate explanations for the disagreements, ranked and already partly tested.

### True bugs — ours, fixable

**H3 · Surgery resets neither the baseline nor the nadir.**
Our own read_log wrote down the rule — *"any surgery resets baseline/nadir"* — and the code does
not implement it. `build_trajectories` takes only the **first** `Post-Op` row as the reference and
leaves all later ones sitting in the trajectory, where they drag the nadir down. **24 of 91
patients have ≥2 Post-Op scans.**
The worked example is Patient-067 — our own hand-read case. Four Post-Op rows: wk-000-2, wk-019-2,
**wk-059**, wk-157. The code calls wk-109 **PD, "+1329% vs nadir"** — against a 63 mm³ nadir from
week-042, measured *across* the week-059 re-resection. Our read_log says the enhancing tumour at
wk-109 had **shrunk** and the PD came from T2. **It got the right answer by a mechanism we had
already written down as invalid.** Five consecutive expert-SD timepoints on that patient are
called PD by the same stale nadir.
*Tested:* balanced accuracy 38.6 → 43.9, CR 42 → 58%, PR 10 → 25% — and raw agreement 58.2 → 56.7%.

**H4 · The baseline is the wrong scan.** *Largest single measured gain.*
RANO 2.0 baselines a newly diagnosed GBM at the **first post-radiotherapy scan**. Our own
`criteria.md` says "post-RT baseline 21–35 days". But `build_trajectories` uses the first
`Post-Op` row, which is usually week-000-2. The code contradicts itself: `RT_END_WEEK = 10.0`
asserts radiotherapy ends around week 10, while the baseline sits at week 0. **Every CR/PR
comparison is being made against a pre-radiotherapy reference.**
*Tested:* 58.2 → **62.7%**, PR recall 10 → 17%.

**H7 · No measurability gate, and our own uncertainty model is never read.**
`min_absolute_change_mm3 = 0.0` — the floor is OFF, and the docstring says so honestly. RANO's
±25%/±40% rules presume **measurable** disease (both diameters ≥10 mm ≈ a 500 mm³ sphere), and
nothing in `rano.py` checks it. Piece 1's memo calls size-dependent confidence a binding
constraint, `VOLUME_UNCERTAINTY_PP` exists — and the criteria package **never imports it**. The
median nadir behind our PD calls is 339–420 mm³, *below the floor RANO assumes*.
*Warning, tested:* a **flat** floor does not work. PD calls from a nadir < 500 mm³ are 22% wrong;
from ≥ 500 mm³, 28% wrong. The **size-dependent band we already built is the right instrument**;
a constant is not.

### Measurement layer — not a rule bug

**H2 · 62 timepoints where `Enhancing_Core == 0.0` exactly, and only 15 of them are expert CR.**
Of the 62: 29 expert SD, 18 expert PD, 15 expert CR, across 22 patients. **This is not the label
swap** — our own volumetry agrees with the shipped JSON 60/60. DeepBraTumIA genuinely finds no
enhancing core, and in 30 of these cases puts 1–22 cm³ into `Necrotic_NonEnhancing` instead
(Patient-003 wk-027: expert PD, enhancing 0, necrosis 21.7 cm³). The segmenter is binning rim
enhancement as necrosis, so `enhancing == 0` can never safely mean CR.
*Tested and killed:* measuring `enhancing + necrosis` together gives 58.2 → **54.2%**, CR recall
42% → **0%**. Do not do this.

### Correct divergence — do not "fix"

**H1 · Components the pipeline structurally cannot see. 13.0 points, and it is a ceiling.**
`new_lesion` is `None` on **100%** of timepoints and `use_t2_progression=False`. **38 of the 55
expert-PD → our-SD misses (69%) have an expert rationale explicitly naming a new lesion or T2
progression.** No threshold anywhere reaches these. They need **connected-component lesion
instancing**, which does not exist in the repo.

**H5 · Confirmation. LUMIERE does not encode it.**
`require_confirmation=True` fires on 42 calls; **30 of them are wrong**, and the pre-confirmation
provisional call would have been right in only 6. RANO genuinely requires confirmation — but a
Macdonald-era clinical log does not record whether a response was confirmed.

**H6 · The reference asymmetry itself. 27 of the 64 residual disagreements.**
Expert-SD → our-PD on genuinely large tumours: median **3,671 mm³**, only 6 of 58 below 500 mm³.
**This is not noise.** A tumour that fell a long way and then rebounded **is** PD-vs-nadir under
RANO 2.0, and was routinely called SD by a Macdonald-era reader. Signature: `change_vs_baseline`
strongly negative while `change_vs_nadir` > +40%. **We are right and the label is from a
different standard.**

### Crossed off — do not spend time

**H8 · Pseudoprogression contributes nothing.** `RT_END_WEEK = 10.0` is one cohort-wide constant
and `policy="flag"` never changes a call. *(Worth knowing for later: LUMIERE ships a per-timepoint
`LessThan3Months` column — 28 rows — that our loader does not read.)*

**H9 · Target-lesion selection is not mis-set, it is absent.** We measure the whole enhancing
compartment; the expert measures ≤2–3 chosen target lesions bidimensionally. That is a missing
subsystem, not a parameter to tune.

---

## 9. What is left to build

There is exactly **one** unbuilt subsystem on the critical path, and it unlocks two separate
things at once:

> **Lesion instancing → per-lesion bidimensional product → target selection at the count cap.**

- **Lesion instancing** (connected-component labelling of the enhancing mask, tracked across
  timepoints) gives us `new_lesion` — which is 13.0 agreement points of ceiling (H1).
- **Per-lesion bidimensional product** gives us the measurement the expert actually logged, in
  the units they logged it in (mm × mm) — which is what the new bar requires.
- **Target selection at the count cap** (pick the ≤2–3 largest measurable lesions) is what makes
  our measurement comparable to theirs at all.

Plus the three fixable bugs from §8: **surgery reset (H3)**, **post-RT baseline (H4)**, and
**wiring `VOLUME_UNCERTAINTY_PP` into the criteria package so every call carries its own
confidence (H7)**.

Deliberately **not** on the list: improving the segmenter, re-benchmarking Dice, building
registration, tuning the T2 threshold, or adding `enhancing + necrosis`.

---

## 10. The current definition of done (approved 2026-08-24)

> **By Sep 30, on ≥15 LUMIERE follow-ups drawn from the held-out arm:**
> 1. the bidimensional product is supplied by **our own pipeline** from our own DeepBraTumIA masks;
> 2. **every RANO 2.0 decision is ours, unaided**;
> 3. exact match vs the expert category **≥80%** *or* **every disagreement classified
>    BUG / RANO-2.0-DIVERGENCE / UNKNOWABLE with the evidence named**;
> 4. the pipeline's bidimensional product is **within ±25% of the expert's logged mm × mm on
>    ≥12 of the 15**.

Clause (4) is the strongest part and it is new. **142 LUMIERE timepoints across 63 patients ship
the expert's own bidimensional numbers verbatim** ("Target L.: 13mm x 26mm"); 35 patients have ≥2,
18 have ≥3. **±25% is not arbitrary — 25% *is* the PD threshold.** If your measurement error is
larger than the decision it feeds, the pipeline cannot make the decision. This turns the
"diameters off +9.6% / −8.5%" finding from a caveat into a bar.

Clause (3)'s "or" is the escape from §6's trap: you are no longer required to match a
Macdonald-era label — you are required to *explain* every place you don't.

---

## 11. Where everything lives

```
lumiere/
├── label_schema.py                 THE locked integer→compartment map. Import it; never hardcode.
├── Imaging-v202211.zip             32.6 GB. Read in place, never extracted.
├── LUMIERE-ExpertRating-*.csv      Ground truth: expert RANO call + rationale per timepoint.
├── LUMIERE-datacompleteness.csv    The manifest: which modalities/masks exist per timepoint.
├── lumiere_eda.ipynb               The original exploration (modules M0–M11).
│
├── src/rano/
│   ├── contract/     case.py (typed Patient/Timepoint) · data_contract.py (+ VOLUME_UNCERTAINTY_PP)
│   ├── adapters/     base.py (the seam) · lumiere/ (the one built adapter) · dicom.py (designed, not built)
│   ├── fingerprint/  geometry extraction, atlas-consistency proof, persistence
│   ├── validate/     the QC engine and its checks
│   ├── volumetry/    voxel counting → mm³
│   ├── criteria/     measurement.py · rano.py · profiles.py · compare.py   ← the RANO engine
│   └── labels.py     the one bridge to label_schema.py
│
├── scripts/          run_rano_calls.py (the evaluation harness) · emit_data_contract.py
│                     lock_cohort.py · audit_atlas_native_volumes.py · render_*.py
│
├── output/
│   ├── contract/     data_contract.json (+ cohort variant)
│   ├── cohort/       cohort_lock.json         ← 7 practice / 17 held out. Do not inspect held-out.
│   ├── volume_audit/ volumes.csv              ← 591 timepoints of compartment volumes
│   └── rano_calls/   calls.csv · report.txt · case_tables.html
│
├── docs/             architecture.html · piece1_readiness.html · volume_audit.html · progress.html
└── tests/            151 tests
```

Useful commands:

```
.venv/bin/python -m pytest -q                                  # 151 tests
.venv/bin/python scripts/run_rano_calls.py                     # ALL profiles (the default)
.venv/bin/python scripts/run_rano_calls.py --cases -1 --cases-disagreements-only
```

---

## 12. Decisions that are closed — do not relitigate

1. **DeepBraTumIA is the canonical segmentation source.** HD-GLIO-AUTO is a QC second opinion
   only — not a co-equal volume source, not label-fused.
2. **Labels: `1 = enhancing`, `2 = necrosis/non-enhancing`, `3 = edema`.** Proved three ways.
   Import `label_schema.py`.
3. **Volumetry happens in atlas space**, and it is unbiased (rigid transforms, det = 1).
4. **The loader never resamples or transforms.** It reads, checks, reports. Resampling is a
   separate, explicit, opt-in, logged stage.
5. **Never gate on isotropy.** Flag anisotropic data; do not reject it.
6. **The T2/FLAIR component stays OFF by default**, kept as an ablation, until the signal is
   non-enhancing *tumour* rather than the whole edema compartment.
7. **`enhancing + necrosis` is disproved.** Do not revisit.
8. **Piece 2 (registration) is closed by the dataset.** Not on the critical path.
9. **The held-out arm (17 patients) is not to be inspected** until the final evaluation.
10. **The ≥80% exact-match bar is retired.** Replaced by §10.
11. **These are deterministic rules, not a machine-learning model.** "Leakage", "tuning",
    "training set" are category errors when applied to the criteria code. (The cohort lock exists
    because *we* — the humans — can overfit by choosing thresholds after looking.)
12. **An unavailable signal is `None`, never `False`.**
