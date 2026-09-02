# Why the pipeline and the doctor disagree — 10 cases, ruled

*Written 2026-08-31. Written so it makes sense to someone who has forgotten everything,
including the person who wrote it.*

---

## 1. What this is, and why it was done

The pipeline reads a patient's scans over time and says whether the tumour is **better, worse,
or unchanged**. We compared its answers against a doctor's answers on 376 scans. It matched
**58.5%** of the time.

That looks like a failing grade. It isn't a useful one, because answering **"worse"** on every
single scan without looking at anything scores **63.3%** on this dataset — the doctor called 63%
of scans progression. So a number that looks bad may just mean the pipeline refuses to guess
"worse" every time.

**So the job was not to raise the number. The job was to find out what is actually wrong.**

We disagree with the doctor on **156 scans**. You cannot fix 156 things. So we took a sample,
worked out what went wrong in each one, and counted — because the thing that goes wrong most
often is the thing worth fixing.

**This document is that exercise, written out in full.**

---

## 2. First, the disagreements were sorted into three piles

Not every disagreement is the same kind of problem. Sorted in this order, so each scan lands in
exactly one pile:

| Pile | What it is | How many |
|---|---|---|
| **1** | The tumour-outlining software reported **zero** enhancing tumour | 44 |
| **2** | The doctor's written reason names a **new lesion** or **T2 growth** — things the pipeline structurally cannot see | 46 |
| **3** | **A genuine disagreement about the rules** — both sides looked at the same number and disagreed | 66 |

Piles 1 and 2 are already understood and are not rule problems. **Pile 3 is the only one where
asking "who is wrong?" makes sense**, so that is where the ten cases came from.

Of the 66 in pile 3, **23 belong to patients in the locked held-out group** and were skipped —
those are reserved for the final September evaluation and looking at them now would spend them.
That left **43** to choose from.

---

## 3. The three possible verdicts

For each disagreement, exactly one of these is true:

| Verdict | Meaning |
|---|---|
| **BUG** | Our program is wrong. It made a comparison the rulebook forbids. |
| **DIVERGENCE** | Our program is right. The doctor used an older rulebook (Macdonald, 1990) and we use the current one (RANO 2.0, 2023), and here the two genuinely give different answers. |
| **UNKNOWABLE** | Nothing in the data can settle it. The doctor's reason refers to something we do not have — steroid doses, how the patient looked in clinic, where the radiation beam was aimed. |

### The order to ask the questions in

**Always ask in this order and stop at the first yes.** Order matters, because a broken
comparison looks exactly like a legitimate disagreement if you do not check for it first.

1. **Did our program compare against a scan the rulebook says it shouldn't have?** → BUG
2. **Was the comparison fair, and do the two rulebooks genuinely disagree here?** → DIVERGENCE
3. **Does nothing in the data settle it?** → UNKNOWABLE

### Two rules learned the hard way while doing this

**Rule A — test a defect by removing it.** For each candidate defect, ask: *if I fixed only this
one thing, would the answer become right?* Several cases had three defects stacked, and this is
the only way to say which one actually caused the wrong answer.

**Rule B — a case that disappears is not a case that was fixed.** Twice, correcting the reference
scan meant the disputed scan became the new starting point — and starting points are never
scored. The disagreement vanishes from the tally without any call being made correctly. That is
not a win, and it must be recorded as such.

---

## 4. Two terms you must keep straight

Almost every ruling below turns on these, and mixing them up sends you to the wrong fix.

| Term | Meaning | Used to judge |
|---|---|---|
| **Baseline** | The reference scan the whole trajectory is measured against. RANO 2.0 says it must be **the first scan taken 21–35 days after radiotherapy ends** — *not* the scan taken right after surgery. | **Improvement** (CR, PR) |
| **Nadir** | The smallest the tumour has ever been so far. | **Worsening** (PD) |

**Any surgery resets both.** After an operation, the old numbers describe tissue that has been
physically cut out. Comparing across a surgery is meaningless.

---

## 5. The ten cases

Every number below came from the pipeline's own output. Volumes are enhancing tumour in mm³,
read from the segmentation software's own measurements.

---

### Case 1 — Patient-003, week 14 · doctor said SD, we said PR

```
week-000-2    15,418    Post-Op   "complete resection of the enhancing tumour"
week-014          59    SD        "less than 3 months, non-target lesion"   <-- this scan
week-027           0    PD
week-038      31,454    PD
```

**What we said:** PR (partial response) — the tumour shrank 100% from the 15,418 mm³ baseline.

**Working it through:**

- 59 mm³ is a lump about **5 mm across**. RANO requires two perpendicular diameters of **at least
  10 mm** before a lesion counts as measurable. There is nothing measurable here.
- The doctor said the same thing in their own words: *"non-target lesion"* means *there is no
  measurable target lesion, only incidental ones*.
- And look at the baseline: 15,418 mm³ of "enhancing tumour" on a scan the doctor labelled
  **complete resection**. That is not tumour. It is the bright rim around the fresh surgical
  cavity. RANO explicitly says to exclude the resection cavity from measurements. We count it.
- Our code has a switch for this (`block_pr_when_non_measurable`) and it is switched **on** — but
  it reads a column in the dataset's spreadsheet that is blank on this row, so it never fired.

> **VERDICT: BUG — no measurability floor.** A response cannot be called on 59 mm³.

---

### Case 2 — Patient-004, week 71 · doctor said PR, we said PD

```
week-000-2       175    Post-Op
week-020       4,031    PD
week-038      30,040    PD
week-041       3,690    Post-Op   <-- SECOND SURGERY
week-057      22,553    PD
week-071         634    PR                                   <-- this scan
week-086       6,199    PD
```

**What we said:** PD — 634 mm³ is **+262%** above the smallest-ever value of 175 mm³.

**Working it through:**

- That 175 mm³ nadir is from **week 0** — thirty weeks and **one operation** earlier.
- A surgery sits at week 41, between the nadir scan and this one. Under RANO the nadir must reset
  there. Comparing across it is void.
- The doctor is comparing week 71 against week 57 (22,553 mm³) and watching it collapse to 634.
  Obviously improving.

> **VERDICT: BUG — stale nadir carried across a surgery.**

**Note:** the *baseline* is also wrong here (it sits at week 0, pre-radiotherapy) — but the call
was "worse", and "worse" is judged against the **nadir**. Recording this as "wrong baseline"
would send you to the wrong fix.

---

### Case 3 — Patient-005, week 15 · doctor said SD, we said PD

```
enhancing 4,721 mm3, up +390% from a 963 mm3 nadir at week-000-2
doctor's reason: "Progression probably within the irradiated area"
```

**Working it through:**

- No surgery in between. 4,721 mm³ is large and unambiguous. The growth is real.
- The doctor is saying: *yes it grew, but it grew where the radiation beam was aimed, so this is
  probably inflammation from treatment, not tumour.*
- **We cannot check this.** The irradiated area is a treatment plan — a dose map drawn on a
  planning scan before treatment. It is not visible on an MRI, and the dataset does not ship one.
- The doctor hedged too: *"probably"*.

> **VERDICT: UNKNOWABLE.** The claim is not checkable from anything we have.

**What this case exposed:** our program *did* notice the risk. It flagged the scan as being inside
the post-radiotherapy window and called progression anyway, because the setting is
`"flag"` (note it, do nothing) rather than `"downgrade"` (refuse to call progression inside the
window). See finding **F1** in §7 — this turned out to matter a great deal.

---

### Case 4 — Patient-009, week 24 · doctor said CR, we said SD

```
week-000-1     7,717    Pre-Op
week-000-2       441    Post-Op   "partial resection"
week-014      14,206    SD        tumour grew back
week-015         784    Post-Op   "COMPLETE resection"      <-- SECOND SURGERY
week-024           1    CR                                  <-- this scan
week-066       8,248    PD
```

**What we said:** SD — via a three-step chain.

1. 1 mm³ is not *exactly* zero, so the code said PR, not CR.
2. A response must be confirmed by a later scan.
3. The next scan is **42 weeks later** and shows the tumour back — so the response was thrown
   away and demoted to SD.

**Working it through, using Rule A (remove one defect at a time):**

| Fix only this | Result | Right? |
|---|---|---|
| Reset the nadir at the surgery | still PR | no |
| Treat 1 mm³ as zero | CR, but confirmation still kills it | no |
| Remove the confirmation rule | PR survives | no — doctor said CR |

Nothing fixes it alone. So record the **last** thing that destroyed a nearly-right answer.

> **VERDICT: BUG — confirmation has no upper time limit.** A scan 42 weeks later is not
> confirming anything; it is the next chapter of the disease.

This is a **different defect** from the known limitation that the dataset does not record whether
responses were confirmed. That limitation is real and unfixable. **This** is our own
implementation being wrong. See finding **F2**.

---

### Case 5 — Patient-014, week 12 · doctor said SD, we said PR

```
week-000     26,991    Pre-Op
week-001        436    Post-Op   "complete resection of the enhancing tumour"
week-012         89    SD                                       <-- this scan
```

That is the **entire patient** — three scans.

**Working it through:**

- 89 mm³ is about 5.5 mm across. Not measurable.
- The 436 mm³ reference is on a scan the doctor labelled *complete resection* — again, surgical
  rim, not tumour.
- **Neither scan carries measurable disease.** There was never anything to call a response on.
  SD is correct, and the doctor left no note because there was nothing to say.

**Rule B applies here:** you might think "the baseline is wrong, it should be the post-radiotherapy
scan." But the first scan at week ≥10 **is week 12 itself**. It would become the starting point,
never get scored, and the disagreement would simply vanish. That is not the same as getting it
right.

> **VERDICT: BUG — no measurability floor.** Same defect as case 1.

---

### Case 6 — Patient-032, week 13 · doctor said CR, we said PR

```
week-000-2     7,174    Post-Op   "complete resection"
week-013           3    CR                                      <-- this scan
```

**What we said:** PR — because 3 mm³ is not *exactly* zero, and our code only calls complete
response on exactly zero.

Three voxels. That is segmentation noise, not disease.

> **VERDICT: BUG — the complete-response test is `== 0` instead of `<= a noise floor`.**

**But do not fix this in isolation** — see finding **F4**. Measured across the whole dataset,
`enhancing == 0 → CR` is *already* wrong about three times out of four. Sweeping near-zero scans
into that bucket would make things worse, not better.

---

### Case 7 — Patient-032, week 85 · doctor said CR, we said PD

```
week-000-2     7,174    Post-Op   "complete resection"     <-- surgery 1
week-013           3    CR
week-027           3    CR
week-040         716    CR        "T2 progression, new non-target lesion"
week-044       1,169    PD
week-046       2,189    Post-Op   "complete resection"     <-- surgery 2
week-055       3,968    PD
week-067       2,702    PD
week-069       7,532    Post-Op   "complete resection"     <-- surgery 3
week-075       3,085    PD
week-085          28    CR                                 <-- this scan
week-096          67    PD
```

**What we said:** PD — **+833%** above the smallest-ever value of 3 mm³.

**Working it through:**

- That "+833% growth" is three voxels becoming twenty-eight. Enormous as a percentage, nothing as
  disease. **Percentages of noise are meaningless** — and note our automated noise check said this
  "clears the band", because the band is itself a percentage. A limitation of the tooling, not
  evidence.
- **Two surgeries** sit between the nadir scan and this one, at weeks 46 and 69.
- Reset the nadir at week 69 (7,532 mm³) and 28 mm³ becomes a **99% drop**. The PD vanishes.

> **VERDICT: BUG — stale nadir carried across (two) surgeries.**

Fixing only the nadir lands on PR, not CR. Landing on CR also needs the noise floor from case 6.
**These bugs are entangled, not independent.**

**Also visible in this table, and important:** the doctor calls week 40 **CR** while 716 mm³ of
"enhancing tumour" is present, and writes *"new non-target lesion"*. Not a contradiction on their
side — see finding **F5**.

---

### Case 8 — Patient-035, week 8 · doctor said SD, we said PD

```
week-000-2        25    Post-Op   "complete resection"
week-008       8,596    SD        "Progression probably within the irradiated area"   <-- this scan
week-019      11,938    SD        "Progression probably within the irradiated area"
week-024      11,176    PD        new non-target lesion
week-036      28,135    PD        new target lesion 21mm x 33mm
week-050      26,498    PD
```

**Working it through:**

- Standard treatment is: surgery, ~4 weeks recovery, then ~6 weeks of radiotherapy. So
  radiotherapy runs roughly **weeks 4 to 10**. **Week 8 is during it.**
- RANO 2.0 requires the baseline to be the first scan **21–35 days after radiotherapy ends** —
  about week 13–15 here.
- **Week 8 is before that.** RANO does not assign better/worse/same to scans taken before the
  starting point. **Our program scored a scan that should never have been scored.**

That holds regardless of what the doctor was thinking, so it does not depend on the unverifiable
"irradiated area" claim — which is what separates this from case 3.

> **VERDICT: BUG — baseline is the post-surgery scan instead of the post-radiotherapy scan.**

---

### Case 9 — Patient-035, week 19 · doctor said SD, we said PD

Same patient, eleven weeks later. 11,938 mm³, **+47,652%** above a 25 mm³ nadir.

**Working it through:**

- Week 19 **is** genuinely after radiotherapy, so case 8's argument does not transfer.
- A tempting calculation: if week 8 were the reference, week 19 is **+38.9%**, below the +40%
  progression threshold — so SD. **Two problems with that.**
  1. **Week 8 was just ruled invalid** (case 8). You cannot use it as the reference one case later.
  2. **+38.9% is not safely below +40%.** The measurement uncertainty on volumes this size is
     ±5.3 percentage points on each side, so the true change lies somewhere between **+28% and
     +50%**. The threshold sits inside the error bar. No confident call is available either way.
- Looking at what is actually available: there is **no scan between week 8 and week 19**. So the
  first valid baseline **is week 19** — this scan. And baselines are not scored.

> **VERDICT: BUG — pre-radiotherapy baseline.** Consequence differs from case 8: the case
> **disappears** rather than resolving (Rule B).

**Discipline note recorded at the time:** the reasoning reached for the reference point that
produced the doctor's answer. It arrived there via a rule rather than via the score, so it was
legitimate — but that instinct is exactly the trap. **Ask what the rulebook says the reference is
first. Never pick the reference that lands where you want.**

---

### Case 10 — Patient-036, week 5 · doctor said SD, we said PD

```
week-002     4,152    Post-Op   (the reference)
week-005     7,574    SD        no reason written     <-- this scan
```

7,574 mm³, **+82%** above the nadir. Large tumour, no surgery in between, no noise question — the
growth is real.

**Working it through:**

- **Week 5 is during radiotherapy.** Same as case 8. Not an assessable scan.
- Tempting move: the doctor wrote nothing, but they used *"progression within the irradiated
  area"* twice on a different patient, so probably that is what they meant here too — which would
  make it UNKNOWABLE.
- **Rejected, deliberately.** Attributing a reason the doctor did not write is inventing evidence.
  If that is allowed, any disagreement can be explained away by imagining a good reason for the
  other side.
- And it is unnecessary, because there is a defect we can **prove** without their reason: the scan
  is before the valid baseline.

> **VERDICT: BUG — pre-radiotherapy baseline.**

**General rule extracted:** *rule on what you can demonstrate about your own program, not on what
you can plausibly guess about the reader.* UNKNOWABLE is a verdict about the **evidence**, not
about the doctor's silence.

---

## 6. The tally

```
BUG           9
DIVERGENCE    0
UNKNOWABLE    1
```

**Which defect, among the nine bugs:**

```
3   baseline is the post-surgery scan, not the post-radiotherapy one
2   no measurability floor
2   nadir carried across a surgery
1   confirmation has no upper time limit
1   complete response requires exactly zero
```

**Do not over-read DIVERGENCE = 0.** See §8, caveat C1 — the ten were taken in patient-ID order,
and the pattern where the two rulebooks genuinely disagree did not appear in this particular ten.
It exists elsewhere in the pile.

---

## 7. Findings that came out along the way

These were not the goal. They emerged from working the cases, and several contradict what was
believed going in.

### F1 · The pseudoprogression setting is doing nothing, and switching it helps

`pseudoprogression_policy` is set to `"flag"` — note the risk, then call progression anyway.
46 scans carry that flag; we disagree with the doctor on 26 of them.

Switching to `"downgrade"` — the strict reading, where progression cannot be called inside the
post-radiotherapy window without tissue proof:

```
flag       (current)   agree 58.5%   balanced 38.8%   SD recall 26%   PD recall 77%
downgrade              agree 62.2%   balanced 44.5%   SD recall 53%   PD recall 72%
```

**This is the only change anyone has tested that moves both numbers in the same direction.** Every
other correct fix trades one against the other.

It also overturns the standing belief that pseudoprogression "contributes nothing" — which was
true of the *current setting* (`flag` never changes a call, so of course it contributes nothing),
not of the concept. **Nobody had tested the other setting.**

*Caveat: rests entirely on the assumption that radiotherapy ends at week 10 for every patient.
See §8, A1.*

### F2 · "Confirmed" is implemented as "the next scan, whenever it happens"

The rule requires a confirming scan **at least** 4 weeks later. There is **no upper limit**.
Measured across all 45 confirmation-driven demotions, the gap to the "confirming" scan in weeks:

```
4, 4, 5, 7, 8, 8, 8, 9, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12,
13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 15, 15, 16, 16, 16, 16, 17,
17, 18, 19, 19, 19, 20, 27, 42, 50
```

A scan **50 weeks later** confirms nothing. **30 of the 45 demotions are wrong.**

### F3 · The measurability floor cannot be a volume threshold — in principle

RANO's definition of measurable disease is a **shape** test, not a size test:

- **2D:** two perpendicular diameters ≥10 mm, on ≥2 consecutive slices
- **3D:** ≥10 mm in **all three** perpendicular planes, requires isotropic acquisition ≤1.5 mm

Volume cannot express a shape test:

```
lesion (mm)               measurable?    volume mm3
10 x 10 x 10  sphere      YES                   524
 9 x  9 x  9  sphere      no                    382
15 x  6 x  6  cigar       no                    283
30 x 30 x  3  pancake     no                  1,414    <-- 3x the smallest measurable lesion
40 x  8 x  8  cigar       no                  1,340
12 x 11 x 10  blob        YES                   691
```

Any floor set at 524 admits the pancake and both cigars. Any floor that excludes them also
excludes legitimate measurable lesions.

**So `min_absolute_change_mm3` is not an under-tuned knob — it is the wrong instrument.** A
correct measurability gate needs **per-lesion diameters**, which needs splitting the mask into
separate lesions.

*Also worth recording: our masks are 1 mm isotropic only because the segmentation software
**resampled** them into a standard brain space. The original scans are often 6 mm slices.
Resampling does not create resolution that was never acquired. So the 3D pathway's acquisition
requirement is not really satisfied, which is a further argument for the 2D pathway.*

### F4 · The "obvious" complete-response fix would backfire

Measured distribution of small enhancing volumes against the doctor's calls:

```
volume          n     doctor said                ours
exactly 0      62     SD 29, PD 18, CR 15        SD 22, CR 40
1-10 mm3        9     CR 3, SD 2, PD 2, PR 2     SD 5, PD 2, PR 2
11-50 mm3      11     PD 6, CR 5                 PR 4, PD 4, SD 3
51-200 mm3     21     PD 12, SD 4, PR 4, CR 1    PD 10, SD 7, PR 4
201-524 mm3    19     PD 14, SD 4, PR 1          PD 10, SD 9
```

**On the 62 scans where the segmenter reports exactly zero enhancing tumour, the doctor calls
complete response only 15 times.** Twenty-nine are stable and eighteen are *progressing*. We emit
CR on 40 of them.

So `enhancing == 0 → CR` is **already wrong about 76% of the time**. Adding a noise floor that
sweeps near-zero scans into that bucket would push ~20 more timepoints into a rule that is already
the worst-performing branch in the engine.

The blocker underneath is the segmentation, not the rule: when it reports no enhancing tumour it
is often binning the tumour rim into the "dead tissue" compartment instead. Already proven
unfixable by adding the two compartments together (that made agreement 58.2 → 54.2% and complete-
response recall 42% → **0%**).

### F5 · The doctor's "complete response" is not our "complete response"

Patient-032, week 40: the doctor calls **CR** with **716 mm³** of enhancing tumour present.

Not a contradiction on their side. RANO says pick **at most 3** *target lesions* and track those
with a ruler; everything else is *non-target* and merely noted as present or absent. And RANO
explicitly says to **exclude from measurement**: cystic components, necrotic centres, and **the
resection cavity**.

Our pipeline measures the **entire enhancing compartment**, cavity rim included. The doctor
measures **three chosen lesions**, cavity excluded.

**We are not measuring a bigger version of their number. We are measuring a different object.**
No threshold reconciles that. It needs lesion splitting, cavity exclusion, and target selection.

### F6 · Moving the baseline wins by shrinking the exam, not by answering better

The wrong baseline was the most-cited defect (3 of 9), which makes it look like the obvious thing
to fix. Measured:

```
as built (baseline = post-op)      n=376   agree 58.5%   balanced 38.8%
baseline = first post-RT scan      n=294   agree 60.9%   balanced 38.3%

82 timepoints disappear — 22% of the entire evaluation set
```

Agreement rises because **a fifth of the hard cases stop being scored**. Balanced accuracy
actually gets slightly *worse*.

**Fix it anyway** — it is what the rulebook says and it is cheap. But it does not buy performance,
and it is not worth spending the remaining schedule on.

### F7 · Four independent routes to the same missing subsystem

| Route | Needs |
|---|---|
| Measurability gate (F3) | per-lesion diameters |
| New-lesion detection — 13 points of ceiling, pile 2 | splitting the mask into separate lesions |
| The mm × mm comparison the September target requires | per-lesion diameters |
| Excluding the resection cavity and necrotic centres (F5) | per-lesion segmentation and target selection |

**One build unlocks all four:** connected-component lesion instancing → per-lesion bidimensional
product → target selection at the count cap.

---

## 8. Caveats and assumptions

Everything here that could be wrong. Read this before quoting any number above.

### Assumptions baked into the numbers

**A1 · Radiotherapy is assumed to end at week 10, for every patient.**
The dataset ships **no radiotherapy dates**. `RT_END_WEEK = 10.0` is one cohort-wide constant
derived from standard-of-care timing (surgery, ~4 weeks recovery, ~6 weeks of treatment). It is
not a per-patient fact. **Rulings 8, 9 and 10 and finding F1 all rest on it.** If a patient's
treatment ran late or early, those rulings move.

**A2 · The volumes come from segmentation software, not from a human.**
Every mm³ in this document is a voxel count from DeepBraTumIA's output. No human verified any
outline. Our own volume calculation reproduces the shipped numbers exactly, so the *arithmetic* is
sound — but that says nothing about whether the outlines are correct. F4 is direct evidence they
often are not.

**A3 · Volume uncertainty is size-dependent, and it is large for small lesions.**
Measured over 5,776 comparisons on the coarsest grids: under 1,000 mm³ the spread is **±36.6
percentage points**; 1,000–5,000 mm³ is ±14.8; 5,000–20,000 mm³ is ±5.3. The progression threshold
is +40%. **For small lesions the error bar is comparable to the decision.** The rule engine does
not currently read this uncertainty at all.

**A4 · "Enhancing tumour" includes the resection cavity rim.**
RANO says to exclude it. We do not. This inflates the reference scan on nearly every patient
(see cases 1 and 5) and is a systematic bias, not random noise.

**A5 · The doctor's labels follow the 1990 rulebook, not the 2023 one.**
They are the target, not an oracle. Some disagreements are correct behaviour on our side.

### Limits of this exercise

**C1 · The ten cases were taken in patient-ID order, not sampled randomly or by shape.**
This is the most important caveat. The "rebound divergence" pattern — where a tumour falls a long
way and rebounds, which the 2023 rulebook calls progression and a 1990 reader called stable —
fires on only **5 of the 43** available cases, and **none of them landed in this ten**. So
**DIVERGENCE = 0 is partly an artifact of the draw**, not evidence that divergence does not exist.
A stratified re-draw would test this and has not been done.

**C2 · 23 of the 66 pile-3 cases were not looked at.** They belong to the locked held-out group and
are reserved for the September evaluation.

**C3 · Ten cases is a small sample.** The defect counts (3 / 2 / 2 / 1 / 1) are indicative, not
statistically meaningful.

**C4 · The pile boundaries are a judgement, not a fact.** Pile 2 is defined by keyword-matching the
doctor's free-text reason for "new" or "T2 progression". A slightly different keyword rule shifts
2–3 cases between piles 2 and 3.

**C5 · Several cases had multiple defects stacked.** Each was recorded under the one that actually
caused the wrong answer (Rule A), but the counts would shift under a different attribution rule.

**C6 · The automated noise check in the worksheet compares percentages.** For tiny volumes
(case 7: 3 mm³ → 28 mm³) percentage-based noise checking is meaningless and reports "clears the
band" when it does not. Do not trust that column below ~500 mm³.

---

## 9. Code fixes required

Ordered by cost. **None of the first four needs more than a few days.**

### Fix 1 · Pseudoprogression policy — *one word*

**Where:** `src/rano/criteria/profiles.py`, `pseudoprogression_policy`
**Change:** `"flag"` → `"downgrade"`
**Why:** RANO says progression inside the post-radiotherapy window is not callable without tissue
proof or a lesion outside the radiation field. The code path already exists and has never been
switched on.
**Measured effect:** agreement 58.5 → 62.2%, balanced 38.8 → 44.5%, stable-disease recall 26 → 53%.
**Blocked on:** nothing. Caveat A1 applies.

### Fix 2 · Confirmation needs an upper time limit — *~10 lines*

**Where:** `src/rano/criteria/profiles.py` (add a field), `src/rano/criteria/rano.py` (the
confirming-scan search in `_confirm`)
**Change:** add `confirmation_max_weeks`. If the only candidate scan falls beyond it, **keep the
original call and flag it as unconfirmable** rather than demoting it to stable disease.
**Why:** the current search takes the first scan *at least* 4 weeks out, with no maximum. Gaps of
27, 42 and 50 weeks are currently treated as confirmation.
**Measured context:** 45 demotions, 30 of them wrong.

### Fix 3 · Surgery resets the baseline and the nadir — *~30 lines, two files*

**Where:** `src/rano/criteria/measurement.py` (add a field marking a scan as a surgery),
`src/rano/criteria/rano.py` (reset the reference state when the loop sees one),
`scripts/run_rano_calls.py` (set the field — currently it takes the *first* post-operative scan as
the reference and silently leaves later ones in the list as ordinary follow-ups)
**Why:** the rulebook says any surgery resets both reference points. 24 of 91 patients have two or
more operations.
**Measured effect:** balanced accuracy 38.6 → 43.9, complete-response recall 42 → 58%,
partial-response recall 10 → 25% — **and raw agreement 58.2 → 56.7%**. Correct, and it costs the
headline number. Expect that and report it honestly.

### Fix 4 · Baseline = first post-radiotherapy scan — *~20 lines*

**Where:** `scripts/run_rano_calls.py`, the reference-selection loop
**Change:** choose the first scan at week ≥ RT end + 3–5 weeks, instead of the first
post-operative scan.
**Why:** RANO 2.0 mandates it, explicitly forbidding post-surgical and pre-radiotherapy scans.
**Measured effect:** 58.5 → 60.9% agreement, but **82 timepoints (22%) stop being scored** and
balanced accuracy slightly drops. **Do it for correctness, not for the number** (F6).

### Fix 5 · The complete-response test — *small change, but see the warning*

**Where:** `src/rano/criteria/rano.py`, the `enhancing == 0` test
**Change:** `== 0` → `<= a noise floor`
**Why:** three voxels should not block a complete response (case 6).
**WARNING:** **do not ship this alone.** F4 shows `enhancing == 0 → CR` is already wrong ~76% of
the time; widening it makes things worse. It should land only alongside work on the segmentation
problem underneath it.

### Fix 6 · The measurability gate — *the expensive one, and it is not a config change*

**Where:** does not exist yet.
**Why the existing switch cannot do it:** `min_absolute_change_mm3` is a volume threshold, and
measurability is a **shape** test (F3). No number works.
**What it actually needs:**
1. **Connected-component labelling** of the enhancing mask — split it into separate lesions
2. **Exclude** the resection cavity and necrotic centres (RANO requires this; we currently count them)
3. **Per-lesion diameters** — longest axis and longest perpendicular
4. **Target selection** — keep the largest 3 measurable lesions, track the same ones over time
5. **Match lesions across timepoints**, which then also yields new-lesion detection

**This is the single remaining subsystem.** It is the only fix for the measurability floor, the
only route to new-lesion detection (13 points of ceiling), the only way to produce the mm × mm
numbers the September target requires, and the only way to measure the same object the doctor
measures. **Four independent routes, one build** (F7).

### Also worth wiring in

`VOLUME_UNCERTAINTY_PP` in `src/rano/contract/data_contract.py` exists, is measured, and has
**never been imported by the rule engine.** Every call should carry its own confidence. Case 9 is
the demonstration: +38.9% against a +40% threshold with a ±10.6 percentage-point band is not a
call, it is a coin flip, and nothing in the current output says so.

---

## 10. What was NOT decided

- **Which single defect gets the remaining schedule.** The evidence points hard at Fix 6 —
  it is the only expensive one, and the only one the September target cannot be met without. That
  decision is recorded elsewhere.
- **Whether the divergence pattern is real.** C1 — needs a stratified re-draw.
- **Anything about the 23 held-out cases.** Deliberately untouched.

---

## 11. How to reproduce everything above

```bash
# all four threshold profiles, the headline comparison
.venv/bin/python scripts/run_rano_calls.py

# the adjudication worksheet: every disagreement, bucketed, evidence attached, verdict blank
.venv/bin/python scripts/adjudicate.py --bucket 3 --exclude-arm held_out --render
```

Files produced:

```
output/adjudication/worksheet_bucket3.csv    machine-readable, one row per disagreement
output/adjudication/worksheet_bucket3.txt    the readable cards used for these rulings
output/adjudication/rulings.md               the raw ruling log written during the session
output/rano_calls/report.txt                 the headline comparison
```
