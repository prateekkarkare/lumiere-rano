# LUMIERE — volumetric RANO from longitudinal glioblastoma MRI

A pipeline that reads a patient's brain MRIs over time and produces a **RANO response
assessment** — complete response, partial response, stable disease, or progression — from
segmented tumour volumes, and then measures itself honestly against expert radiologist ratings.

Built on the [LUMIERE dataset](#dataset) (Suter et al. 2022): 91 glioblastoma patients, 638 study
dates, expert RANO ratings on 616 of them.

## Where it actually stands

`mrano_volumetric` profile, 376 rated timepoints, all 91 patients:

| | agreement | balanced accuracy | CR | PR | SD | PD |
|---|---|---|---|---|---|---|
| **current** | 63.0% | **48.6%** | 50% | 20% | 52% | 72% |
| majority-class baseline ("always say PD") | 63.3% | 25.0% | — | — | — | — |

**Read the balanced-accuracy column, not the agreement column.** The expert called progression on
63% of scans, so answering "PD" on every scan without looking at anything scores 63.3%. Raw
agreement is not yet a meaningful score on this cohort, and quoting it alone would flatter the
pipeline into meaninglessness. Every report in this repo prints the majority baseline next to the
agreement for that reason.

The remaining gap is understood and written down, not mysterious — see
[`docs/ADJUDICATION_2026-08-31.md`](docs/ADJUDICATION_2026-08-31.md), which rules on ten
disagreements one at a time, and [`docs/FIXES_2026-09-02.md`](docs/FIXES_2026-09-02.md), which
records what the resulting fixes were actually worth. The single largest missing piece is
**per-lesion instancing** (connected components → per-lesion diameters → target selection → lesion
matching across timepoints), which is the only route to new-lesion detection, a real measurability
gate, and resection-cavity exclusion.

---

## Setup

Python **3.11+**. The dependency set is deliberately lean — numpy, nibabel, pydantic — so there is
no plotting or ML stack to install; the HTML reports build their own inline SVG.

```bash
python3 -m venv .venv
```

```bash
.venv/bin/pip install -e ".[test]"
```

Verify the install:

```bash
.venv/bin/python -m pytest -q
```

156 tests, ~1 second. They need no dataset — the rule engine, the adapters and the volumetry are
tested against synthetic fixtures on purpose, so a broken checkout is distinguishable from a
missing download.

<a name="dataset"></a>
## The data (you must download it separately)

**LUMIERE is not redistributed here.** It has its own license and every dataset file is
`.gitignore`d. Download it from the source below and place these in the repository root:

| File | Needed by |
|---|---|
| `Imaging-v202211.zip` | everything — masks, shipped volumes, transforms |
| `LUMIERE-ExpertRating-v202211.csv` | every evaluation script (the reference ratings) |
| `LUMIERE-pyradiomics-deepbratumia-features.csv` | label-schema verification |
| `LUMIERE-pyradiomics-hdglioauto-features.csv` | label-schema verification |
| `LUMIERE-MRinfo.csv`, `LUMIERE-datacompleteness.csv`, `LUMIERE-Demographics_Pathology.csv` | fingerprinting and cohort selection |

> Suter, Y., Knecht, U., Valenzuela, W., et al. (2022). *The LUMIERE dataset: Longitudinal
> Glioblastoma MRI with expert RANO evaluation.* Scientific Data 9:768.

Nothing is unzipped. `rano.adapters.lumiere.zip_ref` reads members straight out of the archive, so
the ~30 GB stays as one file on disk.

---

## Running it

### The main evaluation

```bash
.venv/bin/python scripts/run_rano_calls.py
```

Scores every patient, prints the confusion matrix and per-arm breakdown, and writes
`output/rano_calls/calls.csv` (one row per timepoint, with the reason for every call) and
`report.txt`. Useful flags:

- `--profile mrano_volumetric` — one profile instead of all four
- `--cohort practice` — restrict to the practice arm
- `--source volumes-csv` — use our own mask-derived volumetry instead of the shipped JSONs. These
  agree exactly; running both is a live cross-check, not a preference.
- `--cases -1 --cases-disagreements-only` — per-patient tables showing both sides' reasoning

### Everything else

| Command | What it does |
|---|---|
| `scripts/run_fingerprint.py --n 10` | Per-timepoint geometry/modality fingerprint → `output/fingerprints/` |
| `scripts/emit_data_contract.py --all` | The Piece-1 deliverable: one JSON of geometry, volumes, uncertainty and ratings |
| `scripts/lock_cohort.py` | Re-derives the frozen practice/held-out split (deterministic) |
| `scripts/audit_atlas_native_volumes.py` | Computes every volume in atlas space *and* on each native grid |
| `scripts/render_volume_audit.py` | → `docs/volume_audit.html` |
| `scripts/render_case_tables.py` | → `output/rano_calls/case_tables.html`, filterable |
| `scripts/adjudicate.py --bucket all` | Worksheet of every disagreement with the evidence to rule on it — and no verdict of its own |
| `scripts/sweep_guards.py` | Sweeps the two guard constants and prints what each value costs and buys |
| `scripts/rerun_volumetric_rano_check.py` | The original single-signal sanity check, with the corrected labels |

All take `.venv/bin/python` and `--help`.

---

## Layout

```
src/rano/
  criteria/      the RANO rule itself — pure, no dataset/NIfTI/zip knowledge
                   profiles.py     named threshold sets (the choice is a measured result,
                                   not a buried constant)
                   rano.py         nadir/baseline tracking, confirmation, pseudoprogression
                   measurement.py  inputs & outputs; an unavailable signal is None, never False
                   compare.py      agreement reporting, confusion, per-class recall
  adapters/      dataset-specific loading. lumiere/ reads the zip in place; the seam exists so
                 the same criteria object serves this evaluation and a future DICOM pipeline
  fingerprint/   per-image geometry, modality and space extraction
  validate/      conformance checks (mask/grid alignment, atlas consistency)
  volumetry/     voxel counting in a named space
  contract/      the data contract: what is guaranteed, and the size-dependent error bar
scripts/         thin harnesses over the above; all output goes to output/
tests/           156 tests, no dataset required
docs/            written record — see below
label_schema.py  integer -> compartment, the single source of truth. Import it; never hardcode.
```

### Design rules worth not re-litigating

- **An unavailable signal is `None`, never `False`.** New lesions, clinical deterioration and
  steroid dose are not in this dataset. Defaulting them to "no" would turn *we did not look* into
  *we looked and found nothing*. Every assessment reports which components it could not evaluate.
- **Two reference points, never one.** Progression is measured against the **nadir**, response
  against the **baseline**. Collapsing them is the most common way to get RANO wrong.
- **The nadir never includes the timepoint being assessed.** That makes the ratio non-negative by
  construction and progression unreachable — a bug that produces plausible output forever.
- **The rule package knows nothing about LUMIERE.** No zip, no NIfTI, no CSV. That is what makes
  it reusable rather than a one-dataset script.

---

## Two things to know before trusting a number

**The held-out arm.** `output/cohort/cohort_lock.json` freezes a 24-patient cohort split 7
practice / 17 held out. The split is version-controlled precisely because an unversioned one
drifts silently. `run_rano_calls.py` defaults to `--cohort all` and says so in its own header — the
published RANO thresholds are not fitted, so evaluating on everything is legitimate for them. But
any constant *we* invented is different: `scripts/sweep_guards.py` excludes the held-out arm by
default, because choosing a value by reading a curve is selection on data whether or not there is a
model involved.

**`RT_END_WEEK = 10.0` is an assumption, not a fact.** LUMIERE ships no radiotherapy dates, so the
pseudoprogression window is anchored on one cohort-wide constant derived from standard-of-care
timing (resection, ~4 weeks recovery, ~6 weeks chemoradiotherapy). It is the largest single lever
in the current results. Override it with `--rt-end-week`, or pass a negative number to disable the
window entirely.

## The written record

| Document | What it is |
|---|---|
| [`docs/ADJUDICATION_2026-08-31.md`](docs/ADJUDICATION_2026-08-31.md) | Ten disagreements ruled on one at a time: bug, divergence, or unknowable. The diagnosis. |
| [`docs/FIXES_2026-09-02.md`](docs/FIXES_2026-09-02.md) | What the resulting fixes were worth, including the one that turned out to change nothing. |
| [`docs/PROJECT_STORY.md`](docs/PROJECT_STORY.md) | The narrative of how the pipeline got here. |
| `docs/architecture.html` | Piece-1 architecture. |
| `docs/piece1_readiness.html` | The readiness memo that closed Piece 1. |
| `docs/volume_audit.html` | Atlas-vs-native volumetry: the error bar depends on lesion **size**, not compartment. |
