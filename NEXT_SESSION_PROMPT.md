# LUMIERE Pipeline — Session Prompt: Build Piece 1 (Data Loader + Validator)

## Project & ultimate goal
Building a production-grade pipeline on the **LUMIERE** glioblastoma dataset (Suter et al. 2022,
*Scientific Data* 9:768). Ultimate goal: **accept any patient's longitudinal MRI → segment the
tumour → track it across timepoints → automatically produce a RANO response assessment
(volumetric RANO).** This is run as a real industry-deployment project: build step by step, no
skipping, **verify every claim from the data (never guess from memory)**, and keep it pedagogical
— I am a beginner in AI segmentation & registration.

## Working style (important)
- Explain concepts as you go; don't just dictate.
- Prefer verifying from the data over asserting. Silent-failure bugs (e.g. wrong label integer)
  are the enemy.
- Small, reviewable steps. I own the design judgment; you write boilerplate + propose designs
  and argue with me when I'm wrong.

## Where the project stands (decisions already LOCKED — do not relitigate)
1. **Canonical segmentation source = DeepBraTumIA.** 3 classes (necrosis/non-enh, enhancing,
   edema) → yields ET/TC/WT, isolates the T2/FLAIR compartment, excludes necrosis from enhancing;
   1mm isotropic; consistent MNI space across timepoints. HD-GLIO-AUTO (2 classes, per-case space,
   ~6mm slices) is kept only as an independent QC second-opinion (enhancing-vs-enhancing Dice),
   NOT a co-equal volume source, NOT label-fused.
2. **Label schema is decoded & verified** and lives in `label_schema.py` (project root) as the
   single source of truth — import it, never hardcode integers:
   - DeepBraTumIA: `{0:background, 1:necrosis_nonenhancing, 2:enhancing, 3:edema}`
   - HD-GLIO-AUTO: `{0:background, 1:t2_flair_nonenhancing, 2:enhancing}`
   - Composite regions: DeepBraTumIA ET={2}, TC={1,2}, WT={1,2,3}; HD-GLIO ET={2}, WT={1,2}.
   Verified (unique integers == documented set, plausible volumes) on Patient-001/002/003 for both.
3. **Volumes are trustworthy at the group level** (CR → −100% enhancing; two tools agree ~5%) but
   NOT yet validated voxel-wise against manual ground truth (LUMIERE ships no manual masks).
4. **RANO needs 4 signals, not 1.** A pure enhancing-volume threshold reproduces the expert RANO
   call only ~42%; ~62% of missed PDs are driven by NEW LESIONS or T2/FLAIR progression. So the
   eventual assessment engine needs: (a) enhancing volume, (b) T2/FLAIR non-enhancing volume,
   (c) new-lesion detection (⇒ requires registration), (d) reference/nadir + confirmation logic.
5. The **expert RANO labels are independent** of these auto-volumes (hand-drawn 2D Macdonald
   diameters + clinical judgment). They are our ground-truth TARGET, not a pipeline input.

## Data on disk (working dir: /Users/prateekkarkare/Desktop/Personal/Projects/lumiere)
- `Imaging-v202211.zip` (32.6 GB) — per patient/timepoint: raw skull-stripped CT1/T1/T2/FLAIR
  (`.nii.gz`, native), plus `DeepBraTumIA-segmentation/atlas/...` (MNI 1mm: `seg_mask.nii.gz`,
  `measured_volumes_in_mm3.json`, skull_strip images, `.tfm` transforms) and
  `HD-GLIO-AUTO-segmentation/{native,registered}/...`. Members are read directly from the zip
  (no extraction) — see helpers in `lumiere_eda.ipynb` cell "M0" (`load_nifti_from_zip`,
  `load_json_from_zip`, `m_dbt_mask`, `m_hdg_mask`, `m_dbt_vol`, `m_image`, `m_dbt_ss`).
- CSVs: `LUMIERE-datacompleteness.csv` (which modalities/masks exist per timepoint — USE THIS as
  the ground-truth manifest), `LUMIERE-MRinfo.csv` (DICOM geometry/scanner), `LUMIERE-ExpertRating-
  v202211.csv` (RANO labels + rationale), `LUMIERE-Demographics_Pathology.csv`, and two
  `LUMIERE-pyradiomics-*-features.csv`.
- `label_schema.py` — the locked label decode (import it).
- `lumiere_eda.ipynb` — full EDA (M0–M11 + M6c volumetric-RANO check). Reuse its patterns.
- `.venv` has nibabel, numpy, pandas, matplotlib, seaborn, lifelines. Use `source .venv/bin/activate`.

## THIS SESSION: Piece 1 = Data Loader + Validator (+ a thin volume-extractor vertical slice)

### Design intent (the seam that must be right)
LUMIERE is already preprocessed (HD-BET skull-strip, masks provided, DeepBraTumIA in MNI). The
deployment target is raw NEW patients (DICOM, not skull-stripped, arbitrary geometry, no mask).
So "skull-stripped / registered / has a mask" are **checks** when consuming LUMIERE but
**processing steps** when ingesting a real patient. Build Piece 1 as a **data contract + validator
with swappable ingestion adapters**: implement the LUMIERE adapter now; design (don't yet build)
the seam where a future DICOM adapter brings raw data up to the same internal contract. Same
internal case representation, same validator, two front doors.

### Functional requirements
1. **Case data contract.** Define a typed representation: a `Patient` with an ordered list of
   `Timepoint`s; each `Timepoint` exposes available modalities (CT1/T1/T2/FLAIR), the canonical
   mask (DeepBraTumIA), geometry metadata (shape, affine, spacing, orientation), and the space it
   lives in. Lazy-load arrays (stream from zip; don't hold everything in memory).
2. **Ingestion adapter (LUMIERE).** Build the case list from `LUMIERE-datacompleteness.csv` as the
   manifest; resolve zip member paths; parse/sort timepoints chronologically
   (`week-000-1`, `week-000-2`, `week-044`, ...).
3. **Validator → structured QC report (never crash on a bad case).** Per timepoint, run checks and
   emit pass/warn/fail, then bucket cases usable / needs-attention / unusable. Checks:
   - **Modalities present** per configurable contract (default required: CT1 + FLAIR; T1/T2 desired).
   - **Geometry consistency** across modalities of a timepoint (or record that they differ / need
     registration). Report spacing, isotropy, anisotropy ratio — FLAG but DO NOT reject anisotropic.
   - **Mask–image grid alignment**: mask shares the image affine + shape it claims to be in; report
     mismatches. Canonicalise orientation to RAS for reporting.
   - **Label-schema validity**: unique mask integers ⊆ documented set (from `label_schema.py`);
     off-schema integer = fail. Absent compartment = 0 mm³, NOT an error.
   - **Skull-strip sanity**: confirm non-brain background is zeroed / brain mask present.
   - **Longitudinal orderability**: timepoints parse and sort; baseline (post-op) identifiable.
4. **Thin vertical slice (proves the contract is usable).** For one validated patient: load →
   validate → compute enhancing & edema volumes (mm³) from the canonical mask via `label_schema.py`,
   using voxel_count × voxel_volume. Cross-check against the shipped `measured_volumes_in_mm3.json`
   (should match within rounding). This is the seed of the future volumetry stage.
5. **Provenance/logging** on everything the loader touches (source, mask source, space). Keep the
   loader NON-DESTRUCTIVE: it reads, checks, reports — it never resamples or transforms.

### Explicit architectural decisions to honour
- **Accept any geometry; never gate on isotropy.** Resampling to isotropic is a SEPARATE, explicit,
  logged, opt-in stage (linear interp for images, **nearest-neighbour for masks**) — not in the
  loader. Volumetry works at native spacing, so isotropy is only needed later (registration,
  new-lesion detection).
- **Config-driven contract** (required modalities, mask source, target space) — not hardcoded.
- Import `label_schema.py`; do not re-encode integers.

### Open questions to resolve WITH ME at the start (don't assume)
- Repo/module layout: propose a package structure (e.g. `src/lumiere/` with `io/`, `validate/`,
  `schema/`) and whether to keep streaming-from-zip or extract to a working tree.
- Case representation: dataclasses vs pydantic vs plain dicts — argue a choice.
- Which "space" the loader surfaces by default for DeepBraTumIA (atlas/MNI) and how it should
  represent that a timepoint's modalities may be in different native spaces.
- Test strategy: synthetic bad cases (missing modality, misaligned mask, off-schema label) as unit
  tests for the validator.

### DONE WHEN
- A working `Loader` that lists LUMIERE patients/timepoints from the completeness manifest and lazy-
  loads modalities + canonical mask from the zip.
- A `Validator` producing a structured per-case QC report with the checks above, run across the
  cohort with a summary of usable / needs-attention / unusable counts (+ the reasons).
- A thin volume-extractor reproducing the shipped enhancing/edema volumes for ≥1 patient within
  rounding, using `label_schema.py`.
- Unit tests for the validator on synthetic bad cases.
- A short note on what surprised us in the cohort-wide QC (e.g. how many timepoints fail which check).

### After this piece (roadmap, for context — NOT this session)
Piece 2: registration / coordinate-space handling (hard dependency for new-lesion detection).
Piece 3: segmentation stage (run/emulate DeepBraTumIA for new patients).
Piece 4: longitudinal RANO engine (the 4 signals + nadir/baseline/confirmation logic).
