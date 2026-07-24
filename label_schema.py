"""
LUMIERE segmentation label schema — the single source of truth for integer -> compartment.

Every downstream volumetry MUST import these maps instead of hard-coding integers.
A wrong integer here fails SILENTLY (counts the wrong compartment, never errors), so the
mapping below is not guessed from memory — it is triangulated across three independent,
mutually-corroborating sources and then VERIFIED against real mask files.

PROVENANCE / CITATION
---------------------
Dataset: Suter et al. (2022), "The LUMIERE dataset: Longitudinal Glioblastoma MRI with
expert RANO evaluation", Scientific Data 9:768.

Important honesty note: the LUMIERE *descriptor PDF* (LUMIERE-readme.pdf) documents the
compartment NAMES and file layout but does NOT tabulate the integer encodings. The integer
-> compartment mapping is therefore established from the dataset's own machine-readable
artifacts, all shipped with Suter et al. 2022:

  1. LUMIERE-pyradiomics-{deepbratumia,hdglioauto}-features.csv
        -> carry paired `Label` (integer) and `Label name` (string) columns. OFFICIAL.
  2. DeepBraTumIA measured_volumes_in_mm3.json (per timepoint)
        -> keys corroborate the DeepBraTumIA compartment names.
  3. Underlying tool publications:
        - DeepBraTumIA (BraTS-style multi-class output).
        - HD-GLIO / HD-GLIO-AUTO: Kickingereder et al. 2019, Lancet Oncology
          (2 classes: contrast-enhancing tumour; T2/FLAIR non-enhancing).

VERIFICATION (empirical, against real .nii.gz masks)
----------------------------------------------------
Confirmed unique integers == documented set, with anatomically plausible voxel counts, on
Patient-001 (week-000-1), Patient-002 (week-000), Patient-003 (week-000-2), for BOTH sources.
Verified 2026-07-22 via nibabel (see notebook M5 / lumiere_eda.ipynb).
"""

# --- DeepBraTumIA: atlas/segmentation/seg_mask.nii.gz (MNI, 1mm isotropic) ---
# NOTE: label 1 merges necrosis AND non-enhancing solid tumour (JSON key: Necrotic_NonEnhancing).
DEEPBRATUMIA_LABELS = {
    0: "background",
    1: "necrosis_nonenhancing",  # pyradiomics "Necrosis";  JSON Necrotic_NonEnhancing
    2: "enhancing",              # pyradiomics "Contrast-enhancing";  JSON Enhancing_Core
    3: "edema",                  # pyradiomics "Edema";  JSON Edema_Compartment (T2/FLAIR)
}

# --- HD-GLIO-AUTO: registered/segmentation.nii.gz (per-case reference space) ---
HDGLIO_LABELS = {
    0: "background",
    1: "t2_flair_nonenhancing",  # pyradiomics "Non-enhancing" (edema + non-enh tumour, merged)
    2: "enhancing",              # pyradiomics "Contrast-enhancing"
}

LABEL_SCHEMA = {"DeepBraTumIA": DEEPBRATUMIA_LABELS, "HD-GLIO-AUTO": HDGLIO_LABELS}

# --- Composite (BraTS-style) regions used by RANO volumetry ---
# ET = enhancing tumour;  TC = tumour core (ET + necrosis/non-enh);  WT = whole tumour.
COMPOSITE_REGIONS = {
    "DeepBraTumIA": {"ET": {2}, "TC": {1, 2}, "WT": {1, 2, 3}},
    "HD-GLIO-AUTO": {"ET": {2}, "WT": {1, 2}},  # cannot form TC (no necrosis/edema split)
}

# --- Canonical mask decision (the required 3-line note) -----------------------------------
# CANONICAL for RANO compartment volumetry = DeepBraTumIA, because:
#   (1) 3 classes -> yields ET/TC/WT and isolates the T2/FLAIR compartment + excludes necrosis
#       from enhancing (RANO requires this); HD-GLIO's 2 classes cannot give TC.
#   (2) 1mm isotropic -> low partial-volume error; HD-GLIO's ~6mm slices make volumes noisier.
#   (3) consistent MNI space across all timepoints -> longitudinal comparability + new-lesion reg.
# HD-GLIO-AUTO role = independent second opinion (enhancing-vs-enhancing Dice) for QC gating,
# NOT a co-equal volume source and NOT label-fused (the two taxonomies differ).
CANONICAL_SOURCE = "DeepBraTumIA"
