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

*** CORRECTION 2026-07-30 — labels 1 and 2 were SWAPPED here until this date. ***
The 2026-07-22 check above confirmed only that the integer SET was {0,1,2,3} with plausible
volumes. That cannot detect a swap: exchanging two compartment NAMES leaves both the set and
the plausibility intact. And source (1) below was not independent corroboration of the names
after all -- the pyradiomics CSV `Label name` column is ITSELF mislabeled, and this file simply
inherited its error. Three independent checks, all agreeing, establish the true mapping:

  a. Shipped-JSON correspondence, over ALL 599 atlas masks in the archive:
        measured_volumes_in_mm3.json["Enhancing_Core"]        == count(label 1)  in 599/599
        measured_volumes_in_mm3.json["Necrotic_NonEnhancing"] == count(label 2)  in 599/599
     (16/599 also satisfy the reverse -- those are ties where the two counts are equal.)

  b. Contrast-enhancement physics, 60 randomly sampled timepoints (52 with both labels
     present). Enhancing tissue takes up gadolinium and brightens on CT1 relative to
     pre-contrast T1; necrosis does not. Using each scan's own brain median to normalise:
        label 1 enhanced more than label 2 in 51/52 cases
        median normalised (CT1 - T1):  label 1 = +0.587    label 2 = -0.073
     Label 2 LOSES signal after contrast -- it cannot be the enhancing compartment.

  c. Clinical plausibility: Patient-001/week-044 has 5 voxels of label 1 and 18,615 of
     label 2 -- a near-complete response of the enhancing component with residual necrosis,
     which is a textbook post-treatment picture. The swapped reading (no necrosis, 18,605mm3
     of enhancement 44 weeks post-op) is not.

HOW IT HAPPENED (worth recording, because the evidence was already in hand):
lumiere_eda.ipynb had ALREADY derived the correct mapping empirically, by matching mask voxel
counts to the JSON, and printed `{1: 'Enhancing', 2: 'Necrosis', 3: 'Edema'}` directly above the
contradicting CSV table. The contradiction was seen and resolved the wrong way round, concluding
"use the 'Label name' STRING for volumetrics; the mask integers follow a different convention" --
trusting the CSV strings over the dataset's own numbers. This file was then written from the CSV.
The lesson is not "verify empirically" (that was done) but "when two sources disagree, the one
that can be checked against physics wins".

CONSEQUENCE -- narrower than it first appears:
  * Analyses that read the shipped measured_volumes_in_mm3.json by KEY (`Enhancing_Core`) were
    always CORRECT and need no re-run. That includes the Task-1 / notebook-M6c volumetric-RANO
    check and its ~42% agreement figure, which used the JSON, not the mask integers.
  * Analyses that read the MASK and resolved integers through this file would have measured
    NECROSIS as enhancing. No such analysis had shipped when the error was found -- the rano
    package's volumetry was the first consumer, and it was fixed before use.
"""

# --- DeepBraTumIA: atlas/segmentation/seg_mask.nii.gz (MNI, 1mm isotropic) ---
# NOTE: label 2 merges necrosis AND non-enhancing solid tumour (JSON key: Necrotic_NonEnhancing).
# The trailing comments give the JSON key, which is the VERIFIED correspondence (see a. above).
# The pyradiomics `Label name` strings are deliberately NOT cited here: they are wrong.
DEEPBRATUMIA_LABELS = {
    0: "background",
    1: "enhancing",              # JSON Enhancing_Core           -- verified 599/599 + physics
    2: "necrosis_nonenhancing",  # JSON Necrotic_NonEnhancing    -- verified 599/599 + physics
    3: "edema",                  # JSON Edema_Compartment (T2/FLAIR) -- verified 599/599
}

# --- HD-GLIO-AUTO: registered/segmentation.nii.gz (per-case reference space) ---
# RE-VERIFIED 2026-07-30 by the same contrast-enhancement test that caught the DeepBraTumIA
# swap, on 40 sampled timepoints in HD-GLIO's own registered space: label 2 enhanced more in
# 40/40 (median normalised CT1-T1: label 2 = +0.571, label 1 = -0.034). This mapping is CORRECT
# as documented -- only DeepBraTumIA was affected. (The pyradiomics `Label name` column happens
# to be right here and wrong there, which is why per-tool re-verification was necessary.)
HDGLIO_LABELS = {
    0: "background",
    1: "t2_flair_nonenhancing",  # verified: does NOT enhance
    2: "enhancing",              # verified: enhances
}

LABEL_SCHEMA = {"DeepBraTumIA": DEEPBRATUMIA_LABELS, "HD-GLIO-AUTO": HDGLIO_LABELS}

# --- Composite (BraTS-style) regions used by RANO volumetry ---
# ET = enhancing tumour;  TC = tumour core (ET + necrosis/non-enh);  WT = whole tumour.
COMPOSITE_REGIONS = {
    # ET follows the 2026-07-30 correction: enhancing is label 1, NOT label 2.
    # TC and WT are unions of both tumour labels, so the swap never affected them.
    "DeepBraTumIA": {"ET": {1}, "TC": {1, 2}, "WT": {1, 2, 3}},
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
