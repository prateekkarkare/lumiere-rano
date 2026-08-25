"""
Run the fingerprinter against the real LUMIERE archive for a handful of patients.

Prints a compact per-patient/per-timepoint summary to stdout and writes the full pydantic
record to ``output/fingerprints/<patient_id>.json`` (see fingerprint/io.py).

Usage:
    .venv/bin/python scripts/run_fingerprint.py                      # first 3 patients
    .venv/bin/python scripts/run_fingerprint.py --n 10
    .venv/bin/python scripts/run_fingerprint.py --patients Patient-001 Patient-025
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rano.adapters.lumiere.adapter import LumiereAdapter
from rano.fingerprint.extractor import fingerprint_patient
from rano.fingerprint.io import write_patient_fingerprint
from rano.fingerprint.schema import PatientFingerprint

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ZIP = ROOT / "Imaging-v202211.zip"
DEFAULT_MANIFEST = ROOT / "LUMIERE-datacompleteness.csv"
DEFAULT_OUT = ROOT / "output" / "fingerprints"


def _summarize(record: PatientFingerprint) -> str:
    lines = [f"{record.patient_id}  ({record.n_timepoints} timepoints)"]
    for tp in record.timepoints:
        mods = ", ".join(sorted(tp.modalities)) or "(none)"
        mask = "yes" if tp.mask is not None else "no"
        lines.append(f"  {tp.label:<14} modalities=[{mods}]  mask={mask}")
        if tp.mask is not None:
            counts = ", ".join(f"{lc.label}:{lc.n_voxels}" for lc in tp.mask.label_histogram)
            lines.append(f"    mask label histogram: {counts}")
        if tp.mask_reference is not None:
            i = tp.mask_reference.intensity
            lines.append(
                f"    mask_reference intensity (brain-extent): n={i.n_voxels} "
                f"mean={i.mean:.1f} std={i.std:.1f}"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zip", default=str(DEFAULT_ZIP), help="path to Imaging-v202211.zip")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="path to LUMIERE-datacompleteness.csv")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="output directory for per-patient JSON records")
    parser.add_argument("--n", type=int, default=3, help="fingerprint the first N patients (default: 3)")
    parser.add_argument("--patients", nargs="+", help="explicit patient IDs, overrides --n")
    args = parser.parse_args()

    adapter = LumiereAdapter(args.zip, args.manifest)
    patient_ids = args.patients or adapter.patient_ids()[: args.n]

    for pid in patient_ids:
        record = fingerprint_patient(adapter.load_patient(pid))
        path = write_patient_fingerprint(record, args.out)
        print(_summarize(record))
        print(f"  -> {path}\n")


if __name__ == "__main__":
    main()
