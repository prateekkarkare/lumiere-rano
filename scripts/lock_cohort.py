"""
Lock the working cohort and the practice / held-out split. Deterministic and re-runnable.

SELECTION CRITERION (from the LUMIERE descriptor's own arithmetic: 638 study dates, 599 with
automated segmentation, 616 expert-rated — and the two sets do NOT nest, ratings exist where
sequences are missing and vice versa):

    a timepoint is ASSESSABLE  <=>  it has a usable DeepBraTumIA segmentation
                                    AND an expert RANO response label (CR/PR/SD/PD)

Pre-Op and Post-Op are excluded: they are surgical states, not response assessments, so they
cannot be scored against and must not inflate a trajectory's length.

    a patient is IN COHORT     <=>  >= MIN_ASSESSABLE assessable timepoints

The threshold is a trajectory-length requirement, not a quality filter: RANO is longitudinal, so
a patient with two assessable timepoints cannot exercise nadir/confirmation logic at all.

SPLIT. Fully deterministic — no RNG, no seed to lose. Patients are first partitioned by whether
their trajectory contains a COMPLETE RESPONSE, then within each stratum ordered by assessable
count (descending), then by ID for ties; every PRACTICE_EVERY-th patient in each stratum goes to
PRACTICE.

Two stratifications, each for a reason:
  * by trajectory length (the stride, rather than taking a prefix) — so neither arm is
    systematically shorter than the other;
  * by presence of CR — because CR is rare (27 timepoints cohort-wide) and clustered in a few
    patients. An unstratified stride put ALL of them in held-out, which would have left the
    practice arm unable to exercise the complete-response branch at all. Note this looks only at
    the DISTRIBUTION of expert labels to balance the arms; it is a pre-scoring design decision,
    not an inspection of held-out trajectories.

HELD-OUT DISCIPLINE. This script prints per-patient detail for the PRACTICE arm only. For the
held-out arm it prints aggregate counts and nothing else. The IDs are written to the lock file
because the split must be fixed and auditable — but nothing here inspects their trajectories,
volumes, or ratings beyond counting them.

Usage:  .venv/bin/python scripts/lock_cohort.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONTRACT = ROOT / "output" / "contract" / "data_contract_cohort.json"
DEFAULT_OUT = ROOT / "output" / "cohort" / "cohort_lock.json"

MIN_ASSESSABLE = 6      # >= 6 assessable timepoints to form a scorable trajectory
PRACTICE_EVERY = 4      # every 4th patient in the stratified order -> practice
RESPONSE_LABELS = {"CR", "PR", "SD", "PD"}


def assessable_timepoints(patient: dict) -> list[dict]:
    return [
        t for t in patient["timepoints"]
        if t["readiness"] != "unusable" and t.get("expert_rating") in RESPONSE_LABELS
    ]


def summarise(patient: dict) -> dict:
    tps = assessable_timepoints(patient)
    return {
        "patient_id": patient["patient_id"],
        "n_timepoints_total": patient["n_timepoints"],
        "n_assessable": len(tps),
        "assessable_timepoints": [t["label"] for t in tps],
        "rating_mix": dict(Counter(t["expert_rating"] for t in tps).most_common()),
        "week_span": [tps[0]["week_offset"], tps[-1]["week_offset"]] if tps else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--min-assessable", type=int, default=MIN_ASSESSABLE)
    args = ap.parse_args()

    doc = json.loads(Path(args.contract).read_text())
    eligible = [summarise(p) for p in doc["patients"]]
    cohort = [p for p in eligible if p["n_assessable"] >= args.min_assessable]
    cohort.sort(key=lambda p: (-p["n_assessable"], p["patient_id"]))

    # stratify by CR presence, then stride within each stratum (see module docstring)
    practice: list[dict] = []
    held_out: list[dict] = []
    for has_cr in (True, False):
        stratum = [p for p in cohort if ("CR" in p["rating_mix"]) is has_cr]
        for i, p in enumerate(stratum):
            (practice if i % PRACTICE_EVERY == 0 else held_out).append(p)
    practice.sort(key=lambda p: (-p["n_assessable"], p["patient_id"]))
    held_out.sort(key=lambda p: (-p["n_assessable"], p["patient_id"]))

    def mix(group):
        c: Counter = Counter()
        for p in group:
            c.update(p["rating_mix"])
        return dict(c.most_common())

    lock = {
        "locked_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_contract": Path(args.contract).name,
        "criterion": {
            "assessable_timepoint": "usable DeepBraTumIA segmentation AND expert RANO label in "
                                    "{CR, PR, SD, PD}; Pre-Op/Post-Op excluded",
            "min_assessable_per_patient": args.min_assessable,
            "split_rule": f"partition by presence of a CR timepoint; within each stratum order by "
                          f"n_assessable desc then patient_id; every {PRACTICE_EVERY}th -> "
                          f"practice, remainder -> held out",
            "deterministic": True,
        },
        "totals": {
            "patients_screened": len(eligible),
            "patients_in_cohort": len(cohort),
            "assessable_timepoints_in_cohort": sum(p["n_assessable"] for p in cohort),
        },
        "practice": {
            "n_patients": len(practice),
            "n_assessable_timepoints": sum(p["n_assessable"] for p in practice),
            "rating_mix": mix(practice),
            "patients": practice,
        },
        "held_out": {
            "n_patients": len(held_out),
            "n_assessable_timepoints": sum(p["n_assessable"] for p in held_out),
            "rating_mix": mix(held_out),
            "discipline": "DO NOT INSPECT. IDs are recorded so the split is fixed and auditable. "
                          "No trajectory, volume or rating of these patients may be examined "
                          "until the scoring run.",
            "patient_ids": [p["patient_id"] for p in held_out],
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(lock, indent=2))

    print(f"COHORT LOCKED — {len(cohort)} patients, "
          f"{sum(p['n_assessable'] for p in cohort)} assessable timepoints "
          f"(threshold >= {args.min_assessable})\n")
    print(f"PRACTICE  {len(practice)} patients / {lock['practice']['n_assessable_timepoints']} timepoints"
          f"   {lock['practice']['rating_mix']}")
    for p in practice:
        print(f"   {p['patient_id']:<14} {p['n_assessable']:>2} assessable  "
              f"weeks {p['week_span'][0]:.0f}–{p['week_span'][1]:.0f}  {p['rating_mix']}")
    print(f"\nHELD OUT  {len(held_out)} patients / {lock['held_out']['n_assessable_timepoints']} timepoints"
          f"   {lock['held_out']['rating_mix']}")
    print("   (per-patient detail deliberately not printed — held-out set stays unexamined)")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
