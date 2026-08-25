"""
Re-run the volumetric-RANO sanity check with the CORRECTED label mapping.

IMPORTANT SCOPE NOTE. The original check (notebook M6c, ~42% agreement) read the shipped
``measured_volumes_in_mm3.json`` by KEY (``Enhancing_Core``), which was always the correct
compartment. **That result is NOT invalidated by the 2026-07-30 label correction** and does not
need re-running. What the correction endangered was anything reading the MASK integers through
``label_schema.py`` — i.e. this package's own volumetry, which is the first such consumer and was
fixed before it shipped.

This script therefore is not a repair of history. It re-derives the single-signal result from the
MASK (the path the pipeline will actually use) and reports it both ways, so the cost of the bug is
measured rather than asserted, and so the mask-based path is shown to agree with the JSON-based
conclusion:

  * CORRECTED  — enhancing = label 1  (what the pipeline will actually use)
  * SWAPPED    — enhancing = label 2  (reproduces the original, wrong, analysis)

Rule applied (volumetric RANO, Ellingson-style thresholds, deliberately the SIMPLEST possible
version because the point is to show a single signal is insufficient):
    CR  enhancing volume == 0
    PR  <= -65% vs BASELINE (first post-op timepoint with a mask)
    PD  >= +40% vs NADIR (smallest enhancing volume seen so far)
    SD  otherwise
Pre-Op / Post-Op timepoints are excluded — they are surgical states, not response assessments.

Usage:  .venv/bin/python scripts/rerun_volumetric_rano_check.py
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VOLUMES = ROOT / "output" / "volume_audit" / "volumes.csv"
DEFAULT_RATINGS = ROOT / "LUMIERE-ExpertRating-v202211.csv"
DEFAULT_OUT = ROOT / "output" / "rano_check"

PD_INCREASE = 0.40   # >= +40% vs nadir  -> progressive disease
PR_DECREASE = -0.65  # <= -65% vs baseline -> partial response
RESPONSE_LABELS = {"CR", "PR", "SD", "PD"}


def load_ratings(path: Path) -> dict[tuple[str, str], str]:
    rows = list(csv.DictReader(path.open()))
    key = next(k for k in rows[0] if k.startswith("Rating ("))
    return {(r["Patient"].strip(), r["Date"].strip()): r[key].strip() for r in rows}


def load_volumes(path: Path, compartment: str) -> dict[str, list[tuple[str, float]]]:
    """patient -> [(timepoint, atlas volume of ``compartment``)], in CSV (chronological) order."""
    out: dict[str, list[tuple[str, float]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for r in csv.DictReader(path.open()):
        if r["space"] != "atlas" or r["compartment"] != compartment:
            continue
        k = (r["patient"], r["timepoint"])
        if k in seen:
            continue
        seen.add(k)
        out[r["patient"]].append((r["timepoint"], float(r["volume_mm3"])))
    return out


def classify(trajectory: list[tuple[str, float]]) -> list[tuple[str, str]]:
    """Apply the volumetric rule along one patient's trajectory. Returns [(timepoint, call)]."""
    calls: list[tuple[str, str]] = []
    baseline: float | None = None
    nadir: float | None = None
    for tp, vol in trajectory:
        if baseline is None:
            baseline, nadir = vol, vol
            calls.append((tp, "BASELINE"))
            continue
        if vol == 0:
            call = "CR"
        elif nadir is not None and nadir > 0 and (vol - nadir) / nadir >= PD_INCREASE:
            call = "PD"
        elif nadir == 0 and vol > 0:
            call = "PD"                     # re-emergence from a complete response
        elif baseline > 0 and (vol - baseline) / baseline <= PR_DECREASE:
            call = "PR"
        else:
            call = "SD"
        calls.append((tp, call))
        nadir = min(nadir, vol) if nadir is not None else vol
    return calls


def evaluate(volumes, ratings, label: str) -> dict:
    total = correct = 0
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    per_class: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [correct, total]

    for patient, traj in volumes.items():
        for tp, call in classify(traj):
            expert = ratings.get((patient, tp))
            if expert not in RESPONSE_LABELS or call == "BASELINE":
                continue
            total += 1
            confusion[expert][call] += 1
            per_class[expert][1] += 1
            if call == expert:
                correct += 1
                per_class[expert][0] += 1

    return {
        "variant": label,
        "n_assessable_timepoints": total,
        "n_agree": correct,
        "agreement_pct": round(100.0 * correct / total, 1) if total else 0.0,
        "per_expert_class": {
            k: {"n": v[1], "recall_pct": round(100.0 * v[0] / v[1], 1) if v[1] else 0.0}
            for k, v in sorted(per_class.items())
        },
        "confusion_expert_x_predicted": {k: dict(v) for k, v in sorted(confusion.items())},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--volumes", default=str(DEFAULT_VOLUMES))
    ap.add_argument("--ratings", default=str(DEFAULT_RATINGS))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    ratings = load_ratings(Path(args.ratings))
    results = []
    for variant, compartment in (("CORRECTED (enhancing = label 1)", "enhancing"),
                                 ("SWAPPED (enhancing = label 2, the original error)",
                                  "necrosis_nonenhancing")):
        results.append(evaluate(load_volumes(Path(args.volumes), compartment), ratings, variant))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "volumetric_rano_check.json").write_text(json.dumps(results, indent=2))

    print("Volumetric-RANO single-signal check — enhancing volume only\n")
    for r in results:
        print(f"  {r['variant']}")
        print(f"    agreement with expert RANO: {r['agreement_pct']}%  "
              f"({r['n_agree']}/{r['n_assessable_timepoints']} assessable timepoints)")
        print("    recall per expert class: " + "  ".join(
            f"{k}={v['recall_pct']}% (n={v['n']})" for k, v in r["per_expert_class"].items()))
        print()
    print(f"-> {out_dir / 'volumetric_rano_check.json'}")


if __name__ == "__main__":
    main()
