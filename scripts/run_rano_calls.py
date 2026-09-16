#!/usr/bin/env python3
"""
Run the RANO criteria over LUMIERE's shipped volumes and compare the calls to the expert ratings.

This is the LUMIERE-specific *harness*. The rule itself lives in ``rano.criteria`` and knows
nothing about this dataset -- that separation is the point, so the same criteria object serves
the offline evaluation here and the online pipeline later.

WHAT IT MEASURES
    Volumes come from DeepBraTumIA's own ``measured_volumes_in_mm3.json``, read straight out of
    the archive (599 files, ~0.1 s). Those are read BY KEY (``Enhancing_Core``), so they are
    immune to the 2026-07-30 label swap -- see label_schema.py. ``--source volumes-csv`` reads our
    own mask-derived volumetry instead, which reproduces the JSON exactly and is the path the
    pipeline will use; running both is a live check that they still agree.

WHAT IT CANNOT MEASURE, AND SO DOES NOT PRETEND TO
    * new lesions -- needs connected-component tracking across timepoints, not a volume table
    * clinical deterioration, steroid dose -- not in the dataset at all
    Both are passed as ``None``, not ``False``. The report counts how many calls were made with a
    component missing, instead of quietly scoring an incomplete rule as if it were complete.

COHORT NOTE
    ``--cohort all`` (the default) evaluates every patient, INCLUDING the 17 held out by
    output/cohort/cohort_lock.json. That is a deliberate choice by the operator, not an oversight,
    but it does spend the held-out arm: thresholds tuned against these numbers can no longer be
    validated on unseen data. The report always breaks results out by arm so the generalisation
    gap stays visible and so you can see exactly what was spent.

Usage:
    .venv/bin/python scripts/run_rano_calls.py
    .venv/bin/python scripts/run_rano_calls.py --cohort practice --profile mrano_volumetric
    .venv/bin/python scripts/run_rano_calls.py --source volumes-csv --timeline-patients 8
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import zipfile
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rano.adapters.lumiere import paths, weeks  # noqa: E402
from rano.adapters.lumiere.treatment import read_surgical_labels, treatment_from_labels  # noqa: E402
from rano.criteria import (  # noqa: E402
    PROFILES,
    CallPair,
    Response,
    ResponseCriteria,
    TimepointMeasurement,
    assess_trajectory,
    compare_calls,
    format_case_table,
    format_confusion,
    format_summary_line,
    format_timeline,
    split_by_group,
)
from rano.contract.treatment import TreatmentRecord  # noqa: E402
from rano.adapters.lumiere.zip_ref import ZipSource  # noqa: E402
from rano.criteria.timeline import BaselinePolicy, choose_baseline  # noqa: E402
from rano.labels import LABEL_SCHEMA  # noqa: E402
from rano.measurement import has_measurable_disease  # noqa: E402

DEFAULT_ZIP = ROOT / "Imaging-v202211.zip"
DEFAULT_VOLUMES_CSV = ROOT / "output" / "volume_audit" / "volumes.csv"
DEFAULT_RATINGS = ROOT / "LUMIERE-ExpertRating-v202211.csv"
DEFAULT_LOCK = ROOT / "output" / "cohort" / "cohort_lock.json"
DEFAULT_OUT = ROOT / "output" / "rano_calls"

#: Timepoints the expert marked as surgical states rather than response assessments.
PRE_OP, POST_OP = "Pre-Op", "Post-Op"
SCORABLE_RATINGS = {"CR", "PR", "SD", "PD"}

#: Weeks from the week-000 study to the end of chemoradiotherapy, used ONLY to place the
#: pseudoprogression window. Standard-of-care timing (Stupp): resection, ~4 weeks recovery, then
#: 6 weeks of concurrent chemoRT. It is a cohort-level assumption, not a per-patient fact -- the
#: dataset ships no radiotherapy dates. Override with --rt-end-week, or set it to a negative
#: number to disable the window without changing the profile.
RT_END_WEEK = 10.0

#: the enhancing label integer, from the locked schema -- never hardcoded here
ENHANCING_LABEL = next(k for k, v in LABEL_SCHEMA["DeepBraTumIA"].items() if v == "enhancing")

_JSON_MEMBER = re.compile(r"^Imaging/([^/]+)/([^/]+)/.*measured_volumes_in_mm3\.json$")

#: JSON key -> the compartment name our schema uses. Verified 599/599; see label_schema.py.
JSON_KEYS = {
    "Enhancing_Core": "enhancing",
    "Necrotic_NonEnhancing": "necrosis_nonenhancing",
    "Edema_Compartment": "edema",
}


#: Abbreviations the expert-rating CSV uses, expanded from its own column header.
RATIONALE_GLOSS = {
    "CRET": "complete resection of the enhancing tumour",
    "PRET": "partial resection of the enhancing tumour",
    "T2-Progr.": "T2 progression",
}


@dataclass(frozen=True, slots=True)
class ExpertRow:
    rating: str
    non_measurable: bool
    rationale: str

    def expert_rationale_display(self) -> str:
        """The rationale verbatim, with the dataset's own abbreviations spelled out.

        Kept verbatim apart from whole-token expansions: strings like ``Target L.: 1 12mm x 13mm``
        carry measurements that must not be reworded, and "None" is a rationale the reader actually
        entered, not a missing value.
        """
        text = self.rationale.strip()
        if not text or text.lower() == "none":
            return ""
        for abbr, full in RATIONALE_GLOSS.items():
            text = text.replace(abbr, full)
        return text


def _detail(a) -> str:
    """Compact numeric context: the volume, and the two ratios it was judged against."""
    if a.enhancing_mm3 is None:
        return "no enhancing measurement"
    bits = [f"{a.enhancing_mm3:,.0f} mm3"]
    if a.change_vs_nadir is not None:
        bits.append(f"{a.change_vs_nadir:+.0%} vs nadir")
    elif a.enhancing_mm3 > 0:
        bits.append("nadir was 0")
    if a.change_vs_baseline is not None:
        bits.append(f"{a.change_vs_baseline:+.0%} vs baseline")
    return "   ".join(bits)


# --------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------

def load_shipped_volumes(zip_path: Path) -> dict[tuple[str, str], dict[str, float]]:
    """(patient, timepoint) -> {compartment: mm3}, straight from DeepBraTumIA's own JSONs."""
    out: dict[tuple[str, str], dict[str, float]] = {}
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            m = _JSON_MEMBER.match(name)
            if not m:
                continue
            raw = json.loads(zf.read(name))
            out[(m.group(1), m.group(2))] = {
                compartment: float(raw[key]) for key, compartment in JSON_KEYS.items() if key in raw
            }
    return out


def load_volumes_csv(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    """Same shape, from our own atlas-space volumetry."""
    out: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for r in csv.DictReader(path.open()):
        if r["space"] != "atlas":
            continue
        out[(r["patient"], r["timepoint"])][r["compartment"]] = float(r["volume_mm3"])
    return dict(out)


def load_expert(path: Path) -> dict[tuple[str, str], ExpertRow]:
    rows = list(csv.DictReader(path.open()))
    rating_col = next(k for k in rows[0] if k.startswith("Rating ("))
    rationale_col = next(k for k in rows[0] if k.startswith("Rating rationale"))
    return {
        (r["Patient"].strip(), r["Date"].strip()): ExpertRow(
            rating=r[rating_col].strip(),
            non_measurable=r["NonMeasurableLesions"].strip().lower() == "x",
            rationale=r[rationale_col].strip(),
        )
        for r in rows
    }


def load_arms(path: Path) -> dict[str, str]:
    """patient -> 'practice' | 'held_out'. Empty when no lock file exists."""
    if not path.is_file():
        return {}
    lock = json.loads(path.read_text())
    arms: dict[str, str] = {}
    for arm in ("practice", "held_out"):
        block = lock.get(arm, {})
        # the two arms are shaped differently on purpose: the practice arm records per-patient
        # detail, the held-out arm records ONLY ids, so that reading the lock cannot leak the
        # held-out timepoints. Accept both rather than assuming one.
        for entry in block.get("patients", []):
            arms[entry["patient_id"] if isinstance(entry, dict) else entry] = arm
        for pid in block.get("patient_ids", []):
            arms[pid] = arm
    return arms


# --------------------------------------------------------------------------------------
# assembling trajectories
# --------------------------------------------------------------------------------------

def build_trajectories(
    volumes: dict[tuple[str, str], dict[str, float]],
    expert: dict[tuple[str, str], ExpertRow],
    rt_end_week: float,
    *,
    surgical: dict | None = None,
    baseline_rule: str = "post-op-label",
    policy: BaselinePolicy = BaselinePolicy(),
) -> dict[str, tuple[TimepointMeasurement | None, list[TimepointMeasurement], TreatmentRecord]]:
    """patient -> (reference scan or None, chronological measurements to score).

    Pre-operative studies are DROPPED, not merely left unscored: an untreated tumour used as the
    baseline makes the post-surgical drop look like a partial response for the rest of the
    trajectory. The first post-operative study becomes the reference, which is what RANO means by
    baseline for a resected glioma. Later post-operative studies stay in the trajectory -- they
    are genuine observations and must be allowed to lower the nadir -- but their expert rating is
    'Post-Op', so they are never paired against a response class.

    Timepoints with volumes but no expert row are kept for the same reason: dropping them would
    hide a real nadir and make later progression unreachable.

    The treatment record is rebuilt whenever ``surgical`` is given, whichever baseline rule is in
    force: the rule needs the surgery weeks to reset the reference at every operation, which is
    independent of how the FIRST reference was chosen.

    TWO WAYS TO PICK THE REFERENCE
        ``post-op-label``: the first study the expert labelled Post-Op, matched by timepoint LABEL,
        and one cohort-wide ``rt_end_week`` for the pseudoprogression window.

        ``post-rt``: the patient's own treatment record is
        rebuilt (surgery weeks, radiotherapy assumed from each patient's first operation) and the
        reference is the first scan far enough past that patient's radiotherapy -- see
        ``rano.criteria.timeline``. Scans BEFORE it are dropped, not merely unscored: they describe
        an untreated or mid-treatment state, so letting them set the nadir is the same error as
        using a pre-operative baseline. A patient with no qualifying scan is scored not at all.
    """
    per_patient: dict[str, list[tuple[str, dict[str, float]]]] = defaultdict(list)
    for (patient, tp), vols in volumes.items():
        per_patient[patient].append((tp, vols))

    out: dict[str, tuple[TimepointMeasurement | None, list[TimepointMeasurement]]] = {}
    for patient, entries in per_patient.items():
        entries.sort(key=lambda e: weeks.sort_key(e[0]))

        record = TreatmentRecord()
        if surgical is not None:
            record, _ = treatment_from_labels(
                surgical.get(patient), [weeks.week_offset(tp) for tp, _ in entries]
            )
        courses = record.radiotherapy or ()
        patient_rt_end = courses[0].end.week if courses and courses[0].end is not None else None

        measurements: list[TimepointMeasurement] = []
        for tp, vols in entries:
            row = expert.get((patient, tp))
            if row is not None and row.rating == PRE_OP:
                continue
            week = weeks.week_offset(tp)
            if baseline_rule == "post-op-label" or patient_rt_end is None:
                since_rt = (week - rt_end_week) if week is not None and rt_end_week >= 0 else None
            else:
                since_rt = (week - patient_rt_end) if week is not None else None
            measurements.append(
                TimepointMeasurement(
                    timepoint=tp,
                    week=week,
                    enhancing_mm3=vols.get("enhancing"),
                    t2_flair_mm3=vols.get("edema"),
                    new_lesion=None,            # not derivable from a volume table
                    clinical_deterioration=None,  # not in the dataset
                    steroids_increased=None,      # not in the dataset
                    non_measurable_only=row.non_measurable if row is not None else None,
                    weeks_since_rt=since_rt,
                )
            )

        if baseline_rule == "post-rt":
            choice = choose_baseline([m.week for m in measurements], record, policy)
            if not choice.found:
                out[patient] = (None, [], record)   # no eligible reference: nothing is scored
                continue
            out[patient] = (measurements[choice.index], measurements[choice.index + 1:], record)
            continue

        reference = None
        for i, m in enumerate(measurements):
            row = expert.get((patient, m.timepoint))
            if row is not None and row.rating == POST_OP:
                reference = measurements.pop(i)
                break
        out[patient] = (reference, measurements, record)
    return out


# --------------------------------------------------------------------------------------
# measurable disease, on the scans responses are measured AGAINST
# --------------------------------------------------------------------------------------

def reference_scans(trajectories) -> list[tuple[str, str]]:
    """Every scan that serves as a reference: the trajectory baseline, plus the first scan after
    each operation, which the rule makes a new reference. Only these need the ruler -- measuring
    all 599 scans would answer a question nobody asks."""
    wanted: list[tuple[str, str]] = []
    for patient, (reference, measurements, record) in trajectories.items():
        if reference is None:
            continue
        wanted.append((patient, reference.timepoint))
        previous = reference.week
        for m in measurements:
            if m.week is not None and previous is not None and any(
                previous < s.week <= m.week for s in (record.surgeries or ())
            ):
                wanted.append((patient, m.timepoint))
            previous = m.week if m.week is not None else previous
    return wanted


def measurable_flags(keys, zip_path: Path, cache: Path) -> dict[tuple[str, str], bool]:
    """(patient, timepoint) -> does this scan hold a lesion of at least 10 x 10 mm?

    Cached, because it is the only step here that opens masks: about a second per scan, against a
    tenth of a second for the whole volume table. Delete the cache file to recompute.
    """
    known: dict[tuple[str, str], bool] = {}
    if cache.is_file():
        known = {
            (r["patient"], r["timepoint"]): r["measurable"] == "True"
            for r in csv.DictReader(cache.open())
        }
    todo = [k for k in keys if k not in known]
    if todo and zip_path.is_file():
        src = ZipSource(str(zip_path))
        for i, (patient, tp) in enumerate(todo, 1):
            member = paths.dbt_mask(patient, tp)
            if not src.exists(member):
                continue
            mask = np.asarray(src.open_nifti(member).dataobj) == ENHANCING_LABEL
            known[(patient, tp)] = has_measurable_disease(mask)
            if i % 25 == 0:
                print(f"  ruler: {i}/{len(todo)} reference scans", file=sys.stderr)
        cache.parent.mkdir(parents=True, exist_ok=True)
        with cache.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["patient", "timepoint", "measurable"])
            w.writeheader()
            for (patient, tp), flag in sorted(known.items()):
                w.writerow({"patient": patient, "timepoint": tp, "measurable": flag})
    return known


def apply_measurability(trajectories, flags):
    """Stamp ``measurable_disease`` on the scans we measured; everything else stays ``None``."""
    out = {}
    for patient, (reference, measurements, record) in trajectories.items():
        def stamped(m):
            flag = flags.get((patient, m.timepoint))
            return m if flag is None else replace(m, measurable_disease=flag)

        out[patient] = (
            stamped(reference) if reference is not None else None,
            [stamped(m) for m in measurements],
            record,
        )
    return out


# --------------------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------------------

def run_profile(trajectories, expert, arms, criteria: ResponseCriteria):
    """Returns (pairs for comparison, all rows for the CSV)."""
    pairs: list[CallPair] = []
    rows: list[dict] = []
    for patient in sorted(trajectories):
        reference, measurements, record = trajectories[patient]
        result = assess_trajectory(
            patient, measurements, criteria, reference=reference, treatment=record
        )
        arm = arms.get(patient, "unassigned")

        for a in result.assessments:
            row = expert.get((patient, a.timepoint))
            rating = row.rating if row is not None else ""
            rows.append(
                {
                    "profile": criteria.name,
                    "patient": patient,
                    "timepoint": a.timepoint,
                    "week": weeks.week_offset(a.timepoint),
                    "arm": arm,
                    "enhancing_mm3": a.enhancing_mm3,
                    "change_vs_nadir": a.change_vs_nadir,
                    "change_vs_baseline": a.change_vs_baseline,
                    "t2_change_vs_nadir": a.t2_change_vs_nadir,
                    "calc": a.call.value,
                    "provisional": a.provisional_call.value if a.provisional_call else "",
                    "expert": rating,
                    "agree": int(a.call.value == rating) if rating in SCORABLE_RATINGS else "",
                    "driver": a.driver,
                    "reason": a.reason,
                    "unknowns": "|".join(a.unknowns),
                    "pseudoprogression_risk": int(a.pseudoprogression_risk),
                    "expert_rationale": row.expert_rationale_display() if row is not None else "",
                }
            )
            if a.scorable and rating in SCORABLE_RATINGS:
                pairs.append(
                    CallPair(
                        patient=patient,
                        timepoint=a.timepoint,
                        predicted=a.call.value,
                        expert=rating,
                        week=weeks.week_offset(a.timepoint),
                        group=arm,
                        reason=a.reason,
                        expert_reason=row.expert_rationale_display() if row is not None else "",
                        detail=_detail(a),
                        unknowns=a.unknowns,
                        complete=a.complete,
                    )
                )
    return pairs, rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=("json", "volumes-csv"), default="json")
    ap.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    ap.add_argument("--volumes-csv", type=Path, default=DEFAULT_VOLUMES_CSV)
    ap.add_argument("--ratings", type=Path, default=DEFAULT_RATINGS)
    ap.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--cohort", choices=("all", "practice", "held_out"), default="all")
    ap.add_argument("--profile", action="append", help="repeatable; default = every profile")
    ap.add_argument("--rt-end-week", type=float, default=RT_END_WEEK)
    ap.add_argument("--no-measurability", action="store_true",
                    help="skip the ruler pass; the measurable-disease gate then reports unknown")
    ap.add_argument("--measurability-cache", type=Path,
                    default=DEFAULT_OUT.parent / "measurement" / "measurable_references.csv")
    ap.add_argument(
        "--baseline",
        choices=("post-op-label", "post-rt"),
        default="post-op-label",
        help="post-op-label: the expert's first Post-Op study (as built). "
             "post-rt: the first scan far enough past THIS patient's radiotherapy (RANO 2.0).",
    )
    ap.add_argument("--timeline-patients", type=int, default=6, help="0 for none, -1 for all")
    ap.add_argument("--cases", type=int, default=0,
                    help="per-patient case tables with both sides' reasoning; 0 for none, -1 for all")
    ap.add_argument("--cases-disagreements-only", action="store_true")
    args = ap.parse_args()

    if args.source == "json":
        if not args.zip.is_file():
            print(f"archive not found: {args.zip}", file=sys.stderr)
            return 2
        volumes = load_shipped_volumes(args.zip)
        source_desc = f"DeepBraTumIA measured_volumes_in_mm3.json ({args.zip.name})"
    else:
        if not args.volumes_csv.is_file():
            print(f"volumes csv not found: {args.volumes_csv}", file=sys.stderr)
            return 2
        volumes = load_volumes_csv(args.volumes_csv)
        source_desc = f"our atlas volumetry ({args.volumes_csv.relative_to(ROOT)})"

    expert = load_expert(args.ratings)
    arms = load_arms(args.lock)

    if args.cohort != "all":
        keep = {p for p, a in arms.items() if a == args.cohort}
        volumes = {k: v for k, v in volumes.items() if k[0] in keep}

    surgical = read_surgical_labels(str(args.ratings)) if args.ratings.is_file() else None
    trajectories = build_trajectories(
        volumes, expert, args.rt_end_week, surgical=surgical, baseline_rule=args.baseline
    )
    flags = (
        {}
        if args.no_measurability
        else measurable_flags(reference_scans(trajectories), args.zip, args.measurability_cache)
    )
    trajectories = apply_measurability(trajectories, flags)
    profiles = [PROFILES[n] for n in (args.profile or list(PROFILES))]

    args.out.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    # --- provenance -------------------------------------------------------------------
    rated = {k for k, v in expert.items() if v.rating in SCORABLE_RATINGS}
    emit("=" * 78)
    emit("RANO calls vs expert ratings")
    emit("=" * 78)
    emit(f"volumes         {source_desc}")
    emit(f"timepoints      {len(volumes)} with volumes, {len(rated)} with a scorable expert rating")
    emit(f"patients        {len({p for p, _ in volumes})}   cohort filter: {args.cohort}")
    emit(f"baseline rule   {args.baseline}")
    emit(f"ruler           {len(flags)} reference scans measured"
         + ("" if flags else " (the measurable-baseline gate reports unknown)"))
    if args.baseline == "post-rt":
        emit("rt-end-week     per patient: first operation + 10 weeks (assumed; LUMIERE ships no RT dates)")
    else:
        emit(f"rt-end-week     {args.rt_end_week:g} (pseudoprogression window anchor; cohort-level assumption)")
    missing = sorted(k for k in rated if k not in volumes)
    if missing:
        emit(f"unmatched       {len(missing)} rated timepoints have no volumes, e.g. {missing[:3]}")
    emit("")

    all_rows: list[dict] = []
    reports = []
    for criteria in profiles:
        pairs, rows = run_profile(trajectories, expert, arms, criteria)
        all_rows.extend(rows)
        reports.append((criteria, compare_calls(pairs, criteria.name), pairs))

    # --- headline ---------------------------------------------------------------------
    emit("-" * 78)
    emit("PROFILE SUMMARY")
    emit("-" * 78)
    for criteria, report, _ in reports:
        emit(format_summary_line(report))
    emit("")
    for criteria, _, _ in reports:
        emit(f"  {criteria.name}: {criteria.description}")
    emit("")

    # --- per profile ------------------------------------------------------------------
    for criteria, report, pairs in reports:
        emit("=" * 78)
        emit(f"PROFILE  {criteria.name}")
        emit("=" * 78)
        emit(format_confusion(report))
        emit("")
        groups = split_by_group(pairs)
        if len(groups) > 1:
            emit("  by cohort arm:")
            for arm in sorted(groups):
                emit("    " + format_summary_line(compare_calls(groups[arm], arm)))
            emit("")

    # --- timelines for the first profile ----------------------------------------------
    if args.timeline_patients:
        criteria, report, _ = reports[0]
        n = None if args.timeline_patients < 0 else args.timeline_patients
        emit("=" * 78)
        emit(f"TIMELINES  {criteria.name}  (worst agreement first)")
        emit("=" * 78)
        emit(format_timeline(report, max_patients=n))

    if args.cases:
        criteria, report, _ = reports[0]
        n = None if args.cases < 0 else args.cases
        emit("=" * 78)
        emit(f"CASE TABLES  {criteria.name}  (worst agreement first)")
        emit("=" * 78)
        emit(format_case_table(report, max_patients=n,
                               disagreements_only=args.cases_disagreements_only))

    # --- artefacts --------------------------------------------------------------------
    calls_csv = args.out / "calls.csv"
    with calls_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0]))
        w.writeheader()
        w.writerows(all_rows)
    (args.out / "report.txt").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {calls_csv.relative_to(ROOT)} ({len(all_rows)} rows)")
    print(f"wrote {(args.out / 'report.txt').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
