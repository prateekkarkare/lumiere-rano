"""
Measure practice-arm lesions with the ruler and compare against the radiologist's own numbers.

WHY THIS COMPARISON IS THE POINT
--------------------------------
The synthetic shapes prove the ruler measures what it is pointed at. They cannot say whether we
point it at the same thing a radiologist does. The rationale column of the expert ratings carries
their written measurements -- "Target L.: 29mm x 38mm" -- which is the only ground truth we have
for the mm x mm product, and the reason the ruler exists at all.

TWO COLUMNS, DELIBERATELY
-------------------------
``biggest`` is the lesion the stated task picks: the largest surviving component. ``best match``
is whichever of our lesions comes closest to what the radiologist wrote. When the two agree, the
ruler is being judged. When they diverge, the disagreement is about TARGET SELECTION -- which
lesion to measure -- and that is a separate unsolved problem (a radiologist follows a lesion they
chose at an earlier visit, and demotes it when it shrinks; see Patient-072). Reporting only the
first column would blame the ruler for both.

Expect us to read roughly one voxel high: a rasterised lesion genuinely is larger than the smooth
shape it came from (see docs/ruler_validation.html).

Usage:
    .venv/bin/python scripts/run_measurement.py
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rano.adapters.lumiere import paths  # noqa: E402
from rano.adapters.lumiere.zip_ref import ZipSource  # noqa: E402
from rano.measurement import measure_lesion  # noqa: E402

DEFAULT_ZIP = ROOT / "Imaging-v202211.zip"
COHORT_LOCK = ROOT / "output" / "cohort" / "cohort_lock.json"
RATINGS = ROOT / "LUMIERE-ExpertRating-v202211.csv"
OUT = ROOT / "output" / "measurement" / "vs_expert.json"

ENHANCING = 1
FLOOR = 20          # locked in docs/lesion_split_decision.html
MAX_LESIONS = 5     # measuring every speck is pointless; the expert names at most a handful

MEAS = re.compile(r"(\d+(?:\.\d+)?)\s*(?:mm)?\s*[x×X]\s*(\d+(?:\.\d+)?)\s*(?:mm)?")
FWD26 = tuple((dx, dy, dz) for dx in (0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
              if (dx, dy, dz) > (0, 0, 0))


def _shift(d: int):
    if d == 0:
        return slice(None), slice(None)
    return (slice(0, -d), slice(d, None)) if d > 0 else (slice(-d, None), slice(0, d))


def components(mask: np.ndarray, floor: int = FLOOR) -> list[np.ndarray]:
    """26-connected lesions above the floor, biggest first. The locked splitting rule."""
    n = int(mask.sum())
    if n == 0:
        return []
    idx = np.full(mask.shape, -1, np.int64)
    idx[mask] = np.arange(n)
    par = list(range(n))

    def find(x: int) -> int:
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    for off in FWD26:
        sl = [_shift(d) for d in off]
        a, b = idx[tuple(s[0] for s in sl)], idx[tuple(s[1] for s in sl)]
        m = (a >= 0) & (b >= 0)
        for u, v in zip(a[m].tolist(), b[m].tolist()):
            ru, rv = find(u), find(v)
            if ru != rv:
                par[max(ru, rv)] = min(ru, rv)

    roots = np.fromiter((find(i) for i in range(n)), np.int64, n)
    coords = np.argwhere(mask)
    groups = [coords[roots == r] for r in np.unique(roots) if (roots == r).sum() >= floor]
    groups.sort(key=len, reverse=True)
    return groups


#: Within this many mm on BOTH axes, our lesion is taken to be the one the radiologist measured.
#: There is no ground truth for WHICH lesion they meant -- only their numbers -- so this split is
#: partly circular: agreement defines "same lesion", and agreement is then reported. The
#: diagnoses for the far cases do not lean on it; they come from lesion counts and sizes.
SAME_LESION_MM = 5.0


def summarize(records: list[dict]) -> None:
    """The headline figures docs/progress.html quotes, printed so they can be regenerated."""
    ok = [r for r in records if r.get("status") == "ok"]
    err = {id(r): (r["best_match"]["long_mm"] - r["expert_long_mm"],
                   r["best_match"]["perp_mm"] - r["expert_short_mm"]) for r in ok}
    close = [r for r in ok if max(abs(e) for e in err[id(r)]) <= SAME_LESION_MM]
    far = [r for r in ok if max(abs(e) for e in err[id(r)]) > SAME_LESION_MM]
    print(f"\n{len(ok)} measurable timepoints: {len(close)} land within {SAME_LESION_MM:.0f} mm "
          f"on both axes, {len(far)} do not")

    if close:
        dl = np.array([err[id(r)][0] for r in close])
        ds = np.array([err[id(r)][1] for r in close])
        print("\nWHERE WE MEASURED THE SAME LESION THEY DID")
        print(f"  long axis      median {np.median(dl):+.1f} mm   range {dl.min():+.1f} to {dl.max():+.1f}")
        print(f"  perpendicular  median {np.median(ds):+.1f} mm   range {ds.min():+.1f} to {ds.max():+.1f}")

    print("\nWHERE WE DID NOT -- and why")
    for r in sorted(far, key=lambda r: -abs(err[id(r)][0])):
        if len(r["expert_all"]) > 1:
            why = f"they measured {len(r['expert_all'])} separate lesions; we found {r['n_lesions']}"
        elif not r["best_is_biggest"]:
            why = "they follow a lesion that is not our biggest"
        elif err[id(r)][0] > 8:
            why = "our lesion is far larger -- merged, or a different target"
        else:
            why = "our lesion is smaller than the one they measured"
        e = f"{r['expert_long_mm']:.0f}x{r['expert_short_mm']:.0f}"
        o = f"{r['best_match']['long_mm']:.1f}x{r['best_match']['perp_mm']:.1f}"
        print(f"  {r['patient']:<13} {r['timepoint']:<11} {e:>8} {o:>11}  {why}")

    agree = sum(1 for r in ok if r["best_match"]["measurable"]
                == (r["expert_long_mm"] >= 10 and r["expert_short_mm"] >= 10))
    ce = np.array([r["best_match"]["caliper_excess_mm"] for r in ok])
    print(f"\nMEASURABILITY GATE (>= 10 x 10 mm): agrees with their written numbers at {agree}/{len(ok)}")
    print(f"SHAPE WARNING (caliper minus constrained long axis): median {np.median(ce):.2f} mm, "
          f"{int((ce > 3).sum())}/{len(ce)} awkward enough to flag")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zip", default=str(DEFAULT_ZIP))
    ap.add_argument("--lock", default=str(COHORT_LOCK))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    src = ZipSource(a.zip)
    practice = [p["patient_id"] for p in
                json.loads(Path(a.lock).read_text())["practice"]["patients"]]
    rows = list(csv.DictReader(open(RATINGS)))
    rr = next(k for k in rows[0] if k.startswith("Rating rationale"))

    todo = [(r["Patient"], r["Date"], MEAS.findall(r[rr] or ""), (r[rr] or "").strip())
            for r in rows if r["Patient"] in practice and MEAS.search(r[rr] or "")]
    print(f"{len(todo)} practice timepoints carry a written mm x mm measurement\n")

    t0 = time.time()
    records = []
    print(f"  {'patient':<13} {'visit':<12} {'expert':>12} {'biggest lesion':>16} "
          f"{'best match':>16} {'err (best)':>13}")
    for patient, tp, pairs, text in todo:
        member = paths.dbt_mask(patient, tp)
        if not src.exists(member):
            print(f"  {patient:<13} {tp:<12} {'':>12}   (no segmentation in the archive)")
            records.append({"patient": patient, "timepoint": tp, "status": "no segmentation"})
            continue

        enh = np.asarray(src.open_nifti(member).dataobj) == ENHANCING
        comps = components(enh)[:MAX_LESIONS]
        ours = []
        for sel in comps:
            m = np.zeros(enh.shape, bool)
            m[sel[:, 0], sel[:, 1], sel[:, 2]] = True
            res = measure_lesion(m)
            if res is not None:
                ours.append({"volume_mm3": int(len(sel)),
                             "long_mm": res.measurement.long_mm,
                             "perp_mm": res.measurement.perp_mm,
                             "product_mm2": res.product_mm2,
                             "slice": res.slice_index,
                             "caliper_excess_mm": res.measurement.caliper_excess_mm,
                             "measurable": res.measurement.measurable})
        if not ours:
            records.append({"patient": patient, "timepoint": tp, "status": "no lesion above floor"})
            continue

        # the expert's largest written measurement, by product
        exp = max(((float(x), float(y)) for x, y in pairs), key=lambda p: p[0] * p[1])
        e_long, e_short = max(exp), min(exp)

        big = ours[0]
        best = min(ours, key=lambda o: abs(o["long_mm"] - e_long) + abs(o["perp_mm"] - e_short))
        records.append({
            "patient": patient, "timepoint": tp, "status": "ok", "rationale": text,
            "expert_all": [[float(x), float(y)] for x, y in pairs],
            "expert_long_mm": e_long, "expert_short_mm": e_short,
            "n_lesions": len(ours), "lesions": ours,
            "biggest": big, "best_match": best,
            "best_is_biggest": best is big,
        })
        fmt = lambda o: f"{o['long_mm']:.1f} x {o['perp_mm']:.1f}"      # noqa: E731
        err = (f"{best['long_mm'] - e_long:+.1f} / {best['perp_mm'] - e_short:+.1f}")
        flag = "" if best is big else "   <- not the biggest lesion"
        print(f"  {patient:<13} {tp:<12} {f'{e_long:.0f} x {e_short:.0f}':>12} "
              f"{fmt(big):>16} {fmt(best):>16} {err:>13}{flag}", flush=True)

    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(records, indent=1))
    print(f"\nwrote {p.relative_to(ROOT)}  ({time.time()-t0:.0f}s)")
    summarize(records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
