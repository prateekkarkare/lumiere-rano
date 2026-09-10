"""
Read the connectivity sweep JSON and print the decision tables.

Separate from the sweep so the expensive pass runs once and the presentation can be re-run
freely. Every number here comes from output/connectivity/connectivity_sweep.json.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON = ROOT / "output" / "connectivity" / "connectivity_sweep.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", default=str(DEFAULT_JSON))
    a = ap.parse_args()

    d = json.loads(Path(a.json).read_text())
    recs = d["records"]
    floors = [str(f) for f in d["floors_mm3"]]
    n = len(recs)
    empty = [r for r in recs if r["enhancing_mm3"] == 0]
    live = [r for r in recs if r["enhancing_mm3"] > 0]

    print(f"{n} practice timepoints  ({len(empty)} with zero enhancing voxels, "
          f"{len(live)} with disease)")
    print(f"total enhancing volume across the arm: "
          f"{sum(r['enhancing_mm3'] for r in recs):,} mm3\n")

    print("=" * 74)
    print("1. DO 26- AND 6-CONNECTIVITY AGREE ON LESION COUNT?   (n = %d with disease)" % len(live))
    print("=" * 74)
    print(f"  {'floor mm3':>9} | {'agree':>12} | {'26>6':>5} | {'6>26':>5} | {'worst gap':>9}")
    for f in floors:
        agree = sum(1 for r in live if r["counts_by_floor_26"][f] == r["counts_by_floor_6"][f])
        hi = sum(1 for r in live if r["counts_by_floor_26"][f] > r["counts_by_floor_6"][f])
        lo = sum(1 for r in live if r["counts_by_floor_26"][f] < r["counts_by_floor_6"][f])
        gap = max(abs(r["counts_by_floor_26"][f] - r["counts_by_floor_6"][f]) for r in live)
        pct = 100 * agree / len(live)
        print(f"  {f:>9} | {agree:>4}/{len(live):<3} {pct:>5.1f}% | {hi:>5} | {lo:>5} | {gap:>9}")

    print("\n" + "=" * 74)
    print("2. WHAT DOES THE FLOOR THROW AWAY?   (26-connectivity)")
    print("=" * 74)
    print(f"  {'floor mm3':>9} | {'total discarded':>15} | {'% of arm':>8} | "
          f"{'worst single tp':>15} | {'worst tp %':>10}")
    total_vol = sum(r["enhancing_mm3"] for r in recs)
    for f in floors:
        disc = sum(r["volume_discarded_by_floor_26"][f] for r in recs)
        worst = max(live, key=lambda r: r["volume_discarded_by_floor_26"][f] / r["enhancing_mm3"])
        wpct = 100 * worst["volume_discarded_by_floor_26"][f] / worst["enhancing_mm3"]
        print(f"  {f:>9} | {disc:>15,} | {100*disc/total_vol:>7.3f}% | "
              f"{worst['volume_discarded_by_floor_26'][f]:>15,} | {wpct:>9.1f}%")

    print("\n" + "=" * 74)
    print("3. HOW MULTIFOCAL IS THE ARM?   (26-connectivity, timepoints with disease)")
    print("=" * 74)
    print(f"  {'floor mm3':>9} | {'1 lesion':>9} | {'2':>4} | {'3':>4} | {'4-5':>4} | "
          f"{'6+':>4} | {'median':>6} | {'max':>4}")
    for f in floors:
        c = [r["counts_by_floor_26"][f] for r in live]
        cnt = Counter(c)
        b45 = sum(v for k, v in cnt.items() if k in (4, 5))
        b6 = sum(v for k, v in cnt.items() if k >= 6)
        med = sorted(c)[len(c) // 2]
        print(f"  {f:>9} | {cnt.get(1,0):>9} | {cnt.get(2,0):>4} | {cnt.get(3,0):>4} | "
              f"{b45:>4} | {b6:>4} | {med:>6} | {max(c):>4}")

    print("\n" + "=" * 74)
    print("4. DOES ONE LESION DOMINATE?   (share of enhancing volume in the largest component)")
    print("=" * 74)
    fr = sorted(r["largest_volume_fraction"] for r in live)
    q = lambda p: fr[min(int(p * len(fr)), len(fr) - 1)]  # noqa: E731
    print(f"  min {fr[0]:.1%}   p10 {q(.10):.1%}   median {q(.50):.1%}   "
          f"p90 {q(.90):.1%}   max {fr[-1]:.1%}")
    print(f"  timepoints where the largest lesion holds >=90% of enhancing volume: "
          f"{sum(1 for v in fr if v >= .90)}/{len(fr)}")

    print("\n" + "=" * 74)
    print("5. 2D FRAGMENTATION OF THE LARGEST LESION   (bears on the bidimensional measurement)")
    print("=" * 74)
    withsl = [r for r in live if r["largest_n_slices"] > 0]
    tot_sl = sum(r["largest_n_slices"] for r in withsl)
    tot_fr = sum(r["largest_slices_fragmented"] for r in withsl)
    allfrag = [r for r in withsl if r["largest_slices_fragmented"] == r["largest_n_slices"]]
    nofrag = [r for r in withsl if r["largest_slices_fragmented"] == 0]
    print(f"  axial slices occupied by the largest lesion, summed: {tot_sl:,}")
    print(f"  of those, slices where it appears as >1 disjoint piece: {tot_fr:,} "
          f"({100*tot_fr/tot_sl:.1f}%)")
    print(f"  timepoints where EVERY such slice is fragmented: {len(allfrag)}/{len(withsl)}")
    print(f"  timepoints where NO slice is fragmented:         {len(nofrag)}/{len(withsl)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
