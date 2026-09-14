#!/usr/bin/env python3
"""
Sweep the two numeric guards that the 2026-08-31 adjudication left as open numbers, and print
what each value costs and buys.

WHICH TWO, AND WHY THEY ARE SWEPT RATHER THAN CHOSEN
    ``min_absolute_change_mm3``   A ratio guard alone lets three voxels becoming twenty-eight
        read as "+833%, progression" (adjudication case 7). An absolute floor is the cheap half of
        the fix. It is NOT the measurability gate -- finding F3 proved no volume can express
        RANO's shape test -- so the question here is only "below what change is the number noise?"
    ``confirmation_max_weeks``    RANO writes a minimum interval and no maximum, so the code took
        the next scan whenever it arrived, including 42 and 50 weeks out (finding F2).

    Neither number is published. Both were guesses. This script replaces the guess with a curve.

WHAT IT WILL NOT DO
    Pick for you, and not touch the held-out arm. The 17 held-out patients of the locked cohort
    are EXCLUDED by default: choosing a constant by reading a curve is selection on data, and the
    held-out arm exists to be spent once, in September, on the finished rule. ``--cohort all``
    overrides that and says so in the header.

HOW TO READ THE OUTPUT
    ``agree`` is raw agreement; ``bal`` is mean per-class recall. They move in opposite directions
    on almost every honest change, so a value is only interesting when it moves both or holds one
    flat while lifting the other. ``majority`` is what answering PD on every scan would score --
    the bar, and it is high, because the cohort is mostly progression.

Usage:
    .venv/bin/python scripts/sweep_guards.py
    .venv/bin/python scripts/sweep_guards.py --cohort all --profile rano_classic_ported
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rano.criteria import PROFILES, compare_calls  # noqa: E402

import run_rano_calls as H  # noqa: E402

#: mm3. 1 mm isotropic atlas space, so these read directly as voxel counts. Spaced to show the
#: shape of the curve, not to hunt for a winner at three significant figures.
MM3_GRID = [0.0, 1.0, 3.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 524.0, 1000.0]

#: weeks. 4.0 is the profile's own minimum, so it is the degenerate "confirm only on the very next
#: scan if it is exactly at the minimum" end; None is the unlimited behaviour being replaced.
WEEKS_GRID = [6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0, 26.0, 52.0, None]

CLASSES = ("CR", "PR", "SD", "PD")


def score(trajectories, expert, arms, criteria):
    pairs, _ = H.run_profile(trajectories, expert, arms, criteria)
    return compare_calls(pairs, criteria.name)


def row(label: str, rep) -> str:
    recalls = "  ".join(f"{c} {rep.per_class[c].recall:5.0%}" for c in CLASSES)
    return (
        f"  {label:>10}   n {rep.n_compared:4d}   agree {rep.agreement:6.1%}   "
        f"bal {rep.balanced_accuracy:6.1%}   {recalls}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", type=Path, default=H.DEFAULT_ZIP)
    ap.add_argument("--ratings", type=Path, default=H.DEFAULT_RATINGS)
    ap.add_argument("--lock", type=Path, default=H.DEFAULT_LOCK)
    ap.add_argument("--profile", default="mrano_volumetric")
    ap.add_argument("--rt-end-week", type=float, default=H.RT_END_WEEK)
    ap.add_argument("--cohort", choices=("no-held-out", "all", "practice"), default="no-held-out")
    args = ap.parse_args()

    volumes = H.load_shipped_volumes(args.zip)
    expert = H.load_expert(args.ratings)
    arms = H.load_arms(args.lock)

    if args.cohort == "no-held-out":
        volumes = {k: v for k, v in volumes.items() if arms.get(k[0]) != "held_out"}
        cohort_note = "held-out arm EXCLUDED (17 patients reserved for September)"
    elif args.cohort == "practice":
        volumes = {k: v for k, v in volumes.items() if arms.get(k[0]) == "practice"}
        cohort_note = "practice arm only (7 patients)"
    else:
        cohort_note = "ALL patients -- this READS THE HELD-OUT ARM and spends it"

    trajectories = H.build_trajectories(volumes, expert, args.rt_end_week)
    base = PROFILES[args.profile]

    print("=" * 96)
    print(f"GUARD SWEEP  profile={base.name}  rt_end_week={args.rt_end_week:g}")
    print("=" * 96)
    print(f"cohort          {cohort_note}")
    print(f"patients        {len({p for p, _ in volumes})}")
    print(f"held fixed      pseudoprogression_policy={base.pseudoprogression_policy}, "
          f"confirmation_weeks={base.confirmation_weeks:g}")
    print()

    ref = score(trajectories, expert, arms, base)
    print(f"reference       {base.name} as shipped: min_absolute_change_mm3="
          f"{base.min_absolute_change_mm3:g}, confirmation_max_weeks={base.confirmation_max_weeks}")
    print(row("shipped", ref))
    print(f"  {'majority':>10}   answering PD on every scan scores {ref.majority_baseline:.1%}")
    print()

    print("-" * 96)
    print("A · min_absolute_change_mm3   (confirmation_max_weeks held at shipped value)")
    print("-" * 96)
    for v in MM3_GRID:
        print(row(f"{v:g} mm3", score(trajectories, expert, arms,
                                     base.variant(base.name, min_absolute_change_mm3=v))))
    print()

    print("-" * 96)
    print("B · confirmation_max_weeks   (min_absolute_change_mm3 held at shipped value)")
    print("-" * 96)
    for w in WEEKS_GRID:
        print(row("unlimited" if w is None else f"{w:g} wk",
                  score(trajectories, expert, arms,
                        base.variant(base.name, confirmation_max_weeks=w))))
    print()

    print("-" * 96)
    print("C · joint  (rows = mm3, cols = weeks; cell = agreement / balanced accuracy)")
    print("-" * 96)
    cols = [w for w in WEEKS_GRID if w in (8.0, 12.0, 16.0, 20.0, None)]
    head = "".join(("unlim" if w is None else f"{w:g}wk").rjust(15) for w in cols)
    print(f"  {'':>10}{head}")
    for v in MM3_GRID:
        cells = []
        for w in cols:
            r = score(trajectories, expert, arms,
                      base.variant(base.name, min_absolute_change_mm3=v, confirmation_max_weeks=w))
            cells.append(f"{r.agreement:.1%}/{r.balanced_accuracy:.1%}".rjust(15))
        print(f"  {v:g} mm3".ljust(12) + "".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
