"""
Connectivity x volume-floor sweep over the LUMIERE practice arm.

QUESTION
--------
Choosing how to split the enhancing mask into lesions needs two decisions -- the connectivity
(6 vs 26) and the volume floor -- and on one timepoint (Patient-067 week-109) the two turned out
to interact: the connectivities disagreed wildly at floor 0 (9 vs 16 components) and agreed
exactly from 5 mm3 upward, because everything they disagreed about was single-voxel debris.
This sweep asks whether that holds across the whole practice arm, so the decision rests on 93
timepoints instead of one.

It also measures three things the decision depends on:
  * how much enhancing volume a floor actually discards (it must be negligible),
  * how often a timepoint is genuinely multifocal after flooring,
  * how often the LARGEST lesion appears as >1 disjoint piece on its own axial slices --
    which is a statement about the 2D bidimensional measurement, not about lesion counting.

COHORT: practice arm only, patient list read from the cohort lock. Nothing here reads held-out.
SPACE:  atlas, MNI 1 mm isotropic, so 1 voxel == 1 mm3 exactly and counts are volumes.
This script computes an evidence table. It decides nothing and no pipeline code imports it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rano.adapters.lumiere import paths  # noqa: E402
from rano.adapters.lumiere.zip_ref import ZipSource  # noqa: E402

DEFAULT_ZIP = ROOT / "Imaging-v202211.zip"
COHORT_LOCK = ROOT / "output" / "cohort" / "cohort_lock.json"
OUT_DIR = ROOT / "output" / "connectivity"

ENHANCING = 1

#: candidate floors in mm3 (== voxels here). 0 is the no-floor control.
FLOORS = (0, 1, 2, 5, 10, 20, 33, 50, 100)

#: Half of each neighbourhood: unioning a->b covers b->a, so only "forward" offsets are needed.
FWD_26 = tuple(
    (dx, dy, dz)
    for dx in (0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
    if (dx, dy, dz) > (0, 0, 0)
)
FWD_6 = ((0, 0, 1), (0, 1, 0), (1, 0, 0))
FWD_8_INPLANE = ((0, 1), (1, -1), (1, 0), (1, 1))


def _slices(d: int) -> tuple[slice, slice]:
    """Aligned source/target slices for a shift of ``d`` along one axis."""
    if d == 0:
        return slice(None), slice(None)
    if d > 0:
        return slice(0, -d), slice(d, None)
    return slice(-d, None), slice(0, d)


class UnionFind:
    __slots__ = ("parent",)

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        p = self.parent
        while p[x] != x:
            p[x] = p[p[x]]  # path halving
            x = p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def component_sizes(mask: np.ndarray, offsets) -> np.ndarray:
    """Voxel count of each connected component, descending. Vectorised edge extraction."""
    n = int(mask.sum())
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    idx = np.full(mask.shape, -1, dtype=np.int64)
    idx[mask] = np.arange(n)

    uf = UnionFind(n)
    for off in offsets:
        sl = [_slices(d) for d in off]
        a = idx[tuple(s[0] for s in sl)]
        b = idx[tuple(s[1] for s in sl)]
        both = (a >= 0) & (b >= 0)
        if not both.any():
            continue
        for u, v in zip(a[both].tolist(), b[both].tolist()):
            uf.union(u, v)

    roots = np.fromiter((uf.find(i) for i in range(n)), dtype=np.int64, count=n)
    sizes = np.bincount(roots)
    return np.sort(sizes[sizes > 0])[::-1]


def pieces_in_slice(mask2d: np.ndarray) -> int:
    """Number of 8-connected pieces in one axial cross-section."""
    return len(component_sizes(mask2d[:, :, None], [(o[0], o[1], 0) for o in FWD_8_INPLANE]))


def analyse(enh: np.ndarray) -> dict:
    s26 = component_sizes(enh, FWD_26)
    s6 = component_sizes(enh, FWD_6)
    total = int(enh.sum())
    rec: dict = {
        "enhancing_mm3": total,
        "n_components_26": int(s26.size),
        "n_components_6": int(s6.size),
        "largest_mm3": int(s26[0]) if s26.size else 0,
        "largest_volume_fraction": round(float(s26[0]) / total, 4) if total else None,
        "counts_by_floor_26": {str(f): int((s26 >= f).sum()) for f in FLOORS},
        "counts_by_floor_6": {str(f): int((s6 >= f).sum()) for f in FLOORS},
        "volume_discarded_by_floor_26": {
            str(f): int(s26[s26 < f].sum()) for f in FLOORS
        },
    }

    # 2D fragmentation of the largest lesion, on the slices it occupies
    if s26.size:
        idx = np.full(enh.shape, -1, dtype=np.int64)
        n = int(enh.sum())
        idx[enh] = np.arange(n)
        uf = UnionFind(n)
        for off in FWD_26:
            sl = [_slices(d) for d in off]
            a, b = idx[tuple(s[0] for s in sl)], idx[tuple(s[1] for s in sl)]
            m = (a >= 0) & (b >= 0)
            for u, v in zip(a[m].tolist(), b[m].tolist()):
                uf.union(u, v)
        roots = np.fromiter((uf.find(i) for i in range(n)), dtype=np.int64, count=n)
        big_root = np.bincount(roots).argmax()
        big = np.zeros(enh.shape, dtype=bool)
        coords = np.argwhere(enh)
        sel = coords[roots == big_root]
        big[sel[:, 0], sel[:, 1], sel[:, 2]] = True
        zs = np.flatnonzero(big.any(axis=(0, 1)))
        frag = [int(z) for z in zs if pieces_in_slice(big[:, :, z : z + 1]) > 1]
        rec["largest_n_slices"] = int(zs.size)
        rec["largest_slices_fragmented"] = len(frag)
    else:
        rec["largest_n_slices"] = 0
        rec["largest_slices_fragmented"] = 0
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zip", default=str(DEFAULT_ZIP))
    ap.add_argument("--lock", default=str(COHORT_LOCK))
    ap.add_argument("--out", default=str(OUT_DIR / "connectivity_sweep.json"))
    a = ap.parse_args()

    src = ZipSource(a.zip)
    lock = json.loads(Path(a.lock).read_text())
    practice = [p["patient_id"] for p in lock["practice"]["patients"]]

    import re
    pat = re.compile(r"^Imaging/([^/]+)/([^/]+)/DeepBraTumIA-segmentation/atlas/"
                     r"segmentation/seg_mask\.nii\.gz$")
    cases: list[tuple[str, str]] = sorted(
        (m.group(1), m.group(2))
        for m in (pat.match(n) for n in src.names)
        if m and m.group(1) in practice
    )
    print(f"sweeping {len(cases)} practice timepoints across {len(practice)} patients\n")

    records = []
    t0 = time.time()
    for i, (patient, tp) in enumerate(cases, 1):
        enh = np.asarray(src.open_nifti(paths.dbt_mask(patient, tp)).dataobj) == ENHANCING
        rec = {"patient": patient, "timepoint": tp, **analyse(enh)}
        records.append(rec)
        print(f"  [{i:>2}/{len(cases)}] {patient} {tp:<12} "
              f"enh={rec['enhancing_mm3']:>7} mm3  "
              f"26-conn={rec['n_components_26']:>3}  6-conn={rec['n_components_6']:>3}  "
              f"largest={rec['largest_mm3']:>7}", flush=True)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "space": "atlas MNI 1mm isotropic (1 voxel == 1 mm3)",
        "compartment": "enhancing (DeepBraTumIA label 1)",
        "cohort": "practice arm only",
        "floors_mm3": list(FLOORS),
        "n_timepoints": len(records),
        "records": records,
    }, indent=1))
    print(f"\nwrote {out.relative_to(ROOT)}  ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
