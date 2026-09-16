"""
From a mask to a list of lesions, and from lesions to "is any of this measurable?".

THE LOCKED SPLITTING RULE (docs/lesion_split_decision.html)
    Two voxels belong to the same lesion if they touch at a face, an edge or a corner
    (26-connectivity), and a component under ``FLOOR_MM3`` is dropped as noise. 6-connectivity
    shatters thin enhancing rims into fragments that each fall under the floor and vanish.

MEASURABLE IS A SHAPE TEST, NOT A SIZE TEST
    RANO admits a lesion as a measurable target when it spans >= 10 mm in two perpendicular
    directions on one slice -- so no volume threshold can stand in for it. On real LUMIERE masks a
    956 mm3 lesion measures 17.2 x 9.0 mm and fails, while a 691 mm3 blob can pass. The floor above
    is a noise floor and nothing more.

    Volumetry NEVER applies the floor; it counts every voxel. An empty lesion inventory means "no
    measurable target lesion", never "no disease" and never a complete response.
"""

from __future__ import annotations

import numpy as np

from rano.measurement.ruler import LesionMeasurement, measure_lesion

#: Locked: components smaller than this are noise, not lesions.
FLOOR_MM3 = 20

_FWD26 = tuple(
    (dx, dy, dz)
    for dx in (0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) > (0, 0, 0)
)


def _shift(d: int):
    if d == 0:
        return slice(None), slice(None)
    return (slice(0, -d), slice(d, None)) if d > 0 else (slice(-d, None), slice(0, d))


def components(mask: np.ndarray, floor: int = FLOOR_MM3) -> list[np.ndarray]:
    """26-connected lesions at or above ``floor`` voxels, biggest first, as coordinate arrays."""
    mask = np.asarray(mask, dtype=bool)
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

    for off in _FWD26:
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


def measure_lesions(mask: np.ndarray, limit: int | None = None) -> list[LesionMeasurement]:
    """Ruler measurement of each lesion in ``mask``, biggest lesion first.

    ``limit`` measures only the largest N -- measuring every speck answers no question anyone asks.
    """
    out: list[LesionMeasurement] = []
    for sel in components(mask)[:limit]:
        one = np.zeros(mask.shape, bool)
        one[tuple(sel.T)] = True
        measured = measure_lesion(one)
        if measured is not None:
            out.append(measured)
    return out


def has_measurable_disease(mask: np.ndarray, limit: int | None = 5) -> bool:
    """True when at least one lesion clears RANO's 10 x 10 mm target-lesion gate.

    The largest lesion is not necessarily the measurable one -- a big pancake fails the gate while a
    smaller round lesion passes -- so several are measured, largest first.
    """
    return any(m.measurement.measurable for m in measure_lesions(mask, limit=limit))


__all__ = ["FLOOR_MM3", "components", "measure_lesions", "has_measurable_disease"]
