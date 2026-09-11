"""
Bidimensional measurement — the mm x mm product RANO is written in.

WHY THIS EXISTS
---------------
This project reports volume, which is the better measurement. But the expert calls we grade
ourselves against were made with a ruler on a screen (2D Macdonald), and RANO's measurability
gate -- a lesion counts as a target only at 10 mm by 10 mm -- is stated in millimetres, not
cubic millimetres. No volume threshold can answer it. So the ruler is needed twice over: to
read the answer key, and to decide which lesions are allowed into the answer at all.

WHAT IS MEASURED
----------------
On each axial slice of a lesion: the longest straight line across it, then the longest line at
right angles to that one. Multiply. The slice with the biggest product wins.

FOUR DECISIONS, EACH LOAD-BEARING
---------------------------------
1. **Both lines stay inside the tissue.** A caliper you could not physically draw on the image
   is not a measurement. For a convex cross-section this changes nothing; on a crescent it is
   the difference between measuring the lesion and measuring the gap beside it. The
   unconstrained caliper is reported too, as ``caliper_mm`` -- when it and ``long_mm`` diverge,
   the cross-section is an awkward shape and the measurement deserves suspicion.

2. **One contiguous piece at a time.** Half the slices of a typical lesion here show it as two
   or more disjoint islands -- it is joined through neighbouring slices, not within this one. A
   ruler laid across two islands measures the healthy brain between them. Each piece is measured
   alone and the best product wins.

3. **Enclosed holes are filled first.** A ring-enhancing tumour is measured outer edge to outer
   edge, through its dead centre, the way the size of an apple includes its core. Only holes the
   tissue completely surrounds are filled -- a surgical cavity open along one side never fills,
   because the fill runs out of the opening. That distinction is why this is safe and why simply
   adding in the necrosis label would not be: in this cohort label 2 holds resection cavities
   many times larger than the tumour around them.

4. **Voxels are squares, not points.** Lengths are measured between voxel CORNERS, so a run of
   ten 1 mm voxels measures 10.0 mm rather than the 9.0 mm you get centre-to-centre. On a 10 mm
   threshold that one millimetre decides whether a lesion is measurable at all.

Geometry is in voxel units. On the 1 mm isotropic atlas grid one voxel is one millimetre; on any
other grid the caller must convert.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

__all__ = [
    "SliceMeasurement", "LesionMeasurement",
    "measure_piece", "measure_slice", "measure_lesion",
    "fill_enclosed_holes", "pieces_2d", "convex_hull",
]


@dataclass(frozen=True, slots=True)
class SliceMeasurement:
    """One measurement on one contiguous piece of one slice."""

    long_mm: float
    perp_mm: float
    long_endpoints: tuple[tuple[float, float], tuple[float, float]]
    perp_endpoints: tuple[tuple[float, float], tuple[float, float]]
    caliper_mm: float
    caliper_endpoints: tuple[tuple[float, float], tuple[float, float]]
    area_mm2: float
    filled_mm2: float
    """Area after enclosed holes were filled -- compare with ``area_mm2`` to see what filling did."""

    @property
    def product_mm2(self) -> float:
        return self.long_mm * self.perp_mm

    @property
    def measurable(self) -> bool:
        """RANO's target-lesion gate: at least 10 mm in both directions."""
        return self.long_mm >= 10.0 and self.perp_mm >= 10.0

    @property
    def caliper_excess_mm(self) -> float:
        """How much longer an unconstrained caliper would read. Large means an odd shape."""
        return self.caliper_mm - self.long_mm


@dataclass(frozen=True, slots=True)
class LesionMeasurement:
    """The winning slice for a whole lesion."""

    slice_index: int
    piece_index: int
    n_pieces_on_slice: int
    measurement: SliceMeasurement

    @property
    def product_mm2(self) -> float:
        return self.measurement.product_mm2


# --------------------------------------------------------------------------- geometry

def convex_hull(points: np.ndarray) -> np.ndarray:
    """Monotone-chain hull. Returns the vertices in order."""
    pts = sorted({(float(a), float(b)) for a, b in points})
    if len(pts) <= 2:
        return np.array(pts, dtype=float)

    def half(seq):
        out: list[tuple[float, float]] = []
        for p in seq:
            while len(out) >= 2:
                (ax, ay), (bx, by) = out[-2], out[-1]
                if (bx - ax) * (p[1] - ay) - (by - ay) * (p[0] - ax) <= 0:
                    out.pop()
                else:
                    break
            out.append(p)
        return out

    return np.array(half(pts)[:-1] + half(pts[::-1])[:-1], dtype=float)


def _corners(pts: np.ndarray) -> np.ndarray:
    """The four corners of every voxel -- see decision 4 in the module docstring."""
    return np.concatenate([pts, pts + [1, 0], pts + [0, 1], pts + [1, 1]]).astype(float)


def pieces_2d(mask: np.ndarray) -> list[np.ndarray]:
    """Split a slice into 8-connected pieces. A ruler may never span two of them."""
    seen = np.zeros_like(mask, dtype=bool)
    nbrs = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]
    out = []
    for seed in map(tuple, np.argwhere(mask)):
        if seen[seed]:
            continue
        cur, q = [], deque([seed])
        seen[seed] = True
        while q:
            i, j = q.popleft()
            cur.append((i, j))
            for di, dj in nbrs:
                x, y = i + di, j + dj
                if 0 <= x < mask.shape[0] and 0 <= y < mask.shape[1] \
                        and mask[x, y] and not seen[x, y]:
                    seen[x, y] = True
                    q.append((x, y))
        out.append(np.array(cur))
    out.sort(key=len, reverse=True)
    return out


def fill_enclosed_holes(mask: np.ndarray) -> np.ndarray:
    """Fill only what the tissue completely surrounds.

    Flood the background inward from the border; whatever the flood cannot reach is enclosed.
    A cavity open along one side stays unfilled because the flood gets in -- which is exactly
    the property that makes this safe on post-operative scans.
    """
    h, w = mask.shape
    free = ~mask
    seen = np.zeros_like(free)
    q: deque = deque()
    for i in range(h):
        for j in (0, w - 1):
            if free[i, j] and not seen[i, j]:
                seen[i, j] = True
                q.append((i, j))
    for j in range(w):
        for i in (0, h - 1):
            if free[i, j] and not seen[i, j]:
                seen[i, j] = True
                q.append((i, j))
    while q:
        i, j = q.popleft()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if 0 <= a < h and 0 <= b < w and free[a, b] and not seen[a, b]:
                seen[a, b] = True
                q.append((a, b))
    return mask | (free & ~seen)


def _boundary(occ: np.ndarray) -> np.ndarray:
    """Voxels with at least one face on background. Only these can be a chord's endpoint."""
    b = np.zeros_like(occ)
    b[:-1, :] |= occ[:-1, :] & ~occ[1:, :]
    b[1:, :] |= occ[1:, :] & ~occ[:-1, :]
    b[:, :-1] |= occ[:, :-1] & ~occ[:, 1:]
    b[:, 1:] |= occ[:, 1:] & ~occ[:, :-1]
    b[0, :] |= occ[0, :]
    b[-1, :] |= occ[-1, :]
    b[:, 0] |= occ[:, 0]
    b[:, -1] |= occ[:, -1]
    return np.argwhere(b)


def _on_tissue(occ: np.ndarray, a: np.ndarray, b: np.ndarray, step: float = 0.25) -> bool:
    """Does the straight segment a->b stay on tissue the whole way?

    Endpoints are voxel CORNERS, which sit exactly on the boundary, so both ends are pulled half
    a voxel toward the middle before sampling. Without that, a chord running along a flat edge
    rounds outward at its ends and is wrongly rejected.
    """
    v = b - a
    L = float(np.hypot(*v))
    if L < 1e-9:
        return True
    u = v / L
    a2, b2 = a + u * 0.5, b - u * 0.5
    n = max(int(L / step), 2)
    t = np.linspace(0.0, 1.0, n)
    xs = np.floor(a2[0] + t * (b2[0] - a2[0])).astype(int)
    ys = np.floor(a2[1] + t * (b2[1] - a2[1])).astype(int)
    ok = (xs >= 0) & (xs < occ.shape[0]) & (ys >= 0) & (ys < occ.shape[1])
    return bool(ok.all() and occ[xs, ys].all())


def _longest_inside(occ: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Longest chord that never leaves the tissue. Candidates first, longest tested first."""
    bnd = _boundary(occ)
    if len(bnd) == 0:
        return 0.0, np.zeros(2), np.zeros(2)
    cand = np.unique(_corners(bnd), axis=0)
    d2 = ((cand[:, None, :] - cand[None, :, :]) ** 2).sum(2)
    iu = np.triu_indices(len(cand), 1)
    order = np.argsort(d2[iu])[::-1]
    for k in order:
        i, j = iu[0][k], iu[1][k]
        if _on_tissue(occ, cand[i], cand[j]):
            return float(np.sqrt(d2[i, j])), cand[i], cand[j]
    return 0.0, cand[0], cand[0]


def _longest_perpendicular(occ: np.ndarray, p1: np.ndarray, p2: np.ndarray,
                           step: float = 0.25) -> tuple[float, np.ndarray, np.ndarray]:
    """Longest unbroken run of tissue at right angles to p1->p2.

    Swept rather than searched: slide a line across the shape square to the long axis and, at
    each position, take the longest UNBROKEN stretch sitting on tissue. A C-shape puts tissue,
    gap, tissue on one such line -- the gap ends the measurement rather than being spanned.

    The run must also CROSS the long axis. Without that condition a ring-shaped piece answers
    with a long chord taken from the far side of the ring, square to the long axis but nowhere
    near it -- geometrically valid, and not a width anyone would draw. For a convex piece the
    condition costs nothing, because there every perpendicular chord crosses the axis anyway.
    """
    v = p2 - p1
    L = float(np.hypot(*v))
    if L < 1e-9:
        return 0.0, p1, p1
    u = v / L
    w = np.array([-u[1], u[0]])

    pts = np.argwhere(occ).astype(float)
    corners = _corners(pts)
    tu, tw = corners @ u, corners @ w
    ss = np.arange(tw.min(), tw.max() + step, step)
    best, ends = 0.0, (p1, p1)
    for t in np.arange(tu.min(), tu.max(), 0.5):
        px = t * u[0] + ss * w[0]
        py = t * u[1] + ss * w[1]
        xs, ys = np.floor(px).astype(int), np.floor(py).astype(int)
        ok = (xs >= 0) & (xs < occ.shape[0]) & (ys >= 0) & (ys < occ.shape[1])
        ins = np.zeros(ss.size, np.int8)
        ins[ok] = occ[xs[ok], ys[ok]]
        d = np.diff(np.concatenate(([0], ins, [0])))
        st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        if not st.size:
            continue
        axis_w = float(p1 @ w)                      # the long axis sits at this w, for every t
        for a_, b_ in zip(st, en):
            s0, s1 = float(ss[a_]), float(ss[min(b_, ss.size - 1)])
            if not (s0 - step <= axis_w <= s1 + step):
                continue                            # this run does not cross the long axis
            run = float(b_ - a_) * step
            if run > best:
                best = run
                ends = (t * u + s0 * w, t * u + s1 * w)
    return best, ends[0], ends[1]


def measure_piece(pts: np.ndarray, fill_holes: bool = True) -> SliceMeasurement:
    """Measure one contiguous piece of one slice."""
    lo = pts.min(axis=0)
    occ = np.zeros(tuple(pts.max(axis=0) - lo + 1), dtype=bool)
    occ[pts[:, 0] - lo[0], pts[:, 1] - lo[1]] = True
    if fill_holes:
        occ = fill_enclosed_holes(occ)

    raw_area = float(len(pts))
    hull = convex_hull(_corners(np.argwhere(occ).astype(float)))
    caliper, ca, cb = 0.0, np.zeros(2), np.zeros(2)
    if len(hull) >= 2:
        d2 = ((hull[:, None, :] - hull[None, :, :]) ** 2).sum(2)
        i, j = np.unravel_index(d2.argmax(), d2.shape)
        caliper, ca, cb = float(np.sqrt(d2[i, j])), hull[i], hull[j]

    long_mm, a, b = _longest_inside(occ)
    perp_mm, pa, pb = _longest_perpendicular(occ, a, b)
    off = lambda q: (float(q[0] + lo[0]), float(q[1] + lo[1]))  # noqa: E731
    return SliceMeasurement(
        long_mm=long_mm, perp_mm=perp_mm,
        long_endpoints=(off(a), off(b)),
        perp_endpoints=(off(pa), off(pb)),
        caliper_mm=caliper, caliper_endpoints=(off(ca), off(cb)),
        area_mm2=raw_area, filled_mm2=float(occ.sum()),
    )


def measure_slice(mask2d: np.ndarray, fill_holes: bool = True
                  ) -> tuple[SliceMeasurement | None, int, int]:
    """Best product over every contiguous piece on this slice."""
    ps = pieces_2d(mask2d)
    if not ps:
        return None, -1, 0
    best, idx = None, -1
    for k, pts in enumerate(ps):
        m = measure_piece(pts, fill_holes)
        if best is None or m.product_mm2 > best.product_mm2:
            best, idx = m, k
    return best, idx, len(ps)


def measure_lesion(mask3d: np.ndarray, fill_holes: bool = True) -> LesionMeasurement | None:
    """Best product over every axial slice the lesion occupies. Axis 2 is axial."""
    best: LesionMeasurement | None = None
    for z in np.flatnonzero(mask3d.any(axis=(0, 1))):
        m, piece, n = measure_slice(mask3d[:, :, z], fill_holes)
        if m is not None and (best is None or m.product_mm2 > best.product_mm2):
            best = LesionMeasurement(int(z), piece, n, m)
    return best
