"""
Synthetic shapes with known answers, for testing the ruler.

A voxel (i, j) occupies the square [i, i+1] x [j, j+1], so its CENTRE is at (i+0.5, j+0.5).
Every shape here is rasterised by testing the centre, which is what makes the analytic answers
below actually apply -- testing the index instead shifts every shape by half a voxel and
silently biases all of them the same way.

These live in the package rather than in the test file because the validation page renders them
too, and a drawing of a different shape from the one under test would be worse than no drawing.
"""

from __future__ import annotations

import numpy as np

__all__ = ["disc", "ellipse", "square", "rectangle", "annulus", "crescent", "ell_shape",
           "two_blobs", "SHAPES"]


def _grid(n: int) -> tuple[np.ndarray, np.ndarray]:
    i, j = np.indices((n, n))
    return i + 0.5, j + 0.5


def disc(r: float, n: int | None = None) -> np.ndarray:
    n = n or int(2 * r + 8)
    x, y = _grid(n)
    c = n / 2.0
    return ((x - c) ** 2 + (y - c) ** 2) < r * r


def ellipse(a: float, b: float, angle_deg: float = 0.0, n: int | None = None) -> np.ndarray:
    n = n or int(2 * max(a, b) + 8)
    x, y = _grid(n)
    c = n / 2.0
    t = np.radians(angle_deg)
    u = (x - c) * np.cos(t) + (y - c) * np.sin(t)
    v = -(x - c) * np.sin(t) + (y - c) * np.cos(t)
    return ((u / a) ** 2 + (v / b) ** 2) < 1.0


def square(s: int, pad: int = 4) -> np.ndarray:
    m = np.zeros((s + 2 * pad, s + 2 * pad), bool)
    m[pad:pad + s, pad:pad + s] = True
    return m


def rectangle(w: int, h: int, pad: int = 4) -> np.ndarray:
    m = np.zeros((w + 2 * pad, h + 2 * pad), bool)
    m[pad:pad + w, pad:pad + h] = True
    return m


def annulus(outer: float, inner: float, n: int | None = None) -> np.ndarray:
    n = n or int(2 * outer + 8)
    return disc(outer, n) & ~disc(inner, n)


def crescent(r: float, bite_r: float, offset: float, n: int | None = None) -> np.ndarray:
    """A disc with a bite taken out by a second, offset disc. Concave, and encloses nothing --
    so hole-filling cannot rescue it and the inside-the-tissue constraint has to do the work."""
    n = n or int(2 * r + 8)
    x, y = _grid(n)
    c = n / 2.0
    big = ((x - c) ** 2 + (y - c) ** 2) < r * r
    bite = ((x - c - offset) ** 2 + (y - c) ** 2) < bite_r * bite_r
    return big & ~bite


def ell_shape(arm: int, thick: int, pad: int = 4) -> np.ndarray:
    m = np.zeros((arm + 2 * pad, arm + 2 * pad), bool)
    m[pad:pad + thick, pad:pad + arm] = True
    m[pad:pad + arm, pad:pad + thick] = True
    return m


def two_blobs(r: float, gap: float) -> np.ndarray:
    """Two discs with clear space between them. A ruler must never span the gap."""
    n = int(4 * r + gap + 12)
    x, y = _grid(n)
    cy = n / 2.0
    c1, c2 = n / 2.0 - r - gap / 2.0, n / 2.0 + r + gap / 2.0
    return (((x - c1) ** 2 + (y - cy) ** 2) < r * r) | (((x - c2) ** 2 + (y - cy) ** 2) < r * r)


#: A ring that may not be crossed has a known answer too. The longest line that stays on the rim
#: is a chord tangent to the hole, 2*sqrt(R^2 - r^2) long; the longest line square to it is the
#: matching tangent chord on the neighbouring side, the same length by symmetry, and the two
#: cross at distance r*sqrt(2) from the centre -- inside the rim whenever r*sqrt(2) < R.
_RING_R, _RING_r = 25.0, 15.0
_RING_TANGENT_CHORD = 2.0 * np.sqrt(_RING_R ** 2 - _RING_r ** 2)

#: name -> (mask, expected long, expected perp, note, fill_holes). ``None`` for an expected value
#: means there is no clean analytic answer and the test checks a property instead.
SHAPES: dict[str, tuple] = {
    "disc r=10": (disc(10), 20.0, 20.0, "diameter both ways", True),
    "disc r=20": (disc(20), 40.0, 40.0, "diameter both ways", True),
    "square 20": (square(20), 20.0 * np.sqrt(2), 20.0 * np.sqrt(2), "both diagonals", True),
    "ellipse 15x6": (ellipse(15, 6), 30.0, 12.0, "major and minor axes", True),
    "ellipse 15x6 @30deg": (ellipse(15, 6, 30), 30.0, 12.0,
                            "same, turned -- must not be axis-locked", True),
    "ellipse 15x6 @72deg": (ellipse(15, 6, 72), 30.0, 12.0, "same, turned further", True),
    "annulus R25 r15": (annulus(25, 15), 50.0, 50.0,
                        "hole filled first, so it measures as its outer disc", True),
    "annulus R25 r15, unfilled": (annulus(25, 15), _RING_TANGENT_CHORD, _RING_TANGENT_CHORD,
                                  "forbidden to cross the hole: both lines become chords "
                                  "tangent to it, 2*sqrt(R^2-r^2) long", False),
    # bite parameters chosen so the hull's longest line genuinely leaves the tissue -- a shallow
    # bite leaves the full diameter intact and tests nothing.
    "crescent": (crescent(20, 18, 8), None, None, "concave: the caliper must overshoot", True),
    "L-shape": (ell_shape(30, 8), None, None, "concave in the other direction", True),
    "two blobs": (two_blobs(8, 10), 16.0, 16.0,
                  "two pieces: measure one, never across the gap", True),
}
