"""
Tests for the bidimensional ruler, against shapes whose answers are known in advance.

The perpendicular measurement is the reason this file exists. It went through two rewrites --
the first returned zero on every shape, because it looked for endpoints among convex-hull
vertices and a perpendicular chord almost never ends on one. Nothing about the numbers it
produced on real lesions looked wrong; they were simply plausible. A shape whose answer is known
before you measure it is the only thing that catches that class of mistake.

ON THE TOLERANCES
-----------------
A rasterised circle is genuinely bigger than the ideal circle it came from: the voxels near its
diagonal stick out past the smooth boundary, and treating each voxel as the 1 mm square it
actually is measures that. So curved shapes read about ONE VOXEL high and grid-aligned shapes
read exactly right, which is why ``square 20`` is asserted to 0.5 mm and the discs to 1.5 mm.
That is a property of the voxel grid, not slack in the ruler, and it is the same one-voxel
overshoot to expect against a radiologist's calipers.
"""

from __future__ import annotations

import numpy as np
import pytest

from rano.measurement import (
    fill_enclosed_holes,
    measure_lesion,
    measure_piece,
    measure_slice,
    pieces_2d,
)
from rano.measurement.shapes import SHAPES, annulus, crescent, disc, ell_shape, two_blobs

VOXEL = 1.5  # one voxel of rasterisation overshoot, plus a little
TIGHT = 0.5  # grid-aligned shapes have no rasterisation error at all


@pytest.mark.parametrize("name", [k for k, v in SHAPES.items() if v[1] is not None])
def test_known_shapes_measure_as_expected(name):
    mask, want_long, want_perp, _, fill = SHAPES[name]
    m, _, _ = measure_slice(mask, fill_holes=fill)
    tol = TIGHT if name.startswith("square") else VOXEL
    assert m.long_mm == pytest.approx(want_long, abs=tol), f"{name}: long {m.long_mm}"
    assert m.perp_mm == pytest.approx(want_perp, abs=tol), f"{name}: perp {m.perp_mm}"


def test_the_ruler_is_not_locked_to_the_voxel_axes():
    """The same ellipse turned must measure the same. A ruler that only searches along the grid
    would report the bounding box instead, and would shrink and grow as the shape rotates."""
    from rano.measurement.shapes import ellipse

    got = []
    for angle in (0, 30, 45, 72, 90):
        m, _, _ = measure_slice(ellipse(15, 6, angle))
        got.append((m.long_mm, m.perp_mm))
    longs = [g[0] for g in got]
    perps = [g[1] for g in got]
    assert max(longs) - min(longs) < 1.0, f"long axis wobbles with rotation: {longs}"
    assert max(perps) - min(perps) < 1.5, f"perpendicular wobbles with rotation: {perps}"


def test_a_line_that_would_leave_the_tissue_is_rejected():
    """On an L-shape the widest possible caliper runs from one arm tip to the other, straight
    across the empty corner. The measurement must refuse it and stay on the tissue."""
    m, _, _ = measure_slice(ell_shape(30, 8))
    assert m.caliper_mm > m.long_mm + 5, "the constraint did nothing on an L-shape"
    assert m.long_mm == pytest.approx(np.hypot(30, 8), abs=VOXEL)


def test_the_same_on_a_crescent():
    m, _, _ = measure_slice(crescent(20, 18, 8))
    assert m.caliper_mm > m.long_mm + 1, "the constraint did nothing on a crescent"


def test_a_ring_is_measured_through_its_middle():
    """A ring-enhancing lesion is measured outer edge to outer edge, so filling the enclosed
    hole must turn an annulus into its outer disc."""
    ring = annulus(25, 15)
    filled, _, _ = measure_slice(ring, fill_holes=True)
    assert filled.long_mm == pytest.approx(50.0, abs=VOXEL)
    assert filled.perp_mm == pytest.approx(50.0, abs=VOXEL)

    # forbidden to cross the hole, the longest line is a chord tangent to it
    unfilled, _, _ = measure_slice(ring, fill_holes=False)
    assert unfilled.long_mm == pytest.approx(2 * np.sqrt(25 ** 2 - 15 ** 2), abs=VOXEL)
    assert unfilled.long_mm < filled.long_mm - 5, "not filling should give a shorter line"
    assert unfilled.caliper_mm > unfilled.long_mm + 5, "the caliper should still cross the hole"


def test_a_ruler_never_spans_two_separate_pieces():
    """Two discs of radius 8 with a 10 mm gap. Measuring across both would read about 42 mm;
    the right answer is one disc, about 16."""
    m, _, n = measure_slice(two_blobs(8, 10))
    assert n == 2
    assert m.long_mm == pytest.approx(16.0, abs=VOXEL)
    assert m.long_mm < 25, "the ruler spanned the gap between the two pieces"


def test_fill_only_closes_holes_the_tissue_surrounds():
    ring = annulus(12, 6)
    assert fill_enclosed_holes(ring).sum() > ring.sum()

    open_c = ring.copy()          # cut the rim open: the fill must now run out of the opening
    n = ring.shape[0]
    open_c[n // 2 - 1:n // 2 + 2, n // 2:] = False
    assert fill_enclosed_holes(open_c).sum() == open_c.sum(), \
        "a cavity open along one side must not be filled"


def test_the_measurability_gate_trips_at_ten_millimetres():
    assert not measure_slice(disc(4.0))[0].measurable      # 8 mm across
    assert measure_slice(disc(5.5))[0].measurable          # 11 mm across


def test_product_and_area_are_consistent():
    m, _, _ = measure_slice(disc(10))
    assert m.product_mm2 == pytest.approx(m.long_mm * m.perp_mm)
    assert m.area_mm2 == pytest.approx(np.pi * 100, rel=0.05)


def test_pieces_are_returned_biggest_first():
    ps = pieces_2d(two_blobs(8, 10) | disc(3, two_blobs(8, 10).shape[0]))
    assert len(ps) >= 2
    assert all(len(ps[i]) >= len(ps[i + 1]) for i in range(len(ps) - 1))


def test_measuring_a_volume_picks_the_slice_with_the_biggest_product():
    """A stack of discs that grows then shrinks: the winning slice must be the fattest one."""
    vol = np.zeros((60, 60, 7), bool)
    for z, r in enumerate([4, 7, 12, 20, 12, 7, 4]):
        vol[:, :, z] = disc(r, 60)
    best = measure_lesion(vol)
    assert best is not None
    assert best.slice_index == 3
    assert best.measurement.long_mm == pytest.approx(40.0, abs=VOXEL)


def test_the_same_input_always_gives_the_same_answer():
    a, _, _ = measure_slice(crescent(20, 18, 8))
    b, _, _ = measure_slice(crescent(20, 18, 8))
    assert (a.long_mm, a.perp_mm) == (b.long_mm, b.perp_mm)


def test_an_empty_slice_measures_nothing():
    m, idx, n = measure_slice(np.zeros((20, 20), bool))
    assert m is None and idx == -1 and n == 0
    assert measure_lesion(np.zeros((20, 20, 5), bool)) is None
