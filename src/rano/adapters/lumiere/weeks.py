"""
Timepoint label parsing & chronological ordering for LUMIERE.

Labels look like ``week-044`` or ``week-000-2``. The trailing ``-N`` distinguishes multiple
scans within the same week (e.g. two baseline studies), so it is an ordering tie-breaker, not
a separate week. ``week-000-1`` sorts before ``week-000-2`` sorts before ``week-044``.

An unparseable label yields ``None``; such timepoints sort AFTER all parseable ones and carry
``week_offset is None`` in the contract (the longitudinal check flags them, never crashes).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_RE = re.compile(r"week-(\d+)(?:-(\d+))?$")


@dataclass(frozen=True, order=True)
class TimepointKey:
    """Sortable key: primary = week number, secondary = intra-week scan index (0 if absent)."""

    week: int
    sub: int


def parse(label: str) -> TimepointKey | None:
    """``"week-000-2"`` -> ``TimepointKey(0, 2)``; ``"week-044"`` -> ``TimepointKey(44, 0)``; bad -> None."""
    m = _RE.match(str(label).strip())
    if not m:
        return None
    return TimepointKey(int(m.group(1)), int(m.group(2)) if m.group(2) else 0)


def week_offset(label: str) -> float | None:
    """The single float the contract stores (weeks since first scan). ``None`` if unparseable."""
    k = parse(label)
    return float(k.week) if k is not None else None


def sort_key(label: str) -> tuple[int, TimepointKey | tuple[int, int]]:
    """Total order that places unparseable labels last, then by (week, sub)."""
    k = parse(label)
    if k is None:
        return (1, (0, 0))
    return (0, k)
