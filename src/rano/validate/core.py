"""
Validator core — the one result shape every check in ``validate/checks/`` returns.

Per the C4 component map, each check is a pure function; a runner will later wrap the whole
set so one bad check (an exception) becomes a "fail" result instead of crashing the run. That
registry/runner isn't built yet — this module holds only the result type, because right now
there's exactly one check (``mask_grid_alignment``) and nothing to register or run in a loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping

Status = Literal["pass", "warn", "fail"]


@dataclass(frozen=True)
class CheckResult:
    """The outcome of one check against one piece of the case contract.

    ``code`` is a short, stable, machine-matchable identifier (dotted, e.g.
    ``"mask_grid_alignment.shape_mismatch"``); ``detail`` is the human-readable reason;
    ``evidence`` carries the numbers behind the verdict (deltas, tolerances used, ...).
    """

    status: Status
    code: str
    detail: str
    evidence: Mapping[str, object] = field(default_factory=dict)


__all__ = ["CheckResult", "Status"]
