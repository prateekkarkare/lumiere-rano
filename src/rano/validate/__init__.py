"""Validator: pure checks over the case contract, each returning a CheckResult."""

from rano.validate.checks.mask_grid_alignment import check_mask_grid_alignment
from rano.validate.core import CheckResult

__all__ = ["CheckResult", "check_mask_grid_alignment"]
