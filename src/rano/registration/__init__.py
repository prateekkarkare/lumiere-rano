"""Piece 2 — putting a patient's visits into the same frame as each other.

Atlas space gives every visit the same grid; it does not give them the same anatomy in the same
voxels. See ``rigid`` for the evidence and the direction convention.
"""

from rano.registration.rigid import (
    RigidFit,
    dilate,
    fit_rigid,
    registration_mask,
    resample_like_fixed,
)

__all__ = ["RigidFit", "fit_rigid", "registration_mask", "resample_like_fixed", "dilate"]
