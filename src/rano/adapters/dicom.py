"""
DICOM adapter — SEAM ONLY (Piece 1 builds interfaces; the impl is a later piece).

This exists now so the two-front-door design is *proven*, not just asserted: ``DicomAdapter`` is a
real subclass of ``Adapter``, so the abstract contract is demonstrably satisfiable by something
other than LUMIERE. Every method is a stub. When it is built, its job is DECODE ONLY — assemble a
3-D volume + affine from a DICOM series and wire native-space ``ImageRef``s. It will emit no mask
and no skull-strip; the validator reports those deficits and the router sends the case through the
skull-strip / segmentation stages.
"""

from __future__ import annotations

from rano.adapters.base import Adapter
from rano.contract.case import Patient


class DicomAdapter(Adapter):
    """Raw-DICOM front door. Not yet implemented — present to keep the seam honest."""

    name = "dicom-fs"

    def __init__(self, root: str) -> None:
        self.root = root

    def patient_ids(self) -> list[str]:
        raise NotImplementedError("DICOM ingestion is a later piece; the seam is defined, not built.")

    def load_patient(self, patient_id: str) -> Patient:
        raise NotImplementedError("DICOM ingestion is a later piece; the seam is defined, not built.")


__all__ = ["DicomAdapter"]
