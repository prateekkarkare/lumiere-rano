"""
Ingestion adapter contract — the two front doors reduce to one internal representation.

An ``Adapter`` is a *front door*: it resolves some heterogeneous source (a LUMIERE zip now, a
raw DICOM study later) into ``Patient`` objects that obey the ``contract.case`` types. Its one
hard boundary:

    An adapter may only DECODE a storage format. It may not PROCESS pixels.

Decoding a NIfTI member out of a zip, or reconstructing a 3-D volume + affine from a DICOM
series, is decoding — allowed. Resampling, skull-stripping, registration and segmentation are
processing — forbidden here; they are explicit downstream stages. Consequently a DICOM adapter
emits volumes that are native-geometry, not skull-stripped and mask-less: the validator will
*report* those deficits and the router will *route* the case into processing. That asymmetry
(LUMIERE arrives complete, a scanner study arrives raw) is the whole point of the seam — the
same validator serves both because neither the contract nor the validator bakes in "has a mask".
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

from rano.contract.case import Patient


class Adapter(ABC):
    """Resolve a source into ``Patient`` objects. Format-decode only — never transform pixels."""

    #: A short, stable name for provenance/logging, e.g. "lumiere-zip" or "dicom-fs".
    name: str = "adapter"

    @abstractmethod
    def patient_ids(self) -> list[str]:
        """All patient identifiers this source exposes, in a stable order."""

    @abstractmethod
    def load_patient(self, patient_id: str) -> Patient:
        """Build one fully-wired (but lazy) ``Patient``: timepoints sorted, ``ImageRef``s attached.

        Must not read voxel arrays — only whatever headers/metadata are needed to construct the
        ``Geometry`` and wire the refs. Raising is acceptable for a genuinely unresolvable patient
        (e.g. absent from the manifest); per-timepoint defects are data for the validator, not
        exceptions here.
        """

    def iter_patients(self) -> Iterator[Patient]:
        """Convenience: stream every patient. Override if a source can do this more efficiently."""
        for pid in self.patient_ids():
            yield self.load_patient(pid)


__all__ = ["Adapter"]
