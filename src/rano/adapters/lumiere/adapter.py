"""
LUMIERE ingestion adapter — the first front door.

Enumeration is driven by ``LUMIERE-datacompleteness.csv`` (the declared ground-truth manifest):
patients and timepoints come from its rows, in file order, timepoints sorted chronologically.
For every modality/mask the manifest marks present, the adapter attaches a lazy ``ZipNiftiRef``
IF the member actually resolves in the archive. A member that the manifest claims but the zip
lacks is dropped from the case (so the contract reflects what is truly loadable) and recorded by
``audit_manifest_vs_zip`` — the discrepancy is data, surfaced explicitly, never a silent guess.

Spaces (per the locked per-image-space design):
  * raw CT1/T1/T2/FLAIR  -> NATIVE, each its own space key (never presumed co-registered);
  * DeepBraTumIA mask, its skull-strip reference image, and brain mask -> MNI 1mm (shared key).

This adapter only DECODES: it wires refs and reads headers as needed. It never transforms pixels.
"""

from __future__ import annotations

import csv
from collections import OrderedDict

from rano.adapters.base import Adapter
from rano.adapters.lumiere import paths, weeks
from rano.adapters.lumiere.zip_ref import ZipNiftiRef, ZipSource
from rano.contract.case import MaskSource, Modality, Patient, Space, Timepoint

_MODALITY_COLS = {
    "CT1": Modality.CT1,
    "T1": Modality.T1,
    "T2": Modality.T2,
    "FLAIR": Modality.FLAIR,
}
#: order in which we look for an in-space reference image to align the mask against
_REF_PREFERENCE = (Modality.CT1, Modality.T1, Modality.T2, Modality.FLAIR)


def _present(cell: str | None) -> bool:
    return str(cell).strip().lower() == "x"


class LumiereAdapter(Adapter):
    """Resolve LUMIERE (manifest + zip) into the internal contract. Format-decode only."""

    name = "lumiere-zip"

    def __init__(
        self,
        zip_path: str,
        manifest_csv: str,
        mask_source: MaskSource = MaskSource.DEEPBRATUMIA,
    ) -> None:
        if mask_source is not MaskSource.DEEPBRATUMIA:
            raise NotImplementedError(
                f"Only DeepBraTumIA is wired as the canonical mask; got {mask_source!r}."
            )
        self._src = ZipSource(zip_path)
        self._mask_source = mask_source
        #: patient -> list of (tp_label, {Modality: present}, {mask_key: present}) in manifest order
        self._manifest = self._read_manifest(manifest_csv)

    # ---- manifest ---------------------------------------------------------------------
    @staticmethod
    def _read_manifest(manifest_csv: str) -> "OrderedDict[str, list[tuple]]":
        out: "OrderedDict[str, list[tuple]]" = OrderedDict()
        with open(manifest_csv, newline="") as fh:
            for row in csv.DictReader(fh):
                patient = row["Patient"].strip()
                tp = row["Timepoint"].strip()
                mods = {mod: _present(row.get(col)) for col, mod in _MODALITY_COLS.items()}
                masks = {"DeepBraTumIA": _present(row.get("DeepBraTumIA"))}
                out.setdefault(patient, []).append((tp, mods, masks))
        return out

    # ---- Adapter API ------------------------------------------------------------------
    def patient_ids(self) -> list[str]:
        return list(self._manifest.keys())

    def load_patient(self, patient_id: str) -> Patient:
        if patient_id not in self._manifest:
            raise KeyError(f"{patient_id!r} not in LUMIERE manifest")
        rows = sorted(self._manifest[patient_id], key=lambda r: weeks.sort_key(r[0]))
        timepoints = tuple(self._build_timepoint(patient_id, tp, mods, masks) for tp, mods, masks in rows)
        return Patient(id=patient_id, timepoints=timepoints)

    # ---- timepoint assembly -----------------------------------------------------------
    def _build_timepoint(self, patient: str, tp: str, mods: dict, masks: dict) -> Timepoint:
        modalities: dict[Modality, ZipNiftiRef] = {}
        for modality, present in mods.items():
            if not present:
                continue
            member = paths.raw_image(patient, tp, modality)
            if self._src.exists(member):
                modalities[modality] = ZipNiftiRef(self._src, member, Space.native(member))

        mask = mask_reference = brain_mask = None
        mask_src = None
        if masks.get("DeepBraTumIA"):
            mem_mask = paths.dbt_mask(patient, tp)
            if self._src.exists(mem_mask):
                mni = Space.mni152_1mm()
                mask = ZipNiftiRef(self._src, mem_mask, mni)
                mask_src = MaskSource.DEEPBRATUMIA
                mask_reference = self._first_skull_strip(patient, tp, mni)
                mem_brain = paths.dbt_brain_mask(patient, tp)
                if self._src.exists(mem_brain):
                    brain_mask = ZipNiftiRef(self._src, mem_brain, mni)

        return Timepoint(
            id=f"{patient}/{tp}",
            label=tp,
            week_offset=weeks.week_offset(tp),
            modalities=modalities,
            mask=mask,
            mask_source=mask_src,
            mask_reference=mask_reference,
            brain_mask=brain_mask,
        )

    def _first_skull_strip(self, patient: str, tp: str, space: Space) -> ZipNiftiRef | None:
        for modality in _REF_PREFERENCE:
            member = paths.dbt_skull_strip(patient, tp, modality)
            if self._src.exists(member):
                return ZipNiftiRef(self._src, member, space)
        return None

    # ---- integrity audit (manifest claims vs. what the archive actually holds) ---------
    def audit_manifest_vs_zip(self) -> list[dict]:
        """Rows where the manifest and the zip disagree — surfaced, never silently dropped."""
        discrepancies: list[dict] = []
        for patient, rows in self._manifest.items():
            for tp, mods, masks in rows:
                for modality, present in mods.items():
                    member = paths.raw_image(patient, tp, modality)
                    exists = self._src.exists(member)
                    if present != exists:
                        discrepancies.append(
                            {
                                "patient": patient, "timepoint": tp, "item": modality.value,
                                "manifest": present, "zip": exists, "member": member,
                            }
                        )
                member = paths.dbt_mask(patient, tp)
                exists = self._src.exists(member)
                if masks.get("DeepBraTumIA") != exists:
                    discrepancies.append(
                        {
                            "patient": patient, "timepoint": tp, "item": "DeepBraTumIA",
                            "manifest": masks.get("DeepBraTumIA"), "zip": exists, "member": member,
                        }
                    )
        return discrepancies


__all__ = ["LumiereAdapter", "ZipSource"]
