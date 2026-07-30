"""Persist fingerprint records — one JSON file per patient, the on-disk audit trail."""

from __future__ import annotations

from pathlib import Path

from rano.fingerprint.schema import PatientFingerprint


def write_patient_fingerprint(record: PatientFingerprint, output_dir: Path | str) -> Path:
    """Write ``record`` to ``<output_dir>/<patient_id>.json``, creating ``output_dir`` if needed."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{record.patient_id}.json"
    path.write_text(record.model_dump_json(indent=2))
    return path


__all__ = ["write_patient_fingerprint"]
