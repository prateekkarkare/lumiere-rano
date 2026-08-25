"""
Fingerprint extractor — the one place in Piece 1 that deliberately loads voxel arrays.

Inspired by nnU-Net's ``DatasetFingerprintExtractor``: the point of a fingerprint is not to
describe a case for its own sake, it's to measure exactly the facts the next stage (here,
the validator/router) would otherwise have to re-derive. Every field this module produces
is chosen because something downstream needs that number, not because it was easy to compute.

Basis rules, and why:
  * Native modalities (CT1/T1/T2/FLAIR) arrive pre-skull-stripped (see
    ``adapters/lumiere/paths.py``), so zero voxels are background, not tissue. There is no mask
    in native space to restrict against (that would require registration — forbidden here), so
    "foreground" is approximated as nonzero voxels, the same effective basis nnU-Net gets from
    running its fingerprinter on already-nonzero-cropped images.
  * ``mask_reference`` shares the mask's own MNI grid, so two distinct stats are meaningful:
      (a) brain-extent intensity, restricted to ``brain_mask == 1`` (the coarse "is this
          tissue" analog to the native-image nonzero basis);
      (b) per-compartment intensity, restricted to each label found in ``mask`` — the
          nnU-Net-style segmentation-foreground stat that actually drives normalization
          decisions. Every label present (including background=0) is reported; nothing here
          judges whether that label is expected — see ``schema.py``'s note on label semantics.
  * ``mask`` and ``brain_mask`` are categorical/binary: no intensity stats. ``mask`` additionally
    gets a raw label histogram.

Extraction failures (corrupt NIfTI, unreadable zip member, mismatched shapes between mask and
mask_reference) are NOT caught here — they raise and stop the run. A silently-null fingerprint
for a corrupt case is worse than a loud failure: the whole point of this pass is to be the one
place that actually looks at the pixels.
"""

from __future__ import annotations

import numpy as np

from rano.contract.case import Geometry, ImageRef, Patient, Timepoint
from rano.fingerprint.schema import (
    CompartmentIntensity,
    GeometryFingerprint,
    ImageFingerprint,
    IntensityStats,
    LabelCount,
    PatientFingerprint,
    TimepointFingerprint,
)


def _geometry_fingerprint(geo: Geometry) -> GeometryFingerprint:
    return GeometryFingerprint(
        shape=geo.shape,
        spacing=geo.spacing,
        orientation=geo.orientation,
        dtype=geo.dtype,
        anisotropy_ratio=geo.anisotropy_ratio,
        affine=tuple(tuple(float(x) for x in row) for row in geo.affine),
    )


def _intensity_stats(values: np.ndarray) -> IntensityStats:
    """Summarize a basis-filtered array of voxel intensities. Empty basis -> NaN stats, not a crash."""
    n = int(values.size)
    if n == 0:
        nan = float("nan")
        return IntensityStats(n_voxels=0, mean=nan, std=nan, min=nan, max=nan, median=nan, p0_5=nan, p99_5=nan)
    return IntensityStats(
        n_voxels=n,
        mean=float(np.mean(values)),
        std=float(np.std(values)),
        min=float(np.min(values)),
        max=float(np.max(values)),
        median=float(np.median(values)),
        p0_5=float(np.percentile(values, 0.5)),
        p99_5=float(np.percentile(values, 99.5)),
    )


def _label_histogram(labels: np.ndarray) -> tuple[LabelCount, ...]:
    values, counts = np.unique(labels, return_counts=True)
    return tuple(LabelCount(label=int(v), n_voxels=int(c)) for v, c in zip(values, counts))


def _fingerprint_native_image(role: str, ref: ImageRef) -> ImageFingerprint:
    data = np.asarray(ref.load().data)
    return ImageFingerprint(
        role=role,
        source=ref.source,
        space=str(ref.space),
        geometry=_geometry_fingerprint(ref.geometry),
        intensity=_intensity_stats(data[data != 0]),
    )


def _fingerprint_mask(ref: ImageRef) -> ImageFingerprint:
    data = np.asarray(ref.load().data)
    return ImageFingerprint(
        role="mask",
        source=ref.source,
        space=str(ref.space),
        geometry=_geometry_fingerprint(ref.geometry),
        label_histogram=_label_histogram(data),
    )


def _fingerprint_brain_mask(ref: ImageRef) -> ImageFingerprint:
    return ImageFingerprint(
        role="brain_mask",
        source=ref.source,
        space=str(ref.space),
        geometry=_geometry_fingerprint(ref.geometry),
    )


def _fingerprint_mask_reference(
    ref: ImageRef, mask_ref: ImageRef | None, brain_mask_ref: ImageRef | None
) -> ImageFingerprint:
    data = np.asarray(ref.load().data)

    if brain_mask_ref is not None:
        brain = np.asarray(brain_mask_ref.load().data)
        intensity = _intensity_stats(data[brain != 0])
    else:
        intensity = _intensity_stats(data[data != 0])

    compartment_intensity = None
    if mask_ref is not None:
        labels = np.asarray(mask_ref.load().data)
        compartment_intensity = tuple(
            CompartmentIntensity(label=int(lbl), stats=_intensity_stats(data[labels == lbl]))
            for lbl in np.unique(labels)
        )

    return ImageFingerprint(
        role="mask_reference",
        source=ref.source,
        space=str(ref.space),
        geometry=_geometry_fingerprint(ref.geometry),
        intensity=intensity,
        compartment_intensity=compartment_intensity,
    )


def fingerprint_timepoint(tp: Timepoint) -> TimepointFingerprint:
    """Materialize every image attached to one timepoint. Absent slots stay absent (never faked)."""
    modalities = {
        modality.value: _fingerprint_native_image(f"modality:{modality.value}", ref)
        for modality, ref in tp.modalities.items()
    }
    mask_fp = _fingerprint_mask(tp.mask) if tp.mask is not None else None
    brain_mask_fp = _fingerprint_brain_mask(tp.brain_mask) if tp.brain_mask is not None else None
    mask_reference_fp = (
        _fingerprint_mask_reference(tp.mask_reference, tp.mask, tp.brain_mask)
        if tp.mask_reference is not None
        else None
    )

    return TimepointFingerprint(
        id=tp.id,
        label=tp.label,
        week_offset=tp.week_offset,
        modalities=modalities,
        mask=mask_fp,
        mask_source=tp.mask_source.value if tp.mask_source is not None else None,
        mask_reference=mask_reference_fp,
        brain_mask=brain_mask_fp,
    )


def fingerprint_patient(patient: Patient) -> PatientFingerprint:
    """Materialize every image across every timepoint of ``patient`` into one pydantic record."""
    timepoints = tuple(fingerprint_timepoint(tp) for tp in patient)
    return PatientFingerprint(
        patient_id=patient.id,
        n_timepoints=patient.n_timepoints,
        timepoints=timepoints,
    )


__all__ = ["fingerprint_patient", "fingerprint_timepoint"]
