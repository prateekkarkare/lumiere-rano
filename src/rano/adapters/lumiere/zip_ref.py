"""
Streaming NIfTI access straight out of the LUMIERE zip — no extraction of the 32 GB archive.

``ZipSource`` owns the ``ZipFile`` handle and an in-memory set of member names (the central
directory, ~0.1 s to index). ``ZipNiftiRef`` is the concrete ``ImageRef`` the LUMIERE adapter
attaches to timepoints:

  * ``geometry`` decompresses ONLY the NIfTI header (nibabel reads shape/affine/dtype lazily),
    so a whole-cohort fingerprint never materializes voxels. It is cached per ref.
  * ``load`` decompresses the full array on demand. It is intentionally NOT cached — holding
    every array would defeat the laziness the contract promises. A bounded LRU can be layered
    on later if a stage needs repeated reads.

Caveat (honest): streaming from a zip means the compressed member is read into memory and
gunzipped; it defeats nibabel's mmap. That is fine for LUMIERE development but is a dev
convenience, not the production I/O path — which is exactly why this lives behind ``ImageRef``.
NOTE: ``ZipFile`` is not safe for concurrent reads; this layer assumes single-threaded use.
"""

from __future__ import annotations

import gzip
import io
import json
import zipfile

import nibabel as nib
import numpy as np

from rano.contract.case import Geometry, ImageRef, LoadedImage, Space


class ZipSource:
    """A lazily-opened zip archive with a cached member index."""

    def __init__(self, zip_path: str) -> None:
        self.zip_path = str(zip_path)
        self._zf: zipfile.ZipFile | None = None
        self._names: frozenset[str] | None = None

    @property
    def zf(self) -> zipfile.ZipFile:
        if self._zf is None:
            self._zf = zipfile.ZipFile(self.zip_path)
        return self._zf

    @property
    def names(self) -> frozenset[str]:
        if self._names is None:
            self._names = frozenset(self.zf.namelist())
        return self._names

    def exists(self, member: str) -> bool:
        return member in self.names

    def read(self, member: str) -> bytes:
        return self.zf.read(member)

    def read_json(self, member: str) -> dict:
        return json.loads(self.read(member))

    def open_nifti(self, member: str) -> nib.Nifti1Image:
        """Wrap a gzipped NIfTI member as a nibabel image without extracting to disk."""
        raw = self.read(member)
        fh = nib.FileHolder(fileobj=gzip.GzipFile(fileobj=io.BytesIO(raw)))
        return nib.Nifti1Image.from_file_map({"header": fh, "image": fh})

    def close(self) -> None:
        if self._zf is not None:
            self._zf.close()
            self._zf = None


class ZipNiftiRef(ImageRef):
    """A lazy handle to one NIfTI member inside a ``ZipSource``."""

    __slots__ = ("_src", "_member", "_space", "_geometry")

    def __init__(self, src: ZipSource, member: str, space: Space) -> None:
        self._src = src
        self._member = member
        self._space = space
        self._geometry: Geometry | None = None

    @property
    def source(self) -> str:
        return self._member

    @property
    def space(self) -> Space:
        return self._space

    @property
    def geometry(self) -> Geometry:
        if self._geometry is None:
            img = self._src.open_nifti(self._member)  # header decompressed only
            self._geometry = Geometry.from_header(img.shape, img.affine, img.get_data_dtype())
        return self._geometry

    def load(self) -> LoadedImage:
        img = self._src.open_nifti(self._member)
        data = np.asanyarray(img.dataobj)  # full decompress here, and only here
        return LoadedImage(data=data, affine=np.asarray(img.affine, dtype=float), header=img.header)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"ZipNiftiRef({self._member!r}, space={self._space})"
