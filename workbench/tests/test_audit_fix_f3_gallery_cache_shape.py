"""F3: the gallery feature cache must be validated against this run's gallery size.

`_encode_gallery` used to call `_load_memmap(destination, fingerprint)` with no shape, while
`_encode_queries` passed `(len(queries), dimension)`. `_load_memmap` only enforces a shape when one
is given, so a cache file that is complete and readable but holds the wrong number of rows -- a
partially written gallery that also rewrote its sidecar, or a copied cache -- was reused as if it
described this run's gallery. Every row after the truncation point then belongs to a different
image, and the only symptom is a worse score.

A *byte-truncated* `.npy` is a different case: NumPy already refuses it at `np.load` with
`mmap length is greater than file size`, which `_load_memmap` catches. The check here therefore
covers the short-but-valid file, which is the case that no layer below refuses.
"""
from __future__ import annotations

import numpy as np
import pytest

from workbench.composition import runner


class CountingAdapter:
    """Return features derived from the file stem, and count how often encoding ran."""

    def __init__(self) -> None:
        self.encoded = 0

    def encode_targets(self, image_paths, categories):
        self.encoded += 1
        values = np.zeros((len(image_paths), 4), dtype=np.float32)
        for row, path in enumerate(image_paths):
            values[row, 0] = int(str(path).rsplit("img", 1)[1].split(".")[0])
        return values


def _write_cache(path, rows: int, fingerprint: str) -> None:
    mapped = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(rows, 4))
    mapped[:] = np.arange(rows, dtype=np.float32)[:, None]
    mapped.flush()
    del mapped
    # The sidecar's digest must describe the bytes that were written. It used to be built from a
    # fresh `np.zeros`, which the content-integrity check now (correctly) rejects as a mismatch:
    # a digest that does not cover the file is exactly the false assurance this check removes.
    runner._write_memmap_metadata(path, fingerprint, np.load(path, allow_pickle=False))


def _paths(tmp_path, count: int) -> list:
    return [tmp_path / f"img{index}.jpg" for index in range(count)]


def _encode(tmp_path, adapter, image_paths, fingerprint):
    destination = tmp_path / "features" / "gallery_global.npy"
    destination.parent.mkdir(parents=True, exist_ok=True)
    # One batch per pass, so `adapter.encoded` counts encode passes rather than batches.
    return runner._encode_gallery(
        adapter, image_paths, None, destination, fingerprint,
        batch_size=max(1, len(image_paths)), remaining_scopes=1, query_count=1, condition_count=1)


def test_short_gallery_cache_is_not_reused(tmp_path) -> None:
    """A 2-row cache must not answer for a 3-image gallery."""
    fingerprint = "f" * 64
    image_paths = _paths(tmp_path, 3)
    path = tmp_path / "features" / "gallery_global.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_cache(path, rows=2, fingerprint=fingerprint)

    adapter = CountingAdapter()
    mapped, _projection = _encode(tmp_path, adapter, image_paths, fingerprint)

    assert adapter.encoded == 1, "a mis-sized gallery cache must not be reused"
    assert mapped.shape == (3, 4)
    # Rebuilt from the images, not carried over from the stale rows.
    assert [float(row[0]) for row in mapped] == [0.0, 1.0, 2.0]


def test_matching_gallery_cache_is_still_reused(tmp_path) -> None:
    """The shape check must not break warm-cache reuse."""
    fingerprint = "f" * 64
    image_paths = _paths(tmp_path, 3)
    path = tmp_path / "features" / "gallery_global.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_cache(path, rows=3, fingerprint=fingerprint)

    adapter = CountingAdapter()
    mapped, facts = _encode(tmp_path, adapter, image_paths, fingerprint)

    assert adapter.encoded == 0, "a cache of the right shape must be reused"
    assert facts["reused"] == 1
    assert mapped.shape == (3, 4)


def test_byte_truncated_gallery_cache_is_refused_not_raised(tmp_path) -> None:
    """NumPy's own mmap refusal must reach the caller as a cache miss, not an exception."""
    fingerprint = "f" * 64
    image_paths = _paths(tmp_path, 4)
    path = tmp_path / "features" / "gallery_global.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_cache(path, rows=4, fingerprint=fingerprint)
    original = path.read_bytes()
    path.write_bytes(original[: len(original) // 2])

    adapter = CountingAdapter()
    mapped, _projection = _encode(tmp_path, adapter, image_paths, fingerprint)

    assert adapter.encoded == 1
    assert mapped.shape == (4, 4)
