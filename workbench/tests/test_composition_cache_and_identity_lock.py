"""S-04 / S-03: the cache and identity facts the campaign says are broken, checked against HEAD.

Both findings were reported against an older commit and the code now carries the guards. What was
missing is a test that would FAIL if they were removed, so the claims are locked rather than trusted:

* S-04 -- a memmap whose bytes were replaced with a same-dtype, same-shape array must not load; a
  half-written file, a stale sidecar, and a wrong fingerprint must all fail closed. `WB_CACHE_VERIFY=off`
  must still refuse a truncated file (a header check is weaker, not absent).
* S-03 -- the run identity must partition on every arithmetic input: BLAS thread count, the resolved
  query-scope map, and query order. When the artifact predates identity v2 its facts must read
  `unknown`, never be backfilled from today's source.

Old artifact bytes are never rewritten by either test: every fixture is a temporary directory.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workbench.composition import runner


def _cache(tmp_path: Path, name: str = "gallery_global.npy", rows: int = 4, width: int = 3,
           fill: float = 1.0, dtype: str = "float32") -> Path:
    path = tmp_path / "features" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    value = np.full((rows, width), fill, dtype=np.dtype(dtype))
    np.save(path, value)
    runner._write_memmap_metadata(path, "fp-good", value)
    return path


# ------------------------------------------------------------------ S-04

def test_same_shape_and_dtype_with_different_bytes_is_refused(tmp_path: Path) -> None:
    path = _cache(tmp_path)
    np.save(path, np.zeros((4, 3), dtype=np.float32))
    assert runner._load_memmap(path, "fp-good", (4, 3)) is None, (
        "replacing the array with same-shape same-dtype zeros must not load")


def test_a_different_dtype_with_the_same_shape_is_refused(tmp_path: Path) -> None:
    path = _cache(tmp_path)
    np.save(path, np.ones((4, 3), dtype=np.float64))
    assert runner._load_memmap(path, "fp-good", (4, 3)) is None


def test_a_wrong_fingerprint_is_refused(tmp_path: Path) -> None:
    path = _cache(tmp_path)
    assert runner._load_memmap(path, "fp-other", (4, 3)) is None


def test_a_truncated_file_is_refused_even_with_content_verification_off(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`off` skips the digest, never the byte-length and shape checks."""
    path = _cache(tmp_path)
    with open(path, "r+b") as handle:
        handle.truncate(path.stat().st_size // 2)
    monkeypatch.setenv("WB_CACHE_VERIFY", "off")
    assert runner._load_memmap(path, "fp-good", (4, 3)) is None


def test_a_stale_sidecar_shape_is_refused(tmp_path: Path) -> None:
    path = _cache(tmp_path)
    sidecar = path.with_suffix(path.suffix + ".json")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    metadata["shape"] = [99, 3]
    sidecar.write_text(json.dumps(metadata), encoding="utf-8")
    assert runner._load_memmap(path, "fp-good", (4, 3)) is None


def test_the_digest_is_over_content_not_name(tmp_path: Path) -> None:
    a = _cache(tmp_path, "gallery_global.npy", fill=1.0)
    b = _cache(tmp_path, "query_multimodal_global.npy", fill=2.0)
    assert runner._memmap_integrity(a)["content_digest"] != runner._memmap_integrity(b)["content_digest"]


def test_a_legacy_cache_without_a_digest_loads_but_is_never_called_verified(tmp_path: Path) -> None:
    path = _cache(tmp_path)
    sidecar = path.with_suffix(path.suffix + ".json")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    metadata.pop("content_digest")
    sidecar.write_text(json.dumps(metadata), encoding="utf-8")
    assert runner._load_memmap(path, "fp-good", (4, 3)) is not None
    integrity = runner._memmap_integrity(path)
    assert integrity["verified"] is False and integrity["content_digest"] == "absent"


# ------------------------------------------------------------------ S-03

def test_the_identity_partitions_on_the_blas_policy_and_the_scope_map() -> None:
    """The three v2 inputs must be in the identity block the run id is hashed from."""
    import inspect

    source = inspect.getsource(runner.run_composition)
    for field in ("blas_threads_pinned", "query_scope_resolution", "query_order_identity",
                  "cache_verify_mode", "blas_pin_capability"):
        assert f'"{field}"' in source, f"{field} must be hashed into the run identity"


def test_the_backend_refuses_to_backfill_a_pre_v2_identity() -> None:
    """A run recorded before v2 must report `unknown`, never inherit today's policy values."""
    from workbench.backend import composition as backend

    legacy_summary = {"run_id": "r1", "model_id": "limn", "checkpoint_id": "ckpt"}
    # A recognisable RUNNER block: it carries runner-only keys and NO v2 marker. `query_order_identity`
    # is itself a v2 marker, so a genuine pre-v2 block cannot contain it -- including it here would
    # make the fixture assert the opposite of what it tests.
    legacy_recorded = {"model_id": "limn", "checkpoint_id": "ckpt",
                       "gallery_identity": "g", "subset_memberships": ["full"],
                       "adapter_module_source_hash": "a", "scoring_module_source_hash": "s"}
    identity = backend.measurement_identity(legacy_summary, legacy_recorded)
    facts = identity["identity"]
    assert facts.get("identity_schema_version") == "unknown (predates identity v2)", facts
    assert identity["group_id"] is None, (
        "a pre-v2 identity has no v2 inputs, so it cannot be grouped as if it did")
    assert "predates identity v2" in json.dumps(identity)


def test_two_policies_never_share_a_group_id() -> None:
    """A BLAS=1 run and a BLAS=10 run measured different arithmetic; they cannot be duplicates."""
    from workbench.backend import composition as backend

    def summary(pinned: int) -> dict:
        return {"run_id": f"r{pinned}", "model_id": "limn", "checkpoint_id": "ckpt",
                "gallery_identity": "g", "query_order_identity": "q",
                "query_scope_resolution": "s", "cache_verify_mode": "full",
                "blas_threads_pinned": pinned}

    first = backend.measurement_identity(summary(1))
    second = backend.measurement_identity(summary(10))
    assert first["group_id"] is not None and second["group_id"] is not None
    assert first["group_id"] != second["group_id"], (
        "two BLAS policies are two measurements; a shared group id would present them as one")
