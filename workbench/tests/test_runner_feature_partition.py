"""The ENCODE partition is recorded, and a fallback cache can never be read as canonical.

`runner._encode_queries` retries a failed adapter call ONE QUERY AT A TIME. For a batch-sensitive
model that is not a retry, it is a different arithmetic shape: CLVC-Net query rows move by up to
1.4639e-02 between batch 1 and batch 20, and the pinned upstream evaluator moves identically
(87 §3b). The SCORING fallback already records itself (`condition_status.<cond>.fallback_used`);
this is the ENCODE-side record, kept under its own name so the scoring key keeps its meaning.

CPU only: a fixture adapter whose `encode_queries` raises for a multi-row call and answers a
one-row call. No GPU, no checkpoint.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from workbench.composition import runner

DIMENSION = 4

class BatchStrictAdapter:
    """Fails any call carrying more than one row; succeeds one row at a time."""

    def __init__(self) -> None:
        self.calls: list[int] = []

    def encode_queries(self, image_paths, texts, categories):
        self.calls.append(len(image_paths))
        if len(image_paths) > 1:
            raise RuntimeError("fixture batch rejection")
        return np.full((len(image_paths), DIMENSION), 1.0, dtype=np.float32)

class BatchAdapter:
    """Always succeeds; records the call shapes it was handed."""

    def __init__(self) -> None:
        self.calls: list[int] = []

    def encode_queries(self, image_paths, texts, categories):
        self.calls.append(len(image_paths))
        return np.ones((len(image_paths), DIMENSION), dtype=np.float32)

def _queries(tmp_path: Path, count: int = 3):
    queries, paths = [], []
    for index in range(count):
        source = tmp_path / f"q{index}.png"
        Image.new("RGB", (3, 2), (index + 1, 1, 1)).save(source)
        queries.append(SimpleNamespace(qid=f"7:{index + 1}", query_txt=f"text {index}",
                                       category="dress"))
        paths.append(source)
    return queries, paths

def _encode(adapter, tmp_path: Path, destination: Path, queries, paths):
    return runner._encode_queries(adapter, queries, paths, list(range(len(queries))), "multimodal",
                                  tmp_path, destination, "fingerprint-abc", DIMENSION, 2)

def test_mixed_partition_is_recorded_with_qids_and_reason(tmp_path: Path) -> None:
    queries, paths = _queries(tmp_path)
    adapter = BatchStrictAdapter()
    destination = tmp_path / "features" / "query_multimodal_dress.npy"
    features, errors, category_errors = _encode(adapter, tmp_path, destination, queries, paths)

    record = json.loads(destination.with_suffix(destination.suffix + runner.FEATURE_FALLBACK_SUFFIX)
                        .read_text(encoding="utf-8"))
    assert record["partition"] == runner.FEATURE_PARTITION_MIXED
    assert record["fallback_used"] is True
    assert record["fallback_qids"] == ["7:1", "7:2", "7:3"]
    assert "fixture batch rejection" in record["fallback_reason"]
    assert record["runner_batch_size"] == 2
    # one rejected group call, then one call per surviving row
    assert adapter.calls == [3, 1, 1, 1]
    assert record["adapter_calls"] == 4
    assert errors == {} and category_errors == {}
    assert np.isfinite(np.asarray(features)).all()

def test_canonical_partition_is_recorded_and_reused(tmp_path: Path) -> None:
    queries, paths = _queries(tmp_path)
    destination = tmp_path / "features" / "query_multimodal_dress.npy"
    first = BatchAdapter()
    _encode(first, tmp_path, destination, queries, paths)
    record = json.loads(destination.with_suffix(destination.suffix + runner.FEATURE_FALLBACK_SUFFIX)
                        .read_text(encoding="utf-8"))
    assert record["partition"] == runner.FEATURE_PARTITION_CANONICAL
    assert record["fallback_used"] is False and record["fallback_qids"] == []
    assert record["fallback_reason"] is None
    assert record["cache_key"] == "fingerprint-abc"
    assert first.calls == [3]

    second = BatchAdapter()
    _encode(second, tmp_path, destination, queries, paths)
    assert second.calls == [], "the canonical cache must be a HIT: same key, same configuration"

def test_fallback_cache_is_never_reused_as_the_canonical_configuration(tmp_path: Path) -> None:
    """Same inputs, different partition -> different key, so the canonical load must MISS."""
    queries, paths = _queries(tmp_path)
    destination = tmp_path / "features" / "query_multimodal_dress.npy"

    _encode(BatchStrictAdapter(), tmp_path, destination, queries, paths)
    sidecar = json.loads(destination.with_suffix(destination.suffix + ".json").read_text(encoding="utf-8"))
    fallback_key = sidecar["fingerprint"]
    assert fallback_key != "fingerprint-abc"
    assert runner._load_memmap(destination, "fingerprint-abc", (len(queries), DIMENSION)) is None

    # Same configuration, canonical this time: the memmap must be rebuilt, not read.
    adapter = BatchAdapter()
    _encode(adapter, tmp_path, destination, queries, paths)
    assert adapter.calls == [3]
    sidecar = json.loads(destination.with_suffix(destination.suffix + ".json").read_text(encoding="utf-8"))
    assert sidecar["fingerprint"] == "fingerprint-abc"

def test_partition_key_depends_on_the_affected_qids() -> None:
    first = runner._feature_partition_fingerprint("fp", {"fallback_qids": ["7:1"]})
    second = runner._feature_partition_fingerprint("fp", {"fallback_qids": ["7:2"]})
    canonical = runner._feature_partition_fingerprint("fp", {"fallback_qids": []})
    assert canonical == "fp"
    assert first == runner._feature_partition_fingerprint("fp", {"fallback_qids": ["7:1"]})
    assert len({first, second, canonical}) == 3

def test_partition_record_names_the_policy_and_reports_absence(tmp_path: Path) -> None:
    blocks = [{"partition": runner.FEATURE_PARTITION_MIXED, "fallback_qids": ["7:1"],
               "fallback_reason": "RuntimeError: boom", "call_window": 4096,
               "forward_batch_size": 32, "adapter_calls": 5},
              {"partition": runner.FEATURE_PARTITION_CANONICAL, "fallback_qids": [],
               "call_window": 4096, "forward_batch_size": 32, "adapter_calls": 2}]
    merged = runner._feature_partition_record(blocks)
    assert merged["partition"] == runner.FEATURE_PARTITION_MIXED
    assert merged["fallback_qids"] == ["7:1"]
    assert merged["adapter_calls"] == 7
    missing = runner._feature_partition_record([{}])
    assert missing["partition"] is None and missing["fallback_used"] is None
    assert "unknown" in missing["reason"]


def test_partition_read_from_disk_names_the_condition_and_the_unencoded_case(tmp_path: Path) -> None:
    queries, paths = _queries(tmp_path)
    _encode(BatchAdapter(), tmp_path, tmp_path / "features" / "query_multimodal_dress.npy",
            queries, paths)
    on_disk = runner._query_feature_partition(tmp_path, "multimodal", ["dress"])
    assert on_disk["partition"] == runner.FEATURE_PARTITION_CANONICAL
    assert on_disk["runner_batch_size"] == 2 and on_disk["fallback_used"] is False

    never_encoded = runner._query_feature_partition(
        tmp_path, "image_only_empty_text", ["dress"],
        not_encoded_reason="never encoded: ABLATION_UNSUPPORTED")
    assert never_encoded["partition"] == "not_encoded"
    assert "ABLATION_UNSUPPORTED" in never_encoded["reason"]
