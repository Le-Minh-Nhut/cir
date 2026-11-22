from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DIAGNOSTIC = REPO / "workbench/docs/reviews/20261112-parity-repair/scripts/hint_adapter_diagnostic.py"
NATIVE_ROOT = REPO / "workbench/docs/reviews/20261112-parity-repair/raw/third_model/hint-identity-prereq-20261120-final"
RECEIPT = REPO / "workbench/docs/reviews/20261112-parity-repair/raw/third_model/hint-dress-native-20261120/diagnostic-receipt.json"


def load():
    spec = importlib.util.spec_from_file_location("hint_adapter_diagnostic_cpu_test", DIAGNOSTIC)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_compare_uses_native_order_mask_and_adapter_score():
    diagnostic = load()
    gallery_ids = ["ref", "target", "other", "last"]
    refs = ["ref", "other"]
    targets = ["target", "last"]
    source = np.asarray([[-1e11, 0.8, 0.1, 0.0],
                         [0.8, 0.1, -1e11, 0.9]], dtype=np.float32)
    source_order = np.asarray([[1, 2, 3, 0], [3, 0, 1, 2]], dtype=np.int32)
    query_vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    gallery_vectors = np.asarray([[[0.1, 0.0], [0.0, 0.8]],
                                  [[0.7, 0.0], [0.0, 0.1]],
                                  [[0.8, 0.0], [0.0, 0.2]],
                                  [[0.0, 0.0], [0.0, 0.9]]], dtype=np.float32)
    from workbench.composition.model_hint import Adapter

    adapter_scores = Adapter.score(object.__new__(Adapter), query_vectors, gallery_vectors)
    result = diagnostic.compare(source, source_order, adapter_scores, gallery_ids, refs, targets)
    assert result["source_hit_counts"] == {"r1": 2, "r10": 2, "r50": 2}
    assert result["adapter_hit_counts"] == {"r1": 1, "r10": 2, "r50": 2}
    assert result["target_rank_flip_query_indices"] == [0]
    assert result["target_hit_flip_query_indices"]["k1"] == [0]
    assert result["topk_candidate_membership_flip_query_indices"]["k1"] == [0]
    assert np.all(result["adapter_masked_scores"][[0, 1], [0, 2]] == np.float32(-1e11))
    assert result["max_abs_score_delta"] > 0



def test_cli_refuses_existing_output_without_mutating_it(tmp_path, monkeypatch):
    diagnostic = load()
    output_root = tmp_path / "existing"
    output_root.mkdir()
    sentinel = output_root / "diagnostic.json"
    sentinel.write_bytes(b"keep exactly\x00these bytes\n")
    before = sentinel.read_bytes()
    monkeypatch.setattr(sys, "argv", [
        str(DIAGNOSTIC), "--native-root", str(tmp_path / "missing-native"),
        "--native-receipt", str(tmp_path / "missing-receipt.json"),
        "--output-root", str(output_root),
    ])
    assert diagnostic.main() == 1
    assert sentinel.read_bytes() == before
