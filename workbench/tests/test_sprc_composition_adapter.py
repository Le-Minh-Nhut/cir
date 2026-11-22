from __future__ import annotations

import numpy as np
import pytest

from workbench.composition import model_sprc


def test_vitg_checkpoint_is_rejected_before_configuration_resolution() -> None:
    with pytest.raises(ValueError, match="ViT-G is a distinct"):
        model_sprc.Adapter(object(), "sprc_fiq_unresolved")


def test_score_matches_native_query_to_each_target_token_without_mutation() -> None:
    adapter = model_sprc.Adapter.__new__(model_sprc.Adapter)
    queries = np.zeros((2, 256), dtype=np.float32)
    queries[:, :2] = ((3.0, 4.0), (5.0, 12.0))
    targets = np.zeros((3, 8192), dtype=np.float32)
    targets[0, 31 * 256:31 * 256 + 2] = (6.0, 8.0)
    targets[1, 0:2] = (0.0, 2.0)
    targets[2, 12 * 256:12 * 256 + 2] = (-3.0, 4.0)
    original_queries, original_targets = queries.copy(), targets.copy()
    actual = adapter.score(queries, targets)
    expected = np.array([[1.0, 0.8, 0.28], [126.0 / 130.0, 12.0 / 13.0, 33.0 / 65.0]], dtype=np.float32)
    np.testing.assert_allclose(actual, expected, atol=1e-7)
    np.testing.assert_array_equal(queries, original_queries)
    np.testing.assert_array_equal(targets, original_targets)
    assert adapter.rank(queries, targets, [[2], [1]]) == [3, 2]


def test_native_normalization_epsilon_for_near_zero_features() -> None:
    adapter = model_sprc.Adapter.__new__(model_sprc.Adapter)
    query = np.zeros((1, 256), dtype=np.float32)
    query[0, 0] = 1e-15
    target = np.zeros((1, 8192), dtype=np.float32)
    target[0, 0] = 1e-15
    np.testing.assert_allclose(adapter.score(query, target), [[1e-6]], atol=1e-12)
    np.testing.assert_array_equal(adapter.score(np.zeros((1, 256), dtype=np.float32), target), [[0.0]])


def test_sprc_rejects_malformed_native_feature_shapes() -> None:
    adapter = model_sprc.Adapter.__new__(model_sprc.Adapter)
    with pytest.raises(ValueError, match="native SPRC query"):
        adapter.score(np.zeros((1, 8192)), np.zeros((1, 8192)))

def test_native_config_changes_sprc_adapter_identity(tmp_path, monkeypatch) -> None:
    from types import SimpleNamespace

    adapter = model_sprc.Adapter.__new__(model_sprc.Adapter)
    source = tmp_path / "third_party/SPRC/src"
    for relative in (
        "lavis/configs/models/blip2/blip2_pretrain_vitL.yaml",
        "lavis/models/blip2_models/blip2_qformer_cir_align_prompt.py",
        "lavis/processors/blip_processors.py",
        "data_utils.py",
    ):
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    adapter.config = SimpleNamespace(WORKBENCH_THIRD_PARTY_ROOT=tmp_path / "third_party",
                                     WORKBENCH_CHECKPOINT_ROOT=tmp_path / "checkpoints")
    adapter.model = {"source_dir": "SPRC"}
    adapter.model_id = "sprc"
    adapter.checkpoint_id = "sprc_fiq_vitl"
    adapter.checkpoint = {"relative_path": "checkpoint.pt"}
    adapter.query_dimension = 256
    adapter.score_dimension = 8192
    adapter.cache_policy = "reusable"
    adapter.batch_size = 32
    monkeypatch.setattr("workbench.backend.registry.checkpoint_path", lambda *args: checkpoint)
    before = adapter.prepare_identity()
    config = source / "lavis/configs/models/blip2/blip2_pretrain_vitL.yaml"
    config.write_text("changed", encoding="utf-8")
    after = adapter.prepare_identity()
    assert before["native_configuration"]["sha256"] != after["native_configuration"]["sha256"]
