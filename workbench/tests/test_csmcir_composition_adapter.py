from __future__ import annotations

import numpy as np
import pytest

from workbench.composition import adapters, model_csmcir





def test_csmcir_selects_native_checkpoint_payload() -> None:
    state = {"epoch": 4, "Blip2QformerCirAlignPromptTargetImageCaptionshiyan92": {"weight": 1}}
    assert model_csmcir._csmcir_state_dict(state) == {"weight": 1}
    with pytest.raises(ValueError, match="expected one CSMCIR checkpoint state"):
        model_csmcir._csmcir_state_dict({"epoch": 4, "first": {}, "second": {}})

def test_caption_content_changes_adapter_identity(tmp_path) -> None:
    from types import SimpleNamespace

    adapter = model_csmcir.Adapter.__new__(model_csmcir.Adapter)
    third_party = tmp_path / "third_party"
    source = third_party / "CSMCIR" / "src"
    for relative in (
        "lavis/configs/models/blip2/blip2_pretrain.yaml",
        "lavis/processors/blip_processors.py",
        "lavis/models/blip2_models/blip2_qformer_cir_align_prompt_csmcir.py",
        "data_utils_csmcir.py",
    ):
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    caption_root = tmp_path / "assets/assets/csmcir/cf0c19bb346266c295b4b5772ebf03bd1c4f0467/fashioniq_qwen_captions/qwen_captions"
    caption_root.mkdir(parents=True)
    for category in adapter.category_scopes:
        (caption_root / f"{category}_cot_val.json").write_text('{"id":{"Final_Caption":"first"}}', encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    adapter.config = SimpleNamespace(WORKBENCH_THIRD_PARTY_ROOT=third_party,
                                     WORKBENCH_ARTIFACTS_ROOT=tmp_path / "assets")
    adapter._model = {"source_dir": "CSMCIR"}
    adapter._checkpoint = {"resolved_path": checkpoint}
    adapter.batch_size = 32
    adapter.cache_policy = "reusable"
    before = adapter.prepare_identity()
    caption = caption_root / "dress_cot_val.json"
    caption.write_text('{"id":{"Final_Caption":"changed"}}', encoding="utf-8")
    after = adapter.prepare_identity()
    assert before["target_caption_assets"]["dress"]["sha256"] != after["target_caption_assets"]["dress"]["sha256"]

def test_caption_cache_reloads_only_when_content_changes(tmp_path) -> None:
    caption = tmp_path / "dress.json"
    caption.write_text('{"id":{"Final_Caption":"first"}}', encoding="utf-8")
    cache = {}
    first = model_csmcir._load_caption_map(cache, "dress", caption)
    assert model_csmcir._load_caption_map(cache, "dress", caption) is first
    caption.write_text('{"id":{"Final_Caption":"changed"}}', encoding="utf-8")
    second = model_csmcir._load_caption_map(cache, "dress", caption)
    assert second["id"]["Final_Caption"] == "changed"
    assert second is not first

def test_target_encoding_requires_one_explicit_fashioniq_category(tmp_path) -> None:
    adapter = model_csmcir.Adapter.__new__(model_csmcir.Adapter)
    adapter.work_root = tmp_path
    with pytest.raises(ValueError, match="one explicit FashionIQ category scope"):
        adapter.encode_targets([tmp_path / "dress.png"], [None])
def test_score_matches_native_query_to_each_target_token_without_mutation() -> None:
    adapter = model_csmcir.Adapter.__new__(model_csmcir.Adapter)
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
    adapter = model_csmcir.Adapter.__new__(model_csmcir.Adapter)
    query = np.zeros((1, 256), dtype=np.float32)
    query[0, 0] = 1e-15
    target = np.zeros((1, 8192), dtype=np.float32)
    target[0, 0] = 1e-15
    np.testing.assert_allclose(adapter.score(query, target), [[1e-6]], atol=1e-12)
    np.testing.assert_array_equal(adapter.score(np.zeros((1, 256), dtype=np.float32), target), [[0.0]])


def test_csmcir_rejects_malformed_native_feature_shapes() -> None:
    adapter = model_csmcir.Adapter.__new__(model_csmcir.Adapter)
    with pytest.raises(ValueError, match="native query"):
        adapter.score(np.zeros((1, 8192)), np.zeros((1, 8192)))
