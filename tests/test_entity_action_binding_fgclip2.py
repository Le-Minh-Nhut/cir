from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from cache.features import get_dense_features_by_ids, load_dense_image_features
from models.entity_action_binding import (
    EntityActionBindingCIR,
    FeatureWiseAffineFusion,
    SharedRelationBinder,
)
from models.retrieval import multi_positive_retrieval_loss, target_positive_mask


def _inputs(*, dim: int = 16, batch_size: int = 2) -> dict[str, object]:
    torch.manual_seed(7)
    return {
        "reference_global": torch.randn(batch_size, dim),
        "reference_dense": torch.randn(batch_size, 5, dim),
        "reference_dense_mask": torch.tensor(
            [[True, True, True, False, False], [True, True, True, True, False]]
        ),
        "target_global": torch.randn(batch_size, dim),
        "target_dense": torch.randn(batch_size, 6, dim),
        "target_dense_mask": torch.tensor(
            [[True, True, False, False, False, False], [True, True, True, True, True, False]]
        ),
        "text_global": torch.randn(batch_size, dim),
        "text_states": torch.randn(batch_size, 7, dim),
        # BOS/EOS and padding are all excluded, just as cached content_mask does.
        "text_content_mask": torch.tensor(
            [[False, True, True, True, False, False, False], [False, True, True, True, True, False, False]]
        ),
        "target_ids": ["same", "same"],
    }


def test_shared_query_invariant_and_masked_attention() -> None:
    model = EntityActionBindingCIR(dim=16, num_relations=4, fusion_hidden_dim=12)
    relation_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if "relation_queries" in name
    ]
    assert len(relation_parameters) == 1
    assert relation_parameters[0][0] == "binder.relation_queries"

    inputs = _inputs()
    output = model(**{key: value for key, value in inputs.items() if key != "target_ids"})
    for attention_name, mask_name in (
        ("vision_attention", "reference_dense_mask"),
        ("text_attention", "text_content_mask"),
    ):
        attention = output[attention_name]
        mask = inputs[mask_name]
        assert torch.equal(
            attention.masked_select(~mask[:, None, :]),
            torch.zeros_like(attention.masked_select(~mask[:, None, :])),
        )
        torch.testing.assert_close(
            attention.sum(dim=-1), torch.ones_like(attention.sum(dim=-1))
        )


def test_required_1024_shape_and_normalization_contract() -> None:
    model = EntityActionBindingCIR(dim=1024, num_relations=4, fusion_hidden_dim=64)
    inputs = _inputs(dim=1024)
    output = model(**{key: value for key, value in inputs.items() if key != "target_ids"})
    assert output["entity"].shape == (2, 4, 1024)
    assert output["action"].shape == (2, 4, 1024)
    assert output["vision_attention"].shape == (2, 4, 5)
    assert output["text_attention"].shape == (2, 4, 7)
    assert output["query"].shape == (2, 1024)
    assert output["target_embedding"].shape == (2, 1024)
    torch.testing.assert_close(output["query"].norm(dim=-1), torch.ones(2))
    torch.testing.assert_close(output["target_embedding"].norm(dim=-1), torch.ones(2))


def test_losses_are_finite_and_all_trainable_parameters_receive_finite_gradients() -> None:
    model = EntityActionBindingCIR(dim=16, num_relations=4, fusion_hidden_dim=12)
    inputs = _inputs()
    losses = model.compute_loss(inputs)
    for name in ("retrieval_loss", "entity_action_loss", "relation_ortho_loss"):
        assert torch.isfinite(losses[name])
    total = losses["retrieval_loss"] + losses["entity_action_loss"] + losses["relation_ortho_loss"]
    total.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    for value in inputs.values():
        if isinstance(value, torch.Tensor):
            assert value.grad is None


def test_multi_positive_duplicate_targets_are_not_negatives() -> None:
    query = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    target = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    mask = target_positive_mask(["x", "x"], 2, query.device)
    assert mask.all()
    duplicate_loss = multi_positive_retrieval_loss(
        query, target, ["x", "x"], temperature=0.07
    )
    unique_loss = multi_positive_retrieval_loss(
        query, target, ["x", "y"], temperature=0.07
    )
    torch.testing.assert_close(duplicate_loss, torch.zeros_like(duplicate_loss), atol=1e-6, rtol=0)
    assert unique_loss > duplicate_loss


def test_all_invalid_binding_is_stable_zero() -> None:
    binder = SharedRelationBinder(dim=8, num_relations=3)
    slots, attention = binder(torch.randn(2, 4, 8), torch.zeros(2, 4, dtype=torch.bool))
    assert torch.equal(attention, torch.zeros_like(attention))
    assert torch.equal(slots, torch.zeros_like(slots))
    assert torch.isfinite(slots).all()


def test_affine_initialization_is_exactly_reference_preserving() -> None:
    fusion = FeatureWiseAffineFusion(dim=8, hidden_dim=5)
    reference = torch.randn(2, 5, 8)
    text = torch.randn(2, 5, 8)
    assert torch.equal(fusion(reference, text), reference)


def test_target_features_cannot_change_composed_query() -> None:
    model = EntityActionBindingCIR(dim=16, num_relations=4, fusion_hidden_dim=12)
    inputs = _inputs()
    forward_inputs = {key: value for key, value in inputs.items() if key != "target_ids"}
    first = model(**forward_inputs)
    forward_inputs["target_global"] = torch.randn_like(forward_inputs["target_global"])
    forward_inputs["target_dense"] = torch.randn_like(forward_inputs["target_dense"])
    second = model(**forward_inputs)
    assert torch.equal(first["query"], second["query"])
    assert not torch.equal(first["target_embedding"], second["target_embedding"])


def _write_dense_cache(root: Path) -> tuple[np.ndarray, np.ndarray]:
    source = np.arange(5 * 4, dtype=np.float32).reshape(5, 4) / 13
    stored = source.astype(np.float16)
    np.save(root / "values.npy", stored)
    np.save(root / "offsets.npy", np.asarray([0, 2, 5], dtype=np.int64))
    np.save(root / "spatial_shapes.npy", np.asarray([[1, 2], [1, 3]], dtype=np.int32))
    (root / "name_to_idx.json").write_text(
        json.dumps({"short": 0, "long": 1}), encoding="utf-8"
    )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "model_id": "qihoo360/fg-clip2-large",
                "revision": "4d1d5dc35c716902f07c172dbfc23b82a7bc6bf3",
                "feature_kind": "fgclip2_dense_image_tokens_ragged",
                "feature_dim": 4,
                "total_token_count": 5,
            }
        ),
        encoding="utf-8",
    )
    return source, stored


def test_ragged_dense_cache_lookup_padding_masks_and_storage_parity(tmp_path: Path) -> None:
    source, stored = _write_dense_cache(tmp_path)
    cache = load_dense_image_features(tmp_path)
    tokens, mask = get_dense_features_by_ids(["long", "short"], cache)
    assert tokens.shape == (2, 3, 4)
    assert torch.equal(mask, torch.tensor([[True, True, True], [True, True, False]]))
    torch.testing.assert_close(tokens[0], torch.from_numpy(stored[2:].astype(np.float32)))
    assert torch.equal(tokens[1, 2], torch.zeros(4))
    assert float(np.max(np.abs(source - stored.astype(np.float32)))) < 2e-3


def test_corrupted_dense_offsets_fail_loudly(tmp_path: Path) -> None:
    _write_dense_cache(tmp_path)
    np.save(tmp_path / "offsets.npy", np.asarray([0, 3, 5], dtype=np.int64))
    with pytest.raises(ValueError, match="spatial_shapes"):
        load_dense_image_features(tmp_path)
