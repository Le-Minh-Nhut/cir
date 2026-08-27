from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader

from cache.features import get_dense_features_by_ids, load_dense_image_features
from models.entity_action_binding import (
    EntityActionBindingCIR,
    FeatureWiseAffineFusion,
    SharedRelationBinder,
)
from models.retrieval import multi_positive_retrieval_loss, target_positive_mask
from training.entity_action_binding import fit_entity_action_binding


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
    reference_tokens = output["reference_image_tokens"]
    text_tokens = output["text_tokens"]
    torch.testing.assert_close(
        reference_tokens.norm(dim=-1),
        torch.ones_like(reference_tokens[..., 0]),
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        text_tokens.norm(dim=-1),
        torch.ones_like(text_tokens[..., 0]),
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        reference_tokens[:, :1].norm(dim=-1),
        reference_tokens[:, 1:].norm(dim=-1).mean(dim=-1, keepdim=True),
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        text_tokens[:, :1].norm(dim=-1),
        text_tokens[:, 1:].norm(dim=-1).mean(dim=-1, keepdim=True),
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(output["query"].norm(dim=-1), torch.ones(2))
    torch.testing.assert_close(output["target_embedding"].norm(dim=-1), torch.ones(2))

    diagnostics = model.diagnostics(
        output,
        inputs["reference_dense_mask"],
        inputs["text_content_mask"],
    )
    for name in (
        "diagnostic/reference_global_token_norm",
        "diagnostic/entity_token_norm_mean",
        "diagnostic/text_global_token_norm",
        "diagnostic/action_token_norm_mean",
    ):
        torch.testing.assert_close(diagnostics[name], torch.ones_like(diagnostics[name]))


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


def test_empty_relation_tokens_remain_zero_at_composition_interface() -> None:
    model = EntityActionBindingCIR(dim=8, num_relations=3, fusion_hidden_dim=5)
    image = model.encode_image(
        torch.randn(2, 8),
        torch.randn(2, 4, 8),
        torch.zeros(2, 4, dtype=torch.bool),
    )
    text = model.encode_text(
        torch.randn(2, 8),
        torch.randn(2, 5, 8),
        torch.zeros(2, 5, dtype=torch.bool),
    )
    torch.testing.assert_close(image["tokens"][:, 0].norm(dim=-1), torch.ones(2))
    torch.testing.assert_close(text["tokens"][:, 0].norm(dim=-1), torch.ones(2))
    assert torch.equal(image["tokens"][:, 1:], torch.zeros_like(image["tokens"][:, 1:]))
    assert torch.equal(text["tokens"][:, 1:], torch.zeros_like(text["tokens"][:, 1:]))


def test_raw_binder_scale_is_normalized_only_at_composition_interface() -> None:
    torch.manual_seed(19)
    model = EntityActionBindingCIR(dim=16, num_relations=4, fusion_hidden_dim=12)
    inputs = _inputs()
    raw_entity, _ = model.binder(
        inputs["reference_dense"], inputs["reference_dense_mask"]
    )
    raw_action, _ = model.binder(inputs["text_states"], inputs["text_content_mask"])
    assert not torch.allclose(raw_entity.norm(dim=-1), torch.ones(2, 4))
    assert not torch.allclose(raw_action.norm(dim=-1), torch.ones(2, 4))

    image = model.encode_image(
        inputs["reference_global"],
        inputs["reference_dense"],
        inputs["reference_dense_mask"],
    )
    text = model.encode_text(
        inputs["text_global"], inputs["text_states"], inputs["text_content_mask"]
    )
    torch.testing.assert_close(
        image["tokens"].norm(dim=-1), torch.ones(2, 5), atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        text["tokens"].norm(dim=-1), torch.ones(2, 5), atol=1e-5, rtol=1e-5
    )


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


def test_training_persists_all_epoch_metrics_without_wandb(tmp_path: Path) -> None:
    class ToyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))

        def experiment_provenance(self) -> dict[str, object]:
            return {"architecture": "test"}

        def compute_loss(self, _batch: dict[str, object]) -> dict[str, torch.Tensor]:
            return {
                "retrieval_loss": self.weight.square(),
                "diagnostic/token_scale": self.weight.detach() * 0 + 1,
            }

    model = ToyModel()
    fit_entity_action_binding(
        model,
        DataLoader([0], batch_size=1),
        SGD(model.parameters(), lr=0.1),
        lambda _model: {
            "recall_at_10": 10.0,
            "recall_at_50": 20.0,
            "mean_recall": 15.0,
        },
        lambda _batch, _device: {},
        num_epochs=1,
        device=torch.device("cpu"),
        loss_weights={"retrieval_loss": 1.0},
        output_dir=tmp_path,
        use_amp=False,
    )
    rows = (tmp_path / "training_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    metrics = json.loads(rows[0])
    assert metrics["epoch"] == 1
    assert metrics["train/total_loss"] == pytest.approx(1.0)
    assert metrics["train/retrieval_loss"] == pytest.approx(1.0)
    assert metrics["train/diagnostic/token_scale"] == pytest.approx(1.0)
    assert metrics["val/recall_at_10"] == pytest.approx(10.0)
    assert metrics["val/recall_at_50"] == pytest.approx(20.0)
    assert metrics["val/mean_recall"] == pytest.approx(15.0)
