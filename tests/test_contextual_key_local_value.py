from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from cache.features import get_text_features_by_sample_ids, load_text_features
from models.taper import TAPER


class DummyComposeTeacher(nn.Module):
    """Small differentiable stand-in for the frozen CSMCIR compose teacher."""

    def __init__(self, query_dim: int) -> None:
        super().__init__()
        self.query_dim = query_dim
        self.scale = nn.Parameter(torch.ones(query_dim))

    def compose(
        self,
        reference_features: Tensor,
        text_states: Tensor,
        attention_mask: Tensor,
        *,
        normalize: bool = False,
    ) -> Tensor:
        mask = attention_mask.to(text_states.dtype).unsqueeze(-1)
        pooled = (text_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        reference_scale = 1.0 + 0.1 * reference_features.mean(dim=(1, 2)).unsqueeze(1)
        query = pooled[:, : self.query_dim] * reference_scale * self.scale
        if normalize:
            query = torch.nn.functional.normalize(query, dim=-1)
        return query


def make_model() -> TAPER:
    torch.manual_seed(0)
    model = TAPER(
        DummyComposeTeacher(query_dim=3),
        text_dim=5,
        reference_dim=7,
        teacher_text_dim=6,
        teacher_query_dim=3,
        query_dim=3,
        slot_dim=4,
        state_dim=4,
        num_slots=2,
        num_primitives=2,
        counterfactual_chunk_size=2,
        slot_value_source="teacher_raw",
        slot_effect_in_value=False,
    )
    with torch.no_grad():
        model.slot_queries.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0, 0.0],
                ]
            )
        )
        model.slot_query_projection.weight.copy_(torch.eye(4))
        model.text_key_projection.weight.zero_()
        model.text_key_projection.weight[0, 0] = 1.0
    return model


def make_inputs() -> dict[str, Tensor]:
    contextual = torch.zeros(1, 5, 5)
    contextual[0, :, 0] = torch.tensor([0.0, 4.0, -4.0, 1.0, 0.0])
    teacher_raw = torch.tensor(
        [
            [
                [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
                [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                [200.0, 200.0, 200.0, 200.0, 200.0, 200.0],
            ]
        ]
    )
    return {
        "reference_features": torch.randn(1, 7),
        "text_states": contextual,
        "text_attention_mask": torch.tensor([[True, True, True, True, False]]),
        "text_content_mask": torch.tensor([[False, True, True, True, False]]),
        "teacher_reference_features": torch.randn(1, 2, 7),
        "teacher_text_states": teacher_raw,
    }


def build_slots(model: TAPER, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
    return model.build_edit_slots(
        inputs["reference_features"],
        inputs["text_states"],
        inputs["text_attention_mask"],
        text_content_mask=inputs["text_content_mask"],
        teacher_reference_features=inputs["teacher_reference_features"],
        teacher_text_states=inputs["teacher_text_states"],
    )


def clone_inputs(inputs: dict[str, Tensor]) -> dict[str, Tensor]:
    return {name: value.clone() for name, value in inputs.items()}


def test_experiment_contract_and_slot_mlp_input_dimension() -> None:
    model = make_model()

    assert model.experiment_provenance() == {
        "slot_value_source": "teacher_raw",
        "slot_effect_in_value": False,
    }
    assert isinstance(model.slot_mlp[0], nn.Linear)
    assert model.slot_mlp[0].in_features == model.teacher_text_dim == 6
    assert model.slot_mlp[0].in_features != model.text_dim + model.teacher_query_dim

    with pytest.raises(ValueError, match="slot_value_source"):
        TAPER(
            DummyComposeTeacher(3),
            text_dim=5,
            reference_dim=7,
            teacher_text_dim=6,
            teacher_query_dim=3,
            query_dim=3,
            slot_value_source="contextual",
        )
    with pytest.raises(ValueError, match="slot_effect_in_value"):
        TAPER(
            DummyComposeTeacher(3),
            text_dim=5,
            reference_dim=7,
            teacher_text_dim=6,
            teacher_query_dim=3,
            query_dim=3,
            slot_effect_in_value=True,
        )


def test_text_cache_preserves_contextual_raw_token_alignment(tmp_path: Path) -> None:
    states = np.arange(2 * 5 * 5, dtype=np.float32).reshape(2, 5, 5)
    teacher_states = np.arange(2 * 5 * 6, dtype=np.float32).reshape(2, 5, 6)
    attention = np.array(
        [[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]],
        dtype=np.bool_,
    )
    content = np.array(
        [[0, 1, 1, 1, 0], [0, 1, 1, 0, 0]],
        dtype=np.bool_,
    )
    np.save(tmp_path / "states.npy", states)
    np.save(tmp_path / "teacher_states.npy", teacher_states)
    np.save(tmp_path / "attention_mask.npy", attention)
    np.save(tmp_path / "content_mask.npy", content)
    (tmp_path / "sample_to_idx.json").write_text(
        json.dumps({"a": 0, "b": 1}),
        encoding="utf-8",
    )
    (tmp_path / "captions.json").write_text(
        json.dumps({"a": "caption a", "b": "caption b"}),
        encoding="utf-8",
    )
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")

    cache = load_text_features(tmp_path)
    contextual, raw, loaded_attention, loaded_content = (
        get_text_features_by_sample_ids(
            ["b", "a"],
            ["caption b", "caption a"],
            cache,
        )
    )

    assert contextual.shape == (2, 5, 5)
    assert raw.shape == (2, 5, 6)
    torch.testing.assert_close(contextual[0], torch.from_numpy(states[1]))
    torch.testing.assert_close(raw[0], torch.from_numpy(teacher_states[1]))
    torch.testing.assert_close(loaded_attention[0], torch.from_numpy(attention[1]))
    torch.testing.assert_close(loaded_content[0], torch.from_numpy(content[1]))


def test_contextual_key_and_teacher_raw_value_are_separate() -> None:
    model = make_model()
    inputs = make_inputs()
    baseline = build_slots(model, inputs)

    raw_changed = clone_inputs(inputs)
    raw_changed["teacher_text_states"][0, 1, :] += 50.0
    changed_value = build_slots(model, raw_changed)

    torch.testing.assert_close(
        changed_value["ownership_logits"], baseline["ownership_logits"]
    )
    torch.testing.assert_close(changed_value["slot_masks"], baseline["slot_masks"])
    torch.testing.assert_close(
        changed_value["qasa_attention"], baseline["qasa_attention"]
    )
    assert not torch.allclose(changed_value["slot_semantics"], baseline["slot_semantics"])
    assert not torch.allclose(changed_value["edit_slots"], baseline["edit_slots"])

    contextual_changed = clone_inputs(inputs)
    contextual_changed["text_states"][0, 1, 0] = -8.0
    changed_key = build_slots(model, contextual_changed)

    assert not torch.allclose(changed_key["ownership_logits"], baseline["ownership_logits"])
    assert not torch.allclose(changed_key["slot_masks"], baseline["slot_masks"])
    assert not torch.allclose(changed_key["qasa_attention"], baseline["qasa_attention"])


def test_slot_mlp_receives_semantics_only_and_ignores_slot_effects() -> None:
    model = make_model()
    inputs = make_inputs()
    captured_inputs: list[Tensor] = []

    def capture_input(_module: nn.Module, args: tuple[Tensor, ...]) -> None:
        captured_inputs.append(args[0].detach().clone())

    handle = model.slot_mlp[0].register_forward_pre_hook(capture_input)
    baseline = build_slots(model, inputs)
    handle.remove()

    assert len(captured_inputs) == 1
    assert captured_inputs[0].shape[-1] == model.teacher_text_dim
    torch.testing.assert_close(captured_inputs[0], baseline["slot_semantics"])

    reference_changed = clone_inputs(inputs)
    reference_changed["teacher_reference_features"] += 100.0
    changed = build_slots(model, reference_changed)

    assert not torch.allclose(changed["slot_effects"], baseline["slot_effects"])
    torch.testing.assert_close(changed["slot_semantics"], baseline["slot_semantics"])
    torch.testing.assert_close(changed["raw_edit_slots"], baseline["raw_edit_slots"])
    torch.testing.assert_close(changed["edit_slots"], baseline["edit_slots"])
    assert not bool(baseline["slot_effect_used_in_latent"].item())


def test_fixed_support_blocks_contextual_information_from_value() -> None:
    model = make_model()
    inputs = make_inputs()
    fixed_masks = torch.tensor(
        [
            [
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 0.0],
            ]
        ]
    )
    fixed_logits = torch.zeros_like(fixed_masks)

    with patch.object(
        model,
        "_competitive_ownership",
        return_value=(fixed_logits, fixed_masks),
    ):
        baseline = build_slots(model, inputs)

        contextual_changed = clone_inputs(inputs)
        contextual_changed["text_states"] += torch.randn_like(
            contextual_changed["text_states"]
        ) * 100.0
        changed_context = build_slots(model, contextual_changed)
        torch.testing.assert_close(
            changed_context["slot_semantics"], baseline["slot_semantics"]
        )
        torch.testing.assert_close(
            changed_context["raw_edit_slots"], baseline["raw_edit_slots"]
        )
        torch.testing.assert_close(changed_context["edit_slots"], baseline["edit_slots"])

        support_changed = clone_inputs(inputs)
        support_changed["teacher_text_states"][0, 1, :] += 20.0
        changed_support = build_slots(model, support_changed)
        assert not torch.allclose(
            changed_support["slot_semantics"][:, 0],
            baseline["slot_semantics"][:, 0],
        )
        torch.testing.assert_close(
            changed_support["slot_semantics"][:, 1],
            baseline["slot_semantics"][:, 1],
        )

        outside_slot_zero = clone_inputs(inputs)
        outside_slot_zero["teacher_text_states"][0, 3, :] += 30.0
        changed_outside = build_slots(model, outside_slot_zero)
        torch.testing.assert_close(
            changed_outside["slot_semantics"][:, 0],
            baseline["slot_semantics"][:, 0],
        )


def test_zero_slot_and_invalid_tokens_keep_existing_contract() -> None:
    model = make_model()
    inputs = make_inputs()
    zero_masks = torch.zeros(1, model.num_slots, 5)
    semantics, mass, activity = model._mass_aware_slot_pool(
        inputs["teacher_text_states"],
        zero_masks,
    )
    edit_slots = model.slot_mlp(semantics) * activity.unsqueeze(-1)

    assert torch.equal(mass, torch.zeros_like(mass))
    assert torch.equal(activity, torch.zeros_like(activity))
    assert torch.equal(semantics, torch.zeros_like(semantics))
    assert torch.equal(edit_slots, torch.zeros_like(edit_slots))

    output = build_slots(model, inputs)
    invalid = ~(
        inputs["text_attention_mask"].bool()
        & inputs["text_content_mask"].bool()
    )
    assert torch.equal(
        output["slot_masks"].masked_select(invalid[:, None, :]),
        torch.zeros_like(output["slot_masks"].masked_select(invalid[:, None, :])),
    )

    invalid_raw_changed = clone_inputs(inputs)
    invalid_raw_changed["teacher_text_states"][0, 0, :] += 10_000.0
    invalid_raw_changed["teacher_text_states"][0, 4, :] -= 10_000.0
    changed = build_slots(model, invalid_raw_changed)
    torch.testing.assert_close(changed["slot_semantics"], output["slot_semantics"])
    torch.testing.assert_close(changed["edit_slots"], output["edit_slots"])


def test_teacher_stays_frozen_and_local_value_path_backpropagates() -> None:
    model = make_model().train()
    inputs = make_inputs()
    output = build_slots(model, inputs)
    weights = torch.arange(
        output["edit_slots"].numel(),
        dtype=output["edit_slots"].dtype,
    ).reshape_as(output["edit_slots"])
    loss = (output["edit_slots"] * weights).sum()
    loss.backward()

    assert not model.teacher.training
    assert all(not parameter.requires_grad for parameter in model.teacher.parameters())
    assert all(parameter.grad is None for parameter in model.teacher.parameters())
    for parameter in (
        model.slot_queries,
        model.slot_query_projection.weight,
        model.text_key_projection.weight,
        model.slot_mlp[0].weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_full_compute_loss_forward_backward_smoke() -> None:
    model = make_model().train()
    single = make_inputs()
    batch_inputs = {
        name: torch.cat([value, value.clone()], dim=0)
        for name, value in single.items()
    }
    batch_inputs["reference_features"][1] += 0.5
    batch_inputs["teacher_reference_features"][1] -= 0.5
    batch_inputs["text_states"][1, 2, 0] += 2.0
    batch_inputs["teacher_text_states"][1, 2, :] += 3.0
    target_features = torch.randn(2, 1, model.query_dim)

    losses = model.compute_loss(
        {
            **batch_inputs,
            "target_features": target_features,
            "target_ids": ["target-a", "target-b"],
        }
    )
    retrieval_loss = losses["retrieval_loss"]
    assert retrieval_loss.ndim == 0
    assert torch.isfinite(retrieval_loss)
    assert losses["diagnostic/slot_value_source_teacher_raw"].item() == 1.0
    assert losses["diagnostic/slot_effect_used_in_latent"].item() == 0.0

    retrieval_loss.backward()
    trainable_gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert trainable_gradients
    assert all(torch.isfinite(gradient).all() for gradient in trainable_gradients)
    assert all(parameter.grad is None for parameter in model.teacher.parameters())


def test_a31_slot_mlp_shape_is_incompatible() -> None:
    model = make_model()
    a31_state = model.state_dict()
    a31_state["slot_mlp.0.weight"] = torch.randn(
        model.slot_dim,
        model.text_dim + model.teacher_query_dim,
    )

    with pytest.raises(RuntimeError, match="size mismatch for slot_mlp.0.weight"):
        model.load_state_dict(a31_state, strict=False)


def test_checkpoint_records_experiment_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    wandb_stub = types.ModuleType("wandb")
    wandb_stub.run = None
    wandb_stub.log = lambda _data: None
    monkeypatch.setitem(sys.modules, "wandb", wandb_stub)

    from training.engine import taper_checkpoint

    model = make_model()
    checkpoint = taper_checkpoint(model)
    assert checkpoint["experiment_provenance"] == model.experiment_provenance()
    assert checkpoint["model_state_dict"]
    assert not any(
        name.startswith("teacher.")
        for name in checkpoint["model_state_dict"]
    )

    path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, path)
    restored = torch.load(path, map_location="cpu", weights_only=True)
    assert restored["experiment_provenance"] == {
        "slot_value_source": "teacher_raw",
        "slot_effect_in_value": False,
    }


def test_evaluator_rejects_a31_and_wrong_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    omegaconf_stub = types.ModuleType("omegaconf")
    omegaconf_stub.OmegaConf = object
    monkeypatch.setitem(sys.modules, "omegaconf", omegaconf_stub)

    from evaluate_qasa_inference import load_checkpoint

    model = make_model()
    a31_state = model.state_dict()
    a31_state["slot_mlp.0.weight"] = torch.randn(
        model.slot_dim,
        model.text_dim + model.teacher_query_dim,
    )
    a31_path = tmp_path / "a31.pt"
    torch.save(a31_state, a31_path)
    with pytest.raises(RuntimeError, match="must be trained from scratch"):
        load_checkpoint(model, a31_path)

    model_state = {
        name: value
        for name, value in model.state_dict().items()
        if not name.startswith("teacher.")
    }
    wrong_path = tmp_path / "wrong-provenance.pt"
    torch.save(
        {
            "model_state_dict": model_state,
            "experiment_provenance": {
                "slot_value_source": "contextual",
                "slot_effect_in_value": True,
            },
        },
        wrong_path,
    )
    with pytest.raises(RuntimeError, match="provenance mismatch"):
        load_checkpoint(model, wrong_path)

    matching_path = tmp_path / "matching.pt"
    torch.save(
        {
            "model_state_dict": model_state,
            "experiment_provenance": model.experiment_provenance(),
        },
        matching_path,
    )
    load_checkpoint(model, matching_path)
