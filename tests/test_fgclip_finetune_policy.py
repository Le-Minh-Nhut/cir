from __future__ import annotations

import torch
from torch.optim import AdamW

from losses.objective import IAGSRMEObjective, ObjectiveConfig
from models.iag_srme import IAGSRME, IAGSRMEConfig
from training.engine import (
    PrecisionPolicy,
    parameter_count_diagnostics,
    save_checkpoint,
    trainable_parameters,
)

from test_fgclip_v2_backbone import FGCLIPBackbone, MockFGCLIP


def _grad_sum(module: torch.nn.Module) -> float:
    return sum(
        float(parameter.grad.detach().abs().sum())
        for parameter in module.parameters()
        if parameter.grad is not None
    )


def _text_only_backbone(mode: str) -> FGCLIPBackbone:
    readout = "R0-QG" if mode == "learned_qg" else "R0-NCLS"
    return FGCLIPBackbone(
        MockFGCLIP(),
        text_width=4,
        train_vision=False,
        train_text=True,
        train_text_projection=False,
        global_readout_mode=mode,
        readout_experiment=readout,
        finetune_policy="text_only",
        experiment_identity=f"{readout}-TEXT",
    )


def _model(backbone: FGCLIPBackbone) -> IAGSRME:
    return IAGSRME(
        backbone,
        IAGSRMEConfig(
            width=4,
            num_candidates=3,
            max_steps=1,
            num_heads=2,
            exec_dim=4,
            stop_enabled=False,
            score_dropout=0.0,
        ),
    )


def test_full_policy_preserves_trainable_vision_and_text_baseline() -> None:
    backbone = FGCLIPBackbone(
        MockFGCLIP(),
        text_width=4,
        train_vision=True,
        train_text=True,
        train_text_projection=False,
        finetune_policy="full",
    )

    assert all(parameter.requires_grad for parameter in backbone.model.vision_model.parameters())
    assert all(
        parameter.requires_grad for parameter in backbone.model.visual_projection.parameters()
    )
    assert all(parameter.requires_grad for parameter in backbone.model.text_model.parameters())
    assert all(
        not parameter.requires_grad for parameter in backbone.model.text_projection.parameters()
    )


def test_text_only_freezes_visual_weights_but_keeps_text_and_tasks_trainable() -> None:
    backbone = _text_only_backbone("learned_qg")
    model = _model(backbone)
    backbone.train()

    assert all(
        not parameter.requires_grad for parameter in backbone.model.vision_model.parameters()
    )
    assert all(
        not parameter.requires_grad for parameter in backbone.model.visual_projection.parameters()
    )
    assert all(parameter.requires_grad for parameter in backbone.model.text_model.parameters())
    assert all(
        not parameter.requires_grad for parameter in backbone.model.text_projection.parameters()
    )
    assert all(parameter.requires_grad for parameter in backbone.text_adapter.parameters())
    assert backbone.q_G.requires_grad
    assert not backbone.model.vision_model.training
    assert not backbone.model.visual_projection.training
    assert backbone.model.text_model.training
    for module in (
        model.proposal,
        model.grounder,
        model.action_fusion,
        model.executor,
        model.score_net,
    ):
        assert all(parameter.requires_grad for parameter in module.parameters())


def test_native_cls_frozen_readout_backpropagates_to_candidate_state_and_executor() -> None:
    torch.manual_seed(31)
    backbone = _text_only_backbone("native_cls")
    model = _model(backbone).train()
    patches, cls_anchor = backbone.initial_state_with_anchor(torch.randn(2, 3, 2, 2))
    assert not patches.requires_grad
    assert not cls_anchor.requires_grad
    output = model.forward_from_features(
        patches,
        torch.randn(2, 5, 4),
        torch.randn(2, 4),
        torch.ones(2, 5, dtype=torch.bool),
        cls_anchor,
    )
    candidate_states = output["steps"][0]["candidate_states"]
    candidate_states.retain_grad()
    weights = torch.randn_like(output["steps"][0]["candidate_queries"])
    (output["steps"][0]["candidate_queries"] * weights).sum().backward()

    assert candidate_states.grad is not None and candidate_states.grad.abs().sum() > 0
    assert _grad_sum(model.executor) > 0
    assert _grad_sum(backbone.model.vision_model) == 0
    assert _grad_sum(backbone.model.visual_projection) == 0
    assert not backbone.q_G.requires_grad


def test_all_frozen_visual_readouts_remain_differentiable_with_respect_to_state() -> None:
    torch.manual_seed(311)
    backbone = _text_only_backbone("native_cls").train()
    state = torch.randn(2, 4, 6, requires_grad=True)
    anchor = torch.randn(2, 6)
    global_state = backbone.global_readout(state, anchor)
    dense = backbone.dense_readout(state)
    query = backbone.retrieval_from_global(global_state)
    loss = (
        (global_state * torch.randn_like(global_state)).sum()
        + (dense * torch.randn_like(dense)).sum()
        + (query * torch.randn_like(query)).sum()
    )
    loss.backward()

    assert state.grad is not None and state.grad.abs().sum() > 0
    assert _grad_sum(backbone.model.vision_model) == 0
    assert _grad_sum(backbone.model.visual_projection) == 0

    targets = backbone.encode_global_images(torch.randn(2, 3, 2, 2))
    assert not targets.requires_grad


def test_learned_qg_stays_trainable_through_frozen_visual_readout() -> None:
    torch.manual_seed(32)
    backbone = _text_only_backbone("learned_qg").train()
    state = torch.randn(2, 4, 6, requires_grad=True)
    query = backbone.retrieval_readout(state)
    (query * torch.randn_like(query)).sum().backward()

    assert backbone.q_G.grad is not None and backbone.q_G.grad.abs().sum() > 0
    assert state.grad is not None and state.grad.abs().sum() > 0
    assert _grad_sum(backbone.model.vision_model) == 0
    assert _grad_sum(backbone.model.visual_projection) == 0


def test_text_only_optimizer_excludes_frozen_visual_parameters() -> None:
    backbone = _text_only_backbone("learned_qg")
    model = _model(backbone)
    objective = IAGSRMEObjective(ObjectiveConfig())
    optimizer = AdamW(trainable_parameters(model, objective), lr=1e-3)
    optimizer_ids = {
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    }
    frozen_visual_ids = {
        id(parameter)
        for module in (backbone.model.vision_model, backbone.model.visual_projection)
        for parameter in module.parameters()
    }

    assert optimizer_ids.isdisjoint(frozen_visual_ids)
    assert id(backbone.q_G) in optimizer_ids
    assert all(
        id(parameter) in optimizer_ids for parameter in backbone.model.text_model.parameters()
    )
    assert all(id(parameter) in optimizer_ids for parameter in backbone.text_adapter.parameters())
    assert all(id(parameter) in optimizer_ids for parameter in model.executor.parameters())

    counts = parameter_count_diagnostics(model, objective)
    assert counts["trainable_vision_parameters"] == 0
    assert counts["trainable_visual_projection_parameters"] == 0
    assert counts["trainable_text_parameters"] > 0
    assert counts["trainable_text_adapter_parameters"] > 0
    assert counts["trainable_q_g_parameters"] > 0
    assert counts["trainable_iag_srme_parameters"] > 0

    native = _text_only_backbone("native_cls")
    native_model = _model(native)
    native_optimizer_ids = {
        id(parameter) for parameter in trainable_parameters(native_model, objective)
    }
    assert id(native.q_G) not in native_optimizer_ids


def test_checkpoint_records_explicit_finetune_policy(tmp_path) -> None:
    backbone = _text_only_backbone("learned_qg")
    model = _model(backbone)
    objective = IAGSRMEObjective(ObjectiveConfig(lambda_safe=0.1))
    optimizer = AdamW(trainable_parameters(model, objective), lr=1e-3)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path,
        model,
        objective,
        optimizer,
        epoch=0,
        metric=0.0,
        precision=PrecisionPolicy("fp32", False, None, False),
        optimizer_step=17,
        batch_step=19,
        run_metadata={"seed": 42, "gradient_accumulation": 1},
    )
    saved = torch.load(path, weights_only=True)
    metadata = saved["metadata"]

    assert metadata["global_readout_mode"] == "learned_qg"
    assert metadata["readout_experiment"] == "R0-QG"
    assert metadata["finetune_policy"] == "text_only"
    assert metadata["experiment_identity"] == "R0-QG-TEXT"
    assert metadata["train_vision"] is False
    assert metadata["train_text"] is True
    assert metadata["train_text_projection"] is False
    assert saved["optimizer_step"] == 17
    assert saved["batch_step"] == 19
    assert metadata["number_of_optimizer_updates"] == 17
    assert metadata["run"]["seed"] == 42
    assert metadata["objective_config"]["lambda_safe"] == 0.1
