from __future__ import annotations

from types import MethodType

import torch

from models.taper_mag.model import TaperMAG, TaperMAGConfig
from models.taper_mag.rollout import RolloutConfig
from training.taper_mag_losses import stop_anchored_listwise_utility_loss
from training.taper_mag_diagnostics import summarize_training_diagnostics
from test_taper_mag_actor import encoded_batch


def model() -> TaperMAG:
    return TaperMAG(
        TaperMAGConfig(text_dim=20, vision_dim=24, retrieval_dim=32, dropout=0, max_steps=4)
    ).eval()


def test_stop_forbidden_at_t0_then_per_sample_alive_and_repeat_allowed() -> None:
    taper = model()
    calls = {"count": 0}

    def controlled_forward(self, features, *, detach_inputs=False):
        del self, detach_inputs
        calls["count"] += 1
        values = torch.full(features.shape[:2], -2.0, device=features.device)
        values[0, 0] = 2.0  # sample zero repeatedly chooses action 0
        return values

    taper.utility.forward = MethodType(controlled_forward, taper.utility)
    output = taper(encoded_batch(), RolloutConfig(max_steps=3, selection_mode="learned"))
    assert output.trace.actions[0].tolist() == [0, 0, 0]
    assert output.trace.actions[1, 0].item() < 4  # STOP is impossible at t=0
    assert output.trace.actions[1, 1].item() == 4
    assert output.trace.active[1].tolist() == [True, True, False]
    assert calls["count"] == 3  # critic/candidates were recomputed at every state


def test_frozen_order_reuses_t0_values_while_dynamic_recomputes() -> None:
    taper = model()
    call = {"index": 0}

    def changing_forward(self, features, *, detach_inputs=False):
        del self, detach_inputs
        index = call["index"]
        call["index"] += 1
        values = torch.zeros(features.shape[:2], device=features.device)
        values[:, index % 4] = 2.0
        return values

    taper.utility.forward = MethodType(changing_forward, taper.utility)
    dynamic = taper(encoded_batch(), RolloutConfig(max_steps=3, selection_mode="learned"))
    assert dynamic.trace.actions[0].tolist() == [0, 1, 2]
    call["index"] = 0
    frozen = taper(encoded_batch(), RolloutConfig(max_steps=3, selection_mode="frozen_order"))
    assert frozen.trace.actions[0].tolist() == [0, 0, 0]


def test_closed_utility_backward_updates_only_critic() -> None:
    taper = model().train()
    _, state, operators = taper.prepare(encoded_batch())
    from models.taper_mag.utility import HistoryState

    history = HistoryState.initialize(2, 4, 6, state.local.device)
    _, _, _, predicted = taper.preview(
        state, operators, history, step=0, max_steps=1, detach_utility_inputs=True
    )
    loss = stop_anchored_listwise_utility_loss(predicted, torch.randn_like(predicted))
    taper.zero_grad(set_to_none=True)
    loss.backward()
    utility_grad = sum(
        float(parameter.grad.abs().sum())
        for parameter in taper.utility.parameters()
        if parameter.grad is not None
    )
    actor_grad = sum(
        float(parameter.grad.abs().sum())
        for name, parameter in taper.named_parameters()
        if not name.startswith("utility.") and parameter.grad is not None
    )
    assert utility_grad > 0
    assert actor_grad == 0


def test_required_diagnostics_are_finite() -> None:
    taper = model()
    output = taper(encoded_batch(), RolloutConfig(max_steps=2, selection_mode="learned"))
    diagnostics = summarize_training_diagnostics(
        output, torch.randn_like(output.trace.predicted_gain)
    )
    required = {
        "mean_action_count",
        "repeat_action_frequency",
        "candidate_query_variance",
        "oracle_best_gain",
        "critic_top1_agreement",
        "critic_regret",
        "text_attention_entropy",
        "visual_attention_entropy",
        "query_query_cosine_offdiag",
        "operator_operator_cosine_offdiag",
    }
    assert required.issubset(diagnostics)
    assert all(torch.isfinite(torch.tensor(value)) for value in diagnostics.values())
