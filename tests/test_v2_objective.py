from __future__ import annotations

import torch

from losses.objective import IAGSRMEObjective, ObjectiveConfig


def _gradient_sum(module: torch.nn.Module) -> float:
    return sum(
        float(parameter.grad.detach().abs().sum())
        for parameter in module.parameters()
        if parameter.grad is not None
    )


def test_objective_is_terminal_retrieval_only(model, features) -> None:
    output = model.forward_from_features(*features)
    objective = IAGSRMEObjective(ObjectiveConfig())
    components = objective(output, torch.randn(3, 12), ["a", "b", "c"])

    assert torch.equal(components["total"], components["terminal"])
    assert all(
        name in {"total", "terminal"} or name.startswith("slot_")
        for name in components
    )


def test_terminal_gradient_crosses_every_sequential_edit_and_text_path(model) -> None:
    torch.manual_seed(23)
    reference_images = torch.randn(3, 3, 3, 3)
    target_images = torch.randn(3, 3, 3, 3)
    input_ids = torch.randint(0, 32, (3, 6))
    attention_mask = torch.ones(3, 6, dtype=torch.bool)
    content_mask = torch.ones(3, 6, dtype=torch.bool)
    output = model(reference_images, input_ids, attention_mask, content_mask)
    targets = model.encode_global_images(target_images)
    objective = IAGSRMEObjective(ObjectiveConfig())

    objective(output, targets, ["a", "b", "c"])["total"].backward()

    for module in (
        model.proposal,
        model.grounder,
        model.action_fusion,
        model.executor,
        model.backbone.text_embedding,
    ):
        assert _gradient_sum(module) > 0, type(module).__name__
        assert all(
            torch.isfinite(parameter.grad).all()
            for parameter in module.parameters()
            if parameter.grad is not None
        )


def test_terminal_loss_preserves_duplicate_target_positives(model, features) -> None:
    output = model.forward_from_features(*features)
    targets = torch.randn(3, 12)
    objective = IAGSRMEObjective(ObjectiveConfig())

    components = objective(output, targets, ["same", "other", "same"])

    assert torch.isfinite(components["terminal"])
    assert torch.isfinite(components["total"])


def test_slot_diagnostics_are_finite_and_cover_every_transition(model, features) -> None:
    output = model.forward_from_features(*features)
    objective = IAGSRMEObjective(ObjectiveConfig())
    components = objective(output, torch.randn(3, 12), ["a", "b", "c"])
    suffixes = {
        "delta_l2",
        "delta_patch_l2_mean",
        "exec_mask_mean",
        "exec_mask_support",
        "action_l2",
        "edit_l2",
        "state_drift_l2",
        "cumulative_drift_l2",
    }

    for slot in range(model.config.num_context_edits):
        for suffix in suffixes:
            value = components[f"slot_{slot}_{suffix}"]
            assert value.ndim == 0
            assert torch.isfinite(value)


def test_target_shuffle_changes_loss_but_not_target_free_forward(model, features) -> None:
    model.eval()
    first = model.forward_from_features(*features)
    second = model.forward_from_features(*features)
    assert torch.equal(first["query"], second["query"])
    assert torch.equal(first["state"], second["state"])

    targets = torch.randn(3, 12)
    objective = IAGSRMEObjective(ObjectiveConfig())
    normal = objective(first, targets, ["a", "b", "c"])["terminal"]
    shuffled = objective(first, targets.flip(0), ["c", "b", "a"])["terminal"]
    assert not torch.equal(normal, shuffled)
