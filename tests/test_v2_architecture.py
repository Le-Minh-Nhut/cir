from __future__ import annotations

import inspect

import torch

from models.iag_srme import ActionFusion, Executor, Grounder, ProposalNet


def test_proposal_uses_text_and_initial_global_but_has_no_patch_input() -> None:
    torch.manual_seed(1)
    proposal = ProposalNet(16, 8, 4, 4).eval()
    tokens = torch.randn(2, 5, 8)
    text = torch.randn(2, 8)
    global_state = torch.randn(2, 16)
    mask = torch.ones(2, 5, dtype=torch.bool)

    baseline = proposal(tokens, text, global_state, mask)
    assert not torch.allclose(baseline, proposal(tokens.flip(0), text, global_state, mask))
    assert not torch.allclose(baseline, proposal(tokens, text + 1.0, global_state, mask))
    assert not torch.allclose(baseline, proposal(tokens, text, global_state + 1.0, mask))
    assert tuple(inspect.signature(proposal.forward).parameters) == (
        "text_tokens",
        "text_global",
        "current_global",
        "content_mask",
    )


def test_grounder_reads_the_current_state_for_one_edit() -> None:
    torch.manual_seed(2)
    grounder = Grounder(16, 12).eval()
    edit = torch.randn(2, 1, 16)
    dense = torch.randn(2, 9, 12)
    state = torch.randn(2, 9, 16)
    _, alpha, support = grounder(edit, dense)
    entity = torch.einsum("bsn,bnd->bsd", alpha, state)

    assert alpha.shape == support.shape == (2, 1, 9)
    assert torch.allclose(alpha.sum(-1), torch.ones(2, 1), atol=1e-6)
    assert not torch.allclose(alpha.sum(-1), support.sum(-1))
    assert torch.allclose(entity, alpha @ state)


def test_action_fusion_keeps_independent_bounded_gates() -> None:
    torch.manual_seed(3)
    fusion = ActionFusion(16).eval()
    entity = torch.randn(2, 1, 16)
    edit = torch.randn(2, 1, 16)
    action, gamma, beta = fusion(entity, edit)

    assert action.shape == gamma.shape == beta.shape == (2, 1, 16)
    assert ((gamma > 0) & (gamma < 1) & (beta > 0) & (beta < 1)).all()
    assert not torch.allclose(gamma + beta, torch.ones_like(gamma))


def test_executor_reuses_one_slot_dimension_and_zero_mask_is_identity() -> None:
    torch.manual_seed(4)
    executor = Executor(16, 16, 8).eval()
    parent = torch.randn(2, 9, 16)
    actions = torch.randn(2, 1, 16)
    support = torch.rand(2, 1, 9)
    raw, delta, next_states = executor(parent, actions, support, (3, 3))

    assert raw.shape == delta.shape == next_states.shape == (2, 1, 9, 16)
    assert torch.equal(next_states, parent[:, None] + delta)
    _, zero_delta, unchanged = executor(
        parent, actions, torch.zeros_like(support), (3, 3)
    )
    assert torch.equal(zero_delta, torch.zeros_like(zero_delta))
    assert torch.equal(unchanged[:, 0], parent)


def test_proposal_runs_once_and_all_slots_use_one_shared_executor(
    model, features, monkeypatch
) -> None:
    proposal_calls = 0
    executor_parents: list[torch.Tensor] = []
    original_proposal = model.proposal.forward
    original_executor = model.executor.forward

    def counted_proposal(*args, **kwargs):
        nonlocal proposal_calls
        proposal_calls += 1
        return original_proposal(*args, **kwargs)

    def counted_executor(parent, *args, **kwargs):
        executor_parents.append(parent.detach().clone())
        return original_executor(parent, *args, **kwargs)

    monkeypatch.setattr(model.proposal, "forward", counted_proposal)
    monkeypatch.setattr(model.executor, "forward", counted_executor)
    initial = features[0]
    output = model.forward_from_features(*features)

    assert proposal_calls == 1
    assert len(executor_parents) == model.config.num_context_edits == 4
    assert len(output["steps"]) == 4
    assert sum(isinstance(module, Executor) for module in model.modules()) == 1
    assert torch.equal(executor_parents[0], initial)
    for slot, step in enumerate(output["steps"]):
        assert step["slot"] == slot
        assert torch.equal(step["edit"], output["context_edits"][:, slot])
        assert step["state"].shape == initial.shape
        assert torch.equal(step["parent_state"], executor_parents[slot])
        if slot > 0:
            assert torch.equal(step["parent_state"], output["steps"][slot - 1]["state"])
    assert not torch.allclose(output["steps"][1]["parent_state"], initial)
    assert torch.equal(output["state"], output["steps"][-1]["state"])


def test_sequential_model_has_no_selector_stop_or_target_input(model, features) -> None:
    output = model.eval().forward_from_features(*features)

    assert not hasattr(model, "score_net")
    assert not hasattr(model.config, "stop_enabled")
    assert not hasattr(model.config, "epsilon_stop")
    assert "target" not in inspect.signature(model.forward).parameters
    for step in output["steps"]:
        assert not {
            "scores",
            "best_score",
            "best_idx",
            "selected_idx",
            "stopped_now",
        } & step.keys()


def test_final_output_shapes_have_no_slot_axis(model, features) -> None:
    output = model.forward_from_features(*features)

    assert output["query"].shape == (3, 12)
    assert output["state"].shape == (3, 9, 16)
    assert all(step["state"].shape == (3, 9, 16) for step in output["steps"])
