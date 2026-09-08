from __future__ import annotations

import inspect
from dataclasses import replace

import torch

from models.iag_srme import ActionFusion, Executor, Grounder, ProposalNet, ScoreNet


def test_forward_exposes_exact_candidate_global_used_for_query(model, features) -> None:
    state, tokens, text, mask = features
    output = model.forward_from_features(state, tokens, text, mask)

    for step in output["steps"]:
        torch.testing.assert_close(
            model.backbone.retrieval_from_global(step["candidate_global"]),
            step["candidate_queries"],
        )


def test_proposal_uses_text_and_current_global_but_has_no_patch_input() -> None:
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


def test_grounder_read_write_split_and_current_state_entity_pool() -> None:
    torch.manual_seed(2)
    grounder = Grounder(16, 12).eval()
    edit = torch.randn(2, 4, 16)
    dense = torch.randn(2, 9, 12)
    state = torch.randn(2, 9, 16)
    _, alpha, support = grounder(edit, dense)
    entity = torch.einsum("bkn,bnd->bkd", alpha, state)

    assert alpha.shape == support.shape == (2, 4, 9)
    assert torch.allclose(alpha.sum(-1), torch.ones(2, 4), atol=1e-6)
    assert not torch.allclose(alpha.sum(-1), support.sum(-1))
    assert torch.allclose(entity, alpha @ state)
    assert not torch.allclose(entity, alpha @ torch.randn_like(state))


def test_action_fusion_has_independent_bounded_gates_and_is_equivariant() -> None:
    torch.manual_seed(3)
    fusion = ActionFusion(16).eval()
    entity = torch.randn(2, 4, 16)
    edit = torch.randn(2, 4, 16)
    action, gamma, beta = fusion(entity, edit)
    permutation = torch.tensor([2, 0, 3, 1])
    permuted, _, _ = fusion(entity[:, permutation], edit[:, permutation])

    assert action.shape == gamma.shape == beta.shape == (2, 4, 16)
    assert ((gamma > 0) & (gamma < 1) & (beta > 0) & (beta < 1)).all()
    assert not torch.allclose(gamma + beta, torch.ones_like(gamma))
    assert torch.allclose(permuted, action[:, permutation])


def test_executor_identity_zero_mask_same_parent_and_permutation() -> None:
    torch.manual_seed(4)
    executor = Executor(16, 16, 8).eval()
    parent = torch.randn(2, 9, 16)
    actions = torch.randn(2, 4, 16)
    support = torch.rand(2, 4, 9)
    raw, delta, candidates = executor(parent, actions, support, (3, 3))

    assert torch.equal(raw, torch.zeros_like(raw))
    assert torch.equal(delta, torch.zeros_like(delta))
    assert torch.equal(candidates, parent[:, None].expand_as(candidates))

    torch.nn.init.normal_(executor.state_up.weight, std=0.02)
    torch.nn.init.normal_(executor.state_up.bias, std=0.02)
    _, _, no_write = executor(parent, actions, torch.zeros_like(support), (3, 3))
    _, _, written = executor(parent, actions, support, (3, 3))
    permutation = torch.tensor([3, 1, 0, 2])
    _, _, permuted = executor(parent, actions[:, permutation], support[:, permutation], (3, 3))
    assert torch.equal(no_write, parent[:, None].expand_as(no_write))
    assert torch.allclose(permuted, written[:, permutation])


def test_one_shared_scorenet_is_candidate_permutation_equivariant() -> None:
    torch.manual_seed(9)
    scorer = ScoreNet(16, 8, 16, dim=8, dropout=0.0).eval()
    current = torch.randn(2, 16)
    text = torch.randn(2, 8)
    actions = torch.randn(2, 4, 16)
    delta = torch.randn(2, 4, 9, 16)
    support = torch.rand(2, 4, 9)
    candidate_global = torch.randn(2, 4, 16)
    feature = scorer.build_features(current, text, actions, delta, support, candidate_global)
    score = scorer(feature)
    permutation = torch.tensor([2, 3, 1, 0])
    permuted_feature = scorer.build_features(
        current,
        text,
        actions[:, permutation],
        delta[:, permutation],
        support[:, permutation],
        candidate_global[:, permutation],
    )
    assert feature.shape == (2, 4, 40)
    assert torch.allclose(permuted_feature, feature[:, permutation])
    assert torch.allclose(scorer(permuted_feature), score[:, permutation])


def test_rollout_branches_from_one_parent_preserves_v0_and_reproposes(model, features) -> None:
    initial, tokens, text, mask = features
    torch.nn.init.normal_(model.executor.state_up.weight, std=0.03)
    with torch.no_grad():
        model.score_net.net[-1].weight.zero_()
        model.score_net.net[-1].bias.fill_(1.0)
    before = initial.clone()
    output = model.eval().forward_from_features(initial, tokens, text, mask)

    assert torch.equal(initial, before)
    assert torch.equal(output["initial_state"], before)
    assert len(output["steps"]) == 3
    for step in output["steps"]:
        assert torch.equal(step["candidate_states"], step["parent_state"][:, None] + step["delta"])
        assert step["stop_score"].eq(0).all()
    assert not torch.allclose(output["steps"][0]["proposals"], output["steps"][1]["proposals"])


def test_stop_is_exact_absorbing_keep_and_forward_is_target_free(model, features) -> None:
    initial, tokens, text, mask = features
    model.config = replace(model.config, stop_enabled=True)
    with torch.no_grad():
        model.score_net.net[-1].weight.zero_()
        model.score_net.net[-1].bias.fill_(-1.0)
    output = model.eval().forward_from_features(initial, tokens, text, mask)

    assert len(output["steps"]) == 1
    assert output["steps"][0]["selected_idx"].eq(4).all()
    assert output["steps"][0]["stop_score"].eq(0).all()
    assert output["stopped"].all()
    assert torch.equal(output["state"], initial)
    assert "target" not in inspect.signature(model.forward).parameters
