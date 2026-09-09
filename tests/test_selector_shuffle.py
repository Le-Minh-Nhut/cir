from __future__ import annotations

from dataclasses import replace

import torch

import models.iag_srme.model as model_module
from models.iag_srme.model import ScoreNet, gather_candidate_axis


def test_selector_shuffle_scores_inverse_map_to_original_candidates() -> None:
    torch.manual_seed(17)
    scorer = ScoreNet(16, 8, 16, dim=8, dropout=0.0).eval()
    features = torch.randn(3, 4, 40)
    permutation = torch.tensor([[2, 0, 3, 1], [1, 3, 0, 2], [3, 2, 1, 0]])

    expected = scorer(features)
    shuffled = gather_candidate_axis(features, permutation)
    shuffled_scores = scorer(shuffled)
    restored = gather_candidate_axis(shuffled_scores, permutation.argsort(dim=-1))

    assert torch.allclose(restored, expected)


def test_selector_shuffle_is_train_only(model, features, monkeypatch) -> None:
    calls = 0

    def tracked_permutation(batch_size, num_candidates, device):
        nonlocal calls
        calls += 1
        return torch.arange(num_candidates, device=device).expand(batch_size, -1)

    monkeypatch.setattr(model_module, "random_candidate_permutation", tracked_permutation)
    model.config = replace(
        model.config, max_steps=1, selector_shuffle_enabled=True
    )

    model.eval().forward_from_features(*features)
    assert calls == 0

    model.train().forward_from_features(*features)
    assert calls == 1


def test_shuffle_preserves_original_selected_identity_and_hard_commit(
    model, features, monkeypatch
) -> None:
    permutation = torch.tensor([[2, 0, 3, 1], [1, 3, 0, 2], [3, 2, 1, 0]])

    def fixed_permutation(batch_size, num_candidates, device):
        assert (batch_size, num_candidates) == permutation.shape
        return permutation.to(device)

    monkeypatch.setattr(model_module, "random_candidate_permutation", fixed_permutation)
    model.config = replace(
        model.config,
        max_steps=1,
        stop_enabled=False,
        selector_shuffle_enabled=True,
    )
    torch.nn.init.normal_(model.executor.state_up.weight, std=0.03)
    torch.nn.init.normal_(model.executor.state_up.bias, std=0.03)

    unshuffled = model.eval().forward_from_features(*features)
    shuffled = model.train().forward_from_features(*features)
    baseline_step = unshuffled["steps"][0]
    shuffled_step = shuffled["steps"][0]

    assert torch.allclose(shuffled_step["scores"], baseline_step["scores"])
    assert torch.equal(shuffled_step["selected_idx"], baseline_step["selected_idx"])
    selected_states = model._gather_candidate(
        shuffled_step["candidate_states"], shuffled_step["selected_idx"]
    )
    assert torch.allclose(shuffled["state"], selected_states)
    assert torch.equal(
        shuffled_step["selected_idx"], shuffled_step["scores"].argmax(dim=-1)
    )


def test_shuffle_does_not_change_stop_or_target_free_inference_contract(
    model, features
) -> None:
    model.config = replace(
        model.config,
        max_steps=1,
        stop_enabled=True,
        selector_shuffle_enabled=True,
    )
    with torch.no_grad():
        model.score_net.net[-1].weight.zero_()
        model.score_net.net[-1].bias.fill_(-1.0)

    output = model.eval().forward_from_features(*features)

    assert output["steps"][0]["selected_idx"].eq(model.config.num_candidates).all()
    assert output["stopped"].all()
    assert torch.equal(output["state"], output["initial_state"])

