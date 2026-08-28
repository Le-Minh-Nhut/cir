from __future__ import annotations

import torch

from models.taper_mag.contracts import SupervisionBatch
from models.taper_mag.model import TaperMAG, TaperMAGConfig
from models.taper_mag.rollout import RolloutConfig
from training.marginal_gain_teacher import MarginalGainTeacher
from training.negative_bank import NegativeBank
from training.taper_mag_losses import stop_anchored_listwise_utility_loss
from test_taper_mag_actor import encoded_batch


def supervision(target: torch.Tensor) -> SupervisionBatch:
    return SupervisionBatch(
        target_embedding=target,
        target_ids=("positive-a", "positive-b"),
        positive_ids=(("positive-a", "group-a"), ("positive-b",)),
    )


def test_teacher_detached_common_negatives_and_positive_filtering() -> None:
    torch.manual_seed(1)
    current = torch.nn.functional.normalize(torch.randn(2, 8, requires_grad=True), dim=-1)
    candidates = torch.nn.functional.normalize(torch.randn(2, 4, 8, requires_grad=True), dim=-1)
    target = torch.nn.functional.normalize(torch.randn(2, 8), dim=-1)
    bank_ids = ("positive-a", "negative-1", "group-a", "negative-2", "negative-3")
    bank = NegativeBank(torch.randn(5, 8), bank_ids, hard_negatives=3)
    negative_set = bank.mine_once(current, supervision(target))
    for row, positives in zip(negative_set.ids, (("positive-a", "group-a"), ("positive-b",)), strict=True):
        assert set(row).isdisjoint(positives)
    output = MarginalGainTeacher().score(
        current, candidates, supervision(target), negative_set
    )
    assert not output.raw_gain.requires_grad
    assert output.net_values.shape == (2, 5)
    assert (output.net_values[:, -1] == 0).all()
    # A single [B,H] ID set is returned, not action-specific negative IDs.
    assert output.negative_ids == negative_set.ids


def test_target_shuffle_changes_teacher_not_target_free_policy_inputs() -> None:
    torch.manual_seed(2)
    current = torch.nn.functional.normalize(torch.randn(2, 8), dim=-1)
    candidates = torch.nn.functional.normalize(torch.randn(2, 4, 8), dim=-1)
    target = torch.nn.functional.normalize(torch.randn(2, 8), dim=-1)
    bank = NegativeBank(torch.randn(4, 8), ("n0", "n1", "n2", "n3"), hard_negatives=3)
    sup = supervision(target)
    negatives = bank.mine_once(current, sup)
    teacher = MarginalGainTeacher()
    original = teacher.score(current, candidates, sup, negatives).raw_gain
    shuffled_sup = supervision(target.flip(0))
    shuffled = teacher.score(current, candidates, shuffled_sup, negatives).raw_gain
    assert not torch.allclose(original, shuffled)
    model = TaperMAG(
        TaperMAGConfig(text_dim=20, vision_dim=24, retrieval_dim=32, dropout=0, max_steps=2)
    ).eval()
    policy = encoded_batch()
    before = model(policy, RolloutConfig(max_steps=2))
    # Shuffle only supervision; the inference call has no target argument/resource.
    del shuffled_sup
    after = model(policy, RolloutConfig(max_steps=2))
    torch.testing.assert_close(before.final_query, after.final_query)
    torch.testing.assert_close(before.trace.predicted_gain, after.trace.predicted_gain)
    assert torch.equal(before.trace.actions, after.trace.actions)
    state_keys = tuple(model.state_dict())
    assert not any(token in key for key in state_keys for token in ("teacher", "target", "negative_bank"))


def test_utility_loss_has_closed_teacher_gradient() -> None:
    predicted = torch.randn(3, 4, requires_grad=True)
    teacher = torch.randn(3, 4, requires_grad=True)
    loss = stop_anchored_listwise_utility_loss(predicted, teacher)
    loss.backward()
    assert predicted.grad is not None and predicted.grad.abs().sum() > 0
    assert teacher.grad is None
