import torch

from losses.action_claim_binding import ActionClaimBindingLoss
from losses.complementary_claim import ComplementaryClaimLoss, claim_weighted_text_pool
from losses.factor import FactorCompletenessLoss
from losses.marginal import detached_marginal_utilities, entmax15_fenchel_young
from losses.retrieval import TerminalRetrievalLoss
from losses.unique import UniqueContributionLoss


def test_marginal_utility_is_detached_from_construction_and_target() -> None:
    current = torch.randn(3, 12, requires_grad=True)
    candidates = torch.randn(3, 4, 12, requires_grad=True)
    targets = torch.randn(3, 12, requires_grad=True)
    positive = torch.eye(3, dtype=torch.bool)
    utilities = detached_marginal_utilities(current, candidates, targets, positive, 0.1)
    assert not utilities.requires_grad
    scores = torch.randn(3, 5, requires_grad=True)
    target_distribution = torch.softmax(utilities, dim=-1)
    entmax15_fenchel_young(scores, target_distribution).mean().backward()
    assert scores.grad is not None
    assert current.grad is None and candidates.grad is None and targets.grad is None


def test_all_six_losses_are_finite() -> None:
    torch.manual_seed(5)
    batch, candidates, length, width = 4, 4, 7, 16
    queries = torch.randn(batch, 12, requires_grad=True)
    targets = torch.randn(batch, 12, requires_grad=True)
    terminal = TerminalRetrievalLoss()(queries, targets)
    claims = torch.sigmoid(torch.randn(batch, candidates, length, requires_grad=True))
    mask = torch.ones(batch, length, dtype=torch.bool)
    comp = ComplementaryClaimLoss()(claims, mask)
    text = torch.randn(batch, length, width, requires_grad=True)
    intents = torch.randn(batch, candidates, width, requires_grad=True)
    pooled = claim_weighted_text_pool(claims, text, mask)
    bind = ActionClaimBindingLoss(width=width)(intents, pooled)
    factors = torch.randn(batch, candidates, width, requires_grad=True)
    anchor = torch.randn(batch, width, requires_grad=True)
    factor, geometry = FactorCompletenessLoss()(factors, anchor)
    unique, _ = UniqueContributionLoss()(geometry)
    score = torch.randn(batch, candidates + 1, requires_grad=True)
    marginal = entmax15_fenchel_young(score, torch.softmax(torch.randn_like(score), -1)).mean()
    all_losses = torch.stack([terminal, marginal, comp.loss, bind, factor, unique])
    assert torch.isfinite(all_losses).all()
    all_losses.sum().backward()
    assert factors.grad is not None
    # Auxiliary relational anchor is intentionally detached in factor/unique.
    assert anchor.grad is None


def test_claim_padding_has_zero_mass() -> None:
    claims = torch.full((2, 4, 5), 0.5)
    mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 0, 0]], dtype=torch.bool)
    result = ComplementaryClaimLoss()(claims, mask)
    assert torch.equal(result.normalized_claims.masked_select(~mask[:, None]), torch.zeros(20))

