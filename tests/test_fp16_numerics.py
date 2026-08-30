from __future__ import annotations

import torch
from entmax import entmax15
from pytest import MonkeyPatch
from torch import Tensor, nn

from losses.complementary_claim import ComplementaryClaimLoss
from losses.factor import FactorCompletenessLoss
from losses.marginal import entmax15_fenchel_young
from losses.retrieval import TerminalRetrievalLoss
import models.iag_srme.grounding as grounding_module
from models.iag_srme.grounding import AnchorGrounder
from models.iag_srme.readout import cap_vector
from numerics import normalize_fp32


def test_fp16_grounding_entmax_is_finite_normalized_and_sparse(
    monkeypatch: MonkeyPatch,
) -> None:
    torch.manual_seed(181)
    entmax_input_dtypes: list[torch.dtype] = []

    def recording_entmax(logits: Tensor, dim: int) -> Tensor:
        entmax_input_dtypes.append(logits.dtype)
        return entmax15(logits, dim=dim)

    monkeypatch.setattr(grounding_module, "entmax15", recording_entmax)
    grounder = AnchorGrounder(width=4).half()
    intents = (20 * torch.randn(2, 4, 4)).half().requires_grad_()
    anchor = (20 * torch.randn(2, 9, 4)).half().requires_grad_()
    supports = grounder(intents, anchor)

    assert entmax_input_dtypes == [torch.float32]
    assert supports.dtype is torch.float16
    assert torch.isfinite(supports).all()
    assert torch.allclose(supports.sum(dim=-1).float(), torch.ones(2, 4), atol=1e-3)
    assert supports.eq(0).any()
    supports.square().sum().backward()
    assert intents.grad is not None and torch.isfinite(intents.grad).all()


def test_fp16_retrieval_and_fenchel_young_use_finite_fp32_arithmetic() -> None:
    torch.manual_seed(191)
    queries = torch.randn(4, 16, dtype=torch.float16, requires_grad=True)
    targets = torch.randn(4, 16, dtype=torch.float16, requires_grad=True)
    terminal = TerminalRetrievalLoss()(queries, targets)
    scores = torch.tensor(
        [[2.0, -2.0, 0.0, 1.0, -1.0]], dtype=torch.float16, requires_grad=True
    )
    target_distribution = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0, 1.0]], dtype=torch.float16
    )
    marginal = entmax15_fenchel_young(scores, target_distribution).mean()

    assert terminal.dtype is torch.float32 and torch.isfinite(terminal)
    assert marginal.dtype is torch.float32 and torch.isfinite(marginal)
    (terminal + marginal).backward()
    for gradient in (queries.grad, targets.grad, scores.grad):
        assert gradient is not None and torch.isfinite(gradient).all()
    expected_score_gradient = entmax15(scores.detach().float(), dim=-1) - target_distribution.float()
    assert torch.allclose(scores.grad.float(), expected_score_gradient, atol=1e-3, rtol=1e-3)


def test_fp16_auxiliary_log_kl_and_normalization_islands_are_finite() -> None:
    torch.manual_seed(193)
    claims = torch.sigmoid(torch.randn(3, 4, 7, dtype=torch.float16, requires_grad=True))
    mask = torch.ones(3, 7, dtype=torch.bool)
    complementary = ComplementaryClaimLoss()(claims, mask).loss
    factors = torch.randn(3, 4, 8, dtype=torch.float16, requires_grad=True)
    anchors = torch.randn(3, 8, dtype=torch.float16, requires_grad=True)
    factor, _ = FactorCompletenessLoss()(factors, anchors)

    assert complementary.dtype is torch.float32 and torch.isfinite(complementary)
    assert factor.dtype is torch.float32 and torch.isfinite(factor)
    (complementary + factor).backward()
    assert factors.grad is not None and torch.isfinite(factors.grad).all()
    # The semantic factor anchor is intentionally detached by the factor objective.
    assert anchors.grad is None


def test_fp16_cap_normalization_and_layernorm_remain_finite_at_small_scale() -> None:
    values = torch.tensor([[1e-7, -1e-7, 0.0]], dtype=torch.float16)
    capped = cap_vector(values, cap=0.5)
    normalized = normalize_fp32(values, dim=-1)
    layer_norm = nn.LayerNorm(3).half()(values)

    for tensor in (capped, normalized, layer_norm):
        assert tensor.dtype is torch.float16
        assert torch.isfinite(tensor).all()
