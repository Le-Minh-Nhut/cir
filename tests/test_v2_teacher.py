from __future__ import annotations

import torch

from models.iag_srme.utils.retrieval import (
    build_teacher_masks,
    marginal_teacher_utilities,
    teacher_retrieval_loss,
)


def test_duplicate_targets_are_multi_positive_and_never_negative() -> None:
    positive, negative, valid = build_teacher_masks(
        ["11", "24", "11", "48"], torch.device("cpu")
    )
    assert valid.all()
    assert positive[0, 2] and positive[2, 0]
    assert not negative[0, 2] and not negative[2, 0]
    assert not (positive & negative).any()


def test_parent_and_all_siblings_share_pool_and_keep_has_zero_utility() -> None:
    torch.manual_seed(5)
    targets = torch.randn(3, 7)
    current = torch.randn(3, 7)
    candidates = current[:, None].expand(-1, 4, -1).clone()
    positive, negative, _ = build_teacher_masks(["a", "b", "c"], targets.device)
    utility, valid = marginal_teacher_utilities(
        current, candidates, targets, positive, negative, 0.1
    )
    assert valid.all()
    assert torch.equal(utility, torch.zeros_like(utility))
    assert not utility.requires_grad


def test_empty_negative_row_is_invalid_not_fabricated() -> None:
    targets = torch.randn(3, 7)
    current = torch.randn(3, 7)
    candidates = torch.randn(3, 4, 7)
    positive, negative, _ = build_teacher_masks(["same", "same", "same"], targets.device)
    _, valid = marginal_teacher_utilities(
        current, candidates, targets, positive, negative, 0.1
    )
    assert not negative.any()
    assert not valid.any()


def test_teacher_similarity_and_logsumexp_are_fp32() -> None:
    query = torch.randn(2, 5, dtype=torch.float16)
    targets = torch.randn(2, 5, dtype=torch.float16)
    positive, negative, _ = build_teacher_masks(["a", "b"], query.device)
    loss = teacher_retrieval_loss(query, targets, positive, negative, 0.1)
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss).all()
