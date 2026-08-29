import torch

from losses.factor import kl_divergence
from losses.unique import leave_one_out_logits


def test_vectorized_leave_one_out_matches_slow_reference() -> None:
    torch.manual_seed(17)
    log_pi = torch.randn(3, 4, 6).log_softmax(dim=-1)
    vectorized = leave_one_out_logits(log_pi)
    slow = torch.empty_like(vectorized)
    for batch in range(log_pi.shape[0]):
        for removed in range(log_pi.shape[1]):
            kept = [index for index in range(log_pi.shape[1]) if index != removed]
            slow[batch, removed] = log_pi[batch, kept].mean(dim=0)
    assert torch.allclose(vectorized, slow, atol=1e-7)


def test_unique_error_matches_slow_distribution_math() -> None:
    log_pi = torch.randn(2, 4, 5).log_softmax(-1)
    full = torch.rand(2, 5).softmax(-1)
    vectorized = kl_divergence(
        full[:, None].expand(2, 4, 5), leave_one_out_logits(log_pi).softmax(-1)
    )
    slow = torch.empty(2, 4)
    for batch in range(2):
        for removed in range(4):
            kept = torch.cat([log_pi[batch, :removed], log_pi[batch, removed + 1 :]])
            approximation = kept.mean(0).softmax(-1)
            slow[batch, removed] = kl_divergence(full[batch], approximation)
    assert torch.allclose(vectorized, slow, atol=1e-7)

