import torch

from models.iag_srme.grounding import AnchorGrounder


def test_stable_grounding_and_normalization_axis() -> None:
    grounder = AnchorGrounder(width=8).eval()
    intents = torch.randn(2, 4, 8)
    anchor = torch.randn(2, 10, 8)
    support = grounder(intents, anchor)
    changed_state = torch.randn_like(anchor) * 100
    support_again = grounder(intents, anchor)
    assert changed_state.shape == anchor.shape
    assert torch.equal(support, support_again)
    assert torch.allclose(support.sum(dim=-1), torch.ones(2, 4), atol=1e-6)
    assert not torch.allclose(support.sum(dim=1), torch.ones(2, 10))


def test_entmax_produces_exact_sparse_zeros() -> None:
    grounder = AnchorGrounder(width=4, grounding_width=4).eval()
    with torch.no_grad():
        grounder.intent_projection.weight.copy_(torch.eye(4))
        grounder.anchor_projection.weight.copy_(torch.eye(4))
    intents = torch.tensor([[[10.0, 0, 0, 0]]]).expand(1, 4, 4)
    anchor = torch.tensor([[[10.0, 0, 0, 0], [-10.0, 0, 0, 0], [0.0, 1, 0, 0]]])
    support = grounder(intents, anchor)
    assert (support == 0).any()
    assert torch.equal(support[..., 1], torch.zeros_like(support[..., 1]))
