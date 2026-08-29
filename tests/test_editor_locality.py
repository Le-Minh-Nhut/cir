import torch

from models.iag_srme.editor import SharedTokenEditor


def test_exact_support_locality() -> None:
    editor = SharedTokenEditor(width=8, lambda_z=0.1)
    contexts = torch.randn(2, 4, 8)
    anchor = torch.randn(2, 6, 8)
    state = torch.randn(2, 6, 8)
    support = torch.tensor([[[1.0, 0, 0, 0, 0, 0]] * 4, [[0, 0.4, 0.6, 0, 0, 0]] * 4])
    delta, candidates = editor(contexts, support, anchor, state)
    zero_mask = support == 0
    assert torch.equal(delta[zero_mask], torch.zeros_like(delta[zero_mask]))
    assert torch.equal(candidates, state[:, None] + delta)
