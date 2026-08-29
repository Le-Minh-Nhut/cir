import inspect

import torch

from models.iag_srme.intent import TextIntentEncoder


def test_text_only_intent_api_and_determinism() -> None:
    encoder = TextIntentEncoder(width=16, num_candidates=4, num_heads=4).eval()
    text = torch.randn(2, 6, 16)
    mask = torch.ones(2, 6, dtype=torch.bool)
    first = encoder(text, mask)
    unrelated_image_state = torch.randn(2, 9, 16) * 100
    second = encoder(text, mask)
    assert unrelated_image_state.shape == (2, 9, 16)
    assert torch.equal(first, second)
    assert tuple(inspect.signature(encoder.forward).parameters) == ("text_tokens", "content_mask")


def test_query_permutation_has_no_hardcoded_semantics() -> None:
    encoder = TextIntentEncoder(width=16, num_candidates=4, num_heads=4).eval()
    text = torch.randn(2, 6, 16)
    mask = torch.ones(2, 6, dtype=torch.bool)
    baseline = encoder(text, mask)
    permutation = torch.tensor([2, 0, 3, 1])
    with torch.no_grad():
        encoder.query_bank.copy_(encoder.query_bank[permutation])
    permuted = encoder(text, mask)
    assert torch.allclose(permuted, baseline[:, permutation], atol=1e-6)

