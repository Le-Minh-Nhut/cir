from __future__ import annotations

import torch

from models.one_shot_control import FGCLIP2OneShotControl, OneShotControlConfig
from training.taper_mag_losses import terminal_bidirectional_infonce


def test_same_backbone_one_shot_control_smoke() -> None:
    model = FGCLIP2OneShotControl(
        OneShotControlConfig(text_dim=20, retrieval_dim=768, hidden_dim=24, dropout=0)
    )
    reference = torch.nn.functional.normalize(torch.randn(3, 768), dim=-1)
    text = torch.randn(3, 6, 20)
    mask = torch.tensor(
        [[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]],
        dtype=torch.bool,
    )
    query = model(reference, text, mask)
    assert query.shape == (3, 768)
    torch.testing.assert_close(query.norm(dim=-1), torch.ones(3), atol=1e-6, rtol=1e-6)
    targets = torch.nn.functional.normalize(torch.randn(3, 768), dim=-1)
    loss = terminal_bidirectional_infonce(
        query,
        targets,
        ("t0", "t1", "t2"),
        (("t0",), ("t1",), ("t2",)),
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert model.composer[1].weight.grad is not None
    assert model.trainable_parameter_count > 0
