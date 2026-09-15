from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from models.taper import TAPER


def _encoder_orthogonal_regularization(templates: torch.Tensor) -> torch.Tensor:
    batch_size, length, _ = templates.size()
    normalized_templates = F.normalize(templates, p=2, dim=-1)
    cosine_score = torch.matmul(
        normalized_templates,
        normalized_templates.transpose(-2, -1),
    )
    identity = torch.eye(length, device=templates.device)
    identity = identity.unsqueeze(0).repeat(batch_size, 1, 1)
    return F.mse_loss(cosine_score, identity)


def test_slot_query_orthogonal_regularization_matches_encoder() -> None:
    templates = torch.tensor(
        [
            [[1.0, 2.0, 0.0], [2.0, -1.0, 1.0]],
            [[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]],
        ]
    )

    actual = TAPER._orthogonal_regularization(templates)
    expected = _encoder_orthogonal_regularization(templates)

    torch.testing.assert_close(actual, expected)


def test_orthonormal_slot_queries_have_zero_orthogonal_loss() -> None:
    templates = torch.eye(4).unsqueeze(0)

    loss = TAPER._orthogonal_regularization(templates)

    torch.testing.assert_close(loss, torch.zeros_like(loss))


def test_orthogonal_loss_backpropagates_to_slot_queries() -> None:
    model = TAPER(
        torch.nn.Identity(),
        text_dim=2,
        reference_dim=2,
        teacher_query_dim=2,
        query_dim=2,
        slot_dim=2,
        state_dim=2,
        num_slots=2,
        num_primitives=2,
    )
    with torch.no_grad():
        model.slot_queries.copy_(torch.tensor([[1.0, 0.0], [1.0, 1.0]]))

    loss = model._orthogonal_regularization(model.slot_queries.unsqueeze(0))
    loss.backward()

    assert model.slot_queries.grad is not None
    assert torch.isfinite(model.slot_queries.grad).all()
    assert torch.count_nonzero(model.slot_queries.grad) > 0


def test_e2e_objective_weights_orthogonal_loss_at_point_seven() -> None:
    config_path = (
        Path(__file__).resolve().parents[2] / "conf" / "experiment" / "taper_e2e.yaml"
    )
    config = OmegaConf.load(config_path)

    assert config.loss_weights.retrieval_loss == 1.0
    assert config.loss_weights.ortho_loss == 0.7
