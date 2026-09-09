from __future__ import annotations

import torch
import torch.nn.functional as F
import pytest
from torch import Tensor, nn

from models.iag_srme import IAGSRME, IAGSRMEConfig


class TinyBackbone(nn.Module):
    global_readout_mode = "learned_qg"
    state_dim = 16
    text_dim = 8
    dense_dim = 12
    retrieval_dim = 12
    patch_grid = (3, 3)

    def __init__(self) -> None:
        super().__init__()
        self.image_projection = nn.Linear(3, self.state_dim)
        self.text_embedding = nn.Embedding(32, self.text_dim)
        self.global_projection = nn.Linear(self.state_dim, self.state_dim)
        self.dense_projection = nn.Linear(self.state_dim, self.dense_dim)
        self.retrieval_projection = nn.Linear(self.state_dim, self.retrieval_dim)

    def initial_state(self, images: Tensor) -> Tensor:
        patches = images.permute(0, 2, 3, 1).reshape(images.shape[0], 9, 3)
        return self.image_projection(patches)

    def global_readout(self, state: Tensor, cls_anchor: Tensor | None = None) -> Tensor:
        del cls_anchor
        return self.global_projection(state.mean(dim=-2))

    def dense_readout(self, state: Tensor) -> Tensor:
        return self.dense_projection(state)

    def retrieval_readout(self, state: Tensor, cls_anchor: Tensor | None = None) -> Tensor:
        return self.retrieval_from_global(self.global_readout(state, cls_anchor))

    def retrieval_from_global(self, global_state: Tensor) -> Tensor:
        return F.normalize(self.retrieval_projection(global_state), dim=-1)

    def encode_text(
        self, input_ids: Tensor, attention_mask: Tensor, content_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        del attention_mask
        tokens = self.text_embedding(input_ids)
        mask = content_mask[..., None].to(tokens.dtype)
        return tokens, (tokens * mask).sum(1) / mask.sum(1).clamp_min(1)

    def encode_global_images(self, images: Tensor) -> Tensor:
        return self.retrieval_readout(self.initial_state(images))


@pytest.fixture
def features() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    torch.manual_seed(7)
    state = torch.randn(3, 9, 16)
    tokens = torch.randn(3, 6, 8)
    text = torch.randn(3, 8)
    mask = torch.ones(3, 6, dtype=torch.bool)
    return state, tokens, text, mask


@pytest.fixture
def model() -> IAGSRME:
    torch.manual_seed(11)
    return IAGSRME(
        TinyBackbone(),
        IAGSRMEConfig(
            width=8,
            num_context_edits=4,
            num_heads=4,
            exec_dim=8,
        ),
    )
