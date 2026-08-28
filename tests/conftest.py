from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import Tensor, nn


class FakeEncoder(nn.Module):
    def __init__(self, dim: int, blocks: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(blocks)])


class FakeTextModel(nn.Module):
    def __init__(self, vocab: int = 64, dim: int = 16, blocks: int = 12) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(vocab, dim)
        self.encoder = FakeEncoder(dim, blocks)
        self.final_layer_norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, dim)

    def forward(self, input_ids: Tensor, attention_mask: Tensor, **_: object) -> object:
        hidden = self.embeddings(input_ids)
        for block in self.encoder.layers:
            hidden = hidden + torch.tanh(block(hidden))
        hidden = self.final_layer_norm(hidden)
        return SimpleNamespace(last_hidden_state=hidden, pooler_output=self.head(hidden[:, -1]))


class FakeVisionModel(nn.Module):
    def __init__(self, dim: int = 16, blocks: int = 12) -> None:
        super().__init__()
        self.encoder = FakeEncoder(dim, blocks)
        self.projection = nn.Linear(dim, dim)


class FakeFGCLIP2(nn.Module):
    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(
                hidden_size=dim, projection_size=dim, num_hidden_layers=12
            ),
            vision_config=SimpleNamespace(
                hidden_size=dim, patch_size=16, num_hidden_layers=12
            ),
        )
        self.text_model = FakeTextModel(dim=dim)
        self.vision_model = FakeVisionModel(dim=dim)
        self.dense_feature_head = nn.Linear(dim, dim)

    def get_text_features(self, input_ids: Tensor, attention_mask: Tensor, **kwargs: object) -> Tensor:
        return self.text_model(input_ids, attention_mask, **kwargs).pooler_output

    def get_image_features(
        self, pixel_values: Tensor, pixel_attention_mask: Tensor, spatial_shapes: Tensor
    ) -> Tensor:
        del pixel_attention_mask, spatial_shapes
        pooled = pixel_values.mean(dim=1)
        return self.vision_model.projection(pooled)

    def get_image_dense_feature(
        self, pixel_values: Tensor, pixel_attention_mask: Tensor, spatial_shapes: Tensor
    ) -> Tensor:
        del pixel_attention_mask, spatial_shapes
        return self.dense_feature_head(pixel_values)


class FakeBatch(dict):
    def to(self, device: torch.device) -> FakeBatch:
        return FakeBatch({key: value.to(device) for key, value in self.items()})


class FakeTokenizer:
    padding_side = "right"
    truncation_side = "right"
    pad_token_id = 0
    eos_token_id = 1
    bos_token_id = 2
    model_max_length = 64

    def __call__(self, texts: list[str], **_: object) -> dict[str, Tensor]:
        batch = len(texts)
        ids = torch.zeros(batch, 64, dtype=torch.long)
        attention = torch.zeros(batch, 64, dtype=torch.long)
        special = torch.ones(batch, 64, dtype=torch.long)
        for index, text in enumerate(texts):
            length = min(max(len(text.split()), 1), 62)
            ids[index, :length] = torch.arange(3, 3 + length)
            ids[index, length] = 1
            attention[index, : length + 1] = 1
            special[index, :length] = 0
        return {
            "input_ids": ids,
            "attention_mask": attention,
            "special_tokens_mask": special,
        }


class FakeImageProcessor:
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]
    patch_size = 16

    def to_dict(self) -> dict[str, object]:
        return {"image_mean": self.image_mean, "image_std": self.image_std, "patch_size": 16}

    def __call__(self, images: list[object], max_num_patches: int, **_: object) -> FakeBatch:
        batch = len(images)
        real = min(4, max_num_patches)
        values = torch.randn(batch, max_num_patches, 16)
        mask = torch.zeros(batch, max_num_patches, dtype=torch.bool)
        mask[:, :real] = True
        return FakeBatch(
            pixel_values=values,
            pixel_attention_mask=mask,
            spatial_shapes=torch.tensor([[2, 2]] * batch),
        )
