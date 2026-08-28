from __future__ import annotations

import torch

from backbones.fgclip2_base import (
    FGCLIP2BaseBackbone,
    TextTuningConfig,
    VisionTuningConfig,
)
from conftest import FakeFGCLIP2, FakeImageProcessor, FakeTokenizer
from training.text_drift import TextDriftMonitor


def make_backbone() -> FGCLIP2BaseBackbone:
    return FGCLIP2BaseBackbone(
        model=FakeFGCLIP2(),
        tokenizer=FakeTokenizer(),
        image_processor=FakeImageProcessor(),
        text_tuning=TextTuningConfig(
            mode="last_n_blocks",
            num_unfrozen_blocks=4,
            train_final_norm=True,
            train_projection=False,
        ),
        vision_tuning=VisionTuningConfig(),
    )


def test_short_pooling_matches_official_final_token_head_contract() -> None:
    backbone = make_backbone()
    batch = backbone.tokenize_texts(["make it red", "add long sleeves"])
    states = backbone.encode_text_tokens(batch)
    actual = backbone.pool_short_text_states(states)
    expected = backbone.model.text_model.head(states[:, -1, :])
    assert actual.shape == (2, backbone.contract.retrieval_dim)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected)


def test_capture_and_measure_each_use_one_text_transformer_forward() -> None:
    backbone = make_backbone()
    batch = backbone.tokenize_texts(["make it brighter", "remove the sleeves"])
    forward_count = 0

    def count_forward(_module, _inputs, _output) -> None:
        nonlocal forward_count
        forward_count += 1

    handle = backbone.model.text_model.register_forward_hook(count_forward)
    try:
        snapshot = TextDriftMonitor.capture(backbone, batch)
        assert forward_count == 1
        metrics = TextDriftMonitor.measure(backbone, snapshot)
        assert forward_count == 2
    finally:
        handle.remove()

    torch.testing.assert_close(
        torch.tensor(metrics["text_token_cosine"]), torch.tensor(1.0), atol=1e-6, rtol=0
    )


def test_unchanged_snapshot_has_zero_drift() -> None:
    backbone = make_backbone()
    batch = backbone.tokenize_texts(["make it blue", "add a collar"])
    snapshot = TextDriftMonitor.capture(backbone, batch)
    metrics = TextDriftMonitor.measure(backbone, snapshot)
    assert abs(metrics["text_token_cosine"] - 1.0) < 1e-6
    assert abs(metrics["text_token_cosine_drift"]) < 1e-6
    assert abs(metrics["text_pooled_cosine"] - 1.0) < 1e-6
    assert abs(metrics["text_pooled_cosine_drift"]) < 1e-6
    assert metrics["text_parameter_relative_change"] == 0.0


def test_trainable_text_update_is_detected() -> None:
    backbone = make_backbone()
    batch = backbone.tokenize_texts(["make it green", "shorten the sleeves"])
    snapshot = TextDriftMonitor.capture(backbone, batch)
    trainable = next(
        parameter
        for name, parameter in backbone.model.named_parameters()
        if parameter.requires_grad and name.startswith("text_model.encoder.layers.8")
    )
    with torch.no_grad():
        trainable.add_(0.01)
    metrics = TextDriftMonitor.measure(backbone, snapshot)
    assert metrics["text_parameter_relative_change"] > 0
    assert (
        abs(metrics["text_token_cosine_drift"]) > 1e-8
        or abs(metrics["text_pooled_cosine_drift"]) > 1e-8
    )
