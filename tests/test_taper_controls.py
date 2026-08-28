from __future__ import annotations

import inspect
from pathlib import Path

import torch
import yaml

import models.one_shot_control as one_shot_module
import models.taper_controls as controls_module
import run_taper_control as control_runner
from backbones.fgclip2_base import FGCLIP2BaseBackbone, TextTuningConfig, VisionTuningConfig
from conftest import FakeFGCLIP2, FakeImageProcessor, FakeTokenizer
from models.one_shot_control import FGCLIP2OneShotControl, OneShotControlConfig
from models.taper_controls import ReferenceOnlyControl, SimpleSumControl, TextControlConfig, TextOnlyControl
from training.taper_mag_losses import terminal_bidirectional_infonce


def _backbone() -> FGCLIP2BaseBackbone:
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


def test_m0_is_exact_target_free_reference_only_query() -> None:
    model = ReferenceOnlyControl()
    reference = torch.randn(3, 768)
    query = model(reference)
    torch.testing.assert_close(query, torch.nn.functional.normalize(reference, dim=-1))
    torch.testing.assert_close(query.norm(dim=-1), torch.ones(3))
    assert sum(parameter.numel() for parameter in model.parameters()) == 0
    assert "text" not in inspect.signature(model.forward).parameters
    assert "target" not in inspect.signature(model.forward).parameters


def test_m1_uses_online_text_only_and_updates_last_four_text_blocks() -> None:
    backbone = _backbone()
    model = TextOnlyControl(TextControlConfig(text_dim=16, retrieval_dim=32))
    tokenized = backbone.tokenize_texts(["make red", "add long sleeves"])
    text_tokens = backbone.encode_text_tokens(tokenized)
    query = model(text_tokens, tokenized.content_mask)
    assert query.shape == (2, 32)
    torch.testing.assert_close(query.norm(dim=-1), torch.ones(2), atol=1e-6, rtol=1e-6)
    query[:, 0].sum().backward()
    for index, block in enumerate(backbone.model.text_model.encoder.layers):
        gradients = [parameter.grad for parameter in block.parameters()]
        if index < 8:
            assert all(gradient is None for gradient in gradients)
        else:
            assert any(gradient is not None and gradient.abs().sum() > 0 for gradient in gradients)
    assert "reference" not in inspect.signature(model.forward).parameters


def test_m2_is_only_scalar_gated_global_sum_and_gate_gets_gradient() -> None:
    model = SimpleSumControl(TextControlConfig(text_dim=20, retrieval_dim=768))
    reference = torch.randn(3, 768)
    text = torch.randn(3, 5, 20)
    mask = torch.ones(3, 5, dtype=torch.bool)
    query = model(reference, text, mask)
    assert query.shape == (3, 768)
    torch.testing.assert_close(query.norm(dim=-1), torch.ones(3), atol=1e-6, rtol=1e-6)
    query[:, 0].sum().backward()
    assert model.log_alpha.grad is not None and model.log_alpha.grad.abs() > 0
    signature = inspect.signature(model.forward).parameters
    assert "reference_global" in signature
    assert "text_tokens" in signature
    assert "reference_local" not in signature
    assert "taper_mag" not in inspect.getsource(controls_module).lower()


def test_m3_remains_one_shot_taper_independent_and_finite() -> None:
    backbone = _backbone()
    model = FGCLIP2OneShotControl(
        OneShotControlConfig(text_dim=16, retrieval_dim=768, hidden_dim=64, dropout=0)
    )
    reference = torch.nn.functional.normalize(torch.randn(3, 768), dim=-1)
    tokenized = backbone.tokenize_texts(["make red", "add sleeves", "remove pattern"])
    text = backbone.encode_text_tokens(tokenized)
    query = model(reference, text, tokenized.content_mask)
    targets = torch.nn.functional.normalize(torch.randn(3, 768), dim=-1)
    loss = terminal_bidirectional_infonce(
        query, targets, ("t0", "t1", "t2"), (("t0",), ("t1",), ("t2",))
    )
    assert query.shape == (3, 768) and torch.isfinite(loss)
    loss.backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in backbone.model.text_model.encoder.layers[8].parameters()
    )
    assert "taper_mag" not in inspect.getsource(one_shot_module).lower()
    assert "train_taper_mag" not in inspect.getsource(control_runner).lower()
    assert "models.taper_mag" not in inspect.getsource(control_runner).lower()
    signature = inspect.signature(model.forward).parameters
    assert not {"state", "actions", "teacher", "target"}.intersection(signature)


def test_m0_m3_configs_are_parallel_controls_with_matched_update_budgets() -> None:
    root = Path(__file__).parents[1] / "conf"
    paths = {
        "M0": root / "taper_mag_v4_m0_reference_only.yaml",
        "M1": root / "taper_mag_v4_m1_text_only.yaml",
        "M2": root / "taper_mag_v4_m2_simple_sum.yaml",
        "M3": root / "taper_mag_v4_one_shot_control.yaml",
    }
    configs = {
        control: yaml.safe_load(path.read_text(encoding="utf-8"))
        for control, path in paths.items()
    }
    assert {control: config["control"]["id"] for control, config in configs.items()} == {
        control: control for control in paths
    }
    assert {
        configs[control]["training"]["max_optimizer_updates"]
        for control in ("M1", "M2", "M3")
    } == {4260}
    assert all(
        configs[control]["backbone"]["text_tuning"]["mode"] == "last_n_blocks"
        for control in configs
    )
