from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from backbones.fgclip2_base import FGCLIP2BaseBackbone, TextTuningConfig, VisionTuningConfig
from cache.taper_mag import FeatureSourcePolicy, ImageCacheManifest
from conftest import FakeFGCLIP2, FakeImageProcessor, FakeTokenizer
from models.taper_mag.contracts import PolicyBatch, SupervisionBatch
from models.taper_mag.model import TaperMAG, TaperMAGConfig
from training.negative_bank import NegativeBank
from training.checkpointing import load_checkpoint, save_checkpoint
from training.taper_mag_engine import CurriculumStage, EngineConfig, TaperMAGTrainingEngine
from training.taper_mag_optimizer import OptimizerConfig, build_optimizer


def backbone() -> FGCLIP2BaseBackbone:
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


def test_invalid_cache_tuning_combinations_fail_loudly() -> None:
    with pytest.raises(ValueError, match="cached text"):
        FeatureSourcePolicy(True, False, False, True, True, True, True).validate()
    with pytest.raises(ValueError, match="cached reference"):
        FeatureSourcePolicy(False, True, False, False, True, True, True).validate()
    with pytest.raises(ValueError, match="gallery embeddings"):
        FeatureSourcePolicy(False, False, True, False, True, True, True).validate()


def test_manifest_exact_validation() -> None:
    manifest = ImageCacheManifest(
        schema_version=1,
        model_id="qihoo360/fg-clip2-base",
        revision="a" * 40,
        processor_config_hash="p",
        extraction_method="official",
        normalization="L2",
        dtype="bfloat16",
        image_id_mapping_hash="m",
        feature_dim=768,
        patch_policy="official_dynamic_v1",
        split="train",
        spatial_shapes_present=False,
        image_count=100,
        complete_split=True,
    )
    manifest.require_exact(manifest)
    with pytest.raises(RuntimeError, match="feature_dim"):
        manifest.require_exact(replace(manifest, feature_dim=1024))


def test_online_text_to_taper_gradient_contract_and_optimizer_groups() -> None:
    torch.manual_seed(5)
    fg = backbone()
    taper = TaperMAG(
        TaperMAGConfig(text_dim=16, vision_dim=16, retrieval_dim=16, dropout=0, max_steps=1)
    )
    optimizer = build_optimizer(taper, fg, OptimizerConfig())
    group_names = {group["group_name"] for group in optimizer.param_groups}
    assert {"actor_decay", "utility_decay", "text_decay"}.issubset(group_names)
    assert not any(parameter.requires_grad for parameter in fg.model.vision_model.parameters())

    tokenized = fg.tokenize_texts(["make it red", "add sleeves"])
    policy = PolicyBatch(
        reference_ids=("r0", "r1"),
        modification_texts=("make it red", "add sleeves"),
        reference_local=torch.randn(2, 5, 16),
        reference_local_mask=torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.bool),
        reference_global=torch.nn.functional.normalize(torch.randn(2, 16), dim=-1),
        text_input_ids=tokenized.input_ids,
        text_attention_mask=tokenized.attention_mask,
        text_content_mask=tokenized.content_mask,
    )
    target = torch.nn.functional.normalize(torch.randn(2, 16), dim=-1)
    supervision = SupervisionBatch(
        target_embedding=target,
        target_ids=("t0", "t1"),
        positive_ids=(("t0",), ("t1",)),
    )
    negative_bank = NegativeBank(
        torch.randn(4, 16), ("n0", "n1", "n2", "n3"), hard_negatives=3
    )
    engine = TaperMAGTrainingEngine(fg, taper, negative_bank)
    result = engine.step(
        policy,
        supervision,
        EngineConfig(stage=CurriculumStage.ACTOR_WARMUP, horizon=1),
    )
    result.loss.backward()
    for index, block in enumerate(fg.model.text_model.encoder.layers):
        gradients = [parameter.grad for parameter in block.parameters()]
        if index < 8:
            assert all(gradient is None for gradient in gradients)
        else:
            assert any(gradient is not None and gradient.abs().sum() > 0 for gradient in gradients)
    assert all(parameter.grad is None for parameter in fg.model.vision_model.parameters())
    assert taper.operator_generator.text_reader.queries.grad is not None
    assert taper.executor.film.weight.grad is not None
    assert taper.readout.retrieval_projection.weight.grad is not None


def test_checkpoint_resume_restores_model_text_optimizer_scheduler_and_rng(tmp_path) -> None:
    fg = backbone()
    taper = TaperMAG(
        TaperMAGConfig(text_dim=16, vision_dim=16, retrieval_dim=16, dropout=0, max_steps=1)
    ).eval()
    optimizer = build_optimizer(taper, fg, OptimizerConfig())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    checkpoint = tmp_path / "resume.ckpt"
    reference_parameter = next(taper.parameters()).detach().clone()
    reference_text = next(
        parameter for parameter in fg.model.parameters() if parameter.requires_grad
    ).detach().clone()
    save_checkpoint(
        checkpoint,
        model=taper,
        backbone=fg,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=2,
        global_step=17,
        stage="actor_warmup",
        curriculum_state={"horizon": 1},
        resolved_config={"seed": 42},
        manifest_hashes={"train": "abc"},
        best_metrics={"mean_recall": 1.0},
    )
    with torch.no_grad():
        next(taper.parameters()).add_(10)
        next(parameter for parameter in fg.model.parameters() if parameter.requires_grad).add_(10)
    payload = load_checkpoint(
        checkpoint,
        model=taper,
        backbone=fg,
        optimizer=optimizer,
        scheduler=scheduler,
        expected_manifest_hashes={"train": "abc"},
    )
    torch.testing.assert_close(next(taper.parameters()), reference_parameter)
    torch.testing.assert_close(
        next(parameter for parameter in fg.model.parameters() if parameter.requires_grad),
        reference_text,
    )
    assert payload["epoch"] == 2 and payload["global_step"] == 17
