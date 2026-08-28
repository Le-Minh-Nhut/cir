from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from models.taper_mag.contracts import SupervisionBatch
from training.marginal_gain_teacher import MarginalGainTeacher
from training.mixed_precision import runtime_autocast
from training.negative_bank import CommonNegativeSet
from training.taper_mag_audit import TeacherShadowAuditor, dynamic_frozen_audit
from training.taper_mag_engine import CurriculumStage, EngineConfig
from training.taper_mag_profiler import profile_taper_runtime
from test_taper_mag_training_contract import _end_to_end_fixture


def test_runtime_autocast_enables_only_cuda_bf16() -> None:
    context = torch.no_grad()
    with patch("training.mixed_precision.torch.autocast", return_value=context) as autocast:
        with runtime_autocast(torch.device("cuda"), "bf16"):
            pass
    autocast.assert_called_once_with(device_type="cuda", dtype=torch.bfloat16)

    with patch("training.mixed_precision.torch.autocast") as autocast:
        with runtime_autocast(torch.device("cpu"), "bf16"):
            pass
        with runtime_autocast(torch.device("cuda"), "fp32"):
            pass
    autocast.assert_not_called()
    with pytest.raises(ValueError, match="Unsupported TAPER runtime precision"):
        runtime_autocast(torch.device("cpu"), "fp16")


def test_detached_marginal_gain_teacher_outputs_fp32_from_bf16_inputs() -> None:
    current = torch.randn(2, 8, dtype=torch.bfloat16)
    candidates = torch.randn(2, 4, 8, dtype=torch.bfloat16)
    supervision = SupervisionBatch(
        target_embedding=torch.randn(2, 8, dtype=torch.bfloat16),
        target_ids=("t0", "t1"),
        positive_ids=(("t0",), ("t1",)),
    )
    negatives = CommonNegativeSet(
        embeddings=torch.randn(2, 3, 8, dtype=torch.bfloat16),
        ids=(("n0", "n1", "n2"), ("n0", "n1", "n2")),
    )
    output = MarginalGainTeacher().score(
        current, candidates, supervision, negatives
    )
    assert output.raw_gain.dtype == torch.float32
    assert output.net_values.dtype == torch.float32
    assert not output.raw_gain.requires_grad


CUDA_BF16_AVAILABLE = torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def _cuda_bf16_fixture():
    fg, taper, policy, supervision, engine = _end_to_end_fixture()
    device = torch.device("cuda")
    fg.to(device=device, dtype=torch.bfloat16)
    taper.to(device=device)
    policy = replace(
        policy,
        reference_local=policy.reference_local.to(device),
        reference_local_mask=policy.reference_local_mask.to(device),
        reference_global=policy.reference_global.to(device),
        text_input_ids=policy.text_input_ids.to(device),
        text_attention_mask=policy.text_attention_mask.to(device),
        text_content_mask=policy.text_content_mask.to(device),
        spatial_shapes=(
            policy.spatial_shapes.to(device)
            if policy.spatial_shapes is not None
            else None
        ),
    )
    supervision = replace(
        supervision, target_embedding=supervision.target_embedding.to(device)
    )
    return fg, taper, policy, supervision, engine


@pytest.mark.skipif(not CUDA_BF16_AVAILABLE, reason="requires CUDA BF16 autocast")
def test_bf16_dynamic_audit_and_teacher_shadow_keep_fp32_taper_weights() -> None:
    fg, taper, policy, supervision, engine = _cuda_bf16_fixture()
    taper.eval()
    fg.eval()
    with runtime_autocast(torch.device("cuda"), "bf16"):
        encoded = engine.encode_policy(policy)
    assert encoded.text_tokens.dtype == torch.bfloat16
    assert next(taper.parameters()).dtype == torch.float32
    report = dynamic_frozen_audit(
        taper,
        encoded,
        supervision,
        engine.negative_bank,
        engine.teacher,
        max_steps=1,
        precision="bf16",
    )
    assert report["valid"]
    before = {name: value.detach().clone() for name, value in taper.state_dict().items()}
    auditor = TeacherShadowAuditor(
        taper,
        fg.model,
        engine.negative_bank,
        engine.teacher,
        seed=3,
        precision="bf16",
    )
    auditor.update(
        encoded,
        supervision,
        sample_ids=("s0", "s1"),
        reference_ids=policy.reference_ids,
        modification_texts=policy.modification_texts,
    )
    for name, value in taper.state_dict().items():
        torch.testing.assert_close(value, before[name])


@pytest.mark.skipif(not CUDA_BF16_AVAILABLE, reason="requires CUDA BF16 autocast")
def test_profiler_executes_cuda_bf16_runtime_without_mutating_weights() -> None:
    _, taper, policy, supervision, engine = _cuda_bf16_fixture()
    before = {name: value.detach().clone() for name, value in taper.state_dict().items()}
    report = profile_taper_runtime(
        engine,
        policy,
        supervision,
        EngineConfig(stage=CurriculumStage.ACTOR_WARMUP, horizon=1),
        repeats=1,
        precision="bf16",
    )
    assert report["numerical"]["finite"]
    for name, value in taper.state_dict().items():
        torch.testing.assert_close(value, before[name])
