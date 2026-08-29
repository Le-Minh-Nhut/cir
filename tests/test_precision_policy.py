from __future__ import annotations

import pytest
import torch

from training.engine import resolve_precision


def test_precision_policy_keeps_fp16_and_bf16_distinct() -> None:
    cuda = torch.device("cuda")
    fp32 = resolve_precision("fp32", cuda)
    fp16 = resolve_precision("fp16", cuda)
    bf16 = resolve_precision("bf16", cuda)

    assert not fp32.autocast_enabled and fp32.autocast_dtype is None
    assert fp16.autocast_enabled and fp16.autocast_dtype is torch.float16
    assert fp16.scaler_enabled
    assert bf16.autocast_enabled and bf16.autocast_dtype is torch.bfloat16
    assert not bf16.scaler_enabled


def test_cpu_policy_does_not_enable_cuda_gradient_scaler() -> None:
    assert not resolve_precision("fp16", torch.device("cpu")).scaler_enabled
    assert not resolve_precision("bf16", torch.device("cpu")).scaler_enabled


def test_unknown_precision_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported precision"):
        resolve_precision("amp", torch.device("cpu"))
