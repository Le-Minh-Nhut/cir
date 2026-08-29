from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


LOW_PRECISION_DTYPES = (torch.float16, torch.bfloat16)


def fp32_if_low_precision(tensor: Tensor) -> Tensor:
    """Promote AMP-sensitive arithmetic while preserving ordinary FP32/FP64 behavior."""

    return tensor.float() if tensor.dtype in LOW_PRECISION_DTYPES else tensor


def normalize_fp32(tensor: Tensor, dim: int = -1, epsilon: float = 1e-12) -> Tensor:
    """Compute L2 normalization in FP32 under fp16/bf16 and restore input dtype."""

    with torch.autocast(device_type=tensor.device.type, enabled=False):
        working = fp32_if_low_precision(tensor)
        normalized = F.normalize(working, dim=dim, eps=epsilon)
    return normalized.to(tensor.dtype)
