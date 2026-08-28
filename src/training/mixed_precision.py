from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext

import torch


SUPPORTED_RUNTIME_PRECISIONS = frozenset({"bf16", "fp32"})


def runtime_autocast(
    device: torch.device,
    precision: str,
) -> AbstractContextManager[None]:
    """Return the one canonical TAPER runtime autocast context.

    CUDA BF16 uses AMP. CPU and explicit FP32 retain ordinary FP32 execution.
    """
    if precision not in SUPPORTED_RUNTIME_PRECISIONS:
        raise ValueError(
            f"Unsupported TAPER runtime precision {precision!r}; "
            f"expected one of {sorted(SUPPORTED_RUNTIME_PRECISIONS)}"
        )
    if device.type == "cuda" and precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()
