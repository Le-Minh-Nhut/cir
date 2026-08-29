from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.optim import SGD

from canary_train_iag_srme import _complete_amp_step, _gradients_are_finite


class _FiniteForwardInfiniteBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: Tensor) -> Tensor:
        return value.sum()

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor]:
        return (torch.full_like(grad_output.expand(1), float("inf")),)


def test_grad_scaler_skips_initial_overflow_then_updates() -> None:
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_type)
    # CUDA exercises the real fp16 path; CPU exercises the same public scaler
    # skip/backoff contract without requiring accelerator hardware in CI.
    dtype = torch.float16 if device_type == "cuda" else torch.float32
    weight = nn.Parameter(torch.tensor([1.0], device=device, dtype=dtype))
    optimizer = SGD([weight], lr=0.1)
    scaler = torch.amp.GradScaler(
        device_type,
        init_scale=8.0,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=1,
        enabled=True,
    )
    tracked = {"weight": weight}

    initial = weight.detach().clone()
    first_scale = float(scaler.get_scale())
    overflow_loss = _FiniteForwardInfiniteBackward.apply(weight)
    scaler.scale(overflow_loss).backward()
    scaler.unscale_(optimizer)
    assert not _gradients_are_finite([weight])
    second_scale, skipped_deltas = _complete_amp_step(
        scaler,
        optimizer,
        tracked,
        scale_before=first_scale,
        overflow=True,
    )

    assert second_scale < first_scale
    assert skipped_deltas == {"weight": 0.0}
    assert torch.equal(weight.detach(), initial)

    optimizer.zero_grad(set_to_none=True)
    finite_loss = weight.float().square().sum()
    scaler.scale(finite_loss).backward()
    scaler.unscale_(optimizer)
    assert _gradients_are_finite([weight])
    final_scale, successful_deltas = _complete_amp_step(
        scaler,
        optimizer,
        tracked,
        scale_before=second_scale,
        overflow=False,
    )

    assert final_scale > second_scale
    assert successful_deltas["weight"] > 0.0
    assert not torch.equal(weight.detach(), initial)
