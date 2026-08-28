from __future__ import annotations

import statistics
import time
from typing import Any, Callable

import torch
from torch import Tensor
from torch.optim import AdamW, Optimizer

from models.taper_mag.contracts import PolicyBatch, SupervisionBatch
from models.taper_mag.rollout import RolloutConfig
from models.taper_mag.utility import HistoryState
from training.mixed_precision import runtime_autocast
from training.taper_mag_engine import EngineConfig, TaperMAGTrainingEngine


def _timed(device: torch.device, function: Callable[[], Any]) -> tuple[Any, float]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    result = function()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return result, (time.perf_counter() - start) * 1000.0


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def profile_taper_runtime(
    engine: TaperMAGTrainingEngine,
    policy: PolicyBatch,
    supervision: SupervisionBatch,
    config: EngineConfig,
    *,
    optimizer: Optimizer | None = None,
    repeats: int = 3,
    precision: str = "fp32",
) -> dict[str, Any]:
    """Read-only profile: no optimizer step, no persistent weight/state mutation."""
    if repeats <= 0:
        raise ValueError("profile repeats must be positive")
    model = engine.model
    backbone = engine.backbone
    device = next(model.parameters()).device
    model_mode, backbone_mode = model.training, backbone.training
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.eval()
    backbone.eval()

    def model_runtime(function: Callable[[], Any]) -> Any:
        with runtime_autocast(device, precision):
            return function()

    try:
        encoded, text_ms = _timed(
            device, lambda: model_runtime(lambda: engine.encode_policy(policy))
        )
        prepared, actor_ms = _timed(
            device, lambda: model_runtime(lambda: model.prepare(encoded))
        )
        _, state, operators = prepared
        history = HistoryState.initialize(
            state.local.shape[0],
            operators.operators.shape[1],
            state.local.shape[1],
            device,
        )
        preview, preview_ms = _timed(
            device,
            lambda: model_runtime(
                lambda: model.preview_detached_actor(
                    state,
                    operators,
                    history,
                    step=0,
                    max_steps=config.horizon,
                    detach_utility_inputs=True,
                )
            ),
        )
        current, _, candidate_readout, _ = preview

        def teacher_call() -> None:
            negatives = engine.negative_bank.mine_once(current.query, supervision)
            engine.teacher.score(
                current.query,
                candidate_readout.query,
                supervision,
                negatives,
                step_cost=config.step_cost,
            )

        _, teacher_ms = _timed(device, teacher_call)
        model.zero_grad(set_to_none=True)
        backbone.zero_grad(set_to_none=True)

        def forward_backward() -> Tensor:
            result = model_runtime(lambda: engine.step(policy, supervision, config))
            result.loss.backward()
            return result.loss.detach()

        loss, train_ms = _timed(device, forward_backward)
        model.zero_grad(set_to_none=True)
        backbone.zero_grad(set_to_none=True)

        optimizer_step_ms: float | None = None
        if optimizer is not None:
            if not isinstance(optimizer, AdamW):
                raise TypeError("V4 profiler optimizer timing requires the canonical AdamW")
            optimized_parameters = [
                parameter
                for group in optimizer.param_groups
                for parameter in group["params"]
            ]
            snapshots = [parameter.detach().cpu().clone() for parameter in optimized_parameters]
            profile_groups = [
                {key: value for key, value in group.items() if key != "params"}
                | {"params": group["params"]}
                for group in optimizer.param_groups
            ]
            profile_optimizer = AdamW(profile_groups)

            def optimizer_train_step() -> Tensor:
                profile_optimizer.zero_grad(set_to_none=True)
                result = model_runtime(lambda: engine.step(policy, supervision, config))
                result.loss.backward()
                profile_optimizer.step()
                return result.loss.detach()

            _, optimizer_step_ms = _timed(device, optimizer_train_step)
            with torch.no_grad():
                for parameter, snapshot in zip(
                    optimized_parameters, snapshots, strict=True
                ):
                    parameter.copy_(snapshot.to(parameter.device))
            profile_optimizer.zero_grad(set_to_none=True)
            model.zero_grad(set_to_none=True)
            backbone.zero_grad(set_to_none=True)

        latencies: list[float] = []
        last_output = None
        with torch.inference_mode():
            for _ in range(repeats):
                last_output, latency = _timed(
                    device,
                    lambda: model_runtime(
                        lambda: model(
                            encoded,
                            RolloutConfig(
                                max_steps=config.horizon,
                                selection_mode="learned",
                                straight_through=False,
                                exploration_probability=0.0,
                            ),
                            detach_utility_inputs=True,
                        ),
                    ),
                )
                latencies.append(latency)
        assert last_output is not None
        action_count = (
            last_output.trace.active
            & last_output.trace.actions.ne(model.config.num_queries)
        ).sum(dim=1).float()
        stop_histogram = {
            str(step): int(
                (last_output.trace.actions[:, step] == model.config.num_queries).sum()
            )
            for step in range(last_output.trace.actions.shape[1])
        }
        taper_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        report = backbone.parameter_report(taper_parameters=taper_trainable)
        return {
            "schema_version": 1,
            "parameter_counts": report,
            "memory": {
                "peak_allocated_vram_bytes": (
                    int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
                ),
                "peak_reserved_vram_bytes": (
                    int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None
                ),
            },
            "timing_ms": {
                "backbone_text": text_ms,
                "actor_operator": actor_ms,
                "executor_candidate_preview": preview_ms,
                "teacher_negative_mining": teacher_ms,
                "forward_backward_no_optimizer_update": train_ms,
                "train_optimizer_step": optimizer_step_ms,
                "validation_query_p50": statistics.median(latencies),
                "validation_query_p95": _percentile(latencies, 0.95),
            },
            "throughput": {
                "samples_per_second_forward_backward": policy.reference_local.shape[0]
                / max(train_ms / 1000.0, 1e-12),
                "candidate_previews_per_second": (
                    policy.reference_local.shape[0] * model.config.num_queries
                ) / max(preview_ms / 1000.0, 1e-12),
            },
            "rollout": {
                "mean_steps": float(action_count.mean()),
                "p95_steps": float(torch.quantile(action_count, 0.95)),
                "stop_histogram": stop_histogram,
            },
            "numerical": {
                "loss": float(loss),
                "finite": bool(torch.isfinite(loss)),
            },
            "flops": {
                "preview": None,
                "selected_action": None,
                "status": "unavailable_no_reliable_flop_tooling",
            },
            "optimizer_step_timing": {
                "value_ms": optimizer_step_ms,
                "status": (
                    "measured_with_disposable_adamw_and_exact_weight_restore"
                    if optimizer_step_ms is not None
                    else "not_measured_optimizer_not_provided"
                ),
            },
        }
    finally:
        model.train(model_mode)
        backbone.train(backbone_mode)
