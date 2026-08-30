from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from data.images import FashionIQImageCollator, ImageBatch
from datasets.common import DirectoryImageStore
from datasets.fashioniq import FashionIQDataset
from diagnostics.iag_srme import functional_effective_rank, pairwise_cosine
from losses.objective import IAGSRMEObjective, ObjectiveConfig
from losses.retrieval import positive_mask_from_ids
from models.iag_srme import FGCLIPBackbone, FGCLIPRegime, IAGSRME, IAGSRMEConfig, IAGSRMECore
from runtime import configure_torch_runtime, seed_everything
from training.engine import assert_training_setup, resolve_precision, trainable_parameters


BASE_CHECKPOINT = "qihoo360/fg-clip-base"
BASE_REVISION = "454d76372c2cf5eb48fa0d871fd0534481484d97"
CATEGORIES = ("dress", "shirt", "toptee")


def _gradient_norm(parameters: Iterable[nn.Parameter]) -> float:
    squares = [parameter.grad.detach().float().square().sum() for parameter in parameters if parameter.grad is not None]
    if not squares:
        return 0.0
    return math.sqrt(float(torch.stack(squares).sum()))


def _gradients_are_finite(parameters: Iterable[nn.Parameter]) -> bool:
    return all(
        bool(torch.isfinite(parameter.grad).all())
        for parameter in parameters
        if parameter.grad is not None
    )


def _parameter_delta(parameter: nn.Parameter, initial: Tensor) -> float:
    return float((parameter.detach().float() - initial).abs().max())


def _complete_amp_step(
    scaler: torch.amp.GradScaler,
    optimizer: torch.optim.Optimizer,
    tracked: dict[str, nn.Parameter],
    *,
    scale_before: float,
    overflow: bool,
) -> tuple[float, dict[str, float]]:
    """Finish one scaled step and enforce the overflow skip contract."""

    before_step = {
        name: parameter.detach().clone() for name, parameter in tracked.items()
    }
    scaler.step(optimizer)
    scaler.update()
    scale_after = float(scaler.get_scale())
    step_deltas = {
        name: _parameter_delta(parameter, before_step[name])
        for name, parameter in tracked.items()
    }
    if overflow:
        if not scaler.is_enabled():
            raise FloatingPointError("non-finite gradients without an enabled GradScaler")
        if scale_after >= scale_before:
            raise RuntimeError(
                "GradScaler did not reduce its scale after non-finite gradients: "
                f"{scale_before} -> {scale_after}"
            )
        changed = {name: delta for name, delta in step_deltas.items() if delta != 0.0}
        if changed:
            raise AssertionError(
                f"tracked parameters changed during a skipped AMP overflow step: {changed}"
            )
    return scale_after, step_deltas


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def _verify_absorbing_stop(output) -> None:
    for step in output.trace:
        dead = ~step.live_before
        if dead.any():
            if not torch.equal(step.next_state[dead], step.current_state[dead]):
                raise AssertionError("a stopped canary sample changed state")
            if not torch.equal(step.next_query[dead], step.current_query[dead]):
                raise AssertionError("a stopped canary sample changed query")


def _dynamic_change_metrics(output) -> dict[str, float]:
    measurements: defaultdict[str, list[float]] = defaultdict(list)
    for previous, current in zip(output.trace[:-1], output.trace[1:], strict=True):
        executed = previous.selected_index.lt(output.intents.shape[1])
        if not executed.any():
            continue
        pairs = {
            "gt": (previous.current_evidence, current.current_evidence),
            "dt": (previous.accumulated_local_change, current.accumulated_local_change),
            "context": (previous.contexts, current.contexts),
            "delta_z": (previous.delta_z, current.delta_z),
            "scores": (previous.scores, current.scores),
        }
        for name, (left, right) in pairs.items():
            measurements[name].append(float((right[executed] - left[executed]).abs().max()))
    return {name: _mean(values) for name, values in measurements.items()}


def _batch_diagnostics(output) -> dict[str, object]:
    actions = torch.stack([step.selected_index for step in output.trace], dim=1)
    delta_q = torch.stack([step.delta_q for step in output.trace], dim=1)
    supports = (
        output.conditional_supports.float()
        if output.conditional_supports is not None
        else output.supports.float()
    )
    return {
        "stop_by_timestep": actions.eq(output.intents.shape[1]).float().mean(dim=0).tolist(),
        "selected_distribution": torch.nn.functional.one_hot(
            actions, output.intents.shape[1] + 1
        )
        .float()
        .mean(dim=(0, 1))
        .tolist(),
        "grounding_support_fraction": float((supports > 0).float().mean()),
        "grounding_entropy": float(
            -(supports * supports.clamp_min(1e-8).log()).sum(dim=-1).mean()
        ),
        "delta_q_norm": float(delta_q.float().norm(dim=-1).mean()),
        "functional_effect_rank": float(functional_effective_rank(delta_q).mean()),
        "functional_delta_q_cosine": float(pairwise_cosine(delta_q).mean()),
        "dynamic_changes": _dynamic_change_metrics(output),
        "visual_null_probability": (
            None
            if output.visual_null_probabilities is None
            else {
                "mean": float(output.visual_null_probabilities.float().mean()),
                "maximum": float(output.visual_null_probabilities.float().max()),
            }
        ),
    }


def _build_canary_model(device: torch.device) -> tuple[IAGSRME, object, object]:
    regime = FGCLIPRegime(
        checkpoint=BASE_CHECKPOINT,
        revision=BASE_REVISION,
        train_vision=True,
        train_text=True,
        train_text_projection=False,
    )
    backbone = FGCLIPBackbone.from_pretrained(regime, internal_width=256)
    tokenizer, processor = FGCLIPBackbone.load_processor(
        regime.checkpoint, regime.revision, regime.trust_remote_code
    )
    core = IAGSRMECore(
        IAGSRMEConfig(
            width=256,
            num_candidates=4,
            max_steps=3,
            num_heads=8,
            retrieval_dim=backbone.retrieval_dim,
            lambda_z=0.10,
            query_cap=1000.0,
            selector_temperature=1.0,
            selector_gumbel_noise=True,
            enable_visual_null=True,
            visual_null_initial_logit=0.0,
            grounding_normalization="entmax15",
        )
    )
    return IAGSRME(backbone, core).to(device), tokenizer, processor


def _build_loader(args, tokenizer, processor) -> DataLoader[ImageBatch]:
    root = Path(args.dataset_root)
    annotation_root = root / "captions"
    image_root = root / "images"
    if not annotation_root.is_dir() or not image_root.is_dir():
        raise FileNotFoundError(
            f"FashionIQ root must contain captions/ and images/: {root.resolve()}"
        )
    dataset = FashionIQDataset(
        annotation_root,
        "train",
        CATEGORIES,
        caption_policy="ordered_and",
        seed=args.seed,
    )
    collator = FashionIQImageCollator(
        DirectoryImageStore(image_root),
        tokenizer,
        processor,
        max_text_length=77,
        include_targets=True,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collator,
        generator=torch.Generator().manual_seed(args.seed),
    )


def _next_batch(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def main() -> None:
    parser = argparse.ArgumentParser(description="Canonical GPU FashionIQ IAG-SRME canary")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--precision", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument(
        "--dataset-root",
        default=os.environ.get("FASHIONIQ_ROOT", "data/fashionIQ_dataset"),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--exploding-gradient-threshold", type=float, default=1e4)
    parser.add_argument("--accelerator-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 2 or args.log_every < 1:
        raise ValueError("steps/log-every must be positive and batch-size must be at least two")
    if not torch.cuda.is_available():
        raise RuntimeError("the fp16 FashionIQ canary requires a CUDA device")
    device = torch.device(f"cuda:{args.accelerator_index}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    seed_everything(args.seed, deterministic=True)
    configure_torch_runtime(deterministic=True, benchmark=False)
    precision = resolve_precision(args.precision, device)
    model, tokenizer, processor = _build_canary_model(device)
    if model.core.config.query_cap != 1000.0 or not model.core.config.enable_visual_null:
        raise AssertionError("R1b canary requires query_cap=1000 and Visual NULL enabled")
    objective = IAGSRMEObjective(ObjectiveConfig(), width=256).to(device)
    loader = _build_loader(args, tokenizer, processor)
    optimizer = AdamW(
        trainable_parameters(model, objective),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    assert_training_setup(model, objective, optimizer, device)
    scaler = torch.amp.GradScaler("cuda", enabled=precision.scaler_enabled)
    initial_scale = float(scaler.get_scale())
    minimum_scale = initial_scale
    total_amp_overflows = 0
    consecutive_amp_overflows = 0
    successful_optimizer_steps = 0
    skipped_optimizer_steps = 0
    first_successful_step: int | None = None

    parameter_families = {
        "vision": list(model.backbone.model.vision_model.parameters())
        + list(model.backbone.model.visual_projection.parameters()),
        "text_encoder": list(model.backbone.model.text_model.parameters()),
        "intent_queries": [model.core.intent_encoder.query_bank],
        "grounding": list(model.core.grounder.parameters()),
        "visual_null": [
            model.core.grounder.visual_null_key,
            model.core.grounder.visual_null_bias,
        ],
        "editor": list(model.core.editor.parameters()),
        "readout": list(model.core.readout.parameters()),
        "scorer": list(model.core.scorer.parameters()),
    }
    tracked = {
        "vision": next(model.backbone.model.vision_model.parameters()),
        "text": next(model.backbone.model.text_model.parameters()),
        "intent": model.core.intent_encoder.query_bank,
        "editor": model.core.editor.direction.weight,
        "visual_null": model.core.grounder.visual_null_key,
    }
    initial = {name: parameter.detach().float().clone() for name, parameter in tracked.items()}
    calls = {"vision_model": 0, "anchor_projection": 0, "intent": 0, "grounder": 0}

    def count(name):
        def hook(_module, _inputs):
            calls[name] += 1

        return hook

    handles = [
        model.backbone.model.vision_model.register_forward_pre_hook(count("vision_model")),
        model.backbone.anchor_projection.register_forward_pre_hook(count("anchor_projection")),
        model.core.intent_encoder.register_forward_pre_hook(count("intent")),
        model.core.grounder.register_forward_pre_hook(count("grounder")),
    ]
    history: defaultdict[str, list[float]] = defaultdict(list)
    action_counts = torch.zeros(5, dtype=torch.long)
    stop_step_counts = torch.zeros(3, dtype=torch.long)
    sample_steps = 0
    iterator = iter(loader)
    model.train()
    objective.train()
    try:
        for step_index in range(1, args.steps + 1):
            cpu_batch, iterator = _next_batch(iterator, loader)
            batch = cpu_batch.to(device)
            if batch.target_pixels is None or any(target is None for target in batch.target_ids):
                raise ValueError("canary batch requires target pixels and IDs")
            for key in calls:
                calls[key] = 0
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda",
                enabled=precision.autocast_enabled,
                dtype=precision.autocast_dtype,
            ):
                output = model(
                    batch.reference_pixels,
                    batch.input_ids,
                    batch.attention_mask,
                    batch.content_mask,
                )
                target_embeddings = model.encode_global_images(batch.target_pixels)
                positives = positive_mask_from_ids(
                    [str(target) for target in batch.target_ids], device
                )
                losses = objective(output, target_embeddings, positives)
            if not all(torch.isfinite(value).all() for value in losses.values()):
                raise FloatingPointError(f"non-finite canary loss at step {step_index}")
            scale_before = float(scaler.get_scale())
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            gradient_norms = {
                name: _gradient_norm(parameters)
                for name, parameters in parameter_families.items()
            }
            optimizer_parameters = [
                parameter
                for group in optimizer.param_groups
                for parameter in group["params"]
            ]
            amp_overflow_this_step = not _gradients_are_finite(optimizer_parameters)
            if amp_overflow_this_step:
                scale_after, step_parameter_deltas = _complete_amp_step(
                    scaler,
                    optimizer,
                    tracked,
                    scale_before=scale_before,
                    overflow=True,
                )
                minimum_scale = min(minimum_scale, scale_after)
                total_amp_overflows += 1
                consecutive_amp_overflows += 1
                skipped_optimizer_steps += 1
                overflow_record = {
                    "step": step_index,
                    "losses": {
                        name: float(losses[name].detach())
                        for name in ("terminal", "marginal", "total")
                    },
                    "gradient_norms": gradient_norms,
                    "amp_scale_before": scale_before,
                    "amp_scale_after": scale_after,
                    "amp_overflow_this_step": True,
                    "total_amp_overflows": total_amp_overflows,
                    "consecutive_amp_overflows": consecutive_amp_overflows,
                    "successful_optimizer_steps": successful_optimizer_steps,
                    "skipped_optimizer_steps": skipped_optimizer_steps,
                    "first_successful_step": first_successful_step,
                    "parameter_step_max_abs_delta": step_parameter_deltas,
                    "peak_vram_bytes": torch.cuda.max_memory_allocated(device),
                }
                print(json.dumps(overflow_record, sort_keys=True), flush=True)
                if consecutive_amp_overflows > 20:
                    raise RuntimeError(
                        "more than 20 consecutive AMP-overflowed canary iterations"
                    )
                if step_index >= 25 and successful_optimizer_steps == 0:
                    raise RuntimeError(
                        "no successful optimizer step after 25 attempted iterations"
                    )
                continue

            zero_families = [name for name, value in gradient_norms.items() if value == 0]
            if zero_families:
                raise RuntimeError(
                    f"zero expected gradient at step {step_index}: {zero_families}"
                )
            if max(gradient_norms.values()) > args.exploding_gradient_threshold:
                raise RuntimeError(
                    f"exploding gradient at step {step_index}: {gradient_norms}"
                )
            if calls != {
                "vision_model": 2,
                "anchor_projection": 1,
                "intent": 1,
                "grounder": 1,
            }:
                raise AssertionError(f"unexpected canary forward call counts: {calls}")
            _verify_absorbing_stop(output)
            scale_after, step_parameter_deltas = _complete_amp_step(
                scaler,
                optimizer,
                tracked,
                scale_before=scale_before,
                overflow=False,
            )
            minimum_scale = min(minimum_scale, scale_after)
            consecutive_amp_overflows = 0
            successful_optimizer_steps += 1
            if first_successful_step is None:
                first_successful_step = step_index

            diagnostics = _batch_diagnostics(output)
            actions = torch.stack([trace.selected_index.detach().cpu() for trace in output.trace], 1)
            action_counts += torch.bincount(actions.flatten(), minlength=5)
            stop_step_counts += actions.eq(4).sum(dim=0)
            sample_steps += actions.numel()
            for name in ("terminal", "marginal", "total"):
                history[name].append(float(losses[name].detach()))
            history["effect_rank"].append(float(diagnostics["functional_effect_rank"]))
            history["effect_cosine"].append(float(diagnostics["functional_delta_q_cosine"]))
            null_diagnostics = diagnostics["visual_null_probability"]
            if isinstance(null_diagnostics, dict):
                history["p_null_mean"].append(float(null_diagnostics["mean"]))
                history["p_null_max"].append(float(null_diagnostics["maximum"]))
            if (
                step_index == 1
                or step_index == first_successful_step
                or step_index % args.log_every == 0
                or step_index == args.steps
            ):
                record = {
                    "step": step_index,
                    "losses": {name: history[name][-1] for name in ("terminal", "marginal", "total")},
                    "gradient_norms": gradient_norms,
                    "amp_scale_before": scale_before,
                    "amp_scale_after": scale_after,
                    "amp_overflow_this_step": False,
                    "total_amp_overflows": total_amp_overflows,
                    "consecutive_amp_overflows": consecutive_amp_overflows,
                    "successful_optimizer_steps": successful_optimizer_steps,
                    "skipped_optimizer_steps": skipped_optimizer_steps,
                    "first_successful_step": first_successful_step,
                    "parameter_step_max_abs_delta": step_parameter_deltas,
                    "parameter_max_abs_delta": {
                        name: _parameter_delta(parameter, initial[name])
                        for name, parameter in tracked.items()
                    },
                    "vision_call_counts": dict(calls),
                    "diagnostics": diagnostics,
                    "peak_vram_bytes": torch.cuda.max_memory_allocated(device),
                }
                print(json.dumps(record, sort_keys=True), flush=True)
    except torch.OutOfMemoryError as error:
        print(
            json.dumps(
                {
                    "status": "OOM",
                    "peak_vram_bytes": torch.cuda.max_memory_allocated(device),
                    "message": str(error),
                }
            ),
            file=sys.stderr,
            flush=True,
        )
        raise
    finally:
        for handle in handles:
            handle.remove()

    distribution = action_counts.float() / action_counts.sum().clamp_min(1)
    stop_by_timestep = stop_step_counts.float() / (
        max(successful_optimizer_steps, 1) * args.batch_size
    )
    non_stop = action_counts[:4]
    candidate_distribution = non_stop.float() / non_stop.sum().clamp_min(1)
    collapse_flags = {
        "all_stop_t0": bool(stop_by_timestep[0] > 0.99),
        "never_stop": bool(action_counts[4] == 0),
        "single_candidate_monopoly": bool(candidate_distribution.max() > 0.99),
        "identical_candidate_effects": bool(
            _mean(history["effect_rank"]) <= 1.05
            or _mean(history["effect_cosine"]) >= 0.999
        ),
    }
    summary = {
        "status": "complete",
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "steps": args.steps,
        "attempted_steps": args.steps,
        "successful_optimizer_steps": successful_optimizer_steps,
        "skipped_optimizer_steps": skipped_optimizer_steps,
        "amp_overflow_count": total_amp_overflows,
        "initial_scale": initial_scale,
        "final_scale": float(scaler.get_scale()),
        "minimum_scale": minimum_scale,
        "first_successful_step": first_successful_step,
        "precision": args.precision,
        "query_cap": model.core.config.query_cap,
        "enable_visual_null": model.core.config.enable_visual_null,
        "p_null_mean_start": (
            history["p_null_mean"][0] if history["p_null_mean"] else None
        ),
        "p_null_mean_end": (
            history["p_null_mean"][-1] if history["p_null_mean"] else None
        ),
        "peak_vram_bytes": torch.cuda.max_memory_allocated(device),
        "loss_start": {
            name: history[name][0] if history[name] else None
            for name in ("terminal", "marginal", "total")
        },
        "loss_end": {
            name: history[name][-1] if history[name] else None
            for name in ("terminal", "marginal", "total")
        },
        "stop_by_timestep": stop_by_timestep.tolist(),
        "selected_distribution_with_stop": distribution.tolist(),
        "candidate_distribution_conditional_non_stop": candidate_distribution.tolist(),
        "collapse_flags": collapse_flags,
        "finite": True,
        "parameter_max_abs_delta": {
            name: _parameter_delta(parameter, initial[name])
            for name, parameter in tracked.items()
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if successful_optimizer_steps >= 10 and any(collapse_flags.values()):
        raise RuntimeError(f"canary collapse detector fired: {collapse_flags}")


if __name__ == "__main__":
    main()
