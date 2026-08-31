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
R1C1_ALL_STOP_T0_OPERATIONAL_THRESHOLD = 0.99


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


def _reproposal_audit_groups(
    reproposal: nn.Module,
) -> tuple[dict[str, list[nn.Parameter]], dict[str, nn.Parameter]]:
    """Return independently auditable R1c2 branches and representatives."""

    families = {
        "reproposal_output": list(reproposal.output_projection.parameters()),
        "reproposal_state": list(reproposal.state_attention.parameters()),
        "reproposal_change": list(reproposal.change_attention.parameters()),
        "reproposal_text": list(reproposal.text_attention.parameters()),
        "reproposal_state_query": list(
            reproposal.state_query_projection.parameters()
        ),
        "reproposal_fusion": list(reproposal.residual_hidden.parameters()),
    }
    representatives = {
        "reproposal_output": reproposal.output_projection.weight,
        "reproposal_state": reproposal.state_attention.in_proj_weight,
        "reproposal_change": reproposal.change_attention.in_proj_weight,
        "reproposal_text": reproposal.text_attention.in_proj_weight,
        "reproposal_state_query": reproposal.state_query_projection.weight,
        "reproposal_fusion": reproposal.residual_hidden[1].weight,
    }
    return families, representatives


def _semantic_claim_audit_groups(
    claim: nn.Module,
) -> tuple[dict[str, list[nn.Parameter]], dict[str, nn.Parameter]]:
    """Return R2 claim/firewall branches for cumulative learnability checks."""

    families = {
        "semantic_claim_output": list(claim.compatibility[-1].parameters()),
        "semantic_claim_query": list(claim.query_projection.parameters()),
        "semantic_claim_token": list(claim.token_projection.parameters()),
        "semantic_claim_state": list(claim.state_projection.parameters()),
        "semantic_claim_hidden": list(claim.compatibility[:-1].parameters()),
    }
    representatives = {
        "semantic_claim_output": claim.compatibility[-1].weight,
        "semantic_claim_query": claim.query_projection.weight,
        "semantic_claim_token": claim.token_projection.weight,
        "semantic_claim_state": claim.state_projection.weight,
        "semantic_claim_hidden": claim.compatibility[0].weight,
    }
    return families, representatives


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


def _classify_canary_outcomes(
    *,
    r1c1: bool,
    attempted_steps: int,
    successful_optimizer_steps: int,
    stop_t0_occupancy: float,
    total_stop_decisions: int,
    maximum_candidate_share: float,
    mean_effect_rank: float,
    mean_effect_cosine: float,
) -> tuple[dict[str, bool], dict[str, bool]]:
    """Separate implementation readiness from scientific behavior observations.

    Candidate/STOP specialization is a result of the experiment, not a prerequisite
    for a mechanically valid canary. R1c1/R1c2 immediate-STOP saturation is the
    one operational exception: it prevents the run from exercising changed-state
    recurrence, so the canary cannot validate its defining dynamic path.
    """

    mechanical_failure_flags = {
        "no_successful_optimizer_steps": successful_optimizer_steps == 0,
        "insufficient_successful_optimizer_steps": (
            attempted_steps >= 20 and successful_optimizer_steps < 20
        ),
        "all_stop_t0_prevents_recurrence_exercise": (
            r1c1
            and stop_t0_occupancy > R1C1_ALL_STOP_T0_OPERATIONAL_THRESHOLD
        ),
    }
    scientific_warning_flags = {
        "never_stop": total_stop_decisions == 0,
        "single_candidate_monopoly": maximum_candidate_share > 0.99,
        "identical_candidate_effects": (
            mean_effect_rank <= 1.05 or mean_effect_cosine >= 0.999
        ),
    }
    return mechanical_failure_flags, scientific_warning_flags


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
        "temporal_grounding": _dynamic_grounding_diagnostics(output),
        "temporal_intent": _dynamic_intent_diagnostics(output),
        "visual_null_probability": _dynamic_applicability_diagnostics(output),
        "semantic_residual": _semantic_residual_diagnostics(output),
    }


def _semantic_residual_diagnostics(output) -> dict[str, object] | None:
    residuals = output.temporal_semantic_residuals
    claims = output.temporal_semantic_claims
    if residuals is None or claims is None:
        return None
    residuals = residuals.detach().float()
    claims = claims.detach().float()
    rho_means = [float(residuals[:, timestep].mean()) for timestep in range(residuals.shape[1])]
    consumption = []
    claim_cosines = []
    for timestep, step in enumerate(output.trace):
        live = step.live_before
        current_claims = claims[live, timestep]
        if current_claims.numel():
            matrix = torch.nn.functional.cosine_similarity(
                current_claims[:, :, None], current_claims[:, None, :], dim=-1
            )
            off_diagonal = ~torch.eye(
                current_claims.shape[1], dtype=torch.bool, device=matrix.device
            )
            claim_cosines.append(float(matrix[:, off_diagonal].mean()))
        executed = live & step.selected_index.lt(claims.shape[2])
        if executed.any():
            consumption.append(
                step.selected_semantic_consumption[executed].detach().float().sum(-1)
            )
    consumed = torch.cat(consumption) if consumption else torch.empty(0)
    return {
        "rho_mean_by_state": rho_means,
        "selected_consumption_mass_mean": (
            float(consumed.mean()) if consumed.numel() else None
        ),
        "claim_cosine_between_candidates_by_timestep": claim_cosines,
        "claim_dtype": str(claims.dtype),
        "residual_dtype": str(residuals.dtype),
        "same_parent_residual": all(
            step.parent_semantic_residual is not None
            and step.candidate_semantic_residuals is not None
            for step in output.trace
        ),
    }


def _dynamic_intent_diagnostics(output) -> dict[str, object] | None:
    if output.temporal_intents is None or output.initial_intents is None:
        return None
    intents = output.temporal_intents.detach().float()
    base = output.initial_intents.detach().float()
    per_timestep = []
    for timestep, step in enumerate(output.trace):
        valid = step.live_before
        current = intents[valid, timestep]
        if not current.numel():
            per_timestep.append(None)
            continue
        pairwise = torch.nn.functional.cosine_similarity(
            current[:, :, None], current[:, None, :], dim=-1
        )
        off_diagonal = ~torch.eye(
            current.shape[1], dtype=torch.bool, device=current.device
        )[None]
        per_timestep.append(
            {
                "live_parent_count": int(valid.sum()),
                "pairwise_candidate_cosine": float(
                    pairwise[off_diagonal.expand_as(pairwise)].mean()
                ),
                "mean_residual_norm_from_base": float(
                    (current - base[valid]).norm(dim=-1).mean()
                ),
            }
        )
    transitions = []
    for timestep in range(len(output.trace) - 1):
        valid = output.trace[timestep + 1].live_before
        displacement = intents[valid, timestep + 1] - intents[valid, timestep]
        if not displacement.numel():
            transitions.append(None)
            continue
        pairwise_displacement = torch.nn.functional.cosine_similarity(
            displacement[:, :, None], displacement[:, None, :], dim=-1
        )
        off_diagonal = ~torch.eye(
            displacement.shape[1], dtype=torch.bool, device=displacement.device
        )[None]
        transitions.append(
            {
                "live_parent_count": int(valid.sum()),
                "mean_intent_l2_change": float(displacement.norm(dim=-1).mean()),
                "intent_displacement_cosine": float(
                    pairwise_displacement[
                        off_diagonal.expand_as(pairwise_displacement)
                    ].mean()
                ),
            }
        )
    return {
        "enable_dynamic_reproposal": output.dynamic_reproposal,
        "per_timestep": per_timestep,
        "per_transition": transitions,
    }


def _dynamic_grounding_diagnostics(output) -> dict[str, object]:
    if output.temporal_supports is None:
        raise AssertionError("canary output is missing temporal support trace")
    supports = output.temporal_supports.detach().float()
    per_timestep = []
    for timestep, step in enumerate(output.trace):
        valid = step.live_before
        current = supports[valid, timestep]
        if not current.numel():
            per_timestep.append(None)
            continue
        pairwise = torch.nn.functional.cosine_similarity(
            current[:, :, None], current[:, None, :], dim=-1
        )
        off_diagonal = ~torch.eye(
            current.shape[1], dtype=torch.bool, device=current.device
        )[None]
        per_timestep.append(
            {
                "live_parent_count": int(valid.sum()),
                "support_mass_max_abs_error": float(
                    (current.sum(dim=-1) - 1.0).abs().max()
                ),
                "support_fraction": float((current > 0).float().mean()),
                "support_entropy": float(
                    -(current * current.clamp_min(1e-8).log()).sum(dim=-1).mean()
                ),
                "between_candidate_cosine": float(
                    pairwise[off_diagonal.expand_as(pairwise)].mean()
                ),
            }
        )
    transitions = []
    for transition in range(len(output.trace) - 1):
        valid = output.trace[transition + 1].live_before
        before = supports[valid, transition]
        after = supports[valid, transition + 1]
        transitions.append(
            {
                "live_parent_count": int(valid.sum()),
                "same_candidate_temporal_cosine": (
                    float(torch.nn.functional.cosine_similarity(before, after, dim=-1).mean())
                    if before.numel()
                    else None
                ),
                "support_l1_change": (
                    float((after - before).abs().sum(dim=-1).mean())
                    if before.numel()
                    else None
                ),
            }
        )
    return {
        "enable_dynamic_regrounding": output.dynamic_regrounding,
        "per_timestep": per_timestep,
        "per_transition": transitions,
        "support_dtype": str(output.temporal_supports.dtype),
    }


def _dynamic_applicability_diagnostics(output) -> dict[str, object] | None:
    if output.visual_null_probabilities is None or output.visual_confidence is None:
        return None

    def statistics(values: Tensor) -> dict[str, float]:
        values = values.detach().float().flatten()
        return {
            "mean": float(values.mean()),
            "std": float(values.std(unbiased=False)),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
        }

    live_confidence: list[Tensor] = []
    live_null: list[Tensor] = []
    confidence_by_timestep: list[dict[str, float] | None] = []
    null_by_timestep: list[dict[str, float] | None] = []
    candidate_confidence_std: list[Tensor] = []
    candidate_null_std: list[Tensor] = []
    ungated_norms: list[Tensor] = []
    gated_norms: list[Tensor] = []
    actuator_errors: list[Tensor] = []
    for step in output.trace:
        if step.visual_confidence is None or step.visual_null_probability is None:
            return None
        valid = step.live_before
        if not valid.any():
            confidence_by_timestep.append(None)
            null_by_timestep.append(None)
            continue
        confidence = step.visual_confidence[valid].detach().float()
        null = step.visual_null_probability[valid].detach().float()
        live_confidence.append(confidence.flatten())
        live_null.append(null.flatten())
        confidence_by_timestep.append(statistics(confidence))
        null_by_timestep.append(statistics(null))
        candidate_confidence_std.append(confidence.std(dim=-1, unbiased=False))
        candidate_null_std.append(null.std(dim=-1, unbiased=False))
        if step.ungated_delta_z is None:
            raise AssertionError("R1b canary requires the ungated editor effect trace")
        ungated = step.ungated_delta_z[valid].detach().float()
        gated = step.delta_z[valid].detach().float()
        reconstructed = ungated * confidence[..., None, None]
        ungated_norms.append(ungated.flatten(2).norm(dim=-1).flatten())
        gated_norms.append(gated.flatten(2).norm(dim=-1).flatten())
        actuator_errors.append((gated - reconstructed).abs().flatten())

    if not live_confidence:
        return None
    all_confidence = torch.cat(live_confidence)
    all_null = torch.cat(live_null)
    temporal_changes: list[Tensor] = []
    signed_temporal_changes: list[float | None] = []
    for previous, current in zip(output.trace[:-1], output.trace[1:], strict=True):
        if previous.visual_confidence is None or current.visual_confidence is None:
            return None
        valid = current.live_before
        if not valid.any():
            signed_temporal_changes.append(None)
            continue
        change = (
            current.visual_confidence[valid].detach().float()
            - previous.visual_confidence[valid].detach().float()
        )
        temporal_changes.append(change.flatten())
        signed_temporal_changes.append(float(change.mean()))
    absolute_temporal = (
        torch.cat(temporal_changes).abs() if temporal_changes else torch.empty(0)
    )
    confidence_stats = statistics(all_confidence)
    null_stats = statistics(all_null)
    return {
        "p_null_mean": null_stats["mean"],
        "p_null_std": null_stats["std"],
        "p_null_minimum": null_stats["minimum"],
        "p_null_maximum": null_stats["maximum"],
        "confidence_mean": confidence_stats["mean"],
        "confidence_std": confidence_stats["std"],
        "confidence_minimum": confidence_stats["minimum"],
        "confidence_maximum": confidence_stats["maximum"],
        "confidence_mean_by_timestep": [
            None if item is None else item["mean"] for item in confidence_by_timestep
        ],
        "confidence_std_by_timestep": [
            None if item is None else item["std"] for item in confidence_by_timestep
        ],
        "confidence_min_by_timestep": [
            None if item is None else item["minimum"] for item in confidence_by_timestep
        ],
        "confidence_max_by_timestep": [
            None if item is None else item["maximum"] for item in confidence_by_timestep
        ],
        "p_null_mean_by_timestep": [
            None if item is None else item["mean"] for item in null_by_timestep
        ],
        "p_null_std_by_timestep": [
            None if item is None else item["std"] for item in null_by_timestep
        ],
        "p_null_min_by_timestep": [
            None if item is None else item["minimum"] for item in null_by_timestep
        ],
        "p_null_max_by_timestep": [
            None if item is None else item["maximum"] for item in null_by_timestep
        ],
        "confidence_candidate_std_mean": float(
            torch.cat(candidate_confidence_std).mean()
        ),
        "p_null_candidate_std_mean": float(torch.cat(candidate_null_std).mean()),
        "mean_abs_confidence_temporal_change": (
            float(absolute_temporal.mean()) if absolute_temporal.numel() else None
        ),
        "max_abs_confidence_temporal_change": (
            float(absolute_temporal.max()) if absolute_temporal.numel() else None
        ),
        "mean_signed_confidence_change_t0_to_t1": signed_temporal_changes[0],
        "mean_signed_confidence_change_t1_to_t2": signed_temporal_changes[1],
        "ungated_base_delta_z_norm": float(torch.cat(ungated_norms).mean()),
        "gated_delta_z_norm": float(torch.cat(gated_norms).mean()),
        "confidence_to_delta_scale_error": float(
            torch.cat(actuator_errors).max()
        ),
        "applicability_logit_dtype": str(output.trace[0].applicability_logits.dtype),
        "confidence_dtype": str(output.trace[0].visual_confidence.dtype),
        "p_null_dtype": str(output.trace[0].visual_null_probability.dtype),
        "delta_z_dtype": str(output.trace[0].delta_z.dtype),
        "candidate_state_dtype": str(output.trace[0].candidate_states.dtype),
        "population": "candidate actions of samples live before each timestep",
    }


def _build_canary_model(
    device: torch.device,
    *,
    dynamic_regrounding: bool = False,
    dynamic_reproposal: bool = False,
    semantic_residual: bool = False,
) -> tuple[IAGSRME, object, object]:
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
            enable_dynamic_applicability=not dynamic_regrounding,
            initial_applicability=0.98,
            grounding_normalization="entmax15",
            enable_dynamic_regrounding=dynamic_regrounding,
            enable_dynamic_reproposal=dynamic_reproposal,
            enable_semantic_residual=semantic_residual,
            initial_claim_probability=0.99,
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
        default=os.environ.get("FASHIONIQ_ROOT", "data/FashionIQ"),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--exploding-gradient-threshold", type=float, default=1e4)
    parser.add_argument("--accelerator-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    causal_mode = parser.add_mutually_exclusive_group()
    causal_mode.add_argument(
        "--r1c1",
        action="store_true",
        help="canary R1c1 fixed-WHAT/current-state-WHERE instead of R1b applicability",
    )
    causal_mode.add_argument(
        "--r1c2",
        action="store_true",
        help="canary R1c2 dynamic WHAT plus current-state dynamic WHERE",
    )
    causal_mode.add_argument(
        "--r2",
        action="store_true",
        help="canary R2 semantic residual/claim firewall with dynamic WHERE",
    )
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
    model, tokenizer, processor = _build_canary_model(
        device,
        dynamic_regrounding=args.r1c1 or args.r1c2 or args.r2,
        dynamic_reproposal=args.r1c2,
        semantic_residual=args.r2,
    )
    if model.core.config.query_cap != 1000.0:
        raise AssertionError("R1a/R1b/R1c1/R1c2/R2 canary requires query_cap=1000")
    if args.r2:
        if (
            not model.core.config.enable_semantic_residual
            or not model.core.config.enable_dynamic_regrounding
            or model.core.config.enable_dynamic_reproposal
            or model.core.config.enable_dynamic_applicability
        ):
            raise AssertionError(
                "R2 requires claim firewall/dynamic WHERE with reproposal/applicability off"
            )
    elif args.r1c2:
        if (
            not model.core.config.enable_dynamic_regrounding
            or not model.core.config.enable_dynamic_reproposal
            or model.core.config.enable_dynamic_applicability
        ):
            raise AssertionError(
                "R1c2 requires dynamic WHAT/WHERE with R1b applicability off"
            )
    elif args.r1c1:
        if (
            not model.core.config.enable_dynamic_regrounding
            or model.core.config.enable_dynamic_applicability
        ):
            raise AssertionError("R1c1 requires dynamic WHERE with R1b applicability off")
    elif not model.core.config.enable_dynamic_applicability:
        raise AssertionError("R1b canary requires dynamic applicability enabled")
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
        "editor": list(model.core.editor.parameters()),
        "readout": list(model.core.readout.parameters()),
        "scorer": list(model.core.scorer.parameters()),
    }
    if model.core.applicability_head is not None:
        parameter_families["applicability"] = list(
            model.core.applicability_head.parameters()
        )
    reproposal_families: dict[str, list[nn.Parameter]] = {}
    reproposal_representatives: dict[str, nn.Parameter] = {}
    if model.core.reproposal is not None:
        (
            reproposal_families,
            reproposal_representatives,
        ) = _reproposal_audit_groups(model.core.reproposal)
        parameter_families.update(reproposal_families)
    semantic_claim_families: dict[str, list[nn.Parameter]] = {}
    semantic_claim_representatives: dict[str, nn.Parameter] = {}
    if model.core.semantic_claim is not None:
        (
            semantic_claim_families,
            semantic_claim_representatives,
        ) = _semantic_claim_audit_groups(model.core.semantic_claim)
        semantic_claim_families["residual_conditioned_intent"] = list(
            model.core.intent_encoder.parameters()
        )
        semantic_claim_representatives["residual_conditioned_intent"] = (
            model.core.intent_encoder.cross_attention.in_proj_weight
        )
        parameter_families.update(semantic_claim_families)
    tracked = {
        "vision": next(model.backbone.model.vision_model.parameters()),
        "text": next(model.backbone.model.text_model.parameters()),
        "intent": model.core.intent_encoder.query_bank,
        "editor": model.core.editor.direction.weight,
        "grounding_projection": model.core.grounder.anchor_projection.weight,
    }
    if model.core.applicability_head is not None:
        tracked["applicability_weight"] = model.core.applicability_head.projection.weight
        tracked["applicability_bias"] = model.core.applicability_head.projection.bias
    tracked.update(reproposal_representatives)
    tracked.update(semantic_claim_representatives)
    initial = {name: parameter.detach().float().clone() for name, parameter in tracked.items()}
    calls = {
        "vision_model": 0,
        "anchor_projection": 0,
        "intent": 0,
        "grounder": 0,
        "applicability": 0,
        "reproposal": 0,
        "semantic_claim": 0,
    }

    def count(name):
        def hook(_module, _inputs):
            calls[name] += 1

        return hook

    handles = [
        model.backbone.model.vision_model.register_forward_pre_hook(count("vision_model")),
        model.backbone.anchor_projection.register_forward_pre_hook(count("anchor_projection")),
        model.core.intent_encoder.cross_attention.register_forward_pre_hook(
            count("intent")
        ),
        model.core.grounder.register_forward_pre_hook(count("grounder")),
    ]
    if model.core.applicability_head is not None:
        handles.append(
            model.core.applicability_head.register_forward_pre_hook(
                count("applicability")
            )
        )
    if model.core.reproposal is not None:
        handles.append(
            model.core.reproposal.register_forward_pre_hook(count("reproposal"))
        )
    if model.core.semantic_claim is not None:
        handles.append(
            model.core.semantic_claim.register_forward_pre_hook(
                count("semantic_claim")
            )
        )
    history: defaultdict[str, list[float]] = defaultdict(list)
    action_counts = torch.zeros(5, dtype=torch.long)
    stop_step_counts = torch.zeros(3, dtype=torch.long)
    sample_steps = 0
    successful_steps_with_nonzero_applicability_gradient = 0
    successful_steps_with_nonzero_grounding_gradient = 0
    reproposal_nonzero_gradient_steps = {
        name: 0 for name in reproposal_families
    }
    semantic_claim_nonzero_gradient_steps = {
        name: 0 for name in semantic_claim_families
    }
    latest_applicability_diagnostics: dict[str, object] | None = None
    latest_grounding_diagnostics: dict[str, object] | None = None
    latest_intent_diagnostics: dict[str, object] | None = None
    max_observed_applicability_variation = 0.0
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

            applicability_gradient = gradient_norms.get("applicability", 0.0)
            if applicability_gradient > 0:
                successful_steps_with_nonzero_applicability_gradient += 1
            if gradient_norms["grounding"] > 0:
                successful_steps_with_nonzero_grounding_gradient += 1
            for name in reproposal_nonzero_gradient_steps:
                if gradient_norms.get(name, 0.0) > 0:
                    reproposal_nonzero_gradient_steps[name] += 1
            for name in semantic_claim_nonzero_gradient_steps:
                if gradient_norms.get(name, 0.0) > 0:
                    semantic_claim_nonzero_gradient_steps[name] += 1
            allowed_zero_families = {"applicability"}
            if args.r1c1 or args.r1c2 or args.r2:
                # Dynamic-WHERE readiness is based on cumulative grounder learnability;
                # one minibatch with zero grounder gradient is not a failure.
                allowed_zero_families.add("grounding")
            if args.r1c2:
                # Zero-init W_out intentionally blocks upstream reproposal gradients
                # until the output projection has taken an optimizer step.
                allowed_zero_families.update(reproposal_families)
            if args.r2:
                # The zero-initialized claim output may initially block upstream
                # claim projections; readiness is checked cumulatively after updates.
                allowed_zero_families.update(semantic_claim_families)
            zero_families = [
                name
                for name, value in gradient_norms.items()
                if value == 0 and name not in allowed_zero_families
            ]
            if zero_families:
                raise RuntimeError(
                    f"zero expected gradient at step {step_index}: {zero_families}"
                )
            if max(gradient_norms.values()) > args.exploding_gradient_threshold:
                raise RuntimeError(
                    f"exploding gradient at step {step_index}: {gradient_norms}"
                )
            expected_calls = {
                "vision_model": 2,
                "anchor_projection": 1,
                "intent": 1,
                "grounder": (
                    model.core.config.max_steps
                    if args.r1c1 or args.r1c2 or args.r2
                    else 1
                ),
                "applicability": (
                    0
                    if args.r1c1 or args.r1c2 or args.r2
                    else model.core.config.max_steps
                ),
                "reproposal": model.core.config.max_steps - 1 if args.r1c2 else 0,
                "semantic_claim": model.core.config.max_steps if args.r2 else 0,
            }
            if args.r2:
                expected_calls["intent"] = model.core.config.max_steps
            if calls != expected_calls:
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
            temporal_grounding = diagnostics["temporal_grounding"]
            if isinstance(temporal_grounding, dict):
                latest_grounding_diagnostics = temporal_grounding
            temporal_intent = diagnostics["temporal_intent"]
            if isinstance(temporal_intent, dict):
                latest_intent_diagnostics = temporal_intent
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
                latest_applicability_diagnostics = null_diagnostics
                max_observed_applicability_variation = max(
                    max_observed_applicability_variation,
                    float(null_diagnostics["confidence_std"]),
                    float(
                        null_diagnostics["max_abs_confidence_temporal_change"] or 0.0
                    ),
                    float(null_diagnostics["confidence_candidate_std_mean"]),
                )
                history["p_null_mean"].append(float(null_diagnostics["p_null_mean"]))
                history["p_null_max"].append(
                    float(null_diagnostics["p_null_maximum"])
                )
                history["p_null_min"].append(
                    float(null_diagnostics["p_null_minimum"])
                )
                history["confidence_mean"].append(
                    float(null_diagnostics["confidence_mean"])
                )
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
                    "successful_steps_with_nonzero_applicability_gradient": (
                        successful_steps_with_nonzero_applicability_gradient
                    ),
                    **{
                        f"successful_steps_with_nonzero_{name}_gradient": count
                        for name, count in reproposal_nonzero_gradient_steps.items()
                    },
                    **{
                        f"successful_steps_with_nonzero_{name}_gradient": count
                        for name, count in semantic_claim_nonzero_gradient_steps.items()
                    },
                    "applicability_nonzero_gradient_fraction": (
                        successful_steps_with_nonzero_applicability_gradient
                        / successful_optimizer_steps
                    ),
                    "parameter_max_abs_delta": {
                        name: _parameter_delta(parameter, initial[name])
                        for name, parameter in tracked.items()
                    },
                    "vision_call_counts": dict(calls),
                    "diagnostics": diagnostics,
                    "peak_vram_bytes": torch.cuda.max_memory_allocated(device),
                }
                print(json.dumps(record, sort_keys=True), flush=True)
            if successful_optimizer_steps >= 20 and not (
                args.r1c1 or args.r1c2 or args.r2
            ):
                applicability_delta = max(
                    _parameter_delta(
                        tracked["applicability_weight"],
                        initial["applicability_weight"],
                    ),
                    _parameter_delta(
                        tracked["applicability_bias"],
                        initial["applicability_bias"],
                    ),
                )
                if successful_steps_with_nonzero_applicability_gradient == 0:
                    raise RuntimeError(
                        "dynamic applicability received zero gradient on every successful step"
                    )
                if applicability_delta == 0:
                    raise RuntimeError(
                        "dynamic applicability parameters did not change after 20 successful steps"
                    )
                if isinstance(latest_applicability_diagnostics, dict):
                    variation_tolerance = 1e-7
                    if max_observed_applicability_variation <= variation_tolerance:
                        raise RuntimeError(
                            "applicability parameters changed but the FP32 actuator remained "
                            "numerically constant after 20 successful steps"
                        )
            if successful_optimizer_steps >= 20 and (
                args.r1c1 or args.r1c2 or args.r2
            ):
                if successful_steps_with_nonzero_grounding_gradient == 0:
                    raise RuntimeError(
                        "dynamic-WHERE grounder received zero gradient on every successful step"
                    )
                if _parameter_delta(
                    tracked["grounding_projection"], initial["grounding_projection"]
                ) == 0:
                    raise RuntimeError("dynamic-WHERE grounder parameters did not change")
            if successful_optimizer_steps >= 20 and args.r1c2:
                dead_gradient_branches = [
                    name
                    for name, count in reproposal_nonzero_gradient_steps.items()
                    if count == 0
                ]
                if dead_gradient_branches:
                    raise RuntimeError(
                        "R1c2 reproposal branches received no cumulative gradient: "
                        f"{dead_gradient_branches}"
                    )
                unmoved_branches = [
                    name
                    for name in reproposal_representatives
                    if _parameter_delta(tracked[name], initial[name]) == 0
                ]
                if unmoved_branches:
                    raise RuntimeError(
                        "R1c2 reproposal branch parameters did not move: "
                        f"{unmoved_branches}"
                    )
            if successful_optimizer_steps >= 20 and args.r2:
                dead_claim_branches = [
                    name
                    for name, count in semantic_claim_nonzero_gradient_steps.items()
                    if count == 0
                ]
                if dead_claim_branches:
                    raise RuntimeError(
                        "R2 semantic-claim branches received no cumulative gradient: "
                        f"{dead_claim_branches}"
                    )
                unmoved_claim_branches = [
                    name
                    for name in semantic_claim_representatives
                    if _parameter_delta(tracked[name], initial[name]) == 0
                ]
                if unmoved_claim_branches:
                    raise RuntimeError(
                        "R2 semantic-claim branch parameters did not move: "
                        f"{unmoved_claim_branches}"
                    )
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
    mechanical_failure_flags, scientific_warning_flags = _classify_canary_outcomes(
        r1c1=args.r1c1 or args.r1c2 or args.r2,
        attempted_steps=args.steps,
        successful_optimizer_steps=successful_optimizer_steps,
        stop_t0_occupancy=float(stop_by_timestep[0]),
        total_stop_decisions=int(action_counts[4]),
        maximum_candidate_share=float(candidate_distribution.max()),
        mean_effect_rank=_mean(history["effect_rank"]),
        mean_effect_cosine=_mean(history["effect_cosine"]),
    )
    grounder_delta = _parameter_delta(
        tracked["grounding_projection"], initial["grounding_projection"]
    )
    mechanical_failure_flags.update(
        {
            # These checks abort at the point of failure. False in a completed report
            # records what the successful canary actually established.
            "non_finite": False,
            "unrecoverable_amp_overflow": False,
            "incorrect_forward_call_count": False,
            "same_parent_invariant_failure": False,
            "support_normalization_failure": False,
            "r1b_applicability_active": bool(
                (args.r1c1 or args.r1c2 or args.r2)
                and model.core.config.enable_dynamic_applicability
            ),
            "grounder_no_gradient": bool(
                (args.r1c1 or args.r1c2 or args.r2)
                and successful_optimizer_steps > 0
                and successful_steps_with_nonzero_grounding_gradient == 0
            ),
            "grounder_no_parameter_movement": bool(
                (args.r1c1 or args.r1c2 or args.r2)
                and successful_optimizer_steps >= 20
                and grounder_delta == 0.0
            ),
            "r1c2_dynamic_reproposal_disabled": bool(
                args.r1c2 and not model.core.config.enable_dynamic_reproposal
            ),
            "r2_semantic_residual_disabled": bool(
                args.r2 and not model.core.config.enable_semantic_residual
            ),
            **{
                f"{name}_no_gradient": bool(
                    args.r1c2
                    and successful_optimizer_steps >= 20
                    and reproposal_nonzero_gradient_steps[name] == 0
                )
                for name in reproposal_families
            },
            **{
                f"{name}_no_parameter_movement": bool(
                    args.r1c2
                    and successful_optimizer_steps >= 20
                    and _parameter_delta(tracked[name], initial[name]) == 0.0
                )
                for name in reproposal_representatives
            },
            **{
                f"{name}_no_gradient": bool(
                    args.r2
                    and successful_optimizer_steps >= 20
                    and semantic_claim_nonzero_gradient_steps[name] == 0
                )
                for name in semantic_claim_families
            },
            **{
                f"{name}_no_parameter_movement": bool(
                    args.r2
                    and successful_optimizer_steps >= 20
                    and _parameter_delta(tracked[name], initial[name]) == 0.0
                )
                for name in semantic_claim_representatives
            },
        }
    )
    if args.r1c2 and isinstance(latest_intent_diagnostics, dict):
        intent_steps = [
            item
            for item in latest_intent_diagnostics["per_timestep"]
            if isinstance(item, dict)
        ]
        intent_transitions = [
            item
            for item in latest_intent_diagnostics["per_transition"]
            if isinstance(item, dict)
        ]
        scientific_warning_flags.update(
            {
                "dynamic_intents_nearly_static": all(
                    float(item["mean_residual_norm_from_base"]) <= 1e-7
                    for item in intent_steps[1:]
                ),
                "high_intent_candidate_similarity": any(
                    float(item["pairwise_candidate_cosine"]) >= 0.999
                    for item in intent_steps
                ),
                "high_intent_displacement_comotion": any(
                    float(item["intent_displacement_cosine"]) >= 0.999
                    for item in intent_transitions
                ),
            }
        )
    latest_semantic_residual = (
        _semantic_residual_diagnostics(output) if args.r2 else None
    )
    if args.r2 and isinstance(latest_semantic_residual, dict):
        rho = latest_semantic_residual["rho_mean_by_state"]
        claim_cosines = latest_semantic_residual[
            "claim_cosine_between_candidates_by_timestep"
        ]
        scientific_warning_flags.update(
            {
                "high_claim_clone": bool(
                    claim_cosines and max(claim_cosines) >= 0.999
                ),
                "residual_unused": max(rho) - min(rho) <= 1e-7,
                "consume_all_after_first_action": (
                    len(rho) > 1 and rho[1] <= 0.01 * max(rho[0], 1e-8)
                ),
                "global_consumption_warning": bool(
                    claim_cosines and min(claim_cosines) >= 0.999
                ),
            }
        )
    mechanical_status = (
        "FAIL" if any(mechanical_failure_flags.values()) else "PASS"
    )
    summary = {
        "status": "complete",
        "mechanical_status": mechanical_status,
        "mechanical_failure_flags": mechanical_failure_flags,
        "scientific_warning_flags": scientific_warning_flags,
        "scientific_warning_status": (
            "observations only; these flags do not invalidate mechanical readiness "
            "for full training"
        ),
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
        "enable_dynamic_applicability": (
            model.core.config.enable_dynamic_applicability
        ),
        "enable_dynamic_regrounding": model.core.config.enable_dynamic_regrounding,
        "enable_dynamic_reproposal": model.core.config.enable_dynamic_reproposal,
        "enable_semantic_residual": model.core.config.enable_semantic_residual,
        "initial_applicability": model.core.config.initial_applicability,
        "successful_steps_with_nonzero_applicability_gradient": (
            successful_steps_with_nonzero_applicability_gradient
        ),
        "successful_steps_with_nonzero_grounding_gradient": (
            successful_steps_with_nonzero_grounding_gradient
        ),
        **{
            f"successful_steps_with_nonzero_{name}_gradient": count
            for name, count in reproposal_nonzero_gradient_steps.items()
        },
        **{
            f"successful_steps_with_nonzero_{name}_gradient": count
            for name, count in semantic_claim_nonzero_gradient_steps.items()
        },
        "semantic_claim_nonzero_gradient_fraction": {
            name: count / max(successful_optimizer_steps, 1)
            for name, count in semantic_claim_nonzero_gradient_steps.items()
        },
        "grounding_nonzero_gradient_fraction": (
            successful_steps_with_nonzero_grounding_gradient
            / max(successful_optimizer_steps, 1)
        ),
        "applicability_nonzero_gradient_fraction": (
            successful_steps_with_nonzero_applicability_gradient
            / max(successful_optimizer_steps, 1)
        ),
        "applicability_variation_tolerance": 1e-7,
        "max_observed_applicability_variation": (
            max_observed_applicability_variation
        ),
        "p_null_mean_start": (
            history["p_null_mean"][0] if history["p_null_mean"] else None
        ),
        "p_null_mean_end": (
            history["p_null_mean"][-1] if history["p_null_mean"] else None
        ),
        "dynamic_applicability_end": latest_applicability_diagnostics,
        "dynamic_grounding_end": latest_grounding_diagnostics,
        "dynamic_intent_end": latest_intent_diagnostics,
        "semantic_residual_end": latest_semantic_residual,
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
        "finite": True,
        "parameter_max_abs_delta": {
            name: _parameter_delta(parameter, initial[name])
            for name, parameter in tracked.items()
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    failed_mechanical_checks = [
        name for name, failed in mechanical_failure_flags.items() if failed
    ]
    if failed_mechanical_checks:
        raise RuntimeError(
            "canary mechanical readiness failed: "
            f"{failed_mechanical_checks}. Immediate STOP saturation is operationally "
            "invalid only because it prevents exercising changed-state recurrent "
            "regrounding; it is not a scientific judgment against STOP."
        )


if __name__ == "__main__":
    main()
