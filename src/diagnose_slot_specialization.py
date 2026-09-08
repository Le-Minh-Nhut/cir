"""Read-only t=0 functional specialization and gradient-exposure diagnostic."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset

from data.images import FashionIQImageCollator
from datasets.common import DirectoryImageStore
from datasets.fashioniq import FashionIQDataset
from diagnose_candidate_selector import build_objective_from_checkpoint
from diagnostics.cohort import (
    load_or_create_manifest,
    sample_ids_fingerprint,
    teacher_batch_fingerprint,
    validate_processed_manifest,
)
from diagnostics.specialization import (
    concept_mil_responsibility,
    conditional_geometry_summary,
    functional_specialization_summary,
    gradient_interference_summary,
    semantic_specialization_summary,
    summarize_slot_gradients,
)
from evaluate import validate_checkpoint_backbone_metadata
from models.iag_srme.utils.retrieval import (
    build_teacher_masks,
    marginal_teacher_utilities,
)
from models.iag_srme.utils.semantic import PARSER_VERSION, parse_instruction_concepts
from runtime import configure_torch_runtime, resolve_device, seed_everything
from train import CATEGORIES, build_model, git_identity
from training.engine import resolve_precision


LOSS_COMPONENTS = {
    "terminal": "terminal",
    "concept": "concept_loss",
    "bind": "bind_loss",
    "dpp": "dpp_raw",
    "pair": "pair",
    "gain": "gain",
}

LOSS_COEFFICIENTS = {
    "terminal": "terminal_weight",
    "concept": "lambda_c",
    "bind": "lambda_bind",
    "dpp": "lambda_dpp",
    "pair": "lambda_pair",
    "gain": "lambda_gain",
}


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def _state_signature(module: nn.Module) -> str:
    """Low-memory fingerprint used only to assert this diagnostic did not mutate state."""

    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        tensor = value.detach().float()
        record = (
            name,
            tuple(value.shape),
            str(value.dtype),
            float(tensor.sum()),
            float(tensor.square().sum()),
        )
        digest.update(repr(record).encode("utf-8"))
    return digest.hexdigest()


def _resolve_sample_count(value: object, dataset_size: int) -> tuple[int, str]:
    requested = str(value)
    if requested.lower() == "all":
        return dataset_size, "all"
    count = int(requested)
    if count <= 0:
        raise ValueError("diagnostic_samples must be 'all' or a positive integer")
    return min(count, dataset_size), requested


def _validate_checkpoint(cfg: DictConfig, checkpoint: Mapping[str, Any]) -> list[dict[str, object]]:
    return validate_checkpoint_backbone_metadata(
        checkpoint.get("metadata"),
        str(cfg.backbone.checkpoint),
        str(cfg.backbone.revision),
        str(cfg.backbone.global_readout_mode),
        expected_readout_experiment=str(cfg.backbone.readout_experiment),
        expected_finetune_policy=str(cfg.backbone.finetune_policy),
        expected_train_vision=bool(cfg.backbone.train_vision),
        expected_train_text=bool(cfg.backbone.train_text),
        expected_train_text_projection=bool(cfg.backbone.train_text_projection),
        expected_model_config={
            "num_candidates": int(cfg.model.num_candidates),
            "max_steps": int(cfg.model.max_steps),
            "stop_enabled": bool(cfg.model.stop_enabled),
            "epsilon_stop": float(cfg.model.epsilon_stop),
        },
        allow_counterfactual=bool(cfg.get("allow_counterfactual_eval", False)),
    )


def _mil_summary(
    positive_responsibilities: list[Tensor],
    responsibility_by_concept: Mapping[str, list[Tensor]],
    *,
    candidates: int,
    min_support: int,
) -> dict[str, Any]:
    if not positive_responsibilities:
        return {
            "available": False,
            "reason": "no positive instruction concepts were represented by the concept vocabulary",
        }
    values = torch.cat(positive_responsibilities).float()
    entropy = -(values * values.clamp_min(1e-12).log()).sum(dim=-1)
    top = values.argmax(dim=-1)
    conditional = []
    insufficient = []
    for concept in sorted(responsibility_by_concept):
        concept_values = torch.stack(responsibility_by_concept[concept]).float()
        support = concept_values.shape[0]
        if support < min_support:
            insufficient.append({"concept": concept, "support_count": support})
            continue
        concept_top = concept_values.argmax(dim=-1)
        conditional.append(
            {
                "concept": concept,
                "support_count": support,
                "mean_responsibility_per_slot": concept_values.mean(dim=0).tolist(),
                "median_responsibility_per_slot": concept_values.median(dim=0).values.tolist(),
                "mean_responsibility_entropy": float(
                    -(
                        concept_values * concept_values.clamp_min(1e-12).log()
                    ).sum(dim=-1).mean()
                ),
                "top_responsibility_slot_occupancy": [
                    float(concept_top.eq(slot).float().mean()) for slot in range(candidates)
                ],
            }
        )
    return {
        "available": True,
        "positive_concept_occurrence_count": values.shape[0],
        "mean_responsibility_per_slot": values.mean(dim=0).tolist(),
        "median_responsibility_per_slot": values.median(dim=0).values.tolist(),
        "mean_responsibility_entropy": float(entropy.mean()),
        "median_responsibility_entropy": float(entropy.median()),
        "top_responsibility_slot_occupancy": [
            float(top.eq(slot).float().mean()) for slot in range(candidates)
        ],
        "conditional_by_concept": conditional,
        "insufficient_support": insufficient,
    }


def _gradient_for(
    loss: Tensor,
    proposals: Tensor,
    query_rows: Tensor,
    *,
    retain_graph: bool,
) -> tuple[Tensor, Tensor]:
    if not loss.requires_grad:
        return torch.zeros_like(proposals), torch.zeros_like(query_rows)
    proposal_gradient, query_gradient = torch.autograd.grad(
        loss,
        (proposals, query_rows),
        retain_graph=retain_graph,
        allow_unused=True,
    )
    return (
        proposal_gradient if proposal_gradient is not None else torch.zeros_like(proposals),
        query_gradient if query_gradient is not None else torch.zeros_like(query_rows),
    )


def _gradient_attribution(
    model: nn.Module,
    objective: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    precision: Any,
    max_batches: int,
) -> tuple[dict[str, Any], list[str]]:
    proposal_gradients: dict[str, list[Tensor]] = defaultdict(list)
    query_gradients: dict[str, list[Tensor]] = defaultdict(list)
    processed_ids: list[str] = []
    for batch_index, cpu_batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        processed_ids.extend(cpu_batch.sample_ids)
        batch = cpu_batch.to(device)
        if batch.target_pixels is None:
            raise ValueError("gradient diagnostic requires target images")
        with torch.autocast(
            device_type=device.type,
            enabled=precision.autocast_enabled,
            dtype=precision.autocast_dtype,
        ):
            output = model(
                batch.reference_pixels,
                batch.input_ids,
                batch.attention_mask,
                batch.content_mask,
            )
            with torch.no_grad():
                targets = model.encode_global_images(batch.target_pixels)
            components = objective(
                output,
                targets.detach(),
                batch.target_ids,
                batch.modification_texts,
            )
        if not output["steps"]:
            continue
        proposals = output["steps"][0]["proposals"]
        query_rows = model.proposal.queries
        names = list(LOSS_COMPONENTS)
        for index, name in enumerate(names):
            coefficient = float(getattr(objective.config, LOSS_COEFFICIENTS[name]))
            proposal_gradient, query_gradient = _gradient_for(
                coefficient * components[LOSS_COMPONENTS[name]],
                proposals,
                query_rows,
                retain_graph=index < len(names) - 1,
            )
            proposal_gradients[name].append(proposal_gradient.detach().float().cpu())
            query_gradients[name].append(query_gradient.detach().float().cpu()[None])

    if not processed_ids:
        return {"available": False, "reason": "no gradient batches were processed"}, []
    proposal_values = {name: torch.cat(values) for name, values in proposal_gradients.items()}
    query_values = {name: torch.cat(values) for name, values in query_gradients.items()}
    attribution = {
        "available": True,
        "t0_candidate_tensor_gradient": {
            name: summarize_slot_gradients(values)
            for name, values in proposal_values.items()
        },
        "parameter_level_aggregate_query_row_gradient": {
            name: summarize_slot_gradients(values) for name, values in query_values.items()
        },
        "t0_proposal_gradient_interference": gradient_interference_summary(
            proposal_values
        ),
        "score_loss_upstream_detach_control": {
            name: {
                "maximum_absolute_gradient": float(proposal_values[name].abs().max()),
                "passed": bool(float(proposal_values[name].abs().max()) <= 1e-12),
            }
            for name in ("pair", "gain")
        },
        "scope_notes": {
            "t0_candidate_tensor_gradient": (
                "gradient of each exact full objective component with respect to the live "
                "t=0 proposal tensor"
            ),
            "parameter_level_aggregate_query_row_gradient": (
                "gradient with respect to shared proposal query rows; terminal/bind/DPP may "
                "aggregate recurrent uses and must not be called pure t=0 exposure"
            ),
            "loss_scaling": (
                "each gradient uses the checkpoint objective coefficient, so norm ratios "
                "describe the actual configured objective contributions"
            ),
        },
        "objective_component_coefficients": {
            name: float(getattr(objective.config, field))
            for name, field in LOSS_COEFFICIENTS.items()
        },
    }
    return attribution, processed_ids


def _automatic_flags(
    functional: Mapping[str, Any],
    gradients: Mapping[str, Any],
    mil: Mapping[str, Any],
    semantic: Mapping[str, Any],
    *,
    complementarity_epsilon: float,
    gradient_skew_threshold: float,
    mil_skew_threshold: float,
) -> list[dict[str, str]]:
    flags: list[dict[str, str]] = []
    coalition = functional["coalition_oracle"]
    if coalition["all_minus_c3"] is not None:
        if coalition["all_minus_c3"] > complementarity_epsilon:
            flags.append(
                {
                    "code": "NON_C3_COMPLEMENTARITY_PRESENT",
                    "message": "All-K oracle exceeds C3-only beyond the configured tolerance.",
                }
            )
        else:
            flags.append(
                {
                    "code": "NON_C3_COMPLEMENTARITY_WEAK",
                    "message": "All-K oracle is close to C3-only under the configured tolerance.",
                }
            )
        shares = functional["shapley"]["normalized_share_per_slot"]
        if shares is not None and shares[3] >= 0.70:
            flags.append(
                {
                    "code": "C3_DOMINANT_FUNCTIONAL_CONTRIBUTION",
                    "message": "C3 owns at least 70% of mean exact Shapley contribution.",
                }
            )

    if gradients.get("available"):
        terminal = gradients["t0_candidate_tensor_gradient"]["terminal"]["per_slot"]
        if max(row["gradient_energy_fraction"] for row in terminal) >= gradient_skew_threshold:
            flags.append(
                {
                    "code": "TASK_GRADIENT_EXPOSURE_SKEW",
                    "message": "Current t=0 terminal-gradient energy is strongly slot-skewed.",
                }
            )
        for comparison in gradients["t0_proposal_gradient_interference"].values():
            for row in comparison:
                ratio = row["mean_aux_to_terminal_norm_ratio"]
                negative = row["negative_cosine_fraction"]
                if ratio is not None and ratio >= 2.0:
                    flags.append(
                        {
                            "code": "AUX_GRADIENT_DOMINATES_TASK",
                            "message": "An auxiliary gradient norm is at least 2x terminal for a slot.",
                        }
                    )
                    break
                if negative is not None and negative >= 0.50:
                    flags.append(
                        {
                            "code": "NEGATIVE_TASK_AUX_ALIGNMENT",
                            "message": "At least half of valid task/auxiliary gradient cosines are negative.",
                        }
                    )
                    break

    if mil.get("available") and max(mil["mean_responsibility_per_slot"]) >= mil_skew_threshold:
        flags.append(
            {
                "code": "CONCEPT_MIL_RESPONSIBILITY_SKEW",
                "message": "Mean positive-concept MIL responsibility is strongly slot-skewed.",
            }
        )
    if not semantic["reported_concepts"]:
        flags.append(
            {
                "code": "INSUFFICIENT_SEMANTIC_SUPPORT",
                "message": "No parsed concept passed the configured support threshold.",
            }
        )
    # Several slots can cross one threshold; report each evidence type once.
    return list({flag["code"]: flag for flag in flags}.values())


def _render_markdown(report: Mapping[str, Any]) -> str:
    functional = report["functional_specialization"]
    coalition = functional["coalition_oracle"]
    lines = [
        "# IAG-SRME Slot Specialization Diagnostic",
        "",
        "> Read-only checkpoint evidence. Gradient exposure and auxiliary alignment at a final "
        "checkpoint do not establish temporal causality.",
        "",
        "## Cohort",
        "",
        f"- split: `{report['cohort']['split']}`",
        f"- processed samples: `{report['cohort']['processed_sample_count']}`",
        f"- valid t0 teacher rows: `{functional['sample_count']}`",
        f"- sample fingerprint: `{report['cohort']['processed_sample_ids_sha256']}`",
        "",
        "## Functional specialization at t=0",
        "",
        "| slot | selected/execution | mean utility | positive | harmful | oracle/execution | unique wins | Shapley |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in functional["per_slot"]:
        occupancy = row["oracle_occupancy_given_oracle_execute"]
        selected = functional["live_policy"]["selected_fraction_given_execute"][row["slot"]]
        selected_text = f"{selected:.3%}" if selected is not None else "n/a"
        lines.append(
            f"| C{row['slot']} | {selected_text} | {row['mean_utility']:.6f} | "
            f"{row['positive_utility_fraction']:.3%} | {row['harmful_utility_fraction']:.3%} | "
            f"{occupancy:.3%} | {row['unique_oracle_win_rate']:.3%} | "
            f"{row['shapley_contribution']:.6f} |"
            if occupancy is not None
            else f"| C{row['slot']} | {selected_text} | {row['mean_utility']:.6f} | "
            f"{row['positive_utility_fraction']:.3%} | {row['harmful_utility_fraction']:.3%} | "
            f"n/a | {row['unique_oracle_win_rate']:.3%} | "
            f"{row['shapley_contribution']:.6f} |"
        )
    lines += [
        "",
        "## Coalition oracle",
        "",
        f"- all-K value: `{coalition['full_set_value']:.6f}`",
        f"- C3-only value: `{coalition['c3_only_value']}`",
        f"- all-K minus C3: `{coalition['all_minus_c3']}`",
        f"- effective functional K: `{functional['effective_functional_k']}`",
        f"- Shapley efficiency error: `{functional['shapley']['efficiency_absolute_error']:.3e}`",
        "",
        "## Conditional proposal geometry",
        "",
        "PR by slot and functional subset (`n/a` means insufficient support).",
        "",
        "| slot | all | utility > 0 | Shapley > 0 | oracle winner |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in report["conditional_geometry"]["proposal"]["per_slot"]:
        values = []
        for name in (
            "all_valid_t0",
            "positive_utility",
            "positive_shapley",
            "oracle_winner",
        ):
            subset = row["subsets"][name]
            values.append(
                "n/a"
                if subset["insufficient_support"]
                else f"{subset['participation_ratio']:.3f}"
            )
        lines.append(
            f"| C{row['slot']} | {values[0]} | {values[1]} | {values[2]} | {values[3]} |"
        )

    gradients = report["gradient_attribution"]
    lines += ["", "## Current gradient exposure", ""]
    if gradients.get("available"):
        lines += [
            "t0 proposal-tensor gradient energy fraction:",
            "",
            "| loss | "
            + " | ".join(f"C{slot}" for slot in range(functional["num_candidates"]))
            + " |",
            "|---|" + "---:|" * functional["num_candidates"],
        ]
        for name, statistics in gradients["t0_candidate_tensor_gradient"].items():
            energy = [row["gradient_energy_fraction"] for row in statistics["per_slot"]]
            lines.append(
                "| " + name + " | " + " | ".join(f"{value:.3%}" for value in energy) + " |"
            )
        lines += ["", "Pair/gain upstream-detach controls:", ""]
        for name, control in gradients["score_loss_upstream_detach_control"].items():
            lines.append(
                f"- `{name}`: passed=`{control['passed']}`, "
                f"max |grad|=`{control['maximum_absolute_gradient']:.3e}`"
            )
    else:
        lines.append(f"Unavailable: {gradients.get('reason')}")

    mil = report["concept_mil_responsibility"]
    lines += ["", "## Concept-MIL responsibility", ""]
    if mil.get("available"):
        lines.append(
            "- mean responsibility per slot: `"
            + ", ".join(f"{value:.4f}" for value in mil["mean_responsibility_per_slot"])
            + "`"
        )
        lines.append(
            "- top-responsibility occupancy: `"
            + ", ".join(
                f"{value:.3%}" for value in mil["top_responsibility_slot_occupancy"]
            )
            + "`"
        )
        lines.append(
            f"- mean responsibility entropy: `{mil['mean_responsibility_entropy']:.4f}`"
        )
    else:
        lines.append(f"Unavailable: {mil.get('reason')}")

    semantic = report["semantic_specialization"]
    lines += [
        "",
        "## Semantic conditioning",
        "",
        f"- concepts meeting support threshold: `{len(semantic['reported_concepts'])}`",
        f"- concepts below threshold: `{len(semantic['insufficient_support'])}`",
        "- Multi-label conditionals are explanatory, not causal semantic expertise.",
        "",
        "## Automatic descriptive flags",
        "",
    ]
    lines.extend(
        f"- **{flag['code']}** — {flag['message']}"
        for flag in report["automatic_interpretation_flags"]
    )
    lines += [
        "",
        "## Interpretation guardrails",
        "",
        "- High C3 occupancy is not itself collapse.",
        "- Low proposal PR is not itself a dead-slot diagnosis.",
        "- Shapley and all-K-minus-C3 quantify functional complementarity.",
        "- Gradient attribution measures current exposure, not what caused training history.",
        "- Concept-MIL responsibility measures current semantic routing pressure, not causality.",
        "",
        "## Limitations",
        "",
    ]
    lines.extend(f"- {item}" for item in report["limitations"])
    return "\n".join(lines)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    checkpoint_value = cfg.get("checkpoint")
    if checkpoint_value is None:
        raise ValueError("pass +checkpoint=path/to/checkpoint.pt")
    manifest_value = cfg.get("diagnostic_manifest")
    if manifest_value is None:
        raise ValueError("pass +diagnostic_manifest=path/to/persistent_manifest.json")

    checkpoint_path = Path(str(checkpoint_value))
    split = str(cfg.get("diagnostic_split", "val"))
    diagnostic_samples_value = cfg.get("diagnostic_samples", "all")
    batch_size = int(cfg.get("diagnostic_batch_size", 8))
    gradient_batches = int(cfg.get("gradient_batches", 4))
    gradient_batch_size = int(cfg.get("gradient_batch_size", batch_size))
    min_concept_support = int(cfg.get("min_concept_support", 30))
    min_geometry_support = int(cfg.get("min_geometry_support", 20))
    bootstrap_samples = int(cfg.get("bootstrap_samples", 1000))
    seed = int(cfg.seed)

    seed_everything(seed, bool(cfg.runtime.deterministic))
    configure_torch_runtime(
        deterministic=bool(cfg.runtime.deterministic),
        benchmark=bool(cfg.runtime.benchmark),
    )
    device = resolve_device(str(cfg.runtime.device), int(cfg.runtime.accelerator_index))
    precision = resolve_precision(str(cfg.runtime.precision), device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "objective" not in checkpoint:
        raise ValueError(
            "slot specialization requires checkpoint objective state for exact "
            "concept/bind/DPP gradient attribution"
        )
    mismatches = _validate_checkpoint(cfg, checkpoint)
    model, tokenizer, processor = build_model(cfg)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    dataset_root = Path(cfg.dataset.root)
    annotation_root = dataset_root / str(cfg.dataset.annotation_dir)
    image_store = DirectoryImageStore(dataset_root / str(cfg.dataset.image_dir))
    train_dataset = FashionIQDataset(
        annotation_root,
        "train",
        CATEGORIES,
        caption_policy=str(cfg.experiment.train_caption_policy),
        seed=seed,
    )
    caption_policy = (
        str(cfg.experiment.train_caption_policy)
        if split == "train"
        else str(cfg.experiment.val_caption_policy)
    )
    dataset = FashionIQDataset(
        annotation_root,
        split,
        CATEGORIES,
        caption_policy=caption_policy,
        seed=seed,
    )
    sample_count, sample_request = _resolve_sample_count(
        diagnostic_samples_value, len(dataset)
    )
    objective = build_objective_from_checkpoint(
        cfg, checkpoint, model, tokenizer, train_dataset
    ).to(device).eval()
    stored_objective = checkpoint["objective"]
    if not isinstance(stored_objective, Mapping):
        raise ValueError("checkpoint objective state must be a state-dict mapping")
    expected_objective_keys = set(objective.state_dict())
    stored_objective_keys = set(stored_objective)
    if expected_objective_keys != stored_objective_keys:
        raise ValueError(
            "checkpoint objective keys do not exactly match the reconstructed objective: "
            f"missing={sorted(expected_objective_keys - stored_objective_keys)}, "
            f"unexpected={sorted(stored_objective_keys - expected_objective_keys)}"
        )
    collator = FashionIQImageCollator(
        image_store,
        tokenizer,
        processor,
        int(cfg.backbone.max_text_length),
        include_targets=True,
    )
    cohort, manifest = load_or_create_manifest(
        dataset,
        str(manifest_value),
        sample_count=sample_count,
        batch_size=batch_size,
        seed=seed,
        split=split,
        caption_policy=caption_policy,
    )
    loader = DataLoader(
        cohort,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(cfg.experiment.num_workers),
        pin_memory=True,
        collate_fn=collator,
        drop_last=False,
    )

    model_signature_before = _state_signature(model)
    objective_signature_before = _state_signature(objective)
    utilities: list[Tensor] = []
    proposals: list[Tensor] = []
    actions: list[Tensor] = []
    effects: list[Tensor] = []
    selections: list[Tensor] = []
    valid_sample_ids: list[str] = []
    valid_categories: list[str | None] = []
    valid_texts: list[str] = []
    valid_concepts: list[tuple[str, ...]] = []
    processed_sample_ids: list[str] = []
    positive_responsibilities: list[Tensor] = []
    responsibility_by_concept: dict[str, list[Tensor]] = defaultdict(list)
    invalid_teacher_rows = 0

    with torch.no_grad():
        for batch_index, cpu_batch in enumerate(loader):
            processed_sample_ids.extend(cpu_batch.sample_ids)
            batch = cpu_batch.to(device)
            if batch.target_pixels is None:
                raise ValueError("slot specialization requires target images")
            with torch.autocast(
                device_type=device.type,
                enabled=precision.autocast_enabled,
                dtype=precision.autocast_dtype,
            ):
                output = model(
                    batch.reference_pixels,
                    batch.input_ids,
                    batch.attention_mask,
                    batch.content_mask,
                )
                targets = model.encode_global_images(batch.target_pixels)
            if not output["steps"]:
                raise RuntimeError("checkpoint produced no recurrent t=0 step")
            step = output["steps"][0]
            positive, negative, _ = build_teacher_masks(batch.target_ids, targets.device)
            teacher, valid = marginal_teacher_utilities(
                step["current_query"],
                step["candidate_queries"],
                targets.detach(),
                positive,
                negative,
                objective.config.retrieval_temperature,
            )
            invalid_teacher_rows += int((~valid).sum())
            indices = valid.nonzero(as_tuple=False).flatten()
            if indices.numel():
                utilities.append(teacher.index_select(0, indices).float().cpu())
                proposals.append(step["proposals"].index_select(0, indices).float().cpu())
                actions.append(step["actions"].index_select(0, indices).float().cpu())
                effects.append(step["delta_q"].index_select(0, indices).float().cpu())
                selections.append(step["selected_idx"].index_select(0, indices).long().cpu())
                for index in indices.cpu().tolist():
                    valid_sample_ids.append(cpu_batch.sample_ids[index])
                    valid_categories.append(cpu_batch.categories[index])
                    valid_texts.append(cpu_batch.modification_texts[index])
                    valid_concepts.append(
                        parse_instruction_concepts(cpu_batch.modification_texts[index])
                    )

            if objective.concept is not None:
                concept = objective.concept(step["proposals"], batch.modification_texts)
                responsibility = concept_mil_responsibility(
                    concept["candidate_logits"], objective.config.tau_mil
                )
                labels = concept["labels"]
                vocabulary = objective.concept.vocabulary.concepts
                for row in indices.tolist():
                    concept_indices = labels[row].nonzero(as_tuple=False).flatten().tolist()
                    for concept_index in concept_indices:
                        value = responsibility[row, :, concept_index].float().cpu()
                        positive_responsibilities.append(value[None])
                        responsibility_by_concept[vocabulary[concept_index]].append(value)
            print(
                f"[slot-specialization] functional batch={batch_index + 1}/{len(loader)} "
                f"valid_t0={sum(value.shape[0] for value in utilities)}"
            )

    cohort_metadata = validate_processed_manifest(
        manifest, processed_sample_ids, batch_size=batch_size
    )
    if not utilities:
        raise RuntimeError("no valid false-negative-safe teacher rows at t=0")
    utility = torch.cat(utilities)
    proposal = torch.cat(proposals)
    action = torch.cat(actions)
    effect = torch.cat(effects)
    selected = torch.cat(selections)
    functional = functional_specialization_summary(
        utility,
        dpp_temperature=(objective.config.tau_dpp if objective.config.dpp_enabled else None),
        useful_threshold=(
            objective.config.useful_threshold if objective.config.dpp_enabled else None
        ),
    )
    shapley = functional.pop("per_sample_shapley")
    execute = selected < utility.shape[1]
    execute_count = int(execute.sum())
    functional["live_policy"] = {
        "execute_count": execute_count,
        "execute_fraction": float(execute.float().mean()),
        "stop_count": int((~execute).sum()),
        "stop_fraction": float((~execute).float().mean()),
        "selected_count_per_slot": [
            int((execute & selected.eq(slot)).sum()) for slot in range(utility.shape[1])
        ],
        "selected_fraction_given_execute": [
            (
                float((execute & selected.eq(slot)).sum() / execute_count)
                if execute_count
                else None
            )
            for slot in range(utility.shape[1])
        ],
    }
    semantic = semantic_specialization_summary(
        utility,
        shapley,
        valid_concepts,
        min_support=min_concept_support,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    conditional_geometry = {
        "proposal": conditional_geometry_summary(
            proposal, utility, shapley, min_support=min_geometry_support
        ),
        "action": conditional_geometry_summary(
            action, utility, shapley, min_support=min_geometry_support
        ),
        "delta_q": conditional_geometry_summary(
            effect, utility, shapley, min_support=min_geometry_support
        ),
    }
    mil = _mil_summary(
        positive_responsibilities,
        responsibility_by_concept,
        candidates=utility.shape[1],
        min_support=min_concept_support,
    )

    gradient_count = min(len(cohort), gradient_batches * gradient_batch_size)
    gradient_subset = Subset(cohort, list(range(gradient_count)))
    gradient_loader = DataLoader(
        gradient_subset,
        batch_size=gradient_batch_size,
        shuffle=False,
        num_workers=int(cfg.experiment.num_workers),
        pin_memory=True,
        collate_fn=collator,
        drop_last=False,
    )
    gradient_attribution, gradient_sample_ids = _gradient_attribution(
        model,
        objective,
        gradient_loader,
        device=device,
        precision=precision,
        max_batches=gradient_batches,
    )
    model_signature_after = _state_signature(model)
    objective_signature_after = _state_signature(objective)
    if model_signature_after != model_signature_before:
        raise RuntimeError("read-only diagnostic mutated model state")
    if objective_signature_after != objective_signature_before:
        raise RuntimeError("read-only diagnostic mutated objective state")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("autograd diagnostic unexpectedly populated model .grad buffers")
    if any(parameter.grad is not None for parameter in objective.parameters()):
        raise RuntimeError("autograd diagnostic unexpectedly populated objective .grad buffers")

    flags = _automatic_flags(
        functional,
        gradient_attribution,
        mil,
        semantic,
        complementarity_epsilon=float(cfg.get("complementarity_epsilon", 1e-4)),
        gradient_skew_threshold=float(cfg.get("gradient_skew_threshold", 0.75)),
        mil_skew_threshold=float(cfg.get("mil_skew_threshold", 0.60)),
    )
    metadata = checkpoint.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    report = {
        "metadata": {
            **git_identity(),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "checkpoint_metric": checkpoint.get("metric"),
            "checkpoint_experiment_identity": metadata.get("experiment_identity"),
            "configuration_mismatches": mismatches,
            "seed": seed,
            "dataset": str(cfg.dataset.name),
            "dataset_root": str(cfg.dataset.root),
            "teacher_policy": "canonical_identity_aware_false_negative_safe_in_batch",
            "teacher_view": "canonical_inbatch",
            "retrieval_temperature": objective.config.retrieval_temperature,
            "num_candidates": model.config.num_candidates,
            "max_steps": model.config.max_steps,
            "primary_analysis_timestep": 0,
            "concept_parser_version": PARSER_VERSION,
            "concept_vocabulary_size": (
                len(objective.concept.vocabulary.concepts)
                if objective.concept is not None
                else 0
            ),
            "concept_vocabulary_fingerprint": (
                objective.concept.vocabulary.fingerprint
                if objective.concept is not None
                else None
            ),
            "resolved_config": OmegaConf.to_container(cfg, resolve=True),
            "read_only": True,
            "model_state_signature_before": model_signature_before,
            "model_state_signature_after": model_signature_after,
            "objective_state_signature_before": objective_signature_before,
            "objective_state_signature_after": objective_signature_after,
        },
        "cohort": {
            "manifest_path": str(manifest_value),
            "split": split,
            "caption_policy": caption_policy,
            "diagnostic_samples_request": sample_request,
            **cohort_metadata,
            "valid_t0_teacher_row_count": utility.shape[0],
            "valid_t0_sample_ids_sha256": sample_ids_fingerprint(valid_sample_ids),
            "invalid_t0_teacher_rows": invalid_teacher_rows,
        },
        "teacher": {
            "utility_definition": "retrieval_loss(parent)-retrieval_loss(candidate)",
            "stop_utility": 0.0,
            "target_bank_scope": "matched manifest batch",
        },
        "functional_specialization": functional,
        "coalition_oracle": functional["coalition_oracle"],
        "shapley": functional["shapley"],
        "effective_functional_k": functional["effective_functional_k"],
        "semantic_specialization": semantic,
        "conditional_geometry": conditional_geometry,
        "gradient_attribution": {
            **gradient_attribution,
            "gradient_batches": gradient_batches,
            "gradient_batch_size": gradient_batch_size,
            "gradient_sample_count": len(gradient_sample_ids),
            "gradient_sample_ids": gradient_sample_ids,
            "gradient_sample_ids_sha256": sample_ids_fingerprint(gradient_sample_ids),
            "gradient_teacher_batch_grouping_sha256": teacher_batch_fingerprint(
                gradient_sample_ids, gradient_batch_size
            ),
        },
        "gradient_interference": gradient_attribution.get(
            "t0_proposal_gradient_interference", {}
        ),
        "concept_mil_responsibility": mil,
        "automatic_interpretation_flags": flags,
        "limitations": [
            "Static-checkpoint gradients measure current exposure, not temporal causality.",
            "Semantic conditioning is multi-label and explanatory, not causal expertise.",
            "Canonical utilities are conditioned on the matched in-batch negative pool.",
            "No fixed larger negative-bank robustness view is implemented in this patch.",
            "A 160-row manifest is underpowered for rare-specialist claims; use full VAL or 2000+.",
            "OLD and STRONG must be run separately with the same manifest and settings.",
            "Mutual information, stratified permutation tests, and FDR tests are not emitted; "
            "deterministic bootstrap confidence intervals are the implemented robustness view.",
        ],
    }

    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "slot_specialization_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (output_dir / "slot_specialization_report.md").write_text(
        _render_markdown(report), encoding="utf-8"
    )
    with (output_dir / "slot_specialization_samples.jsonl").open(
        "w", encoding="utf-8"
    ) as file:
        for index, sample_id in enumerate(valid_sample_ids):
            file.write(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "category": valid_categories[index],
                        "modification_text": valid_texts[index],
                        "concepts": valid_concepts[index],
                        "teacher_utility": utility[index].tolist(),
                        "shapley": shapley[index].tolist(),
                        "selected_idx": int(selected[index]),
                    }
                )
                + "\n"
            )
    print(f"[slot-specialization] wrote reports to {output_dir}")


if __name__ == "__main__":
    main()
