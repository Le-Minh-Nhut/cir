"""
Candidate/selector diagnostics for CIR IAG-SRME V2 R0.

Put this file at:
    src/diagnose_candidate_selector.py

Example:
    TOKENIZERS_PARALLELISM=false python src/diagnose_candidate_selector.py \
      backbone=fgclip_base_text_native_cls \
      +checkpoint=outputs/r0_ncls_text_strong_aux/best.pt \
      +diagnostic_batches=20 \
      +diagnostic_batch_size=8 \
      hydra.run.dir=outputs/diagnose_selector_new

Read-only: no optimizer.step(), no checkpoint mutation.
Target information is used only for diagnostics.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import hydra
import torch
import torch.nn.functional as F
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import DataLoader

from data.images import FashionIQImageCollator
from datasets.common import DirectoryImageStore
from datasets.fashioniq import FashionIQDataset
from diagnostics.cohort import (
    caption_change_masks,
    category_shuffle_indices,
    load_or_create_manifest,
    sample_ids_fingerprint,
    validate_processed_manifest,
)
from diagnostics.selection import selection_metrics, slot_monopoly, transition_retrieval
from evaluate import validate_checkpoint_backbone_metadata
from evaluation.fashioniq import build_validation_datasets, evaluate_fashioniq
from losses.objective import IAGSRMEObjective, ObjectiveConfig
from models.iag_srme.utils.retrieval import (
    build_teacher_masks,
    teacher_retrieval_loss,
)
from runtime import configure_torch_runtime, resolve_device, seed_everything
from train import (
    CATEGORIES,
    build_concept_vocabulary,
    build_model,
    encode_concept_prototypes,
)
from training.engine import resolve_precision


def safe_div(a: float, b: float) -> float:
    return a / b if b else float("nan")


def matrix_to_list(x: Tensor) -> list[list[float]]:
    return [[float(v) for v in row] for row in x.detach().cpu().tolist()]


def pairwise_cosine_matrix(x: Tensor) -> Tensor:
    z = F.normalize(x.detach().float(), dim=-1)
    return (z @ z.transpose(-1, -2)).mean(dim=0)


def pairwise_jaccard_matrix(mask: Tensor, threshold: float = 0.5) -> Tensor:
    hard = mask.detach() > threshold
    k = hard.shape[1]
    out = torch.zeros(k, k, dtype=torch.float32, device=mask.device)
    for i in range(k):
        for j in range(k):
            inter = (hard[:, i] & hard[:, j]).sum(dim=-1).float()
            union = (hard[:, i] | hard[:, j]).sum(dim=-1).float()
            out[i, j] = (inter / union.clamp_min(1)).mean()
    return out


def pairwise_l2_matrix(x: Tensor) -> Tensor:
    flat = x.detach().float().flatten(2)
    k = flat.shape[1]
    out = torch.zeros(k, k, dtype=torch.float32, device=x.device)
    for i in range(k):
        for j in range(k):
            out[i, j] = (flat[:, i] - flat[:, j]).norm(dim=-1).mean()
    return out


def build_objective_from_checkpoint(
    cfg: DictConfig,
    checkpoint: Mapping[str, Any],
    model: Any,
    tokenizer: Any,
    train_dataset: FashionIQDataset,
) -> IAGSRMEObjective:
    metadata = checkpoint.get("metadata")
    stored = metadata.get("objective_config") if isinstance(metadata, dict) else None
    if isinstance(stored, Mapping):
        objective_config = ObjectiveConfig(**dict(stored))
        print("[diagnostic] objective source: checkpoint")
    else:
        objective_config = ObjectiveConfig(
            **{k: v for k, v in cfg.objective.items() if k != "name"}
        )
        print("[diagnostic] objective source: current config")

    vocabulary = None
    prototypes = None
    if objective_config.concept_enabled:
        vocabulary = build_concept_vocabulary(train_dataset, objective_config)
        prototypes = encode_concept_prototypes(
            model,
            tokenizer,
            vocabulary,
            int(cfg.backbone.max_text_length),
        )

    objective = IAGSRMEObjective(
        objective_config,
        width=int(cfg.model.width),
        state_dim=model.backbone.state_dim,
        concept_vocabulary=vocabulary,
        concept_prototypes=prototypes,
    )

    if "objective" in checkpoint:
        missing, unexpected = objective.load_state_dict(checkpoint["objective"], strict=False)
        if missing or unexpected:
            print(
                "[diagnostic] objective state mismatch:",
                {"missing": missing, "unexpected": unexpected},
            )

    return objective


def proposal_query_prior_cosine(model: Any) -> list[list[float]] | None:
    queries = getattr(getattr(model, "proposal", None), "queries", None)
    if not isinstance(queries, Tensor):
        return None
    q = F.normalize(queries.detach().float(), dim=-1)
    return matrix_to_list(q @ q.T)


def format_matrix(matrix: list[list[float]], labels: list[str]) -> str:
    lines = [
        "| | " + " | ".join(labels) + " |",
        "|---|" + "|".join(["---:"] * len(labels)) + "|",
    ]
    for label, row in zip(labels, matrix):
        lines.append("| " + label + " | " + " | ".join(f"{v:.4f}" for v in row) + " |")
    return "\n".join(lines)


def render_markdown(report: Mapping[str, Any]) -> str:
    k = int(report["num_candidates"])
    cand_labels = [f"C{i}" for i in range(k)]
    decision_labels = cand_labels + ["STOP"]

    lines = [
        "# Candidate / Selector Diagnostic",
        "",
        f"- checkpoint: `{report['checkpoint_path']}`",
        f"- epoch: `{report['checkpoint'].get('epoch')}`",
        f"- metric: `{report['checkpoint'].get('metric')}`",
        "",
        "## Automatic flags",
        "",
    ]
    for flag in report["flags"]:
        lines.append(f"- **{flag['level']} — {flag['code']}**: {flag['message']}")

    lines += [
        "",
        "## Slot statistics",
        "",
        "| Slot | selected/all | selected/execute | oracle/all | oracle/oracle-execute | mean score | mean teacher utility | positive utility % | harmful when selected % | useful-for-DPP % |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["slot_stats"]:
        lines.append(
            "| {slot} | {selected_rate:.3%} | {selected_rate_given_execute:.3%} | "
            "{oracle_rate:.3%} | {oracle_rate_given_oracle_execute:.3%} | {mean_score:.5f} | "
            "{mean_teacher_utility:.5f} | {positive_utility_rate:.3%} | "
            "{harmful_selected_rate:.3%} | {dpp_useful_rate:.3%} |".format(**row)
        )

    sel = report["selector"]
    detailed = report["selection_metrics"]
    stop = detailed["stop"]
    lines += [
        "",
        "## Selector summary",
        "",
        f"- exact oracle accuracy: `{sel['exact_oracle_accuracy']:.4f}`",
        f"- stop/execute accuracy: `{sel['stop_execute_accuracy']:.4f}`",
        f"- execute rate: `{detailed['execute_rate']:.4f}`",
        f"- harmful / executions: `{sel['harmful_execution_fraction_of_executions']:.4f}`",
        f"- harmful / all decisions: `{sel['harmful_execution_fraction_of_decisions']:.4f}`",
        f"- missed-opportunity STOP rate: `{sel['missed_opportunity_stop_rate']:.4f}`",
        f"- selected teacher utility: `{sel['selected_teacher_utility']:.5f}`",
        f"- oracle teacher utility: `{sel['oracle_teacher_utility']:.5f}`",
        f"- oracle regret: `{sel['oracle_regret']:.5f}`",
        f"- ScoreNet/teacher Pearson: `{sel['score_teacher_pearson']:.4f}`",
        f"- selected/oracle agreement: `{detailed['selected_equals_oracle_fraction']:.4f}`",
        f"- selected utility: `{detailed['mean_selected_utility']:.5f}`",
        f"- oracle utility: `{detailed['mean_oracle_utility']:.5f}`",
        f"- regret: `{detailed['mean_regret']:.5f}`",
        (
            f"- STOP precision / recall / F1: `{stop['precision']:.4f}` / "
            f"`{stop['recall']:.4f}` / `{stop['f1']:.4f}`"
        ),
        (
            f"- harmful executions: `{stop['harmful_execution_count']}` / "
            f"`{stop['executed_action_count']}` = "
            f"`{stop['harmful_execution_fraction_of_executions']:.4f}`"
        ),
        "",
        "### Selector vs oracle confusion",
        "",
        "Rows = selector, columns = oracle.",
        "",
        format_matrix(report["selector_vs_oracle_confusion_rate"], decision_labels),
        "",
        "## Proposal cosine matrix",
        "",
        format_matrix(report["pairwise"]["proposal_cosine"], cand_labels),
        "",
        "## Action cosine matrix",
        "",
        format_matrix(report["pairwise"]["action_cosine"], cand_labels),
        "",
        "## delta_q cosine matrix",
        "",
        format_matrix(report["pairwise"]["delta_q_cosine"], cand_labels),
        "",
        "## Write-mask Jaccard matrix",
        "",
        format_matrix(report["pairwise"]["exec_mask_jaccard"], cand_labels),
        "",
        "## Candidate-state L2 matrix",
        "",
        format_matrix(report["pairwise"]["candidate_state_l2"], cand_labels),
    ]

    if report.get("proposal_query_prior_cosine") is not None:
        lines += [
            "",
            "## Learned proposal-query prior cosine",
            "",
            format_matrix(report["proposal_query_prior_cosine"], cand_labels),
        ]

    dpp = report["dpp"]
    lines += [
        "",
        "## DPP / candidate quality",
        "",
        f"- mean useful candidate count: `{dpp['mean_useful_candidate_count']:.4f}`",
        f"- rows with >=2 useful candidates: `{dpp['valid_row_rate']:.4f}`",
        f"- positive candidate fraction: `{dpp['positive_candidate_fraction']:.4f}`",
        "",
        "> Target-derived teacher utility is diagnostic only and is never an inference input.",
    ]
    if report.get("caption_sensitivity"):
        lines += ["", "## Correct vs shuffled-caption sensitivity", ""]
        for name, values in report["caption_sensitivity"].items():
            lines.append(f"- `{name}`: mean `{values['mean']:.6f}` (n={values['count']})")
    if report.get("official_fashioniq"):
        lines += ["", "## Official FashionIQ retrieval", ""]
        for name, value in sorted(report["official_fashioniq"].items()):
            lines.append(f"- `{name}`: `{value:.6f}`")
    return "\n".join(lines)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    checkpoint_value = cfg.get("checkpoint")
    if checkpoint_value is None:
        raise ValueError("pass +checkpoint=path/to/best.pt")

    checkpoint_path = Path(str(checkpoint_value))
    diagnostic_batches = int(cfg.get("diagnostic_batches", 20))
    diagnostic_batch_size = int(cfg.get("diagnostic_batch_size", 8))
    manifest_value = cfg.get("diagnostic_manifest")
    if manifest_value is None:
        raise ValueError(
            "pass +diagnostic_manifest=path/to/shared_manifest.json; "
            "OLD and STRONG must replay the same persistent cohort"
        )
    diagnostic_split = str(cfg.get("diagnostic_split", "train"))
    run_official_evaluation = bool(cfg.get("diagnostic_official_eval", False))

    seed_everything(int(cfg.seed), bool(cfg.runtime.deterministic))
    configure_torch_runtime(
        deterministic=bool(cfg.runtime.deterministic),
        benchmark=bool(cfg.runtime.benchmark),
    )
    device = resolve_device(str(cfg.runtime.device), int(cfg.runtime.accelerator_index))
    precision = resolve_precision(str(cfg.runtime.precision), device)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    validate_checkpoint_backbone_metadata(
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
    model, tokenizer, processor = build_model(cfg)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    dataset_root = Path(cfg.dataset.root)
    annotation_root = dataset_root / str(cfg.dataset.annotation_dir)
    image_store = DirectoryImageStore(dataset_root / str(cfg.dataset.image_dir))

    caption_policy = (
        str(cfg.experiment.train_caption_policy)
        if diagnostic_split == "train"
        else str(cfg.experiment.val_caption_policy)
    )
    concept_dataset = FashionIQDataset(
        annotation_root,
        "train",
        CATEGORIES,
        caption_policy=str(cfg.experiment.train_caption_policy),
        seed=int(cfg.seed),
    )
    diagnostic_dataset = FashionIQDataset(
        annotation_root,
        diagnostic_split,
        CATEGORIES,
        caption_policy=caption_policy,
        seed=int(cfg.seed),
    )
    objective = (
        build_objective_from_checkpoint(cfg, checkpoint, model, tokenizer, concept_dataset)
        .to(device)
        .eval()
    )

    collator = FashionIQImageCollator(
        image_store,
        tokenizer,
        processor,
        int(cfg.backbone.max_text_length),
        include_targets=True,
    )
    cohort, manifest = load_or_create_manifest(
        diagnostic_dataset,
        str(manifest_value),
        sample_count=diagnostic_batches * diagnostic_batch_size,
        batch_size=diagnostic_batch_size,
        seed=int(cfg.seed),
        split=diagnostic_split,
        caption_policy=caption_policy,
    )
    loader = DataLoader(
        cohort,
        batch_size=diagnostic_batch_size,
        shuffle=False,
        num_workers=int(cfg.experiment.num_workers),
        pin_memory=True,
        collate_fn=collator,
        drop_last=False,
    )

    k = int(model.config.num_candidates)
    stop_idx = k
    n_actions = k + 1

    selected_count = torch.zeros(n_actions, dtype=torch.float64)
    oracle_count = torch.zeros(n_actions, dtype=torch.float64)
    confusion = torch.zeros(n_actions, n_actions, dtype=torch.float64)

    score_sum = torch.zeros(k, dtype=torch.float64)
    teacher_sum = torch.zeros(k, dtype=torch.float64)
    teacher_positive = torch.zeros(k, dtype=torch.float64)
    dpp_useful = torch.zeros(k, dtype=torch.float64)
    slot_rows = torch.zeros(k, dtype=torch.float64)
    selected_harmful = torch.zeros(k, dtype=torch.float64)
    selected_slot_count = torch.zeros(k, dtype=torch.float64)

    proposal_cos_sum = torch.zeros(k, k, dtype=torch.float64)
    action_cos_sum = torch.zeros(k, k, dtype=torch.float64)
    deltaq_cos_sum = torch.zeros(k, k, dtype=torch.float64)
    mask_jaccard_sum = torch.zeros(k, k, dtype=torch.float64)
    state_l2_sum = torch.zeros(k, k, dtype=torch.float64)
    pairwise_steps = 0

    all_scores: list[Tensor] = []
    all_teacher: list[Tensor] = []
    all_teacher_matrix: list[Tensor] = []
    all_selected: list[Tensor] = []
    all_delta_norm: list[Tensor] = []
    timestep_values: dict[int, dict[str, list[Tensor]]] = defaultdict(lambda: defaultdict(list))
    transition_values: dict[str, list[Tensor]] = defaultdict(list)
    decision_records: list[dict[str, Any]] = []
    terminal_records: list[dict[str, Any]] = []
    caption_values: dict[str, list[float]] = defaultdict(list)
    caption_shuffle_counts = {
        "requested_shuffle_rows": 0,
        "index_changed_rows": 0,
        "text_changed_rows": 0,
        "text_unchanged_due_to_duplicate_caption_rows": 0,
    }
    processed_sample_ids: list[str] = []
    live_sample_ids_by_t: dict[int, list[str]] = defaultdict(list)

    total_decisions = 0
    teacher_invalid_rows = 0
    total_harmful = 0
    total_executed = 0
    total_missed_stop = 0
    total_stop_execute_correct = 0
    total_exact_oracle = 0
    selected_utility_sum = 0.0
    oracle_utility_sum = 0.0
    regret_sum = 0.0

    useful_count_sum = 0.0
    useful_row_count = 0
    dpp_valid_rows = 0
    positive_candidate_total = 0.0
    positive_candidate_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            processed_sample_ids.extend(batch.sample_ids)

            batch = batch.to(device)
            assert batch.target_pixels is not None

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

                shuffle = category_shuffle_indices(
                    batch.categories, seed=int(cfg.seed) + batch_idx
                ).to(device)
                shuffled_output = model(
                    batch.reference_pixels,
                    batch.input_ids.index_select(0, shuffle),
                    batch.attention_mask.index_select(0, shuffle),
                    batch.content_mask.index_select(0, shuffle),
                )

            positive, negative, _ = build_teacher_masks(batch.target_ids, targets.device)
            initial_query = (
                output["steps"][0]["current_query"] if output["steps"] else output["query"]
            )
            initial_loss = teacher_retrieval_loss(
                initial_query,
                targets.detach(),
                positive,
                negative,
                objective.config.retrieval_temperature,
            )
            terminal = transition_retrieval(
                output["query"],
                output["query"][:, None],
                targets.detach(),
                positive,
                negative,
                objective.config.retrieval_temperature,
            )
            for row, sample_id in enumerate(batch.sample_ids):
                if not bool(terminal["valid"][row]):
                    continue
                terminal_records.append(
                    {
                        "sample_id": sample_id,
                        "category": batch.categories[row],
                        "stopped_early": bool(output["stopped"][row]),
                        "initial_retrieval_loss": float(initial_loss[row].cpu()),
                        "terminal_retrieval_loss": float(terminal["parent_loss"][row].cpu()),
                        "initial_to_terminal_retrieval_improvement": float(
                            (initial_loss[row] - terminal["parent_loss"][row]).cpu()
                        ),
                        "terminal_positive_similarity": float(
                            terminal["parent_positive_similarity"][row].cpu()
                        ),
                        "terminal_hardest_negative_similarity": float(
                            terminal["parent_hardest_negative_similarity"][row].cpu()
                        ),
                        "terminal_positive_minus_hardest_negative_margin": float(
                            terminal["parent_margin"][row].cpu()
                        ),
                    }
                )

            for step in output["steps"]:
                proposal_cos_sum += pairwise_cosine_matrix(step["proposals"]).double().cpu()
                action_cos_sum += pairwise_cosine_matrix(step["actions"]).double().cpu()
                deltaq_cos_sum += pairwise_cosine_matrix(step["delta_q"]).double().cpu()
                mask_jaccard_sum += pairwise_jaccard_matrix(step["exec_mask"]).double().cpu()
                state_l2_sum += pairwise_l2_matrix(step["candidate_states"]).double().cpu()
                pairwise_steps += 1

                live_indices = step["live_indices"]
                live_sample_ids_by_t[int(step["timestep"])].extend(
                    [batch.sample_ids[index] for index in live_indices.cpu().tolist()]
                )
                pos_live = positive.index_select(0, live_indices)
                neg_live = negative.index_select(0, live_indices)
                transition = transition_retrieval(
                    step["current_query"],
                    step["candidate_queries"],
                    targets.detach(),
                    pos_live,
                    neg_live,
                    objective.config.retrieval_temperature,
                )
                teacher = transition["utility"]
                valid_rows = transition["valid"]
                teacher_invalid_rows += int((~valid_rows).sum().cpu())

                if not valid_rows.any():
                    continue

                teacher = teacher[valid_rows].float()
                scores = step["scores"][valid_rows].float()
                selected = step["selected_idx"][valid_rows]

                rows = teacher.shape[0]
                total_decisions += rows
                score_sum += scores.sum(dim=0).double().cpu()
                teacher_sum += teacher.sum(dim=0).double().cpu()
                teacher_positive += (teacher > 0).sum(dim=0).double().cpu()
                slot_rows += rows

                quality = torch.sigmoid(teacher / objective.config.tau_dpp)
                useful = quality > objective.config.useful_threshold
                dpp_useful += useful.sum(dim=0).double().cpu()

                useful_per_row = useful.sum(dim=-1)
                useful_count_sum += float(useful_per_row.float().sum().cpu())
                useful_row_count += rows
                dpp_valid_rows += int((useful_per_row >= 2).sum().cpu())

                positive_candidate_total += float((teacher > 0).float().sum().cpu())
                positive_candidate_count += teacher.numel()

                all_scores.append(scores.detach().cpu().flatten())
                all_teacher.append(teacher.detach().cpu().flatten())
                all_teacher_matrix.append(teacher.detach().cpu())
                all_selected.append(selected.detach().cpu())
                delta_norm = step["delta_q"][valid_rows].detach().float().norm(dim=-1)
                all_delta_norm.append(delta_norm.cpu())
                timestep = int(step["timestep"])
                timestep_values[timestep]["utility"].append(teacher.detach().cpu())
                timestep_values[timestep]["selected"].append(selected.detach().cpu())
                timestep_values[timestep]["delta_norm"].append(delta_norm.cpu())

                for key in (
                    "parent_loss",
                    "candidate_loss",
                    "parent_positive_similarity",
                    "parent_hardest_negative_similarity",
                    "parent_margin",
                    "candidate_positive_similarity",
                    "candidate_hardest_negative_similarity",
                    "candidate_margin",
                ):
                    transition_values[key].append(
                        transition[key][valid_rows].detach().float().cpu()
                    )

                oracle_u, oracle_idx = teacher.max(dim=-1)
                oracle_stop = oracle_u <= float(model.config.epsilon_stop)
                oracle_action = torch.where(
                    oracle_stop,
                    torch.full_like(oracle_idx, stop_idx),
                    oracle_idx,
                )

                for s in selected.detach().cpu().tolist():
                    selected_count[int(s)] += 1
                for o in oracle_action.detach().cpu().tolist():
                    oracle_count[int(o)] += 1
                for s, o in zip(
                    selected.detach().cpu().tolist(),
                    oracle_action.detach().cpu().tolist(),
                ):
                    confusion[int(s), int(o)] += 1

                stop = selected >= k
                gather_idx = selected.clamp_max(k - 1)
                selected_u = teacher.gather(1, gather_idx[:, None]).squeeze(1)
                selected_u = torch.where(stop, torch.zeros_like(selected_u), selected_u)
                oracle_value = torch.where(oracle_stop, torch.zeros_like(oracle_u), oracle_u)
                regret = oracle_value - selected_u

                harmful = (~stop) & (selected_u < 0)
                missed_stop = stop & ~oracle_stop
                stop_execute_correct = stop == oracle_stop
                exact = torch.where(
                    oracle_stop,
                    stop,
                    (~stop) & (gather_idx == oracle_idx),
                )

                total_harmful += int(harmful.sum().cpu())
                total_executed += int((~stop).sum().cpu())
                total_missed_stop += int(missed_stop.sum().cpu())
                total_stop_execute_correct += int(stop_execute_correct.sum().cpu())
                total_exact_oracle += int(exact.sum().cpu())
                selected_utility_sum += float(selected_u.sum().cpu())
                oracle_utility_sum += float(oracle_value.sum().cpu())
                regret_sum += float(regret.sum().cpu())

                for slot in range(k):
                    chosen_slot = (~stop) & (gather_idx == slot)
                    selected_slot_count[slot] += float(chosen_slot.sum().cpu())
                    selected_harmful[slot] += float((chosen_slot & (selected_u < 0)).sum().cpu())

                valid_live = live_indices[valid_rows].detach().cpu().tolist()
                for row, batch_row in enumerate(valid_live):
                    decision_records.append(
                        {
                            "sample_id": batch.sample_ids[batch_row],
                            "category": batch.categories[batch_row],
                            "timestep": timestep,
                            "live": True,
                            "executed": bool(selected[row] < k),
                            "selected_idx": int(selected[row]),
                            "stopped_now": bool(selected[row] >= k),
                            "parent_retrieval_loss": float(
                                transition["parent_loss"][valid_rows][row].detach().cpu()
                            ),
                            "candidate_retrieval_loss": [
                                float(value)
                                for value in transition["candidate_loss"][valid_rows][row]
                                .detach()
                                .cpu()
                            ],
                            "candidate_utility": [
                                float(value) for value in teacher[row].detach().cpu()
                            ],
                            "parent_positive_similarity": float(
                                transition["parent_positive_similarity"][valid_rows][row]
                                .detach()
                                .cpu()
                            ),
                            "parent_hardest_negative_similarity": float(
                                transition["parent_hardest_negative_similarity"][valid_rows][row]
                                .detach()
                                .cpu()
                            ),
                            "parent_positive_minus_hardest_negative_margin": float(
                                transition["parent_margin"][valid_rows][row].detach().cpu()
                            ),
                            "selected_utility": float(selected_u[row].detach().cpu()),
                            "oracle_utility": float(oracle_value[row].detach().cpu()),
                            "selected_vs_oracle_regret": float(regret[row].detach().cpu()),
                            "positive_utility_candidate_count": int(
                                (teacher[row] > 0).sum().detach().cpu()
                            ),
                            "harmful_candidate_count": int((teacher[row] < 0).sum().detach().cpu()),
                        }
                    )

            # Caption sensitivity is paired at t=0, before divergent STOP cohorts.
            if output["steps"] and shuffled_output["steps"]:
                correct = output["steps"][0]
                shuffled = shuffled_output["steps"][0]
                caption_masks = caption_change_masks(batch.modification_texts, shuffle)
                index_changed = caption_masks["index_changed"].to(device)
                changed = caption_masks["text_changed"].to(device)
                duplicate_unchanged = caption_masks["duplicate_caption_unchanged"]
                caption_shuffle_counts["requested_shuffle_rows"] += len(batch.modification_texts)
                caption_shuffle_counts["index_changed_rows"] += int(index_changed.sum())
                caption_shuffle_counts["text_changed_rows"] += int(changed.sum())
                caption_shuffle_counts["text_unchanged_due_to_duplicate_caption_rows"] += int(
                    duplicate_unchanged.sum()
                )
                if changed.any():
                    for name in ("proposals", "actions", "delta_q"):
                        first = correct[name][changed].detach().float()
                        second = shuffled[name][changed].detach().float()
                        caption_values[f"{name}_cosine"].extend(
                            F.cosine_similarity(first, second, dim=-1).flatten().cpu().tolist()
                        )
                        caption_values[f"{name}_norm_difference"].extend(
                            (first - second).norm(dim=-1).flatten().cpu().tolist()
                        )
                    correct_transition = transition_retrieval(
                        correct["current_query"],
                        correct["candidate_queries"],
                        targets.detach(),
                        positive,
                        negative,
                        objective.config.retrieval_temperature,
                    )
                    shuffled_transition = transition_retrieval(
                        shuffled["current_query"],
                        shuffled["candidate_queries"],
                        targets.detach(),
                        positive,
                        negative,
                        objective.config.retrieval_temperature,
                    )
                    paired_valid = (
                        correct_transition["valid"] & shuffled_transition["valid"] & changed
                    )
                    if paired_valid.any():
                        correct_utility = correct_transition["utility"][paired_valid]
                        shuffled_utility = shuffled_transition["utility"][paired_valid]
                        caption_values["candidate_utility_correct_minus_shuffled"].extend(
                            (correct_utility - shuffled_utility).flatten().cpu().tolist()
                        )
                        caption_values["oracle_utility_correct_minus_shuffled"].extend(
                            (correct_utility.max(-1).values - shuffled_utility.max(-1).values)
                            .cpu()
                            .tolist()
                        )
                        selected_caption_values = {}
                        for prefix, step_output, utilities in (
                            ("correct", correct, correct_utility),
                            ("shuffled", shuffled, shuffled_utility),
                        ):
                            selected_caption = step_output["selected_idx"][paired_valid]
                            stopped_caption = selected_caption >= k
                            selected_slot = selected_caption.clamp_max(k - 1)
                            selected_value = utilities.gather(1, selected_slot[:, None]).squeeze(1)
                            selected_value = torch.where(
                                stopped_caption,
                                torch.zeros_like(selected_value),
                                selected_value,
                            )
                            caption_values[f"selected_utility_{prefix}"].extend(
                                selected_value.cpu().tolist()
                            )
                            selected_caption_values[prefix] = selected_value
                        caption_values["selected_utility_correct_minus_shuffled"].extend(
                            (
                                selected_caption_values["correct"]
                                - selected_caption_values["shuffled"]
                            )
                            .cpu()
                            .tolist()
                        )
                    correct_terminal = teacher_retrieval_loss(
                        output["query"],
                        targets.detach(),
                        positive,
                        negative,
                        objective.config.retrieval_temperature,
                    )
                    shuffled_terminal = teacher_retrieval_loss(
                        shuffled_output["query"],
                        targets.detach(),
                        positive,
                        negative,
                        objective.config.retrieval_temperature,
                    )
                    terminal_rows = positive.any(dim=-1) & negative.any(dim=-1) & changed
                    if terminal_rows.any():
                        # Positive means the correct caption has lower retrieval loss.
                        caption_values["terminal_retrieval_correct_advantage"].extend(
                            (shuffled_terminal - correct_terminal)[terminal_rows].cpu().tolist()
                        )

            print(f"[diagnostic] batch={batch_idx + 1}/{len(loader)} decisions={total_decisions}")

    cohort_metadata = validate_processed_manifest(
        manifest, processed_sample_ids, batch_size=diagnostic_batch_size
    )

    if total_decisions == 0:
        raise RuntimeError("no valid decisions found")

    slot_stats = []
    execute_count = int(selected_count[:k].sum())
    oracle_execute_count = int(oracle_count[:k].sum())
    for slot in range(k):
        slot_stats.append(
            {
                "slot": f"C{slot}",
                "selected_rate": safe_div(float(selected_count[slot]), total_decisions),
                "selected_rate_given_execute": safe_div(float(selected_count[slot]), execute_count),
                "oracle_rate": safe_div(float(oracle_count[slot]), total_decisions),
                "oracle_rate_given_oracle_execute": safe_div(
                    float(oracle_count[slot]), oracle_execute_count
                ),
                "mean_score": safe_div(float(score_sum[slot]), float(slot_rows[slot])),
                "mean_teacher_utility": safe_div(float(teacher_sum[slot]), float(slot_rows[slot])),
                "positive_utility_rate": safe_div(
                    float(teacher_positive[slot]), float(slot_rows[slot])
                ),
                "harmful_selected_rate": safe_div(
                    float(selected_harmful[slot]), float(selected_slot_count[slot])
                ),
                "dpp_useful_rate": safe_div(float(dpp_useful[slot]), float(slot_rows[slot])),
            }
        )

    slot_stats.append(
        {
            "slot": "STOP",
            "selected_rate": safe_div(float(selected_count[stop_idx]), total_decisions),
            "selected_rate_given_execute": 0.0,
            "oracle_rate": safe_div(float(oracle_count[stop_idx]), total_decisions),
            "oracle_rate_given_oracle_execute": 0.0,
            "mean_score": 0.0,
            "mean_teacher_utility": 0.0,
            "positive_utility_rate": 0.0,
            "harmful_selected_rate": 0.0,
            "dpp_useful_rate": 0.0,
        }
    )

    score_teacher_pearson = float("nan")
    if all_scores and all_teacher:
        x = torch.cat(all_scores).float()
        y = torch.cat(all_teacher).float()
        x = x - x.mean()
        y = y - y.mean()
        denom = x.norm() * y.norm()
        if float(denom) > 1e-12:
            score_teacher_pearson = float((x @ y / denom).cpu())

    pairwise_steps = max(pairwise_steps, 1)
    proposal_cos = proposal_cos_sum / pairwise_steps
    action_cos = action_cos_sum / pairwise_steps
    deltaq_cos = deltaq_cos_sum / pairwise_steps
    mask_jaccard = mask_jaccard_sum / pairwise_steps
    state_l2 = state_l2_sum / pairwise_steps
    confusion_rate = confusion / max(float(confusion.sum()), 1.0)

    combined_selection = selection_metrics(
        torch.cat(all_teacher_matrix),
        torch.cat(all_selected),
        stop_threshold=float(model.config.epsilon_stop),
        delta_q_norm=torch.cat(all_delta_norm),
    )
    by_timestep = {}
    for timestep, values in sorted(timestep_values.items()):
        by_timestep[str(timestep)] = selection_metrics(
            torch.cat(values["utility"]),
            torch.cat(values["selected"]),
            stop_threshold=float(model.config.epsilon_stop),
            delta_q_norm=torch.cat(values["delta_norm"]),
        )

    flags: list[dict[str, str]] = []
    selected_dominance = slot_monopoly(combined_selection)
    oracle_dominance = slot_monopoly(combined_selection, oracle=True)
    max_selected_slot = int(selected_dominance["slot"])
    max_selected_rate = float(selected_dominance["fraction"])
    max_oracle_slot = int(oracle_dominance["slot"])
    max_oracle_rate = float(oracle_dominance["fraction"])

    if selected_dominance["detected"]:
        flags.append(
            {
                "level": "WARN",
                "code": "SELECTOR_SLOT_COLLAPSE",
                "message": (
                    f"C{max_selected_slot} is selected in {max_selected_rate:.1%} "
                    "of executed edits."
                ),
            }
        )

    if max_selected_slot != max_oracle_slot and max_selected_rate > 0.50:
        flags.append(
            {
                "level": "WARN",
                "code": "SELECTOR_BIAS_NOT_ORACLE_BIAS",
                "message": (
                    f"Selector prefers C{max_selected_slot}, while oracle most often prefers "
                    f"C{max_oracle_slot}."
                ),
            }
        )

    if oracle_dominance["detected"]:
        flags.append(
            {
                "level": "WARN",
                "code": "ORACLE_SLOT_BIAS",
                "message": (
                    f"Oracle itself prefers C{max_oracle_slot} in {max_oracle_rate:.1%} "
                    "of oracle-execute decisions."
                ),
            }
        )

    harmful_rate = safe_div(total_harmful, total_executed)
    if harmful_rate > 0.20:
        flags.append(
            {
                "level": "WARN",
                "code": "HARMFUL_EXECUTIONS",
                "message": f"{harmful_rate:.1%} of executed actions are teacher-negative.",
            }
        )

    exact_acc = safe_div(total_exact_oracle, total_decisions)
    if exact_acc < 0.50:
        flags.append(
            {
                "level": "WARN",
                "code": "LOW_EXACT_ORACLE_ACCURACY",
                "message": f"Exact selector-vs-oracle accuracy is only {exact_acc:.1%}.",
            }
        )

    mean_useful = safe_div(useful_count_sum, useful_row_count)
    valid_dpp_rate = safe_div(dpp_valid_rows, useful_row_count)
    if mean_useful < 2.0:
        flags.append(
            {
                "level": "WARN",
                "code": "DPP_STARVED_FOR_GOOD_CANDIDATES",
                "message": (
                    f"Only {mean_useful:.2f} useful candidates on average; "
                    "diversity exists but candidate quality is insufficient."
                ),
            }
        )

    offdiag = ~torch.eye(k, dtype=torch.bool)
    mean_deltaq_cos = float(deltaq_cos[offdiag].mean())
    if mean_deltaq_cos > 0.95:
        flags.append(
            {
                "level": "FAIL",
                "code": "FUNCTIONAL_COLLAPSE",
                "message": f"Mean sibling delta_q cosine is {mean_deltaq_cos:.4f}.",
            }
        )

    if not flags:
        flags.append(
            {
                "level": "OK",
                "code": "NO_MAJOR_AUTOMATIC_FLAG",
                "message": "No conservative selector/candidate threshold was crossed.",
            }
        )

    metadata = checkpoint.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}

    official_metrics = None
    if run_official_evaluation:
        validation_datasets = build_validation_datasets(
            annotation_root,
            CATEGORIES,
            str(cfg.experiment.val_caption_policy),
            seed=int(cfg.seed),
        )
        validation_collator = FashionIQImageCollator(
            image_store,
            tokenizer,
            processor,
            int(cfg.backbone.max_text_length),
            include_targets=False,
        )
        validation_loaders = {
            category: DataLoader(
                dataset,
                batch_size=int(cfg.experiment.eval_batch_size),
                shuffle=False,
                num_workers=int(cfg.experiment.num_workers),
                collate_fn=validation_collator,
            )
            for category, dataset in validation_datasets.items()
        }
        official_metrics = evaluate_fashioniq(
            model,
            validation_loaders,
            {category: dataset.annotations for category, dataset in validation_datasets.items()},
            protocol=str(cfg.protocol.name),
            split_root=dataset_root / str(cfg.dataset.split_dir),
            split=str(cfg.protocol.split),
            image_store=image_store,
            image_processor=processor,
            device=device,
            gallery_batch_size=int(cfg.experiment.gallery_batch_size),
            num_workers=int(cfg.experiment.num_workers),
        )

    retrieval_behavior = {}
    for name, values in transition_values.items():
        merged = torch.cat(values).float()
        retrieval_behavior[name] = {
            "mean": float(merged.mean()),
            "median": float(merged.median()),
            "count": int(merged.numel()),
        }
        if merged.ndim == 2:
            retrieval_behavior[name]["per_slot"] = [
                {
                    "slot": slot,
                    "mean": float(merged[:, slot].mean()),
                    "median": float(merged[:, slot].median()),
                }
                for slot in range(merged.shape[1])
            ]
    retrieval_behavior["number_positive_utility_candidates"] = {
        "mean": combined_selection["positive_utility_candidate_count_mean"]
    }
    retrieval_behavior["number_harmful_candidates"] = {
        "mean": combined_selection["harmful_candidate_count_mean"]
    }
    merged_delta_norm = torch.cat(all_delta_norm)
    retrieval_behavior["number_nearly_zero_effect_candidates"] = {
        "mean": float((merged_delta_norm <= 1e-6).sum(-1).float().mean()),
        "threshold": 1e-6,
    }
    caption_summary = {
        name: {
            "mean": float(torch.tensor(values).mean()),
            "median": float(torch.tensor(values).median()),
            "count": len(values),
        }
        for name, values in sorted(caption_values.items())
        if values
    }
    terminal_summary = {
        key: float(torch.tensor([float(record[key]) for record in terminal_records]).mean())
        for key in (
            "initial_retrieval_loss",
            "terminal_retrieval_loss",
            "initial_to_terminal_retrieval_improvement",
            "terminal_positive_similarity",
            "terminal_hardest_negative_similarity",
            "terminal_positive_minus_hardest_negative_margin",
        )
    }
    headline = {
        "mean_utility_per_slot": [
            slot["mean_teacher_utility"] for slot in combined_selection["slot_metrics"]
        ],
        "oracle_slot_occupancy": [
            slot["oracle_best_fraction_given_oracle_execute"]
            for slot in combined_selection["slot_metrics"]
        ],
        "selected_slot_occupancy": [
            slot["selected_fraction_given_execute"] for slot in combined_selection["slot_metrics"]
        ],
        "execute_rate": combined_selection["execute_rate"],
        "oracle_execute_rate": combined_selection["oracle_execute_rate"],
        "selected_stop_rate": combined_selection["stop"]["stop_rate"],
        "oracle_stop_rate": combined_selection["stop"]["oracle_stop_rate"],
        "selected_oracle_agreement": combined_selection["selected_equals_oracle_fraction"],
        "selected_utility": combined_selection["mean_selected_utility"],
        "oracle_utility": combined_selection["mean_oracle_utility"],
        "regret": combined_selection["mean_regret"],
        "harmful_execution_fraction_of_executions": combined_selection["stop"][
            "harmful_execution_fraction_of_executions"
        ],
        "harmful_execution_fraction_of_decisions": combined_selection["stop"][
            "harmful_execution_fraction_of_decisions"
        ],
        "stop_precision": combined_selection["stop"]["precision"],
        "stop_recall": combined_selection["stop"]["recall"],
        "caption_correct_advantage": {
            key: value["mean"]
            for key, value in caption_summary.items()
            if "utility" in key or "retrieval" in key
        },
        "official_fashioniq": official_metrics,
        "terminal_cohort": terminal_summary,
    }

    report = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint": {
            "epoch": checkpoint.get("epoch"),
            "metric": checkpoint.get("metric"),
            "experiment_identity": metadata.get("experiment_identity"),
            "global_readout_mode": metadata.get("global_readout_mode"),
            "finetune_policy": metadata.get("finetune_policy"),
            "objective_config": metadata.get("objective_config"),
        },
        "num_candidates": k,
        "diagnostic_batches": diagnostic_batches,
        "diagnostic_batch_size": diagnostic_batch_size,
        "teacher_invalid_rows": teacher_invalid_rows,
        "manifest": {
            "path": str(manifest_value),
            **cohort_metadata,
            "split": diagnostic_split,
            "caption_policy": caption_policy,
        },
        "timestep_cohorts": {
            str(timestep): {
                "live_sample_ids": sample_ids,
                "live_sample_count": len(sample_ids),
                "live_sample_ids_fingerprint": sample_ids_fingerprint(sample_ids),
            }
            for timestep, sample_ids in sorted(live_sample_ids_by_t.items())
        },
        "slot_stats": slot_stats,
        "selector": {
            "exact_oracle_accuracy": exact_acc,
            "stop_execute_accuracy": safe_div(total_stop_execute_correct, total_decisions),
            "harmful_execution_fraction_of_executions": harmful_rate,
            "harmful_execution_fraction_of_decisions": safe_div(total_harmful, total_decisions),
            "executed_action_count": total_executed,
            "missed_opportunity_stop_rate": safe_div(total_missed_stop, total_decisions),
            "selected_teacher_utility": safe_div(selected_utility_sum, total_decisions),
            "oracle_teacher_utility": safe_div(oracle_utility_sum, total_decisions),
            "oracle_regret": safe_div(regret_sum, total_decisions),
            "score_teacher_pearson": score_teacher_pearson,
            "selected_histogram": {str(i): int(selected_count[i]) for i in range(n_actions)},
            "oracle_histogram": {str(i): int(oracle_count[i]) for i in range(n_actions)},
        },
        "selection_metrics": combined_selection,
        "selection_metrics_by_timestep": by_timestep,
        "headline_metrics": headline,
        "retrieval_behavior": retrieval_behavior,
        "retrieval_similarity_definition": {
            "positive_similarity": "maximum cosine over valid positive target IDs",
            "hardest_negative_similarity": "maximum cosine over false-negative-safe negatives",
            "utility": "teacher_loss(parent) - teacher_loss(candidate)",
            "positive_utility": "candidate utility > 0",
            "oracle_execute": "maximum candidate utility > epsilon_stop",
        },
        "metric_definitions": {
            "selected_fraction_of_all_decisions": "selected slot count / all decisions",
            "selected_fraction_given_execute": "selected slot count / executed decisions",
            "oracle_best_fraction_of_all_decisions": "oracle slot count / all decisions",
            "oracle_best_fraction_given_oracle_execute": (
                "oracle slot count / decisions where max utility > epsilon_stop"
            ),
        },
        "decision_records": decision_records,
        "terminal_cohort_records": terminal_records,
        "terminal_cohort_summary": terminal_summary,
        "caption_sensitivity": caption_summary,
        "caption_shuffle_cohort": caption_shuffle_counts,
        "official_fashioniq": official_metrics,
        "selector_vs_oracle_confusion_rate": matrix_to_list(confusion_rate),
        "pairwise": {
            "proposal_cosine": matrix_to_list(proposal_cos),
            "action_cosine": matrix_to_list(action_cos),
            "delta_q_cosine": matrix_to_list(deltaq_cos),
            "exec_mask_jaccard": matrix_to_list(mask_jaccard),
            "candidate_state_l2": matrix_to_list(state_l2),
        },
        "proposal_query_prior_cosine": proposal_query_prior_cosine(model),
        "dpp": {
            "mean_useful_candidate_count": mean_useful,
            "valid_row_rate": valid_dpp_rate,
            "positive_candidate_fraction": safe_div(
                positive_candidate_total, positive_candidate_count
            ),
        },
        "flags": flags,
    }

    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "candidate_selector_diagnostic.json"
    md_path = output_dir / "candidate_selector_diagnostic.md"

    json_path.write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")

    print("\n================ FLAGS ================")
    for flag in flags:
        print(f"[{flag['level']}] {flag['code']}: {flag['message']}")
    print("=======================================")
    print(f"[diagnostic] JSON: {json_path}")
    print(f"[diagnostic] Markdown: {md_path}")


if __name__ == "__main__":
    main()
