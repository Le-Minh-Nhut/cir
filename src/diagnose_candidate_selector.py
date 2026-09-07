from __future__ import annotations

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

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

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
from losses.objective import IAGSRMEObjective, ObjectiveConfig
from models.iag_srme.utils.retrieval import (
    build_teacher_masks,
    marginal_teacher_utilities,
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
        missing, unexpected = objective.load_state_dict(
            checkpoint["objective"], strict=False
        )
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
        lines.append(
            "| " + label + " | " + " | ".join(f"{v:.4f}" for v in row) + " |"
        )
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
        "| Slot | selected % | oracle % | mean score | mean teacher utility | positive utility % | harmful when selected % | useful-for-DPP % |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["slot_stats"]:
        lines.append(
            "| {slot} | {selected_rate:.3%} | {oracle_rate:.3%} | {mean_score:.5f} | "
            "{mean_teacher_utility:.5f} | {positive_utility_rate:.3%} | "
            "{harmful_selected_rate:.3%} | {dpp_useful_rate:.3%} |".format(**row)
        )

    sel = report["selector"]
    lines += [
        "",
        "## Selector summary",
        "",
        f"- exact oracle accuracy: `{sel['exact_oracle_accuracy']:.4f}`",
        f"- stop/execute accuracy: `{sel['stop_execute_accuracy']:.4f}`",
        f"- harmful execution rate: `{sel['harmful_execution_rate']:.4f}`",
        f"- missed-opportunity STOP rate: `{sel['missed_opportunity_stop_rate']:.4f}`",
        f"- selected teacher utility: `{sel['selected_teacher_utility']:.5f}`",
        f"- oracle teacher utility: `{sel['oracle_teacher_utility']:.5f}`",
        f"- oracle regret: `{sel['oracle_regret']:.5f}`",
        f"- ScoreNet/teacher Pearson: `{sel['score_teacher_pearson']:.4f}`",
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
    return "\n".join(lines)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    checkpoint_value = cfg.get("checkpoint")
    if checkpoint_value is None:
        raise ValueError("pass +checkpoint=path/to/best.pt")

    checkpoint_path = Path(str(checkpoint_value))
    diagnostic_batches = int(cfg.get("diagnostic_batches", 20))
    diagnostic_batch_size = int(cfg.get("diagnostic_batch_size", 8))

    seed_everything(int(cfg.seed), bool(cfg.runtime.deterministic))
    configure_torch_runtime(
        deterministic=bool(cfg.runtime.deterministic),
        benchmark=bool(cfg.runtime.benchmark),
    )
    device = resolve_device(str(cfg.runtime.device), int(cfg.runtime.accelerator_index))
    precision = resolve_precision(str(cfg.runtime.precision), device)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
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
        seed=int(cfg.seed),
    )
    objective = build_objective_from_checkpoint(
        cfg, checkpoint, model, tokenizer, train_dataset
    ).to(device).eval()

    collator = FashionIQImageCollator(
        image_store,
        tokenizer,
        processor,
        int(cfg.backbone.max_text_length),
        include_targets=True,
    )
    loader = DataLoader(
        train_dataset,
        batch_size=diagnostic_batch_size,
        shuffle=True,
        num_workers=int(cfg.experiment.num_workers),
        pin_memory=True,
        collate_fn=collator,
        drop_last=True,
        generator=torch.Generator().manual_seed(int(cfg.seed) + 777),
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

    total_decisions = 0
    total_harmful = 0
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

    iterator = iter(loader)
    with torch.no_grad():
        for batch_idx in range(diagnostic_batches):
            try:
                batch = next(iterator)
            except StopIteration:
                break

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

            positive, negative, _ = build_teacher_masks(batch.target_ids, targets.device)

            for step in output["steps"]:
                proposal_cos_sum += pairwise_cosine_matrix(step["proposals"]).double().cpu()
                action_cos_sum += pairwise_cosine_matrix(step["actions"]).double().cpu()
                deltaq_cos_sum += pairwise_cosine_matrix(step["delta_q"]).double().cpu()
                mask_jaccard_sum += pairwise_jaccard_matrix(step["exec_mask"]).double().cpu()
                state_l2_sum += pairwise_l2_matrix(step["candidate_states"]).double().cpu()
                pairwise_steps += 1

                live_indices = step["live_indices"]
                pos_live = positive.index_select(0, live_indices)
                neg_live = negative.index_select(0, live_indices)
                teacher, valid_rows = marginal_teacher_utilities(
                    step["current_query"],
                    step["candidate_queries"],
                    targets.detach(),
                    pos_live,
                    neg_live,
                    objective.config.retrieval_temperature,
                )

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

                oracle_u, oracle_idx = teacher.max(dim=-1)
                oracle_stop = oracle_u <= 0
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
                oracle_value = torch.maximum(oracle_u, torch.zeros_like(oracle_u))
                regret = oracle_value - selected_u

                harmful = (~stop) & (selected_u < 0)
                missed_stop = stop & (oracle_u > 0)
                stop_execute_correct = stop == oracle_stop
                exact = torch.where(
                    oracle_stop,
                    stop,
                    (~stop) & (gather_idx == oracle_idx),
                )

                total_harmful += int(harmful.sum().cpu())
                total_missed_stop += int(missed_stop.sum().cpu())
                total_stop_execute_correct += int(stop_execute_correct.sum().cpu())
                total_exact_oracle += int(exact.sum().cpu())
                selected_utility_sum += float(selected_u.sum().cpu())
                oracle_utility_sum += float(oracle_value.sum().cpu())
                regret_sum += float(regret.sum().cpu())

                for slot in range(k):
                    chosen_slot = (~stop) & (gather_idx == slot)
                    selected_slot_count[slot] += float(chosen_slot.sum().cpu())
                    selected_harmful[slot] += float(
                        (chosen_slot & (selected_u < 0)).sum().cpu()
                    )

            print(
                f"[diagnostic] batch={batch_idx + 1}/{diagnostic_batches} "
                f"decisions={total_decisions}"
            )

    if total_decisions == 0:
        raise RuntimeError("no valid decisions found")

    slot_stats = []
    for slot in range(k):
        slot_stats.append(
            {
                "slot": f"C{slot}",
                "selected_rate": safe_div(float(selected_count[slot]), total_decisions),
                "oracle_rate": safe_div(float(oracle_count[slot]), total_decisions),
                "mean_score": safe_div(float(score_sum[slot]), float(slot_rows[slot])),
                "mean_teacher_utility": safe_div(
                    float(teacher_sum[slot]), float(slot_rows[slot])
                ),
                "positive_utility_rate": safe_div(
                    float(teacher_positive[slot]), float(slot_rows[slot])
                ),
                "harmful_selected_rate": safe_div(
                    float(selected_harmful[slot]), float(selected_slot_count[slot])
                ),
                "dpp_useful_rate": safe_div(
                    float(dpp_useful[slot]), float(slot_rows[slot])
                ),
            }
        )

    slot_stats.append(
        {
            "slot": "STOP",
            "selected_rate": safe_div(float(selected_count[stop_idx]), total_decisions),
            "oracle_rate": safe_div(float(oracle_count[stop_idx]), total_decisions),
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

    flags: list[dict[str, str]] = []
    max_selected_slot = int(selected_count[:k].argmax())
    max_selected_rate = safe_div(float(selected_count[max_selected_slot]), total_decisions)
    max_oracle_slot = int(oracle_count[:k].argmax())
    max_oracle_rate = safe_div(float(oracle_count[max_oracle_slot]), total_decisions)

    if max_selected_rate > 0.60:
        flags.append(
            {
                "level": "WARN",
                "code": "SELECTOR_SLOT_COLLAPSE",
                "message": f"C{max_selected_slot} is selected {max_selected_rate:.1%} of decisions.",
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

    if max_oracle_rate > 0.60:
        flags.append(
            {
                "level": "WARN",
                "code": "ORACLE_SLOT_BIAS",
                "message": f"Oracle itself prefers C{max_oracle_slot} {max_oracle_rate:.1%} of decisions.",
            }
        )

    harmful_rate = safe_div(total_harmful, total_decisions)
    if harmful_rate > 0.20:
        flags.append(
            {
                "level": "WARN",
                "code": "HARMFUL_EXECUTIONS",
                "message": f"{harmful_rate:.1%} of selected executions are teacher-negative.",
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
        "slot_stats": slot_stats,
        "selector": {
            "exact_oracle_accuracy": exact_acc,
            "stop_execute_accuracy": safe_div(total_stop_execute_correct, total_decisions),
            "harmful_execution_rate": harmful_rate,
            "missed_opportunity_stop_rate": safe_div(total_missed_stop, total_decisions),
            "selected_teacher_utility": safe_div(selected_utility_sum, total_decisions),
            "oracle_teacher_utility": safe_div(oracle_utility_sum, total_decisions),
            "oracle_regret": safe_div(regret_sum, total_decisions),
            "score_teacher_pearson": score_teacher_pearson,
            "selected_histogram": {
                str(i): int(selected_count[i]) for i in range(n_actions)
            },
            "oracle_histogram": {
                str(i): int(oracle_count[i]) for i in range(n_actions)
            },
        },
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