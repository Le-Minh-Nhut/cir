from __future__ import annotations

"""
Frozen-checkpoint diagnosis for the A3.2 contextual-key / local-value experiment.

This script does NOT train or mutate the checkpoint. It performs causal
inference-time interventions on the same learned checkpoint to localize why the
hard/local VALUE run may have lost retrieval quality.

Questions answered:
1) Are QASA-selected slots often EMPTY under hard VALUE ownership?
2) Do selected-empty slots still cause executor state changes?
3) Does blocking selected-empty slots improve retrieval?
4) Does hard -> soft VALUE assignment help on the frozen checkpoint?
5) Does raw -> contextual VALUE help on the frozen checkpoint?
6) If neither helps, is the bottleneck likely downstream in slot_mlp/executor?
"""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from audit_taper_merit_p0 import (
    ScalarCollector,
    RecallCollector,
    build_target_indices,
    effective_rank_rows,
    load_runtime,
    mean_pairwise_row_cosine,
    prepare_batch,
    score_gallery,
)
from cache.features import get_features_by_ids
from evaluate_qasa_inference import CATEGORIES
from evaluation.fashioniq import build_fashioniq_gallery


VALUE_MODES = ("soft_shared", "hard_st_exclusive")


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Diagnose A3.2 local/private VALUE failure with frozen-checkpoint "
            "causal interventions. No training and no checkpoint mutation."
        )
    )
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, default=Path("data/FashionIQ"))
    p.add_argument("--cache-root", type=Path, default=Path("features"))
    p.add_argument("--config", type=Path, default=Path("conf/experiment/taper_e2e.yaml"))
    p.add_argument(
        "--slot-value-assignment",
        choices=VALUE_MODES,
        default=None,
        help=(
            "Normally omit this: checkpoint provenance is auto-detected. "
            "Use only for legacy checkpoints without provenance."
        ),
    )
    p.add_argument(
        "--protocol",
        choices=("fashioniq_original", "fashioniq_val"),
        default="fashioniq_original",
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--max-queries-per-category",
        type=int,
        default=256,
        help="0 = full validation set.",
    )
    p.add_argument("--num-examples", type=int, default=30)
    p.add_argument(
        "--json-output",
        type=Path,
        default=Path("reports/taper_local_value_failure_diagnosis.json"),
    )

    # load_runtime() from audit_taper_merit_p0 validates these fields.
    p.add_argument("--hard-negatives", type=int, default=16)
    p.add_argument("--pairwise-margin", type=float, default=0.0)
    p.add_argument("--pairwise-tau", type=float, default=None)
    p.add_argument("--phi-positive-threshold", type=float, default=1e-4)
    p.add_argument("--min-full-gain", type=float, default=1e-4)
    return p.parse_args()


def load_checkpoint_provenance(path: Path) -> dict:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        provenance = obj.get("experiment_provenance")
        if isinstance(provenance, dict):
            return dict(provenance)
    return {}


def safe_ratio(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    num = num.float()
    den = den.float()
    out = torch.full_like(num, float("nan"))
    valid = den > 0
    out[valid] = num[valid] / den[valid]
    return out


def target_rank(scores: torch.Tensor, target_indices: torch.Tensor) -> torch.Tensor:
    target = scores.gather(1, target_indices[:, None])
    return 1 + (scores > target).sum(dim=1)


def target_margin(scores: torch.Tensor, target_indices: torch.Tensor) -> torch.Tensor:
    target = scores.gather(1, target_indices[:, None]).squeeze(1)
    masked = scores.clone()
    masked.scatter_(1, target_indices[:, None], float("-inf"))
    hardest_negative = masked.max(dim=1).values
    return target - hardest_negative


def mean_token_pair_cosine(states: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    b, n, _ = states.shape
    z = F.normalize(states.float(), dim=-1, eps=1e-8)
    sim = z @ z.transpose(1, 2)
    upper = torch.triu(
        torch.ones(n, n, dtype=torch.bool, device=states.device), diagonal=1
    )
    pair_valid = valid[:, :, None] & valid[:, None, :] & upper[None, :, :]
    denom = pair_valid.sum(dim=(1, 2))
    num = (sim * pair_valid.to(sim.dtype)).sum(dim=(1, 2))
    result = num / denom.clamp_min(1)
    return torch.where(
        denom > 0,
        result,
        torch.full_like(result, float("nan")),
    )


def rebuild_edit_slots(model, *, value_states: torch.Tensor, value_masks: torch.Tensor):
    slot_semantics, slot_mass, slot_activity = model._mass_aware_slot_pool(
        value_states, value_masks
    )
    raw_edit_slots = model.slot_mlp(slot_semantics)
    edit_slots = raw_edit_slots * slot_activity.unsqueeze(-1)
    return {
        "slot_semantics": slot_semantics,
        "slot_mass": slot_mass,
        "slot_activity": slot_activity,
        "raw_edit_slots": raw_edit_slots,
        "edit_slots": edit_slots,
    }


def execute_query(
    model,
    *,
    edit_slots: torch.Tensor,
    selected_mask: torch.Tensor,
    z0: torch.Tensor,
    reference_state: torch.Tensor,
):
    execution = model.execute(edit_slots, selected_mask, z0, reference_state)
    query = model.make_query(execution["final_state"])
    return query, execution


def summarize_variant(
    *,
    name: str,
    query: torch.Tensor,
    gallery_norm: torch.Tensor,
    target_indices: torch.Tensor,
    category: str,
    recall: RecallCollector,
    stats: ScalarCollector,
):
    scores = score_gallery(query, gallery_norm)
    recall.update(name, category, scores, target_indices)
    ranks = target_rank(scores, target_indices)
    margins = target_margin(scores, target_indices)
    stats.add(f"retrieval/{name}/target_rank", ranks.float())
    stats.add(f"retrieval/{name}/target_margin", margins)
    return scores, ranks, margins


def get_mean_recall(recall_result: dict, name: str) -> float | None:
    item = recall_result.get(name)
    if not item:
        return None
    return float(item["mean_recall"])


def delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return a - b


def diagnosis_from_report(report: dict) -> list[dict]:
    recall = report.get("recall", {})
    stats = report.get("stats", {})

    def mr(name):
        return get_mean_recall(recall, name)

    def mean_stat(name):
        item = stats.get(name)
        return None if not item else item.get("mean")

    deployed = mr("deployed_qasa")
    nonempty = mr("raw_hard_qasa_nonempty")
    soft = mr("raw_soft_qasa")
    hard_ctx = mr("contextual_hard_qasa_nonempty")
    soft_ctx = mr("contextual_soft_qasa")
    teacher = mr("teacher_full")

    selected_empty = mean_stat("routing/qasa_selected_empty_fraction")
    owned_unselected = mean_stat(
        "routing/hard_owned_token_fraction_unselected_by_qasa"
    )
    hard_k = mean_stat("routing/hard_value_effective_k")
    hard_dom = mean_stat("routing/hard_value_dominant_share")
    empty_shift = mean_stat("executor/empty_selected_actual_change_norm_sum")

    findings = []

    nonempty_gain = delta(nonempty, deployed)
    if selected_empty is not None:
        severity = (
            "high" if selected_empty >= 0.15 else "medium" if selected_empty >= 0.05 else "low"
        )
        findings.append(
            {
                "name": "qasa_vs_hard_value_support_mismatch",
                "severity": severity,
                "evidence": {
                    "mean_qasa_selected_empty_fraction": selected_empty,
                    "mean_recall_gain_when_masking_empty_selected_slots": nonempty_gain,
                    "mean_hard_owned_token_fraction_unselected_by_qasa": owned_unselected,
                },
                "interpretation": (
                    "QASA is computed from soft contextual attention while VALUE may be "
                    "hard-exclusive. A QASA-selected slot can therefore own zero hard VALUE "
                    "tokens, while a hard-nonempty slot can be left unselected."
                ),
            }
        )

    if empty_shift is not None:
        findings.append(
            {
                "name": "empty_slot_executor_is_not_noop_when_selected",
                "severity": "high" if empty_shift > 1e-3 else "low",
                "evidence": {
                    "mean_sum_actual_state_change_norm_empty_selected_only": empty_shift,
                },
                "interpretation": (
                    "An edit slot with zero VALUE can still trigger a nonzero executor "
                    "transition if QASA selects it, because execute() gates on selection, "
                    "not VALUE activity."
                ),
            }
        )

    soft_gain = delta(soft, mr("raw_hard_qasa"))
    if soft_gain is not None:
        severity = "high" if soft_gain >= 2.0 else "medium" if soft_gain >= 0.5 else "low"
        findings.append(
            {
                "name": "hard_assignment_bottleneck_probe",
                "severity": severity,
                "evidence": {
                    "frozen_checkpoint_soft_minus_hard_mean_recall": soft_gain,
                    "raw_hard_qasa_mean_recall": mr("raw_hard_qasa"),
                    "raw_soft_qasa_mean_recall": soft,
                },
                "interpretation": (
                    "This is an inference-only intervention on a checkpoint trained in one "
                    "mode, so it is not a fair final soft-vs-hard comparison. A large positive "
                    "delta nevertheless suggests hard exclusivity itself is a bottleneck."
                ),
            }
        )

    context_gain = delta(hard_ctx, nonempty)
    if context_gain is not None:
        severity = "high" if context_gain >= 2.0 else "medium" if context_gain >= 0.5 else "low"
        findings.append(
            {
                "name": "raw_value_information_bottleneck_probe",
                "severity": severity,
                "evidence": {
                    "contextual_hard_minus_raw_hard_nonempty_mean_recall": context_gain,
                    "raw_hard_qasa_nonempty_mean_recall": nonempty,
                    "contextual_hard_qasa_nonempty_mean_recall": hard_ctx,
                },
                "interpretation": (
                    "A large contextual-V rescue suggests raw mean-pooled word embeddings "
                    "are too weak (e.g. loss of phrase context/order/composition). This is "
                    "a leak probe, not a proposed final architecture."
                ),
            }
        )

    if hard_k is not None or hard_dom is not None:
        collapse = (
            (hard_dom is not None and hard_dom >= 0.80)
            or (hard_k is not None and hard_k <= 1.5)
        )
        findings.append(
            {
                "name": "hard_partition_collapse",
                "severity": "high" if collapse else "medium",
                "evidence": {
                    "mean_hard_value_effective_k": hard_k,
                    "mean_hard_value_dominant_share": hard_dom,
                },
                "interpretation": (
                    "Hard privacy prevents token copying, but it does not itself prevent a "
                    "giant slot from winning most tokens."
                ),
            }
        )

    if deployed is not None and teacher is not None:
        gap = teacher - deployed
        findings.append(
            {
                "name": "global_teacher_upper_bound_gap",
                "severity": "high" if gap >= 5.0 else "medium" if gap >= 2.0 else "low",
                "evidence": {
                    "teacher_full_mean_recall": teacher,
                    "deployed_mean_recall": deployed,
                    "teacher_minus_deployed": gap,
                    "contextual_soft_probe_mean_recall": soft_ctx,
                },
                "interpretation": (
                    "If the teacher/global path is much stronger while VALUE interventions "
                    "do not rescue the model, the remaining bottleneck is likely slot "
                    "construction/executor capacity rather than token assignment alone."
                ),
            }
        )

    return findings


@torch.no_grad()
def run(args):
    provenance = load_checkpoint_provenance(args.checkpoint)
    provenance_mode = provenance.get("slot_value_assignment")
    if args.slot_value_assignment is None and provenance_mode in VALUE_MODES:
        args.slot_value_assignment = provenance_mode

    runtime = load_runtime(args)
    model = runtime["model"]
    device = runtime["device"]

    stats = ScalarCollector()
    recall = RecallCollector()
    examples = []
    total_samples = 0
    max_examples_buffer = max(args.num_examples * 10, args.num_examples)

    for category in CATEGORIES:
        loader = runtime["val_loaders"][category]
        annotations = getattr(loader.dataset, "annotations", None)
        if annotations is None:
            raise AttributeError("FashionIQDataset must expose .annotations")

        gallery_ids = build_fashioniq_gallery(
            protocol=args.protocol,
            split_root=runtime["split_root"],
            split="val",
            category=category,
            annotations=annotations,
        )
        gallery_index = {image_id: i for i, image_id in enumerate(gallery_ids)}
        gallery_features = get_features_by_ids(
            gallery_ids,
            runtime["val_retrieval"],
            runtime["val_retrieval_idx"],
        ).to(device=device, dtype=torch.float32)
        gallery_norm = F.normalize(gallery_features, dim=-1)

        processed = 0
        for batch in tqdm(loader, desc=f"LOCAL-V DIAG [{category}]", dynamic_ncols=True):
            if args.max_queries_per_category and processed >= args.max_queries_per_category:
                break

            x = prepare_batch(runtime, batch)
            b = x["reference_features"].shape[0]
            target_indices = build_target_indices(batch, gallery_index, device)

            slot_output = model.build_edit_slots(
                x["reference_features"],
                x["text_states"],
                x["attention_mask"],
                text_content_mask=x["content_mask"],
                teacher_reference_features=x["reference_native"],
                teacher_text_states=x["teacher_text_states"],
            )

            soft_masks = slot_output["slot_masks"]
            hard_masks = slot_output["value_hard_slot_masks"]
            qasa_selected = slot_output["qasa_selected_mask"]
            valid = x["attention_mask"] & x["content_mask"]

            hard_count = hard_masks.sum(dim=2)
            hard_nonempty = hard_count > 0
            valid_count = valid.sum(dim=1).float()
            hard_k = hard_nonempty.sum(dim=1).float()
            hard_dominant = hard_count.max(dim=1).values / valid_count.clamp_min(1.0)

            selected_count = qasa_selected.sum(dim=1).float()
            selected_nonempty = qasa_selected & hard_nonempty
            selected_empty = qasa_selected & ~hard_nonempty
            selected_nonempty_count = selected_nonempty.sum(dim=1).float()
            selected_empty_count = selected_empty.sum(dim=1).float()

            stats.add("routing/hard_value_effective_k", hard_k)
            stats.add("routing/hard_value_dominant_share", hard_dominant)
            stats.add(
                "routing/hard_value_empty_slot_fraction",
                (~hard_nonempty).float().mean(dim=1),
            )
            stats.add("routing/qasa_selected_count", selected_count)
            stats.add(
                "routing/qasa_selected_nonempty_precision",
                safe_ratio(selected_nonempty_count, selected_count),
            )
            stats.add(
                "routing/qasa_hard_nonempty_recall",
                safe_ratio(selected_nonempty_count, hard_k),
            )
            stats.add(
                "routing/qasa_selected_empty_fraction",
                safe_ratio(selected_empty_count, selected_count),
            )
            owned_unselected_tokens = (
                hard_count * (~qasa_selected).to(hard_count.dtype)
            ).sum(dim=1)
            stats.add(
                "routing/hard_owned_token_fraction_unselected_by_qasa",
                owned_unselected_tokens / valid_count.clamp_min(1.0),
            )

            top2 = soft_masks.topk(min(2, model.num_slots), dim=1).values
            stats.add("routing/soft_top1_probability", top2[:, 0, :][valid])
            if model.num_slots > 1:
                stats.add(
                    "routing/soft_top1_margin",
                    (top2[:, 0, :] - top2[:, 1, :])[valid],
                )
            probs = soft_masks.clamp_min(1e-12)
            entropy = -(probs * probs.log()).sum(dim=1)
            stats.add("routing/soft_entropy_per_token", entropy[valid])

            raw_hard = rebuild_edit_slots(
                model,
                value_states=x["teacher_text_states"],
                value_masks=hard_masks,
            )
            raw_soft = rebuild_edit_slots(
                model,
                value_states=x["teacher_text_states"],
                value_masks=soft_masks,
            )

            contextual_available = x["text_states"].shape[-1] == model.teacher_text_dim
            contextual_hard = contextual_soft = None
            if contextual_available:
                contextual_hard = rebuild_edit_slots(
                    model,
                    value_states=x["text_states"],
                    value_masks=hard_masks,
                )
                contextual_soft = rebuild_edit_slots(
                    model,
                    value_states=x["text_states"],
                    value_masks=soft_masks,
                )

            expected = raw_soft if model.slot_value_assignment == "soft_shared" else raw_hard
            stats.add(
                "smoke/deployed_reconstruction_max_abs_diff",
                (expected["edit_slots"] - slot_output["edit_slots"])
                .abs()
                .amax(dim=(1, 2)),
            )

            z0, reference_state = model.initialize_state(x["reference_features"])
            q_reference = model.make_query(z0)

            queries = {}
            executions = {}

            queries["deployed_qasa"], executions["deployed_qasa"] = execute_query(
                model,
                edit_slots=slot_output["edit_slots"],
                selected_mask=qasa_selected,
                z0=z0,
                reference_state=reference_state,
            )
            queries["raw_hard_qasa"], executions["raw_hard_qasa"] = execute_query(
                model,
                edit_slots=raw_hard["edit_slots"],
                selected_mask=qasa_selected,
                z0=z0,
                reference_state=reference_state,
            )
            (
                queries["raw_hard_qasa_nonempty"],
                executions["raw_hard_qasa_nonempty"],
            ) = execute_query(
                model,
                edit_slots=raw_hard["edit_slots"],
                selected_mask=selected_nonempty,
                z0=z0,
                reference_state=reference_state,
            )
            (
                queries["raw_hard_all_nonempty"],
                executions["raw_hard_all_nonempty"],
            ) = execute_query(
                model,
                edit_slots=raw_hard["edit_slots"],
                selected_mask=hard_nonempty,
                z0=z0,
                reference_state=reference_state,
            )
            queries["raw_soft_qasa"], executions["raw_soft_qasa"] = execute_query(
                model,
                edit_slots=raw_soft["edit_slots"],
                selected_mask=qasa_selected,
                z0=z0,
                reference_state=reference_state,
            )
            all_slots = torch.ones_like(qasa_selected)
            queries["raw_soft_all"], executions["raw_soft_all"] = execute_query(
                model,
                edit_slots=raw_soft["edit_slots"],
                selected_mask=all_slots,
                z0=z0,
                reference_state=reference_state,
            )

            if contextual_hard is not None and contextual_soft is not None:
                (
                    queries["contextual_hard_qasa_nonempty"],
                    executions["contextual_hard_qasa_nonempty"],
                ) = execute_query(
                    model,
                    edit_slots=contextual_hard["edit_slots"],
                    selected_mask=selected_nonempty,
                    z0=z0,
                    reference_state=reference_state,
                )
                (
                    queries["contextual_soft_qasa"],
                    executions["contextual_soft_qasa"],
                ) = execute_query(
                    model,
                    edit_slots=contextual_soft["edit_slots"],
                    selected_mask=qasa_selected,
                    z0=z0,
                    reference_state=reference_state,
                )

            queries["reference_only"] = q_reference
            if slot_output["q_teacher_full"].shape[-1] == model.query_dim:
                queries["teacher_full"] = F.normalize(
                    slot_output["q_teacher_full"].float(), dim=-1
                )

            q_empty_only, empty_execution = execute_query(
                model,
                edit_slots=raw_hard["edit_slots"],
                selected_mask=selected_empty,
                z0=z0,
                reference_state=reference_state,
            )
            queries["empty_selected_only"] = q_empty_only
            empty_actual_change = empty_execution["actual_state_changes"].float().norm(dim=-1)
            stats.add(
                "executor/empty_selected_actual_change_norm_sum",
                empty_actual_change.sum(dim=1),
            )
            stats.add(
                "executor/empty_selected_query_l2_from_reference",
                (q_empty_only - q_reference).float().norm(dim=-1),
            )
            stats.add(
                "executor/empty_selected_query_cosine_distance_from_reference",
                1.0
                - F.cosine_similarity(q_empty_only.float(), q_reference.float(), dim=-1),
            )

            for name, built in (
                ("raw_hard", raw_hard),
                ("raw_soft", raw_soft),
                ("contextual_hard", contextual_hard),
                ("contextual_soft", contextual_soft),
            ):
                if built is None:
                    continue
                stats.add(
                    f"representation/{name}/edit_slot_effective_rank",
                    effective_rank_rows(built["edit_slots"]),
                )
                stats.add(
                    f"representation/{name}/edit_slot_pairwise_cosine",
                    mean_pairwise_row_cosine(built["edit_slots"]),
                )
                stats.add(
                    f"representation/{name}/slot_semantic_norm",
                    built["slot_semantics"].float().norm(dim=-1),
                )

            stats.add(
                "tokens/raw_word_embedding_pairwise_cosine",
                mean_token_pair_cosine(x["teacher_text_states"], valid),
            )
            stats.add(
                "tokens/contextual_embedding_pairwise_cosine",
                mean_token_pair_cosine(x["text_states"], valid),
            )

            if contextual_hard is not None:
                active = hard_nonempty
                raw_sem = F.normalize(raw_hard["slot_semantics"].float(), dim=-1, eps=1e-8)
                ctx_sem = F.normalize(
                    contextual_hard["slot_semantics"].float(), dim=-1, eps=1e-8
                )
                stats.add(
                    "representation/raw_vs_contextual_same_hard_support_cosine",
                    (raw_sem * ctx_sem).sum(dim=-1)[active],
                )

            per_variant_ranks = {}
            for name, query in queries.items():
                _, ranks, _ = summarize_variant(
                    name=name,
                    query=query,
                    gallery_norm=gallery_norm,
                    target_indices=target_indices,
                    category=category,
                    recall=recall,
                    stats=stats,
                )
                per_variant_ranks[name] = ranks

            deployed_query = queries["deployed_qasa"]
            for name, query in queries.items():
                if name == "deployed_qasa":
                    continue
                stats.add(
                    f"query_shift/{name}/l2_from_deployed",
                    (query.float() - deployed_query.float()).norm(dim=-1),
                )
                stats.add(
                    f"query_shift/{name}/cosine_distance_from_deployed",
                    1.0
                    - F.cosine_similarity(query.float(), deployed_query.float(), dim=-1),
                )

            if len(examples) < max_examples_buffer:
                hard_rank = per_variant_ranks["raw_hard_qasa"]
                soft_rank = per_variant_ranks["raw_soft_qasa"]
                nonempty_rank = per_variant_ranks["raw_hard_qasa_nonempty"]
                ctx_rank = per_variant_ranks.get("contextual_hard_qasa_nonempty")

                for i in range(b):
                    row = {
                        "category": category,
                        "sample_id": str(batch.sample_ids[i]),
                        "modification_text": str(batch.modification_texts[i]),
                        "target_id": str(batch.target_ids[i]),
                        "qasa_selected_slots": [
                            s for s in range(model.num_slots) if bool(qasa_selected[i, s].item())
                        ],
                        "hard_nonempty_slots": [
                            s for s in range(model.num_slots) if bool(hard_nonempty[i, s].item())
                        ],
                        "qasa_selected_empty_slots": [
                            s for s in range(model.num_slots) if bool(selected_empty[i, s].item())
                        ],
                        "hard_winner_counts": [
                            int(v) for v in hard_count[i].detach().cpu().tolist()
                        ],
                        "hard_effective_k": int(hard_k[i].item()),
                        "hard_dominant_share": float(hard_dominant[i].item()),
                        "rank_deployed": int(per_variant_ranks["deployed_qasa"][i].item()),
                        "rank_raw_hard": int(hard_rank[i].item()),
                        "rank_raw_hard_nonempty": int(nonempty_rank[i].item()),
                        "rank_raw_soft": int(soft_rank[i].item()),
                        "rank_contextual_hard_nonempty": (
                            int(ctx_rank[i].item()) if ctx_rank is not None else None
                        ),
                    }
                    row["soft_rank_rescue"] = row["rank_raw_hard"] - row["rank_raw_soft"]
                    row["nonempty_rank_rescue"] = (
                        row["rank_raw_hard"] - row["rank_raw_hard_nonempty"]
                    )
                    row["contextual_rank_rescue"] = (
                        row["rank_raw_hard_nonempty"] - row["rank_contextual_hard_nonempty"]
                        if row["rank_contextual_hard_nonempty"] is not None
                        else None
                    )
                    examples.append(row)

            processed += b
            total_samples += b

    recall_result = recall.finalize()
    stats_result = stats.finalize()

    def rescue_score(row):
        values = [
            row.get("soft_rank_rescue"),
            row.get("nonempty_rank_rescue"),
            row.get("contextual_rank_rescue"),
        ]
        values = [v for v in values if isinstance(v, (int, float))]
        return max(values) if values else -math.inf

    examples = sorted(examples, key=rescue_score, reverse=True)[: args.num_examples]

    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_provenance": provenance,
        "loaded_model_provenance": model.experiment_provenance(),
        "num_samples": total_samples,
        "protocol": {
            "dataset_protocol": args.protocol,
            "max_queries_per_category": args.max_queries_per_category,
            "important_caveat": (
                "soft/raw/contextual variants are frozen-checkpoint inference probes. "
                "They localize bottlenecks but are NOT substitutes for retraining each "
                "mode from scratch for a fair ablation."
            ),
            "variant_meanings": {
                "deployed_qasa": "Exact checkpoint deployment path.",
                "raw_hard_qasa": "Raw VALUE + hard exclusive ownership + original QASA.",
                "raw_hard_qasa_nonempty": (
                    "Same as raw_hard_qasa, but QASA-selected hard-empty VALUE slots are blocked."
                ),
                "raw_hard_all_nonempty": (
                    "Execute every hard-nonempty VALUE slot, bypassing QASA selection."
                ),
                "raw_soft_qasa": (
                    "Same checkpoint, but raw VALUE is soft-shared at inference."
                ),
                "raw_soft_all": "Soft-shared raw VALUE and force all slots to execute.",
                "contextual_hard_qasa_nonempty": (
                    "Leak probe: contextual Q-Former VALUE with same hard support; hard-empty selected slots blocked."
                ),
                "contextual_soft_qasa": "Leak upper-bound probe: contextual VALUE + soft sharing.",
                "empty_selected_only": (
                    "Execute only QASA-selected slots that own ZERO hard VALUE tokens."
                ),
                "teacher_full": "Frozen CSMCIR global composed query, when dimensions match.",
                "reference_only": "No edit execution.",
            },
        },
        "recall": recall_result,
        "stats": stats_result,
        "examples": examples,
    }
    report["diagnosis"] = diagnosis_from_report(report)

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n" + "=" * 86)
    print("TAPER A3.2 LOCAL-VALUE FAILURE DIAGNOSIS")
    print("=" * 86)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Samples:    {total_samples}")
    print(f"Mode:       {model.slot_value_assignment}")

    order = [
        "deployed_qasa",
        "raw_hard_qasa",
        "raw_hard_qasa_nonempty",
        "raw_hard_all_nonempty",
        "raw_soft_qasa",
        "raw_soft_all",
        "contextual_hard_qasa_nonempty",
        "contextual_soft_qasa",
        "teacher_full",
        "reference_only",
        "empty_selected_only",
    ]
    print("\nRetrieval interventions:")
    for name in order:
        item = recall_result.get(name)
        if not item:
            continue
        print(
            f"  {name:34s} "
            f"R@10={item['recall_at_10']:6.2f} "
            f"R@50={item['recall_at_50']:6.2f} "
            f"Mean={item['mean_recall']:6.2f}"
        )

    def show_mean(key):
        item = stats_result.get(key)
        return None if item is None else item["mean"]

    print("\nCritical routing/executor diagnostics:")
    for key in (
        "routing/hard_value_effective_k",
        "routing/hard_value_dominant_share",
        "routing/hard_value_empty_slot_fraction",
        "routing/qasa_selected_empty_fraction",
        "routing/qasa_selected_nonempty_precision",
        "routing/qasa_hard_nonempty_recall",
        "routing/hard_owned_token_fraction_unselected_by_qasa",
        "executor/empty_selected_actual_change_norm_sum",
        "executor/empty_selected_query_l2_from_reference",
        "representation/raw_hard/edit_slot_effective_rank",
        "representation/raw_hard/edit_slot_pairwise_cosine",
        "tokens/raw_word_embedding_pairwise_cosine",
        "tokens/contextual_embedding_pairwise_cosine",
    ):
        value = show_mean(key)
        if value is not None:
            print(f"  {key:58s} {value:.6f}")

    print("\nAutomated interpretation:")
    for item in report["diagnosis"]:
        print(f"  [{item['severity'].upper():6s}] {item['name']}")
        for k, v in item["evidence"].items():
            if v is not None:
                if isinstance(v, float):
                    print(f"           {k}: {v:.4f}")
                else:
                    print(f"           {k}: {v}")
        print(f"           -> {item['interpretation']}")

    print(f"\nJSON report: {args.json_output}")


if __name__ == "__main__":
    run(parse_args())