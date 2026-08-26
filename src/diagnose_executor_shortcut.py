from __future__ import annotations

"""
A3.2 Executor Shortcut Forensic
================================

Frozen-checkpoint diagnostic for:
    exp/e2e-a3.2-contextual-key-local-value

This script DOES NOT train or mutate the checkpoint.

It tests three hypotheses:

H1) Compute-ticket shortcut:
    Does one dominant slot become much stronger simply by being repeated across
    multiple recurrent Executor steps?

H2) Slot-content dependence:
    Holding state/reference/primitive fixed, does replacing the real slot with
    ZERO barely change the transition?

H3) Slot-identity dependence:
    Holding state/reference/primitive fixed, does replacing the real slot with
    a slot from another sample barely change the transition?

Important design choices:
- Executor tests bypass QASA selection on purpose.
- "Original full" executes only hard-nonempty VALUE slots.
- Dominant slot is the slot owning the most hard VALUE tokens.
- Repeat-xKeff uses the same number of execution tickets as the sample's
  number of hard-nonempty slots.
- Repeat-xL uses all configured slot/execution tickets.
- Transition dependence uses the SAME state, SAME reference, and SAME primitive
  for real / zero / shuffled slot variants, isolating slot content itself.
"""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from audit_taper_merit_p0 import (
    RecallCollector,
    ScalarCollector,
    build_target_indices,
    load_runtime,
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
            "Frozen-checkpoint forensic for Executor compute-ticket and "
            "slot-content shortcuts in A3.2."
        )
    )
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, default=Path("data/FashionIQ"))
    p.add_argument("--cache-root", type=Path, default=Path("features"))
    p.add_argument(
        "--config",
        type=Path,
        default=Path("conf/experiment/taper_e2e.yaml"),
    )
    p.add_argument(
        "--slot-value-assignment",
        choices=VALUE_MODES,
        default=None,
        help=(
            "Normally omit; checkpoint provenance is auto-detected. "
            "For the target A3.2 run this should resolve to hard_st_exclusive."
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
        help="0 = full validation set. Start with 256/category.",
    )
    p.add_argument(
        "--json-output",
        type=Path,
        default=Path("reports/a3_2_executor_shortcut_forensic.json"),
    )

    # Required by audit_taper_merit_p0.load_runtime().
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
        p = obj.get("experiment_provenance")
        if isinstance(p, dict):
            return dict(p)
    return {}


def target_rank(scores: torch.Tensor, target_indices: torch.Tensor) -> torch.Tensor:
    target = scores.gather(1, target_indices[:, None])
    return 1 + (scores > target).sum(dim=1)


def target_margin(scores: torch.Tensor, target_indices: torch.Tensor) -> torch.Tensor:
    target = scores.gather(1, target_indices[:, None]).squeeze(1)
    masked = scores.clone()
    masked.scatter_(1, target_indices[:, None], float("-inf"))
    hardest_negative = masked.max(dim=1).values
    return target - hardest_negative


def safe_ratio(num: torch.Tensor, den: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    num = num.float()
    den = den.float()
    out = torch.full_like(num, float("nan"))
    valid = den.abs() > eps
    out[valid] = num[valid] / den[valid]
    return out


def cosine_or_nan(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    na = a.float().norm(dim=-1)
    nb = b.float().norm(dim=-1)
    valid = (na > eps) & (nb > eps)
    cos = F.cosine_similarity(a.float(), b.float(), dim=-1, eps=eps)
    return torch.where(valid, cos, torch.full_like(cos, float("nan")))


def execute_query(model, edit_slots, selected_mask, z0, reference_state):
    execution = model.execute(
        edit_slots,
        selected_mask,
        z0,
        reference_state,
    )
    query = model.make_query(execution["final_state"])
    return query, execution


def first_k_mask(k: torch.Tensor, num_slots: int) -> torch.Tensor:
    steps = torch.arange(num_slots, device=k.device)[None, :]
    return steps < k[:, None]


def gather_slot(slots: torch.Tensor, slot_ids: torch.Tensor) -> torch.Tensor:
    b = slots.shape[0]
    batch_ids = torch.arange(b, device=slots.device)
    return slots[batch_ids, slot_ids]


def summarize_retrieval(
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
    rank = target_rank(scores, target_indices).float()
    margin = target_margin(scores, target_indices)
    stats.add(f"retrieval/{name}/target_rank", rank)
    stats.add(f"retrieval/{name}/target_margin", margin)
    return rank, margin


def _mean(report_stats: dict, key: str):
    x = report_stats.get(key)
    return None if x is None else x.get("mean")


def _median(report_stats: dict, key: str):
    x = report_stats.get(key)
    return None if x is None else x.get("median")


def build_verdict(report: dict) -> dict:
    stats = report["stats"]
    recall = report["recall"]

    zero_rel = _median(stats, "transition/real_vs_zero_relative_delta_difference")
    zero_cos = _median(stats, "transition/real_vs_zero_delta_cosine")
    shuf_rel = _median(stats, "transition/real_vs_shuffled_relative_delta_difference")
    shuf_cos = _median(stats, "transition/real_vs_shuffled_delta_cosine")

    repeat_l_move = _median(
        stats, "compute_ticket/repeat_xL_over_single_query_move_ratio"
    )
    repeat_k_move = _median(
        stats, "compute_ticket/repeat_xKeff_over_single_query_move_ratio"
    )

    def mr(name):
        item = recall.get(name)
        return None if item is None else float(item["mean_recall"])

    single_mr = mr("dominant_single")
    repeat_k_mr = mr("dominant_repeat_xKeff")
    repeat_l_mr = mr("dominant_repeat_xL")
    original_mr = mr("original_all_nonempty")

    slot_ignore_score = 0
    if zero_rel is not None and math.isfinite(zero_rel) and zero_rel < 0.25:
        slot_ignore_score += 1
    if zero_cos is not None and math.isfinite(zero_cos) and zero_cos > 0.90:
        slot_ignore_score += 1
    if shuf_rel is not None and math.isfinite(shuf_rel) and shuf_rel < 0.35:
        slot_ignore_score += 1
    if shuf_cos is not None and math.isfinite(shuf_cos) and shuf_cos > 0.85:
        slot_ignore_score += 1

    compute_ticket_score = 0
    if repeat_k_move is not None and math.isfinite(repeat_k_move) and repeat_k_move > 1.25:
        compute_ticket_score += 1
    if repeat_l_move is not None and math.isfinite(repeat_l_move) and repeat_l_move > 1.50:
        compute_ticket_score += 1
    if (
        single_mr is not None
        and repeat_l_mr is not None
        and repeat_l_mr > single_mr + 0.5
    ):
        compute_ticket_score += 1
    if (
        original_mr is not None
        and repeat_l_mr is not None
        and repeat_l_mr >= original_mr - 0.25
    ):
        compute_ticket_score += 1

    slot_dependence = (
        "WEAK"
        if slot_ignore_score >= 3
        else "SUSPICIOUS"
        if slot_ignore_score >= 2
        else "PRESENT"
    )
    compute_ticket = (
        "STRONG"
        if compute_ticket_score >= 3
        else "SUSPICIOUS"
        if compute_ticket_score >= 2
        else "NOT_CONFIRMED"
    )

    return {
        "slot_content_dependence": slot_dependence,
        "compute_ticket_shortcut": compute_ticket,
        "slot_ignore_score_0_to_4": slot_ignore_score,
        "compute_ticket_score_0_to_4": compute_ticket_score,
        "headline_evidence": {
            "median_real_vs_zero_relative_delta_difference": zero_rel,
            "median_real_vs_zero_delta_cosine": zero_cos,
            "median_real_vs_shuffled_relative_delta_difference": shuf_rel,
            "median_real_vs_shuffled_delta_cosine": shuf_cos,
            "median_repeat_xKeff_over_single_query_move_ratio": repeat_k_move,
            "median_repeat_xL_over_single_query_move_ratio": repeat_l_move,
            "mean_recall_original_all_nonempty": original_mr,
            "mean_recall_dominant_single": single_mr,
            "mean_recall_dominant_repeat_xKeff": repeat_k_mr,
            "mean_recall_dominant_repeat_xL": repeat_l_mr,
        },
        "interpretation": (
            "STRONG compute-ticket + WEAK slot-content dependence is the clearest "
            "signature that recurrent Executor depth is substituting for slot "
            "specialization. If slot-content dependence is PRESENT but compute-ticket "
            "is STRONG, Executor still exploits recurrence, but it is not simply "
            "ignoring slot information. If neither is confirmed, the dominant cause "
            "is more likely upstream routing/objective pressure."
        ),
    }


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
    total_samples = 0

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
        for batch in tqdm(
            loader,
            desc=f"EXECUTOR FORENSIC [{category}]",
            dynamic_ncols=True,
        ):
            if args.max_queries_per_category and processed >= args.max_queries_per_category:
                break

            x = prepare_batch(runtime, batch)
            b = x["reference_features"].shape[0]

            # Respect exact per-category cap even when the final batch crosses it.
            if args.max_queries_per_category:
                remaining = args.max_queries_per_category - processed
                if remaining <= 0:
                    break
                if b > remaining:
                    # Current project batches are structured objects, so slicing the
                    # batch itself is awkward. Stop before exceeding the requested cap.
                    break

            target_indices = build_target_indices(batch, gallery_index, device)

            slot_output = model.build_edit_slots(
                x["reference_features"],
                x["text_states"],
                x["attention_mask"],
                text_content_mask=x["content_mask"],
                teacher_reference_features=x["reference_native"],
                teacher_text_states=x["teacher_text_states"],
            )

            edit_slots = slot_output["edit_slots"]
            hard_masks = slot_output["value_hard_slot_masks"]
            hard_count = hard_masks.sum(dim=2)
            hard_nonempty = hard_count > 0
            hard_k = hard_nonempty.sum(dim=1)
            valid_token_count = hard_count.sum(dim=1)
            valid_sample = valid_token_count > 0

            if not valid_sample.all():
                # FashionIQ should not normally hit this. Fail loudly rather than
                # silently fabricate a dominant slot for an empty correction.
                bad = (~valid_sample).nonzero(as_tuple=False).reshape(-1).tolist()
                raise RuntimeError(
                    f"Found samples with zero hard VALUE tokens in batch positions {bad}"
                )

            dominant_ids = hard_count.argmax(dim=1)
            dominant_slots = gather_slot(edit_slots, dominant_ids)

            stats.add("routing/hard_effective_k", hard_k.float())
            stats.add(
                "routing/dominant_hard_token_share",
                hard_count.max(dim=1).values
                / valid_token_count.clamp_min(1.0),
            )

            z0, reference_state = model.initialize_state(x["reference_features"])
            q_reference = model.make_query(z0)

            # ---------------------------------------------------------------
            # A) Original hard-nonempty slots.
            # ---------------------------------------------------------------
            q_original, ex_original = execute_query(
                model,
                edit_slots,
                hard_nonempty,
                z0,
                reference_state,
            )

            # ---------------------------------------------------------------
            # B) Dominant slot, exactly one execution ticket.
            # ---------------------------------------------------------------
            single_mask = F.one_hot(
                dominant_ids,
                num_classes=model.num_slots,
            ).to(torch.bool)

            q_single, ex_single = execute_query(
                model,
                edit_slots,
                single_mask,
                z0,
                reference_state,
            )

            # ---------------------------------------------------------------
            # C) Same dominant slot copied into K_eff tickets.
            # ---------------------------------------------------------------
            repeated_slots = dominant_slots[:, None, :].expand(
                -1, model.num_slots, -1
            ).clone()
            repeat_keff_mask = first_k_mask(hard_k, model.num_slots)

            q_repeat_keff, ex_repeat_keff = execute_query(
                model,
                repeated_slots,
                repeat_keff_mask,
                z0,
                reference_state,
            )

            # ---------------------------------------------------------------
            # D) Same dominant slot copied into all L tickets.
            # ---------------------------------------------------------------
            repeat_l_mask = torch.ones(
                b,
                model.num_slots,
                dtype=torch.bool,
                device=device,
            )
            q_repeat_l, ex_repeat_l = execute_query(
                model,
                repeated_slots,
                repeat_l_mask,
                z0,
                reference_state,
            )

            # Query/state movement: pure model behavior, no target labels needed.
            single_q_move = (q_single - q_reference).float().norm(dim=-1)
            repeat_keff_q_move = (q_repeat_keff - q_reference).float().norm(dim=-1)
            repeat_l_q_move = (q_repeat_l - q_reference).float().norm(dim=-1)
            original_q_move = (q_original - q_reference).float().norm(dim=-1)

            stats.add("compute_ticket/single_query_move", single_q_move)
            stats.add("compute_ticket/repeat_xKeff_query_move", repeat_keff_q_move)
            stats.add("compute_ticket/repeat_xL_query_move", repeat_l_q_move)
            stats.add("compute_ticket/original_all_nonempty_query_move", original_q_move)
            stats.add(
                "compute_ticket/repeat_xKeff_over_single_query_move_ratio",
                safe_ratio(repeat_keff_q_move, single_q_move),
            )
            stats.add(
                "compute_ticket/repeat_xL_over_single_query_move_ratio",
                safe_ratio(repeat_l_q_move, single_q_move),
            )
            stats.add(
                "compute_ticket/repeat_xL_vs_original_query_cosine",
                cosine_or_nan(q_repeat_l - q_reference, q_original - q_reference),
            )

            single_state_move = (
                ex_single["final_state"] - z0
            ).float().norm(dim=-1)
            repeat_keff_state_move = (
                ex_repeat_keff["final_state"] - z0
            ).float().norm(dim=-1)
            repeat_l_state_move = (
                ex_repeat_l["final_state"] - z0
            ).float().norm(dim=-1)

            stats.add("compute_ticket/single_state_move", single_state_move)
            stats.add(
                "compute_ticket/repeat_xKeff_state_move",
                repeat_keff_state_move,
            )
            stats.add(
                "compute_ticket/repeat_xL_state_move",
                repeat_l_state_move,
            )
            stats.add(
                "compute_ticket/repeat_xKeff_over_single_state_move_ratio",
                safe_ratio(repeat_keff_state_move, single_state_move),
            )
            stats.add(
                "compute_ticket/repeat_xL_over_single_state_move_ratio",
                safe_ratio(repeat_l_state_move, single_state_move),
            )

            # ---------------------------------------------------------------
            # E) First-step slot dependence with SAME primitive.
            #
            # Use the primitive that the actual one-slot execution chose.
            # Then call _transition with:
            #   real slot / zero slot / shuffled slot
            # while keeping state/reference/primitive fixed.
            # ---------------------------------------------------------------
            primitive_ids = ex_single["trace_primitive_ids"][:, 0]
            if (primitive_ids < 0).any():
                raise RuntimeError(
                    "Dominant single-slot execution unexpectedly produced invalid primitive id"
                )
            primitive = model.primitive_bank[primitive_ids]
            valid_step = torch.ones(b, dtype=torch.bool, device=device)

            _, _, _, delta_real = model._transition(
                z0,
                dominant_slots,
                primitive,
                reference_state,
                valid_step,
            )
            _, _, _, delta_zero = model._transition(
                z0,
                torch.zeros_like(dominant_slots),
                primitive,
                reference_state,
                valid_step,
            )

            delta_real = delta_real.float()
            delta_zero = delta_zero.float()
            real_norm = delta_real.norm(dim=-1)
            zero_norm = delta_zero.norm(dim=-1)

            stats.add("transition/real_delta_norm", real_norm)
            stats.add("transition/zero_delta_norm", zero_norm)
            stats.add(
                "transition/zero_over_real_delta_norm_ratio",
                safe_ratio(zero_norm, real_norm),
            )
            stats.add(
                "transition/real_vs_zero_delta_cosine",
                cosine_or_nan(delta_real, delta_zero),
            )
            stats.add(
                "transition/real_vs_zero_relative_delta_difference",
                safe_ratio(
                    (delta_real - delta_zero).norm(dim=-1),
                    real_norm,
                ),
            )

            # Cross-sample slot shuffle. roll(1) guarantees no identity mapping
            # for b > 1 and does not touch state/reference/primitive.
            if b > 1:
                shuffled_slots = dominant_slots.roll(shifts=1, dims=0)
                _, _, _, delta_shuffled = model._transition(
                    z0,
                    shuffled_slots,
                    primitive,
                    reference_state,
                    valid_step,
                )
                delta_shuffled = delta_shuffled.float()

                stats.add(
                    "transition/shuffled_delta_norm",
                    delta_shuffled.norm(dim=-1),
                )
                stats.add(
                    "transition/real_vs_shuffled_delta_cosine",
                    cosine_or_nan(delta_real, delta_shuffled),
                )
                stats.add(
                    "transition/real_vs_shuffled_relative_delta_difference",
                    safe_ratio(
                        (delta_real - delta_shuffled).norm(dim=-1),
                        real_norm,
                    ),
                )

            # ---------------------------------------------------------------
            # F) Retrieval consequence.
            # ---------------------------------------------------------------
            ranks = {}
            margins = {}
            for name, query in (
                ("reference_only", q_reference),
                ("original_all_nonempty", q_original),
                ("dominant_single", q_single),
                ("dominant_repeat_xKeff", q_repeat_keff),
                ("dominant_repeat_xL", q_repeat_l),
            ):
                rank, margin = summarize_retrieval(
                    name=name,
                    query=query,
                    gallery_norm=gallery_norm,
                    target_indices=target_indices,
                    category=category,
                    recall=recall,
                    stats=stats,
                )
                ranks[name] = rank
                margins[name] = margin

            ref_rank = ranks["reference_only"]
            ref_margin = margins["reference_only"]
            for name in (
                "original_all_nonempty",
                "dominant_single",
                "dominant_repeat_xKeff",
                "dominant_repeat_xL",
            ):
                stats.add(
                    f"retrieval_gain/{name}/rank_improvement_vs_reference",
                    ref_rank - ranks[name],
                )
                stats.add(
                    f"retrieval_gain/{name}/margin_improvement_vs_reference",
                    margins[name] - ref_margin,
                )

            single_margin_gain = margins["dominant_single"] - ref_margin
            repeat_keff_margin_gain = (
                margins["dominant_repeat_xKeff"] - ref_margin
            )
            repeat_l_margin_gain = margins["dominant_repeat_xL"] - ref_margin
            stats.add(
                "compute_ticket/repeat_xKeff_over_single_positive_margin_gain_ratio",
                safe_ratio(
                    repeat_keff_margin_gain,
                    single_margin_gain,
                ),
            )
            stats.add(
                "compute_ticket/repeat_xL_over_single_positive_margin_gain_ratio",
                safe_ratio(
                    repeat_l_margin_gain,
                    single_margin_gain,
                ),
            )

            processed += b
            total_samples += b

    report = {
        "checkpoint": str(args.checkpoint),
        "experiment_provenance": provenance,
        "loaded_slot_value_assignment": args.slot_value_assignment,
        "num_samples": total_samples,
        "protocol": {
            "dataset_protocol": args.protocol,
            "max_queries_per_category": args.max_queries_per_category,
            "important_semantics": [
                "No training and no checkpoint mutation.",
                "Executor tests bypass QASA to isolate Executor behavior.",
                "Original full executes hard-nonempty VALUE slots only.",
                "Dominant slot is the hard VALUE slot owning the most tokens.",
                "repeat_xKeff gives the dominant slot the same number of execution tickets as the sample's hard-nonempty slot count.",
                "repeat_xL gives the dominant slot all configured execution tickets.",
                "real/zero/shuffled transition tests hold state, reference, and primitive fixed.",
                "Shuffled slot is cross-sample; it is a sensitivity probe, not a semantic counterfactual.",
            ],
        },
        "recall": recall.finalize(),
        "stats": stats.finalize(),
    }
    report["verdict"] = build_verdict(report)

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== A3.2 EXECUTOR SHORTCUT FORENSIC ===")
    print(f"samples: {total_samples}")
    print(f"report : {args.json_output}")

    v = report["verdict"]
    print("\n--- Verdict ---")
    print(f"slot-content dependence : {v['slot_content_dependence']}")
    print(f"compute-ticket shortcut : {v['compute_ticket_shortcut']}")

    e = v["headline_evidence"]
    print("\n--- Headline evidence ---")
    for key, value in e.items():
        print(f"{key}: {value}")

    return report


def main():
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()