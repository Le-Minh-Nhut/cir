from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from cache.features import (
    get_features_by_ids,
    get_text_features_by_sample_ids,
    load_features,
    load_text_features,
)
from datasets.common import collate_cir_samples
from datasets.fashioniq import (
    FashionIQDataset,
    load_correction_dict,
)
from evaluation.fashioniq import build_fashioniq_gallery
from models.taper import TAPER
from teachers.csmcir_compose import CSMCIRComposeTeacher


CATEGORIES = ("dress", "shirt", "toptee")


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose a trained TAPER Competitive-NULL checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to TAPER best.pt / last.pt",)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/FashionIQ"))
    parser.add_argument("--cache-root", type=Path, default=Path("features"))
    parser.add_argument("--config", type=Path, default=Path("conf/experiment/taper_e2e.yaml"))
    parser.add_argument("--protocol", type=str, default="fashioniq_original", choices=("fashioniq_original", "fashioniq_val"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-queries-per-category", type=int, default=0,
        help=(
            "0 = full validation category. "
            "For quick diagnosis use e.g. 256."
        ),
    )
    parser.add_argument("--top-worst", type=int, default=20)
    parser.add_argument("--json-output", type=Path, default=Path("reports/taper_checkpoint_diagnosis.json"))

    return parser.parse_args()


# ============================================================
# DATA
# ============================================================

def load_correction_dicts(annotation_root: Path):
    correction_dicts = {}

    for category in CATEGORIES:
        path = (
            annotation_root
            / f"correction_dict_{category}.json"
        )

        if not path.is_file():
            raise FileNotFoundError(
                f"Missing correction dictionary: {path}"
            )

        correction_dicts[category] = (
            load_correction_dict(path)
        )

    return correction_dicts


def build_val_loaders(
    *,
    annotation_root: Path,
    batch_size: int,
    num_workers: int,
    caption_policy: str,
    correction_dicts,
):
    loaders = {}
    annotations = {}

    for category in CATEGORIES:
        dataset = FashionIQDataset(
            annotation_root=annotation_root,
            split="val",
            categories=[category],
            caption_policy=caption_policy,
            correction_dicts=correction_dicts,
        )

        loaders[category] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_cir_samples,
            pin_memory=True,
        )

        annotations[category] = dataset.annotations

    return loaders, annotations


# ============================================================
# MODEL
# ============================================================

def build_model(
    *,
    cfg,
    device: torch.device,
):
    m = cfg.model

    teacher = CSMCIRComposeTeacher(
        csmcir_root=cfg.teacher.csmcir_root,
        checkpoint_path=cfg.teacher.checkpoint_path,
    ).to(device).eval()

    model = TAPER(
        teacher,
        text_dim=m.text_dim,
        reference_dim=m.reference_dim,
        teacher_text_dim=m.teacher_text_dim,
        teacher_query_dim=m.teacher_query_dim,
        query_dim=m.query_dim,
        slot_dim=m.slot_dim,
        state_dim=m.state_dim,
        num_slots=m.num_slots,
        num_primitives=m.num_primitives,
        mask_temperature=m.mask_temperature,
        router_temperature=m.router_temperature,
        retrieval_temperature=m.retrieval_temperature,
        neutral_mode=m.neutral_mode,
        slot_gate_threshold=m.slot_gate_threshold,
        hard_slot_gating_during_training=
            m.hard_slot_gating_during_training,
        gate_mode=m.gate_mode,
        st_gate_recovery=m.st_gate_recovery,
        alpha_max=m.alpha_max,
        counterfactual_chunk_size=
            m.counterfactual_chunk_size,
    ).to(device)

    return model


def load_taper_checkpoint(
    model: TAPER,
    checkpoint_path: Path,
):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    try:
        state = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        state = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

    # Slight robustness in case a future checkpoint wrapper is used.
    if (
        isinstance(state, dict)
        and "model_state_dict" in state
    ):
        state = state["model_state_dict"]

    elif (
        isinstance(state, dict)
        and "state_dict" in state
    ):
        state = state["state_dict"]

    missing, unexpected = model.load_state_dict(
        state,
        strict=False,
    )

    # Current fit() intentionally excludes frozen teacher.* weights.
    bad_missing = [
        key
        for key in missing
        if not key.startswith("teacher.")
    ]

    if bad_missing:
        raise RuntimeError(
            "Checkpoint is missing non-teacher parameters:\n"
            + "\n".join(bad_missing)
        )

    if unexpected:
        raise RuntimeError(
            "Checkpoint has unexpected parameters:\n"
            + "\n".join(unexpected)
        )

    print(
        f"Loaded checkpoint: {checkpoint_path}"
    )

    print(
        f"Expected frozen teacher missing keys: "
        f"{len(missing) - len(bad_missing)}"
    )


# ============================================================
# REPRESENTATIONAL DIAGNOSTICS
# ============================================================

def pairwise_cosine_per_sample(
    x: torch.Tensor,
):
    """
    x: [B, L, D]

    Returns:
        [B] mean pairwise off-diagonal cosine.
    """
    if x.ndim != 3:
        raise ValueError("x must be [B,L,D]")

    _, num_slots, _ = x.shape

    if num_slots < 2:
        return torch.zeros(
            x.shape[0],
            device=x.device,
            dtype=x.dtype,
        )

    normalized = F.normalize(
        x,
        dim=-1,
        eps=1e-6,
    )

    similarity = (
        normalized
        @ normalized.transpose(1, 2)
    )

    upper = torch.triu(
        torch.ones(
            num_slots,
            num_slots,
            dtype=torch.bool,
            device=x.device,
        ),
        diagonal=1,
    )

    return similarity[:, upper].mean(dim=1)


def ownership_similarity_per_sample(
    slot_masks: torch.Tensor,
    content_mask: torch.Tensor,
):
    """
    Compare token-ownership maps between slots.

    slot_masks:  [B,L,N]
    content_mask:[B,N]
    """
    valid = content_mask.to(slot_masks.dtype)

    masked = (
        slot_masks
        * valid[:, None, :]
    )

    return pairwise_cosine_per_sample(
        masked
    )


def normalized_entropy_per_sample(
    *,
    null_probs: torch.Tensor,
    slot_masks: torch.Tensor,
    content_mask: torch.Tensor,
):
    """
    Normalized by log(L+1), therefore approximately in [0,1].
    """
    all_probs = torch.cat(
        [
            null_probs[:, None, :],
            slot_masks,
        ],
        dim=1,
    )

    entropy = -(
        all_probs.clamp_min(1e-12)
        * all_probs.clamp_min(1e-12).log()
    ).sum(dim=1)

    entropy = (
        entropy
        / math.log(all_probs.shape[1])
    )

    valid = content_mask.to(entropy.dtype)

    denom = valid.sum(dim=1).clamp_min(1.0)

    return (
        entropy * valid
    ).sum(dim=1) / denom


def state_change_similarity_per_sample(
    actual_changes: torch.Tensor,
    valid_steps: torch.Tensor,
):
    """
    actual_changes: [B,S,D]
    valid_steps:    [B,S]

    Pairwise similarity between executed state changes only.
    Returns NaN when sample has fewer than two valid steps.
    """
    batch_size, num_steps, _ = (
        actual_changes.shape
    )

    normalized = F.normalize(
        actual_changes,
        dim=-1,
        eps=1e-6,
    )

    similarity = (
        normalized
        @ normalized.transpose(1, 2)
    )

    upper = torch.triu(
        torch.ones(
            num_steps,
            num_steps,
            dtype=torch.bool,
            device=actual_changes.device,
        ),
        diagonal=1,
    )

    pair_valid = (
        valid_steps[:, :, None]
        & valid_steps[:, None, :]
        & upper[None, :, :]
    )

    values = (
        similarity
        * pair_valid.to(similarity.dtype)
    ).sum(dim=(1, 2))

    counts = pair_valid.sum(dim=(1, 2))

    result = torch.full(
        (batch_size,),
        float("nan"),
        dtype=similarity.dtype,
        device=similarity.device,
    )

    has_pairs = counts > 0

    result[has_pairs] = (
        values[has_pairs]
        / counts[has_pairs]
    )

    return result


# ============================================================
# AGGREGATION
# ============================================================

class ScalarAccumulator:
    def __init__(self):
        self.sum = defaultdict(float)
        self.count = defaultdict(int)

    def add(
        self,
        name: str,
        values: torch.Tensor,
    ):
        values = (
            values.detach()
            .float()
            .reshape(-1)
        )

        finite = torch.isfinite(values)
        values = values[finite]

        if values.numel() == 0:
            return

        self.sum[name] += (
            values.sum().item()
        )

        self.count[name] += (
            values.numel()
        )

    def mean(self, name: str):
        count = self.count[name]

        if count == 0:
            return float("nan")

        return self.sum[name] / count


def target_ranks(
    scores: torch.Tensor,
    target_indices: torch.Tensor,
):
    """
    1-indexed approximate exact rank.

    Ties are extremely unlikely for these continuous retrieval scores.
    """
    target_scores = scores.gather(
        1,
        target_indices[:, None],
    )

    return (
        (scores > target_scores)
        .sum(dim=1)
        + 1
    )


def update_retrieval_statistics(
    *,
    category_stats,
    variant: str,
    category: str,
    ranks: torch.Tensor,
):
    stats = category_stats[variant][category]

    stats["n"] += int(ranks.numel())

    stats["hit10"] += int(
        (ranks <= 10).sum().item()
    )

    stats["hit50"] += int(
        (ranks <= 50).sum().item()
    )


def macro_retrieval_metrics(
    category_stats,
    variant,
):
    category_metrics = {}

    for category in CATEGORIES:
        stats = category_stats[
            variant
        ][category]

        n = stats["n"]

        if n == 0:
            continue

        r10 = (
            100.0
            * stats["hit10"]
            / n
        )

        r50 = (
            100.0
            * stats["hit50"]
            / n
        )

        category_metrics[category] = {
            "recall_at_10": r10,
            "recall_at_50": r50,
        }

    if not category_metrics:
        raise RuntimeError(
            f"No samples for variant {variant}"
        )

    avg_r10 = sum(
        x["recall_at_10"]
        for x in category_metrics.values()
    ) / len(category_metrics)

    avg_r50 = sum(
        x["recall_at_50"]
        for x in category_metrics.values()
    ) / len(category_metrics)

    return {
        "recall_at_10": avg_r10,
        "recall_at_50": avg_r50,
        "mean_recall":
            (avg_r10 + avg_r50) / 2.0,
        "categories": category_metrics,
    }


# ============================================================
# MAIN AUDIT
# ============================================================

@torch.no_grad()
def run_diagnosis(args):
    if args.batch_size < 1:
        raise ValueError(
            "--batch-size must be >= 1"
        )

    if (
        args.max_queries_per_category < 0
    ):
        raise ValueError(
            "--max-queries-per-category "
            "must be >= 0"
        )

    if (
        args.device.startswith("cuda")
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested but unavailable."
        )

    device = torch.device(args.device)

    cfg = OmegaConf.load(args.config)

    dataset_root = args.dataset_root
    annotation_root = (
        dataset_root / "captions"
    )

    split_root = (
        dataset_root / "image_splits"
    )

    correction_dicts = (
        load_correction_dicts(
            annotation_root
        )
    )

    val_loaders, val_annotations = (
        build_val_loaders(
            annotation_root=annotation_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            caption_policy=
                cfg.val_caption_policy,
            correction_dicts=
                correction_dicts,
        )
    )

    feature_root = (
        args.cache_root
        / "fashioniq"
        / "csmcir"
        / "val"
    )

    val_retrieval, val_retrieval_idx = (
        load_features(
            feature_root / "retrieval"
        )
    )

    val_native, val_native_idx = (
        load_features(
            feature_root / "native"
        )
    )

    val_text = load_text_features(
        feature_root / "text"
    )

    print("Device:", device)
    print(
        "Val retrieval:",
        tuple(val_retrieval.shape),
    )
    print(
        "Val native:",
        tuple(val_native.shape),
    )
    print(
        "Val text:",
        tuple(val_text.states.shape),
    )

    print()
    print("Loading model...")

    model = build_model(
        cfg=cfg,
        device=device,
    )

    load_taper_checkpoint(
        model,
        args.checkpoint,
    )

    model.eval()

    num_slots = model.num_slots
    num_primitives = model.num_primitives

    variant_names = (
        ["full", "reference_only"]
        + [
            f"drop_{slot_id}"
            for slot_id in range(num_slots)
        ]
    )

    category_stats = {
        variant: {
            category: {
                "n": 0,
                "hit10": 0,
                "hit50": 0,
            }
            for category in CATEGORIES
        }
        for variant in variant_names
    }

    rank_history = {
        variant: []
        for variant in variant_names
    }

    diagnostics = ScalarAccumulator()

    primitive_counts = torch.zeros(
        num_primitives,
        dtype=torch.long,
    )

    selected_slot_counts = torch.zeros(
        num_slots,
        dtype=torch.long,
    )

    examples = []

    # --------------------------------------------------------
    # Static query specialization
    # --------------------------------------------------------

    projected_slot_queries = (
        model.slot_query_projection(
            model.slot_queries
        )
    )

    query_similarity = (
        pairwise_cosine_per_sample(
            projected_slot_queries[
                None, :, :
            ]
        )[0]
        .item()
    )

    # --------------------------------------------------------
    # Category loop
    # --------------------------------------------------------

    for category in CATEGORIES:
        print()
        print("=" * 80)
        print(
            f"CATEGORY: {category}"
        )
        print("=" * 80)

        loader = val_loaders[category]

        gallery_ids = (
            build_fashioniq_gallery(
                protocol=args.protocol,
                split_root=split_root,
                split="val",
                category=category,
                annotations=
                    val_annotations[category],
            )
        )

        gallery_features = (
            get_features_by_ids(
                gallery_ids,
                val_retrieval,
                val_retrieval_idx,
            )
            .to(
                device=device,
                dtype=torch.float32,
            )
        )

        gallery_index = {
            image_id: index
            for index, image_id
            in enumerate(gallery_ids)
        }

        processed = 0

        progress = tqdm(
            loader,
            desc=f"Diagnose [{category}]",
            dynamic_ncols=True,
        )

        for batch in progress:
            if (
                args.max_queries_per_category
                and processed
                >= args.max_queries_per_category
            ):
                break

            reference_native = (
                get_features_by_ids(
                    batch.reference_ids,
                    val_native,
                    val_native_idx,
                )
                .to(
                    device=device,
                    dtype=torch.float32,
                )
            )

            reference_features = (
                reference_native[:, 0, :]
            )

            (
                text_states,
                teacher_text_states,
                attention_mask,
                content_mask,
            ) = get_text_features_by_sample_ids(
                batch.sample_ids,
                batch.modification_texts,
                val_text,
            )

            text_states = text_states.to(
                device=device,
                dtype=torch.float32,
            )

            teacher_text_states = (
                teacher_text_states.to(
                    device=device,
                    dtype=torch.float32,
                )
            )

            attention_mask = (
                attention_mask.to(
                    device=device,
                    dtype=torch.bool,
                )
            )

            content_mask = (
                content_mask.to(
                    device=device,
                    dtype=torch.bool,
                )
            )

            output = model.forward(
                reference_features,
                text_states,
                attention_mask,
                text_content_mask=
                    content_mask,
                teacher_reference_features=
                    reference_native,
                teacher_text_states=
                    teacher_text_states,
            )

            batch_size = (
                reference_features.shape[0]
            )

            take = batch_size

            if args.max_queries_per_category:
                remaining = (
                    args.max_queries_per_category
                    - processed
                )

                take = min(
                    take,
                    remaining,
                )

            if take <= 0:
                break

            # ==================================================
            # QUERY VARIANTS
            # ==================================================

            queries = {
                "full": output["q0"],
                "reference_only":
                    output["q_reference_only"],
            }

            for slot_id in range(num_slots):
                disabled = torch.zeros(
                    batch_size,
                    num_slots,
                    dtype=torch.bool,
                    device=device,
                )

                disabled[:, slot_id] = True

                drop_execution = model.execute(
                    output["edit_slots"],
                    output["slot_gates"],
                    output["z0"],
                    output["reference_state"],
                    disabled_slots=disabled,
                )

                queries[
                    f"drop_{slot_id}"
                ] = model.make_query(
                    drop_execution[
                        "final_state"
                    ]
                )

            # ==================================================
            # TARGET INDICES
            # ==================================================

            target_ids = list(
                batch.target_ids
            )

            if any(
                target_id is None
                for target_id in target_ids
            ):
                raise ValueError(
                    "Validation sample missing target_id."
                )

            target_indices = []

            for target_id in target_ids:
                if target_id not in gallery_index:
                    raise KeyError(
                        f"Target {target_id} "
                        f"not found in gallery."
                    )

                target_indices.append(
                    gallery_index[target_id]
                )

            target_indices = torch.tensor(
                target_indices,
                dtype=torch.long,
                device=device,
            )

            # ==================================================
            # RETRIEVAL / CAUSAL SLOT DROP
            # ==================================================

            batch_ranks = {}

            for variant, query in queries.items():
                scores = model._retrieval_scores(
                    query,
                    gallery_features,
                )

                ranks = target_ranks(
                    scores,
                    target_indices,
                )[:take]

                batch_ranks[variant] = (
                    ranks.detach()
                    .cpu()
                )

                update_retrieval_statistics(
                    category_stats=
                        category_stats,
                    variant=variant,
                    category=category,
                    ranks=ranks,
                )

                rank_history[
                    variant
                ].extend(
                    ranks.detach()
                    .cpu()
                    .tolist()
                )

            # ==================================================
            # SLOT REPRESENTATION DIAGNOSTICS
            # ==================================================

            entropy = (
                normalized_entropy_per_sample(
                    null_probs=
                        output["null_probs"],
                    slot_masks=
                        output["slot_masks"],
                    content_mask=
                        content_mask,
                )[:take]
            )

            ownership_similarity = (
                ownership_similarity_per_sample(
                    output["slot_masks"],
                    content_mask,
                )[:take]
            )

            semantic_similarity = (
                pairwise_cosine_per_sample(
                    output["slot_semantics"]
                )[:take]
            )

            effect_similarity = (
                pairwise_cosine_per_sample(
                    output["slot_effects"]
                )[:take]
            )

            edit_slot_similarity = (
                pairwise_cosine_per_sample(
                    output["edit_slots"]
                )[:take]
            )

            query_reference_cosine = (
                F.cosine_similarity(
                    output["q0"],
                    output[
                        "q_reference_only"
                    ],
                    dim=-1,
                )[:take]
            )

            query_reference_l2 = (
                (
                    output["q0"]
                    - output[
                        "q_reference_only"
                    ]
                )
                .norm(dim=-1)[:take]
            )

            state_similarity = (
                state_change_similarity_per_sample(
                    output[
                        "actual_state_changes"
                    ],
                    output[
                        "trace_valid_mask"
                    ],
                )[:take]
            )

            diagnostics.add(
                "normalized_assignment_entropy",
                entropy,
            )

            diagnostics.add(
                "ownership_pair_similarity",
                ownership_similarity,
            )

            diagnostics.add(
                "semantic_pair_similarity",
                semantic_similarity,
            )

            diagnostics.add(
                "effect_pair_similarity",
                effect_similarity,
            )

            diagnostics.add(
                "edit_slot_pair_similarity",
                edit_slot_similarity,
            )

            diagnostics.add(
                "query_reference_cosine",
                query_reference_cosine,
            )

            diagnostics.add(
                "query_reference_l2",
                query_reference_l2,
            )

            diagnostics.add(
                "state_change_pair_similarity",
                state_similarity,
            )

            diagnostics.add(
                "slot_gate_mean",
                output["slot_gates"][
                    :take
                ],
            )

            diagnostics.add(
                "slot_effect_norm",
                output[
                    "slot_effects"
                ][:take].norm(dim=-1),
            )

            diagnostics.add(
                "slot_semantic_norm",
                output[
                    "slot_semantics"
                ][:take].norm(dim=-1),
            )

            diagnostics.add(
                "edit_slot_norm",
                output[
                    "edit_slots"
                ][:take].norm(dim=-1),
            )

            diagnostics.add(
                "actual_state_change_norm",
                output[
                    "actual_state_changes"
                ][:take].norm(dim=-1),
            )

            valid_trace = (
                output[
                    "trace_valid_mask"
                ][:take]
            )

            diagnostics.add(
                "valid_execution_steps",
                valid_trace.sum(dim=1),
            )

            diagnostics.add(
                "hard_active_slots",
                output[
                    "hard_active_slot_mask"
                ][:take]
                .sum(dim=1),
            )

            # Only valid execution steps.
            valid_alpha = (
                output[
                    "transition_strengths"
                ][:take][valid_trace]
            )

            diagnostics.add(
                "transition_alpha",
                valid_alpha,
            )

            valid_route_confidence = (
                output[
                    "route_confidences"
                ][:take][valid_trace]
            )

            diagnostics.add(
                "route_confidence",
                valid_route_confidence,
            )

            valid_selected_gate = (
                output[
                    "trace_selected_slot_gates"
                ][:take][valid_trace]
            )

            diagnostics.add(
                "selected_gate",
                valid_selected_gate,
            )

            for slot_id in range(num_slots):
                diagnostics.add(
                    f"slot_{slot_id}_mass",
                    output[
                        "slot_mass"
                    ][:take, slot_id],
                )

                diagnostics.add(
                    f"slot_{slot_id}_gate",
                    output[
                        "slot_gates"
                    ][:take, slot_id],
                )

            # ==================================================
            # ROUTER / PRIMITIVE USAGE
            # ==================================================

            primitive_ids = (
                output[
                    "trace_primitive_ids"
                ][:take]
            )

            slot_ids = (
                output[
                    "trace_slot_ids"
                ][:take]
            )

            valid_flat = (
                valid_trace.reshape(-1)
            )

            primitive_flat = (
                primitive_ids.reshape(-1)[
                    valid_flat
                ]
            )

            slot_flat = (
                slot_ids.reshape(-1)[
                    valid_flat
                ]
            )

            for primitive_id in range(
                num_primitives
            ):
                primitive_counts[
                    primitive_id
                ] += (
                    primitive_flat
                    .eq(primitive_id)
                    .sum()
                    .cpu()
                )

            for slot_id in range(num_slots):
                selected_slot_counts[
                    slot_id
                ] += (
                    slot_flat
                    .eq(slot_id)
                    .sum()
                    .cpu()
                )

            # ==================================================
            # PER-SAMPLE RECORDS
            # ==================================================

            slot_mass_cpu = (
                output[
                    "slot_mass"
                ][:take]
                .detach()
                .cpu()
            )

            null_cpu = (
                output[
                    "null_probs"
                ][:take]
                .detach()
                .cpu()
            )

            primitive_trace_cpu = (
                primitive_ids[:take]
                .detach()
                .cpu()
            )

            valid_trace_cpu = (
                valid_trace[:take]
                .detach()
                .cpu()
            )

            for i in range(take):
                primitive_trace = []

                for step_id in range(
                    primitive_trace_cpu.shape[1]
                ):
                    if bool(
                        valid_trace_cpu[
                            i,
                            step_id,
                        ]
                    ):
                        primitive_trace.append(
                            int(
                                primitive_trace_cpu[
                                    i,
                                    step_id,
                                ]
                            )
                        )

                drop_ranks = [
                    int(
                        batch_ranks[
                            f"drop_{slot_id}"
                        ][i]
                    )
                    for slot_id
                    in range(num_slots)
                ]

                examples.append(
                    {
                        "category": category,
                        "sample_id":
                            str(
                                batch.sample_ids[i]
                            ),
                        "text":
                            str(
                                batch.modification_texts[
                                    i
                                ]
                            ),
                        "target_id":
                            str(target_ids[i]),
                        "full_rank":
                            int(
                                batch_ranks[
                                    "full"
                                ][i]
                            ),
                        "reference_rank":
                            int(
                                batch_ranks[
                                    "reference_only"
                                ][i]
                            ),
                        "drop_ranks":
                            drop_ranks,
                        "slot_mass": [
                            float(x)
                            for x in (
                                slot_mass_cpu[i]
                                .tolist()
                            )
                        ],
                        "null_mean":
                            float(
                                null_cpu[i][
                                    content_mask[
                                        i
                                    ].cpu()
                                ]
                                .mean()
                                .item()
                            ),
                        "entropy":
                            float(
                                entropy[i]
                                .cpu()
                                .item()
                            ),
                        "ownership_similarity":
                            float(
                                ownership_similarity[
                                    i
                                ]
                                .cpu()
                                .item()
                            ),
                        "semantic_similarity":
                            float(
                                semantic_similarity[
                                    i
                                ]
                                .cpu()
                                .item()
                            ),
                        "effect_similarity":
                            float(
                                effect_similarity[
                                    i
                                ]
                                .cpu()
                                .item()
                            ),
                        "edit_slot_similarity":
                            float(
                                edit_slot_similarity[
                                    i
                                ]
                                .cpu()
                                .item()
                            ),
                        "primitive_trace":
                            primitive_trace,
                    }
                )

            processed += take

            progress.set_postfix(
                processed=processed
            )

            if (
                args.max_queries_per_category
                and processed
                >= args.max_queries_per_category
            ):
                break

    # ========================================================
    # SUMMARIZE RETRIEVAL
    # ========================================================

    retrieval = {}

    for variant in variant_names:
        retrieval[variant] = (
            macro_retrieval_metrics(
                category_stats,
                variant,
            )
        )

    full_mean = retrieval[
        "full"
    ]["mean_recall"]

    reference_mean = retrieval[
        "reference_only"
    ]["mean_recall"]

    # ========================================================
    # PRIMITIVE DISTRIBUTION
    # ========================================================

    primitive_total = int(
        primitive_counts.sum().item()
    )

    primitive_fractions = []

    for count in primitive_counts.tolist():
        fraction = (
            count / primitive_total
            if primitive_total
            else 0.0
        )

        primitive_fractions.append(
            fraction
        )

    slot_total = int(
        selected_slot_counts.sum().item()
    )

    selected_slot_fractions = []

    for count in selected_slot_counts.tolist():
        fraction = (
            count / slot_total
            if slot_total
            else 0.0
        )

        selected_slot_fractions.append(
            fraction
        )

    # ========================================================
    # AGGREGATE DIAGNOSTICS
    # ========================================================

    diagnostic_names = [
        "normalized_assignment_entropy",
        "ownership_pair_similarity",
        "semantic_pair_similarity",
        "effect_pair_similarity",
        "edit_slot_pair_similarity",
        "query_reference_cosine",
        "query_reference_l2",
        "state_change_pair_similarity",
        "slot_gate_mean",
        "slot_effect_norm",
        "slot_semantic_norm",
        "edit_slot_norm",
        "actual_state_change_norm",
        "valid_execution_steps",
        "hard_active_slots",
        "transition_alpha",
        "route_confidence",
        "selected_gate",
    ]

    for slot_id in range(num_slots):
        diagnostic_names.extend(
            [
                f"slot_{slot_id}_mass",
                f"slot_{slot_id}_gate",
            ]
        )

    diagnostic_summary = {
        name: diagnostics.mean(name)
        for name in diagnostic_names
    }

    diagnostic_summary[
        "projected_slot_query_pair_similarity"
    ] = query_similarity

    # ========================================================
    # CAUSAL DELTAS
    # ========================================================

    causal_drops = {}

    for slot_id in range(num_slots):
        variant = f"drop_{slot_id}"

        drop_metrics = retrieval[variant]

        causal_drops[
            f"slot_{slot_id}"
        ] = {
            "delta_recall_at_10":
                retrieval["full"][
                    "recall_at_10"
                ]
                - drop_metrics[
                    "recall_at_10"
                ],

            "delta_recall_at_50":
                retrieval["full"][
                    "recall_at_50"
                ]
                - drop_metrics[
                    "recall_at_50"
                ],

            "delta_mean_recall":
                full_mean
                - drop_metrics[
                    "mean_recall"
                ],
        }

    reference_gap = {
        "delta_recall_at_10":
            retrieval["full"][
                "recall_at_10"
            ]
            - retrieval[
                "reference_only"
            ]["recall_at_10"],

        "delta_recall_at_50":
            retrieval["full"][
                "recall_at_50"
            ]
            - retrieval[
                "reference_only"
            ]["recall_at_50"],

        "delta_mean_recall":
            full_mean
            - reference_mean,
    }

    # ========================================================
    # HEURISTIC FLAGS
    #
    # These are diagnostic only.
    # They are NOT training thresholds.
    # ========================================================

    flags = []

    entropy = diagnostic_summary[
        "normalized_assignment_entropy"
    ]

    ownership_sim = diagnostic_summary[
        "ownership_pair_similarity"
    ]

    semantic_sim = diagnostic_summary[
        "semantic_pair_similarity"
    ]

    effect_sim = diagnostic_summary[
        "effect_pair_similarity"
    ]

    edit_sim = diagnostic_summary[
        "edit_slot_pair_similarity"
    ]

    state_sim = diagnostic_summary[
        "state_change_pair_similarity"
    ]

    max_primitive_share = (
        max(primitive_fractions)
        if primitive_fractions
        else 0.0
    )

    drop_deltas = [
        causal_drops[
            f"slot_{slot_id}"
        ]["delta_mean_recall"]
        for slot_id
        in range(num_slots)
    ]

    if (
        entropy >= 0.80
        and ownership_sim >= 0.80
    ):
        flags.append(
            "OWNERSHIP-DIFFUSE/SYMMETRY suspected: "
            "high normalized entropy + highly similar "
            "token ownership maps."
        )

    if (
        semantic_sim >= 0.80
        and effect_sim >= 0.80
    ):
        flags.append(
            "SEMANTIC/FUNCTIONAL SYMMETRY suspected: "
            "slot semantics and teacher effects are "
            "highly similar."
        )

    if (
        edit_sim >= 0.80
        and semantic_sim < 0.80
    ):
        flags.append(
            "SLOT-MLP COLLAPSE suspected: "
            "inputs differ but final Edit Slot latents "
            "are highly similar."
        )

    if reference_gap[
        "delta_mean_recall"
    ] < 5.0:
        flags.append(
            "REFERENCE SHORTCUT suspected: "
            "full retrieval improves less than 5 mean-recall "
            "points over reference-only."
        )

    if (
        max(
            abs(delta)
            for delta in drop_deltas
        ) < 1.5
        and reference_gap[
            "delta_mean_recall"
        ] >= 5.0
    ):
        flags.append(
            "CAUSAL SLOT REDUNDANCY suspected: "
            "text/execution matters globally, but dropping "
            "any individual slot barely changes retrieval."
        )

    if max_primitive_share >= 0.80:
        flags.append(
            "PRIMITIVE COLLAPSE suspected: "
            "one primitive receives >=80% of valid routes."
        )

    if (
        math.isfinite(state_sim)
        and state_sim >= 0.80
    ):
        flags.append(
            "EXECUTION-DIRECTION COLLAPSE suspected: "
            "different execution steps produce highly "
            "similar state-change directions."
        )

    if not flags:
        flags.append(
            "No obvious collapse detected by the coarse "
            "heuristics. Inspect causal deltas and detailed "
            "representational statistics."
        )

    # ========================================================
    # WORST EXAMPLES
    # ========================================================

    worst_examples = sorted(
        examples,
        key=lambda x: x["full_rank"],
        reverse=True,
    )[: args.top_worst]

    result = {
        "checkpoint":
            str(args.checkpoint),
        "protocol":
            args.protocol,
        "max_queries_per_category":
            args.max_queries_per_category,
        "retrieval":
            retrieval,
        "reference_gap":
            reference_gap,
        "causal_slot_drop":
            causal_drops,
        "diagnostics":
            diagnostic_summary,
        "primitive_counts":
            primitive_counts.tolist(),
        "primitive_fractions":
            primitive_fractions,
        "selected_slot_counts":
            selected_slot_counts.tolist(),
        "selected_slot_fractions":
            selected_slot_fractions,
        "heuristic_flags":
            flags,
        "worst_examples":
            worst_examples,
    }

    # ========================================================
    # PRINT
    # ========================================================

    print()
    print("=" * 80)
    print("TAPER CHECKPOINT DIAGNOSIS")
    print("=" * 80)

    print()
    print("=== RETRIEVAL ===")

    for variant in variant_names:
        metrics = retrieval[variant]

        print(
            f"{variant:<16} "
            f"R@10={metrics['recall_at_10']:7.3f} | "
            f"R@50={metrics['recall_at_50']:7.3f} | "
            f"mean={metrics['mean_recall']:7.3f}"
        )

    print()
    print("=== REFERENCE-ONLY GAP ===")

    print(
        "Full - reference-only | "
        f"ΔR@10={reference_gap['delta_recall_at_10']:.3f} | "
        f"ΔR@50={reference_gap['delta_recall_at_50']:.3f} | "
        f"Δmean={reference_gap['delta_mean_recall']:.3f}"
    )

    print()
    print("=== CAUSAL SLOT DROP ===")

    for slot_id in range(num_slots):
        values = causal_drops[
            f"slot_{slot_id}"
        ]

        print(
            f"drop slot {slot_id} | "
            f"ΔR@10={values['delta_recall_at_10']:7.3f} | "
            f"ΔR@50={values['delta_recall_at_50']:7.3f} | "
            f"Δmean={values['delta_mean_recall']:7.3f}"
        )

    print()
    print("=== SLOT REPRESENTATION ===")

    display_names = [
        "projected_slot_query_pair_similarity",
        "normalized_assignment_entropy",
        "ownership_pair_similarity",
        "semantic_pair_similarity",
        "effect_pair_similarity",
        "edit_slot_pair_similarity",
        "slot_effect_norm",
        "slot_semantic_norm",
        "edit_slot_norm",
    ]

    for name in display_names:
        print(
            f"{name:<42} "
            f"{diagnostic_summary[name]:.6f}"
        )

    print()
    print("=== REFERENCE / QUERY ===")

    for name in (
        "query_reference_cosine",
        "query_reference_l2",
    ):
        print(
            f"{name:<42} "
            f"{diagnostic_summary[name]:.6f}"
        )

    print()
    print("=== SLOT MASS / GATES ===")

    for slot_id in range(num_slots):
        print(
            f"slot {slot_id} | "
            f"mass={diagnostic_summary[f'slot_{slot_id}_mass']:.4f} | "
            f"gate={diagnostic_summary[f'slot_{slot_id}_gate']:.4f}"
        )

    print(
        f"valid_execution_steps"
        f"{'':<21} "
        f"{diagnostic_summary['valid_execution_steps']:.4f}"
    )

    print(
        f"hard_active_slots"
        f"{'':<25} "
        f"{diagnostic_summary['hard_active_slots']:.4f}"
    )

    print()
    print("=== ROUTER / EXECUTOR ===")

    print(
        f"{'route_confidence':<42} "
        f"{diagnostic_summary['route_confidence']:.6f}"
    )

    print(
        f"{'transition_alpha':<42} "
        f"{diagnostic_summary['transition_alpha']:.6f}"
    )

    print(
        f"{'selected_gate':<42} "
        f"{diagnostic_summary['selected_gate']:.6f}"
    )

    print(
        f"{'actual_state_change_norm':<42} "
        f"{diagnostic_summary['actual_state_change_norm']:.6f}"
    )

    print(
        f"{'state_change_pair_similarity':<42} "
        f"{diagnostic_summary['state_change_pair_similarity']:.6f}"
    )

    print()
    print("Primitive usage:")

    for primitive_id, fraction in enumerate(
        primitive_fractions
    ):
        print(
            f"  primitive {primitive_id}: "
            f"{100.0 * fraction:6.2f}% "
            f"({int(primitive_counts[primitive_id])})"
        )

    print()
    print("Selected Edit-Slot usage:")

    for slot_id, fraction in enumerate(
        selected_slot_fractions
    ):
        print(
            f"  slot {slot_id}: "
            f"{100.0 * fraction:6.2f}% "
            f"({int(selected_slot_counts[slot_id])})"
        )

    print()
    print("=== HEURISTIC FLAGS ===")

    for flag in flags:
        print(" -", flag)

    print()
    print("=== WORST RETRIEVAL EXAMPLES ===")

    for index, example in enumerate(
        worst_examples,
        start=1,
    ):
        print()
        print(
            f"[{index}] "
            f"{example['category']} | "
            f"rank={example['full_rank']} | "
            f"reference={example['reference_rank']}"
        )

        print(
            "text:",
            example["text"],
        )

        print(
            "drop ranks:",
            example["drop_ranks"],
        )

        print(
            "slot mass:",
            [
                round(x, 3)
                for x
                in example["slot_mass"]
            ],
        )

        print(
            f"null={example['null_mean']:.3f} | "
            f"entropy={example['entropy']:.3f} | "
            f"ownership_sim={example['ownership_similarity']:.3f}"
        )

        print(
            f"semantic_sim={example['semantic_similarity']:.3f} | "
            f"effect_sim={example['effect_similarity']:.3f} | "
            f"edit_sim={example['edit_slot_similarity']:.3f}"
        )

        print(
            "primitive trace:",
            example["primitive_trace"],
        )

    # ========================================================
    # SAVE JSON
    # ========================================================

    args.json_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.json_output.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(
        "Saved report:",
        args.json_output,
    )


def main():
    args = parse_args()

    torch.manual_seed(12345)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(12345)

    run_diagnosis(args)


if __name__ == "__main__":
    main()
