from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm

from cache.features import (
    get_features_by_ids,
    get_text_features_by_sample_ids,
    load_features,
    load_text_features,
)
from evaluate_qasa_inference import (
    CATEGORIES,
    SLOT_VALUE_ASSIGNMENTS,
    SLOT_VALUE_SOURCES,
    build_model,
    build_val_loaders,
    load_checkpoint,
    load_correction_dicts,
)
from evaluation.fashioniq import build_fashioniq_gallery


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "TAPER-MERIT P0 frozen-checkpoint audit for the contextual-key/"
            "local-value QASA teacher branch. "
            "Measures exact slot coalitions, SINGLE/DROP/REPEAT/MEAN recovery, "
            "per-hard-negative functional effects, and QASA-vs-functional agreement. "
            "This script never trains or mutates the checkpoint."
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
        "--slot-value-source",
        choices=SLOT_VALUE_SOURCES,
        default=None,
        help="Override model.slot_value_source; must match checkpoint provenance.",
    )
    p.add_argument(
        "--slot-effect-in-value",
        choices=("true", "false"),
        default=None,
        help="Override model.slot_effect_in_value; must match checkpoint provenance.",
    )
    p.add_argument(
        "--slot-value-assignment",
        choices=SLOT_VALUE_ASSIGNMENTS,
        default=None,
        help="Override model.slot_value_assignment; must match checkpoint provenance.",
    )
    p.add_argument(
        "--functional-ownership-enabled",
        choices=("true", "false"),
        default=None,
        help="Override functional ownership provenance; must match checkpoint.",
    )
    p.add_argument(
        "--protocol",
        type=str,
        default="fashioniq_original",
        choices=("fashioniq_original", "fashioniq_val"),
    )
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--max-queries-per-category",
        type=int,
        default=256,
        help="0 = full validation; 256/category is a good first P0 pass.",
    )
    p.add_argument(
        "--hard-negatives",
        type=int,
        default=16,
        help="Number of fixed hard negatives mined per query.",
    )
    p.add_argument(
        "--negative-source",
        type=str,
        default="qasa_full",
        choices=("qasa_full", "all_slots_full", "reference_only"),
        help=(
            "Query used once to mine the fixed hard-negative bank. "
            "The same negatives are then reused for every intervention."
        ),
    )
    p.add_argument(
        "--pairwise-margin",
        type=float,
        default=0.0,
        help="Margin m in softplus((s_neg - s_pos + m) / tau).",
    )
    p.add_argument(
        "--pairwise-tau",
        type=float,
        default=None,
        help="Pairwise error temperature. Default: model retrieval_temperature.",
    )
    p.add_argument(
        "--phi-positive-threshold",
        type=float,
        default=1e-4,
        help="Minimum positive conditional loss reduction to call a slot/mode useful.",
    )
    p.add_argument(
        "--min-full-gain",
        type=float,
        default=1e-4,
        help="Ratios are reported only when forced-all FULL gain exceeds this value.",
    )
    p.add_argument("--num-examples", type=int, default=20)
    p.add_argument(
        "--json-output",
        type=Path,
        default=Path("reports/taper_merit_p0_audit.json"),
    )
    return p.parse_args()


class ScalarCollector:
    def __init__(self):
        self.values: dict[str, list[torch.Tensor]] = defaultdict(list)

    def add(self, name: str, x):
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        x = x.detach().float().reshape(-1).cpu()
        x = x[torch.isfinite(x)]
        if x.numel():
            self.values[name].append(x)

    def finalize(self) -> dict[str, dict[str, float | int]]:
        result = {}
        quantiles = torch.tensor([0.10, 0.50, 0.90, 0.95, 0.99])
        for name, chunks in self.values.items():
            if not chunks:
                continue
            x = torch.cat(chunks)
            q = torch.quantile(x, quantiles)
            result[name] = {
                "n": int(x.numel()),
                "mean": float(x.mean()),
                "std": float(x.std(unbiased=False)),
                "min": float(x.min()),
                "p10": float(q[0]),
                "median": float(q[1]),
                "p90": float(q[2]),
                "p95": float(q[3]),
                "p99": float(q[4]),
                "max": float(x.max()),
            }
        return result


class RecallCollector:
    def __init__(self):
        self.data = defaultdict(
            lambda: defaultdict(lambda: {"n": 0, "hit10": 0, "hit50": 0})
        )

    def update(
        self,
        variant: str,
        category: str,
        scores: torch.Tensor,
        target_indices: torch.Tensor,
    ):
        if scores.ndim != 2:
            raise ValueError("scores must be [B,G]")
        if target_indices.shape != (scores.shape[0],):
            raise ValueError("target_indices must be [B]")

        max_k = min(50, scores.shape[1])
        top = scores.topk(max_k, dim=1).indices
        target = target_indices[:, None]
        hit10 = top[:, : min(10, max_k)].eq(target).any(dim=1)
        hit50 = top.eq(target).any(dim=1)

        d = self.data[variant][category]
        d["n"] += scores.shape[0]
        d["hit10"] += int(hit10.sum().item())
        d["hit50"] += int(hit50.sum().item())

    def finalize(self):
        out = {}
        for variant, per_category in self.data.items():
            category_result = {}
            r10_values = []
            r50_values = []
            for category in CATEGORIES:
                d = per_category.get(category)
                if not d or d["n"] == 0:
                    continue
                r10 = 100.0 * d["hit10"] / d["n"]
                r50 = 100.0 * d["hit50"] / d["n"]
                category_result[category] = {
                    "n": d["n"],
                    "recall_at_10": r10,
                    "recall_at_50": r50,
                }
                r10_values.append(r10)
                r50_values.append(r50)

            if r10_values:
                macro_r10 = sum(r10_values) / len(r10_values)
                macro_r50 = sum(r50_values) / len(r50_values)
                out[variant] = {
                    "recall_at_10": macro_r10,
                    "recall_at_50": macro_r50,
                    "mean_recall": (macro_r10 + macro_r50) / 2.0,
                    "per_category": category_result,
                }
        return out


def coalition_masks(num_slots: int, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [
            [bool(mask & (1 << slot_id)) for slot_id in range(num_slots)]
            for mask in range(1 << num_slots)
        ],
        dtype=torch.bool,
        device=device,
    )


def coalition_name(mask: int, num_slots: int) -> str:
    slots = [f"S{s}" for s in range(num_slots) if mask & (1 << s)]
    return "+".join(slots) if slots else "EMPTY"


def execute_explicit_variants(
    model,
    *,
    variant_slots: torch.Tensor,
    variant_masks: torch.Tensor,
    z0: torch.Tensor,
    reference_state: torch.Tensor,
) -> torch.Tensor:
    """
    variant_slots: [B,V,L,D]
    variant_masks: [B,V,L] bool

    execute() itself always loops exactly model.num_slots times. Therefore every
    intervention keeps the same graph depth; inactive steps become exact no-op state
    updates under the current TAPER executor.
    """
    if variant_slots.ndim != 4:
        raise ValueError("variant_slots must be [B,V,L,D]")
    if variant_masks.ndim != 3:
        raise ValueError("variant_masks must be [B,V,L]")
    b, v, l, d = variant_slots.shape
    if l != model.num_slots:
        raise ValueError("variant slot count mismatch")
    if variant_masks.shape != (b, v, l):
        raise ValueError("variant mask shape mismatch")
    if variant_masks.dtype != torch.bool:
        raise TypeError("variant_masks must be bool")

    flat_slots = variant_slots.reshape(b * v, l, d)
    flat_masks = variant_masks.reshape(b * v, l)
    flat_z0 = z0[:, None, :].expand(b, v, -1).reshape(b * v, -1)
    flat_reference = (
        reference_state[:, None, :].expand(b, v, -1).reshape(b * v, -1)
    )

    execution = model.execute(
        flat_slots,
        flat_masks,
        flat_z0,
        flat_reference,
    )
    queries = model.make_query(execution["final_state"])
    return queries.reshape(b, v, -1)


def execute_same_slots_variants(
    model,
    *,
    edit_slots: torch.Tensor,
    masks: torch.Tensor,
    z0: torch.Tensor,
    reference_state: torch.Tensor,
) -> torch.Tensor:
    if masks.ndim != 2:
        raise ValueError("masks must be [V,L]")
    b, l, d = edit_slots.shape
    v = masks.shape[0]
    variant_slots = edit_slots[:, None, :, :].expand(b, v, l, d)
    variant_masks = masks[None, :, :].expand(b, v, l)
    return execute_explicit_variants(
        model,
        variant_slots=variant_slots,
        variant_masks=variant_masks,
        z0=z0,
        reference_state=reference_state,
    )


def score_gallery(query: torch.Tensor, gallery_norm: torch.Tensor) -> torch.Tensor:
    if query.ndim != 2:
        raise ValueError("query must be [B,D]")
    if gallery_norm.ndim != 3:
        raise ValueError("gallery must be [G,T,D]")
    q = F.normalize(query.float(), dim=-1)
    token_scores = torch.einsum("bd,gtd->bgt", q, gallery_norm)
    return token_scores.amax(dim=-1)


def score_local_candidates(
    queries: torch.Tensor,
    candidates_norm: torch.Tensor,
) -> torch.Tensor:
    """
    queries: [B,V,D]
    candidates_norm: [B,C,T,D]
    returns: [B,V,C]
    """
    if queries.ndim != 3 or candidates_norm.ndim != 4:
        raise ValueError("Expected queries [B,V,D], candidates [B,C,T,D]")
    q = F.normalize(queries.float(), dim=-1)
    token_scores = torch.einsum("bvd,bctd->bvct", q, candidates_norm)
    return token_scores.amax(dim=-1)


def pairwise_error_from_scores(
    local_scores: torch.Tensor,
    *,
    margin: float,
    tau: float,
) -> torch.Tensor:
    """
    local_scores: [B,V,1+H], positive is candidate 0.
    returns per-negative loss [B,V,H].
    """
    if tau <= 0:
        raise ValueError("pairwise tau must be > 0")
    if local_scores.ndim != 3 or local_scores.shape[-1] < 2:
        raise ValueError("local_scores must be [B,V,1+H]")
    pos = local_scores[:, :, :1]
    neg = local_scores[:, :, 1:]
    return F.softplus((neg - pos + margin) / tau)


def gradient_error_mode_rank(
    source_query: torch.Tensor,
    candidates_norm: torch.Tensor,
    *,
    margin: float,
    tau: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Cheap piecewise-exact gradient-mode diagnostic for the current retrieval head.

    For score(q, image)=max_t q^T c_t, the derivative wrt q is the winning token
    c_{t*} almost everywhere. We form the target-directed negative gradient for
    every hard negative and project it to the tangent plane of the normalized q.

    Returns the participation-ratio effective rank of G for every sample: [B].
    """
    q = F.normalize(source_query.float(), dim=-1)
    if candidates_norm.ndim != 4:
        raise ValueError("candidates_norm must be [B,C,T,D]")

    token_scores = torch.einsum("bd,bctd->bct", q, candidates_norm)
    winning_token = token_scores.argmax(dim=-1)  # [B,C]
    gather_index = winning_token[:, :, None, None].expand(
        -1, -1, 1, candidates_norm.shape[-1]
    )
    chosen = candidates_norm.gather(2, gather_index).squeeze(2)  # [B,C,D]

    image_scores = token_scores.amax(dim=-1)
    pos_score = image_scores[:, :1]
    neg_score = image_scores[:, 1:]
    x = (neg_score - pos_score + margin) / tau
    weight = torch.sigmoid(x) / tau

    pos_vec = chosen[:, :1, :]
    neg_vec = chosen[:, 1:, :]
    target_direction = pos_vec - neg_vec
    tangent = target_direction - (
        target_direction * q[:, None, :]
    ).sum(dim=-1, keepdim=True) * q[:, None, :]
    g = weight[:, :, None] * tangent  # [B,H,D]

    gram = g @ g.transpose(1, 2)
    tr = torch.diagonal(gram, dim1=1, dim2=2).sum(dim=1)
    tr2 = gram.square().sum(dim=(1, 2))
    rank = tr.square() / (tr2 + eps)
    return torch.where(tr > eps, rank, torch.zeros_like(rank))


def effective_rank_rows(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    x: [B,M,D]. Participation-ratio effective rank of row effects.
    """
    gram = x.float() @ x.float().transpose(1, 2)
    tr = torch.diagonal(gram, dim1=1, dim2=2).sum(dim=1)
    tr2 = gram.square().sum(dim=(1, 2))
    rank = tr.square() / (tr2 + eps)
    return torch.where(tr > eps, rank, torch.zeros_like(rank))


def mean_pairwise_row_cosine(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    x: [B,L,D]. Mean cosine only across pairs where both rows have nonzero norm.
    Returns NaN for samples with no valid pair.
    """
    b, l, _ = x.shape
    if l < 2:
        return torch.full((b,), float("nan"), device=x.device)

    norms = x.float().norm(dim=-1)
    z = F.normalize(x.float(), dim=-1, eps=eps)
    sim = z @ z.transpose(1, 2)
    upper = torch.triu(
        torch.ones(l, l, dtype=torch.bool, device=x.device),
        diagonal=1,
    )
    pair_valid = (norms[:, :, None] > eps) & (norms[:, None, :] > eps)
    pair_valid = pair_valid & upper[None, :, :]

    numerator = (sim * pair_valid.to(sim.dtype)).sum(dim=(1, 2))
    denominator = pair_valid.sum(dim=(1, 2))
    result = numerator / denominator.clamp_min(1)
    return torch.where(
        denominator > 0,
        result,
        torch.full_like(result, float("nan")),
    )


def exact_functional_phi(
    coalition_losses: torch.Tensor,
    num_slots: int,
) -> dict[str, torch.Tensor]:
    """
    coalition_losses: [B,2^L,H], lower is better.

    phi(s,j|A) = L_j(A) - L_j(A U {s})
    Positive means slot s reduces that hard-negative error.
    """
    if coalition_losses.ndim != 3:
        raise ValueError("coalition_losses must be [B,2^L,H]")
    expected = 1 << num_slots
    if coalition_losses.shape[1] != expected:
        raise ValueError(f"Expected {expected} coalitions")

    full = expected - 1
    phi_empty = []
    phi_loo = []
    phi_max_conditional = []
    phi_mean_conditional = []

    for s in range(num_slots):
        bit = 1 << s
        phi_empty.append(
            coalition_losses[:, 0, :] - coalition_losses[:, bit, :]
        )
        phi_loo.append(
            coalition_losses[:, full ^ bit, :] - coalition_losses[:, full, :]
        )

        conditional = []
        for mask in range(expected):
            if mask & bit:
                continue
            conditional.append(
                coalition_losses[:, mask, :]
                - coalition_losses[:, mask | bit, :]
            )
        conditional_tensor = torch.stack(conditional, dim=1)  # [B,2^(L-1),H]
        phi_max_conditional.append(conditional_tensor.amax(dim=1))
        phi_mean_conditional.append(conditional_tensor.mean(dim=1))

    return {
        "phi_empty": torch.stack(phi_empty, dim=1),  # [B,L,H]
        "phi_loo": torch.stack(phi_loo, dim=1),
        "phi_max_conditional": torch.stack(phi_max_conditional, dim=1),
        "phi_mean_conditional": torch.stack(phi_mean_conditional, dim=1),
    }


def exact_k_fraction(
    coalition_gain: torch.Tensor,
    num_slots: int,
    fraction: float,
    min_full_gain: float,
) -> torch.Tensor:
    """
    Smallest coalition cardinality reaching fraction * forced-all FULL gain.
    Invalid/negative-FULL samples are NaN.
    """
    full = (1 << num_slots) - 1
    full_gain = coalition_gain[:, full]
    sizes = torch.tensor(
        [int(mask).bit_count() for mask in range(1 << num_slots)],
        device=coalition_gain.device,
        dtype=torch.long,
    )
    target = fraction * full_gain
    qualifies = coalition_gain >= target[:, None]
    qualifies = qualifies & (full_gain[:, None] > min_full_gain)

    huge = torch.full(
        (coalition_gain.shape[0], coalition_gain.shape[1]),
        num_slots + 1,
        dtype=torch.long,
        device=coalition_gain.device,
    )
    candidate_size = torch.where(qualifies, sizes[None, :], huge)
    best = candidate_size.min(dim=1).values.float()
    return torch.where(
        full_gain > min_full_gain,
        best,
        torch.full_like(best, float("nan")),
    )


def safe_recovery_ratio(
    gain: torch.Tensor,
    full_gain: torch.Tensor,
    min_full_gain: float,
) -> torch.Tensor:
    out = torch.full_like(gain, float("nan"), dtype=torch.float32)
    valid = full_gain > min_full_gain
    if gain.ndim == 1:
        out[valid] = gain[valid] / full_gain[valid]
    elif gain.ndim == 2:
        out[valid] = gain[valid] / full_gain[valid, None]
    else:
        raise ValueError("gain must be [B] or [B,K]")
    return out


def build_target_indices(batch, gallery_index: dict[str, int], device):
    target_ids = list(batch.target_ids)
    if any(target_id is None for target_id in target_ids):
        raise ValueError("Validation sample is missing target_id")
    missing = [target_id for target_id in target_ids if target_id not in gallery_index]
    if missing:
        raise KeyError(f"Targets missing from gallery, first few: {missing[:5]}")
    return torch.tensor(
        [gallery_index[target_id] for target_id in target_ids],
        device=device,
        dtype=torch.long,
    )


def prepare_batch(runtime, batch):
    device = runtime["device"]
    reference_native = get_features_by_ids(
        batch.reference_ids,
        runtime["val_native"],
        runtime["val_native_idx"],
    ).to(device=device, dtype=torch.float32)
    reference_features = reference_native[:, 0, :]

    text_states, teacher_text_states, attention_mask, content_mask = (
        get_text_features_by_sample_ids(
            batch.sample_ids,
            batch.modification_texts,
            runtime["val_text"],
        )
    )

    return {
        "reference_native": reference_native,
        "reference_features": reference_features,
        "text_states": text_states.to(device=device, dtype=torch.float32),
        "teacher_text_states": teacher_text_states.to(
            device=device, dtype=torch.float32
        ),
        "attention_mask": attention_mask.to(device=device, dtype=torch.bool),
        "content_mask": content_mask.to(device=device, dtype=torch.bool),
    }


def load_runtime(args):
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.max_queries_per_category < 0:
        raise ValueError("--max-queries-per-category must be >= 0")
    if args.hard_negatives < 1:
        raise ValueError("--hard-negatives must be >= 1")
    if args.phi_positive_threshold < 0:
        raise ValueError("--phi-positive-threshold must be >= 0")
    if args.min_full_gain < 0:
        raise ValueError("--min-full-gain must be >= 0")

    device = torch.device(args.device)
    cfg = OmegaConf.load(args.config)
    if getattr(args, "slot_value_source", None) is not None:
        cfg.model.slot_value_source = args.slot_value_source
    if getattr(args, "slot_effect_in_value", None) is not None:
        cfg.model.slot_effect_in_value = args.slot_effect_in_value == "true"
    if args.slot_value_assignment is not None:
        cfg.model.slot_value_assignment = args.slot_value_assignment
    if args.functional_ownership_enabled is not None:
        cfg.functional_ownership.enabled = (
            args.functional_ownership_enabled == "true"
        )
    annotation_root = args.dataset_root / "captions"
    split_root = args.dataset_root / "image_splits"

    correction_dicts = load_correction_dicts(annotation_root)
    val_loaders = build_val_loaders(
        annotation_root=annotation_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        caption_policy=cfg.val_caption_policy,
        correction_dicts=correction_dicts,
    )

    feature_root = args.cache_root / "fashioniq" / "csmcir" / "val"
    val_retrieval, val_retrieval_idx = load_features(feature_root / "retrieval")
    val_native, val_native_idx = load_features(feature_root / "native")
    val_text = load_text_features(feature_root / "text")

    model = build_model(cfg, device)
    load_checkpoint(model, args.checkpoint)
    model.eval()

    pairwise_tau = (
        float(args.pairwise_tau)
        if args.pairwise_tau is not None
        else float(model.retrieval_temperature)
    )
    if pairwise_tau <= 0:
        raise ValueError("pairwise tau must be > 0")

    return {
        "device": device,
        "cfg": cfg,
        "split_root": split_root,
        "val_loaders": val_loaders,
        "val_retrieval": val_retrieval,
        "val_retrieval_idx": val_retrieval_idx,
        "val_native": val_native,
        "val_native_idx": val_native_idx,
        "val_text": val_text,
        "model": model,
        "pairwise_tau": pairwise_tau,
    }


def selected_precision_recall(
    selected: torch.Tensor,
    useful: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    intersect = (selected & useful).sum(dim=1).float()
    selected_count = selected.sum(dim=1).float()
    useful_count = useful.sum(dim=1).float()

    precision = torch.where(
        selected_count > 0,
        intersect / selected_count.clamp_min(1.0),
        torch.full_like(intersect, float("nan")),
    )
    recall = torch.where(
        useful_count > 0,
        intersect / useful_count.clamp_min(1.0),
        torch.full_like(intersect, float("nan")),
    )
    return precision, recall


def add_example_rows(
    examples: list[dict],
    *,
    limit: int,
    batch,
    gallery_ids: list[str],
    negative_indices: torch.Tensor,
    qasa_selected: torch.Tensor,
    hard_effective_k: torch.Tensor,
    hard_winner_counts: torch.Tensor,
    g_rank: torch.Tensor,
    phi_rank: torch.Tensor,
    useful_slots: torch.Tensor,
    phi_max_conditional: torch.Tensor,
    coalition_gain: torch.Tensor,
    repeat_gain: torch.Tensor,
    mean_gain: torch.Tensor,
    qasa_gain: torch.Tensor,
    full_gain: torch.Tensor,
    k95: torch.Tensor,
    k99: torch.Tensor,
):
    if len(examples) >= limit:
        return
    b = negative_indices.shape[0]
    l = qasa_selected.shape[1]
    full = (1 << l) - 1
    remaining = limit - len(examples)

    for i in range(min(b, remaining)):
        fg = float(full_gain[i].item())
        denom_ok = math.isfinite(fg) and fg > 1e-12

        coalition_row = {
            coalition_name(mask, l): float(coalition_gain[i, mask].item())
            for mask in range(1 << l)
        }
        examples.append(
            {
                "sample_id": str(batch.sample_ids[i]),
                "modification_text": str(batch.modification_texts[i]),
                "target_id": str(batch.target_ids[i]),
                "hard_negative_ids": [
                    gallery_ids[int(idx)]
                    for idx in negative_indices[i].detach().cpu().tolist()
                ],
                "qasa_selected_slots": [
                    s for s in range(l) if bool(qasa_selected[i, s].item())
                ],
                "hard_partition_effective_k": int(hard_effective_k[i].item()),
                "hard_partition_winner_counts": [
                    int(x) for x in hard_winner_counts[i].detach().cpu().tolist()
                ],
                "gradient_error_mode_effective_rank": float(g_rank[i].item()),
                "functional_phi_effective_rank": float(phi_rank[i].item()),
                "functionally_useful_slots": [
                    s for s in range(l) if bool(useful_slots[i, s].item())
                ],
                "per_slot_max_conditional_phi_mean": [
                    float(x)
                    for x in phi_max_conditional[i].mean(dim=-1).detach().cpu().tolist()
                ],
                "forced_all_full_gain": fg,
                "best_repeat_recovery_ratio": (
                    float((repeat_gain[i].max() / full_gain[i]).item())
                    if denom_ok
                    else None
                ),
                "mean_xK_recovery_ratio": (
                    float((mean_gain[i] / full_gain[i]).item())
                    if denom_ok
                    else None
                ),
                "qasa_full_recovery_ratio": (
                    float((qasa_gain[i] / full_gain[i]).item())
                    if denom_ok
                    else None
                ),
                "k95": (
                    int(k95[i].item()) if torch.isfinite(k95[i]) else None
                ),
                "k99": (
                    int(k99[i].item()) if torch.isfinite(k99[i]) else None
                ),
                "coalition_gain": coalition_row,
                "forced_all_mask": coalition_name(full, l),
            }
        )


@torch.no_grad()
def run(args):
    runtime = load_runtime(args)
    model = runtime["model"]
    device = runtime["device"]
    tau = runtime["pairwise_tau"]
    num_slots = model.num_slots

    if num_slots > 8:
        raise ValueError(
            "Exact 2^K audit is intentionally limited to K<=8; "
            f"current num_slots={num_slots}"
        )

    masks = coalition_masks(num_slots, device)
    full_mask_id = (1 << num_slots) - 1
    full_mask = masks[full_mask_id]
    stats = ScalarCollector()
    recall = RecallCollector()
    examples: list[dict] = []
    total_samples = 0

    for category in CATEGORIES:
        loader = runtime["val_loaders"][category]
        annotations = getattr(loader.dataset, "annotations", None)
        if annotations is None:
            raise AttributeError(
                "FashionIQDataset must expose .annotations for gallery construction"
            )

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

        if len(gallery_ids) <= args.hard_negatives:
            raise ValueError(
                f"Gallery has {len(gallery_ids)} images, cannot mine "
                f"{args.hard_negatives} negatives after excluding target."
            )

        processed = 0
        progress = tqdm(
            loader,
            desc=f"MERIT P0 [{category}]",
            dynamic_ncols=True,
        )

        for batch in progress:
            if (
                args.max_queries_per_category
                and processed >= args.max_queries_per_category
            ):
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
            edit_slots = slot_output["edit_slots"]
            z0, reference_state = model.initialize_state(x["reference_features"])
            q_reference = model.make_query(z0)

            # Exact 2^K forced coalitions. These deliberately bypass QASA selection
            # so every learned candidate can be causally tested.
            coalition_queries = execute_same_slots_variants(
                model,
                edit_slots=edit_slots,
                masks=masks,
                z0=z0,
                reference_state=reference_state,
            )  # [B,2^K,D]
            q_all_full = coalition_queries[:, full_mask_id, :]
            q_empty = coalition_queries[:, 0, :]

            # Deployed path: same learned slots, current QASA selected subset.
            qasa_execution = model.execute(
                edit_slots,
                slot_output["qasa_selected_mask"],
                z0,
                reference_state,
            )
            q_qasa = model.make_query(qasa_execution["final_state"])

            # Direct clone adversary: copy each original slot into all L positions,
            # keep all L execution tickets.
            repeated_slots = (
                edit_slots[:, :, None, :]
                .expand(b, num_slots, num_slots, edit_slots.shape[-1])
            )
            repeated_masks = torch.ones(
                b,
                num_slots,
                num_slots,
                dtype=torch.bool,
                device=device,
            )
            repeat_queries = execute_explicit_variants(
                model,
                variant_slots=repeated_slots,
                variant_masks=repeated_masks,
                z0=z0,
                reference_state=reference_state,
            )  # [B,L,D]

            # MEAN x K adversary: destroy slot identity/content differences while
            # keeping K parallel slot positions/tickets.
            mean_slot = edit_slots.mean(dim=1, keepdim=True).expand_as(edit_slots)
            mean_query = execute_explicit_variants(
                model,
                variant_slots=mean_slot[:, None, :, :],
                variant_masks=torch.ones(
                    b, 1, num_slots, dtype=torch.bool, device=device
                ),
                z0=z0,
                reference_state=reference_state,
            )[:, 0, :]

            # Empty coalition must be an exact executor no-op relative to z0 query.
            stats.add(
                "smoke/empty_query_max_abs_diff_from_reference",
                (q_empty - q_reference).abs().amax(dim=1),
            )

            # Full-gallery retrieval is intentionally limited to three headline
            # variants; all expensive causal variants use the same small fixed
            # hard-negative bank below.
            gallery_scores = {
                "qasa_full": score_gallery(q_qasa, gallery_norm),
                "all_slots_full": score_gallery(q_all_full, gallery_norm),
                "reference_only": score_gallery(q_reference, gallery_norm),
            }
            for variant, scores in gallery_scores.items():
                recall.update(variant, category, scores, target_indices)

            source_query = {
                "qasa_full": q_qasa,
                "all_slots_full": q_all_full,
                "reference_only": q_reference,
            }[args.negative_source]
            source_scores = gallery_scores[args.negative_source].clone()
            source_scores.scatter_(
                1,
                target_indices[:, None],
                float("-inf"),
            )
            negative_indices = source_scores.topk(
                args.hard_negatives, dim=1
            ).indices

            candidate_indices = torch.cat(
                [target_indices[:, None], negative_indices],
                dim=1,
            )
            candidates_norm = gallery_norm[candidate_indices]  # [B,1+H,T,D]

            # Per-negative error-mode rank from the fixed negative bank.
            g_rank = gradient_error_mode_rank(
                source_query,
                candidates_norm,
                margin=args.pairwise_margin,
                tau=tau,
            )
            stats.add("error_modes/gradient_effective_rank", g_rank)
            stats.add("error_modes/rank_gt_1_2", (g_rank > 1.2).float())
            stats.add("error_modes/rank_gt_1_5", (g_rank > 1.5).float())
            stats.add("error_modes/rank_gt_2_0", (g_rank > 2.0).float())

            # Exact finite intervention table Phi over every coalition.
            coalition_scores = score_local_candidates(
                coalition_queries,
                candidates_norm,
            )
            coalition_losses = pairwise_error_from_scores(
                coalition_scores,
                margin=args.pairwise_margin,
                tau=tau,
            )  # [B,2^K,H]

            phi = exact_functional_phi(coalition_losses, num_slots)
            phi_max = phi["phi_max_conditional"]
            phi_positive = phi_max.clamp_min(0.0)
            phi_rank = effective_rank_rows(phi_positive)
            phi_cosine = mean_pairwise_row_cosine(phi_positive)
            stats.add("functional/phi_positive_effective_rank", phi_rank)
            stats.add("functional/phi_positive_pairwise_cosine", phi_cosine)

            useful_slots = (
                phi_max.amax(dim=-1) > args.phi_positive_threshold
            )
            stats.add(
                "functional/useful_slot_count",
                useful_slots.sum(dim=1).float(),
            )

            owner_strength, owner_slot = phi_positive.max(dim=1)  # [B,H]
            covered_mode = owner_strength > args.phi_positive_threshold
            owner_1h = F.one_hot(
                owner_slot,
                num_classes=num_slots,
            ).to(torch.bool)
            owner_1h = owner_1h & covered_mode[:, :, None]
            unique_owner_count = owner_1h.any(dim=1).sum(dim=1).float()
            stats.add(
                "functional/unique_positive_mode_owner_count",
                unique_owner_count,
            )

            qasa_selected = slot_output["qasa_selected_mask"]
            qasa_precision, qasa_recall = selected_precision_recall(
                qasa_selected,
                useful_slots,
            )
            stats.add("qasa_vs_function/qasa_selected_count", qasa_selected.sum(dim=1))
            stats.add("qasa_vs_function/precision", qasa_precision)
            stats.add("qasa_vs_function/recall", qasa_recall)

            # Existing QASA hard-partition diagnostics, kept as health metrics only.
            hard_k = slot_output["qasa_inference_effective_k"].float()
            winner_counts = slot_output["qasa_inference_winner_counts"].float()
            valid_count = slot_output["qasa_valid_mask"].sum(dim=1).clamp_min(1)
            dominant_hard_share = (
                winner_counts.max(dim=1).values / valid_count.float()
            )
            stats.add("routing/hard_partition_effective_k", hard_k)
            stats.add("routing/dominant_hard_token_share", dominant_hard_share)

            # Coalition utility: U gain relative to EMPTY/reference on the same
            # per-sample fixed hard-negative bank.
            coalition_mean_loss = coalition_losses.mean(dim=-1)
            empty_loss = coalition_mean_loss[:, 0]
            coalition_gain = empty_loss[:, None] - coalition_mean_loss
            full_gain = coalition_gain[:, full_mask_id]
            stats.add("utility/forced_all_full_gain", full_gain)
            stats.add(
                "utility/positive_forced_all_full_gain",
                (full_gain > args.min_full_gain).float(),
            )

            single_gain = torch.stack(
                [coalition_gain[:, 1 << s] for s in range(num_slots)],
                dim=1,
            )
            drop_gain = torch.stack(
                [
                    coalition_gain[:, full_mask_id ^ (1 << s)]
                    for s in range(num_slots)
                ],
                dim=1,
            )
            loo_marginal = full_gain[:, None] - drop_gain

            repeat_scores = score_local_candidates(
                repeat_queries,
                candidates_norm,
            )
            repeat_losses = pairwise_error_from_scores(
                repeat_scores,
                margin=args.pairwise_margin,
                tau=tau,
            ).mean(dim=-1)
            repeat_gain = empty_loss[:, None] - repeat_losses

            mean_scores = score_local_candidates(
                mean_query[:, None, :],
                candidates_norm,
            )
            mean_loss = pairwise_error_from_scores(
                mean_scores,
                margin=args.pairwise_margin,
                tau=tau,
            )[:, 0, :].mean(dim=-1)
            mean_gain = empty_loss - mean_loss

            qasa_scores_local = score_local_candidates(
                q_qasa[:, None, :],
                candidates_norm,
            )
            qasa_loss = pairwise_error_from_scores(
                qasa_scores_local,
                margin=args.pairwise_margin,
                tau=tau,
            )[:, 0, :].mean(dim=-1)
            qasa_gain = empty_loss - qasa_loss

            single_ratio = safe_recovery_ratio(
                single_gain,
                full_gain,
                args.min_full_gain,
            )
            repeat_ratio = safe_recovery_ratio(
                repeat_gain,
                full_gain,
                args.min_full_gain,
            )
            mean_ratio = safe_recovery_ratio(
                mean_gain,
                full_gain,
                args.min_full_gain,
            )
            qasa_ratio = safe_recovery_ratio(
                qasa_gain,
                full_gain,
                args.min_full_gain,
            )

            stats.add(
                "collapse/best_single_recovery_ratio",
                single_ratio.amax(dim=1),
            )
            stats.add(
                "collapse/best_repeat_xK_recovery_ratio",
                repeat_ratio.amax(dim=1),
            )
            stats.add("collapse/mean_xK_recovery_ratio", mean_ratio)
            stats.add("collapse/qasa_full_recovery_ratio", qasa_ratio)

            k95 = exact_k_fraction(
                coalition_gain,
                num_slots,
                0.95,
                args.min_full_gain,
            )
            k99 = exact_k_fraction(
                coalition_gain,
                num_slots,
                0.99,
                args.min_full_gain,
            )
            stats.add("coalition/k95", k95)
            stats.add("coalition/k99", k99)

            for s in range(num_slots):
                stats.add(f"slot/{s}/singleton_gain", single_gain[:, s])
                stats.add(f"slot/{s}/loo_marginal", loo_marginal[:, s])
                stats.add(f"slot/{s}/repeat_xK_gain", repeat_gain[:, s])
                stats.add(
                    f"slot/{s}/phi_empty_mean",
                    phi["phi_empty"][:, s, :].mean(dim=-1),
                )
                stats.add(
                    f"slot/{s}/phi_loo_mean",
                    phi["phi_loo"][:, s, :].mean(dim=-1),
                )
                stats.add(
                    f"slot/{s}/phi_max_conditional_mean",
                    phi_max[:, s, :].mean(dim=-1),
                )
                stats.add(
                    f"slot/{s}/useful_fraction",
                    useful_slots[:, s].float(),
                )

            add_example_rows(
                examples,
                limit=args.num_examples,
                batch=batch,
                gallery_ids=gallery_ids,
                negative_indices=negative_indices,
                qasa_selected=qasa_selected,
                hard_effective_k=slot_output["qasa_inference_effective_k"],
                hard_winner_counts=slot_output["qasa_inference_winner_counts"],
                g_rank=g_rank,
                phi_rank=phi_rank,
                useful_slots=useful_slots,
                phi_max_conditional=phi_max,
                coalition_gain=coalition_gain,
                repeat_gain=repeat_gain,
                mean_gain=mean_gain,
                qasa_gain=qasa_gain,
                full_gain=full_gain,
                k95=k95,
                k99=k99,
            )

            total_samples += b
            processed += b
            progress.set_postfix(
                n=processed,
                hardK=f"{hard_k.mean().item():.2f}",
                gRank=f"{g_rank.mean().item():.2f}",
                phiRank=f"{phi_rank.mean().item():.2f}",
            )

    scalar_summary = stats.finalize()
    recall_summary = recall.finalize()

    report = {
        "checkpoint": str(args.checkpoint),
        "experiment_provenance": model.experiment_provenance(),
        "num_samples": total_samples,
        "num_slots": num_slots,
        "protocol": {
            "dataset_protocol": args.protocol,
            "negative_source": args.negative_source,
            "hard_negatives": args.hard_negatives,
            "pairwise_margin": args.pairwise_margin,
            "pairwise_tau": tau,
            "phi_positive_threshold": args.phi_positive_threshold,
            "min_full_gain_for_ratios": args.min_full_gain,
            "important_semantics": [
                "qasa_full uses the checkpoint's real qasa_selected_mask.",
                "all_slots_full and the 2^K coalitions force slot activation and bypass QASA selection so every candidate can be causally tested.",
                "Hard negatives are mined once per sample from negative_source and frozen for every intervention.",
                "Target image is used only by this audit as the positive retrieval candidate; no semantic slot labels are required.",
                "REPEAT copies one original slot into all K positions; MEAN copies the per-sample mean slot into all K positions.",
                "Hard-partition metrics are diagnostic argmax partitions; under soft_shared they are not the actual soft VALUE support.",
                "The script is frozen-checkpoint evaluation only and does not modify training.",
            ],
        },
        "recall": recall_summary,
        "stats": scalar_summary,
        "examples": examples,
    }

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    def mean_of(name: str) -> float:
        item = scalar_summary.get(name)
        return float("nan") if item is None else float(item["mean"])

    def median_of(name: str) -> float:
        item = scalar_summary.get(name)
        return float("nan") if item is None else float(item["median"])

    print()
    print("=" * 78)
    print("TAPER-MERIT P0 FUNCTIONAL INTERVENTION AUDIT")
    print("=" * 78)
    print(f"Samples:                         {total_samples}")
    print(
        "Hard partition mean K:           "
        f"{mean_of('routing/hard_partition_effective_k'):.3f}"
    )
    print(
        "Dominant hard token share:       "
        f"{mean_of('routing/dominant_hard_token_share'):.3f}"
    )
    print(
        "Gradient error-mode rank:        "
        f"{mean_of('error_modes/gradient_effective_rank'):.3f}"
    )
    print(
        "Functional Phi effective rank:   "
        f"{mean_of('functional/phi_positive_effective_rank'):.3f}"
    )
    print(
        "Functionally useful slots:       "
        f"{mean_of('functional/useful_slot_count'):.3f}"
    )
    print(
        "QASA functional precision/recall:"
        f" {mean_of('qasa_vs_function/precision'):.3f}"
        f" / {mean_of('qasa_vs_function/recall'):.3f}"
    )
    print(
        "Median best SINGLE/FULL:         "
        f"{median_of('collapse/best_single_recovery_ratio'):.3f}"
    )
    print(
        "Median best REPEAT/FULL:         "
        f"{median_of('collapse/best_repeat_xK_recovery_ratio'):.3f}"
    )
    print(
        "Median MEANxK/FULL:              "
        f"{median_of('collapse/mean_xK_recovery_ratio'):.3f}"
    )
    print(
        "Mean K95 / K99:                  "
        f"{mean_of('coalition/k95'):.3f}"
        f" / {mean_of('coalition/k99'):.3f}"
    )

    for variant in ("qasa_full", "all_slots_full", "reference_only"):
        r = recall_summary.get(variant)
        if r:
            print(
                f"{variant:30s}"
                f" R@10={r['recall_at_10']:.2f}"
                f" R@50={r['recall_at_50']:.2f}"
                f" Mean={r['mean_recall']:.2f}"
            )
    print(f"Report: {args.json_output}")


if __name__ == "__main__":
    run(parse_args())
