from __future__ import annotations

import hashlib
import itertools
from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader


# ============================================================
# Basic helpers
# ============================================================


def _mean(x: Tensor) -> float:
    if x.numel() == 0:
        return 0.0
    return float(x.float().mean().item())


def _median(x: Tensor) -> float:
    if x.numel() == 0:
        return 0.0
    return float(x.float().median().item())


def _max(x: Tensor) -> float:
    if x.numel() == 0:
        return 0.0
    return float(x.float().max().item())


def _quantile(x: Tensor, q: float) -> float:
    if x.numel() == 0:
        return 0.0
    return float(torch.quantile(x.float(), q).item())


def _cat(values: list[Tensor]) -> Tensor:
    if not values:
        return torch.empty(0, dtype=torch.float32)

    return torch.cat(values, dim=0)


def _require_config(
    config: Mapping[str, object],
    key: str,
):
    if key not in config:
        raise KeyError(f"Stage-1 evaluation config is missing required key: {key}")

    return config[key]


# ============================================================
# Teacher gallery
# ============================================================


def _validate_teacher_gallery(
    teacher_gallery_features: Tensor,
    teacher_gallery_name_to_idx: Mapping[str, int],
) -> None:
    """
    Current Stage-1 V0 assumes one teacher-native vector per gallery image:

        gallery: [G, D_teacher]

    Do NOT silently reuse FG-CLIP/TAPER features unless they really are
    the selected teacher's native retrieval gallery representation.
    """

    if teacher_gallery_features.ndim != 2:
        raise ValueError(
            "teacher_gallery_features must be [G,D]. "
            "Current Stage-1 evaluator assumes one teacher-native "
            "gallery vector per image."
        )

    if teacher_gallery_features.shape[0] != len(teacher_gallery_name_to_idx):
        raise ValueError("teacher gallery feature count does not match ID mapping")

    if not torch.isfinite(teacher_gallery_features).all():
        raise ValueError("teacher_gallery_features contains NaN or Inf")

    indices = list(teacher_gallery_name_to_idx.values())

    if len(set(indices)) != len(indices):
        raise ValueError("teacher gallery name_to_idx contains duplicate indices")

    if min(indices, default=0) < 0:
        raise ValueError("teacher gallery mapping contains negative index")

    if max(indices, default=-1) >= teacher_gallery_features.shape[0]:
        raise ValueError("teacher gallery mapping points outside feature tensor")


def _get_category_gallery(
    *,
    category: str,
    gallery_ids_by_category: Mapping[str, Sequence[str]],
    teacher_gallery_features: Tensor,
    teacher_gallery_name_to_idx: Mapping[str, int],
    device: torch.device,
) -> tuple[list[str], Tensor, dict[str, int]]:
    if category not in gallery_ids_by_category:
        raise KeyError(f"Missing teacher gallery IDs for category {category!r}")

    gallery_ids = list(gallery_ids_by_category[category])

    if not gallery_ids:
        raise ValueError(f"Teacher gallery for {category!r} is empty")

    if len(set(gallery_ids)) != len(gallery_ids):
        raise ValueError(f"Teacher gallery for {category!r} contains duplicate IDs")

    missing = [image_id for image_id in gallery_ids if image_id not in teacher_gallery_name_to_idx]

    if missing:
        raise KeyError(f"Teacher gallery cache is missing images for {category!r}: {missing[:5]}")

    global_indices = [teacher_gallery_name_to_idx[image_id] for image_id in gallery_ids]

    category_gallery = teacher_gallery_features[global_indices].to(
        device=device,
        non_blocking=True,
    )

    local_name_to_idx = {image_id: index for index, image_id in enumerate(gallery_ids)}

    return (
        gallery_ids,
        category_gallery,
        local_name_to_idx,
    )


# ============================================================
# Teacher scoring
# ============================================================


def _cosine_scores(
    queries: Tensor,
    gallery: Tensor,
) -> Tensor:
    if queries.ndim != 2:
        raise ValueError("teacher queries must be [Q,D]")

    if gallery.ndim != 2:
        raise ValueError("teacher gallery must be [G,D]")

    if queries.shape[-1] != gallery.shape[-1]:
        raise ValueError(
            "Teacher query/gallery dimension mismatch. "
            "Do not auto-project between unrelated spaces."
        )

    queries = F.normalize(
        queries.float(),
        dim=-1,
    )

    gallery = F.normalize(
        gallery.float(),
        dim=-1,
    )

    scores = queries @ gallery.t()

    if not torch.isfinite(scores).all():
        raise ValueError("Teacher retrieval scores contain NaN or Inf")

    return scores


def _score_in_chunks(
    queries: Tensor,
    gallery: Tensor,
    *,
    chunk_size: int,
) -> Tensor:
    if chunk_size < 1:
        raise ValueError("score_query_chunk_size must be >= 1")

    chunks = []

    for start in range(
        0,
        queries.shape[0],
        chunk_size,
    ):
        end = start + chunk_size

        chunks.append(
            _cosine_scores(
                queries[start:end],
                gallery,
            )
        )

    if not chunks:
        return torch.empty(
            0,
            gallery.shape[0],
            device=gallery.device,
        )

    return torch.cat(
        chunks,
        dim=0,
    )


# ============================================================
# Candidate policy
# ============================================================


def _ids_to_indices(
    image_ids: Sequence[str],
    name_to_idx: Mapping[str, int],
    *,
    device: torch.device,
    field_name: str,
) -> Tensor:
    missing = [image_id for image_id in image_ids if image_id not in name_to_idx]

    if missing:
        raise KeyError(f"{field_name} missing from category teacher gallery: {missing[:5]}")

    return torch.tensor(
        [name_to_idx[image_id] for image_id in image_ids],
        dtype=torch.long,
        device=device,
    )


def _known_positive_exclusions(
    *,
    ground_truth_ids: Sequence[tuple[str, ...]],
    target_ids: Sequence[str],
    local_name_to_idx: Mapping[str, int],
) -> list[list[int]]:
    """
    FashionIQ normally has no extra positives here.

    CIRR can later use ground_truth_ids without changing TCFR arithmetic.
    """

    if len(ground_truth_ids) != len(target_ids):
        raise ValueError("ground_truth_ids and target_ids size mismatch")

    result: list[list[int]] = []

    for positives, target_id in zip(
        ground_truth_ids,
        target_ids,
    ):
        excluded = []

        for positive_id in positives:
            if positive_id == target_id:
                continue

            if positive_id in local_name_to_idx:
                excluded.append(local_name_to_idx[positive_id])

        result.append(excluded)

    return result


def _apply_candidate_exclusions(
    scores: Tensor,
    *,
    reference_indices: Tensor | None,
    extra_excluded_indices: Sequence[Sequence[int]] | None,
) -> Tensor:
    if scores.ndim != 2:
        raise ValueError("scores must be [B,G]")

    scores = scores.clone()

    batch_size = scores.shape[0]
    batch_ids = torch.arange(
        batch_size,
        device=scores.device,
    )

    if reference_indices is not None:
        if reference_indices.shape != (batch_size,):
            raise ValueError("reference_indices must be [B]")

        scores[
            batch_ids,
            reference_indices,
        ] = -torch.inf

    if extra_excluded_indices is not None:
        if len(extra_excluded_indices) != batch_size:
            raise ValueError("extra_excluded_indices must contain one list per query")

        for row, excluded in enumerate(extra_excluded_indices):
            if not excluded:
                continue

            index_tensor = torch.tensor(
                list(excluded),
                dtype=torch.long,
                device=scores.device,
            )

            scores[
                row,
                index_tensor,
            ] = -torch.inf

    return scores


# ============================================================
# Rank
# ============================================================


def compute_target_ranks(
    scores: Tensor,
    target_indices: Tensor,
    *,
    reference_indices: Tensor | None = None,
    extra_excluded_indices: Sequence[Sequence[int]] | None = None,
) -> Tensor:
    """
    Return 1-based target rank.

    rank = 1 means target is the highest-scoring valid gallery item.
    """

    if scores.ndim != 2:
        raise ValueError("scores must be [B,G]")

    batch_size = scores.shape[0]

    if target_indices.shape != (batch_size,):
        raise ValueError("target_indices must be [B]")

    candidate_scores = _apply_candidate_exclusions(
        scores,
        reference_indices=reference_indices,
        extra_excluded_indices=extra_excluded_indices,
    )

    batch_ids = torch.arange(
        batch_size,
        device=scores.device,
    )

    target_scores = candidate_scores[
        batch_ids,
        target_indices,
    ]

    if not torch.isfinite(target_scores).all():
        raise ValueError("Candidate policy removed the positive target")

    # 1-based rank.
    ranks = 1 + (candidate_scores > target_scores[:, None]).sum(dim=1)

    return ranks


# ============================================================
# Fixed hard negatives
# ============================================================


def build_fixed_hard_negatives(
    full_scores: Tensor,
    target_indices: Tensor,
    *,
    hard_negative_k: int,
    reference_indices: Tensor | None = None,
    extra_excluded_indices: Sequence[Sequence[int]] | None = None,
) -> Tensor:
    """
    Negatives are mined ONLY from q_full.

    The returned negative set must then remain fixed when q_minus changes.
    """

    if hard_negative_k < 1:
        raise ValueError("hard_negative_k must be >= 1")

    candidate_scores = _apply_candidate_exclusions(
        full_scores,
        reference_indices=reference_indices,
        extra_excluded_indices=extra_excluded_indices,
    )

    batch_size = candidate_scores.shape[0]

    batch_ids = torch.arange(
        batch_size,
        device=candidate_scores.device,
    )

    # Positive target can never be a negative.
    candidate_scores[
        batch_ids,
        target_indices,
    ] = -torch.inf

    valid_negative_count = torch.isfinite(candidate_scores).sum(dim=1)

    if (valid_negative_count < hard_negative_k).any():
        raise ValueError(
            f"Not enough valid gallery items to build {hard_negative_k} fixed hard negatives"
        )

    return candidate_scores.topk(
        k=hard_negative_k,
        dim=1,
    ).indices


# ============================================================
# TCFR arithmetic
# ============================================================


def compute_target_margin(
    *,
    scores: Tensor,
    target_indices: Tensor,
    hard_negative_indices: Tensor,
) -> Tensor:
    """
    m(q) = target_similarity - mean(fixed_negative_similarity)

    scores:
        [B,G]

    target_indices:
        [B]

    hard_negative_indices:
        [B,H]

    returns:
        [B]
    """

    if scores.ndim != 2:
        raise ValueError("scores must be [B,G]")

    batch_size = scores.shape[0]

    if target_indices.shape != (batch_size,):
        raise ValueError("target_indices must be [B]")

    if hard_negative_indices.ndim != 2 or hard_negative_indices.shape[0] != batch_size:
        raise ValueError("hard_negative_indices must be [B,H]")

    batch_ids = torch.arange(
        batch_size,
        device=scores.device,
    )

    target_similarity = scores[
        batch_ids,
        target_indices,
    ]

    negative_similarity = scores.gather(
        dim=1,
        index=hard_negative_indices,
    ).mean(dim=1)

    return target_similarity - negative_similarity


def compute_slot_target_margin(
    *,
    scores: Tensor,
    target_indices: Tensor,
    hard_negative_indices: Tensor,
) -> Tensor:
    """
    scores:
        [B,L,G]

    returns:
        [B,L]
    """

    if scores.ndim != 3:
        raise ValueError("slot scores must be [B,L,G]")

    batch_size, num_slots, _ = scores.shape

    if target_indices.shape != (batch_size,):
        raise ValueError("target_indices must be [B]")

    if hard_negative_indices.ndim != 2 or hard_negative_indices.shape[0] != batch_size:
        raise ValueError("hard_negative_indices must be [B,H]")

    target_grid = target_indices[
        :,
        None,
        None,
    ].expand(
        batch_size,
        num_slots,
        1,
    )

    target_similarity = scores.gather(
        dim=2,
        index=target_grid,
    ).squeeze(2)

    negative_grid = hard_negative_indices[
        :,
        None,
        :,
    ].expand(
        batch_size,
        num_slots,
        -1,
    )

    negative_similarity = scores.gather(
        dim=2,
        index=negative_grid,
    ).mean(dim=2)

    return target_similarity - negative_similarity


def compute_tcfr(
    *,
    full_margin: Tensor,
    minus_margin: Tensor,
    slot_gates: Tensor,
    gate_threshold: float,
) -> dict[str, Tensor]:
    """
    full_margin:
        [B]

    minus_margin:
        [B,L]

    slot_gates:
        [B,L]
    """

    if full_margin.ndim != 1:
        raise ValueError("full_margin must be [B]")

    if minus_margin.ndim != 2:
        raise ValueError("minus_margin must be [B,L]")

    if minus_margin.shape[0] != full_margin.shape[0]:
        raise ValueError("full/minus margin batch mismatch")

    if slot_gates.shape != minus_margin.shape:
        raise ValueError("slot_gates must match minus_margin [B,L]")

    margin_drop = full_margin[:, None] - minus_margin

    active = slot_gates >= gate_threshold

    active_count = active.sum(dim=1)

    hard_tcfr = torch.where(
        active_count > 0,
        (margin_drop * active.to(margin_drop.dtype)).sum(dim=1)
        / active_count.clamp_min(1).to(margin_drop.dtype),
        # IMPORTANT:
        # zero-active samples contribute ZERO.
        torch.zeros_like(full_margin),
    )

    soft_tcfr = (margin_drop * slot_gates).sum(dim=1) / slot_gates.sum(dim=1).clamp_min(1e-8)

    return {
        "margin_drop": margin_drop,
        "active_mask": active,
        "active_count": active_count,
        "per_sample_tcfr": hard_tcfr,
        "per_sample_tcfr_soft": soft_tcfr,
    }


# ============================================================
# Stage-1 structural health
# ============================================================


def _active_pairwise_cosines(
    values: Tensor,
    active: Tensor,
) -> Tensor:
    """
    values:
        [B,L,D]

    active:
        [B,L]

    Return all upper-triangle active-active pair cosines.
    """

    if values.ndim != 3:
        raise ValueError("values must be [B,L,D]")

    if active.shape != values.shape[:2]:
        raise ValueError("active must match values [B,L]")

    values = F.normalize(
        values.float(),
        dim=-1,
        eps=1e-8,
    )

    pairwise = torch.einsum(
        "bld,bmd->blm",
        values,
        values,
    )

    num_slots = values.shape[1]

    upper_triangle = torch.triu(
        torch.ones(
            num_slots,
            num_slots,
            dtype=torch.bool,
            device=values.device,
        ),
        diagonal=1,
    )

    valid_pairs = active[:, :, None] & active[:, None, :] & upper_triangle[None]

    return pairwise[valid_pairs]


def compute_structural_health(
    *,
    slot_masks: Tensor,
    slot_effects: Tensor,
    slot_gates: Tensor,
    text_attention_mask: Tensor,
    gate_threshold: float,
    text_content_mask: Tensor | None = None,
) -> dict[str, Tensor]:
    if slot_masks.ndim != 3:
        raise ValueError("slot_masks must be [B,L,N]")

    if slot_effects.ndim != 3:
        raise ValueError("slot_effects must be [B,L,D]")

    if slot_effects.shape[:2] != slot_masks.shape[:2]:
        raise ValueError("slot_effects and slot_masks mismatch")

    if slot_gates.shape != slot_masks.shape[:2]:
        raise ValueError("slot_gates must be [B,L]")

    if text_attention_mask.shape != (
        slot_masks.shape[0],
        slot_masks.shape[2],
    ):
        raise ValueError("text_attention_mask must be [B,N]")

    active = slot_gates >= gate_threshold

    active_count = active.sum(dim=1)

    num_slots = slot_masks.shape[1]

    attention_valid = text_attention_mask.to(slot_masks.dtype)

    valid_masks = slot_masks * attention_valid[:, None, :]

    valid_token_count = attention_valid.sum(dim=1).clamp_min(1.0)

    slot_mass = valid_masks.sum(dim=2) / valid_token_count[:, None]

    result = {
        "active_count": active_count.float(),
        "zero_active": (active_count == 0).float(),
        "all_active": (active_count == num_slots).float(),
        "mask_pair_cosines": _active_pairwise_cosines(
            valid_masks,
            active,
        ),
        "effect_pair_cosines": _active_pairwise_cosines(
            slot_effects,
            active,
        ),
        "active_slot_mass": slot_mass[active],
    }

    # --------------------------------------------------------
    # Optional content-aware coverage
    # --------------------------------------------------------

    if text_content_mask is not None:
        if text_content_mask.shape != text_attention_mask.shape:
            raise ValueError("text_content_mask shape mismatch")

        content = text_content_mask.bool() & text_attention_mask.bool()

        active_masks = slot_masks * active[
            :,
            :,
            None,
        ].to(slot_masks.dtype)

        # Union probability-like claim:
        #
        # 1 - Π_l (1 - A_ln)
        #
        union_claim = 1.0 - torch.prod(
            1.0 - active_masks,
            dim=1,
        )

        content_f = content.to(union_claim.dtype)

        content_count = content_f.sum(dim=1)

        has_content = content_count > 0

        coverage = (union_claim * content_f).sum(dim=1) / content_count.clamp_min(1.0)

        result["content_union_coverage"] = coverage[has_content]

    return result


def summarize_structural_health(
    *,
    active_count: Tensor,
    zero_active: Tensor,
    all_active: Tensor,
    mask_pair_cosines: Tensor,
    effect_pair_cosines: Tensor,
    active_slot_mass: Tensor,
    overlap_margin: float,
    effect_diversity_margin: float,
    content_union_coverage: Tensor | None,
) -> dict[str, float]:
    metrics = {
        "stage1/active_slots_mean": _mean(active_count),
        "stage1/active_slots_std": (
            float(active_count.float().std(unbiased=False).item()) if active_count.numel() else 0.0
        ),
        "stage1/zero_active_rate": _mean(zero_active),
        "stage1/all_active_rate": _mean(all_active),
        "stage1/active_mask_overlap_mean": _mean(mask_pair_cosines),
        "stage1/active_mask_overlap_max": _max(mask_pair_cosines),
        "stage1/active_mask_overlap_violation_rate": _mean(
            (mask_pair_cosines > overlap_margin).float()
        ),
        "stage1/active_effect_cosine_mean": _mean(effect_pair_cosines),
        "stage1/active_effect_cosine_max": _max(effect_pair_cosines),
        "stage1/active_effect_redundancy_violation_rate": _mean(
            (effect_pair_cosines > effect_diversity_margin).float()
        ),
        "stage1/active_slot_mass_mean": _mean(active_slot_mass),
        "stage1/active_slot_mass_p50": _quantile(
            active_slot_mass,
            0.50,
        ),
        "stage1/active_slot_mass_p95": _quantile(
            active_slot_mass,
            0.95,
        ),
        "stage1/active_slot_mass_max": _max(active_slot_mass),
    }

    if content_union_coverage is not None:
        metrics["stage1/content_union_coverage_mean"] = _mean(content_union_coverage)

    return metrics


# ============================================================
# ESSS — Edit-Slot Set Stability
# ============================================================


def _exact_slot_assignment(
    cost: Tensor,
) -> Tensor:
    """
    Exact Hungarian-equivalent assignment for small L.

    Current TAPER uses few slots, e.g. L=4.
    For L=4 there are only 4! = 24 permutations.

    This avoids adding scipy solely for Stage-1 evaluation.
    """

    if cost.ndim != 2 or cost.shape[0] != cost.shape[1]:
        raise ValueError("slot matching cost must be square [L,L]")

    num_slots = cost.shape[0]

    if num_slots > 8:
        raise RuntimeError(
            "Exact permutation matching becomes expensive "
            "for num_slots > 8. "
            "Install/use scipy Hungarian if TAPER grows beyond this."
        )

    cpu_cost = cost.detach().float().cpu()

    rows = torch.arange(num_slots)

    best_permutation = None
    best_cost = float("inf")

    for permutation in itertools.permutations(range(num_slots)):
        columns = torch.tensor(
            permutation,
            dtype=torch.long,
        )

        value = float(
            cpu_cost[
                rows,
                columns,
            ]
            .sum()
            .item()
        )

        if value < best_cost:
            best_cost = value
            best_permutation = permutation

    if best_permutation is None:
        raise RuntimeError("Slot assignment failed")

    return torch.tensor(
        best_permutation,
        dtype=torch.long,
    )


def _match_one_sample_by_mask(
    *,
    current_mask: Tensor,
    previous_mask: Tensor,
) -> Tensor:
    """
    current_mask:
        [L,N_valid]

    previous_mask:
        [L,N_valid]

    Returns:
        assignment [L]

    assignment[l] tells which PREVIOUS slot corresponds
    to CURRENT slot l.
    """

    if current_mask.shape != previous_mask.shape:
        raise ValueError("Current/previous ESSS mask shape mismatch")

    if current_mask.ndim != 2:
        raise ValueError("ESSS sample masks must be [L,N]")

    current = F.normalize(
        current_mask.float(),
        dim=-1,
        eps=1e-8,
    )

    previous = F.normalize(
        previous_mask.float(),
        dim=-1,
        eps=1e-8,
    )

    similarity = current @ previous.t()

    cost = 1.0 - similarity

    return _exact_slot_assignment(cost)


def _stable_anchor_ids(
    sample_ids: Sequence[str],
    *,
    anchor_size: int | None,
    anchor_seed: int | None,
) -> list[str]:
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Validation sample_ids contain duplicates")

    if anchor_size is None or anchor_size >= len(sample_ids):
        return sorted(sample_ids)

    if anchor_size < 1:
        raise ValueError("stability anchor_size must be >= 1")

    if anchor_seed is None:
        raise ValueError(
            "stability.anchor_seed must be explicit when anchor_size is smaller than validation set"
        )

    ranked = []

    for sample_id in sample_ids:
        digest = hashlib.sha256(f"{anchor_seed}:{sample_id}".encode("utf-8")).digest()

        score = int.from_bytes(
            digest[:8],
            byteorder="big",
            signed=False,
        )

        ranked.append(
            (
                score,
                sample_id,
            )
        )

    ranked.sort()

    return [sample_id for _, sample_id in ranked[:anchor_size]]


def build_stability_anchor(
    *,
    records: Mapping[str, dict[str, Tensor]],
    config: Mapping[str, object],
) -> dict[str, object]:
    stability_config = config.get(
        "stability",
        {},
    )

    if not isinstance(
        stability_config,
        Mapping,
    ):
        raise TypeError("evaluation.stability must be a mapping")

    anchor_size_raw = stability_config.get("anchor_size")

    if anchor_size_raw is None:
        anchor_size = None
    else:
        anchor_size = int(anchor_size_raw)

    anchor_seed_raw = stability_config.get("anchor_seed")

    anchor_seed = None if anchor_seed_raw is None else int(anchor_seed_raw)

    sample_ids = list(records.keys())

    selected_ids = _stable_anchor_ids(
        sample_ids,
        anchor_size=anchor_size,
        anchor_seed=anchor_seed,
    )

    selected_records = {
        sample_id: {key: value.detach().cpu().clone() for key, value in records[sample_id].items()}
        for sample_id in selected_ids
    }

    return {
        "sample_ids": selected_ids,
        "records": selected_records,
    }


def compute_esss(
    *,
    current_anchor: Mapping[str, object],
    previous_anchor: Mapping[str, object],
    gate_threshold: float,
) -> dict[str, float]:
    current_ids = list(current_anchor["sample_ids"])

    previous_ids = list(previous_anchor["sample_ids"])

    if current_ids != previous_ids:
        raise ValueError("ESSS fixed anchor sample IDs changed")

    current_records = current_anchor["records"]

    previous_records = previous_anchor["records"]

    if not isinstance(
        current_records,
        Mapping,
    ):
        raise TypeError("current ESSS records must be mapping")

    if not isinstance(
        previous_records,
        Mapping,
    ):
        raise TypeError("previous ESSS records must be mapping")

    mask_similarities = []
    effect_similarities = []
    gate_drifts = []
    active_flips = []

    for sample_id in current_ids:
        current = current_records[sample_id]

        previous = previous_records[sample_id]

        current_mask = current["slot_masks"]

        previous_mask = previous["slot_masks"]

        current_effect = current["slot_effects"]

        previous_effect = previous["slot_effects"]

        current_gate = current["slot_gates"]

        previous_gate = previous["slot_gates"]

        assignment = _match_one_sample_by_mask(
            current_mask=current_mask,
            previous_mask=previous_mask,
        )

        previous_mask = previous_mask[assignment]

        previous_effect = previous_effect[assignment]

        previous_gate = previous_gate[assignment]

        current_active = current_gate >= gate_threshold

        previous_active = previous_gate >= gate_threshold

        # IMPORTANT:
        #
        # include:
        #   active -> active
        #   active -> inactive
        #   inactive -> active
        #
        # exclude:
        #   inactive -> inactive
        #
        union_active = current_active | previous_active

        if not union_active.any():
            continue

        mask_cosine = F.cosine_similarity(
            current_mask.float(),
            previous_mask.float(),
            dim=-1,
            eps=1e-8,
        )

        effect_cosine = F.cosine_similarity(
            current_effect.float(),
            previous_effect.float(),
            dim=-1,
            eps=1e-8,
        )

        gate_drift = (current_gate.float() - previous_gate.float()).abs()

        active_flip = (current_active != previous_active).float()

        mask_similarities.append(mask_cosine[union_active])

        effect_similarities.append(effect_cosine[union_active])

        gate_drifts.append(gate_drift[union_active])

        active_flips.append(active_flip[union_active])

    if not mask_similarities:
        # Dead capacity must NOT fake perfect convergence.
        return {
            "stage1/stability_available": 0.0,
        }

    return {
        "stage1/stability_available": 1.0,
        "stage1/matched_mask_stability": _mean(_cat(mask_similarities)),
        "stage1/matched_effect_stability": _mean(_cat(effect_similarities)),
        "stage1/matched_gate_drift": _mean(_cat(gate_drifts)),
        "stage1/matched_active_flip_rate": _mean(_cat(active_flips)),
    }


# ============================================================
# Optional health constraints
# ============================================================


def compute_health_ok(
    metrics: Mapping[str, float],
    config: Mapping[str, object],
) -> bool:
    """
    Thresholds are OPTIONAL.

    We deliberately do not invent scientific defaults.

    Example:

    evaluation:
      health:
        max_zero_active_rate: null
        max_all_active_rate: null
        max_harmful_active_slot_rate: null
        max_mask_overlap_violation_rate: null
        max_effect_redundancy_violation_rate: null
    """

    health_config = config.get(
        "health",
        {},
    )

    if not isinstance(
        health_config,
        Mapping,
    ):
        raise TypeError("evaluation.health must be a mapping")

    checks = (
        (
            "max_zero_active_rate",
            "stage1/zero_active_rate",
        ),
        (
            "max_all_active_rate",
            "stage1/all_active_rate",
        ),
        (
            "max_harmful_active_slot_rate",
            "stage1/harmful_active_slot_rate",
        ),
        (
            "max_mask_overlap_violation_rate",
            "stage1/active_mask_overlap_violation_rate",
        ),
        (
            "max_effect_redundancy_violation_rate",
            "stage1/active_effect_redundancy_violation_rate",
        ),
    )

    for config_key, metric_key in checks:
        threshold = health_config.get(config_key)

        if threshold is None:
            continue

        if metric_key not in metrics:
            return False

        if float(metrics[metric_key]) > float(threshold):
            return False

    return True


# ============================================================
# Build fixed TCFR cache
# ============================================================


@torch.no_grad()
def build_tcfr_cache(
    *,
    model,
    val_loaders: Mapping[str, DataLoader],
    prepare_batch_fn,
    teacher_gallery_features: Tensor,
    teacher_gallery_name_to_idx: Mapping[str, int],
    gallery_ids_by_category: Mapping[
        str,
        Sequence[str],
    ],
    config: Mapping[str, object],
    device: torch.device,
) -> dict[str, dict[str, object]]:
    """
    Run ONCE before Stage-1 training.

    q_full is independent of learned slot removal, therefore:

        q_full
        -> full teacher rank
        -> teacher-qualified
        -> fixed hard negatives

    can be cached for the whole Stage-1 run.
    """

    _validate_teacher_gallery(
        teacher_gallery_features,
        teacher_gallery_name_to_idx,
    )

    teacher_success_k = int(
        _require_config(
            config,
            "teacher_success_k",
        )
    )

    hard_negative_k = int(
        _require_config(
            config,
            "hard_negative_k",
        )
    )

    exclude_reference = bool(
        _require_config(
            config,
            "exclude_reference_from_candidates",
        )
    )

    score_chunk_size = int(
        config.get(
            "score_query_chunk_size",
            256,
        )
    )

    if teacher_success_k < 1:
        raise ValueError("teacher_success_k must be >= 1")

    if hard_negative_k < 1:
        raise ValueError("hard_negative_k must be >= 1")

    model.eval()

    cache: dict[
        str,
        dict[str, object],
    ] = {}

    for category, val_loader in val_loaders.items():
        (
            gallery_ids,
            gallery,
            local_name_to_idx,
        ) = _get_category_gallery(
            category=category,
            gallery_ids_by_category=gallery_ids_by_category,
            teacher_gallery_features=teacher_gallery_features,
            teacher_gallery_name_to_idx=teacher_gallery_name_to_idx,
            device=device,
        )

        for raw_batch in val_loader:
            prepared = prepare_batch_fn(raw_batch)

            reference_features = prepared["reference_features"]

            text_states = prepared["text_states"]

            text_attention_mask = prepared["text_attention_mask"]

            q_full = model.teacher.compose(
                reference_features,
                text_states,
                text_attention_mask,
                normalize=False,
            )

            if not isinstance(
                q_full,
                Tensor,
            ):
                raise TypeError("teacher.compose() must return Tensor")

            if q_full.ndim != 2:
                raise ValueError("teacher q_full must be [B,D]")

            if q_full.shape[-1] != gallery.shape[-1]:
                raise ValueError(
                    "Teacher q_full and teacher gallery "
                    "live in different dimensions. "
                    "Do not auto-project."
                )

            sample_ids = list(raw_batch.sample_ids)

            reference_ids = list(raw_batch.reference_ids)

            target_ids = []

            for target_id in raw_batch.target_ids:
                if target_id is None:
                    raise ValueError("Stage-1 validation sample is missing target_id")

                target_ids.append(target_id)

            if len(sample_ids) != q_full.shape[0]:
                raise ValueError("q_full batch size mismatch")

            if len(set(sample_ids)) != len(sample_ids):
                raise ValueError("Duplicate sample ID inside validation batch")

            for sample_id in sample_ids:
                if sample_id in cache:
                    raise ValueError(f"Duplicate validation sample_id: {sample_id}")

            target_indices = _ids_to_indices(
                target_ids,
                local_name_to_idx,
                device=device,
                field_name="target_id",
            )

            reference_indices = None

            if exclude_reference:
                reference_indices = _ids_to_indices(
                    reference_ids,
                    local_name_to_idx,
                    device=device,
                    field_name="reference_id",
                )

                if (reference_indices == target_indices).any():
                    raise ValueError(
                        "reference_id == target_id while reference exclusion is enabled"
                    )

            extra_exclusions = _known_positive_exclusions(
                ground_truth_ids=(raw_batch.ground_truth_ids),
                target_ids=target_ids,
                local_name_to_idx=local_name_to_idx,
            )

            full_scores = _score_in_chunks(
                q_full,
                gallery,
                chunk_size=score_chunk_size,
            )

            full_ranks = compute_target_ranks(
                full_scores,
                target_indices,
                reference_indices=reference_indices,
                extra_excluded_indices=extra_exclusions,
            )

            hard_negative_indices = build_fixed_hard_negatives(
                full_scores,
                target_indices,
                hard_negative_k=hard_negative_k,
                reference_indices=reference_indices,
                extra_excluded_indices=extra_exclusions,
            )

            full_margin = compute_target_margin(
                scores=full_scores,
                target_indices=target_indices,
                hard_negative_indices=(hard_negative_indices),
            )

            for row, sample_id in enumerate(sample_ids):
                negative_ids = [gallery_ids[index] for index in hard_negative_indices[row].tolist()]

                cache[sample_id] = {
                    "sample_id": sample_id,
                    "category": category,
                    "reference_id": reference_ids[row],
                    "target_id": target_ids[row],
                    "teacher_full_rank": int(full_ranks[row].item()),
                    "teacher_qualified": bool(full_ranks[row].item() <= teacher_success_k),
                    "full_margin": float(full_margin[row].item()),
                    # Store IDs instead of local row indices.
                    # This keeps the cache interpretable.
                    "hard_negative_ids": negative_ids,
                }

    if not cache:
        raise RuntimeError("TCFR cache is empty")

    qualified_count = sum(int(record["teacher_qualified"]) for record in cache.values())

    if qualified_count == 0:
        raise RuntimeError(
            "Teacher-qualified validation coverage is zero. "
            "TCFR cannot be used as Stage-1 primary metric."
        )

    return cache


# ============================================================
# Complete Stage-1 evaluator
# ============================================================


@torch.no_grad()
def evaluate_stage1_edit_slots(
    *,
    model,
    val_loaders: Mapping[str, DataLoader],
    prepare_batch_fn,
    teacher_gallery_features: Tensor,
    teacher_gallery_name_to_idx: Mapping[str, int],
    gallery_ids_by_category: Mapping[
        str,
        Sequence[str],
    ],
    tcfr_cache: Mapping[
        str,
        Mapping[str, object],
    ],
    previous_anchor: Mapping[str, object] | None,
    config: Mapping[str, object],
    device: torch.device,
) -> tuple[
    dict[str, float | bool],
    dict[str, object] | None,
]:
    """
    Stage-1 only:

        build_edit_slots
            ↓
        q_teacher_full / q_teacher_minus
            ↓
        TCFR
            +
        structural health
            +
        ESSS

    No Router.
    No Primitive Bank.
    No Executor.
    No final TAPER retrieval query.
    """

    _validate_teacher_gallery(
        teacher_gallery_features,
        teacher_gallery_name_to_idx,
    )

    score_chunk_size = int(
        config.get(
            "score_query_chunk_size",
            256,
        )
    )

    model_gate_threshold = float(model.slot_gate_threshold)

    if "slot_gate_threshold" in config:
        configured_threshold = float(config["slot_gate_threshold"])

        if abs(configured_threshold - model_gate_threshold) > 1e-8:
            raise ValueError(
                "evaluation.slot_gate_threshold differs from model.slot_gate_threshold"
            )

    gate_threshold = model_gate_threshold

    exclude_reference = bool(
        _require_config(
            config,
            "exclude_reference_from_candidates",
        )
    )

    model.eval()

    # --------------------------------------------------------
    # TCFR values
    # --------------------------------------------------------

    qualified_flags = []
    tcfr_hard_values = []
    tcfr_soft_values = []
    harmful_values = []

    full_rank_values = []
    rank_hurt_values = []

    # --------------------------------------------------------
    # Structural health values
    # --------------------------------------------------------

    active_count_values = []
    zero_active_values = []
    all_active_values = []

    mask_pair_values = []
    effect_pair_values = []

    active_slot_mass_values = []

    content_coverage_values = []
    content_coverage_available = True

    # --------------------------------------------------------
    # ESSS candidate records
    # --------------------------------------------------------

    stability_records: dict[
        str,
        dict[str, Tensor],
    ] = {}

    all_sample_ids = []

    for category, val_loader in val_loaders.items():
        (
            _,
            gallery,
            local_name_to_idx,
        ) = _get_category_gallery(
            category=category,
            gallery_ids_by_category=gallery_ids_by_category,
            teacher_gallery_features=teacher_gallery_features,
            teacher_gallery_name_to_idx=teacher_gallery_name_to_idx,
            device=device,
        )

        for raw_batch in val_loader:
            prepared = prepare_batch_fn(raw_batch)

            for required_key in (
                "reference_features",
                "text_states",
                "text_attention_mask",
            ):
                if required_key not in prepared:
                    raise KeyError(f"prepare_batch_fn missing {required_key!r}")

            reference_features = prepared["reference_features"]

            text_states = prepared["text_states"]

            text_attention_mask = prepared["text_attention_mask"]

            slots = model.build_edit_slots(
                reference_features=(reference_features),
                text_states=text_states,
                text_attention_mask=(text_attention_mask),
            )

            slot_masks = slots["slot_masks"]

            slot_effects = slots["slot_effects"]

            slot_gates = slots["slot_gates"]

            q_full = slots["q_teacher_full"]

            q_minus = slots["q_teacher_minus"]

            tensors_to_check = {
                "slot_masks": slot_masks,
                "slot_effects": slot_effects,
                "slot_gates": slot_gates,
                "q_teacher_full": q_full,
                "q_teacher_minus": q_minus,
            }

            for name, tensor in tensors_to_check.items():
                if not torch.isfinite(tensor).all():
                    raise ValueError(f"{name} contains NaN or Inf")

            batch_size, num_slots = slot_gates.shape

            if q_full.shape != (
                batch_size,
                model.teacher_query_dim,
            ):
                raise ValueError("Unexpected q_teacher_full shape")

            if q_minus.shape != (
                batch_size,
                num_slots,
                model.teacher_query_dim,
            ):
                raise ValueError("Unexpected q_teacher_minus shape")

            if gallery.shape[-1] != model.teacher_query_dim:
                raise ValueError("Teacher query/gallery dimension mismatch. Do not auto-project.")

            sample_ids = list(raw_batch.sample_ids)

            reference_ids = list(raw_batch.reference_ids)

            target_ids = []

            for target_id in raw_batch.target_ids:
                if target_id is None:
                    raise ValueError("Stage-1 validation sample is missing target_id")

                target_ids.append(target_id)

            all_sample_ids.extend(sample_ids)

            # ------------------------------------------------
            # Validate fixed TCFR cache
            # ------------------------------------------------

            cache_rows = []

            for row, sample_id in enumerate(sample_ids):
                if sample_id not in tcfr_cache:
                    raise KeyError(f"TCFR cache missing sample {sample_id}")

                cached = tcfr_cache[sample_id]

                if cached["category"] != category:
                    raise ValueError("TCFR cache category changed")

                if cached["reference_id"] != reference_ids[row]:
                    raise ValueError("TCFR cache reference_id changed")

                if cached["target_id"] != target_ids[row]:
                    raise ValueError("TCFR cache target_id changed")

                cache_rows.append(cached)

            target_indices = _ids_to_indices(
                target_ids,
                local_name_to_idx,
                device=device,
                field_name="target_id",
            )

            hard_negative_indices = []

            for cached in cache_rows:
                negative_ids = cached["hard_negative_ids"]

                hard_negative_indices.append(
                    [local_name_to_idx[image_id] for image_id in negative_ids]
                )

            hard_negative_indices = torch.tensor(
                hard_negative_indices,
                dtype=torch.long,
                device=device,
            )

            full_margin = torch.tensor(
                [float(cached["full_margin"]) for cached in cache_rows],
                dtype=torch.float32,
                device=device,
            )

            full_ranks = torch.tensor(
                [int(cached["teacher_full_rank"]) for cached in cache_rows],
                dtype=torch.long,
                device=device,
            )

            qualified = torch.tensor(
                [bool(cached["teacher_qualified"]) for cached in cache_rows],
                dtype=torch.bool,
                device=device,
            )

            # ------------------------------------------------
            # Score q_minus against teacher-native gallery
            # ------------------------------------------------

            flat_minus = q_minus.reshape(
                batch_size * num_slots,
                model.teacher_query_dim,
            )

            flat_minus_scores = _score_in_chunks(
                flat_minus,
                gallery,
                chunk_size=score_chunk_size,
            )

            minus_scores = flat_minus_scores.reshape(
                batch_size,
                num_slots,
                gallery.shape[0],
            )

            minus_margin = compute_slot_target_margin(
                scores=minus_scores,
                target_indices=target_indices,
                hard_negative_indices=(hard_negative_indices),
            )

            tcfr = compute_tcfr(
                full_margin=full_margin,
                minus_margin=minus_margin,
                slot_gates=slot_gates,
                gate_threshold=gate_threshold,
            )

            # ------------------------------------------------
            # Primary TCFR
            # ------------------------------------------------

            tcfr_hard_values.append(tcfr["per_sample_tcfr"][qualified].cpu())

            tcfr_soft_values.append(tcfr["per_sample_tcfr_soft"][qualified].cpu())

            qualified_flags.append(qualified.float().cpu())

            full_rank_values.append(full_ranks.float().cpu())

            qualified_active = qualified[:, None] & tcfr["active_mask"]

            harmful_values.append((tcfr["margin_drop"][qualified_active] < 0).float().cpu())

            # ------------------------------------------------
            # Rank hurt — secondary diagnostic
            # ------------------------------------------------

            reference_indices = None

            if exclude_reference:
                reference_indices = _ids_to_indices(
                    reference_ids,
                    local_name_to_idx,
                    device=device,
                    field_name="reference_id",
                )

            extra_exclusions = _known_positive_exclusions(
                ground_truth_ids=(raw_batch.ground_truth_ids),
                target_ids=target_ids,
                local_name_to_idx=local_name_to_idx,
            )

            minus_ranks_by_slot = []

            for slot_id in range(num_slots):
                slot_ranks = compute_target_ranks(
                    minus_scores[
                        :,
                        slot_id,
                        :,
                    ],
                    target_indices,
                    reference_indices=(reference_indices),
                    extra_excluded_indices=(extra_exclusions),
                )

                minus_ranks_by_slot.append(slot_ranks)

            minus_ranks = torch.stack(
                minus_ranks_by_slot,
                dim=1,
            )

            rank_hurt = minus_ranks - full_ranks[:, None]

            rank_hurt_values.append(rank_hurt[qualified_active].float().cpu())

            # ------------------------------------------------
            # Structural health — ALL validation samples
            # ------------------------------------------------

            text_content_mask = prepared.get("text_content_mask")

            if text_content_mask is None:
                content_coverage_available = False

            health = compute_structural_health(
                slot_masks=slot_masks,
                slot_effects=slot_effects,
                slot_gates=slot_gates,
                text_attention_mask=(text_attention_mask),
                gate_threshold=gate_threshold,
                text_content_mask=(text_content_mask),
            )

            active_count_values.append(health["active_count"].cpu())

            zero_active_values.append(health["zero_active"].cpu())

            all_active_values.append(health["all_active"].cpu())

            if health["mask_pair_cosines"].numel():
                mask_pair_values.append(health["mask_pair_cosines"].cpu())

            if health["effect_pair_cosines"].numel():
                effect_pair_values.append(health["effect_pair_cosines"].cpu())

            if health["active_slot_mass"].numel():
                active_slot_mass_values.append(health["active_slot_mass"].cpu())

            if "content_union_coverage" in health and health["content_union_coverage"].numel():
                content_coverage_values.append(health["content_union_coverage"].cpu())

            # ------------------------------------------------
            # Save per-sample outputs for ESSS
            # ------------------------------------------------

            for row, sample_id in enumerate(sample_ids):
                if sample_id in stability_records:
                    raise ValueError(f"Duplicate validation sample_id: {sample_id}")

                attention_valid = text_attention_mask[row].bool()

                if not attention_valid.any():
                    raise ValueError(f"Sample {sample_id} has no valid text token")

                # Trim padding before storing.
                # This makes ESSS independent of batch padding length.
                stability_records[sample_id] = {
                    "slot_masks": slot_masks[
                        row,
                        :,
                        attention_valid,
                    ]
                    .float()
                    .cpu(),
                    "slot_effects": slot_effects[row].float().cpu(),
                    "slot_gates": slot_gates[row].float().cpu(),
                }

    # ========================================================
    # Aggregate validation
    # ========================================================

    if not all_sample_ids:
        raise RuntimeError("Stage-1 validation produced no samples")

    if len(set(all_sample_ids)) != len(all_sample_ids):
        raise ValueError("Validation sample_ids are not unique")

    qualified_tensor = _cat(qualified_flags)

    qualified_count = int(qualified_tensor.sum().item())

    total_count = int(qualified_tensor.numel())

    if qualified_count == 0:
        raise RuntimeError(
            "Teacher-qualified coverage is zero. TCFR cannot be used as primary metric."
        )

    tcfr_hard_tensor = _cat(tcfr_hard_values)

    tcfr_soft_tensor = _cat(tcfr_soft_values)

    if tcfr_hard_tensor.numel() != qualified_count:
        raise RuntimeError("Qualified TCFR sample count mismatch")

    harmful_tensor = _cat(harmful_values)

    rank_hurt_tensor = _cat(rank_hurt_values)

    active_count_tensor = _cat(active_count_values)

    zero_active_tensor = _cat(zero_active_values)

    all_active_tensor = _cat(all_active_values)

    mask_pair_tensor = _cat(mask_pair_values)

    effect_pair_tensor = _cat(effect_pair_values)

    active_slot_mass_tensor = _cat(active_slot_mass_values)

    content_coverage_tensor = None

    # Only report content coverage when it was valid/available
    # for the whole validation path.
    if content_coverage_available and content_coverage_values:
        content_coverage_tensor = _cat(content_coverage_values)

    metrics: dict[
        str,
        float | bool,
    ] = summarize_structural_health(
        active_count=active_count_tensor,
        zero_active=zero_active_tensor,
        all_active=all_active_tensor,
        mask_pair_cosines=mask_pair_tensor,
        effect_pair_cosines=effect_pair_tensor,
        active_slot_mass=active_slot_mass_tensor,
        overlap_margin=float(model.overlap_margin),
        effect_diversity_margin=float(model.effect_diversity_margin),
        content_union_coverage=(content_coverage_tensor),
    )

    metrics.update(
        {
            # -----------------------------------------------
            # Primary quality
            # -----------------------------------------------
            "stage1/tcfr_margin_drop": _mean(tcfr_hard_tensor),
            # Diagnostic only.
            "stage1/tcfr_margin_drop_soft": _mean(tcfr_soft_tensor),
            # -----------------------------------------------
            # Teacher coverage
            # -----------------------------------------------
            "stage1/teacher_qualified_coverage": (qualified_count / total_count),
            "stage1/teacher_qualified_count": float(qualified_count),
            "stage1/validation_count": float(total_count),
            "stage1/teacher_full_rank_mean": _mean(_cat(full_rank_values)),
            # -----------------------------------------------
            # Secondary functional diagnostics
            # -----------------------------------------------
            "stage1/tcfr_rank_hurt_mean": _mean(rank_hurt_tensor),
            "stage1/tcfr_rank_hurt_median": _median(rank_hurt_tensor),
            "stage1/tcfr_rank_hurt_positive_rate": _mean((rank_hurt_tensor > 0).float()),
            "stage1/harmful_active_slot_rate": _mean(harmful_tensor),
            # First evaluation has no ESSS comparison yet.
            "stage1/stability_available": 0.0,
        }
    )

    # ========================================================
    # ESSS
    # ========================================================

    stability_config = config.get(
        "stability",
        {},
    )

    if not isinstance(
        stability_config,
        Mapping,
    ):
        raise TypeError("evaluation.stability must be mapping")

    stability_enabled = bool(
        stability_config.get(
            "enabled",
            True,
        )
    )

    current_anchor = None

    if stability_enabled:
        current_anchor = build_stability_anchor(
            records=stability_records,
            config=config,
        )

        if previous_anchor is not None:
            metrics.update(
                compute_esss(
                    current_anchor=(current_anchor),
                    previous_anchor=(previous_anchor),
                    gate_threshold=(gate_threshold),
                )
            )

    # ========================================================
    # Optional health guards
    # ========================================================

    metrics["stage1/health_ok"] = compute_health_ok(
        metrics,
        config,
    )

    return (
        metrics,
        current_anchor,
    )
