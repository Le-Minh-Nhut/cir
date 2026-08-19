from __future__ import annotations
import itertools
from collections.abc import Mapping, Sequence
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

def _mean(x: Tensor) -> float:
    return float(x.float().mean().item()) if x.numel() else 0.0

def _max(x: Tensor) -> float:
    return float(x.float().max().item()) if x.numel() else 0.0

def _q(x: Tensor, q: float) -> float:
    return float(torch.quantile(x.float(), q).item()) if x.numel() else 0.0

def _cat(xs: list[Tensor]) -> Tensor:
    return torch.cat(xs, dim=0) if xs else torch.empty(0, dtype=torch.float32)

def _required(config: Mapping[str, object], key: str):
    if key not in config:
        raise KeyError(f"Missing Stage-1 evaluation config key: {key}")
    return config[key]

def _validate_gallery(gallery: Tensor, name_to_idx: Mapping[str, int], score_mode: str) -> None:
    if score_mode == "cosine" and gallery.ndim != 2:
        raise ValueError("teacher_score_mode='cosine' requires gallery [G,D]")
    if score_mode == "max_token" and gallery.ndim != 3:
        raise ValueError("teacher_score_mode='max_token' requires gallery [G,K,D]")
    if score_mode not in {"cosine", "max_token"}:
        raise ValueError("teacher_score_mode must be 'cosine' or 'max_token'")
    if gallery.shape[0] != len(name_to_idx):
        raise ValueError("teacher gallery feature count != len(name_to_idx)")
    if not torch.isfinite(gallery).all():
        raise ValueError("teacher gallery contains NaN or Inf")
    indices = list(name_to_idx.values())
    if set(indices) != set(range(gallery.shape[0])):
        raise ValueError("teacher gallery name_to_idx must cover contiguous [0,G)")

def _score_queries(queries: Tensor, gallery: Tensor, *, score_mode: str, gallery_chunk_size: int) -> Tensor:
    """[Q,D] x teacher-native gallery -> [Q,G]."""
    if queries.ndim != 2:
        raise ValueError("queries must be [Q,D]")
    if queries.shape[-1] != gallery.shape[-1]:
        raise ValueError("teacher query/gallery dimension mismatch")
    if gallery_chunk_size < 1:
        raise ValueError("gallery_chunk_size must be >= 1")
    queries = F.normalize(queries.float(), dim=-1)
    score_chunks = []
    for start in range(0, gallery.shape[0], gallery_chunk_size):
        g = (
            gallery[start : start + gallery_chunk_size]
            .to(queries.device, non_blocking=True)
            .float()
        )
        g = F.normalize(g, dim=-1)
        if score_mode == "cosine":
            scores = queries @ g.T
        else:
            scores = torch.einsum("qd,gkd->qgk", queries, g).max(dim=-1).values
        score_chunks.append(scores)
    scores = torch.cat(score_chunks, dim=1)
    if not torch.isfinite(scores).all():
        raise ValueError("teacher scorer returned NaN or Inf")
    return scores

def _score_paired_candidates(queries: Tensor, candidates: Tensor, *, score_mode: str) -> Tensor:
    """
    queries: [B,L,D]
    candidates:
      cosine    -> [B,C,D]
      max_token -> [B,C,K,D]
    returns: [B,L,C]
    """
    if queries.ndim != 3:
        raise ValueError("queries must be [B,L,D]")
    if queries.shape[-1] != candidates.shape[-1]:
        raise ValueError("teacher query/candidate dimension mismatch")
    q = F.normalize(queries.float(), dim=-1)
    c = F.normalize(candidates.float(), dim=-1)
    if score_mode == "cosine":
        if c.ndim != 3:
            raise ValueError("cosine candidates must be [B,C,D]")
        scores = torch.einsum("bld,bcd->blc", q, c)
    elif score_mode == "max_token":
        if c.ndim != 4:
            raise ValueError("max_token candidates must be [B,C,K,D]")
        scores = torch.einsum("bld,bckd->blck", q, c).max(dim=-1).values
    else:
        raise ValueError("teacher_score_mode must be 'cosine' or 'max_token'")
    if not torch.isfinite(scores).all():
        raise ValueError("teacher paired scorer returned NaN or Inf")
    return scores

def _get_category_gallery(
    category: str,
    gallery_ids_by_category: Mapping[str, Sequence[str]],
    teacher_gallery_features: Tensor,
    teacher_gallery_name_to_idx: Mapping[str, int],
    device: torch.device,
) -> tuple[list[str], Tensor, dict[str, int]]:
    if category not in gallery_ids_by_category:
        raise KeyError(f"Missing gallery IDs for category={category}")
    gallery_ids = list(gallery_ids_by_category[category])
    if not gallery_ids:
        raise ValueError(f"Empty gallery for category={category}")
    if len(set(gallery_ids)) != len(gallery_ids):
        raise ValueError(f"Duplicate gallery IDs for category={category}")
    missing = [x for x in gallery_ids if x not in teacher_gallery_name_to_idx]
    if missing:
        raise KeyError(f"Teacher gallery cache missing IDs: {missing[:5]}")
    global_indices = [teacher_gallery_name_to_idx[x] for x in gallery_ids]
    gallery = teacher_gallery_features[global_indices].to(device, non_blocking=True)
    local_index = {image_id: i for i, image_id in enumerate(gallery_ids)}
    return (gallery_ids, gallery, local_index)

def _ids_to_indices(
    image_ids: Sequence[str],
    name_to_idx: Mapping[str, int],
    *,
    device: torch.device,
    field_name: str,
) -> Tensor:
    missing = [x for x in image_ids if x not in name_to_idx]
    if missing:
        raise KeyError(f"{field_name} missing from teacher gallery: {missing[:5]}")
    return torch.tensor([name_to_idx[x] for x in image_ids], device=device)

def compute_target_ranks(
    scores: Tensor, target_indices: Tensor, *, reference_indices: Tensor | None = None
) -> Tensor:
    """Return 1-based target ranks using the same argsort semantics as retrieval eval."""
    if scores.ndim != 2 or target_indices.shape != (scores.shape[0],):
        raise ValueError("scores/target_indices must be [B,G] and [B]")
    scores = scores.clone()
    rows = torch.arange(scores.shape[0], device=scores.device)
    if reference_indices is not None:
        if reference_indices.shape != target_indices.shape:
            raise ValueError("reference_indices must be [B]")
        if (reference_indices == target_indices).any():
            raise ValueError("reference_id == target_id while reference exclusion is enabled")
        scores[rows, reference_indices] = -torch.inf
    target_scores = scores[rows, target_indices]
    if not torch.isfinite(target_scores).all():
        raise ValueError("positive target was removed from candidate set")
    ranking = torch.argsort(scores, dim=1, descending=True)
    matches = ranking.eq(target_indices[:, None])
    if not matches.any(dim=1).all():
        raise RuntimeError("target index disappeared from ranking")
    return matches.to(torch.int64).argmax(dim=1) + 1

def build_fixed_hard_negatives(
    full_scores: Tensor,
    target_indices: Tensor,
    *,
    hard_negative_k: int,
    reference_indices: Tensor | None = None,
) -> Tensor:
    """Mine the fixed H_i from q_full only, using retrieval ranking order."""
    if hard_negative_k < 1:
        raise ValueError("hard_negative_k must be >= 1")
    scores = full_scores.clone()
    rows = torch.arange(scores.shape[0], device=scores.device)
    if reference_indices is not None:
        if reference_indices.shape != target_indices.shape:
            raise ValueError("reference_indices must be [B]")
        scores[rows, reference_indices] = -torch.inf
    scores[rows, target_indices] = -torch.inf
    if (torch.isfinite(scores).sum(dim=1) < hard_negative_k).any():
        raise ValueError("not enough valid hard negatives")
    ranking = torch.argsort(scores, dim=1, descending=True)
    negatives = ranking[:, :hard_negative_k]
    if not torch.isfinite(scores.gather(1, negatives)).all():
        raise RuntimeError("invalid candidate entered fixed hard-negative set")
    return negatives

def compute_tcfr(
    *, full_margin: Tensor, minus_margin: Tensor, slot_gates: Tensor, gate_threshold: float
) -> dict[str, Tensor]:
    """
    full_margin: [B]
    minus_margin: [B,L]
    slot_gates: [B,L]
    """
    if full_margin.ndim != 1:
        raise ValueError("full_margin must be [B]")
    if minus_margin.ndim != 2 or minus_margin.shape[0] != full_margin.shape[0]:
        raise ValueError("minus_margin must be [B,L]")
    if slot_gates.shape != minus_margin.shape:
        raise ValueError("slot_gates must be [B,L]")
    margin_drop = full_margin[:, None] - minus_margin
    active = slot_gates >= gate_threshold
    active_count = active.sum(dim=1)
    hard = torch.where(
        active_count > 0,
        (margin_drop * active).sum(dim=1) / active_count.clamp_min(1),
        torch.zeros_like(full_margin),
    )
    soft = (margin_drop * slot_gates).sum(dim=1) / slot_gates.sum(dim=1).clamp_min(1e-08)
    return {
        "margin_drop": margin_drop,
        "active_mask": active,
        "per_sample_tcfr": hard,
        "per_sample_tcfr_soft": soft,
    }

def _active_pair_cosines(values: Tensor, active: Tensor) -> Tensor:
    values = F.normalize(values.float(), dim=-1, eps=1e-08)
    pairwise = torch.einsum("bld,bmd->blm", values, values)
    l = values.shape[1]
    upper = torch.triu(torch.ones(l, l, dtype=torch.bool, device=values.device), diagonal=1)
    keep = active[:, :, None] & active[:, None, :] & upper[None]
    return pairwise[keep]

def compute_stage1_health(
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
    if slot_effects.ndim != 3 or slot_effects.shape[:2] != slot_masks.shape[:2]:
        raise ValueError("slot_effects must be [B,L,D]")
    if slot_gates.shape != slot_masks.shape[:2]:
        raise ValueError("slot_gates must be [B,L]")
    if text_attention_mask.shape != (slot_masks.shape[0], slot_masks.shape[2]):
        raise ValueError("text_attention_mask must be [B,N]")
    active = slot_gates >= gate_threshold
    active_count = active.sum(dim=1)
    if text_content_mask is not None:
        valid_mask = text_content_mask.bool() & text_attention_mask.bool()
    else:
        valid_mask = text_attention_mask.bool()

    valid = valid_mask.to(slot_masks.dtype)[:, None, :]
    masks = slot_masks * valid
    mass = masks.sum(dim=2) / valid.sum(dim=2).clamp_min(1.0)
    out = {
        "active_count": active_count.float(),
        "zero_active": (active_count == 0).float(),
        "all_active": (active_count == slot_masks.shape[1]).float(),
        "mask_pair_cosines": _active_pair_cosines(masks, active),
        "effect_pair_cosines": _active_pair_cosines(slot_effects, active),
        "active_slot_mass": mass[active],
    }
    if text_content_mask is not None:
        if text_content_mask.shape != text_attention_mask.shape:
            raise ValueError("text_content_mask must match text_attention_mask")
        content = text_content_mask.bool() & text_attention_mask.bool()
        active_masks = slot_masks * active[:, :, None].to(slot_masks.dtype)
        union_claim = 1.0 - torch.prod(1.0 - active_masks, dim=1)
        content_f = content.to(union_claim.dtype)
        content_count = content_f.sum(dim=1)
        has_content = content_count > 0
        coverage = (union_claim * content_f).sum(dim=1) / content_count.clamp_min(1.0)
        out["content_union_coverage"] = coverage[has_content]
    return out

def _summarize_health(
    health: dict[str, Tensor], *, overlap_margin: float, effect_diversity_margin: float
) -> dict[str, float]:
    masks = health["mask_pair_cosines"]
    effects = health["effect_pair_cosines"]
    mass = health["active_slot_mass"]
    active_count = health["active_count"]
    metrics = {
        "stage1/active_slots_mean": _mean(active_count),
        "stage1/active_slots_std": float(active_count.std(unbiased=False).item())
        if active_count.numel()
        else 0.0,
        "stage1/zero_active_rate": _mean(health["zero_active"]),
        "stage1/all_active_rate": _mean(health["all_active"]),
        "stage1/active_mask_overlap_mean": _mean(masks),
        "stage1/active_mask_overlap_max": _max(masks),
        "stage1/active_mask_overlap_violation_rate": _mean((masks > overlap_margin).float()),
        "stage1/active_effect_cosine_mean": _mean(effects),
        "stage1/active_effect_cosine_max": _max(effects),
        "stage1/active_effect_redundancy_violation_rate": _mean(
            (effects > effect_diversity_margin).float()
        ),
        "stage1/active_slot_mass_mean": _mean(mass),
        "stage1/active_slot_mass_p50": _q(mass, 0.5),
        "stage1/active_slot_mass_p95": _q(mass, 0.95),
        "stage1/active_slot_mass_max": _max(mass),
    }
    if "content_union_coverage" in health:
        metrics["stage1/content_union_coverage_mean"] = _mean(health["content_union_coverage"])
    return metrics

def _exact_assignment(cost: Tensor) -> Tensor:
    """Exact one-to-one assignment; TAPER V0 has small L (e.g. 4)."""
    if cost.ndim != 2 or cost.shape[0] != cost.shape[1]:
        raise ValueError("cost must be square [L,L]")
    l = cost.shape[0]
    if l > 8:
        raise RuntimeError("Use scipy Hungarian if num_slots > 8")
    cost = cost.detach().float().cpu()
    rows = torch.arange(l)
    best_perm = None
    best_value = float("inf")
    for perm in itertools.permutations(range(l)):
        cols = torch.tensor(perm)
        value = float(cost[rows, cols].sum().item())
        if value < best_value:
            best_value = value
            best_perm = perm
    if best_perm is None:
        raise RuntimeError("slot assignment failed")
    return torch.tensor(best_perm, dtype=torch.long)

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
    if not isinstance(current_records, Mapping) or not isinstance(previous_records, Mapping):
        raise TypeError("ESSS records must be mappings")
    mask_values, effect_values, gate_values, flip_values = ([], [], [], [])
    for sample_id in current_ids:
        cur = current_records[sample_id]
        prev = previous_records[sample_id]
        cm, pm = (cur["slot_masks"], prev["slot_masks"])
        ce, pe = (cur["slot_effects"], prev["slot_effects"])
        cg, pg = (cur["slot_gates"], prev["slot_gates"])
        if cm.shape != pm.shape or ce.shape != pe.shape or cg.shape != pg.shape:
            raise ValueError(f"ESSS shape changed for sample={sample_id}")
        c_norm = F.normalize(cm.float(), dim=-1, eps=1e-08)
        p_norm = F.normalize(pm.float(), dim=-1, eps=1e-08)
        assignment = _exact_assignment(1.0 - c_norm @ p_norm.T)
        pm, pe, pg = (pm[assignment], pe[assignment], pg[assignment])
        cur_active = cg >= gate_threshold
        prev_active = pg >= gate_threshold
        union_active = cur_active | prev_active
        if not union_active.any():
            continue
        mask_cos = F.cosine_similarity(cm.float(), pm.float(), dim=-1, eps=1e-08)
        effect_cos = F.cosine_similarity(ce.float(), pe.float(), dim=-1, eps=1e-08)
        gate_drift = (cg.float() - pg.float()).abs()
        active_flip = (cur_active != prev_active).float()
        mask_values.append(mask_cos[union_active])
        effect_values.append(effect_cos[union_active])
        gate_values.append(gate_drift[union_active])
        flip_values.append(active_flip[union_active])
    if not mask_values:
        return {"stage1/stability_available": False}
    return {
        "stage1/stability_available": True,
        "stage1/matched_mask_stability": _mean(_cat(mask_values)),
        "stage1/matched_effect_stability": _mean(_cat(effect_values)),
        "stage1/matched_gate_drift": _mean(_cat(gate_values)),
        "stage1/matched_active_flip_rate": _mean(_cat(flip_values)),
    }

def _health_ok(metrics: Mapping[str, float | bool], config: Mapping[str, object]) -> bool:
    """Optional rejection gates. None means 'log only'."""
    health = config.get("health", {})
    if not isinstance(health, Mapping):
        raise TypeError("evaluation.health must be a mapping")
    checks = (
        ("max_zero_active_rate", "stage1/zero_active_rate"),
        ("max_all_active_rate", "stage1/all_active_rate"),
        ("max_harmful_active_slot_rate", "stage1/harmful_active_slot_rate"),
        ("max_mask_overlap_violation_rate", "stage1/active_mask_overlap_violation_rate"),
        ("max_effect_redundancy_violation_rate", "stage1/active_effect_redundancy_violation_rate"),
    )
    for config_key, metric_key in checks:
        threshold = health.get(config_key)
        if threshold is not None and float(metrics[metric_key]) > float(threshold):
            return False
    return True

@torch.no_grad()
def build_tcfr_cache(
    *,
    model,
    val_loaders: Mapping[str, DataLoader],
    prepare_batch_fn,
    teacher_galleries: Mapping[str, tuple[Tensor, Mapping[str, int]]],
    gallery_ids_by_category: Mapping[str, Sequence[str]],
    config: Mapping[str, object],
    device: torch.device,
) -> dict[str, dict[str, object]]:
    score_mode = str(_required(config, "teacher_score_mode"))
    teacher_success_k = int(_required(config, "teacher_success_k"))
    hard_negative_k = int(_required(config, "hard_negative_k"))
    exclude_reference = bool(_required(config, "exclude_reference_from_candidates"))
    gallery_chunk_size = int(config.get("gallery_chunk_size", 1024))
    if teacher_success_k < 1 or hard_negative_k < 1:
        raise ValueError("teacher_success_k and hard_negative_k must be >= 1")
    model.eval()
    cache = {}

    for category, val_loader in val_loaders.items():
        if category not in teacher_galleries:
            raise KeyError(f"Missing teacher gallery for category={category}")

        (teacher_gallery_features, teacher_gallery_name_to_idx) = teacher_galleries[category]
        _validate_gallery(teacher_gallery_features, teacher_gallery_name_to_idx, score_mode)
        (gallery_ids, gallery, local_index) = _get_category_gallery(
            category,
            gallery_ids_by_category,
            teacher_gallery_features,
            teacher_gallery_name_to_idx,
            device,
        )
        for raw_batch in val_loader:
            if any(raw_batch.ground_truth_ids):
                raise NotImplementedError("V0 evaluator is FashionIQ/single-positive only. Add benchmark-aware known-positive exclusion before CIRR.")
            batch = prepare_batch_fn(raw_batch)
            q_full = model.teacher.compose(
                batch["reference_features"],
                batch["text_states"],
                batch["text_attention_mask"],
                normalize=False,
            )
            if q_full.ndim != 2 or q_full.shape[-1] != model.teacher_query_dim:
                raise ValueError(f"teacher q_full must be [B,{model.teacher_query_dim}]")
            if not torch.isfinite(q_full).all():
                raise ValueError("teacher q_full contains NaN or Inf")
            sample_ids = list(raw_batch.sample_ids)
            reference_ids = list(raw_batch.reference_ids)
            target_ids = []
            for target_id in raw_batch.target_ids:
                if target_id is None:
                    raise ValueError("validation sample is missing target_id")
                target_ids.append(target_id)
            if any((sample_id in cache for sample_id in sample_ids)):
                raise ValueError("duplicate validation sample_id")
            target_indices = _ids_to_indices(target_ids, local_index, device=device, field_name="target_id")
            reference_indices = (
                _ids_to_indices(
                    reference_ids, local_index, device=device, field_name="reference_id"
                )
                if exclude_reference
                else None
            )
            full_scores = _score_queries(
                q_full, gallery, score_mode=score_mode, gallery_chunk_size=gallery_chunk_size
            )
            full_ranks = compute_target_ranks(
                full_scores, target_indices, reference_indices=reference_indices
            )
            hard_negative_indices = build_fixed_hard_negatives(
                full_scores,
                target_indices,
                hard_negative_k=hard_negative_k,
                reference_indices=reference_indices,
            )
            rows = torch.arange(len(sample_ids), device=device)
            full_margin = full_scores[rows, target_indices] - full_scores.gather(
                1, hard_negative_indices
            ).mean(dim=1)
            for row, sample_id in enumerate(sample_ids):
                cache[sample_id] = {
                    "category": category,
                    "reference_id": reference_ids[row],
                    "target_id": target_ids[row],
                    "teacher_full_rank": int(full_ranks[row].item()),
                    "teacher_qualified": bool(full_ranks[row].item() <= teacher_success_k),
                    "full_margin": float(full_margin[row].item()),
                    "hard_negative_ids": [
                        gallery_ids[i] for i in hard_negative_indices[row].tolist()
                    ],
                }
    if not cache:
        raise RuntimeError("TCFR cache is empty")
    if not any((bool(row["teacher_qualified"]) for row in cache.values())):
        raise RuntimeError("teacher-qualified coverage is zero; TCFR is unusable")
    return cache

@torch.no_grad()
def evaluate_stage1_edit_slots(
    *,
    model,
    val_loaders: Mapping[str, DataLoader],
    prepare_batch_fn,
    teacher_galleries: Mapping[str, tuple[Tensor, Mapping[str, int]]],
    gallery_ids_by_category: Mapping[str, Sequence[str]],
    tcfr_cache: Mapping[str, Mapping[str, object]],
    previous_anchor: Mapping[str, object] | None,
    config: Mapping[str, object],
    device: torch.device,
) -> tuple[dict[str, float | bool], dict[str, object]]:
    score_mode = str(_required(config, "teacher_score_mode"))
    exclude_reference = bool(_required(config, "exclude_reference_from_candidates"))
    gallery_chunk_size = int(config.get("gallery_chunk_size", 1024))
    gate_threshold = float(model.slot_gate_threshold)
    if "slot_gate_threshold" in config:
        if abs(float(config["slot_gate_threshold"]) - gate_threshold) > 1e-08:
            raise ValueError("evaluation gate threshold != model.slot_gate_threshold")
    model.eval()
    qualified_all, tcfr_all, soft_all = ([], [], [])
    harmful_all, rank_hurt_all, full_rank_all = ([], [], [])
    active_count_all, zero_all, all_active_all = ([], [], [])
    mask_pair_all, effect_pair_all, mass_all = ([], [], [])
    coverage_all = []
    coverage_available = True
    stability_records: dict[str, dict[str, Tensor]] = {}
    for category, val_loader in val_loaders.items():
        if category not in teacher_galleries:
            raise KeyError(f"Missing teacher gallery for category={category}")

        (teacher_gallery_features, teacher_gallery_name_to_idx) = teacher_galleries[category]
        _validate_gallery(teacher_gallery_features, teacher_gallery_name_to_idx, score_mode)
        _, gallery, local_index = (_get_category_gallery(category, gallery_ids_by_category, teacher_gallery_features, teacher_gallery_name_to_idx, device))
        for raw_batch in val_loader:
            if any(raw_batch.ground_truth_ids):
                raise NotImplementedError("V0 Stage-1 evaluator is FashionIQ-only")
            batch = prepare_batch_fn(raw_batch)
            out = model.build_edit_slots(
                reference_features=batch["reference_features"],
                text_states=batch["text_states"],
                text_attention_mask=batch["text_attention_mask"],
                text_content_mask=batch.get("text_content_mask"),
            )
            masks = out["slot_masks"]
            effects = out["slot_effects"]
            gates = out["slot_gates"]
            q_full = out["q_teacher_full"]
            q_minus = out["q_teacher_minus"]
            for name, tensor in {
                "slot_masks": masks,
                "slot_effects": effects,
                "slot_gates": gates,
                "q_teacher_full": q_full,
                "q_teacher_minus": q_minus,
            }.items():
                if not torch.isfinite(tensor).all():
                    raise ValueError(f"{name} contains NaN or Inf")
            b, l = gates.shape
            if q_full.shape != (b, model.teacher_query_dim):
                raise ValueError("unexpected q_teacher_full shape")
            if q_minus.shape != (b, l, model.teacher_query_dim):
                raise ValueError("unexpected q_teacher_minus shape")
            if q_full.shape[-1] != gallery.shape[-1]:
                raise ValueError("teacher query/gallery dimension mismatch")
            sample_ids = list(raw_batch.sample_ids)
            reference_ids = list(raw_batch.reference_ids)
            target_ids = []
            for target_id in raw_batch.target_ids:
                if target_id is None:
                    raise ValueError("validation sample is missing target_id")
                target_ids.append(target_id)
            if any((sample_id in stability_records for sample_id in sample_ids)):
                raise ValueError("duplicate validation sample_id")
            cached_rows = []
            for row, sample_id in enumerate(sample_ids):
                if sample_id not in tcfr_cache:
                    raise KeyError(f"TCFR cache missing sample={sample_id}")
                cached = tcfr_cache[sample_id]
                if cached["category"] != category or cached["reference_id"] != reference_ids[row] or cached["target_id"] != target_ids[row]:
                    raise ValueError(f"stale/misaligned TCFR cache for sample={sample_id}")
                cached_rows.append(cached)
            target_indices = _ids_to_indices(target_ids, local_index, device=device, field_name="target_id")
            reference_indices = (_ids_to_indices(reference_ids, local_index, device=device, field_name="reference_id")
                if exclude_reference
                else None
            )
            negative_indices = torch.tensor(
                [[local_index[x] for x in cached["hard_negative_ids"]] for cached in cached_rows],
                device=device,
            )
            full_margin = torch.tensor(
                [cached["full_margin"] for cached in cached_rows],
                dtype=torch.float32,
                device=device,
            )
            full_ranks = torch.tensor([cached["teacher_full_rank"] for cached in cached_rows], device=device)
            qualified = torch.tensor(
                [cached["teacher_qualified"] for cached in cached_rows],
                dtype=torch.bool,
                device=device,
            )
            candidate_indices = torch.cat([target_indices[:, None], negative_indices], dim=1)
            candidates = gallery[candidate_indices]
            full_candidate_scores = _score_paired_candidates(q_full[:, None, :], candidates, score_mode=score_mode)[:, 0]
            current_full_margin = full_candidate_scores[:, 0] - full_candidate_scores[:, 1:].mean(dim=1)
            if not torch.allclose(current_full_margin, full_margin, rtol=1e-4, atol=1e-5):
                max_error = float((current_full_margin - full_margin).abs().max().item())
                raise RuntimeError(
                    "TCFR cache is stale/inconsistent with current q_full; "
                    f"max margin error={max_error:.6g}"
                )
            candidate_scores = _score_paired_candidates(q_minus, candidates, score_mode=score_mode)
            minus_margin = candidate_scores[:, :, 0] - candidate_scores[:, :, 1:].mean(dim=2)
            tcfr = compute_tcfr(
                full_margin=full_margin,
                minus_margin=minus_margin,
                slot_gates=gates,
                gate_threshold=gate_threshold,
            )
            qualified_active = qualified[:, None] & tcfr["active_mask"]
            qualified_all.append(qualified.float().cpu())
            tcfr_all.append(tcfr["per_sample_tcfr"][qualified].cpu())
            soft_all.append(tcfr["per_sample_tcfr_soft"][qualified].cpu())
            harmful_all.append((tcfr["margin_drop"][qualified_active] < 0).float().cpu())
            full_rank_all.append(full_ranks.float().cpu())
            minus_scores = _score_queries(
                q_minus.reshape(b * l, model.teacher_query_dim),
                gallery,
                score_mode=score_mode,
                gallery_chunk_size=gallery_chunk_size,
            ).reshape(b, l, gallery.shape[0])
            minus_ranks = torch.stack(
                [
                    compute_target_ranks(
                        minus_scores[:, slot_id],
                        target_indices,
                        reference_indices=reference_indices,
                    )
                    for slot_id in range(l)
                ],
                dim=1,
            )
            rank_hurt_all.append(
                (minus_ranks - full_ranks[:, None])[qualified_active].float().cpu()
            )
            content_mask = batch.get("text_content_mask")
            if content_mask is None:
                coverage_available = False
            health = compute_stage1_health(
                slot_masks=masks,
                slot_effects=effects,
                slot_gates=gates,
                text_attention_mask=batch["text_attention_mask"],
                gate_threshold=gate_threshold,
                text_content_mask=content_mask,
            )
            active_count_all.append(health["active_count"].cpu())
            zero_all.append(health["zero_active"].cpu())
            all_active_all.append(health["all_active"].cpu())
            if health["mask_pair_cosines"].numel():
                mask_pair_all.append(health["mask_pair_cosines"].cpu())
            if health["effect_pair_cosines"].numel():
                effect_pair_all.append(health["effect_pair_cosines"].cpu())
            if health["active_slot_mass"].numel():
                mass_all.append(health["active_slot_mass"].cpu())
            if "content_union_coverage" in health:
                coverage_all.append(health["content_union_coverage"].cpu())
            for row, sample_id in enumerate(sample_ids):
                content_mask = batch.get("text_content_mask")
                if content_mask is not None:
                    valid_tokens = content_mask[row].bool()
                else:
                    valid_tokens = batch["text_attention_mask"][row].bool()
                if not valid_tokens.any():
                    raise ValueError(f"sample={sample_id} has no valid text token")
                stability_records[sample_id] = {
                    "slot_masks": masks[row, :, valid_tokens].float().cpu(),
                    "slot_effects": effects[row].float().cpu(),
                    "slot_gates": gates[row].float().cpu(),
                }
    qualified = _cat(qualified_all)
    qualified_count = int(qualified.sum().item())
    total_count = int(qualified.numel())
    if total_count == 0:
        raise RuntimeError("Stage-1 validation produced no samples")
    if qualified_count == 0:
        raise RuntimeError("teacher-qualified coverage is zero")
    tcfr_values = _cat(tcfr_all)
    if tcfr_values.numel() != qualified_count:
        raise RuntimeError("qualified TCFR denominator mismatch")
    health_all = {
        "active_count": _cat(active_count_all),
        "zero_active": _cat(zero_all),
        "all_active": _cat(all_active_all),
        "mask_pair_cosines": _cat(mask_pair_all),
        "effect_pair_cosines": _cat(effect_pair_all),
        "active_slot_mass": _cat(mass_all),
    }
    if coverage_available and coverage_all:
        health_all["content_union_coverage"] = _cat(coverage_all)
    metrics: dict[str, float | bool] = _summarize_health(
        health_all,
        overlap_margin=float(model.overlap_margin),
        effect_diversity_margin=float(model.effect_diversity_margin),
    )
    rank_hurt = _cat(rank_hurt_all)
    metrics.update(
        {
            "stage1/tcfr_margin_drop": _mean(tcfr_values),
            "stage1/tcfr_margin_drop_soft": _mean(_cat(soft_all)),
            "stage1/teacher_qualified_coverage": qualified_count / total_count,
            "stage1/teacher_full_rank_mean": _mean(_cat(full_rank_all)),
            "stage1/tcfr_rank_hurt_mean": _mean(rank_hurt),
            "stage1/tcfr_rank_hurt_median": float(rank_hurt.median().item())
            if rank_hurt.numel()
            else 0.0,
            "stage1/tcfr_rank_hurt_positive_rate": _mean((rank_hurt > 0).float()),
            "stage1/harmful_active_slot_rate": _mean(_cat(harmful_all)),
            "stage1/teacher_qualified_count": float(qualified_count),
            "stage1/validation_count": float(total_count),
            "stage1/stability_available": False,
        }
    )
    current_anchor: dict[str, object] = {
        "sample_ids": sorted(stability_records),
        "records": {
            sample_id: stability_records[sample_id] for sample_id in sorted(stability_records)
        },
    }
    if previous_anchor is not None:
        metrics.update(
            compute_esss(
                current_anchor=current_anchor,
                previous_anchor=previous_anchor,
                gate_threshold=gate_threshold,
            )
        )
    metrics["stage1/health_ok"] = _health_ok(metrics, config)
    for key in (
        "stage1/tcfr_margin_drop",
        "stage1/teacher_qualified_coverage",
        "stage1/zero_active_rate",
        "stage1/all_active_rate",
        "stage1/harmful_active_slot_rate",
    ):
        if not torch.isfinite(torch.tensor(float(metrics[key]))):
            raise FloatingPointError(f"non-finite Stage-1 metric: {key}")
    return (metrics, current_anchor)
