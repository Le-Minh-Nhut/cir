from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

if TYPE_CHECKING:
    from datasets.fashioniq import FashionIQDataset
    from teachers.csmcir import CSMCIRStage1Teacher


CATEGORIES = ("dress", "shirt", "toptee")


@dataclass
class ProbeConfig:
    split: str
    max_samples: int
    seed: int
    max_change_patches: int
    pos_threshold: float
    neg_threshold: float
    min_text_effect_norm: float
    min_change_effect_norm: float
    lmax: int
    control_every: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "R0 probe for latent text<->reference-to-target change relations. "
            "This script does NOT train edit slots."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("data/FashionIQ"))
    parser.add_argument("--csmcir-root", type=Path, default=Path("teacher/repos/CSMCIR"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("teacher/checkpoints/csmcir/fashioniq_tuned_clip_best.pt"),
    )
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--categories", nargs="+", choices=CATEGORIES, default=list(CATEGORIES))
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-change-patches", type=int, default=24)

    # Deliberately conservative. Scores in (neg_threshold, pos_threshold) abstain.
    # These are diagnostic thresholds, NOT calibrated semantic labels.
    parser.add_argument("--pos-threshold", type=float, default=0.35)
    parser.add_argument("--neg-threshold", type=float, default=-0.35)
    parser.add_argument("--min-text-effect-norm", type=float, default=1e-4)
    parser.add_argument("--min-change-effect-norm", type=float, default=1e-4)

    parser.add_argument("--lmax", type=int, default=4)
    parser.add_argument(
        "--control-every",
        type=int,
        default=0,
        help=(
            "If >0, every Nth sample also uses a wrong same-category target as a negative control. "
            "0 disables the extra compute."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/stage1_relation_probe/csmcir_dual_counterfactual"),
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def resolve_image_path(image_root: Path, image_id: str, category: str) -> Path:
    candidates: list[Path] = []
    for ext in (".jpg", ".png", ".jpeg"):
        candidates.append(image_root / category / f"{image_id}{ext}")
    for ext in (".jpg", ".png", ".jpeg"):
        candidates.append(image_root / f"{image_id}{ext}")
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Could not find image: category={category}, id={image_id}")


def load_image_tensor(
    *,
    image_root: Path,
    image_id: str,
    category: str,
    preprocess,
) -> torch.Tensor:
    path = resolve_image_path(image_root, image_id, category)
    with Image.open(path) as image:
        return preprocess(image.convert("RGB"))


def infer_spatial_indices(num_vision_tokens: int, device: torch.device) -> torch.Tensor:
    """
    CSMCIR's ViT often exposes CLS + HxW patch tokens.
    If K-1 is a perfect square, skip index 0 as CLS. Otherwise use all tokens.
    """
    if num_vision_tokens < 1:
        raise ValueError("num_vision_tokens must be >= 1")

    candidate = num_vision_tokens - 1
    side = int(math.isqrt(candidate))
    if candidate > 0 and side * side == candidate:
        return torch.arange(1, num_vision_tokens, device=device)

    return torch.arange(num_vision_tokens, device=device)


@torch.inference_mode()
def compose_with_image_mask(
    *,
    teacher: CSMCIRStage1Teacher,
    image_features: torch.Tensor,
    text_states: torch.Tensor,
    text_attention_mask: torch.Tensor,
    image_token_mask: torch.Tensor,
    normalize: bool = False,
) -> torch.Tensor:
    """
    Same CSMCIR Q-Former path as teacher.compose(), except the image-token
    encoder_attention_mask is supplied explicitly so we can do patch deletion.

    This intentionally lives in the probe first: R0 is an audit, not a permanent API.
    """
    if image_features.ndim != 3:
        raise ValueError("image_features must be [B,K,D]")
    if text_states.ndim != 3:
        raise ValueError("text_states must be [B,N,D]")
    if text_attention_mask.shape != text_states.shape[:2]:
        raise ValueError("text_attention_mask shape mismatch")
    if image_token_mask.shape != image_features.shape[:2]:
        raise ValueError("image_token_mask shape mismatch")

    batch_size = image_features.shape[0]
    model = teacher.model
    device = image_features.device

    query_tokens = model.query_tokens.expand(batch_size, -1, -1)
    query_atts = torch.ones(
        query_tokens.shape[:-1],
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.cat(
        [query_atts, text_attention_mask.to(device=device, dtype=torch.long)],
        dim=1,
    )

    output = model.Qformer.bert(
        inputs_embeds=text_states,
        query_embeds=query_tokens,
        attention_mask=attention_mask,
        encoder_hidden_states=image_features,
        encoder_attention_mask=image_token_mask.to(device=device, dtype=torch.long),
        return_dict=True,
    )

    num_query_tokens = query_tokens.shape[1]
    q_pre = model.text_proj(
        output.last_hidden_state[:, num_query_tokens, :]
    )

    if not torch.isfinite(q_pre).all():
        raise FloatingPointError("compose_with_image_mask produced NaN/Inf")

    if normalize:
        q_pre = F.normalize(q_pre, dim=-1)
    return q_pre


@torch.inference_mode()
def encode_text_with_token_ids(
    teacher: CSMCIRStage1Teacher,
    text: str,
) -> dict[str, Any]:
    processed = teacher.txt_processor(text)
    tokenizer = teacher.model.tokenizer

    tokens = tokenizer(
        [processed],
        padding="max_length",
        truncation=True,
        max_length=teacher.model.max_txt_len,
        return_tensors="pt",
    )

    device = next(teacher.model.parameters()).device
    tokens = tokens.to(device)

    text_states = teacher.model.Qformer.bert.embeddings.word_embeddings(
        tokens.input_ids
    )
    attention_mask = tokens.attention_mask.bool()

    special_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
    for special_id in tokenizer.all_special_ids:
        special_mask |= tokens.input_ids == special_id

    content_mask = attention_mask & ~special_mask
    token_strings = tokenizer.convert_ids_to_tokens(
        tokens.input_ids[0].detach().cpu().tolist()
    )

    return {
        "processed_text": processed,
        "text_states": text_states,
        "attention_mask": attention_mask,
        "content_mask": content_mask,
        "input_ids": tokens.input_ids,
        "token_strings": token_strings,
    }


@torch.inference_mode()
def compute_text_deletion_effects(
    *,
    teacher: CSMCIRStage1Teacher,
    reference_features: torch.Tensor,
    text_states: torch.Tensor,
    text_attention_mask: torch.Tensor,
    content_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    c_n = q_full(Ir,m) - q_full(Ir,m with token n masked out)

    Returns:
        token_indices [Nt]
        effects       [Nt,Dq]
    """
    token_indices = content_mask[0].nonzero(as_tuple=False).squeeze(1)
    if token_indices.numel() == 0:
        return token_indices, torch.empty(
            0,
            teacher.model.text_proj.out_features,
            device=reference_features.device,
        )

    q_full = teacher.compose(
        reference_features,
        text_states,
        text_attention_mask,
        normalize=False,
    )

    count = int(token_indices.numel())
    ref_rep = reference_features.expand(count, -1, -1)
    text_rep = text_states.expand(count, -1, -1)
    mask_rep = text_attention_mask.expand(count, -1).clone()

    rows = torch.arange(count, device=reference_features.device)
    mask_rep[rows, token_indices] = False

    q_minus = teacher.compose(
        ref_rep,
        text_rep,
        mask_rep,
        normalize=False,
    )
    effects = q_full.expand_as(q_minus) - q_minus
    return token_indices, effects


@torch.inference_mode()
def match_and_select_change_patches(
    *,
    reference_features: torch.Tensor,
    target_features: torch.Tensor,
    max_change_patches: int,
) -> dict[str, torch.Tensor]:
    """
    Match each target patch to its nearest reference patch in frozen vision space.
    Then keep patches with largest raw mismatch (1 - cosine).

    This is only a proposal filter. It is not treated as edit ground truth.
    """
    if reference_features.shape[0] != 1 or target_features.shape[0] != 1:
        raise ValueError("R0 probe currently expects batch size 1 per sample")

    device = reference_features.device
    ref_ids = infer_spatial_indices(reference_features.shape[1], device)
    tgt_ids = infer_spatial_indices(target_features.shape[1], device)

    ref = F.normalize(reference_features[0, ref_ids].float(), dim=-1)
    tgt = F.normalize(target_features[0, tgt_ids].float(), dim=-1)

    similarity = tgt @ ref.T
    best_sim, best_local_ref = similarity.max(dim=1)
    matched_ref_ids = ref_ids[best_local_ref]

    raw_change_score = 1.0 - best_sim
    keep = min(max_change_patches, int(tgt_ids.numel()))
    if keep < 1:
        raise ValueError("max_change_patches must retain at least one patch")

    top = torch.topk(raw_change_score, k=keep, largest=True).indices

    return {
        "target_patch_ids": tgt_ids[top],
        "reference_patch_ids": matched_ref_ids[top],
        "match_cosine": best_sim[top],
        "raw_change_score": raw_change_score[top],
    }


@torch.inference_mode()
def compute_differential_patch_effects(
    *,
    teacher: CSMCIRStage1Teacher,
    reference_features: torch.Tensor,
    target_features: torch.Tensor,
    text_states: torch.Tensor,
    text_attention_mask: torch.Tensor,
    target_patch_ids: torch.Tensor,
    reference_patch_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    For each selected target patch q matched to reference patch p:

      e_t(q) = Q(It,m) - Q(It with q masked,m)
      e_r(p) = Q(Ir,m) - Q(Ir with p masked,m)
      d_q    = e_t(q) - e_r(p)

    All effects live in the same frozen teacher query space.

    The subtraction is deliberate: generic text-conditioned patch grounding that
    already existed in the reference should cancel; change-specific sensitivity
    should remain.
    """
    count = int(target_patch_ids.numel())
    if count == 0:
        raise ValueError("No selected target patches")
    if reference_patch_ids.shape != target_patch_ids.shape:
        raise ValueError("target/reference patch ids must have same shape")

    Kt = target_features.shape[1]
    Kr = reference_features.shape[1]
    device = target_features.device

    target_mask_full = torch.ones(1, Kt, dtype=torch.bool, device=device)
    reference_mask_full = torch.ones(1, Kr, dtype=torch.bool, device=device)

    q_target_full = compose_with_image_mask(
        teacher=teacher,
        image_features=target_features,
        text_states=text_states,
        text_attention_mask=text_attention_mask,
        image_token_mask=target_mask_full,
        normalize=False,
    )
    q_reference_full = compose_with_image_mask(
        teacher=teacher,
        image_features=reference_features,
        text_states=text_states,
        text_attention_mask=text_attention_mask,
        image_token_mask=reference_mask_full,
        normalize=False,
    )

    target_rep = target_features.expand(count, -1, -1)
    reference_rep = reference_features.expand(count, -1, -1)
    text_rep = text_states.expand(count, -1, -1)
    text_mask_rep = text_attention_mask.expand(count, -1)

    target_masks = torch.ones(count, Kt, dtype=torch.bool, device=device)
    reference_masks = torch.ones(count, Kr, dtype=torch.bool, device=device)

    rows = torch.arange(count, device=device)
    target_masks[rows, target_patch_ids] = False
    reference_masks[rows, reference_patch_ids] = False

    q_target_minus = compose_with_image_mask(
        teacher=teacher,
        image_features=target_rep,
        text_states=text_rep,
        text_attention_mask=text_mask_rep,
        image_token_mask=target_masks,
        normalize=False,
    )
    q_reference_minus = compose_with_image_mask(
        teacher=teacher,
        image_features=reference_rep,
        text_states=text_rep,
        text_attention_mask=text_mask_rep,
        image_token_mask=reference_masks,
        normalize=False,
    )

    target_effect = q_target_full.expand_as(q_target_minus) - q_target_minus
    reference_effect = q_reference_full.expand_as(q_reference_minus) - q_reference_minus
    differential_effect = target_effect - reference_effect

    return {
        "target_effect": target_effect,
        "reference_effect": reference_effect,
        "differential_effect": differential_effect,
    }


def cosine_relation_matrix(
    text_effects: torch.Tensor,
    change_effects: torch.Tensor,
) -> torch.Tensor:
    if text_effects.ndim != 2 or change_effects.ndim != 2:
        raise ValueError("effects must be [N,D] and [Q,D]")
    if text_effects.shape[1] != change_effects.shape[1]:
        raise ValueError("effect dimensions must match")
    if text_effects.numel() == 0 or change_effects.numel() == 0:
        return torch.empty(
            text_effects.shape[0],
            change_effects.shape[0],
            device=text_effects.device,
        )

    t = F.normalize(text_effects.float(), dim=-1, eps=1e-12)
    c = F.normalize(change_effects.float(), dim=-1, eps=1e-12)
    return t @ c.T


def label_relation_edges(
    *,
    scores: np.ndarray,
    text_effect_norms: np.ndarray,
    change_effect_norms: np.ndarray,
    pos_threshold: float,
    neg_threshold: float,
    min_text_effect_norm: float,
    min_change_effect_norm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    labels:
      +1 = candidate same-edit
       0 = abstain
      -1 = candidate different-edit / cannot-link

    IMPORTANT:
    A merely low score is NOT negative.
    Only a genuinely negative cosine below neg_threshold is used as a candidate
    cannot-link. Thresholds are diagnostics until calibrated on controlled data.
    """
    if not neg_threshold < 0 < pos_threshold:
        raise ValueError("Require neg_threshold < 0 < pos_threshold")

    scores = np.asarray(scores, dtype=np.float64)
    tnorm = np.asarray(text_effect_norms, dtype=np.float64)
    cnorm = np.asarray(change_effect_norms, dtype=np.float64)

    valid = (
        (tnorm[:, None] >= min_text_effect_norm)
        & (cnorm[None, :] >= min_change_effect_norm)
        & np.isfinite(scores)
    )

    labels = np.zeros(scores.shape, dtype=np.int8)
    labels[valid & (scores >= pos_threshold)] = 1
    labels[valid & (scores <= neg_threshold)] = -1

    # Confidence is deliberately local: strong absolute functional agreement,
    # tempered by effect norms. No row-centering / fake negatives.
    scale = max(abs(pos_threshold), abs(neg_threshold), 1e-6)
    score_conf = np.clip(np.abs(scores) / scale, 0.0, 1.0)

    tscale = np.median(tnorm[tnorm > 0]) if np.any(tnorm > 0) else 1.0
    cscale = np.median(cnorm[cnorm > 0]) if np.any(cnorm > 0) else 1.0
    tconf = np.clip(tnorm / max(tscale, 1e-12), 0.0, 1.0)
    cconf = np.clip(cnorm / max(cscale, 1e-12), 0.0, 1.0)

    weights = score_conf * np.sqrt(tconf[:, None] * cconf[None, :])
    weights[labels == 0] = 0.0
    return labels, weights


def bcc_disagreement(
    labels: np.ndarray,
    weights: np.ndarray,
    text_assignment: np.ndarray,
    patch_assignment: np.ndarray,
) -> float:
    labels = np.asarray(labels)
    weights = np.asarray(weights, dtype=np.float64)
    text_assignment = np.asarray(text_assignment, dtype=np.int64)
    patch_assignment = np.asarray(patch_assignment, dtype=np.int64)

    same = text_assignment[:, None] == patch_assignment[None, :]
    positive_cut = (labels > 0) & (~same)
    negative_merged = (labels < 0) & same
    return float(weights[positive_cut | negative_merged].sum())


def _kmeans(
    X: np.ndarray,
    k: int,
    *,
    seed: int,
    restarts: int = 16,
    max_iter: int = 100,
) -> tuple[np.ndarray, float]:
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    if not 1 <= k <= n:
        raise ValueError("k must be in [1,n]")
    if k == 1:
        labels = np.zeros(n, dtype=np.int64)
        center = X.mean(axis=0, keepdims=True)
        return labels, float(((X - center) ** 2).sum())

    rng = np.random.default_rng(seed)
    best_labels: np.ndarray | None = None
    best_sse = float("inf")

    for _ in range(restarts):
        centers = X[rng.choice(n, size=k, replace=False)].copy()
        labels = np.zeros(n, dtype=np.int64)

        for _ in range(max_iter):
            distance = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            new_labels = distance.argmin(axis=1)

            # Repair empty clusters by assigning farthest points.
            used = set(new_labels.tolist())
            missing = [cluster for cluster in range(k) if cluster not in used]
            if missing:
                nearest = distance[np.arange(n), new_labels]
                order = np.argsort(-nearest)
                cursor = 0
                for cluster in missing:
                    while cursor < len(order) and np.sum(new_labels == new_labels[order[cursor]]) <= 1:
                        cursor += 1
                    point = order[min(cursor, len(order) - 1)]
                    new_labels[point] = cluster
                    cursor += 1

            new_centers = np.empty_like(centers)
            for cluster in range(k):
                pts = X[new_labels == cluster]
                if len(pts) == 0:
                    new_centers[cluster] = centers[cluster]
                else:
                    new_centers[cluster] = pts.mean(axis=0)

            if np.array_equal(new_labels, labels):
                centers = new_centers
                labels = new_labels
                break
            labels = new_labels
            centers = new_centers

        sse = float(((X - centers[labels]) ** 2).sum())
        if sse < best_sse:
            best_sse = sse
            best_labels = labels.copy()

    assert best_labels is not None
    return best_labels, best_sse


def signed_svd_bcc_scan(
    *,
    labels: np.ndarray,
    weights: np.ndarray,
    lmax: int,
    seed: int,
) -> dict[str, Any]:
    """
    Diagnostic only.

    Build A = Y * W, get joint text/patch coordinates from SVD, cluster the
    stacked coordinates for K=1..Lmax, then choose the lowest weighted BCC
    disagreement. This is an initializer/proposal, not the semantic objective.
    """
    labels = np.asarray(labels, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    n_text, n_patch = labels.shape
    if n_text == 0 or n_patch == 0:
        return {"available": False, "reason": "empty_relation_matrix"}

    A = labels * weights
    if np.allclose(A, 0.0):
        return {"available": False, "reason": "all_abstain"}

    U, singular_values, Vt = np.linalg.svd(A, full_matrices=False)
    max_k = min(lmax, n_text + n_patch)

    candidates: list[dict[str, Any]] = []
    for k in range(1, max_k + 1):
        dim = min(k, len(singular_values))
        sqrt_s = np.sqrt(np.maximum(singular_values[:dim], 0.0))
        text_coord = U[:, :dim] * sqrt_s[None, :]
        patch_coord = Vt.T[:, :dim] * sqrt_s[None, :]
        X = np.vstack([text_coord, patch_coord])

        joint_assignment, _ = _kmeans(
            X,
            k,
            seed=seed + 997 * k,
        )
        text_assignment = joint_assignment[:n_text]
        patch_assignment = joint_assignment[n_text:]

        # TAPER edit slots should be joint text/change units.
        used = set(joint_assignment.tolist())
        modality_valid = all(
            np.any(text_assignment == cluster) and np.any(patch_assignment == cluster)
            for cluster in used
        )

        disagreement = bcc_disagreement(
            labels,
            weights,
            text_assignment,
            patch_assignment,
        )
        normalized = disagreement / max(float(weights[labels != 0].sum()), 1e-12)

        candidates.append(
            {
                "k": k,
                "weighted_disagreement": disagreement,
                "normalized_disagreement": normalized,
                "modality_valid": bool(modality_valid),
                "text_assignment": text_assignment.tolist(),
                "patch_assignment": patch_assignment.tolist(),
            }
        )

    valid = [c for c in candidates if c["modality_valid"]]
    pool = valid if valid else candidates
    best = min(
        pool,
        key=lambda c: (c["weighted_disagreement"], c["k"]),
    )
    return {
        "available": True,
        "best": best,
        "candidates": candidates,
        "singular_values": singular_values[: min(8, len(singular_values))].tolist(),
    }


class DisjointSet:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def relation_graph_diagnostics(
    labels: np.ndarray,
) -> dict[str, Any]:
    labels = np.asarray(labels)
    n_text, n_patch = labels.shape
    dsu = DisjointSet(n_text + n_patch)

    positive_edges = np.argwhere(labels > 0)
    negative_edges = np.argwhere(labels < 0)

    positive_incident = set()
    for ti, pj in positive_edges:
        left = int(ti)
        right = n_text + int(pj)
        dsu.union(left, right)
        positive_incident.add(left)
        positive_incident.add(right)

    roots = sorted({dsu.find(node) for node in positive_incident})
    root_to_component = {root: i for i, root in enumerate(roots)}
    node_component = {
        node: root_to_component[dsu.find(node)]
        for node in positive_incident
    }

    separated_pairs: set[tuple[int, int]] = set()
    for ti, pj in negative_edges:
        left = int(ti)
        right = n_text + int(pj)
        if left not in node_component or right not in node_component:
            continue
        a = node_component[left]
        b = node_component[right]
        if a == b:
            continue
        separated_pairs.add(tuple(sorted((a, b))))

    num_components = len(roots)
    total_pairs = num_components * (num_components - 1) // 2
    separator_coverage = (
        len(separated_pairs) / total_pairs
        if total_pairs > 0
        else None
    )

    return {
        "num_positive_edges": int(len(positive_edges)),
        "num_negative_edges": int(len(negative_edges)),
        "num_abstain_edges": int((labels == 0).sum()),
        "num_positive_components": num_components,
        "negative_separated_component_pairs": len(separated_pairs),
        "possible_component_pairs": total_pairs,
        "negative_separator_coverage": separator_coverage,
    }


def choose_indices(
    dataset: FashionIQDataset,
    *,
    max_samples: int,
    seed: int,
) -> list[int]:
    if max_samples < 1:
        raise ValueError("--max-samples must be >= 1")
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    return indices[: min(max_samples, len(indices))]


def build_wrong_target_lookup(dataset: FashionIQDataset) -> dict[int, int]:
    by_category: dict[str, list[int]] = {category: [] for category in CATEGORIES}
    for index, annotation in enumerate(dataset.annotations):
        by_category[annotation.category].append(index)

    lookup: dict[int, int] = {}
    for category, indices in by_category.items():
        if len(indices) < 2:
            continue
        for pos, index in enumerate(indices):
            lookup[index] = indices[(pos + 1) % len(indices)]
    return lookup


@torch.inference_mode()
def run_one_relation_probe(
    *,
    teacher: CSMCIRStage1Teacher,
    image_root: Path,
    reference_id: str,
    target_id: str,
    category: str,
    modification_text: str,
    max_change_patches: int,
    pos_threshold: float,
    neg_threshold: float,
    min_text_effect_norm: float,
    min_change_effect_norm: float,
    lmax: int,
    seed: int,
) -> dict[str, Any]:
    device = next(teacher.model.parameters()).device

    images = torch.stack(
        [
            load_image_tensor(
                image_root=image_root,
                image_id=reference_id,
                category=category,
                preprocess=teacher.preprocess,
            ),
            load_image_tensor(
                image_root=image_root,
                image_id=target_id,
                category=category,
                preprocess=teacher.preprocess,
            ),
        ]
    ).to(device)

    vision = teacher.encode_reference(images)
    reference_features = vision[0:1]
    target_features = vision[1:2]

    text = encode_text_with_token_ids(teacher, modification_text)
    text_states = text["text_states"]
    attention_mask = text["attention_mask"]
    content_mask = text["content_mask"]

    token_indices, text_effects = compute_text_deletion_effects(
        teacher=teacher,
        reference_features=reference_features,
        text_states=text_states,
        text_attention_mask=attention_mask,
        content_mask=content_mask,
    )

    patch_match = match_and_select_change_patches(
        reference_features=reference_features,
        target_features=target_features,
        max_change_patches=max_change_patches,
    )

    patch_effect = compute_differential_patch_effects(
        teacher=teacher,
        reference_features=reference_features,
        target_features=target_features,
        text_states=text_states,
        text_attention_mask=attention_mask,
        target_patch_ids=patch_match["target_patch_ids"],
        reference_patch_ids=patch_match["reference_patch_ids"],
    )

    differential = patch_effect["differential_effect"]
    relation = cosine_relation_matrix(text_effects, differential)

    text_norm = torch.linalg.vector_norm(text_effects.float(), dim=-1)
    change_norm = torch.linalg.vector_norm(differential.float(), dim=-1)

    relation_np = relation.detach().cpu().numpy()
    text_norm_np = text_norm.detach().cpu().numpy()
    change_norm_np = change_norm.detach().cpu().numpy()

    edge_labels, edge_weights = label_relation_edges(
        scores=relation_np,
        text_effect_norms=text_norm_np,
        change_effect_norms=change_norm_np,
        pos_threshold=pos_threshold,
        neg_threshold=neg_threshold,
        min_text_effect_norm=min_text_effect_norm,
        min_change_effect_norm=min_change_effect_norm,
    )

    graph = relation_graph_diagnostics(edge_labels)
    svd = signed_svd_bcc_scan(
        labels=edge_labels,
        weights=edge_weights,
        lmax=lmax,
        seed=seed,
    )

    token_strings = text["token_strings"]
    selected_tokens = [
        token_strings[int(index)]
        for index in token_indices.detach().cpu().tolist()
    ]

    # Top edge summaries are useful for fast human inspection without opening
    # the full score matrix.
    flat: list[tuple[float, int, int]] = []
    for ti in range(relation_np.shape[0]):
        for pj in range(relation_np.shape[1]):
            flat.append((float(relation_np[ti, pj]), ti, pj))

    top_positive = sorted(flat, reverse=True)[: min(8, len(flat))]
    top_negative = sorted(flat)[: min(8, len(flat))]

    def render_edge(item: tuple[float, int, int]) -> dict[str, Any]:
        score, ti, pj = item
        return {
            "score": score,
            "token": selected_tokens[ti],
            "token_position": int(token_indices[ti]),
            "target_patch_id": int(patch_match["target_patch_ids"][pj]),
            "reference_patch_id": int(patch_match["reference_patch_ids"][pj]),
            "raw_change_score": float(patch_match["raw_change_score"][pj]),
            "label": int(edge_labels[ti, pj]),
            "weight": float(edge_weights[ti, pj]),
        }

    non_abstain = int((edge_labels != 0).sum())
    total_edges = int(edge_labels.size)

    return {
        "reference_id": reference_id,
        "target_id": target_id,
        "category": category,
        "modification_text": modification_text,
        "processed_text": text["processed_text"],
        "tokens": selected_tokens,
        "token_positions": token_indices.detach().cpu().tolist(),
        "target_patch_ids": patch_match["target_patch_ids"].detach().cpu().tolist(),
        "reference_patch_ids": patch_match["reference_patch_ids"].detach().cpu().tolist(),
        "match_cosine": patch_match["match_cosine"].detach().cpu().tolist(),
        "raw_change_score": patch_match["raw_change_score"].detach().cpu().tolist(),
        "text_effect_norm": text_norm_np.tolist(),
        "change_effect_norm": change_norm_np.tolist(),
        "relation_scores": relation_np.tolist(),
        "edge_labels": edge_labels.tolist(),
        "edge_weights": edge_weights.tolist(),
        "edge_stats": {
            "total": total_edges,
            "positive_fraction": float((edge_labels > 0).sum() / max(total_edges, 1)),
            "negative_fraction": float((edge_labels < 0).sum() / max(total_edges, 1)),
            "abstain_fraction": float((edge_labels == 0).sum() / max(total_edges, 1)),
            "non_abstain": non_abstain,
        },
        "graph": graph,
        "signed_svd_bcc": svd,
        "top_positive": [render_edge(x) for x in top_positive],
        "top_negative": [render_edge(x) for x in top_negative],
    }


def aggregate_summary(
    records: list[dict[str, Any]],
    controls: list[dict[str, Any]],
    config: ProbeConfig,
) -> dict[str, Any]:
    if not records:
        raise RuntimeError("No relation probe records")

    def mean(path: tuple[str, ...]) -> float:
        vals = []
        for record in records:
            obj: Any = record
            for key in path:
                obj = obj[key]
            if obj is not None:
                vals.append(float(obj))
        return float(np.mean(vals)) if vals else float("nan")

    k_hist: Counter[str] = Counter()
    bcc_available = 0
    for record in records:
        proposal = record["signed_svd_bcc"]
        if proposal["available"]:
            bcc_available += 1
            k_hist[str(proposal["best"]["k"])] += 1

    control_summary: dict[str, Any] | None = None
    if controls:
        def extract_metric(rs: list[dict[str, Any]], key: str) -> list[float]:
            out = []
            for x in rs:
                proposal = x["signed_svd_bcc"]
                if proposal["available"]:
                    out.append(float(proposal["best"][key]))
            return out

        true_loss = extract_metric(
            [c["true"] for c in controls],
            "normalized_disagreement",
        )
        wrong_loss = extract_metric(
            [c["wrong"] for c in controls],
            "normalized_disagreement",
        )

        pairs = min(len(true_loss), len(wrong_loss))
        control_summary = {
            "num_controls": len(controls),
            "mean_true_best_normalized_bcc": float(np.mean(true_loss)) if true_loss else None,
            "mean_wrong_best_normalized_bcc": float(np.mean(wrong_loss)) if wrong_loss else None,
            "note": (
                "Lower true-target BCC disagreement than wrong-target is desirable, "
                "but this is only a negative control, not proof of semantic factors."
            ),
        }

    return {
        "probe": "csmcir_dual_counterfactual_relation_r0",
        "config": asdict(config),
        "num_samples": len(records),
        "mean_positive_fraction": mean(("edge_stats", "positive_fraction")),
        "mean_negative_fraction": mean(("edge_stats", "negative_fraction")),
        "mean_abstain_fraction": mean(("edge_stats", "abstain_fraction")),
        "mean_positive_component_count": mean(("graph", "num_positive_components")),
        "mean_negative_separator_coverage": mean(("graph", "negative_separator_coverage")),
        "bcc_available_fraction": bcc_available / len(records),
        "candidate_k_histogram": dict(sorted(k_hist.items())),
        "wrong_target_control": control_summary,
        "interpretation_rules": [
            "Do not treat low positive similarity as a negative edge.",
            "High abstention is acceptable if retained edges are precise.",
            "Positive components without negative separators are not identifiable as separate edits.",
            "A low BCC disagreement is not sufficient; shuffled/wrong-target controls must degrade.",
            "This probe does not train slots and does not create hard pseudo coalition labels.",
        ],
    }


def self_test() -> None:
    cpu = torch.device("cpu")

    assert infer_spatial_indices(257, cpu)[0].item() == 1
    assert len(infer_spatial_indices(257, cpu)) == 256
    assert infer_spatial_indices(256, cpu)[0].item() == 0

    scores = np.array(
        [
            [0.8, -0.8, 0.1],
            [-0.7, 0.7, -0.1],
        ]
    )
    labels, weights = label_relation_edges(
        scores=scores,
        text_effect_norms=np.array([1.0, 1.0]),
        change_effect_norms=np.array([1.0, 1.0, 1.0]),
        pos_threshold=0.35,
        neg_threshold=-0.35,
        min_text_effect_norm=1e-4,
        min_change_effect_norm=1e-4,
    )
    assert labels.tolist() == [[1, -1, 0], [-1, 1, 0]]
    assert np.all(weights[labels == 0] == 0)

    # Strongly positive K=1 must stay valid.
    k1 = np.ones((2, 3), dtype=np.int8)
    k1w = np.ones_like(k1, dtype=float)
    assert bcc_disagreement(
        k1,
        k1w,
        np.zeros(2, dtype=int),
        np.zeros(3, dtype=int),
    ) == 0.0

    # Clean K=2: giant pays negative cross-factor edges.
    y = np.array([[1, -1], [-1, 1]], dtype=np.int8)
    w = np.ones_like(y, dtype=float)
    giant = bcc_disagreement(
        y, w,
        np.array([0, 0]),
        np.array([0, 0]),
    )
    correct = bcc_disagreement(
        y, w,
        np.array([0, 1]),
        np.array([0, 1]),
    )
    assert giant == 2.0
    assert correct == 0.0

    proposal = signed_svd_bcc_scan(
        labels=y,
        weights=w,
        lmax=3,
        seed=7,
    )
    assert proposal["available"]
    assert proposal["best"]["k"] == 2
    assert proposal["best"]["weighted_disagreement"] == 0.0

    graph = relation_graph_diagnostics(y)
    assert graph["num_positive_components"] == 2
    assert graph["negative_separator_coverage"] == 1.0

    # All-abstain must not invent a K.
    none = signed_svd_bcc_scan(
        labels=np.zeros((2, 2), dtype=np.int8),
        weights=np.zeros((2, 2), dtype=float),
        lmax=4,
        seed=1,
    )
    assert not none["available"]
    assert none["reason"] == "all_abstain"

    print("ALL R0 RELATION-PROBE SELF TESTS PASSED")


def main() -> None:
    args = parse_args()

    if args.self_test:
        self_test()
        return

    from datasets.fashioniq import FashionIQDataset
    from teachers.csmcir import CSMCIRStage1Teacher

    if not args.neg_threshold < 0 < args.pos_threshold:
        raise ValueError("--neg-threshold must be <0 and --pos-threshold >0")
    if args.max_change_patches < 1:
        raise ValueError("--max-change-patches must be >=1")
    if args.lmax < 1:
        raise ValueError("--lmax must be >=1")
    if args.control_every < 0:
        raise ValueError("--control-every must be >=0")

    dataset_root = args.dataset_root.resolve()
    annotation_root = dataset_root / "captions"
    image_root = dataset_root / "images"

    dataset = FashionIQDataset(
        annotation_root=annotation_root,
        split=args.split,
        categories=args.categories,
        caption_policy="ordered_and",
        seed=args.seed,
    )
    selected_indices = choose_indices(
        dataset,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    wrong_lookup = build_wrong_target_lookup(dataset)

    device = torch.device(args.device)
    teacher = CSMCIRStage1Teacher(
        csmcir_root=args.csmcir_root,
        checkpoint_path=args.checkpoint,
        device=args.device,
    ).to(device).eval()

    config = ProbeConfig(
        split=args.split,
        max_samples=len(selected_indices),
        seed=args.seed,
        max_change_patches=args.max_change_patches,
        pos_threshold=args.pos_threshold,
        neg_threshold=args.neg_threshold,
        min_text_effect_norm=args.min_text_effect_norm,
        min_change_effect_norm=args.min_change_effect_norm,
        lmax=args.lmax,
        control_every=args.control_every,
    )

    output_dir = args.output_root / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / "samples.jsonl"
    controls_path = output_dir / "wrong_target_controls.jsonl"
    summary_path = output_dir / "summary.json"

    records: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []

    with samples_path.open("w", encoding="utf-8") as samples_file:
        controls_file = (
            controls_path.open("w", encoding="utf-8")
            if args.control_every > 0
            else None
        )
        try:
            for ordinal, index in enumerate(
                tqdm(selected_indices, desc="R0 relation probe"),
                start=1,
            ):
                sample = dataset[index]
                if sample.target_id is None or sample.category is None:
                    raise RuntimeError("FashionIQ train/val sample requires target/category")

                record = run_one_relation_probe(
                    teacher=teacher,
                    image_root=image_root,
                    reference_id=sample.reference_id,
                    target_id=sample.target_id,
                    category=sample.category,
                    modification_text=sample.modification_text,
                    max_change_patches=args.max_change_patches,
                    pos_threshold=args.pos_threshold,
                    neg_threshold=args.neg_threshold,
                    min_text_effect_norm=args.min_text_effect_norm,
                    min_change_effect_norm=args.min_change_effect_norm,
                    lmax=args.lmax,
                    seed=args.seed + index,
                )
                record["sample_id"] = sample.sample_id
                record["probe_index"] = index

                records.append(record)
                samples_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                samples_file.flush()

                if (
                    args.control_every > 0
                    and ordinal % args.control_every == 0
                    and index in wrong_lookup
                ):
                    wrong_index = wrong_lookup[index]
                    wrong_sample = dataset[wrong_index]
                    if wrong_sample.target_id is None:
                        continue

                    wrong_record = run_one_relation_probe(
                        teacher=teacher,
                        image_root=image_root,
                        reference_id=sample.reference_id,
                        target_id=wrong_sample.target_id,
                        category=sample.category,
                        modification_text=sample.modification_text,
                        max_change_patches=args.max_change_patches,
                        pos_threshold=args.pos_threshold,
                        neg_threshold=args.neg_threshold,
                        min_text_effect_norm=args.min_text_effect_norm,
                        min_change_effect_norm=args.min_change_effect_norm,
                        lmax=args.lmax,
                        seed=args.seed + 100_000 + index,
                    )
                    control = {
                        "sample_id": sample.sample_id,
                        "true_target_id": sample.target_id,
                        "wrong_target_id": wrong_sample.target_id,
                        "true": record,
                        "wrong": wrong_record,
                    }
                    controls.append(control)
                    assert controls_file is not None
                    controls_file.write(
                        json.dumps(control, ensure_ascii=False) + "\n"
                    )
                    controls_file.flush()
        finally:
            if controls_file is not None:
                controls_file.close()

    summary = aggregate_summary(records, controls, config)
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print()
    print("R0 RELATION PROBE COMPLETE")
    print(f"samples: {samples_path}")
    if args.control_every > 0:
        print(f"controls: {controls_path}")
    print(f"summary: {summary_path}")
    print()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()