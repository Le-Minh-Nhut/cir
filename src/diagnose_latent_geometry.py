from __future__ import annotations

"""
Latent-geometry diagnostic for CIR IAG-SRME V2 R0.

Put this file at:
    src/diagnose_latent_geometry.py

Example:
    TOKENIZERS_PARALLELISM=false python src/diagnose_latent_geometry.py \
      backbone=fgclip_base_text_native_cls \
      +checkpoint=outputs/r0_ncls_text/best.pt \
      +diagnostic_batches=20 \
      +diagnostic_batch_size=8 \
      hydra.run.dir=outputs/diagnose_latent_geometry_old

This script is READ-ONLY:
- no optimizer.step()
- no checkpoint mutation
- no training
- target images are used only for diagnostic comparisons

It measures cross-sample representation geometry for:
- V0 patch-state pooled representation
- current_global at each recurrent timestep
- committed next-state pooled representation
- committed next-state global representation
- proposal representations
- action representations
- candidate_global representations
- delta_q representations
- terminal retrieval query
- target retrieval embeddings

Main questions:
1) Does effective rank decrease across recurrent steps?
2) Does cross-sample cosine increase (cone collapse)?
3) Does explained variance concentrate in fewer directions?
4) Do per-dimension standard deviations collapse?
5) Does the synthetic recurrent state drift away from the real FG-CLIP feature distribution?
6) Does geometry degrade more in the strong-DPP checkpoint than in the old checkpoint?
"""

import json
import math
from collections import defaultdict
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
from runtime import configure_torch_runtime, resolve_device, seed_everything
from train import CATEGORIES, build_model
from training.engine import resolve_precision


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def _to_cpu_float(x: Tensor) -> Tensor:
    return x.detach().float().cpu()


def _flatten_features(x: Tensor) -> Tensor:
    """Return [N,D] for arbitrary feature tensors."""
    if x.ndim == 1:
        return x[:, None]
    if x.ndim == 2:
        return x
    return x.reshape(-1, x.shape[-1])


def _pool_patch_state(x: Tensor) -> Tensor:
    """Mean-pool [B,N,D] patch state into [B,D]."""
    if x.ndim != 3:
        raise ValueError(f"expected patch state [B,N,D], got {tuple(x.shape)}")
    return x.float().mean(dim=1)


def _safe_float(x: Tensor | float) -> float:
    if isinstance(x, Tensor):
        return float(x.detach().float().cpu())
    return float(x)


def _nan() -> float:
    return float("nan")


# ---------------------------------------------------------------------------
# Geometry metrics
# ---------------------------------------------------------------------------

def _offdiag_values(matrix: Tensor) -> Tensor:
    n = matrix.shape[0]
    if n < 2:
        return matrix.new_empty(0)
    mask = ~torch.eye(n, dtype=torch.bool, device=matrix.device)
    return matrix[mask]


def _pairwise_cosine_stats(x: Tensor, max_samples: int = 2048) -> dict[str, float]:
    x = _flatten_features(x).float()
    if x.shape[0] < 2:
        return {
            "mean": _nan(),
            "median": _nan(),
            "p90": _nan(),
            "p99": _nan(),
        }

    if x.shape[0] > max_samples:
        idx = torch.linspace(0, x.shape[0] - 1, max_samples).long()
        x = x.index_select(0, idx)

    z = F.normalize(x, dim=-1)
    sim = z @ z.T
    values = _offdiag_values(sim)

    if values.numel() == 0:
        return {
            "mean": _nan(),
            "median": _nan(),
            "p90": _nan(),
            "p99": _nan(),
        }

    return {
        "mean": float(values.mean()),
        "median": float(values.median()),
        "p90": float(torch.quantile(values, 0.90)),
        "p99": float(torch.quantile(values, 0.99)),
    }


def _spectrum_stats(x: Tensor) -> dict[str, float]:
    """
    Center X, compute covariance-space singular spectrum.

    effective_rank_pr:
        participation ratio on eigenvalues:
        (sum lambda)^2 / sum lambda^2

    effective_rank_entropy:
        exp(entropy(normalized eigenvalues))

    stable_rank:
        Frobenius^2 / spectral^2
        = sum lambda / max(lambda)
    """
    x = _flatten_features(x).float()
    n, d = x.shape

    if n < 2 or d < 1:
        return {
            "effective_rank_pr": _nan(),
            "effective_rank_entropy": _nan(),
            "stable_rank": _nan(),
            "top1_explained_variance": _nan(),
            "top5_explained_variance": _nan(),
            "top10_explained_variance": _nan(),
        }

    xc = x - x.mean(dim=0, keepdim=True)

    # SVD on [N,D]. Number of non-zero singular values <= min(N-1,D).
    s = torch.linalg.svdvals(xc)
    eigen = s.square()
    total = eigen.sum()

    if float(total) <= 1e-20:
        return {
            "effective_rank_pr": 1.0,
            "effective_rank_entropy": 1.0,
            "stable_rank": 1.0,
            "top1_explained_variance": 1.0,
            "top5_explained_variance": 1.0,
            "top10_explained_variance": 1.0,
        }

    p = eigen / total
    pr = total.square() / eigen.square().sum().clamp_min(1e-20)
    entropy = -(p.clamp_min(1e-20) * p.clamp_min(1e-20).log()).sum()
    entropy_rank = entropy.exp()
    stable_rank = total / eigen.max().clamp_min(1e-20)

    def topk(k: int) -> float:
        k = min(k, eigen.numel())
        return float(eigen[:k].sum() / total)

    return {
        "effective_rank_pr": float(pr),
        "effective_rank_entropy": float(entropy_rank),
        "stable_rank": float(stable_rank),
        "top1_explained_variance": topk(1),
        "top5_explained_variance": topk(5),
        "top10_explained_variance": topk(10),
    }


def _dimension_stats(x: Tensor, near_zero_threshold: float = 1e-4) -> dict[str, float]:
    x = _flatten_features(x).float()
    if x.shape[0] < 2:
        return {
            "mean_per_dim_std": _nan(),
            "median_per_dim_std": _nan(),
            "min_per_dim_std": _nan(),
            "max_per_dim_std": _nan(),
            "near_zero_dim_fraction": _nan(),
        }

    std = x.std(dim=0, unbiased=False)
    return {
        "mean_per_dim_std": float(std.mean()),
        "median_per_dim_std": float(std.median()),
        "min_per_dim_std": float(std.min()),
        "max_per_dim_std": float(std.max()),
        "near_zero_dim_fraction": float((std < near_zero_threshold).float().mean()),
    }


def _covariance_stats(x: Tensor, max_dim: int = 1024) -> dict[str, float]:
    x = _flatten_features(x).float()
    if x.shape[0] < 2:
        return {
            "mean_abs_offdiag_cov": _nan(),
            "mean_abs_offdiag_corr": _nan(),
        }

    # Limit dimension only for memory safety.
    if x.shape[1] > max_dim:
        idx = torch.linspace(0, x.shape[1] - 1, max_dim).long()
        x = x.index_select(1, idx)

    xc = x - x.mean(dim=0, keepdim=True)
    cov = (xc.T @ xc) / max(x.shape[0] - 1, 1)

    offdiag = _offdiag_values(cov)
    mean_abs_cov = float(offdiag.abs().mean()) if offdiag.numel() else _nan()

    std = cov.diag().clamp_min(0).sqrt()
    denom = std[:, None] * std[None, :]
    corr = cov / denom.clamp_min(1e-12)
    offdiag_corr = _offdiag_values(corr)
    mean_abs_corr = (
        float(offdiag_corr.abs().mean()) if offdiag_corr.numel() else _nan()
    )

    return {
        "mean_abs_offdiag_cov": mean_abs_cov,
        "mean_abs_offdiag_corr": mean_abs_corr,
    }


def _norm_stats(x: Tensor) -> dict[str, float]:
    x = _flatten_features(x).float()
    norms = x.norm(dim=-1)
    return {
        "mean_norm": float(norms.mean()),
        "std_norm": float(norms.std(unbiased=False)),
        "min_norm": float(norms.min()),
        "max_norm": float(norms.max()),
    }


def geometry_stats(x: Tensor) -> dict[str, float]:
    x = _flatten_features(x).float()
    result: dict[str, float] = {
        "count": float(x.shape[0]),
        "dim": float(x.shape[1]),
    }
    result.update(_norm_stats(x))
    cosine = _pairwise_cosine_stats(x)
    result.update(
        {
            "pairwise_cosine_mean": cosine["mean"],
            "pairwise_cosine_median": cosine["median"],
            "pairwise_cosine_p90": cosine["p90"],
            "pairwise_cosine_p99": cosine["p99"],
        }
    )
    result.update(_spectrum_stats(x))
    result.update(_dimension_stats(x))
    result.update(_covariance_stats(x))
    return result


def aligned_drift_stats(current: Tensor, anchor: Tensor) -> dict[str, float]:
    """
    current/anchor must be aligned [B,D].
    """
    current = current.float()
    anchor = anchor.float()
    if current.shape != anchor.shape or current.numel() == 0:
        return {
            "relative_l2_drift": _nan(),
            "cosine_to_anchor": _nan(),
        }

    delta = (current - anchor).norm(dim=-1)
    denom = anchor.norm(dim=-1).clamp_min(1e-8)
    relative = delta / denom
    cosine = F.cosine_similarity(current, anchor, dim=-1)

    return {
        "relative_l2_drift": float(relative.mean()),
        "cosine_to_anchor": float(cosine.mean()),
    }


def paired_distribution_shift(x: Tensor, ref: Tensor) -> dict[str, float]:
    """
    Compare two [N,D] distributions using simple diagnostic statistics.
    This is not a formal manifold distance.
    """
    x = _flatten_features(x).float()
    ref = _flatten_features(ref).float()

    if x.shape[1] != ref.shape[1]:
        return {
            "mean_shift_l2": _nan(),
            "std_shift_l2": _nan(),
            "norm_mean_ratio": _nan(),
        }

    mean_shift = (x.mean(0) - ref.mean(0)).norm()
    std_x = x.std(0, unbiased=False)
    std_ref = ref.std(0, unbiased=False)
    std_shift = (std_x - std_ref).norm()

    mean_norm_x = x.norm(dim=-1).mean()
    mean_norm_ref = ref.norm(dim=-1).mean().clamp_min(1e-8)

    return {
        "mean_shift_l2": float(mean_shift),
        "std_shift_l2": float(std_shift),
        "norm_mean_ratio": float(mean_norm_x / mean_norm_ref),
    }


# ---------------------------------------------------------------------------
# Accumulator
# ---------------------------------------------------------------------------

class FeatureAccumulator:
    def __init__(self) -> None:
        self.data: dict[str, list[Tensor]] = defaultdict(list)

    def add(self, name: str, value: Tensor) -> None:
        self.data[name].append(_to_cpu_float(value))

    def cat(self, name: str) -> Tensor | None:
        values = self.data.get(name)
        if not values:
            return None
        return torch.cat(values, dim=0)

    def names(self) -> list[str]:
        return sorted(self.data.keys())


# ---------------------------------------------------------------------------
# Diagnostic logic
# ---------------------------------------------------------------------------

def _find_selected_rows(step: dict[str, Any], num_candidates: int) -> tuple[Tensor, Tensor]:
    """
    Return:
        execute_mask [B_live] bool
        best_idx      [B_live] long, clamped in [0,K-1]
    """
    selected = step["selected_idx"]
    execute = selected < num_candidates
    best_idx = selected.clamp_max(num_candidates - 1)
    return execute, best_idx


def _gather_candidate(values: Tensor, indices: Tensor) -> Tensor:
    shape = [values.shape[0], 1] + [1] * (values.ndim - 2)
    index = indices.view(*shape).expand(-1, 1, *values.shape[2:])
    return values.gather(1, index).squeeze(1)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    checkpoint_value = cfg.get("checkpoint")
    if checkpoint_value is None:
        raise ValueError("Pass +checkpoint=path/to/best.pt")

    checkpoint_path = Path(str(checkpoint_value))
    diagnostic_batches = int(cfg.get("diagnostic_batches", 20))
    diagnostic_batch_size = int(cfg.get("diagnostic_batch_size", 8))

    seed_everything(int(cfg.seed), bool(cfg.runtime.deterministic))
    configure_torch_runtime(
        deterministic=bool(cfg.runtime.deterministic),
        benchmark=bool(cfg.runtime.benchmark),
    )

    device = resolve_device(
        str(cfg.runtime.device),
        int(cfg.runtime.accelerator_index),
    )
    precision = resolve_precision(str(cfg.runtime.precision), device)

    print(f"[latent-geometry] device={device} checkpoint={checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

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
        generator=torch.Generator().manual_seed(int(cfg.seed) + 20260907),
    )

    acc = FeatureAccumulator()

    # Aligned per-sample drift buffers.
    drift_current_by_t: dict[int, list[Tensor]] = defaultdict(list)
    drift_anchor_by_t: dict[int, list[Tensor]] = defaultdict(list)
    committed_global_by_t: dict[int, list[Tensor]] = defaultdict(list)
    committed_anchor_global_by_t: dict[int, list[Tensor]] = defaultdict(list)

    # Retrieval geometry.
    retrieval_by_t: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    with torch.no_grad():
        iterator = iter(loader)

        for batch_idx in range(diagnostic_batches):
            try:
                batch = next(iterator)
            except StopIteration:
                break

            batch = batch.to(device)
            if batch.target_pixels is None:
                raise RuntimeError("diagnostic requires include_targets=True")

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

                # encode_global_images() already returns the final normalized
                # 512-D retrieval embedding. Do NOT pass it through
                # retrieval_from_global(), which expects a 768-D pre-projection
                # global visual state.
                target_query = model.backbone.encode_global_images(batch.target_pixels)

            # Real-image reference / target baselines.
            initial_state = output["initial_state"]
            initial_state_pooled = _pool_patch_state(initial_state)

            if getattr(model.backbone, "global_readout_mode", "learned_qg") == "native_cls":
                cls_anchor = output.get("cls_anchor")
                initial_global = model.backbone.global_readout(initial_state, cls_anchor)
            else:
                initial_global = model.backbone.global_readout(initial_state)

            acc.add("V0_pooled", initial_state_pooled)
            acc.add("reference_global", initial_global)

            # Real target image retrieval embedding (already projected + normalized).
            acc.add("target_query", target_query)

            for step in output["steps"]:
                t = int(step["timestep"])
                prefix = f"t{t}"

                parent = step["parent_state"]
                current_global = step["current_global"]
                proposals = step["proposals"]
                actions = step["actions"]
                candidate_states = step["candidate_states"]
                candidate_queries = step["candidate_queries"]
                delta_q = step["delta_q"]
                selected_idx = step["selected_idx"]

                live_indices = step["live_indices"]
                live_initial_state = initial_state.index_select(0, live_indices)
                live_initial_pooled = _pool_patch_state(live_initial_state)

                live_initial_global = initial_global.index_select(0, live_indices)
                live_target_query = target_query.index_select(0, live_indices)

                parent_pooled = _pool_patch_state(parent)

                acc.add(f"{prefix}/current_state_pooled", parent_pooled)
                acc.add(f"{prefix}/current_global", current_global)
                acc.add(f"{prefix}/proposals_all", proposals.reshape(-1, proposals.shape[-1]))
                acc.add(f"{prefix}/actions_all", actions.reshape(-1, actions.shape[-1]))
                acc.add(
                    f"{prefix}/candidate_query_all",
                    candidate_queries.reshape(-1, candidate_queries.shape[-1]),
                )
                acc.add(
                    f"{prefix}/delta_q_all",
                    delta_q.reshape(-1, delta_q.shape[-1]),
                )

                candidate_state_pooled = candidate_states.float().mean(dim=-2)
                acc.add(
                    f"{prefix}/candidate_state_pooled_all",
                    candidate_state_pooled.reshape(
                        -1, candidate_state_pooled.shape[-1]
                    ),
                )

                # Candidate global features can be reconstructed from candidate query path
                # only if stored; current model steps store candidate_queries but not
                # candidate_global. Recompute from candidate_states for diagnostics.
                live_cls = step.get("cls_anchor")
                candidate_global = model.backbone.global_readout(
                    candidate_states,
                    live_cls,
                )
                acc.add(
                    f"{prefix}/candidate_global_all",
                    candidate_global.reshape(-1, candidate_global.shape[-1]),
                )

                # Drift of current state from V0 for the same live sample.
                drift_current_by_t[t].append(_to_cpu_float(parent_pooled))
                drift_anchor_by_t[t].append(_to_cpu_float(live_initial_pooled))

                # Retrieval statistics for the current query.
                current_query = step["current_query"]
                pos_sim = F.cosine_similarity(
                    current_query.float(),
                    live_target_query.float(),
                    dim=-1,
                )
                retrieval_by_t[t]["current_positive_similarity"].extend(
                    pos_sim.detach().cpu().tolist()
                )

                execute, best_idx = _find_selected_rows(
                    step,
                    int(model.config.num_candidates),
                )

                if execute.any():
                    chosen_state = _gather_candidate(
                        candidate_states[execute],
                        best_idx[execute],
                    )
                    chosen_state_pooled = _pool_patch_state(chosen_state)

                    chosen_global = _gather_candidate(
                        candidate_global[execute],
                        best_idx[execute],
                    )
                    chosen_query = _gather_candidate(
                        candidate_queries[execute],
                        best_idx[execute],
                    )

                    acc.add(f"{prefix}/committed_state_pooled", chosen_state_pooled)
                    acc.add(f"{prefix}/committed_global", chosen_global)
                    acc.add(f"{prefix}/committed_query", chosen_query)

                    chosen_live_initial_pooled = live_initial_pooled[execute]
                    chosen_live_initial_global = live_initial_global[execute]

                    committed_global_by_t[t].append(_to_cpu_float(chosen_global))
                    committed_anchor_global_by_t[t].append(
                        _to_cpu_float(chosen_live_initial_global)
                    )

                    chosen_target_query = live_target_query[execute]
                    chosen_pos_sim = F.cosine_similarity(
                        chosen_query.float(),
                        chosen_target_query.float(),
                        dim=-1,
                    )
                    retrieval_by_t[t]["committed_positive_similarity"].extend(
                        chosen_pos_sim.detach().cpu().tolist()
                    )

            # Terminal query.
            terminal_query = output["query"]
            acc.add("terminal_query", terminal_query)

            terminal_pos_sim = F.cosine_similarity(
                terminal_query.float(),
                target_query.float(),
                dim=-1,
            )
            retrieval_by_t[-1]["terminal_positive_similarity"].extend(
                terminal_pos_sim.detach().cpu().tolist()
            )

            print(
                f"[latent-geometry] batch={batch_idx + 1}/{diagnostic_batches} "
                f"steps={len(output['steps'])}"
            )

    # -----------------------------------------------------------------------
    # Aggregate
    # -----------------------------------------------------------------------

    geometry: dict[str, dict[str, float]] = {}
    for name in acc.names():
        value = acc.cat(name)
        if value is not None and value.numel() > 0:
            geometry[name] = geometry_stats(value)

    # Drift against V0.
    drift: dict[str, dict[str, float]] = {}

    for t in sorted(drift_current_by_t):
        current = torch.cat(drift_current_by_t[t], dim=0)
        anchor = torch.cat(drift_anchor_by_t[t], dim=0)
        drift[f"t{t}/current_state_pooled_vs_V0"] = aligned_drift_stats(
            current,
            anchor,
        )

    for t in sorted(committed_global_by_t):
        current = torch.cat(committed_global_by_t[t], dim=0)
        anchor = torch.cat(committed_anchor_global_by_t[t], dim=0)
        drift[f"t{t}/committed_global_vs_initial_global"] = aligned_drift_stats(
            current,
            anchor,
        )

    # Distribution shift against real-image reference/target baselines.
    distribution_shift: dict[str, dict[str, float]] = {}
    reference_global = acc.cat("reference_global")
    target_query_all = acc.cat("target_query")

    if reference_global is not None:
        for name in acc.names():
            value = acc.cat(name)
            if value is None:
                continue
            if name.endswith("global") or "candidate_global" in name:
                distribution_shift[f"{name}_vs_reference_global"] = (
                    paired_distribution_shift(value, reference_global)
                )

    if target_query_all is not None:
        terminal_query_all = acc.cat("terminal_query")
        if terminal_query_all is not None:
            distribution_shift["terminal_query_vs_target_query"] = (
                paired_distribution_shift(terminal_query_all, target_query_all)
            )

    retrieval: dict[str, dict[str, float]] = {}
    for t, metrics in retrieval_by_t.items():
        label = "terminal" if t == -1 else f"t{t}"
        retrieval[label] = {}
        for metric_name, values in metrics.items():
            if not values:
                continue
            tensor = torch.tensor(values, dtype=torch.float32)
            retrieval[label][metric_name + "_mean"] = float(tensor.mean())
            retrieval[label][metric_name + "_std"] = float(
                tensor.std(unbiased=False)
            )

    # -----------------------------------------------------------------------
    # Automatic flags
    # -----------------------------------------------------------------------

    flags: list[dict[str, str]] = []

    v0 = geometry.get("V0_pooled")
    terminal = geometry.get("terminal_query")

    # Compare effective rank only when both exist.
    if v0 and terminal:
        v0_rank = v0.get("effective_rank_pr", _nan())
        term_rank = terminal.get("effective_rank_pr", _nan())
        if math.isfinite(v0_rank) and math.isfinite(term_rank) and v0_rank > 0:
            drop = 1.0 - term_rank / v0_rank
            if drop > 0.30:
                flags.append(
                    {
                        "level": "WARN",
                        "code": "EFFECTIVE_RANK_DROP",
                        "message": (
                            f"Terminal effective rank is {drop:.1%} lower than "
                            f"V0 pooled rank ({v0_rank:.2f} -> {term_rank:.2f})."
                        ),
                    }
                )

    # Monotonic cross-sample cone concentration in current globals.
    timestep_cosines: list[tuple[int, float]] = []
    for name, stats in geometry.items():
        if name.startswith("t") and name.endswith("/current_global"):
            try:
                t = int(name.split("/")[0][1:])
            except ValueError:
                continue
            c = stats.get("pairwise_cosine_mean", _nan())
            if math.isfinite(c):
                timestep_cosines.append((t, c))

    timestep_cosines.sort()
    if len(timestep_cosines) >= 2:
        increase = timestep_cosines[-1][1] - timestep_cosines[0][1]
        if increase > 0.15:
            flags.append(
                {
                    "level": "WARN",
                    "code": "RECURRENT_CONE_CONCENTRATION",
                    "message": (
                        "Mean cross-sample current-global cosine increases by "
                        f"{increase:.3f} from t{timestep_cosines[0][0]} "
                        f"to t{timestep_cosines[-1][0]}."
                    ),
                }
            )

    # Rank degradation across current globals.
    timestep_ranks: list[tuple[int, float]] = []
    for name, stats in geometry.items():
        if name.startswith("t") and name.endswith("/current_global"):
            try:
                t = int(name.split("/")[0][1:])
            except ValueError:
                continue
            r = stats.get("effective_rank_pr", _nan())
            if math.isfinite(r):
                timestep_ranks.append((t, r))

    timestep_ranks.sort()
    if len(timestep_ranks) >= 2 and timestep_ranks[0][1] > 0:
        drop = 1.0 - timestep_ranks[-1][1] / timestep_ranks[0][1]
        if drop > 0.30:
            flags.append(
                {
                    "level": "WARN",
                    "code": "RECURRENT_RANK_COLLAPSE",
                    "message": (
                        f"Current-global effective rank drops {drop:.1%} "
                        f"from t{timestep_ranks[0][0]} to t{timestep_ranks[-1][0]}."
                    ),
                }
            )

    # Large state drift.
    last_state_drift = None
    for key in sorted(drift):
        if "current_state_pooled_vs_V0" in key:
            value = drift[key].get("relative_l2_drift", _nan())
            if math.isfinite(value):
                last_state_drift = (key, value)

    if last_state_drift is not None and last_state_drift[1] > 1.0:
        flags.append(
            {
                "level": "WARN",
                "code": "LARGE_RECURRENT_STATE_DRIFT",
                "message": (
                    f"{last_state_drift[0]} relative L2 drift is "
                    f"{last_state_drift[1]:.3f}."
                ),
            }
        )

    if not flags:
        flags.append(
            {
                "level": "OK",
                "code": "NO_STRONG_LATENT_COLLAPSE_FLAG",
                "message": (
                    "No conservative automatic latent-collapse threshold was crossed. "
                    "Compare OLD vs STRONG reports before concluding geometry is healthy."
                ),
            }
        )

    metadata = checkpoint.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}

    report = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint": {
            "epoch": checkpoint.get("epoch"),
            "metric": checkpoint.get("metric"),
            "architecture": metadata.get("architecture"),
            "experiment_identity": metadata.get("experiment_identity"),
            "global_readout_mode": metadata.get("global_readout_mode"),
            "finetune_policy": metadata.get("finetune_policy"),
        },
        "diagnostic": {
            "batches": diagnostic_batches,
            "batch_size": diagnostic_batch_size,
            "device": str(device),
            "precision": str(cfg.runtime.precision),
        },
        "geometry": geometry,
        "drift": drift,
        "distribution_shift": distribution_shift,
        "retrieval": retrieval,
        "flags": flags,
    }

    # -----------------------------------------------------------------------
    # Markdown summary
    # -----------------------------------------------------------------------

    def fmt(v: Any) -> str:
        try:
            v = float(v)
            if math.isnan(v):
                return "nan"
            return f"{v:.5f}"
        except Exception:
            return str(v)

    md_lines: list[str] = [
        "# CIR IAG-SRME V2 R0 — Latent Geometry Diagnostic",
        "",
        f"- checkpoint: `{checkpoint_path}`",
        f"- epoch: `{checkpoint.get('epoch')}`",
        f"- validation metric: `{checkpoint.get('metric')}`",
        f"- batches: `{diagnostic_batches}`",
        f"- batch size: `{diagnostic_batch_size}`",
        "",
        "## Automatic flags",
        "",
    ]

    for flag in flags:
        md_lines.append(
            f"- **{flag['level']} — {flag['code']}**: {flag['message']}"
        )

    md_lines += [
        "",
        "## Geometry summary",
        "",
        "| Representation | N | mean cos | eff rank(PR) | stable rank | top1 var | top5 var | top10 var | mean dim std | near-zero dims | mean norm |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    important_order = [
        "V0_pooled",
        "reference_global",
        "target_query",
    ]

    timestep_names = sorted(
        [n for n in geometry if n.startswith("t")],
        key=lambda x: (
            int(x.split("/")[0][1:]) if x.split("/")[0][1:].isdigit() else 999,
            x,
        ),
    )

    terminal_names = ["terminal_query"]
    ordered = important_order + timestep_names + terminal_names

    seen = set()
    for name in ordered:
        if name in seen or name not in geometry:
            continue
        seen.add(name)
        s = geometry[name]
        md_lines.append(
            "| {name} | {count:.0f} | {cos} | {rank} | {stable} | {t1} | {t5} | {t10} | {std} | {zero} | {norm} |".format(
                name=name,
                count=s.get("count", float("nan")),
                cos=fmt(s.get("pairwise_cosine_mean")),
                rank=fmt(s.get("effective_rank_pr")),
                stable=fmt(s.get("stable_rank")),
                t1=fmt(s.get("top1_explained_variance")),
                t5=fmt(s.get("top5_explained_variance")),
                t10=fmt(s.get("top10_explained_variance")),
                std=fmt(s.get("mean_per_dim_std")),
                zero=fmt(s.get("near_zero_dim_fraction")),
                norm=fmt(s.get("mean_norm")),
            )
        )

    md_lines += [
        "",
        "## Recurrent drift",
        "",
        "| Comparison | relative L2 drift | cosine to anchor |",
        "|---|---:|---:|",
    ]

    for name, s in sorted(drift.items()):
        md_lines.append(
            f"| {name} | {fmt(s.get('relative_l2_drift'))} | "
            f"{fmt(s.get('cosine_to_anchor'))} |"
        )

    md_lines += [
        "",
        "## Distribution-shift proxies",
        "",
        "| Comparison | mean shift L2 | std shift L2 | norm mean ratio |",
        "|---|---:|---:|---:|",
    ]

    for name, s in sorted(distribution_shift.items()):
        md_lines.append(
            f"| {name} | {fmt(s.get('mean_shift_l2'))} | "
            f"{fmt(s.get('std_shift_l2'))} | {fmt(s.get('norm_mean_ratio'))} |"
        )

    md_lines += [
        "",
        "## Retrieval geometry",
        "",
    ]

    for name, metrics in sorted(retrieval.items()):
        md_lines.append(f"### {name}")
        md_lines.append("")
        for metric_name, value in sorted(metrics.items()):
            md_lines.append(f"- `{metric_name}`: `{fmt(value)}`")
        md_lines.append("")

    md_lines += [
        "## Interpretation rules",
        "",
        "- Sibling diversity and global representation health are different questions.",
        "- A low sibling cosine does **not** prove the global embedding space is healthy.",
        "- Strong evidence of recurrent collapse would be a large rank drop, rising cross-sample cosine, or rapidly increasing variance concentration from early to late timesteps.",
        "- Healthy rank but large real-vs-synthetic distribution shift would indicate off-manifold drift without classical dimensional collapse.",
        "- Always compare this report between OLD and STRONG checkpoints before changing the objective.",
    ]

    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "latent_geometry_report.json"
    md_path = output_dir / "latent_geometry_report.md"

    json_path.write_text(
        json.dumps(report, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print("\n================ LATENT GEOMETRY FLAGS ================")
    for flag in flags:
        print(f"[{flag['level']}] {flag['code']}: {flag['message']}")
    print("=======================================================")
    print(f"[latent-geometry] JSON: {json_path}")
    print(f"[latent-geometry] Markdown: {md_path}")


if __name__ == "__main__":
    main()