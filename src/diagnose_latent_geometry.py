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

from __future__ import annotations

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
from diagnostics.cohort import (
    load_or_create_manifest,
    sample_ids_fingerprint,
    validate_processed_manifest,
)
from diagnostics.geometry import (
    assert_compatible_feature_interface,
    candidate_geometry,
    feature_geometry,
)
from evaluate import validate_checkpoint_backbone_metadata
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


def _nan() -> float:
    return float("nan")


# ---------------------------------------------------------------------------
# Geometry metrics
# ---------------------------------------------------------------------------


def geometry_stats(x: Tensor) -> dict[str, float]:
    # Kept as the public compatibility name used by existing reports/tests.
    return feature_geometry(x)


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


def distribution_shift_stats(x: Tensor, ref: Tensor) -> dict[str, float]:
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


def aligned_distribution_shift(x: Tensor, ref: Tensor) -> dict[str, float]:
    """Distribution shift plus row-wise drift for the same ordered sample cohort."""

    x = _flatten_features(x).float()
    ref = _flatten_features(ref).float()
    if x.shape != ref.shape:
        raise ValueError(
            f"aligned distribution shift requires equal shapes, got {x.shape} and {ref.shape}"
        )
    result = distribution_shift_stats(x, ref)
    result.update(
        {
            "paired_l2_mean": float((x - ref).norm(dim=-1).mean()),
            "paired_cosine_mean": float(F.cosine_similarity(x, ref, dim=-1).mean()),
        }
    )
    return result


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
    manifest_value = cfg.get("diagnostic_manifest")
    if manifest_value is None:
        raise ValueError(
            "pass +diagnostic_manifest=path/to/shared_manifest.json; "
            "OLD and STRONG must replay the same persistent cohort"
        )
    diagnostic_split = str(cfg.get("diagnostic_split", "train"))

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
    dataset = FashionIQDataset(
        annotation_root,
        diagnostic_split,
        CATEGORIES,
        caption_policy=caption_policy,
        seed=int(cfg.seed),
    )
    cohort, manifest = load_or_create_manifest(
        dataset,
        str(manifest_value),
        sample_count=diagnostic_batches * diagnostic_batch_size,
        batch_size=diagnostic_batch_size,
        seed=int(cfg.seed),
        split=diagnostic_split,
        caption_policy=caption_policy,
    )

    collator = FashionIQImageCollator(
        image_store,
        tokenizer,
        processor,
        int(cfg.backbone.max_text_length),
        include_targets=True,
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

    acc = FeatureAccumulator()
    candidate_acc = FeatureAccumulator()

    # Aligned per-sample drift buffers.
    drift_current_by_t: dict[int, list[Tensor]] = defaultdict(list)
    drift_anchor_by_t: dict[int, list[Tensor]] = defaultdict(list)
    committed_global_by_t: dict[int, list[Tensor]] = defaultdict(list)
    committed_anchor_global_by_t: dict[int, list[Tensor]] = defaultdict(list)
    executed_before_by_t: dict[int, list[Tensor]] = defaultdict(list)
    executed_after_by_t: dict[int, list[Tensor]] = defaultdict(list)
    state_records: list[dict[str, Any]] = []
    terminal_records: list[dict[str, Any]] = []
    processed_sample_ids: list[str] = []
    live_sample_ids_by_t: dict[int, list[str]] = defaultdict(list)
    candidate_reference_global_by_t: dict[int, list[Tensor]] = defaultdict(list)
    feature_export: dict[int, dict[str, Any]] = defaultdict(
        lambda: {"sample_ids": [], "features": defaultdict(list)}
    )

    # Retrieval geometry.
    retrieval_by_t: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            processed_sample_ids.extend(batch.sample_ids)

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

                # The first recurrent readout is the exact q0/g0 used by the live
                # forward. A zero-step configuration falls back to the same readout
                # functions under this identical autocast policy.
                if output["steps"]:
                    initial_global = output["steps"][0]["current_global"]
                    initial_query = output["steps"][0]["current_query"]
                else:
                    cls_anchor = output.get("cls_anchor")
                    initial_global = model.backbone.global_readout(
                        output["initial_state"], cls_anchor
                    )
                    initial_query = model.backbone.retrieval_from_global(initial_global)

            # Real-image reference / target baselines.
            initial_state = output["initial_state"]
            initial_state_pooled = _pool_patch_state(initial_state)

            acc.add("V0_pooled", initial_state_pooled)
            acc.add("reference_global", initial_global)
            acc.add("initial_query", initial_query)

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
                current_query = step["current_query"]
                delta_q = step["delta_q"]
                live_indices = step["live_indices"]
                live_ids = [batch.sample_ids[index] for index in live_indices.cpu().tolist()]
                live_sample_ids_by_t[t].extend(live_ids)
                live_initial_state = initial_state.index_select(0, live_indices)
                live_initial_pooled = _pool_patch_state(live_initial_state)

                live_initial_global = initial_global.index_select(0, live_indices)
                live_target_query = target_query.index_select(0, live_indices)

                parent_pooled = _pool_patch_state(parent)

                acc.add(f"{prefix}/current_state_pooled", parent_pooled)
                acc.add(f"{prefix}/current_global", current_global)
                acc.add(f"{prefix}/current_query", current_query)
                candidate_acc.add(f"{prefix}/proposals", proposals)
                candidate_acc.add(f"{prefix}/actions", actions)
                candidate_acc.add(f"{prefix}/candidate_query", candidate_queries)
                candidate_acc.add(f"{prefix}/delta_q", delta_q)

                candidate_state_pooled = candidate_states.float().mean(dim=-2)
                candidate_acc.add(f"{prefix}/candidate_state_pooled", candidate_state_pooled)

                candidate_global = step["candidate_global"]
                with torch.autocast(
                    device_type=device.type,
                    enabled=precision.autocast_enabled,
                    dtype=precision.autocast_dtype,
                ):
                    reconstructed_queries = model.backbone.retrieval_from_global(candidate_global)
                tolerance = (
                    5e-3 if candidate_queries.dtype in (torch.float16, torch.bfloat16) else 2e-5
                )
                if not torch.allclose(
                    reconstructed_queries,
                    candidate_queries,
                    atol=tolerance,
                    rtol=tolerance,
                ):
                    raise AssertionError(
                        "stored candidate_global does not reconstruct stored candidate_queries"
                    )
                candidate_acc.add(f"{prefix}/candidate_global", candidate_global)
                candidate_reference_global_by_t[t].append(_to_cpu_float(live_initial_global))

                exported = feature_export[t]
                exported["sample_ids"].extend(live_ids)
                for name, value in {
                    "proposal": proposals,
                    "action": actions,
                    "delta_q": delta_q,
                    "candidate_global": candidate_global,
                    "candidate_query": candidate_queries,
                    "current_global": current_global,
                    "current_query": current_query,
                }.items():
                    exported["features"][name].append(_to_cpu_float(value))

                # Drift of current state from V0 for the same live sample.
                drift_current_by_t[t].append(_to_cpu_float(parent_pooled))
                drift_anchor_by_t[t].append(_to_cpu_float(live_initial_pooled))

                # Retrieval statistics for the current query.
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
                gathered_state = _gather_candidate(candidate_states, best_idx)
                committed_after_decision = torch.where(
                    execute[:, None, None], gathered_state, parent
                )
                acc.add(
                    f"{prefix}/committed_state_after_decision_pooled",
                    _pool_patch_state(committed_after_decision),
                )
                live_batch_rows = live_indices.detach().cpu().tolist()
                live_batch_set = set(live_batch_rows)
                for batch_row, sample_id in enumerate(batch.sample_ids):
                    if batch_row not in live_batch_set:
                        state_records.append(
                            {
                                "sample_id": sample_id,
                                "timestep": t,
                                "live": False,
                                "executed": False,
                                "selected_idx": None,
                                "stopped_now": False,
                                "state_before_norm": None,
                                "candidate_preview_norms": None,
                                "committed_state_after_norm": None,
                            }
                        )
                for local_row, batch_row in enumerate(live_batch_rows):
                    state_records.append(
                        {
                            "sample_id": batch.sample_ids[batch_row],
                            "timestep": t,
                            "live": True,
                            "executed": bool(execute[local_row]),
                            "selected_idx": int(step["selected_idx"][local_row]),
                            "stopped_now": bool(step["stopped_now"][local_row]),
                            "state_before_norm": float(
                                parent[local_row].detach().float().norm().cpu()
                            ),
                            "candidate_preview_norms": [
                                float(value)
                                for value in candidate_states[local_row]
                                .detach()
                                .float()
                                .flatten(1)
                                .norm(dim=-1)
                                .cpu()
                            ],
                            "committed_state_after_norm": float(
                                committed_after_decision[local_row].detach().float().norm().cpu()
                            ),
                        }
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

                    chosen_live_initial_global = live_initial_global[execute]

                    committed_global_by_t[t].append(_to_cpu_float(chosen_global))
                    committed_anchor_global_by_t[t].append(
                        _to_cpu_float(chosen_live_initial_global)
                    )
                    executed_before_by_t[t].append(_to_cpu_float(parent_pooled[execute]))
                    executed_after_by_t[t].append(_to_cpu_float(chosen_state_pooled))

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
            acc.add("terminal_state_pooled", _pool_patch_state(output["state"]))
            for row, sample_id in enumerate(batch.sample_ids):
                terminal_records.append(
                    {
                        "sample_id": sample_id,
                        "stopped_early": bool(output["stopped"][row]),
                        "terminal_state_norm": float(
                            output["state"][row].detach().float().norm().cpu()
                        ),
                        "terminal_query_norm": float(
                            terminal_query[row].detach().float().norm().cpu()
                        ),
                    }
                )

            terminal_pos_sim = F.cosine_similarity(
                terminal_query.float(),
                target_query.float(),
                dim=-1,
            )
            retrieval_by_t[-1]["terminal_positive_similarity"].extend(
                terminal_pos_sim.detach().cpu().tolist()
            )

            print(
                f"[latent-geometry] batch={batch_idx + 1}/{len(loader)} "
                f"steps={len(output['steps'])}"
            )

    cohort_metadata = validate_processed_manifest(
        manifest, processed_sample_ids, batch_size=diagnostic_batch_size
    )

    # -----------------------------------------------------------------------
    # Aggregate
    # -----------------------------------------------------------------------

    geometry: dict[str, dict[str, float]] = {}
    for name in acc.names():
        value = acc.cat(name)
        if value is not None and value.numel() > 0:
            geometry[name] = geometry_stats(value)

    candidate_geometry_report: dict[str, dict[str, Any]] = {}
    for name in candidate_acc.names():
        value = candidate_acc.cat(name)
        if value is None or value.numel() == 0:
            continue
        report = candidate_geometry(value)
        candidate_geometry_report[name] = report
        # Preserve historical *_all keys and add an unambiguous pooled alias.
        legacy_name = f"{name}_all"
        geometry[legacy_name] = report["raw"]["pooled"]
        geometry[f"{name}_pooled"] = report["raw"]["pooled"]

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
    for t in sorted(executed_before_by_t):
        before = torch.cat(executed_before_by_t[t], dim=0)
        after = torch.cat(executed_after_by_t[t], dim=0)
        drift[f"t{t}/paired_executed_state_after_vs_before"] = aligned_drift_stats(after, before)

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
                distribution_shift[f"{name}_vs_reference_global_distribution"] = (
                    distribution_shift_stats(value, reference_global)
                )
        for t, references in sorted(candidate_reference_global_by_t.items()):
            candidate_value = candidate_acc.cat(f"t{t}/candidate_global")
            if candidate_value is None:
                continue
            live_reference = torch.cat(references, dim=0)
            if candidate_value.shape[0] != live_reference.shape[0]:
                raise ValueError("candidate-global/reference row alignment was lost")
            for slot in range(candidate_value.shape[1]):
                distribution_shift[
                    f"t{t}/candidate_global_slot_{slot}_vs_live_reference_global_aligned"
                ] = aligned_distribution_shift(candidate_value[:, slot], live_reference)
            repeated_reference = live_reference[:, None].expand_as(candidate_value)
            distribution_shift[f"t{t}/candidate_global_pooled_vs_live_reference_global_aligned"] = (
                aligned_distribution_shift(
                    candidate_value.reshape(-1, candidate_value.shape[-1]),
                    repeated_reference.reshape(-1, repeated_reference.shape[-1]),
                )
            )

    if target_query_all is not None:
        terminal_query_all = acc.cat("terminal_query")
        if terminal_query_all is not None:
            distribution_shift["terminal_query_vs_target_query_aligned"] = (
                aligned_distribution_shift(terminal_query_all, target_query_all)
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
            retrieval[label][metric_name + "_std"] = float(tensor.std(unbiased=False))

    # -----------------------------------------------------------------------
    # Automatic flags
    # -----------------------------------------------------------------------

    flags: list[dict[str, str]] = []

    def add_compatible_rank_drop_flag(
        start_name: str, end_name: str, *, interface: str, code: str
    ) -> None:
        start = acc.cat(start_name)
        end = acc.cat(end_name)
        if start is None or end is None:
            return
        assert_compatible_feature_interface(
            start,
            end,
            first_name=start_name,
            second_name=end_name,
            interface=interface,
        )
        start_rank = geometry[start_name].get("effective_rank_pr", _nan())
        end_rank = geometry[end_name].get("effective_rank_pr", _nan())
        if math.isfinite(start_rank) and math.isfinite(end_rank) and start_rank > 0:
            drop = 1.0 - end_rank / start_rank
            if drop > 0.30:
                flags.append(
                    {
                        "level": "WARN",
                        "code": code,
                        "message": (
                            f"{interface} effective rank is {drop:.1%} lower "
                            f"({start_name} {start_rank:.2f} -> {end_name} {end_rank:.2f})."
                        ),
                    }
                )

    add_compatible_rank_drop_flag(
        "initial_query",
        "terminal_query",
        interface="retrieval-query",
        code="RETRIEVAL_QUERY_RANK_DROP",
    )
    add_compatible_rank_drop_flag(
        "V0_pooled",
        "terminal_state_pooled",
        interface="pooled-patch-state",
        code="PATCH_STATE_RANK_DROP",
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
                    f"{last_state_drift[0]} relative L2 drift is {last_state_drift[1]:.3f}."
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

    headline_geometry = {}
    for name, modes in candidate_geometry_report.items():
        if not any(name.endswith(suffix) for suffix in ("/proposals", "/actions", "/delta_q")):
            continue
        raw = modes["raw"]
        headline_geometry[name] = {
            "per_slot_PR": [slot["effective_rank_pr"] for slot in raw["per_slot"]],
            "slot_centered_pooled_PR": raw["slot_centered_pooled"]["effective_rank_pr"],
            "between_slot_variance_fraction": raw["variance_decomposition"][
                "between_slot_variance_fraction"
            ],
        }

    headline_trajectory = {
        "initial_query_PR": geometry.get("initial_query", {}).get("effective_rank_pr", _nan()),
        "terminal_query_PR": geometry.get("terminal_query", {}).get("effective_rank_pr", _nan()),
        "V0_pooled_PR": geometry.get("V0_pooled", {}).get("effective_rank_pr", _nan()),
        "terminal_state_pooled_PR": geometry.get("terminal_state_pooled", {}).get(
            "effective_rank_pr", _nan()
        ),
    }

    timestep_cohorts = {
        str(t): {
            "live_sample_ids": sample_ids,
            "live_sample_count": len(sample_ids),
            "live_sample_ids_fingerprint": sample_ids_fingerprint(sample_ids),
        }
        for t, sample_ids in sorted(live_sample_ids_by_t.items())
    }
    exported_features = {
        "format_version": 1,
        "manifest_sample_ids_sha256": cohort_metadata["processed_sample_ids_sha256"],
        "teacher_batch_grouping_sha256": cohort_metadata["teacher_batch_grouping_sha256"],
        "timesteps": {
            str(t): {
                "sample_ids": values["sample_ids"],
                "features": {
                    name: torch.cat(chunks, dim=0) for name, chunks in values["features"].items()
                },
            }
            for t, values in sorted(feature_export.items())
        },
    }

    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    features_path = output_dir / "candidate_features.pt"

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
            "batches": len(loader),
            "requested_diagnostic_batches": diagnostic_batches,
            "batch_size": diagnostic_batch_size,
            "device": str(device),
            "precision": str(cfg.runtime.precision),
            "manifest_path": str(manifest_value),
            **cohort_metadata,
            "split": diagnostic_split,
            "caption_policy": caption_policy,
            "candidate_features_path": str(features_path),
        },
        "timestep_cohorts": timestep_cohorts,
        "geometry": geometry,
        "candidate_geometry": candidate_geometry_report,
        "headline_candidate_geometry": headline_geometry,
        "headline_trajectory_geometry": headline_trajectory,
        "metric_definitions": {
            "retrieval_query_rank_trajectory": (
                "Cross-input covariance PR in the projected, L2-normalized retrieval "
                "interface: initial_query/current_query/candidate_query/committed_query/"
                "terminal_query only."
            ),
            "patch_state_rank_trajectory": (
                "Cross-input covariance PR of mean-pooled recurrent patch states: "
                "V0/current/committed/terminal state only."
            ),
            "candidate_per_slot_PR": (
                "Cross-input covariance participation ratio computed independently "
                "for each candidate slot."
            ),
            "legacy_sibling_rank": (
                "Not reported here; diagnose_iag_srme.py labels its per-input K-by-D "
                "singular-value metric explicitly."
            ),
        },
        "state_transition_records": state_records,
        "terminal_cohort_records": terminal_records,
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
        except (TypeError, ValueError):
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
        md_lines.append(f"- **{flag['level']} — {flag['code']}**: {flag['message']}")

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
        "initial_query",
        "target_query",
    ]

    timestep_names = sorted(
        [n for n in geometry if n.startswith("t")],
        key=lambda x: (
            int(x.split("/")[0][1:]) if x.split("/")[0][1:].isdigit() else 999,
            x,
        ),
    )

    terminal_names = ["terminal_state_pooled", "terminal_query"]
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
        "## Candidate geometry decomposition",
        "",
        "| Representation | pooled PR | slot-centered PR | mean per-slot PR | min per-slot PR | between-slot variance |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, modes in sorted(candidate_geometry_report.items()):
        raw = modes["raw"]
        md_lines.append(
            f"| {name} | {fmt(raw['pooled']['effective_rank_pr'])} | "
            f"{fmt(raw['slot_centered_pooled']['effective_rank_pr'])} | "
            f"{fmt(raw['per_slot_summary']['mean_per_slot_PR'])} | "
            f"{fmt(raw['per_slot_summary']['min_per_slot_PR'])} | "
            f"{fmt(raw['variance_decomposition']['between_slot_variance_fraction'])} |"
        )
        md_lines.append("")
        for slot in raw["per_slot"]:
            md_lines.append(
                f"- `{name}/slot_{slot['slot']}`: PR={fmt(slot['effective_rank_pr'])}, "
                f"entropy-rank={fmt(slot['effective_rank_entropy'])}, "
                f"mean-std={fmt(slot['mean_per_dim_std'])}, "
                f"zero-variance={slot['zero_total_variance']}"
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
            f"| {name} | {fmt(s.get('relative_l2_drift'))} | {fmt(s.get('cosine_to_anchor'))} |"
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

    json_path = output_dir / "latent_geometry_report.json"
    md_path = output_dir / "latent_geometry_report.md"

    json_path.write_text(
        json.dumps(report, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    torch.save(exported_features, features_path)

    print("\n================ LATENT GEOMETRY FLAGS ================")
    for flag in flags:
        print(f"[{flag['level']}] {flag['code']}: {flag['message']}")
    print("=======================================================")
    print(f"[latent-geometry] JSON: {json_path}")
    print(f"[latent-geometry] Markdown: {md_path}")
    print(f"[latent-geometry] Matched features: {features_path}")


if __name__ == "__main__":
    main()
