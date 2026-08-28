from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader, Dataset, Subset

from backbones.fgclip2_base import (
    FGCLIP2BaseBackbone,
    TextTuningConfig,
    VisionTuningConfig,
)
from cache.taper_mag import FeatureSourcePolicy, FrozenVisionCache, stable_json_hash
from datasets.common import CIRBatch, collate_cir_samples
from datasets.fashioniq import resolve_fashioniq_correction_dicts
from evaluation.fashioniq import (
    build_fashioniq_gallery,
    evaluate_fashioniq_category,
    evaluate_fashioniq_ranking,
    fashioniq_target_ranks,
    macro_average_fashioniq,
)
from models.taper_mag.contracts import PolicyBatch, SupervisionBatch
from models.taper_mag.model import TaperMAG, TaperMAGConfig
from models.taper_mag.rollout import RolloutConfig
from models.taper_mag.utility import HistoryState
from training.checkpointing import load_checkpoint, save_checkpoint
from training.checkpoint_selection import CheckpointSelectionState
from training.ema import ModelEMA, ema_required_for_phase
from training.fashioniq_runtime import CATEGORIES, load_config, make_dataset, seed_everything
from training.marginal_gain_teacher import MarginalGainTeacher
from training.negative_bank import NegativeBank
from training.taper_mag_engine import (
    CurriculumStage,
    EngineConfig,
    TaperMAGTrainingEngine,
    encode_policy_batch,
)
from training.taper_mag_curriculum import CurriculumScheduler, CurriculumState
from training.taper_mag_diagnostics import summarize_training_diagnostics
from training.taper_mag_audit import (
    TeacherShadowAuditor,
    causal_operator_interventions,
    dynamic_frozen_audit,
    mean_audit_reports,
    teacher_shadow_firewall_passes,
    validate_teacher_shadow_provenance,
    write_json,
    write_policy_traces,
)
from training.taper_mag_optimizer import (
    OptimizerConfig,
    build_optimizer,
    format_parameter_report,
)
from training.taper_mag_profiler import profile_taper_runtime
from training.taper_mag_reports import (
    EpochHealthAccumulator,
    GradientRuntimeTracker,
    build_functional_health_report,
    sampled_policy_trace_records,
    static_firewall_report,
)
from training.run_manifest import build_run_manifest, write_run_manifest
from training.text_drift import TextDriftMonitor, text_block_gradient_norms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TAPER-MAG V4 on FashionIQ")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--teacher-shadow-audit", action="store_true")
    parser.add_argument("--profile-runtime", action="store_true")
    parser.add_argument("--audit-samples", type=int)
    return parser.parse_args()


def set_dataset_epoch(dataset: Dataset, epoch: int) -> None:
    base = dataset.dataset if isinstance(dataset, Subset) else dataset
    if hasattr(base, "set_epoch"):
        base.set_epoch(epoch)


def build_policy_batch(
    batch: CIRBatch,
    cache: FrozenVisionCache,
    backbone: FGCLIP2BaseBackbone,
    device: torch.device,
) -> PolicyBatch:
    reference_ids = tuple(batch.reference_ids)
    texts = tuple(batch.modification_texts)
    local, local_mask, shapes = cache.dense_by_ids(reference_ids)
    reference_global = cache.global_by_ids(reference_ids)
    tokenized = backbone.tokenize_texts(texts)
    return PolicyBatch(
        reference_ids=reference_ids,
        modification_texts=texts,
        reference_local=local.to(device).float(),
        reference_local_mask=local_mask.to(device),
        reference_global=reference_global.to(device).float(),
        text_input_ids=tokenized.input_ids.to(device),
        text_attention_mask=tokenized.attention_mask.to(device),
        text_content_mask=tokenized.content_mask.to(device),
        spatial_shapes=shapes.to(device),
    )


def build_supervision_batch(
    batch: CIRBatch,
    cache: FrozenVisionCache,
    device: torch.device,
) -> SupervisionBatch:
    if any(target_id is None for target_id in batch.target_ids):
        raise ValueError("Training/validation FashionIQ sample is missing target ID")
    target_ids = tuple(str(target_id) for target_id in batch.target_ids)
    target_global = cache.global_by_ids(target_ids)
    return SupervisionBatch(
        target_embedding=target_global.to(device).float(),
        target_ids=target_ids,
        positive_ids=tuple((target_id,) for target_id in target_ids),
    )


def validate_cache(cache: FrozenVisionCache, backbone: FGCLIP2BaseBackbone, split: str) -> None:
    manifest = backbone.manifest()
    processor_hash = stable_json_hash(manifest.image_processor_config)
    for cache_manifest, expected_dim, expected_normalization in (
        (cache.global_manifest, backbone.contract.retrieval_dim, "L2"),
        (cache.dense_manifest, backbone.contract.vision_dim, "none"),
    ):
        expected = {
            "model_id": backbone.model_id,
            "revision": backbone.revision,
            "processor_config_hash": processor_hash,
            "feature_dim": expected_dim,
            "patch_policy": manifest.vision_patch_policy,
            "split": split,
            "normalization": expected_normalization,
            "complete_split": True,
            "cache_kind": "global" if expected_normalization == "L2" else "dense_reference",
            "image_scope": "complete_split" if expected_normalization == "L2" else "reference_only",
        }
        actual = cache_manifest.__dict__ if hasattr(cache_manifest, "__dict__") else {
            field: getattr(cache_manifest, field) for field in expected
        }
        differences = {key: (value, actual[key]) for key, value in expected.items() if actual[key] != value}
        if differences:
            raise RuntimeError(f"Cache/backbone contract mismatch: {differences}")


@torch.no_grad()
def validate_fashioniq(
    model: TaperMAG,
    backbone: FGCLIP2BaseBackbone,
    cache: FrozenVisionCache,
    config: dict[str, Any],
    device: torch.device,
    curriculum: CurriculumState,
    correction_dicts: dict[str, dict[str, str]] | None,
) -> dict[str, Any]:
    model.eval()
    backbone.eval()
    category_results: dict[str, dict[str, float]] = {}
    dataset_root = Path(config["data"]["dataset_root"])
    for category in CATEGORIES:
        dataset = make_dataset(config, "val", (category,), correction_dicts)
        loader = DataLoader(
            dataset,
            batch_size=int(config["training"]["eval_batch_size"]),
            shuffle=False,
            num_workers=int(config["training"]["num_workers"]),
            collate_fn=collate_cir_samples,
        )
        gallery_ids = build_fashioniq_gallery(
            protocol=config["data"]["validation_protocol"],
            split_root=dataset_root / "image_splits",
            category=category,
            annotations=dataset.annotations,
            split="val",
        )
        gallery = cache.global_by_ids(gallery_ids).to(device).float()
        scores: list[torch.Tensor] = []
        target_ids: list[str] = []
        reference_ids: list[str] = []
        for raw_batch in loader:
            policy = build_policy_batch(raw_batch, cache, backbone, device)
            use_bf16 = device.type == "cuda" and config["runtime"]["precision"] == "bf16"
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                encoded = encode_policy_batch(backbone, policy)
                if curriculum.phase in {
                    CurriculumStage.ACTOR_WARMUP,
                    CurriculumStage.UTILITY_SHADOW,
                }:
                    rollout = RolloutConfig(max_steps=1, selection_mode="uniform")
                elif curriculum.phase == CurriculumStage.CRITIC_WARMUP:
                    rollout = RolloutConfig(
                        max_steps=1,
                        selection_mode="soft",
                        selection_temperature=curriculum.selection_temperature,
                    )
                else:
                    rollout = RolloutConfig(
                        max_steps=curriculum.horizon,
                        selection_mode="learned",
                        straight_through=False,
                        exploration_probability=0.0,
                    )
                query = model(encoded, rollout).final_query.float()
            scores.append((query @ gallery.T).cpu())
            reference_ids.extend(str(reference_id) for reference_id in raw_batch.reference_ids)
            target_ids.extend(str(target_id) for target_id in raw_batch.target_ids)
        category_results[category] = evaluate_fashioniq_category(
            torch.cat(scores),
            target_ids,
            gallery_ids,
            protocol=str(config["data"]["validation_protocol"]),
            reference_ids=reference_ids,
        )
    average = macro_average_fashioniq(category_results)
    metrics = dict(average)
    metrics["validation_protocol"] = str(config["data"]["validation_protocol"])
    metrics["reference_exclusion"] = (
        config["data"]["validation_protocol"] == "fashioniq_val"
    )
    for category, values in category_results.items():
        metrics.update({f"{category}_{key}": value for key, value in values.items()})
    return metrics


@torch.no_grad()
def validate_functional_controls(
    model: TaperMAG,
    backbone: FGCLIP2BaseBackbone,
    cache: FrozenVisionCache,
    config: dict[str, Any],
    device: torch.device,
    curriculum: CurriculumState,
    correction_dicts: dict[str, dict[str, str]] | None,
    negative_bank: NegativeBank,
    teacher: MarginalGainTeacher,
) -> dict[str, Any]:
    """Audit-only causal controls and dynamic/frozen retrieval on one VAL gallery."""
    protocol = str(config["data"]["validation_protocol"])
    if protocol != "fashioniq_val":
        raise ValueError(
            "Primary TAPER functional audit requires validation_protocol=fashioniq_val"
        )
    model.eval()
    backbone.eval()
    requested = int(
        config.get("diagnostics", {}).get("functional_audit_samples", 32)
    )
    if requested < len(CATEGORIES):
        raise ValueError(
            "diagnostics.functional_audit_samples must include at least one sample "
            "per FashionIQ category"
        )
    base_count, remainder = divmod(requested, len(CATEGORIES))
    category_limits = {
        category: base_count + int(index < remainder)
        for index, category in enumerate(CATEGORIES)
    }
    variants = (
        "full_dynamic",
        "frozen_t0",
        "reference_only",
        "repeat_best",
        "mean_repeat",
        "clone_all_best",
        "clone_all_mean",
        "operator_zero",
        "operator_mean",
    )
    category_metrics: dict[str, dict[str, dict[str, float]]] = {}
    gallery_sizes: dict[str, int] = {}
    ranks_by_variant: dict[str, list[torch.Tensor]] = {
        name: [] for name in variants
    }
    complementary_reports: list[dict[str, Any]] = []
    sample_count = 0
    dataset_root = Path(config["data"]["dataset_root"])

    for category in CATEGORIES:
        full_dataset = make_dataset(config, "val", (category,), correction_dicts)
        audit_dataset = Subset(
            full_dataset,
            range(min(category_limits[category], len(full_dataset))),
        )
        loader = DataLoader(
            audit_dataset,
            batch_size=int(config["training"]["eval_batch_size"]),
            shuffle=False,
            num_workers=int(config["training"]["num_workers"]),
            collate_fn=collate_cir_samples,
        )
        gallery_ids = build_fashioniq_gallery(
            protocol=protocol,
            split_root=dataset_root / "image_splits",
            category=category,
            annotations=full_dataset.annotations,
            split="val",
        )
        gallery = cache.global_by_ids(gallery_ids).to(device).float()
        gallery_sizes[category] = len(gallery_ids)
        query_batches: dict[str, list[torch.Tensor]] = {
            name: [] for name in variants
        }
        target_ids: list[str] = []
        reference_ids: list[str] = []
        for raw_batch in loader:
            policy = build_policy_batch(raw_batch, cache, backbone, device)
            supervision = build_supervision_batch(raw_batch, cache, device)
            encoded = encode_policy_batch(backbone, policy)
            dynamic = model(
                encoded,
                RolloutConfig(
                    max_steps=curriculum.horizon,
                    selection_mode="learned",
                    straight_through=False,
                    exploration_probability=0.0,
                    step_cost=curriculum.step_cost,
                ),
                detach_utility_inputs=True,
            )
            frozen = model(
                encoded,
                RolloutConfig(
                    max_steps=curriculum.horizon,
                    selection_mode="frozen_order",
                    straight_through=False,
                    exploration_probability=0.0,
                    step_cost=curriculum.step_cost,
                ),
                detach_utility_inputs=True,
            )
            _, initial, operators = model.prepare(encoded)
            history = HistoryState.initialize(
                initial.local.shape[0],
                operators.operators.shape[1],
                initial.local.shape[1],
                initial.local.device,
            )
            current, _, candidate_readout, _ = model.preview(
                initial,
                operators,
                history,
                step=0,
                max_steps=curriculum.horizon,
                detach_utility_inputs=True,
            )
            negatives = negative_bank.mine_once(current.query, supervision)
            labels = teacher.score(
                current.query,
                candidate_readout.query,
                supervision,
                negatives,
                step_cost=curriculum.step_cost,
            )
            controls = causal_operator_interventions(
                model,
                encoded,
                labels.raw_gain.argmax(dim=-1),
                max_steps=curriculum.horizon,
                step_cost=curriculum.step_cost,
            )
            batch_queries = {
                "full_dynamic": dynamic.final_query,
                "frozen_t0": frozen.final_query,
                "reference_only": current.query,
                **controls,
            }
            for name in variants:
                query_batches[name].append(batch_queries[name].float().cpu())
            reference_ids.extend(str(reference_id) for reference_id in raw_batch.reference_ids)
            target_ids.extend(str(target_id) for target_id in raw_batch.target_ids)
            complementary_reports.append(
                dynamic_frozen_audit(
                    model,
                    encoded,
                    supervision,
                    negative_bank,
                    teacher,
                    max_steps=curriculum.horizon,
                    step_cost=curriculum.step_cost,
                )
            )
        sample_count += len(target_ids)
        category_metrics[category] = {}
        for name in variants:
            scores = torch.cat(query_batches[name]).to(device) @ gallery.T
            category_metrics[category][name] = evaluate_fashioniq_ranking(
                scores.cpu(),
                target_ids,
                gallery_ids,
                protocol=protocol,
                reference_ids=reference_ids,
            )
            ranks_by_variant[name].append(
                fashioniq_target_ranks(
                    scores.cpu(),
                    target_ids,
                    gallery_ids,
                    protocol=protocol,
                    reference_ids=reference_ids,
                )
            )

    aggregate: dict[str, dict[str, float]] = {}
    for name in variants:
        category_recall = {
            category: category_metrics[category][name] for category in CATEGORIES
        }
        macro = macro_average_fashioniq(category_recall)
        ranks = torch.cat(ranks_by_variant[name]).float()
        aggregate[name] = {
            **macro,
            "mean_target_rank": float(ranks.mean()),
            "median_target_rank": float(ranks.median()),
            "mrr": float(ranks.reciprocal().mean()),
        }

    dynamic_ranks = torch.cat(ranks_by_variant["full_dynamic"]).float()
    frozen_ranks = torch.cat(ranks_by_variant["frozen_t0"]).float()
    if curriculum.horizon == 1:
        dynamic_vs_frozen: dict[str, Any] = {
            "valid": True,
            "status": "not_applicable_horizon_1",
            "validation_protocol": protocol,
            "reference_exclusion": True,
        }
    else:
        dynamic_vs_frozen = {
            "valid": True,
            "status": "audited",
            "validation_protocol": protocol,
            "reference_exclusion": True,
            "dynamic": aggregate["full_dynamic"],
            "frozen": aggregate["frozen_t0"],
            "delta": {
                key: aggregate["full_dynamic"][key]
                - aggregate["frozen_t0"][key]
                for key in (
                    "recall_at_10",
                    "recall_at_50",
                    "mean_recall",
                    "mean_target_rank",
                    "median_target_rank",
                    "mrr",
                )
            },
            "target_rank_improved_fraction": float(
                (dynamic_ranks < frozen_ranks).float().mean()
            ),
            "target_rank_worsened_fraction": float(
                (dynamic_ranks > frozen_ranks).float().mean()
            ),
            "same_gallery": True,
        }

    full_mean = aggregate["full_dynamic"]["mean_recall"]
    reference_mean = aggregate["reference_only"]["mean_recall"]
    intervention_report: dict[str, Any] = {
        "execution_contract": "operator_to_executor_to_state_to_readout",
        "query_delta_arithmetic_used": False,
        "best_selection_rule": "detached_t0_common_negative_teacher_argmax_audit_only",
    }
    for name in variants[3:]:
        metrics = aggregate[name]
        intervention_report[name] = {
            **metrics,
            "paired_delta_vs_full": {
                key: metrics[key] - aggregate["full_dynamic"][key]
                for key in ("recall_at_10", "recall_at_50", "mean_recall")
            },
            "paired_delta_vs_reference": {
                key: metrics[key] - aggregate["reference_only"][key]
                for key in ("recall_at_10", "recall_at_50", "mean_recall")
            },
        }
        if name in {"repeat_best", "mean_repeat"}:
            denominator = full_mean - reference_mean
            intervention_report[name]["rho_repeat"] = (
                (metrics["mean_recall"] - reference_mean) / denominator
                if abs(denominator) > 1e-12
                else None
            )
            intervention_report[name]["rho_repeat_denominator_guarded"] = (
                abs(denominator) <= 1e-12
            )

    complementary = mean_audit_reports(complementary_reports)
    return {
        "schema_version": 1,
        "validation_protocol": protocol,
        "reference_exclusion": True,
        "gallery_semantics": "ordered_unique_union_of_val_reference_and_target_ids_per_category",
        "audit_subset": "first_N_official_validation_triplets_per_category",
        "requested_sample_count": requested,
        "sample_count": sample_count,
        "gallery_sizes": gallery_sizes,
        "categories": category_metrics,
        "variants": aggregate,
        "dynamic_vs_frozen": dynamic_vs_frozen,
        "interventions": intervention_report,
        "complementary_local_diagnostics": complementary,
    }


def engine_config_for(
    curriculum: CurriculumState,
    config: dict[str, Any],
) -> EngineConfig:
    result = EngineConfig(
        stage=curriculum.phase,
        horizon=curriculum.horizon,
        step_cost=curriculum.step_cost,
        retrieval_temperature=float(config["loss"]["retrieval_temperature"]),
        utility_weight=float(config["loss"]["utility_weight"]),
        oracle_mix=curriculum.oracle_mix,
        straight_through=curriculum.straight_through,
        selection_temperature=curriculum.selection_temperature,
        rho_gate=curriculum.rho_gate,
        exploration_probability=curriculum.exploration_probability,
    )
    result.validate()
    return result


def verify_resume_schedule_config(
    payload: dict[str, Any], current_config: dict[str, Any]
) -> None:
    saved_training = payload["config"]["training"]
    current_training = current_config["training"]
    keys = (
        "curriculum_mode",
        "epochs",
        "max_optimizer_updates",
        "health_gate_mode",
        "bypass_health_gates_for_smoke",
    )
    differences = {
        key: {
            "saved": saved_training.get(key),
            "current": current_training.get(key),
        }
        for key in keys
        if saved_training.get(key) != current_training.get(key)
    }
    if differences:
        raise RuntimeError(f"Resume schedule config mismatch: {differences}")
    saved_protocol = payload["config"].get("data", {}).get("validation_protocol")
    current_protocol = current_config.get("data", {}).get("validation_protocol")
    if saved_protocol != current_protocol:
        raise RuntimeError(
            "Resume validation protocol mismatch: "
            f"saved={saved_protocol!r}, current={current_protocol!r}. "
            "Refusing to mix FashionIQ checkpoint-selection metrics across protocols."
        )


def main() -> None:
    args = parse_args()
    exclusive_modes = sum(
        (args.validate_only, args.teacher_shadow_audit, args.profile_runtime)
    )
    if exclusive_modes > 1:
        raise ValueError(
            "--validate-only, --teacher-shadow-audit, and --profile-runtime are mutually exclusive"
        )
    if (args.teacher_shadow_audit or args.profile_runtime) and args.resume is None:
        raise ValueError("Audit/profile modes require --resume")
    if args.audit_samples is not None and args.audit_samples <= 0:
        raise ValueError("--audit-samples must be positive")
    config = load_config(args.config)
    if (
        "actor_warmup_passed" in config["training"].get("approved_health_gates", [])
        and not config["training"].get("bypass_health_gates_for_smoke", False)
        and args.resume is None
    ):
        raise ValueError(
            "A scientific actor_warmup_passed approval must resume the audited epoch-8 checkpoint"
        )
    seed_everything(int(config["seed"]))
    correction_dicts = resolve_fashioniq_correction_dicts(
        Path(config["data"]["dataset_root"]) / "captions",
        str(config["data"]["correction_policy"]),
    )
    curriculum_scheduler = CurriculumScheduler.from_config(
        config["training"], step_cost=float(config["policy"]["step_cost"])
    )
    configured_epochs = int(config["training"]["epochs"])
    if curriculum_scheduler.mode == "canonical_v4" and configured_epochs != 60:
        raise ValueError("canonical_v4 requires training.epochs=60")
    engine_config_for(curriculum_scheduler.state_for_epoch(1), config)
    device = torch.device(
        config["runtime"]["device"]
        if config["runtime"]["device"] != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    text_cfg = TextTuningConfig(**config["backbone"]["text_tuning"])
    vision_cfg = VisionTuningConfig(**config["backbone"]["vision_tuning"])
    if vision_cfg.mode != "frozen" or vision_cfg.num_unfrozen_blocks != 0:
        raise ValueError("This branch requires vision_tuning.mode=frozen and num_unfrozen_blocks=0")
    FeatureSourcePolicy(
        text_encoder_trainable=text_cfg.mode != "frozen",
        vision_encoder_trainable=False,
        gallery_projection_trainable=False,
        use_cached_text_states=False,
        use_cached_reference_dense=True,
        use_cached_reference_global=True,
        use_cached_gallery_global=True,
    ).validate()
    dtype = torch.bfloat16 if config["runtime"]["precision"] == "bf16" else torch.float32
    backbone = FGCLIP2BaseBackbone(
        revision=config["backbone"]["revision"],
        dtype=dtype if device.type == "cuda" else torch.float32,
        text_tuning=text_cfg,
        vision_tuning=vision_cfg,
    ).to(device)
    taper = TaperMAG(
        TaperMAGConfig(
            text_dim=backbone.contract.text_dim,
            vision_dim=backbone.contract.vision_dim,
            retrieval_dim=backbone.contract.retrieval_dim,
            d_model=int(config["model"]["d_model"]),
            num_queries=int(config["model"]["num_queries"]),
            max_steps=int(config["policy"]["max_steps"]),
        )
    ).to(device)
    ema = ModelEMA(decay=float(config["optimizer"]["ema_decay"]))
    print(format_parameter_report(backbone, taper))

    cache_base = (
        Path(config["data"]["cache_root"])
        / "fashioniq"
        / "fgclip2-base"
        / backbone.revision
    )
    train_cache = FrozenVisionCache.load(cache_base / "train")
    val_cache = FrozenVisionCache.load(cache_base / "val")
    validate_cache(train_cache, backbone, "train")
    validate_cache(val_cache, backbone, "val")
    manifest_hashes = {
        "train_global": train_cache.global_manifest.sha256,
        "train_dense": train_cache.dense_manifest.sha256,
        "val_global": val_cache.global_manifest.sha256,
        "val_dense": val_cache.dense_manifest.sha256,
    }

    def verify_new_actor_gate_evidence(payload: dict[str, Any]) -> None:
        saved_gates = payload["curriculum_state"].get("health_gates", {}).get(
            "approved_transitions", []
        )
        current_gates = config["training"].get("approved_health_gates", [])
        if "actor_warmup_passed" in current_gates and "actor_warmup_passed" not in saved_gates:
            if int(payload["epoch"]) + 1 != 8:
                raise RuntimeError(
                    "actor_warmup_passed requires teacher-shadow evidence from the "
                    "completed canonical epoch-8 checkpoint"
                )
            validate_teacher_shadow_provenance(
                config["training"]["teacher_shadow_report"],
                args.resume,
                manifest_hashes,
            )

    if args.validate_only:
        if args.resume is None:
            raise ValueError("--validate-only requires --resume with a trained checkpoint")
        payload = load_checkpoint(
            args.resume,
            model=taper,
            backbone=backbone,
            expected_manifest_hashes=manifest_hashes,
        )
        saved_epoch = int(payload["epoch"]) + 1
        if payload.get("dataset_epoch") not in {None, saved_epoch}:
            raise RuntimeError("Checkpoint dataset epoch is incompatible with epoch-boundary resume")
        verify_resume_schedule_config(payload, config)
        verify_new_actor_gate_evidence(payload)
        curriculum_scheduler.verify_checkpoint(saved_epoch, payload["curriculum_state"])
        saved_curriculum = curriculum_scheduler.state_for_epoch(saved_epoch)
        ema.load_state_dict(
            payload.get("ema"),
            taper,
            backbone.model,
            expected_active=ema_required_for_phase(saved_curriculum.phase.value),
        )
        with ema.average_parameters(taper, backbone.model):
            metrics = validate_fashioniq(
                taper, backbone, val_cache, config, device, saved_curriculum, correction_dicts
            )
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return
    dataset: Dataset = make_dataset(config, "train", CATEGORIES, correction_dicts)
    if args.max_train_samples is not None:
        if args.max_train_samples <= 0:
            raise ValueError("--max-train-samples must be positive")
        dataset = Subset(dataset, range(min(args.max_train_samples, len(dataset))))
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["training"]["num_workers"]),
        collate_fn=collate_cir_samples,
        pin_memory=device.type == "cuda",
    )
    optimizer = build_optimizer(taper, backbone, OptimizerConfig(**config["optimizer"]))
    accumulation = int(config["training"]["gradient_accumulation"])
    epochs = int(config["training"]["epochs"])
    configured_max = config["training"].get("max_optimizer_updates")
    updates = (
        int(configured_max)
        if configured_max is not None
        else max(1, math.ceil(len(loader) / accumulation) * epochs)
    )
    if updates <= 0:
        raise ValueError("max_optimizer_updates must be positive")
    warmup = max(1, round(0.05 * updates))

    def schedule(step: int) -> float:
        if step < warmup:
            return max(step, 1) / warmup
        progress = (step - warmup) / max(updates - warmup, 1)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    bank = NegativeBank(
        train_cache.global_embeddings,
        tuple(image_id for image_id, _ in sorted(train_cache.name_to_idx.items(), key=lambda item: item[1])),
        hard_negatives=int(config["teacher"]["hard_negatives"]),
    )
    engine = TaperMAGTrainingEngine(
        backbone,
        taper,
        bank,
        MarginalGainTeacher(float(config["teacher"]["retrieval_temperature"])),
    )
    output_dir = Path(config["runtime"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (output_dir / "backbone_manifest.json").write_text(
        json.dumps(asdict(backbone.manifest()), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    write_run_manifest(
        output_dir / "run_manifest.json",
        build_run_manifest(config, backbone.manifest(), manifest_hashes),
    )
    firewall = static_firewall_report(taper)
    write_json(output_dir / "firewall_report.json", firewall)

    if args.teacher_shadow_audit or args.profile_runtime:
        payload = load_checkpoint(
            args.resume,
            model=taper,
            backbone=backbone,
            expected_manifest_hashes=manifest_hashes,
        )
        saved_epoch = int(payload["epoch"]) + 1
        if payload.get("dataset_epoch") not in {None, saved_epoch}:
            raise RuntimeError("Checkpoint dataset epoch is incompatible with epoch-boundary resume")
        verify_resume_schedule_config(payload, config)
        verify_new_actor_gate_evidence(payload)
        curriculum_scheduler.verify_checkpoint(saved_epoch, payload["curriculum_state"])
        saved_curriculum = curriculum_scheduler.state_for_epoch(saved_epoch)
        ema.load_state_dict(
            payload.get("ema"),
            taper,
            backbone.model,
            expected_active=ema_required_for_phase(saved_curriculum.phase.value),
        )
        audit_count = args.audit_samples or int(
            config.get("diagnostics", {}).get("teacher_shadow_samples", 256)
        )
        audit_dataset = Subset(dataset, range(min(audit_count, len(dataset))))
        set_dataset_epoch(audit_dataset, saved_epoch)
        audit_loader = DataLoader(
            audit_dataset,
            batch_size=int(config["training"]["eval_batch_size"]),
            shuffle=False,
            num_workers=int(config["training"]["num_workers"]),
            collate_fn=collate_cir_samples,
        )
        if args.profile_runtime:
            raw_batch = next(iter(audit_loader))
            policy = build_policy_batch(raw_batch, train_cache, backbone, device)
            supervision = build_supervision_batch(raw_batch, train_cache, device)
            with ema.average_parameters(taper, backbone.model):
                report = profile_taper_runtime(
                    engine,
                    policy,
                    supervision,
                    engine_config_for(saved_curriculum, config),
                    optimizer=optimizer,
                    repeats=int(config.get("diagnostics", {}).get("profiler_repeats", 3)),
                )
            write_json(output_dir / "compute_report.json", report)
            print(json.dumps(report, indent=2, sort_keys=True))
            return

        with ema.average_parameters(taper, backbone.model):
            taper.eval()
            backbone.eval()
            auditor = TeacherShadowAuditor(
                taper,
                backbone.model,
                bank,
                engine.teacher,
                seed=int(config["seed"]),
                near_tie_band=float(
                    config.get("diagnostics", {}).get("near_tie_band", 0.0)
                ),
            )
            dynamic_reports: list[dict[str, Any]] = []
            for raw_batch in audit_loader:
                policy = build_policy_batch(raw_batch, train_cache, backbone, device)
                supervision = build_supervision_batch(raw_batch, train_cache, device)
                encoded = encode_policy_batch(backbone, policy)
                auditor.update(
                    encoded,
                    supervision,
                    sample_ids=tuple(raw_batch.sample_ids),
                    reference_ids=tuple(raw_batch.reference_ids),
                    modification_texts=tuple(raw_batch.modification_texts),
                )
                dynamic_reports.append(
                    dynamic_frozen_audit(
                        taper,
                        encoded,
                        supervision,
                        bank,
                        engine.teacher,
                        max_steps=int(config["policy"]["max_steps"]),
                        step_cost=float(config["policy"]["step_cost"]),
                    )
                )
            shadow = auditor.finalize(
                checkpoint_path=args.resume,
                cache_manifest_hashes=manifest_hashes,
            )
            functional_retrieval = validate_functional_controls(
                taper,
                backbone,
                val_cache,
                config,
                device,
                saved_curriculum,
                correction_dicts,
                bank,
                engine.teacher,
            )
        write_json(output_dir / "teacher_shadow_report.json", shadow)
        write_policy_traces(output_dir / "policy_trace_sampled.jsonl", auditor.traces)
        shadow_firewall = {
            "pass": teacher_shadow_firewall_passes(shadow["firewall"]),
            **shadow["firewall"],
        }
        dynamic_report = mean_audit_reports(dynamic_reports)
        retrieval_dynamic = functional_retrieval["dynamic_vs_frozen"]
        write_json(output_dir / "firewall_report.json", shadow_firewall)
        write_json(output_dir / "functional_retrieval.json", functional_retrieval)
        functional = {
            "schema_version": 1,
            "retrieval": {
                "validation_protocol": functional_retrieval["validation_protocol"],
                "audit_subset_variants": functional_retrieval["variants"],
            },
            "actor": shadow["candidate_space"],
            "candidate_space": shadow["candidate_space"],
            "critic": shadow["critic_shadow"],
            "stop": {
                "false_stop_rate": shadow["critic_shadow"]["false_stop_rate"],
                "false_continue_rate": shadow["critic_shadow"]["false_continue_rate"],
            },
            "dynamic_policy": {
                **retrieval_dynamic,
                "local_action_diagnostics": dynamic_report or {"status": "not_run"},
            },
            "repeat": {
                "local_staleness": (dynamic_report or {}).get(
                    "repeat", {"status": "not_run"}
                ),
                "causal_retrieval": {
                    name: functional_retrieval["interventions"][name]
                    for name in ("repeat_best", "mean_repeat")
                },
            },
            "response_rank": shadow["response_rank"],
            "clone_controls": functional_retrieval["interventions"],
            "firewall": shadow_firewall,
            "numerical_health": {"pass": shadow["numerical_health"]["finite"], **shadow["numerical_health"]},
        }
        write_json(output_dir / "functional_health.json", functional)
        print(json.dumps(shadow, indent=2, sort_keys=True))
        return
    drift_batch = backbone.tokenize_texts(
        ["make it red", "add long sleeves", "remove the pattern", "make it more formal"]
    )
    drift_snapshot = TextDriftMonitor.capture(backbone, drift_batch)
    start_epoch = 0
    global_step = 0
    best_metrics = {"mean_recall": -float("inf")}
    checkpoint_selection = CheckpointSelectionState()
    if args.resume:
        payload = load_checkpoint(
            args.resume,
            model=taper,
            backbone=backbone,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_manifest_hashes=manifest_hashes,
        )
        saved_epoch = int(payload["epoch"]) + 1
        if payload.get("dataset_epoch") not in {None, saved_epoch}:
            raise RuntimeError("Checkpoint dataset epoch is incompatible with epoch-boundary resume")
        verify_resume_schedule_config(payload, config)
        verify_new_actor_gate_evidence(payload)
        curriculum_scheduler.verify_checkpoint(saved_epoch, payload["curriculum_state"])
        saved_curriculum = curriculum_scheduler.state_for_epoch(saved_epoch)
        if payload["stage"] != saved_curriculum.phase.value:
            raise RuntimeError("Resume checkpoint stage differs from resolved curriculum")
        start_epoch = int(payload["epoch"]) + 1
        global_step = int(payload["global_step"])
        best_metrics = dict(payload["best_metrics"])
        checkpoint_selection = CheckpointSelectionState.from_state_dict(
            payload.get("checkpoint_selection_state")
        )
        ema.load_state_dict(
            payload.get("ema"),
            taper,
            backbone.model,
            expected_active=ema_required_for_phase(saved_curriculum.phase.value),
        )

    optimizer.zero_grad(set_to_none=True)
    last_gradient_norms: dict[str, float] = {}
    last_diagnostics: dict[str, float] = {}
    for epoch in range(start_epoch, epochs):
        set_dataset_epoch(dataset, epoch + 1)
        curriculum = curriculum_scheduler.state_for_epoch(epoch + 1)
        engine_config = engine_config_for(curriculum, config)
        if ema_required_for_phase(curriculum.phase.value) and not ema.active:
            ema.activate(taper, backbone.model)
        taper.train()
        backbone.train()
        epoch_loss = 0.0
        diagnostics_config = config.get("diagnostics", {})
        epoch_health = EpochHealthAccumulator(
            near_tie_band=float(diagnostics_config.get("near_tie_band", 0.0)),
            step_cost=curriculum.step_cost,
            calibration_bins=int(diagnostics_config.get("calibration_bins", 5)),
        )
        gradient_health = GradientRuntimeTracker(taper.config.num_queries)
        trace_remaining = int(
            diagnostics_config.get("sampled_policy_traces_per_epoch", 8)
        )
        epoch_trace_records: list[dict[str, Any]] = []
        for micro_step, raw_batch in enumerate(loader):
            if global_step >= updates:
                break
            policy = build_policy_batch(raw_batch, train_cache, backbone, device)
            supervision = build_supervision_batch(raw_batch, train_cache, device)
            use_bf16 = device.type == "cuda" and config["runtime"]["precision"] == "bf16"
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                result = engine.step(policy, supervision, engine_config)
                scaled_loss = result.loss / accumulation
            scaled_loss.backward()
            epoch_loss += float(result.loss.detach())
            epoch_health.update(result.model_output, result.teacher_gain)
            if trace_remaining > 0:
                records = sampled_policy_trace_records(
                    sample_ids=tuple(raw_batch.sample_ids),
                    reference_ids=tuple(raw_batch.reference_ids),
                    target_ids=tuple(str(value) for value in raw_batch.target_ids),
                    modification_texts=tuple(raw_batch.modification_texts),
                    output=result.model_output,
                    teacher_gain=result.teacher_gain,
                    negative_ids=result.negative_ids,
                    limit=trace_remaining,
                )
                epoch_trace_records.extend(records)
                trace_remaining -= len(records)
            if (micro_step + 1) % accumulation == 0 or micro_step + 1 == len(loader):
                optimized_parameters = [
                    parameter
                    for group in optimizer.param_groups
                    for parameter in group["params"]
                ]
                gradient_norms = [
                    parameter.grad.detach().float().norm()
                    for parameter in optimized_parameters
                    if parameter.grad is not None
                ]
                pre_clip_norm = float(
                    torch.stack(gradient_norms).norm() if gradient_norms else 0.0
                )
                gradient_health.update(
                    taper,
                    backbone.model,
                    pre_clip_global_norm=pre_clip_norm,
                    clip_threshold=float(config["optimizer"]["gradient_clip"]),
                )
                torch.nn.utils.clip_grad_norm_(
                    optimized_parameters,
                    max_norm=float(config["optimizer"]["gradient_clip"]),
                )
                last_gradient_norms = text_block_gradient_norms(backbone)
                optimizer.step()
                scheduler.step()
                if ema.active:
                    ema.update(taper, backbone.model)
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
        last_diagnostics, policy_health = epoch_health.report()
        gradient_report = gradient_health.report()
        if epoch_trace_records:
            trace_path = output_dir / "policy_trace_sampled.jsonl"
            with trace_path.open("a", encoding="utf-8") as file:
                for trace in epoch_trace_records:
                    file.write(json.dumps({"epoch": epoch + 1, **trace}, sort_keys=True) + "\n")
        dynamic_policy_report: dict[str, Any]
        repeat_report: dict[str, Any]
        audit_policy_health: dict[str, Any]
        with ema.average_parameters(taper, backbone.model):
            metrics = validate_fashioniq(
                taper, backbone, val_cache, config, device, curriculum, correction_dicts
            )
            functional_retrieval = validate_functional_controls(
                taper,
                backbone,
                val_cache,
                config,
                device,
                curriculum,
                correction_dicts,
                bank,
                engine.teacher,
            )
            combined_dynamic = dict(
                functional_retrieval["complementary_local_diagnostics"]
            )
            repeat_report = {
                **combined_dynamic.pop("repeat"),
                "causal_retrieval": {
                    name: functional_retrieval["interventions"][name]
                    for name in ("repeat_best", "mean_repeat")
                },
                "valid": True,
                "validation_protocol": functional_retrieval["validation_protocol"],
                "sample_count": functional_retrieval["sample_count"],
            }
            audit_policy_health = combined_dynamic.pop("critic")
            dynamic_policy_report = {
                **functional_retrieval["dynamic_vs_frozen"],
                "local_action_diagnostics": combined_dynamic,
                "sample_count": functional_retrieval["sample_count"],
            }
        drift = TextDriftMonitor.measure(backbone, drift_snapshot)
        functional_health = build_functional_health_report(
            epoch=epoch + 1,
            phase=curriculum.phase.value,
            retrieval=metrics,
            actor=last_diagnostics,
            critic=audit_policy_health,
            gradients=gradient_report,
            firewall=firewall,
            dynamic_policy=dynamic_policy_report,
            repeat=repeat_report,
            clone_controls=functional_retrieval["interventions"],
        )
        write_json(output_dir / "functional_retrieval.json", functional_retrieval)
        write_json(output_dir / "functional_health.json", functional_health)
        record = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "stage": curriculum.phase.value,
            "horizon": curriculum.horizon,
            "oracle_mix": curriculum.oracle_mix,
            "straight_through": curriculum.straight_through,
            "selection_temperature": curriculum.selection_temperature,
            "rho_gate": curriculum.rho_gate,
            "exploration_probability": curriculum.exploration_probability,
            "ema_active": ema.active,
            "ema_updates": ema.num_updates,
            "train_loss": epoch_loss / max(len(loader), 1),
            **metrics,
            **drift,
            **last_gradient_norms,
            **last_diagnostics,
        }
        validation_prefix = (
            "phase_diagnostic"
            if curriculum.phase
            in {
                CurriculumStage.ACTOR_WARMUP,
                CurriculumStage.UTILITY_SHADOW,
                CurriculumStage.CRITIC_WARMUP,
            }
            else "deployable_hard"
        )
        record.update(
            {f"{validation_prefix}/{name}": value for name, value in metrics.items()}
        )
        with (output_dir / "metrics_val.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, sort_keys=True) + "\n")
        print(json.dumps(record, sort_keys=True))
        if (
            functional_health["firewall"]["pass"]
            and functional_health["numerical_health"]["pass"]
            and metrics["mean_recall"] > best_metrics["mean_recall"]
        ):
            best_metrics = dict(metrics)
        checkpoint_names = checkpoint_selection.select(
            retrieval=metrics,
            policy=audit_policy_health,
            functional=functional_health,
        )
        for name, reason in checkpoint_names.items():
            save_checkpoint(
                output_dir / name,
                model=taper,
                backbone=backbone,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                stage=curriculum.phase.value,
                curriculum_state=curriculum_scheduler.checkpoint_state(epoch + 1),
                resolved_config=config,
                manifest_hashes=manifest_hashes,
                best_metrics=best_metrics,
                ema_state=ema.state_dict(),
                checkpoint_reason=reason,
                validation_metrics=metrics,
                policy_metrics=audit_policy_health,
                functional_health_metrics=functional_health,
                selection_state=checkpoint_selection.state_dict(),
                dataset_epoch=epoch + 1,
                micro_step=None,
            )
        if global_step >= updates:
            break


if __name__ == "__main__":
    main()
