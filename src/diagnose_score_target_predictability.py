"""Diagnostic-only target-predictability and teacher-pool stability audit."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import DataLoader

from data.images import FashionIQImageCollator
from datasets.common import DirectoryImageStore
from datasets.fashioniq import FashionIQDataset
from diagnose_score_feature_sufficiency import (
    EXPECTED_VAL_ORACLE_COUNTS,
    EXPECTED_VAL_ORACLE_STOP_RATE,
    EXPECTED_VAL_ORACLE_UTILITY,
    cache_metadata,
    load_cache,
    merge_rows,
    rows_from_shard,
    sha256_file,
    train_probe,
)
from diagnostics.cohort import (
    load_or_create_manifest,
    sample_ids_fingerprint,
    teacher_batch_fingerprint,
    validate_processed_manifest,
)
from diagnostics.feature_sufficiency import COMPACT_FEATURE_FIELDS, LABEL_FIELD, probe_loss, probe_metrics, to_training_precision
from diagnostics.target_predictability import (
    PRIVILEGED_FIELDS,
    add_privileged_features,
    deterministic_alternative_negative_indices,
    freeze_module,
    recompute_utility_with_target_bank,
    stability_metrics,
    target_aware_probe_from_rows,
    validate_target_cache_rows,
)
from evaluate import validate_checkpoint_backbone_metadata
from models.iag_srme.utils.retrieval import build_teacher_masks
from runtime import configure_torch_runtime, resolve_device, seed_everything
from train import CATEGORIES, build_model
from training.engine import resolve_precision

SOURCE_CHECKPOINT = "outputs/2026-09-21/20-02-45/best.pt"
COMPACT_CACHE = "outputs/diagnostics/2026-09-21/score_feature_sufficiency_t0/compact_cache"
TRAIN_MANIFEST = "outputs/diagnostics/2026-09-21/score_feature_probe_train.json"
TRUE_VAL_MANIFEST = "outputs/diagnostics/2026-09-21/shared_val160_true.json"
SEEDS = (42, 43, 44)


def _cache_fingerprint(path: Path) -> str:
    return hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest()


def _target_metadata(
    *, source_sha: str, compact_manifest_sha: str, manifest: Mapping[str, Any], split: str, batch_size: int
) -> dict[str, Any]:
    sample_ids = [str(row["sample_id"]) for row in manifest["samples"]]
    return {
        "format_version": 1,
        "diagnostic_only": True,
        "source_checkpoint_sha256": source_sha,
        "compact_cache_manifest_sha256": compact_manifest_sha,
        "split": split,
        "manifest_sample_ids_sha256": sample_ids_fingerprint(sample_ids),
        "teacher_batch_grouping_sha256": teacher_batch_fingerprint(sample_ids, batch_size),
        "teacher_batch_size": batch_size,
        "fields": ["sample_ids", "target_ids", "target_embeddings_fp16"],
    }


def _target_paths(cache_dir: Path) -> tuple[Path, Path]:
    return cache_dir / "manifest.json", cache_dir / "shards"


def _load_target_cache(cache_dir: Path, expected: Mapping[str, Any]) -> dict[str, Any] | None:
    manifest_path, _ = _target_paths(cache_dir)
    if not manifest_path.is_file():
        return None
    stored = json.loads(manifest_path.read_text(encoding="utf-8"))
    if stored.get("metadata") != dict(expected):
        raise ValueError("privileged target cache is incompatible with source checkpoint or cohort")
    return stored


def _target_rows(cache_dir: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    path = cache_dir / "shards" / str(record["path"])
    if sha256_file(path) != record["sha256"]:
        raise ValueError(f"privileged target cache shard checksum mismatch: {path}")
    rows = torch.load(path, map_location="cpu", weights_only=True)
    validate_target_cache_rows(rows)
    return rows


def _merge_targets(parts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = {
        "target_embeddings": torch.cat([part["target_embeddings"] for part in parts]),
        "sample_ids": [sample_id for part in parts for sample_id in part["sample_ids"]],
        "target_ids": [target_id for part in parts for target_id in part["target_ids"]],
    }
    validate_target_cache_rows(rows)
    return rows


def _collect_targets(
    *,
    model: Any,
    loader: DataLoader,
    cache_dir: Path,
    metadata: Mapping[str, Any],
    device: torch.device,
    precision: Any,
) -> dict[str, Any]:
    """Only frozen source target embeddings; no labels, queries, or model updates."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    _, shard_dir = _target_paths(cache_dir)
    shard_dir.mkdir(exist_ok=True)
    records, processed = [], []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            batch = batch.to(device)
            if batch.target_pixels is None or any(target_id is None for target_id in batch.target_ids):
                raise ValueError("privileged target cache requires one true target per row")
            with torch.autocast(device_type=device.type, enabled=precision.autocast_enabled, dtype=precision.autocast_dtype):
                targets = model.encode_global_images(batch.target_pixels)
            rows = {
                "target_embeddings": targets.detach().float().cpu().to(torch.float16),
                "sample_ids": list(batch.sample_ids),
                "target_ids": [str(target_id) for target_id in batch.target_ids],
            }
            validate_target_cache_rows(rows)
            path = shard_dir / f"target_{batch_index:06d}.pt"
            torch.save(rows, path)
            records.append({"path": path.name, "rows": len(rows["sample_ids"]), "sha256": sha256_file(path)})
            processed.extend(rows["sample_ids"])
            print(f"[target-predictability/cache] batch={batch_index + 1}/{len(loader)} rows={len(processed)}")
    if not records:
        raise RuntimeError("privileged target cache collected no rows")
    (cache_dir / "manifest.json").write_text(
        json.dumps({"metadata": dict(metadata), "shards": records}, indent=2), encoding="utf-8"
    )
    return {"metadata": dict(metadata), "shards": records, "processed_sample_ids": processed}


def _target_training_rows(rows: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    compact = {name: rows[name] for name in (*COMPACT_FEATURE_FIELDS, LABEL_FIELD, "sample_ids")}
    result = to_training_precision(compact, device)
    result.update({name: rows[name].to(device=device, dtype=torch.float32) for name in PRIVILEGED_FIELDS})
    return result


def _evaluate_target_probe(probe: torch.nn.Module, rows: Mapping[str, Any], device: torch.device, epsilon_stop: float) -> dict[str, Any]:
    batch = _target_training_rows(rows, device)
    scores = probe(batch)
    result = probe_metrics(scores, batch[LABEL_FIELD], epsilon_stop=epsilon_stop)
    result["loss"] = float(probe_loss(scores, batch[LABEL_FIELD]))
    return result


def _train_target_probe(
    train_parts: Sequence[Mapping[str, Any]],
    dev_parts: Sequence[Mapping[str, Any]],
    val_rows: Mapping[str, Any],
    *,
    device: torch.device,
    epochs: int,
    epsilon_stop: float,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    probe = target_aware_probe_from_rows(train_parts[0]).to(device).train()
    optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-4, weight_decay=0.01)
    history, started = [], time.perf_counter()
    for epoch in range(epochs):
        losses = []
        for rows in train_parts:
            batch = _target_training_rows(rows, device)
            optimizer.zero_grad(set_to_none=True)
            loss = probe_loss(probe(batch), batch[LABEL_FIELD])
            loss.backward()
            if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in probe.parameters()):
                raise FloatingPointError("non-finite target-aware probe gradient")
            optimizer.step()
            losses.append(float(loss.detach()))
        probe.eval()
        with torch.no_grad():
            history.append({
                "epoch": epoch + 1,
                "mean_training_loss": sum(losses) / len(losses),
                "train": _evaluate_target_probe(probe, _merge_privileged(train_parts), device, epsilon_stop),
                "probe_dev": _evaluate_target_probe(probe, _merge_privileged(dev_parts), device, epsilon_stop),
            })
        probe.train()
    probe.eval()
    with torch.no_grad():
        val = _evaluate_target_probe(probe, val_rows, device, epsilon_stop)
    return probe, {"parameter_count": sum(parameter.numel() for parameter in probe.parameters()), "training_seconds": time.perf_counter() - started, "epochs": history, "true_val": val}


def _merge_privileged(parts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    base = merge_rows(parts)
    return {**base, **{name: torch.cat([part[name] for part in parts]) for name in PRIVILEGED_FIELDS}}


def _mean_std(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keys = ("loss", "exact_oracle_accuracy", "selected_teacher_utility", "oracle_teacher_utility", "oracle_regret", "harmful_execution_fraction_of_executions", "stop_rate", "oracle_stop_rate")
    output: dict[str, Any] = {}
    for key in keys:
        values = torch.tensor([float(result[key]) for result in results])
        output[key] = {"mean": float(values.mean()), "std": float(values.std(unbiased=False))}
    calibration = ("pearson", "spearman", "sign_agreement_at_zero")
    for key in calibration:
        values = torch.tensor([float(result["score_utility_calibration"][key]) for result in results])
        output[key] = {"mean": float(values.mean()), "std": float(values.std(unbiased=False))}
    return output


def _true_val_deltas(target_free: Mapping[str, Any], target_aware: Mapping[str, Any]) -> dict[str, float]:
    free, aware = target_free["mean_std"], target_aware["mean_std"]
    return {
        "pearson": aware["pearson"]["mean"] - free["pearson"]["mean"],
        "spearman": aware["spearman"]["mean"] - free["spearman"]["mean"],
        "sign_agreement_at_zero": aware["sign_agreement_at_zero"]["mean"] - free["sign_agreement_at_zero"]["mean"],
        "exact_oracle_accuracy": aware["exact_oracle_accuracy"]["mean"] - free["exact_oracle_accuracy"]["mean"],
        "selected_teacher_utility": aware["selected_teacher_utility"]["mean"] - free["selected_teacher_utility"]["mean"],
        "oracle_regret": aware["oracle_regret"]["mean"] - free["oracle_regret"]["mean"],
        "harmful_execution_fraction_of_executions": aware["harmful_execution_fraction_of_executions"]["mean"] - free["harmful_execution_fraction_of_executions"]["mean"],
        "stop_calibration_error": abs(aware["stop_rate"]["mean"] - aware["oracle_stop_rate"]["mean"]) - abs(free["stop_rate"]["mean"] - free["oracle_stop_rate"]["mean"]),
    }


def _oracle_sanity(rows: Mapping[str, Any], epsilon_stop: float) -> dict[str, Any]:
    metrics = probe_metrics(rows[LABEL_FIELD], rows[LABEL_FIELD], epsilon_stop=epsilon_stop)
    occupancy = metrics["oracle_slot_occupancy"]
    actual = tuple(int(occupancy[f"C{slot}"]["count"]) for slot in range(4)) + (int(occupancy["STOP"]["count"]),)
    utility, stop_rate = float(metrics["oracle_teacher_utility"]), float(metrics["oracle_stop_rate"])
    passed = actual == EXPECTED_VAL_ORACLE_COUNTS and abs(utility - EXPECTED_VAL_ORACLE_UTILITY) <= 5e-4 and abs(stop_rate - EXPECTED_VAL_ORACLE_STOP_RATE) <= 5e-4
    return {"passed": passed, "actual": {"oracle_slot_counts": actual, "oracle_utility": utility, "oracle_stop_rate": stop_rate}}


def _subset_rows(rows: Mapping[str, Any], indices: Tensor) -> dict[str, Any]:
    return {name: value.index_select(0, indices) if isinstance(value, Tensor) else [value[index] for index in indices.tolist()] for name, value in rows.items()}


def _pool_stability(
    rows: Mapping[str, Any],
    anchor_targets: Mapping[str, Any],
    reservoir: Mapping[str, Any],
    *,
    bank_size: int,
    pool_count: int,
    epsilon_stop: float,
    temperature: float,
    device: torch.device,
    single_positive: Tensor,
) -> dict[str, Any]:
    if list(rows["sample_ids"]) != list(anchor_targets["sample_ids"]):
        raise ValueError("anchor queries and target embeddings are misaligned")
    validate_target_cache_rows(anchor_targets)
    validate_target_cache_rows(reservoir)
    canonical = rows[LABEL_FIELD].float()
    reservoir_ids = [str(value) for value in reservoir["target_ids"]]
    reservoir_embeddings = reservoir["target_embeddings"].float()
    all_alternatives = []
    with torch.no_grad():
        for anchor in range(canonical.shape[0]):
            anchor_target = anchor_targets["target_embeddings"][anchor].to(device=device, dtype=torch.float32)
            anchor_target_id = str(anchor_targets["target_ids"][anchor])
            negatives = deterministic_alternative_negative_indices(
                reservoir_ids,
                anchor_target_id=anchor_target_id,
                anchor_seed_index=anchor,
                bank_size=bank_size,
                seeds=range(pool_count),
            )
            current = rows["current_query"][anchor : anchor + 1].to(device=device, dtype=torch.float32)
            candidates = rows["candidate_queries"][anchor : anchor + 1].to(device=device, dtype=torch.float32)
            utilities = []
            for indices in negatives:
                negative_bank = reservoir_embeddings.index_select(0, torch.tensor(indices))
                target_bank = torch.cat((anchor_target[None], negative_bank), dim=0).to(device=device, dtype=torch.float32)
                utilities.append(recompute_utility_with_target_bank(current, candidates, target_bank, temperature=temperature).cpu())
            all_alternatives.append(torch.stack(utilities))
    alternatives = torch.stack(all_alternatives, dim=1)
    report = stability_metrics(canonical, alternatives, epsilon_stop=epsilon_stop)
    report["pool_count"] = pool_count
    report["pool_seeds"] = list(range(pool_count))
    report["bank_size"] = bank_size
    report["seed_rule"] = "pool seed + anchor * 1000003"
    report["canonical_single_positive_fraction"] = float(single_positive.float().mean())
    report["canonical_multi_positive_fraction"] = float((~single_positive).float().mean())
    if single_positive.any():
        report["single_positive_only"] = stability_metrics(
            canonical[single_positive], alternatives[:, single_positive], epsilon_stop=epsilon_stop
        )
    return report


def _single_positive_mask(target_ids: Sequence[str], batch_size: int) -> Tensor:
    masks = []
    for start in range(0, len(target_ids), batch_size):
        positive, _, _ = build_teacher_masks(target_ids[start : start + batch_size], torch.device("cpu"))
        masks.append(positive.sum(dim=-1).eq(1))
    return torch.cat(masks)


def _interpret(deltas: Mapping[str, float], val_stability: Mapping[str, Any]) -> str:
    improved = deltas["pearson"] > 0.05 and deltas["selected_teacher_utility"] > 0.0
    stable = val_stability["utility_variance"]["pool_induced_to_canonical_std"] < 0.25 and val_stability["candidate_ranking"]["canonical_oracle_action_agreement_mean"] > 0.8
    if improved and stable:
        return "CASE 1: privileged true-target geometry improves predictability while pool stability is high; hidden target relation is a major missing variable."
    if improved:
        return "CASE 2: privileged true-target geometry improves predictability, but evaluator-pool dependence also contributes."
    if not stable:
        return "CASE 3: target-aware improvement is limited and evaluator-pool dependence is material."
    return "CASE 4: target-aware improvement is limited while pool stability is high; semantic grounding or representation generalization is the likelier bottleneck."


def _render(report: Mapping[str, Any]) -> str:
    lines = [
        "# Score Target Predictability Diagnostic",
        "",
        "- privileged target information is diagnostic-only and unavailable at production inference.",
        f"- source checkpoint SHA256: `{report['source_checkpoint']['sha256']}`",
        f"- TRUE VAL oracle sanity: `{report['true_val_oracle_sanity']['passed']}`",
        "",
        "## Matched 3-seed probes",
        "",
        "| probe | split | Pearson | Spearman | sign@0 | exact oracle | selected utility | regret | STOP |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in report["probes"].items():
        for split in ("train_epoch_5", "probe_dev_epoch_5", "true_val"):
            metrics = result[split]["mean_std"]
            lines.append(f"| {name} | {split} | {metrics['pearson']['mean']:.4f} ± {metrics['pearson']['std']:.4f} | {metrics['spearman']['mean']:.4f} ± {metrics['spearman']['std']:.4f} | {metrics['sign_agreement_at_zero']['mean']:.4f} ± {metrics['sign_agreement_at_zero']['std']:.4f} | {metrics['exact_oracle_accuracy']['mean']:.4f} | {metrics['selected_teacher_utility']['mean']:.5f} | {metrics['oracle_regret']['mean']:.5f} | {metrics['stop_rate']['mean']:.4f} |")
    lines += ["", "## TRUE VAL target-aware minus target-free", "", f"`{report['true_val_deltas']}`", "", "## Negative-pool stability", "", f"- TRUE VAL: `{report['true_val_stability']}`", f"- TRAIN: `{report['train_stability']}`", "", "## Interpretation", "", report["interpretation"]]
    return "\n".join(lines)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    source_path = Path(str(cfg.get("checkpoint", SOURCE_CHECKPOINT)))
    compact_dir = Path(str(cfg.get("probe_cache_dir", COMPACT_CACHE)))
    train_manifest_path = Path(str(cfg.get("probe_train_manifest", TRAIN_MANIFEST)))
    val_manifest_path = Path(str(cfg.get("probe_val_manifest", TRUE_VAL_MANIFEST)))
    if not source_path.is_file() or not compact_dir.is_dir() or not val_manifest_path.is_file():
        raise FileNotFoundError("source checkpoint, immutable compact cache, and fixed TRUE VAL manifest are required")
    if not train_manifest_path.is_file():
        raise FileNotFoundError("fixed TRAIN manifest is required; it will not be regenerated")
    seed_everything(int(cfg.seed), bool(cfg.runtime.deterministic))
    configure_torch_runtime(deterministic=bool(cfg.runtime.deterministic), benchmark=bool(cfg.runtime.benchmark))
    device = resolve_device(str(cfg.runtime.device), int(cfg.runtime.accelerator_index))
    precision = resolve_precision(str(cfg.runtime.precision), device)
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=True)
    validate_checkpoint_backbone_metadata(checkpoint.get("metadata"), str(cfg.backbone.checkpoint), str(cfg.backbone.revision), str(cfg.backbone.global_readout_mode), expected_readout_experiment=str(cfg.backbone.readout_experiment), expected_finetune_policy=str(cfg.backbone.finetune_policy), expected_train_vision=bool(cfg.backbone.train_vision), expected_train_text=bool(cfg.backbone.train_text), expected_train_text_projection=bool(cfg.backbone.train_text_projection), expected_model_config={"num_candidates": int(cfg.model.num_candidates), "max_steps": int(cfg.model.max_steps), "stop_enabled": bool(cfg.model.stop_enabled), "epsilon_stop": float(cfg.model.epsilon_stop), "loss_free_balance_enabled": bool(cfg.model.loss_free_balance_enabled), "loss_free_bias_update_rate": float(cfg.model.loss_free_bias_update_rate), "proposal_mode": str(cfg.model.proposal_mode)})
    model, tokenizer, processor = build_model(cfg)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    freeze_module(model)
    metadata = checkpoint.get("metadata")
    objective_config = metadata.get("objective_config") if isinstance(metadata, Mapping) else None
    if not isinstance(objective_config, Mapping):
        raise ValueError("source checkpoint lacks objective config")
    temperature = float(objective_config["retrieval_temperature"])
    root = Path(cfg.dataset.root)
    collator = FashionIQImageCollator(DirectoryImageStore(root / str(cfg.dataset.image_dir)), tokenizer, processor, int(cfg.backbone.max_text_length), include_targets=True)
    annotation_root = root / str(cfg.dataset.annotation_dir)
    train_dataset = FashionIQDataset(annotation_root, "train", CATEGORIES, caption_policy="ordered_and", seed=int(cfg.seed))
    train_cohort, train_manifest = load_or_create_manifest(train_dataset, train_manifest_path, sample_count=len(train_dataset), batch_size=32, seed=int(cfg.seed), split="train", caption_policy="ordered_and")
    val_dataset = FashionIQDataset(annotation_root, "val", CATEGORIES, caption_policy="ordered_and", seed=int(cfg.seed))
    val_cohort, val_manifest = load_or_create_manifest(val_dataset, val_manifest_path, sample_count=160, batch_size=8, seed=int(cfg.seed), split="val", caption_policy="ordered_and")
    train_loader = DataLoader(train_cohort, batch_size=32, shuffle=False, num_workers=int(cfg.experiment.num_workers), pin_memory=True, collate_fn=collator)
    val_loader = DataLoader(val_cohort, batch_size=8, shuffle=False, num_workers=int(cfg.experiment.num_workers), pin_memory=True, collate_fn=collator)
    source_sha, compact_sha = sha256_file(source_path), _cache_fingerprint(compact_dir)
    compact_before = _cache_fingerprint(compact_dir)
    compact_expected = cache_metadata(source_checkpoint_sha256=source_sha, manifest=train_manifest, split="train", batch_size=32)
    # ponytail: reuse the immutable target-free cache; regenerate it only with the separate feature-sufficiency diagnostic.
    compact = load_cache(compact_dir, compact_expected)
    if compact is None:
        raise FileNotFoundError("immutable compact target-free cache is missing; run feature sufficiency collection first")
    train_parts = [rows_from_shard(compact_dir, record) for record in compact["shards"]]
    train_rows = merge_rows(train_parts)
    validate_processed_manifest(train_manifest, train_rows["sample_ids"], batch_size=32)
    privileged_root = output_dir / "privileged_target_cache"
    train_target_metadata = _target_metadata(source_sha=source_sha, compact_manifest_sha=compact_sha, manifest=train_manifest, split="train", batch_size=32)
    train_target_cache = _load_target_cache(privileged_root / "train", train_target_metadata)
    if train_target_cache is None:
        train_target_cache = _collect_targets(model=model, loader=train_loader, cache_dir=privileged_root / "train", metadata=train_target_metadata, device=device, precision=precision)
    train_targets = _merge_targets([_target_rows(privileged_root / "train", record) for record in train_target_cache["shards"]])
    validate_processed_manifest(train_manifest, train_targets["sample_ids"], batch_size=32)
    val_target_metadata = _target_metadata(source_sha=source_sha, compact_manifest_sha=compact_sha, manifest=val_manifest, split="val", batch_size=8)
    val_target_cache = _load_target_cache(privileged_root / "true_val", val_target_metadata)
    if val_target_cache is None:
        val_target_cache = _collect_targets(
            model=model,
            loader=val_loader,
            cache_dir=privileged_root / "true_val",
            metadata=val_target_metadata,
            device=device,
            precision=precision,
        )
    val_targets = _merge_targets(
        [_target_rows(privileged_root / "true_val", record) for record in val_target_cache["shards"]]
    )
    validate_processed_manifest(val_manifest, val_targets["sample_ids"], batch_size=8)
    val_parts = []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            output = model(batch.reference_pixels, batch.input_ids, batch.attention_mask, batch.content_mask)
            if not output["steps"]:
                raise RuntimeError("frozen source emitted no t0 step for TRUE VAL")
            step = output["steps"][0]
            targets = model.encode_global_images(batch.target_pixels)
            positive, negative, _ = build_teacher_masks(batch.target_ids, device)
            from diagnostics.selection import transition_retrieval
            from diagnostics.feature_sufficiency import compact_t0_features

            transition = transition_retrieval(
                step["current_query"], step["candidate_queries"], targets, positive, negative, temperature
            )
            if not transition["valid"].all():
                raise RuntimeError("TRUE VAL has invalid canonical teacher rows")
            val_parts.append(
                {**compact_t0_features(step, transition["utility"]), "sample_ids": list(batch.sample_ids)}
            )
    val_rows = merge_rows(val_parts)
    validate_processed_manifest(val_manifest, val_rows["sample_ids"], batch_size=8)
    sanity = _oracle_sanity(val_rows, float(cfg.model.epsilon_stop))
    if not sanity["passed"]:
        raise RuntimeError(f"TRUE VAL oracle sanity gate failed: {sanity}")
    val_privileged = add_privileged_features(val_rows, val_targets)
    split_index = max(1, int(len(train_parts) * 0.9))
    if split_index == len(train_parts):
        split_index -= 1
    free_train, free_dev = train_parts[:split_index], train_parts[split_index:]
    aware_parts = [add_privileged_features(part, _subset_rows(train_targets, torch.arange(sum(previous[LABEL_FIELD].shape[0] for previous in train_parts[:index]), sum(previous[LABEL_FIELD].shape[0] for previous in train_parts[: index + 1])))) for index, part in enumerate(train_parts)]
    aware_train, aware_dev = aware_parts[:split_index], aware_parts[split_index:]
    probes: dict[str, Any] = {}
    for name in ("target_free_retrieval_independent", "target_aware_retrieval_independent"):
        records = []
        for seed in SEEDS:
            torch.manual_seed(seed)
            if name == "target_free_retrieval_independent":
                _, result = train_probe("retrieval_augmented_independent", free_train, free_dev, val_rows, device=device, epochs=5, learning_rate=1e-4, weight_decay=0.01, epsilon_stop=float(cfg.model.epsilon_stop))
            else:
                _, result = _train_target_probe(aware_train, aware_dev, val_privileged, device=device, epochs=5, epsilon_stop=float(cfg.model.epsilon_stop))
            records.append(result)
            print(f"[target-predictability/probe] {name} seed={seed} true_val_pearson={result['true_val']['score_utility_calibration']['pearson']:.4f}")
        probes[name] = {
            "seeds": list(SEEDS), "runs": records,
            "train_epoch_5": {"mean_std": _mean_std([item["epochs"][-1]["train"] for item in records])},
            "probe_dev_epoch_5": {"mean_std": _mean_std([item["epochs"][-1]["probe_dev"] for item in records])},
            "true_val": {"mean_std": _mean_std([item["true_val"] for item in records])},
        }
    single_val = _single_positive_mask(val_targets["target_ids"], 8)
    if train_rows[LABEL_FIELD].shape[0] < 512:
        raise RuntimeError("TRAIN stability audit requires the first 16 complete teacher groups (512 rows)")
    first_train = _subset_rows(train_rows, torch.arange(512))
    first_targets = _subset_rows(train_targets, torch.arange(512))
    train_stability = _pool_stability(
        first_train,
        first_targets,
        train_targets,
        bank_size=32,
        pool_count=16,
        epsilon_stop=float(cfg.model.epsilon_stop),
        temperature=temperature,
        device=device,
        single_positive=_single_positive_mask(first_targets["target_ids"], 32),
    )
    val_stability = _pool_stability(
        val_rows,
        val_targets,
        val_targets,
        bank_size=8,
        pool_count=32,
        epsilon_stop=float(cfg.model.epsilon_stop),
        temperature=temperature,
        device=device,
        single_positive=single_val,
    )
    if _cache_fingerprint(compact_dir) != compact_before:
        raise RuntimeError("immutable compact target-free cache was modified")
    deltas = _true_val_deltas(probes["target_free_retrieval_independent"]["true_val"], probes["target_aware_retrieval_independent"]["true_val"])
    report = {"source_checkpoint": {"path": str(source_path), "sha256": source_sha}, "compact_target_free_cache": {"path": str(compact_dir), "manifest_sha256": compact_sha, "immutable_verified": True}, "privileged_target_cache": {"path": str(privileged_root), "diagnostic_only": True}, "train_manifest": validate_processed_manifest(train_manifest, train_rows["sample_ids"], batch_size=32), "val_manifest": validate_processed_manifest(val_manifest, val_rows["sample_ids"], batch_size=8), "true_val_oracle_sanity": sanity, "probe_protocol": {"seeds": list(SEEDS), "epochs": 5, "timestep": 0, "optimizer": "AdamW", "learning_rate": 1e-4, "weight_decay": 0.01, "objective": "absolute_gain_loss(huber_delta=1.0)", "precision": "fp32", "true_val_used_for_optimization": False}, "probes": probes, "true_val_deltas": deltas, "true_val_stability": val_stability, "train_stability": train_stability, "interpretation": _interpret(deltas, val_stability), "production_behavior_unchanged": True}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "score_target_predictability_report.json").write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")
    (output_dir / "score_target_predictability_report.md").write_text(_render(report), encoding="utf-8")


if __name__ == "__main__":
    main()
