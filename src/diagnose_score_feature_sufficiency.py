"""Frozen t0 feature-sufficiency probes for the residual-LN source checkpoint.

This is diagnostic-only: the source model is frozen, teacher utilities are labels only,
and no production module, optimizer, checkpoint, or routing policy is modified.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from data.images import FashionIQImageCollator
from datasets.common import DirectoryImageStore
from datasets.fashioniq import FashionIQDataset
from diagnostics.cohort import (
    load_or_create_manifest,
    sample_ids_fingerprint,
    teacher_batch_fingerprint,
    validate_processed_manifest,
)
from diagnostics.feature_sufficiency import (
    COMPACT_FEATURE_FIELDS,
    LABEL_FIELD,
    cache_fingerprint,
    compact_cache_bytes,
    compact_t0_features,
    probe_from_rows,
    probe_loss,
    probe_metrics,
    to_cache_precision,
    to_training_precision,
    validate_compact_features,
)
from diagnostics.selection import transition_retrieval
from evaluate import validate_checkpoint_backbone_metadata
from models.iag_srme.utils.retrieval import build_teacher_masks
from runtime import configure_torch_runtime, resolve_device, seed_everything
from train import CATEGORIES, build_model
from training.engine import resolve_precision

PROBES = (
    "legacy_independent",
    "legacy_set_relative",
    "retrieval_augmented_independent",
)
BASELINES = {
    "source": {
        "pearson": 0.0026,
        "exact_oracle_accuracy": 0.3000,
        "selected_utility": 0.00480,
        "oracle_utility": 0.06073,
        "regret": 0.05592,
        "harmful_execute": 0.5342,
        "stop_rate": 0.0875,
        "oracle_stop_rate": 0.1063,
    },
    "gain_only_refit": {
        "pearson": 0.0682,
        "exact_oracle_accuracy": 0.1875,
        "selected_utility": 0.00620,
        "oracle_utility": 0.06073,
        "regret": 0.05452,
        "harmful_execute": 0.4035,
        "stop_rate": 0.6438,
        "oracle_stop_rate": 0.1063,
    },
    "pair_gain_refit": {
        "pearson": 0.1023,
        "exact_oracle_accuracy": 0.2812,
        "selected_utility": 0.00720,
        "oracle_utility": 0.06073,
        "regret": 0.05353,
        "harmful_execute": 0.4938,
        "stop_rate": 0.0000,
        "oracle_stop_rate": 0.1063,
    },
}
EXPECTED_VAL_ORACLE_COUNTS = (37, 34, 27, 45, 17)
EXPECTED_VAL_ORACLE_UTILITY = 0.06073
EXPECTED_VAL_ORACLE_STOP_RATE = 0.1063
MAX_CACHE_BYTES = 2 * 1024**3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_metadata(
    *, source_checkpoint_sha256: str, manifest: Mapping[str, Any], split: str, batch_size: int
) -> dict[str, Any]:
    sample_ids = [str(row["sample_id"]) for row in manifest["samples"]]
    return {
        "format_version": 1,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "split": split,
        "manifest_sample_ids_sha256": sample_ids_fingerprint(sample_ids),
        "teacher_batch_grouping_sha256": teacher_batch_fingerprint(sample_ids, batch_size),
        "teacher_batch_size": batch_size,
        "compact_feature_fields": list(COMPACT_FEATURE_FIELDS),
        "label_field": LABEL_FIELD,
    }


def cache_paths(cache_dir: Path) -> tuple[Path, Path]:
    return cache_dir / "manifest.json", cache_dir / "shards"


def load_cache(cache_dir: Path, expected: Mapping[str, Any]) -> dict[str, Any] | None:
    manifest_path, _ = cache_paths(cache_dir)
    if not manifest_path.is_file():
        return None
    stored = json.loads(manifest_path.read_text(encoding="utf-8"))
    if stored.get("metadata") != dict(expected):
        raise ValueError("compact feature cache is incompatible with source checkpoint or manifest")
    if stored.get("fingerprint") != cache_fingerprint(stored["metadata"]):
        raise ValueError("compact feature cache fingerprint mismatch")
    return stored


def save_cache_manifest(cache_dir: Path, metadata: Mapping[str, Any], shards: list[dict[str, Any]]) -> None:
    manifest_path, _ = cache_paths(cache_dir)
    payload = {"metadata": dict(metadata), "fingerprint": cache_fingerprint(metadata), "shards": shards}
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def rows_from_shard(cache_dir: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    path = cache_dir / "shards" / str(record["path"])
    if sha256_file(path) != record["sha256"]:
        raise ValueError(f"compact feature cache shard checksum mismatch: {path}")
    rows = torch.load(path, map_location="cpu", weights_only=True)
    validate_compact_features(rows)
    return rows



def cache_disk_bytes(cache_dir: Path, records: list[Mapping[str, Any]]) -> int:
    return sum((cache_dir / "shards" / str(record["path"])).stat().st_size for record in records)



def expected_cache_bytes(model: Any, *, rows: int) -> int:
    """Estimate the FP16 compact tensor cache before source collection writes a shard."""

    candidates = int(model.config.num_candidates)
    state_dim = int(model.backbone.state_dim)
    text_dim = int(model.backbone.text_dim)
    query_dim = int(model.backbone.retrieval_dim)
    fp16_values = state_dim + text_dim + 3 * candidates * state_dim + (1 + 2 * candidates) * query_dim
    return rows * (2 * fp16_values + 4 * candidates)

def collect_t0_cache(
    *,
    model: Any,
    loader: DataLoader,
    retrieval_temperature: float,
    cache_dir: Path,
    metadata: Mapping[str, Any],
    device: torch.device,
    precision: Any,
) -> dict[str, Any]:
    """Run frozen source t0 once and shard only compact target-free probe inputs."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    _, shard_dir = cache_paths(cache_dir)
    shard_dir.mkdir(exist_ok=True)
    shards = []
    processed_ids: list[str] = []
    estimated_bytes = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            batch = batch.to(device)
            if batch.target_pixels is None:
                raise ValueError("feature sufficiency collection requires target images for labels")
            with torch.autocast(
                device_type=device.type,
                enabled=precision.autocast_enabled,
                dtype=precision.autocast_dtype,
            ):
                output = model(
                    batch.reference_pixels, batch.input_ids, batch.attention_mask, batch.content_mask
                )
                targets = model.encode_global_images(batch.target_pixels)
            if not output["steps"]:
                raise RuntimeError("frozen source emitted no t0 step")
            step = output["steps"][0]
            positive, negative, _ = build_teacher_masks(batch.target_ids, device)
            transition = transition_retrieval(
                step["current_query"],
                step["candidate_queries"],
                targets.detach(),
                positive,
                negative,
                retrieval_temperature,
            )
            if not transition["valid"].all():
                raise RuntimeError(
                    "teacher utility is invalid for a frozen cohort batch; do not silently change "
                    "the fixed teacher grouping"
                )
            rows = to_cache_precision(
                {**compact_t0_features(step, transition["utility"]), "sample_ids": list(batch.sample_ids)}
            )
            estimated_bytes += compact_cache_bytes(rows)
            if estimated_bytes > MAX_CACHE_BYTES:
                raise RuntimeError(
                    f"compact feature cache size {estimated_bytes / 1024**3:.2f} GiB exceeds 2 GiB"
                )
            path = shard_dir / f"t0_{batch_index:06d}.pt"
            torch.save(rows, path)
            shards.append(
                {"path": path.name, "rows": len(rows["sample_ids"]), "sha256": sha256_file(path)}
            )
            processed_ids.extend(rows["sample_ids"])
            print(
                f"[feature-probe/collect] batch={batch_index + 1}/{len(loader)} "
                f"rows={len(processed_ids)}"
            )
    if not shards:
        raise RuntimeError("t0 feature collection produced no valid teacher rows")
    save_cache_manifest(cache_dir, metadata, shards)
    return {
        "metadata": dict(metadata),
        "fingerprint": cache_fingerprint(metadata),
        "shards": shards,
        "estimated_bytes": estimated_bytes,
        "processed_sample_ids": processed_ids,
    }


def merge_rows(parts: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not parts:
        raise ValueError("cannot merge zero compact probe shards")
    rows = {
        **{name: torch.cat([part[name] for part in parts], dim=0) for name in (*COMPACT_FEATURE_FIELDS, LABEL_FIELD)},
        "sample_ids": [sample_id for part in parts for sample_id in part["sample_ids"]],
    }
    validate_compact_features(rows)
    return rows


def train_probe(
    name: str,
    train_parts: list[Mapping[str, Any]],
    dev_parts: list[Mapping[str, Any]],
    val_rows: Mapping[str, Any],
    *,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    epsilon_stop: float,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Fit from scratch over frozen whole teacher-batch groups in deterministic order."""

    probe = probe_from_rows(name, train_parts[0]).to(device).train()
    optimizer = torch.optim.AdamW(probe.parameters(), lr=learning_rate, weight_decay=weight_decay)
    history = []
    started = time.perf_counter()
    for epoch in range(epochs):
        losses = []
        for rows in train_parts:
            batch = to_training_precision(rows, device)
            optimizer.zero_grad(set_to_none=True)
            scores = probe(batch)
            loss = probe_loss(scores, batch[LABEL_FIELD])
            loss.backward()
            if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in probe.parameters()):
                raise FloatingPointError(f"non-finite gradient in {name}")
            optimizer.step()
            losses.append(float(loss.detach()))
        probe.eval()
        with torch.no_grad():
            train_eval = evaluate_probe(probe, merge_rows(train_parts), device, epsilon_stop)
            dev_eval = evaluate_probe(probe, merge_rows(dev_parts), device, epsilon_stop)
        history.append(
            {
                "epoch": epoch + 1,
                "mean_training_loss": sum(losses) / len(losses),
                "train": train_eval,
                "probe_dev": dev_eval,
            }
        )
        probe.train()
    probe.eval()
    with torch.no_grad():
        val = evaluate_probe(probe, val_rows, device, epsilon_stop)
    return probe, {
        "parameter_count": sum(parameter.numel() for parameter in probe.parameters()),
        "training_seconds": time.perf_counter() - started,
        "epochs": history,
        "true_val": val,
    }


def evaluate_probe(
    probe: torch.nn.Module, rows: Mapping[str, Any], device: torch.device, epsilon_stop: float
) -> dict[str, Any]:
    batch = to_training_precision(rows, device)
    scores = probe(batch)
    result = probe_metrics(scores, batch[LABEL_FIELD], epsilon_stop=epsilon_stop)
    result["loss"] = float(probe_loss(scores, batch[LABEL_FIELD]))
    return result


def validate_true_val_oracle(metrics: Mapping[str, Any]) -> dict[str, Any]:
    oracle = metrics["oracle_slot_occupancy"]
    actual_counts = tuple(int(oracle[f"C{slot}"]["count"]) for slot in range(4)) + (
        int(oracle["STOP"]["count"]),
    )
    utility = float(metrics["oracle_teacher_utility"])
    stop_rate = float(metrics["oracle_stop_rate"])
    passed = (
        actual_counts == EXPECTED_VAL_ORACLE_COUNTS
        and abs(utility - EXPECTED_VAL_ORACLE_UTILITY) <= 5e-4
        and abs(stop_rate - EXPECTED_VAL_ORACLE_STOP_RATE) <= 5e-4
    )
    return {
        "passed": passed,
        "expected": {
            "oracle_slot_counts": EXPECTED_VAL_ORACLE_COUNTS,
            "oracle_utility": EXPECTED_VAL_ORACLE_UTILITY,
            "oracle_stop_rate": EXPECTED_VAL_ORACLE_STOP_RATE,
        },
        "actual": {
            "oracle_slot_counts": actual_counts,
            "oracle_utility": utility,
            "oracle_stop_rate": stop_rate,
        },
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Frozen T0 Score Feature-Sufficiency Probe",
        "",
        f"- source checkpoint: `{report['source_checkpoint']['path']}`",
        f"- source SHA256: `{report['source_checkpoint']['sha256']}`",
        f"- train manifest: `{report['train_manifest']['path']}`",
        f"- true VAL manifest: `{report['val_manifest']['path']}`",
        "- t0 only; teacher utility is a label and never a probe input.",
        "",
        "## TRUE VAL oracle sanity check",
        "",
        f"- passed: `{report['oracle_sanity_check']['passed']}`",
        f"- actual: `{report['oracle_sanity_check']['actual']}`",
        "",
        "## Comparison",
        "",
        "| probe | Pearson | Spearman | sign@0 | exact oracle | selected utility | oracle utility | regret | harmful / execute | STOP | oracle STOP | dominant slot |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, values in report["probes"].items():
        metrics = values["true_val"]
        calibration = metrics["score_utility_calibration"]
        selected = metrics["selected_slot_occupancy"]
        dominant = max(selected, key=lambda slot: selected[slot]["count"])
        lines.append(
            f"| {name} | {calibration['pearson']:.4f} | {calibration['spearman']:.4f} | "
            f"{calibration['sign_agreement_at_zero']:.4f} | {metrics['exact_oracle_accuracy']:.4f} | "
            f"{metrics['selected_teacher_utility']:.5f} | {metrics['oracle_teacher_utility']:.5f} | "
            f"{metrics['oracle_regret']:.5f} | {metrics['harmful_execution_fraction_of_executions']:.4f} | "
            f"{metrics['stop_rate']:.4f} | {metrics['oracle_stop_rate']:.4f} | {dominant} |"
        )
    lines += ["", "## Known production/refit TRUE VAL t0 baselines", ""]
    for name, values in report["baselines"].items():
        lines.append(f"- `{name}`: {values}")
    return "\n".join(lines)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    source_path = Path(str(cfg.get("checkpoint", "outputs/2026-09-21/20-02-45/best.pt")))
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    cache_dir = Path(str(cfg.get("probe_cache_dir", output_dir / "compact_cache")))
    train_manifest_path = Path(
        str(cfg.get("probe_train_manifest", output_dir.parent / "score_feature_probe_train.json"))
    )
    val_manifest_path = Path(
        str(cfg.get("probe_val_manifest", "outputs/diagnostics/2026-09-21/shared_val160.json"))
    )
    train_batch_size = int(cfg.get("probe_train_teacher_batch_size", 32))
    epochs = int(cfg.get("probe_epochs", 5))
    if not source_path.is_file():
        raise FileNotFoundError(f"source checkpoint does not exist: {source_path}")

    seed_everything(int(cfg.seed), bool(cfg.runtime.deterministic))
    configure_torch_runtime(
        deterministic=bool(cfg.runtime.deterministic), benchmark=bool(cfg.runtime.benchmark)
    )
    device = resolve_device(str(cfg.runtime.device), int(cfg.runtime.accelerator_index))
    precision = resolve_precision(str(cfg.runtime.precision), device)
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=True)
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
            "loss_free_balance_enabled": bool(cfg.model.loss_free_balance_enabled),
            "loss_free_bias_update_rate": float(cfg.model.loss_free_bias_update_rate),
            "proposal_mode": str(cfg.model.proposal_mode),
        },
    )
    model, tokenizer, processor = build_model(cfg)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    metadata = checkpoint.get("metadata")
    objective_config = metadata.get("objective_config") if isinstance(metadata, Mapping) else None
    if not isinstance(objective_config, Mapping):
        raise ValueError("source checkpoint lacks objective_config metadata")
    retrieval_temperature = float(objective_config["retrieval_temperature"])
    root = Path(cfg.dataset.root)
    annotation_root = root / str(cfg.dataset.annotation_dir)
    image_store = DirectoryImageStore(root / str(cfg.dataset.image_dir))
    collator = FashionIQImageCollator(
        image_store, tokenizer, processor, int(cfg.backbone.max_text_length), include_targets=True
    )
    train_dataset = FashionIQDataset(
        annotation_root,
        "train",
        CATEGORIES,
        caption_policy="ordered_and",
        seed=int(cfg.seed),
    )
    train_cohort, train_manifest = load_or_create_manifest(
        train_dataset,
        train_manifest_path,
        sample_count=len(train_dataset),
        batch_size=train_batch_size,
        seed=int(cfg.seed),
        split="train",
        caption_policy="ordered_and",
    )
    train_loader = DataLoader(
        train_cohort,
        batch_size=train_batch_size,
        shuffle=False,
        num_workers=int(cfg.experiment.num_workers),
        pin_memory=True,
        collate_fn=collator,
        drop_last=False,
    )
    expected_train_cache_bytes = expected_cache_bytes(model, rows=len(train_cohort))
    print(
        f"[feature-probe/cache] estimated compact payload={expected_train_cache_bytes / 1024**2:.1f} MiB"
    )
    if expected_train_cache_bytes > MAX_CACHE_BYTES:
        raise RuntimeError(
            f"estimated compact feature cache {expected_train_cache_bytes / 1024**3:.2f} GiB exceeds 2 GiB"
        )
    source_sha = sha256_file(source_path)
    train_metadata = cache_metadata(
        source_checkpoint_sha256=source_sha,
        manifest=train_manifest,
        split="train",
        batch_size=train_batch_size,
    )
    cache = load_cache(cache_dir, train_metadata)
    if cache is None:
        cache = collect_t0_cache(
            model=model,
            loader=train_loader,
            retrieval_temperature=retrieval_temperature,
            cache_dir=cache_dir,
            metadata=train_metadata,
            device=device,
            precision=precision,
        )
    train_processed = cache.get("processed_sample_ids") or [
        sample_id
        for record in cache["shards"]
        for sample_id in rows_from_shard(cache_dir, record)["sample_ids"]
    ]
    train_manifest_metadata = validate_processed_manifest(
        train_manifest, train_processed, batch_size=train_batch_size
    )
    parts = [rows_from_shard(cache_dir, record) for record in cache["shards"]]
    if len(parts) < 2:
        raise RuntimeError("feature sufficiency needs at least two fixed teacher-batch groups")
    split_index = max(1, int(len(parts) * 0.9))
    if split_index == len(parts):
        split_index -= 1
    train_parts, dev_parts = parts[:split_index], parts[split_index:]

    val_dataset = FashionIQDataset(
        annotation_root, "val", CATEGORIES, caption_policy="ordered_and", seed=int(cfg.seed)
    )
    if not val_manifest_path.is_file():
        raise FileNotFoundError(
            f"fixed TRUE VAL manifest is required and will not be regenerated: {val_manifest_path}"
        )
    val_cohort, val_manifest = load_or_create_manifest(
        val_dataset,
        val_manifest_path,
        sample_count=160,
        batch_size=8,
        seed=int(cfg.seed),
        split="val",
        caption_policy="ordered_and",
    )
    val_loader = DataLoader(
        val_cohort,
        batch_size=8,
        shuffle=False,
        num_workers=int(cfg.experiment.num_workers),
        pin_memory=True,
        collate_fn=collator,
        drop_last=False,
    )
    val_parts = []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            assert batch.target_pixels is not None
            with torch.autocast(
                device_type=device.type,
                enabled=precision.autocast_enabled,
                dtype=precision.autocast_dtype,
            ):
                output = model(
                    batch.reference_pixels, batch.input_ids, batch.attention_mask, batch.content_mask
                )
                targets = model.encode_global_images(batch.target_pixels)
            step = output["steps"][0]
            positive, negative, _ = build_teacher_masks(batch.target_ids, device)
            transition = transition_retrieval(
                step["current_query"],
                step["candidate_queries"],
                targets.detach(),
                positive,
                negative,
                retrieval_temperature,
            )
            if not transition["valid"].all():
                raise RuntimeError(
                    "TRUE VAL teacher utility is invalid for a fixed cohort batch; do not interpret probes"
                )
            val_parts.append(
                {**compact_t0_features(step, transition["utility"]), "sample_ids": list(batch.sample_ids)}
            )
    val_rows = merge_rows(val_parts)
    val_processed = [sample_id for part in val_parts for sample_id in part["sample_ids"]]
    val_manifest_metadata = validate_processed_manifest(val_manifest, val_processed, batch_size=8)
    oracle_scores = val_rows[LABEL_FIELD]
    sanity = probe_metrics(oracle_scores, oracle_scores, epsilon_stop=float(model.config.epsilon_stop))
    sanity_check = validate_true_val_oracle(sanity)
    if not sanity_check["passed"]:
        raise RuntimeError(f"TRUE VAL oracle sanity check failed: {sanity_check}")

    probes = {}
    torch.manual_seed(int(cfg.seed))
    for name in PROBES:
        probe, result = train_probe(
            name,
            train_parts,
            dev_parts,
            val_rows,
            device=device,
            epochs=epochs,
            learning_rate=1e-4,
            weight_decay=0.01,
            epsilon_stop=float(model.config.epsilon_stop),
        )
        torch.save(probe.state_dict(), output_dir / f"{name}.pt")
        probes[name] = result

    report = {
        "source_checkpoint": {
            "path": str(source_path),
            "sha256": source_sha,
            "source_git_sha": metadata.get("run", {}).get("git_sha")
            if isinstance(metadata, Mapping)
            else None,
        },
        "source_model_config": metadata.get("model_config") if isinstance(metadata, Mapping) else None,
        "source_objective_config": objective_config,
        "train_manifest": {"path": str(train_manifest_path), **train_manifest_metadata},
        "val_manifest": {"path": str(val_manifest_path), **val_manifest_metadata},
        "compact_cache": {
            "path": str(cache_dir),
            "estimated_bytes": cache.get("estimated_bytes")
            or sum(compact_cache_bytes(part) for part in parts),
            "actual_disk_bytes": cache_disk_bytes(cache_dir, cache["shards"]),
            "expected_bytes": expected_train_cache_bytes,
            "shard_count": len(cache["shards"]),
            "fields": list(COMPACT_FEATURE_FIELDS),
        },
        "probe_config": {
            "seed": int(cfg.seed),
            "epochs": epochs,
            "optimizer": "AdamW",
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "objective": "absolute_gain_loss(huber_delta=1.0)",
            "precision": "fp32",
            "width": 256,
            "teacher_batch_size": train_batch_size,
            "probe_train_batch_groups": len(train_parts),
            "probe_dev_batch_groups": len(dev_parts),
            "timestep": 0,
        },
        "oracle_sanity_check": sanity_check,
        "probes": probes,
        "baselines": BASELINES,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "feature_sufficiency_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=True), encoding="utf-8"
    )
    (output_dir / "feature_sufficiency_report.md").write_text(render_markdown(report), encoding="utf-8")


if __name__ == "__main__":
    main()
