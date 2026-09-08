"""Collect fixed rollout inputs and refit only ScoreNet with absolute gain.

Examples
--------
Collect a deterministic TRAIN cache::

    python src/refit_score_net.py backbone=fgclip_base_text_native_cls \
      +scorer_refit=gain_only scorer_refit.mode=collect \
      scorer_refit.source_checkpoint=outputs/r0_ncls_text_strong_aux/best.pt \
      scorer_refit.cache_dir=outputs/r0_ncls_text_strong_aux_score_gain_refit/cache \
      hydra.run.dir=outputs/r0_ncls_text_strong_aux_score_gain_refit/collect

Refit all ScoreNet parameters and save a normal model checkpoint::

    python src/refit_score_net.py backbone=fgclip_base_text_native_cls \
      +scorer_refit=gain_only scorer_refit.mode=refit \
      scorer_refit.source_checkpoint=outputs/r0_ncls_text_strong_aux/best.pt \
      scorer_refit.cache_dir=outputs/r0_ncls_text_strong_aux_score_gain_refit/cache \
      scorer_refit.output_checkpoint=outputs/r0_ncls_text_strong_aux_score_gain_refit/score_gain_refit.pt \
      hydra.run.dir=outputs/r0_ncls_text_strong_aux_score_gain_refit/refit
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from data.images import FashionIQImageCollator
from datasets.common import CIRSample, DirectoryImageStore
from datasets.fashioniq import FashionIQDataset
from diagnostics.selection import score_utility_calibration
from evaluate import validate_checkpoint_backbone_metadata
from models.iag_srme.utils.retrieval import build_teacher_masks, marginal_teacher_utilities
from runtime import configure_torch_runtime, resolve_device, seed_everything
from train import CATEGORIES, build_model, git_identity
from training.scorer_refit import (
    SCORER_TENSOR_FIELDS,
    build_score_refit_optimizer,
    non_score_parameter_fingerprint,
    parameter_fingerprint,
    scorer_gain_loss,
    scorer_minibatches,
    validate_scorer_batch,
)
from training.engine import resolve_precision


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_fingerprint(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sample_record(sample: CIRSample, dataset_index: int) -> dict[str, object]:
    return {
        "dataset_index": dataset_index,
        "sample_id": sample.sample_id,
        "category": sample.category,
        "reference_id": sample.reference_id,
        "target_id": sample.target_id,
        "modification_text": sample.modification_text,
    }


def _fixed_train_subset(
    dataset: FashionIQDataset, *, sample_count: int, seed: int
) -> tuple[Subset[CIRSample], list[dict[str, object]]]:
    count = len(dataset) if sample_count <= 0 else min(sample_count, len(dataset))
    order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))[:count]
    indices = order.tolist()
    records = [_sample_record(dataset[index], index) for index in indices]
    return Subset(dataset, indices), records


def _source_objective_config(checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = checkpoint.get("metadata")
    objective = metadata.get("objective_config") if isinstance(metadata, Mapping) else None
    if not isinstance(objective, Mapping):
        raise ValueError("source checkpoint lacks objective_config metadata")
    return objective


def _validate_refit_cache_contract(
    cfg: DictConfig,
    manifest: Mapping[str, Any],
    *,
    source_checkpoint_sha256: str,
    source_objective: Mapping[str, Any],
) -> None:
    """Lock every cached-rollout behavior used by the scorer refit."""

    retrieval_temperature = float(source_objective["retrieval_temperature"])
    expected = {
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "num_candidates": int(cfg.model.num_candidates),
        "max_steps": int(cfg.model.max_steps),
        "stop_enabled": bool(cfg.model.stop_enabled),
        "epsilon_stop": float(cfg.model.epsilon_stop),
        "score_dropout": float(cfg.model.score_dropout),
        "retrieval_temperature": retrieval_temperature,
    }
    for field, configured in expected.items():
        stored = manifest.get(field)
        if stored != configured:
            raise ValueError(
                f"scorer cache contract mismatch for {field}: "
                f"stored={stored!r}, configured={configured!r}"
            )
    configured_temperature = float(cfg.objective.retrieval_temperature)
    if configured_temperature != retrieval_temperature:
        raise ValueError(
            "configured retrieval temperature differs from the source checkpoint: "
            f"source={retrieval_temperature!r}, configured={configured_temperature!r}"
        )


def _validate_source_checkpoint(cfg: DictConfig, checkpoint: Mapping[str, Any]) -> None:
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
            "score_dropout": float(cfg.model.score_dropout),
        },
    )


def _detach_rows(value: Tensor, valid_rows: Tensor) -> Tensor:
    return value.index_select(0, valid_rows.nonzero(as_tuple=False).squeeze(-1)).detach().cpu()


def _merge_step_chunks(chunks: Mapping[str, list[Any]]) -> dict[str, Any]:
    return {
        **{name: torch.cat(chunks[name], dim=0) for name in SCORER_TENSOR_FIELDS},
        "sample_ids": [sample_id for values in chunks["sample_ids"] for sample_id in values],
    }


def _write_resolved_config(cfg: DictConfig, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.yaml").write_text(
        OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8"
    )


def collect_scorer_cache(
    cfg: DictConfig,
    checkpoint: Mapping[str, Any],
    source_checkpoint: Path,
    *,
    source_checkpoint_sha256: str,
    device: torch.device,
) -> dict[str, Any]:
    refit = cfg.scorer_refit
    cache_dir = Path(str(refit.cache_dir))
    if cache_dir.exists() and any(cache_dir.iterdir()):
        raise FileExistsError(f"scorer cache directory is not empty: {cache_dir}")
    cache_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, processor = build_model(cfg)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    dataset_root = Path(cfg.dataset.root)
    dataset = FashionIQDataset(
        dataset_root / str(cfg.dataset.annotation_dir),
        str(refit.collection_split),
        CATEGORIES,
        caption_policy=str(cfg.experiment.train_caption_policy),
        seed=int(cfg.seed),
    )
    cohort, sample_records = _fixed_train_subset(
        dataset,
        sample_count=int(refit.sample_count),
        seed=int(cfg.seed),
    )
    batch_size = int(refit.collection_batch_size)
    source_metadata = checkpoint.get("metadata")
    source_metadata = source_metadata if isinstance(source_metadata, Mapping) else {}
    source_run = source_metadata.get("run")
    source_run = source_run if isinstance(source_run, Mapping) else {}
    source_batch_size = source_run.get("batch_size")
    if source_batch_size is not None and int(source_batch_size) != batch_size:
        raise ValueError(
            "collection_batch_size must match the source training batch size because "
            "it changes the in-batch teacher pool"
        )
    loader = DataLoader(
        cohort,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(cfg.experiment.num_workers),
        pin_memory=True,
        drop_last=False,
        collate_fn=FashionIQImageCollator(
            DirectoryImageStore(dataset_root / str(cfg.dataset.image_dir)),
            tokenizer,
            processor,
            int(cfg.backbone.max_text_length),
            include_targets=True,
        ),
    )
    objective_config = _source_objective_config(checkpoint)
    retrieval_temperature = float(objective_config["retrieval_temperature"])
    precision_name = str(cfg.runtime.precision)
    precision = resolve_precision(precision_name, device)
    shards = []
    teacher_batches = []
    invalid_rows = 0
    valid_decisions = 0
    with torch.no_grad():
        for batch_index, cpu_batch in enumerate(loader):
            teacher_batches.append(list(cpu_batch.sample_ids))
            batch = cpu_batch.to(device)
            if batch.target_pixels is None:
                raise ValueError("scorer cache collection requires target images")
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
                targets = model.encode_global_images(batch.target_pixels)
            positive, negative, _ = build_teacher_masks(batch.target_ids, device)
            chunks: defaultdict[str, list[Any]] = defaultdict(list)
            for step in output["steps"]:
                live_indices = step["live_indices"]
                teacher, valid_rows = marginal_teacher_utilities(
                    step["current_query"],
                    step["candidate_queries"],
                    targets.detach(),
                    positive.index_select(0, live_indices),
                    negative.index_select(0, live_indices),
                    retrieval_temperature,
                )
                invalid_rows += int((~valid_rows).sum().cpu())
                if not valid_rows.any():
                    continue
                live_ids = [
                    batch.sample_ids[index] for index in live_indices.detach().cpu().tolist()
                ]
                valid_cpu = valid_rows.detach().cpu()
                chunks["sample_ids"].append(
                    [sample_id for sample_id, valid in zip(live_ids, valid_cpu.tolist()) if valid]
                )
                chunks["current_global"].append(_detach_rows(step["current_global"], valid_rows))
                chunks["text_global"].append(_detach_rows(step["text_global"], valid_rows))
                chunks["actions"].append(_detach_rows(step["actions"], valid_rows))
                chunks["delta"].append(_detach_rows(step["delta"], valid_rows))
                chunks["exec_mask"].append(_detach_rows(step["exec_mask"], valid_rows))
                chunks["candidate_global"].append(
                    _detach_rows(step["candidate_global"], valid_rows)
                )
                chunks["teacher_utility"].append(_detach_rows(teacher, valid_rows).float())
                chunks["timestep"].append(
                    torch.full(
                        (int(valid_rows.sum()),),
                        int(step["timestep"]),
                        dtype=torch.long,
                    )
                )
            if not chunks:
                continue
            shard = _merge_step_chunks(chunks)
            validate_scorer_batch(shard, int(cfg.model.num_candidates))
            shard_path = cache_dir / f"shard_{batch_index:06d}.pt"
            torch.save(shard, shard_path)
            checksum = _sha256_file(shard_path)
            rows = len(shard["sample_ids"])
            valid_decisions += rows
            shards.append({"path": shard_path.name, "rows": rows, "sha256": checksum})
            print(
                f"[score-refit/collect] batch={batch_index + 1}/{len(loader)} "
                f"valid_decisions={valid_decisions}"
            )

    if not shards:
        raise RuntimeError("scorer cache collection produced no valid teacher decisions")
    manifest: dict[str, Any] = {
        "format_version": 2,
        "workflow": "score_net_gain_only_refit",
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "source_git_sha": (
            source_run.get("git_sha")
            or source_metadata.get("git_sha")
            or checkpoint.get("git_commit")
        ),
        "collector_git": git_identity(),
        "collection_split": str(refit.collection_split),
        "caption_policy": str(cfg.experiment.train_caption_policy),
        "seed": int(cfg.seed),
        "sample_count": len(sample_records),
        "collection_batch_size": batch_size,
        "source_training_batch_size": source_batch_size,
        "teacher_batches": teacher_batches,
        "teacher_batch_fingerprint": _json_fingerprint(teacher_batches),
        "dataset_records_fingerprint": _json_fingerprint(sample_records),
        "dataset_records": sample_records,
        "retrieval_temperature": retrieval_temperature,
        "num_candidates": int(cfg.model.num_candidates),
        "max_steps": int(cfg.model.max_steps),
        "stop_enabled": bool(cfg.model.stop_enabled),
        "epsilon_stop": float(cfg.model.epsilon_stop),
        "score_dropout": float(cfg.model.score_dropout),
        "raw_input_fields": list(SCORER_TENSOR_FIELDS),
        "valid_decision_count": valid_decisions,
        "teacher_invalid_row_count": invalid_rows,
        "shards": shards,
    }
    manifest["cache_fingerprint"] = _json_fingerprint(
        {key: manifest[key] for key in manifest if key != "dataset_records"}
    )
    (cache_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def _load_cache_manifest(cache_dir: Path) -> dict[str, Any]:
    path = cache_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"scorer cache manifest does not exist: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("workflow") != "score_net_gain_only_refit":
        raise ValueError("cache manifest is not a gain-only ScoreNet refit cache")
    if _json_fingerprint(manifest.get("dataset_records")) != manifest.get(
        "dataset_records_fingerprint"
    ):
        raise ValueError("scorer cache dataset-record fingerprint mismatch")
    if _json_fingerprint(manifest.get("teacher_batches")) != manifest.get(
        "teacher_batch_fingerprint"
    ):
        raise ValueError("scorer cache teacher-batch fingerprint mismatch")
    fingerprint_payload = {
        key: value
        for key, value in manifest.items()
        if key not in {"cache_fingerprint", "dataset_records"}
    }
    if _json_fingerprint(fingerprint_payload) != manifest.get("cache_fingerprint"):
        raise ValueError("scorer cache manifest fingerprint mismatch")
    return manifest


def _load_shard(
    cache_dir: Path,
    record: Mapping[str, Any],
    candidates: int,
    *,
    verify_checksum: bool = True,
) -> dict[str, Any]:
    path = cache_dir / str(record["path"])
    if verify_checksum and _sha256_file(path) != record["sha256"]:
        raise ValueError(f"scorer cache shard checksum mismatch: {path}")
    batch = torch.load(path, map_location="cpu", weights_only=True)
    validate_scorer_batch(batch, candidates)
    if len(batch["sample_ids"]) != int(record["rows"]):
        raise ValueError(f"scorer cache shard row count mismatch: {path}")
    return batch


def _to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        **{
            name: (
                batch[name].to(device=device, dtype=torch.float32)
                if batch[name].is_floating_point()
                else batch[name].to(device=device)
            )
            for name in SCORER_TENSOR_FIELDS
        },
        "sample_ids": list(batch["sample_ids"]),
    }


def _build_refit_checkpoint(
    checkpoint: Mapping[str, Any],
    model: Any,
    optimizer: torch.optim.Optimizer,
    *,
    metadata: Mapping[str, Any],
    epochs: int,
    optimizer_steps: int,
) -> dict[str, Any]:
    """Build an inference-ready checkpoint without claiming full-trainer resumability."""

    result = dict(checkpoint)
    source_state = dict(checkpoint["model"])
    for name, value in model.score_net.state_dict().items():
        source_state[f"score_net.{name}"] = value.detach().cpu()
    result["model"] = source_state
    result["optimizer"] = None
    result["scaler"] = None
    result["score_refit_optimizer"] = optimizer.state_dict()
    result["optimizer_step"] = optimizer_steps
    result["batch_step"] = optimizer_steps
    result["epoch"] = epochs
    result["metric"] = None
    result["metadata"] = dict(metadata)
    return result


@torch.no_grad()
def _evaluate_cached_scorer(
    model: Any,
    cache_dir: Path,
    manifest: Mapping[str, Any],
    device: torch.device,
    batch_size: int,
    *,
    verify_checksums: bool,
) -> dict[str, Any]:
    model.score_net.eval()
    predicted = []
    teacher = []
    generator = torch.Generator().manual_seed(0)
    for record in manifest["shards"]:
        shard = _load_shard(
            cache_dir,
            record,
            int(manifest["num_candidates"]),
            verify_checksum=verify_checksums,
        )
        for mini in scorer_minibatches(shard, batch_size=batch_size, generator=generator):
            device_batch = _to_device(mini, device)
            _, scores = scorer_gain_loss(
                model.score_net,
                device_batch,
                huber_delta=float(manifest["huber_delta"]),
            )
            predicted.append(scores.cpu())
            teacher.append(device_batch["teacher_utility"].cpu())
    return score_utility_calibration(torch.cat(predicted), torch.cat(teacher))


def refit_score_net(
    cfg: DictConfig,
    checkpoint: Mapping[str, Any],
    source_checkpoint: Path,
    *,
    source_checkpoint_sha256: str,
    device: torch.device,
) -> dict[str, Any]:
    refit = cfg.scorer_refit
    cache_dir = Path(str(refit.cache_dir))
    manifest = _load_cache_manifest(cache_dir)
    source_objective = _source_objective_config(checkpoint)
    _validate_refit_cache_contract(
        cfg,
        manifest,
        source_checkpoint_sha256=source_checkpoint_sha256,
        source_objective=source_objective,
    )
    huber_delta = float(source_objective["huber_delta"])
    manifest["huber_delta"] = huber_delta

    model, _, _ = build_model(cfg)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    before_non_score = non_score_parameter_fingerprint(model)
    before_score = parameter_fingerprint(model, score_net=True)
    optimizer = build_score_refit_optimizer(
        model,
        learning_rate=float(refit.learning_rate),
        weight_decay=float(refit.weight_decay),
    )
    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if not trainable_names or any(not name.startswith("score_net.") for name in trainable_names):
        raise RuntimeError("scorer-only freeze policy was not established")

    output_dir = Path(HydraConfig.get().runtime.output_dir)
    metrics_path = output_dir / "metrics.jsonl"
    initial_calibration = _evaluate_cached_scorer(
        model,
        cache_dir,
        manifest,
        device,
        int(refit.scorer_batch_size),
        verify_checksums=True,
    )
    optimizer_steps = 0
    for epoch in range(int(refit.epochs)):
        model.eval()
        model.score_net.train()
        generator = torch.Generator().manual_seed(int(cfg.seed) + epoch)
        shard_order = torch.randperm(len(manifest["shards"]), generator=generator).tolist()
        loss_sum = 0.0
        score_count = 0
        for shard_index in shard_order:
            record = manifest["shards"][shard_index]
            shard = _load_shard(
                cache_dir,
                record,
                int(manifest["num_candidates"]),
                verify_checksum=False,
            )
            for mini in scorer_minibatches(
                shard,
                batch_size=int(refit.scorer_batch_size),
                generator=generator,
            ):
                device_batch = _to_device(mini, device)
                optimizer.zero_grad(set_to_none=True)
                loss, predicted = scorer_gain_loss(
                    model.score_net,
                    device_batch,
                    huber_delta=huber_delta,
                )
                loss.backward()
                if any(
                    parameter.grad is not None
                    and not torch.isfinite(parameter.grad).all()
                    for parameter in model.score_net.parameters()
                ):
                    raise FloatingPointError("non-finite ScoreNet gradient during refit")
                optimizer.step()
                optimizer_steps += 1
                elements = predicted.numel()
                loss_sum += float(loss.detach()) * elements
                score_count += elements
        record = {
            "record_type": "score_refit_epoch",
            "epoch": epoch + 1,
            "optimizer_steps": optimizer_steps,
            "gain_huber": loss_sum / max(score_count, 1),
            "learning_rate": float(refit.learning_rate),
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(
            f"[score-refit/train] epoch={epoch + 1}/{int(refit.epochs)} "
            f"gain={record['gain_huber']:.6f} steps={optimizer_steps}"
        )

    final_calibration = _evaluate_cached_scorer(
        model,
        cache_dir,
        manifest,
        device,
        int(refit.scorer_batch_size),
        verify_checksums=False,
    )
    after_non_score = non_score_parameter_fingerprint(model)
    after_score = parameter_fingerprint(model, score_net=True)
    if after_non_score != before_non_score:
        raise RuntimeError("a non-ScoreNet parameter changed during scorer-only refit")
    if after_score == before_score:
        raise RuntimeError("ScoreNet parameters did not change during gain-only refit")

    output_checkpoint = Path(str(refit.output_checkpoint))
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    metadata = copy.deepcopy(checkpoint.get("metadata"))
    metadata = metadata if isinstance(metadata, dict) else {}
    source_identity = metadata.get("experiment_identity", "unknown")
    metadata["experiment_identity"] = f"{source_identity}-SCORE-GAIN-REFIT"
    metadata["source_validation_metric"] = checkpoint.get("metric")
    metadata["number_of_optimizer_updates"] = optimizer_steps
    metadata["optimizer_scope"] = "score_net_only"
    metadata["exact_full_training_resume"] = False
    metadata["resume_semantics"] = "warm_start_only_for_full_training"
    metadata["score_refit"] = {
        "workflow": "score_net_gain_only_refit",
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "source_epoch": checkpoint.get("epoch"),
        "source_optimizer_step": checkpoint.get("optimizer_step"),
        "source_batch_step": checkpoint.get("batch_step"),
        "cache_fingerprint": manifest["cache_fingerprint"],
        "collection_split": manifest["collection_split"],
        "sample_count": manifest["sample_count"],
        "collection_batch_size": manifest["collection_batch_size"],
        "retrieval_temperature": manifest["retrieval_temperature"],
        "learning_rate": float(refit.learning_rate),
        "weight_decay": float(refit.weight_decay),
        "optimizer": "AdamW",
        "optimizer_state_field": "score_refit_optimizer",
        "exact_full_training_resume": False,
        "resume_semantics": "warm_start_only_for_full_training",
        "refit_precision": "fp32",
        "epochs": int(refit.epochs),
        "optimizer_steps": optimizer_steps,
        "seed": int(cfg.seed),
        "epsilon_stop": float(cfg.model.epsilon_stop),
        "score_dropout": float(cfg.model.score_dropout),
        "huber_delta": huber_delta,
        "lambda_pair": float(refit.lambda_pair),
        "loss": str(refit.loss),
        "trainable_parameter_names": trainable_names,
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.score_net.parameters()
        ),
        "non_score_parameter_fingerprint_before": before_non_score,
        "non_score_parameter_fingerprint_after": after_non_score,
        "non_score_parameters_unchanged": True,
        "score_net_parameter_fingerprint_before": before_score,
        "score_net_parameter_fingerprint_after": after_score,
        "score_net_parameters_changed": True,
        "initial_cached_calibration": initial_calibration,
        "final_cached_calibration": final_calibration,
        "resolved_config": OmegaConf.to_container(cfg, resolve=True),
    }
    result = _build_refit_checkpoint(
        checkpoint,
        model,
        optimizer,
        metadata=metadata,
        epochs=int(refit.epochs),
        optimizer_steps=optimizer_steps,
    )
    torch.save(result, output_checkpoint)
    report = {
        "output_checkpoint": str(output_checkpoint),
        **metadata["score_refit"],
        "live_rollout_required": True,
        "note": (
            "Cached calibration is offline only. Run candidate diagnostics and official "
            "FashionIQ evaluation from this checkpoint for scientific comparison."
        ),
    }
    (output_dir / "refit_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.get("scorer_refit") is None:
        raise ValueError("pass +scorer_refit=gain_only")
    refit = cfg.scorer_refit
    mode = str(refit.mode)
    if mode not in {"collect", "refit"}:
        raise ValueError("scorer_refit.mode must be collect or refit")
    if float(refit.lambda_pair) != 0.0 or str(refit.loss) != "absolute_gain_huber_only":
        raise ValueError("ScoreNet rescue is fixed to lambda_pair=0 and absolute gain only")
    if str(refit.collection_split) != "train":
        raise ValueError("gain-only ScoreNet refit collection must use the TRAIN split")
    if not (
        str(cfg.backbone.finetune_policy) == "text_only"
        and not bool(cfg.backbone.train_vision)
        and bool(cfg.backbone.train_text)
        and not bool(cfg.backbone.train_text_projection)
    ):
        raise ValueError(
            "this controlled rescue requires the existing text-only backbone policy: "
            "train_vision=false, train_text=true, train_text_projection=false"
        )

    seed_everything(int(cfg.seed), bool(cfg.runtime.deterministic))
    configure_torch_runtime(
        deterministic=bool(cfg.runtime.deterministic),
        benchmark=bool(cfg.runtime.benchmark),
    )
    device = resolve_device(str(cfg.runtime.device), int(cfg.runtime.accelerator_index))
    source_checkpoint = Path(str(refit.source_checkpoint))
    if not source_checkpoint.is_file():
        raise FileNotFoundError(f"source checkpoint does not exist: {source_checkpoint}")
    checkpoint = torch.load(source_checkpoint, map_location="cpu", weights_only=True)
    _validate_source_checkpoint(cfg, checkpoint)
    source_sha256 = _sha256_file(source_checkpoint)
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    _write_resolved_config(cfg, output_dir)

    if mode == "collect":
        report = collect_scorer_cache(
            cfg,
            checkpoint,
            source_checkpoint,
            source_checkpoint_sha256=source_sha256,
            device=device,
        )
        (output_dir / "collection_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(f"[score-refit] cache: {refit.cache_dir}")
    else:
        report = refit_score_net(
            cfg,
            checkpoint,
            source_checkpoint,
            source_checkpoint_sha256=source_sha256,
            device=device,
        )
        print(f"[score-refit] checkpoint: {report['output_checkpoint']}")


if __name__ == "__main__":
    main()
