"""Memorization sanity check for legacy_independent on cached frozen t0 rows only."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from diagnose_score_feature_sufficiency import cache_disk_bytes, merge_rows, rows_from_shard
from diagnostics.feature_sufficiency import LABEL_FIELD, probe_from_rows, probe_loss, probe_metrics, to_training_precision
from runtime import configure_torch_runtime, resolve_device, seed_everything

PROBE_NAME = "legacy_independent"
TEACHER_BATCH_SIZE = 32
COHORT_BATCH_GROUPS = 8
EPOCHS = 100
CHECKPOINT_EPOCHS = frozenset((1, 2, 5, 10, 20, 50, 100))


def load_small_cohort(cache_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load exactly the first eight intact 32-row TRAIN cache shards, in stored order."""

    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"compact feature cache manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = manifest.get("metadata")
    shards = manifest.get("shards")
    if not isinstance(metadata, dict) or metadata.get("split") != "train":
        raise ValueError("small-cohort overfit requires a TRAIN compact feature cache")
    if metadata.get("teacher_batch_size") != TEACHER_BATCH_SIZE:
        raise ValueError("small-cohort overfit requires 32-row fixed teacher-batch groups")
    if not isinstance(shards, list) or len(shards) < COHORT_BATCH_GROUPS:
        raise ValueError("small-cohort overfit requires at least eight compact cache shards")
    parts = [rows_from_shard(cache_dir, record) for record in shards[:COHORT_BATCH_GROUPS]]
    if any(len(part["sample_ids"]) != TEACHER_BATCH_SIZE for part in parts):
        raise ValueError("the first eight compact cache shards must each contain exactly 32 rows")
    return parts, manifest


def metric_row(probe: torch.nn.Module, rows: Mapping[str, Any], device: torch.device) -> dict[str, float | int]:
    """Evaluate the same frozen cohort without changing probe weights."""

    probe.eval()
    with torch.no_grad():
        batch = to_training_precision(rows, device)
        scores = probe(batch)
        summary = probe_metrics(scores, batch[LABEL_FIELD], epsilon_stop=0.0)
        calibration = summary["score_utility_calibration"]
        assert isinstance(calibration, dict)
        return {
            "loss": float(probe_loss(scores, batch[LABEL_FIELD])),
            "pearson": float(calibration["pearson"]),
            "spearman": float(calibration["spearman"]),
            "sign_agreement_at_zero": float(calibration["sign_agreement_at_zero"]),
            "exact_oracle_accuracy": float(summary["exact_oracle_accuracy"]),
            "selected_utility": float(summary["selected_teacher_utility"]),
            "oracle_utility": float(summary["oracle_teacher_utility"]),
            "regret": float(summary["oracle_regret"]),
            "stop_rate": float(summary["stop_rate"]),
            "oracle_stop_rate": float(summary["oracle_stop_rate"]),
            "decision_count": int(summary["decision_count"]),
        }


def train_small_cohort_overfit(
    parts: list[Mapping[str, Any]], *, device: torch.device, epochs: int = EPOCHS
) -> tuple[torch.nn.Module, list[dict[str, float | int]]]:
    """Train the unchanged probe over the fixed groups in their cached order, no held-out data."""

    if len(parts) != COHORT_BATCH_GROUPS:
        raise ValueError(f"small-cohort overfit requires exactly {COHORT_BATCH_GROUPS} cache shards")
    rows = merge_rows(parts)
    if len(rows["sample_ids"]) != COHORT_BATCH_GROUPS * TEACHER_BATCH_SIZE:
        raise ValueError("small-cohort overfit requires exactly 256 cached TRAIN examples")
    probe = probe_from_rows(PROBE_NAME, rows).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-4, weight_decay=0.01)
    history = []
    for epoch in range(1, epochs + 1):
        probe.train()
        for part in parts:
            batch = to_training_precision(part, device)
            optimizer.zero_grad(set_to_none=True)
            loss = probe_loss(probe(batch), batch[LABEL_FIELD])
            loss.backward()
            optimizer.step()
        row = {"epoch": epoch, **metric_row(probe, rows, device)}
        history.append(row)
        if epoch in CHECKPOINT_EPOCHS:
            print(
                "[overfit] "
                + " ".join(
                    f"{name}={row[name]:.5f}" if isinstance(row[name], float) else f"{name}={row[name]}"
                    for name in (
                        "epoch",
                        "loss",
                        "pearson",
                        "spearman",
                        "sign_agreement_at_zero",
                        "exact_oracle_accuracy",
                        "selected_utility",
                        "oracle_utility",
                        "regret",
                        "stop_rate",
                        "oracle_stop_rate",
                    )
                )
            )
    probe.eval()
    return probe, history


def interpretation(final: Mapping[str, float | int]) -> str:
    pearson = float(final["pearson"])
    sign = float(final["sign_agreement_at_zero"])
    if pearson >= 0.8:
        result = (
            "Frozen-cohort capacity/optimization can fit this mapping; poor full-TRAIN/DEV/VAL "
            "performance is primarily a generalization/predictability concern."
        )
    else:
        result = (
            "Pearson remains below the memorization threshold; investigate probe optimization, architecture "
            "capacity, utility scale, or target formulation before feature-sufficiency conclusions."
        )
    if pearson >= 0.8 and sign < 0.8:
        return result + " Signed-zero boundary remains a separate objective problem."
    return result


def render_markdown(report: Mapping[str, Any]) -> str:
    rows = [
        "# legacy_independent 256-row frozen-cohort overfit",
        "",
        f"- compact cache: `{report['compact_cache']['path']}`",
        f"- source checkpoint SHA256: `{report['compact_cache']['source_checkpoint_sha256']}`",
        "- TRAIN only; first 8 stored 32-row teacher groups; t0 only; no VAL or probe-dev.",
        "",
        "| epoch | loss | Pearson | Spearman | sign@0 | exact oracle | selected utility | oracle utility | regret | STOP | oracle STOP |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["checkpoints"]:
        rows.append(
            "| {epoch} | {loss:.5f} | {pearson:.5f} | {spearman:.5f} | {sign_agreement_at_zero:.5f} | "
            "{exact_oracle_accuracy:.5f} | {selected_utility:.5f} | {oracle_utility:.5f} | "
            "{regret:.5f} | {stop_rate:.5f} | {oracle_stop_rate:.5f} |".format(**row)
        )
    return "\n".join(rows + ["", "## Interpretation", "", report["interpretation"]])


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    cache_dir = Path(
        str(
            cfg.get(
                "probe_cache_dir",
                "outputs/diagnostics/2026-09-21/score_feature_sufficiency_t0/compact_cache",
            )
        )
    )
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    seed_everything(42, bool(cfg.runtime.deterministic))
    configure_torch_runtime(
        deterministic=bool(cfg.runtime.deterministic), benchmark=bool(cfg.runtime.benchmark)
    )
    device = resolve_device(str(cfg.runtime.device), int(cfg.runtime.accelerator_index))
    parts, cache_manifest = load_small_cohort(cache_dir)
    _, history = train_small_cohort_overfit(parts, device=device)
    checkpoints = [row for row in history if int(row["epoch"]) in CHECKPOINT_EPOCHS]
    metadata = cache_manifest["metadata"]
    report = {
        "diagnostic": "legacy_independent_small_cohort_overfit",
        "compact_cache": {
            "path": str(cache_dir),
            "disk_bytes": cache_disk_bytes(cache_dir, cache_manifest["shards"]),
            "source_checkpoint_sha256": metadata["source_checkpoint_sha256"],
            "manifest_sample_ids_sha256": metadata["manifest_sample_ids_sha256"],
            "teacher_batch_grouping_sha256": metadata["teacher_batch_grouping_sha256"],
        },
        "protocol": {
            "probe": PROBE_NAME,
            "timestep": 0,
            "split": "train",
            "teacher_batch_groups": COHORT_BATCH_GROUPS,
            "examples": COHORT_BATCH_GROUPS * TEACHER_BATCH_SIZE,
            "epochs": EPOCHS,
            "seed": 42,
            "precision": "fp32",
            "optimizer": "AdamW",
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "objective": "absolute_gain_loss(huber_delta=1.0)",
            "held_out_metrics_used": False,
        },
        "epochs": history,
        "checkpoints": checkpoints,
        "interpretation": interpretation(history[-1]),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "legacy_independent_overfit_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=True), encoding="utf-8"
    )
    (output_dir / "legacy_independent_overfit_report.md").write_text(
        render_markdown(report), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
