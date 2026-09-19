from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.optim import SGD

from data.images import ImageBatch
from losses.objective import IAGSRMEObjective, ObjectiveConfig
from training.engine import (
    resolve_precision,
    save_checkpoint,
    train_one_epoch,
)


class _TinyTrainModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        reference_pixels: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
        content_mask: Tensor,
    ) -> dict[str, Tensor]:
        del input_ids, attention_mask, content_mask
        return {"value": self.weight * reference_pixels.mean()}

    def encode_global_images(self, target_pixels: Tensor) -> Tensor:
        return target_pixels.flatten(1)


class _RecordingObjective(nn.Module):
    def __init__(self, overflow_first: bool = False) -> None:
        super().__init__()
        self.global_steps: list[int] = []
        self.overflow_first = overflow_first

    def forward(
        self,
        output,
        target_embeddings,
        target_ids,
        modification_texts,
        *,
        epoch: int,
        global_step: int,
    ) -> dict[str, Tensor]:
        del target_embeddings, target_ids, modification_texts, epoch
        self.global_steps.append(global_step)
        value = output["value"]
        if self.overflow_first and len(self.global_steps) == 1:
            loss = _FiniteForwardInfiniteBackward.apply(value)
        else:
            loss = value.square()
        return {"total": loss}


class _StageObjective(nn.Module):
    def forward(
        self,
        output,
        target_embeddings,
        target_ids,
        modification_texts,
        *,
        epoch: int,
        global_step: int,
    ) -> dict[str, Tensor]:
        del target_embeddings, target_ids, modification_texts, epoch
        stage = 0 if global_step < 2000 else 1
        value = output["value"]
        return {
            "total": value.square(),
            "dac_stage": value.new_tensor(float(stage)),
            "dac_num_groups": value.new_tensor(float(2**stage)),
            "dac_group_size": value.new_tensor(float(8 // (2**stage))),
            "dac_split_interval_steps": value.new_tensor(2000.0),
        }


class _FiniteForwardInfiniteBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: Tensor) -> Tensor:
        ctx.shape = value.shape
        ctx.device = value.device
        ctx.dtype = value.dtype
        return value.sum()

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor]:
        return (
            torch.full(
                ctx.shape,
                float("inf"),
                device=ctx.device,
                dtype=ctx.dtype,
            )
            * grad_output,
        )


def _batch() -> ImageBatch:
    return ImageBatch(
        sample_ids=["sample"],
        reference_ids=["reference"],
        target_ids=["target"],
        modification_texts=["change"],
        categories=["dress"],
        reference_pixels=torch.ones(1, 1),
        target_pixels=torch.ones(1, 1),
        input_ids=torch.ones(1, 1, dtype=torch.long),
        attention_mask=torch.ones(1, 1, dtype=torch.bool),
        content_mask=torch.ones(1, 1, dtype=torch.bool),
    )


def test_train_one_epoch_passes_pre_update_global_step_and_increments_once() -> None:
    model = _TinyTrainModel()
    objective = _RecordingObjective()
    optimizer = SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cpu", enabled=False)

    metrics, global_step = train_one_epoch(
        model,
        objective,
        [_batch(), _batch()],
        optimizer,
        scaler,
        torch.device("cpu"),
        precision=resolve_precision("fp32", torch.device("cpu")),
        epoch=0,
        global_step=7,
    )

    assert objective.global_steps == [7, 8]
    assert global_step == 9
    assert "total" in metrics


def test_amp_overflow_skipped_update_does_not_advance_global_step() -> None:
    model = _TinyTrainModel()
    objective = _RecordingObjective(overflow_first=True)
    optimizer = SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler(
        "cpu",
        enabled=True,
        init_scale=8.0,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=100,
    )

    _, global_step = train_one_epoch(
        model,
        objective,
        [_batch(), _batch()],
        optimizer,
        scaler,
        torch.device("cpu"),
        precision=resolve_precision("fp32", torch.device("cpu")),
        epoch=0,
        global_step=5,
    )

    assert objective.global_steps == [5, 5]
    assert global_step == 6


def test_step_log_cadence_counts_only_successful_optimizer_updates(
    tmp_path: Path,
) -> None:
    model = _TinyTrainModel()
    objective = _RecordingObjective(overflow_first=True)
    optimizer = SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler(
        "cpu",
        enabled=True,
        init_scale=8.0,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=100,
    )
    log_path = tmp_path / "train_steps.jsonl"

    _, global_step = train_one_epoch(
        model,
        objective,
        [_batch(), _batch()],
        optimizer,
        scaler,
        torch.device("cpu"),
        precision=resolve_precision("fp32", torch.device("cpu")),
        epoch=4,
        global_step=99,
        step_log_path=log_path,
        step_log_interval=100,
    )

    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert objective.global_steps == [99, 99]
    assert global_step == 100
    assert len(records) == 1
    assert records[0]["global_step"] == 100
    assert records[0]["objective_global_step"] == 99
    assert records[0]["epoch"] == 4


def test_epoch_summary_and_step_log_keep_discrete_dac_state(tmp_path: Path) -> None:
    model = _TinyTrainModel()
    objective = _StageObjective()
    optimizer = SGD(model.parameters(), lr=0.1)

    log_path = tmp_path / "train_steps.jsonl"
    metrics, global_step = train_one_epoch(
        model,
        objective,
        [_batch(), _batch()],
        optimizer,
        torch.amp.GradScaler("cpu", enabled=False),
        torch.device("cpu"),
        precision=resolve_precision("fp32", torch.device("cpu")),
        epoch=0,
        global_step=1999,
        step_log_path=log_path,
        step_log_interval=1,
    )

    assert global_step == 2001
    assert metrics["dac_stage"] == 1.0
    assert metrics["dac_num_groups"] == 2.0
    assert metrics["dac_group_size"] == 4.0
    assert metrics["dac_split_interval_steps"] == 2000.0
    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert [
        (
            record["global_step"],
            record["objective_global_step"],
            record["dac_stage"],
        )
        for record in records
    ] == [
        (2000, 1999, 0),
        (2001, 2000, 1),
    ]


def test_checkpoint_records_global_step_and_dac_configuration(
    tmp_path: Path, model
) -> None:
    metadata = {
        "checkpoint": "tiny",
        "revision": "test",
        "readout_experiment": "test",
        "finetune_policy": "test",
        "experiment_identity": "test",
        "train_vision": False,
        "train_text": False,
        "train_text_projection": False,
    }
    for name, value in metadata.items():
        setattr(model.backbone, name, value)
    objective = IAGSRMEObjective(
        ObjectiveConfig(candidate_credit_mode="dac", dac_split_interval_steps=2000)
    )
    optimizer = SGD(model.parameters(), lr=0.1)
    path = tmp_path / "checkpoint.pt"

    save_checkpoint(
        path,
        model,
        objective,
        optimizer,
        epoch=2,
        global_step=6001,
        metric=1.0,
        precision=resolve_precision("fp32", torch.device("cpu")),
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)

    assert checkpoint["global_step"] == 6001
    assert checkpoint["metadata"]["global_step"] == 6001
    assert checkpoint["metadata"]["objective_config"]["candidate_credit_mode"] == "dac"
    assert checkpoint["metadata"]["objective_config"]["dac_split_interval_steps"] == 2000
