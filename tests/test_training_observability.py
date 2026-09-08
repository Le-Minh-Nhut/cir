from __future__ import annotations

import json

import torch
from torch import nn
from torch.optim import SGD

from data.images import ImageBatch
from training.engine import (
    JSONLLogger,
    PrecisionPolicy,
    train_one_epoch,
    trainable_parameters,
)


class _BackboneInner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.text_model = nn.Linear(2, 2)


class _Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _BackboneInner()


class _ObservedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _Backbone()
        self.proposal = nn.Linear(2, 2)
        self.grounder = nn.Linear(2, 2)
        self.action_fusion = nn.Linear(2, 2)
        self.executor = nn.Linear(2, 2)
        self.score_net = nn.Linear(2, 1)

    def forward(self, reference, input_ids, attention_mask, content_mask):
        del attention_mask, content_mask
        value = reference.float() + input_ids.float()
        value = self.backbone.model.text_model(value)
        for module in (
            self.proposal,
            self.grounder,
            self.action_fusion,
            self.executor,
        ):
            value = torch.tanh(module(value))
        return {"value": self.score_net(value).squeeze(-1)}

    def encode_global_images(self, pixels):
        return pixels.float()


class _ObservedObjective(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, output, targets, target_ids, modification_texts):
        del target_ids, modification_texts
        terminal = (output["value"] - targets.mean(dim=-1)).square().mean() * self.scale
        zero = terminal.detach() * 0
        return {
            "total": terminal,
            "terminal": terminal,
            "pair": zero,
            "gain": zero,
            "concept_loss": zero,
            "bind_loss": zero,
            "rel_ortho_loss": zero,
            "dpp_raw": zero,
            "dpp_weighted": zero,
            "stop_rate": zero,
            "mean_rollout_length": zero,
            "useful_candidate_count": zero,
            "dpp_valid_rate": zero,
            "mean_delta_q_norm": zero,
            "safe_harmful_candidate_fraction_c0": zero,
            "safe_positive_candidate_fraction_c0": zero,
            "mean_delta_q_norm_c0": zero,
            "teacher_invalid_rows": zero,
        }


def test_training_jsonl_records_real_optimizer_updates(tmp_path) -> None:
    model = _ObservedModel()
    objective = _ObservedObjective()
    optimizer = SGD(trainable_parameters(model, objective), lr=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    batch = ImageBatch(
        sample_ids=["a", "b"],
        reference_ids=["ra", "rb"],
        target_ids=["ta", "tb"],
        modification_texts=["red", "blue"],
        categories=["dress", "dress"],
        reference_pixels=torch.tensor([[1.0, 2.0], [2.0, 3.0]]),
        target_pixels=torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
        input_ids=torch.tensor([[1, 0], [0, 1]]),
        attention_mask=torch.ones(2, 2, dtype=torch.bool),
        content_mask=torch.ones(2, 2, dtype=torch.bool),
    )
    logger = JSONLLogger(tmp_path / "metrics.jsonl")

    result = train_one_epoch(
        model,
        objective,
        [batch],
        optimizer,
        scaler,
        torch.device("cpu"),
        precision=PrecisionPolicy("fp32", False, None, False),
        epoch=0,
        logger=logger,
    )

    record = json.loads((tmp_path / "metrics.jsonl").read_text().strip())
    assert result["optimizer_step_count"] == 1
    assert record["optimizer_step"] == 1
    assert record["amp_skipped_update_inferred"] is False
    assert record["nonfinite_gradient_count"] == 0
    assert record["total"] > 0
    assert record["safe_harmful_candidate_fraction_c0"] == 0
    assert record["safe_positive_candidate_fraction_c0"] == 0
    assert record["mean_delta_q_norm_c0"] == 0
    for group in (
        "text_encoder",
        "proposal",
        "grounder",
        "action_fusion",
        "executor",
        "score_net",
    ):
        assert record["parameter_updates"][group]["gradient_norm"] > 0
        assert record["parameter_updates"][group]["parameter_delta_norm"] > 0
