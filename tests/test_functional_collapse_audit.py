from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.optim import SGD

from analyze_functional_collapse import STAGES, _heuristic, _records, _summarize
from data.images import ImageBatch
from diagnostics.functional_collapse import (
    _effective_rank,
    alpha_read_diversity,
    exec_mask_diversity,
    functional_collapse_audit,
    generic_diversity,
    selection_quality_audit,
)
from training.engine import JSONLLogger, PrecisionPolicy, train_one_epoch, trainable_parameters


def test_generic_metrics_distinguish_identical_and_orthogonal_candidates() -> None:
    identical = torch.ones(2, 3, 4)
    distinct = torch.eye(3).expand(2, -1, -1)

    collapsed = generic_diversity(identical)
    diverse = generic_diversity(distinct)

    assert collapsed["pairwise_cosine"] == 1.0
    assert collapsed["spread"] == 0.0
    assert collapsed["relative_spread"] == 0.0
    assert collapsed["effective_rank"] == pytest.approx(1.0, abs=3e-4)
    assert diverse["pairwise_cosine"] < collapsed["pairwise_cosine"]
    assert diverse["effective_rank"] > collapsed["effective_rank"]
    assert diverse["spread"] > collapsed["spread"]


def test_low_energy_effects_are_not_reported_as_meaningful_directions() -> None:
    metrics = generic_diversity(torch.full((2, 3, 4), 1e-9))

    assert metrics["low_energy_fraction"] == 1.0
    assert metrics["valid_sibling_effect_fraction"] == 0.0
    assert metrics["pairwise_cosine_valid_pair_fraction"] == 0.0
    assert metrics["effective_rank"] == 0.0


def test_effective_rank_is_scale_stable_and_centered_rank_is_bounded() -> None:
    values = torch.eye(4).unsqueeze(0)
    tiny = generic_diversity(values * 1e-4)
    normal = generic_diversity(values)

    assert tiny["effective_rank"] == pytest.approx(normal["effective_rank"])
    assert normal["centered_effective_rank"] <= 3.0 + 1e-6
    assert normal["valid_sibling_effect_fraction"] == 1.0


def test_zero_effects_are_explicitly_low_energy() -> None:
    metrics = generic_diversity(torch.zeros(2, 4, 3))

    assert metrics["low_energy_fraction"] == 1.0
    assert metrics["valid_candidate_fraction"] == 0.0
    assert metrics["centered_effective_rank"] == 0.0


def test_selection_audit_uses_keep_zero_for_stop_regret() -> None:
    output = {
        "steps": [
            {
                "timestep": 0,
                "scores": torch.tensor([[0.1, 0.0]]),
                "selected_idx": torch.tensor([2]),
                "live_indices": torch.tensor([0]),
                "current_query": torch.tensor([[1.0, 0.0]]),
                "candidate_queries": torch.tensor([[[0.0, 1.0], [1.0, 0.0]]]),
            }
        ]
    }
    audit = selection_quality_audit(
        output,
        torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
        ["target", "negative"],
        0.07,
    )

    assert audit["overall"]["stop_fraction"] == 1.0
    assert audit["overall"]["oracle_utility"] >= audit["overall"]["selected_utility"]
    assert audit["overall"]["one_step_regret"] >= 0.0


def test_selection_audit_subtracts_nonzero_stop_threshold() -> None:
    output = {
        "steps": [
            {
                "timestep": 0,
                "scores": torch.tensor([[0.3, 0.1]]),
                "selected_idx": torch.tensor([0]),
                "live_indices": torch.tensor([0]),
                "current_query": torch.tensor([[1.0, 0.0]]),
                "candidate_queries": torch.tensor([[[0.0, 1.0], [1.0, 0.0]]]),
            }
        ]
    }
    audit = selection_quality_audit(
        output,
        torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
        ["target", "negative"],
        0.07,
        epsilon_stop=0.2,
    )

    assert audit["overall"]["score_margin_to_stop_mean"] == pytest.approx(0.1)

def test_gram_effective_rank_matches_direct_svd_for_flattened_delta() -> None:
    values = torch.randn(3, 5, 2, 7)
    flattened = values.flatten(start_dim=2).float()
    singular_values = torch.linalg.svdvals(flattened)
    expected = (
        singular_values.sum(dim=-1).square()
        / singular_values.square().sum(dim=-1).clamp_min(1e-8)
    ).mean()

    torch.testing.assert_close(_effective_rank(flattened), expected, rtol=1e-5, atol=1e-5)


def test_analyzer_orders_grounder_outputs_before_entities_and_actions() -> None:
    assert STAGES == (
        "proposals",
        "alpha_read",
        "exec_mask",
        "entities",
        "actions",
        "delta",
        "delta_q",
    )


def test_generic_metrics_preserve_dynamic_candidate_and_delta_axes() -> None:
    delta = torch.zeros(2, 5, 3, 4, requires_grad=True)
    metrics = generic_diversity(delta)

    assert len([key for key in metrics if key.startswith("mean_norm_c")]) == 5
    assert all(not isinstance(value, torch.Tensor) for value in metrics.values())


def test_grounding_and_mask_metrics_distinguish_identical_and_distinct_inputs() -> None:
    identical_alpha = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    distinct_alpha = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    identical_mask = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    separate_mask = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])

    alpha_same = alpha_read_diversity(identical_alpha)
    alpha_distinct = alpha_read_diversity(distinct_alpha)
    mask_same = exec_mask_diversity(identical_mask)
    mask_separate = exec_mask_diversity(separate_mask)

    assert alpha_same["pairwise_js_divergence"] == 0.0
    assert alpha_same["argmax_agreement"] == 1.0
    assert alpha_distinct["pairwise_js_divergence"] > 0.0
    assert alpha_distinct["argmax_agreement"] == 0.0
    assert mask_same["soft_iou"] == 1.0
    assert mask_separate["soft_iou"] == 0.0


def test_audit_emits_detached_per_step_and_sample_weighted_summaries() -> None:
    def step(timestep: int, batch: int, value: float) -> dict[str, object]:
        candidates = torch.full((batch, 3, 2), value, requires_grad=True)
        return {
            "timestep": timestep,
            "proposals": candidates,
            "alpha_read": torch.softmax(candidates, dim=-1),
            "entities": candidates,
            "actions": candidates,
            "exec_mask": torch.sigmoid(candidates),
            "delta": candidates.unsqueeze(-2),
            "delta_q": candidates,
        }

    audit = functional_collapse_audit({"steps": [step(0, 2, 1.0), step(1, 1, 2.0)]})

    assert set(audit["by_step"]) == {"t0", "t1"}
    assert audit["overall"]["live_samples"] == 3
    assert audit["overall"]["proposals"]["mean_norm"] == pytest.approx(2**0.5 * 4 / 3)
    assert not any(
        isinstance(value, torch.Tensor)
        for summary in audit["by_step"].values()
        for stage in summary.values()
        for value in (stage.values() if isinstance(stage, dict) else ())
    )


def test_analyzer_uses_audit_records_and_labels_heuristic(tmp_path) -> None:
    audit = functional_collapse_audit(
        {
            "steps": [
                {
                    "timestep": 0,
                    "proposals": torch.ones(1, 2, 3),
                    "alpha_read": torch.ones(1, 2, 3) / 3,
                    "entities": torch.ones(1, 2, 3),
                    "actions": torch.ones(1, 2, 3),
                    "exec_mask": torch.ones(1, 2, 3),
                    "delta": torch.ones(1, 2, 1, 3),
                    "delta_q": torch.ones(1, 2, 3),
                }
            ]
        }
    )
    path = tmp_path / "metrics.jsonl"
    path.write_text(json.dumps({"record_type": "train_update", "functional_collapse_audit": audit}) + "\n")

    records = _records(path)
    summary = _summarize(records)

    assert summary["proposals"]["pairwise_cosine"] == pytest.approx(1.0)
    assert _heuristic(summary) == "proposals"


class _AuditBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = SimpleNamespace(text_model=nn.Linear(2, 2))


class _AuditModel(nn.Module):
    def __init__(self, enabled: bool) -> None:
        super().__init__()
        self.backbone = _AuditBackbone()
        self.proposal = nn.Linear(2, 2)
        self.grounder = nn.Linear(2, 2)
        self.action_fusion = nn.Linear(2, 2)
        self.executor = nn.Linear(2, 2)
        self.score_net = nn.Linear(2, 1)
        self.config = SimpleNamespace(functional_collapse_audit_enabled=enabled)

    def forward(self, reference, input_ids, attention_mask, content_mask):
        del attention_mask, content_mask
        value = self.backbone.model.text_model(reference.float() + input_ids.float())
        for module in (self.proposal, self.grounder, self.action_fusion, self.executor):
            value = torch.tanh(module(value))
        candidates = value[:, None].expand(-1, 3, -1)
        return {
            "value": self.score_net(value).squeeze(-1),
            "steps": [
                {
                    "timestep": 0,
                    "proposals": candidates,
                    "alpha_read": torch.softmax(candidates, dim=-1),
                    "entities": candidates,
                    "actions": candidates,
                    "exec_mask": torch.sigmoid(candidates),
                    "delta": candidates.unsqueeze(-2),
                    "delta_q": candidates,
                }
            ],
        }

    def encode_global_images(self, pixels):
        return pixels.float()


class _AuditObjective(nn.Module):
    def forward(self, output, targets, target_ids, modification_texts):
        del target_ids, modification_texts
        loss = (output["value"] - targets.mean(dim=-1)).square().mean()
        return {"total": loss}


def _batch() -> ImageBatch:
    return ImageBatch(
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


def test_training_jsonl_emits_audit_only_when_enabled(tmp_path) -> None:
    for enabled in (False, True):
        model = _AuditModel(enabled)
        objective = _AuditObjective()
        logger = JSONLLogger(tmp_path / f"{enabled}.jsonl")
        train_one_epoch(
            model,
            objective,
            [_batch()],
            SGD(trainable_parameters(model, objective), lr=0.1),
            torch.amp.GradScaler("cuda", enabled=False),
            torch.device("cpu"),
            precision=PrecisionPolicy("fp32", False, None, False),
            epoch=0,
            logger=logger,
        )
        record = json.loads(logger.path.read_text())
        assert ("functional_collapse_audit" in record) is enabled
        if enabled:
            assert set(record["functional_collapse_audit"]["by_step"]["t0"]) >= {
                "proposals",
                "alpha_read",
                "entities",
                "actions",
                "exec_mask",
                "delta",
                "delta_q",
            }
