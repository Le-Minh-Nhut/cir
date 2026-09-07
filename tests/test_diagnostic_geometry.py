from __future__ import annotations

import torch

from datasets.common import CIRSample
from diagnostics.cohort import load_or_create_manifest, paired_rows
from diagnostics.geometry import candidate_geometry, variance_decomposition
from diagnostics.selection import selection_metrics
from diagnostics.selection import transition_retrieval
from models.iag_srme.utils.retrieval import marginal_teacher_utilities


class _Samples(torch.utils.data.Dataset):
    def __init__(self) -> None:
        self.samples = [
            CIRSample(
                sample_id=f"sample-{index}",
                benchmark_id=None,
                reference_id=f"reference-{index}",
                target_id=f"target-{index}",
                modification_text=f"caption-{index}",
                category="dress",
            )
            for index in range(10)
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> CIRSample:
        return self.samples[index]


def test_slot_offsets_do_not_masquerade_as_input_collapse() -> None:
    generator = torch.Generator().manual_seed(7)
    batch, candidates, dim = 256, 4, 64
    sample_variation = torch.randn(batch, candidates, dim, generator=generator)
    offsets = torch.zeros(candidates, dim)
    offsets[:, :candidates] = 80.0 * torch.eye(candidates)
    report = candidate_geometry(sample_variation + offsets)
    raw = report["raw"]

    assert raw["pooled"]["effective_rank_pr"] < 6.0
    assert raw["per_slot_summary"]["min_per_slot_PR"] > 30.0
    assert raw["slot_centered_pooled"]["effective_rank_pr"] > 40.0
    assert raw["variance_decomposition"]["between_slot_variance_fraction"] > 0.9


def test_manifest_replays_same_ids_captions_and_order(tmp_path) -> None:
    dataset = _Samples()
    path = tmp_path / "manifest.json"
    first, manifest = load_or_create_manifest(
        dataset,
        path,
        sample_count=6,
        batch_size=2,
        seed=17,
        split="train",
        caption_policy="ordered_and",
    )
    second, replayed = load_or_create_manifest(
        dataset,
        path,
        sample_count=2,
        batch_size=2,
        seed=17,
        split="train",
        caption_policy="ordered_and",
    )

    assert manifest == replayed
    assert [sample.sample_id for sample in first] == [sample.sample_id for sample in second]
    assert [sample.modification_text for sample in first] == [
        sample.modification_text for sample in second
    ]


def test_true_per_slot_collapse_and_zero_variance_are_explicit() -> None:
    values = torch.randn(4, 32)
    collapsed = values[None].expand(40, -1, -1).clone()
    report = candidate_geometry(collapsed)["raw"]

    assert report["variance_decomposition"]["within_slot_variance"] < 1e-10
    for slot in report["per_slot"]:
        assert slot["zero_total_variance"] is True
        assert slot["degenerate_spectrum"] is True
        assert slot["effective_rank_pr"] == 0.0


def test_covariance_trace_decomposition_closes() -> None:
    values = torch.randn(31, 4, 17, generator=torch.Generator().manual_seed(11))
    report = variance_decomposition(values)
    assert report["decomposition_ok"] is True
    assert (
        abs(
            report["total_variance"]
            - report["within_slot_variance"]
            - report["between_slot_variance"]
        )
        <= report["decomposition_tolerance"]
    )


def test_paired_cohort_uses_only_executed_sample_ids() -> None:
    ids = [f"sample-{index}" for index in range(160)]
    before_values = torch.arange(160, dtype=torch.float32)[:, None]
    after = {sample_id: before_values[index] + 1 for index, sample_id in enumerate(ids[:153])}

    before, after_values, paired_ids = paired_rows(ids, before_values, after)

    assert paired_ids == ids[:153]
    assert before.shape[0] == after_values.shape[0] == 153
    torch.testing.assert_close(after_values - before, torch.ones_like(before))


def test_slot_and_oracle_metrics_recover_genuinely_best_slot() -> None:
    utility = torch.tensor(
        [
            [-0.2, 0.1, 0.0, 0.8],
            [-0.1, 0.2, 0.1, 0.7],
            [0.0, 0.1, 0.2, 0.9],
            [0.5, 0.1, 0.0, 0.8],
        ]
    )
    selected = torch.tensor([3, 1, 3, 0])
    report = selection_metrics(utility, selected, stop_threshold=0.0)

    slot_three = report["slot_metrics"][3]
    assert slot_three["oracle_best_count"] == 4
    assert slot_three["oracle_best_fraction"] == 1.0
    assert report["selected_equals_oracle_fraction"] == 0.5
    assert report["mean_oracle_utility"] > report["mean_selected_utility"]
    assert report["mean_regret"] > 0


def test_stop_confusion_and_harmful_execution_denominators() -> None:
    utility = torch.tensor(
        [
            [-0.3, -0.2],
            [-0.4, -0.1],
            [-0.2, -0.5],
            [-0.7, -0.6],
        ]
    )
    # STOP, harmful execute, STOP, harmful execute.
    selected = torch.tensor([2, 0, 2, 1])
    stop = selection_metrics(utility, selected, stop_threshold=0.0)["stop"]

    assert (stop["tp"], stop["fp"], stop["tn"], stop["fn"]) == (2, 0, 0, 2)
    assert stop["precision"] == 1.0
    assert stop["recall"] == 0.5
    assert stop["harmful_execution_count"] == 2
    assert stop["executed_action_count"] == 2
    assert stop["harmful_execution_fraction_of_executions"] == 1.0
    assert stop["harmful_execution_fraction_of_decisions"] == 0.5


def test_transition_diagnostic_reuses_teacher_utility_semantics() -> None:
    current = torch.nn.functional.normalize(torch.randn(3, 8), dim=-1)
    candidates = torch.nn.functional.normalize(torch.randn(3, 4, 8), dim=-1)
    targets = torch.nn.functional.normalize(torch.randn(3, 8), dim=-1)
    positive = torch.eye(3, dtype=torch.bool)
    negative = ~positive
    expected, expected_valid = marginal_teacher_utilities(
        current, candidates, targets, positive, negative, 0.07
    )

    report = transition_retrieval(current, candidates, targets, positive, negative, 0.07)

    torch.testing.assert_close(report["utility"], expected)
    torch.testing.assert_close(report["valid"], expected_valid)
