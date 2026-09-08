from __future__ import annotations

import pytest
import torch

from compare_matched_diagnostics import compare_feature_artifacts
from datasets.common import CIRSample
from diagnostics.cohort import (
    caption_change_masks,
    load_or_create_manifest,
    paired_rows,
    validate_processed_manifest,
)
from diagnostics.geometry import (
    assert_compatible_feature_interface,
    candidate_geometry,
    feature_geometry,
    matched_temporal_geometry,
    variance_decomposition,
)
from diagnostics.selection import (
    caption_utility_comparison,
    selection_metrics,
    slot_monopoly,
    transition_retrieval,
)
from models.iag_srme.utils.retrieval import marginal_teacher_utilities


class _Samples(torch.utils.data.Dataset):
    def __init__(self, count: int = 10) -> None:
        self.samples = [
            CIRSample(
                sample_id=f"sample-{index}",
                benchmark_id=None,
                reference_id=f"reference-{index}",
                target_id=f"target-{index}",
                modification_text=f"caption-{index}",
                category="dress",
            )
            for index in range(count)
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
        sample_count=6,
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


def test_val_manifest_creation_and_reuse_are_deterministic(tmp_path) -> None:
    dataset = _Samples(200)
    path = tmp_path / "shared_val_160.json"
    first, manifest = load_or_create_manifest(
        dataset,
        path,
        sample_count=160,
        batch_size=8,
        seed=42,
        split="val",
        caption_policy="ordered_and",
    )
    second, replayed = load_or_create_manifest(
        dataset,
        path,
        sample_count=160,
        batch_size=8,
        seed=42,
        split="val",
        caption_policy="ordered_and",
    )

    assert manifest == replayed
    assert [sample.sample_id for sample in first] == [sample.sample_id for sample in second]
    ids = [sample.sample_id for sample in second]
    metadata = validate_processed_manifest(replayed, ids, batch_size=8)
    assert metadata["manifest_sample_count"] == 160
    assert len(metadata["teacher_batch_grouping_sha256"]) == 64


def test_manifest_rejects_truncation_and_fingerprints_processed_order(tmp_path) -> None:
    dataset = _Samples(160)
    path = tmp_path / "manifest.json"
    cohort, manifest = load_or_create_manifest(
        dataset,
        path,
        sample_count=160,
        batch_size=8,
        seed=17,
        split="train",
        caption_policy="ordered_and",
    )
    with pytest.raises(ValueError, match="cannot be silently truncated"):
        load_or_create_manifest(
            dataset,
            path,
            sample_count=80,
            batch_size=8,
            seed=17,
            split="train",
            caption_policy="ordered_and",
        )

    ids = [sample.sample_id for sample in cohort]
    metadata = validate_processed_manifest(manifest, ids, batch_size=8)
    assert metadata["processed_sample_count"] == 160
    assert len(metadata["processed_sample_ids_sha256"]) == 64
    assert len(metadata["teacher_batch_grouping_sha256"]) == 64
    with pytest.raises(ValueError, match="differs from the manifest"):
        validate_processed_manifest(manifest, ids[:80], batch_size=8)


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
    assert slot_three["oracle_best_fraction_given_oracle_execute"] == 1.0
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


def test_execution_conditional_slot_occupancy_detects_monopoly() -> None:
    utility = torch.ones(100, 4)
    selected = torch.full((100,), 4)
    selected[50:] = 3
    report = selection_metrics(utility, selected, stop_threshold=0.0)
    slot_three = report["slot_metrics"][3]

    assert report["execute_count"] == 50
    assert slot_three["selected_fraction_of_all_decisions"] == 0.5
    assert slot_three["selected_fraction_given_execute"] == 1.0
    assert slot_monopoly(report) == {"detected": True, "slot": 3, "fraction": 1.0}


def test_harmful_execution_denominator_exact_example() -> None:
    utility = torch.ones(100, 2)
    utility[:10, 0] = -1
    selected = torch.full((100,), 2)
    selected[:20] = 0
    stop = selection_metrics(utility, selected, stop_threshold=0.0)["stop"]

    assert stop["executed_action_count"] == 20
    assert stop["harmful_execution_count"] == 10
    assert stop["harmful_execution_fraction_of_executions"] == 0.5
    assert stop["harmful_execution_fraction_of_decisions"] == 0.1


def test_stop_epsilon_distinguishes_positive_utility_from_oracle_execute() -> None:
    utility = torch.tensor([[0.03, -0.2]])
    report = selection_metrics(utility, torch.tensor([2]), stop_threshold=0.05)

    assert report["positive_utility_candidate_count_mean"] == 1.0
    assert report["oracle_execute_count"] == 0
    assert report["stop"]["oracle_stop_count"] == 1


def test_representation_space_guard_rejects_patch_query_comparison() -> None:
    patch = torch.randn(8, 768)
    query = torch.randn(8, 512)
    with pytest.raises(ValueError, match="cannot compare"):
        assert_compatible_feature_interface(
            patch,
            query,
            first_name="V0_pooled",
            second_name="terminal_query",
            interface="trajectory",
        )
    assert_compatible_feature_interface(
        query,
        torch.randn(8, 512),
        first_name="initial_query",
        second_name="terminal_query",
        interface="retrieval-query",
    )


def test_duplicate_caption_strings_are_excluded_from_sensitivity_cohort() -> None:
    texts = ["make it red", "make it red", "remove sleeves"]
    masks = caption_change_masks(texts, torch.tensor([1, 0, 2]))

    assert masks["index_changed"].tolist() == [True, True, False]
    assert masks["text_changed"].tolist() == [False, False, False]
    assert masks["duplicate_caption_unchanged"].tolist() == [True, True, False]


def test_matched_timestep_intersection_uses_only_shared_live_ids() -> None:
    generator = torch.Generator().manual_seed(31)

    def artifact(ids: list[str]) -> dict:
        return {
            "manifest_sample_ids_sha256": "same",
            "teacher_batch_grouping_sha256": "same-batches",
            "timesteps": {
                "1": {
                    "sample_ids": ids,
                    "features": {
                        "proposal": torch.randn(len(ids), 4, 12, generator=generator),
                        "current_query": torch.randn(len(ids), 8, generator=generator),
                    },
                }
            },
        }

    report = compare_feature_artifacts(
        artifact(["A", "B", "C", "D"]),
        artifact(["B", "C", "D", "E"]),
    )["timesteps"]["1"]

    assert report["intersection_sample_ids"] == ["B", "C", "D"]
    assert report["intersection_count"] == 3
    assert report["features"]["proposal"]["matched_old"]["raw"]["pooled"]["count"] == 12

    empty = compare_feature_artifacts(artifact(["A"]), artifact(["B"]))["timesteps"]["1"]
    assert empty["intersection_count"] == 0
    assert empty["features"]["proposal"]["matched_old"]["available"] is False


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


def test_temporal_geometry_matches_survivors_before_drawing_collapse_conclusions() -> None:
    ids_t0 = ["A", "B", "C", "D", "E", "F"]
    ids_t1 = ["A", "B", "C"]
    features_t0 = torch.eye(6)
    features_t1 = features_t0[:3].clone()

    native_t0 = feature_geometry(features_t0)
    native_t1 = feature_geometry(features_t1)
    matched = matched_temporal_geometry(ids_t0, features_t0, ids_t1, features_t1)

    assert native_t0["effective_rank_pr"] != pytest.approx(
        native_t1["effective_rank_pr"]
    )
    assert matched["matched_sample_count"] == 3
    assert matched["matched_sample_ids"] == ids_t1
    assert matched["baseline_matched_PR"] == pytest.approx(matched["current_matched_PR"])
    assert matched["rank_drop_fraction"] == pytest.approx(0.0)
    assert matched["pairwise_cosine_increase"] == pytest.approx(0.0)
    assert not (
        matched["rank_conclusion_valid"] and matched["rank_drop_fraction"] > 0.30
    )
    assert not (
        matched["cosine_conclusion_valid"]
        and matched["pairwise_cosine_increase"] > 0.15
    )


@pytest.mark.parametrize(
    ("epsilon", "correct", "shuffled", "best_advantage", "oracle_advantage"),
    [
        (0.0, [-0.02, -0.03, -0.01, -0.04], [-0.20, -0.10, -0.30, -0.15], 0.09, 0.0),
        (0.05, [0.03, -0.10], [-0.02, -0.20], 0.05, 0.0),
        (0.05, [0.12, -0.10], [0.03, -0.20], 0.09, 0.12),
    ],
)
def test_caption_best_candidate_and_stop_aware_oracle_are_distinct(
    epsilon: float,
    correct: list[float],
    shuffled: list[float],
    best_advantage: float,
    oracle_advantage: float,
) -> None:
    report = caption_utility_comparison(
        torch.tensor([correct]),
        torch.tensor([shuffled]),
        stop_threshold=epsilon,
    )

    assert float(report["best_candidate_utility_correct_minus_shuffled"]) == pytest.approx(
        best_advantage
    )
    assert float(report["oracle_policy_utility_correct_minus_shuffled"]) == pytest.approx(
        oracle_advantage
    )
