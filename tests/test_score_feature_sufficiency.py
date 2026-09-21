from __future__ import annotations

import torch
import pytest

from diagnose_score_feature_sufficiency import evaluate_probe, train_probe
from diagnostics.feature_sufficiency import (
    COMPACT_FEATURE_FIELDS,
    LABEL_FIELD,
    compact_cache_bytes,
    compact_local_mean,
    compact_t0_features,
    probe_from_rows,
    probe_metrics,
    select_probe_scores,
    to_cache_precision,
    to_training_precision,
    validate_compact_features,
)


def _rows(*, batch: int = 5, candidates: int = 4) -> dict[str, object]:
    generator = torch.Generator().manual_seed(117)
    state, text, action, query = 7, 5, 7, 6
    return {
        "current_global": torch.randn(batch, state, generator=generator),
        "text_global": torch.randn(batch, text, generator=generator),
        "actions": torch.randn(batch, candidates, action, generator=generator),
        "local_mean": torch.randn(batch, candidates, state, generator=generator),
        "candidate_global_delta": torch.randn(batch, candidates, state, generator=generator),
        "current_query": torch.randn(batch, query, generator=generator),
        "candidate_queries": torch.randn(batch, candidates, query, generator=generator),
        "delta_q": torch.randn(batch, candidates, query, generator=generator),
        LABEL_FIELD: torch.randn(batch, candidates, generator=generator),
        "sample_ids": [f"sample-{index}" for index in range(batch)],
    }


def _permute(rows: dict[str, object], permutation: torch.Tensor) -> dict[str, object]:
    candidate_fields = {
        "actions",
        "local_mean",
        "candidate_global_delta",
        "candidate_queries",
        "delta_q",
        LABEL_FIELD,
    }
    return {
        name: value.index_select(1, permutation) if name in candidate_fields else value
        for name, value in rows.items()
    }


def test_compact_feature_input_excludes_targets_and_matches_scorenet_local_reduction() -> None:
    generator = torch.Generator().manual_seed(19)
    delta = torch.randn(2, 4, 9, 7, generator=generator)
    mask = torch.rand(2, 4, 9, generator=generator)
    step = {
        "current_global": torch.randn(2, 7, generator=generator),
        "text_global": torch.randn(2, 5, generator=generator),
        "actions": torch.randn(2, 4, 7, generator=generator),
        "delta": delta,
        "exec_mask": mask,
        "candidate_global": torch.randn(2, 4, 7, generator=generator),
        "current_query": torch.randn(2, 6, generator=generator),
        "candidate_queries": torch.randn(2, 4, 6, generator=generator),
        "delta_q": torch.randn(2, 4, 6, generator=generator),
    }
    rows = {**compact_t0_features(step, torch.randn(2, 4, generator=generator)), "sample_ids": ["a", "b"]}

    assert set(rows) == set(COMPACT_FEATURE_FIELDS) | {LABEL_FIELD, "sample_ids"}
    torch.testing.assert_close(rows["local_mean"], compact_local_mean(delta, mask))
    validate_compact_features(rows)
    rows["target_embedding"] = torch.randn(2, 6)
    with pytest.raises(ValueError, match="forbidden"):
        validate_compact_features(rows)


def test_compact_cache_stays_compact_and_round_trips_fp16_features() -> None:
    rows = _rows()
    cache_rows = to_cache_precision(rows)
    assert all(cache_rows[name].dtype == torch.float16 for name in COMPACT_FEATURE_FIELDS)
    assert cache_rows[LABEL_FIELD].dtype == torch.float32
    assert compact_cache_bytes(cache_rows) < 10_000
    restored = to_training_precision(cache_rows, torch.device("cpu"))
    assert all(restored[name].dtype == torch.float32 for name in COMPACT_FEATURE_FIELDS)


def test_probe_evaluation_never_updates_weights() -> None:
    rows = _rows()
    probe = probe_from_rows("legacy_independent", rows, width=8).eval()
    before = {name: parameter.detach().clone() for name, parameter in probe.named_parameters()}
    with torch.no_grad():
        report = evaluate_probe(probe, rows, torch.device("cpu"), epsilon_stop=0.0)
    assert report["decision_count"] == 5
    assert all(torch.equal(before[name], parameter) for name, parameter in probe.named_parameters())


def test_probe_training_keeps_validation_probe_frozen() -> None:
    train_rows, dev_rows, val_rows = _rows(batch=4), _rows(batch=3), _rows(batch=2)
    probe, report = train_probe(
        "legacy_independent",
        [train_rows],
        [dev_rows],
        val_rows,
        device=torch.device("cpu"),
        epochs=1,
        learning_rate=1e-3,
        weight_decay=0.0,
        epsilon_stop=0.0,
    )
    assert probe.training is False
    assert report["parameter_count"] > 0
    assert report["true_val"]["decision_count"] == 2


def test_legacy_probes_ignore_retrieval_features() -> None:
    rows = _rows()
    changed = {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in rows.items()}
    changed["current_query"] += 100.0
    changed["candidate_queries"] += 100.0
    changed["delta_q"] += 100.0
    for name in ("legacy_independent", "legacy_set_relative"):
        probe = probe_from_rows(name, rows, width=8).eval()
        with torch.no_grad():
            torch.testing.assert_close(probe(rows), probe(changed), atol=1e-5, rtol=1e-5)


def test_empty_probe_metrics_are_reported_without_crashing() -> None:
    report = probe_metrics(torch.empty(0, 4), torch.empty(0, 4), epsilon_stop=0.0)
    assert report["available"] is False
    assert report["decision_count"] == 0


def test_invalid_teacher_rows_are_rejected_before_probe_forward() -> None:
    rows = _rows()
    rows[LABEL_FIELD][0, 0] = torch.nan
    with pytest.raises(ValueError, match="teacher_utility contains NaN"):
        probe_from_rows("legacy_independent", rows, width=8)


@pytest.mark.parametrize(
    "name",
    ("legacy_independent", "legacy_set_relative", "retrieval_augmented_independent"),
)
def test_probe_shapes_and_candidate_permutation_equivariance(name: str) -> None:
    rows = _rows()
    probe = probe_from_rows(name, rows, width=8).eval()
    permutation = torch.tensor([2, 0, 3, 1])
    with torch.no_grad():
        baseline = probe(rows)
        permuted = probe(_permute(rows, permutation))
    assert baseline.shape == (5, 4)
    torch.testing.assert_close(permuted, baseline.index_select(1, permutation), atol=1e-5, rtol=1e-5)


def test_independent_probes_do_not_mix_candidate_identity() -> None:
    rows = _rows(batch=1)
    for name in ("legacy_independent", "retrieval_augmented_independent"):
        probe = probe_from_rows(name, rows, width=8).eval()
        changed = {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in rows.items()}
        for field in ("actions", "local_mean", "candidate_global_delta", "candidate_queries", "delta_q"):
            changed[field][:, 1] += 100.0
        with torch.no_grad():
            before, after = probe(rows), probe(changed)
        torch.testing.assert_close(before[:, [0, 2, 3]], after[:, [0, 2, 3]], atol=1e-5, rtol=1e-5)
        assert not torch.allclose(before[:, 1], after[:, 1])


def test_probe_policy_preserves_epsilon_stop_semantics() -> None:
    scores = torch.tensor([[0.0, -0.1, -0.2], [0.02, 0.01, -1.0], [0.01, 0.03, 0.02]])
    selected = select_probe_scores(scores, epsilon_stop=0.015)
    assert selected.tolist() == [3, 0, 1]
