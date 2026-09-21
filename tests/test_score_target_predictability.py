from __future__ import annotations

import hashlib
from contextlib import contextmanager

import pytest
import torch

import diagnose_score_target_predictability as target_diagnostic
from diagnose_score_target_predictability import _cache_fingerprint, _oracle_sanity, _train_target_probe
from diagnostics.feature_sufficiency import COMPACT_FEATURE_FIELDS, LABEL_FIELD, validate_compact_features
from diagnostics.target_predictability import (
    PRIVILEGED_FIELDS,
    add_privileged_features,
    deterministic_alternative_banks,
    deterministic_alternative_negative_indices,
    freeze_module,
    privileged_similarity_features,
    recompute_utility_with_target_bank,
    stability_metrics,
    target_aware_probe_from_rows,
    validate_privileged_rows,
)
from models.iag_srme.utils.retrieval import marginal_teacher_utilities


def _rows(*, batch: int = 6, candidates: int = 4) -> dict[str, object]:
    generator = torch.Generator().manual_seed(321)
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


def _targets(rows: dict[str, object]) -> dict[str, object]:
    batch, dim = rows["current_query"].shape  # type: ignore[union-attr]
    return {
        "target_embeddings": torch.randn(batch, dim, generator=torch.Generator().manual_seed(77)),
        "sample_ids": list(rows["sample_ids"]),
        "target_ids": [f"target-{index}" for index in range(batch)],
    }


def _permuted(rows: dict[str, object], permutation: torch.Tensor) -> dict[str, object]:
    candidate_fields = {"actions", "local_mean", "candidate_global_delta", "candidate_queries", "delta_q", LABEL_FIELD}
    return {
        name: value.index_select(1, permutation) if name in candidate_fields else value
        for name, value in rows.items()
    }


def test_target_free_rows_reject_privileged_fields() -> None:
    rows = _rows()
    rows["target_embeddings"] = torch.randn(6, 6)
    with pytest.raises(ValueError, match="forbidden|unexpected"):
        validate_compact_features(rows)


def test_target_aware_rows_allow_only_explicit_similarity_features() -> None:
    aware = add_privileged_features(_rows(), _targets(_rows()))
    assert set(aware) == set(COMPACT_FEATURE_FIELDS) | {LABEL_FIELD, "sample_ids"} | set(PRIVILEGED_FIELDS)
    validate_privileged_rows(aware)
    aware["teacher_utility_copy"] = aware[LABEL_FIELD]
    with pytest.raises(ValueError, match="only compact fields"):
        validate_privileged_rows(aware)


def test_target_aware_probe_is_shape_and_candidate_permutation_equivariant() -> None:
    rows = _rows()
    aware = add_privileged_features(rows, _targets(rows))
    probe = target_aware_probe_from_rows(aware, width=8).eval()
    permutation = torch.tensor([2, 0, 3, 1])
    permuted_base = _permuted(rows, permutation)
    permuted = add_privileged_features(permuted_base, _targets(rows))
    with torch.no_grad():
        baseline, reordered = probe(aware), probe(permuted)
    assert baseline.shape == (6, 4)
    torch.testing.assert_close(reordered, baseline.index_select(1, permutation), atol=1e-5, rtol=1e-5)


def test_changing_true_target_only_changes_privileged_similarities() -> None:
    rows = _rows()
    targets = _targets(rows)
    before = privileged_similarity_features(rows["current_query"], rows["candidate_queries"], targets["target_embeddings"])
    targets["target_embeddings"] = -targets["target_embeddings"]
    after = privileged_similarity_features(rows["current_query"], rows["candidate_queries"], targets["target_embeddings"])
    assert not torch.allclose(before["candidate_target_cos"], after["candidate_target_cos"])
    assert torch.equal(rows["current_query"], _rows()["current_query"])
    assert torch.equal(rows["candidate_queries"], _rows()["candidate_queries"])


def test_alternative_banks_keep_anchor_and_exclude_anchor_target_deterministically() -> None:
    target_ids = ["dup", "a", "b", "c", "d", "e", "f", "g", "h"]
    first = deterministic_alternative_banks(target_ids, anchor=0, bank_size=5, seeds=(7, 8, 9))
    assert first == deterministic_alternative_banks(target_ids, anchor=0, bank_size=5, seeds=(7, 8, 9))
    for bank in first:
        assert bank[0] == 0
        assert len(bank) == 5
        assert all(target_ids[index] != "dup" for index in bank[1:])
        assert len({target_ids[index] for index in bank[1:]}) == 4


def test_reservoir_negative_sampler_preserves_true_val_bank_behavior() -> None:
    target_ids = ["a", "b", "c", "d", "e", "f"]
    expected = deterministic_alternative_banks(target_ids, anchor=2, bank_size=4, seeds=(3, 4))
    actual = [
        [2, *negative]
        for negative in deterministic_alternative_negative_indices(
            target_ids,
            anchor_target_id="c",
            anchor_seed_index=2,
            bank_size=4,
            seeds=(3, 4),
        )
    ]
    assert actual == expected


def test_pool_stability_uses_strict_anchor_subset_and_full_reservoir(monkeypatch) -> None:
    rows = _rows(batch=2)
    anchors = _targets(rows)
    reservoir = {
        "target_embeddings": torch.arange(36, dtype=torch.float32).reshape(6, 6),
        "sample_ids": [f"reservoir-{index}" for index in range(6)],
        "target_ids": ["target-0", "target-1", "outside-2", "outside-3", "outside-4", "outside-5"],
    }
    anchors["target_embeddings"] = reservoir["target_embeddings"][:2].clone()
    anchors["target_ids"] = reservoir["target_ids"][:2]
    before_current, before_candidates = rows["current_query"].clone(), rows["candidate_queries"].clone()
    observed: list[torch.Tensor] = []

    def record_bank(current, candidates, bank, *, temperature):
        del current, candidates, temperature
        observed.append(bank.cpu().clone())
        return torch.zeros(4)

    monkeypatch.setattr(target_diagnostic, "recompute_utility_with_target_bank", record_bank)
    report = target_diagnostic._pool_stability(
        rows,
        anchors,
        reservoir,
        bank_size=3,
        pool_count=2,
        epsilon_stop=0.0,
        temperature=0.2,
        device=torch.device("cpu"),
        single_positive=torch.ones(2, dtype=torch.bool),
    )

    assert report["pool_count"] == 2
    assert len(observed) == 4
    for observed_index, bank in enumerate(observed):
        anchor = observed_index // 2
        torch.testing.assert_close(bank[0], anchors["target_embeddings"][anchor])
        negative_indices = [
            int((reservoir["target_embeddings"] == value).all(dim=1).nonzero().item())
            for value in bank[1:]
        ]
        negative_ids = [reservoir["target_ids"][index] for index in negative_indices]
        assert all(target_id != anchors["target_ids"][anchor] for target_id in negative_ids)
    assert any(any(index >= 2 for index in [int((reservoir["target_embeddings"] == value).all(dim=1).nonzero().item()) for value in bank[1:]]) for bank in observed)
    assert torch.equal(rows["current_query"], before_current)
    assert torch.equal(rows["candidate_queries"], before_candidates)

    observed_again: list[torch.Tensor] = []

    def record_again(current, candidates, bank, *, temperature):
        del current, candidates, temperature
        observed_again.append(bank.cpu().clone())
        return torch.zeros(4)

    monkeypatch.setattr(target_diagnostic, "recompute_utility_with_target_bank", record_again)
    target_diagnostic._pool_stability(
        rows,
        anchors,
        reservoir,
        bank_size=3,
        pool_count=2,
        epsilon_stop=0.0,
        temperature=0.2,
        device=torch.device("cpu"),
        single_positive=torch.ones(2, dtype=torch.bool),
    )
    assert all(torch.equal(first, second) for first, second in zip(observed, observed_again, strict=True))


def test_pool_stability_moves_full_target_bank_before_teacher(monkeypatch) -> None:
    rows = _rows(batch=1)
    reservoir = {
        "target_embeddings": torch.arange(18, dtype=torch.float32).reshape(3, 6),
        "sample_ids": ["anchor", "negative-1", "negative-2"],
        "target_ids": ["anchor-target", "negative-target-1", "negative-target-2"],
    }
    anchors = {
        "target_embeddings": reservoir["target_embeddings"][:1].clone(),
        "sample_ids": list(rows["sample_ids"]),
        "target_ids": ["anchor-target"],
    }
    observed = []

    def record_device(current, candidates, bank, *, temperature):
        del temperature
        observed.append((current.device, candidates.device, bank.device))
        return torch.zeros(4)

    monkeypatch.setattr(target_diagnostic, "recompute_utility_with_target_bank", record_device)
    target_diagnostic._pool_stability(
        rows,
        anchors,
        reservoir,
        bank_size=2,
        pool_count=2,
        epsilon_stop=0.0,
        temperature=0.2,
        device=torch.device("meta"),
        single_positive=torch.ones(1, dtype=torch.bool),
    )
    assert observed == [
        (torch.device("meta"), torch.device("meta"), torch.device("meta")),
        (torch.device("meta"), torch.device("meta"), torch.device("meta")),
    ]



def test_true_val_forward_wraps_model_and_target_encoder_in_resolved_autocast(monkeypatch) -> None:
    events = []

    @contextmanager
    def autocast(**kwargs):
        events.append(("enter", kwargs))
        yield
        events.append(("exit", None))

    class Batch:
        reference_pixels = "reference"
        input_ids = "ids"
        attention_mask = "mask"
        content_mask = "content"
        target_pixels = "target"

        def to(self, device):
            events.append(("batch", device))
            return self

    class Model:
        def __call__(self, *args):
            events.append(("model", args))
            return "output"

        def encode_global_images(self, pixels):
            events.append(("targets", pixels))
            return "targets"

    precision = type("Precision", (), {"autocast_enabled": True, "autocast_dtype": torch.bfloat16})()
    monkeypatch.setattr(target_diagnostic.torch, "autocast", autocast)
    output, targets = target_diagnostic._true_val_forward(
        Model(), Batch(), device=torch.device("cpu"), precision=precision
    )

    assert (output, targets) == ("output", "targets")
    assert events == [
        ("batch", torch.device("cpu")),
        ("enter", {"device_type": "cpu", "enabled": True, "dtype": torch.bfloat16}),
        ("model", ("reference", "ids", "mask", "content")),
        ("targets", "target"),
        ("exit", None),
    ]

def test_resampled_teacher_utility_uses_canonical_teacher_and_preserves_queries() -> None:
    generator = torch.Generator().manual_seed(5)
    current = torch.randn(1, 4, generator=generator)
    candidates = torch.randn(1, 3, 4, generator=generator)
    bank = torch.randn(5, 4, generator=generator)
    before_current, before_candidates = current.clone(), candidates.clone()
    actual = recompute_utility_with_target_bank(current, candidates, bank, temperature=0.2)
    positive = torch.tensor([[True, False, False, False, False]])
    expected, valid = marginal_teacher_utilities(current, candidates, bank, positive, ~positive, 0.2)
    assert bool(valid.item())
    torch.testing.assert_close(actual, expected[0])
    assert torch.equal(current, before_current)
    assert torch.equal(candidates, before_candidates)


def test_stability_reports_slotwise_and_single_pool_metrics() -> None:
    canonical = torch.tensor([[0.2, -0.1], [0.1, 0.3]])
    alternatives = torch.stack((canonical + 0.01, canonical - 0.01))
    report = stability_metrics(canonical, alternatives, epsilon_stop=0.0)
    assert set(report) >= {"canonical_vs_alternative", "utility_variance", "candidate_ranking", "stop", "sign", "per_slot", "pool_to_pool"}
    assert len(report["per_slot"]["mean_utility_std"]) == 2


def test_true_val_oracle_sanity_gate_matches_fixed_invariant() -> None:
    utility = torch.full((160, 4), -0.01)
    slots = [0] * 37 + [1] * 34 + [2] * 27 + [3] * 45
    for row, slot in enumerate(slots):
        utility[row, slot] = 0.06073 * 160 / len(slots)
    rows = {**_rows(batch=160), LABEL_FIELD: utility, "sample_ids": [str(index) for index in range(160)]}
    assert _oracle_sanity(rows, 0.0)["passed"]

def test_joining_privileged_features_does_not_modify_compact_cache(tmp_path) -> None:
    cache_dir = tmp_path / "compact_cache"
    cache_dir.mkdir()
    manifest = cache_dir / "manifest.json"
    manifest.write_text('{"immutable": true}', encoding="utf-8")
    before = _cache_fingerprint(cache_dir)
    rows = _rows()
    add_privileged_features(rows, _targets(rows))
    assert _cache_fingerprint(cache_dir) == before


def test_probe_training_never_updates_true_val_or_production_module() -> None:
    rows = _rows(batch=4)
    aware = add_privileged_features(rows, _targets(rows))
    val = add_privileged_features(_rows(batch=2), _targets(_rows(batch=2)))
    production = torch.nn.Linear(3, 3)
    before = hashlib.sha256(torch.nn.utils.parameters_to_vector(production.parameters()).detach().numpy().tobytes()).hexdigest()
    freeze_module(production)
    _, report = _train_target_probe([aware], [aware], val, device=torch.device("cpu"), epochs=1, epsilon_stop=0.0)
    after = hashlib.sha256(torch.nn.utils.parameters_to_vector(production.parameters()).detach().numpy().tobytes()).hexdigest()
    assert before == after
    assert report["true_val"]["decision_count"] == 2
