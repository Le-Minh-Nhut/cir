from __future__ import annotations

import copy
import json

import pytest
import torch
from omegaconf import OmegaConf

from models.iag_srme import ScoreNet
from refit_score_net import (
    _build_refit_checkpoint,
    _json_fingerprint,
    _load_cache_manifest,
    _load_shard,
    _sha256_file,
    _validate_refit_cache_contract,
)
from training.scorer_refit import (
    SCORER_TENSOR_FIELDS,
    assert_scorer_optimizer_ownership,
    build_score_refit_optimizer,
    non_score_parameter_fingerprint,
    scorer_gain_loss,
    validate_scorer_batch,
)


def _scorer_batch(
    *,
    rows: int = 8,
    candidates: int = 4,
    state_dim: int = 16,
    text_dim: int = 8,
    patches: int = 9,
) -> dict[str, object]:
    generator = torch.Generator().manual_seed(73)
    return {
        "current_global": torch.randn(rows, state_dim, generator=generator),
        "text_global": torch.randn(rows, text_dim, generator=generator),
        "actions": torch.randn(rows, candidates, state_dim, generator=generator),
        "delta": torch.randn(rows, candidates, patches, state_dim, generator=generator),
        "exec_mask": torch.rand(rows, candidates, patches, generator=generator),
        "candidate_global": torch.randn(
            rows, candidates, state_dim, generator=generator
        ),
        "teacher_utility": torch.randn(rows, candidates, generator=generator),
        "timestep": torch.arange(rows, dtype=torch.long) % 3,
        "sample_ids": [f"sample-{index}" for index in range(rows)],
    }


def test_score_refit_optimizer_owns_every_and_only_scorenet_parameter(model) -> None:
    non_score_before = non_score_parameter_fingerprint(model)
    optimizer = build_score_refit_optimizer(
        model, learning_rate=1e-2, weight_decay=0.0
    )
    assert_scorer_optimizer_ownership(model, optimizer)
    optimizer_ids = {
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    }
    assert optimizer_ids == {id(parameter) for parameter in model.score_net.parameters()}
    assert all(parameter.requires_grad for parameter in model.score_net.parameters())
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if not name.startswith("score_net.")
    )

    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    batch = _scorer_batch()
    optimizer.zero_grad(set_to_none=True)
    loss, _ = scorer_gain_loss(model.score_net, batch, huber_delta=1.0)
    loss.backward()
    for module_name in (
        "context_global",
        "context_text",
        "context_norm",
        "action_projection",
        "local_projection",
        "global_projection",
        "net",
    ):
        module = getattr(model.score_net, module_name)
        gradient_sum = sum(
            float(parameter.grad.abs().sum())
            for parameter in module.parameters()
            if parameter.grad is not None
        )
        assert gradient_sum > 0, module_name
    optimizer.step()

    assert any(
        not torch.equal(before[name], parameter)
        for name, parameter in model.named_parameters()
        if name.startswith("score_net.")
    )
    assert all(
        torch.equal(before[name], parameter)
        for name, parameter in model.named_parameters()
        if not name.startswith("score_net.")
    )
    assert non_score_parameter_fingerprint(model) == non_score_before


def test_refit_cache_contract_rejects_score_dropout_mismatch() -> None:
    cfg = OmegaConf.create(
        {
            "model": {
                "num_candidates": 4,
                "max_steps": 3,
                "stop_enabled": True,
                "epsilon_stop": 0.0,
                "score_dropout": 0.1,
            },
            "objective": {"retrieval_temperature": 0.07},
        }
    )
    manifest = {
        "source_checkpoint_sha256": "source-sha",
        "num_candidates": 4,
        "max_steps": 3,
        "stop_enabled": True,
        "epsilon_stop": 0.0,
        "score_dropout": 0.1,
        "retrieval_temperature": 0.07,
    }
    source_objective = {"retrieval_temperature": 0.07}
    _validate_refit_cache_contract(
        cfg,
        manifest,
        source_checkpoint_sha256="source-sha",
        source_objective=source_objective,
    )

    manifest["score_dropout"] = 0.5
    with pytest.raises(ValueError, match="contract mismatch for score_dropout"):
        _validate_refit_cache_contract(
            cfg,
            manifest,
            source_checkpoint_sha256="source-sha",
            source_objective=source_objective,
        )


def test_refit_checkpoint_is_inference_ready_but_not_full_resume(
    model, features, tmp_path
) -> None:
    optimizer = build_score_refit_optimizer(
        model, learning_rate=1e-3, weight_decay=0.0
    )
    source = {
        "model": copy.deepcopy(model.state_dict()),
        "objective": {},
        "optimizer": {"source_full_optimizer": True},
        "scaler": {"source_scaler": True},
        "metadata": {"experiment_identity": "R0-NCLS-TEXT"},
    }
    metadata = {
        "optimizer_scope": "score_net_only",
        "exact_full_training_resume": False,
        "resume_semantics": "warm_start_only_for_full_training",
    }
    checkpoint = _build_refit_checkpoint(
        source,
        model,
        optimizer,
        metadata=metadata,
        epochs=2,
        optimizer_steps=7,
    )

    assert checkpoint["optimizer"] is None
    assert checkpoint["scaler"] is None
    assert checkpoint["score_refit_optimizer"] == optimizer.state_dict()
    assert checkpoint["metadata"]["optimizer_scope"] == "score_net_only"
    assert checkpoint["metadata"]["exact_full_training_resume"] is False
    assert checkpoint["metadata"]["resume_semantics"] == (
        "warm_start_only_for_full_training"
    )

    path = tmp_path / "score_refit.pt"
    torch.save(checkpoint, path)
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    restored = copy.deepcopy(model)
    restored.load_state_dict(loaded["model"])
    state, tokens, text, mask = features
    model.eval()
    restored.eval()
    with torch.no_grad():
        expected = model.forward_from_features(state, tokens, text, mask)["query"]
        actual = restored.forward_from_features(state, tokens, text, mask)["query"]
    torch.testing.assert_close(actual, expected)


def test_gain_only_refit_learns_negative_scores_for_all_negative_utility() -> None:
    torch.manual_seed(101)
    score_net = ScoreNet(16, 8, 16, dim=8, dropout=0.0)
    batch = _scorer_batch(rows=24)
    teacher = torch.tensor([-0.06, -0.06, -0.04, -0.04]).expand(24, -1).clone()
    batch["teacher_utility"] = teacher
    optimizer = torch.optim.Adam(score_net.parameters(), lr=5e-3)

    with torch.no_grad():
        initial_loss, _ = scorer_gain_loss(score_net, batch, huber_delta=1.0)
    for _ in range(300):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = scorer_gain_loss(score_net, batch, huber_delta=1.0)
        loss.backward()
        optimizer.step()
    score_net.eval()
    with torch.no_grad():
        final_loss, predicted = scorer_gain_loss(score_net, batch, huber_delta=1.0)

    assert final_loss < initial_loss * 0.05
    assert torch.all(predicted < 0)
    assert torch.allclose(predicted.mean(dim=0), teacher[0], atol=0.015)


def test_scorer_cache_integrity_requires_raw_finite_target_free_inputs() -> None:
    batch = _scorer_batch()
    summary = validate_scorer_batch(batch, num_candidates=4)

    assert summary == {"rows": 8, "num_candidates": 4}
    assert set(batch) == {*SCORER_TENSOR_FIELDS, "sample_ids"}
    assert batch["actions"].shape[:2] == (8, 4)
    assert batch["delta"].shape[:2] == (8, 4)
    assert batch["teacher_utility"].shape == (8, 4)
    assert torch.isfinite(batch["teacher_utility"]).all()

    with_target = copy.copy(batch)
    with_target["target_embedding"] = torch.randn(8, 12)
    with pytest.raises(ValueError, match="unexpected=.*target_embedding"):
        validate_scorer_batch(with_target, num_candidates=4)

    nonfinite = copy.copy(batch)
    nonfinite["teacher_utility"] = batch["teacher_utility"].clone()
    nonfinite["teacher_utility"][0, 0] = torch.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        validate_scorer_batch(nonfinite, num_candidates=4)


def test_scorer_cache_manifest_and_shard_integrity(tmp_path) -> None:
    batch = _scorer_batch()
    shard_path = tmp_path / "shard_000000.pt"
    torch.save(batch, shard_path)
    dataset_records = [{"sample_id": value} for value in batch["sample_ids"]]
    teacher_batches = [list(batch["sample_ids"])]
    manifest = {
        "format_version": 1,
        "workflow": "score_net_gain_only_refit",
        "source_checkpoint_sha256": "source-checkpoint",
        "dataset_records": dataset_records,
        "dataset_records_fingerprint": _json_fingerprint(dataset_records),
        "teacher_batches": teacher_batches,
        "teacher_batch_fingerprint": _json_fingerprint(teacher_batches),
        "num_candidates": 4,
        "shards": [
            {
                "path": shard_path.name,
                "rows": len(batch["sample_ids"]),
                "sha256": _sha256_file(shard_path),
            }
        ],
    }
    manifest["cache_fingerprint"] = _json_fingerprint(
        {
            key: value
            for key, value in manifest.items()
            if key not in {"cache_fingerprint", "dataset_records"}
        }
    )
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    loaded_manifest = _load_cache_manifest(tmp_path)
    loaded_batch = _load_shard(tmp_path, loaded_manifest["shards"][0], candidates=4)
    assert loaded_batch["sample_ids"] == batch["sample_ids"]
    torch.testing.assert_close(loaded_batch["teacher_utility"], batch["teacher_utility"])

    corrupted = dict(loaded_manifest)
    corrupted["teacher_batches"] = [["different-sample"]]
    (tmp_path / "manifest.json").write_text(json.dumps(corrupted), encoding="utf-8")
    with pytest.raises(ValueError, match="teacher-batch fingerprint mismatch"):
        _load_cache_manifest(tmp_path)
