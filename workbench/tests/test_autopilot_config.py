from __future__ import annotations

from pathlib import Path

import json

import pytest
import yaml

from workbench.backend import autopilot


def fixture_config(tmp_path: Path, selected="all_registered"):
    dirs = {name: tmp_path / name for name in ("dataset", "artifacts", "checkpoints", "results", "third_party")}
    for directory in dirs.values():
        directory.mkdir()
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text("fake registry", encoding="utf-8")
    config = {
        "schema_version": 1,
        "selected_models": selected,
        "paths": {"registry": str(registry_path), "dataset_root": str(dirs["dataset"]),
                  "artifacts": str(dirs["artifacts"]), "checkpoints": str(dirs["checkpoints"]),
                  "results": str(dirs["results"]), "third_party": str(dirs["third_party"])},
        "budgets": {"max_download_bytes": 1024, "max_models": 20, "top_k": 10},
        "permissions": {key: False for key in autopilot.PERMISSIONS},
    }
    path = tmp_path / "autopilot.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    registry = {"models": [{"model_id": f"model-{index}"} for index in range(3)]}
    return path, config, registry


def test_valid_config_resolves_dynamic_selection_plan(tmp_path):
    path, _, registry = fixture_config(tmp_path)
    registry["models"].extend({"model_id": f"added-{index}"} for index in range(4))
    config = autopilot.load_config(path, registry=registry)
    plan = autopilot.resolved_plan(config, dry_run=True)
    expected = tuple(model["model_id"] for model in registry["models"])
    assert config.selected_model_ids == expected
    assert plan["selected_models"] == list(expected)
    assert [model["model_id"] for model in plan["models"]] == list(expected)
    assert plan["solver_verified"] is False


def test_unknown_config_key_raises(tmp_path):
    path, config, _ = fixture_config(tmp_path)
    config["unexpected"] = True
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown config key"):
        autopilot.load_config(path, registry={"models": [{"model_id": "model-0"}]})


def test_unknown_model_id_raises(tmp_path):
    path, config, registry = fixture_config(tmp_path, ["ghost"])
    with pytest.raises(ValueError, match="unknown model ID.*ghost"):
        autopilot.load_config(path, registry=registry)


def test_dry_run_writes_nothing_and_launches_no_subprocess(tmp_path, monkeypatch):
    import subprocess

    path, _, registry = fixture_config(tmp_path, ["model-0"])
    monkeypatch.setattr(autopilot, "load_registry", lambda path: registry)
    monkeypatch.setattr(autopilot, "_execute", lambda *args, **kwargs: pytest.fail("subprocess orchestration reached"))
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: pytest.fail("subprocess launched"))
    result = autopilot.run(path, apply=False, resume=False, force_models=[], mode="dry-run")
    assert result["summary"]["plan"]["selected_models"] == ["model-0"]
    assert list((tmp_path / "artifacts").glob("**/*")) == []


def test_plan_environments_subprocess_receives_selection_and_source_root(tmp_path, monkeypatch):
    from unittest.mock import patch
    from workbench.scripts import pipeline

    real_reg = autopilot.load_registry(None)
    path, document, _ = fixture_config(tmp_path, ["limn", "csmcir"])
    (tmp_path / "registry.yaml").write_text(Path("workbench/registry/models.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    # Both models must be selectable for this test to observe the whole selection reaching the
    # subprocess: LIMN records no training condition, so the default strict policy would (correctly)
    # block it and only CSMCIR would be left to check.
    document["checkpoint_policy"] = "include_flagged_conditions"
    document["checkpoint_overrides"] = {"csmcir": ["fashioniq"]}
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    config = autopilot.load_config(path, registry=real_reg)
    plan = autopilot.resolved_plan(config, dry_run=True)
    captured_args = []
    monkeypatch.setattr(pipeline, "all_stages", lambda args, config: captured_args.append(args) or [])
    with patch.object(pipeline, "run_stages", return_value=0) as mock_run:
        autopilot._execute(config, plan, resume=False, force_models=[])
        assert captured_args[0].checkpoint_policy == "include_flagged_conditions"
        assert captured_args[0].checkpoint_overrides == {"csmcir": ["fashioniq"]}
        stages = mock_run.call_args[0][0]
        env_plan_stage = stages[0]
        assert env_plan_stage.name == "environment-plan"
        cmd = env_plan_stage.commands[0]
        assert "--third-party-root" in cmd
        assert str(config.paths["third_party"]) in cmd
        assert "--model" in cmd
        model_indices = [idx + 1 for idx, arg in enumerate(cmd) if arg == "--model"]
        assert [cmd[i] for i in model_indices] == ["limn", "csmcir"]


def test_index_failure_does_not_contaminate_model_execution_status(tmp_path):
    reg = {
        "models": [
            {"model_id": "succ", "method_name": "Succ", "native_protocol": "p", "checkpoint_variants": []},
            {"model_id": "fail", "method_name": "Fail", "native_protocol": "p", "checkpoint_variants": []},
            {"model_id": "block", "method_name": "Block", "native_protocol": "p", "checkpoint_variants": []},
        ]
    }
    path, _, _ = fixture_config(tmp_path, ["succ", "fail", "block"])
    config = autopilot.load_config(path, registry=reg)
    plan = {
        "schema_version": 1,
        "configuration_digest": "digest_test",
        "solver_verified": True,
        "environment_plan": {"groups": [{"environment_id": "env", "members": ["succ", "fail", "block"]}]},
        "models": [
            {"model_id": "succ", "status": "NOT_RUN", "blockers": [], "metrics": None},
            {"model_id": "fail", "status": "NOT_RUN", "blockers": [], "metrics": None},
            {"model_id": "block", "status": "BLOCKED_ENVIRONMENT", "blockers": ["conflict"], "metrics": None},
        ],
    }
    execution = {
        "exit_code": 1,
        "stages": {
            "evaluation:succ": {"model_id": "succ", "name": "evaluation", "status": "COMPLETE", "run_id": "r1"},
            "evaluation:fail": {"model_id": "fail", "name": "evaluation", "status": "FAILED", "failure_reason": "crash"},
            "validation-index": {"model_id": None, "name": "validation-index", "status": "FAILED", "failure_reason": "DuckDB err"},
        },
        "missing_permissions": {},
    }
    rows, blockers = autopilot._classify(plan, execution, config)
    result = autopilot._write_reports(config, plan, rows, blockers, execution, planned=False)

    row_map = {r["model_id"]: r for r in rows}
    # The stage says COMPLETE, but no completion proof exists on disk for run `r1`: a stage record is
    # not evidence, so the model must not be reported as a completed evaluation.
    assert row_map["succ"]["status"] == "FAILED_OUTPUT_VALIDATION"
    assert row_map["succ"]["missing_metrics_reason"] == "completion proof missing"
    assert row_map["succ"]["macro_r10"] is None
    assert row_map["fail"]["status"] == "FAILED_EVALUATION"
    assert row_map["block"]["status"] == "BLOCKED_ENVIRONMENT"
    assert result["summary"]["global_index_status"] in ("PARTIAL", "NOT_APPLICABLE", "COMPLETED")


def _config(tmp_path: Path, extra: dict) -> Path:
    registry = tmp_path / "models.yaml"
    registry.write_text(json.dumps({
        "schema_version": 1,
        "protocols": {"fashioniq_original_split": {"label": "Original", "literature_split_label": "original",
                                                   "evaluation_noise_pct": 0}},
        "models": [
            {"model_id": "clean_model", "method_name": "Clean", "source_available": False, "source_dir": None,
             "native_protocol": "fashioniq_original_split",
             "supported_protocols": ["fashioniq_original_split"], "literature_split_label": "original",
             "checkpoint_variants": [{"checkpoint_id": "clean", "filename": "c.pt",
                                      "download_url": "https://example.invalid/c.pt",
                                      "checkpoint_training_noise_pct": 0, "evaluation_noise_pct": 0,
                                      "checkpoint_mapping_status": "VERIFIED_METADATA"}]},
            {"model_id": "noisy_model", "method_name": "Noisy", "source_available": False, "source_dir": None,
             "native_protocol": "fashioniq_original_split",
             "supported_protocols": ["fashioniq_original_split"], "literature_split_label": "original",
             "checkpoint_variants": [{"checkpoint_id": "n50", "filename": "n.pt",
                                      "download_url": "https://example.invalid/n.pt",
                                      "checkpoint_training_noise_pct": 50, "evaluation_noise_pct": 0,
                                      "checkpoint_mapping_status": "VERIFIED_METADATA"}]},
        ],
    }))
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    document = {
        "schema_version": 1, "selected_models": ["clean_model", "noisy_model"],
        "paths": {"registry": str(registry), "dataset_root": str(data),
                  "artifacts": str(tmp_path / "artifacts"),
                  "checkpoints": str(tmp_path / "checkpoints"),
                  "results": str(tmp_path / "results"), "third_party": str(tmp_path / "third_party")},
        "budgets": {"max_download_bytes": 0, "max_models": 16, "top_k": 200},
        "permissions": {"allow_network": False, "allow_large_downloads": False, "allow_env_install": False,
                        "allow_preparation": False, "allow_gpu_eval": False, "allow_index_write": False},
    }
    document.update(extra)
    path = tmp_path / "cfg.yaml"
    path.write_text(json.dumps(document))
    return path


def test_the_checkpoint_policy_defaults_to_strict_clean(tmp_path: Path) -> None:
    plan = autopilot.resolved_plan(autopilot.load_config(_config(tmp_path, {})), dry_run=True)
    assert plan["checkpoint_policy"] == "strict_clean"
    rows = {row["model_id"]: row for row in plan["models"]}
    assert rows["clean_model"]["checkpoint_ids"] == ["clean"]
    assert rows["noisy_model"]["status"] == "BLOCKED_CHECKPOINT"
    assert "50%" in rows["noisy_model"]["blockers"][0]


def test_the_flagged_policy_selects_recorded_noise_checkpoints(tmp_path: Path) -> None:
    plan = autopilot.resolved_plan(autopilot.load_config(_config(tmp_path, {"checkpoint_policy": "include_flagged_conditions"})),
                         dry_run=True)
    rows = {row["model_id"]: row for row in plan["models"]}
    assert rows["noisy_model"]["checkpoint_ids"] == ["n50"]
    assert rows["noisy_model"]["checkpoint_selection"]["strict_clean_checkpoint_available"] is False


def test_an_override_selects_a_specific_variant_and_is_recorded_as_such(tmp_path: Path) -> None:
    plan = autopilot.resolved_plan(autopilot.load_config(_config(tmp_path, {"checkpoint_overrides": {"noisy_model": ["n50"]}})),
                         dry_run=True)
    row = next(row for row in plan["models"] if row["model_id"] == "noisy_model")
    assert row["checkpoint_ids"] == ["n50"]
    assert row["checkpoint_selection"]["policy_overridden"] is True


def test_an_unregistered_policy_or_override_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="checkpoint_policy must be one of"):
        autopilot.load_config(_config(tmp_path, {"checkpoint_policy": "whatever"}))
    with pytest.raises(ValueError, match="unknown model ID"):
        autopilot.load_config(_config(tmp_path, {"checkpoint_overrides": {"not_a_model": ["x"]}}))
    with pytest.raises(ValueError, match="not selected"):
        autopilot.load_config(_config(tmp_path, {"selected_models": ["clean_model"],
                                       "checkpoint_overrides": {"noisy_model": ["n50"]}}))


def test_configuration_digest_includes_checkpoint_policy_and_overrides(tmp_path: Path) -> None:
    from dataclasses import replace

    cfg_clean = autopilot.load_config(_config(tmp_path, {"checkpoint_policy": "strict_clean"}))
    cfg_flagged = replace(cfg_clean, checkpoint_policy="include_flagged_conditions")
    cfg_override = replace(cfg_clean, checkpoint_overrides={"noisy_model": ["n50"]})

    digest_clean = autopilot._configuration_digest(cfg_clean)
    digest_clean_repeat = autopilot._configuration_digest(cfg_clean)
    digest_flagged = autopilot._configuration_digest(cfg_flagged)
    digest_override = autopilot._configuration_digest(cfg_override)

    # Same config produces identical digest repeatedly
    assert digest_clean == digest_clean_repeat
    # Changing policy changes experiment identity
    assert digest_clean != digest_flagged
    # Changing overrides changes experiment identity
    assert digest_clean != digest_override
    assert digest_flagged != digest_override


def _budget_config(tmp_path: Path, *, max_bytes: int, rows: list[dict], registry_models: list[dict],
                   network: bool = True, large: bool = True):
    """A config whose download budget is enforced against a given plan row set."""
    dirs = {name: tmp_path / name for name in ("dataset", "artifacts", "checkpoints", "results", "third_party")}
    for directory in dirs.values():
        directory.mkdir(exist_ok=True)
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text("fake registry", encoding="utf-8")
    config = {
        "schema_version": 1,
        "selected_models": [row["model_id"] for row in rows],
        "paths": {"registry": str(registry_path), "dataset_root": str(dirs["dataset"]),
                  "artifacts": str(dirs["artifacts"]), "checkpoints": str(dirs["checkpoints"]),
                  "results": str(dirs["results"]), "third_party": str(dirs["third_party"])},
        "budgets": {"max_download_bytes": max_bytes, "max_models": 20, "top_k": 10},
        "permissions": {**{key: False for key in autopilot.PERMISSIONS},
                        "allow_network": network, "allow_large_downloads": large},
    }
    path = tmp_path / "autopilot.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return autopilot.load_config(path, registry={"models": registry_models})


def _variant(checkpoint_id: str, size: int, url: str = "https://example.invalid/ckpt.pt") -> dict:
    return {"checkpoint_id": checkpoint_id, "download_url": url, "expected_size_bytes": size,
            "filename": f"{checkpoint_id}.pt"}


def test_the_download_budget_charges_only_the_selected_checkpoints(tmp_path: Path) -> None:
    """A model with several variants must cost the budget exactly what this run selected.

    Charging the first registry variant (or every bundle member) refused runs that would have
    downloaded almost nothing, and mis-reported the reason as a budget violation.
    """
    model = {
        "model_id": "multi",
        "checkpoint_variants": [_variant("small", 100), _variant("huge", 10_000_000)],
        "checkpoint_bundles": [],
    }
    rows = [{"model_id": "multi", "status": "NOT_RUN", "checkpoint_ids": ["small"], "blockers": []}]
    config = _budget_config(tmp_path, max_bytes=1000, rows=rows, registry_models=[model])
    autopilot._enforce_download_budget(config, rows)
    assert rows[0]["status"] == "NOT_RUN", rows[0]

    rows[0]["checkpoint_ids"] = ["huge"]
    autopilot._enforce_download_budget(config, rows)
    assert rows[0]["status"] == "BLOCKED_CHECKPOINT"
    assert "exceeded" in rows[0]["blockers"][0]


def test_the_download_budget_expands_a_selected_bundle_to_its_members(tmp_path: Path) -> None:
    """Selecting a bundle means downloading its members, so the budget must count them."""
    model = {
        "model_id": "bundled",
        "checkpoint_variants": [_variant("a", 400), _variant("b", 400), _variant("other", 10_000_000)],
        "checkpoint_bundles": [{"bundle_id": "bundle-x", "required_checkpoint_ids": ["a", "b"],
                                "structure": "per_category", "mapping_status": "VERIFIED", "notes": ""}],
    }
    rows = [{"model_id": "bundled", "status": "NOT_RUN", "checkpoint_ids": ["bundle-x"], "blockers": []}]
    config = _budget_config(tmp_path, max_bytes=1000, rows=rows, registry_models=[model])
    autopilot._enforce_download_budget(config, rows)
    assert rows[0]["status"] == "NOT_RUN", rows[0]

    # The unrelated third variant is not part of the selected bundle and must not be charged.
    rows[0]["checkpoint_ids"] = ["bundle-x"]
    tight = _budget_config(tmp_path, max_bytes=700, rows=rows, registry_models=[model])
    autopilot._enforce_download_budget(tight, rows)
    assert rows[0]["status"] == "BLOCKED_CHECKPOINT"


def test_an_unknown_selected_checkpoint_size_blocks_rather_than_guessing(tmp_path: Path) -> None:
    model = {"model_id": "unknown-size",
             "checkpoint_variants": [{"checkpoint_id": "x", "download_url": "https://example.invalid/x.pt",
                                      "expected_size_bytes": None, "filename": "x.pt"}],
             "checkpoint_bundles": []}
    rows = [{"model_id": "unknown-size", "status": "NOT_RUN", "checkpoint_ids": ["x"], "blockers": []}]
    config = _budget_config(tmp_path, max_bytes=10_000, rows=rows, registry_models=[model])
    autopilot._enforce_download_budget(config, rows)
    assert rows[0]["status"] == "BLOCKED_CHECKPOINT"
    assert "unknown" in rows[0]["blockers"][0]


def test_offline_blocked_source_acquisition_survives_into_final_rows(tmp_path: Path):
    # The planner caches solve results in a process-global dict keyed by the canonical constraint
    # set, so a failure recorded by an earlier test would make this run be classified
    # BLOCKED_ENVIRONMENT instead of reaching the offline source-acquisition path it is testing.
    from workbench.backend import environment_planner as _planner

    _planner._SOLVE_CACHE.clear()
    dirs = {name: tmp_path / name for name in ("dataset", "artifacts", "checkpoints", "results", "third_party")}
    for d in dirs.values():
        d.mkdir()
    reg_path = tmp_path / "models.yaml"
    reg_path.write_text(Path("workbench/registry/models.yaml").read_text(encoding="utf-8"), encoding="utf-8")

    cfg_data = {
        "schema_version": 1,
        "selected_models": ["csmcir"],
        "paths": {"registry": str(reg_path), "dataset_root": str(dirs["dataset"]),
                  "artifacts": str(dirs["artifacts"]), "checkpoints": str(dirs["checkpoints"]),
                  "results": str(dirs["results"]), "third_party": str(dirs["third_party"])},
        "budgets": {"max_download_bytes": 1024, "max_models": 20, "top_k": 10},
        "permissions": {key: False for key in autopilot.PERMISSIONS},
    }
    cfg_path = tmp_path / "autopilot.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_data), encoding="utf-8")

    result = autopilot.run(cfg_path, apply=True, resume=False, force_models=[], mode="all")

    plan = json.loads((dirs["artifacts"] / "autopilot_execution_plan.json").read_text(encoding="utf-8"))
    assert "source_acquisition" in plan, "plan must retain source_acquisition state"
    assert plan["source_acquisition"]["blocked"], "source_acquisition must record blocked acquisition"

    model_row = next(r for r in result["summary"]["models"] if r["model_id"] == "csmcir")
    assert model_row["status"] == "BLOCKED_SOURCE", f"expected BLOCKED_SOURCE but got {model_row['status']}"
    assert any("pinned source not acquired" in b for b in model_row["blockers"]), "blocker reason must survive into final rows"

    blockers = json.loads((dirs["artifacts"] / "reports" / "blockers.json").read_text(encoding="utf-8"))
    csmcir_blocker = next(b for b in blockers if b["model_id"] == "csmcir")
    assert csmcir_blocker["status"] == "BLOCKED_SOURCE"
