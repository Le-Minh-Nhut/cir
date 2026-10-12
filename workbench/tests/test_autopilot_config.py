from __future__ import annotations

from pathlib import Path

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
