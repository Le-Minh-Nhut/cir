from pathlib import Path

import yaml

from workbench.backend import autopilot


def test_truncated_summary_returns_explicit_failed_state(tmp_path: Path) -> None:
    directories = {"dataset_root": "dataset", "artifacts": "artifacts", "checkpoints": "checkpoints",
                  "results": "results", "third_party": "third_party"}
    for name in directories.values():
        (tmp_path / name).mkdir()
    registry = Path(__file__).resolve().parents[1] / "registry" / "models.yaml"
    config_data = {
        "schema_version": 1,
        "selected_models": ["tme"],
        "paths": {"registry": str(registry), **{key: str(tmp_path / value) for key, value in directories.items()}},
        "budgets": {key: 100 for key in autopilot.BUDGET_KEYS},
        "permissions": {key: False for key in autopilot.PERMISSIONS},
    }
    config_file = tmp_path / "autopilot.yaml"
    config_file.write_text(yaml.safe_dump(config_data), encoding="utf-8")
    config = autopilot.load_config(config_file)
    paths = autopilot._artifact_paths(config)
    paths["summary_json"].parent.mkdir(parents=True, exist_ok=True)
    paths["summary_json"].write_text('{"exit_code": 0, "summary": ', encoding="utf-8")

    state = autopilot._read_state(config)

    assert state["exit_code"] == 1
    assert state["summary"]["status"] == "FAILED"
    assert "corrupt" in state["summary"]["reason"].lower()
