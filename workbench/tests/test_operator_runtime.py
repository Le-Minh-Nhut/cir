from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace



def load_script(name: str):
    path = Path("workbench/scripts") / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_server_command_construction_and_dry_run(tmp_path: Path, capsys) -> None:
    serve = load_script("serve_workbench")
    commands = serve.build_commands(tmp_path, "127.0.0.1", 8100, 5200, True, True, False)
    assert [(command.name, command.argv, command.cwd) for command in commands] == [
        ("backend", ["uvicorn", "workbench.backend.main:app", "--host", "127.0.0.1", "--port", "8100"], tmp_path.parent),
        ("frontend", ["npm", "run", "dev", "--", "--host", "127.0.0.1", "--port", "5200", "--strictPort"], tmp_path / "frontend"),
    ]
    assert serve.main(["--frontend-only", "--production-frontend", "--frontend-port", "5200", "--dry-run"], tmp_path) == 0
    output = capsys.readouterr().out
    assert "npm run preview -- --host 127.0.0.1 --port 5200 --strictPort" in output
    assert "[SKIP] dry-run; no processes started" in output


def test_server_uses_configured_defaults(monkeypatch, tmp_path: Path, capsys) -> None:
    serve = load_script("serve_workbench")
    monkeypatch.setattr(
        serve,
        "resolve_config",
        lambda: SimpleNamespace(WORKBENCH_HOST="0.0.0.0", WORKBENCH_BACKEND_PORT=8100, WORKBENCH_FRONTEND_PORT=5200),
    )

    assert serve.main(["--dry-run"], tmp_path) == 0

    output = capsys.readouterr().out
    assert "uvicorn workbench.backend.main:app --host 0.0.0.0 --port 8100" in output
    assert "npm run dev -- --host 0.0.0.0 --port 5200 --strictPort" in output


def test_server_cli_overrides_configured_defaults(monkeypatch, tmp_path: Path, capsys) -> None:
    serve = load_script("serve_workbench")
    monkeypatch.setattr(
        serve,
        "resolve_config",
        lambda: SimpleNamespace(WORKBENCH_HOST="0.0.0.0", WORKBENCH_BACKEND_PORT=8100, WORKBENCH_FRONTEND_PORT=5200),
    )

    assert serve.main(["--host", "127.0.0.1", "--backend-port", "8300", "--frontend-port", "5300", "--dry-run"], tmp_path) == 0

    output = capsys.readouterr().out
    assert "uvicorn workbench.backend.main:app --host 127.0.0.1 --port 8300" in output
    assert "npm run dev -- --host 127.0.0.1 --port 5300 --strictPort" in output


def test_vite_config_uses_runtime_backend_url_for_dev_and_preview() -> None:
    config = Path("workbench/frontend/vite.config.mts").read_text()

    assert "WORKBENCH_BACKEND_URL" in config
    assert "server: { proxy: { \"/api\": backendUrl } }" in config
    assert "preview: { proxy: { \"/api\": backendUrl } }" in config
    assert ":8000" not in config


def test_server_refuses_missing_frontend_dependencies(tmp_path: Path, capsys) -> None:
    serve = load_script("serve_workbench")
    assert serve.main(["--frontend-only"], tmp_path) == 2
    assert "[BLOCKED] frontend dependencies missing" in capsys.readouterr().err


def test_cleanup_removes_only_supported_generated_outputs(tmp_path: Path, capsys) -> None:
    cleanup = load_script("clean_generated")
    artifacts = tmp_path / "artifacts"
    mock = artifacts / "results" / "fashioniq" / "mock"
    real = artifacts / "results" / "fashioniq" / "real"
    checkpoints = artifacts / "checkpoints"
    data = tmp_path / "data" / "FashionIQ"
    third_party = tmp_path / "third_party" / "upstream"
    for directory in (mock, real, checkpoints, data, third_party, artifacts / "logs"):
        directory.mkdir(parents=True)
    (mock / "run.json").write_text("{}")
    (real / "run.json").write_text("{}")
    (checkpoints / "model.pt").write_text("checkpoint")
    (data / "image.jpg").write_text("data")
    (third_party / "source.py").write_text("source")
    database = artifacts / "workbench.duckdb"
    database.write_text("derived")

    assert cleanup.clean(tmp_path, True, True, True, True) == 0
    assert mock.exists() and database.exists() and (artifacts / "logs").exists()
    assert "[SKIP] dry-run; nothing removed" in capsys.readouterr().out

    assert cleanup.clean(tmp_path, True, True, True, False) == 0
    assert not mock.exists() and not database.exists() and not (artifacts / "logs").exists()
    assert real.exists() and checkpoints.exists() and data.exists() and third_party.exists()
    assert not cleanup.is_safe_target(checkpoints, tmp_path)
    assert not cleanup.is_safe_target(data, tmp_path)
    assert not cleanup.is_safe_target(third_party, tmp_path)
