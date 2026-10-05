from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "sync_upstreams.py"


def load_sync_module():
    spec = importlib.util.spec_from_file_location("sync_upstreams_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def model() -> dict:
    return {
        "model_id": "demo",
        "source_available": True,
        "source_dir": "Demo",
        "upstream_repo_url": "https://example.invalid/demo.git",
        "upstream_commit_sha": "a" * 40,
    }


def test_dry_run_never_calls_git(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    sync = load_sync_module()
    monkeypatch.setattr(sync, "load_registry", lambda: {"models": [model()]})
    monkeypatch.setattr(sync.subprocess, "run", lambda *args, **kwargs: pytest.fail("Git called during dry run"))
    monkeypatch.setattr(sys, "argv", ["sync_upstreams.py", "--model", "demo", "--dry-run", "--output-root", str(tmp_path)])

    sync.main()

    assert not (tmp_path / "Demo").exists()
    assert "[RUN] demo: would clone and checkout" in capsys.readouterr().out


def test_verify_only_never_calls_network(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    sync = load_sync_module()
    monkeypatch.setattr(sync, "load_registry", lambda: {"models": [model()]})
    calls: list[tuple[str, ...]] = []

    def fake_run(command, **kwargs):
        calls.append(tuple(command))
        if command[-2:] == ["rev-parse", "HEAD"]:
            return type("Result", (), {"returncode": 0, "stdout": "b" * 40})()
        if command[-2:] == ["status", "--porcelain"]:
            return type("Result", (), {"returncode": 0, "stdout": ""})()
        pytest.fail(f"unexpected Git command: {command}")

    destination = tmp_path / "Demo"
    destination.mkdir()
    monkeypatch.setattr(sync.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["sync_upstreams.py", "--model", "demo", "--verify-only", "--output-root", str(tmp_path)])

    with pytest.raises(SystemExit, match="1"):
        sync.main()

    assert all("fetch" not in command and "clone" not in command and "checkout" not in command for command in calls)
    assert "[WARN] demo: local SHA" in capsys.readouterr().out


@pytest.mark.parametrize("arguments", [("--list",), ("--all", "--dry-run"), ("--all", "--verify-only")])
def test_ptha_unavailable_source_skips_git_status(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], arguments: tuple[str, ...]
) -> None:
    sync = load_sync_module()
    monkeypatch.setattr(sync, "load_registry", lambda: {"models": [{"model_id": "ptha_mtst", "source_available": False, "source_dir": None}]})
    monkeypatch.setattr(sync, "local_state", lambda path: pytest.fail(f"Git status checked {path}"))
    monkeypatch.setattr(sys, "argv", ["sync_upstreams.py", *arguments])

    sync.main()

    output = capsys.readouterr().out
    assert "local=unavailable" in output
    assert "local=dirty" not in output
    assert "[SKIP] ptha_mtst: source unavailable" in output
