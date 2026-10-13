from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from types import SimpleNamespace

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
        # `local_state` classifies the checkout with the shared porcelain queries the
        # execution gate uses; all of them report a clean tree here.
        if "status" in command and "--porcelain" in command:
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


def test_list_uses_configured_source_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    sync = load_sync_module()
    root = tmp_path / "configured-third-party"
    inspected: list[Path] = []
    monkeypatch.setattr(sync, "load_registry", lambda: {"models": [model()]})
    monkeypatch.setattr(sync, "resolve_config", lambda: SimpleNamespace(WORKBENCH_THIRD_PARTY_ROOT=root))
    monkeypatch.setattr(sync, "local_state", lambda path: inspected.append(path) or ("missing", None, None))
    monkeypatch.setattr(sys, "argv", ["sync_upstreams.py", "--list"])

    sync.main()

    output = capsys.readouterr().out
    assert inspected == [root / "Demo"]
    assert "local=missing" in output
    assert "[INFO] demo: pinned SHA unavailable locally" in output


def test_verify_only_fails_missing_clone(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    sync = load_sync_module()
    monkeypatch.setattr(sync, "load_registry", lambda: {"models": [model()]})
    monkeypatch.setattr(sync, "resolve_config", lambda: SimpleNamespace(WORKBENCH_THIRD_PARTY_ROOT=tmp_path))
    monkeypatch.setattr(sys, "argv", ["sync_upstreams.py", "--model", "demo", "--verify-only"])

    with pytest.raises(SystemExit, match="1"):
        sync.main()


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


def _make_scratch_git_repo(path: Path) -> tuple[str, str]:
    import subprocess
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=path, check=True)
    (path / "file.txt").write_text("initial")
    subprocess.run(["git", "add", "."], cwd=path, check=True)
    subprocess.run(["git", "commit", "-qm", "pin_commit"], cwd=path, check=True)
    pin_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()
    (path / "file.txt").write_text("updated")
    subprocess.run(["git", "commit", "-qam", "other_commit"], cwd=path, check=True)
    other_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()
    return pin_sha, other_sha


def test_clean_checkout_at_wrong_sha_fails_without_update_existing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sync = load_sync_module()
    repo = tmp_path / "Demo"
    pin_sha, other_sha = _make_scratch_git_repo(repo)
    m = model()
    m["upstream_commit_sha"] = pin_sha
    monkeypatch.setattr(sync, "load_registry", lambda: {"models": [m]})
    monkeypatch.setattr(sys, "argv", ["sync_upstreams.py", "--model", "demo", "--output-root", str(tmp_path)])

    with pytest.raises(SystemExit, match="1"):
        sync.main()

    output = capsys.readouterr().out
    assert "BLOCKED_SOURCE_PIN_MISMATCH" in output
    assert f"wanted {pin_sha} actual {other_sha}" in output


def test_clean_checkout_at_wrong_sha_reaches_pin_with_update_existing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import subprocess
    sync = load_sync_module()
    repo = tmp_path / "Demo"
    pin_sha, _ = _make_scratch_git_repo(repo)
    m = model()
    m["upstream_commit_sha"] = pin_sha
    monkeypatch.setattr(sync, "load_registry", lambda: {"models": [m]})
    monkeypatch.setattr(
        sys, "argv", ["sync_upstreams.py", "--model", "demo", "--update-existing", "--output-root", str(tmp_path)]
    )

    sync.main()

    output = capsys.readouterr().out
    assert "[OK] demo: pinned SHA checked out" in output
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    assert actual == pin_sha


def test_dirty_checkout_is_refused_and_unmodified(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sync = load_sync_module()
    repo = tmp_path / "Demo"
    pin_sha, _ = _make_scratch_git_repo(repo)
    (repo / "file.txt").write_text("uncommitted dirty change")
    m = model()
    m["upstream_commit_sha"] = pin_sha
    monkeypatch.setattr(sync, "load_registry", lambda: {"models": [m]})
    monkeypatch.setattr(
        sys, "argv", ["sync_upstreams.py", "--model", "demo", "--update-existing", "--output-root", str(tmp_path)]
    )

    with pytest.raises(SystemExit, match="1"):
        sync.main()

    output = capsys.readouterr().out
    assert "[BLOCKED] demo: repository is dirty; refusing to change it" in output
    assert (repo / "file.txt").read_text() == "uncommitted dirty change"
