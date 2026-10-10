"""Workflow report must be produced for SUCCESS, PARTIAL, FAILED, and BLOCKED terminations."""
from __future__ import annotations

import json
from pathlib import Path

import workbench.scripts.pipeline as pipeline
from workbench.backend.operator_config import WorkbenchConfig


def _cfg(tmp_path: Path) -> WorkbenchConfig:
    data = tmp_path / "FashionIQ"
    (data / "captions").mkdir(parents=True)
    (data / "image_splits").mkdir()
    (data / "images").mkdir()
    for cat in ("dress", "shirt", "toptee"):
        (data / "image_splits" / f"split.{cat}.val.json").write_text("[]")
        (data / "captions" / f"cap.{cat}.val.json").write_text("[]")
    return WorkbenchConfig(tmp_path / "repo", tmp_path / "d", data, "127.0.0.1", 8000, 5173,
                           tmp_path / "ck", tmp_path / "res", tmp_path / "tp")


def test_report_written_when_blocked(tmp_path: Path, monkeypatch):
    cfg = _cfg(tmp_path)
    report = tmp_path / "blocked.json"
    monkeypatch.setattr(pipeline, "resolve_config", lambda: cfg)
    monkeypatch.setattr(pipeline, "run_command", lambda argv: 0)
    rc = pipeline.main(["all", "--model", "csmcir", "--dataset-root", str(cfg.FASHIONIQ_ROOT), "--apply", "--report", str(report)])
    assert rc == 1
    data = json.loads(report.read_text())
    assert data["success"] is False
    assert data["status"] in ("BLOCKED", "PARTIAL", "FAILED")
    assert any(s.get("status") == "BLOCKED_AUTHORIZATION_REQUIRED" for s in data["stages"].values())


def test_report_written_when_failed(tmp_path: Path, monkeypatch):
    cfg = _cfg(tmp_path)
    report = tmp_path / "failed.json"
    monkeypatch.setattr(pipeline, "resolve_config", lambda: cfg)
    monkeypatch.setattr(pipeline, "run_command", lambda argv: 1)
    pipeline.main(["all", "--model", "csmcir", "--dataset-root", str(cfg.FASHIONIQ_ROOT),
                        "--apply", "--allow-network", "--allow-large-downloads", "--allow-env-install",
                        "--allow-preparation", "--allow-gpu-eval", "--report", str(report)])
    assert report.is_file()
    data = json.loads(report.read_text())
    assert data["success"] is False
    assert any(s.get("status") == "FAILED" for s in data["stages"].values())


def test_report_written_on_success(tmp_path: Path):
    report = tmp_path / "ok.json"
    state = tmp_path / "s.json"
    stages = [pipeline.Stage(name="doctor", commands=(["/bin/true"],))]
    rc = pipeline.run_stages(stages, dry_run=False, report_path=report, state_path=state, mode="mock")
    assert rc == 0
    data = json.loads(report.read_text())
    assert data["success"] is True
    assert data["status"] == "SUCCESS"
