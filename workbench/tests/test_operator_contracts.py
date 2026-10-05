from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from workbench.scripts.validate_results import validate_file
from workbench.tests.mock_data import build_mock_runs


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_contracts", SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_doctor_selected_model_json_contract(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    doctor = load_script("doctor")
    report = {
        "overall": "BLOCKED",
        "checks": [{"name": "runtime:encoder", "status": "BLOCKED", "detail": "official command blocked", "evidence": {"checkpoints": []}}],
    }
    selected: dict[str, object] = {}

    def collect(*, model_id: str | None = None, all_models: bool = False, scope: str = "workbench", serving: bool = False) -> dict:
        selected.update(model_id=model_id, all_models=all_models, scope=scope, serving=serving)
        return report

    monkeypatch.setattr(doctor, "collect", collect)

    assert doctor.main(["--model", "encoder", "--json"]) == 2
    assert selected == {"model_id": "encoder", "all_models": False, "scope": "workbench", "serving": False}
    assert json.loads(capsys.readouterr().out) == report


def test_doctor_reports_csmcir_and_encoder_runtime_blockers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    doctor = load_script("doctor")
    from workbench.backend.operator_config import WorkbenchConfig
    from workbench.backend.registry import load_registry

    config = WorkbenchConfig(tmp_path, tmp_path / "data", tmp_path / "FashionIQ", "127.0.0.1", 8000, 5173, tmp_path / "checkpoints", tmp_path / "results", tmp_path / "third_party")
    models = {model["model_id"]: model for model in load_registry()["models"]}
    monkeypatch.setattr(doctor, "git", lambda *_: None)

    csmcir = doctor.model_checks(models["csmcir"], config)[1]["evidence"]["checkpoints"][0]
    encoder = doctor.model_checks(models["encoder"], config)[1]["evidence"]["checkpoints"][0]

    assert csmcir["command_ready"] is False
    assert any(blocker.startswith("CSMCIR fixed dataset layout missing:") for blocker in csmcir["runtime_blockers"])
    assert encoder["command_ready"] is False
    assert f"ENCODER asset missing: {tmp_path / 'third_party' / 'ENCODER' / 'open_clip_pytorch_model.bin'}" in encoder["runtime_blockers"]


def test_strict_real_validation_rejects_missing_provenance(tmp_path: Path) -> None:
    artifact = tmp_path / "real.json"
    payload = build_mock_runs()[0].model_dump(mode="json")
    payload["run"]["data_kind"] = "experiment"
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    output: list[str] = []

    assert not validate_file(artifact, strict_real=True, output=output.append)
    assert output[0] == "[BLOCKED] mock-habit-fiq_n02: missing provenance: upstream_commit, checkpoint_sha256, command_digest, environment_digest"


def test_sync_dry_run_and_verify_only_output_contract(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    sync = load_script("sync_upstreams")
    model = {"model_id": "demo", "source_available": True, "source_dir": "Demo", "upstream_repo_url": "https://example.invalid/demo.git", "upstream_commit_sha": "a" * 40}
    monkeypatch.setattr(sync, "load_registry", lambda: {"models": [model]})
    monkeypatch.setattr(sync.subprocess, "run", lambda *args, **kwargs: pytest.fail("git must not run"))

    monkeypatch.setattr(sys, "argv", ["sync_upstreams.py", "--model", "demo", "--dry-run", "--output-root", str(tmp_path)])
    sync.main()
    assert capsys.readouterr().out.splitlines() == [
        f"demo: url=https://example.invalid/demo.git source_dir=Demo pin={'a' * 40} local=missing",
        f"[RUN] demo: would clone and checkout at {'a' * 40}",
    ]

    monkeypatch.setattr(sys, "argv", ["sync_upstreams.py", "--model", "demo", "--verify-only", "--output-root", str(tmp_path)])
    with pytest.raises(SystemExit, match="1"):
        sync.main()
    assert capsys.readouterr().out.splitlines() == [
        f"demo: url=https://example.invalid/demo.git source_dir=Demo pin={'a' * 40} local=missing",
        "[BLOCKED] demo: pinned SHA unavailable locally",
    ]


@pytest.mark.parametrize("arguments, expected", [(["--list", "--dry-run"], 0), (["--all-runnable", "--dry-run"], 1)])
def test_evaluation_listing_and_dry_run_never_execute_models(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, arguments: list[str], expected: int) -> None:
    evaluate = load_script("evaluate_models")
    monkeypatch.setattr(evaluate, "pinned_revision", lambda _: None)
    monkeypatch.setattr(evaluate.subprocess, "run", lambda *args, **kwargs: pytest.fail("model command must not run"))

    assert evaluate.main(arguments) == expected


@pytest.mark.parametrize("relative", ["artifacts/checkpoints", "data/FashionIQ", "third_party/upstream"])
def test_cleanup_refuses_protected_checkpoint_dataset_and_source_targets(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str], relative: str) -> None:
    cleanup = load_script("clean_generated")
    protected = tmp_path / relative
    protected.mkdir(parents=True)
    (protected / "keep").write_text("protected", encoding="utf-8")
    monkeypatch.setattr(cleanup, "cleanup_targets", lambda *_: [protected])

    assert cleanup.clean(tmp_path, True, False, False, False) == 2
    assert protected.exists() and (protected / "keep").is_file()
    assert "[BLOCKED] refusing unsafe path:" in capsys.readouterr().err
