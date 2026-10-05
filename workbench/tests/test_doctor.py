from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from workbench.backend.operator_config import WorkbenchConfig
from workbench.backend.registry import load_registry


def load_doctor():
    path = Path("workbench/scripts/doctor.py")
    spec = importlib.util.spec_from_file_location("doctor", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config(root: Path) -> WorkbenchConfig:
    return WorkbenchConfig(root, root / "data", root / "FashionIQ", "127.0.0.1", 8000, 5173, root / "checkpoints", root / "results", root / "third_party")


def test_json_output_has_machine_readable_structure(monkeypatch, capsys) -> None:
    doctor = load_doctor()
    monkeypatch.setattr(doctor, "collect", lambda **_: {"overall": "WARNING", "checks": [{"name": "x", "status": "WARNING", "detail": "missing", "evidence": {}}]})

    assert doctor.main(["--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["overall"] == "WARNING"
    assert report["checks"][0] == {"name": "x", "status": "WARNING", "detail": "missing", "evidence": {}}


def test_model_checks_report_missing_csmcir_and_encoder_requirements(tmp_path: Path) -> None:
    doctor = load_doctor()
    models = {model["model_id"]: model for model in load_registry()["models"]}

    csmcir = doctor.model_checks(models["csmcir"], config(tmp_path))
    encoder = doctor.model_checks(models["encoder"], config(tmp_path))
    csmcir_blockers = csmcir[1]["evidence"]["checkpoints"][0]["runtime_blockers"]
    encoder_blockers = encoder[1]["evidence"]["checkpoints"][0]["runtime_blockers"]

    assert any(item.startswith("CSMCIR fixed dataset layout missing: ") for item in csmcir_blockers)
    assert any(str(tmp_path / "third_party" / "CSMCIR" / "fashionIQ_dataset" / "captions") in item for item in csmcir_blockers)
    assert f"ENCODER asset missing: {tmp_path / 'third_party' / 'ENCODER' / 'open_clip_pytorch_model.bin'}" in encoder_blockers


def test_collect_reports_missing_paths(monkeypatch, tmp_path: Path) -> None:
    doctor = load_doctor()
    monkeypatch.setattr(doctor, "load_registry", lambda: {"models": [], "protocols": {}})

    report = doctor.collect(config(tmp_path))
    checks = {item["name"]: item for item in report["checks"]}

    assert checks["fashioniq"]["status"] == "BLOCKED"
    assert checks["fashioniq"]["evidence"]["missing"] == [str(tmp_path / "FashionIQ" / part) for part in ("captions", "image_splits", "images")]
    assert checks["duckdb"]["status"] == "WARNING"
