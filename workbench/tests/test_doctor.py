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


def test_csmcir_real_checks_require_dataset_qwen_and_cot(tmp_path: Path) -> None:
    doctor = load_doctor()
    model = next(model for model in load_registry()["models"] if model["model_id"] == "csmcir")
    settings = config(tmp_path)
    for path in doctor.fashioniq_base_paths(settings.FASHIONIQ_ROOT):
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("[]")
        else:
            path.mkdir(parents=True, exist_ok=True)
    source = settings.WORKBENCH_THIRD_PARTY_ROOT / "CSMCIR"
    source.mkdir(parents=True)
    (source / "fashionIQ_dataset").symlink_to(settings.FASHIONIQ_ROOT, target_is_directory=True)

    checks = {check["name"]: check for check in doctor.model_checks(model, settings)}
    blockers = checks["runtime:csmcir"]["evidence"]["checkpoints"][0]["runtime_blockers"]
    missing_qwen = str(settings.FASHIONIQ_ROOT / "qwen_captions" / "dress_cot_val.json")
    assert f"CSMCIR Qwen captions missing: {missing_qwen}" in blockers
    assert checks["csmcir:base-dataset"]["status"] == "OK"
    assert checks["csmcir:qwen-captions"]["status"] == "BLOCKED"
    assert missing_qwen in checks["csmcir:qwen-captions"]["evidence"]["missing"]

    for path in doctor.csmcir_auxiliary_paths(settings, "Qwen captions"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("[]")
    checks = {check["name"]: check for check in doctor.model_checks(model, settings)}
    blockers = checks["runtime:csmcir"]["evidence"]["checkpoints"][0]["runtime_blockers"]
    missing_cot = str(settings.WORKBENCH_THIRD_PARTY_ROOT / "CSMCIR" / "COT_ours2" / "fashioniq" / "dress_cot_val.json")
    assert f"CSMCIR COT_ours2 captions missing: {missing_cot}" in blockers
    assert "required COT_ours2 captions have no verified automatic acquisition source" in blockers
    assert checks["csmcir:qwen-captions"]["status"] == "OK"
    assert checks["csmcir:cot-captions"]["status"] == "BLOCKED"


def test_ilearn_native_layout_requires_local_unverified_correction_files(tmp_path: Path) -> None:
    doctor = load_doctor()
    model = next(model for model in load_registry()["models"] if model["model_id"] == "hint")
    settings = config(tmp_path)

    checks = {check["name"]: check for check in doctor.model_checks(model, settings)}
    blockers = checks["runtime:hint"]["evidence"]["checkpoints"][0]["runtime_blockers"]
    missing = str(settings.FASHIONIQ_ROOT / "captions" / "correction_dict_dress.json")

    assert f"FashionIQ fashioniq_ilearn_resized requirement missing: {missing}" in blockers


def test_ilearn_present_correction_files_only_clear_presence_blockers(tmp_path: Path) -> None:
    doctor = load_doctor()
    model = next(model for model in load_registry()["models"] if model["model_id"] == "hint")
    settings = config(tmp_path)
    for path in doctor.fashioniq_required_paths(model, settings.FASHIONIQ_ROOT):
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}" if path.name.startswith("correction_dict_") else "[]")
        else:
            path.mkdir(parents=True, exist_ok=True)

    checks = {check["name"]: check for check in doctor.model_checks(model, settings)}
    blockers = checks["runtime:hint"]["evidence"]["checkpoints"][0]["runtime_blockers"]

    assert not any("fashioniq_ilearn_resized requirement missing" in blocker for blocker in blockers)
    assert "checkpoint missing:" in blockers[0] or "source missing:" in blockers[0]



def test_dcnet_reports_generated_caption_pkl_blocker(tmp_path: Path) -> None:
    doctor = load_doctor()
    model = next(model for model in load_registry()["models"] if model["model_id"] == "dcnet")

    checks = {check["name"]: check for check in doctor.model_checks(model, config(tmp_path))}
    blockers = checks["runtime:dcnet"]["evidence"]["checkpoints"][0]["runtime_blockers"]

    assert f"FashionIQ fashioniq_dcnet requirement missing: {tmp_path / 'FashionIQ' / 'captions'}" in blockers
    assert "checkpoint mapping unresolved" in blockers

def test_workbench_scope_ignores_missing_fashioniq(monkeypatch, tmp_path: Path) -> None:
    doctor = load_doctor()
    monkeypatch.setattr(doctor, "git", lambda *_: None)
    monkeypatch.setattr(doctor.importlib.util, "find_spec", lambda _: object())

    report = doctor.collect(config(tmp_path))

    assert report["overall"] == "READY"
    assert "fashioniq" not in {item["name"] for item in report["checks"]}


def test_real_scope_reports_missing_fashioniq_base(monkeypatch, tmp_path: Path) -> None:
    doctor = load_doctor()
    monkeypatch.setattr(doctor, "load_registry", lambda: {"models": [], "protocols": {}})

    report = doctor.collect(config(tmp_path), scope="real")
    checks = {item["name"]: item for item in report["checks"]}

    assert checks["fashioniq"]["status"] == "BLOCKED"
    assert checks["fashioniq"]["evidence"]["missing"][:3] == [str(tmp_path / "FashionIQ" / part) for part in ("captions", "image_splits", "images")]
