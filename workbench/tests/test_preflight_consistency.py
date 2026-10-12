from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from workbench.backend import runtime
from workbench.backend.operator_config import WorkbenchConfig
from workbench.backend.preflight import preflight_blockers
from workbench.backend.registry import checkpoint_path, load_registry
from workbench.scripts import doctor, evaluate_models, prepare_model


def _config(root: Path) -> WorkbenchConfig:
    return WorkbenchConfig(root, root / "data", root / "FashionIQ", "127.0.0.1", 8000, 5173,
                           root / "checkpoints", root / "results", root / "third_party")


def test_doctor_prepare_and_runtime_share_dataset_and_model_blockers(monkeypatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    registry = load_registry()
    models = {model["model_id"]: model for model in registry["models"]}
    monkeypatch.setattr(runtime, "source_clean_and_pinned", lambda *_: (True, "pin", None))
    monkeypatch.setattr(runtime, "environment_blockers", lambda *_: [])
    monkeypatch.setattr(doctor, "environment_blockers", lambda *_: [])
    monkeypatch.setattr(doctor, "environment_status", lambda *_args, **_kwargs: {
        "status": "RUNTIME_READY", "tier": "RUNTIME_READY", "variable": None,
        "interpreter": "/python", "verified": True, "gpu_deferred": False,
        "reason": None, "packages": {},
    })

    for model_id in ("dcnet", "hint", "csmcir"):
        model = deepcopy(models[model_id])
        source = config.WORKBENCH_THIRD_PARTY_ROOT / model["source_dir"]
        source.mkdir(parents=True)
        adapter = doctor.ADAPTERS.get(model_id)
        if adapter and getattr(adapter, "script", None):
            evaluator = source / adapter.script
            evaluator.parent.mkdir(parents=True, exist_ok=True)
            evaluator.write_text("", encoding="utf-8")

        checkpoint = deepcopy(model["checkpoint_variants"][0])
        checkpoint["checkpoint_mapping_status"] = "VERIFIED_METADATA"
        checkpoint["expected_sha256"] = None
        model["checkpoint_variants"] = [checkpoint]
        model["checkpoint_bundles"] = []
        artifact = checkpoint_path(model_id, checkpoint, config.WORKBENCH_CHECKPOINT_ROOT)
        artifact.parent.mkdir(parents=True, exist_ok=True)
        if checkpoint.get("artifact_type") in {"run_directory", "bundle_directory", "bundle_placeholder"}:
            artifact.mkdir()
            for filename in checkpoint.get("required_files", []):
                member = artifact / filename
                member.parent.mkdir(parents=True, exist_ok=True)
                member.write_bytes(b"fixture")
        else:
            artifact.write_bytes(b"fixture")

        doctor_checks = {check["name"]: check for check in doctor.model_checks(model, config)}
        doctor_blockers = doctor_checks[f"runtime:{model_id}"]["evidence"]["checkpoints"][0]["runtime_blockers"]
        prepare_blockers = prepare_model.status(model, config)["blockers"]
        runtime_blocker_set = runtime.runtime_blockers(model, checkpoint, config)
        evaluator_blockers = evaluate_models.blockers(
            model, checkpoint, model.get("native_protocol"), None, config)[3]
        assert set(doctor_blockers) == set(prepare_blockers) == set(runtime_blocker_set) == set(evaluator_blockers)

        assert set(prepare_blockers) == set(preflight_blockers(model, config))
        if model_id == "dcnet":
            assert model.get("preparation_id")
        if model_id == "hint":
            assert model.get("preparation_id") is None
        if model_id == "csmcir":
            assert model["canonical_dataset_link"]
