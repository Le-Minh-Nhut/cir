from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
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


def test_preflight_explicit_dataset_root_honours_canonical_link_and_auxiliary_assets(tmp_path: Path) -> None:
    config = _config(tmp_path)
    model = next(m for m in load_registry()["models"] if m["model_id"] == "csmcir")
    source = config.WORKBENCH_THIRD_PARTY_ROOT / model["source_dir"]
    source.mkdir(parents=True)

    explicit_root = tmp_path / "explicit_fashioniq"
    explicit_root.mkdir()

    link = source / model["canonical_dataset_link"]
    link.symlink_to(explicit_root, target_is_directory=True)

    reqs = doctor.model_requirements(model, config, dataset_root=explicit_root, canonical_root=explicit_root)
    link_blocks = [b for b in reqs["blocks"] if "canonical dataset link" in b]
    assert not link_blocks, f"canonical link resolving to explicit root must not block: {link_blocks}"

    qwen_assets = [a for a in reqs["evidence"]["auxiliary_assets"] if a["name"] == "Qwen captions"]
    assert qwen_assets
    for asset in qwen_assets:
        assert asset["path"].startswith(str(explicit_root))

    qwen_blocks = [b for b in reqs["blocks"] if "Qwen captions" in b]
    assert qwen_blocks
    for block in qwen_blocks:
        assert str(explicit_root) in block

    checkpoint = model["checkpoint_variants"][0]
    evaluator_reasons = evaluate_models.blockers(
        model, checkpoint, model.get("native_protocol"), link, replace(config, FASHIONIQ_ROOT=explicit_root)
    )[3]
    assert not any("canonical dataset link" in b for b in evaluator_reasons)

    link.unlink()
    wrong_root = tmp_path / "wrong_fashioniq"
    wrong_root.mkdir()
    link.symlink_to(wrong_root, target_is_directory=True)

    reqs_wrong = doctor.model_requirements(model, config, dataset_root=explicit_root, canonical_root=explicit_root)
    assert f"canonical dataset link resolves to {wrong_root}, expected {explicit_root}" in reqs_wrong["blocks"]
    evaluator_wrong = evaluate_models.blockers(
        model, checkpoint, model.get("native_protocol"), link, replace(config, FASHIONIQ_ROOT=explicit_root)
    )[3]
    assert f"canonical dataset link resolves to {wrong_root}, expected {explicit_root}" in evaluator_wrong


def test_evaluator_tautology_regression_blocks_symlink_resolving_elsewhere(tmp_path: Path) -> None:
    config = _config(tmp_path)
    model = next(m for m in load_registry()["models"] if m["model_id"] == "csmcir")
    canonical_root = tmp_path / "canonical_fashioniq"
    canonical_root.mkdir()
    wrong_root = tmp_path / "wrong_fashioniq"
    wrong_root.mkdir()

    source = config.WORKBENCH_THIRD_PARTY_ROOT / model["source_dir"]
    source.mkdir(parents=True)
    link = source / model["canonical_dataset_link"]
    link.symlink_to(wrong_root, target_is_directory=True)

    eval_config = replace(config, FASHIONIQ_ROOT=canonical_root)
    blocks_wrong = preflight_blockers(model, eval_config, dataset_root=link)
    assert f"canonical dataset link resolves to {wrong_root}, expected {canonical_root}" in blocks_wrong

    link.unlink()
    link.symlink_to(canonical_root, target_is_directory=True)
    blocks_correct = preflight_blockers(model, eval_config, dataset_root=link)
    assert not any("canonical dataset link" in b for b in blocks_correct)
