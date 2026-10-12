from __future__ import annotations

import os
from pathlib import Path

import pytest

from workbench.backend import registry
from workbench.backend.operator_config import WorkbenchConfig
from workbench.scripts import pipeline


def _workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: dict):
    repo, data, checkpoints, third_party, results = (
        tmp_path / name for name in ("repo", "FashionIQ", "checkpoints", "third_party", "results"))
    for directory in (repo, data, checkpoints, third_party, results):
        directory.mkdir()
    (repo / "workbench" / "replay").mkdir(parents=True)
    (repo / "workbench" / "backend" / "adapters").mkdir(parents=True)
    (repo / "workbench" / "scripts").mkdir(parents=True)
    (repo / "workbench" / "replay" / "__init__.py").write_text("\n")
    (repo / "workbench" / "backend" / "adapters" / "__init__.py").write_text("\n")
    (repo / "workbench" / "replay" / "implementation.py").write_text("replay = 1\n")
    (repo / "workbench" / "backend" / "adapters" / "models.py").write_text("adapter = 1\n")
    (repo / "workbench" / "scripts" / "evaluate_models.py").write_text("operator = 1\n")
    monkeypatch.setattr(pipeline, "REPOSITORY", repo)
    monkeypatch.setattr(registry, "model_by_id", lambda model_id: model)
    checkpoint = checkpoints / "model.pt"
    checkpoint.write_bytes(b"weights")
    (data / "captions.json").write_text("captions")
    config = WorkbenchConfig(
        CIR_REPO_ROOT=repo, CIR_DATA_ROOT=tmp_path, FASHIONIQ_ROOT=data,
        WORKBENCH_HOST="127.0.0.1", WORKBENCH_BACKEND_PORT=8000,
        WORKBENCH_FRONTEND_PORT=5173, WORKBENCH_CHECKPOINT_ROOT=checkpoints,
        WORKBENCH_RESULTS_ROOT=results, WORKBENCH_THIRD_PARTY_ROOT=third_party)
    stage = pipeline.Stage(
        "evaluation", commands=(("synthetic-evaluator",),), model_id=model["model_id"],
        input_paths=(checkpoint, data), required_capabilities=frozenset({"allow_gpu_eval"}),
        evidence=lambda record: (True, None, None))
    return config, stage


def _run(stage: pipeline.Stage, config: WorkbenchConfig, state: Path, monkeypatch,
         calls: list[list[str]], *, resume: bool) -> None:
    monkeypatch.setattr(pipeline, "run_command", lambda argv: calls.append(argv) or 0)
    assert pipeline.run_stages(
        [stage], apply=True, authorized_capabilities={"allow_gpu_eval"}, resume=resume,
        config=config, state_path=state) == 0


@pytest.mark.parametrize(
    ("relative", "old", "new"),
    [
        ("workbench/replay/implementation.py", "replay = 1\n", "replay = 2\n"),
        ("workbench/backend/adapters/models.py", "adapter = 1\n", "adapter = 2\n"),
        ("workbench/scripts/evaluate_models.py", "operator = 1\n", "operator = 2\n"),
    ],
)
def test_code_identity_invalidates_and_unchanged_evaluation_resumes(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative: str,
        old: str, new: str) -> None:
    model = {"model_id": "identity_test", "source_dir": None, "environment_required": False}
    config, stage = _workspace(tmp_path, monkeypatch, model)
    code = config.CIR_REPO_ROOT / relative
    code.parent.mkdir(parents=True, exist_ok=True)
    code.write_text(old)
    before = pipeline.stage_input_fingerprint_info(stage, config)
    unrelated = config.CIR_REPO_ROOT / "unrelated.txt"
    unrelated.write_text("not part of evaluator code")
    assert pipeline.stage_input_fingerprint_info(stage, config)["digest"] == before["digest"]
    assert before["weak"] is False

    calls: list[list[str]] = []
    state = tmp_path / "state.json"
    _run(stage, config, state, monkeypatch, calls, resume=False)
    _run(stage, config, state, monkeypatch, calls, resume=True)
    assert len(calls) == 1, "unchanged execution identity must resume"

    original_mtime = code.stat().st_mtime_ns
    code.write_text(new)
    os.utime(code, ns=(original_mtime, original_mtime))
    after = pipeline.stage_input_fingerprint_info(stage, config)
    assert after["digest"] != before["digest"], "same-mtime source rewrite must invalidate fingerprint"
    assert after["weak"] is False, "derived execution identity must not weaken verified inputs"
    _run(stage, config, state, monkeypatch, calls, resume=True)
    assert len(calls) == 2, "changed code must rerun evaluation"


def test_environment_lock_and_interpreter_content_change_execution_identity(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = {"model_id": "locked_identity", "source_dir": "Model", "environment_required": True,
             "environment": {"python_env_var": "EXECUTION_TEST_PYTHON"}}
    config, stage = _workspace(tmp_path, monkeypatch, model)
    source = config.WORKBENCH_THIRD_PARTY_ROOT / "Model"
    source.mkdir()
    lock = source / "requirements.txt"
    lock.write_text("dependency==1\n")
    interpreter = tmp_path / "model-python"
    interpreter.write_bytes(b"interpreter-v1")
    monkeypatch.setenv("EXECUTION_TEST_PYTHON", str(interpreter))
    before = pipeline.stage_input_fingerprint_info(stage, config)["digest"]

    lock.write_text("dependency==2\n")
    after_lock = pipeline.stage_input_fingerprint_info(stage, config)["digest"]
    assert after_lock != before
    interpreter.write_bytes(b"interpreter-v2")
    after_interpreter = pipeline.stage_input_fingerprint_info(stage, config)["digest"]
    assert after_interpreter != after_lock


def test_approved_stable_dataset_symlink_is_verified_and_retarget_invalidates(
        tmp_path: Path) -> None:
    canonical_root = tmp_path / "datasets"
    first_data = canonical_root / "FashionIQ"
    second_data = canonical_root / "OtherFashionIQ"
    for root, content in ((first_data, b"first dataset"), (second_data, b"second dataset")):
        root.mkdir(parents=True)
        (root / "image.bin").write_bytes(content)
    third_party = tmp_path / "third_party"
    source = third_party / "Model"
    source.mkdir(parents=True)
    link = source / "fashionIQ_dataset"
    link.symlink_to(first_data, target_is_directory=True)
    config = WorkbenchConfig(
        CIR_REPO_ROOT=tmp_path, CIR_DATA_ROOT=canonical_root, FASHIONIQ_ROOT=canonical_root,
        WORKBENCH_HOST="127.0.0.1", WORKBENCH_BACKEND_PORT=8000,
        WORKBENCH_FRONTEND_PORT=5173, WORKBENCH_CHECKPOINT_ROOT=tmp_path / "checkpoints",
        WORKBENCH_RESULTS_ROOT=tmp_path / "results", WORKBENCH_THIRD_PARTY_ROOT=third_party)
    stage = pipeline.Stage("evaluation", input_paths=(link,))

    before = pipeline.stage_input_fingerprint_info(stage, config)
    assert before["weak"] is False
    assert before["inputs"][0]["category"] == pipeline.SYMLINK_VERIFIED
    assert before["inputs"][0]["symlink_target"] == str(first_data)
    assert before["inputs"][0]["target_tree_digest"]

    original_mtime = link.lstat().st_mtime_ns
    link.unlink()
    link.symlink_to(second_data, target_is_directory=True)
    os.utime(link, ns=(original_mtime, original_mtime), follow_symlinks=False)
    after = pipeline.stage_input_fingerprint_info(stage, config)
    assert after["weak"] is False
    assert after["digest"] != before["digest"], "retargeted approved link must invalidate fingerprint"


def test_external_dataset_symlink_remains_unverified(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "image.bin").write_bytes(b"external")
    third_party = tmp_path / "third_party"
    source = third_party / "Model"
    source.mkdir(parents=True)
    link = source / "fashionIQ_dataset"
    link.symlink_to(outside, target_is_directory=True)
    config = WorkbenchConfig(
        CIR_REPO_ROOT=tmp_path, CIR_DATA_ROOT=tmp_path / "data",
        FASHIONIQ_ROOT=tmp_path / "data" / "FashionIQ", WORKBENCH_HOST="127.0.0.1",
        WORKBENCH_BACKEND_PORT=8000, WORKBENCH_FRONTEND_PORT=5173,
        WORKBENCH_CHECKPOINT_ROOT=tmp_path / "checkpoints",
        WORKBENCH_RESULTS_ROOT=tmp_path / "results", WORKBENCH_THIRD_PARTY_ROOT=third_party)

    evidence = pipeline.stage_input_fingerprint_info(pipeline.Stage("evaluation", input_paths=(link,)), config)
    assert evidence["inputs"][0]["category"] == pipeline.SYMLINK_EXTERNAL
    assert evidence["weak"] is True


def test_installed_package_rewrite_invalidates_execution_identity(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A dependency rewritten in place must invalidate resume.

    The interpreter binary, the pinned source and the requirements files are all unchanged in this
    scenario, so an identity built only from those would happily resume a result produced by
    different package content.
    """
    import subprocess
    import sys

    model = {"model_id": "installed_dependency_identity", "source_dir": "Model",
             "environment_required": True,
             "environment": {"python_env_var": "EXECUTION_TEST_PYTHON"}}
    config, stage = _workspace(tmp_path, monkeypatch, model)
    (config.WORKBENCH_THIRD_PARTY_ROOT / "Model").mkdir()

    prefix = tmp_path / "model-env"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(prefix)], check=True)
    monkeypatch.setenv("EXECUTION_TEST_PYTHON", str(prefix / "bin" / "python"))

    site_packages = next((prefix / "lib").glob("python*/site-packages"))
    dist = site_packages / "demo-1.0.dist-info"
    dist.mkdir(parents=True)
    (dist / "METADATA").write_text("Name: demo\nVersion: 1.0\n")
    (dist / "RECORD").write_text("demo.py,sha256=unused,5\n")
    module = site_packages / "demo.py"
    module.write_text("value = 1\n")

    before = pipeline.stage_input_fingerprint_info(stage, config)["digest"]

    # Same length, same path, same interpreter: only the installed bytes differ. A fingerprint
    # built from size and mtime alone would miss this.
    module.write_text("value = 2\n")
    after_rewrite = pipeline.stage_input_fingerprint_info(stage, config)["digest"]
    assert after_rewrite != before

    # And an untouched environment must stay stable, or resume would never reuse anything.
    assert pipeline.stage_input_fingerprint_info(stage, config)["digest"] == after_rewrite
