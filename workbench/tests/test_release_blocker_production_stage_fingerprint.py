"""Fingerprint evidence and production-stage resume regressions (FP-01..FP-12)."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

import workbench.backend.registry as registry
import workbench.backend.runtime as runtime
import workbench.scripts.pipeline as pipeline
from workbench.backend.operator_config import WorkbenchConfig


@pytest.fixture
def workspace(tmp_path: Path):
    repo, data, checkpoints, third_party, results = (tmp_path / name for name in
                                                       ("repo", "FashionIQ", "checkpoints", "third_party", "results"))
    for directory in (repo, data, checkpoints, third_party, results):
        directory.mkdir()
    for category in ("dress", "shirt", "toptee"):
        for folder, name, content in (
                ("image_splits", f"split.{category}.val.json", '["img0"]'),
                ("captions", f"cap.{category}.val.json", '[{"captions": ["caption"]}]')):
            path = data / folder / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
    config = WorkbenchConfig(
        CIR_REPO_ROOT=repo, CIR_DATA_ROOT=tmp_path / "data", FASHIONIQ_ROOT=data,
        WORKBENCH_HOST="127.0.0.1", WORKBENCH_BACKEND_PORT=8000,
        WORKBENCH_FRONTEND_PORT=5173, WORKBENCH_CHECKPOINT_ROOT=checkpoints,
        WORKBENCH_RESULTS_ROOT=results, WORKBENCH_THIRD_PARTY_ROOT=third_party,
    )
    return {"root": tmp_path, "repo": repo, "data": data, "checkpoints": checkpoints,
            "third_party": third_party, "config": config}


def _model(model_id: str, variants: list[dict], *, bundles: list[dict] | None = None,
           source_dir: str | None = None) -> dict:
    return {
        "model_id": model_id, "source_dir": source_dir, "checkpoint_variants": variants,
        "checkpoint_bundles": bundles or [], "environment_required": False,
        "native_protocol": "fashioniq_original_split", "upstream_commit_sha": None,
    }


def _production_eval_stage(workspace, monkeypatch, model: dict) -> pipeline.Stage:
    fake_registry = {"models": [model]}
    monkeypatch.setattr(pipeline, "load_registry", lambda: fake_registry)
    monkeypatch.setattr(registry, "load_registry", lambda: fake_registry)
    monkeypatch.setattr(registry, "model_by_id", lambda model_id: model
                        if model_id == model["model_id"] else (_ for _ in ()).throw(KeyError(model_id)))
    monkeypatch.setattr(registry, "auxiliary_model_ids", lambda: set())
    monkeypatch.setattr(pipeline, "evaluation_evidence",
                        lambda model_dict, args, config, checkpoint_id=None: (lambda record: (True, None, None)))
    args = pipeline.parser_for().parse_args([
        "all", "--model", model["model_id"], "--dataset-root", str(workspace["data"]),
    ])
    return next(stage for stage in pipeline.all_stages(args, workspace["config"])
                if stage.name == "evaluation")


def _checkpoint_inputs(workspace, model: dict) -> list[Path]:
    paths = []
    for variant in model["checkpoint_variants"]:
        path = workspace["checkpoints"] / model["model_id"] / variant["filename"]
        if variant.get("artifact_type") == "run_directory":
            path.mkdir(parents=True)
            (path / "config.json").write_text("{}")
            (path / "trained_model.pth").write_bytes(b"weights-v1")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"checkpoint-v1")
        paths.append(path)
    return paths


def _run_eval(stage, workspace, monkeypatch, state: Path, launched: list[list[str]]) -> int:
    monkeypatch.setattr(pipeline, "run_command", lambda argv: launched.append(argv) or 0)
    return pipeline.run_stages(
        [stage], resume=True, apply=True,
        authorized_capabilities={"allow_gpu_eval", "allow_preparation"},
        config=workspace["config"], state_path=state,
    )


def test_fp01_success_is_content_verified(tmp_path: Path):
    source = tmp_path / "input.json"
    source.write_text('{"value": 1}')
    info = pipeline.stage_input_fingerprint_info(pipeline.Stage("success", input_paths=(source,)))
    assert info["inputs"][0]["category"] == pipeline.CONTENT_VERIFIED
    assert info["inputs"][0]["sha256"] == registry.sha256_file(source)
    assert info["weak"] is False


def test_fp02_oserror_is_unreadable_and_changes_digest(tmp_path: Path, monkeypatch):
    source = tmp_path / "input.bin"
    source.write_bytes(b"input")
    stage = pipeline.Stage("hash-error", input_paths=(source,))
    verified = pipeline.stage_input_fingerprint_info(stage)
    monkeypatch.setattr(registry, "sha256_file",
                        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("read failed")))
    failed = pipeline.stage_input_fingerprint_info(stage)
    assert failed["inputs"][0]["category"] == pipeline.UNREADABLE
    assert failed["weak"] is True
    assert failed["digest"] != verified["digest"]


def test_fp03_unreadable_input_cannot_resume_and_reason_is_recorded(tmp_path: Path, monkeypatch):
    source = tmp_path / "checkpoint.pt"
    source.write_bytes(b"checkpoint")
    stage = pipeline.Stage("hash-resume", commands=(("/synthetic",),), input_paths=(source,))
    calls: list[list[str]] = []
    state = tmp_path / "state.json"
    monkeypatch.setattr(pipeline, "run_command", lambda argv: calls.append(argv) or 0)
    assert pipeline.run_stages([stage], resume=True, state_path=state) == 0
    monkeypatch.setattr(registry, "sha256_file",
                        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("read failed")))
    assert pipeline.run_stages([stage], resume=True, state_path=state) == 0
    assert len(calls) == 2
    record = pipeline.load_pipeline_state(state)["stages"]["hash-resume"]
    assert record["input_fingerprint_evidence"]["inputs"][0]["category"] == pipeline.UNREADABLE
    assert pipeline.UNREADABLE in record["resume_refused_reason"]


def test_fp04_large_same_size_change_forces_rerun(tmp_path: Path, monkeypatch):
    source = tmp_path / "large.bin"
    # Larger than the in-memory hash threshold, so this exercises the streaming path.
    with source.open("wb") as handle:
        handle.truncate(pipeline._STREAM_HASH_THRESHOLD + 1)
    stage = pipeline.Stage("large-input", commands=(("/synthetic",),), input_paths=(source,))
    state = tmp_path / "state.json"
    calls: list[list[str]] = []
    monkeypatch.setattr(pipeline, "run_command", lambda argv: calls.append(argv) or 0)
    assert pipeline.run_stages([stage], resume=True, state_path=state) == 0
    before = source.stat()
    with source.open("r+b") as handle:
        handle.write(b"x")
    os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert source.stat().st_size == before.st_size
    assert pipeline.run_stages([stage], resume=True, state_path=state) == 0
    assert len(calls) == 2


def test_fp05_missing_production_checkpoint_invalidates(tmp_path: Path, workspace, monkeypatch):
    model = _model("synthetic_model", [{"checkpoint_id": "ckpt", "filename": "model.pt"}])
    checkpoint, = _checkpoint_inputs(workspace, model)
    stage = _production_eval_stage(workspace, monkeypatch, model)
    assert checkpoint in stage.input_paths
    launched: list[list[str]] = []
    state = tmp_path / "state.json"
    assert _run_eval(stage, workspace, monkeypatch, state, launched) == 0
    checkpoint.unlink()
    assert _run_eval(stage, workspace, monkeypatch, state, launched) == 0
    assert len(launched) == 2
    record = pipeline.load_pipeline_state(state)["stages"][pipeline.stage_key(stage)]
    evidence = next(item for item in record["input_fingerprint_evidence"]["inputs"]
                    if item["path"] == str(checkpoint))
    assert evidence["category"] == "MISSING"


def test_fp06_changed_limn_bundle_member_invalidates_production_resume(workspace, monkeypatch, tmp_path):
    variants = [
        {"checkpoint_id": f"base_iter0_{category}", "filename": f"0_{category}_best_model.pt"}
        for category in ("dress", "shirt", "toptee")
    ]
    model = _model("limn", variants, bundles=[{
        "bundle_id": "base_iter0_all_categories",
        "required_checkpoint_ids": [variant["checkpoint_id"] for variant in variants],
    }])
    paths = _checkpoint_inputs(workspace, model)
    stage = _production_eval_stage(workspace, monkeypatch, model)
    assert all(path in stage.input_paths for path in paths)
    state, launched = tmp_path / "limn-state.json", []
    assert _run_eval(stage, workspace, monkeypatch, state, launched) == 0
    changed = paths[1]
    changed.write_bytes(changed.read_bytes().replace(b"v1", b"v2"))
    assert _run_eval(stage, workspace, monkeypatch, state, launched) == 0
    assert len(launched) == 2


def test_fp07_changed_dcnet_declared_member_invalidates_production_resume(workspace, monkeypatch, tmp_path):
    model = _model("dcnet", [{
        "checkpoint_id": "fashioniq_run_directory", "filename": "fashioniq_dcnet",
        "artifact_type": "run_directory",
    }])
    checkpoint_dir, = _checkpoint_inputs(workspace, model)
    stage = _production_eval_stage(workspace, monkeypatch, model)
    assert checkpoint_dir in stage.input_paths
    state, launched = tmp_path / "dcnet-state.json", []
    assert _run_eval(stage, workspace, monkeypatch, state, launched) == 0
    member = checkpoint_dir / "trained_model.pth"
    member.write_bytes(b"weights-v2")
    assert _run_eval(stage, workspace, monkeypatch, state, launched) == 0
    assert len(launched) == 2


def test_fp08_unchanged_verified_content_resumes(tmp_path: Path, monkeypatch):
    source = tmp_path / "input.json"
    source.write_text('{"value": 1}')
    stage = pipeline.Stage("verified-resume", commands=(("/synthetic",),), input_paths=(source,))
    state = tmp_path / "state.json"
    calls: list[list[str]] = []
    monkeypatch.setattr(pipeline, "run_command", lambda argv: calls.append(argv) or 0)
    assert pipeline.run_stages([stage], resume=True, state_path=state) == 0
    assert pipeline.run_stages([stage], resume=True, state_path=state) == 0
    assert len(calls) == 1


def test_fp09_a_large_file_is_content_verified_not_metadata_only(tmp_path: Path):
    """Size never downgrades an input to metadata: a checkpoint's bytes are the input.

    An earlier revision stopped hashing files above a 64 MiB threshold and still called the
    result verified evidence, so a checkpoint could change without invalidating resume.
    """
    source = tmp_path / "large.bin"
    with source.open("wb") as handle:
        handle.truncate(pipeline._STREAM_HASH_THRESHOLD + 1)
    info = pipeline.stage_input_fingerprint_info(pipeline.Stage("large", input_paths=(source,)))
    entry = info["inputs"][0]
    assert entry["category"] == pipeline.CONTENT_VERIFIED
    assert entry["sha256"]
    assert info["weak"] is False


def test_fp10_git_provenance_error_cannot_preserve_verified_fingerprint(
        tmp_path: Path, workspace, monkeypatch):
    source = workspace["third_party"] / "SyntheticModel"
    source.mkdir()
    (source / "source.py").write_text("source-v1\n")
    for argv in (["git", "init", "-q"], ["git", "config", "user.email", "t@t.invalid"],
                 ["git", "config", "user.name", "T"], ["git", "add", "."],
                 ["git", "commit", "-qm", "pin"]):
        subprocess.run(argv, cwd=source, check=True, capture_output=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    model = _model("synthetic_model", [], source_dir="SyntheticModel")
    model["upstream_commit_sha"] = commit
    monkeypatch.setattr(registry, "model_by_id", lambda model_id: model)
    monkeypatch.setattr(runtime, "source_provenance", lambda *args: {
        "verified": True, "actual_commit": commit, "dirty": False,
        "provenance_digest": "verified-source", "error": None,
    })
    stage = pipeline.Stage("source-provenance", commands=(("/synthetic",),), model_id="synthetic_model")
    state, calls = tmp_path / "state.json", []
    monkeypatch.setattr(pipeline, "run_command", lambda argv: calls.append(argv) or 0)
    assert pipeline.run_stages([stage], resume=True, config=workspace["config"], state_path=state) == 0
    verified = pipeline.load_pipeline_state(state)["stages"][pipeline.stage_key(stage)]["fingerprint"]
    monkeypatch.setattr(runtime, "source_provenance",
                        lambda *args: (_ for _ in ()).throw(OSError("git metadata unreadable")))
    failed_info = pipeline.stage_input_fingerprint_info(stage, workspace["config"])
    assert failed_info["weak"] is True
    assert failed_info["source_provenance"]["category"] == pipeline.UNREADABLE
    assert failed_info["digest"] != verified
    assert pipeline.run_stages([stage], resume=True, config=workspace["config"], state_path=state) == 0
    assert len(calls) == 2


def test_fp11_non_regular_declared_input_is_not_a_file(tmp_path: Path):
    source = tmp_path / "pipe"
    os.mkfifo(source)
    info = pipeline.stage_input_fingerprint_info(pipeline.Stage("not-a-file", input_paths=(source,)))
    # A FIFO is not a regular file: it has no content digest, so it cannot be verified evidence.
    assert info["inputs"][0]["category"] == pipeline.NOT_A_FILE
    assert info["weak"] is True


def test_fp12_generated_production_stage_records_fields_and_resumes(
        tmp_path: Path, workspace, monkeypatch, capsys):
    model = _model("synthetic_model", [{"checkpoint_id": "ckpt", "filename": "model.pt"}])
    _checkpoint_inputs(workspace, model)
    stage = _production_eval_stage(workspace, monkeypatch, model)
    assert stage.name == "evaluation" and stage.model_id == "synthetic_model"
    state, launched = tmp_path / "state.json", []
    assert _run_eval(stage, workspace, monkeypatch, state, launched) == 0
    record = pipeline.load_pipeline_state(state)["stages"][pipeline.stage_key(stage)]
    assert record["input_fingerprint_evidence"]["inputs"]
    assert all(item["category"] == pipeline.CONTENT_VERIFIED
               for item in record["input_fingerprint_evidence"]["inputs"])
    assert record["input_fingerprint_evidence"]["weak"] is False

    assert _run_eval(stage, workspace, monkeypatch, state, launched) == 0
    assert len(launched) == 1
    assert "[SKIP] resume: already completed" in capsys.readouterr().out


def test_fp13_same_size_member_rewrite_invalidates_a_directory_input(tmp_path: Path):
    """A directory input is the tree, so a rewritten member must change the fingerprint.

    The stage input list names the checkpoint *directory*. Resolving it by the directory's own
    metadata made a changed member invisible: the directory's mtime does not move when a file
    inside it is rewritten, and weight files usually keep their size.
    """
    checkpoint = tmp_path / "run_directory"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    member = checkpoint / "trained_model.pth"
    member.write_bytes(b"weights-v1")
    stage = pipeline.Stage("evaluation", commands=(("/synthetic",),), input_paths=(checkpoint,))

    first = pipeline.stage_input_fingerprint_info(stage)
    assert first["inputs"][0]["category"] == pipeline.TREE_CONTENT_VERIFIED
    assert first["weak"] is False

    member.write_bytes(b"weights-v2")          # same size: 10 bytes either way
    second = pipeline.stage_input_fingerprint_info(stage)
    assert second["digest"] != first["digest"], "a same-size member rewrite must be visible"

    third = pipeline.stage_input_fingerprint_info(stage)
    assert third["digest"] == second["digest"], "an unchanged tree must stay stable"
