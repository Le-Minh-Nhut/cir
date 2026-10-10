"""Regression tests proving resume invalidation on actual production-generated stages."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import workbench.scripts.pipeline as pipeline
from workbench.backend.operator_config import WorkbenchConfig


@pytest.fixture
def prod_env(tmp_path: Path):
    repo = tmp_path / "repo"
    data = tmp_path / "FashionIQ"
    checkpoints = tmp_path / "checkpoints"
    results = tmp_path / "results"
    third_party = tmp_path / "third_party"
    reports = repo / "workbench" / "artifacts" / "reports"

    for d in (repo, data, checkpoints, results, third_party, reports):
        d.mkdir(parents=True)

    captions = data / "captions"
    splits = data / "image_splits"
    captions.mkdir()
    splits.mkdir()

    for cat in ("dress", "shirt", "toptee"):
        (splits / f"split.{cat}.val.json").write_text('["img0"]')
        (captions / f"cap.{cat}.val.json").write_text('[{"captions": ["a"]}]')

    csmcir_source = third_party / "CSMCIR"
    csmcir_source.mkdir()
    subprocess.run(["git", "init"], cwd=csmcir_source, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=csmcir_source, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.invalid"], cwd=csmcir_source, check=True)
    (csmcir_source / "README.md").write_text("CSMCIR")
    subprocess.run(["git", "add", "."], cwd=csmcir_source, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=csmcir_source, check=True)

    csmcir_ckpt = checkpoints / "csmcir" / "fashioniq_tuned_clip_best.pt"
    csmcir_ckpt.parent.mkdir(parents=True)
    csmcir_ckpt.write_bytes(b"initial-checkpoint-bytes")

    return WorkbenchConfig(
        CIR_REPO_ROOT=repo,
        CIR_DATA_ROOT=data.parent,
        FASHIONIQ_ROOT=data,
        WORKBENCH_HOST="127.0.0.1",
        WORKBENCH_BACKEND_PORT=8000,
        WORKBENCH_FRONTEND_PORT=5173,
        WORKBENCH_CHECKPOINT_ROOT=checkpoints,
        WORKBENCH_RESULTS_ROOT=results,
        WORKBENCH_THIRD_PARTY_ROOT=third_party,
    )


# R01: Actual production evaluation stage declares checkpoint input and invalidates when checkpoint bytes change
def test_production_eval_stage_invalidates_on_checkpoint_mutation(prod_env):
    parser = pipeline.parser_for()
    args = parser.parse_args(["all", "--model", "csmcir", "--dataset-root", str(prod_env.FASHIONIQ_ROOT), "--dry-run"])
    stages = pipeline.stages_for(args, prod_env)
    eval_stage = next(s for s in stages if s.name == "evaluation" and s.model_id == "csmcir")

    # Production evaluation stage MUST declare checkpoint in input_paths
    ckpt_path = prod_env.WORKBENCH_CHECKPOINT_ROOT / "csmcir" / "fashioniq_tuned_clip_best.pt"
    assert ckpt_path in eval_stage.input_paths, f"Production eval stage does not declare checkpoint input path! Inputs: {eval_stage.input_paths}"

    # Compute initial fingerprint
    fp1 = pipeline.compute_stage_input_fingerprint(eval_stage, prod_env)

    # Mutate checkpoint bytes
    ckpt_path.write_bytes(b"mutated-checkpoint-bytes-v2")
    fp2 = pipeline.compute_stage_input_fingerprint(eval_stage, prod_env)

    assert fp1 != fp2, "Production eval stage fingerprint did not change after checkpoint mutation!"


# R02: Actual production evaluation stage declares dataset annotations and invalidates when annotations change
def test_production_eval_stage_invalidates_on_annotation_mutation(prod_env):
    parser = pipeline.parser_for()
    args = parser.parse_args(["all", "--model", "csmcir", "--dataset-root", str(prod_env.FASHIONIQ_ROOT), "--dry-run"])
    stages = pipeline.stages_for(args, prod_env)
    eval_stage = next(s for s in stages if s.name == "evaluation" and s.model_id == "csmcir")

    annotation_path = prod_env.FASHIONIQ_ROOT / "captions" / "cap.dress.val.json"
    assert annotation_path in eval_stage.input_paths, f"Production eval stage does not declare annotation input path! Inputs: {eval_stage.input_paths}"

    fp1 = pipeline.compute_stage_input_fingerprint(eval_stage, prod_env)
    annotation_path.write_text('[{"captions": ["mutated-caption"]}]')
    fp2 = pipeline.compute_stage_input_fingerprint(eval_stage, prod_env)

    assert fp1 != fp2, "Production eval stage fingerprint did not change after annotation mutation!"


# R03: Actual production sync stage declares output paths and git resolution uses registry source_dir
def test_production_sync_stage_resolves_registry_source_dir_and_declares_outputs(prod_env):
    parser = pipeline.parser_for()
    args = parser.parse_args(["all", "--model", "csmcir", "--dataset-root", str(prod_env.FASHIONIQ_ROOT), "--dry-run"])
    stages = pipeline.stages_for(args, prod_env)
    sync_stage = next(s for s in stages if s.name == "sync" and s.model_id == "csmcir")

    # Output path must be declared
    csmcir_source = prod_env.WORKBENCH_THIRD_PARTY_ROOT / "CSMCIR"
    assert any(str(csmcir_source) in str(p) for p in sync_stage.output_paths), f"Sync stage does not declare output path in CSMCIR! Outputs: {sync_stage.output_paths}"

    # Git HEAD must be included in fingerprint
    fp1 = pipeline.compute_stage_input_fingerprint(sync_stage, prod_env)
    (csmcir_source / "README.md").write_text("CSMCIR v2")
    subprocess.run(["git", "commit", "-am", "commit2"], cwd=csmcir_source, check=True)
    fp2 = pipeline.compute_stage_input_fingerprint(sync_stage, prod_env)

    assert fp1 != fp2, "Sync stage fingerprint did not change after git HEAD change in CSMCIR source_dir!"


# R04: Dirty upstream source (uncommitted tracked change) invalidates production evaluation
def test_production_eval_stage_invalidates_on_dirty_source(prod_env):
    parser = pipeline.parser_for()
    args = parser.parse_args(["all", "--model", "csmcir", "--dataset-root", str(prod_env.FASHIONIQ_ROOT), "--dry-run"])
    stages = pipeline.stages_for(args, prod_env)
    eval_stage = next(s for s in stages if s.name == "evaluation" and s.model_id == "csmcir")

    fp_clean = pipeline.compute_stage_input_fingerprint(eval_stage, prod_env)

    # Uncommitted tracked modification of the upstream source tree.
    (prod_env.WORKBENCH_THIRD_PARTY_ROOT / "CSMCIR" / "README.md").write_text("locally edited, uncommitted")

    fp_dirty = pipeline.compute_stage_input_fingerprint(eval_stage, prod_env)
    assert fp_clean != fp_dirty, "dirty upstream source did not change the production evaluation fingerprint"
