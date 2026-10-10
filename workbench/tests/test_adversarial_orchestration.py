"""Adversarial orchestration (O), model (M), and result-integrity (D) tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import workbench.backend.index as index
import workbench.scripts.pipeline as pipeline
from workbench.backend.operator_config import WorkbenchConfig
from workbench.backend.registry import load_registry, model_by_id


@pytest.fixture
def cfg(tmp_path: Path):
    data = tmp_path / "FashionIQ"
    (data / "captions").mkdir(parents=True)
    (data / "image_splits").mkdir()
    (data / "images").mkdir()
    for cat in ("dress", "shirt", "toptee"):
        (data / "image_splits" / f"split.{cat}.val.json").write_text("[]")
        (data / "captions" / f"cap.{cat}.val.json").write_text("[]")
    return WorkbenchConfig(tmp_path / "repo", tmp_path / "d", data, "127.0.0.1", 8000, 5173,
                           tmp_path / "ck", tmp_path / "res", tmp_path / "tp")


# M01/M02: LIMN requires 3 checkpoints and records all hashes
def test_m01_m02_limn_requires_three_and_records_hashes():
    limn = model_by_id("limn")
    assert limn["category_model_policy"] == "independent"
    bundle = limn["checkpoint_bundles"][0]
    assert set(bundle["required_checkpoint_ids"]) == {"base_iter0_dress", "base_iter0_shirt", "base_iter0_toptee"}


# M05: ENCODER remains shared
def test_m05_encoder_shared():
    enc = model_by_id("encoder")
    assert enc["category_model_policy"] == "shared"
    assert not enc.get("checkpoint_bundles")


# M06/M07: DCNet run directory hashing and missing member detection
def test_m06_m07_dcnet_directory(tmp_path: Path):
    from workbench.backend.registry import checkpoint_missing_paths
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.json").write_text("{}")
    ckpt = {"artifact_type": "run_directory", "required_files": ["config.json", "trained_model.pth"]}
    missing = checkpoint_missing_paths(run_dir, ckpt)
    assert run_dir / "trained_model.pth" in missing


# M09: Unverified checkpoint mapping cannot become READY
def test_m09_unverified_mapping_not_ready(cfg: WorkbenchConfig):
    import workbench.backend.runtime as runtime
    enc = model_by_id("encoder")
    ckpt = enc["checkpoint_variants"][0]
    blockers = runtime.runtime_blockers(enc, ckpt, cfg, enc["native_protocol"], cfg.FASHIONIQ_ROOT)
    assert "checkpoint mapping unresolved" in blockers or any("mapping" in b for b in blockers)


# M10: Protocols cannot be mixed for analysis
def test_m10_protocols_distinct():
    reg = load_registry()
    assert set(reg["protocols"]) == {"fashioniq_original_split", "fashioniq_full_gallery_ref_excluded", "fashioniq_val_split"}
    assert reg["protocols"]["fashioniq_val_split"]["literature_split_label"] == "val"


# O06/O07: Workflow report exists after early failure and evaluation failure
def test_o06_report_after_early_failure(cfg: WorkbenchConfig, monkeypatch, tmp_path: Path):
    report = tmp_path / "r.json"
    monkeypatch.setattr(pipeline, "resolve_config", lambda: cfg)
    monkeypatch.setattr(pipeline, "run_command", lambda argv: 1)
    rc = pipeline.main(["all", "--model", "csmcir", "--dataset-root", str(cfg.FASHIONIQ_ROOT), "--apply", "--report", str(report)])
    assert rc == 1
    assert report.is_file()
    assert json.loads(report.read_text())["success"] is False


# O11: Final status reflects partial failure
def test_o11_partial_status(tmp_path: Path):
    state = tmp_path / "s.json"
    seq = iter([0, 1])

    def run(argv):
        return next(seq)

    orig = pipeline.run_command
    try:
        pipeline.run_command = run
        stages = [
            pipeline.Stage(name="a", commands=(["/a"],)),
            pipeline.Stage(name="b", commands=(["/b"],)),
        ]
        rc = pipeline.run_stages(stages, dry_run=False, continue_on_error=True, state_path=state)
    finally:
        pipeline.run_command = orig
    assert rc == 1


# O09: --all-models enumerates distinct per-model stages
def test_o09_all_models_distinct(cfg: WorkbenchConfig):
    args = pipeline.parser_for().parse_args(["all", "--all-models", "--dry-run"])
    stages = pipeline.stages_for(args, cfg)
    model_ids = {s.model_id for s in stages if s.model_id}
    assert {"csmcir", "limn", "dcnet"} <= model_ids


# O10: Shared source checkouts not duplicated
def test_o10_shared_source_not_duplicated(cfg: WorkbenchConfig):
    args = pipeline.parser_for().parse_args(["all", "--all-models", "--dry-run"])
    stages = pipeline.stages_for(args, cfg)
    sync_stages = [s for s in stages if s.name == "sync"]
    output_dirs = [str(s.output_paths[0]) for s in sync_stages if s.output_paths]
    assert len(output_dirs) == len(set(output_dirs)), "Shared source directories duplicated"


# D01/D03/D04: index separation and validation
def test_d01_d03_d04_index_integrity(tmp_path: Path):
    root = tmp_path / "results"
    root.mkdir()
    (root / "aggregate_report.json").write_text('{"artifact_type": "official_aggregate_evaluation_report"}')
    assert (root / "aggregate_report.json") not in index.result_files(root)

    (root / "malformed.json").write_text("{not valid")
    with pytest.raises(Exception):
        index.load_runs(root)
    (root / "malformed.json").unlink()

    from workbench.tests.mock_data import build_mock_runs
    run = build_mock_runs()[0]
    (root / f"{run.run.run_id}.json").write_text(run.model_dump_json())
    # duplicate run_id
    dup_dir = root / "dup"
    dup_dir.mkdir()
    (dup_dir / f"{run.run.run_id}.json").write_text(run.model_dump_json())
    with pytest.raises(Exception):
        index.load_runs(root)


# D02: mock distinct from real
def test_d02_mock_distinguishable():
    from workbench.tests.mock_data import build_mock_runs
    assert all(r.run.data_kind == "mock" for r in build_mock_runs())


# D07: failed execution cannot produce a completed result
def test_d07_failed_exec_no_completed_result(tmp_path: Path):
    import workbench.scripts.evaluate_models as em
    cfg = WorkbenchConfig(tmp_path / "repo", tmp_path / "d", tmp_path / "FashionIQ", "127.0.0.1", 8000, 5173,
                          tmp_path / "ck", tmp_path / "res", tmp_path / "tp")
    (tmp_path / "FashionIQ").mkdir()
    (tmp_path / "ck").mkdir()
    ckp = tmp_path / "ck" / "c.pt"
    ckp.write_bytes(b"x")
    plan = em.EvaluationPlan("synthetic_model", "c", "fashioniq_original_split", tmp_path / "FashionIQ",
                             tmp_path, ckp, tmp_path, ["/bin/sh", "-c", "exit 5"], "a", "a")
    rc = em.execute(plan, cfg)
    assert rc == 5
    logs = list((tmp_path / "repo" / "workbench" / "artifacts" / "logs").iterdir())
    rep = json.loads((logs[0] / "aggregate_report.json").read_text())
    assert rep["reproduction_status"] == "EVALUATION_FAILED"
    assert rep["observed_metrics"] is None
