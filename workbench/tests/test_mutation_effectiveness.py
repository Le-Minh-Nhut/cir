"""Targeted mutation testing: each mutation must be killed by at least one relevant test."""
from __future__ import annotations

import json
from pathlib import Path

import workbench.backend.metrics_extraction as mx
import workbench.scripts.pipeline as pipeline


# MUTATION 1: Disable authorization checking -> authorization tests must fail
def test_mutation_1_disabled_authorization_is_detected(monkeypatch, tmp_path: Path, capsys):
    """If authorization gate is removed, an unauthorized eval WOULD run -> detect via real policy test."""
    from workbench.backend.operator_config import WorkbenchConfig
    data = tmp_path / "FashionIQ"
    (data / "captions").mkdir(parents=True)
    (data / "image_splits").mkdir()
    for cat in ("dress", "shirt", "toptee"):
        (data / "image_splits" / f"split.{cat}.val.json").write_text("[]")
        (data / "captions" / f"cap.{cat}.val.json").write_text("[]")
    cfg = WorkbenchConfig(tmp_path / "repo", tmp_path / "d", data, "127.0.0.1", 8000, 5173,
                          tmp_path / "ck", tmp_path / "res", tmp_path / "tp")
    called = []

    # Verify that with the real policy, an unauthorized eval never launches
    monkeypatch.setattr(pipeline, "resolve_config", lambda: cfg)
    monkeypatch.setattr(pipeline, "run_command", lambda argv: called.append(argv) or 0)
    pipeline.main(["real", "--model", "csmcir", "--evaluate"])
    assert not any("evaluate_models.py" in str(a) for a in called)


# MUTATION 2: validate_stage_outputs always True -> output-integrity test must fail
def test_mutation_2_output_validation_detects_corruption():
    # A declared corrupt JSON output must fail validation
    import tempfile
    d = Path(tempfile.mkdtemp())
    bad = d / "out.json"
    bad.write_text("{corrupt")
    s2 = pipeline.Stage(name="s2", commands=([],), output_paths=(bad,))
    assert pipeline.validate_stage_outputs(s2) is False
    bad.write_text('{"ok": true}')
    assert pipeline.validate_stage_outputs(s2) is True


# MUTATION 3: Ignore checkpoint input hashes -> invalidation test must fail
def test_mutation_3_checkpoint_hash_in_fingerprint(tmp_path: Path):
    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"v1")
    stage = pipeline.Stage(name="eval", commands=(["/bin/true"],), input_paths=(ckpt,), model_id="x")
    fp1 = pipeline.compute_stage_input_fingerprint(stage, None)
    ckpt.write_bytes(b"v2-different")
    fp2 = pipeline.compute_stage_input_fingerprint(stage, None)
    assert fp1 != fp2


# MUTATION 4: runtime-preflight dependency failure must prevent eval (evaluator launch safety)
def test_mutation_4_failed_dependency_prevents_eval(tmp_path: Path):
    state = tmp_path / "state.json"
    called = []

    def run(argv):
        called.append(argv)
        return 1 if "doctor.py" in str(argv) else 0

    orig = pipeline.run_command
    try:
        pipeline.run_command = run
        stages = [
            pipeline.Stage(name="runtime-preflight", commands=(["/doctor.py"],), model_id="m"),
            pipeline.Stage(name="evaluation", commands=(["/evaluate_models.py"],), model_id="m", dependencies=("runtime-preflight:m",)),
        ]
        pipeline.run_stages(stages, dry_run=False, continue_on_error=True, state_path=state)
    finally:
        pipeline.run_command = orig
    assert not any("evaluate_models.py" in str(a) for a in called)


# MUTATION 5: Trusting declared LIMN macro must be detected
def test_mutation_5_limn_macro_not_trusted():
    declared = {"macro_r10": 99.0, "macro_r50": 99.0, "macro_mean": 99.0, "complete": True}
    payload = json.dumps({"model_id": "limn", "aggregate": declared, "categories": [
        {"category": "dress", "r10": 50.0, "r50": 70.0},
        {"category": "shirt", "r10": 50.0, "r50": 70.0},
        {"category": "toptee", "r10": 50.0, "r50": 70.0},
    ]})
    assert mx.parse_limn_output(payload).observed_metrics is None


# MUTATION 6: Python-existence-only readiness must be detected
def test_mutation_6_bare_python_not_declared_ready():
    from workbench.backend.runtime import environment_status
    import os
    model = {"environment_required": True, "environment": {"python_env_var": "MUT6_ENV"}}
    os.environ.pop("MUT6_ENV", None)
    st = environment_status(model)
    assert st["status"] != "READY"


# MUTATION 7: Ignoring failed subprocess return codes must be detected
def test_mutation_7_failed_return_propagates(tmp_path: Path):
    import workbench.scripts.evaluate_models as em
    from workbench.backend.operator_config import WorkbenchConfig
    cfg = WorkbenchConfig(tmp_path / "repo", tmp_path / "d", tmp_path / "FashionIQ", "127.0.0.1", 8000, 5173,
                          tmp_path / "ck", tmp_path / "res", tmp_path / "tp")
    (tmp_path / "FashionIQ").mkdir()
    plan = em.EvaluationPlan(
        model_id="synthetic_model", checkpoint_id="ck", protocol="fashioniq_original_split",
        dataset_root=tmp_path / "FashionIQ", source=tmp_path, checkpoint=tmp_path / "ck.pt",
        cwd=tmp_path, command=["/bin/sh", "-c", "exit 7"], pin="a", actual_source_commit="a",
    )
    (tmp_path / "ck.pt").write_bytes(b"x")
    assert em.execute(plan, cfg) == 7


# MUTATION 8: Treating aggregate reports as schema-v2 must be detected
def test_mutation_8_aggregate_not_indexed(tmp_path: Path):
    import workbench.backend.index as index
    root = tmp_path / "results"
    root.mkdir()
    (root / "aggregate_report.json").write_text('{"artifact_type": "official_aggregate_evaluation_report"}')
    assert (root / "aggregate_report.json") not in index.result_files(root)
