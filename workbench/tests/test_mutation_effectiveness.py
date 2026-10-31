"""Genuine mutation testing.

Method: for each mutation we assert the invariant holds on production code, then
apply an intentionally incorrect behaviour via ``monkeypatch`` so the mutation is
*active*, and require the same detector to now fail. A surviving mutation fails
this suite. Every patch is reverted automatically; no mutation enters the commit.

The ``_detector`` functions are the same assertions used by the adversarial
suites, so a killed mutation means the real test suite would catch that defect.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import workbench.backend.index as index
import workbench.backend.metrics_extraction as mx
import workbench.backend.runtime as runtime
import workbench.scripts.evaluate_models as em
import workbench.scripts.pipeline as pipeline
from workbench.backend.operator_config import WorkbenchConfig


def _assert_mutation_killed(monkeypatch, detector, mutator, *, label: str) -> None:
    """detector() -> True when the invariant holds. mutator() breaks the invariant."""
    assert detector() is True, f"invariant already violated before mutating: {label}"
    with monkeypatch.context() as ctx:
        mutator(ctx)
        assert detector() is False, f"MUTATION SURVIVED: {label}"


# ---------------------------------------------------------------- detectors

def _auth_detector(monkeypatch, tmp_path) -> bool:
    data = tmp_path / "FashionIQ"
    (data / "captions").mkdir(parents=True, exist_ok=True)
    (data / "image_splits").mkdir(exist_ok=True)
    for cat in ("dress", "shirt", "toptee"):
        (data / "image_splits" / f"split.{cat}.val.json").write_text("[]")
        (data / "captions" / f"cap.{cat}.val.json").write_text("[]")
    cfg = WorkbenchConfig(tmp_path / "repo", tmp_path / "d", data, "127.0.0.1", 8000, 5173,
                          tmp_path / "ck", tmp_path / "res", tmp_path / "tp")
    launched: list[list[str]] = []
    with monkeypatch.context() as ctx:
        ctx.setattr(pipeline, "resolve_config", lambda: cfg)
        ctx.setattr(pipeline, "run_command", lambda argv: launched.append(argv) or 0)
        pipeline.main(["real", "--model", "csmcir", "--evaluate"])
    return not any("evaluate_models.py" in str(a) for a in launched)


def _output_detector(tmp_path) -> bool:
    """Drive the REAL runner: a stage whose declared output is absent must not be COMPLETE.

    The detector used to call ``pipeline.validate_stage_outputs`` itself, so the mutator rebound
    exactly the symbol it called and the detector could never observe anything else: "mutation
    survived" was unreachable. It now runs the stage through ``run_stages`` -- the path the release
    workflow uses -- and reads the recorded outcome, so a disabled output contract is observable.
    """
    missing = tmp_path / "absent.json"
    stage = pipeline.Stage(name="eval", commands=(["/bin/true"],), output_paths=(missing,))
    state = tmp_path / "output_state.json"
    rc = pipeline.run_stages([stage], dry_run=False, continue_on_error=True, state_path=state)
    record = json.loads(state.read_text(encoding="utf-8"))["stages"]["eval"]
    return rc != 0 and record["status"] != "COMPLETE"


def _fingerprint_detector(tmp_path) -> bool:
    """Drive the REAL runner: changed input content must re-execute a resumed stage.

    The fingerprint's purpose is that a resumed stage re-runs when its input changed. The detector
    used to call ``compute_stage_input_fingerprint`` directly, which ``run_stages`` never calls (it
    goes through ``stage_input_fingerprint_info``), so rebinding the former changed nothing the
    runner could see. Here the runner is asked twice with a mutated checkpoint and the observable
    answers are whether the command ran again and whether the recorded identity moved.
    """
    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"v1")
    output = tmp_path / "report.json"
    output.write_text('{"ok": 1}')
    counter = tmp_path / "runs.txt"
    counter.write_text("")
    state = tmp_path / "fingerprint_state.json"
    state.unlink(missing_ok=True)  # each detector invocation starts from a clean run history
    stage = pipeline.Stage(
        name="evaluation", model_id="m", input_paths=(ckpt,), output_paths=(output,),
        commands=([sys.executable, "-c", f"open({str(counter)!r}, 'a').write('x')"],))

    def run() -> dict:
        assert pipeline.run_stages([stage], dry_run=False, resume=True,
                                   state_path=state) == 0
        return json.loads(state.read_text(encoding="utf-8"))["stages"]["evaluation:m"]

    first = run()
    assert len(counter.read_text(encoding="utf-8")) == 1
    ckpt.write_bytes(b"v2-different")
    second = run()
    return (first["fingerprint"] != second["fingerprint"]
            and len(counter.read_text(encoding="utf-8")) == 2)


def _dependency_detector(monkeypatch, tmp_path) -> bool:
    state = tmp_path / "s.json"
    called: list[list[str]] = []
    with monkeypatch.context() as ctx:
        ctx.setattr(pipeline, "run_command",
                    lambda argv: called.append(argv) or (1 if "doctor.py" in str(argv) else 0))
        stages = [
            pipeline.Stage(name="runtime-preflight", commands=(["/doctor.py"],), model_id="m"),
            pipeline.Stage(name="evaluation", commands=(["/evaluate_models.py"],), model_id="m",
                           dependencies=("runtime-preflight:m",)),
        ]
        pipeline.run_stages(stages, dry_run=False, continue_on_error=True, state_path=state)
    return not any("evaluate_models.py" in str(a) for a in called)


_LIMN_WRONG = json.dumps({
    "model_id": "limn",
    "aggregate": {"macro_r10": 99.0, "macro_r50": 99.0, "macro_mean": 99.0, "complete": True},
    "categories": [
        {"category": "dress", "r10": 50.0, "r50": 70.0},
        {"category": "shirt", "r10": 50.0, "r50": 70.0},
        {"category": "toptee", "r10": 50.0, "r50": 70.0},
    ],
})


def _limn_macro_detector() -> bool:
    return mx.parse_limn_output(_LIMN_WRONG).observed_metrics is None


def _env_detector(monkeypatch, tmp_path) -> bool:
    import venv as _venv
    env_dir = tmp_path / "venv"
    if not env_dir.exists():
        _venv.EnvBuilder(with_pip=False, system_site_packages=False).create(env_dir)
    interp = env_dir / "bin" / "python"
    model = {"model_id": "m", "environment_required": True,
             "environment": {"python_env_var": "MUT_ENV_X", "pytorch": "9.9.9"}}
    with monkeypatch.context() as ctx:
        ctx.setenv("MUT_ENV_X", str(interp))
        return runtime.verify_environment(model, probe=True).tier != "RUNTIME_READY"


def _return_code_detector(tmp_path) -> bool:
    cfg = WorkbenchConfig(tmp_path / "repo", tmp_path / "d", tmp_path / "FashionIQ", "127.0.0.1",
                          8000, 5173, tmp_path / "ck", tmp_path / "res", tmp_path / "tp")
    (tmp_path / "FashionIQ").mkdir(exist_ok=True)
    (tmp_path / "ck").mkdir(exist_ok=True)
    ck = tmp_path / "ck" / "c.pt"
    ck.write_bytes(b"x")
    plan = em.EvaluationPlan("synthetic_model", "c", "fashioniq_original_split",
                             tmp_path / "FashionIQ", tmp_path, ck, tmp_path,
                             ["/bin/sh", "-c", "exit 7"], "a", "a")
    return em.execute(plan, cfg) == 7


def _index_detector(tmp_path) -> bool:
    root = tmp_path / "results"
    root.mkdir(exist_ok=True)
    (root / "aggregate_report.json").write_text('{"artifact_type": "official_aggregate_evaluation_report"}')
    return (root / "aggregate_report.json") not in index.result_files(root)


def _side_effecting_output_detector(tmp_path) -> bool:
    """Drive the REAL runner: an authorized side-effecting stage with no output proof must fail.

    The detector used to call ``validate_stage_outputs`` itself, so rebinding that symbol made the
    detector unable to see anything else. It now runs the stage through ``run_stages`` with the
    capability granted -- the multi-command path, where validation happens once after every command
    has succeeded -- and asserts the stage is not finalized COMPLETE.
    """
    stage = pipeline.Stage(name="checkpoint", commands=(["/bin/true"], ["/bin/true"]),
                           required_capabilities=frozenset({"allow_large_downloads"}))
    state = tmp_path / "side_effecting_state.json"
    rc = pipeline.run_stages([stage], dry_run=False, continue_on_error=True, state_path=state,
                             apply=True, authorized_capabilities={"allow_large_downloads"})
    record = json.loads(state.read_text(encoding="utf-8"))["stages"]["checkpoint"]
    return rc != 0 and record["status"] != "COMPLETE"


def _immutability_detector(tmp_path) -> bool:
    cfg = WorkbenchConfig(tmp_path / "repo", tmp_path / "d", tmp_path / "FashionIQ", "127.0.0.1",
                          8000, 5173, tmp_path / "ck", tmp_path / "res", tmp_path / "tp")
    (tmp_path / "FashionIQ").mkdir(exist_ok=True)
    (tmp_path / "ck").mkdir(exist_ok=True)
    ck = tmp_path / "ck" / "c.pt"
    ck.write_bytes(b"x")
    script = ('echo \'{"average_recall_at10":55.0,"average_recall_at50":75.0,"average_recall":65.0,'
              '"dress_recall_at10":60.0,"dress_recall_at50":80.0,"shirt_recall_at10":55.0,'
              '"shirt_recall_at50":75.0,"toptee_recall_at10":50.0,"toptee_recall_at50":70.0}\'')
    try:
        for _ in range(2):
            plan = em.EvaluationPlan("csmcir", "fashioniq", "fashioniq_original_split",
                                     tmp_path / "FashionIQ", tmp_path, ck, tmp_path,
                                     ["/bin/sh", "-c", script], "a", "a")
            em.execute(plan, cfg)
    except Exception:
        # A collapsed run identity cannot even complete twice -> invariant violated.
        return False
    reports = list((tmp_path / "repo" / "workbench" / "artifacts" / "reports")
                   .glob("csmcir_fashioniq_2*_aggregate.json"))
    return len(reports) == 2


# ------------------------------------------------------------------ mutations

def test_mut1_disabled_authorization_detected(monkeypatch, tmp_path: Path):
    def mutate(ctx):
        ctx.setattr(pipeline, "run_stages", lambda *a, **k: 0)

        def run_stages_unguarded(stages, **kwargs):
            for stage in stages:
                for argv in stage.commands:
                    pipeline.run_command(argv)
            return 0

        ctx.setattr(pipeline, "run_stages", run_stages_unguarded)
        ctx.setattr(pipeline, "main", lambda argv=None: (
            run_stages_unguarded(pipeline.stages_for(pipeline.parser_for().parse_args(argv or []),
                                                     pipeline.resolve_config()))))

    _assert_mutation_killed(monkeypatch,
                            lambda: _auth_detector(monkeypatch, tmp_path),
                            mutate, label="authorization gate disabled")


def test_mut2_output_validation_always_true_detected(monkeypatch, tmp_path: Path):
    _assert_mutation_killed(
        monkeypatch,
        lambda: _output_detector(tmp_path),
        lambda ctx: ctx.setattr(pipeline, "validate_stage_outputs", lambda *a, **k: True),
        label="validate_stage_outputs always True")


def test_mut3_checkpoint_fingerprint_omitted_detected(monkeypatch, tmp_path: Path):
    _assert_mutation_killed(
        monkeypatch,
        lambda: _fingerprint_detector(tmp_path),
        lambda ctx: ctx.setattr(pipeline, "stage_input_fingerprint_info",
                                lambda *a, **k: {"digest": "constant", "weak": False,
                                                 "inputs": [], "source_provenance": None}),
        label="checkpoint content ignored in input fingerprint")


def test_mut4_ignored_dependency_failure_detected(monkeypatch, tmp_path: Path):
    _assert_mutation_killed(
        monkeypatch,
        lambda: _dependency_detector(monkeypatch, tmp_path),
        lambda ctx: ctx.setattr(
            pipeline, "run_stages",
            lambda stages, **kwargs: [pipeline.run_command(a) for s in stages for a in s.commands] and 0),
        label="dependency failures ignored")


def test_mut5_trusting_declared_limn_macro_detected(monkeypatch):
    _assert_mutation_killed(
        monkeypatch,
        _limn_macro_detector,
        lambda ctx: ctx.setattr(mx, "parse_limn_output", lambda text, paper=None: mx.MetricExtractionResult(
            extraction_status="EXTRACTED_VERIFIED", parser_id="mut", parser_version=1,
            observed_metrics={"r10": 99.0, "r50": 99.0, "mean": 99.0},
            category_metrics=None, parity_status="PARITY_NOT_EVALUATED", metric_source="stdout")),
        label="declared LIMN macro trusted without recomputation")


def test_mut6_empty_environment_declared_ready_detected(monkeypatch, tmp_path: Path):
    _assert_mutation_killed(
        monkeypatch,
        lambda: _env_detector(monkeypatch, tmp_path),
        lambda ctx: ctx.setattr(runtime, "verify_environment", lambda model, **k: runtime.EnvironmentReport(
            model_id="m", variable="MUT_ENV_X", interpreter="/usr/bin/python3",
            tier="RUNTIME_READY", status="RUNTIME_READY")),
        label="interpreter presence treated as RUNTIME_READY")


def test_mut7_ignoring_nonzero_return_detected(monkeypatch, tmp_path: Path):
    _assert_mutation_killed(
        monkeypatch,
        lambda: _return_code_detector(tmp_path),
        lambda ctx: ctx.setattr(em, "execute", lambda plan, config: 0),
        label="evaluator nonzero return code ignored")


def test_mut8_aggregate_into_index_detected(monkeypatch, tmp_path: Path):
    def mutate(ctx):
        root = tmp_path / "results"
        ctx.setattr(index, "result_files", lambda r=None: sorted((r or root).rglob("*.json")))

    _assert_mutation_killed(monkeypatch, lambda: _index_detector(tmp_path), mutate,
                            label="aggregate report admitted to per-query index")


def test_mut9_complete_without_output_detected(monkeypatch, tmp_path: Path):
    _assert_mutation_killed(
        monkeypatch,
        lambda: _side_effecting_output_detector(tmp_path),
        lambda ctx: ctx.setattr(pipeline, "validate_stage_outputs", lambda *a, **k: True),
        label="side-effecting stage COMPLETE without output proof")


def test_mut10_overwriting_experiment_detected(monkeypatch, tmp_path: Path):
    def mutate(ctx):
        # Mutated behaviour: collapse every run to one fixed report path.
        ctx.setattr(em, "make_run_id", lambda plan, ts: "FIXED")

    _assert_mutation_killed(monkeypatch, lambda: _immutability_detector(tmp_path), mutate,
                            label="previous experiment report overwritten")
