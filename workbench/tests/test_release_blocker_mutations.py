"""Targeted mutation tests for the release-blocker fixes (MUT-A..MUT-F).

Method: each detector asserts the *invariant* that the corresponding regression
suite protects. The mutation restores the original buggy behaviour by patching
production code, and the detector must then fail. A surviving mutation fails this
suite, so a mutation is never committed and never silently tolerated.
"""
from __future__ import annotations

import json
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from workbench.backend import runtime
from workbench.backend.operator_config import WorkbenchConfig
from workbench.backend.runtime import check_environment_isolation, python_executable
import workbench.scripts.evaluate_models as em


def _probe_false_selection(runtime_module, model) -> str:
    report = runtime_module.verify_environment(model, probe=False)
    if report.tier == "RUNTIME_READY" and report.interpreter:
        return report.interpreter
    raise RuntimeError(f"execution environment not ready [{report.tier}/{report.status}]")


def _assert_mutation_killed(monkeypatch, detector, mutator, *, label: str) -> None:
    assert detector() is True, f"invariant already violated before mutating: {label}"
    with monkeypatch.context() as ctx:
        mutator(ctx)
        assert detector() is False, f"MUTATION SURVIVED: {label}"


def make_venv(root: Path, name: str = "env") -> Path:
    import venv as _venv

    env_dir = root / name
    _venv.EnvBuilder(with_pip=False, system_site_packages=False).create(env_dir)
    return env_dir / "bin" / "python"


def install_stub(interpreter: Path, name: str, version: str) -> None:
    for lib in (interpreter.parent.parent / "lib").glob("python*"):
        dist = lib / "site-packages" / f"{name}-{version}.dist-info"
        dist.mkdir(parents=True, exist_ok=True)
        (dist / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
        return
    raise AssertionError("no site-packages in the test environment")


@pytest.fixture(autouse=True)
def _clear_cache():
    runtime.forget_environment()
    yield
    runtime.forget_environment()


def make_workspace(tmp_path: Path, name: str) -> dict:
    """Minimal real evaluation environment used by the MUT-D / MUT-E detectors."""
    repo = tmp_path / name
    source = repo / "source"
    source.mkdir(parents=True)
    checkpoint = repo / "checkpoint.pt"
    checkpoint.write_bytes(b"synthetic-checkpoint")
    config = WorkbenchConfig(
        CIR_REPO_ROOT=repo,
        CIR_DATA_ROOT=repo / "data",
        FASHIONIQ_ROOT=repo / "FashionIQ",
        WORKBENCH_HOST="127.0.0.1",
        WORKBENCH_BACKEND_PORT=8000,
        WORKBENCH_FRONTEND_PORT=5173,
        WORKBENCH_CHECKPOINT_ROOT=repo / "checkpoints",
        WORKBENCH_RESULTS_ROOT=repo / "results",
        WORKBENCH_THIRD_PARTY_ROOT=repo / "third_party",
    )
    script = repo / "evaluator.py"
    script.write_text(
        "import json, os, sys\n"
        'if os.environ.get("MUT_FAIL"): sys.exit(5)\n'
        'if os.environ.get("MUT_SILENT"): sys.exit(0)\n'
        'print("evaluating")\n'
        "print(json.dumps({'average_recall_at10': 55.0, 'average_recall_at50': 75.0,"
        " 'average_recall': 65.0, 'dress_recall_at10': 60.0, 'dress_recall_at50': 80.0,"
        " 'shirt_recall_at10': 55.0, 'shirt_recall_at50': 75.0,"
        " 'toptee_recall_at10': 50.0, 'toptee_recall_at50': 70.0}))\n"
    )
    return {"config": config, "checkpoint": checkpoint, "source": source,
            "command": [sys.executable, str(script)], "root": repo}


def plan_for(ws: dict) -> em.EvaluationPlan:
    return em.EvaluationPlan("csmcir", "ckpt_v1", "fashioniq_original_split",
                             ws["root"] / "FashionIQ", ws["source"], ws["checkpoint"],
                             ws["source"], ws["command"], "a" * 40, "a" * 40)


# ------------------------------------------------------------------------- MUT-A

def test_mut_a_restoring_probe_false_interpreter_selection_is_detected(tmp_path, monkeypatch):
    """The old bug: python_executable() consulted a non-probing verification."""
    interpreter = make_venv(tmp_path)
    install_stub(interpreter, "stubdep", "1.2.3")
    monkeypatch.setenv("MUT_A_VAR", str(interpreter))
    model = {"model_id": "m", "environment_required": True,
             "environment": {"python_env_var": "MUT_A_VAR", "packages": ["stubdep==1.2.3"]}}
    runtime.forget_environment()

    def detector() -> bool:
        runtime.forget_environment()
        try:
            return python_executable(model) == str(interpreter)
        except RuntimeError:
            return False

    def mutate(ctx):
        # The historical behaviour: python_executable() consulted a *non-probing*
        # verification, whose tier never reaches RUNTIME_READY.
        ctx.setattr(runtime, "require_verified_model_interpreter",
                    lambda model, **kwargs: _probe_false_selection(runtime, model))

    _assert_mutation_killed(monkeypatch, detector, mutate,
                            label="probe=False interpreter selection returned")


# ------------------------------------------------------------------------- MUT-B

def test_mut_b_realpath_only_venv_rejection_is_detected(tmp_path, monkeypatch):
    """The old bug: a venv sharing the system binary was rejected as 'active env'."""
    interpreter = make_venv(tmp_path)
    # Reproduce the real Linux layout: the venv's bin/python is a symlink to the very
    # interpreter binary the orchestrator resolves to.
    interpreter.unlink()
    interpreter.symlink_to(Path(sys.executable).resolve())
    assert interpreter.resolve() == Path(sys.executable).resolve(), "fixture precondition"
    assert check_environment_isolation(interpreter)[0] is True

    def detector() -> bool:
        return runtime.check_environment_isolation(interpreter)[0] is True

    def mutate(ctx):
        def resolve_only_isolation(candidate: Path):
            if Path(candidate).resolve() == Path(sys.executable).resolve():
                return False, "cannot use active workbench environment"
            return True, "ISOLATION_VERIFIED"

        ctx.setattr(runtime, "check_environment_isolation", resolve_only_isolation)

    _assert_mutation_killed(monkeypatch, detector, mutate,
                            label="realpath-only venv rejection restored")


# ------------------------------------------------------------------------- MUT-C

def test_mut_c_treating_declared_runtime_artifacts_as_dirty_is_detected(
        tmp_path, monkeypatch):
    """The old bug: any untracked CSMCIR artifact immediately dirtied the source."""
    repo_root = tmp_path / "repo"
    third_party = repo_root / "third_party"
    source = third_party / "CSMCIR"
    canonical = repo_root / "data" / "FashionIQ"
    source.mkdir(parents=True)
    canonical.mkdir(parents=True)
    for argv in (["git", "init", "-q"], ["git", "config", "user.email", "t@t.invalid"],
                 ["git", "config", "user.name", "T"]):
        subprocess.run(argv, cwd=source, check=True, capture_output=True)
    (source / "eval.py").write_text("x")
    subprocess.run(["git", "add", "."], cwd=source, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-qm", "pin"], cwd=source, check=True, capture_output=True)
    (source / "fashionIQ_dataset").symlink_to(canonical, target_is_directory=True)

    monkeypatch.setenv("CIR_REPO_ROOT", str(repo_root))
    monkeypatch.setenv("FASHIONIQ_ROOT", str(canonical))
    monkeypatch.setenv("WORKBENCH_THIRD_PARTY_ROOT", str(third_party))

    def detector() -> bool:
        return runtime._source_dirty(source) is False

    def mutate(ctx):
        def all_untracked_dirty(candidate: Path, model=None):
            result = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"],
                                    cwd=candidate, capture_output=True, text=True)
            return bool(result.stdout.strip())

        ctx.setattr(runtime, "_source_dirty", all_untracked_dirty)

    _assert_mutation_killed(monkeypatch, detector, mutate,
                            label="declared CSMCIR runtime artifact treated as dirty")


# ------------------------------------------------------------------------- MUT-D

def test_mut_d_accepting_a_leftover_pointer_as_proof_is_detected(tmp_path, monkeypatch):
    """The old bug: any existing latest pointer satisfied the evaluation stage."""
    ws = make_workspace(tmp_path, "mut_d")
    monkeypatch.setenv("WORKBENCH_NO_ENV_CACHE", "1")

    first = uuid.uuid4().hex[:10]
    assert em.execute(plan_for(ws), ws["config"], run_id=first) == 0
    successful = em.reports_root(ws["config"]) / "csmcir_ckpt_v1_latest_successful.json"
    assert successful.is_file()

    monkeypatch.setenv("MUT_SILENT", "1")
    silent = uuid.uuid4().hex[:10]
    assert em.execute(plan_for(ws), ws["config"], run_id=silent) == 0
    monkeypatch.delenv("MUT_SILENT")

    def detector() -> bool:
        evidence = em.completion_validator("csmcir", "ckpt_v1", "fashioniq_original_split",
                                           "a" * 40, ws["config"])
        ok, _, _ = evidence({"run_id": silent})
        return ok is False

    def mutate(ctx):
        def pointer_is_enough(model_id, checkpoint_id, protocol, pin, config, checkpoint_file=None):
            def _validate(record=None):
                if successful.is_file():
                    return True, None, {"completion_proof": str(successful)}
                return False, "no success pointer", None

            return _validate

        ctx.setattr(em, "completion_validator", pointer_is_enough)
        import workbench.scripts.pipeline as pipeline

        ctx.setattr(pipeline, "evaluation_evidence", None, raising=False)

    _assert_mutation_killed(monkeypatch, detector, mutate,
                            label="leftover latest pointer accepted as completion proof")


# ------------------------------------------------------------------------- MUT-E

def test_mut_e_advancing_latest_successful_after_failure_is_detected(tmp_path, monkeypatch):
    """The old bug: a failed evaluator still replaced the latest success pointer."""
    ws = make_workspace(tmp_path, "mut_e")
    monkeypatch.setenv("WORKBENCH_NO_ENV_CACHE", "1")

    good = uuid.uuid4().hex[:10]
    assert em.execute(plan_for(ws), ws["config"], run_id=good) == 0
    successful = em.reports_root(ws["config"]) / "csmcir_ckpt_v1_latest_successful.json"
    assert json.loads(successful.read_text())["run_id"] == good

    def detector() -> bool:
        monkeypatch.setenv("MUT_FAIL", "1")
        bad = uuid.uuid4().hex[:10]
        try:
            em.execute(plan_for(ws), ws["config"], run_id=bad)
        finally:
            monkeypatch.delenv("MUT_FAIL")
        return json.loads(successful.read_text())["run_id"] == good

    def mutate(ctx):
        real_execute = em.execute

        def failing_run_advances_success(plan, config, run_id=None):
            code = real_execute(plan, config, run_id=run_id)
            em._atomic_write_json(em.latest_pointer(config, plan, "successful"),
                                  {"run_id": run_id, "return_code": code})
            return code

        ctx.setattr(em, "execute", failing_run_advances_success)

    _assert_mutation_killed(monkeypatch, detector, mutate,
                            label="failed run advanced latest_successful")


# ------------------------------------------------------------------------- MUT-F

def test_mut_f_restoring_prefix_version_matching_is_detected(monkeypatch):
    """The old bug: startswith() matching accepted 1.12.10 for a required 1.12.1."""

    def detector() -> bool:
        return runtime.version_satisfies("==1.12.1", "1.12.10") is False

    def mutate(ctx):
        def startswith_matching(constraint: str, installed: str | None) -> bool:
            if installed is None:
                return False
            want = constraint.removeprefix("==")
            got = str(installed)
            return got.startswith(want) or want.startswith(got)

        ctx.setattr(runtime, "version_satisfies", startswith_matching)

    _assert_mutation_killed(monkeypatch, detector, mutate,
                            label="startswith version matching restored")


def test_mut_f_bare_version_declaration_is_exact_not_prefix(monkeypatch, tmp_path):
    """A bare declaration is an exact pin: 1.12.10 must not satisfy 1.12.1."""
    monkeypatch.setenv("MUT_F_VAR", str(make_venv(tmp_path)))
    model = {"model_id": "m", "environment_required": True,
             "environment": {"python_env_var": "MUT_F_VAR", "pytorch": "1.12.1"}}
    runtime.forget_environment()

    def detector() -> bool:
        runtime.forget_environment()
        # The bare declaration must be interpreted as an exact pin.
        return runtime.declared_requirement("stubdep", "1.12.1") == "==1.12.1"

    def mutate(ctx):
        ctx.setattr(runtime, "declared_requirement", lambda package, value: value)

    _assert_mutation_killed(monkeypatch, detector, mutate,
                            label="bare version declaration treated as a loose prefix")


# ------------------------------------------------------------------------- MUT-G

def test_mut_g_manage_environment_using_a_non_probing_tier_is_detected(tmp_path, monkeypatch):
    """The old bug: manage_environment reported readiness from a non-probing inspect."""
    import importlib.util
    from dataclasses import replace

    from workbench.backend.operator_config import resolve_config

    spec = importlib.util.spec_from_file_location(
        "manage_environment_mut", Path("workbench/scripts/manage_environment.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    interpreter = make_venv(tmp_path)
    install_stub(interpreter, "stubdep", "1.2.3")
    source_root = tmp_path / "third_party"
    (source_root / "ENV_PROBE_SRC").mkdir(parents=True)
    monkeypatch.setenv("MUT_G_VAR", str(interpreter))
    config = replace(resolve_config(), WORKBENCH_THIRD_PARTY_ROOT=source_root)
    model = {"model_id": "env_probe", "environment_required": True, "source_dir": "ENV_PROBE_SRC",
             "environment": {"python_env_var": "MUT_G_VAR", "packages": ["stubdep==1.2.3"]}}

    def detector() -> bool:
        runtime.forget_environment()
        return module.status(model, config)["ready"] is True

    def mutate(ctx):
        # The old behaviour: readiness came from a non-probing inspect, whose tier
        # never reaches RUNTIME_READY, so a correct environment read as blocked.
        def never_ready_status(model, config):
            report = runtime.verify_environment(model, probe=False)
            return {"ready": False, "runtime_verified": False,
                    "interpreter_status": report.status, "interpreter_tier": report.tier,
                    "interpreter": report.interpreter, "variable": report.variable}

        ctx.setattr(module, "status", never_ready_status)

    _assert_mutation_killed(monkeypatch, detector, mutate,
                            label="manage_environment readiness from a non-probing inspect")
