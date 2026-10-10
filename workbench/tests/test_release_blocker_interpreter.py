"""Release-blocker regression tests: interpreter selection and version matching.

ENV-01..ENV-13, INT-01..INT-09, VER-01..VER-06.

These exercise the *production* environment-selection path. ``python_executable()``
is never monkeypatched to return a success: the tests build real isolated
environments and real dependency contracts, then let the production code decide.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from workbench.backend import runtime
from workbench.backend.adapters.base import EvalRequest
from workbench.backend.adapters.models import DCNetAdapter, LIMNAdapter
from workbench.backend.registry import model_by_id
from workbench.backend.runtime import (
    check_environment_isolation,
    environment_blockers,
    environment_status,
    forget_environment,
    inspect_environment,
    python_executable,
    require_verified_model_interpreter,
    version_satisfies,
)


# --------------------------------------------------------------------------- fixtures

def make_venv(root: Path, name: str = "model_env") -> Path:
    """A real isolated virtual environment. The invocation path is never resolved.

    ``with_pip=False`` keeps each fixture a few hundred KB; these tests never install
    anything, they only need a genuine, verifiable environment identity.
    """
    import venv as _venv

    env_dir = root / name
    _venv.EnvBuilder(with_pip=False, system_site_packages=False, symlinks=True).create(env_dir)
    return env_dir / "bin" / "python"


def install_stub(interpreter: Path, name: str, version: str) -> None:
    """Record a real distribution in the environment's site-packages.

    ``importlib.metadata`` reads exactly this layout, so this is controlled
    dependency metadata for a lightweight test environment (no PyTorch needed).
    """
    for lib in (interpreter.parent.parent / "lib").glob("python*"):
        site_packages = lib / "site-packages"
        site_packages.mkdir(parents=True, exist_ok=True)
        dist = site_packages / f"{name}-{version}.dist-info"
        dist.mkdir(exist_ok=True)
        (dist / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
        )
        return
    raise AssertionError("no site-packages directory in the test environment")


def model_for(python: Path, *, variable: str, python_version: str | None = None,
              packages: list[str] | None = None) -> dict:
    environment: dict = {"python_env_var": variable}
    if python_version:
        environment["python"] = python_version
    if packages:
        environment["packages"] = packages
    return {"model_id": "env_probe", "environment_required": True, "environment": environment}


@pytest.fixture(autouse=True)
def _clear_verification_cache():
    forget_environment()
    yield
    forget_environment()


# ----------------------------------------------------------------- ENV-01, ENV-02

def test_env01_env02_real_venv_is_created_and_identified(tmp_path: Path, monkeypatch):
    interpreter = make_venv(tmp_path)
    monkeypatch.setenv("ENV_01_VAR", str(interpreter))

    cheap = inspect_environment(model_for(interpreter, variable="ENV_01_VAR"))
    assert cheap.tier == "INTERPRETER_PRESENT", "inspect must never claim readiness"
    assert cheap.probe_executed is False

    report = runtime.verify_environment(model_for(interpreter, variable="ENV_01_VAR"), probe=True)
    assert report.interpreter == str(interpreter), "the configured invocation path must be preserved"
    assert report.prefix and report.base_prefix
    assert report.prefix != report.base_prefix, "a real venv reports a distinct sys.prefix"
    assert report.python_version


# ------------------------------------------------------------- ENV-03, ENV-04, ENV-05, ENV-06

def test_env03_env04_env05_env06_verified_runtime_yields_configured_interpreter(
        tmp_path: Path, monkeypatch):
    interpreter = make_venv(tmp_path)
    install_stub(interpreter, "stubdep", "1.2.3")
    monkeypatch.setenv("ENV_04_VAR", str(interpreter))
    model = model_for(interpreter, variable="ENV_04_VAR", packages=["stubdep==1.2.3"])

    report = runtime.verify_environment(model, probe=True)
    assert report.tier == "RUNTIME_READY", report.reason
    assert report.dependency_probe_executed is True
    assert environment_blockers(model) == []
    assert python_executable(model) == str(interpreter)


# ------------------------------------------------------------------------- INT-01..INT-03

def test_int01_valid_linux_venv_accepted(tmp_path: Path):
    """A venv whose bin/python is a symlink to a shared interpreter is still isolated."""
    interpreter = make_venv(tmp_path)
    # Where the platform creates a symlinked bin/python, its resolved target is shared
    # with the orchestrator: exactly the case that must still be accepted.
    if interpreter.is_symlink():
        assert interpreter.resolve() != interpreter
    ok, reason = check_environment_isolation(interpreter)
    assert ok is True, reason


def test_int02_system_and_base_interpreters_rejected(tmp_path: Path):
    for candidate in ("/usr/bin/python3", "/bin/python3", "/usr/local/bin/python3"):
        ok, reason = check_environment_isolation(Path(candidate))
        assert ok is False, f"{candidate} must be rejected"
        assert ("system" in reason.lower() or "identity" in reason.lower()
                or "markers" in reason.lower() or "isolated" in reason.lower()), reason

    # Base conda, if present on this machine, must be rejected too.
    for conda in ("/opt/conda/bin/python3", "/miniconda3/bin/python3", "/anaconda3/bin/python3"):
        if Path(conda).exists():
            ok, reason = check_environment_isolation(Path(conda))
            assert ok is False
            assert "conda" in reason.lower() or "system" in reason.lower()


def test_int03_active_workbench_environment_rejected():
    ok, reason = check_environment_isolation(Path(sys.executable))
    assert ok is False
    assert "workbench" in reason.lower()


def test_int03b_malformed_and_fake_interpreters_rejected(tmp_path: Path):
    fake = tmp_path / "python"
    fake.write_text("#!/bin/sh\nexec /bin/true\n")
    fake.chmod(0o755)
    ok, reason = check_environment_isolation(fake)
    assert ok is False and "identity" in reason.lower()

    broken = make_venv(tmp_path, "broken_env")
    (broken.parent.parent / "pyvenv.cfg").unlink()
    ok, reason = check_environment_isolation(broken)
    assert ok is False, "an environment without markers is not isolated"


# ------------------------------------------------------------------------- INT-04..INT-06

def test_int04_configured_model_interpreter_is_returned(tmp_path: Path, monkeypatch):
    interpreter = make_venv(tmp_path)
    install_stub(interpreter, "stubdep", "1.2.3")
    monkeypatch.setenv("WORKBENCH_ENV_PROBE_PYTHON", str(interpreter))
    monkeypatch.setattr(runtime, "verify_environment", runtime.verify_environment)

    model = model_for(interpreter, variable="WORKBENCH_ENV_PROBE_PYTHON", packages=["stubdep==1.2.3"])
    assert require_verified_model_interpreter(model) == str(interpreter)


def test_int05_int06_adapters_use_the_verified_interpreter(tmp_path: Path, monkeypatch):
    """LIMN and DCNet commands must carry the *verified* interpreter, unpached."""
    interpreter = make_venv(tmp_path)
    install_stub(interpreter, "stubdep", "1.2.3")
    monkeypatch.setenv("WORKBENCH_LIMN_PYTHON", str(interpreter))
    monkeypatch.setenv("WORKBENCH_DCNET_PYTHON", str(interpreter))
    forget_environment()

    limn = model_by_id("limn")
    dcnet = model_by_id("dcnet")
    for model in (limn, dcnet):
        model["environment"] = {"python_env_var": "WORKBENCH_LIMN_PYTHON" if model is limn
                                else "WORKBENCH_DCNET_PYTHON",
                                "packages": ["stubdep==1.2.3"]}
    import workbench.backend.adapters.base as adapter_base

    monkeypatch.setattr(adapter_base, "model_by_id",
                        lambda model_id, registry=None: limn if model_id == "limn" else dcnet)

    request = EvalRequest("limn", "base_iter0_all_categories", "fashioniq_val_split",
                          tmp_path, tmp_path / "o.json")
    limn_command = LIMNAdapter().command(tmp_path / "LIMN", tmp_path / "ck", request)
    assert limn_command[0] == str(interpreter)

    dcnet_request = EvalRequest("dcnet", "fashioniq_run_directory",
                                "fashioniq_full_gallery_ref_excluded", tmp_path, tmp_path / "o.json")
    dcnet_command = DCNetAdapter().command(tmp_path / "DCNet", tmp_path / "run", dcnet_request)
    assert dcnet_command[0] == str(interpreter)


# ------------------------------------------------------------------------- INT-07..INT-08

def test_int07_missing_dependency_blocks_command_construction(tmp_path: Path, monkeypatch):
    interpreter = make_venv(tmp_path)
    monkeypatch.setenv("INT_07_VAR", str(interpreter))
    model = model_for(interpreter, variable="INT_07_VAR", packages=["stubdep==1.2.3"])
    forget_environment()

    report = runtime.verify_environment(model, probe=True)
    assert report.status == "DEPENDENCY_MISSING"
    with pytest.raises(RuntimeError, match="not verified"):
        python_executable(model)
    assert environment_blockers(model), "a missing dependency must surface as a blocker"


def test_int08_wrong_dependency_version_blocks_readiness(tmp_path: Path, monkeypatch):
    interpreter = make_venv(tmp_path)
    install_stub(interpreter, "stubdep", "1.12.10")
    monkeypatch.setenv("INT_08_VAR", str(interpreter))
    model = model_for(interpreter, variable="INT_08_VAR", packages=["stubdep==1.12.1"])
    forget_environment()

    report = runtime.verify_environment(model, probe=True)
    assert report.status == "DEPENDENCY_VERSION_MISMATCH"
    assert python_executable is not None
    with pytest.raises(RuntimeError, match="not verified"):
        python_executable(model)


def test_int08b_wrong_python_version_detected(tmp_path: Path, monkeypatch):
    """ENV-12/ENV-13: an incorrect declared Python version is a mismatch, not a pass."""
    interpreter = make_venv(tmp_path)
    install_stub(interpreter, "stubdep", "1.2.3")
    monkeypatch.setenv("INT_08B_VAR", str(interpreter))
    model = model_for(interpreter, variable="INT_08B_VAR", python_version="2.7.18",
                      packages=["stubdep==1.2.3"])
    forget_environment()

    report = runtime.verify_environment(model, probe=True)
    assert report.status == "PYTHON_VERSION_MISMATCH", report.reason
    with pytest.raises(RuntimeError):
        python_executable(model)


# ------------------------------------------------------------------------------- INT-09

def test_int09_gpu_verification_deferred_on_cpu_laptop(tmp_path: Path, monkeypatch):
    interpreter = make_venv(tmp_path)
    install_stub(interpreter, "stubdep", "1.2.3")
    monkeypatch.setenv("INT_09_VAR", str(interpreter))
    model = model_for(interpreter, variable="INT_09_VAR", packages=["stubdep==1.2.3"])
    model["environment"]["cuda"] = "12.4"
    forget_environment()

    report = runtime.verify_environment(model, probe=True)
    has_cuda = False
    try:
        import torch  # noqa: F401

        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False

    if has_cuda:
        assert report.tier == "CUDA_VERIFIED"
    else:
        assert report.tier == "GPU_VERIFICATION_DEFERRED", report.tier
        assert report.status == "GPU_VERIFICATION_DEFERRED"
        with pytest.raises(RuntimeError, match="GPU verification deferred"):
            require_verified_model_interpreter(model, require_cuda=True)
    status = environment_status(model, probe=True)
    assert status["gpu_deferred"] == (report.tier == "GPU_VERIFICATION_DEFERRED")


# ------------------------------------------------------------------ verification reuse

def test_verified_environment_is_reused_and_content_change_invalidates_cache(
        tmp_path: Path, monkeypatch):
    interpreter = make_venv(tmp_path)
    install_stub(interpreter, "stubdep", "1.2.3")
    monkeypatch.setenv("CACHE_VAR", str(interpreter))
    model = model_for(interpreter, variable="CACHE_VAR", packages=["stubdep==1.2.3"])
    forget_environment()

    first = runtime.verify_environment(model, probe=True)
    assert first.tier == "RUNTIME_READY"
    assert first.environment_fingerprint
    second = runtime.verify_environment(model, probe=True)
    assert second is first, "an unchanged, still-valid verification may be reused"

    # Removing the dependency from the environment must invalidate the cached result.
    for lib in (interpreter.parent.parent / "lib").glob("python*"):
        for dist in (lib / "site-packages").glob("stubdep-*.dist-info"):
            dist.rename(dist.with_suffix(".removed"))
    third = runtime.verify_environment(model, probe=True)
    assert third.status == "DEPENDENCY_MISSING", "cache must not outlive the environment"

    # Changing the declared contract must invalidate it too.
    model["environment"]["packages"] = ["stubdep==9.9.9"]
    fourth = runtime.verify_environment(model, probe=True)
    assert fourth.status in ("DEPENDENCY_MISSING", "DEPENDENCY_VERSION_MISMATCH")


def test_cache_is_not_keyed_on_interpreter_path_alone(tmp_path: Path, monkeypatch):
    """Two environments sharing one binary must never share verification results."""
    first = make_venv(tmp_path, "env_a")
    second = make_venv(tmp_path, "env_b")
    install_stub(first, "stubdep", "1.2.3")
    monkeypatch.setenv("SHARED_A", str(first))
    monkeypatch.setenv("SHARED_B", str(second))
    forget_environment()

    model_a = model_for(first, variable="SHARED_A", packages=["stubdep==1.2.3"])
    model_b = model_for(second, variable="SHARED_B", packages=["stubdep==1.2.3"])

    assert runtime.verify_environment(model_a, probe=True).tier == "RUNTIME_READY"
    report_b = runtime.verify_environment(model_b, probe=True)
    assert report_b.tier != "RUNTIME_READY", "identical paths must not imply identical environments"
    assert report_b.interpreter == str(second)


# ----------------------------------------------------------------------------- VER-01..VER-06

@pytest.mark.parametrize("requirement,installed,expected", [
    ("==1.12.1", "1.12.1", True),
    ("==1.12.1", "1.12", False),
    ("==1.12.1", "1.12.10", False),
    ("==1.12.1", "1.1", False),
    (">=1.12,<1.13", "1.12.1", True),
    (">=1.12,<1.13", "1.13.0", False),
    ("==2.20.0", "2.20.0", True),
    ("==2.20.0", "2.21.0", False),
    ("~=2.20.0", "2.20.5", True),
    ("~=2.20.0", "2.21.0", False),
])
def test_ver01_ver02_ver03_semantic_version_matching(requirement, installed, expected):
    assert version_satisfies(requirement, installed) is expected


def test_ver04_malformed_versions_and_constraints_are_not_accepted():
    assert version_satisfies("garbage-specifier", "1.0") is None
    assert version_satisfies(">=1.0", "not-a-version") is None
    assert version_satisfies("==1.0", None) is False


def test_ver05_missing_dependency_detected(tmp_path: Path, monkeypatch):
    interpreter = make_venv(tmp_path)
    monkeypatch.setenv("VER_05_VAR", str(interpreter))
    model = model_for(interpreter, variable="VER_05_VAR", packages=["absentdep==1.0.0"])
    forget_environment()
    report = runtime.verify_environment(model, probe=True)
    assert report.status == "DEPENDENCY_MISSING"
    assert "absentdep" in (report.reason or "")


def test_ver06_unknown_source_requirement_stays_unverified(tmp_path: Path, monkeypatch):
    """An UNKNOWN declaration must never be silently treated as satisfied."""
    interpreter = make_venv(tmp_path)
    install_stub(interpreter, "stubdep", "1.2.3")
    monkeypatch.setenv("VER_06_VAR", str(interpreter))
    model = model_for(interpreter, variable="VER_06_VAR")
    model["environment"]["packages"] = ["stubdep"]
    forget_environment()

    report = runtime.verify_environment(model, probe=True)
    assert report.status == "VERSION_CONSTRAINT_UNVERIFIED", report.status
    assert report.tier == "ENVIRONMENT_REQUIREMENTS_UNVERIFIED"
    assert environment_blockers(model), "an unverifiable constraint is a blocker, not a pass"


# ------------------------------------------------ manage_environment agreement (INT-04b)

def test_manage_environment_reports_verified_readiness_with_real_venv(tmp_path, monkeypatch):
    """manage_environment, doctor, and guarded_plan must share one readiness verdict."""
    import importlib.util

    from dataclasses import replace

    from workbench.backend.operator_config import resolve_config

    spec = importlib.util.spec_from_file_location(
        "manage_environment_rb", Path("workbench/scripts/manage_environment.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    interpreter = make_venv(tmp_path)
    install_stub(interpreter, "stubdep", "1.2.3")
    source_root = tmp_path / "third_party"
    (source_root / "ENV_PROBE_SRC").mkdir(parents=True)
    monkeypatch.setenv("MANAGE_ENV_PROBE_PYTHON", str(interpreter))
    config = replace(resolve_config(), WORKBENCH_THIRD_PARTY_ROOT=source_root)
    model = {"model_id": "env_probe", "environment_required": True, "source_dir": "ENV_PROBE_SRC",
             "environment": {"python_env_var": "MANAGE_ENV_PROBE_PYTHON",
                             "packages": ["stubdep==1.2.3"]}}
    forget_environment()

    status = module.status(model, config)
    assert status["interpreter"] == str(interpreter), "configured invocation path must be preserved"
    assert status["interpreter_tier"] == "RUNTIME_READY", status
    assert status["ready"] is True
    assert status["runtime_verified"] is True
    # The same model yields the same verdict through the shared runtime path.
    assert runtime.verify_environment(model, probe=True).tier == "RUNTIME_READY"

    # A missing dependency must flip manage_environment to blocked as well.
    for lib in (interpreter.parent.parent / "lib").glob("python*"):
        for dist in (lib / "site-packages").glob("stubdep-*.dist-info"):
            dist.rename(dist.with_suffix(".removed"))
    forget_environment()
    blocked = module.status(model, config)
    assert blocked["ready"] is False
    assert blocked["interpreter_status"] == "DEPENDENCY_MISSING", blocked


# ------------------------------------------- interpreter identity cannot be forged (F1/F2)

def test_a_wrapper_script_is_never_a_model_interpreter(tmp_path: Path):
    """A shell wrapper can reflect argv; it must still never pass as CPython."""
    spoof = tmp_path / "spoof"
    (spoof / "bin").mkdir(parents=True)
    (spoof / "pyvenv.cfg").write_text("home = /usr\n")
    (spoof / "lib" / "python3.13" / "site-packages").mkdir(parents=True)
    script = spoof / "bin" / "python"
    payload = ('{"executable": "%s", "prefix": "%s", "base_prefix": "/usr", '
               '"version": "3.13.12", "implementation": "cpython", "nonce": "%s"}')
    script.write_text("#!/bin/sh\n"
                      f"printf '{payload}\\\\n' \"{script}\" \"{spoof}\" \"$3\"\n")
    script.chmod(0o755)

    ok, reason = check_environment_isolation(script)
    assert ok is False, "a shell wrapper must never be accepted as a Python environment"
    assert "identity" in reason or "CPython" in reason


def test_relocated_interpreter_with_forged_markers_is_rejected(tmp_path: Path):
    """Hand-written venv markers around a relocated system interpreter are not an env."""
    dressed = tmp_path / "dressed"
    (dressed / "bin").mkdir(parents=True)
    (dressed / "lib" / "python3.13").mkdir(parents=True)
    (dressed / "pyvenv.cfg").write_text("home = /usr\n")
    interpreter = dressed / "bin" / "python"
    interpreter.symlink_to(Path(sys.executable).resolve())

    ok, reason = check_environment_isolation(interpreter)
    assert ok is False, "a relocated interpreter with hand-written markers is not an environment"
    assert any(token in reason for token in
               ("installed Python environment", "system", "identity", "markers", "isolated"))


def test_pyvenv_cfg_must_name_its_base_interpreter(tmp_path: Path):
    import venv as _venv

    env_dir = tmp_path / "env"
    _venv.EnvBuilder(with_pip=False, symlinks=True).create(env_dir)
    (env_dir / "pyvenv.cfg").write_text("include-system-site-packages = false\n")

    ok, reason = check_environment_isolation(env_dir / "bin" / "python")
    assert ok is False
    assert "pyvenv.cfg" in reason


def test_chained_venv_identity_is_never_taken_from_the_chain(tmp_path: Path):
    """Identity comes from the environment the interpreter runs as, and must be complete."""
    import venv as _venv

    first, second = tmp_path / "venv_a", tmp_path / "venv_b"
    _venv.EnvBuilder(with_pip=False, symlinks=True).create(first)
    _venv.EnvBuilder(with_pip=False, symlinks=True).create(second)
    (first / "bin" / "python").unlink()
    (first / "bin" / "python").symlink_to(second / "bin" / "python")

    ok, reason = check_environment_isolation(first / "bin" / "python")
    assert ok is True, reason
    assert runtime._interpreter_identity(first / "bin" / "python")["prefix"] == str(first)

    (second / "lib").rename(second / "lib_moved")
    broken = check_environment_isolation(second / "bin" / "python")
    assert broken[0] is False, "an environment with no library tree must be rejected"


# --------------------------------------------- cache identity is per model (F6) and per content (F5)

def test_cache_entries_are_not_shared_between_models(tmp_path, monkeypatch):
    """Two models sharing a variable and contract must not share the cached report."""
    interpreter = make_venv(tmp_path)
    install_stub(interpreter, "stubdep", "1.2.3")
    monkeypatch.setenv("SHARED_MODEL_VAR", str(interpreter))
    forget_environment()
    contract = {"python_env_var": "SHARED_MODEL_VAR", "packages": ["stubdep==1.2.3"]}
    alpha = {"model_id": "model_alpha", "environment_required": True, "environment": dict(contract)}
    beta = {"model_id": "model_beta", "environment_required": True, "environment": dict(contract)}

    first = runtime.verify_environment(alpha, probe=True)
    second = runtime.verify_environment(beta, probe=True)
    assert first is not second, "cache must be keyed per model"
    assert second.model_id == "model_beta"


def test_metadata_rewrite_in_place_invalidates_cache(tmp_path, monkeypatch):
    """A force-reinstall can rewrite metadata without touching directory mtimes."""
    interpreter = make_venv(tmp_path)
    install_stub(interpreter, "stubdep", "1.2.3")
    monkeypatch.setenv("INPLACE_VAR", str(interpreter))
    model = model_for(interpreter, variable="INPLACE_VAR", packages=["stubdep==1.2.3"])
    forget_environment()

    assert runtime.verify_environment(model, probe=True).tier == "RUNTIME_READY"
    for lib in (interpreter.parent.parent / "lib").glob("python*"):
        for dist in (lib / "site-packages").glob("stubdep-*.dist-info"):
            (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: stubdep\nVersion: 9.9.9\n")
    refreshed = runtime.verify_environment(model, probe=True)
    assert refreshed.status == "DEPENDENCY_VERSION_MISMATCH", "rewritten metadata must be noticed"


def test_a_declared_interpreter_digest_is_enforced(tmp_path, monkeypatch):
    """A pinned interpreter digest is how a launcher is excluded by policy.

    An orchestrator cannot distinguish a genuine CPython from a program that speaks the
    same protocol, because the program *is* what it executes. The registry therefore
    records the interpreter digest an operator pinned, and a mismatch is rejected.
    """
    interpreter = make_venv(tmp_path)
    install_stub(interpreter, "stubdep", "1.2.3")
    monkeypatch.setenv("DIGEST_VAR", str(interpreter))
    digest = runtime._interpreter_digest(interpreter)
    assert digest

    matching = {"model_id": "m", "environment_required": True,
                "environment": {"python_env_var": "DIGEST_VAR", "packages": ["stubdep==1.2.3"],
                                "interpreter_sha256": digest}}
    forget_environment()
    assert runtime.verify_environment(matching, probe=True).tier == "RUNTIME_READY"

    mismatched = {"model_id": "m", "environment_required": True,
                  "environment": {"python_env_var": "DIGEST_VAR", "packages": ["stubdep==1.2.3"],
                                  "interpreter_sha256": "0" * 64}}
    forget_environment()
    report = runtime.verify_environment(mismatched, probe=True)
    assert report.status == "UNSAFE_INTERPRETER", report.status
    assert "digest mismatch" in (report.reason or "")
    with pytest.raises(RuntimeError):
        runtime.python_executable(mismatched)


def test_an_unpinned_interpreter_digest_is_not_required(tmp_path, monkeypatch):
    """UNKNOWN/absent digest means unpinned; the environment is judged on its merits."""
    interpreter = make_venv(tmp_path)
    install_stub(interpreter, "stubdep", "1.2.3")
    monkeypatch.setenv("UNPINNED_VAR", str(interpreter))
    for value in (None, "UNKNOWN"):
        environment = {"python_env_var": "UNPINNED_VAR", "packages": ["stubdep==1.2.3"]}
        if value is not None:
            environment["interpreter_sha256"] = value
        model = {"model_id": "m", "environment_required": True, "environment": environment}
        forget_environment()
        assert runtime.verify_environment(model, probe=True).tier == "RUNTIME_READY"


def test_distribution_names_are_normalized_on_both_sides(tmp_path, monkeypatch):
    """PEP 503 normalization: '-' and '_' spellings must reconcile identically."""
    interpreter = make_venv(tmp_path)
    for lib in (interpreter.parent.parent / "lib").glob("python*"):
        site_packages = lib / "site-packages"
        dist = site_packages / "annotated_doc-0.0.4.dist-info"
        dist.mkdir(parents=True, exist_ok=True)
        # The directory spelling and the METADATA spelling deliberately differ.
        (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: annotated-doc\nVersion: 0.0.4\n")
        break
    monkeypatch.setenv("NORMALIZE_VAR", str(interpreter))

    for declared in ("annotated_doc==0.0.4", "annotated-doc==0.0.4", "Annotated.Doc==0.0.4"):
        model = {"model_id": "m", "environment_required": True,
                 "environment": {"python_env_var": "NORMALIZE_VAR", "packages": [declared]}}
        forget_environment()
        report = runtime.verify_environment(model, probe=True)
        assert report.tier == "RUNTIME_READY", f"{declared}: {report.status} {report.reason}"


def test_distribution_name_normalization_is_pep503():
    assert runtime.normalize_distribution_name("PyYAML") == "pyyaml"
    assert runtime.normalize_distribution_name("annotated_doc") == "annotated-doc"
    assert runtime.normalize_distribution_name("Annotated.Doc") == "annotated-doc"
    assert runtime.normalize_distribution_name("open_clip_torch") == "open-clip-torch"
    assert runtime.normalize_distribution_name("comet_ml") == "comet-ml"


def test_an_environment_cannot_fabricate_a_distribution_while_being_probed(tmp_path, monkeypatch):
    """The on-disk inventory is snapshotted before the environment runs any code."""
    interpreter = make_venv(tmp_path)
    for lib in (interpreter.parent.parent / "lib").glob("python*"):
        site_packages = lib / "site-packages"
        site_packages.joinpath("sitecustomize.py").write_text(
            "import pathlib\n"
            "site = pathlib.Path(__file__).parent\n"
            "dist = site / 'torch-2.0.1.dist-info'\n"
            "dist.mkdir(exist_ok=True)\n"
            "(dist / 'METADATA').write_text('Metadata-Version: 2.1\\nName: torch\\nVersion: 2.0.1\\n')\n"
            "import importlib.metadata as m\n"
            "_original = m.version\n"
            "m.version = lambda n: '2.0.1' if n == 'torch' else _original(n)\n")
        break
    monkeypatch.setenv("FABRICATE_VAR", str(interpreter))
    model = {"model_id": "m", "environment_required": True,
             "environment": {"python_env_var": "FABRICATE_VAR", "packages": ["torch==2.0.1"]}}
    forget_environment()
    report = runtime.verify_environment(model, probe=True)

    assert report.tier != "RUNTIME_READY", "a self-fabricated distribution must not verify"
    assert report.status in ("DEPENDENCY_INSTALLATION_UNVERIFIED", "DEPENDENCY_PROBE_FORGED"), report.status
    with pytest.raises(RuntimeError):
        runtime.python_executable(model)


def test_editable_style_install_is_never_ready(tmp_path, monkeypatch):
    """An editable/.pth install has no local dist-info, so it is never verified.

    The dependency probe runs under ``-I``, which ignores ``.pth``-injected paths and
    the user site, so a requirement visible only through an editable install is simply
    not found: the requirement stays unresolved and the environment never reaches
    RUNTIME_READY. Either outcome is acceptable; readiness is not.
    """
    import venv as _venv

    import workbench.backend.runtime as runtime

    env_dir = tmp_path / "env"
    _venv.EnvBuilder(with_pip=False, symlinks=True).create(env_dir)
    site_packages = next((env_dir / "lib").glob("python*/site-packages"))
    project = tmp_path / "checked_out_project"
    project.mkdir()
    (site_packages / "editable.pth").write_text(f"{project}\\n")
    (site_packages / "sitecustomize.py").write_text(
        "import importlib.metadata as m\\n"
        "_original = m.version\\n"
        "m.version = lambda n: '1.0.0' if n == 'myproj' else _original(n)\\n")

    monkeypatch.setenv("EDITABLE_VAR", str(env_dir / "bin" / "python"))
    model = {"model_id": "m", "environment_required": True,
             "environment": {"python_env_var": "EDITABLE_VAR", "packages": ["myproj==1.0.0"]}}
    runtime.forget_environment()
    report = runtime.verify_environment(model, probe=True)

    # Nothing on disk corroborates the requirement, so it is never accepted.
    assert report.tier != "RUNTIME_READY", report.status
    assert report.status in ("DEPENDENCY_INSTALLATION_UNVERIFIED", "DEPENDENCY_MISSING")
    with pytest.raises(RuntimeError, match="not verified"):
        runtime.python_executable(model)


def test_disk_contradiction_is_never_ready(tmp_path, monkeypatch):
    """A probe that contradicts an installed distribution is never accepted."""
    import venv as _venv

    import workbench.backend.runtime as runtime

    env_dir = tmp_path / "env"
    _venv.EnvBuilder(with_pip=False, symlinks=True).create(env_dir)
    site_packages = next((env_dir / "lib").glob("python*/site-packages"))
    dist = site_packages / "stubdep-1.2.3.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text("Metadata-Version: 2.1\\nName: stubdep\\nVersion: 1.2.3\\n")
    # The environment insists it has a different version than the one on disk.
    (site_packages / "sitecustomize.py").write_text(
        "import importlib.metadata as m\\n"
        "_original = m.version\\n"
        "m.version = lambda n: '9.9.9' if n == 'stubdep' else _original(n)\\n")

    monkeypatch.setenv("CONTRADICTION_VAR", str(env_dir / "bin" / "python"))
    model = {"model_id": "m", "environment_required": True,
             "environment": {"python_env_var": "CONTRADICTION_VAR", "packages": ["stubdep==9.9.9"]}}
    runtime.forget_environment()
    report = runtime.verify_environment(model, probe=True)

    assert report.tier != "RUNTIME_READY", report.status
    assert report.status in ("DEPENDENCY_PROBE_FORGED", "DEPENDENCY_MISSING",
                             "DEPENDENCY_VERSION_MISMATCH"), report.status
    with pytest.raises(RuntimeError, match="not verified"):
        runtime.python_executable(model)
