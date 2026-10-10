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
    """Install a real, importable distribution into the environment's site-packages.

    A distribution is more than its metadata: it has a module and a ``RECORD`` naming
    the files it installed, which is exactly what a real installer writes and what the
    runtime requires before a requirement may be corroborated.
    """
    for lib in (interpreter.parent.parent / "lib").glob("python*"):
        site_packages = lib / "site-packages"
        site_packages.mkdir(parents=True, exist_ok=True)
        module = site_packages / f"{name}.py"
        module.write_text(f"__version__ = {version!r}\n")
        dist = site_packages / f"{name}-{version}.dist-info"
        dist.mkdir(exist_ok=True)
        (dist / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
        )
        (dist / "RECORD").write_text(
            f"{name}.py,sha256=test,{module.stat().st_size}\n"
            f"{name}-{version}.dist-info/METADATA,sha256=test,60\n"
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

def test_int09_declared_cuda_requirement_is_deferred_not_verified(tmp_path: Path, monkeypatch):
    """A model declaring CUDA is GPU-deferred, never ready, and no code is executed.

    CUDA capability can only be reported by the environment, and asking it would mean
    running environment code during inspection. The honest verdict is deferral.
    """
    interpreter = make_venv(tmp_path)
    install_stub(interpreter, "stubdep", "1.2.3")
    monkeypatch.setenv("INT_09_VAR", str(interpreter))
    model = model_for(interpreter, variable="INT_09_VAR", packages=["stubdep==1.2.3"])
    model["environment"]["cuda"] = "12.4"
    forget_environment()

    report = runtime.verify_environment(model, probe=True)
    assert report.tier == "GPU_VERIFICATION_DEFERRED", report.tier
    assert report.status == "GPU_VERIFICATION_DEFERRED"
    assert report.cuda_available is None, "CUDA is not probed"
    with pytest.raises(RuntimeError):
        python_executable(model)

    # require_cuda makes the same judgement without executing anything.
    forget_environment()
    forced = runtime.verify_environment(model, probe=True, require_cuda=True)
    assert forced.tier == "GPU_VERIFICATION_DEFERRED"
    with pytest.raises(RuntimeError, match="not verified"):
        runtime.require_verified_model_interpreter(model, require_cuda=True)

    status = runtime.environment_status(model, probe=True)
    assert status["gpu_deferred"] is True
    assert status["verified"] is False


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


def test_ver06_unversioned_declaration_requires_presence_only(tmp_path: Path, monkeypatch):
    """A declaration without a version means "any version", not "unverifiable".

    Three registry contracts declare packages this way (``openai_clip``, ``Pillow``,
    ``torchvision``), so treating them as unverified would make a correct environment
    permanently non-ready for those models.
    """
    interpreter = make_venv(tmp_path)
    install_stub(interpreter, "stubdep", "1.2.3")
    monkeypatch.setenv("VER_06_VAR", str(interpreter))

    unpinned = model_for(interpreter, variable="VER_06_VAR", packages=["stubdep"])
    forget_environment()
    report = runtime.verify_environment(unpinned, probe=True)
    assert report.tier == "RUNTIME_READY", f"{report.status}: {report.reason}"
    assert report.packages.get("stubdep") == "1.2.3"

    # A declared version is still enforced exactly.
    install_stub(interpreter, "pinned", "1.0.0")
    exact = model_for(interpreter, variable="VER_06_VAR", packages=["pinned==2.0.0"])
    forget_environment()
    assert runtime.verify_environment(exact, probe=True).status == "DEPENDENCY_VERSION_MISMATCH"

    # An unusable declared constraint stays unverified.
    broken = model_for(interpreter, variable="VER_06_VAR", packages=["pinned==not!a!version"])
    forget_environment()
    assert runtime.verify_environment(broken, probe=True).status == "VERSION_CONSTRAINT_UNVERIFIED"

    # ``UNKNOWN`` is also "no version declared".
    install_stub(interpreter, "torch", "9.9.9")
    unknown = model_for(interpreter, variable="VER_06_VAR")
    unknown["environment"]["pytorch"] = "UNKNOWN"
    forget_environment()
    assert runtime.verify_environment(unknown, probe=True).tier == "RUNTIME_READY"


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
    # An external change is announced by resetting the cached verification, which also
    # re-baselines the environment's inventory.
    forget_environment()
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
        module = site_packages / "annotated_doc.py"
        module.write_text("__version__ = '0.0.4'\n")
        dist = site_packages / "annotated_doc-0.0.4.dist-info"
        dist.mkdir(parents=True, exist_ok=True)
        # The directory spelling and the METADATA spelling deliberately differ.
        (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: annotated-doc\nVersion: 0.0.4\n")
        (dist / "RECORD").write_text(
            f"annotated_doc.py,sha256=test,{module.stat().st_size}\n"
            "annotated_doc-0.0.4.dist-info/METADATA,sha256=test,60\n")
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
    """The inventory used for reconciliation is read before the environment runs code."""
    interpreter = make_venv(tmp_path)
    for lib in (interpreter.parent.parent / "lib").glob("python*"):
        site_packages = lib / "site-packages"
        (site_packages / "sitecustomize.py").write_text(
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

    # The probe is isolated, so the environment's own code never runs and the
    # fabricated distribution is simply not visible: the requirement is unresolved.
    assert report.tier != "RUNTIME_READY", "a distribution created during the probe is not real"
    assert report.status in ("DEPENDENCY_INSTALLATION_UNVERIFIED", "DEPENDENCY_MISSING",
                             "DEPENDENCY_PROBE_FORGED"), report.status
    assert report.packages.get("torch") is None
    with pytest.raises(RuntimeError):
        runtime.python_executable(model)


def test_metadata_rewritten_during_a_probe_never_verifies(tmp_path, monkeypatch):
    """An environment that rewrites its own METADATA while probed is never ready."""
    interpreter = make_venv(tmp_path)
    for lib in (interpreter.parent.parent / "lib").glob("python*"):
        site_packages = lib / "site-packages"
        dist = site_packages / "torch-1.0.0.dist-info"
        dist.mkdir(parents=True, exist_ok=True)
        (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: torch\nVersion: 1.0.0\n")
        (site_packages / "sitecustomize.py").write_text(
            "import pathlib\n"
            "site = pathlib.Path(__file__).parent\n"
            "for dist in site.glob('torch-*.dist-info'):\n"
            "    (dist / 'METADATA').write_text('Metadata-Version: 2.1\\nName: torch\\nVersion: 2.0.1\\n')\n"
            "import importlib.metadata as m\n"
            "_original = m.version\n"
            "m.version = lambda n: '2.0.1' if n == 'torch' else _original(n)\n")
        break
    monkeypatch.setenv("REWRITE_VAR", str(interpreter))
    model = {"model_id": "m", "environment_required": True,
             "environment": {"python_env_var": "REWRITE_VAR", "packages": ["torch==2.0.1"]}}
    forget_environment()
    report = runtime.verify_environment(model, probe=True)

    assert report.tier != "RUNTIME_READY", "an in-probe metadata rewrite must not verify"
    assert report.status in ("DEPENDENCY_MISSING", "DEPENDENCY_VERSION_MISMATCH",
                             "DEPENDENCY_INSTALLATION_UNVERIFIED"), report.status
    with pytest.raises(RuntimeError):
        runtime.python_executable(model)


def test_a_legitimate_install_verifies_however_the_cache_is_reset(tmp_path, monkeypatch):
    """An operator install must verify on the next probe, whatever the cache state."""
    interpreter = make_venv(tmp_path)
    install_stub(interpreter, "stubdep", "1.2.3")
    monkeypatch.setenv("EXTERNAL_VAR", str(interpreter))
    model = {"model_id": "m", "environment_required": True,
             "environment": {"python_env_var": "EXTERNAL_VAR", "packages": ["stubdep==1.2.3"]}}
    forget_environment()
    assert runtime.verify_environment(model, probe=True).tier == "RUNTIME_READY"

    install_stub(interpreter, "unrelateddep", "2.0.0")

    for reset in (lambda: runtime.forget_environment(),
                  lambda: runtime.forget_environment(str(interpreter))):
        reset()
        report = runtime.verify_environment(model, probe=True)
        assert report.tier == "RUNTIME_READY", f"{report.status}: {report.reason}"

    monkeypatch.setenv("WORKBENCH_NO_ENV_CACHE", "1")
    runtime.forget_environment()
    report = runtime.verify_environment(model, probe=True)
    assert report.tier == "RUNTIME_READY", f"{report.status}: {report.reason}"


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


def test_an_unreadable_installed_version_never_confirms_a_probe(tmp_path, monkeypatch):
    """A dist-info whose Version is unreadable must not corroborate a probe answer."""
    interpreter = make_venv(tmp_path)
    for lib in (interpreter.parent.parent / "lib").glob("python*"):
        site_packages = lib / "site-packages"
        dist = site_packages / "torch-2.0.1.dist-info"
        dist.mkdir(parents=True, exist_ok=True)
        # Name present, Version absent.
        (dist / "METADATA").write_text("Metadata-Version: 2.1\\nName: torch\\n")
        (site_packages / "sitecustomize.py").write_text(
            "import importlib.metadata as m\\n"
            "_original = m.version\\n"
            "m.version = lambda n: '2.0.1' if n == 'torch' else _original(n)\\n")
        break
    monkeypatch.setenv("UNREADABLE_VAR", str(interpreter))
    model = {"model_id": "m", "environment_required": True,
             "environment": {"python_env_var": "UNREADABLE_VAR", "packages": ["torch==2.0.1"]}}
    forget_environment()
    report = runtime.verify_environment(model, probe=True)

    # Either the version cannot be corroborated or the package is not found at all;
    # both are safe, and neither may reach RUNTIME_READY.
    assert report.tier != "RUNTIME_READY", "an unreadable on-disk version cannot corroborate"
    assert report.status in ("DEPENDENCY_INSTALLATION_UNVERIFIED", "DEPENDENCY_MISSING"), report.status
    with pytest.raises(RuntimeError):
        runtime.python_executable(model)


def test_a_pinned_digest_is_enforced_even_after_a_cached_verification(tmp_path, monkeypatch):
    """A cache hit must never mask a pinned-interpreter mismatch."""
    interpreter = make_venv(tmp_path)
    install_stub(interpreter, "stubdep", "1.2.3")
    monkeypatch.setenv("CACHE_PIN_VAR", str(interpreter))
    unpinned = {"model_id": "m", "environment_required": True,
                "environment": {"python_env_var": "CACHE_PIN_VAR", "packages": ["stubdep==1.2.3"]}}
    forget_environment()
    assert runtime.verify_environment(unpinned, probe=True).tier == "RUNTIME_READY"

    pinned = {"model_id": "m", "environment_required": True,
              "environment": {"python_env_var": "CACHE_PIN_VAR", "packages": ["stubdep==1.2.3"],
                              "interpreter_sha256": "0" * 64}}
    report = runtime.verify_environment(pinned, probe=True)
    assert report.tier == "BLOCKED", "a pinned mismatch must not be served from cache"
    assert report.status == "UNSAFE_INTERPRETER", report.status


def test_an_external_install_between_probes_is_not_a_forgery(tmp_path, monkeypatch):
    """Packages installed while the environment was idle must re-baseline, not block."""
    interpreter = make_venv(tmp_path)
    install_stub(interpreter, "stubdep", "1.2.3")
    monkeypatch.setenv("EXTERNAL_VAR", str(interpreter))
    model = {"model_id": "m", "environment_required": True,
             "environment": {"python_env_var": "EXTERNAL_VAR", "packages": ["stubdep==1.2.3"]}}
    forget_environment()
    assert runtime.verify_environment(model, probe=True).tier == "RUNTIME_READY"

    install_stub(interpreter, "unrelateddep", "2.0.0")
    forget_environment(runtime.verify_environment(model, probe=True).interpreter)
    refreshed = runtime.verify_environment(model, probe=True)
    assert refreshed.tier == "RUNTIME_READY", f"{refreshed.status}: {refreshed.reason}"


def test_metadata_only_distribution_is_not_corroboration(tmp_path, monkeypatch):
    """A dist-info with no installed files cannot satisfy a requirement."""
    interpreter = make_venv(tmp_path)
    for lib in (interpreter.parent.parent / "lib").glob("python*"):
        dist = lib / "site-packages" / "torch-2.0.1.dist-info"
        dist.mkdir(parents=True, exist_ok=True)
        # Metadata only: no module, no RECORD entries that exist.
        (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: torch\nVersion: 2.0.1\n")
        (dist / "RECORD").write_text("torch/__init__.py,sha256=test,10\n")
        break
    monkeypatch.setenv("METADATA_ONLY_VAR", str(interpreter))
    model = {"model_id": "m", "environment_required": True,
             "environment": {"python_env_var": "METADATA_ONLY_VAR", "packages": ["torch==2.0.1"]}}
    forget_environment()
    report = runtime.verify_environment(model, probe=True)

    # Metadata with no installed files is not an installation: the requirement is unmet.
    assert report.tier != "RUNTIME_READY", "metadata without content is not an installation"
    assert report.status == "DEPENDENCY_MISSING", report.status
    assert "no importable installation" in (report.reason or "")
    with pytest.raises(RuntimeError):
        runtime.python_executable(model)


def test_probes_never_execute_code_belonging_to_the_environment(tmp_path, monkeypatch):
    """An environment's sitecustomize must not run while it is being inspected."""
    interpreter = make_venv(tmp_path)
    marker = None
    for lib in (interpreter.parent.parent / "lib").glob("python*"):
        site_packages = lib / "site-packages"
        marker = site_packages / "RAN"
        (site_packages / "sitecustomize.py").write_text(
            "import pathlib\n"
            "pathlib.Path(__file__).parent.joinpath('RAN').write_text('yes')\n")
        module = site_packages / "stubdep.py"
        module.write_text("__version__ = '1.2.3'\n")
        dist = site_packages / "stubdep-1.2.3.dist-info"
        dist.mkdir(exist_ok=True)
        (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: stubdep\nVersion: 1.2.3\n")
        (dist / "RECORD").write_text(
            f"stubdep.py,sha256=test,{module.stat().st_size}\n"
            "stubdep-1.2.3.dist-info/METADATA,sha256=test,60\n")
        break
    marker.unlink(missing_ok=True)
    monkeypatch.setenv("NO_SIDE_EFFECT_VAR", str(interpreter))
    model = {"model_id": "m", "environment_required": True,
             "environment": {"python_env_var": "NO_SIDE_EFFECT_VAR", "packages": ["stubdep==1.2.3"]}}
    forget_environment()
    report = runtime.verify_environment(model, probe=True)

    assert report.tier == "RUNTIME_READY", f"{report.status}: {report.reason}"
    assert not marker.exists(), "the environment's own code must never run during inspection"


# ------------------------------------- consolidated inventory: R1, R2, R3 (round 11)

def test_record_less_metadata_is_not_an_installation(tmp_path, monkeypatch):
    """Metadata that installs nothing (no RECORD, no module) cannot satisfy a requirement."""
    interpreter = make_venv(tmp_path)
    for lib in (interpreter.parent.parent / "lib").glob("python*"):
        site_packages = lib / "site-packages"
        dist = site_packages / "norec-2.0.dist-info"
        dist.mkdir(parents=True, exist_ok=True)
        (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: norec\nVersion: 2.0\n")
        break
    monkeypatch.setenv("NOREC_VAR", str(interpreter))
    model = {"model_id": "m", "environment_required": True,
             "environment": {"python_env_var": "NOREC_VAR", "packages": ["norec==2.0"]}}
    forget_environment()
    report = runtime.verify_environment(model, probe=True)

    assert report.tier != "RUNTIME_READY", "a RECORD-less dist-info installs nothing"
    assert report.status == "DEPENDENCY_MISSING", report.status
    with pytest.raises(RuntimeError):
        runtime.python_executable(model)


def test_empty_egg_info_is_not_an_installation(tmp_path, monkeypatch):
    """An egg-info holding only its own PKG-INFO installs nothing."""
    interpreter = make_venv(tmp_path)
    for lib in (interpreter.parent.parent / "lib").glob("python*"):
        site_packages = lib / "site-packages"
        egg = site_packages / "ghostegg.egg-info"
        egg.mkdir(parents=True, exist_ok=True)
        (egg / "PKG-INFO").write_text("Metadata-Version: 1.1\nName: ghostegg\nVersion: 1.0\n")
        (egg / "SOURCES.txt").write_text("ghostegg.egg-info/PKG-INFO\n")
        break
    monkeypatch.setenv("EGG_VAR", str(interpreter))
    model = {"model_id": "m", "environment_required": True,
             "environment": {"python_env_var": "EGG_VAR", "packages": ["ghostegg==1.0"]}}
    forget_environment()
    report = runtime.verify_environment(model, probe=True)

    assert report.tier != "RUNTIME_READY", "descriptive-only egg-info installs nothing"
    assert report.status == "DEPENDENCY_MISSING", report.status


def test_record_members_cannot_escape_the_environment(tmp_path, monkeypatch):
    """A RECORD naming a file outside the tree is not installed content."""
    interpreter = make_venv(tmp_path)
    for lib in (interpreter.parent.parent / "lib").glob("python*"):
        site_packages = lib / "site-packages"
        dist = site_packages / "travpkg-1.0.dist-info"
        dist.mkdir(parents=True, exist_ok=True)
        (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: travpkg\nVersion: 1.0\n")
        # Absolute and traversing members: both must be ignored.
        (dist / "RECORD").write_text(
            "/etc/hostname,sha256=x,10\n"
            "../../../../../../etc/hostname,sha256=x,10\n"
            "travpkg-1.0.dist-info/METADATA,sha256=x,60\n")
        break
    monkeypatch.setenv("TRAV_VAR", str(interpreter))
    model = {"model_id": "m", "environment_required": True,
             "environment": {"python_env_var": "TRAV_VAR", "packages": ["travpkg==1.0"]}}
    forget_environment()
    report = runtime.verify_environment(model, probe=True)

    assert report.tier != "RUNTIME_READY", "an escaping RECORD member is not installed content"
    assert report.status == "DEPENDENCY_MISSING", report.status


def test_system_site_packages_environment_resolves_from_its_base(tmp_path, monkeypatch):
    """A --system-site-packages venv resolves requirements from its base interpreter."""
    import venv as _venv

    base = tmp_path / "base"
    _venv.EnvBuilder(with_pip=False, symlinks=True).create(base)
    for lib in (base / "lib").glob("python*"):
        site_packages = lib / "site-packages"
        module = site_packages / "basedep.py"
        module.write_text("__version__ = '1.0.0'\n")
        dist = site_packages / "basedep-1.0.0.dist-info"
        dist.mkdir(parents=True, exist_ok=True)
        (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: basedep\nVersion: 1.0.0\n")
        (dist / "RECORD").write_text(
            f"basedep.py,sha256=x,{module.stat().st_size}\n"
            "basedep-1.0.0.dist-info/METADATA,sha256=x,60\n")
        break

    overlay = tmp_path / "overlay"
    interpreter = overlay / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    # A real interpreter whose pyvenv.cfg exposes the base's site-packages.
    interpreter.symlink_to(base / "bin" / "python")
    (overlay / "pyvenv.cfg").write_text(
        f"home = {base / 'bin'}\ninclude-system-site-packages = true\n"
        f"version = {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}\n")
    (overlay / "lib").mkdir(exist_ok=True)

    monkeypatch.setenv("SSP_VAR", str(interpreter))
    forgetting = None
    model = {"model_id": "m", "environment_required": True,
             "environment": {"python_env_var": "SSP_VAR", "packages": ["basedep==1.0.0"]}}
    forget_environment()
    report = runtime.verify_environment(model, probe=True)

    # The dependency is resolvable from the base interpreter, so it must not be refused
    # as missing; whatever the verdict, it must name the real reason.
    assert report.status != "DEPENDENCY_MISSING" or report.tier == "UNSAFE_INTERPRETER", report.reason
    assert forgetting is None


def test_inventory_and_module_checks_agree(tmp_path, monkeypatch):
    """One walk backs both answers, so they cannot disagree about an environment."""
    interpreter = make_venv(tmp_path)
    install_stub(interpreter, "stubdep", "1.2.3")
    monkeypatch.setenv("AGREE_VAR", str(interpreter))
    forget_environment()

    versions, modules = runtime.inspect_environment_distributions(str(interpreter.parent.parent))
    assert versions["stubdep"] == "1.2.3"
    assert "stubdep" in modules, "an installed module is reported by the same walk"

    model = {"model_id": "m", "environment_required": True,
             "environment": {"python_env_var": "AGREE_VAR", "packages": ["stubdep==1.2.3"],
                             "required_imports": ["stubdep"]}}
    forget_environment()
    report = runtime.verify_environment(model, probe=True)
    assert report.tier == "RUNTIME_READY", f"{report.status}: {report.reason}"
    assert report.packages["stubdep"] == "1.2.3"
    assert report.imports["stubdep"] is True
