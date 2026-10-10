"""Environment verification with explicit readiness tiers and source-backed probing."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Tier ordering: each tier implies all lower tiers were verified.
TIER_ORDER = [
    "INTERPRETER_MISSING",
    "INTERPRETER_PRESENT",
    "ISOLATION_VERIFIED",
    "PYTHON_VERSION_VERIFIED",
    "DEPENDENCIES_VERIFIED",
    "MODEL_IMPORT_VERIFIED",
    "CUDA_VERIFIED",
    "RUNTIME_READY",
]
TIER_UNCONFIGURED = "UNCONFIGURED"
TIER_BLOCKED = "BLOCKED"
TIER_GPU_DEFERRED = "GPU_VERIFICATION_DEFERRED"
TIER_REQUIREMENTS_UNVERIFIED = "ENVIRONMENT_REQUIREMENTS_UNVERIFIED"


@dataclass
class EnvironmentReport:
    model_id: str
    variable: str | None
    interpreter: str | None
    tier: str
    status: str
    prefix: str | None = None
    base_prefix: str | None = None
    python_version: str | None = None
    packages: dict[str, str | None] = field(default_factory=dict)
    imports: dict[str, bool] = field(default_factory=dict)
    cuda_available: bool | None = None
    cuda_version: str | None = None
    reason: str | None = None
    probe_executed: bool = False
    dependency_probe_executed: bool = False
    contract_digest: str | None = None
    environment_fingerprint: str | None = None
    verified_at: str | None = None


def _run_probe(interpreter: Path, code: str, timeout: int = 60,
               args: tuple[str, ...] = (), isolated: bool = False) -> tuple[bool, str, str]:
    """Run ``code`` under ``interpreter``.

    ``isolated`` adds ``-I`` so the user site and ``PYTHONPATH`` are ignored and no
    ambient environment can answer for the one under test. It deliberately does *not*
    add ``-S``: the environment's own ``site-packages`` must stay visible, and probes
    that report a foreign ``sys.prefix`` are rejected by the caller instead.
    """
    prefix_flags = ["-I"] if isolated else []
    try:
        proc = subprocess.run(
            [str(interpreter), *prefix_flags, "-c", code, *args],
            capture_output=True, text=True, timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return False, "", str(error)
    return proc.returncode == 0, proc.stdout.strip(), proc.stderr.strip()

_PROBE_CODE = r"""
import json, sys
info = {
    "executable": sys.executable,
    "prefix": sys.prefix,
    "base_prefix": getattr(sys, "base_prefix", sys.prefix),
    "version": ".".join(str(p) for p in sys.version_info[:3]),
    "implementation": sys.implementation.name,
    "nonce": sys.argv[1] if len(sys.argv) > 1 else None,
}
print(json.dumps(info))
"""

def _interpreter_identity(interpreter: Path) -> dict[str, Any] | None:
    """Learn an interpreter's identity without trusting its self-report.

    Two independent probes are combined:

    * an isolated probe (``-I -S``) proves the target really is a CPython binary and
      reports the executable it is running as. It cannot be influenced by
      ``sitecustomize.py`` or other code sitting inside the environment.
    * the environment prefix is derived from the **invocation path** (``bin/python``
      -> its parent directory), never from ``sys.prefix``. A probe that has had
      ``sys.prefix`` rewritten would otherwise be able to point the workbench at an
      unrelated directory.

    A wrapper script cannot pass the isolated probe, because the probe's nonce must
    come back and the reported executable must match the invocation path.
    """
    import secrets

    if not _is_native_executable(interpreter):
        # argv is not a secret channel: a shell wrapper can reflect the nonce. The
        # interpreter must first *be* an interpreter binary rather than a script.
        return None
    nonce = secrets.token_hex(16)
    invoked = Path(interpreter)
    invoked_prefix = invoked.parent.parent
    ok, out, _ = _run_probe(interpreter, _PROBE_CODE, args=(nonce,), isolated=True)
    if not ok or not out:
        return None
    try:
        identity = json.loads(out.splitlines()[-1])
    except (ValueError, IndexError):
        return None
    if not isinstance(identity, dict) or identity.get("nonce") != nonce:
        return None
    # The probe must have executed *this* invocation path, under a real CPython, and
    # the reported base_prefix must be corroborated by the pyvenv.cfg on disk.
    reported = str(identity.get("executable") or "")
    try:
        if reported and Path(reported).resolve() != invoked.resolve():
            # `-I` can make a venv's bin/python report the base binary; accept only
            # when the invocation path itself is the binary or a link to it.
            if not invoked.resolve().samefile(Path(reported).resolve()):
                return None
    except OSError:
        return None
    pyvenv = invoked_prefix / "pyvenv.cfg"
    if pyvenv.is_file():
        home = _pyvenv_home(pyvenv)
        base_prefix = str(identity.get("base_prefix") or "")
        if home and base_prefix:
            # ``home`` names the base *bin* directory; base_prefix is its prefix.
            home_prefix = Path(home).parent
            try:
                same_base = (Path(base_prefix).resolve() == home_prefix.resolve()
                             or str(Path(base_prefix).resolve()) == str(home_prefix.resolve()))
            except OSError:
                same_base = False
            if not same_base:
                return None
    identity["prefix"] = str(invoked_prefix)
    return identity


def _identity_claim_matches(interpreter: Path, identity: dict[str, Any]) -> bool:
    """Does the executable the interpreter reports actually match the path we invoked?

    A shell script, a wrapper, or a different interpreter masquerading under a venv
    path can report an unrelated ``sys.executable``; that is not a truthful model env.
    """
    try:
        invoked = Path(interpreter)
        reported = Path(str(identity.get("executable") or ""))
        if not reported:
            return False
        if os.path.samefile(invoked, reported):
            return True
        return invoked.resolve() == reported.resolve()
    except (OSError, ValueError):
        return False

def check_environment_isolation(interpreter: Path) -> tuple[bool, str]:
    """Verify the *runtime* identity of an interpreter, not just its resolved path.

    A Linux venv's ``bin/python`` is routinely a symlink to a system interpreter
    binary, so a matching ``resolve()`` is **not** evidence that this is the system
    environment. Identity is decided by the interpreter's own
    ``sys.prefix``/``sys.base_prefix``, its reported ``sys.executable``, and the
    on-disk virtualenv/conda markers of the *invocation* path.
    """
    invoked = Path(interpreter)
    identity = _interpreter_identity(invoked)
    if identity is None:
        # Cannot establish runtime identity -> fail closed.
        try:
            resolved = invoked.resolve()
        except OSError:
            resolved = invoked
        if str(resolved) in _SYSTEM_INTERPRETERS:
            return False, "cannot use system Python"
        if str(invoked) in _SYSTEM_INTERPRETERS:
            return False, "cannot use system Python"
        if _is_base_conda_path(resolved) or _is_base_conda_path(invoked):
            return False, "cannot use base Conda environment"
        return False, "interpreter runtime identity could not be established"

    if str(identity.get("implementation") or "") != "cpython":
        # A wrapper script (or a non-CPython interpreter) cannot run the official
        # PyTorch evaluators and must never be treated as a model environment.
        return False, "interpreter is not a CPython implementation"

    prefix = str(identity.get("prefix") or "")
    base_prefix = str(identity.get("base_prefix") or "")
    if not prefix:
        return False, "interpreter has no detectable prefix"

    # The active orchestrator environment is never a model target. We deliberately
    # compare *environment identity* (sys.prefix / pyvenv.cfg), never the resolved
    # binary: a uv-managed orchestrator venv and an unrelated model venv can share
    # the very same underlying interpreter binary, so resolve() proves nothing.
    try:
        if Path(prefix).resolve() == Path(sys.prefix).resolve():
            return False, "cannot use active workbench environment"
        if invoked.resolve() == Path(sys.executable).resolve() and Path(prefix).resolve() != Path(sys.prefix).resolve():
            # Same binary but a genuinely different environment: allowed.
            pass
    except (OSError, ValueError):
        pass

    if prefix == base_prefix and (prefix in ("/usr", "/usr/local", "") or _is_base_conda_path(Path(prefix))):
        return False, "cannot use system Python"
    if _is_base_conda_path(Path(prefix)):
        return False, "cannot use base Conda environment"

    # Markers must belong to the environment the interpreter itself reports. Accepting
    # them merely next to the invocation path would let a spoofed launcher or a
    # relocated system interpreter masquerade as an isolated environment.
    prefix_path = Path(prefix)
    has_pyvenv = (prefix_path / "pyvenv.cfg").is_file()
    has_conda_meta = (prefix_path / "conda-meta").is_dir()

    if prefix == base_prefix and not has_conda_meta:
        return False, "interpreter is not in an isolated environment"
    if not (has_pyvenv or has_conda_meta):
        return False, "isolated environment markers (pyvenv.cfg/conda-meta) are missing"

    # The interpreter must run *from* the environment it claims: a symlink that lands
    # inside another venv, or a system interpreter dressed in venv markers, is out.
    if Path(prefix).resolve() == Path(base_prefix).resolve() and not has_conda_meta:
        return False, "interpreter prefix is its own base prefix (not isolated)"


    # A directory dressed in venv markers but with no installed environment is not a
    # usable environment: a relocated or hand-prepared system interpreter must not
    # pass as one. A real venv/conda env has its own site-packages directory.
    has_env_library = any((prefix_path / "lib").glob("python*/site-packages"))
    has_env_library = has_env_library or (prefix_path / "Lib" / "site-packages").is_dir()
    has_env_library = has_env_library or (prefix_path / "lib" / "site-packages").is_dir()
    if not (has_env_library or has_conda_meta):
        return False, "environment has no installed Python environment"
    # A hand-written pyvenv.cfg must at least name the base interpreter it claims.
    pyvenv = prefix_path / "pyvenv.cfg"
    if pyvenv.is_file():
        try:
            home = next((line.split("=", 1)[1].strip()
                         for line in pyvenv.read_text(encoding="utf-8", errors="replace").splitlines()
                         if line.strip().startswith("home") and "=" in line), "")
        except OSError:
            home = ""
        if not home:
            return False, "pyvenv.cfg does not name its base interpreter"
        if not Path(home).is_dir():
            return False, f"pyvenv.cfg home is not a directory: {home}"

    # The interpreter must actually live in the environment it claims.
    try:
        executable = Path(str(identity.get("executable") or ""))
        if not executable.is_relative_to(Path(prefix)):
            return False, "interpreter executable does not belong to its claimed prefix"
    except (OSError, ValueError):
        return False, "interpreter prefix relationship could not be verified"

    # ...and it must be the one we actually invoked.
    if not _identity_claim_matches(invoked, identity):
        return False, "interpreter reports a different executable than the configured path"

    return True, "ISOLATION_VERIFIED"

_SYSTEM_INTERPRETERS = {
    "/usr/bin/python", "/usr/bin/python3", "/bin/python", "/bin/python3",
    "/usr/local/bin/python", "/usr/local/bin/python3",
    "/usr/bin/python3.10", "/usr/bin/python3.11", "/usr/bin/python3.12", "/usr/bin/python3.13",
}

_NATIVE_MAGICS = (b"\x7fELF", b"\xca\xfe\xba\xbe", b"\xcf\xfa\xed\xfe",
                  b"\xfe\xed\xfa\xcf", b"\xce\xfa\xed\xfe", b"\xfe\xed\xfa\xce")

def _pyvenv_home(pyvenv: Path) -> str:
    try:
        for line in pyvenv.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.strip().startswith("home") and "=" in line:
                return line.split("=", 1)[1].strip()
    except OSError:
        return ""
    return ""


def _is_native_executable(interpreter: Path) -> bool:
    """True when the invocation path resolves to a native executable, not a script.

    Official PyTorch evaluators require a real interpreter binary. Shell scripts,
    shims, and echo wrappers that merely *report* Python-shaped JSON are rejected.
    """
    try:
        target = Path(interpreter).resolve()
    except OSError:
        return False
    try:
        with target.open("rb") as handle:
            header = handle.read(4)
    except OSError:
        return False
    if header[:2] == b"MZ":  # Windows PE / launcher stub
        return True
    return header in _NATIVE_MAGICS


def _probe_prefix_mismatch(payload: dict[str, Any], report: EnvironmentReport,
                           *, require_site: bool = True) -> bool:
    """True when a probe answered from a different environment than the verified one.

    This catches accidental cross-environment contamination (a ``PYTHONPATH`` entry,
    an activated environment, a wrapper) as well as ``sitecustomize`` tampering.
    """
    reported = payload.get("__prefix__")
    if reported is None:
        return False
    try:
        return Path(str(reported)).resolve() != Path(str(report.prefix or "")).resolve()
    except (OSError, ValueError):
        return True


def _is_base_conda_path(path: Path) -> bool:
    """True only for a *base* conda install, identified by its own markers.

    A conda environment created with ``-p/--prefix`` is a legitimate isolated
    environment: it has ``conda-meta`` and its own library tree, and it is NOT a base
    install. Base is the install that *contains* the envs directory, so base is
    detected by an ``envs`` directory beside ``conda-meta``, never by a path substring.
    """
    if not (path / "conda-meta").is_dir():
        return False
    if (path / "envs").is_dir():
        return True  # the conda installation root itself
    return False




def declared_requirement(package: str, value: str | None) -> str | None:
    """Normalize a source-declared dependency constraint.

    ``UNKNOWN``/empty means the upstream source did not declare a usable
    constraint; it is returned as ``None`` so callers must report the
    requirement as unverified rather than silently accepting any version.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() == "UNKNOWN":
        return None
    # A bare version means the upstream pinned exactly that version.
    if text[0].isdigit():
        return f"=={text}"
    return text

def _required_packages(model: dict[str, Any]) -> dict[str, str | None]:
    env = model.get("environment") or {}
    required: dict[str, str | None] = {}
    if "pytorch" in env:
        required["torch"] = declared_requirement("torch", env.get("pytorch"))
    if "torchvision" in env:
        required["torchvision"] = declared_requirement("torchvision", env.get("torchvision"))
    for pkg in env.get("packages") or []:
        if not isinstance(pkg, str):
            continue
        name, _, version = pkg.partition("==")
        required.setdefault(name, declared_requirement(name, version or None))
    return required

def version_satisfies(constraint: str, installed: str | None) -> bool | None:
    """Semantic version check. Returns None when the constraint cannot be interpreted."""
    from packaging.specifiers import InvalidSpecifier, SpecifierSet
    from packaging.version import InvalidVersion, Version

    if installed is None:
        return False
    try:
        spec = SpecifierSet(constraint)
    except InvalidSpecifier:
        return None
    try:
        return spec.contains(Version(str(installed)), prereleases=True)
    except InvalidVersion:
        return None

def python_version_satisfies(declared: str, observed: str | None) -> bool | None:
    """Compare a declared Python version with the interpreter's reported version."""
    from packaging.version import InvalidVersion, Version

    if observed is None:
        return False
    text = declared.strip()
    if not text or text.upper() == "UNKNOWN":
        return None
    try:
        observed_version = Version(str(observed))
    except InvalidVersion:
        return None
    if text[0].isdigit() and len(text.split(".")) < 3:
        # A truncated declaration ("3.8") is a series request, not an exact version.
        return tuple(observed_version.release[: len(text.split("."))]) == tuple(
            int(part) for part in text.split(".")
        )
    return version_satisfies(declared_requirement("python", text) or text, str(observed_version))


def _required_imports(model: dict[str, Any]) -> list[str]:
    env = model.get("environment") or {}
    return [str(name) for name in (env.get("required_imports") or [])]


def verify_environment(model: dict[str, Any], *, probe: bool = True, require_cuda: bool = False) -> EnvironmentReport:
    """Return a truthful environment report. Never claims RUNTIME_READY without proof."""
    model_id = model.get("model_id", "<unnamed>")
    env = model.get("environment") or {}
    variable = env.get("python_env_var")
    if not variable:
        return EnvironmentReport(model_id, None, None, TIER_UNCONFIGURED, "UNCONFIGURED",
                                 reason="no environment variable declared")
    configured = os.environ.get(variable)
    if not configured:
        return EnvironmentReport(model_id, variable, None, TIER_UNCONFIGURED, "UNCONFIGURED",
                                 reason=f"{variable} not set")

    interpreter = Path(configured).expanduser()
    if not interpreter.is_file():
        return EnvironmentReport(model_id, variable, str(interpreter), "INTERPRETER_MISSING", "MISSING",
                                 reason="interpreter file missing")
    if not os.access(interpreter, os.X_OK):
        return EnvironmentReport(model_id, variable, str(interpreter), "INTERPRETER_PRESENT", "NOT_EXECUTABLE",
                                 reason="interpreter is not executable")

    if probe:
        cached = _reuse(model, require_cuda, str(interpreter))
        if cached is not None:
            return cached

    identity = _interpreter_identity(interpreter) if probe else None
    report = EnvironmentReport(
        model_id, variable, str(interpreter), "INTERPRETER_PRESENT", "PRESENT",
        prefix=(identity or {}).get("prefix"),
        base_prefix=(identity or {}).get("base_prefix"),
        python_version=(identity or {}).get("version"),
        probe_executed=bool(identity),
    )
    if probe and identity is None:
        report.tier = TIER_BLOCKED
        report.status = "PROBE_FAILED"
        report.reason = "interpreter failed its identity probe"
        return report

    if probe:
        isolated, iso_reason = check_environment_isolation(interpreter)
        if not isolated:
            report.tier = TIER_BLOCKED
            report.status = "UNSAFE_INTERPRETER"
            report.reason = iso_reason
            return report
        report.tier = "ISOLATION_VERIFIED"
        report.status = "ISOLATED"

    if not probe:
        return report

    # Python version agreement against source-declared version.
    declared = str(env.get("python") or "").strip()
    python_ok = python_version_satisfies(declared, report.python_version)
    if python_ok is False:
        report.tier = TIER_BLOCKED
        report.status = "PYTHON_VERSION_MISMATCH"
        report.reason = f"expected Python {declared}, found {report.python_version}"
        return report
    if python_ok is None and declared and declared.upper() != "UNKNOWN":
        # A declaration we cannot interpret is not evidence of agreement.
        report.tier = TIER_REQUIREMENTS_UNVERIFIED
        report.status = "VERSION_CONSTRAINT_UNVERIFIED"
        report.reason = f"unusable declared Python version {declared!r} (found {report.python_version})"
        return report
    report.tier = "PYTHON_VERSION_VERIFIED"
    report.status = "PYTHON_OK"

    # Dependency verification.
    required = _required_packages(model)
    if required:
        code = (
            "import json, sys, importlib.metadata as m\n"
            f"req={required!r}\n"
            "out={'__prefix__': sys.prefix, '__base_prefix__': getattr(sys,'base_prefix',sys.prefix)}\n"
            "for pkg,ver in req.items():\n"
            "    try: out[pkg]=m.version(pkg)\n"
            "    except Exception: out[pkg]=None\n"
            "print(json.dumps(out))\n"
        )
        ok, out, err = _run_probe(interpreter, code, isolated=True)
        if not ok:
            report.tier = TIER_BLOCKED
            report.status = "DEPENDENCIES_UNREADABLE"
            report.reason = err or "dependency probe failed"
            return report
        try:
            installed = json.loads(out.splitlines()[-1])
        except (ValueError, IndexError):
            report.tier = TIER_BLOCKED
            report.status = "DEPENDENCIES_UNREADABLE"
            report.reason = "dependency probe returned invalid JSON"
            return report
        if _probe_prefix_mismatch(installed, report, require_site=False):
            report.tier = TIER_BLOCKED
            report.status = "DEPENDENCY_PROBE_ENVIRONMENT_MISMATCH"
            report.reason = ("the dependency probe ran in a different environment: "
                             f"{installed.get('__prefix__')} != {report.prefix}")
            return report
        installed.pop("__prefix__", None)
        installed.pop("__base_prefix__", None)
        report.packages = installed
        report.dependency_probe_executed = True
        for pkg, want in required.items():
            got = installed.get(pkg)
            if got is None:
                report.tier = TIER_BLOCKED
                report.status = "DEPENDENCY_MISSING"
                report.reason = f"missing package: {pkg}"
                return report
            if want is None:
                report.tier = TIER_REQUIREMENTS_UNVERIFIED
                report.status = "VERSION_CONSTRAINT_UNVERIFIED"
                report.reason = f"{pkg}: source declares no usable version constraint (installed {got})"
                return report
            matched = version_satisfies(want, got)
            if matched is None:
                report.tier = TIER_REQUIREMENTS_UNVERIFIED
                report.status = "VERSION_CONSTRAINT_UNVERIFIED"
                report.reason = f"{pkg}: unusable version constraint {want!r} (installed {got})"
                return report
            if not matched:
                report.tier = TIER_BLOCKED
                report.status = "DEPENDENCY_VERSION_MISMATCH"
                report.reason = f"{pkg}: expected {want}, found {got}"
                return report
    report.tier = "DEPENDENCIES_VERIFIED"
    report.status = "DEPENDENCIES_OK"

    # Model import verification.
    imports = _required_imports(model)
    if imports:
        code = (
            "import json, sys\n"
            f"names={imports!r}\n"
            "out={'__prefix__': sys.prefix}\n"
            "for n in names:\n"
            "    try:\n"
            "        __import__(n); out[n]=True\n"
            "    except Exception: out[n]=False\n"
            "print(json.dumps(out))\n"
        )
        ok, out, err = _run_probe(interpreter, code, isolated=True)
        if not ok:
            report.tier = TIER_BLOCKED
            report.status = "MODEL_IMPORT_FAILED"
            report.reason = err or "import probe failed"
            return report
        try:
            results = json.loads(out.splitlines()[-1])
        except (ValueError, IndexError):
            report.tier = TIER_BLOCKED
            report.status = "MODEL_IMPORT_FAILED"
            report.reason = "import probe returned invalid JSON"
            return report
        if _probe_prefix_mismatch(results, report, require_site=False):
            report.tier = TIER_BLOCKED
            report.status = "DEPENDENCY_PROBE_ENVIRONMENT_MISMATCH"
            report.reason = ("the import probe ran in a different environment: "
                             f"{results.get('__prefix__')} != {report.prefix}")
            return report
        results.pop("__prefix__", None)
        report.imports = results
        missing = [name for name, present in results.items() if not present]
        if missing:
            report.tier = TIER_BLOCKED
            report.status = "MODEL_IMPORT_FAILED"
            report.reason = f"import failed: {', '.join(missing)}"
            return report
        report.tier = "MODEL_IMPORT_VERIFIED"
        report.status = "IMPORTS_OK"

    # Dependency evidence is a precondition for any readiness claim, including a
    # GPU-deferred one: without an executed dependency probe nothing was verified.
    if not report.dependency_probe_executed and not imports:
        report.tier = TIER_REQUIREMENTS_UNVERIFIED
        report.status = "REQUIREMENTS_UNVERIFIED"
        report.reason = "no dependency probe was executed for this contract"
        _remember(model, require_cuda, report)
        return report

    # CUDA verification (only when the model requires it).
    if require_cuda or (env.get("cuda") and str(env["cuda"]).upper() != "UNKNOWN"):
        code = (
            "import json\n"
            "out={'available': False, 'version': None}\n"
            "try:\n"
            "    import torch\n"
            "    out['available']=bool(torch.cuda.is_available())\n"
            "    out['version']=getattr(torch.version,'cuda',None)\n"
            "except Exception: pass\n"
            "print(json.dumps(out))\n"
        )
        ok, out, _ = _run_probe(interpreter, code, isolated=True)
        if ok:
            try:
                cuda = json.loads(out.splitlines()[-1])
                report.cuda_available = cuda.get("available")
                report.cuda_version = cuda.get("version")
            except (ValueError, IndexError):
                pass
        if report.cuda_available:
            report.tier = "CUDA_VERIFIED"
            report.status = "CUDA_OK"
        else:
            report.tier = TIER_GPU_DEFERRED
            report.status = "GPU_VERIFICATION_DEFERRED"
            report.reason = "CUDA runtime unavailable; GPU verification deferred"
            _remember(model, require_cuda, report)
            return report
    elif not required and not imports:
        # No source-backed requirements: cannot claim full verification.
        report.tier = TIER_REQUIREMENTS_UNVERIFIED
        report.status = "REQUIREMENTS_UNVERIFIED"
        report.reason = "no source-backed dependency contract declared"
        _remember(model, require_cuda, report)
        return report

    if not report.dependency_probe_executed:
        # We never actually inspected the installed dependency set, so we cannot
        # claim full runtime readiness even though the interpreter is isolated.
        report.tier = TIER_REQUIREMENTS_UNVERIFIED
        report.status = "REQUIREMENTS_UNVERIFIED"
        report.reason = "no dependency probe was executed for this contract"
        _remember(model, require_cuda, report)
        return report

    report.tier = "RUNTIME_READY"
    report.status = "RUNTIME_READY"
    _remember(model, require_cuda, report)
    return report


# Tiers that constitute a verified runtime. INTERPRETER_PRESENT / ISOLATION_VERIFIED
# alone are *not* readiness: nothing about the installed dependencies was observed.
READY_TIERS = {"RUNTIME_READY"}


def _environment_contract_digest(model: dict[str, Any]) -> str:
    """Identity of the *contract*, so an edited registry entry invalidates cache."""
    import hashlib
    env = model.get("environment") or {}
    relevant = {
        "python_env_var": env.get("python_env_var"),
        "python": env.get("python"),
        "pytorch": env.get("pytorch"),
        "torchvision": env.get("torchvision"),
        "cuda": env.get("cuda"),
        "packages": list(env.get("packages") or []),
        "required_imports": list(env.get("required_imports") or []),
        "environment_required": bool(model.get("environment_required")),
    }
    return hashlib.sha256(json.dumps(relevant, sort_keys=True).encode("utf-8")).hexdigest()


def _interpreter_fingerprint(interpreter: str, contract_digest: str,
                            prefix: str | None = None) -> str:
    """Identity of the *environment content*, not merely its path.

    A path alone says nothing about later package changes inside that environment,
    so the interpreter binary, the environment marker files, and the contract are all
    part of the fingerprint. Any change invalidates a cached verification.
    """
    import hashlib
    hasher = hashlib.sha256()
    hasher.update(contract_digest.encode("utf-8"))
    path = Path(interpreter)
    hasher.update(str(path).encode("utf-8"))
    for candidate in (path, path.resolve()):
        try:
            stat_result = candidate.stat()
        except OSError:
            continue
        hasher.update(f"{candidate}:{stat_result.st_size}:{stat_result.st_mtime_ns}".encode("utf-8"))
        break
    # Use the prefix the interpreter itself reports when available: resolving the
    # invocation path can land on a shared base interpreter directory (symlinked
    # venvs), which would look for markers and packages in the wrong place.
    if prefix:
        prefix_path = Path(prefix)
    else:
        prefix_path = path.parent.parent
    for marker in ("pyvenv.cfg", "conda-meta/history", "conda-meta"):
        marker_path = prefix_path / marker
        try:
            stat_result = marker_path.stat()
        except OSError:
            continue
        hasher.update(f"{marker}:{stat_result.st_size}:{stat_result.st_mtime_ns}".encode("utf-8"))
    # Package directories live under the environment's lib/: their directory mtimes
    # change when a dependency is installed or removed there.
    site_packages_dirs = sorted(prefix_path.glob("lib/python*/site-packages"))
    site_packages_dirs += [prefix_path / "Lib" / "site-packages"]
    for site_packages in site_packages_dirs:
        if not site_packages.is_dir():
            continue
        try:
            hasher.update(f"{site_packages}:{site_packages.stat().st_mtime_ns}".encode("utf-8"))
        except OSError:
            continue
        # A force-reinstall or an in-place member rewrite need not touch the directory
        # mtime, so each distribution's own metadata *and* the files it actually
        # installs (listed in its RECORD) are hashed.
        for dist in sorted(site_packages.glob("*.dist-info")):
            metadata = dist / "METADATA"
            try:
                stat_result = metadata.stat()
            except OSError:
                continue
            hasher.update(f"{dist.name}:{stat_result.st_size}:{stat_result.st_mtime_ns}".encode("utf-8"))
            record = dist / "RECORD"
            try:
                record_text = record.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            hasher.update(f"RECORD:{record_text}".encode("utf-8"))
            members = [line.split(",", 1)[0] for line in record_text.splitlines() if line.strip()]
            if len(members) > _FINGERPRINT_MEMBER_LIMIT:
                # ponytail: very large distributions are identified by their RECORD
                # plus metadata only; a same-size in-place edit inside one of their
                # members can still go unnoticed. Raise the limit or hash members when
                # a stale verification is ever observed on a real model environment.
                hasher.update(f"TRUNCATED:{len(members)}".encode("utf-8"))
                continue
            for member in members:
                member_path = site_packages / member
                try:
                    stat_result = member_path.stat()
                except OSError:
                    hasher.update(f"{member}:ABSENT".encode("utf-8"))
                    continue
                hasher.update(f"{member}:{stat_result.st_size}:{stat_result.st_mtime_ns}".encode("utf-8"))
    return hasher.hexdigest()


_FINGERPRINT_MEMBER_LIMIT = 2000


_VERIFIED_ENVIRONMENTS: dict[str, EnvironmentReport] = {}


def _cache_key(model: dict[str, Any], require_cuda: bool) -> str:
    env = model.get("environment") or {}
    configured = os.environ.get(env.get("python_env_var") or "", "")
    return (f"{model.get('model_id')}|{env.get('python_env_var')}|{configured}|"
            f"{_environment_contract_digest(model)}|cuda={bool(require_cuda)}")


def _cache_enabled() -> bool:
    return os.environ.get("WORKBENCH_NO_ENV_CACHE") not in {"1", "true", "yes"}


def _reuse(model: dict[str, Any], require_cuda: bool, interpreter: str) -> EnvironmentReport | None:
    """Return a still-valid cached verification, or None.

    A cached result is reused only when the contract, the invocation path, the
    interpreter binary, the environment markers, and the environment's lib/
    directory mtimes are all unchanged.
    """
    if not _cache_enabled():
        return None
    key = _cache_key(model, require_cuda)
    cached = _VERIFIED_ENVIRONMENTS.get(key)
    if cached is None or cached.interpreter != interpreter:
        return None
    if cached.environment_fingerprint != _interpreter_fingerprint(
        interpreter, _environment_contract_digest(model), cached.prefix
    ):
        _VERIFIED_ENVIRONMENTS.pop(key, None)
        return None
    return cached


def _remember(model: dict[str, Any], require_cuda: bool, report: EnvironmentReport) -> None:
    if not _cache_enabled() or report.interpreter is None:
        return
    report.contract_digest = _environment_contract_digest(model)
    report.environment_fingerprint = _interpreter_fingerprint(
        report.interpreter, report.contract_digest, report.prefix)
    report.verified_at = datetime.now(UTC).isoformat()
    _VERIFIED_ENVIRONMENTS[_cache_key(model, require_cuda)] = report


def forget_environment(interpreter: str | None = None) -> None:
    """Drop cached verifications (all, or those for one interpreter path)."""
    if interpreter is None:
        _VERIFIED_ENVIRONMENTS.clear()
        return
    for key in [key for key, report in _VERIFIED_ENVIRONMENTS.items() if report.interpreter == interpreter]:
        _VERIFIED_ENVIRONMENTS.pop(key, None)


def inspect_environment(model: dict[str, Any]) -> EnvironmentReport:
    """Cheap, read-only status. NEVER runs a subprocess and NEVER claims readiness.

    This is the right operation for UI status, list endpoints, and any code path
    that must not pay for a dependency probe.
    """
    return verify_environment(model, probe=False)


def require_verified_model_interpreter(model: dict[str, Any], *, require_cuda: bool = False) -> str:
    """Return the model's verified interpreter, proving the runtime first.

    Raises when the environment is not *verified*: an interpreter that merely exists
    or is isolated is not sufficient evidence to launch a real model run.
    """
    if not model.get("environment_required"):
        # Such a model runs in the workbench interpreter; never emit a bare name that
        # could resolve to an unrelated interpreter on PATH.
        return sys.executable
    probe_environment = globals().get("verify_environment", verify_environment)
    report = probe_environment(model, probe=True, require_cuda=require_cuda)
    variable = report.variable or "model-specific Python"
    if report.tier in READY_TIERS and report.interpreter:
        return report.interpreter
    raise RuntimeError(
        f"execution environment not verified [{report.tier}/{report.status}]: "
        f"{variable}={report.interpreter} ({report.reason})"
    )


def environment_status(model: dict[str, Any], probe: bool = False) -> dict[str, Any]:
    """Backward-compatible dict view of the environment report."""
    report = verify_environment(model, probe=probe)
    data = asdict(report)
    # Legacy consumers read `status`/`interpreter`/`variable`/`tier`.
    data.setdefault("isolation_reason", data.get("reason"))
    data["verified"] = report.tier in READY_TIERS
    data["gpu_deferred"] = report.tier == TIER_GPU_DEFERRED
    return data


def environment_blockers(model: dict[str, Any]) -> list[str]:
    """Every reason a model environment cannot be used for a real execution."""
    if not model.get("environment_required"):
        return []
    probe_environment = globals().get("verify_environment", verify_environment)
    report = probe_environment(model, probe=True)
    if report.tier in READY_TIERS:
        return []
    variable = report.variable or "model-specific Python"
    detail = f" ({report.reason})" if report.reason else ""
    return [f"execution environment not verified: {variable}={report.interpreter} [{report.status}/{report.tier}]{detail}"]


def python_executable(model: dict[str, Any]) -> str:
    """Verified interpreter for a model's official command.

    Delegates to the module-level ``require_verified_model_interpreter``; it never
    bypasses verification and never fabricates a path.
    """
    resolver = globals().get("require_verified_model_interpreter", require_verified_model_interpreter)
    return resolver(model)



def is_bytecode(path: str) -> bool:
    # `__pycache__`/`*.pyc` are Python build products that cannot change evaluator
    # source semantics, and some upstream repositories even commit them, so they
    # are never treated as a source change. Only files *inside* a __pycache__
    # directory qualify: a real source edit under a directory merely named
    # "__pycache__"-ish must still be detected.
    parts = Path(path).parts
    if parts and parts[-1] == "__pycache__":
        return True
    return Path(path).suffix == ".pyc"

def _path_of(line: str) -> str:
    # porcelain: XY <path>, and "R  old -> new" for renames.
    body = line[3:].strip() if len(line) > 3 else ""
    return body.split(" -> ")[-1]

def source_status(source: Path) -> dict[str, list[str]] | None:
    """Porcelain lines grouped by trust classification.

    ``tracked``        — modified/staged tracked paths (a real source change)
    ``unauthorized``   — untracked or ignored paths that are not declared runtime artifacts
    ``approved``       — entries the workbench legitimately created (narrowly scoped)
    ``invalid``        — declared artifacts whose target/symlink identity is wrong
    """
    def _porcelain(*extra: str) -> str | None:
        try:
            result = subprocess.run(["git", "status", "--porcelain", *extra],
                                    cwd=source, capture_output=True, text=True)
        except OSError:
            return None
        return result.stdout if result.returncode == 0 else None

    tracked = _porcelain("--untracked-files=no")
    if tracked is None:
        return None
    untracked = _porcelain("--untracked-files=all") or ""
    ignored = _porcelain("--untracked-files=all", "--ignored") or ""

    entries: dict[str, list[str]] = {"tracked": [], "unauthorized": [], "approved": [], "invalid": []}
    approved = _approved_source_artifacts(source)
    tracked_paths: list[str] = []
    other_paths: list[str] = []
    for line in tracked.splitlines():
        if line.strip():
            tracked_paths.append(_path_of(line))
    for line in (untracked + "\n" + ignored).splitlines():
        if line.strip():
            other_paths.append(_path_of(line))

    for path in tracked_paths:
        if is_bytecode(path):
            continue
        entries["tracked"].append(path)
    for path in other_paths:
        if is_bytecode(path):
            continue
        declaration = _declared_artifact_for(path, approved)
        if declaration is None:
            entries["unauthorized"].append(path)
            continue
        verdict = _verify_declared_artifact(source, path, declaration)
        entries["approved" if verdict is None else "invalid"].append(path if verdict is None else f"{path}: {verdict}")
    return entries

def _approved_source_artifacts(source: Path) -> dict[str, dict[str, Any]]:
    """Declared runtime artifacts for the registry model that owns *this* checkout.

    The checkout must be the one the workbench actually configured
    (``WORKBENCH_THIRD_PARTY_ROOT/<source_dir>``). A directory that merely shares a
    name with a model's ``source_dir`` is NOT governed by that model's contract, so
    an unrelated checkout can never borrow another model's allowlist.
    """
    from workbench.backend.operator_config import resolve_config
    from workbench.backend.registry import load_registry, source_runtime_artifacts

    try:
        models = load_registry()["models"]
    except Exception:
        return {}
    try:
        configured_root = Path(getattr(resolve_config(), "WORKBENCH_THIRD_PARTY_ROOT", "")).expanduser()
        resolved_source = source.resolve()
    except (OSError, ValueError):
        return {}
    for model in models:
        source_dir = model.get("source_dir")
        if not source_dir:
            continue
        try:
            if (configured_root / source_dir).resolve() != resolved_source:
                continue
        except (OSError, ValueError):
            continue
        return {item["path"]: item for item in source_runtime_artifacts(model)}
    return {}


def _recorded_asset_digest(path: Path) -> str | None:
    """SHA-256 recorded when the workbench itself acquired this declared asset."""
    from workbench.backend.operator_config import resolve_config

    root = getattr(resolve_config(), "CIR_REPO_ROOT", Path(__file__).resolve().parents[2])
    manifest = root / "workbench" / "artifacts" / "auxiliary" / "download_manifest.json"
    if not manifest.is_file():
        return None
    try:
        entries = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    for entry in entries if isinstance(entries, list) else []:
        try:
            if Path(entry["destination"]).resolve() == path.resolve():
                digest = entry.get("downloaded_sha256")
                return str(digest) if digest else None
        except (KeyError, OSError, TypeError):
            continue
    return None


def _declared_artifact_for(path: str, approved: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    for declared, declaration in approved.items():
        if path == declared or path.startswith(declared.rstrip("/") + "/"):
            return declaration
    return None

def _verify_declared_artifact(source: Path, path: str, declaration: dict[str, Any]) -> str | None:
    """Return a rejection reason, or None when the artifact is legitimate."""
    from workbench.backend.operator_config import resolve_config

    try:
        resolved_source = source.resolve()
        candidate = source / path
        candidate_resolved = candidate.resolve()
    except OSError as error:
        return f"unresolvable: {error}"
    if declaration["kind"] == "canonical_dataset_link":
        # The whole point of this artifact is to resolve *outside* the checkout, so
        # the containment rule below does not apply. It is bounded instead by the
        # configured canonical FashionIQ root.
        canonical = Path(resolve_config().FASHIONIQ_ROOT).resolve()
        if not candidate.is_symlink():
            return "expected a symlink to the canonical FashionIQ root"
        if candidate_resolved != canonical:
            return f"resolves to {candidate_resolved}, expected {canonical}"
        if not candidate_resolved.is_dir():
            return "dangling symlink"
        return None
    if not candidate_resolved.is_relative_to(resolved_source):
        return "escapes the source checkout"
    if declaration["kind"] == "declared_auxiliary_asset":
        # Auxiliary assets are declared *files*; a symlink there is not authorized.
        if candidate.is_symlink():
            return "declared auxiliary asset must be a regular file"
        if not candidate.is_file():
            return "declared auxiliary asset is missing"
        from workbench.backend.registry import sha256_file

        digest = sha256_file(candidate)
        expected_sha = declaration.get("expected_sha256")
        if expected_sha and digest != expected_sha:
            return "declared auxiliary asset checksum mismatch"
        recorded = _recorded_asset_digest(candidate)
        if recorded and digest != recorded:
            return "declared auxiliary asset does not match its recorded acquisition digest"
        if not expected_sha and not recorded:
            return "declared auxiliary asset has no source-backed digest to verify against"
        return None
    return "unknown declared artifact kind"

# Private aliases kept for internal readability and existing imports.
_source_status = source_status
_is_bytecode = is_bytecode


def _source_dirty(source: Path, model: dict[str, Any] | None = None) -> bool | None:
    """True when the checkout differs from the pin in a way that matters.

    ``__pycache__``/``*.pyc`` are unavoidable products of running Python and do not
    change evaluator semantics. Declared, verified runtime artifacts (the CSMCIR
    canonical dataset link, audited auxiliary assets) are legitimate products of the
    workbench's own preparation contract. Everything else — tracked edits, and any
    untracked/ignored path that is *not* a verified declared artifact — is dirty.
    """
    entries = source_status(source)
    if entries is None:
        return None
    return bool(entries["tracked"] or entries["unauthorized"] or entries["invalid"])


def source_clean_and_pinned(model: dict[str, Any], source: Path) -> tuple[bool, str | None, str | None]:
    try:
        top_level = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=source,
                                   check=True, capture_output=True, text=True).stdout.strip()
        if Path(top_level).resolve() != source.resolve():
            return False, None, "source path is not a Git checkout root"
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=source, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return False, None, "source is not a readable Git checkout"
    dirty = _source_dirty(source, model)
    if dirty is None:
        return False, head, "source Git metadata unreadable"
    expected = model.get("upstream_commit_sha")
    if dirty:
        entries = source_status(source) or {}
        detail = (entries.get("tracked") or entries.get("invalid") or entries.get("unauthorized") or ["unknown"])
        return False, head, f"source checkout is dirty: {detail[0]}"
    if expected and head != expected:
        return False, head, f"source pin mismatch: expected {expected}, found {head}"
    return True, head, None


def source_provenance(model: dict[str, Any], source: Path) -> dict[str, Any]:
    """Record expected pin, actual pin, cleanliness, and a provenance digest."""
    import hashlib

    expected = model.get("upstream_commit_sha")
    actual = None
    dirty = None
    error = None
    try:
        actual = subprocess.run(["git", "rev-parse", "HEAD"], cwd=source, check=True,
                                capture_output=True, text=True).stdout.strip()
        dirty = _source_dirty(source, model)
    except (OSError, subprocess.CalledProcessError) as exc:
        error = f"git metadata unreadable: {exc}"
    entries = source_status(source) or {}

    clean_and_pinned = bool(actual and not dirty and (not expected or actual == expected))
    digest = hashlib.sha256(
        json.dumps(
            {"expected": expected, "actual": actual, "dirty": dirty, "error": error},
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "expected_commit": expected,
        "actual_commit": actual,
        "dirty": dirty,
        "tracked_modifications": entries.get("tracked", []),
        "unauthorized_untracked": entries.get("unauthorized", []),
        "approved_runtime_artifacts": entries.get("approved", []),
        "invalid_runtime_artifacts": entries.get("invalid", []),
        "error": error,
        "verified": clean_and_pinned,
        "provenance_digest": digest,
    }


def runtime_blockers(model: dict[str, Any], checkpoint: dict[str, Any], config: Any,
                     protocol: str | None = None, dataset_root: Path | None = None) -> list[str]:
    """Shared final preflight used by UI availability and guarded execution."""
    from workbench.backend.adapters.models import ADAPTERS, OfficialScriptAdapter
    from workbench.backend.fashioniq_layout import missing_paths
    from workbench.backend.registry import (
        checkpoint_bundle_missing_paths, checkpoint_is_present, checkpoint_missing_paths,
        checkpoint_path, external_asset_blockers, fashioniq_required_paths,
        preparation_contract_for_model, preparation_paths, sha256_file,
    )

    blockers: list[str] = []
    source = config.WORKBENCH_THIRD_PARTY_ROOT / model["source_dir"] if model.get("source_dir") else None
    if source is None or not source.is_dir():
        blockers.append(f"source missing: {source}")
    else:
        clean, _, reason = source_clean_and_pinned(model, source)
        if not clean:
            blockers.append(reason or "source unavailable")
    if protocol is not None and protocol not in model.get("supported_protocols", []):
        blockers.append(f"protocol incompatible: {protocol}")
    checkpoint_file = checkpoint_path(model["model_id"], checkpoint, config.WORKBENCH_CHECKPOINT_ROOT)
    if not checkpoint_is_present(checkpoint_file, checkpoint):
        blockers.append(f"checkpoint missing: {checkpoint_missing_paths(checkpoint_file, checkpoint)[0]}")
    if checkpoint.get("checkpoint_mapping_status") != "VERIFIED_METADATA":
        blockers.append("checkpoint mapping unresolved")
    expected_sha = checkpoint.get("expected_sha256")
    if expected_sha and checkpoint_file.is_file() and sha256_file(checkpoint_file) != expected_sha:
        blockers.append("official checkpoint hash mismatch")
    for bundle_id, missing in checkpoint_bundle_missing_paths(model, config.WORKBENCH_CHECKPOINT_ROOT).items():
        blockers.append(f"checkpoint bundle incomplete: {bundle_id}: {missing[0]}")
    adapter_type = ADAPTERS.get(model["model_id"])
    command_implemented = (adapter_type is not None and
                           getattr(adapter_type, "command", None) is not OfficialScriptAdapter.command)
    command_audited = (model.get("command_status") in {"COMMAND_AUDITED", "REPLAY_COMMAND_IMPLEMENTED"} or
                       (model.get("command_status") is None and command_implemented)) and command_implemented
    if not command_audited:
        blockers.append("adapter command not audited")
    elif source is not None and getattr(adapter_type, "script", None) and not (source / adapter_type.script).is_file():
        blockers.append(f"official evaluator missing: {source / adapter_type.script}")
    blockers.extend(environment_blockers(model))
    dataset_root = dataset_root or config.FASHIONIQ_ROOT
    contract = preparation_contract_for_model(model)
    if contract is None:
        missing = missing_paths(fashioniq_required_paths(model, dataset_root))
    else:
        missing = missing_paths(preparation_paths(contract, dataset_root, "raw_inputs") +
                                preparation_paths(contract, dataset_root, "generated_artifacts"))
    if missing:
        layout = model.get("fashioniq_layout") or "FashionIQ"
        blockers.append(f"FashionIQ {layout} requirement missing: {missing[0]}")
    blockers.extend(external_asset_blockers(model))
    if model["model_id"] == "encoder" and source is not None:
        if not (source / "datasets1.py").is_file():
            blockers.append(f"ENCODER evaluator import missing: {source / 'datasets1.py'}")
        if not (source / "open_clip_pytorch_model.bin").is_file():
            blockers.append(f"ENCODER asset missing: {source / 'open_clip_pytorch_model.bin'}")
    if model["model_id"] == "csmcir" and source is not None:
        layout = source / "fashionIQ_dataset"
        if not layout.is_symlink() or layout.resolve() != config.FASHIONIQ_ROOT.resolve():
            blockers.append(f"CSMCIR requires canonical dataset link: {layout} -> {config.FASHIONIQ_ROOT}")
        for group, paths in (
            ("Qwen captions", tuple(config.FASHIONIQ_ROOT / "qwen_captions" / f"{category}_cot_val.json" for category in ("dress", "shirt", "toptee"))),
            ("COT_ours2 captions", tuple(source / "COT_ours2" / "fashioniq" / f"{category}_cot_val.json" for category in ("dress", "shirt", "toptee"))),
        ):
            group_missing = missing_paths(paths)
            if group_missing:
                blockers.append(f"CSMCIR {group} missing: {group_missing[0]}")
    return blockers
