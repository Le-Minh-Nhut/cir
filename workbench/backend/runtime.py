"""Environment verification with explicit readiness tiers and source-backed probing."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
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


def _run_probe(interpreter: Path, code: str, timeout: int = 60) -> tuple[bool, str, str]:
    try:
        proc = subprocess.run(
            [str(interpreter), "-c", code],
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
}
print(json.dumps(info))
"""


def _interpreter_identity(interpreter: Path) -> dict[str, Any] | None:
    """Run the interpreter to learn its real prefix/base_prefix/version."""
    ok, out, _ = _run_probe(interpreter, _PROBE_CODE)
    if not ok or not out:
        return None
    try:
        return json.loads(out.splitlines()[-1])
    except (ValueError, IndexError):
        return None


def check_environment_isolation(interpreter: Path) -> tuple[bool, str]:
    """Verify the *runtime* identity of an interpreter, not just its resolved path.

    A Linux venv's ``bin/python`` is often a symlink to ``/usr/bin/python3``; the
    resolved path alone is therefore not evidence of system Python. Isolation is
    decided by the interpreter's own ``sys.prefix``/``sys.base_prefix`` and
    environment markers.
    """
    resolved = interpreter.resolve()
    if resolved == Path(sys.executable).resolve():
        return False, "cannot use active workbench environment"

    system_names = {"/usr/bin/python", "/usr/bin/python3", "/bin/python", "/bin/python3",
                    "/usr/local/bin/python", "/usr/local/bin/python3", "/usr/bin/python3.11",
                    "/usr/bin/python3.12", "/usr/bin/python3.13"}
    identity = _interpreter_identity(interpreter)
    if identity is None:
        # Cannot establish runtime identity -> fail closed.
        s = str(resolved)
        if s in system_names:
            return False, "cannot use system Python"
        if any(base in s for base in ("/miniconda3/bin/python", "/anaconda3/bin/python", "/opt/conda/bin/python")) and "/envs/" not in s:
            return False, "cannot use base Conda environment"
        return False, "interpreter runtime identity could not be established"

    prefix = str(identity.get("prefix") or "")
    base_prefix = str(identity.get("base_prefix") or "")
    executable = str(identity.get("executable") or "")

    if not prefix:
        return False, "interpreter has no detectable prefix"

    # System / base interpreter: prefix equals base_prefix and points at a system root.
    if prefix == base_prefix and (prefix in ("/usr", "/usr/local") or prefix == ""):
        return False, "cannot use system Python"

    if any(base in prefix for base in ("/miniconda3", "/anaconda3", "/opt/conda")) and "/envs/" not in prefix:
        return False, "cannot use base Conda environment"

    # A virtualenv/conda env has prefix != base_prefix, or env markers on disk.
    prefix_path = Path(prefix)
    has_pyvenv = (prefix_path / "pyvenv.cfg").is_file()
    has_conda_meta = (prefix_path / "conda-meta").is_dir()

    if prefix == base_prefix and not (has_conda_meta and "/envs/" in prefix):
        return False, "interpreter is not in an isolated environment"

    # Verify the interpreter actually lives inside its claimed prefix.
    try:
        executable_path = Path(executable).resolve()
        prefix_resolved = prefix_path.resolve()
        if not executable_path.is_relative_to(prefix_resolved):
            return False, "interpreter executable does not belong to its claimed prefix"
    except (OSError, ValueError):
        return False, "interpreter prefix relationship could not be verified"

    if not (has_pyvenv or has_conda_meta):
        return False, "isolated environment markers (pyvenv.cfg/conda-meta) are missing"

    return True, "ISOLATION_VERIFIED"


def _required_packages(model: dict[str, Any]) -> dict[str, str]:
    env = model.get("environment") or {}
    required: dict[str, str] = {}
    if env.get("pytorch"):
        required["torch"] = str(env["pytorch"])
    if env.get("torchvision"):
        required["torchvision"] = str(env["torchvision"])
    for pkg in env.get("packages") or []:
        if isinstance(pkg, str):
            required.setdefault(pkg.split("==")[0], pkg.split("==")[1] if "==" in pkg else "")
    return required


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
    if declared and declared != "UNKNOWN" and report.python_version:
        if not report.python_version.startswith(declared) and not declared.startswith(report.python_version):
            report.tier = TIER_BLOCKED
            report.status = "PYTHON_VERSION_MISMATCH"
            report.reason = f"expected Python {declared}, found {report.python_version}"
            return report
    report.tier = "PYTHON_VERSION_VERIFIED"
    report.status = "PYTHON_OK"

    # Dependency verification.
    required = _required_packages(model)
    if required:
        code = (
            "import json,importlib.metadata as m\n"
            f"req={required!r}\n"
            "out={}\n"
            "for pkg,ver in req.items():\n"
            "    try: out[pkg]=m.version(pkg)\n"
            "    except Exception: out[pkg]=None\n"
            "print(json.dumps(out))\n"
        )
        ok, out, err = _run_probe(interpreter, code)
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
        report.packages = installed
        report.dependency_probe_executed = True
        for pkg, want in required.items():
            got = installed.get(pkg)
            if got is None:
                report.tier = TIER_BLOCKED
                report.status = "DEPENDENCY_MISSING"
                report.reason = f"missing package: {pkg}"
                return report
            if want and want != "UNKNOWN" and not str(got).startswith(str(want)) and not str(want).startswith(str(got)):
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
            "import json\n"
            f"names={imports!r}\n"
            "out={}\n"
            "for n in names:\n"
            "    try:\n"
            "        __import__(n); out[n]=True\n"
            "    except Exception: out[n]=False\n"
            "print(json.dumps(out))\n"
        )
        ok, out, err = _run_probe(interpreter, code)
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
        report.imports = results
        missing = [name for name, present in results.items() if not present]
        if missing:
            report.tier = TIER_BLOCKED
            report.status = "MODEL_IMPORT_FAILED"
            report.reason = f"import failed: {', '.join(missing)}"
            return report
        report.tier = "MODEL_IMPORT_VERIFIED"
        report.status = "IMPORTS_OK"

    # CUDA verification (only when the model requires it).
    if require_cuda or env.get("cuda"):
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
        ok, out, _ = _run_probe(interpreter, code)
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
            return report
    elif not required and not imports:
        # No source-backed requirements: cannot claim full verification.
        report.tier = TIER_REQUIREMENTS_UNVERIFIED
        report.status = "REQUIREMENTS_UNVERIFIED"
        report.reason = "no source-backed dependency contract declared"
        return report

    if not report.dependency_probe_executed:
        # We never actually inspected the installed dependency set, so we cannot
        # claim full runtime readiness even though the interpreter is isolated.
        report.tier = TIER_REQUIREMENTS_UNVERIFIED
        report.status = "REQUIREMENTS_UNVERIFIED"
        report.reason = "no dependency probe was executed for this contract"
        return report

    report.tier = "RUNTIME_READY"
    report.status = "RUNTIME_READY"
    return report


def environment_status(model: dict[str, Any], probe: bool = False) -> dict[str, Any]:
    """Backward-compatible dict view of the environment report."""
    report = verify_environment(model, probe=probe)
    data = asdict(report)
    # Legacy consumers read `status`/`interpreter`/`variable`/`tier`.
    data.setdefault("isolation_reason", data.get("reason"))
    return data


def environment_blockers(model: dict[str, Any]) -> list[str]:
    if not model.get("environment_required"):
        return []
    report = verify_environment(model, probe=True)
    if report.tier == "RUNTIME_READY":
        return []
    variable = report.variable or "model-specific Python"
    detail = f" ({report.reason})" if report.reason else ""
    return [f"execution environment not verified: {variable}={report.interpreter} [{report.status}/{report.tier}]{detail}"]


def python_executable(model: dict[str, Any]) -> str:
    report = verify_environment(model)
    if report.tier == "RUNTIME_READY" and report.interpreter:
        return report.interpreter
    if model.get("environment_required"):
        raise RuntimeError(
            f"execution environment not ready [{report.tier}/{report.status}]: set {report.variable}"
        )
    return "python"


def _source_dirty(source: Path) -> bool | None:
    """True when tracked files or non-cache untracked files differ from the pin.

    ``__pycache__``/``*.pyc`` are unavoidable products of running Python and do
    not change evaluator semantics, so they are excluded from the dirty check.
    Anything else (tracked modifications, other untracked files) is dirty.
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

    def _path_of(line: str) -> str:
        # porcelain: XY <path>, and "R  old -> new" for renames.
        body = line[3:].strip() if len(line) > 3 else ""
        return body.split(" -> ")[-1]

    def _is_bytecode(path: str) -> bool:
        # `__pycache__`/`*.pyc` are Python build products that cannot change evaluator
        # source semantics, and some upstream repositories even commit them, so they
        # are never treated as a source change. Only files *inside* a __pycache__
        # directory qualify: a real source edit under a directory merely named
        # "__pycache__"-ish must still be detected.
        parts = Path(path).parts
        if parts and parts[-1] == "__pycache__":
            return True
        return Path(path).suffix == ".pyc"

    for line in (tracked + "\n" + untracked + "\n" + ignored).splitlines():
        if not line.strip():
            continue
        if _is_bytecode(_path_of(line)):
            continue
        # Any other tracked modification, untracked file, or ignored local asset
        # (e.g. a locally placed backbone) counts as dirty.
        return True
    return False


def source_clean_and_pinned(model: dict[str, Any], source: Path) -> tuple[bool, str | None, str | None]:
    try:
        top_level = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=source,
                                   check=True, capture_output=True, text=True).stdout.strip()
        if Path(top_level).resolve() != source.resolve():
            return False, None, "source path is not a Git checkout root"
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=source, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return False, None, "source is not a readable Git checkout"
    dirty = _source_dirty(source)
    if dirty is None:
        return False, head, "source Git metadata unreadable"
    expected = model.get("upstream_commit_sha")
    if dirty:
        return False, head, "source checkout is dirty"
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
        dirty = _source_dirty(source)
    except (OSError, subprocess.CalledProcessError) as exc:
        error = f"git metadata unreadable: {exc}"

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
