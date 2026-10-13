"""Plan deterministic, source-backed evaluation environments.

Models are processed by descending declared-constraint count, then model ID.
Each joins the largest group whose complete combined constraints remain
satisfiable; ties use sorted member IDs. Every union is tested from scratch,
never inferred from pairwise compatibility. Solver checks are dry-run only.
"""
from __future__ import annotations

import ast
import contextlib
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
import sys
import uuid
from typing import Any, Callable

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from workbench.backend import runtime

WORKBENCH_ROOT = Path(__file__).resolve().parents[1]
# A solve against conda-forge/pytorch/defaults takes seconds even for a single
# package, so this budget is sized for a real solve rather than a probe.
SOLVER_TIMEOUT_SECONDS = 300
CONDA_CHANNELS = ("conda-forge", "pytorch", "defaults")

_SOLVE_CACHE: dict[tuple[Any, ...], tuple[bool, str]] = {}
_SOLVE_CACHE_OWNERS: dict[int, Callable] = {}

#: Explicit allowlist for package installation and verification subprocesses.
#: Package managers and probes only receive variables essential for resolution and installation,
#: stopping credential propagation (e.g. WORKBENCH_LEDGER_KEY, cloud tokens, API keys).
_PACKAGE_MANAGER_ENV_ALLOWLIST = (
    # Execution basics: locating binaries, user home directory, temporary directories
    "PATH", "HOME", "USER", "LOGNAME", "SHELL", "TMPDIR", "TEMP", "TMP",
    # Locale and encoding: python and package tools require consistent text decoding
    "LANG", "LANGUAGE", "LC_ALL", "LC_CTYPE", "LC_MESSAGES", "TZ",
    # Network and proxy: package index access through enterprise proxies or custom CA bundles
    "HTTP_PROXY", "HTTPS_PROXY", "FTP_PROXY", "ALL_PROXY", "NO_PROXY",
    "http_proxy", "https_proxy", "ftp_proxy", "all_proxy", "no_proxy",
    "SSL_CERT_FILE", "SSL_CERT_DIR", "CURL_CA_BUNDLE", "REQUESTS_CA_BUNDLE",
    # CUDA and GPU discovery: package wheels or builds selecting matching CUDA architectures
    "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES",
    # Conda environment discovery and package caches
    "CONDA_EXE", "CONDA_PYTHON_EXE", "CONDA_ROOT", "CONDA_PREFIX", "CONDA_PKGS_DIRS",
    "CONDA_SHLVL", "CONDA_DEFAULT_ENV",
    # Pip and library caches
    "PIP_CACHE_DIR", "TORCH_HOME", "HF_HOME", "TRANSFORMERS_CACHE",
    "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME",
    # Python runtime flags
    "PYTHONIOENCODING", "PYTHONHASHSEED",
)

_PACKAGE_MANAGER_ENV_DENY_PREFIXES = (
    "WORKBENCH_LEDGER_KEY", "WORKBENCH_SECRET_",
    "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN",
    "WANDB_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
    "GITHUB_TOKEN", "GH_TOKEN", "GITLAB_TOKEN",
    "AWS_", "AZURE_", "GOOGLE_", "GCP_",
    "DOCKER_", "KUBECONFIG", "SLACK_", "STRIPE_", "NPM_",
)


def _package_manager_env(extra_env: dict[str, str] | None = None) -> dict[str, str]:
    """Build an explicit allowlist environment for provisioning and verification subprocesses."""
    env: dict[str, str] = {}
    for key in _PACKAGE_MANAGER_ENV_ALLOWLIST:
        val = os.environ.get(key)
        if val is not None:
            env[key] = val
    for key in list(env):
        if any(key == p or key.startswith(p) for p in _PACKAGE_MANAGER_ENV_DENY_PREFIXES):
            env.pop(key, None)
    if extra_env:
        for k, v in extra_env.items():
            if not any(k == p or k.startswith(p) for p in _PACKAGE_MANAGER_ENV_DENY_PREFIXES):
                env[k] = str(v)
    return env


MARKER_FILENAME = ".workbench_managed_env.json"

def _is_staging_path(path: Path | str | None) -> bool:
    return path is not None and any(part.startswith(".staging-") for part in Path(path).parts)


def _is_workbench_managed(prefix: Path, environment_id: str | None = None) -> bool:
    if _is_staging_path(prefix):
        return False
    marker = Path(prefix) / MARKER_FILENAME
    if not marker.is_file():
        return False
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
        if data.get("managed_by") != "workbench":
            return False
        if environment_id is not None and data.get("environment_id") != environment_id:
            return False
        return True
    except Exception:
        return False


def _mark_workbench_managed(prefix: Path, environment_id: str) -> None:
    marker = Path(prefix) / MARKER_FILENAME
    data = {
        "managed_by": "workbench",
        "environment_id": environment_id,
    }
    marker.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


@contextlib.contextmanager
def _environment_lock(directory: Path, environment_id: str):
    """Serialize provisioning of one environment across processes.

    Two concurrent runs that planned the same group would otherwise both write into the same
    prefix: one renames a half-installed tree while the other is still populating it. The lock is
    advisory and held for the whole create/install/verify sequence, never for reuse.
    # ponytail: one lock per environment_id, not per prefix; sufficient here because a group is the
    # unit of provisioning and a split derives a new id.
    """
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / f".provision-{environment_id}.lock"
    handle = lock_path.open("a+")
    try:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except ImportError:
            pass
        handle.seek(0)
        handle.truncate()
        handle.write(f"{os.getpid()}\n")
        handle.flush()
        yield
    finally:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except ImportError:
            pass
        handle.close()


_SOLVERS = ("conda", "mamba", "micromamba")
_UNKNOWN = {"", "UNKNOWN", "NONE", "NULL"}
#: Official PyTorch/torchvision pairings, from the upstream compatibility matrix
#: (github.com/pytorch/pytorch/wiki/PyTorch-Versions). Patch releases of torch keep the
#: torchvision minor listed for their minor series, which is why 1.12.1 may pair with 0.13.1.
#: 1.7.0 paired with BOTH 0.8.0 and 0.8.1 upstream; omitting 0.8.0 made a correct pinned pair
#: look unsatisfiable.
_TORCHVISION_PAIRS = {
    ("1.0.0", "0.2.2"), ("1.1.0", "0.3.0"), ("1.2.0", "0.4.0"),
    ("1.3.0", "0.4.1"), ("1.3.1", "0.4.2"),
    ("1.4.0", "0.5.0"), ("1.5.0", "0.6.0"), ("1.5.1", "0.6.1"),
    ("1.6.0", "0.7.0"),
    ("1.7.0", "0.8.0"), ("1.7.0", "0.8.1"), ("1.7.1", "0.8.2"),
    ("1.8.0", "0.9.0"), ("1.8.1", "0.9.1"),
    ("1.9.0", "0.10.0"), ("1.9.1", "0.10.1"),
    ("1.10.0", "0.11.0"), ("1.10.0", "0.11.1"), ("1.10.1", "0.11.2"),
    ("1.10.2", "0.11.3"),
    ("1.11.0", "0.12.0"), ("1.12.0", "0.13.0"), ("1.12.1", "0.13.1"),
    ("1.13.0", "0.14.0"), ("1.13.1", "0.14.1"),
    ("2.0.0", "0.15.1"), ("2.0.1", "0.15.2"),
    ("2.1.0", "0.16.0"), ("2.1.1", "0.16.1"), ("2.1.2", "0.16.2"),
    ("2.2.0", "0.17.0"), ("2.2.1", "0.17.1"), ("2.2.2", "0.17.2"),
    ("2.3.0", "0.18.0"), ("2.3.1", "0.18.1"), ("2.4.0", "0.19.0"),
    ("2.4.1", "0.19.1"), ("2.5.0", "0.20.0"), ("2.5.1", "0.20.1"),
    ("2.6.0", "0.21.0"),
}


def _unknown(value: Any) -> bool:
    return value is None or str(value).strip().upper() in _UNKNOWN | {"*"}


def _package_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value.strip().lower())


def _parse_requirement(line: str) -> tuple[str, str] | None:
    line = line.split(" #", 1)[0].split(";", 1)[0].strip()
    if not line or line.startswith(("#", "-")):
        return None
    match = re.match(r"^([A-Za-z0-9][A-Za-z0-9_.-]*)(?:\[[^]]+\])?\s*(.*)$", line)
    if not match:
        return None
    name, spec = _package_name(match.group(1)), match.group(2).strip()
    if spec.startswith("@"):
        return name, "UNKNOWN"
    return name, spec or "*"


def resolve_third_party_root(explicit: Path | str | None = None) -> Path:
    if explicit is not None:
        return Path(explicit).resolve()
    env = os.environ.get("WORKBENCH_THIRD_PARTY_ROOT")
    if env:
        return Path(env).resolve()
    return (WORKBENCH_ROOT / "third_party").resolve()


def _requirements_path(model: dict, third_party_root: Path | None = None) -> Path | None:
    source_dir = model.get("source_dir")
    if not source_dir:
        return None
    source = resolve_third_party_root(third_party_root) / str(source_dir)
    return next((path for path in (source / "requirements.txt", source / "requirement.txt") if path.is_file()), None)

_IMPORT_DISTRIBUTIONS = {
    "attr": ("attrs",), "bs4": ("beautifulsoup4",), "cv2": ("opencv-python", "opencv-python-headless"),
    "dateutil": ("python-dateutil",), "fitz": ("pymupdf",), "lavis": ("salesforce-lavis",),
    "open_clip": ("open-clip-torch",), "pil": ("pillow",), "skimage": ("scikit-image",),
    "sklearn": ("scikit-learn",), "yaml": ("pyyaml",),
}



def _evaluator_imports(model: dict, third_party_root: Path | None = None) -> set[str] | None:
    """Return third-party top-level imports reachable from the pinned evaluator."""
    source_dir = model.get("source_dir")
    if not source_dir:
        return None
    from workbench.backend.adapters.models import ADAPTERS, OfficialScriptAdapter

    adapter = ADAPTERS.get(model.get("model_id"))
    if adapter is None or not issubclass(adapter, OfficialScriptAdapter):
        return None
    source = resolve_third_party_root(third_party_root) / str(source_dir)
    if model.get("model_id") == "limn":
        entry, roots = WORKBENCH_ROOT / "replay" / "limn.py", [WORKBENCH_ROOT.parent, source / "LIMN", source]
    else:
        script = getattr(adapter, "script", None)
        if not script:
            return None
        script_path = Path(script)
        entry, roots = source / script_path, [source / script_path.parent, source]
    if not entry.is_file():
        return None

    standard = sys.stdlib_module_names | set(sys.builtin_module_names) | {"__future__"}
    seen: set[Path] = set()
    external: set[str] = set()

    def local_path(module: str) -> Path | None:
        parts = module.split(".")
        relative = Path(*parts)
        for root in roots:
            for candidate in (root / relative.parent / f"{relative.name}.py", root / relative / "__init__.py"):
                if candidate.is_file():
                    return candidate.resolve()
        return None

    def visit(path: Path) -> bool:
        path = path.resolve()
        if path in seen:
            return True
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, SyntaxError):
            return False
        seen.add(path)
        module_package = ""
        for root in roots:
            try:
                relative = path.relative_to(root.resolve()).with_suffix("")
            except ValueError:
                continue
            parts = relative.parts if path.name == "__init__.py" else relative.parts[:-1]
            module_package = ".".join(parts)
            break

        imports: list[tuple[str, str]] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend((alias.name, "") for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level:
                    parts = module_package.split(".") if module_package else []
                    parts = parts[:max(0, len(parts) - node.level + 1)]
                    module = ".".join(parts + ([module] if node.module else []))
                if module:
                    imports.append((module, ""))
                imports.extend((module, alias.name) for alias in node.names if alias.name != "*")
            elif isinstance(node, ast.Call):
                function = node.func
                dynamic = (isinstance(function, ast.Name) and function.id == "__import__") or (
                    isinstance(function, ast.Attribute) and function.attr == "import_module")
                if dynamic and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                    imports.append((node.args[0].value, ""))

        for module, member in imports:
            if not module:
                continue
            local = local_path(module)
            member_path = local_path(f"{module}.{member}") if member else None
            if local:
                if not visit(local):
                    return False
                if member_path and not visit(member_path):
                    return False
            elif member_path:
                if not visit(member_path):
                    return False
            elif module.startswith("."):
                return False
            elif module.split(".", 1)[0] not in standard:
                external.add(module.split(".", 1)[0])
        return True

    return external if visit(entry) else None

def _distribution_names(module: str, available: set[str]) -> set[str]:
    candidates = _IMPORT_DISTRIBUTIONS.get(module.lower(), (module,))
    return {_package_name(name) for name in candidates} & available


def _add_package(packages: dict[str, str], ranges: dict[str, str], name: str, spec: str) -> None:
    exact, bounded = [], []
    for part in spec.split(","):
        part = part.strip()
        if part == "*":
            continue
        if part == "UNKNOWN" or part.startswith("@"):
            packages[name] = "UNKNOWN"
            return
        if match := re.fullmatch(r"(?:===|==)\s*(.+)", part):
            exact.append(match.group(1))
        elif re.match(r"^(?:>=|<=|>|<|!=|~=)", part):
            bounded.append(part)
        else:
            packages[name] = "UNKNOWN"
            return
    if exact:
        previous = packages.get(name)
        values = [*([previous] if previous not in (None, "*", "UNKNOWN") else []), *exact]
        packages[name] = values[0] if len(set(values)) == 1 else ",".join(f"=={value}" for value in values)
    elif name not in packages:
        packages[name] = "*"
    if bounded:
        previous = ranges.get(name)
        ranges[name] = ",".join(filter(None, (previous, *bounded)))


def model_constraints(model: dict, third_party_root: Path | None = None) -> dict:
    """Extract evaluator imports and their source-backed version evidence."""
    env = model.get("environment") or {}
    requirements = _requirements_path(model, third_party_root)
    packages: dict[str, str] = {}
    ranges: dict[str, str] = {}
    pip_only: set[str] = set()
    evidence = [f"registry model metadata: {model.get('model_id') or '<unnamed>'}"]
    manifest = []
    declarations = env.get("packages") or []
    if isinstance(declarations, dict):
        declarations = [f"{name}{value if str(value).startswith(('=', '<', '>', '!', '~')) else '==' + str(value)}"
                        for name, value in declarations.items()]
    pip_declarations = env.get("pip_packages") or env.get("pip") or []
    if isinstance(pip_declarations, dict):
        pip_declarations = [f"{name}{value if str(value).startswith(('=', '<', '>', '!', '~')) else '==' + str(value)}"
                            for name, value in pip_declarations.items()]
    for decl in pip_declarations:
        if parsed := _parse_requirement(str(decl)):
            name, spec = parsed
            _add_package(packages, ranges, name, "UNKNOWN" if spec == "*" else spec)
            pip_only.add(name)
    if requirements:
        root = resolve_third_party_root(third_party_root)
        try:
            evidence.append(str(requirements.relative_to(root)))
        except ValueError:
            try:
                evidence.append(str(requirements.relative_to(WORKBENCH_ROOT)))
            except ValueError:
                evidence.append(str(requirements))
        manifest = [parsed for line in requirements.read_text(encoding="utf-8", errors="replace").splitlines()
                    if (parsed := _parse_requirement(line))]
    registry_pins = [parsed for declaration in declarations
                     if (parsed := _parse_requirement(str(declaration)))]
    imports = _evaluator_imports(model, third_party_root)
    if imports is None:
        selected = manifest if requirements else registry_pins
    else:
        source_pins = manifest if requirements else registry_pins
        available = {name for name, _ in source_pins}
        modules = imports - {"torch", "torchvision"}
        selected_names = {name for module in modules for name in _distribution_names(module, available)}
        if requirements:
            selected_names.update(available & {"torch", "torchvision"})
        selected = [item for item in source_pins if item[0] in selected_names]
        selected_names.update(name for name, _ in selected)
        for module in modules:
            names = _distribution_names(module, available)
            if not names:
                names = {_package_name(name) for name in _IMPORT_DISTRIBUTIONS.get(module.lower(), (module,))}
            for name in names:
                if name not in selected_names:
                    selected.append((name, "UNKNOWN"))
                    selected_names.add(name)
    for name, spec in selected:
        _add_package(packages, ranges, name, "UNKNOWN" if spec == "*" else spec)

    for name in set(packages) | set(ranges):
        if name in _NON_CONDA_PACKAGES:
            pip_only.add(name)

    torch = str(env.get("pytorch") or "UNKNOWN")
    torchvision = str(env.get("torchvision") or "UNKNOWN")
    if requirements:
        for package, key in (("torch", "torch"), ("torchvision", "torchvision")):
            if package in packages or package in ranges:
                if key == "torch":
                    torch = packages.pop(package, ranges.pop(package, "UNKNOWN"))
                else:
                    torchvision = packages.pop(package, ranges.pop(package, "UNKNOWN"))
                ranges.pop(package, None)
    python, cuda = str(env.get("python") or "UNKNOWN"), str(env.get("cuda") or "UNKNOWN")
    if env.get("confidence"):
        evidence.append(f"registry environment declaration ({env['confidence']})")
    for key, value in (("python", python), ("pytorch", torch), ("torchvision", torchvision), ("cuda", cuda)):
        if not _unknown(value):
            evidence.append(f"registry environment {key}={value}")
    if not requirements and not env:
        evidence.append("no pinned requirements.txt or registry environment declaration")
    return {"python": python, "torch": torch, "torchvision": torchvision, "cuda": cuda,
            "packages": packages, "ranges": ranges, "pip_only": sorted(pip_only),
            "confidence": str(env.get("confidence") or ("VERIFIED_REQUIREMENTS" if requirements else "UNKNOWN")),
            "evidence": evidence}


def _terms(value: str, field: str = "") -> list[tuple[str, str]]:
    if _unknown(value) or value == "*":
        return []
    text = value.strip()
    if field == "python" and re.fullmatch(r"\d+\.\d+", text):
        major, minor = map(int, text.split("."))
        text = f">={major}.{minor},<{major}.{minor + 1}"
    elif not re.match(r"^(===|==|!=|~=|>=|<=|>|<)", text):
        text = "==" + text
    terms = []
    for part in text.split(","):
        match = re.fullmatch(r"\s*(===|==|!=|~=|>=|<=|>|<)\s*(.+?)\s*", part)
        if match:
            terms.append((match.group(1), match.group(2)))
    return terms


def _version(value: str) -> Version | None:
    try:
        return Version(value)
    except InvalidVersion:
        return None


def _has_solution(terms: list[tuple[str, str]]) -> bool:
    if not terms:
        return True
    try:
        specifier = SpecifierSet(",".join(op + value for op, value in terms))
    except InvalidSpecifier:
        return False
    exact = [value for op, value in terms if op in ("==", "===") and not value.endswith(".*")]
    if exact:
        return any(specifier.contains(value, prereleases=True) for value in exact)
    candidates = {"0", "0.0.1"}
    for _, raw in terms:
        version = _version(raw.rstrip(".*"))
        if version is None:
            continue
        release = list(version.release)
        candidates.add(str(version))
        for length in range(1, len(release) + 1):
            prefix = release[:length]
            candidates.update((".".join(map(str, prefix + [1])),
                               ".".join(map(str, prefix + [0, 1]))))
            if prefix[-1]:
                candidates.add(".".join(map(str, prefix[:-1] + [prefix[-1] - 1, 1])))
            candidates.add(".".join(map(str, prefix[:-1] + [prefix[-1] + 1])))
    return any(specifier.contains(candidate, prereleases=True) for candidate in candidates)


def _constraints(specs: list[dict]) -> tuple[dict | None, str | None]:
    combined: dict[str, Any] = {"python": [], "torch": [], "torchvision": [], "cuda": [],
                                "packages": {}, "ranges": {}, "pip_only": [], "evidence": []}
    for spec in specs:
        combined["evidence"].extend(spec["evidence"])
        combined["pip_only"].extend(spec.get("pip_only", []))
        for field in ("python", "torch", "torchvision", "cuda"):
            if not _unknown(spec[field]):
                combined[field].extend(_terms(spec[field], field))
        for field in ("packages", "ranges"):
            for name, value in spec[field].items():
                if value not in {"UNKNOWN", "*"}:
                    combined[field].setdefault(name, []).extend(_terms(value, name))
                else:
                    combined[field].setdefault(name, [])
    for field in ("python", "torch", "torchvision", "cuda"):
        combined[field] = sorted(set(combined[field]))
        if not _has_solution(combined[field]):
            return None, f"{field} constraints do not intersect"
    for name in sorted(set(combined["packages"]) | set(combined["ranges"])):
        combined["packages"][name] = sorted(set(combined["packages"].get(name, [])))
        combined["ranges"][name] = sorted(set(combined["ranges"].get(name, [])))
        if not _has_solution(combined["packages"][name] + combined["ranges"][name]):
            return None, f"{name} constraints do not intersect"
    torch, vision = combined["torch"], combined["torchvision"]
    if torch and vision and not any(
            _has_solution(torch + [("==", torch_version)]) and
            _has_solution(vision + [("==", vision_version)])
            for torch_version, vision_version in _TORCHVISION_PAIRS):
        return None, "PyTorch/torchvision constraints have no official binary pairing"
    combined["pip_only"] = sorted(set(combined["pip_only"]))
    combined["evidence"] = sorted(set(combined["evidence"]))
    combined["evidence"].append("conda solve covers Python and conda-resolvable packages; PyTorch, torchvision, CUDA and pip-index packages remain pinned but conda-unresolved")
    return combined, None



def _solver_command() -> str | None:
    return next((name for name in _SOLVERS if shutil.which(name)), None)


_NON_CONDA_PACKAGES = {
    "apex", "clip", "cuda-version", "decord", "fairscale", "ftfy", "horovod", "iopath",
    "open-clip-torch", "opencv-python", "opencv-python-headless", "salesforce-lavis",
    "torch", "torchvision",
}

def _conda_constraints(constraints: dict) -> dict:
    """Keep only constraints conda's configured channels can resolve."""
    pip_only = set(constraints.get("pip_only", ())) | _NON_CONDA_PACKAGES
    return {
        "python": constraints["python"],
        "packages": {name: terms for name, terms in constraints["packages"].items()
                     if name not in pip_only and terms},
        "ranges": {name: terms for name, terms in constraints["ranges"].items()
                   if name not in pip_only and terms},
    }


def _solver_solve(command: str, constraints: dict) -> tuple[bool, str]:
    specs = [f"python{'=' if op in ('==', '===') else op}{value}"
             for op, value in constraints["python"]]
    for name in sorted(set(constraints["packages"]) | set(constraints["ranges"])):
        if name in _NON_CONDA_PACKAGES:
            continue
        terms = constraints["packages"].get(name, []) + constraints["ranges"].get(name, [])
        specs.extend([name] if not terms else
                     [f"{name}{'=' if op in ('==', '===') else op}{value}" for op, value in terms])
    with tempfile.TemporaryDirectory(prefix="workbench-env-solve-") as temp:
        argv = [command, "create", "--dry-run", "--json", "--prefix", str(Path(temp) / "solve"),
                "--override-channels"]
        for channel in CONDA_CHANNELS:
            argv.extend(("-c", channel))
        argv.extend(specs)
        try:
            process = subprocess.run(argv, env=_package_manager_env(), capture_output=True, text=True, check=False,
                                     timeout=SOLVER_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired as exc:
            return False, f"{command} timed out after {exc.timeout} seconds"
        except OSError as exc:
            return False, str(exc)
    message = (process.stderr or process.stdout or f"{command} exited {process.returncode}").strip()
    return process.returncode == 0, message[:500]


def _solver_outcome(message: str) -> str:
    lowered = message.lower()
    if any(marker in lowered for marker in (
            "packagesnotfounderror", "not available from current channels",
            "package not found", "no packages found")):
        return "CHANNEL_GAP"
    if any(marker in lowered for marker in (
            "unsatisfiable", "conflicts with", "inconsistent", "dependency conflict")):
        return "VERSION_CONFLICT"
    return "SOLVER_ERROR"


def _solver_reason(message: str, constraints: dict, scope: str,
                   channels: tuple[str, ...] = CONDA_CHANNELS) -> tuple[str, str]:
    outcome = _solver_outcome(message)
    if outcome == "CHANNEL_GAP":
        detail = f"package unavailable from configured channels: {message or 'no solution'}"
    elif outcome == "VERSION_CONFLICT":
        detail = f"incompatible package/version constraints: {message or 'no solution'}"
    else:
        detail = message or "no solution"
    searched = ", ".join(channels) if channels else "none (injected solver)"
    return outcome, (f"{outcome}: solver rejected {scope} constraints ({_constraint_names(constraints)}): "
                    f"{detail}; channels searched: {searched}")


def _solve(solver: Callable | None, command: str | None, constraints: dict) -> tuple[bool, str, bool]:
    canonical = json.dumps({**{key: constraints[key] for key in
                               ("python", "torch", "torchvision", "cuda", "packages", "ranges")},
                            "channels": CONDA_CHANNELS if solver is None else ()},
                           sort_keys=True, separators=(",", ":"))
    context = ("command", command or "") if solver is None else ("injected", id(solver))
    if solver is not None:
        _SOLVE_CACHE_OWNERS[id(solver)] = solver
    key = (*context, canonical)
    if key in _SOLVE_CACHE:
        ok, detail = _SOLVE_CACHE[key]
        return ok, detail, True
    try:
        # The injected solver receives the full constraint set; conda receives only packages
        # it can resolve, while its configured channels remain explicit on the command line.
        result = (solver(constraints) if solver is not None
                  else _solver_solve(command, _conda_constraints(constraints)))
    except subprocess.TimeoutExpired as exc:
        result = (False, f"{type(exc).__name__}: solver timed out after "
                         f"{exc.timeout or SOLVER_TIMEOUT_SECONDS} seconds")
    except Exception as exc:
        result = (False, f"{type(exc).__name__}: {exc}")
    if isinstance(result, tuple):
        result = (bool(result[0]), str(result[1]) if len(result) > 1 else "")
    elif isinstance(result, dict):
        result = (bool(result.get("success", result.get("ok", False))),
                  str(result.get("reason", result.get("message", ""))))
    else:
        result = (bool(result), "" if result else "solver rejected combined requirements")
    _SOLVE_CACHE[key] = result
    return result[0], result[1], False



def _reusable_interpreter(members: list[str], by_id: dict[str, dict], specs: dict[str, dict]) -> str | None:
    paths = []
    for mid in members:
        variable = (by_id[mid].get("environment") or {}).get("python_env_var")
        path = os.environ.get(variable or "")
        if not path or _is_staging_path(path):
            return None
        paths.append(path)
    try:
        if len({str(Path(path).resolve()) for path in paths}) != 1:
            return None
    except OSError:
        return None
    for mid in members:
        try:
            report = runtime.verify_environment(_runtime_model(by_id[mid], specs[mid]), probe=True)
        except Exception:
            return None
        if report.tier not in runtime.READY_TIERS or report.status == "MODEL_IMPORT_FAILED":
            return None
    return paths[0]


def _has_source_or_checkpoint(model: dict) -> bool:
    return ((model.get("source_available") is True or
             (model.get("source_available") is not False and bool(model.get("source_dir")))) or
            bool(model.get("checkpoint_variants")) or bool(model.get("checkpoint_bundles")) or
            "source_available" not in model)


def _constraint_names(constraints: dict) -> str:
    names = [field for field in ("python", "torch", "torchvision", "cuda") if constraints[field]]
    names.extend(sorted(name for name, terms in constraints["packages"].items()
                        if terms or constraints["ranges"].get(name)))
    return ", ".join(names) or "python, torch, torchvision, cuda, packages"


def _model_id(model: dict) -> str:
    return str(model.get("model_id") or "<unnamed>")


def _environment_digest(members: list[str], constraints: dict) -> str:
    lock = {"members": sorted(members), "constraints": {key: constraints[key] for
            key in ("python", "torch", "torchvision", "cuda", "packages", "ranges")}}
    return hashlib.sha256(json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _environment_id(members: list[str], constraints: dict) -> str:
    return f"cir-{_environment_digest(members, constraints)[:16]}"


def _source_reason(message: str, specs: list[dict]) -> str:
    sources = sorted({evidence for spec in specs for evidence in spec["evidence"]})
    provenance = ", ".join(sources) if sources else "registry/model metadata (no source-file evidence available)"
    return f"{message}; source: {provenance}"


def _display(terms: list[tuple[str, str]]) -> str | None:
    if not terms:
        return None
    exact = {value for op, value in terms if op in ("==", "===")}
    return next(iter(exact)) if len(exact) == 1 else ",".join(op + value for op, value in terms)


def _runtime_model(model: dict, spec: dict) -> dict:
    result = dict(model)
    env = dict(model.get("environment") or {})
    env["packages"] = [f"{name}=={version}" if version not in ("UNKNOWN", "*") else name
                       for name, version in spec["packages"].items()]
    env["packages"].extend(name for name in spec["ranges"])
    if not _unknown(spec["torch"]):
        env["pytorch"] = spec["torch"]
    elif "pytorch" not in (model.get("environment") or {}):
        env.pop("pytorch", None)
    if not _unknown(spec["torchvision"]):
        env["torchvision"] = spec["torchvision"]
    elif "torchvision" not in (model.get("environment") or {}):
        env.pop("torchvision", None)
    if not _unknown(spec["python"]):
        env["python"] = spec["python"]
    result["environment"] = env
    return result


def _group_status(group: dict, models: dict[str, dict], specs: dict[str, dict], solver_status: str):
    paths, reports = [], []
    for mid in group["members"]:
        model = models[mid]
        variable = (model.get("environment") or {}).get("python_env_var")
        path = os.environ.get(variable or "")
        if not path:
            continue
        paths.append(path)
        if _is_staging_path(path):
            reports.append(None)
            group["unresolved"].append(f"{mid}: interrupted staging environment cannot be reused")
            continue
        if Path(path).resolve() == Path(sys.executable).resolve():
            reports.append(None)
            group["unresolved"].append(f"{mid}: system Python cannot be a model environment target")
            continue
        try:
            reports.append(runtime.verify_environment(_runtime_model(model, specs[mid]), probe=True))
        except Exception as exc:
            reports.append(None)
            group["unresolved"].append(f"{mid}: runtime verification failed: {exc}")
    same = len(paths) == len(group["members"]) and len({str(Path(path).resolve()) for path in paths}) == 1
    ready = len(reports) == len(group["members"]) and all(
        report is not None and report.tier in runtime.READY_TIERS and report.status != "MODEL_IMPORT_FAILED"
        for report in reports)
    import_failed = any(report is None or report.status == "MODEL_IMPORT_FAILED" for report in reports)
    if not any(models[mid].get("environment_required") for mid in group["members"]) and not paths:
        installation = "NOT_REQUIRED"
    elif same and ready:
        installation = "REUSED"
    elif paths:
        installation = "EXISTING_UNVERIFIED"
    else:
        installation = "NOT_INSTALLED"
    if same and ready:
        verification = "READY"
    elif solver_status == "unavailable":
        verification = "SOLVER_UNAVAILABLE"
    elif import_failed:
        verification = "IMPORT_CHECK_FAILED"
    elif paths:
        verification = "NOT_READY"
    else:
        verification = "NOT_CONFIGURED"
    return installation, verification, paths[0] if same and ready else None


def plan_environments(models: list[dict], *, solver=None,
                      artifacts_root: Path | None = None,
                      third_party_root: Path | str | None = None) -> dict:
    """Build largest safe groups; solver-check each proposed full set when available."""
    command = _solver_command() if solver is None else None
    solver_name = "injected" if solver is not None else command or "unavailable"
    by_id = {_model_id(model): model for model in models}
    solver_status = "available" if solver is not None or command else "unavailable"
    specs = {mid: model_constraints(model, third_party_root=third_party_root) for mid, model in by_id.items()}
    order = sorted(by_id, key=lambda mid: (
        -(len(specs[mid]["packages"]) + len(specs[mid]["ranges"]) +
          sum(not _unknown(specs[mid][key]) for key in ("python", "torch", "torchvision", "cuda"))),
        mid))
    solver_invoked = False
    solver_refused = False

    drafts, blocked = [], []
    for mid in order:
        model = by_id[mid]
        if not _has_source_or_checkpoint(model):
            blocked.append({"model_id": mid, "reason":
                            f"source unavailable and no checkpoint variants; source: registry model record for {mid} (source_available=false, checkpoint_variants empty)"})
            continue
        rejected, joined = [], False
        for group in sorted(drafts, key=lambda item: (-len(item["members"]), item["members"])):
            members = group["members"] + [mid]
            union = [specs[item] for item in members]
            constraints, conflict = _constraints(union)
            if conflict:
                reason = _source_reason(conflict, union)
            else:
                reused = _reusable_interpreter(members, by_id, specs)
                if reused:
                    group["members"].append(mid)
                    group["constraints"] = constraints
                    group["split_reasons"].extend(rejected)
                    group["reuse"] = reused
                    group["solver_verification_status"] = "NOT_RUN_REUSED"
                    joined = True
                    break
                if solver_status == "available":
                    solver_invoked = True
                    ok, detail, cached = _solve(solver, command, constraints)
                    if not ok:
                        outcome, message = _solver_reason(
                            detail, constraints, "combined", CONDA_CHANNELS if solver is None else ())
                        group["solver_outcomes"].append(outcome)
                        group["channels_used"] = group["channels_used"] or solver is None
                        reason = _source_reason(message, union)
                    else:
                        group["members"].append(mid)
                        group["constraints"] = constraints
                        group["split_reasons"].extend(rejected)
                        group["solver_verification_status"] = "CACHED_VERIFIED" if cached else "VERIFIED"
                        group["channels_used"] = group["channels_used"] or solver is None
                        joined = True
                        break
                else:
                    group["members"].append(mid)
                    group["constraints"] = constraints
                    group["split_reasons"].extend(rejected)
                    group["solver_verification_status"] = "UNAVAILABLE"
                    joined = True
                    break
            group["split_reasons"].append(reason)
            rejected.append(reason)
        if joined:
            continue
        constraints, conflict = _constraints([specs[mid]])
        if conflict:
            blocked.append({"model_id": mid, "reason": _source_reason(conflict, [specs[mid]])})
            continue
        reused = _reusable_interpreter([mid], by_id, specs)
        if reused:
            solve_status = "NOT_RUN_REUSED"
        elif solver_status == "available":
            solver_invoked = True
            ok, detail, cached = _solve(solver, command, constraints)
            if not ok:
                solver_refused = True
                outcome, message = _solver_reason(
                    detail, constraints, "standalone", CONDA_CHANNELS if solver is None else ())
                blocked.append({"model_id": mid, "solver_outcome": outcome,
                                "reason": _source_reason(message, [specs[mid]])})
                continue
            solve_status = "CACHED_VERIFIED" if cached else "VERIFIED"
        else:
            solve_status = "UNAVAILABLE"
        drafts.append({"members": [mid], "constraints": constraints, "split_reasons": rejected,
                       "reuse": reused, "solver_verification_status": solve_status,
                       "solver_outcomes": [],
                       "channels_used": solver is None and solve_status in ("VERIFIED", "CACHED_VERIFIED")})

    if artifacts_root is None:
        from workbench.backend.operator_config import resolve_config

        artifacts_root = resolve_config().WORKBENCH_ARTIFACTS_ROOT
    groups = []
    for draft in drafts:
        members, constraints = sorted(draft["members"]), draft["constraints"]
        environment_id = _environment_id(members, constraints)
        env_path = artifacts_root / "environments" / environment_id
        unresolved = []
        for mid in members:
            spec = specs[mid]
            unresolved.extend(f"{mid}: {field} constraint UNKNOWN (source does not establish value)"
                              for field in ("python", "torch", "torchvision", "cuda") if _unknown(spec[field]))
            unresolved.extend(f"{mid}: {name} version UNKNOWN (source does not establish value)"
                              for name, value in spec["packages"].items() if value == "UNKNOWN")
            unresolved.extend(f"{mid}: {name} is unversioned in source"
                              for name, value in spec["packages"].items() if value == "*")
        group_packages = {name: _display(constraints["packages"].get(name, []) +
                                         constraints["ranges"].get(name, [])) or "*"
                          for name in sorted(set(constraints["packages"]) | set(constraints["ranges"]))}
        group_packages.update({"torch": _display(constraints["torch"]) or "*",
                               "torchvision": _display(constraints["torchvision"]) or "*",
                               "cuda-version": _display(constraints["cuda"]) or "*"})
        group = {"environment_id": environment_id, "members": members,
                 "python": _display(constraints["python"]),
                 "packages": group_packages,
                 "cuda_runtime": _display(constraints["cuda"]), "conda_env_path": str(env_path),
                 "interpreter": str(env_path / ("python.exe" if os.name == "nt" else "bin/python")),
                 "lock_digest": _environment_digest(members, constraints),
                 "evidence": constraints["evidence"] + [
                     f"Configured conda channels: {', '.join(CONDA_CHANNELS)}"] + (
                     [f"Conda channels searched: {', '.join(CONDA_CHANNELS)}"]
                     if draft["channels_used"] else []),
                 "split_reasons": sorted(set(draft["split_reasons"])),
                 "installation_status": "NOT_INSTALLED", "runtime_verification_status": "NOT_CONFIGURED",
                 "solver_verification_status": draft["solver_verification_status"],
                 "solver_outcomes": sorted(set(draft["solver_outcomes"])),
                 "unresolved": sorted(set(unresolved)),
                 "constraints": constraints,
                 "channels_used": draft["channels_used"]}
        installation, verification, reused = _group_status(group, by_id, specs, solver_status)
        group["installation_status"], group["runtime_verification_status"] = installation, verification
        if reused:
            group["interpreter"] = reused
            group["conda_env_path"] = str(Path(reused).parent.parent)
            group["runtime_evidence"] = f"verified configured interpreter: {reused}"
        elif draft["reuse"]:
            group["runtime_evidence"] = f"verified configured interpreter: {draft['reuse']}"
        if installation == "REUSED" and verification == "READY":
            state = "RUNTIME_VERIFIED"
        elif draft["solver_verification_status"] in ("VERIFIED", "CACHED_VERIFIED"):
            state = "SOLVER_COMPATIBLE"
        else:
            state = "CONSTRAINTS_COMPATIBLE"
        group["state"] = state
        group["evidence"].append(f"state: {state}")
        groups.append(group)
    if solver_status == "unavailable":
        overall_solver_status = "UNAVAILABLE"
    elif not solver_invoked:
        overall_solver_status = "NOT_RUN"
    elif solver_refused:
        overall_solver_status = "PARTIAL"
    else:
        overall_solver_status = "VERIFIED"
    if not solver_invoked:
        for group in groups:
            if group["solver_verification_status"] != "NOT_RUN_REUSED":
                group["solver_verification_status"] = "NOT_RUN"
    plan = {"schema_version": 1, "solver": solver_name,
            "solver_verification_status": overall_solver_status, "groups": groups,
            "blocked": sorted(blocked, key=lambda item: item["model_id"])}
    plan["interpreter_mapping"] = apply_interpreter_mapping(models, plan)
    plan["third_party_root"] = str(resolve_third_party_root(third_party_root))
    return plan


def apply_interpreter_mapping(models: list[dict], plan: dict,
                              environ: dict[str, str] | None = None) -> dict[str, str | None]:
    """Return model-to-interpreter map without replacing explicit operator variables."""
    environ = os.environ if environ is None else environ
    by_id = {_model_id(model): model for model in models}
    mapping = {}
    for group in plan.get("groups", []):
        for mid in group["members"]:
            model = by_id.get(mid)
            variable = (model.get("environment") or {}).get("python_env_var") if model else None
            interpreter = environ.get(variable) if variable and environ.get(variable) else group.get("interpreter")
            mapping[mid] = (None if _is_staging_path(group.get("conda_env_path")) or _is_staging_path(interpreter)
                            else interpreter)
    return mapping


def _pip_specs_for_group(group: dict, models: dict[str, dict]) -> list[str]:
    constraints = group["constraints"]
    specs = []
    torch_terms = constraints.get("torch", [])
    if torch_terms:
        torch_spec = _display(torch_terms)
        if torch_spec and not _unknown(torch_spec):
            specs.append(f"torch{'' if any(torch_spec.startswith(op) for op in ('==', '>=', '<=', '>', '<', '~=', '!=')) else '=='}{torch_spec}")
    vision_terms = constraints.get("torchvision", [])
    if vision_terms:
        vision_spec = _display(vision_terms)
        if vision_spec and not _unknown(vision_spec):
            specs.append(f"torchvision{'' if any(vision_spec.startswith(op) for op in ('==', '>=', '<=', '>', '<', '~=', '!=')) else '=='}{vision_spec}")

    pip_only_names = set(constraints.get("pip_only", [])) | _NON_CONDA_PACKAGES
    for name in sorted(set(constraints.get("packages", {})) | set(constraints.get("ranges", {}))):
        if name in ("cuda-version", "torch", "torchvision"):
            continue
        if name in pip_only_names:
            terms = constraints.get("packages", {}).get(name, []) + constraints.get("ranges", {}).get(name, [])
            disp = _display(terms)
            if disp and not _unknown(disp):
                specs.append(f"{name}{'' if any(disp.startswith(op) for op in ('==', '>=', '<=', '>', '<', '~=', '!=')) else '=='}{disp}")
            else:
                specs.append(name)
    return specs


def _verify_group_interpreter(
    group: dict,
    by_id: dict[str, dict],
    specs: dict[str, dict],
    third_party_root: Path | None = None,
) -> tuple[bool, str, dict]:
    interpreter = Path(group["interpreter"])
    if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
        return False, f"Interpreter {interpreter} is missing or not executable", {}

    details: dict[str, Any] = {"verified_versions": {}, "evaluator_imports": {}, "probes": []}
    try:
        pip_res = subprocess.run([str(interpreter), "-m", "pip", "check"],
                                 env=_package_manager_env(),
                                 capture_output=True, text=True, check=False, timeout=60)
    except Exception as exc:
        return False, f"pip check invocation failed: {exc}", details
    details["pip_check"] = {"returncode": pip_res.returncode, "stdout": pip_res.stdout.strip(), "stderr": pip_res.stderr.strip()}
    if pip_res.returncode != 0:
        return False, f"pip check failed: {pip_res.stdout.strip()} {pip_res.stderr.strip()}", details

    prefix = str(Path(group["conda_env_path"]).resolve())
    try:
        installed, installed_modules, _ = runtime.inspect_environment_distributions(prefix)
    except Exception as exc:
        return False, f"Failed to inspect environment distributions: {exc}", details

    constraints = group["constraints"]
    for name in sorted(set(constraints.get("packages", {})) | set(constraints.get("ranges", {}))):
        if name == "cuda-version":
            continue
        terms = constraints.get("packages", {}).get(name, []) + constraints.get("ranges", {}).get(name, [])
        if terms:
            spec_str = ",".join(f"{op}{val}" for op, val in terms)
            norm_name = runtime.normalize_distribution_name(name)
            inst_ver = installed.get(norm_name)
            details["verified_versions"][name] = {"required": spec_str, "installed": inst_ver}
            if inst_ver is None:
                return False, f"Required package {name} ({spec_str}) is not installed", details
            satisfies = runtime.version_satisfies(spec_str, inst_ver)
            if satisfies is False or satisfies is None:
                return False, f"Package {name} version mismatch: required {spec_str}, installed {inst_ver}", details

    for key, field_name in (("torch", "torch"), ("torchvision", "torchvision")):
        terms = constraints.get(key, [])
        if terms:
            spec_str = ",".join(f"{op}{val}" for op, val in terms)
            norm_name = runtime.normalize_distribution_name(field_name)
            inst_ver = installed.get(norm_name)
            details["verified_versions"][field_name] = {"required": spec_str, "installed": inst_ver}
            if inst_ver is None:
                return False, f"Required package {field_name} ({spec_str}) is not installed", details
            satisfies = runtime.version_satisfies(spec_str, inst_ver)
            if satisfies is False or satisfies is None:
                return False, f"{field_name} version mismatch: required {spec_str}, installed {inst_ver}", details
    probe_modules = []
    for pkg in sorted(set(constraints.get("packages", {})) | {"torch", "torchvision"}):
        if pkg in ("cuda-version",) or pkg not in installed:
            continue
        cand = [k for k, v in _IMPORT_DISTRIBUTIONS.items() if pkg in v]
        mod = cand[0] if cand else pkg.replace("-", "_")
        probe_modules.append(mod)

    if probe_modules:
        code = "import " + ", ".join(probe_modules)
        try:
            res = subprocess.run([str(interpreter), "-c", code],
                                 env=_package_manager_env(),
                                 capture_output=True, text=True, check=False, timeout=60)
        except Exception as exc:
            return False, f"Native module import probe failed: {exc}", details
        if res.returncode != 0:
            return False, f"Native module import probe failed ({code}): {res.stderr.strip()}", details

    for mid in group["members"]:
        model = by_id.get(mid)
        if model is None:
            continue
        spec = specs.get(mid) or {}
        eval_imports = _evaluator_imports(model, third_party_root)
        if eval_imports is None:
            # `None` means the trace could not run, which in practice means the pinned checkout is
            # not present yet. Treating that as "nothing to check" would mark an environment
            # RUNTIME_VERIFIED without ever having looked at what the evaluator imports -- and the
            # same interpreter then fails once the source arrives. An unverifiable requirement is a
            # refusal, not a pass.
            source = (resolve_third_party_root(third_party_root) / str(model.get("source_dir"))).resolve() \
                if model.get("source_dir") else None
            detail = f"pinned source not present ({source})" if source and not source.is_dir() \
                else "no evaluator entry point could be resolved for this model"
            details["evaluator_imports"][mid] = {"modules": None, "returncode": None, "reason": detail}
            return False, f"{mid}: cannot verify evaluator imports: {detail}", details
        if eval_imports:
            code = "; ".join(f"import {m}" for m in sorted(eval_imports))
            try:
                res = subprocess.run([str(interpreter), "-c", code],
                                     env=_package_manager_env(),
                                     capture_output=True, text=True, check=False, timeout=60)
            except Exception as exc:
                return False, f"{mid}: evaluator import probe failed: {exc}", details
            details["evaluator_imports"][mid] = {"modules": list(eval_imports), "returncode": res.returncode}
            if res.returncode != 0:
                return False, f"{mid}: evaluator import probe failed ({code}): {res.stderr.strip()}", details

        var = (model.get("environment") or {}).get("python_env_var") or f"WORKBENCH_{mid.upper()}_PYTHON"
        old_val = os.environ.get(var)
        try:
            os.environ[var] = str(interpreter)
            rmodel = _runtime_model(model, spec)
            rmodel.setdefault("environment", {})["python_env_var"] = var
            report = runtime.verify_environment(rmodel, probe=True)
            if report.tier not in runtime.READY_TIERS or report.status in ("MODEL_IMPORT_FAILED", "UNSAFE_INTERPRETER"):
                return False, f"{mid}: runtime.verify_environment failed: {report.status} ({report.reason})", details
        finally:
            if old_val is not None:
                os.environ[var] = old_val
            else:
                os.environ.pop(var, None)

    return True, "All checks passed", details


def _build_group_dict(
    members: list[str],
    constraints: dict,
    specs: dict[str, dict],
    artifacts_root: Path,
    solver_cmd: str | None,
) -> dict:
    members = sorted(members)
    environment_id = _environment_id(members, constraints)
    env_path = artifacts_root / "environments" / environment_id
    unresolved = []
    for mid in members:
        spec = specs[mid]
        unresolved.extend(f"{mid}: {field} constraint UNKNOWN (source does not establish value)"
                          for field in ("python", "torch", "torchvision", "cuda") if _unknown(spec[field]))
        unresolved.extend(f"{mid}: {name} version UNKNOWN (source does not establish value)"
                          for name, value in spec["packages"].items() if value == "UNKNOWN")
        unresolved.extend(f"{mid}: {name} is unversioned in source"
                          for name, value in spec["packages"].items() if value == "*")
    group_packages = {name: _display(constraints["packages"].get(name, []) +
                                     constraints["ranges"].get(name, [])) or "*"
                      for name in sorted(set(constraints["packages"]) | set(constraints["ranges"]))}
    group_packages.update({"torch": _display(constraints["torch"]) or "*",
                           "torchvision": _display(constraints["torchvision"]) or "*",
                           "cuda-version": _display(constraints["cuda"]) or "*"})
    return {
        "environment_id": environment_id,
        "members": members,
        "python": _display(constraints["python"]),
        "packages": group_packages,
        "cuda_runtime": _display(constraints["cuda"]),
        "conda_env_path": str(env_path),
        "interpreter": str(env_path / ("python.exe" if os.name == "nt" else "bin/python")),
        "lock_digest": _environment_digest(members, constraints),
        "evidence": list(constraints["evidence"]),
        "split_reasons": [],
        "installation_status": "NOT_INSTALLED",
        "runtime_verification_status": "NOT_CONFIGURED",
        "solver_verification_status": "VERIFIED" if solver_cmd else "UNAVAILABLE",
        "solver_outcomes": [],
        "unresolved": sorted(set(unresolved)),
        "constraints": constraints,
        "channels_used": bool(solver_cmd),
        "state": "SOLVER_COMPATIBLE" if solver_cmd else "CONSTRAINTS_COMPATIBLE",
    }


def _provision_group(
    group: dict,
    by_id: dict[str, dict],
    specs: dict[str, dict],
    artifacts_root: Path,
    third_party_root: Path | None,
    allowed: bool,
    solver_cmd: str | None,
    network_allowed: bool,
) -> tuple[bool, str]:
    interpreter = Path(group["interpreter"])
    prefix = Path(group["conda_env_path"])
    if _is_staging_path(prefix) or _is_staging_path(interpreter):
        group["installation_status"] = "FAILED"
        group["runtime_verification_status"] = "FAILED"
        group["state"] = "FAILED"
        return False, "Interrupted staging prefixes cannot be adopted as environments"

    if interpreter.is_file() and os.access(interpreter, os.X_OK) and not prefix.name.startswith(".staging-"):
        ok, reason, details = _verify_group_interpreter(group, by_id, specs, third_party_root)
        if ok:
            console_ok, console_reason = _verify_console_scripts(prefix, interpreter)
            group["evidence"].append(f"console-script check (reuse): {console_reason}")
            if not console_ok:
                group["installation_status"] = "EXISTING_UNVERIFIED"
                group["runtime_verification_status"] = "CONSOLE_SCRIPT_STALE"
                group["state"] = "FAILED"
                return False, console_reason
            group["installation_status"] = "REUSED"
            group["runtime_verification_status"] = "READY"
            group["state"] = "RUNTIME_VERIFIED"
            group["evidence"].append(f"RUNTIME_VERIFIED: reused verified interpreter: {interpreter}")
            return True, "Reused existing verified environment"

    if not allowed:
        group["installation_status"] = "NOT_INSTALLED"
        group["runtime_verification_status"] = "NOT_CONFIGURED"
        group["evidence"].append("Installation not authorized (allowed=False)")
        return False, "Installation not authorized"

    if not solver_cmd:
        group["installation_status"] = "FAILED"
        group["runtime_verification_status"] = "SOLVER_UNAVAILABLE"
        return False, "No conda/mamba/micromamba solver available to create environment"

    # Installing means the package manager must fetch packages. `allow_env_install` authorizes
    # writing an environment, not reaching the network, and conda/pip open a connection even when a
    # local cache happens to satisfy the request. Refuse rather than silently egress.
    if not network_allowed:
        group["installation_status"] = "FAILED"
        group["runtime_verification_status"] = "NOT_CONFIGURED"
        group["state"] = "BLOCKED_AUTHORIZATION_REQUIRED"
        group["evidence"].append("installation refused: allow_network is not authorized")
        return False, "environment installation requires network authorization (allow_network)"

    # Build at the final, content-derived prefix directly. A conda prefix is not relocatable once
    # created: the interpreter records its own prefix, console scripts get that absolute path in
    # their shebang, and conda-meta holds absolute paths for every installed package. Creating at
    # `.staging-...` and renaming only worked because nothing had checked a shebang yet, which is
    # why the rename was followed by a verification that could no longer detect a stale path.
    #
    # An occupied prefix is never overwritten and never deleted: the previous successful
    # environment may be the only working one on the machine. A replacement is built beside it
    # under a new name and published only after it verifies.
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with _environment_lock(prefix.parent, group["environment_id"]):
        # Chosen under the lock. Selecting it first was a time-of-check/time-of-use race: two
        # provisioners picked the same free path, the loser's `conda create` failed on an occupied
        # prefix, and its cleanup deleted the environment the winner had just verified.
        target_prefix = prefix
        if target_prefix.exists():
            # Someone else may have installed it while we waited for the lock: reuse it if it now
            # verifies, rather than building a duplicate beside it.
            existing_interp = target_prefix / ("python.exe" if os.name == "nt" else "bin/python")
            if existing_interp.is_file() and os.access(existing_interp, os.X_OK):
                probe = dict(group)
                probe["conda_env_path"] = str(target_prefix)
                probe["interpreter"] = str(existing_interp)
                ok, _, _ = _verify_group_interpreter(probe, by_id, specs, third_party_root)
                if ok and _verify_console_scripts(target_prefix, existing_interp)[0]:
                    group["conda_env_path"] = str(target_prefix)
                    group["interpreter"] = str(existing_interp)
                    group["installation_status"] = "REUSED"
                    group["runtime_verification_status"] = "READY"
                    group["state"] = "RUNTIME_VERIFIED"
                    group["evidence"].append(
                        f"RUNTIME_VERIFIED: adopted concurrently provisioned environment {target_prefix}")
                    return True, "Reused environment provisioned concurrently"
            idx = 1
            while (prefix.parent / f"{prefix.name}-{idx}").exists():
                idx += 1
            target_prefix = prefix.parent / f"{prefix.name}-{idx}"
            group["evidence"].append(
                f"existing prefix left untouched: {prefix}; replacement built at {target_prefix}")

        staging_prefix = target_prefix
        staging_interp = staging_prefix / ("python.exe" if os.name == "nt" else "bin/python")
        return _install_at_prefix(group, by_id, specs, staging_prefix, staging_interp, target_prefix,
                                  artifacts_root, third_party_root, solver_cmd,
                                  network_allowed=network_allowed)


def _install_at_prefix(
    group: dict,
    by_id: dict[str, dict],
    specs: dict[str, dict],
    staging_prefix: Path,
    staging_interp: Path,
    target_prefix: Path,
    artifacts_root: Path,
    third_party_root: Path | None,
    solver_cmd: str,
    network_allowed: bool,
    ) -> tuple[bool, str]:
    """Create and verify the environment at its own final path, then mark it managed."""

    conda_specs = []
    py_terms = group["constraints"].get("python", [])
    if py_terms:
        for op, val in py_terms:
            conda_specs.append(f"python{'=' if op in ('==', '===') else op}{val}")
    else:
        conda_specs.append("python=3.10")
    conda_specs.append("pip")

    conda_constraints = _conda_constraints(group["constraints"])
    for name in sorted(set(conda_constraints["packages"]) | set(conda_constraints["ranges"])):
        terms = conda_constraints["packages"].get(name, []) + conda_constraints["ranges"].get(name, [])
        if not terms:
            conda_specs.append(name)
        else:
            for op, val in terms:
                conda_specs.append(f"{name}{'=' if op in ('==', '===') else op}{val}")

    argv = [solver_cmd, "create", "--prefix", str(staging_prefix), "--override-channels"]
    for ch in CONDA_CHANNELS:
        argv.extend(("-c", ch))
    argv.extend(conda_specs)
    argv.append("--yes")

    group["evidence"].append(f"conda create: {' '.join(argv)}")
    try:
        proc = subprocess.run(argv, env=_package_manager_env(), capture_output=True, text=True, check=False, timeout=SOLVER_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as exc:
        shutil.rmtree(staging_prefix, ignore_errors=True)
        group["evidence"].append(f"conda create timed out after {exc.timeout}s")
        group["installation_status"] = "FAILED"
        group["state"] = "FAILED"
        return False, f"conda create timed out after {exc.timeout}s"
    except OSError as exc:
        shutil.rmtree(staging_prefix, ignore_errors=True)
        group["installation_status"] = "FAILED"
        group["state"] = "FAILED"
        return False, f"conda create failed: {exc}"

    group["evidence"].append(f"conda create exited {proc.returncode}")
    if proc.returncode != 0:
        shutil.rmtree(staging_prefix, ignore_errors=True)
        group["installation_status"] = "FAILED"
        group["state"] = "FAILED"
        err = (proc.stderr or proc.stdout).strip()[:500]
        group["evidence"].append(f"conda create error: {err}")
        return False, f"conda create exited {proc.returncode}: {err}"

    if not staging_interp.is_file():
        shutil.rmtree(staging_prefix, ignore_errors=True)
        group["installation_status"] = "FAILED"
        group["state"] = "FAILED"
        return False, f"Target interpreter {staging_interp} not found after conda create"

    pip_specs = _pip_specs_for_group(group, by_id)
    if pip_specs:
        pip_argv = [str(staging_interp), "-m", "pip", "install", *pip_specs]
        group["evidence"].append(f"pip install: {' '.join(pip_argv)}")
        try:
            pip_proc = subprocess.run(pip_argv, env=_package_manager_env(), capture_output=True, text=True, check=False, timeout=SOLVER_TIMEOUT_SECONDS)
        except Exception as exc:
            shutil.rmtree(staging_prefix, ignore_errors=True)
            group["installation_status"] = "FAILED"
            group["state"] = "FAILED"
            return False, f"pip install failed: {exc}"

        group["evidence"].append(f"pip install exited {pip_proc.returncode}")
        if pip_proc.returncode != 0:
            shutil.rmtree(staging_prefix, ignore_errors=True)
            group["installation_status"] = "FAILED"
            group["state"] = "FAILED"
            err = (pip_proc.stderr or pip_proc.stdout).strip()[:500]
            group["evidence"].append(f"pip install error: {err}")
            return False, f"pip install exited {pip_proc.returncode}: {err}"

    # Verified where it was created, so the checks below see the paths the environment will
    # actually use: a console-script shebang, an entrypoint execution, the model imports.
    staging_group = dict(group)
    staging_group["conda_env_path"] = str(staging_prefix)
    staging_group["interpreter"] = str(staging_interp)
    ok, reason, details = _verify_group_interpreter(staging_group, by_id, specs, third_party_root)
    group["verification_details"] = details
    if not ok:
        shutil.rmtree(staging_prefix, ignore_errors=True)
        group["installation_status"] = "FAILED"
        group["runtime_verification_status"] = "IMPORT_CHECK_FAILED"
        group["state"] = "FAILED"
        group["evidence"].append(f"Verification failed: {reason}")
        return False, reason

    console_ok, console_reason = _verify_console_scripts(staging_prefix, staging_interp)
    group["evidence"].append(f"console-script check: {console_reason}")
    if not console_ok:
        shutil.rmtree(staging_prefix, ignore_errors=True)
        group["installation_status"] = "FAILED"
        group["runtime_verification_status"] = "CONSOLE_SCRIPT_STALE"
        group["state"] = "FAILED"
        return False, console_reason

    # Nothing is published as ready until every check above passed at this exact path.
    _mark_workbench_managed(target_prefix, group["environment_id"])
    group["conda_env_path"] = str(target_prefix)
    group["interpreter"] = str(staging_interp)
    group["installation_status"] = "INSTALLED"
    group["state"] = "RUNTIME_VERIFIED"
    group["runtime_verification_status"] = "READY"
    group["evidence"].append("INSTALLED: conda and pip package installation succeeded")
    group["evidence"].append("RUNTIME_VERIFIED: pip check, exact versions, and member evaluator imports verified")
    return True, "Provisioning succeeded"


def _verify_console_scripts(prefix: Path, interpreter: Path) -> tuple[bool, str]:
    """Every python console script in this prefix must run a python that lives in this prefix.

    This is the check that distinguishes a real environment from a relocated one. `conda create
    --prefix` writes absolute shebangs: `bin/pip`, `bin/pydoc3.11` and friends name the python
    binary by its path *at install time*. Renaming the directory leaves those shebangs pointing at
    a path that no longer exists, and the environment looks fine right up to the first console
    script invocation. Shell helpers (`bzcmp`, `xzdiff`, `*-config`) are `#!/bin/sh` by design and
    are not this prefix's business, so only python shebangs are judged.
    """
    bin_dir = prefix / ("Scripts" if os.name == "nt" else "bin")
    if not bin_dir.is_dir():
        return True, "no bin directory to inspect"

    # The set of interpreters that legitimately belong to this prefix: whatever `bin/python*`
    # executables exist here, resolved. A console script may name any of them.
    acceptable: set[str] = set()
    for candidate in sorted(bin_dir.glob("python*")):
        if candidate.is_file() and candidate.name.startswith("python"):
            try:
                acceptable.add(str(candidate.resolve()))
            except OSError:
                continue
    if not acceptable:
        try:
            acceptable.add(str(interpreter.resolve()))
        except OSError:
            return False, f"cannot resolve the environment interpreter: {interpreter}"

    inspected = 0
    for script in sorted(bin_dir.iterdir()):
        if script.is_dir() or script.suffix == ".exe":
            continue
        if script.name.startswith("python"):
            # `bin/python`, `bin/python3`, `bin/python3.11` *are* the environment's interpreters
            # (a conda prefix ships them as binaries, a virtualenv as a small stub), so they are not
            # console scripts to be judged against themselves.
            continue
        try:
            if not os.access(script, os.R_OK) or script.stat().st_size == 0:
                continue
            head = script.open("rb").readline()
        except OSError:
            continue
        if not head.startswith(b"#!"):
            continue
        if head[2:].strip().endswith(b"\r"):
            head = head[:-1] + b"\n"
        parts = head[2:].decode("utf-8", "replace").strip().split()
        if not parts:
            return False, f"console script {script.name} has a truncated shebang"
        uses_env = len(parts) > 1 and Path(parts[0]).name == "env"
        shebang = parts[-1] if uses_env else parts[0]
        # Decide by basename, not by the whole string: a conda shebang is an absolute path like
        # `/prefix/bin/python3.11`, and testing the full path for a "python" prefix skipped exactly
        # the case this check exists for.
        if not Path(shebang).name.startswith("python"):
            continue
        if uses_env and Path(shebang).name in {"python", "python3"}:
            # `#!/usr/bin/env python` deliberately defers to PATH, which inside this prefix is this
            # prefix's python; there is no absolute path that can go stale.
            continue
        inspected += 1
        try:
            resolved = str(Path(shebang).resolve())
        except OSError:
            return False, f"console script {script.name} points at an unresolvable {shebang}"
        if resolved not in acceptable:
            return False, (f"console script {script.name} points at {shebang}, which is not a python "
                           f"of this environment ({sorted(acceptable)[0]} expected)")
    return True, f"{inspected} python console-script shebang(s) point into this environment"


def _provision_all_groups(
    initial_groups: list[dict],
    by_id: dict[str, dict],
    specs: dict[str, dict],
    artifacts_root: Path,
    third_party_root: Path | None,
    allowed: bool,
    solver_cmd: str | None,
    network_allowed: bool,
) -> list[dict]:
    queue = list(initial_groups)
    completed = []

    while queue:
        group = queue.pop(0)
        attempts = group.get("_attempts", 0)
        ok, reason = _provision_group(group, by_id, specs, artifacts_root, third_party_root, allowed,
                                      solver_cmd, network_allowed=network_allowed)
        if ok or not allowed or len(group["members"]) <= 1 or attempts >= 2:
            if not ok and allowed and group.get("state") != "BLOCKED_AUTHORIZATION_REQUIRED":
                # A refusal that already recorded a precise state (for example a missing network
                # authorization) keeps it; flattening it to FAILED loses the reason the operator
                # needs in order to grant the right permission.
                group["installation_status"] = "FAILED"
                group["state"] = "FAILED"
                group["runtime_verification_status"] = "FAILED"
                group["unresolved"].append(f"provisioning failed: {reason}")
            elif not ok and group.get("state") == "BLOCKED_AUTHORIZATION_REQUIRED":
                group["unresolved"].append(f"provisioning refused: {reason}")
            completed.append(group)
            continue

        affected_mid = None
        v_details = group.get("verification_details", {})
        for mid in group["members"]:
            imp_res = v_details.get("evaluator_imports", {}).get(mid)
            if imp_res and imp_res.get("returncode", 0) != 0:
                affected_mid = mid
                break
        if not affected_mid:
            for name, vinfo in v_details.get("verified_versions", {}).items():
                req = vinfo.get("required", "")
                spec_req = req if any(req.startswith(op) for op in ("==", ">=", "<=", ">", "<", "!=", "~=")) else f"=={req}"
                if vinfo.get("installed") is None or not runtime.version_satisfies(spec_req, vinfo.get("installed")):
                    cand = [m for m in group["members"] if name in specs[m].get("packages", {}) or name in specs[m].get("ranges", {})]
                    if len(cand) == 1:
                        affected_mid = cand[0]
                        break
        if not affected_mid:
            affected_mid = group["members"][-1]

        remaining_members = [m for m in group["members"] if m != affected_mid]
        rem_constraints, _ = _constraints([specs[m] for m in remaining_members])
        rem_group = _build_group_dict(remaining_members, rem_constraints, specs, artifacts_root, solver_cmd)
        rem_group["split_reasons"] = list(group.get("split_reasons", [])) + [f"Split {affected_mid}: {reason}"]
        rem_group["_attempts"] = attempts + 1

        split_constraints, _ = _constraints([specs[affected_mid]])
        split_group = _build_group_dict([affected_mid], split_constraints, specs, artifacts_root, solver_cmd)
        split_group["split_reasons"] = list(group.get("split_reasons", [])) + [f"Split from {remaining_members}: {reason}"]
        split_group["_attempts"] = attempts + 1

        queue.append(rem_group)
        queue.append(split_group)

    return completed


def write_environment_assignment(assignment: dict, path: Path | str) -> Path:
    p = Path(path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    temp_path = p.with_suffix(f".tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}")
    temp_path.write_text(json.dumps(assignment, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp_path.replace(p)
    return p


def load_environment_assignment(path: Path | str) -> dict:
    p = Path(path).resolve()
    with p.open(encoding="utf-8") as handle:
        return json.load(handle)


def provision_environment_plan(
    models: list[dict],
    plan: dict,
    *,
    artifacts_root: Path | str | None = None,
    allowed: bool = False,
    environ: dict[str, str] | None = None,
    third_party_root: Path | str | None = None,
    network_allowed: bool,
) -> dict:
    """Authoritatively provision environments for a planned model set."""
    if artifacts_root is None:
        from workbench.backend.operator_config import resolve_config
        artifacts_root = resolve_config().WORKBENCH_ARTIFACTS_ROOT
    artifacts_root = Path(artifacts_root).resolve()
    resolved_third_party = resolve_third_party_root(third_party_root)

    by_id = {_model_id(m): m for m in models}
    specs = {mid: model_constraints(m, third_party_root=resolved_third_party) for mid, m in by_id.items()}
    solver_cmd = _solver_command()

    groups = plan.get("groups", [])
    provisioned_groups = _provision_all_groups(
        groups, by_id, specs, artifacts_root, resolved_third_party, allowed, solver_cmd,
        network_allowed=network_allowed,
    )
    plan["groups"] = provisioned_groups
    # Only a group that is actually RUNTIME_VERIFIED publishes an interpreter. A failed group's
    # prefix has been removed, so publishing its path handed every later stage a nonexistent
    # interpreter and the autopilot propagated it to child processes as WORKBENCH_<MODEL>_PYTHON.
    plan["interpreter_mapping"] = {
        mid: (group.get("interpreter") if group.get("state") == "RUNTIME_VERIFIED" else None)
        for group in provisioned_groups for mid in group["members"]
    }

    models_assignment = {}
    for g in provisioned_groups:
        for mid in g["members"]:
            # A group that is not RUNTIME_VERIFIED has no usable environment: whatever path the
            # planner recorded is either absent (cleaned up after a failure) or unverified.
            interp = g.get("interpreter") if g.get("state") == "RUNTIME_VERIFIED" else None
            models_assignment[mid] = {
                "environment_id": g["environment_id"],
                "interpreter": interp,
                "conda_env_path": g.get("conda_env_path"),
                "state": g.get("state", "UNKNOWN"),
                "installation_status": g.get("installation_status"),
                "runtime_verification_status": g.get("runtime_verification_status"),
                "shared": len(g["members"]) > 1,
                "group_members": list(g["members"]),
            }
    for b in plan.get("blocked", []):
        mid = b["model_id"]
        models_assignment[mid] = {
            "environment_id": None,
            "interpreter": None,
            "conda_env_path": None,
            "state": "BLOCKED",
            "installation_status": "BLOCKED",
            "runtime_verification_status": "BLOCKED",
            "shared": False,
            "group_members": [],
            "reason": b.get("reason"),
        }

    assignment = {
        "schema_version": 1,
        "artifacts_root": str(artifacts_root),
        "plan_solver": plan.get("solver"),
        "solver_verification_status": plan.get("solver_verification_status"),
        "groups": provisioned_groups,
        "groups_by_id": {g["environment_id"]: g for g in provisioned_groups},
        "models": models_assignment,
        "interpreter_mapping": plan.get("interpreter_mapping", {}),
        "blocked": plan.get("blocked", []),
        "plan": plan,
    }

    has_failures = any(
        g.get("state") == "FAILED" or g.get("installation_status") == "FAILED"
        for g in provisioned_groups
    )
    if not has_failures and (not allowed or all(g.get("state") == "RUNTIME_VERIFIED" for g in provisioned_groups)):
        assignment_path = artifacts_root / "environment_assignment.json"
        write_environment_assignment(assignment, assignment_path)
        plan["environment_assignment_path"] = str(assignment_path)
    else:
        plan["environment_assignment_path"] = None
    return assignment


def main(argv: list[str] | None = None) -> int:
    import argparse
    from workbench.backend.operator_config import resolve_config
    from workbench.backend.registry import load_registry

    config = resolve_config()
    parser = argparse.ArgumentParser(description="Plan deterministic, source-backed evaluation environments.")
    parser.add_argument("--output", type=Path, default=config.WORKBENCH_ARTIFACTS_ROOT / "environment_plan.json")
    parser.add_argument("--artifacts-root", type=Path, default=config.WORKBENCH_ARTIFACTS_ROOT)
    parser.add_argument("--third-party-root", "--source-root", dest="third_party_root", type=Path, default=None)
    parser.add_argument("--model", dest="models", action="append", default=[], help="Model ID to plan (repeatable)")
    parser.add_argument("--selected-models", dest="selected_models", default=None, help="Comma-separated model IDs")
    parser.add_argument("--allow-env-install", "--provision", dest="allow_install", action="store_true", help="Authorize environment provisioning")
    parser.add_argument("--allow-network", dest="allow_network", action="store_true",
                        help="Authorize the package manager to reach its index; required to install")
    parser.add_argument("--json", action="store_true", help="Print plan as JSON to stdout")
    parser.add_argument("--list", action="store_true", help="List groups and blocked models")
    args = parser.parse_args(argv)

    third_party_root = None
    if args.third_party_root is not None:
        third_party_root = Path(args.third_party_root).resolve()
        if not third_party_root.exists():
            raise FileNotFoundError(f"Configured third_party_root does not exist: {third_party_root}")
    elif os.environ.get("WORKBENCH_THIRD_PARTY_ROOT"):
        third_party_root = Path(os.environ["WORKBENCH_THIRD_PARTY_ROOT"]).resolve()
        if not third_party_root.exists():
            raise FileNotFoundError(f"Configured WORKBENCH_THIRD_PARTY_ROOT does not exist: {third_party_root}")

    registry = load_registry()
    models = registry["models"]

    selected_ids = list(args.models)
    if args.selected_models:
        selected_ids.extend([m.strip() for m in args.selected_models.split(",") if m.strip()])
    if selected_ids:
        selected_set = set(selected_ids)
        models = [m for m in models if m.get("model_id") in selected_set]

    plan = plan_environments(
        models,
        artifacts_root=args.artifacts_root,
        third_party_root=third_party_root,
    )

    if args.allow_install:
        # The written plan must be the provisioned one: `provision_environment_plan` mutates and
        # returns the plan whose groups carry their real state, and writing the pre-provisioning
        # draft would report CONSTRAINTS_COMPATIBLE for a group that is RUNTIME_VERIFIED.
        provisioned = provision_environment_plan(models, plan, artifacts_root=args.artifacts_root,
                                                  network_allowed=bool(getattr(args, "allow_network", False)),
                                                allowed=True, third_party_root=third_party_root)
        if isinstance(provisioned, dict):
            plan = provisioned

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(plan, indent=2, sort_keys=True))
    elif args.list:
        for group in plan.get("groups", []):
            print(f"Group {group['environment_id']}: members={group['members']} status={group.get('runtime_verification_status')} state={group.get('state')}")
        for b in plan.get("blocked", []):
            print(f"Blocked {b['model_id']}: {b.get('reason')}")
    else:
        print(args.output)
    return 0
