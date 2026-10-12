"""Plan deterministic, source-backed evaluation environments.

Models are processed by descending declared-constraint count, then model ID.
Each joins the largest group whose complete combined constraints remain
satisfiable; ties use sorted member IDs. Every union is tested from scratch,
never inferred from pairwise compatibility. Solver checks are dry-run only.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
import sys
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


def _requirements_path(model: dict) -> Path | None:
    source_dir = model.get("source_dir")
    if not source_dir:
        return None
    source = WORKBENCH_ROOT / "third_party" / str(source_dir)
    return next((path for path in (source / "requirements.txt", source / "requirement.txt") if path.is_file()), None)

_IMPORT_DISTRIBUTIONS = {
    "attr": ("attrs",), "bs4": ("beautifulsoup4",), "cv2": ("opencv-python", "opencv-python-headless"),
    "dateutil": ("python-dateutil",), "fitz": ("pymupdf",), "lavis": ("salesforce-lavis",),
    "open_clip": ("open-clip-torch",), "pil": ("pillow",), "skimage": ("scikit-image",),
    "sklearn": ("scikit-learn",), "yaml": ("pyyaml",),
}



def _evaluator_imports(model: dict) -> set[str] | None:
    """Return third-party top-level imports reachable from the pinned evaluator."""
    source_dir = model.get("source_dir")
    if not source_dir:
        return None
    from workbench.backend.adapters.models import ADAPTERS, OfficialScriptAdapter

    adapter = ADAPTERS.get(model.get("model_id"))
    if adapter is None or not issubclass(adapter, OfficialScriptAdapter):
        return None
    source = WORKBENCH_ROOT / "third_party" / str(source_dir)
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


def model_constraints(model: dict) -> dict:
    """Extract evaluator imports and their source-backed version evidence."""
    env = model.get("environment") or {}
    requirements = _requirements_path(model)
    packages: dict[str, str] = {}
    ranges: dict[str, str] = {}
    evidence = [f"registry model metadata: {model.get('model_id') or '<unnamed>'}"]
    manifest = []
    declarations = env.get("packages") or []
    if isinstance(declarations, dict):
        declarations = [f"{name}{value if str(value).startswith(('=', '<', '>', '!', '~')) else '==' + str(value)}"
                        for name, value in declarations.items()]
    if requirements:
        evidence.append(str(requirements.relative_to(WORKBENCH_ROOT)))
        manifest = [parsed for line in requirements.read_text(encoding="utf-8", errors="replace").splitlines()
                    if (parsed := _parse_requirement(line))]
    registry_pins = [parsed for declaration in declarations
                     if (parsed := _parse_requirement(str(declaration)))]
    imports = _evaluator_imports(model)
    if imports is None:
        selected = registry_pins
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
            "packages": packages, "ranges": ranges,
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
                                "packages": {}, "ranges": {}, "evidence": []}
    for spec in specs:
        combined["evidence"].extend(spec["evidence"])
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
    return {
        "python": constraints["python"],
        "packages": {name: terms for name, terms in constraints["packages"].items()
                     if name not in _NON_CONDA_PACKAGES and terms},
        "ranges": {name: terms for name, terms in constraints["ranges"].items()
                   if name not in _NON_CONDA_PACKAGES and terms},
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
            process = subprocess.run(argv, capture_output=True, text=True, check=False,
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



def _source_reason(message: str, specs: list[dict]) -> str:
    sources = sorted({evidence for spec in specs for evidence in spec["evidence"]})
    return f"{message}; source: {', '.join(sources) if sources else 'model source metadata unavailable'}"


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
    env["pytorch"], env["torchvision"], env["python"] = spec["torch"], spec["torchvision"], spec["python"]
    result["environment"] = env
    return result


def _reusable_interpreter(members: list[str], by_id: dict[str, dict], specs: dict[str, dict]) -> str | None:
    paths = []
    for mid in members:
        variable = (by_id[mid].get("environment") or {}).get("python_env_var")
        path = os.environ.get(variable or "")
        if not path:
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
    env["pytorch"], env["torchvision"], env["python"] = spec["torch"], spec["torchvision"], spec["python"]
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
    return installation, verification, paths[0] if same else None


def plan_environments(models: list[dict], *, solver=None,
                      artifacts_root: Path | None = None) -> dict:
    """Build largest safe groups; solver-check each proposed full set when available."""
    command = _solver_command() if solver is None else None
    solver_name = "injected" if solver is not None else command or "unavailable"
    by_id = {_model_id(model): model for model in models}
    solver_status = "available" if solver is not None or command else "unavailable"
    specs = {mid: model_constraints(model) for mid, model in by_id.items()}
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
        lock = {"members": members, "constraints": {key: constraints[key] for
                key in ("python", "torch", "torchvision", "cuda", "packages", "ranges")}}
        digest = hashlib.sha256(json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        environment_id = f"cir-{digest[:16]}"
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
                 "lock_digest": digest,
                 "evidence": constraints["evidence"] + [
                     f"Configured conda channels: {', '.join(CONDA_CHANNELS)}"] + (
                     [f"Conda channels searched: {', '.join(CONDA_CHANNELS)}"]
                     if draft["channels_used"] else []),
                 "split_reasons": sorted(set(draft["split_reasons"])),
                 "installation_status": "NOT_INSTALLED", "runtime_verification_status": "NOT_CONFIGURED",
                 "solver_verification_status": draft["solver_verification_status"],
                 "solver_outcomes": sorted(set(draft["solver_outcomes"])),
                 "unresolved": sorted(set(unresolved))}
        installation, verification, reused = _group_status(group, by_id, specs, solver_status)
        group["installation_status"], group["runtime_verification_status"] = installation, verification
        if reused:
            group["interpreter"] = reused
            group["conda_env_path"] = str(Path(reused).parent.parent)
            group["runtime_evidence"] = f"verified configured interpreter: {reused}"
        elif draft["reuse"]:
            group["runtime_evidence"] = f"verified configured interpreter: {draft['reuse']}"
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
            variable = (model.get("environment") or {}).get("python_env_var")
            mapping[mid] = environ.get(variable) if variable and environ.get(variable) else group["interpreter"]
    return mapping


def main(argv: list[str] | None = None) -> int:
    import argparse
    from workbench.backend.registry import load_registry
    parser = argparse.ArgumentParser(description="Plan compatible model environments.")
    from workbench.backend.operator_config import resolve_config

    config = resolve_config()
    parser.add_argument("--output", type=Path, default=config.WORKBENCH_ARTIFACTS_ROOT / "environment_plan.json")
    args = parser.parse_args(argv)
    plan = plan_environments(load_registry()["models"], artifacts_root=config.WORKBENCH_ARTIFACTS_ROOT)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)
    return 0
