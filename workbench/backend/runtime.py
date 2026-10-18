"""Environment verification with explicit readiness tiers and source-backed probing."""
from __future__ import annotations

from packaging.requirements import InvalidRequirement, Requirement
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from workbench.backend.vendored_source import (
    PROVENANCE_FILENAME,
    source_clean_and_pinned as vendor_source_clean_and_pinned,
    source_provenance as vendor_provenance,
)

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
# A CUDA-declaring model whose interpreter, version, and dependencies are verified but
# whose GPU capability has not been probed yet. It is a *legitimate pending state*, not a
# dead end: verify_gpu_runtime() moves it to GPU_RUNTIME_READY on a real GPU host.
TIER_GPU_PENDING = "GPU_VERIFICATION_PENDING"
TIER_GPU_READY = "GPU_RUNTIME_READY"
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
    gpu_capability: dict[str, Any] | None = None
    reason: str | None = None
    probe_executed: bool = False
    dependency_probe_executed: bool = False
    contract_digest: str | None = None
    environment_fingerprint: str | None = None
    verified_at: str | None = None
    #: Provenance of the verified contract: None for the historical pins, or the registry's
    #: `adapted_runtime.target_compute_capability` when an operator-selected adaptation applied.
    runtime_adaptation: str | None = None
    advisory_fingerprint: str | None = None
    content_verified: bool = False

_PROBE_ENV_DENY_PREFIXES = (
    "WORKBENCH_LEDGER_KEY", "WORKBENCH_SECRET_",
    "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN",
    "WANDB_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
    "GITHUB_TOKEN", "GH_TOKEN", "GITLAB_TOKEN",
    "AWS_", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
    "AZURE_", "GOOGLE_", "GOOGLE_APPLICATION_CREDENTIALS", "GCP_",
    "DOCKER_", "KUBECONFIG", "SLACK_", "SLACK_TOKEN", "STRIPE_", "NPM_", "NPM_TOKEN",
)


def _probe_child_env() -> dict[str, str]:
    """Filtered child environment for probes running untrusted model code."""
    return {
        key: value for key, value in os.environ.items()
        if not any(key == prefix or key.startswith(prefix)
                   for prefix in _PROBE_ENV_DENY_PREFIXES)
    }


def _run_probe(interpreter: Path, code: str, timeout: int = 60,
               args: tuple[str, ...] = (), isolated: bool = False) -> tuple[bool, str, str]:
    """Run ``code`` under ``interpreter``.

    ``isolated`` runs ``-I -S``: the user site and ``PYTHONPATH`` are ignored, and ``site``
    does not import ``sitecustomize``/``usercustomize``. The environment's directories are
    deliberately *not* added to ``sys.path``, because ``site.addsitedir`` executes the
    environment's ``.pth`` files. Nothing belonging to the environment runs, so it cannot
    answer for itself. This is used only for identity, which is not a package question.
    """
    prefix_flags = ["-I", "-S"] if isolated else []
    # The probe executes code from the model's environment, which is untrusted. It must not
    # be handed the operator's secrets. Deny sensitive credential prefixes.
    child_env = _probe_child_env()
    try:
        proc = subprocess.run(
            [str(interpreter), *prefix_flags, "-c", code, *args],
            capture_output=True, text=True, timeout=timeout, env=child_env,
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

def check_environment_isolation(interpreter: Path,
                                env: dict[str, Any] | None = None) -> tuple[bool, str]:
    """Verify the *runtime* identity of an interpreter, not just its resolved path.

    A Linux venv's ``bin/python`` is routinely a symlink to a system interpreter
    binary, so a matching ``resolve()`` is **not** evidence that this is the system
    environment. Identity is decided by the interpreter's own
    ``sys.prefix``/``sys.base_prefix``, its reported ``sys.executable``, and the
    on-disk virtualenv/conda markers of the *invocation* path.
    """
    env = env or {}
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

    # When the registry pins the interpreter binary, the resolved file must match. This
    # is the sound way to bind a verified interpreter: an orchestrator cannot tell a
    # genuine CPython from a launcher that speaks the same protocol, because the launcher
    # *is* what it executes. A declared digest resolves that by policy.
    declared_digest = str((env.get("interpreter_sha256") or "")).strip().lower()
    if declared_digest and not declared_digest.upper() == "UNKNOWN":
        actual_digest = _interpreter_digest(invoked)
        if actual_digest is None:
            return False, "interpreter binary could not be hashed"
        if actual_digest != declared_digest:
            return False, (f"interpreter digest mismatch: registry declares {declared_digest[:16]}…, "
                           f"resolved binary is {actual_digest[:16]}…")

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

def _interpreter_digest(interpreter: Path) -> str | None:
    """SHA-256 of the interpreter binary an operator pinned in the registry."""
    import hashlib

    try:
        resolved = Path(interpreter).resolve()
        with resolved.open("rb") as handle:
            digest = hashlib.sha256()
            while True:
                chunk = handle.read(1 << 20)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


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


def _site_packages_dirs(prefix: Path) -> list[Path]:
    """Every installed-distribution directory belonging to an environment prefix.

    Debian/Ubuntu lay distributions out in ``dist-packages`` rather than
    ``site-packages`` (``/usr/lib/python3/dist-packages``), and a
    ``system_site_packages`` virtualenv resolves requirements from exactly there, so both
    spellings are read.
    """
    candidates: list[Path] = []
    for pattern in ("lib/python*/site-packages", "lib/python*/dist-packages"):
        candidates += sorted(prefix.glob(pattern))
    for name in ("lib", "Lib"):
        for leaf in ("site-packages", "dist-packages"):
            candidates.append(prefix / name / leaf)
    result: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen or not candidate.is_dir():
            continue
        seen.add(key)
        result.append(candidate)
    return result


_NORMALIZE_RE = None

def normalize_distribution_name(name: str) -> str:
    """PEP 503 canonical name: lower-cased, runs of ``-_.`` collapsed to ``-``.

    ``importlib.metadata`` accepts any spelling, and an installed distribution's
    ``METADATA`` name need not match its directory name, so both sides of the
    probe/disk reconciliation must be normalised or a correct environment is
    wrongly reported as forged.
    """
    import re

    global _NORMALIZE_RE
    if _NORMALIZE_RE is None:
        _NORMALIZE_RE = re.compile(r"[-_.]+")
    return _NORMALIZE_RE.sub("-", name).strip().lower()


def _metadata_name_version(metadata: Path) -> tuple[str | None, str | None]:
    """The ``Name`` and ``Version`` recorded in a distribution's own metadata file."""
    try:
        text = metadata.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None, None
    name = version = None
    for line in text.splitlines():
        if line.lower().startswith("name:") and name is None:
            name = line.split(":", 1)[1].strip()
        elif line.lower().startswith("version:") and version is None:
            version = line.split(":", 1)[1].strip()
        if name and version:
            break
    return name, version


_DESCRIPTIVE_METADATA = {"METADATA", "PKG-INFO", "RECORD", "INSTALLER", "WHEEL", "LICENSE",
                         "LICENSE.txt", "entry_points.txt", "SOURCES.txt", "requires.txt",
                         "dependency_links.txt", "top_level.txt", "not-zip-safe", "zip-safe"}


def _directory_is_importable(directory: Path) -> bool:
    """True when a directory would actually be importable as a package.

    A namespace package (PEP 420) is importable because it exists, but an *empty*
    directory left behind by an interrupted or wiped install is not usable code, and a
    directory holding only metadata is not a module either.
    """
    try:
        if not directory.is_dir():
            return False
    except OSError:
        return False
    try:
        entries = list(directory.iterdir())
    except OSError:
        return False
    if not entries:
        # PEP 420: an empty directory *is* a namespace package and can be imported.
        return True
    for entry in entries:
        name = entry.name
        if name in ("__init__.py", "__main__.py"):
            return True
        if name.endswith((".py", ".so", ".pyd", ".pyc")):
            return True
        if entry.is_dir() and name not in ("__pycache__",):
            return True
    return False


def _distribution_owner_names(declared_name: str, dist_name: str) -> list[str]:
    """Candidate module names a distribution may install beside its metadata.

    Distribution names use ``-``/``_``/``.`` where module names use ``_``, so the
    candidates are the normalised name plus its underscore spelling, then the same for
    the metadata directory's own name.
    """
    candidates: list[str] = []
    for raw in (str(declared_name), dist_name.rsplit(".", 2)[0].rsplit("-", 1)[0]):
        if not raw:
            continue
        for spelling in (raw, raw.replace("-", "_"), raw.replace(".", "_"),
                         raw.replace("-", "_").replace(".", "_")):
            if spelling and spelling not in candidates:
                candidates.append(spelling)
    return candidates


def _declared_top_level_modules(dist: Path) -> list[str]:
    """Module names a legacy egg-info declares, or the directories it ships."""
    names: list[str] = []
    top_level = dist / "top_level.txt"
    if top_level.is_file():
        try:
            names += [line.strip() for line in
                      top_level.read_text(encoding="utf-8", errors="replace").splitlines()
                      if line.strip()]
        except OSError:
            pass
    return names


def _holds_importable_content(directory: Path, depth: int = 4) -> bool:
    """True when a directory holds at least one importable Python file.

    Bounded depth keeps a large tree cheap; a namespace package legitimately exposes
    subdirectories, but an empty one installs nothing.
    """
    if depth <= 0:
        return False
    try:
        entries = list(directory.iterdir())
    except OSError:
        return False
    for entry in entries:
        if entry.is_file() and entry.name.endswith((".py", ".pyc", ".so")):
            return True
        if entry.is_dir() and (entry / "__init__.py").is_file():
            return True
        if entry.is_dir() and _holds_importable_content(entry, depth - 1):
            return True
    return False


def _safe_record_member(site_packages: Path, member: str, dist_name: str) -> Path | None:
    """Resolve a RECORD entry to a file inside ``site_packages``, or None.

    RECORD entries are untrusted text: they may be absolute, may traverse upwards, or may
    name the distribution's own metadata. Only a real file inside the tree counts as
    installed content.
    """
    member = member.strip()
    if not member or member.startswith(("/", "\\")):
        return None
    # Normalise "./x" and repeated separators before deciding; RECORD entries are
    # untrusted text and a "." segment must not hide the distribution's own metadata.
    parts = [part for part in PurePosixPath(member).parts if part not in (".", "")]
    if not parts or ".." in parts:
        return None
    relative = PurePosixPath(*parts)
    if relative.is_absolute() or relative.parts[0] == dist_name or member.startswith(f"{dist_name}/"):
        return None
    if relative.parts[0] == dist_name or str(relative).startswith(f"{dist_name}/"):
        return None
    return site_packages / relative


def inspect_environment_distributions(prefix: str | None,
                                      base_prefix: str | None = None
                                      ) -> tuple[dict[str, str | None], set[str], set[str]]:
    """Read an environment's installed distributions from disk.

    One walk produces everything the callers need, so the version inventory and the
    "does this distribution actually install anything / which top-level modules exist"
    answers can never disagree:

    * ``versions``       — normalized name -> version (or None when unreadable).
    * ``modules``        — top-level importable names the environment installs.
    * ``unconfirmed``    — names whose installation cannot be confirmed from disk: a
      RECORD-less metadata directory whose distribution name differs from its module name
      (``beautifulsoup4`` installs ``bs4``) is not decidable here, so it is reported rather
      than guessed at or silently accepted.

    Nothing is executed: no ``.pth`` is run, no module is imported, and no directory is
    added to ``sys.path``. ``base_prefix`` covers a ``system_site_packages`` environment,
    which resolves requirements from its base interpreter.
    """
    from importlib.metadata import PackageNotFoundError, distributions

    versions: dict[str, str | None] = {}
    modules: set[str] = set()
    unconfirmed: set[str] = set()
    # Names confirmed to install real content from their environment prefix. A
    # `--system-site-packages` environment may also see an unconfirmable distribution of
    # the same name in its base: the environment's own confirmed install wins.
    confirmed: set[str] = set()
    primary_roots: list[Path] = []
    base_roots: list[Path] = []
    if prefix:
        primary_roots = _site_packages_dirs(Path(prefix))
    if base_prefix and str(base_prefix) != str(prefix):
        base_roots = [root for root in _site_packages_dirs(Path(base_prefix))
                      if root not in primary_roots]
    roots: list[Path] = primary_roots + base_roots

    for site_packages in roots:
        # Distribution metadata, without importing anything.
        try:
            found = list(distributions(path=[str(site_packages)]))
        except (OSError, ValueError):
            found = []
        for item in found:
            try:
                metadata = item.metadata
                if metadata is None:
                    continue
                name = metadata["Name"]
                version = metadata["Version"]
            except (PackageNotFoundError, KeyError):
                continue
            if name:
                versions.setdefault(normalize_distribution_name(str(name)), version or None)

        # Directories in the tree are importable without any metadata at all.
        try:
            entries = list(site_packages.iterdir())
        except OSError:
            entries = []
        for entry in entries:
            entry_name = entry.name
            if entry.is_dir():
                if entry_name.endswith((".dist-info", ".egg-info", ".libs", "__pycache__")):
                    continue
                # A namespace package is importable because it exists; a directory with
                # nothing importable inside it is not a module, and must not be evidence
                # that a distribution installed anything.
                if _directory_is_importable(entry):
                    modules.add(entry_name.split(".")[0])
            elif entry_name.endswith(".py"):
                modules.add(entry_name[: -len(".py")])

        for dist in list(site_packages.glob("*.dist-info")) + list(site_packages.glob("*.egg-info")):
            dist_name = dist.name
            metadata_path = dist / ("METADATA" if dist_name.endswith(".dist-info") else "PKG-INFO")
            declared_name, declared_version = _metadata_name_version(metadata_path)
            if not declared_name:
                declared_name = dist_name.split("-")[0]
            key = normalize_distribution_name(str(declared_name))

            # Installed *content* is what could actually be imported. Metadata that merely
            # describes a distribution is not content, and a distribution whose files are
            # gone describes an install that never completed (or one that was wiped).
            installed_content = False
            record = dist / "RECORD"
            if record.is_file():
                try:
                    lines = record.read_text(encoding="utf-8", errors="replace").splitlines()
                except OSError:
                    lines = []
                for line in lines:
                    member = line.split(",", 1)[0]
                    resolved = _safe_record_member(site_packages, member, dist_name)
                    if resolved is None:
                        continue
                    if resolved.is_dir():
                        # A directory entry installs content only when it actually holds
                        # importable files (a namespace package still counts, but an empty
                        # directory does not).
                        if any((resolved / name).is_file() for name in ("__init__.py", "__init__.pyc")):
                            installed_content = True
                            modules.add(resolved.name.split(".")[0])
                        elif _holds_importable_content(resolved):
                            installed_content = True
                            modules.add(resolved.name.split(".")[0])
                    elif resolved.is_file():
                        installed_content = True
                        if resolved.name.endswith(".py"):
                            modules.add(resolved.name[: -len(".py")])
            else:
                # No RECORD: a legacy egg-info names its modules in ``top_level.txt``, and
                # those modules must really exist. Descriptive files do not count.
                try:
                    siblings = list(dist.iterdir())
                except OSError:
                    siblings = []
                for module_name in _declared_top_level_modules(dist):
                    head = module_name.split(".", 1)[0]
                    if (site_packages / f"{head}.py").is_file() or (site_packages / head).is_dir():
                        installed_content = True
                        modules.add(head)
                if any(sibling.is_file() and sibling.name not in _DESCRIPTIVE_METADATA
                       for sibling in siblings):
                    installed_content = True
                for sibling in siblings:
                    if sibling.is_dir():
                        installed_content = True
                        modules.add(sibling.name.split(".")[0])
                # Debian's ``deb``-installed dist-infos carry no RECORD and no
                # ``top_level.txt``: the installed code simply sits beside the metadata as a
                # module or package directory named after the distribution. Recognise that
                # layout too, or a distro environment looks empty.
                for candidate in _distribution_owner_names(declared_name, dist_name):
                    module_file = site_packages / f"{candidate}.py"
                    package_dir = site_packages / candidate
                    if module_file.is_file() or _directory_is_importable(package_dir):
                        installed_content = True
                        modules.add(candidate)
                        break

            if installed_content:
                versions.setdefault(key, declared_version or None)
                confirmed.add(key)
            elif key in confirmed:
                # Already confirmed by this environment's own prefix.
                pass
            elif not record.is_file() and (dist / "INSTALLER").is_file():
                # A package manager installed this (Debian's ``deb`` drops INSTALLER and no
                # RECORD) and the module it provides is named differently from the
                # distribution. That mapping is not decidable from disk, so the entry is
                # reported as unconfirmed rather than dropped or accepted.
                versions.setdefault(key, declared_version or None)
                unconfirmed.add(key)
            else:
                versions.pop(key, None)
    unconfirmed -= confirmed
    return versions, modules, unconfirmed


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
        # A package entry may carry an inline requirement spec (`open_clip_torch>=2.24,<2.27`);
        # split it so the map is keyed by the canonical distribution name and the version
        # constraint survives as the expected value.
        try:
            req = Requirement(pkg)
        except InvalidRequirement:
            # An unusable constraint is reported as unverified, not missing: fall back to the
            # bare-name split so the key matches the installed distribution name.
            name, _, version = pkg.partition("==")
            required.setdefault(name, declared_requirement(name, version or None))
            continue
        required.setdefault(req.name, declared_requirement(req.name, str(req.specifier) or None))
    return required

#: Operator-selected runtime adaptation. Set to the registry's `adapted_runtime`
#: `target_compute_capability` (e.g. ``sm_120_blackwell``) to verify a model against its declared
#: adapted package constraints instead of the historical pins. The value must match a declared
#: adaptation exactly — this never relaxes a requirement the registry did not write down.
ADAPTATION_ENV = "WORKBENCH_RUNTIME_ADAPTATION"

def runtime_adaptation(model: dict[str, Any], *, undeclared_is_error: bool = True) -> dict[str, Any] | None:
    """The declared adapted runtime for the operator-selected adaptation target, or None.

    Historical pins remain the contract unless the operator explicitly selects an adaptation the
    registry declares. Any other ADAPTATION_ENV value is refused rather than silently ignored —
    except for models that declare no adaptation at all, which keep their historical contract
    (pass ``undeclared_is_error=False``): a process-global adaptation selection must not break
    unrelated models, but a model that DECLARES an adaptation must never silently verify
    against a different, unknown target.
    """
    requested = os.environ.get(ADAPTATION_ENV, "").strip()
    if not requested:
        return None
    from workbench.backend.errors import WorkbenchError

    declared = (model.get("environment") or {}).get("adapted_runtime") or {}
    target = str(declared.get("target_compute_capability") or "").strip()
    if requested != target:
        if not target and not undeclared_is_error:
            return None
        raise WorkbenchError(
            "UNKNOWN_RUNTIME_ADAPTATION",
            f"unknown {ADAPTATION_ENV}={requested!r} for model "
            f"{model.get('model_id', '<unnamed>')!r}: declared adaptation is {target or 'none'}",
            {"model_id": model.get("model_id"), "requested": requested, "declared": target or None})
    return declared

def resolved_runtime_environment(model: dict[str, Any]) -> dict[str, Any]:
    """The environment block with the operator-selected adaptation applied, or the historical one.

    This is THE canonical resolved runtime contract: the environment planner, solver, installer,
    assignment and provenance all consume it, so the installer can never plan a different
    dependency set from the one the verifier validates. The adaptation applies only when the
    registry declares a matching `adapted_runtime.target_compute_capability`; any other model is
    returned exactly as declared (historical pins stay the default contract).

    A selection the model never declared is NOT an error here: a single operator-wide
    `WORKBENCH_RUNTIME_ADAPTATION` applies to whichever models declare that profile, and every
    other model keeps its historical contract. Refusing instead made the planner raise for every
    non-adapted model in a run that mentions the variable at all. A model that DOES declare an
    adapted_runtime the selection fails to match is still an operator error and refuses loudly.
    """
    env = dict(model.get("environment") or {})
    adapted = runtime_adaptation(model, undeclared_is_error=False)
    if not adapted:
        return env
    resolved = {**env}
    for source, target in (("pytorch", "pytorch"), ("torchvision", "torchvision"),
                           ("cuda", "cuda"), ("python", "python")):
        if source in adapted:
            resolved[target] = adapted[source]
    declared = list(resolved.get("packages") or [])
    adapted_names = set()
    for pkg in adapted.get("packages") or []:
        declaration = str(pkg)
        name = re.split(r"[<>=!~]", declaration, maxsplit=1)[0].strip()
        adapted_names.add(_package_key(name))
        declared = [d for d in declared
                    if _package_key(str(d).split("==")[0].split(">=")[0].strip()) != _package_key(name)]
        declared.append(declaration)
    resolved["packages"] = declared
    # The wheel index that ships the CUDA build matching the adapted CUDA version (e.g. the
    # cu128 index for sm_120): without it `pip install torch>=2.7` resolves the default PyPI
    # build, whose arch list lacks sm_120, and the mismatch only surfaces at the first kernel.
    for key in ("pip_index_url",):
        if adapted.get(key):
            resolved[key] = adapted[key]
    return resolved


def _package_key(name: str) -> str:
    return re.sub(r"[-_]+", "-", name.strip().lower())


def _adapted_packages(model: dict[str, Any], adapted: dict[str, Any]) -> dict[str, str | None]:
    """Same shape as _required_packages, from the adapted block's range constraints."""
    required: dict[str, str | None] = {}
    if "pytorch" in adapted:
        required["torch"] = declared_requirement("torch", adapted.get("pytorch"))
    if "torchvision" in adapted:
        required["torchvision"] = declared_requirement("torchvision", adapted.get("torchvision"))
    for pkg in adapted.get("packages") or []:
        if not isinstance(pkg, str):
            continue
        # Split the distribution name off at the first constraint operator character.
        name = re.split(r"[<>=!~]", pkg, maxsplit=1)[0].strip()
        constraint = pkg[len(name):].strip() or None
        required.setdefault(name, constraint)
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


# Statuses that mean the environment itself is unusable, so a GPU probe must not run.
_BLOCKED_STATUSES = {
    "PROBE_FAILED", "UNSAFE_INTERPRETER", "PYTHON_VERSION_MISMATCH", "DEPENDENCIES_UNREADABLE",
    "DEPENDENCY_MISSING", "DEPENDENCY_VERSION_MISMATCH", "VERSION_CONSTRAINT_UNVERIFIED",
    "MODEL_IMPORT_FAILED", "REQUIREMENTS_UNVERIFIED",
}


def _requires_gpu(model: dict[str, Any]) -> bool:
    """True when a model's contract declares a CUDA requirement.

    ``cuda: UNKNOWN`` means the upstream source did not state one, so it is not a
    requirement. Any other value is a real requirement that must be verified on a GPU host.
    """
    env = model.get("environment") or {}
    declared = env.get("cuda")
    if declared is None:
        return False
    return str(declared).upper() != "UNKNOWN"


def gpu_capability_directory(config: Any) -> Path:
    artifacts_root = getattr(config, "WORKBENCH_ARTIFACTS_ROOT", None)
    if artifacts_root is None:
        artifacts_root = Path(getattr(config, "CIR_REPO_ROOT", Path(__file__).resolve().parents[2])) / "workbench" / "artifacts"
    return Path(artifacts_root) / "gpu_capabilities"


def gpu_capability_path(config: Any, capability_key: str | None = None) -> Path:
    """Where one GPU capability record lives.

    Capabilities are keyed by model + interpreter + dependency fingerprint, so more than one
    model can be verified on the same host. A single fixed ``gpu_capability.json`` meant
    verifying model B overwrote model A's record, and a fresh process could no longer find A's
    verification at all. The directory keeps them side by side; the key is a digest, so it is
    safe as a filename.
    """
    directory = gpu_capability_directory(config)
    if capability_key is None:
        return directory / "gpu_capability.json"
    return directory / f"{capability_key}.json"


_GPU_CAPABILITY: dict[str, dict[str, Any]] = {}


def forget_gpu_capability() -> None:
    _GPU_CAPABILITY.clear()


def record_gpu_capability(model: dict[str, Any], result: dict[str, Any],
                          config: Any | None = None) -> dict[str, Any]:
    """Persist a verified GPU capability result, keyed to the environment and host it describes.

    The key includes the interpreter fingerprint, the dependency fingerprint and the host
    identity (machine, driver, visible devices), so a changed interpreter, a changed dependency
    set, or a different host invalidates it rather than silently reusing a stale verification.
    """
    key = _gpu_capability_key(model)
    payload = {
        **result,
        "model_id": model.get("model_id"),
        "capability_key": key,
        # What this record covers, stated on the record:
        "host_identity": result.get("host_identity") or _host_identity(),
        "revalidation": result.get(
            "revalidation",
            "re-probe when the interpreter, dependency set, host, driver or device visibility changes"),
        "expires_on": result.get("expires_on") or [
            "interpreter change", "dependency change", "host change",
            "driver change", "CUDA_VISIBLE_DEVICES change"],
    }
    signature = _gpu_capability_signature(payload, config)
    if signature:
        payload["signature"] = signature
    _GPU_CAPABILITY[key] = payload
    if config is not None:
        try:
            path = gpu_capability_path(config, key)
            path.parent.mkdir(parents=True, exist_ok=True)
            # Atomic replace: a reader either sees the previous record or the new one, never a
            # half-written file that would read as a forged record.
            temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            os.replace(temporary, path)
        except OSError:
            pass
    return payload


def _host_identity() -> dict[str, str]:
    """Identity of the host a capability was recorded on.

    A capability says "this environment could use CUDA on this machine at this time". Moving the
    same environment to a different host, or changing the driver, changes what that claim is worth,
    so the host and driver identity are part of the key and a record does not silently transfer.
    """
    identity: dict[str, str] = {}
    try:
        identity["node"] = os.uname().nodename
        identity["machine"] = os.uname().machine
    except OSError:
        pass
    # The visible devices are part of the claim: a capability recorded for device 0 must not
    # authorize a run that is now restricted to no device.
    #
    # `NVIDIA_VISIBLE_DEVICES` is read too, and it is the variable the container runtime actually
    # acts on: a record written with it unset (every device) stayed `AUTHORIZED` when it was set to
    # a specific device or to none, because only the CUDA spelling was in the identity -- while the
    # repo's own environment allowlist names all three. A claim about which devices are usable
    # cannot ignore the lever that selects them.
    identity["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    identity["nvidia_visible_devices"] = os.environ.get("NVIDIA_VISIBLE_DEVICES", "<unset>")
    identity["nvidia_driver_capabilities"] = os.environ.get("NVIDIA_DRIVER_CAPABILITIES", "<unset>")
    for path in ("/proc/driver/nvidia/version", "/sys/module/nvidia/version"):
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                line = handle.readline().strip()
        except OSError:
            continue
        if line:
            identity["driver"] = line
            break
    return identity

def resolve_model_interpreter(model: dict[str, Any], config: Any | None = None) -> str | None:
    """The interpreter path this model's environment currently resolves to.

    Checks the declared environment variable first. If unset, falls back to the persisted
    provisioning assignment (`environment_assignment.json`) under this operator's artifacts root,
    provided the record is RUNTIME_VERIFIED and located within this artifacts root.
    """
    env = model.get("environment") or {}
    variable = env.get("python_env_var")
    if not variable:
        return None
    configured = os.environ.get(variable)
    if configured:
        return configured
    try:
        from workbench.backend.environment_planner import load_environment_assignment
        from workbench.backend.operator_config import resolve_config

        cfg = config if config is not None else resolve_config()
        artifacts_root = Path(getattr(cfg, "WORKBENCH_ARTIFACTS_ROOT", None) or "")
        assignment_path = artifacts_root / "environment_assignment.json"
        if assignment_path.is_file():
            model_id = model.get("model_id", "<unnamed>")
            persisted = (load_environment_assignment(assignment_path).get("models") or {}).get(model_id) or {}
            interp = str(persisted.get("interpreter") or "")
            # Only adopt an interpreter this operator's own artifacts root created: another
            # root's assignment (different install/host) must not silently become this
            # environment's interpreter.
            interp_p = Path(interp)
            root_p = artifacts_root.resolve()
            # Only adopt an interpreter this operator's own artifacts root created: another
            # root's assignment (different install/host) must not silently become this
            # environment's interpreter.
            if (persisted.get("state") == "RUNTIME_VERIFIED" and interp
                    and interp_p.is_absolute()
                    and (interp_p.is_relative_to(artifacts_root)
                         or interp_p.parent.parent.resolve().is_relative_to(root_p)
                         or interp_p.resolve().is_relative_to(root_p))):
                return interp
    except Exception:
        pass
    return None

_resolve_model_interpreter = resolve_model_interpreter


def _gpu_capability_key(model: dict[str, Any], authoritative: bool = True) -> str:
    import hashlib

    configured = resolve_model_interpreter(model) or ""
    interpreter = Path(configured).expanduser() if configured else None
    fingerprint = None
    if interpreter is not None and interpreter.is_file():
        fingerprint = _interpreter_fingerprint(
            str(interpreter), _environment_contract_digest(model),
            str(interpreter.parent.parent), authoritative=authoritative)
    return hashlib.sha256(
        json.dumps({
            "model_id": model.get("model_id"),
            "interpreter": configured,
            "interpreter_fingerprint": fingerprint,
            "contract": _environment_contract_digest(model),
            # Host identity: a capability describes a host, not only an environment.
            "host": _host_identity(),
        }, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _gpu_capability_signature(record_without_signature: dict[str, Any],
                              config: Any | None = None) -> str | None:
    """HMAC of a GPU capability record, using the operator key kept outside the artifacts.

    ``config`` is passed through when the caller has one: resolving a fresh global config here would
    read a different repository root than the record was written under, so a capability that exists
    on disk would fail its own signature check and be re-probed.
    """
    try:
        key = _gpu_cache_key(config)
    except Exception:
        return None
    if not key:
        return None
    import hmac as _hmac

    canonical = json.dumps(record_without_signature, sort_keys=True,
                           separators=(",", ":")).encode("utf-8")
    return _hmac.new(key, canonical, hashlib.sha256).hexdigest()


def _gpu_cache_key(config: Any | None = None) -> bytes:
    """The operator key for GPU capability records: the LEDGER's decision, not a second one.

    Two resolvers used to coexist with different preferences -- this one took the repository-root
    key whenever the checkout was writable, while ``evaluate_models.ledger_hmac_key`` preferred the
    durable ``$XDG_CONFIG_HOME/cir`` key -- so on a writable repo each created a DIFFERENT key file
    and the next run failed with "two different ledger keys exist". The capability cache and the
    evidence ledger then disagreed about which host they were describing.

    The ledger is the owner because it has the stronger requirement: a ledger signed with the other
    key is unverifiable, so it refuses to guess. This function adopts the same decision, and treats
    the refusal as "unverified" rather than "trusted" -- a capability record that cannot be signed
    with a known key must not be accepted.

    Failures return b"", which makes every persisted record unverified rather than trusted.
    """
    try:
        from workbench.scripts.evaluate_models import LedgerKeyError, ledger_hmac_key
        from workbench.backend.operator_config import resolve_config

        return ledger_hmac_key(config or resolve_config())
    except LedgerKeyError as error:
        # Re-emit the owner's message verbatim: it names both paths and the remedy, and an operator
        # who saw it from the ledger must see the same words here.
        print(f"workbench: {error}", file=sys.stderr)
        return b""
    except Exception:
        # The import is lazy only to keep the module graph acyclic (evaluate_models imports this
        # module); if it ever fails, an unverified record is the safe answer.
        return b""


def _gpu_capability_signature_valid(payload: dict[str, Any], config: Any | None = None) -> bool:
    claimed = payload.get("signature")
    if not isinstance(claimed, str) or not claimed:
        return False
    body = {key: value for key, value in payload.items() if key != "signature"}
    expected = _gpu_capability_signature(body, config)
    return bool(expected) and hmac_compare(expected, claimed)


def hmac_compare(expected: str, claimed: str) -> bool:
    import hmac as _hmac

    return _hmac.compare_digest(expected, claimed)


#: Test-only switch: allows a synthetic probe (and cached synthetic records) to satisfy the GPU
#: gate. Never set in production; production launch gates accept only interpreter-probed evidence.
GPU_PROBE_OVERRIDE_ENV = "WORKBENCH_ALLOW_SYNTHETIC_GPU_PROBE"


def _record_environment_fingerprint(model: dict[str, Any],
                                          interpreter: str | None) -> str | None:
    """The authoritative environment fingerprint a capability record is bound to.

    Both the writer and every reader compute this value, so it must match the fingerprint the
    capability KEY is derived from (`_gpu_capability_key`): same interpreter string, same
    prefix derivation (the interpreter's parent's parent, unresolved — resolving can land on a
    shared base interpreter for a symlinked venv and would then disagree with the key), same
    contract digest. Computing it here is affordable because it happens once per capability
    probe, never on a status read.
    """
    if not interpreter:
        return None
    from pathlib import Path as _Path
    return _interpreter_fingerprint(
        str(interpreter), _environment_contract_digest(model),
        str(_Path(interpreter).parent.parent), authoritative=True)

def _cached_gpu_capability(model: dict[str, Any], report: EnvironmentReport,
                           authoritative: bool = True) -> dict[str, Any] | None:
    """A previously verified capability for exactly this environment, or None.

    On the advisory/readiness path (`authoritative=False`), this reads existing signed
    records without unbounded content hashing. On the authoritative verification path,
    it checks the full byte-exact fingerprint.
    """
    configured = getattr(report, "interpreter", None) or resolve_model_interpreter(model) or ""
    from workbench.backend.operator_config import resolve_config

    config_for_records = resolve_config()
    synthetic_ok = os.environ.get(GPU_PROBE_OVERRIDE_ENV) == "1"
    expected_fingerprint = _record_environment_fingerprint(model, configured)

    if not authoritative:
        for cached_payload in list(_GPU_CAPABILITY.values()):
            if not isinstance(cached_payload, dict):
                continue
            probe_source = cached_payload.get("probe_source", "interpreter")
            if not (probe_source == "interpreter" or (synthetic_ok and probe_source == "override")):
                continue
            if (cached_payload.get("model_id") in (None, model.get("model_id"))
                    and cached_payload.get("interpreter") == configured
                    and cached_payload.get("dependency_fingerprint") == _environment_contract_digest(model)
                    and cached_payload.get("host_identity") == _host_identity()
                    and (not expected_fingerprint or cached_payload.get("interpreter_fingerprint") == expected_fingerprint)
                    and _gpu_capability_signature_valid(cached_payload, config_for_records)):
                return cached_payload
        directory = gpu_capability_directory(config_for_records)
        if directory.is_dir():
            for candidate in sorted(directory.glob("*.json")):
                try:
                    payload = json.loads(candidate.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                if not isinstance(payload, dict):
                    continue
                probe_source = payload.get("probe_source", "interpreter")
                if not (probe_source == "interpreter" or (synthetic_ok and probe_source == "override")):
                    continue
                if (payload.get("model_id") in (None, model.get("model_id"))
                        and payload.get("interpreter") == configured
                        and payload.get("dependency_fingerprint") == _environment_contract_digest(model)
                        and payload.get("host_identity") == _host_identity()
                        and (not expected_fingerprint or payload.get("interpreter_fingerprint") == expected_fingerprint)
                        and _gpu_capability_signature_valid(payload, config_for_records)):
                    if payload.get("capability_key"):
                        _GPU_CAPABILITY[payload["capability_key"]] = payload
                    return payload
        return None

    key = _gpu_capability_key(model, authoritative=True)
    cached = _GPU_CAPABILITY.get(key)
    if cached is not None:
        probe_source = cached.get("probe_source", "interpreter") if isinstance(cached, dict) else "interpreter"
        if not (probe_source == "interpreter" or (synthetic_ok and probe_source == "override")):
            cached = None
        elif not _gpu_capability_signature_valid(cached, config_for_records):
            _GPU_CAPABILITY.pop(key, None)
            cached = None
    if cached is None:
        payload = None
        try:
            path = gpu_capability_path(config_for_records, key)
            if not path.is_file():
                legacy = gpu_capability_path(config_for_records)
                path = legacy if legacy.is_file() else path
            if path.is_file():
                payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            payload = None
        probe_source = payload.get("probe_source", "interpreter") if isinstance(payload, dict) else "interpreter"
        if (isinstance(payload, dict) and payload.get("capability_key") == key
                and (probe_source == "interpreter" or (synthetic_ok and probe_source == "override"))):
            if _gpu_capability_signature_valid(payload, config_for_records):
                cached = payload
                _GPU_CAPABILITY[key] = payload
    if cached is None:
        return None
    # The record must belong to THIS environment's interpreter. The comparison uses the
    # AUTHORITATIVE fingerprint on both sides — the same value the capability key is derived
    # from and the value `_record_environment_fingerprint` stores when the record is written —
    # so a record produced by a different interpreter (or one whose environment was mutated)
    # cannot certify this one. Comparing against `report.environment_fingerprint` alone silently
    # accepted any record on the advisory path, where that field is deliberately unset.
    if expected_fingerprint:
        if cached.get("interpreter_fingerprint") != expected_fingerprint:
            return None
    return cached

def verify_environment(model: dict[str, Any], *, probe: bool = True, require_cuda: bool = False,
                       authoritative: bool | None = None, advisory: bool = False) -> EnvironmentReport:
    """Return a truthful environment report. Never claims RUNTIME_READY without proof."""
    model_id = model.get("model_id", "<unnamed>")
    env = model.get("environment") or {}
    variable = env.get("python_env_var")
    if not variable:
        return EnvironmentReport(model_id, None, None, TIER_UNCONFIGURED, "UNCONFIGURED",
                                 reason="no environment variable declared")
    configured = resolve_model_interpreter(model)
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

    is_authoritative = False if advisory else (True if authoritative is None else authoritative)
    if probe:
        cached = _reuse(model, require_cuda, str(interpreter), authoritative=is_authoritative)
        if cached is not None:
            return cached

    identity = _interpreter_identity(interpreter) if probe else None
    report = EnvironmentReport(
        model_id, variable, str(interpreter), "INTERPRETER_PRESENT", "PRESENT",
        prefix=(identity or {}).get("prefix"),
        base_prefix=(identity or {}).get("base_prefix"),
        python_version=(identity or {}).get("version"),
        probe_executed=bool(identity),
        content_verified=is_authoritative,
    )
    if probe and identity is None:
        report.tier = TIER_BLOCKED
        report.status = "PROBE_FAILED"
        report.reason = "interpreter failed its identity probe"
        return report

    if probe:
        isolated, iso_reason = check_environment_isolation(interpreter, env)
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

    # Dependency verification. The inventory is read from disk by the orchestrator
    # itself: no code belonging to the environment runs, so the environment cannot
    # report packages, versions, or modules that it does not actually have.
    adaptation = runtime_adaptation(model, undeclared_is_error=False)
    required = _adapted_packages(model, adaptation) if adaptation else _required_packages(model)
    imports = _required_imports(model)
    if required or imports:
        report.packages, installed_modules, unconfirmed = inspect_environment_distributions(
            report.prefix, report.base_prefix)
        import sys as _sys

        installed_modules |= set(_sys.stdlib_module_names)
        installed_modules |= set(_sys.builtin_module_names)
        declared_import_modules = {name.split(".")[0] for name in imports}
        report.dependency_probe_executed = True
        for pkg, want in required.items():
            key = normalize_distribution_name(pkg)
            if key in unconfirmed:
                # The distribution was installed by a package manager and the module it
                # provides cannot be derived from its metadata (`beautifulsoup4` installs
                # `bs4`). If the model declares the module explicitly and that module is
                # really installed, the requirement *is* verified.
                if not any(name.split(".")[0] in declared_import_modules for name in imports):
                    report.tier = TIER_REQUIREMENTS_UNVERIFIED
                    report.status = "DEPENDENCY_INSTALLATION_UNVERIFIED"
                    report.reason = (f"{pkg} was installed by a package manager but the module "
                                     f"it provides cannot be derived from its metadata; declare "
                                     f"that module in required_imports to verify it")
                    _remember(model, require_cuda, report)
                    return report
            if key not in report.packages:
                report.tier = TIER_BLOCKED
                report.status = "DEPENDENCY_MISSING"
                report.reason = (f"missing package: {pkg} (no importable installation found "
                                 f"in the environment)")
                _remember(model, require_cuda, report)
                return report
            got = report.packages[key]
            if want is None:
                # No version declared: any installed version satisfies presence.
                continue
            matched = version_satisfies(want, str(got))
            if matched is None:
                report.tier = TIER_REQUIREMENTS_UNVERIFIED
                report.status = "VERSION_CONSTRAINT_UNVERIFIED"
                report.reason = f"{pkg}: unusable version constraint {want!r} (installed {got})"
                _remember(model, require_cuda, report)
                return report
            if not matched:
                report.tier = TIER_BLOCKED
                report.status = "DEPENDENCY_VERSION_MISMATCH"
                report.reason = f"{pkg}: expected {want}, found {got}"
                _remember(model, require_cuda, report)
                return report
        report.tier = "DEPENDENCIES_VERIFIED"
        report.status = "DEPENDENCIES_OK"
    else:
        imports = []
    report.runtime_adaptation = str(adaptation.get("target_compute_capability")) if adaptation else None

    # Declared importable modules are checked against the same on-disk inventory: a
    # top-level module name must correspond to something the environment installs.
    if imports:
        import sys as _sys

        installed_modules |= set(_sys.stdlib_module_names)
        installed_modules |= set(_sys.builtin_module_names)
        missing = [name for name in imports if name.split(".")[0] not in installed_modules]
        report.imports = {name: name not in missing for name in imports}
        if missing:
            report.tier = TIER_BLOCKED
            report.status = "MODEL_IMPORT_FAILED"
            report.reason = f"import unavailable: {', '.join(missing)}"
            _remember(model, require_cuda, report)
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

    # Read-only inspection never probes CUDA: the answer can only come from running the
    # environment's own code, which inspection refuses to do. A CUDA-declaring model is
    # therefore reported as pending — a legitimate state with a real way forward — and only
    # verify_gpu_runtime() can move it to GPU_RUNTIME_READY.
    if _requires_gpu(model):
        # Interpreter, Python version, and dependencies are verified; GPU capability is a
        # separate question that inspection must never answer. The model is pending, and
        # verify_gpu_runtime() is the authorized path that resolves it on the GPU host.
        cached_gpu = _cached_gpu_capability(model, report, authoritative=is_authoritative)
        if cached_gpu is not None and cached_gpu.get("capability_verified"):
            report.gpu_capability = cached_gpu
            report.cuda_available = bool(cached_gpu.get("cuda_available"))
            report.cuda_version = cached_gpu.get("torch_cuda_version")
            report.tier = TIER_GPU_READY
            report.status = TIER_GPU_READY
            _remember(model, require_cuda, report)
            return report
        report.tier = TIER_GPU_PENDING
        report.status = TIER_GPU_PENDING
        report.reason = ("CUDA capability must be verified on the GPU host with "
                         "verify_gpu_runtime() before this model can be launched")
        _remember(model, require_cuda, report)
        return report

    if not report.dependency_probe_executed and not imports:
        report.tier = TIER_REQUIREMENTS_UNVERIFIED
        report.status = "REQUIREMENTS_UNVERIFIED"
        report.reason = "no dependency probe was executed for this contract"
        _remember(model, require_cuda, report)
        return report

    if not required and not imports:
        # No source-backed requirements: cannot claim full verification.
        report.tier = TIER_REQUIREMENTS_UNVERIFIED
        report.status = "REQUIREMENTS_UNVERIFIED"
        report.reason = "no source-backed dependency contract declared"
        _remember(model, require_cuda, report)
        return report



    report.tier = "RUNTIME_READY"
    report.status = "RUNTIME_READY"
    _remember(model, require_cuda, report)
    return report


# Tiers that constitute a verified *interpreter* runtime (interpreter, Python version,
# dependencies, declared imports). GPU capability is a separate question: a model that
# declares CUDA is interpreter-ready but not launch-ready until its GPU is probed.
READY_TIERS = {"RUNTIME_READY", TIER_GPU_PENDING, TIER_GPU_READY}
# Tiers that permit *launching* a real evaluation. A CUDA-declaring model must reach
# GPU_RUNTIME_READY first; CPU-only models never leave RUNTIME_READY.
LAUNCH_READY_TIERS = {"RUNTIME_READY", TIER_GPU_READY}


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
        "interpreter_sha256": env.get("interpreter_sha256"),
        "environment_required": bool(model.get("environment_required")),
        # The selected adaptation changes which constraints verify the environment, so two
        # contracts (historical vs adapted) must never share a cache entry.
        "adapted_runtime": runtime_adaptation(model, undeclared_is_error=False),
    }
    return hashlib.sha256(json.dumps(relevant, sort_keys=True).encode("utf-8")).hexdigest()


def _member_digest(path: Path) -> str:
    """Content digest of an installed file.

    Hashed every time. The memo that used to live here was keyed on (path, size, mtime), which
    is exactly the triple an in-place rewrite can preserve — os.utime restores mtime — so the
    memo could hand back a digest for content that had changed and a stale GPU capability would
    stay valid. Verification runs during environment checks, not per query: a full hash is
    affordable, and a digest that can be wrong is not evidence.
    """
    from workbench.backend.registry import sha256_file
    from pathlib import Path as _Path

    return sha256_file(_Path(path))


_ADVISORY_INTERPRETER_FINGERPRINT_CACHE: dict[tuple, str] = {}

def _interpreter_fingerprint(interpreter: str, contract_digest: str,
                            prefix: str | None = None,
                            *, authoritative: bool = True) -> str:
    """Identity of the *environment content*, not merely its path.

    On the advisory/status path (`authoritative=False`), this performs a cheap presence
    and metadata check: interpreter binary, environment marker files, site-packages directory
    mtimes, distribution metadata, and RECORD file text (which records package-manager digests).

    On the authoritative verification path (`authoritative=True`), every distribution member
    is hashed byte-by-byte via `_member_digest` without caching.
    """
    import hashlib

    path = Path(interpreter)
    if prefix:
        prefix_path = Path(prefix)
    else:
        prefix_path = path.parent.parent

    site_packages_dirs = sorted(prefix_path.glob("lib/python*/site-packages"))
    site_packages_dirs += [prefix_path / "lib" / "site-packages",
                           prefix_path / "Lib" / "site-packages"]

    cache_key = None
    if not authoritative:
        try:
            st = path.stat()
            sp_mtimes = tuple((str(sp), sp.stat().st_mtime_ns) for sp in site_packages_dirs if sp.is_dir())
            # ctime is in the key because a same-length rewrite with mtime restored moves neither
            # size nor mtime, and ctime is the one inode field userspace cannot set backwards.
            cache_key = (str(path), st.st_size, st.st_mtime_ns, st.st_ctime_ns, str(prefix_path),
                         contract_digest, sp_mtimes)
            cached = _ADVISORY_INTERPRETER_FINGERPRINT_CACHE.get(cache_key)
            if cached is not None:
                return cached
        except OSError:
            cache_key = None

    hasher = hashlib.sha256()
    hasher.update(contract_digest.encode("utf-8"))
    hasher.update(str(path).encode("utf-8"))
    for candidate in (path, path.resolve()):
        try:
            stat_result = candidate.stat()
        except OSError:
            continue
        # The interpreter's CONTENT, not just its metadata. `size:mtime` cannot witness a
        # same-length in-place replacement followed by `os.utime` -- the same attack this file's
        # capability-record helper already rejects by hashing byte-for-byte (`_member_digest`)
        # and the same one the receipt cache had to be rewritten to survive. The binary is the
        # thing the whole environment claim is anchored to, so it is hashed; the cost is one pass
        # over a ~15 MB file on the authoritative path, which already hashes every distribution
        # member byte-by-byte, and the advisory path memoizes its answer.
        digest = _interpreter_digest(candidate)
        hasher.update(
            f"{candidate}:{stat_result.st_size}:{stat_result.st_mtime_ns}:{digest}".encode("utf-8"))
        break
    for marker in ("pyvenv.cfg", "conda-meta/history", "conda-meta"):
        marker_path = prefix_path / marker
        try:
            stat_result = marker_path.stat()
        except OSError:
            continue
        hasher.update(f"{marker}:{stat_result.st_size}:{stat_result.st_mtime_ns}".encode("utf-8"))

    for site_packages in site_packages_dirs:
        if not site_packages.is_dir():
            continue
        try:
            hasher.update(f"{site_packages}:{site_packages.stat().st_mtime_ns}".encode("utf-8"))
        except OSError:
            continue
        for dist in sorted(list(site_packages.glob("*.dist-info")) + list(site_packages.glob("*.egg-info"))):
            metadata = dist / "METADATA" if dist.name.endswith(".dist-info") else dist / "PKG-INFO"
            try:
                stat_result = metadata.stat()
            except OSError:
                continue
            hasher.update(f"{dist.name}:{stat_result.st_size}:{stat_result.st_mtime_ns}".encode("utf-8"))
            record = dist / "RECORD"
            if not record.is_file():
                for member in sorted(dist.iterdir()):
                    try:
                        member_stat = member.stat()
                    except OSError:
                        continue
                    hasher.update(f"{dist.name}/{member.name}:{member_stat.st_size}:"
                                  f"{member_stat.st_mtime_ns}".encode("utf-8"))
                continue
            try:
                record_text = record.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            hasher.update(f"RECORD:{record_text}".encode("utf-8"))
            if authoritative:
                members = [line.split(",", 1)[0] for line in record_text.splitlines() if line.strip()]
                for member in members:
                    member_path = site_packages / member
                    try:
                        hasher.update(f"{member}:{_member_digest(member_path)}".encode("utf-8"))
                        continue
                    except (OSError, ValueError):
                        pass
                    try:
                        stat_result = member_path.stat()
                    except OSError:
                        hasher.update(f"{member}:ABSENT".encode("utf-8"))
                        continue
                    hasher.update(f"{member}:{stat_result.st_size}:{stat_result.st_mtime_ns}".encode("utf-8"))

    digest = hasher.hexdigest()
    if cache_key is not None:
        _ADVISORY_INTERPRETER_FINGERPRINT_CACHE[cache_key] = digest
    return digest






_VERIFIED_ENVIRONMENTS: dict[str, EnvironmentReport] = {}


def _cache_key(model: dict[str, Any], require_cuda: bool) -> str:
    """Everything a cached verification's claim depends on.

    The host identity is part of it because a *GPU capability* is a claim about this machine,
    driver and visible devices, and the cache could otherwise serve a `GPU_RUNTIME_READY` report
    recorded elsewhere -- after a driver upgrade, on another node, or with the device selection
    changed -- to a launch that the capability record itself correctly refuses. The report's own
    `expires_on` names exactly those conditions (``runtime.py`` `record_gpu_capability`), so
    omitting them from the key made the record's honest expiry advisory while the cache stayed
    authoritative.
    """
    env = model.get("environment") or {}
    configured = os.environ.get(env.get("python_env_var") or "", "")
    host = hashlib.sha256(
        json.dumps(_host_identity(), sort_keys=True).encode("utf-8")).hexdigest()
    return (f"{model.get('model_id')}|{env.get('python_env_var')}|{configured}|"
            f"{_environment_contract_digest(model)}|cuda={bool(require_cuda)}|host={host}")


def _cache_enabled() -> bool:
    return os.environ.get("WORKBENCH_NO_ENV_CACHE") not in {"1", "true", "yes"}


def _reuse(model: dict[str, Any], require_cuda: bool, interpreter: str,
           authoritative: bool = True) -> EnvironmentReport | None:
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
    if cached.gpu_capability is not None:
        synthetic_ok = os.environ.get(GPU_PROBE_OVERRIDE_ENV) == "1"
        probe_source = cached.gpu_capability.get("probe_source", "interpreter")
        if not (probe_source == "interpreter" or (synthetic_ok and probe_source == "override")):
            _VERIFIED_ENVIRONMENTS.pop(key, None)
            return None
        # The record's own host identity is compared as well as the key: the key is a hash, and a
        # capability that reports another host may never authorize this one's launch even if the
        # key somehow agreed.
        recorded_host = cached.gpu_capability.get("host_identity")
        if isinstance(recorded_host, dict) and recorded_host != _host_identity():
            _VERIFIED_ENVIRONMENTS.pop(key, None)
            return None
    if authoritative:
        if cached.environment_fingerprint is None:
            return None
        if cached.environment_fingerprint != _interpreter_fingerprint(
            interpreter, _environment_contract_digest(model), cached.prefix,
            authoritative=True,
        ):
            _VERIFIED_ENVIRONMENTS.pop(key, None)
            return None
    else:
        cached_fp = cached.advisory_fingerprint or cached.environment_fingerprint
        current_fp = _interpreter_fingerprint(
            interpreter, _environment_contract_digest(model), cached.prefix,
            authoritative=False,
        )
        if cached_fp != current_fp:
            _VERIFIED_ENVIRONMENTS.pop(key, None)
            return None
    # A pinned-interpreter mismatch must never be masked by a cached verification.
    declared_digest = str((model.get("environment") or {}).get("interpreter_sha256") or "").strip().lower()
    if declared_digest and declared_digest.upper() != "UNKNOWN":
        if _interpreter_digest(interpreter) != declared_digest:
            _VERIFIED_ENVIRONMENTS.pop(key, None)
            return None
    return cached


def _remember(model: dict[str, Any], require_cuda: bool, report: EnvironmentReport) -> None:
    # The contract digest is the identity of what was just verified, not a cache artifact: a
    # GPU capability record stores it and every reader compares it against the model's current
    # contract. Only the fingerprints below are cache state, so only they are gated.
    report.contract_digest = _environment_contract_digest(model)
    if not _cache_enabled() or report.interpreter is None:
        return
    is_authoritative = getattr(report, "content_verified", False)
    if is_authoritative:
        report.environment_fingerprint = _interpreter_fingerprint(
            report.interpreter, report.contract_digest, report.prefix,
            authoritative=True)
        report.advisory_fingerprint = report.environment_fingerprint
    else:
        report.advisory_fingerprint = _interpreter_fingerprint(
            report.interpreter, report.contract_digest, report.prefix,
            authoritative=False)
        report.environment_fingerprint = None
    report.verified_at = datetime.now(timezone.utc).isoformat()
    _VERIFIED_ENVIRONMENTS[_cache_key(model, require_cuda)] = report


def forget_environment(interpreter: str | None = None) -> None:
    """Drop cached verifications (all, or those for one interpreter path)."""
    _ADVISORY_INTERPRETER_FINGERPRINT_CACHE.clear()
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


def require_launch_ready(model: dict[str, Any], *, require_cuda: bool = True) -> str:
    """Return the interpreter for a model that is ready to *launch*, GPU included.

    Building an evaluation *plan* only needs a verified interpreter; starting a real
    evaluation of a CUDA-declaring model additionally needs a verified GPU capability.
    This is the gate the evaluator and the pipeline use before launching.
    """
    interpreter = require_verified_model_interpreter(model, require_cuda=require_cuda)
    if not _requires_gpu(model):
        return interpreter
    report = verify_environment(model, probe=True, require_cuda=require_cuda, authoritative=True)
    if report.tier in LAUNCH_READY_TIERS:
        return interpreter
    raise GpuVerificationError(
        f"GPU runtime not verified [{report.tier}/{report.status}]: {report.reason or ''}".strip())


# The authorized GPU probe. It runs a tiny, bounded computation; it never loads a model,
# never downloads anything, and never allocates substantial device memory.
GPU_PROBE_CODE = r"""
import json, sys

# The probe must import the environment's real dependencies (torch and its runtime), so the
# directories named on the command line are added to sys.path. They are passed explicitly by
# the orchestrator, derived from the environment's invocation path, because `-S` prevents
# sysconfig from applying the environment's own site configuration.
for _candidate in list(sys.argv[1:]):
    if _candidate and _candidate not in sys.path:
        sys.path.insert(0, _candidate)
out = {
    "probe": "gpu_capability",
    "cuda_available": False,
    "device_count": 0,
    "torch_version": None,
    "torch_cuda_version": None,
    "device_name": None,
    "compute_capability": None,
    "torch_arch_list": [],
    "allocated_tensor": False,
    "computed": False,
    "torch_module_file": None,
    "torch_has_compiled_core": False,
    "torch_core_origin": None,
    "torch_file_digest": None,
    "error": None,
}
try:
    import torch
except Exception as error:
    out["error"] = f"torch unavailable: {error}"
    print(json.dumps(out))
    sys.exit(0)
out["torch_version"] = getattr(torch, "__version__", None)
# Corroboration that this is a real installation rather than a single stub module: a real
# PyTorch ships a compiled core extension, and its module lives inside the environment.
try:
    import torch as _torch
    out["torch_module_file"] = getattr(_torch, "__file__", None)
    # "Compiled core" means a real native extension, not merely a file named `_C`: a Python
    # file satisfies find_spec, and a file with a .so suffix can contain anything. The
    # content is therefore inspected for a native binary header, and the extension must live
    # inside this environment.
    try:
        import importlib.util as _ilu

        _spec = _ilu.find_spec("torch._C")
        _origin = getattr(_spec, "origin", None) if _spec is not None else None
        out["torch_core_origin"] = _origin
        if isinstance(_origin, str):
            try:
                with open(_origin, "rb") as _handle:
                    _header = _handle.read(4)
            except OSError:
                _header = b""
            _native = (_header[:4] == b"\x7fELF" or _header[:2] in (b"MZ",)
                       or _header[:4] in (b"\xcf\xfa\xed\xfe", b"\xfe\xed\xfa\xcf",
                                          b"\xca\xfe\xba\xbe"))
            out["torch_has_compiled_core"] = bool(
                _native and _origin.endswith((".so", ".pyd", ".dylib", ".dll")))
        else:
            out["torch_has_compiled_core"] = False
    except Exception:
        out["torch_has_compiled_core"] = False
except Exception:
    pass
out["torch_cuda_version"] = getattr(getattr(torch, "version", None), "cuda", None)
cuda_module = getattr(torch, "cuda", None)
out["cuda_available"] = bool(getattr(cuda_module, "is_available", lambda: False)())
if out["cuda_available"]:
    out["device_count"] = int(getattr(cuda_module, "device_count", lambda: 0)())
if out["cuda_available"]:
    try:
        index = 0
        out["device_name"] = torch.cuda.get_device_name(index)
        major, minor = torch.cuda.get_device_capability(index)
        out["compute_capability"] = f"{major}.{minor}"
        # Which GPU architectures this torch build compiled kernels for: an adaptation claim
        # (e.g. sm_120 on Blackwell) is only credible if the target arch is in this list.
        try:
            out["torch_arch_list"] = list(torch.cuda.get_arch_list())
        except Exception:
            out["torch_arch_list"] = []
        # A one-element tensor is enough to prove the device is usable.
        device = torch.device("cuda", index)
        tensor = torch.ones(1, device=device)
        out["allocated_tensor"] = True
        doubled = (tensor * 2).sum().item()
        out["computed"] = doubled == 2.0
    except Exception as error:
        out["error"] = f"cuda device unusable: {error}"
print(json.dumps(out))
"""

# A test-only probe override. It is consulted ONLY when the named integration-test opt-in
# environment variable is set, so production execution can never select it.
_GPU_PROBE_OVERRIDE: Any = None

def set_gpu_probe_for_tests(probe: Any) -> None:
    """Install a deterministic GPU probe for integration tests only."""
    global _GPU_PROBE_OVERRIDE
    _GPU_PROBE_OVERRIDE = probe


def clear_gpu_probe_for_tests() -> None:
    global _GPU_PROBE_OVERRIDE
    _GPU_PROBE_OVERRIDE = None


_CORE_PROBE_CODE = r"""
import ctypes, json, sys

out = {"pyinit": False, "error": None}
try:
    library = ctypes.CDLL(sys.argv[1])
except Exception as error:
    out["error"] = f"{type(error).__name__}: {error}"
else:
    try:
        getattr(library, sys.argv[2])
    except AttributeError as error:
        out["error"] = f"{type(error).__name__}: {error}"
    else:
        out["pyinit"] = True
print(json.dumps(out))
"""


def _exports_python_init(path: Path, symbol: str = "PyInit__C") -> bool:
    """True when *path* is a loadable extension exporting CPython's ``PyInit__C``.

    The load happens in a child process running *this* interpreter, never in the orchestrator:
    dlopen executes the library's initialisers, and the library belongs to an environment this
    workbench does not trust. The child receives a filtered environment (no operator secrets).
    """
    try:
        import subprocess
    except Exception:  # pragma: no cover - stdlib import cannot fail
        return False
    child_env = _probe_child_env()
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-c", _CORE_PROBE_CODE, str(path), symbol],
            capture_output=True, text=True, timeout=60, env=child_env, cwd="/")
    except (OSError, ValueError):
        return False
    if completed.returncode != 0:
        return False
    try:
        return bool(json.loads(completed.stdout.splitlines()[-1]).get("pyinit"))
    except (ValueError, IndexError):
        return False


def verify_gpu_runtime(model: dict[str, Any], config: Any, *, authorized: bool = False,
                       require_cuda: bool = True, force: bool = False) -> dict[str, Any]:
    """Run the authorized GPU capability probe for a model and record the result.

    This is deliberately *not* part of inspection: it executes code from the selected
    interpreter, which is untrusted. It must therefore be explicitly authorized, it
    receives no secrets, and it never updates completion proofs or claims reproduction.

    Returns a structured capability result. Raises ``GpuVerificationError`` when the probe
    cannot establish that PyTorch can really use a CUDA device.
    """
    if not authorized:
        raise GpuVerificationError(
            "GPU verification requires explicit authorization (pass authorized=True)")

    if not force:
        # A capability already verified for exactly this environment is a recorded
        # observation, not a guess: an evaluation launch consumes the record instead of
        # re-running the probe. `force=True` re-probes (the operator's explicit re-check).
        existing = _cached_gpu_capability(model, verify_environment(model, probe=True, advisory=True))
        if existing is not None and existing.get("capability_verified"):
            return existing

    report = verify_environment(model, probe=True, authoritative=True)
    if report.tier in (TIER_UNCONFIGURED, "INTERPRETER_MISSING", TIER_GPU_DEFERRED) or not report.interpreter:
        raise GpuVerificationError(
            f"cannot probe GPU before the interpreter is usable [{report.tier}/{report.status}]")
    if report.tier == TIER_BLOCKED or report.status in _BLOCKED_STATUSES:
        raise GpuVerificationError(
            f"cannot probe GPU: the environment is blocked [{report.status}] {report.reason or ''}".strip())
    if report.tier not in READY_TIERS:
        raise GpuVerificationError(
            f"cannot probe GPU: the environment is not verified [{report.tier}/{report.status}]")

    interpreter = Path(report.interpreter)
    override = None
    if os.environ.get(GPU_PROBE_OVERRIDE_ENV) == "1":
        override = _GPU_PROBE_OVERRIDE
    if override is not None:
        result = dict(override(model, report))
    else:
        # `-I -S`: the probe must not import environment-provided site code either.
        probe_paths = tuple(str(path) for path in _site_packages_dirs(interpreter.parent.parent))
        ok, out, err = _run_probe(interpreter, GPU_PROBE_CODE, isolated=True, timeout=300,
                                  args=probe_paths)
        if not ok or not out:
            raise GpuVerificationError(f"GPU probe did not run: {err or 'no output'}")
        try:
            result = json.loads(out.splitlines()[-1])
        except (ValueError, IndexError) as error:
            raise GpuVerificationError(f"GPU probe returned unreadable output: {error}") from error

    torch_file = result.get("torch_module_file")
    torch_digest = None
    # The module must be inside the environment whose dependencies were just verified, so a
    # same-content module elsewhere on the machine cannot satisfy the digest. The reported
    # `_C` origin is a claim made by model code, so it is not trusted either: the orchestrator
    # loads the file itself and requires it to export CPython's `PyInit__C`. A copy of an
    # unrelated binary (or a hand-written file with a plausible header) therefore cannot
    # corroborate, whatever the probe's mutable metadata says.
    environment_root = interpreter.parent.parent
    if isinstance(torch_file, str) and torch_file:
        try:
            candidate = Path(torch_file).resolve()
            if candidate.is_file() and candidate.is_relative_to(environment_root.resolve()):
                from workbench.backend.registry import sha256_file

                torch_digest = sha256_file(candidate)
            else:
                result["torch_module_outside_environment"] = str(candidate)
        except (OSError, ValueError):
            torch_digest = None

    core_origin = result.get("torch_core_origin")
    core_loadable = False
    if isinstance(core_origin, str) and core_origin:
        try:
            core_candidate = Path(core_origin).resolve()
            core_loadable = bool(
                core_candidate.is_file()
                and core_candidate.is_relative_to(environment_root.resolve())
                and _exports_python_init(core_candidate, "PyInit__C"))
        except (OSError, ValueError):
            core_loadable = False
        if not core_loadable:
            result["torch_core_not_loadable"] = str(core_origin)
    result["torch_core_loadable"] = core_loadable
    result["torch_file_digest"] = torch_digest
    # A capability requires a real torch installation inside the verified environment, not a
    # single importable stub that answers the probe's questions.
    capability = bool(result.get("cuda_available") and result.get("allocated_tensor")
                      and result.get("computed") and result.get("torch_has_compiled_core")
                      and core_loadable and torch_digest)
    # An operator-selected adaptation makes a specific compute capability part of the contract:
    # torch must actually ship kernels for it (arch list) and the probed device must be one
    # (device capability). Without this check a cu128-less build could pass a generic CUDA probe
    # and then raise at the first real kernel launch.
    adaptation_target = report.runtime_adaptation
    if capability and adaptation_target:
        target = adaptation_target.replace("_blackwell", "").strip().lower()
        # torch's arch list carries the `sm_` prefix and may carry a `+PTX` suffix; the
        # adaptation names the capability either way, so both sides are normalized to the
        # bare capability number before comparison. Comparing a stripped target against a
        # prefixed list made the gate unsatisfiable and closed the launch path on the very
        # hardware the adaptation exists for (Blackwell sm_120).
        def _capability_number(value: str) -> str:
            text = value.strip().lower().split("+")[0]
            return text[3:] if text.startswith("sm_") else text

        arch_list = [str(a).lower() for a in result.get("torch_arch_list") or []]
        target_number = _capability_number(target)
        arch_numbers = [_capability_number(a) for a in arch_list]
        device_cap = str(result.get("compute_capability") or "").replace(".", "")
        if target_number not in arch_numbers:
            capability = False
            result["adaptation_mismatch"] = (
                f"adapted runtime targets {adaptation_target} but torch arch list lacks it: {arch_list}")
        elif device_cap and device_cap != target_number:
            capability = False
            result["adaptation_mismatch"] = (
                f"adapted runtime targets {adaptation_target} but the probed device is sm_{device_cap}")
    payload = {
        **result,
        "capability_verified": capability,
        "runtime_adaptation": adaptation_target,
        "interpreter": str(interpreter),
        # The fingerprint that ties this record to its environment must be the AUTHORITATIVE
        # one, because that is the fingerprint the capability key is derived from and the one
        # the reader recomputes when it decides whether the record still describes this
        # environment. Recording `report.environment_fingerprint` stored null on any probe
        # reached without a full environment verification, which silently disabled the record's
        # own identity check: a capability probed by one interpreter could satisfy another.
        # This runs once per capability probe — the place where hashing is affordable — not on
        # a status read.
        "interpreter_fingerprint": _record_environment_fingerprint(model, interpreter),
        "dependency_fingerprint": report.contract_digest,
        "probe_source": "override" if override is not None else "interpreter",
        "probed_at": datetime.now(timezone.utc).isoformat(),
        # A capability is bound to this host, driver and device visibility. It is evidence, not
        # proof against a hostile environment, and it expires whenever any of that identity
        # changes — the key changes with it, so a stale record is simply not found.
        "host_identity": _host_identity(),
        "revalidation": "re-probe when the interpreter, dependency set, host, driver or device visibility changes",
        "expires_on": ["interpreter change", "dependency change", "host change",
                       "driver change", "CUDA_VISIBLE_DEVICES change"],
    }
    if require_cuda and not capability:
        if result.get("cuda_available") and not result.get("torch_has_compiled_core"):
            reason = ("the imported torch has no compiled core extension, so the CUDA report "
                      "is not corroborated by a real PyTorch installation")
        else:
            reason = result.get("error") or "CUDA unavailable or unusable"
        record_gpu_capability(model, payload, config)
        raise GpuVerificationError(f"GPU capability not verified: {reason}")

    record_gpu_capability(model, payload, config)
    forget_environment()
    return payload


class GpuVerificationError(RuntimeError):
    """Raised when a GPU capability requirement cannot be verified."""


def environment_status(model: dict[str, Any], probe: bool = False) -> dict[str, Any]:
    """Backward-compatible dict view of the environment report."""
    report = verify_environment(model, probe=probe, advisory=True)
    data = asdict(report)
    # Legacy consumers read `status`/`interpreter`/`variable`/`tier`.
    data.setdefault("isolation_reason", data.get("reason"))
    data["verified"] = report.tier in READY_TIERS
    data["launch_ready"] = report.tier in LAUNCH_READY_TIERS
    data["gpu_deferred"] = report.tier == TIER_GPU_DEFERRED
    data["gpu_pending"] = report.tier == TIER_GPU_PENDING
    data["gpu_verified"] = report.tier == TIER_GPU_READY
    return data


def environment_blockers(model: dict[str, Any]) -> list[str]:
    """Every reason a model environment cannot be used for a real execution."""
    if not model.get("environment_required"):
        return []
    probe_environment = globals().get("verify_environment", verify_environment)
    report = probe_environment(model, probe=True, advisory=True)
    if report.tier in LAUNCH_READY_TIERS:
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
    other_paths: list[str] = []
    for line in (untracked + "\n" + ignored).splitlines():
        if line.strip():
            other_paths.append(_path_of(line))

    # A refusal must say WHY, and once. `git status --porcelain` lists a modified tracked file once,
    # while the untracked+ignored union can list an ignored path in both of its columns, so the same
    # divergence appeared up to three times in one list and an operator had to dedupe it by eye to see
    # whether one path or three were at fault.
    _dedupe = lambda items: list(dict.fromkeys(items))
    for line in tracked.splitlines():
        if not line.strip():
            continue
        path = _path_of(line)
        status_code = line[:2]
        if is_bytecode(path) and status_code == " D":
            # A *deleted* committed .pyc is generated bytecode that happened to be committed: four of
            # the pinned checkouts ship `__pycache__` (AirKnow 602 files, CSMCIR 498, INTENT 312,
            # SPRC 160), and Python removes stale caches just by being run against them. Treating that
            # as a source change would refuse every checkout the moment it is imported.
            continue
        # A tracked edit is normally a real divergence, and a *modified* committed .pyc is the
        # dangerous case: CPython imports a cache whose header matches the source, so a tampered .pyc
        # changes the code that runs while the .py stays byte-identical to the pin. But an approved
        # source patch is *also* a tracked edit -- that is what applying a patch does. It reaches the
        # same allowlist every other declared artifact uses, so a declared patch whose derived digest
        # matches is recorded as approved, and anything else is tracked.
        declaration = _declared_artifact_for(path, approved)
        if declaration is not None:
            verdict = _verify_declared_artifact(source, path, declaration)
            entries["approved" if verdict is None else "invalid"].append(
                path if verdict is None else f"{path}: {verdict}")
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
    for key in ("tracked", "unauthorized", "approved", "invalid"):
        entries[key] = _dedupe(entries[key])
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

    config = resolve_config()
    manifest = config.WORKBENCH_ARTIFACTS_ROOT / "auxiliary" / "download_manifest.json"
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
    if declaration["kind"] == "approved_source_patch":
        # An upstream defect may make the pinned source unusable (CSMCIR's 98 unguarded imports of
        # modules the public revision does not contain). The workbench cannot fix that in place, and
        # it will not: the patch is applied to a derived copy, and this declaration is what makes that
        # copy an approved variant rather than an unauthorized source change.
        #
        # Approval is exact. The base file and the derived file must both hash to what the registry
        # recorded, so any other edit -- to this file or to a sibling -- is not covered by the patch
        # and still blocks.
        if candidate.is_symlink():
            return "approved patch target must be a regular file"
        if not candidate.is_file():
            return "approved patch target is missing"
        from workbench.backend.registry import sha256_file

        digest = sha256_file(candidate)
        base = declaration.get("base_sha256")
        expected = declaration.get("derived_sha256")
        # Order matters for the message an operator reads: the published file is "not yet patched",
        # which is a different instruction from "someone edited this beyond the approved delta".
        if base and digest == base:
            return "the approved patch has not been applied"
        if expected and digest != expected:
            return ("modified source in an approved-patch checkout: the file matches neither the "
                    "published base nor the recorded post-patch digest")
        return None
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

def _source_dirty(source: Path, model: dict[str, Any] | None = None) -> bool | None:
    """True when the checkout differs from the pin in a way that matters.

    A *vendored* checkout (the delivered layout: upstream source tracked in this repository with
    the nested Git metadata removed) has no ``.git`` to run ``status`` against, so integrity comes
    from the recorded per-file digest manifest instead — see ``vendored_source``. It is also the
    stronger check: it cannot be fooled by a modification made after the pin.

    A nested checkout is still supported for an operator who syncs upstream into a scratch root.
    ``__pycache__``/``*.pyc`` are unavoidable products of running Python and do not change
    evaluator semantics. Declared, verified runtime artifacts (the canonical dataset link, audited
    auxiliary assets) are legitimate products of the workbench's own preparation contract.
    Everything else — tracked edits, and any untracked/ignored path that is *not* a verified
    declared artifact — is dirty.
    """
    if (source / PROVENANCE_FILENAME).is_file():
        provenance = vendor_provenance(model, source)
        if provenance.get("error"):
            return True
        return None
    entries = source_status(source)
    if entries is None:
        return None
    return bool(entries["tracked"] or entries["unauthorized"] or entries["invalid"])


def source_clean_and_pinned(model: dict[str, Any], source: Path) -> tuple[bool, str | None, str | None]:
    if (source / PROVENANCE_FILENAME).is_file() and not (source / ".git").exists():
        # Vendored layout: the pin is the recorded upstream commit and integrity is the content
        # manifest, so this needs no Git call at all. A checkout that still carries nested Git
        # metadata is NOT this layout — it falls through to the Git-backed path below, which is the
        # authority on what that repository actually has checked out.
        return vendor_source_clean_and_pinned(model, source)
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
    # A checkout that exists but has no recorded pin cannot be reproduced: the SHA it happens to
    # sit on is whatever someone left there. Reproducibility is the point of this contract, so an
    # unpinned checkout is refused even though there is nothing to compare against.
    if not expected:
        return False, head, "source pin missing: registry records no upstream_commit_sha"
    if head != expected:
        return False, head, f"source pin mismatch: expected {expected}, found {head}"
    return True, head, None


def source_provenance(model: dict[str, Any], source: Path) -> dict[str, Any]:
    """Record expected pin, actual pin, cleanliness, and a provenance digest.

    Shape is identical for both layouts so every consumer keeps working: a vendored checkout fills
    the same keys from the recorded manifest instead of from ``git status``.
    """
    import hashlib

    if (source / PROVENANCE_FILENAME).is_file() and not (source / ".git").exists():
        # Vendored layout only. A checkout that still carries nested Git metadata is answered by the
        # Git-backed path below: that repository's own HEAD is the authority on what is checked out,
        # and delegating to the manifest instead would let a git-based checkout — including a dirty
        # one — be reported as a verified vendored pin.
        return vendor_provenance(model, source)

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

    # A refusal must say WHY. `_source_dirty` can reject for three different reasons -- a tracked
    # edit, an undeclared path, or a declared artifact whose verification failed -- and this function
    # reported none of them: `error` stayed None while `verified` was False, so a caller could see the
    # refusal but not the cause. That is what made an approved-patch rejection look like a silent
    # failure, and it is the same shape of defect as reporting a dirty checkout as clean, only in the
    # opposite direction: the answer is right and the explanation is missing.
    if not error and dirty:
        reason = (entries.get("tracked") or entries.get("invalid") or entries.get("unauthorized")
                  or ["unknown divergence"])
        error = f"source checkout is dirty: {reason[0]}"

    clean_and_pinned = bool(actual and not dirty and (not expected or actual == expected))
    if not error and expected and actual and actual != expected:
        error = f"source pin mismatch: expected {expected}, found {actual}"
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


def runtime_plan_blockers(model: dict[str, Any], checkpoint: dict[str, Any], config: Any,
                           protocol: str | None = None,
                           dataset_root: Path | None = None) -> list[str]:
    """Prerequisites for *building* an evaluation plan, GPU excluded.

    Constructing a command is a planning act: it needs a verified interpreter, a resolved
    checkpoint, a pinned source, and the declared layout. Launching that command on a
    GPU-dependent model additionally needs a verified GPU capability, which is what
    ``runtime_blockers`` (the launch gate) enforces.
    """
    return [blocker for blocker in runtime_blockers(model, checkpoint, config, protocol, dataset_root)
            if not blocker.startswith("execution environment not verified:")]


def runtime_blockers(model: dict[str, Any], checkpoint: dict[str, Any], config: Any,
                     protocol: str | None = None, dataset_root: Path | None = None) -> list[str]:
    from workbench.backend.registry import (
        checkpoint_bundle_missing_paths, checkpoint_is_present, checkpoint_missing_paths,
        checkpoint_path, sha256_file,
    )
    from workbench.backend.preflight import preflight_blockers

    blockers = preflight_blockers(model, config, dataset_root=dataset_root)
    if protocol is not None and protocol not in model.get("supported_protocols", []):
        blockers.append(f"protocol incompatible: {protocol}")
    checkpoint_file = checkpoint_path(model["model_id"], checkpoint, config.WORKBENCH_CHECKPOINT_ROOT)
    if not checkpoint_is_present(checkpoint_file, checkpoint, model_id=model["model_id"], config=config):
        blockers.append(f"checkpoint missing: {checkpoint_missing_paths(checkpoint_file, checkpoint, model_id=model['model_id'], config=config)[0]}")
    if checkpoint.get("checkpoint_mapping_status") != "VERIFIED_METADATA":
        blockers.append("checkpoint mapping unresolved")
    expected_sha = checkpoint.get("expected_sha256")
    if expected_sha and checkpoint_file.is_file():
        # The availability digest is manifest-backed when the workbench downloaded the file
        # itself, so building a per-model readiness view does not re-read gigabytes for every
        # model on every request.
        from workbench.backend.registry import _availability_digest
        if _availability_digest(config, model["model_id"], checkpoint, checkpoint_file) != expected_sha:
            blockers.append("official checkpoint hash mismatch")
    # Only when a BUNDLE is what was selected. `checkpoint_availability` calls this for each
    # variant too, and a single-category checkpoint is not made unrunnable by a sibling category
    # that has not been downloaded -- reporting it as such marked every member unrunnable, which
    # disabled the frontend's per-checkpoint button and made an intentional single-category
    # evaluation (./pc.sh --checkpoint base_iter0_dress) impossible to launch.
    if checkpoint.get("bundle_id"):
        for bundle_id, missing in checkpoint_bundle_missing_paths(
                model, config.WORKBENCH_CHECKPOINT_ROOT, config=config).items():
            blockers.append(f"checkpoint bundle incomplete: {bundle_id}: {missing[0]}")

    return list(dict.fromkeys(blockers))
