"""Environment verification with explicit readiness tiers and source-backed probing."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
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

    ``isolated`` runs ``-I -S``: the user site and ``PYTHONPATH`` are ignored, and ``site``
    does not import ``sitecustomize``/``usercustomize``. The environment's directories are
    deliberately *not* added to ``sys.path``, because ``site.addsitedir`` executes the
    environment's ``.pth`` files. Nothing belonging to the environment runs, so it cannot
    answer for itself. This is used only for identity, which is not a package question.
    """
    prefix_flags = ["-I", "-S"] if isolated else []
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
    roots: list[Path] = []
    for candidate in (prefix, base_prefix):
        if candidate:
            for root in _site_packages_dirs(Path(candidate)):
                if root not in roots:
                    roots.append(root)

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
                    if (site_packages / f"{candidate}.py").is_file() or (site_packages / candidate).is_dir():
                        installed_content = True
                        modules.add(candidate)
                        break

            if installed_content:
                versions.setdefault(key, declared_version or None)
            elif not record.is_file() and (dist / "INSTALLER").is_file():
                # A package manager installed this (Debian's ``deb`` drops INSTALLER and no
                # RECORD) and the module it provides is named differently from the
                # distribution. That mapping is not decidable from disk, so the entry is
                # reported as unconfirmed rather than dropped or accepted.
                versions.setdefault(key, declared_version or None)
                unconfirmed.add(key)
            else:
                versions.pop(key, None)
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
    required = _required_packages(model)
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

    # CUDA capability is deliberately NOT probed. Its answer can only come from the
    # environment (``torch.cuda.is_available()``), and running environment code during
    # inspection is exactly what this module refuses to do: an environment could answer
    # for itself. A model declaring a CUDA requirement is reported as GPU verification
    # deferred, which is honest and is never accepted as readiness.
    if require_cuda or (env.get("cuda") and str(env["cuda"]).upper() != "UNKNOWN"):
        report.tier = TIER_GPU_DEFERRED
        report.status = "GPU_VERIFICATION_DEFERRED"
        report.reason = ("CUDA capability cannot be verified by inspecting the environment; "
                         "verify it on the GPU host")
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
        "interpreter_sha256": env.get("interpreter_sha256"),
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
    site_packages_dirs += [prefix_path / "lib" / "site-packages",
                           prefix_path / "Lib" / "site-packages"]
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
        for dist in sorted(list(site_packages.glob("*.dist-info")) + list(site_packages.glob("*.egg-info"))):
            metadata = dist / "METADATA" if dist.name.endswith(".dist-info") else dist / "PKG-INFO"
            try:
                stat_result = metadata.stat()
            except OSError:
                continue
            hasher.update(f"{dist.name}:{stat_result.st_size}:{stat_result.st_mtime_ns}".encode("utf-8"))
            record = dist / "RECORD"
            if not record.is_file():
                # A legacy egg-info has no RECORD: hash everything it contains.
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
    # A pinned-interpreter mismatch must never be masked by a cached verification.
    declared_digest = str((model.get("environment") or {}).get("interpreter_sha256") or "").strip().lower()
    if declared_digest and declared_digest.upper() != "UNKNOWN":
        if _interpreter_digest(interpreter) != declared_digest:
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
