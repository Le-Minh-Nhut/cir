"""The single source-identity rule used by planning, provisioning and runtime verification.

A dependency can be declared as a DIRECT REFERENCE -- `name @ <url>`, optionally with a `@<rev>` --
and then the installed distribution must come from that source, not merely share its name. Two
separate implementations of that rule existed and both were wrong in opposite directions:

* the planner compared the raw declaration text, so `name @  git+URL@rev` (a legal, if unusual,
  spacing) never matched the recorded metadata and a CORRECT install was refused;
* the runtime read only `Requirement.name` and `Requirement.specifier`, so `Requirement.url` was
  discarded and a WRONG source -- PyPI `clip==0.2.0` where OpenAI CLIP was declared -- was accepted
  as `RUNTIME_READY`, and then reused.

Both now resolve through this module, so "the same source" means one thing everywhere.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

try:  # `packaging` is a workbench dependency; the fallback keeps the module importable without it.
    from packaging.requirements import InvalidRequirement, Requirement
except ImportError:  # pragma: no cover - packaging is declared in workbench/backend/requirements.txt
    InvalidRequirement = ValueError  # type: ignore[assignment,misc]
    Requirement = None  # type: ignore[assignment]


@dataclass(frozen=True)
class SourceIdentity:
    """Where a distribution must come from, in comparable form.

    ``repository`` is the scheme-less, prefix-less, trailing-slash-less form of the URL
    (`github.com/openai/CLIP.git`), which is what a recorded `direct_url.json` carries once it is
    normalized the same way. ``revision`` is the declared revision, or None when the declaration
    pins none -- which is itself information: a revision-less declaration cannot be used to accept or
    reject a specific commit.
    """

    name: str
    repository: str | None
    revision: str | None
    raw: str

    @property
    def requires_source(self) -> bool:
        return bool(self.repository)


def _normalize_repository(url: str) -> str:
    """A URL reduced to the part that identifies the REPOSITORY.

    `git+https://github.com/openai/CLIP.git@rev`, `https://github.com/openai/CLIP.git` and
    `https://github.com/openai/CLIP` are the same repository. VCS prefixes (`git+`), userinfo and a
    trailing `.git`/`/` carry no identity, so they are removed rather than compared as text.
    """
    text = url.strip()
    for prefix in ("git+", "hg+", "svn+", "bzr+"):
        if text.startswith(prefix):
            text = text[len(prefix):]
    text = text.strip()
    parts = urlsplit(text)
    host = parts.netloc.rsplit("@", 1)[-1].lower()          # drop any userinfo
    path = parts.path.rstrip("/")
    if path.endswith(".git"):
        path = path[: -len(".git")]
    return urlunsplit((parts.scheme.lower(), host, path, "", ""))


def split_reference(request: str, *, has_name_prefix: bool = True) -> tuple[str, str]:
    """`(repository_url, revision_or_empty)` for a `url[@rev]` or `name @ url[@rev]` string.

    Parsed structurally rather than by string surgery. The revision separator only counts AFTER the
    URL has started, and the old `partition("@")[2]` split broke in two ways: it left the whitespace
    of a legal `name @ git+URL@rev` attached to the URL, and -- when handed a bare URL, which is what
    `packaging` returns in `Requirement.url` -- it treated the revision as everything after the first
    `@` and the URL as the part before it.

    `has_name_prefix` says which shape is being read, because the two are genuinely ambiguous from
    the text alone: `clip @ git+https://host/x.git@abc` has the name before the first `@`, while
    `git+https://host/x.git@abc` does not.
    """
    text = request.strip()
    if has_name_prefix:
        _, separator, remainder = text.partition("@")
        if not separator:
            return "", ""
        text = remainder.strip()
    if not text:
        return "", ""
    if "@" not in text:
        return text, ""
    # The revision is the final `@segment`. A URL may itself contain `@` (userinfo), so the split is
    # taken from the right and only when the tail plausibly is a revision rather than a host: a tail
    # containing `/` or a `.` before any candidate separator is part of the path.
    head, _, tail = text.rpartition("@")
    if not tail or "/" in tail:
        return text, ""
    return head, tail


def identity_of_declaration(declaration: str) -> SourceIdentity | None:
    """The source identity a registry/requirements declaration names, or None if it names none.

    A requirement that is not a direct reference (a plain name, or `name==version`) has no source
    identity: it says nothing about where the distribution comes from, and callers must not treat
    its absence as either proof or contradiction.
    """
    text = declaration.strip()
    if not text:
        return None
    if Requirement is not None:
        try:
            requirement = Requirement(text)
        except (InvalidRequirement, ValueError):
            requirement = None
        if requirement is not None and requirement.url:
            # `Requirement.url` is the URL alone, with no `name @` prefix.
            repository, revision = split_reference(requirement.url, has_name_prefix=False)
            return SourceIdentity(requirement.name, _normalize_repository(repository) or None,
                                  revision or None, text)
        if requirement is not None and not requirement.url:
            return None
    # A declaration `packaging` cannot parse but that carries `name @ url` explicitly.
    name, separator, remainder = text.partition("@")
    if not separator or not remainder.strip():
        return None
    repository, revision = split_reference(remainder)
    if "://" not in repository and not repository.startswith("git+"):
        return None
    return SourceIdentity(name.strip(), _normalize_repository(repository) or None,
                          revision or None, text)


def recorded_source(interpreter: Path, distribution: str) -> dict | None:
    """The `direct_url.json` a package manager wrote for an installed distribution, or None.

    Read as structured data, never as a concatenated string: `vcs_info` carries the resolved commit
    for a VCS install and `url` carries the location, and comparing them requires knowing which part
    is which. None means the distribution was installed from an index, or is not installed at all --
    both of which are "no source recorded", which is a refusal when a source was declared.
    """
    probe = (
        "import json, sys\n"
        "from importlib.metadata import PackageNotFoundError, distribution\n"
        "try:\n"
        "    meta = distribution(sys.argv[1])\n"
        "except PackageNotFoundError:\n"
        "    raise SystemExit(0)\n"
        "raw = meta.read_text('direct_url.json')\n"
        "if raw:\n"
        "    sys.stdout.write(raw)\n")
    try:
        result = subprocess.run([str(interpreter), "-c", probe, distribution],
                                capture_output=True, text=True, check=False, timeout=60)
    except Exception:
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    try:
        payload = json.loads(result.stdout)
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


def mismatch(declared: SourceIdentity, recorded: dict | None) -> str | None:
    """Why the recorded source does not satisfy the declared one, or None when it does.

    Every refusal names WHICH part differs, because "the source differs" is not actionable when the
    difference is a revision versus a repository.
    """
    if not declared.requires_source:
        return None
    if recorded is None:
        return (f"{declared.name} was declared from {declared.raw!r}, but the environment records no "
                "source for it: the installed distribution did not come from the declared source")
    recorded_url = str(recorded.get("url") or "")
    vcs = recorded.get("vcs_info") or {}
    recorded_commit = str(vcs.get("resolved_commit_id") or vcs.get("commit_id") or "")
    recorded_repository = _normalize_repository(recorded_url) if recorded_url else None
    if declared.repository and declared.repository != recorded_repository:
        return (f"{declared.name} was declared from repository {declared.repository!r}, but the "
                f"environment installed it from {recorded_repository or recorded_url!r}")
    if declared.revision and declared.revision != recorded_commit:
        # A resolved commit that differs from a requested revision is a different build even when the
        # repository matches -- and a `requested_revision` in the metadata does not repair that,
        # because the resolved commit is what was actually installed.
        return (f"{declared.name} was declared at revision {declared.revision!r}, but the "
                f"environment resolved {recorded_commit or 'no commit'!r}")
    return None
