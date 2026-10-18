"""Vendored-source provenance.

The upstream checkouts under ``third_party/<source_dir>`` are tracked source in *this*
repository, not nested Git repositories: the pinned upstream tree was copied in and its
Git metadata removed, so the delivered branch is fetchable without network access to the
sixteen upstreams. ``vendor_upstreams.py`` writes ``provenance.json`` next to each
checkout; this module is the one place that reads it.

Integrity is content-based rather than ``git status``-based. A ``git status`` check is
impossible against a tree with no ``.git``, and it is also weaker: it cannot tell a
reviewed patch from a silent edit made after the pin. Instead the vendoring step records
a digest per file plus an aggregate digest of the manifest, and monitoring re-reads the
files. Files the workbench is allowed to create at runtime (symlinks to the canonical
dataset, downloaded auxiliary captions) are declared in the registry and excluded from the
digest, so preparation does not make a checkout look tampered with.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

PROVENANCE_FILENAME = "provenance.json"

# The registry's `source_available` flag used to mean "a nested checkout exists". It now
# means "this repository carries the pinned source", and the execution contract reads the
# same names, so nothing downstream has to learn a second vocabulary.
VENDORED_STATE = "vendored_in_repo"


def load_vendor_provenance(source: Path) -> dict[str, Any] | None:
    try:
        return json.loads((source / PROVENANCE_FILENAME).read_text())
    except (OSError, ValueError):
        return None


def _declared_runtime_artifacts(model: dict[str, Any], source: Path | None = None) -> set[str]:
    """Paths preparation is allowed to create inside the checkout, relative to its root.

    These are artifacts the *workbench* creates at run time — the canonical FashionIQ symlink and
    any registered auxiliary asset whose destination is inside this checkout — so their bytes are
    not part of the pinned upstream tree and they are excluded from the digest.

    A recorded *local fix* is deliberately NOT in this set. A fix is a real source file with a known
    expected digest, so it stays in the digest and is verified like every other file; excluding it
    here would make an edit made after the fix was recorded invisible.
    """
    declared: set[str] = set()
    link = model.get("canonical_dataset_link")
    if isinstance(link, str) and link:
        declared.add(link)
    source_dir = model.get("source_dir")
    if isinstance(source_dir, str) and source_dir:
        # Imported here, not at module scope: registry.py owns the asset declarations and this
        # module is imported by runtime.py, which registry.py already depends on.
        from workbench.backend.registry import auxiliary_assets_for_model

        for asset in auxiliary_assets_for_model(model.get("model_id", "")):
            if asset.get("destination_scope") != "model_source":
                continue
            destination = asset.get("destination_path")
            if isinstance(destination, str) and destination:
                declared.add(destination)
    return declared


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def live_digest(source: Path, model: dict[str, Any] | None = None) -> dict[str, str]:
    """Content digest of every regular file in the checkout, minus declared artifacts."""
    declared = _declared_runtime_artifacts(model or {}, source)
    digests: dict[str, str] = {}
    for path in sorted(source.rglob("*")):
        relative = str(path.relative_to(source))
        if relative == PROVENANCE_FILENAME:
            continue
        if _is_excluded(path, relative, declared):
            continue
        if path.is_symlink():
            # A symlink is a link, not content: hash its target text so a repointed
            # canonical dataset link is still a visible change.
            digests[relative] = "symlink:" + str(path.readlink())
            continue
        if path.is_file():
            digests[relative] = _hash_file(path)
    return digests


def _is_excluded(path: Path, relative: str, declared: set[str]) -> bool:
    if "__pycache__" in path.parts or path.suffix in (".pyc", ".pyo"):
        return True
    if relative in declared:
        return True
    # A declared artifact may be a directory (an asset root); exclude its whole subtree.
    return any(relative.startswith(f"{entry}/") for entry in declared)


def manifest_digest(digests: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(digests, sort_keys=True).encode("utf-8")).hexdigest()


def source_provenance(model: dict[str, Any], source: Path) -> dict[str, Any]:
    """Expected upstream commit, what is on disk, and whether the two still agree."""
    record = load_vendor_provenance(source)
    expected = model.get("upstream_commit_sha")
    if (source / ".git").exists():
        # A checkout that carries BOTH the vendored manifest and nested Git metadata is the exact
        # state vendoring removes: the path reported as the pin would be whatever the nested
        # repository happens to point at, and the digest manifest would describe a tree nobody
        # version-controlled. Refusing it is the point of the contract, so neither path is preferred.
        return {
            "expected_commit": expected,
            "actual_commit": None,
            "local_fixes": [],
            "dirty": True,
            "tracked_modifications": [".git (nested Git metadata inside a vendored checkout)"],
            "unauthorized_untracked": [],
            "approved_runtime_artifacts": [],
            "invalid_runtime_artifacts": [],
            "error": "vendored checkout still carries nested Git metadata",
            "verified": False,
            "provenance_digest": None,
            "state": "vendored_with_nested_git",
        }
    if record is None:
        return {
            "expected_commit": expected,
            "actual_commit": None,
            "local_fixes": [],
            "dirty": None,
            "tracked_modifications": [],
            "unauthorized_untracked": [],
            "approved_runtime_artifacts": [],
            "invalid_runtime_artifacts": [],
            "error": f"{PROVENANCE_FILENAME} missing: checkout is not a vendored pin",
            "verified": False,
            "provenance_digest": None,
            "state": "unvendored",
        }
    actual = record.get("upstream_commit")
    manifest = record.get("file_manifest") or {}
    local_fixes = record.get("local_fixes") or []
    fixed_paths = {entry["path"] for entry in local_fixes}
    # A fixed file IS in the digest; what changes is the digest it must match. Upstream's bytes are
    # replaced by the delivered ones, so a fix is approved content and an edit made afterwards is a
    # modification like any other.
    expected_digests = dict(manifest)
    for entry in local_fixes:
        recorded = entry.get("delivered_sha256") or entry.get("upstream_sha256")
        if recorded:
            expected_digests[entry["path"]] = recorded
    live = live_digest(source, model)
    modified = sorted(
        path for path, digest in expected_digests.items()
        if path in live and live[path] != digest
    )
    removed = sorted(path for path in expected_digests if path not in live)
    # A recorded fix whose *file* is absent is reported by `removed` above; `added` covers content
    # nobody declared, which matters more now that the checkout is committed, because a stray file
    # there would ship with the branch.
    added = sorted(path for path in live if path not in manifest and path not in fixed_paths)
    approved = sorted(path for path in fixed_paths if path in live and path not in modified)
    error = None
    if modified or removed:
        error = (f"vendored source changed since vendoring: "
                 f"{len(modified)} modified, {len(removed)} removed"
                 + (f" (first: {modified[0]})" if modified else f" (first: {removed[0]})"))
    elif added:
        error = f"undeclared files present in checkout: {len(added)} (first: {added[0]})"
    verified = bool(actual and not error and (not expected or actual == expected))
    return {
        "expected_commit": expected,
        "actual_commit": actual,
        "local_fixes": local_fixes,
        "dirty": bool(error),
        "tracked_modifications": modified or removed,
        "unauthorized_untracked": added,
        "approved_runtime_artifacts": approved,
        "invalid_runtime_artifacts": [],
        "error": error,
        "verified": verified,
        "provenance_digest": manifest_digest(manifest) if manifest else None,
        "state": VENDORED_STATE,
        "resolved_url": record.get("resolved_url"),
        "upstream_commit_date": record.get("upstream_commit_date"),
        "license_file": record.get("license_file"),
        "vendored_at": record.get("vendored_at"),
    }

def source_clean_and_pinned(model: dict[str, Any], source: Path) -> tuple[bool, str | None, str | None]:
    """Pinned, unmodified vendored source. Signature preserved for existing callers."""
    provenance = source_provenance(model, source)
    if provenance.get("error"):
        return False, provenance.get("actual_commit"), provenance["error"]
    expected = model.get("upstream_commit_sha")
    actual = provenance.get("actual_commit")
    if not expected:
        return False, actual, "source pin missing: registry records no upstream_commit_sha"
    if actual != expected:
        return False, actual, f"source pin mismatch: expected {expected}, found {actual}"
    return True, actual, None
