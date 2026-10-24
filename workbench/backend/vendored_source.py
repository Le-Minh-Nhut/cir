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
        record = json.loads((source / PROVENANCE_FILENAME).read_text())
    except (OSError, ValueError):
        return None
    if (not isinstance(record, dict) or not isinstance(record.get("upstream_commit"), str)
            or not record["upstream_commit"] or not isinstance(record.get("file_manifest"), dict)
            or not isinstance(record.get("local_fixes", []), list)
            or any(not isinstance(fix, dict) or not isinstance(fix.get("path"), str)
                   for fix in record.get("local_fixes", []))):
        return None
    return record


def _declared_runtime_artifacts(model: dict[str, Any]) -> set[str]:
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
        from workbench.backend.registry import (auxiliary_assets_for_model,
                                                evaluator_relative_asset_paths)

        for asset in auxiliary_assets_for_model(model.get("model_id", "")):
            if asset.get("destination_scope") != "model_source":
                continue
            destination = asset.get("destination_path")
            if isinstance(destination, str) and destination:
                declared.add(destination)
        # An external asset whose declared location IS a path inside the evaluator's checkout (the
        # pinned loader opens it by a literal relative name and no variable can redirect it) is
        # installed by preparing, not by editing source. Without this the four files authorized
        # preparation creates for TME -- the two base weights and the two feature-cache files --
        # made the checkout report `verified=False, dirty=True` with every one of them listed as
        # unauthorized, which blocked the source gate BEFORE any model loader ran and made an
        # authorized preparation look like tampering.
        #
        # Narrow on purpose: these are exactly the declared assets and generated artifacts of this
        # model's own contract, not the `features`, `weight`, `checkout` or `third_party` tree at
        # large. A real source edit and an undeclared executable are still reported.
        declared.update(evaluator_relative_asset_paths(model))
        from workbench.backend.registry import preparation_contract_for_model, preparation_paths

        contract = preparation_contract_for_model(model)
        if contract:
            # `generated_artifacts` declared under the `evaluator_cwd` scope are, by definition,
            # files inside this checkout. Resolving them against a placeholder root yields the
            # checkout-relative path without needing the configured third-party root here.
            for path in preparation_paths(contract, Path("."), "generated_artifacts",
                                          evaluator_root=Path(".")):
                relative = Path(str(path))
                if not relative.is_absolute() and ".." not in relative.parts:
                    declared.add(relative.as_posix())
    return declared


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def live_digest(source: Path, model: dict[str, Any] | None = None) -> dict[str, str]:
    """Content digest of every regular file in the checkout, minus declared artifacts."""
    declared = _declared_runtime_artifacts(model or {})
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


#: Editor/notebook by-products that are never part of a delivered snapshot. These are the SAME
#: exclusions the vendoring script applies when it copies a checkout in, so the manifest it computes
#: afterwards and the tree Git actually commits describe one snapshot rather than two. Keeping the
#: predicate in one place is what makes that true: a manifest that names files the branch does not
#: contain makes every fresh-clone verification of that checkout fail.
UNWANTED_DIRS = {"__pycache__", ".ipynb_checkpoints", ".idea"}
UNWANTED_SUFFIXES = (".pyc", ".pyo")
UNWANTED_NAME_FRAGMENTS = (" copy.", "-Copy.", "copy ", "conflicted copy")


def is_unwanted(relative: Path) -> bool:
    """True for a path that is not part of the delivered snapshot."""
    if set(relative.parts) & UNWANTED_DIRS:
        return True
    if relative.suffix in UNWANTED_SUFFIXES:
        return True
    return any(fragment in relative.name for fragment in UNWANTED_NAME_FRAGMENTS)


def _is_excluded(path: Path, relative: str, declared: set[str]) -> bool:
    if is_unwanted(Path(relative)):
        return True
    if relative in declared:
        return True
    # A declared artifact may be a directory (an asset root); exclude its whole subtree.
    return any(relative.startswith(f"{entry}/") for entry in declared)


def manifest_digest(digests: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(digests, sort_keys=True).encode("utf-8")).hexdigest()


def _git_backed_provenance(model: dict[str, Any], source: Path, expected: str | None) -> dict[str, Any]:
    """The Git-backed answer for a checkout that carries a nested repository.

    Imported lazily: `runtime` imports this module at load time, so a module-scope import of it here
    would be circular. `runtime.source_provenance` sends a checkout *with* nested Git metadata down
    its own Git path, so this delegates rather than keeping a second copy of that logic that could
    drift from it.
    """
    from workbench.backend.runtime import source_provenance as runtime_source_provenance

    provenance = runtime_source_provenance(model, source)
    provenance.setdefault("local_fixes", [])
    provenance.setdefault("state", "nested_git_checkout")
    return provenance


def source_provenance(model: dict[str, Any], source: Path) -> dict[str, Any]:
    """Expected upstream commit, what is on disk, and whether the two still agree."""
    record = load_vendor_provenance(source)
    expected = model.get("upstream_commit_sha")
    if (source / ".git").exists():
        # Nested Git metadata means this is a checkout someone synced, not the vendored layout. It
        # wins over any manifest that happens to sit beside it: the repository's own HEAD is the
        # authority on what is checked out, and honouring a stale manifest instead would let a
        # git-based checkout — including a dirty one — be reported as a verified vendored pin.
        return _git_backed_provenance(model, source, expected)
    if record is None:
        reason = (f"{PROVENANCE_FILENAME} corrupt: invalid vendored provenance record"
                  if (source / PROVENANCE_FILENAME).exists()
                  else f"{PROVENANCE_FILENAME} missing: checkout is not a vendored pin")
        return {
            "expected_commit": expected,
            "actual_commit": None,
            "local_fixes": [],
            "dirty": None,
            "tracked_modifications": [],
            "unauthorized_untracked": [],
            "approved_runtime_artifacts": [],
            "invalid_runtime_artifacts": [],
            "error": reason,
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
    # A model may declare one APPROVED PATCH for a file its pinned source cannot run without. The
    # declaration is exact in both directions -- the file must match the published base digest or the
    # recorded post-patch digest, and nothing else -- so the approved delta is accepted here exactly
    # as the nested-Git path accepts it in `runtime._verify_declared_artifact`. Without this the
    # documented CSMCIR recipe failed its own integrity check: applying the registry's approved patch
    # made the file hash to `derived_sha256` and it was still reported as an unauthorized modification.
    patch_contracts = {
        patch["path"]: patch for patch in (model.get("approved_source_patches") or [])
        if patch.get("path")
    }
    live = live_digest(source, model)
    modified: list[str] = []
    patch_rejections: list[str] = []
    for path, digest in sorted(expected_digests.items()):
        if path not in live or live[path] == digest:
            continue
        contract = patch_contracts.get(path)
        if contract is None:
            modified.append(path)
            continue
        base = contract.get("base_sha256")
        derived = contract.get("derived_sha256")
        if derived and live[path] == derived:
            continue  # the approved delta is applied, byte-for-byte
        if base and live[path] == base:
            patch_rejections.append(f"{path}: the approved patch has not been applied")
        else:
            patch_rejections.append(
                f"{path}: modified source in an approved-patch checkout (matches neither the "
                f"published base nor the recorded post-patch digest)")
    # A manifest path that is *also* a registry-declared runtime artifact is not missing: the
    # declaration means preparation may create it, and this branch now SHIPS it (CSMCIR's CoT captions
    # are both a published asset and tracked source). `live_digest` excludes declared artifacts from
    # the digest, so without this the checkout reported three files it actually contained as removed
    # -- and the same mistake in the reverse direction made every declared artifact look like
    # undeclared content. The reported state has to describe the tree, not the digest's exclusions.
    declared_paths = _declared_runtime_artifacts(model or {}) if model else set()
    on_disk = {str(path.relative_to(source)) for path in source.rglob("*") if path.is_file()}
    removed = sorted(path for path in expected_digests if path not in live and path not in on_disk)
    approved = sorted(set(
        [path for path in declared_paths if path in on_disk and path in expected_digests]
        + [path for path in fixed_paths if path in live and path not in modified]
        + [path for path in patch_contracts
           if path in live and live.get(path) == patch_contracts[path].get("derived_sha256")]
    ))
    # A recorded fix whose *file* is absent is reported by `removed` above; `added` covers content
    # nobody declared, which matters more now that the checkout is committed, because a stray file
    # there would ship with the branch.
    added = sorted(path for path in live if path not in manifest and path not in fixed_paths
                   and path not in patch_contracts and path not in declared_paths)
    error = None
    if modified or removed:
        error = (f"vendored source changed since vendoring: "
                 f"{len(modified)} modified, {len(removed)} removed"
                 + (f" (first: {modified[0]})" if modified else f" (first: {removed[0]})"))
    elif patch_rejections:
        error = patch_rejections[0]
    elif added:
        error = f"undeclared files present in checkout: {len(added)} (first: {added[0]})"
    verified = bool(actual and not error and (not expected or actual == expected))
    return {
        "expected_commit": expected,
        "actual_commit": actual,
        "local_fixes": local_fixes,
        "dirty": bool(error),
        "tracked_modifications": (modified or removed or patch_rejections),
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
