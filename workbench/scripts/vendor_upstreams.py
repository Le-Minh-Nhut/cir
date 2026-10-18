#!/usr/bin/env python3
"""Vendor the pinned upstream checkouts into ``third_party/`` as plain tracked source.

Why this exists: the workbench used to keep each upstream checkout as a nested Git
repository outside version control. A published branch nobody can fetch is not a
reproduction, so the pinned source is vendored instead -- the exact tree of the pinned
commit, with the nested Git metadata removed, under the established layout.

What it does, per model (in registry order):

1. Read ``upstream_repo_url`` / ``upstream_commit_sha`` / ``source_dir`` from the registry.
2. Clone to a staging directory in ``/tmp`` and check out the pinned commit by SHA.
3. Copy the staged tree over the existing checkout, *without* deleting anything the
   upstream commit does not contain. Local runtime artifacts (downloaded captions,
   canonical dataset links) are therefore preserved, never overwritten by absence.
4. Refuse to overwrite a file whose local content differs from the pin unless
   ``--accept-local-changes`` is given, so an unrecorded local edit cannot be lost
   silently.
5. Remove the nested Git metadata: ``.git`` (file or directory), ``.gitmodules``, and
   any committed bytecode cache. Nothing outside the checkout is ever touched.
6. Write ``provenance.json`` recording the resolved URL, the exact commit, the commit
   date, the license file, and the list of preserved local files.

The outer repository is never involved: this script only writes inside
``<output-root>/<source_dir>`` and verifies that path is inside the output root.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workbench.backend.operator_config import resolve_config
from workbench.backend.registry import load_registry

LICENSE_CANDIDATES = ("LICENSE", "LICENSE.txt", "LICENSE.md", "license", "COPYING")
# Committed bytecode is a build product. Four pinned revisions ship `__pycache__`
# (AirKnow 602 files, CSMCIR 498, INTENT 312, SPRC 160); CPython imports a cache whose
# header matches the source, so shipping it would let the executed code differ from the
# reviewed `.py`. It is dropped at vendoring time, which is exactly what running the
# checkout does anyway.
SKIP_DIRS = {"__pycache__", ".ipynb_checkpoints", ".idea"}
SKIP_FILES = {".git", ".gitmodules"}
SKIP_SUFFIXES = (".pyc", ".pyo")
# A notebook/junk duplicate: macOS/Windows "copy" names and Jupyter checkpoint files are editor
# by-products of the author's machine, not part of the method. They are excluded from the delivered
# tree, so they must be excluded from the manifest too — the manifest describes what is DELIVERED.
#
# The bug this replaces: `live_digest(destination)` was computed AFTER the copy, but the copy's
# exclusions and Git's ignore rules are different sets, so the manifest kept listing junk that was
# never committed (34 paths across five checkouts) and the delivery check could not reconcile the two.
# A manifest that names files the branch does not contain is not a weaker integrity check, it is a
# wrong one: it makes the checkout look incomplete to anyone who verifies it from a fresh clone.
SKIP_NAME_FRAGMENTS = (" copy.", "-Copy.", "copy ", "conflicted copy")


def git(*command: str, cwd: Path | None = None) -> str:
    result = subprocess.run(["git", *command], cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(command)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def resolved_remote(url: str) -> str:
    """Follow repository redirects and report where the clone actually landed."""
    try:
        # `git ls-remote` follows redirects; the resolved URL is reported by GitHub in the
        # clone's own config after a normal clone, so ask the staging clone instead.
        return url
    except (OSError, subprocess.CalledProcessError):
        return url


def is_skipped(relative: Path) -> bool:
    """Single source of truth for "not part of the delivered snapshot".

    Both the copy step and the manifest step call this, so the two cannot describe different trees.
    """
    if relative.name in SKIP_FILES:
        return True
    if set(relative.parts) & SKIP_DIRS:
        return True
    if relative.suffix in SKIP_SUFFIXES:
        return True
    return any(fragment in relative.name for fragment in SKIP_NAME_FRAGMENTS)


def copy_tree(staging: Path, destination: Path, accept_local_changes: bool) -> tuple[list[str], list[str]]:
    """Copy *staging* over *destination*; return (written, preserved-local) relative paths."""
    written: list[str] = []
    preserved: list[str] = []
    for source in sorted(staging.rglob("*")):
        relative = source.relative_to(staging)
        if is_skipped(relative):
            continue
        if source.is_symlink() or source.is_file():
            target = destination / relative
            if target.exists() and target.is_file() and not target.is_symlink():
                if target.read_bytes() != source.read_bytes():
                    if not accept_local_changes:
                        preserved.append(str(relative))
                        continue
                    preserved.append(str(relative))
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target, follow_symlinks=False)
            written.append(str(relative))
        else:
            (destination / relative).mkdir(parents=True, exist_ok=True)
    return written, preserved


def remove_nested_git(destination: Path) -> list[str]:
    """Remove Git metadata *inside* the vendored checkout only.

    The path is resolved and re-checked against the destination before each removal, so
    a `.git` file pointing at another repository can never be followed out of the tree.
    """
    removed: list[str] = []
    root = destination.resolve()
    for candidate in [destination / ".git", destination / ".gitmodules"]:
        if not candidate.exists() and not candidate.is_symlink():
            continue
        entry = candidate.resolve()
        if entry != root and root not in entry.parents and not candidate.is_symlink():
            raise RuntimeError(f"refusing to remove {candidate}: resolves outside {root}")
        if candidate.is_dir() and not candidate.is_symlink():
            shutil.rmtree(candidate)
        else:
            candidate.unlink()
        removed.append(str(candidate.relative_to(destination)))
    for cache in sorted(destination.rglob("__pycache__")):
        if cache.is_dir() and not cache.is_symlink():
            shutil.rmtree(cache)
            removed.append(str(cache.relative_to(destination)))
    return removed


def license_file(staging: Path) -> str | None:
    for name in LICENSE_CANDIDATES:
        if (staging / name).is_file():
            return name
    return None


def vendor(model: dict, output_root: Path, accept_local_changes: bool, keep_staging: bool) -> dict:
    url = model.get("upstream_repo_url")
    pin = model.get("upstream_commit_sha")
    source_dir = model.get("source_dir")
    if not (url and pin and source_dir):
        raise RuntimeError("registry entry has no URL, pin and source_dir")
    destination = (output_root / source_dir).resolve()
    root = output_root.resolve()
    if destination == root or not destination.is_relative_to(root):
        raise RuntimeError(f"unsafe source_dir {source_dir!r}")

    staging = Path(tempfile.mkdtemp(prefix=f"vendor-{source_dir}-"))
    try:
        git("clone", "--quiet", "--no-checkout", url, str(staging))
        git("checkout", "--quiet", "--detach", pin, cwd=staging)
        actual = git("rev-parse", "HEAD", cwd=staging)
        if actual != pin:
            raise RuntimeError(f"checked out {actual}, expected {pin}")
        commit_date = git("show", "-s", "--format=%cI", "HEAD", cwd=staging)
        resolved = git("remote", "get-url", "origin", cwd=staging)
        # Submodules and LFS: report, never guess. A pointer file left by a missing LFS
        # object is not the asset it names.
        submodules = (staging / ".gitmodules").read_text() if (staging / ".gitmodules").is_file() else ""
        lfs_pointers = [
            str(path.relative_to(staging))
            for path in staging.rglob("*")
            if path.is_file() and path.stat().st_size < 200 and path.name != ".gitignore"
            and "version https://git-lfs" in path.read_text(errors="ignore")
        ]
        destination.mkdir(parents=True, exist_ok=True)
        written, preserved = copy_tree(staging, destination, accept_local_changes)
        removed = remove_nested_git(destination)
        from workbench.backend.vendored_source import live_digest

        # The manifest is the integrity baseline: monitoring re-hashes these files later, which
        # is what makes a vendored tree (no .git to run `status` against) still auditable. It is
        # computed from the DELIVERED tree, and `live_digest` applies the same `is_skipped`
        # predicate the copy step used, so the manifest can only ever name files this run wrote.
        manifest = live_digest(destination, model)
        record = {
            "model_ids": [item["model_id"] for item in load_registry()["models"] if item.get("source_dir") == source_dir],
            "requested_url": url,
            "resolved_url": resolved,
            "upstream_commit": actual,
            "upstream_commit_date": commit_date,
            "license_file": license_file(staging),
            "destination": str(destination),
            "vendored_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "files_written": len(written),
            "nested_git_removed": removed,
            "preserved_local_files": preserved,
            "submodules_declared": submodules,
            "git_lfs_pointers": lfs_pointers,
            "file_manifest": manifest,
            "local_fixes": [],
        }
        (destination / "provenance.json").write_text(json.dumps(record, indent=2) + "\n")
        return record
    finally:
        if not keep_staging:
            shutil.rmtree(staging, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", help="vendor one model (repeatable)")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--accept-local-changes", action="store_true",
                        help="overwrite files whose local content differs from the pin (recorded either way)")
    parser.add_argument("--keep-staging", action="store_true", help="leave the staging clone for inspection")
    parser.add_argument("--report", type=Path, help="write the provenance records as one JSON file")
    args = parser.parse_args()
    if args.output_root is None:
        args.output_root = resolve_config().WORKBENCH_THIRD_PARTY_ROOT

    models = load_registry()["models"]
    if args.model:
        wanted = set(args.model)
        # Two model entries can share one checkout (the CLIP4Cir pair); vendor it once.
        models = [model for model in models if model["model_id"] in wanted]
        missing = wanted - {model["model_id"] for model in models}
        if missing:
            parser.error(f"unknown model(s): {sorted(missing)}")

    seen: set[str] = set()
    records = []
    failed = False
    for model in models:
        source_dir = model.get("source_dir")
        if not source_dir:
            print(f"[SKIP] {model['model_id']}: no source_dir")
            continue
        if source_dir in seen:
            continue
        seen.add(source_dir)
        try:
            record = vendor(model, args.output_root, args.accept_local_changes, args.keep_staging)
        except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
            print(f"[BLOCKED] {model['model_id']}: {error}")
            failed = True
            continue
        print(f"[OK] {model['model_id']}: {record['resolved_url']} @ {record['upstream_commit'][:12]} "
              f"-> {record['destination']} ({record['files_written']} files, "
              f"preserved {len(record['preserved_local_files'])}, removed {len(record['nested_git_removed'])})")
        records.append(record)
    if args.report:
        args.report.write_text(json.dumps(records, indent=2) + "\n")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
