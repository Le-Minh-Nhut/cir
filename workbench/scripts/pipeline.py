#!/usr/bin/env python3
"""Orchestrate existing workbench operator scripts without reimplementing them."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import stat
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workbench.backend.operator_config import WorkbenchConfig, resolve_config
from workbench.backend.registry import auxiliary_model_ids, load_registry
SCRIPTS = Path(__file__).resolve().parent
REPOSITORY = SCRIPTS.parents[1]
def _artifacts_root(config: WorkbenchConfig | None = None) -> Path:
    cfg = config or resolve_config()
    configured = getattr(cfg, "WORKBENCH_ARTIFACTS_ROOT", None)
    return Path(configured) if configured is not None else Path(getattr(cfg, "CIR_REPO_ROOT", REPOSITORY)) / "workbench" / "artifacts"


@dataclass(frozen=True)
class Stage:
    name: str
    commands: tuple[list[str], ...] = ()
    skipped: str | None = None
    model_id: str | None = None
    checkpoint_id: str | None = None
    protocol: str | None = None
    dependencies: tuple[str, ...] = ()
    input_paths: tuple[Path, ...] = ()
    output_paths: tuple[Path, ...] = ()
    required_capabilities: frozenset[str] = frozenset()
    receipt: bool = False
    run_id: str | None = None
    evidence: Callable[[dict | None], tuple[bool, str | None, dict | None]] | None = None
    resume_refusal_reason: str | None = None
    checkpoint_policy: str | None = None
    checkpoint_selection: tuple[str, ...] = ()
    checkpoint_overrides: tuple[tuple[str, tuple[str, ...]], ...] = ()


def stage_key(stage: Stage) -> str:
    parts = [stage.name]
    if stage.model_id:
        parts.append(stage.model_id)
    if stage.checkpoint_id:
        parts.append(stage.checkpoint_id)
    if stage.protocol:
        parts.append(stage.protocol)
    return ":".join(parts)


def dependency_key(stage: Stage) -> str:
    """Checkpoint/protocol-independent key used by stage dependency edges."""
    return f"{stage.name}:{stage.model_id}" if stage.model_id else stage.name



# Entry-level evidence categories. A digest alone does not say whether every required byte was
# read, so the category travels with it and decides whether resume may trust the fingerprint.
CONTENT_VERIFIED = "CONTENT_VERIFIED"      # the file's bytes were hashed
METADATA_ONLY = "METADATA_ONLY"            # too large to hash inline: name/size/mtime only
UNREADABLE = "UNREADABLE"                  # present but could not be read
MISSING = "MISSING"                        # declared but absent
SYMLINK_VERIFIED = "SYMLINK_VERIFIED"      # a symlink resolved inside the walked root
SYMLINK_DANGLING = "SYMLINK_DANGLING"      # a symlink whose target does not resolve
SYMLINK_EXTERNAL = "SYMLINK_EXTERNAL"      # a symlink leaving the root: content unverifiable
UNREADABLE_DIRECTORY = "UNREADABLE_DIRECTORY"   # a directory the walk could not enter
NOT_A_FILE = "NOT_A_FILE"                  # neither a regular file, directory, nor symlink

# Tree-level evidence categories.
TREE_CONTENT_VERIFIED = "TREE_CONTENT_VERIFIED"   # every entry inside was content-verified
TREE_PARTIAL = "TREE_PARTIAL"                     # some entry was not content-verified
TREE_TRUNCATED = "TREE_TRUNCATED"                 # the inventory budget was exhausted
TREE_UNREADABLE = "TREE_UNREADABLE"               # the tree itself could not be walked
TREE_MISSING = "TREE_MISSING"                     # the tree does not exist

# Only these authorize a verified resume: a fingerprint that claims strong verification must be
# backed by content for everything it covers.
# Metadata is not content: only a real digest or a recorded link target authorizes skip.
VERIFIED_ENTRY_CATEGORIES = frozenset({CONTENT_VERIFIED, SYMLINK_VERIFIED})
#: Never verified evidence, and always a reason to refuse a verified resume.
UNVERIFIED_ENTRY_CATEGORIES = frozenset({
    UNREADABLE, MISSING, NOT_A_FILE, SYMLINK_DANGLING, SYMLINK_EXTERNAL,
    UNREADABLE_DIRECTORY, METADATA_ONLY,
})
VERIFIED_TREE_CATEGORIES = frozenset({TREE_CONTENT_VERIFIED})

# Files above this size are hashed by streaming rather than loaded into memory. There is no size
# at which content stops being required for verification: a checkpoint is a scientific input and
# its bytes are what the fingerprint must bind. The limit below only chooses the read strategy.
_STREAM_HASH_THRESHOLD = 8 * 1024 * 1024
_STREAM_CHUNK = 4 * 1024 * 1024
# A declared directory is walked with a fixed budget so a pathological tree cannot run forever.
# A truncated walk is *partial* evidence, never verified.
_WALK_LIMIT = 200_000
_WALK_SIZE_LIMIT = 64 * 1024 * 1024 * 1024


def _stream_digest(path: Path) -> str:
    """SHA-256 of a file read in bounded chunks, so a multi-GiB input needs no large allocation."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_STREAM_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_digest(path: Path) -> str:
    """Content digest of a regular file, streaming when it is large."""
    try:
        if path.stat().st_size > _STREAM_HASH_THRESHOLD:
            return _stream_digest(path)
    except OSError:
        pass
    from workbench.backend.registry import sha256_file

    return sha256_file(path)


def _entry_evidence(path: Path, root: Path, inventory: "hashlib._Hash") -> tuple[str, int]:
    """Fold one filesystem entry into *inventory* and report its evidence category.

    Every entry contributes its relative path and kind; regular files contribute their content
    digest, symlinks the target string they point at. Nothing here silently downgrades a failure
    into verified evidence: an unreadable file, a dangling link and a non-regular object each
    have their own category, and the caller turns any of them into PARTIAL.
    """
    relative = path.relative_to(root)
    try:
        details = path.lstat()
    except OSError:
        inventory.update(f"{MISSING}\0{relative}\0".encode("utf-8"))
        return MISSING, 0

    size = details.st_size
    if stat.S_ISLNK(details.st_mode):
        # A symlink's identity is the target it names, not the bytes it resolves to, so a retarget
        # invalidates the fingerprint even when both targets hold identical content.
        try:
            target = os.readlink(path)
        except OSError:
            inventory.update(f"{UNREADABLE}\0{relative}\0".encode("utf-8"))
            return UNREADABLE, size
        resolved = path.resolve()
        if not resolved.exists():
            category = SYMLINK_DANGLING
        elif _inside_root(resolved, root):
            # A link that stays inside the walked root is covered by the target's own entry in
            # this same walk, so its content is accounted for.
            category = SYMLINK_VERIFIED
        else:
            # A link that leaves the root points at content this walk never reads. Treating that as
            # verified evidence would let an external file change without invalidating resume, so
            # it is recorded as unverified rather than assumed.
            category = SYMLINK_EXTERNAL
        inventory.update(f"L\0{relative}\0{target}\0{category}\0".encode("utf-8"))
        return category, size

    if stat.S_ISDIR(details.st_mode):
        inventory.update(f"D\0{relative}\0".encode("utf-8"))
        return CONTENT_VERIFIED, 0

    if not stat.S_ISREG(details.st_mode):
        inventory.update(f"{NOT_A_FILE}\0{relative}\0".encode("utf-8"))
        return NOT_A_FILE, size

    try:
        digest = _file_digest(path)
    except OSError:
        inventory.update(f"{UNREADABLE}\0{relative}\0{size}\0".encode("utf-8"))
        return UNREADABLE, size
    inventory.update(f"F\0{relative}\0{size}\0{digest}\0".encode("utf-8"))
    return CONTENT_VERIFIED, size


def _is_verified_entry(category: str) -> bool:
    """True only for evidence that a verified resume may rely on.

    Written as an explicit test so a newly introduced category cannot default to *verified* by
    being absent from a list someone forgot to update.
    """
    return category in VERIFIED_ENTRY_CATEGORIES


def _inside_root(candidate: Path, root: Path) -> bool:
    """True when *candidate* lies within *root* (both already resolved)."""
    try:
        root_resolved = root.resolve()
    except OSError:
        return False
    return candidate == root_resolved or root_resolved in candidate.parents


def _walk_evidence(root: Path, *, skip_paths: frozenset[Path] = frozenset()) -> tuple["hashlib._Hash | None", str, int, int, bool]:
    """Inventory a directory and classify its evidence.

    Returns ``(hash, category, files, bytes, truncated)``; ``hash`` is None only when the tree
    could not be walked at all. Follows no symlinks, so a link cycle cannot expand the walk, and
    the tree-level category is PARTIAL whenever a single entry was not content-verified.
    """
    if not root.exists():
        return None, TREE_MISSING, 0, 0, False
    if not root.is_dir():
        return None, TREE_UNREADABLE, 0, 0, False

    inventory = hashlib.sha256()
    files = 0
    total = 0
    truncated = False
    partial = False
    walk_errors: list[str] = []

    def _on_error(error: OSError) -> None:
        """Record a directory the walk could not enter instead of silently skipping it.

        `os.walk` ignores an unreadable directory by default, so a tree whose contents were never
        read could otherwise be reported as fully content-verified.
        """
        walk_errors.append(str(getattr(error, "filename", "?")))

    # Python build products are not inputs, and this is where it matters most: merely RUNNING an
    # evaluator writes `__pycache__` into its own checkout. Counting those bytes made the evaluation
    # input digest change as a side effect of the previous evaluation, so the stage invalidated
    # itself, `--resume` was refused, and the run got a new `run_id` after evaluating nothing
    # differently. `runtime.is_bytecode` already states the policy for the source-provenance path
    # ("`__pycache__`/*.pyc are Python build products that cannot change evaluator semantics");
    # the walk now asks the same predicate instead of inventing a second answer. Nothing unverified
    # is lost: a `.pyc` is a cache of a declared `.py` input, whose bytes this walk still hashes.
    from workbench.backend.runtime import is_bytecode

    try:
        for current, directories, names in os.walk(root, followlinks=False, onerror=_on_error):
            directories.sort()
            names.sort()
            # Directory symlinks arrive through `directories` and are inventory entries in their
            # own right: a retargeted link to another directory must be visible even though the
            # walk deliberately does not descend into it.
            for name in list(directories):
                candidate = Path(current) / name
                if candidate in skip_paths:
                    directories.remove(name)
                    continue
                if candidate.is_symlink():
                    category, size = _entry_evidence(candidate, root, inventory)
                    files += 1
                    total += size
                    partial = partial or not _is_verified_entry(category)
            for name in names:
                candidate = Path(current) / name
                if candidate in skip_paths or is_bytecode(str(candidate)):
                    continue
                category, size = _entry_evidence(candidate, root, inventory)
                files += 1
                total += size
                partial = partial or not _is_verified_entry(category)
                if files >= _WALK_LIMIT or total >= _WALK_SIZE_LIMIT:
                    truncated = True
                    break
            if truncated:
                break
    except OSError:
        return None, TREE_UNREADABLE, files, total, truncated

    if truncated:
        inventory.update(f"{TREE_TRUNCATED}:{files}:{total}".encode("utf-8"))
        return inventory, TREE_TRUNCATED, files, total, True
    if walk_errors:
        # A directory that could not be entered has contents this inventory never saw.
        for failed in sorted(walk_errors):
            inventory.update(f"{UNREADABLE_DIRECTORY}\0{failed}\0".encode("utf-8"))
        inventory.update(TREE_PARTIAL.encode("utf-8"))
        return inventory, TREE_PARTIAL, files, total, False
    if partial:
        inventory.update(TREE_PARTIAL.encode("utf-8"))
        return inventory, TREE_PARTIAL, files, total, False
    inventory.update(TREE_CONTENT_VERIFIED.encode("utf-8"))
    return inventory, TREE_CONTENT_VERIFIED, files, total, False


def _walk_digest(root: Path) -> tuple[str, int, int, bool]:
    """Backwards-compatible view of :func:`_walk_evidence`: ``(digest, files, bytes, truncated)``."""
    inventory, category, files, total, truncated = _walk_evidence(root)
    digest = inventory.hexdigest() if inventory is not None else ""
    return digest, files, total, truncated or category != TREE_CONTENT_VERIFIED


def _stage_input_fingerprint_evidence(stage: Stage, config: WorkbenchConfig | None = None,
                                      state: dict | None = None) -> dict:
    import hashlib
    import stat
    hasher = hashlib.sha256()
    hasher.update(stage.name.encode("utf-8"))
    if stage.model_id:
        hasher.update(f"MODEL:{stage.model_id}".encode("utf-8"))
    if stage.checkpoint_id:
        hasher.update(f"CKPT:{stage.checkpoint_id}".encode("utf-8"))
    if stage.protocol:
        hasher.update(f"PROTO:{stage.protocol}".encode("utf-8"))
    for cmd in stage.commands:
        hasher.update(shlex.join(cmd).encode("utf-8"))

    if stage.name == "environment-plan":
        for cmd in stage.commands:
            for idx, arg in enumerate(cmd):
                if arg in ("--model", "-m") and idx + 1 < len(cmd):
                    hasher.update(f"MODEL_SEL:{cmd[idx+1]}".encode("utf-8"))
                elif arg in ("--selected-models",) and idx + 1 < len(cmd):
                    hasher.update(f"MODELS_SEL:{cmd[idx+1]}".encode("utf-8"))
                elif arg in ("--third-party-root", "--source-root") and idx + 1 < len(cmd):
                    hasher.update(f"SOURCE_ROOT:{cmd[idx+1]}".encode("utf-8"))
                elif arg in ("--artifacts-root",) and idx + 1 < len(cmd):
                    hasher.update(f"ARTIFACTS_ROOT:{cmd[idx+1]}".encode("utf-8"))
                elif arg in ("--output",) and idx + 1 < len(cmd):
                    hasher.update(f"OUTPUT:{cmd[idx+1]}".encode("utf-8"))
    # The preprocessing recipe is part of what a result means: the same data evaluated under a
    # different layout, caption policy or image extension is a different experiment. It is a
    # derived identity rather than a file, so a model whose registry declaration changes
    # invalidates resume even though no input file moved.
    if stage.model_id and config:
        from workbench.backend.registry import model_by_id

        try:
            recipe_digest = preparation_recipe_digest(
                model_by_id(stage.model_id),
                checkpoint_policy=stage.checkpoint_policy if stage.name == "evaluation" else None,
                checkpoint_selection=stage.checkpoint_selection if stage.name == "evaluation" else (),
                checkpoint_overrides=stage.checkpoint_overrides if stage.name == "evaluation" else ())
        except Exception:
            recipe_digest = "UNKNOWN_RECIPE"
        hasher.update(f"RECIPE:{recipe_digest}".encode("utf-8"))

    # Evaluation identity covers the code that executes the experiment without making it an input
    # or weakening resume: implementation digests are derived identity, like the recipe.
    if stage.name == "evaluation" and stage.model_id and config:
        from workbench.backend.registry import model_by_id

        try:
            execution_identity = _execution_identity(model_by_id(stage.model_id), config)
        except Exception as error:
            execution_identity = {"digest": f"UNKNOWN:{type(error).__name__}:{error}"}
        hasher.update(f"EXECUTION:{execution_identity['digest']}".encode("utf-8"))

    inputs = []
    weak = False
    for path in stage.input_paths:
        item = {"path": str(path)}
        hasher.update(str(path).encode("utf-8"))
        try:
            details = path.lstat()
        except FileNotFoundError:
            category = MISSING
        except OSError as error:
            category = UNREADABLE
            item["error"] = f"{type(error).__name__}: {error}"
        else:
            if stat.S_ISLNK(details.st_mode):
                try:
                    target = os.readlink(path)
                except OSError as error:
                    category = UNREADABLE
                    item["error"] = f"{type(error).__name__}: {error}"
                else:
                    item["symlink_target"] = target
                    try:
                        resolved = path.resolve(strict=True)
                    except (FileNotFoundError, RuntimeError):
                        category = SYMLINK_DANGLING
                        hasher.update(f"L:{target}:{category}".encode("utf-8"))
                    except OSError as error:
                        category = UNREADABLE
                        item["error"] = f"{type(error).__name__}: {error}"
                    else:
                        item["resolved_target"] = str(resolved)
                        roots = [config.WORKBENCH_THIRD_PARTY_ROOT] if config else []
                        if config:
                            roots.append(config.FASHIONIQ_ROOT)
                        for declared in stage.input_paths:
                            if declared.name in {"captions", "image_splits", "images", "resized_image", "resized_images"}:
                                roots.append(declared.parent)
                            elif declared.parent.name in {"captions", "image_splits", "images", "resized_image", "resized_images"}:
                                roots.append(declared.parent.parent)
                        if not any(_inside_root(resolved, root) for root in roots):
                            category = SYMLINK_EXTERNAL
                            hasher.update(f"L:{target}:{resolved}:{category}".encode("utf-8"))
                        else:
                            try:
                                target_details = resolved.stat()
                                if stat.S_ISDIR(target_details.st_mode):
                                    skip_paths = frozenset()
                                    inventory, target_category, files, total, truncated = _walk_evidence(resolved)
                                    item.update(target_tree_files=files, target_tree_bytes=total,
                                                target_tree_truncated=truncated,
                                                target_tree_category=target_category)
                                    if inventory is None:
                                        category = UNREADABLE
                                    else:
                                        target_digest = inventory.hexdigest()
                                        item["target_tree_digest"] = target_digest
                                        category = (SYMLINK_VERIFIED
                                                    if target_category == TREE_CONTENT_VERIFIED else UNREADABLE)
                                        hasher.update(
                                            f"L:{target}:{resolved}:{target_category}:{target_digest}".encode("utf-8"))
                                elif stat.S_ISREG(target_details.st_mode):
                                    target_digest = _file_digest(resolved)
                                    category = SYMLINK_VERIFIED
                                    item["target_sha256"] = target_digest
                                    hasher.update(f"L:{target}:{resolved}:{target_digest}".encode("utf-8"))
                                else:
                                    category = NOT_A_FILE
                                    item["error"] = "symlink target is not a regular file or directory"
                            except OSError as error:
                                category = UNREADABLE
                                item["error"] = f"{type(error).__name__}: {error}"
            elif stat.S_ISDIR(details.st_mode):
                skip_paths = frozenset()
                if stage.name == "evaluation" and stage.model_id and config:
                    from workbench.backend.registry import model_by_id
                    from workbench.backend.fashioniq_layout import canonical_dataset_link_path

                    model = model_by_id(stage.model_id)
                    declared_link = canonical_dataset_link_path(model, config.WORKBENCH_THIRD_PARTY_ROOT)
                    if declared_link and path == declared_link.parent:
                        skip_paths = frozenset({declared_link})
                inventory, category, walk_files, walk_bytes, walk_truncated = _walk_evidence(
                    path, skip_paths=skip_paths)
                item.update(tree_files=walk_files, tree_bytes=walk_bytes,
                            tree_truncated=walk_truncated)
                if inventory is None:
                    item["error"] = f"tree could not be walked: {path}"
                    hasher.update(f"{category}\0".encode("utf-8"))
                else:
                    item["tree_digest"] = inventory.hexdigest()
                    hasher.update(f"TREE:{inventory.hexdigest()}:{category}".encode("utf-8"))
            elif stat.S_ISREG(details.st_mode):
                try:
                    content_digest = _file_digest(path)
                except OSError as error:
                    category = UNREADABLE
                    item["error"] = f"{type(error).__name__}: {error}"
                else:
                    category = CONTENT_VERIFIED
                    item["sha256"] = content_digest
                    hasher.update(content_digest.encode("utf-8"))
            else:
                category = NOT_A_FILE
                item["error"] = "path is not a regular file, directory, or symlink"
        item["category"] = category
        inputs.append(item)
        hasher.update(f":{category}\0".encode("utf-8"))
        if not _is_verified_entry(category) and category not in VERIFIED_TREE_CATEGORIES:
            weak = True

    source_info = None
    if stage.model_id and config:
        from workbench.backend.registry import model_by_id
        from workbench.backend.runtime import source_provenance

        try:
            model = model_by_id(stage.model_id)
        except KeyError:
            pass
        except Exception as error:
            source_info = {"category": "HASH_FAILED", "error": f"{type(error).__name__}: {error}"}
            hasher.update(f"SOURCE:{source_info['category']}:{source_info['error']}".encode("utf-8"))
            weak = True
        else:
            source_name = model.get("source_dir")
            if source_name:
                source_dir = config.WORKBENCH_THIRD_PARTY_ROOT / source_name
                source_info = {"path": str(source_dir)}
                try:
                    source_stat = source_dir.stat()
                except FileNotFoundError:
                    source_info["category"] = "MISSING"
                except OSError as error:
                    source_info.update(category="UNREADABLE", error=f"{type(error).__name__}: {error}")
                else:
                    if not stat.S_ISDIR(source_stat.st_mode):
                        source_info["category"] = "UNREADABLE"
                    else:
                        # ONE call, and only the category depends on whether the checkout has
                        # nested Git metadata. The vendored layout deliberately has NO `.git` (the
                        # pinned tree is committed in this repository), so an absent `.git` is the
                        # normal case and is answered by the vendored-provenance authority -- exactly
                        # as `source_provenance` itself decides. Probing `(source_dir / ".git").stat()`
                        # and treating its FileNotFoundError as an I/O failure was a real defect:
                        # EVERY vendored model's fingerprint was marked weak, so `--resume` refused
                        # every completed stage that depends on source.
                        try:
                            provenance = source_provenance(model, source_dir)
                        except Exception as error:
                            source_info.update(category=UNREADABLE,
                                               error=f"{type(error).__name__}: {error}")
                        else:
                            source_info["actual_commit"] = provenance.get("actual_commit")
                            source_info["dirty"] = provenance.get("dirty")
                            source_info["provenance_digest"] = provenance.get("provenance_digest")
                            if provenance.get("actual_commit") and provenance.get("dirty") is False:
                                source_info["category"] = "CONTENT_VERIFIED"
                            elif provenance.get("error"):
                                source_info.update(category="UNREADABLE", error=provenance["error"])
                            else:
                                source_info["category"] = "UNVERIFIED"
                hasher.update(f"SOURCE:{source_info['category']}:{source_info.get('provenance_digest')}"
                              .encode("utf-8"))
                if source_info.get("error"):
                    hasher.update(source_info["error"].encode("utf-8"))
                if source_info["category"] != "CONTENT_VERIFIED":
                    weak = True

    if state and stage.dependencies:
        stages_rec = state.get("stages", {})
        for dep in stage.dependencies:
            dep_data = stages_rec.get(dep, {})
            hasher.update(f"DEP:{dep}:{dep_data.get('status')}:{dep_data.get('fingerprint')}".encode("utf-8"))

    return {"digest": hasher.hexdigest(), "weak": weak, "inputs": inputs,
            "source_provenance": source_info}


def compute_stage_input_fingerprint(stage: Stage, config: WorkbenchConfig | None = None,
                                    state: dict | None = None) -> str:
    return _stage_input_fingerprint_evidence(stage, config, state)["digest"]


def stage_input_fingerprint_info(stage: Stage, config: WorkbenchConfig | None = None,
                                 state: dict | None = None) -> dict:
    """Return a digest plus per-input evidence; only content-verified inputs are resumable."""
    return _stage_input_fingerprint_evidence(stage, config, state)


def stage_receipt_path(config: WorkbenchConfig, stage: Stage) -> Path:
    key = stage_key(stage).replace(":", "_")
    return _artifacts_root(config) / "receipts" / f"{key}.json"


def write_stage_receipt(config: WorkbenchConfig, stage: Stage, *, return_code: int,
                        fingerprint: str | None = None) -> Path:
    """Durable completion proof for a provisioning stage that produces no artifact file.

    The receipt records the input fingerprint that produced it. Without it, the path — a pure
    function of the stage key — would let a receipt from an earlier, different invocation satisfy
    output validation for a later one, so a stage could be finalized `COMPLETE` on the strength of a
    run whose inputs no longer apply.
    """
    path = stage_receipt_path(config, stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage_id": stage_key(stage),
        "model_id": stage.model_id,
        "commands": [list(c) for c in stage.commands],
        "return_code": return_code,
        "input_fingerprint": fingerprint,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "kind": "stage_completion_receipt",
    }
    temp = path.with_suffix(f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)
    return path




def compute_stage_output_fingerprints(stage: Stage) -> dict[str, str]:
    from workbench.backend.registry import sha256_file

    res = {}
    for p in stage.output_paths:
        if not p.exists():
            res[str(p)] = "MISSING"
        elif p.is_file():
            if p.stat().st_size == 0:
                res[str(p)] = "EMPTY"
            else:
                try:
                    res[str(p)] = sha256_file(p)
                except Exception:
                    res[str(p)] = "ERROR"
        elif p.is_dir():
            if not any(p.iterdir()):
                res[str(p)] = "EMPTY"
            else:
                # Hash the resolved target so a symlinked dataset link is content-verified.
                target = p.resolve()
                try:
                    res[str(p)] = sha256_file(target)
                except Exception:
                    res[str(p)] = "DIR"
    return res


def database_path(config: WorkbenchConfig) -> Path:
    """Where the derived result index lives."""
    return _artifacts_root(config) / "workbench.duckdb"


def validate_stage_outputs(stage: Stage, recorded: dict[str, str] | None = None,
                           config: WorkbenchConfig | None = None,
                           record: dict | None = None) -> bool:
    from workbench.backend.registry import sha256_file

    if stage.evidence is not None:
        # The stage's completion proof is supplied by the stage itself: for an
        # evaluation this is the run-bound completion artifact of THIS invocation,
        # so a leftover report from an earlier run can never satisfy it.
        ok, _, _ = stage.evidence(record)
        return bool(ok)

    if stage.receipt and not stage.output_paths:
        cfg = config or resolve_config()
        path = stage_receipt_path(cfg, stage)
        if not path.is_file() or path.stat().st_size == 0:
            return False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return False
        if payload.get("stage_id") != stage_key(stage) or payload.get("return_code") != 0:
            return False
        # A receipt is only evidence for the invocation that wrote it. A record without a
        # fingerprint (or one recorded before this field existed) cannot be tied to the inputs at
        # hand, so it is not trusted; the stage simply runs again.
        recorded_fingerprint = (record or {}).get("fingerprint")
        if recorded_fingerprint is None:
            return False
        return payload.get("input_fingerprint") == recorded_fingerprint

    if not stage.output_paths:
        # Read-only stages may be skipped on fingerprint alone. A side-effecting stage
        # must carry a durable completion receipt; without one it is not trusted.
        return not stage.required_capabilities
    for p in stage.output_paths:
        if not p.exists():
            return False
        if p.is_dir():
            # A declared directory output must resolve to a populated tree. A symlinked
            # directory is acceptable only when its target is itself populated (the
            # CSMCIR dataset link is exactly this case); dangling/empty links fail.
            if not any(p.iterdir()):
                return False
        if p.is_file():
            if p.stat().st_size == 0:
                return False
            if p.suffix == ".json":
                try:
                    payload = json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    return False
                # A run manifest (anything named *_latest.json) must reference an
                # existing non-empty run artifact; an empty/garbage manifest is not proof.
                if p.name.endswith("_latest.json"):
                    if not isinstance(payload, dict) or not isinstance(payload.get("report"), str):
                        return False
                    referenced = Path(payload["report"])
                    if not referenced.is_file() or referenced.stat().st_size == 0:
                        return False
                    try:
                        json.loads(referenced.read_text(encoding="utf-8"))
                    except Exception:
                        return False
        if recorded and str(p) in recorded:
            expected = recorded[str(p)]
            if expected in ("MISSING", "EMPTY"):
                return False
            try:
                if p.is_dir():
                    actual = sha256_file(p.resolve())
                elif p.is_file():
                    actual = sha256_file(p)
                else:
                    actual = None
                if actual != expected:
                    return False
            except Exception:
                return False
    return True


_UNUSABLE_DEP_STATUSES = {"FAILED", "BLOCKED_AUTHORIZATION_REQUIRED", "SKIPPED_DEPENDENCY", "INTERRUPTED"}


class WorkflowLockError(RuntimeError):
    """Raised when another live workflow already owns this state file."""


def _lock_path(state_path: Path) -> Path:
    return state_path.with_suffix(state_path.suffix + ".lock")


_LOCK_HANDLES: dict[str, object] = {}


def acquire_workflow_lock(state_path: Path, *, timeout: float = 0.0):
    """Acquire an exclusive advisory lock on the workflow state.

    Uses ``flock`` so the lock is released automatically if the process dies.
    Returns the live file object; the caller must keep it referenced and pass it
    to ``release_workflow_lock``. Raises WorkflowLockError if held by a live owner
    and ``timeout`` elapses.
    """
    import fcntl
    import time

    path = _lock_path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "w")
    deadline = time.monotonic() + max(timeout, 0.0)
    while True:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            handle.write(f"{os.getpid()}\n")
            handle.flush()
            _LOCK_HANDLES[str(path)] = handle
            return handle
        except OSError:
            if time.monotonic() >= deadline:
                handle.close()
                raise WorkflowLockError(
                    f"another workflow owns {state_path} (lock held); retry later"
                )
            time.sleep(0.05)


def release_workflow_lock(handle) -> None:
    import fcntl
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass
    try:
        handle.close()
    except OSError:
        pass
    for key, value in list(_LOCK_HANDLES.items()):
        if value is handle:
            del _LOCK_HANDLES[key]


def load_pipeline_state(path: Path | None = None) -> dict:
    path = path or _artifacts_root() / "pipeline_state.json"
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def save_pipeline_state(state: dict, path: Path | None = None) -> None:
    import os
    path = path or _artifacts_root() / "pipeline_state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def new_run_id() -> str:
    """Invocation identity shared by the master and the evaluator it launches."""
    import uuid

    return f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}_{uuid.uuid4().hex[:8]}"


def command(script: str, *arguments: str | Path) -> list[str]:
    """Launch a workbench operator script under the controller interpreter (sys.executable).

    Auxiliary asset acquisition (download_auxiliary_assets.py) executes under this controller
    interpreter rather than model interpreters, so shared multi-model assets are acquired and
    verified before model-specific environments are provisioned.
    """
    return [sys.executable, str(SCRIPTS / script), *(str(argument) for argument in arguments)]


def run_stage(
    name: str,
    *commands: list[str],
    skipped: str | None = None,
    model_id: str | None = None,
    checkpoint_id: str | None = None,
    protocol: str | None = None,
    checkpoint_policy: str | None = None,
    checkpoint_selection: tuple[str, ...] = (),
    checkpoint_overrides: tuple[tuple[str, tuple[str, ...]], ...] = (),
    dependencies: tuple[str, ...] = (),
    input_paths: tuple[Path, ...] = (),
    output_paths: tuple[Path, ...] = (),
    required_capabilities: frozenset[str] | set[str] | tuple[str, ...] = frozenset(),
    required_capability: str | None = None,
    receipt: bool = False,
    run_id: str | None = None,
    evidence: Callable[[dict | None], tuple[bool, str | None, dict | None]] | None = None,
    resume_refusal_reason: str | None = None,
) -> Stage:
    caps = frozenset(required_capabilities)
    if required_capability:
        caps = caps | {required_capability}
    return Stage(
        name=name,
        commands=commands,
        skipped=skipped,
        model_id=model_id,
        checkpoint_id=checkpoint_id,
        checkpoint_policy=checkpoint_policy,
        checkpoint_selection=checkpoint_selection,
        checkpoint_overrides=checkpoint_overrides,
        dependencies=dependencies,
        input_paths=input_paths,
        output_paths=output_paths,
        required_capabilities=caps,
        receipt=receipt,
        run_id=run_id,
        evidence=evidence,
        resume_refusal_reason=resume_refusal_reason,
    )


def selection(args: argparse.Namespace, *, runnable: bool = False) -> list[str]:
    return ["--model", args.model] if args.model else (["--all-runnable"] if runnable else ["--all"])


def dataset_root(args: argparse.Namespace, config: WorkbenchConfig) -> Path:
    return args.dataset_root or config.FASHIONIQ_ROOT


def dataset_check(args: argparse.Namespace, config: WorkbenchConfig) -> Stage:
    root = dataset_root(args, config)
    val_splits = tuple(root / "image_splits" / f"split.{c}.val.json" for c in ("dress", "shirt", "toptee"))
    return run_stage("dataset", command("prepare_dataset.py", "--dataset-root", root, "--check-only"), input_paths=val_splits)


def layout_prepare(args: argparse.Namespace, config: WorkbenchConfig) -> Stage:
    active = args.model == "csmcir" or (args.model is None and any((args.sync_sources, args.download_checkpoints, args.download_auxiliary_assets, args.evaluate)))
    if not active:
        return Stage("dataset-link", skipped="CSMCIR preparation not selected")
    link = config.WORKBENCH_THIRD_PARTY_ROOT / "CSMCIR" / "fashionIQ_dataset"
    target = dataset_root(args, config)
    is_valid_link = False
    if link.is_symlink():
        try:
            is_valid_link = link.resolve() == target.resolve()
        except (OSError, RuntimeError):
            is_valid_link = False
    if is_valid_link:
        return run_stage(
            "dataset-link",
            command("prepare_dataset.py", "--dataset-root", target, "--model", "csmcir", "--check-only"),
            model_id="csmcir",
            output_paths=(link,),
            required_capabilities=frozenset(),
        )
    return run_stage(
        "dataset-link",
        command("prepare_dataset.py", "--dataset-root", target, "--model", "csmcir"),
        model_id="csmcir",
        output_paths=(link,),
        required_capability="allow_preparation",
    )


def _execution_identity(model: dict, config: WorkbenchConfig) -> dict:
    root = REPOSITORY
    digest = hashlib.sha256()

    def add_file(label: str, path: Path) -> None:
        try:
            content = _file_digest(path)
        except OSError as error:
            content = f"UNREADABLE:{type(error).__name__}:{error}"
        digest.update(f"{label}:{content}\0".encode("utf-8"))

    code_files = [*(root / "workbench" / "replay").rglob("*.py"),
                  *(root / "workbench" / "backend" / "adapters").glob("*.py"),
                  root / "workbench" / "scripts" / "evaluate_models.py"]
    for path in sorted(set(code_files)):
        if "__pycache__" not in path.parts:
            add_file(path.relative_to(root).as_posix(), path)

    env = model.get("environment") or {}
    variable = env.get("python_env_var")
    from workbench.backend.model_settings import setting as _setting

    configured = (_setting(variable) or "") if variable else ""
    interpreter = configured or (sys.executable if not variable and not model.get("environment_required")
                                 else "UNCONFIGURED")
    digest.update(f"INTERPRETER:{interpreter}\0".encode("utf-8"))
    if interpreter != "UNCONFIGURED":
        interpreter_path = Path(interpreter).expanduser()
        # The environment prefix must come from the *invoked* path: a virtualenv's `python` is a
        # symlink to the base interpreter, so resolving it first would point the package scan at
        # the base installation instead of the environment being evaluated.
        environment_prefix = interpreter_path.parent.parent
        try:
            interpreter_path = interpreter_path.resolve(strict=True)
        except (OSError, RuntimeError):
            digest.update(b"INTERPRETER_CONTENT:UNAVAILABLE\0")
        else:
            if interpreter_path.is_file():
                add_file("INTERPRETER_CONTENT", interpreter_path)
            else:
                digest.update(b"INTERPRETER_CONTENT:UNAVAILABLE\0")
            # The interpreter binary alone does not describe the environment. Installing,
            # upgrading or force-reinstalling a dependency changes what the evaluator imports
            # without touching the interpreter file, the requirements files or our source, so a
            # run would resume and reuse a result produced by different package content. The
            # installed-distribution fingerprint already hashes each distribution's metadata and
            # the files its RECORD lists, which is the evidence the GPU capability cache uses —
            # reuse it rather than inventing a second package identity.
            try:
                from workbench.backend import runtime as _runtime

                digest.update(f"ENVIRONMENT_CONTENT:{_runtime._interpreter_fingerprint(str(interpreter_path), _runtime._environment_contract_digest(model), str(environment_prefix))}\0".encode("utf-8"))
            except Exception as error:
                # An unreadable environment is not a reason to claim identity we could not
                # establish: record the gap so the fingerprint differs from a verified one.
                digest.update(f"ENVIRONMENT_CONTENT:UNVERIFIED:{type(error).__name__}\0".encode("utf-8"))

    source_name = model.get("source_dir")
    if source_name:
        source = config.WORKBENCH_THIRD_PARTY_ROOT / source_name
        lock_names = {"environment.yml", "environment.yaml", "conda-lock.yml", "conda-lock.yaml",
                      "environment.lock.yml", "environment.lock.yaml", "requirements.txt",
                      "poetry.lock", "uv.lock", "Pipfile.lock"}
        if source.is_dir():
            paths = [source / name for name in lock_names]
            paths.extend(source.glob("requirements*.lock"))
            for path in sorted(set(paths)):
                if path.is_file():
                    add_file(f"LOCK:{path.relative_to(source).as_posix()}", path)
    return {"digest": digest.hexdigest()}


def preparation_recipe_digest(model_dict: dict, *, checkpoint_policy: str | None = None,
                               checkpoint_selection: tuple[str, ...] = (),
                               checkpoint_overrides: tuple[tuple[str, tuple[str, ...]], ...] = ()) -> str:
    """Digest of the preprocessing recipe this model is evaluated under.

    The recipe is what turns raw FashionIQ into the model's inputs: layout, caption policy,
    image extension, auxiliary assets and checkpoint selection. Recording it as part of the
    evaluation fingerprint means changing the recipe invalidates resume even when no data file
    moved.
    """
    relevant = {
        "model_id": model_dict.get("model_id"),
        "fashioniq_layout": model_dict.get("fashioniq_layout"),
        "fashioniq_image_extensions": model_dict.get("fashioniq_image_extensions"),
        "preparation_contract": model_dict.get("preparation_contract"),
        "preparation_id": model_dict.get("preparation_id"),
        "native_protocol": model_dict.get("native_protocol"),
        "supported_protocols": model_dict.get("supported_protocols"),
        "literature_split_label": model_dict.get("literature_split_label"),
        "validation_caption_policy": model_dict.get("validation_caption_policy"),
        "correction_policy": model_dict.get("correction_policy"),
    }
    if checkpoint_policy is not None or checkpoint_selection or checkpoint_overrides:
        relevant["checkpoint_policy"] = checkpoint_policy
        relevant["checkpoint_selection"] = list(checkpoint_selection)
        relevant["checkpoint_overrides"] = {mid: list(ids) for mid, ids in checkpoint_overrides}
    return hashlib.sha256(
        json.dumps(relevant, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def recipe_path(config: WorkbenchConfig, model_id: str) -> Path:
    return config.WORKBENCH_RESULTS_ROOT / "__preparation_recipe__" / f"{model_id}.json"


def write_preparation_recipe(model_dict: dict, config: WorkbenchConfig) -> Path:
    """Record this model's recipe so later runs can prove they used the same one."""
    path = recipe_path(config, model_dict["model_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_id": model_dict["model_id"],
        "recipe_digest": preparation_recipe_digest(model_dict),
        "upstream_commit_sha": model_dict.get("upstream_commit_sha"),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return path


def model_eval_inputs(model_dict: dict, args: argparse.Namespace, config: WorkbenchConfig, checkpoint_ids: Sequence[str] | None = None) -> tuple[Path, ...]:
    """Every scientific input this model's evaluation actually consumes.

    The evaluation stage is fingerprinted over this list, so a file that changes without appearing
    here changes the model's answer without invalidating resume. The list is therefore built from
    the model's source-backed layout, its registered auxiliary assets, and its checkpoint
    selection — not from a shared guess about what FashionIQ looks like.

    The image *directories* are included per model because the evaluators open image files whose
    names come from the splits: a rewritten image inside a declared directory must invalidate the
    stage, and the directory walk content-hashes every member.
    """
    from workbench.backend.registry import (
        auxiliary_assets_for_model, auxiliary_destination, checkpoint_path,
        external_asset_statuses, fashioniq_required_paths, load_registry,
        preparation_contract_for_model, preparation_paths,
    )

    inputs: list[Path] = []
    mid = model_dict["model_id"]

    # Checkpoint inputs must match what the evaluator resolves when no selection is explicit.
    ckpts = model_dict.get("checkpoint_variants") or []
    selected_ckpt_ids = _expand_selection(
        model_dict, list(checkpoint_ids) if checkpoint_ids is not None
        else evaluated_checkpoint_ids(model_dict, args, config.WORKBENCH_CHECKPOINT_ROOT))
    selected_set = set(selected_ckpt_ids)
    ckpts = [c for c in ckpts if c["checkpoint_id"] in selected_set]
    for variant in ckpts:
        inputs.append(checkpoint_path(mid, variant, config.WORKBENCH_CHECKPOINT_ROOT))
    # 2. The FashionIQ inputs this model's layout declares. `fashioniq_required_paths` already
    #    encodes the layout's captions, splits, correction dictionaries, resized-image and
    #    annotation requirements, so a model-specific requirement is picked up automatically
    #    instead of being re-listed (and forgotten) here.
    ds_root = dataset_root(args, config)
    layout_paths = fashioniq_required_paths(model_dict, ds_root)
    if layout_paths:
        inputs.extend(layout_paths)
    else:
        for category in ("dress", "shirt", "toptee"):
            inputs.append(ds_root / "image_splits" / f"split.{category}.val.json")
            inputs.append(ds_root / "captions" / f"cap.{category}.val.json")

    # 3. Preparation-contract inputs, which carry the source-required train annotations, caches
    #    and generated artifacts the layout check alone does not name.
    contract = preparation_contract_for_model(model_dict)
    if contract is not None:
        inputs.extend(preparation_paths(contract, ds_root, "raw_inputs"))
        # An `evaluator_cwd` artifact resolves against the model checkout; without the evaluator root
        # it would be dropped from this list, silently omitting a required input from the pipeline's
        # view of what the run reads.
        evaluator_root = (config.WORKBENCH_THIRD_PARTY_ROOT / model_dict["source_dir"]
                          if model_dict.get("source_dir") else None)
        inputs.extend(preparation_paths(contract, ds_root, "generated_artifacts",
                                        evaluator_root=evaluator_root))

    # 4. Registered auxiliary assets, including the ones placed inside the model's own checkout
    #    (CSMCIR's COT captions, for example). These are evaluation inputs, not decoration.
    for asset in auxiliary_assets_for_model(mid):
        if asset.get("required_for") != "official_evaluation":
            continue
        inputs.append(auxiliary_destination(asset, config.WORKBENCH_THIRD_PARTY_ROOT, ds_root))


    # 4b. External pretrained runtime assets declared as required_for_evaluation
    for asset_status in external_asset_statuses(model_dict):
        raw_path = asset_status.get("path")
        if raw_path:
            asset_path = Path(raw_path)
            if asset_path.exists():
                inputs.append(asset_path)
    # 5. The evaluator implementation itself: a changed script changes the result.
    from workbench.backend.fashioniq_layout import canonical_dataset_link_path

    source_dir = model_dict.get("source_dir")
    if source_dir:
        inputs.append(config.WORKBENCH_THIRD_PARTY_ROOT / source_dir)
    link_path = canonical_dataset_link_path(model_dict, config.WORKBENCH_THIRD_PARTY_ROOT)
    if link_path:
        inputs.append(link_path)

    # 6. The preprocessing recipe is not a file to stat: it is a derived identity, folded into the
    #    fingerprint below by `_stage_input_fingerprint_evidence`. Requiring a written recipe
    #    artifact here would make every run weak before preparation had ever run.

    # De-duplicate while preserving order: identical paths twice add no evidence.
    seen: set[str] = set()
    unique: list[Path] = []
    for path in inputs:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return tuple(unique)


def _bundle_anchor_for_selection(model_dict: dict[str, Any] | None, selected: Sequence[str] | set[str] | None) -> str | None:
    """Return bundle_id when selection names the bundle or all its members, else None."""
    if not model_dict or not selected:
        return None
    selected_set = {selected} if isinstance(selected, str) else set(selected)
    for bundle in model_dict.get("checkpoint_bundles") or []:
        bundle_id = bundle.get("bundle_id")
        if bundle_id and selected_set == {bundle_id}:
            return bundle_id
        required = set(bundle.get("required_checkpoint_ids") or [])
        if required and selected_set == required:
            return bundle_id
    return None

def evaluated_checkpoint_ids(model_dict: dict, args: argparse.Namespace,
                             checkpoint_root: Path | None = None) -> list[str]:
    """The checkpoint ids evaluate_models.py will actually evaluate for this model and args.

    `checkpoint_root` is accepted for callers that already hold a configuration; selection here is
    registry-fact based, so it does not change the answer. Kept in the signature because the two
    callers that resolve a selection from a workspace pass it, and a second, subtly different
    resolver is what this helper exists to prevent.
    """
    selection = _checkpoint_ids(args, model_dict["model_id"], model_dict)
    if selection is not None:
        if not selection:
            return []
        anchor = _bundle_anchor_for_selection(model_dict, selection)
        if anchor:
            return [anchor]
        return list(selection)
    anchor = _bundle_anchor_for_selection(model_dict, None)
    if anchor:
        return [anchor]
    bundles = model_dict.get("checkpoint_bundles") or []
    if model_dict.get("model_id") == "limn" and bundles:
        return [str(bundles[0]["bundle_id"])]
    cid = resolved_checkpoint_id(model_dict, args)
    return [cid] if cid else []


def resolved_checkpoint_id(model_dict: dict, args: argparse.Namespace) -> str:
    """The checkpoint id evaluate_models.py will actually use for this model.

    LIMN evaluates as one bundle candidate keyed by its bundle id, so the proof
    filename must use the bundle id, not the first category variant.
    """
    bundles = model_dict.get("checkpoint_bundles") or []
    selections = getattr(args, "checkpoint_ids", None)
    if isinstance(selections, dict) and model_dict["model_id"] in selections:
        selected = selections[model_dict["model_id"]]
        anchor = _bundle_anchor_for_selection(model_dict, selected)
        if anchor:
            return anchor
        values = _checkpoint_ids(args, model_dict["model_id"], model_dict)
        if values:
            return str(values[0])
        return ""
    explicit = getattr(args, "checkpoint", None)
    if explicit is not None:
        anchor = _bundle_anchor_for_selection(model_dict, explicit)
        if anchor:
            return anchor
        values = _checkpoint_ids(args, model_dict["model_id"], model_dict)
        if values:
            return str(values[0])
        return ""
    if model_dict["model_id"] == "limn" and bundles:
        return str(bundles[0]["bundle_id"])
    variants = model_dict.get("checkpoint_variants") or []
    return str(variants[0]["checkpoint_id"]) if variants else ""

def resolved_checkpoint_file(model_dict: dict, args: argparse.Namespace, config: WorkbenchConfig,
                             checkpoint_id: str | None = None) -> Path:
    """The primary checkpoint file the evaluator will be handed for this selection."""
    from workbench.backend.registry import checkpoint_path

    mid = model_dict["model_id"]
    variants = {variant["checkpoint_id"]: variant for variant in model_dict.get("checkpoint_variants") or []}
    ckpt_id = checkpoint_id or resolved_checkpoint_id(model_dict, args)
    variant = variants.get(ckpt_id)
    if variant is None:
        for bundle in model_dict.get("checkpoint_bundles") or []:
            if bundle["bundle_id"] == ckpt_id:
                variant = variants.get(bundle["required_checkpoint_ids"][0])
                break
    if variant is None and variants:
        variant = next(iter(variants.values()))
    try:
        return checkpoint_path(mid, variant, config.WORKBENCH_CHECKPOINT_ROOT) if variant else Path(".")
    except (KeyError, TypeError):
        return Path(".")


def evaluation_evidence(model_dict: dict, args: argparse.Namespace, config: WorkbenchConfig,
                        checkpoint_id: str | None = None):
    """Run-bound completion evidence for one evaluation stage."""
    from workbench.scripts.evaluate_models import completion_validator

    mid = model_dict["model_id"]
    protocol = getattr(args, "protocol", None) or model_dict.get("native_protocol")
    pin = model_dict.get("upstream_commit_sha")

    if checkpoint_id is not None:
        cids = [checkpoint_id]
    else:
        cids = evaluated_checkpoint_ids(model_dict, args, config.WORKBENCH_CHECKPOINT_ROOT)

    if not cids:
        cid = resolved_checkpoint_id(model_dict, args)
        cids = [cid] if cid else []

    if len(cids) <= 1:
        cid = cids[0] if cids else ""
        val = completion_validator(mid, cid, protocol, pin, config,
                                   resolved_checkpoint_file(model_dict, args, config, checkpoint_id=cid))
        setattr(val, "checkpoint_ids", tuple(cids))
        return val

    validators = {
        cid: completion_validator(mid, cid, protocol, pin, config,
                                  resolved_checkpoint_file(model_dict, args, config, checkpoint_id=cid))
        for cid in cids
    }

    def _validate_all(record: dict | None = None) -> tuple[bool, str | None, dict | None]:
        if record is None:
            return False, "no evaluation invocation recorded", None
        base_run_id = record.get("run_id")
        if not base_run_id:
            return False, "recorded evaluation has no invocation run id", None
        details: dict[str, Any] = {}
        for cid, val in validators.items():
            sub_run_id = f"{base_run_id}_{mid}_{cid}"
            ok, reason, item_detail = val({"run_id": sub_run_id})
            if not ok:
                return False, f"checkpoint {cid}: {reason}", None
            details[cid] = item_detail
        return True, None, {"checkpoints": details, "run_id": base_run_id}

    setattr(_validate_all, "checkpoint_ids", tuple(cids))
    return _validate_all


def model_eval_outputs(model_dict: dict, args: argparse.Namespace, config: WorkbenchConfig,
                       checkpoint_id: str | None = None) -> tuple[Path, ...]:
    """Declared outputs are informational only; run-bound proof is validated separately."""
    mid = model_dict["model_id"]
    if checkpoint_id is not None:
        cids = [checkpoint_id]
    else:
        cids = evaluated_checkpoint_ids(model_dict, args, config.WORKBENCH_CHECKPOINT_ROOT)
    return tuple(_artifacts_root(config) / "reports" / f"{mid}_{cid}_latest_successful.json" for cid in cids)
def real_stages(args: argparse.Namespace, config: WorkbenchConfig) -> list[Stage]:
    from workbench.backend.fashioniq_layout import canonical_dataset_link_path
    from workbench.backend.registry import checkpoint_path, model_by_id
    model_obj = model_by_id(args.model) if args.model else None
    sdir = model_obj.get("source_dir") if model_obj else None
    sync_outputs = (config.WORKBENCH_THIRD_PARTY_ROOT / sdir,) if sdir else ()

    sync = (
        run_stage(
            "sync",
            command("sync_upstreams.py", *selection(args), "--output-root", config.WORKBENCH_THIRD_PARTY_ROOT),
            model_id=args.model,
            output_paths=sync_outputs,
            required_capability="allow_network",
        )
        if args.sync_sources
        else Stage("sync", skipped="pass --sync-sources")
    )
    link = layout_prepare(args, config)
    # The operator may name a bundle; the acquisition and verification stages then need its members.
    cli_selection = _checkpoint_ids(args, args.model, model_obj)
    forced_download_args = _checkpoint_args(args, args.model, model_obj)
    download_args = [*selection(args), *forced_download_args,
                     "--output-root", config.WORKBENCH_CHECKPOINT_ROOT]
    ckpt_outputs = ()
    if model_obj:
        ckpts = model_obj.get("checkpoint_variants", [])
        if cli_selection is not None:
            ckpts = [c for c in ckpts if c["checkpoint_id"] in set(cli_selection)]
        elif forced_download_args:
            forced_ids = {forced_download_args[i + 1] for i in range(0, len(forced_download_args), 2) if forced_download_args[i] == "--checkpoint"}
            if forced_ids:
                ckpts = [c for c in ckpts if c["checkpoint_id"] in forced_ids]
        ckpt_outputs = tuple(checkpoint_path(args.model, c, config.WORKBENCH_CHECKPOINT_ROOT) for c in ckpts)
    if not args.download_checkpoints:
        checkpoints = Stage("checkpoint", skipped="pass --download-checkpoints")
    elif not ckpt_outputs:
        checkpoints = Stage("checkpoint", skipped="no registered checkpoint output for this selection")
    else:
        checkpoints = run_stage(
            "checkpoint",
            command("download_checkpoints.py", *download_args),
            model_id=args.model,
            checkpoint_id=cli_selection[0] if cli_selection is not None and len(cli_selection) == 1 else None,
            output_paths=ckpt_outputs,
            required_capabilities={"allow_large_downloads", "allow_network"},
        )
    selected_auxiliary_models = [args.model] if args.model in auxiliary_model_ids() else ([] if args.model else sorted(auxiliary_model_ids()))
    if args.download_auxiliary_assets and selected_auxiliary_models:
        auxiliary_args = ["--model", selected_auxiliary_models[0]] if args.model else ["--all"]
        if getattr(args, "stage_caption_aliases", False):
            auxiliary_args.append("--stage-caption-aliases")
        auxiliary = run_stage("auxiliary-assets", command("download_auxiliary_assets.py", *auxiliary_args), model_id=args.model, receipt=True, required_capabilities={"allow_network", "allow_large_downloads"})
    elif args.download_auxiliary_assets:
        auxiliary = Stage("auxiliary-assets", skipped="selected model has no registered auxiliary assets")
    else:
        auxiliary = Stage("auxiliary-assets", skipped="pass --download-auxiliary-assets")
    final_doctor_args: list[str] = ["--scope", "real", "--dataset-root", str(dataset_root(args, config))]
    if getattr(args, "verify_gpu", False):
        # This preflight runs immediately before the evaluation stage that carries
        # --verify-gpu, so a GPU whose authorized probe has not run yet must not stop the
        # workflow here: the probe is the next thing to happen, and a failed probe still
        # stops it, because the evaluation stage consumes a real capability.
        final_doctor_args.append("--allow-gpu-pending")
    if args.model:
        final_doctor_args.extend(_checkpoint_args(args, args.model, model_obj))
        final_doctor_args.extend(["--model", args.model])
        strict_runtime = run_stage("runtime-preflight", command("doctor.py", *final_doctor_args), model_id=args.model, dependencies=("sync", "checkpoint")) if args.evaluate else Stage("runtime-preflight", skipped="pass --evaluate")
    else:
        strict_runtime = Stage("runtime-preflight", skipped="per-model strict guards run inside --all-runnable evaluation")
    if args.evaluate:
        evaluation_args = list(selection(args, runnable=True))
        link_path = canonical_dataset_link_path(model_obj, config.WORKBENCH_THIRD_PARTY_ROOT) if model_obj else None
        if link_path:
            evaluation_args.extend([
                "--dataset-root", str(link_path),
                "--canonical-dataset-root", str(dataset_root(args, config)),
            ])
        else:
            evaluation_args.extend([
                "--dataset-root", str(dataset_root(args, config)),
            ])
        if model_obj:
            evaluation_args.extend(_checkpoint_args(args, args.model, model_obj))
        elif args.checkpoint:
            if isinstance(args.checkpoint, (list, tuple, set)):
                for cp in args.checkpoint:
                    evaluation_args.extend(["--checkpoint", str(cp)])
            else:
                evaluation_args.extend(["--checkpoint", str(args.checkpoint)])
        if args.protocol:
            evaluation_args.extend(["--protocol", args.protocol])
        evaluation_args.extend(["--top-k", str(args.top_k)])
        if getattr(args, "verify_gpu", False):
            evaluation_args.append("--verify-gpu")
        if args.continue_on_error:
            evaluation_args.append("--continue-on-error")
        eval_selection = evaluated_checkpoint_ids(model_obj, args, config.WORKBENCH_CHECKPOINT_ROOT) if model_obj else None
        if eval_selection is not None and not eval_selection:
            evaluation = Stage(
                "evaluation",
                skipped=f"refusing evaluation run: checkpoint selection for {args.model} is explicitly empty",
                model_id=args.model,
                checkpoint_selection=(),
            )
        elif model_obj is None:
            # `--all-runnable` evaluates whatever the registry currently allows, so no
            # single model owns the stage: there is no run identity to bind a completion
            # proof to and therefore no output contract that could ever be satisfied.
            # Rather than execute evaluators and then report a guaranteed failure, the
            # stage is planned as skipped and the operator is told how to run it per model
            # so each result gets a run-bound proof.
            evaluation = Stage(
                "evaluation",
                skipped=("--all-runnable evaluates per model; pass --model MODEL_ID so the "
                         "run gets a run-bound completion proof"),
            )
        else:
            eval_inputs = model_eval_inputs(model_obj, args, config, checkpoint_ids=eval_selection) if model_obj else ()
            eval_outputs = model_eval_outputs(model_obj, args, config) if model_obj else ()
            evaluation = run_stage(
                "evaluation",
                command("evaluate_models.py", *evaluation_args),
                model_id=args.model,
                checkpoint_id=resolved_checkpoint_id(model_obj, args),
                protocol=args.protocol,
                checkpoint_policy=getattr(args, "checkpoint_policy", None),
                checkpoint_selection=tuple(eval_selection or ()),
                checkpoint_overrides=tuple(sorted(
                    (mid, tuple(ids)) for mid, ids in getattr(args, "checkpoint_overrides", {}).items())),
                dependencies=("runtime-preflight",),
                input_paths=eval_inputs,
                output_paths=eval_outputs,
                required_capabilities=_evaluation_capabilities(model_obj, config),
                run_id=new_run_id(),
                evidence=evaluation_evidence(model_obj, args, config),
            )
    else:
        evaluation = Stage("evaluation", skipped="pass --evaluate")
    validation_index = (
        run_stage(
            "validation-index",
            command("validate_results.py", "--root", config.WORKBENCH_RESULTS_ROOT, "--strict-real"),
            command("rebuild_index.py", "--results-root", config.WORKBENCH_RESULTS_ROOT,
                "--database", database_path(config)),
            dependencies=("evaluation",),
            input_paths=(config.WORKBENCH_RESULTS_ROOT,),
            output_paths=(database_path(config),),
            required_capabilities={"allow_index_write"},
        )
        if args.rebuild_index
        else Stage("validation-index", skipped="pass --rebuild-index after schema-v2 result JSON exists")
    )
    serve = run_stage("serve", command("serve_workbench.py")) if args.serve else Stage("serve", skipped="pass --serve")
    early_doctor_args: list[str] = ["--scope", "workbench"]
    if args.serve:
        early_doctor_args.append("--serve")
    return [
        run_stage("doctor", command("doctor.py", *early_doctor_args)),
        dataset_check(args, config),
        sync,
        link,
        checkpoints,
        auxiliary,
        strict_runtime,
        evaluation,
        validation_index,
        serve,
    ]

def probe_gpu_capability(model_id: str, config: WorkbenchConfig | None = None) -> int:
    from workbench.backend.operator_config import resolve_config
    from workbench.backend.registry import model_by_id
    from workbench.backend import runtime

    cfg = config or resolve_config()
    model = model_by_id(model_id)
    gpu = (model.get("environment") or {}).get("cuda")
    gpu_required = gpu is not None and str(gpu).upper() != "UNKNOWN"
    if not gpu_required:
        print(f"[SKIP] {model_id} declares no CUDA requirement")
        return 0

    try:
        report = runtime.verify_environment(model, probe=True)
        cached = runtime._cached_gpu_capability(model, report)
        if cached is not None and cached.get("capability_verified"):
            print(f"[REUSE] valid GPU capability exists for {model_id}: {cached.get('capability_key')}")
            return 0
    except Exception:
        pass

    print(f"[PROBE] probing GPU capability for {model_id}...")
    try:
        capability = runtime.verify_gpu_runtime(
            model, cfg, authorized=True, require_cuda=True, force=False
        )
        print(f"[OK] GPU capability verified and recorded for {model_id}: "
              f"cuda={capability.get('cuda_available')} verified={capability.get('capability_verified')} "
              f"compute_capability={capability.get('compute_capability')}")
        return 0
    except (runtime.GpuVerificationError, RuntimeError, Exception) as exc:
        print(f"[BLOCKED] GPU verification failed: {exc}", file=sys.stderr)
        return 1


def run_validation_index(config: WorkbenchConfig | None = None) -> int:
    from workbench.backend.operator_config import resolve_config
    from workbench.backend.index import database_path, load_runs, rebuild_index
    from workbench.scripts.validate_results import validate_root

    cfg = config or resolve_config()
    db = database_path(cfg)
    results_root = cfg.WORKBENCH_RESULTS_ROOT

    try:
        runs = load_runs(results_root)
    except Exception as error:
        print(f"[WARN] validation-index: error loading indexable runs: {error}", file=sys.stderr)
        runs = []

    # An aggregate-only evaluation legitimately produces no schema-v2 per-query run. That is not a
    # reason to fail the stage: the stage's declared output is the derived database, and an empty
    # database is the honest result of indexing nothing. Skipping the rebuild instead left the
    # declared output missing, which the output contract then reported as a blocked stage.
    if not runs:
        try:
            count = rebuild_index(results_root, db, config=cfg)
        except Exception as error:
            print(f"[BLOCKED] index rebuild failed: {error}", file=sys.stderr)
            return 1
        print(f"[OK] no schema-v2 indexable runs found; wrote an empty index ({count} runs) to {db}")
        return 0

    valid = validate_root(results_root, strict_real=True)
    if not valid:
        print("[BLOCKED] validation failed for indexable runs", file=sys.stderr)
        return 1

    try:
        count = rebuild_index(results_root, db, config=cfg)
        print(f"[OK] indexed {count} runs into {db}")
        return 0
    except Exception as error:
        print(f"[BLOCKED] index rebuild failed: {error}", file=sys.stderr)
        return 1


def _expand_selection(model: dict[str, Any] | None, values: list[str]) -> list[str]:
    """Expand any declared bundle id into the checkpoint ids it requires.

    The evaluator already treats naming a bundle as naming its members, so the surrounding stages
    must agree: given `base_iter0_all_categories` the acquisition and verification stages have to
    plan all three category files, not look for a variant that does not exist under that name.
    """
    if not model or not values:
        return list(values)
    bundles = {bundle["bundle_id"]: list(bundle.get("required_checkpoint_ids") or [])
               for bundle in (model.get("checkpoint_bundles") or [])}
    expanded: list[str] = []
    for value in values:
        expanded.extend(bundles.get(value) or [value])
    return list(dict.fromkeys(expanded))


def _dataset_link_ready(model: dict[str, Any] | None, config: WorkbenchConfig) -> bool:
    """True when the model's declared fixed dataset link already resolves to the configured root.

    An evaluation whose link is already correct has nothing to prepare, so it must not demand
    `allow_preparation`: that capability authorizes creating the link, not using it. Requiring it
    unconditionally refused entirely-authorized GPU evaluations on a prepared machine.
    """
    from workbench.backend.fashioniq_layout import canonical_dataset_link_path

    if model is None:
        return False
    link = canonical_dataset_link_path(model, config.WORKBENCH_THIRD_PARTY_ROOT)
    if not link:
        return False
    try:
        return link.is_symlink() and link.resolve() == config.FASHIONIQ_ROOT.resolve()
    except OSError:
        return False


def _evaluation_capabilities(model: dict[str, Any] | None, config: WorkbenchConfig) -> set[str]:
    """Capabilities one model's evaluation stage must be granted.

    Evaluation itself is GPU work. It also writes under the results root, which is the preparation
    boundary, so that capability is required only when the model's declared link still needs
    creating.
    """
    capabilities = {"allow_gpu_eval"}
    if not _dataset_link_ready(model, config):
        capabilities.add("allow_preparation")
    return capabilities


def _checkpoint_ids(args: argparse.Namespace, model_id: str, model: dict[str, Any] | None = None) -> list[str] | None:
    """Return expanded IDs, or None when no selection was supplied."""
    resolved = getattr(args, "checkpoint_ids", None)
    if isinstance(resolved, dict) and model_id in resolved:
        raw = resolved[model_id]
        items = list(raw) if isinstance(raw, (list, tuple, set)) else [raw]
        return _expand_selection(model, items)
    single = getattr(args, "checkpoint", None)
    if single is not None:
        items = list(single) if isinstance(single, (list, tuple, set)) else [single]
        return _expand_selection(model, items)
    return None


def _checkpoint_args(args: argparse.Namespace, model_id: str, model: dict[str, Any] | None = None) -> list[str]:
    values = _checkpoint_ids(args, model_id, model)
    if values is None:
        model_dict = model
        if model_dict is None:
            try:
                registry = load_registry()
                model_dict = next((m for m in registry["models"] if m["model_id"] == model_id), None)
            except Exception:
                model_dict = None
        if model_dict is not None:
            cid = resolved_checkpoint_id(model_dict, args)
            if cid:
                expanded = _expand_selection(model_dict, [cid])
                argv: list[str] = []
                for val in expanded:
                    argv.extend(["--checkpoint", val])
                return argv
        return []
    argv: list[str] = []
    for value in values:
        argv.extend(["--checkpoint", value])
    return argv


def all_stages(args: argparse.Namespace, config: WorkbenchConfig) -> list[Stage]:
    from workbench.backend.fashioniq_layout import canonical_dataset_link_path
    from workbench.backend.registry import checkpoint_path, auxiliary_destination, auxiliary_assets_for_model
    registry = load_registry()
    models = registry["models"]
    # An explicit selection wins over "everything". `--model` names one model and
    # `selected_model_ids` names a set; without either, every registered model is planned, which
    # is the documented meaning of an unqualified `all`. A caller that has already resolved a
    # selection (the autopilot) must be able to hand it in, or a run configured for one model
    # would silently plan all sixteen.
    requested = getattr(args, "selected_model_ids", None)
    if requested is not None:
        wanted = list(requested)
        known = {m["model_id"] for m in models}
        unknown = [mid for mid in wanted if mid not in known]
        if unknown:
            raise ValueError(f"unknown --model value(s): {', '.join(sorted(unknown))}")
        selected_models = [m for m in models if m["model_id"] in set(wanted)]
    elif args.model:
        selected_models = [m for m in models if m["model_id"] == args.model]
    else:
        selected_models = models

    early_doctor = run_stage("doctor", command("doctor.py", "--scope", "workbench"))
    ds_check = dataset_check(args, config)
    stages = [early_doctor, ds_check]

    synced_sources = set()
    for m in selected_models:
        mid = m["model_id"]
        sdir = m.get("source_dir")
        sync_dep = f"sync:{mid}"
        sync_out = (config.WORKBENCH_THIRD_PARTY_ROOT / sdir,) if sdir else ()
        is_source_verified = False
        if sdir:
            source_dir = config.WORKBENCH_THIRD_PARTY_ROOT / sdir
            if source_dir.is_dir():
                from workbench.backend.runtime import source_clean_and_pinned
                try:
                    clean_pinned, _, _ = source_clean_and_pinned(m, source_dir)
                    is_source_verified = bool(clean_pinned)
                except Exception:
                    is_source_verified = False

        if sdir and sdir in synced_sources:
            first_m = next(m2["model_id"] for m2 in selected_models if m2.get("source_dir") == sdir)
            sync_dep = f"sync:{first_m}"
        else:
            if sdir:
                synced_sources.add(sdir)
            if is_source_verified:
                stages.append(run_stage(
                    "sync",
                    command("sync_upstreams.py", "--model", mid, "--verify-only", "--output-root", config.WORKBENCH_THIRD_PARTY_ROOT),
                    model_id=mid,
                    output_paths=sync_out,
                    receipt=True,
                    required_capabilities=frozenset(),
                ))
            else:
                stages.append(run_stage(
                    "sync",
                    command("sync_upstreams.py", "--model", mid, "--output-root", config.WORKBENCH_THIRD_PARTY_ROOT),
                    model_id=mid,
                    output_paths=sync_out,
                    receipt=True,
                    required_capabilities={"allow_network"},
                ))

        # Only an actual provisioning action is side-effecting; plain verification is read-only.
        will_create_env = bool(m.get("environment_required")) and getattr(args, "allow_env_install", False) and getattr(args, "apply", False)
        env_cmd = ["--model", mid]
        if will_create_env:
            # `required_capabilities` below names `allow_network`, so the runner refuses to execute
            # this command unless the capability was granted: passing the flag unconditionally is
            # safe, and keeping the command identical between runs is what lets a resume reuse the
            # stage instead of rerunning it after a permission change.
            env_cmd.extend(["--create", "--allow-env-install", "--allow-network"])
        stages.append(run_stage(
            "environment",
            command("manage_environment.py", *env_cmd),
            model_id=mid,
            dependencies=(sync_dep,),
            receipt=True,
            required_capabilities=({"allow_env_install", "allow_network"} if will_create_env else frozenset()),
        ))


        selected_checkpoints = _checkpoint_ids(args, mid, m)
        forced_args = _checkpoint_args(args, mid, m)
        ckpt_args = ["--model", mid, "--output-root", str(config.WORKBENCH_CHECKPOINT_ROOT),
                     *forced_args]
        ckpts = m.get("checkpoint_variants", [])
        if selected_checkpoints is not None:
            ckpts = [c for c in ckpts if c["checkpoint_id"] in set(selected_checkpoints)]
        elif forced_args:
            forced_ids = {forced_args[i + 1] for i in range(0, len(forced_args), 2) if forced_args[i] == "--checkpoint"}
            if forced_ids:
                ckpts = [c for c in ckpts if c["checkpoint_id"] in forced_ids]
        ckpt_outs = tuple(checkpoint_path(mid, c, config.WORKBENCH_CHECKPOINT_ROOT) for c in ckpts)
        if not ckpt_outs:
            # No registry checkpoint exists for this model: nothing to acquire.
            stages.append(Stage("checkpoint", skipped="model has no registered checkpoint variant"))
        else:
            from workbench.scripts.download_checkpoints import verify as verify_checkpoint
            all_ckpts_verified = bool(ckpt_outs) and all(
                p.is_file() and verify_checkpoint(m, c, p)
                for c, p in zip(ckpts, ckpt_outs)
            )
            if all_ckpts_verified:
                stages.append(run_stage(
                    "checkpoint",
                    command("download_checkpoints.py", "--verify-only", *ckpt_args),
                    model_id=mid,
                    checkpoint_id=selected_checkpoints[0] if selected_checkpoints is not None and len(selected_checkpoints) == 1 else None,
                    output_paths=ckpt_outs,
                    required_capabilities=frozenset(),
                ))
            else:
                stages.append(run_stage(
                    "checkpoint",
                    command("download_checkpoints.py", *ckpt_args),
                    model_id=mid,
                    checkpoint_id=selected_checkpoints[0] if selected_checkpoints is not None and len(selected_checkpoints) == 1 else None,
                    output_paths=ckpt_outs,
                    required_capabilities={"allow_large_downloads", "allow_network"},
                ))

        if mid in auxiliary_model_ids():
            aux_outs = tuple(auxiliary_destination(a, config.WORKBENCH_THIRD_PARTY_ROOT, config.FASHIONIQ_ROOT) for a in auxiliary_assets_for_model(mid))
            from workbench.scripts.download_auxiliary_assets import assets as get_aux_assets
            aux_items = get_aux_assets({mid}, config.WORKBENCH_THIRD_PARTY_ROOT, config.FASHIONIQ_ROOT)
            # ONE command, always, and let the script decide what to do with the state it finds. The
            # command string is part of the stage fingerprint (`_stage_input_fingerprint_evidence`),
            # so choosing `--verify-only` once everything was staged while a cold run chose the
            # downloading form meant the fingerprint CHANGED the moment the stage succeeded -- and a
            # changed fingerprint refuses a verified resume, so this stage re-ran on every
            # invocation. A script that has nothing to fetch already does nothing and costs no
            # capability: its own `main()` reports every asset and only acquires what is missing.
            # Whether the RUNNER may execute this stage is a per-invocation decision about work that
            # is actually owed, and it is deliberately not part of the command: the capability is
            # not fingerprinted, so an offline run over an already-complete asset set still runs the
            # same command with no permission required, while a run that must fetch something is
            # refused by name rather than silently skipping.
            from workbench.scripts.download_auxiliary_assets import (
                auxiliary_asset_needs_acquisition,
                load_manifest,
                manifest_path,
            )
            manifest = load_manifest(manifest_path(config.WORKBENCH_ARTIFACTS_ROOT))
            managed_digest = {entry["destination"]: entry.get("downloaded_sha256") for entry in manifest}

            owed = [item for item in aux_items if item.required and item.automatic_download_available
                    and auxiliary_asset_needs_acquisition(item, managed_manifest=managed_digest)]
            capability = {"allow_network", "allow_large_downloads"} if owed else frozenset()
            # Auxiliary asset acquisition runs under controller sys.executable via command().
            stages.append(run_stage(
                "auxiliary-assets",
                command("download_auxiliary_assets.py", "--model", mid),
                model_id=mid,
                output_paths=aux_outs,
                receipt=True,
                required_capabilities=capability,
            ))

        gpu = (m.get("environment") or {}).get("cuda")
        gpu_required = gpu is not None and str(gpu).upper() != "UNKNOWN"
        has_valid_gpu = False
        if gpu_required:
            try:
                from workbench.backend import runtime
                rep = runtime.verify_environment(m, probe=True)
                cached_cap = runtime._cached_gpu_capability(m, rep)
                if cached_cap is not None and cached_cap.get("capability_verified"):
                    has_valid_gpu = True
            except Exception:
                has_valid_gpu = False

        gpu_cmd = [sys.executable, "-c",
                   "import sys, workbench.scripts.pipeline as p; sys.exit(p.probe_gpu_capability(sys.argv[1]))",
                   mid]
        # Whether this stage needs `allow_gpu_eval` is only true when no verified capability is
        # cached. Kept in a variable because the PREPARATION stage later decides whether it may
        # depend on this stage, and that decision must use the exact requirement recorded here
        # rather than recompute (and possibly disagree with) it.
        gpu_verification_caps = ({"allow_gpu_eval"} if (gpu_required and not has_valid_gpu)
                                 else frozenset())
        stages.append(run_stage(
            "gpu-verification",
            gpu_cmd,
            model_id=mid,
            dependencies=(f"environment:{mid}",),
            required_capabilities=gpu_verification_caps,
            receipt=True,
        ))
        prep_link = canonical_dataset_link_path(m, config.WORKBENCH_THIRD_PARTY_ROOT)
        prep_owner = next((candidate["model_id"] for candidate in selected_models
                           if prep_link and canonical_dataset_link_path(
                               candidate, config.WORKBENCH_THIRD_PARTY_ROOT) == prep_link), None)
        target_root = dataset_root(args, config)
        is_link_valid = False
        if prep_link and prep_link.is_symlink():
            try:
                is_link_valid = prep_link.resolve() == target_root.resolve()
            except (OSError, RuntimeError):
                is_link_valid = False
        prep_cmd = ["--model", mid, "--dataset-root", str(target_root)]
        # Preparation owns two kinds of work: the fixed dataset link a pinned evaluator derives for
        # itself, and the artifacts a preparation contract declares as generable (a source-shipped
        # dictionary, a documented resize). Gating it on the link alone meant a model like LIMN,
        # which declares no link but does declare a resize and a dictionary, never had its
        # preparation attempted at all.
        from workbench.backend.registry import preparation_contract_for_model
        from workbench.scripts import prepare_model
        prep_contract = preparation_contract_for_model(m)
        executable_artifacts = [a for a in (prep_contract or {}).get("generated_artifacts", [])
                                if a.get("generator") in ("pinned_source_file", "fashioniq_resize_images",
                                                          prepare_model.TME_FEATURE_GENERATOR)]
        # A resize is complete only when its receipt exists, so the receipt -- not the directory --
        # is what the stage declares. Otherwise an interrupted run leaves a partially filled
        # directory, the directory check passes on any child file, and the stage resumes as done.
        # Anchored to `target_root`, the dataset the preparation command is actually given: a CLI
        # `--dataset-root` override would otherwise make the stage look for its outputs under the
        # configured root, fail permanently, and break resume.
        prep_dataset_root = target_root
        prep_artifacts: list[Path] = []
        for artifact in executable_artifacts:
            generator = artifact.get("generator")
            if generator == "fashioniq_resize_images":
                layout = (artifact.get("parameters") or {}).get("layout")
                if layout:
                    prep_artifacts.append(prepare_model.resize_receipt_path(
                        replace(config, FASHIONIQ_ROOT=prep_dataset_root), layout))
            elif generator == prepare_model.TME_FEATURE_GENERATOR:
                # An `evaluator_cwd` artifact lives inside the model CHECKOUT, not the dataset root,
                # and the files it names are 10.47 GiB of derived tensors. The stage therefore
                # declares the RECEIPT (written under the configured artifacts root) rather than the
                # directory: a directory check passes on any child file, so an interrupted generation
                # would resume as done. `tme_features_complete` below still verifies the files the
                # receipt names, so an emptied cache re-owes the stage.
                tme_checkout = config.WORKBENCH_THIRD_PARTY_ROOT / (m.get("source_dir") or "")
                prep_artifacts.append(prepare_model.tme_receipt_path(config, tme_checkout, prep_contract))
            else:
                prep_artifacts.extend(prep_dataset_root / relative for relative in artifact["paths"])
        owed_artifacts = tuple(p for p in prep_artifacts if not p.is_file())
        tme_completion = None
        if not owed_artifacts and prep_contract:
            # A receipt can outlive the files it promised, so a declared resize is only satisfied
            # when its receipt AND the images it recorded are still present. Without this a resumed
            # run would skip preparation over a dataset whose resized set had been emptied. The same
            # rule applies to the TME feature cache, whose receipt also binds the split files it was
            # derived from.
            if prepare_model.resize_is_complete(replace(config, FASHIONIQ_ROOT=prep_dataset_root),
                                                m, prep_contract) is False:
                owed_artifacts = prep_artifacts
            else:
                tme_completion = prepare_model.tme_features_complete(config, m, prep_contract)
            if tme_completion is False:
                owed_artifacts = prep_artifacts
        resume_refusal_reason = (
            "preparation self-check failed: TME feature receipt does not bind current source, "
            "gallery bytes, or resolved generation parameters"
            if tme_completion is False else None)
        link_is_owed = bool(prep_link) and prep_owner == mid and not is_link_valid

        # The command is built from the DECLARATION, never from whether work happens to be owed: the
        # command string is part of the stage fingerprint, so a command that gained `--execute` on
        # the second run invalidated its own stage and every dependent, rerunning the whole
        # evaluation instead of resuming it. Whether to execute is the per-run authorization
        # decision, and the runner already skips a completed stage whose fingerprint matches.
        declares_preparation = bool(prep_link) and prep_owner == mid or bool(executable_artifacts)
        # `--apply` is a property of the invocation, constant across a workflow's runs, whereas
        # whether work is owed changes as soon as the first run produces it. Keying the command on
        # `--apply` therefore gives the identical string on every run, which is what resume needs;
        # `--execute` on a read-only invocation would still be refused by the runner for lacking the
        # capability, but a plan run must not claim to execute anything at all.
        prep_deps = ["dataset", sync_dep]
        if declares_preparation and getattr(args, "apply", False):
            prep_cmd.extend(["--execute", "--allow-preparation"])
        # NO `checkpoint:{mid}` edge. No declared preparation generator CONSUMES a checkpoint:
        # every generator in `prepare_model.py` reads the dataset (resize, TME feature precompute)
        # or the pinned source, and none of them opens a weight file. An edge to a stage this stage
        # does not use is not a harmless ordering nicety -- it is a capability trap. `checkpoint`
        # carries `allow_large_downloads`/`allow_network`, so an operator who authorizes PREPARATION
        # without authorizing a download would have had their preparation refused as
        # SKIPPED_DEPENDENCY, i.e. a permission they deliberately withheld would cancel work they
        # explicitly authorized. Ordering is expressed by the append order in `all_stages`.
        #
        # `auxiliary-assets` IS a real prerequisite, but only for a model whose CONTRACT declares an
        # external asset as `required_for_preparation`: that is the declaration that says generation
        # cannot run before the asset is staged. Gating it on the asset being auxiliary-declared is
        # what keeps the edge honest in both directions -- a model with a preparation-required asset
        # that is NOT an auxiliary asset (dcnet's contract-declared ones) must not gain an edge to a
        # stage that does not exist for it, and a model whose auxiliary assets are only needed for
        # EVALUATION (TME's BLIP-2 and EVA-ViT-g, both `required_for_preparation: false`) must not
        # be cancelled because an evaluation-only download failed.
        prep_asset_owed = any(
            asset.get("required_for_preparation")
            for asset in (prep_contract or {}).get("external_assets", []))
        if prep_asset_owed and any(stage.name == "auxiliary-assets" and stage.model_id == mid
                                   for stage in stages):
            prep_deps.append(f"auxiliary-assets:{mid}")
        # The TME generator is a CUDA-only job that produces ~10.47 GiB of derived features, so it
        # carries its own capability rather than riding on `allow_preparation`. Whether the operator
        # granted it is a fact about THIS invocation, not about the model: the command string must
        # stay identical between runs (it is part of the stage fingerprint) and the DECISION is made
        # by the runner, which refuses a stage whose capability was not authorized and reports the
        # missing permission by name. The `environment:{mid}` edge orders the environment stage --
        # which resolves the pinned TME interpreter -- before preparation; whether that interpreter
        # really has CUDA is established by the handler at run time, not guessed here.
        tme_generator = bool(prep_contract) and any(
            artifact.get("generator") == prepare_model.TME_FEATURE_GENERATOR
            for artifact in prep_contract.get("generated_artifacts", []))
        if tme_generator:
            from workbench.backend.model_settings import setting as _setting

            tme_interpreter = (m.get("environment") or {}).get("python_env_var")
            if tme_interpreter and (_setting(tme_interpreter) or m.get("environment_required")):
                prep_deps.append(f"environment:{mid}")
            if getattr(args, "apply", False):
                # Same rule as `--execute` above: `--apply` is a property of the invocation, so the
                # string stays identical between the runs of one workflow; a read-only plan must
                # claim no capability and plan no execution at all.
                prep_cmd.append("--allow-gpu-preparation")
        if prep_owner and prep_owner != mid:
            prep_deps.append(f"preparation:{prep_owner}")
        # A model whose declared preparation is owed cannot be prepared without authorization, so an
        # `--apply` run that lacks it is refused as unauthorized rather than allowed to reach the
        # read-only verify step, exit nonzero, and surface as an unexplained stage failure.
        needs_preparation = declares_preparation and (link_is_owed or bool(owed_artifacts)) and getattr(args, "apply", False)
        prep_capabilities: set[str] = {"allow_preparation"} if needs_preparation else set()
        if tme_generator and getattr(args, "apply", False):
            # Declared for a model that owns the CUDA-only generator: the runner refuses the stage
            # when the operator did not authorize it and reports the exact missing permission, which
            # is a better outcome than silently skipping a declared preparation. A read-only plan
            # claims nothing, so an unauthorized dry run is still permitted to describe itself.
            prep_capabilities.add("allow_gpu_preparation")
        if "allow_gpu_preparation" in prep_capabilities and (
                not gpu_verification_caps or getattr(args, "allow_gpu_eval", False)):
            # The edge is only sound when the GPU stage cannot be refused for lack of a capability.
            # `BLOCKED_AUTHORIZATION_REQUIRED` counts as an UNUSABLE dependency, so depending on a
            # stage the operator did not authorize would cancel preparation with
            # SKIPPED_DEPENDENCY -- an ungranted EVALUATION permission silently killing
            # preparation the operator explicitly authorized with `--allow-preparation
            # --allow-gpu-preparation`. Two authorizations, both naming preparation, must be
            # enough; demanding `allow_gpu_eval` as a third would be a permission this stage does
            # not use. When the operator DID authorize it, the edge is kept and the verified
            # capability is ordered before the CUDA-only generation, which is the point.
            prep_deps.append(f"gpu-verification:{mid}")
        prep_outputs = (prep_link,) if prep_link and prep_owner == mid else tuple(prep_artifacts)
        stages.append(run_stage(
            "preparation",
            command("prepare_model.py", *prep_cmd),
            model_id=mid,
            dependencies=tuple(prep_deps),
            output_paths=tuple(prep_outputs),
            required_capabilities=frozenset(prep_capabilities),
            resume_refusal_reason=resume_refusal_reason,
        ))

        # The auxiliary assets are declared inputs of this model's official evaluation, so they
        # have to be resolved before the preflight can judge readiness. Without the edge the
        # preflight ran before the assets had been fetched or verified, and a model whose captions
        # were about to arrive was reported blocked while its evaluation was already scheduled.
        preflight_deps = (sync_dep, f"checkpoint:{mid}", f"preparation:{mid}", f"environment:{mid}",
                          f"gpu-verification:{mid}", f"auxiliary-assets:{mid}")
        preflight_args = ["--scope", "real", "--dataset-root", str(dataset_root(args, config)),
                          "--model", mid]
        if getattr(args, "verify_gpu", False):
            # The evaluation stage that follows carries --verify-gpu, so a GPU whose
            # authorized probe has not run yet is the state this workflow is about to change.
            preflight_args.append("--allow-gpu-pending")
        # The runtime preflight must inspect the SAME selection the run executes. Gating this on
        # `mid == args.model` dropped the selection on every autopilot invocation (which passes
        # `selected_model_ids` + `checkpoint_ids` and `model=None`), so doctor fell back to "inspect
        # every registered variant" and refused a run whose selected variant was the only one
        # installed: TME's unselected CIRR weight, HABIT's and INTENT's 50%/80% variants, and LIMN's
        # bundle members all blocked evaluations that had nothing to do with them. A missing SELECTED
        # variant must still block -- that is what the doctor stage checks -- so the rule is
        # "forward the resolved selection whenever there is one", not "forward it for one model".
        preflight_selection = _checkpoint_ids(args, mid, m)
        if preflight_selection is None:
            # No explicit selection means the run derives one: a declared bundle anchor, or the
            # model's own preference. The preflight inspects THAT derived set rather than every
            # registered variant -- and when the run resolves nothing at all (a policy that refuses
            # the model), nothing is forwarded and doctor reports the refusal from the same policy.
            preflight_selection = evaluated_checkpoint_ids(m, args, config.WORKBENCH_CHECKPOINT_ROOT) or None
        if preflight_selection:
            for value in preflight_selection:
                preflight_args.extend(["--checkpoint", value])
        if getattr(args, "checkpoint_policy", None):
            preflight_args.extend(["--checkpoint-policy", args.checkpoint_policy])
        stages.append(run_stage(
            "runtime-preflight",
            command("doctor.py", *preflight_args),
            model_id=mid,
            dependencies=preflight_deps,
        ))

        eval_args = ["--model", mid]
        link_path = canonical_dataset_link_path(m, config.WORKBENCH_THIRD_PARTY_ROOT)
        if link_path:
            eval_args.extend([
                "--dataset-root", str(link_path),
                "--canonical-dataset-root", str(dataset_root(args, config)),
            ])
        else:
            eval_args.extend([
                "--dataset-root", str(dataset_root(args, config)),
            ])
        eval_selection = evaluated_checkpoint_ids(m, args, config.WORKBENCH_CHECKPOINT_ROOT)
        if eval_selection is not None and not eval_selection:
            stages.append(Stage(
                "evaluation",
                skipped=f"refusing evaluation run: checkpoint selection for {mid} is explicitly empty",
                model_id=mid,
                checkpoint_selection=(),
            ))
            continue
        eval_args.extend(_checkpoint_args(args, mid, m))
        if args.protocol:
            eval_args.extend(["--protocol", args.protocol])
        eval_args.extend(["--top-k", str(args.top_k)])
        if getattr(args, "verify_gpu", False):
            eval_args.append("--verify-gpu")
        if args.continue_on_error:
            eval_args.append("--continue-on-error")

        eval_deps = (f"runtime-preflight:{mid}",)
        eval_ins = model_eval_inputs(m, args, config, checkpoint_ids=eval_selection)
        eval_outs = model_eval_outputs(m, args, config)
        stages.append(run_stage(
            "evaluation",
            command("evaluate_models.py", *eval_args),
            model_id=mid,
            checkpoint_id=resolved_checkpoint_id(m, args),
            protocol=args.protocol,
            checkpoint_policy=getattr(args, "checkpoint_policy", None),
            checkpoint_selection=tuple(eval_selection or ()),
            checkpoint_overrides=tuple(sorted(
                (override_mid, tuple(ids))
                for override_mid, ids in getattr(args, "checkpoint_overrides", {}).items())),
            dependencies=eval_deps,
            input_paths=eval_ins,
            output_paths=eval_outs,
            required_capabilities=_evaluation_capabilities(m, config),
            run_id=new_run_id(),
            evidence=evaluation_evidence(m, args, config),
        ))

    all_eval_deps = tuple(f"evaluation:{m['model_id']}" for m in selected_models)
    val_cmd = [sys.executable, "-c",
               "import sys, workbench.scripts.pipeline as p; sys.exit(p.run_validation_index())"]
    stages.append(run_stage(
        "validation-index",
        val_cmd,
        dependencies=all_eval_deps,
        input_paths=(config.WORKBENCH_RESULTS_ROOT,),
        output_paths=(database_path(config),),
        required_capabilities={"allow_index_write"},
    ))
    return stages
def setup_stages(args: argparse.Namespace, config: WorkbenchConfig) -> list[Stage]:
    all_s = all_stages(args, config)
    return [s for s in all_s if s.name in ("doctor", "dataset", "sync", "environment", "preparation", "checkpoint", "auxiliary-assets", "gpu-verification")]


def reproduce_stages(args: argparse.Namespace, config: WorkbenchConfig) -> list[Stage]:
    all_s = all_stages(args, config)
    return [s for s in all_s if s.name in ("gpu-verification", "runtime-preflight", "evaluation")]


def analyze_stages(args: argparse.Namespace, config: WorkbenchConfig) -> list[Stage]:
    """Analysis mode is the explicit opt-in to rebuild the derived index."""
    all_s = all_stages(args, config)
    selected = []
    for stage in all_s:
        if stage.name == "validation-index":
            selected.append(replace(stage, required_capabilities=frozenset({"allow_index_write"})))
    return selected

def stages_for(args: argparse.Namespace, config: WorkbenchConfig) -> list[Stage]:
    if args.mode == "mock":
        doctor_args = ["--scope", "workbench", *( ["--serve"] if args.serve else [])]
        stages = [
            run_stage("doctor", command("doctor.py", *doctor_args)),
            run_stage("load mock", command("load_mock_results.py", "--output-root", config.WORKBENCH_RESULTS_ROOT),
                      input_paths=(config.WORKBENCH_RESULTS_ROOT,),
                      receipt=True, required_capabilities={"allow_mock_write"}),
            run_stage("validate", command("validate_results.py", "--root", config.WORKBENCH_RESULTS_ROOT),
                      input_paths=(config.WORKBENCH_RESULTS_ROOT,),
                      receipt=True, required_capabilities={"allow_mock_write"}),
            run_stage("rebuild", command("rebuild_index.py", "--results-root", config.WORKBENCH_RESULTS_ROOT),
                      input_paths=(config.WORKBENCH_RESULTS_ROOT,),
                      receipt=True, required_capabilities={"allow_index_write"}),
        ]
        return [*stages, run_stage("serve", command("serve_workbench.py")) if args.serve else Stage("serve", skipped="pass --serve")]
    if args.mode == "prepare":
        return [dataset_check(args, config), layout_prepare(args, config)]
    if args.mode == "real":
        return real_stages(args, config)
    if args.mode == "serve":
        return [run_stage("serve", command("serve_workbench.py"))]
    if args.mode == "setup":
        return setup_stages(args, config)
    if args.mode == "reproduce":
        return reproduce_stages(args, config)
    if args.mode == "analyze":
        return analyze_stages(args, config)
    if args.mode == "all":
        return all_stages(args, config)
    if args.mode == "report":
        report_args = ["--scope", "real", "--json"]
        if args.model:
            report_args.extend(["--model", args.model])
        else:
            report_args.append("--all")
        return [run_stage("report", command("doctor.py", *report_args))]
    return [run_stage("doctor", command("doctor.py", *( ["--model", args.model] if args.model else [])))]


def run_command(argv: list[str]) -> int:
    try:
        from workbench.backend.registry import hf_hub_cache_root

        env = os.environ.copy()
        env["HF_HOME"] = str(hf_hub_cache_root())
        return subprocess.run(argv, cwd=REPOSITORY, env=env, check=False).returncode
    except OSError as error:
        print(f"[BLOCKED] cannot start {argv[1]}: {error}", file=sys.stderr)
        return 127


def run_stages(
    stages: list[Stage],
    *,
    dry_run: bool = False,
    continue_on_error: bool = False,
    resume: bool = False,
    force_stages: set[str] | None = None,
    state_path: Path | None = None,
    apply: bool = False,
    authorized_capabilities: set[str] | None = None,
    config: WorkbenchConfig | None = None,
    mode: str = "real",
    report_path: Path | None = None,
) -> int:
    cfg = config or resolve_config()
    if state_path is None:
        state_path = _artifacts_root(cfg) / "pipeline_state.json"
    failed = False
    force_set = set(force_stages or ())
    if force_set:
        known = {s.name for s in stages} | {stage_key(s) for s in stages} | {dependency_key(s) for s in stages}
        unknown = sorted(force_set - known)
        if unknown:
            raise WorkflowLockError(f"unknown --force-stage value(s): {', '.join(unknown)}")
    authorized = set(authorized_capabilities or ())
    lock_fd = None
    if not dry_run:
        lock_fd = acquire_workflow_lock(state_path)
    state = load_pipeline_state(state_path) if resume else {}
    stage_records = state.setdefault("stages", {})
    rerun_stages = set()
    failed_stages = set()
    completed_stages = set()
    planning_mode = dry_run or (mode in ("all", "setup", "reproduce") and not apply)
    if mode in ("all", "setup", "reproduce") and not apply and not dry_run:
        print("  [PLAN] planning mode by default; pass --apply with capability authorization flags to execute")

    def write_report() -> None:
        if report_path and not dry_run:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            workflow_report = {
                "mode": mode,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "success": not failed,
                "status": "SUCCESS" if not failed else ("PARTIAL" if any(s.get("status") == "COMPLETE" for s in stage_records.values()) else "FAILED"),
                "stages": stage_records,
            }
            temp_rep = report_path.with_suffix(f".tmp.{os.getpid()}")
            temp_rep.write_text(json.dumps(workflow_report, indent=2) + "\n", encoding="utf-8")
            temp_rep.replace(report_path)
    try:
        for number, stage in enumerate(stages, 1):
            skey = stage_key(stage)
            print(f"[{number}/{len(stages)}] {stage.name}")
            if not stage.commands:
                print(f"  [SKIP] {stage.skipped}")
                continue

            is_side_effecting = bool(stage.required_capabilities)
            stage_planning = planning_mode or (is_side_effecting and not apply)

            # Dependency execution prerequisite check. A dependency that is failing in
            # this sweep OR persisted as failing/blocked in the state file must block.
            def _dep_unusable(dep: str) -> bool:
                if dep in failed_stages:
                    return True
                # A dependency string may omit the checkpoint/protocol qualifier that the
                # recorded key carries. Qualify it with THIS stage's own model so the lookup
                # never matches a different model's stage.
                own = stage.model_id

                def _bad(key: str) -> bool:
                    if key in failed_stages:
                        return True
                    data = stage_records.get(key)
                    return isinstance(data, dict) and data.get("status") in _UNUSABLE_DEP_STATUSES

                # Exact match only: a bare edge name ("sync") must never match another
                # model's qualified key ("sync:othermodel").
                if _bad(dep):
                    return True
                # Same edge qualified by THIS stage's model (dep -> dep:<own>), including
                # any checkpoint/protocol variants of that same edge.
                if own and ":" not in dep:
                    qualified = f"{dep}:{own}"
                    if _bad(qualified):
                        return True
                    for recorded, data in stage_records.items():
                        if recorded.startswith(qualified + ":") and isinstance(data, dict) \
                                and data.get("status") in _UNUSABLE_DEP_STATUSES:
                            return True
                # A dependency that already names a model: allow its checkpoint variants.
                if ":" in dep:
                    for recorded, data in stage_records.items():
                        if recorded.startswith(dep + ":") and isinstance(data, dict) \
                                and data.get("status") in _UNUSABLE_DEP_STATUSES:
                            return True
                return False

            unmet_deps = [dep for dep in stage.dependencies if _dep_unusable(dep)]
            if unmet_deps:
                print(f"  [SKIP] {stage.name}: SKIPPED_DEPENDENCY: dependency {unmet_deps[0]} failed", file=sys.stderr)
                failed = True
                failed_stages.add(stage.name)
                failed_stages.add(dependency_key(stage))
                if not stage_planning:
                    stage_records[skey] = {
                        "stage_id": skey,
                        "name": stage.name,
                        "model_id": stage.model_id,
                        "status": "SKIPPED_DEPENDENCY",
                        "fingerprint": None,
                        "finished_at": datetime.now(timezone.utc).isoformat(),
                        "return_code": 125,
                    }
                    save_pipeline_state(state, state_path)
                continue

            # Explicit authorization check: every required capability must be granted.
            if is_side_effecting:
                missing_caps = sorted(stage.required_capabilities - authorized)
                if not apply and not dry_run:
                    flags = " ".join(f"--{c.replace('_', '-')}" for c in sorted(stage.required_capabilities))
                    print(f"  [PLAN] {stage.name}: side-effecting stage requires --apply and {flags} to execute")
                elif apply and missing_caps:
                    flags = " ".join(f"--{c.replace('_', '-')}" for c in missing_caps)
                    print(f"  [BLOCKED] {stage.name}: BLOCKED_AUTHORIZATION_REQUIRED: pass {flags}", file=sys.stderr)
                    failed = True
                    failed_stages.add(stage.name)
                    failed_stages.add(dependency_key(stage))
                    stage_records[skey] = {
                        "stage_id": skey,
                        "name": stage.name,
                        "model_id": stage.model_id,
                        "status": "BLOCKED_AUTHORIZATION_REQUIRED",
                        "fingerprint": None,
                        "finished_at": datetime.now(timezone.utc).isoformat(),
                        "return_code": 126,
                    }
                    if not dry_run:
                        save_pipeline_state(state, state_path)
                    if not continue_on_error:
                        return 1
                    continue

            fingerprint_info = stage_input_fingerprint_info(stage, cfg, state)
            fingerprint = fingerprint_info["digest"]

            # Check dependency invalidation
            dep_invalidated = any(dep in rerun_stages for dep in stage.dependencies)

            # Resume check
            can_skip = False
            resume_block_reason = stage.resume_refusal_reason
            if resume and not dep_invalidated and (not force_set or (stage.name not in force_set and skey not in force_set)):
                prev = stage_records.get(skey) or stage_records.get(stage.name)
                if prev and prev.get("status") == "COMPLETE" and not fingerprint_info["weak"]:
                    if prev.get("fingerprint") == fingerprint:
                        if not resume_block_reason and validate_stage_outputs(
                                stage, prev.get("output_fingerprints"), config=cfg, record=prev):
                            can_skip = True

            # Report every input that is *not* verified evidence, using the same vocabulary the
            # fingerprint itself used to decide weakness — otherwise a fully content-verified
            # directory tree would be listed as if it were weak.
            weak_categories = [
                f"{item['path']} ({item['category']})"
                for item in fingerprint_info["inputs"]
                if not _is_verified_entry(item["category"])
                and item["category"] not in VERIFIED_TREE_CATEGORIES
            ]
            source = fingerprint_info.get("source_provenance")
            if source and not _is_verified_entry(source.get("category")):
                provenance_line = f"{source.get('path', 'source provenance')} ({source['category']})"
                # The checkout is also a declared input; do not report the same path twice.
                if not any(line.startswith(str(source.get("path", ""))) for line in weak_categories):
                    weak_categories.append(provenance_line)
            if not resume_block_reason:
                resume_block_reason = (f"weak_input_fingerprint: {', '.join(weak_categories)}"
                                       if fingerprint_info["weak"] else None)
            if resume and not can_skip and resume_block_reason:
                print(f"  [INFO] resume refused: {resume_block_reason}")
            if can_skip:
                print(f"  [SKIP] resume: already completed with matching fingerprint and verified outputs")
                completed_stages.add(stage.name)
                completed_stages.add(skey)
                continue

            rerun_stages.add(stage.name)
            rerun_stages.add(skey)

            if not stage_planning:
                stage_records[skey] = {
                    "stage_id": skey,
                    "name": stage.name,
                    "model_id": stage.model_id,
                    "status": "RUNNING",
                    "fingerprint": fingerprint,
                    "input_fingerprint_evidence": fingerprint_info,
                    "resume_refused_reason": resume_block_reason,
                    "run_id": stage.run_id,
                    "started_at": datetime.now(timezone.utc).isoformat(),
                }
                save_pipeline_state(state, state_path)


            stage_failed = False
            for argv in stage.commands:
                if stage.run_id and stage.name == "evaluation" and "--run-id" not in argv:
                    argv = [*argv, "--run-id", stage.run_id]
                print(f"  [{'PLAN' if stage_planning else 'RUN'}] {shlex.join(argv)}")
                if stage.name == "evaluation":
                    print("  [INFO] official reproduction only; no instrumented ranking export")
                if stage_planning:
                    continue
                status = run_command(argv)
                if status != 0:
                    stage_failed = True
                    failed = True
                    failed_stages.add(stage.name)
                    failed_stages.add(dependency_key(stage))
                    print(f"  [BLOCKED] {stage.name} exited with status {status}", file=sys.stderr)
                    stage_records[skey].update({
                        "status": "FAILED",
                        "finished_at": datetime.now(timezone.utc).isoformat(),
                        "return_code": status,
                    })
                    save_pipeline_state(state, state_path)
                    break
                # Single-command stages can be output-validated per command; multi-command
                # stages validate once after all commands succeed (below).
                if stage.receipt and not stage.output_paths:
                    continue  # proof is written after all commands succeed (below)
                if len(stage.commands) == 1 and not validate_stage_outputs(stage, config=cfg, record=stage_records.get(skey)):
                    stage_failed = True
                    failed = True
                    failed_stages.add(stage.name)
                    failed_stages.add(dependency_key(stage))
                    print(f"  [BLOCKED] {stage.name}: declared outputs missing or invalid after exit 0", file=sys.stderr)
                    stage_records[skey].update({
                        "status": "FAILED",
                        "finished_at": datetime.now(timezone.utc).isoformat(),
                        "return_code": 0,
                        "failure_reason": "output_contract_not_satisfied",
                    })
                    save_pipeline_state(state, state_path)
                    break

            if stage_failed:
                if not continue_on_error:
                    return 1
                continue

            # Provisioning stages that produce no artifact file get a durable receipt
            # as their completion proof (validated on later resumes).
            if not stage_planning and stage.receipt and not stage.output_paths:
                write_stage_receipt(cfg, stage, return_code=0,
                                    fingerprint=stage_records.get(skey, {}).get("fingerprint"))

            # Output proof for multi-command stages.
            if not stage_planning and not validate_stage_outputs(stage, config=cfg, record=stage_records.get(skey)):
                failed = True
                failed_stages.add(stage.name)
                failed_stages.add(dependency_key(stage))
                stage_records[skey].update({
                    "status": "FAILED",
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "return_code": 0,
                    "failure_reason": "output_contract_not_satisfied",
                })
                save_pipeline_state(state, state_path)
                _, evidence_reason, _ = stage.evidence(stage_records.get(skey)) if stage.evidence else (False, None, None)
                detail = f": {evidence_reason}" if evidence_reason else ""
                print(f"  [BLOCKED] {stage.name}: declared outputs missing or invalid after exit 0{detail}", file=sys.stderr)
                if not continue_on_error:
                    return 1
                continue

            if not stage_planning:
                completed_stages.add(stage.name)
                completed_stages.add(skey)
                evidence_detail = None
                if stage.evidence is not None:
                    _, _, evidence_detail = stage.evidence(stage_records.get(skey))
                stage_records[skey].update({
                    "status": "COMPLETE",
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "return_code": 0,
                    "run_id": stage.run_id,
                    "evidence": evidence_detail,
                    "output_fingerprints": compute_stage_output_fingerprints(stage),
                })
                save_pipeline_state(state, state_path)
    finally:
        write_report()
        if lock_fd is not None:
            release_workflow_lock(lock_fd)

    return 1 if failed else 0


def parser_for() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("mock", "prepare", "real", "serve", "status", "setup", "reproduce", "analyze", "all", "report"))
    parser.add_argument("--dataset-root", type=Path, help="canonical FashionIQ dataset root")
    parser.add_argument("--sync-sources", action="store_true", help="sync selected official source repositories")
    parser.add_argument("--download-checkpoints", action="store_true", help="download selected registry checkpoints")
    parser.add_argument("--download-auxiliary-assets", action="store_true", help="obtain selected model auxiliary evaluation assets")
    parser.add_argument("--stage-caption-aliases", action="store_true",
                        help="with --download-auxiliary-assets, stage a published source-compatible caption at a "
                             "registered asset whose own source was never published (CSMCIR COT_ours2). The run is "
                             "then a labelled variant, not official parity: workbench/docs/COT_OURS2_AUDIT.md")
    parser.add_argument("--evaluate", action="store_true", help="run guarded official reproduction commands")
    parser.add_argument("--rebuild-index", action="store_true", help="validate and rebuild the derived result index")
    parser.add_argument("--serve", action="store_true", help="serve after pipeline stages")
    parser.add_argument("--dry-run", action="store_true", help="print ordered stage plan without subprocesses or mutations")
    parser.add_argument("--continue-on-error", action="store_true", help="continue pipeline after a failed stage")
    parser.add_argument("--resume", action="store_true", help="skip stages already completed with matching input fingerprints")
    parser.add_argument("--force-stage", action="append", default=[], help="force re-execution of a stage even when --resume is set")
    parser.add_argument("--apply", action="store_true", help="authorize execution of expensive operations")
    parser.add_argument("--allow-network", action="store_true", help="authorize network access (git clone/fetch, downloads)")
    parser.add_argument("--allow-large-downloads", action="store_true", help="authorize large checkpoint downloads")
    parser.add_argument("--allow-env-install", action="store_true", help="authorize environment creation and package installation")
    parser.add_argument("--allow-preparation", action="store_true", help="authorize model-specific preparation execution")
    parser.add_argument("--allow-gpu-eval", action="store_true", help="authorize model evaluation subprocess execution")
    parser.add_argument("--allow-gpu-preparation", action="store_true",
                        help="authorize the CUDA-only preparation generators (TME validation image features)")
    parser.add_argument("--verify-gpu", action="store_true",
                        help="run the authorized GPU capability probe before evaluating (GPU hosts only)")
    parser.add_argument("--allow-index-write", action="store_true", help="authorize rebuilding the derived DuckDB index")
    parser.add_argument("--model", help="registry model selection")
    parser.add_argument("--all-models", action="store_true", help="select all registry models")
    parser.add_argument("--protocol", help="official protocol selection for evaluation")
    parser.add_argument("--checkpoint", help="checkpoint selection")
    parser.add_argument("--top-k", type=int, default=200, help="requested saved retrieval depth")
    parser.add_argument("--report", type=Path, help="write machine-readable execution report JSON to path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = parser_for().parse_args(argv)
    if args.top_k < 1:
        parser_for().error("--top-k must be positive")
    if args.model and args.model not in {model["model_id"] for model in load_registry()["models"]}:
        parser_for().error(f"unknown model: {args.model}")
    config = resolve_config()
    if args.mode == "prepare" and args.dataset_root is None and not config.FASHIONIQ_ROOT:
        parser_for().error("prepare requires --dataset-root or FASHIONIQ_ROOT")
    force_stages = set(args.force_stage) if args.force_stage else None
    if args.all_models and args.model:
        parser_for().error("--all-models cannot be combined with --model")
    if args.mode in ("all", "setup", "reproduce") and not args.model and not args.all_models:
        parser_for().error(f"{args.mode} requires --model MODEL_ID or --all-models (explicit selection)")
    if force_stages:
        known = {s.name for s in stages_for(args, config)}
        unknown = sorted(force_stages - known)
        if unknown:
            parser_for().error(f"unknown --force-stage value(s): {', '.join(unknown)}")
    if args.dry_run and args.apply:
        parser_for().error("--dry-run cannot be combined with --apply")
    authorized_caps = set()
    if args.allow_network:
        authorized_caps.add("allow_network")
    if args.allow_large_downloads:
        authorized_caps.add("allow_large_downloads")
    if args.allow_env_install:
        authorized_caps.add("allow_env_install")
    if args.allow_preparation:
        authorized_caps.add("allow_preparation")
    if args.allow_gpu_eval:
        authorized_caps.add("allow_gpu_eval")
    if getattr(args, "allow_gpu_preparation", False):
        authorized_caps.add("allow_gpu_preparation")
    if args.allow_index_write:
        authorized_caps.add("allow_index_write")
    if args.mode == "mock":
        # `mock` is an explicit, self-contained development workflow; it carries its own
        # authorization for writing mock results and rebuilding the derived index only.
        authorized_caps.update({"allow_mock_write", "allow_index_write"})
    if args.mode == "analyze":
        # `analyze` exists solely to rebuild the derived index from validated results.
        authorized_caps.add("allow_index_write")
    # `mock` and `analyze` are explicit, self-contained operator workflows: they carry
    # their own authorization for the exact writes they perform. `--dry-run` still plans only.
    effective_apply = args.apply or args.mode in ("mock", "analyze")
    return run_stages(
        stages_for(args, config),
        dry_run=args.dry_run,
        continue_on_error=args.continue_on_error,
        resume=args.resume,
        force_stages=force_stages,
        apply=effective_apply,
        authorized_capabilities=authorized_caps,
        config=config,
        mode=args.mode,
        report_path=args.report,
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except WorkflowLockError as error:
        # A held lock is an operator situation, not a crash: one line, nonzero exit, no traceback.
        print(f"[BLOCKED] {error}", file=sys.stderr)
        raise SystemExit(3) from None
