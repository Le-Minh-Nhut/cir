#!/usr/bin/env python3
"""Inspect or pin official source repositories; never downloads checkpoints."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workbench.backend.operator_config import resolve_config

from workbench.backend.registry import load_registry


def git_output(destination: Path, *command: str) -> str | None:
    result = subprocess.run(
        ["git", *command], cwd=destination, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )
    return result.stdout.strip() if result.returncode == 0 else None


def local_state(destination: Path) -> tuple[str, str | None, bool | None]:
    if not destination.exists():
        return "missing", None, None
    head = git_output(destination, "rev-parse", "HEAD")
    if head is None:
        return "not a git repository", None, None
    # Bytecode is a build product, never a source change: use the same
    # classification the execution gate and doctor use, so the three cannot disagree.
    from workbench.backend.runtime import source_status

    entries = source_status(destination)
    if entries is None:
        return "unreadable", head, None
    dirty = bool(entries["tracked"] or entries["unauthorized"] or entries["invalid"])
    return "dirty" if dirty else "clean", head, dirty


def report(model: dict, state: str, head: str | None) -> None:
    print(
        f"{model['model_id']}: url={model.get('upstream_repo_url')} "
        f"source_dir={model.get('source_dir')} pin={model.get('upstream_commit_sha')} "
        f"local={state}" + (f" head={head}" if head else "")
    )


def destination_for(root: Path, model: dict) -> Path:
    source_dir = model.get("source_dir")
    if not isinstance(source_dir, str) or not source_dir:
        raise ValueError("source_dir must be a non-empty relative path")
    relative = Path(source_dir)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe source_dir: {source_dir}")
    root = root.resolve()
    destination = (root / relative).resolve()
    if destination == root or not destination.is_relative_to(root):
        raise ValueError(f"unsafe source_dir: {source_dir}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="list pinned repositories")
    group.add_argument("--model", help="sync one model")
    group.add_argument("--all", action="store_true", help="sync all source-available models")
    parser.add_argument("--fetch", action="store_true", help="fetch pinned commit before detached checkout")
    parser.add_argument("--update-existing", action="store_true", help="allow clean existing repositories to checkout pin")
    parser.add_argument("--dry-run", action="store_true", help="report planned operations without Git calls or changes")
    parser.add_argument("--verify-only", action="store_true", help="report local pin state without network or changes")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    if args.output_root is None:
        args.output_root = resolve_config().WORKBENCH_THIRD_PARTY_ROOT
    if args.verify_only and args.fetch:
        parser.error("--verify-only cannot be used with --fetch")
    models = load_registry()["models"]
    if args.model:
        models = [model for model in models if model["model_id"] == args.model]
        if not models:
            parser.error(f"unknown model: {args.model}")

    failed = False
    groups: dict[Path, list[dict]] = {}
    for model in models:
        if not model["source_available"]:
            report(model, "unavailable", None)
            print(f"[SKIP] {model['model_id']}: source unavailable")
            continue
        try:
            destination = destination_for(args.output_root, model)
        except (OSError, ValueError) as error:
            report(model, "unsafe source path", None)
            print(f"[BLOCKED] {model['model_id']}: {error}")
            failed = True
            continue
        groups.setdefault(destination, []).append(model)

    for destination, group_models in groups.items():
        model = group_models[0]
        pin = model.get("upstream_commit_sha")
        if any(item.get("upstream_commit_sha") != pin or item.get("upstream_repo_url") != model.get("upstream_repo_url") for item in group_models):
            for item in group_models:
                report(item, "conflicting source", None)
                print(f"[BLOCKED] {item['model_id']}: shared source has conflicting pin or URL")
            failed = True
            continue
        if args.dry_run:
            state, head, dirty = ("missing", None, None) if not destination.exists() else ("uninspected", None, None)
        else:
            state, head, dirty = local_state(destination)
        for item in group_models:
            report(item, state, head)
        names = ", ".join(item["model_id"] for item in group_models)
        if args.list:
            status = "pinned SHA present" if head == pin else "pinned SHA unavailable locally" if head is None else "local SHA differs from pinned"
            label = "OK" if head == pin else "INFO"
            for item in group_models:
                print(f"[{label}] {item['model_id']}: {status}")
            continue
        if head is not None and head != pin:
            for item in group_models:
                print(f"[WARN] {item['model_id']}: local SHA {head} differs from pinned {pin}")
        if dirty:
            for item in group_models:
                print(f"[BLOCKED] {item['model_id']}: repository is dirty; refusing to change it")
            failed = True
            continue
        if args.verify_only:
            for item in group_models:
                if head == pin:
                    print(f"[OK] {item['model_id']}: pinned SHA present")
                elif head is None:
                    print(f"[BLOCKED] {item['model_id']}: pinned SHA unavailable locally")
                else:
                    print(f"[BLOCKED] {item['model_id']}: local SHA differs from pinned")
            failed |= head != pin
            continue
        if args.dry_run:
            action = "clone and checkout" if state == "missing" else "checkout existing repository"
            print(f"[RUN] {names}: would {action} at {pin}")
            continue
        if state == "missing":
            staging = None
            try:
                destination.parent.mkdir(parents=True, exist_ok=True)
                staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.clone-", dir=destination.parent))
                staging.rmdir()
                subprocess.run(["git", "clone", "--no-checkout", model["upstream_repo_url"], str(staging)], check=True)
                if args.fetch:
                    subprocess.run(["git", "fetch", "origin", pin], cwd=staging, check=True)
                subprocess.run(["git", "checkout", "--detach", pin], cwd=staging, check=True)
                actual = git_output(staging, "rev-parse", "HEAD")
                if actual != pin:
                    raise RuntimeError(f"checked out {actual}, expected {pin}")
                staging.rename(destination)
            except (OSError, subprocess.CalledProcessError, RuntimeError) as error:
                if staging is not None:
                    shutil.rmtree(staging, ignore_errors=True)
                for item in group_models:
                    print(f"[BLOCKED] {item['model_id']}: clone/pin failed: {error}")
                failed = True
                continue
        elif state != "clean":
            for item in group_models:
                print(f"[BLOCKED] {item['model_id']}: {state}")
            failed = True
            continue
        elif not args.update_existing:
            for item in group_models:
                print(f"[SKIP] {item['model_id']}: existing repository needs --update-existing")
            continue
        else:
            try:
                if args.fetch:
                    print(f"[RUN] {names}: fetching pinned SHA")
                    subprocess.run(["git", "fetch", "origin", pin], cwd=destination, check=True)
                print(f"[RUN] {names}: checking out pinned SHA")
                subprocess.run(["git", "checkout", "--detach", pin], cwd=destination, check=True)
            except (OSError, subprocess.CalledProcessError) as error:
                for item in group_models:
                    print(f"[BLOCKED] {item['model_id']}: Git operation failed: {error}")
                failed = True
                continue
        actual = git_output(destination, "rev-parse", "HEAD")
        if actual != pin:
            for item in group_models:
                print(f"[BLOCKED] {item['model_id']}: checked out {actual}, expected {pin}")
            failed = True
        else:
            for item in group_models:
                print(f"[OK] {item['model_id']}: pinned SHA checked out")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
