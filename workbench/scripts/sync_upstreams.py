#!/usr/bin/env python3
"""Inspect or pin official source repositories; never downloads checkpoints."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workbench.backend.registry import ROOT, load_registry


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
    dirty = bool(git_output(destination, "status", "--porcelain"))
    return "dirty" if dirty else "clean", head, dirty


def report(model: dict, state: str, head: str | None) -> None:
    print(
        f"{model['model_id']}: url={model.get('upstream_repo_url')} "
        f"source_dir={model.get('source_dir')} pin={model.get('upstream_commit_sha')} "
        f"local={state}" + (f" head={head}" if head else "")
    )


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
    parser.add_argument("--output-root", type=Path, default=ROOT / "third_party")
    args = parser.parse_args()
    if args.verify_only and args.fetch:
        parser.error("--verify-only cannot be used with --fetch")
    models = load_registry()["models"]
    if args.model:
        models = [model for model in models if model["model_id"] == args.model]
        if not models:
            parser.error(f"unknown model: {args.model}")

    failed = False
    for model in models:
        if not model["source_available"]:
            report(model, "unavailable", None)
            print(f"[SKIP] {model['model_id']}: source unavailable")
            continue
        destination = args.output_root / model["source_dir"] if model.get("source_dir") else args.output_root
        if args.dry_run:
            state, head, dirty = ("missing", None, None) if not destination.exists() else ("uninspected", None, None)
        else:
            state, head, dirty = local_state(destination)
        report(model, state, head)
        pin = model["upstream_commit_sha"]
        if head is not None and head != pin:
            print(f"[WARN] {model['model_id']}: local SHA {head} differs from pinned {pin}")
            failed = args.verify_only or failed
        if dirty:
            print(f"[BLOCKED] {model['model_id']}: repository is dirty; refusing to change it")
            failed = True
            continue
        if args.list or args.verify_only:
            if head == pin:
                print(f"[OK] {model['model_id']}: pinned SHA present")
            elif head is None:
                print(f"[BLOCKED] {model['model_id']}: pinned SHA unavailable locally")
                failed = True
            continue
        if args.dry_run:
            action = "clone and checkout" if state == "missing" else "checkout existing repository"
            print(f"[RUN] {model['model_id']}: would {action} at {pin}")
            continue
        if state == "missing":
            print(f"[RUN] {model['model_id']}: cloning then checking out pinned SHA")
            subprocess.run(["git", "clone", "--no-checkout", model["upstream_repo_url"], str(destination)], check=True)
        elif state != "clean":
            print(f"[BLOCKED] {model['model_id']}: {state}")
            failed = True
            continue
        elif not args.update_existing:
            print(f"[SKIP] {model['model_id']}: existing repository needs --update-existing")
            continue
        if args.fetch:
            print(f"[RUN] {model['model_id']}: fetching pinned SHA")
            subprocess.run(["git", "fetch", "origin", pin], cwd=destination, check=True)
        print(f"[RUN] {model['model_id']}: checking out pinned SHA")
        subprocess.run(["git", "checkout", "--detach", pin], cwd=destination, check=True)
        actual = git_output(destination, "rev-parse", "HEAD")
        if actual != pin:
            print(f"[BLOCKED] {model['model_id']}: checked out {actual}, expected {pin}")
            failed = True
        else:
            print(f"[OK] {model['model_id']}: pinned SHA checked out")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
