#!/usr/bin/env python3
"""Clone pinned official source repositories; never downloads checkpoints."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workbench.backend.registry import ROOT, load_registry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="list pinned repositories")
    group.add_argument("--model", help="sync one model")
    group.add_argument("--all", action="store_true", help="sync all source-available models")
    parser.add_argument("--fetch", action="store_true", help="fetch refs before detached checkout")
    parser.add_argument("--output-root", type=Path, default=ROOT / "third_party")
    args = parser.parse_args()
    models = load_registry()["models"]
    if args.model:
        models = [model for model in models if model["model_id"] == args.model]
        if not models:
            parser.error(f"unknown model: {args.model}")
    for model in models:
        if not model["source_available"]:
            print(f"{model['model_id']}: source unavailable")
            continue
        print(f"{model['model_id']}: {model['upstream_repo_url']} @ {model['upstream_commit_sha']}")
        if args.list:
            continue
        destination = args.output_root / model["source_dir"]
        if not destination.exists():
            subprocess.run(["git", "clone", "--no-checkout", model["upstream_repo_url"], str(destination)], check=True)
        if args.fetch:
            subprocess.run(["git", "fetch", "origin"], cwd=destination, check=True)
        subprocess.run(["git", "checkout", "--detach", model["upstream_commit_sha"]], cwd=destination, check=True)
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=destination, text=True).strip()
        if head != model["upstream_commit_sha"]:
            raise RuntimeError(f"{model['model_id']}: checked out {head}, expected pin")


if __name__ == "__main__":
    main()
