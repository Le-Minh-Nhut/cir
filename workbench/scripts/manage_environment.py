#!/usr/bin/env python3
"""Inspect or create isolated model evaluation environments. Read-only without --create.

Never installs into the workbench backend environment. Requires --create --allow-env-install
to create a new environment from a source-backed specification. Never guesses packages or
synthesizes requirements not present in the pinned official source tree.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workbench.backend.operator_config import resolve_config
from workbench.backend.registry import load_registry


_ENV_SPEC_CANDIDATES = ("environment.yml", "environment.yaml", "requirements.txt")
_CONDA_COMMANDS = ("conda", "mamba", "micromamba")


def _find_env_spec(source: Path) -> Path | None:
    for name in _ENV_SPEC_CANDIDATES:
        candidate = source / name
        if candidate.is_file():
            return candidate
    return None


def _conda_available() -> str | None:
    for cmd in _CONDA_COMMANDS:
        if shutil.which(cmd):
            return cmd
    return None


def _interpreter_from_env_var(model: dict) -> str | None:
    env = model.get("environment") or {}
    var = env.get("python_env_var")
    if not var:
        return None
    return os.environ.get(var)


def _interpreter_status(model: dict, source_root: Path) -> dict:
    env = model.get("environment") or {}
    var = env.get("python_env_var")
    configured = _interpreter_from_env_var(model)
    if configured:
        path = Path(configured).expanduser().resolve()
        if path.is_file() and os.access(path, os.X_OK):
            return {"status": "READY", "interpreter": str(path), "variable": var}
        return {"status": "CONFIGURED_MISSING", "interpreter": str(path), "variable": var}
    return {"status": "UNCONFIGURED", "interpreter": None, "variable": var}


def status(model: dict, config) -> dict:
    model_id = model["model_id"]
    source_dir = model.get("source_dir")
    source = (config.WORKBENCH_THIRD_PARTY_ROOT / source_dir).resolve() if source_dir else None
    env_meta = model.get("environment") or {}
    spec = _find_env_spec(source) if source and source.is_dir() else None
    interp = _interpreter_status(model, source) if source else {"status": "NO_SOURCE", "interpreter": None, "variable": None}
    return {
        "model_id": model_id,
        "environment_required": bool(model.get("environment_required")),
        "source_synced": source is not None and source.is_dir(),
        "source_path": str(source) if source else None,
        "env_spec_found": spec is not None,
        "env_spec_path": str(spec) if spec else None,
        "env_spec_type": spec.name if spec else None,
        "interpreter_status": interp["status"],
        "interpreter": interp["interpreter"],
        "env_var": interp["variable"],
        "documented_python": env_meta.get("python"),
        "documented_pytorch": env_meta.get("pytorch"),
        "documented_cuda": env_meta.get("cuda"),
        "environment_confidence": env_meta.get("confidence"),
        "ready": interp["status"] == "READY",
    }


def print_status(s: dict) -> None:
    label = "OK" if s["ready"] else "BLOCKED"
    print(f"[{label}] {s['model_id']}: interpreter={s['interpreter_status']} env_var={s['env_var']}")
    if s["env_spec_found"]:
        print(f"  spec: {s['env_spec_path']} ({s['env_spec_type']})")
    else:
        print(f"  spec: not found in {s['source_path'] or 'source not synced'}")
    if s["interpreter"]:
        print(f"  interpreter: {s['interpreter']}")
    if s["documented_python"]:
        print(f"  documented: python={s['documented_python']} pytorch={s['documented_pytorch']} cuda={s['documented_cuda']}")


def create_env(model: dict, s: dict, dry_run: bool) -> bool:
    model_id = model["model_id"]
    if not s["source_synced"]:
        print(f"[BLOCKED] {model_id}: source not synced — sync first with sync_upstreams.py", file=sys.stderr)
        return False
    spec = s["env_spec_path"]
    spec_type = s["env_spec_type"]
    if not spec:
        print(f"[BLOCKED] {model_id}: no environment spec found in pinned source ({', '.join(_ENV_SPEC_CANDIDATES)})", file=sys.stderr)
        return False
    if spec_type in ("environment.yml", "environment.yaml"):
        conda_cmd = _conda_available()
        if not conda_cmd:
            print(f"[BLOCKED] {model_id}: no conda/mamba found; required for {spec}", file=sys.stderr)
            return False
        env_name = f"workbench-{model_id}"
        argv = [conda_cmd, "env", "create", "--name", env_name, "--file", spec, "--yes"]
        print(f"[RUN] {model_id}: {' '.join(argv)}")
        if dry_run:
            print(f"[OK] dry-run: would create conda env {env_name!r} from {spec}")
            return True
        result = subprocess.run(argv)
        if result.returncode != 0:
            print(f"[BLOCKED] {model_id}: conda env create failed", file=sys.stderr)
            return False
        print(f"[OK] {model_id}: environment created; activate and set {s['env_var'] or 'model env var'} to interpreter path")
        return True
    # requirements.txt — pip into a venv
    env_var = s["env_var"]
    if not env_var:
        print(f"[BLOCKED] {model_id}: no environment variable declared; cannot locate target interpreter", file=sys.stderr)
        return False
    configured = os.environ.get(env_var)
    if not configured:
        print(f"[BLOCKED] {model_id}: {env_var} not set; point it at a pre-created venv or interpreter", file=sys.stderr)
        return False
    interpreter = Path(configured).expanduser().resolve()
    if not interpreter.is_file():
        print(f"[BLOCKED] {model_id}: {env_var}={interpreter} not found", file=sys.stderr)
        return False
    argv = [str(interpreter), "-m", "pip", "install", "-r", spec]
    print(f"[RUN] {model_id}: {' '.join(argv)}")
    if dry_run:
        print(f"[OK] dry-run: would install {spec} into {interpreter}")
        return True
    result = subprocess.run(argv)
    if result.returncode != 0:
        print(f"[BLOCKED] {model_id}: pip install failed", file=sys.stderr)
        return False
    print(f"[OK] {model_id}: packages installed into {interpreter}")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="list all registry model environment statuses")
    group.add_argument("--model", help="inspect or create environment for one model")
    parser.add_argument("--create", action="store_true", help="create environment from source-backed spec")
    parser.add_argument("--allow-env-install", action="store_true",
                        help="authorize package installation; required with --create")
    parser.add_argument("--dry-run", action="store_true", help="show planned operations without changes")
    parser.add_argument("--json", action="store_true", help="emit machine-readable status")
    args = parser.parse_args(argv)

    if args.create and not args.allow_env_install:
        parser.error("--create requires --allow-env-install (explicit authorization)")

    config = resolve_config()
    registry = load_registry()
    models = registry["models"]
    if args.model:
        models = [m for m in models if m["model_id"] == args.model]
        if not models:
            parser.error(f"unknown model: {args.model}")

    statuses = [status(m, config) for m in models]

    if args.json:
        import json
        print(json.dumps(statuses if args.list else statuses[0], indent=2))
        return 0 if all(s["ready"] for s in statuses) else 1

    if args.list:
        for s in statuses:
            print_status(s)
        blocked = [s for s in statuses if s["environment_required"] and not s["ready"]]
        print(f"\n{len(blocked)} model(s) with required environment not configured.")
        return 0

    s = statuses[0]
    model = models[0]
    print_status(s)

    if args.create:
        ok = create_env(model, s, args.dry_run)
        return 0 if ok else 1

    return 0 if s["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
