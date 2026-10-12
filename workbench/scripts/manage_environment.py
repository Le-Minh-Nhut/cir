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
from workbench.backend.runtime import READY_TIERS, check_environment_isolation, verify_environment


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
        # Keep the configured invocation path exactly as given: resolving a venv's
        # bin/python can point at the shared system binary and destroy the context.
        path = Path(configured).expanduser()
        if not path.is_file() or not os.access(path, os.X_OK):
            return {"status": "CONFIGURED_MISSING", "interpreter": str(path), "variable": var}
        # This is an explicit operator inspection tool, so it pays for the probe and
        # reports the same tier definition doctor and guarded_plan use.
        report = verify_environment(model, probe=True)
        return {"status": report.status, "tier": report.tier, "interpreter": str(path), "variable": var,
                "verified": report.tier in READY_TIERS, "reason": report.reason,
                "packages": report.packages, "probe_executed": report.probe_executed}
    return {"status": "UNCONFIGURED", "interpreter": None, "variable": var, "tier": "UNCONFIGURED",
            "verified": False, "reason": None}

def _check_shared_assignment(model_id: str, config) -> dict | None:
    assignment_path = config.WORKBENCH_ARTIFACTS_ROOT / "environment_assignment.json"
    if assignment_path.is_file():
        try:
            from workbench.backend.environment_planner import load_environment_assignment
            data = load_environment_assignment(assignment_path)
            model_info = data.get("models", {}).get(model_id)
            if model_info and model_info.get("environment_id"):
                return model_info
        except Exception:
            pass
    plan_path = config.WORKBENCH_ARTIFACTS_ROOT / "environment_plan.json"
    if plan_path.is_file():
        try:
            import json
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            for group in plan.get("groups", []):
                members = group.get("members", [])
                if model_id in members and len(members) >= 1:
                    return {
                        "environment_id": group.get("environment_id"),
                        "interpreter": group.get("interpreter"),
                        "conda_env_path": group.get("conda_env_path"),
                        "group_members": members,
                        "shared": len(members) > 1,
                    }
        except Exception:
            pass
    return None


def status(model: dict, config) -> dict:
    model_id = model["model_id"]
    source_dir = model.get("source_dir")
    source = (config.WORKBENCH_THIRD_PARTY_ROOT / source_dir).resolve() if source_dir else None
    env_meta = model.get("environment") or {}
    spec = _find_env_spec(source) if source and source.is_dir() else None
    interp = _interpreter_status(model, source) if source else {"status": "NO_SOURCE", "interpreter": None, "variable": None}
    res = {
        "model_id": model_id,
        "environment_required": bool(model.get("environment_required")),
        "source_synced": source is not None and source.is_dir(),
        "source_path": str(source) if source else None,
        "env_spec_found": spec is not None,
        "env_spec_path": str(spec) if spec else None,
        "env_spec_type": spec.name if spec else None,
        "interpreter_status": interp["status"],
        "interpreter_tier": interp.get("tier"),
        "interpreter": interp["interpreter"],
        "env_var": interp["variable"],
        "runtime_verified": bool(interp.get("verified")),
        "runtime_reason": interp.get("reason"),
        "documented_python": env_meta.get("python"),
        "documented_pytorch": env_meta.get("pytorch"),
        "documented_cuda": env_meta.get("cuda"),
        "environment_confidence": env_meta.get("confidence"),
        # "ready" means the *runtime* is verified, matching doctor and guarded_plan.
        # A present-but-unverified interpreter is explicitly not ready.
        "ready": bool(interp.get("verified")) or not model.get("environment_required"),
    }
    shared_info = _check_shared_assignment(model_id, config)
    if shared_info:
        res["shared_group"] = shared_info.get("environment_id")
        res["shared_members"] = shared_info.get("group_members")
        res["shared_interpreter"] = shared_info.get("interpreter")
        if shared_info.get("interpreter") and not res.get("interpreter"):
            res["interpreter"] = shared_info["interpreter"]
    return res


def print_status(s: dict) -> None:
    label = "OK" if s["ready"] else "BLOCKED"
    tier = s.get("interpreter_tier") or s["interpreter_status"]
    print(f"[{label}] {s['model_id']}: interpreter={s['interpreter_status']} tier={tier} env_var={s['env_var']}")
    if s.get("shared_group"):
        print(f"  shared: group={s['shared_group']} members={', '.join(s.get('shared_members') or [])}")
    if s.get("runtime_reason"):
        print(f"  runtime: {s['runtime_reason']}")
    if s["env_spec_found"]:
        print(f"  spec: {s['env_spec_path']} ({s['env_spec_type']})")
    else:
        print(f"  spec: not found in {s['source_path'] or 'source not synced'}")
    if s["interpreter"]:
        print(f"  interpreter: {s['interpreter']}")
    if s["documented_python"]:
        print(f"  documented: python={s['documented_python']} pytorch={s['documented_pytorch']} cuda={s['documented_cuda']}")


def create_env(model: dict, s: dict, dry_run: bool, config=None) -> bool:
    model_id = model["model_id"]
    if config is None:
        config = resolve_config()
    shared_info = _check_shared_assignment(model_id, config)
    if shared_info:
        members_str = ", ".join(shared_info.get("group_members", []))
        env_id = shared_info.get("environment_id")
        interp = shared_info.get("interpreter")
        print(f"[GROUP] {model_id}: assigned to shared group {env_id} (members: {members_str}); interpreter: {interp}")
        if interp and not Path(interp).is_file() and not dry_run:
            from workbench.backend.environment_planner import plan_environments, provision_environment_plan
            from workbench.backend.registry import load_registry
            registry = load_registry()
            group_models = [m for m in registry["models"] if m["model_id"] in shared_info.get("group_members", [])]
            plan_path = config.WORKBENCH_ARTIFACTS_ROOT / "environment_plan.json"
            if plan_path.is_file():
                try:
                    import json
                    plan = json.loads(plan_path.read_text(encoding="utf-8"))
                except Exception:
                    plan = plan_environments(group_models, artifacts_root=config.WORKBENCH_ARTIFACTS_ROOT)
            else:
                plan = plan_environments(group_models, artifacts_root=config.WORKBENCH_ARTIFACTS_ROOT)
            provision_environment_plan(registry["models"], plan, artifacts_root=config.WORKBENCH_ARTIFACTS_ROOT, allowed=True)
        return True
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
    # Preserve the configured invocation path: resolving it would hand pip a
    # different interpreter than the environment the operator configured.
    interpreter = Path(configured).expanduser()
    if not interpreter.is_file():
        print(f"[BLOCKED] {model_id}: {env_var}={interpreter} not found", file=sys.stderr)
        return False
    is_isolated, iso_msg = check_environment_isolation(interpreter, model.get("environment") or {})
    if not is_isolated:
        print(f"[BLOCKED] {model_id}: refusing installation into unsafe target interpreter: {iso_msg}", file=sys.stderr)
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
        ok = create_env(model, s, args.dry_run, config=config)
        return 0 if ok else 1

    return 0 if s["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
