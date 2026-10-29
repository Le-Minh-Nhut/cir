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

from workbench.backend.errors import WorkbenchError
from workbench.backend.model_settings import setting
from workbench.backend.operator_config import resolve_config
from workbench.backend.registry import load_registry
from workbench.backend.environment_planner import _is_staging_path, _package_manager_env
from workbench.backend.runtime import READY_TIERS, check_environment_isolation, verify_environment


_ENV_SPEC_CANDIDATES = ("environment.yml", "environment.yaml", "requirements.txt")
_CONDA_COMMANDS = ("conda", "mamba", "micromamba")


def _verify_assigned_interpreter(model: dict, interpreter: str, config) -> "object | None":
    """The CURRENT verification of the assignment's interpreter against the selected contract.

    A persisted `state == "RUNTIME_VERIFIED"` records a past install; it is not evidence that the
    interpreter still satisfies the contract the operator selects now. The probe runs the same
    verifier the planner, doctor and launcher use, pointed at the assignment's own interpreter, so
    reuse is decided by measurement rather than by the record it is about to reuse.

    Returns None when the model declares no `python_env_var`: such a model has no contract to
    verify the assignment against, so the record stays the only evidence there is.
    """
    environment = dict(model.get("environment") or {})
    variable = environment.get("python_env_var")
    if not variable:
        return None
    previous = os.environ.get(variable)
    os.environ[variable] = str(interpreter)
    try:
        return verify_environment({**model, "environment": environment}, probe=True, config=config)
    finally:
        if previous is None:
            os.environ.pop(variable, None)
        else:
            os.environ[variable] = previous


def _assigned_interpreter_verifies(model: dict, interpreter: str, config) -> bool:
    """Whether a persisted assignment may be reused: it must not fail current verification.

    A model with no declared contract (see `_verify_assigned_interpreter`) keeps its record. A
    declared contract that does not currently pass — blocked, or merely unverified — is not
    reusable: the assignment is stale, and the caller must repair rather than reuse it.
    """
    report = _verify_assigned_interpreter(model, interpreter, config)
    return report is None or report.tier in READY_TIERS


def _resolved_install_plan(model: dict, config) -> tuple[list[str], str | None] | None:
    """The pip install plan for this model alone, or None when no adaptation is selected.

    The constraints are the planner's own (`model_constraints` + `_constraints` + the same spec
    renderer), so installer and verifier cannot disagree about what 'the resolved environment'
    is. A venv has no conda to install the conda-resolvable half, so every declared package goes
    through pip: the plan is the resolved constraint set, not a second interpretation of it.
    """
    from workbench.backend import environment_planner as planner
    from workbench.backend.runtime import runtime_adaptation

    if not runtime_adaptation(model, undeclared_is_error=False):
        return None
    third_party = getattr(config, "WORKBENCH_THIRD_PARTY_ROOT", None)
    constraints, conflict = planner._constraints(
        [planner.model_constraints(model, third_party_root=third_party)])
    if conflict:
        raise WorkbenchError("ENVIRONMENT_CONSTRAINTS_CONFLICT", conflict,
                             {"model_id": model.get("model_id")})
    constraints["pip_only"] = sorted(set(constraints["packages"]) | set(constraints["ranges"]))
    return (planner._pip_specs_for_group({"constraints": constraints}, {}),
            constraints.get("pip_index_url"))


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
    from workbench.backend.model_settings import setting

    return setting(var)


def _interpreter_status(model: dict, source_root: Path) -> dict:
    env = model.get("environment") or {}
    var = env.get("python_env_var")
    configured = _interpreter_from_env_var(model)
    if configured and _is_staging_path(configured):
        return {"status": "INTERRUPTED_STAGING", "tier": "NOT_READY", "interpreter": configured,
                "variable": var, "verified": False, "reason": "interrupted staging environment is not authoritative"}
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
                for group in data.get("groups", []):
                    if (group.get("environment_id") == model_info["environment_id"] and
                            model_id in group.get("members", [])):
                        return {**group, "group_members": list(group["members"]),
                                "shared": len(group["members"]) > 1}
                if data.get("groups"):
                    return None
                if (not data.get("groups") and
                        model_id in model_info.get("group_members", []) and
                        not any(_is_staging_path(model_info.get(key))
                                for key in ("interpreter", "conda_env_path"))):
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
                if any(_is_staging_path(group.get(key)) for key in ("interpreter", "conda_env_path")):
                    continue
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
        # Readiness follows the declared interpreter; `environment_required` only controls provisioning.
        "ready": bool(interp.get("verified")) or not env_meta.get("python_env_var"),
    }
    shared_info = _check_shared_assignment(model_id, config)
    if shared_info:
        res["shared_group"] = shared_info.get("environment_id")
        res["shared_members"] = shared_info.get("group_members")
        res["shared_interpreter"] = shared_info.get("interpreter")
        assigned = shared_info.get("interpreter")
        state = shared_info.get("state")
        if assigned:
            res["interpreter"] = assigned
            if state in ("FAILED", "BLOCKED"):
                res["ready"] = False
                res["runtime_reason"] = f"assigned group {res['shared_group']} is {state}"
            elif state == "RUNTIME_VERIFIED" and Path(assigned).is_file() and os.access(assigned, os.X_OK):
                # The persisted record only says a past install verified. Whether the assignment
                # is reusable is decided by the CURRENT verification of that same interpreter
                # against the selected contract: a row that says ready while the probe says
                # BLOCKED is the stale-assignment contradiction this check exists to end.
                report = _verify_assigned_interpreter(model, assigned, config)
                if report is None:
                    res["interpreter_status"] = "GROUP_READY"
                    res["runtime_verified"] = True
                    res["ready"] = True
                else:
                    res["interpreter_status"] = report.status
                    res["interpreter_tier"] = report.tier
                    res["runtime_verified"] = report.tier in READY_TIERS
                    res["runtime_reason"] = report.reason
                    res["ready"] = res["runtime_verified"]
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


def create_env(model: dict, s: dict, dry_run: bool, config=None, network_allowed: bool = False) -> bool:
    model_id = model["model_id"]
    if config is None:
        config = resolve_config()
    shared_info = _check_shared_assignment(model_id, config)
    if shared_info:
        members_str = ", ".join(shared_info.get("group_members", []))
        env_id = shared_info.get("environment_id")
        interp = shared_info.get("interpreter")
        print(f"[GROUP] {model_id}: assigned to shared group {env_id} (members: {members_str}); interpreter: {interp}")
        if (shared_info.get("state") == "RUNTIME_VERIFIED" and interp
                and not _is_staging_path(interp) and Path(interp).is_file() and os.access(interp, os.X_OK)
                and _assigned_interpreter_verifies(model, interp, config)):
            # Reused without reinstalling only because the assignment still verifies NOW against the
            # selected contract. A persisted RUNTIME_VERIFIED record over a broken environment is not
            # evidence of anything, and returning here was what made `--create` refuse to repair it.
            if dry_run:
                print(f"[OK] dry-run: would reuse group {env_id}")
            return True
        if dry_run:
            # Read-only AND result-stable: a dry run provisions nothing, so it does not drop the
            # assignment either. It reports the planned group exactly as before the
            # current-contract check existed, and the verdict cannot depend on --allow-network.
            # The stale→repair drop belongs to a REAL run, which the branch below handles.
            if shared_info.get("state") == "RUNTIME_VERIFIED":
                print(f"[NOTICE] {model_id}: assigned group {env_id} does not verify against the "
                      f"current contract; a real run repairs it instead of reusing it")
            print(f"[OK] dry-run: would provision or reuse group {env_id}")
            return True
    if shared_info:
        from workbench.backend.environment_planner import plan_environments, provision_environment_plan
        from workbench.backend.registry import load_registry
        registry = load_registry()
        group_members = shared_info.get("group_members", [model_id])
        group_models = [m for m in registry["models"] if m["model_id"] in group_members]
        if not group_models:
            group_models = [model]
        plan_path = config.WORKBENCH_ARTIFACTS_ROOT / "environment_plan.json"
        plan = None
        if plan_path.is_file():
            try:
                import json
                plan = json.loads(plan_path.read_text(encoding="utf-8"))
            except Exception:
                plan = None
        if plan is None:
            plan = plan_environments(group_models, artifacts_root=config.WORKBENCH_ARTIFACTS_ROOT)
        # Provision ONLY the group(s) the requested model belongs to. The plan file may carry
        # groups for every registered model; installing all of them on a single-model request
        # pulls gigabytes of unrelated packages the operator never asked for.
        wanted = {m["model_id"] for m in group_models}
        scoped = {**plan, "groups": [g for g in plan.get("groups", [])
                                     if wanted & set(g.get("members", []))]}
        assignment = provision_environment_plan(
            group_models,
            scoped,
            artifacts_root=config.WORKBENCH_ARTIFACTS_ROOT,
            allowed=True,
            network_allowed=network_allowed,
        )
        m_info = (assignment.get("models") or {}).get(model_id) or {}
        g_for_model = next((g for g in assignment.get("groups", []) if model_id in g.get("members", [])), None)

        if (
            m_info.get("state") in ("FAILED", "BLOCKED")
            or m_info.get("installation_status") in ("FAILED", "BLOCKED")
            or (g_for_model and g_for_model.get("state") in ("FAILED", "BLOCKED"))
            or (g_for_model and g_for_model.get("installation_status") in ("FAILED", "BLOCKED"))
            or (g_for_model and g_for_model.get("runtime_verification_status") not in ("READY", "NOT_REQUIRED"))
        ):
            print(f"[FAILED] {model_id}: provisioning failed for group {env_id}", file=sys.stderr)
            return False

        assigned_interp = m_info.get("interpreter") or (g_for_model.get("interpreter") if g_for_model else None)
        if not assigned_interp or not Path(assigned_interp).is_file() or not os.access(assigned_interp, os.X_OK):
            print(f"[FAILED] {model_id}: interpreter missing or not executable: {assigned_interp}", file=sys.stderr)
            return False

        return True
    if not s["source_synced"]:
        print(f"[BLOCKED] {model_id}: source not synced — sync first with sync_upstreams.py", file=sys.stderr)
        return False
    # An operator-preset interpreter (WORKBENCH_*_PYTHON set) must still go through the
    # isolation check in the venv branch below; the planner route is only for fresh installs
    # with no preset interpreter. The lookup is the shared one, so an export and a
    # `workbench.env` entry select the SAME target with the documented precedence.
    preset_interpreter = s.get("env_var") and setting(s["env_var"])
    if not preset_interpreter:
            # No shared assignment yet: route through the environment planner, which derives the
        # group from the model's own spec, provisions it under the workbench artifacts root, and
        # records the assignment. The direct venv/conda branches below remain for operators who
        # configure WORKBENCH_*_PYTHON themselves; a fresh Docker pilot has no preset interpreter,
        # so without this routing every fresh `prepare` would stop here asking for a variable.
        # Dry-run stays read-only: report what the planner WOULD form without provisioning.
        if dry_run:
            from workbench.backend.environment_planner import plan_environments
            from workbench.backend.registry import load_registry
            registry = load_registry()
            planned = plan_environments(
                [next(m for m in registry["models"] if m["model_id"] == model_id)],
                artifacts_root=config.WORKBENCH_ARTIFACTS_ROOT)
            groups = planned.get("groups") or []
            if groups:
                print(f"[OK] dry-run: would provision or reuse group {groups[0]['environment_id']}")
                return True
            print(f"[BLOCKED] {model_id}: planner could not form an environment group "
                  f"({(planned.get('blocked') or [{}])[0].get('reason', 'no reason recorded')})",
                  file=sys.stderr)
            return False
        from workbench.backend.environment_planner import plan_environments, provision_environment_plan
        from workbench.backend.registry import load_registry
        registry = load_registry()
        group_models = [next((m for m in registry["models"] if m["model_id"] == model_id), model)]
        plan = plan_environments(group_models, artifacts_root=config.WORKBENCH_ARTIFACTS_ROOT)
        wanted = {model_id}
        scoped = {**plan, "groups": [g for g in plan.get("groups", [])
                                     if wanted & set(g.get("members", []))]}
        if not scoped["groups"]:
            # The planner refused to form a group for this model; fall through to the manual
            # interpreter branches below so the operator sees the same guidance as before.
            pass
        else:
            assignment = provision_environment_plan(
                group_models,
                scoped,
                artifacts_root=config.WORKBENCH_ARTIFACTS_ROOT,
                allowed=True,
                network_allowed=network_allowed,
            )
            m_info = (assignment.get("models") or {}).get(model_id) or {}
            g_for_model = next((g for g in assignment.get("groups", []) if model_id in g.get("members", [])), None)
            if (
                m_info.get("state") in ("FAILED", "BLOCKED")
                or m_info.get("installation_status") in ("FAILED", "BLOCKED")
                or (g_for_model and g_for_model.get("installation_status") in ("FAILED", "BLOCKED"))
                or (g_for_model and g_for_model.get("runtime_verification_status") not in ("READY", "NOT_REQUIRED"))
            ):
                print(f"[FAILED] {model_id}: planner provisioning failed for group "
                      f"{(g_for_model or {}).get('environment_id')}", file=sys.stderr)
                return False
            assigned_interp = m_info.get("interpreter") or (g_for_model.get("interpreter") if g_for_model else None)
            if not assigned_interp or not Path(assigned_interp).is_file() or not os.access(assigned_interp, os.X_OK):
                print(f"[FAILED] {model_id}: interpreter missing or not executable: {assigned_interp}", file=sys.stderr)
                return False
            print(f"[OK] {model_id}: environment provisioned; interpreter {assigned_interp}")
            return True
    spec = s["env_spec_path"]
    spec_type = s["env_spec_type"]
    if not spec:
        print(f"[BLOCKED] {model_id}: no environment spec found in pinned source ({', '.join(_ENV_SPEC_CANDIDATES)})", file=sys.stderr)
        return False
    # When the operator selected a declared adaptation, the RESOLVED contract — the same
    # constraints the planner, verifier and assignment all consume — is what gets installed. The
    # raw historical freeze (or environment.yml) is a different dependency set and would undo the
    # migration. `_resolved_install_plan` returns None with no adaptation, so the historical
    # branches below are byte-identical in that case.
    try:
        resolved_specs, resolved_index = _resolved_install_plan(model, config) or (None, None)
    except WorkbenchError as error:
        print(f"[BLOCKED] {model_id}: {error.message}", file=sys.stderr)
        return False
    if spec_type in ("environment.yml", "environment.yaml") and resolved_specs is None:
        conda_cmd = _conda_available()
        if not conda_cmd:
            print(f"[BLOCKED] {model_id}: no conda/mamba found; required for {spec}", file=sys.stderr)
            return False
        env_name = f"workbench-{model_id}"
        if not network_allowed:
            print(f"[BLOCKED] {model_id}: conda env create requires network authorization "
                  f"(--allow-network)", file=sys.stderr)
            return False
        argv = [conda_cmd, "env", "create", "--name", env_name, "--file", spec, "--yes"]
        print(f"[RUN] {model_id}: {' '.join(argv)}")
        if dry_run:
            print(f"[OK] dry-run: would create conda env {env_name!r} from {spec}")
            return True
        result = subprocess.run(argv, env=_package_manager_env())
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
    configured = setting(env_var)
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
    if not network_allowed:
        print(f"[BLOCKED] {model_id}: pip install requires network authorization (--allow-network)",
              file=sys.stderr)
        return False
    if resolved_specs is None:
        argv = [str(interpreter), "-m", "pip", "install", "-r", spec]
    else:
        argv = [str(interpreter), "-m", "pip", "install", *resolved_specs]
        # An EXTRA index, not a replacement: the pytorch wheel index hosts the CUDA build the
        # adapted profile pins (cu128 for sm_120) and not the regular PyPI packages beside it.
        if resolved_index:
            argv += ["--extra-index-url", str(resolved_index)]
    print(f"[RUN] {model_id}: {' '.join(argv)}")
    if dry_run:
        print(f"[OK] dry-run: would install {spec if resolved_specs is None else 'resolved constraints'} "
              f"into {interpreter}")
        return True
    result = subprocess.run(argv, env=_package_manager_env())
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
    parser.add_argument("--allow-network", action="store_true",
                        help="authorize the package manager to reach its index; required to install")
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
        ok = create_env(model, s, args.dry_run, config=config, network_allowed=args.allow_network)
        return 0 if ok else 1

    # Optional provisioning must not mask later preflight or authorization blockers; status still
    # reports a declared-but-unverified interpreter as blocked.
    return 0 if not model.get("environment_required") else int(not s["ready"])


if __name__ == "__main__":
    raise SystemExit(main())
