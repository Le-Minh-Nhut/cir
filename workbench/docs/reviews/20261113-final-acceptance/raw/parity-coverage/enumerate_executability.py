"""Step 1 of the paired-parity coverage lane: one row per registry model, from MEASURED facts.

Why this file exists: the acceptance requirement is "every executable, in-scope configuration", and
the only way to answer it without registry prose is to ask the code that already computes each
status. So this calls, per model:

  * `registry.load_registry()`                -- the 17 entries, declared protocol, category policy
  * `runtime.resolve_model_interpreter()`     -- the interpreter the model's env var RESOLVES to
  * `registry.checkpoint_availability()`      -- presence, size, mapping, runnable, blocking reasons
  * `preflight.preflight_blockers()`          -- the shared blocker list (data, source, env, assets)
  * `<composition_adapter>.Adapter.status()`  -- the adapter's own readiness verdict
  * `os.path.exists` on the interpreter path  -- follows the symlink, so a dangling env is visible

Nothing here re-derives a status the production code already computes; each column names its source.
The verdict per row is a small classifier over those measured facts, and the classifier is printed
alongside so a reader can disagree with it.

Writes `executability.json` beside itself. No GPU, no model load, no network.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path("/home/heheboiz/data/cir")
sys.path.insert(0, str(REPO))

OUT = Path(__file__).resolve().parent

#: The environment files that hold the model interpreters. Read to say WHICH variable was consulted,
#: never to change one (the operator's `workbench.env` is not ours to edit).
ENV_FILES = (REPO / "workbench/config/workbench.env",)


def _declared_interpreter(variable: str | None) -> dict:
    """What the env files declare for *variable*, verbatim, plus what the symlink resolves to."""
    record = {"variable": variable, "declared_in": None, "declared_value": None,
              "resolves_to": None, "exists": False, "dangling_target": None}
    if not variable:
        return record
    for path in ENV_FILES:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith(f"{variable}="):
                value = stripped.split("=", 1)[1].strip().strip('"').strip("'")
                record.update(declared_in=str(path.relative_to(REPO)), declared_value=value)
                break
        if record["declared_value"]:
            break
    value = record["declared_value"]
    if value:
        target = Path(value)
        record["exists"] = target.exists()
        if target.is_symlink() and not target.exists():
            record["dangling_target"] = os.readlink(target)
        elif target.exists():
            record["resolves_to"] = str(Path(value).resolve())
    return record


def _checkpoint_rows(model: dict, config) -> list[dict]:
    from workbench.backend.registry import checkpoint_availability

    rows = []
    for variant in model.get("checkpoint_variants") or []:
        availability = checkpoint_availability(model, variant, None)
        path = Path(availability["checkpoint_path"])
        if path.is_file():
            size, kind = path.stat().st_size, "file"
        elif path.is_dir():
            size = sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
            kind = "dir"
        else:
            size, kind = None, "absent"
        rows.append({
            "checkpoint_id": variant.get("checkpoint_id"),
            "filename": variant.get("filename"),
            "mapping_status": variant.get("checkpoint_mapping_status"),
            "path": str(path),
            "kind": kind,
            "size_bytes": size,
            "downloaded": availability["checkpoint_downloaded"],
            "mapping_verified": availability["mapping_verified"],
            "official_sha256_match": availability["official_sha256_match"],
            "runnable": availability["runnable"],
            "blocking_reasons": availability["blocking_reasons"],
        })
    return rows


def _adapter_status(model: dict, config) -> dict:
    """The composition adapter's own `status()`, which is the production readiness verdict."""
    module_name = model.get("composition_adapter")
    if not module_name:
        return {"module": None, "exists": False, "ready": None,
                "reason": "no composition adapter is declared for this model"}
    module_path = REPO / (module_name.replace(".", "/") + ".py")
    try:
        module = __import__(module_name, fromlist=["Adapter"])
        status = module.Adapter.status(config)
        payload = status.to_dict() if hasattr(status, "to_dict") else dict(status.__dict__)
    except Exception as error:                                   # noqa: BLE001 - recorded
        return {"module": module_name, "exists": module_path.is_file(), "ready": False,
                "reason": f"Adapter.status() raised {type(error).__name__}: {error}"}
    return {"module": module_name, "exists": module_path.is_file(),
            "ready": bool(payload.get("ready")), "reason": payload.get("reason"),
            "checkpoint_ids": payload.get("checkpoint_ids"),
            "per_mode": payload.get("per_mode"), "notes": payload.get("notes")}


def _category_scopes(adapter_module: str | None) -> list | None:
    if not adapter_module:
        return None
    try:
        module = __import__(adapter_module, fromlist=["Adapter"])
        return list(getattr(module.Adapter, "category_scopes", []) or [])
    except Exception:                                            # noqa: BLE001 - recorded as absent
        return None


def _verdict(row: dict) -> tuple[str, str]:
    """EXECUTABLE / BLOCKED(<code>) / NO_ADAPTER, from the measured columns only."""
    if not row["adapter"]["exists"]:
        return "NO_ADAPTER", "no composition adapter module on disk"
    interpreter = row["interpreter"]
    if not interpreter["variable"]:
        return "BLOCKED(NO_PYTHON_ENV_VAR)", "the registry declares no python_env_var"
    if not interpreter["declared_value"]:
        return "BLOCKED(ENV_VAR_UNSET)", f"{interpreter['variable']} is not set in any env file"
    if not interpreter["exists"]:
        return ("BLOCKED(INTERPRETER_BROKEN_SYMLINK)",
                f"{interpreter['declared_value']} -> {interpreter['dangling_target']} (absent)")
    if not row["adapter"]["ready"]:
        return "BLOCKED(ADAPTER_NOT_READY)", str(row["adapter"]["reason"])
    usable = [item for item in row["checkpoints"] if item["downloaded"]]
    if not usable:
        return "BLOCKED(NO_CHECKPOINT_ON_DISK)", "no declared checkpoint is present"
    return "EXECUTABLE", "interpreter resolves, adapter ready, at least one checkpoint present"


def main() -> int:
    from workbench.backend.operator_config import resolve_config
    from workbench.backend.preflight import preflight_blockers
    from workbench.backend.registry import load_registry
    from workbench.backend.runtime import resolve_model_interpreter

    config = resolve_config()
    registry = load_registry()
    rows = []
    for model in registry["models"]:
        model_id = model["model_id"]
        declared = (model.get("environment") or {}).get("python_env_var")
        resolved = resolve_model_interpreter(model, config)
        interpreter = _declared_interpreter(declared)
        # The resolver may reach an interpreter the env file does not declare (a persisted
        # provisioning assignment); record both, because they can disagree.
        if resolved and not interpreter["declared_value"]:
            interpreter.update(declared_value=resolved, exists=Path(resolved).exists(),
                               resolves_to=str(Path(resolved).resolve()))
        adapter = _adapter_status(model, config)
        row = {
            "model_id": model_id,
            "python_env_var": declared,
            "interpreter": interpreter,
            "resolver_returned": resolved,
            "checkpoints": _checkpoint_rows(model, config),
            "adapter": adapter,
            "category_scopes": _category_scopes(model.get("composition_adapter")),
            "native_protocol": model.get("native_protocol"),
            "supported_protocols": model.get("supported_protocols"),
            "category_model_policy": model.get("category_model_policy"),
            "reproduction_status": model.get("reproduction_status"),
            "command_status": model.get("command_status"),
            "replay_status": model.get("replay_status"),
            "source_dir": model.get("source_dir"),
            "preflight_blockers": preflight_blockers(model, config),
            "web_publication": model.get("web_publication"),
        }
        row["executability"], row["executability_reason"] = _verdict(row)
        rows.append(row)
        print(f"{model_id:24s} {row['executability']:32s} {row['executability_reason'][:80]}",
              flush=True)

    document = {
        "generated_by": "workbench/docs/reviews/20261113-final-acceptance/raw/parity-coverage/"
                        "enumerate_executability.py",
        "interpreter": sys.executable,
        "registry_count": len(rows),
        "verdicts": {value: sum(1 for row in rows if row["executability"] == value)
                     for value in sorted({row["executability"] for row in rows})},
        "note": "each column names the production function it came from; nothing is re-derived here",
        "models": rows,
    }
    # `web_publication.decided` is a YAML date, so the registry holds `datetime.date` objects.
    # Serialised as ISO strings rather than dropped: the deferral decision date is part of the row.
    (OUT / "executability.json").write_text(json.dumps(document, indent=1, default=str) + "\n",
                                            encoding="utf-8")
    print(json.dumps(document["verdicts"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
