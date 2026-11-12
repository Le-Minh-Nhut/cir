"""Step 4: a reproducible blocker record for every non-executable model.

Each blocker is a COMMAND plus its VERBATIM error output, run here, so another person can rerun the
command and get the same text. No prose summary stands in for a measurement: the `command` field is
copy-pasteable and `stdout`/`stderr` are what it printed in this run.

Writes `blockers.json` and `blockers.txt` beside itself.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path("/home/heheboiz/data/cir")
CONTROL = REPO / "workbench/envs/ctl312h/bin/python"
OUT = Path(__file__).resolve().parent

PROBE = r"""
import json, sys
from pathlib import Path
sys.path.insert(0, "/home/heheboiz/data/cir")
from workbench.backend.operator_config import resolve_config
from workbench.backend.registry import load_registry, checkpoint_availability
from workbench.backend.preflight import preflight_blockers
from workbench.backend.runtime import resolve_model_interpreter
model_id = sys.argv[1]
config = resolve_config()
registry = load_registry()
model = next(row for row in registry["models"] if row["model_id"] == model_id)
variable = (model.get("environment") or {}).get("python_env_var")
resolved = resolve_model_interpreter(model, config)
print(json.dumps({
    "model_id": model_id,
    "python_env_var": variable,
    "resolver_returned": resolved,
    "interpreter_exists": bool(resolved) and Path(resolved).is_file(),
    "preflight_blockers": preflight_blockers(model, config),
    "adapter_module": model.get("composition_adapter"),
    "adapter_module_exists": bool(model.get("composition_adapter")) and (
        Path("/home/heheboiz/data/cir") / (str(model["composition_adapter"]).replace(".", "/") + ".py")
    ).is_file(),
    "checkpoints": [
        {"checkpoint_id": variant.get("checkpoint_id"),
         "path": checkpoint_availability(model, variant, None)["checkpoint_path"],
         "downloaded": checkpoint_availability(model, variant, None)["checkpoint_downloaded"],
         "mapping_verified": checkpoint_availability(model, variant, None)["mapping_verified"]}
        for variant in (model.get("checkpoint_variants") or [])
    ],
}, indent=1))
"""


def probe_script(model_id: str) -> Path:
    """Write the probe to a real file: `python -c CODE arg` leaves `sys.argv[1]` unset, so a printed
    `-c` command would not reproduce. A file plus argv is exactly rerunnable."""
    path = OUT / f"probe_{model_id}.py"
    path.write_text(PROBE.replace("model_id = sys.argv[1]",
                                  f"model_id = {model_id!r}"), encoding="utf-8")
    return path


def run(command: list[str], *, env: dict | None = None) -> dict:
    completed = subprocess.run(command, cwd=str(REPO), capture_output=True, text=True,
                               env=env, timeout=600)
    return {"command": " ".join(command), "exit": completed.returncode,
            "stdout": completed.stdout[-4000:], "stderr": completed.stderr[-4000:]}


def main() -> int:
    records: list[dict] = []

    # The dangling shared target: ONE removed conda env that five declared interpreters point at.
    records.append({
        "blocker": "INTERPRETER_BROKEN_SYMLINK (shared)",
        "models": ["airknow", "pair", "sprc", "combiner_rn50x4_noft", "clip4cir_rn50x4_fullft"],
        "asset_missing": "/home/heheboiz/miniconda3/envs/teacher-sprc/bin/python",
        "note": "five DECLARED env symlinks point at one removed conda env; repairing it is the "
                "operator's, not this lane's",
        **run(["readlink", "-f", "workbench/envs/airknow-adapted/bin/python"]),
        "also": [run(["test", "-e", f"workbench/envs/{name}/bin/python"])
                 | {"symlink": f"workbench/envs/{name}/bin/python"}
                 for name in ("airknow-adapted", "pair-adapted", "sprc-adapted",
                              "clip4cir-adapted", "limn-adapted")],
    })

    # The unset variables. `workbench.env` is the operator's file and is NOT edited by this lane.
    records.append({
        "blocker": "PYTHON_ENV_VAR_UNSET",
        "models": ["airknow", "pair", "dcnet", "combiner_rn50x4_noft", "clip4cir_rn50x4_fullft",
                   "sprc"],
        "asset_missing": "the WORKBENCH_*_PYTHON variable for the model",
        **run(["grep", "-n", "-E", r"WORKBENCH_(AIRKNOW|PAIR|DCNET|CLIP4CIR|SPRC)_PYTHON",
               "workbench/config/workbench.env"]),
        "note": "grep exits 1 and prints nothing: the variable is absent, which is the blocker",
    })

    # Each non-executable model, probed through the production functions.
    for model_id in ("csmcir", "airknow", "pair", "dcnet", "combiner_rn50x4_noft",
                     "clip4cir_rn50x4_fullft", "sprc"):
        records.append({"blocker": f"PROBE {model_id}", "models": [model_id],
                        **run([str(CONTROL), str(probe_script(model_id))])})

    # PAIR's own declared blocker, the one the ledger says must not be re-derived.
    records.append({
        "blocker": "DEFERRED_FROM_WEB (pair)",
        "models": ["pair"],
        "asset_missing": "a published source class for the released pickle",
        **run([str(CONTROL), "-c",
               "import json,sys; sys.path.insert(0,'/home/heheboiz/data/cir'); "
               "from workbench.backend.registry import load_registry, web_deferred_models; "
               "reg=load_registry(); print(json.dumps(web_deferred_models(reg), indent=1))"]),
    })

    (OUT / "blockers.json").write_text(json.dumps(records, indent=1) + "\n", encoding="utf-8")
    lines = []
    for record in records:
        lines.append("=" * 78)
        lines.append(f"# {record['blocker']}  models={record.get('models')}")
        lines.append(f"$ {record.get('command')}")
        lines.append(f"exit={record.get('exit')}")
        if record.get("stdout"):
            lines.append("--- stdout ---")
            lines.append(record["stdout"].rstrip())
        if record.get("stderr"):
            lines.append("--- stderr ---")
            lines.append(record["stderr"].rstrip())
    (OUT / "blockers.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[-40:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
