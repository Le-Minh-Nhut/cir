"""Baseline: per-model interpreter as declared by the registry, resolved through the
workbench's own resolver, plus the version that interpreter reports.

Declared: registry.models[*].environment.python_env_var + workbench.env value.
Reported: `<interpreter> -c "import sys; print(sys.version)"`.
"""
import json
import subprocess
from pathlib import Path

import yaml

from workbench.backend.operator_config import read_env_file

env = read_env_file()
reg = yaml.safe_load(Path("workbench/registry/models.yaml").read_text(encoding="utf-8"))
print(f"# workbench.env keys read: {len(env)}")
print("# upstream_python = registry environment.python (what the upstream README pins)")
print("# adapted_python  = registry environment.adapted_runtime.python (what the operator env implements)")
print(f"# {'model_id':22s} {'var':32s} {'upstream':10s} {'adapted':9s} {'resolved':58s} reported")
for m in reg["models"]:
    mid = m["model_id"]
    e = m.get("environment") or {}
    var = e.get("python_env_var") or "-"
    declared = e.get("python") or "-"
    adapted = (e.get("adapted_runtime") or {}).get("python") or "-"
    resolved = env.get(var) if var in env else None
    if resolved is None:
        # fall back to the workbench interpreter the resolver would use
        resolved = "workbench/envs/ctl312h/bin/python (WORKBENCH default)"
        reported = subprocess.run(["workbench/envs/ctl312h/bin/python", "-c",
                                   "import sys;print(sys.version.split()[0])"],
                                  capture_output=True, text=True).stdout.strip()
    else:
        try:
            reported = subprocess.run([resolved, "-c", "import sys;print(sys.version.split()[0])"],
                                      capture_output=True, text=True, timeout=60).stdout.strip()
            if not reported:
                reported = "NO OUTPUT (exit %d)" % subprocess.run(
                    [resolved, "-c", "pass"], capture_output=True).returncode
        except FileNotFoundError:
            reported = "MISSING (FileNotFoundError)"
        except subprocess.TimeoutExpired:
            reported = "TIMEOUT"
    exists = "exists" if Path(resolved).exists() or resolved.startswith("workbench/") else "ABSENT"
    print(f"  {mid:22s} {var:32s} {declared:10s} {adapted:9s} {resolved:58s} {reported}  [{exists}]")
