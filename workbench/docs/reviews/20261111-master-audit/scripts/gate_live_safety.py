"""Is the gate change live-safe? Compare the SERVED verdicts against a direct in-process call.

The backend on :8000 was started before `parity_gate.py` changed, so it holds the OLD module in
memory. This script does not restart it (another worker owns the service). It records:

  1. the JSON the service is serving right now (fetched by the caller, passed in as a path);
  2. the verdicts this tree's module computes in-process;
  3. a field-by-field comparison of everything a consumer can branch on.

A difference in `missing` / `status` / `failure_count` is reported explicitly, because those are the
fields a UI count could key on.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/heheboiz/data/cir")
from workbench.backend import parity_gate as gate

served_path = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/live_gate.json")
served = json.loads(served_path.read_text(encoding="utf-8"))
local = gate.evaluate_all()

COMPARE = ("status", "passed", "complete", "failure_count", "missing",
           "applied_standard", "requested_standard")

out: list[str] = []
out.append(f"served_keys={sorted(served)}")
out.append(f"local_keys={sorted(local)}")
out.append(f"served_standard={served.get('standard')!r} local_standard={local.get('standard')!r}")
out.append(f"served_requested={served.get('requested_standard')!r} "
           f"local_requested={local.get('requested_standard')!r}")
out.append(f"served_passed_models={served.get('passed_models')} "
           f"local_passed_models={local.get('passed_models')}")
out.append("")

identical = True
for model_id in sorted(set(served.get("models", {})) | set(local["models"])):
    s = served.get("models", {}).get(model_id)
    l = local["models"].get(model_id)
    if s is None:
        out.append(f"{model_id}: PRESENT LOCALLY ONLY")
        identical = False
        continue
    if l is None:
        out.append(f"{model_id}: SERVED ONLY")
        identical = False
        continue
    diffs = {k: (s.get(k), l.get(k)) for k in COMPARE if s.get(k) != l.get(k)}
    identical = identical and not diffs
    out.append(f"{model_id}: status served={s.get('status')!r} local={l.get('status')!r} "
               f"failure_count served={s.get('failure_count')} local={l.get('failure_count')}")
    out.append(f"    missing served={s.get('missing')!r}")
    out.append(f"    missing local ={l.get('missing')!r}")
    out.append("    differing fields: " + (json.dumps(diffs) if diffs else "none"))
    out.append(f"    checks served={len(s.get('checks', []))} local={len(l.get('checks', []))}")
    # A consumer that branches on `passed` must see the same answer both ways.
    out.append(f"    passed served={s.get('passed')} local={l.get('passed')}")
    out.append("")

out.append(f"ALL_COMPARED_FIELDS_IDENTICAL={identical}")
out.append("")
out.append("VERDICT: " + (
    "the running service's answer and the new module's answer agree on every field a consumer "
    "branches on, so activating the new module changes no count and no badge"
    if identical else
    "the two disagree; the service must be restarted by its owner and the UI re-checked before "
    "publishing"))
print("\n".join(out))
