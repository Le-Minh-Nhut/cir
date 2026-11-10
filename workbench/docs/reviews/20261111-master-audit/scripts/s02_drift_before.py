"""S-02 failing-first: on HEAD (92ed66de) a stolen pool is still reported `verified`.

Loads the PRE-FIX runner straight out of git (`git show 92ed66de:workbench/composition/runner.py`)
and runs exactly what the scoring path did: pin to 1, let a concurrent job set the process-global
pool to 10, then ask what the record says. On HEAD `verified` is True -- the artifact would publish a
pin that no longer governed the arithmetic. The fixed module, same experiment, detects the drift.
"""
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/home/heheboiz/data/cir")

REVISION = "92ed66de"
REPO = "/home/heheboiz/data/cir"


def load_pre_fix_runner():
    source = subprocess.run(["git", "show", f"{REVISION}:workbench/composition/runner.py"],
                            cwd=REPO, capture_output=True, text=True, check=True).stdout
    handle = Path(tempfile.mkdtemp()) / "runner_head.py"
    handle.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("runner_head", handle)
    module = importlib.util.module_from_spec(spec)
    sys.modules["runner_head"] = module
    spec.loader.exec_module(module)
    return module


head = load_pre_fix_runner()
print("pre-fix runner from", REVISION)

pin = head._blas_thread_limit(head.BLAS_THREADS_PINNED)
print("HEAD pin record:", {k: pin[k] for k in ("requested", "previous", "effective", "verified")})
# A concurrent job changes the PROCESS-GLOBAL pool while the pinned region is in flight.
head._blas_thread_limit(10)
print("pool after a concurrent setter:", head._current_openblas_threads())
head._record_effective_blas_pin(pin)
print("policy_observed on HEAD:", head.observed_policy_record()["status"])
print("HEAD has blas_pin_scope:", hasattr(head, "blas_pin_scope"))
head._blas_thread_limit(20)

from workbench.composition import runner as fixed

with fixed.blas_pin_scope() as record:
    fixed._blas_thread_limit(10)
    fixed._current_openblas_threads()
print("FIXED drift_detected:", record.get("drift_detected"), "verified:", record.get("verified"))
print("FIXED reason:", (record.get("reason") or "")[:110])
fixed._blas_thread_limit(20)
