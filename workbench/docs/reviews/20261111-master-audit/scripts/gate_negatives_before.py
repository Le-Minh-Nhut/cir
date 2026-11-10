"""The seven required negatives, run against the PRE-FIX gate to record failing-first evidence.

The gate now refuses to load the pre-fix module from this tree, so the "before" numbers in the
report are the ones recorded at the time (raw/gate_negatives_before.txt) and reproduced here by
re-running the same fixtures against `git show 92ed66de:workbench/backend/parity_gate.py`.
"""
import importlib.util, json, sys, tempfile
from pathlib import Path

sys.path.insert(0, "/home/heheboiz/data/cir")
spec = importlib.util.spec_from_file_location("parity_gate_head", "/tmp/parity_gate_head.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
print("loaded pre-fix gate from", spec.origin)
print("has INCOMPLETE:", hasattr(gate, "INCOMPLETE"))
print("EVIDENCED_MODELS:", gate.EVIDENCED_MODELS)
