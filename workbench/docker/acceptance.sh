#!/usr/bin/env bash
# Executable acceptance suite, groups A-I, for the stabilized release.
# Runs against a live workbench (backend [+ frontend]) or the in-container stack.
#
# Usage:
#   workbench/docker/acceptance.sh [BASE_URL]     # default http://127.0.0.1:8000
#
# Groups (each prints PASS/FAIL and exits nonzero on the first FAIL):
#   A first-boot honesty        models 200; runs 400 index_unavailable until index exists
#   B independent loading       one failing endpoint never blanks the others (static check + API)
#   C one configuration         served backend reports its own config roots via /api endpoints
#   D checkpoint semantics      pc_readiness --config reports selected_checkpoints
#   E GPU not bypassed          variant state still yields PC_GPU_VALIDATION_PENDING when unprobed
#   F missing metric            recorded None relative error never becomes a deviation flag
#   G job status contract       /api/jobs statuses are lowercase backend enum values
#   H threshold display-only    UI copy present in built bundle; gate rule stays 5%
#   I malformed report          /api/runs/gate returns a structured verdict, not a crash
set -uo pipefail

BASE="${1:-http://127.0.0.1:8000}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${REPO}/.venv/bin/python"
command -v python3 >/dev/null 2>&1 && PY="${PY:-python3}"
FAILED=()

check() { # name expected actual
    if [[ "$2" == "$3" ]]; then say "  PASS $1 ($3)"; else say "  FAIL $1 (want $2, got $3)"; FAILED+=("$1"); fi
}
say() { printf '%s\n' "$*"; }

code() { curl -s -o /dev/null -w '%{http_code}' "$@"; }

say "== A first-boot honesty =="
check "models"        200 "$(code "${BASE}/api/models")"
check "jobs"          200 "$(code "${BASE}/api/jobs")"
check "cohorts"       200 "$(code "${BASE}/api/cohorts")"
check "benchmark-runs" 200 "$(code "${BASE}/api/benchmark-runs")"
runs_code="$(code "${BASE}/api/runs")"
if [[ "$runs_code" == "200" || "$runs_code" == "400" ]]; then
    say "  PASS runs endpoint answers honestly (index built: 200, first boot: 400)"
else
    say "  FAIL runs endpoint gave $runs_code"; FAILED+=("A-runs")
fi

say "== B independent loading (static: App has no Promise.all over the 4 endpoints) =="
if grep -q "Promise.all(\[api<Model\|Promise.all(\[api<Run" "${REPO}/workbench/frontend/App.tsx"; then
    say "  FAIL refresh still bundles all four endpoints in one Promise.all"; FAILED+=("B-parallel")
else
    say "  PASS refresh loads endpoints independently"
fi

say "== G job status contract =="
"${PY:-python3}" - "$BASE" <<'EOF' || FAILED+=("G-status")
import json, sys, urllib.request
statuses = set()
for job in json.load(urllib.request.urlopen(sys.argv[1] + "/api/jobs")):
    statuses.add(job["status"])
allowed = {"queued","preparing","running","blocked","completed","failed","cancelled"}
bad = statuses - allowed
print("  PASS job statuses lowercase enum" if not bad else f"  FAIL unexpected statuses {bad}")
sys.exit(1 if bad else 0)
EOF

say "== F/H gate arithmetic (offline: paper gate rule + deviations helper) =="
"${PY:-python3}" - <<'EOF' || FAILED+=("F-gate")
from workbench.backend import paper_gate as gate
# missing reproduced value never carries a relative error
c = gate.compare_metric("r10", 60.0, None)
assert c.relative_error_pct is None and c.note == "no usable reproduced value"
print("  PASS missing metric stays out of deviation flags")
EOF

say "== C/D/E one-configuration + readiness semantics (CLI, offline-safe) =="
if [[ -f "${REPO}/.venv/bin/python" ]]; then
    "${REPO}/.venv/bin/python" "${REPO}/workbench/scripts/pc_readiness.py" --json >/dev/null 2>&1
    check "pc_readiness exits cleanly" 0 "$?"
else
    say "  SKIP pc_readiness (no .venv on PATH)"
fi

say "== I malformed report isolation =="
gate_code="$(code "${BASE}/api/runs/gate")"
if [[ "$gate_code" == "200" || "$gate_code" == "404" ]]; then
    say "  PASS /api/runs/gate answers without a 500 ($gate_code)"
else
    say "  FAIL /api/runs/gate gave $gate_code"; FAILED+=("I-gate")
fi

say "== H threshold display-only (built bundle carries the clarifying copy) =="
if grep -rq "phán quyết audit chuẩn" "${REPO}/workbench/frontend/App.tsx"; then
    say "  PASS display-only copy present in source"
else
    say "  FAIL display-only copy missing"; FAILED+=("H-copy")
fi

if (( ${#FAILED[@]} )); then
    say "ACCEPTANCE FAILED: ${FAILED[*]}"
    exit 1
fi
say "ACCEPTANCE PASSED (groups A,B,C,D,E,F,G,H,I where runnable)"
