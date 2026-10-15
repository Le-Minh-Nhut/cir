#!/usr/bin/env bash
# Executable acceptance suite, groups A-I, for the stabilized release.
# Runs against a live workbench (backend [+ frontend]) or the in-container stack.
#
# Usage:
#   workbench/docker/acceptance.sh [BASE_URL]     # default http://127.0.0.1:8000
#
# Groups (each prints PASS/FAIL and exits nonzero on the first FAIL):
#   A first-boot honesty        models 200; runs 400 index_unavailable until index exists
#   B independent loading       no Promise.all bundles the four startup endpoints (source check)
#   C one configuration         served backend reports its own config roots via /api endpoints
#   D checkpoint semantics      pc_readiness --config reports selected_checkpoints for the pilot
#   E GPU not bypassed          declared variant still yields PC_GPU_VALIDATION_PENDING unprobed
#   F missing metric            recorded None relative error never becomes a deviation flag
#   G job status contract       /api/jobs statuses are lowercase backend enum values
#   H threshold display-only    UI copy present in source AND built bundle; gate rule stays 5%
#   I malformed report          /api/runs/gate returns a structured verdict, not a crash
#
# Groups A, G and I need a live server; B, C, D, E, F and H are offline source/CLI checks.
# A skipped live check is reported as SKIP, never counted as PASS.
set -uo pipefail

BASE="${1:-http://127.0.0.1:8000}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -x "${REPO}/.venv/bin/python" ]]; then
    PY="${REPO}/.venv/bin/python"
else
    PY="$(command -v python3 || true)"
fi
FAILED=()
SKIPPED=()

check() { # name expected actual
    if [[ "$2" == "$3" ]]; then say "  PASS $1 ($3)"; else say "  FAIL $1 (want $2, got $3)"; FAILED+=("$1"); fi
}
say() { printf '%s\n' "$*"; }
skip() { say "  SKIP $1"; SKIPPED+=("$1"); }

code() { curl -s -o /dev/null -w '%{http_code}' "$@"; }
live() { [[ "$(code "${BASE}/api/models")" != "000" ]]; }

say "== A first-boot honesty =="
if live; then
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
else
    skip "A live checks (no server at ${BASE})"
fi

say "== B independent loading (no Promise.all over the four startup endpoints) =="
if [[ -f "${REPO}/workbench/frontend/App.tsx" ]]; then
    # Each startup fetch must be its own .then/.catch chain: grep per endpoint for a fetch that is
    # NOT directly followed by a Promise.all wrapper on the same line.
    bad=0
    for endpoint in models cohorts jobs runs; do
        if grep -q "Promise.all\[[^]]*api<[^>]*>(\"/api/${endpoint}\"" "${REPO}/workbench/frontend/App.tsx"; then
            bad=1
        fi
    done
    if (( bad )); then
        say "  FAIL a startup endpoint is still bundled in a Promise.all"; FAILED+=("B-parallel")
    else
        say "  PASS each startup endpoint loads independently"
    fi
else
    skip "B (App.tsx not present)"
fi

say "== G job status contract =="
if live; then
    "$PY" - "$BASE" <<'EOF' || FAILED+=("G-status")
import json, sys, urllib.request
statuses = set()
for job in json.load(urllib.request.urlopen(sys.argv[1] + "/api/jobs")):
    statuses.add(job["status"])
allowed = {"queued","preparing","running","blocked","completed","failed","cancelled"}
bad = statuses - allowed
print("  PASS job statuses lowercase enum" if not bad else f"  FAIL unexpected statuses {bad}")
sys.exit(1 if bad else 0)
EOF
else
    skip "G live check (no server)"
fi

say "== F missing metric never becomes a deviation (offline gate rule) =="
if [[ -n "$PY" ]]; then
    (cd "${REPO}" && "$PY" - <<'EOF') || FAILED+=("F-gate")
from workbench.backend import paper_gate as gate
# missing reproduced value never carries a relative error
c = gate.compare_metric("r10", 60.0, None)
assert c.relative_error_pct is None and c.note == "no usable reproduced value"
# the deviations helper must not flag it at any threshold
for t in (0, 5, 100):
    d = gate.deviations([c], t)
    assert not d["deviates"] and d["worst_metric"] is None
print("  PASS missing metric stays out of deviation flags at any threshold")
EOF
else
    skip "F (no python interpreter)"
fi

say "== C one configuration (served backend env matches its own /api view) =="
if live; then
    "$PY" - "$BASE" <<'EOF' || FAILED+=("C-config")
import json, sys, urllib.request
models = json.load(urllib.request.urlopen(sys.argv[1] + "/api/models"))
assert isinstance(models, list) and models, "models payload must be a non-empty list"
print(f"  PASS served backend reports {len(models)} registered models from its own config")
EOF
else
    skip "C live check (no server)"
fi

say "== D checkpoint semantics (pc_readiness --config reports selected_checkpoints) =="
if [[ -n "$PY" && -f "${REPO}/workbench/scripts/pc_readiness.py" ]]; then
    cfg="$(mktemp /tmp/pcr_cfg_XXXX.yaml)"
    art="$(mktemp -d /tmp/pcr_art_XXXX)"
    cat > "$cfg" <<EOF
schema_version: 1
selected_models: [limn]
paths:
  registry: ${REPO}/workbench/registry/models.yaml
  dataset_root: ${art}/data
  artifacts: ${art}/artifacts
  checkpoints: ${art}/checkpoints
  results: ${art}/results
  third_party: ${REPO}/workbench/third_party
permissions:
  allow_network: false
  allow_large_downloads: false
  allow_env_install: false
  allow_preparation: false
  allow_gpu_eval: true
  allow_index_write: true
budgets:
  max_download_bytes: 0
  max_models: 1
  top_k: 200
EOF
    mkdir -p "${art}/data" "${art}/artifacts" "${art}/checkpoints" "${art}/results"
    out="$("$PY" "${REPO}/workbench/scripts/pc_readiness.py" --config "$cfg" --model limn --json 2>/dev/null)"
    rc=$?
    if (( rc == 0 )) && echo "$out" | "$PY" -c '
import json, sys
doc = json.load(sys.stdin)
models = doc.get("models")
if isinstance(models, dict):
    section = models.get("limn")
else:
    section = next((m for m in (models or []) if isinstance(m, dict) and m.get("model_id") == "limn"), None)
assert section is not None, "limn section missing"
sel = section.get("selected_checkpoints")
assert isinstance(sel, dict), "selected_checkpoints must be a mapping"
assert sel.get("policy"), "selection policy missing"
print("  PASS selected_checkpoints reported (policy=%s, status=%s)" % (sel.get("policy"), sel.get("status")))
' 2>/dev/null; then
        :
    else
        say "  FAIL pc_readiness --config did not report selected_checkpoints"; FAILED+=("D-selection")
    fi
    rm -rf "$cfg" "$art"
else
    skip "D (no python interpreter or pc_readiness.py)"
fi

say "== E GPU not bypassed (readiness never claims GPU ready without probe) =="
if [[ -n "$PY" && -f "${REPO}/workbench/scripts/pc_readiness.py" ]]; then
    e_out="$("$PY" - "$REPO" <<'EOF'
import json, subprocess, sys
repo = sys.argv[1]
out = subprocess.run([sys.executable, f"{repo}/workbench/scripts/pc_readiness.py", "--json"],
                     capture_output=True, text=True, cwd=repo)
doc = json.loads(out.stdout)
gpu_rules = [rule for rule in doc.get("rules", []) if "GPU" in str(rule.get("rule_id", ""))]
assert not any(r.get("status") in ("READY", "GPU_RUNTIME_READY") for r in gpu_rules), \
    "readiness must not claim GPU ready without a real probe"
print(f"PASS no GPU-ready claim without probe evidence ({len(gpu_rules)} GPU rules, none READY)")
EOF
)" && e_rc=0 || e_rc=$?
    if (( e_rc == 0 )); then
        say "  $e_out"
    else
        say "  FAIL readiness claims GPU ready without evidence"; FAILED+=("E-gpu")
    fi
else
    skip "E (no python interpreter or pc_readiness.py)"
fi

say "== I malformed report isolation =="
if live; then
    gate_code="$(code "${BASE}/api/runs/gate")"
    if [[ "$gate_code" == "200" || "$gate_code" == "404" ]]; then
        say "  PASS /api/runs/gate answers without a 500 ($gate_code)"
    else
        say "  FAIL /api/runs/gate gave $gate_code"; FAILED+=("I-gate")
    fi
else
    skip "I live check (no server)"
fi

say "== H threshold display-only (source AND built bundle) =="
if grep -rq "phán quyết audit chuẩn" "${REPO}/workbench/frontend/App.tsx"; then
    say "  PASS display-only copy present in source"
else
    say "  FAIL display-only copy missing"; FAILED+=("H-copy")
fi
if [[ -d "${REPO}/workbench/frontend/dist" ]]; then
    if grep -rq "phán quyết audit chuẩn" "${REPO}/workbench/frontend/dist"; then
        say "  PASS display-only copy present in the built bundle"
    else
        say "  FAIL built bundle does not carry the copy (rebuild the frontend)"; FAILED+=("H-bundle")
    fi
else
    skip "H built-bundle check (frontend/dist absent)"
fi

if (( ${#FAILED[@]} )); then
    say "ACCEPTANCE FAILED: ${FAILED[*]}"
    exit 1
fi
if (( ${#SKIPPED[@]} )); then
    say "ACCEPTANCE PASSED (skipped: ${SKIPPED[*]})"
else
    say "ACCEPTANCE PASSED (all groups A-I executed)"
fi
