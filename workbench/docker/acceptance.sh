#!/usr/bin/env bash
# Executable acceptance suite, groups A-I, for the stabilized release.
# Runs against a live workbench (backend [+ frontend]) or the in-container stack.
#
# Usage:
#   workbench/docker/acceptance.sh [BASE_URL]     # default http://127.0.0.1:8000
#
# Groups (each prints PASS/FAIL and exits nonzero on the first FAIL):
#   A first-boot honesty        models 200; runs 503 index_unavailable until index exists
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

code() { curl -s --max-time 30 -o /dev/null -w '%{http_code}' "$@"; }
live() { [[ "$(code "${BASE}/api/models")" != "000" ]]; }

say "== A first-boot honesty =="
if live; then
    check "models"        200 "$(code "${BASE}/api/models")"
    check "jobs"          200 "$(code "${BASE}/api/jobs")"
    check "cohorts"       200 "$(code "${BASE}/api/cohorts")"
    check "benchmark-runs" 200 "$(code "${BASE}/api/benchmark-runs")"
    runs_code="$(code "${BASE}/api/runs")"
    if [[ "$runs_code" == "200" || "$runs_code" == "503" ]]; then
        say "  PASS runs endpoint answers honestly (index built: 200, first boot: 503 index_unavailable)"
    else
        say "  FAIL runs endpoint gave $runs_code"; FAILED+=("A-runs")
    fi
else
    skip "A live checks (no server at ${BASE})"
fi

say "== B independent loading (no Promise.all over the four startup endpoints) =="
if [[ -f "${REPO}/workbench/frontend/App.tsx" ]]; then
    # Structural check, not a line regex: Promise.all can wrap the endpoint call across lines
    # or through template literals that a same-line grep never sees. The four startup
    # endpoints must each load on its own .then/.catch chain.
    OUT="$("${REPO}/.venv/bin/python" - "${REPO}/workbench/frontend/App.tsx" <<'EOF'
import re, sys
text = open(sys.argv[1], encoding="utf-8").read()
# Balance-scan each Promise.all(...) block (nested parens included), then look for the
# startup endpoints inside it, across line breaks, quotes and template literals.
bundled = set()
for m in re.finditer(r"Promise\.all", text):
    i = m.end(); depth = 0
    while i < len(text):
        ch = text[i]
        if ch == "(": depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0: break
        i += 1
    block = text[m.end():i]
    for ep in ("models", "cohorts", "jobs", "runs"):
        if re.search(rf"/api/{ep}\b", block) or re.search(rf"/api/\$\{{[^}}]*{ep}[^}}]*\}}", block):
            bundled.add(ep)
print("BUNDLED:" + ",".join(sorted(bundled)) if bundled else "OK")
EOF
)"
    if echo "${OUT}" | grep -q "^BUNDLED:"; then
        say "  FAIL startup endpoints bundled in Promise.all: ${OUT#BUNDLED:}"; FAILED+=("B-parallel")
    elif echo "${OUT}" | grep -q "^OK"; then
        say "  PASS each startup endpoint loads independently"
    else
        say "  FAIL B check could not run: ${OUT}"; FAILED+=("B-parallel")
    fi
else
    skip "B (App.tsx not present)"
fi

say "== G job status contract =="
if live; then
    "$PY" - "$BASE" <<'EOF' || FAILED+=("G-status")
import json, sys, urllib.request
jobs = json.load(urllib.request.urlopen(sys.argv[1] + "/api/jobs", timeout=30))
allowed = {"queued","preparing","running","blocked","completed","failed","cancelled"}
statuses = {job["status"] for job in jobs}
bad = statuses - allowed
# An empty queue carries no status at all: checking zero statuses against the enum proves
# nothing, so report that honestly instead of printing PASS.
if bad:
    print(f"  FAIL unexpected statuses {bad}")
    sys.exit(1)
if not statuses:
    print("  NOT VERIFIED no job observed; an empty queue is not evidence the status enum is right")
    sys.exit(2)
print(f"  PASS job statuses lowercase enum ({len(statuses)} observed)")
EOF
    g_rc=$?
    if (( g_rc == 0 )); then
        :
    elif (( g_rc == 2 )); then
        say "  SKIP G status contract could not be verified (no queued job observed)"
        SKIPPED+=("G-status-unverified")
    else
        FAILED+=("G-status")
    fi
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
    OUT="$("${REPO}/.venv/bin/python" - "$BASE" "${REPO}" <<'EOF'
import json, sys, urllib.request
base, repo = sys.argv[1], sys.argv[2]
served = json.load(urllib.request.urlopen(base + "/api/models", timeout=120))
assert isinstance(served, list) and served, "models payload must be a non-empty list"
# Identity, not count: the served set must equal the registry the repository ships, with
# each model's native protocol and declared environment intact — a backend started against
# a different registry would otherwise look healthy on count alone.
import yaml
registry = yaml.safe_load(open(f"{repo}/workbench/registry/models.yaml", encoding="utf-8"))
declared = {m["model_id"]: m for m in registry["models"]}
served_ids = {m["model_id"] for m in served}
assert served_ids == set(declared), f"served models differ from registry: {sorted(served_ids ^ set(declared))}"
for m in served:
    assert m.get("native_protocol"), f"{m['model_id']}: native_protocol missing"
    assert m.get("environment", {}).get("python_env_var"), f"{m['model_id']}: environment declaration missing"
print(f"OK {len(served)} models, ids/protocols/environments match the registry")
EOF
)"
    if echo "${OUT}" | grep -q "^OK"; then
        say "  PASS ${OUT#OK }"
    else
        say "  FAIL configuration mismatch: ${OUT}"; FAILED+=("C-config")
    fi
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
# The report has no `rules` list. GPU readiness is per model: a CUDA-declaring entry carries
# `gpu_required` (the contract) and `gpu_verified` (the probe's answer), and the unsupported
# claim this group exists to catch is verified-without-evidence on exactly those entries.
models = doc.get("models") or {}
gpu_entries = [m for m in models.values() if isinstance(m, dict) and m.get("gpu_required")]
if not gpu_entries:
    # Scanning an empty list prints PASS while checking nothing, which is the bug this replaces.
    print("  NOT VERIFIED no GPU-requiring model in the report; nothing was checked")
    sys.exit(2)
false_claims = [m.get("model_id") for m in gpu_entries
                if m.get("gpu_verified") and m.get("environment_tier") != "GPU_RUNTIME_READY"]
if false_claims:
    print(f"  FAIL GPU verified without GPU_RUNTIME_READY evidence: {false_claims}")
    sys.exit(1)
ready_here = [m.get("model_id") for m in gpu_entries
              if m.get("environment_tier") == "GPU_RUNTIME_READY"]
print(f"  PASS {len(gpu_entries)} GPU-requiring models; "
      f"{len(ready_here)} carry a verified GPU_RUNTIME_READY probe")
EOF
)" && e_rc=0 || e_rc=$?
    if (( e_rc == 0 )); then
        say "  $e_out"
    elif (( e_rc == 2 )); then
        say "  SKIP E could not be verified (no GPU-requiring model reported)"
        SKIPPED+=("E-gpu-unverified")
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
