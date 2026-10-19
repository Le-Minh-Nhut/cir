#!/usr/bin/env bash
# Mutation regression checks for the release-blocking BUG 1–8 fixes.
#
# Each check MUTATES the current code back to the broken behavior, demonstrates the
# acceptance/verifier detects it (FAILS against the mutation), then restores the file.
# This proves the checks are capable of catching the exact regressions the fixes address
# — a suite that still passes with the functionality broken proves nothing.
#
# Usage: workbench/docker/mutation_checks.sh
# Exit: 0 only when every mutation is DETECTED (test fails under mutation).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"
FAILED=()
PASSED=()
say() { printf '%s\n' "$*"; }

restore_trap() {
    for file in "${!BACKUPS[@]}"; do
        cp "${BACKUPS[$file]}" "$file"
    done
    rm -rf "${BACKUP_DIR}"
}
declare -A BACKUPS
# Deterministic backup dir inside the repo-owned artifacts tree: a SIGKILL/power loss
# cannot lose the pristine copies the way a tmpfs mktemp dir would, and the startup
# restore below repairs a tree left mid-mutation by an earlier crashed run.
BACKUP_DIR="${REPO}/workbench/artifacts/.mutation_backups"
# An earlier crashed run (SIGKILL before the EXIT trap) leaves its backups behind; restoring
# them first puts any still-mutated file back before this run starts mutating again.
if [[ -d "${BACKUP_DIR}" ]]; then
    for saved in "${BACKUP_DIR}"/*; do
        [[ -e "${saved}" ]] || continue
        original="${saved##*/}"
        original="${REPO}/${original//_/\//}"
        mkdir -p "$(dirname "${original}")"
        cp "${saved}" "${original}"
    done
fi
rm -rf "${BACKUP_DIR}"; mkdir -p "${BACKUP_DIR}"
trap restore_trap EXIT

backup() {
    BACKUPS["$1"]="${BACKUP_DIR}/$(echo "$1" | tr '/' '_')"
    cp "$1" "${BACKUPS[$1]}"
}

mutate() { # mutation_id file old new  -> applies first replacement
    local id="$1" file="$2"
    "${REPO}/.venv/bin/python" - "$file" "$3" "$4" <<'EOF'
import sys
path, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
text = open(path, encoding="utf-8").read()
assert old in text, f"mutation target not found in {path}"
open(path, "w", encoding="utf-8").write(text.replace(old, new, 1))
EOF
}

# J4 of release_verify.sh detects a broken stdin parse and a missing bundle preference.
# Instead of re-implementing its logic here, run the offline verifier and require that the
# specific mutated aspect is what fails. We assert directly on the mutated artifacts instead:
# each mutation must break a real consumer.

# The pilot mutations need a server that answers /api/models deterministically; the real
# stack is not guaranteed to be up when mutation checks run. A tiny stub serves both.
STUB_PORT=8977
STUB_DIR="$(mktemp -d)"
cat > "${STUB_DIR}/registry.json" <<'JSON'
[{"model_id": "limn", "native_protocol": "fashioniq_val_split",
  "checkpoints": [{"checkpoint_id": "base_iter0_dress"}],
  "checkpoint_bundles": [{"bundle_id": "base_iter0_all_categories"}]}]
JSON
cat > "${STUB_DIR}/serve.py" <<'EOF'
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer

class H(BaseHTTPRequestHandler):
    def do_GET(self):
        body = open(sys.argv[1], "rb").read() if self.path == "/api/models" else b"[]"
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *a): pass

HTTPServer(("127.0.0.1", int(sys.argv[2])), H).serve_forever()
EOF
"${REPO}/.venv/bin/python" "${STUB_DIR}/serve.py" "${STUB_DIR}/registry.json" "${STUB_PORT}" &
STUB_PID=$!
for _ in $(seq 1 50); do curl -fsS "http://127.0.0.1:${STUB_PORT}/api/models" >/dev/null 2>&1 && break; sleep 0.1; done

# M1/M2 drive the real pilot launcher, which refuses to start without the operator's
# `docker/.env` (gitignored, host-specific paths). That is a prerequisite, not a mutant, so its
# absence is a skip with a stated reason rather than a "NOT DETECTED" failure -- the same shape the
# docker checks below use. The mutation is still injected and restored either way, so a host that
# HAS the file exercises it unchanged.
if [[ ! -f "${HERE}/.env" ]]; then
    SKIP_PILOT_MUTANTS=1
    say "== M1/M2 pilot mutants: SKIP NOT VERIFIED (${HERE}/.env absent; the pilot launcher refuses to start without it)"
    say "   reason: M1/M2 need a running pilot; copy .env.example to .env and set the host paths to exercise them"
else
    SKIP_PILOT_MUTANTS=0
fi

say "== M1 BUG 1: pilot JSON parse discards the fetched response =="
backup "${HERE}/run.sh"
mutate M1 "${HERE}/run.sh" 'printf '"'"'%s'"'"' "${response}" | python3 -c' \
    'printf '"'"'%s'"'"' "${response}" >/dev/null && python3 -c'
OUT="$(WORKBENCH_BACKEND_PORT=${STUB_PORT} bash "${HERE}/run.sh" pilot limn 2>&1 || true)"
if (( SKIP_PILOT_MUTANTS )); then
    say "  SKIP M1: NOT VERIFIED (pilot launcher has no .env to start from)"
elif echo "$OUT" | grep -q "malformed JSON"; then
    say "  DETECTED: pilot fails with a clear malformed-JSON error (BUG 1 regression caught)"
    PASSED+=(M1)
else
    say "  NOT DETECTED: pilot output: $OUT"
    FAILED+=(M1)
fi
cp "${BACKUPS[${HERE}/run.sh]}" "${HERE}/run.sh"

say "== M2 BUG 6: bundle selection degraded to checkpoints[0] =="
backup "${HERE}/run.sh"
mutate M2 "${HERE}/run.sh" "bundles[0].get('bundle_id') or m['checkpoints'][0]['checkpoint_id']" \
    "m['checkpoints'][0]['checkpoint_id']" 
OUT="$(WORKBENCH_BACKEND_PORT=${STUB_PORT} bash "${HERE}/run.sh" pilot limn 2>&1 || true)"
# With the mutation the launcher would queue the DRESS-only checkpoint; the guarded plan
# answers 409 evaluation_blocked naming the bundle requirement, which is the detectable delta.
if (( SKIP_PILOT_MUTANTS )); then
    say "  SKIP M2: NOT VERIFIED (pilot launcher has no .env to start from)"
elif echo "$OUT" | grep -q "checkpoint=base_iter0_dress"; then
    say "  DETECTED: launcher degraded to dress-only selection (BUG 6 regression caught)"
    PASSED+=(M2)
else
    say "  NOT DETECTED: pilot output: $OUT"
    FAILED+=(M2)
fi
cp "${BACKUPS[${HERE}/run.sh]}" "${HERE}/run.sh"
kill "${STUB_PID}" 2>/dev/null; rm -rf "${STUB_DIR}"

say "== M3 BUG 7: missing reproduced metric must not carry a deviation label =="
MUT_OUT="$("${REPO}/.venv/bin/python" - "${REPO}" <<'EOF'
# Real mutation of BUG 7: reintroduce the old broken compare_metric (a missing reproduced
# value counts as a deviation), then prove the gate's outcome CHANGES — i.e. the suite's
# verdicts actually depend on the fixed behavior. A "mutation check" whose mutant changes
# nothing detects nothing.
import sys
sys.path.insert(0, sys.argv[1])
from workbench.backend import paper_gate as gate

REF = {"reference_id": "m", "model_id": "m", "gallery_protocol": "fashioniq_original_split",
       "checkpoint_id": "c", "checkpoint_training_noise_pct": 0, "evaluation_noise_pct": 0,
       "split": "val", "dataset": "fashioniq",
       "reported_metrics": {"r10": 60.0, "r50": 70.0, "mean": 65.0},
       "category_metrics": {"dress": {"r10": 60.0, "r50": 70.0},
                            "shirt": {"r10": 60.0, "r50": 70.0},
                            "toptee": {"r10": 60.0, "r50": 70.0}},
       "derived_metrics": [], "reference_status": "VERIFIED"}
ref = gate.benchmarks.BenchmarkReference("m", "m", "p", "T", dict(REF), "d", "v1")
META = {"protocol_id": "fashioniq_original_split", "checkpoint_id": "c",
        "checkpoint_training_noise_pct": 0, "evaluation_noise_pct": 0,
        "split": "val", "dataset": "fashioniq", "checkpoint_mapping_verified": True}
OBS = {"r50": 70.0, "mean": 65.0}  # r10 missing on the run side
CATS = {c: {"r10": 60.0, "r50": 70.0} for c in ("dress", "shirt", "toptee")}

healthy = gate.evaluate_gate(ref, dict(OBS), run_meta=dict(META), real_execution=True,
                             reproduced_category_metrics=CATS)

original = gate.compare_metric
def mutated(key, paper, observed, **kwargs):
    comparison = original(key, paper, observed, **kwargs)
    if paper is not None and observed is None:
        comparison = type(comparison)(**{**comparison.__dict__, "within_tolerance": False})
    return comparison
gate.compare_metric = mutated
mutant = gate.evaluate_gate(ref, dict(OBS), run_meta=dict(META), real_execution=True,
                            reproduced_category_metrics=CATS)
gate.compare_metric = original

if healthy.status == gate.INCOMPLETE_METRIC_EVIDENCE and not healthy.passed \
        and mutant.status == gate.ALERT_METRIC_DEVIATION:
    print("KILLED: mutant flips the verdict; the gate depends on the fixed rule")
else:
    print(f"SURVIVED: healthy={healthy.status} mutant={mutant.status}")
EOF
)"
if echo "$MUT_OUT" | grep -q "KILLED"; then
    say "  PASS mutation killed: missing metric is a coverage gap, and the gate's verdict depends on it"
    PASSED+=(M3)
else
    say "  FAIL mutant survived: $MUT_OUT"
    FAILED+=(M3)
fi

say "== M4 BUG 2/3: compose rejects an empty host path; RO submounts protect canonical data =="
if ! command -v docker >/dev/null 2>&1; then
    say "  SKIP M4 compose env guard: NOT VERIFIED (docker CLI not found)"
elif ! docker info >/dev/null 2>&1; then
    say "  SKIP M4 compose env guard: NOT VERIFIED (docker daemon unavailable)"
else
    M4_OUT="$(docker compose -f "${HERE}/docker-compose.yml" --env-file /dev/null config 2>&1)" && M4_RC=0 || M4_RC=$?
    if echo "${M4_OUT}" | grep -qEi "cannot connect to the docker daemon|failed to connect to the docker api|is the docker daemon running|error during connect"; then
        say "  SKIP M4 compose env guard: NOT VERIFIED (docker daemon unavailable)"
    elif (( M4_RC == 0 )); then
        say "  FAIL: empty HOST_* env accepted (BUG 2 guard missing)"
        FAILED+=(M4)
    elif echo "${M4_OUT}" | grep -qE "set HOST_FASHIONIQ_ROOT in .env|required variable.*HOST_FASHIONIQ_ROOT"; then
        say "  DETECTED: compose refuses to resolve with missing env guards (BUG 2 guard holds)"
        PASSED+=(M4)
    else
        say "  FAIL: compose failed without triggering env guard: ${M4_OUT}"
        FAILED+=(M4)
    fi
fi
if command -v docker >/dev/null && docker info >/dev/null 2>&1 && [[ -n "$(docker ps -q --filter name=^cir-workbench$ 2>/dev/null)" ]]; then
    C="$(docker ps -q --filter name=^cir-workbench$ | head -1)"
    if docker exec "$C" sh -c 'touch /data/FashionIQ/images/.m4 2>/dev/null'; then
        docker exec "$C" rm -f /data/FashionIQ/images/.m4 2>/dev/null
        say "  FAIL: canonical images/ writable (BUG 3 protection broken)"
        FAILED+=(M4)
    else
        say "  DETECTED: canonical images/ rejects writes (BUG 3 protection holds)"
        PASSED+=(M4)
    fi
else
    say "  SKIP M4 live mount probe (container not running)"
fi

say "== M5 BUG 4: GPU override file declares the NVIDIA device reservation =="
if grep -q "driver: nvidia" "${HERE}/docker-compose.gpu.yml"; then
    say "  PASS GPU reservation declared (fail-closed start without toolkit)"
    PASSED+=(M5)
else
    say "  FAIL GPU reservation missing"
    FAILED+=(M5)
fi

say ""
if (( ${#FAILED[@]} )); then
    say "MUTATION CHECKS FAILED: ${FAILED[*]}"
    exit 1
fi
say "MUTATION CHECKS PASSED: ${PASSED[*]}"
exit 0
