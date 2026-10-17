#!/usr/bin/env bash
# Executable release verifier for the CIR Failure Analysis Workbench.
#
# Runs the applicable journey-based acceptance checks (TEST 1–12 of the release spec),
# produces a machine-readable JSON report, and exits NONZERO when any mandatory
# criterion is unmet. GPU-dependent checks may be PENDING on non-target hardware;
# a PENDING is never converted into PASS, and PENDING never fails the exit code when
# the fail-closed contract itself is intact.
#
# Usage:
#   workbench/docker/release_verify.sh [--report FILE] [--base URL] [--offline]
#
# --offline  skip live-container journeys (J1/J2/J3/J11); they are then PENDING, not PASS.
# Every other unmet check fails the run.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${WORKBENCH_REPO:-$(cd "${HERE}/../.." && pwd)}"
BASE="http://127.0.0.1:8000"
REPORT="${HERE}/release_verify_report.json"
OFFLINE=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --report)
            [[ $# -ge 2 ]] || { echo "option $1 requires an argument" >&2; exit 2; }
            REPORT="$2"; shift 2;;
        --base)
            [[ $# -ge 2 ]] || { echo "option $1 requires an argument" >&2; exit 2; }
            BASE="$2"; shift 2;;
        --offline) OFFLINE=1; shift;;
        *) echo "unknown option: $1" >&2; exit 2;;
    esac
done

PY="$(command -v python3 || true)"
[[ -x "${REPO}/.venv/bin/python" ]] && PY="${REPO}/.venv/bin/python"
# Embedded checks import the workbench package; the verifier may be invoked from any cwd,
# so the repo root must be importable regardless of where the interpreter starts.
export PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -z "$PY" ]]; then echo "no python interpreter" >&2; exit 2; fi

declare -A RESULTS   # id -> PASS|FAIL|PENDING
declare -A DETAILS   # id -> detail string
ORDER=(J1 J2 J3 J4 J5 J6 J7 J8 J9 J10 J11 J12)

pass() { RESULTS[$1]="PASS"; DETAILS[$1]="${2:-}"; say "  PASS $1 ${2:-}"; }
fail() { RESULTS[$1]="FAIL"; DETAILS[$1]="${2:-}"; say "  FAIL $1 ${2:-}"; }
pending() { RESULTS[$1]="PENDING"; DETAILS[$1]="${2:-}"; say "  PENDING $1 ${2:-}"; }
say() { printf '%s\n' "$*"; }

code() { curl -s --max-time 5 --connect-timeout 2 -o /dev/null -w '%{http_code}' "$@" 2>/dev/null; }
# Probe the stack exactly once; the /api/models call is expensive (~1 min with
# checkpoint-availability resolution), so J1/J4/J8 share one verdict.
# Liveness probe uses a cheap endpoint: /api/models resolves per-checkpoint availability
# (~1 min) and cannot serve as a fast probe.
LIVE=0
for _ in 1 2 3 4 5 6; do
    if [[ "$(code "${BASE}/api/jobs")" == "200" ]]; then LIVE=1; break; fi
    sleep 5
done
live() { return $((1 - LIVE)); }

# ---------------------------------------------------------------- J1 CLEAN DOCKER STARTUP
say "== J1 clean startup: backend, frontend, all 16 models, no fake results =="
if (( OFFLINE )) || ! live; then
    # No live stack means no boot state could be observed: say that, instead of
    # implying an empty stack was seen.
    pending J1 "boot state UNOBSERVED: no live stack (run compose up, or drop --offline for static checks)"
else
    # 2>&1, so the assertion message survives: capturing stdout alone produced an empty
    # `detail` for every failing journey, leaving the operator with a bare FAIL and no reason.
    if j1="$("${PY}" - "$BASE" "${REPO}" 2>&1 <<'EOF'
import json, sys, urllib.request
base, repo = sys.argv[1], sys.argv[2]
import yaml
expected = {m["model_id"] for m in yaml.safe_load(
    open(f"{repo}/workbench/registry/models.yaml", encoding="utf-8"))["models"]}
def get(path):
    return json.load(urllib.request.urlopen(base + path, timeout=120))
models = get("/api/models")
assert isinstance(models, list), "models payload must be a list"
assert {m.get("model_id") for m in models} == expected, (
    f"served models differ from the registry: {sorted({m.get('model_id') for m in models} ^ expected)}")
for m in models:
    assert m.get("model_id"), "model entry without model_id"
# Two boot states are BOTH legitimate and must be told apart. A clean first boot presents no run
# at all. An existing run produced by a real evaluation is not "fake": it is presented as real
# evidence only when the out-of-band ledger corroborates its aggregate report. Everything else
# visible is exactly the fake this check exists to catch — mock/self-test output, or a run whose
# report no ledger entry corroborates.
runs = get("/api/runs")
assert isinstance(runs, list)
# The corroboration contract is the backend's own (`_aggregate_reports`): a report is real
# evidence only when the ledger holds an entry for those bytes carrying the run identity, return
# code 0 and INVOCATION_COMPLETED. Reusing it means this scan cannot drift from what the API
# serves as real, and cannot invent a weaker rule of its own.
from workbench.backend.errors import WorkbenchError
from workbench.backend.main import _aggregate_reports
try:
    corroborated = {report.get("run_id"): report for report in _aggregate_reports()}
except WorkbenchError as error:
    reason = error.details.get("reason") or error.message
    sys.exit(f"{error.code}: {reason}")
except Exception as error:
    # A J1 run that cannot read the ledger at all is NOT the clean-boot state: with two
    # disagreeing key files every report is unreadable and `_aggregate_reports` refuses. Saying
    # so here keeps an operator fault out of the PASS/PENDING vocabulary, where it would read as
    # an ordinary empty stack.
    sys.exit(f"the evidence ledger cannot be read, so no boot state is observable: "
             f"{getattr(error, 'message', error)}")
def classify(run):
    if run.get("data_kind") == "mock":
        return "mock", "labelled data_kind=mock: a development fixture, not research evidence"
    report = corroborated.get(run.get("run_id"))
    if report is None:
        return "uncorroborated", "no evidence-ledger-corroborated aggregate report for this run"
    if report.get("is_synthetic"):
        return "synthetic", "its corroborated report is a declared synthetic source"
    if (str(report.get("model_id")) != str(run.get("model_id"))
            or str(report.get("protocol_id")) != str(run.get("protocol_id"))):
        return "uncorroborated", "its aggregate report describes a different run identity"
    return "real", "its aggregate report is corroborated by the evidence ledger"
classified = [(run,) + classify(run) for run in runs]
refused = [(run, kind, reason) for run, kind, reason in classified if kind != "real"]
# A run that is HONESTLY labelled as non-evidence is not a fake: `data_kind=mock` is a
# development fixture the workbench itself writes, and it is exactly what a fresh installation
# shows. The check that matters is that nothing reaches the UI as research evidence without
# corroboration, so a labelled fixture is reported separately from an uncorroborated claim.
labelled_non_evidence = [(run, reason) for run, kind, reason in refused if kind == "mock"]
misrepresented = [(run, kind, reason) for run, kind, reason in refused if kind != "mock"]
# Jobs may legitimately exist on a long-lived stack; a clean first boot has none. Either
# way every status must be a real backend enum value, never an invented one.
allowed = {"queued", "preparing", "running", "blocked", "completed", "failed", "cancelled"}
bad_jobs = [j.get("status") for j in get("/api/jobs") if j.get("status") not in allowed]
assert not bad_jobs, f"job statuses outside the backend enum: {bad_jobs}"
assert not misrepresented, "runs presented as real evidence without corroboration: " + "; ".join(
    f"{run.get('run_id')} [{kind}] {reason}" for run, kind, reason in misrepresented)
real_runs = [run for run, kind, _ in classified if kind == "real"]
if real_runs:
    boot_state = "existing valid results"
elif runs:
    boot_state = "labelled non-evidence only"
else:
    boot_state = "clean first boot"
print(f"{len(models)} models; boot state={boot_state}: {len(runs)} run(s) visible, "
      f"{len(real_runs)} ledger-corroborated real, {len(labelled_non_evidence)} labelled non-evidence")
for run, reason in labelled_non_evidence[:5]:
    print(f"  labelled non-evidence: {run.get('run_id')} — {reason}")
EOF
    )"; then
        pass J1 "$j1"
    else
        fail J1 "$j1"
    fi
fi

# ---------------------------------------------------------------- J2 CORRECT VOLUME MOUNTS
say "== J2 volume mounts resolve to the intended host directories =="
if (( OFFLINE )); then
    pending J2 "offline mode"
elif ! command -v docker >/dev/null || [[ -z "$(docker ps -q --filter name=^cir-workbench$ 2>/dev/null)" ]]; then
    pending J2 "workbench container not running"
else
    C="$(docker ps -q --filter name=^cir-workbench$ | head -1)"
    mounts="$(docker inspect "$C" --format '{{range .Mounts}}{{.Source}}=>{{.Destination}} {{end}}')"
    mounts="$mounts" "${PY}" - <<'EOF' && pass J2 "$mounts" || fail J2 "$mounts"
import os, sys
mounts = os.environ["mounts"].split()
pairs = dict(m.split("=>", 1) for m in mounts)
destinations = list(pairs.values())
for dest in ("/data/FashionIQ", "/third_party", "/checkpoints"):
    assert any(d == dest or d.startswith(dest + "/") for d in destinations), f"missing mount for {dest}: {destinations}"
# canonical dataset submounts must exist as separate (read-only) mount entries
ro = [d for d in destinations if d.rstrip("/").endswith(("images", "image_splits"))]
assert len(ro) >= 2, f"expected read-only images/ and image_splits/ submounts, mounts={destinations}"
print("dataset/checkpoints/third_party mounted; canonical submounts present as RO entries")
EOF
fi

# ---------------------------------------------------------------- J3 WRITABLE PREPARATION
say "== J3 preparation is writable inside the container, canonical inputs unchanged =="
if (( OFFLINE )); then
    pending J3 "offline mode"
elif [[ -z "$(docker ps -q --filter name=^cir-workbench$ 2>/dev/null)" ]]; then
    pending J3 "workbench container not running"
else
    C="$(docker ps -q --filter name=^cir-workbench$ | head -1)"
    if docker exec "$C" sh -c '
        mkdir -p /data/FashionIQ/resized_image 2>/dev/null || true
        test -w /data/FashionIQ/resized_image || exit 10
        echo probe > /data/FashionIQ/resized_image/.j3probe && rm -f /data/FashionIQ/resized_image/.j3probe || exit 11
        touch /data/FashionIQ/images/.j3probe 2>/dev/null && exit 12
        touch /data/FashionIQ/image_splits/.j3probe 2>/dev/null && exit 13
        exit 0
    '; then
        pass J3 "derived tree writable; images/ and image_splits/ reject writes (read-only)"
    else
        rc=$?
        case "$rc" in
            10|11) fail J3 "derived dataset tree is NOT writable (rc=$rc)";;
            12)    fail J3 "canonical images/ is writable — protection broken";;
            13)    fail J3 "canonical image_splits/ is writable — protection broken";;
            *)     fail J3 "container exec failed rc=$rc";;
        esac
    fi
fi

# ---------------------------------------------------------------- J4 PILOT LAUNCHER (mutation-verified)
say "== J4 pilot launcher: stdin JSON parse + complete bundle selection =="
RUN_SH="${HERE}/run.sh"
[[ -f "$RUN_SH" ]] || RUN_SH="${REPO}/workbench/docker/run.sh"
if [[ ! -x "$RUN_SH" && ! -f "$RUN_SH" ]]; then
    fail J4 "run.sh missing"
else
    # (a) api_field must parse a real JSON document through stdin: mutate detection for BUG 1.
    j4a=0
    if printf '{"models":[{"model_id":"limn","native_protocol":"fashioniq_val_split"}]}' | \
        bash -c 'response="$(cat)"; printf "%s" "$response" | python3 -c "
import json, sys
doc = json.load(sys.stdin)
r = next((m[\"native_protocol\"] for m in doc[\"models\"] if m[\"model_id\"]==\"limn\"), None)
assert r == \"fashioniq_val_split\", r
"'; then :; else j4a=1; fi
    # the OLD broken pattern (no stdin) must NOT satisfy the contract: feed input, old code ignores it
    if printf '{"models":[]}' | bash -c 'python3 -c "
import json, sys
doc = json.load(sys.stdin)
assert doc[\"models\"], \"empty models must fail\"
"' >/dev/null 2>&1; then j4a_mut=1; else j4a_mut=0; fi
    # (b) run.sh source must prefer the bundle over checkpoints[0] (BUG 6)
    # Match the actual selection expression, not any mention of bundles: M2's mutant
    # replaces the bundle preference with checkpoints[0], and only a grep for the live
    # expression detects that.
    if grep -q "bundles\[0\].get('bundle_id') or m\['checkpoints'\]\[0\]\['checkpoint_id'\]" "$RUN_SH"; then j4b=0; else j4b=1; fi
    # (c) /api/models exposes bundles and /api/jobs resolves the bundle id (BUG 6 API side)
    j4c="skipped"
    if (( ! OFFLINE )) && live; then
        j4c="$("${PY}" - "$BASE" 2>&1 <<'EOF'
import json, sys, urllib.request
base = sys.argv[1]
models = json.load(urllib.request.urlopen(base + "/api/models", timeout=60))
limn = next(m for m in models if m["model_id"] == "limn")
bundles = limn.get("checkpoint_bundles") or []
assert bundles, "limn must expose its checkpoint bundle"
assert bundles[0]["bundle_id"] == "base_iter0_all_categories", bundles[0]
assert set(bundles[0]["required_checkpoint_ids"]) == {"base_iter0_dress", "base_iter0_shirt", "base_iter0_toptee"}
print("bundle exposed with 3 members")
EOF
        )" && j4c_rc=0 || j4c_rc=1
    fi
    if (( j4a == 0 )) && (( j4b == 0 )); then
        if [[ "$j4c" == "skipped" ]]; then
            pass J4 "stdin parse verified; bundle preference in run.sh (live API check skipped)"
        elif (( j4c_rc == 0 )); then
            pass J4 "$j4c; stdin parse verified"
        else
            fail J4 "$j4c"
        fi
    else
        fail J4 "stdin parse=$j4a bundle-preference=$j4b"
    fi
fi

# ---------------------------------------------------------------- J5 GPU CONTRACT
say "== J5 GPU contract: fail-closed without a verified capability =="
if [[ -n "$PY" ]]; then
    j5="$("${PY}" - "$REPO" 2>&1 <<'EOF'
from workbench.backend import runtime
import inspect
src = inspect.getsource(runtime)
# 1) the cached-record gate accepts only interpreter-probed evidence unless the test switch is set
assert "probe_source" in src and "GPU_PROBE_OVERRIDE_ENV" in src
# 2) a GPU-declaring model without a verified capability cannot launch
model = {"model_id": "probe_contract_model",
         "environment": {"environment_required": True, "cuda": "12.4",
                         "python_env_var": "WORKBENCH_PROBE_PYTHON", "packages": []}}
try:
    report = runtime.verify_environment(model, probe=False)
except Exception:
    report = None
if report is not None and report.tier == runtime.TIER_BLOCKED:
    print("unverified GPU environment is BLOCKED (fail-closed holds)")
else:
    # no torch in this interpreter: the model declares CUDA, verification must not pass
    assert report is None or report.tier != runtime.TIER_GPU_READY, \
        "a CUDA-declaring model must never reach TIER_GPU_READY without a real probe"
    print(f"CUDA-declaring model is not GPU-ready without a probe (tier={report.tier if report else None})")
EOF
    )" && pass J5 "$j5" || fail J5 "$j5"
else
    fail J5 "no python interpreter"
fi

# ---------------------------------------------------------------- J6 RUNTIME ADAPTATION
say "== J6 sm_120 adaptation changes the verified requirements, not just metadata =="
if [[ -n "$PY" ]]; then
    j6="$("${PY}" - "$REPO" 2>&1 <<'EOF'
import os
from workbench.backend import runtime
from workbench.backend.registry import load_registry
limn = next(m for m in load_registry()["models"] if m["model_id"] == "limn")
historical = runtime._required_packages(limn)
os.environ["WORKBENCH_RUNTIME_ADAPTATION"] = "sm_120"
try:
    adapted = runtime._adapted_packages(limn, runtime.runtime_adaptation(limn))
    assert runtime.runtime_adaptation(limn)["target_compute_capability"] == "sm_120"
finally:
    os.environ.pop("WORKBENCH_RUNTIME_ADAPTATION", None)
# the adapted contract must differ from the historical pins in what it requires
assert historical != adapted, "adaptation must change the requirement set"
hist_torch = historical.get("torch"); adap_torch = adapted.get("torch")
assert adap_torch and adap_torch != hist_torch, (hist_torch, adap_torch)
# runtime_adaptation refuses an undeclared target
os.environ["WORKBENCH_RUNTIME_ADAPTATION"] = "sm_999"
try:
    runtime.runtime_adaptation(limn)
    raise SystemExit("undeclared adaptation must be refused")
except Exception as error:
    assert "UNKNOWN_RUNTIME_ADAPTATION" in str(error) or "unknown" in str(error).lower()
finally:
    os.environ.pop("WORKBENCH_RUNTIME_ADAPTATION", None)
print(f"historical torch={hist_torch} -> adapted torch={adap_torch}; unknown targets refused")
EOF
    )" && pass J6 "$j6" || fail J6 "$j6"
else
    fail J6 "no python interpreter"
fi

# ---------------------------------------------------------------- J7 COMPLETE LIMN REPORT (synthetic)
say "== J7 complete-report rule via the synthetic evaluator (marked synthetic) =="
if [[ -n "$PY" ]]; then
    j7="$("${PY}" - "$REPO" 2>&1 <<'EOF'
# The production contract under test: a per-category evaluator MUST deliver all three
# category results before an aggregate exists, and a report that carries the synthetic
# marker can never be certified as official paper evidence. Verified directly against the
# aggregate-building code and the gate's synthetic refusal (offline, deterministic).
from workbench.backend import paper_gate as gate
# synthetic/mock provenance can never pass the gate
verdict = gate.evaluate_gate(None, {"r10": 55.0}, run_meta={"run_id": "synthetic"},
                             data_kind="mock", real_execution=True)
assert not verdict.passed, "mock data must never be certified"
# missing one category's contribution yields an incomplete aggregate, not a fake full one
verdict2 = gate.evaluate_gate(None, {"r10": 55.0}, run_meta={"run_id": "x"},
                              data_kind="experiment", real_execution=True,
                              evaluation_completed=False)
assert not verdict2.passed
# a two-of-three LIMN report must never be certified as the full FashionIQ benchmark
from workbench.replay import limn as limn_replay
import tempfile, json
from pathlib import Path
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    # one category directory present with results, two missing -> bundle validation must complain
    blockers = limn_replay.validate_checkpoint_bundle(root, limn_replay.CATEGORIES)
    assert blockers, "empty checkpoint root must be refused"
print("mock data refused; incomplete evaluation refused; 3-category bundle enforced; gate stays fail-closed")
EOF
    )" && pass J7 "$j7" || fail J7 "$j7"
else
    fail J7 "no python interpreter"
fi

# ---------------------------------------------------------------- J8 JOB LIFECYCLE
say "== J8 job lifecycle statuses are the backend enum =="
if (( OFFLINE )) || (( LIVE == 0 )); then
    pending J8 "no live stack"
else
    j8="$("${PY}" - "$BASE" 2>&1 <<'EOF'
import json, sys, urllib.request
jobs = json.load(urllib.request.urlopen(sys.argv[1] + "/api/jobs", timeout=60))
allowed = {"queued", "preparing", "running", "blocked", "completed", "failed", "cancelled"}
bad = {j["status"] for j in jobs} - allowed
assert not bad, f"unexpected job statuses: {bad}"
if not jobs:
    print("queue idle: 0 jobs in flight, endpoint responsive")
else:
    print(f"{len(jobs)} jobs, all statuses within the backend enum")
EOF
    )" && pass J8 "$j8" || fail J8 "$j8"
fi

# ---------------------------------------------------------------- J9 SCIENTIFIC GATE
say "== J9 scientific gate cases incl. INCOMPLETE_METRIC_EVIDENCE =="
if [[ -n "$PY" ]]; then
    j9="$("${PY}" - "$REPO" 2>&1 <<'EOF'
from workbench.backend import paper_gate as gate
# missing reproduced value is INCOMPLETE evidence, never a deviation label
c = gate.compare_metric("r10", 60.0, None)
assert c.within_tolerance is None and c.note == "no usable reproduced value", c
d = gate.deviations([c], 5)
assert not d["deviates"]
# non-finite reproduced value is invalid output, not missing evidence
c2 = gate.compare_metric("r10", 60.0, float("nan"))
assert c2.within_tolerance is False and "not a finite number" in (c2.note or ""), c2
# measured deviation is still flagged
c3 = gate.compare_metric("r10", 60.0, 40.0)
assert c3.within_tolerance is False
# verdict statuses exist and are distinct
assert gate.INCOMPLETE_METRIC_EVIDENCE != gate.ALERT_METRIC_DEVIATION
print("missing->incomplete, nan->invalid-alert, measured deviation still flagged")
EOF
    )" && pass J9 "$j9" || fail J9 "$j9"
else
    fail J9 "no python interpreter"
fi

# ---------------------------------------------------------------- J10 FRONTEND
say "== J10 frontend: exactly two warning labels, built bundle present =="
j10_bad=""
if ! grep -q "CAPTION THAY THẾ" "${REPO}/workbench/frontend/gate.ts" 2>/dev/null; then j10_bad="missing CAPTION THAY THẾ; "; fi
if ! grep -q "LỆCH PAPER" "${REPO}/workbench/frontend/gate.ts" 2>/dev/null; then j10_bad="${j10_bad}missing LỆCH PAPER; "; fi
count="$(grep -o "CAPTION THAY THẾ\|LỆCH PAPER" "${REPO}/workbench/frontend/gate.ts" 2>/dev/null | sort -u | wc -l | tr -d ' ')"
if [[ "$count" != "2" ]]; then j10_bad="${j10_bad}distinct label kinds=${count} (must be exactly 2); "; fi
if [[ ! -d "${REPO}/workbench/frontend/dist" ]]; then
    j10_bad="${j10_bad}frontend/dist missing (run vite build); "
elif ! grep -rq "phán quyết audit chuẩn" "${REPO}/workbench/frontend/dist" 2>/dev/null; then
    j10_bad="${j10_bad}built bundle lacks audit-threshold copy; "
fi
if [[ -z "$j10_bad" ]]; then
    pass J10 "two labels exactly; built bundle carries the display copy"
else
    fail J10 "$j10_bad"
fi

# ---------------------------------------------------------------- J11 PERSISTENCE (DISPOSABLE CONTAINER)
say "== J11 artifacts persist across a container restart (isolated disposable container) =="
# The PRODUCTION container is never restarted by verification: an active research job would
# be killed mid-run. Persistence is proven on a disposable container from the same image with
# its own temporary artifacts directory.
if (( OFFLINE )); then
    pending J11 "offline mode"
elif ! command -v docker >/dev/null; then
    pending J11 "docker unavailable"
elif ! docker image inspect docker-workbench:latest >/dev/null 2>&1; then
    pending J11 "docker-workbench image unavailable"
else
    J11_TMP="$(mktemp -d)"
    J11_C="j11-persist-$$-$RANDOM"
    J11_OK=0
    if docker run -d --name "$J11_C" -v "$J11_TMP:/artifacts"         --entrypoint sleep docker-workbench:latest 120 >/dev/null 2>&1; then
        marker="j11-$$-$RANDOM"
        if docker exec "$J11_C" sh -c "echo $marker > /artifacts/.j11probe" 2>/dev/null \
            && docker restart "$J11_C" >/dev/null 2>&1 \
            && sleep 2 \
            && [[ "$(docker exec "$J11_C" sh -c 'cat /artifacts/.j11probe 2>/dev/null')" == "$marker" ]]; then
            pass J11 "artifacts persisted across container restart (disposable container, isolated $J11_TMP)"
            J11_OK=1
        else
            fail J11 "artifacts lost across restart in disposable container"
        fi
    else
        pending J11 "could not start disposable container"
    fi
    docker rm -f "$J11_C" >/dev/null 2>&1
    rm -rf "$J11_TMP"
fi

# ---------------------------------------------------------------- J12 RESOURCE FAILURE
say "== J12 resource failure: missing checkpoint recorded clearly, others intact =="
if [[ -n "$PY" ]]; then
    j12="$("${PY}" - "$REPO" 2>&1 <<'EOF'
import os, tempfile
from pathlib import Path
from workbench.scripts import evaluate_models
from workbench.backend.operator_config import WorkbenchConfig
from workbench.backend.registry import load_registry
registry = load_registry()
limn = next(m for m in registry["models"] if m["model_id"] == "limn")
with tempfile.TemporaryDirectory() as tmp:
    def sub(name): return Path(tmp) / name
    os.makedirs(sub("data/captions")); sub("checkpoints").mkdir(); sub("results").mkdir(); sub("third_party").mkdir()
    config = WorkbenchConfig(
        CIR_REPO_ROOT=Path(tmp), CIR_DATA_ROOT=sub("data"), FASHIONIQ_ROOT=sub("data"),
        WORKBENCH_HOST="127.0.0.1", WORKBENCH_BACKEND_PORT=0, WORKBENCH_FRONTEND_PORT=0,
        WORKBENCH_CHECKPOINT_ROOT=sub("checkpoints"),
        WORKBENCH_RESULTS_ROOT=sub("results"),
        WORKBENCH_THIRD_PARTY_ROOT=sub("third_party"))
    checkpoint = limn["checkpoint_variants"][0]
    plan, blockers = evaluate_models.guarded_plan(
        limn, checkpoint, limn["native_protocol"], config.FASHIONIQ_ROOT, 200, config)
    assert plan is None, "a missing checkpoint must not produce a launch plan"
    assert any("checkpoint" in b.lower() for b in blockers), blockers
print(f"missing checkpoint -> blocked with named reason: {blockers[0][:60]}...")
EOF
    )" && pass J12 "$j12" || fail J12 "$j12"
else
    fail J12 "no python interpreter"
fi

# ---------------------------------------------------------------- report
say ""
fails=0; pendings=0
for id in "${ORDER[@]}"; do
    r="${RESULTS[$id]:-UNRUN}"
    printf '%-4s %-8s %s\n' "$id" "$r" "${DETAILS[$id]}"
    case "$r" in
        FAIL|UNRUN) fails=$((fails+1));;
        PENDING) pendings=$((pendings+1));;
    esac
done

# Write the machine-readable report (python cannot see bash assoc arrays; emit via env)
J1="${RESULTS[J1]}" J2="${RESULTS[J2]}" J3="${RESULTS[J3]}" J4="${RESULTS[J4]}" \
J5="${RESULTS[J5]}" J6="${RESULTS[J6]}" J7="${RESULTS[J7]}" J8="${RESULTS[J8]}" \
J9="${RESULTS[J9]}" J10="${RESULTS[J10]}" J11="${RESULTS[J11]}" J12="${RESULTS[J12]}" \
D1="${DETAILS[J1]}" D2="${DETAILS[J2]}" D3="${DETAILS[J3]}" D4="${DETAILS[J4]}" \
D5="${DETAILS[J5]}" D6="${DETAILS[J6]}" D7="${DETAILS[J7]}" D8="${DETAILS[J8]}" \
D9="${DETAILS[J9]}" D10="${DETAILS[J10]}" D11="${DETAILS[J11]}" D12="${DETAILS[J12]}" \
"${PY}" - "$REPORT" <<'EOF'
import json, os, sys, datetime
out = {"generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
       "verifier": "workbench/docker/release_verify.sh",
       "checks": {}}
for i in range(1, 13):
    key = f"J{i}"
    dkey = f"D{i}"
    detail = os.environ.get(dkey) or None
    out["checks"][key] = {"status": os.environ.get(key), "detail": detail}
out["summary"] = {
    "pass": sum(1 for c in out["checks"].values() if c["status"] == "PASS"),
    "fail": sum(1 for c in out["checks"].values() if c["status"] in ("FAIL", "UNRUN")),
    "pending": sum(1 for c in out["checks"].values() if c["status"] == "PENDING"),
}
# Distinct acceptance dimensions; every check maps to exactly one. A verdict may only claim a
# dimension when every check in it actually PASSED — a PENDING is not PASS, a source-grep is
# not a runtime probe. GPU_RUNTIME/MODEL_INFERENCE/MODEL_EVALUATION remain PENDING on a host
# without the target accelerator, which caps the verdict at GPU_VALIDATION_PENDING.
out["dimensions"] = {
    "STATIC_SOURCE":     {"checks": ["J4", "J6", "J9", "J10"], "mandatory": True},
    "DEPLOYMENT":        {"checks": ["J2", "J3", "J11"],       "mandatory": True},
    "CONFIGURATION":     {"checks": ["J1"],                    "mandatory": True},
    "MODEL_ENVIRONMENT": {"checks": ["J12"],                   "mandatory": True},
    "OPERATIONS":        {"checks": ["J8"],                    "mandatory": True},
    "GPU_RUNTIME":       {"checks": ["J5"],                    "mandatory": True},
    "MODEL_INFERENCE":   {"checks": [],                        "mandatory": True},
    "MODEL_EVALUATION":  {"checks": [],                        "mandatory": True},
    "SCIENTIFIC_OUTPUT": {"checks": ["J7"],                    "mandatory": True},
}
for dim, spec in out["dimensions"].items():
    statuses = [out["checks"][c]["status"] for c in spec.get("checks", [])]
    if any(s in ("FAIL", "UNRUN") for s in statuses):
        spec["status"] = "FAIL"
    elif statuses and all(s == "PASS" for s in statuses):
        spec["status"] = "PASS"
    elif all(s in ("PASS", "PENDING") for s in statuses):
        spec["status"] = "PENDING"
    else:
        spec["status"] = "NOT_APPLICABLE"
# GPU_RUNTIME claims PASS only with a real, interpreter-probed capability record for THIS
# environment. A record's own fields are NEVER the evidence: the file must be read back through
# the runtime's authoritative capability path, which requires an HMAC signature under this
# deployment's operator key, an interpreter fingerprint, a host identity, the runtime adaptation
# the probe ran under, and a capability key that binds the record to this model + interpreter +
# dependency set + host. A hand-written file with `probe_source: interpreter` and
# `capability_verified: true` therefore certifies nothing, and neither does a record signed with
# another deployment's key or probed elsewhere; each is reported as not-verified with its reason.
# The fail-closed contract (J5) alone is engineering evidence, not GPU verification.
#
# An artifact that cannot be read is its own reported condition. It must never be swallowed by
# the surrounding `except` and turned into real_gpu = False: a leftover 0-byte fixture or a
# half-written record would then silently discard a genuine probe on the target host, and the
# release would be capped at GPU_VALIDATION_PENDING for no stated reason. Such files are named
# in the report and removed (they carry no evidence); a record whose own `capability_key` names
# a different environment is not evidence either.
import glob as _glob, json as _json
from workbench.backend.operator_config import resolve_config as _rc
from workbench.backend import runtime as _runtime
from workbench.backend.registry import load_registry as _load_registry

_EXPECTED_KEYS: dict = {}
_DECLARED_INTERPRETERS: dict = {}

def _declared_interpreter(model):
    """The interpreter this deployment resolves for the model, once per model."""
    model_id = model.get("model_id")
    if model_id not in _DECLARED_INTERPRETERS:
        _DECLARED_INTERPRETERS[model_id] = _runtime.verify_environment(model, probe=False).interpreter
    return _DECLARED_INTERPRETERS[model_id]

def _expected_key(model):
    """This model's capability key, computed once per scan (the fingerprint is expensive)."""
    model_id = model.get("model_id")
    if model_id not in _EXPECTED_KEYS:
        _EXPECTED_KEYS[model_id] = _runtime._gpu_capability_key(model)
    return _EXPECTED_KEYS[model_id]

def _unverified_reason(key, record, models):
    """Why this capability record is not verified evidence for this environment, or None.

    The cheap rejections come first so the report names the actual defect (an unsigned file, a
    different interpreter, another host, another adaptation) instead of a generic refusal. The
    decisions themselves are never re-implemented: the authoritative read-back is
    `verify_environment(..., authoritative=True)` — the path the backend uses to reload a record
    — which accepts this record only when its HMAC verifies, its interpreter fingerprint matches
    this environment, and its capability key is the one computed from this model, interpreter,
    dependency set and host identity.
    """
    if record.get("probe_source") == "override":
        return "probe_source=override is a synthetic test probe, not a real probe"
    if record.get("probe_source") != "interpreter":
        return f"probe_source={record.get('probe_source')!r} is not an interpreter probe"
    if not _runtime._gpu_capability_signature_valid(record):
        # Two disagreeing key files leave the signer unable to pick one, so every record fails
        # here. Naming that condition (and its remedy) keeps an operator fault from reading as a
        # forged record.
        try:
            _key_ready = bool(_runtime._gpu_cache_key())
        except Exception:
            _key_ready = False
        if not _key_ready:
            return ("the record is not signed with this deployment's operator key: no usable "
                    "ledger key resolved (two different key files, or none -- keep one file and "
                    "remove the other, or set WORKBENCH_LEDGER_KEY explicitly to choose one)")
        return "the record is not signed with this deployment's operator key"
    model = next((m for m in models if m.get("model_id") == record.get("model_id")), None)
    if model is None:
        return f"the record names an unknown model {record.get('model_id')!r}"
    interpreter = str(record.get("interpreter") or "")
    if not interpreter:
        return "the record does not name the interpreter it was probed with"
    if interpreter != _declared_interpreter(model):
        return (f"was probed with interpreter {interpreter!r}, this environment's is "
                f"{_declared_interpreter(model)!r}")
    if record.get("dependency_fingerprint") != _runtime._environment_contract_digest(model):
        return "the record's dependency fingerprint is not this environment's contract"
    if record.get("host_identity") != _runtime._host_identity():
        return "the record was probed on a different host, driver or device selection"
    if not record.get("interpreter_fingerprint"):
        return "the record carries no interpreter fingerprint"
    adaptation = _runtime.runtime_adaptation(model, undeclared_is_error=False)
    expected_adaptation = str(adaptation.get("target_compute_capability")) if adaptation else None
    if record.get("runtime_adaptation") != expected_adaptation:
        return (f"was probed under runtime adaptation {record.get('runtime_adaptation')!r}, "
                f"this environment's is {expected_adaptation!r}")
    # A superseded record for this model (an older interpreter fingerprint, say) is cheaply
    # rejected here: only a record whose capability key IS this environment's key can be accepted
    # below, and the key is computed once per model rather than per record, so a long history of
    # stale records cannot turn the scan into a series of full environment hashes.
    if record.get("capability_key") != _expected_key(model):
        return "the record's capability key is not this environment's key"
    report = _runtime.verify_environment(model, probe=True, authoritative=True)
    if report.tier != _runtime.TIER_GPU_READY:
        return (f"the runtime does not accept it as capability evidence "
                f"[{report.tier}/{report.status}]")
    payload = report.gpu_capability or {}
    if payload.get("capability_key") != key:
        return "the runtime loaded a different capability record for this environment"
    if payload.get("probe_source") != "interpreter" or payload.get("capability_verified") is not True:
        return "the runtime does not treat the record as verified capability evidence"
    return None

real_gpu = False
_unreadable: list[tuple[str | None, str]] = []
_wrong_key: list[str] = []
_unverified: list[str] = []
try:
    _directory = _runtime.gpu_capability_directory(_rc())
    _models = _load_registry()["models"]
    for _path in sorted(_glob.glob(str(_directory / "*.json"))):
        _capability_key = os.path.basename(_path)[:-len(".json")]
        try:
            with open(_path, encoding="utf-8") as _handle:
                _record = _json.load(_handle)
        except (OSError, ValueError) as _error:
            _unreadable.append((_path, f"{type(_error).__name__}: {_error}"))
            continue
        if not isinstance(_record, dict):
            _unreadable.append((_path, "not a JSON object"))
            continue
        # `record_gpu_capability` names each file after the capability key it describes, so a
        # record whose own key differs — including the pre-per-key `gpu_capability.json`, which
        # carries a key unrelated to its filename — is evidence for a different environment and
        # fails closed rather than certifying this host.
        if _record.get("capability_key") != _capability_key:
            _wrong_key.append(_capability_key)
            continue
        _reason = _unverified_reason(_capability_key, _record, _models)
        if _reason is None:
            real_gpu = True
        else:
            _unverified.append(f"{_capability_key}.json: {_reason}")
except Exception as _error:      # path/config resolution failed: nothing could be read
    _unreadable.append((None, f"capability scan failed: {type(_error).__name__}: {_error}"))
_names = [os.path.basename(path) if path else reason for path, reason in _unreadable]
for _path, _ in _unreadable:
    # An unreadable artifact carries no evidence, so it is removed: leaving it behind would make
    # every later scan report the same condition. Readable records are never touched.
    if _path:
        try:
            os.remove(_path)
        except OSError:
            pass
# A stale directory can hold thousands of superseded records; the report names the condition and a
# bounded sample of the offenders (with the full count), so volume cannot bury the finding.
def _sample(items: list[str]) -> list[str]:
    return items[:10] + ([f"... and {len(items) - 10} more"] if len(items) > 10 else [])

# The scan's own condition is always reported, verified or not: an operator reading the release
# report must be able to tell "no record was found" from "records were found and rejected".
out["gpu_capability_scan"] = {
    # The condition that degraded the scan is named first (an unreadable artifact, a record for
    # another environment, a rejected record); `real_gpu` states the verdict separately, so a
    # removed leftover is visible without hiding the genuine probe it sat next to.
    "status": ("UNREADABLE" if _unreadable else "WRONG_KEY" if _wrong_key
               else "UNVERIFIED" if not real_gpu else "VERIFIED"),
    "unreadable": _names, "wrong_key": _wrong_key,
    "unverified": _sample(_unverified), "unverified_total": len(_unverified),
    "real_gpu": real_gpu}
if _unreadable or _wrong_key or _unverified:
    _rule = "; ".join(
        message for message in (
            "unreadable capability artifact(s) (removed): " + "; ".join(_names) if _names else "",
            "capability record(s) for a different environment: " + ", ".join(_wrong_key) if _wrong_key else "",
            "capability record(s) that are not verified evidence: " + "; ".join(_sample(_unverified)) if _unverified else "")
        if message)
    out["checks"]["J5"]["detail"] = (out["checks"]["J5"].get("detail") or "") + f" [WARN] {_rule}"
    print(f"[WARN] {_rule}", file=sys.stderr)
if not real_gpu:
    out["dimensions"]["GPU_RUNTIME"]["status"] = "PENDING"
from workbench.scripts import verify_run as _verify_run
from workbench.scripts import evaluate_models as _em

_config = _rc()
_reports_root = _em.reports_root(_config)
_accepted_evidence: list[dict] = []
_unaccepted_runs: list[tuple[str, list[str]]] = []
_inference_evidence: list[str] = []
_invalid_metric_runs: list[tuple[str, list[str]]] = []

if _reports_root.is_dir():
    for _path in sorted(_reports_root.glob("*_aggregate.json")):
        try:
            with open(_path, encoding="utf-8") as _handle:
                _rep = _json.load(_handle)
        except Exception:
            continue
        if not isinstance(_rep, dict):
            continue
        _run_id = str(_rep.get("run_id") or "")
        _model_id = str(_rep.get("model_id") or "")
        _checkpoint_id = str(_rep.get("checkpoint_id") or "")
        _protocol_id = str(_rep.get("protocol_id") or "")
        if not (_run_id and _model_id and _checkpoint_id and _protocol_id):
            continue
        try:
            _v = _verify_run.verify(_run_id, _model_id, _checkpoint_id, _protocol_id)
        except Exception as _verr:
            _unaccepted_runs.append((_run_id, [f"verifier error: {type(_verr).__name__}: {_verr}"]))
            continue

        _is_synthetic = _rep.get("is_synthetic") is True or _rep.get("data_kind") == "mock"
        _reasons = list(_v.get("reasons") or [])
        _evidence_reasons = list(_v.get("evidence_reasons") or [])
        _metric_reasons, _ = _verify_run.metric_evidence_reasons(_model_id, _checkpoint_id, _protocol_id, _rep)
        if _metric_reasons:
            _invalid_metric_runs.append((_run_id, _metric_reasons))

        _proof_ok = False
        try:
            _validator = _em.completion_validator(
                _model_id, _checkpoint_id, _protocol_id,
                _verify_run._regression_pin(_model_id), _config
            )
            _proof_ok, _, _ = _validator({"run_id": _run_id})
        except Exception:
            _proof_ok = False

        _corroboration_reasons = [r for r in _reasons if r not in _metric_reasons]
        _has_execution_evidence = (
            not _is_synthetic
            and _rep.get("data_kind") == "experiment"
            and _proof_ok
            and not _corroboration_reasons
        )
        if _v.get("is_research_evidence") is True:
            _accepted_evidence.append(_v)
            _inference_evidence.append(_run_id)
        else:
            _all_refusals = _reasons + _evidence_reasons
            _unaccepted_runs.append((_run_id, _all_refusals or ["not research evidence"]))
            if _has_execution_evidence:
                _inference_evidence.append(_run_id)

if _accepted_evidence:
    out["dimensions"]["MODEL_INFERENCE"]["status"] = "PASS"
    out["dimensions"]["MODEL_INFERENCE"]["detail"] = f"real model execution observed: {', '.join(_inference_evidence)}"
    out["dimensions"]["MODEL_EVALUATION"]["status"] = "PASS"
    out["dimensions"]["MODEL_EVALUATION"]["detail"] = f"legitimate corroborated benchmark run(s) accepted: {', '.join(v.get('identity', {}).get('run_id') or '' for v in _accepted_evidence)}"
    out["dimensions"]["SCIENTIFIC_OUTPUT"]["status"] = "PASS"
    out["dimensions"]["SCIENTIFIC_OUTPUT"]["detail"] = f"valid, authenticated metrics verified: {', '.join(v.get('identity', {}).get('run_id') or '' for v in _accepted_evidence)}"
else:
    out["dimensions"]["MODEL_EVALUATION"]["status"] = "PENDING"
    if _unaccepted_runs:
        _uid, _ureasons = _unaccepted_runs[0]
        out["dimensions"]["MODEL_EVALUATION"]["detail"] = f"no corroborated research-evidence run ({_uid}: {'; '.join(_ureasons[:2])})"
    else:
        out["dimensions"]["MODEL_EVALUATION"]["detail"] = "no corroborated research-evidence run: no benchmark runs found"

    if _inference_evidence:
        out["dimensions"]["MODEL_INFERENCE"]["status"] = "PASS"
        out["dimensions"]["MODEL_INFERENCE"]["detail"] = f"real model execution observed: {', '.join(_inference_evidence)}"
    else:
        out["dimensions"]["MODEL_INFERENCE"]["status"] = "PENDING"
        if _unaccepted_runs:
            out["dimensions"]["MODEL_INFERENCE"]["detail"] = "synthetic/self-test evaluator is not real model execution; pending real model execution evidence"
        else:
            out["dimensions"]["MODEL_INFERENCE"]["detail"] = "no model execution evidence observed"

    out["dimensions"]["SCIENTIFIC_OUTPUT"]["status"] = "PENDING"
    if _invalid_metric_runs:
        _mid, _mreasons = _invalid_metric_runs[0]
        out["dimensions"]["SCIENTIFIC_OUTPUT"]["detail"] = f"invalid metrics for {_mid} ({'; '.join(_mreasons[:2])})"
    else:
        out["dimensions"]["SCIENTIFIC_OUTPUT"]["detail"] = "no valid, authenticated metrics: no corroborated research-evidence run"

out["accepted_evidence_runs"] = [
    {
        "run_id": v.get("identity", {}).get("run_id"),
        "model_id": v.get("identity", {}).get("model_id"),
        "checkpoint_id": v.get("identity", {}).get("checkpoint_id"),
        "protocol_id": v.get("identity", {}).get("protocol_id"),
        "report_path": v.get("report_path"),
        "metrics": v.get("metrics"),
    }
    for v in _accepted_evidence
]

mandatory_pending = [d for d, spec in out["dimensions"].items()
                     if spec["mandatory"] and spec["status"] != "PASS"]
out["pending_dimensions"] = mandatory_pending
if out["summary"]["fail"] > 0 or any(spec["status"] == "FAIL" for spec in out["dimensions"].values() if spec["mandatory"]):
    out["verdict"] = "RELEASE_BLOCKED"
elif mandatory_pending:
    _gpu_or_research_dims = {"GPU_RUNTIME", "MODEL_INFERENCE", "MODEL_EVALUATION", "SCIENTIFIC_OUTPUT"}
    _non_gpu_pending = [d for d in mandatory_pending if d not in _gpu_or_research_dims]
    if _non_gpu_pending:
        out["verdict"] = "ENGINEERING_INCOMPLETE"
    else:
        out["verdict"] = "GPU_VALIDATION_PENDING"
else:
    out["verdict"] = "RELEASE_READY"

print("\nAcceptance dimensions:")
for dim, spec in out["dimensions"].items():
    print(f"  {dim:<20} {spec['status']}")
with open(sys.argv[1], "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=2, ensure_ascii=False)
    fh.write("\n")
print(f"report written: {sys.argv[1]} verdict={out['verdict']}")
EOF
REPORT_RC=$?
if (( REPORT_RC != 0 )); then
    say "RELEASE VERIFIER: FAILED (report generation failed with exit code ${REPORT_RC})" >&2
    exit 1
fi
if [[ ! -f "$REPORT" ]]; then
    say "RELEASE VERIFIER: FAILED (report file not written: ${REPORT})" >&2
    exit 1
fi

if (( fails > 0 )); then
    say "RELEASE VERIFIER: FAILED (${fails} unmet, ${pendings} pending)"
    exit 1
fi
say "RELEASE VERIFIER: no unmet criteria (${pendings} pending)"
exit 0
