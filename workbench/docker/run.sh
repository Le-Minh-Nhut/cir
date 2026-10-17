#!/usr/bin/env bash
# CIR Failure Analysis Workbench — Docker launcher for the target PC.
#
# Usage: run.sh <command>
#   up            build and start the workbench (backend + frontend)
#   doctor        honest health check: docker, config, dataset, checkpoints, GPU visibility
#   prepare limn  run the guarded preparation pipeline for one model (sync, env, dataset,
#                 checkpoints, GPU verification) inside the container
#   pilot limn    prepare (if needed) then queue an API evaluation; the backend's launch guard
#   pilot csmcir  refuses to start a job whose GPU capability is unverified (no --verify-gpu flag here)
#   status        job list from the API
#   logs          follow the container logs
#   stop          stop the stack
#
# The GPU check is honest: it reports exactly what the container can see. If the host lacks
# nvidia-container-toolkit (or the compose GPU override is disabled), it says PENDING with the
# reason — it never fabricates a capability. A GPU-declaring model is evaluated only through the
# guarded pipeline, which refuses to launch when the GPU is not verified (fail closed).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

say() { printf '%s\n' "$*"; }

# ---------------------------------------------------------------------------
# ONE environment-resolution contract for every command.
# Precedence: explicit shell override > validated .env > built-in default.
# .env is parsed strictly (KEY=VALUE, comments, no eval): a malformed line is
# an error, never silently ignored — a mistyped port must not route requests
# to a stack the operator did not configure.
# ---------------------------------------------------------------------------
env_value() { # env_value KEY DEFAULT
    local key="$1" default="$2" line value k
    if [[ -n "${!key:-}" ]]; then
        printf '%s\n' "${!key}"; return 0
    fi
    if [[ -f "${HERE}/.env" ]]; then
        # `|| [[ -n "$line" ]]` so a final line without a trailing newline is not silently
        # dropped: without it this function answered the DEFAULT for the last variable while
        # pc.sh and compose answered the file's value, and the three disagreed about the stack.
        while IFS= read -r line || [[ -n "$line" ]]; do
            # Strip a trailing comment, then ONE layer of matching quotes -- the same order and
            # the same rules compose applies, and the same ones pc.sh implements. Stripping the
            # comment first without honouring quotes left `"8000" ` here (quotes retained, because
            # the closing quote was no longer last), which `validate_env` then rejected as a
            # non-integer port: a well-formed .env failed only when read through this path.
            line="${line%$'\r'}"
            k="${line%%=*}"
            k="${k#"${k%%[![:space:]]*}"}"
            k="${k%"${k##*[![:space:]]}"}"
            if [[ "$k" == "$key" ]]; then
                value="${line#*=}"
                value="${value%$'\r'}"
                if [[ "$value" == \"* ]]; then
                    value="${value:1}"
                    value="${value%%\"*}"
                elif [[ "$value" == \'* ]]; then
                    value="${value:1}"
                    value="${value%%\'*}"
                else
                    value="${value%%#*}"
                    value="${value#"${value%%[![:space:]]*}"}"
                    value="${value%"${value##*[![:space:]]}"}"
                fi
                printf '%s\n' "$value"; return 0
            fi
        done < "${HERE}/.env"
    fi
    printf '%s\n' "$default"
}

validate_env() { # validate_env KEY VALUE
    case "$2" in
        ''|*[!0-9]*) say "invalid ${1}=${2}: must be an integer port"; exit 2;;
    esac
    if (( $2 < 1 || $2 > 65535 )); then
        say "invalid ${1}=${2}: port out of range"; exit 2
    fi
}

need_env() {
    if [[ ! -f "${HERE}/.env" ]]; then
        say "missing ${HERE}/.env — copy .env.example to .env and set the host paths first."
        exit 2
    fi
}

BACKEND_PORT="$(env_value WORKBENCH_BACKEND_PORT 8000)"
FRONTEND_PORT="$(env_value WORKBENCH_FRONTEND_PORT 5173)"
validate_env WORKBENCH_BACKEND_PORT "$BACKEND_PORT"
validate_env WORKBENCH_FRONTEND_PORT "$FRONTEND_PORT"
ENABLE_GPU="$(env_value WORKBENCH_ENABLE_GPU 0)"
case "$ENABLE_GPU" in 0|1) ;; *) say "invalid WORKBENCH_ENABLE_GPU=${ENABLE_GPU}: must be 0 or 1"; exit 2;; esac
BASE="http://127.0.0.1:${BACKEND_PORT}"

COMPOSE_FILES=(-f "${HERE}/docker-compose.yml")
if [[ "$ENABLE_GPU" == "1" ]]; then
    COMPOSE_FILES+=(-f "${HERE}/docker-compose.gpu.yml")
fi
compose() { docker compose "${COMPOSE_FILES[@]}" --env-file "${HERE}/.env" "$@"; }

cmd_up() {
    need_env
    compose up -d --build
    say "backend: ${BASE}  frontend: http://127.0.0.1:${FRONTEND_PORT}"
    cmd_doctor || true
}

cmd_stop() { compose down; }

cmd_logs() { compose logs -f; }

cmd_status() {
    curl -fsS --max-time 30 "${BASE}/api/jobs" | python3 -m json.tool
}

cmd_doctor() {
    say "== docker =="
    docker version --format '{{.Server.Version}}' 2>/dev/null || { say "docker daemon unreachable"; return 1; }
    say "== container =="
    compose ps
    say "== API =="
    for endpoint in models runs jobs benchmark-runs cohorts; do
        code="$(curl -s --max-time 120 -o /dev/null -w '%{http_code}' "${BASE}/api/${endpoint}" || true)"
        say "  /api/${endpoint}: ${code}"
    done
    say "== GPU verification (resolved configuration) =="
    if [[ "$ENABLE_GPU" != "1" ]]; then
        say "WORKBENCH_ENABLE_GPU is not 1: the GPU compose override is not part of this stack"
        say "GPU validation: PENDING — set WORKBENCH_ENABLE_GPU=1 (shell or .env) and re-run"
        return 3
    fi
    say "compose override in effect: docker-compose.gpu.yml"
    local probe_code=0 probe
    if C="$(docker ps -q --filter name=^cir-workbench$ | head -1)" && [[ -n "$C" ]]; then
        # Probe the ACTUAL running container: a throwaway compose-run container describes a
        # different configuration than the one serving the API.
        probe="$(docker exec "$C" python -c '
import json, subprocess
result = {"nvidia_smi": None, "torch_cuda": None, "cuda_tensor": None, "error": None}
try:
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"], text=True)
    result["nvidia_smi"] = out.strip()
except Exception as error:
    result["error"] = f"nvidia-smi: {error}"
try:
    import torch
    result["torch_cuda"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        x = torch.randn(8, 8, device="cuda")
        result["cuda_tensor"] = bool((x @ x).sum().isfinite().item())
        result["compute_capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
        result["device_name"] = torch.cuda.get_device_name(0)
except Exception as error:
    result["error"] = f"torch: {error}"
print(json.dumps(result))
' 2>&1)" || probe_code=1
    else
        # Nothing running: probe through the same resolved configuration other commands use.
        if probe="$(compose run --rm --quiet workbench python -c '
import json, subprocess
result = {"nvidia_smi": None, "torch_cuda": None, "cuda_tensor": None, "error": None}
try:
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"], text=True)
    result["nvidia_smi"] = out.strip()
except Exception as error:
    result["error"] = f"nvidia-smi: {error}"
try:
    import torch
    result["torch_cuda"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        x = torch.randn(8, 8, device="cuda")
        result["cuda_tensor"] = bool((x @ x).sum().isfinite().item())
        result["compute_capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
        result["device_name"] = torch.cuda.get_device_name(0)
except Exception as error:
    result["error"] = f"torch: {error}"
print(json.dumps(result))
' 2>&1)"; then probe_code=0; else probe_code=1; fi
    fi
    python3 - "$probe" <<'PYDOC' || probe_code=1
import json, sys
try:
    data = json.loads(sys.argv[1].strip().splitlines()[-1])
except Exception:
    print("gpu probe produced no parseable result"); sys.exit(1)
print("nvidia-smi:", data.get("nvidia_smi") or "unavailable")
print("torch.cuda.is_available():", data.get("torch_cuda"))
if data.get("torch_cuda"):
    print("CUDA tensor op:", "OK" if data.get("cuda_tensor") else "FAILED",
          "| compute capability:", data.get("compute_capability"), "| device:", data.get("device_name"))
if data.get("error"):
    print("probe error:", data["error"])
if not (data.get("torch_cuda") and data.get("cuda_tensor")):
    print("GPU validation: PENDING — install nvidia-container-toolkit on the host, then start")
    print("the stack with the GPU override (WORKBENCH_ENABLE_GPU=1 ./run.sh up) and re-run doctor.")
    sys.exit(1)
print("GPU validation: VERIFIED (CUDA tensor executed)")
PYDOC
    if (( probe_code == 0 )); then
        say "gpu probe finished (see above)"
    else
        say "gpu probe could not verify CUDA execution (exit $probe_code)"
        return "$probe_code"
    fi
}

# Read one JSON document from the API and extract a field with a real parser. The response is
# piped into python on stdin, so a malformed or empty response is an explicit error, never a
# silent empty variable.
cmd_prepare() {
    local model="${1:-}"
    case "${model}" in
        limn|csmcir) ;;
        *) say "usage: run.sh prepare <limn|csmcir>"; return 2;;
    esac
    need_env
    say "preparing ${model} through the guarded pipeline (sync, environment, dataset, checkpoints, GPU)…"
    compose exec -T workbench python workbench/scripts/pipeline.py setup \
        --model "${model}" \
        --dataset-root "$(env_value WORKBENCH_DATASET_ROOT /data/FashionIQ)" \
        --sync-sources --download-checkpoints --apply \
        --allow-network --allow-large-downloads --allow-env-install --allow-preparation \
        --allow-gpu-eval --allow-index-write --resume
}

cmd_pilot() {
    local model="${1:-}"
    case "${model}" in
        limn|csmcir) ;;
        *) say "usage: run.sh pilot <limn|csmcir>"; return 2;;
    esac
    need_env
    # One /api/models fetch for both selections: the registry build is the slow part and the
    # endpoint re-serves it per call, so two calls double the latency for no information.
    # protocol_id must be the model's own native protocol (limn: fashioniq_val_split,
    # csmcir: fashioniq_original_split) or the guarded plan rejects it with 409.
    # The registered bundle is the correct selection: LIMN's complete macro result needs all
    # three per-category checkpoints (dress, shirt, toptee) as one declared bundle. Picking the
    # first variant would silently reduce the benchmark to one category.
    local response
    if ! response="$(curl -fsS --max-time 120 "${BASE}/api/models")"; then
        say "API request failed: ${BASE}/api/models"; return 1
    fi
    local protocol checkpoint
    read -r protocol checkpoint <<<"$(printf '%s' "${response}" | python3 -c "
import json, sys
try:
    doc = json.load(sys.stdin)
except json.JSONDecodeError as error:
    sys.exit(f'malformed JSON from api/models: {error}')
matches = [m for m in doc if m.get('model_id') == '${model}']
if not matches:
    sys.exit(f'api/models: model ${model} not found')
m = matches[0]
protocol = m['native_protocol']
bundles = m.get('checkpoint_bundles') or [{}]
checkpoint = bundles[0].get('bundle_id') or m['checkpoints'][0]['checkpoint_id']
print(protocol, checkpoint)
")" || return 1
    # A failed parse leaves both empty while `read` still succeeds on the empty line.
    [[ -n "${protocol}" && -n "${checkpoint}" ]] || return 1
    say "model=${model} checkpoint=${checkpoint} protocol=${protocol}"
    # Preparation first: a fresh installation must not depend on undocumented manual steps.
    cmd_prepare "${model}" || { say "preparation for ${model} did not complete — see the pipeline output above"; return 1; }
    say "queueing official evaluation through the API; the backend's guarded plan refuses"
    say "anything that has not passed preflight, and the queue will not start an unverified GPU run."
    local response
    if ! response="$(curl -fsS --max-time 60 -X POST "${BASE}/api/jobs" -H 'Content-Type: application/json' \
        -d "{\"model_id\":\"${model}\",\"checkpoint_id\":\"${checkpoint}\",\"protocol_id\":\"${protocol}\",\"top_k\":200}" 2>&1)"; then
        say "${response}"
        say "run 'run.sh up' first, check 'run.sh doctor', then retry; see PC_DOCKER_QUICKSTART.md"
        return 1
    fi
    say "${response}"
    say ""
    say "queued; watch with: run.sh status / run.sh logs"
}

case "${1:-}" in
    up) cmd_up;;
    doctor) cmd_doctor;;
    prepare) shift; cmd_prepare "$@";;
    pilot) shift; cmd_pilot "$@";;
    status) cmd_status;;
    logs) cmd_logs;;
    stop) cmd_stop;;
    *) say "usage: run.sh <up|doctor|prepare limn|pilot limn|pilot csmcir|status|logs|stop>"; exit 2;;
esac
