#!/usr/bin/env bash
# CIR Failure Analysis Workbench — Docker launcher for the target PC.
#
# Usage: run.sh <command>
#   up            build and start the workbench (backend + frontend)
#   doctor        honest health check: docker, config, dataset, checkpoints, GPU visibility
#   prepare limn  run the guarded preparation pipeline for one model (sync, env, dataset,
#                 checkpoints, GPU verification) inside the container
#   pilot limn    prepare (if needed) then evaluate — the same guarded pipeline, with --verify-gpu
#   pilot csmcir
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
BACKEND_PORT="${WORKBENCH_BACKEND_PORT:-8000}"
FRONTEND_PORT="${WORKBENCH_FRONTEND_PORT:-5173}"
BASE="http://127.0.0.1:${BACKEND_PORT}"

say() { printf '%s\n' "$*"; }

need_env() {
    if [[ ! -f "${HERE}/.env" ]]; then
        say "missing ${HERE}/.env — copy .env.example to .env and set the host paths first."
        exit 2
    fi
}

COMPOSE_FILES=(-f "${HERE}/docker-compose.yml")
if [[ "${WORKBENCH_ENABLE_GPU:-0}" == "1" ]]; then
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
    curl -fsS "${BASE}/api/jobs" | python3 -m json.tool
}

cmd_doctor() {
    say "== docker =="
    docker version --format '{{.Server.Version}}' 2>/dev/null || { say "docker daemon unreachable"; return 1; }
    say "== container =="
    compose ps
    say "== API =="
    for endpoint in models runs jobs benchmark-runs cohorts; do
        code="$(curl -s -o /dev/null -w '%{http_code}' "${BASE}/api/${endpoint}" || true)"
        say "  /api/${endpoint}: ${code}"
    done
    say "== GPU visibility (honest) =="
    if docker compose -f "${HERE}/docker-compose.yml" --env-file "${HERE}/.env" \
        run --rm --quiet workbench python -c "
import json, subprocess
try:
    out = subprocess.check_output(['nvidia-smi', '--query-gpu=name,compute_cap', '--format=csv,noheader'], text=True)
    print('nvidia-smi:', out.strip())
except Exception as error:
    print('nvidia-smi unavailable inside container:', error)
    print('GPU validation: PENDING — install nvidia-container-toolkit on the host, then start')
    print('with WORKBENCH_ENABLE_GPU=1 ./run.sh up (docker-compose.gpu.yml), and re-run doctor.')
"; then
        say "gpu probe finished (see above)"
    else
        say "gpu probe could not run"
    fi
}

# Read one JSON document from the API and extract a field with a real parser. The response is
# piped into python on stdin, so a malformed or empty response is an explicit error, never a
# silent empty variable.
api_field() { # api_path python_expr
    local response
    if ! response="$(curl -fsS "${BASE}/$1")"; then
        say "API request failed: ${BASE}/$1"; return 1
    fi
    printf '%s' "$response" | python3 -c "
import json, sys
try:
    doc = json.load(sys.stdin)
except json.JSONDecodeError as error:
    sys.exit(f'malformed JSON from $1: {error}')
result = ($2)
if not result:
    sys.exit(f'$1: field not found')
print(result[0] if isinstance(result, list) else result)
"
}

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
        --dataset-root "${WORKBENCH_DATASET_ROOT:-/data/FashionIQ}" \
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
    # protocol_id must be the model's own native protocol (limn: fashioniq_val_split,
    # csmcir: fashioniq_original_split) or the guarded plan rejects it with 409.
    local protocol checkpoint
    local protocol_expr='[m["native_protocol"] for m in doc if m["model_id"]=="'"${model}"'"]'
    protocol="$(api_field "api/models" "${protocol_expr}")" || return 1
    # The registered bundle is the correct selection: LIMN's complete macro result needs all
    # three per-category checkpoints (dress, shirt, toptee) as one declared bundle. Picking the
    # first variant would silently reduce the benchmark to one category.
    local bundle_expr='[(m.get("checkpoint_bundles") or [{}])[0].get("bundle_id") or m["checkpoints"][0]["checkpoint_id"] for m in doc if m["model_id"]=="'"${model}"'"]'
    checkpoint="$(api_field "api/models" "${bundle_expr}")" || return 1
    say "model=${model} checkpoint=${checkpoint} protocol=${protocol}"
    # Preparation first: a fresh installation must not depend on undocumented manual steps.
    cmd_prepare "${model}" || { say "preparation for ${model} did not complete — see the pipeline output above"; return 1; }
    say "queueing official evaluation through the API; the backend's guarded plan refuses"
    say "anything that has not passed preflight, and the queue will not start an unverified GPU run."
    local response
    if ! response="$(curl -fsS -X POST "${BASE}/api/jobs" -H 'Content-Type: application/json' \
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
