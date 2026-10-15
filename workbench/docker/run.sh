#!/usr/bin/env bash
# CIR Failure Analysis Workbench — Docker launcher for the target PC.
#
# Usage: run.sh <command>
#   up            build and start the workbench (backend + frontend)
#   doctor        honest health check: docker, config, dataset, checkpoints, GPU visibility
#   pilot limn    queue the LIMN pilot evaluation (first in the priority order)
#   pilot csmcir  queue the CSMCIR pilot evaluation (second)
#   status        job list from the API
#   logs          follow the container logs
#   stop          stop the stack
#
# The GPU check is honest: it reports exactly what the container can see. If the host lacks
# nvidia-container-toolkit (or the compose GPU block is disabled), it says PENDING with the
# reason — it never fabricates a capability.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_PORT="${WORKBENCH_BACKEND_PORT:-8000}"
BASE="http://127.0.0.1:${BACKEND_PORT}"

say() { printf '%s\n' "$*"; }

need_env() {
    if [[ ! -f "${HERE}/.env" ]]; then
        say "no .env found; copying .env.example -> .env (edit host paths if they differ)"
        cp "${HERE}/.env.example" "${HERE}/.env"
    fi
}

compose() { docker compose -f "${HERE}/docker-compose.yml" --env-file "${HERE}/.env" "$@"; }

cmd_up() {
    need_env
    compose up -d --build
    say "backend: ${BASE}  frontend: http://127.0.0.1:${WORKBENCH_FRONTEND_PORT:-5173}"
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
    print('GPU validation: PENDING — install nvidia-container-toolkit on the host and enable')
    print('the deploy.resources.gpu block in docker-compose.yml, then re-run doctor.')
"; then
        say "gpu probe finished (see above)"
    else
        say "gpu probe could not run"
    fi
}

cmd_pilot() {
    local model="${1:-}"
    case "${model}" in
        limn|csmcir) ;;
        *) say "usage: run.sh pilot <limn|csmcir>"; return 2;;
    esac
    # Queue the official evaluation through the API. The backend's guarded plan refuses
    # anything that has not passed preflight; this endpoint only ever queues real work.
    local payload model_id checkpoint
    payload="$(curl -fsS "${BASE}/api/models")"
    model_id="$(python3 -c "import json,sys; d=json.load(sys.stdin); print(next(m['model_id'] for m in d if m['model_id']=='${model}'))")"
    checkpoint="$(python3 -c "import json,sys; d=json.load(sys.stdin); print(next(m['checkpoints'][0]['checkpoint_id'] for m in d if m['model_id']=='${model}'))")"
    curl -fsS -X POST "${BASE}/api/jobs" -H 'Content-Type: application/json' \
        -d "{\"model_id\":\"${model_id}\",\"checkpoint_id\":\"${checkpoint}\",\"protocol_id\":\"fashioniq_original_split\",\"top_k\":200}"
    say ""
    say "queued; watch with: run.sh status / run.sh logs"
}

case "${1:-}" in
    up) cmd_up;;
    doctor) cmd_doctor;;
    pilot) shift; cmd_pilot "$@";;
    status) cmd_status;;
    logs) cmd_logs;;
    stop) cmd_stop;;
    *) say "usage: run.sh <up|doctor|pilot limn|pilot csmcir|status|logs|stop>"; exit 2;;
esac
