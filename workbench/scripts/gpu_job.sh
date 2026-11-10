#!/usr/bin/env bash
# Run one GPU job under a campaign-wide lock, refusing to start while another process holds the GPU.
#
# Why a lock and not just a check: `nvidia-smi --query-compute-apps` is a snapshot, so two workers that
# both sample an idle GPU start together. `flock -n` makes the second one fail to acquire rather than
# proceed, and this wrapper then waits and retries. It never kills the holder and never runs unlocked.
set -euo pipefail

LOCK=/tmp/cir_gpu.lock
MAX_WAIT=${MAX_WAIT:-7200}
POLL=${POLL:-30}

usage() {
  echo "usage: $0 <command> [args...]" >&2
  exit 64
}

[[ $# -ge 1 ]] || usage

deadline=$(( $(date +%s) + MAX_WAIT ))
while :; do
  # `flock -n 9` on a held descriptor is the only reliable non-blocking probe in bash: `flock -c`
  # re-executes a shell and needs its command quoted, which loses the caller's argv.
  exec 9>"$LOCK"
  if flock -n 9; then
    break
  fi
  exec 9>&-
  if (( $(date +%s) > deadline )); then
    echo "[gpu_job] lock still held after ${MAX_WAIT}s; not starting" >&2
    exit 75
  fi
  echo "[gpu_job] lock held; retrying in ${POLL}s" >&2
  sleep "$POLL"
done

# Both conditions the operator asked for must hold AT LAUNCH, inside the lock: no other compute app.
apps=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader)
if [[ -n "${apps//[[:space:]]/}" ]]; then
  echo "[gpu_job] GPU already has compute apps:" >&2
  echo "$apps" >&2
  exit 76
fi

echo "[gpu_job] lock held, GPU empty, launching: $*" >&2
nvidia-smi --query-gpu=uuid,memory.used,memory.total --format=csv,noheader >&2
# fd 9 stays open for the whole child lifetime, which is what holds the lock.
"$@"
