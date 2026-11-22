#!/usr/bin/env bash
# Run one campaign role in a bounded user service; preserve command argv end-to-end.
set -euo pipefail
ROOT=/home/heheboiz/data/cir
GPU_JOB=${CIR_GPU_JOB:-$ROOT/workbench/scripts/gpu_job.sh}
CPU_LOCK=${CIR_CPU_LOCK:-/tmp/cir-fix-all-20261120-cpu.lock}

usage() {
  echo "usage: $0 --unit NAME --output-dir ABSOLUTE_DIR --min-free-bytes BYTES -- COMMAND [ARG ...]" >&2
  exit 64
}

mode=${1:-}
case "$mode" in
  --inner|--locked) shift ;;
  *) mode=outer ;;
esac

unit= output_dir= min_free_bytes=
while (($#)); do
  case "$1" in
    --unit) (($# >= 2)) || usage; unit=$2; shift 2 ;;
    --output-dir) (($# >= 2)) || usage; output_dir=$2; shift 2 ;;
    --min-free-bytes) (($# >= 2)) || usage; min_free_bytes=$2; shift 2 ;;
    --) shift; break ;;
    *) usage ;;
  esac
done
[[ $unit =~ ^[A-Za-z0-9_.@:-]+$ && $unit != *.service ]] || usage
[[ $output_dir == /* && -d $output_dir ]] || usage
[[ $min_free_bytes =~ ^[0-9]+$ && $min_free_bytes -gt 0 ]] || usage
(($#)) || usage

if [[ $mode != outer ]]; then
  expected_unit="$unit.service"
  cgroup_path=$(awk -F: '$1 == "0" { print $3; exit }' /proc/self/cgroup)
  [[ -n $cgroup_path && $cgroup_path == *"/$expected_unit"* ]] || {
    echo "[role-service] wrong cgroup for $expected_unit: ${cgroup_path:-missing}" >&2
    exit 78
  }
  base="/sys/fs/cgroup$cgroup_path"
  memory_max=$(<"$base/memory.max")
  memory_swap_max=$(<"$base/memory.swap.max")
  oom_group=$(<"$base/memory.oom.group")
  [[ $memory_max == 12884901888 && $memory_swap_max == 0 && $oom_group == 1 ]] || {
    printf '[role-service] cgroup limits mismatch: memory.max=%s memory.swap.max=%s memory.oom.group=%s\n' \
      "$memory_max" "$memory_swap_max" "$oom_group" >&2
    exit 78
  }
  identity=$(systemctl --user show "$expected_unit" -p Id --value)
  [[ $identity == "$expected_unit" ]] || {
    echo "[role-service] service identity mismatch: expected=$expected_unit got=$identity" >&2
    exit 78
  }
  printf 'waiting_for_cpu_lock=1\n' >&2
  exec 9>"$CPU_LOCK"
  flock -x 9
  printf 'cpu_lock_acquired=1\n' >&2
  available_bytes=$(df -PB1 "$output_dir" | awk 'NR == 2 {print $4}')
  printf 'disk_check_after_cpu_lock=1 free_bytes=%s required_bytes=%s\n' "$available_bytes" "$min_free_bytes" >&2
  [[ $available_bytes =~ ^[0-9]+$ && $available_bytes -ge $min_free_bytes ]] || {
    echo "[role-service] insufficient disk space at $output_dir: available=${available_bytes:-unknown} required=$min_free_bytes" >&2
    exit 73
  }
  printf 'unit=%s\ncgroup=%s\nmemory.max=%s\nmemory.swap.max=%s\nmemory.oom.group=%s\noutput_dir=%s\nfree_bytes=%s\nmin_free_bytes=%s\n' \
    "$expected_unit" "$cgroup_path" "$memory_max" "$memory_swap_max" "$oom_group" \
    "$output_dir" "$available_bytes" "$min_free_bytes" >&2
  cd "$ROOT"
  exec env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 "$@"
fi

[[ $0 == /* ]] || ROOT=$(cd "$(dirname "$0")/../../../../.." && pwd)
script="$ROOT/workbench/docs/reviews/20261117-fix-all/execution-20261120-runtime/run-role-service.sh"
service_log_dir="$ROOT/workbench/docs/reviews/20261117-fix-all/execution-20261120-runtime/service-logs"
mkdir -p "$service_log_dir"
stdout="$service_log_dir/${unit}.stdout.log"
stderr="$service_log_dir/${unit}.stderr.log"
# Logs stay outside output_dir so sealed child artifacts remain immutable. The outer shell owns
# capture; with systemd --pipe, service streams belong to the caller, not append properties.
exec >>"$stdout" 2>>"$stderr"
printf '[role-service] unit=%s stdout=%s stderr=%s\n' "$unit" "$stdout" "$stderr"
if "$GPU_JOB" systemd-run --user --wait --pipe --collect --same-dir --service-type=exec \
  --unit="$unit" -p MemoryMax=12G -p MemorySwapMax=0 -p OOMPolicy=kill \
  "$script" --inner --unit "$unit" --output-dir "$output_dir" \
  --min-free-bytes "$min_free_bytes" -- "$@"; then
  status=0
else
  status=$?
fi
printf '[role-service] unit=%s wrapper_exit=%s\n' "$unit" "$status"
exit "$status"
