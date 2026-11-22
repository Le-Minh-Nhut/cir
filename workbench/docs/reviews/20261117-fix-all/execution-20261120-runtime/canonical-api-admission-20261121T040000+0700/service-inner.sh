#!/usr/bin/env bash
set -euo pipefail
root=/home/heheboiz/data/cir
run=/home/heheboiz/data/cir/workbench/docs/reviews/20261117-fix-all/execution-20261120-runtime/canonical-api-admission-20261121T040000+0700
unit=cir-dcnet-canonical-api-20261121-v3
base=/sys/fs/cgroup$(systemctl --user show "$unit.service" -p ControlGroup --value)
printf 'unit=%s\ncgroup=%s\nmemory.max=%s\nmemory.swap.max=%s\nmemory.oom.group=%s\n' "$unit" "$base" "$(cat "$base/memory.max")" "$(cat "$base/memory.swap.max")" "$(cat "$base/memory.oom.group")" > "$run/guard.receipt"
test "$(cat "$base/memory.max")" = 12884901888
test "$(cat "$base/memory.swap.max")" = 0
test "$(cat "$base/memory.oom.group")" = 1
cd "$root"
export WORKBENCH_RUNTIME_ADAPTATION=sm_120
export WORKBENCH_DCNET_PYTHON=/home/heheboiz/data/cir/workbench/envs/dcnet310-sm120/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
set +e
flock -x /tmp/cir-fix-all-20261120-cpu.lock env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  workbench/envs/ctl312h/bin/python workbench/docs/reviews/20261117-fix-all/execution-20261120-runtime/canonical_dcnet_admission.py \
  --run-root "$run/controller" --timeout 1800
status=$?
printf 'controller_exit=%s\n' "$status" > "$run/controller.exit"
exit "$status"
