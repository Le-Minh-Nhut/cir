#!/usr/bin/env bash
set -euo pipefail
root=/home/heheboiz/data/cir
run=$root/workbench/docs/reviews/20261117-fix-all/execution-20261120-runtime/canonical-api-admission-20261121T035500+0700
unit=cir-dcnet-canonical-api-20261121
mkdir -p "$run"
printf 'unit=%s\nrun_root=%s\n' "$unit" "$run" > "$run/launch-intent.txt"
cat > "$run/service-inner.sh" <<'INNER'
#!/usr/bin/env bash
set -euo pipefail
root=/home/heheboiz/data/cir
run=/home/heheboiz/data/cir/workbench/docs/reviews/20261117-fix-all/execution-20261120-runtime/canonical-api-admission-20261121T035500+0700
unit=cir-dcnet-canonical-api-20261121
base=/sys/fs/cgroup$(systemctl --user show "$unit.service" -p ControlGroup --value)
printf 'unit=%s\ncgroup=%s\nmemory.max=%s\nmemory.swap.max=%s\nmemory.oom.group=%s\n' "$unit" "$base" "$(cat "$base/memory.max")" "$(cat "$base/memory.swap.max")" "$(cat "$base/memory.oom.group")" > "$run/guard.receipt"
test "$(cat "$base/memory.max")" = 12884901888
test "$(cat "$base/memory.swap.max")" = 0
test "$(cat "$base/memory.oom.group")" = 1
cd "$root"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
set +e
flock -x /tmp/cir-fix-all-20261120-cpu.lock env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  workbench/envs/dcnet310-sm120/bin/python workbench/docs/reviews/20261117-fix-all/execution-20261120-runtime/canonical_dcnet_admission.py \
  --run-root "$run/controller" --timeout 1800
status=$?
printf 'controller_exit=%s\n' "$status" > "$run/controller.exit"
exit "$status"
INNER
chmod 700 "$run/service-inner.sh"
cat > "$run/service-outer.sh" <<'OUTER'
#!/usr/bin/env bash
set -euo pipefail
root=/home/heheboiz/data/cir
run=/home/heheboiz/data/cir/workbench/docs/reviews/20261117-fix-all/execution-20261120-runtime/canonical-api-admission-20261121T035500+0700
unit=cir-dcnet-canonical-api-20261121
set +e
"$root/workbench/scripts/gpu_job.sh" /bin/bash "$run/service-inner.sh"
status=$?
printf 'gpu_wrapper_exit=%s\n' "$status" > "$run/service.exit"
exit "$status"
OUTER
chmod 700 "$run/service-outer.sh"
printf 'systemd-run --user --unit=%s --property=MemoryMax=12G --property=MemorySwapMax=0 --property=OOMPolicy=kill --property=StandardOutput=append:%s/service.stdout.log --property=StandardError=append:%s/service.stderr.log --collect /bin/bash %s/service-outer.sh\n' "$unit" "$run" "$run" "$run" > "$run/launch-command.txt"
printf 'Prepared only; service not started.\n' > "$run/launch.state"
