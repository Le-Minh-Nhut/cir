#!/bin/bash
set -Eeuo pipefail
cd /home/heheboiz/data/cir
RUN_ROOT=workbench/docs/reviews/20261117-fix-all/execution-20261120-runtime/dcnet-lifecycle-20261120T2358+0700
CGROUP_PATH=$(awk -F: '$1 == "0" {print $3}' /proc/self/cgroup)
CGROUP_DIR="/sys/fs/cgroup$CGROUP_PATH"
[[ $(cat "$CGROUP_DIR/memory.max") == 12884901888 ]]
[[ $(cat "$CGROUP_DIR/memory.swap.max") == 0 ]]
[[ $(cat "$CGROUP_DIR/memory.oom.group") == 1 ]]
printf 'cgroup=%s\nmemory.max=%s\nmemory.swap.max=%s\nmemory.oom.group=%s\n' "$CGROUP_DIR" "$(cat "$CGROUP_DIR/memory.max")" "$(cat "$CGROUP_DIR/memory.swap.max")" "$(cat "$CGROUP_DIR/memory.oom.group")" > "${RUN_ROOT}-cgroup-assertions.txt"
exec workbench/scripts/gpu_job.sh flock -x /tmp/cir-fix-all-20261120-cpu.lock workbench/envs/ctl312h/bin/python workbench/docs/reviews/20261117-fix-all/execution-20261120-runtime/dcnet_lifecycle_helper.py --run-root "/home/heheboiz/data/cir/$RUN_ROOT" --port 18764 --timeout 300
