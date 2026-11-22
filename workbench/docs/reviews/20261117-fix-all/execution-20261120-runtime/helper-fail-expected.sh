#!/bin/bash
set -Eeuo pipefail
cd /home/heheboiz/data/cir
RUN_ROOT=workbench/docs/reviews/20261117-fix-all/execution-20261120-runtime/helper-fail-expected
mkdir -p "$RUN_ROOT"
echo occupied > "$RUN_ROOT/marker.txt"
if workbench/envs/ctl312h/bin/python workbench/docs/reviews/20261117-fix-all/execution-20261120-runtime/dcnet_lifecycle_helper.py --run-root "$RUN_ROOT" --port 18765 --timeout 1; then
    status=0
else
    status=$?
fi
printf 'helper_exit=%s\n' "$status"
[[ $status -eq 1 ]]
