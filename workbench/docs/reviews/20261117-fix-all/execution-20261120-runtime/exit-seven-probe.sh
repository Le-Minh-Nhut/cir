#!/bin/bash
set -Eeuo pipefail
cd /home/heheboiz/data/cir
if workbench/scripts/gpu_job.sh bash -c 'exit 7'; then
    status=0
else
    status=$?
fi
printf 'wrapper_exit=%s\n' "$status"
[[ $status -eq 7 ]]
