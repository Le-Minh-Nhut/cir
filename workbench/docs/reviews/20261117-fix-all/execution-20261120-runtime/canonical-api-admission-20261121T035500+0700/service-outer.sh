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
