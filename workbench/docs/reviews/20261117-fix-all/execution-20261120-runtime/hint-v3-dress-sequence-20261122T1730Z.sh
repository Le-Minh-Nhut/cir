#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/heheboiz/data/cir
RUN="$ROOT/workbench/docs/reviews/20261112-parity-repair/raw/third_model/hint-v3-dress-identity-20261122T172030Z"
PLAN="$RUN/evidence/g1_g2.hint.dress.v3-plan.json"
EXPECTED_PLAN=e975ed9a87bd621e5e45880f560b6365957955a307b4e9b77f52aa7302fb4495
PY="$ROOT/workbench/envs/ctl312h/bin/python"
ACTUAL_PLAN=$(sha256sum "$PLAN" | cut -d' ' -f1)
test "$ACTUAL_PLAN" = "$EXPECTED_PLAN"
test "$(jq -r '.category' "$PLAN")" = dress
test "$(jq -r '.population_scope.query_count' "$PLAN")" = 2017
test "$(jq -r '.population_scope.gallery_count' "$PLAN")" = 2628
test "$(jq '.runs|length' "$PLAN")" = 10
jq -r '.source[]|[.path,.sha256]|@tsv' "$PLAN" | while IFS=$'\t' read -r path expected; do
  test "$(sha256sum "$ROOT/$path" | cut -d' ' -f1)" = "$expected"
done
CHECKPOINT=$(jq -r '.path' "$RUN/raw/checkpoint_identity_hint_fashioniq.json")
CHECKPOINT_SHA=$(jq -r '.checkpoint_sha256' "$PLAN")
test "$(sha256sum "$CHECKPOINT" | cut -d' ' -f1)" = "$CHECKPOINT_SHA"
ROLES=(native_batch_1 native_batch_n native_repeat_1 native_repeat_2 native_adapter_native adapter_batch_1 adapter_batch_n adapter_repeat_1 adapter_repeat_2 native_adapter_adapter)
for role in "${ROLES[@]}"; do
  test -n "$(jq -r --arg role "$role" '.runs[$role].execution.command' "$PLAN")"
  test ! -e "$RUN/raw/roles/dress/$role"
done
printf 'ADMISSION identity=exact plan_sha256=%s query_count=2017 gallery_count=2628 sample_indices=[0,14,18,816,2016] gallery_sample=20 role_count=10\n' "$ACTUAL_PLAN"
for role in "${ROLES[@]}"; do
  command=$(jq -r --arg role "$role" '.runs[$role].execution.command' "$PLAN")
  printf 'ROLE_START %s %s\n' "$role" "$(date -Is)"
  bash -lc "$command"
  printf 'ROLE_EXIT=0 %s %s\n' "$role" "$(date -Is)"
done
printf 'AGGREGATE_START %s\n' "$(date -Is)"
"$PY" "$ROOT/workbench/docs/reviews/20261112-parity-repair/scripts/third_model_parity.py" --model hint --phase aggregate --role aggregate --category dress --run-root "$RUN" --controller-interpreter "$PY"
printf 'GATE_START %s\n' "$(date -Is)"
PYTHONPATH="$ROOT" "$PY" - <<'PY'
import json
from pathlib import Path
from workbench.backend.parity_gate import evaluate_model
root = Path('/home/heheboiz/data/cir/workbench/docs/reviews/20261112-parity-repair/raw/third_model/hint-v3-dress-identity-20261122T172030Z/evidence')
verdict = evaluate_model('hint', categories=('dress',), root=root)
print(json.dumps(verdict.to_dict(), indent=2, sort_keys=True))
PY
