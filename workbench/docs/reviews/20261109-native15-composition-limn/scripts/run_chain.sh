#!/usr/bin/env bash
# Sequential GPU work for the LIMN lane: one job at a time on this shared host.
#   1. wait for any already-running LIMN job of this lane to finish (no overlapping GPU work);
#   2. native evaluation for the categories whose JSON is not yet on disk;
#   3. G1/G2 parity harness.
# Everything appends to raw/ and writes its own JSON, so a failure leaves prior results intact.
set -u
R=/home/heheboiz/data/cir/workbench/docs/reviews/20261109-native15-composition-limn
cd /home/heheboiz/data/cir || exit 1

ENVIRON=(
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
  WORKBENCH_DECODE_WORKERS=2 WORKBENCH_LIMN_BATCH_SIZE=16 WORKBENCH_RUNTIME_ADAPTATION=sm_120
  PYTHONUNBUFFERED=1
)

wait_for_quiet() {
  local pattern="$1"
  while pgrep -f "$pattern" >/dev/null 2>&1; do sleep 10; done
}

wait_for_quiet "t3_native_eval.py --category shirt"

for category in toptee dress; do
  if [ -f "$R/limn_native.$category.json" ]; then
    echo "[skip] $category already measured" >> "$R/raw/t3_chain.log"
    continue
  fi
  echo "[start] $category $(date -Is)" >> "$R/raw/t3_chain.log"
  env "${ENVIRON[@]}" workbench/envs/limn310-adapted/bin/python \
      "$R/scripts/t3_native_eval.py" --category "$category" \
      > "$R/raw/t3_native_$category.log" 2>&1
  echo "[done] $category exit=$? $(date -Is)" >> "$R/raw/t3_chain.log"
done

# Aggregate the per-category evidence into the single report the docs cite.
env "${ENVIRON[@]}" workbench/envs/limn310-adapted/bin/python "$R/scripts/t3_native_eval.py" --aggregate-only \
    > "$R/raw/t3_native_aggregate.log" 2>&1
echo "[done] aggregate exit=$? $(date -Is)" >> "$R/raw/t3_chain.log"

echo "[start] parity $(date -Is)" >> "$R/raw/t3_chain.log"
env "${ENVIRON[@]}" workbench/envs/limn310-adapted/bin/python "$R/scripts/t4_parity.py" \
    > "$R/raw/t4_parity.log" 2>&1
echo "[done] parity exit=$? $(date -Is)" >> "$R/raw/t3_chain.log"
