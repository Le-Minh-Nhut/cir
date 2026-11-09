#!/bin/bash
# Native FashionIQ validation of clip4cir_rn50x4_fullft.
# Same entrypoint, same flags, same cwd as the sibling combiner_rn50x4_noft run
# (run_id 20261108T064137301769Z_combiner_rn50x4_noft_..._020a70b0, return_code 0),
# with --clip-model-path added so the fine-tuned CLIP tower is used instead of stock.
set -u
ROOT=/home/heheboiz/data/cir
CKPT=$ROOT/workbench/artifacts/checkpoint_selection/20261106T002617+0700/checkpoints/clip4cir/FashionIQ/RN50x4_fullft
OUT=$ROOT/workbench/docs/reviews/20261109-native15-composition-limn/raw/native
PY=$ROOT/workbench/artifacts/environments/clip4cir-38/bin/python
export TMPDIR=/tmp/cir-native15-runs/clip4cir_fullft
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
mkdir -p "$TMPDIR"
cd "$ROOT/workbench/third_party/CLIP4Cir"
START=$(date +%s)
/usr/bin/time -v -o "$OUT/clip4cir_fullft.time.txt" \
  "$PY" src/validate.py \
    --dataset fashionIQ \
    --combining-function combiner \
    --combiner-path "$CKPT/fiq_comb_RN50x4_fullft.pt" \
    --projection-dim 2560 \
    --hidden-dim 5120 \
    --clip-model-name RN50x4 \
    --clip-model-path "$CKPT/fiq_clip_RN50x4_fullft.pt" \
    --target-ratio 1.25 \
    --transform targetpad \
  >"$OUT/clip4cir_fullft.stdout.log" 2>"$OUT/clip4cir_fullft.stderr.log"
CODE=$?
END=$(date +%s)
echo "EXITCODE=$CODE WALL_SECONDS=$((END-START))" >> "$OUT/clip4cir_fullft.meta.txt"
rm -rf "$TMPDIR"
