#!/bin/bash
# Native FashionIQ validation of dcnet (CENet / tirg+fd, val split).
# The only source patch involved is the pre-existing workbench fix in
# workbench/third_party/DCNet/base/base_trainer.py::_resume_checkpoint
# (map_location=self.device; weights_only=False only when torch >= 1.13 exposes it).
# No timeout: the previous attempt was cut at 2400 s (EXIT=124) after "Running test on cpu".
set -u
ROOT=/home/heheboiz/data/cir
CKPT=$ROOT/workbench/artifacts/checkpoint_selection/20261106T002617+0700/checkpoints/dcnet
OUT=$ROOT/workbench/docs/reviews/20261109-native15-composition-limn/raw/native
PY=$ROOT/workbench/artifacts/environments/dcnet-377/bin/python
export TMPDIR=/tmp/cir-native15-runs/dcnet
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
mkdir -p "$TMPDIR"
cd "$ROOT/workbench/third_party/DCNet"
START=$(date +%s)
/usr/bin/time -v -o "$OUT/dcnet.time.txt" \
  "$PY" test.py --resume "$CKPT/" --mode val \
  >"$OUT/dcnet.stdout.log" 2>"$OUT/dcnet.stderr.log"
CODE=$?
END=$(date +%s)
echo "EXITCODE=$CODE WALL_SECONDS=$((END-START))" >> "$OUT/dcnet.meta.txt"
stat -c 'ATT_RESULT_MTIME=%y SIZE=%s' "$CKPT/val/att_result.pt" >> "$OUT/dcnet.meta.txt" 2>&1
rm -rf "$TMPDIR"
