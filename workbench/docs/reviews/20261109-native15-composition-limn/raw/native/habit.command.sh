#!/bin/bash
cd /home/heheboiz/data/cir
export WORKBENCH_RUNTIME_ADAPTATION=sm_120 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
C=workbench/artifacts/checkpoint_selection/20261106T002617+0700/cohort/downloads
OUT=workbench/docs/reviews/20261109-native15-composition-limn/raw/native
START=$(date +%s)
workbench/envs/ilearn39-adapted/bin/python workbench/replay/habit.py \
  --checkpoint-root $C/habit \
  --dataset-root data/FashionIQ \
  --source-root workbench/third_party/HABIT \
  --variant fiq_n02 \
  --batch-size 64 \
  --output $OUT/habit.fiq_n02.json
CODE=$?
END=$(date +%s)
echo "EXITCODE=$CODE WALL_SECONDS=$((END-START))" 
echo "EXITCODE=$CODE WALL_SECONDS=$((END-START))" >> $OUT/habit.meta.txt
