#!/bin/bash
cd /home/heheboiz/data/cir
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
C=workbench/artifacts/checkpoint_selection/20261106T002617+0700/cohort/downloads
OUT=workbench/docs/reviews/20261109-native15-composition-limn/raw/native
START=$(date +%s)
workbench/artifacts/environments/cir-005ef4e1b02999a5/bin/python workbench/replay/ptha_mtst.py \
  --checkpoint-root $C/ptha_mtst \
  --dataset-root data/FashionIQ \
  --source-root workbench/third_party/MTST_PTHA \
  --transfer-cirr-to-fashioniq --checkpoint-id cirr_transfer \
  > $OUT/ptha_mtst.json
CODE=$?
END=$(date +%s)
echo "EXITCODE=$CODE WALL_SECONDS=$((END-START))"
echo "EXITCODE=$CODE WALL_SECONDS=$((END-START))" >> $OUT/ptha_mtst.meta.txt
