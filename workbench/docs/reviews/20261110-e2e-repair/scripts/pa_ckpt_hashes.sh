#!/bin/bash
# Phase A: recompute checkpoint digests actually cited by the protocol matrix.
set -u
R=/home/heheboiz/data/cir/workbench/artifacts/checkpoint_selection/20261106T002617+0700
printf "name\tbytes\tsha256\n"
for f in \
  "$R/cohort/downloads/encoder/fashioniq.pt" \
  "$R/cohort/downloads/hint/fashioniq.pt" \
  "$R/checkpoints/limn/0_dress_best_model.pt" \
  "$R/checkpoints/limn/0_shirt_best_model.pt" \
  "$R/checkpoints/limn/0_toptee_best_model.pt" \
  "$R/checkpoints/clip4cir/FashionIQ/RN50x4_fullft/fiq_comb_RN50x4_fullft.pt" \
  "$R/checkpoints/clip4cir/FashionIQ/RN50x4_fullft/fiq_clip_RN50x4_fullft.pt" \
  "$R/checkpoints/dcnet/trained_model.pth" \
  "$R/cohort/downloads/tme/fashioniq_0_best_model.pth" \
  "$R/cohort/downloads/habit/HABIT-FIQ_N0.2.pt" \
  "$R/cohort/downloads/intent/intent_fiq_0.2.pt" \
  "$R/cohort/downloads/conesep/ConeSep-FIQ-N0.2.pt" \
  "$R/cohort/downloads/ptha_mtst/tuned_mtst_cirr.pt" \
  "$R/cohort/downloads/clip4cir/FashionIQ/RN50x4_noft/fiq_comb_RN50x4_noft.pt" \
  "$R/cohort/downloads/airknow/airknow_fiq_0.5.pt" ; do
  if [ ! -f "$f" ]; then printf "%s\tMISSING\tMISSING\n" "$(basename "$f")"; continue; fi
  printf "%s\t%s\t%s\n" "$(basename "$f")" "$(stat -c %s "$f")" "$(sha256sum "$f" | cut -d' ' -f1)"
done
