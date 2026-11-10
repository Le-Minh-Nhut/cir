# ptha_mtst — EXECUTED_PROVEN (log + command + measurement all present)

## workbench/artifacts/checkpoint_selection/20261106T233409+0700/logs/ptha_transfer_run1.log  [719 lines]
### head
   FashionIQ val - ['dress'] dataset in relative mode initialized
   FashionIQ val - ['dress'] dataset in classic mode initialized
   extracting fashionIQ ['dress'] - val index features
   
     0%|          | 0/60 [00:00<?, ?it/s]/home/heheboiz/data/cir/workbench/third_party/MTST_PTHA/MTST_PTHA/src/lavis/models/blip2_models/blip2.py:42: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
     return torch.cuda.amp.autocast(dtype=dtype)
   
     2%|▏         | 1/60 [00:01<01:17,  1.32s/it]
     3%|▎         | 2/60 [00:02<00:56,  1.03it/s]
     5%|▌         | 3/60 [00:02<00:49,  1.16it/s]
     7%|▋         | 4/60 [00:03<00:45,  1.23it/s]
     8%|▊         | 5/60 [00:04<00:42,  1.28it/s]
### tail
       "r10": 54.99,
       "r50": 76.28,
       "mean": 65.64,
       "mismatch": "training dataset/checkpoint differs; numerical context only, not reproduction",
       "differences_transfer_minus_paper": {
         "r10": -17.09215588410696,
         "r50": -16.443610378901163,
         "mean": -16.77288313150406
       }
     }
   }
   exit=0

## measurement macro_metrics: {"r10": 37.897844115893044, "r50": 59.83638962109884, "mean": 48.86711686849594}
## category_metrics: {"dress": {"r10": 33.11849236488342, "r50": 56.02380037307739}, "shirt": {"r10": 38.861629366874695, "r50": 59.02845859527588}, "toptee": {"r10": 41.71341061592102, "r50": 64.45690989494324}}

## command file workbench/docs/reviews/20261109-native15-composition-limn/raw/native/ptha_mtst.command.sh
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


## ptha_mtst.meta.txt: EXITCODE=0 WALL_SECONDS=240 PEAK_VMHWM_KB=27037580 PEAK_GiB=25.79
