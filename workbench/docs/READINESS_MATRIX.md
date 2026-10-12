# Registry readiness matrix (checkpoint policy: strict_clean)

- dataset root: `/home/heheboiz/data/cir/data/FashionIQ`
- upstream root: `/home/heheboiz/data/cir/workbench/third_party`
- checkpoint root: `/home/heheboiz/data/cir/workbench/artifacts/checkpoints`
- models: 16

## States verified

| State | Models |
|---|---|
| SOURCE_VERIFIED | 15 |
| ENVIRONMENT_PLANNED | 15 |
| ENVIRONMENT_INSTALLED | 0 |
| RUNTIME_VERIFIED | 0 |
| GPU_VERIFIED | 0 |
| CHECKPOINT_READY | 0 |
| EVALUATOR_READY | 0 |
| SCIENTIFICALLY_REPRODUCED | 0 |

## Per model

| Model | Protocol | Source pin | Env group | Env state | Selected checkpoints | Evaluator | Blocker |
|---|---|---|---|---|---|---|---|
| `csmcir` | fashioniq_original_split | 774f94e2076f | cir-6e01f50ea9b1f2a1 | SOLVER_COMPATIBLE | fashioniq | BLOCKED | FashionIQ fashioniq_standard requirement missing: /home/heheboiz/data/cir/data/FashionIQ/captions; FashionIQ fashioniq_standard requirement missing: / |
| `airknow` | fashioniq_full_gallery_ref_excluded | 6ab6fa95b746 | cir-810793e28bed4b5c | SOLVER_COMPATIBLE | — | BLOCKED | FashionIQ fashioniq_ilearn_resized requirement missing: /home/heheboiz/data/cir/data/FashionIQ/captions; FashionIQ fashioniq_ilearn_resized requiremen |
| `conesep` | fashioniq_full_gallery_ref_excluded | bd4bf629ebe6 | cir-810793e28bed4b5c | SOLVER_COMPATIBLE | — | BLOCKED | FashionIQ fashioniq_ilearn_resized requirement missing: /home/heheboiz/data/cir/data/FashionIQ/captions; FashionIQ fashioniq_ilearn_resized requiremen |
| `ptha_mtst` | — | none | — | — | — | BLOCKED | source unavailable, dirty, or not pinned; environment: source unavailable and no checkpoint variants; source: registry model record for ptha_mtst (sou |
| `habit` | fashioniq_full_gallery_ref_excluded | 6ddc2cd63eac | cir-810793e28bed4b5c | SOLVER_COMPATIBLE | — | BLOCKED | FashionIQ fashioniq_ilearn_resized requirement missing: /home/heheboiz/data/cir/data/FashionIQ/captions; FashionIQ fashioniq_ilearn_resized requiremen |
| `intent` | fashioniq_full_gallery_ref_excluded | 2a75fc6f57d1 | cir-3fd550bd5e4d35ed | SOLVER_COMPATIBLE | — | BLOCKED | FashionIQ fashioniq_ilearn_resized requirement missing: /home/heheboiz/data/cir/data/FashionIQ/captions; FashionIQ fashioniq_ilearn_resized requiremen |
| `hint` | fashioniq_val_split | bec50b6c8c19 | cir-810793e28bed4b5c | SOLVER_COMPATIBLE | fashioniq | BLOCKED | FashionIQ fashioniq_ilearn_resized requirement missing: /home/heheboiz/data/cir/data/FashionIQ/captions; FashionIQ fashioniq_ilearn_resized requiremen |
| `encoder` | fashioniq_val_split | 29a2a31d6a56 | cir-18b83b96e59da7cb | SOLVER_COMPATIBLE | — | BLOCKED | FashionIQ fashioniq_ilearn_resized requirement missing: /home/heheboiz/data/cir/data/FashionIQ/captions; FashionIQ fashioniq_ilearn_resized requiremen |
| `pair` | fashioniq_val_split | bfb0b98cdb4c | cir-6a03c13ceccd2be7 | SOLVER_COMPATIBLE | — | BLOCKED | FashionIQ fashioniq_ilearn_resized requirement missing: /home/heheboiz/data/cir/data/FashionIQ/captions; FashionIQ fashioniq_ilearn_resized requiremen |
| `clvc_net` | fashioniq_val_split | bd9b68894898 | cir-fd52d7701637810a | SOLVER_COMPATIBLE | — | BLOCKED | raw benchmark input missing: /home/heheboiz/data/cir/data/FashionIQ/captions/cap.dress.train.json; raw benchmark input missing: /home/heheboiz/data/ci |
| `dcnet` | fashioniq_full_gallery_ref_excluded | 68c79d38569f | cir-ef7df5af7640f33e | SOLVER_COMPATIBLE | — | BLOCKED | raw benchmark input missing: /home/heheboiz/data/cir/data/FashionIQ/captions/cap.dress.train.json; raw benchmark input missing: /home/heheboiz/data/ci |
| `combiner_rn50x4_noft` | fashioniq_original_split | dfed9f748a8a | cir-b73d398bd0cd416e | SOLVER_COMPATIBLE | — | BLOCKED | raw benchmark input missing: /home/heheboiz/data/cir/data/FashionIQ/captions/cap.dress.val.json; raw benchmark input missing: /home/heheboiz/data/cir/ |
| `clip4cir_rn50x4_fullft` | fashioniq_original_split | dfed9f748a8a | cir-b73d398bd0cd416e | SOLVER_COMPATIBLE | — | BLOCKED | raw benchmark input missing: /home/heheboiz/data/cir/data/FashionIQ/captions/cap.dress.val.json; raw benchmark input missing: /home/heheboiz/data/cir/ |
| `tgcir` | fashioniq_val_split | 65fa78eaf8ca | cir-1c8839f691612ed0 | SOLVER_COMPATIBLE | — | BLOCKED | raw benchmark input missing: /home/heheboiz/data/cir/data/FashionIQ/captions/cap.dress.train.json; raw benchmark input missing: /home/heheboiz/data/ci |
| `sprc` | fashioniq_original_split | 2935a5397732 | cir-6e01f50ea9b1f2a1 | SOLVER_COMPATIBLE | — | BLOCKED | raw benchmark input missing: /home/heheboiz/data/cir/data/FashionIQ/captions/cap.dress.val.json; raw benchmark input missing: /home/heheboiz/data/cir/ |
| `limn` | fashioniq_val_split | 7d7bc9b116f5 | cir-1bbd9872cf166f0d | SOLVER_COMPATIBLE | — | BLOCKED | raw benchmark input missing: /home/heheboiz/data/cir/data/FashionIQ/captions/cap.dress.train.json; raw benchmark input missing: /home/heheboiz/data/ci |
