# Registry readiness matrix (checkpoint policy: strict_clean)

- dataset root: `/home/heheboiz/data/cir/data/fashionIQ_dataset`
- upstream root: `/home/heheboiz/data/cir/workbench/third_party`
- checkpoint root: `/home/heheboiz/data/cir/workbench/artifacts/checkpoints`
- models: 16

## States verified

| State | Models |
|---|---|
| SOURCE_VERIFIED | 15 |
| ENVIRONMENT_PLANNED | 14 |
| ENVIRONMENT_INSTALLED | 0 |
| RUNTIME_VERIFIED | 0 |
| GPU_VERIFIED | 0 |
| CHECKPOINT_READY | 0 |
| EVALUATOR_READY | 0 |
| SCIENTIFICALLY_REPRODUCED | 0 |

## Per model

| Model | Protocol | Source pin | Env group | Env state | Selected checkpoints | Evaluator | Blocker |
|---|---|---|---|---|---|---|---|
| `csmcir` | fashioniq_original_split | 774f94e2076f | cir-6e01f50ea9b1f2a1 | SOLVER_COMPATIBLE | fashioniq | BLOCKED | CSMCIR Qwen captions missing: /home/heheboiz/data/cir/data/fashionIQ_dataset/qwen_captions/dress_cot_val.json; CSMCIR Qwen captions missing: /home/heh |
| `airknow` | fashioniq_full_gallery_ref_excluded | 6ab6fa95b746 | cir-4c8b682368302099 | SOLVER_COMPATIBLE | — | BLOCKED | FashionIQ fashioniq_ilearn_resized requirement missing: /home/heheboiz/data/cir/data/fashionIQ_dataset/resized_image; FashionIQ fashioniq_ilearn_resiz |
| `conesep` | fashioniq_full_gallery_ref_excluded | bd4bf629ebe6 | cir-4c8b682368302099 | SOLVER_COMPATIBLE | — | BLOCKED | FashionIQ fashioniq_ilearn_resized requirement missing: /home/heheboiz/data/cir/data/fashionIQ_dataset/resized_image; FashionIQ fashioniq_ilearn_resiz |
| `ptha_mtst` | — | none | — | — | — | BLOCKED | source unavailable, dirty, or not pinned; environment: source unavailable and no checkpoint variants; source: registry model record for ptha_mtst (sou |
| `habit` | fashioniq_full_gallery_ref_excluded | 6ddc2cd63eac | cir-4c8b682368302099 | SOLVER_COMPATIBLE | — | BLOCKED | FashionIQ fashioniq_ilearn_resized requirement missing: /home/heheboiz/data/cir/data/fashionIQ_dataset/resized_image; FashionIQ fashioniq_ilearn_resiz |
| `intent` | fashioniq_full_gallery_ref_excluded | 2a75fc6f57d1 | cir-40c39aa201822895 | SOLVER_COMPATIBLE | — | BLOCKED | FashionIQ fashioniq_ilearn_resized requirement missing: /home/heheboiz/data/cir/data/fashionIQ_dataset/resized_image; FashionIQ fashioniq_ilearn_resiz |
| `hint` | fashioniq_val_split | bec50b6c8c19 | cir-4c8b682368302099 | SOLVER_COMPATIBLE | fashioniq | BLOCKED | FashionIQ fashioniq_ilearn_resized requirement missing: /home/heheboiz/data/cir/data/fashionIQ_dataset/resized_image; FashionIQ fashioniq_ilearn_resiz |
| `encoder` | fashioniq_val_split | 29a2a31d6a56 | — | — | — | BLOCKED | environment: SOLVER_ERROR: solver rejected standalone constraints (python, torch, torchvision, einops, numpy, open-clip-torch, tqdm, transformers): {
 |
| `pair` | fashioniq_val_split | bfb0b98cdb4c | cir-6a03c13ceccd2be7 | SOLVER_COMPATIBLE | — | BLOCKED | FashionIQ fashioniq_ilearn_resized requirement missing: /home/heheboiz/data/cir/data/fashionIQ_dataset/resized_image; FashionIQ fashioniq_ilearn_resiz |
| `clvc_net` | fashioniq_val_split | bd9b68894898 | cir-0c796ec0077c3560 | SOLVER_COMPATIBLE | — | BLOCKED | required generated artifact missing: /home/heheboiz/data/cir/data/fashionIQ_dataset/resized_image/dress; required generated artifact missing: /home/he |
| `dcnet` | fashioniq_full_gallery_ref_excluded | 68c79d38569f | cir-969b4b5e38c644b4 | SOLVER_COMPATIBLE | — | BLOCKED | required generated artifact missing: /home/heheboiz/data/cir/data/fashionIQ_dataset/captions/cap.dress.glove.val.pkl; required generated artifact miss |
| `combiner_rn50x4_noft` | fashioniq_original_split | dfed9f748a8a | cir-71a8cc1dc12aa508 | SOLVER_COMPATIBLE | — | BLOCKED | FashionIQ image extension requirement: 3817 of 3817 dress validation images lack .jpg in /home/heheboiz/data/cir/data/fashionIQ_dataset/images (first: |
| `clip4cir_rn50x4_fullft` | fashioniq_original_split | dfed9f748a8a | cir-71a8cc1dc12aa508 | SOLVER_COMPATIBLE | — | BLOCKED | FashionIQ image extension requirement: 3817 of 3817 dress validation images lack .jpg in /home/heheboiz/data/cir/data/fashionIQ_dataset/images (first: |
| `tgcir` | fashioniq_val_split | 65fa78eaf8ca | cir-2967430f2b81b512 | SOLVER_COMPATIBLE | — | BLOCKED | required generated artifact missing: /home/heheboiz/data/cir/data/fashionIQ_dataset/resized_image/dress; required generated artifact missing: /home/he |
| `sprc` | fashioniq_original_split | 2935a5397732 | cir-6e01f50ea9b1f2a1 | SOLVER_COMPATIBLE | — | BLOCKED | LAVIS BLIP-2 pretrain weights unverified: /home/heheboiz/.cache/lavis/blip2_pretrained.pth (operator-supplied unverified provenance); EVA-ViT-g backbo |
| `limn` | fashioniq_val_split | 7d7bc9b116f5 | cir-1bbd9872cf166f0d | SOLVER_COMPATIBLE | — | BLOCKED | required generated artifact missing: /home/heheboiz/data/cir/data/fashionIQ_dataset/resized_image/dress; required generated artifact missing: /home/he |
