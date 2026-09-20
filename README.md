# TAPER: Sequential Edit-Slot Reasoning for Composed Image Retrieval

This repository implements TAPER, a composed image retrieval (CIR) model that
retrieves a target image from a reference image and a textual modification.
TAPER builds edit slots from contextualized text, executes the active edits
sequentially over a reference state, and scores cached CSMCIR gallery tokens.

The CSMCIR teacher is frozen. Image and text features are precomputed, while
TAPER's slot construction, gating, routing, state transitions, and query head
are trained end to end with retrieval supervision.

## Repository layout

```text
conf/                         Hydra datasets, protocols, and experiments
src/cache/                    Feature-cache loading utilities
src/datasets/                 FashionIQ and CIRR annotations/loaders
src/evaluation/               Legacy and ENCODER-compatible metrics
src/models/taper.py           TAPER model and retrieval scoring
src/teachers/                 Frozen CSMCIR integration
src/training/engine.py        Shared end-to-end training loop
src/precompute_csmcir_stage1.py
                              FashionIQ image feature precomputation
src/precompute_taper_e2e_text.py
                              FashionIQ contextual text precomputation
src/precompute_cirr.py        CIRR image and text precomputation
src/train_fashioniq_encoder.py
src/train_cirr_encoder.py     Final ENCODER-compatible training entrypoints
src/evaluate_fashioniq_encoder.py
src/evaluate_cirr_encoder.py Final validation entrypoints
src/predict_cirr_encoder_test.py
                              Official CIRR test1 JSON generation
teacher/README.md             Expected external CSMCIR layout
tests/unit/                   Protocol and model unit tests
```

The older `src/train.py`, `src/train_cirr.py`, and the non-encoder evaluation
modules remain available for legacy experiment reproduction. The commands
below use the final isolated ENCODER-compatible pipelines.

## Environment setup

Python 3.11 or newer is required. Install the PyTorch/torchvision build that
matches the local CUDA driver first, following the official PyTorch
instructions. Then install this project:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The project declares only its direct dependencies in `pyproject.toml`.
CSMCIR is an external research dependency and may require additional packages
from its own environment (for example its LAVIS stack). Install those versions
according to the upstream CSMCIR distribution; do not replace them with an
arbitrary LAVIS release.

## External CSMCIR dependency

Place the upstream CSMCIR source tree and FashionIQ checkpoint as follows:

```text
teacher/
├── repos/CSMCIR/
│   └── src/
│       ├── data_utils_csmcir.py
│       └── lavis/
└── checkpoints/csmcir/
    └── fashioniq_tuned_clip_best.pt
```

The checkpoint is distributed in the authors' Hugging Face repository
`peng12138/CSMCIR`. The source tree must provide the model name
`blip2_cir_align_prompt_csmcir`. See [teacher/README.md](teacher/README.md) for
the loader contract. Both source and weights are intentionally ignored by Git.

To use other locations, override `--csmcir-root` and `--checkpoint` during
precomputation, and override
`experiment.teacher.csmcir_root`/`experiment.teacher.checkpoint_path` during
training or evaluation.

## Dataset preparation

By default, datasets live below `data/` and feature caches below `features/`.
Hydra paths can instead be controlled with `CIR_DATA_ROOT`, `CIR_CACHE_ROOT`,
or explicit `dataset.root`/`paths.cache_root` overrides.

### FashionIQ

Expected layout:

```text
data/FashionIQ/
├── captions/
│   ├── cap.dress.{train,val,test}.json
│   ├── cap.shirt.{train,val,test}.json
│   ├── cap.toptee.{train,val,test}.json
│   ├── correction_dict_dress.json
│   ├── correction_dict_shirt.json
│   └── correction_dict_toptee.json
├── image_splits/
│   └── split.{dress,shirt,toptee}.{train,val,test}.json
└── images/
```

The three correction dictionaries are the ENCODER normalization dictionaries,
not files from the original FashionIQ release. They must match the dictionaries
used to produce the reported caches and checkpoints.

### CIRR

Expected annotation layout:

```text
data/CIRR/
├── captions/
│   └── cap.rc2.{train,val,test1}.json
├── image_splits/
│   └── split.rc2.{train,val,test1}.json
└── {train,dev,test1}/...      # official layout
```

The alternative `data/CIRR/img_raw/{train,dev,test1}/...` layout is also
supported. Image paths are resolved from each `split.rc2.*.json`; do not rewrite
the split ordering.

## Feature precomputation

### FashionIQ image caches

```bash
python src/precompute_csmcir_stage1.py \
  --dataset-root data/FashionIQ \
  --csmcir-root teacher/repos/CSMCIR \
  --checkpoint teacher/checkpoints/csmcir/fashioniq_tuned_clip_best.pt \
  --output-root features/fashioniq/csmcir \
  --batch-size 32 \
  --device cuda
```

Despite its historical filename, this script produces the `retrieval` and
`native` image caches required by the final end-to-end pipeline.

### FashionIQ text caches

Run after the image caches exist:

```bash
python src/precompute_taper_e2e_text.py \
  --dataset-root data/FashionIQ \
  --cache-root features/fashioniq/csmcir \
  --csmcir-root teacher/repos/CSMCIR \
  --checkpoint teacher/checkpoints/csmcir/fashioniq_tuned_clip_best.pt \
  --batch-size 32 \
  --device cuda
```

### CIRR caches

Generate train, validation, and test1 caches in one run:

```bash
python src/precompute_cirr.py \
  --dataset-root data/CIRR \
  --cache-root features/cirr/csmcir \
  --splits train val test1 \
  --csmcir-root teacher/repos/CSMCIR \
  --checkpoint teacher/checkpoints/csmcir/fashioniq_tuned_clip_best.pt \
  --batch-size 32 \
  --device cuda
```

The resulting cache contract is:

```text
features/{fashioniq,cirr}/csmcir/{split}/
├── retrieval/{images.npy,name_to_idx.json}
├── native/{images.npy,name_to_idx.json}
└── text/{states.npy,teacher_states.npy,attention_mask.npy,
         content_mask.npy,sample_to_idx.json,captions.json,manifest.json}
```

## Training

FashionIQ:

```bash
python src/train_fashioniq_encoder.py \
  dataset=fashioniq \
  protocol=fashioniq_encoder \
  experiment=taper_e2e
```

CIRR:

```bash
python src/train_cirr_encoder.py \
  dataset=cirr \
  protocol=cirr_encoder \
  experiment=taper_e2e
```

The orthogonal slot-query experiment is selected with
`experiment=taper_e2e_ortho07`. Checkpoints and metrics are written below the
Hydra run directory:

```text
outputs/YYYY-MM-DD/HH-MM-SS/fashioniq/encoder/{best.pt,last.pt,metrics_*.json}
outputs/YYYY-MM-DD/HH-MM-SS/cirr/encoder/{best.pt,last.pt,metrics_*.json}
```

Checkpoint files contain TAPER state only; the frozen teacher is reloaded from
the configured external source/checkpoint.

## Evaluation

FashionIQ validation:

```bash
python src/evaluate_fashioniq_encoder.py \
  dataset=fashioniq \
  protocol=fashioniq_encoder \
  experiment=taper_e2e \
  +checkpoint=/absolute/path/to/best.pt \
  +metrics_output=/absolute/path/to/fashioniq_metrics.json
```

This reports per-category R@1, R@10, and R@50 plus the unchanged encoder
checkpoint-selection score.

CIRR validation:

```bash
python src/evaluate_cirr_encoder.py \
  dataset=cirr \
  protocol=cirr_encoder \
  experiment=taper_e2e \
  +checkpoint=/absolute/path/to/best.pt \
  +metrics_output=/absolute/path/to/cirr_metrics.json
```

This reports R@1/5/10/50, subset Rs@1/2/3, the existing selection score, and
`mean_recall = (R@5 + Rs@1) / 2`. Reference exclusion and subset filtering use
the benchmark protocol implemented in `src/evaluation/cirr_encoder.py`.

Use the same experiment config that produced a checkpoint. For example, pass
`experiment=taper_e2e_ortho07` when evaluating an orthogonal-loss checkpoint.

## CIRR test1 submission

Test1 has no public target labels, so this command only produces rankings:

```bash
python src/predict_cirr_encoder_test.py \
  dataset=cirr \
  protocol=cirr_encoder \
  experiment=taper_e2e \
  +checkpoint=/absolute/path/to/best.pt \
  +submission_output_dir=/absolute/path/to/submissions/cirr \
  +submission_name=taper_test1
```

Output files:

```text
CIRR_pred_ranks_recall_taper_test1.json
CIRR_pred_ranks_recall_subset_taper_test1.json
```

TAPER's `model.retrieve()` scores the full test gallery. The reference is
removed, global top-50 is emitted, and subset top-3 is obtained by filtering
that same sorted global ranking by the query's `img_set.members`. The subset is
not rescored.

## Verification

Checks that do not require datasets or checkpoints:

```bash
python -m compileall src tests
pytest -q
ruff check src tests
```

The unit suite covers TAPER slot-query orthogonality, FashionIQ and CIRR metric
semantics, encoder/legacy isolation, CIRR reference exclusion, global/subset
ordering, and official test1 JSON metadata.

Three optional parity utilities require the external CSMCIR code, checkpoint,
and FashionIQ caches:

```text
src/check_csmcir_compose_parity.py
src/check_taper_text_cache_parity.py
src/check_taper_chunk_parity.py
```

## Reproducibility notes

- The default random seed is 42 and deterministic PyTorch execution is enabled.
- Caption strings are checked against the text-cache manifest at runtime.
- Text-cache manifests record the CSMCIR checkpoint SHA-256.
- Gallery ordering comes from the benchmark split JSON files.
- CSMCIR parameters remain frozen and in evaluation mode.
- Dataset images, caches, checkpoints, Hydra outputs, W&B runs, reports, and
  generated submissions are intentionally not version-controlled.

## License and acknowledgements

This repository is released under the [MIT License](LICENSE). It relies on the
FashionIQ and CIRR benchmarks and on the external CSMCIR implementation and
weights. Follow the licenses and citation requirements of those projects when
using this code.
