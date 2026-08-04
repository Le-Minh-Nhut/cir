# CIR Research Framework

A clean and reproducible research framework for **Composed Image Retrieval (CIR)**.

The project is being developed for experiments on datasets such as:

* FashionIQ
* CIRR
* CIRCO
* HP-FashionIQ

The framework is designed to support multiple vision-language backbones, including a QuRe-compatible BLIP-2 baseline and future alternatives such as FG-CLIP2.

## Project status

The repository is currently in the infrastructure bootstrap stage.

Current work:

* Hydra configuration
* runtime and reproducibility utilities
* dataset contracts
* benchmark protocol implementation
* BLIP-2 baseline reproduction

Training and evaluation commands shown below will become available as their corresponding modules are implemented.

## Design goals

* Explicit Hydra-based experiment configuration
* Multiple datasets and evaluation protocols
* BLIP-2 as a reproducible baseline
* Support for alternative VLM backbones
* Online raw-image training
* Optional frozen-encoder feature caching
* Model-independent benchmark evaluation
* Modular objectives and negative sampling
* Reliable checkpointing and experiment metadata
* Feasibility on a 16 GB GPU

## Planned project structure

```text
cir/
├── conf/
│   ├── config.yaml
│   ├── dataset/
│   ├── model/
│   ├── protocol/
│   ├── objective/
│   └── experiment/
│
├── src/
│   ├── datasets/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   ├── cache/
│   ├── runtime.py
│   ├── precompute.py
│   ├── train.py
│   └── evaluate.py
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── regression/
│
├── outputs/
├── README.md
└── pyproject.toml
```

## Intended workflow

```text
optional feature precomputation
              ↓
           training
              ↓
      validation/evaluation
              ↓
    test submission generation
```

The feature-precomputation path is only valid when the cached encoder stage and its preprocessing remain frozen.

## Configuration

Experiments are configured through Hydra.

The root configuration is located at:

```text
conf/config.yaml
```

Configuration groups include:

```text
dataset/
model/
protocol/
objective/
experiment/
```

A future training command will look like:

```bash
python src/train.py \
  dataset=fashioniq \
  model=blip2_qure \
  protocol=fashioniq_original \
  objective=qure_pairwise
```

A future evaluation command will look like:

```bash
python src/evaluate.py \
  dataset=fashioniq \
  model=blip2_qure \
  protocol=fashioniq_original
```

## Development setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the project and development dependencies:

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

Run lint checks:

```bash
ruff check .
```

Format the code:

```bash
ruff format .
```

## Implementation roadmap

1. Bootstrap Hydra and project tooling
2. Implement runtime utilities
3. Define common CIR sample and batch contracts
4. Implement FashionIQ and CIRR loaders
5. Implement benchmark metrics and golden tests
6. Define the common model interface
7. Reproduce the QuRe-compatible BLIP-2 baseline
8. Implement training with random negatives
9. Add frozen feature precomputation and caching
10. Add QuRe hard-negative mining
11. Add FG-CLIP2
12. Add CIRCO and HP-FashionIQ
13. Add new CIR research mechanisms

## Research references

The initial engineering design studies ideas and behavior from:

* QuRe: Query-Relevant Retrieval through Hard Negative Sampling in Composed Image Retrieval
* TME: Learning with Noisy Triplet Correspondence for Composed Image Retrieval

The framework itself is being independently structured for reusable CIR research.
