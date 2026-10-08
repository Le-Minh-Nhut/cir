# Model environments

Do not install upstream models into workbench backend environment. Build a separate environment from each pinned source tree on GPU host after checkpoint provenance is confirmed.

Known upstream guidance from audited READMEs:

- CSMCIR: Python 3.9, Torch 2.0.1, torchvision 0.15.2, upstream `requirements.txt`.
- ENCODER: Python 3.9 and upstream `requirements.txt`; source warns OpenCLIP version changes metrics.
- PAIR: Python 3.8.10, Torch 2.0.0; upstream repository does not currently expose a verified FashionIQ checkpoint mapping.
- Air-Know, ConeSep, HABIT, INTENT, HINT: audited READMEs document Python 3.8 and Torch 2.1.0/CUDA 12.1-era LAVIS stacks.
- HINT: Python 3.8.10, CUDA 12.6, Torch 2.1.0/CUDA 12.1-compatible, OpenCLIP 2.24.0, Transformers 4.25.0, LAVIS 1.0.2, timm 0.9.16 according to its README.
- NEUCORE: Python 3.8, CUDA 11.1, `requirements.txt`, NLTK concept extraction, repository-local data/vocab/checkpoint paths, and a Google Drive model folder with no verified FashionIQ file mapping. It is not an integrated runnable workbench model.
- CLVC-Net: Python 3.7.6, Torch 1.6.0 from README; torchvision/CUDA unpinned. Pinned `test.py` has no standalone replay CLI. Evaluation needs externally acquired ResNet-50 ImageNet weights.
- DCNet: Python 3.7.7, Torch 1.4.0, torchvision 0.5.0, NLTK 3.5, spaCy 2.3.0 from README/requirements. Standard evaluation needs ImageNet ResNet-50 weights; spaCy/NLTK assets are preprocessing-only once author GloVe PKLs exist. `--resume` expects `config.json` plus `trained_model.pth` directory.
- Combiner / CLIP4Cir: Python 3.8, Torch 1.11.0, torchvision 0.12.0, pandas 1.4.2, OpenAI CLIP. RN50x4 base CLIP cache is required. Requested noft/fullft artifact mappings remain unresolved; fullft needs separate Combiner and CLIP states.
- TG-CIR: Torch 1.7.0, torchvision 0.8.0, `clip==0.2.0`, numpy 1.22.3, pandas 1.4.2; Python/CUDA unspecified. ViT-B/16 asset/cache required. No source replay CLI.
- SPRC: Python 3.9, Torch 2.0.1, torchvision 0.15.2; source requirements include Transformers 4.36.2, timm 0.9.12, spaCy 3.7.2. LAVIS BLIP-2 pretrain asset required; `sprc_fiq.pt` model/backbone mapping unresolved.
- LIMN base iteration 0: CUDA 12.4, Torch 1.12.1, torchvision 0.13.1, open-clip-torch 2.20.0; Python unspecified. User-provided DataComp OpenCLIP weights and all three category model files required; no replay CLI.

These have not been created or executed here. Do not claim compatibility until each official unmodified evaluation has run.
