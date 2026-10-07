# Model environments

Do not install upstream models into workbench backend environment. Build a separate environment from each pinned source tree on GPU host after checkpoint provenance is confirmed.

Known upstream guidance from audited READMEs:

- CSMCIR: Python 3.9, Torch 2.0.1, torchvision 0.15.2, upstream `requirements.txt`.
- ENCODER: Python 3.9 and upstream `requirements.txt`; source warns OpenCLIP version changes metrics.
- PAIR: Python 3.8.10, Torch 2.0.0; upstream repository does not currently expose a verified FashionIQ checkpoint mapping.
- Air-Know, ConeSep, HABIT, INTENT, HINT: audited READMEs document Python 3.8 and Torch 2.1.0/CUDA 12.1-era LAVIS stacks.
- HINT: Python 3.8.10, CUDA 12.6, Torch 2.1.0/CUDA 12.1-compatible, OpenCLIP 2.24.0, Transformers 4.25.0, LAVIS 1.0.2, timm 0.9.16 according to its README.
- NEUCORE: Python 3.8, CUDA 11.1, `requirements.txt`, NLTK concept extraction, repository-local data/vocab/checkpoint paths, and a Google Drive model folder with no verified FashionIQ file mapping. It is not an integrated runnable workbench model.
- CLVC-Net: Python 3.7.6, Torch 1.6.0, CUDA; pretrained ResNet download/cache and checkpoint loading command remain unresolved.
- DCNet: Python 3.7.7, Torch 1.4.0, torchvision 0.5.0, spaCy 2.3.0 plus `en_vectors_web_lg`, NLTK `punkt`; generated GloVe PKLs required.
- Combiner / CLIP4Cir: Python 3.8, Torch 1.11.0, torchvision 0.12.0, OpenAI CLIP; target checkpoint pairing unresolved.
- TG-CIR: `clip==0.2.0`, Torch 1.7.0, torchvision 0.8.0, CUDA; archive checkpoint member mapping unresolved.
- SPRC: Python 3.9, Torch 2.0.1, torchvision 0.15.2, source `requirements.txt`; selected BLIP-2 model/checkpoint mapping unresolved.
- LIMN base iteration 0: CUDA 12.4, Torch 1.12.1, torchvision 0.13.1, open-clip-torch 2.20.0; artifact exists but replay command/loading parity remains unaudited.

These have not been created or executed here. Do not claim compatibility until each official unmodified evaluation has run.
