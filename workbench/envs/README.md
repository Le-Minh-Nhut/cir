# Model environments

Do not install upstream models into workbench backend environment. Build a separate environment from each pinned source tree on GPU host after checkpoint provenance is confirmed.

Known upstream guidance from audited READMEs:

- CSMCIR: Python 3.9, Torch 2.0.1, torchvision 0.15.2, upstream `requirements.txt`.
- ENCODER: Python 3.9 and upstream `requirements.txt`; source warns OpenCLIP version changes metrics.
- PAIR: Python 3.8.10, Torch 2.0.0; upstream repository does not currently expose a verified FashionIQ checkpoint mapping.
- Air-Know, ConeSep, HABIT, INTENT, HINT: audited READMEs document Python 3.8 and Torch 2.1.0/CUDA 12.1-era LAVIS stacks.

These have not been created or executed here. Do not claim compatibility until each official unmodified evaluation has run.
