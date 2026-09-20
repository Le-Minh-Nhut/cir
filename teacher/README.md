# External CSMCIR assets

The project does not vendor CSMCIR source code or model weights. Place the
upstream repository and checkpoint at the paths used by the default Hydra
configuration:

```text
teacher/
├── repos/
│   └── CSMCIR/
│       └── src/
└── checkpoints/
    └── csmcir/
        └── fashioniq_tuned_clip_best.pt
```

`src/teachers/csmcir.py` loads `data_utils_csmcir` and `lavis` from
`teacher/repos/CSMCIR/src`. The checkpoint is expected to be the upstream
CSMCIR FashionIQ checkpoint; it is available from the authors' CSMCIR model
distribution on Hugging Face (`peng12138/CSMCIR`). These locations can be
overridden through the precomputation CLI arguments or
`experiment.teacher.{csmcir_root,checkpoint_path}`.

Both directories are ignored by Git because they contain third-party code and
large binary weights.
