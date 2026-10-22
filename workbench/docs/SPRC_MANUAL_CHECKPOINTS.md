# SPRC FashionIQ checkpoint installation

Download these files yourself from the author links. Both are FashionIQ SPRC checkpoints, but they are separate backbone configurations and must never share a filename or checkpoint ID:

| Variant | Download | Registry ID / destination | Architecture |
|---|---|---|---|
| Main (paper configuration) | [OneDrive](https://1drv.ms/u/s!Aj0q22vyiZbabnUya4mnufIBtYI?e=n4ZVKj); Baidu share password `t1at` | `sprc_fiq_unresolved` / `workbench/artifacts/checkpoints/sprc/sprc_fiq.pt` | `blip2_cir_align_prompt`, `pretrain`, EVA-CLIP ViT-g (`eva_clip_g`) |
| Separate ViT-L variant | [Google Drive](https://drive.google.com/file/d/11p95msuuTAU6Pej7d_a-KjUHHpP3kH8Y/view) | `sprc_fiq_vitl` / `workbench/artifacts/checkpoints/sprc/sprc_fiq_vitl.pt` | `blip2_cir_align_prompt`, `pretrain_vitL`, CLIP ViT-L (`clip_L`) |

The registry's `WORKBENCH_CHECKPOINT_ROOT` defaults to `<repo>/workbench/artifacts/checkpoints`. If you configured another root, use that same root for placement and pass it as `--output-root` below. Put the downloaded main file at `sprc/sprc_fiq.pt`, and the ViT-L file at `sprc/sprc_fiq_vitl.pt` under that root. Example with the default root:

```sh
mkdir -p workbench/artifacts/checkpoints/sprc
cp ~/Downloads/<main-download-name> workbench/artifacts/checkpoints/sprc/sprc_fiq.pt
cp ~/Downloads/<vitl-download-name> workbench/artifacts/checkpoints/sprc/sprc_fiq_vitl.pt
.venv/bin/python workbench/scripts/download_checkpoints.py --model sprc --checkpoint sprc_fiq_unresolved --verify-only
.venv/bin/python workbench/scripts/download_checkpoints.py --model sprc --checkpoint sprc_fiq_vitl --verify-only
```

Override example: add `--output-root "$WORKBENCH_CHECKPOINT_ROOT"` to each command. The downloader's `--verify-only` mode checks presence, nonzero size, and any registry-published digest; it prints a complete local SHA-256. No author SHA-256 is published for either Drive file, so a local digest is **not** an author checksum. The existing downloader does not verify PyTorch serialization or model compatibility; its `REQUIRED PROVENANCE UNRESOLVED` warning is expected and is not a validation pass.

**Acceptance before evaluation:** inspect deserialization without executing pickle objects (`torch.load(..., weights_only=True)`); require a tensor state dictionary under the exact `Blip2QformerCirAlignPrompt` class key used by the pinned evaluator; compare every supplied key and tensor shape against the pinned model instantiated with the corresponding configuration above; report missing, unexpected, and shape-mismatched keys; reject any shape mismatch or wrong/missing class key. Only then record the full-file SHA-256 and request registry metadata/mapping updates. The native evaluator uses `strict=False`, which can silently accept incomplete or wrong-backbone weights; do not treat it as validation. Until that inspection and registry mapping are complete, installation proves only file placement—not a usable checkpoint or a reproducible evaluation. The base SPRC choice is the registered align-prompt model, not an unavailable reranking model.