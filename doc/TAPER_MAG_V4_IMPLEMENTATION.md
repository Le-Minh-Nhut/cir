# TAPER-MAG V4 implementation notes

This branch is a clean implementation of
`CIR_TAPER_CANONICAL_END_TO_END_METHOD_SPEC_V4_2026-08-28.md` with the explicit
FG-CLIP2-Base/text-finetuning experiment overrides.

## Pinned backbone contract

- model: `qihoo360/fg-clip2-base`
- revision: `430fbc8a912c86fd4de601381b6245a0edab22f0`
- runtime-inspected text/vision/retrieval dimensions: 768/768/768
- runtime-inspected text/vision blocks: 12/12
- text traversal: `short`, max length 64, online during training
- default trainable text modules: blocks 8–11 plus final text LayerNorm
- default frozen text module: short-walk projection head (not used by TAPER token input)
- vision backbone, dense-feature head, official global image pooler: frozen
- patch policy: official dynamic budgets; dense tokens are ragged patch tokens with no CLS
- TAPER width/query count: 256/4; retrieval output: normalized 768-D

The runtime smoke observed different real token counts (for example 567 and 560), so caches store a
flat ragged tensor plus offsets/spatial shapes. Cache manifests are immutable and partial debug
caches are marked `complete_split=false`; training rejects them.

## Scientific boundaries

There is no slot ownership, QASA, hard routing, entmax, primitive bank, semantic slot label,
independently writable global state, target input to policy, learned free STOP logit, or teacher
gradient. ASROA is not implemented or enabled.

Curriculum advancement is explicit in config (`actor_warmup`, `utility_shadow`, `critic_warmup`,
`dagger_t2`, `st_bridge`, `harden`) rather than automatic. Change stage/horizon/oracle mix only after
the preceding health gate has been reviewed.

## Entry points

```bash
PYTHONPATH=src python src/precompute_taper_mag_vision.py --dataset-root data/fashionIQ_dataset --cache-root features --split train
PYTHONPATH=src python src/precompute_taper_mag_vision.py --dataset-root data/fashionIQ_dataset --cache-root features --split val
pytest -q
PYTHONPATH=src python src/train.py --config conf/taper_mag_v4_base.yaml
PYTHONPATH=src python src/train.py --config conf/taper_mag_v4_base.yaml --max-train-samples 32
PYTHONPATH=src python src/train.py --config conf/taper_mag_v4_base.yaml --resume runs/taper-mag-v4-fgclip2-base-last4/last.ckpt
PYTHONPATH=src python src/train.py --config conf/taper_mag_v4_base.yaml --resume runs/taper-mag-v4-fgclip2-base-last4/best_retrieval_valid.ckpt --validate-only
```

The supplied FashionIQ tree does not currently contain the three configured
`correction_dict_{dress,shirt,toptee}.json` files. The main configs deliberately use the validated
`correction_policy=fashioniq` and therefore fail loudly until those exact dictionaries are supplied.
`correction_policy=none` is an explicit supported control, not a silent fallback.
