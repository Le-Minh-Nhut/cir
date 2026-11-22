# PAIR source and checkpoint blocker — 2026-11-21

## Finding

PAIR remains deferred. Available evidence does not establish a complete author-pinned FashionIQ reproduction path or map either local checkpoint to the paper's B1/B2 results.

- Author repository: https://github.com/iLearn-Lab/ICASSP25-PAIR, pinned at `bfb0b98cdb4c44f6b9190baaa84d783a77dd888f`. Public branch API exposes only `main`; tags are empty.
- Author-linked Drive folder: https://drive.google.com/drive/folders/1aVhi7SkZAXCP8bnoD2fQT15Ozrq75xLU?usp=sharing. Available listing does not expose `pair-B1.pt`, `pair-B2.pt`, or a FashionIQ checkpoint mapping. Hugging Face API requests returned HTTP 401.
- Vendored `workbench/third_party/PAIR/models/MHTransformer_rev.py` is absent. Existing reverse-attention code is an experimental reconstruction, not verified author source. No author FashionIQ evaluator CLI is present.
- Local B1/B2 checkpoint hashes appear in the current checkpoint audit, but no source evidence binds either file to author B1/B2 or a FashionIQ configuration. Filename-based mapping is not evidence.

## Decision

Do not run or report PAIR as a paper-faithful FashionIQ reproduction. Do not infer checkpoint mapping, replace the missing author implementation with the experimental reverse-attention reconstruction, or promote parity. Reopen only when the pinned author implementation, evaluator protocol, and source-to-checkpoint/config mapping are available and verifiable.

## Source evidence pointer

`workbench/docs/reviews/20261117-fix-all/ASSET_CAPTION_AUDIT.md:21` records the repository pin, asset access results, missing evaluator/source, and unverified checkpoint mapping. This note scopes those findings to the PAIR blocker; it adds no new upstream retrieval or checkpoint claims.