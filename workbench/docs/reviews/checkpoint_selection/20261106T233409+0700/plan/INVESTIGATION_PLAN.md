# Pre-fix investigation record — session 20261106T233409+0700

Written **before** any new full-run score was viewed in this session. Previous session
(`20261106T002617+0700`) is preserved unmodified and is the historical baseline.

## Repository identity at start
- branch `research/cir-failure-workbench`, HEAD `9cce69fb00e123e8c86d210c8c1b41d26701b5cf`
- working tree: only `CHECKPOINT_SELECTION_REPORT.md` may be modified by this session
- GPU: RTX 5070 Ti, 16303 MiB, driver 580.173.02; CUDA 12.8 wheels, torch 2.7.1+cu128
- No shared registry/protocol/wrapper default is to be changed without a recorded decision.

## HINT

**Claim under test.** macro R@10 `59.8034`, R@50 `80.1308` vs paper `58.48 / 77.93`.

**Hypotheses, ordered:**
- H1 (reproducibility): the documented rerun command is wrong, not the result —
  the report's LA VIS loop passes `--variant` to a parser that has no such flag (exit 2,
  reproduced). *Predicted:* dedicated command reproduces the stored measurement.
- H2 (module resolution): `__import__("datasets")` resolves to the HuggingFace `datasets`
  package instead of the pinned `HINT/datasets.py`, silently changing query/gallery
  construction. *Predicted:* `datasets.__file__` is inside `third_party/HINT`.
- H3 (protocol): the native evaluator's `sims.squeeze().max(-1)` is not what the wrapper
  claims, e.g. it collapses queries instead of Q-Former tokens. *Predicted:* measured shapes
  show `[N,M,32]` → max over the 32 tokens → `[N,M]`, i.e. per-token max is the native op.
- H4 (state): dropout or train-mode leaks into scoring. *Predicted:* `model.training == False`.

**Acceptance criteria.**
1. The documented command exits 0 as written and writes a fresh result file.
2. `datasets`, `test`, `data_utils` all resolve inside the pinned checkout.
3. Independently hand-transcribed native ranking reproduces stored R@10/R@50 to <1e-9.
4. Recomputing macro from stored per-query ranks equals the reported macro.
5. A second batch size changes nothing beyond float-associativity noise (<1e-6).

**Not allowed:** tuning toward the paper; changing caption order; dropping queries.

## PTHA/MTST  (CIRR-trained → FashionIQ evaluation, cross-dataset transfer)

**Claim under test.** `BLOCKED_ARCHITECTURAL_MISMATCH` because "the checkpoint supplies 768
where 256 is expected".

**Hypotheses:**
- P1 (already contradicted): the checkpoint lacks a usable projection head. *Contradicted
  before scoring:* checkpoint holds `vision_proj.weight (256,768)` (196 864 params) and loads
  with `strict=True`; the earlier run failed **after** a successful strict load.
- P2 (leading): the effective `extract_target_features` (the *second* definition wins) returns
  **unprojected** Q-Former states `[B,32,768]`, while `inference()` expects the projected
  `[B,32,256]`; the trained `vision_proj` is orphaned for that call path.
- P3: the failure is a config/class mismatch (wrong class for this checkpoint). *Predicted if
  true:* the state dict would not load strictly, or keys would be missing.
- P4: causal confusion with the sibling class `blip2_qformer_tgir_concatembedding_sim`, which
  projects inside `inference()`. *Predicted:* a universal projection fix would double-project it.

**Acceptance criteria.**
1. Real before/after operand shapes recorded on a real FashionIQ batch.
2. Projection applied only through the checkpoint's own trained `vision_proj`, at the boundary
   where the source expects projected features; no new/random weights, no slicing/padding.
3. `inference_target(unprojected)` vs the wrapper's boundary must agree elementwise (max abs
   diff 0.0) on identical real tensors.
4. Double-projection guard: a second application must be detected/refused.
5. Full 3-category transfer run with 100% coverage, ranks preserved, macro recomputed.

**Label stays** `CIRR-trained → FashionIQ evaluation, cross-dataset transfer`. Not a
FashionIQ-trained reproduction.

## PAIR
Preserve all 4 measurements. Correct only causal overstatement: report must not claim the
2×4 transformer or the 5.20% untouched parameters *prove* incompatibility, only that full
original-source equivalence is unresolved.

## LIMN / TME / DCNet
Make proven session repairs model-scoped and reproducible; no global default changes.
