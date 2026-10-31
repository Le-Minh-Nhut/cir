# A38 W3 scientific validation

HEAD: `a8af3d2df45cb6d0aff8775cd3402f8c97b742cf`.

Status definitions: `PASS` requires empirical execution evidence (probes, tests, dynamic drivers). Static code/schema inspections without execution evidence are designated `STATICALLY_CHECKED`.
## T1 — ranking

| check | method | status | evidence | finding_id |
|---|---|---|---|---|
| Top-K; direction | Hand-built vectors via real `CompositionAdapter.score` | PASS | `/dev/shm/a38-w3/rank_probe.py`: order `[1,0]`, scores `[0.0,1.0,-1.0]`; higher cosine wins. `adapters.py:141-146` normalizes then dot-products. | none |
| L2 normalization | Magnitude-scaled vector comparison | PASS | `[1.0,0.0]`; `adapters.py:142-145`. | none |
| Exact target rank | Positive at gallery index 1 | PASS | Output `[1]`. | none |
| Ties | Two identical target embeddings | PASS | Target at position 1 receives rank 2; `adapters.py:167-168` breaks ties by earlier gallery positions. | none |
| NaN/Inf | NaN score present | PASS | Rank `[None]`, never rank 1; `adapters.py:160` invalidates rows unless all finite. | none |
| Empty gallery | rank + score on (1,0) feature matrix | PASS | Scores shape `(1,0)`; rank `[None]`. No exception. | none |
| Batch boundaries | Same 33 targets scored together and chunks of 7 | PASS | both exact ranks 19, score arrays equal. Production runner chunks query rows, each still scored against same full target matrix (`runner.py:360-374`). | none |
| Duplicate image IDs | Rank API has target positions, not IDs | NOT_APPLICABLE | Passing repeated positions `[1,1]` returns rank 1; duplicate identity dedup handled by position construction (`runner.py:188-197`). Probe does not establish handling of duplicated gallery ID entries. | none |

## T2 — conditions and caches

| check | method | status | evidence | finding_id |
|---|---|---|---|---|
| Actual condition inputs | Wrapped `encode_queries` seam in `_encode_queries`, image captured from forwarded path | PASS | Probe output: multimodal `(2,3,3), mean 83, min 9, max 180, text `'make red'`; text-only black `(2,3,3), mean/min/max 0, text `'make red'`; image-only `(2,3,3), mean 83, min 9, max 180, text `''`. `inputs.py:19-27`; `runner.py:262-271,302-320`. | none |
| Cache isolation | Replicated feature-key digest construction for each condition | PASS | Exact digests: MM `59b525c0f945969ec4a07e2e6338eff39c9cef4f31834d9916904a807882c2ff`; text `157a316b8beabf126ad48ef643dfa25ab25d285e3b7c8f0cfb95bd985b24cfa4`; image `556a67f72f792bfcbb88931ddc6d75414016e15ce54fc9a38ef1773a2f24e65e`; same-key comparison False. Feature key has condition (`runner.py:735-737`). Gallery feature cache is intentionally condition-independent; it encodes gallery images, not query condition (`runner.py:715-720`). Run identity includes conditions (`runner.py:623-650`). | none |

## T3 — independent metrics

Production definitions quoted: `metrics.py:27-30`: `ndcg_full(rank) = 1.0 / math.log2(rank + 1)`; `metrics.py:73-75`: Recall@10/50 are `100 * hits / scored_count`, nDCG is mean over scored ranks. `metrics.py:138-149`: gap = `1 - max(mean T nDCG, mean I nDCG) / mean MM nDCG` over common all-condition exact-rank cohort after averaging.

Independent implementation at `/dev/shm/a38-w3/metric_probe.py` consumed stored per-query ranks only (no production metric calls), run `1410522b57d4fe79083c60a6`.

| metric | production stored | independent | absolute difference | status |
|---|---:|---:|---:|---|
| MM NDCG | 0.25066906940328237 | 0.25066906940328237 | 0 | PASS |
| T NDCG | 0.12021218172563472 | 0.12021218172563472 | 0 | PASS |
| I NDCG | 0.084990055229265143 | 0.084990055229265143 | 0 | PASS |
| composition gap | 0.52043472291256454 | 0.52043472291256454 | 0 | PASS |
| MM Recall@1 | unavailable in stored schema | 12.5 | UNVERIFIED | No stored Recall@1 field. |
| MM Recall@10 | 12.5 | 12.5 | 0 | PASS |
| MM Recall@50 | 25 | 25 | 0 | PASS |

The quote/spec and implementation agree: all three ranks present, no missing values, so common cohort equals cohort; ratio is after means; Recall expressed percent. Raw MM query count 8, valid targets 8, missing 0. Production denominator uses scored query count (`metrics.py:59-76`), not raw count when missing; missing ranks excluded from means/Recall and reduce coverage, never counted as zero (`metrics.py:55-67,99-102`). This run cannot empirically distinguish denominator behavior with zero missing records.

## T4 — dataset integrity and leakage

| check | method | status | evidence | finding_id |
|---|---|---|---|---|
| Release path/split identity | Read loader and selection | STATICALLY_CHECKED | Paths resolve M-BEIR query/candidate JSONL, SF, validated parts, aggregated audit (`data.py:38-68`). Query qid/text/reference/positive required (`:146-163`); duplicate query/candidate IDs rejected (`:182-188,:231-237`). | none |
| Category boundary | Reference stem against FashionIQ validation splits/test annotations | STATICALLY_CHECKED | Category inferred only from reference; ambiguous references unscoped (`data.py:318-387`). | none |
| Gallery membership/order | Loader validates exact basename equality, unique identities and pinned counts; query subsets only | STATICALLY_CHECKED | `data.py:268-278,390-399`; spec says release gallery 74,381, paper-stated 18,853 not claimed (`CIRCUS_COMPOSITION_SPEC.md:7-11`). | none |
| Missing image/decode | Resolver + worker paths | BLOCKED | Resolver returns None when unresolved (`runner.py:118-125`); runner records missing query ref (`:298-301`) and blocks missing gallery (`:581-596`). Images are decoded in staged seam and model-specific worker (`runner.py:263-265`; e.g. `model_encoder.py:320`); no real local images were available in this inspection environment, so decode behavior not exercised. | none |
| Target/reference leakage | Compare release data contract, rank target positions and reference policy | UNVERIFIED | Runner maps positive IDs to gallery indices (`runner.py:188-197`) and explicitly requires reference-eligible adapters (`:607-608`); spec says references remain eligible and no masking (`CIRCUS_COMPOSITION_SPEC.md:11`). No runtime data audit proving no query target equals its reference was run. | none |
| Full gallery on subset run | Read query selection + run | STATICALLY_CHECKED | `select_queries` selects queries only, leaves gallery unchanged (`data.py:390-399`); runner resolves all `data.gallery.image_ids` (`runner.py:567-583`) and ranks positives against `len(gallery_ids)` (`:813-815`). Stored smoke identity count 74,381 (`full.json` lines 329-331). | none |

## T5 — active models/checkpoints

Registry declares nine adapters: CSMCIR, ConeSep, PTHA+MTST, HABIT, INTENT, HINT, ENCODER, LIMN, TME (`models.yaml:28-29,194-195,301-302,416-417,513-514,607-608,682-683,1279-1280,1375-1376`). Registry/spec says none have native full target-GPU inference proven by this review. No checkpoint/GPU inference executed here; status BLOCKED for inference evidence, not model quality. Spec `CIRCUS_COMPOSITION_SPEC.md:88-109` documents host smoke evidence for encoder, habit, limn, TME and remaining blockers; not independently re-executed.

| model | checkpoint/source, inputs/outputs, batching, conditions, errors, serialization | status |
|---|---|---|
| limn | Registry adapter `model_limn`; per-scope subprocess; batch configurable; three conditions; exact scoped score/rank; worker errors serialized by runner. Checkpoint loader not fully reviewed. | BLOCKED — no native inference performed |
| encoder | `model_encoder`; expected hash checked; `strict=False` followed by `_assert_checkpoint_loaded` (`model_encoder.py:297-303`); worker subprocess batches and returns feature arrays. | BLOCKED — no native inference performed |
| csmcir | `model_csmcir`; all condition modes declared but unsupported pending target caption feature dependency; `load_state_dict(..., strict=False)` return ignored (`:197-201`); feature worker. | BLOCKED; S-01 loader risk |
| ptha_mtst | `model_ptha_mtst`; loader uses strict=True (`:306`); subprocess outputs feature arrays; condition support declared. | BLOCKED — no native inference performed |
| tme | `model_tme`; `strict=False` then explicitly errors on missing/unexpected keys (`:362-367`); CUDA required (`:357-359`); batched worker. | BLOCKED — no native inference performed |
| intent | `model_intent`; adapter and all three conditions in registry; model worker contract not fully read. | BLOCKED — no native inference performed |
| habit | `model_habit`; adapter and all conditions; worker contract not fully read. | BLOCKED — no native inference performed |
| conesep | `model_conesep`; adapter and all conditions; worker contract not fully read. | BLOCKED — no native inference performed |
| hint | `model_hint`; adapter and all conditions; worker contract not fully read. | BLOCKED — no native inference performed |

Deferred eight models status: UNVERIFIED in this specific examination; spec says deferred remain deferred (`CIRCUS_COMPOSITION_SPEC.md:312`) but registry selection policy not exhaustively exercised. Registry contains other entries without `composition_adapter`; dispatch capability is registry-declared (`adapters.py:183-212`).

## T6 — experiment identity

| dimension | verdict | evidence |
|---|---|---|
| model ID | INCLUDED | `runner.py:623-625` |
| checkpoint ID | INCLUDED | `runner.py:624` |
| checkpoint content | INCLUDED, subject to provided `checkpoint_identity` correctness | identity includes `spec.checkpoint_identity` (`runner.py:622-626`); its producer not independently validated for every model. |
| model source | MISSING | Identity records adapter source hashes only; no upstream model source revision/hash (`runner.py:634-643`). Stale checkpoint/model source could reuse ranks. |
| adapter source | INCLUDED | module source hash + method source hashes (`runner.py:634-640`). |
| preprocessing | INCLUDED (declared fields) | preprocessing id and selected registry values (`runner.py:630-633`); runtime transform behavior itself not content-hashed. |
| dataset/release identity | INCLUDED | protocol, ordered gallery IDs, query fingerprints/subsets, release image bytes (`runner.py:623-633`; `:176-185,530-540`). Release artifact source-file hashes are summary provenance, not explicit identity fields. |
| query identity | INCLUDED | query qid/index/text/reference/positives/category/subsets fingerprint (`runner.py:182-185`). |
| gallery identity | INCLUDED | ordered IDs plus candidate map (`runner.py:176-179`), actual resolved image bytes (`:626-628`). |
| evaluation protocol | INCLUDED | protocol id, conditions and reference eligible (`runner.py:623-645`). |
| condition | INCLUDED | conditions list in identity and per-condition query feature key (`runner.py:629,735-737`). |
| relevant config | PARTIAL / MISSING | batch size, seed, adapter batch included (`runner.py:645-648`); arbitrary model config/runtime preprocessing values not fingerprinted, a stale reuse risk if they alter inference. |
| runtime identity | INCLUDED (reported identity) | `runtime_identity` and GPU included (`runner.py:642-643`); model interpreter identity's fields at `:386-450`. |
| image content hash: bytes | PASS | `_image_identity` hashes each file with `sha256_file` (`runner.py:530-540`; `data.py:71-77`); same-byte same identity and same-size edit with restored mtime changed digest in scratch probe. |
| hash missing file | PASS | `FileNotFoundError`, fail-closed, not stale reuse. |
| hash memoization | PASS | No memoization in `_image_identity`; each invocation calls streamed SHA-256. |
| hash overhead | UNVERIFIED | 1 MiB scratch file took 0.001842922 s. No real gallery image tree available locally here; full-gallery timing unavailable. |

## Finding summary

One confirmed-unfixed HIGH issue: S-01, CSMCIR checkpoint load silently tolerates incompatible state keys. No other confirmed defect observed. No production or test files modified.