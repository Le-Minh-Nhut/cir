# ENCODER stable-ID regressions: did the unseeded relation cause the two rank disagreements?

Repro:

```
cd /home/heheboiz/data/cir
.venv/bin/python workbench/docs/reviews/20261112-parity-repair/scripts/stable_id_regressions.py
```

Reads only tracked evidence plus the regenerated run's intermediates
(`$ENCODER_PARITY_RAW`, default `/tmp/encparity_raw`); writes only `dress.json`, `shirt.json` here.
CPU only, no GPU.

## Conclusion

Both disagreements are gone in the regenerated pair (`dress` `7:1565` 1216 vs 1215 → **1222 vs 1222**;
`shirt` `7:3362` 1372 vs 1373 → **1372 vs 1372**), so the verdict is `RESOLVED_BY_RELATION_SEED` — but
the stored scores qualify *why*: the pre-regeneration sides already agreed to `1.1324882507324219e-06`
(dress) / `1.648440957069397e-06` (shirt) cross-side feature distance, i.e. both sides were computed
with the **same** relation tensor, while the era shift between runs moves the same features by
`0.013255782425403595` / `0.01691335439682007`. The historical 1-rank difference therefore lived
*inside* one shared-relation state, on a target whose deciding margin
(`1.043081283569336e-07` / `1.1920928955078125e-07`) was only **0.125 / 0.153 of that state's own
cross-side float32 noise** — a knife edge, not a policy question; the seeded re-run moves both queries
by 3.4e4x / 8.2e3x that margin and destroys the edge.

## The historical claim (confirmed from `git show HEAD:`)

`g1_g2.encoder.{dress,shirt}.adapter.json` → `adapter.scoring_ranking.rank_differences_vs_native_val_control`:

```
dress  {"count": 1, "head": [1569], "classification": [{"index": 1569, "adapter_rank": 1216,
        "native_rank": 1215, "adapter_matrix_gap": 0.0006574243307113647,
        "native_matrix_gap": 0.0006571710109710693, "min_gap": 0.0006571710109710693,
        "explained_by_tie_epsilon": false}], "non_tie_count": 1,
        "min_gap_abs": 0.0006571710109710693}
shirt  {"count": 1, "head": [1352], "classification": [{"index": 1352, "adapter_rank": 1372,
        "native_rank": 1373, "adapter_matrix_gap": 0.00016057491302490234,
        "native_matrix_gap": 0.00016041100025177002, "min_gap": 0.00016041100025177002,
        "explained_by_tie_epsilon": false}], "non_tie_count": 1,
        "min_gap_abs": 0.00016041100025177002}
```

Neither artifact carried `relation_state` at HEAD (the key is absent in both), which is consistent with
the producer having no relation record to write at that point.

## The regenerated pair

`rank_differences_vs_native_val_control` is now `{"count": 0, "head": [], "classification": [],
"non_tie_count": 0, "min_gap_abs": 0.0}` in **both** categories, and the gate's
`<category>.ranks.non_tie_changes` reads `0` (it takes `non_tie_count` directly,
`parity_gate.py:536-548`), versus `1` before. `relation_state` now reports
`seed 20261111`, `sha256 20db0c27c979e9a0938c71af5142458dfa2070468d0ad3fcb6f24134c2f9ce92`,
`matches_pinned_digest: true` on both adapter and native artifacts.

Recomputed independently from the regenerated features under the native protocol:

| | dress `7:1565` | shirt `7:3362` |
|---|---|---|
| reference / target | `B004DTTRA8` / `B0048OYGF4` | `B00AMD5I7U` / `B005LIXPPI` |
| pre ranks (native, adapter) | 1215, 1216 | 1373, 1372 |
| **post ranks (native, adapter)** | **1222, 1222** | **1372, 1372** |
| post agreement | yes | yes |
| target score pre → post (native) | 0.13198302686214447 → 0.1320323795080185 | 0.17480586469173431 → 0.17513398826122284 |
| native score row `max_abs`, pre vs post | 0.0035641342401504517 | 0.0009765774011611938 |
| pre deciding margin | 1.043081283569336e-07 | 1.1920928955078125e-07 |
| pre cross-side row `max_abs` | 8.344650268554688e-07 | 7.7858567237854e-07 |
| pre margin ÷ that noise | 0.125 | 0.1531 |
| movement ÷ pre margin | 34169.28571428572 | 8192.125 |

## Is the 1215-vs-1216 delta a relation-draw change?

**Qualified yes for "which relation was drawn", no for "the two sides used different relations".**
The stored arrays answer this directly:

- pre: adapter vs native cross-side feature distance `1.1324882507324219e-06` (dress) /
  `1.648440957069397e-06` (shirt), which reproduces the HEAD artifact's own
  `population.features_vs_native_val_control.max_abs` (`1.0998919606208801e-06`) and
  `gallery_vs_native.max_abs` (`1.1324882507324219e-06`) to the digit;
- era shift (pre → post, same side): `0.013255782425403595` / `0.01691335439682007`.

A changed relation moves these features ~1e4x more than the pre-era two-sided difference, so the
pre-era pair shared one relation. Reseeding changed *which* tensor both sides use, which moved the
target from rank 1215/1216 to 1222/1222 and removed the boundary; it did not align two previously
different relations, because they were not different.

**What would be needed to prove a tie-policy cause for the historical numbers**: a run whose two
sides share one relation *and* whose target score exactly equals a gallery row's score, so
`1 + #greater + #earlier-equal` and a plain argsort position disagree on the same matrix. In the
pre-era arrays no row ties the target exactly (0 equal scores on both sides, both categories), and the
closed form equals the argsort position on all four rows — so the historical difference cannot be a
tie under the stored data; it is sub-noise score movement. A tie-policy conclusion would need a
relation state in which the target ties a gallery row exactly, which no stored era exhibits.

## Protocol used (quoted)

- gallery: `workbench/third_party/ENCODER/datasets.py:186-195` (`get_test_data`, `val-split`) —
  first appearance of `source_img_id` then `target_img_id`; rebuilt from
  `data/FashionIQ/captions/cap.<cat>.val.json` and asserted identical to the archived `gallery_ids`
  (2628 / 3089 rows, same order).
- normalize: `test.py:54-58` per-row L2, float32. score: `test.py:62`. mask: `test.py:67-68`
  (`sims[i, test_targets_id.index(source_img_id)] = -10e10`). rank: `test.py:71` `np.argsort(-sims[i])`,
  1-based.
- adapter metric path: `workbench/composition/adapters.py:393-409`
  (`1 + #greater + #earlier-equal`).
- relation draw: `model_try2.py:93` (`torch.randn(N_p, dim)` inside `BindingDecoder.__init__`);
  production pin `workbench/composition/model_encoder.py:59` (`RELATION_SEED = 20261111`), applied at
  `model_encoder.py:500` (`torch.manual_seed(relation_seed)` immediately before `Encoder(...)`).
  The committed producer had no `manual_seed` before construction; the current one does
  (`encoder_parity.py:144-148`).

## Caveats a reader needs

1. The regenerated intermediates live in `/tmp/encparity_raw` (the producer now honours
   `ENCODER_PARITY_RAW` rather than overwriting the tracked record). The **pre**-era arrays under
   `raw/encoder_parity/` are unmodified (`git status` clean for that directory).
2. The regenerated artifact's own auxiliary counters do not all reproduce from the arrays I read:
   `adapter_vs_argsort_unmasked.count` (28 dress / 7 shirt) and `native_mask_only_effect.count`
   (1093 dress) differ from recomputation (1095/1131 and 1094/1132), and `adapter_style_recall.r1`
   for shirt reads 19.479882237487733 vs 19.528949950932 recomputed. Those fields mix the adapter's
   `rank()` call with dot-product scores; the two decisions this review rests on
   (`rank_differences_vs_native_val_control` = 0, and the per-query ranks) DO reproduce exactly from
   the stored arrays, and are what the verdict uses. The discrepancy is recorded, not papered over.
3. Both cases sit at rank ~1222/~1372, far outside every cutoff, so no hit count moves. That is why
   `min_gap_abs: 0.0` with `non_tie_count: 0` is safe here — and it is also why the historical
   1-rank difference never affected a reported metric.
