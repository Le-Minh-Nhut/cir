# Negative test: the new `metrics_parity` check can fail

Run: `python3 workbench/docs/reviews/20261112-parity-repair/scripts/limn_hits.py --negative`
(the script copies the six artifacts to a fresh `/tmp` dir, mutates ONE recorded side, and calls
`evaluate_model('limn', root=<tmp>)`; the real artifacts stay untouched).

## 1. One side's recorded hit count flipped (`adapter.hits.r10 = 0`, `equality.r10 = false`, flag false)

```
PARITY_MANDATORY_SUBCHECK_FAILED
detail: 4 mandatory subcheck(s) failed: dress.batch_1_vs_n.max_rel,
        dress.metrics.hit_count_equality, shirt.batch_1_vs_n.max_rel,
        toptee.batch_1_vs_n.max_rel
dress check: ('dress.metrics.hit_count_equality', ok=False, measured=False)
```

The gate cites `dress.metrics.hit_count_equality` by name with `ok=False`. Before this repair the
same mutation could not have failed anything: the check did not exist.

## 2. The flag is a string (`"true"`), i.e. a non-boolean claim

```
PARITY_EVIDENCE_MALFORMED
detail: malformed parity artifact: dress.native_metrics_match_feature_rerank is 'true':
        only a recorded boolean can certify or refuse hit-count equality
```

A truthy string cannot smuggle a pass through `_metrics_checks`.

Both outcomes are in `negative_test.json` (machine-readable, same dir).
