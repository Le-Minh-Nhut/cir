# 69 — ENCODER rerun: does a run id that covers the relation change any number?

Date 2026-11-11. Repo `/home/heheboiz/data/cir`, branch `research/cir-failure-workbench`,
HEAD `226f57a9f899a98bb093603a1b5c9a97b2ef5e3e` (verified with `git rev-parse HEAD`).
R03 (`f856338`) is an ancestor of HEAD (`git merge-base --is-ancestor f856338 HEAD` → yes).
Controller interpreter `workbench/envs/ctl312h/bin/python`; model interpreter resolved by the
registry to `workbench/envs/encoder310-adapted/bin/python` (Python 3.10.20, torch 2.7.1+cu128).

Verdict up front: **the run succeeded, the identity covers the relation, and every number is
bit-identical** (0 rank differences in 18,009 query×condition cells). One **NEW DEFECT** was exposed
by the run: the durable `construction_state/relation-seed20261111.npy` record is now **never
written**. Reported in §7, source not touched.

---

## 1. Preflight

```
$ pgrep -af pytest                     -> empty (rc=1)
$ nvidia-smi --query-compute-apps=pid,process_name --format=csv
pid, process_name                      -> no rows
$ nvidia-smi --query-gpu=uuid,memory.used,memory.total --format=csv,noheader
GPU-9bc596d9-19e9-6f43-dc2c-a80da924263c, 769 MiB, 16303 MiB      (RTX 5070 Ti, desktop only)
$ df -h .                              -> 100G free on /dev/nvme0n1p5
```

Launch was wrapped in `workbench/scripts/gpu_job.sh`, which held `flock -n /tmp/cir_gpu.lock` for the
whole child lifetime. Its own launch check printed `[gpu_job] lock held, GPU empty, launching: ...`.
No other GPU job ran during the campaign; the lock was free at launch (no wait loop iterations).

## 2. Exact command

Previous canonical invocation, recovered from artifacts rather than guessed:

* `workbench/artifacts/composition/summary/encoder_fashioniq_ce89a9a27c4cceadeaaf30b0/*.json` →
  `subset` ∈ {full, sf, validated}; `protocol_id: circus_fashioniq_release_v1`;
  `gallery_count: 74381`; `query_count` 6003 / 4069 / 586; `conditions` = multimodal,
  text_only_black_image, image_only_empty_text; `topk_export.requested: true`, `depth: 100`,
  `status: TOPK_EXPORT_COMPLETE`; `cache_verify_mode: full`; `reused_query_result_count: 0`.
* `runs/ce89a9a27c4cceadeaaf30b0/ranks/encoder_fashioniq.json` → `identity.subset_memberships =
  ["full","sf","validated"]`, `identity.adapter_batch_size: 32`, `identity.batch_size: 32`,
  `identity.blas_threads_pinned: 1`, `checkpoint_id: fashioniq`.
* `runs/ce89a9a2…/candidates/` exists → top-K was persisted, so the rerun must persist it too.
* `--config` was required: the default policy (`strict_clean`) refuses ENCODER
  (`no checkpoint variant records a training condition ... policy strict_clean`); the stored run was
  made under a config with `checkpoint_policy: include_flagged_conditions` and an explicit
  `encoder: [fashioniq]` override. The same private config is still on disk at
  `/home/heheboiz/cir_rerun_20261109.yaml` and resolves ENCODER `READY` in `--plan`.

```bash
cd /home/heheboiz/data/cir
nohup workbench/scripts/gpu_job.sh \
  workbench/envs/ctl312h/bin/python workbench/scripts/composition_eval.py \
  --config /home/heheboiz/cir_rerun_20261109.yaml \
  --models encoder --subsets full,sf,validated \
  --topk-depth 100 --keep-features \
  > /tmp/encoder_rerun_20261111.log 2>&1 &
```

`--models encoder` only. `WORKBENCH_COMPOSITION_BATCH_SIZE` was **not** set (default 32, matching the
stored `adapter_batch_size: 32`). `WORKBENCH_BLAS_THREADS` was **not** set: the stored run's
`policy_observed.record` is `requested 1 / effective 1 / previous 10 / verified true`, i.e. the
runner's own `BLAS_THREADS_PINNED=1`, not an operator override — so defaults reproduce it, and the
new summary records the same pin.

Wall time: launch `2026-11-11T11:12:36+07:00` (log created 11:12:39), run directory born
11:13:10, final summary written 11:17:33 → **~4m57s end to end** (262 s from run-dir creation).
Gallery encode (74,381 images) finished at 11:16:21; the three query passes plus ranking and the
three candidate exports finished by 11:17:29.

## 3. New run id

**`9b806c22221e944ff10e514e`**

```
workbench/artifacts/composition/summary/encoder_fashioniq_9b806c22221e944ff10e514e/{full,sf,validated}.json
workbench/artifacts/composition/runs/9b806c22221e944ff10e514e/{ranks,candidates,work/encoder/features}
fingerprint 9b806c22221e944ff10e514ec7ff1b12bab3ffb03f0813bef495f27feea0d4cf
```

`identity.relation_state` (in `runs/9b806c22221e944ff10e514e/ranks/encoder_fashioniq.json`) is now
populated, where `ce89a9a2` has `null`.

## 4. `relation_state` block, verbatim (from `full.json`; `sf.json` and `validated.json` carry the identical block)

```json
  "relation_state": {
    "expected_sha256": "20db0c27c979e9a0938c71af5142458dfa2070468d0ad3fcb6f24134c2f9ce92",
    "matches_pinned_digest": true,
    "seed": 20261111,
    "sha256": "20db0c27c979e9a0938c71af5142458dfa2070468d0ad3fcb6f24134c2f9ce92",
    "status": "recorded"
  }
```

`status: "recorded"`, `sha256 == expected_sha256 == 20db0c27…ce92`, `matches_pinned_digest: true`.
The R03 fix reached the run path; this is not a `not_recorded` run.

## 5. Status and metrics — new run

`execution_status: COMPLETED` for all three subsets. Per-condition `status` is `COMPLETED` for all
three conditions on all three subsets, with `unscored_count: 0`, `category_scope_failure_count: 0`,
`fallback_used: false`. `reused_query_result_count: 0`, `cache_verify_mode: full`.
`topk_export: {requested: true, depth: 100, status: TOPK_EXPORT_COMPLETE, missing_query_count: 0,
partial: false}`; each `.jsonl` has exactly 6003 rows.

| subset | q | gallery | condition | scored | cov | R@10 | R@50 | nDCG | MRR |
|---|---|---|---|---|---|---|---|---|---|
| full | 6003 | 74381 | multimodal | 6003 | 1.0 | 31.017824 | 49.558554 | 0.295680 | 0.169458 |
| full | 6003 | 74381 | text_only_black_image | 6003 | 1.0 | 8.512410 | 18.024321 | 0.155412 | 0.045982 |
| full | 6003 | 74381 | image_only_empty_text | 6003 | 1.0 | 2.648676 | 7.512910 | 0.107327 | 0.011066 |
| sf | 4069 | 74381 | multimodal | 4069 | 1.0 | 18.726960 | 36.323421 | 0.218785 | 0.091253 |
| sf | 4069 | 74381 | text_only_black_image | 4069 | 1.0 | 1.032195 | 5.529614 | 0.107574 | 0.006701 |
| sf | 4069 | 74381 | image_only_empty_text | 4069 | 1.0 | 0.540673 | 4.177931 | 0.097657 | 0.003860 |
| validated | 586 | 74381 | multimodal | 586 | 1.0 | 34.470990 | 55.290102 | 0.296207 | 0.161541 |
| validated | 586 | 74381 | text_only_black_image | 586 | 1.0 | 1.877133 | 7.679181 | 0.115890 | 0.008758 |
| validated | 586 | 74381 | image_only_empty_text | 586 | 1.0 | 1.023891 | 6.996587 | 0.107763 | 0.006141 |

Aggregate: `composition_gap_ndcg` 0.474394 (full, cohort 6003), 0.508313 (sf, 4069),
0.608755 (validated, 586).

## 6. Comparison against `ce89a9a27c4cceadeaaf30b0`

Deltas computed by subtraction of the two `metrics` dicts, not eyeballed. **Every delta is exactly
`0.0`** — all 36 per-condition metric values (3 subsets × 3 conditions × R@10/R@50/nDCG/MRR) and all
five gap aggregates.

| subset | condition | metric | ce89a9a2 | 9b806c22 | Δ |
|---|---|---|---|---|---|
| full | multimodal | R@10 | 31.01782442112277 | 31.01782442112277 | 0.0 |
| full | multimodal | R@50 | 49.558554056305184 | 49.558554056305184 | 0.0 |
| full | multimodal | nDCG | 0.295680489014025 | 0.295680489014025 | 0.0 |
| full | text_only | R@10 | 8.512410461435948 | 8.512410461435948 | 0.0 |
| full | text_only | R@50 | 18.02432117274696 | 18.02432117274696 | 0.0 |
| full | text_only | nDCG | 0.1554115204524066 | 0.1554115204524066 | 0.0 |
| full | image_only | R@10 | 2.6486756621689156 | 2.6486756621689156 | 0.0 |
| full | image_only | R@50 | 7.512910211560886 | 7.512910211560886 | 0.0 |
| full | image_only | nDCG | 0.10732704390474265 | 0.10732704390474265 | 0.0 |
| sf | multimodal | R@10 | 18.726959941017448 | 18.726959941017448 | 0.0 |
| sf | multimodal | R@50 | 36.32342098795773 | 36.32342098795773 | 0.0 |
| sf | multimodal | nDCG | 0.21878521362499326 | 0.21878521362499326 | 0.0 |
| sf | text_only | R@10 | 1.0321946424182846 | 1.0321946424182846 | 0.0 |
| sf | text_only | R@50 | 5.529614155812239 | 5.529614155812239 | 0.0 |
| sf | image_only | R@10 | 0.5406733841238633 | 0.5406733841238633 | 0.0 |
| sf | image_only | R@50 | 4.177930695502581 | 4.177930695502581 | 0.0 |
| validated | multimodal | R@10 | 34.47098976109215 | 34.47098976109215 | 0.0 |
| validated | multimodal | R@50 | 55.2901023890785 | 55.2901023890785 | 0.0 |
| validated | multimodal | nDCG | 0.2962070356673842 | 0.2962070356673842 | 0.0 |
| validated | text_only | R@10 | 1.8771331058020477 | 1.8771331058020477 | 0.0 |
| validated | text_only | R@50 | 7.679180887372014 | 7.679180887372014 | 0.0 |
| validated | image_only | R@10 | 1.023890784982935 | 1.023890784982935 | 0.0 |
| validated | image_only | R@50 | 6.996587030716723 | 6.996587030716723 | 0.0 |

Aggregates: `composition_gap_ndcg` 0.47439372489324116 → 0.47439372489324116 (Δ 0.0);
`gap_multimodal_ndcg`, `gap_text_only_ndcg`, `gap_image_only_ndcg`, `mrr_gap` also Δ 0.0 on all
three subsets.

**Rank differences: 0.** Per-query comparison over the full key set of each condition
(`ranks[condition]`, 6003 queries × 3 conditions = 18,009 cells):

```
image_only_empty_text   0 moved / 6003 keys
multimodal              0 moved / 6003 keys
text_only_black_image   0 moved / 6003 keys
total moved: 0
query_failures: {} in both runs, all conditions
condition_status: byte-identical between runs
```

Stronger evidence than the metric equality: the artifacts are byte-identical where it matters.

| artifact | ce89a9a2 | 9b806c22 |
|---|---|---|
| `work/encoder/features/gallery_global.npy` sha256 | `cc9914115cbaf7fe…` | `cc9914115cbaf7fe…` (same) |
| `work/encoder/features/query_*_global.npy` (3 conditions) | — | md5 identical to ce89a9a2 |
| `candidates/…/multimodal.jsonl` md5 | `9bcd49e3e4f7a56b…` | `9bcd49e3e4f7a56b…` (same) |
| `candidates/…/image_only_empty_text.jsonl` md5 | `da97cf5265a0f4ec…` | `da97cf5265a0f4ec…` (same) |
| `candidates/…/text_only_black_image.jsonl` md5 | `3440ff289d9c0b3d…` | `3440ff289d9c0b3d…` (same) |
| feature sidecar `content_digest` | `49eb0d68…a540` | `49eb0d68…a540` (same; only `fingerprint` differs) |

So the run id changed and nothing else moved. The tensor really is the same pinned draw
(`20db0c27…` in both — `ce89a9a2`'s record is on disk at
`runs/ce89a9a27c4cceadeaaf30b0/construction_state/relation-seed20261111.npy.json` with that digest),
and a run id that partitions it costs no measurement change.

**Caveat on why the id changed — it is not ONLY the relation.** Diffing
`runs/<id>/ranks/encoder_fashioniq.json::identity` between the two runs gives exactly three
differing keys:

```
relation_state                null -> {… "20db0c27…", status "recorded" …}
adapter_module_source_hash    df68824817… -> 920d30945b…
scoring_module_source_hash    ea50afef12… -> 7ec1e34d12…
```

`adapter_module_source_hash` moved because R03 itself edited `model_encoder.py`; the scoring-module
hash moved with later edits to the scoring path. Both are legitimate identity inputs. The point
stands: none of the three changes alters a number.

## 7. NEW DEFECT exposed by this run — the durable tensor record is never written

`runs/9b806c22221e944ff10e514e/` has **no `construction_state/`** directory. `find` over the whole
filesystem for `relation-seed20261111*` newer than today's launch returns nothing; the only such file
is the one `ce89a9a2` had **re-derived by a script** after its own copy was deleted.

Cause (read, not inferred from the symptom): the worker constructs the model exactly once per
process, in `_worker_main` (`workbench/composition/model_encoder.py:453`), and the persist block is
guarded by

```python
# model_encoder.py:491
if job.get("phase") != HANDSHAKE_PHASE:
    ... np.save(relation_path, relation.numpy(), ...)   # :500
```

R03 made the worker's **first** job the handshake (`prepare_identity` → `_exchange(HANDSHAKE_PHASE,
…)`, `:250`). So on every fresh run the guard is false, no tensor is saved, and every subsequent
encode job is handled by `run_one(current)` (`:531`) — an already-loaded model, so the construction
block never runs again. The comment at `:489-495` states the intent ("the first real encode job
writes the durable copy"), but no encode job ever enters that block: it is outside `run_one`.

Consequence, precisely scoped: the run's **identity and summary are correct** (the handshake digest
is a property of the same seed and construction order, and
`runner.py`'s `relation_state_drift` check compares the two and passed), but the only on-disk copy
of the drawn tensor — the artifact `62-encoder-relation.md` §3 and `MODEL_LIMN_ENCODER_VERIFICATION`
rely on for hậu kiểm — is not produced. Reproducing it now requires the same `restore_relation_record.py`
re-derivation that `ce89a9a2` needed, and that re-derivation is only honest because the digest in the
summary pins it.

Not fixed here: production source is out of scope for this job, and the fix is a design choice
(write at handshake into a scratch dir and move, or persist on the first non-handshake job from
`run_one`). Suggested minimal fix for the owner: hoist the persist block into a helper called from
`run_one`'s non-handshake branch on its first invocation, or drop the guard and let the handshake
write into `_scratch_root()/construction-state` and rely on the existing move-under-`work/`. The
second is smaller but still lands under `work/`, which the keep-features cleanup deletes — so the
first is the one that survives, matching `construction_state_root()`'s stated purpose.

**Nothing was papered over**: the summary claims `relation_state.status: recorded` because it
recorded the digest it actually hashed, and that claim is true. The missing `.npy` is a separate,
additional gap.

## 8. Tree movement

```
$ git status --porcelain
?? CLEANUP_REPORT.md.save
```

That single line pre-dates this job (it was already untracked at `git rev-parse` time). The rerun
added **zero** tracked-file changes: `workbench/artifacts/**` is gitignored
(`.gitignore:85`, verified with `git check-ignore -v`), so the new run's artifacts are on disk but
untracked. Sizes for the operator's cleanup budget:

| path | size |
|---|---|
| `runs/9b806c22221e944ff10e514e/` total | 229 MB |
| `…/work/encoder/features/` (kept by `--keep-features`) | 181 MB |
| `…/candidates/encoder_fashioniq/` (3 × ~16 MB JSONL) | 48 MB |

Nothing was committed, deleted or moved; the four prior ENCODER runs are byte-untouched.

## 9. Which run is canonical NOW

**`9b806c22221e944ff10e514e`** is the canonical ENCODER run.

Reasons, in order: it is the only ENCODER run whose identity partitions the relation tensor
(`relation_state.sha256 = 20db0c27…`, `matches_pinned_digest: true`, `status: recorded`); it is
`COMPLETED` on all three subsets with full coverage, zero unscored queries, `cache_verify_mode: full`,
`reused: 0`, and top-K depth 100 complete; and its measurements are **identical to `ce89a9a2`'s in
every digit and in every per-query rank**, so promoting it costs no number change while gaining an
identity that states what the features were computed with.

`ce89a9a27c4cceadeaaf30b0` becomes the **predecessor** of the same pinned draw: same tensor, same
numbers, but an identity that cannot state it (`relation_state: null`). The five documents that call
`ce89a9a2` canonical and quote `31.0178 / 49.5586` remain numerically correct — those are still
ENCODER's multimodal `full` R@10/R@50 on this protocol — and the correct edit is to re-point the
canonical run id, not to change the figures. `5dee7425…`, `6a528e87…`, `eda30a37…` stay superseded
(unpinned generator, and `5dee7425` was also `--subsets full` only).

## 10. Files

* Run artifacts (new, gitignored): `workbench/artifacts/composition/runs/9b806c22221e944ff10e514e/`,
  `workbench/artifacts/composition/summary/encoder_fashioniq_9b806c22221e944ff10e514e/`.
* Console log kept for inspection: `/tmp/encoder_rerun_20261111.log` (63 lines; the trailing JSON
  block is the run report).
* This evidence file. No production source or test was edited.

---

# Rerun 2 — persistence fix `34aaa10`

Date 2026-11-11, after `34aaa10` ("Persist ENCODER relation tensor at construction, not on a later
job", HEAD at launch `34aaa10c3984e310b94e36068c118d828064c9c8`). The §7 defect above is the one it
closes: the persist block is now unconditional at construction, and `_adopt_scratch_relation()`
(`workbench/composition/model_encoder.py:262`) renames the scratch directory into
`<run>/construction_state/` on the first exchange that has a `work_root` — before the keep-features
cleanup runs.

## R2.1 Preflight

```
$ pgrep -af pytest                                     -> empty (rc=1)
$ nvidia-smi --query-compute-apps=pid,process_name --format=csv
pid, process_name                                      -> no rows
$ nvidia-smi --query-gpu=uuid,memory.used,memory.total --format=csv,noheader
GPU-9bc596d9-19e9-6f43-dc2c-a80da924263c, 771 MiB, 16303 MiB
$ git status --porcelain                               -> ?? CLEANUP_REPORT.md.save   (1 line, pre-existing)
```

`workbench/scripts/gpu_job.sh` held `flock -n /tmp/cir_gpu.lock` for the child's whole lifetime; its
launch check printed `[gpu_job] lock held, GPU empty, launching: ...`. No wait-loop iterations: the
lock was free. No other GPU job, no concurrent pytest.

## R2.2 Command (identical to `9b806c22`, same scope)

```bash
cd /home/heheboiz/data/cir
nohup workbench/scripts/gpu_job.sh \
  workbench/envs/ctl312h/bin/python workbench/scripts/composition_eval.py \
  --config /home/heheboiz/cir_rerun_20261109.yaml \
  --models encoder --subsets full,sf,validated \
  --topk-depth 100 --keep-features \
  > /tmp/encoder_rerun2_20261111.log 2>&1 &
```

`--models encoder` only; `WORKBENCH_COMPOSITION_BATCH_SIZE` unset (default 32, as recorded in both
prior identities); `WORKBENCH_BLAS_THREADS` unset (runner's own `BLAS_THREADS_PINNED=1`, recorded
`requested 1 / effective 1 / previous 10 / verified true` in `policy_observed`).

Wall time: log created `11:32:22+07:00`, run directory born `11:32:59`, final summary written
`11:37:19` → **~4m57s end to end** (260 s from run-dir creation), the same as rerun 1.

## R2.3 New run id

**`acc9b20e1aea48487f65f944`**

```
summary/encoder_fashioniq_acc9b20e1aea48487f65f944/{full,sf,validated}.json
runs/acc9b20e1aea48487f65f944/{ranks,candidates,construction_state,work/encoder/features}
```

## R2.4 `relation_state` block, verbatim (`full.json`; `sf.json` and `validated.json` identical)

```json
  "relation_state": {
    "expected_sha256": "20db0c27c979e9a0938c71af5142458dfa2070468d0ad3fcb6f24134c2f9ce92",
    "matches_pinned_digest": true,
    "seed": 20261111,
    "sha256": "20db0c27c979e9a0938c71af5142458dfa2070468d0ad3fcb6f24134c2f9ce92",
    "status": "recorded"
  }
```

`status: recorded`, `sha256 == expected_sha256 == 20db0c27…ce92`, `matches_pinned_digest: true`,
`seed: 20261111`. The same block is present in `runs/acc9b20e1aea48487f65f944/ranks/encoder_fashioniq.json::identity`.

## R2.5 `construction_state/` — the tensor file EXISTS this time

```
$ ls -l workbench/artifacts/composition/runs/acc9b20e1aea48487f65f944/construction_state/
-rw-rw-r-- 1 heheboiz heheboiz 4144 Nov 11 11:32 relation-seed20261111.npy
-rw-rw-r-- 1 heheboiz heheboiz 1219 Nov 11 11:32 relation-seed20261111.npy.json
```

Both files present. `.npy` mtime `11:32:49` **predates** the run directory's birth `11:32:59` — the
rename from the handshake scratch directory preserved the mtime, which is direct evidence the file
was written at construction and then adopted, exactly as `34aaa10` describes.

`sha256` of the tensor payload (the digest definition the pin uses: raw `tobytes()` of the
`float32 (2,512)` array):

```
$ sha256sum .../relation-seed20261111.npy            -> 6ce23bbc6a367c34f9536a612c9e32e3b9cec213687c7bd73378a16001ccf244
  (that is the FILE digest, including the .npy header)
```

The file digest is not the pinned value, and must not be — the pin is over the tensor bytes, not the
numpy envelope. Digested as the run digests it:

```
numpy.load(...) -> dtype float32, shape (2, 512)
sha256(array.tobytes()) = 20db0c27c979e9a0938c71af5142458dfa2070468d0ad3fcb6f24134c2f9ce92
  == RELATION_EXPECTED_DIGEST                     -> MATCH
sidecar .npy.json "sha256" = 20db0c27c979e9a0938c71af5142458dfa2070468d0ad3fcb6f24134c2f9ce92  -> same
array_equal vs ce89a9a2's .npy                     -> True
cmp <ce89a9a2 .npy> <acc9b20e .npy>                -> byte-identical (file digest 6ce23bbc… in both)
```

So: the file on disk reproduces the pinned digest exactly, and it is byte-for-byte the same `.npy`
that `ce89a9a2` carries (which was itself re-derived by script after its own copy was deleted — this
run's copy needed no such recovery, it was written by the run).

## R2.6 Status and metrics

`execution_status: COMPLETED` on all three subsets; every condition `COMPLETED` with
`unscored_count: 0`; `cache_verify_mode: full`; `reused_query_result_count: 0`;
`topk_export`: `TOPK_EXPORT_COMPLETE`, depth 100, `missing_query_count: 0`, `partial: false`, and each
`.jsonl` has exactly 6003 rows. `failed_expectations: []`. `gallery_count: 74381`,
`query_count`: 6003 / 4069 / 586.

| subset | condition | R@10 | R@50 | nDCG | composition_gap_ndcg |
|---|---|---|---|---|---|
| full | multimodal | 31.01782442112277 | 49.558554056305184 | 0.295680489014025 | 0.47439372489324116 |
| full | text_only_black_image | 8.512410461435948 | 18.02432117274696 | 0.1554115204524066 | (same aggregate) |
| full | image_only_empty_text | 2.6486756621689156 | 7.512910211560886 | 0.10732704390474265 | (same aggregate) |
| sf | multimodal | 18.726959941017448 | 36.32342098795773 | 0.21878521362499326 | 0.5083127388173692 |
| sf | text_only_black_image | 1.0321946424182846 | 5.529614155812239 | 0.10757390247452973 | (same) |
| sf | image_only_empty_text | 0.5406733841238633 | 4.177930695502581 | 0.09765738736689415 | (same) |
| validated | multimodal | 34.47098976109215 | 55.2901023890785 | 0.2962070356673842 | 0.6087546543920695 |
| validated | text_only_black_image | 1.8771331058020477 | 7.679180887372014 | 0.11588962404118631 | (same) |
| validated | image_only_empty_text | 1.023890784982935 | 6.996587030716723 | 0.10776273212981008 | (same) |

Gap cohorts 6003 / 4069 / 586 — unchanged.

## R2.7 Delta vs `9b806c22` and vs `ce89a9a2`

Machine-subtracted, not eyeballed. **Non-zero deltas: 0.**

* 36 per-condition metric values (3 subsets × 3 conditions × R@10/R@50/nDCG/MRR) × 2 baselines
  → all exactly `0.0`.
* 5 gap aggregates per subset (`composition_gap_ndcg`, `gap_multimodal_ndcg`,
  `gap_text_only_ndcg`, `gap_image_only_ndcg`, `mrr_gap`) × 3 subsets × 2 baselines → all `0.0`.
* `gap_cohort_count` identical (6003 / 4069 / 586).

| metric | `ce89a9a2` | `9b806c22` | `acc9b20e` | Δ vs both |
|---|---|---|---|---|
| full multimodal R@10 | 31.01782442112277 | 31.01782442112277 | 31.01782442112277 | 0.0 |
| full multimodal R@50 | 49.558554056305184 | 49.558554056305184 | 49.558554056305184 | 0.0 |
| full multimodal nDCG | 0.295680489014025 | 0.295680489014025 | 0.295680489014025 | 0.0 |
| sf multimodal R@10 | 18.726959941017448 | 18.726959941017448 | 18.726959941017448 | 0.0 |
| sf multimodal R@50 | 36.32342098795773 | 36.32342098795773 | 36.32342098795773 | 0.0 |
| validated multimodal R@10 | 34.47098976109215 | 34.47098976109215 | 34.47098976109215 | 0.0 |
| validated multimodal R@50 | 55.2901023890785 | 55.2901023890785 | 55.2901023890785 | 0.0 |
| full `composition_gap_ndcg` | 0.47439372489324116 | 0.47439372489324116 | 0.47439372489324116 | 0.0 |

Byte-level confirmation:

| artifact | `9b806c22` | `acc9b20e` |
|---|---|---|
| `candidates/…/multimodal.jsonl` md5 | `ecb307cab615…` | `ecb307cab615…` |
| `candidates/…/text_only_black_image.jsonl` md5 | `3440ff289d9c…` | `3440ff289d9c…` |
| `candidates/…/image_only_empty_text.jsonl` md5 | `da97cf5265a0…` | `da97cf5265a0…` |
| `work/encoder/features/gallery_global.npy` md5 | `857eb56a6dbb…` | `857eb56a6dbb…` |

Identity diff vs `9b806c22` is exactly one key: `adapter_module_source_hash`
`920d30945b…` → `57a37a34…` (the `34aaa10` edit to `model_encoder.py`). `scoring_module_source_hash`
and `relation_state` are unchanged.

## R2.8 Rank differences

**0 of 18,009** query × condition cells moved, against **both** `9b806c22` and `ce89a9a2`:

```
vs 9b806c22: image_only_empty_text 0/6003, multimodal 0/6003, text_only_black_image 0/6003  -> 0
vs ce89a9a2: image_only_empty_text 0/6003, multimodal 0/6003, text_only_black_image 0/6003  -> 0
query_failures: {} for all conditions; condition_status byte-identical to both
```

No finding to report: the expected identity holds. Same seed, same draw, same features, same ranks,
same numbers — only the run id moved, now for the two reasons that justify a new id (relation
recorded at mint time, plus the `34aaa10` adapter source hash).

## R2.9 Tree movement

```
before launch:  ?? CLEANUP_REPORT.md.save
after run:      ?? CLEANUP_REPORT.md.save
```

Identical. The rerun added **zero** tracked-file changes; `workbench/artifacts/**` is gitignored
(`.gitignore:85`), so the new artifacts exist on disk untracked. Nothing under
`workbench/artifacts/` was committed, deleted or moved — the four earlier ENCODER runs and the
`ce89a9a2` tensor file are byte-untouched. New disk cost: 229 MB
(`runs/acc9b20e1aea48487f65f944/`: 181 MB `work/encoder/features/`, 48 MB `candidates/`, 4 KB
`construction_state/`).

## R2.10 Which run is canonical NOW

**`acc9b20e1aea48487f65f944` is the canonical ENCODER run.**

It is the **FIRST** ENCODER run that has **both** properties at once: an identity that partitions the
relation tensor (`relation_state.status: recorded`, `sha256 20db0c27…ce92`,
`matches_pinned_digest: true`, present in the summary AND in `ranks/<model>_<checkpoint>.json::identity`)
**and** the persisted tensor file itself (`construction_state/relation-seed20261111.npy` + `.json`,
payload digest equal to the pin, byte-identical to the draw `ce89a9a2` scored with). Every earlier run
had at most one of the two.

Superseded, with the reason for each:

| run id | status now | reason |
|---|---|---|
| `ce89a9a27c4cceadeaaf30b0` | superseded | identity hashed while `relation_state` was `None` — the run id does not partition the relation (`not_recorded`) |
| `9b806c22…` (`9b806c22221e944ff10e514e`) | superseded | identity recorded the digest, but the tensor file was absent — no durable copy of the draw it claims |
| `5dee7425dd46c5f154808aac` | superseded | relation drawn from an unpinned generator; also `--subsets full` only |
| `6a528e877c328772e2c7bb28` | superseded | relation drawn from an unpinned generator; no `cache_verify_mode`/`relation_state` |
| `eda30a37c0bc17ac43844f09` | superseded | relation drawn from an unpinned generator; no `cache_verify_mode`/`relation_state` |

The published figures `31.0178 / 49.5586` remain correct for ENCODER multimodal `full` — they are
reproduced to the last digit by `acc9b20e`, with 0 rank differences. The documents that quote them
should re-point the canonical run id to `acc9b20e1aea48487f65f944`; no number changes.

## R2.11 Files

* Run artifacts (new, gitignored): `workbench/artifacts/composition/runs/acc9b20e1aea48487f65f944/`,
  `workbench/artifacts/composition/summary/encoder_fashioniq_acc9b20e1aea48487f65f944/`.
* Console log kept for inspection: `/tmp/encoder_rerun2_20261111.log`.
* No production source or test was edited by this job.
