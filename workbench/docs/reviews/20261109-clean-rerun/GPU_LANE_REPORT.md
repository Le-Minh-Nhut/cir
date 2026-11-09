# 2026-11-09 clean rerun — GPU lane report

Eight full composition runs, each with `--topk-depth 100 --keep-features`, one GPU job at a
time. Every run re-verified from its artifact, not from console output. Evidence:
`RUN_LOG.csv`, `BLAS_IDENTITY_FINDINGS.md`, `logs/<model>.log`, `logs/*.vram.txt`.

## Step 0 — BLAS gate (runtime setter/getter, OPENBLAS/OMP unset)

`blas_gate.py`, run per interpreter.

| interpreter | py | numpy | previous | after set 1 | setter | getter | verified | source |
|---|---|---|---:|---:|---|---|---|---|
| encoder310-adapted | 3.10.20 | 1.26.4 | 20 | 1 | yes | yes | true | libopenblas64_p-r0-0cf96a72.3.23.dev.so |
| limn310-adapted | 3.10.20 | 1.26.4 | 20 | 1 | yes | yes | true | same |
| ilearn39-adapted | 3.9.25 | 1.26.4 | 20 | 1 | yes | yes | true | same |
| tme-adapted | 3.10.22 | 1.26.4 | 20 | 1 | yes | yes | true | same |
| cir-005ef4e1b02999a5 (ptha_mtst) | 3.10.22 | 1.26.4 | — | — | **no** | **no** | **false** | — (MKL only) |

`ctl312h` runner import: `BLAS_THREADS_PINNED = 1`.

ptha_mtst env is MKL-only (`libblas.so.3 -> libmkl_rt.so.2`, `numpy.libs/` empty).
`mkl_set_num_threads` segfaults that interpreter (exit 139) in every form tried;
`MKL_NUM_THREADS=1` works but only before load. That model's encode path is therefore unpinnable.

## Step 1 — harness

`workbench/scripts/composition_eval.py`, flags used:
`--config <cfg> --models <id> --subsets full --topk-depth 100 --keep-features`.
No `--resume`, no `--smoke`. No invented flags.

All eight checkpoints were already staged under
`workbench/artifacts/checkpoint_selection/20261106T002617+0700/cohort/downloads/<model>/`
with sizes matching the registry. Rather than copy ~19 GB on an 87 GB-free disk, a private
config `~/cir_rerun_20261109.yaml` points `paths.checkpoints` there. Plan then reported all
eight READY, zero unavailable.

## Step 2 — runs

| model | variant | run_id | start→end | wall (contaminated) | cov | depth | pin | verdict |
|---|---|---|---|---|---|---|---|---|
| encoder | fashioniq | 5dee7425dd46c5f154808aac | 03:30→03:35 | ~5m14s | 6003/6003 | 100 | verified | PASS |
| conesep | fiq_n02 | a4f786d8af4d46ac2a85ab9c | 03:36→03:59 | ~22m33s | 6003/6003 | 100 | verified | PASS |
| habit | fiq_n02 | c6eb9d5775ae636a3b8addbc | 04:00→04:28 | ~28m16s | 6003/6003 | 100 | verified | PASS |
| intent | fiq_n02 | 4901da139ac40b874d2363c5 | 04:29→04:56 | ~27m56s | 6003/6003 | 100 | verified | PASS |
| hint | fashioniq | f461f391c6d4696011be8067 | 04:57→05:25 | ~28m19s | 6003/6003 | 100 | verified | PASS |
| limn | base_iter0_all_categories | 9e4918a6b5c243412cbc58c9 | 05:25→07:04 | ~1h38m | **6001/6003** | 100, partial | verified | PASS |
| ptha_mtst | cirr_transfer | f54a77450ebd2126590d4385 | 07:04→07:29 | ~24m25s | 6003/6003 | 100 | scoring only | PASS-with-caveat |
| tme | fashioniq_0 | cd1a696d8bca0fc4bce6925b | 07:29→07:58 | ~29m18s | 6003/6003 | 100 | verified | PASS |

All wall times measured with two other lanes active. **Not a baseline.** Throughput columns left
`not_measured (concurrent load)`.

## Step 3 — invariants (from artifacts)

`verify_run.py <run_id> --model <m> --checkpoint <c>` checks, per condition: every in-scope query
has a rank or an explicit unscored reason; candidate export present at the claimed depth with
in-range positions, rows strictly ordered by score, and `target_rank` equal to the metric path's;
and the identity records the BLAS pin and identifies as schema v2.

Final pass over all eight: `PASS`, `failed: []`.

limn's two unscored queries carry reasons in the artifact (`7:3038`, `7:3790`: "category
unresolved has no adapter embedding scope (dress, shirt, toptee)"). Its candidate manifests are
`partial: true`, 6001 records against 6003 expected. Coverage is 0.9996668332500417, never 100%.

## What the pinned rerun changed

Per-query rank diffs (not aggregate metrics) are in `BLAS_IDENTITY_FINDINGS.md`.

* **encoder** — the only model with material movement: 12727 of 18009 ranks changed, max |Δ| 1706,
  gap 0.4723 → 0.4699. Pre-pin and pinned encoder runs are different measurements.
  **CORRECTION 2026-11-09:** that movement is NOT the BLAS pin (a controlled A/B
  `d3b_encoder_aggregate_scan.txt` moves 1 rank/condition, r10/r50 delta 0.0). It is a changed adapter RANK method (adapter_module_source_hash 1a8b0dd2->bf60d855, adapter_source_hashes.rank 5f828b13->f0914b12, score hash identical) plus an interpreter change (pair-adapted py3.11.15 unverified -> encoder310 py3.10.20 verified).
  R@10 rose while R@50 fell, impossible for a thread-count change.
* **conesep**, **ptha_mtst** — weakly sensitive: 17 and 46 ranks moved by 1–2, from ~1e-7 score
  ties, no k-boundary crossing, so R@10/R@50 unchanged. Not bit-identical.
* **habit, intent, hint, tme** — bit-identical: 0/6003 ranks moved, per-query nDCG delta exactly 0.
  Reruns bought depth-100 and a recorded policy, not corrected numbers.

## Resume check

`resume_check.py <run_id> <model> <checkpoint>` recomputes each memmap fingerprint the way
`_cached_features` does, then loads them. On every checked run: all memmaps load under the
recomputed fingerprint, sidecar fingerprints match, `content_digest` present and verified.
`resume_would_reuse_cache: true` (encoder, conesep, habit, intent, limn, tme). Cache keying is
stable within one identity; previous generation caches are unreachable, as expected.

## Open items

1. `runner._record_effective_blas_pin()` fills `_effective_blas_pin`, but no code path writes it
   into the run identity. Artifacts record the requested pin plus a capability probe, never the
   per-call readback. Evidence the pin applied is the gate measurement plus the metric movement.
2. ptha_mtst's encode path cannot be pinned in its host env (MKL-only, setter segfaults). Its run
   is labelled accordingly and must not be presented as equivalent to an OpenBLAS-pinned run.
3. HEAD moved `ffb6ae8` → `20b83e5` at 03:53 by another lane mid-session. The four runs launched
   before that were re-checked: their recorded `scoring_module_source_hash` and
   `condition_transform_source_hash` equal the live values, so they describe the current tree.
4. `--keep-features` left ~2.4–2.9 GB per run (~21 GB); disk went 87 G → 77 G free. Delete
   deliberately. No previous artifact was removed.
