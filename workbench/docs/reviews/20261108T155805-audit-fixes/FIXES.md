# Audit-fix pass — four confirmed audit defects (F1, F3, F4, F6) plus one functional blocker (F7)

Branch `research/cir-failure-workbench`. Nothing committed, nothing pushed, no git state touched.
Applied one defect at a time: apply → test → verify → next.

Source of truth: `workbench/docs/reviews/20261108T082857-audit-pipeline/FINDINGS.md` and its
`patches/`. GPU untouched; the live cohort evaluator (`composition_eval.py --models intent`,
pid 264541) owned the GPU for this entire pass. Every test here is CPU-only.

Files changed: `workbench/composition/runner.py`, `workbench/scripts/composition_eval.py`,
`workbench/composition/model_encoder.py`, `workbench/replay/limn_observer.py`, plus three new test
files. F4 and F6 were authored as patches while those files were owned by other agents, then applied
once ownership was released; the patches remain here as the change record.

| ID | Applied here? | Where |
|---|---|---|
| F1 | yes | `workbench/composition/runner.py`, `workbench/scripts/composition_eval.py` |
| F3 | yes | `workbench/composition/runner.py` |
| F4 | yes (via `ENCODER_MAPPING_STATUS.patch`) | `workbench/composition/model_encoder.py` |
| F6 | yes (via `LIMN_OBSERVER_COMMENT.patch`) | `workbench/replay/limn_observer.py` |
| F7 | yes — found during F4, escalated as a blocker | `workbench/composition/model_encoder.py`, `workbench/composition/model_ptha_mtst.py` |

New tests: `test_audit_fix_f1_seed_provenance.py` (4),
`test_audit_fix_f3_gallery_cache_shape.py` (3),
`test_audit_fix_f4_encoder_mapping_status.py` (5),
`test_audit_fix_encoder_helpers_imported.py` (23, 20 selected by the subset filter).

---

## F1 — `--seed` was inert: chose option (b), remove it from `identity`

### The decision, and why (b) is the honest one

The task offered (a) seed the generators so the field is meaningful, or (b) drop it from `identity`
and rename the flag. (a) is only correct if something actually draws randomness at encode time.
**Nothing does.** Established by reading, before writing any code:

1. **Every encode-time transform is deterministic.** `<model>_transform` in the BLIP-2 family is
   `TargetPad → Resize(BICUBIC) → CenterCrop → Normalize` (`third_party/HABIT/data_utils.py:92`,
   same body in `AirKnow`, `ConeSep`, `HINT`, `INTENT`, `CLIP4Cir`, `SPRC`). ENCODER uses
   open_clip `preprocess_val` (`model_encoder.py:321`); LIMN uses
   `open_clip.image_transform(..., is_train=False)` (`model_limn.py:527-528`).
2. **Every adapter encodes under `.eval()` + `torch.no_grad()`** — `conesep:296,307`,
   `csmcir:265,272`, `encoder:320,349`, `habit:334,354`, `hint:328,340`, `intent:324,336`,
   `limn:524,562`, `ptha_mtst:316,330`, `tme:367,380`. So `nn.Dropout` is inert, and so is
   open_clip's `PatchDropout`, whose forward begins
   `if not self.training or self.prob == 0.: return x`
   (`envs/limn310-adapted/.../open_clip/transformer.py:58-59`).
3. **No sampling decoder.** `grep '\.generate(\|do_sample\|num_beams\|multinomial\|bernoulli'` over
   `workbench/composition/` is empty.
4. **No RNG inside the forward passes the adapters call.** Checked the `extract_retrieval_compose`
   bodies of `AirKnow/blip2_cir.py`, `ConeSep/ConeSep.py`, `HABIT/HABIT.py`, `HINT/HINT.py` for
   `torch.rand*`/`randn`/`randint`/`multinomial`/`bernoulli`/`F.dropout`: none.
5. **A `strict=False` load cannot leave a random parameter reachable.** ENCODER refuses on
   `missing_keys` (`replay/encoder.py:230-236`), CSMCIR refuses (`model_csmcir.py:222-240`), TME
   refuses (`model_tme.py:362-366`), PTHA uses `strict=True` (`:315`); HABIT/HINT/INTENT/CONEPSEP
   load a whole pickled module, which has no missing keys at all.

The one thing that *does* draw randomness is `metrics.paired_bootstrap_gap`
(`metrics.py:209-226`), which takes its own explicit `seed` **argument** and is never called by the
runner (`grep paired_bootstrap_gap workbench/composition/runner.py` → nothing). That is a function
parameter, not a run variable; it belongs in neither `identity` nor the CLI.

So (a) would have been a lie: seeding globals would make `--seed` change a value nothing reads, which
is the same defect with extra code. **Chose (b).**

### Independent corroboration

A separate read-only audit of all nine adapters plus their third-party forwards returned the same
verdict, `NO_RNG_DRAWS`: every encoder runs `.eval()` before any forward and inside `torch.no_grad()`,
every transform is the deterministic one, and each randomness-consuming op in the third-party code is
`self.training`-gated or lives on a training/loss path the worker never calls. Specifically:

- `HINT.py:211-216` `np.random.rand`/`permutation` sit under `if self.training` in `txt_enc`,
  reachable only from `forward_emb:319`, never from `extract_retrieval_*`.
- `ConeSep.py:307-308,336` `torch.randn` + `random.choice` live in `forward()` (the loss path), not in
  the `extract_retrieval_*` methods the worker calls.
- `LIMN/LIMN/Transformer.py:48` `torch.bernoulli` is commented out.
- `INTENT/blip2_cir.py:18` `import random` is unused on this path.

Two construction-time draws are worth recording because they *do* burn global RNG state and a future
reader could mistake them for a reason to seed:

1. **`CSMCIR` `blip2_qformer_cir_align_prompt_csmcir.py:99`** draws `torch.randn` into a **plain tensor
   attribute**, not a buffer or `Parameter`. No checkpoint can overwrite it and no missing-key check
   can catch it — so unlike every other random initialisation here, a `strict=False` load would not
   protect it. It is nonetheless unreachable: it is read only by `forward()` (`:353` onward), and the
   adapter calls `inference_tsen` (`:454-500`), which never touches it. The adapter is also not
   runnable at all — `mode_supported` returns `False` for every condition
   (`model_csmcir.py:88-91`) and `encode_targets` raises (`:100-106`).
2. **Other init draws are all `Parameters`** — `TME/.../blip2.py:60` `query_tokens.normal_`,
   `blip2_qformer_cir_image_diff_features.py:85` `prompt_tokens`, `MTST_PTHA/.../*:115`,
   `ENCODER/model_try2.py:93` `learnable_relation = torch.randn`. Being in `state_dict()` means the
   refuse-on-`missing_keys` load covers them. The inner `create_eva_vit_g(..., strict=False)` with a
   discarded report (`TME/src/lavis/models/eva_vit.py:449`) is inert for the same reason: the outer
   registry checkpoint supplies every key, so even though that report is thrown away, the outer check
   has already failed the run if anything were missing. This is the one place where "`strict=False`
   could leave random init" is *structurally* possible and is closed only by the outer gate — noted
   here rather than treated as safe by construction.

`uuid.uuid4()` is used for job/attempt/tmp **names** (`runner.py:319,326,954`, `topk.py:117,167`, and
per-adapter job dirs) and never feeds a feature. It is why two identical runs produce differently
named temp directories while their features stay byte-identical.

### What was applied

`workbench/composition/runner.py` — removed `"seed": seed` from `identity`, replaced with a comment
stating why, so the next reader does not re-add it:

```python
# `seed` is deliberately NOT hashed here. It is recorded in the summary for
# provenance, but no seeding call exists anywhere in the composition path: ...
"batch_size": batch_size,
```

The summary keeps `"seed": seed` for provenance and now also states `"seed_applied": False`, so an
artifact cannot be read as if a seed had been applied.

`workbench/scripts/composition_eval.py` — help text corrected from
`"seed recorded for deterministic summaries"` (which promised determinism the flag never delivered)
to:

```
recorded in the summary only: no step of the composition path draws randomness, so this value is
provenance, not a reproducibility control, and it does not change the run id
```

I kept the flag name `--seed` rather than renaming to `--seed-recorded`: the flag name is referenced
in `CIRCUS_COMPOSITION_SPEC.md`, `A38_FEATURE_MATRIX.csv` and the baseline help transcript, and the
corrected help text now says exactly what the flag does. A rename is a docs-wide change, not part of
this defect.

### Test

`workbench/tests/test_audit_fix_f1_seed_provenance.py`, 4 checks:

| Test | Pins |
|---|---|
| `test_changing_seed_does_not_change_the_run_id` | `--seed 0` vs `--seed 7` → same `run_id`; both still **record** their value; `seed_applied is False` |
| `test_changing_seed_does_not_change_the_results` | identical `metrics` and identical per-query ranks |
| `test_seed_is_absent_from_the_run_identity` | `"seed"` is not inside the `identity = {...}` block |
| `test_cli_documents_seed_as_recorded_only` | help text says *record* and no longer says *deterministic summaries* |

### Before / after

Without the fix (production hunks reverted, test kept):

```
3 failed, 1 passed in 0.27s
FAILED ...::test_changing_seed_does_not_change_the_run_id - seed ... must not mint a second run id
FAILED ...::test_seed_is_absent_from_the_run_identity - seed must not be hashed into identity
FAILED ...::test_cli_documents_seed_as_recorded_only - old help text implied determinism
```

With the fix: `4 passed in 0.25s`.

Note on the one assertion I had to relax: two runs in the same artifact tree share a `run_id`, so the
second attempt directory is correctly named `<run_id>-retry-1` by the existing `_attempt_id` logic.
The test asserts `second_row["attempt"].startswith(first_row["run_id"])` rather than exact attempt
equality — the shared run id *is* the point, and the retry suffix is pre-existing behaviour.

### run_id consequence — **YES, this changes `identity`**

Removing `seed` from `identity` changes the digest for **every** run, including the default
`--seed 0`. Every existing `run_id` becomes a pre-fix artifact. They must be labelled, not silently
replaced:

- all artifacts under `composition/runs/<run_id>/`, `composition/results/**/<attempt>/` and
  `composition/summary/**` produced before this change carry a run id computed with `seed` in the
  identity;
- the published numbers (`recall_at_10 = 31.02%`, `composition_gap_ndcg 0.3887`, the ENCODER smoke
  attempts `17ef444e30b829ac22f2d3c8` / `1410522b57d4fe79083c60a6`) keep their existing ids and
  remain valid results, but they no longer recompute to those ids;
- the feature caches key off `_digest(identity)` too, so a resumed pre-fix run will re-encode once.

No artifact was deleted or rewritten. This is a labelling obligation for whoever publishes.

---

## F3 — gallery cache accepted without a shape check

### What was actually true (the audit's framing was partly wrong)

`_encode_gallery` called `_load_memmap(destination, fingerprint)` with no shape while
`_encode_queries` passed `(len(queries), dimension)`; `_load_memmap` only enforces a shape when given
one. But I probed the two failure modes separately before fixing, and **a byte-truncated `.npy` is
already refused** — NumPy raises `ValueError: mmap length is greater than file size`, which
`_load_memmap` catches and turns into a cache miss. Measured against the real helper:

```
valid, no shape arg                   -> ACCEPTED shape=(200, 4)
truncated body, header intact         -> REFUSED(None)     <- NumPy, not us
complete 100 rows (gallery has 200)   -> ACCEPTED shape=(100, 4)   <- the real hole
complete 100 rows, shape (200,4)      -> REFUSED(None)
```

So the surviving hole is a cache that is **complete and readable but holds the wrong number of
rows** — a partially written gallery whose sidecar was also rewritten, or a cache copied between
runs. Every row after the shortfall then describes a different image, and the only symptom is a
worse score. That is what the fix targets.

### What was applied

`workbench/composition/runner.py`:

```python
def _memmap_metadata_shape(path: Path) -> tuple[int, ...] | None:
    """The shape the sidecar records for this cache, or None when it is missing or unusable."""
```

and in `_encode_gallery`:

```python
metadata_shape = _memmap_metadata_shape(destination)
expected_shape = (len(image_paths), *metadata_shape[1:]) if metadata_shape else None
cached = _load_memmap(destination, fingerprint, expected_shape)
```

Row count comes from the sidecar's *recorded* shape combined with this run's gallery size, so the
comparison is "what the cache claims to be **vs** what this run needs". Feature width cannot be known
before encoding (it is discovered from the first batch), so it is taken from the sidecar and only the
row count is checked against the run — a cache of the right width but wrong length is refused, which
is the case that produces wrong ranks.

### Test

`workbench/tests/test_audit_fix_f3_gallery_cache_shape.py`, 3 checks:

| Test | Pins |
|---|---|
| `test_short_gallery_cache_is_not_reused` | 2-row cache vs 3-image gallery → re-encodes, shape `(3,4)`, values rebuilt from images |
| `test_matching_gallery_cache_is_still_reused` | right shape → `encoded == 0`, `reused == 1` (no warm-cache regression) |
| `test_byte_truncated_gallery_cache_is_refused_not_raised` | truncation surfaces as a cache miss, not an exception |

### Before / after

Without the fix:

```
2 failed, 2 passed
FAILED ...::test_short_gallery_cache_is_not_reused - a mis-sized gallery cache must not be reused
```
(with the width test still present at that point). After removing the over-specified width test and
applying the fix:

```
1 failed, 2 passed   # fix reverted, short-cache test fails
3 passed             # fix in place
```

### run_id consequence — **NO**

The fingerprint and the identity are unchanged; this changes only whether a cache file is *trusted*.
Existing `run_id`s stay valid. A pre-fix artifact whose gallery cache was mis-sized could have been
computed from a wrong matrix — that is a data-integrity caveat on that artifact, not an identity
change.

---

## F4 — ENCODER hard-codes `checkpoint_mapping_status` (**APPLIED**)

Ownership of `workbench/composition/model_*.py` was released, so the patch was applied with
`git apply`. The generated patch remains in this directory as the change record.
`workbench/tests/test_audit_fix_f4_encoder_mapping_status.py` (5 checks) proves it.

The change is `ENCODER_MAPPING_STATUS.patch` in this directory, now applied.

### Why the literal mattered, and why the patch needs two hunks

The literal is in the **worker**, not the parent (`model_encoder.py:303`), because the worker builds
its own checkpoint record inside `_worker_main`:

```python
checkpoint = {"checkpoint_id": "fashioniq", "expected_sha256": job["expected_sha256"],
              "checkpoint_mapping_status": "VERIFIED_METADATA"}
```

That record is passed straight to `_assert_checkpoint_loaded` → `_unused_checkpoint_keys`
(`replay/encoder.py:73-102`), which uses `checkpoint_mapping_status` as one of its four gates. A
one-line change at `:303` alone would read `job[...]`, so the parent must also put the registry value
into the job payload. Hence two hunks:

1. `_run_job` payload gains `"checkpoint_mapping_status": self._checkpoint.get("checkpoint_mapping_status")`.
2. The worker constructor reads `job.get("checkpoint_mapping_status")` instead of the literal.

`.get` rather than `[...]`: a job missing the key yields `None`, and `None != "VERIFIED_METADATA"`
therefore **refuses** the allowance rather than defaulting to it. That is fail-closed on purpose.

### Verified, not assumed

- **The registry does carry the value.** Read live from the registry: `encoder` /
  `per_query_checkpoint → fashioniq` → `checkpoint_mapping_status = 'VERIFIED_METADATA'`,
  `expected_sha256 = 0f31c501…9173`. So the field the patch reads exists; this is not a substitution
  of one hard-code for another.
- **The gate really is keyed on that field.** `_unused_checkpoint_keys` contains
  `if variant.get("checkpoint_mapping_status") != "VERIFIED_METADATA": return frozenset()`.
- **The patched file parses** (`ast.parse` ok) and the literal is gone from it.
- **The behaviour changes as intended.** Driving the real gate function with a `FakeModel` that does
  not declare the two keys, on the real pinned source root (`source_pin_status → True`):

```
VERIFIED_METADATA        permitted=['backbone.binding_decoder.fc.0.bias', '...weight']
                         assert -> ACCEPTED (the two tensors were skipped)
downgraded UNVERIFIED    permitted=[]
                         assert -> REFUSED: ENCODER checkpoint does not match the model: ...
absent (None)            permitted=[]
                         assert -> REFUSED: ENCODER checkpoint does not match the model: ...
```

So with the patch a registry downgrade tightens the loader by construction; before it, the literal
made that downgrade unobservable. The other three gates (checkpoint digest, clean source pin, module
not declaring the names) are untouched — this patch removes the hard-coded claim, it does not weaken
any gate.

### run_id consequence — **YES, for ENCODER** (corrected after applying the patch)

`checkpoint_mapping_status` itself is not in `identity`, but **editing `model_encoder.py` is**: the
adapter's own module file is hashed into `identity.adapter_module_source_hash`
(`runner.py:802-808`, `_module_source_hash`). Measured:

```
sha256 now  : 3f4efc789448106478f8c73386b0669be4d85167bcde033e2c244da556d70687
sha256 HEAD : 1a8b0dd25827ca0ba90f504d81f2fa190cede1ee3fb3bfe335e70ab359005823
```

So ENCODER's `run_id` moves on this edit independently of F1. The **numbers do not change**: the
registry status is `VERIFIED_METADATA`, the same weights load, and the same two unused tensors are
skipped — `test_verified_status_still_accepts_the_two_unused_tensors` pins that. The new id is a
namespace consequence of the source hash, not a numerical one.

`limn_observer.py` is *not* in `identity` (`grep limn_observer runner.py adapters.py` → empty), so
F6 does not move any run id.

---

## F6 — LIMN tie-order comment states a false reason (**APPLIED**)

Ownership of `workbench/replay/**` was released, so `LIMN_OBSERVER_COMMENT.patch` was applied with
`git apply`; the patch remains here as the change record. Comment-only: the AST of
`limn_observer.py` is byte-identical before and after (`ast.dump(ast.parse(...))` comparison), and
the 6 LIMN test modules pass (67 tests together with the F1/F3/F4 files).

`limn_observer.py:168-169` said:

> `argsort` is stable in NumPy, so reproducing the native call reproduces its tie order too.

That is false. Verified against the interpreter actually used here (numpy 1.26.4):
`inspect.signature(np.argsort)` is `(a, axis=-1, kind=None, order=None)` and its own docstring says
*"The default is 'quicksort'."* — introsort, not stable. The behaviour is nonetheless correct,
because `limn_observer.py:170` issues the **same call on the same input** as the native
`test.py` (`np.argsort(-sims[i, :])`, `third_party/LIMN/LIMN/test.py:85,155,244`), and that call is
deterministic for a given dtype and array size. Only the stated reason was wrong.

The replacement comment says exactly that, and additionally warns the future reader that making the
kind explicit — or doing so on only one side — would change the tie order and break the reproduction
the surrounding comment exists to justify. That was the real risk the false sentence was masking.

### run_id consequence — **NO**

A comment. No identity field, no code path.

---

## Verification runs

Mandatory subset:
`workbench/envs/ctl312h/bin/python -m pytest workbench/tests -q -p no:cacheprovider -k "composition or capability or topk or seed or cache or circus"`

| Run | Result |
|---|---|
| Baseline, before any edit | **7 failed, 321 passed, 7 skipped** (21.48s, `-n 4`) |
| After F1 + F3 | **7 failed, 328 passed, 7 skipped** (15.97s, `-n 4`) |
| After F1 + F3 + F4 + F6 | **7 failed, 328 passed, 7 skipped** (14.35s, `-n 4`) |
| After F7 (helper imports) | **7 failed, 348 passed, 7 skipped** (14.61s, `-n 4`) |

The failing id set is byte-identical across **all four** runs (`diff` empty, 7/7). The deltas are
fully accounted for:

* **+7** (321 → 328): the F1 and F3 test files.
* **+0** on the F4/F6 run: `test_audit_fix_f4_encoder_mapping_status.py` is **not selected** by that
  `-k` filter — verified with `--collect-only`: 0 collected. Also expected: F6 is comment-only and
  adds no test.
* **+20** (328 → 348): `test_audit_fix_encoder_helpers_imported.py` — `--collect-only` reports
  `20/23 tests collected (3 deselected)` under that filter.

The F4 file is therefore run explicitly, together with the F1/F3/F7 files and the six LIMN modules
that exercise the file F6 touches:

```
pytest test_audit_fix_encoder_helpers_imported.py test_audit_fix_f1_seed_provenance.py \
       test_audit_fix_f3_gallery_cache_shape.py test_audit_fix_f4_encoder_mapping_status.py
  ->  35 passed in 1.26s
```

`test_circus_metrics.py`: 8 passed separately.

**No existing test changed state**, so nothing was weakened and nothing regressed.

The 7 baseline failures are pre-existing and not mine — they come from in-flight `model_*.py` work by
another agent plus unset interpreter env vars in this shell:

```
AttributeError: 'Adapter' object has no attribute 'batch_size'
  test_circus_adapters_blip.py::... [conesep], [ptha_mtst]   (x4)
RuntimeError: execution environment not verified [UNCONFIGURED/...]
  WORKBENCH_ENCODER_PYTHON=None, WORKBENCH_LIMN_PYTHON=None   (x2)
FileNotFoundError: .../output.npy                            (x1)
```

No existing test was weakened. I added one new test file per applied fix and changed no test outside
them.

### F7 — missing helper imports: a functional blocker, **FIXED**

Found while writing the F4 test, then escalated to priority as a real blocker. Two adapters called the
parallel-decode helpers without importing them, so both raised `NameError` inside `_worker_main` before
producing any feature; each worker's own `except Exception` converted that into
`reply(False, error_type="NameError", ...)`, so the run failed at model load and looked like a model
problem. **ENCODER and PTHA+MTST — two of the models with published numbers — could not produce a fresh
run from this tree at all.**

| adapter | call sites | missing imports |
|---|---|---|
| `model_encoder.py` | `:330` `limit_torch_threads()`, `:331` `resolved_decode_workers()`, `:353` `prefetch_rows(...)` | all three |
| `model_ptha_mtst.py` | `:320`, `:321`, `:345` | all three |

Both fixed by adding the three names to the existing multi-line
`from workbench.composition.adapters import (...)` blocks. Every call site was re-verified in the file
before editing.

**Static audit of all nine adapters** (helper set derived from `adapters.py`'s public callables):
exactly those two were broken; all nine are now clean. `habit`, `hint`, `intent`, `conesep`, `tme`
(import at `:15`/`:16`/`:20`) and `limn` (multi-line, `:45-47`) were already correct;
`csmcir` has no decode pool. `limn` and `tme` also *call* all three, so they are not affected by this
class of defect.

**Runtime proof**, not reading: in `workbench/envs/encoder310-adapted/bin/python`,
`limit_torch_threads() -> 10`, `resolved_decode_workers() -> 10`, `prefetch_rows` callable; and both
fixed modules resolve all three names in `_worker_main.__globals__` to the **shared** implementations
(`is adapters.X`).

**Regression test**: `test_audit_fix_encoder_helpers_imported.py`, 23 checks, parameterised over all
nine adapters, using `symtable` global-reference analysis (an AST walk produced false positives on
`reply`/`transform`, which are bound in enclosing scopes). Two layers — the derived helper set, and
*every* unbound global against a one-entry allowlist — plus a self-check that strips a real import and
asserts the detector fires. Proven by reverting each import:

```
encoder imports reverted    -> 3 FAILED
ptha_mtst imports reverted  -> 4 FAILED
both restored               -> 23 passed
```

**Why no existing test caught it**: `test_circus_adapters_*.py` import the module and construct
`Adapter()`, never entering `_worker_main`; a `NameError` in a function body is invisible until that
line executes. The check had to be static.

**run_id**: both modules' hashes changed, so each re-keys. For ENCODER this is the **same** new id as
F4 (both edits are in one file; the hash is computed from file content at run time, so nothing was
minted from an intermediate state). PTHA+MTST gets a separate, independent id.

**Consequence for published numbers** — see `MODEL_REPRODUCIBILITY_VERDICT.md`: ENCODER
(31.02/49.58) and PTHA+MTST (19.47/34.45) are labelled `pre-generated artifact, not runnable from
HEAD` until the fix is in **and** the model is re-run. Their artifacts stay valid — the code that made
them was correct — but the tree could not regenerate them.

---

## Files in this directory

```
ENCODER_MAPPING_STATUS.patch              F4 change record (now applied)
LIMN_OBSERVER_COMMENT.patch               F6 change record (now applied)
FIXES.md                                  this report
PRE_F1_ARTIFACTS_AND_CACHE_VERDICT.md     pre-F1 run dir inventory + cache-reuse verdict
MODEL_REPRODUCIBILITY_VERDICT.md          ENCODER/PTHA runnability + per-model matrix
baseline_subset.txt                       subset run before any edit
final_subset.txt                          subset run after F1 + F3
final_subset_after_patches.txt            subset run after F1 + F3 + F4 + F6
final_subset_after_f7.txt                 subset run after F7 (helper imports)
helpers_test.txt                          the F7 static test on its own (23 passed)
```

The revert-proof outputs for F7 (3 failed / 4 failed / 23 passed) are quoted inline above and are
reproducible by removing the two import lines and re-running the test.
