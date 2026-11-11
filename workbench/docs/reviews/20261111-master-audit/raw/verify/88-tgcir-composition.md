# 88 — TG-CIR composition adapter: real worker, parity, runtime

Branch `research/cir-failure-workbench`, 2026-11-12. Owner: this file, `workbench/composition/model_tgcir.py`, `workbench/tests/test_tgcir_composition_parity.py`. No `workbench/frontend/`, no `models.yaml`, no `workbench.env`, no CLVC file touched.

Every GPU command below ran through `workbench/scripts/gpu_job.sh`; the lock was retried when held and never bypassed. Each hold is one small forward pass set (5 queries + 20-image gallery, or fewer).

Interpreter split, and why:
* **model side** — `workbench/envs/tgcir39-la/bin/python` (CPython 3.9.25, torch 2.7.1+cu128, torchvision 0.22.1+cu128, `import clip` works). Runs the native reference harness and the worker subprocess.
* **control side** — `workbench/envs/ctl312h/bin/python`. Drives the composition adapter, because `runtime.check_environment_isolation` refuses to gate an interpreter whose `sys.prefix` is the process's own (`RuntimeError: cannot use active workbench environment`). Measured: running the adapter *inside* `tgcir39-la` fails the gate; running it from `ctl312h` with `WORKBENCH_TGCIR_PYTHON` pointing at `tgcir39-la` passes.

## 1. What `_run` now does

`_run(adapter, operation, image_paths, texts, categories)` — signature unchanged — launches (or
reuses) one worker subprocess, exactly following `model_encoder.py`'s resident-worker pattern: one
JSON job per line, reply prefixed `CIRCUS_WORKER:`, matrix in `.npy`, worker log beside the job dir,
`_stop_worker`/`close`/`restart`.

* Model loaded ONCE per adapter instance, all three categories (registry `category_model_policy: shared`).
* Checkpoint SHA-256 **verified in the worker before `torch.load`** (`_load_verified_model`: size, then
  digest, then `load_trusted_whole_model`). A mismatch raises `CheckpointDigestMismatch`; a missing
  declared digest refuses to unpickle at all.
* Repairs reused from `workbench/replay/tgcir.py` — `_load_tgcir` for the `_LinearWithBias` alias and the
  same `batch_first=False` loop for the 24 `nn.MultiheadAttention` modules, plus `torch.Tensor.cuda`
  neutralisation when there is no GPU. All recorded in `load_state["adaptations"]`.
* `require_verified_interpreter()` still runs immediately before the first launch (env-gate test keeps
  passing; `test_circus_env_gate_omp.py` recognises the `Popen` and requires the gate line to precede it).

### PID reuse (proof the model is not reloaded per batch)

```
composition harness, 3 categories + a 5-call batch-1 pass, one adapter instance:
  worker.pid       1238225
  worker.pid_after 1238225
  load_state.loads 1
```
The reply carries `pid` and `loads`; `Adapter._check_worker_pid` raises
`adapter_worker_reloaded` if a later reply comes from a different PID. Unit-tested
(`test_worker_pid_change_is_a_named_refusal`).

## 2. Locked sample (declared BEFORE measuring)

`cap.<cat>.val.json` first 5 rows whose reference AND target `.jpg` exist, gallery = the pair-union of
those endpoints in first-seen order then filled from `split.<cat>.val.json` to 20. Deterministic,
model-independent. Script: `/tmp/tg88_sample.py`.

| cat | queries (reference → target) |
|---|---|
| dress | B005X4PL1G→B0084Y8XIU, B008XODTD0→B00AKLK08G, B00BPYP69K→B00CMPE0C0, B000QSGNOI→B004UO3XYC, B00FQANLX2→B007U6KROG |
| shirt | B00CZ7QJUG→B005AD7WZI, B0083I6W08→B00BPD4N5E, B0083WOL16→B008VNAKSU, B005Y4KFPM→B005Y4JJ0Y, B006QHPYIY→B005NYBUF2 |
| toptee | B008CFZW76→B008CG1JJ0, B0088WRQVS→B001AS562I, B00BM27KWQ→B007TGJOMS, B00H03TS7Q→B0085YC1G4, B00B02GQ1E→B00BB1UAMU |

Galleries (20 each, first 10 are the pair-union endpoints in that order):

```
dress  B005X4PL1G B0084Y8XIU B008XODTD0 B00AKLK08G B00BPYP69K B00CMPE0C0 B000QSGNOI B004UO3XYC B00FQANLX2 B007U6KROG B009PMCJLW B007HNF4QI B004KHOPDW B0083E6W7U B007L3WP5W B0091PLEKA B004P7TNIY B0077SM1Z0 B007UYY43I B00762GDB0
shirt  B00CZ7QJUG B005AD7WZI B0083I6W08 B00BPD4N5E B0083WOL16 B008VNAKSU B005Y4KFPM B005Y4JJ0Y B006QHPYIY B005NYBUF2 B000KENMD8 B0013EO152 B00DI9YY8O B00C40R3FY B008LIH5VU B0081SF8U0 B007KFQVGK B0051UW60Q B004DJ0A0Y B000GYAKW8
toptee B008CFZW76 B008CG1JJ0 B0088WRQVS B001AS562I B00BM27KWQ B007TGJOMS B00H03TS7Q B0085YC1G4 B00B02GQ1E B00BB1UAMU B00D2EVETW B009P4JF1W B004C1XF96 B003R7235S B000F7SSDO B0015O8JLC B005CTBNN2 B00528E466 B009CC9S5A B00BSWGT18
```

Declared tolerances, fixed before the numbers: **target `atol=0, rtol=0`**; **PASS ≤ 1e-6**; anything above is a failure to explain, not to widen.

## 3. Same-input parity (native vs composition worker)

Native = `workbench/replay/tgcir.py::_load_tgcir` + its repair block + the **pinned**
`datasets.FashionIQ.concat_text`, scoring as `test.py:53-67`. Composition = `Adapter.encode_queries /
encode_targets / score`. Same images, same checkpoint, same transform, reference masked in both.

| category | query rows max &#124;Δ&#124; | target rows max &#124;Δ&#124; | score vector max &#124;Δ&#124; | masked score max &#124;Δ&#124; |
|---|---|---|---|---|
| dress | **0.0** | **0.0** | **0.0** | **0.0** |
| shirt | **0.0** | **0.0** | **0.0** | **0.0** |
| toptee | **0.0** | **0.0** | **0.0** | **0.0** |

Bit-identical on all 12 arrays (15 query rows + 60 target rows). Ranks (masked reference, 1-based,
adapter's `_ranks_from_scores` tie policy):

| category | ranks |
|---|---|
| dress | 1, 1, 11, 1, 2 |
| shirt | 2, 1, 1, 2, 3 |
| toptee | 2, 3, 1, 2, 1 |

Rank vectors identical to the native computation on all 15 queries (`rank_vectors_equal: true`, all
three categories), and `Adapter.rank()` equals `_ranks_from_scores` applied to the native score
matrix (`adapter_rank_equals_tiebreak_on_native: true`). Full 20-deep candidate order identical,
masked and unmasked, in every category.

**Why `score()` needed no change:** `test.py:53-61` is row-normalize + one dot product, which is
exactly `CompositionAdapter.score`. Measured max &#124;Δ&#124; **0.0** on real 512-d embeddings, so the shared
implementation is the upstream expression — nothing was reimplemented.

## 4. CIRCUS intentional differences (same sample, reference NOT masked)

Named differences, each attributable:

1. **reference not masked** (`test.py:63-67` not inherited). Reference self-score and rank movement:

| cat | reference position (gallery idx) | ref self-score | ranks masked → unmasked | reference is top-1 when unmasked |
|---|---|---|---|---|
| dress | 0,2,4,6,8 | 0.3139, 0.4748, 0.4483, 0.3185, 0.5709 | 1,1,11,1,2 → 1,1,11,1,2 | false,false,true,false,true |
| shirt | 0,2,4,6,8 | 0.6477, 0.3063, 0.2189, 0.5405, 0.3976 | 2,1,1,2,3 → 2,1,1,2,3 | true,false,false,true,false |
| toptee | 0,2,4,6,8 | 0.5587, 0.5135, 0.3364, 0.5228, 0.3731 | 2,3,1,2,1 → 2,3,1,2,1 | true,true,false,true,false |

No rank moved on this sample: a reference that is top-1 sits at rank 1 either way, and one that is
not top-1 is below a *different* candidate, so removing it does not change `1 + #greater`. What DOES
change is the candidate window: `reference_is_top1_when_unmasked` is true for 6 of 15 queries, so for
those the reference occupies a top-1 slot that CIRCUS hands to a real candidate. That is the
documented `reference_eligible: true` policy, measured rather than asserted.

2. **gallery = CIRCUS release gallery, not the pair-union subset.** The native pair-union contains only
   the endpoints of the val queries; a CIRCUS run ranks over the release gallery. Nothing in this
   sample changes that, so it is named as a difference and not measured here.

3. **three conditions** (dress, real worker, one loaded model, PID 1238648):

| condition | ranks | max &#124;Δ&#124; vs multimodal (per query row) | argmax positions |
|---|---|---|---|
| multimodal | 1, 1, 10, 1, 1 | — | 1, 3, 12, 7, 9 |
| text_only_black_image | 1, 1, 9, 1, 4 | 0.23617, 0.29617, 0.26807, 0.23876, 0.26486 | 1, 3, 12, 7, 6 |
| image_only_empty_text | 3, 2, 15, 15, 4 | 0.71064, 0.79967, 0.79812, 0.79282, 0.81398 | 15, 5, 12, 1, 16 |

Each movement is one named input change, and the reply proves the text side did what it claims:
`corrected_digest` is **identical** (`46a18d32…`) for multimodal and text-only (same edit text, same
corrections) and **different** (`545c38b0…`) for image-only, whose `corrected_first` is `""` — the
empty text reached the model as empty, not replaced by a default caption. Text-only's pixels are a
zero image of the same size; image-only's text is empty. Those are the only differences from
multimodal.

## 5. Batch 1 vs batch N, ties, cache

* **batch 1 vs batch N** (5 dress queries, one row per call vs one 5-row call, same worker):
  max &#124;Δ&#124; per row `2.898e-06, 1.611e-06, 2.027e-06, 1.714e-06, 1.211e-06`; overall **2.898e-06**.
  Row ORDER identical (`order_identical: true`), ranks identical (`1,1,10,1,1` both). This is cuDNN
  batch-shape nondeterminism, not a plumbing difference: the same tolerance appears between two batch
  shapes on the native path (recorded in `common.py`'s own notes). It is above the 1e-6 same-INPUT
  gate only because it is not the same input shape; the same-input parity in §3 is 0.0.
* **near-tie determinism.** With the first gallery image duplicated at the end (21 entries), the two
  copies score within `2.98e-08` (float32 noise) and one row is an exact tie. `Adapter.rank()` and a
  hand-applied `_ranks_from_scores` agree row for row (`equal: true`), so the tie policy is
  deterministic and identical to the base class. The natural sample has **0** duplicated score values
  in every category (`ties: {dress: 0, shirt: 0, toptee: 0}`).
* **cache on/off.** `WORKBENCH_REPLAY_CACHE_ROOT` set vs unset around the same call: max &#124;Δ&#124; **0.0**,
  bit-identical, and the directory was never created (`cache_root_written: false`). The composition
  path reads no replay cache at all, which is why the setting cannot move a number.

## 6. Caption correction: one owner, proven once

Contract (module docstring, enforced by code): `texts` handed to `encode_queries` are **RAW** captions
in the protocol's joined form (`cap0 + " and " + cap1`, as the CIRCUS release stores `query_txt`). The
adapter transports them unchanged (`test_encode_queries_transports_texts_verbatim`, and the shared
`test_composition_adapters_missing.py::test_adapter_does_not_rewrite_the_texts_it_is_given[tgcir]`
passes). The **WORKER** applies the pinned correction once, immediately before the forward pass.

`Corrections` reproduces `datasets.py:71-80` exactly: correct each caption of the pair SEPARATELY, join
with `" and "`. Correcting the joined string instead differs on **3 of 6016** val rows (all with an
empty first caption), so it is not an equivalent function and is not used for annotation rows.
Category resolution order: explicit caller category → (reference stem, raw text) → raw text → stem;
disagreement between categories is REFUSED. Measured on all 6003 release queries: 6000 resolve, 25
resolve to two categories whose dictionaries produce the **same** corrected text, **3 unresolved**
(`7:3938`, `7:4723`, `7:5123` — their `query_txt` is a rephrased caption, not a `cap.*.val.json` row;
they fall back to the resolved dictionary for the whole string, which is what the pinned function does
to a single string).

Exactly-once, measured on REAL embeddings (`toptee/B00C9NQNSY`, the one non-idempotent val row):

```
raw                'The silicone coverUps are pink in color. and They’re coverup cutlets & not clothes'
pinned_corrected   "the silicone coverup are pink in color and they're coverup cutlets not clothes"
twice_correction   "the silicone coverup are pink in color and they re coverup cutlets not clothes"

native(pinned_corrected) vs adapter(raw_in)      max |Δ| = 0.0        ← exactly once
native(raw)              vs adapter(raw_in)      max |Δ| = 0.04678698629140854
adapter(raw_in)          vs adapter(corrected_in) max |Δ| = 0.021270141005516052
adapter(corrected_in)    vs adapter(twice_in)     max |Δ| = 0.0        ← second application is a no-op
```

So the corrected string is what reaches the model, exactly once, and a double correction would be
visible (`0.0213`). `_correct_text` is asserted byte-identical to the pinned function, executed out of
`datasets.py`, on a fixture (punctuation → space, lowercase, per-token substitution).

## 7. Real runtime on the GPU

```
device_name            NVIDIA GeForce RTX 5070 Ti
capability             (12, 0)   -> sm_120
total VRAM             15806.1 MiB (16303 MiB per nvidia-smi)
torch                  2.7.1+cu128
torchvision            0.22.1+cu128
cuda                   12.8
clip                   1.0 (github openai/CLIP@a1d071733d7111c9c014f024669f959182114e33)
numpy / pandas / PIL   1.22.3 / 1.4.2 / 11.3.0
checkpoint             workbench/artifacts/checkpoint_selection/20261106T002617+0700/downloads/tgcir/x/checkpoints/Fashioniq.pt
checkpoint bytes       607928637
checkpoint sha256      c37768f374c75bebff1299e870518c23bf6db3dc8a75c4d1f729ccd6dd02e966
                       (registry expected_sha256, verified in-worker before torch.load)
```

One real forward: 1 image (`encode_targets`, 1 row) and a batch of 8. VRAM, measured from nvidia-smi in
the same process (torch's own counters report 0 in the driver process because the forward runs in the
WORKER; the worker's counters are in the reply):

```
nvidia-smi before           858 MiB
nvidia-smi after 1 image   1769 MiB   (+911)
nvidia-smi after batch 8   1877 MiB   (+108)
nvidia-smi after close      848 MiB   (back to baseline)
worker reply (dress, 5 queries): allocated 593.1 MiB, reserved 892.0 MiB,
                                 peak reserved 892.0 MiB, device total 15806.1 MiB
```

Peak device memory held by a TG-CIR worker: **892 MiB reserved (~5.6 % of the card)**, and it returns
to baseline on `close()`. No OOM; `resolved_batch_size` default (32) is far inside
`require_batch_fits`'s 0.85 fraction.

### `adapted_runtime` block the lead must add (measured values)

The shared gate refuses the historical contract (`torch: expected ==1.7.0, found 2.7.1+cu128`). With
this block overlaid **in memory only** (`/tmp/tg88_gate.py`), `verify_environment` reaches
`GPU_VERIFICATION_PENDING`, a READY tier, and every declared package resolves. Insert under `tgcir`'s
`environment:`

```yaml
  adapted_runtime:
    target_compute_capability: sm_120
    python: '3.9'
    pytorch: ==2.7.1
    torchvision: ==0.22.1
    cuda: '12.8'
    packages:
    - clip==1.0
    - numpy==1.22.3
    - pandas==1.4.2
    - Pillow
    pip_index_url: https://download.pytorch.org/whl/cu128
```

Two notes the lead needs:
* `clip==1.0` (not the VCS URL) — the environment has `direct_url.json` pinning
  `openai/CLIP@a1d0717…`, and `_adapted_packages` splits a requirement at the first operator, so a
  `clip @ git+…` entry is looked up as the distribution name `clip @ git+https:…` and reports
  `DEPENDENCY_MISSING` (measured). Bare `Pillow` resolves to the installed `pillow` distribution;
  `pillow==11.3.0` as written does not (measured: `Pillow==11.3.0: null`).
* `python: '3.9'` matches the provisioned interpreter exactly (3.9.25).
* `WORKBENCH_TGCIR_PYTHON` must also be set to
  `/home/heheboiz/data/cir/workbench/envs/tgcir39-la/bin/python`. **I did not set it** — `workbench.env`
  is the lead's.

Environment used, verbatim:
`workbench/envs/tgcir39-la/bin/python -c "import torch, torchvision; torch.cuda.is_available()"` →
`2.7.1+cu128 0.22.1+cu128 True`.

## 8. Tests

```
workbench/envs/ctl312h/bin/python -m pytest workbench/tests/test_tgcir_composition_parity.py workbench/tests/test_tgcir_replay.py -q
...................                                                      [100%]
19 passed in 1.05s
```
(13 in the new file, 6 in the pre-existing replay file.)

`test_composition_adapters_missing.py::test_status_refuses_rather_than_claiming_readiness[tgcir]`
still passes unchanged — `status.ready` stays False because the registry still declares no
`adapted_runtime`, which is exactly the state the lead is being asked to change. Nothing in that shared
file was edited.

What the new file covers without GPU/weights/network: pinned-`correct_text` equality on a fixture;
correction applied exactly once (pinned per-caption join, 40 real val rows, plus a non-idempotent row
proving double correction is observable); unresolvable text and unknown category refused; caller
category wins; `encode_queries` transports texts verbatim; query/target row order preserved; all-`None`
target gallery accepted and a non-`None` target category refused; digest mismatch refused before
`torch.load` (loader patched, asserted never called — plus size and missing-digest variants); the
right digest does reach the trusted loader once; reply shape mismatch raises
`adapter_worker_invalid_matrix`; PID change raises `adapter_worker_reloaded`; first reply records the
PID.

## Còn chặn

1. **`registry/models.yaml` has no `adapted_runtime` for `tgcir`** — the block is in §7. Until the lead
   adds it, `Adapter.status()` stays not-ready and a real composition run refuses at
   `require_verified_interpreter`. Not my file.
2. **`WORKBENCH_TGCIR_PYTHON` is unset in `workbench/config/workbench.env`** — needs
   `/home/heheboiz/data/cir/workbench/envs/tgcir39-la/bin/python`. Not my file.
3. **`GPU_VERIFICATION_PENDING`** — after (1), the gate needs `verify_gpu_runtime()` to record the
   sm_120 capability for this interpreter in the operator's workspace before `require_launch_ready`
   will let a full evaluation start. That is a provisioning action, not an adapter change.
4. **The composition adapter has not been run end-to-end through `workbench/composition/runner.py`**
   (CIRCUS release gallery, 15 415 queries, three conditions, all categories). This evidence is the
   adapter + worker contract, parity and runtime, at the declared 5-query/20-image scale. The
   campaign's native TG-CIR numbers (`raw/tgcir_run3cat_gpu.json`: macro R@10 49.7208 / R@50 71.1023 /
   mean 60.4115) are untouched by this work and remain the comparison targets for that run.
5. **`clipWeight` transform**: the worker builds a SECOND CLIP tower on the CPU purely to get
   `.preprocess` (the pinned model already carries its own visual weights from the pickle). It costs
   ~340 MB of host RAM and 0 MiB of VRAM; the transform it yields is the same object
   `train.py:87,90` uses. Recorded in `load_state["transform"]`.
