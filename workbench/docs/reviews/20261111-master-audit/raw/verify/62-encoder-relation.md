# R03 — ENCODER relation state: identity handshake + native replay pin

Date 2026-11-11. Repo `/home/heheboiz/data/cir`, branch `research/cir-failure-workbench`.
Interpreter `workbench/envs/ctl312h/bin/python` (model worker `workbench/envs/encoder310-adapted/bin/python`).

## 1. Hook name + call site

* `CompositionAdapter.prepare_identity()` — base implementation is a NO-OP returning `None`
  (`workbench/composition/adapters.py:306`). Every adapter that draws nothing therefore mints the
  byte-identical run id it minted before the hook existed.
* New class attribute `CompositionAdapter.relation_bearing = False`; ENCODER sets it `True`
  (`model_encoder.py:83`).
* `workbench/composition/runner.py:1635` — called immediately after
  `adapter = spec.adapter_factory(config, checkpoint_id)` (line 1609) and BEFORE
  `identity = {` (line 1665), i.e. before the run id, the fingerprint, `run_dir`, `work_root`,
  `rank_path` and the model lock exist.
* Refusal: `runner.py:1636-1646` — if `relation_bearing` is true and the hook left no recorded digest,
  the run raises `relation_state_unrecorded` and is never published as `not_recorded`.
* ENCODER's implementation (`model_encoder.py:237`) starts the worker with phase `handshake`
  (`HANDSHAKE_PHASE = "handshake"`, `model_encoder.py:71`). `_worker_main` constructs the model under
  `torch.manual_seed(RELATION_SEED)`, digests `learnable_relation`, and replies with
  `{relation_digest, relation_seed, handshake: true}` — no image is opened, no forward pass runs
  (`model_encoder.py:526-533`). The handshake reply writes no tensor (there is no run directory yet);
  the first real encode job persists `relation-seed<seed>.npy` + `.json` under
  `runs/<run_id>/construction_state/`.
* Handshake scratch dir: `tempfile.mkdtemp(prefix="encoder-handshake-")`, moved under
  `work/<model>/` by the first exchange that has a `work_root` (`_job_root`, same filesystem), so the
  preserved `worker.stderr.log` still lands inside the run.

## 2. What the summary records now

* `runner.py:2178-2194`: after every encode call, `effective_relation = _relation_identity(
  adapter.relation_state)`. If `effective_relation["sha256"] != identity["relation_state"]["sha256"]`
  the run raises `relation_state_drift` (FAILED, no summary written claiming a digest).
* `runner.py:2239`: `summary["relation_state"] = dict(effective_relation)` — the EFFECTIVE block.
  `identity["relation_state"]` remains the fingerprint input (unchanged role).
* `_write_failed_results` now takes `relation_state=`; a failed summary carries the block beside
  `execution_status: FAILED` and the reason, so a refused run is legible as a refusal.
* Cache-key correctness: `runner.py:1771` adds `_recorded_relation_matches(saved,
  identity["relation_state"])` to the reuse condition. For a run whose current identity records a
  digest, the stored payload must record the SAME digest; a payload recording none (every payload the
  old code wrote) is not reused. For a model that draws nothing the check returns `True` and the
  fingerprint alone decides, so no other model's resume behaviour changes.

## 3. Native replay seed placement + artifact fields

* `workbench/replay/encoder.py`: `torch.manual_seed(RELATION_SEED)` at line 270, immediately before
  `model_try2.Encoder(...)` at 272 — inside `evaluate_categories`, not at module import and not after
  construction.
* `RELATION_SEED` / `RELATION_EXPECTED_DIGEST` are IMPORTED from
  `workbench.composition.model_encoder` (line 36), so the two paths cannot drift. No cycle:
  `composition.model_encoder` imports `replay.encoder` only inside `_worker_main`, after the module is
  loaded. Verified importable in both interpreters (`ctl312h`, `encoder310-adapted`).
* Docstring states the policy: "POLICY, not restoration … the released checkpoint does not contain
  `learnable_relation`, so the drawn tensor is reproducible under this seed and is not claimed to be
  the authors'".
* `_relation_record()` (`replay/encoder.py:240`) returns
  `{status, seed, sha256, expected_sha256, matches_pinned_digest, in_state_dict, seed_placement,
  policy, tensor, shape, dtype, path}` and writes `relation-seed<seed>.npy` + `.json` into the parent's
  observation directory (`WORKBENCH_OBSERVATION_OUT.parent`) when one is named — same filename
  convention as the composition worker.
* The dict is returned in BOTH places the native path exposes a result:
  `aggregate["relation_state"]` and each `ReplayResult.provenance["relation_state"]`, so the printed
  wrapper document (which the launcher parses and archives) carries it.
* A whole-pickled-model checkpoint (not a `state_dict`) sets `drawn=False`: the tensor then came from
  that file, and the record says so instead of claiming a seeded draw. A model object exposing no
  `backbone.binding_decoder.learnable_relation` yields `status: not_recorded` with a reason.

## 4. Tests (before → after)

`workbench/tests/test_encoder_relation_handshake.py` (8 tests, real `run_composition`, NumPy adapter):

| test | before (hook removed) | after |
|---|---|---|
| `test_the_identity_covers_the_state_the_handshake_produced` | FAILED `'not_recorded' == 'recorded'` | PASS |
| `test_encoding_with_a_different_relation_fails_the_run` | PASS (drift already raised) | PASS |
| `test_a_relation_bearing_adapter_that_cannot_name_its_draw_is_refused` | FAILED (`COMPLETED`, published) | PASS |
| `test_a_stale_payload_recording_no_relation_is_not_reused` | FAILED `True is False` | PASS |
| `test_the_hook_runs_before_the_identity_is_hashed` | FAILED `ValueError: substring not found` | PASS |
| `test_the_summary_block_carries_the_identity_fields[3 params]` | PASS | PASS |

Before/after command (runner edit reverted in place, then restored):

```
python -m pytest workbench/tests/test_encoder_relation_handshake.py -q   # before: 4 failed, 4 passed
python -m pytest workbench/tests/test_encoder_relation_handshake.py -q   # after:  8 passed
```

`workbench/tests/test_encoder_relation_seed.py` gained `test_the_handshake_phase_does_no_image_work`
(the handshake branch replies with the digest and calls no `Image.open`, no `extract_retrieval*`, no
`prefetch_rows`, no `np.save`).

`workbench/tests/test_encoder_relation_native_parity.py` (7 tests, stub vendor tree, real
`evaluate_categories`; runtime order recorded by substituting `torch.manual_seed`):

```
before (seed line removed): 2 failed  -> AssertionError: ['checkpoint', 'seed:0'] ; assert ([])
after:                      7 passed
```

The runtime order check is the load-bearing one: with the pin removed the first recorded event is
`checkpoint` and the only `manual_seed` call is the later `seeded_dataset_construction` (seed 0).

## 5. Bundled probe

Copied `/tmp/cir-review-06e30e8c/review_06e30e8c/protocol_probes/encoder_relation_runner_probe.py`
to `/tmp/r03_encoder_relation_runner_probe.py`, repointed `sys.path` to `/home/heheboiz/data/cir`, ran
it (exit 0).

* Before (recorded in `encoder_relation_runner_probe.output.json`): `summary_states == [{"status":
  "not_recorded", ...}]` while `adapter_state_after_encoding.matches_pinned_digest == true`.
* After: `summary_states == [{"expected_sha256": "worker-digest", "matches_pinned_digest": true,
  "seed": 20261111, "sha256": "worker-digest", "status": "recorded"}]`.
* That probe's synthetic adapter has no `prepare_identity`, so the digest is captured only when it
  encodes; the run now FAILS at the effective-state check with
  `ValueError: relation_state_drift: … computed its features with relation 'worker-digest' but the run
  id was minted over None`. That is the requested behaviour (a run may not claim a relation its
  features were not computed with) and the live digest IS what the summary then records.

## 6. Two-process parity

Checkpoint present: `workbench/artifacts/checkpoint_selection/20261106T002617+0700/cohort/downloads/
encoder/fashioniq.pt` (sha256 `0f31c501…9173`); GPU present (RTX 5070 Ti). Every launch went through
`workbench/scripts/gpu_job.sh` (one GPU job at a time).

* Construction path, two clean processes (`torch.manual_seed(RELATION_SEED)` then
  `model_try2.Encoder(...)`, `encoder310-adapted`):
  both `shape=(2, 512) digest=20db0c27c979e9a0938c71af5142458dfa2070468d0ad3fcb6f24134c2f9ce92
  matches_pin=True`.
* Real handshake through `model_encoder.Adapter.prepare_identity()`, two clean processes
  (`ctl312h`, worker in `encoder310-adapted`): both produced identity projection
  `{"seed": 20261111, "sha256": "20db0c27…ce92", "expected_sha256": "20db0c27…ce92",
  "matches_pinned_digest": true}`.

Full digests in `/tmp/r03_p1.log`, `/tmp/r03_p2.log`, `/tmp/r03_hs1.log`, `/tmp/r03_hs2.log`.

## 7. Superseded artifacts

Four published composition runs were produced with the unpinned relation and are superseded, because
their identity (`relation_state: not_recorded`) can no longer be minted:

| run id | execution_status | relation_state |
|---|---|---|
| `5dee7425dd46c5f154808aac` | COMPLETED | absent (predates the field) |
| `6a528e877c328772e2c7bb28` | COMPLETED | absent |
| `ce89a9a27c4cceadeaaf30b0` | COMPLETED | absent |
| `eda30a37c0bc17ac43844f09` | COMPLETED | absent |

Nothing was rewritten: no artifact was edited, deleted or moved. A new run mints a NEW run id (its
identity now records the digest) — that is the intended, stated consequence. Re-runnable on this host:
the checkpoint, the pinned source, `open_clip_pytorch_model.bin`, the FashionIQ root
(`data/FashionIQ` with `captions`, `image_splits`, `resized_image`) and the CUDA interpreter are all
present, so a fresh ENCODER composition run can be launched; it was NOT launched here (the full
gallery is ~74k images and no re-run was requested).

Same for the native numbers: a native replay now reproduces its relation draw from the code, so
prior native ENCODER numbers should be re-measured to carry the digest.

## 8. pytest (required selection)

`pytest workbench/tests -k "encoder or relation or composition" -q --basetemp=/tmp/r03_bt`
→ **291 passed, 6 failed, 1 skipped, 2668 deselected**. None of the six failures is caused by this
change; each was reproduced with my edits reverted (`git show HEAD:<file>` swapped in):

* `test_circus_adapters_blip.py::test_resident_worker_reuse_and_cleanup[conesep|ptha_mtst]` and
  `…::test_resident_worker_failure_uses_shared_error_and_stays_failed[conesep|ptha_mtst]` —
  `AttributeError: 'Adapter' object has no attribute 'batch_size'`. `_resident_adapter` builds the
  adapter with `__new__` and never sets `batch_size`, while `model_conesep.py:183` /
  `model_ptha_mtst.py` read `adapter.batch_size`. Pre-existing (reproduced on HEAD
  `adapters.py`); those two model modules are owned by another agent.
* `test_circus_tme_lifecycle.py::test_encoder_and_limn_failed_child_latch_and_preserve_log[encoder|limn]`
  — `RuntimeError: execution environment not verified … WORKBENCH_ENCODER_PYTHON not set`. Reproduced
  on HEAD `model_encoder.py` (2 failed in 0.23 s): the fixture never sets the per-model interpreter
  variable and the environment gate refuses before the worker is spawned. Pre-existing.

## 9. Files

Owned files changed: `workbench/composition/adapters.py`, `workbench/composition/runner.py`,
`workbench/composition/model_encoder.py`, `workbench/replay/encoder.py`.
New tests: `workbench/tests/test_encoder_relation_handshake.py`,
`workbench/tests/test_encoder_relation_native_parity.py`.

`adapters.py` is not in the explicit ownership list but is the file the brief names for the contract
("inspect `workbench/composition/adapters.py` first and follow its style") and the base hook has to
live there; the diff is 24 added lines, no existing behaviour changed.
