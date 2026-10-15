# C — model reproduction path (adapters / LIMN replay / registry / scripts)

Model audited: **limn** (deep trace) + **csmcir**, and readiness/refusal of the other 14 registered
models. Branch `research/cir-failure-workbench`, HEAD `0b0f3a6c3e1c640bbff8d658565f10090e406c7c`
(working tree: `pc.sh`, `workbench/tests/fixtures/`, `workbench/tests/test_pc_launcher.py`,
`workbench/docs/PC_ONE_COMMAND_GUIDE.vi.md` untracked; `workbench/backend/registry.py`,
`workbench/backend/paper_gate.py`, `workbench/tests/test_final_hardening_regressions.py` modified —
all inspected as the current source). READ-ONLY audit; the only file written is this report. No GPU
inference was run; every claim below is **STATIC_AUDIT** except where an observed command output is
quoted.

Scope files: `workbench/backend/adapters/base.py`, `workbench/backend/adapters/models.py`,
`workbench/replay/limn.py`, `workbench/replay/common.py`, `workbench/scripts/evaluate_models.py`,
`workbench/scripts/prepare_model.py`, `workbench/registry/models.yaml`.

---

## A) Files inspected

| file | lines read | role |
|---|---|---|
| `workbench/backend/adapters/base.py` | 1–43 (all) | `EvalRequest`, `validate_request` — the only integrity gate every adapter inherits |
| `workbench/backend/adapters/models.py` | 1–197 (all) | 15 adapter classes, `ADAPTERS`, `LIMNAdapter.command`, synthetic-adapter opt-in |
| `workbench/replay/limn.py` | 1–257 (all) | source-faithful LIMN runner: preflight, pinned `test()` call, bundle completeness |
| `workbench/replay/common.py` | 1–108 (all) | `ReplayResult`, hash-checked `torch.load`, git pin check, cache isolation, macro |
| `workbench/scripts/evaluate_models.py` | 60–200, 200–470, 470–600, 1000–1345, 1346–1625, 1626–1775, 1774–1902 | plan → launch → report → gate → proof |
| `workbench/scripts/prepare_model.py` | 39–86, 86–300, 477–670, 700–753 | derived-artifact preparation (resize, staged dictionaries, fixed dataset link) |
| `workbench/registry/models.yaml` | 1–300, 300–989 (all 16 models) | registry of record |
| `workbench/registry/preparation_contracts.yaml` | `limn_fashioniq` 249–300 | declared LIMN data contract |
| `workbench/registry/paper_benchmarks/papers.yaml` | `limn_fashioniq_base_iter0_table1` 1589–1650; all 33 rows' mapping field | committed benchmark reference |
| `workbench/backend/registry.py` | 20–240, 240–363, 600–743 | registry validation, `checkpoint_*` helpers, availability view |
| `workbench/backend/runtime.py` | 770–860, 1179–1400, 1600–1660, 1810–1900, 2200–2322 | interpreter resolution, adaptation, env blockers, launch blockers |
| `workbench/backend/preflight.py` | 1–216 (all) | shared requirement/blocker list incl. approved-patch state |
| `workbench/backend/paper_gate.py`, `benchmark_references.py`, `checkpoint_selection.py`, `metrics_extraction.py` | 1–473 / 1–195 / 100–220 / 60–215 | gate, reference selection, selection policy, LIMN parser |
| `workbench/backend/main.py` | 100–260, 360–467 | `/api/models`, `/api/jobs`, aggregate-report corroboration |
| `workbench/backend/index.py`, `storage.py`, `analysis.py` | 1–120 / 60–120 / 18–84 | served results: aggregate-only, no per-query rows for these models |
| `workbench/docker/run.sh`, `docker-compose*.yml`, `.env.example` | 1–272 / all / all | the launcher the new `pc.sh` drives |
| `pc.sh`, `workbench/docs/PC_ONE_COMMAND_GUIDE.vi.md`, `workbench/tests/test_pc_launcher.py`, `workbench/tests/fixtures/pc_launcher/**` | untracked, read | the one-command deliverable being built in this mission |
| `workbench/third_party/LIMN/**`, `workbench/third_party/CSMCIR/**` | `LIMN/{test.py,datasets.py,model.py,train.py}`, `CSMCIR/src/{validate_blip_csmcir.py,data_utils_csmcir.py,lavis/models/__init__.py}` | pinned upstream source |

## B) Execution paths traced

**LIMN, bundle `base_iter0_all_categories`.**
1. `pc.sh:585–613` reads `GET /api/models`, takes `checkpoint_bundles[0].bundle_id` →
   `base_iter0_all_categories`; `pc.sh:682–684` POSTs `{"model_id":"limn","checkpoint_id":
   "base_iter0_all_categories","protocol_id":"fashioniq_val_split","top_k":200}`.
2. `main.py:373–398` resolves the bundle to its anchor variant (`{**dress_variant,
   "checkpoint_id":"base_iter0_all_categories","bundle_id":...}`), then `guarded_plan`
   (`evaluate_models.py:331–404`).
3. `blockers()` (`:320–329`) calls `runtime_plan_blockers` → `preflight_blockers` +
   `checkpoint_bundle_missing_paths(model, root)` (`registry.py:327–335`) → blocks unless **all
   three** member files exist. Then `adapter_type().command(...)` (`:350`) → `LIMNAdapter.command`
   (`models.py:129–139`).
4. Launch: ledger key + duplicate-run-id pre-check (`:1358–1380`), `runtime.require_launch_ready`
   (`main.py:413–425`), `invoke_evaluator` (`:500–585`), report + gate + proof (`:1420–1620`).
5. Child: `workbench/replay/limn.py` re-validates source pin, all three checkpoints (SHA-256),
   dataset contract and environment (`:33–130`), imports the **pinned** modules
   (`:151–158`), `torch.load(weights_only=False)` only after the author hash matches
   (`common.py:38–50`), then `native_test.test(SimpleNamespace(local_rank=0, batch_size=32), model,
   dataset, category)` (`limn.py:178`), `macro_aggregate` (`:199–201`, `common.py:96–108`), JSON to
   stdout (`:222–244`).
6. `metrics_extraction.parse_limn_output` (`:74–215`) recomputes the macro from the categories and
   rejects an internally inconsistent aggregate.
7. Gate: `benchmark_references.select_reference` (`:149–195`) with
   `checkpoint_training_noise_pct=None` → `limn_fashioniq_base_iter0_table1` "exact condition match";
   `paper_gate.evaluate_gate` → `PASS_*` or a named refusal.
8. Serving: `main.py:128–196` serves a report ONLY if a ledger entry matches its digest, identity
   fields, `return_code == 0` and `invocation_status == "INVOCATION_COMPLETED"`.

**Dress-only run.** `--model limn --checkpoint base_iter0_dress` → `candidates()` returns the plain
variant (`evaluate_models.py:216–250`); `LIMNAdapter.command` appends `--category dress`
(`models.py:137–138`); the replay's `aggregate` is `{..., "complete": False}`
(`limn.py:200–202`); `parse_limn_output` returns `EVALUATION_COMPLETED_METRICS_UNPARSED` and
`observed_metrics=None` (`metrics_extraction.py:118–133`); the gate can only return a non-pass.
Verified by simulation below.

**CSMCIR.** `CSMCIRAdapter.command` builds `python src/validate_blip_csmcir.py --dataset fashionIQ
--blip-model-path <ckpt>` (`models.py:28–32`), cwd `<third_party>/CSMCIR/src`
(`evaluate_models.py:207–208`), and needs an approved patch to the pinned checkout plus a
`fashionIQ_dataset` symlink beside `src/`.

## C) Confirmed defects

### C-1 — P0 — the production plan path constructs the LIMN bundle command from a checkpoint id that does not exist in the registry, so `--model limn` can never launch

- Location: `workbench/scripts/evaluate_models.py:350` (calls `adapter_type().command`, bypassing
  `build_command`) + `workbench/backend/adapters/base.py:30` (`checkpoint_by_id(self.model,
  request.checkpoint_id)`) + `workbench/backend/adapters/models.py:137`.
- Root cause: two distinct code paths. `evaluate_models.guarded_plan` injects a **synthetic** bundle
  dict (`candidates()` builds `{**anchor, "checkpoint_id": bundle_id, "bundle_id": bundle_id}`,
  `:245–250`) and calls `adapter_type().command(...)` directly. `run_eval.py:25` calls
  `build_command`, which routes through `validate_request` → `registry.checkpoint_by_id`
  (`registry.py:628–633`), a function that only looks at `checkpoint_variants` and therefore raises
  `KeyError` for a bundle id. Only the replay's own CLI and `LIMNAdapter.command` implement the
  bundle exception; `validate_request` — which the docstring calls the safety property inherited by
  every adapter — does not.
- Reproduction (read-only, `.venv` interpreter from the repo root; monkeypatch used only to unblock
  the separate interpreter precondition):
  - `.venv/bin/python workbench/scripts/run_eval.py --model limn --checkpoint
    base_iter0_all_categories --protocol fashioniq_val_split --dataset-root /data/FashionIQ
    --output /tmp/o.json` →
    `File ".../registry.py", line 632, in checkpoint_by_id / KeyError: 'base_iter0_all_categories'`
  - `.venv/bin/python workbench/scripts/run_eval.py --model limn --checkpoint base_iter0_dress ...`
    → past the KeyError, fails on `execution environment not verified ... WORKBENCH_LIMN_PYTHON=None`.
  - With the two preconditions monkeypatched away (interpreter + bundle resolution),
    `LIMNAdapter().command(...)` returns the bundle command correctly — the defect is the
    resolution path, not the command text.
- Expected vs actual: expected `run_eval.py` (the documented "print the official command" tool,
  listed `FULLY_REVIEWED` in `workbench/docs/SOURCE_COVERAGE_MANIFEST.md:55`) to print the audited
  command for the registry's own bundle selection; actual `KeyError` traceback. Same for any
  future caller of `build_command`.
- Impact: on the target PC the API path is safe (it pre-resolves the bundle), but the CLI path
  crashes with an unhandled traceback, and `validate_request`'s integrity checks (checkpoint
  presence, mapping status, protocol) are silently skipped for every bundled model on that path.
- Minimal fix: one line in `validate_request` (`base.py:30`): fall back to the bundle anchor —
  `checkpoint = next(({**v, "checkpoint_id": cid, "bundle_id": cid} for b in self.model.get("checkpoint_bundles", []) if b["bundle_id"] == request.checkpoint_id for v in self.model["checkpoint_variants"] if v["checkpoint_id"] == b["required_checkpoint_ids"][0]), None)` — or have `run_eval.py` call `command()` with the same bundle-resolving helper `candidates()` uses. Do not add a second convention.

### C-2 — P0 — "complete FashionIQ macro result" is not derivable from the artifact the system serves

- Location: `workbench/scripts/evaluate_models.py:1465` (`"per_query_export_available": False`),
  `:1514–1519` (report written), `workbench/backend/main.py:128–262` (serving), `workbench/backend/index.py:47–54`.
- Root cause: `execute()` writes only an `official_aggregate_evaluation_report`; `index.py:47`
  deliberately excludes that artifact type from the DuckDB index, and no code path in the repository
  constructs a schema-v2 `ResultRun` (`grep -rln "ResultRun(" workbench | grep -v tests` → only the
  schema module itself). So `/api/queries`, `/api/compare`, top-K output and per-query rankings have
  no rows for any model evaluated through the guarded path.
- Reproduction: `grep -rn "top_k" workbench/backend/adapters/base.py` → the plan carries `top_k`
  (`base.py:18`, `evaluate_models.py:348`) and no adapter or replay consumes it; `grep -rn "ResultRun(" workbench`
  shows no producer.
- Expected vs actual: the UI/analysis contract (`workbench/backend/schemas/results.py:63`,
  `analysis.py:18–23`) requires per-query rows with `top_k_saved`; actual: aggregate-only runs, and
  `top_k` is a no-op for every registered model.
- Impact: after a *successful* LIMN run the operator gets aggregate R@10/R@50 (and per-category)
  but no failure analysis — no worst-ranked queries, no common distractors, no per-query rankings.
  Calling this an "aggregate report" is honest; calling the deliverable a failure-analysis workbench
  reproduction is not yet supported by the produced artifact.
- Minimal fix: state the limitation in the report header/serving layer (already partly done via
  `per_query_export_available`/`index_available`, `autopilot.py:954–968`) AND either have
  `workbench/replay/limn.py` emit a per-query `ResultRun` from the same `sims` matrix it already
  computes, or delete the top-K/analysis surface for aggregate-only models. Do not leave both.

### C-3 — P1 — the gate's own vocabulary check silently demotes every paper row whose mapping the reference registry spells `VERIFIED`

- Location: `workbench/backend/paper_gate.py:413–414` (`if reference_mapping and reference_mapping
  not in ("VERIFIED", "VERIFIED_METADATA", "NO_CHECKPOINT")` — the working-tree copy; the committed
  HEAD version at the same site lacks `"VERIFIED"`), row side
  `workbench/registry/paper_benchmarks/papers.yaml:1601` (`checkpoint_mapping_status: VERIFIED` for
  LIMN; 8 rows use `VERIFIED`), validator `workbench/scripts/build_paper_registry.py:43`
  (`VALID_MAPPING = {"VERIFIED","UNVERIFIED","NO_CHECKPOINT"}`) vs `workbench/backend/registry.py:27`
  (`mapping_statuses = {"VERIFIED_METADATA","UNVERIFIED","ARTIFACT_ASSOCIATED_UNVALIDATED"}`).
- Root cause: two registries use different words for the same fact, and the gate reads the paper
  registry's word with the model registry's vocabulary, capping the verdict.
- Reproduction (read-only simulation with the current working-tree code):
  `evaluate_gate(limn_fashioniq_base_iter0_table1, paper-exact metrics + paper-exact per-category
  metrics, run_meta{... "checkpoint_mapping_verified": True})` → `PASS_WITHIN_5_PERCENT,
  passed=True`. The same call with the reference row's own string `VERIFIED` (i.e. **HEAD's code**)
  → `PASS_METRICS_ONLY_MAPPING_UNVERIFIED, passed=False`. Observed in the live tree:
  `sprc_fiq_val_ours_vitg` (row status `UNVERIFIED`) → `PASS_METRICS_ONLY_MAPPING_UNVERIFIED`.
- Expected vs actual: a paper row that records its checkpoint mapping as established (`VERIFIED`)
  should not cap a run that also carries `plan.checkpoint_mapping_verified=True`; at HEAD it does.
  The working-tree edit fixes it but has not landed, and the vocabulary divergence is not validated
  anywhere: `benchmark_references.load_registry` (`:110–118`) checks only the top-level shape, and
  `build_paper_registry.py` (not run by any gate) normalises an unknown word to `UNVERIFIED`.
- Impact: at HEAD, LIMN's paper-exact run reports `passed=False`; on a release tag taken from a clean
  checkout of HEAD this is the shipped behaviour.
- Minimal fix: keep the working-tree `"VERIFIED"` addition **and** validate the field on load in
  `benchmark_references.load_registry` against a single declared vocabulary (or normalise
  `VERIFIED`→`VERIFIED_METADATA` once at load). Regression test in G-3.

### C-4 — P1 — `evaluate_models.py` exits 0 when a model was refused as un-runnable

- Location: `workbench/scripts/evaluate_models.py:334–335` (`return None, ["SKIPPED: adapter command
  not audited"]`), `:1831–1835` (`failed = failed or any(not reason.startswith("SKIPPED:") ...)`).
- Reproduction: `cd /home/heheboiz/data/cir && .venv/bin/python workbench/scripts/evaluate_models.py
  --model airknow` → prints `[SKIP] SKIPPED: adapter command not audited` twice,
  `echo $?` → **0**. Same for `hint`, `pair`, `conesep`, `habit`, `intent`, `tgcir`, `sprc`.
  `--model limn` → exit **1** (blocked); `--dry-run` → exit 1.
- Expected vs actual: an argparser-driven invocation that evaluated nothing should not exit 0 when
  its selection contains models with no audited evaluator; a CI/launcher that checks the exit code
  cannot distinguish "evaluated" from "refused".
- Impact: masks a whole cohort (the cohort (airknow, conesep, habit, intent, hint, pair, clvc_net, tgcir, sprc)) as success.
- Minimal fix: make `not command_is_audited` return a blocking reason (drop the `SKIPPED:` prefix for
  reasons that are not operator-requested skips), or return 1 when no model in the selection
  produced a plan. `workbench/scripts/pc_readiness.py` already reports these as `BLOCKED_IMPLEMENTATION`;
  the CLI exit code should agree with it.

### C-5 — P2 — an unavailable CSMCIR repair path is described by the CLI one-command surface as an automated step, but no code applies it

- Location: `workbench/registry/models.yaml:69–79` (`approved_source_patches` with
  `patch_file: workbench/patches/csmcir_models_init_compat.patch`), attempts to consume it:
  `workbench/backend/preflight.py:128–156` (`approved_patch_state`), `evaluate_models.py:162`,
  `runtime.py:2161`. No applier exists: `grep -rn "git apply|patch -p" workbench/scripts workbench/backend workbench/docker` → only the reader above.
- Reproduction (read-only, in `/tmp`): copy `workbench/third_party/CSMCIR/src/lavis/models/__init__.py`
  into a `src/lavis/models/` tree, `patch -p1 --batch < workbench/patches/csmcir_models_init_compat.patch`
  → exit 0 and the resulting file hashes `aa68fe5e20bcda5e17d978799deb8ec0ff4636e5b3b2a6ef637ce5e385b0c818`,
  exactly the declared `derived_sha256`. The patch is genuine and applicable; nothing applies it.
  Live state: `approved_patch_state(...)` → `{'applied': False, ... 'src/lavis/models/__init__.py is
  the unmodified published file; apply ... and register that checkout as the model source'}`.
  `pc.sh:544–548` path: `run.sh prepare csmcir` → `pipeline.py setup --model csmcir ... --apply`
  → the `dataset-link` stage and `prepare_model` run, but no patch stage exists, so CSMCIR stays
  `source cannot run` and its checkpoint stays undownloaded (the `checkpoint` stage is gated by the
  pipeline's sync/preparation but the download is a separate stage; observed registry/pipeline code,
  not a live run).
- Expected vs actual: the registry declares a named, hash-verified repair with an exact derived
  digest; the operator is told to apply it by hand. `pc.sh` docs present the CSMCIR flow as
  supported.
- Impact: the second deep-traced model cannot be made runnable by the one-command flow.
- Minimal fix: either add a `apply-approved-patch` pipeline stage that copies the pinned checkout to
  a derived directory and applies the declared patch, verifying `patch_sha256`/`base_sha256`/
  `derived_sha256`, or state in the report/guide that CSMCIR requires a manual operator step.

### C-6 — P2 — `prepare_model.py --dataset-root` prepares the configured root instead of the one it was given

- Location: `workbench/scripts/prepare_model.py:718` (`_fixed_link(config, ...)`) and `:477–670`
  (`_run_resize` writing under `config.FASHIONIQ_ROOT`), `:744` (`if
  model.get("canonical_dataset_link") ... _fixed_link`), versus the CLI `--dataset-root`
  (`:668–753`).
- Root cause: every preparation writer resolves from the frozen `WorkbenchConfig` built by
  `resolve_config()`; `--dataset-root` is parsed and (for the status path) used for checks, but the
  mutating paths never receive it, unlike `pipeline.py:1639–1663` which threads `target_root` into
  the stage outputs.
- Expected vs actual: `prepare_model.py --model limn --dataset-root /mnt/big/fiq --execute` should
  write `resized_image/*` and stage `captions/correction_dict_*.json` under `/mnt/big/fiq`; actual
  writes under `config.FASHIONIQ_ROOT`.
- Impact: silent split-brain dataset state on any host whose override is not also exported as
  `FASHIONIQ_ROOT`; the evaluator (which uses the container's `/data/FashionIQ`) then reports missing
  artifacts that were in fact produced elsewhere.
- Minimal fix: pass the parsed `--dataset-root` into `_run_resize`/`_stage_source_assets`/
  `_fixed_link` (a `replace(config, FASHIONIQ_ROOT=root)` at entry, which `pipeline.py` already does).

### C-7 — P2 — the GPU launch gate can be passed by a capability probe that never executes an sm_120 kernel

- Location: `workbench/backend/runtime.py:1814–1900` (`verify_gpu_runtime`), `:1676–1740`
  (`compute_capability` read from `nvidia-smi`), `workbench/docker/docker-compose.gpu.yml`
  (`count: all`).
- Root cause: the recorded capability describes the *device* (`compute_capability` from
  `nvidia-smi`) plus a torch CUDA probe; the LIMN historical pins (`torch==1.12.1`,
  `torchvision==0.13.1`, `models.yaml:936–937`) have no sm_120 kernels, so the recorded capability
  says `sm_120` while the model's own interpreter would fail at the first kernel with an
  "illegal instruction". The registry's own comment (`models.yaml:940–944`) states exactly this and
  offers `adapted_runtime` as the remedy; whether the recorded capability distinguishes which
  interpreter performed the tensor probe is not established by the lines read.
- Expected vs actual: expected the launch gate to refuse a launch whose interpreter cannot execute a
  kernel on the recorded device; actual behaviour not established here (`UNKNOWN` for the
  interpreter-vs-device binding).
- Impact: if the probe interpreter and the evaluator interpreter differ, the first real LIMN run on
  an RTX 5070 Ti dies at the first CUDA kernel — the precise failure the mission asks about.
- Minimal fix: probe inside the model's own resolved interpreter (the probe already runs a CUDA
  tensor in *a* model environment — confirm it is the same interpreter `require_launch_ready`
  returns) and record the executing interpreter's torch build; refuse when the interpreter's
  `torch.cuda.get_arch_list()` lacks the device's `sm_`.

### C-8 — P2 — `torch.load(weights_only=False)` is unconditional in the LIMN replay

- Location: `workbench/replay/common.py:38–50` — `inspect.signature(torch_module.load)` will always
  contain `weights_only` on torch ≥ 2.6, so the fallback branch is dead and the call is always
  `weights_only=False`.
- Root cause: a guard that reads as version-conditional but is not.
- Reproduction: `grep -n "weights_only" workbench/replay/common.py` → the `in` check and the
  unconditional `weights_only=False`; the author hash match happens first (`:42–44`), so the trust
  argument holds.
- Impact: low — the pins do not support `weights_only=True`, and the byte hash precedes the load.
  Risk is that the code reads as if it were constrained when it is not.
- Minimal fix: replace the branch with a plain `weights_only=False` plus the existing comment, or a
  real version comparison.

### C-9 — P3 — `checkpoint_availability` hashes whole checkpoints on the request path; the working-tree cache is in-process only

- Location: `workbench/backend/registry.py:709` (HEAD) / working tree `_cached_availability_digest`
  (uncommitted), consumed by `main.py:108`.
- Reproduction: `checkpoint_bundle_missing_paths(limn, checkpoints_root)` → `{}` (all three 1.79 GB
  files present and hash-correct; `sha256sum` verified `96a335d4…`, `9260c7f7…`, `79f663a1…` against
  `models.yaml:963,972,981`). A 5.4 GB hash per `/api/models` request made readiness polling time
  out; the working-tree cache (keyed by size+mtime_ns) fixes it for a single process, and
  `workbench/tests/test_final_hardening_regressions.py` (working tree) asserts it.
- Impact: P3 after the cache lands; P2 if an operator runs HEAD.
- Minimal fix: land the cache (already written) and keep the authoritative hashing uncached, as the
  comment on the new helper states.

### C-10 — P3 — the declared `expected_size_bytes` is not enforced at verification time

- Location: `workbench/registry/models.yaml:962,972,981`;
  `workbench/scripts/download_checkpoints.py:241–248` (checked at download);
  `workbench/backend/runtime.py:2311–2318` and `workbench/replay/limn.py:40–48` (presence + SHA-256).
- Reproduction: `grep -rn "expected_size_bytes" workbench/backend workbench/replay` → no reader.
  Verified all three LIMN files are exactly 1792010157 bytes, matching the declaration.
- Expected vs actual: a truncated-but-hash-colliding file is impossible, so size adds little; it is
  simply an unused declaration on the verification path.
- Impact: negligible; noted for registry hygiene.
- Minimal fix: either check it next to the SHA-256 comparison or drop the field from the schema for
  entries whose hash is already verified.

## D) Probable defects needing reproduction

- **D-1 (needs the real LIMN run): does the adapted sm_120 stack deserialize and run the LIMN
  checkpoint at all?** The pickle references `open_clip.model CLIP`, `open_clip.tokenizer tokenize`,
  `open_clip.transformer {LayerNorm,ResidualAttentionBlock,Transformer,VisionTransformer}` and
  `model.{LIMN,Backbone,PositionalEncoding}` (observed with `pickletools.genops` over the
  checkpoint's `archive/data.pkl`). The adapted profile pin is `open_clip_torch>=2.24,<2.27`
  (`models.yaml:950`) while the pinned source is `2.20.0` (`:939`). Whether unpickling
  `CLIP`/`tokenize` survives the module reshuffles between 2.20 and 2.24+ is **UNVERIFIED** and can
  only be settled on the GPU host.
- **D-2 (needs the real run): wall-clock feasibility.** `limn.py:178` calls the pinned `test()` with
  `batch_size=32` and no DataLoader; the pinned dataset decodes and caches every image twice per
  category (`datasets.py:136–170` `get_test_img` on `source_img_data` and `target_img_data`) and
  caches `{name}_train_data.json`, `test_queries_{name}.pkl` and `test_targets_{name}.pkl` into the
  replay's temp root (`datasets.py:50–66`). Peak host RAM for one category is therefore the whole
  decoded val image set. No host-RAM check exists in the launcher path; `pc.sh`/`run.sh` check disk
  and docker only. **UNVERIFIED**.
- **D-3 (needs a real report): whether a bundle report can ever be recorded under the bundle id.**
  `run_report_path`/`completion_proof_path` (`evaluate_models.py:1319–1324`) are
  `{model_id}_{checkpoint_id}_{run_id}_aggregate.json`, i.e. `limn_base_iter0_all_categories_…`.
  The API resolves the bundle to the anchor variant carrying `checkpoint_id = bundle_id`, so this
  should be consistent — but `candidates()` and `main.queue_job` are two independent bundle
  resolvers, and no test drives the second. **UNVERIFIED** without a live queue run.

## E) Test coverage gaps

1. **No test calls `build_command`/`validate_request` for a bundle id.** `grep -rn "validate_request"
   workbench/tests` → 0 hits; every LIMN test calls `adapter.command(...)` directly
   (`test_evaluate_models.py:177–178`, `test_release_blocker_interpreter.py:200`,
   `test_strict_acceptance_matrix.py:91–96`). This is exactly why C-1 survived every wave.
2. **`run_eval.py` has no test at all** (not referenced by any test file).
3. **No test asserts the CLI exit code for a model whose command is un-audited** — C-4.
4. **No cross-registry vocabulary test** — `VERIFIED` (papers.yaml) vs `VERIFIED_METADATA`
   (models.yaml); C-3.
5. **No test applies the declared CSMCIR patch end-to-end** (compute `derived_sha256` and compare) —
   C-5. The digest was verified by hand here only.
6. **No test exercises `prepare_model.py --execute --dataset-root X` with X ≠ `FASHIONIQ_ROOT`** —
   C-6.
7. **No test asserts per-query export** for any model; `per_query_export_available` is hardcoded
   `False` in three places in `autopilot.py` and the evaluator, with no test requiring top-K rows for
   a successful run — C-2.

## F) Recommended minimal fixes

1. `base.py:30` — resolve a declared bundle id to its anchor in `validate_request` (5 lines, one
   convention, mirrors `main.queue_job:382–392`). Fixes C-1 for every bundled model and closes the
   skipped-integrity-check hole.
2. Land the `"VERIFIED"` vocabulary fix **and** validate `checkpoint_mapping_status` in
   `benchmark_references.load_registry` against one declared vocabulary. Fixes C-3 at the source
   instead of at the reader.
3. Make "no audited command" a blocking reason in `evaluate_models.main` (or return 1 when the
   selection produced no plan). Fixes C-4.
4. Either add an approved-patch pipeline stage (verify the three declared digests, copy to a derived
   checkout, apply) or mark CSMCIR's patch as a manual operator step in the registry, the guide and
   the readiness output. Fixes C-5.
5. Thread the parsed `--dataset-root` into `prepare_model.py`'s mutating paths. Fixes C-6.
6. Record the interpreter whose torch build was probed in the GPU capability (and its
   `get_arch_list()`), and refuse a launch whose arch list lacks the device's `sm_`; run the probe
   inside the model's resolved interpreter. Closes the C-7/D-1 uncertainty.
7. Delete the dead `weights_only` branch (C-8); land the availability digest cache (C-9); check or
   drop `expected_size_bytes` (C-10).
8. State, in the report/README, that these models produce aggregate-only evidence (no per-query
   rankings) until `workbench/replay/limn.py` emits a schema-v2 `ResultRun`, or remove the top-K UI
   affordances for them (C-2).

## G) Exact regression tests required

- **G-1** `test_run_eval_prints_bundle_command_for_limn`: call `LIMNAdapter().build_command(EvalRequest("limn","base_iter0_all_categories", ...))`
  with a monkeypatched interpreter and a present-checkpoint registry; assert no exception and
  `"--category" not in command`. Fails at HEAD with `KeyError` (C-1).
- **G-2** `test_validate_request_accepts_declared_bundle_only`: `validate_request` must still reject an
  unknown id (`nope`) with `KeyError` and a bundle whose member is missing with
  `FileNotFoundError`; must accept the declared bundle (C-1).
- **G-3** `test_reference_mapping_vocabulary_is_accepted`: for
  `limn_fashioniq_base_iter0_table1` at paper-exact metrics and paper-exact category metrics, assert
  `status == "PASS_WITHIN_5_PERCENT"` and `passed is True` at HEAD (not only in the working tree);
  plus a row asserting `build_paper_registry`/`load_registry` reject an unknown mapping word (C-3).
- **G-4** `test_unaudited_model_selection_exits_nonzero`: run `evaluate_models.main(["--model","airknow"])`
  and assert the return code is non-zero (C-4).
- **G-5** `test_csmcir_patch_produces_declared_derived_digest`: apply
  `workbench/patches/csmcir_models_init_compat.patch` to a copy of the pinned file in `tmp_path` and
  assert `sha256 == approved_source_patches[0]["derived_sha256"]`; a second assertion that the
  pipeline offers a stage whose output is that derived checkout (C-5).
- **G-6** `test_prepare_model_writes_under_dataset_root_override`: with `config.FASHIONIQ_ROOT` set to
  A and `--dataset-root B`, assert every write lands under B and nothing is created under A (C-6).
- **G-7** `test_gpu_capability_records_interpreter_and_arch_list`: assert the recorded capability
  names the interpreter's `torch.cuda.get_arch_list()` and that a capability whose arch list lacks
  the device's `sm_` is refused at `require_launch_ready` (C-7/D-1).
- **G-8** `test_limn_single_category_is_never_a_complete_benchmark`: with a fake `replay` returning
  `("base_iter0_dress", "dress")`, assert `evaluate_categories(..., categories=("dress",))["complete"] is False`,
  `parse_limn_output` yields `observed_metrics is None`, and `evaluate_gate` returns a non-passing
  status for the same evidence (already partly covered by `test_evaluate_models.py:164–181`; add the
  gate half).
- **G-9** `test_limn_replay_refuses_partial_bundle`: with only `0_dress_best_model.pt` in the
  checkpoint root, assert `validate_checkpoint_bundle(root)` names the two missing categories and
  `preflight` returns them (observed: `{'base_iter0_all_categories': ['…/0_dress…','…/0_shirt…','…/0_toptee…']}`).
- **G-10** `test_limn_dress_only_report_never_becomes_official` (needs a live queue): assert a
  dress-only run's completion proof/report either is never written under the official reports
  directory or is refused by `main._aggregate_reports` (bundle-id identity check) (D-3).

---

## Checklist answers

| # | question | verdict | evidence | severity | action |
|---|---|---|---|---|---|
| 1 | Registry integrity (16 models, 3 protocols) | **VERIFIED** | `registry.load_registry()` loads; 16 model ids and 3 protocol ids printed; `registry.py:20–123` validates protocols, layouts, link names, extensions, mapping/source vocabularies, bundle membership | NONE | none |
| 2 | Source/checkpoint mapping | **VARIANT** | `models.yaml:956–983` lists author Hub URLs at revision `30560ad5…`, hashes and exact sizes; `LIMN` checkout HEAD `7d7bc9b1…` equals `upstream_commit_sha`; three local files hash-match the declared digests (`sha256sum`, observed) | NONE | none |
| 3 | Checkpoint SHA-256 validation | **VERIFIED** | `limn.py:40–48` (per category), `common.py:38–50` (before unpickle), `runtime.py:2316–2318` (plan side), `registry.py:283–297` (`checkpoint_missing_paths`); `expected_size_bytes` unused (`grep` → download path only) | P3 (C-10) | check or drop the size field |
| 4 | Checkpoint bundle selection | **VERIFIED** | `models.yaml:984–989`; `registry.py:327–335`; `runtime.py:2319–2320`; `evaluate_models.py:216–250, 359–364`; run with one of three files present → `checkpoint bundle incomplete: base_iter0_all_categories: …` | NONE | none |
| 5 | Preprocessing | **VERIFIED** | `limn.py:168–172` builds train/val transforms from the checkpoint's own `visual.image_size/image_mean/image_std` (present in the pickle: `strings` → `image_size`, `image_mean`, `image_std`); pinned `train.py:78` uses `open_clip.create_model_and_transforms('ViT-L-14', pretrained=<author path>)`, so an import-free checkpoint is expected to carry the same values | NONE (UNKNOWN only for the exact values) | none; record `visual.image_size` in the run report when the real run happens |
| 6 | Dataset layout | **VERIFIED** | `models.yaml:916–917` (`fashioniq_ilearn_resized_training`, `.jpg`); `limn.py:52–67` requires train+val captions, correction dicts, val splits, `resized_image/<cat>`; consumers pinned at `datasets.py:44,68,70,112,121,137,139,141` | NONE | none |
| 7 | CUDA requirements | **VARIANT (needs hardware)** | `models.yaml:938,945–953` historical `12.4`, adapted `sm_120` profile; `limn.py:120–123` refuses a run without `torch.cuda.is_available()`; the recorded capability may describe a different interpreter than the evaluator's (C-7) | P2 (C-7) | record/probe the interpreter's arch list |
| 8 | Native evaluation commands | **VERIFIED** for LIMN-replay; **FALSE_POSITIVE** for "official CLI" | LIMN has no upstream eval CLI in the pinned tree (`train.py` only; `test.py` has no `__main__`), and the registry declares `replay_type: SOURCE_FAITHFUL_WORKBENCH_REPLAY` (`models.yaml:901–905`), which the replay honours by importing the pinned modules and asserting `module.__file__` (`limn.py:151–158`) | NONE | none |
| 9 | Adapter argument construction | **VERIFIED** except the bundle resolution path | `models.py:129–139`; forced call returns `[…/replay/limn.py, --checkpoint-root <root>, --dataset-root <root>, --source-root <root>]` + `--category dress` for a category run; C-1 for `build_command` | P0 (C-1) | fix `validate_request` |
| 10 | Model imports | **VERIFIED** | `model.LIMN`, `model.Backbone`, `Transformer.*`, `open_clip.*` are the pickle's globals (`pickletools` over `archive/data.pkl`); replay inserts only the pinned `LIMN` dir (`limn.py:148`); D-1 flags the 2.20→2.24 open_clip jump | NONE (P2 risk D-1) | runtime check on the GPU host |
| 11 | Checkpoint deserialization | **VERIFIED with caveat** | `common.py:42–50` hash-first then `torch.load(weights_only=False)`; `limn.py:165–166` asserts `isinstance(model, native_model.LIMN)`; the `weights_only` branch is dead code (C-8) | P2 (C-8) | delete the dead branch |
| 12 | Per-category execution | **VERIFIED** | `limn.py:188–201` replays exactly `CATEGORIES`, rejects duplicates/unknowns, verifies each result's identity, and returns `complete: False` for any subset | NONE | none |
| 13 | Evaluation output completeness | **VARIANT** | JSON carries `complete_benchmark`, per-category rows and `aggregate` (`limn.py:222–236`); `parse_limn_output` refuses a partial/duplicated/inconsistent set (`metrics_extraction.py:105–190`); but the report is aggregate-only — no per-query rows ever exist (C-2) | P0 (C-2) | emit `ResultRun` or declare aggregate-only |
| 14 | Per-query rankings | **NOT_RUN** | no producer: `grep -rln "ResultRun(" workbench \| grep -v tests` → only `schemas/results.py`; `evaluate_models.py:1465` hardcodes `per_query_export_available: False` | P0 (C-2) | see C-2 |
| 15 | Top-K output validity | **BLOCKED** | `EvalRequest.top_k` (`base.py:18`) reaches the plan (`evaluate_models.py:348`) and is consumed by no adapter or replay (`grep`); `analysis.require_top_k` (`analysis.py:18–23`) can therefore never be satisfied for a guarded run | P0 (C-2) | same |
| 16 | Model-specific protocol behaviour | **VERIFIED** | `models.yaml:920–922` native/supported = `fashioniq_val_split` only; `base.py:28–29` refuses other protocols; the pinned evaluator masks the source image and uses a pair-union gallery (`test.py:80–81,86–93`; `datasets.py:143–170`) and `limn.py` hardcodes `protocol="fashioniq_val_split"` | NONE | none |
| 17 | LIMN bundle: selection → command → output → report → gate | **VARIANT** | selection/command/output/report traced end to end (B, E); the gate reaches `PASS_*` only with the working-tree vocabulary patch (C-3), and `build_command` cannot construct the bundle command at all (C-1) | P0/P1 | C-1, C-3 |
| 18 | Can a dress-only run be recorded/served as a complete FashionIQ reproduction? | **NO — VERIFIED** | `limn.py:188–202` (`complete: False` for a subset) → `parse_limn_output` (`metrics_extraction.py:118–133`) `observed_metrics=None` → gate: dress-only `checkpoint_id` → `BLOCKED_CHECKPOINT_MISMATCH`; a bundle-id run with only dress evidence → `PASS_WITH_UNCHECKED_CATEGORIES` with `passed=False`; `limn.py:236–243` forbids `--output` under a `reports` directory; `evaluate_models.py:1509–1519` only writes official reports; `main.py:128–196` serves only ledger-corroborated official reports. All observed by simulation | NONE | none (defence in depth is real) |
| 19 | CSMCIR readiness and refusal honesty | **VERIFIED (refusal) / BLOCKED (readiness)** | `validate_blip_csmcir.py:501–515` CLI matches the adapter; pinned `src/lavis/models/__init__.py` is byte-identical to `base_sha256` (`sha256sum`); the declared patch applies cleanly and reproduces `derived_sha256` exactly (tested in `/tmp`); preflight reports `source cannot run: …`, the missing `fashionIQ_dataset` link and the missing checkpoint; but no code applies the patch (C-5) | P2 (C-5) | add the patch stage or document the manual step |
| 20 | Remaining 14 models: readiness / safe refusal | **VERIFIED** | `evaluate_models --dry-run` per model: `airknow`, `conesep`, `habit`, `intent`, `hint`, `pair`, `clvc_net`, `tgcir`, `sprc` → `SKIPPED: adapter command not audited` (no invented command); `encoder`, `dcnet`, `combiner_rn50x4_noft`, `clip4cir_rn50x4_fullft`, `csmcir`, `limn` → named blockers (missing imports/assets/checkpoints/dataset artifacts, patch not applied, interpreter unset); `ptha_mtst` → no output, non-zero exit | P1 (C-4, exit code) | C-4 |
| 21 | Unavailable author resources never appear available | **VERIFIED** | `checkpoint_selection.select_checkpoints(model_by_id("limn"))` → `BLOCKED / BLOCKED_CHECKPOINT`, "no checkpoint variant records a training condition, so none can be called clean"; unverified mappings are refused by `base.py:34–35`, `runtime.py:2314–2315`; assets with no checksum are `EXISTS_UNVERIFIED_CHECKSUM`/`UNRESOLVED_PROVENANCE` (`registry.py:240–270`) | NONE (P1 note: LIMN's null training condition blocks the autopilot's default `strict_clean`, so per-category runs are refused with an accurate reason — honest, but it means `ptha_mtst` and LIMN need explicit policy/override) | none |
| 22 | `pc.sh` feasibility (the deliverable being written now) | **BLOCKED** | `pc.sh` now exists (untracked, 40 KB, `-rwx`), picks the bundle id for LIMN via `/api/models`, calls `run.sh prepare`, POSTs `/api/jobs`, polls and prints the report; it depends on `docker`, `.env`, and — for any metric verdict — on C-1/C-3/C-6 landing, since `/api/models` hashes 5.4 GB per poll at HEAD | P0 | land C-1, C-3, C-6, C-9 before shipping the tag |

## Execution trace

- Registry: `load_registry()` → 16 models, 3 protocols; `checkpoint_bundle_missing_paths(limn)` → `{}`.
- LIMN source: `git -C workbench/third_party/LIMN rev-parse HEAD` → `7d7bc9b116f594a65ac22457491edf28a88d3c3e` (= pin); `git status --porcelain` → only `__pycache__` entries (ignored by `source_checkout_status`, `common.py:59–77`).
- LIMN checkpoints: `stat` → 1792010157 bytes each; `sha256sum` → `96a335d4…`, `9260c7f7…`, `79f663a1…`, all equal to `models.yaml:963,972,981`.
- Command construction: `run_eval.py --model limn --checkpoint base_iter0_all_categories …` → `KeyError: 'base_iter0_all_categories'` at `registry.py:632`; category variant → only the interpreter precondition fails; monkeypatched `LIMNAdapter().command(...)` → correct bundle argv.
- Plan path: `evaluate_models.py --model limn --dry-run` → `[BLOCKED] official command construction failed: execution environment not verified [UNCONFIGURED/UNCONFIGURED]: WORKBENCH_LIMN_PYTHON=None`, exit 1.
- Refusal sweep: 15 `--dry-run` invocations, outputs quoted in checklist #20; `--model airknow` exit 0 (C-4).
- Gate simulations (read-only, `evaluate_gate` called directly): paper-exact LIMN run → `PASS_WITHIN_5_PERCENT` (working tree) / `PASS_METRICS_ONLY_MAPPING_UNVERIFIED` (HEAD vocabulary); dress-only → `BLOCKED_CHECKPOINT_MISMATCH`; bundle with 1 or 2 categories → `PASS_WITH_UNCHECKED_CATEGORIES, passed=False`; `sprc` exact-match row → `PASS_METRICS_ONLY_MAPPING_UNVERIFIED`; all 33 rows simulated with each row's own reported values as the observed run (`Counter`): 15 `PASS_WITH_UNCHECKED_CATEGORIES` (passed=False), 13 `PASS_METRICS_ONLY_MAPPING_UNVERIFIED` (passed=False), 5 `METRIC_REFERENCE_UNVERIFIED`. No row produces `PASS_WITHIN_5_PERCENT` because none pits its category table against a run that observed it, which is the category-coverage behaviour recorded in C-2 and checklist row 13.
- Patch check (in `/tmp`): `patch -p1 --batch < workbench/patches/csmcir_models_init_compat.patch` → exit 0, derived `sha256 = aa68fe5e…` = declared `derived_sha256`.
- Targeted tests: `pytest workbench/tests/test_evaluate_models.py -q` → 18 passed; `pytest workbench/tests -q -k "bundle or checkpoint_selection"` → 39 passed; `grep -rn "validate_request" workbench/tests` → 0 hits.
- No GPU command, no container build, no download, no full suite.

## Final verdict

**BLOCKED** — the LIMN reproduction path is source-faithful, correctly refuses to run on this host
(missing prepared artifacts, unset `WORKBENCH_LIMN_PYTHON`, no GPU probe), and cannot record or serve
a dress-only run as a complete FashionIQ reproduction (**VERIFIED, with observed gate/parser
evidence**). But the exact command the system builds for `--model limn` on a fresh PC cannot
currently be produced by the audited command builder, and the gate at HEAD demotes a paper-exact LIMN
run because of a vocabulary mismatch between the two registries.

**Exact command line the system builds for `--model limn` on a fresh PC** (forced through
`LIMNAdapter.command` with the interpreter precondition satisfied; verified for shape):

```
/artifacts/environments/<env-hash>/bin/python /repo/workbench/replay/limn.py \
  --checkpoint-root /checkpoints/limn \
  --dataset-root /data/FashionIQ \
  --source-root /third_party/LIMN
```

(no `--category`, i.e. all three categories; `--checkpoint-root` is `checkpoint.parent`, and the
bundle/argv/exit-code handling downstream is complete). **It is correct in content — the replay's
CLI matches these four options, and every `--category` default is the full `CATEGORIES` tuple — but
it is not reachable through `build_command`/`run_eval.py`, which raise `KeyError:
'base_iter0_all_categories'` at `adapters/base.py:30` via `registry.py:632`; the production
`evaluate_models.py` plan path reaches it only by bypassing `build_command`, and it is additionally
gated by the environment, bundle-integrity, source-pin and dataset preflights, which on a clean host
block until the environment, the three 1.79 GB checkpoints, the resized images and the staged
correction dictionaries all exist.**

**Can this implementation fail during the real first run on an RTX 5070 Ti PC? YES.** Three
independent ways, in likelihood order: (1) `LIMNAdapter.build_command`/`run_eval.py` raise `KeyError`
for the bundle id — C-1; (2) `prepare_model.py` writes derived artifacts under the *configured*
dataset root rather than the `--dataset-root` it was given, so the evaluator can report artifacts
missing that were just produced elsewhere — C-6; (3) the recorded GPU capability can be produced by a
probe whose interpreter is not the one that will run LIMN, so a stack without sm_120 kernels can pass
the launch gate and die at the first kernel — C-7/D-1. Even when all three are cleared, the first run
can only ever yield aggregate evidence (no per-query rankings, no top-K), and at HEAD the resulting
paper-exact numbers are reported with `passed: false` because of the registry vocabulary mismatch —
C-2, C-3.
