# Slice B — GPU / MODEL RUNTIME (Blackwell sm_120) — independent source audit

Auditor: independent read-only reviewer (slice B). Branch `research/cir-failure-workbench`, HEAD `e9cafaf`
plus the main agent's **uncommitted working-tree edits** for the P0-A/P0-B wave (that wave was still being
written while this audit ran; every finding below was re-verified against the tree as it stood at the end).

Revision audited (md5 of the working tree; verified by md5sum, read-only):

```
a263d40955fbf6222c0cad210953b9e6  workbench/backend/runtime.py
6cd262b53c6177edc43840f60760218e  workbench/backend/environment_planner.py
b572e2282a91eaebc8665e0e128cd1d5  workbench/scripts/evaluate_models.py
b4adee7a1f7b0f14253ee43096629a4c  workbench/replay/limn.py
24c795da907e9c7da44b377fa372d3e7  workbench/replay/common.py
012378335b16d32dc583dde53196e17a  workbench/registry/models.yaml
c3fd90270c1edab17bfbd7885c54aa01  workbench/scripts/manage_environment.py
d557ef46f36c8de26130f32e29bda247  workbench/scripts/pc_readiness.py
```

Every verdict is `STATIC_AUDIT` (source read + in-process reproduction of pure logic) unless the command
line is quoted; **no GPU inference was run and no runtime result is claimed as observed**. `docker` was not
invoked (slice A owns it). All workbench docs, including prior audits, were treated as hypotheses only.

Honest caveat, stated first: the P0-A/P0-B repair is a *moving target*. Three findings below (B-3, B-9, B-10)
changed state between 23:49 and 00:0x because the main agent was editing `runtime.py`,
`environment_planner.py`, `evaluate_models.py`, `registry/models.yaml` and `docker/run.sh` concurrently with
this audit. Each is marked with the state observed at the hashes above.

---

## 1. Checklist answers

### Q1. Environment identity / fingerprint — does `_environment_id` incorporate the selected runtime profile?

**Verdict: VARIANT (no collision; the risk is real but not currently reachable for `limn`) — severity P2**

- `_environment_id(members, constraints)` = `f"cir-{_environment_digest(...)[:16]}"`
  (`workbench/backend/environment_planner.py:780`), and `_environment_digest` hashes
  `members` + `constraints[python|torch|torchvision|cuda|packages|ranges]`
  (`environment_planner.py:774-776`). The profile name is **not** a key: two profiles that produced the
  same constraint tuples would share one `environments/cir-<hash>` directory.
- There is therefore **no structural guard** that "historical" and "adapted" can never share a directory.
  Today the two plans differ anyway, because the adapted block changes torch/torchvision/cuda/packages.
- Reproduced in-process (`python3 - <<EOF` with an injected solver, key `_SOLVE_CACHE.clear()`d between runs):
  ```
  HIST : ('cir-1bbd9872cf166f0d', [], ([('==','1.12.1')], [('==','12.4')], [('==','2.20.0')]),
          '.../artifacts/environments/cir-1bbd9872cf166f0d')
  ADAPT: ('cir-19b08a0cbb9354e1', [], ([('<','3'),('>=','2.7')], [('==','12.8')], []),
          '.../artifacts/environments/cir-19b08a0cbb9354e1')
  COLLIDE env_id: False
  ```
  The `cir-1bbd9872cf166f0d` value is the *same* id as the already-provisioned
  `workbench/artifacts/environments/cir-1bbd9872cf166f0d` (torch 1.12.1 confirmed on disk in
  `lib/python3.10/site-packages/torch-1.12.1.dist-info/METADATA`) — i.e. the historical id is stable and
  the adapted id is derived from a genuinely different lock, not from the profile name.
- The N=1 experiment above is only valid with the current pair table; before the main agent's edit the
  adapted plan never formed a group at all (see B-9), so no collision was reachable.
- **Recommended action (P2, cheap):** add the adaptation target to the digest — one line in
  `_environment_digest`'s `lock` dict (`"runtime_adaptation": constraints.get("runtime_adaptation")`)
  using the key `model_constraints` already emits at `environment_planner.py:511`. That makes the identity
  property structural instead of incidental.

STATIC_AUDIT.

---

### Q2. Solver spec generation under adaptation (`_conda_constraints` / `_solver_solve` / `_conda_specs_for_group`, `torch>=2.7` ranges)

**Verdict: VARIANT — severity P2 (mitigated by a concurrent main-agent fix; verified at the hashes above)**

Trace, all in `workbench/backend/environment_planner.py`:
- `_TORCHVISION_PAIRS` (`:171-190`) ended at `("2.6.0","0.21.0")` at HEAD (`git show HEAD:...`), while the
  adapted profile requires `torch>=2.7,<3` + `torchvision>=0.22,<1`. `_constraints` rejects any pair absent
  from that table (`:563-567`), and `_conda_constraints` (`:625`) merges `_NON_CONDA_PACKAGES`
  (`:601-608`, which includes `torch`, `torchvision`, `open-clip-torch`) into `pip_only`, so conda **never
  receives a torch spec** — the range can only fail at the pairing check, never at the solver.
- Observed at HEAD logic (reproduced with the pair table truncated to its HEAD content):
  `plan_environments([limn], solver=...)` → `groups: []`,
  `blocked: [{"model_id":"limn","reason":"PyTorch/torchvision constraints have no official binary pairing; ..."}]`.
  So the whole adapted journey was **dead on arrival** via the planner.
- At the hashes above the main agent added `("2.7.0","0.22.0"), ("2.7.1","0.22.1"), ("2.8.0","0.23.0"),
  ("2.9.0","0.24.0"), ("2.9.1","0.24.1")` (`environment_planner.py:186-188`) and the adapted group now
  forms.
- Will micromamba resolve? `python` is what reaches the solver: for `limn` the registry declares
  `python: UNKNOWN` → `_terms` returns `[]` → `_conda_specs_for_group` falls back to `python=3.10`
  (`environment_planner.py:1088-1096`, reproduced: `conda specs: ['python=3.10','pip','numpy=1.21.5',
  'packaging=23.1','pillow=9.4.0','pyyaml','tqdm=4.65.0']`). No torch, no cuda, no open-clip reaches conda.
  The `_solve` cache key excludes the adaptation for the injected-solver path but the constraints differ,
  so no stale hit. **The conda side of the adapted solve is sound within these channels.**
- **Remaining sharp edge:** the pair table is a hand-maintained list, so the *next* torch major silently
  re-breaks adaptation exactly the way 2.7 did. **Recommended action:** replace the table lookup for the
  un-pinned-range case with the same "any overlap" rule used for exact pins (or add a unit assertion that
  every adapted profile in the registry has a matching pair entry).

STATIC_AUDIT.

---

### Q3. Python version bounds vs adapted torch — solver feasibility

**Verdict: PASS for `limn` (no conflict), UNKNOWN as a general guard — severity P3**

- `limn` declares `python: UNKNOWN` (`registry/models.yaml:931`), so the planner defaults to `python=3.10`
  — compatible with `torch>=2.7,<3`. No conflict for the model this slice owns.
- Cross-check of the most recent torch resolutions (network, `pypi.org/pypi/torch/<v>/json`):
  `2.7.0 requires_python >=3.9.0`, `2.7.1 >=3.9.0`, `2.9.0 >=3.10`. `python>=3.9,<3.10` pins (csmcir
  `models.yaml:25`, intent `:306`, …) are therefore **unsatisfiable with torch 2.9** but satisfiable with
  2.7/2.8 — the constraint pair is not inherently contradictory.
- Those models declare no `adapted_runtime` (verified: only `limn` has one), so this is latent, not a live
  blocker. **Recommended action:** if a second model ever declares an adaptation, express the torch pin as
  a range the solver can intersect with `requires_python` (`torch>=2.7,<2.9`) rather than `>=2.7,<3`.
  Note the registry's `adapted_runtime` block carries no `python` key at all (`models.yaml:941-949`), so
  adaptation can never move the interpreter version — the default `python=3.10` is what makes it work.

STATIC_AUDIT.

---

### Q4. Installer pip steps — does the adapted wheel index reach `pip install`?

**Verdict at HEAD: BLOCKED (P0 for the adapted journey). Verdict at the hashes above: fixed by a
concurrent main-agent edit — P2 (residual risk only).**

- At HEAD, `git show HEAD:workbench/backend/environment_planner.py | grep -n "index"` returns no
  `--index-url`/`--extra-index-url` anywhere: `_install_at_prefix` built
  `pip_argv = [str(staging_interp), "-m", "pip", "install", *pip_specs]` (`HEAD:1424`) with no `-f`/`--index-url`,
  and `_pip_specs_for_group` emitted `torch<3,>=2.7` as a bare requirement
  (`environment_planner.py:1111-1130`). `pip install "torch>=2.7"` therefore resolves from **default PyPI**.
  Confirmed from primary source that the default PyPI torch build never ships sm_120 kernels: `torch>=2.7`
  from PyPI is the CUDA 12.6 build (`torch.version.cuda == "12.6"`, arch list sm_50…sm_90), so the probe below
  would fail later at the first kernel launch.
- At the hashes above the main agent added `pip_index_url: https://download.pytorch.org/whl/cu128` to
  `limn.environment.adapted_runtime` (`registry/models.yaml:947-949`, `pip_index_url` at `:949`), propagated it in
  `runtime.resolved_runtime_environment` (`runtime.py:817-819`), carried it through `model_constraints`
  (`environment_planner.py:511-513`) and `_constraints` (conflict check at `:568-577`), and appended
  `--index-url <index>` to the pip argv (`environment_planner.py:1496-1499`). Reproduced:
  `pip_index_url: https://download.pytorch.org/whl/cu128`,
  `pip specs: ['torch<3,>=2.7','torchvision<1,>=0.22','open-clip-torch>=2.24']`.
- **Residual risk 1 (P2, unresolved): `--index-url` *replaces* the index.** Reproduced: the pip argv contains
  `--index-url`, not `--extra-index-url`. `open-clip-torch` — declared in the same adapted block
  (`models.yaml:946`) — is **not** on that index:
  ```
  $ curl -sS -o /dev/null -w "%{http_code}\n" https://download.pytorch.org/whl/cu128/open-clip-torch/
  403
  $ curl -sS -o /dev/null -w "%{http_code}\n" https://download.pytorch.org/whl/cu128/torch/
  200
  ```
  The whole pip invocation is all-or-nothing, so this is one `pip install` that must fetch `torch`,
  `torchvision` **and** `open-clip-torch`; with `--index-url` alone the third fails and `_install_at_prefix`
  deletes the staging prefix and reports `pip install exited 1`. **Action:** use `--extra-index-url` for the
  torch index (torch's runtime deps were checked to exist there too — `sympy filelock typing-extensions
  networkx jinja2 fsspec setuptools` all HTTP 200 — so `--extra-index-url` keeps the resolution working while
  leaving PyPI as the base), or split the pip step into torch-first (torch index) and remainder (PyPI).
- **Residual risk 2 (P3):** if the pip step is skipped for an adapted run (`pip_specs` empty), the wheel
  index is silently dropped — cosmetic today, since the adapted profile always has pip specs.

STATIC_AUDIT (network fetch only for HTTP status and PyPI metadata).

---

### Q5. Runtime `verify_environment` — `_required_packages` vs `_adapted_packages` (P0-A/P0-B wave)

**Verdict: CONFIRMED-VARIANT (a real inconsistency survives the wave) — severity P1**

I did **not** re-derive P0-A/P0-B. What I found is a *third* path that disagrees with both of them:

- After the wave, the planner routes every model through `runtime.resolved_runtime_environment(model)`
  (`environment_planner.py:377`→`runtime.py:786`), which returns the historical block unchanged for a model
  that does not declare `adapted_runtime` (`runtime.py:795-798`).
- But `verify_environment` still selects its contract with the **raw registry** call
  `runtime.runtime_adaptation(model)` (`runtime.py:1257`), which **raises** `UNKNOWN_RUNTIME_ADAPTATION`
  for any model whose environment does not declare that exact target (`runtime.py:779-783`).
- Consequence, reproduced with `WORKBENCH_RUNTIME_ADAPTATION=sm_120` and a *configured* interpreter:

  ```
  $ WORKBENCH_RUNTIME_ADAPTATION=sm_120 WORKBENCH_CSMCIR_PYTHON=/usr/bin/python3.14 python3 -c ...
  verify RAISES WorkbenchError ('UNKNOWN_RUNTIME_ADAPTATION',
      "unknown WORKBENCH_RUNTIME_ADAPTATION='sm_120' for model 'csmcir': declared adaptation is none",
      {'model_id': 'csmcir', 'requested': 'sm_120', 'declared': None})
  environment_blockers -> []
  inspect -> PRESENT
  ```

  and at the digest/plan level (no interpreter needed):

  ```
  csmcir UC -> UNCONFIGURED
  csmcir digest RAISES WorkbenchError ('UNKNOWN_RUNTIME_ADAPTATION',
      "unknown WORKBENCH_RUNTIME_ADAPTATION='sm_120' for model 'csmcir': declared adaptation is none")
  encoder digest RAISES WorkbenchError (... model 'encoder' ...)
  limn   digest -> c7968165
  ```

  (Configured csmcir is the honest repro; unconfigured csmcir returns `UNCONFIGURED` at the top guard
  `runtime.py:1171-1197` before it ever reaches the digest.)
- Consumer blast radius, from the source: `_environment_contract_digest` (`runtime.py:1397`, called at
  `:1413`) → `_cache_key` (`:1536`), `_reuse`/`_remember` (`:1547`,`:1574`), `_interpreter_fingerprint`
  (`:1433`), `_gpu_capability_key` (`:1002`, calls the digest at `:1009`) → `record_gpu_capability`
  (`:935`) and `verify_gpu_runtime` (`:1796`), and `scripts/pipeline.py:899` which hashes the environment
  content into the run's execution identity. Every one of those raises for a non-adapted model while any
  adaptation is selected.
- Also inconsistent *within* the wave: `model_constraints` catches this exact error and keeps the
  historical contract for models that declare no adaptation (`environment_planner.py:379-395`), while
  `verify_environment` does not. And `environment_blockers` (`runtime.py:~1935`) returns `[]` for csmcir in
  that same process, so "blocked model" and "environment that raises on inspection" can coexist.
- Scope today: `WORKBENCH_RUNTIME_ADAPTATION` is auto-forwarded by the compose file
  (`workbench/docker/docker-compose.yml:30`, fed by `.env:39`) and is now also allowlisted into the
  evaluator child (`scripts/evaluate_models.py:446`, main-agent edit). Under the primary Blackwell
  configuration the variable is process-global, so **every** non-adaptive read of **every other model**
  hits this.
- **Recommended action (root cause, one place):** make `verify_environment` (and `_environment_contract_digest`)
  consume `resolved_runtime_environment(model)`'s decision instead of calling `runtime_adaptation` directly
  — i.e. select the adapted contract only when the model itself declares the matching target, and treat an
  unknown target on a non-declaring model the same way `model_constraints` already does. Do **not** patch
  each caller.

STATIC_AUDIT.

---

### Q6. GPU capability cache — does the key include the runtime profile?

**Verdict: PASS (no collision) — severity NONE**

- `_gpu_capability_key` (`runtime.py:1002-1020`) hashes `model_id`, the configured interpreter path, the
  interpreter fingerprint, `_environment_contract_digest(model)` and `_host_identity()`. The contract digest
  includes `"adapted_runtime": runtime_adaptation(model)` (`runtime.py:1413`), so a historical probe record
  and an adapted probe record cannot share a key (`_cached_gpu_capability`, `:1107`, requires
  `payload["capability_key"] == key`).
- Observed, in-process (interpreter not present, so the fingerprint term is `None` — the contract term is
  what differs):
  ```
  None   raw=cc876e27c19c123b spec-model=0286e7fedd8df59d DIFFERENT
  sm_120 raw=c79681651245aab7 spec-model=61ceaa98862403cd DIFFERENT
  ```
  The `raw` vs `spec-model` difference is expected (the planner's model carries the resolved dependency
  set) and is a *fourth* identity for the same physical environment — see B-1.
- The adapted probe additionally verifies the arch list and the device capability
  (`runtime.py:1893-1907`: `target not in arch_list` → `capability=False` with `adaptation_mismatch`
  evidence; `device_cap != target` likewise). Keys consumed by that check
  (`torch_arch_list`, `compute_capability`, `cuda_available`, `allocated_tensor`,
  `torch_has_compiled_core`, `torch_module_file`, `torch_core_origin`) all exist in `GPU_PROBE_CODE`
  (`runtime.py:1642-1672`).
- Stored-record hygiene, observed on the artifact tree: `workbench/artifacts/gpu_capabilities/` holds 1564
  records; `python3` count by `probe_source` → `{'override': 1171, 'interpreter': 293, 'forged': 99,
  'unreadable': 1}`; **0** records carry `runtime_adaptation=sm_120` (consistent with adaptation never
  having been provisioned). Synthetic `override` records are refused outside
  `WORKBENCH_ALLOW_SYNTHETIC_GPU_PROBE=1` (`runtime.py:1145-1150`), and there is no "real adaptation
  capability" claim anywhere in the tree.

STATIC_AUDIT.

---

### Q7. Checkpoint deserialization — torch>=2.7 loading torch-1.12-era whole-model pickles

**Verdict: VARIANT (mechanism sound, one load-bearing route unverified) — severity P1**

- The workbench's own load path is explicit and correct: `load_trusted_whole_model`
  (`workbench/replay/common.py:36-48`) requires a non-`None` author SHA-256, hashes the bytes, and only then
  calls `torch.load(path, map_location="cpu", weights_only=False)` when the parameter exists
  (`common.py:46-47`) — i.e. it does not rely on any torch default. Under torch 2.6+/2.7 the default flip to
  `weights_only=True` would otherwise break whole-model unpickling; this path opts out deliberately and
  documents that the allowance is gated on matching trusted author bytes. **STATIC_AUDIT: PASS.**
- The load-bearing route, however, is `replay_category` (`workbench/replay/limn.py:136`), not the workbench
  helper: `model = load_trusted_whole_model(...)`, then `model.cuda().eval()`, then
  `open_clip.image_transform(visual.image_size, is_train=..., mean=visual.image_mean, std=visual.image_std)`
  (`limn.py:169-176`), then `native_test.test(...)`.
- Under the adapted profile (torch 2.7 / open_clip >=2.24) this code is reading attributes off an object that
  was unpickled from **torch 1.12 / open_clip 2.20-era pickled state**. `image_mean`/`image_std` exist on
  `model.visual` in 2.20 (`factory.py:247-248` sets them) and modern open_clip still sets them for backward
  compatibility (`open_clip v3.3.0 factory.py:876-879 set_model_preprocess_cfg`: `# legacy attribute, keeping
  for bwd compat`), and `image_transform`'s first four parameters are signature-compatible through v3.3.0.
  So the *access* holds — but whether the unpickled 1.12-era `nn.Module` graph survives torch 2.7's
  `_load_from_state_dict`/`__setstate__` machinery was **not run**, and I cannot run GPU inference.
- **Recommended action:** treat adapted-LIMN checkpoint loading as an explicitly owed runtime check
  (`model.cuda()` on all three `0_*_best_model.pt`), and record the outcome in the run's evidence; do not
  present an adapted LIMN number as a reproduction until it is.

STATIC_AUDIT; the deserialization itself is **UNKNOWN** (never executed).

---

### Q8. LIMN replay imports under the adapted runtime (open_clip >=2.24 API drift vs 2.20-based code)

**Verdict: VARIANT, low severity for the pinned call site (open_clip is only used in one place) — severity P2**

- Pinned LIMN source: `open_clip` is imported at `workbench/third_party/LIMN/LIMN/model.py:5`; the model
  factory call is `model.py:15` (`create_model_and_transforms('ViT-L-14', pretrained=os.path.join('[CLIP path]/
  laionCLIP-ViT-L-14-DataComp.XL-s13B-b90K','open_clip_pytorch_model.bin'))`) — that call is inside the
  `Backbone.__init__` path, which **is not** re-executed when `load_trusted_whole_model` unpickles the whole
  model. Plus `model.py:17` (`open_clip.get_tokenizer('ViT-L-14')`) and `train.py:78`. The replay's only
  open_clip use is `limn.py:172-176`.
- API verified across versions (primary source, `raw.githubusercontent.com/mlfoundations/open_clip`):
  `image_transform(image_size, is_train, mean=None, std=None, ...)` exists in v2.20.0 (`transform.py:56-64`),
  v2.24.0 (`:274-283`), v2.26.1 (`:274-283`) and v3.3.0 (`:324-333`) — **no positional or keyword break**.
  v2.24/v2.26 add `resize_mode`/`interpolation` *before* `fill_color` and remove `resize_longest_max`, but the
  replay passes nothing after `std`, so it is unaffected; the resize default is shortest-edge bicubic in both.
- The declared floor `open-clip-torch>=2.24` is also open-ended: PyPI latest is 3.3.0
  (`requires_python >=3.9`), so an unpinned install gets 3.3.0, not 2.24. Verified the call site and the
  `image_mean`/`image_std` attributes still hold there (above). Residual risk is behavioural drift inside the
  transform (interpolation defaults were equal for the arguments actually passed) — not an import failure.
- **Recommended action:** pin the adapted block to a tested minor (`open-clip-torch>=2.24,<2.27`) instead of
  `>=2.24`, so the transform identity is a declared fact rather than an open range.

STATIC_AUDIT.

---

### Q9. CSMCIR environment contract (adaptation selected in the same process)

**Verdict: BLOCKED (cross-model crash) — severity P1**

Covered in Q5; here is the *facet this slice owns* plus one more:

- The crash is not csmcir-specific: it is any model without a matching `adapted_runtime`. Reproduced for
  `csmcir` and `encoder` (Q5 output). `limn` is the only model that resolves.
- **Extra facet — CSMCIR declares `cuda: UNKNOWN` with a hard CUDA dependency.** `registry/models.yaml:28`
  (`cuda: UNKNOWN`) and `environment_required` is **unset** for csmcir (verified by parsing the registry:
  `csmcir env_required= None cuda= UNKNOWN`; only clvc_net/dcnet/combiner_rn50x4_noft/clip4cir_rn50x4_fullft/
  tgcir/sprc/limn carry `environment_required: true`). `_requires_gpu` returns **False** for `cuda: UNKNOWN`
  (`runtime.py:893-903`), and `environment_blockers` returns `[]` when `environment_required` is falsy
  (`runtime.py:1955-1960`). Meanwhile the pinned evaluator hard-selects `cuda` whenever it is available:
  `workbench/third_party/CSMCIR/src/utils_csmcir.py:13-16`
  (`if torch.cuda.is_available(): device = torch.device("cuda") else: ... "cpu"`), and
  `validate_blip_csmcir.py:448` (`clip.tokenize(...).to(device)`), `:522`, `:524`
  (`torch.load(blip_model_path, map_location=device)`). So CSMCIR can be launched with **no GPU capability
  gate at all** and will silently run on `cuda` if the box happens to have one. This is the same shape as the
  P0-A/P0-B family but on the *declaration* side, and P0-B's allowlist fix does not touch it.
- **Recommended action:** either declare `cuda` for CSMCIR (so the gate applies) or make the evaluator's
  device selection an explicit, recorded contract; do not leave "UNKNOWN + hard `.to(device)`" as the
  CSMCIR environment contract.

STATIC_AUDIT.

---

### Q10. Python version bounds conflict (registry `python<3.10` vs adapted torch `>=2.7` needing `>=3.9`)

**Verdict: PASS for `limn`, latent for others — severity P3.** See Q3. `limn` has `python: UNKNOWN`; the
planner's `python=3.10` default (`environment_planner.py:1088-1096`) is the only thing keeping the adapted
solve feasible. Nothing in the code checks that the default is compatible with the adapted torch range.

STATIC_AUDIT.

---

### Q11. `variant_provenance` — is the whole chain marked `ADAPTED_RUNTIME` and never presented as an exact reproduction?

**Verdict: VARIANT — the run-report chain is correctly marked; two label-level gaps — severity P2**

Marked correctly (traced end-to-end):
1. `_variant_provenance(model, config)` (`workbench/scripts/evaluate_models.py:262`) sets
   `adapted_runtime = {"runtime_adaptation": "sm_120", "pytorch": ">=2.7,<3", "torchvision": ">=0.22,<1",
   "cuda": "12.8", "packages": [...], "note": "ADAPTED_RUNTIME: ... not the authors' historical environment,
   not an exact reproduction claim"}` (`:296-309`), and that dict is attached to the plan
   (`:398 variant_provenance=_variant_provenance(...)`).
2. Emitted into the run report: `"variant_provenance": getattr(plan, "variant_provenance", None) or {}`
   (`evaluate_models.py:1472`).
3. Consumed as a comparability fact: `run_meta["declared_variant"] = bool(variant.get("approved_source_patch"))
   or bool(variant.get("adapted_runtime"))` (`evaluate_models.py:162`), and `VARIANT_KEYS = ("caption_alias_staged",
   "declared_variant")` (`backend/paper_gate.py:83`) → `variant = any(...)` (`paper_gate.py:397`) →
   `PASS_WITHIN_5_PERCENT_VARIANT` instead of `PASS_WITHIN_5_PERCENT` (`paper_gate.py:409`).
4. API/UI passthrough: `backend/main.py:159-160` and `backend/autopilot.py:848,965`.

Gaps:
- **G1 (P2) — the canonical schema-v2 result cannot carry the marker at all.** `RunMetadata` is
  `model_config = ConfigDict(extra="forbid")` (`workbench/backend/schemas/results.py:51`) and has no
  variant/adaptation field (fields listed at `:50-84`). Any run that writes a schema-v2 result **must drop**
  the adapted runtime, because adding it would raise a validation error. The index deliberately ignores
  aggregate reports (`backend/index.py:47` skips `official_aggregate_evaluation_report`), so the *indexed*
  research artifact is exactly the one that cannot say "adapted".
  **Action:** add an optional `variant_provenance` (or `runtime_adaptation`) field to `RunMetadata` and carry
  it through `rebuild_index`.
- **G2 (P3) — the readiness label conflates the adapted runtime with a source variant.**
  `pc_readiness.model_section` reclassifies only for an applied source patch
  (`scripts/pc_readiness.py:246-254`), while the adapted runtime is only *reported* as a separate field
  (`environment_section`, `:157`). So a host with `WORKBENCH_RUNTIME_ADAPTATION=sm_120` and a verified adapted
  environment can still print `READY_FOR_FULL_EVALUATION` — a phrase that reads as "exact reproduction"
  (`READY_WITH_DECLARED_VARIANT` is reserved for the patch case).
  **Action:** give the adapted runtime the same treatment as the approved patch in `pc_readiness`.
- **G3 (P3) — the run report's environment block does not name the adaptation.**
  `evaluate_models.py:499-509` builds the evaluator identity with `orchestrator_*`, `model_interpreter`,
  `evaluation_run_id`, `evaluation_output_root` only; the forward of `WORKBENCH_RUNTIME_ADAPTATION` into the
  child (main-agent edit at `:440-443`) makes the model's own preflight see it, but the recorded environment
  dict the report persists does not.

STATIC_AUDIT.

---

### Q12. LIMN replay: `validate_environment` accepts both contracts (historical + adapted)

**Verdict: VARIANT — correctness is sound, contract sharing is not — severity P3**

- `replay/limn.py:84-127` resolves the adaptation through the registry block it fetches itself
  (`_registry_adapted_runtime`, `:77-82`) and checks `torch`/`torchvision`/`open_clip_torch` against the
  selected contract, then requires `torch.cuda.is_available()`. An unknown target falls back to the
  historical pins (`except Exception: adaptation = None`, `:96-97`). That is fail-safe and does **not**
  widen the contract.
- The package set it checks is the adapted block's own (verified in-process:
  `registry-path adapted required: {'torch': '>=2.7,<3', 'torchvision': '>=0.22,<1', 'open_clip_torch': '>=2.24'}`),
  not the planner's.
- **B-1 (P2) — three names for one contract.** In one adapted process, the *same* LIMN environment yields:
  the planner's `_runtime_model` package list (normalized `open-clip-torch` with no constraint),
  `runtime.verify_environment`'s set (raw registry packages `open-clip-torch==2.20.0` + adapted ranges),
  and `replay/limn.py`'s set (`open_clip_torch>=2.24`). Reproduced side by side:
  ```
  planner-path _runtime_model packages: ['numpy==1.21.5','packaging==23.1','pillow==9.4.0','tqdm==4.65.0',
                                        'pyyaml','open-clip-torch','open-clip-torch']
  planner-path required: {'torch':'>=2.7,<3','torchvision':'>=0.22,<1','numpy':'==1.21.5','packaging':'==23.1',
                          'pillow':'==9.4.0','tqdm':'==4.65.0','pyyaml':None,'open-clip-torch':None}
  registry-path adapted required: {'torch':'>=2.7,<3','torchvision':'>=0.22,<1','open_clip_torch':'>=2.24'}
  ```
  The first two derive from the malformed string `"open-clip-torch"` produced by
  `models.yaml:946 packages: [open_clip_torch>=2.24]` — the declaration block does not normalize PEP-503
  separators, so the name is a `_`-form while `model_constraints`' `_package_name` normalizes to `-`-form,
  and `_add_package` splits on `==`/`>=` only in `model_constraints`, not in `_adapted_packages`
  (which uses `re.split(r"[<>=!~]", ...)`, hence the `>=2.24` range works there). Net effect:
  `open-clip-torch` is verified **unversioned** (`None`) on the planner/verify path — i.e. the `>=2.24`
  floor is dropped exactly where adaptation matters — while `_add_package` saw `>=2.24` in `ranges`.
  **Action:** normalize package names at the `adapted_runtime` parse boundary (one place:
  `_adapted_packages` and `_add_package` should share one parser), and re-check that the verifier enforces
  `open-clip-torch>=2.24`.
- `replay_category` ignores `--category`-derived partial completeness in the aggregate (`evaluate_categories`
  refuses duplicates/unknowns, `limn.py:188-193`; macro only when all three complete, `replay/common.py:
  macro_aggregate`). Not part of this slice; noted only because the adapted run depends on it.

STATIC_AUDIT.

---

### Q13. `_runtime_model` drops the adapted pip index and the adaptation marker

**Verdict: CONFIRMED — severity P2**

- `_runtime_model` (`environment_planner.py:797-816`) rebuilds `environment["packages"]` from
  `spec["packages"]` (+ names from `spec["ranges"]`) and copies `pytorch`/`torchvision`/`python` — but never
  re-attaches `spec["pip_index_url"]` (set at `:510-512`) nor the adapted `packages` declarations from the
  registry block.
- Consequence: the top-level `_verify_group_interpreter` path (`:1144`) and the `_reusable_interpreter`
  path (`:733`, which builds its model via `_runtime_model`, `:679`) verify a model that (a) sees the
  *resolved* torch/torchvision fields, and (b) surfaces the planner's clone-derived package list with
  duplicate names. This is the same family as P0-A (one contract consumed by two different shapes), and the
  main-agent wave did not close this edge.
- **Action:** have `_runtime_model` carry `runtime_adaptation`/`pip_index_url` through (or, better, make it
  call `runtime.resolved_runtime_environment` so there is literally one resolver), so the verifier's
  `expected packages` string equals the installer's.

STATIC_AUDIT.

---

### Q14. `_gpu_capability_key` consistency at an interpreter context

**Verdict: UNKNOWN (cannot be exercised here) — severity P2**

- `_gpu_capability_key(model)` computes the fingerprint with `str(interpreter.parent.parent)`
  (`runtime.py:1005-1010`), while `_remember` computes it with `report.prefix` (`runtime.py:1578-1579`,
  `report.prefix` = the invocation-path-derived prefix from `_interpreter_identity`, `:174`).
  `_cached_gpu_capability` compares `cached["interpreter_fingerprint"] != report.environment_fingerprint`
  (`:1152-1154`).
- Wrapper vs real interpreter would read the same *invocation* path, and for the provisioned layout
  `/artifacts/environments/cir-…/bin/python` the two coincide — which is why the prior audit reports a
  cached-capability hit. I could not exercise a case where they differ without a real environment plus a
  probe, so I record this as **UNKNOWN**, not a defect.
- **Action:** make both call sites use `report.prefix` (or one helper), which makes the invariant explicit
  and removes the need for this reasoning.

STATIC_AUDIT; **UNKNOWN**.

---

### Q15. Whole-model checkpoints exist but were never exercised

**Verdict: NOT_RUN (as designed) — severity P3, documentation**

- Observed on disk (`ls -l workbench/artifacts/checkpoints/limn/`):
  `0_dress_best_model.pt`, `0_shirt_best_model.pt`, `0_toptee_best_model.pt`, each 1 792 010 157 bytes
  (matches `expected_size_bytes: 1792010157` in `registry/models.yaml:957,966,975`). The environment on disk
  is the **historical** one (`torch-1.12.1.dist-info`, `open_clip_torch-2.20.0.dist-info`,
  `numpy-1.21.5.dist-info` under `environments/cir-1bbd9872cf166f0d/lib/python3.10/site-packages`).
- `registry/models.yaml:926` says `reproduction_status: BLOCKED_RUNTIME_PREREQUISITES`, which is consistent:
  the adapted runtime has never been provisioned and no `runtime_adaptation=sm_120` GPU capability record
  exists (0 of 1564 records). Nothing in the tree claims an adapted LIMN number.
- **Action:** none required for honesty; the report should simply not be read as "adapted path exercised".

STATIC_AUDIT.

---

## 2. Findings ledger

| ID | Severity | Class | One-line root cause | File:line |
|----|----------|-------|---------------------|-----------|
| **B-1** | P2 | CONFIRMED | Three different resolved contracts (`_runtime_model`, `verify_environment`, `replay/limn`) for one environment; `_runtime_model` drops `pip_index_url`/adaptation marker and emits a duplicate, unversioned `open-clip-torch` | `environment_planner.py:797-816`, `runtime.py:1257`, `replay/limn.py:84-127` |
| **B-2** | P1 | CONFIRMED | `verify_environment` + `_environment_contract_digest` call `runtime_adaptation` on the **raw** registry model; any non-adaptive model **raises `UNKNOWN_RUNTIME_ADAPTATION`** when an adaptation is selected process-globally (compose auto-forwards it) | `runtime.py:1257`, `:1413`, `:779-783`; `docker-compose.yml:30` |
| **B-3** | P1→P2 | CONFIRMED (fix landed mid-audit) | pip installer sent the adapted profile to **default PyPI** (no `-f`/`--index-url`); PyPI torch has no sm_120 kernels. Now fixed, but `--index-url` (not `--extra-index-url`) excludes `open-clip-torch`, which is **403** on the cu128 index — same pip invocation fails | `environment_planner.py:1496-1499`; `registry/models.yaml:947-949`; curl 403/200 |
| **B-4** | P1 | CONFIRMED | CSMCIR declares `cuda: UNKNOWN` + `environment_required` unset while its evaluator does `if torch.cuda.is_available(): device=cuda` → runs on CUDA with **no GPU capability gate** | `models.yaml:28`; `third_party/CSMCIR/src/utils_csmcir.py:13-16` |
| **B-5** | P2 | CONFIRMED | `_environment_id` hashes only `members`+constraint tuples — the adaptation target is not part of the environment identity; no structural guarantee two profiles cannot collide | `environment_planner.py:774-781` |
| **B-6** | P2 | CONFIRMED | `_adapted_packages` does not normalize package names, so the adapted `open-clip-torch>=2.24` floor is verified as `None` (unversioned) on the planner/verify path | `runtime.py:827-843`, `models.yaml:946` |
| **B-7** | P2 | CONFIRMED | Adapted `RunMetadata` is `extra="forbid"` with no variant field, so the schema-v2 (indexed) artifact cannot carry `ADAPTED_RUNTIME`; the aggregate report that does carry it is skipped by the index | `schemas/results.py:51`; `index.py:47` |
| **B-8** | P2 | CONFIRMED | `open_clip` adapted floor is open-ended (`>=2.24`, latest 3.3.0); the pinned call site is signature-compatible through 3.3.0 (verified), so this is a transform-identity risk, not an import break | `models.yaml:946`; `replay/limn.py:170-172` |
| **B-9** | P2 | CONFIRMED | `_TORCHVISION_PAIRS` ended at torch 2.6.0 → the adapted profile had **no binary pairing** and `plan_environments` blocked it (fixed mid-audit by adding 2.7/2.8/2.9 pairs) | `environment_planner.py:171-190` (fix at `:186-188`) |
| **B-10** | P2 | CONFIRMED | Adapted whole-model deserialization (torch-1.12 pickle → torch 2.7) was never executed; `model.cuda()` on a 1.12-era `nn.Module` graph is the unverified link | `replay/limn.py:169-172`; `replay/common.py:36-48` |
| **B-11** | P2 | UNKNOWN | `_gpu_capability_key` uses `interpreter.parent.parent` where `_remember` uses `report.prefix`; equality for a wrapper interpreter was not exercisable | `runtime.py:1005-1010`, `:1578-1579`, `:1152` |
| **B-12** | P3 | CONFIRMED | Registry comment still advertises `sm_120_blackwell` (doc says SCI-4 fixed this) although every code path uses `sm_120` | `runtime.py:765-766` |
| **B-13** | P3 | CONFIRMED | `pc_readiness` labels the adapted runtime only as a reported field, not as `READY_WITH_DECLARED_VARIANT` → a verified adapted host can print `READY_FOR_FULL_EVALUATION` | `pc_readiness.py:157`, `:246-254` |
| **B-14** | P3 | CONFIRMED | Run report's recorded `environment` block (the evaluator identity) never names the runtime adaptation | `evaluate_models.py:499-509` |
| **B-15** | P3 | CONFIRMED | Group evidence (`group["evidence"]`) carries the "adapted runtime selected" sentence added by `model_constraints`, but the group dict has no `runtime_adaptation`/`pip_index_url` field of its own | `environment_planner.py:511`, `:1313`, `:1494-1501` |
| **B-16** | P3 | CONFIRMED | python bound vs adapted torch is unchecked: `python: UNKNOWN` → planner default `python=3.10` is the only thing making `torch>=2.7,<3` feasible; torch 2.9 needs `>=3.10` while several registry entries pin `<3.10` | `environment_planner.py:1088-1096`; `models.yaml:25,306`; PyPI metadata |

Positive, independently re-verified: `load_trusted_whole_model`'s `weights_only=False` is gated on a matching
author SHA-256 (`replay/common.py:36-48`); the GPU cache key *does* include the runtime profile
(`runtime.py:1413` → `:1002-1020`); the adapted GPU probe enforces arch-list **and** device capability
(`runtime.py:1893-1907`); synthetic capability records are refused outside the explicit test switch
(`runtime.py:1145-1150`); the planner's conda solve never receives torch/torchvision/open-clip
(`_NON_CONDA_PACKAGES`, `:601-608`) and the adapted conda spec set is `python=3.10 + pip + 5 conda-resolvable
pins` (reproduced); `variant_provenance` is threaded plan → run report → `paper_gate` →
`PASS_WITHIN_5_PERCENT_VARIANT` → API/UI.

---

## 3. Execution trace

All commands read-only; throwaway output under `/tmp`. `docker` was never invoked. `md5sum` above fixes the
revision.

1. `git log --oneline -3`, `git status --short` → HEAD `e9cafaf`; working tree modified (main agent in flight).
2. `git show HEAD:workbench/backend/environment_planner.py | grep -n "index"` → no `--index-url` anywhere;
   HEAD `_TORCHVISION_PAIRS` ends at `("2.6.0","0.21.0")`; HEAD runtime has no `pip_index`/`index_url`.
3. `git diff` (runtime/planner/evaluate_models/models.yaml/run.sh) → the in-flight P0-A/P0-B wave, including
   `resolved_runtime_environment`, the pair-table extension, `pip_index_url` plumbing and the
   `WORKBENCH_RUNTIME_ADAPTATION` allowlist entry. Re-read after further edits.
4. `python3 - <<EOF` (in-process, injected solver, `_SOLVE_CACHE.clear()`): historical vs adapted
   `plan_environments([limn])` → distinct env ids; no collision. `environment_id` is not profile-keyed.
5. Same harness with the pair table truncated to HEAD content → adapted plan **blocked**
   ("no official binary pairing"); with the current tree → group forms.
6. Same harness on the current tree → `conda specs` / `pip specs` / `pip_index_url` / `_runtime_model`
   packages captured. `pip_index_url` = cu128; `_runtime_model` shows the duplicate unversioned
   `open-clip-torch`.
7. `python3 -c` with `WORKBENCH_RUNTIME_ADAPTATION=sm_120` (+ `WORKBENCH_CSMCIR_PYTHON=/usr/bin/python3.14`) →
   `verify_environment` and `_environment_contract_digest` **raise** `UNKNOWN_RUNTIME_ADAPTATION` for csmcir;
   `environment_blockers` → `[]`; `inspect_environment` → `PRESENT`. Same for encoder.
8. `python3 -c` comparing contract digests → historical `cc876e27…` / adapted `c7968165…` for the raw registry
   model vs `0286e7fe…` / `61ceaa98…` for the planner-shaped model → four identities for one environment.
9. `curl -sS https://download.pytorch.org/whl/cu128/{open-clip-torch,torch,...}` → `403` / `200`;
   `curl` on `sympy filelock typing-extensions networkx jinja2 fsspec setuptools` → all `200`.
10. `curl` `raw.githubusercontent.com/mlfoundations/open_clip/{v2.20.0,v2.24.0,v2.26.1,v3.3.0}/src/open_clip/
    {transform.py,factory.py,model.py}` → `image_transform` signature compatibility and the
    `image_mean`/`image_std` legacy attributes.
11. `curl https://pypi.org/pypi/{torch,open-clip-torch}/json` → `requires_python` bounds; open-clip-torch
    latest `3.3.0`.
12. `python3 -c` parsing `artifacts/gpu_capabilities/*.json` → 1564 records,
    `{'override':1171,'interpreter':293,'forged':99,'unreadable':1}`, **0** with `runtime_adaptation`.
13. `ls -l artifacts/checkpoints/limn/`, `cat .../torch-1.12.1.dist-info/METADATA` → checkpoints present,
    historical environment provisioned, adapted environment not provisioned.

Verification **not** performed and therefore not claimed: any GPU execution, any docker build/compose action,
`pytest`, `release_verify.sh`, an actual conda/micromamba solve, and deserialization of the LIMN checkpoints.

---

## 4. Final verdict

**VARIANT.**

The adapted-runtime (Blackwell sm_120) contract is *declared and provenance-marked honestly end-to-end, but
most of its execution path has never run, and the pieces that would run it are still inconsistent*. The
runtime-profile identity is correctly excluded from capability reuse (Q6) and the `weights_only` decision is
correctly gated (Q7), but:

- selecting the adaptation process-globally **breaks every other model's environment verification** with a
  raised `UNKNOWN_RUNTIME_ADAPTATION` (B-2) — and the compose file forwards that variable by default (P1);
- the adapted pip install is aimed at an index that cannot serve all of its own requirements (B-3) (P1);
- CSMCIR can be launched onto CUDA with no GPU gate at all (B-4) (P1);
- one logical environment still has three different "expected package" definitions and the adapted
  `open-clip-torch` floor is verified as unversioned (B-1, B-6) (P2);
- and the indexed schema-v2 artifact structurally cannot carry the `ADAPTED_RUNTIME` marker (B-7) (P2).

None of this is a fabricated claim: `registry/models.yaml:926` records LIMN as
`BLOCKED_RUNTIME_PREREQUISITES`, no `sm_120` capability record exists, and no adapted number appears in the
tree. So the honest verdict for the reproduction path itself is **NOT_RUN on the adapted axis, VARIANT on the
historical axis** — with B-2/B-3/B-4 as the items that must land before an adapted run could even start.
