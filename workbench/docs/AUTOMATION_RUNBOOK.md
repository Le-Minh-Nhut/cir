# CIR Failure Analysis Workbench — Automation Runbook

Operator guide for running, resuming, troubleshooting, and verifying CIR model reproduction workflows.

## Principles

1. **Research integrity first**: Never fake model reproduction. Never report a noisy checkpoint as clean.
2. **Explicit authorization**: Environment installations and preprocessing require dedicated `--allow-*` flags.
3. **Fail closed**: Missing provenance, mismatched checksums, and unverified mappings block execution.
4. **Resumable pipelines**: Stage states and input fingerprints are persisted in `workbench/artifacts/pipeline_state.json`.

---

## 1. Master Pipeline Commands

The master orchestrator is `workbench/scripts/pipeline.py` (or `workbench/run.sh`).

### A. One-Command Complete Workflow (`all`)
Executes full reproduction pipeline for selected model: preflight doctor, dataset check, source sync, dataset link, checkpoint acquisition, auxiliary asset download, runtime preflight, evaluation, schema validation, and derived index rebuild.

```bash
# Planning mode (default for all, setup, reproduce without --apply):
python workbench/scripts/pipeline.py all --model csmcir

# Explicit dry run (zero mutations, prints stage plans):
python workbench/scripts/pipeline.py all --model csmcir --dry-run

# Authorized complete execution with resume capability:
python workbench/scripts/pipeline.py all --model csmcir --apply \
  --allow-network --allow-large-downloads --allow-env-install \
  --allow-preparation --allow-gpu-eval --resume
```

### B. Master Setup Workflow (`setup`)
Executes preparation stages: doctor, dataset, sync, environment, preparation, checkpoint, auxiliary-assets.

```bash
# Planning mode:
python workbench/scripts/pipeline.py setup --model csmcir

# Authorized execution with resume capability:
python workbench/scripts/pipeline.py setup --model csmcir --apply \
  --allow-network --allow-large-downloads --allow-env-install \
  --allow-preparation --resume
```
### C. Master Reproduction Workflow (`reproduce`)
Runs runtime preflight and guarded evaluation.

```bash
# Planning mode:
python workbench/scripts/pipeline.py reproduce --model csmcir

# Authorized evaluation with resume:
python workbench/scripts/pipeline.py reproduce --model csmcir --apply --allow-gpu-eval --resume
```
### D. Master Analysis Workflow (`analyze`)
Validates result artifacts and rebuilds DuckDB index.

```bash
python workbench/scripts/pipeline.py analyze
```

### E. Master Report Workflow (`report`)
Generates machine-readable JSON status of all models and prerequisites.

```bash
python workbench/scripts/pipeline.py report --model csmcir
python workbench/scripts/pipeline.py report
```

---

## 2. Standalone Stage Operator Commands

Every workflow stage has a working standalone CLI entrypoint:

| Stage | Script | Purpose |
|---|---|---|
| Doctor / Preflight | `workbench/scripts/doctor.py` | Inspect local prerequisites without mutations |
| Environment Manager | `workbench/scripts/manage_environment.py` | Inspect or create isolated model environments |
| Model Preparation | `workbench/scripts/prepare_model.py` | Inspect or execute model-specific preparation |
| Dataset Layout | `workbench/scripts/prepare_dataset.py` | Validate standard FashionIQ & prepare CSMCIR link |
| Source Sync | `workbench/scripts/sync_upstreams.py` | Clone or pin upstream git repositories |
| Checkpoint Download | `workbench/scripts/download_checkpoints.py` | Acquire registry model checkpoints |
| Auxiliary Assets | `workbench/scripts/download_auxiliary_assets.py` | Acquire verified auxiliary files (e.g. Qwen captions) |
| Model Evaluation | `workbench/scripts/evaluate_models.py` | Guard and execute official evaluator |
| Result Validation | `workbench/scripts/validate_results.py` | Validate canonical schema-v2 JSON artifacts |
| Index Rebuilder | `workbench/scripts/rebuild_index.py` | Rebuild DuckDB index from canonical JSON |
| Workbench UI Server | `workbench/scripts/serve_workbench.py` | Launch FastAPI backend and Vite frontend |
| Artifact Cleanup | `workbench/scripts/clean_generated.py` | Remove derived mock/index/log artifacts safely |

### Standalone Examples:

```bash
# Doctor inspection
python workbench/scripts/doctor.py --model csmcir --json
python workbench/scripts/doctor.py --all

# Environment management
python workbench/scripts/manage_environment.py --list
python workbench/scripts/manage_environment.py --model csmcir
# Creation requires explicit authorization:
python workbench/scripts/manage_environment.py --model csmcir --create --allow-env-install --dry-run

# Model preparation
python workbench/scripts/prepare_model.py --list
python workbench/scripts/prepare_model.py --model csmcir
# Execution requires explicit authorization:
python workbench/scripts/prepare_model.py --model csmcir --execute --allow-preparation --dry-run

# Checkpoints
python workbench/scripts/download_checkpoints.py --list
python workbench/scripts/download_checkpoints.py --model csmcir --dry-run
python workbench/scripts/download_checkpoints.py --model csmcir --verify-only

# Auxiliary assets
python workbench/scripts/download_auxiliary_assets.py --list
python workbench/scripts/download_auxiliary_assets.py --model csmcir --dry-run
python workbench/scripts/download_auxiliary_assets.py --model csmcir --verify-only

# Upstream source sync
python workbench/scripts/sync_upstreams.py --list
python workbench/scripts/sync_upstreams.py --model csmcir --verify-only

# Evaluation dry-run
python workbench/scripts/evaluate_models.py --model csmcir --dry-run
```

---

## 3. Resumability and Failure Recovery

Stage states are tracked in `workbench/artifacts/pipeline_state.json`.

- **Resume**: `--resume` skips stages that completed successfully with identical input fingerprints.
- **Force specific stage**: `--force-stage <STAGE_NAME>` forces re-execution of a stage even when marked complete.
- **Fingerprint invalidation**: If stage arguments, commands, or inputs change, the fingerprint mismatch automatically triggers re-execution.
- **Atomic state writes**: Stage transitions (PENDING → RUNNING → COMPLETE / FAILED) are written via temporary file replacement (`.tmp` → `pipeline_state.json`), preventing corrupted states on process interruption.
- **Failure isolation**: A failed stage preserves records of previously completed stages.

---

## 4. Current Model Reproduction Readiness

| Model | Status | Primary Blocker |
|---|---|---|
| **CSMCIR** | Setup Ready / Evaluation Guarded | COT_ours2 captions require manual author placement |
| **Air-Know** | Blocked | Only noisy (50%, 80%) checkpoints published; evaluator unaudited |
| **ConeSep** | Blocked | Only noisy (20%, 50%, 80%) checkpoints published; evaluator unaudited |
| **HABIT** | Blocked | Only noisy (20%, 50%, 80%) checkpoints published; evaluator unaudited |
| **INTENT** | Blocked | Only noisy (20%, 50%, 80%) checkpoints published; evaluator unaudited |
| **HINT** | Blocked | Pinned test.py CLI not audited |
| **ENCODER** | Blocked | Google Drive checkpoint unresolved; missing datasets1.py in upstream |
| **PAIR** | Blocked | pair-B1.pt vs pair-B2.pt checkpoint mapping unverified |
| **PTHA + MTST**| Blocked | No verified FashionIQ checkpoint published |
| **CLVC-Net** | Blocked | Pinned test.py has no standalone CLI loader; ResNet50 weights unverified |
| **DCNet** | Command Audited | Google Drive run directory hash unverified; ResNet50 weights unverified |
| **Combiner (noft)**| Blocked | Drive checkpoint mapping unresolved; CLIP RN50x4 unverified |
| **CLIP4Cir (fullft)**| Blocked | Paired checkpoint bundle unresolved; CLIP RN50x4 unverified |
| **TG-CIR** | Blocked | Pinned test.py has no standalone CLI; checkpoint mapping unverified |
| **SPRC** | Blocked | sprc_fiq.pt backbone mapping unverified; LAVIS asset unverified |
| **LIMN** | Replay Code Ready | Three category models required; CUDA 12.4 / OpenCLIP env unverified |

---

## 5. Distinction Between Artifact Tiers

1. **Mock Artifacts (`data_kind: "mock"`)**:
   Deterministic fixture runs generated for UI testing. **Never research evidence.**
2. **Official Aggregate Evaluation Reports (`aggregate_report.json`)**:
   Recorded in `workbench/artifacts/logs/<timestamp>_<model>_<checkpoint>/`.
   Contains verbatim `stdout.log`, `stderr.log`, `command.json`, paper sanity reference metrics, and `per_query_export_available: false`.
   **Does not constitute reproduced per-query predictions.**
3. **Canonical Schema-v2 Results (`workbench/artifacts/results/`)**:
   Strict JSON containing per-query retrievals, target ranks, and verified parity metrics. Ingested into DuckDB only after validation.
---

## 6. Aggregate Metric Extraction and Bundle Provenance

### A. Model-Specific Parsers (`workbench/backend/metrics_extraction.py`)
- **LIMN (`limn_structured_json`)**:
  Extracts macro R@10, R@50, mean and per-category metrics from structured stdout JSON. Requires all three categories (`dress`, `shirt`, `toptee`) to be complete; arithmetic category macro mean is validated.
- **CSMCIR (`csmcir_stdout_json`)**:
  Extracts `average_recall_at10`, `average_recall_at50`, `average_recall` and per-category recalls from official `validate_blip_csmcir.py` stdout JSON.
- **Unaudited Models**:
  Marked `AGGREGATE_PARSER_UNAVAILABLE`. No fabricated Recall values; exit code 0 alone does not imply metric parity.

### B. LIMN Bundle Provenance
A complete LIMN evaluation run records the exact individual paths and SHA-256 hashes of all three category models (`0_dress_best_model.pt`, `0_shirt_best_model.pt`, `0_toptee_best_model.pt`), plus a deterministic `bundle_manifest_digest`. Dress is never treated as the sole identity of the three-model benchmark experiment.

---

## 7. Experiment run identity, completion proofs, and immutability

Every evaluation is bound to an explicit *invocation identity* (`run_id`). The
master generates it, exports it to the evaluator, and only accepts the stage as
COMPLETE when the current invocation produced a valid completion proof.

```text
workbench/artifacts/logs/<run_id>/
    command.json          # argv, cwd, digests, checkpoint manifest, interpreter, run_id
    stdout.log            # verbatim evaluator stdout
    stderr.log            # verbatim evaluator stderr
    aggregate_report.json # this run's extraction result
workbench/artifacts/reports/
    <model>_<ckpt>_<run_id>_aggregate.json    # immutable run-scoped report
    <model>_<ckpt>_<run_id>_completion.json   # run-bound completion proof
    <model>_<ckpt>_latest_attempt.json        # atomic pointer: latest run attempted
    <model>_<ckpt>_latest_successful.json     # atomic pointer: latest PROVEN success
```

### Completion contract

A stage may reach COMPLETE only when **all** hold:

1. subprocess exit code `0`;
2. a completion proof exists for **this** `run_id` (never an older pointer);
3. the proof parses and its `run_id` equals the current invocation;
4. `model_id`, `checkpoint_id`, and `protocol_id` match the plan;
5. the recorded source commit matches the pinned commit;
6. the recorded checkpoint digest matches the selected checkpoint;
7. `report` and `report_digest` resolve, and the report lives under the artifact root;
8. the invocation evidence digest matches the `stdout.log`/`stderr.log` actually written;
9. the invocation produced observable output (exit 0 with no output mints no proof).

A pre-existing report, a pre-existing latest pointer, or a newer timestamp is
never sufficient. Execution completion is explicitly distinct from metric
verification: a run that exits 0 with unusable aggregate output still counts as a
completed execution, with `metric_extraction_available: false`.

### latest_attempt vs latest_successful

- `latest_attempt` advances on **every** run, success or failure.
- `latest_successful` advances only after a proof-backed success.
- A failed run therefore never displaces a previous success, and all earlier
  run-scoped reports and proofs remain byte-identical.
- Pointer updates are atomic (`write` + `replace`) and every run-scoped artifact is
  created exclusively (`open("x")`), so a colliding run id raises instead of
  overwriting an experiment. A crash mid-update leaves the previous pointer intact
  and no temp file behind.

### Resume

Resume uses the run-bound proof, not the pointer: a stage recorded as COMPLETE is
re-executed whenever its invocation proof is missing or no longer valid.

## 7b. Interpreter readiness, verification reuse, and version policy

Two distinct operations exist and must not be conflated:

| Operation | Cost | May claim readiness |
|---|---|---|
| `runtime.inspect_environment(model)` | no subprocess | **never** |
| `runtime.verify_environment(model, probe=True)` | runs probes | yes, `RUNTIME_READY` |
| `runtime.require_verified_model_interpreter(model)` | runs probes (cached) | yes; raises otherwise |
| `runtime.python_executable(model)` | delegates to the above | yes; raises otherwise |

`python_executable()` never bypasses verification and never returns an arbitrary
configured path: it raises `RuntimeError` unless the runtime is *verified*.
`doctor.py`, `manage_environment.py`, and `guarded_plan()` share this one definition.

### Linux virtualenv identity

A venv's `bin/python` is routinely a symlink to a shared interpreter binary, so a
matching `resolve()` proves nothing. Identity is established without trusting the
target's self-report:

- **the prefix comes from the invocation path** (`<env>/bin/python` → `<env>`), never
  from `sys.prefix`, so a probe that has had `sys.prefix` rewritten cannot point the
  workbench at an unrelated directory;
- the invocation path must resolve to a **native executable**, not a script, so a
  shell wrapper cannot answer for an environment;
- the nonce probe runs under **`-I`** (ignores user site and `PYTHONPATH`) and must
  return the nonce and the invocation path it executed as;
- a venv's **`pyvenv.cfg` must name its real base interpreter**;
- the environment must contain an **installed Python tree** (`lib/python*/site-packages`,
  `Lib/site-packages`, or `conda-meta`), so a system interpreter dressed in
  hand-written markers is rejected.

Rejected: system Python, base Conda (an install owning an `envs/` directory), the
active workbench environment, wrapper scripts, relocated or hand-dressed interpreters,
and environments with no installed Python tree. A legitimate conda environment created
with `-p/--prefix`, a copied-`bin` venv, a symlinked venv, and a venv living under a
path containing "miniconda3" are all accepted.

Dependency and import probes run under `-I` (but **not** `-S`, so the environment's own
site-packages stay visible) and must report the same `sys.prefix` as the verified
environment; a mismatch is blocked as `DEPENDENCY_PROBE_ENVIRONMENT_MISMATCH`.

**No environment code runs while it is inspected.** The dependency and import
requirements are checked against the environment's *on-disk* inventory, read by the
orchestrator itself via `importlib.metadata.distributions(path=...)`. Nothing is executed
inside the environment, so a `.pth` file, a `sitecustomize` module, or any other code
standing in `site-packages` cannot install or claim a package, and cannot influence the
answer. Two traps are deliberately avoided: the environment's directories are never added
to `sys.path` (`site.addsitedir` would execute its `.pth` files), and CUDA capability is
**not probed** — it can only be reported by running environment code, so a declared CUDA
requirement is reported as `GPU_VERIFICATION_DEFERRED` and is never readiness.

Both `site-packages` and the Debian/Ubuntu `dist-packages` spelling are read, so a
`--system-site-packages` virtualenv over a `dist-packages` base resolves its requirements.

A declared distribution must also have **real installed content**, judged by a single walk
of the environment (`inspect_environment_distributions`) so the version answer and the
"is it actually installed" answer can never disagree:

- a `dist-info` `RECORD` member must be a real file **inside** the tree, outside the
  distribution's own metadata directory, and neither absolute nor traversing
  (`..`/`.`). A `RECORD` listing only itself, or one naming `/etc/hostname`, contributes
  nothing;
- a `RECORD`-less `dist-info` contributes nothing by itself;
- a legacy `egg-info` counts only when the module named in its `top_level.txt` really
  exists, or it keeps a non-descriptive sibling file.

A Debian/Ubuntu `deb`-installed layout is recognised too: those drop `INSTALLER`/`WHEEL`
and no `RECORD`, and the code sits beside the metadata as a package named after the
distribution. Where a distribution's module name cannot be derived from its metadata
(`beautifulsoup4` installs `bs4`), the entry is reported as
`DEPENDENCY_INSTALLATION_UNVERIFIED` rather than guessed at or silently accepted; declare
the module in `required_imports` to verify it.

Metadata that installs nothing describes an installation that never completed (or one
that was wiped), so it cannot satisfy a requirement: the package is reported missing. A declared import module must correspond to a top-level name the environment
actually installs (standard-library and builtin module names always count).

The outcome is three-way:

| Situation | Verdict |
|---|---|
| the declared package is installed on disk at an acceptable version, with installed content | verified |
| a declared constraint cannot be parsed at all | `VERSION_CONSTRAINT_UNVERIFIED` |
| the package is absent, its version cannot be read, or it has metadata but no installed files | `DEPENDENCY_MISSING` / `DEPENDENCY_INSTALLATION_UNVERIFIED` — never `RUNTIME_READY` |

An editable or `system_site_packages` layout that the orchestrator cannot see on disk is
reported unverified rather than accepted; point the model's `WORKBENCH_*_PYTHON` at the
environment that actually holds the dependencies.

There is no change-attribution heuristic: nothing tries to guess *why* the on-disk set
changed, because that is not decidable in process. Isolation removes the need for it.

An unversioned declaration (`openai_clip`, `Pillow`, `torchvision`) means "any version":
presence is the requirement. `VERSION_CONSTRAINT_UNVERIFIED` is reserved for a constraint
that cannot be parsed at all.

### Verification reuse

A verified report may be reused only when the contract digest, the invocation path,
the interpreter binary, the environment markers, and the installed distribution
contents are all unchanged. The fingerprint reads each installed distribution's
`RECORD` and hashes its members, so an install, an uninstall, a force-reinstall, or an
in-place member rewrite invalidates cached verification. Entries are keyed per
**model**, never on the interpreter path alone.

`ponytail:` a distribution listing more than 2000 members is identified by its `RECORD`
and metadata only, so a same-size in-place edit inside one of its members can still go
unnoticed; raise `_FINGERPRINT_MEMBER_LIMIT` or hash members if a stale verification is
ever observed on a real model environment.

Directory/bundle checkpoint digests hash the **member name list first** (length-prefixed),
then each member's name and bytes, so adding or removing an undeclared file in a run
directory cannot re-identify the artifact, and member-name/content substitutions are
detected. Run ids are validated at the entry point: a separator or `..` is refused, so an
orchestrator-supplied `--run-id` can never name a path outside `workbench/artifacts`.

`ponytail:` a same-size, same-mtime rewrite of a package member is likewise invisible to
an mtime/size fingerprint. Hashing package contents would cost seconds per probe; the
existing `WORKBENCH_NO_ENV_CACHE=1` escape hatch forces a full re-probe when that matters.

### Dependency version policy

Constraints are evaluated with `packaging.specifiers`, not string prefixes.

| Declaration | Interpretation |
|---|---|
| `1.12.1` (bare) | exact pin: `==1.12.1` |
| `==1.12.1` | exact |
| `>=1.12,<1.13` | declared range |
| `~=2.20.0` | compatible release |
| `UNKNOWN` / empty | unverified → `VERSION_CONSTRAINT_UNVERIFIED` |

`==1.12.1` rejects `1.12` and `1.12.10`. Malformed constraints or versions never
silently pass: they report `VERSION_CONSTRAINT_UNVERIFIED`. Declared Python versions
with fewer than three components are series requests (`3.8` matches `3.8.x`).
Pre-releases and local version suffixes are handled by `packaging` semantics.

## 7c. Source runtime artifact policy

A model checkout must stay pin-clean, yet some audited contracts legitimately place
files inside it. Source status is classified, never blanket-ignored:

| Class | Meaning |
|---|---|
| tracked modification | a real source change → **dirty** |
| unauthorized untracked/ignored | undeclared file → **dirty** (`.gitignore` never whitelists) |
| approved runtime artifact | declared, provenance-verified → allowed |
| invalid runtime artifact | declared but wrong target/digest → **dirty** |

Declared artifacts are derived from audited registry contracts only:

- CSMCIR `fashionIQ_dataset` — accepted only as a symlink resolving exactly to the
  configured canonical `FASHIONIQ_ROOT`; wrong target, dangling link, or a plain
  directory is rejected.
- Model-source auxiliary assets (e.g. `COT_ours2/fashioniq/*_cot_val.json`) —
  accepted only as regular files whose SHA-256 matches the registry's
  `expected_sha256` or the digest recorded in
  `workbench/artifacts/auxiliary/download_manifest.json` at acquisition time.
  Symlinks, missing files, tampered content, or assets with no verifiable digest are
  rejected. Declared auxiliary artifact *directories* are never trusted.
- The `COT_ours2` captions have **no verified automatic acquisition source**
  (`provenance_status: UNVERIFIED`, no URL, no published checksum). Manual placement
  therefore leaves the checkout **dirty** and blocks strict evaluation until a
  verifiable digest exists — by design, because an unverifiable author artifact is not
  evidence. Closing this needs either a published checksum in the registry or a
  supported `--record-manual` provenance path; neither is implemented, and the
  pipeline reports the blocker explicitly rather than hiding it.
- The allowlist applies only to the checkout the workbench configured
  (`WORKBENCH_THIRD_PARTY_ROOT/<source_dir>`). A directory elsewhere that merely shares
  a model's `source_dir` name is not governed by that model's contract.

`doctor.py` and the execution gate share this exact classification, so preflight and
execution cannot disagree.

## 8. Workflow ownership and locking

`run_stages` takes an exclusive `flock` on `pipeline_state.json.lock` while
executing non-dry-run work. Two concurrent workflows cannot both write state;
the second is refused with a clear lock error. The lock is released when the
process exits, including on crash, so no stale lock survives.

## 9. Compound capability policy

Side-effecting stages declare a set of required capabilities. All must be granted:

| Stage | Required capabilities |
|---|---|
| source sync | `allow_network`, `allow_preparation` |
| checkpoint acquisition | `allow_large_downloads`, `allow_network` |
| auxiliary assets | `allow_network`, `allow_large_downloads` |
| environment provisioning | `allow_env_install`, `allow_network` |
| model preparation (mutating) | `allow_preparation`, `allow_network` |
| evaluation | `allow_gpu_eval`, `allow_preparation` |
| derived index rebuild | `allow_index_write` |
| mock result load/validate | `allow_mock_write` |

Verification-only stages declare no capabilities and remain read-only. `all`,
`setup` and `reproduce` require `--model MODEL_ID` or `--all-models`.

Stages that provision without producing a tracked artifact file (source sync,
environment probe, auxiliary assets, mock load/validate, index rebuild) obtain a
durable completion receipt under `workbench/artifacts/receipts/<stage>.json`.
The receipt is the stage's completion proof and is re-validated on every resume.
`mock` and `analyze` are explicit operator workflows and self-authorize only the
writes they perform; `--dry-run` still plans without executing.
