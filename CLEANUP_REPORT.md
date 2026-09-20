# Repository Cleanup Report

This report records the final source-code cleanup performed for the academic
submission. The cleanup was deliberately conservative: no retrieval formula,
TAPER transition, slot/routing behavior, teacher composition, checkpoint
loading rule, or benchmark metric was changed.

## Audit and classification

The repository was audited through Python imports, Hydra defaults and config
key reads, shell/README references, unit tests, entrypoint composition, data and
cache loaders, and tracked-file reachability.

### Final tracked tree

```text
.
├── .gitignore
├── CLEANUP_REPORT.md
├── LICENSE
├── README.md
├── pyproject.toml
├── conf/
│   ├── config.yaml
│   ├── dataset/{cirr,fashioniq}.yaml
│   ├── experiment/{default,taper_e2e,taper_e2e_ortho07}.yaml
│   └── protocol/{cirr_encoder,cirr_val,fashioniq_encoder,
│                 fashioniq_original}.yaml
├── src/
│   ├── cache/{__init__,features}.py
│   ├── datasets/{__init__,cirr,common,fashioniq}.py
│   ├── evaluation/{__init__,cirr,cirr_encoder,fashioniq,
│   │              fashioniq_encoder}.py
│   ├── models/{__init__,taper}.py
│   ├── teachers/{__init__,csmcir,csmcir_compose}.py
│   ├── training/{__init__,engine}.py
│   ├── check_{csmcir_compose,taper_chunk,taper_text_cache}_parity.py
│   ├── precompute_{cirr,csmcir_stage1,taper_e2e_text}.py
│   ├── train.py
│   ├── train_cirr.py
│   ├── train_{cirr,fashioniq}_encoder.py
│   ├── evaluate_{cirr,fashioniq}_encoder.py
│   ├── predict_cirr_encoder_test.py
│   └── runtime.py
├── teacher/README.md
└── tests/unit/
    ├── test_cirr.py
    ├── test_cirr_encoder.py
    ├── test_cirr_encoder_submission.py
    ├── test_cirr_metrics.py
    ├── test_encoder_mode_isolation.py
    ├── test_fashioniq_encoder.py
    └── test_taper_slot_query_ortho.py
```

Ignored data, feature, checkpoint, output, environment, and external teacher
directories are not shown.

### KEEP

- `src/models/taper.py`: the submitted model, including edit-slot refinement,
  residual allocation, routing, gating, execution, query construction, and
  `model.retrieve()` scoring.
- `src/teachers/csmcir.py` and `src/teachers/csmcir_compose.py`: the runtime
  boundary to the frozen external CSMCIR teacher.
- `src/datasets/`, `src/cache/`, and `src/runtime.py`: shared dataset, feature
  cache, and runtime infrastructure used by the submitted entrypoints.
- `src/precompute_csmcir_stage1.py`, `src/precompute_taper_e2e_text.py`, and
  `src/precompute_cirr.py`: the precomputation path required to create the
  FashionIQ and CIRR caches consumed by TAPER. The first filename is historical,
  but the script is part of the current image-cache pipeline.
- `src/train_fashioniq_encoder.py`, `src/evaluate_fashioniq_encoder.py`,
  `src/train_cirr_encoder.py`, `src/evaluate_cirr_encoder.py`, and
  `src/predict_cirr_encoder_test.py`: the final isolated training, validation,
  and CIRR test1 submission entrypoints.
- `src/evaluation/fashioniq_encoder.py` and
  `src/evaluation/cirr_encoder.py`: final benchmark evaluation semantics.
- `src/train.py`, `src/train_cirr.py`, `src/evaluation/fashioniq.py`, and
  `src/evaluation/cirr.py`: executable legacy baselines retained for
  reproducibility and for the encoder/legacy isolation guarantee.
- `src/check_csmcir_compose_parity.py`,
  `src/check_taper_text_cache_parity.py`, and
  `src/check_taper_chunk_parity.py`: focused parity checks for the external
  composition and feature caches, retained because they can validate numerical
  equivalence when the external artifacts are available.
- `conf/experiment/taper_e2e.yaml` and
  `conf/experiment/taper_e2e_ortho07.yaml`: the base and orthogonal-loss final
  experiments. Their model settings and loss values remain unchanged.
- `conf/protocol/cirr_val.yaml`: used by the retained legacy CIRR evaluator for
  configurable global and subset recall cutoffs.
- All tests under `tests/unit/`: useful protocol, isolation, cache, metric, and
  model tests; none were removed.
- `teacher/README.md`: the tracked contract for external CSMCIR source and
  checkpoint placement. `teacher/repos/` and `teacher/checkpoints/` remain
  intentionally untracked external-asset locations.

### REMOVE

- `.idea/`: editor-specific project metadata, unrelated to execution or
  reproduction.
- `reports/*.json`: generated diagnostic outputs, not inputs to source,
  configs, tests, training, or evaluation.
- `teacher/audit/`: historical teacher-comparison/audit infrastructure and its
  generated data. Its references were internal to that audit subtree; the
  current runtime imports `src/teachers/` instead.
- `teacher/adapters/`: adapters for historical teacher comparisons. No final
  runtime, Hydra config, test, or README command imported them.
- `src/diagnose_a5.py`,
  `src/diagnose_slot_functional_specialization_a51c.py`,
  `src/diagnose_taper_checkpoint.py`, and `src/forensic_taper_a3.py`: large
  investigation scripts tied to abandoned A3/A5 diagnostics and the removed
  reports, outside the final reproduction graph.
- `src/train_stage1.py`, `src/evaluation/edit_slot_stage1.py`,
  `src/probe_stage1_relations.py`, and `conf/experiment/taper_stage1.yaml`: an
  obsolete standalone Stage-1 experiment. It called
  `TAPER.compute_stage1_loss()`, which intentionally rejects that formulation
  because it is incompatible with the current competitive NULL allocation; it
  is not part of the final end-to-end pipeline.
- `conf/model/blip2_qure.yaml` and `conf/objective/qure_pairwise.yaml`: one-line
  bootstrap markers whose config groups were never read by current code.
- `conf/protocol/fashioniq_val.yaml`: a zero-byte config with no values and no
  final command reference. The legacy evaluator's programmatic
  `fashioniq_val` branch was left intact.
- `requirements.txt`: a machine-specific environment freeze containing
  transitive CUDA/NVIDIA packages; replaced by the direct dependencies already
  managed in `pyproject.toml`.

The exact removed tracked files are:

```text
.idea/.gitignore
.idea/cir-research.iml
.idea/inspectionProfiles/Project_Default.xml
.idea/inspectionProfiles/profiles_settings.xml
.idea/modules.xml
.idea/vcs.xml
conf/experiment/taper_e2e_ortho07_encoder.yaml
conf/experiment/taper_stage1.yaml
conf/model/blip2_qure.yaml
conf/objective/qure_pairwise.yaml
conf/protocol/fashioniq_val.yaml
reports/a5_1c_forensic.json
reports/a5_1c_forensic_15ep.json
reports/a5_1c_functional_specialization.json
reports/taper_a3_forensic.json
reports/taper_a3_forensic_depth.json
reports/taper_checkpoint_diagnosis.json
requirements.txt
src/diagnose_a5.py
src/diagnose_slot_functional_specialization_a51c.py
src/diagnose_taper_checkpoint.py
src/evaluation/edit_slot_stage1.py
src/forensic_taper_a3.py
src/probe_stage1_relations.py
src/train_stage1.py
teacher/adapters/csmcir.py
teacher/adapters/encoder.py
teacher/adapters/hint.py
teacher/adapters/sprc.py
teacher/adapters/tgcir.py
teacher/adapters/tme.py
teacher/audit/build_fashioniq_cases.py
teacher/audit/fashioniq_val_cases.json
teacher/audit/full_audit.py
teacher/audit/geometry_screen_from_artifacts.py
teacher/audit/metrics.py
teacher/audit/provenance/csmcir_compat_provenance.txt
teacher/audit/provenance/csmcir_models_init_compat.patch
teacher/audit/run.py
teacher/audit/run_full.py
teacher/audit/teacher_envs.json
```

### MOVE/REORGANIZE

No source module was moved or renamed. Avoiding such moves keeps public
entrypoints and import paths stable. The root `teacher/` area was reduced to a
documented external-asset boundary rather than being merged into
`src/teachers/`: these locations have different responsibilities.

### MERGE/DEDUPLICATE

- Removed `conf/experiment/taper_e2e_ortho07_encoder.yaml`; it was byte-for-byte
  equivalent in effective settings to
  `conf/experiment/taper_e2e_ortho07.yaml`. The canonical selector is now
  `experiment=taper_e2e_ortho07` for both datasets.
- Removed unused root Hydra groups `model` and `objective` from
  `conf/config.yaml`. The actual model/teacher settings continue to live in the
  selected experiment config, exactly where all runtime entrypoints read them.
- Removed unused config values (`dataset.split`, derived annotation paths,
  `runtime.compile`, and bootstrap logging values) only after confirming no
  source or test reads them.

### UNCERTAIN / conservatively retained

- The legacy train/evaluation modules may not be needed for the final reported
  encoder runs, but they are functional historical baselines and are covered by
  isolation expectations. They were retained rather than risking a loss of
  experimental reproducibility.
- The repository does not identify an immutable upstream CSMCIR source commit
  or source URL. The runtime contract and checkpoint location are documented,
  but an exact upstream source revision cannot be reconstructed from the
  repository alone. No URL or revision was fabricated.
- The three parity utilities cannot be classified as unnecessary without the
  external teacher and caches. They are small, focused validation tools and
  were retained.

## Removed

The files listed under **REMOVE** were deleted only after searches confirmed
that they were outside the final Python import graph, Hydra configuration
graph, tests, and current reproduction commands. Generated reports were not
moved into documentation because the final paper/report does not reference
them from this repository.

## Kept intentionally

The final FashionIQ and CIRR encoder pipelines, CIRR test1 submission support,
the legacy baseline pipeline, all protocol tests, and parity utilities were
kept. No FashionIQ or CIRR dataset semantic, gallery ordering, reference
exclusion rule, metric, checkpoint selection rule, or retrieval score path was
changed.

## Consolidated

The duplicate orthogonal experiment config was consolidated under
`taper_e2e_ortho07.yaml`. Empty/unused Hydra groups and values were removed,
while all values read by final and retained legacy entrypoints remain.

## Dependency changes

- `pyproject.toml` is the single project dependency declaration.
- Added package build metadata so an editable install can discover packages
  below `src/`.
- Added `wandb` as a direct dependency because `src/training/engine.py` imports
  it unconditionally.
- Kept the existing unpinned PyTorch and core dependency strategy to avoid an
  arbitrary CUDA or CSMCIR compatibility change.
- Removed the environment-freeze `requirements.txt`; setup now instructs users
  to install the appropriate PyTorch/torchvision build first and then install
  the project with `python -m pip install -e ".[dev]"`.
- External CSMCIR/LAVIS compatibility remains the responsibility of the
  upstream CSMCIR environment and is explicitly documented.

## README changes

The old failed-experiment narrative was replaced by a final-project README
covering the method, repository layout, environment, CSMCIR contract, exact
FashionIQ/CIRR layouts, cache precomputation, both training commands, both
validation commands, CIRR test1 submission, expected outputs, tests,
reproducibility, license, and acknowledgements.

Every displayed command was checked against the corresponding argument parser
or Hydra configuration. The test1 documentation preserves full-gallery
`model.retrieve()` scoring, reference removal, global top-50, and subset top-3
filtering without subset rescoring.

## Verification

The following checks were executed after cleanup:

```text
python -m compileall src tests
pytest -q
ruff check src tests
pip install --dry-run --break-system-packages --no-deps \
  --no-build-isolation -e .
```

Hydra composition was checked with `--cfg job` for:

```text
src/train_fashioniq_encoder.py (taper_e2e and taper_e2e_ortho07)
src/evaluate_fashioniq_encoder.py
src/train_cirr_encoder.py
src/evaluate_cirr_encoder.py
src/predict_cirr_encoder_test.py
src/train.py
src/train_cirr.py
```

Argument-parser help was checked for all three precomputation entrypoints.
Important entrypoint imports, deleted-symbol references, README paths, and Git
whitespace were also checked.

Final outcomes:

- Compile: passed with no errors.
- Unit tests: **22 passed**.
- Ruff: **all checks passed** for `src` and `tests`.
- Editable package metadata: passed; pip prepared editable metadata and reported
  `Would install cir-research-0.1.0`. Build isolation was disabled because the
  verification environment has no network access; its installed setuptools
  68.1.2 satisfies the declared `setuptools>=68` build requirement.
- Final encoder Hydra commands: all six train/evaluate/test1 compositions
  passed, including the canonical orthogonal config.
- Legacy config composition: both `fashioniq_original` and `cirr_val` passed.
  Direct legacy script imports require the newly declared `wandb` dependency,
  which is not installed in the current test virtualenv. An import-reachability
  smoke test with only the absent `wandb` module stubbed imported all 12
  important entrypoint/runtime modules successfully.
- Precomputation parsers: all three `--help` checks passed.
- Local FashionIQ loader smoke: loaded 2,017 dress validation samples and a
  3,817-image validation gallery using the available official annotations and
  the uncorrected `ordered_and` policy.
- `git diff --check`: passed.
- Deleted-module/config reference search: no stale runtime references were
  found. The surviving `fashioniq_val` string is the intentionally retained
  programmatic legacy evaluator branch, not a reference to the removed empty
  Hydra file.
- README path check: every source/config/test path exists. The only absent
  paths are the explicitly external `teacher/repos/CSMCIR` source and CSMCIR
  checkpoint documented as user-supplied artifacts.

## Remaining limitations

- No local CIRR dataset is available, so CIRR annotation/image loading and a
  live evaluation batch could not be executed.
- The local FashionIQ copy contains official annotations/images but lacks the
  three ENCODER correction dictionaries required by the final caption policy.
  Only a loader smoke test with the uncorrected `ordered_and` policy can be run.
- The external `teacher/repos/CSMCIR` source, CSMCIR checkpoint, TAPER
  checkpoints, and feature caches are absent. Teacher construction, checkpoint
  restoration, parity checks, and a live retrieval batch therefore require the
  user-supplied research artifacts documented in the README.
- Because the repository does not record the exact external CSMCIR source
  revision, bit-for-bit environment reconstruction still requires that
  revision to be supplied alongside an academic artifact release.
