# DAC Candidate Responsibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a paper-faithful DAC candidate-credit mode with K=8 and a 2,000-successful-update binary split schedule while preserving hard recurrent rollout and existing baselines.

**Architecture:** Pure DAC partition/routing functions live in `src/losses/dac.py`; the objective applies their detached weights to existing differentiable candidate retrieval losses. Training owns one successful-update counter, while configs, diagnostics, tests, and documentation expose the adaptation without changing inference.

**Tech Stack:** Python 3, PyTorch, pytest, Hydra/OmegaConf.

**Spec:** `docs/superpowers/specs/2026-09-20-dac-candidate-responsibility-design.md`

## Global Constraints

- Branch from `exp/e2e-iag-srme-v2-awta-r0`; do not modify that source ref.
- Primary DAC candidate count is exactly K=8 and must be a power of two.
- Split interval defaults to exactly 2,000 successful optimizer updates.
- Preserve ScoreNet hard argmax/STOP rollout and target-free inference.
- Preserve `none`, `awta`, and `hard_wta` candidate-credit modes and all non-credit losses.
- DAC assignment uses detached per-candidate losses; winning-group member losses remain differentiable.

## Review Focus

- Exact boundary convention: the batch seeing `global_step=2000` must use stage 1.
- AMP overflow: skipped optimizer steps must not advance DAC stage.
- Ties: detached `argmin` must deterministically select the first minimum and its group.
- Empty/invalid teacher rows must produce finite zero credit and diagnostics.
- Diagnostics must not accidentally use targets in model inference.

---

### Task 1: Pure DAC hierarchy and routing

**Files:**
- Create: `src/losses/dac.py`
- Create: `tests/test_dac_candidate_credit.py`

**Interfaces:**
- Produces: `validate_dac_candidate_count(int) -> int`, `dac_stage(int, int, int) -> int`, `dac_groups(int, int) -> tuple[tuple[int, ...], ...]`, `dac_responsibility_weights(Tensor, int) -> tuple[Tensor, Tensor]`.
- Consumes: PyTorch tensors with candidates on the final axis.

- [ ] **Step 1: Write failing hierarchy and boundary tests**

Use literal K=8 groups for stages 0–3 and literal stage expectations for steps `0, 1999, 2000, 3999, 4000, 5999, 6000, 12345`. Add validation tests for K=6, zero interval, and negative step.

- [ ] **Step 2: Run tests and verify import failure**

Run: `pytest -q tests/test_dac_candidate_credit.py`

Expected: collection fails because `losses.dac` does not exist.

- [ ] **Step 3: Implement validation, stage, and groups**

Use integer power-of-two validation and contiguous group size `K // 2**stage`, clamping stage at `log2(K)`.

- [ ] **Step 4: Write failing routing/gradient tests**

Cover gradients `[1/8]*8`, `[1/4]*4+[0]*4`, the correct stage-2 pair, and one-hot leaf behavior; cover left winner 3, right winner 6, different groups in one batch, detached weights, and hard-WTA leaf equivalence.

- [ ] **Step 5: Implement detached group weights**

Find detached global argmin, derive group index by integer division, and scatter `1/group_size` over that contiguous group. Return detached weights and winning group indices.

- [ ] **Step 6: Verify Task 1**

Run: `pytest -q tests/test_dac_candidate_credit.py`

Expected: all Task 1 tests pass.

### Task 2: Objective integration and DAC metrics

**Files:**
- Modify: `src/losses/objective.py`
- Modify: `tests/test_dac_candidate_credit.py`
- Verify: `tests/test_awta_candidate_credit.py`

**Interfaces:**
- Consumes: Task 1 helpers and keyword-only `global_step`.
- Produces: DAC candidate-credit loss plus scalar `dac_*` metrics, including dynamic per-slot and per-current-group frequencies.

- [ ] **Step 1: Write failing objective tests**

Monkeypatch existing teacher helpers with literal K=8 losses to verify stage routing, batch groups, invalid-row finite zeros, target detachment, and nonzero candidate-query gradients only for winning-group candidates.

- [ ] **Step 2: Run objective tests and verify expected failures**

Run: `pytest -q tests/test_dac_candidate_credit.py tests/test_awta_candidate_credit.py`

Expected: DAC mode/config/global-step tests fail while existing aWTA tests pass.

- [ ] **Step 3: Integrate DAC mode**

Add `dac_split_interval_steps=2000`, mode validation, `global_step=0`, one shared retrieval-loss calculation, DAC weights, and separate metrics. Do not rename aWTA metrics or alter its epoch schedule.

- [ ] **Step 4: Verify objective integration**

Run: `pytest -q tests/test_dac_candidate_credit.py tests/test_awta_candidate_credit.py tests/test_v2_objective.py tests/test_v2_teacher.py`

Expected: all selected tests pass.

### Task 3: Successful optimizer-update plumbing and checkpoint metadata

**Files:**
- Modify: `src/training/engine.py`
- Create: `tests/test_dac_global_step.py`

**Interfaces:**
- Consumes: objective `global_step` keyword.
- Produces: `train_one_epoch(..., global_step: int) -> tuple[dict[str, float], int]`; checkpoints store `global_step`.

- [ ] **Step 1: Write failing success/overflow counter tests**

Use small real modules/loaders or focused fake scaler behavior to show the objective observes pre-update steps and that successful updates increment while an overflow-skipped step does not.

- [ ] **Step 2: Run and verify signature/behavior failures**

Run: `pytest -q tests/test_dac_global_step.py`

Expected: failure because no global-step interface exists.

- [ ] **Step 3: Implement one counter source of truth**

Pass pre-update step into the objective; compare scaler scale before/after update and treat a decrease as skipped when scaling is enabled. Thread the resulting count through `fit` and `save_checkpoint`.

- [ ] **Step 4: Verify engine tests**

Run: `pytest -q tests/test_dac_global_step.py tests/test_optimizer_device_order.py tests/test_precision_policy.py tests/test_canary_amp_overflow.py`

Expected: all selected tests pass.

### Task 4: K=8 configs, architecture smoke, and diagnostics

**Files:**
- Create: `conf/model/iag_srme_k8.yaml`
- Create: `conf/objective/core_dac.yaml`
- Create: `conf/objective/core_dac_no_dpp.yaml`
- Modify: `src/train.py`
- Modify: `src/diagnose_candidate_selector.py`
- Modify: `src/diagnose_iag_srme.py`
- Modify: `tests/test_dac_candidate_credit.py`

**Interfaces:**
- Consumes: DAC helpers, checkpoint `global_step`, and K=8 Hydra configs.
- Produces: matched launch configs, K=8 tensor-shape proof, per-slot/per-group metrics, and candidate-credit gradient routing.

- [ ] **Step 1: Write failing K=8 model smoke and config-composition tests**

Assert K=8 proposal, grounding, executor, candidate-query, score shapes; hard rollout returns exactly one selected index per live row and forward remains target-free. Compose each K=8 objective/model pair through Hydra.

- [ ] **Step 2: Run and verify missing-config failures**

Run: `pytest -q tests/test_dac_candidate_credit.py`

Expected: configuration tests fail because DAC/K8 configs do not exist.

- [ ] **Step 3: Add configs and early K validation**

Inherit historical configs and override only experimental fields. Validate model K during objective construction when DAC is active.

- [ ] **Step 4: Extend diagnostics**

Use checkpoint global step for objective calls; include candidate credit in loss decomposition/gradient probes; calculate DAC responsibilities from candidate retrieval losses in the selector diagnostic; render DAC metrics without changing inference.

- [ ] **Step 5: Verify Task 4**

Run: `pytest -q tests/test_dac_candidate_credit.py tests/test_v2_architecture.py tests/test_track_b_readout_configs.py`

Expected: all selected tests pass.

### Task 5: Research note, smoke proof, and complete verification

**Files:**
- Create: `doc/DAC_CANDIDATE_RESPONSIBILITY_ADAPTATION_2026-09-20.md`
- Modify as required by verified findings from prior tasks.

**Interfaces:**
- Consumes: final implementation, configs, test outputs, and paper research.
- Produces: reproducible commands and an evidence-backed implementation record.

- [ ] **Step 1: Write the implementation note**

Include all 14 requested research/documentation topics, exact commands, official-code result, and the paper-vs-CIR adaptation caveat.

- [ ] **Step 2: Run focused mathematical and CPU smoke verification**

Run the DAC, aWTA, objective, engine, and architecture test files, plus a standalone CPU forward/backward command if the tests do not already expose all requested gradient facts.

- [ ] **Step 3: Run the full suite**

Run: `pytest -q`

Expected: all tests pass with only established environment-dependent skips.

- [ ] **Step 4: Inspect source-ref integrity and final diff**

Run `git rev-parse exp/e2e-iag-srme-v2-awta-r0`, `git diff --check`, `git status --short`, and inspect `git diff 57954676...HEAD` for unrelated changes.

- [ ] **Step 5: Commit finished implementation**

Commit message: `feat: add paper-faithful DAC candidate responsibility with K=8`
