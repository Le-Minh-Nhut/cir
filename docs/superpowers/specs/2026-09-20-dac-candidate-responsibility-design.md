# Paper-Faithful DAC Candidate Responsibility Design

## Intent

Adapt the Divide-and-Conquer (DAC) responsibility assignment from Narayanan et al., CVPR 2021, to CIR's existing recurrent sibling-candidate training path. The adaptation replaces or ablates aWTA candidate credit without changing the target-free model interface, ScoreNet's hard recurrent selection, STOP semantics, or the non-credit objective terms.

Success means that a K=8 experiment trains every candidate in the winning binary-tree group, moves from shared responsibility to hard WTA at exact 2,000-successful-optimizer-update boundaries, preserves matched K=8 aWTA and hard-WTA controls, exposes enough diagnostics to distinguish DAC credit from hard-rollout gradients, and passes mathematical gradient-routing and model smoke tests.

## Authoritative Sources

The authoritative method specification is:

- Sriram Narayanan, Ramin Moslemi, Francesco Pittaluga, Buyu Liu, and Manmohan Chandraker. “Divide-and-Conquer for Lane-Aware Diverse Trajectory Prediction.” CVPR 2021, pp. 15799–15808. arXiv:2104.08277.
- The official CVPR 2021 supplementary material.

Algorithm 1 chooses the set whose minimum member loss is lower than every other set's minimum, then returns the mean of all hypothesis losses in that winning set. The supplement states that DAC hypotheses are split every 2,000 training iterations. No official public author implementation was found through the CVF record, the lead author's publication page, GitHub searches, or contemporary CVPR code indexes. Unrelated projects named DAC are not implementation sources.

## Existing Architecture and Preserved Semantics

At each recurrent timestep, ProposalNet, Grounder, ActionFusion, and Executor produce K parallel sibling alternatives from one parent state. ScoreNet scores those siblings, hard argmax selects exactly one candidate (or STOP), and only the selected state becomes the next recurrent parent.

DAC is an auxiliary training responsibility path over each timestep's candidate retrieval losses. It does not change candidate generation, recurrent state transitions, ScoreNet, STOP, or inference. Target embeddings remain privileged training supervision and never enter the model's forward signature.

The existing terminal retrieval loss still backpropagates through the hard-executed recurrent path. DAC separately supplies differentiable task credit to every candidate in the winning group. Diagnostics will report terminal and DAC gradient routing separately so this architectural difference from the paper's direct multi-hypothesis output setting remains measurable.

## DAC Rule

For one valid training row with differentiable candidate retrieval losses

`L = [L_0, ..., L_7]`,

the routing decision uses `L.detach()`:

1. Find `k* = argmin_k L_k` with deterministic PyTorch first-index tie behavior.
2. Find the current fixed contiguous group `G*` containing `k*`.
3. Return `mean(L_k for k in G*)`.

Equivalently, construct a detached weight vector with value `1 / |G*|` for candidates in `G*` and zero outside it, then take its dot product with the differentiable losses. No assignment gradient, group-mean routing, softmax, temperature, random grouping, balancing, cloning, or perturbation is introduced.

Invalid teacher rows are skipped exactly as in the existing candidate-credit path. STOP is not a candidate and never enters DAC.

## K=8 Hierarchy and Schedule

DAC mode requires a positive power-of-two candidate count. The primary configuration uses exactly K=8:

- Stage 0: `[[0,1,2,3,4,5,6,7]]`
- Stage 1: `[[0,1,2,3], [4,5,6,7]]`
- Stage 2: `[[0,1], [2,3], [4,5], [6,7]]`
- Stage 3: `[[0], [1], [2], [3], [4], [5], [6], [7]]`

For split interval `S=2000` and completed successful optimizer-update count `global_step`, the zero-based stage is:

`min(global_step // S, log2(K))`.

The first attempted batch sees `global_step=0`. The objective uses the number of successful updates completed before the current batch. A successful optimizer update increments the counter once; an AMP-overflow-skipped step does not. Thus stages change before updates numbered 2001, 4001, and 6001 in one-based human counting, at zero-based global steps 2000, 4000, and 6000.

The paper calls the unsplit state “Depth 1”; repository stage 0 is the same state. This naming translation is the only schedule convention difference.

## Components and Interfaces

### Pure DAC helpers

Create `src/losses/dac.py` with focused pure functions to:

- validate power-of-two K and positive schedule inputs;
- calculate stage from global step;
- build the exact contiguous binary partition;
- create detached per-row responsibility weights and winning-group indices.

These helpers operate on the last candidate axis and are independently unit-tested.

### Objective integration

Extend `ObjectiveConfig` with `dac_split_interval_steps: int = 2000`. Accept `candidate_credit_mode` values `none`, `awta`, `hard_wta`, and `dac`.

`IAGSRMEObjective.forward` receives keyword-only `global_step: int = 0` alongside the existing epoch. Epoch remains the sole input to the unchanged aWTA temperature schedule. DAC uses only global step and split interval.

The objective continues to call the existing identity-safe `teacher_retrieval_loss` once for valid candidate rows. aWTA and hard WTA keep their current detached weights. DAC requests its detached group weights and calculates the same row-normalized candidate-credit aggregation.

DAC-only output metrics include stage, number of groups, group size, split interval, gradient-receiving fraction, responsibility concentration, per-candidate responsibility frequency, and per-group winning frequency. aWTA names remain aWTA-specific.

### Training counter and checkpoints

`train_one_epoch` accepts a starting global step and returns both averaged metrics and the updated count. It passes the pre-update count into the objective, performs backward/scaler step, detects an AMP-overflow skip from the scaler's scale transition, and increments only on a successful optimizer update.

`fit` owns the single counter across epochs. Checkpoints store it at top level and in metadata. Existing `objective_config` metadata records the DAC interval and mode. There is no unrelated resume refactor, but the stored counter is compatible with future resume support.

### Configurations

Add:

- `conf/model/iag_srme_k8.yaml`, inheriting the historical model config and overriding only the name and K;
- `conf/objective/core_dac.yaml`, inheriting `core`, enabling DAC credit with weight 1.0 and interval 2000;
- `conf/objective/core_dac_no_dpp.yaml`, inheriting the DAC config and disabling/zeroing DPP for the isolated specialization ablation.

Matched commands use the same K=8 model with `core_dac`, `core_awta`, or `core_hard_wta`. Historical K=4 configs remain unchanged.

### Diagnostics

Extend the objective and diagnostic scripts without changing model behavior. Report:

- DAC stage, group count/size, interval, credit loss, gradient-receiving fraction, responsibility concentration;
- per-candidate DAC responsibility frequency and current-stage group win frequency;
- existing candidate utility by slot, selected/oracle histograms, useful rates, functional cosine, and effective rank;
- separate candidate-credit gradient routing into proposal, grounding, action fusion, executor, and other trainable paths, alongside terminal routing.

Early-stage similarity is not automatically classified as DAC failure because stage 0 intentionally shares every training row across all candidates.

## Testing Strategy

Development follows red-green-refactor. Tests use literal expected partitions and gradients rather than reproducing helper logic.

Coverage includes:

- exact K=8 partitions at stages 0–3;
- steps 0, 1999, 2000, 3999, 4000, 5999, 6000, and later;
- exact gradients 1/8, 1/4, 1/2, and 1 for winning groups;
- left- and right-half winners and different winning groups within one batch;
- detached assignment and final-stage numerical/gradient equivalence with hard WTA;
- rejection of non-power-of-two K and invalid schedule inputs;
- invalid teacher rows remaining finite and skipped;
- global-step propagation and no increment for a scaler-skipped optimizer update;
- K=8 ProposalNet, Grounder, Executor, candidate-query, and ScoreNet shapes;
- hard rollout still selects one sibling, STOP remains outside DAC, and inference remains target-free;
- existing aWTA/hard-WTA tests and the full existing suite.

A CPU forward-backward smoke test isolates DAC credit, retains candidate-query gradients, verifies nonzero gradients for all candidates in the winning group and zeros outside it, and confirms leaf equivalence to hard WTA.

## Documentation and Delivery

Create a concise implementation note under `doc/` containing the citation, exact algorithm, K=8 hierarchy, schedule, DAC/aWTA/DPP distinctions, recurrent adaptation point, changed files, commands, verification evidence, ambiguity notes, and official-code search result.

Before the final commit, inspect the complete diff, rerun targeted and full tests, run the CPU smoke check, verify the source branch still resolves to its original commit, and commit on `exp/e2e-iag-srme-v2-dac-r0-k8` with a descriptive message.

## Explicit Non-Goals

This is not a reproduction of ALAN or the entire trajectory-prediction paper. It does not add noise, cloning, clustering, load balancing, entropy regularization, Sinkhorn routing, Gumbel-Softmax, mixture rollout, target-conditioned inference, new slot embeddings, or DPP changes. It is a paper-faithful DAC responsibility adaptation to the existing CIR architecture.
