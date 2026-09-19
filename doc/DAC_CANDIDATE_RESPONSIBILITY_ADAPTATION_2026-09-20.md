# Paper-Faithful DAC Responsibility Adaptation

## Citation and source status

Sriram Narayanan, Ramin Moslemi, Francesco Pittaluga, Buyu Liu, and Manmohan Chandraker. “Divide-and-Conquer for Lane-Aware Diverse Trajectory Prediction.” CVPR 2021, pp. 15799–15808. arXiv:2104.08277.

Authoritative sources:

- Paper: <https://openaccess.thecvf.com/content/CVPR2021/papers/Narayanan_Divide-and-Conquer_for_Lane-Aware_Diverse_Trajectory_Prediction_CVPR_2021_paper.pdf>
- Supplement: <https://openaccess.thecvf.com/content/CVPR2021/supplemental/Narayanan_Divide-and-Conquer_for_Lane-Aware_CVPR_2021_supplemental.pdf>

No official public author implementation was found. The CVF record and lead author's publication page link the paper, supplement, blog, and talk but not code; contemporary CVPR code indexes also list no implementation. The paper and supplement are therefore the authoritative specification. No code from unrelated projects named “DAC” was used.

## Implemented algorithm

For differentiable candidate retrieval losses `L[0:K]`, assignment uses detached losses:

1. `k* = argmin(L.detach())`.
2. Select the fixed current-stage group `G*` containing `k*`.
3. `L_DAC = mean(L[k] for k in G*)`.

Equivalently, the detached responsibility weight is `1 / |G*|` inside the winning group and zero outside. Consequently, the exact derivative with respect to each candidate loss is `1 / |G*|` inside the group and zero outside. At the leaf stage this is numerically and gradient-wise identical to hard WTA.

This follows Algorithm 1: group selection is by the minimum member, followed by the mean of every loss in that group. It does not select by group mean, update only the argmin before the leaf, average groups, use a temperature, or randomly regroup candidates. PyTorch's deterministic first-index `argmin` behavior resolves exact ties, which the paper does not discuss.

Rows without valid identity-safe retrieval negatives retain the existing skip behavior and contribute no candidate-credit loss or responsibility frequency.

## Mapping to CIR

The paper trains direct multi-hypothesis outputs. CIR instead produces K sibling candidate states at every recurrent timestep and hard-selects one using ScoreNet for the next recurrent state. DAC is adapted only to the explicit candidate retrieval credit path:

`candidate states -> candidate retrieval queries -> teacher_retrieval_loss -> DAC responsibility`.

The regular terminal retrieval loss still follows the hard-executed recurrent path. Thus terminal gradients may favor the executed candidate, while DAC independently provides task gradient to every candidate in the winning group. The diagnostic gradient-routing table reports terminal and `candidate_credit` routes separately so their relative effects can be inspected.

Inference remains target-free and unchanged. ScoreNet still chooses exactly one candidate or STOP per timestep. STOP is not a ninth DAC hypothesis.

## K=8 hierarchy

K=8 is used because it forms an exact balanced binary tree. DAC mode rejects non-power-of-two candidate counts.

- Stage 0: `[[0,1,2,3,4,5,6,7]]`
- Stage 1: `[[0,1,2,3], [4,5,6,7]]`
- Stage 2: `[[0,1], [2,3], [4,5], [6,7]]`
- Stage 3: `[[0], [1], [2], [3], [4], [5], [6], [7]]`

The effective output count evolves as `1 -> 2 -> 4 -> 8`.

## Split schedule

The supplement states that hypotheses are split every 2,000 training iterations. Here an iteration is a successful optimizer update:

`stage = min(global_step // dac_split_interval_steps, log2(K))`.

With the default interval 2,000:

- steps 0–1999: stage 0;
- steps 2000–3999: stage 1;
- steps 4000–5999: stage 2;
- steps 6000 onward: stage 3.

`global_step` is the number of successful updates completed before the current batch. AMP-overflow-skipped steps do not increment it. Checkpoints store the counter at top level and in metadata. The repository has no resume path, so no unrelated resume refactor was added; the stored value is available for future resume support.

The paper calls the initial unsplit state “Depth 1”; this implementation calls it stage 0. The partitions and timing are otherwise the same.

## DAC versus aWTA and DPP

aWTA uses detached soft assignments `softmax(-L / temperature)` with its existing epoch-based temperature schedule. DAC uses discrete binary-tree groups and no temperature. aWTA, hard WTA, and `none` remain available unchanged as candidate-credit modes.

DAC controls hierarchical task responsibility. Functional DPP remains an independent auxiliary that encourages diversity in retrieval-effect space. The matched DAC config preserves DPP and every other source-branch non-credit setting. `core_dac_no_dpp` disables DPP for an isolated specialization ablation; it does not otherwise change DAC.

## Diagnostics

The objective and diagnostic scripts expose:

- DAC stage, group count, group size, and split interval;
- candidate-credit loss and weighted loss;
- per-candidate responsibility frequency;
- per-current-group winning frequency;
- fraction of candidates receiving DAC gradient per valid row;
- responsibility concentration;
- candidate utility/useful rate by slot;
- selected and oracle slot histograms;
- functional pairwise cosine and effective rank;
- a separate candidate-credit gradient-routing probe.

Stage-0 candidate similarity is not automatically reported as candidate collapse because shared responsibility is intentional at that stage.

## Files changed

- `src/losses/dac.py`: pure DAC validation, schedule, partitions, and detached routing.
- `src/losses/objective.py`: DAC candidate-credit integration and metrics.
- `src/training/engine.py`: successful-update global step and checkpoint metadata.
- `src/train.py`: early DAC K validation.
- `src/diagnose_candidate_selector.py`: DAC responsibility and group-frequency diagnostics.
- `src/diagnose_iag_srme.py`: DAC metrics, stage-aware flags, and candidate-credit gradient routing.
- `conf/model/iag_srme_k8.yaml`: matched K=8 model.
- `conf/objective/core_dac.yaml`: paper schedule with the source objective settings.
- `conf/objective/core_dac_no_dpp.yaml`: isolated no-DPP ablation.
- `tests/test_dac_candidate_credit.py`: hierarchy, boundaries, routing, gradients, invalid rows, configs, and K=8 model smoke tests.
- `tests/test_dac_global_step.py`: successful/skipped update and checkpoint tests.
- `docs/superpowers/specs/2026-09-20-dac-candidate-responsibility-design.md`: approved design record.
- `docs/superpowers/plans/2026-09-20-dac-candidate-responsibility.md`: implementation plan.

## Commands

DAC K=8:

```bash
python src/train.py objective=core_dac model=iag_srme_k8 ...
```

aWTA K=8 control:

```bash
python src/train.py objective=core_awta model=iag_srme_k8 ...
```

Hard-WTA K=8 control:

```bash
python src/train.py objective=core_hard_wta model=iag_srme_k8 ...
```

DAC without DPP:

```bash
python src/train.py objective=core_dac_no_dpp model=iag_srme_k8 ...
```

Dataset, backbone, experiment, output, and runtime overrides represented by `...` should be identical across matched runs.

## Verification evidence

Focused mathematical, objective, engine, architecture, and configuration suite:

```text
63 passed in 1.58s
```

Full suite:

```text
104 passed, 2 skipped in 1.82s
```

The tests explicitly establish:

- gradients of `1/8`, `1/4`, `1/2`, and `1` at stages 0–3;
- zero DAC-credit gradient outside the winning group;
- independent winning groups for rows in one batch;
- detached assignment and leaf equivalence to hard WTA;
- exact schedule boundaries through and beyond step 6000;
- invalid-row finite zero behavior;
- target detachment with four candidate-query recipients at stage 1;
- no global-step increment on a scaler-skipped optimizer update;
- K=8 ProposalNet, Grounder, Executor, retrieval-query, and ScoreNet shapes;
- one hard selected candidate per live row and a target-free inference signature.

## Scope and uncertainty

This is a paper-faithful DAC responsibility adaptation, not an exact reproduction of ALAN or the whole trajectory-prediction paper. The paper does not specify tie handling; first-index `argmin` is used. The paper says “iterations” while the repository uses optimizer updates; successful optimizer updates are the least ambiguous faithful mapping and prevent AMP-overflow skips from advancing the hierarchy.
