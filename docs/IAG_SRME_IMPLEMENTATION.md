# IAG-SRME V2 implementation map

The canonical source of truth is
`doc/CIR_IAG_SRME_UNIFIED_CANONICAL_ARCHITECTURE_AND_TRAINING_SPEC_V2_2026-09-06.md`.

```text
src/models/iag_srme/
├── __init__.py
├── model.py
└── utils/
    ├── __init__.py
    ├── backbone.py
    ├── retrieval.py
    └── semantic.py
```

`model.py` contains `ProposalNet`, `Grounder`, `ActionFusion`, `Executor`,
`ScoreNet`, and the hard recurrent `IAGSRME` rollout in reading order.

`utils/backbone.py` contains the verified FG-CLIP v1 calls:

- `initial_state`: `hidden_states[-2][:, 1:]`;
- `dense_readout`: `forward_without_attn -> post_layernorm -> visual_projection`;
- `global_readout`: learned checkpoint-initialized `q_G` attending to current patches
  through the native final-block projections/LNs/MLP;
- `retrieval_readout`: the same global readout followed by native visual projection and
  normalization for current, candidate, and terminal states.

Dense parity with the official `get_image_dense_features` helper is exact. The q_G
readout is intentionally not claimed equal to the checkpoint's original
`get_image_features` output: that helper uses an image-dependent penultimate CLS token,
whereas V2 removes CLS from recurrent memory and requires one learned q_G. Its audited
contracts are native final-block parameter reuse, current-state sensitivity, and
vectorized/loop equality.

`utils/retrieval.py` builds stable-identity multi-positive masks and the detached FP32
common-pool marginal teacher. `utils/semantic.py` contains the deterministic,
versioned instruction-only concept parser and training-split vocabulary.
`src/losses/objective.py` combines terminal retrieval, confidence-weighted pairwise
ranking, absolute Huber gain calibration, set-level concept coverage, relation
binding/prototype regularization, and history-conditioned Functional DPP.

The exported model forward accepts only reference images and instruction tensors. It
returns a plain dictionary with terminal query/state, immutable initial state, STOP
status, and per-step research diagnostics.

All A0-A6 terms are independently switchable. The checked-in `objective=core` config
enables the A6 objective: `L_bind`, `L_c(t=0)`, relation-bank orthogonality, and
Functional DPP in `delta_q` consequence space. Correspondence is explicitly disabled.
The implementation record, measured checks, and open Track-B parity item are in
`doc/CIR_IAG_SRME_V2_R0_EXPERIMENT_IMPLEMENTATION_STATUS_2026-09-06.md`.
