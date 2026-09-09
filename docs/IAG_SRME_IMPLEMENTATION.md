# IAG-SRME V2 Sequential R0 implementation map

This branch is a sequential-context-edit experiment derived from the historical V2 R0
candidate-selection implementation. Historical specifications and diagnostic reports under
`doc/` describe the source architecture; they are retained for provenance, not as the active
architecture contract for this branch.

```text
src/models/iag_srme/
├── __init__.py
├── model.py
└── utils/
    ├── __init__.py
    └── backbone.py
```

`model.py` contains the retained ProposalNet, Grounder, ActionFusion, Executor, and the
sequential `IAGSRME` forward:

```text
V0 → ProposalNet once → C0,C1,C2,C3
V0,C0 → V1
V1,C1 → V2
V2,C2 → V3
V3,C3 → V4
V4 → terminal retrieval query
```

The context edits remain fixed after their initial generation. Grounding, entity pooling,
action fusion, and execution are recomputed from the evolving state. All four transitions
reuse one Executor instance.

`utils/backbone.py` retains the verified FG-CLIP state, dense readout, global readout, and
retrieval interfaces. Both `learned_qg` and `native_cls` readout modes remain supported, as do
the `full` and `text_only` fine-tuning policies.

`src/losses/objective.py` applies only the bidirectional final-state retrieval loss. Stable
duplicate target identities remain multi-positive through `src/losses/retrieval.py`.
Per-slot state/action/mask norms are detached diagnostics and do not alter optimization.

The exported model forward accepts only reference images and instruction tensors. Official
FashionIQ evaluation consumes only `output["query"]`, which is read from final state `V4`.
