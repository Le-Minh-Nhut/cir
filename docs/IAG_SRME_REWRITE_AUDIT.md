# IAG-SRME Clean-Rewrite Audit

Source commit: `47be3fa` on `exp/e2e-competitive-null-slots`. This audit treats
`doc/CIR_TAPER_IAG_SRME_UNIFIED_MASTER_ARCHITECTURE_AND_LOSS_SPEC_V1_2026-08-29.md`
as the sole architecture and loss authority. The existing TAPER code is historical evidence,
not an implementation base.

## A. Files to KEEP

| Files | Reason |
|---|---|
| `.gitignore`, `LICENSE` | Repository metadata and license are architecture-independent. |
| `.idea/.gitignore`, `.idea/cir-research.iml`, `.idea/inspectionProfiles/Project_Default.xml`, `.idea/inspectionProfiles/profiles_settings.xml`, `.idea/modules.xml`, `.idea/vcs.xml` | User/project IDE metadata; unrelated to runtime architecture. |
| `doc/CIR_TAPER_IAG_SRME_UNIFIED_MASTER_ARCHITECTURE_AND_LOSS_SPEC_V1_2026-08-29.md` | Authoritative research specification and the only permitted historical prose match for old names. |
| `src/datasets/common.py` | Generic CIR sample/batch and image-store protocol; it already keeps model features outside the dataset. |
| `src/datasets/fashioniq.py` | Correct generic FashionIQ annotation parsing, stable IDs, caption composition, and protocol helpers. |
| `src/datasets/__init__.py` | Package marker. |
| `src/runtime.py` | Generic seeding, deterministic runtime, and device selection. |
| `src/evaluation/__init__.py` | Package marker. |
| `src/training/__init__.py` | Package marker. |
| `conf/dataset/fashioniq.yaml` | Dataset identity/root remain valid; it will be extended only with image layout. |
| `conf/protocol/fashioniq_original.yaml`, `conf/protocol/fashioniq_val.yaml` | Protocol names/splits are generic and correct. |

## B. Files to REWRITE

| Files | Reason |
|---|---|
| `README.md` | Replace historical bootstrap text with the clean IAG-SRME entry points and documentation links. |
| `pyproject.toml`, `requirements.txt` | Declare the actual FG-CLIP/transformer and entmax-compatible dependency contract and test tooling. |
| `conf/config.yaml` | Replace old model/objective defaults and stale cache semantics. |
| `src/models/__init__.py` | Export IAG-SRME only. |
| `src/train.py` | New raw-image, current-backbone, end-to-end training entry point. |
| `src/training/engine.py` | Generic IAG-SRME training/checkpoint loop with no cached model-specific representations. |
| `src/evaluation/fashioniq.py` | Preserve metric/protocol math, replace all model interaction and cached-feature assumptions. |

## C. Files to DELETE

| Files | Reason |
|---|---|
| `teacher/` (all adapters, audits, JSON cases, and provenance files) | Entire teacher-selection/audit subsystem is forbidden and has no role in IAG-SRME. |
| `src/teachers/` | Runtime CSMCIR adapters and compose path are forbidden. |
| `src/models/taper.py` | Monolithic old slots/primitives/router/gate/transition architecture; must not be retrofitted. |
| `src/cache/__init__.py`, `src/cache/features.py` | Cache schema is explicitly tied to native/retrieval/text states from the old model; no generic legal FG-CLIP cache contract exists here. |
| `src/check_csmcir_compose_parity.py` | Old compose parity. |
| `src/check_taper_chunk_parity.py`, `src/check_taper_text_cache_parity.py` | Old counterfactual/text-cache parity. |
| `src/diagnose_taper_checkpoint.py`, `src/forensic_taper_a3.py` | Old TAPER checkpoint diagnostics. |
| `src/precompute.py`, `src/precompute_csmcir_stage1.py`, `src/precompute_taper_e2e_text.py` | Old stale/model-specific feature paths. A future frozen-vision cache requires a new explicit namespace and manifest. |
| `src/probe_stage1_relations.py`, `src/train_stage1.py`, `src/evaluation/edit_slot_stage1.py` | Legacy Stage-1 and edit-slot experiments. |
| `conf/model/blip2_qure.yaml` | Wrong backbone and method. |
| `conf/objective/qure_pairwise.yaml` | Wrong objective. |
| `conf/experiment/default.yaml`, `conf/experiment/taper_e2e.yaml`, `conf/experiment/taper_stage1.yaml` | Historical experiment graphs and forbidden fields. |
| `conf/dataset/cirr.yaml`, `conf/protocol/cirr_val.yaml` | No implemented clean CIRR image/protocol pipeline in this rewrite; retaining it would advertise unsupported behavior. |
| `reports/taper_a3_forensic.json`, `reports/taper_a3_forensic_depth.json`, `reports/taper_checkpoint_diagnosis.json` | Artifacts for deleted historical architecture. |

## D. Teacher/CSMCIR dependency graph

Runtime/config files currently depending directly on teacher/CSMCIR are:

- `src/train.py` -> `src/teachers/csmcir_compose.py` -> external CSMCIR repository/checkpoint.
- `src/training/engine.py` -> `src/cache/features.py` -> cached teacher text/native/retrieval arrays.
- `src/evaluation/fashioniq.py` -> `src/cache/features.py` and model arguments carrying teacher reference/text states.
- `src/models/taper.py` -> injected compose object and teacher counterfactual composition.
- `src/train_stage1.py`, `src/evaluation/edit_slot_stage1.py`, `src/precompute_csmcir_stage1.py`, `src/precompute_taper_e2e_text.py`, parity/diagnostic/probe scripts -> teacher adapters or old cache schemas.
- `conf/experiment/taper_e2e.yaml`, `conf/experiment/taper_stage1.yaml` -> external repository roots, checkpoints, teacher dimensions, and old primitive/gate settings.
- `teacher/**`, `src/teachers/**` -> the teacher implementation/audit roots themselves.

All nodes in this graph will be removed or rewritten. No compatibility shim will remain.

## E. Proposed new package structure

```text
src/models/iag_srme/
  backbone.py       # FG-CLIP adapter, projections, freeze contract
  intent.py         # full-text learnable-query intents; optional claim head
  grounding.py      # independent entmax-1.5 anchor supports
  grounded_reader.py
  context.py
  editor.py         # shared bounded support-gated token editor
  readout.py        # bounded accumulated-change retrieval readout
  scorer.py         # shared target-free consequence scorer
  selector.py       # hard-forward ST action/STOP and absorbing state
  outputs.py        # typed trajectory structures
  model.py          # recurrence orchestration only
src/losses/
  retrieval.py
  marginal.py
  complementary_claim.py
  action_claim_binding.py
  factor.py
  unique.py
  objective.py
src/data/images.py
src/evaluation/fashioniq.py
src/training/engine.py
src/diagnostics/iag_srme.py
```

## F. Risks / unresolved specification points

- `K=4` proposal identities does not imply four true edits. Factor activity/semantic NULL is unresolved; STOP cannot serve that role. `L_unique` stays disabled by default and accepts explicit activity weights.
- Applying normalized claim consistency to inactive candidates may force artificial responsibilities; raw claim mass must be logged and `L_comp` remains optional.
- Relational factor objectives can learn secret-sharing or sample codes; they need held-out/intervention audits.
- Representation specialization does not imply distinct functional `delta_q`; matched-compute repeat/clone/effect-rank controls are required.
- The public FG-CLIP checkpoint API can vary. The adapter will use an explicit configured repository/checkpoint contract and fail clearly rather than substitute another model.
- Full vision fine-tuning forbids persistent image features. Frozen-vision caching is intentionally not implemented until a checkpoint/preprocessing manifest can make cache legality testable.
