# TAPER-MAG V4 implementation notes

This branch is a clean implementation of
`CIR_TAPER_CANONICAL_END_TO_END_METHOD_SPEC_V4_2026-08-28.md` with the explicit
FG-CLIP2-Base/text-finetuning experiment overrides.

## Pinned backbone contract

- model: `qihoo360/fg-clip2-base`
- revision: `430fbc8a912c86fd4de601381b6245a0edab22f0`
- runtime-inspected text/vision/retrieval dimensions: 768/768/768
- runtime-inspected text/vision blocks: 12/12
- text traversal: `short`, max length 64, online during training
- default trainable text modules: blocks 8–11 plus final text LayerNorm
- default frozen text module: short-walk projection head (not used by TAPER token input)
- vision backbone, dense-feature head, official global image pooler: frozen
- patch policy: official dynamic budgets; dense tokens are ragged patch tokens with no CLS
- TAPER width/query count: 256/4; retrieval output: normalized 768-D

The runtime smoke observed hundreds of real tokens per image, so image caches are split into an
all-image normalized global store and an annotation-derived reference-only dense store. Both use
NumPy mmap (`global.npy`; `dense_values.npy` plus offsets/spatial shapes); batch lookup copies only
requested rows. Cache manifests explicitly type `complete_split` versus `reference_only`, and
partial debug caches are marked `complete_split=false`; full training rejects them.

## Primary FashionIQ validation protocol

The primary TAPER-MAG validation protocol on this branch is `fashioniq_val`, using the
ENCODER/OFFSET/MELT-style VAL-split convention:

1. the per-category gallery is the ordered, duplicate-free union of every reference ID and target
   ID appearing in that category's validation annotations;
2. each query's own reference image is excluded from that query's retrieval ranking;
3. R@1, R@10, R@50, target rank, and MRR are computed on the remaining gallery.

Category R@10/R@50 are macro-averaged, and Mean Recall is the mean of macro R@10 and macro R@50.
The resolved protocol and `reference_exclusion=true` are written to the run manifest, validation
metrics, functional reports and checkpoint metadata; resume rejects a protocol mismatch so scores
cannot be mixed in one checkpoint-selection history.

`fashioniq_original` remains an explicit, separate protocol whose gallery comes from
`split.<category>.val.json`. Scores from `fashioniq_val` and `fashioniq_original` are not directly
comparable without their protocol label. This scoped pass does not alter the separate M0–M3 control
configs.

## Scientific boundaries

There is no slot ownership, QASA, hard routing, entmax, primitive bank, semantic slot label,
independently writable global state, target input to policy, learned free STOP logit, or teacher
gradient. ASROA is not implemented or enabled.

Curriculum advancement is explicit in config (`actor_warmup`, `utility_shadow`, `critic_warmup`,
`dagger_t2`, `st_bridge`, `harden`) rather than automatic. Change stage/horizon/oracle mix only after
the preceding health gate has been reviewed. Hard non-ST rollout previews all candidates with the
actor graph detached, then recomputes only the selected transition from the same parent with
gradients. The ST bridge retains its live all-candidate surrogate.

## Exact-same-backbone controls

M0–M3 are parallel experiments, never curriculum stages and never imported into TAPER:

| ID | Control | Inputs | Composer parameters | Optimizer updates |
|---|---|---|---:|---:|
| M0 | reference-only | normalized reference global | 0 | 0 |
| M1 | text-only | online contextual content tokens | 592,128 | 4,260 |
| M2 | normalized scalar-gated sum | reference global + online text | 592,129 | 4,260 |
| M3 | gated MLP combiner | reference global + online text | 4,729,344 | 4,260 |

M1–M3 use the same last-four-block text tuning, frozen vision, official global cache, caption and
correction policy, bidirectional multi-positive InfoNCE, effective batch, AdamW family, evaluator
implementation and best-retrieval checkpoint rule as TAPER. Their explicitly configured gallery
protocol must be checked before numerical comparison. M0 has no text input and no trainable
parameters. M3 is one-shot and has no local state, candidate action, teacher, critic or STOP.

The full pinned text-tuning stratum has 28,353,024 trainable parameters. Thus M3 has 33,082,368
total trainable parameters; TAPER has 34,217,494 (4,348,163 actor, 1,516,307 utility, plus text).

## Canonical V4 curriculum

`training.curriculum_mode=canonical_v4` resolves centrally to:

| Epoch | Phase | T | Oracle mix | ST | Temperature | rho_gate | Exploration |
|---|---|---:|---:|---|---:|---:|---:|
| 1–8 | actor warm-up (uniform candidate mixture) | 1 | 0 | no | 1.0 | 0 | 0 |
| 9–14 | critic warm-up (utility-weighted soft mixture) | 1 | 0 | no | 1.0 | 0 | 0 |
| 15–26 | DAgger | 2 | 0.8→0.3 | no | 1.0 | 0 | 0 |
| 27–40 | ST bridge | 3 | 0.3→0 | yes | 1.0→0.5 | 0→0.25 | 0 |
| 41–46 | predicted T4 | 4 | 0 | no | 0.5 | 0.25 | 0.05 |
| 47–52 | predicted T4 decay | 4 | 0 | no | 0.5→0.25 | 0.25 | 0.05→0 |
| 53–60 | harden | 4 | 0 | no | 0.25 | 0.25 (inactive without ST) | 0 |

`rho_gate` implements V4 `grad_scale` only on ST utility weights. Upstream utility inputs remain
detached (`rho_up=0`). Hard non-ST phases retain detached K-preview plus selected-only
recomputation. The model and optimizer stream remain continuous; no actor module is frozen or
trained as a sequential submodel. `curriculum_mode=manual` remains available for controlled debug
runs.

The epoch ranges are reference boundaries, not automatic promotions. Canonical runs use
`health_gate_mode=manual_approval`; each boundary fails loudly until its named gate is listed in
`training.approved_health_gates`. `bypass_health_gates_for_smoke` is false by default and is only a
non-scientific test override. Gate approvals are checkpointed and cannot be removed on resume.
EMA is initialized from live TAPER plus trainable text weights at utility/critic warm-up, updated
after each optimizer step with decay 0.999, and temporarily installed only for validation.

`actor_warmup_passed` additionally requires a parameter-free teacher-shadow report generated from
the exact resumed epoch-8 checkpoint and current cache manifests. The audit writes
`teacher_shadow_report.json`, `functional_health.json`, `firewall_report.json`, and a bounded
`policy_trace_sampled.jsonl`; target IDs and teacher values are confined to the explicitly named
supervision-audit section. It does not call an optimizer. Configure the reviewed report path in
`training.teacher_shadow_report` before adding the approval.

Every training epoch refreshes `functional_health.json` with actor/candidate, regret distribution,
STOP, pairwise/calibration, response-rank, gradient-coverage, clipping, firewall and numerical
sections. Checkpoint selection is fixed: retrieval maximizes valid mean Recall; policy minimizes
mean regret with median-regret then Recall tie-breaks; functional selection first rejects
firewall/numerical/exact-collapse failures and then uses Recall and response rank. `last.ckpt` is
always updated. Resume is deterministic at epoch boundaries only; mid-epoch resume is rejected.

`functional_retrieval.json` evaluates dynamic and frozen-t0 policy on the same `fashioniq_val`
gallery, applies the same per-query reference exclusion to every functional variant, and reports
R@10, R@50, Mean Recall, target-rank mean/median, MRR and paired rank changes.
Repeat-best, mean-repeat, clone-all-best, clone-all-mean, operator-zero and operator-mean are causal
audit interventions: operator anchors are replaced before execution and every result passes through
the shared executor, recurrent local state and real readout. Query-space constructions such as
`q + n*delta_q` are not used as intervention results. Horizon-one dynamic/frozen state dependence
is explicitly marked not applicable.

## Entry points

```bash
PYTHONPATH=src python src/precompute_taper_mag_vision.py --dataset-root data/fashionIQ_dataset --cache-root features --split train
PYTHONPATH=src python src/precompute_taper_mag_vision.py --dataset-root data/fashionIQ_dataset --cache-root features --split val
pytest -q
PYTHONPATH=src python src/run_taper_control.py --config conf/taper_mag_v4_m0_reference_only.yaml
PYTHONPATH=src python src/run_taper_control.py --config conf/taper_mag_v4_m1_text_only.yaml
PYTHONPATH=src python src/run_taper_control.py --config conf/taper_mag_v4_m2_simple_sum.yaml
PYTHONPATH=src python src/run_taper_control.py --config conf/taper_mag_v4_one_shot_control.yaml
PYTHONPATH=src python src/train.py --config conf/taper_mag_v4_base.yaml
PYTHONPATH=src python src/train.py --config conf/taper_mag_v4_base.yaml --max-train-samples 32
PYTHONPATH=src python src/train.py --config conf/taper_mag_v4_base.yaml --resume runs/taper-mag-v4-fgclip2-base-last4/last.ckpt
PYTHONPATH=src python src/train.py --config conf/taper_mag_v4_base.yaml --resume runs/taper-mag-v4-fgclip2-base-last4/best_retrieval_valid.ckpt --validate-only
PYTHONPATH=src python src/train.py --config conf/taper_mag_v4_base.yaml --resume runs/taper-mag-v4-fgclip2-base-last4/last.ckpt --teacher-shadow-audit --audit-samples 256
PYTHONPATH=src python src/train.py --config conf/taper_mag_v4_base.yaml --resume runs/taper-mag-v4-fgclip2-base-last4/last.ckpt --profile-runtime --audit-samples 32
PYTHONPATH=src python src/train.py --config conf/taper_mag_v4_smoke.yaml --max-train-samples 32
PYTHONPATH=src python src/train_one_shot_control.py --config conf/taper_mag_v4_one_shot_control.yaml
```

The profiler writes `compute_report.json` with parameter counts, peak CUDA allocation/reservation
when available, text/operator/preview/teacher/backward timings, throughput, query latency, rollout
steps and STOP histogram. Optimizer-step timing uses a disposable AdamW instance and restores live
weights exactly, leaving the training optimizer untouched. FLOPs remain explicitly `null` when no
reliable tooling is available; values are never extrapolated.

The supplied FashionIQ tree does not currently contain the three configured
`correction_dict_{dress,shirt,toptee}.json` files. The main configs deliberately use the validated
`correction_policy=fashioniq` and therefore fail loudly until those exact dictionaries are supplied.
`correction_policy=none` is an explicit supported control, not a silent fallback.
