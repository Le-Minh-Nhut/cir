# FINAL SCIENTIFIC COMPARABILITY ASSESSMENT: CIR FAILURE ANALYSIS ACROSS 16 MODELS

**Date:** 2026-10-18  
**Repository:** `/home/heheboiz/data/cir` (HEAD `b70d092`)  
**Target Path:** `workbench/docs/reviews/FINAL_CIR_SCIENTIFIC_COMPARABILITY.md`  
**Reference Codes:** `workbench/registry/models.yaml`, `workbench/backend/registry.py`, `src/evaluation/fashioniq.py`, `workbench/backend/storage.py`, `workbench/backend/analysis.py`, `workbench/scripts/evaluate_models.py`, `workbench/backend/schemas/results.py`.

---

## 1. PROTOCOL DEFINITIONS, GALLERY CONSTRUCTION & REFERENCE POLICY

FashionIQ evaluation in composed image retrieval (CIR) is divided into three mutually non-commensurable evaluation protocols. In `workbench/registry/models.yaml:7-19`, the registry specifies:

```yaml
protocols:
  fashioniq_original_split:
    label: FashionIQ - Original Split (Full Gallery, Reference Included)
    literature_split_label: original
    evaluation_noise_pct: 0
  fashioniq_full_gallery_ref_excluded:
    label: FashionIQ - Original Full Gallery, Reference Excluded
    literature_split_label: original
    evaluation_noise_pct: 0
  fashioniq_val_split:
    label: FashionIQ - Val Split (Pair-Union Gallery, Reference Excluded)
    literature_split_label: val
    evaluation_noise_pct: 0
```

### 1.1 Executable Protocol Implementation

The gallery construction and candidate eligibility mechanics are implemented in `src/evaluation/fashioniq.py:155-220`:

```python
@dataclass(frozen=True, slots=True)
class FashionIQProtocol:
    """One FashionIQ evaluation cohort, defined by what a comparison depends on.

    ``gallery`` and ``reference_excluded`` are the two properties that change a target's rank,
    so they are declared together rather than inferred from caller arguments. Every entry point
    resolves a protocol through :data:`FASHIONIQ_PROTOCOLS`; an unknown name is an error, never a
    silent fallback to reference-eligible evaluation.
    """

    protocol_id: str
    gallery: str            # "full_split" | "pair_union"
    reference_excluded: bool
    aliases: tuple[str, ...] = ()


FASHIONIQ_PROTOCOLS: dict[str, FashionIQProtocol] = {
    "fashioniq_original_split": FashionIQProtocol(
        protocol_id="fashioniq_original_split", gallery="full_split",
        reference_excluded=False, aliases=("fashioniq_original",)),
    "fashioniq_full_gallery_ref_excluded": FashionIQProtocol(
        protocol_id="fashioniq_full_gallery_ref_excluded", gallery="full_split",
        reference_excluded=True),
    "fashioniq_val_split": FashionIQProtocol(
        protocol_id="fashioniq_val_split", gallery="pair_union",
        reference_excluded=True, aliases=("fashioniq_val",)),
}


def build_fashioniq_gallery(protocol: str, split_root: str | Path, category: str, annotations: Sequence[FashionIQAnnotation], split: str,) -> list[str]:
    """Build the candidate universe for one protocol.

    Both the registry protocol ids and the shorter internal aliases resolve here, so a caller
    holding either spelling gets the same cohort. Gallery construction follows the protocol
    table, never a caller argument.
    """
    resolved = resolve_fashioniq_protocol(protocol)

    if resolved.gallery == "full_split":
        # The whole validation split in its declared order. Reference exclusion is a per-query
        # property applied at recall time, not a change to the candidate universe: the gallery
        # is what the upstream evaluator scores against.
        return build_original_gallery(split_root=split_root, category=category, split=split)

    return build_val_gallery(annotations)
```

The candidate retrieval ranking mechanics for reference-included vs reference-excluded protocols are defined in `src/evaluation/fashioniq.py:9-77`:

```python
def recall_at_k(scores: torch.Tensor, target_ids: Sequence[str], gallery_ids: Sequence[str], k: int) -> float:
    ...
    rankings = torch.argsort(scores, dim=1, descending=True,)
    top_k = rankings[:, :k]
    target_tensor = torch.tensor(target_indices, device=rankings.device,).unsqueeze(1) 
    hits_per_query = top_k.eq(target_tensor).any(dim=1)
    return hits_per_query.float().mean().item() * 100.0

def recall_at_k_reference_excluded(
    scores: torch.Tensor,
    target_ids: Sequence[str],
    reference_ids: Sequence[str],
    gallery_ids: Sequence[str],
    k: int,
) -> float:
    """Recall@k with each query's own reference removed from the ranked gallery.

    This is the iLearn full-gallery protocol. The candidate universe is the whole validation
    split in its declared order, and a query cannot retrieve its own reference.

    The pinned evaluators *mask* the reference score to a very negative value and then sort
    (`sims[i, targets.index(source)] = -10e10` before `argsort`). Removing the column and masking
    it are equivalent for the top-k hit test whenever all other scores exceed the mask, but the
    two are not the same operation, and the mask depends on that bound holding. This helper
    removes the column explicitly, which is what the *cohort* means, and the tests pin the
    behaviour rather than relying on the equivalence.
    """
    ...
    for row, (target_id, reference_id) in enumerate(zip(target_ids, reference_ids)):
        ...
        reference_column = gallery_index[reference_id]
        ranking = torch.argsort(scores[row], descending=True)
        ranking = ranking[ranking != reference_column]
        target_column = gallery_index[target_id]
        hits.append(bool(ranking[: min(k, len(ranking))].eq(target_column).any()))

    return float(torch.tensor(hits, dtype=torch.float32).mean().item() * 100.0)
```

### 1.2 Protocol Resolution & Semantic Enforcement in `workbench/backend/registry.py`

In `workbench/backend/registry.py:40-78`, protocol resolution enforces that every model's native protocol and supported protocols belong to the closed set, match the literature split label, and cannot be conditionally asserted:

```python
    for model in registry["models"]:
        native_protocol = model["native_protocol"]
        supported_protocols = model["supported_protocols"]
        if native_protocol is not None and native_protocol not in registry["protocols"]:
            raise ValueError(f"unknown protocol for {model['model_id']}")
        if any(protocol not in registry["protocols"] for protocol in supported_protocols):
            raise ValueError(f"unknown supported protocol for {model['model_id']}")
        if native_protocol is not None and native_protocol not in supported_protocols:
            raise ValueError(f"native protocol unsupported for {model['model_id']}")
        if model.get("literature_split_label") not in {"original", "val", None}:
            raise ValueError(f"unknown literature split label for {model['model_id']}")
        if native_protocol is not None and model.get("literature_split_label") != registry["protocols"][native_protocol]["literature_split_label"]:
            raise ValueError(f"literature split label mismatch for {model['model_id']}")
        ...
        if model.get("native_protocol_condition") and native_protocol is not None:
            raise ValueError(f"conditional native protocol must remain unavailable for {model['model_id']}")
```

**Meaning of `native_protocol: null`:**  
`native_protocol: null` designates a registered model whose evaluation protocol cannot be resolved from pinned source code or published paper evidence. For example, `ptha_mtst` (`models.yaml:234-235`) has `native_protocol: null` and `supported_protocols: []`.  
As documented in `docker/acceptance.sh:163-164`:
> `# A registry row may legitimately declare NO runnable protocol (ptha_mtst is a source_blocker row), so the check is equality with the declaration, not presence.`  

When `native_protocol` is `null`, execution is unconditionally rejected by `workbench/scripts/evaluate_models.py:2304-2305`:
```python
        protocol = args.protocol or model["native_protocol"]
        if protocol is None:
            print("[BLOCKED] protocol unavailable"); failed = True; continue
```

### 1.3 Mathematical Proof of Rank Divergence Across Protocols

In `workbench/tests/test_limn_query_export_protocols.py:7-16`, the three protocols are proven to generate three conflicting target ranks for the identical query and similarity scores:

```
  full split:  ["imgA", "imgB", "imgZ", "imgC", "imgD"]   (imgZ is a distractor no annotation uses)
  query:       reference imgA, target imgB
  scores:      imgA 0.9 (the reference), imgZ 0.8 (a stranger), imgB 0.7 (the target)

  Protocol A  reference ELIGIBLE, full gallery -> imgA, imgZ, imgB -> target rank 3
  Protocol B  reference EXCLUDED, full gallery -> imgZ, imgB       -> target rank 2
  Protocol C  reference EXCLUDED, pair-union   -> imgB (imgZ absent) -> target rank 1
```

Because candidate universe size and reference masking differ fundamentally, cross-protocol rank comparisons and metrics are non-commensurable.

---

## 2. PROTOCOL MEMBERSHIP TABLE (ALL 16 MODELS)

The 16 models in `workbench/registry/models.yaml` map to the 3 protocols and 1 unresolved state:

| Model ID | Method Name | Research Generation | Pub. Year | Native Protocol | Literature Label | Reported R@10 | Reported R@50 | Reported Mean | Category Policy | Per-Query Observer | Registry Citation |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **csmcir** | CSMCIR | recent | null | `fashioniq_original_split` (A) | original | 57.07 | 77.27 | 67.17 | shared (single) | None | `models.yaml:21-60` |
| **airknow** | Air-Know | recent | 2026 | `fashioniq_full_gallery_ref_excluded` (B) | original | 55.75 | 75.72 | 65.73 | shared (single) | None | `models.yaml:97-125` |
| **conesep** | ConeSep | recent | 2026 | `fashioniq_full_gallery_ref_excluded` (B) | original | 55.58 | 75.88 | 65.73 | shared (single) | None | `models.yaml:150-182` |
| **ptha_mtst** | PTHA + MTST | recent | 2025 | `null` (Unresolved) | null | 54.99 | 76.28 | 65.64 | null | None | `models.yaml:217-238` |
| **habit** | HABIT | recent | 2026 | `fashioniq_full_gallery_ref_excluded` (B) | original | 55.38 | 75.20 | 65.29 | shared (single) | None | `models.yaml:243-270` |
| **intent** | INTENT | recent | 2026 | `fashioniq_full_gallery_ref_excluded` (B) | original | 55.31 | 75.16 | 65.24 | shared (single) | None | `models.yaml:305-333` |
| **hint** | HINT | recent | 2026 | `fashioniq_val_split` (C) | val | 58.48 | 77.93 | 68.21 | shared (single) | None | `models.yaml:368-395` |
| **encoder** | ENCODER | recent | 2025 | `fashioniq_val_split` (C) | val | 56.13 | 77.59 | 66.86 | `shared` | None | `models.yaml:415-449` |
| **pair** | PAIR | recent | 2025 | `fashioniq_val_split` (C) | val | 52.76 | 74.51 | 63.64 | shared (single) | None | `models.yaml:477-505` |
| **clvc_net** | CLVC-Net | legacy | 2021 | `fashioniq_val_split` (C) | val | 30.70 | 58.41 | 44.56 | `independent` | None | `models.yaml:538-577` |
| **dcnet** | DCNet | legacy | 2021 | `fashioniq_full_gallery_ref_excluded` (B) | original | null | null | 40.83 | `shared` | None | `models.yaml:606-633` |
| **combiner_rn50x4_noft** | Combiner RN50x4 noft | legacy | 2023 | `fashioniq_original_split` (A) | original | null | null | null | `shared` | None | `models.yaml:660-690` |
| **clip4cir_rn50x4_fullft** | CLIP4Cir RN50x4 fullft | legacy | 2023 | `fashioniq_original_split` (A) | original | 43.78 | 66.93 | 55.36 | `shared` | None | `models.yaml:721-756` |
| **tgcir** | TG-CIR | legacy | 2023 | `fashioniq_val_split` (C) | val | 51.32 | 73.09 | null | `shared` | None | `models.yaml:785-810` |
| **sprc** | SPRC | legacy | 2024 | `fashioniq_original_split` (A) | original | 54.92 | 74.97 | 64.85 | `shared` | None | `models.yaml:843-872` |
| **limn** | LIMN | legacy | 2024 | `fashioniq_val_split` (C) | val | 55.91 | 77.82 | 66.87 | `independent` | `LIMN_QUERY_OBSERVER_V1` | `models.yaml:900-932` |

---

## 3. PER-MODEL AUDIT: AGGREGATE METRICS VS. PER-QUERY CAPABILITIES

### 3.1 CSMCIR (`csmcir`)
- **Native Aggregate Metric:** Present. `reported_r10: 57.07`, `reported_r50: 77.27`, `reported_mean: 67.17` (`models.yaml:58-60`). Evaluator `CSMCIR/src/validate_blip_csmcir.py:compute_fiq_val_metrics` calculates per-category Recall@10/@50 and mean.
- **Per-Query Results Today:** **NO**. `per_query_observer: null` (`models.yaml:21-63`).
- **Code Requirements to Produce Per-Query Data:**
  1. `CSMCIR/src/validate_blip_csmcir.py:compute_fiq_val_metrics` computes Q-Former similarity tensor `sims` across query and full gallery (`FashionIQDataset`), ranking by `torch.argsort(1 - score)` without masking reference.
  2. A dedicated observer module must intercept `sims` before reduction, capture per-query `top_results` (up to Top-K depth), record canonical FashionIQ query IDs (`category:index:reference:target`), and compute target ranks.
  3. Preflight runtime blockers must be resolved: environment CUDA 11.7 / torch 2.0.1 wheel dependencies, missing LAVIS package imports (`models.yaml:63`), image filename extension `.png` requirement (`models.yaml:54`), and Qwen/COT auxiliary asset verification (`models.yaml:37`).
  4. Cross-check observer Recall with native `compute_fiq_val_metrics` hit counts for exact parity.

### 3.2 Air-Know (`airknow`)
- **Native Aggregate Metric:** Present. `reported_r10: 55.75`, `reported_r50: 75.72`, `reported_mean: 65.73` (`models.yaml:123-125`). Evaluator `AirKnow/test_BLIP2.py:test` calculates category recalls (`:169-175`).
- **Per-Query Results Today:** **NO**. `per_query_observer: null` (`models.yaml:97-125`).
- **Code Requirements to Produce Per-Query Data:**
  1. Pinned `AirKnow/test_BLIP2.py:test` is a bare function without an entry point (`__main__`) or CLI parser (`models.yaml:98`); `AirKnowAdapter` in `workbench/backend/adapters/models.py` has no implemented command constructor. A standalone CLI runner must be implemented.
  2. Checkpoint availability: Only noise-trained checkpoints (`fiq_n05`, `fiq_n08`) exist; the paper's 0%-noise clean checkpoint is unpublished (`models.yaml:98`).
  3. Upstream `test_BLIP2.py:151` applies `-10e10` reference mask to `sims[i, test_targets_id.index(t['source_img_id'])]`. Observer must extract candidate rankings post-mask, resolve image paths from `resized_image/{category}`, serialize `ResultRun`, and verify hit-count parity.

### 3.3 ConeSep (`conesep`)
- **Native Aggregate Metric:** Present. `reported_r10: 55.58`, `reported_r50: 75.88`, `reported_mean: 65.73` (`models.yaml:180-182`). Evaluator `ConeSep/test.py:test` computes category recall.
- **Per-Query Results Today:** **NO**. `per_query_observer: null` (`models.yaml:150-182`).
- **Code Requirements to Produce Per-Query Data:**
  1. `ConeSep/test.py` is a callable without a CLI; `ConeSepAdapter` has no implemented command; checkpoint variants are noise-trained (`fiq_n02`, `fiq_n05`, `fiq_n08`).
  2. Implement execution harness to instantiate `model`, load checkpoint, and call `test.py`.
  3. Extract full ranking from `sims` where `test.py:68` sets reference mask `sims[i, test_targets_id.index(t['source_img_id'])] = -10e10`.
  4. Stage required category correction dictionaries and resized images; verify instrumentation parity.

### 3.4 PTHA + MTST (`ptha_mtst`)
- **Native Aggregate Metric:** Present only as paper text. `reported_r10: 54.99`, `reported_r50: 76.28`, `reported_mean: 65.64` (`models.yaml:236-238`), transcribed in `workbench/registry/paper_benchmarks/extracted/ptha_mtst_scaleup_mtst.json`.
- **Per-Query Results Today:** **NO**. Complete blocker: `source_available: false`, `source_dir: null`, `native_protocol: null`, `checkpoint_variants: []` (`models.yaml:231-242`).
- **Code Requirements to Produce Per-Query Data:**
  1. Author repository `https://github.com/zyinan99/MTST_PTHA` exists but is not synced, pinned, or audited (`models.yaml:224-227`). Must clone, establish pinned SHA, and add source tree.
  2. Obtain and verify fine-tuned FashionIQ checkpoint (none verified currently; paper reports only).
  3. Identify gallery construction and reference policy from code to assign protocol.
  4. Write adapter CLI, evaluation pipeline, and per-query observer.

### 3.5 HABIT (`habit`)
- **Native Aggregate Metric:** Present. `reported_r10: 55.38`, `reported_r50: 75.20`, `reported_mean: 65.29` (`models.yaml:268-270`). Evaluator `HABIT/test.py:test` computes category recalls.
- **Per-Query Results Today:** **NO**. `per_query_observer: null` (`models.yaml:243-270`).
- **Code Requirements to Produce Per-Query Data:**
  1. Pinned `HABIT/test.py` lacks a CLI parser; `HabitAdapter` command is unimplemented; checkpoints are noise-trained variants (`fiq_n02`, `fiq_n05`, `fiq_n08`).
  2. Implement replay wrapper loading weights and calling `test.py`.
  3. Capture ranking from `sims` before `argsort` (`test.py:151` sets `sims[i, test_targets_id.index(t['source_img_id'])] = -10e10`).
  4. Resolve external BERT tokenizer (`/root/bert-base-uncased`) and LAVIS base weights; verify parity.

### 3.6 INTENT (`intent`)
- **Native Aggregate Metric:** Present. `reported_r10: 55.31`, `reported_r50: 75.16`, `reported_mean: 65.24` (`models.yaml:331-333`). Evaluator `INTENT/test.py:test` computes category recall.
- **Per-Query Results Today:** **NO**. `per_query_observer: null` (`models.yaml:305-333`).
- **Code Requirements to Produce Per-Query Data:**
  1. `INTENT/test.py` is a callable function without CLI entry; `IntentAdapter` command is unimplemented; variants are noise-trained (`fiq_n02`, `fiq_n05`, `fiq_n08`).
  2. Create CLI loader, load checkpoint, hook `test.py:150` reference mask `-10e10`, export `top_results` and `target_rank`.
  3. Validate hit count parity against `test.py` return metrics.

### 3.7 HINT (`hint`)
- **Native Aggregate Metric:** Present. `reported_r10: 58.48`, `reported_r50: 77.93`, `reported_mean: 68.21` (`models.yaml:393-395`). Evaluator `HINT/test.py:test` computes category recall.
- **Per-Query Results Today:** **NO**. `per_query_observer: null` (`models.yaml:368-395`).
- **Code Requirements to Produce Per-Query Data:**
  1. Upstream `HINT/test.py:test` has no `__main__` entry point; `evaluate_models.py` marks `command=not audited`.
  2. Implement evaluation runner that initializes `HINT` model, loads `fashioniq` checkpoint, and calls `test()`.
  3. Extract similarity rankings across the pair-union gallery (`datasets.py:154-166`), with reference score masked.
  4. Export canonical `QueryResult` records and cross-check hit counts.

### 3.8 ENCODER (`encoder`)
- **Native Aggregate Metric:** Present. `reported_r10: 56.13`, `reported_r50: 77.59`, `reported_mean: 66.86` (`models.yaml:447-449`). Evaluator `ENCODER/test.py:test` computes category recalls.
- **Per-Query Results Today:** **NO**. `per_query_observer: null` (`models.yaml:415-449`).
- **Code Requirements to Produce Per-Query Data:**
  1. Upstream packaging blocker: `ENCODER/evaluate_model.py` attempts to import missing `datasets1.py`. Packaging defect must be resolved.
  2. Base OpenCLIP binary `open_clip_pytorch_model.bin` and verified `fashioniq.pt` checkpoint must be resolved and staged.
  3. Intercept `sims` in `ENCODER/test.py:39-49` (pair-union gallery with reference masked to `-10e10`), save top-K candidates, and verify parity.

### 3.9 PAIR (`pair`)
- **Native Aggregate Metric:** Present. `reported_r10: 52.76`, `reported_r50: 74.51`, `reported_mean: 63.64` (`models.yaml:503-505`). Evaluator `PAIR/test.py:test` computes category recall.
- **Per-Query Results Today:** **NO**. `per_query_observer: null` (`models.yaml:477-505`).
- **Code Requirements to Produce Per-Query Data:**
  1. Checkpoint mapping is `UNVERIFIED`; `PairAdapter` command is unimplemented. Pinned tree has no evaluator CLI.
  2. Verify author FashionIQ checkpoint and acquire base OpenCLIP weights.
  3. Implement CLI evaluator, extract `sims` from `PAIR/test.py:test`, record top-K rankings with reference excluded, and cross-check parity.

### 3.10 CLVC-Net (`clvc_net`)
- **Native Aggregate Metric:** Present. `reported_r10: 30.70`, `reported_r50: 58.41`, `reported_mean: 44.56` (`models.yaml:575-577`). Evaluator `CLVCNet/test.py:test` computes category R@10/R@50.
- **Per-Query Results Today:** **NO**. `per_query_observer: null` (`models.yaml:538-577`).
- **Code Requirements to Produce Per-Query Data:**
  1. `category_model_policy: independent` (`models.yaml:545`): CLVC-Net requires separate models for dress, shirt, and toptee. Author Drive checkpoint bundle is unresolved (`author_drive_unresolved`).
  2. Pinned source has callable `test()` only; `CLVCNetAdapter` raises `NotImplementedError`.
  3. Implement bundle evaluator, extract dot-product similarities across pair-union gallery, map query rankings, and verify parity against native hit counts.

### 3.11 DCNet (`dcnet`)
- **Native Aggregate Metric:** Only mean present in registry. `reported_mean: 40.83` (`models.yaml:633`), `reported_r10: null`, `reported_r50: null`. Upstream `DCNet/trainer/trainer_joint.py:TrainerJoint._valid_epoch` computes category recalls.
- **Per-Query Results Today:** **NO**. `per_query_observer: null` (`models.yaml:606-633`).
- **Code Requirements to Produce Per-Query Data:**
  1. `DCNetAdapter` routes execution via `python test.py --resume <run_directory>`, calling `TrainerJoint._valid_epoch`.
  2. In `TrainerJoint._valid_epoch` (`trainer/trainer_joint.py:216-218`), capture dual-branch fused scores (`F.log_softmax(score_comp) + F.log_softmax(score_corr)`) where reference score is masked.
  3. Extract query rankings across the full split, serialize `top_results` and `target_rank`, and cross-check hit counts.
  4. Verify author Drive checkpoint `config.json` and `trained_model.pth`.

### 3.12 Combiner RN50x4 noft (`combiner_rn50x4_noft`)
- **Native Aggregate Metric:** None declared in registry. `reported_r10: null`, `reported_r50: null`, `reported_mean: null` (`models.yaml:688-690`). Pinned evaluator `src/validate.py:validate_fiq` in `CLIP4Cir` computes Recall@10/@50.
- **Per-Query Results Today:** **NO**. `per_query_observer: null` (`models.yaml:660-690`).
- **Code Requirements to Produce Per-Query Data:**
  1. Resolve unverified checkpoint `rn50x4_noft_unresolved`.
  2. Hook cosine similarity matrix in `src/validate.py:validate_fiq` (full gallery, reference eligible).
  3. Serialize query rankings to canonical schema; verify parity against native `validate_fiq`.

### 3.13 CLIP4Cir RN50x4 fullft (`clip4cir_rn50x4_fullft`)
- **Native Aggregate Metric:** Present. `reported_r10: 43.78`, `reported_r50: 66.93`, `reported_mean: 55.36` (`models.yaml:754-756`). Pinned evaluator `src/validate.py:validate_fiq` in `CLIP4Cir` computes category recalls.
- **Per-Query Results Today:** **NO**. `per_query_observer: null` (`models.yaml:721-756`).
- **Code Requirements to Produce Per-Query Data:**
  1. Resolve unverified checkpoint `rn50x4_fullft_unresolved`.
  2. Intercept similarity matrix in `validate_fiq` (full split gallery, reference retained as candidate).
  3. Export top-K retrieved images and target ranks; verify parity against native hit counts.

### 3.14 TG-CIR (`tgcir`)
- **Native Aggregate Metric:** Present. `reported_r10: 51.32`, `reported_r50: 73.09`, `reported_mean: null` (`models.yaml:808-810`). Evaluator `TGCIR/test.py:test` computes category recall.
- **Per-Query Results Today:** **NO**. `per_query_observer: null` (`models.yaml:785-810`). `replay_status: PREFLIGHT_ONLY`.
- **Code Requirements to Produce Per-Query Data:**
  1. Checkpoint `tgcir_archive_unresolved` must be resolved.
  2. `TGCIR/test.py` is a bare function; `workbench/replay/tgcir.py` is preflight-only. Implement evaluation runner.
  3. Extract similarity rankings across pair-union gallery, excluding reference. Verify hit count parity.

### 3.15 SPRC (`sprc`)
- **Native Aggregate Metric:** Present. `reported_r10: 54.92`, `reported_r50: 74.97`, `reported_mean: 64.85` (`models.yaml:870-872`). Evaluator `SPRC/src/validate_blip_sprc.py:blip_validate_fashioniq` computes category recalls.
- **Per-Query Results Today:** **NO**. `per_query_observer: null` (`models.yaml:843-872`).
- **Code Requirements to Produce Per-Query Data:**
  1. Resolve unverified checkpoint `sprc_fiq_unresolved` and model class ambiguity (`blip2_qformer_cir_rerank` vs `align_prompt`).
  2. In `blip_validate_fashioniq`, intercept query-to-gallery similarity matrix (full split, reference eligible).
  3. Export top-K rankings and target ranks; verify parity against native hits.

### 3.16 LIMN (`limn`)
- **Native Aggregate Metric:** Present. `reported_r10: 55.91`, `reported_r50: 77.82`, `reported_mean: 66.87` (`models.yaml:930-932`). Pinned `LIMN/test.py:test` computes category recalls.
- **Per-Query Results Today:** **YES**. Registered observer: `per_query_observer: LIMN_QUERY_OBSERVER_V1` (`models.yaml:904`).
- **Implementation in Code:**
  1. `workbench/replay/limn_observer.py`: Extracts per-query rankings using the native model (`compose_query`, `compose_target`) and dataset (`test_queries`, `test_targets`), applies native mask `-10e10`, maps integer positions to canonical image IDs, and records `ObservedQuery`.
  2. `workbench/replay/limn_export.py:observe_and_verify`: Executes native `test.py:test` alongside `observe_category()`, calling `cross_check_with_native()` for strict hit-count parity.
  3. `workbench/replay/limn_publish.py`: Validates schema, writes atomic output, and verifies disk reload.
- **Summary:** LIMN is the **only model among all 16** with an implemented per-query observer and publication pipeline.

---

## 4. COMPARABILITY RULES & ENFORCEMENT CALLSITES

### 4.1 The Fundamental Comparability Invariant
Two runs or models may only be compared if:
1. They share the **exact same FashionIQ protocol ID** (`fashioniq_original_split`, `fashioniq_full_gallery_ref_excluded`, or `fashioniq_val_split`).
2. They share the same `data_kind` (`experiment` runs cannot be compared with `mock` runs).
3. At the query level, they possess an **identical query universe** (same count, same IDs, identical reference/target pairs).

### 4.2 Code Enforcement: Refusal Quotations

#### Refusal 1: Cross-Protocol Analysis Selection
`workbench/backend/storage.py:174-178` (`validate_analysis_selection`):
```python
def validate_analysis_selection(run_ids: list[str], *, k: int | None = None, require_top_k: bool = True, database_path: Path | None = None) -> tuple[list[dict[str, Any]], dict[str, dict[str, dict[str, Any]]]]:
    runs = get_runs(run_ids, database_path)
    protocols = {run["protocol_id"] for run in runs}
    if len(protocols) != 1:
        raise WorkbenchError("cross_protocol", "Cross-protocol analysis is invalid. Select runs from one FashionIQ protocol.", {"protocol_ids": sorted(protocols)})
```

#### Refusal 2: Query Alignment Enforcement
`workbench/backend/analysis.py:11-15` (`require_same_protocol`):
```python
def require_same_protocol(runs: list[ResultRun]) -> str:
    protocols = {run.run.protocol_id for run in runs}
    if len(protocols) != 1:
        raise WorkbenchError("cross_protocol", "Cross-protocol analysis is invalid. Select runs from one FashionIQ protocol.", {"protocol_ids": sorted(protocols)})
    return protocols.pop()
```

#### Refusal 3: Cohort Creation Mismatch
`workbench/backend/main.py:968-971` (`create_saved_cohort`):
```python
    runs, maps = validate_analysis_selection(payload.run_ids)
    protocol = runs[0]["protocol_id"]
    if protocol != payload.protocol_id:
        raise HTTPException(400, {"error": "cohort_protocol_mismatch", "message": "Cohort protocol does not match selected runs.", "details": {"payload_protocol_id": payload.protocol_id, "runs_protocol_id": protocol}})
```

#### Refusal 4: Paper Gate & Benchmark Reference Matching
`workbench/backend/benchmark_references.py:283-293` (`select_reference`):
```python
        if protocol_id is not None:
            recorded = record.get("gallery_protocol")
            if recorded is None:
                # A row whose protocol nobody established does NOT match every protocol. Treating
                # an absent value as a wildcard let a paper that states no gallery construction
                # bind to any run -- an identity asserted by default rather than by evidence. The
                # row is still reported; it simply cannot be chosen as the reference.
                return False
            if recorded != protocol_id:
                return False
        return True
```
And `workbench/backend/paper_gate.py:287-288`:
```python
    if "protocol_id" in joined or "split" in joined or "dataset" in joined:
        return BLOCKED_PROTOCOL_MISMATCH
```

#### Refusal 5: Runtime Plan and Adapter Protocol Compatibility
`workbench/backend/runtime.py:2490-2491` (`runtime_blockers`):
```python
    if protocol is not None and protocol not in model.get("supported_protocols", []):
        blockers.append(f"protocol incompatible: {protocol}")
```
And `workbench/backend/adapters/base.py:28-29` (`validate_request`):
```python
    def validate_request(self, request: EvalRequest) -> dict[str, Any]:
        if request.protocol_id not in self.model["supported_protocols"]:
            raise ValueError(f"{self.model_id} does not support {request.protocol_id}")
```

#### Refusal 6: Evaluation Invocation Main Loop
`workbench/scripts/evaluate_models.py:2302-2306`:
```python
    for model, checkpoint in items:
        protocol = args.protocol or model["native_protocol"]
        print(f"[RUN] {model['model_id']} / {checkpoint['checkpoint_id']}")
        if protocol is None:
            print("[BLOCKED] protocol unavailable"); failed = True; continue
        plan, reasons = guarded_plan(model, checkpoint, protocol, dataset_root, args.top_k, config, dry_run=args.dry_run)
```

### 4.3 Legally Comparable Model Pairs
- **Protocol A Pairs (Aggregate Only):** Any pair within `{csmcir, combiner_rn50x4_noft, clip4cir_rn50x4_fullft, sprc}` can be compared on reported aggregate metrics (except combiner has null values). Per-query comparisons are currently impossible.
- **Protocol B Pairs (Aggregate Only):** Any pair within `{airknow, conesep, habit, intent, dcnet}` can be compared on reported aggregate metrics. Per-query comparisons are currently impossible.
- **Protocol C Pairs:** Any pair within `{hint, encoder, pair, clvc_net, tgcir, limn}` can be compared on reported aggregate metrics. At the per-query level, only `limn` can produce data; since pairwise comparison requires $\ge 2$ runs, zero cross-model comparisons are runnable today.
- **Cross-Protocol Pairs (e.g. `limn` vs `csmcir`, `limn` vs `habit`):** **REFUSED BY POLICY AND API**.

---

## 5. RESULT SCHEMA ADMISSION REQUIREMENTS (`results.py::admission`)

To be admitted into DuckDB and analysis endpoints, a per-query result artifact must satisfy `workbench/backend/schemas/results.py:240-371`.

### 5.1 Admission Rules & Verification States
The admission states are:
- `SYNTHETIC_OR_MOCK`: Admitted as fixture data, never research evidence.
- `PER_QUERY_UNVERIFIED`: Provenance incomplete or awaiting ledger corroboration.
- `PER_QUERY_PARITY_PENDING`: Parity not recorded.
- `PER_QUERY_PARITY_FAILED`: Observed ranking failed hit count parity.
- `PER_QUERY_VERIFIED`: Corroborated by HMAC evidence ledger and completion proof.

### 5.2 Specific Checks in `admission(result: ResultRun)`:
1. **Experiment Data Kind (`results.py:254-266`):**
   ```python
   if run.data_kind == "mock":
       return SYNTHETIC_OR_MOCK, [f"data_kind is {run.data_kind!r}: development data, never evidence"]
   if run.data_kind != "experiment":
       return PER_QUERY_UNVERIFIED, [
           f"data_kind is {run.data_kind!r}: a declared self-test, not evidence about reproduction"]
   ```
2. **Category Coverage (`results.py:272-276`):**
   Must cover all 3 categories: `FASHIONIQ_CATEGORIES = ("dress", "shirt", "toptee")`:
   ```python
   present = {query.category for query in result.queries}
   if present != set(FASHIONIQ_CATEGORIES):
       missing_categories = sorted(set(FASHIONIQ_CATEGORIES) - present)
       reasons.append(
           f"an experiment must cover all three FashionIQ categories; missing: {missing_categories}")
   ```
3. **Category Checkpoint Integrity (`results.py:280-288`):**
   ```python
   if run.category_checkpoint_ids is not None:
       values = list(run.category_checkpoint_ids.values())
       if len(set(values)) != len(values):
           reasons.append("category_checkpoint_ids reuses one checkpoint for more than one "
                          "category; the categories are independently trained models")
       unmapped = sorted({query.category for query in result.queries}
                         - set(run.category_checkpoint_ids))
       if unmapped:
           reasons.append(f"category_checkpoint_ids is missing categories present in queries: {unmapped}")
   ```
4. **Provenance Records (`results.py:293-298`):**
   Must contain `command_digest` and `environment_digest`, plus `checkpoint_sha256` or `bundle_manifest_digest`:
   ```python
   missing = [field for field in EXPERIMENT_REQUIRED_PROVENANCE if not getattr(run, field)]
   if not run.checkpoint_sha256 and not run.bundle_manifest_digest:
       missing.append("checkpoint_sha256 or bundle_manifest_digest")
   if missing:
       reasons.append("the artifact does not record the provenance its verification needs: "
                      + ", ".join(missing))
   ```
5. **Instrumentation Parity (`results.py:299-312`):**
   ```python
   if run.instrumentation_parity == "INSTRUMENTATION_PARITY_MISMATCH":
       reasons.append("instrumentation parity failed: the observed ranking does not reproduce the native hit counts")
       return PER_QUERY_PARITY_FAILED, reasons
   if run.instrumentation_parity is None:
       reasons.append("instrumentation parity has not been recorded for this artifact")
       return PER_QUERY_PARITY_PENDING, reasons
   if run.instrumentation_parity not in INSTRUMENTATION_PARITY_STATES:
       reasons.append("unrecognized instrumentation parity state "
                      f"{run.instrumentation_parity!r}; expected one of "
                      f"{sorted(INSTRUMENTATION_PARITY_STATES)}")
       return PER_QUERY_UNVERIFIED, reasons
   ```

### 5.3 Promotion to Verified Evidence (`promote`)
In `results.py:321-371`, an artifact cannot promote itself. It can only reach `PER_QUERY_VERIFIED` when the authoritative verifier (`scripts/verify_run.py`) produces a receipt where:
- `certified == True` and `is_research_evidence == True` (`:347`).
- `verifier_verdict.get("per_query_artifact_digest")` strictly matches the disk `artifact_digest` (`:358-364`).
- `model_id` and `checkpoint_id` match the run (`:366-370`).

---

## 6. FAILURE ANALYSIS COMBINATIONS & CURRENT FEASIBILITY

### 6.1 Feasibility Truth Today
**Every cross-model query-level failure analysis is IMPOSSIBLE today on real experimental evidence.**  
- `validate_analysis_selection` requires at least two runs with identical protocols and aligned query universes.
- Exactly 1 model (`limn`) has an implemented per-query observer.
- The other 15 models are **aggregate-only**.
- All query-level comparison endpoints (`/api/analysis`, `/api/analysis/failure-jaccard`, `/api/compare/{query_id}`, `/api/queries`) currently operate **only on mock data fixtures** (`workbench/scripts/load_mock_results.py`).

### 6.2 The 15 Aggregate-Only Models
The following 15 models contribute **only their aggregate metrics** (Recall@10, Recall@50, Mean Recall) and **cannot contribute per-query rankings**:
1. `csmcir`
2. `airknow`
3. `conesep`
4. `ptha_mtst` (unresolved protocol, unverified checkpoint)
5. `habit`
6. `intent`
7. `hint`
8. `encoder`
9. `pair`
10. `clvc_net`
11. `dcnet`
12. `combiner_rn50x4_noft`
13. `clip4cir_rn50x4_fullft`
14. `tgcir`
15. `sprc`

### 6.3 Most Valuable Model Combinations When Unlocked

#### Protocol C Combinations (`fashioniq_val_split`):
1. **LIMN vs. ENCODER**
   - *Target Analysis:* **Old-Correct / New-Wrong (Regressions)** and **Persistent Failures**.
   - *Scientific Motivation:* LIMN represents 2024 legacy CIR (CLIP ViT-B/32, modular independent category networks, mean 66.87). ENCODER represents 2025 recent CIR (shared backbone, mean 66.86). Despite virtually identical aggregate mean recalls (66.87 vs 66.86), do they fail on identical queries, or does ENCODER regress on specific categories where LIMN succeeded?
   - *What Unlocks It:* Fix ENCODER packaging (`datasets1.py` import), stage base OpenCLIP weights, resolve `fashioniq.pt`, implement per-query observer on `ENCODER/test.py:test`.
2. **LIMN vs. HINT**
   - *Target Analysis:* **Common Failures** and **Failure Jaccard**.
   - *Scientific Motivation:* HINT achieves the highest reported aggregate in Protocol C (68.21 mean). Comparing LIMN with HINT reveals whether the +1.34 point gain eliminates hard failure modes or merely shifts boundary ranks.
   - *What Unlocks It:* Implement evaluation CLI for `HINT/test.py` and hook per-query observer into its similarity matrix.
3. **LIMN vs. CLVC-Net**
   - *Target Analysis:* **Persistent Failures Across Generations** (2021 vs 2024).
   - *Scientific Motivation:* CLVC-Net (30.70 R@10, 44.56 mean) vs LIMN (55.91 R@10, 66.87 mean) maps the historical trajectory of pair-union CIR. Queries failed by both represent foundational visual-linguistic ambiguities in FashionIQ.
   - *What Unlocks It:* Resolve CLVC-Net category checkpoint bundle from author Drive, write runner, and implement observer.

#### Protocol B Combinations (`fashioniq_full_gallery_ref_excluded`):
1. **DCNet vs. Air-Know / ConeSep**
   - *Target Analysis:* **Persistent Failures** and **Old-Correct / New-Wrong**.
   - *Scientific Motivation:* DCNet is legacy 2021 (ResNet-50 + GloVe, mean 40.83). Air-Know/ConeSep are 2026 recent architectures (BLIP-2 / Q-Former, mean 65.73). Highlights multi-modal architectural evolution under full-split reference-masked evaluation.
   - *What Unlocks It:* Intercept dual-branch fused scores in `DCNet` `TrainerJoint._valid_epoch`; implement standalone execution harness and observer for Air-Know or ConeSep.
2. **Air-Know vs. ConeSep vs. HABIT vs. INTENT**
   - *Target Analysis:* **Top-K Rank Disagreement** and **Failure Jaccard**.
   - *Scientific Motivation:* Contemporary 2026 iLearn family models with tightly clustered reported means (65.24 - 65.73). Comparing them determines whether architectural nuances (cone separation vs habit memory vs intent disentanglement) produce distinct query-level behavior or represent equivalent representations.
   - *What Unlocks It:* Acquire 0%-noise clean checkpoints (currently only noise-trained checkpoints exist) and implement observer wrappers around `test.py`.

#### Protocol A Combinations (`fashioniq_original_split`):
1. **CLIP4Cir RN50x4 fullft vs. SPRC vs. CSMCIR**
   - *Target Analysis:* **Common Failures** and **Top-K Rank Disagreement**.
   - *Scientific Motivation:* Traces representation learning from standard fine-tuned CLIP (CLIP4Cir, 55.36 mean) to similarity-preserving re-ranking (SPRC, 64.85 mean) to CoT memory alignment (CSMCIR, 67.17 mean).
   - *What Unlocks It:* Verify checkpoints for CLIP4Cir and SPRC; resolve CSMCIR LAVIS/CUDA/image extension preflight blockers; implement observers hooking query-gallery similarity tensors.

### 6.4 Permanently Impossible Comparisons
- **Cross-Protocol Comparisons (e.g. LIMN vs CSMCIR, LIMN vs HABIT):**
  - Permanently rejected by workbench schema, runtime planning, and database index.
  - Can only ever be compared if an authoring team evaluates a model under an alternative protocol. However, each model adapter strictly validates `request.protocol_id in model['supported_protocols']`. CSMCIR only supports Protocol A; HABIT only supports Protocol B; LIMN only supports Protocol C.
  - Asserting comparability across these boundaries is mathematically false and rejected by `validate_analysis_selection` with `WorkbenchError("cross_protocol", ...)`.
