# FINAL 16-MODEL REPRODUCIBILITY REASSESSMENT

**Tree**: `b70d092` + the Phase A fixes in this pass. **Every per-model claim below cites the
per-model report in `workbench/docs/reviews/model_reassessment/<model_id>.md`, which cites `file:line`
in the pinned checkout.** Where a report could not establish something, this matrix says **UNKNOWN**
rather than filling the gap.

**The previous `EXTERNAL_BLOCKER` label is not used here.** It conflated an installable package, a
missing wrapper, an unpublished weight and a scientific unknown into one word, which cannot be
planned against. Every blocker below carries a precise classification, and a model with three distinct
problems has three distinct entries.

---

## 1. Scope, verified against the registry

| Quantity | Value | Source |
|---|---|---|
| Model entries | **16** | `workbench/registry/models.yaml` |
| `source_available: true` | **15** | same |
| Distinct upstream URLs among those 15 | **14** | `combiner_rn50x4_noft` and `clip4cir_rn50x4_fullft` share `ABaldrati/CLIP4Cir` |
| Vendored checkouts on disk | **14** | `workbench/third_party/` |
| Source-less entries | **1** (`ptha_mtst`) | `upstream_commit_sha: null`, no `source_dir` |
| Models with a per-query observer | **1** (`limn`) | `per_query_observer: LIMN_QUERY_OBSERVER_V1` |

Shared repositories were shared for **acquisition only**; each variant was audited independently
against its own architecture, checkpoint, command and output semantics.

## 2. Master matrix

Legend — **Proto**: A = `fashioniq_original_split` (full gallery, reference eligible) · B =
`fashioniq_full_gallery_ref_excluded` · C = `fashioniq_val_split` (pair-union, reference excluded).
**Entry**: `CLI_READY` · `CALLABLE_NO_CLI` · `ADAPTER_NOT_IMPLEMENTED` · `SOURCE_MISSING`. **Lvl**:
achieved verification level 1–6. **P/Q**: per-query capability today.

| model_id | year | source / pinned SHA | proto | entry | wrap | ckpt | asset | lvl | P/Q | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| `limn` | 2024 | `iLearn-Lab/TPAMI24-LIMN` `7d7bc9b1` | C | `REPLAY_COMMAND_IMPLEMENTED` | done | VERIFIED (3 sha256) | correction dicts in-tree | **3** | **observer** | **ACTUALLY_READY_PENDING_GPU** |
| `clip4cir_rn50x4_fullft` | 2023 | `ABaldrati/CLIP4Cir` `dfed9f74` | A | `CLI_READY` | — | **upstream now ships a checksummed release (members UNKNOWN without download)** | RN50x4 base | 2 | none | CHECKPOINT_ARCHIVE_PUBLIC + MEMBER_MAPPING_PENDING_DOWNLOAD |
| `combiner_rn50x4_noft` | 2023 | `ABaldrati/CLIP4Cir` `dfed9f74` | A | `CLI_READY` | — | **`fiq_comb_RN50x4_noft.pt` sha256 published** | same | 2 | none | CHECKPOINT_ARCHIVE_PUBLIC + MEMBER_MAPPING_PENDING_DOWNLOAD |
| `clvc_net` | 2021 | `iLearn-Lab/SIGIR21-CLVC-Net` `bd9b6889` | C | `CALLABLE_NO_CLI` | MODERATE | whole-module pickle, 6 members | asset-blocked | 2 | none | REQUIRES_EVALUATION_WRAPPER + LEGACY_RUNTIME (2 numerics-neutral edits proved) |
| `dcnet` | 2021 | `ozmig77/dcnet` `68c79d38` | B | `CLI_READY` | — | 2 files, VERIFIED | **OOV unseeded → NONDETERMINISTIC** | 2 | none | REQUIRES_UPSTREAM_ASSET (author cache unavailable, see §5) |
| `encoder` | 2025 | `iLearn-Lab/AAAI25-ENCODER` `29a2a31d` | C | `CLI_READY` | — | measured 631,879,365 B, sha256 unrecorded | LAION ViT-B/32 not provisioned | **3** | none | RECOVERY_PLAN_IDENTIFIED (alias + asset + parser NOT yet implemented) |
| `hint` | 2026 | `iLearn-Lab/ICASSP26-HINT` `bec50b6c` | C | `CALLABLE_NO_CLI` | MODERATE | **no FIQ loader exists** | correction dicts present | 2 | none | REQUIRES_EVALUATION_WRAPPER |
| `pair` | 2025 | `iLearn-Lab/ICASSP25-PAIR` `bfb0b98c` | C | `CALLABLE_NO_CLI` | BLOCKED_BY_MISSING_SOURCE | mapping unverified | — | 2 | none | **REQUIRES_SIGNIFICANT_RECONSTRUCTION** |
| `airknow` | 2026 | `iLearn-Lab/CVPR26-Air-Know` `6ab6fa95` | B | `CALLABLE_NO_CLI` | MODERATE | whole-module pickle, url live | correction dicts **unpublished** | 2 | none | REQUIRES_EVALUATION_WRAPPER + REQUIRES_UPSTREAM_ASSET |
| `habit` | 2026 | `iLearn-Lab/AAAI26-HABIT` `6ddc2cd6` | B | `CALLABLE_NO_CLI` | MODERATE | **VERIFIED** (size+sha per file) | correction dicts present | 2 | none | REQUIRES_EVALUATION_WRAPPER |
| `intent` | 2026 | `iLearn-Lab/AAAI26-INTENT` `2a75fc6f` | B | `CALLABLE_NO_CLI` | MODERATE | NO_CLEAN_CHECKPOINT (correct) | correction dicts unpublished | 2 | none | REQUIRES_EVALUATION_WRAPPER + REQUIRES_UPSTREAM_ASSET |
| `conesep` | 2026 | `Lee-zixu/ConeSep` → `iLearn-Lab/CVPR26-ConeSep` `bd4bf629` | B | `CALLABLE_NO_CLI` | MODERATE | NO_CLEAN_CHECKPOINT (correct) | correction dicts present | 2 | none | REQUIRES_EVALUATION_WRAPPER |
| `csmcir` | — | `qzp2018/CSMCIR` `774f94e2` | A | `CLI_READY` | — | VERIFIED (2.72 GB, oid) | **CoT val captions unpublished** | 2 | none | REQUIRES_UPSTREAM_ASSET |
| `sprc` | 2024 | `chunmeifeng/SPRC` `2935a539` | A | `ADAPTER_NOT_IMPLEMENTED` | MODERATE | **URL 403** | vendored LAVIS complete | 2 | none | REQUIRES_UPSTREAM_ASSET (checkpoint republish) |
| `tgcir` | 2023 | `iLearn-Lab/MM23-TG-CIR` `65fa78ea` | C | `CALLABLE_NO_CLI` | COMPLEX | zip, mapping UNKNOWN | CLIP ViT-B/16 unprovisioned | 2 | none | REQUIRES_EVALUATION_WRAPPER + NEEDS_CHECKPOINT_MAPPING |
| `ptha_mtst` | 2025 | **none vendored**; `zyinan99/MTST_PTHA` is the authors' own repo | **UNRESOLVED** | `SOURCE_MISSING` | BLOCKED | **no FashionIQ weight published** | — | 1 | none | **SOURCE_NOT_INTEGRATED** + CHECKPOINT_UNAVAILABLE + PROTOCOL_UNVERIFIED |

## 3. What the reassessment changed about the previous picture

Six findings that a documentation-level review would not have produced:

1. **`ptha_mtst` is not "source unavailable".** `zyinan99/MTST_PTHA` is the paper's **own** repository
   (README line 1 is the paper title; owner is the first author; commits are his). What is missing is
   the **FashionIQ checkpoint**: the author's Hub holds only three CIRR files, and `fashioniq.pt` /
   `tuned_mtst_fashioniq.pt` return 404. The pinned revision also cannot import —
   `models/__init__.py:51` names a module that 404s. Source exists; the artifacts do not.
2. **The CLIP4Cir upstream moved and FIXED its release.** `56bb547` (one commit past the pin,
   README-only) replaced an undocumented Drive folder with GitHub release `weights`: five
   `pretrained_models.tar.gz.part-*` assets **with GitHub-published SHA-256 digests**, verified here
   via the releases API. This is the single largest recoverability change in the set and it affects
   **two** registered models.
3. **`encoder`'s "missing module" is a one-token typo, not a distribution split.** The tree ships
   `datasets.py` defining exactly the four symbols the importer needs, with matching constructor
   signatures; the author's own Drive folder is byte-identical and contains no `datasets1.py` either.
   The existing audit note labels it `SOURCE_EVIDENCE_MISSING`, which is wrong in kind.
4. **Four models' correction dictionaries are already on disk.** `conesep`, `hint`, `intent` (and
   `tgcir`) declare them missing; the files are present and **byte-identical** to LIMN's
   `correct_files/` (`9e4a5987…`, `ca29e29e…`, `3dfbf15f…`). Prior documents are stale here.
5. **`dcnet`'s blocker is scientific, not engineering — and not permanent.**
   `preprocess/process_cap.py:31` draws out-of-vocabulary embeddings from `np.random.normal` with
   **no seeding call anywhere in `preprocess/`** (verified), so
   `ORIGINAL_PREPROCESSING_PARITY_UNVERIFIED`; the authors published no cache containing the
   embeddings they drew, so `AUTHOR_CACHE_UNAVAILABLE`. A seeded reimplementation is reproducible
   (`DETERMINISTIC_VARIANT_POSSIBLE`) but is not evidence of the original draw, so
   `ORIGINAL_PAPER_PARITY_NOT_ESTABLISHED`. Missing public data today is not proof that the
   authors' cache can never surface, so this is reported as an unestablished parity rather than as
   a permanent impossibility.
6. **`clvc_net`'s legacy problem is two edits, and both are numerically neutral.** `resnet.py:3`'s
   removed re-export and the `weights_only` flag; the audit proved bit-identical outputs and equal
   gradients. But the same audit found the `pretrained=` default now resolves to a **different weight
   file** (`IMAGENET1K_V2` vs the declared `resnet50-19c8e357.pth`) — a trap no prior audit mentions.

## 4. Two models REFUTED by execution

| Prior claim | Status | Evidence |
|---|---|---|
| `intent` has a tensor-shape failure in the similarity computation (P0) | **REFUTED** | `blip2_cir.py:769` appends axes → `(B,256,1,1)`; `:792` permutes to `(B,256,32)`; `np.matmul` broadcasts to `(M,N,1,32)`, and `test.py:150`'s indexing is well-formed. The registry had **already withdrawn** the claim (`models.yaml:306`); the brief and the old audit were both stale. |
| The LIMN observer's ranking derivation differs from native | **REFUTED** | 200/200 trials reproduce the full order including ties, against the native code quoted line by line. |
| `sprc`'s vendored LAVIS is incomplete (98 unguarded imports) | **REFUTED** | That is **CSMCIR's** defect. SPRC's vendored LAVIS resolves 47/47 imports. |
| `sprc`'s checkpoint URL is merely "restricted" | **REFUTED — worse** | measured `301 → 403`; the same-repo CIRR link is also 403. Author must republish. |
| The paper contains no out-of-the-box FashionIQ table | **REFUTED in form, right in conclusion** | arXiv 2308.11485 Table 1 does carry the `Combiner/✗/✗` row — but for **RN-50**, and the registry's variant is RN-50**x4**, so the nulls stay defensible. |

## 5. What cannot be fixed from this repository

| Blocker | Model(s) | Why not |
|---|---|---|
| `NONDETERMINISTIC_PREPROCESSING` | `dcnet` | unseeded OOV embeddings; regeneration ≠ the authors' run and no `.pkl` was published |
| `CHECKPOINT_UNPUBLISHED` | `ptha_mtst` | no FashionIQ weight exists in any public location |
| `CHECKPOINT_URL_INACCESSIBLE` | `sprc` | author's URL returns 403; only the authors can republish |
| `AUXILIARY_ASSET_MISSING` (unpublished) | `csmcir`, `airknow`, `intent` | the CoT val captions / correction dictionaries exist in no public tree |
| `UNPUBLISHED_SOURCE_MODULE` | `pair` | `MHTransformer_rev` is absent, and no code path can instantiate the revision class that IS present |
| `SCIENTIFIC_PROTOCOL_UNVERIFIED` | `ptha_mtst` | protocol cannot be established without the source and weights |

## 6. Per-model reports

`workbench/docs/reviews/model_reassessment/<model_id>.md` — 16 files, each with identity, source
status (both snapshots), native workflow, dependencies, checkpoints, auxiliary assets, protocol,
workbench integration, confirmed defects, remaining blockers, implementation plan, verdict and evidence.

## 7. What this does NOT claim

No model was executed on a GPU. No checkpoint over ~600 MB was downloaded. `ptha_mtst`'s archive
members and `tgcir`'s archive contents are **UNKNOWN** because the archive was not fetched, and that
gap is stated in both reports rather than filled. Level 5 and 6 are unreachable on this host by
construction, and no model is labelled above its achieved level.
