# csmcir — reproducibility reassessment

Independent adversarial, READ-ONLY audit. Nothing under `/home/heheboiz/data/cir` was written except
this file. Scratch work in `/tmp/aud`. Everything is `STATIC_AUDIT` except lines explicitly marked
`OBSERVED` (a command I ran). **No GPU inference was run and no checkpoint was loaded into a model:
every runtime number is absent, not estimated.**

---

## 1. MODEL IDENTITY

| Field | Value | Source |
|---|---|---|
| model_id | `csmcir` | `workbench/registry/models.yaml:21` |
| method_name | CSMCIR | `models.yaml:38` |
| paper_title | "CSMCIR: CoT-Enhanced Symmetric Alignment with Memory Bank for Composed Image Retrieval" | `models.yaml:43` |
| paper_url | `https://arxiv.org/abs/2601.03728` | `models.yaml:44` |
| upstream_repo_url | `https://github.com/qzp2018/CSMCIR` | `models.yaml:45` |
| upstream_commit_sha | `774f94e2076ff17ea91703a6239d2a08f0e1a44e` | `models.yaml:46` |
| source_dir | `CSMCIR` | `models.yaml:48` |
| native_protocol | `fashioniq_original_split` | `models.yaml:56` |
| literature_split_label | `original` | `models.yaml:55` |
| reported_r10 / r50 / mean | 57.07 / 77.27 / 67.17 | `models.yaml:58-60` |
| checkpoint variant | `fashioniq` → `fashioniq_tuned_clip_best.pt` | `models.yaml:83-84` |
| reproduction_status | `NOT_RUN` | `models.yaml:61` |

`publication_year: null` (`models.yaml:39`) although the pinned `README.md:4` says
"**(ACL206 Findings)**" (literal typo for ACL 2026 Findings). P3.

**OBSERVED** pin check:
```
$ git -C workbench/third_party/CSMCIR rev-parse HEAD
774f94e2076ff17ea91703a6239d2a08f0e1a44e
```
Matches the registry exactly. **VERIFIED.**

## 2. SOURCE STATUS

**Accessible: yes. Complete: no.** Two independent classes of missing module (both below).
**SNAPSHOT 2 (current upstream), OBSERVED:**
```
$ git -C workbench/third_party/CSMCIR ls-remote origin HEAD
774f94e2076ff17ea91703a6239d2a08f0e1a44e	HEAD
```
Current upstream HEAD equals the pin. The upstream authors have **not** added a CLI, fixed the
import block, or published a checkpoint or `COT_ours2/` since the pin. The pin is current.

The checkout carries 856 `git ls-files` entries, of which **498 are tracked `.pyc`** (OBSERVED):
```
$ git ls-files '*.pyc' | wc -l   -> 498
$ git ls-files '*.py'  | wc -l   -> 160
```
The author committed compiled bytecode of files that are not in the repository. This is not cosmetic
— it is the only surviving trace of the missing modules (see 3).

## 3. NATIVE EVALUATION WORKFLOW

### 3.1 Entrypoint, function chain, CLI

- Entrypoint: `src/validate_blip_csmcir.py` (adapter `workbench/backend/adapters/models.py:30`).
- `__main__` at `src/validate_blip_csmcir.py:891-892` → `main()` at `:501`.
- CLI (`:503-508`): `--dataset` (default `fashionIQ`), `--blip-model-name`
  (default `blip2_cir_align_prompt_csmcir`), `--blip-model-path`, `--target-ratio` (1.25),
  `--transform` (`targetpad`).
- Adapter command (`workbench/backend/adapters/models.py:32-33`):
  `python src/validate_blip_csmcir.py --dataset fashionIQ --blip-model-path <checkpoint>`.
  The omitted `--blip-model-name` defaults to exactly the registered class name.

Function chain for FashionIQ:
`main:514` → `blip_validate_fashioniq:565` → `load_model_and_preprocess:568`
→ `torch.load`+`load_state_dict:570-571` → `FashionIQDataset('val',[cat],'relative',preprocess):596`
and `FashionIQDataset('val',[cat],'classic',preprocess):598`
→ `extract_index_blip_caption_features:612` → `compute_fiq_val_metrics:613`
→ `generate_fiq_val_predictions:149` → `DataLoader(..., collate_fn=collate_fn):164`
→ `blip_model.inference(...):199` → `compute_fiq_val_metrics` asserts and returns R@10/R@50
(`:54-55`) → JSON printed at `:630`.

Note: pinned `README.md:115` documents `python src/blip_validate_csmcir.py … --model-path` but no
such file and no such flag exist (OBSERVED `ls` → `MISSING src/blip_validate_csmcir.py`); the real
file accepts `--blip-model-path`. The workbench adapter uses the working spelling. P3 doc bug.

### 3.2 What a wrapper must call (if the source were importable)

A wrapper would need only the adapter's command; the source ships its own `__main__`. **No wrapper is
needed** — this model is one of the few with a real CLI. The blocker is not the CLI.

### 3.3 Missing components (this is the finding)

**(a) The pinned `lavis.models` package cannot be imported.** OBSERVED, independently recomputed:

```
$ python: parse src/lavis/models/__init__.py, resolve each `from lavis.X import Y` against the tree
TOTAL lavis imports in models/__init__.py: 146
PRESENT: 48
MISSING: 98
first missing line: 44
CSMCIR import line: 152
```

Breakdown of the 98 absent modules: 90 are
`lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan{4..92}`, and 8
are other author working-copy variants (`…_caption_query`, `…_daul_contrastive`, `…_alignloss`,
`…_target_image_caption`, `…_logsumexp`, `…_attentionpooling`, `…_gatepool`, `…_aggregator`).
None of lines 44-151 is wrapped in `try`/`except` (OBSERVED: `any try/except in 40-155` → none), so
the first absent module raises `ModuleNotFoundError` and `src/validate_blip_csmcir.py:12`
(`from lavis.models import load_model_and_preprocess`) dies during import.

**(b) NEW — the checkpoint's own class module does not exist either.** This is the decisive finding
and it is not in any existing workbench document.

The pinned `src/lavis/models/blip2_models/` directory contains 20 tracked `.py` model files plus
**129 tracked `.pyc`** in `__pycache__/` (OBSERVED). Many `.pyc` files have no `.py` counterpart.
OBSERVED, extracting the top-level key set from the published checkpoint's `data.pkl` (read from the
Hub by HTTP range request; see 5):

```
$ python: pickletools.genops over tuned_clip_best/data.pkl
   first constants: ['epoch', 'Blip2QformerCirAlignPromptTargetImageCaptionshiyan92', 'query_tokens', ...]
```

`epoch 9`, and the state-dict key is the literal string
`Blip2QformerCirAlignPromptTargetImageCaptionshiyan92` — the class name defined by
`blip2_qformer_cir_align_prompt_target_image_caption_shiyan92.py`. OBSERVED:

```
$ ls src/lavis/models/blip2_models/*shiyan*.py   -> no matches
$ git ls-files | grep shiyan92
src/lavis/models/blip2_models/__pycache__/blip2_qformer_cir_align_prompt_target_image_caption_shiyan92.cpython-{310,39}.pyc
```

The evaluator loads with
`blip_model.load_state_dict(checkpoint[blip_model.__class__.__name__], strict=False)`
(`src/validate_blip_csmcir.py:571`). `blip_model.__class__.__name__` is
`Blip2QformerCirAlignPromptCSMCIR` (registered at
`blip2_qformer_cir_align_prompt_csmcir.py:27-28`). The checkpoint key is `…shiyan92`. **The two names
do not match, so the lookup `checkpoint["Blip2QformerCirAlignPromptCSMCIR"]` raises `KeyError`
regardless of the class modules being present.** `strict=False` only forgives missing tensors inside a
successfully retrieved dict; it does not forgive a missing dict key.

I have **not** excluded the possibility that the evaluator was meant to run against a different
`--blip-model-name`; the registry declares none, and the default is used. What is established is: the
declared checkpoint for the declared model name cannot be selected by that name.

**(c) Two required class modules are absent and the checkpoint does not carry their weights.** Even
if the name mismatch in (b) fitted, `Blip2QformerCirAlignPromptCSMCIR` builds a Memory Bank
(`blip2_qformer_cir_align_prompt_csmcir.py:99-110`): `memory_bank_visual` (257*1408 x 512),
`memory_bank_text`, `memory_bank_text_mask`, `memory_bank_cls_tokens`, `memory_bank_ptr`,
`memory_sample_age`. The checkpoint is a `state_dict`, not a whole module (evidenced by its
`data.pkl` holding an `OrderedDict` of tensors and no `__main__`/class global). OBSERVED string counts
in `data.pkl`:

```
visual_encoder 1062   Qformer 713   vision_proj 3   text_proj 3   query_tokens 1
temp 2   token_importance 1   itm_head 3   ln_vision 3   attn_proj 2
memory_bank 0   alpha 0   prompt_tokens 0
```

`memory_bank*` → **0 occurrences**. `load_state_dict(..., strict=False)` would leave the four Memory
Bank buffers at their `__init__` values — `memory_bank_visual` is `F.normalize(randn(257*1408,512))`,
a **random** buffer. For CSMCIR that may be benign on the eval path: the FashionIQ gallery path calls
`extract_target_caption_features` (`:498`) and `inference` (`:411`, `:438-451`), and the string
`memory_bank` does not appear in either. It is used in `forward` (`:350-374`) only. So the evaluation
path never touches the bank. I record the observation without claiming an effect.

**Missing components summary:** the package (98 imports), two class modules, a matching checkpoint
key. The registered entrypoint cannot reach `compute_fiq_val_metrics`.

## 4. DEPENDENCIES

Dependency categories D1-D8 were **not defined in any file I can read** (`grep` over `workbench/`
for `D1–D8`, `DEPENDENCY_CATEGORY`, `dependency categor` → no definition). Rather than invent the
mapping I give the concrete category per import in words, and mark the label I am applying.

| import | file:line | owner | required version | installable? | py/torch/cuda | safe remedy | test result |
|---|---|---|---|---|---|---|---|
| `from lavis.models import load_model_and_preprocess` | `src/validate_blip_csmcir.py:12` | upstream CSMCIR | local tree | n/a — **local module tree, not a package** | 3.9 / 2.0.1 / 11.7 | repair the import block (approved patch) **and** ship the two class modules | NOT_RUN — module absent |
| `from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_csmcir import Blip2QformerCirAlignPromptCSMCIR` | `src/lavis/models/__init__.py:152` | upstream CSMCIR | local tree | module **present** | 3.9 / 2.0.1 | none | resolvable (OBSERVED) |
| 98 imports of `…_shiyan{4..92}` / variant modules | `src/lavis/models/__init__.py:44-151` | upstream CSMCIR | local tree | **NOT installable — unpublished source** | n/a | approved patch guards them | resolvable only after patch |
| `from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_target_image_caption_shiyan92 import …` *(implied by the checkpoint key)* | module absent; only `.pyc` at `…/__pycache__/…shiyan92.cpython-39.pyc` | upstream CSMCIR | unpublished | **NOT installable** | n/a | none — recover from `.pyc` (decompilation) or retrain | NOT_RUN |
| `transformers.modeling_utils.apply_chunking_to_forward` | `src/lavis/models/blip2_models/Qformer.py:41` (also `med.py:42`, `nlvr_encoder.py:21`) | third-party | transformers 4.36.2 per `requirements.txt:11` | yes | 3.9 / 2.0.1 | pin the declared version; **it is an environment mismatch, not a source defect** | OBSERVED FAIL under the workbench venv (transformers 5.x): `ImportError: cannot import name 'apply_chunking_to_forward' from 'transformers.modeling_utils'` |
| torch/torchvision/torchaudio | `README.md:14`; `models.yaml:25-27` | third-party | 2.0.1 / 0.15.2 / 2.0.2 | yes | CUDA 11.7 declared | environment creation | NOT_RUN |
| opencv-python, timm 0.9.12, einops, fairscale, spacy, pycocoevalcap, pandas, omegaconf, iopath, decord, webdataset, scikit-image | `requirements.txt:1-14` | third-party | as pinned | yes | — | environment creation | NOT_RUN |
| `torch.cuda.empty_cache()` (hard CUDA) | `src/validate_blip_csmcir.py:618`; `src/utils_csmcir.py:13-16` | upstream CSMCIR | — | n/a | CUDA required | none; GPU host needed | NOT_RUN |

The `transformers` failure is the only observed import error, and it is caused by my probe environment
(workbench venv, transformers 5.x), **not** by the model's declared environment. It is reported here
because it is the error a naive local import test sees first, and it must not be mistaken for the real
blocker. Not a model defect. P3.

## 5. CHECKPOINTS

Declared: `models.yaml:83-96`.

| field | declared | OBSERVED | verdict |
|---|---|---|---|
| filename | `fashioniq_tuned_clip_best.pt` | Hub lists it | MATCH |
| URL | `https://huggingface.co/peng12138/CSMCIR/resolve/main/fashioniq_tuned_clip_best.pt` | HTTP 200, `Content-Length: 2720615859` | MATCH |
| expected_size_bytes | 2720615859 | 2720615859 | MATCH |
| expected_sha256 | `c8812071de0640d647c5ba2a70628b80a3289ee2c47676f7a1b6abaa313fa49e` | LFS pointer `oid sha256:c8812071de0640d647c5ba2a70628b80a3289ee2c47676f7a1b6abaa313fa49e`, `size 2720615859` | MATCH |
| mapping status | `VERIFIED_METADATA` | mapping is verified **as metadata** | correct as labelled |
| training noise | 0 | not asserted by the author anywhere | UNRESOLVED |

OBSERVED (author Hub, revision `cf0c19bb346266c295b4b5772ebf03bd1c4f0467`):
`https://huggingface.co/api/models/peng12138/CSMCIR` lists 22 siblings including four `*.pt`
(`fashioniq_tuned_clip_best.pt`, `cirr_tuned_clip_arithmetic.pt`, `LaSCo_tuned_clip_best.pt`,
`shoes_tuned_clip_best.pt`), so the FashionIQ file is unambiguous.

**Where would the weights come from?** The author Hub, as declared. There is no local copy: no
`artifacts/checkpoints/csmcir/` payload was observed. So the bytes are reachable and the metadata
matches, but **no loader or run exists**.

**What the container is.** OBSERVED, HTTP range reads of the ZIP central directory:
```
size 2720615859, entries 861
first member: tuned_clip_best/data.pkl (181696 bytes)
```
ZIP prefix `tuned_clip_best/` is what `torch.save(obj, "…/tuned_clip_best.pt")` produces from the
filename — so the archive prefix does not by itself distinguish a whole-module pickle from a
state_dict. The `data.pkl` content does: it unpickles into an `OrderedDict` whose values are tensors
(`_rebuild_tensor_v2`), and its first keys are `epoch`, `Blip2QformerCirAlignPromptTargetImageCaptionshiyan92`,
`query_tokens`, … . I verified the prefix convention against a known-good local `torch.save`
(OBSERVED, torch 2.14.0+cpu: a state_dict saved to `state_dict.pt` yields prefix `state_dict/`; a
whole `nn.Module` yields `whole_module/`). **Conclusion: `state_dict` keyed by class name** — which is
what `validate_blip_csmcir.py:571` assumes, and exactly why finding 3.3(b) bites.

**Availability class:** `URL_VERIFIED` / `CHECKPOINT_URL_INACCESSIBLE` does not apply — the URL works.
The blocking class is that the declared checkpoint is keyed to a class that does not exist in the
pinned revision.

## 6. AUXILIARY ASSETS

Registry: `workbench/registry/auxiliary_assets.yaml`.

| asset | required for | declared source | OBSERVED | class |
|---|---|---|---|---|
| Qwen captions `{dress,shirt,toptee}_cot_val.json` | official_evaluation | author Hub `fashioniq_qwen_captions/qwen_captions/…` | HTTP 200, real JSON content, digest recorded per file | `OFFICIAL_PUBLIC_DOWNLOAD` / AVAILABLE |
| `COT_ours2/fashioniq/{dress,shirt,toptee}_cot_val.json` | official_evaluation | `source_type: none`, `source_repo: null`, `automatic_download_available: false`, `provenance_status: UNVERIFIED` (`auxiliary_assets.yaml:72-95`) | not on the author Hub (22-file listing, no `COT_ours2`); `git ls-files COT_ours2` in the pinned tree → empty; a local untracked directory exists | `SOURCE_EVIDENCE_MISSING` / BLOCKED |

**Which file does the evaluation path need?** Both, in different roles, both unconditional for a
`FashionIQDataset`:

1. `COT_ours2/fashioniq/{cat}_cot_val.json` — read by `src/utils_csmcir.py:100` inside
   `extract_index_blip_caption_features`, which the official path calls at
   `src/validate_blip_csmcir.py:612`. It supplies the **gallery** caption features via
   `blip_model.extract_target_caption_features` (`utils_csmcir.py:161-162` → model `:498`), i.e. it
   decides the gallery embeddings that R@10/R@50 are computed against. The FashionIQ branch reads
   **all three** files regardless of which category the dataset holds (`utils_csmcir.py:97-103`).
2. `fashionIQ_dataset/qwen_captions/{cat}_cot_val.json` — read by
   `src/data_utils_csmcir.py:408` in `FashionIQDataset.__init__`, feeding `__getitem__`'s
   `image_captions_cot` (`:435`), which reaches `inference`'s `text_cot` argument.

`text_cot` is **never read** inside `inference` (OBSERVED: `grep -n text_cot` in the model file →
`:411`, `:454` signatures only). So (2) is dead weight on the query path; (1) is genuinely
load-bearing and **unpublished**.

**Downloadable?** (1) no; (2) yes. **Regenerable?** No. `COT_ours2` is the author's own CoT caption
set (`README.md` names only the *qwen* captions as downloadable), produced by an unspecified
generating model whose weights, prompt and randomness are nowhere in the tree. Regenerating cannot
reproduce the author bytes; a regenerated set would be a different experiment. The workbench's
`caption_alias_from` mechanism (`auxiliary_assets.yaml:90-94`) stages the published Qwen file at the
`COT_ours2` path and labels the run a variant — correct engineering, honestly labelled, but it is a
**substitution**: `auxiliary_assets.yaml:75` itself says
`SOURCE_COMPATIBLE_EQUIVALENCE_UNKNOWN`. OBSERVED: the local `COT_ours2/fashioniq/*` files are staged
and byte-identical in size to the Qwen assets (2817386 / 5015979 / 3929303). P0 for official parity.

`keywords_in_mods_{cat}.json` (`src/data_utils_csmcir.py:216`) and `bert_captions`
(`:223`, `:226`) are read only by `FashionIQDataset_imageedit` (class at `:175`), which has **no
callsite** anywhere in the tree (OBSERVED `grep`), so they are not required. Correctly not registered.

## 7. FASHIONIQ PROTOCOL

Registry: `native_protocol: fashioniq_original_split` (`models.yaml:56`),
`supported_protocols: [fashioniq_original_split]` (`:57`), `literature_split_label: original` (`:55`).
**Quote the authoritative declaration:** Protocol A = `fashioniq_original_split` — full gallery,
reference eligible. I did not and will not propose changing it.

- **Gallery:** the full ordered validation split per category.
  `FashionIQDataset.__init__` builds `image_names` from
  `fashionIQ_dataset/image_splits/split.{cat}.val.json` (`src/data_utils_csmcir.py:416-419`); the
  `classic` mode `__getitem__` yields `(name, preprocess(Image.open(images/{name}.png)))`
  (`:466-470`). No pair union, no dedup inside the evaluator.
- **Reference policy:** reference **eligible**. `compute_fiq_val_metrics`
  (`src/validate_blip_csmcir.py:24-57`) builds `labels` only against `target_names` (`:49-50`) and
  never masks or removes `reference_names`; `generate_fiq_val_predictions` uses the reference only to
  build the query (`:169`, `:193-194`). No `-10e10` mask exists on this path. Matches Protocol A.
- **Query construction:** the two relative captions concatenated deterministically as
  `"{cap[0].strip('.?, ').capitalize()} and {cap[1].strip('.?, ')}"` (`:180-183`), then
  `txt_processors["eval"]` (`:185`).
- **Categories:** `['dress','toptee','shirt']` (`:592`), each evaluated separately.
- **Recall computation:** `distances = 1 - pred_sim` (`:44`), `argsort` (`:45`),
  `labels = sorted_index_names == repeat(target_names, len(index_names))` (`:49-50`),
  `assert sum(labels,-1) == 1` (`:51`), `R@10 = sum(labels[:,:10])/len(labels)*100` (`:54`),
  R@50 at `:55`. Macro = unweighted mean of the three categories (`:620-627`), then
  `(mean50+mean10)/2` (`:627`) — the paper's convention.

Known source-level property to keep visible: 121 FashionIQ ids appear in both the shirt and toptee
val splits, and `extract_index_blip_caption_features` merges all three `COT_ours2` dicts in
`dress, shirt, toptee` order, so for those ids the toptee caption wins even inside the shirt gallery.
That is upstream behaviour, orthogonal to the caption-source problem, and it is not a reason to change
the protocol. P2 documentation note.

## 8. WORKBENCH INTEGRATION

- **Adapter:** `workbench/backend/adapters/models.py:28-33` — `CSMCIRAdapter(OfficialScriptAdapter)`,
  `script = "src/validate_blip_csmcir.py"`, real `command()` override → **audited**
  (`workbench/scripts/evaluate_models.py:130-131`).
- **Planner / autopilot:** no `csmcir` reference in `workbench/backend/autopilot.py` outside the
  registry-driven path; `workbench/configs/pc_autopilot.pilot_csmcir.yaml` exists as a
  reference-blocked profile (`FINAL_PC_READINESS.md:75-76`).
- **Checkpoint prep:** `workbench/backend/registry.py` `checkpoint_path`/`checkpoint_member_root`;
  the declared artifact is a **single file** (`models.yaml:84`) — `main.py:356-360` records that the
  single-file member resolution was recently fixed so such artifacts no longer hash to `"unresolved"`.
- **Preflight:** `workbench/backend/preflight.py:40-116`. It (a) resolves `paths_for_layout`,
  (b) runs `image_extension_problems` against the declared `.png`
  (`fashioniq_layout.py:92-136`), (c) calls `approved_patch_state`
  (`preflight.py:119-158`) and — when the derived digest matches — records
  "source repaired by the approved patch … the published-source blocker no longer applies and the run
  is an approved compatibility variant, not official parity", (d) validates the auxiliary assets and
  accepts a staged caption alias as a labelled variant (`preflight.py:74-81`).
  `base_sha256` was independently confirmed: OBSERVED
  `sha256sum workbench/third_party/CSMCIR/src/lavis/models/__init__.py` →
  `9d8c914b551a7de614e062f9a241f44626fd591a31a0ccf1cc79f0ae23c9db01`, equal to `models.yaml:77`.
- **Metric parser:** `workbench/backend/metrics_extraction.py:219-305` `parse_csmcir_output`;
  dispatched from `:404`. It requires all six per-category keys plus the three averages, validates
  0-100, recomputes the macro and rejects a disagreement > 0.05, then `_check_parity(tolerance=0.05)`.
  Matches the source key names of `src/validate_blip_csmcir.py:620-630`. **VERIFIED (static).**
- **Per-query support:** none for this model. The evaluator returns three macro numbers and prints no
  per-query data; `inference` returns only `sim_i2t`. No per-query artifacts can exist without
  instrumenting the official implementation, which the project forbids before parity. Correct.

## 9. CONFIRMED DEFECTS

**D-1 (P0) — 98 unguarded absent imports before the registration.**
Root cause: the author's working copy diverged from the published revision; `git ls-files` shows the
`.pyc` of the missing modules shipped while the `.py` were not (498 tracked `.pyc` vs 160 `.py`).
Reproduction: OBSERVED script; `98 44 151`; `any try/except in 40-155` → none. Entry import at
`src/validate_blip_csmcir.py:12`. Fix: already declared as an approved patch (see D-2).
Regression result: NOT_RUN (no GPU).

**D-2 (P0) — declared checkpoint is keyed to a class the revision does not contain.** *New finding.*
Root cause: `src/validate_blip_csmcir.py:571` indexes the state_dict by
`blip_model.__class__.__name__`, but the checkpoint's key is
`Blip2QformerCirAlignPromptTargetImageCaptionshiyan92` while the registered class is
`Blip2QformerCirAlignPromptCSMCIR`. Evidence: pickletools constants + the missing `.py`.
Reproduction: read the checkpoint's `data.pkl` by HTTP range (commands in 13).
Fix: the author must publish the module the checkpoint was trained with (or a renamed dict).
Regression result: NOT_RUN.

**D-3 (P0) — `COT_ours2/fashioniq/*_cot_val.json` has no public source** and drives gallery features.
See 6. Fix: operator must supply the author artifact, or the run must be labelled a variant, which
`auxiliary_assets.yaml:75` already does.

**D-4 (P1) — the four Memory Bank buffers are absent from the checkpoint** (`memory_bank` count 0 in
`data.pkl`). With `strict=False` they stay at their random `__init__` values
(`blip2_qformer_cir_align_prompt_csmcir.py:99-110`). No effect found on the eval path
(`inference`/`extract_target_caption_features` never read the bank). Recorded so it is not
rediscovered as a mystery. Severity P1 because it is a silent-state difference between the author's
training graph and the evaluator's graph.

## 10. REMAINING BLOCKERS

| # | class | ownership | evidence | remediation |
|---|---|---|---|---|
| B1 | `UPSTREAM_IMPORT_BUG` | upstream (qzp2018/CSMCIR) | `src/lavis/models/__init__.py:44-151`; 98 absent modules | approved patch declared; not applied by default. Fine. |
| B2 | `UNPUBLISHED_SOURCE_MODULE` | upstream | `blip2_qformer_cir_align_prompt_target_image_caption_shiyan92.py` absent; only `.pyc`; checkpoint key names it | author must publish; otherwise the `.pyc` is the only recoverable form (decompilation), which is not parity |
| B3 | `PUBLIC_ASSET_NOT_PREPARED` | operator | `COT_ours2/fashioniq/*_cot_val.json` unpublished; `auxiliary_assets.yaml:72-95` UNVERIFIED | operator supplies the author file, or the labelled variant path is used |
| B4 | `CHECKPOINT_NOT_MAPPED` (new class needed) | upstream | checkpoint key ≠ registered class name | author republishes or the loader key is mapped, which changes the official implementation |
| B5 | `GPU_RUNTIME_UNTESTED` | workbench host | `reproduction_status: NOT_RUN`; no GPU observed | another host |

`EXTERNAL_BLOCKER` is not used: the taxonomy in the task supplies precise tokens and every blocker
above has an owner.

## 11. IMPLEMENTATION PLAN

I am not proposing to make csmcir runnable, and I explicitly decline to make it runnable by any of
the forbidden means (no gallery change, no reference-policy change, no caption substitution presented
as parity, no checkpoint substitution).

1. **Record D-2** in `workbench/patches/csmcir.md` (or a sibling doc) as a second, independent
   blocker with the evidence in 13. Complexity: LOW (documentation). Scientific risk: none.
   Expected output: the next reader does not spend a day rediscovering that the state_dict key is
   `…shiyan92`.
2. **Do not extend the approved patch** to add the Memory Bank keys or to remap the checkpoint key:
   either would change which weights execute, i.e. a different experiment. If a run is ever made,
   it must be labelled VARIANT with the missing-buffer note.
3. **Leave `native_protocol` alone.** `fashioniq_original_split` is correct for this source.
4. Optional P3: fix `publication_year: null` and the `README.md:115` `--model-path` doc bug.

## 12. FINAL VERDICT

**BLOCKED.**

- `native_protocol: fashioniq_original_split` is **VERIFIED** against the pinned source
  (`data_utils_csmcir.py:416-419,466-470`; `validate_blip_csmcir.py:49-50`).
- The approved patch is **VERIFIED to do exactly what it claims and nothing else** (it can be applied
  cleanly with `patch -p1`, reproduces the declared `derived_sha256` exactly, and moves exactly the 98
  import statements of absent modules into 4 `try:` blocks — 98 imports inside `try` bodies, 0 of them
  resolvable from the tree, 0 unresolvable imports left unguarded; `ast.parse` accepts the result;
  no import is dropped, added or reordered).
- The checkpoint **URL, size and sha256 are VERIFIED** against the author Hub.
- The model is nevertheless unreproducible from the pinned source: the package cannot import, the
  class module named by the checkpoint does not exist, and the gallery caption file was never
  published.
- `reproduction_status: NOT_RUN` is **honest and correct**.

## 13. EVIDENCE

Repo: `https://github.com/qzp2018/CSMCIR`. Pin: `774f94e2076ff17ea91703a6239d2a08f0e1a44e`.

Commands and observed output (all read-only; scratch in `/tmp/aud`):

```
$ git -C workbench/third_party/CSMCIR rev-parse HEAD
774f94e2076ff17ea91703a6239d2a08f0e1a44e
$ git -C workbench/third_party/CSMCIR ls-remote origin HEAD
774f94e2076ff17ea91703a6239d2a08f0e1a44e	HEAD

$ sha256sum workbench/patches/csmcir_models_init_compat.patch
0e703584a7ecb5df3781574c2516fd406745c414491b6aa487fbb3e76532b9bd
$ sha256sum workbench/third_party/CSMCIR/src/lavis/models/__init__.py
9d8c914b551a7de614e062f9a241f44626fd591a31a0ccf1cc79f0ae23c9db01
# -> both match models.yaml:76 and :77

$ git archive HEAD src/lavis/models/__init__.py | tar -x -C /tmp/aud/tree
$ cd /tmp/aud/tree && patch --dry-run -p1 < /tmp/aud/csmcir_models_init_compat.patch
checking file src/lavis/models/__init__.py
DRYRUN_EXIT=0
$ patch -p1 --no-backup-if-mismatch < … && sha256sum src/lavis/models/__init__.py
aa68fe5e20bcda5e17d978799deb8ec0ff4636e5b3b2a6ef637ce5e385b0c818
# -> equals models.yaml:78 (derived_sha256)

# semantic check of the patched file (AST)
try blocks: 4 imports inside: 98 imports inside that RESOLVE: []
unguarded (col-0) imports that do not resolve: []
orig lavis import lines: 146 new: 146 | identical as multiset: True

# import probes
$ python -c "import sys;sys.path.insert(0,'…/CSMCIR/src'); import lavis.models"
ImportError: cannot import name 'apply_chunking_to_forward' from 'transformers.modeling_utils'
# (this venv's transformers is 5.x; the declared 4.36.2 is what requirements.txt pins — env mismatch, not source)

# checkpoint metadata
$ HEAD https://huggingface.co/peng12138/CSMCIR/resolve/main/fashioniq_tuned_clip_best.pt
200 Content-Length: 2720615859
$ GET …/raw/main/fashioniq_tuned_clip_best.pt
version https://git-lfs.github.com/spec/v1
oid sha256:c8812071de0640d647c5ba2a70628b80a3289ee2c47676f7a1b6abaa313fa49e
size 2720615859

# checkpoint container
$ HEAD … -> 2720615859 ; ZIP EOCD -> 861 entries ; first member tuned_clip_best/data.pkl (181696)
$ pickletools.genops(data.pkl) first constants:
['epoch', 'Blip2QformerCirAlignPromptTargetImageCaptionshiyan92', 'query_tokens', …]
$ string counts in data.pkl:
visual_encoder 1062  Qformer 713  vision_proj 3  text_proj 3  query_tokens 1
temp 2  token_importance 1  itm_head 3  ln_vision 3  attn_proj 2
memory_bank 0  alpha 0  prompt_tokens 0
$ ls src/lavis/models/blip2_models/*shiyan*.py   -> no matches
$ git ls-files | grep -c shiyan                   -> 189 (all .pyc or a .ipynb_checkpoints copy)
$ git ls-files '*.pyc' | wc -l ; git ls-files '*.py' | wc -l   -> 498 ; 160

# auxiliary assets
$ HF api models/peng12138/CSMCIR -> sha cf0c19bb…, 22 siblings, no COT_ours2 path
$ GET …/raw/cf0c19bb…/fashioniq_qwen_captions/qwen_captions/dress_cot_val.json -> 200, real JSON
$ ls COT_ours2/fashioniq/ -> dress 2817386  shirt 5015979  toptee 3929303  (local, staged)
```

`file:line` index: `models.yaml:21,38,43-48,55-61,73-96,97-149`;
`workbench/backend/adapters/models.py:28-33`; `workbench/backend/preflight.py:40-116,119-158,74-81`;
`workbench/backend/metrics_extraction.py:219-305,404`;
`workbench/backend/fashioniq_layout.py:83-89,92-136`;
`workbench/scripts/evaluate_models.py:130-131`;
`workbench/registry/auxiliary_assets.yaml:3-71,72-95`;
`workbench/patches/csmcir.md:1-97`; `workbench/patches/csmcir_models_init_compat.patch` (235 lines);
`CSMCIR/src/lavis/models/__init__.py:44-151,152,156`;
`CSMCIR/src/lavis/models/blip2_models/blip2_qformer_cir_align_prompt_csmcir.py:27-28,99-110,350-374,411-451,498-535`;
`CSMCIR/src/validate_blip_csmcir.py:12,24-57,149-208,501-508,565-630,618,891-892`;
`CSMCIR/src/utils_csmcir.py:13-16,97-103,161-162`;
`CSMCIR/src/data_utils_csmcir.py:16,175,216,223,226,408,416-419,435,466-470`;
`CSMCIR/requirements.txt:1-14`; `CSMCIR/README.md:4,14,115,117`;
`CSMCIR/src/lavis/configs/models/blip2/blip2_pretrain.yaml:10`.

## 14. ACHIEVED VERIFICATION LEVEL

**LEVEL 1** for the reproduction itself (no artifact, no run). Within static analysis I reached
LEVEL 4 for: pin identity, protocol identity, checkpoint metadata identity, patch identity and
patch effect, and the import-blocker count. I did **not** reach LEVEL 5 or 6: no checkpoint was
loaded into a model, no image was decoded, no metric was computed. Anything about what CSMCIR
*measures* is UNKNOWN.

Claims I am explicitly NOT making: that the declared checkpoint would reach 57.07/77.27/67.17 (UNKNOWN);
that the Memory Bank omission changes the score (UNKNOWN — no effect found on the eval path, and that
is all); that `COT_ours2` is recoverable anywhere (UNKNOWN — only the public trees were searched).
