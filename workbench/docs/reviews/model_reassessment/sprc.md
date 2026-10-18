# sprc — reproducibility reassessment (independent, adversarial, READ-ONLY)

Auditor: independent reassessment pass. Nothing under `/home/heheboiz/data/cir` was modified except
this file. All probe output went to `/tmp`. Every claim is STATIC_AUDIT (file read / command run,
no GPU), and each carries `file:line` or command+output. No RUNTIME_VERIFIED claim exists for this
model. Existing workbench audits were treated as hypotheses and re-checked against the pinned tree;
where my reading disagrees I say so and show the code.

Pinned checkout: `workbench/third_party/SPRC` at `2935a5397732260d1db6fa577e5926f963e36f0f`.

---

## 1. MODEL IDENTITY

| field | value | evidence |
|---|---|---|
| model_id | `sprc` | `workbench/registry/models.yaml:843` |
| method | SPRC — "Sentence-level Prompts Benefit Composed Image Retrieval" | `models.yaml:853`, `:857` |
| year | 2024 (ICLR 2024 Spotlight; pinned README line 2 says `【ICLR 2024, Spotlight】`) | `models.yaml:856`; `workbench/third_party/SPRC/README.md:2` |
| paper | https://arxiv.org/abs/2310.05473 | `models.yaml:858` |
| source URL | https://github.com/chunmeifeng/SPRC | `models.yaml:859` |
| pinned commit | `2935a5397732260d1db6fa577e5926f963e36f0f` | `models.yaml:861` |
| checkpoint variant | `sprc_fiq_unresolved` / `sprc_fiq.pt` / author OneDrive | `models.yaml:889-899` |
| reported metrics | R@10 54.92, R@50 74.97, mean 64.85 (+ discrepancy note) | `models.yaml:870-873` |
| native protocol | `fashioniq_original_split` | `models.yaml:868` |

STATIC_AUDIT.

**Agreement with the paper, verified from primary source.** ICLR 2024 camera-ready Table 1, Ours (SPRC)
row: `49.18 72.43 | 55.64 73.89 | 59.35 78.58 | 54.92 74.97 | 64.85`
(`https://proceedings.iclr.cc/paper_files/paper/2024/file/8204f54f6cd8f0db1765a274e94ceaac-Paper-Conference.pdf`,
fetched this pass, Table 1 body ~lines 120-140 of the extracted text). The paper states results are on
the FashionIQ **validation** set because test labels are undisclosed (same extraction, line ~150).
Reported values in the registry match the paper's printed aggregate row exactly.

The registry's own `reported_score_note` (`models.yaml:873`) is independently confirmed:
(49.18+55.64+59.35)/3 = 54.7233 ≠ 54.92 — the printed R@10 is not the mean of its printed categories.
UNKNOWN whether that is a transcription or a different aggregation; the paper gives no reconciliation.

## 2. SOURCE STATUS

- `source_available: true` (`models.yaml:862`), `source_dir: SPRC` (`:863`).
- OBSERVED `git -C workbench/third_party/SPRC rev-parse HEAD` → `2935a5397732260d1db6fa577e5926f963e36f0f` — equals `upstream_commit_sha` (`models.yaml:861`). Pin exact.
- OBSERVED tree: `README.md`, `requirements.txt`, `src/` with `__init__`-less top-level scripts
  `utils.py`, `validate_blip.py`, `validate_blip_rerank.py`, `cirr_test_submission.py`,
  `data_utils.py`, `blip_fine_tune_2.py`, `blip_validate.py`, plus a **vendored LAVIS** at `src/lavis/`
  (690 files, 82 dirs).
- Complete? The vendored LAVIS is NOT stripped. OBSERVED a mechanical check of
  `src/lavis/models/__init__.py`: **47 `from lavis...` import lines, 0 of them missing on disk**
  (script run from `src/lavis/`, comparing each dotted module against `.py` or `/__init__.py`). So the
  vendored package is internally complete — and this is a **correction of a hypothesis**: the
  `csmcir` registry entry's `source_blocker` (`models.yaml:~80`) describes 98 unguarded imports of
  absent modules in *CSMCIR's* copy; that defect does **not** apply to SPRC's copy. The two checkouts
  differ. Do not generalise the CSMCIR blocker to SPRC.
- Missing modules on the evaluation path: none from the source tree. The missing module is a
  third-party import (`clip`, §4).

SNAPSHOT 2 (current upstream), OBSERVED this pass:
- `git ls-remote origin HEAD` (from the pinned checkout) → `2935a5397732260d1db6fa577e5926f963e36f0f	HEAD`
- `git ls-remote --heads origin` → `2935a5397732260d1db6fa577e5926f963e36f0f	refs/heads/main`
- `git fetch origin --dry-run` → no output (nothing to fetch)
- `git rev-list -1 origin/main` == `git rev-list -1 HEAD` == `2935a539…`

**The pinned commit IS upstream `main` HEAD.** Upstream has not moved since 2024-04-16
(`git log -1 --format=%H %ad %an %s` → `2935a5397732260d1db6fa577e5926f963e36f0f 2024-04-16 21:29:29 +0800 Yang Bai Update README.md`).
The pin was NOT changed. No authors' CLI was added, no dependency fixed, no checkpoint published,
no move. Upstream is dormant.

Note: `upstream_default_branch: null` (`models.yaml:860`) while the remote's default branch is
`main` (observed via `ls-remote --heads` + `remotes/origin/HEAD -> origin/main`). Cosmetic registry
gap, P3.

## 3. NATIVE EVALUATION WORKFLOW

**Entrypoint** `src/blip_validate.py` (the adapter's `script`, `workbench/backend/adapters/models.py:121`).

**Argument parser, quoted IN FULL** — `workbench/third_party/SPRC/src/blip_validate.py:141-155`:

```
141 if __name__ == '__main__':
142     parser = ArgumentParser()
143     parser.add_argument("--dataset", type=str, required=True, help="should be either 'CIRR' or 'fashionIQ'")
144     parser.add_argument("--blip-model-name", default="blip2_cir_rerank_learn", type=str)
145     parser.add_argument("--backbone", type=str, default="pretrain", help="pretrain for vit-g, pretrain_vitL for vit-l")
146     parser.add_argument("--model-path", type=str)
147
148     args = parser.parse_args()
149     if args.dataset.lower() not in ['fashioniq', 'cirr']:
150         raise ValueError("Dataset should be either 'CIRR' or 'FashionIQ")
151
152     if args.dataset.lower() == 'cirr':
153         blip_validate_cirr(args.blip_model_name, args.backbone, args.model_path)
154     elif args.dataset.lower() == 'fashioniq':
155         clip_finetune_fiq(['dress', 'toptee', 'shirt'], args.blip_model_name, args.backbone, args.model_path)
```

That is the complete argument surface: exactly four flags.

| argument | required | default | type |
|---|---|---|---|
| `--dataset` | **YES** (`required=True`) | — | str; must lower-case into `{'fashioniq','cirr'}` else `ValueError` |
| `--blip-model-name` | no | **`"blip2_cir_rerank_learn"`** | str |
| `--backbone` | no | `"pretrain"` | str |
| `--model-path` | no (no `required=True`) | **`None`** | str |

**Function chain** (all in the pinned tree):
`blip_validate.py:155` → `clip_finetune_fiq` (`:26`) → `load_model_and_preprocess(name=blip_model_name,
model_type=backbone, is_eval=False, device=device)` (`:45`) → `torch.load(model_path)` (`:48`) →
`blip_model.load_state_dict(checkpoint[blip_model.__class__.__name__], strict=False)` (`:49`) →
`targetpad_transform(1.25, 224)` (`:53`) → per category `dress,toptee,shirt`: `FashionIQDataset('val',[cat],'relative',preprocess)`
(`:64`) and `('classic')` (`:66`) → `extract_index_blip_features` (`:80`) → `compute_fiq_val_metrics`
(`:81`, defined `validate_blip.py:24`) → `blip_model.inference(reference_image_features, feature_curr, input_captions)`
(`validate_blip.py:198`) → prints one JSON object (`:88-98`).

**CLI?** Yes, but with a hard-coded model class name supplied by the caller and no dataset-root flag.

**What a wrapper must call:** `python src/blip_validate.py --dataset FashionIQ --blip-model-name <registered LAVIS name> --backbone <pretrain|pretrain_vitL> --model-path <sprc_fiq.pt>`,
executed with the checkout's `src/` importable and `<checkout>/fashionIQ_dataset` present
(`data_utils.py:12`, `:149`, `:155`, `:185`; no dataset-root option exists).

**Missing components** (see §4, §5): `clip` import; `transformers` version compat; a locally present
checkpoint; a resolvable model class name; the gallery-id extension contract.

### Does the workbench adapter's command line satisfy that parser?

The adapter is `workbench/backend/adapters/models.py:119-121`:

```
119 class SPRCAdapter(OfficialScriptAdapter):
120     model_id = "sprc"
121     script = "src/blip_validate.py"
```

It declares **no `command()`**. OBSERVED, executed:

```
$ .venv/bin/python -c "from workbench.backend.adapters.models import ADAPTERS, OfficialScriptAdapter; \
    print(ADAPTERS['sprc']().command(None,None,None))"
NotImplementedError upstream evaluation command requires audit before execution
```

and

```
$ .venv/bin/python -c "... from workbench.scripts.evaluate_models import command_is_audited; \
    print(command_is_audited(ADAPTERS['sprc']))"
command_is_audited: False
command is base impl: True
```

`OfficialScriptAdapter.command` raises (`adapters/models.py:22-23`); `command_is_audited` is exactly
"the class overrides `command`" (`workbench/scripts/evaluate_models.py:130-131`). Therefore:

- OBSERVED `guarded_plan(model, checkpoint, 'fashioniq_original_split', None, 50)` for `sprc` →
  `(None, ['SKIPPED: adapter command not audited'])`.
- `blockers()` still reports `command not runnable: COMMAND_BLOCKED_MODEL_MAPPING` via
  `workbench/backend/preflight.py:184-206` (`command_status` at `models.yaml:875`).

**Answer: there is no command line to compare. The adapter cannot satisfy the parser because it
constructs no command at all**, and a hypothetical `command()` would have to pass
`--dataset`, `--blip-model-name`, `--backbone`, `--model-path` — all four are parsed, three would
need explicit values (the two defaults are unusable, §4/§9). Also, `build_command` would first run
`validate_request` (`adapters/models.py:27-36` in `adapters/base.py:27-36`), which raises
`checkpoint_mapping_unverified` because `checkpoint_mapping_status: UNVERIFIED`
(`models.yaml:899`). Severity **P0** (the model has no executable path).

## 4. DEPENDENCIES

The task's D1–D8 categories are not defined anywhere in this repo (grepped: `D1`, "dependency
categor" → only unrelated hits in `workbench/docs/reviews/final_pc_audit/`). I therefore classify by a
stated scheme and name it explicitly: **D-import = the import statement's own failure mode**
(D-missing-package, D-version-incompatible, D-unvendored-local, D-dead-import). I do not invent
D1–D8 numbers I cannot source.

| import | file:line | owner | required version | installable? | py/torch/cuda | safe remedy | test result |
|---|---|---|---|---|---|---|---|
| `import clip` | `src/blip_validate.py:8` | Openai CLIP (upstream, 3rd-party) | not pinned anywhere in the tree | from source: `pip install git+https://github.com/openai/CLIP.git` (PyPI name `clip` is an unrelated 2012 clipboard CLI, verified `https://pypi.org/pypi/clip/json` → "A CLI clipboard manager", 0.2.0, 2013) | py3.9/torch2.0.1, CPU-only import | add the distribution to the env spec + a pin; the FashionIQ path does not *use* it (§ note) | OBSERVED `ModuleNotFoundError: No module named 'clip'` |
| `from lavis.models import load_model_and_preprocess` | `src/blip_validate.py:16` → `src/validate_blip.py:12` | vendored at `src/lavis` (in-tree) | n/a (vendored) | yes, in-tree | py3.9 | none needed for the file | OBSERVED `ImportError: cannot import name 'apply_chunking_to_forward' from 'transformers.modeling_utils'` under the workbench venv (transformers 4.57.6) |
| `transformers` (transitively, `lavis.models.med:40-43`, `lavis.models.blip2_models.Qformer:41`, `lavis.models.blip_models.nlvr_encoder:21`) | `src/lavis/models/med.py:40` | 3rd-party | **must export `apply_chunking_to_forward` from `transformers.modeling_utils`** | yes (a pin) | py3.9 | pin the version the vendored LAVIS was written against | OBSERVED: `transformers` 4.36.2 DOES export it — `curl -sSL .../transformers/v4.36.2/src/transformers/modeling_utils.py | grep -c apply_chunking_to_forward` → `1` (line 46). The workbench venv has 4.57.6, which does not (`hasattr` → `False`). So the registry's declared `transformers==4.36.2` (`models.yaml:885`) is the correct dissolver of THIS failure; the venv that produced the ImportError is not the declared env |
| `pandas` | `src/blip_validate.py:10` | 3rd-party | `pandas==2.2.0` (`requirements.txt:1`) | yes | — | none | dead import on both paths (`pd.` never used in the file) — D-dead-import |
| `numpy`, `torch`, `torch.nn.functional`, `tqdm`, `torch.optim.lr_scheduler` | `src/blip_validate.py:9,11,12,15,17` | 3rd-party | as README (`README.md:12-17`: `torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2`) | yes | py3.9 / torch 2.0.1 | none | present in the workbench venv |
| `salesforce-lavis==1.0.2` | declared `models.yaml:885` | 3rd-party (NOT used by the evaluator — the tree vendors LAVIS) | 1.0.2 | yes (`https://pypi.org/pypi/salesforce-lavis/json`, 1.0.2 exists) | — | none; it is a fallback, and it would be a *second* divergent LAVIS next to the vendored one | not exercised |
| `spacy==3.7.2`, `timm==0.9.12` | `models.yaml:885`, `requirements.txt:5,13` | 3rd-party | pinned | yes | — | none | see §10 for the timm caveat |

**Verdict on §4:** the registry's `environment` block (`models.yaml:876-885`, `confidence: VERIFIED_REQUIREMENTS`)
correctly transcribes the two README pins and the `requirements.txt` packages, and it **omits
`clip`**, which the README recipe also omits — i.e. the registry faithfully mirrors an incomplete
upstream recipe. The dependency gap is upstream's, and its category is
**INSTALLABLE_DEPENDENCY_MISSING** (P1) plus **DEPENDENCY_VERSION_CONFLICT** for the
`transformers` lower bound — the latter is resolvable by honouring the already-declared
`transformers==4.36.2`, verified above.

Honest qualifier on the `clip` gap: on the FashionIQ path `clip` is *not called* — `clip.load` is
commented out at `src/blip_validate.py:44`, and the only live `clip` uses are in the CIRR paths
(`validate_blip.py:444`, `validate_blip_rerank.py:8,12`). The import at line 8 still executes first,
so it remains a hard blocker for the process, but it is a **one-line env fix, not a science risk**.

## 5. CHECKPOINTS

Declared variant (`models.yaml:889-899`): `sprc_fiq.pt`, `download_url: null`,
`source_url: https://1drv.ms/u/s!Aj0q22vyiZbabnUya4mnufIBtYI?e=n4ZVKj`, `source_type: author_onedrive`,
`expected_size_bytes: null`, `expected_sha256: null`, `status: CHECKPOINT_MAPPING_UNVERIFIED`.

Published links in the pinned README (`README.md:123-128`): OneDrive `sprc_fiq.pt` and `sprc_cirr.pt`;
BaiduCloud mirror with password `t1at`; Google-Drive `sprc_fiq_vitl.pt` / `sprc_cirr_vitl.pt` ("models
based on vit-l"). `README.md:130-132` "Todo: code and pre-trained weights for rerank model".

**URL test (no body downloaded), as instructed:**

```
$ curl -sS -I -L --max-time 30 "https://1drv.ms/u/s!Aj0q22vyiZbabnUya4mnufIBtYI?e=n4ZVKj"
HTTP/2 301
location: https://onedrive.live.com/:u:/g/personal/DA9689F26BDB2A3D/s!Aj0q22vyiZbabnUya4mnufIBtYI?resid=DA9689F26BDB2A3D!110&e=n4ZVKj&migratedtospo=true&redeem=...
HTTP/2 403
date: Sun, 18 Oct 2026 06:59:58 GMT
```

**HTTP 403 on the final hop. Classification: `CHECKPOINT_URL_INACCESSIBLE` (P0), evidence above.**

Controls run this pass to prove the 403 is OneDrive-specific and not a local network/UA artifact:

```
# same repo's CIRR link, same host:
$ curl -sS -I -L --max-time 30 "https://1drv.ms/u/s!Aj0q22vyiZbabxR_hjpQ_DmYX3s?e=k9fmt1"
HTTP/2 301 ... HTTP/2 403

# README's Google-Drive link resolves fine from this host:
$ curl -sS -o /dev/null -w "%{http_code} %{url_effective}\n" -L --max-time 30 \
    "https://drive.google.com/file/d/11p95msuuTAU6Pej7d_a-KjUHHpP3kH8Y/view?usp=sharing"
200 https://drive.google.com/file/d/11p95msuuTAU6Pej7d_a-KjUHHpP3kH8Y/view?usp=sharing

# with a browser UA and &download=1 the OneDrive link lands on a sign-in page:
$ curl ... -A "<Chrome UA>" "https://1drv.ms/...&download=1"
status=200 size=36188 ctype=text/html final=https://login.live.com/login.srf?wa=wsignin1.0&...
```

Interpretation (STATIC_AUDIT, careful): a plain UA-less HEAD reaches `onedrive.live.com` and 403s; a
browser UA with `&download=1` bounces to `login.live.com`. So the sharing link is **not anonymously
downloadable by a non-browser client from this host**. Whether a human browser session can still
fetch it is UNKNOWN — I did not and cannot click through. The registry's decision to leave
`expected_sha256`/`expected_size_bytes` null is therefore **honest and correct**; a digest cannot be
computed without the bytes.

Second published variant `sprc_fiq_vitl.pt` (README line 128) is **not registered** — `models.yaml:888-899`
has one variant. If the paper's row used the ViT-L model, the registered artifact is the wrong one.
UNKNOWN which was used for the paper's Table 1; the paper says "initialized from the BLIP-2 pretrained
model with ViT-G" (camera-ready, §4.1 prose).

Mapping confidence: **UNVERIFIED**, see §9. Local checkpoint absent: OBSERVED
`workbench/artifacts/checkpoints/` contains only `.gitkeep` + a LIMN bundle + `download_manifest.json`
(LIMN entries only); `find workbench/artifacts -iname "*sprc*"` → one mock result JSON.

Loader contract: `torch.load(model_path, map_location=device)` then
`checkpoint[blip_model.__class__.__name__]` with `strict=False` (`src/blip_validate.py:48-49`) —
i.e. the checkpoint must be a dict keyed by the **class name of whatever LAVIS class the
`--blip-model-name` resolved to**, and a mismatch is silently tolerated (only `missing_keys` printed).
That makes a wrong-class run *silently* produce near-random metrics rather than crash — a
scientific-integrity hazard, P1 (see §9).

## 6. AUXILIARY ASSETS

| name | purpose | source | availability | generation possible? | evidence |
|---|---|---|---|---|---|
| LAVIS BLIP-2 pretrain weights | base weights the pinned `blip2_pretrain.yaml` loads | `https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained.pth` | **reachable**: `curl -I` → `HTTP/2 200`, `content-length: 746998955`, `x-goog-stored-content-length: 746998955` | no (must be downloaded) | `src/lavis/configs/models/blip2/blip2_pretrain.yaml:10`; `workbench/registry/auxiliary_assets.yaml:204-218` declares `expected_size_bytes: 746998955` — **matches the observed header** |
| EVA-ViT-g backbone weights | vision tower for `vit_model="eva_clip_g"` | `https://storage.googleapis.com/.../eva_vit_g.pth` | **reachable**: `HTTP/2 200`, `content-length: 2025249237` | no | `src/lavis/models/eva_vit.py:442-443` hard-coded URL; `auxiliary_assets.yaml:219-232` declares `2025249237` — **matches** |
| bert tokenizer/`bert-base-uncased` config | `Blip2Base.init_tokenizer`, Q-Former init | Hugging Face hub via `transformers` | reachable (network fetch at runtime) | no | `src/lavis/models/blip2_models/blip2.py:32,48,55`. Not declared as an auxiliary asset for sprc — P3 documentation gap (other LAVIS models declare it; `auxiliary_assets.yaml` has exactly two sprc rows) |
| FashionIQ images/captions/splits | the benchmark | operator-supplied; `canonical_dataset_link: fashionIQ_dataset` (`models.yaml:848`) | link **not staged** — `ls workbench/third_party/SPRC/fashionIQ_dataset` → `No such file or directory`; source exists at `data/fashionIQ_dataset` | no | `data_utils.py:12,149,155,185`; `README contracts` `workbench/registry/preparation_contracts.yaml:220-248` |

**Finding (P1), disagreeing with the workbench's own primary list.** The two cache assets are
**decoys on this host**. OBSERVED:

```
$ ls -la /home/heheboiz/.cache/lavis/blip2_pretrained.pth
-rw-rw-r-- 1 heheboiz heheboiz 22 Oct 14 14:20 .../blip2_pretrained.pth
$ file .../blip2_pretrained.pth
ASCII text, with no line terminators
$ ls /home/heheboiz/.cache/torch/hub/checkpoints/
ls: cannot access '.../checkpoints/': No such file or directory
```

22 ASCII bytes where 746,998,955 binary bytes are required, and the declared EVA-ViT-g destination
does not exist at all. `workbench/docs/READINESS_MATRIX.md:63` reports these as
"operator-supplied unverified provenance" — true, but weaker than the fact: they are **not the
assets**, and the registry declares `expected_size_bytes` for both while the *preflight* is reported
as a provenance problem. Any future run must not treat their presence as readiness. Registry declares
no `expected_sha256` for either (`auxiliary_assets.yaml:217,232`), so size is the only check available;
the declared sizes are verified correct above, so a size gate is installable now.

## 7. FASHIONIQ PROTOCOL

Registry: `native_protocol: fashioniq_original_split`, `supported_protocols: [fashioniq_original_split]`,
`literature_split_label: original` (`models.yaml:867-869`). Workbench definition:
`src/evaluation/fashioniq.py:172-174` — `protocol_id="fashioniq_original_split", gallery="full_split",
reference_excluded=False, aliases=("fashioniq_original",)`.

Verified from the pinned evaluator:

- **Gallery** = the complete ordered `image_splits/split.{cat}.val.json` list, one category per pass:
  `data_utils.py:153-156` loads `self.image_names` from exactly that file; `utils.py:54-55` uses
  `DataLoader(..., shuffle` unset) so order is preserved; `validate_blip.py:45-46` sorts the full
  `index_names` array. `fashioniq_original_split.gallery == "full_split"` ✓.
- **Reference policy** = reference REMAINS ELIGIBLE. `validate_blip.py:44-57` builds `labels` against
  `target_names` only, sorts all gallery names, and never masks or removes the reference. Contrast the
  CIRR path, which *does* mask: `validate_blip.py:258-260` and `:331-333`
  (`reference_mask = ... sorted_index_names[reference_mask]`). `reference_excluded=False` ✓.
- **Query construction**: relative-mode batch yields `(reference_name, target_name, captions)` for
  `split=='val'` (`data_utils.py:174-176`); captions are two strings joined
  `f"{f0.strip('.?, ').capitalize()} and {f1.strip('.?, ')}"` (`validate_blip.py:180-183`) then passed
  through `txt_processors["eval"]`. Deterministic. ✓
- **Categories**: hard-coded `['dress','toptee','shirt']` at `blip_validate.py:155`. ✓
- **Recall computation**: `distances = 1 - pred_sim` (`validate_blip.py:44`), ascending `argsort`
  (`:45`), one-hot label by target equality (`:49-50`), one-hit assert (`:51`),
  `recall_at10 = labels[:, :10].sum()/len(labels)*100`, `recall_at50` likewise (`:54-55`).
  Aggregation `mean` per metric + `average_recall = (mean_r50 + mean_r10)/2` (`blip_validate.py:88-96`). ✓
- **Metric parser**: `workbench/backend/metrics_extraction.py:408-411` routes `sprc` to
  `parse_csmcir_output`, whose key set (`{cat}_recall_at10/_at50`, `average_recall_at10/_at50`,
  `average_recall`) is exactly what `blip_validate.py:88-96` prints, and which cross-checks the
  declared macro against the categories (`metrics_extraction.py:271-279`). ✓

**One semantic caveat, carried forward from the prior audit and re-verified:** the *ranking model*
does not condition on the reference for the class the default names. OBSERVED
`blip2_qformer_cir_rerank.py:373-397`: `inference(...)` uses `reference_embeds` **only** for
`.device` (`:381`), then scores `text_proj(text CLS)` against `target_feats` (`:387-396`). By contrast
`blip2_qformer_cir_align_prompt.py:313-340` **does** cross-attend the reference. This is a
model-identity matter, not a gallery matter: `reference_excluded=False` describes the ranking pool and
is unaffected. P2, restated for the class-unknown case in §9.

**Protocol matches the paper?** PARTIALLY, and I mark the remainder UNKNOWN. The paper reports
FashionIQ **validation** Recall@K with no candidate-set statement (camera-ready §4.1, visible in the
fetched text: "the ground-truth labels for the test set ... not publicly disclosed, we adopt the
results on the validation set"). `literature_split_label: original` is consistent with that; the
"full split, reference eligible" reading is an **inference from code**, which is the only authority
available. Registry is right to record the code-derived protocol; it does not over-claim.

## 8. WORKBENCH INTEGRATION

| layer | state | evidence |
|---|---|---|
| adapter | `SPRCAdapter` with `script` only; **no `command()`** → not audited | `adapters/models.py:119-121`; `evaluate_models.py:130-131`; OBSERVED `command_is_audited` → False |
| planner | `guarded_plan` short-circuits: `(None, ['SKIPPED: adapter command not audited'])` | `evaluate_models.py:361-365`; OBSERVED |
| preflight | reports `command not runnable: COMMAND_BLOCKED_MODEL_MAPPING`, asset provenance, missing canonical link, unverified env | `preflight.py:184-208`; `models.yaml:875`; `READINESS_MATRIX.md:63` |
| checkpoint prep | none: `download_url: null` so no automatic download; `source_url` is a 403 OneDrive share | `models.yaml:891-892`; OBSERVED 403 |
| metric parser | wired: `sprc → parse_csmcir_output` | `metrics_extraction.py:408-411` |
| per-query support | none declared | no `per_query_observer` key on the sprc entry (`models.yaml:843-899`) |
| status fields | `reproduction_status: BLOCKED_CHECKPOINT_MAPPING`, `command_status: COMMAND_BLOCKED_MODEL_MAPPING`, `replay_status: BLOCKED_CHECKPOINT_MAPPING` | `models.yaml:850,874,875` |
| `evaluate_models.py` "search for sprc" | **the file contains ZERO occurrences of `sprc`** — OBSERVED `grep -c sprc workbench/scripts/evaluate_models.py` → `0` | assignment asked me to find the adapter there; it is not there, it is in `workbench/backend/adapters/models.py:119-121` (registered into `ADAPTERS` at `:155`) |

The assignment's premise ("find it in `workbench/scripts/evaluate_models.py`") is **REFUTED**:
`evaluate_models.py` consumes the adapter registry (`from workbench.backend.adapters.models import
ADAPTERS, OfficialScriptAdapter`, `evaluate_models.py:27`) but names no model id in its own source.
P3 documentation/navigation note only; the audit trail is still intact.

## 9. CONFIRMED DEFECTS

**F1 — `--blip-model-name` default names a class the pinned repo does not register.** P0.
- Root cause: `blip_validate.py:144` default `"blip2_cir_rerank_learn"`. OBSERVED `grep -rn
  "rerank_learn"` over the checkout (excluding `__pycache__`) → exactly **one** hit, that same line.
  Every registered name in the tree: `grep -rn "@registry.register_model" src/lavis/models/blip2_models/*.py`
  → `blip2`/`blip2_feature_extractor` (`blip2_qformer.py:25-26`), `blip2_image_text_matching`,
  `blip2_instruct_cir`, `blip2_opt`, `blip2_t5*`, `blip2_cir_align_prompt`
  (`blip2_qformer_cir_align_prompt.py:25`), `blip2_cir_cat` (`:26`), `blip2_cir_cls`,
  `blip2_cir_full`, `blip2_cir_neg`, `blip2_cir_neg_rel`, `blip2_cir_rel`, **`blip2_cir_rerank`
  (`blip2_qformer_cir_rerank.py:26`)**, `blip2_cir_slt`, `blip2_cir_z_learn_pos_align`, `blip2_cir_prompt`.
  No `*_learn`.
- Reproduction (static): `load_model_and_preprocess(name="blip2_cir_rerank_learn", ...)` →
  `lavis.models.__init__:223` `model_cls = registry.get_model_class(name)` → `registry.py:236-237`
  `return cls.mapping["model_name_mapping"].get(name, None)` → `None` → `model_cls.from_pretrained`
  raises `AttributeError: 'NoneType' object has no attribute 'from_pretrained'`
  (`lavis/models/__init__.py:226`). I could not execute this in the venv because the import of
  `lavis` itself fails first (§4), so the failure is **derived from source, not observed**. The prior
  audit's line (`workbench/docs/reviews/models/sprc.md`: "`registry.get_model_class` is …") agrees; I
  confirm it independently from `registry.py:236-237`.
- Fix: the class is a **property of `sprc_fiq.pt`**, which is unreachable (§5). Passing a wrong-but-
  registered name is not a fix: `load_state_dict(..., strict=False)` (`blip_validate.py:49`) will
  accept it and print only `Missing keys` — a silent wrong-model run. **Do not guess.**
- Regression: cannot be run.

**F2 — the checkpoint cannot be fetched.** P0, `CHECKPOINT_URL_INACCESSIBLE` (§5). `download_url: null`
means the workbench has no automatic path; the share URL 403s for non-browser clients.

**F3 — `clip` is imported but never installed by the README recipe.** P1,
`INSTALLABLE_DEPENDENCY_MISSING`. `blip_validate.py:8`; `requirements.txt` (14 lines, quoted in §4)
has no `clip` line; README install block (`README.md:9-17`) installs only `torch*` + `-r requirements.txt`.
OBSERVED `ModuleNotFoundError: No module named 'clip'`.

**F4 — vendored-LAVIS vs modern `transformers`.** P1, `DEPENDENCY_VERSION_CONFLICT`.
OBSERVED in the workbench venv (3.13.12 / transformers 4.57.6):
`ImportError: cannot import name 'apply_chunking_to_forward' from 'transformers.modeling_utils'`
raised at `src/lavis/models/med.py:40` while importing `lavis.models`. OBSERVED the declared
`transformers==4.36.2` exports it. So this is an env-selection defect, **not** an upstream source
defect; the honest root cause is "whoever ran it used the workbench venv instead of the declared
`sprc` env".

**F5 — decoy auxiliary assets pass as present.** P1 (§6). 22-byte `blip2_pretrained.pth`; EVA-ViT-g
absent. Registry has declared correct sizes (`auxiliary_assets.yaml:216,231`) that match the live
`content-length` (746998955 / 2025249237) → a size gate is available and would catch both.

**F6 — second published checkpoint variant unregistered.** P2 (§5). `README.md:128`
`sprc_fiq_vitl.pt` (ViT-L) exists; `models.yaml` registers only the ViT-g-shaped `sprc_fiq.pt`.
UNKNOWN which one produced the paper row.

**F7 — assignment's premise about the adapter location.** P3 (§8). Not a workbench defect; recorded so
the audit trail is not silently "corrected".

**F8 — `upstream_default_branch: null` while remote default is `main`.** P3 (`models.yaml:860` vs
OBSERVED `refs/heads/main`). Metadata only; the pin is a SHA, so nothing depends on it.

**Withdrawn/refuted hypotheses (stated because they are widely repeated in the repo's docs):**
- "the pinned SPRC LAVIS cannot import because 98 unguarded imports reference absent modules" — this is
  **CSMCIR's** blocker, not SPRC's. SPRC's vendored tree is complete (47/47 imports resolve). Do not
  reuse that sentence for this model.
- "`prepare_dataset.py:prepare_csmcir` is CSMCIR-only and sprc is not auto-linked" (in
  `workbench/docs/reviews/models/sprc.md`'s refresh block) — outside this ticket's scope; not
  re-verified here, so I neither confirm nor deny it. Marked UNKNOWN deliberately.

## 10. REMAINING BLOCKERS

| # | classification | owner | evidence | remediation |
|---|---|---|---|---|
| B1 | **FIRST_PARTY_ADAPTER_MISSING** | workbench | `adapters/models.py:119-121` has no `command()`; `command_is_audited` False; `guarded_plan` → `SKIPPED` | implement `SPRCAdapter.command` once the class mapping is known; it must emit `--dataset FashionIQ --blip-model-name <name> --backbone <pretrain|pretrain_vitL> --model-path <ckpt>` and must refuse a mapping-unverified checkpoint (the existing `validate_request` already refuses — keep it) |
| B2 | **CHECKPOINT_URL_INACCESSIBLE** | upstream (author) | OBSERVED 403 on `https://1drv.ms/u/s!Aj0q22vyiZbabnUya4mnufIBtYI?e=n4ZVKj` | author re-publishes on an anonymous-download host; only the author can. `CHECKPOINT_UNPUBLISHED` is NOT the right token — the file is published, the URL just refuses anonymous clients |
| B3 | **INSTALLABLE_DEPENDENCY_MISSING** | workbench env spec | `blip_validate.py:8`; `requirements.txt` (no `clip`) | add the Openai-CLIP distribution to the sprc env spec with a pin; harmless even though the FashionIQ path never calls it |
| B4 | **DEPENDENCY_VERSION_CONFLICT** | workbench env selection | OBSERVED ImportError in venv transformers 4.57.6; declared pin 4.36.2 exports the symbol | build the env from `models.yaml:876-885`; do not run the evaluator under the workbench venv |
| B5 | **PUBLIC_ASSET_NOT_PREPARED** | operator | 22-byte `blip2_pretrained.pth`; EVA-ViT-g destination absent; canonical dataset link absent | fetch both (URLs verified 200, sizes match the registry's declarations) and stage `fashionIQ_dataset` |
| B6 | **SCIENTIFIC_PROTOCOL_UNVERIFIED** | upstream paper | paper names the val set and Recall@K but no candidate set | keep the source-derived `fashioniq_original_split`; do not change the gallery/reference policy |
| B7 | **GPU_RUNTIME_UNTESTED** | environment | no GPU run performed here | the whole §7 path is unexecuted |
| B8 | **UNPUBLISHED_SOURCE_MODULE** (candidate, must not be assumed) | upstream | `blip_validate.py:144` names `blip2_cir_rerank_learn`; nothing in the tree registers it; README line 130-132 "Todo: code and pre-trained weights for rerank model" | treat as an upstream descriptive gap; do **not** substitute `blip2_cir_rerank` unless the checkpoint's key proves it |

**What must NOT be done:** do not switch the protocol to Protocol B/C to sidestep anything (the
registry's `native_protocol: fashioniq_original_split` at `models.yaml:868` is source-derived and
correct); do not change the caption input, the noise condition, or the gallery; do not write an
invented `command()` that hard-codes `blip2_cir_rerank`.

## 11. IMPLEMENTATION PLAN

Change list, in dependency order (all mine would be workbench-side except C1/C3 which need upstream):

- **C1 (workbench, needed now):** add the Openai-CLIP distribution to the sprc env spec and record its
  pin next to `models.yaml:885`. Complexity: trivial (one line + an env-verification entry). Scientific
  risk: none (unused on the FashionIQ path). Expected output: `import clip` stops failing.
- **C2 (workbench, needed now):** make the preflight asset check reject size implausibility using the
  already-declared `expected_size_bytes` (`auxiliary_assets.yaml:216,231`), because the registry's
  declared values match the live `content-length` (verified). Complexity: small. Risk: none.
- **C3 (blocked on upstream):** obtain `sprc_fiq.pt` and read its top-level key. Only that key
  authorises `--blip-model-name`. Complexity: one `torch.load` + key read once the bytes exist. Risk:
  high if skipped — `strict=False` hides a wrong class.
- **C4 (workbench, blocked on C3):** write `SPRCAdapter.command` with the four flags quoted in §3 and
  an explicit `--backbone` derived from the same evidence. Complexity: small. Risk: none if C3 holds.
- **C5 (bookkeeping):** set `upstream_default_branch: main`; register `sprc_fiq_vitl.pt` as a second
  variant with `checkpoint_mapping_status: UNVERIFIED`; add the `bert-base-uncased` tokenizer as an
  auxiliary asset. Complexity: trivial.

Nothing in this plan touches `src/evaluation/fashioniq.py`, the gallery, the reference policy, the
captions or the noise condition.

## 12. FINAL VERDICT

**BLOCKED** — with a precise split, so no reader can mistake what is verified for what is not:

- The **source** is accessible, complete, exactly pinned, and upstream has not moved since 2024-04-16.
- The **protocol** (`fashioniq_original_split`), **gallery** (full ordered val split per category),
  **reference policy** (reference eligible, no mask on the FashionIQ path), **query construction**,
  **preprocessing** (TargetPad 1.25 @ 224) and **metric mathematics** all trace faithfully to the pinned
  source, and the paper's own values are recorded correctly (Table 1 Ours row matches to the digit).
- The **model cannot be run**: no adapter command exists (`FIRST_PARTY_ADAPTER_MISSING`), the
  checkpoint URL returns 403 (`CHECKPOINT_URL_INACCESSIBLE`), the checkpoint's class key is
  undeterminable from README+code (`UNPUBLISHED_SOURCE_MODULE` candidate), the `clip` import is not in
  the upstream recipe, the declared `transformers` pin is not what the observed venv had, and the
  declared cache assets on this host are not the assets.

Per the required vocabulary: **FIRST_PARTY_ADAPTER_MISSING** (P0) · **CHECKPOINT_URL_INACCESSIBLE**
(P0) · **CHECKPOINT_UNPUBLISHED — NO, explicitly rejected** · **INSTALLABLE_DEPENDENCY_MISSING** (P1) ·
**DEPENDENCY_VERSION_CONFLICT** (P1) · **PUBLIC_ASSET_NOT_PREPARED** (P1) · **SCIENTIFIC_PROTOCOL_UNVERIFIED**
(P3) · **GPU_RUNTIME_UNTESTED**.

Two of the three model-side repairs (C3, and the class name C4 depends on) **require upstream
cooperation or a human who already holds the checkpoint** — the author must either republish the
weights on an anonymous-download host or someone must supply the `.pt` file.

## 13. EVIDENCE

Repo: `https://github.com/chunmeifeng/SPRC` · pinned `2935a5397732260d1db6fa577e5926f963e36f0f`
(== `git ls-remote origin HEAD`, == upstream `refs/heads/main`).

Files read (all under `workbench/third_party/SPRC`, pinned):
`README.md`, `requirements.txt`, `src/blip_validate.py`, `src/validate_blip.py`, `src/data_utils.py`,
`src/utils.py`, `src/lavis/models/__init__.py`, `src/lavis/models/base_model.py`,
`src/lavis/common/registry.py`, `src/lavis/configs/models/blip2/blip2_pretrain.yaml`,
`src/lavis/models/blip2_models/blip2.py`, `blip2_qformer_cir_align_prompt.py`,
`blip2_qformer_cir_rerank.py`, `blip2_qformer_cir_cat.py`, `src/lavis/models/eva_vit.py`;
plus `workbench/registry/models.yaml`, `auxiliary_assets.yaml`, `preparation_contracts.yaml`,
`workbench/backend/adapters/models.py`, `adapters/base.py`, `workbench/backend/preflight.py`,
`workbench/backend/metrics_extraction.py`, `workbench/backend/registry.py`,
`workbench/scripts/evaluate_models.py`, `src/evaluation/fashioniq.py`, `workbench/docs/READINESS_MATRIX.md`,
and the prior hypotheses `workbench/docs/reviews/models/sprc.md` / `final_model_audit/sprc.md` (tested,
not paraphrased).

Commands run and observed results (read-only; all artifacts under `/tmp`):
1. `git -C workbench/third_party/SPRC rev-parse HEAD` → `2935a5397732260d1db6fa577e5926f963e36f0f`
2. `git -C … ls-remote origin HEAD` → same SHA; `ls-remote --heads` → `refs/heads/main` same SHA;
   `fetch origin --dry-run` → empty
3. `git -C … log -1 --format='%H %ad %an %s' --date=iso` → `2935a5397732… 2024-04-16 21:29:29 +0800 Yang Bai Update README.md`
4. `curl -sS -I -L --max-time 30 "<oneDrive sprc_fiq>"` → `HTTP/2 301` → **`HTTP/2 403`**
5. `curl -sS -o /dev/null -w '%{http_code}' -L "<google drive sprc_fiq_vitl>"` → `200`
6. `curl -sS -I -L "<GCS blip2_pretrained.pth>"` → `200`, `content-length: 746998955`;
   `<GCS eva_vit_g.pth>` → `200`, `content-length: 2025249237`
7. `grep -rn "rerank_learn" src/` → 1 hit, `src/blip_validate.py:144`
8. `grep -rn "@registry.register_model" src/lavis/models/blip2_models/*.py` → 19 registrations, none `*_learn`
9. mechanical import check of `src/lavis/models/__init__.py` → 47 lavis imports, 0 missing
10. `.venv/bin/python -c "import clip"` → `ModuleNotFoundError: No module named 'clip'`
11. `.venv/bin/python -c "from lavis.models import load_model_and_preprocess"` →
    `ImportError: cannot import name 'apply_chunking_to_forward' from 'transformers.modeling_utils'`
12. `curl -sSL https://raw.githubusercontent.com/huggingface/transformers/v4.36.2/src/transformers/modeling_utils.py | grep -c apply_chunking_to_forward` → `1`
13. `.venv/bin/python -c "…command_is_audited(ADAPTERS['sprc'])…"` → `False`; `.command(None,None,None)` →
    `NotImplementedError upstream evaluation command requires audit before execution`
14. `.venv/bin/python -c "guarded_plan(sprc, variant, 'fashioniq_original_split', None, 50)"` →
    `(None, ['SKIPPED: adapter command not audited'])`
15. `grep -c sprc workbench/scripts/evaluate_models.py` → `0`
16. `ls -la /home/heheboiz/.cache/lavis/blip2_pretrained.pth` → 22 bytes, `file` → `ASCII text`;
    `/home/heheboiz/.cache/torch/hub/checkpoints/` → `No such file or directory`
17. ICLR 2024 camera-ready PDF fetched; Table 1 Ours (SPRC) row and the val-set sentence read from it.

## 14. ACHIEVED VERIFICATION LEVEL

**LEVEL 2 of 6** — (1) registry/identity, (2) static source + protocol auditing are complete and
independently evidenced; **no** dependency installation, **no** checkpoint acquisition, **no** process
execution of the evaluator, **no** metric extraction from a real run, **no** numerical comparison to
the paper. Nothing in this report is RUNTIME_VERIFIED; §3's chain and §7's protocol are STATIC_AUDIT.
