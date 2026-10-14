s stated base commit. `git -C /home/heheboiz/data/cir status --porcelain` reports ` M workbench/backend/registry.py`, ` M workbench/scripts/download_auxiliary_assets.py`, `?? workbench/backend/paper_gate.py`, `?? workbench/tests/test_paper_gate.py`, `?? workbench/patches/csmcir_models_init_compat.patch` — but **not** `models.yaml`. The `conesep` block read above is therefore the committed content at `d9431fe9632b67a329771cc06567487b591afb4b`. All registry line numbers in this audit were re-derived by printing the file (`awk 'NR>=133 && NR<=195'`).

## PINNED SOURCE COMMIT

| Claim | Evidence | Verdict |
|---|---|---|
| Registry pin `bd4bf629ebe68045c1a54dcc61e4a3ea65f8759b` (`models.yaml:150`) equals the actual checkout | `git -C workbench/third_party/ConeSep rev-parse HEAD` → `bd4bf629ebe68045c1a54dcc61e4a3ea65f8759b` | **MATCH** |
| `upstream_repo_url: https://github.com/iLearn-Lab/CVPR26-ConeSep` (`:148`) matches the checkout's remote | `git -C workbench/third_party/ConeSep config --get remote.origin.url` → `https://github.com/iLearn-Lab/CVPR26-ConeSep.git` | **MATCH**. (`git remote get-url origin` prints the push rewrite `git@github.com:iLearn-Lab/CVPR26-ConeSep.git`; fetch URL agrees.) |
| `upstream_default_branch: main` (`:149`) | `git -C workbench/third_party/ConeSep branch -a --contains HEAD` → `* main`, `remotes/origin/HEAD -> origin/main` | **MATCH** |
| Checkout is not dirty in tracked files | `git -C workbench/third_party/ConeSep status --porcelain` → only untracked `__pycache__/` directories (18 entries + `lavis/**/__pycache__`), no modified tracked file | **CLEAN** (bytecode-only untracked noise; `runtime.is_bytecode`, `workbench/backend/runtime.py:1829-1838`, treats these as non-source) |
| History depth | `git -C workbench/third_party/ConeSep rev-list --count HEAD` → `1`; `git log -1 --format='%H %ci %s'` → `bd4bf629ebe68045c1a54dcc61e4a3ea65f8759b 2026-05-25 17:59:18 +0800 Update README.md` | Single-commit checkout. The tree content is resolved at the pin; **no upstream history is available to diff against**, so "this file was never different" claims cannot be made from the checkout alone. |

**Verdict: VERIFIED (pin).** The registry commit string, the remote, the branch and the working tree all agree. This is the strongest claim in the audit.

## CHECKPOINT IDENTITY

STATIC_AUDIT, registry + HTTP HEAD inspection only. **No checkpoint was downloaded; no payload was inspected; no weights were loaded.**

Registry declares three variants (`workbench/registry/models.yaml:164-194`):

| checkpoint_id | filename (`:line`) | training noise | eval noise | size | sha256 | status |
|---|---|---|---|---|---|---|
| `fiq_n02` | `ConeSep-FIQ-N0.2.pt` (`:166`) | 20 (`:169`) | 0 (`:170`) | `null` (`:171`) | `null` (`:172`) | `NO_CLEAN_CHECKPOINT` (`:173`) |
| `fiq_n05` | `ConeSep-FIQ-N0.5.pt` (`:176`) | 50 (`:179`) | 0 (`:180`) | `null` (`:181`) | `null` (`:182`) | `NO_CLEAN_CHECKPOINT` (`:183`) |
| `fiq_n08` | `ConeSep-FIQ_N0.8.pt` (`:186`) | 80 (`:189`) | 0 (`:190`) | `null` (`:191`) | `null` (`:192`) | `NO_CLEAN_CHECKPOINT` (`:193`) |

All three carry `checkpoint_mapping_status: VERIFIED_METADATA` (`:174`, `:184`, `:194`).

**Independent check against the author Hub** (`read https://huggingface.co/iLearn-Lab/CVPR26-ConeSep/tree/main` and `.../tree/main/fiq`):

- Root listing contains exactly `fiq/` (a directory), `.gitattributes`, `README.md`. The `fiq/` listing contains exactly three files: `ConeSep-FIQ-N0.2.pt`, `ConeSep-FIQ-N0.5.pt`, `ConeSep-FIQ_N0.8.pt` — uploaded in one commit `36139d8`, repo total 8.16 GB.
- `curl -s -o /dev/null -D - https://huggingface.co/iLearn-Lab/CVPR26-ConeSep/resolve/main/fiq/<file>` per file: all three answer `HTTP/2 302` then `HTTP/2 200`, with `x-linked-size: 2721428966` and distinct `x-linked-etag` values:
  - `ConeSep-FIQ-N0.2.pt` → `"5a864ee5a78c1d249320c4f997cc7d14b529b329b2efba5577c82dbe019ed9d2"`
  - `ConeSep-FIQ-N0.5.pt` → `"1953ec2f9269a5cb24d61bdf89d05ed8430fb3bdc6dd18cef08fe9e3dfbe1e26"`
  - `ConeSep-FIQ_N0.8.pt` → `"7c2bd0fbaea423a1a64809956b3fda72e191a6f7e5ed6ecfa708f1e3a8730eae"`
- The three URLs in the registry (`:167`, `:177`, `:187`) are the exact canonical resolve URLs for those files. Registry ↔ Hub file set **matches**, including the inconsistent `N0.8` underscore spelling which is faithfully copied.

**Gaps (severity P2):** `expected_size_bytes` and `expected_sha256` are `null` for all three variants even though the Hub publishes a size (`2721428966`) and a content hash (`x-linked-etag`, SHA-256 of the LFS/Xet blob). `scripts/download_checkpoints.py:110-114` prints `LOCAL SHA256 COMPUTED — official SHA256 unavailable` and `hash_valid = True` when `expected_sha256 is None`, and `:117` accepts `checkpoint_mapping_status == VERIFIED_METADATA` + a `download_url` as `provenance_valid`. So a download of any 2.72 GB blob from that URL would pass the gate with **no content binding**. This is not a blocker for the negative verdict, but it is a real provenance hole worth closing (the Hub exposes the value; recording it costs nothing).

**Upstream publishes no checkpoint list at all.** `grep -n -i "huggingface|hub|\.pt" third_party/ConeSep/README.md` returns only `README.md:123` (LAVIS clone URL), `README.md:238` ("best model weights ... saved in `best_model.pt`"), and `README.md:245` (`python src/cirr_test_submission.py checkpoints/cirr_noise0.5/`). **There is no checkpoint download section, no Hub link, and no clean (0%) FashionIQ checkpoint.** The three filenames exist only on the Hub, not in the pinned tree.

### Clean-checkpoint question (assignment focus)

- Upstream README states the evaluated noise ratios as `0.0`, `0.2`, `0.5`, `0.8` (`README.md:208`) and its FashionIQ results table contains a full `0%` block (`README.md:78`: "The **0% noise** setting in the tables below is equivalent to the **traditional fully-supervised CIR** paradigm").
- The published Hugging Face file set contains **only** the 20/50/80 variants — confirmed by the HF directory listing above. **No clean FashionIQ checkpoint is published.**
- Therefore `strict_clean_checkpoint_available: false` (`workbench/registry/models.yaml:162`) is **correct**, and `notes: Official Hub lists only FashionIQ 20%, 50%, and 80% noise-trained weights.` (`:163`) is **accurate**.
- Independently executed: `PYTHONPATH=. python3 -c "... select_checkpoints(models['conesep'], policy=...)"` →
  - `strict_clean` → `BLOCKED BLOCKED_CHECKPOINT False ()` reason `no checkpoint is recorded as clean-trained under policy strict_clean: fiq_n02=20%, fiq_n05=50%, fiq_n08=80%.`
  - `include_flagged_conditions` → `BLOCKED BLOCKED_CHECKPOINT_SELECTION False ()` reason `more than one source-backed variant is registered and none is declared a bundle: fiq_n02, fiq_n05, fiq_n08.`
- **Consequence (P1, scientific):** the paper row the registry pins — R@10 55.58 / R@50 75.88 / mean 65.73 — is the **0% noise (clean) row** of Table 1 (verified below). Evaluating any registered checkpoint would produce a **20/50/80%-noise-trained** model's clean-evaluation score, which is the row `54.93 / 75.01 / 64.97` (20%), `53.39 / 74.15 / 63.77` (50%), or `49.25 / 70.30 / 59.77` (80%). Comparing any of those to 65.73 would be a **wrong-row comparison**. `workbench/backend/paper_gate.py:34-43` lists `checkpoint_training_noise_pct` among `COMPARABILITY_FIELDS`, so the gate is designed to refuse this — but the registry row itself carries no warning beyond `:163`, and the three variants all say `evaluation_noise_pct: 0` (`:170,180,190`), which is true of the *evaluation* but is easily read as "this is a clean run". Explicit labelling of the target paper row as `0% noise / training condition clean` is still missing from the variant records.

## EVALUATION PROTOCOL

Registry: `native_protocol: fashioniq_full_gallery_ref_excluded` (`workbench/registry/models.yaml:156`), `supported_protocols: [fashioniq_full_gallery_ref_excluded]` (`:157`), `literature_split_label: original` (`:155`). Protocol definition: `workbench/registry/models.yaml:12-15` (`label: FashionIQ - Original Full Gallery, Reference Excluded`, `literature_split_label: original`, `evaluation_noise_pct: 0`).

**Independently checked against the pinned source. All five sub-claims hold.**

1. **Exact evaluator.** `third_party/ConeSep/test.py:9` `def test(params, model, testset, category, txt_processors)`; categories `dress|shirt|toptee|shoes` selected at `test.py:12-18`. There is **no** `if __name__ == '__main__'` block and no `ArgumentParser` in `test.py`; the only callsite is `third_party/ConeSep/train.py:216` `t = test.test(args, model, trainset, category, txt_processors)`, inside `train_and_evaluate()`. `grep -n "__main__" third_party/ConeSep/test.py` → no match. **No standalone FashionIQ evaluation CLI exists.**
2. **Gallery construction.** `third_party/ConeSep/datasets.py:28` `self.split = 'original-split'` (a hardcoded default; `FashionIQ.__init__` signature at `:25` takes only `path, transform, noise_ratio` — no split switch). `get_test_data()` (`:164`) reads `image_splits/split.{name}.val.json` (`:166`) into `images`, then takes the `elif self.split == 'original-split':` branch (`:201`), which appends **every** index of `images` in file order:
   ```python
   elif self.split == 'original-split':
       for id, image_name in enumerate(images):
           test_targets_id.append(id)
           ...
   ```
   → full ordered validation split. The `val-split` pair-union branch (`:191-200`) is dead code for this model. Registry claim **confirmed**.
3. **Reference exclusion.** `test.py:63-68`:
   ```python
   test_targets_id = []
   for i in test_targets:
       test_targets_id.append(i['target_img_id'])
   for i, t in enumerate(test_queries):
       sims[i, test_targets_id.index(t['source_img_id'])] = -10e10
   ```
   The reference score is **masked to `-10e10` before the sort** — not removed. Registry `docs/FASHIONIQ_PROTOCOL_PARITY.md:71` says "masked before ranking"; `docs/audits/PHASE2_SOURCES_D.md` and `PHASE2_CROSS_REVIEW.md:73` agree. Registry claim **confirmed**; the workbench doc also correctly distinguishes mask vs. column-removal (`docs/FASHIONIQ_PROTOCOL_PARITY.md:27-33`).
4. **Recall@K maths.** `test.py:71/:75` `nn_result = [np.argsort(-sims[i, :])[:50] for i in range(sims.shape[0])]`; `:76` `for k in [1, 10, 50]:`; `:79` `if test_targets_id.index(test_queries[i]['target_img_id']) in nns[:k]: r += 1`; `:81` `r = 100 * r / len(nn_result)`. Hit fraction × 100, denominator = number of queries. **Confirmed.** Note `R@1` is computed in addition to R@10/R@50; the paper/registry report only R@10 and R@50.
5. **Sort direction / tie-breaking.** `np.argsort(-sims[i, :])` = descending by similarity. Ties are broken by numpy's default quicksort (introsort) on the negated array — **unstable and unspecified by the source**; the pinned code makes no tie policy explicit, and neither does the workbench (`docs/FASHIONIQ_PROTOCOL_PARITY.md:35-36`: "Ties remain an ambiguity of the ranking itself, which the workbench does not claim to pin"). Practically inert here: query/target features are `F.normalize(..., dim=-1)` outputs of a continuous model (`third_party/ConeSep/lavis/models/blip2_models/ConeSep.py:488-490`, `:514`), so exact float ties are measure-zero. **UNKNOWN-by-design, not a defect.**

### Structural finding in the pinned evaluator (P3, does not affect the maths)

`test.py:59-61`:
```python
sims = np.matmul(all_queries, all_imgs) #all_queries.dot(all_imgs.T)
sims = sims.squeeze()
sims = sims.max(-1)
```
`extract_retrieval_compose` returns shape `(N, 1, 1, d)` (`ConeSep.py:492` `return fusion_feats.unsqueeze(1).unsqueeze(1)` with `fusion_feats` of dim `d = embed_dim`), and `extract_retrieval_target` returns `(M, d, 32)` (`ConeSep.py:515` `return image_features.permute(0, 2, 1)` where `image_features` is `(M, 32, d)`). Matmul gives `(N, M, 1, 32)`, `.squeeze()` gives `(N, M, 1, 32)`, and `.max(-1)` gives `(N, M, 1)`.

`sims` is therefore **4-D, not 2-D**, when it reaches the masking line. Consequences:
- `sims[i, test_targets_id.index(t['source_img_id'])] = -10e10` — with `sims` of shape `(N, M, 1)`, `sims[j, k]` is the length-1 array `sims[j, k, :]`, so **numpy writes through the view and the mask still lands on `sims[j, k, 0]`**. Masking is not silently dropped. This is why the upstream code works despite the odd shape.
- `np.argsort(-sims[i, :])` then yields shape `(M, 1)` instead of `(M,)`. `nns[:k]` (e.g. `[:10]`) gives `(10, 1)`, and `test_targets_id.index(...) in nns[:k]` tests an integer shape-`(10,1)` membership — numpy broadcasts the scalar against the `(10,1)` array, so the hit test still reduces to "is the target index among the top-k", consistent with the `(M,)` case. **The observed numbers are not explained by a broadcast accident.**
- The comment `all_queries.dot(all_imgs.T)` on `:59` is stale and does not describe the code.

I could only verify the shape algebra by direct evaluation of numpy semantics in `/tmp` (`np.matmul` on `(5,1,1,256)×(7,256,32)` → `(5,7,1,32)`; `.max(-1)` → `(5,7,1)`); I did **not** run the model. So: the shape claim is STATIC_AUDIT-verified, the "numbers still come out right" claim is argued from numpy semantics and is **NOT** RUNTIME_VERIFIED.

## DATASET CONTRACT

Registry: `fashioniq_layout: fashioniq_ilearn_resized` (`models.yaml:153`), `fashioniq_image_extensions: [".jpg"]` (`:154`), `preparation_contract: manual_ilearn_resized` (`:145`).

Source paths (all relative to a root that **must end with a separator**, because `__init__` concatenates): `datasets.py:31-33`:
```python
self.image_dir = self.path + 'resized_image'
self.split_dir = self.path + 'image_splits'
self.caption_dir = self.path + 'captions'
```

Required inputs, by callsite:

| Input | Source callsite | Registry layout checker |
|---|---|---|
| `image_splits/split.{dress,shirt,toptee}.val.json` | `datasets.py:166` | `fashioniq_layout.py:33` |
| `captions/cap.{cat}.val.json` | `datasets.py:168` | `fashioniq_layout.py:31` |
| `captions/correction_dict_{cat}.json` | `datasets.py:170` | `fashioniq_layout.py:32` |
| `captions/cap.{cat}.train.json` | `datasets.py:94` (only on cache miss) | present in `ilearn_resized_training_paths` (`:37-41`), **absent** from `ilearn_resized_paths` |
| `resized_image/{cat}/{id}.jpg` | `datasets.py:137` (`get_img`), `:147` (`get_test_img`) | `fashioniq_layout.py:29-30` + extension check `:79-89,92-…` |
| `fashion_iq_data.json`, `test_{queries,targets}_{cat}.pkl` | `datasets.py:35,38,44-58` | **not checked** |

`fashioniq_layout.py:25-34` (`ilearn_resized_paths`) covers captions dir, splits dir, `resized_image/`, the three category dirs, the three val caption files, the three correction dictionaries and the three val split files. `_LAYOUT_IMAGE_DIRECTORY["fashioniq_ilearn_resized"] = ("resized_image", True)` (`fashioniq_layout.py:85`) matches the source's per-category `.jpg` path. **Layout and extension claims confirmed against source.**

**Two real gaps:**
1. **Cache branch has no freshness validation.** `datasets.py:35` gates the whole preparation on `if not os.path.exists(os.path.join(self.path, 'fashion_iq_data.json'))`. If that file exists, `:52-58` loads `fashion_iq_data.json` and the six PKLs and **never re-reads the split/caption/correction files**. All the layout preflight checks would pass against freshly-edited JSONs while the evaluator silently used stale pickles. **P2.** (`docs/audits/PHASE2_DATA_CONTRACT.md:65`, `PHASE2_SOURCES_D.md:55` flag this; I confirm it from `datasets.py:35-58`.)
2. **The correction dictionaries have no provenance.** `datasets.py:170` (val) and `:96` (train init) open `captions/correction_dict_{cat}.json`; earlier greps confirm the pinned tree neither tracks nor generates them, and `workbench/registry/auxiliary_assets.yaml` has **no `conesep` row** (`grep -c conesep registry/auxiliary_assets.yaml` → `0`; the file's 14 assets cover only `clip4cir_rn50x4_fullft`, `clvc_net`, `combiner_rn50x4_noft`, `csmcir`, `sprc`). They are a content-transforming input (see below), so an operator supplying a differently-derived dictionary gets a different caption string and a different Recall. **P2.**

**Local readiness (observed):** `PYTHONPATH=. python3 -c "from workbench.backend.preflight import model_requirements; ..."` returns **8 blocks** for `conesep`:
```
FashionIQ fashioniq_ilearn_resized requirement missing: /home/heheboiz/data/cir/data/fashionIQ_dataset/resized_image
FashionIQ fashioniq_ilearn_resized requirement missing: .../resized_image/dress
FashionIQ fashioniq_ilearn_resized requirement missing: .../resized_image/shirt
FashionIQ fashioniq_ilearn_resized requirement missing: .../resized_image/toptee
FashionIQ fashioniq_ilearn_resized requirement missing: .../captions/correction_dict_dress.json
FashionIQ fashioniq_ilearn_resized requirement missing: .../captions/correction_dict_shirt.json
FashionIQ fashioniq_ilearn_resized requirement missing: .../captions/correction_dict_toptee.json
FashionIQ image extension requirement: image directory missing: .../resized_image
```
`ls /home/heheboiz/data/cir/data/fashionIQ_dataset` → `captions  image_splits  images  qwen_captions` — the standard (un-resized) layout, **not** the iLearn resized layout. `data/` also holds a separate `FashionIQ/` tree. So the registered layout is genuinely absent locally; `docs/READINESS_MATRIX.md:27` recorded the same blocker.

## PREPROCESSING CONTRACT

Two independent text stages plus one image stage. Registry's `docs/SOURCE_TO_WORKBENCH_PARITY_MATRIX.md:28` records caption stage `EQUIVALENT_WITH_EVIDENCE` and image stage `EQUIVALENT_WITH_EVIDENCE` for `conesep`; `docs/audits/PHASE2_DATA_CONTRACT.md:25` records the sources. My reading:

**Stage 1 — in-dataset, dictionary-corrected caption composition** (`datasets.py:108-117`):
```python
def correct_text(self, text, correction_dict):
    trans=str.maketrans({key: ' ' for key in string.punctuation})
    tokens = str(text).lower().translate(trans).strip().split()
    text = " ".join([correction_dict.get(word) if word in correction_dict else word for word in tokens])
    return text

def concat_text(self, captions, correction_dict):
    text = "{} and {}".format(self.correct_text(captions[0], correction_dict), self.correct_text(captions[1], correction_dict))
```
Applied to validation captions at `datasets.py:175`. So: lowercase → **all** ASCII `string.punctuation` → spaces → whitespace tokenize → exact per-token dictionary substitution → rejoin with spaces → `cap1 + " and " + cap2` in source order. **Confirmed.**

[…113ln elided…]

**CF-4 (P2) — Dataset contract has an unchecked, unversioned cache and unprovenanced correction dictionaries.** `datasets.py:35-58` (staleness) and `datasets.py:96,170` + absent registry rows (dictionary provenance). Both change captions/gallery without failing any preflight check.

## MINOR FINDINGS

**MF-1 (P2) — No size/sha binding on the three checkpoints.** `models.yaml:171-172,181-182,191-192` are `null`; the Hub publishes `x-linked-size: 2721428966` and per-file `x-linked-etag` (recorded above). `scripts/download_checkpoints.py:110-114,117` treats an absent expected hash as pass. Cheap to fix, and it is the difference between "downloaded a file" and "downloaded *the* file".
**MF-2 (P2) — Hardcoded `/root/bert-base-uncased`** at `blip2.py:32,48,55`, with no registry record. Not portable; no fallback in source.
**MF-3 (P2) — Registry records no preprocessing or auxiliary assets for `conesep`.** `preparation_contracts.yaml` and `auxiliary_assets.yaml` both contain zero `conesep` rows (counted). LAVIS `blip2_pretrained.pth` (`blip2_pretrain.yaml:10`) and the correction dictionaries are runtime prerequisites that the registry does not name. The richer per-asset audit exists at `workbench/registry/audits/cluster_b_ilearn.yaml:281-274`, but `auxiliary_assets_for_model` (`backend/registry.py:334-335`) reads only `auxiliary_assets.yaml`, so that audit content **does not reach preflight**.
**MF-4 (P3) — Two tolerance regimes.** `metrics_extraction._check_parity` uses absolute 0.05 (`:32`); `paper_gate.TOLERANCE_PCT` uses relative 5.0 (`:11`). Same three metrics, different acceptance rules.
**MF-5 (P3) — Stale comment.** `test.py:59` `#all_queries.dot(all_imgs.T)` does not describe the code (the arrays are 4-D/3-D; a `.T` transpose would be wrong).
**MF-6 (P3) — Dead `val-split` branch.** `datasets.py:191-200` is unreachable because `self.split` is hardcoded `'original-split'` at `:28` and no constructor argument overrides it. Harmless, but it means `fashioniq_val_split` is not a supported protocol for this model even though the code contains the branch.
**MF-7 (P3) — No upstream history.** Single-commit checkout (`rev-list --count HEAD` → 1). The pin is faithful to the registry string, but the audit cannot confirm the pin is the current upstream `main` (that would require a fetch, which the read-only rule forbids). Marked **UNKNOWN** rather than PASS.
**MF-8 (P3) — Empty `dataset_root` string in dataset builders.** `get_img`/`get_test_img` require a trailing separator (`self.path + 'resized_image'`), which the README's `/path/to/FashionIQ/` (`README.md:215`) supplies. A path without the trailing slash silently reads a sibling path. Not a defect of the registered model, but a footgun the layout checker does not test for (it uses `Path` joins, `fashioniq_layout.py:25-34`).

## UNRESOLVED EVIDENCE

1. **Checkpoint payload identity.** No weight file was downloaded or opened. `checkpoint_training_noise_pct` values (20/50/80) are **Hub filename metadata plus registry assertion**, not a verified property of the bytes. The Hub exposes only sizes and per-file hashes — nothing that states a training condition. **UNKNOWN.**
2. **Checkpoint loadability.** Whether `torch.load("ConeSep-FIQ-N*.pt")` yields an object whose architecture matches `lavis/models/blip2_models/ConeSep.py:ConeSep` is **UNKNOWN**. The Hub's pickle-import list for all three files includes `lavis.models.blip2_models.ConeSep.ConeSep`, `lavis.models.eva_vit.VisionTransformer`, `lavis.models.blip2_models.Qformer.BertLMHeadModel` and `transformers.models.bert.tokenization_bert.BertTokenizer` — so the files are **whole-model pickles**, not `state_dict` dicts (consistent with `train.py:251` `torch.save(model, ...)`). That means the loader would have to be `torch.load(..., weights_only=False)` on a pickle, and would need the `lavis` package importable at unpickle time; it also means `utils.load_checkpoint`'s `checkpoint['state_dict']` would **fail** on these files. This raises checkpoint-handling risk but was **not executed** — STATIC_AUDIT only.
3. **Whether the real `-10e10` mask and the workbench column-removal produce identical per-query rankings for the actual ConeSep score distribution.** Argued (synthetic probe, unit-norm features), not demonstrated on real outputs. **UNKNOWN.**
4. **Whether the upstream pin is current `main`.** No fetch performed (read-only). **UNKNOWN.**
5. **Whether the iLearn correction dictionaries' contents match the ones used for the paper table.** No provenance anywhere. **UNKNOWN.**
6. **Whether the `resized_image` staging recipe (which resizer, what JPEG quality) matches what produced the published numbers.** No source script exists. **UNKNOWN.**
7. **The 55-item checklist's canonical question text.** The assignment file specifies "all 55 questions" and a strict `N. <question short form> — verdict — evidence — severity — action` format, but the list itself is not present in `/tmp/audit_assignments.md`, `/tmp/audit_context.md`, or anywhere under `/home/heheboiz/data/cir` (searched `*.md` for `55 question|55-item|CHECKLIST|checklist`). The 55 probes below are therefore **my own**: they cover every artefact the assignment enumerates (registry, source, dataset, preprocessing, gallery, metric, parity, runtime, checkpoint, verdict), one probe per fact. Recorded here so the substitution is explicit rather than silent.

---

## FINAL VERDICT

**BLOCKED.**

The registry's static claims about `conesep` are, where I could check them, **correct**: the pin matches the checkout, the protocol `fashioniq_full_gallery_ref_excluded` is exactly what `datasets.py:28,201-206` + `test.py:63-82` implement, `strict_clean_checkpoint_available: false` is right because the author Hub publishes only 20/50/80% weights, and 55.58 / 75.88 / 65.73 reproduce the paper's 0%-noise row and are internally consistent with `mean=(r10+r50)/2`. But the model cannot be reproduced: the pinned tree has no standalone evaluator and no checkpoint loader, the workbench adapter raises `NotImplementedError`, the registered dataset layout is absent locally, and the published row's row-to-checkpoint mapping is unlabelled. No runtime claim was made or observed.

---

## EXECUTION TRACE

```
paper/README/config
  -> source dataset
  -> preprocessing
  -> official evaluator
  -> workbench invocation
  -> checkpoint
  -> output
  -> metric comparison
```

**1. paper/README/config → nothing downstream is pinned by config.**
- Paper: `https://arxiv.org/abs/2604.20358` (HTTP 200; title/authors match `README.md:8-13`); Table 1 text from `https://arxiv.org/html/2604.20358v1`.
- README: `third_party/ConeSep/README.md:78` (0% = fully-supervised), `:83` (Table 1 caption), `:150-181` (FashionIQ directory layout), `:208` (noise ratios `0.0/0.2/0.5/0.8`), `:213-221` (train command), `:238` (`best_model.pt`), `:245` (CIRR submission).
- Config: `lavis/configs/models/blip2/blip2_pretrain.yaml:10` → `pretrained: https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained.pth`; `image_size: 224`, `num_query_token: 32`. **Not registered as an auxiliary asset for `conesep`** (`auxiliary_assets.yaml` has 0 conesep rows).
- ⇒ *Arrow status:* paper ✓ read; README ✓ read; config ✓ read; **registry pin of config/backbone weights ✗ MISSING.**

**2. source dataset → present, but wrong layout.**
- Required: `resized_image/{dress,shirt,toptee}/*.jpg`, `image_splits/split.{cat}.val.json`, `captions/cap.{cat}.val.json`, `captions/correction_dict_{cat}.json` (`datasets.py:31-33,137,147,166,168,170`).
- Present: `/home/heheboiz/data/cir/data/fashionIQ_dataset/{captions,image_splits,images,qwen_captions}` — no `resized_image/`. Split/caption JSONs do exist and were counted (3817/6346/5373 images, unique, all query ids inside the gallery).
- ⇒ *Arrow status:* **BLOCKED** (8 preflight blocks, executed).

**3. preprocessing → confirmed in source, not exercised.**
- `train.py:85-100` targetpad 1.25 @224; `data_utils.py:92-106`; `datasets.py:108-117` caption correction+composition; `test.py:35` LAVIS eval processor; `blip_processors.py:49-59`; `blip2.py:32` tokenizer path; `ConeSep.py:469-475,128` `max_length=32`.
- ⇒ *Arrow status:* **STATIC_AUDIT VERIFIED**; no image was ever transformed.

**4. official evaluator → exists as a function; not reachable standalone.**
- `test.py:9-84`; invoked only from `train.py:216` inside `train_and_evaluate()`; no `__main__`.
- ⇒ *Arrow status:* **NOT RUNNABLE as an evaluator** (CF-1).

**5. workbench invocation → does not exist.**
- `workbench/backend/adapters/models.py:59-61` → inherited `command()` raises. **Executed:** `NotImplementedError: upstream evaluation command requires audit before execution`.
- `workbench/replay/` has no `conesep` module (`ls` → `common.py, limn.py, tgcir.py`). `grep -rn conesep workbench/replay/` → empty.
- `src/evaluation/fashioniq.py` full-gallery machinery has **test-only** callsites.
- ⇒ *Arrow status:* **BLOCKED.**

**6. checkpoint → identity partially established, bytes never touched.**
- Registry `models.yaml:165-194`; Hub listing (3 files exactly, `fiq/` only); `x-linked-size 2721428966` and three distinct `x-linked-etag` values for all three URLs. No download. `artifacts/checkpoints/` holds only `.gitkeep`.
- Training-condition labels are filename/registry metadata, not payload-derived (UNRESOLVED #1).
- ⇒ *Arrow status:* **PARTIAL — URL/identity verified; bytes, loadability, and training condition UNKNOWN.**

**7. output → none.**
- No evaluation was run; there is no stdout, no `metrics_best.json`, no workbench result record for `conesep` (`reproduction_status: NOT_RUN`, `models.yaml:161`).
- ⇒ *Arrow status:* **NOT_RUN.**

**8. metric comparison → only against the paper, not against a run.**
- Registry `:158-160` vs README image `assets/results-fiq.png` (`0%` row) vs arXiv Table 1 → all `55.58 / 75.88 / 65.73`, **identical**.
- `(55.58+75.88)/2 = 65.73`; `(50.96+56.98+58.80)/3 = 55.58`; `(73.02+75.22+79.40)/3 = 75.88` (executed).
- ⇒ *Arrow status:* **transcription VERIFIED; reproduction PARITY_NOT_EVALUATED.**

---

## CHECKLIST

*Provenance note: the assignment's canonical 55-question list is not present in any file I can read (see UNRESOLVED #7). The 55 probes below are mine, stated as short forms.*

1. **Registered `model_id` present and unique** — VERIFIED — `workbench/registry/models.yaml:133`; `tests/test_checkpoint_selection.py:127-133` iterates `conesep` by id — NONE — none.
2. **`method_name`/`paper_title` match the upstream README title** — VERIFIED — `models.yaml:142,146` vs `third_party/ConeSep/README.md:5` — NONE — none.
3. **`paper_url` resolves and identifies the same work** — VERIFIED — `read https://arxiv.org/abs/2604.20358` HTTP 200, same title/authors — NONE — none.
4. **Registry `upstream_commit_sha` equals the checkout HEAD** — VERIFIED — `models.yaml:150` = `git -C workbench/third_party/ConeSep rev-parse HEAD` — NONE — none.
5. **Checkout remote equals `upstream_repo_url`** — VERIFIED — `git config --get remote.origin.url` → `https://github.com/iLearn-Lab/CVPR26-ConeSep.git` — NONE — none.
6. **`upstream_default_branch` matches** — VERIFIED — `git branch -a --contains HEAD` → `* main` — NONE — none.
7. **Checkout has no modified tracked files** — VERIFIED — `git status --porcelain` → `__pycache__` only — P3 — keep `.gitignore`d; bytecode already excluded by `runtime.is_bytecode`.
8. **Pin equals current upstream `main`** — UNKNOWN — single-commit checkout; no fetch permitted — P3 — re-verify at run time with a fetch on a writable host.
9. **`source_available`/`source_dir` correct** — VERIFIED — `models.yaml:151-152`; tree present under `workbench/third_party/` — NONE — none.
10. **Declared `native_protocol` matches the source evaluator's gallery** — VERIFIED — `datasets.py:28,201-206` full ordered split; `docs/FASHIONIQ_PROTOCOL_PARITY.md:71` — NONE — none.
11. **`supported_protocols` is exactly the native one** — VERIFIED — `models.yaml:157`; dead `val-split` branch `datasets.py:191-200` is unreachable — NONE — none.
12. **`literature_split_label: original` consistent with the gallery** — VERIFIED — `models.yaml:155`; protocol table `:12-15`; PAPER says "standard metrics" for FashionIQ — NONE — none.
13. **Reference excluded before ranking** — VERIFIED — `test.py:68` mask precedes `test.py:71/:75` `argsort` — NONE — none.
14. **Reference mask value and mechanism recorded (mask, not removal)** — VERIFIED — `test.py:68` `-10e10`; registry doc `FASHIONIQ_PROTOCOL_PARITY.md:71` — P3 — keep the mask-vs-removal distinction visible.
15. **Sort is descending by similarity** — VERIFIED — `test.py:71,75` `np.argsort(-sims[i, :])` — NONE — none.
16. **Tie-breaking policy pinned** — UNKNOWN — numpy default quicksort, unspecified in source and workbench (`FASHIONIQ_PROTOCOL_PARITY.md:35-36`) — P3 — accept; measure-zero for normalized float features.
17. **Recall@K maths correct (`100*hits/N`)** — VERIFIED — `test.py:79-81` — NONE — none.
18. **Denominator is the query count** — VERIFIED — `test.py:81` `len(nn_result)` = `len(test_queries)` — NONE — none.
19. **R@1 additionally computed (superset of the paper rows)** — VERIFIED — `test.py:76` `for k in [1, 10, 50]` — P3 — harmless; document it.
20. **Macro/mean definition matches the paper (`(avg R@10 + avg R@50)/2`)** — VERIFIED — executed `(55.58+75.88)/2 = 65.73`; paper `AVG. 65.73` — NONE — none.
21. **`reported_r10` matches the paper** — VERIFIED — `models.yaml:158`; README image + arXiv Table 1 both `55.58` — NONE — none.
22. **`reported_r50` matches the paper** — VERIFIED — `models.yaml:159`; both sources `75.88` — NONE — none.
23. **`reported_mean` matches the paper** — VERIFIED — `models.yaml:160`; both sources `65.73` — NONE — none.
24. **`reported_mean` equals `(reported_r10 + reported_r50)/2`** — VERIFIED — executed; `65.73` both ways — NONE — none.
25. **Category cells are consistent with the average row** — VERIFIED — executed `50.96/56.98/58.80 → 55.58`; `73.02/75.22/79.40 → 75.88` — NONE — none.
26. **Reported row's noise condition is labelled** — VARIANT — registry does not state the row is 0% noise; the three variants say `evaluation_noise_pct: 0` (`:170,180,190`) — P1 — add `paper_row_noise_pct: 0` (or equivalent) to the entry.
27. **Gallery equals the full val split file order** — VERIFIED — `datasets.py:203` `for id, image_name in enumerate(images)` — NONE — none.
28. **Gallery sizes match the real FashionIQ val splits** — VERIFIED — counted 3817/6346/5373, unique — NONE — none.
29. **Every reference/target lies inside its gallery** — VERIFIED — counted `ids subset of split? True` for all three — NONE — none.
30. **Layout id `fashioniq_ilearn_resized` matches source paths** — VERIFIED — `fashioniq_layout.py:25-34` vs `datasets.py:31-33,166,168,170` — NONE — none.
31. **Image extension `.jpg` matches source** — VERIFIED — `datasets.py:137,147` `+ ".jpg"`; `fashioniq_layout.py:85` — NONE — none.
32. **Layout checker covers the eval-required files** — VERIFIED — `fashioniq_layout.py:31-33` covers val captions, dictionaries, val splits; `:29-30` covers image dirs — NONE — none.
33. **Layout checker covers *all* files the evaluator opens** — VARIANT — cache JSON + 6 PKLs (`datasets.py:35,52-58`) unchecked; train captions only in the `_training` variant (`fashioniq_layout.py:37-41`) — P2 — add the cache branch or invalidate-on-mismatch.
34. **Cache staleness guarded** — NOT_RUN (absent) — `datasets.py:35` loads PKLs without re-reading sources — P2 — require a provenance stamp inside `fashion_iq_data.json`.
35. **Correction dictionaries present and compared** — VARIANT — `datasets.py:96,170`; `auxiliary_assets.yaml` has 0 conesep rows — P2 — register the three dictionaries with a digest.
36. **Caption normalization matches source (lowercase, all punctuation, token map)** — VERIFIED — `datasets.py:108-113` — NONE — none.
37. **Caption composition order and `" and "` joiner** — VERIFIED — `datasets.py:116` — NONE — none.
38. **LAVIS eval caption processor applied to the composed string** — VERIFIED — `test.py:35`; `blip_processors.py:28-59` — NONE — none.
39. **Tokenizer path pinned in the registry** — VARIANT — source hardcodes `/root/bert-base-uncased` (`blip2.py:32,48,55`); no registry record — P2 — record it as an auxiliary asset or patch to an env var.
40. **`max_txt_len` recorded** — VERIFIED (source) / UNKNOWN (registry) — `ConeSep.py:128` default 32, `from_config :556`; LAVIS config does not set it — P3 — note it in the prep contract.
41. **Image transform recipe matches source** — VERIFIED — `train.py:85-100`; `data_utils.py:92-106` — NONE — none.
42. **Staged resized images have a pinned generation recipe** — VARIANT — no resize script in the tree; `auxiliary_assets.yaml`/`preparation_contracts.yaml` have no conesep row — P2 — pin a resizer + JPEG quality, or hash the staged tree.
43. **Noise injection touches only training data** — VERIFIED — `datasets.py:62-89`; called only at `train.py:109,117`; `grep shuffle test.py` empty — NONE — none.
44. **Evaluator runs in eval mode / no grad** — VERIFIED — `test.py:10` `model.eval()`; `:20` `torch.no_grad()` — NONE — none.
45. **Checkpoint variants declared with noise provenance** — VERIFIED — `models.yaml:165-194` — NONE — none.
46. **Checkpoint URLs resolve** — VERIFIED (STATIC) — three `curl` HEADs → `302` then `200` — NONE — none.
47. **Checkpoint sizes recorded** — VARIANT — all `null` (`:171,181,191`) though the Hub reports `2721428966` — P2 — record them.
48. **Checkpoint sha256 recorded** — VARIANT — all `null` (`:172,182,192`) though the Hub exposes per-file `x-linked-etag` — P2 — record them.
49. **Checkpoint file set matches the author Hub** — VERIFIED — HF listing = exactly the three registered filenames — NONE — none.
50. **A clean (0% noise) checkpoint is published** — VERIFIED-NEGATIVE — HF `fiq/` holds only 20/50/80; README has no checkpoint section — NONE (registry is correct) — none.
51. **`strict_clean_checkpoint_available` is accurate** — VERIFIED — `models.yaml:162` false; executed `select_checkpoints` → BLOCKED, `strict_clean_checkpoint_available False` — NONE — none.
52. **Checkpoint loadable by pinned code** — UNKNOWN — no loader exists (`--restore_file` never read, `utils.load_checkpoint` uncalled, files are whole-model pickles per Hub import list) — P1 — see CF-1; requires a run to settle.
53. **Standalone official evaluator exists** — VERIFIED-NEGATIVE — `test.py` has no `__main__`; only `train.py:216` — P1 — write and audit an eval entrypoint.
54. **Workbench adapter produces a command** — VERIFIED-NEGATIVE — `adapters/models.py:59-61`; executed `NotImplementedError` — P1 — implement `ConeSepAdapter.command` after CF-1 is resolved.
55. **Workbench evaluator reaches the model's protocol / run output exists and parity was compared** — BLOCKED — full-gallery helpers are test-only (`src/evaluation/fashioniq.py:35-77,159-206`; callsites only in `tests/unit/`); `reproduction_status: NOT_RUN` (`models.yaml:161`); no result record — P1 — no `conesep` number can be produced today.

---

**FINAL VERDICT: BLOCKED** — the registry's static facts for `conesep` check out against the pinned tree (pin, protocol, `strict_clean=false`, paper-metric transcription), but there is no runnable path to any number: the pinned tree has no standalone evaluator or checkpoint loader, the workbench adapter raises `NotImplementedError`, the registered dataset layout is absent locally, and the published row is the 0%-noise row for which no checkpoint exists.


[Showing lines 1-174 and 288-439 of 439; 113 middle lines (17.5KB) elided. Read artifact://11511 for full output]