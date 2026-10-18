# limn — reproducibility reassessment

Independent, adversarial, read-only re-audit of the CIR Failure Analysis Workbench's LIMN
reproduction path against the pinned upstream checkout. Every prior workbench document was treated
as a hypothesis. Nothing here was copied from one; every claim is re-derived from the file and line
cited, or from a command whose output is printed. No GPU inference was executed and none is claimed.

**Snapshot 1 (decides what must be reproduced)** = `workbench/third_party/LIMN` at
`7d7bc9b116f594a65ac22457491edf28a88d3c3e`, the registry's `upstream_commit_sha`
(`workbench/registry/models.yaml:916`). Verified:

```
$ git -C workbench/third_party/LIMN rev-parse HEAD
7d7bc9b116f594a65ac22457491edf28a88d3c3e
```

**Snapshot 2 (current upstream)** = same commit; see §2.

---

## 1. MODEL IDENTITY

| field | value | source |
|---|---|---|
| method | LIMN (base, self-training iteration 0) | `workbench/registry/models.yaml:909,958` |
| paper | *Self-Training Boosted Multi-Factor Matching Network for Composed Image Retrieval*, IEEE TPAMI 2024, DOI 10.1109/TPAMI.2023.3346434 | `models.yaml:911,912`; `third_party/LIMN/README.md:1,155-164` |
| year | 2024, `research_generation: legacy` | `models.yaml:910,911` |
| source URL | https://github.com/iLearn-Lab/TPAMI24-LIMN | `models.yaml:914` |
| pinned commit | `7d7bc9b116f594a65ac22457491edf28a88d3c3e` | `models.yaml:916`, verified above |
| checkpoint variant | `base_iter0_all_categories` bundle = `0_{dress,shirt,toptee}_best_model.pt` | `models.yaml:960-992`; `third_party/LIMN/README.md:110` |
| reported metrics | r10 55.91, r50 77.82, mean 66.87 (WORKBENCH_DERIVED mean) | `models.yaml:930-932` |
| native protocol | `fashioniq_val_split` (Protocol C: ordered pair-union gallery, reference excluded) | `models.yaml:924` |
| serialization | `whole_model` | `models.yaml:907` |

`native_protocol` is authoritative and is quoted verbatim: `native_protocol: fashioniq_val_split`
with `supported_protocols: [fashioniq_val_split]` (`models.yaml:924-925`); decisive
`literature_split_label: val` (`models.yaml:923`). Verified independently that the pinned source
implements exactly that protocol (Protocol C) and not Protocol A/B — §7.

The paper row is separately recorded and independently verified from the PDF:

```
$ sed -n '1589-1660p' workbench/registry/paper_benchmarks/papers.yaml
  reference_id: limn_fashioniq_base_iter0_table1
  checkpoint_id: base_iter0_all_categories
  gallery_protocol: fashioniq_val_split
  reference_excluded: true
  reported_metrics: {r10: 55.91, r50: 77.82}
  category_metrics: dress 50.72/74.52, shirt 56.08/77.09, toptee 60.94/81.85
  review_status: CONFIRMED
```

Arithmetic re-checked by hand: (50.72+56.08+60.94)/3 = 55.9133 → 55.91; (74.52+77.09+81.85)/3 =
77.82; (55.91+77.82)/2 = 66.865 → 66.87 with the declared round-half-up policy
(`papers.yaml:1620-1627`). All three reproduce. **STATIC_AUDIT.**

---

## CHECKLIST ANSWERS (the six assignment questions)

| # | question | verdict | severity |
|---|---|---|---|
| 1 | observer ranking derivation == native computation (candidates, masking, caption preprocessing, similarity, argsort)? | **VERIFIED** — every step quoted and matched, and the full order (ties included) reproduces a literal transcription of `test.py` on the same raw features, 200/200 trials (§9 table, §9.4) | NONE |
| 2 | `-10e10` in-row masking and `eligible = gallery_size - 1` consequence | **VERIFIED** (the mask and the gallery size); the `eligible_count` derivation is **UNKNOWN** against the native code (native has no such variable) and it bounds *stored* depth by `min(top_k, G-1)` (§9.5) | P2 |
| 3 | three-category bundle: files, upstream load, workbench declaration, silent substitution | files/digests **VERIFIED**; upstream load path **VERIFIED**; the *evaluated* weights cannot be substituted, but the *recorded* member identity is self-declared and CAN be — REPRODUCED (§9.2) | P2 |
| 4 | `per_query_state` can never be `PER_QUERY_VERIFIED` from the producer | **VERIFIED** — the run can only return `PER_QUERY_EXPORTED_UNVERIFIED` (§9.1, code is a 3-branch function) | NONE |
| 5 | pinned source requiring a dependency the workbench env does not provision | **VERIFIED** — none when provisioned from `requirements.txt`; observed import set == provisioned set; a `pip check` clean (§4, §9.6) | NONE |
| 6 | SNAPSHOT 2: upstream changed since the pin? | **VERIFIED NO** — `ls-remote` == pin; `fetch --dry-run` fetches nothing (§2) | NONE |

---

## 2. SOURCE STATUS

**Accessible?** Yes. Real git checkout, single commit, clean except bytecode:

```
$ git -C workbench/third_party/LIMN status --porcelain
?? LIMN+/__pycache__/
?? LIMN/__pycache__/
$ git -C workbench/third_party/LIMN rev-list --count HEAD
1
$ git -C workbench/third_party/LIMN log --format='%H %ci %s' -1
7d7bc9b116f594a65ac22457491edf28a88d3c3e 2026-04-05 19:33:05 +0800 Update README.md
```

`source_pin_status` accepts this: `limn.validate_source(Path('workbench/third_party/LIMN').resolve())`
→ `[]` (OBSERVED, §13). The `__pycache__` entries are `is_bytecode`-filtered by
`common.py:76-81`, which is the workbench's documented and correct behaviour.

**Complete?** For the FashionIQ path, yes. `LIMN/{model,Transformer,MHTransformer,datasets,test,train,
utils,cirr_test_submission}.py` all present. Cross-family modules that do not exist in the pin are
never imported by the FashionIQ path; see §4. Also present: `correct_files/correction_dict_*.json`
(4 files, byte-identical to the staged dataset dictionaries — §6).

**Missing modules** (upstream, not workbench):
- No `__main__`/CLI in `LIMN/test.py`: `grep -n __main__ LIMN/test.py` → no match (rc=1); the file
  defines exactly `test` (:23), `test_cirr_valset` (:99), `test_fashion200k_dataset` (:182). The
  reproducible entry point really is a *function*, reached upstream only from training
  (`train.py:231 t = test.test(args, model, dataset_list[_id], _name[1])`).
- `LIMN/model.py:15` hard-codes an unedited placeholder:
  `open_clip.create_model_and_transforms('ViT-L-14', pretrained=os.path.join('[CLIP path]/laionCLIP-ViT-L-14-DataComp.XL-s13B-b90K', 'open_clip_pytorch_model.bin'))`.
  Copying `model.py` onto `sys.path` (which `workbench/replay/limn.py:163-168` does) makes
  `import open_clip` in `model.py:5` an ABSOLUTE import that resolves to the installed package, and
  `create_model_and_transforms` is only reached from `Backbone.__init__` — which is NOT reached
  because the checkpoint is unpickled (§5). Verified: no `open_clip_pytorch_model` string exists in
  any of the three checkpoint pickles (`mentions filepath: -1`).

**SNAPSHOT 2 — current upstream state:**

```
$ git -C workbench/third_party/LIMN ls-remote origin HEAD
7d7bc9b116f594a65ac22457491edf28a88d3c3e	HEAD
$ git -C workbench/third_party/LIMN ls-remote origin
7d7bc9b116f594a65ac22457491edf28a88d3c3e	HEAD
7d7bc9b116f594a65ac22457491edf28a88d3c3e	refs/heads/main
$ git -C workbench/third_party/LIMN fetch origin --dry-run
(no output; exit 0)
```

`origin/main` is the tip and equals the pin. Upstream has added **no CLI**, **no dependency fix** and
**no new checkpoint** for FashionIQ since the pin. The only post-pin commit message recorded anywhere
in this checkout is the pin's own `Update README.md`; the checkout is shallow (1 commit), so earlier
history is not present locally and is not claimed.

Separately, the author Hub repo referenced by the paper's README was queried live (read-only, no
download of weights):

```
$ curl -s https://huggingface.co/api/models/iLearn-Lab/TPAMI24-LIMN
{"id":"iLearn-Lab/TPAMI24-LIMN","sha":"30560ad575a56c39cc39047d1a474269338c77c0",
 "lastModified":"2024-07-13T04:19:12.000Z","gated":false, ...}
$ curl -s "https://huggingface.co/api/models/iLearn-Lab/TPAMI24-LIMN?blobs=true"
0_dress_best_model.pt   1792010157 96a335d4eb8b1e77d35575aab9177637208e2eb6c59e12f833f67e5cb0e7fc1c
0_shirt_best_model.pt   1792010157 9260c7f77d3017d8cc10e6652cc8cfe275dbc9704366fa2bf581a24808930d74
0_toptee_best_model.pt  1792010157 79f663a14d89d455fdc3612b0b36d011c06d7957d1475a654755e1053a6cece2
```

All three registry `expected_size_bytes` and `expected_sha256` values (`models.yaml:965-966,974-975,
983-984`) match the Hub's LFS metadata byte-for-byte, and the Hub revision in the registry URL
(`.../resolve/30560ad5...`) matches the repo's current `sha`. Hub `lastModified` 2024-07-13 →
upstream has not moved. **STATIC_AUDIT.**

`README.md:110` documents checkpoint filenames as `checkpoint/0_{dataset}_best_model.pt` and
`README.md:151` links `HaokunWen/LIMN` (which 301-redirects to `iLearn-Lab/TPAMI24-LIMN`:
`curl .../api/models/HaokunWen/LIMN` → `Temporary Redirect ... /api/models/iLearn-Lab/TPAMI24-LIMN`).

**Verdict: accessible, complete for the FashionIQ path, no upstream movement, no upstream CLI.**

---

## 3. NATIVE EVALUATION WORKFLOW

Entrypoint (function, not a script): `third_party/LIMN/LIMN/test.py:23
def test(params, model, testset, category)`.

Function chain, quoted:

```
test.py:24      model.eval()
test.py:25      (test_queries, test_targets, name) = (testset.test_queries, testset.test_targets, category)
test.py:28      with torch.no_grad():
test.py:33-39   for t in tqdm(test_queries, ...): imgs += [t['source_img_data']]; mods += [t['mod']['str']]
                ... imgs = torch.stack(imgs).float().cuda()
test.py:41-42   dataset_type = torch.ones(imgs.shape[0]).cuda() * dataset_name_dict[category]
                f = model.compose_query(imgs, mods, dataset_type)[0]
test.py:52-61   gallery: imgs += [t['target_img_data']] ... imgs = model.compose_target(imgs, dataset_type)[0]
test.py:67-70   for i in range(all_queries.shape[0]): all_queries[i,:] /= np.linalg.norm(all_queries[i,:])
test.py:74      sims = all_queries.dot(all_imgs.T)
test.py:76-78   test_targets_id = [i['target_img_id'] for i in test_targets]
test.py:80-82   if name != 'birds': for i, t in enumerate(test_queries):
                    sims[i, test_targets_id.index(t['source_img_id'])] = -10e10
test.py:85      nn_result = [np.argsort(-sims[i, :])[:50] for i in range(sims.shape[0])]
test.py:88-96   for k in [1, 10, 50]: hits / len(nn_result) * 100  ->  ('{name}_r{k}', r)
```

Dataset construction (the gallery and the captions):

```
datasets.py:44-48   self.image_dir = self.path + 'resized_image'   # NOTE: concatenation, no separator
                    self.dataset_type = dataset_name_dict[self.name]
datasets.py:136-137 images = json.load(open(os.path.join(self.split_dir, "split.{}.val.json".format(self.name))))
datasets.py:139-146 ref_captions = json.load(...cap.{name}.val.json); correction_dict = json.load(...)
                    mod_str = self.concat_text(caption['captions'], correction_dict)
datasets.py:150-152 out['source_img_id'] = images.index(candidate)
                    out['target_img_id'] = images.index(target)
datasets.py:156-165 test_targets_id = first-seen ordered union of source_img_id then target_img_id
                    test_targets = [{target_img_id, target_img_data} for i in test_targets_id]
datasets.py:82-91   correct_text/concat_text: punctuation -> ' ', lower, split, dict lookup, " and ".join
datasets.py:120-127 get_test_img(...) -> img = self.transform[1](img)     # VAL transform
datasets.py:50-64   caches test_queries_/test_targets_ to .pkl and reloads unconditionally if present
```

Environment: `README.md:18-24` (GPU A100 40G, CUDA 12.4, torch 1.12.1, torchvision 0.13.1,
open-clip-torch 2.20.0); `requirements.txt` pins `torch==1.12.1`, `torchvision==0.13.1`,
`open-clip-torch==2.20.0`, `numpy==1.21.5`, `Pillow==9.4.0`, `tqdm==4.65.0`, `packaging==23.1` (§4).

**Function chain a wrapper must call**: load a whole-model checkpoint, build the native `FashionIQ`
dataset with `transform=[preprocess_train, preprocess_val]`, then
`test(params, model, testset, category)` with a `params` object carrying `local_rank` and
`batch_size`, on a CUDA device. `test.py:24,39,41` are why: `.cuda()`, `torch.ones(...).cuda()` and
`dataset_name_dict[...]` are unconditional.

**Missing components**: no CLI (`test.py`), no inference-time weight acquisition path (`model.py:15`
placeholder), no release of the CLIP data-comp backbone weights inside the checkpoint (they are, in
fact, inside — §5), no source-provided standalone loader in `LIMN/`. Also note `datasets.py:44-48`
concatenates `path + 'resized_image'`, so a path without a trailing slash produces
`<root>resized_image`; `workbench/replay/limn.py:190` passes `path=f"{cache_root}/"` (correct), and
the workbench test `test_limn_query_export_e2e.py` happens to pass a path WITH separators, so this
latent upstream bug is neutralised by construction, not by accident.

---

## 4. DEPENDENCIES

Method: `environment_planner._evaluator_imports` statically walks from
`workbench/replay/limn.py` with roots `[repo, third_party/LIMN/LIMN, third_party/LIMN]`
(`environment_planner.py:250-251`).

```
$ .venv/bin/python -c "... ep._evaluator_imports(model_by_id('limn')) ..."
IMPORTS: ['PIL', 'numpy', 'open_clip', 'packaging', 'torch', 'tqdm', 'yaml']
```

The `yaml` module is not imported by any LIMN source file (verified: `grep -n yaml` over the
checkout's `*.py` → no match); it is pulled by `workbench/backend/registry.py:9 import yaml`, which
`limn.py:18` imports. Every walked name resolves to a distribution that the recorded environment
actually installed:

```
$ .venv/bin/python -c "... ep.model_constraints(model_by_id('limn')) ..."
PACKAGES: {"numpy":"1.21.5","open-clip-torch":"2.20.0","packaging":"23.1",
           "pillow":"9.4.0","tqdm":"4.65.0","pyyaml":"UNKNOWN"}
RANGES: {}   PIP_ONLY: ['open-clip-torch','torch','torchvision']
```

Recorded plan and install (`workbench/artifacts/environment_assignment.json`, group
`cir-1bbd9872cf166f0d`, members `["limn"]`):

```
conda create: micromamba create --prefix .../cir-1bbd9872cf166f0d --override-channels \
   -c conda-forge -c pytorch -c defaults python=3.10 pip numpy=1.21.5 packaging=23.1 \
   pillow=9.4.0 pyyaml tqdm=4.65.0 --yes        -> exited 0
pip install:  .../bin/python -m pip install torch==1.12.1 torchvision==0.13.1 open-clip-torch==2.20.0 -> exited 0
console-script check: 20 python console-script shebang(s) point into this environment
INSTALLED: conda and pip package installation succeeded
RUNTIME_VERIFIED: pip check, exact versions, and member evaluator imports verified
```
```
"packages": {"cuda-version":"12.4","numpy":"1.21.5","open-clip-torch":"2.20.0","packaging":"23.1",
             "pillow":"9.4.0","pyyaml":"*","torch":"1.12.1","torchvision":"0.13.1","tqdm":"4.65.0"}
"unresolved": ["limn: python constraint UNKNOWN (source does not establish value)",
               "limn: pyyaml version UNKNOWN (source does not establish value)"]
"verification_details": {"evaluator_imports": {"limn": {"modules": ["yaml","torch","numpy","tqdm",
   "open_clip","PIL","packaging"], "returncode": 0}}, "pip_check": {"returncode": 0, ...}}
```

| import | file:line (pinned) | owner | required version | installable? | py/torch/cuda | remedy | test result |
|---|---|---|---|---|---|---|---|
| `torch` | `test.py:1`, `datasets.py:5`, `model.py:1` | upstream | 1.12.1 | yes (PyPI) | py3.10, cu10.2 build | keep pin, or adapted profile | installed; `python -c "import torch"` now FAILS, §9.6 |
| `torchvision` | via `open_clip` only | upstream transitively | 0.13.1 | yes | — | keep | installed, same load failure |
| `numpy` | `test.py:1`, `datasets.py:2` | upstream | 1.21.5 | yes | any | keep | installed, imports |
| `open_clip` | `limn.py:168` (workbench), `model.py:5`/`train.py:11` | upstream+workbench | 2.20.0 | yes (PyPI) | — | keep | installed, same load failure |
| `PIL` | `datasets.py:4` | upstream | 9.4.0 | yes | any | keep | installed, imports |
| `tqdm` | `test.py:3` | upstream | 4.65.0 | yes | any | keep | installed, imports |
| `packaging` | `limn.py:96-100` | workbench | 23.1 | yes | any | keep | installed, imports |
| `yaml` | `workbench/backend/registry.py:9` | workbench | UNKNOWN (any) | yes | any | keep | installed 6.0.3, imports |
| `pydantic` | deliberately absent from this child | workbench | — | n/a | — | must stay out (below) | OBSERVED absent, by design |

Dependency-category classification for the missing items: **none fall in D1–D8** — there is no
missing import in the pinned FashionIQ path. The two `UNKNOWN`s above are not missing *packages*:
D5-class-only note is that `python: UNKNOWN` in the registry (`models.yaml:938`) is supplied by the
planner's own default (`environment_planner.py:1098 conda_specs.append("python=3.10")`), and
`pyyaml` is `*` (installed as 6.0.3).

`pydantic` deliberately absent from the model interpreter is load-bearing and correct:
`limn_export.py:28-36` documents and the design enforces that the model-side exporter must not import
the canonical schema, and `evaluate_models.py:1694-1700` runs the publisher as a separate child
under the workbench interpreter. `open_clip_torch` is `pip_only` because torch/torchvision/open-clip
are pinned but conda-unresolved (same assignment file, `evidence` list).

**Verdict: no unprovisioned dependency. The environment contract is internally consistent and its
import probe passed. STATIC_AUDIT + observed probe output.**

---

## 5. CHECKPOINTS

Files (bundle `base_iter0_all_categories`, `models.yaml:987-992`):

| member | size | sha256 (registry) | Hub LFS sha256 | local file |
|---|---|---|---|---|
| `0_dress_best_model.pt` | 1792010157 | `96a335d4eb8b…e7fc1c` | identical | present, hash matches |
| `0_shirt_best_model.pt` | 1792010157 | `9260c7f77d30…930d74` | identical | present, hash matches |
| `0_toptee_best_model.pt` | 1792010157 | `79f663a14d89…a6cece2` | identical | present, hash matches |

```
$ sha256sum workbench/artifacts/checkpoints/limn/*.pt
96a335d4eb8b1e77d35575aab9177637208e2eb6c59e12f833f67e5cb0e7fc1c  0_dress_best_model.pt
9260c7f77d3017d8cc10e6652cc8cfe275dbc9704366fa2bf581a24808930d74  0_shirt_best_model.pt
79f663a14d89d455fdc3612b0b36d011c06d7957d1475a654755e1053a6cece2  0_toptee_best_model.pt
$ stat -c '%n %s' .../*.pt   -> 1792010157 for all three
$ limn.validate_checkpoint_bundle(Path('workbench/artifacts/checkpoints/limn'))  -> []
```

Availability class: **AVAILABLE**, URL_VERIFIED + VERIFIED_METADATA (`models.yaml:967-968,976-977,
985-986`), `strict_clean_checkpoint_available: null` (`models.yaml:957`) — which is *not* a defect
in itself since these are clean whole-model saves.

Mapping confidence: **VERIFIED_METADATA**. The names are bound by two independent author statements:
`README.md:110` (`0_{dataset}_best_model.pt`) and `train.py:263`
(`torch.save(model, os.path.join(args.model_dir, "{}_{}_best_model.pt".format(args.i, _name[1])))`
with `--i` default `0`, `train.py:67`). Cross-checked against the authorship's own evaluation logs
published next to the same Hub revision:
`dress_0_metrics_best.json` = `{"dress_r1": 18.542…, "dress_r10": 50.7188…, "dress_r50": 74.5166…}`
→ 50.72/74.52; `shirt_0_*` → 56.08/77.09; `toptee_0_*` → 60.94/81.85. These are exactly the
category cells in `papers.yaml:1628-1635` and the exact macro in `models.yaml:930-931`. The revision
label sequence on the Hub (0/1/2 per category) matches LIMN's iteration numbering.

Checksum status: **verified locally for all three members**, and verified against author-published
LFS metadata fetched live.

Loader contract: whole-model pickles (`checkpoint_serialization: whole_model`,
`models.yaml:907`); `common.py:43-55 load_trusted_whole_model` requires a non-`None` author hash,
hashes the bytes, and only then calls `torch.load(..., weights_only=False)`. Verified structurally
that each pickle needs `model` and `Transformer` on `sys.path` (put there by
`limn.py:163-168`) plus `open_clip.model`, `open_clip.transformer`, `open_clip.tokenizer` — with
**no** external-weights reference: `mentions filepath: -1` for `open_clip_pytorch_model`, and
`conv1`/`positional_embedding`/`transformer` present in `archive/data.pkl`. So no CLIP download is
required at inference. Confirmed by the registry that no auxiliary asset is declared for this model
(§6). `UNSAFE_CHECKPOINT_SERIALIZATION` is real but is already handled by that loader path.

---

## 6. AUXILIARY ASSETS

| name | purpose | source | availability | generation possible? |
|---|---|---|---|---|
| `split.{cat}.val.json` ×3 | ordered val split; the integer positions the native dataset indexes into | FashionIQ official / operator staging | AVAILABLE (`data/fashionIQ_dataset/image_splits/`) | no — must be the official split |
| `cap.{cat}.val.json` ×3 | 2017/2038/1961 annotations; supplies `candidate`, `target`, `captions[2]` | official | AVAILABLE | no |
| `cap.{cat}.train.json` ×3 | not read by eval but asserted by `validate_dataset` (`limn.py:50-54`) | official | AVAILABLE | no |
| `correction_dict_{cat}.json` ×3 | spell correction applied to both captions | author `correct_files/` | AVAILABLE, **byte-identical to author** | no |
| `resized_image/{cat}/` ×3 | 256×256 `.jpg` the native path opens | operator resize per `README.md:51` (`resize_images.py`, ANTIALIAS/LANCZOS) | AVAILABLE, complete | yes (deterministic resize) |
| CLIP `laionCLIP-ViT-L-14-DataComp.XL-s13B-b90K` | backbone weights the author used | `model.py:15` placeholder | **NOT NEEDED at inference** — weights are inside the checkpoint (§5) | no |

Verified byte-identity of the dictionaries (the observer's alignment proof depends on this):

```
$ sha256sum data/fashionIQ_dataset/captions/correction_dict_dress.json \
            workbench/third_party/LIMN/correct_files/correction_dict_dress.json
9e4a59877098fe26036b70759387be589990eb4068692427e890113dcba2edda  (both)
$ ... shirt: ca29e29e23c0a1987d6b121219d9580bdae9872df0ea219c5aea870ee2f8567e  (both)
$ ... toptee: 3dfbf15fa6d3a30dc0f5cf7e2908ca36d1775e42aa6598f1dba7899267dc1b71  (both)
```
and record counts `dress 363 / shirt 360 / toptee 391` (author `correct_files/` sizes identical).
Dictionary *modification* texts: the pin's `CONTEXT`/README plus `AUXILIARY_ASSET_AUDIT.md:116` say
the dict is the author's; only the byte proof above was re-derived here.

Image dimensions observed: every file in each `resized_image/<category>/` is exactly 256×256
(`dress {(256,256): 3817}`, `shirt {(256,256): 6346}`, `toptee {(256,256): 5373}`), i.e. the resize
step happened and the `Resize(224)+CenterCrop(224)` val transform is a 224-crop of a 256 source.

---

## 7. FASHIONIQ PROTOCOL

Registry: **Protocol C** — `native_protocol: fashioniq_val_split` (`models.yaml:924`), whose
declaration is `fashioniq_val_split: {label: … Val Split (Pair-Union Gallery, Reference Excluded),
literature_split_label: val, evaluation_noise_pct: 0}` (`models.yaml:12-16`). The assignment's
Protocol A/B are excluded by construction, and `supported_protocols` lists only C
(`models.yaml:925`).

- **Gallery**: ordered pair-union, first-seen, reference first — `datasets.py:156-165`
  (`if i['source_img_id'] not in test_targets_id: append`, then target). Not sorted, not the full
  split. Re-derived on the real data by instantiating the pinned dataset (§13): dress 2628,
  shirt 3089, toptee 2902 entries.
- **Reference policy**: excluded by masking, never removed — `test.py:80-82`. The reference
  therefore still occupies a rank slot.
- **Query construction**: one query per val annotation, `annotation_index == i` because
  `datasets.py:144 for idx in range(len(ref_captions))` is a straight sequential loop with no
  filtering or reordering.
- **Caption input**: `concat_text` (`datasets.py:82-91`): punctuation→' ', lower, split, dict
  lookup, `"{} and {}"`. No special tokens at inference (unlike the `fashion200k` path in the same
  file, which differs).
- **Recall computation**: native integer hit counts over `len(nn_result)`, k ∈ {1,10,50}
  (`test.py:88-96`); macro = arithmetic mean over the three categories,
  `replay/common.py:105-115 macro_aggregate`; mean = (r10+r50)/2. `metrics_extraction.py:74-215`
  re-derives the macro independently and refuses a >0.05 disagreement with the declared aggregate.

Nothing in this audit proposes changing gallery, reference policy, caption input or noise
condition; the observed preflight and protocol agree with Protocol C as declared.

---

## 8. WORKBENCH INTEGRATION

**Adapter**: `backend/adapters/models.py:125-153 LIMNAdapter(OfficialScriptAdapter)` with
`script = "LIMN/test.py"` (:127) and a bespoke `command()` (:130-153) that execs
`<repo>/workbench/replay/limn.py` with `--checkpoint-root <member_root> --dataset-root
<request.dataset_root> --source-root <source>` and `--category <c>` when the selection is not the
full bundle (`:150-152`). Reached only when `command_is_audited` is true and
`command_status: REPLAY_COMMAND_IMPLEMENTED` (`models.yaml:934`). All 14 adapters are registered at
`models.py:155`.

**Planner**: `environment_planner.py:250-251` special-cases LIMN's trace entry
(`workbench/replay/limn.py`, roots `[repo, LIMN/LIMN, LIMN]`) — which is why `limn_export.py:10-16`
insists the observer must not live in `limn.py`. Env plan/install/verify recorded in §4; the one
prepared environment is shared by exactly one model. `_build_group_dict` reports `python: null`
because the registry declares `python: UNKNOWN` and the planner's own default is not folded back into
the display (`environment_planner.py:1098`); cosmetically confusing, not a defect.

**Checkpoint prep**: `evaluate_models.py:387-401` sets `artifact_type =
"category_specific_whole_models"` for `base_iter0_all_categories`, builds the member manifest over
`0_{dress,shirt,toptee}_best_model.pt` with digests read from disk, and `:424` resolves
`member_root` from the DECLARATION (`registry.py:866-887 checkpoint_member_root`) rather than from
`is_dir()` — the fix for the historical one-level-too-high bug.

**Preflight**: `limn.py:67-72 preflight` = source pin + bundle hashes + dataset layout + environment.
The dataset check (`limn.py:50-57`) requires `cap.{c}.train.json` (LIMN-specific
`fashioniq_ilearn_resized_training` layout, `registry.py:37-41`), `cap.{c}.val.json`,
`correction_dict_{c}.json`, `split.{c}.val.json`, `resized_image/{c}`. The shared
`preflight_blockers` (`backend/preflight.py:162-…`) additionally requires the interpreter tier.

**Metric parser**: `metrics_extraction.py:74-215 parse_limn_output` reads the ONE compact JSON object
`limn.py:271-279` prints, refuses duplicate/unexpected categories, requires all three + `complete`,
requires finite recall values, recomputes the macro independently and rejects >0.05 drift, then
returns `EXTRACTED_VERIFIED`. `metrics_extraction.py:402-403` routes `model_id == "limn"` here.

**Per-query support**: registry declares `per_query_observer: LIMN_QUERY_OBSERVER_V1`
(`models.yaml:904`); `evaluate_models.py:1586-1597 per_query_capability` reads it.
`evaluate_models.py:1853-1874` runs the observer only for a full-bundle run (`--category` runs are
explicitly marked `CAPABILITY_ABSENT_FOR_DIAGNOSTIC`) and only when the aggregate exited 0.
`_run_observation` (`:1620-1755`) starts `python -m workbench.replay.limn_export` in the model
interpreter with the plan's `member_root`, then publishes through a second child
`python -m workbench.replay.limn_publish` (`:1694-1700`), parses the child's one-line JSON via
`_status_object` (`:1600-1617`), re-checks the artifact's `run_id`/`model_id`/`protocol_id`
(`:1730-1740`), and computes the artifact digest **itself** (`:1741`). The publisher refuses on a
parity mismatch (`publication.py:243-252`), and the artifact is then re-read from disk and validated
(`publication.py:196-199`).

Orchestration surface: `limn.py:238-283 main()` runs `evaluate_categories` (`:216-232`) and prints the
aggregate JSON; `limn_export.py:106-175 main()` is the observer CLI. The adapter command bypasses
`limn.py main()` and therefore does **not** produce the per-query artifact: only
`evaluate_models.py` does. No `--observe` flag exists on `limn.py` `main()` (verified: no such
argument at `:239-248`; the only names, besides `--category`, are `--preflight-only` and
`--output`). This supersedes the prior audit's claim that per-query wiring was absent — expected:
it now lives in `evaluate_models.py`, which is where it belongs, and the same earlier audit's cited
line (`evaluate_models.py:1453 "per_query_export_available": False`) no longer exists; the field now
reads the real artifact (`evaluate_models.py:1890`). Those two prior claims are **STALE**, not facts.

---

## 9. CONFIRMED DEFECTS

Ranking derivation re-checked line by line, both sides quoted:

| step | pinned native | observer | same? |
|---|---|---|---|
| candidate list | `test.py:25` `testset.test_queries/test_targets` | `observer:267-268 dataset.test_queries/test_targets` | YES (identical objects — `limn_export.py:41-53` runs both consumers on ONE loaded model/dataset) |
| query features | `test.py:41-42` `compose_query(imgs, mods, dataset_type)[0]` | `observer:290-294` identical call, same batching rule (`>= batch_size or is last`) | YES |
| target features | `test.py:60-61` `compose_target(imgs, dataset_type)[0]` | `observer:301-305` identical | YES |
| dataset type | `test.py:41` `torch.ones(n).cuda() * dataset_name_dict[category]` | `observer:291,302` `torch.ones(n).cuda() * dataset.dataset_type`, and `datasets.py:48 self.dataset_type = dataset_name_dict[self.name]` | YES |
| normalization | `test.py:67-70` per-row `/= np.linalg.norm(...)` | `observer:139-149` row norms then divide | YES |
| similarity | `test.py:74` `all_queries.dot(all_imgs.T)` | `observer:148` `queries.dot(targets.T)` | YES |
| mask | `test.py:80-82` in-row `-10e10` at the reference column | `observer:150` `sims[index, gallery_index] = NATIVE_REFERENCE_MASK` (`:38 = -10e10`) | YES |
| gallery mapping | `test.py:76-78` `test_targets_id`, `datasets.py:156-165` | `observer:276-280` reuses `test_targets` verbatim; `position_of` inverts the SAME list (`observer:280`) | YES |
| argsort | `test.py:85` `np.argsort(-sims[i,:])[:50]` | `observer:162` `np.argsort(-sims[index,:])` (full; `[:50]` is a *save* bound) | YES on the full order |
| recall | `test.py:88-96` hits/len*100 | `observer:52-56 recall(k)` and `cross_check_with_native:352-370` exact-equality on rounded values | YES |

Empirical confirmation on the real dataset, using the pinned native dataset object and the
workbench's own alignment proof (§13 command `native_dataset_check.py`): 2017/2038/1961 queries,
2628/3089/2902 gallery entries, `assert_alignment: PASS` for all three categories — meaning every
native query's integer positions map to the annotation's own `candidate`/`target` and the recorded
model text equals the recomposed text from the on-disk dictionary. A random parity harness
(`/tmp/audit/parity2.py`, 80 trials, RAW features into both paths) shows **0 failures** over recall,
top-50 sets and rank positions.

### 9.1 — NONE — `per_query_state` cannot be `PER_QUERY_VERIFIED` from the producer (question 4 CONFIRMED)

`evaluate_models.py:1574-1583`:

```
def per_query_state(capability: str | None, artifact: Path | None, note: str) -> str:
    if not capability:  return "CAPABILITY_ABSENT"
    if artifact is None: return "PER_QUERY_EXPORT_FAILED"
    return "PER_QUERY_EXPORTED_UNVERIFIED"
```

Three branches, none of which can return the fourth vocabulary member
(`evaluate_models.py:1565-1571` defines `PER_QUERY_VERIFIED`). The only promotion is the launcher's,
gated on a real verifier verdict plus a parent-computed artifact digest (`pc.sh:1864-1866`), and the
schema side refuses self-certification (`results.py:249-372`). Severity NONE; no action.

### 9.2 — P2 — the per-query artifact's recorded bundle identity is self-declared and nothing compares it to the plan (question 3)

Files: three whole-model checkpoints (bundle `base_iter0_all_categories`, `models.yaml:987-992`).
Upstream load: `train.py:263` writes them; `test.py` receives ONE model per category and never
touches a sibling. Workbench declaration: `models.yaml:960-992` (three `checkpoint_variants` + one
bundle) with local hashes verified (§5).

**What is safe.** The *evaluated* weights cannot be substituted. Both processes that load weights
hash every member against the registry before loading:
`limn.py:67-72 preflight` → `validate_checkpoint_bundle` (`limn.py:37-47`, per-file
`sha256_file` vs `expected_sha256`, refuses on mismatch) → `common.py:43-55
load_trusted_whole_model` (hashes again, and only then `torch.load(..., weights_only=False)`). The
observation process re-runs the same `preflight` (`limn_export.py:121`), and the aggregate side is
additionally bound by the completion proof: `evaluate_models.py:737-810 _members_live` recomputes a
manifest over the live member files and compares it with the recorded
`bundle_manifest_digest`, which `_recorded_report_manifest` (`:2096-2105`) reads from the run's own report
(`:806-809`). So a swapped member on disk would be caught on the aggregate path.

**What is not safe.** The canonical PER-QUERY artifact's bundle identity is written by the publisher
from the payload map, verbatim (`limn_publish.py:76-85`):

```
members = payload.get("checkpoint_members") or {}
bundle_digest = manifest_digest(members) if members else None
... checkpoint_sha256 = next((member.get("sha256") for member in members.values()), None)
```

Those `filename` / `sha256` values flow straight into `RunMetadata.checkpoint_bundle` /
`bundle_manifest_digest` / `checkpoint_sha256` (`publication.py:105-140`). Nothing recomputes them
and nothing compares them with the parent's own `plan.bundle_manifest_digest`:
`_run_observation` re-checks only `run_id`, `model_id` and `protocol_id` on the artifact
(`evaluate_models.py:1729-1740`), and the admission policy enforces only
`category_checkpoint_ids` uniqueness/completeness (`results.py:280-288`) plus "one of
`checkpoint_sha256` / `bundle_manifest_digest` is non-empty" (`results.py:294-295`). Grep of every
reader of the field: `evaluate_models.py:764,838` (the aggregate proof, above),
`schemas/results.py:294-295`, `limn_publish.py:97`, `publication.py:125` — no consumer compares the
per-query artifact's manifest with the report's.

Reproduction (synthetic payload, no writes under `/home/heheboiz/data/cir`; output in `/tmp`):

```
$ .venv/bin/python -c "... publish_payload(payload with members={'dress': {'filename':
  '0_dress_best_model.pt','checkpoint_id':'base_iter0_dress','sha256':'aa'}}) ..."
PUBLISHED with bogus sha256=aa and a single-member map
{'checkpoint_id': 'base_iter0_all_categories',
 'bundle_manifest_digest': 'b59177c8fe02062023bf6e62c6e99f28ebf48417bd979e91fb3a8e2c76d25589',
 'checkpoint_bundle': {'dress': {'filename': '0_dress_best_model.pt',
     'checkpoint_id': 'base_iter0_dress', 'sha256': 'aa'}},
 'category_checkpoint_ids': {'dress': 'base_iter0_dress'},
 'checkpoint_sha256': 'aa',
 'data_kind': 'experiment',
 'instrumentation_parity': 'INSTRUMENTATION_PARITY_VERIFIED'}
```

A two-character fake digest, a one-member "bundle", and no `path` key were all accepted; the
artifact validates and `verify_run_identity` passes (`publication.py:201-232` requires only a
plausible non-`dev-` run id and non-empty command/environment digests, both supplied by the parent
from the invocation rather than hashed from weights). The same is structural for real runs: the
honest path copies registry constants (`limn_export.py:96-100`), so a *wrong record on a correct
execution* is possible only through a producer bug or a mutating actor — but the consequence is
invisible either way, which is exactly the property F-12 asserts ("a substituted weight would be
invisible in the recorded provenance", `results.py:280-288`). Also note the single-category
diagnostic path records no bundle field at all (`evaluate_models.py:387-409` only fires for the
full bundle; `replay_category` returns just `checkpoint_sha256`, `limn.py:211-214`), so a
per-category record carries no per-category weight identity — mitigated, not closed, by the
schema's "category / category_gallery_sizes" rules.

Fix shape (~8 lines, machinery already present): in `_run_observation`, after parsing the published
artifact, compare its `run.bundle_manifest_digest` with the parent's own
`plan.bundle_manifest_digest` (or recompute `manifest_digest(_plan_members(plan))`) and refuse on
disagreement. Regression check: mutate one member digest in the staged payload and assert the
parent refuses the artifact.

### 9.3 — P2 — `data_kind` is inferred from a run-id prefix, so the identity contract has a soft edge

`limn_publish.py:108-110`: `data_kind="experiment" if not str(payload["run_id"]).startswith("dev-")
else "mock"`. Combined with `publication.py:230-234` (a `dev-` id on an `experiment` is refused, and
a non-`dev-` id passes), the only thing separating a declared experiment from a development artifact
is the absence of three leading characters. The parent normally overrides this by passing the
orchestrator's `run_id` (`evaluate_models.py:1676`), so this is not exploitable through the official
path; a direct invocation of `limn_export`/`limn_publish` can label a hand-made payload enough to
reach `PER_QUERY_UNVERIFIED → PER_QUERY_VERIFIED` given a verifier receipt, because the promotion
check is digest-based and the digest is computed by the parent over whatever bytes exist. Severity
P2 (defence in depth). Action: derive `data_kind` from an orchestrator-supplied field with a
recorded provenance, or require the run id to match the orchestrator's documented format
(`evaluate_models.validate_run_id`) rather than a negative prefix test.

### 9.4 — NONE — REJECTED_FINDING: an earlier draft of this audit claimed an out-of-bounds rank defect; the claim was an artifact of a bad control

I first suspected that extending the ranking past the native `[:50]` boundary (`test.py:85`) changes
the *order*, because the masked reference column is written as the Python float `-10e10`
(`observer:38,150`, `test.py:82`) and I believed that write upcasts the `float32` similarity matrix
to `float64`, changing `np.argsort`'s tie-breaking for bit-identical gallery scores.

That suspicion is **wrong**, and the control I used to establish it was invalid: it compared
`np.argsort(-n[i,:])` on an already-normalized `float32` matrix against
` np.argsort(-(q[i:i+1].dot(t.T))[0,:])` on the same normalized inputs, i.e. the second operand was
re-normalized (a no-op numerically but an upcast to `float64` in `dot`), so the two operands were
not the same numbers. Re-tested with the SAME raw features fed to both paths:

```
$ .venv/bin/python /tmp/audit/dtype_check.py
200 trials, order mismatches: 0
upcast dtypes: float32 float32 identical order: True
```

The second line is the decisive one: in this NumPy, item assignment of the Python float `-10e10`
into a `float32` array does **not** upcast it (`c1.dtype` is `float32`, and the sort order is
identical to writing `np.float32(-10e10)`). The dtype is therefore the same on both sides, and the
observer's full order equals a literal transcription of the native mask+argsort over the whole row,
ties included. The faithful end-to-end harness (`/tmp/audit/parity2.py`, raw features into both
paths, 80 trials) independently reports **0 failures** over recall, top-50 sets and rank positions,
including trials constructed to be full of exact ties.

Residual, non-defect difference, stated for completeness: the native path can only ever report
positions inside its `[:50]` cut, while the observer walks the full row; that is the entire purpose
of the observer, and the parity check (`observer:352-370`) deliberately compares only the
r1/r10/r50 hit counts, which are in-bounds by definition. Severity NONE, finding REJECTED — recorded
here so nobody re-derives the same false positive. No code change is warranted; the earlier draft's
"sort on a copy before writing the mask" advice is withdrawn.

### 9.5 — P2 — the `eligible_count` derivation is a workbench assumption the native code does not make

`observer:344-347`:

```
gallery_size=len(gallery_ids),
eligible_count=len(gallery_ids) - 1,
```

and `observer:313 eligible_depth = min(top_k, len(gallery_ids) - 1)`. The native code has **no**
eligible-candidate variable: `test.py:85` sorts the whole row of `G` candidates with the reference
pinned at `-10e10`, and `test.py:88-96` compares positions, never counts. So `eligible = G - 1` is a
*derived* artifact of the mask semantics, and `results.py:34-41` (Protocol C ∈
`REFERENCE_EXCLUDED_PROTOCOLS`) then treats `G-1` as a hard bound: a rank > `G-1` raises
(`publication.py:100-108`; `results.py:165-200`). Reproduced:

```
$ .venv/bin/python -c "... mk(2627) ... mk(2628) ..."
rank 2627 -> 2627
rank 2628 -> REFUSED: ValueError dress:0:R:T: target_rank 2628 exceeds the eligible candidate count for dress (2627)
```

Two consequences worth naming: (a) at the last saved depth the slot can hold the masked reference, so
a stored Top-K of `min(top_k, G-1)` can be one entry shorter than native's own Top-50 would be if a
native run ever saved depth ≥ G — it does not, so this is latent; (b) `results.py:171-200` requires
`top_results` length == the effective depth, so a legitimate target at rank `G` (possible only if the
reference is not in the gallery) would be refused. Verified on the real data this cannot arise for
LIMN: the reference is always in the pair-union gallery, and there is no `candidate == target`
annotation (`dress/shirt/toptee candidate==target: 0 0 0`), so each query masks exactly one column.
Severity P2, latent. Action: keep `eligible_count` as a declaration but derive the *rank* bound from
the actual candidate set (`G - |masked columns|`) instead of from `G-1`, or record the exclusion
explicitly so the two can never be conflated. **Do not** change gallery or reference policy.

### 9.6 — P1 — the provisioned environment does not load on this host

```
$ workbench/artifacts/environments/cir-1bbd9872cf166f0d/bin/python -c "import torch"
  File ".../torch/__init__.py", line 202, in <module>
    from torch._C import *  # noqa: F403
ImportError: libtorch_cpu.so: cannot enable executable stack as shared object requires: Invalid argument
$ readelf -lW .../torch/lib/libtorch_cpu.so | grep GNU_STACK
  GNU_STACK  ... RWE 0x10
$ .../bin/python -c "import torch,..."   -> torch FAIL, torchvision FAIL, open_clip FAIL,
                                            numpy OK 1.21.5, PIL OK 9.4.0, tqdm OK 4.65.0,
                                            packaging OK 23.1, yaml OK 6.0.3, pydantic FAIL (by design)
```

Cause: `torch==1.12.1` from the PyPI wheel (the Linux wheel's `libtorch_cpu.so` requires an
executable stack) against the loader policy of this kernel (`linux 7.0.0-38-generic`), so the
prepared interpreter cannot stand up its native evaluator on THIS machine. Classification:
`FIRST_PARTY_ENVIRONMENT_CONFIGURATION` + `LEGACY_RUNTIME_INCOMPATIBILITY` (a workbench-provisionable
environment whose install was record-verified then cannot run; also note this torch bundles
`libcudart 10.2`, `version.py:3 cuda = '10.2'`, against the registry's declared `cuda: 12.4`
(`models.yaml:941`)). Severity P1. Action: record the loader/policy fact on the environment report
(refuse `RUNTIME_VERIFIED` for a host where `import torch` inside the interpreter fails — the
assignment's own gate already requires this class of check), and on such a host provision either a
newer `1.12.x` build (a variant, badged as such) or the declared `adapted_runtime`
(`models.yaml:948-956`, `pip_index_url: https://download.pytorch.org/whl/cu128`). Do **not** treat
this as an upstream fix: the authors' own pin is CUDA 12.4 + torch 1.12.1 and the checkpoint loads
with `map_location="cpu"` (`common.py:54`).

### 9.7 — P3 — stale prior-audit citations (not defects in the code)

`final_model_audit/limn.md:391` names `workbench/replay/limn.py:replay_category_with_observation()`
and `:396` points at `evaluate_models.py:1453 "per_query_export_available": False`. Re-derived:
`replay_category_with_observation` does not exist anywhere in the tree except inside that document
and one phrase in `FINAL_ADVERSARIAL_REVIEW.md:75` (the observer is invoked from
`evaluate_models._run_observation`), and `evaluate_models.py:1453` is now inside `report_identity`
while the field lives at `:1890` reading the real artifact. The cited *conclusion* ("export
implemented, not wired into the pipeline") is now false. Severity P3 (documentation integrity).
Note: `final_model_audit/limn.md:403-420` cite `datasets.py:44` and `datasets.py:58-64`; in THIS pin
those are `:44 self.image_dir` and `:50-64` (the pickle cache), so those line numbers are also off
by a few lines. I did not use those documents for any finding above.

---

## 10. REMAINING BLOCKERS

1. **GPU_RUNTIME_UNTESTED / ACTUALLY_READY_PENDING_GPU — ownership: operator + workbench.**
   No inference was executed (no GPU here; `validate_environment` would also refuse). The blocking
   evidence needed is one run of `limn_export` on one category on the target host, plus the
   aggregate `python workbench/replay/limn.py` run, both on the provisioned interpreter.
2. **AUXILIARY_ASSET_MISSING — ownership: upstream (image resizing).** No resized-image
   distribution is published by the authors (`README.md:51` is instructions only, pointing at
   FashionIQ's `start_kit/resize_images.py`). The staged dataset is complete on disk (3817/6346/5373
   images, all 256×256), but "complete" is asserted by the workbench's own resize receipt
   (`resized_image/.workbench_resize_receipt.json`), not by an author checksum. Severity P2, and it
   is *the* residual risk for the pilot's number: if the resize filter or the source images differed
   from what the authors used, the six published digits would drift. Evidence needed to settle it:
   the authors' own resized-image digest, or a documented FashionIQ release hash. This is
   `NONDETERMINISTIC_PREPROCESSING` only if the filter differs; the receipt claims LANCZOS and I did
   not independently verify the filter against the upstream script (network fetch of
   `XiaoxiaoGuo/fashion-iq` was not attempted).
3. **PUBLIC_ASSET_NOT_PREPARED — the CLIP backbone (`model.py:15` placeholder).** Not a blocker for
   THIS reproduction because the backbone weights are inside the checkpoint (§5), but it *is* a
   blocker for anything that constructs `Backbone.__init__` (training, LIMN+). If any future path
   touches that constructor, evidence needed: the exact
   `laionCLIP-ViT-L-14-DataComp.XL-s13B-b90K/open_clip_pytorch_model.bin` bytes.
4. **UPSTREAM_CLI_MISSING — ownership: upstream, permanent.** `LIMN/test.py` is a module of
   functions with no `__main__`; the workbench wrapper is the only reason a CLI exists. Not
   actionable upstream and not a defect of the workbench — recorded so it is never mistaken for one.
5. **Suspicion not settled (needs one real run), P3: whether the observer's in-process
   re-derivation reproduces the native call *bit-for-bit on GPU***, as opposed to reproducing
   its arithmetic. `test.py:24 model.eval()` is the only eval-mode statement in the pinned
   test code, and `limn_observer.observe_category` has no matching `model.eval()`; it is safe
   only because `native_test.test()` ran first on the same object (`limn_export.py:44-52` calls
   `native_test.test(...)` before `observe_category(...)`), and every `nn.Dropout` in the model
   stack defaults to `p=0` (`model.py:118 dropout=0.0`; `Transformer.py:56,62` `dropout=0.`).
   Verified: no module in the stack sets a non-zero dropout (`grep -n nn.Dropout` → 4 sites, all
   fed the 0 defaults), and no batch norm is present. So train mode is a numerical no-op here.
   Evidence needed anyway: one run printing `model.training` inside both consumers, so a future
   nonzero-dropout variant cannot silently make the observation differ from the aggregate.

No `EXTERNAL_BLOCKER` classification is used anywhere in this report; each of the above is stated
with one of the assignment's tokens.

---

## 11. IMPLEMENTATION PLAN

Ordered, smallest-first, no new abstraction:

1. **Parent cross-checks the per-query artifact's bundle manifest** (`P2`, §9.2). In
   `evaluate_models._run_observation`, after reading the published artifact, compare
   `run.bundle_manifest_digest` with the parent's own `plan.bundle_manifest_digest` (or
   `manifest_digest(_plan_members(plan))`) and refuse on disagreement. ~8 lines, reuses
   existing helpers. Complexity: low. Scientific risk: none (it can only refuse). Regression
   check: a test that publishes with a mutated `sha256` and asserts the parent rejects it.
2. **Derive the rank bound from masked columns, not `G-1`** (`P2`, §9.5). Record
   `eligible_count = gallery_size - len(set(reference_positions))` and keep the schema's Protocol C
   ceiling as a *ceiling* only. Do not touch gallery or reference policy. Changes only refusal
   boundaries that are unreachable on the real data.
3. **Have the environment report carry the native-load result** (P1, §9.6): refuse
   `RUNTIME_VERIFIED` when `import torch` inside the interpreter fails, and record the loader error
   verbatim. On this host, provision the declared `sm_120` profile instead of the historical pin.
4. **`data_kind` from a recorded provenance field, not a prefix test** (`P2`, §9.3).
5. **Docs** (§9.7): correct `final_model_audit/limn.md`'s stale symbols/lines or mark the document
   superseded.

Expected outputs after 1–5: the aggregate path is unchanged; the per-query artifact either
matches the on-disk bundle or is refused; the eligibility edge is closed and the run-id-based
`data_kind` edge is removed. No expected change to any reported metric.

---

## 12. FINAL VERDICT

- **SOURCE**: **VERIFIED** — accessible, complete for the FashionIQ path, pin exact, upstream
  unmoved (SNAPSHOT 2 == SNAPSHOT 1).
- **PROTOCOL**: **VERIFIED** — `fashioniq_val_split` (Protocol C) as declared; gallery and reference
  policy re-derived from the pinned code and reproduced on the real data.
- **COMMAND**: **VERIFIED** (static) — `LIMN/test.py:test()` is invoked faithfully by
  `replay_category`; the entry point is a function upstream, and the wrapper is the CLI.
- **OBSERVER DERIVATION**: **VERIFIED** — same objects, same calls, same grouping, same mask
  expression, same argsort; the observer's full order reproduces a literal transcription of
  `test.py` on the same raw features (200/200 trials), and `assert_alignment` passes on the
  real dataset for all three categories. One earlier suspicion (§9.4) was REFUTED.
- **CHECKPOINT**: **VERIFIED** — three files, sizes and sha256 match author LFS metadata and the
  local bytes; mapping corroborated by the authors' own iteration-0 metric files.
- **ENVIRONMENT**: **BLOCKED** on this host — provisioned and import-verified at install time, but
  `import torch` now fails with `libtorch_cpu.so: cannot enable executable stack` (§9.6).
- **PER-QUERY CAPABILITY**: **VARIANT** — implemented, upstream-compatible, and structurally unable
  to self-promote; one P2 self-declared-provenance hole (`§9.2`) and one P1 runtime gap
  (`§9.6`) stand before a pilot claim.
- **RELEASE READINESS**: **ACTUALLY_READY_PENDING_GPU** for the static/pipeline axis, **BLOCKED** on
  the runtime axis for this host (`GPU_RUNTIME_UNTESTED`).

No runtime result is claimed. Nothing about this model's reproduction is RUNTIME_VERIFIED.

---

## 13. EVIDENCE

Repo: https://github.com/iLearn-Lab/TPAMI24-LIMN · pinned
`7d7bc9b116f594a65ac22457491edf28a88d3c3e` · workbench HEAD `b70d092e9d2232e34886f819e30c7327b74f3a76`.

Commands run (all read-only; the only writes were to `/tmp/audit/*` and to this report path):

```
git -C workbench/third_party/LIMN rev-parse HEAD            -> 7d7bc9b116f594a65ac22457491edf28a88d3c3e
git -C workbench/third_party/LIMN ls-remote origin HEAD     -> 7d7bc9b116f594a65ac22457491edf28a88d3c3e
git -C workbench/third_party/LIMN fetch origin --dry-run    -> (no output, exit 0)
git -C workbench/third_party/LIMN status --porcelain        -> only ?? LIMN*/__pycache__/
sha256sum workbench/artifacts/checkpoints/limn/*.pt         -> matches models.yaml:966,975,984
curl https://huggingface.co/api/models/iLearn-Lab/TPAMI24-LIMN?blobs=true
     -> the same three sha256 + size 1792010157; repo sha 30560ad5…; lastModified 2024-07-13
.venv/bin/python -c "limn.validate_source/validate_checkpoint_bundle/validate_dataset/validate_environment"
     -> [], [], [], ['environment version mismatch torch …','… torchvision …','missing open_clip_torch …']
.venv/bin/python -c "environment_planner._evaluator_imports(model_by_id('limn'))"
     -> ['PIL','numpy','open_clip','packaging','torch','tqdm','yaml']
.venv/bin/python -c "environment_planner.model_constraints(model_by_id('limn'))"
     -> packages/  pip_only ['open-clip-torch','torch','torchvision']
.venv/bin/python /tmp/audit/native_dataset_check.py
     -> dress 2017/2628/3817/2017 type=0 eligible=2627 ; shirt 2038/3089/6346/2038 type=1 ;
        toptee 1961/2902/5373/1961 type=2 ; assert_alignment PASS ×3
.venv/bin/python /tmp/audit/parity2.py                      -> trials 80, failures: 0
.venv/bin/python /tmp/audit/dtype_check.py                  -> 200 trials, order mismatches: 0
                                                                upcast dtypes: float32 float32 identical order: True
.venv/bin/python /tmp/audit/parity_check.py                 -> 8 deltas, ALL traced to that harness's
                                                                invalid control (it re-normalized one operand); superseded
                                                                by dtype_check.py and parity2.py
.venv/bin/python /tmp/audit/dbg.py                          -> the same invalid control in isolation
.venv/bin/python /tmp/audit/caption_parity.py               -> 0 mismatches over 2016+2037+1960 records
.venv/bin/python -c "publish_payload(bogus members)"        -> PUBLISHED with sha256 'aa' (§9.2)
.venv/bin/python -c "mk(2627)/mk(2628)"                     -> accepts 2627, refuses 2628 (§9.5)
.venv/bin/python -m pytest workbench/tests/test_limn_query_export_ranking.py -q  -> 41 passed
workbench/artifacts/environments/cir-1bbd9872cf166f0d/bin/python -c "import torch"
     -> ImportError: libtorch_cpu.so: cannot enable executable stack as shared object requires
readelf -lW .../libtorch_cpu.so | grep GNU_STACK           -> RWE 0x10
python3 -c "PIL.Image.open over every resized_image/<cat>/*" -> all 256×256 (3817/6346/5373)
sha256sum correction_dict_{dress,shirt,toptee}.json (dataset vs correct_files) -> identical ×3
```

Source files read (Snapshot 1): `LIMN/test.py:23-96`, `LIMN/datasets.py:39-165`,
`LIMN/model.py:1-84,115-211`, `LIMN/Transformer.py:55-94`, `LIMN/train.py:1-140,220-275`,
`README.md`, `requirements.txt`, `correct_files/*`.
Workbench files read: `replay/limn.py`, `replay/limn_observer.py`, `replay/limn_export.py`,
`replay/limn_publish.py`, `replay/publication.py`, `replay/common.py`,
`backend/schemas/results.py`, `backend/environment_planner.py`, `backend/registry.py`,
`backend/adapters/models.py`, `backend/metrics_extraction.py`, `backend/index.py`,
`backend/preflight.py`, `backend/runtime.py`, `scripts/evaluate_models.py`, `pc.sh`,
`registry/models.yaml`, `registry/paper_benchmarks/papers.yaml`,
`artifacts/environment_assignment.json`.

`STATIC_AUDIT` applies to everything except: the observed preflight outputs, the observed native
dataset instantiation and alignment proof, the observed parity/caption checks, the observed
publisher defect reproduction, the observed torch load failure, and the observed hashes/sizes.
No claim in this report is `RUNTIME_VERIFIED`.

---

## 14. ACHIEVED VERIFICATION LEVEL

**LEVEL 3 of 6 — static/fixture verification against the pinned source, with the data-side native
path actually executed on the real dataset but no model inference.**

- Level 1 (source pinned/accessible): achieved.
- Level 2 (protocol re-derived from source): achieved.
- Level 3 (observer-vs-native derivation re-derived and exercised on real data/real dataset objects,
  plus reproduced workbench-side defects): achieved.
- Level 4 (checkpoint bytes mapped to a published metric row): partly achieved — hashes and the
  authors' iteration-0 metric files match, but no run of this workload was executed.
- Level 5 (GPU inference on one category): **not achieved**.
- Level 6 (full three-category reproduction with an accepted metric): **not achieved**.

The pilot must not be described as verified on the strength of this document.
