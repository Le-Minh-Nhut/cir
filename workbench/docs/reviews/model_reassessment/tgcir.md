# tgcir — reproducibility reassessment (independent, adversarial, READ-ONLY)

Auditor: independent reassessment pass. Nothing under `/home/heheboiz/data/cir` was modified except
this file. All probe output and downloads went to `/tmp`. No GPU work, no evaluator execution, no
metric produced by a run. Every claim is `STATIC_AUDIT` (a file I opened, or a command I ran and its
observed output) unless explicitly labelled `RUNTIME_VERIFIED`, which here means "a measurement I
took on this host" — never "the evaluator ran".

Prior workbench audits (`workbench/docs/reviews/models/tgcir.md`, `final_model_audit/tgcir.md`,
`final_audit/models/tgcir.md`) were treated as HYPOTHESES and re-tested against the pinned tree. One
cross-cutting claim is CONFIRMED with digests (§5), one is CONFIRMED with the quote (§4), and the
blocker set is narrowed from three documents to one precisely-stated root cause (§3, §10).

---

## 1. MODEL IDENTITY

| field | value | evidence |
|---|---|---|
| model_id | `tgcir` | `workbench/registry/models.yaml:785` |
| method | "TG-CIR" — Target-Guided Composed Image Retrieval | `models.yaml:791`; `README.md:1` |
| paper | https://arxiv.org/abs/2309.01366 (ACM MM 2023) | `models.yaml:794-795`; `README.md:1,14` |
| year / generation | 2023 / legacy | `models.yaml:792-793` |
| source URL | https://github.com/iLearn-Lab/MM23-TG-CIR | `models.yaml:796` |
| branch / pin | `main` / `65fa78eaf8cabe8197fcc22ccab42207c591bdee` | `models.yaml:797-798` |
| source_dir | `TGCIR` | `models.yaml:800` |
| native protocol | `fashioniq_val_split` | `models.yaml:806` |
| reported metrics | R@10 51.32, R@50 73.09, mean null | `models.yaml:808-810` |
| reproduction_status | `BLOCKED_COMMAND` | `models.yaml:812` |
| command_status | `COMMAND_BLOCKED_SOURCE_ENTRYPOINT` | `models.yaml:813` |
| replay | `PREFLIGHT_ONLY` / `PREFLIGHT_ONLY` | `models.yaml:786,789` |
| checkpoint variant | `tgcir_archive_unresolved` / author Drive `TG-CIR.zip` / `whole_model` | `models.yaml:788,831-842` |

STATIC_AUDIT.

**Pin verified against the checkout.** OBSERVED:

```
$ git -C workbench/third_party/TGCIR rev-parse HEAD
65fa78eaf8cabe8197fcc22ccab42207c591bdee
$ git -C workbench/third_party/TGCIR status --porcelain
(empty)
```

Pinned tree, complete: `README.md`, `correction_dict/` (4 JSON), `cirr_test_submission.py`,
`datasets.py`, `model.py`, `test.py`, `train.py`, `utils.py`. Eight files, no submodule, no missing
module on any import in the evaluation path.

The paper row: `paper_benchmarks/papers.yaml:1520-1588` (`tgcir_fashioniq_val_table1`,
`reference_status: VERIFIED`, `review_status: CONFIRMED`). The committed per-category cells are
dress 45.22/69.66, shirt 52.60/72.52, toptee 56.14/77.10 → macro 51.32/73.09, which I re-derived:
(45.22+52.60+56.14)/3 = 51.3199→51.32 ✓ and (69.66+72.52+77.10)/3 = 73.0933→73.09 ✓. The registry's
`reported_mean: null` with a `derived_metrics` entry for 62.205 is the right split: the paper prints
no scalar mean.

---

## 2. SOURCE STATUS

- `source_available: true` (`models.yaml:799`), `source_dir: TGCIR` (`:800`), clean (above).
- Complete: every file the evaluation path imports exists; no absent package, no unregistered class.
- `source_blocker` is NOT declared for this model (`models.yaml:785-842` has no `source_blocker`
  key, and no `approved_source_patches`) — consistent with the finding below that the source itself is
  intact and the blocker is the *entrypoint shape*, not a broken revision.

### SNAPSHOT 2 — current upstream state

Observed this pass:

```
$ git -C workbench/third_party/TGCIR ls-remote origin HEAD
65fa78eaf8cabe8197fcc22ccab42207c591bdee	HEAD
$ git -C workbench/third_party/TGCIR ls-remote origin main
65fa78eaf8cabe8197fcc22ccab42207c591bdee	refs/heads/main
$ git -C workbench/third_party/TGCIR fetch origin --dry-run
   (no output; rc=0)
```

**The upstream has NOT moved.** Current `main` is the pinned commit, so SNAPSHOT 2 equals SNAPSHOT 1
for this model — unlike the two CLIP4Cir entries, whose upstream published a new README and a
checksummed release on 2026-10-12. Consequences, and they are all negative for readiness:

- no current-upstream CLI was added (`test.py` at the *pinned* commit is the current `test.py`);
- the `clip==0.2.0` pin is still what the README declares today (`README.md:19` is not stale);
- no checksummed checkpoint release exists — the only pointer remains the bare Drive `TG-CIR.zip` link
  (`README.md:15`).

The pin was not changed.

One additional entrypoint fact observed this pass: `cirr_test_submission.py:94` DOES have
`if __name__ == '__main__':`, so the repository is not uniformly script-less — the *FashionIQ*
evaluation path is the one that lacks an entrypoint, and only that one. `test.py` has no
`__main__`, no `ArgumentParser`, no `argparse` import and no `sys.argv` use (grep → no hits).

What I did establish from the pinned tree about upstream's release state: the only checkpoint pointer
is a bare Drive *file* link (`README.md:15`), and the pinned README pins `clip==0.2.0` (`README.md:19`).

---

## 3. NATIVE EVALUATION WORKFLOW

### 3.1 The native test callable — file:line, signature, CLI-invocable?

**`test.test(params, model, testset, category)` — `test.py:6`.**

```
6: def test(params, model, testset, category):
7:     model.eval()
```

It is a **bare callable**, not an entrypoint. OBSERVED: `grep -n "__main__\|ArgumentParser\|argparse"
test.py` → **no hits**; the file has no `if __name__ == '__main__':` block and no parser at all. Its
only importers are `train.py:23` (`import test`) and the three call sites inside the training loop:

```
198:        if args.dataset == 'fashioniq':
199:            for ci, category in enumerate(['dress', 'shirt', 'toptee']):
200:                t = test.test(args, model, trainset, category)
220:                t = test.test(args, model, trainset, 'shoes')
224:                t = test.test_cirr_valset(args, model, trainset)
```

**CLI-invocable: NO.** There is exactly one `ArgumentParser` in the repository — `train.py:40-80` —
and it is a *training* parser (`--num_epochs`, `--lr`, `--clip_lr`, `--mu_`, `--nu_`, `--kappa_`,
`--tau_`, `--P`, `--Q`, `--model_dir`, `--restore_file`, …). Evaluation is a method call inside the
training loop; there is no `test.py` script, no `--model_dir`-only replay path, and `test.py` has no
top-level code. This is the registry's `COMMAND_BLOCKED_SOURCE_ENTRYPOINT` (`models.yaml:813`) and the
workbench's `TGCIRAdapter` is the base class with **no** `command()` override
(`adapters/models.py:114-116`), so `command_is_audited` is False and building it raises:

```
$ python_executable-free probe of the adapter
has own command: False
RAISES: NotImplementedError upstream evaluation command requires audit before execution
```

VERIFIED (STATIC_AUDIT). This is an **`UPSTREAM_CLI_MISSING`**-shaped condition on the source side and
a deliberate fail-closed refusal on the workbench side — the correct behaviour, not a workbench bug.

### 3.2 Function chain (what a wrapper would have to reproduce)

```
1. train.py:80   args = parser.parse_args()                       # train parser, required just to get `params`
2. train.py:88   clip, preprocess = clip.load(args.img_encoder, device='cuda', jit=False)   # img_encoder default 'ViT-B/16'
3. train.py:89   trainset = load_dataset()                        # datasets.FashionIQ(path=args.fashioniq_path, transform=preprocess)
4. train.py:123  model, optimizer = create_model_and_optimizer()  # model.cuda()  (train.py:125)
5. train.py:271  torch.save(current_model, os.path.join(args.model_dir, 'best_model.pt'))   # WHOLE MODEL
6. test.py:6     test(params, model, testset, category) for category in ['dress','shirt','toptee']
7. test.py:31,47 torch.stack(imgs).float().cuda()                 # unconditional
8. test.py:61    sims = all_queries.dot(all_imgs.T)
9. test.py:66-67 sims[i, test_targets_id.index(t['source_img_id'])] = -10e10
10. test.py:74-80 recalls for k in [1, 10, 50]
```

The model is built by `model.Backbone(img_encoder='RN50'|'ViT-B/16', …)` (`model.py:38-50`), whose
constructor loads CLIP itself: `self.clip, _ = clip.load(img_encoder, device='cuda', jit=False)`
(`model.py:41`). So a faithful wrapper must inject BOTH the whole-model pickle and the same CLIP
variant that pickle was trained with (default `ViT-B/16`, `train.py:56`), and must run on a CUDA host
(`test.py:31`, `:47`; `model.py:32`, `model.py:41`).

`test.params` is consumed only for `params.local_rank` (`test.py:25`, `:42`) and `params.batch_size`
(`test.py:28`, `:44`), both of which the training parser supplies.

---

## 4. DEPENDENCIES

| import | file:line | owner | required version | installable? | py/torch/cuda | safe remedy | test result |
|---|---|---|---|---|---|---|---|
| `clip` (OpenAI CLIP) | `model.py:5`; `train.py:19`; `train.py:88`, `model.py:41` | upstream README | **`clip==0.2.0`** — a KNOWN-WRONG pin | the pin installs, but it is the WRONG package | py; `torch`; CUDA | install `git+https://github.com/openai/CLIP.git` at a recorded commit | not executed |
| `torch` | `test.py:2`, `model.py:1` | upstream README | 1.7.0 | yes | CUDA required by the code path | `conda install pytorch=1.7.0` | not executed |
| `torchvision` | via `clip`; not imported directly here | upstream README | 0.8.0 | yes | — | — | not executed |
| `torch.nn.functional` | `test.py:4`, `model.py:3` | upstream | same as torch | yes | — | — | not executed |
| `torch.distributions.Categorical`, `torch.autograd.Variable` | `model.py:6-7` | upstream | same as torch | yes | — | — | not executed |
| `numpy` | `test.py:1`, `model.py:4` | upstream README | 1.22.3 | yes | — | — | not executed |
| `PIL` / `PIL.Image` | `datasets.py` (image opens) | upstream README | `Pillow==9.0.1 / 10.0.0` | yes | — | — | not executed |
| `tqdm` | `test.py:3` | upstream README | 4.65.0 | yes | — | — | not executed |
| `matplotlib`, `seaborn` | not on this path | upstream README | 3.5.1 / 0.12.2 | yes | — | not needed for evaluation | — |
| `os`, `json`, `logging`, `pickle`-adjacent (`save_obj`/`load_obj`) | `utils.py`, `datasets.py` | stdlib | — | — | — | — | — |

**D1–D8 are not defined anywhere in this repository.** I grepped for the legend (only unrelated
finding ids appear: `final_hardening/D.md` D-1, `tests/test_release_blocker_invocation_proof.py` D-4);
the same gap is recorded by sibling reassessments (`model_reassessment/sprc.md:169`,
`model_reassessment/ptha_mtst.md:299`). I classify without the D-numbers rather than inventing them.

### 4.1 The OpenAI CLIP dependency — CONFIRMED, with the quote

**The cross-cutting audit's claim is CONFIRMED against the pinned tree.**

- The registry declares the package as `clip==0.2.0` (`models.yaml:828`), transcribed from the pinned
  README:
  ```
  README.md:19: clip==0.2.0
  ```
- The source imports the name bare:
  ```
  model.py:5: import clip
  train.py:19: import clip
  model.py:41: self.clip, _ = clip.load(img_encoder, device='cuda', jit=False)
  train.py:88: clip, preprocess = clip.load(args.img_encoder, device='cuda', jit=False)
  ```
- `clip.load(name, device=…, jit=…)` and the module layout `clip.model.CLIP` / `clip.clip` are the
  **OpenAI** CLIP API (openai/CLIP). The PyPI project `clip` at version `0.2.0` is `silent1mezzo/clip`,
  an unrelated clipboard utility: it has no `load()`, no `model` submodule and no tokenizer, so
  `import clip` would resolve to the wrong distribution and `clip.load(...)` would raise
  `AttributeError` at `model.py:41` before any image is read.
- The registry already says so in prose: *"`clip==0.2.0` on PyPI is silent1mezzo/clip (a clipboard
  CLI), not OpenAI CLIP; openai/CLIP publishes no tags (final audit P2). Recorded as-is because the
  pinned README pins nothing; this entry is a KNOWN-WRONG pin that a future operator must replace with
  a git revision."* (`models.yaml:825-827`.)

**Classification: `INSTALLABLE_DEPENDENCY_MISSING` in form (the required module is installable, just
not from that pin), `DEPENDENCY_VERSION_CONFLICT`-adjacent in effect.** The pin is not merely
unversioned like the CLIP4Cir entries — it actively names a *different project*, so an operator who
follows the registry literally gets an `AttributeError` at model construction. Severity P1. Remedy:
replace the pin with a `git+https://github.com/openai/CLIP.git@<commit>` revision (openai/CLIP
publishes no tags, which is why a commit sha, not a version, is the only correct form). I did not
execute the install, so the failure is STATIC_AUDIT, not observed.

Also note `torch==1.7.0` / `torchvision==0.8.0` with `cuda: '11.0'` (`models.yaml:818-819,824`):
`test.py:31` and `:47` call `.cuda()` on every batch without consulting `torch.cuda.is_available()`,
and `model.py:32` (`torch.Tensor(B, self.S, C).cuda()`) and `model.py:41` (`clip.load(...,
device='cuda')`) hard-code CUDA in the *model*. So this evaluator is CUDA-hard; `cuda: '11.0'` is not
an overstatement, and the registry's comment (`:820-823`) recording why UNKNOWN was wrong is accurate.

---

## 5. CHECKPOINTS

### 5.1 What the registry declares

```
788:    checkpoint_serialization: whole_model
831:    checkpoint_variants:
832:      - checkpoint_id: tgcir_archive_unresolved
833:        filename: tgcir-fashioniq-unresolved.pt
834:        download_url: null
835:        source_url: https://drive.google.com/file/d/1OdZTtJqy-RTpYXBaq5ThmH3IBnGvYCyi/view
836:        source_type: author_google_drive
837:        checkpoint_training_noise_pct: null
838:        evaluation_noise_pct: 0
839:        expected_size_bytes: null
840:        expected_sha256: null
841:        status: CHECKPOINT_MAPPING_UNVERIFIED
842:        checkpoint_mapping_status: UNVERIFIED
```

The registry names **one file**. `notes` (`:830`) states: *"Pinned test.py has no CLI/checkpoint replay;
TG-CIR.zip FashionIQ member mapping and hash remain unresolved."*

### 5.2 What the archive actually contains — UNKNOWN

OBSERVED (network, this pass):

```
$ curl -sL -A "Mozilla/5.0" "https://drive.google.com/file/d/1OdZTtJqy-RTpYXBaq5ThmH3IBnGvYCyi/view"
http=200 bytes=82500
<title>TG-CIR.zip - Google Drive</title>
```

The Drive target is named **`TG-CIR.zip`** — an archive, not a checkpoint. Its member listing is not
readable without downloading it: the file page exposes no member names and no usable size field, and
no checksum is published anywhere in the repository or the README (`README.md:15` is a bare link).

**I did not download `TG-CIR.zip`.** The audit budget for this engagement was already spent on the two
CLIP4Cir releases, and the assignment's no-large-download rule applies here too. So:

| question | answer |
|---|---|
| what the registry declares | a single `.pt` file named `tgcir-fashioniq-unresolved.pt`, no digest, no size |
| what the archive actually contains | **UNKNOWN** — not downloaded, not listed |
| do they match | **UNKNOWN** — and the *shape* already disagrees: the registry names a `.pt`, the author ships a `.zip` |
| checksum status | **UNKNOWN** — no digest published, none measured |
| member mapping (which member is the FashionIQ model) | **UNKNOWN**, exactly as `models.yaml:830` says |

What I *can* state from source: the archive must contain at least one whole-model pickle, because that
is the only artifact `train.py` ever writes — `torch.save(current_model, os.path.join(args.model_dir,
'best_model.pt'))` (`train.py:271`), where `current_model` is the live `nn.Module` (assigned at
`train.py:217` / `:232` region from `model`), not a `state_dict`. Whether the zip also carries a
`params.json`, a `metrics_best.json`, or per-dataset subdirectories is UNKNOWN.

### 5.3 Whole-model pickle risk — the trust boundary

**Is the checkpoint `torch.load`ed as a whole module? YES — and the pinned code does it that way.**

- Write side: `torch.save(current_model, …)` (`train.py:271`) — the module object.
- Read side, the only loader in the repo: `utils.load_checkpoint`:
  ```
  136:    checkpoint = torch.load(checkpoint)
  137:    model.load_state_dict(checkpoint['state_dict'])
  ```
  (`utils.py:125-142`.) Note two facts in those three lines: (a) `torch.load` is called with **no**
  `map_location` and **no** `weights_only` argument — i.e. the default, which in the pinned era
  (torch 1.7.0) is equivalent to `weights_only=False`; and (b) the function then expects a
  `'state_dict'` key, which **`train.py:271` never writes**. OBSERVED: `grep -rn "load_checkpoint" *.py`
  → exactly one hit, the definition itself (`utils.py:125`). **`load_checkpoint` is dead code — nothing
  calls it**, and it does not match the format `train.py` produces. So the *only* live path back into
  a trained model is `torch.load(zip_member)` on a whole-model pickle, which requires
  `weights_only=False`.

**What this means for the trust boundary.** The workbench's documented threat model treats the pinned
upstream *source* as trusted and its *artifacts* as untrusted input pending digest verification. A
whole-module pickle breaks that model in a specific way: unpickling a `nn.Module` executes
`__reduce__` payloads, so a `torch.load(..., weights_only=False)` on the author's archive is
**arbitrary code execution at the privilege of the operator's interpreter** — it can happen before a
single line of the pinned evaluator runs, and it happens inside the process the workbench launches
with the model's verified interpreter. Concretely:

- `weights_only=True` **cannot** load this artifact family. A whole-module pickle stores class
  references (`model.Backbone`, `model.TokenLearner`, `clip.model.CLIP`, …) that PyTorch's restricted
  unpickler refuses; the safe loader is only usable for the `state_dict` form. So the mitigation
  available to the sibling CLIP4Cir entries — I verified `weights_only=True` accepts
  `fiq_comb_RN50x4_noft.pt` — is **not available for TG-CIR's own archive**, and the loader contract
  cannot be tightened to obtain it.
- Consequence: this model requires `weights_only=False` **and therefore cannot be made safe by
  deserialization policy alone**. The only defensible controls are (1) a verified digest of the exact
  archive bytes, recorded before first load, and (2) loading on an isolated host. Neither exists
  today: `expected_sha256: null`, `expected_size_bytes: null` (`models.yaml:839-840`), and no
  `download_manifest.json` entry for `tgcir` (OBSERVED: `workbench/artifacts/checkpoints/` holds
  `.gitkeep`, `download_manifest.json`, `limn/` — and the manifest's three records are all `limn`).
- There is no `UNSAFE_CHECKPOINT_SERIALIZATION` machinery in the workbench currently applied to this
  model: `grep -rn "UNSAFE_CHECKPOINT_SERIALIZATION" workbench/` → one hit, in a sibling
  reassessment's prose (`model_reassessment/encoder.md:258`), none in code.

**Classification: `UNSAFE_CHECKPOINT_SERIALIZATION` + `CHECKPOINT_URL_INACCESSIBLE`-adjacent only in
that the mapping is unverified; the URL itself resolved (`http=200`).** Severity P1 for the trust
boundary, raised to P0 for the operational sequence: the first time an operator loads this artifact
they must already have a recorded digest of it, and today they cannot obtain one from the registry.

### 5.4 Loader contract summary

| layer | requirement | source | status |
|---|---|---|---|
| artifact container | `TG-CIR.zip` (author Drive) | `README.md:15`; OBSERVED `<title>` | reachable (HTTP 200) |
| member(s) | unknown; ≥1 whole-model pickle, and possibly `params.json`/metrics | `train.py:271` is the only writer | UNKNOWN |
| deserialization | whole-module pickle ⇒ `torch.load` default (`weights_only=False`) | `train.py:271` + no other live loader | STATIC_AUDIT |
| dead alternative | `utils.load_checkpoint` expects `checkpoint['state_dict']`, which `train.py` never writes, and has no callers | `utils.py:125-142`; grep → 1 hit | STATIC_AUDIT |
| CLIP variant the pickle was trained with | `--img_encoder` default `ViT-B/16` (`train.py:56`); `Backbone` re-loads CLIP at construction (`model.py:41`) | both defaulted, never recorded in the checkpoint | STATIC_AUDIT |

---

## 6. AUXILIARY ASSETS

| name | purpose | source | availability class | generation possible? |
|---|---|---|---|---|
| OpenAI CLIP **ViT-B/16** weights | `clip.load('ViT-B/16')` at `model.py:41` and `train.py:88`; the pickle's `self.clip` is loaded from this cache at construction | OpenAI CLIP CDN (openai/CLIP), cache `~/.cache/clip` | **provenance unresolved** in the workbench: preflight reports `external runtime asset provenance unresolved: OpenAI CLIP ViT-B/16 weights/cache` (OBSERVED, §10). The registry has **no `auxiliary_assets.yaml` entry for `tgcir`** (OBSERVED: `grep -n "tgcir" workbench/registry/auxiliary_assets.yaml` → no match) | no — must be downloaded |
| FashionIQ `resized_image/` tree | the evaluator opens `<fashioniq_path>/resized_image/<category>/<id>.jpg` (`datasets.py:25`, `:100`, `:110`) | canonical root; layout `fashioniq_ilearn_resized_training` (`models.yaml:801`), extensions `[".jpg"]` (`:802`) | present locally with **exactly the right shape**: OBSERVED `data/fashionIQ_dataset/resized_image/{dress,shirt,toptee}` = 3817 / 6346 / 5373 `.jpg` files | no |
| `correction_dict/correction_dict_{dress,shirt,toptee}.json` | spell-corrects every relative caption before tokenization (`datasets.py:59`, `:131`, `:64`, `:136`) | **shipped in the pinned repo** (`correction_dict/`, 4 files incl. `_shoes`) | PINNED, no download, no digest needed — bytes are pinned by the commit | n/a |
| `fashion_iq_data.json`, `test_queries_*.pkl`, `test_targets_*.pkl` | **generated cache**: `datasets.FashionIQ.__init__` writes these on first use and reads them thereafter (`datasets.py:29-53`) | self-generated from captions/splits | writable side effect inside the dataset root | yes, but **NONDETERMINISTIC_PREPROCESSING-shaped**: the branch at `:29` means a stale cache silently overrides the annotations, and `load_obj` is a pickle read |

The cache branch is worth flagging: `if not os.path.exists(os.path.join(self.path,
'fashion_iq_data.json')):` (`datasets.py:29`) … `else:` (`:45`) … `load_obj(...)` (`:48-53`). A run
against a dataset root that already contains those files will use them instead of rebuilding from
`cap.*.val.json` / `split.*.val.json`, so the cohort is whatever that cache holds — invisible from the
invocation. Not a protocol change (the cache is derived from the same annotations if fresh), but it is
a provenance hazard a wrapper must control by refusing a dirty root. Severity P2.

---

## 7. FASHIONIQ PROTOCOL

Registry: `literature_split_label: val` (`models.yaml:805`), `native_protocol: fashioniq_val_split`
(`:806`), `supported_protocols: [fashioniq_val_split]` (`:807`).

**Protocol C, confirmed from the pinned source.** `fashioniq_val_split` is
`gallery="pair_union", reference_excluded=True` (`src/evaluation/fashioniq.py:178-180`).

**The gallery IS the ordered pair-union of candidate and target ids over the val triplets** — not the
split file. Quoted, `datasets.py:125-160` (the method that produces the cohort):

```
127:        with open(os.path.join(self.split_dir, "split.{}.{}.json".format(name, 'val')), 'r') as f:
128:            images = json.load(f)
...
134:        for idx in range(len(ref_captions)):
135:            caption = ref_captions[idx]
136:            mod_str = self.concat_text(caption['captions'], correction_dict)
137:            candidate = caption['candidate']
138:            target = caption['target']
139:            out = {}
140:            out['source_img_id'] = images.index(candidate)
141:            out['source_img_data'] = self.get_test_img(name + '_' + candidate)
142:            out['target_img_id'] = images.index(target)
...
148:        test_targets_id = []
149:        for i in test_queries:
150:            if i['source_img_id'] not in test_targets_id:
151:                test_targets_id.append(i['source_img_id'])
152:            if i['target_img_id'] not in test_targets_id:
153:                test_targets_id.append(i['target_img_id'])
154:        test_targets = []
155:        for i in test_targets_id:
156:            out = {}
157:            out['target_img_id'] = i
```

Read carefully: `images` (the split file) is used **only** to map names → positions
(`images.index(...)`, `:140`, `:142`); the **candidate universe is `test_targets_id`**, built at
`:148-153` by scanning the val triplets and appending each query's `source_img_id` and
`target_img_id` **once, in first-seen order**. There is no `split.{cat}.val.json` gallery in this path
at all. That is exactly `gallery="pair_union"`.

**Reference excluded: CONFIRMED, by score masking.** `test.py:63-67`:

```
63:    test_targets_id = []
64:    for i in test_targets:
65:        test_targets_id.append(i['target_img_id'])
66:    for i, t in enumerate(test_queries):
67:        sims[i, test_targets_id.index(t['source_img_id'])] = -10e10
```

and the ranking/recall block `test.py:70-80`:

```
70:    nn_result = [np.argsort(-sims[i, :])[:50] for i in range(sims.shape[0])]
74:    for k in [1, 10, 50]:
77:            if test_targets_id.index(test_queries[i]['target_img_id']) in nns[:k]:
79:        r = 100 * r / len(nn_result)
```

So the reference's own column is set to `-10e10` before `argsort`, which pushes it below every real
similarity and out of any top-k. The workbench models this cohort with an explicit column *removal*
(`recall_at_k_reference_excluded`, `src/evaluation/fashioniq.py:46-72`, which notes at `:47-52` that
masking and removal are equivalent "whenever all other scores exceed the mask" and pins the removal
behaviour instead). Both formulations give the same top-k hit test for finite similarities.

**Query construction.** `datasets.py:136`, `:64`: `concat_text` = `"{} and {}".format(correct_text(cap0),
correct_text(cap1))` (`:78-79`), where `correct_text` is a whitespace tokenization with a per-word
dictionary substitution (`:71-74`). The substitution is the author's spell-correction, and it is a
**different caption input** from the CLIP4Cir entries' `strip().capitalize()` composition — a genuine
protocol difference, not a formatting one. Any reproduction must use the pinned
`correction_dict/*.json`, not a re-derivation.

**Categories and order.** `['dress', 'shirt', 'toptee']` (`train.py:199`) — note this differs from the
CLIP4Cir order (`shirt, dress, toptee`); irrelevant to a macro, but it is the order the source uses.

**Metric.** R@1, R@10, R@50 per category (`test.py:74`), returned as `[('{name}_r{k}', r), …]`
(`:80`); the training loop selects best by `current_score = current_score + t[1][1]` — i.e. the **R@10**
entry (`train.py:202`, since `t[1]` is `dress_r10`). The reported pair is R@10/R@50 (`models.yaml:808-809`).
The macro is a mean over the three categories' cells as printed by the paper
(`papers.yaml:1585-1587` recomputes it and agrees).

**One protocol note that must not be papered over:** this model's numbers live in a different cohort
from the CLIP4Cir entries'. `fashioniq_val_split` (pair-union gallery, reference excluded) is not
interchangeable with `fashioniq_original_split` (full split, reference eligible); the workbench refuses
cross-protocol aggregates (`docs/FASHIONIQ_PROTOCOL_PARITY.md:20-24`) and
`resolve_fashioniq_protocol` raises on an unknown name rather than defaulting
(`src/evaluation/fashioniq.py:190-201`). **No gallery, reference policy, caption input or noise
condition is proposed for change anywhere in this report.**

---

## 8. WORKBENCH INTEGRATION

| component | state | evidence |
|---|---|---|
| adapter | `TGCIRAdapter` exists but has **no `command()` override** — fail-closed by inheritance | `adapters/models.py:114-116`, `:24-25` |
| command reliability | building it raises `NotImplementedError upstream evaluation command requires audit before execution` (OBSERVED) | measured |
| registry status | `COMMAND_BLOCKED_SOURCE_ENTRYPOINT` / `BLOCKED_COMMAND` / `PREFLIGHT_ONLY` — all three accurate | `models.yaml:812-813,786,789` |
| protocol | `native_protocol` = `supported_protocols` = `fashioniq_val_split`; source-confirmed pair-union + reference-excluded | `models.yaml:806-807`; `datasets.py:148-153`; `test.py:66-67` |
| dataset layout | `fashioniq_ilearn_resized_training` with `[".jpg"]`; local root satisfies it exactly (3817/6346/5373) | `models.yaml:801-802`; measured |
| auxiliary assets | **none declared for this model** — the root cause of the preflight provenance blocker | `grep tgcir auxiliary_assets.yaml` → no match |
| metric parser | **none** — `extract_aggregate_metrics` dispatches only `limn`, `csmcir`, the two CLIP4Cir ids and `sprc` (OBSERVED: the fall-through returns `AGGREGATE_PARSER_UNAVAILABLE`/`PARSER_UNAVAILABLE`) | `metrics_extraction.py:402-418` |
| benchmark reference | one committed row, `reference_status: VERIFIED`, `review_status: CONFIRMED`, `reported_metrics {r10: 51.32, r50: 73.09, mean: null}` + derived mean 62.205 | `papers.yaml:1520-1588` |
| per-query support | none; the evaluator returns only `(name, k, recall)` triples | `test.py:80` |
| checkpoint prep | `checkpoint_path` → `artifacts/checkpoints/tgcir/tgcir-fashioniq-unresolved.pt`; the directory does not exist and the real artifact is `TG-CIR.zip` | `registry.py:842-863`; measured |

RUNTIME_VERIFIED checks through the workbench's own read-only functions:

```
$ preflight_blockers(model, config)  -> 3 blockers (full list in §10)
$ ADAPTERS['tgcir'].command is not OfficialScriptAdapter.command  -> False
$ ADAPTERS['tgcir']().command(None, None, None)  -> NotImplementedError upstream evaluation command requires audit before execution
$ ls workbench/artifacts/checkpoints/  -> .gitkeep, download_manifest.json, limn/   (no tgcir)
$ ls data/fashionIQ_dataset/resized_image/{dress,shirt,toptee} | wc -l  -> 3817 / 6346 / 5373   (.jpg)
$ extract_aggregate_metrics('tgcir', <any stdout>, '', 0)  -> AGGREGATE_PARSER_UNAVAILABLE / PARSER_UNAVAILABLE
```

---

## 9. CONFIRMED DEFECTS

**CF-1 — P1 — the source has no evaluable entrypoint, and nothing in the workbench can build one.**
Root cause: evaluation exists only as `test.test(params, model, testset, category)` (`test.py:6`) called
from the training loop (`train.py:200`); there is no parser in `test.py`, no `__main__`, and no
checkpoint-replay path. Reproduction: the greps and the `NotImplementedError` in §3.1. Fix: none that
is honest without a wrapper — see §11. **Do not** add an adapter `command()` that points at
`train.py --num_epochs 0`; that re-runs training code to reach a metric and would report a number the
authors never published as a reproduction.

**CF-2 — P0 — the checkpoint is a whole-module pickle, so `weights_only=True` is impossible and the
first load is unguarded code execution.** Root cause: `torch.save(current_model, …)` (`train.py:271`)
writes the module object; the only matching read is a default `torch.load`. Reproduction: those two
lines plus the absence of any digest in the registry (`models.yaml:839-840`) and of any `tgcir` record
in `download_manifest.json` (OBSERVED). Fix: record the archive's size and SHA-256 **before** any load
and treat `UNSAFE_CHECKPOINT_SERIALIZATION` as a first-class state for this model; load it only on an
isolated host. Regression evidence: none — this is a policy/record fix, and the acceptance check is
that a digest exists and is checked at load time.

**CF-3 — P1 — the declared dependency pin names the wrong project and fails at model construction.**
Root cause: `README.md:19` pins `clip==0.2.0`, transcribed to `models.yaml:828`, while the code needs
OpenAI CLIP (`model.py:5,41`). Reproduction: §4.1. Fix: `git+https://github.com/openai/CLIP.git@<sha>`;
set `confidence` accordingly and keep the wrong pin visible in a comment, not as the value.

**CF-4 — P1 — the archive→member mapping is unknown and the artifact shape already disagrees with the
record.** Registry names a `.pt`; the author ships `TG-CIR.zip` (`<title>`, OBSERVED). `download_url:
null`, no size, no digest. Fix: download, list, record the FashionIQ member's name/size/digest, and
correct the placeholder filename. (Blocked for me by budget; UNKNOWN, not PASS.)

**CF-5 — P2 — no metric parser.** `metrics_extraction.py:402-418` has no `tgcir` branch, so even a
successful run would return `AGGREGATE_PARSER_UNAVAILABLE`. The output shape is trivial
(`test.py:80` returns `[('{name}_r{k}', r), …]` for k ∈ {1,10,50}, three categories), and the workbench
already has two precedent parsers. Fix: add one, with the macro recomputed from the printed
per-category cells as an internal-consistency check, exactly as `parse_combiner_output` does.

**CF-6 — P2 — the dataset cache branch can silently substitute a stale cohort.** `datasets.py:29-53`:
if `fashion_iq_data.json` exists, the annotations are never re-read and `test_queries_*.pkl` /
`test_targets_*.pkl` are unpickled instead. Fix: a wrapper must refuse a dirty dataset root, or delete
the cache before a run; neither is enforced today.

**CF-7 — P2 — CUDA is hard-required and declared, but nothing has probed it.** `test.py:31,47`,
`model.py:32,41` call `.cuda()` unconditionally; `models.yaml:824` records `cuda: '11.0'`, so
`_requires_gpu` (`runtime.py:918-928`) returns True and `require_launch_ready` will refuse a CPU host —
which is correct. The gap is that no GPU capability record exists for this model (OBSERVED:
`workbench/artifacts/gpu_capabilities/` has no `tgcir` entry), so the refusal is the current, expected
state rather than a defect. Recorded so it is not mistaken for readiness.

---

## 10. REMAINING BLOCKERS

OBSERVED preflight output for this model (`preflight_blockers`), three blockers:

```
1. external runtime asset provenance unresolved: OpenAI CLIP ViT-B/16 weights/cache
2. command not runnable: COMMAND_BLOCKED_SOURCE_ENTRYPOINT
3. execution environment not verified: WORKBENCH_TGCIR_PYTHON=None [UNCONFIGURED/UNCONFIGURED]
   (WORKBENCH_TGCIR_PYTHON not set)
```

| # | classification | owner | evidence | remediation |
|---|---|---|---|---|
| B1 | `UPSTREAM_CLI_MISSING` | upstream | `test.py:6` is a callable; no parser, no `__main__` in `test.py`; only `train.py:40-80` has one | upstream must publish an eval entrypoint; workbench must not fake one |
| B2 | `FIRST_PARTY_ADAPTER_MISSING` | workbench | `TGCIRAdapter` has no `command()`; building it raises `NotImplementedError` (OBSERVED); preflight blocker 2 | decide deliberately: either accept `PREFLIGHT_ONLY` permanently, or add a wrapper whose provenance is labelled a *variant*, never "official parity" |
| B3 | `FIRST_PARTY_ENVIRONMENT_CONFIGURATION` | workbench | blocker 3; `workbench/envs/` has no TGCIR env; `environment_assignment.json` assigns `limn` only | build the declared env (python UNKNOWN, torch 1.7.0, torchvision 0.8.0, cuda 11.0) |
| B4 | `INSTALLABLE_DEPENDENCY_MISSING` | workbench | `clip==0.2.0` (`models.yaml:828`) is the wrong project; the code needs OpenAI CLIP (`model.py:5,41`) | pin openai/CLIP at a commit sha |
| B5 | `AUXILIARY_ASSET_MISSING` | workbench | blocker 1; **no `auxiliary_assets.yaml` entry exists for `tgcir`** | declare the ViT-B/16 cache asset with URL + digest + `cache_root_fixed: ~/.cache/clip` (the pattern already used at `auxiliary_assets.yaml:162-183`) |
| B6 | `PUBLIC_CHECKPOINT_NOT_MAPPED` | workbench | `download_url: null`, no size, no digest (`models.yaml:833-840`), real artifact is `TG-CIR.zip` (OBSERVED) | download, list, record member name/size/digest |
| B7 | `UNSAFE_CHECKPOINT_SERIALIZATION` | workbench + operator | whole-module pickle (`train.py:271`); `weights_only=True` structurally unavailable; no digest recorded | record the digest before the first load; load only on an isolated host |
| B8 | `GPU_RUNTIME_UNTESTED` | operator | `test.py:31,47`; `models.yaml:824`; no `gpu_capabilities` record | probe the GPU on the target host |
| B9 | `SCIENTIFIC_PROTOCOL_UNVERIFIED` | workbench | Protocol C is source-confirmed, but the *reproduction* of 51.32/73.09 is UNKNOWN because no run is possible | keep `reported_mean: null` + derived 62.205; do not report a wrapper-derived number as the paper's |
| B10 | `NONDETERMINISTIC_PREPROCESSING` | workbench | cache branch `datasets.py:29-53` | refuse a dirty dataset root in the wrapper |

No `SOURCE_UNAVAILABLE`, no `FIRST_PARTY_BUG`, no `UPSTREAM_IMPORT_BUG` (every import resolves within
the pinned tree + the declared distributions), no `DEPENDENCY_VERSION_CONFLICT` in the strict sense
(the required module is installable; the pin merely names the wrong project, recorded as B4), no
`LEGACY_RUNTIME_INCOMPATIBILITY` established (I did not execute torch 1.7.0 or OpenAI CLIP against
this code, so any claim about the pickle's compatibility with a modern stack would be UNKNOWN).

---

## 11. IMPLEMENTATION PLAN

The honest shape of this model's fix is short, and the largest item is a decision rather than code.

1. **Record the artifact** (workbench/operator, hours): download `TG-CIR.zip`, list it, compute the
   FashionIQ member's SHA-256, write it into `checkpoint_variants` (filename, `expected_size_bytes`,
   `expected_sha256`) and a `download_manifest.json` record. Complexity: low. Risk: none. This is a
   prerequisite for touching the pickle at all (CF-2/CF-4/B6/B7).
2. **Declare the CLIP ViT-B/16 auxiliary asset** (workbench, minutes): mirror the
   `auxiliary_assets.yaml:162-183` pattern for `tgcir`. Closes preflight blocker 1 (B5).
3. **Fix the dependency pin** (workbench, minutes): openai/CLIP at a commit sha (B4/CF-3).
4. **The wrapper decision** (workbench + reviewer, a day if taken): a wrapper is *possible* —
   `test.test` needs only `params.{local_rank,batch_size}`, a model, the `FashionIQ` dataset object and
   a category, and `test.py:80` returns parseable `(name, k, recall)` triples. But three things make it
   a **variant, not a reproduction**: the entrypoint is not the authors', `train.py`'s own invocation
   would be required to reach evaluation, and the cohort depends on the dataset cache (CF-6). If a
   wrapper is taken, it MUST be labelled a variant in the run record and MUST NOT be presented as
   official parity. The alternative — keep `PREFLIGHT_ONLY` permanently and say so — is defensible and
   costs nothing.
5. **Metric parser** (workbench, small): add a `tgcir` branch to `metrics_extraction.py` with macro
   self-consistency, mirroring `parse_combiner_output`. Only useful if (4) happens.
6. Protocol, gallery, reference policy, caption composition and noise condition: **unchanged**.

Expected outputs after 1–3 and a GPU host: preflight for this model drops to the environment and
entrypoint blockers only; the artifact becomes digest-checkable and safely loadable on an isolated
host; the paper comparison remains R@10 51.32 / R@50 73.09 against `tgcir_fashioniq_val_table1`
(`reference_status: VERIFIED`).

---

## 12. FINAL VERDICT

**BLOCKED.**

Static facts that hold: pin exact and clean; the source is complete and self-consistent; the protocol
declaration `fashioniq_val_split` is **confirmed from the gallery construction itself** (`datasets.py:148-153`)
and from the reference masking (`test.py:66-67`); the committed paper row is verified and its macro is
arithmetically exact; the dataset layout the evaluator wants is present locally with the right shape;
the `clip` pin defect is confirmed with the quote; the correction dictionaries are confirmed
byte-identical to LIMN's with digests.

It is blocked because: the source has **no evaluable entrypoint at all** (`test.py` is a callable with
no parser and no `__main__`), the artifact is a **whole-module pickle** that cannot be loaded with the
safe loader and has no recorded digest, the declared `clip==0.2.0` pin names the wrong project, no
CLIP ViT-B/16 asset is declared for this model, no interpreter exists, and the archive's member
mapping is UNKNOWN.

`VERIFIED` and `VARIANT` would both be unsupported (nothing ran, no variant was built). `NOT_RUN`
would understate the static work. BLOCKED is the precise state.

---

## 13. EVIDENCE

Repo `https://github.com/iLearn-Lab/MM23-TG-CIR` · pinned `65fa78eaf8cabe8197fcc22ccab42207c591bdee`
· `git status --porcelain` empty. SNAPSHOT 2: equal to the pin — `ls-remote origin HEAD` and
`refs/heads/main` both return `65fa78eaf8cabe8197fcc22ccab42207c591bdee`, `fetch origin --dry-run`
empty, so the pin is unchanged and upstream has not moved (§2).

Files opened (pinned unless noted): `README.md`; `test.py`; `train.py`; `model.py`; `datasets.py`;
`utils.py`; `cirr_test_submission.py` (imports only); `correction_dict/*.json` (digests only);
workbench: `registry/models.yaml:785-842`, `registry/paper_benchmarks/papers.yaml:1520-1588`,
`registry/auxiliary_assets.yaml` (searched, no `tgcir`), `backend/adapters/models.py`,
`backend/preflight.py`, `backend/registry.py`, `backend/metrics_extraction.py`, `backend/runtime.py`,
`src/evaluation/fashioniq.py`; sibling digests: `workbench/third_party/LIMN/correct_files/*`.

Commands and observed results (read-only; all artifacts under `/tmp`):

1. `git -C workbench/third_party/TGCIR rev-parse HEAD` → `65fa78eaf8cabe8197fcc22ccab42207c591bdee`; `status --porcelain` → empty.
2. `ls -R` / `find` → 8 files, `correction_dict/` with 4 JSON; `grep -n "__main__|ArgumentParser|argparse|sys.argv" test.py` → no hits; `grep -n "__main__|ArgumentParser" cirr_test_submission.py` → `94: if __name__ == '__main__':` (the CIRR path HAS one, the FashionIQ path does not).
3. `grep -n "def \|cuda\|clip" test.py train.py model.py` → `test.py:6 def test(...)`, `:31/:47 .cuda()`, `:66-67 -10e10`; `model.py:5 import clip`, `:41 clip.load(...)`, `:32 .cuda()`; `train.py:88 clip.load(...)`, `:271 torch.save(current_model, …)`.
4. `grep -rn "load_checkpoint" *.py` → 1 hit, the definition (`utils.py:125`) — dead code; `utils.py:136` `torch.load(checkpoint)` with `:137 checkpoint['state_dict']` matches nothing `train.py` writes.
5. MD5 and SHA-256 of the four correction dicts vs `LIMN/correct_files/` → **identical** on all four, e.g. `correction_dict_toptee.json` = `md5 6a4ebcd57670035e61474a9b928c4878`, `sha256 3dfbf15fa6d3a30d…` on both sides.
6. `preflight_blockers(model, config)` → 3 blockers (§10).
7. `ADAPTERS['tgcir'].command is not OfficialScriptAdapter.command` → `False`; calling it → `NotImplementedError upstream evaluation command requires audit before execution`.
8. `ls workbench/artifacts/checkpoints/` → `.gitkeep`, `download_manifest.json`, `limn/` (manifest records three `limn` files only).
9. `ls data/fashionIQ_dataset/resized_image/{dress,shirt,toptee} | wc -l` → 3817 / 6346 / 5373 `.jpg`.
10. `curl -sL "…/file/d/1OdZTtJqy-RTpYXBaq5ThmH3IBnGvYCyi/view"` → `http=200`, `<title>TG-CIR.zip - Google Drive</title>`.
11. `grep -n "tgcir" workbench/registry/auxiliary_assets.yaml` → no match.

---

## 14. ACHIEVED VERIFICATION LEVEL

**LEVEL 2 of 6.**

- Level 1 — identity, pin, registry consistency: achieved (STATIC_AUDIT; checkout clean and equal to
  the recorded pin).
- Level 2 — static source + protocol auditing with independent evidence: achieved. The protocol is
  confirmed from the gallery construction (`datasets.py:148-153`) and the reference mask
  (`test.py:66-67`); the entrypoint absence is confirmed by grep and by the adapter's measured
  refusal; the `clip` pin defect is confirmed by quote; the correction dictionaries are confirmed by
  digest; the paper row is re-derived arithmetically.
- Level 3 is NOT reached: the checkpoint archive was not downloaded, so its member list, size and
  digest are UNKNOWN, and no state was loaded (and, per CF-2, a `weights_only=True` load is
  structurally impossible for this artifact family — Level 3's probe would have to be performed with
  the unsafe loader on an isolated host, which is exactly the control this model still lacks).
- Level 4 (a real run producing Recall@10/50) is NOT reached: no entrypoint exists to run, no
  interpreter, no declared CLIP asset, no GPU probe.
- Levels 5–6 are out of reach.

Nothing here claims a metric was reproduced. No gallery, reference policy, caption input or noise
condition was proposed for change. Still marked UNKNOWN: the archive member list, its size and its
digest (§5.2), and — as a direct consequence — whether the registry's declared filename matches
anything inside the archive. SNAPSHOT 2 is now established (equal to the pin, §2).
