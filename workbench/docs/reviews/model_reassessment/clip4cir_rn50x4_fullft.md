# clip4cir_rn50x4_fullft — reproducibility reassessment (independent, adversarial, READ-ONLY)

Auditor: independent reassessment pass. Nothing under `/home/heheboiz/data/cir` was modified except
this file. All probe output and downloads went to `/tmp`. No GPU work, no evaluator execution, no
metric produced by a run. Every claim is `STATIC_AUDIT` (a file I opened, or a command I ran and its
observed output) unless explicitly labelled `RUNTIME_VERIFIED`, which here means "bytes I fetched and
decoded on this host", never "the evaluator ran".

This model SHARES its repository with `combiner_rn50x4_noft` (same `source_dir`, same pin
`dfed9f748a8a4d05abf613164e06d59b22dbdf13`) but is a DIFFERENT variant: different checkpoint contract,
different CLI (it passes `--clip-model-path`), different paper reference. It is audited independently;
nothing is inherited from the sibling report except where I cite it explicitly for a shared fact.

Prior workbench audits (`workbench/docs/reviews/models/clip4cir_rn50x4_fullft.md`) were treated as
HYPOTHESES and re-tested. Two of their findings are REJECTED below (§5.2, §5.4) and two are confirmed
with the mechanism identified.

---

## 1. MODEL IDENTITY

| field | value | evidence |
|---|---|---|
| model_id | `clip4cir_rn50x4_fullft` | `workbench/registry/models.yaml:721` |
| method | "CLIP4Cir RN50x4 fullft" | `models.yaml:733` |
| paper | "Composed Image Retrieval using Contrastive Learning and Task-oriented CLIP-based Features" | `models.yaml:737` |
| paper URL / repo | https://arxiv.org/abs/2308.11485 / https://github.com/ABaldrati/CLIP4Cir | `models.yaml:738-739` |
| year / generation | 2023 / legacy | `models.yaml:734-735` |
| pinned commit | `dfed9f748a8a4d05abf613164e06d59b22dbdf13` | `models.yaml:741` |
| source_dir | `CLIP4Cir` | `models.yaml:743` |
| native protocol | `fashioniq_original_split` | `models.yaml:748` |
| reported metrics | R@10 43.78, R@50 66.93, mean 55.36 (derived) | `models.yaml:754-756` |
| reproduction_status | `BLOCKED_CHECKPOINT_BUNDLE` | `models.yaml:757` |
| checkpoint variant | `rn50x4_fullft_unresolved` / `bundle_placeholder` / `required_files: [combiner_state.pt, clip_state.pt]` | `models.yaml:768-784` |

STATIC_AUDIT. The paper's own row is the paper's headline RN-50x4 configuration: arXiv v1 §4.3 "we
achieve the best results on both datasets when we fine-tune both encoders", Table 9
`Ours RN-50x4 Transformer 44.41 65.26 39.46 64.55 47.48 70.98 43.78 66.93` (verified in §7.3).

---

## 2. SOURCE STATUS

Pin and cleanliness: identical to the sibling and verified the same way. OBSERVED:

```
$ git -C workbench/third_party/CLIP4Cir rev-parse HEAD
dfed9f748a8a4d05abf613164e06d59b22dbdf13
$ git -C workbench/third_party/CLIP4Cir status --porcelain   -> (empty)
$ .venv/bin/python -c "…source_clean_and_pinned(model, source)…"
clip4cir_rn50x4_fullft (True, 'dfed9f748a8a4d05abf613164e06d59b22dbdf13', None)
```

(`workbench/backend/runtime.py:2403-2427`.) The pinned tree is complete: 7 files under `src/`, all
imports of the evaluation path resolvable (`src/validate.py:1-18`).

### SNAPSHOT 2 — current upstream state

```
$ git -C workbench/third_party/CLIP4Cir ls-remote origin HEAD
56bb547909d57bd8b5d2ef7ba55149962f77d482	HEAD
$ git -C workbench/third_party/CLIP4Cir fetch origin --dry-run
   dfed9f7..56bb547  master     -> origin/master
 * [new tag]         weights    -> weights
```

One commit ahead (`Update README`, 2026-10-12T19:50:59Z, `README.md` +4 −4 only; API compare:
`ahead_by 1`, one file). **The pin was not changed.** The README now points at a GitHub release
instead of Google Drive, and the author published a release `weights` (published 2026-10-12T19:50:01Z,
`target_commitish: master`) whose five `pretrained_models.tar.gz.part-*` assets each carry a
GitHub-published SHA-256 (full table in the sibling report §2; identical facts, not re-derived here
except where this model needs them, §5.2).

This is decisive for this variant specifically: **the pinned README names one generic folder and no
per-variant file**, whereas the new release ships a per-variant directory tree and states the pairing
rule outright: "Use CLIP and Combiner checkpoints from the **same dataset and configuration**. The
`noft` configurations include only a Combiner checkpoint and use the corresponding original CLIP
backbone without fine-tuning."

---

## 3. NATIVE EVALUATION WORKFLOW

Entrypoint `src/validate.py`; adapter `CLIP4CirFullftAdapter` (`adapters/models.py:97-111`),
`script = "src/validate.py"` `:99`.

Function chain (STATIC_AUDIT):

1. `main()` `src/validate.py:263`; argparse `:264-277`.
2. `clip_model, clip_preprocess = clip.load(args.clip_model_name, device=device, jit=False)` `:279` —
   **base RN50x4 is constructed first, always**, even when fine-tuned weights are supplied.
   `input_dim = clip_model.visual.input_resolution` `:280`; `feature_dim = clip_model.visual.output_dim` `:281`.
3. `if args.clip_model_path:` `:283` → `saved_state_dict = torch.load(args.clip_model_path,
   map_location=device)` `:285` → `clip_model.load_state_dict(saved_state_dict["CLIP"])` `:286` →
   prints `'CLIP model loaded successfully'` `:287`. **LOADED FOR THIS VARIANT.**
4. `targetpad_transform(args.target_ratio, input_dim)` `:291`.
5. `elif args.combining_function.lower() == 'combiner':` `:304` → `Combiner(feature_dim,
   args.projection_dim, args.hidden_dim)` `:305` → `state_dict = torch.load(args.combiner_path,
   map_location=device)` `:306` → `combiner.load_state_dict(state_dict["Combiner"])` `:307` →
   `combiner.eval()` `:308`.
6. `elif args.dataset.lower() == 'fashioniq':` `:325` → per category `shirt`/`dress`/`toptee`
   `:329/:334/:339` → `fashioniq_val_retrieval` `:113` → gallery `FashionIQDataset(…,'classic',…)`
   `:127` + `extract_index_features` `:128` (this is the step that consumes the **fine-tuned image
   encoder**) → `FashionIQDataset(…,'relative',…)` `:129` → `compute_fiq_val_metrics` `:131` `:21-57`.
7. `print(f"average recall10 = {mean(…)}")` `:353-354`.

CLI present: YES (`main()`, `README.md:256-269`). What a wrapper must call (exactly what the adapter
builds, `adapters/models.py:107-111`):

```
$WORKBENCH_CLIP4CIR_PYTHON <checkout>/src/validate.py \
  --dataset fashionIQ --combining-function combiner \
  --combiner-path <bundle>/combiner_state.pt --clip-model-path <bundle>/clip_state.pt \
  --projection-dim 2560 --hidden-dim 5120 --clip-model-name RN50x4 \
  --target-ratio 1.25 --transform targetpad
```

Missing to run as-is: both bundle members, the base RN50x4 cache, the `<checkout>/fashionIQ_dataset`
link, `.jpg` images, and a verified interpreter (§10).

---

## 4. DEPENDENCIES

Identical module set to the sibling — same file, same path, same interpreter — so the table is not
duplicated in full. The load-bearing rows for this variant:

| import | file:line | owner | required version | installable? | py/torch/cuda | safe remedy | test result |
|---|---|---|---|---|---|---|---|
| `torch` (+ `torch.load` of the CLIP overlay) | `src/validate.py:10,285` | upstream | registry declares 1.11.0 | yes | py3.8; CPU-capable | build the declared env | not executed |
| `torchvision` | `src/data_utils.py:7-9` | upstream | 0.12.0 | yes | py3.8 | — | not executed |
| `clip` / `clip.model.CLIP` | `src/validate.py:8,12` | upstream README `README.md:118` | **unpinned git install** | yes | needs `torch`, `torchvision`, `ftfy`, `regex`, `tqdm`, `Pillow` | pin a commit sha | not executed |
| `numpy` | `src/validate.py:9` | upstream (undocumented) | not declared | yes | — | add to env spec | not executed |
| `tqdm` | `src/validate.py:14`; `src/utils.py:11` | upstream (undocumented) | not declared | yes | — | add to env spec | not executed |
| `PIL`/`PIL.Image` | `src/data_utils.py:5-6` | upstream (undocumented) | not declared | yes | — | add to env spec | not executed |
| `pandas`, `comet_ml` | `src/combiner_train.py:11,1` | training only, not this path | 1.4.2 / 3.21.0 | yes | — | none needed | n/a |

The `clip` package is the load-bearing dependency for this variant twice over: it defines the
`CLIP` class whose `state_dict` the overlay targets, and it owns the `RN50x4` architecture the
overlay must fit. An unpinned git install means the class definition at evaluation time is whatever
`main` happened to ship — a real reproducibility hole, upstream's own (`README.md:118`).

**D1–D8 are not defined anywhere in this repository** (grepped: `workbench/`, all `*.py`/`*.md`; the
only `D1` hits are unrelated finding ids). Two sibling reassessments record the same gap
(`model_reassessment/sprc.md:169`, `model_reassessment/ptha_mtst.md:299`). I therefore do not invent
the mapping: the only dependency-shaped issue here is the undocumented-but-installable
`numpy`/`tqdm`/`Pillow` set (`INSTALLABLE_DEPENDENCY_MISSING` in form, documentation gap in fact, P2)
and the unpinned `clip` (P2). Nothing missing is uninstallable.

OBSERVED: no interpreter. `WORKBENCH_CLIP4CIR_PYTHON` unset; `workbench/envs/` holds only
`README.md` + `csmcir-import.txt`; `workbench/artifacts/environment_assignment.json` assigns `limn`
only. `python_executable(model)` → `RuntimeError execution environment not verified
[UNCONFIGURED/UNCONFIGURED]: WORKBENCH_CLIP4CIR_PYTHON=None` (`runtime.py:1781-1799`).

---

## 5. CHECKPOINTS

### 5.1 The two-state bundle: what the two states are

Registry (`models.yaml:768-784`), verbatim:

```
768:    checkpoint_variants:
769:      - checkpoint_id: rn50x4_fullft_unresolved
770:        filename: rn50x4-fullft-unresolved.pt
771:        artifact_type: bundle_placeholder
772:        required_files: [combiner_state.pt, clip_state.pt]
773:        download_url: null
774:        source_url: https://drive.google.com/drive/folders/1ny2hhzP8HZBnXhjvDEni8P8G4_inCNTv?usp=sharing
779:        checkpoint_training_noise_pct: 0
780:        evaluation_noise_pct: 0
781:        expected_size_bytes: null
782:        expected_sha256: null
783:        status: CHECKPOINT_MAPPING_UNVERIFIED
784:        checkpoint_mapping_status: UNVERIFIED
```

**State 1 — the fine-tuned CLIP encoders.** Saved by stage 1. `src/clip_fine_tune.py:226` /
`:228` (FashionIQ) and `:408/:411/:414/:416` (CIRR) call
`save_model('tuned_clip_best'|'tuned_clip_<epoch>'|…, epoch, clip_model, training_path)`, and
`src/utils.py:117-131` writes:

```
128:    torch.save({
129:        'epoch': cur_epoch,
130:        model_name: model_to_save.state_dict(),
131:    }, str(models_path / f'{name}.pt'))
```

with `model_name = model_to_save.__class__.__name__` (`:127`). For `clip.model.CLIP` that key is
`'CLIP'`. It contains BOTH encoders' weights: the pinned README says the stage-1 recipe is
"we update the weights of both CLIP encoders" (`README.md:71-73`), and the paper's Table 9 row is the
`IFT=✓, TFT=✓` configuration (§7.3).

**State 2 — the trained Combiner.** Saved by stage 2. `src/combiner_train.py:225` / `:227` and
`:411/:414/:417/:419` call `save_model('combiner'|'combiner_<epoch>'|…, epoch, combiner, …)`; via the
same `src/utils.py:127-131` the key is `'Combiner'` (class `Combiner`, `src/combiner.py:6`).

**Which does the evaluation path load? BOTH, always, in this order.**

```
279:    clip_model, clip_preprocess = clip.load(args.clip_model_name, device=device, jit=False)
283:    if args.clip_model_path:
285:        saved_state_dict = torch.load(args.clip_model_path, map_location=device)
286:        clip_model.load_state_dict(saved_state_dict["CLIP"])
304:    elif args.combining_function.lower() == 'combiner':
305:        combiner = Combiner(feature_dim, args.projection_dim, args.hidden_dim).to(device, non_blocking=True)
306:        state_dict = torch.load(args.combiner_path, map_location=device)
307:        combiner.load_state_dict(state_dict["Combiner"])
```

Plus a third artifact that the registry's two-member bundle does not name: the **base** RN50x4
checkpoint, downloaded by `clip.load` at `:279` before the overlay is applied. So the runtime contract
is three artifacts: base RN50x4 (a cache asset, declared separately at `auxiliary_assets.yaml:184-203`)
+ `…_clip_….pt` + `…_comb_….pt`.

**Is the second (CLIP) state required to reproduce the published numbers? YES, and it is load-bearing
for correctness, not just fidelity.** Evidence:
- `src/validate.py:283-287` — it is the only place the fine-tuned encoders enter the process, and
  `:128` extracts the gallery features from `clip_model` AFTER that overlay.
- If it is omitted, the script does not fail: it silently evaluates the **noft** configuration, i.e. it
  becomes the sibling model's experiment. Since the offset is one CLI flag, a wrapper that forgets it
  produces a plausible wrong number. This is the single most important correctness hazard in the
  variant.
- The published row is the fine-tuned configuration by its own text: the paper's §4.3 says the best
  results come from fine-tuning both encoders, and the author's own release README requires
  "Use CLIP and Combiner checkpoints from the **same dataset and configuration**" and states the `noft`
  configurations "include only a Combiner checkpoint".

**The registry's two-artefact claim is therefore CORRECT but incomplete**: `combiner_state.pt` +
`clip_state.pt` are genuinely both needed, and the base RN50x4 weights are a third requirement it
delegates to the auxiliary-asset registry.

### 5.2 The available bundle is richer than the registry records — but the registry's *shape* choice is defensible

The prior audit asserted (`docs/reviews/models/clip4cir_rn50x4_fullft.md:232`) that "an unauthenticated
fetch of the folder HTML yields **zero** `.pt` occurrences … so the individual file listing needs
JS/auth and the archive→variant mapping remains unproven." **REJECTED.** The listing is readable
without JS, via the `embeddedfolderview` endpoint. OBSERVED (this pass, unauthenticated):

```
$ curl -sL -A "Mozilla/5.0" "https://drive.google.com/embeddedfolderview?id=1fnGeZQA4XMfEERo-qDCosJXILl1UdQ_l#list"
   title "FashionIQ"; entries RN50_fullft, RN50_imageft, RN50_noft, RN50_textft, RN50x4_fullft, RN50x4_noft
$ curl -sL -A "Mozilla/5.0" "https://drive.google.com/embeddedfolderview?id=1BPE33_XSm33Min0OaGW2Sl9rcddZ-WF-#list"
   title "RN50x4_fullft"; entries:
     fiq_clip_RN50x4_fullft.pt  -> id 1cQS8vFhNBpZOhyDNe15STz00wMvYEQEb   (11/3/22)
     fiq_comb_RN50x4_fullft.pt  -> id 1MxFkozS7RJy5VOoDBC2gJkoYqB4jD5tB   (11/3/22)
```

So the variant's Drive folder holds exactly two files — **one `_clip_` state and one `_comb_`
state** — matching the registry's `required_files: [combiner_state.pt, clip_state.pt]` one-for-one and
matching the release README's rule "Filenames containing `_clip_` hold the fine-tuned CLIP weights,
while `_comb_` files hold the trained Combiner weights." The names in the registry are placeholders,
not upstream filenames (the sibling's `rn50x4-noft-unresolved.pt` has the same property).

Also observed: the pinned README's Drive folder contains no file listing at all in its HTML, which is
why the earlier audit saw nothing — the landing page is JS-rendered; the *subfolder* endpoints are not.
Severity P1 for the stale prose, not for the model's verdict.

### 5.3 The pairing question: what happens on a mismatch

Three questions, three different answers, and only the third is dangerous:

**(a) Is a matching backbone required? YES — and the mismatch is loud.** The Combiner's input width is
the CLIP feature dim: `Combiner(clip_feature_dim, projection_dim, hidden_dim)` (`src/combiner.py:11`)
with `text_projection_layer: Linear(clip_feature_dim, projection_dim)` `:18` and
`output_layer: Linear(hidden_dim, clip_feature_dim)` `:25`. The RN-50 family's `d = 1024` and the
RN-50x4 family's `d = 640` (the CLIP paper's own table, and the author's Drive README: "in the
RN50-based pretrained models is equal to 4096, in the RN50x4-based is equal to 2560" — 4d in both
cases). RUNTIME_VERIFIED probe against the released noft state (scratch, pinned class, no repo writes):

```
$ .venv/bin/python /tmp/probe_mismatch.py
RN50-recipe dims -> RuntimeError Error(s) in loading state_dict for Combiner:
RN50x4 dims -> <All keys matched successfully>
```

So a cross-backbone pairing fails at `load_state_dict` with a shape error. Safe.

**(b) Does the loader demand a matching PAIR (clip state vs combiner state)? NO.** `src/validate.py`
loads the two files independently (`:285-286` for `"CLIP"`, `:306-307` for `"Combiner"`) with two
independent `if`/`elif` branches. **There is no cross-check of any kind** — no shared metadata, no
manifest, no digest, no dimension reconciliation between the two files, and no assertion that they
came from the same training run. `torch.load` yields whatever the file holds; a `'CLIP'` key from the
CIRR experiment and a `'Combiner'` key from FashionIQ would load together without a warning.

**(c) Does anything in the file name or contents identify the variant? NO.**

```
$ .venv/bin/python -c "…torch.load('fiq_comb_RN50x4_noft.pt', weights_only=True)…"
top-level keys: ['epoch', 'Combiner']     # 'epoch' is an int; no dataset tag, no variant tag, no config
```

There is **no** dataset/category/variant metadata inside the released members — verified for the noft
member (RUNTIME_VERIFIED, sibling report §5.3). Combined with §5.1, this produces the real defect:
because `clip4cir_rn50x4_fullft` and `combiner_rn50x4_noft` share one architecture
(`d=640, proj=2560, hid=5120`), **`fiq_comb_RN50x4_noft.pt` satisfies the fullft loader with "All keys
matched successfully"** (measured). The only thing that prevents a wrong combination is the pair of
file *names*, and the registry discards them in favour of the placeholders
`combiner_state.pt` / `clip_state.pt` (`models.yaml:772`). So the bundle's correctness currently rests
on an operator transcribing the names correctly with nothing verifying the transcription. That is
`checkpoint_mismatch`-class risk, not a theoretical one.

**Ownership:** the missing cross-check is the pinned source's design (upstream), but the workbench's
job is to bind the *published names and digests* to the slots it invented — and it does not, because
`expected_sha256` is `null` for both members (`models.yaml:781-782`; the members inherit the anchor's
nulls, see §5.4). Fix by recording per-member digests, not by changing the source.

### 5.4 The registry/adapter path divergence — CONFIRMED, with the mechanism

The prior audit called this a P1 and proposed two remedies. Confirmed here, root-caused, and shown to
be narrower than "the registry and adapter disagree": one shared function already resolves members
correctly, and **the adapter does not use it**.

`checkpoint_path` (`workbench/backend/registry.py:842-863`) resolves a bundle-shaped artifact to
`<root>/<model_id>/<filename>` — i.e. literally `…/clip4cir_rn50x4_fullft/rn50x4-fullft-unresolved.pt`
— and `checkpoint_missing_paths` (`:365-416`) then treats *that* as the member directory, looking for
`required_files` **inside** it (`:371-373`, `:377-378`). The adapter instead does
`combiner_path = checkpoint / "combiner_state.pt"` (`adapters/models.py:105`) where `checkpoint` IS
`…/rn50x4-fullft-unresolved.pt` (`OfficialScriptAdapter.build_command` → `checkpoint_path(...)`,
`adapters/models.py:18-22`), yielding `…/rn50x4-fullft-unresolved.pt/combiner_state.pt` at build time
but reading the members from `…/checkpoint_file.parent` per the comment at `:102-104`. The code's own
comment claims agreement with `checkpoint_missing_paths`; measured, it does not agree — the two
searches are off by one directory.

RUNTIME_VERIFIED, using the workbench's own pure functions and a staged stub tree in `/tmp`:

```
$ .venv/bin/python - (registry.py helpers only)
checkpoint_path -> /home/heheboiz/data/cir/workbench/artifacts/checkpoints/clip4cir_rn50x4_fullft/rn50x4-fullft-unresolved.pt
members (registry view): ['combiner_state.pt', 'clip_state.pt']
member location registry requires: [.../rn50x4-fullft-unresolved.pt/combiner_state.pt, .../rn50x4-fullft-unresolved.pt/clip_state.pt]
member location adapter requires: [.../clip4cir_rn50x4_fullft/combiner_state.pt,            .../clip4cir_rn50x4_fullft/clip_state.pt]
agree? False

# staged tree A: members INSIDE the .pt directory  (what checkpoint_missing_paths accepts)
registry gate (members inside .pt dir): []
adapter dir (members as siblings):      [PosixPath('…/clip4cir_rn50x4_fullft/combiner_state.pt'),
                                         PosixPath('…/clip4cir_rn50x4_fullft/clip_state.pt')]
```

So a bundle staged to satisfy the registry gate fails the adapter, and vice versa. **But the fix is
smaller than the prior audit thought**: `checkpoint_member_root` (`registry.py:866-891`) already
returns the member directory from the DECLARATION (`:883-888`, `bundle_placeholder` ∈
`is_multi_member`), and it returns `…/rn50x4-fullft-unresolved.pt` — the registry's answer, not the
adapter's. The adapter (and only the adapter) hard-codes `.parent`, and no LATER consumer disagrees:
`evaluate_models.py:424-425` uses `checkpoint_member_root`, `limn.py:133-142` uses it too (the
comment at `adapters/models.py:133-142` explains that this very `.parent` bug "looked one level too
high" for LIMN's `checkpoint_root / "<member>.pt"`). Fix: make `CLIP4CirFullftAdapter.command` take
its member directory from `checkpoint_member_root`, exactly as `LIMNAdapter` does — or, equivalently,
note that the adapter's build_command already names the right files and that only the *check* should
be reconciled. Do NOT invent a third convention.

**Related, P2 — `bundle_placeholder` is under-instrumented for this model.** `evaluate_models.py:402`
builds a member manifest only for `("run_directory", "bundle_directory")`; this entry is
`bundle_placeholder` and declares no `checkpoint_bundles` (`true` observed: the model dict has no
`checkpoint_bundles` key; the only bundle in `models.yaml` is LIMN's `base_iter0_all_categories`,
`:988`). Therefore `checkpoint_bundle` and `directory_members` stay `None` and the change-detection
gate (`:749-754`) computes `multi_member = False` → returns True without checking any member digest. A
staged fullft bundle would be run with no per-member integrity binding, unlike LIMN.

### 5.5 What the author release provides (the fix upstream already published)

RUNTIME_VERIFIED, decoded from bounded prefixes of the author's `pretrained_models.tar.gz.part-000`
(the archive is one `cat`-concatenated `tar.gz` split across five assets; tar headers and every member's
data are plain bytes in the concatenation, so a prefix yields every member whose start is inside it):

```
$ curl -sL -r 0-1199999999 -o /tmp/p0z .../pretrained_models.tar.gz.part-000
$ gzip -dc /tmp/p0z | tar -tv | head
drwxrwxr-x alberto/alberto         0 2026-10-09 18:48 FashionIQ/
drwxrwxr-x alberto/alberto         0 2026-10-09 18:47 FashionIQ/RN50x4_noft/
-rw-rw-r-- alberto/alberto 236017921 2022-11-03 09:08 FashionIQ/RN50x4_noft/fiq_comb_RN50x4_noft.pt
drwxrwxr-x alberto/alberto         0 2026-10-09 18:48 FashionIQ/RN50_imageft/
-rw-rw-r-- alberto/alberto 604118785 2022-11-03 12:54 FashionIQ/RN50_imageft/fiq_comb_RN50_imageft.pt
-rw-rw-r-- alberto/alberto 408412975 2022-11-03 09:00 FashionIQ/RN50_imageft/fiq_clip_RN50_imageft.pt
drwxrwxr-x alberto/alberto         0 2026-10-09 18:48 FashionIQ/RN50x4_fullft/
-rw-rw-r-- alberto/alberto 713874695 2022-11-03 09:01 FashionIQ/RN50x4_fullft/fiq_clip_RN50x4_fullft.pt
```

**Both members of this variant are named, sized and hashed by the author release.** RUNTIME_VERIFIED
(I decoded the complete `part-000` — 1,900,000,000 bytes, fetched this pass — and extracted every
member whose bytes lie inside it; `/tmp/extract_fullft.py`, `/tmp/probe_clip_census.py`):

```
member                                       size          sha256 (measured this pass)
FashionIQ/RN50x4_fullft/fiq_clip_RN50x4_fullft.pt   713874695   4111ac139cf96ac0cf31b30bf8f55b654247b06f0f5fbee4fe8dcc8c1c7cfd2f
FashionIQ/RN50x4_fullft/fiq_comb_RN50x4_fullft.pt   236017921   (tar header size observed; data lies
                                                                  beyond the part-000 boundary, so its
                                                                  digest is UNKNOWN here)
```

(Name/size/digest of the CLIP member are measured; the Combiner member's *size* is read from its tar
header in the same decode, and its digest was not computed because its data block falls past the
1.9 GB part boundary — I did not reassemble `part-001..004`, which the assignment's no-download
rule and `/tmp`'s size make unattractive. It is UNKNOWN, not assumed.)

The CLIP overlay's own census (safe loader, no unpickling):

```
top-level keys: ['epoch', 'CLIP']      669 tensors, 178,410,446 parameters
visual.attnpool.positional_embedding [82, 2560]     # the RN50x4 visual tower's 288px/width-2560 grid
text_projection [640, 640]                          # d = 640
```

Two consequences worth stating: (1) the release puts this variant's two members in
`FashionIQ/RN50x4_fullft/` with names that say which is which — `_clip_` vs `_comb_` — matching the
release README's rule and the Drive listing in §5.2 one-for-one; (2) the fullft Combiner member is
**the same 236,017,921 bytes in size as the noft Combiner member**, consistent with §5.3(b): the two
variants share one Combiner architecture, so nothing in the file's dimensions distinguishes them and
the filename is the only signal in the release.

This replaces the registry's `rn50x4-fullft-unresolved.pt` + `combiner_state.pt`/`clip_state.pt`
placeholders with real names in a real tree.

---

## 6. AUXILIARY ASSETS

| name | purpose | source | availability class | generation possible? |
|---|---|---|---|---|
| OpenAI CLIP RN50x4 base weights | `clip.load('RN50x4')` `src/validate.py:279` — still required even though fine-tuned weights are supplied | `auxiliary_assets.yaml:188` (same CDN object as the sibling's `:166`); destination `~/.cache/clip/RN50x4.pt` `:192`; `cache_root_fixed: ~/.cache/clip` `:196` | `automatic_download_available: true` `:198`, `provenance_status: VERIFIED` `:203`, size `421854225` `:201`, digest `7e526bd1…` `:202` | no |
| FashionIQ val captions + splits + images | the evaluator's input | canonical root; staged only through the declared link `fashionIQ_dataset` (`models.yaml:726`) | operator-provided, present but `.png`-only (§7) | no |
| `bpe_simple_vocab_16e6.txt.gz` | `clip.tokenize` | vendored in the `clip` package | transitively available | n/a |

The pinned source states the base weights are needed even with `--clip-model-path`: the overlay is a
`load_state_dict` on an already-constructed module (`:279-281` then `:283-286`), not a `torch.load` of
a whole model, so the architecture must exist before the overlay. `auxiliary_assets.yaml:199-200`
records the same reasoning. OBSERVED locally: `ls ~/.cache/clip` → absent, and preflight reports
`OpenAI CLIP RN50x4 base weights missing: /home/heheboiz/.cache/clip/RN50x4.pt`.

---

## 7. FASHIONIQ PROTOCOL

Registry: `literature_split_label: original` (`models.yaml:747`), `native_protocol:
fashioniq_original_split` (`:748`), `supported_protocols: [fashioniq_original_split]` (`:749`),
layout `fashioniq_standard` (`:744`), extensions `[".jpg"]` (`:727`), `reported_r10: 43.78`,
`reported_r50: 66.93` (`:754-755`).

### 7.1 Gallery and candidate list construction — the reference STAYS

Workbench table `src/evaluation/fashioniq.py:171-181`:
`fashioniq_original_split` = `gallery="full_split"`, `reference_excluded=False`.

Source (`src/validate.py`), the candidate list is built from the classic-mode dataset and nothing
removes a member:

```
113: def fashioniq_val_retrieval(dress_type, combining_function, clip_model, preprocess):
124:     clip_model = clip_model.float().eval()
127:     classic_val_dataset = FashionIQDataset('val', [dress_type], 'classic', preprocess)
128:     index_features, index_names = extract_index_features(classic_val_dataset, clip_model)
129:     relative_val_dataset = FashionIQDataset('val', [dress_type], 'relative', preprocess)
131:     return compute_fiq_val_metrics(relative_val_dataset, clip_model, index_features, index_names, combining_function)
```

`extract_index_features` (`src/utils.py:21-44`) loads `classic_val_dataset` with `shuffle` at its
default (False, `:30-31`) and appends every item:

```
32:     index_features = torch.empty((0, feature_dim)).to(device, non_blocking=True)
33:     index_names = []
38:     for names, images in tqdm(classic_val_loader):
41:             batch_features = clip_model.encode_image(images)
42:             index_features = torch.vstack((index_features, batch_features))
43:             index_names.extend(names)
```

and the gallery inventory itself is the whole category val split, in file order:
`src/data_utils.py:144-147` (`self.image_names.extend(json.load(f))` over
`fashionIQ_dataset/image_splits/split.{dress_type}.val.json` `:146`).

Scoring and ranking, `compute_fiq_val_metrics`:

```
41:     index_features = F.normalize(index_features, dim=-1).float()
44:     distances = 1 - predicted_features @ index_features.T
45:     sorted_indices = torch.argsort(distances, dim=-1).cpu()
46:     sorted_index_names = np.array(index_names)[sorted_indices]
49:     labels = torch.tensor(
50:         sorted_index_names == np.repeat(np.array(target_names), len(index_names)).reshape(len(target_names), -1))
```

**There is no reference removal anywhere in this file's FashionIQ path** — no mask, no column drop, no
`-10e10` sentinel, no re-rank. The reference is, moreover, *used as* the query image out of the same
gallery map: `name_to_feat = dict(zip(index_names, index_features))` `:80` and
`itemgetter(*reference_names)(name_to_feat)` `:101-104`. I grepped for the mechanisms that could
exclude it: the only `reference_mask` in the repository is on the CIRR path
(`src/validate.py:162-166`; `src/cirr_test_submission.py:98-103`). That internal contrast is decisive:
this codebase knows how to exclude the reference and does so for CIRR, not for FashionIQ.

**Registry claim CONFIRMED from source**: `fashioniq_original_split` keeps the reference in the
candidate gallery for this model, and the protocol label is correct.

### 7.2 Query construction, categories, metric

- Query: `reference_name = self.triplets[index]['candidate']` (`src/data_utils.py:155`),
  `target_name = ...['target']` `:166`.
- Captions: deterministic ordered composition, `src/validate.py:89-93` —
  `np.array(captions).T.flatten().tolist()` then
  `f"{a.strip('.?, ').capitalize()} and {b.strip('.?, ')}"`, tokenized with `context_length=77` and no
  `truncate=True` (over-long captions raise rather than being silently cut).
- Categories and order: `shirt`, `dress`, `toptee` `:329`, `:334`, `:339`.
- Recall: `recall_at10 = torch.sum(labels[:, :10]) / len(labels) * 100` `:54`; `:55` for R@50; macro =
  `statistics.mean` over the three `:353-354` (unweighted, not query-weighted).
- One correct target per query, asserted: `:51`.

### 7.3 The paper rows this variant covers

Parsed every `<table>` element of `https://ar5iv.labs.arxiv.org/html/2308.11485` (326,987 bytes, 11
table elements = 10 numbered tables + the loss equation):

- **Table 9** (FashionIQ **validation** SotA): `Ours RN-50x4 Transformer | 44.41 65.26 | 39.46 64.55 |
  47.48 70.98 | 43.78 66.93`. Arithmetic I executed: (44.41+39.46+47.48)/3 = 43.7833 ✓ Average R@10;
  (65.26+64.55+70.98)/3 = 66.93 ✓ Average R@50. The Average columns are the unweighted category macro,
  matching `validate.py:353-354` exactly.
- **Table 10** (CIRR **test** SotA) is the sibling SotA row for the other dataset; this model's
  protocol is FashionIQ, so it is out of scope.
- **Table 1** (FashionIQ val, `CF × IFT × TFT`) is RN-50-only and is the variant grid; its `Combiner`,
  `IFT=✓`, `TFT=✓` row is `39.87 60.84 | 37.67 63.16 | 44.88 68.59 | 40.80 64.20` — the RN-50 sibling
  of this model, not this model.
- §4.3 fixes the backbone claim for the ablation grids ("Throughout all the experiments, we use the
  RN-50 CLIP model") and identifies the fully fine-tuned configuration as the headline one.

**Does this variant's protocol match the `reported_r10/r50` the registry transcribes?** YES, and the
transcription is arithmetically exact. `models.yaml:754-755` records 43.78 / 66.93 and `:756` records
`reported_mean: 55.36`, with the comment at `:750-753` marking the mean as `WORKBENCH_DERIVED`
(`(43.78+66.93)/2 = 55.355`). The committed benchmark row
(`paper_benchmarks/papers.yaml:211-292`, `clip4cir_rn50x4_fullft_fiq_tab9`) carries
`reported_metrics: {r10: 43.78, r50: 66.93, mean: null}` plus a `derived_metrics` entry with the
formula. The paper prints no scalar mean, so leaving it null in the reference and deriving it in the
registry is the correct split.

One structural caveat that should not be papered over: the paper prints **no per-category numbers for
the RN-50x4 backbone other than the ones inside Table 9's cells**, and Table 9's caption says the upper
section holds methods "not directly comparable to our proposed approach" — its own rows are the
lower/comparable section. The registered row is therefore the right one, and it is against a cohort
whose *reference policy* the paper never states (§7.1 shows only the code does).

### 7.4 Cross-method comparability — recorded so nobody breaks it

The protocol invariant holds and is not proposed for change: full ordered split, reference eligible,
deterministic caption composition, 0% evaluation noise. Numbers produced under `fashioniq_original_split`
are NOT interchangeable with `fashioniq_full_gallery_ref_excluded` cohorts (Air-Know/ConeSep/HABIT/
INTENT/DCNet), and the workbench enforces that separation
(`docs/FASHIONIQ_PROTOCOL_PARITY.md:20-24`; `resolve_fashioniq_protocol` raises on an unknown name,
`src/evaluation/fashioniq.py:190-201`).

---

## 8. WORKBENCH INTEGRATION

| component | state | evidence |
|---|---|---|
| adapter | present, real `command()` override, passes BOTH `--combiner-path` and `--clip-model-path` | `adapters/models.py:97-111` |
| bundle member directory | **divergent** from `checkpoint_missing_paths` (off by one directory) | measured in §5.4 |
| request validation | refuses an unverified mapping before anything else | `adapters/base.py:28-35` (`checkpoint_mapping_unverified`) |
| planner | `guarded_plan` refuses early on preflight blockers | OBSERVED: `(None, [4 blockers…])` |
| preflight | 4 blockers, identical for both CLIP4Cir entries | `preflight.py:42-44, 86-100, 110-112` |
| metric parser | present and dispatched by model_id; validates the evaluator's own macro against its categories | `metrics_extraction.py:309-386`, dispatch `:406-407` |
| benchmark gate | reference VERIFIED, comparable; run_meta carries protocol/checkpoint/noise/split/dataset | `papers.yaml:208` (`reference_status: VERIFIED`); `paper_gate.py:154-160`, `:260-277` |
| change detection for the bundle | **no-op for `bundle_placeholder`** — no manifest is built | `evaluate_models.py:402, 421-425, 749-754` (§5.4) |
| per-query support | none; the evaluator prints only per-category and macro recalls | `src/validate.py:344-354` |

RUNTIME_VERIFIED checks through the workbench's own read-only functions:

```
$ source_clean_and_pinned(model, checkout) -> (True, 'dfed9f748a8a…', None)
$ preflight_blockers(model, config)        -> 4 blockers (identical list to the sibling; §10)
$ guarded_plan(model, variant, 'fashioniq_original_split', None, 50) -> (None, [same 4 blockers])
$ python_executable(model) -> RuntimeError execution environment not verified [UNCONFIGURED/UNCONFIGURED]
$ ADAPTERS['clip4cir_rn50x4_fullft']().build_command(request)
  -> FileNotFoundError: checkpoint missing: …/clip4cir_rn50x4_fullft/rn50x4-fullft-unresolved.pt
$ select_reference_outcome('clip4cir_rn50x4_fullft')
  -> reference clip4cir_rn50x4_fullft_fiq_tab9 (reference_status VERIFIED, r10 43.78, r50 66.93)
$ extract_aggregate_metrics('clip4cir_rn50x4_fullft', <Table-9-shaped stdout>, paper_metrics={r10:43.78,…})
  -> EXTRACTED_VERIFIED / clip4cir_stdout_recalls / PARITY_VERIFIED
```

The two models route to the same parser and the same preflight blockers, and they differ exactly where
they must: the adapter's `--clip-model-path` and the checkpoint contract.

---

## 9. CONFIRMED DEFECTS

**CF-1 — P1 — a wrapper that omits `--clip-model-path` silently evaluates the WRONG variant.** Root
cause: the frozen base encoders are constructed at `src/validate.py:279` and the fine-tuned overlay is
conditional (`:283`), with no assertion and no warning if it is skipped; the two configurations differ
by one flag but produce different numbers. Reproduction: read `:279-287` against `:304-309`; the sibling
model's adapter (`adapters/models.py:90-94`) is exactly this command minus the flag. Fix: the adapter
already passes it, so nothing in the workbench needs changing — but any external/manual invocation must
be pinned by the plan, not by convention, and the registry should record that the flag is mandatory for
this model_id (it does, implicitly, via the two-member bundle). Do not add a "fallback to noft" path.

**CF-2 — P1 — the registry and the adapter disagree about where the bundle members live, so no staged
tree can satisfy both.** Measured in §5.4 (`agree? False`; the two staged-tree probes return
complementary results). Root cause: `CLIP4CirFullftAdapter` hard-codes `checkpoint_member_root`'s
answer instead of calling it (`adapters/models.py:102-106` vs `registry.py:883-888`), and the same
mistake is documented as already fixed for LIMN (`adapters/models.py:133-142`). Fix: delegate to
`checkpoint_member_root`, as `LIMNAdapter` and `evaluate_models.py:424-425` already do.

**CF-3 — P1 — the pairing is unverifiable by construction: two same-architecture variants share one
shape and one free CLI.** `src/validate.py` loads `"CLIP"` and `"Combiner"` independently (`:285-286`,
`:306-307`), checks nothing across them, and the released members carry no variant metadata (measured:
keys are `['epoch','Combiner']`; the noft and fullft Combiner states are both
`d=640/proj=2560/hid=5120` and each loads into the other's slot with "All keys matched successfully").
Reproduction: §5.3 (a)/(b)/(c). Fix: record per-member filenames + `expected_sha256` in the registry so
the manifest binds the published bytes to the slots the workbench invented; the base architecture check
is already loud (RuntimeError), so no source change is needed or wanted.

**CF-4 — P0 — the dataset the evaluator would read does not contain the images it opens, and the
failure would be silent.** `src/data_utils.py:158,161,170,176` open `f"{name}.jpg"`; the canonical root
holds `74381 png`, **0 jpg** (OBSERVED), and preflight reports `3817 of 3817 dress`, `6346 of 6346
shirt`, `5373 of 5373 toptee` validation ids lacking `.jpg`. A missing image does not raise:
`src/data_utils.py:151-184` catches every exception per item and returns `None`, and
`collate_fn` (`src/utils.py:81-88`) filters `None` out — so a missing gallery item shrinks the gallery
and a missing query item changes the denominator, without a single non-zero exit. Fix: prepare a
`.jpg`-extension view and declare it as the generated artifact of `clip4cir_fashioniq`
(`preparation_contracts.yaml:154` is `generated_artifacts: []` today).

**CF-5 — P1 — no verified interpreter, no base weights, no dataset link.** OBSERVED, all three:
`python_executable` raises `[UNCONFIGURED/UNCONFIGURED]`; `~/.cache/clip` absent; the checkout has no
`fashionIQ_dataset` entry. Fix: build/verify the environment, stage the auxiliary asset, run the
authorized link step (`prepare_dataset.py:28-67`; refuses to overwrite `:54-56`, refuses a wrong target
`:49-50`).

**CF-6 — P2 — `bundle_placeholder` builds no manifest, so a staged bundle has no per-member integrity
binding.** `evaluate_models.py:402` handles only `run_directory`/`bundle_directory`;
`change detection` (`:749-754`) then short-circuits to `True`. Fix: extend the manifest condition to
include `bundle_placeholder`, or declare a `checkpoint_bundles` entry for this model (the LIMN
precedent, `models.yaml:988`).

**CF-7 — P1 — the record that the author's release names both members (sizes included) is absent from
the registry, and the local Drive-based prose in the sibling documents is wrong.** Root cause: the same
JS-rendered-page misread as CF-2 in the sibling report; plus a registry that predates the 2026-10-12
release. Reproduction: §5.2 (Drive listing), §5.5 (archive listing). Fix: record the real names and the
release URL, correct the prose.

---

## 10. REMAINING BLOCKERS

| # | classification | owner | evidence | remediation |
|---|---|---|---|---|
| B1 | `FIRST_PARTY_ENVIRONMENT_CONFIGURATION` | workbench | `python_executable` → `RuntimeError … WORKBENCH_CLIP4CIR_PYTHON=None`; no CLIP4Cir env in `workbench/envs/` | build + verify env |
| B2 | `PUBLIC_ASSET_NOT_PREPARED` | workbench | preflight: base RN50x4 weights missing in `~/.cache/clip` | stage via the auxiliary manifest |
| B3 | `PUBLIC_ASSET_NOT_PREPARED` | workbench | preflight: `canonical dataset link missing: …/CLIP4Cir/fashionIQ_dataset -> …/data/fashionIQ_dataset` | authorized `prepare_dataset.py --model clip4cir_rn50x4_fullft` |
| B4 | `AUXILIARY_ASSET_MISSING` | workbench | preflight: 3817/6346/5373 val images "lack .jpg" | `.jpg`-extension preparation (CF-4) |
| B5 | `PUBLIC_CHECKPOINT_NOT_MAPPED` | workbench | registry holds placeholders (`models.yaml:770,772`) with `download_url: null`, no size, no digest (`:781-782`), while the author names both members in a checksummed release (§5.5: CLIP member 713,874,695 bytes sha256 `4111ac13…` measured; Combiner member 236,017,921 bytes, digest UNKNOWN) and Drive lists exactly two files (§5.2) | record real filenames + per-member digests + release URL |
| B6 | `CHECKPOINT_MAPPING_UNVERIFIED` (mapping semantics, not bytes) | workbench | no cross-check binds the CLIP state to the Combiner state (`src/validate.py:285-286, 306-307`); same-architecture members load into each other's slots | per-member digests in the bundle manifest (CF-3) |
| B7 | `GPU_RUNTIME_UNTESTED` | operator | `environment.cuda: UNKNOWN` (`models.yaml:764`); `_requires_gpu` treats UNKNOWN as "no requirement" (`runtime.py:918-928`), which is correct here (`src/utils.py:15-18` falls back to CPU) | optional: record `cuda` as not required |

No `UPSTREAM_CLI_MISSING` (the CLI exists and the adapter matches it), no `UPSTREAM_IMPORT_BUG`, no
`DEPENDENCY_VERSION_CONFLICT`, no `SOURCE_UNAVAILABLE`, no `CHECKPOINT_UNPUBLISHED` (published and
reachable), no `CHECKPOINT_URL_INACCESSIBLE`, no `NONDETERMINISTIC_PREPROCESSING` (the val path is
deterministic: no augmentation, fixed caption composition, `torch.no_grad()` extraction), and no
`UNSAFE_CHECKPOINT_SERIALIZATION` established — the safe loader (`weights_only=True`) accepted both
extracted members, so no untrusted-pickle hazard was observed (see §12).

---

## 11. IMPLEMENTATION PLAN

1. **Bundle record** (workbench, hours): replace the placeholders with the author's names —
   `fiq_clip_RN50x4_fullft.pt` (713,874,695 bytes, measured this pass) and `fiq_comb_RN50x4_fullft.pt`
   — plus per-member `expected_sha256` once each is measured, `source_type: author_github_release`, and
   `checkpoint_mapping_status: VERIFIED_METADATA`. Scientific risk: none (data only). Effect: the
   `checkpoint_mapping_unverified` refusal stops firing and the members become digest-checkable.
2. **Adapter/bundle path reconciliation** (workbench, ~1 line): `checkpoint_member_root` (CF-2).
3. **Manifest coverage for `bundle_placeholder`** (workbench, small): CF-6.
4. **Dataset `.jpg` preparation** (workbench, small): declared generated artifact (CF-4).
5. **Environment + assets** (operator): B1–B3, then a first real run.
6. **Documentation corrections** (workbench, minutes): CF-7, and the sibling's Drive prose.
7. Protocol, gallery, reference policy, caption composition and noise condition are untouched. Nothing
   in this plan changes Protocol A or makes the reference excludable.

Expected outputs after 1–6: one command line, six per-category numbers, two macro lines parsed by
`parse_combiner_output` into `EXTRACTED_VERIFIED`, and a gate comparison against
`clip4cir_rn50x4_fullft_fiq_tab9` (43.78/66.93, tolerance per `paper_gate.compare_metric`).

---

## 12. FINAL VERDICT

**BLOCKED.**

The static facts hold where they can be checked: pin exact and clean; a native CLI exists and the
adapter matches it, including the flag that defines this variant; the two-state bundle is genuinely
two-state and both states are needed to reach the published numbers; the reference-eligible
`fashioniq_original_split` protocol is confirmed from the candidate-list construction itself; the
transcribed paper row 43.78/66.93 is arithmetically exact against Table 9; the parser exists and was
exercised. The author released, on 2026-10-12, a checksummed archive that names both members.

It is blocked because: nothing can run (no interpreter, no base weights, no dataset link, no `.jpg`
tree with a silent-skip failure mode), the registry's bundle names do not exist upstream and its member
paths disagree with its own adapter, and the CLIP↔Combiner pairing is unverifiable by any mechanism in
the source or in the record. `VERIFIED` would be unsupported (nothing ran). `VARIANT` would understate
real blockers. `NOT_RUN` would understate what was established.

---

## 13. EVIDENCE

Repo `https://github.com/ABaldrati/CLIP4Cir` · pinned `dfed9f748a8a4d05abf613164e06d59b22dbdf13` ·
SNAPSHOT 2 `56bb547909d57bd8b5d2ef7ba55149962f77d482` (not adopted).

Files opened (pinned unless noted): `README.md`; `src/validate.py` (in full, 360 lines);
`src/utils.py` (`save_model:117-131`, `extract_index_features:21-44`, `collate_fn:81-88` —
an empty `__init__.py` is absent, so package semantics were confirmed from the file list);
`src/combiner.py`; `src/data_utils.py`; `src/clip_fine_tune.py` (argparse `:421-439`, save sites
`:223-228`, `:405-416`); `src/combiner_train.py` (argparse `:424-442`, save sites `:222-227`,
`:408-419`); `src/cirr_test_submission.py` (`:98-103`);
workbench: `registry/models.yaml`, `registry/auxiliary_assets.yaml`,
`registry/preparation_contracts.yaml:140-169`, `registry/paper_benchmarks/papers.yaml:211-292`,
`backend/adapters/models.py`, `backend/adapters/base.py`, `backend/registry.py`,
`backend/runtime.py`, `backend/preflight.py`, `backend/metrics_extraction.py`,
`backend/benchmark_references.py`, `backend/paper_gate.py`, `scripts/evaluate_models.py`,
`docs/FASHIONIQ_PROTOCOL_PARITY.md`; hypotheses only: `docs/reviews/models/clip4cir_rn50x4_fullft.md`.

Commands and observed results (read-only; all artifacts under `/tmp`):

1. `git -C … rev-parse HEAD` → `dfed9f748a8a…`; `status --porcelain` → empty; `source_clean_and_pinned` → `(True, …, None)`.
2. `git -C … ls-remote origin HEAD` → `56bb5479…`; `fetch --dry-run` → `dfed9f7..56bb547 master`, `[new tag] weights`.
3. `curl api.github.com/…/compare/dfed9f7…56bb547` → ahead 1, one file, `README.md +4 −4`; commit message `Update README`, `2026-10-12T19:50:59Z`.
4. `curl api.github.com/…/releases/tags/weights` → 5 assets with `sha256:` digests and sizes.
5. Drive: `embeddedfolderview?id=1fnGeZQA…` → 6 subfolders incl. `RN50x4_fullft`;
   `?id=1BPE33_XSm33Min0OaGW2Sl9rcddZ-WF-` → `fiq_clip_RN50x4_fullft.pt` + `fiq_comb_RN50x4_fullft.pt` (ids `1cQS8vFhNBpZOhyDNe15STz00wMvYEQEb`, `1MxFkozS7RJy5VOoDBC2gJkoYqB4jD5tB`).
6. `curl -r 0-1899999999 -o /tmp/pz …/part-000` (1,900,000,000 bytes, complete) → decoded with `/tmp/extract_fullft.py`
   → `FashionIQ/RN50x4_fullft/fiq_clip_RN50x4_fullft.pt` 713,874,695 bytes sha256 `4111ac13…`;
   `FashionIQ/RN50x4_fullft/fiq_comb_RN50x4_fullft.pt` header size 236,017,921 (data beyond part-000);
   `FashionIQ/RN50x4_noft/fiq_comb_RN50x4_noft.pt` 236,017,921 bytes sha256 `b31882fe…`.
6b. `/tmp/probe_clip_census.py` → keys `['epoch','CLIP']`; 669 tensors; 178,410,446 params; `visual.attnpool.positional_embedding [82,2560]`; `text_projection [640,640]`.
7. `.venv/bin/python` (registry helpers) → `checkpoint_path` = `…/clip4cir_rn50x4_fullft/rn50x4-fullft-unresolved.pt`; `agree? False`; staged-tree probes complementary (§5.4).
8. `/tmp/probe_load.py`, `/tmp/probe_strict.py`, `/tmp/probe_mismatch.py` → safe loader accepted; keys `['epoch','Combiner']`; 12 tensors / 59,003,521 params; strict load OK against the pinned class; RN50-recipe dims raise `RuntimeError`.
9. `preflight_blockers` / `guarded_plan` → `(None, [4 blockers])`; `python_executable` → `RuntimeError [UNCONFIGURED/UNCONFIGURED]`; `build_command` → `FileNotFoundError checkpoint missing: …`.
10. `select_reference_outcome('clip4cir_rn50x4_fullft')` → reference `clip4cir_rn50x4_fullft_fiq_tab9`, `VERIFIED`, r10 43.78 / r50 66.93.
11. `curl ar5iv.labs.arxiv.org/html/2308.11485` → 326,987 bytes; Table 9 `44.41 65.26 39.46 64.55 47.48 70.98 43.78 66.93`; Table 1 `Combiner ✓ ✓ 39.87 … 64.20`.
12. `ls data/fashionIQ_dataset/images | sed 's/.*\.//' | sort | uniq -c` → `74381 png`.
13. `ls ~/.cache/clip` → absent; `ls workbench/envs/` → `README.md`, `csmcir-import.txt`.

---

## 14. ACHIEVED VERIFICATION LEVEL

**LEVEL 3 of 6.**

- Level 1 — identity, pin, registry consistency: achieved.
- Level 2 — static source + protocol auditing with independent evidence (candidate-list construction,
  paper-row transcription, SNAPSHOT 2): achieved.
- Level 3 — the author's release was fetched (the complete 1.9 GB `part-000`), decoded, and BOTH members
  of this variant were measured: the CLIP overlay with its full key/shape census (669 tensors,
  178,410,446 params, `[82,2560]` positional grid, `[640,640]` text projection) and its SHA-256
  `4111ac13…`, and the Combiner member's header size (236,017,921); the extracted noft Combiner state
  was strict-loaded into the *pinned* `Combiner` class and cross-slot/cross-backbone mismatch behaviour
  was measured. Achieved (RUNTIME_VERIFIED, scratch only). The fullft Combiner member's own digest is
  UNKNOWN (its data falls past the part-000 boundary) and is reported as such.
- Level 4 (a real evaluator run producing Recall@10/50) is NOT reached: no verified interpreter, no
  base RN50x4 cache, no dataset link, no `.jpg` tree, and the bundle is blocked before launch.
- Levels 5–6 are out of reach.

Nothing here claims a metric was reproduced. No threshold, gallery, reference policy or noise condition
was proposed for change.
