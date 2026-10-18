# combiner_rn50x4_noft — reproducibility reassessment (independent, adversarial, READ-ONLY)

Auditor: independent reassessment pass. Nothing under `/home/heheboiz/data/cir` was modified except
this file. All probe output and downloads went to `/tmp`. No GPU work, no evaluator execution, no
metric produced by a run: nothing here is a runtime reproduction. Every claim is `STATIC_AUDIT`
(a file I opened, or a command I ran and its observed output) unless it is explicitly labelled
`RUNTIME_VERIFIED`, which in this report means "bytes I fetched and decoded on this host", never
"the evaluator ran".

Prior workbench audits (`workbench/docs/reviews/models/combiner_rn50x4_noft.md`,
`workbench/docs/reviews/final_audit/models/combiner_rn50x4_noft.md`) were treated as HYPOTHESES and
re-tested against the pinned tree and live upstream. Two of their claims are REJECTED below (§2, §5);
the rest are confirmed with independent evidence.

---

## 1. MODEL IDENTITY

| field | value | evidence |
|---|---|---|
| model_id | `combiner_rn50x4_noft` | `workbench/registry/models.yaml:660` |
| method | "Combiner RN50x4 noft" | `models.yaml:672` |
| paper | "Composed Image Retrieval using Contrastive Learning and Task-oriented CLIP-based Features" | `models.yaml:675` |
| paper URL | https://arxiv.org/abs/2308.11485 | `models.yaml:676` |
| year / generation | 2023 / legacy | `models.yaml:673-674` |
| source URL | https://github.com/ABaldrati/CLIP4Cir | `models.yaml:677` |
| branch | `master` | `models.yaml:678` |
| pinned commit | `dfed9f748a8a4d05abf613164e06d59b22dbdf13` | `models.yaml:679` |
| source_dir | `CLIP4Cir` | `models.yaml:681` |
| native protocol | `fashioniq_original_split` | `models.yaml:686` |
| reported metrics | R@10 null, R@50 null, mean null | `models.yaml:688-690` |
| reproduction_status | `BLOCKED_CHECKPOINT_MAPPING` | `models.yaml:691` |
| checkpoint variant | `rn50x4_noft_unresolved` / `rn50x4-noft-unresolved.pt` / author Google Drive folder | `models.yaml:703-720` |

STATIC_AUDIT.

**This model is one of TWO registry entries that share the same upstream repository.** The sibling
`clip4cir_rn50x4_fullft` (`models.yaml:721`) is a distinct `model_id`, with a distinct adapter
(`CombinerNoftAdapter` `workbench/backend/adapters/models.py:86-94` vs `CLIP4CirFullftAdapter`
`:97-111`), a distinct `reproduction_status` (`BLOCKED_CHECKPOINT_MAPPING` `:691` vs
`BLOCKED_CHECKPOINT_BUNDLE` `:757`) and a distinct checkpoint record (`:703-720` vs `:768-784`).
Nothing established for the sibling is reused here.

---

## 2. SOURCE STATUS

**Pinned checkout verified.** OBSERVED:

```
$ git -C workbench/third_party/CLIP4Cir rev-parse HEAD
dfed9f748a8a4d05abf613164e06d59b22dbdf13
$ git -C workbench/third_party/CLIP4Cir status --porcelain
(empty)
$ git -C workbench/third_party/CLIP4Cir log -1 --format='%H %ci %s'
dfed9f748a8a4d05abf613164e06d59b22dbdf13 2023-09-05 09:31:52 +0200 Update README.md
```

Equal to `upstream_commit_sha` (`models.yaml:679`). The workbench's own gate agrees:
`.venv/bin/python -c "…source_clean_and_pinned(model, source)…"` →
`(True, 'dfed9f748a8a4d05abf613164e06d59b22dbdf13', None)` for BOTH entries, read from
`workbench/backend/runtime.py:2403-2427` (dirty → refuse; missing pin → refuse; mismatch → refuse).

Checkout contents, complete (`git ls-tree -r --name-only`): `.gitignore`, `LICENSE`, `README.md`,
`images/{Combiner-architecture,cir-overview,clip-fine-tuning,combiner-training}.png`,
`src/{cirr_test_submission,clip_fine_tune,combiner,combiner_train,data_utils,utils,validate}.py`.
No missing module in the pinned revision: every import in the evaluation path resolves within the tree
or to a declared distribution (`src/validate.py:1-18`, `src/data_utils.py:1-9`, `src/utils.py:1-13`).

### SNAPSHOT 2 — current upstream state (this changed on 2026-10-12)

```
$ git -C workbench/third_party/CLIP4Cir ls-remote origin HEAD
56bb547909d57bd8b5d2ef7ba55149962f77d482	HEAD
$ git -C workbench/third_party/CLIP4Cir fetch origin --dry-run
   dfed9f7..56bb547  master     -> origin/master
 * [new tag]         weights    -> weights
```

One commit ahead; the pin was NOT changed. `curl api.github.com/repos/ABaldrati/CLIP4Cir/compare/
dfed9f7…56bb547` → `status ahead, ahead_by 1, behind_by 0, total_commits 1`, files: `README.md` +4 −4,
nothing else. `curl …/commits/56bb547` → message `Update README`, date `2026-10-12T19:50:59Z`.

**What changed, and it is decisive for this model:** the README no longer points at Google Drive. The
pinned version (`README.md:204-205`) says the pre-trained models are "via [Google
Drive](https://drive.google.com/drive/folders/1ny2hhzP8HZBnXhjvDEni8P8G4_inCNTv?usp=sharing)". The
current upstream README says instead:

> Pre-trained CLIP and Combiner checkpoints for **FashionIQ** and **CIRR** are available in the
> [GitHub releases](https://github.com/ABaldrati/CLIP4Cir/releases).

and the author published a release `weights` (published `2026-10-12T19:50:01Z`, `target_commitish:
master`) with five assets, each carrying a GitHub-published SHA-256:

| asset | size (published) | sha256 (published) |
|---|---|---|
| `pretrained_models.tar.gz.part-000` | 1900000000 | `7a70c0e9f52edbfb2c479b747430391f25bba2f34a01d292fbd672a955964106` |
| `pretrained_models.tar.gz.part-001` | 1900000000 | `8a976c834d36f206f8b3c5782b6abbb33331589313de5fe602d530def643fcb3` |
| `pretrained_models.tar.gz.part-002` | 1900000000 | `849e141225607e096f74748527cc3acbb936f71109779eca0e09f7ab2b16378e` |
| `pretrained_models.tar.gz.part-003` | 1900000000 | `45098bbf6d01c09856cdcce745a301458a1243f2b33904318f5c38cd5f5df89c` |
| `pretrained_models.tar.gz.part-004` | 945777978 | `8f999d94ed58edb7b3fe20bf01beb534a9c6f95d98a40b546793fd9655d10a1c` |

(Source: `curl api.github.com/repos/ABaldrati/CLIP4Cir/releases/tags/weights`; the `digest` field is
GitHub's own `sha256:` claim per asset.) OBSERVED reachability: `curl -r 0-1023 -o /dev/null -w
'%{http_code}'` → `206` for parts 000/001/002; `curl -sIL …/part-000` → `HTTP/2 302` →
`HTTP/2 200`, `content-disposition: attachment; filename=pretrained_models.tar.gz.part-000`,
`content-length: 1900000000` (exactly the published size).

**Verdict on SNAPSHOT 2: the upstream authors FIXED the release.** They did not add a CLI, they did
not change the code, they did not move — they replaced an undocumented Drive folder with a
checksummed GitHub release. The pinned revision still points at the Drive folder, so the pin decides
what must be reproduced and nothing here proposes changing it. This fix is not visible to any audit
that reads only the pinned README.

---

## 3. NATIVE EVALUATION WORKFLOW

Entrypoint: `src/validate.py` (`models.yaml:667-671`, `replay_type: OFFICIAL_CLI`);
`CombinerNoftAdapter.script = "src/validate.py"` (`adapters/models.py:88`).

Function chain (all STATIC_AUDIT, pinned tree):

1. `main()` `src/validate.py:263` → argparse `:264-277` → `args = parser.parse_args()` `:277`.
2. `clip_model, clip_preprocess = clip.load(args.clip_model_name, device=device, jit=False)` `:279`;
   `input_dim = clip_model.visual.input_resolution` `:280`; `feature_dim = clip_model.visual.output_dim` `:281`.
3. `if args.clip_model_path:` `:283` — **not taken for this variant** (the adapter never passes it, §3 CLI).
4. `targetpad_transform(args.target_ratio, input_dim)` `:291` (default transform, `:274-275`).
5. `elif args.combining_function.lower() == 'combiner':` `:304` → `Combiner(feature_dim,
   args.projection_dim, args.hidden_dim)` `:305` → `state_dict = torch.load(args.combiner_path,
   map_location=device)` `:306` → `combiner.load_state_dict(state_dict["Combiner"])` `:307` →
   `combiner.eval()` `:308` → `combining_function = combiner.combine_features` `:309`.
6. `elif args.dataset.lower() == 'fashioniq':` `:325` → `fashioniq_val_retrieval('shirt'|'dress'|'toptee', …)`
   `:329/:334/:339` → `fashionIQ_val_retrieval` `:113` → `FashionIQDataset('val',[dress_type],'classic',preprocess)`
   `:127`, `extract_index_features` `:128` (`src/utils.py:21-44`), `FashionIQDataset(…,'relative',…)` `:129`,
   `compute_fiq_val_metrics` `:131`.
7. `compute_fiq_val_metrics` `:21-57`: `generate_fiq_val_predictions` `:35`; `F.normalize(index_features)`
   `:41`; `distances = 1 - predicted_features @ index_features.T` `:44`; `torch.argsort` `:45`;
   `labels` `:49-50`; assert `:51`; `recall_at10/50` `:54-55`.
8. `print(f"average recall10 = {mean(…)}")` `:353-354`.

**CLI present: YES, and it is a real CLI** (`main()` behind `if __name__ == '__main__':` `:359-360`,
README recipe `README.md:256-269`). What a wrapper must call (exact shape the adapter builds,
`adapters/models.py:90-94`):

```
$WORKBENCH_CLIP4CIR_PYTHON <checkout>/src/validate.py \
  --dataset fashionIQ --combining-function combiner --combiner-path <combiner.pt> \
  --projection-dim 2560 --hidden-dim 5120 --clip-model-name RN50x4 \
  --target-ratio 1.25 --transform targetpad
```

Missing components to run as-is: base RN50x4 weights in `~/.cache/clip`, the `<checkout>/fashionIQ_dataset`
link, `.jpg` images, a verified interpreter, and the checkpoint itself. All five are confirmed missing
in §10.

---

## 4. DEPENDENCIES

Module-level imports of the executed path, and where each comes from. "Owner" is who must fix it.

| import | file:line | owner | required version | installable? | py/torch/cuda | safe remedy | test result |
|---|---|---|---|---|---|---|---|
| `torch` | `src/validate.py:10`; `src/combiner.py:1`; `src/utils.py:6` | upstream | registry declares 1.11.0 | yes (`conda -c pytorch`) | py3.8; torch 1.11.0; CPU-capable | build the declared env | not executed (no env) |
| `torch.nn.functional` | `src/validate.py:11`, `src/combiner.py:3` | upstream | same | yes | — | — | not executed |
| `torch.utils.data.DataLoader` | `src/validate.py:13`, `src/utils.py:10` | upstream | same | yes | — | — | not executed |
| `torchvision.transforms*` | `src/data_utils.py:7-9` | upstream | registry declares 0.12.0 | yes | py3.8 | build the declared env | not executed |
| `clip` / `clip.model.CLIP` | `src/validate.py:8,12`; `src/utils.py:8` | upstream README (`README.md:118`) | **unpinned** — `pip install git+https://github.com/openai/CLIP.git` | yes | py3.8; needs `torch`, `torchvision`, `ftfy`, `regex`, `tqdm`, `Pillow` transitively | record a commit sha | not executed |
| `numpy` | `src/validate.py:9` | upstream (README omits it) | not declared | yes (PyPI) | — | add to the env spec | not executed |
| `tqdm` | `src/validate.py:14`; `src/utils.py:11` | upstream (README omits it) | not declared | yes (PyPI) | — | add to the env spec | not executed |
| `PIL` / `PIL.Image` | `src/data_utils.py:5-6` | upstream (README omits it) | not declared | yes (PyPI) | — | add to the env spec | not executed |
| `pandas` | `src/combiner_train.py:11` (training only, not this path) | upstream README | 1.4.2 | yes | — | — | n/a |
| `comet_ml.Experiment` | `src/combiner_train.py:1` (training only, not this path) | upstream README | 3.21.0 | yes | — | — | n/a |
| `statistics`, `argparse`, `operator`, `pathlib`, `multiprocessing`, `typing`, `random`, `json` | `src/validate.py:1-6` etc. | stdlib | — | — | — | none | — |

**Dependency categories D1–D8 are not defined anywhere in this repository.** I grepped for the legend
(`grep -rn "D1" workbench/ --include=*.py --include=*.md` → only unrelated finding-ids such as
`final_hardening/D.md` D-1 and `tests/test_release_blocker_invocation_proof.py` D-4); the same gap is
already recorded by two sibling reassessments (`model_reassessment/sprc.md:169`,
`model_reassessment/ptha_mtst.md:299`). I therefore classify without the D-numbers rather than invent
their mapping: **`numpy`, `tqdm` and `Pillow` are transitively installed by `torch`/`clip` but
declared nowhere** — that is an `INSTALLABLE_DEPENDENCY_MISSING`-shaped documentation gap (P2), not a
runtime blocker. `clip` being an unpinned git install is the upstream's own reproducibility weakness
(`README.md:118`; registry copies it verbatim as the bare string `clip`, `models.yaml:700`).

OBSERVED: no interpreter exists. `ls workbench/envs/` → `README.md`, `csmcir-import.txt` only; no
CLIP4Cir environment. `WORKBENCH_CLIP4CIR_PYTHON` is unset (OBSERVED python `None`).
`workbench/artifacts/environment_assignment.json` lists exactly one assigned model (`limn`,
`cir-1bbd9872cf166f0d`, `RUNTIME_VERIFIED`); neither CLIP4Cir entry appears.

Dependency probe (RUNTIME_VERIFIED on this host, purely local):
`.venv/bin/python -c "python_executable(model)"` → `RuntimeError: execution environment not verified
[UNCONFIGURED/UNCONFIGURED]: WORKBENCH_CLIP4CIR_PYTHON=None (WORKBENCH_CLIP4CIR_PYTHON not set)` for
this model. That is `workbench/backend/runtime.py:1781-1799` refusing to fabricate an interpreter.

---

## 5. CHECKPOINTS

### 5.1 What the variant is, and which file it is

Registry record (`models.yaml:703-720`) quoted verbatim:

```
703:    checkpoint_variants:
704:      - checkpoint_id: rn50x4_noft_unresolved
705:        filename: rn50x4-noft-unresolved.pt
706:        artifact_type: file
707:        download_url: null
711:        source_url: https://drive.google.com/drive/folders/1ny2hhzP8HZBnXhjvDEni8P8G4_inCNTv?usp=sharing
712:        source_type: author_google_drive
715:        checkpoint_training_noise_pct: 0
716:        evaluation_noise_pct: 0
717:        expected_size_bytes: null
718:        expected_sha256: null
719:        status: CHECKPOINT_MAPPING_UNVERIFIED
720:        checkpoint_mapping_status: UNVERIFIED
```

**REJECTED HYPOTHESIS.** `workbench/docs/AUXILIARY_ASSET_AUDIT.md:221`,
`workbench/registry/assets/cluster_b4_classic.yaml:79` and
`workbench/registry/audits/cluster_c_classic.yaml:61` all claim:

> "Author Google Drive folder … does not document which file maps to the RN50x4 noft Combiner
> checkpoint, nor does it publish a checksum."

The first half is false. OBSERVED (live, unauthenticated, this pass):
`https://drive.google.com/embeddedfolderview?id=1fnGeZQA4XMfEERo-qDCosJXILl1UdQ_l#list` → title
`FashionIQ`, entries `RN50_fullft, RN50_imageft, RN50_noft, RN50_textft, RN50x4_fullft, RN50x4_noft`;
`…?id=1OHmPOU9VYxDynoCyDfCIY8ijTsSltxuL` → title `RN50x4_noft`, **exactly one** entry,
`fiq_comb_RN50x4_noft.pt` (id `1Hb3lp3Vk0tSpAcMoq8KG5tFsw-jymryi`, last-modified `11/3/22`). The
folder's own README (`https://drive.usercontent.google.com/download?id=1G7kZo7paiaRd3mif9_saERyhPevRCM88&export=download`,
HTTP 200, 2449 bytes) documents the release structure AND states the keys and dims:

> "2) RN50x4-based Combiner trained on out-of-the-box CLIP features - CVPR22 Demo paper"
> "in the RN50x4-based is equal to 2560 … in the RN50x4-based is equal to 5120"
> `combiner.load_state_dict(combiner_state_dict['Combiner'])`

The second half (no checksum published **by the author**) stood at the time those documents were
written; §5.2 shows the authors have since published one. **P1**: the prose is wrong on both halves now
and would otherwise license the sibling fullft entry to reuse the same (also wrong) finding.

### 5.2 Does upstream ship it, link it, or neither? — the answer CHANGED on 2026-10-12

Pinned revision: **links it only** (Google Drive folder, `README.md:204-205`; no file name, no digest,
no size anywhere in the repo). Current upstream: **ships it, with a digest** (release `weights`).

RUNTIME_VERIFIED (this host, bytes I fetched and decoded):

```
$ curl -sL -r 0-1199999999 -o /tmp/p0z https://github.com/ABaldrati/CLIP4Cir/releases/download/weights/pretrained_models.tar.gz.part-000
$ gzip -dc /tmp/p0z | tar -tv | head
drwxrwxr-x alberto/alberto         0 2026-10-09 18:48 FashionIQ/
drwxrwxr-x alberto/alberto         0 2026-10-09 18:47 FashionIQ/RN50x4_noft/
-rw-rw-r-- alberto/alberto 236017921 2022-11-03 09:08 FashionIQ/RN50x4_noft/fiq_comb_RN50x4_noft.pt
drwxrwxr-x alberto/alberto         0 2026-10-09 18:48 FashionIQ/RN50_imageft/
-rw-rw-r-- alberto/alberto 604118785 2022-11-03 12:54 FashionIQ/RN50_imageft/fiq_comb_RN50_imageft.pt
```

The `part-000` prefix contains the noft member in full, so I decoded it out of the tar stream
(`/tmp/probe_ckpt.py`, scratch only):

```
members seen in the prefix: [("FashionIQ/",0), ("FashionIQ/RN50x4_noft/",0),
  ("FashionIQ/RN50x4_noft/fiq_comb_RN50x4_noft.pt", 236017921),
  ("FashionIQ/RN50_imageft/",0), ("FashionIQ/RN50_imageft/fiq_comb_RN50_imageft.pt", 604118785)]
extracted: ["FashionIQ/RN50x4_noft/fiq_comb_RN50x4_noft.pt", 236017921,
            "b31882fe6f7e3e194cb9cfda048926844ee0ccc9d4ebc5483fb3bacfa5972955"]
```

So the authoritative, independently measured facts for this variant's checkpoint are:

- **exact path in the author archive:** `FashionIQ/RN50x4_noft/fiq_comb_RN50x4_noft.pt`
- **exact size:** 236,017,921 bytes
- **sha256 of the extracted member:** `b31882fe6f7e3e194cb9cfda048926844ee0ccc9d4ebc5483fb3bacfa5972955`
- **container digest (GitHub-published, per asset):** `part-000` =
  `7a70c0e9f52edbfb2c479b747430391f25bba2f34a01d292fbd672a955964106` (I did not hash the 1.9 GB part
  myself — I retrieved a bounded prefix, so that value is QUOTED, not measured.)

### 5.3 Loader contract — RUNTIME_VERIFIED against the pinned class

`src/validate.py:306-307` requires `state_dict["Combiner"]`. I loaded the released member with the
pinned `Combiner` class (`/tmp/probe_strict.py`; `sys.dont_write_bytecode = True`, so the checkout was
not touched — `git status --porcelain` stayed empty afterwards):

```
$ .venv/bin/python /tmp/probe_strict.py
strict load OK: <All keys matched successfully>
forward is eval-only?  True
combine_features output shape: (3, 640) | L2 norms: [1.0, 1.0, 1.0]
```

Key census (`/tmp/probe_load.py`, `torch.load(..., weights_only=True)` — the SAFE loader accepted the
file, so there is no deserialization hazard to report):

```
top-level keys: ['epoch', 'Combiner']            # 'epoch' is an int; the state is under 'Combiner'
12 tensors, 59,003,521 parameters:
 text_projection_layer.weight [2560, 640]      text_projection_layer.bias [2560]
 image_projection_layer.weight [2560, 640]     image_projection_layer.bias [2560]
 combiner_layer.weight [5120, 5120]            combiner_layer.bias [5120]
 output_layer.weight [640, 5120]               output_layer.bias [640]
 dynamic_scalar.0.weight [5120, 5120]          dynamic_scalar.0.bias [5120]
 dynamic_scalar.3.weight [1, 5120]             dynamic_scalar.3.bias [1]
```

These dims are exactly `Combiner(clip_feature_dim=640, projection_dim=2560, hidden_dim=5120)`
(`src/combiner.py:11-31`) with RN50x4's CLIP `d = 640` (CLIP paper Table 19 as cited by the workbench;
the author's Drive README states the same 2560/5120 for the RN50x4 family), and the adapter passes
precisely `--projection-dim 2560 --hidden-dim 5120` (`adapters/models.py:93`). **The registry's
`rn50x4_noft_unresolved` is a placeholder name that does not exist anywhere upstream; the real file
name, size and digest are now established.**

Mapping confidence: **HIGH** (single file in a variant-named folder, author README names the variant,
strict load against the pinned class succeeds, dims match the adapter). Checksum status: **measured on
this host** for the member; the container digest is the author's published claim.

---

## 6. AUXILIARY ASSETS

| name | purpose | source | availability class | generation possible? |
|---|---|---|---|---|
| OpenAI CLIP RN50x4 base weights | `clip.load('RN50x4')` at `src/validate.py:279` constructs the backbone and downloads it if absent | `https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt` (`auxiliary_assets.yaml:166`; destination `~/.cache/clip/RN50x4.pt` `:170`, `cache_root_fixed` `:174`) | `automatic_download_available: true` (`:176`), `provenance_status: VERIFIED` (`:183`); digest `7e526bd1…` (`:182`) equals the CDN object name — OpenAI's own URL-convention hash, which upstream CLIP self-verifies | no — must be downloaded |
| FashionIQ val captions + splits + images | the evaluator's whole input | canonical root `/home/heheboiz/data/cir/data/fashionIQ_dataset`; staged into the checkout only via the declared link `fashionIQ_dataset` (`models.yaml:665`) | operator-provided (present locally, but `.png`-only, §7) | no |
| `bpe_simple_vocab_16e6.txt.gz` BPE table | `clip.tokenize` | vendored inside the installed `clip` package | transitively available | n/a |

The base-weight cache path is recorded correctly for the consumer: `clip/clip.py` resolves
`download_root or os.path.expanduser("~/.cache/clip")` (OBSERVED from `raw.githubusercontent.com/
openai/CLIP/main/clip/clip.py`, line ~121) and `auxiliary_assets.yaml:171-174` says exactly that.
OBSERVED locally: `ls ~/.cache/clip` → `No such file or directory`, and the workbench's own preflight
reports it missing (§10). The published CDN length is `421854225`, recorded at `auxiliary_assets.yaml:181`
this engagement's HEAD; I did not hash the bytes (a ~400 MiB download), so VERIFIED means URL + published
digest, not bytes-on-disk.

---

## 7. FASHIONIQ PROTOCOL

Registry: `literature_split_label: original` (`models.yaml:685`), `native_protocol:
fashioniq_original_split` (`:686`), `supported_protocols: [fashioniq_original_split]` (`:687`),
`fashioniq_layout: fashioniq_standard` (`:682`), `fashioniq_image_extensions: [".jpg"]` (`:666`).

Workbench protocol table `src/evaluation/fashioniq.py:171-181`:

```
172:    "fashioniq_original_split": FashionIQProtocol(
173:        protocol_id="fashioniq_original_split", gallery="full_split",
174:        reference_excluded=False, aliases=("fashioniq_original",)),
```

**Gallery.** `fashioniq_val_retrieval` (`src/validate.py:113`) builds
`FashionIQDataset('val', [dress_type], 'classic', preprocess)` `:127` and calls
`extract_index_features(classic_val_dataset, clip_model)` `:128`. The gallery is therefore
`image_splits/split.{category}.val.json` in file order: `src/data_utils.py:144-147`
(`self.image_names.extend(json.load(f))`), iterated whole with `shuffle` at its default (False) by
`src/utils.py:30-31` and consumed at `src/utils.py:38-43`. Per-category, full split, no dedup.

**Reference policy: the reference STAYS in the gallery.** `compute_fiq_val_metrics`
(`src/validate.py:21-57`) computes `distances = 1 - predicted_features @ index_features.T` `:44`,
sorts ascending `:45`, and builds labels against **target names only** `:49-50`, asserting exactly one
hit per query `:51`. There is no reference mask, no removal, no `-10e10`, no re-ranking that drops a
name. I grepped the file for the mechanisms that would do it: the only `reference_mask` in the
repository is on the CIRR path (`src/validate.py:162-166`, and `src/cirr_test_submission.py:98-103`).
That internal contrast — removal present for CIRR, absent for FashionIQ — is the strongest available
evidence that reference eligibility here is a deliberate property of this evaluator.

**Query construction.** `reference_name = self.triplets[index]['candidate']` (`src/data_utils.py:155`),
`target_name = self.triplets[index]['target']` `:166`; captions composed deterministically in
`src/validate.py:89-92` (`np.array(captions).T.flatten()` then
`f"{a.strip('.?, ').capitalize()} and {b.strip('.?, ')}"`), tokenized with
`clip.tokenize(input_captions, context_length=77)` `:93` and **no `truncate=True`** (an over-length
caption raises instead of being silently cut). The reference's own feature is taken from the gallery map,
`name_to_feat = dict(zip(index_names, index_features))` `:80` and `itemgetter(*reference_names)(name_to_feat)`
`:101-104` — so the reference is not merely *a* gallery member, it is *read out of* the gallery.

**Categories.** `shirt`, `dress`, `toptee`, evaluated in that literal order `:329`, `:334`, `:339`.
**Recall computation.** `recall_at10 = torch.sum(labels[:, :10]) / len(labels) * 100` `:54`;
`recall_at50` `:55` with `labels[:, :50]`. Macro = `statistics.mean` over the three appended values
`:331-342, :353-354` — an unweighted macro, not query-weighted.

**Does the protocol match the `reported_r10/r50` the registry transcribes?** The registry transcribes
**nothing** for this model (`null`/`null`/`null`, `models.yaml:688-690`), and that is correct — but for a
different reason than the nulls alone suggest, and this is where I disagree with the prior audits' framing.

The cited paper (arXiv 2308.11485, ACM TOMM 2023) prints TEN result tables. I fetched the full text
(`https://ar5iv.labs.arxiv.org/html/2308.11485`, 326,987 bytes) and parsed every `<table>` element:
**Table 1** is the FashionIQ val `CF × IFT × TFT` grid and its `Combiner | ✗ | ✗` row — the
out-of-the-box-CLIP Combiner, i.e. the no-fine-tuning configuration — is
`31.85 52.50 | 27.22 50.62 | 33.81 57.57 | avg 30.96 53.56`. **Table 2** is the same grid on CIRR
(`31.26 …` → `avg 30.96/53.56` equivalent). Tables 3–8 are ablations. §4.3 states it explicitly:
"Throughout all the experiments, we use the RN-50 CLIP model." **Table 9** (the FashionIQ SotA
comparison) contains exactly one RN-50x4 row, `Ours RN-50x4 Transformer` = `44.41 65.26 | 39.46 64.55 |
47.48 70.98 | avg 43.78 66.93`, and Table 9's own caption says the upper section holds methods "not
directly comparable to our proposed approach" — it is the comparison table, and its RN-50x4 row is the
fully fine-tuned model, not this variant.

So: **this variant's configuration is measured in the paper (Table 1, Combiner/✗/✗, 30.96/53.56), but
NOT on the paper's RN-50x4 backbone** — Table 1 is RN-50-only. There is no RN-50x4 frozen-encoder
FashionIQ row anywhere in the cited paper. The registry's `null`s are therefore **defensible**, and the
prior audit's statement that the paper "contains no out-of-the-box/FashionIQ table" is wrong in form
(Table 1 IS such a table, just on RN-50) though right in the conclusion that no RN-50x4 noft number
exists there. Severity P2: fix the prose, keep the nulls.

An author-published **noft RN-50x4** FashionIQ row does exist, in a paper the registry does not cite:
CVPR 2022 demo paper, `https://openaccess.thecvf.com/content/CVPR2022/papers/Baldrati_Effective_Conditioned_and_Composed_Image_Retrieval_Combining_CLIP-Based_Features_CVPR_2022_paper.pdf`
(fetched this pass, 4,392,102 bytes). Its Table 1 `Proposed approach` row is
`36.36 58.00 | 31.63 56.67 | 38.19 62.42 | avg 35.39 59.03`, on the Fashion-IQ **validation** set; §3.2
states "In the experiments, the CLIP encoders have been kept frozen and the only trained part of the
model is the Combiner function"; §3.2 also fixes `RN50x4`, 288×288 input, 640-dim features, target ratio
`1.25`; §4.3 states the category scoping. The Drive folder README attributes exactly this configuration
(item 2) to that paper — so the author, in two independent places, ties `fiq_comb_RN50x4_noft.pt` to
36.36/58.00/31.63/56.67/38.19/62.42/35.39/59.03. (Residual UNKNOWN: no per-file hash in either paper
ties the *published bytes* to that *published row*; the attribution is documentary, not cryptographic.)

Arithmetic I executed on those cells: shirt 33.995, dress 57.335, toptee 50.305 → macro 47.2117;
and the registry's own `reported_mean` convention, (35.39+59.03)/2 = 47.21. Both conventions agree.

**Protocol invariant preserved: no gallery, reference policy, caption input or noise condition is
proposed for change anywhere in this report.** `fashioniq_original_split` (Protocol A) is the registry's
declared `native_protocol` and the source confirms it.

---

## 8. WORKBENCH INTEGRATION

| component | state | evidence |
|---|---|---|
| adapter | present, real `command()` override | `adapters/models.py:86-94` (model_id, script, exact flags) |
| adapter command vs source | byte-level match of the flags this variant must pass; `--clip-model-path` correctly ABSENT | `:90-94` vs `src/validate.py:263-277`; contrast `:101-111` (fullft passes it) |
| planner / protocol | `native_protocol` + `supported_protocols` = `fashioniq_original_split`; unknown names raise | `models.yaml:686-687`; `src/evaluation/fashioniq.py:190-201` |
| checkpoint prep | `checkpoint_path` → `artifacts/checkpoints/combiner_rn50x4_noft/rn50x4-noft-unresolved.pt` (file shape); the real file name is `fiq_comb_RN50x4_noft.pt` and lives nowhere in the registry | `registry.py:842-863`; measured in §5.2 |
| preflight | reports 4 blockers (`.jpg`, base weights, link, interpreter) — see §10 | `workbench/backend/preflight.py:42-44, 86-100, 110-112`; output below |
| metric parser | **present** — `parse_combiner_output`, dispatched for this model_id, with macro self-consistency | `metrics_extraction.py:309-386`, dispatch `:406-407` |
| per-query support | none, and none possible: the evaluator prints only per-category and macro recalls | `src/validate.py:344-354` |
| benchmark reference | three committed rows, all `UNVERIFIED`, all-null → selection returns `NO_PUBLISHED_COMPARISON` (permanent) | `paper_benchmarks/papers.yaml:293-355`; observed selection below |
| replay code | `replay_status: REPLAY_CODE_IMPLEMENTED` (`models.yaml:667`), `replay_type: OFFICIAL_CLI` (`:670`) — there is no `workbench/replay/*` module for this model; the "replay" is the adapter command | `ls workbench/replay/` → limn/tgcir/publication only |

RUNTIME_VERIFIED checks I ran through the workbench's own read-only functions:

```
$ preflight_blockers(model, config)  -> 4 data/runtime blockers (full list in §10)
$ source_clean_and_pinned(model, checkout) -> (True, 'dfed9f748a8a…', None)
$ extract_aggregate_metrics('combiner_rn50x4_noft', <CVPR22 numbers as stdout>, paper_metrics=…)
  -> EXTRACTED_VERIFIED / clip4cir_stdout_recalls / {'r10': 35.3933, 'r50': 59.03, 'mean': 47.2117}
     / PARITY_VERIFIED / per-category {shirt 36.36/58.00, dress 31.63/56.67, toptee 38.19/62.42}
$ python_executable(model) -> RuntimeError: execution environment not verified [UNCONFIGURED/UNCONFIGURED]
$ ADAPTERS['combiner_rn50x4_noft']().build_command(request)
  -> FileNotFoundError: checkpoint missing: …/combiner_rn50x4_noft/rn50x4-noft-unresolved.pt
$ select_reference_outcome('combiner_rn50x4_noft') -> NO_PUBLISHED_COMPARISON (author_published_no_reference=True)
```

So the metric path is complete and demonstrably correct on the numbers that exist; what is missing is
everything upstream of it. **Correction to the prior audit**: its CF-5 ("no metric parser … AGGREGATE_PARSER_UNAVAILABLE") is stale — `parse_combiner_output` now exists and is dispatched (`metrics_extraction.py:309-386, :406-407`), and `test_adversarial_metrics.py:197-202` asserts both CLIP4Cir ids route to it.

---

## 9. CONFIRMED DEFECTS

**CF-1 — P1 — the checkpoint mapping is no longer "unverified"; the release resolved it and the
registry was not updated.** Root cause: the registry's checkpoint record encodes a Drive folder
(`models.yaml:707-712`) and a placeholder filename (`:705`) that no upstream artifact carries, plus
`expected_size_bytes: null` / `expected_sha256: null` (`:717-718`). Reproduction: `curl …/releases/tags/weights`
→ 5 digested assets; decode `part-000` → `FashionIQ/RN50x4_noft/fiq_comb_RN50x4_noft.pt`, 236,017,921
bytes, sha256 `b31882fe…`; strict-load across the pinned `Combiner` → all keys matched. Fix: record the
real filename, the measured size and digest, and the release URL; then the model's
`checkpoint_mapping_status` can move off `UNVERIFIED` and the `WorkbenchError("checkpoint_mapping_unverified")`
gate (`adapters/base.py:34-35`) stops firing. Regression evidence: none needed — this is a registry
data fix, and the strict-load probe above is the acceptance check.

**CF-2 — P1 — the "the Drive folder does not document the mapping" claim in three workbench documents
is false.** Root cause: the earlier audit read the JS-rendered Drive page (`grep -o '\.pt'` → 0) instead
of the `embeddedfolderview` endpoint, and generalised from that. Reproduction: the endpoint listings in
§5.1. Fix: correct `AUXILIARY_ASSET_AUDIT.md:221`, `cluster_b4_classic.yaml:79`,
`cluster_c_classic.yaml:61`; the remaining, real gap was the missing digest, which the 2026-10-12
release also closed.

**CF-3 — P0 — the dataset the evaluator would read does not hold the images it opens, and the failure
mode is silent.** `src/data_utils.py:158,161,170,176` open `f"{name}.jpg"`. OBSERVED:
`ls data/fashionIQ_dataset/images | sed 's/.*\.//' | sort | uniq -c` → `74381 png`, **0 jpg**; the
workbench's own image-extension gate reports `3817 of 3817 dress`, `6346 of 6346 shirt`,
`5373 of 5373 toptee` validation ids lacking `.jpg`. `models.yaml:662-663` asserts the root "opens
`.jpg` images there" and that the link is staged by an authorized step, but no step converts or aliases
extensions: `preparation_contracts.yaml:154` declares `generated_artifacts: []`. Worse, a missing image
does not raise: `src/data_utils.py:151-184` wraps every `__getitem__` in `except Exception as e: print(...)`
and returns `None`, and `collate_fn` (`src/utils.py:81-88`) filters `None` out of the batch — so a failed
*gallery* item silently shrinks the gallery and a failed *query* item silently changes the denominator.
Fix: stage a `.jpg`-extension tree (hardlink) as the declared generated artifact, or record the
decision explicitly. Do not run against `.png`.

**CF-4 — P1 — the declared dataset link is absent, and `EvalRequest.dataset_root` never reaches the
evaluator.** OBSERVED `test -e workbench/third_party/CLIP4Cir/fashionIQ_dataset` → absent. The adapter
(`adapters/models.py:90-94`) ignores `request.dataset_root` entirely and the source has no root flag
(`src/validate.py:263-277`), so the only viable path is that link, created by
`prepare_dataset.py:28-67` (refuses if the destination exists and is not the expected link `:54-56`,
refuses a wrong target `:49-50`, `symlink_to` only under authorization `:65`). Fix: run the authorized
staging step; do not add a hidden link.

**CF-5 — P1 — the interpreter does not exist, so no command can be constructed.**
`WORKBENCH_CLIP4CIR_PYTHON` unset; `python_executable(model)` raises
`execution environment not verified [UNCONFIGURED/UNCONFIGURED]`; `workbench/envs/` has no CLIP4Cir env;
`environment_assignment.json` assigns `limn` only. Fix: build and verify the declared 3.8 / torch 1.11.0 /
torchvision 0.12.0 environment (`models.yaml:693-700`), pinning `clip` to a commit sha (see §4).

**CF-6 — P2 — the base RN50x4 cache is absent, and the published digest has not been checked against
bytes.** `ls ~/.cache/clip` → absent. `auxiliary_assets.yaml:166-183` records URL + digest + size; the
digest is OpenAI's URL-convention name and upstream CLIP self-verifies against it, but no run on this
host has hashed the served object. Fix: stage through the workbench's manifest so the digest is recorded
from bytes, not from the URL.

---

## 10. REMAINING BLOCKERS

Precise classifications only; no `EXTERNAL_BLOCKER`.

| # | classification | owner | evidence | remediation |
|---|---|---|---|---|
| B1 | `FIRST_PARTY_ENVIRONMENT_CONFIGURATION` | workbench | `python_executable(model)` → `RuntimeError … WORKBENCH_CLIP4CIR_PYTHON=None`; `ls workbench/envs/` lacks a CLIP4Cir env | build + verify the declared 3.8/1.11.0/0.12.0 environment |
| B2 | `PUBLIC_ASSET_NOT_PREPARED` | workbench | preflight: `OpenAI CLIP RN50x4 base weights missing: /home/heheboiz/.cache/clip/RN50x4.pt`; `ls ~/.cache/clip` → absent | stage via the auxiliary-asset manifest |
| B3 | `PUBLIC_ASSET_NOT_PREPARED` | workbench | preflight: `canonical dataset link missing: …/CLIP4Cir/fashionIQ_dataset -> …/data/fashionIQ_dataset` | authorized `prepare_dataset.py --model combiner_rn50x4_noft` |
| B4 | `AUXILIARY_ASSET_MISSING` | workbench | preflight: 3817/6346/5373 validation images "lack .jpg"; canonical root is 74381 `.png` / 0 `.jpg` | prepare a `.jpg`-extension view of the canonical root (CF-3) |
| B5 | `PUBLIC_CHECKPOINT_NOT_MAPPED` | workbench | registry names `rn50x4-noft-unresolved.pt`, `download_url: null`, no digest (`models.yaml:705-718`) while the real member is `FashionIQ/RN50x4_noft/fiq_comb_RN50x4_noft.pt` (measured, §5.2) | record filename + size + digest + release URL; do not invent a local file name (`download_checkpoint` is out of scope here) |
| B6 | `SCIENTIFIC_PROTOCOL_UNVERIFIED` | workbench | the committed rows for this model are all `UNVERIFIED`/all-null (`papers.yaml:293-355`), so `select_reference_outcome` returns `NO_PUBLISHED_COMPARISON`; the only author-published noft RN-50x4 row is in the CVPR22 demo paper, which the registry does not cite | decide deliberately: cite the CVPR22 row as a sanity reference, or keep the permanent no-comparison verdict and say why (the registry already does the latter) |
| B7 | `GPU_RUNTIME_UNTESTED` | operator | `environment.cuda: UNKNOWN` (`models.yaml:699`), and `_requires_gpu` (`runtime.py:918-928`) treats UNKNOWN as "not a requirement" — correct here because `src/utils.py:15-18` falls back to CPU and no unconditional `.cuda()` exists in this path | optionally record `cuda` as not required rather than UNKNOWN |

No `FIRST_PARTY_BUG`, no `UPSTREAM_IMPORT_BUG`, no `UPSTREAM_CLI_MISSING`, no
`DEPENDENCY_VERSION_CONFLICT`, no `UNPUBLISHED_SOURCE_MODULE`, no `CHECKPOINT_UNPUBLISHED`, no
`CHECKPOINT_URL_INACCESSIBLE`, no `NONDETERMINISTIC_PREPROCESSING`, no `UNSAFE_CHECKPOINT_SERIALIZATION`
is established for this model. The checkpoint is published and the URL is reachable; the mapping is
simply not recorded.

---

## 11. IMPLEMENTATION PLAN

1. **Registry checkpoint record for `rn50x4_noft_unresolved`** (workbench, hours): real filename
   (`fiq_comb_RN50x4_noft.pt`), `expected_size_bytes: 236017921`,
   `expected_sha256: b31882fe6f7e3e194cb9cfda048926844ee0ccc9d4ebc5483fb3bacfa5972955`, `source_url`
   = the `weights` release asset, `source_type: author_github_release`, and
   `checkpoint_mapping_status: VERIFIED_METADATA`. Complexity: low. Scientific risk: none (data only).
   Expected effect: `checkpoint_missing_paths` becomes meaningful; the `checkpoint_mapping_unverified`
   refusal stops firing; `blocked_models` drops this model's checkpoint reason.
2. **Documentation correction** (workbench, minutes): CF-2.
3. **Dataset preparation step** (workbench, hours): add the `.jpg`-extension staging as a declared
   generated artifact for `clip4cir_fashioniq` (`preparation_contracts.yaml:154` is `[]` today) so CF-3
   is fixed where the contract says it should be. Complexity: low. Scientific risk: none — the pixel
   content is unchanged; only the basenames the evaluator opens are resolvable.
4. **Environment + assets** (operator, hours): B1, B2, B3, then a first real run.
5. Do NOT change the protocol, the gallery, the reference policy, the caption composition or the noise
   condition. Nothing in this plan touches Protocol A.

Expected outputs after 1–4: one command line (§3), six per-category numbers and two macro lines, parsed
by `parse_combiner_output` into `EXTRACTED_VERIFIED` with per-category evidence, and a gate verdict of
`NO_PUBLISHED_COMPARISON` (permanent) — by design, not by failure.

---

## 12. FINAL VERDICT

**BLOCKED** — with one strong positive correction.

Confirmed by independent evidence: pin exact and clean (`dfed9f748a8a…`, `source_clean_and_pinned` →
True); native CLI real and present; `fashioniq_original_split` correct (full ordered split, reference
eligible, no mask anywhere on the FashionIQ path); preprocessing exact (TargetPad 1.25 → 288, CLIP
normalization, no eval-time augmentation); the adapter command matches the audited invocation and
correctly omits `--clip-model-path`; the metric parser exists and was exercised successfully on the
only published noft numbers; the checkpoint is now **published, reachable, measured and strictly
loadable**.

Blocking: no verified interpreter, no base RN50x4 cache, no `<checkout>/fashionIQ_dataset` link, no
`.jpg` tree (with a silent `None`-skip path that would corrupt ranking rather than raise), and a
registry mapping that names a file that does not exist while the real one does.

`VERIFIED` alone would be wrong (nothing ran), `VARIANT` would understate a real blocker, and
`NOT_RUN`/`UNKNOWN` would understate what was established. BLOCKED is the precise state.

---

## 13. EVIDENCE

Repo: `https://github.com/ABaldrati/CLIP4Cir` · pinned `dfed9f748a8a4d05abf613164e06d59b22dbdf13`
· SNAPSHOT 2 `56bb547909d57bd8b5d2ef7ba55149962f77d482` (not adopted).

Files opened (pinned checkout unless noted): `README.md`, `src/validate.py` (360 lines, in full),
`src/combiner.py`, `src/data_utils.py`, `src/utils.py`, `src/combiner_train.py` (argparse + save sites),
`src/clip_fine_tune.py` (argparse + save sites), `src/cirr_test_submission.py` (CIRR reference mask);
workbench: `workbench/registry/models.yaml`, `registry/auxiliary_assets.yaml`,
`registry/assets/cluster_b4_classic.yaml`, `registry/audits/cluster_c_classic.yaml`,
`registry/preparation_contracts.yaml`, `registry/paper_benchmarks/papers.yaml`,
`registry/paper_benchmarks/extracted/clip4cir_tomm2023.json`, `backend/adapters/models.py`,
`backend/adapters/base.py`, `backend/registry.py`, `backend/preflight.py`, `backend/runtime.py`,
`backend/metrics_extraction.py`, `backend/benchmark_references.py`, `backend/paper_gate.py`,
`scripts/prepare_dataset.py`, `src/evaluation/fashioniq.py`, `tests/test_adversarial_metrics.py`,
`tests/test_release_blocker_invocation_proof.py`; hypotheses only: `docs/reviews/models/combiner_rn50x4_noft.md`.

Commands and observed results (read-only; every artifact under `/tmp`):

1. `git -C workbench/third_party/CLIP4Cir rev-parse HEAD` → `dfed9f748a8a4d05abf613164e06d59b22dbdf13`; `status --porcelain` → empty.
2. `git -C … ls-remote origin HEAD` → `56bb5479…`; `fetch origin --dry-run` → `dfed9f7..56bb547 master`, `[new tag] weights`; pin untouched.
3. `curl api.github.com/repos/ABaldrati/CLIP4Cir/compare/dfed9f7…56bb547` → ahead 1, `README.md +4 −4`.
4. `curl api.github.com/repos/ABaldrati/CLIP4Cir/releases/tags/weights` → 5 assets with `sha256:` digests and sizes quoted in §2.
5. `curl -r 0-1199999999 -o /tmp/p0z …/part-000` → `gzip -dc /tmp/p0z | tar -tv` → 5 members, noft member 236,017,921 bytes.
6. `/tmp/probe_ckpt.py` → extracted member sha256 `b31882fe6f7e3e194cb9cfda048926844ee0ccc9d4ebc5483fb3bacfa5972955`.
7. `/tmp/probe_load.py` → `weights_only=True` accepted; keys `['epoch','Combiner']`; 12 tensors; 59,003,521 params; dims as listed.
8. `/tmp/probe_strict.py` → `<All keys matched successfully>`; `combine_features` → `(3, 640)`, L2 = 1.0.
9. `/tmp/probe_mismatch.py` → RN50-recipe dims (1024/4096/8192) raise `RuntimeError: Error(s) in loading state_dict for Combiner` — a cross-backbone mismatch is loud; a same-backbone mismatch is silent (see the sibling report §5.3).
10. `preflight_blockers(model, config)` → the 4 blockers quoted in B2/B3/B4.
11. `source_clean_and_pinned(model, checkout)` → `(True, 'dfed9f748a8a…', None)`.
12. `extract_aggregate_metrics('combiner_rn50x4_noft', <CVPR22 stdout>, …)` → `EXTRACTED_VERIFIED`, `{'r10': 35.3933, 'r50': 59.03, 'mean': 47.2117}`, `PARITY_VERIFIED`.
13. `python_executable(model)` → `RuntimeError execution environment not verified [UNCONFIGURED/UNCONFIGURED]`.
14. `ls ~/.cache/clip` → absent; `ls workbench/envs/` → `README.md`, `csmcir-import.txt`; `ls workbench/artifacts/checkpoints/` → `.gitkeep`, `download_manifest.json`, `limn/`.
15. `ls data/fashionIQ_dataset/images | sed 's/.*\.//' | sort | uniq -c` → `74381 png`.
16. `curl ar5iv.labs.arxiv.org/html/2308.11485` → 326,987 bytes; 11 `<table>` elements parsed; Table 1 `31.85/52.50…`, Table 9 `44.41 … 43.78/66.93`.
17. `curl openaccess.thecvf.com/…/Baldrati_Effective_…CVPR_2022_paper.pdf` → 4,392,102 bytes; Table 1 `Proposed approach 36.36 58.00 31.63 56.67 38.19 62.42 35.39 59.03`; §3.2 frozen-encoder sentence.
18. Drive: `embeddedfolderview?id=1fnGeZQA…` → `RN50x4_noft` among six; `?id=1OHmPOU9…` → sole file `fiq_comb_RN50x4_noft.pt`; folder README fetch → 2449 bytes, keys/dims quoted.

---

## 14. ACHIEVED VERIFICATION LEVEL

**LEVEL 3 of 6.**

- Level 1 — identity, pin, registry consistency: achieved (STATIC_AUDIT + `source_clean_and_pinned`).
- Level 2 — static source + protocol auditing, independently evidenced, including the SNAPSHOT 2
  comparison: achieved (STATIC_AUDIT).
- Level 3 — the checkpoint artifact's bytes were actually obtained from the author release, the member
  was decoded, its size and SHA-256 were measured, its key set and tensor shapes were censused with the
  safe loader, and it was strict-loaded into the *pinned* `Combiner` class on this host: achieved
  (RUNTIME_VERIFIED, scratch only).
- Level 4 (a real evaluator run producing Recall@10/50) is NOT reached: no verified interpreter, no
  base RN50x4 cache, no dataset link, no `.jpg` tree.
- Levels 5–6 are out of reach.

Nothing in this report claims a metric was reproduced. The `EXTRACTED_VERIFIED` result in §8 is the
parser exercised on **published paper numbers typed as a synthetic stdout**, and is labelled as such —
it is evidence about the parser, not about the model.
