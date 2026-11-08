# Protocol findings — mismatches, wrong citations, conditional paths

Legend: **CONFIRMED** = verified from the pinned source (and executed where marked).
**REFUTED** = the claim does not hold against the pinned source. **UNVERIFIED** = could not be decided
from source + local data; say exactly why.

Environment note for every item: 16 repositories were cloned and detached at the registry pins
(`CLONE_MANIFEST.tsv`), working trees pristine. No upstream evaluator was executed — a cohort GPU
evaluation owns the card (pid 264541). So every "computes" statement is source-verified, not measured.

---

## 0. Headline

The user's three-way classification is **correct for all 17 variants**, and the registry's
`native_protocol` / `supported_protocols` agree with it on all 17. **No model needs a protocol
correction.** The real defects found are (a) four inaccurate line numbers, (b) three "entrypoint"
misstatements where the cited file is a library, not a runnable program, (c) one registry architecture-
level binding that cannot express a per-run protocol, and (d) two vendored files modified without a
registry `approved_source_patches` declaration.

---

## 1. Group assignment per repo — all CONFIRMED

| repo | gallery source | gallery count | reference | verdict |
|---|---|---:|---|---|
| CSMCIR | `split.<cat>.val.json` | 15536 | kept | A, CONFIRMED |
| MTST_PTHA | `split.<cat>.val.json` (+exists filter) | 15536 (0 dropped locally) | kept | A, CONFIRMED |
| CLIP4Cir | `split.<cat>.val.json` | 15536 | kept | A, CONFIRMED |
| SPRC | `split.<cat>.val.json` | 15536 | kept | A, CONFIRMED |
| TME | `split.<cat>.val.json`, per category | 3817/6346/5373 | kept (mask is CIRR-only) | A, CONFIRMED |
| AirKnow | `split.<cat>.val.json` via `original-split` | 15536 | masked | B, CONFIRMED |
| ConeSep | same | 15536 | masked | B, CONFIRMED |
| HABIT | same | 15536 | masked | B, CONFIRMED |
| INTENT | same | 15536 | masked | B, CONFIRMED |
| DCNet | `split.<cat>.val.json` via `val_trg` | 15536 | masked (both branches) | B, CONFIRMED |
| HINT | pair union | 8619 | masked | C, CONFIRMED |
| ENCODER | pair union (`val-split`) | 8619 | masked | C, CONFIRMED |
| PAIR | pair union (`val-split`) | 8619 | masked | C, CONFIRMED |
| CLVC-Net | pair union (`gallery_all=False`) | 8619 | masked | C, CONFIRMED |
| TG-CIR | pair union | 8619 | masked | C, CONFIRMED |
| LIMN (base/iter-0) | pair union | 8619 | masked (`name != 'birds'`) | C, CONFIRMED |

Counts EXECUTED: full split 3817/6346/5373/15536; pair union 2628/3089/2902/8619
(`build_fashioniq_gallery` on the real `data/FashionIQ` splits; identical to the independently counted
JSON lengths).

---

## 2. Wrong or imprecise citations — CONFIRMED

| # | repo | cited | what the cited line actually is | correct line |
|---|---|---|---|---|
| C1 | MTST_PTHA | `src/data_utils.py:164` | `# get the image names` (a comment) | gallery read `:167-168`; existence filter `:169-172` |
| C2 | DCNet | `data_loader/CE_dataset.py:24` | `self.no_target = (split == 'val_trg' or split == 'test_trg')` — the flag, not the load | the split read is `:27` |
| C3 | DCNet | `trainer/trainer_joint.py:215` | `# Delete reference == target` (a comment) | masks at `:217` and `:218` |
| C4 | CLVC-Net | `datasets.py:233` | `def get_test_targets(self):` (function header) | `:234` `if self.gallery_all:` / `:242` `else:` |
| C5 | HINT | `datasets.py:154` | `test_targets_id = []` (block sentinel) | construction `:154-165` (claim is right, the line is the head) |
| C6 | TG-CIR | `datasets.py:148` | `test_targets_id = []` (block sentinel) | construction `:148-159` |
| C7 | LIMN | `datasets.py:158` | `test_targets_id = []` (block sentinel) | construction `:158-169` |

C1–C4 are genuinely wrong (comment/flag/header instead of the operative statement). C5–C7 point at the
first line of the correct block, so the *claim* is correct while the *line* is a sentinel — flagged but
not a scientific error.

Accurate as cited (CONFIRMED): CSMCIR 419/24, CLIP4Cir 146/21, SPRC 155/24, TME 175/83,
AirKnow 218/151, ConeSep 201/68, HABIT 205/151, INTENT 203/150, ENCODER 185/68, PAIR 148/68,
TG-CIR mask 67, LIMN mask 82, CLVC mask 135.

---

## 3. Entrypoint claims that do not hold — CONFIRMED

| repo | user/implied entrypoint | reality |
|---|---|---|
| MTST_PTHA | `src/validate_blip.py` | its `main()` dispatches **CIRR only** (`:497-509`); no FashionIQ argument branch. Runnable path: `src/blip_validate.py:__main__` → `clip_finetune_fiq(...)` `:200` |
| SPRC | `src/validate_blip.py` | same defect: `main()` is CIRR-only (`:508-509`). Runnable path: `src/blip_validate.py:__main__` `:155` |
| AirKnow | `test_BLIP2.py` | has **no `__main__`** (0 occurrences); invoked as a module from `train_BLIP2.py:20,298` |
| ConeSep / HABIT / INTENT / TGCIR / LIMN / PAIR / ENCODER / HINT | `test.py` | **no `__main__`** in any of them (EXECUTED count = 0 each). Each is a bare `test(params, model, testset, category)` called from the repo's train script |
| CLVC-Net | README says `python test.py` | `test.py` has no `__main__`; the documented command is inert. Real driver `train.py:259` |

Consequence for us: for these repos "which entrypoint runs" is answered by **our replay wrapper**, which
is what the adapters already do (`launcher = "replay"`). The wrapper supplying the missing shell is
legitimate; the registry/docs should not describe the upstream file as a runnable evaluator.

---

## 4. Conditional paths whose runtime value matters — CONFIRMED

| repo | condition | runtime value on the FashionIQ default path | effect |
|---|---|---|---|
| TME | `if category == 'CIRR':` before the `10e10` mask (`precompute_evaluation.py:83`) | `category` ∈ {dress,shirt,toptee} for FashionIQ (`:15-17`) | mask **never** runs → reference kept |
| CLVC-Net | `if self.gallery_all:` (`datasets.py:234`) | `False`: `__init__` default `:153`, and `train.py:86-96` never passes it; `grep gallery_all` finds only `datasets.py:153,163,234` | pair-union branch |
| ENCODER | `if self.split == 'val-split':` (`datasets.py:185`) | `'val-split'`: `__init__` default `:64`, `train.py:35`, `evaluate_model.py:15` | pair-union branch |
| PAIR | `if self.split == 'val-split':` (`datasets.py:148`) | `'val-split'`: `__init__` default `:23`, `train.py:29` | pair-union branch |
| HABIT | `self.split` from ctor default (`:28`) | `'original-split'`; `train.py:109-112` omits it | full split |
| INTENT | same, but `train_INTENT.py:61 --split default='original-split'` forwarded at `:99` | `'original-split'` | full split |
| LIMN | `if name != 'birds':` around the mask (`test.py:80`) | `name` ∈ {dress,shirt,toptee} | mask runs |
| AirKnow / ConeSep | `self.split` **hard-coded** `'original-split'` (`:31` / `:28`) | no override path exists | full split |
| MTST_PTHA | `if not file_path.exists():` (`:169-172`) | 0 missing locally | filter is a no-op here, not guaranteed elsewhere |
| DCNet | `split.split('_')[0]` on `'val_trg'` (`CE_dataset.py:27`) | `'val'` | full val split file |

---

## 5. Registry defect: protocol bound to the model, not to the run — CONFIRMED (design), FAIL-SAFE today

`workbench/registry/models.yaml` stores `native_protocol` per **model**, and the run protocol is
`args.protocol or model["native_protocol"]` (`scripts/evaluate_models.py:2738`, `scripts/pipeline.py:1208`).
Two separate facts:

- **`--protocol` can be passed** (`evaluate_models.py:519`, choices = registry protocols) and is then
  checked against `supported_protocols` (`backend/runtime.py:3075`, `adapters/base.py:28`). So the run is
  not *forced* onto the native protocol.
- **But every replay wrapper hard-codes its protocol as a module constant** and never reads the request:
  `PROTOCOL = "fashioniq_val_split"` (`encoder.py:48`, `clvc_net.py:46`, `hint.py:65`, `pair.py:44`),
  `PROTOCOL = "fashioniq_full_gallery_ref_excluded"` (`airknow.py:51`, `habit.py:46`, `intent.py:46`,
  `dcnet.py:25`), and `limn.py:243` passes the literal `protocol="fashioniq_val_split"`.
  `_replay_command` (`adapters/models.py:44-70`) builds the child argv and **does not pass any protocol
  flag**; no replay file has an `add_argument("--protocol")` (grep: 0 hits).

Therefore `--protocol` on a replay-launched model changes what the *plan/proof metadata* claims while the
child still computes its constant cohort. Today this cannot mislabel a completed run, because
`supported_protocols` is a single element for every model — the only accepted `--protocol` equals the
constant. It becomes a silent mislabel the moment any model is given a second supported protocol (e.g.
"these models can also be scored on our release gallery"). Correct fix is a per-run protocol plumbed into
the wrapper's argv and emitted in its proof payload, not a per-model constant. **Registry correction
proposed? No** — this is code, not `models.yaml`; see §6 for the one-line registry-side guard I do
recommend (currently unnecessary because each `supported_protocols` has one entry).

Rank-arithmetic check (EXECUTED, our evaluator, gallery `[REF, WRONG, TARGET]`, scores 0.9/0.8/0.7):
target rank 3 when the reference is kept (`recall_at_k(...,k=3)=100`, `k=1` = 0) and rank 2 when excluded
(`recall_at_k_reference_excluded(...,k=2)=100`, `k=1` = 0). Trap 8's arithmetic is confirmed against our
own code.

Cross-protocol comparison guard (CONFIRMED): `backend/analysis.py:11-15` `require_same_protocol` raises
`WorkbenchError("cross_protocol", ...)` when a run set mixes protocol ids; `paper_gate.py` and
`benchmark_references.py:283-291` refuse a reference row whose `gallery_protocol` ≠ the run's. So a Group A
number cannot be averaged with a Group B number through the supported analysis path.

---

## 6. Trap-by-trap verdicts

1. **All three native groups use validation queries; `original_split` is not a training gallery —
   CONFIRMED.** Every Group A read is `split.<cat>.val.json` or `cap.<cat>.val.json`; TME's train-split
   pair-union branch (`data.py:162-171`) is unreachable at eval because `precompute_test.py:139` and
   `precompute_train.py:81` ask for `'val'`. Registry already records this (models.yaml TME note).
2. **`original-split` naming does not imply reference-kept — CONFIRMED.** AirKnow, ConeSep, HABIT and
   INTENT all set `self.split = 'original-split'` and all four mask the reference (`test*.py:151/68/151/150`).
   DCNet is the same shape under a different name (`val_trg`). Repos where the name is a true Group A:
   CSMCIR, MTST_PTHA, CLIP4Cir, SPRC, TME (none of them has a split *named* `original-split` except via
   the `'val'` literal — their group comes from the gallery file, not a label).
3. **Protocol attach-to-run vs attach-to-model — CONFIRMED defect, fail-safe today.** §5.
4. **Branch selections — CONFIRMED**, all six resolved: ENCODER `val-split` (default in both entry
   points), PAIR `val-split` (default), HABIT `original-split` (ctor default, not overridden), INTENT
   `original-split` (`--split` default, forwarded), CLVC-Net `gallery_all=False` (default, never set),
   LIMN `name != 'birds'` → mask on.
5. **"Reference excluded" is a ranking-time candidate rule only — CONFIRMED, and our code does not
   over-apply it.** Upstream: the -10e10 write is into the score matrix just before `argsort`; the
   reference image is still the model's query input in every one of these repos. Ours:
   `src/evaluation/fashioniq.py:336-342` passes `reference_ids` only when
   `resolved.reference_excluded`, and `recall_at_k_reference_excluded` (`:35`) drops the *column* at rank
   time (`:70-73`: `reference_column = gallery_index[reference_id]` … `ranking[ranking != reference_column]`);
   the model still receives `batch.reference_ids` features (`:305-306`) in every protocol. Composition adapters go further and assert non-inheritance:
   `composition/model_encoder.py:41,223`, `composition/adapters.py:409-453`.
6. **No CIRR mask leakage in either direction — CONFIRMED** by function-level scan of every file that has
   both evaluators. `mask=True` is confined to `compute_cirr_val_metrics` / `get_results` /
   `test_cirr_valset` (CSMCIR, MTST_PTHA, CLIP4Cir, SPRC) and to TME's `if category == 'CIRR':`.
   `compute_fiq_val_metrics` is `mask=False` in all five Group A files. Reverse direction: the Group B/C
   FashionIQ `test()` mask is unconditional on FashionIQ categories and does not consult the CIRR path.
7. **PTHA's loader drops absent images — CONFIRMED, and it is a no-op on this host.** Filter at
   `MTST_PTHA/src/data_utils.py:169-172`, keyed on `<id>.png`, so it is an extension test as much as an
   existence test. EXECUTED on the real root: 0 of 3817/6346/5373 val ids lack a `.png`, so gallery
   membership = 15536 = the split file. Our replay calls the *native* `FashionIQDataset` for the gallery
   (`replay/ptha_mtst.py:365-366`), so it reproduces the filter by construction rather than re-implementing
   it — including the drop, if a host ever has a missing file.
8. **`[reference, wrong, target]` → rank 3 vs rank 2 — CONFIRMED** (§5, executed), and the cross-protocol
   comparison guard exists (§5). No Group A number can reach a Group B mean through
   `require_same_protocol`.

---

## 7. Vendored-vs-clean divergence

Method: 16 clean clones diffed against `workbench/third_party/<dir>` with `--exclude=.git
--exclude=__pycache__ --exclude='*.pyc'` (full unified diffs: `raw/vendored_diffs.txt`). Nine repositories
are byte-identical apart from generated caches and declared data links: **SPRC, CLIP4Cir, LIMN, TGCIR,
HINT, HABIT, INTENT, ConeSep** (8 of them), plus **PAIR** and **CLVCNet** whose only *code* deltas are
non-evaluator files. The user's "4 of 33 key files" undercounts: I find **7 modified files + 2 added
files across 7 repositories**.

| repo | file | nature | touches gallery/reference semantics? |
|---|---|---|---|
| CSMCIR | `src/validate_blip_csmcir.py` | query batch size env override; `--visualize` opt-in for the post-metric dump | **NO** — metrics are printed before the dump; batch size only |
| CSMCIR | `src/lavis/models/__init__.py` | guarded imports + re-registration of an author-published bytecode class | **NO** |
| MTST_PTHA | `MTST_PTHA/src/data_utils.py` | `base_path = Path(os.environ.get("MTST_DATA_ROOT", "path_to_data_dir"))` | **NO** — root only; the gallery read (`:167`) and the existence filter (`:169-172`) are untouched |
| MTST_PTHA | `MTST_PTHA/src/lavis/models/__init__.py` | `try/except ImportError` around one absent module | **NO** |
| TME | `src/precompute_evaluation.py` | `Qformer_bs`/`query_bs` env overrides, documented defaults 256/32 preserved | **NO** — the `if category == 'CIRR'` mask block is not in the diff |
| TME | `src/precompute.py` | preallocate + slot-copy instead of list + `torch.cat` | **NO** — same order, same dtype; `train_gallery` membership untouched |
| TME | `src/utility.py` | fail loudly when CUDA is absent (upstream silently fell back to CPU) | **NO** |
| TME | `precompute.sh` | drops a stray background `nohup` CIRR job, documents the foreground FashionIQ command | **NO** |
| AirKnow | `data_utils.py` | `_resolve_img_path` falls back `.jpg` → `.png` | **NO** — and it is *unreachable* from the evaluator: `datasets.py` does not import `data_utils` (grep: 0 hits); only `train_BLIP2.py:23` and `cirr_sub_BLIP2.py:99` do |
| AirKnow | `lavis/models/__init__.py` | `try/except ImportError` for `blip_cir` | **NO** |
| ENCODER | `evaluate_model.py` | `import datasets1` → `import datasets as datasets1` | **NO** — name binding only |
| ENCODER | `datasets1.py` (**added**, absent upstream) | 1 line: `from datasets import *` | **NO** |
| CLVCNet | `resnet.py` | `torch.hub.load_state_dict_from_url` with torchvision fallback | **NO** — weights loader |
| DCNet | `base/base_trainer.py` | `weights_only=False` at one checkpoint load | **NO** |
| PAIR | `models/MHTransformer_rev.py` (**added**) | reconstructed module the checkpoint needs | **NO** — model definition |

Verification of the two *declared* patch sets (EXECUTED, sha256 of every live file against the registry):
all five `approved_source_patches` entries (CSMCIR ×2, DCNet ×1, TME ×2) hash to their declared
`derived_sha256`. All 13 `provenance.local_fixes` digests match their declared `delivered_sha256`.

**Defect (CONFIRMED):** `MTST_PTHA` and `AirKnow` carry *modified* files that are **not** declared in
`models.yaml` `approved_source_patches` (their `approved_source_patches` lists are empty) — the changes
are only recorded in `third_party/<repo>/provenance.json` `local_fixes`. `vendored_source.py:218-232`
accepts a file when it matches either digest, so integrity checks still pass; the gap is that the
registry — the document a reader checks for "what did we change" — says nothing. `ENCODER/datasets1.py`
and `PAIR/models/MHTransformer_rev.py` are added files with the same status (declared only in
provenance, and `preflight.py:119,321` already expects `datasets1.py`). No gallery/reference change is
hidden by any of them.

---

## 8. Points that remain UNVERIFIED, and why

| item | why unverified |
|---|---|
| **Numbers.** Whether any repo's *reported* R@10/R@50 was actually produced by the code path I read. | Needs each repo's environment + checkpoints on GPU; the card is owned by the running cohort evaluation. I verified the scorer's rules, not the artefacts that were scored. |
| **TME gallery membership at run time.** | `data.py:250-255` loads `features/FashionIQ/val_images.pt` + `val_name_to_idx.json`; locally those are a 1-byte file and `{}` (recorded in `CIRCUS_COMPOSITION_SPEC.md`). The code fixes membership to that cache; I cannot read the author's cache. |
| **DCNet `val_trg` ordering vs `id2targetidx`.** | The mask indexes `id2targetidx[elem['candidate']]`, built in the same loop that concatenates `val_experts` — order is consistent by construction, but I did not execute it, so I am not asserting the index arithmetic numerically. |
| **Whether a `gallery_all=True` run ever produced a published CLVC-Net number.** | No release artefact in the pinned tree records the flag; the checkpoint is unresolved (`author_drive_unresolved`). |
| **AirKnow / ConeSep / HABIT / INTENT: which of `test.py:test` vs any offline scoring script produced the paper tables.** | Those repos ship no scoring CLI and no result JSON; the only call site present is the training-loop validation. A paper may have used code not in the repo. |
| **PAIR / HINT / TGCIR / CLVC-Net / SPRC / CLIP4Cir / CSMCIR / PTHA checkpoints** | Mapping status unchanged by this audit; registry already marks PAIR/SPRC variants `UNVERIFIED` and CLVC-Net `author_drive_unresolved`. |
| **`LIMN+` protocol.** | Same code shape as LIMN base (pair union + `name != 'birds'` mask), but the registry registers no LIMN+ variant, so no row is claimed. Explicitly not extended, as the user required. |
