# ptha_mtst — reproducibility reassessment (independent, adversarial, READ-ONLY)

Auditor: independent reassessment pass. Nothing under `/home/heheboiz/data/cir` was modified except
this file. All probe output went to `/tmp`. No GPU work; there is no pinned checkout for this model, so
nothing here can be RUNTIME_VERIFIED and nothing is presented as such. Prior audits
(`workbench/docs/reviews/models/ptha_mtst.md`, `final_model_audit/ptha_mtst.md`) were treated as
hypotheses and re-tested against live upstream and the registry.

**Headline, because it must not be missed: the repository the registry links is the model's OWN
source, authored by the paper's first author — it is NOT a different project's repo.** Details in §2.

---

## 1. MODEL IDENTITY

| field | value | evidence |
|---|---|---|
| model_id | `ptha_mtst` | `workbench/registry/models.yaml:217` |
| method | "PTHA + MTST" | `models.yaml:218` |
| paper | "Scale Up Composed Image Retrieval Learning via Modification Text Generation" | `models.yaml:222` |
| paper URL | https://arxiv.org/abs/2504.05316 | `models.yaml:223` |
| year | 2025 | `models.yaml:220` |
| source url (recorded) | https://github.com/zyinan99/MTST_PTHA | `models.yaml:228` |
| branch (recorded) | `main` | `models.yaml:229` |
| pinned commit | `null` | `models.yaml:230` |
| source_available | `false` | `models.yaml:231` |
| source_dir | `null` | `models.yaml:232` |
| native_protocol | `null` | `models.yaml:234` |
| supported_protocols | `[]` | `models.yaml:235` |
| reported metrics | R@10 54.99, R@50 76.28, mean 65.64 | `models.yaml:236-238` |
| reproduction_status | `BLOCKED_NO_VERIFIED_FASHIONIQ_CHECKPOINT` | `models.yaml:239` |
| checkpoint_variants | `[]` | `models.yaml:241` region (see §5) |

Paper identity verified this pass from the primary source: `read https://arxiv.org/abs/2504.05316` →
title identical, authors `Yinan Zhou, Yaxiong Wang, Haokun Lin, Chen Ma, Li Zhu, Zhedong Zheng`,
submitted 2025-02-21, categories cs.IR/cs.AI/cs.CV. The abstract defines MTST = "Modification
Text-oriented Synthetic Triplets" (a generator + synthetic pretraining/finetuning data) and describes "a
two-hop alignment strategy" (the PTHA retrieval mechanism). So the registered method name denotes **a
data-generation pipeline plus a retrieval network**, not one downloadable artifact. STATIC_AUDIT.

## 2. SOURCE STATUS

Registry values quoted **verbatim** (`models.yaml:217-241`):

```
217:  - model_id: ptha_mtst
218:    method_name: PTHA + MTST
219:    research_generation: recent
220:    publication_year: 2025
221:    preparation_contract: unavailable
222:    paper_title: Scale Up Composed Image Retrieval Learning via Modification Text Generation
223:    paper_url: https://arxiv.org/abs/2504.05316
224:    # Independence review (workbench/docs/reviews/models/ptha_mtst.md) refuted the earlier "no source"
225:    # conclusion: an author repository exists. It is recorded as evidence, not registered as a source_dir
226:    # or a commit pin, because registering it would claim a checkout the workbench has not synced, audited
227:    # or verified. Sync and audit it before promoting it to `source_dir`.
228:    upstream_repo_url: https://github.com/zyinan99/MTST_PTHA
229:    upstream_default_branch: main
230:    upstream_commit_sha: null
231:    source_available: false
232:    source_dir: null
233:    source_blocker: "author repository https://github.com/zyinan99/MTST_PTHA exists and is reachable (README title matches the paper, commits authored by the paper's first author Yinan Zhou, HEAD 6a02ee4763fc2260b6abfb18be78853facb91d86 at 2025-02-09) but is not synced, pinned or audited here, and no FashionIQ fine-tuned checkpoint is verified; the earlier 'no source exists' note was incomplete"
234:    native_protocol: null
235:    supported_protocols: []
236:    reported_r10: 54.99
237:    reported_r50: 76.28
238:    reported_mean: 65.64
239:    reproduction_status: BLOCKED_NO_VERIFIED_FASHIONIQ_CHECKPOINT
240:    strict_clean_checkpoint_available: false
241:    notes: No author-linked FashionIQ fine-tuned checkpoint verified. CIRR pretraining evidence is not eligible.
```

**Registry ↔ machine, checked, not assumed** (executed, read-only):

```
$ .venv/bin/python -c "from workbench.backend.registry import load_registry; …"
source_available = False
upstream_repo_url = 'https://github.com/zyinan99/MTST_PTHA'
upstream_commit_sha = None
native_protocol = None
supported_protocols = []
source_dir = None
checkpoint_variants = []
preparation_contract = 'unavailable'
reproduction_status = 'BLOCKED_NO_VERIFIED_FASHIONIQ_CHECKPOINT'
strict_clean_checkpoint_available = False
reported_r10 / r50 / mean = 54.99 / 76.28 / 65.64
```

- **Confirmed: no pin, no `source_dir`, no protocol, empty checkpoint list.** The `upstream_repo_url`
  is present but the registry explicitly declines to treat it as a synced source (comment lines
  224-227). That is an honest and internally consistent state.
- **No checkout exists in the tree.** `ls workbench/third_party/` → `AirKnow CLIP4Cir CLVCNet CSMCIR
  ConeSep DCNet ENCODER HABIT HINT INTENT LIMN PAIR SPRC TGCIR` (14 entries; none is MTST/PTHA).
  Registry ↔ filesystem consistent.
- Why unregistered rather than wrong: registering `upstream_repo_url` while leaving
  `source_available: false` is the correct way to record *evidence* without claiming an audited
  checkout. My independent reading agrees with the registry here — and I note this because the prior
  audit's "REFUTED (P1)" framing of the same facts (`workbench/docs/reviews/models/ptha_mtst.md`,
  which was written when the registry had `upstream_repo_url: null`) is **now stale**: the current
  registry entry already carries the discovery. The hypothesis has been absorbed;
  re-asserting it as an open defect would be wrong.

### SNAPSHOT 2 — current upstream state (no pin exists, so this is the only state)

Because there is no pin, `ls-remote` is not merely a snapshot comparison — it is the **only** access
point. Executed this pass:

```
$ git ls-remote https://github.com/zyinan99/MTST_PTHA
6a02ee4763fc2260b6abfb18be78853facb91d86	HEAD
6a02ee4763fc2260b6abfb18be78853facb91d86	refs/heads/main

$ git ls-remote --heads https://github.com/zyinan99/MTST_PTHA
6a02ee4763fc2260b6abfb18be78853facb91d86	refs/heads/main
```

Exactly one branch (`main`), one SHA, and it equals the SHA the registry's `source_blocker` records
(`models.yaml:233`). `git ls-remote` was used for the whole probe; **no clone was performed and no
pin was created or changed.**

## 3. IS `https://github.com/zyinan99/MTST_PTHA` THE MODEL'S OWN SOURCE?

**YES — the model's own source repository.** It is not a different project's repo, not a mirror, and
not a fork. Evidence, each item fetched this pass:

| test | command / URL | observed result |
|---|---|---|
| title identity | `read https://raw.githubusercontent.com/zyinan99/MTST_PTHA/main/README.md` | line 1 is exactly `# Scale Up Composed Image Retrieval Learning via Modification Text Generation` — the registered `paper_title` (`models.yaml:222`) |
| author identity | `read https://api.github.com/users/zyinan99` | `"name": "Yinan Zhou"` — the paper's first author |
| commit authorship | `read https://api.github.com/repos/zyinan99/MTST_PTHA/commits?per_page=10` | every commit authored/committed by `Yinan Zhou <1359149292@qq.com>` or `80240887+zyinan99@users.noreply.github.com` |
| provenance | `read https://api.github.com/repos/zyinan99/MTST_PTHA` | `created_at 2024-12-26T06:16:27Z`, `pushed_at 2025-02-09T10:52:31Z`, `size 7906`, `license MIT`, `fork: false`, `stargazers 0`, `default_branch main`, `open_issues 0` |
| paper title inside repo | `read …/api.github.com/repos/zyinan99/MTST_PTHA/git/trees/main` | top-level entries `MTST_PTHA/`, `README.md`, `requirements.txt`, `LICENSE`, `.gitattributes`, `.gitignore` |
| code is the paper's | `read .../MTST_PTHA/src/lavis/models/__init__.py` (raw, at SHA `6a02ee47…`) | imports `blip2_qformer_tgir_concatembedding_sim`, `blip2_qformer_tgir_concatembedding_sim_zyn`, `blip2_vicuna_instruct_tgir_modifiercaption`; `__all__` includes `Blip2Qformer4TGIRCESIM`, `Blip2Qformer4TGIRCESIMZYN`, `Blip2VicunaInstructTGIR4MC` |
| model class registered | `read .../blip2_qformer_tgir_concatembedding_sim.py` (raw, SHA `6a02ee47…`) | `@registry.register_model("blip2_tgir_ce_sim")` → `class Blip2Qformer4TGIRCESIM`, whose `forward()` consumes `samples["text_rev"]` (reverse modification text), computes a two-hop composed query (`composed_query_output` → `sequence_output` → second `Qformer.bert(query_embeds=sequence_output)`), an implicit-prototype loss `loss_p2p = mseloss(p1,p2.detach())`, and `loss_itc`/`loss_rtc` — i.e. exactly the paper's PTHA two-hop alignment and the reverse-modifier usage |
| README training recipe names that class | README (fetched) | `--blip-model-name 'blip2_tgir_ce_sim'`, `--backbone {"pretrain_mtst" for pretrained model using mtst, "pretrain" for initialized model from blip-2}` |
| MTST-specific config exists | `read .../MTST_PTHA/src/lavis/configs/models/blip2/blip2_pretrain_mtst.yaml` (raw, SHA `6a02ee47…`) | present; `pretrained: "PATH_TO_PRETRAINED_WEIGHTS"` in the **payload** it downloads |
| MTST-specific loader exists | `read .../MTST_PTHA/src/lavis/models/blip2_models/blip2_qformer_tgir_concatembedding_sim.py` (raw, SHA `6a02ee47…`) | `PRETRAINED_MODEL_CONFIG_DICT = {"pretrain": "configs/models/blip2/blip2_pretrain.yaml", "pretrain_mtst": "configs/models/blip2/blip2_pretrain_mtst.yaml"}` |
| evaluation path exists | HTTP probes at the **pinned-to-nothing** SHA `6a02ee4763fc2260b6abfb18be78853facb91d86` | `MTST_PTHA/src/blip_validate.py` → `200`; `.../src/data_utils.py` → `200`; `.../src/validate_blip.py` → `200`; `.../src/utils.py` → `200` |

**Answer: this is the model's own official source** (self-published by the first author, MIT-licensed,
0 stars, never forked). It is *not* a different paper's repository, which is the outcome the assignment
asked me to be alert for; I report the opposite and here is why I am confident: the README title is the
paper's title verbatim, the account is the first author, and the code contains the paper's two named
contributions (MTST synthetic-triplet generator usage and PTHA two-hop alignment) as registered model
classes.

Residual caution, stated rather than glossed: the repository is **not** a full reproduction bundle on
its own — see §5/§10. It is the right repo; it is not sufficient by itself.

## 4. NATIVE EVALUATION WORKFLOW (from the only reachable revision)

No pin exists, so the following is read from the current `main` HEAD
`6a02ee4763fc2260b6abfb18be78853facb91d86` and labelled **STATIC_AUDIT / UNPINNED**. It must be
re-derived once a pin is recorded.

Argument parser, from the raw file (byte-verified this pass; `sha256sum /tmp/mtst_blip_validate.py` →
`fb040a92e9534405e13b5992dc6686ea650b1925e5931442462d1c25a76655bb`, 9002 bytes):

```
186 if __name__ == '__main__':
187     parser = ArgumentParser()
188     parser.add_argument("--dataset", type=str, required=True, help="should be either 'CIRR' or 'fashionIQ'")
189     parser.add_argument("--blip-model-name", default="blip2_cir_rerank_learn", type=str)
190     parser.add_argument("--backbone", type=str, default="pretrain", help="pretrain for vit-g, pretrain_vitL for vit-l")
191     parser.add_argument("--model-path", type=str)
193     args = parser.parse_args()
194     if args.dataset.lower() not in ['fashioniq', 'cirr']:
195         raise ValueError("Dataset should be either 'CIRR' or 'FashionIQ")
197     if args.dataset.lower() == 'cirr':
198         blip_validate_cirr(args.blip_model_name, args.backbone, args.model_path)
199     elif args.dataset.lower() == 'fashioniq':
200         clip_finetune_fiq(['dress', 'toptee', 'shirt'], args.blip_model_name, args.backbone, args.model_path)
```

This parser is byte-for-byte the same shape as SPRC's (`workbench/third_party/SPRC/src/blip_validate.py:141-155`),
including the same **unregistered default** `blip2_cir_rerank_learn`. README's Evaluation block passes
only `--dataset`, `--blip-model-name {trained model name}`, `--model-path {for path}` — notably **no
`--backbone`**, whose default `pretrain` maps to `configs/models/blip2/blip2_pretrain.yaml`, whereas the
MTST fine-tuning instructions require `--backbone pretrain_mtst`. So the published evaluation recipe,
run as written, would load the wrong backbone config for an MTST-pretrained weight. STATIC_AUDIT; P2.

Data contract (STATIC_AUDIT, UNPINNED): `MTST_PTHA/src/data_utils.py:15` is
`base_path = Path("path_to_data_dir")` — a literal placeholder the README instructs the operator to
edit; there is no CLI data-root option. It opens
`fashionIQ_dataset/captions/cap.{cat}.{split}.json` (`:153`), a **line-delimited** extra training file
`cap_{split}_{cat}_rev.json` (`:157`), `fashionIQ_dataset/image_splits/split.{cat}.{split}.json`
(`:167`), and images with a **hard-coded `.png`** suffix (`:170,:206`). It also *silently drops* split
ids whose `.png` is missing (`:169-172`), which would change the candidate universe without an error —
a determinism hazard, P2 (UNPINNED).

Metric path (STATIC_AUDIT, UNPINNED): `MTST_PTHA/src/validate_blip.py:24-57` —
`distances = 1 - pred_sim` (`:44`), ascending `argsort` (`:45`), labels by target equality (`:49-50`),
one-hit assert (`:51`), `recall_at10`/`recall_at50` over `len(labels)` (`:54-55`). No `reference_mask`
on the FashionIQ path; the mask appears only in the CIRR helpers (`:258-260`, `:331-333`).

**Protocol implication — which I deliberately do NOT register.** The code shape above is consistent
with the workbench's `fashioniq_original_split` definition (`src/evaluation/fashioniq.py:172-174`:
`gallery="full_split"`, `reference_excluded=False`). But this is derived from an **unpinned** tree whose
`.png`-filtering, `.replace("jpg","png")` id rewriting and line-delimited training file can each change
what "the full split" means. Registering it now would convert an unpinned inference into a protocol
claim. See §8.

## 5. CHECKPOINTS AND AUXILIARY ASSETS

Registry: `checkpoint_variants: []` (`models.yaml:241` region; OBSERVED `[]` from the loader),
`strict_clean_checkpoint_available: false` (`:240`), `notes:` "No author-linked FashionIQ fine-tuned
checkpoint verified. CIRR pretraining evidence is not eligible." (`:241`).

Independent hunt, re-run this pass (network metadata only; **no weight byte was downloaded**):

| artifact | path | size | LFS sha256 | evidence |
|---|---|---|---|---|
| pretrained (CIRR) | `tuned_mtst_cirr.pt` | 2,720,352,973 | `1c4dab86a9f66dfe10aff75b39790432eb6cc746cf27b6cfab74b66d93b7588f` | `read https://huggingface.co/api/models/yinanzhou1/mtst_ptha/tree/main?recursive=true&expand=true` |
| generated pretraining data (CIRR) | `cirr_modifier_generated_5p_completed_rev.json` | 298,419,312 | `6cb4c02170f703da5ae40ae6d988a6aae272c2fa821d30c1d090de671ddfbd4d` | same |
| generated finetuning data (CIRR) | `cirr_train_modifier_rev_long.json` | 15,591,788 | `b1b7e72b3416e4479e4ddb9a807e789210e8d4f9d32210da460a22882f90d15d` | same |

Hub repo `yinanzhou1/mtst_ptha`: `sha b5e669c8ef28b0084f43398d184a5a0116903ccf`, `lastModified
2024-12-26T07:42:49Z`, `gated false`, `usedStorage 3034364073`, tags `license:apache-2.0`, no model card
(`read https://huggingface.co/api/models/yinanzhou1/mtst_ptha`). Note the repo SHA's **last commit
message is a rename** (`"Rename cirrr_modifier_generated_5p_completed_rev.json to
cirr_modifier_generated_5p_completed_rev.json"`) — the artifact was republished once.

The upstream README (lines 85-87, fetched this pass) links **exactly three** downloads, and all three
are the CIRR artifacts above. There is **no FashionIQ weight link anywhere in the README**.

Probes I ran to try to falsify the registry's `BLOCKED_NO_VERIFIED_FASHIONIQ_CHECKPOINT`:

```
$ curl -I -L "https://huggingface.co/yinanzhou1/mtst_ptha/resolve/main/tuned_mtst_cirr.pt"     → 200
$ curl -I -L "https://huggingface.co/yinanzhou1/mtst_ptha/resolve/main/fashioniq.pt"           → 404
$ curl -I -L "https://huggingface.co/yinanzhou1/mtst_ptha/resolve/main/tuned_mtst_fashioniq.pt" → 404
$ curl "https://huggingface.co/api/models?search=mtst&limit=20"        → only `yinanzhou1/mtst_ptha` is CIR-related
$ curl "https://huggingface.co/api/models?search=ptha&limit=20"        → no MTST/PTHA CIR model
$ curl "https://huggingface.co/api/models/yinanzhou1/mtst_ptha/discussions" → {"discussions":[],"count":0}
```

Auxiliary assets: the FashionIQ benchmark itself and a BLIP-2/ViT-g backbone. `preparation_contract:
unavailable` (`models.yaml:221`) means the registry asserts **no** data requirement for this model, so
`model_requirements(...)` returns empty blocks — the absence is enforced by the registry's allow-list,
not accidental. No `auxiliary_assets.yaml` rows exist for `ptha_mtst` (grep: only `sprc`-family and
other model ids). Nothing can be prepared for a model with no source and no checkpoint.

`checkpoint_serialization` is **not declared** for this model (other entries declare `state_dict` or
`whole_model`). Unavoidable while no checkpoint exists; P3 bookkeeping once one does.

**Verdict: the registry's checkpoint claims hold.** `strict_clean_checkpoint_available: false`,
`reproduction_status: BLOCKED_NO_VERIFIED_FASHIONIQ_CHECKPOINT` (`:239`) and the "CIRR pretraining
evidence is not eligible" note (`:241`) are all correct: the only published weight is a CIRR
MTST-pretrained model (`tuned_mtst_cirr.pt`), and using a CIRR-pretrained weight to score a FashionIQ
row would be exactly the kind of substitution the workbench's protocol rules forbid.
STATIC_AUDIT (metadata only; **no weight was downloaded or loaded**).

## 6. SOURCE TREE COMPLETENESS — a fresh finding (P1, unpinned)

The prior audit described the repo's environment as a copy of SPRC's. I checked the same class of
defect SPRC was suspected of — imports referencing modules the revision does not contain — by reading
the upstream file and comparing each dotted module against the tree listing:

OBSERVED, from the fetched `MTST_PTHA/src/lavis/models/__init__.py` (50 `from lavis...` import lines)
cross-checked against `read https://api.github.com/repos/zyinan99/MTST_PTHA/git/trees/main`:

- line 51 `from lavis.models.blip2_models.blip2_qformer_tgir_concatembedding_sim_zyn import
  Blip2Qformer4TGIRCESIMZYN`
- line 37 `from lavis.models.blip2_models.blip2_vicuna_instruct_tgir_modifiercaption import
  Blip2VicunaInstructTGIR4MC`

Two targeted HTTP probes at the branch HEAD SHA:

```
MTST_PTHA/src/lavis/models/blip2_models/blip2_qformer_tgir_concatembedding_sim.py          → 200
MTST_PTHA/src/lavis/models/blip2_models/blip2_qformer_tgir_concatembedding_sim_zyn.py      → 404
MTST_PTHA/src/lavis/models/blip2_models/blip2_vicuna_instruct_tgir_modifiercaption.py      → 200
```

**The `_zyn` module imported unguarded at `models/__init__.py:51` is absent from the repository.**
Because that import sits before the registration line for every class, `import lavis.models` fails —
so even *with* a source checkout, the MTST evaluator could not import its own package. Classification:
**UNPUBLISHED_SOURCE_MODULE**, P1, **upstream** ownership. This is precisely the CSMCIR defect class
(`workbench/patches/csmcir_models_init_compat.patch`), independently present here. It matters because
it means syncing and pinning the repo is **necessary but not sufficient**; an approved source patch
would also be required, and the workbench's `approved_source_patches` mechanism is the right home for
it once the tree is pinned.

Caveat on method: my tree check enumerated imports from the file and probed the two non-standard ones
directly rather than diffing the full recursive tree in memory, so I claim exactly what I observed —
two MTST-specific modules, one present, one 404 — and I do **not** claim to have proven the absence of
any other missing module. A full audit requires the pinned checkout.

## 7. DEPENDENCIES

Environment (upstream README "Prerequisites (Following SPRC)"; `requirements.txt` at the repo root,
fetched this pass, 14 lines): `tqdm==4.66.1, omegaconf==2.3.0, iopath==0.1.10, timm==0.9.12,
decord==0.6.0, webdataset==0.2.86, opencv-python, transformers==4.36.2, fairscale==0.4.13,
einops==0.7.0, scikit-image==0.22.0, spacy==3.7.2, pycocoevalcap==1.2` (line 1 `# pandas==2.2.0`
commented out), plus `conda create -n mtst python=3.9` and
`pip install torch==2.0.1 torchvision==0.15.2 torchaudio==0.2.0`.

Dependency categories D1–D8 are not defined anywhere in this repo (I grepped for `D1`, "dependency
categor" — no definition exists; the only `D1` hits are unrelated in
`workbench/docs/reviews/final_pc_audit/`). I therefore classify with a stated scheme and do not invent
numbers. Notable facts:

- `lavis` is again **vendored** (`MTST_PTHA/src/lavis`), and `requirements.txt` has **no lavis line**
  (grep for `lavis` → no match). Same gap as SPRC.
- No `clip` line in `requirements.txt`, while `src/blip_validate.py` (SPRC lineage) imports `clip` at
  line 8 — I verified the MTST file by HTTP fetch; the import list is the SPRC one.
  `INSTALLABLE_DEPENDENCY_MISSING` (P1) if the MTST entrypoint is used.
- `transformers==4.36.2` is declared and, as verified in the SPRC report, that version does export
  `apply_chunking_to_forward` from `transformers.modeling_utils` — the vendored LAVIS needs it.
- `timm==0.9.12`: the vendored LAVIS imports `timm.models.registry`/`hub`/`layers` (deprecated shims).
  Whether 0.9.12 still provides them for the code paths used is **UNKNOWN** — I did not install it.
- The environment's *number* cannot be fixed at all until a pin exists; every item above is graded
  against a moving `main`.

## 8. FASHIONIQ PROTOCOL — deliberately NOT guessed

Registry: `native_protocol: null`, `supported_protocols: []` (`models.yaml:234-235`).

**I do not propose a value, and here is the reason in evidence, not style.** The protocol is a property
of the *pinned evaluator's data flow*. For this model there is no pin, and the only reachable revision
has three properties that each independently change the candidate universe and are each unverifiable
from a paper:

1. `data_utils.py:169-172` **silently deletes** split ids whose image file is missing, so the gallery is
   `split ∩ {files present}` rather than the published split.
2. `data_utils.py:180,188,196` rewrite ids with `.replace("jpg","png")`, so the mapping from
   FashionIQ ids to files is revision-specific.
3. `validate_blip.py:44-55` has no reference mask on the FashionIQ path (→ reference eligible), which
   is consistent with the workbench's `fashioniq_original_split` (`src/evaluation/fashioniq.py:172-174`)
   — but this is one of the three, and the other two are unresolved.

The paper offers no help: `read https://arxiv.org/abs/2504.05316` gives no candidate-set statement, and
the arXiv HTML contains **no code URL** (I fetched `https://arxiv.org/html/2504.05316v1`, 304,040 bytes,
and grepped for `github|huggingface|1drv|drive.google` → only arXiv's own `html_feedback` and LaTeXML
issue links). So "the paper and the code agree" cannot be asserted, and the workbench's rule is
explicit: never change a gallery, reference policy, caption input or noise condition to make a model
runnable — and equally never *invent* one.

**Evidence that would be required to establish a protocol (this is the answer):**
1. A **pinned commit** of `https://github.com/zyinan99/MTST_PTHA`, with `source_dir` registered and the
   checkout's `git rev-parse HEAD` equal to the recorded SHA. Without it nothing is auditable, since
   `main` is a moving target (6 commits, last push 2025-02-09).
2. The pinned `src/data_utils.py` + `src/validate_blip.py` read line-by-line to determine, in the pinned
   revision: the gallery source file(s), whether the `.png`-existence filter is active, the exact id
   transformation, and the presence/absence of a reference mask on the FashionIQ path.
3. The **FashionIQ fine-tuned checkpoint** for the evaluated row, plus its top-level key, so the class
   name (hence which `inference()` the protocol's query construction goes through) is known rather than
   guessed from the README's `blip2_tgir_ce_sim` hint.
4. Confirmation of which variant produced the reported R@10 54.99 / R@50 76.28 / mean 65.64
   (`models.yaml:236-238`) — ViT-g vs ViT-L, and whether the row is the MTST-pretrained model.
5. A paper-side or author-side statement of the candidate set (the paper currently gives none).

Items 1, 2 and 4 are obtainable by the workbench alone (network sync + audit). Item 3 requires the
authors to publish the weights, or a human who holds them. Item 5 requires the authors.

## 9. WORKBENCH INTEGRATION

| layer | state | evidence |
|---|---|---|
| adapter | **none** — no `PTHAAdapter`/`MTSTAdapter`; `ADAPTERS` has no such entry | `grep -rn "ptha\|mtst\|MTST" workbench/backend/adapters/models.py` → no match; `ADAPTERS = {...}` at `adapters/models.py:155` lists 15 ids, none `ptha_mtst` |
| planner | unreachable: `ADAPTERS.get('ptha_mtst')` → `None` → `guarded_plan` returns `(None, ['SKIPPED: adapter command not audited'])` | `workbench/scripts/evaluate_models.py:361-365` |
| preflight | `source_available: false` + `source_dir: null` → blocker `upstream source unavailable`; plus `command not runnable: COMMAND_UNAUDITED`; plus `environment_required` unset so no env blocker | `workbench/backend/preflight.py:178-181`, `:205-206` |
| checkpoint prep | nothing to prepare (`checkpoint_variants: []`); `download_url` doesn't exist | `models.yaml:241` region |
| metric parser | none — the model never reaches extraction; `metrics_extraction.py:403-419` has no `ptha_mtst` branch | `workbench/backend/metrics_extraction.py:403-419` |
| per-query support | none declared | no `per_query_observer` key on `models.yaml:217-241` |
| asset registry | no rows for `ptha_mtst` | grep over `workbench/registry/assets/cluster_b5_ptha_mtst.yaml` shows `assets:` entries all `classification: OFFICIAL_ASSET_UNAVAILABLE`, `acquisition_status: BLOCKED` (`:22-31`, `:43-52`) |
| stale documentation | `workbench/registry/assets/cluster_b5_ptha_mtst.yaml:3` and `:14-16` still say "no obtainable source checkout", "No official repository has been pinned", `search_locations_checked` lists only two *local* places (`:58-60`) — that is now wrong in the same way the old registry note was, because the author repo IS known and recorded at `models.yaml:228` | `cluster_b5_ptha_mtst.yaml:3,14-16,32-33,58-60` vs `models.yaml:228` |

Two documentation defects follow from the above, both P3 bookkeeping and both mine to report, not to
fix (read-only):

- `cluster_b5_ptha_mtst.yaml:32-33` "the workbench does not invent one" / "No official repository has
  been pinned" contradicts `models.yaml:228` which records the repository. The cluster file should
  either be regenerated from the registry or annotated.
- `cluster_b5_ptha_mtst.yaml:22` classifies the source checkout as `OFFICIAL_ASSET_UNAVAILABLE`. The
  honest token is now **not** "unavailable" but "known and not yet synced/pinned" — i.e. the asset
  exists, the workbench has not acquired it.

## 10. CONFIRMED DEFECTS

**D1 — no pin, therefore nothing is auditable.** P1, owner **workbench**.
`upstream_commit_sha: null` (`models.yaml:230`), `source_dir: null` (`:232`); OBSERVED the remote has a
single branch `main` at `6a02ee4763fc2260b6abfb18be78853facb91d86` and the last push was 2025-02-09, so
the target is stable enough to pin today. Fix: sync + pin; complexity low; scientific risk none.

**D2 — the pinned revision would not import.** P1, owner **upstream**, `UNPUBLISHED_SOURCE_MODULE`.
`models/__init__.py:51` imports `blip2_qformer_tgir_concatembedding_sim_zyn`, which is **404** on
`main` (§6). Reproduction: `curl -o /dev/null -w '%{http_code}'
https://raw.githubusercontent.com/zyinan99/MTST_PTHA/6a02ee4763fc2260b6abfb18be78853facb91d86/MTST_PTHA/src/lavis/models/blip2_models/blip2_qformer_tgir_concatembedding_sim_zyn.py`
→ `404`. Fix: an approved `models/__init__.py` patch of exactly this delta (the CSMCIR pattern), plus a
`source_blocker` on the registry entry so preflight reports it ahead of downstream blockers. Regression:
not runnable without the bytes.

**D3 — the published evaluation recipe mis-specifies the backbone.** P2, owner **upstream**.
README Evaluation passes no `--backbone`, so the default `pretrain` (`blip_validate.py:190`) applies,
while the Training section requires `--backbone pretrain_mtst` (`PRETRAINED_MODEL_CONFIG_DICT` in
`blip2_qformer_tgir_concatembedding_sim.py`, config `blip2_pretrain_mtst.yaml`). A faithful wrapper
MUST pass `--backbone pretrain_mtst`; following the README verbatim would load the wrong config.

**D4 — the default `--blip-model-name` is not a registered class.** P2, owner **upstream**, same class
of defect as SPRC's (`blip_validate.py:189` default `blip2_cir_rerank_learn`; observed `grep` over the
upstream tree returns no `rerank_learn` registration). The correct value here is at least *nameable*
(unlike SPRC's): the README trains with `blip2_tgir_ce_sim`, which **is** registered
(`@registry.register_model("blip2_tgir_ce_sim")` in `blip2_qformer_tgir_concatembedding_sim.py`).
Still not provable without the checkpoint's top-level key; do not hard-code it.

**D5 — silent gallery mutation + id rewriting.** P2, owner **upstream**, UNPINNED.
`data_utils.py:169-172` drops ids whose file is absent; `:180,188,196` rewrite ids. Either can change
the candidate universe with no error and no log line beyond the removal itself.

**D6 — harness-side dead ends that read as readiness.** P3, owner **workbench**. `cluster_b5_ptha_mtst.yaml`
statements contradict `models.yaml:228` and classify a known asset as `OFFICIAL_ASSET_UNAVAILABLE`
(§9).

**No defect is claimed for**: the registry's `null` protocol (correct — see §8), the empty
`checkpoint_variants` (correct — §5), or `upstream_repo_url` being present with
`source_available: false` (correct and deliberate — §2).

## 11. REMAINING BLOCKERS

| # | classification | owner | evidence | remediation |
|---|---|---|---|---|
| B1 | **UNPUBLISHED_SOURCE_MODULE** | upstream | `models/__init__.py:51` → 404 on `blip2_qformer_tgir_concatembedding_sim_zyn.py` | approved source patch (CSMCIR pattern) after pinning |
| B2 | **CHECKPOINT_UNPUBLISHED** | upstream | README lines 85-87 link only CIRR artifacts; `fashioniq.pt`/`tuned_mtst_fashioniq.pt` → 404; HF repo has 5 files, none FashionIQ | authors publish a FashionIQ fine-tuned weight; nothing the workbench can do |
| B3 | **FIRST_PARTY_ADAPTER_MISSING** | workbench | no `ptha_mtst` entry in `ADAPTERS` (`adapters/models.py:155`) | implement after B1+B2; must pass `--backbone pretrain_mtst` (D3) |
| B4 | **SCIENTIFIC_PROTOCOL_UNVERIFIED** | upstream paper | paper gives no candidate set; arXiv HTML has no code URL | §8's five-item evidence set |
| B5 | **UNPUBLISHED_SOURCE_MODULE** / sync gap | workbench | `upstream_commit_sha: null`, `source_dir: null` | sync + pin (§10 D1) — obtainable without upstream cooperation |
| B6 | **INSTALLABLE_DEPENDENCY_MISSING** | workbench env spec | no `lavis`, no `clip` line in the upstream requirements | resolve at env-build time, after the pin |
| B7 | **GPU_RUNTIME_UNTESTED** | environment | no source, no checkpoint, no run | unreachable |

Note the vocabulary constraint: none of these is `EXTERNAL_BLOCKER`. The two genuine
upstream-cooperation blockers are B1 and B2; **B1 is arguably fixable by the workbench** via the
approved-patch mechanism once the tree is pinned, so the only hard upstream dependency is B2.

## 12. FINAL VERDICT

**BLOCKED — `UNPUBLISHED_SOURCE_MODULE` (P1) + `CHECKPOINT_UNPUBLISHED` (P0 for reproduction)**, with
these independently established facts:

- The registry entry is **accurate and honest**: `source_available: false`, `upstream_commit_sha: null`,
  `source_dir: null`, `native_protocol: null`, `supported_protocols: []`, `checkpoint_variants: []`,
  `strict_clean_checkpoint_available: false` — all verified against the file and the machine-readable
  registry.
- The linked repository **is the model's own source**, not another project's: paper title verbatim,
  authored by the first author Yinan Zhou, containing both named contributions (MTST synthetic-triplet
  usage and PTHA two-hop alignment) as registered model classes. **The conditional in the assignment
  ("if the only public artifact is a different paper's repo") fails; I say so explicitly.**
- The repository **cannot currently be run as published**: an unguarded import of a module absent from
  the repo (`models/__init__.py:51`) prevents `import lavis.models`.
- **No FashionIQ checkpoint is published** — only CIRR artifacts, and a CIRR-pretrained weight is not
  eligible for a FashionIQ row.
- **No protocol can be established** without a pin plus the FashionIQ weight; per §8, the required
  evidence is a pinned commit, line-by-line audit of the pinned data/metric path, the FashionIQ
  checkpoint's key, variant identity for the reported row, and an author statement of the candidate
  set. I deliberately record `native_protocol: null` as **correct**, not as a gap to fill by
  inference — inventing a protocol here would be the exact failure mode the workbench rules forbid.

## 13. EVIDENCE

Repo: `https://github.com/zyinan99/MTST_PTHA` (no pin in the registry). HEAD observed
`6a02ee4763fc2260b6abfb18be78853facb91d86` on `refs/heads/main`, sole branch, last push 2025-02-09.
Hub: `https://huggingface.co/yinanzhou1/mtst_ptha` @ `b5e669c8ef28b0084f43398d184a5a0116903ccf`.

Files read (registry, local, read-only): `workbench/registry/models.yaml:216-241`;
`workbench/registry/assets/cluster_b5_ptha_mtst.yaml`; `workbench/backend/adapters/models.py:155`;
`workbench/backend/preflight.py:170-208`; `workbench/backend/metrics_extraction.py:403-419`;
`workbench/scripts/evaluate_models.py:361-365`; `src/evaluation/fashioniq.py:165-180`;
plus the prior hypotheses `workbench/docs/reviews/models/ptha_mtst.md`,
`workbench/docs/reviews/final_model_audit/ptha_mtst.md` (re-tested, not paraphrased).

Upstream files read at the observed SHA `6a02ee47…`: `README.md`, `requirements.txt`,
`MTST_PTHA/src/blip_validate.py` (sha256 `fb040a92e9534405e13b5992dc6686ea650b1925e5931442462d1c25a76655bb`),
`MTST_PTHA/src/data_utils.py`, `MTST_PTHA/src/validate_blip.py`, `MTST_PTHA/src/utils.py`,
`MTST_PTHA/src/lavis/models/__init__.py`,
`MTST_PTHA/src/lavis/models/blip2_models/blip2_qformer_tgir_concatembedding_sim.py`,
`MTST_PTHA/src/lavis/configs/models/blip2/blip2_pretrain_mtst.yaml`.

Commands / reads with observed results:
1. `git ls-remote https://github.com/zyinan99/MTST_PTHA` → `6a02ee4763fc2260b6abfb18be78853facb91d86 HEAD` + `refs/heads/main`
2. `git ls-remote --heads …` → one line, `refs/heads/main`
3. `read https://api.github.com/repos/zyinan99/MTST_PTHA` → `fork: false`, `created_at 2024-12-26`, `pushed_at 2025-02-09`, `default_branch main`, MIT, 0 stars
4. `read https://api.github.com/users/zyinan99` → `"name": "Yinan Zhou"`
5. `read https://api.github.com/repos/zyinan99/MTST_PTHA/commits?per_page=10` → all commits by Yinan Zhou
6. `read https://api.github.com/repos/zyinan99/MTST_PTHA/git/trees/main` → 6 top-level entries, `truncated: false`
7. `read https://raw.githubusercontent.com/…/README.md` → title == paper title; downloads at lines 85-87, all CIRR
8. `read https://raw.githubusercontent.com/…/requirements.txt` → 14 lines, no `lavis`, no `clip`
9. `read https://arxiv.org/abs/2504.05316` → title/authors/date match; MTST and two-hop alignment named in abstract
10. `curl https://arxiv.org/html/2504.05316v1` → 304,040 bytes; grep for code hosts → no author code URL
11. `read https://huggingface.co/api/models/yinanzhou1/mtst_ptha/tree/main?recursive=true&expand=true` → 5 files, 3 with LFS oids/sizes
12. `read https://huggingface.co/api/models/yinanzhou1/mtst_ptha` → `gated false`, `usedStorage 3034364073`, last commit = a rename
13. `curl -I` on `tuned_mtst_cirr.pt` / `fashioniq.pt` / `tuned_mtst_fashioniq.pt` → `200` / `404` / `404`
14. `curl https://huggingface.co/api/models?search=mtst&limit=20` and `?search=ptha&limit=20` → no other CIR model
15. `curl https://huggingface.co/api/models/yinanzhou1/mtst_ptha/discussions` → `{"discussions":[],"count":0}`
16. targeted HTTP probes at SHA `6a02ee47…`: `blip_validate.py` 200, `data_utils.py` 200, `validate_blip.py` 200, `utils.py` 200, `blip2_qformer_tgir_concatembedding_sim.py` 200, `blip2_qformer_tgir_concatembedding_sim_zyn.py` **404**, `blip2_vicuna_instruct_tgir_modifiercaption.py` 200, `blip2_pretrain_mtst.yaml` 200
17. `.venv/bin/python -c "load_registry()…['ptha_mtst']"` → fields quoted in §2
18. `ls workbench/third_party/` → 14 dirs, none MTST/PTHA

## 14. ACHIEVED VERIFICATION LEVEL

**LEVEL 1 of 6** — identity, registry consistency and network-reachability auditing only.
There is no pinned checkout, so no source-level audit is possible; there is no checkpoint, so no
dependency installation, no process execution, no metric extraction and no comparison to the paper's
reported row. Nothing here is RUNTIME_VERIFIED; §4 through §8 are STATIC_AUDIT against an **unpinned**
revision and are labelled as such throughout.
