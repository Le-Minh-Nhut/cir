# FINAL 16-MODEL RECOVERY PRIORITY

Ranked by **what the user can act on**, not by benchmark Recall. The user studies persistent CIR
failures across generations, so the ranking weights: (1) trustworthiness of the result, (2) protocol
compatibility, (3) ability to produce **per-query** rankings, (4) cost of the next step. Evidence for
each cell is in `FINAL_16_MODEL_REPRODUCIBILITY_REASSESSMENT.md` and that model's own report.

> **Read this first.** Exactly **one** model produces per-query rankings today (`limn`). Multi-model
> failure analysis — common failures, persistent failures, regressions, rank disagreement, Jaccard —
> requires ≥2 runs under the **same** protocol, and the API refuses anything else
> (`storage.validate_analysis_selection`). So every recommendation below is really a statement about
> **which second observer to build**, and in which protocol.

---

## GROUP 1 — QUICK WINS

### `limn` — the only model that can run a real pilot today
Observer, per-query export, parity gate, three verified checkpoints, correction dictionaries in-tree,
native entrypoint callable, dependency set identical to the provisioned one.
**Next step**: run it on the RTX 5070 Ti. Nothing else is required.
**Risk**: an adapted runtime (`sm_120`) makes it a *variant*, never official parity — the registry
already records that.

### `clip4cir_rn50x4_fullft` — nearest to runnable after LIMN
`CLI_READY`; the author moved upstream and published a **checksummed GitHub release**; the loader
contract is known (`state_dict["CLIP"]`, then `state_dict["Combiner"]`).
**Next step**: pin the release digest, provision the RN50x4 base, stage the dataset link.
**Why high value**: it is Protocol **A**, and it is the newest model in that protocol — it is the
**anchor for any "old model right / new model wrong" comparison in A**.

### `combiner_rn50x4_noft` — same repository, same release, one variant over
Byte-identical pinned SHA, same release assets; its member is `FIQ_combiner_...` with a published
sha256 and a verified strict-load (`<All keys matched successfully>`, 12 tensors, dims `640→2560→5120→640`).
**Next step**: same as above.
**Why high value**: it is the **no-fine-tune ablation of the same architecture**, so pairing it with
`clip4cir_rn50x4_fullft` isolates fine-tuning as a variable — a comparison the user cannot make any
other way in Protocol A.

---

## GROUP 2 — MODERATE ENGINEERING

### `encoder` — cheapest real fix in the set
Three bounded items: a one-token alias (`import datasets as datasets1` — every needed symbol exists),
staging the public LAION `ViT-B-32` asset the code asks for, and writing a metric parser.
**Next step**: add the alias as an `approved_source_patches` entry (the CSMCIR pattern already exists),
add the asset to the auxiliary manifest, implement the parser.
**Why high value**: Protocol **C**, same as LIMN — **this is the second observer to build.**

### `hint` — Protocol C, checkpoint mapping clean
Correction dictionaries already on disk; checkpoint mapping verified; no CLI but the call chain is
short. `torchaudio` is a real, installable blocker (it is the first failure of `from lavis.models
import …`); `setproctitle` is **not** (training-only).
**Next step**: wrapper + `torchaudio` + parser.

### `habit`, `intent`, `conesep` — Protocol B, same shape of work
All three: callable evaluator, no CLI, correction dictionaries present (contrary to the old notes),
checkpoint mapping verified where clean weights exist. `habit` and `conesep` have verified checkpoints;
`intent`'s 0 %-noise row has **no** published checkpoint, which is a genuine artifact gap.
**Next step**: wrapper + parser; `intent` additionally needs the authors to publish a 0 % checkpoint.

### `clvc_net`, `dcnet` — Protocol C / B, but read the caveats
`clvc_net`: two edits, both **proved numerically neutral** by execution, plus a wrapper; watch the
`pretrained=` default, which today picks a **different** ImageNet weight file than the declared one.
`dcnet`: `CLI_READY` and CPU-capable, but see Group 3 — the scientific blocker dominates.

---

## GROUP 3 — REQUIRES UPSTREAM ARTIFACTS

### `csmcir` — Protocol A, blocked on one unpublished file
The CLI exists and needs **no wrapper**. The approved patch is verified (base and derived digests
match; 98 absent modules guarded, 146 import lines unchanged, semantics preserved). The blocker is
`COT_ours2/fashioniq/*_cot_val.json`, which **drives gallery features** — the author's downloadable
`qwen_captions/*` feeds a different field that inference never reads, so it is not a substitute.
Plus the checkpoint is keyed for a class name the evaluator does not compute.
**Next step**: ask the authors for the CoT val captions, or accept a variant.

### `sprc` — Protocol A, blocked on a 403
The vendored LAVIS is **complete** (the old 98-import claim belongs to CSMCIR); `transformers==4.36.2`
is pinned and sufficient; the argparse contract is four flags and satisfiable. The blockers are the
**403** on the author's checkpoint URL and an unregistered model class name in the evaluator's own
default. The adapter also builds no command line at all — that is first-party work, not upstream.
**Next step**: authors republish the weights; then write the adapter command.

### `airknow` — Protocol B, blocked on two unpublished things
Callable evaluator, live checkpoint URLs, both whole-module pickles. The correction dictionaries are
needed unconditionally (`datasets.py:187`) and are published nowhere. There is also **no checkpoint
loader anywhere** in the tree, so a wrapper must supply both the loader and the call sequence.
**Next step**: request the dictionaries; then MODERATE wrapper.

### `tgcir` — Protocol C, blocked on a mapping AND an unsafe artifact
`test.py` has no `__main__`; only the FashionIQ path lacks an entrypoint (the CIRR path has one). The
checkpoint is a **whole-module pickle** — `weights_only=True` is structurally impossible, so the first
load is unguarded code execution in the evaluator's interpreter, and the registry records **no digest**
for it. The registry also declares a `.pt` where the author ships a `.zip` whose members are UNKNOWN.
**Next step**: fetch and digest the archive (isolated host), then map members; only then a wrapper.

### `dcnet` — Protocol B, permanently unreproducible as published
`process_cap.py:31` draws OOV embeddings from `np.random.normal` with **no seed anywhere in
`preprocess/`**. Regeneration produces a different experiment, and the authors published no `.pkl`.
**Decision required, not work**: either accept a variant whose numbers are not the paper's, or drop
the model from failure analysis. Do not spend engineering time on this expecting parity.

---

## GROUP 4 — RESEARCH RECONSTRUCTION / HIGH RISK

### `pair` — Protocol C, missing architecture module
`MHTransformer_rev` is imported by `models/model.py:13` and by no other file, and it exists nowhere in
the pinned tree or upstream history. The class it would supply is partly present (`Transformer` at
`MHTransformer.py:119`, `MultiHeadAttentionRev` at `:61`), but `TransformerLayer.__init__` hardcodes
`MultiHeadAttention` (`:115`), so **no code path can instantiate the revision variant**. The repo is
additionally not importable (`datasets` under `data/`, no `__init__.py`, no `sys.path` handling) and
`train.py`'s `__main__` never calls `train()` or `test.test()`.
**Verdict**: reconstruction would mean writing a component the authors did not publish. Not advised.

### `ptha_mtst` — no source, no weights, no protocol
The repository linked in the registry **is** the paper's own source (verified: title, first author,
his commits) — so "source unavailable" was the wrong label. What is missing is everything else: no
FashionIQ checkpoint (Hub holds three CIRR files; `fashioniq.pt` → 404), and the pinned revision
cannot even import (`models/__init__.py:51` names a module that 404s). The protocol is genuinely
**UNRESOLVED**, and guessing it would violate the invariant that a model's protocol comes from
evidence.
**Verdict**: needs the authors. No amount of local work reaches it.

---

## The one thing to do first

Build **`encoder` as the second Protocol-C observer**. It shares LIMN's protocol, its three blockers
are each bounded and independently testable, and it is the cheapest path to the state where the
workbench can answer the user's actual question — *which queries fail across generations* — instead of
reporting one model's failure profile. Protocol A is already well supplied with runnable models; C is
where the dataset is thinnest and where LIMN lives.
