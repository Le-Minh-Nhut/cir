# PREMISES — audit of the G1/G2 · CLVC · LIMN · PAIR campaign spec

Read-only audit of every premise in `CIR_G1_G2_CLVC_LIMN_PAIR_REPAIR_PROMPT.en.md`.
Tree: `/home/heheboiz/data/cir`, branch `research/cir-failure-workbench`, HEAD
`40c897c31bbdd76b5a76f65935c7dd0f7fef0e94` = `origin` (`git rev-list --left-right --count` → `0	0`).
Porcelain at audit start: `?? CLEANUP_REPORT.md.save` only. **Concurrent, not mine:** at
`2026-11-12 19:39-19:40 +0700` three tracked files acquired uncommitted edits —
`workbench/backend/composition.py` (+116/−31), `workbench/tests/test_composition_api_contracts.py`
(+83), `workbench/tests/test_review_defects_evaluator.py` (+75). `workbench/backend/parity_gate.py`,
every `g1_g2.*.json`, every `raw/…` artifact and every `workbench/artifacts/…` summary cited here are
**unmodified**, so all verdicts below stand; only the `composition.py` API-layer claims would need a
re-read if that diff lands. No source edit by this audit, no stage, no commit, no GPU run, no service
restart, **zero pytest invocations**. Gate verdicts were produced by *importing*
`workbench.backend.parity_gate` from a `python3` one-liner under `/tmp` (read-only; writes nothing).

Verdict counts: **HOLDS 8 · REVISED 5 · FALSE 0** (claims 1,2,3,6,7,8,10,13 hold; 4,5,9,11,12 hold
with corrections that change the work item).

Every artifact written by an earlier campaign is cited by path; where a doc and an artifact
disagree, the artifact wins and the doc is called out.

## Verdict table

| id | claim (summary) | verdict | evidence (file:line / artifact / command) | correction (REVISED only) |
|---|---|---|---|---|
| 1 | gate exists; `GET /api/runs/parity-gate` discovers records by scanning | **HOLDS** | `parity_gate.py:1144` `_discover_models`, scan `:1152-1155`; root `:75`+`:208-210`; route `main.py:1172-1184`; fields `:407-419`, `:456`, `:461`; measured `UNPAIRED: PARITY_EVIDENCE_MISSING`, `MISMATCH: PARITY_MANDATORY_SUBCHECK_FAILED failures 22` | — |
| 2 | v2 request leaves v1 verdicts on v1; `_declared_standard` rejects v2 | **HOLDS** | `parity_gate.py:104-106`, `:749-753`; measured `requested …/v2 applied parity-relative/v1 advisory ['…/v2']`; v2-labelled copy → `PARITY_EVIDENCE_MALFORMED`; missing writer (`t4_parity.py:400-410`, `encoder_parity.py:976-988`, no `standard` key) **and** missing reader (`evaluate_abs_rel` `:343` has no production caller) | — |
| 3 | producer scripts + paired JSONs exist; identity fields | **HOLDS** | `scripts/{t4_parity,encoder_parity,t2_dual_scope,t3_native_eval}.py`; 12 `g1_g2.*.json` with byte counts + sha256 in §3; encoder pair carries `model_id/protocol`, LIMN pair carries none (all 7 `IDENTITY_FIELDS` `None`) | — |
| 4 | 17 registered, 15 without parity, SPRC checkpoint absent, CSMCIR composition deferred | **REVISED** | `models.yaml` 17 ids (§4 line list); `INVENTORY.csv:2-18` parity `Y` only for encoder/limn; `checkpoints/sprc/sprc_fiq_vitl.pt` = 1 459 756 541 B, `sha256 1938610…bea08` = `models.yaml:1452`; `INVENTORY.csv:2` csmcir `USER-DEFERRED` | SPRC's **ViT-L** checkpoint is present and digest-verified; only the **paper's ViT-g** is absent (`models.yaml:1423-1425`, `MODELS_REST_VERIFICATION.vi.md:459-462`). Replace "checkpoint absent" with "no ViT-g checkpoint; ViT-L measured but not comparable to the paper row" |
| 5 | relation-state pin wired through both processes; provenance of `learnable_relation` | **REVISED** | pin `model_encoder.py:59,65`; adapter applies `:500` before `Encoder` `:504`; native imports `replay/encoder.py:36`, seeds `:324`, records `:253-270`; producer `encoder_parity.py:388` unseeded, `grep -c manual_seed / RELATION_SEED / relation_state` → 0 matches each; `model_try2.py:93` `nn.Parameter(torch.randn(...)).cuda()`; proof `relation_proof_construct.out` (`IN_STATE_DICT False`, 417 keys) | Pin **is** wired in both production paths and `learnable_relation` is constructed (not a `state_dict` key). The unwired process is the G1/G2 **producer**, which never seeds and emits no `relation_state`, so the gate must fail `relation.<cat>.digest` |
| 6 | two rank disagreements archived; stable IDs recoverable | **HOLDS** | `g1_g2.encoder.dress.adapter.json` `rank_differences_vs_native_val_control` index 1569/1215 vs 1216; shirt 1352/1373 vs 1372; `raw/encoder_parity/inputs.{dress,shirt}.json`; join to `circus_release/mbeir_fashioniq_task7_test.jsonl` → **dress `7:1565`, shirt `7:3362`** | — (IDs, not just indices: add them to the run identity) |
| 7 | CLVC shares LIMN resolver; key is `(stem, normalized caption)` | **HOLDS** | `model_clvc_net.py:68-71` import, `:256-257` call; `model_limn.py:189-227`, key line `:213`, index build `:150-153` | — (path: `workbench/composition/model_clvc_net.py`; there is no `composition/adapters/`) |
| 8 | `7:3038`,`7:3790` are the last 2 of 119 unscoped | **HOLDS** | `raw/m02/m02_recount.json:288-297` (119/117/2, shirt 46 toptee 71); digests `:7-10`; `derived/category_routing_inputs/clvc_net_da0a2162…json` `resolver_routed_count 117`, both qids absent; `m02_resolver.txt:42-45` | — |
| 9 | two conflicting CLVC native R@10 artifacts; provenance; reproducible? | **REVISED** | A `raw/clvc_run3cat_gpu.json` 30.02079703540087 / 58.67189400437203; B `measurements/clvc_run1.json` 29.970746517160162 / 58.62171212901515; both `mrr` absent, batch/mask/digest absent from artifacts; commands `BLOCKED_MODELS_VERIFICATION.vi.md:214-218` (A), `CHECKPOINT_SELECTION_REPORT.md:1697-1699` (B) | `mrr` is **not recorded** in either native artifact; batch and mask are prose-only. A controlled rerun must emit per-query ranks + tie-break + batch/mask provenance, else it yields a third unresolvable macro |
| 10 | LIMN parity: matching ranks, missing hit-count evidence | **HOLDS** | adapter `scoring_ranking.rank_differences 0`, `unexplained 0`, `native_aggregate_ranks_compared 2017/2038/1961`; `native_metrics_match_feature_rerank` absent in all six LIMN files (`grep -c` → 0), gate `missing ['metrics_parity']`; prose `INPUT_FORWARD_PARITY.vi.md:255-260` says PASS while gate says `PARITY_MANDATORY_SUBCHECK_FAILED` (3× batch `max_rel`) | — (do not cite the prose PASS; the gate verdict is authoritative) |
| 11 | PAIR blocked `BLOCKED_MISSING_SOURCE_CLASSES`; checkpoint modules; 5.20 %/2×4 measured? | **REVISED** | `raw/pair_absence_evidence.json:3-21`; producer `replay/pair_absence.py:220`; `102-…inventory.md:224`; registry says `BLOCKED_CHECKPOINT_MAPPING_UNVERIFIED` (`models.yaml:825`); geometry `.raw/fp2_pair_B1.json:2183,2217` (transformer_a 2×4), `transformer_c/d`, `weight_chanelImg*`, `cross_t2i` present; `pair_rederive.out` `PARAMS_TOTAL 172814185` / `PARAMS_BYPASSED 8986448` / `PCT 5.2001` | The **registry** blocker is `BLOCKED_CHECKPOINT_MAPPING_UNVERIFIED`; `BLOCKED_MISSING_SOURCE_CLASSES` is the replay/audit verdict. 5.20 % is measured **against a hypothesised class**, 2×4 is measured **from the checkpoint** |
| 12 | report §0.1 block, §4.4b metrics, §9 first 5 rows; `PARITY.vi.md:219`; `102-…md:220` | **REVISED** | `FINAL_REPORT_20261112.vi.md:73-106`, `:990-997`, `:1387-1393` quoted in full; §4.4b re-read from `…/clvc_net_…da0a2162…/full.json`; `PARITY.vi.md:219` is blank (content at `:214-232`, LIMN recount `:249-258`); `102-…md:220` is the header `### 7.2`, content at `:224` | §0.1 is a `cd9ced2` snapshot; HEAD is `40c897c` (live: same MainPIDs, `0 0`, GPU `903 MiB`, bundle `index-NnjrdZLG.js`). Both "neighbourhood" citations point at blank/header lines |
| 13 | any doc still says `assets/index-DDviexaW.js` | **HOLDS** | `grep -rn "index-DDviexaW" workbench/docs` → 9 hits/7 files (14 hits/10 files for both names); 9 stale mentions in 7 files (§13 table); served = `dist` = `index-NnjrdZLG.js`, `sha256 721d34d0…80d0`, 392 769 B | — |

---

## A. Gate and contract

### 1. Gate exists; discovery is a filename scan of ONE fixed directory

**Verdict: HOLDS** (one correction: single directory, `iterdir`, not a recursive glob over
"directories").

- Gate module: `workbench/backend/parity_gate.py` (1192 lines). Route: `workbench/backend/main.py:1172`
  `@app.get("/api/runs/parity-gate")`, body `:1173-1184`, which does
  `from workbench.backend.parity_gate import evaluate_all` (`:1181`) and `return evaluate_all()`.
- Discovery function: `_discover_models` at `parity_gate.py:1144`; the scan itself at `:1152-1155`:

  ```python
      try:
          names = sorted(path.name for path in base.iterdir())
  ```

  `base` is the single directory `evidence_root()` (`:208-210`) →
  `PARITY_EVIDENCE_RELATIVE` (`:75`) = `("workbench", "docs", "reviews",
  "20261109-native15-composition-limn")`, resolved against `Path(__file__).resolve().parents[2]`.
  Not recursive, not request-derived, no glob.
- Filename grammar accepted (`:1156-1168`): `g1_g2.<model>.<category>.<side>.json` where
  `<side> ∈ {native, adapter}` and `<category> ∈ {dress, shirt, toptee}`; the bare
  `g1_g2.<category>.<side>.json` form resolves to the model the last-but-one component would name,
  and `LEGACY_BARE_MODELS = ("limn",)` (`:260`) is the only model allowed to pair via those bare
  names (`_artifact_names` `:269`, `_artifact_paths` `:278-292`).
- Required fields for a verdict to rest on anything (`evaluate_model` `:839`):
  - native `layer1_identity` block, keys `gallery_order_identical`, `reference_ids_identical`,
    `target_ids_identical`, `model_text_identical_to_author_composition` plus the two count pairs
    (`_identity_checks` `:407`, required tuple `:416-417`);
  - adapter body (`adapter` or `layer2_4_forward`, `_body` `:757`), and for every numeric block
    `max_abs`/`max_rel`/`min_cosine` plus the declared tolerance keys (`_relative_checks` `:348`);
  - cross-side identity (`IDENTITY_FIELDS` `:456`, `_identity_cross_checks` `:461`): `model_id`,
    `checkpoint_id`, `checkpoint_identity`, `protocol_id`, `protocol`, `query_order_sha256`,
    `gallery_order_sha256` — compared value-to-value, one-sided fields fail.
- Unpaired model: if **all** expected names are absent → `PARITY_NOT_RUN` (`:860-866`); if **some**
  are absent → `PARITY_EVIDENCE_MISSING` (`:868-871`). Measured (read-only import):

  ```
  UNPAIRED: PARITY_EVIDENCE_MISSING | missing parity artifact(s): ['g1_g2.encoder.dress.adapter.json', 'g1_g2.encoder.shirt.adapter.json', 'g1_g2.encoder.toptee.adapter.json']
  ```
- Identity-mismatched pair: verdict `PARITY_MANDATORY_SUBCHECK_FAILED` with per-field failures, never
  a silent pass. Measured with an adapter whose `model_id`/`protocol` were altered:

  ```
  MISMATCH: PARITY_MANDATORY_SUBCHECK_FAILED failures 22 passed False
      dress.input.identity_agreement.model_id {'native': 'encoder', 'adapter': 'encoder_other'}
      dress.input.identity_agreement.protocol {'native': 'fashioniq_val_split', 'adapter': 'fashioniq_full_gallery_ref_excluded'}
  ```
- Live verdict at HEAD (same call): `standard parity-relative/v1 requested parity-relative/v1 passed []`;
  `encoder PARITY_MANDATORY_SUBCHECK_FAILED failures=16 missing=[]`;
  `limn PARITY_MANDATORY_SUBCHECK_FAILED failures=3 missing=['metrics_parity']`. Failures include
  `relation.dress.digest measured='native=not_recorded, adapter=not_recorded'` and the three
  `*.batch_1_vs_n.max_rel` blocks.

Repro:

```
cd /home/heheboiz/data/cir && python3 -c "
import sys; sys.path.insert(0,'.')
from workbench.backend.parity_gate import evaluate_all
r=evaluate_all(); print(r['standard'], r['passed_models'])
for m,v in r['models'].items(): print(m, v['status'], v['failure_count'], v['missing'])"
```

### 2. `parity-absolute-relative/v2` — request does not relabel; `_declared_standard` refuses v2 artifacts

**Verdict: HOLDS.** Both code sites quoted below. Missing: **writer absent entirely; reader only
refuses — the componentwise comparison path is unwired.**

- Constants: `parity_gate.py:104-106`

  ```python
  STANDARD_RELATIVE_V1 = "parity-relative/v1"
  STANDARD_ABS_REL_V2 = "parity-absolute-relative/v2"
  STANDARDS = (STANDARD_RELATIVE_V1, STANDARD_ABS_REL_V2)
  ```

- Site 1 — a v2 *request* cannot move a v1 verdict. `evaluate_model` docstring `:841-846` and
  `evaluate_all` `:1170-1192`: `applied` is computed from the verdicts, `requested_standard` reported
  beside it, and `evaluate_all` returns `"standard": applied[0] ...` (`:1183`) with
  `"advisory_standards": sorted(set(STANDARDS) - set(applied))`. Measured:

  ```
  requested parity-absolute-relative/v2 applied parity-relative/v1 advisory ['parity-absolute-relative/v2'] passed []
  encoder PARITY_MANDATORY_SUBCHECK_FAILED requested parity-absolute-relative/v2 applied parity-relative/v1
  limn PARITY_MANDATORY_SUBCHECK_FAILED requested parity-absolute-relative/v2 applied parity-relative/v1
  ```

- Site 2 — `_declared_standard` (`:728`) rejects a v2-declaring artifact outright, `:749-753`:

  ```python
      if declared == STANDARD_ABS_REL_V2:
          raise _Malformed(
              "the artifact declares parity-absolute-relative/v2 but records only `max_abs`/`max_rel` "
              "aggregates; that standard is defined per component (abs(a-b) <= atol + rtol*abs(b)) and "
              "cannot be evaluated from the recorded statistics, so it must never be applied")
  ```

  Measured by copying the three LIMN pairs to `/tmp` and adding `"standard":
  "parity-absolute-relative/v2"` to both sides: `PARITY_EVIDENCE_MALFORMED | malformed parity
  artifact: the artifact declares parity-absolute-relative/v2 but records only …`.
- The v2 rule itself exists as `evaluate_abs_rel` (`:343`) but has **no production caller**:
  `grep -rn "evaluate_abs_rel" --include=*.py .` → `parity_gate.py:343` (definition) and
  `workbench/tests/test_gate_parity_subchecks.py:312,313,315` only.
- **What is missing — writer AND reader.**
  - *Writer*: both surviving G1/G2 producers emit no `standard` key at all —
    `grep -c "standard" t4_parity.py encoder_parity.py` → `0` matches for the standard field; the
    writers are `t4_parity.py:400-410` (`payload = {"category":…, "phase":…, "tolerances": TOL, …}`)
    and `encoder_parity.py:976-988`. `python3` scan of all 12 `g1_g2.*.json` for a top-level `standard`
    key → `[]`. They also record only `max_abs`/`max_rel` aggregates (`t4_parity.py:222-226`), so they
    cannot produce the per-component payload v2 is defined over.
  - *Reader*: v1 judgement is complete, but the v2 reader is a refusal only (`:749`). To consume a
    genuine v2 pair the gate needs (a) a stored per-component/ digest-bound chunk array and (b) an
    evaluation branch that applies `evaluate_abs_rel` per element — neither exists. `DIMENSIONS` (`:109-111`),
    `REQUIRED_DIMENSIONS` (`:101-102`) and every `ParityCheck` construction are v1-shaped
    (`standard=STANDARD_RELATIVE_V1`, e.g. `:378,392,403`).

Repro:

```
cd /home/heheboiz/data/cir && python3 -c "
import sys; sys.path.insert(0,'.')
from workbench.backend import parity_gate as gate
print(gate.evaluate_all(standard=gate.STANDARD_ABS_REL_V2)['advisory_standards'])"
grep -rn "evaluate_abs_rel" --include=*.py /home/heheboiz/data/cir | grep -v pycache
```

### 3. Producers and paired JSONs under `20261109-native15-composition-limn/`

**Verdict: HOLDS.**

Producer scripts (`scripts/`, sha256 first 16 hex, line counts):

| script | writes | lines | sha256[0:16] |
|---|---|---|---|
| `t4_parity.py` | `g1_g2.{dress,shirt,toptee}.{native,adapter}.json` (LIMN, legacy bare names) — `OUT` `:31`, target `:400`, write `:410` | 433 | `c6c417b65dc0a268` |
| `encoder_parity.py` | `g1_g2.encoder.{cat}.{phase}.json` — `OUT` `:37`, target `:976`, write `:988` | 1015 | `84fe9a44adadec46` |
| `t2_dual_scope.py` | `limn_dual_scope.json` (`:370`) + 6 `limn_dual_scope.<scope>.<cond>.jsonl` | 381 | `0254cc63b0f06fea` |
| `t3_native_eval.py` | `limn_native.json` (`:102`, `:273`) + `limn_native.<cat>.json[l]` | 281 | `95b6e4adf54c7ea2` |
| also present | `dump_registry.py`, `pair_pickle_globals.py`, `pair_rederive.py`, `pair_probe.py`, `probe_envs.py`, `probe_airknow.py`, `hash_ckpts*.py`, `run_chain.sh`, `ui_shots.py`, `failure_diff.py`, `reg_full.py`, `t1_provenance.py` | | |

Paired JSONs (12 files, bytes, sha256[0:16]):

| file | bytes | sha256[0:16] |
|---|---|---|
| `g1_g2.dress.native.json` | 6631 | `767a131280664ca5` |
| `g1_g2.dress.adapter.json` | 2695 | `b9187c562be4b6d4` |
| `g1_g2.shirt.native.json` | 6581 | `c85f306d84218623` |
| `g1_g2.shirt.adapter.json` | 2427 | `2bed688c68d4486b` |
| `g1_g2.toptee.native.json` | 6661 | `88912b89718a4b68` |
| `g1_g2.toptee.adapter.json` | 2493 | `91ab769994d9101a` |
| `g1_g2.encoder.dress.native.json` | 8443 | `97dc5f91755f968e` |
| `g1_g2.encoder.dress.adapter.json` | 6963 | `58eee7f86504af82` |
| `g1_g2.encoder.shirt.native.json` | 8379 | `fff4c2a967855ec2` |
| `g1_g2.encoder.shirt.adapter.json` | 6792 | `6438f08e5f754821` |
| `g1_g2.encoder.toptee.native.json` | 7326 | `b2f6224468243b47` |
| `g1_g2.encoder.toptee.adapter.json` | 6376 | `d3da63158bebafb0` |

Identity fields present, two of them (`python3` read of the top level):

- `g1_g2.encoder.shirt.native.json` → top keys `['category','phase','model_id','protocol','tolerances','native']`;
  `model_id='encoder'`, `phase='native'`, `protocol='fashioniq_val_split'`, `category='shirt'`.
  Same for its adapter pair (`phase='adapter'`).
- `g1_g2.shirt.native.json` → top keys `['category','phase','tolerances','layer1_identity']`;
  **`model_id=None`, `protocol=None`** — the LIMN pair carries *no* `model_id`, `protocol`,
  `checkpoint_id`, `checkpoint_identity`, `protocol_id`, `query_order_sha256`, or
  `gallery_order_sha256` at the top level (all seven `IDENTITY_FIELDS` are `None`). Its identity lives
  only inside `layer1_identity` (`query_count_native`/`query_count_annotations` =
  2017/2038/1961 pairs, `gallery_count_*`, the four `*_identical` booleans).

Tolerances (identical on both sides of every pair, e.g. encoder shirt): `forward_max_abs 1e-05`,
`forward_max_rel 1e-04`, `forward_min_cosine 0.9999999`, `batch_max_abs 1e-04`,
`batch_max_rel 0.001`, `score_max_abs 1e-06`, `tie_epsilon 1e-06`.

---

## B. Registry / inventory

### 4. 17 registered configurations; 15 without parity; SPRC checkpoint; CSMCIR composition deferred

**Verdict: REVISED.** 17/15 holds; CSMCIR-composition-deferred holds; **"SPRC's checkpoint is
absent" is wrong — `sprc_fiq_vitl.pt` is on disk with the registry's exact digest. What is absent is
the paper's ViT-g checkpoint.**

- 17 configurations, `workbench/registry/models.yaml`, `yaml.safe_load(...)['models']` → `COUNT 17`
  (`scripts/dump_registry.py`, 14 lines). `model_id` lines:
  `csmcir:28`, `airknow:133`, `conesep:208`, `ptha_mtst:315`, `habit:430`, `intent:527`, `hint:621`,
  `encoder:696`, `pair:795`, `clvc_net:876`, `dcnet:989`, `combiner_rn50x4_noft:1085`,
  `clip4cir_rn50x4_fullft:1168`, `tgcir:1265`, `sprc:1358`, `limn:1475`, `tme:1582`.
- 15 without formal parity evidence: `workbench/docs/reviews/20261111-master-audit/INVENTORY.csv:2-18`,
  `has_parity_artifact` = `Y` for exactly `encoder` and `limn`; `N` for the other 15.
- SPRC checkpoint **present**: `workbench/artifacts/checkpoint_selection/20261106T002617+0700/checkpoints/sprc/sprc_fiq_vitl.pt`,
  1 459 756 541 B, `sha256 19386104271c01ea03ee2177575c0e55f98edd3ba61a116d78f2e61ab5dbea08` —
  byte-equal to `models.yaml:1452 expected_sha256` (`status: URL_VERIFIED` at `:1453`), registered
  `expected_size_bytes: 1459756541`. Registry comment `:1423-1425` states why ViT-L is a scientific
  variant: *"The paper's row is ViT-g while the released Google Drive checkpoints are ViT-L (upstream
  issue #4, acknowledged by the author)"*. The ViT-g file is what is missing.
  `MODELS_REST_VERIFICATION.vi.md:459-462` says the same and calls the UI string "chưa tải
  checkpoint" a defect: *"trong khi `sprc_fiq_vitl.pt` đã tải, đã đo"*.
- CSMCIR: deferred as a **composition** target — `INVENTORY.csv:2` `has_composition_run=N`,
  `USER-DEFERRED (ngoài cohort 15)`; no `csmcir` dir under
  `workbench/artifacts/composition/summary/` (38 dirs enumerated, model set below). A historical
  native measurement exists (`checkpoint_selection/20261107T021500+0700/csmcir/measurement.json`,
  `counted=True artifact_backed=True`), consistent with the spec's "account for any valid
  author-checkpoint native result".
- **Which of the 17 the gate reports now:** only `encoder` and `limn` — `_discover_models` returns
  `['encoder', 'limn']`, both `PARITY_MANDATORY_SUBCHECK_FAILED`, `passed_models: []`. The other 15
  are absent from `models` entirely (no pair on disk), i.e. the endpoint has nothing to report for
  them, not `NOT_RUN` rows.
- **Native vs composition coverage** (recounted from artifacts, not from prose):

  | model | native rows (total / `counted` & with metrics) | composition `full.json` |
  |---|---|---|
  | csmcir | 2 / 1 (`20261107T021500+0700/csmcir/measurement.json`) | none (deferred) |
  | airknow | 10 / **3** (`airknow_run3cat_gpu_run{1..4}` + `airknow_fiq_n0{5,8}*` + `airknow_variant_80`) | none |
  | conesep | 1 / 1 | 2 COMPLETED |
  | ptha_mtst | 2 / 1 (+1 `ptha_mtst_transfer_blocker.json`, counted, no metrics) | 2 COMPLETED |
  | habit | 1 / 1 | 2 COMPLETED + 1 FAILED |
  | intent | 1 / 1 | 2 COMPLETED |
  | hint | 2 / 1 | 2 COMPLETED |
  | encoder | 1 / 1 | 6 COMPLETED |
  | pair | 13 / **4** (`pair_b1`/`pair_b2` × `H_normal`/`H_rev`) + 1 `pair_blocker.json` (counted, no metrics) + 4 uncounted mirror rows | none |
  | clvc_net | 5 / **2 in mutual conflict** (`clvc_run1.json`, `raw/clvc_run3cat_gpu.json`) | 6 PARTIAL |
  | dcnet | 2 / 1 (+1 `selection_decisions.json`, uncounted) | none |
  | combiner_rn50x4_noft | 1 / 1 | none |
  | clip4cir_rn50x4_fullft | 2 / **1** (`measurements/clip4cir_parsed.json`) | none |
  | tgcir | 4 / 1 | 3 COMPLETED + 2 PARTIAL |
  | sprc | 2 / 1 (`measurements/sprc_vitl_run1.json`) | none |
  | limn | 1 / 1 | 1 PARTIAL + 1 COMPLETED + 1 PARTIAL |
  | tme | 3 / 1 (+1 `tme_blocker.json` counted w/o metrics) | 2 COMPLETED |

  Native-only (no composition result): `csmcir, airknow, pair, dcnet, combiner_rn50x4_noft,
  clip4cir_rn50x4_fullft, sprc`. Composition-only: none. Both: the other 9 (`clvc_net` included, whose
  composition runs are all `PARTIAL`). Counts come from
  `workbench.backend.native_measurements.list_native_measurements` (56 rows, 50 blocks with metrics,
  23 with metrics served, `models_with_metrics 17`, `unbacked_counted 0`).

  **Caveat that must travel with this table:** `INVENTORY.csv` is dated
  `2026-11-11 03:16` (commit `be14393`) and records `has_composition_run=N` for `clvc_net` and
  `tgcir`, yet both have `full.json` summaries written `2026-11-12 05:16`/`09:15`. The CSV column is
  stale for those two; the artifact scan above is the corrected reading.

Repro:

```
cd /home/heheboiz/data/cir && python3 -c "
import yaml,json; ms=yaml.safe_load(open('workbench/registry/models.yaml'))['models']
print(len(ms), [m['model_id'] for m in ms])"
sha256sum workbench/artifacts/checkpoint_selection/20261106T002617+0700/checkpoints/sprc/sprc_fiq_vitl.pt
```

---

## C. ENCODER

### 5. Relation-state pin: wired through production on both sides, absent from the G1/G2 artifacts; `learnable_relation` is constructed, not loaded

**Verdict: REVISED.** The pin **is** wired through both production processes (one constant, imported
on the native side). What is *not* wired is the G1/G2 producer: `encoder_parity.py` is a third path
that never seeds, so the artifacts carry no `relation_state` and the gate must fail
`relation.<cat>.digest`.

- Pin constant, one definition — `workbench/composition/model_encoder.py:59` and `:65`:

  ```python
  RELATION_SEED = 20261111
  RELATION_EXPECTED_DIGEST = "20db0c27c979e9a0938c71af5142458dfa2070468d0ad3fcb6f24134c2f9ce92"
  ```
- Workbench path — applied immediately before construction, `model_encoder.py:500` /
  `:505-507`:

  ```python
  torch.manual_seed(relation_seed)
  ...
      model = model_try2.Encoder(
          hidden_dim=512, dropout=0.5, local_token_num=3, t=0.1,
          wc=2, N_p=2, weighted=True)
  ```

  digest computed at `:509-510` and compared to the pin in `_relation_record` (`:231-239`:
  `"matches_pinned_digest": response["relation_digest"] == RELATION_EXPECTED_DIGEST`,
  `"in_state_dict": False`). Enforced: `runner.py:1880-1883` raises `relation_state_unrecorded`,
  `runner.py:2011` hashes `relation_state` into the run identity, `runner.py:2465-2469` raises
  `relation_state_drift` if the effective post-encode digest differs.
- Native replay path — **the same constants, imported** (`workbench/replay/encoder.py:36`:
  `from workbench.composition.model_encoder import RELATION_EXPECTED_DIGEST, RELATION_SEED`),
  applied at `:324` (`torch.manual_seed(RELATION_SEED)`) immediately before the `Encoder(...)` call
  at `:330`, recorded by `_relation_record` (`:253-270`, same three fields), surfaced at `:433-435`.
- **Third path, unpinned:** the artifact producer `scripts/encoder_parity.py:388` builds
  `model_try2.Encoder(hidden_dim=512, …)` with no `torch.manual_seed`, and
  `grep -c "manual_seed\|RELATION_SEED\|relation_state" encoder_parity.py t4_parity.py` → `0`/`0`.
  The six `g1_g2.encoder.*.json` therefore have no `relation_state` key at any depth, so
  `parity_gate._relation_checks` (`:777`) reports `native=not_recorded, adapter=not_recorded` and,
  because `RELATION_DRAWING_MODELS = ("encoder",)` (`:267`), that is a **mandatory** failure
  (measured above).
- Origin of `learnable_relation`: **constructed at `__init__`, not a `state_dict` key.**
  `workbench/third_party/ENCODER/model_try2.py:93`:

  ```python
          self.learnable_relation = nn.Parameter(torch.randn(N_p, dim, dtype=torch.float32)).cuda()
  ```

  `.cuda()` on a `Parameter` yields a plain `Tensor`, so the attribute is not registered: no
  `state_dict()` entry, no restore. Measured in
  `raw/encoder_parity/relation_proof_construct.out`: `CHECKPOINT_KEY_COUNT 417`,
  `KEYS_MATCHING_learnable_relation []`, `KEYS_MATCHING_relation []`, `IN_STATE_DICT False`,
  `SHAPE [2, 512]`. Both loaders only restore the registered weights
  (`model_encoder.py:511-514` `state = load_trusted_whole_model(...)` then
  `model.load_state_dict(state, strict=False)`; `replay/encoder.py:351` likewise) and read the
  relation from the constructed model.

Repro:

```
cd /home/heheboiz/data/cir && grep -n "manual_seed\|RELATION_SEED\|relation_state" \
  workbench/docs/reviews/20261109-native15-composition-limn/scripts/encoder_parity.py   # → no match
grep -n "learnable_relation" workbench/third_party/ENCODER/model_try2.py
grep -a "IN_STATE_DICT\|CHECKPOINT_KEY_COUNT" \
  workbench/docs/reviews/20261109-native15-composition-limn/raw/encoder_parity/relation_proof_construct.out
```

### 6. The two historical rank disagreements are archived, and the stable query IDs are recoverable

**Verdict: HOLDS.** Indices and both ranks match the artifact exactly; the archived
`inputs.<cat>.json` maps index → reference image + caption, and that pair joins 1:1 to the CIRCUS
release. **Stable IDs: dress `7:1565`, shirt `7:3362`.**

- Rank dumps, `g1_g2.encoder.dress.adapter.json` → `adapter.scoring_ranking.rank_differences_vs_native_val_control`:
  `{"count": 1, "head": [1569], "classification": [{"index": 1569, "adapter_rank": 1216,
  "native_rank": 1215, "adapter_matrix_gap": 0.0006574243307113647,
  "native_matrix_gap": 0.0006571710109710693, "explained_by_tie_epsilon": false}], "non_tie_count": 1,
  "min_gap_abs": 0.0006571710109710693}` — matches "1569 (1215 vs 1216)".
- `g1_g2.encoder.shirt.adapter.json`: `{"count": 1, "head": [1352], "classification": [{"index": 1352,
  "adapter_rank": 1372, "native_rank": 1373, …, "explained_by_tie_epsilon": false}],
  "non_tie_count": 1}` — matches "1352 (1373 vs 1372)". `toptee` → `count 0`.
- Input manifest: `raw/encoder_parity/inputs.dress.json` / `inputs.shirt.json`, arrays
  `full_reference_ids`, `full_target_ids`, `full_query_texts`, `full_query_paths`,
  `reference_positions`, `target_positions`, all length 2017 / 2038.
  - `dress[1569]`: `full_reference_ids = B004DTTRA8`, `full_target_ids = B0048OYGF4`,
    `full_query_texts = "is more intimate and cami and has no pink in it"`, positions 1077 / 2205.
  - `shirt[1352]`: `full_reference_ids = B00AMD5I7U`, `full_target_ids = B005LIXPPI`,
    `full_query_texts = "is blue and has a lighter logo"`, positions 492 / 2238.
- Recovery is unambiguous: the reference id is unique in each list
  (`[j for j,x in enumerate(refs) if x==ref]` → `[1569]` and `[1352]`), and joining
  `(image stem, normalized caption)` against
  `workbench/artifacts/circus_release/mbeir_fashioniq_task7_test.jsonl` (6003 rows) yields exactly one
  hit each:

  ```
  dress 1569 B004DTTRA8 | is more intimate and cami and has no pink in it | hits [('7:1565', 'Is more intimate and cami and has no pink in it.')]
  shirt 1352 B00AMD5I7U | is blue and has a lighter logo | hits [('7:3362', 'Is blue and has a lighter logo.')]
  ```

  So **dress → `7:1565`, shirt → `7:3362`**. (The manifest itself stores only image ids, not M-BEIR
  qids; the qid is recovered from the release file, which is why the join step is mandatory and why a
  digest of that file belongs in the run identity.)

Repro:

```
cd /home/heheboiz/data/cir && python3 -c "
import json
b='workbench/docs/reviews/20261109-native15-composition-limn/raw/encoder_parity/'
for cat,i in (('dress',1569),('shirt',1352)):
    d=json.load(open(b+f'inputs.{cat}.json'))
    print(cat,i,d['full_reference_ids'][i],d['full_query_texts'][i])"
```

---

## D. LIMN / CLVC-Net

### 7. CLVC-Net shares LIMN's resolver; the key is `(reference stem, normalized edit caption)`

**Verdict: HOLDS** (path correction: the adapter lives at
`workbench/composition/model_clvc_net.py`, there is no `workbench/composition/adapters/` directory).

- Import, `model_clvc_net.py:68-71`:

  ```python
  from workbench.composition.model_limn import (
      caption_scope_index,
      resolve_query_scope,
  )
  ```

  with the contract stated at `:64-67` ("The routing contract is SHARED with LIMN, not reimplemented
  here …").
- Call site, `model_clvc_net.py:256-257`:

  ```python
          index = self.caption_scope_index()
          record = resolve_query_scope(index, query, candidate_names)
  ```
- Resolver, `model_limn.py:189-227`; the decisive lookup at `:213`:

  ```python
      holders = index.get((stem, normalize_query_text(getattr(query, "query_txt", ""))), frozenset())
  ```

  Index construction (`:150-153`) keys on `(Path(row["candidate"]).stem, text)` where
  `text = normalize_query_text(" and ".join(...))`; `normalize_query_text` (`:55-57`) casefolds,
  strips punctuation, collapses whitespace. `query_reference_stem` (`:172-177`) supplies the stem.
- Therefore the key is **`(reference stem, normalized edit caption)`, not stem-only** — the resolver
  **does consume the edit caption**, and two captions on the same reference can map to different
  categories. That is exactly the 117/2 mechanism: a reference present in two val splits resolves only
  when one `cap.<cat>.val.json` row reconstructs the queried text; when both files carry a
  byte-identical caption pair, `len(holders) > 1` → `unresolved:annotation_names_two_categories`
  (`:223-225`).

### 8. `7:3038` and `7:3790` are the only 2 of the historical 119 still unscoped

**Verdict: HOLDS.**

- `workbench/docs/reviews/20261111-master-audit/raw/m02/m02_recount.json:288-297`:

  ```json
    "ambiguous_qid_count": 119,
    "resolved_count": 117,
    "unresolved_count": 2,
    "resolved_scope_counts": { "shirt": 46, "toptee": 71 },
    "unresolved_qids": [ "7:3038", "7:3790" ],
  ```

  with self-checks `:340-344` (`"ambiguous_count_is_119": true, "resolved_count_is_117": true,
  "unresolved_are_exactly_7:3038_7:3790": true`).
- Annotation digests, `m02_recount.json:7-10`: `dress 5e5117d4…b3adb7de9`,
  `shirt 7b7ca379…febacea6d7feb`, `toptee b4e09f42…f15cdc5eb` — identical to the digests the live
  resolver records in
  `workbench/artifacts/composition/derived/category_routing_inputs/clvc_net_da0a2162be0e159a1a456cec_full.json`
  (`annotation.files.<cat>.sha256`, `"complete": true`), so the counts and the digests belong to the
  same bytes.
- Recovered-117 list: the same derived artifact, `routing` block — `resolver_key =
  reference_stem+edit_caption`, `loader_scoped_count = 5884`, `resolver_routed_count = 117`,
  `still_unscoped_count = 2`, `artifact_adapter_routed = 6001`, `replay_agrees_with_artifact = true`,
  `resolver_routed_qids` = 117 entries, of which `7:3038 in? False`, `7:3790 in? False`.
  Independent recount `m02_resolver.txt:42-45`; per-qid map `qid_scope_map.json` (`by_reason
  caption_annotation 117 / loader_identity 5884`, `scored 6001`, `unresolved 2`).
- Level distinction to keep: `provenance_category_resolution.unresolved_count = 0` (loader left no
  out-of-scope category) vs resolver-level `unresolved_count = 2`. Both true, different layers.

### 9. CLVC-Net's two conflicting native R@10 artifacts

**Verdict: REVISED.** Both artifacts exist and disagree by 0.0501 pp as charged, **but neither records
`mrr`**, and batch / mask / checkpoint digest are absent from both.

| | A `workbench/docs/reviews/20261111-master-audit/raw/clvc_run3cat_gpu.json` | B `workbench/artifacts/checkpoint_selection/20261106T002617+0700/measurements/clvc_run1.json` |
|---|---|---|
| macro R@10 | **30.02079703540087** | **29.970746517160162** |
| macro R@50 | **58.67189400437203** | **58.62171212901515** |
| macro mean | 44.34634551988645 | 44.29622932308766 |
| mrr | **absent** (no `mrr`/`MRR` key; only `r1/r10/r50`) | **absent** |
| per-cat R@10 (dress/shirt/toptee) | 29.449677739216657 / 27.77232580961727 / 32.840387557368686 | 29.350520575111553 / 27.77232580961727 / 32.78939316675166 |
| script | `workbench/replay/clvc_net.py` (`-m workbench.replay.clvc_net`) | `checkpoint_selection/…/scripts/run_clvc.py` |
| checkpoint | paths only (`…/checkpoints/clvc/{dress,shirt,toptee}_{local,global}_model.pt`), `source_commit bd9b6889489806cb60676597880f91d918b279cb`, `native_evaluator test.py:test`, `runtime_label NATIVE_CUDA`, `device cuda` | paths only (`local`/`global`), `source_commit bd9b6889…79cb`, `native_evaluator` missing |
| split | `fashioniq_val_split`, `split=val`, per-category pair-union galleries 2628/3089/2902 | same protocol; same galleries |
| batch | **not in artifact**; prose only: batch 32 (`BLOCKED_MODELS_VERIFICATION.vi.md:310`) | **not in artifact**; command passes `48` |
| mask | **not in artifact**; pinned evaluator masks the reference row at `test.py:135` | **not in artifact** |
| recorded_at / replay_type | `2026-11-10T13:31:28.628383+00:00` / `SOURCE_FAITHFUL_WORKBENCH_REPLAY` | none |

Both are still `counted: true` with mutual `conflict_with` (verified: two rows, `counted=True
backed=True`, same `protocol_id`). The second run pair of A (`clvc_run3cat_gpu_run1.json`,
`_run2.json`) is byte-identical in the numbers (`Δ +0.0000`), so A is repeatable; B's log
(`$D1/logs/clvc_run1.log`) is not in this tree.

Reproducibility from recorded commands:

- **A**: command recorded at `workbench/docs/reviews/20261111-master-audit/BLOCKED_MODELS_VERIFICATION.vi.md:214-218`
  (`… -m workbench.replay.clvc_net --checkpoint-root … --dataset-root data/FashionIQ --source-root
  workbench/third_party/CLVCNet --batch-size 32 --output /tmp/clvc_run_gpu.json`), execution record
  `:310` (`EXIT=0`, wall 235 s, batch 32, cache warm). The recorded `--output` is a `/tmp` path, so
  the command→artifact link is prose, not a logged invocation; no `RUN_LOG.csv` exists in
  `20261111-master-audit/`.
- **B**: command recorded at `CHECKPOINT_SELECTION_REPORT.md:1697-1699`
  (`$CIR/workbench/envs/airknow-adapted/bin/python "$EVID/scripts/run_clvc.py" "$EVID/checkpoints/clvc"
  "$DS" "$EVID/measurements/clvc_rerun.json" 48`), table row `:1904`. Note the recorded output name is
  `clvc_rerun.json`, not `clvc_run1.json`.

So: **both are reproducible in principle from recorded commands; neither is reproducible as a
byte-identical *artifact* from its recorded command, and no `mrr` exists to compare.** The
0.0501 pp gap is 1–2 queries per category (dress R@1 137→138, R@10 592→594, R@50 1146→1145;
`102-queue-gpu-and-blocker-inventory.md:227`), i.e. a tie-break/order difference, not a provenance
difference — `per_query_published: false` in both files.

### 10. LIMN parity: matching ranks, missing hit-count evidence

**Verdict: HOLDS.** Ranks do match (`rank_differences 0`) and hit counts are absent on **both** sides.
One caveat that matters for the repair: the record's G1/G2 verdicts **cannot be quoted from the prose
as a pass** — `INPUT_FORWARD_PARITY.vi.md` says PASS, the gate says
`PARITY_MANDATORY_SUBCHECK_FAILED` (batch `max_rel`), and `metrics_parity` is `missing`.

- LIMN parity record = the six `g1_g2.{dress,shirt,toptee}.{native,adapter}.json` files (section 3
  table) plus the prose verdict `INPUT_FORWARD_PARITY.vi.md:255-260`:

  ```
  | **G1 input parity** | **PASS** | gallery order, reference/target ids, and the model text are exact against the annotations …
  | **G2 forward parity** | **PASS (absolute)** | … batch-1-vs-16 `max_abs ≤ 2.0e-07` but its declared `max_rel ≤ 1e-3` is **FAILED** for a documented denominator reason
  | **G2 scoring parity** | **PASS** | score `max_abs ≤ 5.4e-07` (declared 1e-6)
  | **G2 ranking parity** | **PASS** | **0** rank differences vs the native aggregate ranks in all three categories; … (47 total, gap 0.0, 0 unexplained)
  ```

  The artifact fields that back "matching ranks" (adapter `layer2_4_forward.scoring_ranking`):
  `rank_differences = 0`, `unexplained_rank_differences = 0`, `max_abs_rank_delta = 0`,
  `native_aggregate_ranks_compared = 2017/2038/1961`, `first_unexplained_index = None`,
  `score_max_abs_delta_vs_dot = 5.36e-07 / 5.36e-07 / 4.17e-07` vs `score_max_abs 1e-06`.
- **The gate does not agree with the prose** (measured at HEAD, section 1):
  `limn PARITY_MANDATORY_SUBCHECK_FAILED failures=3 missing=['metrics_parity']`, the three failures
  being `dress/shirt/toptee.batch_1_vs_n.max_rel`. Machine record:
  `workbench/docs/reviews/20261111-master-audit/web_evidence/w05_parity_gate_after.json` →
  `models.limn` `"status": "PARITY_MANDATORY_SUBCHECK_FAILED"`, `"missing": ["metrics_parity"]`,
  `"failure_count": 3`; `GUARDS_FIXES.vi.md:448-478` documents the same contradiction.
- **Hit counts: absent on both sides.** The gate's `metrics_parity` is fed only by a boolean
  `native_metrics_match_feature_rerank`, `parity_gate.py:693-704`:

  ```python
      if "native_metrics_match_feature_rerank" not in native:
          return []
  ```

  `grep -c native_metrics_match_feature_rerank g1_g2.{dress,shirt,toptee}.{native,adapter}.json` →
  all `0` (the three `encoder.*.native.json` files do carry it once each, which is why ENCODER's
  `metrics_parity` is `true` and LIMN's is `missing`). Neither side stores `hits` / `hit_count` /
  `top_k_hits`; the adapter stores only recall percentages (`adapter_style_recall`,
  `native_style_recall`, e.g. dress 50.52057511155181 vs 51.165096678235, r50 identical
  74.26871591472484). So `metrics_parity` is **`missing`, not measured-and-equal** — exactly the
  premise the campaign wants recomputed.

---

## E. PAIR

### 11. PAIR's blocker, checkpoint contents, and whether 5.20 % / 2×4 are measured

**Verdict: REVISED.** The three facts hold (classes unpublished; checkpoint geometry measured;
5.20 % measured) but the **blocker string quoted is not the registry's** — the registry says
`BLOCKED_CHECKPOINT_MAPPING_UNVERIFIED`, and `BLOCKED_MISSING_SOURCE_CLASSES` is the replay/audit
verdict.

- Registry: `models.yaml:825` `reproduction_status: BLOCKED_CHECKPOINT_MAPPING_UNVERIFIED`;
  `:828` `command_status: COMMAND_BLOCKED_SOURCE_ENTRYPOINT`; both variants
  (`pair_b1` `:800-…`, `pair_b2` `:865-…`) `checkpoint_mapping_status: UNVERIFIED`.
  `workbench/composition/model_pair.py:3-12` refuses on `UNVERIFIED`.
- The string `BLOCKED_MISSING_SOURCE_CLASSES` is produced at `workbench/replay/pair_absence.py:220`
  and stored machine-readable at
  `workbench/docs/reviews/20261111-master-audit/raw/pair_absence_evidence.json:3`, with
  `required_globals` `:4-12` = `model_vssp.{Encoder,Backbone,FeatureWiseAffine}` +
  `MHTransformer_dis.{Transformer,TransformerLayer,MultiHeadAttention,MlpTransformer}`, every one
  `false` in `required_globals_present_as_module_or_package` (`:13-21`). Asserted by
  `workbench/tests/test_blocked_models_verification.py:158`. Human statement:
  `20261111-master-audit/raw/verify/102-queue-gpu-and-blocker-inventory.md:224` (`PAIR-SRC`), which
  also records the upstream check: full clone at `/tmp/pair_up` = `bfb0b98cdb…dd888f`,
  `git rev-list --all --count` → **21** commits, `git grep` on every rev → 0 hits (verified live:
  `/tmp/pair_up` exists, HEAD `bfb0b98…`, count `21`).
  `COMPOSITION_BLOCKERS.vi.md` does **not** mention PAIR at all.
- Checkpoint contents (measured, stub-unpickler walk
  `workbench/docs/reviews/20261108T114646+0700-ckpt-mapping/.raw/fp2_pair_B1.json`, root class
  `:17` `"root_class": "model_vssp.Encoder"`):
  - `backbone.transformer_a` → **2 layers × 4 heads**: `:2183`
    `"backbone.transformer_a.layers.0.attn.num_heads": 4`, `:2217`
    `"backbone.transformer_a.layers.1.attn.num_heads": 4`, layer indices `[0, 1]` only;
    `"enc_dec": false`.
  - `transformer_b` 2×4; `transformer_c`, `transformer_d` present, 2 layers each;
    `weight_chanelImg`, `weight_chanelImg_t`, `cross_t2i` present (all `true` in the artifact's
    module tree) — none of which the pinned class declares.
  - `backbone.clip` ViT-B/32 (`conv1.weight [768,3,32,32]`, `positional_embedding [77,512]`,
    `vocab 49408`), `backbone.transformer` 1 layer × 8 heads.
  - Digests on disk: `pair-B1.pt` 694 958 565 B `c4d861ecd0baa3f1427fc7a853943f5614baf87def1cdd9c4bb7583ee925fbf2`
    ( = registry `pair_b1.expected_sha256`), `pair-B2.pt` 694 872 371 B
    `74346fd7de6efdafe8dde70458316286a60ceb2d7b419efed55193418aeb3f03` ( = `models.yaml:872`).
- **Measured, with an explicit caveat:** `20261109-native15-composition-limn/scripts/pair_rederive.out`
  contains `MISSING_IN_CKPT: 0` / `UNEXPECTED_IN_CKPT: 74` / `SHAPE_MISMATCH: 70` /
  `PARAMS_TOTAL: 172814185` / `PARAMS_BYPASSED: 8986448` /
  `PCT_BYPASSED_PLUS_UNCLASS: 5.2001` (recomputed: `8986448/172814185 = 5.2000639…` %). Same numbers in
  `UNFINISHED.vi.md:126,129`, `CKPT_MAPPING.vi.md:369,760`, `EVIDENCE_INDEX.md:296-298,374`.
  The 5.20 % therefore **is** measured, but against a **hypothesised** public class (the alias loader
  stands in for the absent `model_vssp`), so it quantifies the gap to the nearest public class, not
  the author's geometry. The 2-layer/4-head `transformer_a` is measured **from the checkpoint**, not
  against a hypothesis, and is the stronger evidence.

---

## F. Reports cited by the prompt

### 12. `FINAL_REPORT_20261112.vi.md` blocks, and the two "neighbourhood" lines

**Verdict: REVISED.** The blocks exist and are quoted below; **§0.1 is a `cd9ced2` snapshot while
HEAD is `40c897c`**, and the `PARITY.vi.md:219` / `102-…md:220` pointers land on a blank line and on a
section header, not on the content the prompt expects.

§0.1 live-state block, `workbench/docs/reviews/20261111-master-audit/FINAL_REPORT_20261112.vi.md:73-106`
(header at `:73`, fence `:78-106`):

```
### 0.1 Số liệu sống **đo lại** lần cuối (2026-11-12T13:36+07)
…
```
```text
$ date -Iseconds                                                           # :79-80
2026-11-12T13:36:19+07:00
$ git rev-parse HEAD                                                       # :82-83
cd9ced2155a2d98e9021c549b4ca7310f3fac51b
$ git rev-parse origin/research/cir-failure-workbench                      # :85-86
cd9ced2155a2d98e9021c549b4ca7310f3fac51b
$ git rev-list --left-right --count origin/research/cir-failure-workbench...HEAD   # :88-89
0       0
$ systemctl --user show -p MainPID -p ActiveEnterTimestamp -p NRestarts \  # :91-95
    cir-workbench-backend.service cir-workbench-frontend.service
MainPID=1758096
NRestarts=0
ActiveEnterTimestamp=Thu 2026-11-12 12:20:57 +07
                                                            # (frontend block, :96-99: MainPID=1221148, NRestarts=0, 23:48:44 +07)
$ pgrep -af composition_eval.py                                            # :101-102
(none)
$ nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader    # :104-105
373 MiB, 16303 MiB
```
Porcelain snapshot at `:108-113` (`?? CLEANUP_REPORT.md.save`).

Live re-measure now: HEAD `40c897c31bbdd76b5a76f65935c7dd0f7fef0e94` = origin (`0	0`), backend
`MainPID=1758096 NRestarts=0`, frontend `MainPID=1221148 NRestarts=0`, GPU `903 MiB, 16303 MiB`,
served bundle `assets/index-NnjrdZLG.js`. So §0.1's **structural** claims (two units, same MainPIDs,
idle GPU, no composition process) reproduce; its **HEAD/GPU-memory digits** are a past snapshot — the
report says so itself at `:4-6` and `:118-121`.

§4.4b metrics block, `:990-997` (verbatim, header `:965`):

```
Metric theo điều kiện, **6001/6003 chấm được**, coverage **`0.9996668332500417`** (đọc từ
`metrics.<role>`, với `metrics.image_only.condition == "clvc_image_encoder_only"`):

| condition | R@1 | R@10 | R@50 | mrr | ndcg_full |
|---|---|---|---|---|---|
| `multimodal` | 1.3997667055490752 | 7.198800199966672 | 18.180303282786202 | 0.03564730414020548 | 0.14921544731411218 |
| `text_only_black_image` | 0.11664722546242293 | 1.3997667055490752 | 4.49925012497917 | 0.006595896893970061 | 0.09988226530955799 |
| `clvc_image_encoder_only` | 0.11664722546242293 | 1.1998000333277787 | 2.9495084152641224 | 0.005437399476599689 | 0.08644584977266369 |
```

Independently re-read from
`workbench/artifacts/composition/summary/clvc_net_author_drive_unresolved_da0a2162be0e159a1a456cec/full.json`:
every digit matches, `execution_status PARTIAL`, `query_count 6003`,
`composition_gap_ndcg 0.33061712371309204`, `scored_count 6001` per condition. §4.4b's own caveat at
`:999-1001` (cohort 6001 contains 117 caption-routed queries; strict caption-free cohort B = 5884) is
the load-bearing qualification.

§9 command table, first 5 data rows, `:1387-1393` (header `:1387-1388`):

```
| khẳng định | lệnh | kết quả mong đợi / đã thấy |
| HEAD và khoảng cách với `origin` (§0.1) | `git rev-parse HEAD` · `git rev-parse origin/…` · `git rev-list --left-right --count …` | **đo lại:** `cd9ced2155a2d98e9021c549b4ca7310f3fac51b` · cùng sha · `0<TAB>0` — đã push |
| Hai unit sống đúng MainPID (§0.1, §1.5) | `systemctl --user show -p MainPID -p ActiveEnterTimestamp -p NRestarts cir-workbench-backend.service cir-workbench-frontend.service` | **đo lại:** backend `1758096` / `Thu 2026-11-12 12:20:57 +07` / `0` · frontend `1221148` / `Wed 2026-11-11 23:48:44 +07` / `0` |
| Không còn run composition nào chạy (§0.1, §4.4b) | `pgrep -af composition_eval.py` | **đo lại:** rỗng `(none)` — mọi run đã thoát |
| GPU đang bận bao nhiêu (§0.1) | `nvidia-smi --query-gpu=…` · `nvidia-smi --query-compute-apps=…` | **đo lại:** `… 373 MiB, 16303 MiB` · không có compute app |
| TG-CIR Full là `COMPLETED` 6003 (§4.1b, §7) | `python3 -c "import json;d=json.load(open('…/tgcir_tgcir_archive_unresolved_7b1fa85007bffdab931f7f19/full.json'));print(…)"` | `COMPLETED 6003 {'label': None, 'limited': False, 'query_limit': None} 0.43853620852911723` |
```

Row 5 reproduces exactly (`tgcir COMPLETED 6003 0.43853620852911723`).

- `workbench/docs/reviews/20261110-e2e-repair/PARITY.vi.md:219` is **blank**. The neighbourhood the
  prompt wants is `:214-232`: heading `:214` `## R2 × R9 — cái nào thật sự là failure`, `:220`
  `**Sàn native-vs-native (cùng model, cùng query, khác transform):**`, table `:222-226`
  (`dress max_abs 0.120392, mean_abs 0.003350`; `shirt 0.122951 / 0.003715`;
  `toptee 0.125173 / 0.003375`), and `:228-232` the classification of the four failing blocks
  (`NOT_COMPARABLE`, `NOT_COMPARABLE`, `DENOMINATOR_ARTEFACT`, `GENUINE_FAIL`). Note `:210` claims
  `encoder 2 rank non-tie` while the gate now counts **1 per category** (`dress` `non_tie_count 1`,
  `shirt` `1`, `toptee` `0` → 2 total, consistent), and `:249-258` gives the LIMN G2 recount
  ("3 bảo vệ được", `1.140e-02 / 2.698e-03 / 7.509e-03` vs declared `1e-3`), which matches the
  artifacts measured above.
- `workbench/docs/reviews/20261111-master-audit/raw/verify/102-queue-gpu-and-blocker-inventory.md:220`
  is the **section header** `### 7.2 Genuinely open blockers`; its table starts `:222`. The row the
  prompt is after is `:224` `**PAIR-SRC** | BLOCKED_MISSING_SOURCE_CLASSES …` (quoted in section 11).
  Nearby and equally load-bearing: `:225` `G1G2-15`, `:227` `CLVC-NUM`, `:228` `LIMN-2`.

### 13. Bundle identity

**Verdict: HOLDS** — docs still say `index-DDviexaW.js`, and the currently served bundle is
`index-NnjrdZLG.js`, so those lines are stale.

`grep -rn "index-DDviexaW\|index-NnjrdZLG" workbench/docs | head -40` → 14 matches / 10 files;
9 mentions of the stale name in 7 files:

| file | stale mentions |
|---|---|
| `workbench/docs/reviews/20261111-master-audit/FINAL_REPORT_20261112.vi.md` | 2 (`:274`, `:1415`) |
| `workbench/docs/reviews/20261111-master-audit/raw/verify/102-report-staleness.md` | 1 (`:256`) |
| `workbench/docs/reviews/20261111-master-audit/raw/verify/105-sweep-1d5a66c1.md` | 2 (`:25`, `:26`) |
| `workbench/docs/reviews/20261111-master-audit/raw/verify/105-sweep-preflight-1d5a66c1.txt` | 1 (`:59`) |
| `workbench/docs/reviews/20261111-master-audit/raw/verify/106-sweep-117fc172.md` | 1 (`:61`) |
| `workbench/docs/reviews/20261111-master-audit/raw/verify/106-sweep-preflight-117fc172.txt` | 1 (`:59`) |
| `workbench/docs/reviews/20261112-parity-repair/BASELINE.md` | 1 (`:366`) |

`:274` reads *"bundle đang phục vụ hiện nay là `assets/index-DDviexaW.js`"* — a live-state claim, and
false now. `:1415` (the §9 table row "Bundle đang phục vụ (§1.5)") quotes the same name plus
`sha256 58eed6d48640fa1f93d2a64a96d87812a044a3b07b324fb872f6126162bf5369`, 392 261 B, mtime
`09:34:18.594`. Live measurement now:

```
$ curl -s http://127.0.0.1:5173/ | grep -o 'assets/index-[^"]*\.js'
assets/index-NnjrdZLG.js
$ curl -s http://127.0.0.1:5173/assets/index-NnjrdZLG.js | sha256sum
721d34d043858e80c88e76d72866ba8583508318baeebac599cebd1e1c2c80d0  -
$ sha256sum workbench/frontend/dist/assets/index-NnjrdZLG.js
721d34d043858e80c88e76d72866ba8583508318baeebac599cebd1e1c2c80d0  workbench/frontend/dist/assets/index-NnjrdZLG.js   # 392 769 B, 2026-11-12 12:20:56
```

So the served and `dist/` bundles agree with each other and with
`raw/verify/107-sweep-aea143bd.md:25-26` / `107-sweep-preflight-aea143bd.txt:31-33`
(`served=assets/index-NnjrdZLG.js`). The `105`/`106` sweep records are historical and correctly
name their own bundle; only `FINAL_REPORT:274,:1415`, `102-report-staleness.md:256`, and
`BASELINE.md:366` present the old name as current.

---

## What the campaign can actually do

Premises that hold, in the order worth spending time on. Each is CPU-or-light-GPU and does not
require a new full evaluation.

1. **Make the G1/G2 producer write the relation state (ENCODER) — no GPU needed beyond the existing
   capture.** The pin already exists and is shared (`model_encoder.py:59,65`, imported at
   `replay/encoder.py:36`); the gap is that `scripts/encoder_parity.py:388` constructs `Encoder`
   unseeded and never emits `relation_state`. Add `torch.manual_seed(RELATION_SEED)` +
   `_relation_record`-shaped output to that harness, re-emit the six artifacts, and the gate's
   `relation.<cat>.digest` becomes a real check rather than `not_recorded`. This is the cheapest
   genuine unblock on the board and needs the encoder sample capture only, not a full eval.
2. **Recompute LIMN hit counts from stored features (premise 10).** Ranks already match
   (`rank_differences 0`) and the artifacts already hold `native_style_recall` /
   `adapter_style_recall`; what is missing is the boolean/`hits` evidence. This is deterministic
   re-scoring of `raw/encoder_parity/*.npz` + `raw/native/*.json`, on CPU. It closes LIMN's only
   `missing` dimension (`metrics_parity`).
3. **Resolve the v2 gap deliberately (premise 2).** Because the writers emit no `standard` field and
   only `max_abs`/`max_rel`, "implement v2" is a **contract change**, not a patch: per-component
   capture must be added to the producers *and* a componentwise evaluator added to the gate
   (`evaluate_abs_rel` exists at `:343`, unused). Do the writer and reader in one versioned change, and
   keep v1 verdicts readable (the gate already does). Do **not** ship a v2 label over `max_abs` data —
   the gate correctly marks that malformed.
4. **Two stable-ID regressions (premise 6) — ready to run.** `dress 7:1565` (index 1569, 1215 vs
   1216) and `shirt 7:3362` (index 1352, 1373 vs 1372). The manifest and the release file both exist;
   the join is unambiguous. Recompute target rank under the real eligibility/tie policy from archived
   features; no GPU.
5. **LIMN/CLVC resolver work is already shared and already tested (premises 7, 8).** Any further
   `7:3038` / `7:3790` work is provenance research only — the code path is one function with a
   `(stem, normalised caption)` key; there is nothing to "unify". Treat the 2 as permanently unscoped
   unless upstream publishes category metadata.
6. **PAIR exploratory work can only be exploratory (premise 11).** Measured facts are solid
   (checkpoint `transformer_a` 2×4, `transformer_c/d` + `weight_chanelImg*` + `cross_t2i` present,
   74 unexpected / 70 shape mismatch / 5.2001 % bypassed). But the percentage is against a
   hypothesis, and no revision of the public repo has the classes. Do not upgrade PAIR's status from
   an exploratory mapping.

Must be re-scoped before any GPU time is spent:

- **"SPRC's checkpoint is absent" (premise 4).** It is present and digest-verified
  (`1938610…bea08`, 1 459 756 541 B); only the paper's ViT-g file is missing. Re-scoping to "no ViT-g
  checkpoint" is honest; scheduling a download for a file already on disk is wasted work.
- **"Reproduce CLVC's two native numbers with a controlled rerun" (premise 9).** A third macro number
  cannot settle it: neither artifact publishes per-query ranks and neither records `mrr`, batch, mask
  or a checkpoint digest. The rerun must be specified as *emit per-query ranks + tie-break rule +
  batch/mask provenance*, or it will reproduce the same unresolvable conflict.
- **"Implement v2 and generate new v2 evidence" (premise 2).** Generating evidence needs a writer
  change first; a GPU run today would emit v1-only artifacts that the gate must keep reporting under
  v1.
- **`INVENTORY.csv` as the scheduling input (premise 4).** Its `has_composition_run` column predates
  the `clvc_net`/`tgcir` summaries (CSV mtime `2026-11-11 03:16`, summaries `2026-11-12 05:16`/`09:15`).
  Re-derive the matrix from the artifacts before assigning jobs.
- **Anything that cites `FINAL_REPORT_20261112.vi.md:274`/`:1415` for the live bundle (premise 13).**
  Those lines name `index-DDviexaW.js`; the served and `dist/` bundle is `index-NnjrdZLG.js`
  `721d34d0…80d0`. Fix the report line before it is used as a live-state citation.
- **Any claim that the ENCODER pin is "not wired through both processes" (premise 5).** It is wired;
  the defect is in the producer harness. Scoping "wire the pin" would duplicate working code and miss
  the actual gap.

### Exact repro commands used in this audit

```
cd /home/heheboiz/data/cir
git rev-parse HEAD origin/research/cir-failure-workbench
git rev-list --left-right --count origin/research/cir-failure-workbench...HEAD
git status --porcelain

# gate, both standards, and the v2 request/applied split
python3 -c "import sys;sys.path.insert(0,'.');from workbench.backend.parity_gate import evaluate_all;
r=evaluate_all();print(r['standard'],r['requested_standard'],r['passed_models']);
[print(m,v['status'],v['failure_count'],v['missing']) for m,v in r['models'].items()]"
python3 -c "import sys;sys.path.insert(0,'.');from workbench.backend import parity_gate as g;
print(g.evaluate_all(standard=g.STANDARD_ABS_REL_V2)['advisory_standards'])"

# discovery surface and identity fields
python3 -c "import sys;sys.path.insert(0,'.');from workbench.backend import parity_gate as g;
print(g.evidence_root());print(g._discover_models(g.evidence_root()))"

# registry count and statuses
python3 -c "import yaml;ms=yaml.safe_load(open('workbench/registry/models.yaml'))['models'];
print(len(ms));[print(m['model_id'],m.get('reproduction_status'),m.get('command_status'),
bool(m.get('composition_adapter'))) for m in ms]"

# SPRC checkpoint, byte-verified against the registry digest
sha256sum workbench/artifacts/checkpoint_selection/20261106T002617+0700/checkpoints/sprc/sprc_fiq_vitl.pt
python3 -c "import yaml;m=[x for x in yaml.safe_load(open('workbench/registry/models.yaml'))['models']
if x['model_id']=='sprc'][0];[print(c['checkpoint_id'],c.get('expected_sha256'),c.get('expected_size_bytes'))
for c in m['checkpoint_variants']]"

# relation pin: definition, both production sites, and the unpinned producer
grep -n "RELATION_SEED\|RELATION_EXPECTED_DIGEST\|manual_seed" workbench/composition/model_encoder.py workbench/replay/encoder.py
grep -c "manual_seed\|RELATION_SEED\|relation_state" workbench/docs/reviews/20261109-native15-composition-limn/scripts/encoder_parity.py
grep -n "learnable_relation" workbench/third_party/ENCODER/model_try2.py
grep -a "CHECKPOINT_KEY_COUNT\|IN_STATE_DICT\|SHAPE " workbench/docs/reviews/20261109-native15-composition-limn/raw/encoder_parity/relation_proof_construct.out

# archived rank disagreements -> stable IDs
python3 -c "import json;b='workbench/docs/reviews/20261109-native15-composition-limn/raw/encoder_parity/';
[print(c,i,json.load(open(b+f'inputs.{c}.json'))['full_reference_ids'][i],
json.load(open(b+f'inputs.{c}.json'))['full_query_texts'][i]) for c,i in (('dress',1569),('shirt',1352))]"
python3 -c "import json;rows=[json.loads(l) for l in open('workbench/artifacts/circus_release/mbeir_fashioniq_task7_test.jsonl')];
norm=lambda s:' '.join(s.lower().replace('.',' ').split());
print([r['qid'] for r in rows if r['query_img_path'].endswith('B004DTTRA8.jpg') and norm(r['query_txt'])==norm('is more intimate and cami and has no pink in it')],
      [r['qid'] for r in rows if r['query_img_path'].endswith('B00AMD5I7U.jpg') and norm(r['query_txt'])==norm('is blue and has a lighter logo')])"

# resolver sharing + key
grep -n "model_limn import\|resolve_query_scope\|caption_scope_index" workbench/composition/model_clvc_net.py
grep -n "index.get((stem\|normalize_query_text\|Path(row\[\"candidate\"\]).stem" workbench/composition/model_limn.py

# 119 / 117 / 2
python3 -c "import json;d=json.load(open('workbench/artifacts/composition/derived/category_routing_inputs/clvc_net_da0a2162be0e159a1a456cec_full.json'))['routing'];
print(d['resolver_key'],d['loader_scoped_count'],d['resolver_routed_count'],d['still_unscoped_count'],len(d['resolver_routed_qids']),
'7:3038' in d['resolver_routed_qids'],'7:3790' in d['resolver_routed_qids'])"

# CLVC native pair (two conflicting numbers, and the absence of mrr/batch/mask)
python3 -c "import json;
[print(p, json.load(open(p))['aggregate']) for p in
 ('workbench/artifacts/checkpoint_selection/20261106T002617+0700/measurements/clvc_run1.json',
  'workbench/docs/reviews/20261111-master-audit/raw/clvc_run3cat_gpu.json')]"
grep -c '"mrr"' workbench/artifacts/checkpoint_selection/20261106T002617+0700/measurements/clvc_run1.json \
  workbench/docs/reviews/20261111-master-audit/raw/clvc_run3cat_gpu.json

# PAIR blocker + geometry + 5.20%
cat workbench/docs/reviews/20261111-master-audit/raw/pair_absence_evidence.json
grep -n "transformer_a.layers.[01].attn.num_heads" workbench/docs/reviews/20261108T114646+0700-ckpt-mapping/.raw/fp2_pair_B1.json
grep -n "PARAMS_TOTAL\|PARAMS_BYPASSED\|PCT_BYPASSED" workbench/docs/reviews/20261109-native15-composition-limn/scripts/pair_rederive.out
sha256sum workbench/artifacts/checkpoint_selection/20261106T002617+0700/checkpoints/pair/*.pt

# bundle identity
grep -rn "index-DDviexaW\|index-NnjrdZLG" workbench/docs | head -40
curl -s http://127.0.0.1:5173/ | grep -o 'assets/index-[^"]*\.js'
curl -s http://127.0.0.1:5173/assets/index-NnjrdZLG.js | sha256sum
sha256sum workbench/frontend/dist/assets/index-NnjrdZLG.js
```
