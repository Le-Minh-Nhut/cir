# MODEL_MATRIX — per-configuration disposition (2026-11-12)

Machine-readable twin: `MODEL_MATRIX.csv` (same 25 columns, same order, every cell identical).
Rows: **17 registered configurations in registry order** plus **1 in-scope FashionIQ native target
that is not a registry id** — `clip4cir_rn50x4_noft`, the noft sibling of the registered
`combiner_rn50x4_noft` (same upstream `ABaldrati/CLIP4Cir`, same checkpoint bytes, separate row
because it carries its own native macro) = **18 rows**. Registry order is the order of
`workbench/registry/models.yaml`; the extra row is last and its `blocker_code` is `ROLLUP_ONLY`.

Tree: `/home/heheboiz/data/cir`, branch `research/cir-failure-workbench`, HEAD
`40c897c31bbdd76b5a76f65935c7dd0f7fef0e94`. Every cell below is read from an artifact, a registry
field, or a command run in this session. `INVENTORY.csv` is used **only** as a cross-check (§
"Cross-check against INVENTORY.csv"), never as a source. The gate (`workbench.backend.parity_gate`,
`evaluate_all()`) is authoritative for G1/G2; where prose disagrees the gate wins and the prose is
named in § "Superseded".

**Snapshot note (read this before quoting a G1/G2 cell).** The `g1_g2.*` evidence set and
its producers (`scripts/t4_parity.py`, `scripts/encoder_parity.py`) carry **uncommitted
working-tree changes made by other workers during this session**: twelve artifacts were rewritten at
`20:07–20:15 +0700`, and a **third model (`hint`) gained a brand-new six-file paired set** (untracked;
`parity_gate.py` also gained v2 support, commits `23e5b8f`/`84812a4`). Discovery therefore now returns
**three** ids, and the gate reads each pair under the standard that pair declares — `encoder`/`limn`
under `parity-relative/v1`, `hint` under `parity-absolute-relative/v2`, i.e. *after*
this audit's premise pass. The G1/G2 cells below are the gate's verdict on the **working tree as it
is now**, because that is what `GET /api/runs/parity-gate` serves. The committed baseline at HEAD
`40c897c` differs and is recorded in footnote 36. Nothing in this file was written by the matrix
generator; both snapshots are stated so no reader has to guess which one a number belongs to.

Status vocabulary is restricted to `PASS`, `PASS_WITH_DECLARED_FAILURES`, `INCOMPLETE`, `FAIL`,
`NOT_RUN`, `BLOCKED`, `USER_DEFERRED`. Nothing is blank; absent evidence is `NOT_RUN`/`NONE`.

---

## 1. Identity — source, checkpoint, interpreter

| model_id | checkpoint_id | checkpoint_sha256 | checkpoint_digest_source | interpreter | interpreter_present | native_verified |
|---|---|---|---|---|---|---|
| csmcir | fashioniq | c8812071de0640d647c5ba2a70628b80a3289ee2c47676f7a1b6abaa313fa49e | AUTHOR | workbench/artifacts/environments/cir-005ef4e1b02999a5/bin/python | Y | N |
| airknow | fiq_n05 | f4d25ed9e532f2ee3671f6edadb1048e73113246ec056b413bde75dfe3f79bc6 | AUTHOR | workbench/envs/airknow-adapted/bin/python | N | N |
| conesep | fiq_n02 | 5a864ee5a78c1d249320c4f997cc7d14b529b329b2efba5577c82dbe019ed9d2 | AUTHOR | workbench/envs/ilearn39-adapted/bin/python | Y | N |
| ptha_mtst | cirr_transfer | 1c4dab86a9f66dfe10aff75b39790432eb6cc746cf27b6cfab74b66d93b7588f | AUTHOR | workbench/artifacts/environments/cir-005ef4e1b02999a5/bin/python | Y | N |
| habit | fiq_n02 | 814d2d0ceb29ef910f37562e8bea8a76cf38aa546fbc2102e495df687c276b03 | AUTHOR | workbench/envs/ilearn39-adapted/bin/python | Y | N |
| intent | fiq_n02 | 2a504cadfd6b1c2486a2c508538b0160c5653f62c7cd11f4f3d8a9534dc26afc | AUTHOR | workbench/envs/ilearn39-adapted/bin/python | Y | N |
| hint | fashioniq | bf9c72d42206b667ed586758c15c648970df821857e4067f26155a114a6517d4 | AUTHOR | workbench/envs/ilearn39-adapted/bin/python | Y | N |
| encoder | fashioniq | 0f31c50150a7bda3495e3378df13b4cec7924b52efd23b5e5dd9731459da9173 | AUTHOR | workbench/envs/encoder310-adapted/bin/python | Y | N |
| pair | pair_b1 | c4d861ecd0baa3f1427fc7a853943f5614baf87def1cdd9c4bb7583ee925fbf2 | AUTHOR | workbench/envs/pair-adapted/bin/python | N | N |
| clvc_net | author_drive_unresolved | b504245dc028d534e6e421eb73d5884d6cbda9cbb3ae03b95ab5c8b8e90ee6b2 | LOCAL_ONLY | workbench/envs/clvc39-la/bin/python | Y | N |
| dcnet | fashioniq_run_directory | a7f9ce0eb86fc1ee3c26e73ed5ac75259b588dbe305d33c12cf12f556ecaaaa4 | LOCAL_ONLY | workbench/artifacts/environments/dcnet-377/bin/python | Y | N |
| combiner_rn50x4_noft | rn50x4_noft_unresolved | b31882fe6f7e3e194cb9cfda048926844ee0ccc9d4ebc5483fb3bacfa5972955 | AUTHOR | workbench/artifacts/environments/clip4cir-38/bin/python | Y | N |
| clip4cir_rn50x4_fullft | rn50x4_fullft_unresolved | 9d13b110dcf92a92164908b65b4e846e9eba7de8ab73b0d42b413c105dcae8b4 | LOCAL_ONLY | workbench/artifacts/environments/clip4cir-38/bin/python | Y | N |
| tgcir | tgcir_archive_unresolved | c37768f374c75bebff1299e870518c23bf6db3dc8a75c4d1f729ccd6dd02e966 | AUTHOR | workbench/envs/tgcir39-la/bin/python | Y | N |
| sprc | sprc_fiq_vitl | 19386104271c01ea03ee2177575c0e55f98edd3ba61a116d78f2e61ab5dbea08 | AUTHOR | workbench/artifacts/environments/sprc-39/bin/python | Y | N |
| limn | base_iter0_dress\|shirt\|toptee | 4c6d1f562ca6c27fbb2077dd3258d2e5fc5ba324ed34edc2b42674e6dee23574 | AUTHOR | workbench/envs/limn310-adapted/bin/python | Y | N |
| tme | fashioniq_0 | fea7fc08760dad1f5a66184ea7e65b45388f9781157cbd33a3b64db74103968a | LOCAL_ONLY | workbench/envs/tme-adapted/bin/python | Y | N |
| clip4cir_rn50x4_noft | rn50x4_noft_unresolved | b31882fe6f7e3e194cb9cfda048926844ee0ccc9d4ebc5483fb3bacfa5972955 | AUTHOR | workbench/artifacts/environments/clip4cir-38/bin/python | Y | N |

Footnotes for §1
- **1** `checkpoint_digest_source=AUTHOR` means the registry carries a `expected_sha256` for that
  variant; `LOCAL_ONLY` means no author digest exists and the sha256 shown was computed on disk by
  `workbench/docs/reviews/20261109-native15-composition-limn/scripts/hash_ckpts_canonical.py`
  (output `scripts/checkpoint_hashes_canonical.json`). Per-variant matches:
  `AUTHOR` rows all matched byte-for-byte; bundles whose members are AUTHOR-hashed and matched
  individually: `clvc_net` 6/6 (`models.yaml:969-974`), `clip4cir_rn50x4_fullft` 2/2,
  `dcnet` 2/2, `limn` 3/3 (`models.yaml:1443`-region for `sprc`).
- **2** `limn` has three per-category checkpoints, each with its own AUTHOR digest
  (`base_iter0_dress 96a335d4…fc1c`, `shirt 9260c7f7…0d74`, `toptee 79f663a1…cece2`, all three
  hashed on disk to the same values); the cell shows the author hasher's directory digest for the
  set (`4c6d1f56…3574`, `kind=directory`).
- **3** `clvc_net`, `clip4cir_rn50x4_fullft`, `dcnet` show a **directory** digest
  (`scripts/checkpoint_hashes_canonical.json` `kind=directory`), because each is a bundle, not a
  file. Their per-member AUTHOR digests matched (footnote 1).
- **4** `tme` is `LOCAL_ONLY`: `models.yaml` declares `expected_sha256: null` for both variants
  (`fashioniq_0`, `cirr_0`). On-disk digest `fea7fc08…968a` (2 728 382 104 B).
- **5** `pair` row shows `pair_b1` (`c4d861ec…fbf2`, AUTHOR, matched); `pair_b2`
  (`74346fd7…3f03`, AUTHOR, matched) carries the same status. Neither variant maps to a published
  class (see §5).
- **6** `sprc` row shows `sprc_fiq_vitl` — **present**, 1 459 756 541 B,
  `sha256 19386104271c01ea03ee2177575c0e55f98edd3ba61a116d78f2e61ab5dbea08`, equal to
  `models.yaml:1452`. The *other* variant, `sprc_fiq_unresolved`, has `expected_sha256: null` and
  `status: CHECKPOINT_MAPPING_UNVERIFIED`; the paper's **ViT-g** checkpoint is what is missing
  (`models.yaml:1423-1425`), not SPRC's checkpoint.
- **7** `interpreter_present=N` for `airknow` and `pair` is a **dangling symlink**, not an absent
  build: `workbench/envs/airknow-adapted/bin/python` and `workbench/envs/pair-adapted/bin/python`
  both point at `/home/heheboiz/miniconda3/envs/teacher-sprc/bin/python`, which does not exist
  (`test -e` → false; `ls -la` shows the symlink). Working alternates exist but are **not** the
  declared interpreter: `workbench/envs/airknow310-la/bin/python` (3.10.20) and
  `workbench/envs/pair39-la/bin/python` (3.9.25).
- **8** `clvc_net` and `tgcir` declare **no interpreter** in the registry
  (`NATIVE_EVAL_MATRIX.csv` `interpreter` = `none (no interpreter declared in registry)` /
  `none (no interpreter declared; registry python UNKNOWN)`), yet both have a recorded 3-category
  native run. The env named in the cell is the one the recorded command used
  (`BLOCKED_MODELS_VERIFICATION.vi.md:215` for CLVC, `:295` region for TG-CIR), verified present and
  reporting 3.9.25.
- **9** `interpreter` values for the nine models that ran under a shared adapted env are the
  intervals the **artifacts record**, not the registry's `environment.python`: e.g. `conesep`
  registry says 3.8.10 but `cohort/measurements/conesep_fiq_n02.json`
  `runtime.python = …/envs/airknow-adapted/bin/python`; `ilearn39-adapted` is 3.9.25. Both facts are
  true and describe different things (declared env vs. executed env).
- **10** `native_verified=N` everywhere, **including `limn` and `encoder`**, follows the campaign's
  own definition: `native_verified=Y` requires a real upstream evaluator run **and** G1+G2 parity
  (`EVIDENCE_INDEX.md:139-143`). Independently recomputed with the gate's rule (G1 and G2 both at
  zero mandatory failures): **0 models** qualify. The historical `NATIVE_EVAL_MATRIX.csv` also says
  `Y` only for `limn`; that `Y` predates the gate's `relation_parity`/`metrics_parity` dimensions and
  is superseded here (§ Superseded, item 2).
- **11** `clip4cir_rn50x4_noft` (row 18) shares the `combiner_rn50x4_noft` checkpoint id and bytes:
  both are the "noft" combination of the same two upstream files; `models.yaml` registers only the
  `combiner_*` id. Its R@10 35.14068226019541 appears in `measurements/clip4cir_parsed.json` under
  key `clip4cir_noft`, and the native service row for it is `counted=false,
  artifact_backed=false` (an unbacked roll-up row), which is why it cannot be counted as a
  configuration even though the number exists.

---

## 2. Native FashionIQ result

| model_id | native_protocol | native_result | native_run_id_or_artifact |
|---|---|---|---|
| csmcir | fashioniq_original_split | R@10 57.05073873202006 / R@50 77.29859352111816 | workbench/artifacts/checkpoint_selection/20261107T021500+0700/csmcir/measurement.json |
| airknow | fashioniq_full_gallery_ref_excluded | R@10 52.54661498980954 / R@50 72.9026186457433 | workbench/artifacts/checkpoint_selection/20261106T002617+0700/measurements/airknow_fiq_n05_run1.json |
| conesep | fashioniq_full_gallery_ref_excluded | R@10 54.76130615726771 / R@50 75.21514359343843 | workbench/artifacts/checkpoint_selection/20261106T002617+0700/cohort/measurements/conesep_fiq_n02.json |
| ptha_mtst | fashioniq_original_split | R@10 37.897844115893044 / R@50 59.83638962109884 | workbench/artifacts/checkpoint_selection/20261106T233409+0700/measurements/ptha_transfer_run1.json |
| habit | fashioniq_full_gallery_ref_excluded | R@10 54.87332239571265 / R@50 74.50337396905957 | workbench/artifacts/checkpoint_selection/20261106T002617+0700/cohort/measurements/habit_fiq_n02.json |
| intent | fashioniq_full_gallery_ref_excluded | R@10 54.86624335289025 / R@50 75.43191079283865 | workbench/artifacts/checkpoint_selection/20261106T002617+0700/cohort/measurements/intent_fiq_n02.json |
| hint | fashioniq_val_split | R@10 59.80338494972219 / R@50 80.13076097500765 | workbench/artifacts/checkpoint_selection/20261106T233409+0700/measurements/hint_doc_command.json |
| encoder | fashioniq_val_split | **`native_primary`** R@10 56.22580193539884 / R@50 77.53315673870537 (seeded, reproducible — the only cross-run-reproducible ENCODER figure) · **`native_val_control`** R@10 56.17095419043685 / R@50 77.49899018875801 (`on_adapter_features == val_control` byte-equal on all 9 cells; the `native_primary` delta is the **query transform** `preprocess_train` vs `preprocess_val`, NOT the mask — both are reference-masked) · registry draw 56.223101675386054 / 77.66738539271432 `NOT_REPRODUCIBLE` (the one row the API serves) · review draw 56.290792549557 / 77.633389132303 `NOT_REPRODUCIBLE` and **served nowhere** (`raw/native/encoder.json` states no `replay_type`, so it is neither a row nor a backing report) | seeded `native_primary`: `docs/reviews/20261109-native15-composition-limn/g1_g2.encoder.{dress,shirt,toptee}.native.json` → `native.native_metrics_recheck` · `native_val_control`: `g1_g2.encoder.{…}.adapter.json` → `adapter.scoring_ranking.native_convention_recall_val_control` · registry: `artifacts/checkpoint_selection/20261106T002617+0700/cohort/measurements/encoder_fashioniq.json` · review: `docs/reviews/20261109-native15-composition-limn/raw/native/encoder.json` |
| pair | fashioniq_val_split | **WITHHELD FROM WEB PUBLICATION (2026-11-13); retained in the registry.** Its four counted exploratory rows are readable **by id only**: `61ccfc4d63fb69eb` B1/H_normal 34.03503453772374 / 56.59567753134161, `27a9b77d146b9b97` B1/H_rev 35.014736041655844 / 57.57531573029066, `f4655ef992695f90` B2/H_normal 22.156393011046543 / 43.486402136659365, `8cb65b73274a86f8` B2/H_rev 20.086769084793715 / 41.49879132733318 (all `mapping_status: EXPLORATORY_PUBLIC_MAPPING`). The **8** variant-keyed roll-up rows remain **listed** and uncounted — `B1/H_normal`, `B1/H_rev`, `B2/H_normal`, `B2/H_rev` × 2 files, each `rollup: true`, `counted: false`, `protocol_id: null`, `unit_source: inferred_from_magnitude` | `workbench/artifacts/checkpoint_selection/20261106T002617+0700/pair_compat/measurements/pair_b1_H_normal.json` (+ `pair_b1_H_rev.json`, `pair_b2_H_normal.json`, `pair_b2_H_rev.json`) · roll-ups: `pair_compat/manifests/pair_compat_verdict.json` and `pair_compat/measurements/summary.json` · deferral: `workbench/registry/models.yaml` `web_publication.status: DEFERRED` |
| clvc_net | fashioniq_val_split | R@10 30.02079703540087 / R@50 58.67189400437203 at batch 32 (SELECTED: composition runs pin 32) · R@10 29.970746517160162 / R@50 58.62171212901515 at batch 48 (lineage) | workbench/artifacts/checkpoint_selection/20261106T002617+0700/measurements/clvc_run1.json (batch 48) · workbench/docs/reviews/20261111-master-audit/raw/clvc_run3cat_gpu.json (batch 32) · per-query ranks and the reproduction proof: workbench/docs/reviews/20261112-parity-repair/CLVC_NATIVE.md §3-§6 + raw/clvc_native/<pass>/<cat>.jsonl |
| dcnet | fashioniq_full_gallery_ref_excluded | R@10 27.929730914548916 / R@50 53.76075764301528 (**CPU replay**, the served pair) · roll-up `measurements/dcnet_run1.json` states 27.929730914548923 / 53.72723331879208 — a 2026-11-07 CUDA run, **not** the served pair | served backing: `workbench/docs/reviews/20261109-native15-composition-limn/raw/native/dcnet.json` (`replay_type SOURCE_NATIVE_ENTRYPOINT`) · roll-up: `workbench/artifacts/checkpoint_selection/20261106T002617+0700/measurements/dcnet_run1.json` |
| combiner_rn50x4_noft | fashioniq_original_split | R@10 35.1078 / R@50 58.4756 | workbench/artifacts/checkpoint_selection/20261106T002617+0700/measurements/selection_decisions.json |
| clip4cir_rn50x4_fullft | **validated** `fashioniq_original_split` — the recorded label `fashioniq_full_gallery_ref_excluded` is `MISLABELED` (`PROTOCOL_MATRIX.csv`: the FashionIQ branch of `third_party/CLIP4Cir/src/validate.py:20-56` applies **no reference mask**; the only `reference_mask` is CIRR-only at `:160-166`) | R@10 43.8016931215922 / R@50 66.98358456293742 — macro = **unweighted mean** of the three category recalls (`validate.py:5` `from statistics import mean`, `:281-297`, `:308-309`; a pooled recomputation would be 43.76662307359437, a different quantity) · roll-up `measurements/clip4cir_parsed.json` states 43.81821850935618 / 66.95005893707275 over dress/toptee cells that exist in **no other file**, served only as `raw_metrics` | workbench/docs/reviews/20261109-native15-composition-limn/raw/native/clip4cir_fullft.json (`backed_by_sha256 fb97cfab…398e`; stdout log prints `average recall10 = 43.8016931215922`) |
| tgcir | fashioniq_val_split | R@10 49.720772235486656 / R@50 71.10231942031965 | workbench/artifacts/checkpoint_selection/20261106T002617+0700/measurements/tgcir_run1.json |
| sprc | fashioniq_original_split | R@10 51.038571198781334 / R@50 73.32633932431538 | workbench/artifacts/checkpoint_selection/20261106T002617+0700/measurements/sprc_vitl_run1.json |
| limn | fashioniq_val_split | R@10 55.79193333333333 / R@50 77.6378 | workbench/artifacts/checkpoint_selection/20261106T002617+0700/cohort/measurements/limn_base_iter0.json |
| tme | fashioniq_original_split | R@10 54.922901590665184 / R@50 74.83842174212138 | workbench/artifacts/checkpoint_selection/20261106T233409+0700/measurements/tme_normal_path_full.json |
| clip4cir_rn50x4_noft | fashioniq_original_split (inherited from the roll-up, not declared per row) | R@10 35.14068226019541 / R@50 58.44364364941915 | workbench/artifacts/checkpoint_selection/20261106T002617+0700/measurements/clip4cir_parsed.json |

Footnotes for §2
- **12** Each `native_run_id_or_artifact` is the **aggregate-bearing** file, not a roll-up:
  - `csmcir` → `checkpoint_selection/20261107T021500+0700/csmcir/measurement.json`
    (`macro_r10 57.05073873202006`, `macro_r50 77.29859352111816`, `protocol fashioniq_original_split`,
    checkpoint sha256 in-file = registry).
  - `airknow` → `measurements/airknow_fiq_n05_run1.json` (`aggregate.complete=true`). The `fiq_n08`
    variant is 50.64166565555885 / 71.45559570455528 and is **not** the registered selection.
  - `conesep` → `cohort/measurements/conesep_fiq_n02.json`.
  - `ptha_mtst` → `20261106T233409+0700/measurements/ptha_transfer_run1.json`
    (`macro_metrics`), labelled `CIRR checkpoint -> FashionIQ | cross-dataset transfer |
    exploratory` — a transfer, not a FashionIQ reproduction (`EVIDENCE_INDEX.md:108-111`).
  - `habit` → `cohort/measurements/habit_fiq_n02.json`; `intent` → `…/intent_fiq_n02.json`.
  - `hint` → `20261106T233409+0700/measurements/hint_doc_command.json` (the later, counted copy;
    `cohort/measurements/hint_fashioniq.json` states the identical numbers and is marked duplicate).
  - `encoder` → **three** aggregates exist, and only one is cross-run reproducible. **Two of the
    three are the same kind of quantity in two SCOPES, and they must never be collapsed into one
    number** (`ENCODER_RELATION.md` §11b):
    **`native_primary`** = `56.22580193539884 / 77.53315673870537`, the mean over
    `g1_g2.encoder.{dress,shirt,toptee}.native.json` → `native.native_metrics_recheck`; queries go
    through `preprocess_train` (stochastic `RandomResizedCrop`), **reference-masked**.
    **`native_val_control`** = `56.17095419043685 / 77.49899018875801`, the mean over
    `g1_g2.encoder.{…}.adapter.json` → `adapter.scoring_ranking.native_convention_recall_val_control`;
    queries go through `preprocess_val`, **also reference-masked**. `on_adapter_features` is
    **byte-equal to `val_control` on all 9 cells** (r1/r10/r50 × 3 categories, asserted this turn).
    The `native_primary ↔ val_control` gap is therefore the **query transform, not the mask** — and
    this is why neither may be quoted as "the" ENCODER native number.
    The two *unseeded draws* are
    `cohort/measurements/encoder_fashioniq.json` (`recorded_at 2026-11-06T14:39:45Z`, 56.223101675386054
    / 77.66738539271432) and `raw/native/encoder.json` (`recorded_at 2026-11-09T13:23:23Z`,
    56.290792549557 / 77.633389132303) are **both unseeded draws** — `learnable_relation` is drawn
    from the global RNG and neither file records a `relation_state`, `seed`, or `batch_size`
    (`ENCODER_RELATION.md` §9b, key-set table). `RELATION_SEED` landed 2026-11-11, after both. Only the
    **seeded parity native side** reproduces across runs: R@10 **56.225801935399** / R@50
    **77.533156738705**, recomputed this session as the mean of
    `g1_g2.encoder.{dress,shirt,toptee}.native.json` → `native.layer1_identity.native_metrics_recheck`
    (`dress 51.71046108081309 / shirt 54.90677134445535 / toptee 62.0601733809281`; r50
    `76.30143777887952 / 75.3189401373896 / 80.97909229984701`). Both older numbers are marked
    `NOT_REPRODUCIBLE` in the cell, not deleted.
  - `pair` → `pair_compat/measurements/pair_b1_H_normal.json`; the four counted rows are
    B1/H_normal 34.03503453772374, B1/H_rev 35.014736041655844, B2/H_normal 22.156393011046543,
    B2/H_rev 20.086769084793715 — all `mapping_status: EXPLORATORY_PUBLIC_MAPPING`. **Deferred from
    web publication 2026-11-13** (§4b): these four ids are reachable only at
    `/api/native-measurements/{measurement_id}`. The **8** variant-keyed roll-up rows
    (`B1/H_normal`, `B1/H_rev`, `B2/H_normal`, `B2/H_rev`, two files each) remain **listed** and stay
    `counted: false`, `protocol_id: null`, `unit_source: inferred_from_magnitude` — they are keyed by
    variant, not by `model_id: pair`, so the deferral filter does not touch them and must not be
    described as removing every `B1/`/`B2/` row.
  - `clvc_net` → **both** aggregates are correct and are labelled by batch size:
    `measurements/clvc_run1.json` = **batch 48** (`29.970746517160162 / 58.62171212901515`, all nine
    cells reproduced), `raw/clvc_run3cat_gpu.json` = **batch 32** (`30.02079703540087 /
    58.67189400437203`, all six cells reproduced). Proof, per-query rank JSONL and the determinism
    control: `20261112-parity-repair/CLVC_NATIVE.md` §3-§6; raw passes under
    `20261112-parity-repair/raw/clvc_native/<pass>/<cat>.jsonl` (2017/2038/1961 rows per category,
    all fields complete, `target_rank` never null). The **batch-32** figure is the one to quote
    against any composition run, because `workbench/replay/clvc_net.py:268` defaults
    `--batch-size 32` and every recorded composition measurement used 32.
  - `dcnet` → `measurements/dcnet_run1.json`; the artifact carries `paper` cells
    (`macro_r10 27.78`) so it is comparable in shape, but no parity exists.
  - `combiner_rn50x4_noft` → `measurements/selection_decisions.json`
    `selected_2026-11-06.combiner_rn50x4_noft.metrics` (35.1078 / 58.4756). The **run artifact** with
    the same numbers is
    `workbench/artifacts/reports/combiner_rn50x4_noft_…020a70b0_aggregate.json`
    (`EXECUTION_COMPLETED`, `observed_metrics {r10 35.1078, r50 58.4756, mean 46.7917}`).
  - `clip4cir_rn50x4_fullft` → `20261109-native15-composition-limn/raw/native/clip4cir_fullft.json`
    (`aggregate.macro_r10 43.8016931215922`, `macro_r50 66.98358456293742`,
    `replay_type SOURCE_NATIVE_ENTRYPOINT`). The roll-up file
    `measurements/clip4cir_parsed.json` states a **different** pair
    (`43.81821850935618 / 66.95005893707275`) for its own `clip4cir_fullft` key, and
    `selection_decisions.json` repeats `43.8182`. Two different runs; **the replay is the correct
    aggregate** because the macro rule is the unweighted three-category mean
    (`validate.py:5,281-309`) and the roll-up's dress/toptee cells exist in no other file —
    footnote 33. The roll-up's pair is served only as `raw_metrics`.
    `docs/reviews/20261109-native15-composition-limn/raw/native/encoder.json` is the contrasting
    case: it states an `aggregate` but **no `replay_type`**, so it is neither a row nor a backing
    report and its `56.290792549557 / 77.633389132303` is served **nowhere**.
  - `tgcir` → `measurements/tgcir_run1.json`; `raw/tgcir_run3cat_gpu.json` states the identical
    numbers (`49.720772235486656`) and is marked duplicate/uncounted.
  - `sprc` → `measurements/sprc_vitl_run1.json` (`aggregate.complete=true`).
  - `limn` → `cohort/measurements/limn_base_iter0.json`; note its `note` field states the
    per-category rows came from the completed run and **the aggregate was recomputed here** after
    the runner's own aggregate helper raised.
  - `tme` → `20261106T233409+0700/measurements/tme_normal_path_full.json`.
- **13** `native_protocol` for `combiner_rn50x4_noft` is `fashioniq_original_split` per the row, and
  the run artifact `…020a70b0_completion.json` states `protocol_id: fashioniq_original_split` too —
  consistent. `clip4cir_rn50x4_fullft` is **`fashioniq_original_split` at the validated protocol**;
  its recorded label `fashioniq_full_gallery_ref_excluded` (in `raw/native/clip4cir_fullft.json:5`) is
  `MISLABELED`. Verified in source, not taken from the matrix: the FashionIQ branch
  (`workbench/third_party/CLIP4Cir/src/validate.py:20-56`, `compute_fiq_val_metrics`) sorts the index,
  builds `labels` from `sorted_index_names` and computes `recall_at10/50` — **no reference masking
  anywhere**; `generate_fiq_val_predictions` (`:58-112`) consumes the reference features but never
  removes the reference row. The file's **only** `reference_mask` is at `:160-166` inside
  `compute_cirr_val_metrics` (CIRR). The registry agrees (`native_protocol:
  fashioniq_original_split` on `/api/models`). The metrics are unaffected by the label; the cohort
  (3817/6346/5373) is the same either way.
- **14** No native artifact records a `run_id` field except the two `workbench/artifacts/reports/`
  runs used for `combiner`; the others are identified by file path + `recorded_at` (present only for
  `encoder 2026-11-06T14:39:45Z` and `tme 2026-11-06T18:53:49Z`). That is why the column is named
  `native_run_id_or_artifact`.
- **15** `pair`'s R@50 shown is B1/H_normal's `56.59567753134161`; the reason the R@10 cell lists all
  four rows is that "one number" would misrepresent a model whose two checkpoints × two hypotheses
  disagree by up to 18.6 pp (52.76 paper vs 34.04 best public mapping).

---

## 3. Composition + export + benchmark

| model_id | composition_run_ids | composition_status | topk_available | benchmark_status |
|---|---|---|---|---|
| csmcir | NONE | NOT_RUN | N/A | NOT_RUN |
| airknow | NONE | NOT_RUN | N/A | NOT_RUN |
| conesep | 30f6de423861cc1d6c78c361; a4f786d8af4d46ac2a85ab9c | PASS | Y | NOT_RUN |
| ptha_mtst | 315dd590eeb03d2c78d7937a; f54a77450ebd2126590d4385 | PASS | Y | NOT_RUN |
| habit | 497c25c57fbeba885ecfb9d1; c6eb9d5775ae636a3b8addbc; e0dd04a46ef7a58334b9e353 | INCOMPLETE | Y | NOT_RUN |
| intent | 37ee72aa298b276d1bfda0fa; 4901da139ac40b874d2363c5 | PASS | Y | NOT_RUN |
| hint | 4bc9893a977e501b82378e76; f461f391c6d4696011be8067 | PASS | Y | NOT_RUN |
| encoder | 5dee7425dd46c5f154808aac; 6a528e877c328772e2c7bb28; 9b806c22221e944ff10e514e; acc9b20e1aea48487f65f944; ce89a9a27c4cceadeaaf30b0; eda30a37c0bc17ac43844f09 | PASS | Y | NOT_RUN |
| pair | NONE | NOT_RUN | N/A | NOT_RUN |
| clvc_net | 889bc330d60e8cbab956944f; d541e40cde1a949026f98854; da0a2162be0e159a1a456cec; eab10106b2df8e97fb7a5ce5 | INCOMPLETE | PARTIAL | NOT_RUN |
| dcnet | NONE | NOT_RUN | N/A | NOT_RUN |
| combiner_rn50x4_noft | NONE | NOT_RUN | N/A | INCOMPLETE |
| clip4cir_rn50x4_fullft | NONE | NOT_RUN | N/A | NOT_RUN |
| tgcir | 1eee4c0adbc93f7708205a5e; 6396454ce29c2bea9ebfbe00; 7b1fa85007bffdab931f7f19; 9f115afeb58eb1adf75d88ba | INCOMPLETE | Y | NOT_RUN |
| sprc | NONE | NOT_RUN | N/A | NOT_RUN |
| limn | 10173dc45dea8c90fd2a284b; 3f86f58483b368377e2c151e; 9e4918a6b5c243412cbc58c9 | INCOMPLETE | PARTIAL | NOT_RUN |
| tme | ad477312aa57c1b9114f9e51; cd1a696d8bca0fc4bce6925b | PASS | Y | NOT_RUN |
| clip4cir_rn50x4_noft | NONE | NOT_RUN | N/A | NOT_RUN |

Footnotes for §3
- **16** `composition_run_ids` are the `run_id`s of the `full.json` summaries under
  `workbench/artifacts/composition/summary/`. The two populations must not be mixed:
  **33 distinct ids on disk** (36 dirs; 2 `SYNTHETIC_*`, 1 `released_audit_task7_mm_any`, and 5
  query-limited smokes) versus **26 distinct ids served by default** by
  `GET /api/composition/runs` (smoke, fixtures and imported rows are opt-in). The six smoke ids are
  `3f86f584` (limn, 8 q), `eda30a37` (encoder, 8 q), `1eee4c0a` + `6396454c` (tgcir, 5 q),
  `889bc330` + `d541e40c` (clvc_net, 5 q) — all `smoke.limited: true`. Per-model default-served
  counts are `encoder 5, habit 3, limn 4, tgcir 2, clvc_net 2, conesep 2, hint 2, intent 2,
  ptha_mtst 2, tme 2`; measured this turn. See §Totals.
- **17** `composition_status`: `PASS` only when **every** referenced attempt is `COMPLETED`;
  `INCOMPLETE` when some attempt completed and at least one was `PARTIAL`/`FAILED`; `FAIL` when none
  completed. Exact per-attempt states:
  `encoder` 6×COMPLETED; `conesep` 2×COMPLETED; `intent` 2×COMPLETED; `hint` 2×COMPLETED;
  `tme` 2×COMPLETED; `ptha_mtst` 2×COMPLETED; `habit` 2×COMPLETED + 1 FAILED (`e0dd04a4`,
  interpreter mismatch); `tgcir` 3×COMPLETED + 2×PARTIAL (`9f115afe` and its `-retry-1`);
  `limn` 1×COMPLETED (8-query smoke `3f86f584`) + 2×PARTIAL (`10173dc4`, `9e4918a6`);
  `clvc_net` **6 dirs / 4 ids, every one PARTIAL** (`889bc330`, `d541e40c` are 5-query smokes;
  `da0a2162` and `eab10106b2`+2 retries are 6003).
- **18** `topk_available`: `Y` = a 6003-cohort attempt recorded `TOPK_EXPORT_COMPLETE`;
  `PARTIAL` = only `TOPK_EXPORT_COMPLETE_PARTIAL` (CLVC `da0a2162` missing 6 = 2 qids × 3
  conditions; LIMN `9e4918a6` missing 6); `N` = runs exist but no 6003 attempt exported Top-K
  (`encoder`'s `6a528e87` has `topk_export = null` yet other attempts are complete);
  `N/A` = no composition run at all.
- **19** `benchmark_status` is the paper-gate state, **not** the composition state. Only
  `workbench/artifacts/reports/*_aggregate.json` carries it — **19 aggregate files**, counted this
  session by `model_id`: `combiner_rn50x4_noft` ×2, `csmcir` ×5, `dcnet` ×1, `mock` ×1,
  `synthetic_model` ×10 (`synthetic_model` and `mock` are fixtures, never configurations). `combiner_rn50x4_noft` = `INCOMPLETE`
  because its completed run `020a70b0` is `EXECUTION_COMPLETED` with
  `benchmark_gate.status = METRIC_REFERENCE_UNVERIFIED`, `passed: false` — the run reproduced a
  number but no committed reference describes the experiment. `csmcir`'s two
  `PASS_WITHIN_5_PERCENT_VARIANT` runs (footnote 20) and `dcnet`'s one `EXECUTION_FAILED` run are the
  only other real-model reports; every other model has no aggregate report at all, so `NOT_RUN`
  (not "passed").
- **20** `csmcir` does have two `PASS_WITHIN_5_PERCENT_VARIANT` benchmark runs (`f564570e`,
  `f03887d2`, `observed_metrics 57.0507/77.2986/67.1747`). They are **not** reflected in
  `benchmark_status` as `PASS` because the row is `USER_DEFERRED`: the campaign is not spending
  composition work on it. The benchmark fact is recorded here so the deferral is not mistaken for
  absence.

---

## 4. Parity — gate-authoritative

| model_id | g1_status | g2_status | g1_failing_checks | g2_failing_checks |
|---|---|---|---|---|
| csmcir | NOT_RUN | NOT_RUN | NONE | NONE |
| airknow | NOT_RUN | NOT_RUN | NONE | NONE |
| conesep | NOT_RUN | NOT_RUN | NONE | NONE |
| ptha_mtst | NOT_RUN | NOT_RUN | NONE | NONE |
| habit | NOT_RUN | NOT_RUN | NONE | NONE |
| intent | NOT_RUN | NOT_RUN | NONE | NONE |
| hint | PASS | FAIL | NONE | shirt.batch_1_vs_n.max_rel; toptee.batch_1_vs_n.max_rel |
| encoder | PASS | FAIL | NONE | dress.population.features_vs_native_val_control.max_rel; dress.population.gallery_vs_native.max_rel; dress.batch_1_vs_n.max_rel; shirt.population.features_vs_native_val_control.max_rel; shirt.population.gallery_vs_native.max_rel; shirt.batch_1_vs_n.max_rel; toptee.population.features_vs_native_val_control.max_rel; toptee.population.gallery_vs_native.max_rel; toptee.batch_1_vs_n.max_rel |
| pair | NOT_RUN | NOT_RUN | NONE | NONE |
| clvc_net | NOT_RUN | NOT_RUN | NONE | NONE |
| dcnet | NOT_RUN | NOT_RUN | NONE | NONE |
| combiner_rn50x4_noft | NOT_RUN | NOT_RUN | NONE | NONE |
| clip4cir_rn50x4_fullft | NOT_RUN | NOT_RUN | NONE | NONE |
| tgcir | NOT_RUN | NOT_RUN | NONE | NONE |
| sprc | NOT_RUN | NOT_RUN | NONE | NONE |
| limn | PASS | FAIL | NONE | dress.batch_1_vs_n.max_rel; shirt.batch_1_vs_n.max_rel; toptee.batch_1_vs_n.max_rel |
| tme | NOT_RUN | NOT_RUN | NONE | NONE |
| clip4cir_rn50x4_noft | NOT_RUN | NOT_RUN | NONE | NONE |

Footnotes for §4
- **21** Source of every G1/G2 cell: `from workbench.backend.parity_gate import evaluate_model`,
  iterating `_discover_models(evidence_root())`, which at handoff returns
  `['conesep','encoder','habit','hint','intent','limn','tgcir']` — **7 models, discovered from their
  own filenames, not from a seed list** (re-measured 2026-11-13: 5 carry a verdict — encoder, habit,
  hint, limn, tgcir — and `conesep`/`intent` sit at `PARITY_EVIDENCE_MISSING`). The figure "three
  models" recorded here earlier was the state when HINT was the newest producer and HABIT/TG-CIR had
  no pair yet. `evaluate_all()` returns `passed_models: []` and a mixed `standard` list
  (`['parity-absolute-relative/v2', 'parity-relative/v1']`), because each pair is judged under the
  standard it declares; the per-model `applied_standard` is recorded in the `parity_evidence_path`
  cell.
- **22** `encoder` **G1 now PASSES** (30 mandatory checks, 0 failing). The three
  `relation.{dress,shirt,toptee}.digest` checks are recorded on both sides with the pinned digest:
  `relation_state = {status: recorded, seed: 20261111, sha256: 20db0c27c979e9a0938c71af5142458dfa2
  070468d0ad3fcb6f24134c2f9ce92, expected_sha256: <same>, matches_pinned_digest: true}` in each
  `g1_g2.encoder.*.native.json` and `…adapter.json`. This is the working-tree state produced by
  `scripts/encoder_parity.py`, which now imports `RELATION_SEED` from the production module and calls
  it immediately before `Encoder(...)` (`git diff` of that file, hunk at the `native_phase`
  construction). At HEAD these three checks were `not_recorded` — footnote 36.
- **23** `encoder` **G2 failures are now 9** (was 13 at HEAD; the two `ranks.*` failures per
  category are gone because the re-measurement produced `rank_differences_vs_native_val_control =
  {count: 0, non_tie_count: 0}` in all three categories). Remaining, per category:
  `population.features_vs_native_val_control.max_rel` = 0.7499999820540181 / 4.236841887544606 /
  6.2777780084400545 (declared 1e-4; `max_abs` 1.125037670135498e-06 / 1.296401023864746e-06 /
  1.0021030902862549e-06, i.e. **under** the 1e-5 atol),
  `population.gallery_vs_native.max_rel` = 10.461538814286977 / 0.5312498017651329 /
  3.133333651667554 (`max_abs` 1.1399388313293457e-06 / 1.650303602218628e-06 /
  1.0132789611816406e-06), `batch_1_vs_n.max_rel` = 0.003176237364982574 / 0.0027347281757461243 /
  0.0015533629650869047 (declared 1e-3; `max_abs` 3.3527612686157227e-07 / 2.8312206268310547e-07 /
  3.203749656677246e-07). Every one is a **relative-term-only** failure: the absolute deltas are
  1–2 orders of magnitude below their atol and `min_cosine ≥ 0.99999999998` in all nine blocks.
- **24** `limn` G1 has **zero** mandatory failures (18 mandatory checks, `input_parity` only); its
  three `relation.*.digest` checks are non-mandatory (`absence=True`) because
  `limn ∉ RELATION_DRAWING_MODELS` and neither LIMN artifact carries the key. G1 is `PASS` while the
  overall status is `PARITY_MANDATORY_SUBCHECK_FAILED` — the two are different claims.
- **25** `limn` G2 failures (3): `{dress,shirt,toptee}.batch_1_vs_n.max_rel` =
  0.011399774792037631 / 0.002698280178837889 / 0.007509343787328878 vs declared 1e-3 (`max_abs`
  1.751e-07 / 1.974e-07 / 1.714e-07, `min_cosine` 0.99999999999919 / …894 / …934). Only the
  relative term fails.
- **26** `limn` `metrics_parity` is now **measured**, not missing: the three
  `g1_g2.{dress,shirt,toptee}.native.json` files carry
  `native_metrics_match_feature_rerank: true` plus a `native_metrics_parity_detail` block
  (`declared_rule: "exact integer hit-count equality at every cutoff"`, `cutoffs [1,10,50]`,
  `reference_mask: native test.py:92 convention applied on BOTH sides`, `ranking_convention`). At
  HEAD the key was absent in all six LIMN files and the dimension reported `missing` — footnote 36.
  The producer change that adds it is `scripts/t4_parity.py`'s new `refresh_metrics_block()`, which
  re-derives the block from `20261112-parity-repair/scripts/limn_hits.py` after each rewrite.
- **27** The six LIMN artifacts' identity fields (`model_id`, `checkpoint_id`, `protocol_id`,
  `protocol`, `query_order_sha256`, `gallery_order_sha256`) are **all null**; input parity rests on
  `layer1_identity`'s `query_count_native`/`query_count_annotations` pairs
  (2017/2038/1961) and the four `*_identical` booleans. This is a real weakness of the pair, not a
  gate defect — recorded, not scored.
- **28** Fifteen configurations have `NOT_RUN` because **no paired artifact exists**:
  `_discover_models` returns exactly `['encoder','limn']`. The number 15 = 17 − 2 and is
  corroborated by `INVENTORY.csv` (`has_parity_artifact` `Y` for encoder, limn only).

---

## 4b. Publication scope — registry 17, published 16, PAIR deferred

**Two different counts, and neither is the other.**

* **Registry configurations: 17.** `yaml.safe_load(models.yaml)['models']` → 17. PAIR is still one of
  them, with its source history, both checkpoint variants (`pair_b1`, `pair_b2`), its offline CLI and
  every artifact under `artifacts/checkpoint_selection/…/pair_compat/`.
* **Published on the web: 16.** `workbench/backend/registry.py:160-170` `web_deferred_models()` reads
  the registry's own flag; `/api/models`, `/api/native-measurements` and `POST /api/jobs` subtract it.
  `models.yaml:828` (working tree) declares `web_publication: {status: DEFERRED, reason: …,
  decided: 2026-11-13, offline: …}` for `pair` and for no other model.
  `workbench/tests/test_pair_web_deferral.py` asserts both numbers in one test:
  `len(registry["models"]) == 17` and `17 - len(deferred) == 16`.

**What the deferral removes, and what it does not.** Removed: PAIR's **listing** in `/api/models` and
`/api/native-measurements`, and a false `404 model_not_found` for a new PAIR job. Kept: the registry
entry, the offline CLI, and every measurement **by its own id** —
`/api/native-measurements/{measurement_id}` is deliberately not filtered, so an auditor holding
`61ccfc4d63fb69eb` still reads `mapping_status: EXPLORATORY_PUBLIC_MAPPING` and the artifact path.

**The 8 roll-up rows survive the deferral.** `B1/H_normal`, `B1/H_rev`, `B2/H_normal`, `B2/H_rev` ×
two files (`pair_compat/manifests/pair_compat_verdict.json`, `pair_compat/measurements/summary.json`)
are keyed by **variant**, not by `model_id: pair`; they carry `rollup: true`, `counted: false`,
`protocol_id: null`, `unit_source: inferred_from_magnitude`, and they were never counted. Filtering
`pair` leaves all 8 in the listing (measured: filtered census still returns 22 roll-up rows, of which
these 8). This is the correct outcome — they state a number no artifact backs and were never a
publication claim — but it means **"PAIR is deferred" is not the same as "no `B1/`/`B2/` row is
listed"**, and no count of published *models* may be read as a count of *rows*.

Measured effect of the filter on the native catalogue (disk code, `list_native_measurements` with
`exclude_model_ids={'pair'}`): 56 rows → **51**; `files 39 → 34`; `blocks_with_metrics 50 → 46`;
`with_metrics 23 → 19`; `models_with_metrics 17 → 16`; `signposted_blockers 4 → 3`;
`rollup_rows`, `duplicate_rows`, `unbacked_rows`, `verified` unchanged (18 / 27 / 18 / 0).

---

## 5. Disposition

| model_id | disposition | blocker_code | blocker_one_line |
|---|---|---|---|
| csmcir | USER_DEFERRED | USER_DEFERRED_COMPOSITION | operator deferred; native aggregate exists but no parity and no CIRCUS run |
| airknow | INCOMPLETE | INTERPRETER_BROKEN_SYMLINK | native_rows_measured, parity absent; declared interpreter is a dangling symlink |
| conesep | INCOMPLETE | NO_PARITY_EVIDENCE | native + 2 CIRCUS runs, no G1/G2 pair |
| ptha_mtst | INCOMPLETE | TRANSFER_ONLY | CIRR-checkpoint transfer, not a FashionIQ reproduction; no parity |
| habit | FAIL | PARITY_MANDATORY_SUBCHECK_FAILED | complete 3-category pair since round F1; **2** failures, both `batch_1_vs_n.max_rel` (shirt 0.0050964738236841, toptee 0.006168472315303178); `max_abs` 1.94e-07 / 1.63e-07 inside atol 1e-4. Its three `ranks.evidence_present` checks were **closed** by recording the adapter-vs-argsort comparison (5 → 2). `metrics_parity` **missing** (no stored native per-query-rank control) |
| intent | INCOMPLETE | NO_PARITY_EVIDENCE | native + 2 CIRCUS runs, no G1/G2 pair |
| hint | FAIL | PARITY_MANDATORY_SUBCHECK_FAILED | v2 pair on both sides; **2** failures, both `batch_1_vs_n.max_rel` (shirt 0.005049934976857354, toptee 0.04213516386699195); `metrics_parity` **measured true** (12 checks, 4 per category, all ok) |
| encoder | FAIL | PARITY_MANDATORY_SUBCHECK_FAILED | gate: G1 PASS, G2 9 relative-term-only failures (3 population + 3 gallery + 3 batch); relation pin now recorded on both sides |
| pair | BLOCKED | BLOCKED_MISSING_SOURCE_CLASSES | model_vssp / MHTransformer_dis unpublished; numbers exist only under an exploratory alias |
| clvc_net | INCOMPLETE | NATIVE_MACRO_RESOLVED_BY_BATCH | both historical macros reproduce exactly at fixed batch sizes (32 -> 30.0208, 48 -> 29.9707); batch size is the mechanism; --batch-size is part of the measurement identity |
| dcnet | INCOMPLETE | NO_PARITY_EVIDENCE | native aggregate present but its only ledger entry is EXECUTION_FAILED |
| combiner_rn50x4_noft | INCOMPLETE | METRIC_REFERENCE_UNVERIFIED | run 020a70b0 EXECUTION_COMPLETED, benchmark gate not passed |
| clip4cir_rn50x4_fullft | INCOMPLETE | TWO_NATIVE_MACROS | replay 43.8017 vs roll-up 43.8182; no parity |
| tgcir | FAIL | PARITY_MANDATORY_SUBCHECK_FAILED | **INCOMPLETE-class, sample-scope** — 3 failures, all `batch_1_vs_n.max_rel` (dress 0.24029609477008945, shirt 0.07048293904534349, toptee 0.029507544996198128), each denominator-driven (`max_abs` ≤ 2.90e-06 vs atol 1e-4, `min_cosine` ≥ 0.9999999997); `population_scope: sample` → `coverage: chunked` preview over 5 queries / 20 images; `metrics_parity` **missing**. **Do not quote beside encoder's 9 as a same-class complete-population failure.** See `20261113-final-acceptance/raw/parity-coverage/TGCIR_BATCH.md` |
| sprc | USER_DEFERRED | USER_DEFERRED_NO_VITG | ViT-L measured 51.04/73.33; the paper's ViT-g checkpoint is absent |
| limn | FAIL | PARITY_MANDATORY_SUBCHECK_FAILED | gate: 3 G2 batch max_rel failures; metrics_parity now measured true |
| tme | INCOMPLETE | NO_PARITY_EVIDENCE | native + 2 CIRCUS runs, no G1/G2 pair |
| clip4cir_rn50x4_noft | INCOMPLETE | ROLLUP_ONLY | macro exists only in a roll-up file; row is uncounted and unbacked |

Footnotes for §5
- **29** `FAIL` is used only where an artifact pair exists **and** the gate rejects it:
  `encoder`, `limn`. A configuration with no artifact is `INCOMPLETE`/`NOT_RUN`, never `FAIL`.
  Both current `FAIL`s are **relative-term-only**: every failing check's `max_abs` sits below its
  declared atol and `min_cosine ≥ 0.9999999999`, so the verdict is about a vanishing denominator,
  not about the models disagreeing in absolute terms. `blocker_one_line` states the split per model
  (encoder: G1 PASS, G2 9; limn: 3).
- **30** `BLOCKED` is `pair` only, and the code is the **replay** verdict
  (`workbench/docs/reviews/20261111-master-audit/raw/pair_absence_evidence.json:3`, produced by
  `workbench/replay/pair_absence.py:220`, asserted by
  `workbench/tests/test_blocked_models_verification.py:158`). The **registry** field for the same
  model is `reproduction_status: BLOCKED_CHECKPOINT_MAPPING_UNVERIFIED` (`models.yaml:825`) +
  `command_status: COMMAND_BLOCKED_SOURCE_ENTRYPOINT` (`:828`). Both are true; they are different
  vocabularies and the cell shows the precise one.
- **31** `USER_DEFERRED` is `csmcir` and `sprc`, per `EVIDENCE_INDEX.md:17-18` and `GATES.csv`
  row `Scope/accounting`. Their reasons differ: `csmcir` = composition deferred by the operator;
  `sprc` = no ViT-g checkpoint (ViT-L measured but not comparable to the paper row,
  `models.yaml:1423-1425`).
- **32** `clvc_net` `blocker_code = NATIVE_MACRO_RESOLVED_BY_BATCH` (was `NATIVE_MACRO_CONFLICT`).
  A controlled rerun settled it, both directions, bit-exactly (`CLVC_NATIVE.md` §3, §5, §6):
  - fixed **batch 32** reproduces `raw/clvc_run3cat_gpu.json` on **all six** R@10/R@50 cells
    (macro `30.02079703540087 / 58.67189400437203`, Δ `0.0`), and the prose command that recorded
    `--batch-size 32` is `BLOCKED_MODELS_VERIFICATION.vi.md:214-218`;
  - fixed **batch 48** reproduces `measurements/clvc_run1.json` on **all nine** R@1/R@10/R@50 cells
    (macro `29.970746517160162 / 58.62171212901515`, Δ `0.0`), matching the batch-`48` command at
    `CHECKPOINT_SELECTION_REPORT.md:1697-1699`;
  - three independent batch-32 passes (`batch32`, `batch32_r2`, `batch32_r3`) are **bit-identical**:
    0 ranks moved, max |Δ| `0.000e+00`, same qid order (`CLVC_NATIVE.md` §5);
  - per-query ranks now exist (`CLVC_NATIVE.md` §6), so the old "cannot be named" caveat is dead.
  **Measured sensitivity — the reason the rule exists** (`CLVC_NATIVE.md` §4): batch 1 vs 32 moves
  **1548 of 2017** dress ranks (max |Δ score| 0.644522, R@10 28.805156 → 29.449678); batch 48 vs 32
  moves **427 / 378 / 372** ranks in dress/shirt/toptee (one category at batch 1 only, stated rather
  than extrapolated).
  **Rule that replaces "quote neither":** quote **batch 32** for anything compared against a
  composition run, and **always name the batch** — `--batch-size` is part of the measurement identity.
- **33** `clip4cir_rn50x4_fullft` `blocker_code = TWO_NATIVE_MACROS`: replay `43.8016931215922`
  vs roll-up `43.81821850935618`. **Which one is correct is settled, not open:** the replay is. The
  rule is in the evaluator — `third_party/CLIP4Cir/src/validate.py:5` `from statistics import mean`,
  `:281-297` collects one value per category, `:308-309` prints `average recall10/50 =
  mean(list)` — so a macro is the **unweighted mean of the three category R@10 values**, not a pooled
  recomputation (pooled over 2017/2038/1961 queries the same cells give `43.76662307359437`, a
  different quantity). The replay's three cells recompute to exactly `43.8016931215922`, and its
  `raw/native/clip4cir_fullft.stdout.log` prints the same figure. The roll-up
  (`checkpoint_selection/20261106T002617+0700/measurements/clip4cir_parsed.json`, key
  `clip4cir_fullft`) is the **wrong** one: its `dress_r10 39.11750018596649` and
  `toptee_r50 71.23916149139404` appear in **no other file** under `workbench/` (grep → 1 hit, the
  roll-up itself), so its macro is built over cells no artifact backs. The API therefore serves the
  replay and keeps the roll-up's pair in `raw_metrics` — `backend/native_measurements.py` documents
  this as two different runs.
- **34** `airknow` `blocker_code = INTERPRETER_BROKEN_SYMLINK` is the smallest actionable defect:
  every other blocker on this board is missing science, this one is a dangling symlink
  (`workbench/envs/airknow-adapted/bin/python -> /home/heheboiz/miniconda3/envs/teacher-sprc/bin/python`,
  absent).

---

## Totals

Counted independently from the table, each from the command shown in §"Commands used". No count is
derived from another.

| quantity | value | names / detail |
|---|---|---|
| registered configurations | **17** | `yaml.safe_load(models.yaml)['models']`; ids at `models.yaml:28,133,208,315,430,527,621,696,795,876,989,1085,1168,1265,1358,1475,1582` |
| **configurations published on the web** | **16** | 17 − 1 deferred (`pair`); `web_deferred_models()` in `backend/registry.py:160-170`, asserted by `tests/test_pair_web_deferral.py` (`17 - len(deferred) == 16`). **Never quote 16 as the registry size or 17 as the published count** — §4b |
| extra in-scope native targets (not registry ids) | **1** | `clip4cir_rn50x4_noft` — aggregate present, row uncounted/unbacked |
| matrix rows | **18** | 17 + 1 |
| ids carrying a native aggregate artifact (any id) | **18** | Superseded by the line below; kept so the old figure is not silently reused |
| distinct model keys in the served native census | **25** | `airknow, airknow_n05, airknow_n08, airknow_variant_80, clip4cir_rn50x4_fullft, clip4cir_rn50x4_noft, clvc_net, combiner_rn50x4_noft, conesep, csmcir, dcnet, encoder, habit, hint, intent, limn, pair, ptha_mtst, sprc, tgcir, tme, B1/H_normal, B1/H_rev, B2/H_normal, B2/H_rev` — 21 model ids + 4 variant-keyed roll-up keys |
| **models with a counted, artifact-backed native row** | **17** | the registry's 17 ids, one each (`pair` contributes 4 counted rows; `airknow` 3) |
| counted artifact-backed native **rows** | **23** | `counts.with_metrics`; ≠ 17 models and ≠ 25 keys |
| ids carrying a native aggregate (registered only) | **17** | every registered id has at least one |
| registered ids with **no** aggregate artifact | **0** | — |
| **counts no reader may quote as "evaluated"** | — | `verified: 0` is the only verification count. **23** is counted artifact-backed *rows*, **27** is `duplicate_rows` (repeats of one identity), **18** is `unbacked_rows` (roll-up claims no artifact backs), **17** is *models* with a number, **25** is *model keys* incl. variants, **14** is protocol-eligible configurations. None of them is "evaluated", "verified", or "reproduced" — see the two separate questions below |
| **valid native FashionIQ evaluations** (protocol-eligible aggregate; parity-independent) | **14** | airknow, clip4cir_rn50x4_fullft\*, clvc_net, conesep, csmcir†, dcnet, encoder, habit, hint, intent, limn, sprc†, tgcir, tme — exclusions and reasons in footnote 37 |
| **native parity-verified** (G1+G2, zero mandatory failures) | **0** | `passed_models: []`; the 19 failures are encoder 9, habit 2, hint 2, limn 3, tgcir 3 (re-measured at handoff) — every one a relative-or-componentwise term on a vanishing denominator (footnotes 29, 38, 39) |
| native executed **but not verified** | **16** | airknow, clip4cir_rn50x4_fullft, clvc_net, combiner_rn50x4_noft, conesep, csmcir, dcnet, encoder, habit, hint, intent, limn, ptha_mtst, sprc, tgcir, tme |
| …of which `NATIVE_EVAL_MATRIX.csv` records `native_executed=Y` | **12** | csmcir, conesep, ptha_mtst, habit, intent, hint, encoder, dcnet, combiner_rn50x4_noft, clip4cir_rn50x4_fullft, limn, tme |
| …4 aggregates newer than that matrix snapshot | **4** | airknow, clvc_net, sprc, tgcir (matrix says `native_executed=N`; the aggregate file exists — footnote 35) |
| native blocked | **1** | pair (`BLOCKED_MISSING_SOURCE_CLASSES`; its 4 counted numbers are `EXPLORATORY_PUBLIC_MAPPING`) |
| CIRCUS composition runs referenced (distinct `run_id`s with a summary) | **33** all · **30** non-synthetic non-audit · **24** with a 6003-query cohort | 36 `full.json` dirs; 2 `SYNTHETIC_*`; 1 `released_audit_task7_mm_any`; 6 ids are 5-or-8-query smokes |
| parity measured (≥1 paired `g1_g2` artifact set) | **5** | **encoder** (v1), **limn** (v1), **hint** (v2), **habit** (v2), **tgcir** (v2, sample-scope) — `pairs_on_disk.json:complete_pair_count`, re-measured 2026-11-13 |
| parity PASS per the gate (`evaluate_all()`, not prose) | **0** | `passed_models: []` — the *parity* verdict is held by **5** models, none passing; the **14** above is the native-evaluation count and does not depend on it |
| parity blocked / missing artifact | **12** | the other 12 registered ids (17 − 5 measured: 2 G1-only `conesep`/`intent`, 1 `NO_ADAPTER` `sprc`, 1 deferred `pair`, 1 scope-decision `ptha_mtst`, 2 specified-not-built `clvc_net`/`tme`, 5 blocked on interpreter/asset) |
| G1 PASS models (mandatory checks) | **5** | encoder 0/30 failing, habit 0/36, hint 0/36, limn 0/33, tgcir 0/36 — all five pass G1 on mandatory checks; each also carries 3 failing **non-mandatory** `relation.*.digest` checks (recorded as absent where the model has no relation tensor). At HEAD encoder was `FAIL` on 3 relation digests (footnote 36) |
| G2 PASS models | **0** | encoder 9 failures, hint 2, limn 3; every one is a relative/componentwise-term-only failure |
| models with a 6003-query CIRCUS run **and** a native aggregate | **9** | encoder, conesep, habit, intent, hint, limn, ptha_mtst, tgcir, tme (+clvc_net = 10 with a 6003 run; its runs are all PARTIAL) |

Footnote 35 — **the four aggregates the matrix column predates.** `NATIVE_EVAL_MATRIX.csv`
(`native_executed`) was written before four aggregate files existed:
`airknow` (`measurements/airknow_fiq_n05_run1.json`), `clvc_net` (`measurements/clvc_run1.json`),
`sprc` (`measurements/sprc_vitl_run1.json`), `tgcir` (`measurements/tgcir_run1.json`). Each has
`complete: true` and three per-category rows. The count of **16** comes from the artifacts; **12** is
what that column says. The artifacts win (same staleness class as `has_composition_run`,
§Cross-check).

Footnote 37 — **the 14 protocol-eligible native evaluations, derived (not incremented from the old
zero).** Command: intersect the models that have a native aggregate artifact
(`list_native_measurements()` metric-bearing rows, 18 ids) with the rows of
`workbench/docs/reviews/20261110-e2e-repair/PROTOCOL_MATRIX.csv` whose `row_kind = native_config`
(17 rows: protocol/loader/gallery/category attributes verified per row), then subtract the four
exclusions below. 18 ids − 4 exclusions = **14**.

Exclusions, each named with its reason:

| id | why it is not a valid native FashionIQ evaluation |
|---|---|
| `ptha_mtst` | **CIRR checkpoint → FashionIQ cross-dataset transfer**, not a FashionIQ reproduction. The artifact itself labels it `CIRR checkpoint -> FashionIQ | cross-dataset transfer | exploratory` and `PROTOCOL_MATRIX` notes read *"the reported number is cross-dataset transfer, not a reproduction"*. Its protocol *is* verified (`fashioniq_original_split`, CONFIRMED) — eligibility fails on the checkpoint's training dataset, not on the protocol. |
| `combiner_rn50x4_noft` | **roll-up-only served row.** The number 35.1078/58.4756 is read from `measurements/selection_decisions.json`; the only run artifact (`…020a70b0_aggregate.json`) is `EXECUTION_COMPLETED` but its `benchmark_gate` is `METRIC_REFERENCE_UNVERIFIED` with *"no committed reference describes this experiment"*, and `per_query_export_available: false`. A run exists; the evaluation is not certified. |
| `clip4cir_rn50x4_noft` | **roll-up / unbacked row.** The macro lives only in `measurements/clip4cir_parsed.json` under key `clip4cir_noft`, and the native service row for it is `counted=false, artifact_backed=false`. No protocol row of its own exists in `PROTOCOL_MATRIX.csv` (it shares the `combiner` registration). |
| `pair` | **exploratory public-class alias only.** Its four counted rows carry `mapping_status: EXPLORATORY_PUBLIC_MAPPING`; `PROTOCOL_MATRIX` says *"Not a native run; not admitted as acceptance evidence."* |

Two footnotes on the kept 14:

* `\*clip4cir_rn50x4_fullft` — protocol **eligible but mislabelled**: its recorded label is
  `fashioniq_full_gallery_ref_excluded` while the validated protocol is `fashioniq_original_split`
  (`label_verdict: MISLABELED`, evidence `third_party/CLIP4Cir/src/validate.py:21-57` — the FashionIQ
  branch has no reference mask; the only `reference_mask` is CIRR-only at `:162-166`). It is counted,
  at its **validated** protocol, not at its recorded one.
* `†csmcir`, `†sprc` — protocol **CONFIRMED** and aggregates present, but both are `USER_DEFERRED`
  (see footnote 31), so they are eligible evaluations that the campaign is deliberately not spending
  work on. Counting them keeps the row honest about what exists; the deferred disposition is carried
  in the Disposition table, not hidden here.

The remaining 12 (14 − 2 deferred) are the active, protocol-eligible, parity-independent native
evaluations.

Footnote 38 — **`hint` is the third parity-measured model, and the only one on v2.**
Its six artifacts (`g1_g2.hint.{dress,shirt,toptee}.{native,adapter}.json`, currently **untracked**)
declare `standard: parity-absolute-relative/v2` on both sides, and the gate detects it:
`evaluate_model('hint').applied_standard == 'parity-absolute-relative/v2'` — the first pair in this
repo judged componentwise (`abs(a-b) <= atol + rtol*abs(b)` per element) rather than on `max_abs`/
`max_rel` aggregates. Result: **G1 PASS** (36 mandatory checks, 0 failing) and **G2 FAIL** on exactly
two subchecks, `shirt.batch_1_vs_n.max_rel` and `toptee.batch_1_vs_n.max_rel` (measured
0.005049934976857354 and 0.04213516386699195 in `THIRD_MODEL_PARITY.md` §7). No mandatory dimension is
unmeasured (`missing: []`). Consequence for the vocabulary: `hint` moves from `INCOMPLETE`
(`NO_PARITY_EVIDENCE`) to `FAIL` (`PARITY_MANDATORY_SUBCHECK_FAILED`) — an artifact pair now exists and
the gate rejects it, which is exactly the rule footnote 29 states.
Prose source: `20261112-parity-repair/THIRD_MODEL_PARITY.md` §7 (do not restate its tables).

Footnote 39 — **`hint`'s G2 count settled at 2; the "8" recorded here was a transient between two states.**
Both the served endpoint and the current tree now report **2** mandatory failures
(`shirt.batch_1_vs_n.max_rel` 0.005049934976857354, `toptee.batch_1_vs_n.max_rel`
0.04213516386699195) with `status: PARITY_MANDATORY_SUBCHECK_FAILED` and
`applied_standard: parity-absolute-relative/v2`.

The "8" this footnote used to record was measured while the gate lane's `_metrics_checks` additions
existed on disk but the six `{dress,shirt,toptee}.metrics.eligible_{denominators,query_set_identical}`
checks had no counts behind them — they read `None` and failed. They are now **measured and ok**
(`native`/`adapter` denominators 2017/2038/1961 on both sides, membership `true`), so they are no
longer failures at all: the presence checks passed, they did not merely stop being mandatory.
Re-measured 2026-11-13 at handoff on the tree at `TESTED_SOURCE_SHA`: `evaluate_model('hint')` →
**2** failures, **12** metrics checks, all mandatory, all `ok: true`. The served number and the tree
number now agree, because the service was restarted onto this tree at `Fri 2026-11-13 10:10:07 +07`
(backend pid 2816638). The *status* is `FAIL` either way and `passed_models` stays `[]`.

Footnote 36 — **the committed baseline at HEAD differs, and that is why two verdicts moved.**
Evaluating the gate against `git show HEAD:…` copies of the same twelve artifacts (written to a
`/tmp` directory, never into the repo) gives:

| model | HEAD `40c897c` | working tree now |
|---|---|---|
| encoder G1 | `FAIL`, 3 failing (`relation.{dress,shirt,toptee}.digest` = `native=not_recorded, adapter=not_recorded`) | `PASS`, 0 failing |
| encoder G2 | `FAIL`, 13 failing (adds `{dress,shirt}.ranks.non_tie_changes` + `.min_gap_vs_tie_epsilon`) | `FAIL`, 9 failing (the three `population.*` + three `batch_1_vs_n` families) |
| encoder `scoring_parity` | `false` | `true` |
| limn `metrics_parity` | `missing` (key absent from all six files) | `true` (`native_metrics_match_feature_rerank: true`) |
| limn G2 | `FAIL`, 3 (`batch_1_vs_n.max_rel`) | `FAIL`, 3 (same three, same values) |
| hint | **no artifacts at all** (`NOT_RUN`, undiscovered) | **`FAIL`**: `PASS` G1 / `FAIL` G2 on 2 `batch_1_vs_n.max_rel`, judged componentwise under v2 |
| models discovered | 2 (`encoder`, `limn`) | 3 (`encoder`, `hint`, `limn`) |
| `passed_models` | `[]` | `[]` |

So **no overall verdict changed** (all three models are `PARITY_MANDATORY_SUBCHECK_FAILED` and
`passed_models` stays empty), but **three things moved**: encoder's G1 (`relation_parity`),
LIMN's `metrics_parity`, and a **new third pair** (`hint`) appearing on the v2 standard. The historical rank disagreements the
audit reconstructed from index dumps (dress index 1569 → `7:1565`, shirt index 1352 → `7:3362`) live
**only in the HEAD bytes** — the working-tree files record `count: 0`. Both are true statements about
different snapshots, which is exactly why the pair must be re-measured under one declared run
identity before either is cited as current.

**Two different questions, two different numbers — do not read one as the other.**

* **Does a native FashionIQ evaluation exist?** Yes, for **14** ids. That count is parity-independent:
  it asks only whether the protocol, loader, gallery and category scope were verified and whether an
  aggregate artifact exists at that evidence level. It is *the* number to quote when the question is
  "how much of this cohort has been measured", and it is the one the campaign spec protects when it
  says a genuine native-only aggregate stays visible at its actual evidence level.
* **Is any configuration parity-verified?** **0.** That asks a stricter question: a native **and** a
  Workbench run over identical inputs, agreeing per the gate with zero mandatory failures. Three models
  have a parity pair at all (`encoder`, `limn` on v1; `hint` on v2) and none passes (9, 3 and 2 G2
  failures respectively, all relative-or-componentwise-only).

Everything between those two numbers is evidence level, not existence: 16 ids executed an evaluator,
12 of them are `native_executed=Y` in `NATIVE_EVAL_MATRIX.csv`, 4 more aggregates postdate that
column (footnote 35), and 9 ids hold both a native aggregate and at least one completed 6003-query
CIRCUS run.

One more pair of counts that are also distinct, and neither of which is "verified":

* **16 "executed"** counts ids with a metric-bearing aggregate artifact, whatever produced it. That
  includes `clvc_net` and `tgcir`, whose runs came from `workbench/replay/*.py` shells rather than
  the upstream entrypoint (the pinned `test.py` in both has no CLI/loader), and `airknow`, whose
  registry env still declares the dangling symlink — which is why "executed" (16) is larger than
  "protocol-eligible" (14).
* **12 "native_executed=Y"** is the older matrix column, which additionally required
  `EXECUTION_COMPLETED` reporting on the historical path. It misses four later aggregates
  (footnote 35) and predates `clvc_net`/`tgcir`'s replayed runs.

`native_verified` stays `N` on all 18 rows because it means parity, and parity is `PASS` for
**zero** models.

---

## Cross-check against `INVENTORY.csv` (used only as a check)

`workbench/docs/reviews/20261111-master-audit/INVENTORY.csv` (18 lines, mtime `2026-11-10
19:30:11`) disagrees with the artifact-derived matrix in four places:

| INVENTORY.csv says | artifacts say | why |
|---|---|---|
| `clvc_net has_composition_run=N` | 6 `full.json` dirs / 4 ids, all PARTIAL | CSV predates the 2026-11-12 summaries (`da0a2162` written `09:15`) |
| `tgcir has_composition_run=N` | 5 dirs / 4 ids, 3 COMPLETED | same cause (`7b1fa850` written `05:16`) |
| `clip4cir_rn50x4_fullft has_native_measurement=N` | a counted, artifact-backed native row exists (`raw/native/clip4cir_fullft.json`) | the CSV was written before that artifact was committed |
| `sprc has_native_measurement=Y, status=BLOCKED_CHECKPOINT_MAPPING` | native aggregate present (`sprc_vitl 51.0386/73.3263`), disposition `USER_DEFERRED` | the CSV's Y is right; the blocker reads as "nothing measured", which is what the audit corrected |

Also note the CSV's `native_verified` column: `Y` for `limn` only, `N` for the rest — the table
above reports `N` for **all** seventeen because the gate adds `relation_parity` and
`metrics_parity`, neither of which existed when that column was written (§ Superseded, item 2).

---

## Superseded

Results this audit newly supersedes. Each line: what, where, why.

1. **`INPUT_FORWARD_PARITY.vi.md:255-260` — LIMN "G2 forward/scoring/ranking parity PASS".**
   Superseded by the gate (`evaluate_all()` → `PARITY_MANDATORY_SUBCHECK_FAILED`, 3 failures). The
   prose quotes `max_abs` values that do hold but omits that the mandatory `max_rel` subchecks do
   not; `GUARDS_FIXES.vi.md:448-478` already flagged the contradiction and this audit re-measured it.
   Note the *other* half of that prose ("41 hit-count evidence missing") **has since been fixed** in
   the working tree: `metrics_parity` is now measured `true` (footnote 26), so a current citation may
   say LIMN is missing hit counts only if it dates itself to HEAD.
2. **Every `native_verified=Y` for LIMN** (`NATIVE_EVAL_MATRIX.csv` row `limn`,
   `EVIDENCE_INDEX.md:113,141-143`, `FINAL_REPORT.vi.md:137`). Superseded: under the gate's
   definition (G1 **and** G2 with zero mandatory failures) it is `N`, because G2 has three
   `batch_1_vs_n.max_rel` failures — and that reason is unchanged by the `metrics_parity` fix.
3. **Both halves of the ENCODER relation story are now superseded, in sequence.**
   *(a)* "The pin is unwired in both processes" — never true: it is wired in both production paths
   (`model_encoder.py:500`, `replay/encoder.py:324`, one shared constant pair at `:59,65` imported at
   `replay/encoder.py:36`); the defect was the **producer** `scripts/encoder_parity.py`, which
   constructed `Encoder` unseeded and emitted no `relation_state`.
   *(b)* "ENCODER G1 fails on `relation.<cat>.digest`, therefore the pair is unusable" — superseded by
   the 2026-11-12 20:07–20:13 working-tree re-measurement: `scripts/encoder_parity.py` now imports
   `RELATION_SEED` and records `relation_state` with `matches_pinned_digest: true`, and the gate
   reports encoder **G1 PASS**. Encoder still fails G2 (9 relative-only checks), so its overall
   verdict did **not** improve past `PARITY_MANDATORY_SUBCHECK_FAILED` — see footnote 36 for the
   HEAD-vs-working-tree table.
4. **`CLVC`'s native macro as an unresolved conflict, and *this file's own earlier* "quote neither"
   rule.** Both are superseded by the controlled rerun in `CLVC_NATIVE.md` §3-§6: `30.02079703540087
   / 58.67189400437203` is the **batch-32** number (reproduced on all six R@10/R@50 cells),
   `29.970746517160162 / 58.62171212901515` the **batch-48** number (all nine cells), three batch-32
   passes are bit-identical (0 ranks moved, max |Δ| 0.0), and per-query ranks exist. Quote batch 32
   against composition runs and always name the batch. The 0.0501 pp gap is a batch-size effect of an
   unchanged evaluator, not a defect in either artifact.
5. **`FINAL_REPORT_20261112.vi.md:274` and `:1415` — "bundle đang phục vụ hiện nay là
   `assets/index-DDviexaW.js`".** Superseded: served == `dist` == `assets/index-NnjrdZLG.js`,
   `sha256 721d34d043858e80c88e76d72866ba8583508318baeebac599cebd1e1c2c80d0`, 392 769 B. Same for
   `102-report-staleness.md:256` and `20261112-parity-repair/BASELINE.md:366`.
6. **`INVENTORY.csv` as a scheduling input.** Its `has_composition_run` / `has_native_measurement`
   columns are stale for `clvc_net`, `tgcir`, `clip4cir_rn50x4_fullft` (§Cross-check).
7. **"SPRC's checkpoint is absent."** Superseded: `sprc_fiq_vitl.pt` is on disk,
   `sha256 1938610…bea08`, matching `models.yaml:1452`; only the paper's ViT-g is missing.
8. **The ENCODER native aggregate as a single reproducible number.** Superseded: the registry's
   56.223101675386054 / 77.66738539271432 and the review artifact's 56.290792549557 /
   77.633389132303 are **two independent unseeded draws** (`learnable_relation` is constructed from
   the global RNG and neither file records a relation state or a seed), so neither reproduces across
   runs. The seeded parity native side is the only cross-run-reproducible figure: **R@10
   56.225801935399 / R@50 77.533156738705**. See `ENCODER_RELATION.md` §8-§9b.
9. **`CLVC`'s `image_only_empty_text` as a supported condition.** Superseded by
   `eab10106b2`'s `ABLATION_UNSUPPORTED` (CLVC-Net cannot encode an empty edit,
   `text_model.py:84-86`); `da0a2162` drops it in favour of `clvc_image_encoder_only`.

---

## Commands used (every number in this file)

```
cd /home/heheboiz/data/cir

# row source + registry order
python3 -c "import yaml;ms=yaml.safe_load(open('workbench/registry/models.yaml'))['models'];print(len(ms));[print(m['model_id']) for m in ms]"

# gate authority (G1/G2 status, failing checks, dimensions, missing)
python3 -c "import sys;sys.path.insert(0,'.');from workbench.backend import parity_gate as g;
r=g.evaluate_all();print(r['passed_models']);
[print(m,v['status'],v['failure_count'],v['missing']) for m,v in r['models'].items()]"

# per-gate mandatory/failing rollup and every failing check name
python3 -c "import sys;sys.path.insert(0,'.');from workbench.backend import parity_gate as g;
[print(m,g.evaluate_model(m).status) or [print(' ',c.gate,c.name,c.measured,c.declared) for c in g.evaluate_model(m).checks if c.mandatory and not c.ok] for m in g._discover_models(g.evidence_root())]"

# composition summaries: dirs, run ids, query counts, topk, per-model states
python3 -c "import json,os;b='workbench/artifacts/composition/summary';
[print(d,json.load(open(os.path.join(b,d,'full.json'))).get('run_id'),json.load(open(os.path.join(b,d,'full.json'))).get('execution_status'),json.load(open(os.path.join(b,d,'full.json'))).get('query_count')) for d in sorted(os.listdir(b)) if os.path.exists(os.path.join(b,d,'full.json'))]"

# native rows as the service sees them (counted, artifact_backed, protocol, metrics, path)
python3 -c "import sys;sys.path.insert(0,'.');from pathlib import Path;from workbench.backend.native_measurements import list_native_measurements as L;r=L(Path('workbench/artifacts'));
[print(x['model_id'],x['counted'],x['artifact_backed'],x['protocol_id'],x['metrics'],x['source_path']) for x in r['measurements'] if x['metrics']];print(r['counts'])"

# checkpoint digests as computed on disk
python3 -c "import json;d=json.load(open('workbench/docs/reviews/20261109-native15-composition-limn/scripts/checkpoint_hashes_canonical.json'));
[print(k,v['kind'],v['sha256']) for k,v in d.items()]"
sha256sum workbench/artifacts/checkpoint_selection/20261106T002617+0700/checkpoints/sprc/sprc_fiq_vitl.pt
sha256sum workbench/artifacts/checkpoint_selection/20261106T002617+0700/checkpoints/pair/*.pt

# interpreter presence (dangling symlinks)
test -e workbench/envs/airknow-adapted/bin/python && echo EXISTS || echo BROKEN_SYMLINK
test -e workbench/envs/pair-adapted/bin/python && echo EXISTS || echo BROKEN_SYMLINK
for e in ctl312h ilearn39-adapted encoder310-adapted limn310-adapted tme-adapted clvc39-la tgcir39-la pair39-la airknow310-la; do printf "%-24s " "$e"; workbench/envs/$e/bin/python -c "import sys;print(sys.version.split()[0])"; done
for d in workbench/artifacts/environments/*/; do printf "%-50s " "$d"; "$d/bin/python" -c "import sys;print(sys.version.split()[0])"; done

# benchmark (paper) gate states
python3 -c "import json,glob
for p in sorted(glob.glob('workbench/artifacts/reports/*_aggregate.json')):
    d=json.load(open(p));bg=d.get('benchmark_gate') or {}
    print(d.get('model_id'),d.get('run_id','')[-8:],d.get('execution_status'),bg.get('status'),bg.get('passed'))"

# the SAME gate against the committed HEAD artifacts, for footnote 36 (writes only to /tmp)
python3 -c "import sys,subprocess,pathlib,tempfile;sys.path.insert(0,'.');
from workbench.backend import parity_gate as g
names=[f'g1_g2.{c}.{s}.json' for c in ('dress','shirt','toptee') for s in ('native','adapter')]+[f'g1_g2.encoder.{c}.{s}.json' for c in ('dress','shirt','toptee') for s in ('native','adapter')]
t=pathlib.Path(tempfile.mkdtemp())
[ (t/n).write_text(subprocess.run(['git','show','HEAD:workbench/docs/reviews/20261109-native15-composition-limn/'+n],capture_output=True,text=True).stdout) for n in names]
[ print(m, g.evaluate_model(m,root=t).status, g.evaluate_model(m,root=t).missing) for m in ('encoder','limn') ]"

# artifact working-tree vs HEAD digests (the 20:07-20:15 change)
git status --porcelain -- workbench/docs/reviews/20261109-native15-composition-limn | head -20
git diff --stat -- workbench/docs/reviews/20261109-native15-composition-limn

# CLVC-Net: recompute the two macros from the per-query rank JSONL, and the batch sensitivity
python3 -c "
import json, statistics
B='workbench/docs/reviews/20261112-parity-repair/raw/clvc_native/'
for p in ('batch32','batch48','batch32_r2','batch32_r3','batch1'):
    cats={'batch1':('dress',)}.get(p,('dress','shirt','toptee'))
    r10=[];r50=[]
    for c in cats:
        rows=[json.loads(l) for l in open(B+p+'/'+c+'.jsonl')]
        g=len({r['reference_gallery_index'] for r in rows} | {r['target_gallery_index'] for r in rows})
        r10.append(sum(1 for r in rows if r['target_rank']<10)*100.0/len(rows))
        r50.append(sum(1 for r in rows if r['target_rank']<50)*100.0/len(rows))
    print(p, 'macro_r10=%.15f macro_r50=%.15f' % (statistics.fmean(r10), statistics.fmean(r50)))
"
python3 -c "import json;d=json.load(open('workbench/docs/reviews/20261112-parity-repair/raw/clvc_native/summary.json'));
print(json.dumps(d['batch_sensitivity_vs_batch32'], indent=1)[:1200])"

# protocol-eligible native evaluations: 18 aggregate ids minus the 4 named exclusions
python3 -c "import csv
P='workbench/docs/reviews/20261110-e2e-repair/PROTOCOL_MATRIX.csv'
M='workbench/docs/reviews/20261112-parity-repair/MODEL_MATRIX.csv'
proto={r['model_id']:r for r in csv.DictReader(open(P)) if r['row_kind']=='native_config'}
mx={r['model_id']:r for r in csv.DictReader(open(M))}
EXCL={'ptha_mtst','combiner_rn50x4_noft','clip4cir_rn50x4_noft','pair'}
cand=[m for m in mx if mx[m]['native_run_id_or_artifact']!='NONE']
elig=[m for m in cand if m in proto and m not in EXCL]
print('aggregate ids', len(cand), 'protocol-eligible', len(elig)); print(sorted(elig));
print('excluded', sorted(EXCL))"

# the seeded encoder macro (the only cross-run-reproducible encoder figure)
python3 -c "
import json, statistics
B='workbench/docs/reviews/20261109-native15-composition-limn/'
r10=[];r50=[]
for c in ('dress','shirt','toptee'):
    m=json.load(open(B+f'g1_g2.encoder.{c}.native.json'))['native']['native_metrics_recheck']
    r10.append(m[c+'_r10']); r50.append(m[c+'_r50'])
print('seeded macro_r10=%.15f macro_r50=%.15f' % (statistics.fmean(r10), statistics.fmean(r50)))"

# third parity model + which standard each pair is judged under (footnote 38)
python3 -c "
import sys; sys.path.insert(0,'.')
from workbench.backend import parity_gate as g
print(g._discover_models(g.evidence_root()))
[print(m, g.evaluate_model(m).status, g.evaluate_model(m).applied_standard,
       len([c for c in g.evaluate_model(m).checks if c.gate=='G1' and c.mandatory]),
       len([c for c in g.evaluate_model(m).failures]))
 for m in g._discover_models(g.evidence_root())]"

# bundle identity (superseded item 5)
grep -rn "index-DDviexaW" workbench/docs | wc -l
curl -s http://127.0.0.1:5173/ | grep -o 'assets/index-[^"]*\.js'
sha256sum workbench/frontend/dist/assets/index-NnjrdZLG.js
```
