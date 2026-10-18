# FINAL 16-MODEL AUDIT MATRIX

Every row is an independent audit of ONE registered model workflow against its **pinned upstream
source**, cloned at the exact `upstream_commit_sha` in `workbench/registry/models.yaml` under
`/tmp/upstream/<source_dir>` at audit time. The audits were performed by subagents across four model
families; the raw reports are the evidence for this table and are reproduced verbatim in the
`verdict` column's reasoning.

- **Tested source at the time of THIS matrix**: `84ec6f1`. Superseded; the current release metadata
  is `FINAL_PC_RELEASE_VERDICT.md`.
- **Audit inputs**: 15 distinct upstream repositories, all fetched and checked out at their pinned
  SHAs. `git rev-parse HEAD` in each clone matched the registry exactly.
- **What "adapter" means**: `workbench/backend/adapters/models.py`. The adapter's only job is to
  invoke the PINNED UPSTREAM code with correct arguments. An adapter that names a script without a
  CLI entrypoint, or passes a flag the script does not define, is a defect regardless of whether the
  model could run for other reasons.

## How to read the verdict column

| Verdict | Meaning |
|---|---|
| `EXTERNAL_BLOCKER` | The workflow cannot execute here, and the cause is outside this repository: missing/not-public source, an unpublished checkpoint or auxiliary asset, a missing public dependency, or no CLI entrypoint in the pinned upstream. Not a defect in this workbench. |
| `CONFIRMED_DEFECT` | A defect in first-party code, reproduced, with a regression test. |
| `BY_DESIGN` | The workbench refuses the model deliberately and says so. Not a defect. |
| `REJECTED_FINDING` | An allegation that was investigated and disproved. Recorded so it is not re-raised. |

## The matrix

| # | model_id | pinned source | adapter entrypoint | protocol (verified in source) | per-query | audit verdict | what remains |
|---|---|---|---|---|---|---|---|
| 01 | `csmcir` | `qzp2018/CSMCIR` @ `774f94e` | `src/validate_blip_csmcir.py`, `--dataset fashionIQ --blip-model-path` — **matches upstream argparse** | `fashioniq_original_split` — full `split.{cat}.val.json`, reference never masked. **Verified** | `AGGREGATE_ONLY` | `EXTERNAL_BLOCKER` | Pinned `lavis/models/__init__.py` has 98 unguarded imports of modules absent from the repo, before the line that registers the model; evaluation also needs an unpublished `COT_ours2/...cot_val.json`. |
| 02 | `airknow` | `iLearn-Lab/CVPR26-Air-Know` @ `6ab6fa9` | `test_BLIP2.py` — exists, but defines only an in-memory function: no `__main__`, no argparse | `fashioniq_full_gallery_ref_excluded` — full val split in split-file order, reference masked `-10e10`. **Verified** | `AGGREGATE_ONLY` | `EXTERNAL_BLOCKER` (adapter gap is a defect of the upstream, not of this repo) | `lavis.models` imports two files (`blip_cir`, `blip_cir_caption`) that exist only as bytecode; `test_BLIP2.py` evaluates in `model.train()` mode with dropout active; hardcodes `/root/bert-base-uncased`; mandatory correction dictionaries absent. |
| 03 | `conesep` | `Lee-zixu/ConeSep` @ `bd4bf62` | `test.py` — in-memory function only, no CLI | `fashioniq_full_gallery_ref_excluded` — full val split, reference masked. **Verified** | `AGGREGATE_ONLY` | `EXTERNAL_BLOCKER` | No CLI entrypoint; hardcoded `/root/bert-base-uncased`; correction dictionaries unpublished; entire `val-split` branch in its dataset code is dead. |
| 04 | `ptha_mtst` | `zyinan99/MTST_PTHA` — **no pinned commit** (`source_available: false`) | none | **unresolved by design** — registry correctly leaves `native_protocol: null` | n/a | `EXTERNAL_BLOCKER` + one first-party defect, **fixed** | The launcher substituted `fashioniq_val_split` for a model whose protocol is not established. Fixed: no default protocol, an undeclared protocol is refused (`pc.sh`, `F-18`). |
| 05 | `habit` | `iLearn-Lab/AAAI26-HABIT` @ `6ddc2cd` | `test.py` — in-memory helper only, no CLI | `fashioniq_full_gallery_ref_excluded` — full val split; the `val-split` pair-union branch is dead code. Reference masked. **Verified** | `AGGREGATE_ONLY` | `EXTERNAL_BLOCKER` | No CLI entrypoint; no clean 0%-noise checkpoint published; correction dictionaries unpublished; hardcoded `/root/bert-base-uncased` and `/export/home/.cache/lavis`; cache writing requires a writable dataset root. |
| 06 | `intent` | `iLearn-Lab/AAAI26-INTENT` @ `2a75fc6` | `test.py` — in-memory helper only, no CLI | `fashioniq_full_gallery_ref_excluded` — same construction as HABIT. **Verified** | `AGGREGATE_ONLY` | `EXTERNAL_BLOCKER` | Same class as HABIT. Additionally: `train_INTENT.py --split` can switch the protocol to `val-split`, which the registry's preparation contract does not mention — a documentation gap, not a protocol change. |
| 07 | `hint` | `iLearn-Lab/ICASSP26-HINT` @ `bec50b6` | `test.py` — in-memory helper only, no CLI | `fashioniq_val_split` — pair-union gallery, reference masked. **Verified** | `AGGREGATE_ONLY` | `EXTERNAL_BLOCKER` | No CLI entrypoint; the pinned tree contains **no FashionIQ checkpoint loader** (only CIRR); `setproctitle` and `torchaudio` missing from the declared environment; `correction_dict_{cat}.json` unpublished; `train.py` passes an argument the dataset constructor does not accept. |
| 08 | `encoder` | `iLearn-Lab/AAAI25-ENCODER` @ `29a2a31` | `evaluate_model.py`, `--dataset fashioniq --fashioniq_split val-split --fashioniq_path --ckpt_path` — **matches upstream argparse** | `fashioniq_val_split` — dedup ordered union of source then target ids; reference masked. **Verified** | `AGGREGATE_ONLY` | `EXTERNAL_BLOCKER` | The pinned evaluator dies at import: `import datasets1` (the file is named `datasets.py`); it also needs an unversioned local `./open_clip_pytorch_model.bin`; loads with `strict=False`, so missing keys leave submodules randomly initialised; checkpoint mapping unverified. |
| 09 | `pair` | `iLearn-Lab/ICASSP25-PAIR` @ `bfb0b98` | none — `PairAdapter` has no `command()` (base raises) | `fashioniq_val_split` — same union construction; reference masked. **Verified** | `AGGREGATE_ONLY` | `BY_DESIGN` (workbench) / `EXTERNAL_BLOCKER` (upstream) | The pinned `test.py` has no `__main__` and no argparse, and `model.py` imports a non-existent `MHTransformer_rev`. The workbench fails closed and records `COMMAND_BLOCKED_SOURCE_ENTRYPOINT`, which is the honest outcome. |
| 10 | `clvc_net` | `iLearn-Lab/SIGIR21-CLVC-Net` @ `bd9b688` | none | `fashioniq_val_split` — pair-union **reproduced from host data**: dress 2628, shirt 3089, toptee 2902, all strictly below the full-split sizes. Reference masked. **Verified** | `AGGREGATE_ONLY` | `EXTERNAL_BLOCKER` | No CLI entrypoint; the six whole-model members the trainer writes are unmapped from the author's Drive folder; legacy Python 3.7 / torch 1.6 / CUDA 10.2 environment, and `torchvision.models.utils` no longer exists in any modern version. |
| 11 | `dcnet` | `ozmig77/dcnet` @ `68c79d3` | `test.py --resume <dir>` — flag exists and is required; `--resume` is the only flag that can be passed | `fashioniq_full_gallery_ref_excluded` — full val split; reference masked in both score branches before `log_softmax` combination. **Verified** | `AGGREGATE_ONLY` | `EXTERNAL_BLOCKER` | The dataset root comes from `config.json` inside the checkpoint directory, not a flag, and the registry does not declare that; GloVe caption artifacts are generated with an **unseeded** RNG for OOV words, so they are not reproducible without the author's published `.pkl`; the official run directory is an unmapped Drive folder. |
| 12 | `combiner_rn50x4_noft` | `ABaldrati/CLIP4Cir` @ `dfed9f7` | `src/validate.py` with `--projection-dim 2560 --hidden-dim 5120 --clip-model-name RN50x4 --target-ratio 1.25 --transform targetpad` — **all flags verified against argparse and against the model's own dimensions** (6404 → 2560, 6408 → 5120) | `fashioniq_original_split` — full gallery, reference **kept**. **Verified** | `AGGREGATE_ONLY` | `EXTERNAL_BLOCKER` | Checkpoint mapping unverified. The registry publishes **no** paper row for this variant, which is correct: the pinned paper has no RN50x4 noft FashionIQ row to cite. |
| 13 | `clip4cir_rn50x4_fullft` | same repo @ `dfed9f7` | `src/validate.py` + `--clip-model-path <ckpt>/clip_state.pt`; the loader executes `load_state_dict(saved_state_dict["CLIP"])` | `fashioniq_original_split` — full gallery, reference kept. **Verified** | `AGGREGATE_ONLY` | `EXTERNAL_BLOCKER` | Two-state bundle (`combiner_state.pt` + `clip_state.pt`) with no published pairing manifest or checksums; registry marks `BLOCKED_CHECKPOINT_BUNDLE` / `CHECKPOINT_MAPPING_UNVERIFIED`. The published row (43.78 / 66.93 / 55.36) was checked against the pinned paper's table and is transcribed correctly, with the mean derived as the two-value mean. |
| 14 | `tgcir` | `iLearn-Lab/MM23-TG-CIR` @ `65fa78e` | none | `fashioniq_val_split` — accumulated dedup union of reference and target ids; reference masked. **Verified** | `AGGREGATE_ONLY` | `BY_DESIGN` (workbench) / `EXTERNAL_BLOCKER` (upstream) | No CLI entrypoint in `test.py`; the Drive archive has no member mapping or digest; the model is saved as a **whole-model pickle** (`torch.save(model)`), which is an arbitrary-code-execution hazard on load and is why the registry will not pin a digest for it; the declared `clip==0.2.0` PyPI pin is not OpenAI CLIP. |
| 15 | `sprc` | `chunmeifeng/SPRC` @ `2935a53` | **none** — `SPRCAdapter` has no `command()` (base raises). Upstream requires `--model-path`, not `--blip-model-path` | `fashioniq_original_split` — full gallery, reference kept. **Verified** | `AGGREGATE_ONLY` | `CONFIRMED_DEFECT` (adapter) + `EXTERNAL_BLOCKER` | Defect: the adapter registered a script it never wraps, and a sibling flag name that upstream does not accept. Left unwrapped deliberately for now: wrapping it would still not run, because the argparse default `blip2_cir_rerank_learn` is not a LAVIS-registered model, `clip` is missing from `requirements.txt`, and the OneDrive checkpoint URL returns HTTP 403. Recorded as `BLOCKED_CHECKPOINT_MAPPING`. |
| 16 | `limn` | `iLearn-Lab/TPAMI24-LIMN` @ `7d7bc9b` | **`workbench/replay/limn.py`**, not `LIMN/test.py` — the pinned `test.py` has **zero** argparse and no `__main__`, and the adapter correctly execs the replay driver, which takes `--checkpoint-root/--dataset-root/--source-root` (all required) and optional `--category` | `fashioniq_val_split` — first-seen union of query endpoints, reference masked in-row to `-10e10` and **retained as a column** (which is why `eligible = gallery_size - 1` is the right bound). **Verified against the pinned file byte for byte** (`diff -r` of the clone against `workbench/third_party/LIMN` was empty) | **the ONLY model with a per-query path**: two-stage observer → parity → canonical publication | `EXTERNAL_BLOCKER` (GPU, dataset, checkpoints) — **no defect on the success path** | Nothing in the LIMN chain blocked the pilot. The four findings the cross-check produced were all P2/P3 first-party defects, now fixed: the diagnostic reason was overwritten (`F-15`), the publisher re-implemented the bundle digest with different keying (`F-16`), the provenance list named an alternative twice (`F-17`), and an unresolvable checkpoint id made the proof self-referential (`F-14`, GLM). |

## Cross-model conclusions

1. **Protocols are correct.** For every one of the 16 models, the candidate universe and reference
   policy were read out of the PINNED source, not from the registry: `fashioniq_original_split` keeps
   the reference in a full gallery, `fashioniq_full_gallery_ref_excluded` masks it in a full gallery,
   and `fashioniq_val_split` masks it in a deduplicated ordered union. In every case the registry's
   `native_protocol` matched the code. **No model's protocol needed changing, and none was changed.**

2. **Only one model has a per-query path.** LIMN. The other fifteen are `AGGREGATE_ONLY` because
   their upstream code prints category and macro recalls and nothing else. None of them may be
   presented as query-level capable, and none is.

3. **The dominant blocker is upstream, not first-party.** Ten of the sixteen cannot execute here for
   reasons outside this repository: five have no CLI entrypoint at all (HABIT, INTENT, HINT, CLVC-Net,
   TG-CIR), and five need an unpublished artifact (correction dictionaries, GloVe caches generated
   with an unseeded RNG, a clean 0%-noise checkpoint, a whole-model archive with no digest).

4. **The adapter is fail-closed by construction.** `OfficialScriptAdapter.command` raises
   `NotImplementedError` rather than inventing a command line. Every model without an audited command
   is therefore refused at planning time and recorded with a reason, which is why ten
   `EXTERNAL_BLOCKER` rows did not become twelve silent-failure rows.

5. **Two adapter-level defects were confirmed and are now documented rather than papered over**:
   `sprc` registered a script with no wrapper and a flag name upstream does not accept, and the
   `ptha_mtst` launcher path invented a protocol. The second is fixed in code; the first is recorded
   with the three independent external reasons that make wrapping it pointless today.
