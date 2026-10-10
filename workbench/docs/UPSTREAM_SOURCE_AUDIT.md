# Upstream source audit (cloned, pin-verified)

Audit date: 2026-10-10. All official upstream repositories were cloned source-only
(`GIT_LFS_SKIP_SMUDGE=1`) into `workbench/third_party/` (git-ignored). No model
checkpoints, datasets, pretrained weights, or LFS objects were downloaded. No
package was installed. No inference was run.

Cloning is acquisition only. A cloned source tree is **not** evidence that a model
can be loaded, evaluated, or reproduced.

## Pin verification method

```bash
cd workbench/third_party/<dir>
git rev-parse HEAD                       # must equal registry upstream_commit_sha
git status --porcelain                   # tracked modifications must be empty
```

Plain `git status` shows `__pycache__`/`*.pyc` entries in several trees. Some
upstream repositories even commit `.pyc` files. Bytecode is a Python build product
that cannot change evaluator source semantics, so the workbench's `_source_dirty`
check excludes it from both the tracked and untracked/ignored sets while still
treating any other tracked change, untracked file, or locally placed (even
git-ignored) asset as dirty.

## Acquisition results

| Model(s) | Directory | Commit (verified) | Bytes | Evaluator file |
| --- | --- | --- | --- | --- |
| CSMCIR | `CSMCIR` | `774f94e2076f` | 11M | `src/validate_blip_csmcir.py` ✅ |
| Air-Know | `AirKnow` | `6ab6fa95b746` | 91M | `test_BLIP2.py` ✅ |
| ConeSep | `ConeSep` | `bd4bf629ebe6` | 91M | `test.py` ✅ |
| HABIT | `HABIT` | `6ddc2cd63eac` | 92M | `test.py` ✅ |
| INTENT | `INTENT` | `2a75fc6f57d1` | 90M | `test.py` ✅ |
| HINT | `HINT` | `bec50b6c8c19` | 114M | `test.py` ✅ |
| ENCODER | `ENCODER` | `29a2a31d6a56` | 33M | `evaluate_model.py` ✅ |
| PAIR | `PAIR` | `bfb0b98cdb4c` | 31M | `test.py` ✅ |
| CLVC-Net | `CLVCNet` | `bd9b68894898` | 348K | `test.py` ✅ |
| DCNet | `DCNet` | `68c79d38569f` | 5.7M | `test.py` ✅ |
| Combiner / CLIP4Cir | `CLIP4Cir` | `dfed9f748a8a` | 1.9M | `src/validate.py` ✅ |
| TG-CIR | `TGCIR` | `65fa78eaf8ca` | 368K | `test.py` ✅ |
| SPRC | `SPRC` | `2935a5397732` | 9.0M | `src/blip_validate.py` ✅ |
| LIMN | `LIMN` | `7d7bc9b116f5` | 12M | `LIMN/test.py` ✅ |

Total on disk: **≈582 MB**. All 15 source-bearing registry models report
`verified=True, dirty=False` through `runtime.source_provenance`. PTHA + MTST has no verified upstream repository in the
registry and was not cloned (nothing invented).

**Deduplication:** Combiner RN50x4 noft and CLIP4Cir RN50x4 fullft share
`ABaldrati/CLIP4Cir` at the same commit; one checkout serves both model IDs.

## Source-level findings confirmed against the pinned code

- **ENCODER**: `evaluate_model.py` imports `datasets1` (line 7) and
  `model_try2`; only `datasets.py` is present. The missing module is a real
  source-level blocker, not a guess. Confirms `command_status: COMMAND_AUDITED`
  plus a prerequisite blocker.
- **LIMN / TG-CIR / CLVC-Net**: pinned `test.py` defines an evaluation function
  only — no `argparse` and no `if __name__` entrypoint. A standalone replay CLI
  genuinely does not exist in the pinned sources, so the registry blockers
  (`BLOCKED_NO_SOURCE_REPLAY` / `COMMAND_BLOCKED_SOURCE_ENTRYPOINT`) are correct.
  LIMN is served by the reviewed workbench replay wrapper instead.
- **SPRC**: `src/blip_validate.py` does have an `argparse` entrypoint, but the
  checkpoint's model/backbone class-key mapping remains unverified; the blocker is
  checkpoint mapping, not CLI absence.
- **DCNet**: run directory layout (`config.json`, `trained_model.pth` under
  `experiments/…`) matches the registry `run_directory` artifact contract.

## What cloning does and does not change

- **Does:** make evaluator entrypoints inspectable, allow source-faithful adapter
  auditing, and give `doctor.py` real source pin/dirty evidence.
- **Does not:** verify checkpoints, verify metrics, or authorize evaluation.
  Model readiness is unchanged; see `workbench/docs/UPSTREAM_AUDIT.md` for the
  per-model scientific blockers.

## Environment footprint

Source-only clones, no LFS. Disk budget respected (≥5 GiB free maintained). No
downloads of checkpoints, datasets, or pretrained backbones.
