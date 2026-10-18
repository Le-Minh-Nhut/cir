# R8 — FINAL REPORT: RELEASE REPAIR, EVIDENCE INTEGRITY & PC HANDOFF

## A. GIT STATE

| | |
|---|---|
| INITIAL_HEAD_SHA | `2bc0cd223028057b577b861d83d9e2795b040840` — matched the expected start |
| RELEASE_SOURCE_SHA | `59949ff` — the R8 production changes |
| **TESTED_SOURCE_SHA** | **`17d3947`** — every gate in this report ran on it (docs-only successor of `59949ff`, production tree identical) |
| **FINAL_HEAD_SHA** | `17d39476c08d32c5c78cdd5d7dfb4eb7abbb9d55` |
| Remote HEAD | `17d3947` — local equals remote |
| `origin/main` | `8ebcc74d5fd1934df4f3772a8ca5a96dbd00f333` — **unchanged** |
| Worktree | clean; no checkpoint, dataset, cache or secret tracked; nothing rebased, reset or amended |

**The self-reference is avoided by construction.** A tracked document cannot contain its own commit's
SHA, so the documents name an immutable *source* revision and `git rev-parse HEAD` is the authority for
the tip. Two commits, deliberately separate: `59949ff` (production) and `17d3947` (documentation
naming it), with `git diff 59949ff 17d3947 -- . ':(exclude)workbench/docs' ':(exclude)workbench/tests'`
**empty** — so the handoff installs the code that was gated.

## B. CONFIRMED DEFECTS

| ID | Verdict | Reproduction | Fix | Regression |
|---|---|---|---|---|
| **R8-01** | CONFIRMED P1 | `git show 11ba3df:pc.sh` contains **0** occurrences of `sys.exit(7)` and of `REGISTRY_UNKNOWN`; the tested revision has **1** and **5**. The handoff instructed `git checkout 11ba3df` while naming exit 7 on the same page | Production-tree equality, working-tree containment, pinned-revision only, documents advanced | 3 tests + the obsolete-ancestor counterexample; documents edited and the guard still refused |
| **R8-02** | CONFIRMED P2 | `E2BIG` reproduced at the real boundary: 131072 bytes in one argv element raises `OSError: [Errno 7]`; `/api/benchmark-runs` crosses it at 30–37 runs | Four sites pass file paths; EXIT-trap lifetime | Gate driven past the ceiling (must succeed); same body as an argv element (must still raise E2BIG); unreachable/empty/malformed preserved |
| **R8-03** | CONFIRMED P1 in-gate / NOT REACHABLE end-to-end | A development receipt carries `certified=true` and `is_research_evidence=true` with no proof validated; the gate read only those | Both signals checked by name | Development-mode alone, certification-mode alone, both, neither, + deployment positive control |
| **R8-04** | CONFIRMED P1 | Publisher **published** a one-member "three-category" artifact with no planned digest: `rc=0`, measured | Parent binds the artifact to the plan-derived digest (same expression the proof records); publisher validates against the **registry's** declared bundle | Partial bundle refused by name; registry-correct bundle publishes; single-category checkpoint still publishes |
| **R8-05** | CONFIRMED P2 | `FETCH_HEAD` offered as checkout; "thirteen" against twelve rows; a false negative control in the ledger; manifest stale in **45 of 126** rows | All corrected | Stale count now refused |

## C. ADDITIONAL DEFECTS (found by review, in this pass's own work)

| ID | Sev | Finding | Evidence |
|---|---|---|---|
| **R8-06** | **P1** | The gate prints its token on stdout and its notes on stderr *so they cannot mix*; the shell read it with `2>&1`. The note poisoned the token, no summary branch matched, and a proof-unavailable run was reported **pilot-ready with exit 0** instead of exit 7 | Reproduced through the real gate program and the real summary |
| **R8-07** | **P1** | Temp files leaked on every non-zero exit (`fail_stage` and `sys.exit` leave before the cleanup line) | Measured: pre-fix launcher leaves `cir_verify_state.*` after a proof-unavailable run and after a failed index rebuild; fixed tree leaves nothing |
| **R8-08** | P2 | The cleanup test asserted a source substring — the class the brief forbids — and passed on a tree that leaks | Replaced; the new test was checked **RED** against the pre-fix launcher first |
| **R8-09** | P2 | Status doc called `command_status` a registry field (no such key); said encoder had no `command()` (it does, `adapters/models.py:36-41`); described parsers as general (5 of 16) | Matrix regenerated with every cell read from the code |

## D. LARGE JSON INTEGRATION

Kernel limits measured on this host: `ARG_MAX` = 2097152, `MAX_ARG_STRLEN` = **131072** (32 pages).
131071 bytes passes, 131072 raises `E2BIG`. Growth drivers: `/api/runs` full table dump (43–65 runs),
`/api/benchmark-runs` one row per report (30–37 runs); the 48-run workload of this project yields
≈173 KiB. All four carriage sites now use paths; the tests drive bodies past the ceiling and prove the
old form still fails on the same input.

## E. DEVELOPMENT CERTIFICATE SECURITY

Production certification requires `certification_mode == "deployment"` and `development_mode == false`,
and a development process can only reach that from a report WITH a completion proof. The gate now
refuses on either signal by name. End-to-end the heavy verifier already refused first, so the operator's
exit code was already correct — the gate is now correct independently, which is the point: the program
that decides the success token must not depend on another program for its own integrity.

## F. LIMN SCIENTIFIC EVIDENCE

The independent science review **rejected all five allegations of drift**: the change is `+20/−0` in one
hunk after artifact admission; failure only flips `per_query_state`; every ranking, Recall, gallery,
eligibility and checkpoint loader is outside the diff; all three on-disk checkpoints match the registry
digests; Protocol C and the Top-K contract are untouched. Both digest sides route through one
`manifest_digest`; measured equal across keying shapes. The payload-vs-payload hole was real and is
closed by the parent check against the plan.

## G. SCIENTIFIC INVARIANTS

Protocol A `fashioniq_original_split` (4 models) · B `fashioniq_full_gallery_ref_excluded` (5) · C
`fashioniq_val_split` (6) · `ptha_mtst` **UNRESOLVED**. No protocol, reference-eligibility rule, gallery
construction, Recall calculation, checkpoint variant or upstream pin was changed. `limn` remains the
only model with a per-query observer.

## H. SIXTEEN MODEL STATUS

`workbench/docs/reviews/r8/R8_16_MODEL_STATUS.md`, every cell read from the registry and the code.
**Readiness: 1 READY_FOR_PC_PILOT (`limn`) · 1 READY_AFTER_SMALL_FIX (`encoder`) · 2
NEEDS_CHECKPOINT_MAPPING (the CLIP4Cir pair) · 1 NEEDS_UPSTREAM_ASSET (`dcnet`) · 1 NEEDS_AUTHOR_INPUT
(`ptha_mtst`) · 10 NEEDS_WRAPPER.** Runtime verification level is **0 for all sixteen**; GPU-verified
count is **0**.

## I–L. ENCODER · CLIP4CIR · PTHA+MTST · DCNET

**ENCODER**: `RECOVERY_PLAN_IDENTIFIED`, not fixed. Its adapter **does** override `command()`; missing
are the import alias patch, the LAION ViT-B/32 asset, and the metric parser. **CLIP4CIR/COMBINER**:
release verified independently — tag `weights`, published **2026-10-12**, five parts, **8.55 GB**, all
five SHA-256 digests published. Status `CHECKPOINT_ARCHIVE_PUBLIC` + `MEMBER_MAPPING_PENDING_DOWNLOAD`,
**not** `CHECKPOINT_READY`. **PTHA+MTST**: `PUBLIC_AUTHOR_SOURCE_EXISTS` / `SOURCE_NOT_INTEGRATED` /
`CHECKPOINT_UNAVAILABLE` / `PROTOCOL_UNVERIFIED` — "source does not exist" is no longer claimed.
**DCNET**: `ORIGINAL_PREPROCESSING_PARITY_UNVERIFIED` + `AUTHOR_CACHE_UNAVAILABLE`; "permanently
unreproducible" is withdrawn, native preprocessing and scoring untouched.

## M. FULL TEST RESULTS

| Command | Exit | Result |
|---|---|---|
| `pytest workbench/tests -q` | **0** | **1427 passed, 0 failed** (26:16) |
| `bash workbench/docker/acceptance.sh` | 0 | ACCEPTANCE PASSED (1 honest skip) |
| `bash workbench/docker/mutation_checks.sh` | 0 | PASSED M1 M2 M3 M4 M5 |
| `bash workbench/docker/release_verify.sh --offline` | 0 | no unmet criteria; 5 pending (not RELEASE_READY) |
| `python -m compileall -q src workbench/…` | 0 | clean (repo source; the root-owned provisioned env under `artifacts/` is excluded) |
| `bash -n pc.sh` | 0 | clean |
| `npx tsc -b` · `npx vite build` | 0 | clean |
| `docker compose config --quiet` | 0 | clean |
| `git diff --check` | 0 | clean |

Every figure above is from the committed tree `17d3947`.

## N. INDEPENDENT REVIEW RESULTS

Five models, four perspectives, all read-only on a frozen candidate and all told to **disprove**
readiness: **gemini-audit** (runtime), **v41review/DeepSeek** (evidence), **qwen-audit** (science),
**v41audit** (release), plus the three-scout reconnaissance wave.

| Review | Verdict | Confirmed | Rejected |
|---|---|---|---|
| Runtime | 4 claims attacked | **3 P1 leaks** (→R8-07, R8-08) | remaining argv sites bounded by the 12,000-char job-tail cap; sentinel semantics preserved; subshell capture correct |
| Evidence | SOUND_WITH_RESERVATIONS | **P1 token/notes merge** (→R8-06), P2, P3 | 15 receipt shapes swept; nested/missing/malformed receipts refused; dev cert refused for every falsy variant |
| Science | **REJECTED_FINDING** | none P0 | all five drift allegations disproved with quoted code and measured digests |
| Release | VARIANT | 7 findings (→R8-01/05/09) | 12 allegations itemised as disproved |

## O. PC COMMANDS

```bash
git clone git@github.com:Le-Minh-Nhut/cir.git && cd cir
git fetch origin research/cir-failure-workbench
git checkout 17d3947
git rev-parse HEAD              # must print 17d39476c08d32c5c78cdd5d7dfb4eb7abbb9d55
./pc.sh                         # the one operator command
```

`./pc.sh` needs a FashionIQ dataset path, the three declared LIMN bundle members under the checkpoint
root, a `torch` + `open_clip` interpreter, and exactly one ledger key file. Exit codes: 0 by declared
capability · 3 aggregate reached but a declared observer exported nothing · 4 artifact unverified or
proof unavailable for an observer model · 5 the launcher could not read what the run recorded · 6
summary and index disagree · 7 invocation authenticated but **not** scientifically certified.

## P. EXTERNAL VALIDATION LIMITS

No GPU, no checkpoint loading, no inference, no Recall, no scientific reproduction. **Runtime
verification level 0 for every model including LIMN**; all LIMN claims are static or fixture-level.
The `differs from the paper` question remains open until `./pc.sh` runs on the RTX 5070 Ti.

## Q. FINAL VERDICT

# `PC_PILOT_HANDOFF_READY`

# `GPU_SCIENTIFIC_VALIDATION_PENDING`

# `SIXTEEN_MODEL_REPRODUCIBILITY: 1 runnable pending GPU · 1 one-fix-away · 2 need checkpoint mapping · 1 needs an upstream asset · 1 needs the authors · 10 need wrappers`

Zero known unresolved P0/P1/P2 defects in the audited target workflows. Zero confirmed regressions:
the one regression this pass introduced (`NameError` in the payload builder) was caught by the suite
and fixed before the commit. All applicable gates pass on the final tree.

No claim of mathematical certainty is made. The honest reading: the pilot path is ready for hardware,
the other fifteen are documented at the level they have actually reached, and the two reviewers who
were asked to break this pass's own work succeeded three times — which is the reason to believe the
rest of it.
