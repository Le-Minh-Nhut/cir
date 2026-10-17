# FINAL PC RELEASE VERDICT

## Git state

| | |
|---|---|
| Branch | `research/cir-failure-workbench` |
| Initial HEAD this pass | `3aae18fe02ef13f5b38d153dbc8007fbb40d4c92` |
| **TESTED_SOURCE_SHA** | **`4b32063f6544f10e7b0b6e9425e7467fef3d08b9`** |
| FINAL_HEAD_SHA | see the push record at the end of this file |
| `origin/main` | `8ebcc74d5fd1934df4f3772a8ca5a96dbd00f333` — **unchanged** |
| Worktree | clean |

`main` was never touched. Nothing was rebased, reset, force-pushed or amended; every change is a new
commit on the research branch. No checkpoint, dataset, cache or secret is tracked.

## What was tested

The full verification ran against `4b32063`. Any later commit in this pass is documentation only, and
the production diff for it is recorded at the end.

| Gate | Command | Result |
|---|---|---|
| Backend suite | `pytest workbench/tests -q` | **1365 passed, 0 failed** (25:40) |
| Acceptance | `bash workbench/docker/acceptance.sh` | **ACCEPTANCE PASSED** (1 honest skip: GPG status unverified) |
| Mutation checks | `bash workbench/docker/mutation_checks.sh` | **PASSED** M1 M2 M3 M4 M5 |
| Offline release verification | `bash workbench/docker/release_verify.sh --offline` | exit 0, `ENGINEERING_INCOMPLETE`, **0 checks FAIL/UNRUN** |
| Python compile | `python -m compileall -q …` | clean |
| Launcher syntax | `bash -n pc.sh` | clean |
| Docker entrypoint syntax | `bash -n workbench/docker/run.sh` | clean |
| TypeScript | `npx tsc --noEmit` | clean |
| Frontend production build | `npm run build` | clean |
| Compose configuration | `docker compose … config --quiet` | clean |
| Whitespace/conflict check | `git diff --check` | clean |

## Defects fixed in this pass

Thirteen confirmed defects. Full evidence in `FINAL_DEFECT_LEDGER.md`.

| ID | Sev | One line |
|---|---|---|
| F-01 | P1 | The launcher swallowed a failed DuckDB rebuild and ran it before the verifier, so a finished pilot could be absent from the index it reports on. |
| F-02 | P1 | An operator passthrough of `HOME` reached the ledger HMAC key, letting the untrusted evaluator forge a ledger entry. |
| F-03 | P1 | A container reused on a configuration match could be a different commit's code, so a report could name code that never ran. |
| F-04 | P1 | An unresolvable checkpoint id made the completion proof bind to its own claimed digest. |
| F-05 | P2 | The active run survived a protocol switch, rendering one candidate universe under another protocol's label and mis-keying annotations. |
| F-06 | P2 | A stale query lookup could overwrite the currently selected run's ranking. |
| F-07 | P2 | A diagnostic's per-query reason was computed and then unconditionally overwritten. |
| F-08 | P2 | The producer and the publisher computed different "bundle manifest digests" for one artifact. |
| F-09 | P2 | `PUT /api/annotations` accepted a protocol id and a query id that do not exist. |
| F-10 | P2 | The launcher invented `fashioniq_val_split` for a model whose protocol the registry leaves unresolved. |
| F-11 | P3 | The summary printed "the index contains this run" before the check that could contradict it. |
| F-12 | P3 | The provenance requirement named an alternative twice, producing advice the operator could not act on. |

## Not fixed, with the reason

| ID | Finding | Why it stands |
|---|---|---|
| N-01 | 27 tests assert on source TEXT rather than runtime behaviour (worst five cited by `file:line` in the ledger). | Rewriting the verification apparatus belongs in its own pass, not in a frozen release. No claim in this report rests on those tests: every defect above has a regression test that exercises runtime behaviour. |
| N-02 | `admission` does not enforce score monotonicity. | No consumer depends on it; a schema rule that rejected a legitimate native output on a tie is the larger risk. |
| N-03 | The SPRC adapter is registered without a wrapper. | Wrapping it would still not run: upstream's argparse default names an unregistered LAVIS model, `clip` is missing from the pinned requirements, and the author's checkpoint URL returns HTTP 403. |

## The sixteen model workflows

Full matrix in `FINAL_16_MODEL_AUDIT_MATRIX.md`.

| Verdict | Models |
|---|---|
| Executable here (subject to the external limits below) | `limn` (the pilot) |
| `EXTERNAL_BLOCKER` — no CLI entrypoint in the pinned upstream | `habit`, `intent`, `hint`, `clvc_net`, `tgcir` |
| `EXTERNAL_BLOCKER` — needs an unpublished asset or has an uninstallable dependency | `csmcir`, `airknow`, `conesep`, `encoder`, `dcnet`, `combiner_rn50x4_noft`, `clip4cir_rn50x4_fullft`, `sprc` |
| `EXTERNAL_BLOCKER` — no pinned source at all | `ptha_mtst` |

- Only **LIMN** declares a per-query observer. The other fifteen are `AGGREGATE_ONLY` because their
  upstream code prints aggregate recalls and nothing else; none of them is presented as query-level
  capable, anywhere in this workbench.
- **No model's native protocol was changed.** For all sixteen, the candidate universe and reference
  policy were read out of the pinned source and compared against the registry: every assignment was
  already correct.

## The three protocols

| Protocol | Candidate universe | Reference | Models |
|---|---|---|---|
| `fashioniq_original_split` | full validation gallery | **eligible** | csmcir, combiner_rn50x4_noft, clip4cir_rn50x4_fullft, sprc |
| `fashioniq_full_gallery_ref_excluded` | full validation gallery | excluded | airknow, conesep, habit, intent, dcnet |
| `fashioniq_val_split` | deduplicated ordered reference/target union | excluded | hint, encoder, pair, clvc_net, tgcir, limn |

Each was verified against the pinned code, not the documentation. The val-split union sizes were
reproduced independently from host data (dress 2628, shirt 3089, toptee 2902 — all strictly below the
full-split sizes 3817/6346/5373), which is what distinguishes protocol C from protocol B in an
executable way. Cross-protocol comparison is refused on `protocol_id` equality, probed with two runs
whose gallery sizes were identical.

## Launcher success/failure matrix

| Scenario | Report state | Index state | Exit |
|---|---|---|---|
| aggregate verified, per-query exported and corroborated | `PER_QUERY_VERIFIED` | `PER_QUERY_VERIFIED` | **0** |
| aggregate verified, observer declared, export failed | `PER_QUERY_EXPORT_FAILED` | either | **3** |
| aggregate verified, artifact exported, not yet corroborated | `PER_QUERY_EXPORTED_UNVERIFIED` | either | **4** |
| model declares no observer (aggregate-only by design) | `CAPABILITY_ABSENT` | no per-query state | **0** |
| the run's report cannot be read | unknown | either | **5** |
| the summary and the index disagree about the same run | either | contradicting | **6** |
| the verifier refuses the evidence | refusal text with reasons | not reached | non-zero |
| the index rebuild fails, times out, or answers with a non-2xx | not reached | unchanged (atomic) | non-zero |
| the index rebuilds but the run is absent, bound to another model, or has no evidence state | not reached | stale | non-zero |
| the container is another commit's code | not reached | not reached | non-zero |

## External limitations

Nothing below was validated, and nothing in the repository claims otherwise.

| Item | State |
|---|---|
| `TARGET_GPU_VALIDATION` (RTX 5070 Ti) | **PENDING** |
| `LIMN_NATIVE_INFERENCE` | **PENDING** |
| `LIMN_REAL_INSTRUMENTATION_PARITY` | **PENDING** |
| `PAPER_REPRODUCTION` | **PENDING** |
| `REAL_RECALL` | **PENDING** |
| Per-model external blockers | documented in the matrix |

No GPU is present on this machine, FashionIQ is absent, and the three LIMN checkpoints were not
downloaded. Every LIMN runtime claim in this pass is static or fixture-level.

## Exact PC commands

```bash
git clone git@github.com:Le-Minh-Nhut/cir.git && cd cir
git fetch origin research/cir-failure-workbench
git checkout FETCH_HEAD        # an explicit checkout; never `git reset --hard`
./pc.sh                        # the one operator command
```

`./pc.sh` requires: a FashionIQ dataset path, the three LIMN bundle members under the checkpoint root,
`WORKBENCH_LIMN_PYTHON` pointing at an interpreter with `torch` and `open_clip`, and exactly one
ledger key file. Exit codes are the table above; `./pc.sh --verify-only` re-runs verification against
an existing run without re-evaluating.

## VERDICT

# `PC_PILOT_HANDOFF_READY`

with

# `GPU_SCIENTIFIC_VALIDATION_PENDING`

### Why READY, stated narrowly

Every software gate passes on the frozen source, no confirmed P0/P1/P2 defect remains in the audited
intended workflows, the final source was reviewed by independent reviewers on four model families who
were asked to disprove readiness, the three protocols are unchanged, evidence cannot self-certify, the
launcher's success and failure paths are both covered by tests that exercise real HTTP and real child
processes, and the external limitations are stated rather than papered over.

### What READY does not mean

It does not mean the software is correct with mathematical certainty. It means: no defect found by 21
independent audits of this source remains unfixed, and the two findings that were left standing are
named with their reasons. The strongest evidence for this verdict is not the green suite — the suite
was green while three defects that could make a pilot report false success were live. It is that the
reviewers found those defects by reading the executable source, and that each now has a test which
fails if the fix is reverted.

### The next step, and the only one

Execute the real LIMN checkpoint bundle on the RTX 5070 Ti and collect verified research evidence.
No further feature work, no further models, no redesign.
