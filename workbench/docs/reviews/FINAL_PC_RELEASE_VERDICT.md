# FINAL PC RELEASE VERDICT

## Observed host capacity (2026-10-24, R20)

| | |
|---|---|
| RAM | 16 GB total, 16 GB swap; the encoder replay's documented peak for ONE three-category construction is 11.87 GB (`evaluate_models.py:715-716`) |
| Aggregate evaluation | **completes and verifies** on this host: encoder, all three categories, `verify_run.py` `verified=true / is_research_evidence=true`. ATTRIBUTION: the run was executed at 14:16 against `0fc3563` + the then-uncommitted R20 working tree; the later `874ddcc` changed only stage-fingerprint classification (`pipeline.py`), never the replay/evaluation path that produced these metrics, so the result stands for the tested production tree as well — but the revision that ran it is `0fc3563` + the R20 tree, and that is what is recorded here |
| Per-query observer | **host-limited**: the observer child is killed with `-9` under the same ceiling when it is scheduled alongside the aggregate. The aggregate report for that invocation is complete and the failure is recorded as `INVOCATION_FAILED` (`return_code -9`) rather than as a pass |
| Disk | ~17 GB free at the time of that run; the ~11.9 GB replay cache fits, and `evaluate_models.py` removes it after the children finish |

This is a capacity observation about the machine, not a model or software defect: the same software
produced a verified aggregate result on the same host. A 32 GB host (or a serialized observer after
the aggregate) is the concrete unblock.

## Git state

| | |
|---|---|
| Branch | `research/cir-failure-workbench` |
| Initial HEAD this pass | `3aae18fe02ef13f5b38d153dbc8007fbb40d4c92` |
| **TESTED_SOURCE_SHA** | **`ab38ac7260e0788aa33ca8f1f61423dc3a1b5239`** — the production revision under test; documentation-only successors preserve this production tree. R25 suite counts below describe the pre-composition revision. R23 remains historical. |
| **FINAL_HEAD_SHA** | not recorded here by design: a tracked file cannot name the commit that contains it. `git rev-parse HEAD` on the branch tip is the authority. |
| Production tree | compare against the named revision with `git diff ab38ac7260e0788aa33ca8f1f61423dc3a1b5239 HEAD -- . ':(exclude)workbench/docs' ':(exclude)workbench/tests'`; the release guard requires byte-identical production source. |
| `origin/main` | `8ebcc74d5fd1934df4f3772a8ca5a96dbd00f333` — **unchanged** |
| Worktree | R25 changes committed; preserved untracked user files: `src/train_cirr.py`, `uv.lock` |

`main` was never touched. Nothing was rebased, reset, force-pushed or amended; every change is a new
commit on the research branch. No checkpoint, dataset, cache or secret is tracked.

## Release metadata (the single authoritative record)

| Field | Value |
|---|---|
| Branch | `research/cir-failure-workbench` |
| Suite accounting | **2060 passed, 5 failed**, 15 skipped, 0 errors = **2080 collected** (`workbench/tests`) in the full invocation on `a3f90403e79a31424498ef9dfb30f9534907acc9`, wall 1304.19 s. Including `tests/unit`: 2139 collected, 2119 passed, 5 failed, 15 skipped; unit scope 59 passed. Subsequent focused runs are separate evidence. |
| TESTED_SOURCE_SHA | `ab38ac7260e0788aa33ca8f1f61423dc3a1b5239` — the production revision named for current release accounting |
| Previous composition accounting | `d05149b9a71546166f5d5d38c813744c2a84c9e9`: 2029 collected, 2017 passed, 1 failed, 11 skipped. Retained as historical measurement. |
| FINAL_HEAD_SHA | `git rev-parse HEAD` on the branch tip; `TESTED_SOURCE_SHA` identifies the production tree under test. |
| `origin/main` | `8ebcc74d5fd1934df4f3772a8ca5a96dbd00f333` — unchanged |
| Registry models | 17 model entries; **17 `source_available`** (every entry has vendored source); **16 distinct upstream URLs** among them (`combiner_rn50x4_noft` and `clip4cir_rn50x4_fullft` share `ABaldrati/CLIP4Cir`); `ptha_mtst` is vendored and pinned at `6a02ee4763fc`, but its sole registered checkpoint is CIRR-trained for CIRR→FashionIQ transfer, not a FashionIQ reproduction. |
| Protocols | A `fashioniq_original_split` (full gallery, reference eligible) · B `fashioniq_full_gallery_ref_excluded` (full gallery, reference excluded) · C `fashioniq_val_split` (ordered pair-union, reference excluded) · `ptha_mtst` unresolved |
| Launcher exit codes | 0 success by declared capability · 3 aggregate reached, declared observer has no artifact · 4 artifact exported but not verified, or proof unavailable for an observer model · 5 launcher could not read what the run recorded · 6 summary and index disagree · 7 invocation authenticated but **not** scientifically certified |
| `--verify-only` | GPU runtime probe only. It does not verify a run, rebuild the index, or download a checkpoint. |

`./pc.sh` requires: a FashionIQ dataset path, the model's declared bundle members under the checkpoint
root, the model's provisioned interpreter, and exactly one ledger key file.

## What was tested

Full invocation on `a3f90403e79a31424498ef9dfb30f9534907acc9`: 2139 collected across workbench/tests and tests/unit; 2119 passed, 5 failed, 15 skipped, 0 errors, 1304.19 s. Workbench scope: 2080 collected, 2060 passed, 5 failed, 15 skipped, 0 errors. Unit scope: 59 passed. The later blocked-row deduplication and blank-HF export correction in `ab38ac72` is covered by a separate 123-passing CIRCUS/runtime run; the full suite was not rerun after that narrow change. Separate reruns do not turn the recorded full invocation green.

The raw failures are retained in the audit package. Two launcher fixtures exceeded 20 s under
parallel load; missing-input and Autopilot resume assertions also failed; the count guard saw the
old 2071-case metadata. Focused rechecks and the updated guard are recorded separately.
The final source differs from the full-run source only by the blocked-row CLI correction, blank-HF export normalization and their
stronger assertions; all 123 CIRCUS/runtime checks pass after that correction.
Torch 2.7.1+cpu / torchvision 0.22.1+cpu also passed all 59 unit tests in an isolated environment.
No target sm_120 GPU or complete per-model CUDA installation is claimed.

Earlier R23/R25/composition measurements are historical and do not certify this runtime profile.

| Gate | Command | Result |
|---|---|---|
| Backend suite | One full invocation on `a3f90403`; `-n 6 --dist=loadfile` with documented audit-host tool adaptations | **2060 passed, 5 failed**, 15 skipped, 0 errors = **2080 collected** for workbench/tests. Unit scope 59 passed. This is not a green full-suite claim. |
| Acceptance | `bash workbench/docker/acceptance.sh` | **ACCEPTANCE PASSED at `59949ff`** (1 honest skip: GPG status unverified). At `9a53ee1` group C fails only against the pre-existing `cir-workbench` container, whose image predates the vendoring and still serves the old registry (it reports `ptha_mtst source_available: false`); rerun against a backend started from current source and group C passes — `PASS 17 models, ids/protocols/environments match the registry`. Rebuild the image before re-running acceptance. |
| Mutation checks | `bash workbench/docker/mutation_checks.sh` | **PASSED** M1 M2 M3 M4 M5 |
| Offline release verification | `bash workbench/docker/release_verify.sh --offline` | exit 0, `ENGINEERING_INCOMPLETE`, **0 checks FAIL/UNRUN** |
| Python compile | `python -m compileall -q …` | clean |
| Vendored-source contract | `pytest workbench/tests/test_vendored_source_integrity.py` | **13 passed** — every delivered checkout verifies against its recorded pin and digest manifest |
| Vendored approved patch | `pytest workbench/tests/test_vendored_approved_patch.py` | **11 passed** — the registry's own CSMCIR patch is accepted; an edit after it, a wrong digest and an unrelated edit are refused |
| Fresh-clone delivery | `pytest workbench/tests/test_fresh_clone_delivery.py` | **10 passed** — a real clone from the remote contains every manifest path with matching bytes, no nested `.git`, no gitlinks |
| Wrapper parser coverage | `pytest workbench/tests/test_replay_output_parser.py` | **18 passed** — every wrapper model parses; partial, contradictory and foreign documents are refused |
| Launcher syntax | `bash -n pc.sh` | clean |
| Docker entrypoint syntax | `bash -n workbench/docker/run.sh` | clean |
| TypeScript | `npx tsc --noEmit` | clean |
| Frontend production build | `npm run build` | clean |
| Compose configuration | `docker compose … config --quiet` | clean |
| Whitespace/conflict check | `git diff --check` | clean |

## Defects fixed in this pass

Twelve confirmed defects in this pass, plus the A-series and F-series entries the ledger
records for later passes. Full evidence in `FINAL_DEFECT_LEDGER.md`.

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
| N-04 | The dataset cache's query order is not hash-bound to the caption file. | Per-position alignment is checked for every query, so a stale, truncated or reordered cache is caught; only a permutation applied consistently to both files escapes, which is a trust assumption about the operator's dataset root. Closing it belongs with dataset verification. |
| N-05 | Rank order under exact score ties is not pinned across NumPy builds. | The observer and the native code call the same `argsort` on the same objects within a run, so in-run parity is guaranteed; cross-build tie order is unobservable here and is a property of the pinned upstream ranking. Diverging from it would break the reproduction. |
| N-06 | One DCNet row publishes an aggregate mean without per-category rows. | The model has no evaluator, so nothing can produce a reproduced number for it; the paper gate's tolerance bounds any comparison; the registry claims nothing per-category. |

## The model workflows (17 registered; the historical audit covered 16)

Full matrix in `FINAL_16_MODEL_AUDIT_MATRIX.md`.

| Verdict | Models |
|---|---|
| Executable here (subject to the external limits below) | `limn` (the pilot) |
| `EXTERNAL_BLOCKER` — requested noise condition unavailable | `airknow` (20% requested; only 50%/80% registered) |
| `EXTERNAL_BLOCKER` — unverified checkpoint mapping and blocked evaluator entrypoint | `pair` (`pair_b1`/`pair_b2` mapping unverified) |
| `EXTERNAL_BLOCKER` — unresolved archive mapping, no source replay, missing ResNet-50 | `clvc_net` |
| `EXTERNAL_BLOCKER` — unresolved author run-directory mapping, data preparation and assets | `dcnet` |
| `EXTERNAL_BLOCKER` — unresolved checkpoint mapping, RN50x4 weights and compatible data | `combiner_rn50x4_noft` |
| `EXTERNAL_BLOCKER` — unresolved two-state bundle, checkpoint mapping, RN50x4 weights and compatible data | `clip4cir_rn50x4_fullft` |
| `EXTERNAL_BLOCKER` — blocked source entrypoint, archive mapping and asset provenance | `tgcir` |
| `EXTERNAL_BLOCKER` — unverified model-class/key mapping and assets | `sprc` |

- Only **LIMN** declares a per-query observer. The other sixteen are `AGGREGATE_ONLY` because their
  upstream code prints aggregate recalls and nothing else; none of them is presented as query-level
  capable, anywhere in this workbench.
- **No model's native protocol was changed.** For every model, the candidate universe and reference
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

This snapshot's external limitations are about the TARGET hardware, not about whether any GPU work
happened: the RTX 5070 Ti was never available, so every claim about it stays PENDING. Separately and
independently of this page, a real GPU evaluation DID run on the project's own host for one model --
`encoder`, all three FashionIQ categories on an RTX 3050 Laptop (sm 8.6), aggregate report + completion
proof + verifier output recorded, gate `PASS_METRICS_ONLY_MAPPING_UNVERIFIED`. That is sm86 evidence for
one model and does NOT certify sm120, and it does not make the LIMN claims above anything other than
static or fixture-level.

## Exact PC commands

```bash
git clone git@github.com:Le-Minh-Nhut/cir.git && cd cir
git fetch origin research/cir-failure-workbench
git checkout ab38ac72           # the fixed TESTED_SOURCE_SHA above; never `git reset --hard`
git rev-parse HEAD             # must print ab38ac7260e0788aa33ca8f1f61423dc3a1b5239
./pc.sh                        # the one operator command
```

`./pc.sh` requires: a FashionIQ dataset path, the three LIMN bundle members under the checkpoint root,
`WORKBENCH_LIMN_PYTHON` pointing at an interpreter with `torch` and `open_clip`, and exactly one
ledger key file. Exit codes are the table above.

`./pc.sh --verify-only` does **one** thing: it probes the model's GPU runtime inside the container and
exits. It does **not** verify a recorded run, does **not** rebuild the DuckDB index, and does **not**
download a checkpoint. Verifying an existing run and refreshing the index is what a plain `./pc.sh`
does — it reuses a corroborated run instead of re-evaluating.

## VERDICT

# `PC_PILOT_HANDOFF_READY`

with

# `GPU_SCIENTIFIC_VALIDATION_PENDING`

### Why READY, stated narrowly

Within the audited scope this verdict covers, at its recorded review time, the gates below passed on the named revision and no defect the audits reported remained open in those paths. That is a bounded claim about what was examined, not a statement that no defect existed anywhere. The R13 round found five more defects in paths a previous round had already certified, which is evidence for that boundary. Ordinary coding/review work in this round ran on `cx/gpt-6-luna` as resolved by the harness at `9router/cx/gpt-6-luna:max` (the operator's own `task.agentModelOverrides` maximum-effort setting for both worker tiers), the advisor role was left unchanged, and no Gemini model was invoked. The three protocols are unchanged, evidence cannot self-certify, the launcher's success and failure paths are covered by tests exercising real HTTP and real child processes, and external limitations are stated rather than papered over.

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
