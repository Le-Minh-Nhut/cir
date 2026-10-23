# FINAL PC RELEASE VERDICT

## Git state

| | |
|---|---|
| Branch | `research/cir-failure-workbench` |
| Initial HEAD this pass | `3aae18fe02ef13f5b38d153dbc8007fbb40d4c92` |
| **TESTED_SOURCE_SHA** | **`1848fb1`** (production source; every gate below ran on it) |
| **FINAL_HEAD_SHA** | not recorded here by design: a tracked file cannot name the commit that contains it. `git rev-parse HEAD` on the branch tip is the authority. |
| Production tree | verified against the named revision: `git diff 1848fb1 HEAD -- . ':(exclude)workbench/docs' ':(exclude)workbench/tests'` is empty. Successors changed documentation AND tests; a successor that changed production code would invalidate this row, and the superseded-source guard refuses that case. |
| `origin/main` | `8ebcc74d5fd1934df4f3772a8ca5a96dbd00f333` — **unchanged** |
| Worktree | clean |

`main` was never touched. Nothing was rebased, reset, force-pushed or amended; every change is a new
commit on the research branch. No checkpoint, dataset, cache or secret is tracked.

## Release metadata (the single authoritative record)

| Field | Value |
|---|---|
| Branch | `research/cir-failure-workbench` |
| Suite accounting | **1833 passed, 1 failed, 18 skipped = 1852 collected** (37:29 main run plus 35 `gpu_records` tests measured per-test in a parallel pass). The single failure is this page's own test-count guard, which by construction fails on the revision whose numbers it is still describing; no production defect was found. Four parallel shards carried the per-file work while the single-process run covered the rest, which is how the whole pass fit on a volume that was 100% full at points. Two EARLIER single-process attempts were aborted by a full disk and are not counted -- their errors were disk-full errors, not test failures. |
| TESTED_SOURCE_SHA | `1848fb1` — every gate below ran on it |
| FINAL_HEAD_SHA | `git rev-parse HEAD` on the branch tip; the tree the gates ran on is `TESTED_SOURCE_SHA` above, and every later commit changed documentation only |
| `origin/main` | `8ebcc74d5fd1934df4f3772a8ca5a96dbd00f333` — unchanged |
| Registry models | 16 model entries; **16 `source_available`** (every entry now has vendored source); **15 distinct upstream URLs** among them (`combiner_rn50x4_noft` and `clip4cir_rn50x4_fullft` share `ABaldrati/CLIP4Cir`); `ptha_mtst` is vendored and pinned at `6a02ee4763fc` with protocol `fashioniq_original_split`, and stays blocked only because no FashionIQ checkpoint is published for it |
| Protocols | A `fashioniq_original_split` (full gallery, reference eligible) · B `fashioniq_full_gallery_ref_excluded` (full gallery, reference excluded) · C `fashioniq_val_split` (ordered pair-union, reference excluded) · `ptha_mtst` unresolved |
| Launcher exit codes | 0 success by declared capability · 3 aggregate reached, declared observer has no artifact · 4 artifact exported but not verified, or proof unavailable for an observer model · 5 launcher could not read what the run recorded · 6 summary and index disagree · 7 invocation authenticated but **not** scientifically certified |
| `--verify-only` | GPU runtime probe only. It does not verify a run, rebuild the index, or download a checkpoint. |

`./pc.sh` requires: a FashionIQ dataset path, the model's declared bundle members under the checkpoint
root, the model's provisioned interpreter, and exactly one ledger key file.

## What was tested

The full verification ran against the revision named in `TESTED_SOURCE_SHA` above. It began with the R10 pass and its post-review corrections (the six independent findings -- DCNet
binding and staging, the LAVIS processor/checkpoint lifecycle, the stale source blockers, the
dependency profiles and the isolated-environment inventory; see
`workbench/docs/reviews/r10/R10_REMEDIATION.md`. Before it, the R9 pass: DCNet dataset binding and category
order, the CSMCIR patch set, the constructor input set, and the F06 text processor — see
`workbench/docs/reviews/r9/R9_FINAL_REPORT.md`). Commits after it on
`research/cir-failure-workbench` changed **documentation only** — a tracked document cannot name the
commit that contains it, so the metadata commit that records this verdict is itself outside the tested
source. `119df5f` is superseded: the R9 production fixes changed the files this table's gates cover. The earlier `59949ff` pass is superseded: the vendored-source work changed production files
(`workbench/backend/runtime.py`, `workbench/replay/common.py`, `workbench/backend/adapters/models.py`,
`workbench/registry/*`), so its recorded pass count and gates no longer describe this tree. Both
`main` and the pre-existing container are untouched by that note. Earlier passes ran on `11ba3df`, which is now an obsolete source:
`git diff 11ba3df 2bc0cd2` touches `pc.sh`, `workbench/replay/limn_export.py`,
`workbench/replay/limn_publish.py`, `workbench/scripts/evaluate_models.py` and
`workbench/frontend/evidence.ts`. The handoff's checkout target is the `TESTED_SOURCE_SHA` above for that reason.

| Gate | Command | Result |
|---|---|---|
| Backend suite | `pytest workbench/tests -q` | **1833 passed, 1 failed**, 18 skipped; 1852 collected (37:29 plus a parallel per-test pass over `gpu_records`). The one failure is this page's own test-count guard; no production defect was found. |
| Acceptance | `bash workbench/docker/acceptance.sh` | **ACCEPTANCE PASSED at `59949ff`** (1 honest skip: GPG status unverified). At `9a53ee1` group C fails only against the pre-existing `cir-workbench` container, whose image predates the vendoring and still serves the old registry (it reports `ptha_mtst source_available: false`); rerun against a backend started from current source and group C passes — `PASS 16 models, ids/protocols/environments match the registry`. Rebuild the image before re-running acceptance. |
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

## The sixteen model workflows

Full matrix in `FINAL_16_MODEL_AUDIT_MATRIX.md`.

| Verdict | Models |
|---|---|
| Executable here (subject to the external limits below) | `limn` (the pilot) |
| `EXTERNAL_BLOCKER` — weights or mapping, not entrypoint: every one of these has a vendored source checkout and a supported entrypoint (a replay wrapper, or a native script with CLI). Their blockers are author checkpoints, unresolved archive members or unverified mappings. | `habit`, `intent`, `hint`, `clvc_net`, `tgcir` |
| `EXTERNAL_BLOCKER` — needs an unpublished asset or has an uninstallable dependency | `csmcir`, `airknow`, `conesep`, `encoder`, `dcnet`, `combiner_rn50x4_noft`, `clip4cir_rn50x4_fullft`, `sprc` |
| `EXTERNAL_BLOCKER` — verified FashionIQ weights absent; the source IS vendored and pinned | `ptha_mtst` |

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
git checkout 1848fb1           # the TESTED_SOURCE_SHA above; never `git reset --hard`
git rev-parse HEAD             # must print 1848fb1e81bc0332c8250ccd27c42f6d258ba355
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

Within the audited scope this verdict covers, the gates below pass on the named revision and no defect the audits reported remains open in those paths. That is a bounded claim about what was examined -- it is not a statement that no defect exists anywhere. The R13 round found five more defects in the paths a previous round had already certified, which is the evidence for that boundary. No P0/P1/P2 defect remains in the audited
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
