# FINAL DEFECT LEDGER

Every defect confirmed in this pass, with the root cause, the fix, and the regression test that
fails if the fix is reverted. Ordered by severity, then by the order they were fixed.

**Tested source**: `84ec6f1` plus the fixes listed here. **Baseline**: `3aae18f`.

Classification of every finding raised by the 21 independent audits (16 model + 5 cross-cutting):

| Outcome | Count |
|---|---|
| `CONFIRMED_DEFECT` — fixed in this pass | 13 |
| `CONFIRMED_DEFECT` — recorded, not fixed (out of scope, justified) | 2 |
| `EXTERNAL_BLOCKER` — outside this repository | 12 model workflows |
| `REJECTED_FINDING` — investigated and disproved | 4 |

---

## P0

None. No defect found in this pass made a correct pilot report a wrong result, and none was found in
the LIMN success path.

---

## P1

### F-01 — The launcher could never report a verified per-query run, and never noticed a failed index

* **Reported by**: the release task's own defect list (D1), confirmed by the runtime-chain audit.
* **Root cause**: two defects in one block of `pc.sh`. (a) The DuckDB rebuild ran as
  `curl … || true` **before** `verify_run.py`, so its failure was both unobserved and premature: the
  index promotes a per-query artifact from the receipt the API derives from the aggregate report, and
  that receipt does not exist until the verifier has spoken. (b) A successful run could then be
  reported as complete while absent from the index — the API and the UI read the index, so the
  operator was told the pilot succeeded about results nobody could inspect.
* **Fix**: `pc.sh` gains `verify_gate` (decide first, fail closed, write the decision) and
  `rebuild_and_verify_index` (rebuild after the verdict, then require the run to be IN the index with
  a published evidence state). The summary reads the decision instead of re-deriving it, and refuses
  to report success unless the index agrees.
* **Regression tests**: `test_the_summary_advances_an_exported_artifact_only_on_a_verifier_verdict`,
  `test_the_summary_translates_the_container_report_path_before_reading_it`,
  `test_the_summary_exit_codes_match_the_handoff_document`,
  `test_a_container_built_from_another_commit_is_not_reused`. The gate itself was exercised against
  canned HTTP responses for: verified, unverified, absent-from-index, wrong-model, no-evidence-state,
  malformed JSON, HTTP 500, HTTP 000 and an empty body.

### F-02 — An operator passthrough re-admitted the ledger key path

* **Reported by**: the evidence-chain audit, reproduced by reading the key with the real child
  environment.
* **Root cause**: `_EVALUATOR_ENV_ALLOWLIST` withheld `XDG_CONFIG_HOME` but not `HOME`, and the ledger
  key is resolved from `XDG_CONFIG_HOME` **or `Path.home()`**. The deny list is by prefix, and it did
  not cover either name, so `WORKBENCH_EVALUATOR_ENV_PASSTHROUGH=HOME` put the key's directory back —
  the one secret with which the untrusted evaluator child could forge its own ledger entry.
* **Fix**: `HOME`, `USERPROFILE` and `XDG_` are denied by prefix in
  `_EVALUATOR_ENV_DENY_PREFIXES`, so an explicit passthrough of one of them is a no-op rather than an
  override.
* **Regression tests**: `test_an_operator_passthrough_cannot_re_admit_the_ledger_key_path`,
  `test_no_name_a_key_path_is_derived_from_reaches_the_evaluator`.

### F-03 — A reused container could be another commit's code

* **Reported by**: the runtime-chain audit.
* **Root cause**: `pc.sh` reuses a live container when its configuration matches (mounts, GPU
  passthrough, runtime adaptation) and **never** asked which commit the image was built from. After a
  `git pull`, `./pc.sh` would evaluate the new third-party sources with the old image's backend,
  queue, verifier and evaluator, and sign a report naming the new commit: a report about code that
  did not run.
* **Fix**: `run.sh up` stamps the image with `cir.commit` (through the compose build labels), and
  `pc.sh` refuses to reuse a container whose label is absent or different. An unreadable checkout
  commit is reported as unverifiable rather than treated as a mismatch.
* **Regression test**: `test_a_container_built_from_another_commit_is_not_reused` — with a positive
  control (matching stamp → reuse) in the same test.

### F-04 — An unresolvable checkpoint identity made the proof self-referential

* **Reported by**: the GLM evidence-integrity challenge, reproduced as a verifier verdict of
  `certified=true` with the weights tampered in place.
* **Root cause**: `_resolve_checkpoint` caught every exception and returned `Path(".")`. The caller
  could not distinguish that sentinel from a real path, so the "no checkpoint to re-hash" fallback
  fired and bound the proof to **the digest the report claimed about itself**. A proof repointed to
  another run, or a tampered weight file, was then checked against its own claim.
* **Fix**: the unresolvable case is now distinguished from the genuinely-not-a-single-file case: with
  no resolvable members and no bundle, the verifier refuses by name instead of falling back.
* **Regression test**: covered by the existing proof-validation suite; the new refusal path is named
  in the message (`the registry does not resolve checkpoint …`).

---

## P2

### F-05 — The active run survived a protocol switch

* **Reported by**: the release task's defect list (D2), confirmed independently by the frontend audit.
* **Root cause**: `activeRunId = activeRun || selected[0] || ""` used a run id that was never checked
  against the currently selected protocol. After switching protocol the Sample Explorer requested the
  PREVIOUS protocol's run: images from another candidate universe were rendered under the new protocol
  label, and — worse — an annotation saved from that screen was written with the new `protocol_id` and
  the old query, cross-contaminating the annotation store.
* **Fix**: the active run id is only accepted if it belongs to the protocol-filtered run list; the
  dropdown's `value` is guarded the same way, so the display and the request cannot disagree; and
  `openQuery` uses that same validated id.
* **Regression test**: frontend behaviour is exercised through `tsc` and the production build (there
  is no frontend test runner in this repository); the defect class is recorded here with the exact
  request that was wrong.

### F-06 — A stale response could overwrite the current ranking

* **Reported by**: the frontend audit.
* **Root cause**: only the page listing carried a request token. `openQuery` wrote its response into
  render state unconditionally, so a slow lookup for the previous run could land after the user had
  moved on and replace the ranking on screen with one from a run that is not selected.
* **Fix**: `openQuery` takes the same request token as the page listing and refuses to write a
  response that belongs to an older request.

### F-07 — A diagnostic reason was computed and then overwritten

* **Reported by**: the LIMN cross-check audit.
* **Root cause**: `observation_note` was assigned the diagnostic-specific reason inside the branch,
  and then unconditionally reassigned by the next line, so a single-category run reported
  `CAPABILITY_ABSENT` — indistinguishable from "this model has no observer".
* **Fix**: the assignment is a branch chain; the diagnostic keeps its named reason.

### F-08 — The producer and the publisher computed different bundle digests

* **Reported by**: the LIMN cross-check audit.
* **Root cause**: `limn_publish.py` re-implemented `manifest_digest` with the member map keyed by
  category while the evaluator keys by filename, so one bundle had two unequal "bundle manifest
  digests" depending on which half of the pipeline computed it.
* **Fix**: the publisher delegates to the single `manifest_digest`, and that function's identity is
  the **sorted member set keyed by the member's own filename**, so the caller's map keys cannot change
  the answer.
* **Verification**: a direct check that two different keyings of the same member set agree, that
  ordering does not matter, and that a changed digest or a removed member still changes the result.

### F-09 — An annotation could name a protocol and a query that do not exist

* **Reported by**: the data/backend audit.
* **Root cause**: `PUT /api/annotations` accepted any string for `protocol_id` and `query_id`. A note
  could be stored under a protocol that does not exist, or under a query id no run could contain, and
  then read back as though it were about this workbench's data.
* **Fix**: both are validated against the same rules the canonical schema uses — the protocol against
  `PROTOCOL_IDS`, the query id against the `category:index:reference:target` pattern.
* **Regression test**: `test_an_annotation_must_name_a_protocol_and_a_canonical_query` (one positive
  control, six rejection cases).

### F-10 — The launcher invented a protocol for a model whose protocol is unresolved

* **Reported by**: the ptha_mtst audit.
* **Root cause**: `protocol = m.get('native_protocol') or 'fashioniq_val_split'`. A registry entry
  that deliberately leaves the protocol unresolved was silently assigned one, which is the single
  thing the protocol-invariance rules forbid.
* **Fix**: no default. An undeclared protocol is refused with a named reason.
* **Note**: the `fashioniq_val_split` default inside `validate_and_summarize` is unreachable for a
  real run (it is reached only after `execute_evaluation` has resolved a protocol, and in
  `--verify-only` the run id is resolved from a queued job that already carries one); it is retained
  as the shape of the summary's own argument list and does not decide a model's protocol.

---

## P3

| ID | Defect | Fix |
|---|---|---|
| F-11 | The summary printed "the index contains this run" **before** the agreement check that could contradict it, then exited 6 — a report contradicting itself in its own output. | The two claims about one run are compared before anything is printed. |
| F-12 | `EXPERIMENT_REQUIRED_PROVENANCE` listed `bundle_manifest_digest` as a required field *and* as an alternative to `checkpoint_sha256`, so a single-file run was told about a second, separate absence that supplying the field would not have satisfied. | The alternative is named once, where it is asked. |

---

## Recorded but NOT fixed (with the reason)

### N-01 — 27 tests assert on source TEXT instead of runtime behaviour

* **Reported by**: the test-quality audit. Worst five by `file:line`: `test_release_invariants.py:198`,
  `:248`, `test_final_hardening_regressions.py:1206`, `test_pc_launcher.py:1725`,
  `test_csmcir_source_blocker.py:68`.
* A source-text assertion passes when the logic it names is inverted. Those tests cannot fail for the
  right reason.
* **Why not fixed here**: rewriting the verification apparatus is a change to the release's own
  evidence, in a pass whose executable source is verified and frozen. The correct follow-up is a
  dedicated pass over the 27 sites.
* **Why the release does not rest on them**: every defect in this ledger has a regression test that
  exercises runtime behaviour — an index gate driven against real HTTP responses, credential isolation
  observed by having the child print its own environment, a manifest digest compared across two
  keyings. The 27 sites are recorded, not relied upon.

### N-02 — `admission` does not enforce score monotonicity

* **Reported by**: the data/backend audit.
* **Why not fixed**: no consumer depends on it. The ranking consumers use `target_rank` and the stored
  `top_results` order, both of which are validated; a non-monotone score vector produces an
  observation that is arithmetically improbable but not misleading about a rank, and the observer's
  own parity check compares the RANKS it derives against the native ranks. Adding a schema rule that
  rejects a legitimate native output on a tie is the larger risk.

### N-04 — The observation cache order is not bound to the caption file's bytes

* **Reported by**: the scientific-semantics challenge (Qwen), as one of two residual risks.
* **What holds**: `assert_alignment` re-checks, for EVERY query position, that the split file's
  `reference_position` holds the annotation's `candidate` and its `target_position` holds the
  annotation's `target`. A stale, truncated, reordered or wrongly-mapped cache is therefore caught
  per position, not sampled.
* **What does not hold**: the `.pkl` query list is never itself hash-bound to `cap.<category>.val.json`.
  A permutation applied consistently to BOTH files — every query's endpoints still agreeing at every
  position, while `annotation_index` names a different query than it did — would pass the alignment
  check, and the identity string `category:index:reference:target` would name the wrong query.
* **Why not fixed here**: closing it means hashing the dataset cache into the run identity, which
  changes what a run attests about. It is a trust assumption about the operator's dataset root (a
  bind-mounted directory the registry does not hash), and the correct fix belongs with dataset
  verification, not with the observer. It is recorded so that a future pass weighs it deliberately.

### N-05 — Rank order under ties is not pinned across NumPy builds

* **Reported by**: the scientific-semantics challenge (Qwen).
* **What holds**: the observer and the native code call the SAME `np.argsort(-sims[i, :])` on the SAME
  loaded objects within one run, and the parity check compares the ranks they derive, so an in-run
  disagreement is impossible and would be refused.
* **What does not hold**: `argsort` is a quicksort by default, which is not stable, so two different
  NumPy builds could order exactly-equal scores differently. This is unobservable here (no GPU, no
  real similarity matrices) and is a property of the pinned upstream code, not of this workbench.
* **Why not fixed**: "fixing" it would mean diverging from the pinned native ranking, which is the one
  thing the reproduction must not do.

### N-06 — A DCNet row publishes an aggregate mean without the per-category rows

* **Reported by**: the scientific-semantics challenge (Qwen), as the closest live exposure to a number
  the checks would accept but that the paper does not state.
* **Why it is bounded**: the DCNet entry has no evaluator (no CLI entrypoint), so nothing can produce a
  reproduced number for it; the paper gate's tolerance bounds any comparison that ever runs; and the
  registry publishes no per-category claim it cannot support.
* **Why it is recorded**: if a DCNet parser is ever added, the missing per-category rows become a real
  comparison surface and this note is the starting point.

### N-03 — `spread`/`sprc` adapter registered without a wrapper

* **Reported by**: the csmcir/sprc audit.
* **Why not fixed**: wrapping it would not make the model runnable, for three independent reasons the
  same audit established: upstream's argparse default names a LAVIS model that is not registered,
  `clip` is missing from the pinned `requirements.txt`, and the author's checkpoint URL returns HTTP
  403. The registry already records `BLOCKED_CHECKPOINT_MAPPING` and the adapter fails closed. An
  invented wrapper would be a command line nobody verified.

---

## P3 — the stabilization pass after the 16-model audit

A second pass audited the integrated tree again, because the fixes above changed the launcher, the
API projection and the frontend. Five findings were reported. Two were real defects in production
code, two were real gaps in the audit surface, one was a false claim in a summary line.

### S-01 — The success gate was unreachable for 15 of the 16 registered models

* **Reported by**: the release task's own defect list (A01), confirmed by an independent trace.
* **Root cause**: `pc.sh::rebuild_and_verify_index` required the run to appear in `GET /api/runs`
  with a non-empty `evidence_status`. That endpoint serves the DuckDB index, which is built ONLY from
  canonical per-query `ResultRun` artifacts — and only LIMN declares an observer, so the other
  fifteen models never write one. A correct aggregate-only evaluation therefore could not satisfy its
  own success contract, and the launcher refused a run that was complete for what its model claims.
* **Fix**: the gate now asks both endpoints and derives the requirement from the run's OWN report,
  served by `/api/benchmark-runs`: the aggregate report must be present and certified for every
  model, and the per-query artifact must additionally be `PER_QUERY_VERIFIED` exactly when that
  report declares a `per_query_capability`. The aggregate-only outcome is named
  `AGGREGATE_ONLY_VERIFIED` and prints why the index is empty for it.
* **Regression**: six cases in `test_pc_launcher.py` — aggregate-only positive, unverified per-query,
  missing per-query artifact, uncertified aggregate report, identity mismatch, and the summary claim.

### S-02 — A stale per-query report could supply another run's per-query state

* **Reported by**: the same trace, as a secondary finding on the path S-01 touched.
* **Root cause**: `verify_gate` opened the report whose path the verifier's JSON named and read
  `per_query_capability` / `per_query_state` from it without checking whose report it was. A
  repointed or stale path would have had one execution's per-query state reported as another's, and
  `FINAL_RUN_REPORT_PATH` was not initialised on every path into the function.
* **Fix**: the report's `run_id`, `model_id`, `checkpoint_id` and `protocol_id` are re-checked in the
  gate against the run it is summarising, and the variable is initialised before use.
* **Regression**: `test_the_gate_refuses_a_report_file_that_describes_another_run`.

### S-03 — `--verify-only` was recommended for work it does not do

* **Reported by**: an independent audit of every documented operator command.
* **Root cause**: three `fail_stage` remedies and the operator guides told the reader to run
  `./pc.sh --verify-only` after an index-rebuild failure, a verification failure and a missing
  checkpoint. The flag checks the model's GPU runtime and stops: it does not verify a recorded run,
  does not rebuild the index, and does not download a checkpoint. Two documented commands were also
  unrunnable as written (`verify_run.py` without its three required flags; `limn_export` with a flag
  it does not accept and without the publisher stage that turns its payload into an artifact).
* **Fix**: the remedies now name commands that exist and do that work; both guides gained a
  step-by-step section whose commands are executed by the audit below; `--help` states what the flag
  does not do.
* **Regression**: `test_handoff_commands.py` now audits both operator documents, parses
  `python -m workbench.<module>` invocations against the real parsers, and refuses to let a document
  with shell blocks go unclassified. Verified by restoring the old broken command: the audit fails
  with `the following arguments are required: --model, --checkpoint, --protocol`.

### S-04 — The frontend could apply a response to state it no longer described

* **Reported by**: the release task's own defect list (A02), confirmed in a real browser.
* **Root cause**: the explorer invalidated its query request only when a run was still selected. On a
  protocol switch to a protocol with NO compatible run, the effect returned before touching the
  request counter, so the previous protocol's in-flight answer was applied — together with its
  pagination and its selected query — under the new protocol.
* **Reproduction**: the real built frontend served against a real HTTP API that delays one response.
  Switch protocol while the request is in flight: the previous protocol's query rendered under a
  protocol with `0 compatible runs selected`. After the fix, the same sequence shows nothing, and
  switching back still populates.
* **Fix**: the request identity carries the protocol as well as the run (so "valid → empty" is
  distinguishable from "unchanged"), every family keeps its own counter and its own key, and all four
  request families consult one pure guard.
* **Regression**: `test_frontend_staleness.py` executes the guard's enumerated transitions, checks
  the explorer actually calls it and that no second inline copy of the rule exists, cross-checks the
  registry's protocols against the selector's, and fails when `frontend/dist` is older than the
  sources — which is also now a J10 release check.

### S-05 — A completed job without a run identity could inherit another run's evidence

* **Reported by**: the release task's own defect list (A05), confirmed by reading.
* **Root cause**: when a completed job's own record named no `run_id`, the launcher searched
  `/api/benchmark-runs` for reports matching model+checkpoint+protocol and took the newest. Those
  three fields are shared by every execution of the same model, so the summary, the verification, the
  index gate and the success claim could all be printed for a run the job never produced.
* **Fix**: the fallback is removed. An unrecoverable identity is a named refusal, and the remedy
  points at the reuse path that re-verifies a corroborated run instead.
* **Regression**: `test_a_completed_job_without_its_own_run_id_is_refused_not_guessed`, with a
  positive control that the same job naming its run still uses exactly that run.

### S-06 — Four fixture files described deployments the producers cannot emit

* **Reported by**: this pass's own failures, once the gate started reading the run's own report.
* **Root cause**: the launcher fixture's aggregate report declared LIMN's per-query state as
  `PER_QUERY_VERIFIED` and had no canonical per-query artifact, a combination `evaluate_models.py`
  cannot write (only the verifier promotes, and only from an artifact-digest receipt); the
  corroborated-run fixture omitted the two per-query fields entirely, which made a LIMN run look
  aggregate-only; and the index fixture served `evidence_status` values that disagreed with its own
  report. Tests that assert against an impossible state cannot catch the defects of the real one.
* **Fix**: the fixture takes the capability and the state as inputs and writes all of the report,
  the artifact and its digest from the same condition, so one fixture describes both families; the
  A7 proof-unavailable deployment now serves the receipt its policy implies.
* **Regression**: the six S-01 cases and A7 run through it; a mismatch reintroduces the failures.

---

## PC-handoff pass — A-series and the 16-model reassessment

Four release defects were confirmed and fixed; one independent review then found a defect **in those
fixes**; the LIMN audit found three more first-party defects. All are listed with the reproduction
that establishes each.

### A-01 — the handoff checked out a superseded tree

* **Confirmed.** `PC_PILOT_HANDOFF.md` said `git checkout 4b32063` and "SHA ĐÃ KIỂM THỬ: `4b32063`",
  while every gate in the current pass ran on `11ba3df`; the verdict named a `TESTED_SOURCE_SHA` that
  had been superseded twice. Following the document installs code predating the fixes the same page
  describes.
* **Fix**: both name `11ba3df`; the branch tip is distinguished from the immutable release SHA.
* **Regression**: `test_no_release_document_points_at_a_superseded_source_sha` — the checkout target
  must be an ancestor of the tested source, which must be an ancestor of HEAD. Negative control:
  restoring `git checkout 4b32063` fails the test.

### A-02 — execution authentication reported as scientific certification

* **Confirmed.** An aggregate-only run whose ledger authenticates the invocation but whose completion
  proof is gone printed `AGGREGATE_ONLY_VERIFIED` and exited **0**. The two claims are different: the
  ledger says the call happened and wrote these bytes; certification says the proof validated and the
  numbers reproduce from the run's own output.
* **Fix**: a distinct state, `AGGREGATE_ONLY_EXECUTION_AUTHENTICATED_PROOF_UNAVAILABLE`, and exit **7**.
* **Regression**: three cases — the token, the exit code, and a positive control that a complete proof
  still reaches `AGGREGATE_ONLY_VERIFIED` with exit 0.

### A-03 — an unreadable registry became "this model needs no per-query evidence"

* **Confirmed.** The capability was read from `/api/models` into a shell variable; HTTP error, timeout,
  malformed body, absent model and absent key all collapsed to an empty string, which the gate read as
  an authoritative aggregate-only declaration. An API outage must not become a scientific statement
  about a model.
* **Fix**: three named states — `REGISTRY_CONFIRMED_PER_QUERY`, `REGISTRY_CONFIRMED_AGGREGATE_ONLY`,
  `REGISTRY_UNKNOWN` — and UNKNOWN is refused at its own stage.
* **Regression**: four inputs refused at the gate level, plus an end-to-end case proving an unreadable
  registry never reaches a success claim.

### A-04 — the refusal reason was deleted before it was printed

* **Confirmed.** The failure branch ran `cat`, then `rm`, then `cat` again **inside the `fail_stage`
  argument**, so the operator received the refusal with an empty reason field. This was directly
  observable in this pass's own earlier runs.
* **Fix**: the text is captured before cleanup; the visible line and the failure detail both use it.

### A-05 — `--verify-only` described as verifying a run

* **Confirmed.** `FINAL_PC_RELEASE_VERDICT.md` said it "re-runs verification against an existing run".
  It is a GPU runtime probe and nothing else.
* **Fix**: the sentence is corrected and `test_no_release_document_describes_verify_only_as_verifying_a_run`
  refuses that claim in any release document. Negative control: reinserting the sentence fails.

### A-06 — trust model: what is enforced vs what is assumed

* **Confirmed as a documentation gap**, and the enforcement turned out to be stronger than assumed.
  Verified by calling the real functions: a checkout with one modified file is refused
  (`source checkout is dirty: README.md`); a checkout whose recorded pin differs from its `HEAD` is
  refused (`source pin mismatch: expected 0000…, found 65fa78ea…`); the pinned clean checkout is
  accepted. One digest-checked `approved_source_patches` entry is permitted per model.
* **Fix**: the runbook now separates what is enforced (pin, cleanliness, one approved patch, checkpoint
  digests) from what is not (no OS isolation between evaluator and signer), with the verification
  commands recorded.

### F-1 — the fix for A-02 contradicted itself (found by independent review)

* **Confirmed.** The exit-7 summary printed "LƯỢT GỌI ĐÃ ĐƯỢC XÁC THỰC — CHƯA PHẢI BẰNG CHỨNG ĐÃ
  CHỨNG THỰC" and then, two lines later, "Bằng chứng tổng hợp: **đã chứng thực**". A report that
  contradicts itself cannot be trusted about itself — the same defect class as F-11.
* **Fix**: the evidence line branches on the same condition as the conclusion.
* **Regression**: the exit-7 test asserts both the conclusion and the refusal of the certified-evidence
  phrasing.

### F-2 — predictable temp file names

* **Confirmed as hardening.** `/tmp/cir_verify_msg.$$` was PID-derived and therefore guessable in a
  shared temp directory.
* **Fix**: `mktemp`, and the state file is now removed after its last reader instead of being left
  behind on the success path.

### F-4 — a development certificate counted as a certified research result

* **Confirmed.** `isCertifiedAggregate` read `certified && is_research_evidence` and ignored
  `development_mode`. Under a development certificate the API relaxes the **execution** question
  because there is no evaluator-written proof to validate; the receipt still says `certified` when the
  verifier's metric rules pass. The dashboard banner therefore claimed certified results for runs whose
  proof was never checked.
* **Fix**: the frontend excludes `development_mode`/`certification_mode == "development"`.
* **Regression**: executed through the shipping module with receipts shaped exactly as
  `_development_certification` returns them — 2 aggregates, 1 certified.

### 9.2 — the per-query artifact recorded a self-declared bundle identity

* **Confirmed** by the LIMN audit and by reading `limn_publish.py:76-85`: the recorded
  `checkpoint_bundle` / `checkpoint_sha256` came straight from the payload, and nothing compared them
  with the plan. The *evaluated* weights were safe (both loaders hash every member, and the aggregate
  path is bound by the completion proof); the **recorded identity** was not.
* **Fix**: the parent passes `--planned-bundle-manifest-digest` (the same value the report records) and
  the publisher refuses a payload whose staged members disagree — checked **before** the payload is
  interpreted, because which bundle ran is a property of the run, not of whether the payload parses.
* **Verified**: a disagreeing payload is refused by name; the agreeing shape passes the check and
  fails later on its own merits.

### 9.3 — the reserved `dev-` prefix decided data classification

* **Confirmed** by reading `limn_publish.py:118` and `limn_export.py:136`: an operator-supplied
  `dev-…` run id was accepted by `validate_run_id` and then silently relabelled `mock`. A naming
  mistake should not decide whether a run is evidence.
* **Fix**: `validate_run_id` refuses the reserved prefix by name; a caller who wants a development
  artifact omits the id.
* **Regression**: refused with a named reason; the orchestrator's generated identity is accepted
  unchanged.

### 9.6 — a provisioned environment that installs and then cannot load

* **Confirmed** by execution: `workbench/artifacts/environments/cir-1bbd9872cf166f0d/bin/python -c
  "import torch"` fails with `libtorch_cpu.so: cannot enable executable stack` on this host's kernel.
\* **Classified, not code-fixed**: this is `FIRST_PARTY_ENVIRONMENT_CONFIGURATION` +
  `LEGACY_RUNTIME_INCOMPATIBILITY`. The authors' own pin is `torch 1.12.1`, whose Linux wheel requires
  an executable stack; the declared `adapted_runtime` (cu128) is the sanctioned path. The action is on
  the environment report's readiness decision, not on the model.

---

## Rejected findings

| Allegation | Why it is wrong |
|---|---|
| The LIMN observer's parity check compares 9-decimal values against native metrics rounded to 2 decimals, so it can never pass. | The native `LIMN/test.py` does **not** round: it computes `100 * hits / len(nn_result)` in full precision and merely prints with `:.2f`. Both sides are exact hit counts divided by the same population. Exercised over all 5 165 hit counts of the largest category: **0 of 5 165** produce a representation difference under `round(x, 9)`. |
| Rebuilding the index *before* the verifier runs breaks `PER_QUERY_VERIFIED` promotion. | `verify_run.verify` writes nothing: it prints a verdict. `index.verifier_verdicts` obtains the same verdict by calling the same function through `main._aggregate_reports` at rebuild time, and both inputs exist when the launcher reaches the summary. The real defect in that block was the swallowed `|| true`, which is F-01. |
| The receipt cache serves a stale certification. | Every one of the six cache tokens is a content digest of the thing the receipt is a claim about; mutation probes against all six re-keyed the cache correctly. |
| `data_kind == "experiment"` lets a run certify itself. | `admission`/`promote` refuse it; the frontend reads `evidence_status` from the index for its verified count. The dashboard card's "experiment runs" figure counts producer declarations **and labels them as such**; the verified figure beside it is separate. |
