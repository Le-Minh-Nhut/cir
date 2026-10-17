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

### N-03 — `spread`/`sprc` adapter registered without a wrapper

* **Reported by**: the csmcir/sprc audit.
* **Why not fixed**: wrapping it would not make the model runnable, for three independent reasons the
  same audit established: upstream's argparse default names a LAVIS model that is not registered,
  `clip` is missing from the pinned `requirements.txt`, and the author's checkpoint URL returns HTTP
  403. The registry already records `BLOCKED_CHECKPOINT_MAPPING` and the adapter fails closed. An
  invented wrapper would be a command line nobody verified.

---

## Rejected findings

| Allegation | Why it is wrong |
|---|---|
| The LIMN observer's parity check compares 9-decimal values against native metrics rounded to 2 decimals, so it can never pass. | The native `LIMN/test.py` does **not** round: it computes `100 * hits / len(nn_result)` in full precision and merely prints with `:.2f`. Both sides are exact hit counts divided by the same population. Exercised over all 5 165 hit counts of the largest category: **0 of 5 165** produce a representation difference under `round(x, 9)`. |
| Rebuilding the index *before* the verifier runs breaks `PER_QUERY_VERIFIED` promotion. | `verify_run.verify` writes nothing: it prints a verdict. `index.verifier_verdicts` obtains the same verdict by calling the same function through `main._aggregate_reports` at rebuild time, and both inputs exist when the launcher reaches the summary. The real defect in that block was the swallowed `|| true`, which is F-01. |
| The receipt cache serves a stale certification. | Every one of the six cache tokens is a content digest of the thing the receipt is a claim about; mutation probes against all six re-keyed the cache correctly. |
| `data_kind == "experiment"` lets a run certify itself. | `admission`/`promote` refuse it; the frontend reads `evidence_status` from the index for its verified count. The dashboard card's "experiment runs" figure counts producer declarations **and labels them as such**; the verified figure beside it is separate. |
