# FINAL ADVERSARIAL REVIEW

What independent reviewers were asked, what they found, what was disproved, and — most importantly —
what the reviewers got from the code rather than from the documentation.

**Tested source at the time of this review**: `84ec6f1` + the fixes in `FINAL_DEFECT_LEDGER.md`.
Superseded; the current release metadata is `FINAL_PC_RELEASE_VERDICT.md`.
**Baseline before this pass**: `3aae18f`

## 1. Current-round attribution

The worker model was `cx/gpt-6-luna`, resolved by the harness as `9router/cx/gpt-6-luna:max` (the operator's own `task.agentModelOverrides` maximum-effort setting); the advisor role was unchanged, and no Gemini model was invoked. Route note (added later): the current harness configuration pins all worker overrides to a different route, so this historical attribution can no longer be re-established from live configuration; it is retained as the record of the round that wrote this document.

## 2. The method every reviewer was given

1. **Pinned source first.** All 15 upstream repositories were cloned into `/tmp/upstream/<name>` at
   the exact `upstream_commit_sha` in the registry, and each clone's `git rev-parse HEAD` was compared
   against that pin before any audit began. An audit that reasons about "the current source" of a
   project whose evaluation code has since changed is worthless.
2. **`file:line` or a command's observed output.** Claims that were not observed are marked
   `[INFERENCE]` by the reviewer.
3. **Disprove readiness.** Reviewers were told their job was to construct a false PASS: what exact
   input or state makes the workbench report success when it should not?
4. **Read the callers.** Before alleging a defect in a function, grep its call sites. Several
   allegations died here — a guard that already exists elsewhere is not a finding.
5. **Never turn UNKNOWN into PASS.** No GPU, no FashionIQ dataset, no multi-GB checkpoints on this
   machine: those are `EXTERNAL_BLOCKER`, not failures and not passes.
6. **Docs are hypotheses.** Every existing audit document in `workbench/docs/` was to be treated as a
   claim to be checked, never as evidence.

## 3. Disagreements, and how each was settled

Independent reviewers did not merely agree. Three substantive disagreements arose, and each was
resolved with an executable test rather than by seniority.

### 3.1 The LIMN parity precision claim — REJECTED by experiment

One model audit reported a P1: the observer's cross-check compares values at 9 decimal places while
the native code rounds to 2, so a real run would fail parity and block publication.

The cross-check audit disagreed. Settled by running it: the pinned `LIMN/test.py` computes
`100 * hits / len(nn_result)` in full precision and rounds **only when printing** (`:.2f`); the
observer recomputes the same expression. Over the largest category's 5 165 possible hit counts,
**0 diverged** under `round(x, 9)`. The claim was false, and the false positive would have cost a
release-blocking scare.

### 3.2 Rebuild-before-verify — REJECTED, but it hid a real defect

The same cross-check audit rejected the premise that rebuilding the index before `verify_run.py` breaks
`PER_QUERY_VERIFIED` promotion: `verify_run.verify` performs **zero writes**, and the index obtains the
same verdict by calling the same function through `main._aggregate_reports` at rebuild time.

That rejection is correct, and it is what focused attention on the actual defect in the same block:
`curl … || true`. The rebuild's failure *and* its result were both unobserved, so a run could be fully
evaluated and verified yet absent from the API and the UI. See `F-01`.

### 3.3 The two LIMN audits cited different line numbers — RE-VERIFIED

The second LIMN audit found the first audit's citations stale and two named symbols non-existent
(`_run_category`, `replay_category_with_observation`; the real name is `replay_category`), and noted
that the first audit never mentioned `checkpoint_member_root` at all. The conclusions of the first
audit survived; its citations did not. Re-derived from the file: the same conclusion holds, with
current line numbers.

## 4. False-pass attempts that FAILED (the system held)

These are the attacks that were tried and refused. They are listed because a review that only reports
successes is not a review.

| Attack | Result |
|---|---|
| Nominate your own artifact digest | Refused: `promote` requires the digest from the bytes the reader computed AND the digest the verified report recorded, and refuses when the artifact's own field is used as a fallback. |
| Promote a per-query artifact with no aggregate receipt | Refused: `promote` returns early when no verdict exists. |
| Import a one-category artifact as a full-bundle result | Refused: `admission` requires the three categories, and `promote` re-checks admission rather than trusting a matching digest. |
| Reuse one checkpoint for two categories | Refused: `category_checkpoint_ids` must be distinct per category. |
| Repoint a completion proof to another run | Refused across 8 constructed attempts on an honest A/B pair. |
| Edit the ledger body, or hand-sign a duplicate entry | Refused: the HMAC covers the entry, and duplicates are handled first-writer-wins. |
| Serve a stale certification from the receipt cache | Refused: all six cache tokens are content digests; mutation probes against each re-keyed the cache. |
| Truncate or corrupt a saved ranking | Refused: depth and identity are validated on the index path as well as the canonical path. |
| Compare two protocols that happen to share a gallery size | Refused: isolation is keyed on `protocol_id` equality, and the probe used two runs whose gallery sizes were equal (260 == 260). |
| Pass `HOME` through to reach the ledger key | **Initially succeeded** — see `F-02`. Now refused. |

## 5. Findings that were fixed as a direct result

Full detail in `FINAL_DEFECT_LEDGER.md`. In summary, the independent audits produced:

* 3 P1 defects in first-party code (`F-02` credential re-admission, `F-03` stale image reuse,
  `F-04` self-referential checkpoint identity),
* the two defects the release task itself named (`F-01` launcher index, `F-05` active-run protocol
  leak), both confirmed by independent reviewers who were not told what to look for,
* 5 more P2 and 2 P3 defects across the launcher, the observer, the publisher, the schema and the API.

Two findings were confirmed and deliberately **not** fixed, with the reason recorded (`N-01` the 27
source-text tests, `N-02` score monotonicity). Three recordings were made rather than repairs because
repairing them would change the verification apparatus itself in a frozen release.

### 3.4 The scientific-semantics challenge found no defect on the reported-number path

Qwen was asked five questions, including "if you had to make this workbench report a scientifically
wrong number that every existing check would accept, what would you do?". Five attacks were attempted
(forged artifact, forged report, wrong-protocol run, a number absent from the native output, and
cross-protocol analytics) and all five were refused at a named line.

Two residuals were returned as `UNVERIFIED_RISK` rather than defects, and both are recorded as such in
the ledger (`N-04` the dataset cache order is not hash-bound; `N-05` tie order is not pinned across
NumPy builds) along with `N-06`, a bounded exposure in a model that has no evaluator. The second is
unobservable on this host and is a property of the PINNED upstream ranking: diverging from it to make
ties deterministic would break the reproduction, which is the one thing that must not happen.

### 3.5 The parity gate is a refusal, not a warning — checked after an allegation said otherwise

An allegation that parity only warns was rejected with three layers shown: the exporting child exits
non-zero, the parent refuses before publishing, the publisher raises `ParityMismatch`, and
`PER_QUERY_PARITY_FAILED` is a state `promote` cannot lift. That is four refusals for one failure,
which is what a fail-closed gate looks like.

## 6. What the reviewers could NOT establish

Stated plainly, because the alternative is a report that implies more confidence than it has.

* **No GPU validation.** Nothing in this pass ran on an RTX 5070 Ti. `TARGET_GPU_VALIDATION`,
  `LIMN_NATIVE_INFERENCE`, `LIMN_REAL_INSTRUMENTATION_PARITY` and `PAPER_REPRODUCTION` are all
  `PENDING`.
* **No real dataset or checkpoints.** FashionIQ and the three LIMN checkpoints were absent, so every
  runtime claim about LIMN is static or fixture-level, never a real measurement.
* **The Docker deployment path was exercised statically.** Compose configuration, the build, and the
  GPU override were validated as configuration; a real containerised pilot was not run.
* **A false-PASS state could not be constructed on the happy path** — but the GLM challenge's two real
  gaps required either an out-of-band checkpoint id or a stale container image, and the second of
  those was reachable in ordinary operation and is now fixed.

## 7. Why this pass did not stop at the first green suite

The suite was green before this pass began, at 1 358 passing tests, and it was green while seven
defects were live — including three that would make a pilot report success when it had not succeeded.
That is the whole justification for the 21 audits: a green suite is evidence about the tests, not about
the system. Two of the three P1s above were found by reviewers reading the executable source, not by a
test, and the regression tests for them were written **after** the defect was reproduced.
