# R8 — Independent review, EVIDENCE perspective

**Candidate under review:** uncommitted working-tree changes on top of `2bc0cd2`
(`git status` at review time: modified `pc.sh`, `workbench/scripts/evaluate_models.py`,
`workbench/tests/test_pc_launcher.py`, `workbench/tests/test_handoff_commands.py`, 4 docs; new
`workbench/docs/reviews/r8/`).

**Tree state actually reviewed** (all citations are against these bytes):

```
5d74979ade95adc06afa4a976a12c324e933941057531f56c244ee94c276a984  pc.sh
035e4b757f1761e32384f2b2b805d821f92e32e8437b44138cb16a58776d54e5  workbench/scripts/evaluate_models.py
975dfd3916c94bd5b2a68ce8e9d93d1c2fc84f95abaa23439f7f750e734a61ae  workbench/replay/limn_publish.py
```

> **Caveat on the review window.** `pc.sh` was modified *during* this review (the diff grew from
> 202 to 234 insertions between my first and second read). All probes and every line citation below
> were re-run/re-derived against the hash above; the index-gate program and the summary program were
> re-extracted from that revision and both e2e results were reproduced with it.

**Method.** No file in the repo was modified. Probes live under `/tmp/r8/`. The gate and the summary
were never re-implemented: both programs were **extracted from `pc.sh` itself** and driven as
separate processes (the extractor is `_index_gate_program()` /
`test_pc_launcher.py:2002`), and the shell-side read was reproduced with the **verbatim** `pc.sh`
line, in real `bash`, not an emulation.

---

## CLAIMS ATTACKED

### Claim 1 — "A development certificate can never satisfy the production release gate."

**Not disproved.** `pc.sh:1769` refuses both signals, and the refusal pre-empts the evidence test at
`pc.sh:1778`, so a lying dev receipt (`certified=false, is_research_evidence=false,
certification_mode="development"`) is still refused by name. 15 receipt shapes driven through the
real gate (`/tmp/r8/sweep.py`, `/tmp/r8/sweep2.py`, `/tmp/r8/sweep3.py`):

| # | receipt shape | rc | stdout |
|---|---|---|---|
| 0 | exactly as `_development_certification` emits it (`development_mode=true`, `certification_mode="development"`) | 1 | — (dev refusal) |
| 1 | `development_mode` **absent**, `certification_mode="development"` | 1 | — (dev refusal) |
| 2 | `development_mode="true"` (string), `certification_mode="development"` | 1 | — (dev refusal) |
| 3 | `development_mode=1` (int), `certification_mode="development"` | 1 | — (dev refusal) |
| 4 | `development_mode=None`, `certification_mode="development"` | 1 | — (dev refusal) |
| 5 | `certification_mode="Development"` / `" development "` / `"DEVELOPMENT"` / `"development\n"` | 1 | — (dev refusal) |
| 6 | `certification_mode` absent entirely, `development_mode=true` | 1 | — (dev refusal) |
| 7 | dev flag on the **report**, not inside `verification` | 0 | `AGGREGATE_ONLY_VERIFIED` |
| 8 | **both keys absent**, `certified=true`, `is_research_evidence=true` | 0 | `AGGREGATE_ONLY_VERIFIED` |
| 9 | both keys present but `None` | 0 | `AGGREGATE_ONLY_VERIFIED` |
| 10 | `certification_mode` an unexpected string (`"proof_unavailable"`, `""`, `True`, `["development"]`) | 0 | `AGGREGATE_ONLY_VERIFIED` |
| 11 | `report_digest` absent | 1 | — (`pc.sh:1789` refuses first) |
| 12 | `verification` absent entirely | 1 | — (not evidence) |
| 13 | `verification` is a string / a list / `{"receipt": …}` | 1 | — (not evidence) |
| 14 | deployment receipt (positive control) | 0 | `AGGREGATE_ONLY_VERIFIED` |

Shapes 7–10 are **accepted**. That is not exploitable through `/api/benchmark-runs`, because
`_aggregate_reports` always builds the receipt through `main.py:763-784`, which passes
`development_mode=bool(...)` and `certification_mode=receipt.get(...)` with the mode defaulted
(`main.py:771-772`), so a keyless receipt is unreachable from the API the gate reads. They are
recorded here as the gate's demonstrated failure boundary, matching
`test_pc_launcher.py:2857` (`test_the_development_refusal_survives_a_receipt_that_omits_both_fields`,
which documents the same acceptance).

**Residual (reachable, different trigger — CONFIRMED, see D-1):** the *gate program is correct*; the
**shell consumer of its output is not**. On the proof-unavailable branch the gate writes a
diagnostic to **stderr** (`pc.sh:1831`) while `pc.sh:1910` captures **stdout and stderr merged**
(`2>&1`) and assigns the result **verbatim** to `INDEX_EVIDENCE_STATUS` (`pc.sh:1919`). The compared
value is then a two-line string, and every `INDEX_EVIDENCE_STATUS == "…"` comparison in the summary
becomes false. This is not a missing development-certificate check; it is a broken channel between
two checks that both exist.

**Reproducer** (`/tmp/r8/chain.sh`, real `bash`; drives the extracted verify-gate program, then the
verbatim `pc.sh:1910` read, then the extracted summary program):

```
$ bash /tmp/r8/chain.sh
INDEX_EVIDENCE_STATUS=$'ghi chú: báo cáo khai \'PROOF_UNAVAILABLE_AND_CAPABILITY_ABSENT\' cho một mô hình
 không có trình quan sát\nAGGREGATE_ONLY_EXECUTION_AUTHENTICATED_PROOF_UNAVAILABLE'
SUMMARY EXIT = 0
KẾT LUẬN: SẴN SÀNG CHO PILOT (theo khả năng đã khai báo của mô hình).
  Bằng chứng tổng hợp: đã chứng thực và API phục vụ được lượt chạy này.
```

### Claim 2 — "The parent binds the published bundle identity to the plan."

**Partly disproved.**

* The check **fires** where the plan carries a digest. Forged artifact digest vs. plan digest →
  refused (`/tmp/r8/probe_plan_binding.py`); artifact recording exactly the plan digest → accepted
  (`/tmp/r8/probe_binding2.py`, `CTRL`).
* It is **reachable on the official path** even for a bundle whose weights are not on disk:
  `guarded_plan` builds `checkpoint_bundle` from the **declared** member names
  (`evaluate_models.py:399`), and `build_artifact_manifest` records `"sha256": None` for a member
  that is not a file (`evaluate_models.py:1149`), so both `plan.bundle_manifest_digest` and the
  `manifest_digest(_plan_members(plan))` fallback yield a **non-None** digest over the declared
  member set. `planned_digest` is therefore truthy and the check is not vacuous for LIMN
  (`/tmp/r8/probe_binding2.py`, `VAIN`).
* **But the check is vacuous whenever the plan resolves no members at all.** With
  `checkpoint_bundle=None` and `directory_members=None`, `_plan_members` returns `None`
  (`evaluate_models.py:1125-1126,1131`), `manifest_digest(None)` returns `None`
  (`evaluate_models.py:1164-1165`), and `planned_digest` is falsy, so the whole block
  (`evaluate_models.py:1762-1769`) is skipped and an artifact recording an arbitrary bundle digest
  is **accepted**:

```
$ .venv/bin/python -u /tmp/r8/probe_binding2.py
VAC2  planned_digest = None
VAC2  artifact/note  = (PosixPath('/tmp/r8/b2c/canonical.json'), "artifact exported (data_kind='experiment'); admission is the verifier's to decide")
```

  On the *official* path this is not reachable for LIMN: `candidates` substitutes
  `checkpoint_by_id(model, bundle_id)` (`evaluate_models.py:248`) **before** `guarded_plan` runs, so
  the plan is built from the bundle record and carries its members. The residue is any caller that
  reaches `execute()` with a hand-built plan, and `limn_publish` invoked directly (claim 3).
* The observer and publisher **cannot** disagree without the parent seeing it: the publisher writes
  `bundle_manifest_digest = manifest_digest(members) if members else None`
  (`limn_publish.py:101`), which is derivable only from the payload's `checkpoint_members`, and the
  parent compares the *same* recomputation against the *plan*.

### Claim 3 — "Missing planned digest cannot become accepted evidence."

**Disproved for the official path, disproved for the parent, CONFIRMED for the publisher CLI.**

* Official path: the parent always passes `--planned-bundle-manifest-digest` when the plan resolves
  members (`evaluate_models.py:1690-1692`), and re-derives the digest itself after publication
  (`evaluate_models.py:1761`). Closed.
* **But `limn_publish.publish_payload` still publishes a self-declared bundle identity when the
  payload carries no `planned_bundle_manifest_digest`** (`limn_publish.py:80`: `if planned_digest:`).
  The gate is skipped, `members` are whatever the payload declares, and
  `bundle_manifest_digest = manifest_digest(members)` (`limn_publish.py:101`) is recorded from those
  self-declared members:

```
$ .venv/bin/python -u /tmp/r8/probe_publish_cli.py
publisher rc: 0 {"published":"/tmp/r8/pubcli/results/fashioniq_val_split/limn/20260101T000000000000Z_limn_ckpt_probe.json"}
artifact records bundle_manifest_digest: e06dc6118867742de27ca2575eb48a44cbc4ecc2d32e926fd8bdfb341d67f869
digest over the ONE member staged       : e06dc6118867742de27ca2575eb48a44cbc4ecc2d32e926fd8bdfb341d67f869
payload had planned digest?             : False None
checkpoint_bundle members recorded      : ['dress']
```

  The published artifact claims `checkpoint_id = base_iter0_all_categories` (a three-category bundle)
  while its recorded bundle is **one** member. Residual exposure: any direct invocation of the
  documented CLI (`limn_publish.py:124-138`) or a hand-staged payload — the module's own docstring
  documents the CLI as the interchange, and `_run_observation` is not the only way to reach it.
  Precise boundary: the **parent** closes it on the path the launcher uses; the **module** does not.

### Claim 4 — "Execution authentication is never presented as scientific certification."

**Not disproved at the token level; disproved at the consumer level.**

* The token vocabulary is genuinely separated in the gate: `pc.sh:1843` prints
  `AGGREGATE_ONLY_EXECUTION_AUTHENTICATED_PROOF_UNAVAILABLE`,
  `pc.sh:1845` prints `AGGREGATE_ONLY_VERIFIED`, and the proof-policy branch is gated on
  `certification_mode == "proof_unavailable"` **and** `execution_authenticated is True`
  **and** `state in ("proof_unavailable", "refused")` (`pc.sh:1754-1757`). No other line prints a
  success claim for an unverified run (`grep -n "SẴN SÀNG CHO PILOT\|THÀNH CÔNG\|ĐẠT —" pc.sh` →
  `2287, 2292, 2298, 2394`; `2394` is the `--verify-only` GPU banner, unrelated).
* **But the conclusion and the evidence line ARE printed for an unverified run on the real path**,
  and they contradict the gate's own token — see D-1. `pc.sh:2268`/`pc.sh:2294`/`pc.sh:2336` are
  correct as written; they are simply never reached because `INDEX_EVIDENCE_STATUS` is two lines
  long. The guard that was written to prevent exactly this (`pc.sh:2299-2307`, "The evidence line
  must agree with the conclusion two lines above it") is defeated by the same pollution.

---

## CONFIRMED DEFECTS

### D-1 · P1 · The index gate's stderr note is merged into `INDEX_EVIDENCE_STATUS`, so a proof-unavailable run is reported as a verified, pilot-ready result and exits 0

**Evidence.**
* The gate prints a diagnostic to **stderr** on the aggregate-only branch:
  `pc.sh:1830-1832` — `if state and state not in ("CAPABILITY_ABSENT",): print("ghi chú: …",
  file=sys.stderr)`.
* `pc.sh:1817` sets `state = "PROOF_UNAVAILABLE_AND_" + (state or "CAPABILITY_ABSENT")` **before**
  that branch, so on the proof-unavailable path `state` is *always*
  `PROOF_UNAVAILABLE_AND_…` and the note is *always* printed.
* `pc.sh:1910` captures with `2>&1`; `pc.sh:1919` assigns the merged text verbatim to
  `INDEX_EVIDENCE_STATUS`.
* The summary compares that value by equality: `pc.sh:2268`, `pc.sh:2294`, `pc.sh:2303`,
  `pc.sh:2308`, `pc.sh:2274`, `pc.sh:2295`, `pc.sh:2304`, `pc.sh:2311`, `pc.sh:2336`.

**Reachable trigger.** `FINAL_RUN_PROOF_UNAVAILABLE=1` is reachable without any malformed input: the
reuse reader returns exit 2 for a completed, ledger-corroborated candidate whose *only* failing
reasons name a missing completion proof (`pc.sh:1424-1454`), and `pc.sh:1458-1465` then sets
`FINAL_RUN_PROOF_UNAVAILABLE=1`, which `pc.sh:2152` passes to `verify_gate`. Proved with the
verbatim `pc.sh:1364-1457` program:

```
$ bash /tmp/r8/probe_dedup.sh
check_rc=2  (0=verified reuse, 2=proof-unavailable reuse -> FINAL_RUN_PROOF_UNAVAILABLE=1, 1=skip)
```

**ACTUAL behaviour.**

```
$ bash /tmp/r8/e2e.sh
INDEX_EVIDENCE_STATUS=$'ghi chú: báo cáo khai \'PROOF_UNAVAILABLE_AND_CAPABILITY_ABSENT\' cho một mô hình không có trình quan sát\nAGGREGATE_ONLY_EXECUTION_AUTHENTICATED_PROOF_UNAVAILABLE'
SUMMARY: -> ordinary 'AGGREGATE_ONLY_VERIFIED' line (lines 2246-2249)
SUMMARY: KẾT LUẬN: SẴN SÀNG CHO PILOT + 'Bằng chứng tổng hợp: đã chứng thực'
…
KẾT LUẬN: SẴN SÀNG CHO PILOT (theo khả năng đã khai báo của mô hình).
  Bằng chứng tổng hợp: đã chứng thực và API phục vụ được lượt chạy này.
SUMMARY_EXIT=0
```

**EXPECTED behaviour.** `pc.sh:2336` — "Not 0. The invocation is authenticated and the numbers are
recorded, but nothing verified them, and a zero here would present an unverified run as a passed
pilot." The branch at `pc.sh:2336-2339` is designed to `sys.exit(7)`; the pollution makes it
unreachable.

**Severity P1.** Not P0: the gate's own scientific verdict (`AGGREGATE_ONLY_EXECUTION_AUTHENTICATED_PROOF_UNAVAILABLE`)
is correct and is still printed, so the text is present in the operator's output. It is P1 because
the machine-readable success signal is wrong in the two ways the launcher contract is defined by —
the conclusion line and the **exit code** — and the run is presented to the operator as a certified,
pilot-ready result.

**Existing tests do not catch it.** `test_pc_launcher.py:2470` asserts
`proc.stdout.strip() == "AGGREGATE_ONLY_EXECUTION_AUTHENTICATED_PROOF_UNAVAILABLE"` on
`process.stdout` only, and the state this test can reach is `CAPABILITY_ABSENT`
(`_report(... per_query_state="CAPABILITY_ABSENT")`), which is precisely the one state where the note
is **not** printed (`pc.sh:1830`). `test_pc_launcher.py:2536`
(`test_the_summary_exits_seven_for_an_authenticated_but_unverified_aggregate_run`) hands the summary
the clean token string directly, bypassing the shell channel. Positive control on the per-query path:
a real deployment receipt still yields a clean `PER_QUERY_VERIFIED` (`/tmp/r8/probe_pq.py`), so the
damage is confined to the proof-unavailable branch.

**Reproducer (minimal, real bash, no emulation):**

```bash
# /tmp/r8/probe_fixture2.sh — `checked=` line is verbatim pc.sh:1910
per_query_state=CAPABILITY_ABSENT
  INDEX_EVIDENCE_STATUS=$'ghi chú: báo cáo khai \'PROOF_UNAVAILABLE_AND_CAPABILITY_ABSENT\' cho một mô hình không có trình quan sát\nAGGREGATE_ONLY_EXECUTION_AUTHENTICATED_PROOF_UNAVAILABLE'
```

Any receipt reaching the aggregate-only branch with `state` other than `CAPABILITY_ABSENT` produces
the two-line value; `proof_policy` guarantees it on the proof-unavailable path.

**Pre-existing.** The `2>&1` read is at `2bc0cd2:pc.sh:1767`, the note at `2bc0cd2:pc.sh:1690`, and
the same summary comparisons at `2bc0cd2:pc.sh:2122/2131`. Reproduced against the base revision's own
programs (`/tmp/r8/e2e_base.sh`): base output is byte-identical in effect — `SẴN SÀNG CHO PILOT`,
`Bằng chứng tổng hợp: đã chứng thực`, exit 0. So this is not a regression introduced by R8-01..R8-05;
it is an unfixed defect that the candidate's own new token vocabulary (A-02) was written to prevent.

---

### D-2 · P2 · `_development_certification` omits the binding fields its sibling receipt carries, so `promote` fails closed only by accident

**Evidence.** `main.py:563-565` returns a receipt whose keys are
`{report_digest, verified, is_research_evidence, reasons, evidence_reasons, certified,
certification_mode, development_mode}` — notably **no** `per_query_artifact_digest`, `model_id` or
`checkpoint_id`. The deployment receipt carries all three (`main.py:604-607`), and
`main.py:781-783` copies them into `report["verification"]` for both.

**ACTUAL behaviour.** `promote` reads only the receipt — `if not all(verifier_verdict.get(key) is
True for key in ("certified", "is_research_evidence"))` (`results.py:347`) — with **no**
`development_mode` / `certification_mode` term. A development receipt passes that gate by
construction (`certified=true`, `is_research_evidence=true`), and is then blocked only by the
next check, a missing `per_query_artifact_digest` (`results.py:353,359`):

```
$ .venv/bin/python -u /…/ -
admission: ('PER_QUERY_UNVERIFIED', ['awaiting corroboration by the authoritative verifier (evidence ledger + completion proof)'])
as-emitted      -> PER_QUERY_UNVERIFIED
dev+bound digest-> ('PER_QUERY_VERIFIED', [])
```

**EXPECTED behaviour.** "A development certificate is not a production certification" is a property
`promote` should assert, not one it should obtain from an absent field. One extra key in
`main.py:563` (the per-query artifact digest the report already holds) turns this into a
`PER_QUERY_VERIFIED` promotion with no verifier having spoken.

**Severity P2 — not currently exploitable, and I could not reach it.** `verify_gate` never writes
`PER_QUERY_VERIFIED` into `per_query_artifact_digest`; it only copies the field the verifier's own
JSON carries (`pc.sh:2021-2023`), and `_development_certification` produces no artifact digest. All
producers of a `promote` verdict are `main._aggregate_reports` and
`index.verifier_verdicts`, both of which emit the same field set. Demonstrated only by construction,
so the *reachability* of the bad state is UNVERIFIED; the *absence of the guard* is CONFIRMED by the
two-line probe above.

---

### D-3 · P3 · The plan-binding comparison prefers the plan-time digest over the digest of the members on disk, inverting the module's own stated rule

**Evidence.** `evaluate_models.py:1750-1769` compares the artifact's digest against
`plan.bundle_manifest_digest or manifest_digest(_plan_members(plan))` (`:1761`). The comment at
`:1758-1760` states the opposite precedence: "`_plan_members` hashes the members on **DISK**, before
the observer runs, and `plan.bundle_manifest_digest` is derived from that." `_plan_members`'
docstring already records that a plan's recorded digests may be placeholders
(`evaluate_models.py:1112-1114`), and `build_artifact_manifest` writes `"sha256": None` for a member
that is not yet a file (`evaluate_models.py:1149`).

**ACTUAL behaviour.** A plan whose member digests were recorded while the weights were absent, with
the weights present by the time the observer runs, refuses a perfectly valid export:

```
$ .venv/bin/python -u /tmp/r8/probe_stale.py
plan-time (files absent) digest: 39319874348df98f741716993c0ceea56028ee531ea6b1e5ebda40a5d00b3b39
digest over the real weights    : 89fc4ac72d442082a3e493b7465b937543eb9ffd9b50f56d23e1d02ce94e5f0d | differs: True
planned_digest the parent compares: 39319874348df98f741716993c0ceea56028ee531ea6b1e5ebda40a5d00b3b39
  ...but _plan_members recomputes  : 89fc4ac72d442082a3e493b7465b937543eb9ffd9b50f56d23e1d02ce94e5f0d
VERDICT: REFUSED —
```

**EXPECTED behaviour.** Either precedence, consistently: the check is meant to be against the bytes
the run evaluated, and `plan.bundle_manifest_digest` is the *run-time* digest for the official path
(it is computed once at plan time from the same on-disk members, since the aggregate evaluator hashes
the live bundle later). Both are self-consistent in production; the inversion only bites when the
plan's manifest is a placeholder, which is exactly the case `_plan_members` exists to correct.

**Severity P3.** Non-fatal and fail-closed (the failure mode is a refusal, not a false certification),
it degrades a legitimate observer export to `PER_QUERY_EXPORT_FAILED` for a capability model. I could
not reach it on the official path: for LIMN, `guarded_plan` builds `checkpoint_bundle` from the
registry's declared members and there is a mandatory preparation stage between plan and evaluation.
So: the *precedence divergence* is CONFIRMED by probe; *production reachability* is UNVERIFIED.

---

## DISPROVED ALLEGATIONS

* **"A development certificate reaches a success token through an alternative receipt shape."**
  Disproved for shapes 0–6, 11–13 above, including the falsy-value forms (`"true"`, `"True"`, `1`,
  `None`, `[]`, `{}`, `"yes"`, `"TRUE"`) when the other signal is present, and for a receipt that
  lies in the opposite direction (`certified=false` with `certification_mode="development"` still
  refused by name). Shapes 7–10 are accepted but unreachable through `_aggregate_reports`.
* **"The gate can be reached with the report nested differently, or with `verification` missing."**
  Disproved: `verification` is required to be a `dict` (`pc.sh:1747`), and a missing/`str`/`list`/
  `{"receipt": …}` verification falls to "không công bố … bằng chứng nghiên cứu đã chứng thực".
* **"`report_digest` absent still reaches a token."** Disproved: `pc.sh:1789` refuses before any
  branch.
* **"The parent's new bundle check is always vacuous."** Disproved: forged digest → refused with a
  named reason; a bundle whose weights are absent still yields a non-`None` plan digest because the
  member set is *declared* (`evaluate_models.py:399`) and `sha256` is nullable
  (`evaluate_models.py:1149`); a plan whose weights are present is also bound (control passes).
* **"The observer and the publisher can disagree such that the artifact records a digest the parent
  never sees."** Disproved: `limn_publish.py:101` derives the recorded digest from the payload's
  `members` with the same function the parent uses, so no third value can appear.
* **"`limn_publish`'s `if planned_digest:` gap is closed by the parent everywhere."** Disproved —
  the parent closes it *only on its own call path*; the CLI still publishes a one-member
  "three-category" artifact (probe above).
* **"Execution authentication is presented as scientific certification anywhere in the token
  vocabulary."** Disproved at the token level: both tokens are distinct, the proof policy is
  triply-gated (`pc.sh:1754-1757`), and no other success line exists. The defect is in the consumer
  (D-1), not the vocabulary.
* **"The proof-unavailable aggregate-only run already exits non-zero via a different branch."**
  Disproved: `/tmp/r8/e2e.sh` and `/tmp/r8/chain.sh` both end `SUMMARY_EXIT=0` through the real
  summary program driven by the real verify-gate program.

## UNVERIFIED

* **Production reachability of D-1's trigger.** Argued from code (`pc.sh:1458-1465` → `pc.sh:2152`,
  plus the fact that `completion_validator` refuses an absent checkpoint directory at
  `evaluate_models.py:2249+` and the pipeline prunes/never wrote proofs for the current run), and
  the `check_rc=2` path is proved with the real program. I did not stage a deployed run with a pruned
  proof end-to-end (`workbench/artifacts/reports/` holds no `*_aggregate.json`), so I cannot show the
  operator's exact terminal. The pollution mechanism itself is proved at the process boundary.
* **Reachability of D-2's bad state** (a development receipt carrying a per-query artifact digest).
  No producer does it today; judged by reading every writer, not by a probe.
* **Whether D-2's `per_query_artifact_digest` key would ever be populated for a development receipt**
  — `_development_certification` never computes one and the report field it would need is not passed
  in.
* **Production reachability of D-3** — see above.
* **`workbench/scripts/verify_run.py` and `workbench/tests/test_handoff_commands.py`.** I read
  `verify_run.verify`'s output contract (`verify_run.py:219` `report_path`) to build D-1's chain, and
  I confirmed `test_handoff_commands.py` is a test-only change (production-tree comparison instead of
  ancestry) that cannot affect shipped behaviour. I did not audit the release-metadata guard's
  correctness — out of this ticket's perspective.

## VERDICT

**SOUND_WITH_RESERVATIONS.**

The five R8 claims are individually defensible: the development-certificate gate works for every
reachable receipt shape, the API's receipt normalisation closes the null/absent forms the gate itself
would accept, the parent's plan binding fires and is reachable even without weights on disk, and the
token vocabulary does keep execution authentication separate from scientific certification.

What the candidate does not survive is its own new separation. The proof-unavailable token
(`AGGREGATE_ONLY_EXECUTION_AUTHENTICATED_PROOF_UNAVAILABLE`) is only ever produced together with a
stderr diagnostic, and `pc.sh:1910` merges stderr into the value the summary compares (`pc.sh:1919`),
so the summary's own exit-7 branch is unreachable and the operator is told the run is
`SẴN SÀNG CHO PILOT` / "đã chứng thực" with exit 0. That is the exact false claim the candidate's own
comments (`pc.sh:1833-1841`, `pc.sh:2336-2339`) say must never be made — and the new tests cannot see
it because they assert on `stdout` alone and pin the one state (`CAPABILITY_ABSENT`) where the
diagnostic is not printed. The same "both checks exist but the channel between them is broken" shape
reappears as a narrower version in D-3, and its mirror image (a check that `promote` gets right only
by accident of an omitted field) in D-2.

D-1 must be fixed before readiness is claimed: capture the gate's stdout alone (or compare on the
last line), and add a regression that drives the *merged* capture the shell actually performs.
