# R8 — INDEPENDENT RELEASE REVIEW (revisions, documents, commands, test provenance, model readiness)

READ-ONLY audit of the uncommitted working tree on top of `2bc0cd2`. Exactly one file was created:
this one. No repo state was mutated (`git status --porcelain` at the end names 12 modified paths and
the untracked `workbench/docs/reviews/r8/`, all pre-existing except this file). No GPU inference was
run; no runtime behaviour of `pc.sh` inside Docker was executed. Every claim below is `STATIC_AUDIT`
unless the line quotes a command and its observed output (`RUNTIME_VERIFIED`).

Baseline at the moment this review started:

```
$ git rev-parse HEAD; git rev-parse origin/main
2bc0cd223028057b577b861d83d9e2795b040840
8ebcc74d5fd1934df4f3772a8ca5a96dbd00f333          # ground truth confirmed, RUNTIME_VERIFIED
```

**Concurrency warning (not caused by me):** `pc.sh`, `workbench/tests/test_pc_launcher.py` and
`workbench/scripts/evaluate_models.py` were being edited *while* this review ran (`ls -l
--time-style=+%H:%M:%S`: `pc.sh` 17:15:52, `test_pc_launcher.py` 17:20:23; line count moved 2408 →
2876 → `def test_` 98 → 1344 total). Collection count for the working tree therefore moved
**1419 → 1422** during the audit. Measurements against the *committed* revision `2bc0cd2` are
unaffected and are marked as such.

---

# CLAIMS ATTACKED

1. "Every release document now agrees with the actual repository." → **DISPROVED in part**; full
   claim table in §5. `SOURCE_COVERAGE_MANIFEST.md` is stale in **45 of 126 rows**; the verdict's
   install block still names an unpinned `FETCH_HEAD`; `FINAL_PC_RELEASE_VERDICT.md` is simultaneously
   "TESTED_SOURCE_SHA = `2bc0cd2`" (committed) and a re-run of exactly the same gates in the handoff
   (pending commit), while the fixes under review are **not** in `2bc0cd2`.
2. "The guard test cannot be satisfied by an obsolete ancestor." → **DISPROVED.** Built and ran.
3. "The count is right." → **MATCH for the committed tree, MISMATCH for the working tree.**
4. "The handoff installs the verified code." → **PARTIAL.** `git checkout 2bc0cd2` resolves and its
   production tree is a superset-of-`11ba3df` tree, but it does **not** contain the changes under
   review.
5. Model readiness rows in `R8_16_MODEL_STATUS.md` → **two wrong classes of row** (§6).

---

# CONFIRMED DEFECTS

## D-1 (P1) — the release metadata names a tree that does not contain the changes being released

`workbench/docs/reviews/FINAL_PC_RELEASE_VERDICT.md:9` (`TESTED_SOURCE_SHA` = `2bc0cd2`) and
`workbench/docs/PC_PILOT_HANDOFF.md:8-9` ("1406 passed ... mọi test, acceptance, mutation và
release_verify ... đã chạy trên đúng cây đó") assert that the gates ran on `2bc0cd2`. The changes this
task exists to review are **uncommitted**:

```
$ git diff --numstat 2bc0cd2 -- . ':(exclude)workbench/docs' ':(exclude)workbench/tests'
229  58  pc.sh
20   0   workbench/scripts/evaluate_models.py
```

Trigger: operator follows the handoff, installs `2bc0cd2`, and gets a launcher **without** the argv
path-carriage fix, the development-certificate gate and the parent-side bundle-digest binding.
Reachable: yes, it is the documented install path.
Actual: documented tree lacks 249 added production lines that the same documents describe as fixed.
Expected: either commit them and advance `TESTED_SOURCE_SHA`, or state plainly that the reviewed
revision is the working tree.
Reproducer: `git show 2bc0cd2:pc.sh | grep -c 'MAX_ARG_STRLEN'` → `0` (working tree → `5`);
`git show 2bc0cd2:pc.sh | grep -c 'development_mode'` → `0` (working tree → `3`);
`git show 2bc0cd2:workbench/scripts/evaluate_models.py | grep -c 'must describe the members'` → `0`.

## D-2 (P1) — the production-tree guard is satisfiable by an obsolete recorded source

`workbench/tests/test_handoff_commands.py:334-443`. The test compares **the handoff's checkout SHA
against the verdict's TESTED_SOURCE_SHA**, and never compares either against the code that is actually
going to ship. A history where both documents are edited together to name an obsolete commit passes
all seven tests while production differs between the recorded source and HEAD.

Reproducer (`/tmp/g3`, clone of this repo, no commit on the branch): base `2bc0cd2` + a commit that
appends one line to `pc.sh`; verdict and handoff both edited to name `11ba3df` and `1330`:

```
$ git log --oneline -1              # 678ad96 obs prod
$ git diff --numstat 11ba3df HEAD -- . ':(exclude)workbench/docs' ':(exclude)workbench/tests'
103  22  pc.sh
12   1   workbench/frontend/evidence.ts
13   1   workbench/replay/limn_export.py
18   2   workbench/replay/limn_publish.py
43   5   workbench/scripts/evaluate_models.py
$ .venv/bin/python -m pytest workbench/tests/test_handoff_commands.py -q -p no:cacheprovider
.......                                                                  [100%]
7 passed in 0.63s
```

A stronger variant only edits the **verdict's** `TESTED_SOURCE_SHA` and keeps the handoff's
`git checkout 2bc0cd2`; the guard still passes, even though the handoff then installs code that the
verdict denies was tested (`/tmp/g2`: `superseded_source_sha` → 1 passed). Severity **P1** because this
is precisely the defect class R8-01 was written to close, and the replacement closes only one
direction of it.

## D-3 (P2) — the guard's count check is insensitive to the difference between sources and collection

`workbench/tests/test_handoff_commands.py:439-471`: when `TESTED_SOURCE_SHA` == `HEAD` the test
returns immediately (`:466`); otherwise it compares `def test_` counts only. The verdict's `1406` is
therefore bound to **pytest collection**, which the test never computes:

```
committed 2bc0cd2 tree (clean extract of `git archive 2bc0cd2`, /tmp/head_tree),
$ .venv/bin/python -m pytest workbench/tests --collect-only -q -p no:cacheprovider
1406 tests collected in 3.59s                       # MATCHES the documents exactly

working tree, 17:2x, same command
1422 tests collected in 1.46s                       # documents still say 1406

static `def test_` count: 2bc0cd2 = 1329, working tree = 1344
```

So the recorded `1406` is correct **for `2bc0cd2` only**, and the delta (+22 collected, +15 test
functions) is the uncommitted work — the same work D-1 says the documents do not name.

## D-4 (P2) — `SOURCE_COVERAGE_MANIFEST.md` is stale in 45 of its 126 rows

Recipe confirmed (`sha256[:16]`, line count) against rows that still match; 45 rows disagree:

```
$ .venv/bin/python /tmp/mf.py
rows: 126   checked: 126   mismatched: 45
  ('workbench/tests/test_handoff_commands.py', '205', 502, '855c29b0ba42f90f', 'd173d08c915d88e6')
  ('workbench/backend/main.py',               '361', 1169, '2e808c406343580d', 'ac94f9b328992a2d')
  ('workbench/scripts/evaluate_models.py',    '1884', 2417, '5e40501f0ca2171c', '035e4b757f1761e3')
  ('workbench/backend/registry.py',           '743', 1007, 'fefa7e515b4303fb', '6bbcc004dcdac14b')
  ... 41 more
```

`workbench/docs/SOURCE_COVERAGE_MANIFEST.md:89` is the row R8-01 flagged. R8-01 spot-checked four
neighbours and called the rest fine; **that hypothesis is wrong** — 45 rows are stale, including the
file the manifest lists as `TESTED_PATH` for the very test that guards release metadata.

## D-5 (P2) — `FINAL_PC_RELEASE_VERDICT.md` still installs an unpinned target

`workbench/docs/reviews/FINAL_PC_RELEASE_VERDICT.md:152`:

```
git fetch origin research/cir-failure-workbench
git checkout FETCH_HEAD        # an explicit checkout; never `git reset --hard`
```

`FETCH_HEAD` after that fetch is the moving branch tip; the handoff (`PC_PILOT_HANDOFF.md:30`)
explicitly forbids branch names for this reason. The guard's regex requires a hex SHA
(`test_handoff_commands.py:365`) and `test_pc_launcher.py`/`test_handoff_commands.py` read only the
handoff (`:365`) and the verdict's **bold hex** tokens (`:395`), so this line is untested. Verified
that the branch tip is in fact still `2bc0cd2`:

```
$ git ls-remote origin refs/heads/research/cir-failure-workbench refs/heads/main
8ebcc74d5fd1934df4f3772a8ca5a96dbd00f333  refs/heads/main
2bc0cd223028057b577b861d83d9e2795b040840  refs/heads/research/cir-failure-workbench
```
(`RUNTIME_VERIFIED` — network reachable; the remote tip equals the local HEAD today, which is why the
defect is latent rather than active.)

## D-6 (P3) — the defect ledger's own regression record is false and now contradicts the test it names

`workbench/docs/reviews/FINAL_DEFECT_LEDGER.md:358-361` is inside `RELEASE_DOCS`
(`test_handoff_commands.py:296-302` includes it):

```
* **Fix**: both name `11ba3df`; ...                          # stale: the documents now name 2bc0cd2
* **Regression**: ... the checkout target must be an ancestor of the tested source ... Negative
  control: restoring `git checkout 4b32063` fails the test.
```

Both halves are wrong against the current test. The test asserts **production-tree equality**, not
ancestry (`:372-384`), and the "negative control" is false:

```
$ git merge-base --is-ancestor 4b32063 11ba3df; echo $?   → 0
$ git merge-base --is-ancestor 4b32063 2bc0cd2; echo $?   → 0
```
`4b32063` is an ancestor of both, and **no test pins the handoff's `git checkout` to the tested SHA**
(see D-2's second variant), so restoring `git checkout 4b32063` at `PC_PILOT_HANDOFF.md:30` alongside
`TESTED_SOURCE_SHA 2bc0cd2` leaves the suite green.

## D-7 (P3) — the defect-count sentence and the table disagree

`workbench/docs/reviews/FINAL_PC_RELEASE_VERDICT.md:56` says "Thirteen confirmed defects"; the table
below it has **12** `F-` rows and `F-13` appears nowhere in the tree
(`grep -c F-13` → 0 in both the verdict and the ledger). Pre-existing since `b55fc29`, not introduced
by this working tree, and not about revisions — reported for completeness.

---

# DISPROVED ALLEGATIONS

| # | allegation | how it was disproved |
|---|---|---|
| A1 | "The guard cannot be satisfied by an obsolete ancestor" (`test_handoff_commands.py:407-436`) | `/tmp/g3`: ancestor `11ba3df`, production moved by 189 numstat lines, handoff and verdict both name the obsolete commit and `1330` → **7 passed**. The test's own docstring says "Ancestry proves a commit is in the history; it proves nothing about the bytes" — true of the *old* test, and still true of the new one when the documents are edited together. |
| A2 | "The guard detects that the handoff's checkout tree differs from the tested tree" (the fix's stated purpose, `PC_PILOT_HANDOFF.md:17-19`) | `/tmp/g4`: handoff `git checkout 11ba3df`, verdict `TESTED_SOURCE_SHA 11ba3df` → `test_no_release_document_points_at_a_superseded_source_sha` **1 passed**, even though `_production_tree("11ba3df") != _production_tree("2bc0cd2")`. The comparison is internal to the document pair. |
| A3 | "`test_the_verdict_names_the_same_test_count_the_handoff_does` ties the handoff's count to the verdict's" (name) | `:496-502` reads **only** the verdict and asserts `len(set(counts)) == 1`. The handoff's `(1406 passed; …)` is never parsed. This is the exact hole R8-01 found at `R8_01_METADATA_AUDIT.md::213`; it is unchanged in the tree under review. |
| A4 | "the count check binds the recorded number to what pytest collects" | `:439-471` compares `def test_` source counts, never collection: 1344 static in the working tree vs 1422 collected. Any recorded count ≤ 1344 passes. |
| A5 | R8-01's implied scope ("the manifest mismatch is the `test_handoff_commands.py` row") | 45 of 126 rows mismatch (D-4). |
| A6 | "`11ba3df` and `b70d092` differ only in documentation" | true and re-verified: `_production_tree("11ba3df") == _production_tree("b70d092") == 49092d843e4b4bee…`. The claim is honest; it is the *later* commits that move production. |
| A7 | "`11ba3df` contains no `sys.exit(7)`/`REGISTRY_UNKNOWN`" (handoff's justification) | confirmed, not disproved: `git show 11ba3df:pc.sh | grep -c` → `0`/`0`; at `2bc0cd2` → `1`/`5`. |
| A8 | "the handoff's documented exit-code list is wrong" | **no**: `0/3/4/5/6/7` all exist in `pc.sh` (`sys.exit(3)`:2341, `(4)`:2345, `(5)`:2332, `(6)`:2209/2214, `(7)`:2339); the handoff's table (`:81-90`) is consistent with `pc.sh` and with the verdict's one-line summary (`FINAL_PC_RELEASE_VERDICT.md:27`). The handoff omits 6; it never contradicts it. |
| A9 | "`--verify-only` is described in the release documents as verifying a run" | **not found**. All five `RELEASE_DOCS` describe it correctly (`PC_PILOT_HANDOFF.md:187-189`, `PC_ONE_COMMAND_GUIDE.vi.md:231`, `FINAL_PC_RELEASE_VERDICT.md:28,160`); `test_no_release_document_describes_verify_only_as_verifying_a_run` passes. Static reading of `pc.sh:2350-2367` confirms `VERIFY_ONLY == 1` → `verify_gpu_in_container` → `exit 0`, and preparation/checkpoint work is skipped. |
| A10 | "`AUTOMATION_RUNBOOK.md` quotes a repo SHA/test count that has gone stale" | correct as written: its `--verify-only` uses are **script** flags (`:120,125,129`) and `download_checkpoints.py`/`download_auxiliary_assets.py`/`sync_upstreams.py` each define one (`grep -c '\"--verify-only\"'` → 1/1/1). It states no repo SHA and no suite count. |
| A11 | "`PC_ONE_COMMAND_GUIDE.vi.md` clones a branch instead of a revision" | true but **not a defect I can call a factual mismatch**: `:70` `git clone -b research/cir-failure-workbench …` is a moving target, the same class as D-5. Kept as MISMATCH in §5, severity P3, because the guide is one of the `RELEASE_DOCS` and carries no SHA anywhere. |
| A12 | "the historical reports contradict the current metadata" | mostly **not**: `FINAL_DEFECT_LEDGER.md:6`, `FINAL_16_MODEL_AUDIT_MATRIX.md:9`, `FINAL_FIRST_PARTY_SOURCE_COVERAGE.md:7`, `FINAL_ADVERSARIAL_REVIEW.md:6` were all re-labelled "at the time of THIS …" by the working tree, so they are now explicitly HISTORICAL. |

---

# UNVERIFIED

1. `1406 passed, 0 failed` — I collected 1406 tests on `2bc0cd2` and executed **no** full suite (forbidden). The *pass* count is `UNVERIFIED`; the *collection* count is verified.
2. Acceptance / mutation / `release_verify.sh --offline` / `tsc` / `npm run build` / compose results in `FINAL_PC_RELEASE_VERDICT.md:44-51` — not run; no repo mutation permitted.
3. Any GPU/`pc.sh` end-to-end behaviour: Docker was not exercised. All `pc.sh` claims here are static source reads.
4. The paper-level claims (protocol A/B/C gallery sizes 2628/3089/2902; Recall rows) — out of this ticket's scope and not attacked here.
5. `test_pc_launcher.py`'s new R8-02/R8-03 regressions: file was still being written while I audited (17:20:23); I ran the file once at an earlier state (120 s, no failures reported) and did not re-run it after the last write, deliberately, to avoid racing a live editor. Verdict `NOT_RUN` as of the final byte count.

---

# §5 — FULL CLAIM TABLE

Legend: **MATCH** = repository agrees; **MISMATCH** = it does not; **HISTORICAL** = explicitly labelled
as a past pass/baseline (permitted). Discovery was a full sweep of `workbench/docs/**/*.md` for hex
tokens on checkout/tested/HEAD/count lines (`/tmp/sweep.txt`), not a hand-picked list.

## 5a. Source SHAs presented as current

| # | file:line | claim | repository says | verdict |
|---|---|---|---|---|
| 1 | `PC_PILOT_HANDOFF.md:8` | `SHA ĐÃ KIỂM THỬ (nguồn sản xuất): 2bc0cd2` | resolves; but the reviewed changes are uncommitted (D-1) | **MISMATCH** (as the tested tree) |
| 2 | `PC_PILOT_HANDOFF.md:12-16` | `11ba3df` = obsolete source; diff touches `pc.sh`, `limn_export.py`, `limn_publish.py`, `evaluate_models.py`, `evidence.ts`; no `sys.exit(7)`/`REGISTRY_UNKNOWN` there | `git diff --numstat 11ba3df 2bc0cd2` → those **are** the diff (5 files, 167/31); token counts 0/0 vs 1/5 | **MATCH** |
| 3 | `PC_PILOT_HANDOFF.md:30` | `git checkout 2bc0cd2` | resolves; `_production_tree` equals itself; best fixed target available | **MATCH** as a revision (D-1/D-2 for what it installs) |
| 4 | `PC_PILOT_HANDOFF.md:33-34` | branch may advance past `2bc0cd2`; `TESTED_SOURCE_SHA` is the invariant to compare against | `git ls-remote` tip = `2bc0cd2` today | **MATCH** |
| 5 | `FINAL_PC_RELEASE_VERDICT.md:9,22` | `TESTED_SOURCE_SHA = 2bc0cd2` | resolves; equals the invariant the handoff names | **MATCH** as a pair (D-1 for correctness against the working tree) |
| 6 | `FINAL_PC_RELEASE_VERDICT.md:10` | `FINAL_HEAD_SHA 2bc0cd2 … production files changed across 11ba3df..2bc0cd2` | `git diff --numstat` gives exactly those 5 files | **MATCH** |
| 7 | `FINAL_PC_RELEASE_VERDICT.md:23` | `FINAL_HEAD_SHA` = see `git rev-parse HEAD`; gates ran on `TESTED_SOURCE_SHA` | HEAD = `2bc0cd2`; but "the gates ran on it" is false for the pending changes | **MISMATCH** (P1, same root as D-1) |
| 8 | `FINAL_PC_RELEASE_VERDICT.md:152` | `git checkout FETCH_HEAD` | moving branch tip; unpinned install | **MISMATCH** (D-5, P2) |
| 9 | `FINAL_16_MODEL_REPRODUCIBILITY_REASSESSMENT.md:3` | `Tree: b70d092 + the Phase A fixes in this pass` | `b70d092` resolves; production tree **equals** `11ba3df`'s, so "Phase A fixes" must be read as uncommitted work — not stated | **HISTORICAL** (partial) |
| 10 | `FINAL_CIR_SCIENTIFIC_COMPARABILITY.md:4` | `HEAD b70d092` | resolves; ancestor; superseded | **HISTORICAL** |
| 11 | `FINAL_RTX5070TI_COMPATIBILITY.md:5` | `Repository HEAD: b70d092` | ditto | **HISTORICAL** |
| 12 | `FINAL_DEFECT_LEDGER.md:6` | "Tested source at the time of this ledger's original pass: 84ec6f1" | resolves; `_production_tree` of `84ec6f1` ≠ current | **HISTORICAL** (re-labelled by this working tree) |
| 13 | `FINAL_DEFECT_LEDGER.md:7` | `Baseline: 3aae18f` | resolves | **HISTORICAL** |
| 14 | `FINAL_DEFECT_LEDGER.md:355` | "**Fix**: both name `11ba3df`" | both now name `2bc0cd2` | **MISMATCH** (D-6, P3) |
| 15 | `FINAL_DEFECT_LEDGER.md:358-361` | guard asserts ancestry; negative control `4b32063` fails the test | test asserts production-tree equality; `4b32063` is an ancestor of both revisions → rc 0 | **MISMATCH** (D-6, P3) |
| 16 | `FINAL_16_MODEL_AUDIT_MATRIX.md:9` | "Tested source at the time of THIS matrix: 84ec6f1" | resolves; superseded | **HISTORICAL** |
| 17 | `FINAL_FIRST_PARTY_SOURCE_COVERAGE.md:7` | same, re-labelled | resolves | **HISTORICAL** |
| 18 | `FINAL_ADVERSARIAL_REVIEW.md:6,8` | `84ec6f1` + `3aae18f` baseline | resolve | **HISTORICAL** |
| 19 | `PC_PILOT_HANDOFF.md:34` | `TESTED_SOURCE_SHA` in the verdict is the invariant | reads as a cross-reference, not a literal SHA | **MATCH** |
| 20 | `PC_ONE_COMMAND_GUIDE.vi.md:70` | `git clone -b research/cir-failure-workbench …` | moving target; no SHA anywhere in the file (`grep` for 7-40 hex: none) | **MISMATCH** (P3, unguarded install path) |
| 21 | `PC_TRANSFER_CHECKLIST.md:85-86` | `git clone <repo>` / `git checkout <the frozen commit>` | placeholders; cannot be wrong, cannot be followed | **VARIANT** (no defect asserted) |
| 22 | `SOURCE_COVERAGE_MANIFEST.md` (126 rows) | line count + `sha256[:16]` per path | 45 rows mismatch (D-4) | **MISMATCH** (P2) |
| 23 | `FINAL_DEFECT_LEDGER.md:382`-class A/F-series prose | pass-specific trees named per entry | resolves where checked (`2ed4934`, `b55fc29`, `2bc0cd2`) | **HISTORICAL** |
| 24 | `FINAL_16_MODEL_RECOVERY_PRIORITY.md` (whole file) | no SHA, no count | verified by grep | **MATCH** — none claimed |
| 25 | `AUTOMATION_RUNBOOK.md` (whole file) | no repo SHA, no suite count | verified by grep | **MATCH** — none claimed |
| 26 | `READINESS_MATRIX.md:4-5` | `models: 16`; upstream root path | registry has 16 entries | **MATCH** |
| 27 | `FINAL_PC_READINESS.md:10-11` | base `70851dd3…`; `main` = `8ebcc74…` | both resolve; `main` exact | **HISTORICAL** |
| 28 | `final_pc_audit/FINAL_ADVERSARIAL_RELEASE_REPORT.vi.md:14-18` | `ddcb71f` tested, `0943042` docs-only, `1171 passed` | `_production_tree("ddcb71f") == _production_tree("0943042") == 86543635c4e6576a…` — the docs-only claim is **true** | **HISTORICAL** (permitted) |
| 29 | `final_pc_audit/FINAL_ADVERSARIAL_RELEASE_REPORT.vi.md:278` | `git reset --hard ddcb71f` | superseded target and the exact command the handoff forbids | **MISMATCH** as an install instruction (P3; historical campaign) |
| 30 | `final_pc_audit/iterative_audit/round_01/SOURCE_MANIFEST.md:3-4` | `SOURCE_SHA be083e62…`, `SOURCE_TREE 36c47b66…` | `git rev-parse be083e62^{tree}` → `36c47b66fd1729d4e8cce6995994f790c9dd6997` | **MATCH** (tree hash independently reproduced) |
| 31 | `final_model_audit/*`, `reviews/models/*`, `final_hardening/*`, `final_pc_audit/{A..F}.md` | audit-time HEADs (`5726d5f`, `eb36aa5`, `d9431fe`, `e9cafaf`, `0b0f3a6`, `6e3c979`, `92ecbd9`) | all resolve and are ancestors | **HISTORICAL** |
| 32 | `r8/R8_01_METADATA_AUDIT.md:89,213` | manifest row 20 stale; self-referential count test | row 20 stale **confirmed**; but 45 rows are stale (A5) | **MATCH** in substance, **understated** in scope |
| 33 | `r8/R8_03_DEV_CERT_TRACE.md:15,186` | `pc.sh:1637` at `2bc0cd2` checks only `is_research_evidence`+`certified` | `git show 2bc0cd2:pc.sh | sed -n 1637p` → exactly that; the development-mode refusal is at working-tree `:1769` | **MATCH** |
| 34 | `r8/R8_02_ARGV_INVENTORY.md:4,25-33` | `pc.sh` at `2bc0cd2` is 2232 lines; argv carriage at 1269/1306/1767/1788/1950 | `git show 2bc0cd2:pc.sh | wc -l` → 2232; those are the argv call sites | **MATCH** |

## 5b. Test counts

| # | file:line | claim | measured | verdict |
|---|---|---|---|---|
| 35 | `PC_PILOT_HANDOFF.md:9` | `1406 passed` on the tested tree | `2bc0cd2` **collects 1406** (`git archive` extract, `--collect-only`) | **MATCH** for `2bc0cd2`; the pass count is UNVERIFIED |
| 36 | `FINAL_PC_RELEASE_VERDICT.md:42` | `1406 passed, 0 failed` (25:43) | ditto | **MATCH** (collection); duration UNVERIFIED |
| 37 | working tree at the same moment | same documents | **1422 collected** (also 1419 earlier in the audit; file being edited) | **MISMATCH** for the tree under review (P2) |
| 38 | `final_pc_audit/*.vi.md` (`1171`, `1134`, `951`, `1030`) | pass-specific counts | not re-run; each names its own frozen tree | **HISTORICAL** |
| 39 | `FINAL_DEFECT_LEDGER.md:361` | "Negative control: restoring `git checkout 4b32063` fails the test" | no; `4b32063` is an ancestor of both candidate SHAs and no test pins the handoff to the tested SHA | **MISMATCH** (D-6) |

## 5c. `git checkout` / install instructions

| # | file:line | instruction | verdict |
|---|---|---|---|
| 40 | `PC_PILOT_HANDOFF.md:26-30` | clone, fetch branch, `git checkout 2bc0cd2`, with an explicit "never `git reset --hard`" warning | **MATCH** (fixed revision; correct warning) |
| 41 | `FINAL_PC_RELEASE_VERDICT.md:150-152` | clone, fetch, `git checkout FETCH_HEAD` | **MISMATCH** (D-5) |
| 42 | `PC_ONE_COMMAND_GUIDE.vi.md:70` | `git clone -b research/cir-failure-workbench` | **MISMATCH** (P3) |
| 43 | `PC_TRANSFER_CHECKLIST.md:85-86` | placeholder clone/checkout | **VARIANT** |
| 44 | `FINAL_ADVERSARIAL_RELEASE_REPORT.vi.md:278` | `git reset --hard ddcb71f` | **MISMATCH** (historical campaign, superseded target) |
| 45 | guard coverage of the same | `test_handoff_commands.py:365` regexes **only** `PC_PILOT_HANDOFF.md`; `:395` only bold-hex tokens in `RELEASE_DOCS` | rows 41 (bare `FETCH_HEAD`) and 42 (`-b <branch>`, no hex) are **not covered** |

## 5d. Exit-code lists

| # | file:line | claim | verified against `pc.sh` (working tree) | verdict |
|---|---|---|---|---|
| 46 | `PC_PILOT_HANDOFF.md:81-90` | 0 / 3 / 4 / 5 / 7 table, 4-or-7 for proof-unavailable | `sys.exit(3)` :2341, `(4)` :2345, `(5)` :2332, `(7)` :2339; `AGGREGATE_ONLY_*` tokens printed at :1812-1820 | **MATCH** (omits 6, consistent with `FINAL_PC_RELEASE_VERDICT.md:27`; guard input rejects 6 by design) |
| 47 | `FINAL_PC_RELEASE_VERDICT.md:27` | one-line list 0/3/4/5/6/7 with meanings | `sys.exit(6)` :2209 and :2214 (`PER_QUERY_VERIFIED` ↔ index disagreement) | **MATCH** |
| 48 | `PC_PILOT_HANDOFF.md:96-101` | 7 = authenticated, not certified; 5 = "unknown", not "absent"; 3 = aggregate reached, full pilot not ready | :2339 exit 7 after printing the not-certified conclusion; :1835-1838 `AGGREGATE_ONLY_EXECUTION_AUTHENTICATED_PROOF_UNAVAILABLE`; :1803-1808 "Unknown is not absent" → refusable | **MATCH** |
| 49 | `FINAL_PC_RELEASE_VERDICT.md:98-108` | launcher success/failure matrix incl. "the summary and the index disagree … 6" | `:2209`/`:2214` are exactly that check | **MATCH** |
| 50 | `R8_02_ARGV_INVENTORY.md:20` | E2BIG previously surfaced as shell rc 126 and aborted at `INDEX_VERIFY` | static read of `2bc0cd2:pc.sh:1573-1600` is consistent (bodies in argv); no execution performed | **UNVERIFIED** at runtime, consistent statically |

## 5e. `--verify-only` descriptions

| # | file:line | claim | verdict |
|---|---|---|---|
| 51 | `PC_PILOT_HANDOFF.md:187-189` | GPU probe only; does **not** verify a run, rebuild the index or download a checkpoint | **MATCH** to `pc.sh:2350-2367` (static) |
| 52 | `FINAL_PC_RELEASE_VERDICT.md:28` and `:160-163` | same sentence | **MATCH** |
| 53 | `PC_ONE_COMMAND_GUIDE.vi.md:231` | same sentence + "run `./pc.sh` without flags to verify an existing run and rebuild the index" | **MATCH** |
| 54 | `PC_PILOT_HANDOFF.md:51` | flag list `--model --force --doctor --verify-only --no-gpu --dataset --checkpoint --timeout --help` | all present in `pc.sh:199-222` (plus `--self-check`, which `pc.sh:193` help text lists) | **MATCH** |
| 55 | `PC_PILOT_HANDOFF.md:205` | `./pc.sh --self-check` is a documented step | `pc.sh:221` handles it; help at `:193` | **MATCH** |
| 56 | `AUTOMATION_RUNBOOK.md:120,125,129` | `--verify-only` for the **download/sync scripts** | each script defines it (`grep -c` → 1/1/1) | **MATCH** (different flag, different program) |
| 57 | guard coverage | `test_no_release_document_describes_verify_only_as_verifying_a_run` has a `"not "` exemption (`:491`) | any line containing "not " is exempt, including the whole paragraph it targets | **VARIANT** (P3, weak guard — pre-existing) |

---

# §6 — MODEL READINESS ROWS IN `R8_16_MODEL_STATUS.md` vs `workbench/registry/models.yaml`

Derived every column from the registry (`yaml.safe_load`, 16 entries) and from
`workbench/backend/adapters/models.py` / `workbench/backend/metrics_extraction.py`.

**Correct**: `protocol`, `src_avail`, `src_pinned`, `ckpt_variants`, `mapping_verified`,
`observer`, `remaining_blockers` — all 16 rows agree with the registry (protocols A/B/C assignments
match `native_protocol`; `ptha_mtst` has `source_available:false`, `upstream_commit_sha:null`, no
variants; only `limn` carries `per_query_observer: LIMN_QUERY_OBSERVER_V1`).

**Wrong class 1 — `native_evaluator` is stated in a vocabulary that exists nowhere in the repository.**
`R8_16_MODEL_STATUS.md:10` calls it "the workbench's own status field for the model's upstream
entrypoint". No such field exists:

```
$ grep -rn "native_evaluator" workbench --include=*.py   → (no output)
$ python -c "keys of every models.yaml entry"            → no 'native_evaluator'
```

The values written there (`CLI_READY` at `:30`, `COMMAND_AUDITED`, `COMMAND_BLOCKED_*`,
`REPLAY_COMMAND_IMPLEMENTED`, `NONE`) are the vocabulary of
`FINAL_16_MODEL_REPRODUCIBILITY_REASSESSMENT.md:33-34`, not of the registry. The registry's actual
field is `command_status` (present for 8 models, absent for 8) with a documented fallback in
`workbench/backend/registry.py:981-986`.

Consequentially, the **encoder** row (`:30`) is wrong on both cells it names:
`native_evaluator = CLI_READY` is 0 in the registry and `command_status` is **absent** for `encoder`
(`grep -n "command_status:"` → lines 509,579,635,692,736,813,875,934 only), while
`workbench/backend/adapters/models.py:37-44` shows `EncoderAdapter` **does** override `command`
(`grep` of the adapter registry: `encoder overrides_command=True script=evaluate_model.py`). Any
consumer resolving "CLI_READY" from this doc gets a status string that appears in three Markdown files
and in zero code paths.

**Wrong class 2 — the static level ignores a first-party, testable implementation that exists.**
`:46` says "Metric parser run at verification time | any model whose aggregate report passes the
verifier's metric rules". Measured, per model, by calling the real function:

```
$ .venv/bin/python -c "from workbench.backend.metrics_extraction import extract_aggregate_metrics; ..."
limn                    parser_id=limn_structured_json
csmcir                  parser_id=csmcir_stdout_json
combiner_rn50x4_noft    parser_id=clip4cir_stdout_recalls
clip4cir_rn50x4_fullft  parser_id=clip4cir_stdout_recalls
sprc                    parser_id=csmcir_stdout_json        # shared shape, deliberate (mx:406-411)
dcnet, encoder, habit, intent, hint, pair, clvc_net, tgcir, airknow, conesep, ptha_mtst
                        AGGREGATE_PARSER_UNAVAILABLE
```

So a parser exists for **5 of 16** models (`workbench/backend/metrics_extraction.py:402-411`), a fact
absent from the 16-row matrix, whose `static_lvl` column is `2` for every model including the five.
`R8_16_MODEL_STATUS.md:7` promises the document "states only what is implemented in this tree", so the
omission of implemented, callable parsers is a completeness defect, not a wording quibble. Severity
**P3** (documentation of capability, not a scientific-contract claim); the columns that decide
scientific behaviour are correct, which is the part the document claims about itself at `:70-74`.

---

# EXECUTION TRACE

| # | command (working directory) | observed |
|---|---|---|
| 1 | `git status --porcelain`, `git log --oneline -5`, `git rev-parse HEAD origin/main` | HEAD `2bc0cd2`, `origin/main` `8ebcc74…`; 12 modified + 1 untracked dir |
| 2 | `git diff --stat`, `git diff pc.sh`, `git diff …/evaluate_models.py`, `git diff …/test_handoff_commands.py`, `git diff …/test_pc_launcher.py`, `git diff workbench/docs` | 8 files, +682/−108; contents summarised in §CONFIRMED DEFECTS |
| 3 | `git diff --name-only 11ba3df 2bc0cd2` | 35 files (5 production, 30 docs/tests) |
| 4 | `git diff --numstat 11ba3df 2bc0cd2 -- . ':(exclude)workbench/docs' ':(exclude)workbench/tests'` | 101/22 `pc.sh`, 12/1 `evidence.ts`, 13/1 `limn_export.py`, 18/2 `limn_publish.py`, 23/5 `evaluate_models.py` |
| 5 | `git diff --numstat 2bc0cd2 -- . ':(exclude)…docs' ':(exclude)…tests'` | 229/58 `pc.sh`, 20/0 `evaluate_models.py` → D-1 |
| 6 | `git archive 2bc0cd2 \| tar -x -C /tmp/head_tree`; `pytest --collect-only -q` | **1406 collected** |
| 7 | working tree, same command (twice, 17:15 and 17:21) | **1419**, then **1422** collected |
| 8 | `git grep -c -h -E '^\s*def test_' {2bc0cd2,HEAD} -- workbench/tests/` | 1329 at `2bc0cd2`; 1344 in the working tree |
| 9 | `/home/heheboiz/data/cir/.venv/bin/python /tmp/mf.py` | 126 manifest rows checked, **45 mismatched** |
| 10 | clone → `/tmp/g2` (handoff `11ba3df`, tested `11ba3df`, count `1394`) | superseded-source test **passes** despite a 5-file production gap → A2 |
| 11 | clone → `/tmp/g3` (commit appending to `pc.sh`; both docs name `11ba3df`) | `git diff --numstat 11ba3df HEAD -- …` non-empty; guard file **7 passed** → D-2 |
| 12 | clone → `/tmp/g4` (tested `11ba3df`, handoff `2bc0cd2`, count `1329`) | superseded-source test passes although the parity property holds against `2bc0cd2`; only the count test fails → A2/A4 |
| 13 | clone → `/tmp/g5` (consistent obsolete pair: tested `11ba3df`, handoff `11ba3df`, count `1329`) | **7 passed** → D-2 |
| 14 | `git merge-base --is-ancestor 4b32063 {11ba3df,2bc0cd2}` | rc 0, rc 0 → D-6 |
| 15 | `git show {11ba3df,2bc0cd2}:pc.sh \| grep -c …` | `sys.exit(7)` 0→1, `REGISTRY_UNKNOWN` 0→5, `MAX_ARG_STRLEN` 0→0 (working tree 5) |
| 16 | `git rev-parse be083e62^{tree}` | `36c47b66fd1729d4e8cce6995994f790c9dd6997` — matches round_01/02 manifest |
| 17 | production-tree fingerprint (sha256 over `git ls-tree -r -z`) | `11ba3df` = `b70d092` = `49092d…`; `2bc0cd2` = `881c00…`; `ddcb71f` = `0943042` = `865436…`; `4b32063` = `17192e…` |
| 18 | `git ls-remote origin refs/heads/…` | branch tip `2bc0cd2`, `main` `8ebcc74…` |
| 19 | `yaml.safe_load` of `models.yaml`; `ADAPTERS` command-override probe; `extract_aggregate_metrics` per model | §6 |
| 20 | `git grep -rn … -- workbench/docs` sweeps for hex tokens, `git checkout`, `passed`, `--verify-only`, exit codes | §5 |
| 21 | `.venv/bin/python -m pytest workbench/tests/test_handoff_commands.py -q -p no:cacheprovider` (in the real repo) | 7 passed |
| 22 | `.venv/bin/python -m pytest workbench/tests/test_pc_launcher.py` (real repo, one run) | 120 s, timeout/no failure output — recorded as `NOT_RUN` (file still being written) |

No full-suite run. Nothing was written under `/home/heheboiz/data/cir/workbench/docs/` except this
file; every counterexample lives under `/tmp/{g2,g3,g4,g5,head_tree,recompa,t2}`.

---

# FINAL VERDICT

**VARIANT.**

- The revision metadata is *internally consistent* in a way the previous pass was not: the handoff now
  names a fixed SHA (`2bc0cd2`), that SHA's collection count (`1406`) is independently reproduced,
  the five-file production diff out of obsolete `11ba3df` is exactly as quoted, `origin/main` is
  unchanged, the exit-code lists match `pc.sh`, and `--verify-only` is described correctly in all five
  release documents. Those are real improvements, and seven of the eight allegations I could test in
  that direction were disproved.
- It is not `VERIFIED`, because the two claims this pass exists to make are still open: the metadata
  names a tree that does not contain the reviewed changes (**P1**), and the guard that is supposed to
  make the metadata self-checking is satisfiable by an obsolete recorded source (**P1**) and does not
  compare the handoff's checkout target to the tested source at all. `SOURCE_COVERAGE_MANIFEST.md` is
  stale in 45/126 rows (**P2**), the verdict's install block is unpinned (**P2**), and the count check
  compares sources rather than collection (**P2**).
- It is not `BLOCKED` (every claim was reachable read-only) and not `NOT_RUN` (twenty-two command
  groups were executed; three counterexample histories were built and observed).
