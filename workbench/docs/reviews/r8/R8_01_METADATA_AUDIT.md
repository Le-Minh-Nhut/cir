# R8-01 — release-metadata audit: superseded source SHA in the operator handoff

READ-ONLY audit. No production file was modified. This is the only file created.
Every claim below is `STATIC_AUDIT` unless marked `RUNTIME_VERIFIED` (a command was executed and its
output is quoted). No GPU inference was run; no runtime claim about model behaviour is made.

Baseline observed while writing this report:

```
$ git rev-parse HEAD; git rev-parse origin/research/cir-failure-workbench; git rev-parse origin/main
2bc0cd223028057b577b861d83d9e2795b040840
2bc0cd223028057b577b861d83d9e2795b040840
8ebcc74d5fd1934df4f3772a8ca5a96dbd00f333
$ git rev-parse --abbrev-ref HEAD
research/cir-failure-workbench
$ git status --porcelain | wc -l
0
```

So the assigned ground truth is confirmed exactly: HEAD `2bc0cd2`, remote branch `2bc0cd2`,
`origin/main` `8ebcc74`. Worktree clean at the moment of measurement. `RUNTIME_VERIFIED`.

Note on a concurrent change (I did not cause it, and I did not revert it): at the end of this audit
`git status --porcelain` reported `M pc.sh` — a modification to the launcher that appeared while this
audit was running. `git rev-parse HEAD` and `git rev-parse origin/research/cir-failure-workbench` still
return `2bc0cd2…` both before and after, so **every commit-referenced measurement in this report is
unaffected** (all evidence is taken from `11ba3df`, `HEAD`, `b70d092`, `2ed4934`, `b55fc29` and
`origin/main`, never from the working tree). The two claims that were taken from the working tree —
`sha256`/line-count spot checks in §1a and the `1406 tests collected` figure in §4 — were measured
before that edit appeared; `workbench/tests/test_handoff_commands.py` still hashes to
`9f90cbc16aa16cf8` and is still 369 lines, so §1a row 20 stands after the edit. I left the working-tree
change exactly as I found it.

---

## 1. Claim table — every SHA / test-count / checkout claim under `workbench/docs/`

Discovery method (not a hand-picked list): every line of every `workbench/docs/**/*.md` that contains a
hex token of 7–40 chars was extracted together with its surrounding line, then classified by whether the
line carries commit/checkout/`HEAD`/tested/baseline context. The enumeration below is the result; the
four named documents are marked, the rest were found by the sweep.

Classification: **CURRENT** = written as the tree to install / the state now; **HISTORICAL** = explicitly
labelled as a past pass, freeze, baseline or initial HEAD.

### 1a. Claims presented as CURRENT (defect class)

| # | file:line | claim as written | what git says | MATCH / MISMATCH |
|---|---|---|---|---|
| 1 | `workbench/docs/PC_PILOT_HANDOFF.md:8` | `**SHA ĐÃ KIỂM THỬ (nguồn sản xuất): `11ba3df`** — mọi test, acceptance, mutation và release_verify mô tả trên trang này đã chạy trên đúng cây đó` | `11ba3df` resolves and is an ancestor of HEAD, but the branch tip is `2bc0cd2` and **production differs** between the two (§2) | **MISMATCH** as the tree the page hands the operator |
| 2 | `workbench/docs/PC_PILOT_HANDOFF.md:9` | `(1394 passed; …)` | HEAD tree collects **1406** (`pytest --collect-only`, §4); the `11ba3df` tree has 12 fewer `def test_` (1317 vs 1329 static count, §4) | **MISMATCH / UNREPRODUCIBLE** — 1394 is not the count of either tree |
| 3 | `workbench/docs/PC_PILOT_HANDOFF.md:10-11` | ``Commit `b70d092` sau nó **chỉ sửa tài liệu**: `git diff 11ba3df b70d092` trên `pc.sh`, `workbench/`, `src/` cho ra **0 dòng**`` | reproduced: only `workbench/docs/reviews/FINAL_PC_RELEASE_VERDICT.md`, `2 +2 -2`, rc=0 | **MATCH** (as written) |
| 4 | `workbench/docs/PC_PILOT_HANDOFF.md:23` | `git checkout 11ba3df` | resolves; ancestor of HEAD; but its production tree ≠ `2bc0cd2`'s in 5 files | **MATCH as a revision, MISMATCH as an install target** |
| 5 | `workbench/docs/PC_PILOT_HANDOFF.md:26` | `Cây mới nhất (gồm cả tài liệu cập nhật) là `b70d092`; khác biệt so với `11ba3df` chỉ là tài liệu.` | newest commit on the branch is `2bc0cd2`, not `b70d092` | **MISMATCH** |
| 6 | `workbench/docs/reviews/FINAL_PC_RELEASE_VERDICT.md:9` | ``**TESTED_SOURCE_SHA** | **`11ba3df`** (production source; every gate in this pass ran on it)`` | as #1 | **MISMATCH** as a current record |
| 7 | `workbench/docs/reviews/FINAL_PC_RELEASE_VERDICT.md:10` | ``**FINAL_HEAD_SHA** | `11ba3df` — the stabilization pass.`` | no ref equals `11ba3df`; HEAD is `2bc0cd2` | **MISMATCH** |
| 8 | `workbench/docs/reviews/FINAL_PC_RELEASE_VERDICT.md:22` | ``TESTED_SOURCE_SHA | `11ba3df` — every gate below ran on it`` | as #1 | **MISMATCH** |
| 9 | `workbench/docs/reviews/FINAL_PC_RELEASE_VERDICT.md:23` | ``FINAL_HEAD_SHA | see `git rev-parse HEAD` on the branch tip; everything after `11ba3df` is documentation and tests`` | `git rev-parse HEAD` → `2bc0cd2`; **3 of the 4** commits after `11ba3df` touch production (`2ed4934` pc.sh, `b55fc29` 5 files, `2bc0cd2` limn_export.py) | **MISMATCH** — and it contradicts line 10 of the same table |
| 10 | `workbench/docs/reviews/FINAL_PC_RELEASE_VERDICT.md:35` | `The full verification ran against `11ba3df`. The only commit after it is documentation only, and the production diff for it is recorded at the end.` | 4 commits after `11ba3df`; only `b70d092` is docs-only; the "production diff … recorded at the end" **does not exist** — the file is 191 lines and ends at the "next step" paragraph | **MISMATCH** (two false statements) |
| 11 | `workbench/docs/reviews/FINAL_PC_RELEASE_VERDICT.md:40` | `pytest workbench/tests -q` → ``**1394 passed, 0 failed** (26:08)`` | HEAD collects 1406; delta between trees = +12 tests | **MISMATCH / UNREPRODUCIBLE** |
| 12 | `workbench/docs/reviews/FINAL_PC_RELEASE_VERDICT.md:148-150` | `git clone git@github.com:Le-Minh-Nhut/cir.git && cd cir` / `git fetch origin research/cir-failure-workbench` / `git checkout FETCH_HEAD` | after that fetch, `FETCH_HEAD` = branch tip = `2bc0cd2` — an unpinned, moving target | **MISMATCH** as an install instruction (no frozen SHA), and **not covered by the guard test** (§3) |
| 13 | `workbench/docs/PC_ONE_COMMAND_GUIDE.vi.md:70` | `git clone -b research/cir-failure-workbench https://github.com/Le-Minh-Nhut/cir` | clones the moving branch tip; no SHA anywhere in the document (grep: no `[0-9a-f]{7,}` commit and no `TESTED_SOURCE_SHA`) | **MISMATCH** as an install instruction |
| 14 | `workbench/docs/reviews/FINAL_DEFECT_LEDGER.md:6` | `**Tested source**: `84ec6f1` plus the fixes listed here. **Baseline**: `3aae18f`.` | both exist and are ancestors; **8** production commits separate `84ec6f1` from HEAD | **MISMATCH** (stale current-tense claim; no install instruction attached) |
| 15 | `workbench/docs/reviews/FINAL_DEFECT_LEDGER.md:356-358` | `the checkout target must be an ancestor of the tested source, which must be an ancestor of HEAD. Negative control: restoring git checkout 4b32063 fails the test.` | the first sentence is exactly what the test asserts (**MATCH**); the negative control is **false**: `git merge-base --is-ancestor 4b32063 11ba3df` → rc=0 | **MISMATCH** on the negative control |
| 16 | `workbench/docs/reviews/FINAL_16_MODEL_AUDIT_MATRIX.md:9` | `- **Tested source**: `84ec6f1` (the production tree every gate ran against)` | as #14 | **MISMATCH** (stale) |
| 17 | `workbench/docs/reviews/FINAL_FIRST_PARTY_SOURCE_COVERAGE.md:7` | `- **Tested source**: `84ec6f1`` | as #14 | **MISMATCH** (stale) |
| 18 | `workbench/docs/reviews/FINAL_ADVERSARIAL_REVIEW.md:6` | `**Tested source**: `84ec6f1` + the fixes in `FINAL_DEFECT_LEDGER.md`` | as #14 | **MISMATCH** (stale) |
| 19 | `workbench/docs/reviews/final_pc_audit/FINAL_ADVERSARIAL_RELEASE_REPORT.vi.md:278` | `git reset --hard ddcb71f          # TESTED_SOURCE_SHA (nguồn sản xuất đã kiểm thử)` | `ddcb71f` exists, ancestor of HEAD, superseded; and it is a `git reset --hard` — the exact command `PC_PILOT_HANDOFF.md:21` forbids | **MISMATCH** as an install instruction |
| 20 | `workbench/docs/SOURCE_COVERAGE_MANIFEST.md:89` | ``| `workbench/tests/test_handoff_commands.py` | workbench | 205 | `855c29b0ba42f90f` | TESTED_PATH |`` | actual: 369 lines, `sha256[:16]` = `9f90cbc16aa16cf8` | **MISMATCH** — the manifest's digest of the very test that is supposed to guard this metadata is stale |

Spot-check of that manifest's digest recipe against three neighbours (all `sha256[:16]`, `RUNTIME_VERIFIED`):

| row | manifest claim | measured now | verdict |
|---|---|---|---|
| `test_operator_config.py` | 44 / `bb7b6d227620607c` | 44 / `bb7b6d227620607c` | MATCH (recipe confirmed) |
| `test_credential_boundary.py` | 157 / `665c3a345a25977e` | 224 / `71fe1123c145540a` | MISMATCH |
| `test_evaluate_models.py` | 514 / `fac01b77dfd45233` | 1091 / `533352916987aa9c` | MISMATCH |
| `test_handoff_commands.py` | 205 / `855c29b0ba42f90f` | 369 / `9f90cbc16aa16cf8` | MISMATCH |

### 1b. Claims that are explicitly historical (permitted — they name a past pass)

| # | file:line | claim | git truth | verdict |
|---|---|---|---|---|
| 21 | `workbench/docs/reviews/FINAL_PC_RELEASE_VERDICT.md:8` | `Initial HEAD this pass: 3aae18fe02ef13f5b38d153dbc8007fbb40d4c92` | resolves; ancestor of HEAD; commit message "docs: state that 84ec6f1 is the tested production source" | MATCH |
| 22 | `workbench/docs/reviews/FINAL_PC_RELEASE_VERDICT.md:11,24` | `origin/main 8ebcc74d5fd1934df4f3772a8ca5a96dbd00f333 — unchanged` | **exact match**, unchanged | MATCH |
| 23 | `workbench/docs/reviews/FINAL_PC_RELEASE_VERDICT.md:12` | `Worktree clean` | `git status --porcelain` empty | MATCH |
| 24 | `workbench/docs/reviews/FINAL_DEFECT_LEDGER.md:351-354` | A-01 description: handoff said `git checkout 4b32063` while gates ran on `11ba3df` | consistent with git: `4b32063` is an ancestor of `11ba3df`; both are ancestors of HEAD. It describes a **past** file state (confirmed: `git show 11ba3df:workbench/docs/PC_PILOT_HANDOFF.md` really did say `4b32063`) | MATCH as historical |
| 25 | `workbench/docs/FINAL_PC_READINESS.md:10-11` | `Base commit for this document: 70851dd3…`; `main remains 8ebcc74d…` | both resolve; `70851dd3` is an ancestor of HEAD; `main` matches | MATCH |
| 26 | `workbench/docs/reviews/final_pc_audit/FINAL_ADVERSARIAL_RELEASE_REPORT.vi.md:14-18` | `TESTED_SOURCE_SHA ddcb71f`; `RELEASE_COMMIT_SHA 0943042` ("chỉ tài liệu"); `cf52311`; `a2861b2`; `1171 passed` | all SHAs resolve and are ancestors; **`git diff ddcb71f 0943042` over production = 0 files** — the "docs only" claim is true | MATCH as historical |
| 27 | `workbench/docs/reviews/final_pc_audit/iterative_audit/round_01/SOURCE_MANIFEST.md:3-4` (and `round_02`) | `SOURCE_SHA: be083e6264ebb0b12442a97d6bfefa57fc0207b4` / `SOURCE_TREE: 36c47b66fd1729d4e8cce6995994f790c9dd6997` | `git rev-parse be083e62^{tree}` → `36c47b66fd1729d4e8cce6995994f790c9dd6997` | MATCH exactly (tree hash independently reproduced) |
| 28 | `workbench/docs/reviews/final_pc_audit/iterative_audit/round_01/ROOT_CAUSE_MATRIX.md:7` | `Frozen tree: a2861b2. Consolidated fix commit: 04ae2ad.` | both resolve; ancestors | MATCH as historical |
| 29 | `workbench/docs/reviews/final_pc_audit/iterative_audit/round_01/AUDIT_A.md:5` | `HEAD 92ecbd94052bc7b4829abd4314af8b63d7d6bea3` | resolves (`92ecbd9`), ancestor | MATCH |
| 30 | `workbench/docs/PC_TRANSFER_CHECKLIST.md:85-86` | `git clone <repo> && cd cir` / `git checkout <the frozen commit>` | placeholders, no literal SHA — cannot be wrong, but also cannot be followed | VARIANT (no defect asserted) |
| 31 | `workbench/docs/AUTOMATION_RUNBOOK.md` (whole file, 723 lines) | no repo-level SHA, no test count | verified by grep: the only hex-like tokens are model pins (`65fa78ea` as a *pin-mismatch example string*) and `SHA-256` prose | MATCH — none claimed |
| 32 | `workbench/docs/reviews/final_model_audit/*.md`, `.../models/*.md` | audit-time HEADs `5726d5f`, `eb36aa5`, `d9431fe` and upstream model pins (`7d7bc9b1`, `65fa78ea`, `29a2a31d`, …) | all audit HEADs and all *repo-local* SHAs resolve; upstream pins are a different metadata class (`upstream_commit_sha`, §5) and were verified against `workbench/registry/models.yaml` by their own documents | no repo-release defect found here |

Counted: **20 CURRENT-class claims, 12 of which are MISMATCH** (rows 1–20 above); the historical set (rows
21–32) contains no mismatch except the false negative-control sentence in row 15, which is a claim about
the *test*, not about a commit.

---

## 2. Is `11ba3df` a docs-only ancestor, or an obsolete source? — git-diff verdict

Exact command (production = everything outside `workbench/docs/` and `workbench/tests/`):

```
$ git diff 11ba3df HEAD -- . ':(exclude)workbench/docs' ':(exclude)workbench/tests'
```

Output — **NOT empty** (`RUNTIME_VERIFIED`):

```
 pc.sh                                | 123 ++++++++++++++++++++++++++++-------
 workbench/frontend/evidence.ts       |  13 +++-
 workbench/replay/limn_export.py      |  14 +++-
 workbench/replay/limn_publish.py     |  20 +++++-
 workbench/scripts/evaluate_models.py |  28 ++++++--
 5 files changed, 167 insertions(+), 31 deletions(-)
```

Numstat of the same command: `101 22 pc.sh`, `12 1 workbench/frontend/evidence.ts`, `13 1
workbench/replay/limn_export.py`, `18 2 workbench/replay/limn_publish.py`, `23 5
workbench/scripts/evaluate_models.py`.

Per-commit attribution of the four commits between `11ba3df` and HEAD (`RUNTIME_VERIFIED`):

```
$ for c in $(git rev-list 11ba3df..HEAD); do echo "$c $(git diff --numstat $c^ $c \
    -- . ':(exclude)workbench/docs' ':(exclude)workbench/tests' | wc -l) $(git log -1 --format=%s $c)"; done
2bc0cd2 1 fix: the planned-digest assignment belonged where the other parent-owned values are set
b55fc29 5 fix: close the LIMN-audit defects and record the 16-model reassessment
2ed4934 1 fix: close four release defects found by the PC-handoff audit
b70d092 0 docs: record the stabilization pass in the release verdict
```

**Verdict: `11ba3df` is an OBSOLETE SOURCE, not a docs-only ancestor.** Exactly one of the four later
commits (`b70d092`) is documentation-only. Three touch production. The `test_no_release_document_points_at_a_superseded_source_sha`
docstring's own premise — "the tested SHA must be an ancestor of HEAD", i.e. "an ancestor is therefore
safe" — is refuted by the repository's own history.

Control measurement, same command shape over other candidate pairs (`RUNTIME_VERIFIED`):

| pair | production files changed |
|---|---|
| `11ba3df..b70d092` | **0** (this is the pair the handoff quotes, and it is honest) |
| `11ba3df..2ed4934` | 1 |
| `11ba3df..b55fc29` | 5 |
| `11ba3df..2bc0cd2` (HEAD) | 5 |
| `ddcb71f..0943042` | 0 |
| `4b32063..11ba3df` | 5 |
| `84ec6f1..11ba3df` | 10 |

The defect is **not** an arithmetic slip: the handoff and the verdict both quote `11ba3df` as the tree to
install, while the fixes recorded on the same pages live in `2ed4934`, `b55fc29` and `2bc0cd2`.

### The concrete consequence (why this is P1, not cosmetic)

The handoff documents behaviour that the tree it tells the operator to check out **does not contain**:

| token | `git show 11ba3df:pc.sh` | `workbench/docs/PC_PILOT_HANDOFF.md` | HEAD `pc.sh` |
|---|---|---|---|
| `sys.exit(7)` | **0** occurrences | documented as the exit code for `AGGREGATE_ONLY_EXECUTION_AUTHENTICATED_PROOF_UNAVAILABLE` (:47) | 1 |
| `AGGREGATE_ONLY_EXECUTION_AUTHENTICATED_PROOF_UNAVAILABLE` | 0 | 1 | 6 |
| `REGISTRY_UNKNOWN` / `REGISTRY_CONFIRMED_AGGREGATE_ONLY` | 0 / 0 | described as the "unknown is not absent" contract (:50) | 5 / 2 |
| `reserved development prefix` in `evaluate_models.py` | 0 | (the `dev-` classification fix) | 1 |
| `planned_bundle_manifest_digest` in `limn_publish.py` | 0 | (the bundle-identity fix) | 1 |

An operator who follows `PC_PILOT_HANDOFF.md:23` installs a launcher that can never print the exit code
the same page defines, and a publisher that does not compare the planned bundle digest. This is the
`verify-only`-class defect the guard test was written for, in mirror image.

---

## 3. The guard test — what it asserts, what it does not, and the counterexample

### 3.1 Statement of the test

`workbench/tests/test_handoff_commands.py:301-338` (`test_no_release_document_points_at_a_superseded_source_sha`).
Assertions verbatim, in order:

1. `:312-313` reads `head = git rev-parse HEAD`.
2. `:316-317` extracts the tested SHA from `FINAL_PC_RELEASE_VERDICT.md` with
   `r"\*\*TESTED_SOURCE_SHA\*\* \| \*\*`([0-9a-f]{7,40})`\*\*"`; asserts a match exists.
3. `:319-321` asserts `git rev-parse <tested_sha>` succeeds.
4. `:323-327` asserts `git merge-base --is-ancestor <tested_sha> <head>` → rc 0.
5. `:329-331` extracts `re.findall(r"^git checkout ([0-9a-f]{7,40})", handoff, re.M)` from
   `PC_PILOT_HANDOFF.md`; asserts the list is non-empty.
6. `:332-338` for each: asserts it resolves, and asserts
   `git merge-base --is-ancestor <checkout_sha> <tested_sha>` → rc 0.

Observed dynamic extraction (`RUNTIME_VERIFIED`):

```
TESTED regex: 11ba3df
count regex (verdict): ['1394']
checkout regex (handoff): ['11ba3df']
verdict checkout regex hits: []            # <-- `git checkout FETCH_HEAD` is not matched
handoff bold count matches: []             # <-- handoff's 1394 is written unbolded, unparsed
handoff any 'N passed': ['1394']
```

### 3.2 What it fails to assert (each item is a hole, with the reason it exists)

| # | missing assertion | why it matters | evidence |
|---|---|---|---|
| a | `tested_sha == head`, or `== origin/<branch>`, or `== the shipped checkout target` | ancestry is satisfied by any ancestor, including an obsolete one | `:323-327` and `:335-338` are the only git assertions in the function |
| b | that the **tree at** the tested SHA equals the working tree **in any path** — there is not one `git diff` in the whole test | the property that actually protects the operator is production-tree equality | `grep -n 'diff' workbench/tests/test_handoff_commands.py` → no `git diff` invocation |
| c | that production paths are equal between the recorded source and the shipped commit | this is R8-01 itself | §2: 5 production files differ while the test passes |
| d | that the recorded test count corresponds to any run of any tree | `1394` is bound to nothing | §4 |
| e | any check of `PC_ONE_COMMAND_GUIDE.vi.md` (or the verdict's own commands) for a checkout/fetch target — although the guide **is** in `RELEASE_DOCS` at `:296`, this function never reads it | row 13 and row 12 above are unchecked | `:314` and `:329` read only two literals: the verdict and the handoff |
| f | `git checkout FETCH_HEAD` (`FINAL_PC_RELEASE_VERDICT.md:150`) — the regex requires a literal hex SHA | the verdict's own second command installs the moving branch tip | `verdict checkout regex hits: []` |
| g | that the handoff's prose count equals the verdict's count | `test_the_verdict_names_the_same_test_count_the_handoff_does` (`:363-368`) compares the verdict **to itself** (`len(set(counts)) == 1`, `:368`); the handoff's `1394` is never parsed | `:366` regex `\*\*(\d+) passed, 0 failed\*\*` matched only in the verdict; handoff writes `(1394 passed; …)` |

Two further notes that make the guard weaker than its docstring claims:

* `RELEASE_DOCS` (`:294-298`) lists three documents; `FINAL_DEFECT_LEDGER.md` — the document whose
  regression record points at this test — is **not** in it, so the false negative-control sentence
  (§1 row 15) is untested.
* `test_no_release_document_describes_verify_only_as_verifying_a_run` (`:341-360`) has a
  `if "không" not in lowered and "does not" not in lowered and "not " not in lowered` escape (`:357`):
  any line merely containing "not " is exempt, which is most of the paragraphs it is aimed at.
  (Recorded here because it shares the `RELEASE_DOCS` failure mode; not separately pursued.)

### 3.3 Counterexample — live, then synthetic

**Live (`RUNTIME_VERIFIED`).** The current repository *is* the counterexample: recorded source
`11ba3df`, shipped commit `2bc0cd2`, production differs by 5 files, and the test passes.

```
$ .venv/bin/python -m pytest workbench/tests/test_handoff_commands.py -p no:cacheprovider -q \
    -k 'superseded_source_sha or same_test_count'
..                                                                       [100%]
2 passed, 3 deselected in 0.06s

$ git diff --numstat 11ba3df HEAD -- . ':(exclude)workbench/docs' ':(exclude)workbench/tests'
101	22	pc.sh
12	1	workbench/frontend/evidence.ts
13	1	workbench/replay/limn_export.py
18	2	workbench/replay/limn_publish.py
23	5	workbench/scripts/evaluate_models.py
```

**Synthetic, built in `/tmp` with plumbing only — no commit on the branch (`RUNTIME_VERIFIED`).**
Shapes the history so that production differs between the recorded source and the shipped commit while
every assertion of the guard still holds:

```
$ git log --oneline
6c309bb c4: docs only
cf3588b c3: docs only
5bad013 c2: production FIX (production.py changed)
bd13ccc c1: tested source, production.py v1
$ git merge-base --is-ancestor bd13ccc 6c309bb; echo $?      # assertion 4 (tested -> HEAD)
0
$ git merge-base --is-ancestor bd13ccc bd13ccc; echo $?      # assertion 6 (checkout -> tested)
0
$ git diff --stat bd13ccc 6c309bb -- production.py
 production.py | 2 +-
 1 file changed, 1 insertion(+), 1 deletion(-)
```

Both ancestor predicates are `0` (pass) and production moved. `bd13ccc` plays the role of `11ba3df`,
`5bad013` of `2ed4934`/`b55fc29`/`2bc0cd2`, `cf3588b`/`6c309bb` of `b70d092`. Nothing in the test's
six assertions distinguishes the fixed history from the broken one — because none of them compares a
tree.

### 3.4 A finding in the opposite direction: the recorded negative control is false

`FINAL_DEFECT_LEDGER.md:358` claims `restoring git checkout 4b32063 fails the test`. It does not:

```
$ git merge-base --is-ancestor 4b32063 11ba3df; echo $?
0
```

With `git checkout 4b32063` restored at `PC_PILOT_HANDOFF.md:23` and `TESTED_SOURCE_SHA 11ba3df` kept at
`FINAL_PC_RELEASE_VERDICT.md:9`, assertion 6 (`4b32063` ancestor of `11ba3df`) still returns 0 and the
test still passes. The recorded "regression test fails if the fix is reverted" therefore does not hold
for the very defect it was written to pin. **Severity P2** — the ledger's own standard ("a test which
fails if the fix is reverted") is not met here. Note also that at `11ba3df` the test **does not exist**
(`git show 11ba3df:workbench/tests/test_handoff_commands.py | grep -c test_no_release_document_points…`
→ 0); it first appears in `2ed4934`. The negative control is unrunnable in the tree it names as tested.

---

## 4. The test-count claim

`RUNTIME_VERIFIED`, current tree:

```
$ .venv/bin/python -m pytest workbench/tests -p no:cacheprovider -q --collect-only
1406 tests collected in 1.33s
```

Static per-revision count of test functions (`git show <rev>:<file>` over
`git ls-tree -r <rev> -- workbench/tests`, counting `^\s*def test_`):

| revision | static `def test_` count |
|---|---|
| `11ba3df` | 1317 |
| `2bc0cd2` (HEAD) | 1329 |

Difference: **+12** test functions between the recorded source and HEAD — the number `1394` matches
neither, and no run artifact for `11ba3df` is stored in the repository that would settle it
(`grep -rn '1406 passed\|1394' --include='*.md' --include='*.py' --include='*.sh' --include='*.txt' .`
returns only the two prose citations themselves). The ground-truth `1406 passed` for the current tree
is consistent with the collection count above; the documents' `1394` is stale by construction, and the
test that is supposed to catch a contradiction (`:363-368`) compares the verdict only against itself
(§3.2 g). **Severity P2.**

---

## 5. Complete consumer list for a one-place fix

Everything that reads or states release/commit metadata, so the repair has a single owner per class.
`class A` = repo-release metadata (this defect); `class B` = model-pin metadata (`upstream_commit_sha`,
a different record, listed so it is not accidentally folded in).

**Class A — readers of repo-release metadata**

| consumer | location | what it reads | note |
|---|---|---|---|
| guard test | `workbench/tests/test_handoff_commands.py:294-298` (`RELEASE_DOCS`) | 3 document paths | the natural single registry for "which documents are release documents" |
| guard test | `workbench/tests/test_handoff_commands.py:301-338` | verdict `TESTED_SOURCE_SHA`; handoff `git checkout <sha>` | the failing guard |
| guard test | `workbench/tests/test_handoff_commands.py:363-368` | verdict `**N passed, 0 failed**` | self-referential; never reads the handoff's count |
| command-audit test | `workbench/tests/test_handoff_commands.py:19-32` (`DOCS`) | `PC_REPRODUCTION_HANDOFF.md`, `AUTOMATION_RUNBOOK.md`, `workbench/README.md`, `PC_PILOT_HANDOFF.md`, `PC_ONE_COMMAND_GUIDE.vi.md` | parses every ```bash block; does **not** check SHAs. Note the overlap: `AUTOMATION_RUNBOOK.md` is audited here but is **absent** from `RELEASE_DOCS` |
| launcher contract tests | `workbench/tests/test_pc_launcher.py:1786-1802`, `:1804-1817` | `PC_PILOT_HANDOFF.md` flag list vs `pc.sh` and `--help` | doc↔code agreement, not SHAs |
| launcher contract test | `workbench/tests/test_pc_launcher.py:1850-1860` | `PC_PILOT_HANDOFF.md` exit codes and per-query tokens vs `pc.sh` | **this is the test that would have caught §2's consequence, and it cannot** — it asserts only that `sys.exit(3/4/5)` and four state tokens exist in `pc.sh`, not that the *documented* exit 7 exists in the tree being shipped |
| launcher runtime identity | `pc.sh:277-280`, `:676-690` | `git rev-parse HEAD` of the checkout, `docker inspect "cir.commit"` label | runtime counterpart of `TESTED_SOURCE_SHA`; refuses a reused container from another commit |
| launcher doc pointers | `pc.sh:53`, `:1773`, `:1903` | hard-coded `workbench/docs/PC_ONE_COMMAND_GUIDE.vi.md` (3 sites) | the guide is referenced by the code, so it is a release document by contract and must stay in `RELEASE_DOCS` |
| backend schema prose | `workbench/backend/schemas/results.py:270` | comment naming the `PC_PILOT_HANDOFF` admission conditions | comment only |
| docs stating a current/`FINAL_HEAD` SHA | `PC_PILOT_HANDOFF.md:8-11,23,26`; `reviews/FINAL_PC_RELEASE_VERDICT.md:9,10,22,23,35,40,148-150` | the claims of §1 | the two documents the fix must touch |
| docs stating a stale tested SHA (no install instruction) | `reviews/FINAL_DEFECT_LEDGER.md:6`; `reviews/FINAL_16_MODEL_AUDIT_MATRIX.md:9`; `reviews/FINAL_FIRST_PARTY_SOURCE_COVERAGE.md:7`; `reviews/FINAL_ADVERSARIAL_REVIEW.md:6` | `84ec6f1` | second-order staleness |
| historical pass reports | `reviews/final_pc_audit/FINAL_ADVERSARIAL_RELEASE_REPORT.vi.md:14-18,275-278`; `.../iterative_audit/round_0{1,2}/SOURCE_MANIFEST.md`, `.../round_01/ROOT_CAUSE_MATRIX.md`, `.../round_01/AUDIT_A.md`; `reviews/final_model_audit/*.md`; `reviews/final_audit/*.md` | older SHAs | labelled historical; one (`:278`) teaches `git reset --hard` |
| per-file digest manifest | `workbench/docs/SOURCE_COVERAGE_MANIFEST.md` (382 rows) | line count + `sha256[:16]` per tracked file, incl. `test_handoff_commands.py:89` | stale for 3 of 4 spot-checked rows; it has no generator in the repo (`grep -rln SOURCE_COVERAGE_MANIFEST --include=*.py workbench/` → none), so it is hand-maintained |
| placeholder clone doc | `PC_TRANSFER_CHECKLIST.md:85-86` | `git checkout <the frozen commit>` | no literal SHA |
| branch-clone doc | `PC_ONE_COMMAND_GUIDE.vi.md:70` | `git clone -b research/cir-failure-workbench` | installs the moving tip |

**Class B — model-pin metadata (not this defect; listed for disambiguation)**

`workbench/registry/models.yaml` (`upstream_commit_sha`, read by `doctor.py:45-64`,
`verify_run.py:86`, `pipeline.py:961`, `readiness_matrix.py:121`, `download_auxiliary_assets.py`,
`sync_upstreams.py:37,113,165,198`) and the runtime pin check `workbench/backend/runtime.py`
`source_clean_and_pinned` (documented at `AUTOMATION_RUNBOOK.md:272-289`). These pins are per-model
upstream checkouts and are enforced at launch; they are structurally different from the repo-release
SHA and must not be collapsed into it.

### Fix in one place — what the single owner should check

The property that protects the operator is **production-tree equality**, not ancestry. The lazy
correct assertion set, all in `test_no_release_document_points_at_a_superseded_source_sha` (or one
shared helper it calls):

1. `tested_sha == git rev-parse <shipped ref>` (or == `origin/<branch>` when the branch is the ship
   target), and
2. `git diff --quiet <checkout_target> <tested_sha> -- . ':(exclude)workbench/docs' ':(exclude)workbench/tests'`
   must be empty, and
3. the `RELEASE_DOCS` tuple must gain every document that carries a checkout, a `FETCH_HEAD` or a
   tested SHA (`FINAL_DEFECT_LEDGER.md`, `AUTOMATION_RUNBOOK.md`, `PC_ONE_COMMAND_GUIDE.vi.md`), with
   `git checkout FETCH_HEAD` and `-b <branch>` clone lines rejected by the same rule, and
4. the recorded count must be compared **across** documents and bound to a run, not compared to itself.

That is the one-place owner: the guard test's `RELEASE_DOCS` tuple plus the two assertions above. The
documents then only need to name `2bc0cd2` (or whatever is the frozen release) consistently.

---

## 6. Execution trace

| step | command (abridged) | observed |
|---|---|---|
| 1 | `git rev-parse HEAD origin/research/cir-failure-workbench origin/main` | `2bc0cd2…`, `2bc0cd2…`, `8ebcc74…` |
| 2 | `git status --porcelain \| wc -l` | `0` |
| 3 | sweep of every `workbench/docs/**/*.md` for `[0-9a-f]{7,40}` with commit/checkout context | 32 claim rows (§1) |
| 4 | `git diff 11ba3df HEAD -- . ':(exclude)workbench/docs' ':(exclude)workbench/tests'` | 5 production files, `+167 −31` → **obsolete source** |
| 5 | per-commit attribution over `git rev-list 11ba3df..HEAD` | `b70d092` 0 production files; `2ed4934` 1; `b55fc29` 5; `2bc0cd2` 1 |
| 6 | control pairs (`ddcb71f..0943042`, `4b32063..11ba3df`, `84ec6f1..11ba3df`) | 0 / 5 / 10 |
| 7 | `git show <rev>:pc.sh \| grep -c …` for exit 7, `REGISTRY_UNKNOWN`, `AGGREGATE_ONLY_EXECUTION_AUTHENTICATED_PROOF_UNAVAILABLE` | 0 at `11ba3df`; 1 / 5 / 6 at HEAD → handoff documents behaviour absent from the installed tree |
| 8 | read `test_handoff_commands.py:294-369`; regex probes | 1 ancestor assertion per SHA; `FETCH_HEAD` unmatched; handoff count unparsed; no `git diff` anywhere |
| 9 | `pytest workbench/tests/test_handoff_commands.py -k 'superseded_source_sha or same_test_count'` | **2 passed** while production differs → live counterexample |
| 10 | synthetic 4-commit repo in `/tmp/r8demo` (no commit on the branch) | both ancestor predicates pass, `git diff <tested> HEAD -- production.py` = 1 file → test blind |
| 11 | `git merge-base --is-ancestor 4b32063 11ba3df` | rc=0 → the ledger's negative control is false |
| 12 | `pytest workbench/tests --collect-only -q` | `1406 tests collected`; static `def test_` per rev: 1317 (`11ba3df`) vs 1329 (HEAD); `1394` matches neither |
| 13 | manifest recipe spot-check (`wc -l` + `sha256sum[:16]`) | recipe confirmed on `test_operator_config.py`; stale on `test_handoff_commands.py`, `test_credential_boundary.py`, `test_evaluate_models.py` |
| 14 | `git rev-parse be083e62^{tree}` | `36c47b66…` == the `SOURCE_TREE` the round-01 manifest records → manifest recipe for SHAs verified |
| 15 | grep for the guard test in `11ba3df`'s tree | absent; introduced by `2ed4934` |

No GPU inference was executed. No command in this audit wrote to `/home/heheboiz/data/cir` except the
creation of this file. Nothing was added to the branch, no ref was moved, no commit was created
(the counterexample lives only in `/tmp/r8demo`).

---

## 7. Final verdict

**R8-01: CONFIRMED — P1. The defect is real and reproduced on the live tree, and the guard test is
structurally incapable of detecting it.**

* `git diff 11ba3df HEAD` over production paths → **5 files, 167 insertions, 31 deletions**. `11ba3df`
  is an obsolete source, not a docs-only ancestor. **VERIFIED.**
* The guard test asserts only `merge-base --is-ancestor` twice and never diffs a tree. It passes today,
  on this repository, while the handoff installs an older production tree. **VERIFIED** (2 passed).
* The synthetic counterexample reproduces the same blind spot with git plumbing only. **VERIFIED.**
* Consequence, not mere bookkeeping: the handoff documents exit code 7 and two new launcher states that
  `11ba3df`'s `pc.sh` does not implement (0 occurrences), and the publisher's bundle-digest guard is
  absent from `11ba3df`. **VERIFIED.**

Secondary findings: false recorded negative control (`FINAL_DEFECT_LEDGER.md:358`) — P2;
`FINAL_HEAD_SHA` stale and self-contradictory, "only commit after it is documentation only" false, and
"the production diff … recorded at the end" absent (`FINAL_PC_RELEASE_VERDICT.md:10,23,35`) — P2;
`1394 passed` unreproducible and guarded by a self-referential test — P2;
`git checkout FETCH_HEAD` / `-b <branch>` clone as the stated install path
(`FINAL_PC_RELEASE_VERDICT.md:150`, `PC_ONE_COMMAND_GUIDE.vi.md:70`) — P2;
`git reset --hard ddcb71f` in a shipped review document — P3;
`git checkout 4b32063`-era and `84ec6f1`-era tested-source claims in five documents — P3;
stale per-file digest of the guard file in `SOURCE_COVERAGE_MANIFEST.md:89` — P3.

**NOT VIOLATED / disproved:** the handoff's own claim at `PC_PILOT_HANDOFF.md:10-11`
(`git diff 11ba3df b70d092` over `pc.sh`, `workbench/`, `src/` = 0 lines) is **true** — reproduced with
rc=0 and a single docs file changed. `FINAL_PC_RELEASE_VERDICT.md:11` and `:24` (`origin/main` =
`8ebcc74…`, unchanged) are **exact matches**. The round-01 manifest's `SOURCE_TREE` is an **exact**
match for `be083e62^{tree}`. Five documents that were automatically flagged in the sweep carry no
repo-release claim and hide no defect (`AUTOMATION_RUNBOOK.md` entire file; `PC_TRANSFER_CHECKLIST.md`
placeholders).

**UNKNOWN (not passed):** whether any gate ever ran on `11ba3df`, and therefore whether `1394` is that
run's count. No run artifact bound to a commit exists in the repository, and this audit cannot run the
suite or the GPU path. Both numbers are reported as unreproducible rather than adjudicated.
