# 102 — exact corrections list for `FINAL_REPORT_20261112.vi.md`

The main report is **not** rewritten here; the lead does the integration pass. This file lists every
statement that is now FALSE because both composition Full runs exist, with the exact line, the current
text, and the correct value. All line numbers are 1-based against
`workbench/docs/reviews/20261111-master-audit/FINAL_REPORT_20261112.vi.md` at HEAD `82cb1044`.

Verified facts behind every correction:

| run | `execution_status` | `query_count` | `smoke` | gap | file |
|---|---|---|---|---|---|
| `tgcir` `7b1fa85007bffdab931f7f19` | `COMPLETED` | 6003 | `{label: null, limited: false, query_limit: null}` | `0.43853620852911723`, cohort 6003 | `workbench/artifacts/composition/summary/tgcir_tgcir_archive_unresolved_7b1fa85007bffdab931f7f19/full.json` (109 793 B, mtime 2026-11-12 05:16:29.257 +0700) |
| `clvc_net` `eab10106b2df8e97fb7a5ce5` attempt `-retry-2` | `PARTIAL` | 6003 | `{label: null, limited: false, query_limit: null}` | `null`, reason `no queries have exact ranks in all three conditions` | `…/summary/clvc_net_author_drive_unresolved_eab10106b2df8e97fb7a5ce5-retry-2/full.json` (17 029 B, mtime 04:53:22.955 +0700) |

Source of record: `raw/verify/98-tgcir-full-run.md` §1-§3.

---

## 1. Stale statements, line by line

### C1 — §4 verdict header (lines **485-486**)

Current:

```
**Phán quyết: CHƯA ĐẠT** — TG-CIR Full chưa tồn tại; CLVC Full đã có nhưng là `PARTIAL` với một điều
kiện không đo được và `composition_gap_ndcg` là `null`.
```

Correct: both Full runs exist. `tgcir` is `COMPLETED` over 6003 queries with a three-condition gap of
`0.43853620852911723`; `clvc_net` is `PARTIAL` with `composition_gap_ndcg: null` because
`image_only_empty_text` is `ABLATION_UNSUPPORTED`. The verdict itself is a judgement for the lead; the
**reason** sentence must change.

### C2 — §4.1 heading (line **488**)

Current: `### 4.1 TG-CIR: cái đang có là smoke 5 query, **không phải** Full`

Correct: TG-CIR Full `7b1fa85007bffdab931f7f19` exists. The smoke `1eee4c0adbc93f7708205a5e` still
exists and is still a smoke; the section can keep it but must stop saying it is all there is.

### C3 — §4.1 closing paragraph (lines **519-525**)

Current: `**Nói thẳng: đây là smoke 5 query, không phải Full run…**` … `Số **Full** của TG-CIR: **chưa
đo** — không có summary Full nào cho `tgcir` trong bằng chứng của đợt này.`

Correct last clause: a Full summary **does** exist now — `…_7b1fa850…/full.json`, `COMPLETED`,
6003 queries, `smoke.limited: false`. Replace "chưa đo" with the pointer and the metrics:

| condition | R@10 | R@50 | mrr | ndcg_full | scored |
|---|---|---|---|---|---|
| `multimodal` | 23.33833083458271 | 42.11227719473597 | 0.12098114437415375 | 0.24702051966534347 | 6003/6003 |
| `text_only_black_image` | 6.546726636681659 | 13.65983674829252 | 0.03674930344151411 | 0.1386930775424115 | 6003/6003 |
| `image_only_empty_text` | 2.798600699650175 | 7.163085124104614 | 0.010877930517005432 | 0.10452130761687346 | 6003/6003 |

`topk_export`: `TOPK_EXPORT_COMPLETE`, depth 100, three conditions, `missing_query_count 0`,
`partial: false`. `shortcut_outcomes`: `{"both": 9, "composition_required": 956, "image_only": 159,
"neither_scored": 0, "text_only": 384, "unresolved": 4495}`. **R@1 is still not produced** — that part
of §4.1 remains true and must not be deleted.

### C4 — §4.4 heading (line **564**)

Current: `### 4.4 CLVC Full: đã có một run hoàn tất, và một lần chạy lại đang chạy`

Correct: the retry finished. Heading should read that there are two completed attempts and no run in
flight.

### C5 — §4.4 retry status paragraph (lines **594-606**)

Current: `**Trạng thái lần chạy lại hiện tại — ĐANG CHẠY, chưa xong.**` … PID `1366364` …

Correct: PID `1366364` **exited**; its attempt produced
`…_eab10106b2df8e97fb7a5ce5-retry-2/full.json` (17 029 B, mtime `04:53:22.955`), `execution_status`
`PARTIAL`, wall `1867.2 s (31 min 7 s)` (log birth `04:22:15.752` → summary mtime `04:53:22.956`),
92 390 images → `0.0202 s/image`, and `--smoke` was **not** used (`smoke.limited: false`). Peak VRAM was
**not sampled** for this run (it was already in flight when the session began), so no VRAM number may be
attached to it. The sentence `**không** được gán bất kỳ con số Full nào cho nó` becomes obsolete: the
numbers above **are** its numbers.

### C6 — §4.4 last line (line **608**)

Current: `Không có con số Full nào khác: **chưa đo**.`

Correct: the CLVC Full numbers are measured (§4.4 table is correct as written, but cite
`…-retry-2/full.json`, not `…_eab10106…/full.json` — see C7). TG-CIR Full numbers are measured (C3).
What remains unmeasured is **CLVC peak VRAM only**.

### C7 — §4.4 metric provenance line (lines **580-581**)

Current: `` `raw/verify/90-…md` §4.1; đọc lại từ
`workbench/artifacts/composition/summary/clvc_net_author_drive_unresolved_eab10106b2df8e97fb7a5ce5/full.json` ``

Correct path: `…_eab10106b2df8e97fb7a5ce5-retry-2/full.json`. The numbers in the table are identical in
both attempts (re-read here), but the completed attempt is `-retry-2`, and `…_eab10106…/full.json` is the
earlier superseded one.

### C8 — §7 verdict row 4 (line **684**)

Current: `| 4 | Kết quả composition trên Full | **CHƯA ĐẠT** | TG-CIR Full chưa tồn tại (chỉ có smoke 5 query), còn CLVC Full là `PARTIAL` với `composition_gap_ndcg` `null`. |`

Correct: `TG-CIR Full chưa tồn tại` is false — `7b1fa85007bffdab931f7f19` is `COMPLETED`, 6003 queries,
gap `0.43853620852911723`. CLVC remains `PARTIAL` with `composition_gap_ndcg` `null`. The verdict word
(`CHƯA ĐẠT` vs a partial pass) is the lead's call; the **reason string** must be replaced.

### C9 — §6 evidence table, `90-circus-composition-runs.md` row (line **663**)

Current: `| raw/verify/90-circus-composition-runs.md | CLVC Full `eab10106` §4.1 (PARTIAL, 4489 s, …) … |`

Still accurate for `90`, but the table has no row for `98-tgcir-full-run.md`, which is now the primary
source for **both** Full runs. Add:
`| raw/verify/98-tgcir-full-run.md | tgcir Full 7b1fa850 (COMPLETED, 6003 q, gap 0.4385) §2; clvc_net Full eab10106-retry-2 (PARTIAL, 1867.2 s, 0.0202 s/img) §1; side-by-side §3; API cross-check §4 |`

### C10 — §0 live snapshot (lines **33-38**)

Current: `## pgrep -af composition_eval.py` listing `1366356` / `1366364`, and
`## ps -o pid,etime,%cpu,nlwp,stat -p 1366364` → `1366364 20:23 767 19 Sl`.

This was true **at 04:43 +07** and is a timestamped snapshot, so it is not wrong as a snapshot — but
both processes have exited, and any reader who treats §0 as current state will be misled. Add one line
under the block: `Hai PID này đã thoát; xem §4.4 (đã sửa) và raw/verify/98-…md §1.`

### C11 — §5 item 4 (line **~625**, "Gallery encode … 0.0138 vs 0.0027 s/ảnh")

The ~5× discrepancy remains **unexplained**, so the item stays open — but it should note the two
completed Full runs' independent rates: `clvc_net` `0.0202 s/img` and `tgcir` `0.00603 s/img`. Neither
reproduces `0.0027`, which narrows the standalone reading rather than confirming it. Add, do not delete.

### C12 — §5 item 5 (line **~640**, TG-CIR `adapter_worker_exited`)

Unchanged in substance: the guard is shipped and CPU-proven, the latch change stays reverted, and no real
run reached the guard. But the sentence "chưa có bằng chứng từ run thật" now has a companion fact: the
TG-CIR Full `7b1fa850` reached `COMPLETED` on 6003/6003 queries, i.e. a real run **did not** enter the
guard. That is consistent, not a contradiction; state it so the reader does not read "unproven" as
"possibly broken".

---

## 2. Statements that are still TRUE and must not be "corrected"

* **R@1 does not exist.** `ConditionMetrics` declares only `recall_at_10`/`recall_at_50`
  (`workbench/composition/metrics.py:38-49`); no `recall_at_1` anywhere in `workbench/composition/`.
  Both Full runs and both API hosts return `<ABSENT>`. Do not turn this into a zero.
* **CLVC `PARTIAL` is a real 119-query hole** (1.98 % of 6003), 5884 scored per condition, 238 =
  119 × 2 missing top-K windows. Re-verified from `-retry-2/full.json`.
* **`composition_gap_ndcg` for CLVC is `null`** with reason `no queries have exact ranks in all three
  conditions`. `metrics.py:126` builds `common` as the qids ranked in all three conditions; `image_only`
  has zero ranks, so the cohort is empty. This is the mechanism, not a phrasing.
* **Composition ≠ native.** `tgcir` 23.3383 (composition) vs 49.7208 (native macro, `raw/tgcir_run3cat_gpu.json`)
  are different protocols — different query text, different candidate pool.
* **PAIR is still blocked upstream** (`model_vssp` + `MHTransformer_dis` absent across all 21 commits of
  `iLearn-Lab/ICASSP25-PAIR` at `bfb0b98`; `git grep` over every rev = 0 hits). Verified again for this
  round; see `raw/verify/102-queue-gpu-and-blocker-inventory.md` §7.2.
* **`TESTED_SOURCE_SHA` is still `96d86010…` and still red by design** — 14 production files moved past
  it. It is **not** advanced here; the lead schedules the stationary sweep. Preconditions in
  `raw/verify/102-…-inventory.md` §7.2 row `SHA-PIN`.
* **`gpu_may_be_held` is still a live refusal.** `raw/verify/102-…-inventory.md` §3-§4 prove the blocked
  and released directions; the blocker is kept.

---

## 3. Addendum — two corrections are themselves now stale (concurrent commits, 2026-11-12 07:53)

Written after §1-§2, when two commits landed that were not on `origin` at `82cb104`. They change the
**status** of two items the main report calls open. The lead should apply these in the same pass.

### A1 — R@1 is no longer absent from the code (affects §2 bullet 1 of this file, and §4.1 of the report)

`8484ef7` added `recall_at_1` to `ConditionMetrics` (`workbench/composition/metrics.py:48`), computed as
`hit_at_k(rank, 1)` over the same scored denominator, with the tie policy left to the producer. So the
report's sentence "R@1 không tồn tại ở cả hai phía" (`FINAL_REPORT_20261112.vi.md:515-517`) is **stale
in its reason** even though its **conclusion about the published runs still holds**: the backfill script
`workbench/scripts/backfill_recall_at_1.py` has not been run — **0 of 35** `summary/*/full.json` files
carry a `recall_at_1` key, including both Full runs in §7.4. Correct wording: R@1 is **computed by the
pipeline now** but **not present in any published artifact yet**, so it must still not be quoted — and
still not as 0. Re-running the backfill over the stored `ranks/*.json` is what would close it, and that
is a separate, cheap, CPU-only step the lead can schedule.

### A2 — the encode fallback is now recorded (affects §5 item 3 of the report)

`5c6c781` makes `runner.py` record the feature partition per condition (`partition`, `fallback_used`,
`fallback_qids`, `fallback_reason`, `runner.py:657-700`) and keys the memmap cache on
`_feature_partition_fingerprint` (`:668`), so a mixed-partition cache cannot be mistaken for a canonical
one. `FINAL_REPORT_20261112.vi.md` §5 item 3 ends "**Chưa vá.**" — that is now **false for runs made
after this commit**. It stays true for the two published Full runs, whose feature caches predate the
field and therefore report `partition: None` with a reason. Correct wording: "fixed going forward;
the existing runs cannot prove their partition, so a cross-run feature comparison involving them is
still not licensed."

### A3 — nothing else in §1-§2 changes

`tgcir` Full `7b1fa850…` and `clvc_net` Full `…-retry-2` are still the two runs on disk, still
`COMPLETED` and `PARTIAL` respectively, still 6003 queries, still `smoke.limited: false`, and the CLVC gap
is still `null` for the same `ABLATION_UNSUPPORTED` reason. The PAIR absence, the SHA pin, the 5 rerun
cases, the 15 `NOT_RUN`, LIMN `7:3038`/`7:3790` and the 46/50 inferred units are unaffected.

---

## 4. Status — APPLIED, do not apply again

All twelve items (C1-C12) and all three addendum items (A1-A3) were applied to
`FINAL_REPORT_20261112.vi.md` on 2026-11-12, in one commit on top of `d0e0eb7` (report file only,
pushed nowhere). Line numbers in §1 are therefore stale by construction — they refer to `82cb1044`.
Three statements in the report were also corrected for the same reason without being in this list:
§2.4 (`_MODELS_CACHE` was fixed by `abba57e`, so "still serves a stale document" is no longer true),
the §2 verdict line, and §1.5's "current state" sentence (backend `MainPID` is `1476678` now, not
`1221379`; frontend is unchanged at `1221148`).

Two limits of that pass, stated so the next reader does not assume more: R@1 is still quoted only
from the derived file (0 of 35 published `summary/*/full.json` carry the key), and the CLVC run in
flight under the new profile (`da0a2162be0e159a1a456cec`, no `summary/` yet) contributed no number.

**That second limit is now closed** (second commit, same branch, report file only). The run finished
and published `…da0a2162be0e159a1a456cec/full.json` (17 956 B, mtime `2026-11-12 09:15:18.964 +0700`):
`PARTIAL`, 6003 queries, conditions `(multimodal, text_only_black_image, clvc_image_encoder_only)`,
6001/6003 scored, `composition_gap_ndcg 0.33061712371309204` over 6001. Source of record is
`104-clvc-full-image-only.md`. The report now carries it as **§4.4b**, keeps the old
`eab10106…-retry-2` as **§4.4a** with its own untouched numbers, and states in three places (§4.2,
§4.4b, §7 row 4) that this gap runs on a **declared variant** image condition and is therefore not the
protocol gap and not comparable to TG-CIR's `0.4385`.

Two things C1-C12 could not have anticipated and that the second pass had to settle from the artifacts
rather than from the brief: LIMN's rank artifact leaves `7:3038`, `7:3790`, `7:2179` and `7:4066` all
`null` in all three conditions (5884 non-null / 119 null per condition), so CLVC and LIMN now agree on
2 qids and differ on 117 — the shared cohort is 5884, not 6001; and `backfill_recall_at_1.py` refused
this run because `METRIC_BLOCKS` mapped only protocol names — which `104` §6.1 records, and which
`b887c7e` has since fixed (re-running it now exits 0 and reproduces 1.3997667055490752 /
0.11664722546242293 over `scored_count 6001`). The report states both halves; the derived file that
verification produced was deleted, because this run publishes R@1 itself.

**Third pass — the caption-free figure is measured, and it replaces every `chưa tính xong`.**
`workbench/artifacts/composition/derived/category_routing_inputs/clvc_net_da0a2162be0e159a1a456cec_full.json`
(4 623 B) publishes both cohorts from the same rank table: A = `0.33061712371309204` over 6001 (the
published figure, `reproduces_published: true`) and B = **`0.33400015944039774`** over 5884
(`caption_free: true`), differing by `0.0033830357273053`. The report quotes both wherever the gap is
stated (§4 header, §4.3, §4.4b, §7 row 4, §8) and states that the finished summary is immutable and
carries no `category_routing_inputs`, with the §7.7 patch spec named as the open item (§5.2 item 12).
No section of the report describes the routing input as stem-only any more; the two remaining `stem`
mentions in §4.3 are the refutation ("routing chỉ dùng stem" is listed as forbidden) and the
26-of-119 stem-ambiguity argument, which is about why stem alone would be worse, not about what the
resolver reads.

**Fourth pass — live-state rows re-pinned.** After the pin advanced to `117fc17220d0e35651c6778c1afb62e271c8d944`
and the campaign was pushed, the report's own live-state rows had gone stale. §0.1 is now the single
authority (`HEAD a0c08c0 == origin`, 0/0, 36 commits since `82cb1044`; backend `MainPID=1510910`,
frontend `1221148`; no composition process, GPU 373 MiB with no compute app) and §9 repeats it rather
than holding a second copy. The `TESTED_SOURCE_SHA` row is no longer "still old": it records the
advance, the sweep (`3123 passed, 20 failed, 47 skipped = 3190`, `3586.55 s`, `106-sweep-117fc172.md`),
the guard at 8 passed, the 29-file / +3505 −246 production diff from the OLD pin as history, and the 20
remaining failures by class (18 baseline ids plus 2 never in that baseline). The guard-extension row is
new: `git checkout` and `must print` are checked in every `RELEASE_DOCS` entry, with the bite proof.

One row was certifying an output it no longer produced: `R@1 vắng khỏi artifact đã công bố` printed
`35` and `35` → 0/35. Re-measured: **`35` and `36` → 1/36**, the one file being
`clvc_net_author_drive_unresolved_da0a2162be0e159a1a456cec/full.json`. §4.1's "R@1 absent" is now
scoped to the three runs it is true for (smoke `1eee4c0a` and both older Full runs), and §5.1 item 6
carries the same qualification. The served bundle is recorded as `assets/index-DDviexaW.js`, sha256
`58eed6d48640fa1f93d2a64a96d87812a044a3b07b324fb872f6126162bf5369`, with the older names labelled
historical.
