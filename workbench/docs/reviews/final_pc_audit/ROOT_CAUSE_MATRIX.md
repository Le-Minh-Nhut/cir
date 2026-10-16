# ROOT_CAUSE_MATRIX — final PC wave

Base: `0b0f3a6` · Fix commit: `6e3c979` · Branch `research/cir-failure-workbench`
Target: clean Ubuntu host + RTX 5070 Ti (Blackwell sm_120, 16 GB VRAM), driven by `./pc.sh`

Six audits ran against the real source (reports `A.md`–`F.md`), then a fourth-way review wave
(`REV1`–`REV4`). This matrix merges every finding by ROOT CAUSE across the mission's dependency
graph, and records what was fixed, how it was proved, and what remains.

## 1. Boundary graph and status

| # | Transition | Input → Output | Status |
|---|---|---|---|
| 1 | PC host → one-command launcher | host facts → preflight verdict | FIXED (launcher added; capability detection, not a version string) |
| 2 | launcher → host dependency checks | driver/toolkit/docker → actionable remedy | VERIFIED (labels PENDING honestly; RTX 3050 here is reported as what it is) |
| 3 | host checks → docker configuration | `.env` → compose files | FIXED (GPU flag written where compose reads it) |
| 4 | docker config → GPU runtime | compose override → device reservation | UNVERIFIED (no toolkit on this host) |
| 5 | container startup → dataset validation | mounts → writable/read-only split | VERIFIED (`:ro` submounts kernel-enforced) |
| 6 | dataset → source sync | pinned upstreams → checkouts | VERIFIED (pipeline receipts) |
| 7 | source → checkpoint validation | files → SHA256 vs registry | FIXED (manifest-backed, was 85 s/request) |
| 8 | checkpoint → environment planning | registry → resolved contract | FIXED (tolerant for undeclared adaptations) |
| 9 | planning → installation | resolver → installed env | UNVERIFIED (needs target host) |
| 10 | installation → GPU verification | interpreter → capability record | FIXED (arch gate + signing key) |
| 11 | GPU verification → model loading | capability → launch authorisation | FIXED (gate was unsatisfiable on sm_120) |
| 12 | model loading → inference | checkpoint bundle → 3 categories | VERIFIED (bundle resolvable; dress-only refused) |
| 13 | inference → benchmark evaluation | stdout → aggregate | UNVERIFIED (needs target host) |
| 14 | evaluation → scientific validation | report → gate verdict | FIXED (vocabulary cap + corroboration) |
| 15 | validation → result indexing | reports → DuckDB | FIXED (empty-root `[OK]` corrected) |
| 16 | index → backend API | index → endpoints | FIXED (latency, duplicate 409) |
| 17 | API → frontend | JSON → UI | VERIFIED (bundle selectable, `blocked` terminal) |
| 18 | frontend → research analysis | runs → analysis views | VERIFIED (unchanged this wave) |

## 2. Confirmed defects by root cause

### RC-1 — a status/readiness path performed authoritative work
**Symptom:** `GET /api/models` took **85 s**; any readiness poll timed out and `./pc.sh` could never
reach evaluation.

**Root cause (two independent cost centres, same mistake):**
- `registry.checkpoint_availability` hashed every downloaded checkpoint on every request (LIMN: 3 ×
  1.79 GB).
- `runtime._interpreter_fingerprint` (reached via `verify_environment` → `environment_blockers` →
  `preflight_blockers`) hashed every installed distribution member: **97,317 `sha256_file` calls**
  for one model's environment, measured with cProfile.

**Fix:** the availability/readiness view now answers from what is already recorded — the checkpoint
download manifest (guarded by a size comparison) and a cheap presence/metadata fingerprint — while
the authoritative path (explicit verification, GPU probe, launch gate, receipts, ledger) keeps
byte-exact content hashing. The advisory result is named honestly so it cannot be read as a content
verification.

**Proof:** `checkpoint_availability(limn)` worst-of-3 **11.06 s → 0.97 s**; `sha256_file` calls
**129,751 → 0**; live endpoint **85 s → 6 s**. Two regression tests, each shown failing pre-fix.

### RC-2 — a composed capability must resolve on the worker's account, not the builder's
**Symptom:** the GPU launch gate could never open in the deployed container: capability records were
written unsigned and then rejected by their own reader.

**Root cause:** `_gpu_cache_key` resolved only `<CIR_REPO_ROOT>/.workbench_ledger_key`. The image is
built by root and runs as the operator's uid, so the key resolved to `b""` → signature omitted →
`_cached_gpu_capability` discarded every record → `require_launch_ready` raised → 409
`gpu_verification_required` forever. The evidence ledger had already migrated to
`XDG_CONFIG_HOME`; this reader had not.

**Fix:** same resolution order as the ledger — `WORKBENCH_LEDGER_KEY`, then a legacy
repository-root key if present, then `$XDG_CONFIG_HOME/cir/workbench_ledger_key` (created 0600).

**Proof:** with the legacy path absent, the key resolves non-empty, is created 0600, and is stable
across calls; the record round-trips.

### RC-3 — a gate comparison whose two sides used different vocabularies
**Symptom:** a run whose numbers equal the paper's could never certify
(`PASS_METRICS_ONLY_MAPPING_UNVERIFIED`).

**Root cause:** `paper_gate` capped the identity claim at the reference's own
`checkpoint_mapping_status`, accepting only `VERIFIED_METADATA`. `papers.yaml` spells a confirmed
mapping `VERIFIED` (8 rows, including the LIMN row); only `models.yaml` uses `VERIFIED_METADATA`.

**Fix:** both spellings count as verified; an unrecognised or explicitly unverified mapping still
caps the verdict.

**Proof:** the LIMN reference row now yields `PASS_WITHIN_5_PERCENT passed=True` for paper-exact
metrics; conesep (`UNVERIFIED`) still yields `PASS_METRICS_ONLY_MAPPING_UNVERIFIED`. Both directions
asserted. Regression test shown failing pre-fix.

### RC-4 — an architecture comparison that could never match
**Symptom:** on the very hardware the adaptation exists for, the capability probe could never
succeed.

**Root cause:** the adaptation target `sm_120` had its `sm_` prefix stripped before comparison,
while torch's arch list carries `sm_`-prefixed names (`sm_70`, …, `sm_120`) and may carry a `+PTX`
suffix. `'120' in ['sm_120']` is False.

**Fix:** normalize both sides to the bare capability number (strip `sm_`, drop `+PTX`).

**Proof:** table-driven check — matching list verifies, non-matching list refuses, `_blackwell`
suffix and `+PTX` suffix handled.

### RC-5 — a process-global selection treated as per-model error
**Symptom:** under `WORKBENCH_RUNTIME_ADAPTATION=sm_120`, planning crashed for every model that
declares no adaptation (e.g. CSMCIR), and the crash landed *after* pip install, orphaning a prefix.

**Root cause:** `resolved_runtime_environment` called `runtime_adaptation` in strict mode
(`undeclared_is_error=True`), while `model_constraints` deliberately tolerated the same situation.

**Fix:** the resolved contract is tolerant for models that declare nothing (they keep historical
pins) and still refuses loudly for a model that DOES declare an adaptation the selection fails to
match — the operator-error case is preserved.

**Proof:** csmcir resolves to historical `2.0.1 / cuda 11.7`, LIMN to adapted `>=2.7,<3 / cu128`;
a wrong value for a declaring model still raises.

### RC-6 — a first-class selection the shared lookup did not know
**Symptom:** the adapter path (`build_command` / `run_eval.py`) raised `KeyError` on LIMN's
legitimate bundle `base_iter0_all_categories`; only `evaluate_models.py` reached evaluation, by
bypassing the adapter.

**Root cause:** `registry.checkpoint_by_id` searched only `checkpoint_variants`; the API handler had
its own bundle resolution, so the two disagreed.

**Fix:** bundle resolution lives in the shared lookup (anchor on the first required member, carry
`bundle_id` + member list), so every caller agrees.

**Proof:** the bundle resolves and `validate_request` accepts it; an unknown id still raises; the
adapter now refuses only for the real reason (interpreter unconfigured on this host).

### RC-7 — a guard whose authoritative half was unreachable
**Symptom:** two concurrent identical submissions could both be accepted, double-booking the GPU.

**Root cause:** `SingleGpuQueue.enqueue(..., refuse_duplicate=True)` performs the check atomically
under the queue lock, but the API handler only ran its own pre-check and never passed the flag
(TOCTOU).

**Fix:** the handler now passes `refuse_duplicate=True` and translates `DuplicateJobError` into the
documented 409 body; the pre-check remains as a cheap fast path, documented as non-authoritative.

**Proof:** concurrent-submission test passes; the same test fails pre-fix with an escaped
`DuplicateJobError` (500 instead of 409).

### RC-8 — a child that survives its supervisor
**Symptom:** stopping the container left an orphaned evaluator holding VRAM.

**Root cause:** the evaluator runs with `start_new_session=True`, so the SIGTERM delivered to the
backend could never reach its process group, and the worker thread died without reaping it.

**Fix:** a queue `shutdown()` reusing the existing group-terminate helper (SIGTERM → grace →
SIGKILL), wired to the app shutdown event, with `serve_workbench.py` forwarding the stop to the
backend child.

**Proof:** a spawned child's group is gone after shutdown; pre-fix the method does not exist.

### RC-9 — a verifier that swallowed evidence
**Symptom:** a real GPU probe was discarded; the whole GPU verdict was decided by a stray 0-byte
file.

**Root cause:** the capability scan wrapped everything in `try/except Exception: real_gpu = False`,
so an unreadable artifact (left by an unisolated test) overrode readable evidence.

**Fix:** each artifact is read individually; unreadable ones are named, removed, and reported as a
condition, never as a downgrade of observed evidence. A readable record must also match the key in
its own filename.

**Proof:** with one good record plus one 0-byte file, the scan reports `UNREADABLE` and keeps
`real_gpu: True` (pre-fix: `PENDING`).

### RC-10 — scanners that passed on empty input
**Symptom:** acceptance reported PASS while checking nothing.

**Root cause (two sites):** Group E read `doc.get("rules", [])` from `pc_readiness.py --json`, a
document that has no `rules` key (real keys: `config, dataset, environment, gpu, intended_gpu,
models, papers, read_only, repository, summary`), so the filter was always empty; Group G diffed
`/api/jobs` statuses against an allowed set and passed when the queue was empty.

**Fix:** Group E consults the keys the document actually has and reports NOT VERIFIED when no
GPU-requiring model is present; Group G refuses to pass on an empty queue and fails only on a real
out-of-enum status.

**Proof:** both new tests fail against the pre-fix script and pass after; the live E snippet now
prints the real count (11 GPU-requiring models).

### RC-11 — a launcher polling a surface too expensive to poll
**Symptom:** readiness could never be reached on a host with checkpoints on disk.

**Root cause:** `pc.sh` used `/api/models` (RC-1: 85 s) for the already-running probe, the readiness
loop, and registry resolution, all with `--max-time 5` or `30`.

**Fix:** readiness polls `/api/jobs` (a cheap in-memory listing); registry resolution is a separate
single call with a generous timeout and one retry; the `.env` GPU flag is written where compose
reads it.

**Proof:** `--doctor` 0.3 s; the launcher harness (20 cases) drives the real script to completion.

## 3. Verified sound (no change needed)

- compose override precedence; relative `HOST_*` resolution from `workbench/docker`; dataset
  `:ro` submounts kernel-enforced; persistent storage; restart recovery; container-name consumer
  consistency.
- `fashioniq_val_split` agrees across `models.yaml`, `schemas/results.py`, the frontend protocol
  list, `src/evaluation/fashioniq.py`, `replay/limn.py`, and `papers.yaml`.
- Cohort correctness re-derived from pinned source: gallery = first-seen ref-then-target union,
  reference masked before ranking, macro mean = paper arithmetic. No protocol alteration found.
- CPU-only orchestrator + GPU-only isolated model environment is the real design and holds at every
  subprocess boundary; the packaged CPU torch in the orchestrator is correct.

## 4. Remaining, explicitly not-PASS

| Item | Status | Why |
|---|---|---|
| Real GPU passthrough on RTX 5070 Ti | **PENDING** | no `nvidia-container-toolkit` on this host; this machine has an RTX 3050 |
| sm_120 kernel execution in the isolated LIMN env | **PENDING** | needs the target host; arch-list format anchored on torch source, not an empirical modern-wheel capture |
| Environment installation on the target | **PENDING** | historical `torch==1.12.1` is not on plain PyPI; requires an index or the adapted profile |
| Real LIMN inference / FashionIQ evaluation | **PENDING** | needs the target host |
| Paper metric parity | **NOT CLAIMED** | a deviation is a research finding, not an infrastructure result |
| Any test exercising sm_120/Blackwell kernels | **GAP** | all mocked capability tests use sm_89 |

---

# Đợt “FINAL MISSION” (baseline a517cf9) — ma trận nguyên nhân gốc

## Phase 1 — bốn defect đã xác nhận

| ID | Nguyên nhân gốc | Bằng chứng | Trạng thái |
|---|---|---|---|
| P0-1 | `pc.sh` chạy `verify_run.py` bằng `python3` **host** (Python 3.14, thiếu torch/yaml) với `CIR_REPO_ROOT`/`PYTHONPATH` host. Host còn có **hai** khóa ledger xung đột (`.workbench_ledger_key` và `~/.config/cir/...`) nên host `LedgerKeyError` — không thể xác minh ledger do container ghi. | `grep -n 'python3 "$verify_script"' pc.sh` (2 chỗ); `.venv/bin/python -c "em.ledger_hmac_key(...)"` → `LedgerKeyError two different ledger keys exist` | ĐÃ SỬA — `run_verifier_in_container` dùng `docker exec -T cir-workbench python3 workbench/scripts/verify_run.py`; container down → `fail_stage` có tên, không fallback host |
| P0-2 | `verify_run.py` chỉ kiểm category cho LIMN; không kiểm metric extraction đã chạy chưa, không đối chiếu macro với category, không từ chối synthetic | Audit F break-attempt 3: report synthetic + macro bịa (`r10 99.9` vs trung bình category 27.93) vẫn `verified: true` | ĐÃ SỬA — `metric_evidence_reasons`: `metric_extraction_available is True` + `extraction_status == EXTRACTED_VERIFIED`; macro tái lập từ chính category với dung sai 0.05 điểm %; từ chối `is_synthetic`/`data_kind != experiment`; mọi model khác phải có so sánh publishable |
| P1-3 | `pc.sh` quyết định tái dùng container mà **không** kiểm mount thật; container trỏ dataset khác vẫn được dùng → sai dữ liệu khoa học | `docker inspect --format '{{json .Mounts}}'` không được đọc ở đâu trong `pc.sh` | ĐÃ SỬA — so `Source` thật với `HOST_FASHIONIQ_ROOT`/`HOST_WORKBENCH_CHECKPOINT_ROOT`/`HOST_WORKBENCH_ARTIFACTS_ROOT` (chuẩn hoá `realpath`); mount thiếu → lý do có tên; `docker inspect` không đọc được → UNKNOWN → fail closed |
| P2-4 | Nhánh “cấu hình không khớp” gọi `run.sh up`, rồi khối `container_running == 0` ngay dưới gọi **lần nữa** | OLD: `UP CALLS COUNT: 2`; FIXED: `1` (test `test_incompatible_configuration_correction_invokes_docker_startup_exactly_once`) | ĐÃ SỬA — set `container_running=1` sau lần up đầu |

## Phase 2/3 — defect phát hiện bởi audit độc lập

| ID | Nguyên nhân gốc | Bằng chứng | Trạng thái |
|---|---|---|---|
| A-E1 (P1) | Container đang chạy nhưng API không trả 200 → `pc.sh` **bỏ qua** kiểm tra active-job rồi `run.sh up` → có thể phá job đang chạy | check active-job nằm trong nhánh `code == 200`; nhánh else không kiểm gì | ĐÃ SỬA — nhánh API-không-trả-lời hỏi cùng câu hỏi, đọc thêm `queue_state.json` dưới artifacts (bind mount host đọc được); có job → refuse; không đọc được → UNKNOWN → fail closed; 2 test mới |
| E-1 (P1) | Boot lần đầu không có `workbench.duckdb` (gitignored) → `/api/runs` 503 vĩnh viễn tới khi ai đó gọi rebuild | `main.py:_startup` không kiểm DB | ĐÃ SỬA — build index nền khi thiếu; index có sẵn thì giữ nguyên; build lỗi → 503 trung thực kèm lý do thật (không bịa payload) |
| E-2 (P1) | Export cohort Markdown format `{item['r1']:.2f}` không guard `None` → HTTP 500 | TypeError tại `main.py:364` | ĐÃ SỬA — `_format_metric` render `N/A`, không bao giờ bịa số |
| D-1 (HIGH) | Hợp đồng hoàn thành cho model không phải LIMN yếu hơn hẳn: bỏ qua extraction/category/macro, dùng `any(...)` | probe 1.7: run thiếu extraction vẫn `verified: True` | ĐÃ SỬA (đang xác minh) |
| D-2 (HIGH) | `select_reference` trả `None` khi hai row VERIFIED khớp cùng điều kiện → **mọi** run `ptha_mtst` hợp lệ bị từ chối | probe 2.5: legitimate run → refused | ĐÃ SỬA (đang xác minh) |
| D-3 (MED) | `schemas/results.py:79` `Literal["experiment","mock"]` nhưng pipeline phát `data_kind="synthetic"` | `evaluate_models.py:185` | ĐÃ SỬA (đang xác minh) |
| D-4 (MED) | `combiner_rn50x4_noft` đã đăng ký nhưng mọi reference UNVERIFIED → run hợp lệ không thể certify, không phân biệt với run giả | probe 2.6 | ĐÃ SỬA (đang xác minh) |
| F-1 (P2) | `mutation_checks.sh` M4 đọc exit code `docker compose config`; daemon down → non-zero bị hiểu là “đã phát hiện” → PASS giả | FAIL-before log: `M4 ... DETECTED` khi daemon down | ĐÃ SỬA — kiểm docker CLI/daemon trước, không chạy được → SKIP NOT VERIFIED, không tính là detected |

## Điểm đã xác nhận ĐÚNG (không sửa)

- Xác minh hai chiều trên `verify_run.py`: report sửa sau ledger → từ chối; proof replay → từ chối; macro không khớp (LIMN) → từ chối; extraction chưa chạy (LIMN) → từ chối; synthetic → từ chối; metric NaN/ngoài khoảng/bool → từ chối. Và run hợp lệ với Recall đúng biên 0.0 / 100.0 vẫn được chấp nhận.
- `macros` của LIMN tái lập được từ chính category (dung sai 0.05 điểm % — cùng bound parser extraction đã dùng).
- Sai lệch số so với paper vẫn là **phát hiện nghiên cứu** (`ALERT_METRIC_DEVIATION`, `passed=False`), `verify_run.py` vẫn `verified: true`: thực thi và tái lập là hai câu hỏi khác nhau — cố ý.
- UI không thể trình bày run bịa/không hợp lệ như bằng chứng đã chứng thực.
