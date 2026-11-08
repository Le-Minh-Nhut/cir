# CIR — báo cáo cuối phiên clean-rerun (2026-11-09)

Revision khi viết: `20a406a`. `TESTED_SOURCE_SHA` vẫn là `80842661` — guard đỏ, xem §7.
Mọi con số dưới đây dẫn từ file có trong `EVIDENCE_INDEX.md`; không có số nào là ước lượng.

## 1. Trạng thái tổng quan

Sáu lỗi đường chạy mà bản review độc lập tại `d699b66` nêu đã được sửa và **mỗi lỗi có test fail
trước / pass sau** (19 failed–2 passed trước, 25 passed sau, cùng một file test). Giao diện sáng đã
lên bản chạy thật với 17 model luôn hiển thị, Top-K thật theo từng query, và phạm vi protocol/subset
không còn lệch giữa các trang. Phát hiện quan trọng nhất của phiên không phải là một lỗi code mà là
một sự thật về dữ liệu: **toàn bộ artifact CIRCUS trên máy này được sinh trước commit pin BLAS
`bf6a448`**, nên chưa có số nào được chấm dưới một policy BLAS được ghi lại — và đo được thì **chỉ
`encoder` bị ảnh hưởng số**. Cuối cùng, một điều không thể làm trong phiên: **không thể backfill**
bất kỳ run cũ nào, vì khoá cache là chính digest của identity (giải thích ở §6).

Tóm lại: code đúng hơn, giao diện thật hơn, hiểu biết về dữ liệu chính xác hơn — nhưng **chưa có
gate khoa học nào được chứng nhận lại bằng một lượt chạy đầy đủ**, và guard release vẫn đỏ.

## 2. Sáu lỗi evaluator đã sửa

Tất cả trong `workbench/tests/test_review_defects_evaluator.py`.

| # | Nguyên nhân | Hàm đã đổi | Test |
|---|---|---|---|
| 1 | `_blas_thread_limit` chỉ được áp trong `_rank_groups_concurrently`; `_backfill_export` và nhánh fallback theo batch gọi thẳng `adapter.rank*` **sau khi** helper đó đã restore, nên chấm ở thread mặc định của tiến trình | `_backfill_export` (pin + restore trong `finally`), vòng fallback, `BackfillRankDisagreement` mới | `test_backfill_runs_under_the_same_blas_policy_as_the_normal_path` |
| 2 | `if setter is None: return int(previous)` là "thành công im lặng", và `previous` đọc từ biến môi trường chứ không từ runtime | `_openblas_symbols`, `_blas_thread_limit` (trả record, **đọc lại** sau khi set), `_current_openblas_threads`, `_blas_pin_capability` | 4 test: silent-success, verified, readback-disagrees, int-coercion |
| 5 | `blas_threads_pinned` / `query_scope_resolution` / `query_order_identity` không nằm trong payload đã hash, nên hai run khác policy vẫn cùng `group_id` và bị gọi là `duplicate` | `measurement_identity` (payload + `identity_schema_version`) | 6 test, gồm `test_identity_v1_artifacts_are_unknown_not_regrouped` |
| 6 | `scoring_impl` suy từ source **hiện tại** của adapter, rồi gắn lên artifact lịch sử | `scoring_implementation(..., summary=, live=)`, `_recorded_source_matches_disk` | 3 test |
| 7 | Loader chỉ kiểm fingerprint + shape; file cùng shape khác nội dung vẫn load (đo được: sum 6.0 → 0.0, `accepted=true`) | `_load_memmap`, `_write_memmap_metadata` (digest theo chunk), `_memmap_integrity` | 6 test, gồm truncated / stale-sidecar / legacy-absent |
| 8 | `assert … or True` không bao giờ fail (`test_composition_topk_runner.py:131`, `test_capability_composition.py:81`) | hai assert thay bằng kiểm tra thật + ca âm (sai hash, sai thứ tự, sai độ dài, id trùng) | `test_gallery_identity_assertion_is_not_vacuous` |

Thêm một lỗi do review bắt trong chính bản sửa của tôi: `cache_verify_mode` được đọc bằng
`summary.get(...)` trong khi runner **không hề ghi khoá đó vào summary**, nên mọi artifact thật mang
`None` và run `header` bị gộp nhóm với run `full`. Đã sửa (ghi vào summary + đọc qua
`_recorded_or_summary`) và thêm 4 test.

Chi phí cache đã đo trên gallery thật 74.381×512 (152 MB): digest 43,6 ms, load có verify **+43,3 ms**
so với mmap trần, trong khi encode gallery mất hàng phút. Chế độ `header` +0,0 ms; `off` 0,09 ms.

## 3. Sự thật quan trọng nhất: pin BLAS và mức ảnh hưởng thật

`git log -S blas_threads_pinned` → trường này do **`bf6a448`** thêm, commit
**2026-11-08 20:33:11 +0700**. Summary composition mới nhất trên đĩa: **2026-11-08 17:42**. Số summary
sinh sau 20:33: **0**. Vậy **mọi artifact CIRCUS đều được chấm khi chưa hề pin BLAS**.

Nhưng điều đó chỉ quan trọng với model mà **thread count thật sự đổi kết quả**. Đo, không suy từ
source (seed 20261109, gallery 3000, 8 query, 6 near-tie mỗi query, BLAS 1 vs 20 qua setter runtime):

| model | biểu thức chấm | max \|Δscore\| | rank đổi | max \|Δrank\| | tie cluster đảo | nhạy BLAS? |
|---|---|---:|---:|---:|---:|---|
| conesep | einsum(optimize=True) | 0.000e+00 | 0 / 400 | 0 | 0 / 8 | không |
| ptha_mtst | einsum(optimize=True) | 0.000e+00 | 0 / 400 | 0 | 0 / 8 | không |
| habit | einsum | 0.000e+00 | 0 / 400 | 0 | 0 / 8 | không |
| intent | einsum | 0.000e+00 | 0 / 400 | 0 | 0 / 8 | không |
| hint | einsum | 0.000e+00 | 0 / 400 | 0 | 0 / 8 | không |
| tme | einsum | 0.000e+00 | 0 / 400 | 0 | 0 / 8 | không |
| **encoder** | matmul theo batch (M=8) | **1.788e-07** | **29 / 400** | **5** | **6 / 8** | **CÓ** |
| limn | matmul 1 query (M=1) | 0.000e+00 | 0 / 400 | 0 | 0 / 8 | không |

Cơ chế đã cô lập: GEMM theo batch (M>1) thay đổi theo thread count; matvec một query thì không.
`np.linalg.norm(axis=1)` bất biến. Và dù `einsum(optimize=True)` **có** đi qua đường đa luồng (nhanh
~2,1× ở 20 thread), kết quả vẫn **bit-identical**.

**Hệ quả:** với bảy model còn lại, việc thiếu pin là **lỗ provenance**, **không phải lỗi số** — câu
"rerun" cho chúng không có nghĩa "số cũ sai". Chỉ `encoder` là có khả năng sai số thật.

## 4. Frontend

- Nền sáng `#f5f7fa` (`rgb(245,247,250)`), chữ 17px mặc định và tuỳ chọn 19px (`style.css:51-52`).
- 17 model luôn hiển thị, kể cả chưa chạy/bị chặn; đếm riêng model / run / pending / deferred.
- Subset **luôn** được ghi vào URL → Full không còn bị biến thành SF khi copy/reload/back/forward.
- Top-K thật: K thuộc identity của request (run+subset+condition+query+K), menu bị chặn theo depth đã
  export, response cũ không ghi đè ngữ cảnh mới. Chọn Top 50 render đủ 50 ảnh.
- `outcomeLabel()` để badge không in trần chữ `unresolved` (đọc thành `unscored`, nghĩa ngược lại).
- `ScoringImplNotice` bỏ câu quá rộng "không so sánh được ở mức từng query".

Harness chạy lại live trên `127.0.0.1:5173` (dữ liệu thật, không fixture): **cold 3, composition 8,
mobile 4, samples 2 check — tất cả exit 0**. 13 test node pass. Ảnh chụp:
`ui/overview-1920x1080.png`, `ui/overview-2560x1440.png`, `ui/composition-1440x900.png`,
`ui/samples-top50-1920x1080.png`, `ui/mobile-390x844.png`. Không tràn ngang ở 1920/2560/390
(`body.scrollWidth == innerWidth`), 0 ô bị cắt.

## 5. Bảng gate G0–G12 (trạng thái thật)

| gate | trạng thái | phạm vi / lý do |
|---|---|---|
| G0 inventory | **PASS** | 17 model id, 29 variant, `MODEL_MATRIX.csv` |
| G1 input correctness | **NOT_RUN** | không chạy differential native↔adapter nào trong phiên này |
| G2 forward/scoring | **NOT_RUN** | không forward pass nào được đối chứng; đo BLAS là bằng chứng G7 |
| G3 policy/identity | **PASS** (chỉ về **code**) | identity v2 + readback được test; **chưa artifact công bố nào mang v2** |
| G4 cache/resume/fallback | **PASS** | cơ chế + ca âm; cache cũ nay không truy cập được (đúng thiết kế) |
| G5 clean run/coverage | **NOT_RUN** | run cũ nhãn `unknown (predates the BLAS pin)`; chỉ encoder chạy lại và **không phải baseline** |
| G6 rank/export | **FAIL** | 4 run depth 50, 4 run không có export; Top-100 cần rerun |
| G7 optimization parity | **PASS** | bảng đo BLAS + chứng minh song song theo trục query; **không claim speedup** |
| G8 measured performance | **NOT_RUN** | hai lane khác đang chạy → không có timing sạch |
| G9 frontend correctness | **PASS** | 4 session browser trên dữ liệu thật, exit 0 |
| G10 frontend usability | **PASS** | sáng, 17/19px, 1920/2560/390, 0 ô cắt |
| G11 integration/release | **FAIL** | guard đỏ (`TESTED_SOURCE_SHA` không thể tiến khi chưa sweep) |
| G12 coordination | **PASS** | có một sai lệch đã ghi: 4 lane chạy song song so với cap 3, lane thứ tư đã dừng |

**7 PASS · 2 FAIL · 4 NOT_RUN.** Không có gate nào được PASS mà thiếu bằng chứng trên đĩa.

## 6. Rerun: đã làm gì, chưa làm gì, vì sao

**Đã chạy lại trong phiên này — đúng 1 model**, theo `RUN_LOG.csv`:

- `encoder` (`5dee7425dd46c5f154808aac`): `6003/6003`, coverage 1.0, depth 100,
  `blas_verified=true` (readback 20→1, restore 20), `identity_schema_version=2`, `invariants=PASS`,
  `verdict=PASS`. **Nhưng** cột wall-time tự ghi *"measured under concurrent load from two other
  lanes — not a baseline"*, throughput `not_measured`. Nên đây là bằng chứng **cơ chế**, không phải
  số đo hiệu năng.

**Chưa chạy:** 11 hàng `RERUN_FULL` còn lại (conesep, ptha_mtst, habit, intent, hint, tme, 3 variant
limn, 2 variant airknow) và toàn bộ hàng `BLOCKED`.

**Vì sao không thể "chỉ export thêm" cho run cũ — điều này chốt phương án:** khoá cache là
`_digest({"run": fingerprint, ...})` với `fingerprint = _digest(identity)`. Identity nay có các
trường v2 mà artifact cũ không có, nên fingerprint đổi. Tính trên artifact conesep thật:
`30f6de4238…` → `cb92b4b4fd…`, khoá cache gallery `439d49b3…` → `4f5e49f0…`. Loader fail kiểm
fingerprint → `_cached_features` trả `{},{}` → `--resume` từ chối. **File `.npy` còn trên đĩa nhưng
không dùng được.** Đây là hành vi đúng (cache khác policy không được tái dùng), nhưng nghĩa là
`--resume` **không thể** bắc qua một thay đổi schema identity. Vì vậy mọi run cần export sâu hơn đều
phải **RERUN_FULL + `--topk-depth 100`** trong một lượt GPU, không phải rerun rồi backfill.

Tổng theo `RERUN_DECISIONS.csv`: **12 hàng RERUN_FULL · 14 BLOCKED · 3 DEFERRED** (29 hàng).

## 7. Tồn đọng

- **Guard đỏ (G11).** `test_handoff_commands.py` → **1 failed, 7 passed**:
  `TESTED_SOURCE_SHA 80842661… does not contain the production code currently on disk`. Không thể
  nâng SHA vì cần một serial sweep đầy đủ (~58 phút) mà GPU đang bận. Đã **không** sửa test, **không**
  nới gate.
- **Sweep gần nhất tại revision hiện tại: chưa có.** Sweep tại `ffb6ae8` đã bị huỷ ở **22%**, không
  có count (`sweep_ffb6ae8_ABORTED.txt`, exit code cố ý không ghi).
- **G8 NOT_RUN.** Không có timing sạch vì hai lane khác chạy đồng thời.
- **BLOCKED (14 hàng), lý do cụ thể trong `MODEL_MATRIX.csv`:** thiếu byte checkpoint (conesep n0.5/n0.8,
  habit n0.5/n0.8, intent n0.5/n0.8, tme cirr_0); chưa có composition adapter (pair ×2, clvc_net,
  dcnet, combiner_rn50x4_noft, clip4cir_rn50x4_fullft, tgcir); `dcnet` còn có blocker `weights_only`
  và interpreter duy nhất không có torch.
- **DEFERRED (3 hàng):** `csmcir`, `sprc` — theo yêu cầu người dùng.
- **`airknow`:** adapter đã có và contract score/rank **chỉ verify trên input tổng hợp**; **chưa có
  forward GPU**, layout target chưa xác nhận với checkpoint thật, `status()` trả `ready=False`
  ("no interpreter environment resolved"), cả hai variant `NO_CLEAN_CHECKPOINT`.
- **`limn`:** coverage **5.884/6.003**. Resolver nay gán 6.001 với `7:3038`, `7:3790` chưa resolve.
  **6.001/6.003 KHÔNG phải 100%** và không lượt nào được phép lặng lẽ đổi mẫu số.
- **Lỗi ordering có sẵn, không sửa:** `test_circus_cli.py::test_cli_3d_full_gallery_resume_and_gallery_invalidation`
  fail `'REUSED' != 'COMPLETED'` khi `test_composition_topk_runner.py` chạy trước. Đã chứng minh
  **có trước** thay đổi này (stash hai file production vẫn fail). Thuộc defect #5 của sweep
  `20261108T212500`, là lỗi isolation của test — không thuộc phạm vi lane này.

## 8. Commit, push, dịch vụ

Năm commit của phiên, tách theo lane:

| SHA | nội dung |
|---|---|
| `7ef4cc4` | sáu lỗi evaluator + test (19 failed/2 passed → 25 passed) |
| `f6796b3` | CLIP4Cir: member theo `role` từ registry + test |
| `97f9d35` | bằng chứng lane A + `RERUN_IMPACT.md` + sweep bị huỷ |
| `87ff419` | frontend light workspace (15 file) |
| `20a406a` | adapter Air-Know (ghi rõ chưa sẵn sàng) |
| *(commit này)* | evidence pack: matrix, decisions, gates, index, báo cáo |

Dịch vụ: `http://127.0.0.1:5173` (200) và `http://127.0.0.1:8000` (200); asset đang phục vụ
`index-9MwV5MAk.js` khớp đúng `dist/assets/`.

Push và trạng thái remote: xem mục cuối của `GATE_RESULTS.json` (G11) và xác nhận
`git rev-parse HEAD` == `git rev-parse origin/research/cir-failure-workbench` sau khi push.

**Không** claim "tất cả đã sạch": G1, G2, G5, G8 chưa chạy; G6 và G11 đang FAIL.
