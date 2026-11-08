# BÁO CÁO CUỐI — lượt `research/cir-failure-workbench`

Thời điểm viết: `2026-11-09T00:32:09+0700`. Nhánh `research/cir-failure-workbench`.
Báo cáo này **tổng hợp lại bằng chứng đã có**, không chạy experiment mới, không chạy GPU, không sửa
source ngoài thư mục này.

**Quy tắc số liệu của báo cáo này:** mỗi con số phải mở được file raw chứa nó. Số chỉ tồn tại trong
prose của một `.md` khác được ghi rõ **`chỉ có trong prose, không có file raw`** và không được trình
bày như một phép đo. Lý do: một lượt trước đã ghi `2192 img/s`, con số **không xuất hiện trong bất kỳ
file raw nào**, và bảng prose của chính nó ngụ ý ~1024 ảnh/dòng — một thí nghiệm khác với `vdecode.txt`
(512 ảnh). Lỗi đó không được lặp lại ở đây.

**Thẻ dùng trong toàn báo cáo:**

| thẻ | nghĩa |
|---|---|
| **quan sát được** | mở file raw / chạy lệnh và đọc được; số chép nguyên, không làm tròn |
| **suy luận** | đọc code/shape/control-flow mà suy ra, chưa có phép đo trực tiếp |
| **chưa giải quyết** | bằng chứng hiện có không quyết định được |

---

## 1. Tóm tắt một trang

### 1.1 Lượt này làm gì

Lượt này **không sinh số mới**. Nó kiểm toán đường *đọc ảnh → tiền xử lý → inference → scoring* của
workbench, đối chiếu từng model với source upstream tại đúng commit pin, và viết lại các con số
throughput bằng những gì file raw thực sự nói. Các phép đo đã có từ những lượt trước (decode, scoring,
BLAS, sweep test) được **đọc lại từ file raw**, không được đo lại.

### 1.2 Năm sự thật quan trọng nhất

1. **ENCODER và PTHA+MTST không chạy được từ HEAD** trước khi sửa. Hai adapter gọi ba helper song song
   hoá decode mà không import → `NameError` **bên trong `_worker_main`**, bị `except Exception` của
   chính worker nuốt thành `reply(False, error_type="NameError")`, nên hiện ra như lỗi model.
   **quan sát được** (`20261108T155805-audit-fixes/MODEL_REPRODUCIBILITY_VERDICT.md` §1; sửa đã áp:
   `model_encoder.py:35-37`, `model_ptha_mtst.py:21`).
2. **`optimize=True` / chunked GEMM bị từ chối vì lý do khoa học, không phải kỹ thuật.** Nhanh hơn
   4.6–17.8× nhưng đổi **207 / 18,009** rank. **quan sát được** (`ein_threads.txt`,
   `metric_gate.json`, `d2`-family).
3. **Không gate nào nói "feature matrix đúng".** Toàn bộ G2-fwd (`forward feature bit-identity`) là
   `NOT_RUN` vì GPU thuộc cohort `intent` suốt cửa sổ kiểm toán. Pixel bằng nhau **không** suy ra
   feature bằng nhau. **quan sát được** (`EQUIVALENCE.md` gate table).
4. **Suite test không xanh.** Bản ghi hiện hành `1129c58`: `2467 passed, 18 failed, 47 skipped =
   2532`, exit 1. **8 trong 18** failure là defect nằm trong chính range đã giao. **quan sát được**
   (`full_suite_1129c58b.txt`, `FAILURES.md`).
5. **Không con số nào trong cohort CIRCUS là paper reproduction.** Gallery 74,381 ảnh so với 18,853
   ảnh mà paper nêu; khoảng cách R@10 của `habit` là **19.76 điểm** và phần lớn là protocol, không
   phải model hỏng. **quan sát được** (`full.json` → `paper_vs_release_gallery` + `metrics`;
   `registry/models.yaml` → `reported_r10`).

### 1.3 Những gì **KHÔNG** được xác lập

- Không có đo **bit-identity của feature forward** cho bất kỳ model nào (G2-fwd `NOT_RUN`).
- Không có so sánh **cold vs warm cache** (G4d `NOT_RUN`) — cần re-encode 74,381 ảnh.
- Không có baseline G5 full-coverage: sidecar chưa bao giờ kiểm **byte** của mảng feature.
- Không có throughput **cả pipeline** từ file raw; chỉ có decode-only, batch, BLAS-thread.
- Không có số nào của `csmcir` — adapter từ chối có chủ đích.
- Không có **paper reproduction** cho bất kỳ model nào trong 17.
- Định nghĩa của **G1, G5, G6, G7** không tồn tại ở bất kỳ file nào trong corpus; các chữ G1–G7 xuất
  hiện lại trong `a40/` và `final_pc_audit/` với nghĩa **khác hẳn**. **chưa giải quyết**.

---

## 2. Trạng thái repo / service

### 2.1 Git

| đại lượng | giá trị | nguồn |
|---|---|---|
| `git rev-parse HEAD` | `992b25d254367eec919ea729b2d0df9c09343419` | chạy trong lượt này |
| `git rev-parse origin/research/cir-failure-workbench` | `992b25d254367eec919ea729b2d0df9c09343419` | chạy trong lượt này |
| `git rev-parse origin/main` | `8ebcc74d5fd1934df4f3772a8ca5a96dbd00f333` | chạy trong lượt này |
| `git status --short` | `?? workbench/docs/reviews/20261109-facts/` (thư mục do agent khác, không phải của lượt này) | chạy trong lượt này |
| commit gần nhất | `992b25d` 2026-11-09 00:14:34 +0700 "Replace every unsourced throughput number with what the raw files actually say" | `git log` |

Năm commit sau `1129c58` đều **chỉ chạm `workbench/docs/`**: `116c869`, `1191d26`, `f57fb0d`,
`d0252bb`, `992b25d`. **quan sát được**:

```
git diff --stat 1129c58ba8fa0566c5840cfe820292ae4e9e84f3 HEAD \
  -- . ':(exclude)workbench/docs' ':(exclude)workbench/tests'
→ (rỗng)
```

### 2.2 `TESTED_SOURCE_SHA` là gì và nghĩa là gì

`**TESTED_SOURCE_SHA** | **`1129c58ba8fa0566c5840cfe820292ae4e9e84f3`**` trong
`workbench/docs/reviews/FINAL_PC_RELEASE_VERDICT.md`. Nó là **revision sản xuất đã được đo**, không
phải đầu nhánh. Nghĩa chính xác:

- Cây production tại revision đó **phải byte-identical** với cây đang trên đĩa (trừ `workbench/docs`
  và `workbench/tests`). Test `test_no_release_document_points_at_a_superseded_source_sha`
  (`test_handoff_commands.py:435-500`) assert đúng điều này.
- Vì vậy commit **docs-only không đẩy `TESTED_SOURCE_SHA` đi**, và test vẫn pass. **quan sát được**:
  chạy `pytest workbench/tests/test_handoff_commands.py -q -p no:cacheprovider` → **8 passed in 7.26s**.
- Con số sweep gắn với **`1129c58`**, không gắn với `HEAD`: `2467 passed, 18 failed, 47 skipped =
  2532`, exit 1, 3483.79 s. **quan sát được** (`full_suite_1129c58b.txt` header).

### 2.3 Service đang chạy (đọc từ `/proc`, không suy đoán)

| port | process | cmdline | nguồn |
|---:|---|---|---|
| 5173 | node | `node .../frontend/node_modules/.bin/vite preview --host 127.0.0.1 --port 5173 --strictPort` | `/proc/1019145/cmdline` |
| 8000 | python | `workbench/envs/ctl312h/bin/python -m uvicorn workbench.backend.main:app --host 127.0.0.1 --port 8000` | `/proc/1019132/cmdline` |
| 5432 | — | PostgreSQL (không thuộc workbench) | `ss -ltnp` |
| 20241 | cloudflared | tunnel (không thuộc workbench) | `ss -ltnp` |

### 2.4 Bundle được phục vụ vs `dist/assets`

| kiểm | kết quả | nguồn |
|---|---|---|
| `GET http://127.0.0.1:5173/` → 200, entry `/assets/index-KTV7ID1e.js` | đúng | `curl` |
| sha256 của bundle qua HTTP | `40f34f6d6e61efbf43c34337aa9073d6a1c1d0f9d990d4cb5e1937dbe3418caf` | `curl \| sha256sum` |
| sha256 file trên đĩa `workbench/frontend/dist/assets/index-KTV7ID1e.js` | **cùng giá trị** | `sha256sum` |
| mtime `dist/assets/index-KTV7ID1e.js` | `2026-11-08 22:46:23.748` | `stat` |
| mtime `EvaluationPage.tsx` / `catalog.ts` / `types.ts` | `22:46:06.516` / `22:46:01.284` / `22:45:38.016` | `stat` |
| `workbench/frontend/dist/` có được track không | **không** — `.gitignore:95` | `git check-ignore -v` |
| SPA fallback `/nonexistent-xyz` | trả cùng `index.html` (200) | `curl` |
| bundle có chứa chuỗi `einsum` / `matmul` | **0 lần**; có `scoring_impl` 5 lần, `scoring impl` 1 lần | `grep -o` |

⇒ **Bundle đang phục vụ khớp `dist/assets` trên đĩa, và `dist` được build SAU ba file production của
`224734b`.** **quan sát được.** `dist` là artifact gitignored — nó không nằm trong `TESTED_SOURCE_SHA`,
và `npm run build` không nằm trong sweep nên **không có số nào chứng minh bundle hiện tại tương ứng
chính xác `1129c58`**; cái chứng minh được chỉ là nó mới hơn ba file đó. **suy luận**

`/api/models` trả **17** entry; `/api/composition/runs` trả **27** row nhưng chỉ **9 `run_id` phân
biệt** (mỗi run có 3 row theo condition). **quan sát được** (`curl` + đếm).

---

## 3. Kê khai — 16 / 17 / 29 / 21 / 2

Mọi con số dưới đây **đếm lại trong lượt này** bằng resolver thật, không chép từ prose.

| đại lượng | giá trị | file / cách đếm | trạng thái |
|---|---:|---|---|
| repository upstream duy nhất | **16** | `upstream_repo_url` khác nhau trên 17 entry của `workbench/registry/models.yaml`; bằng đúng **16** thư mục dưới `workbench/third_party/` | **quan sát được** |
| biến thể đăng ký (`model_id`) | **17** | `len(models)` trong `models.yaml`; `/api/models` cũng trả 17 | **quan sát được** |
| biến thể checkpoint khai báo | **29** | một dòng cho mỗi `checkpoint_variants[]` element | **quan sát được** |
| biến thể checkpoint đã staged | **21 / 29** | `registry.checkpoint_path(model_id, variant).exists()` | **quan sát được** |
| checkout không phải model | **2** | `teacher/repos/SPN4CIR` (`BUAADreamer/SPN4CIR` @ `84005f64`) + `data/cirr` (`Cuberick-Orion/CIRR` @ `2276fba2`) | **quan sát được** |
| model có composition adapter | **9** | đếm `composition_adapter` trong registry | **quan sát được** |
| biến `WORKBENCH_*_PYTHON` trong `workbench/config/workbench.env` | **9 dòng** | `grep -n` | **quan sát được** |
| `workbench/artifacts/environment_assignment.json` | **không tồn tại** | `ls` | **quan sát được** |

**8 checkpoint vắng** (không có bytes ở declared root): `conesep/fiq_n05`, `conesep/fiq_n08`,
`habit/fiq_n05`, `habit/fiq_n08`, `intent/fiq_n05`, `intent/fiq_n08`, `sprc/sprc_fiq_unresolved`,
`tme/cirr_0`. **quan sát được** (resolver chạy trong lượt này, khớp `REPOSITORY_INVENTORY.md` §4.1).

### 3.1 CLIP4Cir: một repository, hai biến thể

`combiner_rn50x4_noft` và `clip4cir_rn50x4_fullft` cùng trỏ `ABaldrati/CLIP4Cir`, cùng
`source_dir: CLIP4Cir`, cùng **một** `provenance.json` với `model_ids = ["combiner_rn50x4_noft",
"clip4cir_rn50x4_fullft"]`, cùng biến `WORKBENCH_CLIP4CIR_PYTHON`. Đây là **toàn bộ** lý do 17 ≠ 16;
nó **không** phải repository thứ 17 và không được tách thành con số 17. **quan sát được**
(`REPOSITORY_INVENTORY.md` §2.4).

Đây cũng là cặp **duy nhất** có thể đối chiếu noft-vs-fullft trên cùng một protocol — và hiện cặp này
không chạy được `fullft` (xem §10, B1).

### 3.2 Hai checkout không phải model

| đường dẫn | remote | HEAD | vai trò |
|---|---|---|---|
| `teacher/repos/SPN4CIR` | `BUAADreamer/SPN4CIR` | `84005f643ccaacf999982694ad5631df92cef098` | không registry entry, không adapter, không provenance |
| `data/cirr` | `Cuberick-Orion/CIRR` | `2276fba2d2c53862c8ae79ea18e1f1ae03719f56` | dataset repo, không phải model |

⇒ **18 source tree trên đĩa, 16 thuộc phạm vi workbench.** "16/17" là **registry count** và phải gọi
đúng tên như vậy. **quan sát được**.

### 3.3 Bất đối xứng TME — pin registry không tồn tại trong clone teacher

`teacher/repos/TME` có remote `https://github.com/He-Changhao/2025-CVPR-TME.git` và HEAD
`7fe811992820d30828e24065eb0c7a7ba099dd3b`, còn registry pin `li-shuxian/TME` @ `394aad79df2c…`.
`git -C teacher/repos/TME cat-file -t 394aad79…` → `fatal: could not get object info`. Adapter chạy
tree `third_party` (manifest khớp 349/351, 2 file lệch đúng 2 patch đã duyệt), nên **execution không
bị ảnh hưởng**; cái sai là mọi câu "TME checkout đang ở commit pin". **quan sát được**.

---

## 4. Protocol

### 4.1 Ba nhóm + CIRCUS là protocol **riêng**

| nhóm | protocol id | gallery | reference | số ảnh gallery |
|---|---|---|---:|---:|
| **A** | `fashioniq_original_split` | full val split `split.<cat>.val.json` | **được giữ** | 3817+6346+5373 = **15,536** |
| **B** | `fashioniq_full_gallery_ref_excluded` | full val split | **bị mask `-10e10`** | **15,536** |
| **C** | `fashioniq_val_split` | pair-union `candidate ∪ target`, giữ thứ tự | bị mask | 2628+3089+2902 = **8,619** |
| **CIRCUS** | `circus_fashioniq_release_v1` | danh sách 74,381 identity phát hành, giữ thứ tự | **eligible** (cố ý không kế thừa mask) | **74,381** |

Counts 15,536 và 8,619 **đếm lại trong lượt này** từ `data/FashionIQ/image_splits/split.*.val.json`
và `data/FashionIQ/captions/cap.*.val.json`. **quan sát được**.

CIRCUS **không** phải nhóm thứ tư của upstream — nó là protocol của workbench, chồng lên gallery
phát hành. Vì vậy mọi số cohort CIRCUS khác upstream **đồng thời** ở kích thước gallery, policy
reference, cách pool cohort và bộ metric.

### 4.2 Gán protocol theo từng biến thể — đã verify

`PROTOCOL_MATRIX.md` §"Reconciliation against `workbench/registry/models.yaml`": **17/17 MATCH**, **0
MISMATCH**, **0 UNVERIFIED**. Registry had **0 wrong** — không cần sửa protocol nào.

| nhóm | model_id |
|---|---|
| A | `csmcir`, `ptha_mtst`, `combiner_rn50x4_noft`, `clip4cir_rn50x4_fullft`, `sprc`, `tme` |
| B | `airknow`, `conesep`, `habit`, `intent`, `dcnet` |
| C | `hint`, `encoder`, `pair`, `clvc_net`, `tgcir`, `limn` |

### 4.3 Bốn cái bẫy đã ghi lại

1. **Tên nói dối về việc mask reference.** `airknow`, `conesep`, `habit`, `intent` đều đặt
   `self.split = 'original-split'` **và cả bốn đều mask reference** (`test*.py:151/68/151/150`).
   `dcnet` cùng hình dạng dưới tên `val_trg`. ⇒ tên split **không** nói lên reference policy.
   **quan sát được**.
2. **`original-split` ≠ gallery training.** Mọi nhóm đọc `split.<cat>.val.json`; nhánh train-split của
   TME (`data.py:162-171`) **không chạm tới được** lúc eval vì `precompute_test.py:139` và
   `precompute_train.py:81` hỏi `'val'`. **quan sát được**.
3. **CIRR mask không rò rỉ.** `mask=True` chỉ nằm trong `compute_cirr_val_metrics` / `get_results` /
   `test_cirr_valset` (CSMCIR, MTST_PTHA, CLIP4Cir, SPRC) và trong `if category == 'CIRR':` của TME;
   `compute_fiq_val_metrics` là `mask=False` ở cả 5 file nhóm A. Chiều ngược lại: mask FashionIQ của
   nhóm B/C không đọc đường CIRR. **quan sát được**.
4. **Filter ảnh thiếu của PTHA.** `MTST_PTHA/src/data_utils.py:169-172` lọc trên `<id>.png` — vừa là
   test tồn tại vừa là test phần mở rộng. Trên host này **0** file thiếu, nên gallery hiệu dụng =
   15,536. Nó **là** no-op ở đây nhưng không được bảo đảm ở host khác; replay gọi chính
   `FashionIQDataset` native nên tái hiện filter theo cấu trúc. **quan sát được**.

### 4.4 Defect tài liệu của chính lượt protocol audit (không phải lỗi protocol)

- **4 citation sai thật** (trỏ comment/flag/header thay vì câu lệnh): MTST_PTHA `:164` → thật `:167-172`;
  DCNet `:24` → `:27`; DCNet `:215` → `:217-218`; CLVC-Net `:233` → `:234/:242`.
- **3 "entrypoint" thực chất là library**: `test_BLIP2.py` (AirKnow) và `test.py` của 8 repo **không có
  `__main__`**; `src/validate_blip.py:main()` của MTST_PTHA/SPRC **chỉ dispatch CIRR**.
  README của CLVC-Net ghi `python test.py` — lệnh đó **không làm gì**.
- **Diff vendored thật: 15 khối = 13 file sửa + 2 file thêm trên 8 repo.**
  `PROTOCOL_FINDINGS.md` §7 viết "7 modified + 2 added across 7 repositories" — **prose sai, file raw
  đúng**. Đếm lại: `grep -c '^======== ' raw/vendored_diffs.txt` → **15**. **quan sát được**.
  Đồng thời "vendored tree khớp manifest 100%" **không** có nghĩa "giống upstream": manifest sinh từ
  bytes **sau** `local_fixes`.

---

## 5. Ma trận gate G0–G7

**Copy nguyên văn roll-up từ `GATE_MATRIX.md`. Không nâng bất kỳ ô nào.**

| gate | subject (theo bằng chứng) | phán quyết | con số |
|---|---|---|---|
| **G0** | inventory | **PASS** | 16 repos / 17 variants / 29 checkpoint variants / 21 staged; 2 non-model checkouts |
| **G1** | protocol assignment per pinned upstream | **PASS 17/17** | 0 MISMATCH; 4 wrong line citations + 3 entrypoint misstatements (documentation defects, not protocol errors) |
| **G2** | decode/preprocess bit-identity | **PASS where measured** | habit 10 rows (4 + 6 batch), one sha256 `5fc8002d…`; encoder 16/16 rows, one sha256 `4f6a14f9…` |
| **G2-fwd** | forward feature bit-identity | **NOT_RUN** | GPU owned by the `intent` cohort the whole window |
| **G3** | query-axis regrouping | **PASS** on the shipped path | `d2_gate_full_6003x3.txt`: **18,009/18,009** query-condition pairs rank-identical to the published artifact (multimodal 32.8 q/s, text_only 30.6, image_only 32.6; gallery 74381×32×256, 8 threads, chunk 32). Secondary: 0 mismatches at batch 1 / 7 / 256 vs 32 over a 96-query slice per condition, 0 at repeat, and 512 real queries at 2/4/8 threads. Measured on **habit**'s features; other adapters share the same shipped call path but are not separately measured — do not read this as a per-model pass |
| **G3b** | changed summation order | **FAIL** | 207 / 18,009 rank changes; 0 K=10; 0 K=50; 19 order-only |
| **G4** | axes (batch, tail, repeat, cold/warm) | **PARTIAL** | PASS: repeat determinism (`exact_equal: true`, max 0.0) and batch 1 vs 32 on the **shipped** path. **FAIL on the GEMM fast path**: 6/6003 multimodal rank changes at batch 1 vs 32 (`7:666`, `7:1988`, `7:2003`, `7:2142`, `7:2253`, `7:4730`) — the reason that path was rejected. **NOT_RUN**: cold vs warm cache, the batch axis for `text_only`/`image_only`, cache-disabled, and the incomplete-final-batch case |
| **G5** | cache identity | **DEFECT, partially remediated** — NOT a pass | recorded verdict: 7 identity inputs absent. Closed since: BLAS thread count (`blas_threads_pinned`) and `scoring_module_source_hash`. Still open: TF32/cuDNN flags, GPU capability/driver/CUDA version, transform source (registry labels only), byte-level array verification, and the 851 duplicate gallery feature rows over 2,386 positions |
| **G6** | throughput | **PARTIAL — cannot be PASS for any variant** | Raw files support only: batch saturation (`batch.txt` 17.6/18.0/17.9/18.0 img/s at bs 64/128/256/512), BLAS threads flat for the einsum path (`ein_threads.txt` 6.5/6.0/6.2/5.6/6.2 q/s at 1/2/4/8/10), parallel decode (`vdecode.txt` serial 109.1 → **682.7 img/s at 4 workers**, 588.5 at 8, delta 0.0). **No raw file backs any full-pipeline throughput** |
| **G7** | publishable provenance | **needs verification** — no row passes | **2** artifacts still labelled "pre-generated artifact, not runnable from HEAD" (ENCODER, PTHA+MTST; blocker fixed, **not re-run**); **4** models had published numbers re-keyed by the decode fix; 1 PARTIAL (limn, superseded), 1 FAILED (habit), 2 smoke 8-query |

### 5.1 "Thiết lập được gì / KHÔNG thiết lập được gì", theo từng gate

| gate | **thiết lập được** | **KHÔNG thiết lập được** |
|---|---|---|
| G0 | danh tính repo/pin/checkpoint có thật, hash khớp bytes | model có chạy được không; tương đương float |
| G1 | code upstream tại pin **sẽ** tính gallery/reference như registry khai | số tác giả đã công bố; không chạy evaluator upstream nào |
| G2 | pixel matrix **bit-identical** qua worker×batch cho 2 họ transform | **feature** matrix; cold/warm cache |
| G2-fwd | — | bất cứ điều gì |
| G3 | chia trục query không đổi rank trên **habit**, 18,009 cặp | per-model: 8 adapter còn lại chưa đo riêng |
| G3b | đổi thứ tự cộng **có** đổi rank (207) | "đổi nhưng metric không đổi" là **an toàn** — chỉ là quan sát trên mẫu |
| G4 | repeat determinism; batch 1 vs 32 trên đường shipped | cold/warm; batch axis cho 2 condition còn lại; cache-disabled |
| G5 | BLAS policy + scoring hash nay là identity field | byte-level; GPU capability; transform source |
| G6 | decode song song 6.2× với delta 0.0 | throughput cả pipeline (không có raw) |
| G7 | — | tái lập được từ tree; mọi row đều không PASS |

**Cảnh báo tên gate:** G0 được đặt tên bởi chính tiêu đề `REPOSITORY_INVENTORY.md`; G2/G3/G4 được định
nghĩa bởi `TOLERANCES.md`; **G1, G5, G6, G7 không được định nghĩa ở đâu trong corpus**. Bảng trên dùng
cách đọc rộng nhất bảo vệ được bằng bằng chứng, và nói rõ cách đọc. **chưa giải quyết**.

**Hai chỗ tôi cố ý KHÔNG copy nguyên văn, và lý do** (mọi phán quyết PASS/FAIL/PARTIAL/NOT_RUN đều giữ
nguyên):

1. Ô **G3** trong `GATE_MATRIX.md` có đoạn *"and 512 real queries at 2/4/8 threads / 1.81×–5.67× with 0
   mismatch"*. Tôi **bỏ phần speedup** và giữ phần rank-identity, vì `1.81×`/`5.67×` **chỉ có trong
   prose** — không file raw nào trong `20261108T-throughput/` chứa chúng (§8.2). Tôi cũng bỏ câu
   "einsum shipped 5.6 q/s ⇒ ~55 phút/cấu hình" vì 5.6 q/s có raw (`ein_threads.txt`) còn **~55 phút
   là phép nhân**, không phải phép đo.
2. Ô **G4** nhắc *"the incomplete-final-batch case, which is a structural argument rather than a
   measurement"*; tôi giữ `NOT_RUN` nhưng rút gọn câu, không đổi nghĩa. `6003 % 32 = 19` và lập luận
   tail không ảnh hưởng các batch trước là **suy luận cấu trúc**, không phải phép đo.

Không ô nào bị **nâng** lên. Không ô `NOT_RUN` nào bị viết thành PASS.

---

## 6. Kết quả cohort CIRCUS — 9 model

Mọi ô số dưới đây **đọc trực tiếp** từ `workbench/artifacts/composition/summary/<run>/full.json`
bằng `json.load` (không qua grep/selector — các file này trả `null` giả dưới `?q=`). Không làm tròn.

### 6.1 Bảng cohort

| model | checkpoint | điều kiện nhiễu train | coverage / mẫu số | MM R@10 | MM R@50 | MM nDCG | composition gap (nDCG) | Top-K present? | `scoring_impl` | BLAS policy | run_id |
|---|---|---|---:|---:|---:|---:|---:|---|---|---|---|
| `habit` | `fiq_n02` | **train 20%** noise | 6003/6003 = 1.0 | 35.61552557054806 | 54.3728135932034 | 0.32828280509551133 | 0.388704727214254 | ✅ `TOPK_EXPORT_COMPLETE` depth 50 | `einsum` | BLAS-neutral (Δ 0.0) | `497c25c57fbeba885ecfb9d1` |
| `conesep` | `fiq_n02` | **train 20%** noise | 6003/6003 = 1.0 | 35.48225887056472 | 54.13959686823255 | 0.3294414713812836 | 0.3942551173824983 | ✅ depth 50 | `einsum-optimized` | `unknown (predates the pin)` — GEMM path | `30f6de423861cc1d6c78c361` |
| `intent` | `fiq_n02` | **train 20%** noise | 6003/6003 = 1.0 | 34.39946693320007 | 54.00633016824921 | 0.3184910253798891 | 0.37538250032494136 | ✅ depth 50 | `einsum` | BLAS-neutral (Δ 0.0) | `37ee72aa298b276d1bfda0fa` |
| `hint` | `fashioniq` | không khai (`null`) | 6003/6003 = 1.0 | 32.816924870897886 | 51.77411294352824 | 0.30926916692748224 | 0.43870420817223343 | ✅ depth 50 | `einsum` | BLAS-neutral (Δ 0.0) | `4bc9893a977e501b82378e76` |
| `tme` | `fashioniq_0` | **0%** (clean) | 6003/6003 = 1.0 | 34.01632517074796 | 53.95635515575545 | 0.32038840538867464 | 0.3586012581618244 | ❌ absent | `einsum` | BLAS-neutral (Δ 0.0) | `ad477312aa57c1b9114f9e51` |
| `ptha_mtst` | `cirr_transfer` | **CIRR→FashionIQ transfer** (train trên CIRR) | 6003/6003 = 1.0 | 19.473596535065802 | 34.44944194569382 | 0.22493423438490692 | 0.24362219946484542 | ❌ absent | `einsum-optimized` | `unknown (predates the pin)` | `315dd590eeb03d2c78d7937a` |
| `encoder` | `fashioniq` | không khai (`null`) | 6003/6003 = 1.0 | 31.01782442112277 | 49.5752123938031 | 0.29586071325251023 | 0.47230210227676583 | ❌ absent | `matmul` | `unknown (predates the pin)` | `6a528e877c328772e2c7bb28` |
| `limn` | `base_iter0_all_categories` | không khai (`null`) | **5884/6003** = 0.980176578377478 | 30.064581917063222 | 48.759347382732834 | 0.2908449032087251 | 0.5417445849251903 | ❌ absent | `matmul` | `unknown (predates the pin)` | `10173dc45dea8c90fd2a284b` (**PARTIAL, superseded**) |
| `csmcir` | `fashioniq` | 0% | **0/6003** | — | — | — | — | — | `einsum-optimized` | n/a | **chưa chạy** |

Nguồn từng cột: mẫu số + R@10/R@50/nDCG + gap ← `full.json` → `metrics.multimodal.{scored_count,
recall_at_10, recall_at_50, ndcg_full}` và `metrics.composition_gap_ndcg`; coverage ←
`metrics.multimodal.coverage`; noise ← `checkpoint_training_noise_pct`; Top-K ← `topk_export.status` +
`depth`; `scoring_impl` ← `/api/composition/runs` (backend đọc thân `score()`, không suy từ hash); BLAS
policy ← `BLAS-POLICY.vi.md` §3–§4 và việc identity của các run đó **không có** khoá thread/BLAS;
run_id ← tên thư mục.

Kiểm chéo `scoring_impl` với source: `model_habit.py:258`, `model_intent.py:248`, `model_hint.py:250`,
`model_tme.py:252` dùng `np.einsum("nd,gqd->ngq").max(axis=-1)` **không** `optimize`;
`model_conesep.py:79`, `model_csmcir.py:116`, `model_ptha_mtst.py:87` dùng `optimize=True`;
`model_encoder.py:237` dùng `queries @ gallery.T`; `model_limn.py` chấm từng row `query @ gallery[...].T`.
**quan sát được**.

### 6.2 Ba cảnh báo bắt buộc khi đọc bảng

1. **`csmcir` KHÔNG chạy.** `encode_targets` raise `_TARGET_REASON` (`model_csmcir.py:106`), và
   `mode_supported()` trả `False` cho cả ba condition (`:88-91`). Lý do khoa học, không phải thiếu nỗ lực:
   `score()` của adapter là một embedding text 256-d nhân bản ×32, trong khi evaluator native dùng
   attention-weighted token importance + max trên 32 token — **hai đại lượng khác nhau**. Thêm nữa
   caption chỉ phủ **15,415 / 74,381 = 20.72%** gallery (`model_csmcir.py:58-59`).
2. **`ptha_mtst` là CIRR→FashionIQ transfer, không bao giờ là hàng paper.** `full.json` →
   `checkpoint_transfer: "CIRR→FashionIQ transfer"`, `checkpoint_training_dataset: CIRR`,
   `reproduction_status: SELECTED_TRANSFER_ONLY`. Không được so `19.473596535065802` với
   `reported_r10 54.99`.
3. **Hai `scoring_impl` khác nhau trong cùng một bảng** ⇒ rank từng query **không so sánh được** giữa
   nhóm `einsum` (habit/intent/hint/tme) và nhóm GEMM-path (conesep/ptha_mtst/encoder/limn). Chênh lệch
   aggregate là chênh lệch **metric**, không phải "model này tốt hơn model kia".

### 6.3 Caveat so sánh gallery: 74,381 vs 18,853

`full.json` → `paper_vs_release_gallery` = `{paper_stated_images: 18853, released_images: 74381,
discrepancy: "paper states 18,853; pinned released audit artifact contains 74,381"}`. **quan sát được**.

Khoảng cách R@10 của `habit` (paper 55.38 theo `registry/models.yaml` → `reported_r10`; ours
35.61552557054806 theo `full.json`):

```
55.38 − 35.61552557054806 = 19.76447442945194
```

⇒ **~19.76 điểm**, và phần lớn là **protocol** (gallery 74,381 vs 18,853, policy reference, pool cohort,
bộ metric), **không phải model hỏng**. Con số "roughly 19 points of HABIT's gap is protocol" trong đề bài
là **đúng về bậc độ lớn và đúng về nguồn (registry + artifact)**; phần phân bổ chính xác bao nhiêu điểm
do gallery và bao nhiêu do policy **chưa được đo tách rời** — **chưa giải quyết**.

Cùng phép tính cho các model khác (cùng nguồn, cùng công thức):

| model | paper `reported_r10` | ours | lệch |
|---|---:|---:|---:|
| `habit` | 55.38 | 35.61552557054806 | **19.7645** |
| `conesep` | 55.58 | 35.48225887056472 | **20.0977** |
| `intent` | 55.31 | 34.39946693320007 | **20.9105** |
| `tme` | 55.15 | 34.01632517074796 | **21.1337** |
| `encoder` | 56.13 | 31.01782442112277 | **25.1122** |
| `hint` | 58.48 | 32.816924870897886 | **25.6631** |
| `limn` | 55.91 | 30.064581917063222 | **25.8454** (mẫu số 5884, không so được ở coverage 1.0) |
| `ptha_mtst` | 54.99 | 19.473596535065802 | **35.5164** — vô nghĩa: transfer, không cùng điều kiện train |

Bốn model có adapter nhưng **không** có hàng trong bảng cohort (`airknow`, `pair`, `clvc_net`, `dcnet`,
`tgcir`, `sprc`, `combiner_rn50x4_noft`, `clip4cir_rn50x4_fullft`) không có composition adapter hoặc bị
blocker — xem §10.

### 6.4 Artifact bị loại khỏi mọi claim metric

| run | lý do |
|---|---|
| `habit_fiq_n02_e0dd04a46ef7a58334b9e353` | `FAILED` — `RuntimeError: execution environment not verified [BLOCKED/PYTHON_VERSION_MISMATCH]: …/airknow-adapted/bin/python (expected Python 3.8.10, found 3.11.15)`; `metrics.multimodal.unavailable_reason: "no queries have an exact rank"` |
| `encoder_fashioniq_eda30a37c0bc17ac43844f09` | **smoke 8 query** (`smoke.query_limit: 8`), R@10 12.5 |
| `limn_base_iter0_all_categories_3f86f58483b368377e2c151e` | **smoke 8 query**, R@10 25.0 |
| `SYNTHETIC_encoder_fashioniq_fixture001`, `SYNTHETIC_habit_fiq_n02_fixture002` | fixture (`is_synthetic: true`) |
| `released_audit_task7_mm_any_…` | import baseline, **không phải model run** (`provenance_source`), R@10 52.84024654339497 |

**quan sát được** (14 thư mục `summary/`, đọc từng `full.json`).

---

## 7. Lỗi đã xác nhận và đã sửa

Mỗi mục: (a) sai gì, (b) chứng minh thế nào, (c) sửa gì, (d) blast radius, (e) số cũ có bị vô hiệu không.

### 7.1 F1 — `--seed` là trang trí → **ĐÃ SỬA**

- **(a)** `--seed` đi vào `identity` nên mint `run_id` khác, nhưng **không chỗ nào trong đường
  composition áp nó**. `grep manual_seed|np.random.seed|random.seed|deterministic` trên
  `runner.py`, `adapters.py`, `composition_eval.py` → **0 lời gọi**.
- **(b)** Chứng minh "không có gì rút ngẫu nhiên": mọi transform là
  `TargetPad → Resize(BICUBIC) → CenterCrop → Normalize`; mọi adapter `.eval()` + `no_grad()`;
  `grep '\.generate(\|do_sample\|num_beams\|multinomial\|bernoulli'` trên `workbench/composition/`
  **rỗng**; mọi draw trong third-party đều `self.training`-gated. **quan sát được**.
- **(c)** Bỏ `"seed": seed` khỏi `identity` (comment lý do tại chỗ); summary **vẫn** giữ `"seed"` cho
  provenance và thêm `"seed_applied": False`; help text CLI nói rõ đây là provenance, không đổi run id.
  Giữ tên flag `--seed` vì nó được tham chiếu trong spec/baseline.
- **(d) Blast radius:** **namespace only**. 14 summary trên đĩa: **2** có `seed_applied: false`
  (`hint_4bc9893…`, `conesep_30f6de4…`); **12** không có key đó (pre-F1 hoặc fixture/import).
  Cache 2.3 GB của habit: load dưới fingerprint cũ `True`, dưới post-F1 `False` — hai khoá độc lập
  (path `runs/<run_id>/work/...` và sidecar fingerprint) đều từ chối.
- **(e) Số cũ KHÔNG bị vô hiệu.** `seed` chưa bao giờ được computation đọc, nên bỏ nó khỏi `identity`
  không thể đổi kết quả. Cái mất là **addressability**: run id cũ không tái tính được.
- Test: `test_audit_fix_f1_seed_provenance.py`, 4 check; revert-proof **3 failed / 1 passed** khi gỡ fix.

### 7.2 F3 — cache gallery nhận mà không kiểm shape → **ĐÃ SỬA**

- **(a)** `_encode_gallery` gọi `_load_memmap(destination, fingerprint)` **không** shape, còn
  `_encode_queries` truyền `(len(queries), dimension)`.
- **(b)** Đo lại 4 trường hợp trên helper thật: `.npy` truncate **đã** bị NumPy từ chối
  (`ValueError: mmap length is greater than file size` → cache miss). Lỗ **thật** là cache đọc được
  nhưng **sai số hàng**: `complete 100 rows (gallery 200) -> ACCEPTED shape=(100, 4)`.
- **(c)** `_memmap_metadata_shape(path)` + `expected_shape = (len(image_paths), *metadata_shape[1:])`
  (`runner.py:443-444`).
- **(d)** Không đổi số. Nhưng một artifact pre-fix có cache sai kích thước **có thể** đã tính từ ma trận
  sai — đó là caveat về tính toàn vẹn của artifact đó.
- **(e)** **Không** invalidate cache: fingerprint/identity không đổi; fix chỉ đổi việc **có tin** file.
- Test: `test_audit_fix_f3_gallery_cache_shape.py`, 3 check.

### 7.3 F4 — ENCODER hard-code `checkpoint_mapping_status` → **ĐÃ SỬA**

- **(a)** `model_encoder.py:302-303` (worker) dựng record với `"checkpoint_mapping_status":
  "VERIFIED_METADATA"` **literal**. Literal đó là **một trong bốn gate** của `_unused_checkpoint_keys`
  (`replay/encoder.py:73-102`) → registry hạ status **không** làm adapter chặt hơn.
- **(b)** Đo hành vi sau patch: `VERIFIED_METADATA` → `ACCEPTED`; `UNVERIFIED` → `REFUSED`;
  `None` → `REFUSED`. Khôi phục literal → `REACHED_PAST_MAPPING_GATE`.
- **(c)** Payload thêm `"checkpoint_mapping_status": self._checkpoint.get(...)`; worker đọc
  `job.get(...)`. Dùng `.get` để **fail-closed** khi thiếu key.
- **(d)** Không đổi số. Registry status là `VERIFIED_METADATA`, cùng trọng số được nạp.
- **(e)** Không vô hiệu số; **có** đổi `run_id` vì `_module_source_hash` hash chính file adapter —
  nhưng F4 và F7 nằm **cùng một file**, nên chỉ mint **một** id mới.
- Test: 5 check, trong đó 3 check đường **downgrade** (happy-path-only sẽ pass dù literal còn).

### 7.4 F6 — comment LIMN nêu lý do sai → **ĐÃ SỬA**

- **(a)** `replay/limn_observer.py:162-169` nói tie order tái lập được vì "argsort is stable in NumPy".
  **Sai**: `np.argsort` default là `'quicksort'` (introsort), **không** stable.
- **(b)** Kiểm bằng chính interpreter: numpy 1.26.4, `inspect.signature` = `(a, axis=-1, kind=None,
  order=None)`, docstring nói *"The default is 'quicksort.'"*.
- **(c)** Comment thay bằng câu đúng, cộng cảnh báo rằng làm `kind` explicit — hoặc chỉ làm ở một phía —
  sẽ đổi tie order và phá chính reproduction mà comment đó tồn tại để biện minh. AST của file
  **byte-identical** trước/sau.
- **(d)** Không đổi số. **(e)** Không invalidate cache — `limn_observer.py` **không** nằm trong
  `identity` (`grep limn_observer runner.py adapters.py` rỗng).

### 7.5 F7 — ENCODER + PTHA+MTST **không chạy được từ HEAD** (`NameError`) → **ĐÃ SỬA**

**Đây là phát hiện nặng nhất của cả chương trình.**

- **(a)** Sáu call site gọi helper không import:

  | adapter | call site | helper | import? |
  |---|---|---|---|
  | `model_encoder.py` | `:330` | `limit_torch_threads()` | **không** |
  | `model_encoder.py` | `:331` | `resolved_decode_workers()` | **không** |
  | `model_encoder.py` | `:353` | `prefetch_rows(...)` | **không** |
  | `model_ptha_mtst.py` | `:320` | `limit_torch_threads()` | **không** |
  | `model_ptha_mtst.py` | `:321` | `resolved_decode_workers()` | **không** |
  | `model_ptha_mtst.py` | `:345` | `prefetch_rows(...)` | **không** |

  Hệ quả không hiển nhiên: `NameError` raise **bên trong `_worker_main`**, và `except Exception` của
  chính worker biến nó thành `reply(False, error_type="NameError")` — run chết ở bước nạp model và
  **trông như lỗi model**.
- **(b)** Audit tĩnh toàn bộ 9 adapter với tập 15 helper suy từ public callable của `adapters.py`:
  **đúng hai file hỏng**. Runtime xác nhận: trong `encoder310-adapted`,
  `limit_torch_threads() -> 10`, `resolved_decode_workers() -> 10`, `prefetch_rows` callable; trong
  `_worker_main.__globals__` cả sáu cặp `(module, helper)` trả `in globals=True callable=True` và
  **`is adapters.X`**.
- **(c)** Thêm 3 tên vào block `from workbench.composition.adapters import (...)` sẵn có ở cả hai file
  (HEAD: `model_encoder.py:35-37`, `model_ptha_mtst.py:21`).
- **(d) Blast radius:** ENCODER (`31.01782442112277 / 49.5752123938031`) và PTHA+MTST
  (`19.473596535065802 / 34.44944194569382`) phải mang nhãn **"pre-generated artifact, not runnable from
  HEAD"** cho tới khi có run mới. Module hash đổi ⇒ re-key (`model_encoder.py` `bf60d855be3d72dd…` vs
  HEAD `1a8b0dd25827ca0b…`; `model_ptha_mtst.py` `768bc3ae9a2bd0af…` vs HEAD `546d1a52d2b6ca58…`).
- **(e)** Số **không sai** — code tạo ra chúng đã đúng — nhưng cây hiện tại **không thể** sinh lại
  chúng. Blocker đã gỡ, **chưa re-run**.
- Test: `test_audit_fix_encoder_helpers_imported.py`, 23 check, dùng phân tích `symtable` (bound ở
  enclosing scope nên AST walk thuần sẽ false-positive trên `reply`/`transform`). Revert-proof: gỡ import
  encoder → **3 FAILED**; gỡ ptha → **4 FAILED**; khôi phục → **23 passed**.
- **Vì sao test cũ không bắt:** `test_circus_adapters_*.py` import module và construct `Adapter()`,
  **không bao giờ vào `_worker_main`**. `NameError` trong thân hàm là invisible cho tới khi dòng đó chạy.

### 7.6 LIMN — `query_scope()` có nhưng runner **không bao giờ gọi** → **ĐÃ SỬA**

- **(a)** `model_limn.py` có `def query_scope(...)` (`:186`) đọc category từ chính caption annotation,
  nhưng `runner._scope_for_query` (`:395`) chỉ đọc `query.category` do loader gán. Kết quả: **119 query
  không có category** → PARTIAL, coverage `0.980176578377478`, mẫu số **5884/6003**.
- **(b)** `scope_resolution.json` (đo trên release thật + annotation thật + `model_limn.Adapter` thật,
  không worker, không GPU): `loader_assigned: 5884`, `changed_loader_assigned: []`,
  `dress 2010 / shirt 2035 / toptee 1956`, `None: 2`, `nulls: ["7:3038","7:3790"]`.
  ⇒ **6001/6003 resolvable, 2 genuinely ambiguous**.
- **(c)** Wire resolver với precedence giữ nguyên (`query.category in scopes` return trước, resolver
  không được hỏi); một giá trị ngoài `scopes` **không** được tin; scope đã resolve **phải** tới adapter
  (`_encode_queries`, rank group cats, `_backfill_export`) nếu không hook chỉ là trang trí. Thêm
  `identity.query_scope_resolution` = digest của `sorted((qid, scope))` (`runner.py:1170`) vì map này
  **quyết định ma trận gallery nào mỗi query được chấm**, còn `query_order_identity` hash
  `query.category` vốn `None` đúng ở những query resolver sửa.
- **(d)** `identity_proof.json`: run cũ `10173dc45dea8c90fd2a284b` recompute **đúng** từ identity đã
  lưu; id mới `2a4b19f335f5ec281375d09f`; `resume_would_reuse_old: false`. ⇒ artifact cũ **không** bị
  tái dùng im lặng.
- **(e)** Số `30.064581917063222` là số **mẫu số 5884**, **superseded**; chưa có run 6001/6003.
- Test: `test_limn_scope_runner.py`, 9 test (gồm `test_resolver_value_outside_declared_scopes_is_unresolved_not_trusted`,
  `test_resolved_scope_map_changes_the_run_identity`, `test_resolver_returning_none_leaves_the_query_unscored`).

### 7.7 D1 — `prefetch_rows` submit **toàn bộ** future → **ĐÃ SỬA**

- **(a)** `ThreadPoolExecutor.map` submit cả ~2,327 lô ngay lập tức; `islice` chỉ giới hạn số **kết quả
  kéo về mỗi lần gọi**, không giới hạn số tensor đã decode đang nằm trong bộ nhớ. Decode đo được
  2,089 img/s còn forward ~84 img/s ⇒ pool chạy trước ~25× và giữ các lô đã xong trong future của
  chúng: khoảng **45 GB** cho toàn gallery. (Con số 45 GB là **tính toán từ hai throughput**, không
  phải phép đo bộ nhớ — **suy luận**, và 2089 img/s là prose-only, xem §8.)
- **(b)** Revert-proof (`d1_revert_proof.txt`): `AssertionError: prefetch_rows held 12 batches in flight
  for 64 batches; a bounded pipeline must never exceed workers*2 = 8`.
- **(c)** Cửa sổ trượt thật: chỉ `workers * 2` future mở tại một thời điểm; yield future cũ nhất rồi
  submit thêm đúng một lô (`adapters.py:146-154`, `window = max(1, workers * 2 ...)`).
- **(d)** Test cũ ghi `live["peak"]` mà **không assert** — pass ở mọi độ sâu. Nay assert
  `peak <= workers * 2` và đếm số lô "đã submit nhưng chưa consume" (future đã xong vẫn giữ tensor).
- **(e)** Không đổi số.

### 7.8 D2 — `WORKBENCH_SCORE_THREADS` **đổi cách gộp phép cộng** → **ĐÃ SỬA**

- **(a)** `rank_in_query_chunks` đặt `size = ceil(total / workers)`. Call site cũ truyền `vectors` tối đa
  `batch_size = 32`, nên ở mặc định 8 thread nhóm là **4 query/lần gọi `adapter.score`**, còn ở
  `WORKBENCH_SCORE_THREADS=1` là **32**. Đó là **cách gộp phép cộng khác nhau trong đường publishable**,
  điều khiển bằng env var, mà `run_id` không đổi — cùng lớp lỗi với F1.
- **(b)** Revert-proof (`d2_revert_proof.txt`): `AssertionError: threads=1 produced group sizes [64]`
  thay vì `{16}`.
- **(c)** Kích thước nhóm do **caller** quyết định (`chunk_size`); song song hoá nằm **giữa các nhóm**
  (`_rank_groups_concurrently`). Nên knob chỉ đổi thời gian.
- **(d)/(e)** Chứng minh ở đúng độ chi tiết production, trên **18,009** query-condition,
  `chunk_size=32`, threads=8, so từng rank với artifact `habit_fiq_n02.json` đã publish:
  multimodal 6003 q / 183.0 s / 32.8 q/s / **0 mismatch**; text_only 6003 / 196.0 s / 30.6 / **0**;
  image_only 6003 / 184.3 s / 32.6 / **0**. Tổng **563 s (9.4 phút)**, **0 mismatch**. **quan sát được**
  (`d2_gate_full_6003x3.txt` + `.json`).

### 7.9 D3 / D3b — số thread BLAS **đổi rank** và **không tất định giữa hai lần chạy** → **ĐÃ SỬA**

- **(a)** Song song theo query nhân với BLAS đa luồng sẽ oversubscribe (8 nhóm × 10 thread = 80 thread
  trên 20 core); và tệ hơn: `_rank_groups_concurrently` đặt `_blas_thread_limit(1)` **sau** nhánh
  `if threads <= 1`, nên nhánh **serial** chạy với BLAS = env (10) còn nhánh concurrent chạy BLAS=1.
  Số thread BLAS **không trung tính về số học**.
- **(b)** Ba phép đo:
  - `d3_serial_blas_compare.txt`: serial BLAS=1 vs serial BLAS=10 trên **512 query, scorer `@` thật,
    gallery 74,381 × 512** → `ranks_identical = False`, `queries_changed_rank = 11`,
    `max_abs_drank = 1`, `score_max_abs_delta = 4.172e-07`. Ví dụ: query 77 rank 38550 → 38549; 112
    39674 → 39673; 127 7510 → 7509.
  - `blas_encoder_v2.txt` (encode **thật**, 6003 query): multimodal dNDCG **3.42e-06**, image_only
    **2.18e-05**, text_only **1.30e-08**; 0 vượt K=10/K=50/K=100; R@10/R@50 delta **0.0** cả ba.
  - **Không tất định giữa hai lần chạy cùng máy**: chạy hai lần cùng script, multimodal cho dNDCG
    **8.33e-05** (lần 1) và **3.42e-06** (lần 2) — khác **24×**; text_only đổi từ 3 sang 4 query.
    (`BLAS-POLICY.vi.md` §1b, `D1-D3-DEFECTS.vi.md` D3c).
- **(c)** Hoist `_blas_thread_limit(BLAS_THREADS_PINNED)` **trước** cả hai nhánh, bọc `try/finally`
  (`runner.py:775`, hằng `BLAS_THREADS_PINNED = 1` ở `:597`). Toggle **một lần**, không trong worker —
  phiên bản toggle-trong-worker đã đo được là **gây race làm đổi rank ở 4 và 8 thread**.
  Thêm `identity.blas_threads_pinned` (`runner.py:1198`) vì `_rank_groups_concurrently` nằm trong
  `runner.py`, file mà `_module_source_hash` **không bao giờ** hash.
- **(d) Blast radius:** 5 adapter GEMM-path (encoder, limn, csmcir, ptha_mtst, conesep) tính dưới một
  policy BLAS **không xác định**; 4 adapter naive-einsum (habit, intent, hint, tme) **không** bị ảnh
  hưởng (contraction không có đường BLAS, đo phẳng ở 1/2/4/8/10 thread, delta **đúng 0.0**).
- **(e)** R@10/R@50 tái lập được; **nDCG không** tái lập chính xác dưới policy BLAS=1 (lệch tới
  2.18e-05). Các ΔnDCG chỉ nên đọc như **bậc độ lớn** (1e-8 … 8e-5), **không** phải hằng số.
- Test: `test_both_serial_and_concurrent_branches_pin_blas` (revert-proof: `assert 1 in calls` fail,
  `calls == []` tức nhánh serial không pin lần nào), `test_blas_pin_applies_with_a_single_group`,
  `test_concurrent_groups_are_identical_to_serial_for_matmul`,
  `test_changing_the_blas_policy_changes_the_fingerprint` (`blas_resume_proof.json`: fingerprint
  pin=1 `95bb975daaac3902…`, absent `05dafbbbfed84c2d…`, pin=10 `8f7997872334f49d…`).

### 7.10 Hai mục **ghi nhận nhưng KHÔNG áp dụng**

| ID | nội dung | vì sao không áp |
|---|---|---|
| G1 | `require_batch_fits` under-reserve: projection `batch_size * per_image_bytes` vs `total * 0.85`, **bỏ qua floor cố định ~3.96 GiB** và bộ nhớ đã cư trú. Hồi quy tuyến tính từ `batch.txt`: **~3.96 GiB + ~17.5 MiB/ảnh**. Guard hiện dùng 14 MiB/ảnh, không trừ floor ⇒ ở **bs ≈ 989** nó sẽ duyệt một batch thực tế cần **~20.1 GiB** trên card 16.3 GiB. Guard cũng **chỉ** được nối trong `model_habit.py`; `--batch-size` **không được bảo vệ** ở 7 adapter còn lại | Đề xuất đã viết ra, chưa áp; nằm ngoài phạm vi lượt |
| G2 | `measurement_identity` trong `backend/composition.py` (~520–530) thiếu `scoring_module_source_hash` và `adapter_module_source_hash` ⇒ hai run khác đường scoring hash ra cùng `group_id` và bị gắn nhãn `duplicate` — **sai** | File do agent khác sở hữu |

---

## 8. Tối ưu và số đo — chỉ số có file raw

### 8.1 Bảng số có raw

| phép đo | file raw | kết quả | ghi chú |
|---|---|---|---|
| **Decode song song** (512 ảnh thật, decode+transform **thuần**) | `vdecode.txt` | serial **4.69 s = 109.1 img/s**; 2 worker `1.02 s` (4.6×); **4 worker `0.75 s` (6.2×) = 682.7 img/s**; 8 worker `0.87 s` (5.4×) = 588.5 img/s; `bitwise_equal=True`, `maxabs_delta=0.0` | **8 worker chậm hơn 4** |
| **Bão hoà batch** (2000 ảnh, habit) | `batch.txt` | fp32 bs 64/128/256/512 = **17.6 / 18.0 / 17.9 / 18.0 img/s**; bf16 17.7 / 17.1 / 16.2 | phẳng ⇒ nâng batch **không** phải speedup; peak alloc 3484 → 9544 MiB |
| **BLAS thread cho đường einsum** | `ein_threads.txt` | `old` ở 1/2/4/8/10 thread = **6.5 / 6.0 / 6.2 / 5.6 / 6.2 q/s** | phẳng trên cả 5 mức |
| `optimize=True` vs `old` | `ein_threads.txt` | 27.3 / 22.1 / 33.9 / 25.7 / 22.9 q/s ⇒ **~4.6×** ở 8 thread | `maxdelta 3.576e-07` |
| chunked GEMM 4096 vs `old` | `ein_threads.txt` | 31.2 / 56.2 / 86.3 / **99.7** / 87.7 q/s ⇒ **17.8×** ở 8 thread (`99.7 / 5.6`) | **18× là của GEMM, không phải của `optimize=True`** |
| **Query-axis scoring, đường shipped** | `d2_gate_full_6003x3.txt` + `.json` | multimodal 6003 q / 183.0 s / 32.8 q/s / **mismatches=0**; text_only 196.0 s / 30.6 / 0; image_only 184.3 s / 32.6 / 0. `all_identical: true` | **18,009/18,009 rank-identical** vs artifact đã publish |
| **`optimize=True` bị từ chối** | `eq_fast.txt`, `eq_opt.txt`, `metric_decision_optimize_true.txt`, `metric_gate.json` | 400-query slice: mismatch 0 / 3 / 11 = **14/1200**; full 6003×3: **changed 4 + 35 + 168 = 207**, `cross10: 0`, `cross50: 0`, `accept: true` (accept = "metric aggregate không đổi", **không** phải "rank không đổi") | xem §8.3 |
| **`np.dot` theo thread** | `blas_dot_threads.json` | 1/2/4/8 thread: delta `0.0`, `ranks_changed_vs_1: 0`, cùng fingerprint scores `f8dc27de33a36d7f`; **20 thread: delta `6.103515625e-05`**, ranks 0 | ngưỡng thay đổi ở 20, không ở 8 |
| **BLAS pin là load-bearing** | `d3_serial_blas_compare.txt`/`.json` | serial BLAS=1 vs 10: **11/512 query đổi rank**, max abs drank 1, `score_max_abs_delta 4.172325134277344e-07` | scorer `@`, **dữ liệu tổng hợp** (`np.random.default_rng(17)`) — xem §8.4 |
| **Hoist proof** | `d3_hoist_proof.txt`/`.json` | threads=1 (BLAS pin 1) `210.9 q/s`; serial **không** pin (BLAS=10) `301.8 q/s`; threads 2/4/8 (pin 1) = `310.6 / 549.9 / 556.5`, `ranks_identical_to_threads_1=True` | ⇒ pin làm nhánh serial **chậm ~30%**, đổi lấy tái lập được |
| **LIMN không nhạy BLAS** | `d3b_limn_aggregate_scan.txt`/`.json` | 600 query thật × 3 condition: **changed 0**, max drank 0, mọi cross-K 0, ΔR@10/ΔR@50/ΔnDCG = 0 | LIMN chấm từng row ⇒ BLAS không có chiều M để đổi |
| **ENCODER nhạy BLAS** | `blas_encoder_v2.txt`/`.json` | dscore 2.38e-07 / 2.38e-07 / 2.09e-07; changed **2 / 4 / 1**; ΔnDCG **3.42e-06 / 1.30e-08 / 2.18e-05**; R@10/R@50 delta 0.0 cả ba | encode thật, gallery 74,381 × 512 trong 1.8 min |

### 8.2 Số **KHÔNG** có file raw — phải ghi rõ

| con số | nằm ở đâu | vấn đề |
|---|---|---|
| **`21.2 img/s`** (BEFORE) | chỉ prose `THROUGHPUT.vi.md` | **không có file raw**. `vdecode.txt` đo serial **109.1 img/s**. Hai số chỉ tương thích nếu cấu hình khác nhau — nhưng không file nào ghi cấu hình của 109.1 |
| **`2089.2 img/s`** (AFTER) | chỉ prose | **không có file raw**. `vdecode.txt` cho tối đa **682.7 img/s ở 4 worker**; không có dòng 2089.2 ở bất kỳ file nào trong `20261108T-throughput/` |
| **`2192 img/s`** (8 worker) | chỉ prose | **không có file raw**; `wall × img/s` của chính bảng đó ≈ **1024 ảnh/dòng**, tức **thí nghiệm khác** với `vdecode.txt` (512 ảnh). `grep 2192` trên mọi `.txt`/`.json` của thư mục → **0 hit**. Đây đúng là lỗi mà báo cáo này không được lặp lại |
| **`2291.9 img/s`** (intent AFTER) | chỉ prose | **không có file raw** |
| Bảng `workers / wall / img/s` (618/780/1672/2192/1915) | chỉ prose | **không có file raw**; **và nó nói 8 worker tốt nhất, trong khi `vdecode.txt` đo 4 worker nhanh hơn 8** — tự mâu thuẫn |
| **`5.67×`** (và 1.81× / 3.26×) query-axis | chỉ prose (`THROUGHPUT.vi.md` §5b bảng 106.0/58.7/32.6/18.7 s) | **PROSE-ONLY**. Cột **"rank identical"** thì có nguồn khác: `d2_gate_full_6003x3.txt` (18,009/18,009). Bản thân `THROUGHPUT.vi.md` đã ghi rõ bảng wall/q/s **không có raw**; phép đo authoritative cho `WORKBENCH_SCORE_THREADS` là test `test_query_parallel_ranking_is_identical_to_serial`, nó khẳng định **tính đồng nhất**, không đo throughput |
| `27–43 q/s optimize, 70–75 q/s GEMM dưới BLAS=2` | prose `EQUIVALENCE.md` | `ein_threads.txt` ở 2 thread ghi `optimize` **22.1** và `gemm` **56.2** — **không khớp** |
| `66.5 s cho 6003 query` (trong `metric_gate.json`) | JSON, nhưng | 66.5 s cho 6,003 query ⇒ ~90 q/s, trong khi `ein_threads.txt` đo `optimize=True` ở 22–34 q/s. Một trong hai dùng cấu hình khác, **không file nào nói** |
| "45 GB cho toàn gallery" | prose `D1-D3-DEFECTS.vi.md` | **tính từ** 2,089 img/s vs ~84 img/s; không phải phép đo bộ nhớ |
| "98×" | prose | suy ra từ 21.2 → 2089.2, tức từ hai số prose-only |
| `~5×` end-to-end | prose | suy ra từ `batch.txt` (17.6–18.0 img/s) + forward ~84 img/s; **forward 84 img/s không có raw file** trong template này (nó nằm trong bảng prose `THROUGHPUT.vi.md` §1) |

### 8.3 Vì sao `optimize=True` bị từ chối — lý do G3b, không phải lý do kỹ thuật

| cách | q/s | max abs delta vs shipped | rank changes / 18,009 |
|---|---:|---:|---:|
| `einsum("nd,gqd->ngq")` (production) | **5.6** | 0 (chuẩn) | 0 |
| `einsum(..., optimize=True)` | **25.7** | 3.576e-07 | **207** |
| chunked GEMM (4096) | **99.7** | 3.576e-07 | **207** (cùng tập query, cùng rank mới) |

Phân bố 207: multimodal **4** (max |Δrank| 1), text_only **35** (2), image_only **168** (5).
**0 vượt K=10, 0 vượt K=50**, Top-K **set** khác **0/18,009**, Top-K **order** khác **19**.
ΔR@10 = ΔR@50 = **0.0** cả ba; ΔnDCG = +1.4044159724058147e-09 / +1.5525044860797976e-09 /
−1.5751937930693848e-08; gap −2.1140054151658205e-09.

Ba lý do từ chối, cụ thể:

1. **207 query đổi rank là artifact đã công bố** — Top-K trả cho UI, người dùng thấy thứ tự khác.
2. **0 query vượt biên K ở cohort này là quan sát, không phải bảo đảm** — query ở đúng hạng 9/10/11 sẽ
   vượt biên chỉ vì 1 ULP; không có gì trong phép tính ngăn điều đó.
3. **`run_id` không đổi nếu chỉ sửa thân `score()`** — một rank bị lật sẽ nằm im trong artifact cũ,
   loại lỗi tệ nhất vì không ai truy được.

Cơ chế: mọi rank đổi nằm trên cặp gallery row gần-trùng. Hai cơ chế **khác nhau, không được gộp**:
(i) tie chính xác bị tách bởi 1 ULP (`7:574`: shipped `0.3319489359855652`, multiplicity 2, rank 1628;
fast `0.3319489657878876`, multiplicity 1, rank 1627; margin −2.98e-08); (ii) near-tie thật sự được
quyết định bởi bit cuối (`7:690`: `0.08333621919155121` rank 63717 vs `0.08333626389503479` rank 63716;
margin −4.47e-08). Max |score margin| quan sát: **8.940696716308594e-08**. Số tie interaction trong
các rank đã đổi, **đếm lại từ `fastpath_vs_stored.json` → `paths.opt.margins`** (một phía có
`tie_multiplicity > 1`): **2/4** (multimodal), **18/35** (text_only), **68/120** (image_only — mẫu cap
120 của 168). **quan sát được**. (Prose cũ ghi 2/4, 16/35, 59/120 — xem §14(j): 16 và 59 **không truy
vết được**.)

**Phạm vi:** con số 207 là **một** model (habit) với **một** layout token 32×256. Conesep/csmcir/ptha
dùng subscript `mkd`/`mdk` và **vốn đã** ship `optimize=True` từ trước; encoder và limn dùng `@`.
⇒ hiện **không có một scoring function thống nhất trên 9 adapter**, và độ nhạy rank mới được đo cho
**đúng một** trong số chúng.

### 8.4 Một điều phải nói rõ về bằng chứng BLAS

`d3_serial_blas_compare.py` và `d3_hoist_proof.py` khởi tạo `CompositionAdapter.__new__(...)` và dùng
**dữ liệu tổng hợp** `np.random.default_rng(17)` cho queries/targets (gallery 74,381 × 512 ngẫu nhiên),
không dùng feature thật. Số của chúng trả lời đúng câu hỏi **"BLAS thread có đổi rank của một phép `@`
thật không"** (11/512), nhưng **không** chứng minh ảnh hưởng rank của encoder/limn trên dữ liệu thật.
Bằng chứng trên dữ liệu thật là `blas_encoder_v2.txt` (encode thật) và `d3b_limn_aggregate_scan`
(feature thật). **quan sát được** (đọc source của hai script).

Đồng thời `d3b_encoder_aggregate_scan.txt` báo **ba dòng giống hệt nhau** (mỗi dòng 1 query đổi rank,
ΔnDCG 2.18e-05) — đó **không** phải ba phép đo độc lập: script cũ truyền **cùng một ảnh tham chiếu và
cùng một text gốc ở mọi vòng lặp** và **không gọi `adapt_query_inputs`**, nên cả ba condition chấm
**cùng một input**. Bản sửa (`blas_encoder_v2.txt`) dùng `runner._stage_query_path` và cho ba condition
**khác nhau thật**. Xem §14(b) — đây là một chỗ hai file mâu thuẫn và tôi chọn `v2` là đúng.

---

## 9. Test suite

### 9.1 Hai sweep serial

| | `c2d8cdeb` | `1129c58` (**hiện hành**) |
|---|---:|---:|
| collected | **2528** | **2532** |
| passed | 2463 | **2467** |
| failed | 18 | **18** |
| skipped | 47 | 47 |
| duration | 3464.03 s (0:57:44) | 3483.79 s (0:58:03) |
| exit | 1 | 1 |
| log | `20261108T212500+0700-sweep/full_suite_c2d8cdeb.txt` | `20261108T225338+0700-sweep/full_suite_1129c58b.txt` |
| HEAD before == after | `c2d8cdeb0eabc3232b28070f55b5a10d63ccffcb` | `1129c58ba8fa0566c5840cfe820292ae4e9e84f3` |

`2463 + 18 + 47 = 2528`; `2467 + 18 + 47 = 2532`. **quan sát được** (header của cả hai log).

Delta: +4 id (`test_frontend_scoring_impl.py` đóng góp 4, tất cả pass), **cùng 18 id failure, không
mới, không sửa**. Cả hai lượt đều **serial, không `-n`** — đúng yêu cầu `AGENTS.md` cho số liệu release.

### 9.2 Tally attribution của 18 failure

| hạng | n | id | nghĩa |
|---|---:|---|---|
| **(a)** gây ra bên trong `b0f38bd..HEAD` | **8** | #1–#4, #5, #6–#7, #8 | #1–#4: `adapter.batch_size` được đọc ở `model_conesep.py:183`, thêm bởi `aa0f530`; fixture dựng `Adapter.__new__` và không set. #5: tương tác **thứ tự test** với `test_composition_topk_runner.py`, thêm bởi `9b6b5b6`. #6–#7: test **không thể pass** — fixture set `adapter.interpreter` nhưng gate re-resolve từ registry qua `setting()`, autouse `isolate_operator_state` xoá mọi `WORKBENCH_*`, và `test_circus_tme_lifecycle.py:209` patch **stdlib** `subprocess.Popen` nên probe của chính gate nhận child `sys.exit(9)`. #8: worker chết với `module 'torch' has no attribute 'set_num_threads'` — `adapters.py:182` (thêm bởi `aa0f530`) gặp stub `torch` chỉ có stdlib |
| **(h)** harness/test defect | **0** | — | đã reclassify vào (a) |
| **(b)** host state | **9** | #9, #11–#17, #18 | #9, #18: symlink **untracked** `workbench/third_party/DCNet/dataset/fashioniq` do agent khác tạo 13:43 → `.resolve()` trả tree thật; **proven**: pass trong worktree sạch của `c2d8cdeb`. #11–#17: `release_verify.sh` → `verdict=RELEASE_BLOCKED` vì `workbench/artifacts/gpu_capabilities/` bị nhiễm record |
| **(c)** pre-existing tại `b0f38bd` | **0** | — | — |
| **(d)** test giữ behaviour lỗi thời (đã đóng) | **1** | #10 | `test_no_release_document_points_at_a_superseded_source_sha` — đóng bằng cách đẩy `TESTED_SOURCE_SHA` |
| **tổng** | **18** | | `8 + 0 + 9 + 0 + 1 = 18` |

Đề bài yêu cầu tally **8 / 9 / 1 / 0** — đúng bốn con số này, theo thứ tự (a) / (b) / (d) / (c).

### 9.3 Phát biểu đúng về suite

- **Suite KHÔNG xanh.** 18 failure, exit 1.
- **8 trong 18 ship bên trong range đã giao** (`b0f38bd..HEAD`): 6 do production (#1–#4, #5, #8) và 2
  test **không thể pass trên fresh clone** (#6, #7).
- Cả 18 **không nằm trên đường đo** ⇒ không vô hiệu hoá số của revision này; nhưng "operator state giải
  thích các failure" là **sai**, và ghi chú cũ nói vậy đã bị rút.
- **33 → 18 failure dùng chung 0 id.** Số nhỏ hơn **không** phải bằng chứng code tốt lên: thay đổi
  test-isolation (`9b6b5b6` thêm `isolate_operator_state`) đã xoá 27 id operator-state, trong khi thay
  đổi production của range tạo ra 6 trong 18 cái còn lại. **Đúng chính cái fixture đó** — dùng bởi một
  test **cần** interpreter được cấu hình — là nguyên nhân của #6/#7.

### 9.4 `#11–#17` là **chưa giải quyết**

Chúng được xếp **(b)** dựa trên (i) thư mục bị nhiễm và (ii) cùng họ `RELEASE_BLOCKED` xuất hiện ở
`b0f38bd`. Đếm lại thư mục trong lượt này: **648** file, trong đó **372** `(probe_source: override,
device_name: "Synthetic GPU")`, **114** `(interpreter, None)`, **70** `(override, None)`, **51**
`(forged, None)`, **35** `(override, "fake GPU")`, **6** `(interpreter, "NVIDIA GeForce RTX 5070 Ti")`;
lỗi: 540 `null`, 70 `torch reports CUDA unavailable`, 35 `CUDA out of memory`. Con số
"632 / 465 / 363 / 34" trong `FAILURES.md` là **đếm của lượt trước**, không phải của lượt này —
`FAILURES.md` đã nói rõ nó tự sửa sau một review độc lập.

**Test sẽ quyết định:** chuyển `workbench/artifacts/gpu_capabilities/` ra chỗ khác rồi chạy lại 7 test
(`test_release_blocker_gpu_records.py::test_defect_b_g_with_no_target_gpu_verdict_is_pending_not_pass`
+ 5 × `test_release_verify_documented_invocations_produce_parseable_report` +
`test_offline_verdict_names_deployment_reason`). Pass với thư mục rỗng ⇒ nguyên nhân là nhiễm probe;
vẫn fail ⇒ nguyên nhân nằm trong logic verifier và hạng đổi thành (a) hoặc (c). **Chưa làm** vì thư
mục thuộc run của agent khác.

---

## 10. Blocker còn lại

| # | blocker | ảnh hưởng | cách gỡ | trạng thái |
|---|---|---|---|---|
| B1 | **`csmcir`**: `encode_targets` raise `_TARGET_REASON` (`model_csmcir.py:106`); `mode_supported()` False cả 3 condition; caption phủ 15,415/74,381 = 20.72% | model này **không có** số nào | (a) tái hiện đúng `inference()` attention-weighted thay vì `score()` hiện tại; (b) caption cho 58,966 ảnh còn lại từ nguồn chính thức | **mở**, và **không nên** vá bằng cách đổi semantics |
| B2 | **LIMN 2 query** `7:3038`, `7:3790`: cùng một cặp caption nằm dưới **cả** `shirt` và `toptee` | coverage trần **6001/6003**, không phải 6003 | không có; phải nêu tên trong mọi claim coverage | **chưa giải quyết** (vĩnh viễn với dữ liệu đã pin) |
| B3 | **LIMN chưa re-run** dưới `run_id` mới `2a4b19f335f5ec281375d09f` | số `30.064581917063222` là mẫu số 5884, không so được ở coverage 1.0 | chạy `composition_eval.py --models limn` khi GPU trống | **mở** |
| B4 | **G5 còn 5 input vắng** trong identity: TF32/cuDNN flags, GPU capability/driver/CUDA version, transform source (registry chỉ ghi nhãn), **byte-level array verification**, duplicate rows. (BLAS thread count và `scoring_module_source_hash` **đã đóng**) | cache có thể bị dùng lại dưới một phép tính khác | ghi thêm field; verify byte khi `_load_memmap` | **mở một phần** — không phải gate xanh |
| B5 | **4 artifact publish trước policy BLAS** — chính xác là 4 artifact **GEMM-path** (nhóm mà BLAS thread **có** đổi rank/nDCG): `encoder 6a528e8` (kết thúc 00:00), `limn 10173dc` (04:51), `ptha 315dd59` (06:43), `conesep 30f6de4` (17:42). Khối env BLAS vào ở `3d0e2b8` **17:55:36** và pin ở `bf6a448` **20:33:11**, nên **cả 9 run** đều trước policy — 5 run còn lại (`habit e0dd04a` 05:00, `tme ad47731` 06:14, `habit 497c25c` 15:21, `intent 37ee72a` 16:38, `hint 4bc9893` 17:07) là naive-einsum, BLAS-neutral (Δ đúng 0.0) | 4 artifact GEMM-path **không** so sánh trực tiếp được với run mới dưới policy BLAS=1; identity của chúng có 27–30 khoá top-level và **0** khoá thread/blas/omp | dán nhãn `unknown (predates the pin)` — **đã dán** | **ghi nhận**, không sửa bằng cách suy "=10" (giá trị do shell/ambient quyết định, không được ghi lại) |
| B6 | **ENCODER + PTHA artifact trước fix import** (`6a528e8`, `315dd59`) | nhãn "pre-generated artifact, not runnable from HEAD" vẫn hiệu lực | re-run dưới id mới | **mở** |
| B7 | **Browser harness lệch UI**: `ui_smoke.py` + `e2e_lane.py` viết cho UI một-trang; `4e61f2f` tách trang và không cập nhật. Fail ngay ở session mặc định: chờ `div.cards b` đọc `"17 registered models"` (nay ở `[data-testid="area-overview"] .stat-value`), và `open_page(page,"Models")` trong khi nav nay là tiếng Việt (`Mô hình`, `Khám phá mẫu`, …) | **chưa** browser-verify các khu vực khác lần này; chỉ khu evaluation được probe riêng | re-derive từ `routes.ts` (`AREAS × views`), thay 2 overview wait, re-point control selectors, rồi chạy `--session cold` → `fixture` → viewport lane | **mở**; sửa là **production change** nên cần sweep riêng |
| B8 | **`clip4cir_rn50x4_fullft`**: `workbench/backend/adapters/models.py:374-375` hard-code `combiner_state.pt`/`clip_state.pt`, trên đĩa chỉ có `fiq_comb_RN50x4_fullft.pt`/`fiq_clip_RN50x4_fullft.pt` | variant không chạy được (`FileNotFoundError` ngay) | một dòng: đọc `required_files` từ variant | **mở**, chờ chủ `workbench/backend/**` |
| B9 | **`dcnet`**: patch `weights_only=False` (đúng cho torch ≥2.6) vs pin `torch==1.4.0` (không có tham số đó) — mâu thuẫn hai chiều trên cp37 | `TypeError: 'weights_only' is an invalid keyword argument for load()`, chạy thật đã fail | nâng pin lên 1.13.1, hoặc patch không dùng `weights_only`, hoặc patch có điều kiện version | **mở**, thuộc registry/patch |
| B10 | **`sprc/sprc_fiq_vitl`**: 3 preflight blocker — link canonical dataset, `BLOCKED_EXTERNAL_ASSET_PROVENANCE` cho BLIP-2 (file có, tác giả không công bố sha), và `command_status: COMMAND_BLOCKED_MODEL_MAPPING` là **stale** | không chạy được | tách `command_status` theo variant; khai asset | **mở** |
| B11 | **`clvc_net`, `tgcir`**: 4 lời gọi `.cuda()` không guard trên đường chấm; torch 1.6 chỉ ship kernel tới sm_86, host capability 12.0 | không chạy được trên host này | GPU + torch ≥2.7 có sm_120, hoặc bản `test.py` guard device | **mở** |
| B12 | **`pair`**: MAPPING_CONTRADICTED (pickle cần `model_vssp` + `MHTransformer_dis` mà repo pin không ship; 74 key lạ, 70 shape mismatch) | không chạy được | đúng revision module, hoặc revision PAIR khác | **chưa giải quyết** |
| B13 | **8/17 model không có interpreter resolve được**; `workbench/artifacts/environment_assignment.json` **không tồn tại**; `workbench.env` chỉ set **9** biến `*_PYTHON` | airknow, pair, clvc_net, dcnet, combiner noft, fullft, tgcir, sprc | operator set biến, hoặc provision `environment_assignment.json` với `state: RUNTIME_VERIFIED` | **mở** (operator-owned) |
| B14 | **`tme`**: `full.json` **không** chứa `variant_provenance` ⇒ badge `sm_120` không được ghi vào artifact | artifact đơn lẻ bị đọc như reproduction hợp lệ | một dòng trong `runner.py` ghi `variant_provenance` | **mở** |
| B15 | **`preparation_contracts.yaml` thiếu package**: `setuptools` (CLIP4Cir), `einops` (DCNet), `decord`/`pycocoevalcap`/`webdataset`/`einops`/`scikit-image` (SPRC) | `verify_environment` báo `DEPENDENCIES_OK` trên env **không import nổi** evaluator | bổ sung contract | **mở** |
| B16 | **Không có số nào là paper reproduction.** Mọi bảng so sánh với paper đều qua protocol khác | — | không có cách "gỡ"; phải ghi nhãn | **thuộc tính thường trực** |

---

## 11. Phạm vi kiểm chứng

| hạng mục | đã **executed** | **static-only** | **NOT_RUN** và vì sao |
|---|---|---|---|
| Kê khai 16/17/29/21, 8 vắng | ✅ resolver trên registry, `exists()` trên đĩa | — | — |
| Protocol assignment 17/17 | ✅ đếm split thật (15536, 8619), rank check trên `[REF, WRONG, TARGET]` | đọc source upstream tại pin cho từng repo | evaluator upstream (**không chạy** — cần env riêng + GPU) |
| G0 identity / hash | ✅ sha256 từng checkpoint staged, re-hash 100% manifest vendored | — | — |
| G2 decode/transform | ✅ 2 họ transform × grid worker×batch (habit 4+6 row, encoder 16 row), delta **0.0** | — | 5 họ transform còn lại **chưa** đo riêng (CSMCIR không có vòng lặp lô để song song hoá) |
| **G2-fwd** feature forward | ❌ | — | **NOT_RUN** — GPU thuộc cohort `intent` toàn bộ cửa sổ; chạy song song bị cấm |
| G3 query-axis shipped | ✅ **18,009/18,009** rank-identical, 563 s | — | — |
| G3b `optimize`/GEMM | ✅ full 6003×3, 207 rank changes | — | — |
| G4 batch/tail/repeat | ✅ repeat `exact_equal: true`; batch 1 vs 32 trên đường shipped (96-query slice); GEMM 6/6003 | lập luận cấu trúc cho batch cuối không chia hết | batch axis cho `text_only`/`image_only` (**killed after multimodal**) |
| **G4d cold vs warm** | ❌ | — | **NOT_RUN** — chỉ có **một** warm habit cache (2.4 GB); bản cold cần re-encode **74,381** ảnh |
| G4 cache-disabled | ❌ | — | **NOT_RUN** |
| **G5 full-coverage baseline** | ❌ | sidecar đọc `dtype`/`fingerprint`/`shape` | **không có baseline** — cần **re-encode 74,381 ảnh**; và chưa bao giờ kiểm **byte** của mảng |
| G6 throughput | ✅ decode 512 ảnh (4/2/8 worker), batch 7 dòng, BLAS-thread grid, BLAS=1-vs-10 rank (dữ liệu tổng hợp), encode thật cho ENCODER | — | throughput **cả pipeline** từ raw (**không tồn tại**); `WORKBENCH_SCORE_THREADS` (**chưa đo** bởi audit equivalence — nó tự nói vậy) |
| G7 publishable provenance | đọc 14 `full.json`, so run_id, `seed_applied`, `topk_export` | so source adapter với `scoring_impl` của API | **không row nào PASS** — chưa re-run ENCODER/PTHA/LIMN |
| Cohort CIRCUS 9 model | ✅ 8 model có số đọc trực tiếp; `csmcir` bị từ chối | — | `csmcir` (**BLOCKED_ADAPTER**); LIMN ở coverage 0.98018 |
| Test suite | ✅ 2 sweep serial thật (3464 s, 3484 s) | — | — |
| Browser UI | ✅ khu **evaluation** được probe riêng (9 row, mixed-impl notice, 0 console error) | — | các khu khác **chưa re-verify**: harness lệch (§10 B7) |
| Paper reproduction | ❌ | — | **không có** cho bất kỳ model nào |

**Những gì "chưa từng" xảy ra, phải nói thẳng:** chưa có một phép đo feature forward nào; chưa có phép
đo cold/warm; chưa có một evaluator upstream nào được chạy; chưa có một số nào của `csmcir`; và chưa
có paper reproduction nào.

---

## 12. Đường dẫn

### 12.1 Báo cáo (prose)

| đường dẫn | một dòng mô tả |
|---|---|
| `workbench/docs/reviews/20261109T003209+0700-final/BAO_CAO_CUOI.vi.md` | **báo cáo này** |
| `workbench/docs/reviews/20261108T173736+0700/INPUT_PIPELINE_AND_INFERENCE_AUDIT.vi.md` | audit tổng hợp đường đọc→inference→scoring, 1,458 dòng |
| `workbench/docs/reviews/20261108T173736+0700/GATE_MATRIX.md` | ma trận 29 biến thể × 11 gate + roll-up (authority của §5) |
| `workbench/docs/reviews/20261108T173736+0700/EVIDENCE_INDEX.md` | mỗi thư mục bằng chứng chứng minh gì / không chứng minh gì |
| `workbench/docs/reviews/20261108T152852-audit-inventory/REPOSITORY_INVENTORY.md` | G0: 16 repo, 17 variant, 29 checkpoint, 21 staged, hash từng file |
| `workbench/docs/reviews/20261108T152852-audit-inventory/INVENTORY_GAPS.md` | 11 khoảng trống của kê khai (A1–A3, B1–B4, C1–C3) |
| `workbench/docs/reviews/20261108T155252-protocol-audit/PROTOCOL_MATRIX.md` | G1: gallery/reference/entrypoint từng repo + đối chiếu registry 17/17 |
| `workbench/docs/reviews/20261108T155252-protocol-audit/PROTOCOL_FINDINGS.md` | 4 citation sai, 3 entrypoint sai, 8 bẫy protocol |
| `workbench/docs/reviews/20261108T082857-audit-pipeline/PIPELINE_TRACE.md` | transform/tokenizer/pooling/scoring từng họ, kèm `file:line` |
| `workbench/docs/reviews/20261108T082857-audit-pipeline/UPSTREAM_PARITY.md` | bảng parity HABIT (28 dòng) và ENCODER (22 dòng) |
| `workbench/docs/reviews/20261108T082857-audit-pipeline/DIFFERENTIAL_TESTS.md` | trục A1–A5 + tolerance khai trước |
| `workbench/docs/reviews/20261108T082857-audit-pipeline/FINDINGS.md` | F1–F8, S1–S5 |
| `workbench/docs/reviews/20261108T153045-audit-equivalence/TOLERANCES.md` | 5 lớp so sánh, khai **trước** khi đo |
| `workbench/docs/reviews/20261108T153045-audit-equivalence/EQUIVALENCE.md` | G2/G3/G4 + 207/18,009 + cơ chế 1-ULP |
| `workbench/docs/reviews/20261108T153045-audit-equivalence/FINDINGS.md` | F1–F7 của audit equivalence |
| `workbench/docs/reviews/20261108T155805-audit-fixes/FIXES.md` | F1/F3/F4/F6/F7 applied, kèm revert-proof |
| `workbench/docs/reviews/20261108T155805-audit-fixes/PRE_F1_ARTIFACTS_AND_CACHE_VERDICT.md` | 8 run pre-F1 + cache không tái dùng được (2 khoá) |
| `workbench/docs/reviews/20261108T155805-audit-fixes/MODEL_REPRODUCIBILITY_VERDICT.md` | ENCODER/PTHA không chạy được từ HEAD + audit tĩnh 9 adapter |
| `workbench/docs/reviews/20261108T-throughput/THROUGHPUT.vi.md` | batch/decode/scoring, kèm cảnh báo prose-only |
| `workbench/docs/reviews/20261108T-throughput/D1-D3-DEFECTS.vi.md` | D1 bounded prefetch, D2 group size, D3/D3b BLAS pin |
| `workbench/docs/reviews/20261108T-throughput/BLAS-POLICY.vi.md` | vì sao pin BLAS, và policy của từng run đã publish |
| `workbench/docs/reviews/20261108T212500+0700-sweep/FAILURES.md` | 18 failure @ `c2d8cdeb`, attribution + lịch sử tự sửa |
| `workbench/docs/reviews/20261108T231500+0700-harness/UI_HARNESS_STALE.md` | harness browser lệch UI sau `4e61f2f` |
| `workbench/docs/reviews/20261108T113610-eval/MODEL_RUN_MATRIX.md` | ma trận 9 model có adapter × 3 condition, `scoring_impl`, BLAS policy |
| `workbench/docs/reviews/20261108T113610-eval/BLOCKERS.vi.md` | B1–B9, mỗi blocker 6 mục |
| `workbench/docs/reviews/20261108T180000+0700-limn-scope/EVIDENCE.md` | wire `query_scope()` vào runner + identity proof |
| `workbench/docs/reviews/FINAL_PC_RELEASE_VERDICT.md` | verdict + `TESTED_SOURCE_SHA` + bảng sweep |
| `workbench/docs/PC_PILOT_HANDOFF.md` | hợp đồng bàn giao sang máy RTX 5070 Ti |

### 12.2 Bằng chứng raw đáng mở nhất

| đường dẫn | một dòng mô tả |
|---|---|
| `…/20261108T-throughput/vdecode.txt` | 4 dòng: serial 109.1 img/s, 2/4/8 worker, `bitwise_equal=True` |
| `…/20261108T-throughput/batch.txt` | 7 dòng: bs × dtype → img/s + peak VRAM |
| `…/20261108T-throughput/ein_threads.txt` | 5 mức BLAS × 3 cách scoring, kèm `maxdelta` |
| `…/20261108T-throughput/d2_gate_full_6003x3.txt` + `.json` | 18,009 cặp, `mismatches=0`, `all_identical: true` |
| `…/20261108T-throughput/metric_gate.json` | 6003×3, `changed`/`cross10`/`cross50`/`seconds` cho `opt` và `gemm` |
| `…/20261108T-throughput/eq_fast.txt`, `eq_opt.txt` | slice 400 query: 14/1200 và 3/800 mismatch |
| `…/20261108T-throughput/d3_serial_blas_compare.txt` + `.json` | 11/512 query đổi rank, delta 4.172e-07 |
| `…/20261108T-throughput/d3_hoist_proof.txt` + `.json` | 210.9 / 301.8 / 310.6 / 549.9 / 556.5 q/s, ranks identical |
| `…/20261108T-throughput/blas_encoder_v2.txt` + `.json` | encode thật, 3 condition khác nhau, ΔnDCG 3.42e-06 / 1.30e-08 / 2.18e-05 |
| `…/20261108T-throughput/d3b_limn_aggregate_scan.txt` | 600 query thật, changed 0 ở cả 3 condition |
| `…/20261108T-throughput/blas_dot_threads.json` | 1/2/4/8 thread delta 0.0; 20 thread 6.103515625e-05 |
| `…/20261108T-throughput/blas_resume_proof.json` | 3 fingerprint khác nhau cho pin=1 / absent / pin=10 |
| `…/20261108T-throughput/d1_revert_proof.txt` | `held 12 batches in flight … 12 <= (4 * 2)` |
| `…/20261108T-throughput/d2_revert_proof.txt` | `threads=1 produced group sizes [64]` |
| `…/20261108T-throughput/published_run_timestamps.txt` | mtime rank artifact của 12 run + `now` |
| `…/20261108T212500+0700-sweep/full_suite_c2d8cdeb.txt` | sweep serial thứ nhất, header + `FAILED` lines |
| `…/20261108T225338+0700-sweep/full_suite_1129c58b.txt` | sweep serial hiện hành, 18 `FAILED` lines |
| `…/20261108T153045-audit-equivalence/decode_habit.json`, `decode_habit_batch.json`, `decode_encoder.json` | grid worker×batch, `bit_identical: true`, 1 sha256 mỗi file |
| `…/20261108T153045-audit-equivalence/divergence_detail.json` | `order_only_divergences` = 19; `fastpath_agreement` |
| `…/20261108T153045-audit-equivalence/cache_and_ties.json` | 7 input vắng; 851 nhóm trùng / 2386 vị trí / lớn nhất 54 |
| `…/20261108T153045-audit-equivalence/gap_and_axes.json` | aggregate stored vs `opt` vs `gemm1024` full precision |
| `…/20261108T153045-audit-equivalence/encoder_transform.json` | train-vs-val: max 3.8269808292388916; cùng seed 0.0 |
| `workbench/artifacts/composition/summary/*/full.json` (14) | nguồn **duy nhất** của mọi số cohort ở §6 |
| `…/20261108T180000+0700-limn-scope/scope_resolution.json` | dress 2010 / shirt 2035 / toptee 1956 / `None` 2 |
| `…/20261108T180000+0700-limn-scope/identity_proof.json` | run cũ recompute đúng; id mới; `resume_would_reuse_old: false` |
| `workbench/artifacts/composition/runs/*/ranks/*.json` | `identity` + `fingerprint` + `ranks` của từng run (28–29 khoá top-level) |

### 12.3 Lệnh tái lập rẻ nhất (tất cả **không cần GPU**, < 10 s)

```bash
# 1) Đếm lại 16 repos / 17 variant / 29 checkpoint / 21 staged
workbench/envs/ctl312h/bin/python - <<'PY'
import yaml, sys; sys.path.insert(0, '.')
from workbench.backend import registry
d = yaml.safe_load(open('workbench/registry/models.yaml'))
urls = {m['upstream_repo_url'] for m in d['models']}
rows = [(m['model_id'], v) for m in d['models'] for v in m['checkpoint_variants']]
print(len(d['models']), 'variants;', len(urls), 'repos;', len(rows), 'checkpoints;',
      sum(1 for mid, v in rows if registry.checkpoint_path(mid, v).exists()), 'staged')
PY
# kỳ vọng: 17 variants; 16 repos; 29 checkpoints; 21 staged

# 2) Guard handoff (8 test) — cũng là thứ chứng minh TESTED_SOURCE_SHA vẫn đúng
workbench/envs/ctl312h/bin/python -m pytest workbench/tests/test_handoff_commands.py \
  -q -p no:cacheprovider
# kỳ vọng: 8 passed

# 3) Cây production có khớp revision đã test không
git diff --quiet 1129c58ba8fa0566c5840cfe820292ae4e9e84f3 HEAD -- . \
  ':(exclude)workbench/docs' ':(exclude)workbench/tests' && echo "production tree identical"

# 4) Đọc số cohort đúng cách (json.load, KHÔNG dùng ?q= selector)
workbench/envs/ctl312h/bin/python - <<'PY'
import json, glob, os
for p in sorted(glob.glob('workbench/artifacts/composition/summary/*/full.json')):
    j = json.load(open(p, encoding='utf-8'))
    mm = j['metrics']['multimodal']
    print(f"{os.path.basename(os.path.dirname(p))[:44]:44s} {mm['scored_count']:5d} "
          f"{mm['recall_at_10']!s:22s} {j['metrics']['composition_gap_ndcg']}")
PY

# 5) Pre-F1 vs post-F1
grep -o '"seed_applied": [a-z]*' workbench/artifacts/composition/summary/*/full.json
# chỉ hint_4bc9893… và conesep_30f6de4… có false; phần còn lại KHÔNG có key

# 6) 207 rank changes có đúng nguồn không
workbench/envs/ctl312h/bin/python -c "
import json; d=json.load(open('workbench/docs/reviews/20261108T-throughput/metric_gate.json'))
print(sum(v['opt']['changed'] for v in d['report'].values()),
      [v['opt']['cross10'] for v in d['report'].values()])"
# kỳ vọng: 207 [0, 0, 0]

# 7) 19 order-only divergence + loại tie
workbench/envs/ctl312h/bin/python -c "
import json, collections
d=json.load(open('workbench/docs/reviews/20261108T153045-audit-equivalence/divergence_detail.json'))
o=d['order_only_divergences']
print(sum(v['count'] for v in o.values()),
      collections.Counter(r['kind'][:20] for v in o.values() for r in v['rows']))"
# kỳ vọng: 19  Counter({'NOT a shipped-path ti': 16, 'exact tie broken by': 3})

# 8) 15 khối diff vendored (không tin prose "7+2")
grep -c '^======== ' \
  workbench/docs/reviews/20261108T155252-protocol-audit/raw/vendored_diffs.txt
# kỳ vọng: 15

# 9) Bundle đang phục vụ == dist trên đĩa
curl -s http://127.0.0.1:5173/assets/index-KTV7ID1e.js | sha256sum
sha256sum workbench/frontend/dist/assets/index-KTV7ID1e.js
```

---

## 13. Việc tiếp theo, theo thứ tự ưu tiên

| # | việc | vì sao trước | chi phí ước lượng |
|---:|---|---|---|
| 1 | **Re-run ENCODER và PTHA+MTST** dưới `run_id` mới (`--topk-depth 50 --keep-features`) | Đây là bước **duy nhất** chuyển "pre-generated artifact, not runnable from HEAD" thành artifact tái lập được. Blocker đã gỡ, chưa re-run | ~15–25 phút/model nếu pipeline đã nhanh; serial, GPU độc quyền ⇒ **~1 h cho cả hai** |
| 2 | **Re-run LIMN** với scope resolver (id dự kiến `2a4b19f335f5ec281375d09f`) | Số 30.06 hiện là mẫu số 5884, superseded; coverage sẽ là 6001/6003 | ~15–25 phút + encode 3 checkpoint |
| 3 | **Đo G2-fwd** cho **ít nhất 2 họ transform** (một lavis, một open_clip) khi GPU trống | Gate duy nhất còn trống **giữa** "decode đúng" và "số đúng" | encode cùng 512 ảnh hai cấu hình batch/worker, so **feature matrix**: ~10–20 phút + GPU |
| 4 | **Đo `WORKBENCH_SCORE_THREADS` bằng một file raw** (cùng harness `equivalence_fastpath.py --verify-sample`) | Để `5.67×` có nguồn, hoặc vĩnh viễn ghi nó là prose-only | ~15 phút CPU |
| 5 | **Đo cold vs warm cache** (G4d) trên habit, **sau** (1) | Cần một cache cold ⇒ re-encode **74,381** ảnh | ~15 phút encode + ~10 phút đo |
| 6 | **Đóng nốt G5**: ghi TF32/cuDNN flags + GPU capability/driver/CUDA version vào `identity`; **verify byte** mảng feature khi đọc cache | 2/7 đã xong; 5 còn lại. Byte-level là mục quan trọng nhất | 1 buổi; byte-verify có thể tốn I/O trên 2.4 GB/cache (giây đến phút/cache) |
| 7 | **Quyết định scoring function** — giữ bất đối xứng (4 einsum / 5 GEMM) **kèm caveat rank giữa hai nhóm không so được**, hoặc chuyển đồng loạt sang `optimize=True`/GEMM với run_id mới và **re-run toàn bộ** | Trạng thái hiện tại — hai thứ tự cộng trong một cohort đã publish — là lựa chọn **tệ hơn cả hai** | Quyết định: 0 phút. Nếu chuyển: 8 model × ~20 phút ≈ **3 h** + viết lại bảng |
| 8 | **Sửa B8** (`adapters/models.py:374-375`, một dòng) để mở khoá `clip4cir_rn50x4_fullft` | Mở khoá luôn khả năng đối chiếu cặp noft/fullft trên cùng protocol — cặp duy nhất làm được việc đó | 1 dòng + preflight; chờ chủ file |
| 9 | **Sửa browser harness** (B7) | Sửa là **production change** ⇒ cần **một sweep serial ~58 phút** + advance `TESTED_SOURCE_SHA` | re-derive ~1 buổi + 58 phút sweep |
| 10 | **Đóng #11–#17** bằng cách chuyển `workbench/artifacts/gpu_capabilities/` ra chỗ khác rồi chạy lại 7 test | Quyết định dứt điểm hạng (b) vs (a)/(c) cho 7 failure | ~5 phút + 7 test |
| 11 | **Bổ sung package thiếu** vào `preparation_contracts.yaml` (B15) | `verify_environment` đang báo `DEPENDENCIES_OK` trên env không import nổi evaluator | ~30 phút + đo lại 3 env |
| 12 | **Ghi `variant_provenance` vào summary** (B14) | Artifact TME `sm_120` hiện bị đọc như reproduction hợp lệ | 1 dòng + sweep nếu là production |
| 13 | **Quyết định `csmcir`** và **quyết định SPN4CIR** (đăng ký biến thể 18 hay xoá clone) | Cả hai là quyết định sản phẩm, không phải kỹ thuật | 0 phút (quyết định) |
| 14 | **Ghi lại định nghĩa G1/G5/G6/G7** nếu chương trình audit có chúng ở nơi khác; nếu không, bỏ các nhãn đó khỏi tài liệu release | §5 và §1.3 phụ thuộc vào việc này | 0 phút (quyết định) + sửa docs |

**Không được làm, và lý do nằm trong chính báo cáo này:** bật `optimize=True` làm mặc định "cho nhanh"
(207 rank change, §8.3); nới tolerance cho decode-only (che vi phạm, `TOLERANCES.md` lớp B); suy
BLAS policy của run cũ là "10" (giá trị do shell quyết định, không ghi lại); và gọi suite là xanh khi
còn 18 failure với 8 defect trong range (§9.3).

---

## 14. Những chỗ nguồn mâu thuẫn, và tôi chọn bên nào

| # | mâu thuẫn | tôi chọn | vì sao |
|---:|---|---|---|
| a | `PROTOCOL_FINDINGS.md` §7 nói "7 modified + 2 added across 7 repositories"; `raw/vendored_diffs.txt` có **15 khối diff** (13 sửa + 2 thêm, 8 repo) | **file raw** | Đếm lại `grep -c '^======== '` = 15. Prose sai, file đúng |
| b | `d3b_encoder_aggregate_scan.txt` báo **3 dòng giống hệt** (mỗi dòng 1 query đổi rank, ΔnDCG 2.18e-05); `blas_encoder_v2.txt` báo **3 condition khác nhau** | **`blas_encoder_v2.txt`** | Script cũ truyền cùng một input cho cả 3 vòng và **không gọi `adapt_query_inputs`**; bản v2 dùng `runner._stage_query_path` + assertion intervention. D1-D3-DEFECTS đã rút bảng cũ |
| c | `EQUIVALENCE.md` §G3b nói "optimize 27–43 q/s, GEMM 70–75 q/s dưới BLAS=2"; `ein_threads.txt` ở 2 thread ghi **22.1** và **56.2** | **`ein_threads.txt`** | File raw thắng prose; và đây đánh dấu toàn bộ nhóm "27–43" là prose-only |
| d | `audit-equivalence/FINDINGS.md` F2 nói "4 trong 19 là exact-tie artefact"; `divergence_detail.json` cho **3** (theo `shipped_exactly_equal`) và **9** (theo "một trong hai phía") | **JSON** | Dữ liệu thô thắng prose. Không đổi phán quyết (FAIL về rank identity), nhưng "4/19" là số **không có nguồn** |
| e | `THROUGHPUT.vi.md` bảng prose nói **8 worker là điểm tốt nhất** (2192); `vdecode.txt` đo **4 worker nhanh hơn 8** (682.7 vs 588.5) | **`vdecode.txt`** | File raw. Bảng prose là thí nghiệm khác (~1024 ảnh/dòng) |
| f | `REPOSITORY_INVENTORY.md` §2 nói HEAD là `b2fca876`; thực tế đó là commit **cuối cùng chạm `models.yaml`**, HEAD repo lúc đó là `6421188` | **diễn giải "base so sánh của file registry"** | `git diff` cho cùng thống kê 189/46 với cả hai mốc, nên kết luận không đổi |
| g | `FAILURES.md` nói `gpu_capabilities/` có 632 record (465 override, 363 Synthetic, 34 CUDA-OOM); lượt này đếm **648** file (372/114/70/51/35/6) | **không chọn bên nào — cả hai là đếm tại thời điểm khác nhau** | Thư mục dùng chung, bị ghi thêm giữa hai lượt; chính `FAILURES.md` đã ghi rõ nó tự sửa sau review độc lập. Kết luận "thư mục bị nhiễm" giữ nguyên |
| h | `INPUT…AUDIT.vi.md` §6.6 ghi habit `seed_applied: null` (pre-F1); thực tế `full.json` **không có key** `seed_applied` | **"key vắng mặt"** | Đọc raw: 12/14 summary không có key; chỉ `hint` và `conesep` có `false`. `null` và "vắng" cùng nghĩa vận hành (pre-F1) nhưng khác nhau về JSON |
| i | `MODEL_REPRODUCIBILITY_VERDICT.md` ghi SHA file adapter **sau fix** (`bf60d855…`, `768bc3ae…`); mọi run artifact hiện có mang SHA **trước fix** | **cả hai đúng ở mốc của nó** | File đã đổi sau khi run được tạo; đó chính là lý do nhãn "pre-generated artifact" còn hiệu lực |
| j | `EQUIVALENCE.md:96` và `INPUT…AUDIT.vi.md:1203` nói tie interaction là **2/4, 16/35, 59/120**; `fastpath_vs_stored.json` → `paths.opt.margins` (đếm lại trong lượt này: `shipped_recomputed.tie_multiplicity > 1` **hoặc** `fastpath_tie_multiplicity > 1`) cho **2/4, 18/35, 68/120** | **JSON** | Dữ liệu thô thắng prose. "16/35" và "59/120" là số **không truy vết được**; không đổi phán quyết (FAIL về rank identity) và không đổi 207 |
| k | `PC_PILOT_HANDOFF.md` §"Kết quả R14" nói LIMN `R10 55.6315` "khớp paper trong 0.3 điểm"; cohort CIRCUS nói `30.064581917063222` | **không mâu thuẫn** | Hai thứ khác nhau: R14 là **native** `fashioniq_val_split` trên gallery 8,619 với replay riêng; cohort CIRCUS là **composition** trên gallery 74,381. Không con số nào của cohort này là reproduction, kể cả khi cùng model |

---

## Kết luận ngắn

Lượt này **không** nói "cả 17 model đều đúng". Nó nói:

- **Kê khai đứng vững** (G0 PASS, 16/17/29/21/2), **protocol đứng vững** (G1 PASS 17/17, 0 sửa registry),
  **decode bit-identical ở nơi đã đo** (G2 PASS cho 2 họ transform).
- **Feature forward chưa từng được đo** (G2-fwd `NOT_RUN`); **cache identity vẫn là defect** (G5, còn 5
  input vắng + 851 nhóm trùng); **throughput cả pipeline không có raw file** (G6 PARTIAL theo cấu trúc);
  **không artifact nào PASS G7**.
- **8/9 model có adapter có số**; `csmcir` không chạy được và việc không chạy là **đúng**. `ptha_mtst` là
  transfer, `conesep`/`habit`/`intent` mang nhãn nhiễu 20%, `limn` ở coverage 0.98018.
- **Suite không xanh**: 2467/18/47 = 2532, và 8 trong 18 failure là defect ship trong range.
- **Mọi số cohort là số của protocol CIRCUS trên gallery 74,381, không phải paper.** Khoảng cách ~19.76
  điểm R@10 của `habit` phần lớn là protocol.
