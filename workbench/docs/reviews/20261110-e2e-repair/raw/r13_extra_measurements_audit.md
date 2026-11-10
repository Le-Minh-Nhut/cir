# R13 extra — ba measurement file bị báo là "missing native"

Phạm vi: chỉ đọc. Không chạy model nào. Gốc: `workbench/artifacts/checkpoint_selection/20261106T002617+0700/`
(viết tắt `$D`). HEAD `2e2e2d2`.

## 0. Kết luận trước

| file | verdict | bằng chứng quyết định (một dòng) |
|---|---|---|
| `measurements/tgcir_run1.json` | **EXECUTED_PROVEN** | `$D/logs/tgcir_run2.log` chứa đúng ba dòng per-category của file này + `aggregate` khớp tới từng chữ số, sau khi chạy hết 2017/2038/1961 query và 2628/3089/2902 gallery |
| `measurements/clvc_run1.json` | **EXECUTED_PROVEN** | `$D/logs/clvc_run1.log` chứa đúng ba dòng per-category và `aggregate {"macro_r10": 29.970746517160162, ...}`, kết thúc `EXIT=0` |
| `measurements/airknow_fiq_n05_run1.json` | **EXECUTED_PROVEN** | dòng cuối `$D/logs/airknow_n05_run1.log`: `[OK] result written: .../measurements/airknow_fiq_n05_run1.json` |

Cả ba **đều** là native run đã thực thi. Chi tiết dưới đây, kèm hai điều cần nói rõ vì chúng dễ bị
đọc sai.

## 1. Run directory dưới `workbench/artifacts/logs/`

`ls workbench/artifacts/logs` → **21 thư mục**, và **không** có thư mục nào cho tgcir, clvc, airknow:

```
ls -d workbench/artifacts/logs/*tgcir*  → No such file or directory
ls -d workbench/artifacts/logs/*clvc*   → No such file or directory
ls -d workbench/artifacts/logs/*airknow*→ No such file or directory
```

21 thư mục đó thuộc `combiner_rn50x4_noft` (2), `csmcir` (7), `dcnet` (1), `synthetic_model` (11).
Vậy câu trả lời cho (1) là: **không có run directory kiểu `<model>__<checkpoint>__<run_id>`**.

**Nhưng kết luận "không có log" là SAI.** Ba model này chạy trước khi có quy ước `artifacts/logs/`;
log của chúng nằm ở `$D/logs/` — cùng cây với measurement, không phải `artifacts/logs/`:

```
$D/logs/airknow_n05_run1.log   15 374 B
$D/logs/airknow_n08_run1.log   15 374 B
$D/logs/clvc_run1.log          20 353 B
$D/logs/tgcir_run1.log          4 413 B   <-- run THẤT BẠI
$D/logs/tgcir_run2.log         15 793 B   <-- run THÀNH CÔNG
```

`$D/logs/` có 18 file log, mtime 2026-11-06 00:44 – 01:17, tức trước `artifacts/logs/` (2026-11-08).
Đây là lý do một audit chỉ nhìn `artifacts/logs/` sẽ kết luận sai rằng ba model này chưa từng chạy.

## 2. File measurement có tự khai evaluator / command / run_id không?

**Không, ở cả ba.** Top-level keys:

| file | top-level keys | `run_id` / `command` / `cmd` / `invocation` / `provenance` |
|---|---|---|
| `tgcir_run1.json` | `aggregate, categories, model_id, protocol` | tất cả **None** |
| `clvc_run1.json` | `aggregate, categories, model_id, protocol` | tất cả **None** |
| `airknow_fiq_n05_run1.json` | `aggregate, categories, category_model_policy, checkpoint_training_noise_pct, complete_benchmark, eval_mode_variant, evaluation_noise_pct, model_id, protocol, replay_type, runtime, timestamp` | `run_id`/`command`/`cmd`/`invocation`/`provenance` **None**; chỉ có `timestamp = 2026-11-05T17:44:46.422038+00:00` |

`provenance` chỉ tồn tại **trong từng category**, và nội dung khác nhau:

`tgcir_run1.json`, category `dress` — đúng như bạn nêu, **không có evaluator**:

```json
{"source_commit": "65fa78eaf8cabe8197fcc22ccab42207c591bdee",
 "checkpoint": ".../downloads/tgcir/x/checkpoints/Fashioniq.pt",
 "img_encoder": "ViT-B/16"}
```

`clvc_run1.json`, category `dress` — cũng **không có evaluator**:

```json
{"source_commit": "bd9b6889489806cb60676597880f91d918b279cb",
 "local": ".../checkpoints/clvc/dress_local_model.pt",
 "global": ".../checkpoints/clvc/dress_global_model.pt"}
```

`airknow_fiq_n05_run1.json`, category `dress` — **có** evaluator, và nhiều hơn thế:

```json
{"native_evaluator": "test_BLIP2.py:test",
 "native_output": {"dress_r1": 19.186911254338124,
                   "dress_r10": 47.892910262766485,
                   "dress_r50": 70.84779375309866},
 "checkpoint_training_noise_pct": 50, "evaluation_noise_pct": 0,
 "model_mode": "train (native behavior)",
 "checkpoint_sha256": "f4d25ed9e532f2ee3671f6edadb1048e73113246ec056b413bde75dfe3f79bc6",
 "checkpoint_size_bytes": 2724056638,
 "preprocessing": "targetpad transform, target_ratio 1.25, input_dim 224",
 "text_processor": {"config": "configs/models/blip2/blip2_pretrain.yaml", "loaded_via": "lavis.models"}}
```

⇒ Chỉ **airknow** tự khai entrypoint. `tgcir` và `clvc` **không**, nên với hai model này file
measurement một mình **không đủ**; phải dựa vào log.

## 3. `tgcir` — số không phải transcription, và có gì trên đĩa

### 3.1 Không phải transcription của tác giả

`$D/downloads/tgcir/x/checkpoints/Fashioniq_metrics.json` (file của tác giả):

```
dress_r10  45.21566683192861   shirt_r10  52.600588812561334   toptee_r10  56.144824069352374
dress_r50  69.65790778383739   shirt_r50  72.52208047105005    toptee_r50  77.10351861295257
```

macro ba category = **51.3204** (r10). Measurement nói **49.7208** với dress 43.3813 / shirt 51.4720 /
toptee 54.3090. Khác biệt có hệ thống và **thấp hơn** ở cả ba category. Vậy đúng: **không** phải
chép lại số tác giả — nó là một lần chạy thật cho ra số khác.

### 3.2 Có gì trên đĩa ghi lại cách sinh số

**Có, và đủ.** Ba thứ:

1. **Script**: `$D/scripts/run_tgcir.py` (59 dòng). Docstring:

   > "Session runner for pinned TG-CIR `test.test(params, model, testset, category)`. Why needed: the
   > pinned `test.py` has no CLI; the registry wrapper is preflight-only. This calls the native
   > function with the pinned FashionIQ constructor and the author transform (`train.py:88-93`,
   > `clip.load('ViT-B/16').preprocess`) and the checkpoint's own configuration (`local_token_num=8`,
   > `global_token_num=4` from the author's `Fashioniq_metrics.json`)."

   Nó `sys.path.insert` vào `workbench/third_party/TGCIR`, `import test as tg_test`, rồi gọi
   `tg_test.test(...)` — tức **gọi hàm native của upstream**, không viết lại metric.

2. **Log của run thất bại**: `$D/logs/tgcir_run1.log` (4 411 B) là **traceback**, `EXIT=1`:

   ```
   File ".../scripts/run_tgcir.py", line 39, in <module>
     recalls = dict(tg_test.test(...))
   ...
   AttributeError: 'MultiheadAttention' object has no attribute 'batch_first'
   ```

   Đây là bằng chứng mạnh: nó chứng minh `run_tgcir.py` **đã thực sự chạy** và **đã đi vào**
   `TGCIR/test.py:32` → `model.py:202` → `model.py:191` → `model.py:96` trước khi chết.

3. **Log của run thành công**: `$D/logs/tgcir_run2.log` (15 793 B) có 6 progress bar — 3 cho query
   (2017 / 2038 / 1961) và 3 cho gallery (2628 / 3089 / 2902) — rồi:

   ```
   {"category": "dress",  "r1": 14.377788795240455, "r10": 43.38125929598414, "r50": 68.368864650471}
   {"category": "shirt",  "r1": 22.42394504416094,  "r10": 51.47203140333661, "r50": 69.8233562315996}
   {"category": "toptee", "r1": 22.59051504334523,  "r10": 54.30902600713922, "r50": 75.11473737888832}
   aggregate {"macro_r10": 49.720772235486656, "macro_r50": 71.10231942031965, "macro_mean": 60.411545827903154, "complete": true}
   ```

   Ba dòng `r10`/`r50` **trùng tới từng chữ số** với `measurements/tgcir_run1.json`
   (`43.38125929598414`, `51.47203140333661`, `54.30902600713922`), và `aggregate` trùng
   `49.720772235486656` / `71.10231942031965`.

### 3.3 Điều phải nói rõ: tên file lệch

`measurements/tgcir_run1.json` **không** được sinh bởi `logs/tgcir_run1.log` — log đó thất bại
(`EXIT=1`, không có dòng `category` nào). Nó được sinh bởi **`logs/tgcir_run2.log`**. Tên file
measurement lệch một đơn vị so với log sinh ra nó.

Đây là khiếm khuyết ghi chép thật, nhưng **không** làm số liệu mất hiệu lực: nội dung là số của run
thành công, và run thành công có log đầy đủ với kích thước dataset đúng (2017/2038/1961 query,
2628/3089/2902 gallery — khớp `split.*.val.json`). Cần đổi tên hoặc thêm `run_log` field vào
measurement.

## 4. `clvc` và `airknow` — chi tiết

### `clvc_run1.json` / `logs/clvc_run1.log`

- Log 326 dòng, mở đầu bằng progress bar 2017 query, kết thúc:

  ```
  {"category": "toptee", "r1": 7.8531361550229475, "r10": 32.78939316675166, "r50": 63.895971443141256}
  aggregate {"macro_r10": 29.970746517160162, "macro_r50": 58.62171212901515, "macro_mean": 44.29622932308766, "complete": true}
  EXIT=0
  ```

- Ba dòng per-category trong log trùng **tuyệt đối** với `measurements/clvc_run1.json`
  (`29.350520575111553`, `27.77232580961727`, `32.78939316675166`).
- `provenance.source_commit = bd9b688…` trùng `upstream_commit` trong
  `workbench/third_party/CLVCNet/provenance.json`.
- `native_evaluator` **None**: không khai entrypoint. Nhưng log + checkpoint path + `EXIT=0` đủ.

### `airknow_fiq_n05_run1.json` / `logs/airknow_n05_run1.log`

- 249 dòng; mở đầu `FashionIQ dataset initialized with 18000 samples.`; có
  `FutureWarning` từ `workbench/third_party/AirKnow/lavis/models/blip2_models/blip2.py:42` — tức
  **code upstream đang chạy**, không phải reimplementation.
- Dòng cuối: `[OK] result written: .../measurements/airknow_fiq_n05_run1.json` — tự khai file đích.
- `provenance.native_evaluator = "test_BLIP2.py:test"` + `native_output` + `checkpoint_sha256`.
- `replay_type: SOURCE_FAITHFUL_WORKBENCH_REPLAY` **không** có nghĩa synthetic:
  `workbench/replay/airknow.py:1-12` nói module "runs the native FashionIQ evaluator" và chỉ bù
  invocation shell vì `test_BLIP2.py` không có `__main__` hay checkpoint loader.

## 5. Headline "11/15 native, bốn missing" có phải đổi không?

**Có.** Câu hiện tại (`FINAL_REPORT.vi.md`) nói airknow, clvc_net, tgcir, pair là bốn cái còn thiếu
native. Ba trong bốn (`airknow`, `clvc_net`, `tgcir`) có measurement + log thực thi. Chỉ `pair` là
thật sự không có.

Nhưng **không** được đổi thành "14/15". Đếm theo hai chuẩn khác nhau:

- **Đã thực thi, có log chứng minh** — 14/15: thêm `airknow`, `clvc_net`, `tgcir`; còn `pair`.
- **Đã thực thi VÀ tự khai entrypoint upstream trong measurement** — chỉ 12/15: `airknow` có
  `native_evaluator`; `clvc_net` và `tgcir` **không** có field đó (bằng chứng nằm ở log, không ở
  measurement).

Câu bảo vệ được, nói đúng cả hai chuẩn:

> "14 trong 15 target active có native evaluation đã thực thi, mỗi cái có log stdout trong
> `$D/logs/` khớp tới từng chữ số với measurement tương ứng. Trong đó **12** khai được entrypoint
> upstream ngay trong measurement (`provenance.native_evaluator`); `clvc_net` và `tgcir` chỉ chứng
> minh được qua log + script gọi hàm native. `pair` là target duy nhất chưa có native run. Hai
> khiếm khuyết ghi chép cần sửa: `tgcir`'s measurement tên `run1` nhưng được sinh bởi `run2.log`,
> và `clvc`/`tgcir` thiếu field `native_evaluator`."

Không đổi được thành 15/15: `pair` không có measurement mang macro, và `measurements/pair_b*.json`
chỉ có hypothesis/batch probe, không có recall.

## 6. Điều chỉnh cho census trước đó

`raw/native_measurement_census.txt` đếm **12 model_id** có `aggregate.macro_r10`. Con số đó **vẫn
đúng** — nó đếm *file có macro*, không phải *model đã chạy*. Ba model này đã nằm trong 12
(`airknow`, `clvc_net`, `tgcir` đều có mặt). Cái census **không** phân biệt được là model nào trong
12 có log chứng minh thực thi; đó là việc của file này.

## 7. ĐÍNH CHÍNH — census v1 của tôi đã đếm thiếu

`raw/native_measurement_census.txt` chỉ quét `**/measurements/*.json` với đúng một hình dạng
`aggregate.macro_r10`. Nó **bỏ sót** bốn hình dạng khác, nên câu "5 active config không có file
native" ở §0 của file này là **SAI**. Quét lại toàn bộ (`raw/r13_census_v2.md`) tìm thấy:

| model | file | giá trị | hình dạng |
|---|---|---|---|
| **csmcir** | `20261107T021500+0700/csmcir/measurement.json` | r10 **57.0507** / r50 **77.2986** | top-level `macro_r10` |
| **ptha_mtst** | `20261106T233409+0700/measurements/ptha_transfer_run1.json` | r10 **37.8978** / r50 **59.8364** | `macro_metrics.{r10,r50}` |
| **combiner_rn50x4_noft** | `measurements/clip4cir_parsed.json` | r10 **35.1407** / r50 **58.4436** | `clip4cir_noft.macro_r10` |
| **clip4cir_rn50x4_fullft** | `measurements/clip4cir_parsed.json` | r10 **43.8182** / r50 **66.9501** | `clip4cir_fullft.macro_r10` |
| **pair** | `pair_compat/measurements/pair_b1_H_normal.json` (+ 3 file) | r10 **34.0350** / r50 **56.5957** | `macro.macro_r10` |

Nên **17/17 model đều có ít nhất một file mang macro**. "Không có file" không còn đúng cho model nào.

### 7.1 Nhưng "có macro" ≠ "native run hợp lệ" — ba trường hợp phải loại

- **pair**: `mapping_status = "EXPLORATORY_PUBLIC_MAPPING"`, `class_mapping` là ánh xạ tự chế
  (`"model_vssp.Encoder": "model.PAIR"`, `"MHTransformer_dis.MlpTransformer": "MHTransformer.MlpTransformer"`).
  Đây là **replay thử nghiệm**, không phải native evaluation của checkpoint đã release. Số 34.04 nằm
  rất xa `paper_reference.macro_r10 = 52.76` — chính file tự ghi reference. ⇒ **không đếm**.
- **csmcir**: `measurement.json` có `evidence_class` và `equivalence_to_historical_cot_input` — cần
  đọc hai field đó trước khi đếm; nó thuộc nhóm **deferred** trong registry, nên dù có macro nó
  **không** nằm trong 15 active.
- **ptha_mtst**: `macro_metrics` có thật, `protocol_id = fashioniq_original_split`, nhưng **không** có
  `provenance.native_evaluator`, không `run_id`, không log trong `$D/logs/`. Chưa xác minh được nó
  được sinh thế nào ⇒ xếp **EXECUTED_UNPROVEN** cho tới khi tìm được log/script.

### 7.2 Headline đã sửa lại lần hai

Câu "14/15 đã thực thi có log, chỉ pair thiếu" ở §5 **cũng cần chỉnh**, vì ptha chưa chứng minh được:

> **13/15** target active có native evaluation đã thực thi **và** có bằng chứng sinh số (log stdout
> hoặc `native_evaluator` trong measurement). `ptha_mtst` có measurement mang macro nhưng **chưa**
> có log/entrypoint ⇒ EXECUTED_UNPROVEN. `pair` chỉ có replay thử nghiệm ⇒ không đếm. `csmcir` và
> `sprc` là deferred, ngoài 15 active.

Đây là con số bảo vệ được. Muốn nâng lên 14/15 thì phải tìm được log của ptha; muốn nâng pair thì
phải có native mapping, không phải `EXPLORATORY_PUBLIC_MAPPING`.

## 8. ptha_mtst — ĐÃ tìm được bằng chứng thực thi (nâng lên 14/15)

Câu "ptha chưa chứng minh được" ở §7.1 **sai** — tôi tìm chưa hết. Bằng chứng đầy đủ
(`raw/r13_ptha_proof.md`):

| thứ | đường dẫn | nội dung quyết định |
|---|---|---|
| log | `20261106T233409+0700/logs/ptha_transfer_run1.log` (719 dòng) | progress bar `0/60` → `60/60`, `FutureWarning` từ `third_party/MTST_PTHA/.../blip2.py:42` (code upstream đang chạy), dòng cuối `exit=0` |
| command | `20261109-native15-composition-limn/raw/native/ptha_mtst.command.sh` | `python workbench/replay/ptha_mtst.py --checkpoint-root .../ptha_mtst --transfer-cirr-to-fashioniq` |
| meta | `.../ptha_mtst.meta.txt` | `EXITCODE=0 WALL_SECONDS=240 PEAK_GiB=25.79` |
| measurement | `20261106T233409+0700/measurements/ptha_transfer_run1.json` | `macro_metrics {r10: 37.8978, r50: 59.8364}` |

Lưu ý quan trọng về **bản chất**: log ghi `"mismatch": "training dataset/checkpoint differs;
numerical context only, not reproduction"` và `differences_transfer_minus_paper` = −17.09 r10 /
−16.44 r50. Đây là **CIRR→FashionIQ transfer**, không phải FashionIQ-trained checkpoint. Nó **là**
native evaluation của một checkpoint thật, nhưng **không** phải reproduction của paper. Đếm vào
"đã thực thi", không đếm vào "tái lập paper".

## 9. Con số cuối

| nhóm | n | model |
|---|---:|---|
| đã thực thi + có bằng chứng sinh số | **14/15** | tất cả trừ `pair` |
| trong đó khai `native_evaluator` ngay trong measurement | 12 | + ptha chỉ có log/command, không field |
| không đếm | 1 | `pair` (`EXPLORATORY_PUBLIC_MAPPING`) |
| ngoài 15 active | 2 | `csmcir`, `sprc` (deferred) |

> **14/15** target active có native evaluation đã thực thi, mỗi cái có log stdout hoặc command+meta
> với `EXITCODE=0`, khớp với measurement mang macro. `pair` là cái duy nhất còn thiếu: chỉ có replay
> `mapping_status = "EXPLORATORY_PUBLIC_MAPPING"`, 34.04 so với `paper_reference.macro_r10 = 52.76`
> do chính file đó ghi.
