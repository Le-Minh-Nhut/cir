# Phase A — PROTOCOL_MATRIX (báo cáo ngắn)

Ngày 2026-11-10. HEAD `2e2e2d2`, branch `research/cir-failure-workbench`.
Sản phẩm: `PROTOCOL_MATRIX.csv` (26 dòng × 58 cột), sinh bằng
`scripts/pa_protocol_matrix.py` từ các file đo được trong `raw/`:

| file | nội dung | script sinh ra |
|---|---|---|
| `raw/protocol_cohorts.json` | cohort của 3 protocol FashionIQ: gallery theo thứ tự, sha256, đếm query/gallery, dup, target/ref trong gallery | `scripts/pa_protocol_hashes.py` |
| `raw/circus_cohorts.json` | gallery 74381 + hash, subset full/sf/validated, multi-positive, category resolution | `scripts/pa_circus_cohorts.py` |
| `raw/membership.json` | membership + sự tồn tại file ảnh (`.png` vs `.jpg`), union 15415 | `scripts/pa_membership.py` |
| `raw/checkpoint_hashes.tsv` | sha256 của 15 checkpoint, resolve qua `checkpoint_path()` | `scripts/pa_ckpt_hashes.py` |
| `raw/registry_dump.txt` | protocol + model facts đọc từ registry | `scripts/pa_registry_dump.py` |
| `raw/scout/*.txt` | trích nguyên văn `path:line` cho từng evaluator | 4 scout (read-only) |

## 1. Bốn protocol key được nêu — kết quả

| label | gallery | ref | đếm | kết luận |
|---|---|---|---|---|
| `fashioniq_original_split` | full `split.<cat>.val.json`, đúng thứ tự file | **eligible** | 3817/6346/5373; 6016 query | **CONFIRMED** — `src/evaluation/fashioniq.py:172-186`, `models.yaml:15-18` |
| `fashioniq_full_gallery_ref_excluded` | **cùng** gallery trên (sha256 trùng từng category) | **masked** | như trên | **CONFIRMED** — `fashioniq.py:187-189`, `models.yaml:19-22`. Khác duy nhất ở mask |
| `fashioniq_val_split` | pair union (ref trước, target sau) | masked | 2628/3089/2902 | **CONFIRMED** — `fashioniq.py:190-192`, `models.yaml:23-26` |
| `circus_fashioniq_release_v1` | `gallery_paths` của audit artifact, 74381, 0 dup | eligible (ref nằm trong gallery nhưng không phải positive) | 6003 / 4069 / 586 | **CONFIRMED** — `workbench/composition/data.py:11-13,196-315` |

Kiểm chứng thêm (đo, không suy luận):

* Gallery pair-union do workbench dựng **trùng thứ tự** với union của upstream
  (`images.index(candidate)` rồi `images.index(target)`) trên cả 3 category.
* `fashioniq_original_split` và `fashioniq_full_gallery_ref_excluded` có **cùng** `gallery_sha256_ordered`
  (`dress d193f087…`, `shirt 2c2a2382…`, `toptee cae7b6f0…`) — hai tên, một cohort ứng viên.
* Đơn vị: native evaluator trả **phần trăm**; riêng DCNet trả **phân số** (`0.2793` = 27.93 %).
  Cột `raw_units` ghi từng dòng. Matrix không làm tròn ở đâu cả.

## 2. Label tìm thêm được (và một label sai)

* `fashioniq_val_split_ref_included` — **không phải protocol riêng**; chỉ còn là alias evaluator-side
  trỏ về `fashioniq_original_split` (`fashioniq.py:175-186`). Alias này **không** được
  `RunMetadata` chấp nhận cho artifact đã lưu (`schemas/results.py:12-16,136`).
  Còn sót một fixture cũ khai nó là `native_protocol` của TME:
  `workbench/tests/fixtures/pc_launcher/models.json:1723-1725` — **stale**, lệch registry.
  (Bàn giao: fixture thuộc worker khác; đề nghị sửa 3 dòng đó thành `fashioniq_original_split`.)
* `p1` — chuỗi trong `artifacts/reports/mock_c1_burn-probe-final_*.json` (`model_id=mock`,
  `checkpoint_id=c1`). **Không phải protocol**; đã đánh dấu DISPROVED để không ai đếm nó vào cohort.

## 3. Run mà tôi cho là ghi sai protocol — kèm bằng chứng

**`clip4cir_rn50x4_fullft`** ghi `fashioniq_full_gallery_ref_excluded`; đúng phải là
`fashioniq_original_split`.

* `raw/native/clip4cir_fullft.json` → `protocol`, `protocol_source` đều nói "reference is excluded
  by masking its own index entry".
* Pinned upstream **không** làm vậy: `third_party/CLIP4Cir/src/validate.py:21-57`
  (`compute_fiq_val_metrics`) không có lệnh ghi mask nào; nó `F.normalize` index (`:41`),
  `distances = 1 - pred @ index.T` (`:44`), `argsort` ascending (`:45`), label theo target (`:49-50`).
  `reference_mask` **chỉ** tồn tại trong `compute_cirr_val_metrics` (`:162-166`, hàm bắt đầu `:135`).
* Registry đã đúng từ trước: `models.yaml:1184` → `fashioniq_original_split`,
  `supported_protocols: [fashioniq_original_split]`. Vậy **sai ở artifact, không phải ở registry**.
* Gallery của chính run: 3817/6346/5373 = full split ⇒ không phải cohort pair-union.
* Đây là **lỗi provenance**, không phải lỗi ranking: cùng gallery, chỉ khác luật loại reference, và
  luật đó không hề chạy. Không cần rerun theo R1 để sửa nhãn.

Đã sửa bằng migration có thể kiểm lại (không sửa tay artifact gốc của chiến dịch trước):

```
workbench/scripts/migrate_protocol_label.py            # công cụ
workbench/tests/test_protocol_label_migration.py       # 5 test, PASS
workbench/docs/reviews/20261110-e2e-repair/raw/r1_migration_evidence.json
workbench/docs/reviews/20261110-e2e-repair/raw/clip4cir_fullft.migrated.json          # bản đã migrate
workbench/docs/reviews/20261110-e2e-repair/raw/clip4cir_fullft.migrated.json.migration.json
```

Migration từ chối khi (a) không có citation, (b) `gallery_count` trong artifact mâu thuẫn cohort được
yêu cầu, (c) đổi sang chính nhãn cũ. Nó đóng băng `aggregate`, `categories`, `checkpoint`, `source`,
`runtime`, `protocol_source` và assert bằng nhau; đã kiểm: `aggregate identical True`,
`categories identical True`, khác biệt duy nhất là `migrated_from` + nhãn.

Chạy lại:
```
PYTHONPATH=$PWD workbench/envs/ctl312h/bin/python -m pytest workbench/tests/test_protocol_label_migration.py -q
```

## 4. Việc chưa làm (Phase B/C)

Phase B (`RUN_MATRIX.csv`) và Phase C (R2, R3, R4, R10, R11) **chưa bắt đầu** — đúng thứ tự operator
yêu cầu: báo Phase A trước, không khởi động job GPU nhiều giờ. Không có số nào của R2–R11 trong tài
liệu này được coi là đã đo.
