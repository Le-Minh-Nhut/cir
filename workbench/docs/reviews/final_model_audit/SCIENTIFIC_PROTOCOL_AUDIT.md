# SCIENTIFIC PROTOCOL AUDIT — FASHIONIQ

Ba protocol là **bất biến khoa học** của workbench. Tài liệu này chứng minh từng cái bằng code đã
ghim, và chứng minh rằng chúng **không thể trộn** — bằng cả test lẫn đường từ chối ở runtime.

## 1. Ba protocol, chứng minh từ source

| | A `fashioniq_original_split` | B `fashioniq_full_gallery_ref_excluded` | C `fashioniq_val_split` |
|---|---|---|---|
| Gallery | `split.{cat}.val.json`, thứ tự file | `split.{cat}.val.json`, thứ tự file | union duy nhất reference+target, thứ tự xuất hiện |
| Reference | **ELIGIBLE** (không mask) | EXCLUDED, mask `-10e10` | EXCLUDED, mask `-10e10` |
| Kích thước | dress 3817 / shirt 6346 / toptee 5373 | như A | dress 2628 / shirt 3089 / toptee 2902 |
| Thành viên đã kiểm chứng | csmcir, combiner_rn50x4_noft, clip4cir_rn50x4_fullft, sprc | airknow, conesep, habit, intent, dcnet | hint, encoder, pair, clvc_net, tgcir, limn |

### Bằng chứng cho Protocol A (reference vẫn eligible)
`workbench/third_party/CLIP4Cir/src/validate.py:44-55` và `SPRC/src/blip_validate.py:43-56`:
`distances = 1 - predicted_features @ index_features.T` rồi `argsort` **trực tiếp**, không có bước
mask nào. Đối chiếu: cùng file đó, nhánh CIRR (`validate.py:164`) **có** mask — chứng tỏ tác giả mask
có ý thức khi protocol yêu cầu, nên sự vắng mặt ở FashionIQ là đặc điểm của protocol, không phải lỗi.

### Bằng chứng cho Protocol B (reference bị loại, gallery đầy đủ)
`AirKnow/test_BLIP2.py:151`, `ConeSep/test.py:68`, `HABIT/test.py:151`, `INTENT/test.py:150`:
```python
sims[i, test_targets_id.index(t['source_img_id'])] = -10e10
```
kèm `datasets.py` mặc định `split='original-split'` và lặp toàn bộ `split.{cat}.val.json`.
`DCNet` mask ở **cả hai nhánh điểm** (`trainer/trainer_joint.py:216-218`) trước khi fuse
`F.log_softmax(score_comp) + F.log_softmax(score_corr)` — kiểm tra riêng vì hợp nhất hai nhánh là chỗ
dễ sót mask.

### Bằng chứng cho Protocol C (union + loại reference)
`LIMN/datasets.py:158-169`, `HINT/datasets.py:154-166`, `ENCODER/datasets.py:185-196`,
`PAIR/data/datasets.py:148-159`, `CLVCNet/datasets.py:243-256`, `TGCIR/datasets.py:148-160`:
dựng danh sách theo thứ tự xuất hiện, thêm reference rồi target nếu chưa có. Mask ở
`LIMN/test.py:82` (và tương ứng ở các model khác).

## 2. Sự khác biệt được chứng minh bằng số

Test `workbench/tests/test_limn_query_export_protocols.py` dựng một fixture mà **cùng một query** cho
ra **ba rank khác nhau**, nên không thể pass nếu implementation trộn protocol:

```
split:   ["imgA", "imgB", "imgZ", "imgC", "imgD"]   # imgZ không thuộc annotation nào
query:   reference=imgA, target=imgB
scores:  imgA 0.9 (chính là reference), imgZ 0.8 (người lạ), imgB 0.7 (target)

A: gallery đầy đủ + reference eligible  -> imgA, imgZ, imgB  -> rank 3
B: gallery đầy đủ + reference excluded  -> imgZ, imgB        -> rank 2
C: union         + reference excluded   -> imgB              -> rank 1
```

Cơ chế khác biệt được kiểm chứng **trực tiếp** (chạy `torch.argsort`, in ra `[0, 2, 1, 3, 4]`), không
suy luận. Test cũng chốt rằng ở `k=1` cả A lẫn B đều là 0 — `imgZ` che target trong cả hai — nên
protocol **không thể phân biệt bằng một `k` nông duy nhất**; đó là lý do các khẳng định trong test
được đặt ở `k` phù hợp thay vì ép một đẳng thức sai.

## 3. Cơ chế chống trộn ở runtime

| Lớp | Cơ chế | Bằng chứng |
|---|---|---|
| Schema | `RunMetadata.validate_metadata` buộc `literature_split_label` khớp `protocol_id` | `schemas/results.py` |
| Schema | `PROTOCOL_IDS` chỉ có đúng ba tên | `test_the_protocol_ids_are_the_three_declared_names` |
| Registry | `native_protocol` phải thuộc ba tên, hoặc `null` | `test_registry_membership_matches_the_declared_protocols` |
| Phân tích | `validate_analysis_selection` từ chối nhiều protocol (`cross_protocol`) | `test_cross_protocol_analysis_is_refused_by_the_api` |
| Phân tích | `aligned_queries` buộc cùng `raw_captions`, khác `model_input_text` thì được | `test_runs_with_different_raw_captions_cannot_be_compared` |
| Benchmark | `COMPARABILITY_FIELDS` gồm `protocol_id`, `split`, `dataset`; lệch → `BLOCKED_PROTOCOL_MISMATCH` | `paper_gate.py` |
| Tham chiếu | dòng không khai protocol **không** khớp bất kỳ protocol nào | `test_an_unresolved_reference_protocol_never_wildcard_matches_a_run` |

## 4. Vụ `ptha_mtst` — đã sửa

Đây là vi phạm protocol **thật** và là phát hiện nghiêm trọng nhất về mặt khoa học của lượt này:

* Bài báo chỉ nói *"R@10 and R@50 on the val set in three categories"* — **không** nêu cách dựng
  gallery hay chính sách reference.
* Registry đúng khi ghi `native_protocol: null`. Nhưng hai dòng benchmark lại gán
  `gallery_protocol: fashioniq_original_split` + `reference_excluded: false`, viện dẫn
  `validate_blip.py:51` của một repo **chưa từng được sync hay pin** (`source_available: false`).
* Hệ quả nếu để nguyên: bộ chọn so khớp sẽ bind một run vào reference có protocol **chưa ai chứng
  minh**, và `benchmark_gate` có thể chứng thực một so sánh giữa hai candidate universe khác nhau —
  đúng loại P0 mà mục 8 của yêu cầu cấm.
* Đã sửa ở ba tầng: dữ liệu extracted, `papers.yaml`, và `matches()`. Kèm refusal nói đúng sự thật
  ("publish metrics but declare no protocol") để người vận hành không đi tìm một dòng không tồn tại.

## 5. Điều **không** được làm

* Không tính Jaccard thất bại xuyên protocol.
* Không so sánh common distractor xuyên protocol.
* Không tính đồng thuận xuyên protocol.
* Không gán protocol cho `ptha_mtst` chỉ để cho đủ 16.
* Không dùng nhãn văn liệu "original"/"val" thay cho semantics thật của evaluator — hai model
  `airknow`/`conesep`/`habit`/`intent` mang nhãn `literature_split_label: original` nhưng protocol
  thật là B (reference excluded), **khác** A. Nhãn văn liệu và protocol là hai trục riêng biệt trong
  schema, và điều đó được test chốt.
