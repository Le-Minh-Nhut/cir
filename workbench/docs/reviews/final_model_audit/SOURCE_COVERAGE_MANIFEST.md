# SOURCE COVERAGE MANIFEST

Phân biệt rõ: **inventoried** (liệt kê được) ≠ **reviewed** (đọc và truy vết tới evaluator). Không có
phần trăm coverage nào được bịa. Mọi con số dưới đây đếm được bằng `git ls-files` / `find`.

## 1. Source bên thứ ba đã ghim (upstream)

| Repo | File inventoried | File reviewed (đường tới evaluator) | Loại trừ và lý do |
|---|---|---|---|
| `LIMN` (`7d7bc9b1`) | 88 | `LIMN/test.py`, `LIMN/datasets.py`, `LIMN/model.py`, `LIMN/Transformer.py`, `correct_files/correction_dict_{dress,shirt,toptee}.json` | `LIMN+/` là biến thể self-training lặp (hệ thống khác, không tương đương); `cirr_test_submission.py` chỉ CIRR; `MHTransformer.py` là dead code |
| `CSMCIR` (`774f94e2`) | 1140 | `src/validate_blip_csmcir.py`, `src/data_utils_csmcir.py`, `src/utils_csmcir.py`, `src/lavis/models/blip2_models/blip2_qformer_cir_align_prompt_csmcir.py`, `src/lavis/models/__init__.py` | phần lớn 1140 file là vendored LAVIS (thư viện, không phải first-party) |
| `HINT` (`bec50b6c`) | 1821 | `datasets.py`, `test.py`, `train.py`, `README.md` | vendored LAVIS; nhánh CIRR |
| `ENCODER` (`29a2a31d`) | 62 | `evaluate_model.py`, `datasets.py`, `test.py`, `model_try2.py`, `token_wise_matching.py`, `train.py` | `cirr_test_submission.py` chỉ CIRR |
| `PAIR` (`bfb0b98c`) | 60 | `models/model.py`, `test.py`, `train.py`, `data/datasets.py`, `README.md` | — |
| `AirKnow` (`6ab6fa95`) | 1842 | `test_BLIP2.py`, `train_BLIP2.py`, `datasets.py`, `lavis/models/__init__.py` | vendored LAVIS; nhánh CIRR/LaSCo |
| `ConeSep` (`bd4bf629`) | 1808 | `test.py`, `datasets.py`, `train.py`, `blip2.py` | vendored LAVIS |
| `HABIT` (`6ddc2cd6`) | 1821 | `test.py`, `train.py`, `datasets.py`, `blip2.py`, `default.yaml` | vendored LAVIS |
| `INTENT` (`2a75fc6f`) | 1819 | `test.py`, `datasets.py`, `data_utils.py`, `train_INTENT.py`, `lavis/models/blip2_models/blip2_cir.py` | vendored LAVIS; `test.py:178-357` (CIRR/LaSCo) |
| `TGCIR` (`65fa78ea`) | 56 | toàn bộ 11 file first-party | — |
| `CLVCNet` (`bd9b6889`) | 56 | `test.py`, `datasets.py`, `model.py`, `text_model.py`, `resnet.py` | — |
| `DCNet` (`68c79d38`) | 107 | `data_loader/CE_dataset.py`, `trainer/trainer_joint.py`, `test.py`, `parse_config.py`, `preprocess/process_cap.py`, `base/base_trainer.py` | PDF camera-ready đi kèm được kiểm digest |
| `CLIP4Cir` (`dfed9f74`) | 60 | `src/validate.py`, `src/data_utils.py`, `src/utils.py`, `src/models/combiner.py` | nhánh CIRR |
| `SPRC` (`2935a539`) | 715 | `src/validate_blip.py`, `src/blip_validate.py`, `src/data_utils.py`, `src/validate_blip_rerank.py` | vendored LAVIS |
| `ptha_mtst` | **0** | **0** | repo tác giả chưa được sync/pin (`source_available: false`, `upstream_commit_sha: null`). Đây là `EXTERNAL_BLOCKER` được báo cáo, **không** phải một lượt review. |

Mỗi repo đã ghim đều được kiểm **SHA của checkout khớp `upstream_commit_sha`** trong
`workbench/registry/models.yaml` trước khi đọc.

## 2. First-party (mã của workbench)

| Nhóm | Số file `.py` (tracked) |
|---|---|
| `workbench/**` (trừ `third_party`, `tests`) | 52 |
| `src/**` | 11 |
| `workbench/tests/**` | 74 (trước lượt này) |
| Frontend `.ts`/`.tsx` (mức 1) | 11 |

### Đã đọc và truy vết trong lượt này (first-party)

Đường thực thi và khoa học:
`pc.sh`, `workbench/docker/{run.sh,docker-compose.yml,Dockerfile,release_verify.sh,acceptance.sh,mutation_checks.sh}`,
`workbench/scripts/{evaluate_models.py,verify_run.py,validate_results.py,pc_readiness.py,pipeline.py,build_paper_registry.py}`,
`workbench/backend/{main.py,runtime.py,environment_planner.py,operator_config.py,registry.py,checkpoint_selection.py,index.py,storage.py,analysis.py,paper_gate.py,metrics_extraction.py,benchmark_references.py,preflight.py,errors.py,fashioniq_layout.py,adapters/models.py,adapters/base.py,schemas/results.py}`,
`workbench/replay/{common.py,limn.py,tgcir.py}`,
`workbench/frontend/{types.ts,evidence.ts,gate.ts,client.ts,App.tsx,components.tsx,src.tsx,protocols.ts,gate_types.ts}`,
`src/evaluation/{fashioniq.py,fashioniq_encoder.py}`, `src/datasets/fashioniq.py`.

Được đối chiếu chéo ở hai audit độc lập (`FIRST_PARTY_SCIENCE_AUDIT.md`,
`FIRST_PARTY_RUNTIME_AUDIT.md`) chứ không chỉ đọc một lượt.

### Loại trừ có lý do (first-party)

| Đường | Lý do |
|---|---|
| `workbench/third_party/**` | source thượng nguồn đã ghim, kiểm riêng theo bảng trên |
| `workbench/envs/**`, `src/teachers/**`, `src/training/**`, `src/models/**`, `src/losses/**` | đường **huấn luyện**, không nằm trên đồ thị thực thi đánh giá; lượt này không huấn luyện gì |
| `workbench/artifacts/**` | artifact sinh ra, không phải source |

## 3. Điều manifest này KHÔNG được hiểu thành

* Không có nghĩa "100% source đã đọc". Bảng trên liệt kê chính xác file nào được truy vết tới điểm
  sinh metric; các file vendored LAVIS được coi là **phụ thuộc** với version ghim, không đọc từng dòng.
* Không có nghĩa "mọi model đã audit được". `ptha_mtst` là 0 file — đó là kết luận, không phải thiếu sót.
* Không có nghĩa "mọi đường đi đã chạy". Coverage ở đây là **đọc và truy vết**, còn `RUNTIME_VERIFIED`
  chỉ áp dụng cho những gì thực sự chạy trên máy này, và mỗi báo cáo model ghi rõ là loại nào.
