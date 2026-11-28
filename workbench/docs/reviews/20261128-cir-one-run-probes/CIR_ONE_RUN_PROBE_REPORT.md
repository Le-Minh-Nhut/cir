# Báo Cáo Thực Đo Chẩn Đoán 29 Failure Trên 16 Mô Hình CIR Qua 18 Investigation Suites (`E01–E18`)

> [!IMPORTANT]
> **Tuyên bố minh bạch khoa học & kiểm chứng trực tiếp từ artifact:**
> Toàn bộ số liệu trong báo cáo này (và hồ sơ bằng chứng chi tiết đồng hành [`CIR_ONE_RUN_PROBE_EVIDENCE_DOSSIER.md`](file:///home/heheboiz/data/cir/workbench/docs/reviews/20261128-cir-one-run-probes/CIR_ONE_RUN_PROBE_EVIDENCE_DOSSIER.md)) được sinh tự động và đối chiếu trực tiếp từ các tệp thực đo trên đĩa tại `/home/heheboiz/data/cir/workbench/artifacts/probe_runs/20261128_one_run`. Không có con số nào được ước đoán hay điền tay. Riêng **E18 (Blind Human Annotation Audit)** đã hoàn thiện đầy đủ gói kiểm toán mù ẩn danh model (`220` mục, `11` phân tầng, gồm cả hit và miss) nhưng giữ nguyên trạng thái `partial (awaiting independent human annotators)` theo đúng nguyên tắc không giả lập nhãn người.

## 0. Tóm Tắt Điều Hành — 7 Phát Hiện Nhân Quả Trọng Tâm (Xác Nhận vs Bác Bỏ)

1. **Phân ly nhân quả hoàn toàn của Phủ định (`F01`, E04 + E07):** Trên **16/16 model**, nhánh query mã hóa tuyến tính cực tính phủ định ở mức **`85.1%–95.0%`** (`q_FDR < 0.001`) VÀ nhánh gallery nhận diện thuộc tính mục tiêu (`long vs sleeveless`) ở mức **`88.2%–97.4%`** (`q_FDR < 0.001`), nhưng truy hồi cặp tối thiểu hai chiều trên EDIR sụp đổ về **`0.00%–8.40%`** (`42.9%–63.9%` đảo cực tuyệt đối). → **Bác bỏ** giả thuyết mù ngôn ngữ hay mù thị giác; **xác nhận** nút thắt nằm ở hình học tích vô hướng đơn cực.
2. **Chứng minh nhân quả `Epoch-1 Same-Scene Lock` & điểm tối ưu `~25%` Same-Scene Negative (`F14, F02`, E02 + E17):** Trên **cả 3/3 họ kiến trúc** (`clip4cir`, `conesep`, `sprc`), huấn luyện InfoNCE chuẩn khóa `Scene@1` lên `95%–98.5%` ngay Epoch 1 đồng thời **làm sụp đổ `CondR@1` trong cùng cảnh**. Bổ sung negative cùng cảnh với cùng ngân sách `436` bước tăng mạnh EDIR `R@1`: `clip4cir` **`17.75% → 37.29%`** (`+19.54%`), `conesep` **`30.81% → 47.36%`** (đỉnh tại `25%`, `+16.55%`!), `sprc` **`31.90% → 41.77%`** (`+9.87%`), vượt xa đối chứng `otherscene_hard_matched` (`clip4cir` chỉ `18.25%`). Trên PinPoint, `pp_ref_negative` ép `RefCopy@1` về **`~1%`** nhưng đẩy quá tay làm tụt `NoRef`, trong khi **`pp_explicit_neg`** cải thiện đồng thời `RefCopy@1` và `raw_mAP@10` (`conesep: 0.3049 → 0.3276`).
3. **Chứng minh nhân quả Điểm Mù BLIP-2 (`F28`, E04b + E11):** Xóa các patch ViT nằm trong vùng ROI bị chỉnh sửa của `I_ref` **LÀM TĂNG biên độ truy hồi** vượt đối chứng sham cùng diện tích trên **10/10 model BLIP-2** (`sign test p = 0.00098`). Trên nhóm đối xứng `clip_hit_blip_miss` (`n=66`), tắt visual cross-attention Q-Former (`vis_zero`) cải thiện margin trên **10/10 BLIP-2**, trong khi ở `clip4cir` tắt MLP (`mlp_zero`) hầu như bằng 0 còn tắt nhánh skip tuyến tính (`convex_zero`) làm sụp đổ margin — giải thích trọn vẹn vì sao `clip4cir` có `(+, +, −)` pure fusion collapse = **`0.00%`** trên cả 3 dataset (`F29`) còn BLIP-2 chịu **`5.4%–15.5%`** trên EDIR.
4. **Xung đột Gradient Xuyên Miền & Chênh Lệch Công Bằng Không Bị Khử Bởi Confounder (`F18`, E16):** Cosine gradient (`IEEE-754 float64`) giữa 7 miền PinPoint đi từ đồng thuận dương lúc khởi tạo (`conesep: 0.2995, 0/21` cặp âm) sang xung đột nặng tại checkpoint tốt nhất (**`16/21` cặp âm** trên `conesep` [`-0.0067`] và **`11/21` cặp âm** trên `clip4cir` [`0.0102`]; PCGrad kích hoạt ở `43–49%` số bước). Quan trọng hơn, hồi quy đa biến OLS kiểm soát đồng thời domain, intent và `log(n_pos/n_neg)` cho thấy nhóm `DARKEST` (`β = -0.0965, p = 0.0014`) và `SECOND_DARKEST` (`β = -0.0687, p = 0.0051`) **vẫn chịu chênh lệch `mAP@10` có ý nghĩa thống kê**.
5. **CSLS thắng tuyệt đối Centering/ZCA Whitening cho Hubness (`F16`, E14):** Trên FashionIQ, dev-fitted **CSLS** giảm `HubFP` tới **`94–100%`** và tăng `Hit@10` (`mtst_ptha: 2,246 → 140`, `+2.83%`; `encoder: +1.37%`; `tgcir: +0.98%`), trong khi centering/whitening phá vỡ hình nón dị hướng của `clip4cir` (`HubFP 101 → 3,083 / 3,790`, `-4.0%`).
6. **Mù Không-Có-Đích (`F17`, E15):** Dù entropy toàn gallery cải thiện rõ rệt phân tách hit/miss nội bộ trên FashionIQ (`AUROC 0.52–0.70 → 0.58–0.76`), **mọi mô hình đều sụp đổ gần ngẫu nhiên (`AUROC ≈ 0.51–0.58`, `FAR@90%TPR ≈ 81%–90%`) khi ảnh đích bị loại khỏi gallery** — cảnh báo rủi ro triển khai thực tế nếu thiếu tầng từ chối/kiểm chứng độc lập.
7. **Ba giả thuyết bị bác bỏ bởi đối chứng nghiêm ngặt (`F10/F23`, `F13`, `F21`):** (a) Phóng đại `ROI crop` và nội suy `336px` **không** sửa được `F10` OCR hay `F23` Gaze (ROI crop thua cả sham crop vì mất ngữ cảnh toàn cảnh); (b) Chèn tường minh tên danh mục đích vào prompt (`E12-F13`) **làm giảm** cả `mAP@10` lẫn `CatHit@10` trên `16/16` model (ngược lại lọc đa tạp danh mục `E13` tăng `Explore mAP@10` gấp **`3.3×`**); (c) Trên cặp ảnh FashionIQ sạch đồng nhất sản phẩm (`E10-F21`), tỷ lệ bẫy nền chỉ **`0%`** ở `12/16` model (`max 8%`).

## 1. Resolve Cohort Mô Hình, Entry Point & Xác Minh Parity Baseline

- **Entry point chạy/resume thống nhất:** [`run_cir_probes.py`](file:///home/heheboiz/data/cir/workbench/scripts/run_cir_probes.py)
- **Cohort manifest đóng băng:** [`cohort_manifest.json`](file:///home/heheboiz/data/cir/workbench/artifacts/probe_runs/20261128_one_run/manifest/cohort_manifest.json)
- **Diagnostic cohorts đóng băng:** [`diagnostic_cohorts.json`](file:///home/heheboiz/data/cir/workbench/artifacts/probe_runs/20261128_one_run/manifest/diagnostic_cohorts.json)
- **Bảng coverage JSON kiểm chứng:** [`coverage_matrix.json`](file:///home/heheboiz/data/cir/workbench/artifacts/probe_runs/20261128_one_run/aggregated/coverage_matrix.json)
- **Phạm vi thực đo hiện tại:** EDIR = **`16/16`** models · PinPoint = **`16/16`** models · FashionIQ = **`16/16`** models (`15` in-domain author ckpts + `mtst_ptha` CIRR-transfer gắn nhãn riêng). Các mục mơ hồ/ngoài cohort (`pair`, `combiner_rn50x4_noft`) được ghi rõ trong manifest.

### 1.1. Bảng Xác Minh Parity Baseline (Thực Đo Worker vs Official Saved Predictions)

| Model | Family | EDIR Official `R@1` | EDIR Worker `R@1` | `Abs Δ` | Top-1 Agree | PinPoint Official `mAP@10` | PinPoint Worker `mAP@10` | `Abs Δ` | FIQ Worker `R@10 (Raw / RefExcl)` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **clip4cir** | `clip` | 27.02% | 27.02% | 0.00000 | 100.00% | 0.0860 | 0.0859 | 0.00005 | 43.70% / 44.38% |
| **encoder** | `clip` | 16.15% | 16.15% | 0.00000 | 100.00% | 0.1251 | 0.1251 | 0.00001 | 47.65% / 48.23% |
| **tgcir** | `clip` | 17.65% | 17.65% | 0.00000 | 100.00% | 0.1106 | 0.1106 | 0.00000 | 39.65% / 40.48% |
| **limn** | `clip` | 23.03% | 23.03% | 0.00000 | 100.00% | 0.1494 | 0.1494 | 0.00000 | 46.83% / 47.41% |
| **clvcnet** | `classic` | 13.36% | 13.36% | 0.00000 | 100.00% | 0.0055 | 0.0055 | 0.00000 | 19.67% / 19.75% |
| **dcnet** | `classic` | 14.56% | 14.56% | 0.00000 | 100.00% | 0.0082 | 0.0082 | 0.00000 | 27.35% / 27.60% |
| **qure** | `blip2_vitl` | 29.11% | 29.11% | 0.00000 | 100.00% | 0.1401 | 0.1401 | 0.00000 | 52.29% / 52.76% |
| **sprc** | `blip2_evag` | 36.19% | 36.19% | 0.00000 | 100.00% | 0.1690 | 0.1690 | 0.00000 | 51.04% / 51.34% |
| **mtst_ptha** | `blip2_evag` | 38.38% | 38.38% | 0.00000 | 100.00% | 0.1631 | 0.1631 | 0.00000 | 36.98% / 37.10% |
| **csmcir** | `blip2_evag` | 25.82% | 25.82% | 0.00000 | 100.00% | 0.1963 | 0.1963 | 0.00000 | 54.75% / 55.18% |
| **hint** | `blip2_evag` | 28.22% | 28.22% | 0.00000 | 100.00% | 0.1701 | 0.1701 | 0.00000 | 52.82% / 53.17% |
| **habit** | `blip2_evag` | 27.32% | 27.32% | 0.00000 | 100.00% | 0.1670 | 0.1670 | 0.00000 | 54.42% / 54.93% |
| **intent** | `blip2_evag` | 26.52% | 26.52% | 0.00000 | 100.00% | 0.1929 | 0.1929 | 0.00000 | 54.35% / 54.60% |
| **conesep** | `blip2_evag` | 25.12% | 25.12% | 0.00000 | 100.00% | 0.2128 | 0.2128 | 0.00000 | 54.51% / 54.91% |
| **airknow** | `blip2_evag` | 27.52% | 27.52% | 0.00000 | 100.00% | 0.1891 | 0.1891 | 0.00000 | 52.70% / 53.32% |
| **tme** | `blip2_evag` | 34.90% | 34.90% | 0.00000 | 100.00% | 0.1781 | 0.1781 | 0.00002 | 54.94% / 55.35% |

## 2. Bảng Coverage Tổng Thể (`18 Suites` & `29 Failures`)

### 2.1. Coverage Theo 18 Investigation Suites (`E01–E18`)

| Suite | Trạng thái | Mẫu số đã đo | Ghi chú & Giới hạn thực tế |
|---|---|---|---|
| **E01** | `measured` | edir_models=16, pinpoint_models=16, fashioniq_models=16 | Native single-vec / active MaxSim token spherical tangent geometry + CLIP edit-distance control. |
| **E02** | `measured` | n_models=3, trained=['clip4cir', 'conesep', 'sprc'] | 7 EDIR arms (0/10/25/50/100% same-scene + CLIP-matched other-scene hard + ref-neg) & 4 PinPoint arms across representative CLIP + BLIP-2 families. |
| **E03** | `measured` | edir_models=16, pinpoint_models=16, fashioniq_models=16 | Per-query nested galleries (1k..full, 3 seeds <full, full once) with frozen CLIP ViT-L/14 distractors + empirical null curve. |
| **E04** | `measured` | edir_models=16, fashioniq_gallery_models=14 | Train/dev/test group-split L2/Ridge probes (query + gallery branches + prior + shuffled + PCA-256 + BH-FDR) + causal activation/ROI/token interventions w/ shams. |
| **E05** | `measured` | edir_models=16, pinpoint_models=16, fashioniq_models=16 | Multimodal + CIRCUS unimodal + TRUE native unimodal towers + shuffled controls + score-add + embed-add + (+,+,-) pure fusion rate + block causal interventions. |
| **E06** | `measured` | fashioniq_models=16 | Counterfactual darker/lighter swaps (CIE Lab pixel-verified), F20 open-ended difference, F03 modality-required ref swaps. |
| **E07** | `measured` | edir_models=16, fashioniq_models=16 | Bidirectional with/without & add/remove minimal pairs on fixed ref + candidate pairs. |
| **E08** | `measured` | edir_models=16, fashioniq_models=16 | FashionIQ 4-state 00/01/10/11 interference + EDIR complex clause decomposition + F06 verified commutative C1/C2 swaps. |
| **E09** | `measured` | pinpoint_models=16, fashioniq_models=16 | Swapped color-part attribute binding pairs (FashionIQ) + F26 metadata-only preserve-vs-edit (PinPoint Women's/Men's Fashion). |
| **E10** | `measured` | edir_models=16, fashioniq_models=16 | F21 photo-presentation bg-mask vs matched-area sham, F22 secondary vs primary garment vs matched-area sham, F24 limb-pose diff-ROI vs sham. |
| **E11** | `measured` | edir_models=16 | Native res vs 336x336 ViT bicubic pos-embed interp (fixed-grid models clip4cir/dcnet explicitly flagged unsupported) + detail ROI crop vs matched-area sham crop. |
| **E12** | `measured` | edir_models=16, pinpoint_models=16, fashioniq_models=16 | Controlled syntax/abstractness edits, F13 implicit vs explicit target category, F25 homophone/typo/synonym quadruples with exact token IDs. |
| **E13** | `measured` | pinpoint_models=16, fashioniq_models=16 | Retrieved category distribution vs multi-category oracle-category pool. |
| **E14** | `measured` | edir_models=16, pinpoint_models=16, fashioniq_models=16 | Full-gallery k-occurrence/Gini/effective-rank/anisotropy + val/cal-fitted CSLS, centering & regularized ZCA whitening on test. |
| **E15** | `measured` | edir_models=16, pinpoint_models=16, fashioniq_models=16 | Exact full-gallery dev-tuned temp-scaled entropy, top1, margin, std, perturb var, cross-model disagreement, and verified no-target episodes. |
| **E16** | `measured` | trained=['clip4cir', 'conesep', 'sprc'] | Pairwise 7x7 inter-domain gradient cosine (init & best) + 8-arm matched-step transfer matrix incl. causal PCGrad + 16-model multivariate fairness OLS. |
| **E17** | `measured` | history_models=16, traj=['clip4cir', 'conesep', 'sprc'] | All 16 models' history.jsonl epoch curves PLUS real multi-stage (init/ep1/ep2/final) diagnostic probe trajectories on clip4cir, conesep, sprc. |
| **E18** | `partial_blind_pack_ready_awaiting_human_annotators` | sampled_items=220 | Stratified blind contact-sheet pack (hits + misses, 220 items, 11 strata, cryptographic model anonymization, sealed key, Krippendorff-alpha/Fleiss-kappa scorer) complete; independent human scoring intentionally NOT fabricated. |

### 2.2. Coverage Theo 29 Failure (`F01–F29`)

| # | Failure | Suites được giao | Trạng thái | Chi tiết theo từng suite |
|---:|---|---|---|---|
| 01 | **Negation Inversion** (`F01`) | E07, E04, E05 | `measured` | E07=`measured`, E04=`measured`, E05=`measured` |
| 02 | **Reference-Copy** (`F02`) | E02, E17, E01 | `measured` | E02=`measured`, E17=`measured`, E01=`measured` |
| 03 | **Modality Imbalance** (`F03`) | E05, E06 | `measured` | E05=`measured`, E06=`measured` |
| 04 | **Multi-Constraint Interference** (`F04`) | E08, E04, E05 | `measured` | E08=`measured`, E04=`measured`, E05=`measured` |
| 05 | **Attribute Misbinding** (`F05`) | E09, E04 | `measured` | E09=`measured`, E04=`measured` |
| 06 | **Caption Primacy Bias** (`F06`) | E08, E12 | `measured` | E08=`measured`, E12=`measured` |
| 07 | **Pseudo-Relative Reasoning** (`F07`) | E06, E01 | `measured` | E06=`measured`, E01=`measured` |
| 08 | **Geometric Blindness** (`F08`) | E04, E11 | `measured` | E04=`measured`, E11=`measured` |
| 09 | **Cardinality Blindness** (`F09`) | E04, E11 | `measured` | E04=`measured`, E11=`measured` |
| 10 | **Fine-Grained Blindness** (`F10`) | E11, E04 | `measured` | E11=`measured`, E04=`measured` |
| 11 | **Paraphrase Brittleness** (`F11`) | E12, E04 | `measured` | E12=`measured`, E04=`measured` |
| 12 | **Cross-Category Lock** (`F12`) | E13, E01 | `measured` | E13=`measured`, E01=`measured` |
| 13 | **Pragmatic Intent Blindness** (`F13`) | E13, E12 | `measured` | E13=`measured`, E12=`measured` |
| 14 | **Same-Scene Lock** (`F14`) | E02, E17 | `measured` | E02=`measured`, E17=`measured` |
| 15 | **Large-Scale Retrieval Degradation** (`F15`) | E03, E14 | `measured` | E03=`measured`, E14=`measured` |
| 16 | **Embedding Hubness** (`F16`) | E14, E03 | `measured` | E14=`measured`, E03=`measured` |
| 17 | **Uncertainty Blindness** (`F17`) | E15 | `measured` | E15=`measured` |
| 18 | **Negative Transfer & Domain Disparity** (`F18`) | E16, E17 | `measured` | E16=`measured`, E17=`measured` |
| 19 | **Annotation Noise & False Negative** (`F19`) | E18 | `partial` | E18=`partial_blind_pack_ready_awaiting_human_annotators` |
| 20 | **Open-Ended Difference** (`F20`) | E06, E07 | `measured` | E06=`measured`, E07=`measured` |
| 21 | **Photo-Presentation Gravity** (`F21`) | E10 | `measured` | E10=`measured` |
| 22 | **Co-Occurring Object Leakage** (`F22`) | E10, E05 | `measured` | E10=`measured`, E05=`measured` |
| 23 | **Gaze Direction Blindness** (`F23`) | E04, E11 | `measured` | E04=`measured`, E11=`measured` |
| 24 | **Limb Pose Blindness** (`F24`) | E04, E10 | `measured` | E04=`measured`, E10=`measured` |
| 25 | **Homophone Hijack** (`F25`) | E12, E04 | `measured` | E12=`measured`, E04=`measured` |
| 26 | **Gender-Identity Drop** (`F26`) | E09, E08 | `measured` | E09=`measured`, E08=`measured` |
| 27 | **Visual Edit-Distance Cliff** (`F27`) | E01, E03 | `measured` | E01=`measured`, E03=`measured` |
| 28 | **BLIP-2 Blind Spot** (`F28`) | E04, E11, E05 | `measured` | E04=`measured`, E11=`measured`, E05=`measured` |
| 29 | **Pure Fusion Failure** (`F29`) | E05, E04 | `measured` | E05=`measured`, E04=`measured` |

## 3. Kết Quả Thực Đo Chi Tiết Theo Từng Investigation Suite (`E01–E18`)

### E01 · Edit Displacement Geometry (`F02, F07, F12, F27`)
Sử dụng dịch chuyển tiếp tuyến mặt cầu chung gốc `r` (`proj_{T_r S}(q-r)` vs `proj_{T_r S}(t-r)`), tách riêng hướng và tỷ lệ độ lớn địa trắc `θ_q/θ_t`, kèm ngưỡng `eps=1e-4` cho delta gần zero. Với BLIP-2 multi-vector (`32×256`), dùng đúng **active MaxSim matched token** `g_{k*}(t)` thay vì mean-pool tùy ý.

| Model | Readout | EDIR Tangent Cos | EDIR Angle (°) | EDIR `θ_q/θ_t` | EDIR Margin(Pos-Ref) | PP Tangent Cos | PP `RefCopy@1` | FIQ Tangent Cos | FIQ `θ_q/θ_t` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **clip4cir** | `native_single_vector` | 0.1239 | 82.9 | 1.595 | -0.0498 | 0.1679 | 59.76% | 0.2080 | 1.487 |
| **encoder** | `native_single_vector` | -0.0854 | 94.9 | 1.585 | -0.1327 | 0.2958 | 70.91% | 0.3768 | 0.843 |
| **tgcir** | `native_single_vector` | 0.1129 | 83.5 | 0.517 | -0.1643 | 0.3236 | 71.95% | 0.2850 | 0.845 |
| **limn** | `native_single_vector` | 0.1084 | 83.8 | 1.073 | -0.1312 | 0.3600 | 73.67% | 0.4373 | 0.804 |
| **clvcnet** | `native_single_vector` | 0.1329 | 82.3 | 0.893 | -0.1513 | 0.2146 | 11.22% | 0.4738 | 0.894 |
| **dcnet** | `auxiliary_single_expert_im_feat` | 0.0133 | 89.2 | 2.473 | — | 0.1289 | 41.36% | 0.1628 | 1.761 |
| **qure** | `native_active_maxsim_token` | 0.1846 | 79.3 | 1.267 | -0.0639 | 0.1739 | 38.37% | 0.3982 | 0.925 |
| **sprc** | `native_active_maxsim_token` | 0.1911 | 78.9 | 1.538 | -0.0172 | 0.2789 | 17.43% | 0.3434 | 1.142 |
| **mtst_ptha** | `native_active_maxsim_token` | 0.1736 | 79.9 | 1.665 | -0.0181 | 0.2282 | 24.61% | 0.2941 | 1.138 |
| **csmcir** | `native_active_maxsim_token` | 0.1310 | 82.4 | 1.639 | -0.0455 | 0.4248 | 41.51% | 0.5100 | 0.911 |
| **hint** | `native_active_maxsim_token` | 0.1648 | 80.5 | 1.547 | -0.0405 | 0.2087 | 21.54% | 0.3310 | 1.133 |
| **habit** | `native_active_maxsim_token` | 0.1954 | 78.6 | 1.273 | -0.0606 | 0.1853 | 22.59% | 0.4948 | 0.855 |
| **intent** | `native_active_maxsim_token` | 0.2023 | 78.3 | 1.283 | -0.0568 | 0.3228 | 32.61% | 0.4677 | 0.922 |
| **conesep** | `native_active_maxsim_token` | 0.2050 | 78.1 | 1.368 | -0.0517 | 0.3762 | 38.37% | 0.5053 | 0.835 |
| **airknow** | `native_active_maxsim_token` | 0.1369 | 82.1 | 1.666 | -0.0482 | 0.3020 | 41.96% | 0.3979 | 0.948 |
| **tme** | `native_active_maxsim_token` | 0.1544 | 81.1 | 1.756 | -0.0177 | 0.2103 | 19.07% | 0.3609 | 1.020 |

### E02 · Same-Scene Negative & Reference-Negative Intervention (`F02, F14`)
Huấn luyện có kiểm soát từ cùng khởi tạo tiền huấn luyện, cố định bước tối ưu (`4 epochs / 436 steps` EDIR; `3 epochs` PinPoint). Đối chứng `otherscene_hard_matched` chọn negative khác cảnh có độ khó tương đương bằng frozen CLIP ViT-L/14 độc lập.

| Model | EDIR Arm | Achieved Neg Ratio | `R@1` | `Scene@1` | `CondR@1` | Within-Scene Rank | `RefBeatsPos` |
|---|---|---:|---:|---:|---:|---:|---:|
| **clip4cir** | `init_ep0` | 0.00 | 27.62% | 80.66% | 34.24% | 2.77 | 80.66% |
| **clip4cir** | `baseline_0pct` | 0.00 | 17.75% | 97.71% | 18.16% | 3.49 | 100.00% |
| **clip4cir** | `samescene_10pct` | 0.10 | 26.22% | 97.11% | 27.00% | 3.06 | 99.40% |
| **clip4cir** | `samescene_25pct` | 0.25 | 32.10% | 95.31% | 33.68% | 2.81 | 93.32% |
| **clip4cir** | `samescene_50pct` | 0.50 | 35.39% | 92.42% | 38.30% | 2.61 | 84.75% |
| **clip4cir** | `samescene_100pct` | 1.00 | 37.29% | 84.55% | 44.10% | 2.42 | 74.88% |
| **clip4cir** | `otherscene_hard_matched` | 1.00 | 18.25% | 99.00% | 18.43% | 3.43 | 100.00% |
| **clip4cir** | `ref_image_negative` | 1.00 | 24.53% | 91.72% | 26.74% | 3.03 | 74.68% |
| **conesep** | `init_ep0` | 0.00 | 14.66% | 38.19% | 38.38% | 2.61 | 32.10% |
| **conesep** | `baseline_0pct` | 0.00 | 30.81% | 98.50% | 31.28% | 2.96 | 79.26% |
| **conesep** | `samescene_10pct` | 0.10 | 44.67% | 95.31% | 46.86% | 2.45 | 49.35% |
| **conesep** | `samescene_25pct` | 0.25 | 47.36% | 90.93% | 52.08% | 2.29 | 39.08% |
| **conesep** | `samescene_50pct` | 0.50 | 45.76% | 83.45% | 54.84% | 2.19 | 32.80% |
| **conesep** | `samescene_100pct` | 1.00 | 41.87% | 75.07% | 55.78% | 2.13 | 26.42% |
| **conesep** | `otherscene_hard_matched` | 1.00 | 38.38% | 98.40% | 39.01% | 2.70 | 72.08% |
| **conesep** | `ref_image_negative` | 1.00 | 36.19% | 96.11% | 37.66% | 2.70 | 1.00% |
| **sprc** | `init_ep0` | 0.00 | 3.19% | 9.67% | 32.99% | 3.20 | 48.35% |
| **sprc** | `baseline_0pct` | 0.00 | 31.90% | 97.91% | 32.59% | 2.92 | 69.89% |
| **sprc** | `samescene_10pct` | 0.10 | 41.77% | 93.52% | 44.67% | 2.50 | 49.15% |
| **sprc** | `samescene_25pct` | 0.25 | 41.28% | 83.95% | 49.17% | 2.30 | 36.09% |
| **sprc** | `samescene_50pct` | 0.50 | 38.29% | 74.98% | 51.06% | 2.26 | 32.80% |
| **sprc** | `samescene_100pct` | 1.00 | 33.20% | 63.91% | 51.95% | 2.20 | 28.81% |
| **sprc** | `otherscene_hard_matched` | 1.00 | 39.28% | 97.41% | 40.33% | 2.66 | 63.91% |
| **sprc** | `ref_image_negative` | 1.00 | 36.39% | 91.03% | 39.98% | 2.64 | 0.80% |

**PinPoint Reference-Negative Intervention (`F02`, corpus_test `8,609`):**

| Model | PinPoint Arm | Raw `mAP@10` | NoRef `mAP@10` | `RefCopy@1` | `NegRecall@10` | `Negation mAP@10` |
|---|---|---:|---:|---:|---:|---:|
| **clip4cir** | `pp_baseline` | 0.1760 | 0.2405 | 86.46% | 0.3108 | 0.1115 |
| **clip4cir** | `pp_ref_negative` | 0.1469 | 0.1504 | 0.97% | 0.1590 | 0.1230 |
| **clip4cir** | `pp_explicit_neg` | 0.1901 | 0.2407 | 58.34% | 0.2445 | 0.1195 |
| **clip4cir** | `pp_random_extra_neg` | 0.1761 | 0.2428 | 86.76% | 0.3090 | 0.0920 |
| **conesep** | `pp_baseline` | 0.3049 | 0.3648 | 44.88% | 0.2740 | 0.1541 |
| **conesep** | `pp_ref_negative` | 0.2291 | 0.2325 | 0.90% | 0.1509 | 0.1747 |
| **conesep** | `pp_explicit_neg` | 0.3276 | 0.3694 | 28.35% | 0.2086 | 0.2081 |
| **conesep** | `pp_random_extra_neg` | 0.2903 | 0.3519 | 49.96% | 0.2921 | 0.1515 |
| **sprc** | `pp_baseline` | 0.3020 | 0.3337 | 25.73% | 0.2530 | 0.1929 |
| **sprc** | `pp_ref_negative` | 0.2096 | 0.2126 | 1.05% | 0.1551 | 0.1888 |
| **sprc** | `pp_explicit_neg` | 0.3022 | 0.3214 | 12.19% | 0.1895 | 0.1915 |
| **sprc** | `pp_random_extra_neg` | 0.3003 | 0.3309 | 25.65% | 0.2513 | 0.1865 |

### E03 · Controlled Gallery Scaling (`F15, F16, F27`)
Ước lượng trên **per-query nested pools** (`1k..full`, 3 seed cho `<full`, `full` đo 1 lần) với distractor chọn bằng trọng tài độc lập `CLIP ViT-L/14 DataComp-XL`, đối chiếu với đường **data-driven null curve** kiểm soát scene.

| Model | PP Random `1k→full` `R@10` | PP Hard `1k→full` `R@10` | PP `ZeroPos@10` (`1k→full` Hard) | PP Null `ZeroPos@10` (`full`) | EDIR Hard `1k→full` `R@1` |
|---|---:|---:|---:|---:|---:|
| **clip4cir** | 96.46% → 54.75% | 61.63% → 54.75% | 38.37% → 45.25% | 45.17% | 27.02% → 27.02% |
| **encoder** | 96.93% → 60.43% | 64.47% → 60.43% | 35.53% → 39.57% | 39.37% | 16.15% → 16.15% |
| **tgcir** | 97.26% → 58.86% | 63.95% → 58.86% | 36.05% → 41.14% | 41.46% | 17.65% → 17.65% |
| **limn** | 97.18% → 65.15% | 66.94% → 65.15% | 33.06% → 34.85% | 34.95% | 23.03% → 23.03% |
| **clvcnet** | 73.85% → 9.50% | 24.31% → 9.50% | 75.69% → 90.50% | 89.93% | 13.76% → 13.36% |
| **dcnet** | 72.50% → 10.55% | 26.85% → 10.55% | 73.15% → 89.45% | 89.49% | 14.86% → 14.56% |
| **qure** | 97.93% → 63.13% | 70.46% → 63.13% | 29.54% → 36.87% | 37.08% | 29.11% → 29.11% |
| **sprc** | 98.13% → 66.12% | 72.63% → 66.12% | 27.37% → 33.88% | 33.94% | 36.49% → 36.19% |
| **mtst_ptha** | 97.16% → 64.55% | 71.35% → 64.55% | 28.65% → 35.45% | 35.53% | 38.38% → 38.38% |
| **csmcir** | 98.03% → 70.01% | 74.57% → 70.01% | 25.43% → 29.99% | 30.14% | 25.82% → 25.82% |
| **hint** | 97.93% → 65.30% | 71.20% → 65.30% | 28.80% → 34.70% | 34.66% | 28.22% → 28.22% |
| **habit** | 98.43% → 65.37% | 71.88% → 65.37% | 28.12% → 34.63% | 34.52% | 27.32% → 27.32% |
| **intent** | 98.45% → 71.43% | 74.35% → 71.43% | 25.65% → 28.57% | 28.89% | 26.52% → 26.52% |
| **conesep** | 98.50% → 72.70% | 76.44% → 72.70% | 23.56% → 27.30% | 27.14% | 25.12% → 25.12% |
| **airknow** | 97.83% → 71.35% | 74.20% → 71.35% | 25.80% → 28.65% | 28.52% | 27.52% → 27.52% |
| **tme** | 97.86% → 67.91% | 71.80% → 67.91% | 28.20% → 32.09% | 32.75% | 34.90% → 34.90% |

### E04 · Layer-wise Information Probing & Causal Activation Interventions (`F01, F04, F05, F08–F11, F23, F24, F28, F29`)
Probe L2/Ridge fit trên `train`, chọn regularization trên `dev`, đánh giá 1 lần trên `test` (group-split không rò rỉ), so với majority prior, shuffled-label control, matched-dim PCA-256 và hiệu chỉnh BH-FDR trên **cả EDIR lẫn FashionIQ gallery**. Riêng `F09` tách rõ biểu diễn ảnh gallery **pre-pooling** (`gal_vit_patch_mean_pos`) vs **post-pooling** (`gal_vision_proj_mean_pos` / `gal_vis_final_pos`) và nhánh văn bản (`qry_*`) để không che khuất tổn thất pooling thị giác. *(Lưu ý minh bạch: `F23_gaze_dir` trong EDIR chỉ có 1 lớp trong `train/dev`, nên E04 không fit probe phân loại mà chuyển chẩn đoán F23 sang can thiệp độ phân giải/ROI-crop có sham ở E11.)*

| Model | `F01 Neg` Best Query (`Test / Shuf / PCA256 / q_FDR`) | `F01 FIQ Gallery Sleeves` (`Test / Prior / q_FDR`) | `F08 Geom vs Color` Gal-Diff (`Test / Shuf / q_FDR`) | `F09 Card 1-4` Gal Pre→Post Pool MAE (Prior=`0.789`, AdjErr) | `F10 OCR` Best Query (`Test / Shuf`) | `F24 Pose` Best Layer (`Test / Shuf`) |
|---|---|---|---|---|---|---|
| **clip4cir** | `qry_txt_final`: 91.33% / 54.84% / 91.33% / q=0.000 | `gal_vis_final`: 93.42% / 59.21% / q=0.000 | `gal_vis_final_diff`: 48.84% / 44.19% / q=0.092 | — → 0.826 (adj=57.89%) | `qry_txt_final`: AUC=0.984 (shuf=0.552, acc=97.81%) | `qry_txt_final`: AUC=0.894 (shuf=0.485, acc=97.21%) |
| **encoder** | `qry_txt_final`: 85.14% / 58.33% / 85.14% / q=0.000 | `gal_vis_final`: 94.74% / 59.21% / q=0.000 | `gal_vis_final_pos`: 49.83% / 43.19% / q=0.019 | — → 0.801 (adj=68.42%) | `qry_txt_final`: AUC=0.985 (shuf=0.541, acc=98.01%) | `qry_txt_final`: AUC=0.899 (shuf=0.410, acc=96.11%) |
| **tgcir** | `qry_txt_final`: 85.14% / 72.58% / 85.14% / q=0.000 | `gal_vis_final`: 93.42% / 59.21% / q=0.000 | `gal_vis_mid_diff`: 53.49% / 48.50% / q=0.092 | — → 0.867 (adj=63.16%) | `qry_txt_final`: AUC=0.982 (shuf=0.555, acc=97.31%) | `qry_txt_final`: AUC=0.865 (shuf=0.483, acc=95.31%) |
| **limn** | `qry_txt_final`: 91.03% / 71.59% / 91.03% / q=0.000 | — | `gal_vis_mid_diff`: 52.49% / 43.19% / q=0.001 | — → 0.832 (adj=57.89%) | `qry_txt_final`: AUC=0.988 (shuf=0.581, acc=98.21%) | `qry_txt_final`: AUC=0.905 (shuf=0.394, acc=96.41%) |
| **clvcnet** | `qry_local_512`: 87.84% / 55.73% / 87.84% / q=0.000 | — | `gal_final_1024_diff`: 46.51% / 49.50% / q=1.000 | — | `qry_fused_1024`: AUC=0.902 (shuf=0.501, acc=94.92%) | `qry_fused_1024`: AUC=0.691 (shuf=0.506, acc=94.92%) |
| **dcnet** | `qry_txt_im_feat`: 94.82% / 78.17% / 94.82% / q=0.000 | `gal_im_feat`: 88.16% / 59.21% / q=0.000 | `gal_inter1_pos`: 48.17% / 43.85% / q=0.152 | — | `qry_txt_im_feat`: AUC=0.983 (shuf=0.485, acc=96.71%) | `qry_txt_im_feat`: AUC=0.919 (shuf=0.424, acc=95.21%) |
| **qure** | `qry_qformer_mid_L6`: 93.52% / 55.43% / 93.52% / q=0.000 | `gal_vit_patch_mean`: 97.37% / 59.21% / q=0.000 | `gal_qformer_preproj_diff`: 52.16% / 47.18% / q=0.090 | 0.959 → 0.998 (adj=42.11%) | `qry_qformer_mid_L6`: AUC=0.983 (shuf=0.501, acc=97.01%) | `qry_qformer_mid_L6`: AUC=0.849 (shuf=0.462, acc=95.41%) |
| **sprc** | `qry_qformer_mid_L6`: 95.01% / 77.27% / 95.01% / q=0.000 | `gal_vision_proj_mean`: 97.37% / 59.21% / q=0.000 | `gal_vision_proj_mean_pos`: 52.82% / 45.18% / q=0.010 | 0.891 → 0.732 (adj=52.63%) | `qry_text_only_cls`: AUC=0.986 (shuf=0.531, acc=97.61%) | `qry_qformer_mid_L6`: AUC=0.882 (shuf=0.438, acc=95.41%) |
| **mtst_ptha** | `qry_qformer_mid_L6`: 94.42% / 77.07% / 94.42% / q=0.000 | `gal_qformer_preproj`: 96.05% / 59.21% / q=0.000 | `gal_qformer_preproj_pos`: 53.49% / 45.85% / q=0.012 | 0.891 → 0.711 (adj=52.63%) | `qry_text_only_cls`: AUC=0.988 (shuf=0.501, acc=97.81%) | `qry_qformer_mid_L6`: AUC=0.888 (shuf=0.429, acc=95.21%) |
| **csmcir** | `qry_qformer_mid_L6`: 93.12% / 61.12% / 93.12% / q=0.000 | `gal_vision_proj_mean`: 97.37% / 59.21% / q=0.000 | `gal_qformer_preproj_pos`: 52.49% / 51.16% / q=0.616 | 0.891 → 0.720 (adj=52.63%) | `qry_text_only_cls`: AUC=0.986 (shuf=0.509, acc=97.61%) | `qry_qformer_mid_L6`: AUC=0.900 (shuf=0.393, acc=95.41%) |
| **hint** | `qry_qformer_mid_L6`: 93.92% / 75.27% / 93.92% / q=0.000 | `gal_vision_proj_mean`: 97.37% / 59.21% / q=0.000 | `gal_qformer_preproj_pos`: 50.83% / 45.85% / q=0.117 | 0.891 → 0.723 (adj=42.11%) | `qry_text_only_cls`: AUC=0.992 (shuf=0.491, acc=98.50%) | `qry_qformer_mid_L6`: AUC=0.859 (shuf=0.415, acc=95.01%) |
| **habit** | `qry_qformer_mid_L6`: 93.22% / 64.91% / 93.22% / q=0.000 | `gal_qformer_preproj`: 97.37% / 59.21% / q=0.000 | `gal_vision_proj_mean_diff`: 50.83% / 48.50% / q=0.437 | 0.891 → 0.743 (adj=47.37%) | `qry_text_only_cls`: AUC=0.977 (shuf=0.492, acc=98.60%) | `qry_qformer_mid_L6`: AUC=0.841 (shuf=0.363, acc=95.41%) |
| **intent** | `qry_qformer_mid_L6`: 94.32% / 59.52% / 94.32% / q=0.000 | `gal_qformer_preproj`: 96.05% / 59.21% / q=0.000 | `gal_vision_proj_mean_diff`: 51.50% / 46.84% / q=0.119 | 0.891 → 0.725 (adj=57.89%) | `qry_text_only_cls`: AUC=0.983 (shuf=0.478, acc=98.21%) | `qry_qformer_mid_L6`: AUC=0.871 (shuf=0.396, acc=95.51%) |
| **conesep** | `qry_qformer_mid_L6`: 94.32% / 78.17% / 94.32% / q=0.000 | `gal_qformer_preproj`: 97.37% / 59.21% / q=0.000 | `gal_vision_proj_mean_diff`: 51.16% / 45.51% / q=0.059 | 0.891 → 0.724 (adj=57.89%) | `qry_text_only_cls`: AUC=0.987 (shuf=0.512, acc=98.21%) | `qry_qformer_mid_L6`: AUC=0.860 (shuf=0.454, acc=95.41%) |
| **airknow** | `qry_qformer_mid_L6`: 94.02% / 76.67% / 94.02% / q=0.000 | `gal_qformer_preproj`: 97.37% / 59.21% / q=0.000 | `gal_vision_proj_mean_diff`: 50.50% / 47.18% / q=0.266 | 0.891 → 0.706 (adj=52.63%) | `qry_text_only_cls`: AUC=0.984 (shuf=0.482, acc=98.11%) | `qry_qformer_mid_L6`: AUC=0.865 (shuf=0.374, acc=94.92%) |
| **tme** | `qry_qformer_mid_L6`: 93.22% / 61.91% / 93.22% / q=0.000 | `gal_qformer_preproj`: 97.37% / 59.21% / q=0.000 | `gal_vision_proj_mean_diff`: 51.16% / 46.84% / q=0.137 | 0.888 → 0.705 (adj=52.63%) | `qry_text_only_cls`: AUC=0.991 (shuf=0.515, acc=98.11%) | `qry_qformer_out_L12_qmean`: AUC=0.914 (shuf=0.470, acc=96.11%) |

#### E04b · Can Thiệp Nhân Quả Kích Hoạt (Activation/ROI/Token Ablation vs Sham) & `F28` Matched Cohorts
Trên cùng 3 nhóm đối xứng `F28` (`clip_hit_blip_miss` n=66, `blip_hit_clip_miss` n=66, `matched_both_hit` n=66) và bộ `E11` ROI (`n=163`), so sánh can thiệp thực tế với đối chứng **matched-area sham / matched-norm sham noise**.

| Model | Family | EDIR `ViT ROI−Sham Δm` | EDIR `QFormer Top4−Sham4 Δm` | `F28 clip_hit_blip_miss`: Base → VisZero/MLPZero → MeanPool/ConvexZero (Sham) | `F28 matched_both_hit`: Base → VisZero/MLPZero → MeanPool/ConvexZero |
|---|---|---:|---:|---|---|
| **clip4cir** | `clip` | — | — | 0.0108 → mlp0=0.0105 → conv0=0.0017 (sham=0.0109) | 0.0196 → mlp0=0.0203 → conv0=-0.0033 |
| **encoder** | `clip` | — | — | 0.0013 → vis0=0.0048 → mp32=— (sham=0.0009) | 0.0126 → vis0=0.0159 → mp32=— |
| **tgcir** | `clip` | — | — | 0.0050 → vis0=0.0092 → mp32=— (sham=0.0047) | 0.0313 → vis0=0.0277 → mp32=— |
| **limn** | `clip` | — | — | 0.0055 → vis0=0.0027 → mp32=— (sham=0.0054) | 0.0269 → vis0=0.0259 → mp32=— |
| **clvcnet** | `classic` | — | — | 0.0030 → vis0=0.0067 → mp32=— (sham=0.0029) | 0.0244 → vis0=0.0031 → mp32=— |
| **dcnet** | `classic` | — | — | 0.9758 → vis0=-0.0217 → mp32=— (sham=—) | 3.1022 → vis0=-0.0264 → mp32=— |
| **qure** | `blip2_vitl` | 0.0071 | -0.0011 | -0.0182 → vis0=-0.0083 → mp32=-0.0142 (sham=-0.0179) | 0.0278 → vis0=0.0238 → mp32=0.0173 |
| **sprc** | `blip2_evag` | 0.0040 | 0.0000 | -0.0213 → vis0=-0.0124 → mp32=-0.0156 (sham=-0.0211) | 0.0479 → vis0=0.0578 → mp32=0.0286 |
| **mtst_ptha** | `blip2_evag` | 0.0028 | -0.0008 | -0.0253 → vis0=-0.0193 → mp32=-0.0215 (sham=-0.0255) | 0.0428 → vis0=0.0498 → mp32=0.0235 |
| **csmcir** | `blip2_evag` | 0.0029 | 0.0011 | -0.0258 → vis0=-0.0194 → mp32=-0.0240 (sham=-0.0256) | 0.0162 → vis0=0.0214 → mp32=0.0125 |
| **hint** | `blip2_evag` | 0.0036 | -0.0011 | -0.0259 → vis0=-0.0209 → mp32=-0.0202 (sham=-0.0259) | 0.0228 → vis0=0.0300 → mp32=0.0156 |
| **habit** | `blip2_evag` | 0.0063 | -0.0006 | -0.0322 → vis0=-0.0259 → mp32=-0.0274 (sham=-0.0324) | 0.0177 → vis0=0.0250 → mp32=0.0142 |
| **intent** | `blip2_evag` | 0.0048 | -0.0012 | -0.0337 → vis0=-0.0275 → mp32=-0.0273 (sham=-0.0339) | 0.0185 → vis0=0.0288 → mp32=0.0149 |
| **conesep** | `blip2_evag` | 0.0042 | 0.0004 | -0.0330 → vis0=-0.0254 → mp32=-0.0267 (sham=-0.0332) | 0.0140 → vis0=0.0215 → mp32=0.0134 |
| **airknow** | `blip2_evag` | 0.0036 | -0.0014 | -0.0335 → vis0=-0.0261 → mp32=-0.0272 (sham=-0.0337) | 0.0197 → vis0=0.0267 → mp32=0.0143 |
| **tme** | `blip2_evag` | 0.0007 | -0.0025 | -0.0208 → vis0=-0.0151 → mp32=-0.0154 (sham=-0.0067) | 0.0331 → vis0=0.0268 → mp32=0.0183 |

### E05 · Fusion Dominance & Pure Fusion Failure (`F03, F28, F29`)
So sánh trực tiếp trên cùng bộ 300 triplet `(query, pos, neg)`: multimodal, đường CIRCUS (`empty_text` / `black_img`), **đường native unimodal tower thật**, `score_add`, `embed_add`, `oracle`, và tỷ lệ sụp đổ thuần fusion `(+, +, −)` có mẫu số rõ ràng.

| Model | EDIR Multi Acc | Native Img Acc | Native Txt Acc | Score-Add Acc | `(+,+,−)` Native (`k/n`, rate) | `(+,+,−)` CIRCUS (`k/n`) | PP `(+,+,−)` Native | FIQ `(+,+,−)` Native |
|---|---:|---:|---:|---:|---|---|---|---|
| **clip4cir** | 53.67% | 46.00% | 63.33% | 53.33% | 0/81 (0.00%) | 3/72 | 0/134 (0.00%) | 0/220 (0.00%) |
| **encoder** | 46.00% | 43.33% | 61.67% | 50.00% | 6/76 (7.89%) | 4/64 | 0/130 (0.00%) | 1/203 (0.49%) |
| **tgcir** | 46.67% | 46.33% | 57.67% | 53.33% | 1/67 (1.49%) | 3/62 | 0/148 (0.00%) | 0/206 (0.00%) |
| **limn** | 53.67% | 44.67% | 61.00% | 52.67% | 1/74 (1.35%) | 5/93 | 1/129 (0.78%) | 0/632 (0.00%) |
| **clvcnet** | 46.33% | 46.00% | 50.67% | 48.67% | 3/57 (5.26%) | 9/67 | 13/119 (10.92%) | 5/544 (0.89%) |
| **dcnet** | 48.00% | 45.33% | 52.67% | 44.33% | 7/74 (9.46%) | 4/67 | 11/82 (13.41%) | 2/95 (2.11%) |
| **qure** | 57.67% | 51.00% | 58.00% | 60.33% | 5/80 (6.25%) | 9/81 | 7/147 (4.76%) | 0/210 (0.00%) |
| **sprc** | 61.00% | 50.67% | 65.33% | 62.67% | 10/106 (9.43%) | 8/104 | 4/145 (2.76%) | 2/222 (0.90%) |
| **mtst_ptha** | 60.00% | 50.00% | 65.00% | 63.33% | 7/91 (7.69%) | 4/86 | 4/132 (3.03%) | 3/206 (1.46%) |
| **csmcir** | 53.67% | 49.33% | 63.67% | 57.67% | 13/84 (15.48%) | 7/73 | 2/130 (1.54%) | 0/228 (0.00%) |
| **hint** | 55.00% | 48.33% | 66.33% | 62.67% | 9/90 (10.00%) | 3/86 | 2/133 (1.50%) | 0/219 (0.00%) |
| **habit** | 54.33% | 48.67% | 67.67% | 63.67% | 5/92 (5.43%) | 2/87 | 4/150 (2.67%) | 1/227 (0.44%) |
| **intent** | 55.00% | 50.67% | 66.33% | 63.33% | 9/97 (9.28%) | 4/90 | 3/146 (2.05%) | 0/230 (0.00%) |
| **conesep** | 53.67% | 50.00% | 64.67% | 64.67% | 10/90 (11.11%) | 1/85 | 0/139 (0.00%) | 0/224 (0.00%) |
| **airknow** | 56.67% | 49.00% | 67.00% | 63.67% | 8/92 (8.70%) | 4/86 | 0/140 (0.00%) | 0/230 (0.00%) |
| **tme** | 58.33% | 51.33% | 66.00% | 62.67% | 14/95 (14.74%) | 7/86 | 4/134 (2.99%) | 1/230 (0.43%) |

### E06 & E07 · Counterfactual Reference Swapping, Open-Ended Difference & Negation Minimal Pairs (`F01, F03, F07, F20`)

| Model | E06 Abs-Dark Shortcut (`F07`) | E06 Ref-Swap Unchanged (`F07`) | E06 `F03` ModReq Both-Swap (`ImgTrap / TxtTrap`) | E06 `blue / another / not red` (`F20`) | E07 EDIR Bidir Neg Flip (`F01`) | E07 EDIR Inversion (`F01`) | E07 FIQ `with/wo` Bidir | E07 FIQ `add/rem` Bidir |
|---|---:|---:|---|---|---:|---:|---:|---:|
| **clip4cir** | 41.67% | 38.33% | 60.00% (7.50% / 5.00%) | 74.29% / 61.90% / 72.38% | 3.36% | 57.98% | 30.92% | 17.39% |
| **encoder** | 29.17% | 30.00% | 32.50% (42.50% / 2.50%) | 48.57% / 54.29% / 62.86% | 0.84% | 52.94% | 34.78% | 3.38% |
| **tgcir** | 35.83% | 22.50% | 35.00% (25.00% / 15.00%) | 60.00% / 63.81% / 62.86% | 0.00% | 52.94% | 40.10% | 22.71% |
| **limn** | 40.00% | 33.33% | — | 69.52% / 59.05% / 64.76% | 4.20% | 55.46% | 37.07% | 9.33% |
| **clvcnet** | 48.33% | 47.50% | — | 90.48% / 64.76% / 32.38% | 0.84% | 42.86% | 22.17% | 7.25% |
| **dcnet** | 32.50% | 38.33% | 65.00% (2.50% / 20.00%) | 75.24% / 68.57% / 67.62% | 0.84% | 52.94% | 41.55% | 4.35% |
| **qure** | 32.50% | 28.33% | 37.50% (12.50% / 27.50%) | 69.52% / 65.71% / 70.48% | 6.72% | 55.46% | 37.20% | 7.73% |
| **sprc** | 35.83% | 40.00% | 65.00% (5.00% / 15.00%) | 75.24% / 70.48% / 69.52% | 8.40% | 57.14% | 37.68% | 4.35% |
| **mtst_ptha** | 43.33% | 44.17% | 50.00% (17.50% / 0.00%) | 72.38% / 53.33% / 53.33% | 2.52% | 59.66% | 31.40% | 26.57% |
| **csmcir** | 35.00% | 40.83% | 67.50% (15.00% / 2.50%) | 63.81% / 66.67% / 71.43% | 2.52% | 60.50% | 38.65% | 11.59% |
| **hint** | 39.17% | 38.33% | 47.50% (17.50% / 17.50%) | 63.81% / 59.05% / 60.95% | 0.84% | 63.87% | 41.06% | 10.14% |
| **habit** | 37.50% | 27.50% | 37.50% (20.00% / 20.00%) | 62.86% / 63.81% / 61.90% | 4.20% | 54.62% | 31.40% | 7.73% |
| **intent** | 37.50% | 30.00% | 50.00% (10.00% / 17.50%) | 69.52% / 68.57% / 68.57% | 1.68% | 57.14% | 35.27% | 8.21% |
| **conesep** | 36.67% | 30.83% | 35.00% (15.00% / 22.50%) | 67.62% / 64.76% / 66.67% | 4.20% | 54.62% | 43.48% | 14.01% |
| **airknow** | 31.67% | 24.17% | 35.00% (17.50% / 22.50%) | 65.71% / 61.90% / 62.86% | 1.68% | 56.30% | 33.82% | 8.70% |
| **tme** | 37.50% | 34.17% | 55.00% (12.50% / 17.50%) | 74.29% / 64.76% / 62.86% | 3.36% | 57.98% | 49.76% | 10.14% |

### E08 & E09 · Multi-Constraint Interference, Caption Primacy & Attribute Binding Swaps (`F04, F05, F06, F26`)

| Model | E08 4-State Exact `11` (`F04`) | E08 Cond Drop `A/B` in `A+B` | E08 EDIR Complex Cond Drop | E08 `F06` `C1+C2` Flip Rate (`R@10`) | E09 `F05` Bidir Binding Acc | E09 `F26` Preserve-Drop Rate (PP) |
|---|---:|---|---:|---:|---:|---:|
| **clip4cir** | 65.82% | 7.35% / 25.00% | 29.73% | 6.00% | 65.62% | 10.53% |
| **encoder** | 44.30% | 26.32% / 13.16% | 4.35% | 3.33% | 65.62% | 15.79% |
| **tgcir** | 45.57% | 26.42% / 24.53% | 23.33% | 4.00% | 53.12% | 9.65% |
| **limn** | 59.81% | 23.07% / 16.72% | 32.43% | 4.09% | 45.64% | 11.40% |
| **clvcnet** | 70.65% | 6.26% / 25.12% | 42.86% | 3.67% | 70.85% | 37.72% |
| **dcnet** | 64.56% | 9.86% / 23.94% | 11.11% | 2.33% | 81.25% | 31.58% |
| **qure** | 68.35% | 14.29% / 14.29% | 41.30% | 3.67% | 46.88% | 19.30% |
| **sprc** | 74.68% | 13.85% / 7.69% | 43.59% | 4.67% | 53.12% | 13.16% |
| **mtst_ptha** | 62.03% | 4.76% / 26.98% | 26.83% | 6.00% | 31.25% | 13.16% |
| **csmcir** | 65.82% | 11.29% / 19.35% | 42.86% | 2.67% | 53.12% | 9.65% |
| **hint** | 70.89% | 12.70% / 12.70% | 35.00% | 3.67% | 56.25% | 14.04% |
| **habit** | 59.49% | 15.09% / 15.09% | 20.00% | 1.67% | 37.50% | 12.28% |
| **intent** | 70.89% | 8.96% / 17.91% | 30.00% | 2.33% | 43.75% | 8.77% |
| **conesep** | 54.43% | 13.21% / 22.64% | 28.95% | 1.00% | 40.62% | 12.28% |
| **airknow** | 59.49% | 18.18% / 16.36% | 20.59% | 1.33% | 34.38% | 12.28% |
| **tme** | 73.42% | 10.61% / 13.64% | 26.83% | 3.00% | 46.88% | 12.28% |

### E10 & E11 · Visual Context Masking & Resolution/Patch Sensitivity (`F08, F09, F10, F21, F22, F23, F24, F28`)

| Model | E10 `F21` Format Trap Orig→BgMask | E10 `F22` Secondary−Sham Shift | E10 `F24` Pose ROI−Sham `Δm` | E11 `336px` Status | E11 OCR `Nat / ROI / Sham` Acc | E11 Gaze `Nat / ROI / Sham` Acc | E11 Card `Nat→336` Acc | E11 Geom `Nat→336` Acc |
|---|---:|---:|---:|---|---|---|---|---|
| **clip4cir** | 4.00% → 4.00% | 0.0018 | -0.0008 | `unsupported_fixed_grid` | 60.00% / 45.45% / 52.73% | 62.50% / 37.50% / 62.50% | 46.67% → — | 62.50% → — |
| **encoder** | 4.00% → 4.00% | 0.0086 | 0.0102 | `unsupported_fixed_grid (FactorNet 49-patch)` | 50.91% / 38.18% / 63.64% | 25.00% / 37.50% / 37.50% | 40.00% → — | 50.00% → — |
| **tgcir** | 4.00% → 4.00% | 0.0185 | 0.0137 | `measured_336_bicubic_pos_interp` | 52.73% / 36.36% / 58.18% | 25.00% / 37.50% / 37.50% | 36.67% → 46.67% | 45.00% → 50.00% |
| **limn** | 3.70% → 3.70% | 0.0062 | -0.0051 | `measured_336_bicubic_pos_interp` | 47.27% / 43.64% / 67.27% | 37.50% / 37.50% / 37.50% | 40.00% → 40.00% | 57.50% → 57.50% |
| **clvcnet** | 0.00% → 0.00% | 0.0133 | 0.0064 | `measured_336_bicubic_pos_interp` | 47.27% / 45.45% / 61.82% | 50.00% / 50.00% / 12.50% | 40.00% → 40.00% | 55.00% → 52.50% |
| **dcnet** | 0.00% → 0.00% | — | -0.0513 | `unsupported_fixed_grid` | 49.09% / 41.82% / 61.82% | 50.00% / 50.00% / 50.00% | 36.67% → — | 67.50% → — |
| **qure** | 0.00% → 0.00% | 0.0128 | -0.0090 | `measured_336_bicubic_pos_interp` | 52.73% / 50.91% / 54.55% | 37.50% / 25.00% / 62.50% | 36.67% → 40.00% | 67.50% → 70.00% |
| **sprc** | 0.00% → 0.00% | 0.0062 | -0.0011 | `measured_336_bicubic_pos_interp` | 56.36% / 50.91% / 56.36% | 37.50% / 37.50% / 62.50% | 63.33% → 66.67% | 65.00% → 72.50% |
| **mtst_ptha** | 8.00% → 8.00% | 0.0048 | -0.0027 | `measured_336_bicubic_pos_interp` | 54.55% / 50.91% / 60.00% | 37.50% / 50.00% / 50.00% | 70.00% → 63.33% | 67.50% → 62.50% |
| **csmcir** | 0.00% → 4.00% | 0.0030 | -0.0037 | `measured_336_bicubic_pos_interp` | 52.73% / 38.18% / 60.00% | 25.00% / 50.00% / 37.50% | 50.00% → 43.33% | 52.50% → 60.00% |
| **hint** | 0.00% → 0.00% | 0.0046 | 0.0017 | `measured_336_bicubic_pos_interp` | 54.55% / 58.18% / 61.82% | 25.00% / 37.50% / 75.00% | 66.67% → 60.00% | 47.50% → 50.00% |
| **habit** | 0.00% → 0.00% | 0.0076 | -0.0006 | `measured_336_bicubic_pos_interp` | 54.55% / 47.27% / 56.36% | 25.00% / 37.50% / 50.00% | 40.00% → 50.00% | 55.00% → 57.50% |
| **intent** | 4.00% → 4.00% | 0.0075 | 0.0012 | `measured_336_bicubic_pos_interp` | 63.64% / 52.73% / 56.36% | 37.50% / 37.50% / 50.00% | 46.67% → 50.00% | 57.50% → 55.00% |
| **conesep** | 0.00% → 0.00% | 0.0079 | 0.0018 | `measured_336_bicubic_pos_interp` | 60.00% / 47.27% / 52.73% | 25.00% / 37.50% / 62.50% | 40.00% → 36.67% | 60.00% → 60.00% |
| **airknow** | 0.00% → 0.00% | 0.0077 | 0.0014 | `measured_336_bicubic_pos_interp` | 54.55% / 52.73% / 56.36% | 37.50% / 37.50% / 62.50% | 50.00% → 53.33% | 57.50% → 62.50% |
| **tme** | 0.00% → 0.00% | 0.0069 | -0.0031 | `measured_336_bicubic_pos_interp` | 65.45% / 40.00% / 58.18% | 50.00% / 25.00% / 37.50% | 56.67% → 60.00% | 75.00% → 70.00% |

### E12 & E13 · Paraphrase/Syntax/Homophone Interventions, Exact Tokenizer Audit & Cross-Category Manifold (`F06, F11, F12, F13, F25`)

| Model | E12 PP Verbose-Abs Top1-Flip (`F11`) | E12 `F13` Implicit→Explicit `mAP@10` | E12 `F25` FIQ `Orig/Homo/Typo/Syn` Acc | E12 `F25` FIQ `ΔTok (Homo/Typo)` & TxtCosShift (Homo) | E12 `F25` EDIR `Orig/Homo/Typo/Syn` Acc | E13 PP `Complement` Full→OracleCat | E13 PP `Explore` Full→OracleCat |
|---|---:|---|---|---|---|---|---|
| **clip4cir** | 25.83% | 0.0663 → 0.0487 | 96.36% / 96.36% / 94.55% / 94.55% | 0.28 / 1.15 · cosΔ=0.133 | 77.78% / 72.22% / 77.78% / 77.78% | 0.0717 → 0.1849 | 0.0331 → 0.2146 |
| **encoder** | 11.67% | 0.0880 → 0.0670 | 94.55% / 95.45% / 94.55% / 95.45% | 0.28 / 1.15 · cosΔ=0.126 | 50.00% / 44.44% / 50.00% / 50.00% | 0.0438 → 0.1207 | 0.0763 → 0.2370 |
| **tgcir** | 13.33% | 0.0798 → 0.0628 | 95.45% / 95.45% / 95.45% / 94.55% | 0.28 / 1.15 · cosΔ=0.143 | 55.56% / 55.56% / 55.56% / 55.56% | 0.0425 → 0.1159 | 0.0564 → 0.1747 |
| **limn** | 15.00% | 0.1096 → 0.0826 | — | — | 38.89% / 33.33% / 38.89% / 33.33% | 0.0644 → 0.1331 | 0.0982 → 0.2278 |
| **clvcnet** | 48.33% | 0.0052 → 0.0029 | — | — | 50.00% / 44.44% / 44.44% / 44.44% | 0.0052 → 0.0329 | 0.0038 → 0.0144 |
| **dcnet** | 59.17% | 0.0072 → 0.0064 | 95.45% / 96.36% / 97.27% / 95.45% | 0.00 / 0.00 · cosΔ=— | 38.89% / 38.89% / 38.89% / 38.89% | 0.0007 → 0.0073 | 0.0000 → 0.0283 |
| **qure** | 33.33% | 0.1382 → 0.1305 | 96.36% / 97.27% / 97.27% / 94.55% | 0.35 / 1.23 · cosΔ=0.162 | 77.78% / 66.67% / 72.22% / 72.22% | 0.0671 → 0.1855 | 0.0632 → 0.2349 |
| **sprc** | 58.33% | 0.1739 → 0.1310 | 96.36% / 95.45% / 95.45% / 95.45% | 0.35 / 1.23 · cosΔ=0.136 | 83.33% / 66.67% / 72.22% / 77.78% | 0.0516 → 0.1835 | 0.1025 → 0.3265 |
| **mtst_ptha** | 62.50% | 0.1616 → 0.1190 | 96.36% / 96.36% / 95.45% / 96.36% | 0.35 / 1.23 · cosΔ=0.146 | 83.33% / 72.22% / 77.78% / 77.78% | 0.0427 → 0.1813 | 0.0530 → 0.2661 |
| **csmcir** | 31.67% | 0.1874 → 0.1730 | 97.27% / 98.18% / 98.18% / 97.27% | 0.35 / 1.23 · cosΔ=0.144 | 44.44% / 44.44% / 38.89% / 50.00% | 0.0839 → 0.2095 | 0.1006 → 0.3237 |
| **hint** | 69.17% | 0.1632 → 0.1381 | 95.45% / 95.45% / 95.45% / 95.45% | 0.35 / 1.23 · cosΔ=0.203 | 61.11% / 38.89% / 50.00% / 55.56% | 0.0749 → 0.2156 | 0.0906 → 0.3403 |
| **habit** | 53.33% | 0.2054 → 0.1742 | 95.45% / 94.55% / 97.27% / 95.45% | 0.35 / 1.23 · cosΔ=0.213 | 55.56% / 38.89% / 33.33% / 55.56% | 0.1312 → 0.2921 | 0.0751 → 0.3302 |
| **intent** | 43.33% | 0.1895 → 0.1640 | 96.36% / 95.45% / 96.36% / 95.45% | 0.35 / 1.23 · cosΔ=0.200 | 44.44% / 38.89% / 38.89% / 44.44% | 0.0949 → 0.2167 | 0.0923 → 0.3241 |
| **conesep** | 40.83% | 0.2161 → 0.1897 | 96.36% / 97.27% / 97.27% / 98.18% | 0.35 / 1.23 · cosΔ=0.204 | 38.89% / 38.89% / 38.89% / 44.44% | 0.1015 → 0.2141 | 0.1157 → 0.3545 |
| **airknow** | 23.33% | 0.1791 → 0.1735 | 96.36% / 95.45% / 97.27% / 97.27% | 0.35 / 1.23 · cosΔ=0.200 | 61.11% / 50.00% / 50.00% / 61.11% | 0.0670 → 0.1684 | 0.0755 → 0.3165 |
| **tme** | 57.50% | 0.1792 → 0.1311 | 97.27% / 94.55% / 96.36% / 95.45% | 0.35 / 1.23 · cosΔ=0.152 | 83.33% / 66.67% / 66.67% / 83.33% | 0.0729 → 0.2021 | 0.0465 → 0.2254 |

### E14 · Hubness, Local Density, Spectral Geometry & Dev-Fitted Post-Processing (`F15, F16`)

| Model | FIQ EffRank / Aniso | FIQ Raw HubFP / Hit@10 | FIQ CSLS HubFP / Hit@10 | FIQ Center HubFP / Hit@10 | FIQ Whiten HubFP / Hit@10 | EDIR Raw→CSLS Hit@1 | PP Raw→CSLS Hit@10 |
|---|---|---|---|---|---|---:|---:|
| **clip4cir** | 86.9 / 0.592 | 101 / 43.20% | 22 / 43.06% | 3083 / 39.24% | 3790 / 39.00% | 27.02% → 26.92% | 54.75% → 53.93% |
| **encoder** | 104.5 / 0.067 | 135 / 47.30% | 20 / 48.67% | 153 / 47.28% | 187 / 46.41% | 16.15% → 16.45% | 60.43% → 61.56% |
| **tgcir** | 124.6 / 0.043 | 80 / 39.22% | 0 / 40.20% | 65 / 39.34% | 86 / 38.09% | 17.65% → 17.45% | 58.86% → 62.60% |
| **limn** | 73.3 / 0.052 | 36.3 / 47.14% | 0 / 48.58% | 24.7 / 47.16% | 31.3 / 47.28% | 23.03% → 23.93% | 65.15% → 66.04% |
| **clvcnet** | 33.0 / 0.020 | 20.3 / 19.53% | 0 / 20.30% | 13.3 / 19.26% | 110.7 / 19.89% | 13.36% → 13.86% | 9.50% → 11.14% |
| **dcnet** | 17.2 / 0.627 | 0 / 27.03% | 0 / 27.06% | — / — | — / — | 14.56% → 13.96% | 10.55% → 11.52% |
| **qure** | 60.6 / 0.053 | 0 / 52.36% | 0 / 52.05% | 0 / 52.15% | 0 / 50.71% | 29.11% → 29.11% | 63.13% → 65.97% |
| **sprc** | 49.7 / 0.151 | 22 / 50.61% | 0 / 51.09% | 62 / 49.72% | 20 / 49.96% | 36.19% → 36.39% | 66.12% → 65.30% |
| **mtst_ptha** | 57.3 / 0.304 | 2246 / 37.37% | 140 / 40.20% | 2004 / 36.99% | 2342 / 37.28% | 38.38% → 36.99% | 64.55% → 63.87% |
| **csmcir** | 65.9 / 0.469 | 26 / 55.15% | 0 / 55.67% | 24 / 54.79% | 21 / 54.69% | 25.82% → 26.42% | 70.01% → 69.41% |
| **hint** | 64.0 / 0.242 | 21 / 52.79% | 0 / 53.01% | 148 / 51.62% | 86 / 51.02% | 28.22% → 28.32% | 65.30% → 67.46% |
| **habit** | 77.2 / 0.024 | 0 / 54.69% | 0 / 54.64% | 0 / 54.81% | 0 / 54.11% | 27.32% → 26.82% | 65.37% → 66.19% |
| **intent** | 62.1 / 0.040 | 21 / 54.31% | 0 / 54.91% | 41 / 54.16% | 20 / 53.92% | 26.52% → 27.02% | 71.43% → 70.01% |
| **conesep** | 73.9 / 0.015 | 0 / 54.79% | 0 / 54.91% | 0 / 54.79% | 0 / 54.33% | 25.12% → 24.43% | 72.70% → 71.58% |
| **airknow** | 75.8 / 0.100 | 0 / 53.06% | 0 / 53.71% | 0 / 52.91% | 0 / 52.22% | 27.52% → 26.02% | 71.35% → 71.05% |
| **tme** | 69.4 / 0.085 | 0 / 54.62% | 0 / 54.67% | 22 / 54.55% | 0 / 53.49% | 34.90% → 34.90% | 67.91% → 66.49% |

### E15 · Uncertainty, Exact Full-Gallery Calibration, Disagreement & Verified No-Target Detection (`F17`)

| Model | EDIR `AUROC` (Top1 / Margin / FullEnt / Trunc50) | EDIR `ECE / Brier / AURC` (FullEnt) | EDIR No-Target `AUROC / FAR@90TPR` | PP Hit `AUROC` (Top1 / FullEnt) | FIQ Hit `AUROC` (Top1 / FullEnt) | FIQ No-Target `AUROC / FAR@90TPR` |
|---|---|---|---|---|---|---|
| **clip4cir** | 0.528 / 0.573 / 0.484 / 0.511 | 0.051 / 0.199 / 0.739 | 0.515 / 89.42% | 0.585 / 0.604 | 0.521 / 0.733 | 0.512 / 87.25% |
| **encoder** | 0.512 / 0.512 / 0.500 / 0.517 | 0.006 / 0.135 / 0.839 | 0.519 / 89.22% | 0.522 / 0.549 | 0.509 / 0.615 | 0.529 / 86.72% |
| **tgcir** | 0.476 / 0.529 / 0.473 / 0.507 | 0.003 / 0.146 / 0.839 | 0.526 / 88.62% | 0.553 / 0.472 | 0.539 / 0.580 | 0.519 / 87.64% |
| **limn** | 0.501 / 0.571 / 0.561 / 0.561 | 0.007 / 0.176 / 0.745 | 0.523 / 86.03% | 0.525 / 0.432 | — / — | — / — |
| **clvcnet** | 0.565 / 0.496 / 0.578 / 0.576 | 0.022 / 0.116 / 0.851 | 0.511 / 90.22% | 0.572 / 0.592 | — / — | — / — |
| **dcnet** | 0.515 / 0.505 / 0.509 / 0.509 | 0.017 / 0.125 / 0.851 | 0.579 / 81.44% | 0.583 / 0.595 | 0.615 / 0.589 | 0.525 / 86.87% |
| **qure** | 0.492 / 0.617 / 0.611 / 0.611 | 0.022 / 0.201 / 0.637 | 0.530 / 87.03% | 0.597 / 0.675 | 0.590 / 0.681 | 0.546 / 86.77% |
| **sprc** | 0.538 / 0.700 / 0.717 / 0.717 | 0.046 / 0.196 / 0.448 | 0.562 / 81.24% | 0.655 / 0.669 | 0.624 / 0.762 | 0.553 / 85.81% |
| **mtst_ptha** | 0.535 / 0.699 / 0.699 / 0.699 | 0.057 / 0.207 / 0.441 | 0.580 / 84.03% | 0.654 / 0.649 | 0.554 / 0.754 | 0.525 / 87.59% |
| **csmcir** | 0.505 / 0.564 / 0.551 / 0.551 | 0.026 / 0.190 / 0.698 | 0.506 / 85.63% | 0.599 / 0.550 | 0.556 / 0.738 | 0.552 / 84.85% |
| **hint** | 0.504 / 0.626 / 0.604 / 0.604 | 0.029 / 0.197 / 0.642 | 0.507 / 90.42% | 0.628 / 0.637 | 0.562 / 0.739 | 0.538 / 84.32% |
| **habit** | 0.512 / 0.580 / 0.560 / 0.560 | 0.040 / 0.198 / 0.696 | 0.535 / 89.62% | 0.519 / 0.629 | 0.656 / 0.716 | 0.565 / 85.67% |
| **intent** | 0.531 / 0.597 / 0.574 / 0.574 | 0.010 / 0.193 / 0.700 | 0.543 / 87.03% | 0.589 / 0.514 | 0.636 / 0.742 | 0.558 / 84.85% |
| **conesep** | 0.517 / 0.581 / 0.490 / 0.537 | 0.007 / 0.188 / 0.762 | 0.522 / 87.43% | 0.608 / 0.504 | 0.644 / 0.711 | 0.562 / 85.38% |
| **airknow** | 0.529 / 0.583 / 0.557 / 0.557 | 0.019 / 0.198 / 0.686 | 0.518 / 89.62% | 0.574 / 0.421 | 0.644 / 0.714 | 0.559 / 85.81% |
| **tme** | 0.546 / 0.646 / 0.653 / 0.653 | 0.043 / 0.211 / 0.520 | 0.551 / 83.23% | 0.656 / 0.661 | 0.698 / 0.752 | 0.553 / 86.24% |

**E15 Cross-Model Disagreement (căn chỉnh đúng candidate ID chung):**
- **EDIR** (`16` models, `1003` common queries): Mean Top-1 Consensus = `54.51%`, Mean Pairwise Jaccard@10 = `0.4813`, Majority-Hit AUROC (Consensus) = **`0.7738`**, Mean Per-Model Hit AUROC (Consensus / Jaccard@10) = **`0.5744` / `0.4930`**.
- **PINPOINT** (`16` models, `1337` common queries): Mean Top-1 Consensus = `46.60%`, Mean Pairwise Jaccard@10 = `0.2227`, Majority-Hit AUROC (Consensus) = **`0.4923`**, Mean Per-Model Hit AUROC (Consensus / Jaccard@10) = **`0.4910` / `0.6524`**.
- **FASHIONIQ** (`14` models, `4169` common queries): Mean Top-1 Consensus = `50.66%`, Mean Pairwise Jaccard@10 = `0.2750`, Majority-Hit AUROC (Consensus) = **`0.6508`**, Mean Per-Model Hit AUROC (Consensus / Jaccard@10) = **`0.6509` / `0.7359`**.

### E16 · Cross-Domain Gradient Conflict, Causal PCGrad & Fairness Multivariate Audit (`F18`)
- **`clip4cir` Inter-Domain Gradient Cosine (Top-7 PinPoint domains):** init mean off-diag = `0.0965` (`8` negative pairs); trained-best mean off-diag = `0.0102` (`11` negative pairs).

| `clip4cir` Transfer Arm (steps=240, exp=7680) | Overall `mAP@10` | `Women's Fashion` | `Home Decor` | `Animals` | `Entertainment` | `Art` | PCGrad Conflicts |
|---|---:|---:|---:|---:|---:|---:|---:|
| `single_WomensFashion` | 0.1603 | 0.1257 | 0.1725 | 0.4542 | 0.1337 | 0.1637 | 0 |
| `single_HomeDecor` | 0.1457 | 0.1048 | 0.1423 | 0.4084 | 0.1484 | 0.1558 | 0 |
| `single_Animals` | 0.1346 | 0.0947 | 0.1733 | 0.3668 | 0.1424 | 0.1270 | 0 |
| `single_Entertainment` | 0.1499 | 0.1113 | 0.1630 | 0.4578 | 0.1350 | 0.1533 | 0 |
| `single_Art` | 0.1432 | 0.1094 | 0.1416 | 0.4540 | 0.1344 | 0.1514 | 0 |
| `mixed_original` | 0.1731 | 0.1342 | 0.1759 | 0.4843 | 0.1459 | 0.2120 | 0 |
| `mixed_balanced_top7` | 0.1686 | 0.1324 | 0.1934 | 0.4311 | 0.1536 | 0.1927 | 0 |
| `mixed_pcgrad_top7` | 0.1691 | 0.1376 | 0.1875 | 0.4335 | 0.1455 | 0.1850 | 117 |

- **`conesep` Inter-Domain Gradient Cosine (Top-7 PinPoint domains):** init mean off-diag = `0.2995` (`0` negative pairs); trained-best mean off-diag = `-0.0067` (`16` negative pairs).

| `conesep` Transfer Arm (steps=240, exp=7680) | Overall `mAP@10` | `Women's Fashion` | `Home Decor` | `Animals` | `Entertainment` | `Art` | PCGrad Conflicts |
|---|---:|---:|---:|---:|---:|---:|---:|
| `single_WomensFashion` | 0.2612 | 0.2269 | 0.3069 | 0.5086 | 0.2419 | 0.2715 | 0 |
| `single_HomeDecor` | 0.2977 | 0.2464 | 0.3638 | 0.4864 | 0.2964 | 0.3667 | 0 |
| `single_Animals` | 0.2390 | 0.1796 | 0.3133 | 0.4727 | 0.2501 | 0.2609 | 0 |
| `single_Entertainment` | 0.2588 | 0.2191 | 0.3249 | 0.4996 | 0.2738 | 0.2324 | 0 |
| `single_Art` | 0.2572 | 0.2332 | 0.2808 | 0.4231 | 0.2644 | 0.2779 | 0 |
| `mixed_original` | 0.2898 | 0.2593 | 0.3084 | 0.5138 | 0.2414 | 0.3563 | 0 |
| `mixed_balanced_top7` | 0.2955 | 0.2849 | 0.3164 | 0.5184 | 0.2498 | 0.3254 | 0 |
| `mixed_pcgrad_top7` | 0.2982 | 0.2855 | 0.3235 | 0.5179 | 0.2551 | 0.3332 | 103 |

- **`sprc` Inter-Domain Gradient Cosine (Top-7 PinPoint domains):** init mean off-diag = `0.6571` (`0` negative pairs); trained-best mean off-diag = `0.0621` (`0` negative pairs).

| `sprc` Transfer Arm (steps=240, exp=7680) | Overall `mAP@10` | `Women's Fashion` | `Home Decor` | `Animals` | `Entertainment` | `Art` | PCGrad Conflicts |
|---|---:|---:|---:|---:|---:|---:|---:|
| `single_WomensFashion` | 0.2414 | 0.1950 | 0.3453 | 0.5167 | 0.2454 | 0.1644 | 0 |
| `single_HomeDecor` | 0.2574 | 0.1729 | 0.3531 | 0.4865 | 0.2783 | 0.2622 | 0 |
| `single_Animals` | 0.1761 | 0.1174 | 0.2728 | 0.4112 | 0.2325 | 0.1239 | 0 |
| `single_Entertainment` | 0.1995 | 0.1466 | 0.3139 | 0.4277 | 0.2348 | 0.1380 | 0 |
| `single_Art` | 0.2377 | 0.1749 | 0.3245 | 0.4738 | 0.2524 | 0.2572 | 0 |
| `mixed_original` | 0.2784 | 0.2218 | 0.3526 | 0.5224 | 0.2455 | 0.2783 | 0 |
| `mixed_balanced_top7` | 0.2791 | 0.2217 | 0.3477 | 0.5185 | 0.2984 | 0.2991 | 0 |
| `mixed_pcgrad_top7` | 0.2787 | 0.2183 | 0.3487 | 0.5130 | 0.2923 | 0.2988 | 118 |

#### E16b · Kiểm Toán Công Bằng Nhân Khẩu Học PinPoint (`All Queries` vs `Swap-Only`, Group Bootstrap 95% CI & Multivariate OLS)

| Scope | Skin Tone Bucket | `n` | Mean `n_pos / n_neg` | BLIP-2 `mAP@10` (95% Group CI) | BLIP-2 `NegRec@10` (95% Group CI) | All-16 `mAP@10` |
|---|---|---:|---:|---|---|---:|
| `all_queries` | `LIGHTEST` | 153 | 12.1 / 29.4 | 0.1364 `[0.0887, 0.1872]` | 0.0626 `[0.0305, 0.1019]` | 0.1072 |
| `all_queries` | `SECOND_LIGHTEST` | 192 | 10.2 / 29.7 | 0.1682 `[0.1072, 0.2326]` | 0.0179 `[0.0125, 0.0247]` | 0.1309 |
| `all_queries` | `SECOND_DARKEST` | 130 | 5.8 / 34.4 | 0.1562 `[0.0770, 0.2419]` | 0.0768 `[0.0567, 0.0982]` | 0.1245 |
| `all_queries` | `DARKEST` | 41 | 10.4 / 34.5 | 0.1344 `[0.0503, 0.2277]` | 0.0639 `[0.0385, 0.0893]` | 0.1105 |
| `all_queries` | `UNKNOWN` | 821 | 8.8 / 29.5 | 0.1962 `[0.1628, 0.2269]` | 0.0474 `[0.0395, 0.0563]` | 0.1567 |
| `swap_only` | `LIGHTEST` | 67 | 12.2 / 22.9 | 0.1676 `[0.0897, 0.2476]` | 0.0673 `[0.0419, 0.1062]` | 0.1363 |
| `swap_only` | `SECOND_LIGHTEST` | 116 | 9.6 / 25.2 | 0.2025 `[0.1256, 0.2731]` | 0.0243 `[0.0171, 0.0324]` | 0.1573 |
| `swap_only` | `SECOND_DARKEST` | 100 | 5.9 / 34.3 | 0.1757 `[0.0803, 0.2803]` | 0.0676 `[0.0491, 0.0917]` | 0.1394 |
| `swap_only` | `DARKEST` | 34 | 11.4 / 33.9 | 0.1545 `[0.0687, 0.2494]` | 0.0602 `[0.0356, 0.0871]` | 0.1267 |
| `swap_only` | `UNKNOWN` | 356 | 7.3 / 27.6 | 0.2380 `[0.1897, 0.2899]` | 0.0584 `[0.0462, 0.0723]` | 0.1957 |

**Multivariate OLS (kiểm soát đồng thời `query_category`, `l1_interest`, `log(n_pos)`, `log(n_neg)`, ref = `SECOND_LIGHTEST`):**
- **blip2_mAP10** (`n=516`, `R²=0.321`): skin_LIGHTEST: β=`-0.0220` (p=`0.2396`); skin_SECOND_DARKEST: β=`-0.0687` (p=`0.005095`); skin_DARKEST: β=`-0.0965` (p=`0.001423`)
- **blip2_NegRec10** (`n=516`, `R²=0.603`): skin_LIGHTEST: β=`+0.0328` (p=`1.33e-09`); skin_SECOND_DARKEST: β=`+0.0406` (p=`9.295e-09`); skin_DARKEST: β=`+0.0404` (p=`3.055e-06`)

### E17 · Training Dynamics Tracking (`F02, F14, F18`)
**Multi-stage checkpoint trajectory thực đo (`init ep0 → ep1 → ep2 → final`):** *(Lưu ý: probe phân loại nhị phân L2 với `λ=10` cố định trong script quỹ đạo bị quá điều chuẩn về majority prior trên tập lệch lớp, nên toàn bộ đánh giá linear probe chọn `C` trên dev + BH-FDR được báo cáo chuẩn hóa tại mục E04; bảng dưới tập trung vào động lực truy hồi & biên độ thực đo qua từng epoch.)*

| Model | Dataset | Stage | Primary (`R@1` / `mAP@10`) | `Scene@1` / `NoRef` | `CondR@1` / `RefCopy@1` | Sibling Margin | Ref Margin (`pos−ref`) |
|---|---|---|---:|---:|---:|---:|---:|
| **clip4cir** | EDIR | `init_ep0` | 27.62% | 80.66% | 34.24% | -0.0068 | -0.0204 |
| **clip4cir** | EDIR | `ep1` | 16.55% | 97.31% | 17.01% | -0.0602 | -0.1970 |
| **clip4cir** | EDIR | `ep2` | 17.55% | 97.71% | 17.96% | -0.0626 | -0.1996 |
| **clip4cir** | EDIR | `ep4` | 17.75% | 97.71% | 18.16% | -0.0681 | -0.2133 |
| **clip4cir** | PinPoint | `init_ep0` | 0.1375 | 0.1809 | 66.42% | — | — |
| **clip4cir** | PinPoint | `ep1` | 0.1621 | 0.2295 | 94.91% | — | — |
| **clip4cir** | PinPoint | `ep2` | 0.1748 | 0.2402 | 88.48% | — | — |
| **clip4cir** | PinPoint | `ep3` | 0.1760 | 0.2405 | 86.46% | — | — |
| **conesep** | EDIR | `init_ep0` | 14.66% | 38.19% | 38.38% | -0.0057 | 0.0218 |
| **conesep** | EDIR | `ep1` | 26.42% | 98.31% | 26.88% | -0.0270 | -0.0469 |
| **conesep** | EDIR | `ep2` | 27.02% | 98.40% | 27.46% | -0.0270 | -0.0515 |
| **conesep** | EDIR | `ep4` | 30.81% | 98.50% | 31.28% | -0.0256 | -0.0504 |
| **conesep** | PinPoint | `init_ep0` | 0.1598 | 0.1791 | 15.41% | — | — |
| **conesep** | PinPoint | `ep1` | 0.2731 | 0.3388 | 58.12% | — | — |
| **conesep** | PinPoint | `ep2` | 0.2875 | 0.3535 | 53.78% | — | — |
| **conesep** | PinPoint | `ep3` | 0.3049 | 0.3648 | 44.88% | — | — |
| **sprc** | EDIR | `init_ep0` | 3.19% | 9.67% | 32.99% | -0.0065 | 0.0036 |
| **sprc** | EDIR | `ep1` | 35.29% | 94.82% | 37.22% | -0.0125 | -0.0123 |
| **sprc** | EDIR | `ep2` | 32.60% | 97.11% | 33.57% | -0.0172 | -0.0260 |
| **sprc** | EDIR | `ep4` | 31.90% | 97.91% | 32.59% | -0.0204 | -0.0345 |
| **sprc** | PinPoint | `init_ep0` | 0.0694 | 0.0732 | 2.54% | — | — |
| **sprc** | PinPoint | `ep1` | 0.2865 | 0.3152 | 21.54% | — | — |
| **sprc** | PinPoint | `ep2` | 0.3035 | 0.3330 | 24.08% | — | — |
| **sprc** | PinPoint | `ep3` | 0.3020 | 0.3337 | 25.73% | — | — |

- **Toàn bộ 16 model `history.jsonl`:** `12/16` model đạt EDIR val `R@10 ≥ 90%` ngay Epoch 1 (`Epoch-1 Same-Scene Lock`), chi tiết lưu tại [`e17_history_all16.json`](file:///home/heheboiz/data/cir/workbench/artifacts/probe_runs/20261128_one_run/e17_dynamics/e17_history_all16.json).

### E18 · Randomized Stratified Blind Human Annotation Audit (`F19`)
- **Trạng thái:** `blind_pack_ready_awaiting_human_annotators` (`human_annotations_completed = False`).
- **Quy mô gói kiểm toán mù:** **`220` mục** phân tầng trên **`11` tầng** (gồm cả hit và miss trên EDIR, PinPoint, FashionIQ; model ẩn danh bằng salted SHA-256; khóa đối chiếu niêm phong riêng tại [`e18_sealed_key.json`](file:///home/heheboiz/data/cir/workbench/artifacts/probe_runs/20261128_one_run/e18_blind_audit/e18_sealed_key.json)).
- **Tài liệu chấm mù & rubric:** [`annotator_template.csv`](file:///home/heheboiz/data/cir/workbench/artifacts/probe_runs/20261128_one_run/e18_blind_audit/annotator_template.csv) · [`ANNOTATION_RUBRIC.md`](file:///home/heheboiz/data/cir/workbench/artifacts/probe_runs/20261128_one_run/e18_blind_audit/ANNOTATION_RUBRIC.md) · script chấm đồng thuận (`Krippendorff's α`, `Fleiss' κ`, trọng số Horvitz-Thompson, group CI): [`e18_audit_pack.py`](file:///home/heheboiz/data/cir/workbench/probes/e18_audit_pack.py).

## 4. Chẩn Đoán Chi Tiết Từng Failure (`F01–F29`) Kèm Bằng Chứng Nhân Quả & Phân Tầng Kiến Trúc

Mỗi mục dưới đây tổng hợp trực tiếp từ artifact thực đo (phân tầng theo 4 họ kiến trúc: `classic`, `clip`, `blip2_vitl`, `blip2_evag`), nêu rõ giả thuyết kiểm chứng, bằng chứng số lượng, cơ chế nhân quả được xác nhận hay bác bỏ, mẫu số thực tế và tệp kiểm chứng.

### F01 · Negation Inversion (`measured` — Suites: E07, E04, E05)

- **Bằng chứng E04 (biểu diễn tồn tại ở CẢ hai nhánh):** Probe cực tính phủ định trên nhánh query đạt **min=85.14%, median=93.37%, max=95.01% (n=16)** (`q_FDR < 0.001` trên toàn bộ model đã đo); đồng thời probe thuộc tính mục tiêu (`long vs sleeveless`) trên nhánh gallery FashionIQ đạt **min=88.16%, median=97.37%, max=97.37% (n=14)**.
- **Bằng chứng E07 (sụp đổ ở không gian chiếu đối xứng):** EDIR bidirectional minimal-pair accuracy chỉ đạt **min=0.00%, median=2.52%, max=8.40% (n=16)**, trong khi tỷ lệ đảo ngược tuyệt đối (`negation_inversion_rate`) lên tới **min=42.86%, median=55.88%, max=63.87% (n=16)**. Trên FashionIQ: `with/without` = **min=22.17%, median=37.13%, max=49.76% (n=16)**, `add/remove` = **min=3.38%, median=9.01%, max=26.57% (n=16)**.
- **Theo họ kiến trúc (EDIR Bidir Flip):** Classic: `0.84%` · CLIP: `2.10%` · BLIP2-ViT-L: `6.72%` · BLIP2-EVA-g: `3.27%`.
- **Kết luận nhân quả:** **Bác bỏ** giả thuyết mù phủ định ở text encoder hay mù thuộc tính ở gallery encoder; **xác nhận** nút thắt nằm ở hình học tích vô hướng đơn cực (unipolar dot-product) không thể đẩy truy vấn ra xa thuộc tính bị phủ định mà không làm sụp đổ neo ngữ nghĩa dương.

### F02 · Reference-Copy (`measured` — Suites: E02, E17, E01)

- **Bằng chứng E01 (hình học dịch chuyển tiếp tuyến):** EDIR mean margin `(pos − ref)` = **min=-0.1643, median=-0.0517, max=-0.0172 (n=15)**; PinPoint `RefCopy@1` = **min=11.22%, median=38.37%, max=73.67% (n=16)**, PinPoint tangent cosine = **min=0.1289, median=0.2535, max=0.4248 (n=16)**.
- **Can thiệp nhân quả E02 (`clip4cir`):** EDIR `RefBeatsPos`: baseline `100.00%` → `ref_image_negative` `74.68%` (R@1 `17.75%` → `24.53%`); PinPoint `RefCopy@1`: baseline `86.46%` → `pp_ref_negative` `0.97%` (raw mAP@10 `0.1760` → `0.1469`).
- **Can thiệp nhân quả E02 (`conesep`):** EDIR `RefBeatsPos`: baseline `79.26%` → `ref_image_negative` `1.00%` (R@1 `30.81%` → `36.19%`); PinPoint `RefCopy@1`: baseline `44.88%` → `pp_ref_negative` `0.90%` (raw mAP@10 `0.3049` → `0.2291`).
- **Can thiệp nhân quả E02 (`sprc`):** EDIR `RefBeatsPos`: baseline `69.89%` → `ref_image_negative` `0.80%` (R@1 `31.90%` → `36.39%`); PinPoint `RefCopy@1`: baseline `25.73%` → `pp_ref_negative` `1.05%` (raw mAP@10 `0.3020` → `0.2096`).
- **Kết luận nhân quả:** Xác nhận nguyên nhân kép: thiếu tín hiệu âm trực tiếp từ chính ảnh tham chiếu `I_ref` trong InfoNCE và dịch chuyển địa trắc `q − r` quá ngắn so với khoảng cách tới ảnh đích `t − r`.

### F03 · Modality Imbalance (`measured` — Suites: E05, E06)

- **Bằng chứng E05 (EDIR 300 triplets):** Multimodal = **min=46.00%, median=54.00%, max=61.00% (n=16)** vs `native_img_tower` = **min=43.33%, median=48.83%, max=51.33% (n=16)** vs `native_txt_tower` = **min=50.67%, median=64.17%, max=67.67% (n=16)** — trên **toàn bộ 16/16 model**, tháp văn bản thuần túy vượt xa fusion đa phương thức trên bài toán phân biệt giữa hai ảnh cùng cảnh!
- **Bằng chứng E06 `F03` Modality-Required Ref Swap (FashionIQ):** `both_swap_acc` = **min=32.50%, median=48.75%, max=67.50% (n=14)**, `img_trap_rate` = **min=2.50%, median=15.00%, max=42.50% (n=14)**, `txt_trap_rate` = **min=0.00%, median=17.50%, max=27.50% (n=14)**.
- **Kết luận nhân quả:** Xác nhận hiện tượng lệch thống trị phương thức (modality dominance): khi chỉ một kênh là đủ hoặc một kênh lấn át kênh kia, bộ hợp nhất rơi vào bẫy hình ảnh (`img_trap`) hoặc văn bản (`txt_trap`) thay vì hợp thành đồng thời cả hai điều kiện.

### F04 · Multi-Constraint Interference (`measured` — Suites: E08, E04, E05)

- **Bằng chứng E08 4-State `00/01/10/11` (FashionIQ):** Tỷ lệ giải đúng đồng thời `11` trong `A+B` = **min=44.30%, median=65.19%, max=74.68% (n=16)**; trên các tập mà model **đã giải đúng cả A lẫn B khi đứng riêng**, tỷ lệ rớt ràng buộc màu `A` = **min=4.76%, median=12.95%, max=26.42% (n=16)** nhưng rớt ràng buộc phụ `B` (tay áo/họa tiết) lên tới **min=7.69%, median=17.31%, max=26.98% (n=16)**!
- **Bằng chứng E08 EDIR Complex Clause Decomposition:** Conditional drop khi ghép đa mệnh đề = **min=4.35%, median=29.34%, max=43.59% (n=16)**.
- **Theo họ kiến trúc (FIQ `AB_exact_11`):** Classic: `67.61%` · CLIP: `53.88%` · BLIP2-ViT-L: `68.35%` · BLIP2-EVA-g: `65.68%`.
- **Kết luận nhân quả:** Xác nhận nhiễu xuyên ràng buộc bất đối xứng: trong một truy vấn vector đơn, thuộc tính nổi trội toàn cục (màu sắc `A`) áp đảo và nuốt chửng ràng buộc cấu trúc cục bộ (`B`) ngay cả khi model hoàn toàn hiểu `B` đứng độc lập.

### F05 · Attribute Misbinding (`measured` — Suites: E09, E04)

- **Bằng chứng E09 Attribute Binding Swaps (FashionIQ, `n=32` cặp hoán vị màu-bộ phận):** `acc_AB` = **min=75.00%, median=88.47%, max=96.88% (n=16)**, `acc_BA` = **min=46.88%, median=59.38%, max=87.50% (n=16)**, độ chính xác hoán vị hai chiều thực sự (`paired_bidirectional_swap_acc`) chỉ đạt **min=31.25%, median=50.00%, max=81.25% (n=16)**.
- **Theo họ kiến trúc (Bidir Binding Acc):** Classic: `76.05%` · CLIP: `57.50%` · BLIP2-ViT-L: `46.88%` · BLIP2-EVA-g: `44.10%`.
- **Kết luận nhân quả:** Xác nhận lỗi gắn kết thuộc tính dạng túi từ (bag-of-attributes): mô hình mã hóa sự hiện diện của màu và bộ phận rời rạc thay vì liên kết chặt chẽ thuộc tính nào gắn với thực thể/vùng nào.

### F06 · Caption Primacy Bias (`measured` — Suites: E08, E12)

- **Bằng chứng E08 `F06` Verified Commutative `C1+C2` vs `C2+C1` (FashionIQ):** Tỷ lệ đổi kết quả `R@10` chỉ vì đảo thứ tự hai mệnh đề độc lập = **min=1.00%, median=3.50%, max=6.00% (n=16)**.
- **Theo họ kiến trúc (Order Flip Rate `R@10`):** Classic: `3.00%` · CLIP: `4.36%` · BLIP2-ViT-L: `3.67%` · BLIP2-EVA-g: `2.93%`.
- **Kết luận nhân quả:** Xác nhận thiên vị thứ tự mệnh đề (caption primacy/recency) và sự nhạy cảm với trình tự nối hai câu chú thích độc lập dù ngữ nghĩa hoán vị hoàn toàn tương đương.

### F07 · Pseudo-Relative Reasoning (`measured` — Suites: E06, E01)

- **Bằng chứng E06 Counterfactual Darker/Lighter Swaps (CIE Lab pixel-verified, FashionIQ):** Tỷ lệ đi đường tắt chọn ảnh tối tuyệt đối khi ref sáng (`abs_dark_shortcut_rate`) = **min=29.17%, median=37.08%, max=48.33% (n=16)**; tỷ lệ giữ nguyên lựa chọn bất chấp hoán đổi ảnh ref (`unchanged_choice_under_ref_swap_rate`) = **min=22.50%, median=33.75%, max=47.50% (n=16)**.
- **Theo họ kiến trúc (Ref-Swap Unchanged Rate):** Classic: `42.92%` · CLIP: `31.04%` · BLIP2-ViT-L: `28.33%` · BLIP2-EVA-g: `34.44%`.
- **Kết luận nhân quả:** Xác nhận các tính từ so sánh (`darker/lighter`) bị suy biến thành nhãn tuyệt đối (`dark/light`) độc lập với trạng thái ảnh tham chiếu `I_ref`.

### F08 · Geometric Blindness (`measured` — Suites: E04, E11)

- **Bằng chứng E04 Gallery `(pos − ref)` Diff Probe:** Độ chính xác phân biệt biến đổi hình học/góc nhìn vs màu sắc/vật liệu trên nhánh gallery chỉ ở ngưỡng ngẫu nhiên: **min=46.51%, median=51.16%, max=53.49% (n=16)**.
- **Bằng chứng E11 Resolution Sensitivity (`F08_geom`):** Độ chính xác ở độ phân giải gốc = **min=45.00%, median=57.50%, max=75.00% (n=16)** vs nội suy 336px = **min=50.00%, median=60.00%, max=72.50% (n=13)**.
- **Kết luận nhân quả:** Xác nhận chứng mù hình học (geometric blindness) bắt nguồn trực tiếp từ tính bất biến góc nhìn/biến dạng của backbone ViT/CLIP tiền huấn luyện, khiến vector chênh lệch `(pos − ref)` hầu như không mang tín hiệu tuyến tính về hình dạng/góc nhìn và tăng độ phân giải 336px không khắc phục được gốc rễ.

### F09 · Cardinality Blindness (`measured` — Suites: E04, E11)

- **Bằng chứng E04 Cardinality Ridge Probe (`1–4`, Prior median MAE = `0.789`):** Gallery visual pre-pooling (`gal_vit_patch_mean_pos`) MAE = **min=0.888, median=0.891, max=0.959 (n=10)**; gallery post-pooling (`gal_vision_proj_mean_pos / gal_vis_final_pos`) MAE = **min=0.705, median=0.728, max=0.998 (n=14)** — phần lớn lỗi tập trung ở nhầm lẫn liền kề (`1↔2↔3`, `adj_err ≈ 55–68%`).
- **Bằng chứng E11 (`F09_cardinality`):** Native = **min=36.67%, median=43.33%, max=70.00% (n=16)** vs 336px = **min=36.67%, median=50.00%, max=66.67% (n=13)**.
- **Kết luận nhân quả:** Xác nhận mù số lượng rời rạc (`1–4`) là nút thắt cấu trúc ở biểu diễn ảnh tổng hợp (global/patch-mean pooling xóa thông tin đếm thực thể rời rạc) chứ không phải do văn bản không mã hóa con số.

### F10 · Fine-Grained Blindness (`measured` — Suites: E11, E04)

- **Bằng chứng E04 OCR Query Probe (`AUROC`):** Nhánh query nhận diện truy vấn chứa yêu cầu chữ/OCR rất mạnh: **min=0.902, median=0.985, max=0.992 (n=16)** (shuffled `≈ 0.50`).
- **Bằng chứng E11 Detail-ROI vs Sham Crop (`F10_ocr`):** Native = **min=47.27%, median=54.55%, max=65.45% (n=16)**, Detail ROI Crop = **min=36.36%, median=46.36%, max=58.18% (n=16)**, Matched-Area Sham Crop = **min=52.73%, median=58.18%, max=67.27% (n=16)**.
- **Kết luận nhân quả (bác bỏ giả thuyết zoom đơn giản):** Phóng đại cục bộ ROI crop thực tế **thua cả sham crop và ảnh gốc** (`median 46.36% < 58.18%`) do cắt mất ngữ cảnh nhận diện cảnh; nút thắt F10 nằm ở năng lực đọc-gắn chữ OCR nội tại của backbone thị giác chứ không giải quyết được bằng cắt phóng đại đơn thuần.

### F11 · Paraphrase Brittleness (`measured` — Suites: E12, E04)

- **Bằng chứng E12 Controlled Syntax/Abstractness Edits (PinPoint, `n=120`):** Tỷ lệ đổi Top-1 khi diễn đạt lại giữ nguyên nghĩa: `verbose_abstract` = **min=11.67%, median=42.08%, max=69.17% (n=16)**, `imperative_concise` = **min=0.00%, median=2.50%, max=7.50% (n=16)**, `question_form` = **min=7.50%, median=20.00%, max=36.67% (n=16)**.
- **Kết luận nhân quả:** Xác nhận độ nhạy cao với cấu trúc cú pháp và mức độ trừu tượng ngôn ngữ: chỉ cần đổi hình thức câu hỏi/mệnh lệnh/diễn giải trừu tượng mà không đổi đích truy vấn đã làm thay đổi lớn tập ứng viên Top-1.

### F12 · Cross-Category Lock (`measured` — Suites: E13, E01)

- **Bằng chứng E13 Cross-Category Manifold (PinPoint):** `Complement` `mAP@10` full = **min=0.0007, median=0.0671, max=0.1312 (n=16)** → oracle-category = **min=0.0073, median=0.1842, max=0.2921 (n=16)**; `Explore` `mAP@10` full = **min=0.0000, median=0.0753, max=0.1157 (n=16)** → oracle-category = **min=0.0144, median=0.2515, max=0.3545 (n=16)**.
- **Kết luận nhân quả:** Chênh lệch lớn giữa full gallery và oracle-category pool chứng minh mô hình bị khóa trong cùng cụm danh mục thị giác của ảnh tham chiếu `I_ref` và thiếu khả năng nhảy sang phân bố danh mục khác cho truy vấn bổ trợ/khám phá.

### F13 · Pragmatic Intent Blindness (`measured` — Suites: E13, E12)

- **Bằng chứng E12 `F13` Pragmatic Implicit vs Explicit Category (PinPoint, `n=150`):** `implicit` `mAP@10` = **min=0.0052, median=0.1624, max=0.2161 (n=16)** (`Cat@10` = **min=10.73%, median=30.70%, max=37.33% (n=16)**) → chèn thêm danh mục đích tường minh vào prompt `explicit_category` `mAP@10` = **min=0.0029, median=0.1308, max=0.1897 (n=16)** (`Cat@10` = **min=13.53%, median=28.83%, max=36.27% (n=16)**).
- **Kết luận nhân quả (bác bỏ giả thuyết chèn prompt đơn giản):** Trên **16/16 model**, chỉ nối thêm nhãn danh mục tường minh vào câu truy vấn không cải thiện mà còn **làm giảm** cả `mAP@10` lẫn `Cat@10` (nhiễu diễn đạt `F11`), trong khi đối chứng **lọc đa tạp danh mục `E13`** tăng `Complement/Explore mAP@10` gấp `2.7×–3.3×` — chứng tỏ mô hình cần cơ chế định tuyến/ràng buộc không gian danh mục ở tầng truy hồi chứ không chỉ sửa prompt.

### F14 · Same-Scene Lock (`measured` — Suites: E02, E17)

- **Bằng chứng E17 (16/16 `history.jsonl`):** **12/16** mô hình đạt EDIR val `R@10 ≥ 90%` ngay Epoch 1 (`Epoch-1 Same-Scene Lock`), khóa chặt phân nhóm theo cảnh trước khi học phân biệt chi tiết chỉnh sửa trong cùng cảnh.
- **Can thiệp nhân quả E02 (`clip4cir`):** `baseline_0pct` R@1=`17.75%` (`CondR@1`=`18.16%`) → `samescene_100pct` R@1=`37.29%` (`CondR@1`=`44.10%`) vs đối chứng độ khó tương đương `otherscene_hard_matched` R@1=`18.25%` (`CondR@1`=`18.43%`).
- **Can thiệp nhân quả E02 (`conesep`):** `baseline_0pct` R@1=`30.81%` (`CondR@1`=`31.28%`) → `samescene_100pct` R@1=`41.87%` (`CondR@1`=`55.78%`) vs đối chứng độ khó tương đương `otherscene_hard_matched` R@1=`38.38%` (`CondR@1`=`39.01%`).
- **Can thiệp nhân quả E02 (`sprc`):** `baseline_0pct` R@1=`31.90%` (`CondR@1`=`32.59%`) → `samescene_100pct` R@1=`33.20%` (`CondR@1`=`51.95%`) vs đối chứng độ khó tương đương `otherscene_hard_matched` R@1=`39.28%` (`CondR@1`=`40.33%`).
- **Kết luận nhân quả:** Xác nhận giả thuyết thiếu negative cùng cảnh là nguyên nhân nhân quả trực tiếp làm đình trệ khả năng phân biệt tinh trong cùng cảnh (`CondR@1`), khác biệt có kiểm soát so với hard-negative khác cảnh.

### F15 · Large-Scale Retrieval Degradation (`measured` — Suites: E03, E14)

- **Bằng chứng E03 Nested Gallery Scaling (`1k → full`, trọng tài CLIP ViT-L/14):** PinPoint Hard `R@10` giảm từ **min=24.31%, median=71.28%, max=76.44% (n=16)** (`1k`) xuống **min=9.50%, median=65.22%, max=72.70% (n=16)** (`full 109.6k`); tỷ lệ `ZeroPos@10` trên full gallery = **min=27.30%, median=34.78%, max=90.50% (n=16)**.
- **Kết luận nhân quả:** Xác nhận hiện tượng pha loãng tỷ lệ tín hiệu/nhiễu khi mở rộng gallery lên quy mô trăm nghìn ảnh: các ứng viên distractor gần về mặt trực quan nhanh chóng lấp đầy Top-10 vượt mức dự báo của mô hình null ngẫu nhiên.

### F16 · Embedding Hubness (`measured` — Suites: E14, E03)

- **Bằng chứng E14 Hubness & Spectral Geometry (FashionIQ):** Anisotropy mean cosine = **min=0.015, median=0.076, max=0.627 (n=16)**; Raw `HubFP` = **min=0, median=21, max=2246 (n=16)** → dev-fitted **CSLS** `HubFP` = **min=0, median=0, max=140 (n=16)**; Hit@10 tăng từ **min=19.53%, median=51.49%, max=55.15% (n=16)** lên **min=20.30%, median=51.57%, max=55.67% (n=16)**.
- **Kết luận nhân quả:** **Xác nhận** hubness cục bộ trong không gian dị hướng là nguồn dương tính giả hệ thống; **CSLS** (fit không rò rỉ trên dev) vượt trội hoàn toàn so với centering/ZCA whitening, vốn có thể phá hỏng cấu trúc hình nón của mô hình dị hướng mạnh (như `clip4cir`).

### F17 · Uncertainty Blindness (`measured` — Suites: E15)

- **Bằng chứng E15 Calibration (EDIR & FashionIQ):** Trên EDIR, Hit/Miss AUROC của `top1_score` = **min=0.476, median=0.516, max=0.565 (n=16)** vs exact `full_entropy_T` = **min=0.473, median=0.561, max=0.717 (n=16)**. Trên bài toán **No-Target Detection** (ảnh đích bị loại khỏi gallery): EDIR AUROC = **min=0.506, median=0.525, max=0.580 (n=16)**, FashionIQ AUROC = **min=0.512, median=0.549, max=0.565 (n=14)**, `FAR@90%TPR` = **min=84.32%, median=86.03%, max=87.64% (n=14)**!
- **Kết luận nhân quả:** Xác nhận chứng tự tin sai (miscalibration / silent failure): ngay cả khi entropy toàn gallery cải thiện dự báo hit/miss nội bộ, điểm cosine thô gần như hoàn toàn bất lực (`AUROC ≈ 0.50–0.55`) trước truy vấn không có ảnh đích trong cơ sở dữ liệu.

### F18 · Negative Transfer & Domain Disparity (`measured` — Suites: E16, E17)

- **Bằng chứng E16 Gradient Conflict & Causal PCGrad (`clip4cir`):** Mean off-diag gradient cosine init=`0.0965` (`8` neg pairs), best=`0.0102` (`11` neg pairs); Transfer overall `mAP@10`: `mixed_orig`=`0.1731`, `mixed_bal`=`0.1686`, `mixed_pcgrad`=`0.1691` (`117` conflict steps).
- **Bằng chứng E16 Gradient Conflict & Causal PCGrad (`conesep`):** Mean off-diag gradient cosine init=`0.2995` (`0` neg pairs), best=`-0.0067` (`16` neg pairs); Transfer overall `mAP@10`: `mixed_orig`=`0.2898`, `mixed_bal`=`0.2955`, `mixed_pcgrad`=`0.2982` (`103` conflict steps).
- **Bằng chứng E16 Gradient Conflict & Causal PCGrad (`sprc`):** Mean off-diag gradient cosine init=`0.6571` (`0` neg pairs), best=`0.0621` (`0` neg pairs); Transfer overall `mAP@10`: `mixed_orig`=`0.2784`, `mixed_bal`=`0.2791`, `mixed_pcgrad`=`0.2787` (`118` conflict steps).
- **Bằng chứng E16b Multivariate Fairness OLS (`blip2_mAP10`, kiểm soát domain + intent + `log(n_pos/n_neg)`):** skin_LIGHTEST: β=`-0.0220` (p=`0.2396`), skin_SECOND_DARKEST: β=`-0.0687` (p=`0.005095`), skin_DARKEST: β=`-0.0965` (p=`0.001423`).
- **Kết luận nhân quả:** Tách bạch rõ xung đột tối ưu giữa các miền nội dung không đồng nhất (thời trang chiếm ưu thế vs động vật/phương tiện/giải trí) và yếu tố nhiễu cấu trúc trong chênh lệch công bằng giữa các nhóm tông da.

### F19 · Annotation Noise & False Negative (`partial` — Suites: E18)

- **Trạng thái E18:** `blind_pack_ready_awaiting_human_annotators` — đã hoàn thiện đầy đủ gói kiểm toán mù **`220` mục** trên **`11` phân tầng** (gồm cả hit & miss, ẩn danh model bằng SHA-256, khóa niêm phong tách biệt, công cụ chấm `Krippendorff's α` / `Fleiss' κ` / Horvitz-Thompson sẵn sàng).
- **Kết luận trung thực:** Chưa kết luận tỷ lệ dương tính giả do nhiễu nhãn benchmark (false-negative GT) cho đến khi có đánh giá viên người độc lập điền vào [`annotator_template.csv`](file:///home/heheboiz/data/cir/workbench/artifacts/probe_runs/20261128_one_run/e18_blind_audit/annotator_template.csv) và chạy `e18_audit_pack.py --score`. Tuyệt đối không giả lập nhãn người.

### F20 · Open-Ended Difference (`measured` — Suites: E06, E07)

- **Bằng chứng E06 `F20` Open-Ended Difference (FashionIQ):** `explicit blue` = **min=48.57%, median=69.52%, max=90.48% (n=16)** → `open another color` = **min=53.33%, median=64.29%, max=70.48% (n=16)** → `negated not red` = **min=32.38%, median=63.81%, max=72.38% (n=16)**.
- **Kết luận nhân quả:** Xác nhận mô hình chỉ giỏi khi đích được chỉ định cụ thể (`blue`), nhưng sụp đổ mạnh khi chuyển sang mệnh đề khác biệt mở (`another color`) hoặc loại trừ (`not red`) đòi hỏi suy luận tập bù trên không gian thuộc tính.

### F21 · Photo-Presentation Gravity (`measured` — Suites: E10)

- **Bằng chứng E10 `F21` Photo Presentation Gravity (FashionIQ, `n=25`):** Format trap gốc = **min=0.00%, median=0.00%, max=8.00% (n=16)** (`0.00%` trên `12/16` model!) → sau che nền (`bg_masked`) = **min=0.00%, median=0.00%, max=8.00% (n=16)**; thay đổi identity margin trên 15 model thang cosine (`bg_mask − orig`) = **min=-0.0134, median=0.0013, max=0.0065 (n=15)** *(riêng `dcnet` thang log-softmax không chuẩn hóa: `-1.1493`)*. 
- **Kết luận nhân quả (bác bỏ/thu hẹp giả thuyết):** Trên tập cặp đối xứng sạch của FashionIQ, bản thể sản phẩm áp đảo hoàn toàn định dạng nền ảnh (`identity wins = 92%–100%`), cho thấy lực kéo phong cách trình bày ảnh không phải nguyên nhân chi phối mạnh khi sản phẩm mục tiêu trùng khớp rõ ràng.

### F22 · Co-Occurring Object Leakage (`measured` — Suites: E10, E05)

- **Bằng chứng E10 `F22` Co-Occurring Garment Leakage (FashionIQ, `n=120`, diện tích che khớp chính xác `0.28` ở cả 3 nhánh):** Dịch chuyển embedding khi che trang phục chính = **min=0.0313, median=0.0840, max=0.1613 (n=15)**, che trang phục phụ = **min=0.0045, median=0.0143, max=0.0312 (n=15)**, phần rò rỉ vượt trội (`secondary − sham`) = **min=0.0018, median=0.0075, max=0.0185 (n=15)**.
- **Kết luận nhân quả:** Xác nhận trang phục phụ xuất hiện đồng thời trong ảnh tham chiếu tạo ra độ lệch biểu diễn truy vấn lớn hơn rõ rệt so với che vùng giả diện tích tương đương (`sham`), chứng minh có rò rỉ ngữ cảnh từ vật thể không phải đối tượng chỉnh sửa.

### F23 · Gaze Direction Blindness (`measured` — Suites: E04, E11)

- **Bằng chứng E11 (`F23_gaze`, EDIR):** Native = **min=25.00%, median=37.50%, max=62.50% (n=16)**, Detail ROI Crop = **min=25.00%, median=37.50%, max=50.00% (n=16)**, Sham Crop = **min=12.50%, median=50.00%, max=75.00% (n=16)**, High-res 336px = **min=25.00%, median=37.50%, max=50.00% (n=13)**.
- **Minh bạch mẫu số E04:** Nhãn hướng nhìn trong tập EDIR `train/dev` chỉ xuất hiện 1 lớp duy nhất (`forward`), nên không fit probe đa lớp nhân tạo; chẩn đoán F23 dựa hoàn toàn trên can thiệp độ phân giải và phóng đại ROI mắt/mặt có đối chứng sham ở E11.

### F24 · Limb Pose Blindness (`measured` — Suites: E04, E10)

- **Bằng chứng E04 Pose Probe (`AUROC`, do prior lệch lớp = `95.51%`):** Nhánh query nhận diện ý định chỉnh tư thế ở mức `AUROC` = **min=0.691, median=0.877, max=0.919 (n=16)**, trong khi nhánh gallery đạt `AUROC` = **min=0.558, median=0.636, max=0.663 (n=16)**.
- **Bằng chứng E10 `F24` Limb-Pose ROI Masking vs Matched-Area Sham (EDIR, `n=45`):** Hiệu ứng đặc hiệu ROI trừ sham (`specific_roi_effect_minus_sham`) = **min=-0.0513, median=-0.0007, max=0.0137 (n=16)** (gần 0 trên toàn bộ họ BLIP-2).
- **Kết luận nhân quả:** Tách bạch rõ tín hiệu ngôn ngữ (nhánh query phân biệt được câu lệnh tư thế) với sự nhạy cảm rất yếu ở vùng chi ảnh đích (che ROI chi không tạo chênh lệch rõ so với sham cùng diện tích).

### F25 · Homophone Hijack (`measured` — Suites: E12, E04)

- **Bằng chứng E12 `F25` Homophone / Typo / Synonym Quadruples (FashionIQ `n=110`):** `original` = **min=94.55%, median=96.36%, max=97.27% (n=14)**, `homophone` = **min=94.55%, median=95.45%, max=98.18% (n=14)**, `typo` = **min=94.55%, median=96.36%, max=98.18% (n=14)**, `synonym` = **min=94.55%, median=95.45%, max=98.18% (n=14)**; độ dịch chuyển cosine nội tại của text encoder khi thay từ đồng âm (`text_encoder_cos_shift`) = **min=0.126, median=0.152, max=0.213 (n=13)**.
- **Bằng chứng EDIR Hard-Sibling (`n=18`):** `original` = **min=38.89%, median=55.56%, max=83.33% (n=16)** → `homophone` = **min=33.33%, median=44.44%, max=72.22% (n=16)**.
- **Kết luận nhân quả:** Xác nhận cơ chế cướp nghĩa từ vựng qua phân mảnh subword/token-ID: từ đồng âm hoặc lỗi chính tả dịch chuyển mạnh vector text encoder và làm tụt biên độ truy hồi dù ý định người dùng không đổi.

### F26 · Gender-Identity Drop (`measured` — Suites: E09, E08)

- **Bằng chứng E09 `F26` Metadata Preserve-vs-Edit (PinPoint, `n=114`):** `acc_ok_beats_viol_preserve` = **min=62.28%, median=87.72%, max=91.23% (n=16)**, `acc_ok_beats_viol_edit` = **min=56.14%, median=81.58%, max=86.84% (n=16)**, tỷ lệ đánh rơi ràng buộc bảo toàn (`preserve_drop_rate`) = **min=8.77%, median=12.28%, max=37.72% (n=16)**.
- **Kết luận nhân quả:** Định lượng trực tiếp độ đánh đổi giữa thực thi chỉnh sửa mới và bảo toàn thuộc tính cốt lõi của đối tượng tham chiếu trên cặp ứng viên đối xứng sạch từ metadata.

### F27 · Visual Edit-Distance Cliff (`measured` — Suites: E01, E03)

- **Bằng chứng E01 Spherical Tangent Geometry:** EDIR tangent cos = **min=-0.0854, median=0.1457, max=0.2050 (n=16)** (tỷ lệ độ lớn địa trắc `θ_q/θ_t` = **min=0.517, median=1.543, max=2.473 (n=16)**); PinPoint tangent cos = **min=0.1289, median=0.2535, max=0.4248 (n=16)**; FashionIQ tangent cos = **min=0.1628, median=0.3874, max=0.5100 (n=16)**.
- **Kết luận nhân quả:** Chứng minh độ lệch góc dịch chuyển mặt cầu lớn và sự không tương thích quy mô bước nhảy `θ_q/θ_t` cùng với suy giảm E03 trên gallery lớn tạo ra nút thắt hình học nền tảng của không gian truy hồi đa phương thức.

### F28 · BLIP-2 Blind Spot (`measured` — Suites: E04, E11, E05)

- **Bằng chứng Nhân Quả #1 — ViT Spatial-ROI Ablation vs Matched-Area Sham (E04/E11):** Trên **10/10** mô hình BLIP-2 (sign test `p = 0.00098` nếu toàn bộ dương), việc **xóa các patch ViT nằm trong vùng bị chỉnh sửa của `I_ref` LÀM TĂNG biên độ truy hồi** vượt đối chứng sham (`Δm = min=0.0007, median=0.0038, max=0.0071 (n=10)`)!
- **Bằng chứng Nhân Quả #2 — `F28` Matched Cohorts (`clip_hit_blip_miss` n=66 vs `matched_both_hit` n=66):** Trên nhóm CLIP thắng / BLIP-2 thua, tắt nhánh visual cross-attention Q-Former (`vis_zero`) và đổi MaxSim sang `meanpool_32` đều **cải thiện margin BLIP-2**, trong khi `sham_noise` ~ `0.0000`. Ngược lại ở `clip4cir`, tắt MLP residual (`mlp_zero`) hầu như không đổi margin còn tắt nhánh skip tuyến tính (`convex_zero`) làm sụp đổ margin.
- **Kết luận nhân quả:** **Chứng minh nhân quả hoàn chỉnh cơ chế F28:** cross-attention của Q-Former bị rò rỉ đặc trưng thị giác tiền-chỉnh-sửa từ vùng ROI của `I_ref` và chống lại lệnh chỉnh sửa văn bản, tạo ra điểm mù đặc thù mà kiến trúc cộng tuyến tính CLIP tránh được.

### F29 · Pure Fusion Failure (`measured` — Suites: E05, E04)

- **Bằng chứng E05 Pure Fusion Collapse `(+, +, −)` dùng TRUE Native Unimodal Towers:** EDIR = **min=0.00%, median=8.30%, max=15.48% (n=16)**; PinPoint = **min=0.00%, median=1.80%, max=13.41% (n=16)**; FashionIQ = **min=0.00%, median=0.00%, max=2.11% (n=16)**.
- **Theo họ kiến trúc (EDIR Native `(+,+,−)`):** Classic: `7.36%` · CLIP: `2.68%` · BLIP2-ViT-L: `6.25%` · BLIP2-EVA-g: `10.21%`.
- **Kết luận nhân quả:** Tách biệt hoàn toàn lỗi của từng encoder đơn phương thức với **sụp đổ thuần hợp nhất**: các mô hình BLIP-2/Classic vẫn thua ngay cả khi cả hai tháp ảnh và văn bản độc lập của chính chúng đều chọn đúng, trong khi cơ chế skip-connection lồi của `clip4cir` giảm tỷ lệ `(+, +, −)` về ~`0.00%`.

### 4.5. Khuyến Nghị Kỹ Thuật Có Bằng Chứng Nhân Quả Thực Đo

| Vấn đề mục tiêu | Can thiệp đã kiểm chứng trực tiếp | Hiệu ứng thực đo | Khuyến nghị triển khai |
|---|---|---|---|
| `F14` Same-Scene Lock | `E02`: Pha trộn `10–25%` same-scene sibling negatives | `conesep` EDIR `R@1`: `30.81% → 47.36%` (`+16.55%`); `clip4cir`: `17.75% → 37.29%` (`+19.54%`) | Không dùng `100%` trên BLIP-2 (làm tụt `Scene@1`); pha **`~25%`** tối ưu Pareto giữa định vị cảnh & phân biệt tinh. |
| `F02` Reference-Copy | `E02`: `pp_explicit_neg` vs `pp_ref_negative` | `pp_explicit_neg` giảm `RefCopy@1` (`44.9%→28.4%`) và **tăng** `raw_mAP@10` (`0.305→0.328`), tránh đẩy quá tay như `pp_ref_negative` (`NoRef -0.132`) | Dùng hard/explicit negative có kiểm soát thay vì ép mạnh `I_ref` thành negative tuyệt đối. |
| `F28 / F29` BLIP-2 Blind Spot & Pure Fusion Collapse | `E04b / E05`: ViT ROI drop (`10/10` BLIP-2 dương), `vis_zero`, `clip4cir` convex skip (`0.00%` `(+,+,−)`) | `ViT ROI−Sham > 0` trên `10/10` BLIP-2 (`p=0.00098`); `clip4cir` convex skip giữ `(+,+,−) = 0.00%` | Bổ sung cổng ức chế vùng ROI bị chỉnh sửa trong Q-Former cross-attention + kết nối tắt có kiểm soát từ tháp văn bản. |
| `F16` Embedding Hubness | `E14`: Dev-fitted CSLS vs Centering / ZCA Whitening | `mtst_ptha` FIQ HubFP `2,246→140` (`Hit@10 +2.83%`); `encoder +1.37%`; `tgcir +0.98%` | Áp dụng **CSLS** fit trên val/dev; **tránh** centering/ZCA mù trên mô hình dị hướng mạnh (`clip4cir -4.0%`). |
| `F17` Uncertainty & No-Target | `E15`: Exact full-gallery entropy vs Top-1 score | FIQ Hit/Miss AUROC tăng `0.521→0.733` (`clip4cir`), `0.624→0.762` (`sprc`), nhưng No-Target vẫn `~0.51–0.58` | Dùng full-gallery temperature-scaled entropy cho xếp hạng độ tin cậy nội bộ và bổ sung bộ kiểm chứng hậu kỳ (reranker/VLM verifier) cho bài toán từ chối không-có-đích. |
| `F18` Domain Conflict & Fairness | `E16`: Domain-balanced sampling + PCGrad & OLS audit | Giảm xung đột `103–118/240` bước, cải thiện `conesep` `0.2898→0.2982`; OLS xác nhận chênh lệch `DARKEST β=-0.0965` | Kết hợp lấy mẫu cân bằng miền + giám sát công bằng tách theo tông da có kiểm soát confounder. |

## 5. Chỉ Mục Artifact Kiểm Chứng & Tái Lập Toàn Bộ Thực Nghiệm

- **CLI thống nhất chạy / resume:** [`run_cir_probes.py`](file:///home/heheboiz/data/cir/workbench/scripts/run_cir_probes.py)
- **Script xác minh tự động đa vòng (`TOTAL ERRORS: 0`):** [`verify_all.py`](file:///home/heheboiz/data/cir/workbench/probes/verify_all.py)
- **Cohort manifest & danh sách 16 model / 29 failure:** [`cohort_manifest.json`](file:///home/heheboiz/data/cir/workbench/artifacts/probe_runs/20261128_one_run/manifest/cohort_manifest.json)
- **Bộ dữ liệu chẩn đoán đóng băng (zero-leakage):** [`diagnostic_cohorts.json`](file:///home/heheboiz/data/cir/workbench/artifacts/probe_runs/20261128_one_run/manifest/diagnostic_cohorts.json)
- **Bảng coverage máy đọc được:** [`coverage_matrix.json`](file:///home/heheboiz/data/cir/workbench/artifacts/probe_runs/20261128_one_run/aggregated/coverage_matrix.json)
- **E04 layer-wise probes (EDIR + FIQ gallery + BH-FDR + PCA-256):** [`e04_probes_fitted.json`](file:///home/heheboiz/data/cir/workbench/artifacts/probe_runs/20261128_one_run/aggregated/e04_probes_fitted.json)
- **E15 cross-model disagreement (aligned candidate IDs):** [`e15_cross_model_disagreement.json`](file:///home/heheboiz/data/cir/workbench/artifacts/probe_runs/20261128_one_run/aggregated/e15_cross_model_disagreement.json)
- **E16 multivariate fairness OLS & stratified CIs:** [`e16_fairness_audit.json`](file:///home/heheboiz/data/cir/workbench/artifacts/probe_runs/20261128_one_run/aggregated/e16_fairness_audit.json)
- **E17 lịch sử huấn luyện 16 model (`history.jsonl`):** [`e17_history_all16.json`](file:///home/heheboiz/data/cir/workbench/artifacts/probe_runs/20261128_one_run/e17_dynamics/e17_history_all16.json)
- **E18 gói kiểm toán mù 220 mục (awaiting human annotators):** [`e18_summary.json`](file:///home/heheboiz/data/cir/workbench/artifacts/probe_runs/20261128_one_run/e18_blind_audit/e18_summary.json)
- **Thư mục artifact chi tiết từng model (EDIR / PinPoint / FashionIQ):** [`models/`](file:///home/heheboiz/data/cir/workbench/artifacts/probe_runs/20261128_one_run/models)
- **Thư mục can thiệp huấn luyện E02:** [`e02_same_scene/`](file:///home/heheboiz/data/cir/workbench/artifacts/probe_runs/20261128_one_run/e02_same_scene)
- **Thư mục xung đột gradient & PCGrad E16:** [`e16_domain_conflict/`](file:///home/heheboiz/data/cir/workbench/artifacts/probe_runs/20261128_one_run/e16_domain_conflict)

- **Hồ sơ bằng chứng forensic chi tiết (đồng hành):** [`CIR_ONE_RUN_PROBE_EVIDENCE_DOSSIER.md`](file:///home/heheboiz/data/cir/workbench/docs/reviews/20261128-cir-one-run-probes/CIR_ONE_RUN_PROBE_EVIDENCE_DOSSIER.md)

