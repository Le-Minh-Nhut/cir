# PARITY.vi.md — Phase C verdicts (R1, R2, R3, R4), R10/R11 chưa đo

Mỗi finding: verdict + lệnh chạy lại. Tolerances đã tuyên bố **trước** khi đo và **không** bị chỉnh sau.

---

## R1 — CLIP4Cir FullFT: **CONFIRMED (lỗi provenance, không phải lỗi ranking)**

Nhãn ghi `fashioniq_full_gallery_ref_excluded`; nhãn đúng `fashioniq_original_split`.

Bằng chứng:
* `workbench/third_party/CLIP4Cir/src/validate.py:21-57` — `compute_fiq_val_metrics` **không** có lệnh
  ghi mask nào; `F.normalize(index)` ở `:41`, `distances = 1 - pred @ index.T` ở `:44`, `argsort`
  ascending `:45`, label theo target `:49-50`.
* `reference_mask` chỉ tồn tại trong `compute_cirr_val_metrics` (`:162-166`, hàm bắt đầu `:135`) —
  nhánh CIRR, không chạy cho FashionIQ.
* Registry đã đúng: `workbench/registry/models.yaml:1184` → `fashioniq_original_split`.
* Gallery của chính run: 3817/6346/5373 ⇒ full split, **không** phải pair-union.

Trả lời câu hỏi "reference có bao giờ rank trên chính target của nó không": **có thể** — vì reference
vẫn là ứng viên hợp lệ trong cohort này, nó chỉ không phải target. Không có mask nào loại nó.

Xử lý: migration có kiểm chứng, **không** rerun (ranking không sai):
```
PYTHONPATH=$PWD workbench/envs/ctl312h/bin/python \
  workbench/scripts/migrate_protocol_label.py \
  workbench/docs/reviews/20261110-e2e-repair/raw/clip4cir_fullft.migrated.json \
  --to fashioniq_original_split \
  --evidence workbench/docs/reviews/20261110-e2e-repair/raw/r1_migration_evidence.json
PYTHONPATH=$PWD workbench/envs/ctl312h/bin/python -m pytest workbench/tests/test_protocol_label_migration.py -q
```
Kết quả: 5 test PASS; `aggregate` và `categories` **identical** với bản gốc; chỉ thêm `migrated_from`
và đổi nhãn; sidecar giữ sha256 trước/sau. Artifact gốc của chiến dịch trước **không** bị sửa.

---

## R2 — ENCODER transform: **CONFIRMED (bất đối xứng là thật, và nó ở phía native)**

Đo trên cùng một ảnh (`data/FashionIQ/resized_image/dress/B0007WIZYE.jpg`, 256×256), ba nhánh tách
riêng — `raw/r2_transform_tensors.json`:

| nhánh | max_abs | nghĩa |
|---|---|---|
| `preprocess_val` vs `preprocess_val` (control) | **0.0** (equal) | control hợp lệ |
| `preprocess_train` lặp lại trên cùng ảnh | **3.8270** | transform train **ngẫu nhiên** |
| `preprocess_train` vs `preprocess_val` | **3.8270** | khác biệt thật, không phải nhiễu đo |

`preprocess_train` = `RandomResizedCrop((224,224), scale=(0.9,1.0), ratio=(0.75,1.3333), bicubic)`;
`preprocess_val` = `Resize(224, bicubic)` + `CenterCrop(224)`.

**Câu hỏi "category nào dùng transform nào" — trả lời bằng source:** `datasets.py:139,141,177,178`
dùng `stage=0` (= `preprocess_train`) cho **ảnh query**, và `:195,202` dùng `stage=1`
(= `preprocess_val`) cho **gallery**. Nhưng ở nhánh val-split bài toán, target của query cũng đi qua
`:178` với `stage=0`. Vậy: **query-side = train transform (ngẫu nhiên), gallery-side = val transform.**

Điều này **không** do adapter gây ra: adapter dùng `preprocess_val` cho mọi ảnh
(`workbench/composition/model_encoder.py:329,348`). Ba thứ được tách đúng như yêu cầu:
(1) native fidelity — transform train thật sự ngẫu nhiên, đo được;
(2) same-input control — val vs val bằng 0 tuyệt đối;
(3) CIRCUS adaptation — adapter chọn val transform, một lựa chọn **khác** native ở phía query,
    và đó là nguồn khác biệt thật, không phải lỗi triển khai.

Chạy lại:
```
PYTHONPATH=$PWD workbench/envs/encoder310-adapted/bin/python \
  workbench/docs/reviews/20261110-e2e-repair/scripts/pc_r2_transform_tensors.py
```

---

## R3 — parity harness (mask cho rank vs mask cho tie probe): **CONFIRMED**

Hai lớp bằng chứng.

**(a) Ca đồ chơi tái lập được** — `raw/r3_tie_harness.json`, reference 1.0, ảnh khác 0.8,
target 0.5, candidate 0.5000001:

| đại lượng | giá trị |
|---|---|
| rank không mask | **4** |
| rank có mask reference | **3** ⇒ target **#4 → #3** đúng như mô tả |
| gap trong ma trận **không mask** | **1.192e-07** (≈ mức float32, dưới 1e-6) |
| gap trong ma trận **có mask** | **99999997952.0** (= 1e11 sai khác do -10e10) |
| vị trí thực sự vượt | **chỉ `0`** (chính reference) |
| target vs candidate | **1.192e-07** |

Kết luận: "gap thay đổi ~1e-7" và "rank đổi" là hai **phép đo khác mask**. Mask dùng cho rank phải là
mask dùng cho probe; nếu không, gap bị chi phối bởi giá trị `-10e10`, không bởi score.

Khác biệt nhỏ so với mô tả, ghi lại cho đúng: với gallery **4** phần tử (có thêm ảnh 0.8) target đi
**#4 → #3**; bỏ ảnh 0.8 còn **3** phần tử thì đúng **#3 → #2** — cả hai đã đo, cả hai đều nằm trong
`raw/r3_tie_harness.json`. Delta `1.192e-07` trùng khít ở cả hai cách dựng; đó mới là đại lượng được kiểm.

**(b) Ca thật từ NPZ** (`raw/r3_real_case.txt`) — `native.{cat}.npz` + `adapter.{cat}.npz`,
rank tính lại với **một** mask duy nhất:

* `dress`: query 3, target position 2627 — `preprocess_train` rank **28**, `preprocess_val` rank
  **31**; vị trí vượt: `595, 2558, 1228`; gap nhỏ nhất **1.559e-3** (train) và **3.639e-4** (val).
  Cả hai **> 1e-6** ⇒ **không phải tie**.
* **Giới hạn phạm vi, phải nói rõ:** chỉ **1/4** target của sample khớp chính xác một hàng trong
  gallery (3 target còn lại không định danh được), nên đây là **một** ca, không phải population.
  So sánh adapter-vs-native trên population trong file là **không hợp lệ** ở script này (thứ tự
  `adapter.queries` không được chứng minh trùng thứ tự 4 sample) — các con số đó **không** được dùng
  làm kết luận.

Verdict: cơ chế harness **CONFIRMED** (mask lệch giữa rank và probe tạo ra kết luận tie giả). Nhưng
kết luận "hai ca thật là tie" **DISPROVED** cho ca đo được: gap 1.6e-3, không phải 1e-7.

Chạy lại:
```
PYTHONPATH=$PWD workbench/envs/ctl312h/bin/python workbench/docs/reviews/20261110-e2e-repair/scripts/pc_r3_tie_harness.py
PYTHONPATH=$PWD workbench/envs/ctl312h/bin/python workbench/docs/reviews/20261110-e2e-repair/scripts/pc_r3_real_case.py
```

---

## R4 — ENCODER `learnable_relation`: **CONFIRMED, và nặng hơn mô tả**

Đo độc lập, hai process sạch (`raw/r4_*`):

| đại lượng | giá trị đo |
|---|---|
| checkpoint keys | **417** |
| key chứa `relation` | **0** (`[]`) |
| `state_dict()` entries | **415** |
| `learnable_relation` có trong `state_dict` | **False** |
| kiểu attribute | `Tensor` (không phải `Parameter`) ⇒ **không được đăng ký** |
| `named_parameters()` chứa `relation` | **[]** |
| device | `cuda:0` (`.cuda()` đã chuyển, nhưng vẫn là Tensor thường) |
| requires_grad | `True` (vô nghĩa: không nằm trong optimizer) |
| **unexpected keys** khi `load_state_dict` | `backbone.binding_decoder.fc.0.bias`, `...fc.0.weight` |
| hai construction không seed, cùng process | max_abs **4.923079013824463** |
| cùng seed trước construction | max_abs **0.0** |
| hai seed khác nhau | max_abs **4.862130165100098** |

Nguồn: `workbench/third_party/ENCODER/model_try2.py:93`
`self.learnable_relation = nn.Parameter(torch.randn(N_p, dim, dtype=torch.float32)).cuda()` —
`.cuda()` trả Tensor thường nên attribute **không** vào `_parameters`.

Số của operator (417/0/415/~4.2) **đúng về bản chất**; giá trị chênh lệch tôi đo được là
**4.923** (không seed) chứ không phải 4.2 — chênh lệch phụ thuộc RNG, không phải hằng số.

Thêm một phát hiện operator chưa nêu: checkpoint có **2 key mà model không dùng**
(`binding_decoder.fc.0.*`). Nghĩa là `miss=0, unexpected=2` — `strict=False` che mất.

Chính sách khuyến nghị (chưa áp dụng — Phase C không được tự đổi khoa học):
1. **Seed trước construction** là đủ và rẻ nhất: `torch.manual_seed(S)` trước `Encoder(...)`,
   chứng minh bằng `seeded_same_seed_max_abs = 0.0` ở trên. Ghi `S` vào run identity/cache.
2. Nếu không muốn seed toàn cục: lưu tensor đã rút ra + sha256 vào run dir, đọc lại khi replay.
3. Không dùng tensor `arange` làm "released weight" — checkpoint không chứa relation, nên mọi giá trị
   đều là lựa chọn của workbench và phải được khai báo như vậy.

Chạy lại:
```
PYTHONPATH=$PWD workbench/envs/encoder310-adapted/bin/python workbench/docs/reviews/20261110-e2e-repair/scripts/pc_r4_relation.py --seed 0 --repeat 2
PYTHONPATH=$PWD workbench/envs/encoder310-adapted/bin/python workbench/docs/reviews/20261110-e2e-repair/scripts/pc_r4_determinism.py
```

---

## R10 (LIMN scope): **CONFIRMED — giữ PARTIAL với denominator canonical**

Đo lại trên artifact của chiến dịch (`limn_dual_scope.json`), không suy luận:

* `canonical_status` = `PARTIAL`, `scored_count 6001`, `unscored_count 2`, `unscored_qids`
  `["7:3038","7:3790"]`; `canonical_cohort.full.query_count = 6003` (denominator canonical).
* `gallery.count = 74381`, `protocol_id = circus_fashioniq_release_v1`, nguyên văn:
  "the FULL released gallery for every scope: no category filtering". Vậy scope quyết định
  **checkpoint/embedding space**, **không** quyết định gallery — đọc sai chỗ này sẽ biến một
  diagnostic thành "đổi phạm vi để cải thiện điểm".
* `queries["7:3038"].release.loader_category = null`, `matched_subsets = ["full"]` ⇒ loader **không**
  gán được category cho 2 query này. Đó là lý do chúng unscored, không phải vì rank kém.
* Không có scope nào được chọn làm canonical: `note` ghi "no scope is chosen as 'the' answer", và
  `rank_convention_crosscheck`: 36/36 query đã-score, `all_agree: true` ⇒ thứ tự gallery + tie rule
  của cohort canonical đúng là cái đã ghi.

Verdict: **CONFIRMED**, canonical giữ **PARTIAL 6001/6003**; diagnostics shirt/toptee ở bảng riêng.
Không được chọn scope theo rank/score, không được suy category từ target — cả hai điều đó artifact
này tuân thủ (category = `null` cho 2 query, không bị lấp bằng suy đoán).

## R11 (ties/BLAS): **CONFIRMED phạm vi, DISPROVED là nguyên nhân — chỉ cho các nhóm đã đo**

Đo lại trên `COMPOSITION_MATRIX.csv` (`blas_pin`), phạm vi ghi rõ từng dòng:

| model | threads | đường tính | kết quả A/B | suy ra |
|---|---|---|---|---|
| `encoder` | requested=1 / effective=1 (verified) | GEMM `queries @ gallery.T` | `changed=1`, `max_drank=1`, r10/r50 **delta 0.0**, nDCG 2.18e-05 | BLAS **không** phải nguyên nhân chênh cũ→mới; nguyên nhân là đổi `adapter_module_source_hash` + đổi interpreter |
| `conesep` | 1/1 | `np.einsum(optimize=True)` → gemm | 17/18009 rank đổi **1** bậc, nDCG ~1e-7, R@10/R@50 **y hệt**, cùng interpreter | GEMM near-tie **thật**, nhưng không đổi metric |
| `limn` | 1/1 | `query @ gallery.T` theo block 8192 dòng | `changed=0`, `max_drank=0`, delta **0.0** | không có ca nào để kết luận |

Ba điều **không** được rút ra (theo đúng bài học): margin tới #1, cùng source hash, hay R@10/R@50
ngược chiều — mỗi thứ đó **không đủ** để chỉ định nguyên nhân. Ở đây chỉ có một kết luận dương tính
hẹp: với `conesep`, tồn tại near-tie ở mức 1e-7 **và** nó không đổi metric; với `encoder`, BLAS bị
**loại trừ** bằng A/B có kiểm soát. Ngoài ba model này, **không** có phạm vi nào được tuyên bố.

Verdict: **CONFIRMED** cho phần đã đo; **BLOCKED** cho phần chưa đo (các model còn lại).

---

## Tổng hợp verdict

| finding | verdict |
|---|---|
| R1 CLIP4Cir FullFT | **CONFIRMED** (provenance; đã migrate + test) |
| R2 ENCODER transform | **CONFIRMED** (train transform ngẫu nhiên; control = 0.0) |
| R3 harness tie/mask | **CONFIRMED** cơ chế; **DISPROVED** "ca thật là tie" (gap 1.6e-3) |
| R4 `learnable_relation` | **CONFIRMED** (+ 2 unexpected checkpoint keys) |
| R10 LIMN scope | **CONFIRMED** — canonical giữ PARTIAL 6001/6003, gallery 74381 không lọc |
| R11 ties/BLAS | **CONFIRMED** phạm vi (encoder loại trừ BLAS; conesep near-tie 1e-7 không đổi metric; limn 0 ca); phần còn lại **BLOCKED** |
| R2 × R9 (phân loại 4 block) | **CONFIRMED** — 3 GENUINE_FAIL (batch), 6 NOT_COMPARABLE, 3 DENOMINATOR_ARTEFACT; encoder 2 rank non-tie, LIMN 0 |

---

## R2 × R9 — cái nào thật sự là failure

Đo lại độc lập, không import script cũ: `raw/encoder_parity/native.{cat}.npz` có **cả hai** population
(`population_native` = native chạy `preprocess_train`, `population_val` = native chạy `preprocess_val`),
nên "sàn so sánh" đo được trực tiếp thay vì suy ra.

**Sàn native-vs-native (cùng model, cùng query, khác transform):**

| category | native(train-tf) vs native(val-tf), 2017/2038/1961 query |
|---|---|
| dress | max_abs **0.120392**, mean_abs 0.003350 |
| shirt | max_abs **0.122951**, mean_abs 0.003715 |
| toptee | max_abs **0.125173**, mean_abs 0.003375 |

**Bốn block `within_tolerance: false` của `g1_g2.encoder.*.adapter.json` — phân loại:**

| block (dress / shirt / toptee) | số đo | phân loại |
|---|---|---|
| `layer4_forward_sample.query_native_tensors` | max_abs **0.021249 / 0.041421 / 0.023708** — **trùng tới 6 chữ số** với sàn sample-level native(train-tf) vs native(val-tf) (**0.021249 / 0.041421 / 0.023708**) | **NOT_COMPARABLE** |
| `population.features_vs_native_train_query` | max_abs **0.120392 / 0.122951 / 0.125173** — **trùng tới 6 chữ số** với sàn population (bảng trên) | **NOT_COMPARABLE** (max_rel 223487 là `0.120392 / 5.39e-7`, tức denominator ~5.4e-7) |
| `population.features_vs_native_val_control` | max_abs **1.100e-6 / 1.278e-6 / 0.987e-6** (≈9× dưới `forward_max_abs=1e-5`), `min_cosine` 0.99999999999 ≥ 0.9999999 **PASS**; chỉ max_rel 3.35 / 0.75 / 7.85 fail, với denominator `1.1e-6/3.35 = 3.3e-7` | **DENOMINATOR_ARTEFACT** |
| `batch_size_1_vs_n` | max_abs **3.204e-07 / 2.813e-07 / 3.278e-07** (≈312× dưới `batch_max_abs=1e-4`), `min_cosine` 0.9999999999992; chỉ max_rel 7.80e-3 / 3.13e-3 / 1.16e-2 fail, denominator `3.2e-7/7.8e-3 = 4.1e-5` | **GENUINE_FAIL** — tái lập được (block trên feature tất định), nhưng *genuine* ở đây nghĩa là "batch effect thật và lặp lại", **không** có nghĩa là sai số lớn |

Sửa lại một chi tiết trong mô tả trước: `features_vs_native_val_control` fail `max_abs` ở mức **~9×
dưới** atol, không phải 3 orders; cái vượt 3 orders là `max_rel` (3.35 so với 1e-4).

**Đếm nào bảo vệ được.** Với encoder, mỗi category có **1** block thất bại thật (`batch_size_1_vs_n`)
⇒ **3** block trên 3 category, cộng **2** thay đổi rank không-tie (dưới top-50, không ảnh hưởng
R@10/R@50) ⇒ **5** sai lệch thật, trong đó **2** cái có khả năng ảnh hưởng metric và **0** cái thực sự
đổi metric. Vậy:
* "**13**" (gate) **không** bảo vệ được — nó đếm cả 6 block `NOT_COMPARABLE`/`DENOMINATOR_ARTEFACT`.
* "**2**" (báo cáo trước) **không** bảo vệ được — bỏ sót 3 block `batch_size_1_vs_n`.
* **"5"** (3 batch-rel + 2 rank) là con số duy nhất đứng được; nếu chỉ muốn nói "cái có thể đổi metric"
  thì phải viết **2**, và phải nói rõ 2 đó **không** đổi R@10/R@50.

**Cập nhật theo artifact hiện hành — chỉ phần đếm hạng.** `scoring_ranking.rank_differences_vs_native_val_control`
của `g1_g2.encoder.{dress,shirt,toptee}.adapter.json` nay là `count: 0, non_tie_count: 0, min_gap_abs:
0.0` ở cả ba category, nên số thay đổi rank không-tie là **0**, không phải **2**. Đếm hiện hành:
**3** block `batch_size_1_vs_n`, **0** rank, **0** đổi R@10/R@50. Hai `max_abs` và taxonomy
`NOT_COMPARABLE`/`DENOMINATOR_ARTEFACT` trong bảng trên **không** đổi (`ENCODER_RELATION.md` §11: chín
`max_rel` ở lại nguyên). Nguồn của thay đổi: `ENCODER_RELATION.md` §4 (dress `7:1565` 1215/1216 →
**1222/1222**; shirt `7:3362` 1373/1372 → **1372/1372**).

**limn G2 — đã đo lại, "3" bảo vệ được, "1" thì không.** `g1_g2.{dress,shirt,toptee}.adapter.json`
(4 block/category, không có file `.limn.` riêng — tên file dùng tên category):
mỗi category có **đúng 1** block `within_tolerance: false`, và cả ba đều là cùng một block
`layer2_4_forward/batch_size_1_vs_n`:

| category | max_abs | max_rel | declared `batch_max_rel` | `min_cosine` |
|---|---|---|---|---|
| dress | 1.751e-07 | **1.140e-02** | 1e-3 | 0.99999999999919 |
| shirt | 1.974e-07 | **2.698e-03** | 1e-3 | 0.99999999999894 |
| toptee | 1.714e-07 | **7.509e-03** | 1e-3 | 0.99999999999934 |

`max_abs` nằm dưới `batch_max_abs=1e-4` khoảng 500–580×, `min_cosine` PASS; chỉ `max_rel` vượt, và
denominator ở đây cỡ `1.75e-7/1.14e-2 = 1.5e-5`. Đây **cùng loại** với block `batch_size_1_vs_n` của
encoder ⇒ xếp **GENUINE_FAIL** (hiệu ứng batch tái lập được), không phải artefact thuần tuý.
LIMN có **0** chênh lệch rank (`layer2_4_forward` là block duy nhất fail). Vậy:
* "**3**" (3 block, mỗi category 1) là con số bảo vệ được;
* "**1**" trong báo cáo trước **sai** — nó đếm một block chung cho cả 3 category;
* "**13**" của gate là đếm chéo encoder+limn, không phải của riêng LIMN.

**Một câu kết luận cho R3:** tie probe của harness **không hợp lệ** vì nó đo gap trong cửa sổ không áp
dụng eligibility mask, **và** riêng ca thật vẫn là non-tie lớn hơn epsilon **1559×** (train) / **364×**
(val) — nên sửa harness **không** cứu được verdict "tie", nó chỉ làm phép đo trung thực hơn.

## Xác nhận `seed_recorded` trong `PROTOCOL_MATRIX.csv`

Đã grep lại để chắc chắn cột `NOT_RECORDED` không phải là khẳng định ẩu:

```
grep -il "seed" workbench/docs/reviews/20261109-native15-composition-limn/raw/native/*.json *.log
  -> chỉ khớp: raw/native/dcnet.stdout.log
grep -rln '"seed"|manual_seed|seed=' workbench/artifacts/logs/*/command*.json workbench/registry/models.yaml
  -> không khớp file nào
grep -l "SEED" .../raw/native/*.json -> không khớp
```

Dòng khớp duy nhất **không** phải seed đánh giá: `dcnet.stdout.log:2` in
`Namespace(..., seeds='0')` — đó là `--seeds 0` của DCNet, và chuỗi `'0'` được truyền vào chỗ mà
upstream mong đợi **một list**, nên giá trị hiệu dụng chưa được chứng minh. Không file evidence nào
khác (report JSON, run dir, env, registry) ghi seed. Kết luận: **`NOT_RECORDED` đúng cho 14/15 cấu
hình**; riêng `dcnet` cột `seed_recorded` **đã được sửa** thành
`--seeds '0' on the command line (stdout.log:2), passed where upstream expects a list:
EFFECTIVE_SEED_UNPROVEN` — tức có cờ seed trên command line nhưng **chưa** chứng minh giá trị đó có
hiệu lực, và không được đọc như một seed đã ghi nhận.
