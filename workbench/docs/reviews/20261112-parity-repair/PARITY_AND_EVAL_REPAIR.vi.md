# BÁO CÁO GIAO HÀNG — chiến dịch sửa parity & eval 20261112

Đây là báo cáo **giao hàng** của chiến dịch `research/cir-failure-workbench`, nhánh công việc
`20261112-parity-repair`. Mỗi con số dưới đây được **chép nguyên** từ một record đã viết trong
`workbench/docs/reviews/20261112-parity-repair/`, kèm đường dẫn trích ngay trong dòng. Chỗ nào record
ghi `UNDETERMINED` / `NOT_COMPLETED` thì báo cáo giữ nguyên chữ đó. Không câu nào ở đây khẳng định
một phép kiểm chưa được chạy.

Báo cáo này chỉ tạo **một** file mới. Không sửa file nào khác, không stage, không commit, không chạy
GPU, không restart service.

---

## §0 — Trạng thái đo

Các số trong báo cáo được đo tại **revision của chiến dịch**, không phải tại một tip sha. Điều kiện để
đọc chúng là **bất biến cây production**, không phải `HEAD` bằng đúng một sha:

```
$ git diff --quiet aea143bd8894881ad7d5b341c21dbeff25cfaaaf HEAD -- . \
    ':(exclude)workbench/docs' ':(exclude)workbench/tests'
exit 0        # đo tại HEAD 40c897c31bbdd76b5a76f65935c7dd0f7fef0e94
```

Nguồn: `BASELINE.md:387-393`, `raw/baseline/03-prod-tree-invariant.txt` (`prod_tree_equals_aea143bd=0`),
`20261111-master-audit/FINAL_REPORT_20261112.vi.md:75-85`. Các commit **chỉ-tài-liệu** sau mốc đó không
làm sai các số này, vì chúng không chạm cây production (`BASELINE.md:55-56`: pin cách HEAD 4 commit, cả
bốn chỉ-tài-liệu).

**Cây production hiện KHÁC pin, và đây là sửa lỗi production của chính chiến dịch này.** Đo lại lúc
viết báo cáo (`HEAD = cf6ba468af5b3a6f47174bdd0d50866f23b7006f`):

```
$ git diff --quiet aea143bd HEAD -- . ':(exclude)workbench/docs' ':(exclude)workbench/tests'
exit 1
$ git diff --stat aea143bd HEAD -- . ':(exclude)workbench/docs' ':(exclude)workbench/tests'
 workbench/backend/composition.py | 117 +++++++++++----
 workbench/backend/parity_gate.py | 317 +++++++++++++++++++++++++++++++++++++--
 workbench/frontend/gate.ts       |  11 ++
 3 files changed, 400 insertions(+), 45 deletions(-)
```

Ba file đó đến từ ba commit production của chiến dịch: `cb0b228` (so hai hash module riêng; phục vụ
điều kiện run tự khai), `23e5b8f` (writer+reader `parity-absolute-relative/v2`), `7df87ab` (`standard`
là chuỗi + `standards` breakdown). Vì vậy **mốc đo của báo cáo là pin `aea143bd`**, còn cây đang chạy
mang thêm ba sửa lỗi đó — và pin **phải được nâng**; hệ quả đo được ghi ở §9.

Bất biến dạng "so với chính cây trên đĩa" hiện **fail** (exit `1`), đúng như record đã dự báo và đúng
như `test_handoff_commands.py::test_no_release_document_points_at_a_superseded_source_sha` phát hiện:
`1 failed, 7 passed in 16.41s` (đo trong lượt này, `--basetemp=/tmp/pin_cf6`).

---

## §1 — Phương pháp và những gì đã bị bác bỏ

### 1.1 Bảng phán quyết 13 tiền đề

Nguồn: `PREMISES.md:22-40` (bảng phán quyết, `HOLDS 8 · REVISED 5 · FALSE 0`). Mỗi dòng một tiền đề.

| # | tiền đề (tóm) | phán quyết | đính chính |
|---|---|---|---|
| 1 | gate tồn tại; `GET /api/runs/parity-gate` khám phá record bằng quét | **HOLDS** | — |
| 2 | request v2 để verdict v1 nguyên; `_declared_standard` từ chối v2 | **HOLDS** | — |
| 3 | script producer + JSON cặp tồn tại; trường identity | **HOLDS** | — |
| 4 | 17 đăng ký, 15 không parity, **checkpoint SPRC vắng**, CSMCIR hoãn | **REVISED** | SPRC **ViT-L CÓ** trên đĩa (`sprc_fiq_vitl.pt`, 1 459 756 541 B, `sha256 1938610…bea08` = `models.yaml:1452`); chỉ **ViT-g của paper** là vắng |
| 5 | pin relation-state nối qua **cả hai** process; provenance `learnable_relation` | **REVISED** | Pin **đã** nối ở cả hai đường production; `learnable_relation` được **construct**, không nằm trong `state_dict`. Chỗ hở là **producer** G1/G2 |
| 6 | hai bất đồng hạng được lưu; stable ID khôi phục được | **HOLDS** | — (thêm ID vào run identity) |
| 7 | CLVC-Net dùng chung resolver LIMN; khoá là `(stem, caption)` | **HOLDS** | — |
| 8 | `7:3038`,`7:3790` là 2/119 cuối chưa scope được | **HOLDS** | — |
| 9 | hai artifact native R@10 CLVC mâu thuẫn; có tái lập được không? | **REVISED** | `mrr` **không** được ghi trong cả hai; batch và mask chỉ có trong văn xuôi. Cần rerun có kiểm soát ghi per-query rank + tie-break + batch/mask |
| 10 | LIMN parity: hạng khớp, thiếu bằng chứng hit-count | **HOLDS** | — (đừng trích PASS trong văn xuôi; gate là thẩm quyền) |
| 11 | PAIR bị chặn `BLOCKED_MISSING_SOURCE_CLASSES`; hình học checkpoint; 5.20 %/2×4 đã đo? | **REVISED** | Blocker **của registry** là `BLOCKED_CHECKPOINT_MAPPING_UNVERIFIED`; `BLOCKED_MISSING_SOURCE_CLASSES` là verdict của replay/audit. 5.20 % đo **so với một lớp giả định**; 2×4 đo **từ checkpoint** |
| 12 | block §0.1 của report, metric §4.4b, 5 dòng §9; `PARITY.vi.md:219`; `102-…md:220` | **REVISED** | §0.1 là snapshot `cd9ced2`; HEAD là `40c897c`. Cả hai trích dẫn "neighbourhood" trỏ vào dòng trống / header |
| 13 | tài liệu nào còn ghi `assets/index-DDviexaW.js` | **HOLDS** | — (9 lần nhắc cũ / 7 file; served = `dist` = `index-NnjrdZLG.js`) |

### 1.2 Ba tầng luật chiến dịch đã áp

1. **Verdict suy từ artifact, không từ văn xuôi.** Gate (`workbench.backend.parity_gate`,
   `evaluate_all()`) là thẩm quyền; nơi văn xuôi lệch, văn xuôi bị nêu tên trong mục "Superseded"
   (`MODEL_MATRIX.md:568-618`). Ví dụ: `INPUT_FORWARD_PARITY.vi.md:255-260` nói LIMN "G2 PASS" trong
   khi gate nói `PARITY_MANDATORY_SUBCHECK_FAILED` với 3 failure.
2. **Không có bằng chứng là `INCOMPLETE`, không bao giờ là PASS.** 15 id không có cặp artifact là
   `NOT_RUN`/`INCOMPLETE`, không phải `FAIL` (`MODEL_MATRIX.md:369-375`, footnote 29).
   `native_verified=N` trên **cả 18 dòng** vì nó nghĩa là parity, và parity PASS cho **0** model.
3. **Chỉ ship một bản vá khi lỗi của nó được tái lập.** `BACKEND_DEFECTS.md` phân loại rõ
   `REPRODUCED — fixed` (2 lỗi) so với `REPRODUCED — spec only` (2 lỗi); lỗi 2 còn được ghi là
   "REPRODUCED at a different site" vì cơ sở bị chỉ sai chỗ.

### 1.3 Ba khẳng định REVISED đã suýt gây công vô ích

| khẳng định sai | sự thật đo được | nguồn |
|---|---|---|
| "checkpoint SPRC vắng" | `sprc_fiq_vitl.pt` **có**, 1 459 756 541 B, digest khớp registry; chỉ **ViT-g** của paper vắng | `PREMISES.md:221-296`; `MODEL_MATRIX.md:77-81` |
| "pin ENCODER chưa nối ở cả hai process" | Pin **đã** nối: adapter `model_encoder.py:500` trước `Encoder` `:504`; native `replay/encoder.py:324`, hằng số chung `:59,65` import ở `:36`. Chỗ hở là **producer** `scripts/encoder_parity.py`: `grep -c "manual_seed\|RELATION_SEED\|relation_state"` → **0** | `ENCODER_RELATION.md:8-30`; `PREMISES.md:298-364` |
| "hai macro native CLVC xung đột, không quyết được" | **Cả hai đúng**, ở hai batch size khác nhau; batch 32 và batch 48 tái lập **bit-exact** cả hai artifact | `CLVC_NATIVE.md:73-126`; `MODEL_MATRIX.md:594-603` |

---

## §2 — Ma trận cấu hình

Nguồn máy đọc được: `MODEL_MATRIX.csv` (25 cột, 18 dòng dữ liệu) và bản văn xuôi `MODEL_MATRIX.md`
(cùng thứ tự, cùng ô). Mỗi ô đọc từ artifact / trường registry / lệnh chạy, không suy từ `INVENTORY.csv`
(file đó chỉ dùng làm cross-check, `MODEL_MATRIX.md:550-564`).

**Kích thước:** **17** cấu hình đăng ký (registry order) + **1** target native trong phạm vi nhưng
**không** là id registry (`clip4cir_rn50x4_noft`, `blocker_code: ROLLUP_ONLY`) = **18 dòng**
(`MODEL_MATRIX.md:4-8`). Đếm lại CSV: 18 dòng.

**Từ vựng trạng thái** — chỉ 7 giá trị, không ô nào trống; bằng chứng vắng là `NOT_RUN`/`NONE`
(`MODEL_MATRIX.md:29-30`):

> `PASS` · `PASS_WITH_DECLARED_FAILURES` · `INCOMPLETE` · `FAIL` · `NOT_RUN` · `BLOCKED` · `USER_DEFERRED`

`FAIL` chỉ dùng khi **có** cặp artifact **và** gate từ chối nó; không có artifact thì là
`INCOMPLETE`/`NOT_RUN`, không bao giờ `FAIL` (footnote 29).

### 2.1 Năm cấu hình đổi disposition vì chiến dịch này

| id | trước | sau | lý do (một dòng) | nguồn |
|---|---|---|---|---|
| `encoder` | G1 `FAIL` (3 `relation.<cat>.digest` = `not_recorded`), G2 13 failure | G1 **`PASS`** (0/30), G2 **9** failure | seed relation trong producer; 7 check lật, không phải 3 | `ENCODER_RELATION.md:79-133`; `MODEL_MATRIX.md:498-507` |
| `sprc` | "checkpoint vắng" → `BLOCKED_CHECKPOINT_MAPPING` | **`USER_DEFERRED`** / `USER_DEFERRED_NO_VITG` | ViT-L **có** và đã đo 51.0386/73.3263; chỉ ViT-g vắng | `MODEL_MATRIX.md:364`; `PREMISES.md:236-240` |
| `limn` | `metrics_parity` = `missing` | `metrics_parity` = **measured `true`**, hit count bằng nhau ở cả 3 category; vẫn `FAIL` với **3** `batch_1_vs_n.max_rel` | `t4_parity.py`'s `refresh_metrics_block()` sinh block từ `scripts/limn_hits.py` | `MODEL_MATRIX.md:327-334`; `raw/limn_hits/hit_counts.txt` |
| `clvc_net` | `NATIVE_MACRO_CONFLICT` | **`NATIVE_MACRO_RESOLVED_BY_BATCH`** | batch 32 và 48 tái lập chính xác cả hai artifact; batch size là cơ chế | `MODEL_MATRIX.md:387-403` |
| `clip4cir_rn50x4_fullft` | "hai macro, chưa phân xử" | `INCOMPLETE` / **`TWO_NATIVE_MACROS`** | replay `43.8016931215922` vs roll-up `43.81821850935618`; không parity | `MODEL_MATRIX.md:404-407`, footnote 33 |

Ngoài ra `hint` đi từ **chưa được khám phá** lên `FAIL` / `PARITY_MANDATORY_SUBCHECK_FAILED` — không
nằm trong bảng trên vì trước đó nó không có verdict nào để đổi.

---

## §3 — G1/G2: cái gì đo được, cái gì không

Gate được đọc bằng cách import `workbench.backend.parity_gate` (read-only), không chạy pytest để lấy
verdict. `_discover_models(evidence_root())` trả **`['encoder', 'hint', 'limn']`** — ba model, khám phá
từ **tên file của chính chúng** (`MODEL_MATRIX.md:292-298`).

### 3.1 `encoder`

Artifact: `workbench/docs/reviews/20261109-native15-composition-limn/g1_g2.encoder.{dress,shirt,toptee}.{native,adapter}.json`
(6 file, regenerate 2026-11-12). Verdict: **G1 `PASS`** (30 mandatory check, 0 fail), **G2 `FAIL`** với
**9** failure.

Chín tên check và giá trị đo (`ENCODER_RELATION.md:105-117, 460-470`):

| check | `max_abs` | `max_rel` | declared `max_rel` |
|---|---|---|---|
| `dress.population.features_vs_native_val_control.max_rel` | 1.125e-06 | 0.7499999820540181 | 1e-4 |
| `dress.population.gallery_vs_native.max_rel` | 1.140e-06 | 10.461538814286977 | 1e-4 |
| `dress.batch_1_vs_n.max_rel` | 3.353e-07 | 0.003176237364982574 | 1e-3 |
| `shirt.population.features_vs_native_val_control.max_rel` | 1.296e-06 | 4.236841887544606 | 1e-4 |
| `shirt.population.gallery_vs_native.max_rel` | 1.650e-06 | 0.5312498017651329 | 1e-4 |
| `shirt.batch_1_vs_n.max_rel` | 2.831e-07 | 0.0027347281757461243 | 1e-3 |
| `toptee.population.features_vs_native_val_control.max_rel` | 1.002e-06 | 6.2777780084400545 | 1e-4 |
| `toptee.population.gallery_vs_native.max_rel` | 1.013e-06 | 3.133333651667554 | 1e-4 |
| `toptee.batch_1_vs_n.max_rel` | 3.204e-07 | 0.0015533629650869047 | 1e-3 |

**Toàn bộ chín failure là `max_rel`-only**, với `max_abs` ≤ **1.65e-06**, tức 1–2 bậc dưới atol khai
báo (`forward_max_abs 1e-5`, `batch_max_abs 1e-4`), và `min_cosine ≥ 0.99999999998` ở cả chín block.
`max_rel` chia cho thành phần |reference| **nhỏ nhất** trong ma trận, nên một thành phần near-zero
(~1e-7…2e-4) biến noise float32 thành tỷ số lớn (`ENCODER_RELATION.md:454-459`). Hai tolerance
(`forward_max_rel 1e-4`, `batch_max_rel 1e-3`) **không bị dời** — `max_abs` phẳng so với lần chạy
trước pin trong ~2 % (`:472-476`).

**Cần gì để clear:** hoặc (a) siết `max_abs`-only cho các scope này như một quyết định hiển ngôn có
record riêng, hoặc (b) một ngưỡng `max_rel` có mẫu số sàn (relative-to-row-norm thay vì
relative-to-min-component). Cả hai đều là **đổi định nghĩa đo**, không phải sửa model; không làm trong
chiến dịch này. Điều **không** được làm: dời `forward_max_rel`/`batch_max_rel` cho đến khi xanh.

### 3.2 `limn`

Artifact: `g1_g2.{dress,shirt,toptee}.{native,adapter}.json` (6 file). Verdict: **G1 `PASS`** (18
mandatory check, `input_parity`; ba `relation.*.digest` là non-mandatory vì `limn ∉
RELATION_DRAWING_MODELS`), **G2 `FAIL`** với **3** failure.

`metrics_parity` **nay MEASURED và bằng nhau** — hit count r1/r10/r50 giống hệt hai phía, từng category
(`raw/limn_hits/hit_counts.txt`; `MODEL_MATRIX.md:327-334`):

| category | query | native hits r1/r10/r50 | adapter hits r1/r10/r50 | eq | ranks identical |
|---|---|---|---|---|---|
| dress | 2017 | 375 / 1032 / 1498 | 375 / 1032 / 1498 | true | true |
| shirt | 2038 | 517 / 1141 / 1561 | 517 / 1141 / 1561 | true | true |
| toptee | 1961 | 511 / 1181 / 1609 | 511 / 1181 / 1609 | true | true |

Gate trước → sau (`raw/limn_hits/gate_before.txt` → `gate_after.txt`): `missing ['metrics_parity']` ở
cả ba category → `[]` ở cả ba, và ba check `*.metrics.hit_count_equality` = `True`/`True`.

Ba failure còn lại (`raw/limn_hits/remaining_failures.txt`; `MODEL_MATRIX.md:323-326`):

| check | measured | declared | `max_abs` | `min_cosine` |
|---|---|---|---|---|
| `dress.batch_1_vs_n.max_rel` | **0.011399774792037631** | 1e-3 | 1.751e-07 | 0.99999999999919 |
| `shirt.batch_1_vs_n.max_rel` | **0.002698280178837889** | 1e-3 | 1.974e-07 | …894 |
| `toptee.batch_1_vs_n.max_rel` | **0.007509343787328878** | 1e-3 | 1.714e-07 | …934 |

Chỉ số hạng tương đối fail; tuyệt đối 1.7–2.0e-7. **Cần gì để clear:** cùng lớp quyết định như §3.1
định nghĩa `max_rel`; không phải việc của model.

`metrics_parity` có **negative test** chứng minh nó cắn được: một phía bị lật hit count →
`PARITY_MANDATORY_SUBCHECK_FAILED` nêu tên `dress.metrics.hit_count_equality` `ok=False`; cờ là chuỗi
`"true"` → `PARITY_EVIDENCE_MALFORMED` ("only a recorded boolean can certify or refuse hit-count
equality") (`raw/limn_hits/NEGATIVE_TEST.md`).

Một điểm yếu của cặp LIMN phải nêu: sáu trường identity (`model_id`, `checkpoint_id`, `protocol_id`,
`protocol`, `query_order_sha256`, `gallery_order_sha256`) **đều null**; `input_parity` dựa trên
`layer1_identity` (2017/2038/1961) và bốn boolean `*_identical` (`MODEL_MATRIX.md:335-339`). Đó là
khiếm khuyết thật của cặp, không phải lỗi gate — ghi nhận, không tính điểm.

### 3.3 `hint` (model THỨ BA)

Artifact: `g1_g2.hint.{dress,shirt,toptee}.{native,adapter}.json` — 6 file, viết bởi capture thật
2026-11-12 (native 170 s, adapter 3× ~105 s, dưới flock của `gpu_job.sh`). Producer:
`scripts/third_model_parity.py`. Verdict: **G1 `PASS`** (36 mandatory check, 0 fail), **G2 `FAIL`** với
**2** failure.

| check | measured | declared `batch_max_rel` | `batch_1_vs_n.max_abs` |
|---|---|---|---|
| `shirt.batch_1_vs_n.max_rel` | **5.049934976857354e-03** | 1e-3 | 1.527e-07 |
| `toptee.batch_1_vs_n.max_rel` | **0.04213516386699195** | 1e-3 | 1.527e-07 |

`dress` pass ở 1.526e-04. `min_cosine ≈ 0.9999999999999`, `batch_max_abs ≈ 1.5e-07` ở cả hai category
fail: chênh batch-1-vs-N là ~1e-7 tuyệt đối; số tương đối lớn chỉ vì `max_rel` chia cho thành phần
|reference| nhỏ nhất — cùng artefact mẫu số như §3.1 (`THIRD_MODEL_PARITY.md:230-244`).

Mọi chiều bắt buộc khác **đã đo và giữ**: forward và pixel **bitwise 0.0** ở cả ba category;
`rank_differences` 492/382/323 nhưng **toàn bộ** là sentinel `0` ("ngoài top-50 hoặc bị mask"), nên
**0** khác biệt trên rank native thật; `unexplained_rank_differences` = 0; `score_max_abs_delta_vs_dot`
2.09e-07 / 1.79e-07 / 1.79e-07 trong 1e-6 (`THIRD_MODEL_PARITY.md:185-228`).

Hit count bằng nhau hai phía, từng category (`THIRD_MODEL_PARITY.md:214-218`):

| category | native hits r1/r10/r50 | adapter hits (native protocol) | equal |
|---|---|---|---|
| dress | **432 / 1034 / 1525** | 432 / 1034 / 1525 | true |
| shirt | **663 / 1272 / 1656** | 663 / 1272 / 1656 | true |
| toptee | **628 / 1289 / 1638** | 628 / 1289 / 1638 | true |

`relation_state` **không áp dụng** và **không được thêm**: draw ngẫu nhiên duy nhất trong `__init__`
HINT là `prompt_tokens`, mà grep trên cả cây pinned chỉ thấy nó ở khai báo (`:295`) và init (`:298`) —
forward được đánh giá dùng `self.query_tokens` (`:410`), có trong `state_dict`. Ghi digest cho một
tensor forward không đọc sẽ đặt một giá trị trông-như-đo cạnh một check phải giữ non-mandatory.

**Cần gì để clear:** cùng lớp quyết định `max_rel` như §3.1/§3.2. Tolerance **không bị dời sau khi đọc
số** — `t4_parity.py:35-42` được import, không restate (`THIRD_MODEL_PARITY.md:160-173, 240-244`).

### 3.4 Câu chuyện ENCODER: pin đã nối, **producer** mới là chỗ hở

Pin đã có trong **cả hai** đường production từ trước: adapter `workbench/composition/model_encoder.py:500`
(`torch.manual_seed(relation_seed)` trước `Encoder(...)` ở `:504`, record ở `:231-239`); native replay
`workbench/replay/encoder.py:324` (`torch.manual_seed(RELATION_SEED)`), record `:253-270`; hằng số ở
`model_encoder.py:59` / `:65`, import ở `replay/encoder.py:36` (`ENCODER_RELATION.md:10-18`).

`BindingDecoder.learnable_relation` là `nn.Parameter(torch.randn(N_p, dim)).cuda()`
(`third_party/ENCODER/model_try2.py:93`) — **construct**, vắng khỏi `state_dict()` và khỏi checkpoint
417 khoá (`raw/encoder_parity/relation_proof_construct.out`: `IN_STATE_DICT False`, `SHAPE [2, 512]`).
Không có gì pin draw đó, nên tensor là hàm của RNG toàn cục lúc construct.

**Bản vá** (`ENCODER_RELATION.md:57-77`, 39 thêm / 3 xoá): producer `scripts/encoder_parity.py` import
`RELATION_SEED` từ module production và `manual_seed` ngay trước `Encoder(...)`; record lấy từ
recorder production `replay._relation_record(model, None)` — **không** công thức digest thứ ba; record
được nâng lên gốc document nơi `_relation_checks` đọc. Không tolerance nào dời, không check nào bị xoá,
không FAIL nào bị relabel.

**Sau vá:** cả **sáu** artifact ghi `relation_state.sha256
20db0c27c979e9a0938c71af5142458dfa2070468d0ad3fcb6f24134c2f9ce92` với
`matches_pinned_digest: true`, `seed: 20261111` trên cả sáu (`ENCODER_RELATION.md:175-200`). Gate đi
**16 → 9** failure — **bảy** check lật, không phải ba: ba `relation.<cat>.digest` +
`dress.ranks.non_tie_changes` 1.0 → 0.0 + `dress.ranks.min_gap_vs_tie_epsilon` 6.5717e-4 → 0.0 +
`shirt.ranks.non_tie_changes` 1.0 → 0.0 + `shirt.ranks.min_gap_vs_tie_epsilon` 1.6041e-4 → 0.0
(`ENCODER_RELATION.md:124-133`).

**Hai bất đồng hạng lịch sử NAY ĐÃ HẾT** (`raw/stable_ids/README.md`, `ENCODER_RELATION.md:140-173`):

| | dress `7:1565` | shirt `7:3362` |
|---|---|---|
| reference / target | `B004DTTRA8` / `B0048OYGF4` | `B00AMD5I7U` / `B005LIXPPI` |
| trước (native, adapter) | 1215, 1216 (index **1569**) | 1373, 1372 (index **1352**) |
| **sau (native, adapter)** | **1222, 1222** | **1372, 1372** |
| margin quyết định trước | 1.043081283569336e-07 | 1.1920928955078125e-07 |
| `max_abs` cross-side của **chính trạng thái đó** (float32 noise) | 8.344650268554688e-07 | 7.7858567237854e-07 |
| **margin ÷ noise** | **0.125** | **0.1531** |
| era shift ÷ margin | 34169.28571428572 | 8192.125 |

**Nguyên nhân đo được, chính xác:** cặp **trước** vá dùng **MỘT** relation chung — khoảng cách
cross-side `1.1324882507324219e-06` (dress) / `1.648440957069397e-06` (shirt), tái lập đúng
`max_abs` của chính artifact HEAD tới từng chữ số. Bất đồng là một **knife edge** ở **0.125–0.153 của
noise float32 của chính trạng thái đó**; reseeding đổi **relation nào** cả hai phía dùng, và dịch
feature `0.013255782425403595` / `0.01691335439682007` — 3.4e4× / 8.2e3× margin đó — phá cạnh và đưa
cả hai query về `1222`/`1372` ở **cả hai** phía. Đây **KHÔNG** phải tie (0 cặp score bằng nhau ở mọi
era đã lưu; `1 + #greater + #earlier-equal` bằng đúng vị trí argsort trên cả bốn dòng), và **KHÔNG**
phải "hai relation khác nhau". Cả hai hạng nằm ngoài mọi cutoff nên không hit count nào dịch.

Cross-confirmation: composition run đã seed `encoder_fashioniq_9b806c22221e944ff10e514e/full.json`
cũng ghi `relation_state.sha256 = 20db0c27…`, `matches_pinned_digest: true` — cùng byte, ở một run
không chạm harness này. Pin tái lập qua **ba** process độc lập (`ENCODER_RELATION.md:190-200`).
Negative test: một digest bị đổi trong bản `/tmp` → thêm **đúng một** failure
`relation.shirt.digest` (`:202-213`).

### 3.5 Macro native ENCODER: số đã seed là số duy nhất tái lập được

| id | file | `recorded_at` | macro R@10 | macro R@50 | verdict |
|---|---|---|---|---|---|
| **seeded** | `g1_g2.encoder.<cat>.native.json` | 2026-11-12 | **56.225801935399** | **77.533156738705** | tái lập theo construct |
| registry | `cohort/measurements/encoder_fashioniq.json` | 2026-11-06T14:39:45Z | 56.223101675386 | 77.667385392714 | **NOT_REPRODUCIBLE** |
| review | `raw/native/encoder.json` | 2026-11-09T13:23:23Z | 56.290792549557 | 77.633389132303 | **NOT_REPRODUCIBLE** |

Ngưỡng khai báo **trước khi đọc số**: 1e-3 điểm phần trăm, trên **từng ô**. Kết quả: **không cặp nào**
khớp trong 1e-3 pp trên bất kỳ ô nào; ô nhỏ nhất là `dress` r50 seeded−registry = 0.04958 pp, vẫn
**49.6×** ngưỡng, và đúng bằng **một** hit R@50 trên 2017 query (bước lượng tử `100/2017`). Ô lớn nhất
0.35696 pp. **Macro trông khít hơn chính các category của nó** vì chín ô `seeded − registry` **triệt
tiêu nhau**: tổng `+0.00809` pp, trung bình `+0.00270` pp. Vì vậy một tolerance **macro-only** sẽ gọi
chúng là "reproduced" — đó chính là lý do **bảng per-category là bắt buộc**
(`ENCODER_RELATION.md:306-400`).

Cả hai file lưu **không** ghi relation, seed hay batch size (`relation_state`/`seed`/`batch_size` đều
vắng; `per_query_published: false`), nên chúng ghi **hai hoặc nhiều hơn** draw của một tensor không ai
viết lại — không có một con số nào từng là một phép đo duy nhất. Phát hiện thứ ba: tồn tại **hai**
aggregate "native ENCODER" và registry không biết cái thứ hai (`MODEL_MATRIX.md` chỉ trích registry;
grep `raw/native/encoder.json`, `56.2908`, `77.6333` trong `MODEL_MATRIX.*`/`BASELINE`/`PREMISES` →
không hit) — đúng lớp `TWO_NATIVE_MACROS` mà ma trận gắn cho `clip4cir_rn50x4_fullft` nhưng **chưa**
gắn cho `encoder` (`ENCODER_RELATION.md:401-425`).

---

## §4 — CLVC-Net: hai macro native và hai batch size

Nguồn: `CLVC_NATIVE.md` + `raw/clvc_native/` (per-query JSONL + JSON per category + `summary.json`).
Producer `scripts/clvc_native_ranks.py`. Bundle checkpoint `b504245d…6b2` (repo canonical hasher,
`kind=directory`) tái lập trong run này; source pin `bd9b6889489806cb60676597880f91d918b279cb`,
`dirty=false`.

### 4.1 Verdict tái lập — cả hai artifact đúng

| artifact lịch sử | batch | ô tái lập | macro R@10 / R@50 | Δ |
|---|---|---|---|---|
| `20261111-master-audit/raw/clvc_run3cat_gpu.json` | **32** | **6/6** (R@10/R@50 × 3 category) | **30.02079703540087 / 58.67189400437203** | **0.0** |
| `artifacts/checkpoint_selection/20261106T002617+0700/measurements/clvc_run1.json` | **48** | **9/9** (R@1/R@10/R@50 × 3) | **29.970746517160162 / 58.62171212901515** | **0.0** |

`CLVC_NATIVE.md:75-101` — in dòng theo category. Mỗi lệnh văn xuôi khớp: `--batch-size 32`
(`BLOCKED_MODELS_VERIFICATION.vi.md:214-218`), positional `48` (`CHECKPOINT_SELECTION_REPORT.md:1697-1699`).

**Determinism control:** ba pass riêng biệt ở batch 32 (`batch32`, `batch32_r2`, `batch32_r3`) là
**bit-identical** — **0** rank dịch, **max |Δ| `0.000e+00`**, cùng thứ tự qid, ở cả ba category
(`CLVC_NATIVE.md:148-165`). Đây là điều làm hai phép tái lập ở trên có nghĩa, không phải may.

**Cơ chế bị loại trừ đến chỉ còn batch size** (`CLVC_NATIVE.md:104-126`): checkpoint (cùng 6 file,
digest `b504245d…`), gallery order (2628/3089/2902 ảnh mọi pass, union pair-order), mask (`-10e10`,
`test.py:135`, reference nằm **đúng cuối** 2628/2628 dress và 3089/3089 shirt nên không bao giờ chiếm
slot top-50), text correction (không tồn tại ở đâu trong `test.py`/`datasets.py`/`text_model.py`).
Tie **không** phải nguyên nhân (40/12/14 ở batch 32 vs 36/12/14 ở batch 48) nhưng là confound thật ở
biên. Batch 48 dress wall **268.6 s** ≈ **5.5×** batch 32 dress (44.8–49.6 s) trên cùng GPU — xác nhận
đường `params.batch_size` thật sự đổi vòng lặp.

### 4.2 Độ nhạy batch

| so sánh | category | max |Δ R@1/R@10/R@50| | rank target dịch | query |
|---|---|---|---|---|---|
| batch 1 vs 32 | dress | **0.644522** (R@10 28.805156 → 29.449678) | **1548** | 2017 |
| batch 48 vs 32 | dress | 0.099157 | 427 | 2017 |
| batch 48 vs 32 | shirt | 0.098135 | 378 | 2038 |
| batch 48 vs 32 | toptee | 0.101989 | 372 | 1961 |

**1548/2017 rank dress dịch giữa batch 1 và 32** (76.7 %). Chỉ **một** category chạy batch 1 vì chi phí
— `shirt`/`toptee` **không** chạy batch 1, nói thẳng chứ không ngoại suy (`CLVC_NATIVE.md:128-146`).

### 4.3 Luật rút ra

> **`--batch-size` là một phần của measurement identity của CLVC-Net.** Không được đổi nó cho bất kỳ
> phép đo nào sẽ được báo cáo hay so sánh. Trích **batch 32** cho mọi thứ so với một composition run
> (`replay/clvc_net.py:268` mặc định 32 và mọi composition measurement đã ghi dùng 32). Một con số CLVC
> không kèm `batch_size` **không** so sánh được với các artifact này — đó chính là khiếm khuyết đã sinh
> ra cuộc điều tra này.

Cái **chưa** settled: batch size nào là canonical — không thứ gì trong upstream pinned hay registry chỉ
định một cái; `test.py` chỉ đọc `params.batch_size` (`CLVC_NATIVE.md:207-215`).

---

## §5 — CLVC dual-scope diagnostic

Nguồn: `CLVC_DIAGNOSTIC.md` + `raw/clvc_diag/` (`clvc_dual_scope.json` + 6 JSONL export).
Producer `scripts/clvc_dual_scope.py`.

**Trạng thái: DIAGNOSTIC ONLY. Not canonical. Never a metric.** 12 cell = 2 qid × 2 scope (`shirt`,
`toptee`) × 3 điều kiện CLVC.

| qid | scope | condition | kind | target | rank | score@target | score@rank1 | margin |
|---|---|---|---|---|---|---|---|---|
| 7:3038 | shirt | multimodal | protocol | 7:3416 | 848 | 0.463620 | 0.736086 | 0.272467 |
| 7:3038 | shirt | text_only_black_image | protocol | 7:3416 | 304 | 0.477407 | 0.654882 | 0.177476 |
| 7:3038 | shirt | clvc_image_encoder_only | ablation | 7:3416 | 8702 | 0.303293 | 1.000000 | 0.696707 |
| 7:3038 | toptee | multimodal | protocol | 7:3416 | 1253 | 0.565981 | 0.736116 | 0.170134 |
| 7:3038 | toptee | text_only_black_image | protocol | 7:3416 | 1549 | 0.458470 | 0.666364 | 0.207893 |
| 7:3038 | toptee | clvc_image_encoder_only | ablation | 7:3416 | 1267 | 0.592141 | 1.000000 | 0.407859 |
| 7:3790 | shirt | multimodal | protocol | 7:42304 | 412 | 0.571143 | 0.795686 | 0.224543 |
| 7:3790 | shirt | text_only_black_image | protocol | 7:42304 | 1466 | 0.444272 | 0.757220 | 0.312948 |
| 7:3790 | shirt | clvc_image_encoder_only | ablation | 7:42304 | 5481 | 0.259081 | 1.000000 | 0.740919 |
| 7:3790 | toptee | multimodal | protocol | 7:42304 | 259 | 0.743001 | 0.842818 | 0.099816 |
| 7:3790 | toptee | text_only_black_image | protocol | 7:42304 | 568 | 0.618858 | 0.747598 | 0.128739 |
| 7:3790 | toptee | clvc_image_encoder_only | ablation | 7:42304 | 4535 | 0.445611 | 1.000000 | 0.554389 |

`target_within_export_depth` **false** ở mọi cell (rank 259…8702, ngoài Top-100) — nên rank chính xác
được lưu riêng, không suy từ cửa sổ export (`workbench/composition/topk.py:11-16`).

**Schema dùng lại của LIMN, có trích dòng, không phát minh lại** (`CLVC_DIAGNOSTIC.md:9-31`): skeleton
`t2_dual_scope.py:120-146`; block per-query `:274-291`; block per-cell `:305-327`; row top-100 export
`:330-346`; qid identity `:105-134` (block `release` là thẩm quyền); guard target-là-chính-query
`:66-83`, `:270-273`, `:349-352` (quét toàn report **trước** khi ghi file); rank crosscheck
`:230-268`; depth `:40` `TOP_K = 100`; đặt tên file `:348-352`.

**Bằng chứng không rò rỉ** (`CLVC_DIAGNOSTIC.md:136-160`) — đo, không khẳng định:

| grep | kết quả |
|---|---|
| `clvc_diag` khắp `workbench/` trừ thư mục này + producer | **0** hit (chỉ `__pycache__` của producer) |
| `clvc_dual_scope` trong `workbench/composition/`, `workbench/backend/`, `workbench/tests/` | **0** hit |
| `raw/` hoặc `docs/reviews` trong `workbench/composition/metrics.py` | **0** hit |
| `_review_run_rows` glob `docs/reviews/*/raw/*.json` vs file này | **không** khớp (sâu hơn một cấp) |
| `native_measurements._review_run_payload(payload)` | **`None`** — predicate từ chối (không có `model_id`/`replay_type`/`aggregate`), nên nó không thể thành một served measurement row |
| row `_review_run_rows()` chứa `clvc_diag` | `[]` |

Mọi summary/aggregate đều được đánh dấu `diagnostic: true`, `canonical: false` ở top level, trong cả
12 cell và cả 12 export row; mỗi cell mang `canonical_status` nêu qid vẫn unscored trong metric canonical
6001/6003 (`canonical_status` = `{status: PARTIAL, scored_count: 6001, unscored_count: 2, unscored_qids:
['7:3038','7:3790']}`, đọc từ `raw/clvc_diag/clvc_dual_scope.json`).

**`image_only_empty_text` = `ABLATION_UNSUPPORTED`, không bao giờ gộp với `clvc_image_encoder_only`.**
Nó chỉ được ghi dưới `unsupported_conditions`, không phải một cell. Lý do (nguyên văn từ artifact):
text tower của CLVC tokenise empty edit thành **0** token rồi index `x[0][0]`
(`third_party/CLVCNet/text_model.py:84-86`), raise `IndexError` cho **mọi** hàng
(`model_clvc_net.py:418-435`, `EMPTY_TEXT_REASON` ở `:96`). Không fake.

**Ghi chú trung thực về `clvc_image_encoder_only`:** rank-1 của nó **là chính ảnh reference**, score
≈ **1.0** (`rank_1_gallery_id` bằng reference ở **cả 4** cell ablation). Vì vậy R@1 của điều kiện này là
**self-match**, không phải retrieval — đúng chữ ký của việc rank một gallery có reference trên hàng của
chính ảnh reference. Đừng đọc R@1 đó như chất lượng truy hồi; hãy đọc `target_rank` (8702 / 1267 /
5481 / 4535).

**Delta CPU/GPU được đo, không tuyên bố tương đương** (`CLVC_DIAGNOSTIC.md:107-134`): recompute 6 query
đã chấm từ gallery cache → `max_abs_row_delta` **9.32e-05**, `max_rank_delta` **10** trên gallery
74 381 ảnh; batch-1 vs batch-2 CPU lệch ≤ 1.2e-07, tức dịch chuyển là **DEVICE**, không phải batch
composition. `NOT_COMPUTED_GPU_PENDING`: **không** — cả 12 cell đã tính trên CPU.

---

## §6 — Hợp đồng v2 và các lỗi backend đã vá

### 6.1 `parity-absolute-relative/v2` là thay đổi hợp đồng writer+reader

Nguồn: `V2_STANDARD.md`; code `23e5b8f` (writer+reader) và `7df87ab` (`standard` chuỗi + `standards`).

v1 phán từ **hai aggregate** (`max_abs > atol OR max_rel > rtol`); `max_rel` chia cho thành phần
reference nên một embedding unit-norm với thành phần ~1e-8 biến lỗi tuyệt đối 1e-7 thành **7.8**. v2
khai báo luật **theo từng thành phần**:

```
fail if  abs(adapter - native) > atol + rtol * abs(native)   for ANY component
```

Native là phía **REFERENCE** (`b`), adapter là phía đo (`a`).

**Hình dạng block** (`V2_STANDARD.md:33-78`) — một khoá top-level mới `component_parity`, cộng
`"standard": "parity-absolute-relative/v2"` trên **cả hai** phía:

```json
"component_parity": {
  "standard": "parity-absolute-relative/v2",
  "declared_rule": "abs(adapter - native) <= atol + rtol * abs(native)",
  "reference_side": "native",
  "chunk_rule": "full when components_total <= 4096, else the first 512 components plus every component whose normalized violation exceeds 1",
  "scopes": {"<scope name>": {"dimension": "forward_parity" | "batch_parity",
    "shape": [4,512], "components_total": 2048, "components_emitted": 2048,
    "coverage": "full" | "chunked", "indices": [...], "reference": [...], "actual": [...],
    "sha256": {"reference": "…", "actual": "…", "delta": "…"},
    "mismatch_count": 0, "max_abs": 1.2e-07, "worst_normalized": 0.41, "offenders": [...]}}
}
```

`<scope name>` là tên so sánh của chính artifact (`sample_forward.query`, `sample_forward.target`,
`layer4_forward_sample.query_val_pixels`, `population.features_vs_native_val_control`,
`population.gallery_vs_native`). `dimension` chỉ nhận `forward_parity` cho luật thành phần (chỉ ở đó cả
hai producer giữ một mảng native căn với mảng adapter); `batch_parity`/`cache_off_restart` so adapter với
output **của chính nó**, nên gán nhãn "native" cho mảng trái sẽ mô tả sai phép đo. `sha256` trên **toàn**
mảng, canonical little-endian float64 (`np.ascontiguousarray(a, dtype="<f8").tobytes()`).

**Luật chunk:** mảng đầy đủ khi `components_total <= 4096`; trên ngưỡng đó, dump **512** thành phần đầu
**cộng mọi** thành phần vi phạm (tăng dần, khử trùng) — giữ một scope toàn dân số (~1.5 M thành phần)
trong một file review được mà vẫn phát ra mọi toạ độ mà một verdict failure có thể nêu tên.
`coverage: "chunked"` là một phần của record.

**Aggregate không bao giờ suy được thành kết quả componentwise** (`V2_STANDARD.md:120-130`): gate tự
suy `max_abs` và vi phạm chuẩn hoá tệ nhất từ các giá trị đã phát; khi `coverage == "full"`, các giá trị
đã ghi **phải khớp** phép suy (`math.isclose`, rel `1e-9`) nếu không artifact tự mâu thuẫn và
malformed; khi `chunked`, chúng chỉ được **≥** giá trị suy ra, **không bao giờ** dùng để gọi một scope
là ok. Tolerances **không** đọc từ block — `atol`/`rtol` lấy từ bảng `tolerances` của cặp, chọn theo
`dimension`, nên một block không thể nới tolerance mà nó bị phán.

**Sáu kết cục reader suy ra** (`V2_STANDARD.md:79-106`) — gate chạy lại `evaluate_abs_rel` trên các cặp
đã phát, không bao giờ suy từ `max_abs`/`max_rel`/`mismatch_count`/`worst_normalized`:

| # | kết cục | verdict |
|---|---|---|
| 1 | v1 pair (không khoá `standard`) | phán dưới `parity-relative/v1`, y như cũ |
| 2 | v2 pair, mọi cặp đã phát thoả luật | check ok, `applied_standard` = v2 |
| 3 | v2 pair, một cặp vi phạm | check **FAILS**, `measured.offending_coordinates` nêu toạ độ |
| 4 | block thiếu trong khi khai `standard: v2` | cặp `_Malformed` |
| 5 | một giá trị đã phát không hữu hạn | cặp `_Malformed` |
| 6 | cặp **trộn** v1/v2 (một phía v1-by-default, một phía khai v2) | cặp `_Malformed` |

Cộng một luật nữa: block có thể nằm ở **một** phía (bình thường: adapter, phía giữ **cả hai** mảng); nếu
nằm ở **cả hai** phía thì mọi scope chung phải **giống hệt**, nếu không artifact tự mâu thuẫn.

**Test:** file mới `workbench/tests/test_parity_absolute_relative_v2.py` — **19** def, collect **27**
case (một `parametrize` 8 case cho các kiểu malformed). `23e5b8f` mang 17 def; `7df87ab` thêm **2** def
(`test_a_two_standard_evidence_set_keeps_standard_a_string`,
`test_the_aggregate_label_is_v1_when_nothing_was_judged`).

**25+ test mới** là tổng đếm được qua `--collect-only` và qua `git show` từng commit, không phải suy
từ một con số:

| nguồn | đếm |
|---|---|
| `test_parity_absolute_relative_v2.py` (mới hoàn toàn) | **27** case |
| `test_composition_api_contracts.py`, `cb0b228` | **+3** def (7 → 10) |
| `test_review_defects_evaluator.py`, `cb0b228` | **+3** def (25 → 28) |
| tổng cộng | **33** case/def mới |

`pytest test_gate_parity_subchecks.py test_parity_absolute_relative_v2.py -q` → **46 passed**
(`V2_STANDARD.md:186-189`).

Test cũ trong `test_gate_parity_subchecks.py` từng pin **bốn giá trị ĐO ĐƯỢC** từ `g1_g2.encoder.*`;
seed relation làm chúng dịch hợp pháp, nên bốn test fail vì giá trị ngẫu nhiên chứ không vì regression.
Mỗi cái được thay bằng **bất biến** mà check tồn tại để bảo vệ — không check nào bị xoá, không float nào
re-pin — kèm **mutation proof** cho cả bốn (`V2_STANDARD.md:161-189`).

**Luật tương thích kế thừa:** artifact v1 **giữ nguyên verdict v1** (cùng check, cùng tolerance, cùng
verdict; không khoá `standard`, không block, không thay đổi); một request v2 trên artifact v1 vẫn **chỉ
advisory**, `applies: False`; một nhãn v2 trần trên aggregate **vẫn là `MALFORMED`** — không bao giờ là
một kết quả v2 (`V2_STANDARD.md:140-151`).

### 6.2 Bốn lỗi backend: hai đã vá, hai đặc tả chưa làm

Nguồn: `BACKEND_DEFECTS.md` (bảng verdict ở `:9-14`).

| # | lỗi | verdict | xử lý |
|---|---|---|---|
| 1 | `_recorded_source_matches_disk` hash **một** module và so **một** module khác, nên `scoring_impl` là `unknown` cho **mọi** run | REPRODUCED — **fixed** | `workbench/backend/composition.py:689-729` |
| 2 | `served_conditions` quảng cáo một điều kiện run **chưa** đo | REPRODUCED **ở chỗ khác** — **fixed** | `composition.py:23-38`, `:40-78`, `:1593-1608` |
| 3 | `gallery_identity_sha256` có **hai** công thức dưới một khoá | REPRODUCED — **spec only** | `GALLERY_DIGEST_PATCH_SPEC.md` |
| 4 | `adapter_query_scope_resolution` đếm "không route" thành "không scope được" | REPRODUCED — **spec only** | `SCOPE_RESOLUTION_PATCH_SPEC.md` |

**Lỗi 1 — hai hash, hai module.** `scoring_module_source_hash` là hash của `workbench/composition/adapters.py`
**dùng chung**, nhưng guard so với `_module_source_hash(capability(model_id))` — module `model_<id>.py`
của chính model. Hai file khác nhau, nên phép so sai cho **mọi** artifact từng ghi. Bản vá đọc **cả**
`scoring_module_source_hash` **và** `adapter_module_source_hash`; thiếu **hoặc** `"unavailable"` ở một
trong hai → `False` (không bao giờ `live`); cả hai phải khớp. Đo: guard post-fix `True`, pre-fix
`False`, post-fix với adapter hash bị nhiễu `False` (`BACKEND_DEFECTS.md:158-179`). Run
`da0a2162be0e159a1a456cec` phục vụ `unknown` **trước và sau** — và đó là câu trả lời trung thực: nó
`adapter_module_source_hash 3b8e1bb5…` = `git show 1e37f02:workbench/composition/model_clvc_net.py |
sha256sum`, còn `a2c2ad7` đổi file đó sau (nay `7f150b6a…`), nên run đó thực sự lịch sử. Cái **hỏng**
là guard không phân biệt được: nó cũng trả `False` cho một run có byte **giống hệt** hôm nay — đúng
điều summary tổng hợp trong probe chứng minh (`True` post-fix, `False` pre-fix). Ba test qua
`get_composition_run_detail` (`test_review_defects_evaluator.py`).

**Lỗi 2 — điều kiện chưa đo được quảng cáo.** Cơ sở bị chỉ sai: `served_conditions` (dòng 1551 khi đó)
đã được `07a3d10` thu hẹp đúng. Lỗi còn sống nằm sớm hơn một hàm: `displayed_conditions` (dòng 26 ở
HEAD) **vô điều kiện** thêm `CONDITIONS` ở cấp module. Với run **có** bảng rank, filter của bảng che
nó; với run **aggregate-only** (`ranks_data is None`, nhánh fallback) triple protocol quay lại và
`image_only_empty_text` được quảng cáo dù run đo ba điều kiện khác:

```
PRE-FIX : ['multimodal','text_only_black_image','image_only_empty_text','clvc_image_encoder_only']
POST-FIX: ['multimodal','text_only_black_image','clvc_image_encoder_only']
```

Summary legacy **không** có trường `conditions` phải **không** đổi: trước và sau đều
`('multimodal','text_only_black_image','image_only_empty_text')` (`BACKEND_DEFECTS.md:239-244`). Ba test
trong `test_composition_api_contracts.py`.

**Lỗi 3 — hai công thức `gallery_identity_sha256` (spec only).** Trên cùng một release: nhánh
plan/blocked (`composition_eval.py:298-300`, payload `{protocol_id, image_ids}`) →
`0bfd43b9cc0162885f06c60cfc751602ee9f3bed2fb4d206ffcebe3f7b443462`; mọi run hoàn tất
(`runner.py:512-515`, `_gallery_fingerprint`, payload `{protocol_id, ordered_gallery, candidate_names}`)
→ `1db68c59d96f014cdbecf936e108653de98abef0bd9a98a8abc2fe08b5dec4b4`; `gallery_count 74381` ở **cả
hai**. Vì sao **chưa** làm: `workbench/scripts/composition_eval.py` và `workbench/composition/runner.py`
**không thuộc lane này** — spec là change request, không phải change. Giải pháp chọn: dùng lại
`_gallery_fingerprint` (xoá công thức thứ hai), không phải đổi tên khoá. Cả hai lựa chọn **không** phá
digest nào đã công bố vì không artifact on-disk nào chứa `0bfd43b9…` ngoài transcript audit dưới
`workbench/docs/` (historical record, không được viết lại).

**Lỗi 4 — routing tính thành failure (spec only).** Run `7b1fa85007bffdab931f7f19`: `adapter_routed 0`,
`still_unscoped 6003`, `unscoped_qids` 6003 entry, `category_scopes [null]`, `resolver null`, và mọi
`condition_status.*.unscored_count` = **0**. Nguyên nhân `runner.py:561-562` (`if None in scopes: return
None, None`) + `runner.py:2534` đếm mọi `None` scope là unscoped. `_scope_name(None) == "global"`
(`runner.py:257-258`), nên gallery nó chấm là **một** ma trận thật, đúng. Vì sao **chưa** làm: fix
đúng nằm ở **runner**, không ở `backend/composition.py`; viết lại ở backend sẽ làm payload phục vụ lệch
artifact trên đĩa và lệch `derive_routing_cohort_gap.py`, thứ replay trực tiếp artifact. File đó cũng
không thuộc lane này.

### 6.3 Lỗi `standard`-as-list phát hiện khi cặp v2 của HINT landed

`evaluate_all` nhét vào ô `standard` duy nhất:

```python
"standard": applied[0] if len(applied) == 1 else (applied or STANDARD_RELATIVE_V1)
```

nên trường là **CHUỖI** chỉ khi mọi model có bằng chứng khai cùng một standard. Khoảnh khắc standard
thứ hai xuất hiện — đúng việc ship v2 — nó thành **LIST**. Hai hậu quả sống, cả hai tái lập trước khi
vá bằng cách đặt một cặp v2 cạnh một cặp v1 trong một thư mục (`V2_STANDARD.md:195-216`):

```
standard      = ['parity-absolute-relative/v2', 'parity-relative/v1'] | list
page renders  = Chuẩn ['parity-absolute-relative/v2', 'parity-relative/v1']
```

1. `frontend/gate.ts` type trường là `string` và `EvaluationPage.tsx:71` render ``Chuẩn ${gate.standard}``,
   nên trang sống in một **array** vào câu;
2. `test_v1_remains_the_default_standard` gọi `evaluate_all()` **không** root — tức thư mục evidence
   thật — và assert `== STANDARD_RELATIVE_V1`, nên nó vỡ ngay khi artifact gắn nhãn v2 của HINT landed.
   Đó là fact về **thư mục**, không phải về default.

**Bản vá (`7df87ab`):** `standard` **LUÔN** là chuỗi, chọn tất định — standard của model **đầu tiên theo
thứ tự model-id đã sắp** mà verdict không phải `NOT_RUN`, hoặc `parity-relative/v1` khi không có gì được
phán. Thứ tự sắp, **không** thứ tự artifact đến, **không** đa số, nên nhãn không thể dịch chỉ vì một cặp
landed trừ khi cặp đó sắp đầu. Thêm trường `standards` ánh xạ mỗi standard khai báo sang các model-id đã
khai nó, mỗi model được phán xuất hiện dưới **đúng một** khoá:

```json
"standard":  "parity-relative/v1",
"standards": {"parity-absolute-relative/v2": ["hint"],
              "parity-relative/v1": ["encoder", "limn"]}
```

`models[*].applied_standard` vẫn là sự thật per-model, **không** đổi; `standard` chỉ là summary
routing/display và **không bao giờ** thay thế nó — verdict đọc từ `models`, không suy từ nhãn aggregate.

**Bằng chứng sống** (đo trong lượt viết báo cáo, `curl` tới backend đang chạy):

```
$ curl -s -o /tmp/pg.json -w '%{http_code} %{size_download}\n' http://127.0.0.1:8000/api/runs/parity-gate
200 94699
standard : 'parity-relative/v1'                                   ← str
standards: {'parity-absolute-relative/v2': ['hint'],
            'parity-relative/v1': ['encoder', 'limn']}
passed   : []
models   : encoder PARITY_MANDATORY_SUBCHECK_FAILED (v1) 9 failures
           hint    PARITY_MANDATORY_SUBCHECK_FAILED (v2) 2 failures
           limn    PARITY_MANDATORY_SUBCHECK_FAILED (v1) 3 failures
```

Frontend không cần sửa TSX: `gate.ts` đã type `standard: string`, và với backend trả chuỗi,
`Chuẩn ${gate.standard}` đúng như đang viết. Type được mở thêm hai trường nó còn thiếu để **ghi lại**
payload, không phải để sửa render.

---

## §7 — Siêu dữ liệu tài liệu đã sửa

### 7.1 Identity của bundle

| fact | giá trị | nguồn |
|---|---|---|
| bundle được trang phục vụ trỏ | `assets/index-NnjrdZLG.js` | `raw/baseline/13-served-index-html.txt` |
| `dist/assets/index-NnjrdZLG.js` | **392 769** B, sha256 **`721d34d043858e80c88e76d72866ba8583508318baeebac599cebd1e1c2c80d0`** | `raw/baseline/12-frontend-bundle.txt` |
| `dist/assets/index-DecQnta9.css` | 26 552 B, sha256 `e7e826a332ec9ab171f97051b901ee86fa69e2588937c3023914f3ac41ffe196` | same |
| served == `dist` | **có**, cùng sha256 cả JS và CSS | `BASELINE.md:130-131` |
| qua proxy? | `curl …/assets/index-NnjrdZLG.js \| sha256sum` = cùng digest | `raw/baseline/13-…txt` (`=== served asset sha ===`) |
| `dist/` tracked? | **không** — `.gitignore:95:workbench/frontend/dist/` | `BASELINE.md:132` |

`index-DDviexaW.js`, `index-DYYSN7oG.js`, `index-Lne9CV3w.js` là tên **lịch sử**, không phải hiện tại.
Chín lần nhắc tên cũ còn lại nằm trong bảy file; các sweep record `105`/`106` ghi tên của **chính**
revision chúng và được annotate, không viết lại (`PREMISES.md:715-751`).

### 7.2 Cách diễn đạt pin đổi

Pin không còn được phát biểu như "HEAD phải bằng sha X", mà là **mốc đo + bất biến**:
`FINAL_REPORT_20261112.vi.md:75-85` (thêm) và `:1388-1401` (đổi câu) — §0.1 đo tại `cd9ced2`, mọi hậu
duệ chỉ-tài-liệu có **cùng cây production** (`git diff --quiet aea143bd HEAD -- …` rỗng), và **đó** là
điều kiện để đọc số, "không phải `HEAD` bằng đúng sha nào". Dòng §9 tương ứng đổi từ một khẳng định
"số liệu sống" sang hai lần đo ở hai mốc: *"Hai lần đo hai mốc; không mốc nào là 'sha phải giữ'"*
(`git diff 40c897c HEAD` của report, khối `+20 −5`).

### 7.3 Dòng baseline của verdict

`FINAL_PC_RELEASE_VERDICT.md:24` (row "Production tree") đã nêu đúng base `aea143bd8894881ad7d5b341c21dbeff25cfaaaf`;
row này **cho người đọc**, guard **không** đọc nó — guard đọc `TESTED_SOURCE_SHA` (row trên). Đo lại
sau khi row được sửa: base đúng, nhưng bất biến không còn đúng so với `HEAD` hoặc so với đĩa, vì hai
commit production `cb0b228` và `23e5b8f` (cộng `7df87ab`) landed sau pin — xem §0 (`BASELINE.md:385-393`).

### 7.4 Sửa giá trị ENCODER ở các tài liệu trích số pre-seed

`ENCODER_RELATION.md:427-452` liệt kê **19** điểm trích, phân loại `LIVE — must be corrected` /
`Ambiguous` / `Historical record — leave` / `False positive`. Các sửa đã landed ở `243b4d2`, ví dụ
`20261109-native15-composition-limn/ENCODER_PARITY.vi.md` (`+31 −14`):

| chỗ | trước (pre-seed) | sau (seeded) |
|---|---|---|
| `§5.2` heading | "0 differences in toptee, 1 each in dress/shirt, both non-ties" | "bản ghi cũ … **nay 0 everywhere**" |
| `§5.3` native re-run dress | 21.2692 / 51.9584 / 76.3510 | **21.1205 / 51.7105 / 76.3014** |
| `§5.3` native re-run shirt | 25.3680 / 54.9558 / 75.4661 | **25.3189 / 54.9068 / 75.3189** |
| `§5.3` native re-run toptee | 29.3728 / 61.8052 / 81.0301 | **29.1178 / 62.0602 / 80.9791** |
| `§5.3` Δ range | "R@1 by up to 0.198, R@10 0.153, R@50 0.102" | "R@1 by up to **0.347**, R@10 **0.248**, R@50 **0.153**", 9/9 ô ô Recall khác bảng hit count đã lưu |
| `§5.2` probe | dress 1215/1216, shirt 1373/1372 | dress **1222/1222**, shirt **1372/1372** |
| `§5.4` transform table | R@10 `51.9584 / 54.9558 / 61.9582` | **51.7105 / 54.9068 / 62.0602** |

Macro seeded `R@10 56.225801935399 / R@50 77.533156738705` được ghi kèm. Các ghi chú lịch sử
(`raw/native/encoder.json`, log pre-fix, `relation_proof_native_pair.txt`) **giữ nguyên** — chúng là
bằng chứng *đã thiết lập* hai draw độc lập. `habitat.json:15,22` = **false positive** (HABIT, checkpoint
và protocol khác, trùng số ở r1).

### 7.5 Dựng lại bảng digest của BASELINE

`BASELINE.md:298-320` dựng lại từ `raw/baseline/19-annotation-sha256.txt` — **21** digest, **21** path,
không elision, không marker `+N more`. Hai cặp bị **dịch** và ba dòng **placeholder** được điền:

| ô | trước (sai) | sau (đúng) |
|---|---|---|
| `cap.shirt.train.json` | `1552c05d…98b9983` | `0999ce5da01a38f0cef1547941b738f4c56e225fdd9b8e03e2a6edaf292f94c3` |
| `split.shirt.train.json` | `1552c05d…` prefix collision note | `1552c05d866d31b43321a719b2cb8a65f02c53e9a6bc355861fe1d62498b9983` (raw:18) |
| `split.dress.train.json` | `0999ce5d…292f94c3` | `dcb1c7329003e85e39a3010147a620cee4b7de6a5ddf2626f253bc9a2cdd78e5` (raw:15) |
| `cap.toptee.test.json` | "— not in captured head" | `db822aa52340ce1f09c3000064f79b60b6c476b50b98bf34200dd2996c06e146` (raw:8) |
| `split.toptee.val.json` | "— see raw file" | `ee42b2505275dd7a19b11ddb88c9264aab2d77f4e59b075fb1c7c3095d02f412` (raw:22) |
| `cap.toptee.val.json` | placeholder | `b4e09f428e6c255cc4b4ec46e77908305ab83075d82840c960be81af15cdc5eb` (raw:10, không đổi) |

**Nguyên nhân dịch — một khẳng định đã bị bác bỏ:** đoạn văn cũ nói "transcript rendering interleaved
some long paths". Sai: file raw **không** interleaved; mỗi dòng là `<digest>  data/FashionIQ/<dir>/<file>`,
nên mọi digest và path đọc ra đều rõ ràng. Chính ghi chú sai đó sinh ra các cặp bị dịch, và **không
được trích lại** (`BASELINE.md:317-320`). Sizes cũng bị làm thô (`1.1M` cho ba file 1.18M–1.20M); nay là
byte chính xác.

### 7.6 Digest routing KHÔNG bị ảnh hưởng

Ba digest của caption routing được xác nhận **trên đĩa** trong lượt viết báo cáo và có trong mọi
artifact downstream:

```
$ sha256sum data/FashionIQ/captions/cap.{dress,shirt,toptee}.val.json
5e5117d45695df9c3ca91fac3e0bc49e6422ae9c8b3793f0ac9b3ab83adb7de9  cap.dress.val.json
7b7ca3797b85dddfd83e74cdb4454cb63e3e7c09acef2590155efbacea6d7feb  cap.shirt.val.json
b4e09f428e6c255cc4b4ec46e77908305ab83075d82840c960be81af15cdc5eb  cap.toptee.val.json
```

Xuất hiện lại ở **19** file dưới `workbench/` (`grep -rl "5e5117d45695df9c" workbench/`). Phân bố đo
được, và điều đáng chú ý là **một trong ba chỗ ngoài `docs/reviews` là CODE**:

| khu vực | số file | vai trò |
|---|---|---|
| `workbench/docs/reviews/` | 15 | record/transcript — ghi lại phép đo, không thực thi |
| `workbench/composition/__pycache__/` | 2 | bytecode dẫn xuất, không phải nguồn |
| **`workbench/composition/data.py`** | **1** | **mã nguồn**: `:27` khai `"cap.dress.val.json": {"sha256": "5e5117d45695df9c…"}` như **kỳ vọng được cưỡng chế** — digest là hằng số kiểm tra trong code, không chỉ là con số trong báo cáo |
| `workbench/artifacts/composition/derived/…/clvc_net_da0a2162…_full.json` | 1 | artifact derived của run |
| `workbench/artifacts/checkpoint_selection/20261106T002617+0700/manifests/protocol_identity.json` | 1 | manifest protocol |

Vì `workbench/composition/data.py:27` giữ digest này như một hằng số được kiểm, con số
`cap.dress.val 5e5117d4…` không chỉ "có trong báo cáo" mà **được cưỡng chế lúc chạy**; và bảng digest
dựng lại ở §7.5 **không** chạm tới nó. Cùng kết luận cho `cap.shirt.val 7b7ca379…` và
`cap.toptee.val b4e09f42…` — cả ba đọc từ đĩa khớp từng byte.

---

## §8 — Kiểm chứng trên bề mặt sống

**Phép kiểm UI được chạy trong một browser thật** — Playwright + `channel="chrome"`, headless,
viewport 1920×1080, qua `vite preview` ở `:5173`. Script: `scripts/live_probe.py`; output
`raw/live/report.json` + 6 ảnh chụp. Đây là probe **dev-only, không ship**.

### 8.1 Identity bundle + service

| fact | giá trị |
|---|---|
| URL | `http://127.0.0.1:5173` |
| shell status | **200**; scripts `['/assets/index-NnjrdZLG.js']`; stylesheets `['/assets/index-DecQnta9.css']` |
| bundle identity | sha256 `721d34d043858e80c88e76d72866ba8583508318baeebac599cebd1e1c2c80d0`, 392 769 B — khớp `dist/` và bản qua proxy |

PID/timestamp: `BASELINE.md:110-113` ghi backend `MainPID=1758096`, `ActiveEnterTimestamp=Thu
2026-11-12 12:20:57 +07`, `NRestarts=0`; frontend `MainPID=1221148`, `Wed 2026-11-11 23:48:44 +07`,
`NRestarts=0`. **Đo lại lúc viết báo cáo hai unit đã được restart** (để nạp `dist` mới, mtime
`2026-11-12 23:22:36`): backend `MainPID=2162645`, frontend `MainPID=2162647`, cả hai
`ActiveEnterTimestamp=Thu 2026-11-12 23:22:41 +07`, `NRestarts=0`, `ActiveState=active`. Không
restart nào do lượt viết báo cáo này thực hiện; đây là chép trạng thái đo được.

### 8.2 Endpoint

| endpoint | HTTP / bytes |
|---|---|
| `http://127.0.0.1:8000/api/models` | `200 86658` |
| `http://127.0.0.1:8000/api/benchmark-runs` | `200 54761` |
| `http://127.0.0.1:8000/api/runs/parity-gate` | `200 94699` (đo trong lượt viết) / `200 65460` (`BASELINE.md:119`, trước khi `standard`+`standards` landed) |
| `http://127.0.0.1:5173/` | `200 271` |

`standard` là **chuỗi** `'parity-relative/v1'`, `standards` breakdown
`{'parity-absolute-relative/v2': ['hint'], 'parity-relative/v1': ['encoder','limn']}`, `passed_models` =
`[]` (§6.3).

### 8.3 UI: gate, điều kiện, query

| điều claim | đo được |
|---|---|
| header render standard như **chuỗi** | `contains_Chuẩn_parity-relative/v1: true`, `contains_bracket_list: **false**`; subtitle `Chuẩn parity-relative/v1 · tính từ artifact đo, không đọc từ văn bản` |
| badge verdict | `KHÔNG mô hình nào PASS (0/3)` |
| summary | `Không mô hình nào đạt: 3/3 mô hình có subcheck bắt buộc FAIL. Đây là kết luận từ artifact, không phải lỗi tải.` |
| ba dòng model | `encoder` 9 subcheck, `hint` 2, `limn` 3 — tất cả `data-passed="false"` |
| **HINT được khám phá** | có mặt trên trang (ba dòng, `parity-row-hint`), không phải chỉ trong log |
| `conditions_display` đúng **ba** điều kiện | `['multimodal','text_only_black_image','clvc_image_encoder_only']`, nhãn `['Image + text','Text only','Image only (image encoder)']`; `image_only_empty_text` **không** có |
| run được chọn | `active-run` = `c:da0a2162be0e159a1a456cec` |
| `7:2179` | title `Query 7:2179`; caption `Has writing on the t-shirt and is gray with black lettering.`; multimodal **#1,580** · text-only **#4,549** · image-only (image encoder) **#2,100**; outcome `unresolved` |
| console errors | `[]` → **0** |
| page errors | **0** (listener `pageerror` không ghi gì; `report.json` chỉ có `console_errors: []`) |
| failed requests | **0** (listener `requestfailed` ghi cùng sink; sink rỗng) |

Ba hạng 1580/4549/2100 khớp API: `GET /api/composition/runs/da0a2162…/queries?…&query_id=7:2179`
trả `target_rank 2100`, `ranks {multimodal: 1580, text_only_black_image: 4549,
clvc_image_encoder_only: 2100}`, `outcome: 'unresolved'`, `outcome_counts {'unresolved': 1}` (đo trong
lượt viết báo cáo). Nhãn UI cho `unresolved` là `Unresolved — mọi hạng ngoài @10`
(`workbench/frontend/routes.ts:114-124`), cố ý khác chữ trần `unresolved` để không đọc thành "unscored".

**Ảnh chụp:**

- `raw/live/01-shell.png`
- `raw/live/02-parity-card.png`
- `raw/live/02-parity-page.png`
- `raw/live/03-samples-filters.png`
- `raw/live/04-query-7-2179-inspector.png`
- `raw/live/04-query-7-2179-page.png`

---

## §9 — Bảng tổng hợp số lượng

Mỗi dòng đếm **độc lập** từ bảng/lệnh nêu kèm; **không** dòng nào suy từ dòng khác.

| đại lượng | giá trị | nguồn / danh sách |
|---|---|---|
| cấu hình đăng ký | **17** | `yaml.safe_load(models.yaml)['models']`; `MODEL_MATRIX.md:422` |
| dòng ma trận | **18** | 17 + `clip4cir_rn50x4_noft`; đếm lại `MODEL_MATRIX.csv` |
| **đánh giá native FashionIQ HỢP LỆ** (aggregate đủ điều kiện protocol, độc lập parity) | **14** | airknow, clip4cir_rn50x4_fullft\*, clvc_net, conesep, csmcir†, dcnet, encoder, hint, intent, limn, sprc†, tgcir, tme |
| …loại trừ, nêu tên | **4** | `ptha_mtst` (checkpoint CIRR → transfer, không phải FashionIQ reproduction), `combiner_rn50x4_noft` (roll-up-only, `METRIC_REFERENCE_UNVERIFIED`), `clip4cir_rn50x4_noft` (roll-up/unbacked, `counted=false`), `pair` (`EXPLORATORY_PUBLIC_MAPPING`) |
| **native parity-verified** | **0** | `passed_models: []` |
| native đã thực thi nhưng chưa verified | **16** | `MODEL_MATRIX.md:430` |
| native bị chặn | **1** | `pair` |
| CIRCUS composition run được tham chiếu | **33** all · **30** non-synthetic non-audit · **24** cohort 6003 | 36 dir `full.json`; `MODEL_MATRIX.md:434` |
| **parity đã đo** (≥1 bộ `g1_g2` cặp) | **3** | encoder (v1), limn (v1), hint (v2) |
| **parity PASS** | **0** | `MODEL_MATRIX.md:436` |
| parity blocked/thiếu artifact | **14** | 17 − 3 = 14 id còn lại |
| G1 PASS | **3** | encoder (0/30), hint (0/36, v2), limn (0/18) |
| G2 PASS | **0** | encoder 9 failure, hint 2, limn 3 |
| test đã collect (toàn suite, cây hiện tại) | **3226** | `pytest workbench/tests --collect-only -q` |
| test đã collect tại pin `aea143bd` | **3193** | `107-sweep-aea143bd.md` §1 (`--collect-only -q`) |
| sweep tại pin: passed / failed / skipped | **3126 / 20 / 47 = 3193**, wall **3684.01 s (1:01:24)**, `^ERROR` 0, `Errno 28` 0 | `107-sweep-aea143bd.md` §2 |
| sweep: diff tập failure so sweep trước (`117fc172`) | **0 mới, 0 rơi, 20 y nguyên** | `107-sweep-aea143bd.md` §3 |
| sweep tại **chính chiến dịch này** (cây sau 3 commit production) | **`cf6ba468`**: 3159 passed / 20 failed / 47 skipped = **3226** collected, wall **4037.65 s (1:07:17)**, `^ERROR` 0, `Errno 28` 0; diff tập failure so sweep `aea143bd`: **0 mới, 0 rơi, 20 y nguyên** | `SWEEP_cf6ba468.md` §2-§3 (log `/tmp/sweep_cf6ba468.txt`) |
| pin hiện tại | **`cf6ba468af5b3a6f47174bdd0d50866f23b7006f`** — nâng sau một sweep stationary có tập failure **trùng khít** tập của pin trước (`aea143bd`: 0 mới, 0 rơi, 20 y nguyên) | `FINAL_PC_RELEASE_VERDICT.md` / `PC_PILOT_HANDOFF.md`, `TESTED_SOURCE_SHA` |
| test mới cho v2 | **25+** | 19 def / 27 case trong `test_parity_absolute_relative_v2.py`; +3 backend; +2 `standard`-as-string; `V2_STANDARD.md:186-189` → **46 passed** cho hai file |
| test `test_composition_api_contracts.py` | **10** passed (7 trước `cb0b228`) | `BACKEND_DEFECTS.md:293` |
| test `test_review_defects_evaluator.py` | **28** passed (25 trước) | `BACKEND_DEFECTS.md:294` |
| 14 file composition-related | **143** passed, 0 failed | `BACKEND_DEFECTS.md:297` |
| `test_api_outcome_triple.py` / `test_derived_subset_status.py` | **9** / **5** passed | `BACKEND_DEFECTS.md:295-296` |
| `test_handoff_commands.py` (đo trong lượt này) | **1 failed, 7 passed**, 16.41 s | `pytest … --basetemp=/tmp/pin_cf6` |

**Hai mục đỏ-dự-kiến (expected-red)** — cả hai là guard siêu-dữ-liệu release, không phải regression
logic; chỉ mục 1 đỏ trong lượt đo này:

1. `test_handoff_commands.py::test_no_release_document_points_at_a_superseded_source_sha` — **ĐỎ, đo
   được**: `1 failed, 7 passed in 16.41s`. Lý do: nó assert `TESTED_SOURCE_SHA` chứa **mọi** byte
   production trên đĩa, và ba commit production của chiến dịch (`cb0b228`, `23e5b8f`, `7df87ab`) landed
   sau `aea143bd`. Đó là **lỗi của pin**, không phải của code: với `workbench/backend/composition.py`
   được revert, file này pass **8/8** (`BACKEND_DEFECTS.md:289-304`). Trong sweep **tại pin**, cùng id
   này cũng đỏ có chủ ý vì sweep chạy **trước** khi nâng pin, rồi xanh lại sau khi nâng (`107 §3`;
   `107 §5`: `8 passed in 14.01s`).
2. `test_handoff_commands.py::test_the_release_documents_quote_the_test_count_they_actually_ran` và
   `::test_the_verdict_names_the_same_test_count_the_handoff_does` — **XANH trong lượt này** (nằm trong
   `7 passed`), và đỏ-dự-kiến chỉ khi số `passed/failed` trong release document không khớp số test mà
   **revision được nêu tên** thật sự collect. Docstring của chính file ghi rõ lớp này: *"the
   release-metadata guards are part of the suite and they FAIL on the very revision they describe, so a
   document that can only record zero failures cannot describe its own run honestly"*
   (`workbench/tests/test_handoff_commands.py:666-668`). Chúng xanh sau khi nâng pin sang
   `cf6ba468`, vì document nay ghi đúng `3159 passed, 20 failed, 47 skipped` của chính revision đó;
   **advance pin mà không cập nhật số sẽ làm chúng đỏ**.

**Hai con số khác nhau, đừng đọc cái này như cái kia** (`MODEL_MATRIX.md:517-532`):

- *"Có một đánh giá native FashionIQ không?"* → **14**. Đếm này **độc lập parity**: chỉ hỏi protocol,
  loader, gallery và category scope đã verified và có aggregate ở mức bằng chứng đó chưa.
- *"Có cấu hình nào parity-verified không?"* → **0**. Câu hỏi chặt hơn: native **và** một run Workbench
  trên cùng input, khớp theo gate với 0 mandatory failure. Ba model có cặp (9, 3 và 2 failure G2).

---

## §10 — Lệnh tái lập

Chép từ mục "Commands used" của các record. Không thêm flag nào không có trong đó.

### 10.1 Gate (thẩm quyền G1/G2)

```bash
cd /home/heheboiz/data/cir

# khám phá + verdict aggregate + standard mỗi cặp được phán dưới
python3 -c "
import sys; sys.path.insert(0,'.')
from workbench.backend import parity_gate as g
print(g._discover_models(g.evidence_root()))
r=g.evaluate_all(); print(r['standard'], r.get('standards'), r['passed_models'])
[print(m, v['status'], v['failure_count'], v['missing']) for m,v in r['models'].items()]"

# mọi check bắt buộc FAIL, kèm tên và giá trị đo/khai báo
python3 -c "
import sys;sys.path.insert(0,'.');from workbench.backend import parity_gate as g
[print(m,g.evaluate_model(m).status) or
 [print(' ',c.gate,c.name,c.measured,c.declared) for c in g.evaluate_model(m).checks
  if c.mandatory and not c.ok] for m in g._discover_models(g.evidence_root())]"
```

Nguồn: `MODEL_MATRIX.md:626-636`.

### 10.2 Tái lập CLVC-Net (cần GPU)

```bash
# baseline batch 32 + probe độ nhạy batch 1 (deliverable)
MAX_WAIT=3600 setsid nohup workbench/scripts/gpu_job.sh \
  workbench/envs/ctl312h/bin/python \
  workbench/docs/reviews/20261112-parity-repair/scripts/clvc_native_ranks.py \
  --batch-size 32 > /tmp/clvc_native.log 2>&1 < /dev/null &

# control cho verdict: batch 48 (run1) và một repeat batch 32 (determinism)
MAX_WAIT=3600 workbench/scripts/gpu_job.sh \
  workbench/envs/ctl312h/bin/python \
  workbench/docs/reviews/20261112-parity-repair/scripts/clvc_native_ranks.py \
  --summarise-only --extra-pass 48 --extra-pass 32

# merge thư mục pass đã có mà không chạy gì
workbench/envs/ctl312h/bin/python \
  workbench/docs/reviews/20261112-parity-repair/scripts/clvc_native_ranks.py --summarise-only
```

Nguồn: `CLVC_NATIVE.md:225-243`.

### 10.3 Producer parity (ENCODER / LIMN / HINT)

```bash
# ENCODER: regenerate cặp G1/G2 (GPU bắt buộc — BindingDecoder.__init__ gọi .cuda())
ENCODER_PARITY_RAW=/tmp/encparity_raw workbench/scripts/gpu_job.sh \
  <model interpreter> \
  workbench/docs/reviews/20261109-native15-composition-limn/scripts/encoder_parity.py

# LIMN: block metrics_parity sinh lại từ hit counts
python3 workbench/docs/reviews/20261112-parity-repair/scripts/limn_hits.py
python3 workbench/docs/reviews/20261112-parity-repair/scripts/limn_hits.py --negative

# HINT: native rồi adapter, đúng interpreter mỗi pha
env OMP_NUM_THREADS=2 ... workbench/scripts/gpu_job.sh \
  workbench/envs/ilearn39-adapted/bin/python \
  workbench/docs/reviews/20261112-parity-repair/scripts/third_model_parity.py --phase native
env OMP_NUM_THREADS=2 ... WORKBENCH_COMPOSITION_BATCH_SIZE=32 workbench/scripts/gpu_job.sh \
  workbench/envs/ctl312h/bin/python \
  workbench/docs/reviews/20261112-parity-repair/scripts/third_model_parity.py --phase adapter
```

Nguồn: `ENCODER_RELATION.md:536-546` (điều kiện chạy), `THIRD_MODEL_PARITY.md:303-311`,
`scripts/limn_hits.py` (chế độ `--negative`). Native **phải** là interpreter của model
(`transformers==4.25.0`); adapter **phải** là controller.

### 10.4 Sweep

```bash
setsid nohup workbench/envs/ctl312h/bin/python -m pytest workbench/tests \
  -q -p no:cacheprovider -rfE --tb=short --basetemp=/tmp/sweep_basetemp_cf6ba468 \
  > /tmp/sweep_cf6ba468.txt 2>&1 < /dev/null &
grep -c '^ERROR ' /tmp/sweep_cf6ba468.txt
grep -c "Errno 28" /tmp/sweep_cf6ba468.txt
```

Mẫu lệnh + cách kiểm accounting: `107-sweep-aea143bd.md` §2 (basetemp là thứ duy nhất đổi tên).

### 10.5 Probe sống

```bash
curl -s -o /dev/null -w '%{http_code} %{size_download}\n' http://127.0.0.1:8000/api/models
curl -s -o /dev/null -w '%{http_code} %{size_download}\n' http://127.0.0.1:8000/api/benchmark-runs
curl -s -o /dev/null -w '%{http_code} %{size_download}\n' http://127.0.0.1:8000/api/runs/parity-gate
curl -s -o /dev/null -w '%{http_code} %{size_download}\n' http://127.0.0.1:5173/
curl -s http://127.0.0.1:5173/ | grep -o 'assets/index-[^"]*\.js'
sha256sum workbench/frontend/dist/assets/index-NnjrdZLG.js
curl -s http://127.0.0.1:5173/assets/index-NnjrdZLG.js | sha256sum

# UI: Playwright + chrome qua vite preview
python3 workbench/docs/reviews/20261112-parity-repair/scripts/live_probe.py
```

### 10.6 Lỗi backend (reproduction, không GPU)

```bash
# lưu script trong BACKEND_DEFECTS.md §"Reproduction" vào /tmp/delta.py rồi:
PYTHONPATH=$PWD workbench/envs/ctl312h/bin/python /tmp/delta.py
```

Nguồn: `BACKEND_DEFECTS.md:22-30` (in ra bốn khối A/B/C/D đã ghi ở `:158-179`).

---

## §11 — Còn chặn và điều KHÔNG được làm

### 11.1 Còn chặn, mỗi mục có blocker tái lập được

| # | mục | blocker tái lập được |
|---|---|---|
| 1 | **PAIR** (`BLOCKED_MISSING_SOURCE_CLASSES` / registry `BLOCKED_CHECKPOINT_MAPPING_UNVERIFIED`) | `raw/pair_absence_evidence.json:3-21`: `model_vssp.{Encoder,Backbone,FeatureWiseAffine}` + `MHTransformer_dis.{Transformer,TransformerLayer,MultiHeadAttention,MlpTransformer}` đều `false`; full clone upstream 21 commit, `git grep` mọi rev → **0** hit. 5.20 % đo 8986448/172814185 = 5.2001 % **so với một lớp giả định** |
| 2 | **2 query nhập nhằng** `7:3038`, `7:3790` | `raw/m02/m02_recount.json:288-297` (119/117/2; shirt 46, toptee 71); release không có khoá category; mọi luật phân xử đều phá luật "chỉ gán từ đúng một category được nêu". Cần metadata upstream, không phải vá code |
| 3 | **`image_only_empty_text` không hỗ trợ cho CLVC** | `third_party/CLVCNet/text_model.py:84-86` index `x[0][0]` sau khi tokenise empty → `IndexError` cho **mọi** hàng (`model_clvc_net.py:418-435`, `EMPTY_TEXT_REASON` ở `:96`). Ghi `ABLATION_UNSUPPORTED`, không fake |
| 4 | **15 id không có cặp parity** | `_discover_models` trả **3** id; `INVENTORY.csv` `has_parity_artifact` = `Y` chỉ cho encoder/limn. 17 − 2 = 15 lúc audit; nay 17 − 3 = 14 |
| 5 | **công thức gallery-digest kép** | `0bfd43b9…` (plan/blocked) vs `1db68c59…` (run hoàn tất), `gallery_count 74381` cả hai — spec ở `GALLERY_DIGEST_PATCH_SPEC.md`, file sở hữu nằm ngoài lane |
| 6 | **trường scope-resolution** | `7b1fa850…`: `adapter_routed 0`, `still_unscoped 6003`, `condition_status.*.unscored_count 0` — mâu thuẫn nội tại; fix ở `runner.py:2528-2537`, ngoài lane (`SCOPE_RESOLUTION_PATCH_SPEC.md`) |
| 7 | **`reference_metadata_seen` chưa có trong run identity** | `grep -rn reference_metadata_seen workbench/composition workbench/backend workbench/scripts workbench/tests` → **0** hit; chỉ có trong `raw/verify/99-clvc-scope-and-image-only.md:125,314`. Chưa ship vì nó mint run id mới cho mọi run hiện có |
| 8 | **20 test fail pre-existing** | `107-sweep-aea143bd.md` §3: **18** id baseline (blip ×4, `circus_cli`, `tme_lifecycle` ×2, `tme_precision`, `f08_dcnet`, `handoff_commands` SHA guard, `release_blocker_gpu_records` ×7, `prepare_model`) + **2** id chưa từng trong baseline 18 (`test_doctor.py::test_dcnet_runtime_asset_blocks_after_preparation_is_present`, `test_evaluate_models.py::test_dcnet_audited_command_reports_non_command_blockers`). Ba trong số đó phụ thuộc trạng thái host nên pass trong worktree trần và fail trong cây thật (`106` §6) — ghi nhận, không sửa |
| 9 | **người gửi SIGTERM lịch sử** | `UNDETERMINED`. Đường nhận đã biết (`serve_workbench.py:152-166`), `jobs.py` bị loại, test/script trong repo bị loại, session reaping bị loại; bốn chữ ký chết đã commit (`raw/service-death-log{,-2,-3,-4}.txt`). *"That inference is for a future investigation, not evidence about the past"* |
| 10 | **hành vi `systemctl stop`** | `NOT_COMPLETED` — ca `TimeoutStopSec=30` worst-case (job sống phớt lờ SIGTERM hết 10 s grace) **chưa đo**, chỉ bị chặn bằng số học (`GRACE_SECONDS` 10 + 5 SIGKILL + 10 join ≤ 30) |
| 11 | **batch size nào canonical cho CLVC** | `UNDETERMINED` — không upstream pinned lẫn registry chỉ định; `test.py` chỉ đọc `params.batch_size` |
| 12 | **hai aggregate native ENCODER cùng tồn tại, registry chỉ biết một** | `ENCODER_RELATION.md:401-425`; lớp `TWO_NATIVE_MACROS` chưa được gắn cho `encoder` trong ma trận |

### 11.2 Những thứ KHÔNG được "sửa"

| thứ | vì sao không được |
|---|---|
| `COMMAND_BLOCKED_SOURCE_ENTRYPOINT` (CLVC `clvc_net`, TG-CIR) | Nó ghi đúng sự thật: entrypoint nguồn không chạy được từ repo. Đổi nó thành "chạy được" là đổi nhãn bằng chứng, không phải sửa lỗi |
| 2 query `7:3038` / `7:3790` | Mọi luật phân xử đều **phá** luật "chỉ gán từ đúng một category được nêu". Cần metadata upstream, không phải code |
| `image_only_empty_text` | `ABLATION_UNSUPPORTED` là kết quả đo của một `IndexError` xác định. "Sửa" nó = fake một phép đo |
| `--batch-size` > 32 cho một measurement sẽ báo cáo | Batch 1 vs 32 dịch **1548/2017** rank dress. `--batch-size` là một phần measurement identity |
| R@1 đọc như **0** | Rank của CLVC nằm ngoài @10 nhưng **có** giá trị (1580/4549/2100). Đọc "0" = đọc `unscored` như `unresolved`; UI đã cố ý đặt nhãn `Unresolved — mọi hạng ngoài @10` |
| dời bất kỳ tolerance nào để biến một `max_rel` failure thành xanh | 9 failure encoder, 3 limn, 2 hint đều là **denominator-unstable near-zero reference**. `max_abs` ≤ 1.65e-06 và `min_cosine ≥ 0.9999999999`. Dời ngưỡng sau khi đọc số là moving the measurement to fit the result — đổi định nghĩa đo phải là quyết định hiển ngôn có record riêng |

---
