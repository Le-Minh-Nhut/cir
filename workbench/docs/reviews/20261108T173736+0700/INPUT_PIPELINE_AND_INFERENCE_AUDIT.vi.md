# KIỂM TOÁN INPUT PIPELINE VÀ INFERENCE — báo cáo tổng hợp

Nhánh `research/cir-failure-workbench`. Không commit, không push, không đụng git state.
Đây là báo cáo **tổng hợp lại bằng chứng đã có**, không phải điều tra mới: không chạy experiment mới,
không chạy GPU, không sửa file nào ngoài thư mục này.

Thư mục bằng chứng gốc: xem `EVIDENCE_INDEX.md` cùng thư mục. Ma trận gate: `GATE_MATRIX.md`.

**Thẻ dùng trong toàn báo cáo:**

| thẻ | nghĩa |
|---|---|
| **quan sát được (observed)** | có phép đo hoặc file:line trong bằng chứng gốc; số liệu chép nguyên, không làm tròn |
| **suy luận (inference)** | đọc code/shape/control-flow mà suy ra, chưa có phép đo trực tiếp |
| **câu hỏi chưa giải quyết (unresolved)** | bằng chứng hiện có không quyết định được |

Không có mục nào trong báo cáo này nói "cả 16 model đều đúng". Mỗi kết luận đi kèm phạm vi kiểm
chứng cụ thể (số case đã đo / chưa đo), và các gate chưa đo được ghi đúng là `NOT_RUN`.

---

## 1. Kê khai repository / model thực tế

### 1.1 Hai con số, và con số thứ ba

| đại lượng | giá trị | cách đếm | trạng thái |
|---|---|---|---|
| repository upstream duy nhất | **16** | `upstream_repo_url` khác nhau trên 17 entry registry; bằng đúng 16 thư mục dưới `workbench/third_party/`, mỗi thư mục một `provenance.json` | **quan sát được** |
| biến thể đăng ký (`model_id`) | **17** | `len(models)` trong `workbench/registry/models.yaml` | **quan sát được** |
| biến thể checkpoint khai báo | **29** | một dòng cho mỗi phần tử `checkpoint_variants[]` | **quan sát được** |
| biến thể checkpoint staged | **21 / 29** | `registry.checkpoint_path(model_id, variant).exists()` với `WORKBENCH_CHECKPOINT_ROOT` hiện tại | **quan sát được**, tái lập độc lập trong lượt viết báo cáo này |

Con số **29 là checkpoint, không phải repository**. Ai đọc "17" thành "17 thứ cần evaluate" sẽ
đếm thiếu 12 file trọng số; ai đọc "29" thành "29 repository" sẽ đếm thừa 13. Cả hai đều đã từng
xảy ra trong tài liệu trước.

**Tái lập độc lập (observed):** chạy lại resolver trên registry hiện tại cho đúng 21 present / 8
absent, và 8 cái absent là `conesep/fiq_n05`, `conesep/fiq_n08`, `habit/fiq_n05`, `habit/fiq_n08`,
`intent/fiq_n05`, `intent/fiq_n08`, `sprc/sprc_fiq_unresolved`, `tme/cirr_0` — khớp `REPOSITORY_INVENTORY.md` §4.1.

### 1.2 CLIP4Cir: một repository, hai biến thể — nêu tường minh

`combiner_rn50x4_noft` và `clip4cir_rn50x4_fullft` cùng trỏ `ABaldrati/CLIP4Cir`, cùng
`source_dir: CLIP4Cir`, cùng một `provenance.json` có `model_ids = ["combiner_rn50x4_noft",
"clip4cir_rn50x4_fullft"]`, và cùng một biến môi trường `WORKBENCH_CLIP4CIR_PYTHON`. **quan sát được**

Khác nhau ở artifact và ở cách nạp:

| biến thể | artifact | bytes | sha256 | nạp |
|---|---|---|---|---|
| `combiner_rn50x4_noft` | một file `fiq_comb_RN50x4_noft.pt` | 236,017,921 | `b31882fe6f7e3e19…` | chỉ `Combiner` trên CLIP gốc |
| `clip4cir_rn50x4_fullft` | bundle 2 file: `fiq_clip_RN50x4_fullft.pt` (713,874,695) + `fiq_comb_RN50x4_fullft.pt` (236,017,921) | 949,892,616 | `4111ac139cf96ac0…` + `7c7c0a10fb4c995c…` | `Combiner` **và** CLIP đã fine-tune |

Cùng một `CLIP4Cir/src/validate.py` phục vụ cả hai; chế độ do `--clip-model-path` quyết định
(`PROTOCOL_MATRIX.md` §Group A / CLIP4Cir). **quan sát được**

### 1.3 29 biến thể checkpoint không phải repository

Chúng là file trọng số. Ba cái là artifact nhiều member (`clvc_net` 6 file, `dcnet` run directory 3
member khai báo + một `val/` không khai báo, `clip4cir_rn50x4_fullft` 2 file). **quan sát được**

### 1.4 Hai checkout không phải model

| đường dẫn | remote | HEAD | vai trò |
|---|---|---|---|
| `teacher/repos/SPN4CIR` | `BUAADreamer/SPN4CIR` | `84005f643ccaacf999982694ad5631df92cef098` | không có registry entry, không adapter, không provenance; được tham chiếu bởi đúng một file (`teacher/outputs/full_audit/tgcir/full_metrics.json`) |
| `data/cirr` | `Cuberick-Orion/CIRR` | `2276fba2d2c53862c8ae79ea18e1f1ae03719f56` | dataset repo, không phải model |

⇒ 16 + 2 = **18 source tree trên đĩa**, trong đó 16 thuộc phạm vi workbench. Con số 16/17 là
**registry count**, phải gọi đúng tên như vậy. **quan sát được**

### 1.5 Năm repository có `.git` lồng trong `teacher/repos/`, mười hai thì không

`teacher/repos/` có 5 checkout (`CSMCIR`, `AAAI25-ENCODER`, `TME`, `SPRC`, `SPN4CIR`) cộng `data/cirr`.
`workbench/third_party/*` **không** có `.git` nào — vendor step đã bỏ và giữ `provenance.json` gồm
`upstream_commit` + `file_manifest` sha256 từng file. **quan sát được**

Hệ quả: với 12 repo vendor-only, ta verify được **nội dung** (re-hash 100% manifest) nhưng không
chứng minh được `upstream_commit` trỏ đúng commit trên remote. **suy luận**

### 1.6 Bất đối TME — pin registry không tồn tại trong clone teacher

| trường | giá trị | trạng thái |
|---|---|---|
| registry `upstream_repo_url` | `https://github.com/li-shuxian/TME` | observed (registry) |
| registry `upstream_commit_sha` | `394aad79df2c104feb32a2c96132582012765062` | observed (registry) |
| `third_party/TME/provenance.json` | cùng URL, cùng commit, vendored 2026-10-23 | observed |
| `teacher/repos/TME` remote | `https://github.com/He-Changhao/2025-CVPR-TME.git` | **quan sát được lại trong lượt này** |
| `teacher/repos/TME` HEAD | `7fe811992820d30828e24065eb0c7a7ba099dd3b` (2025-06-09) | **quan sát được lại** |
| `git -C teacher/repos/TME cat-file -t 394aad79…` | `fatal: git cat-file: could not get object info` | **quan sát được lại** |

Vendored tree khớp manifest riêng 349/351 file; 2 file lệch đúng là 2 patch đã duyệt. Nên **execution
không bị ảnh hưởng**; cái sai là mọi câu "TME checkout đang ở commit pin" nếu nói về `teacher/repos/TME`.
Hai tree cũng khác nhau ở `README.md` và `precompute.sh` (fork trỏ FashionIQ, bản gốc trỏ CIRR, cộng
một job `nohup` bị bỏ) — nên đây không phải re-clone cùng nội dung ở commit khác mà là **một fork
khác chủ**. **quan sát được** (diff nằm ở `20261108T155252-protocol-audit/raw/vendored_diffs.txt`)

### 1.7 CSMCIR: file trong teacher tree ở trạng thái thứ ba

| locus | sha256 | trạng thái |
|---|---|---|
| `base_sha256` khai báo | `9d8c914b551a7de614e062f9a241f44626fd591a31a0ccf1cc79f0ae23c9db01` | upstream như công bố |
| `derived_sha256` khai báo | `0ec134431697af6f202b5256e4e30ab725a44191a279e3f8bfdb3a62495b3774` | sau patch đã duyệt |
| `workbench/third_party/CSMCIR/src/lavis/models/__init__.py` | `0ec134431697af6f…` | **derived — đúng** |
| `teacher/repos/CSMCIR/src/lavis/models/__init__.py` (working tree) | `8e9cd6a8303bc0c6…` | **không phải base, không phải derived — trạng thái thứ ba** |
| `teacher/repos/CSMCIR` @ HEAD (`git show`) | `9d8c914b551a7de6…` | base — HEAD sạch, lệch là do sửa chưa commit (102+/102−, 236 entry dirty) |

**quan sát được, tự re-hash cả ba trong lượt này.** Adapter launch `third_party`, nên evaluation hôm
nay không đọc file đó; nhưng bất kỳ run nào trỏ vào `teacher/repos/CSMCIR` sẽ chạy một biến thể
**chưa được review**. Cùng pattern nhưng nhỏ hơn: `validate_blip_csmcir.py` trong teacher tree ở
**base** trong khi bản vendored ở **derived** — tức teacher copy **thiếu** patch đã duyệt.

### 1.8 Registry đang dirty so với HEAD

`HEAD` = `6421188f4f93574b7fe093a804dd235d39e630e8`; `workbench/registry/models.yaml` lệch
**189 insertion / 46 deletion**. **Hình dạng đếm bất biến**: HEAD và worktree đều 17 `model_id`,
16 URL, 29 checkpoint variant, cùng danh sách `checkpoint_id`. Cái thay đổi là **trường bằng chứng**:
12 variant có thêm `expected_sha256` + `expected_size_bytes`, 5 variant lên `VERIFIED_METADATA`,
5 `filename` placeholder được resolve. **quan sát được**

⚠️ `REPOSITORY_INVENTORY.md` §2 nói HEAD là `b2fca876` (2026-11-07 13:57:43 +0700). Đó là commit
**cuối cùng chạm `models.yaml`**, không phải HEAD của repo (`6421188`, 32 commit sau). `git diff`
cho cùng một thống kê 189/46 với cả hai mốc, nên kết luận không đổi — nhưng câu chữ trong inventory
phải đọc là "base so sánh của file registry", không phải "HEAD repo". **quan sát được**

---

## 2. Tóm tắt điều hành + ma trận gate theo biến thể

### 2.1 Điều hành

1. **Kê khai là vững.** 16/17/29/21-29 và hai non-model checkout đã được kiểm lại độc lập. Không có
   repository thứ 17 nào xuất hiện. TME là chỗ duy nhất mà "pin" và "clone trên đĩa" là hai thứ khác
   nhau, và nó không ảnh hưởng execution.
2. **Phân nhóm protocol là vững.** 17/17 variant khớp giữa phân loại đọc-từ-source và registry. Cái
   sai là **tài liệu**: 4 số dòng chỉ sai chỗ (comment/flag/header thay vì câu lệnh thật), 3 chỗ gọi
   một file library là "entrypoint", và 2 repo có file sửa mà registry không khai.
3. **Pipeline đọc được ở mức source rất chi tiết, nhưng forward chưa từng được đo.** Mọi gate liên
   quan tới *feature matrix* đều `NOT_RUN` vì GPU thuộc về cohort `intent` suốt cửa sổ kiểm toán.
   Decode/transform thì đo rất kỹ và **bit-identical**.
4. **Đường scoring nhanh bị từ chối vì lý do khoa học, không phải lý do kỹ thuật.** `optimize=True`
   và GEMM chia khối nhanh hơn 4.6–17.8× nhưng đổi **207/18,009** rank. R@10/R@50 không đổi (0 vượt
   biên K), nhưng rank và thứ tự Top-K là artifact đã công bố.
5. **Song song theo trục query thì được** — 5.67× ở 8 thread, so với rank đã lưu của run thật, 0
   mismatch trên 512 query. Đây là khác biệt bản chất: chia theo query không đổi thứ tự cộng dồn
   bên trong bất kỳ query nào.
6. **Bốn defect code đã sửa và một blocker chức năng đã sửa.** ENCODER và PTHA+MTST **không chạy
   được từ HEAD** trước khi sửa `NameError`. Đó là phát hiện nặng nhất của cả chương trình.
7. **Ba artifact full-run mới tồn tại trên đĩa mà corpus kiểm toán không nhắc tới.** `intent`
   `37ee72aa298b276d1bfda0fa` (`seed_applied: null` ⇒ pre-F1), `hint`
   `4bc9893a977e501b82378e76` và `conesep` `30f6de423861cc1d6c78c361` (cả hai `seed_applied: false` ⇒
   post-F1). Xem §6.5/§6.6 — **quan sát được**, nhưng chưa có báo cáo nào chịu trách nhiệm về chúng.

### 2.2 Ma trận gate

G0 và G2/G3/G4 có định nghĩa trong bằng chứng. **G1, G5, G6, G7 không được định nghĩa ở đâu trong
corpus** — xem §10.3. Bảng dưới dùng cách đọc rộng nhất có thể bảo vệ được bằng bằng chứng, và nói rõ
cách đọc đó.

| gate | câu hỏi (theo cách đọc được bảo vệ) | phán quyết | con số |
|---|---|---|---|
| **G0** kê khai | repo/pin/checkpoint có danh tính thật? | **PASS** | 16 repo / 17 variant / 29 checkpoint / 21 staged; 2 non-model checkout |
| **G1** protocol | `native_protocol` có khớp code upstream tại pin? | **PASS 17/17**, với ngoại lệ tài liệu được nêu tên | 0 MISMATCH; 4 citation sai thật; 3 entrypoint sai; 2 repo thiếu khai patch |
| **G2** decode | ma trận pixel có bit-identical qua worker/batch? | **PASS** ở nơi đã đo | habit: 10 dòng đo, 1 sha256 `5fc8002d…`; encoder: 16 dòng, 1 sha256 `4f6a14f9…` |
| **G2-fwd** forward | ma trận feature có bit-identical? | **NOT_RUN** | GPU bận toàn bộ cửa sổ; "pixel bằng nhau không suy ra feature bằng nhau" |
| **G3** scoring, cùng thứ tự cộng | chia trục query (batch size) có đổi rank? | **PASS trên đường shipped** | 96-query slice/condition: **0 mismatch** ở batch 1 / 7 / 256 vs 32, 0 khi repeat, batch cuối không chia hết không đổi 32 hàng đầu. Cùng đường shipped, đo song song theo trục query: 512 query thật, 2/4/8 thread ⇒ 1.81× / 3.26× / 5.67×, **0 mismatch** so với artifact đã lưu |
| **G3b** scoring, khác thứ tự cộng | `optimize=True`/GEMM có đổi rank? | **FAIL** | **207 / 18,009**; 0 vượt K=10; 0 vượt K=50; **19** cửa sổ Top-K đổi thứ tự |
| **G4** axes | batch 1 vs 32, batch cuối, repeat, cold/warm | **PARTIAL** | **PASS**: repeat determinism (`exact_equal: true`, max 0.0) và batch 1 vs 32 trên **đường shipped** (0 mismatch). **FAIL trên GEMM fast path**: **6/6003** rank đổi ở batch 1 vs 32 (multimodal; `7:666`, `7:1988`, `7:2003`, `7:2142`, `7:2253`, `7:4730`) — chính là lý do path đó bị từ chối. **NOT_RUN**: cold vs warm cache, batch axis cho `text_only`/`image_only` (bị kill sau multimodal), cache disabled, và batch cuối không chia hết — đó là **lập luận cấu trúc**, không phải phép đo |
| **G5** cache identity | cache có thể bị dùng lại dưới một phép tính khác? | **DEFECT, đã đóng một phần** — **không phải PASS** | verdict gốc: **7 input** vắng. Đã đóng: BLAS thread count (`blas_threads_pinned`) và `scoring_module_source_hash`. **Còn mở**: TF32/cuDNN flags, byte-level cache verification (sidecar chỉ kiểm fingerprint, **không bao giờ** kiểm bytes mảng), và **851** hàng feature gallery trùng (nhóm lớn nhất 54) làm tie hệ thống. Xem §7.7 |
| **G6** throughput | nhanh hơn bao nhiêu, có đổi kết quả không? | **PARTIAL — không thể PASS cho bất kỳ variant nào** | **CÓ raw**: bão hoà theo batch (`batch.txt`: bs 64/128/256/512 = 17.6/18.0/17.9/18.0 img/s, phẳng); BLAS thread không giúp gì cho đường einsum (`ein_threads.txt`: 6.5/6.0/6.2/5.6/6.2 q/s ở 1/2/4/8/10); decode song song (`vdecode.txt`: 512 ảnh thật, **chỉ decode+transform**, serial 109.1 img/s → **682.7 img/s ở 4 worker** (6.2×), 502 ở 2, 588.5 ở 8 worker — 4 worker là tối ưu, 8 chậm hơn; `bitwise_equal=True` delta 0.0). **Không file raw nào chống lưng cho throughput cả pipeline**: `21.2` baseline và `2089.2` after chỉ có trong prose; `2192 img/s` **không xuất hiện trong bất kỳ file raw nào** và bảng prose của chính nó ngụ ý ~1024 ảnh mỗi dòng (một thí nghiệm khác với 512 ảnh của `vdecode.txt`); bảng wall/q/s theo trục query là prose-only. Xem §8.8 |
| **G7** provenance công bố | artifact trong tree có tái lập được từ tree? | **cần xác minh** | **2** artifact mang nhãn "pre-generated artifact, not runnable from HEAD" (ENCODER, PTHA+MTST — blocker đã gỡ, **chưa re-run**); **4** model có số đã publish bị re-key bởi fix decode (encoder, limn, ptha_mtst, tme) ⇒ run id cũ không tái lập; 1 artifact PARTIAL (limn), 1 FAILED (habit), 2 smoke 8-query |

Ma trận chi tiết theo từng biến thể: `GATE_MATRIX.md`.

---

## 3. Mỗi repository một mục, tách tiểu mục cho từng biến thể đăng ký

Ghi chú chung cho cả mục: "adapter" = `workbench/composition/model_*.py` (đường CIRCUS 3-condition);
"native" = `workbench/replay/*` + `workbench/scripts/evaluate_models.py`. Mọi số dưới đây là
**quan sát được** trừ khi ghi khác.

### 3.1 `CSMCIR` — qzp2018/CSMCIR @ `774f94e2076f…` — 1 biến thể

`/csmcir` · checkpoint `fashioniq` · `fashioniq_tuned_clip_best.pt` · 2,720,615,859 B · sha `c8812071de0640d6…` (MATCH) · `fashioniq_original_split`

- **Kê khai:** vendored tree lệch manifest 2/359 file, đúng 2 patch đã duyệt; `teacher/repos/CSMCIR` ở trạng thái thứ ba (§1.7).
- **Protocol:** nhóm A, reference **được giữ**, gallery = `split.<cat>.val.json` = 15,536. Không có mask trên đường FashionIQ; mask chỉ tồn tại trong `compute_cirr_val_metrics` (`src/validate_blip_csmcir.py:261-264`).
- **Pipeline:** adapter `workbench.composition.model_csmcir`; score dùng `np.einsum("nd,mkd->nmk")` sau khi L2 chuẩn hoá target (`model_csmcir.py:115-116`); nạp bằng `load_state_dict(strict=False)` (`:217`).
- **Trạng thái:** adapter **từ chối cả 3 condition** (`mode_supported()` → `False`). Lý do nguyên văn: `score()` là một embedding text 256-d nhân bản ×32 thành dot product 8192-d, trong khi evaluator native dùng attention-weighted token importance và max trên 32 token — hai đại lượng khác nhau. Cộng thêm caption chỉ phủ **15,415 / 74,381 = 20.72%** gallery.
- **Verdict:** `NOT_RUN`, và việc không chạy là **đúng**, không phải thiếu nỗ lực. Sửa cho khớp = đổi semantics đánh giá; điền caption = bịa dữ liệu.

### 3.2 `AirKnow` — iLearn-Lab/CVPR26-Air-Know @ `6ab6fa95b746…` — 1 biến thể

`/airknow` · `fiq_n05` (50% noise) và `fiq_n08` (80% noise) · 2,724,056,638 B mỗi file (cùng size, khác sha) · `fiq_n05` `f4d25ed9e532f2ee…`, `fiq_n08` `756fc0bdc43d9e0f…`

- **Kê khai:** vendored 1218/1218 khớp manifest; **2 file sửa so với clean clone** (`data_utils.py` fallback `.jpg`→`.png`; `lavis/models/__init__.py` guard import `blip_cir`) mà `approved_source_patches` là **rỗng** — chỉ ghi trong `provenance.local_fixes`.
- **Protocol:** nhóm B. `self.split = 'original-split'` **hard-code** (`datasets.py:31`); mask reference `-10e10` ở `test_BLIP2.py:151`, nằm trong nhánh `else` của `if category == 'fashion200k'` nên **chạy** với FashionIQ. Gallery 15,536.
- **Entrypoint:** `test_BLIP2.py` **không có `__main__`**; được gọi từ `train_BLIP2.py:20,298` trong vòng validate.
- **Trạng thái:** `reproduction_status: NOT_RUN`; `WORKBENCH_AIRKNOW_PYTHON` **không được set** trong `workbench.env`, và `workbench/artifacts/environment_assignment.json` **không tồn tại** → interpreter `None`. Checkpoint ở declared root: **không có** (chỉ có ở tree phụ).
- **Verdict:** không chạy được. Hai variant là 50%/80% noise — kể cả chạy được cũng không phải "clean reproduction".

### 3.3 `ConeSep` — Lee-zixu/ConeSep @ `bd4bf629ebe6…` — 3 biến thể

`/conesep` · `fiq_n02` staged (2,721,428,966 B, `5a864ee5a78c1d24…`); `fiq_n05`, `fiq_n08` **ABSENT** · `fashioniq_full_gallery_ref_excluded`

- **Kê khai:** `git ls-remote` trên cả `Lee-zixu/ConeSep` và `iLearn-Lab/CVPR26-ConeSep` trả **cùng HEAD** `bd4bf629…` ⇒ không phải model thứ hai. **quan sát được (executed)**
- **Protocol:** nhóm B; `self.split = 'original-split'` hard-code (`datasets.py:28`); mask `test.py:68` **không nằm trong `if` nào**.
- **Pipeline:** transform `targetpad_transform(1.25,224)`; score `np.einsum("nd,mdk->nmk", …, optimize=True).max(axis=-1)` (`model_conesep.py:79`); nạp whole-model `.cuda().eval()` (`:296`).
- **Bất đối xứng đáng ghi:** ConeSep dùng subscript `mdk`, còn csmcir/ptha dùng `mkd`. Mỗi cái tự nhất quán với encode path của nó, nhưng đây là kiểu bất đối xứng biến một tensor bị transpose thành điểm "trông hợp lý". **suy luận**
- **Trạng thái:** có adapter, checkpoint staged, `fiq_n02` = **train noise 20%** — nhãn bắt buộc, không phải clean reproduction. **Có artifact mới trên đĩa (sinh trong lúc báo cáo này đang được viết):** `conesep_fiq_n02_30f6de423861cc1d6c78c361` **COMPLETED**, 6,003 query, gallery 74,381, cả **3 condition** `COMPLETED` với `unscored_count: 0`, `seed_applied: false` ⇒ **post-F1**, `topk_export` `TOPK_EXPORT_COMPLETE` depth 50, `missing_query_count: 0`, `reference_eligibility.measured: true, reference_rank: 1`, `checkpoint_mapping_status: VERIFIED_METADATA`, `reproduction_status: NOT_RUN`. Số đo: MM R@10 `35.48225887056472` / R@50 `54.13959686823255` / nDCG `0.3294414713812836`; text R@10 `15.092453773113442` / R@50 `26.869898384141262` / nDCG `0.19955748541119267`; image R@10 `3.1984007996001997` / R@50 `9.611860736298517` / nDCG `0.11206403391502913`; gap `0.3942551173824983`, cohort 6,003, coverage 1.0 cả ba. **quan sát được**, không có trong bất kỳ audit nguồn nào.
- **Verdict:** runnable; `fiq_n05`/`fiq_n08` không thể chạy (không có bytes, không có digest). `fiq_n08` còn có lỗi tên file: `ConeSep-FIQ_N0.8.pt` (underscore) khác hai sibling (`-N0.2`/`-N0.5`).

### 3.4 `MTST_PTHA` — zyinan99/MTST_PTHA @ `6a02ee4763fc…` — 1 biến thể

`/ptha_mtst` · `cirr_transfer` · `tuned_mtst_cirr.pt` · 2,720,352,973 B · `1c4dab86a9f66dfe…` · `fashioniq_original_split`

- **Kê khai:** 360/360 khớp manifest; **2 file sửa so với clean clone** (`MTST_PTHA/src/data_utils.py` đọc `MTST_DATA_ROOT`; `lavis/models/__init__.py` guard một import trỏ tới file không tồn tại) — lại là **khai thiếu trong registry**.
- **Protocol:** nhóm A; gallery `split.<cat>.val.json` 15,536; **có filter tồn tại** `data_utils.py:169-172` trên `<id>.png` → trên host này **0** file thiếu nên gallery hiệu dụng = 15,536. **quan sát được (executed trên split thật)**
- **Entrypoint:** `src/validate_blip.py:main()` **chỉ dispatch CIRR** (`:497-509`). Đường chạy được là `src/blip_validate.py:__main__` → `clip_finetune_fiq(...)` (`:200`).
- **Pipeline:** target layout `(M,k,d)`; **adapter duy nhất dùng `load_state_dict(strict=True)`** (`model_ptha_mtst.py:315`) — lựa chọn an toàn hơn 8 adapter còn lại.
- **Blocker đã gỡ:** trước fix F7 (`§7.5`), `model_ptha_mtst.py` gọi `limit_torch_threads`, `resolved_decode_workers`, `prefetch_rows` mà **không import** → `NameError` trong `_worker_main`, bị `except Exception` biến thành `reply(False, error_type="NameError")` nên trông như lỗi model. **Không thể sinh run mới từ HEAD.**
- **Nhãn bắt buộc:** `SELECTED_TRANSFER_ONLY`; checkpoint train trên CIRR, eval trên FashionIQ. Số `19.473596535065802 / 34.44944194569382` là **số của transfer**, không so được với paper row.
- **Verdict:** artifact `315dd590eeb03d2c78d7937a` COMPLETED, 6,003 query — nhưng là **pre-F1** và **pre-F7** ⇒ "pre-generated artifact, not runnable from HEAD" tại thời điểm artifact đó được tạo.

### 3.5 `HABIT` — iLearn-Lab/AAAI26-HABIT @ `6ddc2cd63eac…` — 3 biến thể

`/habit` · `fiq_n02` staged (2,730,715,930 B, `814d2d0ceb29ef91…`); `fiq_n05`, `fiq_n08` ABSENT · `fashioniq_full_gallery_ref_excluded`

- **Kê khai:** 1234/1234 khớp manifest, không sửa gì so với upstream.
- **Protocol:** nhóm B; ctor default `split='original-split'` (`datasets.py:28`) và `train.py:109-112` **không truyền** → nhánh pair-union `:193` **không thể chạm tới** từ entrypoint của repo. Mask `test.py:151`.
- **Pipeline (đo được ở mức source rất chi tiết):** `targetpad_transform(1.25,224)`; caption qua `BlipCaptionProcessor(prompt="", max_words=50)` **thực thi được**; tokenizer BERT `padding="max_length", max_tokens=32`; **attention mask có dùng**; pooling là **slot cố định `[:,32,:]`**, không phải mean có mask; target layout `(G,32,256)`; score `np.einsum("nd,gqd->ngq").max(-1)`.
- **Artifact:** `habit_fiq_n02_497c25c57fbeba885ecfb9d1` **COMPLETED**, 6,003 query, **MM R@10 35.61552557054806 / R@50 54.3728135932034**, gap `0.388704727214254`, `topk_export` depth 50 `TOPK_EXPORT_COMPLETE`, `missing_query_count: 0`. Đây là **anchor run** của audit equivalence. `seed_applied: null` ⇒ pre-F1.
- **Artifact thứ hai:** `habit_fiq_n02_e0dd04a46ef7a58334b9e353` **FAILED**, `no queries have an exact rank`.
- **Verdict:** model chạy được, số đầy đủ; nhưng là **pre-F1** nên `run_id` không tái lập được, và là pre-decode-change nên không phải artifact của code hiện tại.

### 3.6 `INTENT` — iLearn-Lab/AAAI26-INTENT @ `2a75fc6f57d1…` — 3 biến thể

`/intent` · `fiq_n02` staged (2,722,478,522 B, `2a504cadfd6b1c24…`); `fiq_n05`, `fiq_n08` ABSENT

- **Kê khai:** 1228/1228 khớp manifest.
- **Protocol:** nhóm B; ctor default `'original-split'` (`datasets.py:26`) **và** `train_INTENT.py:61` khai `--split` default `'original-split'`, forward ở `:99` — nên bất kỳ override nào cũng phải do operator chủ động. Mask `test.py:150`. Đây là ca mà cảnh báo "tên split không nói lên reference policy" đúng nhất: nhãn nói `original-split`, scorer **loại** reference.
- **Pipeline:** `targetpad_transform(1.25,224)` (`model_intent.py:325`); tokenizer trong model `blip2_cir.py:744-751`; mask `:753-763`; pooling index 32 `:767-769`; whole pickled module + SHA-256 bắt buộc (`model_intent.py:316-317`, `common.py:45-56` từ chối nếu thiếu digest).
- **Artifact:** **có artifact mới trên đĩa, corpus không nhắc:** `intent_fiq_n02_37ee72aa298b276d1bfda0fa` **COMPLETED**, 6,003 query, **MM R@10 34.39946693320007 / R@50 54.00633016824921**, gap `0.37538250032494136`, `topk_export` `TOPK_EXPORT_COMPLETE` depth 50, `missing_query_count: 0`, `reference_eligibility.measured: true, reference_rank: 1`, `seed_applied: null`. Timestamp file 16:38 hôm nay. **quan sát được** — nhưng `RUNS.vi.md` chỉ ghi `intent` là QUEUED/NOT_RUN. Xem §6.5.

### 3.7 `HINT` — iLearn-Lab/ICASSP26-HINT @ `bec50b6c8c19…` — 1 biến thể

`/hint` · `fashioniq` · 2,724,451,182 B · `bf9c72d42206b667…` · `fashioniq_val_split`

- **Kê khai:** 1225/1225 khớp manifest.
- **Protocol:** nhóm C — gallery = **hợp pair** `source ∪ target` theo thứ tự, 2,628 / 3,089 / 2,902 = **8,619**; mask `test.py:82-83` không điều kiện. Lớp dataset này **không có** tuỳ chọn `original-split`.
- **Pipeline:** cùng substrate BLIP-2 như habit/intent; `targetpad_transform(1.25,224)`; score `einsum("nd,gqd->ngq").max(-1)`.
- **Artifact:** `hint_fashioniq_4bc9893a977e501b82378e76` **COMPLETED**, 6,003 query, **MM R@10 32.816924870897886 / R@50 51.77411294352824**, gap `0.43870420817223343`, `seed_applied: false` ⇒ **post-F1**, `topk_export` `TOPK_EXPORT_COMPLETE`. Timestamp 17:07 hôm nay. **quan sát được**, cũng không được `RUNS.vi.md` nhắc (nó ghi `hint` NOT_RUN).

### 3.8 `ENCODER` — iLearn-Lab/AAAI25-ENCODER @ `29a2a31d6a56…` — 1 biến thể

`/encoder` · `fashioniq` · 631,879,365 B · `0f31c50150a7bda3…` · `fashioniq_val_split`

- **Kê khai:** 26/26 khớp manifest; **2 file** khác clean clone: `evaluate_model.py` đổi `import datasets1` → `import datasets as datasets1`, và **thêm file** `datasets1.py` (1 dòng `from datasets import *`). Cả hai khai trong `provenance.local_fixes`, **không** trong `approved_source_patches` — dù `preflight.py:119,321` đã kỳ vọng `datasets1.py` tồn tại.
- **Protocol:** nhóm C; `val-split` là default ở **cả hai** entrypoint (`datasets.py:64`, `train.py:35`, `evaluate_model.py:15`); mask `test.py:67-68`.
- **Pipeline:** `create_model_and_transforms("ViT-B-32", pretrained=open_clip_pytorch_model.bin)` và **chỉ lấy phần tử thứ ba** (`model_encoder.py:321`) ⇒ `preprocess_val` cho **mọi** input. Upstream dùng `preprocess_train` cho reference và `preprocess_val` cho gallery (`datasets.py:177` vs `:195`).
- **Đo divergence (executed):** `preprocess_train` là `RandomResizedCrop(224, scale=(0.9,1.0))`, **stochastic, không seed**. Trên 64 ảnh thật: mean abs 0.24868646082674112, max abs **3.8269808292388916**; ba lần rút khác nhau lệch nhau đúng 3.8269808292388916; cùng seed → 0.0; khác seed → 0.357215940952301. Ở mức feature ViT-B-32: max abs 1.2528373748064041, mean 0.13496107971573323, cosine min 0.8261280059814453 / mean 0.9276208877563477.
- **Tác động rank (PROXY):** mini-gallery 128 ảnh thật, 64 query: **48/64 rank đổi**, max |Δrank| **51**, R@10 24→23 trên 64, R@50 48→48, nDCG 0.2947158195324436 → 0.29442625175981996. **Đây là PROXY, không phải metric của gallery 74,381** — tự file nói vậy (`kind`).
- **Blocker đã gỡ:** 3 helper không import → `NameError` (§7.5).
- **Artifact:** `6a528e877c328772e2c7bb28` COMPLETED 6,003 query, **MM R@10 31.01782442112277 / R@50 49.5752123938031**, gap `0.47230210227676583`, `seed_applied: null`. Artifact thứ hai `eda30a37c0bc17ac43844f09` là **smoke 8 query** (R@10 12.5) — phải loại khỏi mọi claim.
- **Verdict:** số đã công bố là **pre-generated artifact, not runnable from HEAD** tại thời điểm audit-fix; blocker đã gỡ nhưng **chưa re-run**, và `run_id` đã đổi nên không thể tái lập đúng id đó.

### 3.9 `PAIR` — iLearn-Lab/ICASSP25-PAIR @ `bfb0b98cdb4c…` — 2 biến thể

`/pair` · `pair_b1` (694,958,565 B, `c4d861ecd0baa3f1…`), `pair_b2` (694,872,371 B, `74346fd7de6efdaf…`) · cả hai **không staged ở declared root** (chỉ ở tree phụ)

- **Kê khai:** 22/22 khớp manifest; **thêm file** `models/MHTransformer_rev.py` (reconstruct module mà checkpoint cần) — khai trong provenance, không trong registry.
- **Protocol:** nhóm C; `val-split` default (`data/datasets.py:23`, `train.py:29`); mask `test.py:67-68`.
- **Entrypoint:** `test.py` chỉ có `test(params, model, testset, category)`, **không `__main__`**; README chỉ công bố lệnh CIRR submission.
- **Trạng thái:** registry `checkpoint_mapping_status: UNVERIFIED` cho cả hai; `reproduction_status: BLOCKED_CHECKPOINT_MAPPING_UNVERIFIED`. Không có composition adapter.
- **Verdict:** không chạy được. Báo cáo trước ghi MAPPING_CONTRADICTED (pickle cần `model_vssp` + `MHTransformer_dis` mà repo pin không ship; 74 key lạ, 70 shape mismatch, 5.20% tham số ngoài forward) — **unresolved**, chưa kiểm lại trong lượt này.

### 3.10 `CLVCNet` — iLearn-Lab/SIGIR21-CLVC-Net @ `bd9b68894898…` — 1 biến thể

`/clvc_net` · `author_drive_unresolved` · bundle 6 file: `dress_global_model.pt` 176,437,419 / `dress_local_model.pt` 217,168,235 / `shirt_global_model.pt` 177,872,427 / `shirt_local_model.pt` 218,603,243 / `toptee_global_model.pt` 177,578,027 / `toptee_local_model.pt` 218,308,843 · tổng 1,185,968,194 B

- **Kê khai:** 9/9 khớp manifest; `resnet.py` sửa `torch.hub.load_state_dict_from_url` + fallback torchvision. 6 digest khớp `selected_bundle_digests.json`.
- **Protocol:** nhóm C **vì `gallery_all=False`**. `__init__` default `False` (`datasets.py:153`) và `train.py:86-96` không truyền ⇒ nhánh `else` (`:242`) chạy ⇒ pair union 8,619. `grep gallery_all` chỉ ra `datasets.py:153,163,234` — không config, không CLI.
- **Entrypoint:** README nói `python test.py` nhưng `test.py` có **0** `__main__` ⇒ lệnh đó không làm gì; driver thật là `train.py:259`.
- **Blocker:** **4 lời gọi `.cuda()` không guard** trên đường chấm ⇒ buộc có GPU; torch 1.6 chỉ ship kernel tới sm_86, host là compute capability 12.0. Đây là blocker hardware/dtype thật.
- **Mã hoá nội bộ 6 file: chưa đọc** (`UNVERIFIED`) — tên `dress_local`/`dress_global` *gợi ý* split local/global nhưng không chứng minh.

### 3.11 `DCNet` — ozmig77/dcnet @ `68c79d38569f…` — 1 biến thể

`/dcnet` · `fashioniq_run_directory` · `trained_model.pth` 2,877,310,172 B (`130f031e0f7808cc…`) + `config.json` 1,734 B + `val/` 4,096 B **không khai báo** · `fashioniq_full_gallery_ref_excluded`

- **Kê khai:** 38/39 khớp manifest, 1 file lệch đúng patch đã duyệt (`base/base_trainer.py` → `weights_only=False`). `trained_model.pth` là **legacy non-zip pickle** — không đọc được nếu không execute, nên cấu trúc key **UNVERIFIED**.
- **Protocol:** nhóm B dưới tên `val_trg`: `split.split('_')[0]` → `val` ⇒ `split.<cat>.val.json`, 15,536, thứ tự nguyên vẹn. Mask **cả hai nhánh** `score_matrix_comp` và `score_matrix_corr` (`trainer_joint.py:217-218`), và cả log-softmax sum.
- **Đo được (executed):** chạy thật `20261108T072226547971Z_dcnet_fashioniq_run_directory_f969aef3` → `EXECUTION_FAILED`, `return_code: 1`, `TypeError: 'weights_only' is an invalid keyword argument for load()`.
- **Nguyên nhân = mâu thuẫn hai chiều:** patch đúng cho torch ≥ 2.6 (nơi `weights_only` là default), pin đúng cho 1.4.0 (nơi **không có** tham số đó), và không tồn tại tổ hợp trên cp37: `'weights_only' in signature == False` trên 1.4.0; `version_satisfies('==1.4.0','1.13.1') == False`; wheel `torch-1.13.1+cpu-cp37` **có**, `torch-2.0.1` cp37 **không có**.
- **Verdict:** `FAIL` có bằng chứng, không phải "nghi ngờ". Cách gỡ đều thuộc registry/patch (ngoài phạm vi các audit này).

### 3.12 `CLIP4Cir` — ABaldrati/CLIP4Cir @ `dfed9f748a8a…` — 2 biến thể

`/combiner_rn50x4_noft` · `rn50x4_noft_unresolved` · `fiq_comb_RN50x4_noft.pt` · 236,017,921 B · `b31882fe6f7e3e194cb9cfda048926844ee0ccc9d4ebc5483fb3bacfa5972955`

- **Protocol:** nhóm A; `split.<cat>.val.json` 15,536; **không mask** (`src/validate.py:21-57`); mask duy nhất ở `:162-166` trong `compute_cirr_val_metrics`. `--dataset` là `required=True` nên README chọn.
- **Chạy thật:** `run_id 20261108T064137301769Z_combiner_rn50x4_noft_rn50x4_noft_unresolved_020a70b0`, exit **0**, ~81 phút CPU, `EXECUTION_COMPLETED` / `EXTRACTED_VERIFIED` / `COMPLETED`, interpreter `clip4cir-38` (py 3.8.20, torch 1.11.0+cpu), `runtime_adaptation: null`.

  | category | gallery | R@10 | R@50 |
  |---|---:|---:|---:|
  | dress | 3817 | 30.639565 | 56.767476 |
  | shirt | 6346 | 36.947989 | 57.262021 |
  | toptee | 5373 | 37.735850 | 61.397249 |
  | **macro** | — | **35.1078** | **58.4756** |

  `mean` (workbench-derived) = **46.7917**.
- **Không so được với paper:** `benchmark_gate.status = METRIC_REFERENCE_UNVERIFIED`, lý do nguyên văn *"no publishable comparison — every committed row for it is `['UNVERIFIED']`"*. Verifier: `verified=false` nhưng `evidence_reasons=[]` — tức **không có lý do bằng chứng** nào bị từ chối, chỉ thiếu hàng paper. Đây là thuộc tính thường trực của những gì tác giả công bố.
- **Blocker đã gỡ để chạy được:** registry khai `fashioniq_image_extensions: [".jpg"]` nhưng link trong checkout trỏ `prepared/clip4cir/fashionIQ_dataset`; dữ liệu **đủ** (`images/*.jpg` là symlink tới `data/FashionIQ/images/*.png`, 15,415 file, 3817/3817 + 6346/6346 + 5373/5373). Cách gỡ **không đổi ngữ nghĩa**: truyền `--dataset-root` **và** `--canonical-dataset-root` cùng trỏ một thư mục.

`/clip4cir_rn50x4_fullft` · `rn50x4_fullft_unresolved` · bundle 2 file 949,892,616 B

- **Protocol:** như trên (cùng `validate.py`).
- **Preflight:** dry-run exit 0. Nhưng **lệnh nó dựng trỏ tới hai file không tồn tại**: `RN50x4_fullft/combiner_state.pt` và `RN50x4_fullft/clip_state.pt`, trong khi trên đĩa chỉ có tên tác giả `fiq_comb_RN50x4_fullft.pt` / `fiq_clip_RN50x4_fullft.pt` (registry khai đúng hai tên này trong `required_files`).
- **Nguyên nhân:** `workbench/backend/adapters/models.py:374-375` hard-code tên thay vì đọc `required_files`.
- **Verdict:** `NOT_RUN`; sẽ chết ngay `FileNotFoundError`. Sửa một dòng, chờ chủ file `workbench/backend/**`.
- **Dữ liệu:** hai member đều đã staged và hash khớp; nhưng **không có manifest/checksum của tác giả cho việc pairing** hai state này.

### 3.13 `TGCIR` — iLearn-Lab/MM23-TG-CIR @ `65fa78eaf8ca…` — 1 biến thể

`/tgcir` · `tgcir_archive_unresolved` · `Fashioniq.pt` · 607,928,637 B · `c37768f3…` (MATCH declared)

- **Kê khai:** 11/11 khớp manifest; **whole-model pickle (RCE hazard)** nên registry từ chối pin digest theo một số đường. Cấu trúc trong **chưa đọc** (`UNVERIFIED`).
- **Protocol:** nhóm C; pair union 8,619; mask `test.py:66-67` không điều kiện; entrypoint `train.py:23` → `:200`, `test.py` không `__main__`.
- **Blocker:** như `clvc_net` — 4 `.cuda()` không guard ⇒ cần GPU + kernel mới.
- **Verdict:** `BLOCKED`.

### 3.14 `SPRC` — chunmeifeng/SPRC @ `2935a5397732…` — 2 biến thể

`/sprc` · `sprc_fiq_unresolved` (**không có bytes**, `find /` 0 kết quả) · `sprc_fiq_vitl` (1,459,756,541 B, `19386104271c01ea…`, `scientific_variant: true`) · `fashioniq_original_split`

- **Kê khai:** 354/354 khớp manifest, không sửa gì.
- **Protocol:** nhóm A; gallery `split.<cat>.val.json` 15,536; **không mask**; không filter tồn tại. Entrypoint: `validate_blip.py:main()` **chỉ CIRR**; đường thật là `src/blip_validate.py:__main__` (`:155`).
- **Preflight:** dry-run exit 1 với **3 blocker khác nhau**: (a) canonical dataset link trỏ `prepared/sprc/fashionIQ_dataset`; model này **không có** `--dataset-root` flag nên không gỡ được như CLIP4Cir; (b) `BLOCKED_EXTERNAL_ASSET_PROVENANCE` cho BLIP-2 — file *có* (698.9 MB `blip2_pretrained_vitL.pth`, 712.4 MB `blip2_pretrained.pth`) nhưng tác giả **không công bố sha**, và `asset_admission_status` từ chối mọi `UNRESOLVED_PROVENANCE`; (c) `command_status: COMMAND_BLOCKED_MODEL_MAPPING` là **stale** — nó mô tả `sprc_fiq_unresolved` (file không tồn tại), không mô tả variant ViT-L đã `MAPPING_CONFIRMED` 628/628.
- **Ghi nhận về môi trường:** env `sprc-39` (py 3.9.25, torch 2.0.1+cpu) phải cài thêm **5 package** mà registry contract không khai (`decord==0.6.0`, `pycocoevalcap==1.2`, `webdataset`, `einops`, `scikit-image`) thì mới `import blip_validate` được. Tức `verify_environment` báo `DEPENDENCIES_OK` trên một env **không import nổi evaluator**.
- **Đường CPU (nếu chạy được):** SPRC khai `vit_precision: "fp16"` và `clip_vit.py:242-260` gọi `convert_weights_to_fp16`, **nhưng** `lavis/models/__init__.py:243-244` có `if device == "cpu": model = model.float()`. Đo thật: `vit dtype torch.float32`, forward `[1,257,1024] float32`, không lỗi. ⇒ chạy CPU là **fp32**, một khác biệt số học phải khai trên nhãn artifact.
- **Verdict:** `NOT_RUN`; `sprc_fiq_vitl` là ViT-L, **không so được** với row ViT-g của paper dù chạy được.

### 3.15 `LIMN` — iLearn-Lab/TPAMI24-LIMN @ `7d7bc9b116f5…` — 3 biến thể

`/limn` · `base_iter0_dress` / `base_iter0_shirt` / `base_iter0_toptee` · mỗi file **1,792,010,157 B** với **ba digest khác nhau** (`96a335d4eb8b1e77…` / `9260c7f77d3017d8…` / `79f663a14d89d455…`) · `fashioniq_val_split`

- **Kê khai:** 33/33 khớp manifest. Cùng size + khác digest được kiểm là **hợp lý** (cùng kiến trúc, khác trọng số), không phải file trùng.
- **Protocol:** nhóm C; pair union 8,619; mask dưới `if name != 'birds'` (`test.py:80-82`) — với `dress|shirt|toptee` thì **chạy**. Guard `!='birds'` tồn tại vì loader Birds là cái duy nhất bật `gallery_all` (`datasets.py:315-322,419`).
- **Pipeline:** transform dựng từ **visual tower của checkpoint đã nạp**: `open_clip.image_transform(visual.image_size, is_train=False, mean=visual.image_mean, std=visual.image_std)` (`model_limn.py:527-528`). Đo: checkpoint **không mang** `image_mean`/`image_std` ⇒ rơi về default open_clip `(0.48145466,…)` / `(0.26862954,…)` = **giống upstream**; `image_size` 224. Một transform cho query và target — **khớp** upstream `get_test_img` → `transform[1]` (`datasets.py:123-130,151,153,169`); `transform[0]` chỉ dùng ở `__getitem__` (`:104-105`) tức training.
- **Scoping:** 3 whole-model checkpoint theo category; `dataset_type` `{"dress":0,"shirt":1,"toptee":2}` khớp upstream `datasets.py:21-24`. 119/6,003 query **không gán được category** (reference dùng chung shirt/toptee) → vào `scope_errors`, **giảm coverage, không phải retrieval miss**.
- **Artifact:** `10173dc45dea8c90fd2a284b` **PARTIAL**, 6,003 query, cohort **5,884**, coverage `0.98018`, MM R@10 `30.064581917063222` / R@50 `48.759347382732834`, gap `0.5417445849251903`. `seed_applied: null`. Artifact thứ hai `3f86f58483b368377e2c151e` là **smoke 8 query** (R@10 25.0) — phải loại.
- **Verdict:** chạy được; `reference_eligibility.measured: false` vì `score()` cần `Mapping` gallery thật — đây là câu trả lời **trung thực**, không phải `false` giả.

### 3.16 `TME` — li-shuxian/TME @ `394aad79df2c…` (registry) — 2 biến thể

`/tme` · `fashioniq_0` (2,728,382,104 B, `fea7fc08760dad1f…`, **PRESENT_NO_DECLARED_SHA**) · `cirr_0` (**ABSENT**) · `fashioniq_original_split`

- **Kê khai:** 349/351 khớp manifest, 2 file lệch đúng 2 patch đã duyệt (`src/precompute.py`, `src/precompute_evaluation.py`). **Pin registry không tồn tại trong `teacher/repos/TME`** (§1.6) — không ảnh hưởng execution.
- **Protocol:** nhóm A. Gallery = `split.<cat>.val.json` **theo từng category**: `3817 / 6346 / 5373`, không phải 15,536 gộp. Reference **được giữ**: mask nằm trong `if category == 'CIRR':` (`precompute_evaluation.py:83`), mà `category ∈ {dress,shirt,toptee}` cho FashionIQ (`:15-17`) ⇒ **không bao giờ chạy**.
- **Điểm chưa giải quyết:** membership gallery lúc chạy lấy từ cache `features/FashionIQ/val_images.pt` + `val_name_to_idx.json` (`data.py:250-255`). Trên host này hai file là **1 byte / `{}`** — không phải cache dùng lại được, và **không đọc được cache của tác giả** ⇒ identity của membership verified từ code, nội dung cache thì không.
- **Pipeline:** transform từ `utility.targetpad_transform` của chính TME (`model_tme.py:353,368`); `load_state_dict(strict=False)` (`:362`) — trong khi `TME/src/lavis/models/eva_vit.py:449` gọi `create_eva_vit_g(..., strict=False)` và **vứt báo cáo**; cái chặn rủi ro là outer registry checkpoint cung cấp đủ key nên outer check đã fail nếu thiếu. **suy luận**, đã ghi rõ là điểm hẹp.
- **Artifact:** `ad477312aa57c1b9114f9e51` COMPLETED, 6,003 query, MM R@10 `34.01632517074796` / R@50 `53.95635515575545`, gap `0.3586012581618244`, `seed_applied: null`. **Nhưng `full.json` không chứa key `variant_provenance`** ⇒ badge `sm_120` adaptation không được ghi vào artifact, chỉ nằm ở `cohort_status.txt`. Đây là khuyết điểm provenance thật: người đọc một artifact đơn lẻ sẽ tưởng đây là reproduction hợp lệ.
- **Verdict:** chạy được dưới `sm_120`; `cirr_0` không có bytes.

---

## 4. So sánh upstream vs adapter

Mức kiểm chứng: **EXECUTED** = có phép đo trên dữ liệu thật; **STATIC** = đọc source có `file:line`,
chưa chạy. Không có dòng nào dưới đây là "adapter so với adapter".

### 4.1 HABIT — `third_party/HABIT/*` vs `workbench/composition/model_habit.py`

| trục | upstream | adapter | mức |
|---|---|---|---|
| dựng transform | `data_utils.py:92-105` `targetpad_transform(1.25,224)` | cùng hàm, cùng đối số (`model_habit.py:340`) | STATIC (cùng object từ cùng checkout) |
| chuỗi | TargetPad(1.25) → Resize(224,BICUBIC) → CenterCrop → rgb → ToTensor → CLIP Normalize (`:99-104`) | y hệt | STATIC |
| xử lý aspect | pad tới tỉ lệ 1.25, **không stretch**; nếu `max/min < 1.25` thì **không pad gì** (`:64-76`) | y hệt | STATIC |
| query vs gallery | **một** transform cho cả hai (`datasets.py:139,141,195,202`) | một `transform`, một `decode` | STATIC |
| augmentation lúc eval | không (TargetPad/Resize/CenterCrop deterministic) | không | STATIC |
| text processor | `blip_caption` → `BlipCaptionProcessor(prompt="", max_words=50)` (`blip_processors.py:29-73`) | `processors[0]["eval"]` từ cùng config | **EXECUTED**: lowercase, strip `.!\"()*#:;~`, gộp whitespace, 50 từ |
| chuỗi caption | upstream ghép 2 caption bằng `" and "` (`datasets.py:120`) | dùng `query_txt` nguyên văn, **không** re-split | **EXECUTED** (string test) |
| tokenizer | BERT từ `/root/bert-base-uncased`, `[DEC]` là bos (`blip2.py:31-34`) | trong model, không adapter tokenizer | STATIC + **EXECUTED** ở mức id |
| pooling | slot cố định `[:,32,:]`; attention mask che phần pad (`HABIT.py:399-407`) | y hệt | STATIC |
| target layout | `(G,32,256)` | y hệt, transpose trong worker | STATIC |
| similarity | `np.matmul` rồi `.max(-1)` (`test.py:141-143`) | `einsum("nd,gqd->ngq").max(-1)` | **EXECUTED**: batch-1 vs batch-32 delta ≤ 6.6e-7, **0 rank change** |
| gallery | danh sách per-category đã dedupe | danh sách 74,381 phát hành, giữ thứ tự | **khác có chủ ý** |
| reference | mask `-10e10` (`test.py:151`) | **eligible**, đo được rank 1 | **khác có chủ ý** |
| metric | percent recall, **in riêng từng category**, không pool | Recall percent + nDCG/MRR fraction + gap, pool trên cohort | **khác có chủ ý** |
| kích thước gallery upstream | `split.<cat>.val.json` = **3817 / 6346 / 5373** | 74,381 | **khác có chủ ý**, định lượng được |

### 4.2 ENCODER — `third_party/ENCODER/*` vs `workbench/composition/model_encoder.py`

| trục | upstream | adapter | mức |
|---|---|---|---|
| dựng transform | `create_model_and_transforms('ViT-B-32', …)` → `[preprocess_train, preprocess_val]` (`evaluate_model.py:40,43`) | cùng call, **chỉ lấy thứ ba** (`model_encoder.py:321`) | STATIC |
| transform reference | `get_img(..., stage=0)` = **`preprocess_train`**, `RandomResizedCrop(224, scale=(0.9,1.0))`, **stochastic, không seed** (`datasets.py:177`) | `preprocess_val`, deterministic | **EXECUTED**: mean abs 0.24868646082674112, max 3.8269808292388916; ba draw lệch nhau 3.8269808292388916; cùng seed 0.0; khác seed 0.357215940952301 |
| transform gallery | `stage=1` = `preprocess_val` (`:195,202`) | `preprocess_val` | STATIC |
| `target_img_data` của query | tính ở `:178` với `stage=0` rồi **không được đọc** (`test.py:27` chỉ dùng `source_img_data`) | n/a | STATIC ⇒ dead code upstream |
| tokenizer | `open_clip.get_tokenizer('ViT-B-32')`, 77 slot, **không truyền attention mask** (`model_try2.py:137,385`; tower chỉ dùng causal mask `:343`) | y hệt | **EXECUTED**: `''` → `[49406, 49407, 0×75]` |
| pooling | `mean(fuse_local, dim=1)` trên **6** token `cat([global(1), action(2), selected(3)])` (`:412,428,490`) | y hệt; shape propagation qua `FeatureWiseAffine` cho `(B,6,512)` | STATIC + shape check |
| ảnh hưởng của pad | 77-token context nuôi `text_local_fea`; `ScoreNet(sparse_ratio=0.6)` giữ `ceil(77*0.6)=47`, `FactorNet` softmax 47 → 3 (`:42-50,77-89`) | y hệt | **suy luận** — chưa có forward để đo |
| similarity | dot sau L2 từng hàng (`test.py:55-62`) | L2 hai phía + dot, chunk 8192 (`model_encoder.py:221-238`) | **EXECUTED**: 0 rank diff |
| mask reference | `sims[i, targets.index(source)] = -10e10` (`test.py:68`) | **không** | **khác có chủ ý** |
| rank | `argsort(-sims)[:50]` rồi membership 0-based | rank chính xác 1-based trên toàn gallery | **khác có chủ ý** |
| nạp checkpoint | `load_state_dict(model, strict=False)` (`evaluate_model.py:75`) | `strict=False` **+ `_assert_checkpoint_loaded`** (`model_encoder.py:315-318`) | STATIC; adapter chặt hơn |

### 4.3 LIMN — `third_party/LIMN/LIMN/datasets.py` vs `model_limn.py`

| trục | upstream | adapter | mức |
|---|---|---|---|
| transform eval | `get_test_img` → `transform[1]` (val) cho reference, target, **và mọi gallery row** (`:151,153,169`) | `image_transform(..., is_train=False, …)` cho cả ba vai (`:527`) | **khớp** |
| rò rỉ transform train | `transform[0]` chỉ qua `__getitem__` (`:104-105`) | không dùng | STATIC |
| hằng số normalize | open_clip default | đọc từ tower; checkpoint không có buffer mean/std ⇒ = default | **EXECUTED** (probe key) |
| `dataset_type` | `{dress:0, shirt:1, toptee:2}` (`:21-24`) | cùng literal (`:512`) | STATIC |
| mask reference | không mask trên đường val | không mask | STATIC |
| rank | argsort riêng | 1-based, tie thắng theo vị trí sớm hơn (`:400-416`) | STATIC + đo tie stability |

### 4.4 BLIP-2 family còn lại — khác biệt theo trục

| model | target layout | width | nạp | ghi chú |
|---|---|---|---|---|
| `habit` / `intent` / `hint` / `tme` | `(G,32,256)` | 256 | whole/trusted; `tme` `strict=False` | `einsum("nd,gqd->ngq").max(-1)` |
| `conesep` | `(M,d,k)` = `mdk` | 8192 | whole, `.cuda().eval()` | `optimize=True` |
| `csmcir` | `(M,k,d)` = `mkd` | 8192 | `strict=False` | `optimize=True` |
| `ptha_mtst` | `(M,k,d)` = `mkd` | 32×256 | **`strict=True`** | `optimize=True` |

Ba điểm phải nêu, đã nêu trong audit gốc và **vẫn đúng**:

1. `encoder` là model duy nhất **không có attention mask** để loại pad, nên `image_only_empty_text` của nó là một truy vấn `[SOT][EOT]` **do pad chi phối**, không phải "không có text" — **không so ngang** được với các family khác.
2. `conesep` dùng `mdk` còn `csmcir`/`ptha_mtst` dùng `mkd`, cả ba kết thúc bằng `.max(axis=-1)`. Mỗi cái tự nhất quán; nhưng đây là loại bất đối xứng biến tensor transpose thành điểm hợp lý về shape.
3. `strict=False` xuất hiện ở 5/9 adapter (`tme`, `csmcir`, `encoder`, cộng các đường whole-module bypass key matching). Chỉ `ptha_mtst` fail to nếu lệch key.

### 4.5 Chỗ hai audit nói khác nhau về diff vendored

`PROTOCOL_FINDINGS.md` §7 nói *"7 modified files + 2 added files across 7 repositories"*. Nhưng
`raw/vendored_diffs.txt` — chính file nó trích — có **15 khối diff**: **13 file sửa + 2 file thêm**,
trải trên **8 repository** (`MTST_PTHA` 2, `TME` 4, `AirKnow` 2, `DCNet` 1, `ENCODER` 2, `CLVCNet` 1,
`CSMCIR` 2, `PAIR` 1).

Audit nào đúng: **file raw đúng**, con số trong prose sai. **quan sát được** (đếm khối `======== `).

Đồng thời `REPOSITORY_INVENTORY.md` §3.2 nói chỉ **3 repo lệch manifest** (CSMCIR 2, DCNet 1, TME 2
= 5 file) và 13 repo khớp 100%. Hai bên **không mâu thuẫn**: chúng so với hai baseline khác nhau.
Manifest được sinh từ bytes **sau** `local_fixes`, nên một repo có thể khớp manifest 100% mà vẫn khác
clean clone — đúng trường hợp `AirKnow` (1218/1218 manifest, 2 file khác clone), `MTST_PTHA`
(360/360, 2 file), `ENCODER` (26/26, 2 file), `CLVCNet` (9/9, 1 file), `PAIR` (22/22, 1 file thêm).
**Đây là chỗ dễ đọc sai nhất của bộ bằng chứng**: câu "vendored tree khớp provenance 100%" **không**
có nghĩa "vendored tree giống upstream". **suy luận từ hai nguồn observed**

---

## 5. Adaptation protocol cố ý

Mỗi mục dưới đây là **quyết định có chủ ý**, không phải defect. Chúng phải được nêu trong mọi summary
vì chúng làm con số **không so được** với upstream.

### 5.1 ENCODER: transform deterministic nơi upstream là stochastic

- **Upstream:** reference image qua `preprocess_train` = `RandomResizedCrop`, **không seed** ở script
  eval của tác giả (`evaluate_model.py:38-44` + `datasets.py:177`). **quan sát được**
- **Adapter:** `preprocess_val` cho **mọi** input. **quan sát được**
- **Biện luận:** upstream **không tái lập được** — ba lần rút lệch nhau 3.8269808292388916, cùng độ
  lớn với khoảng cách train-vs-val. Thêm nữa, gallery val **dedupe `source ∪ target`**, nên ảnh
  reference **cũng là một gallery row**: upstream encode **cùng một file** hai lần, một lần ngẫu
  nhiên (query) và một lần tất định (gallery) — protocol của chính upstream không nhất quán. Với
  protocol reference-eligible, encode reference ngẫu nhiên sẽ phá chính rank mà ta định đo.
- **Hệ quả lên số đã công bố:** **không**. `31.01782442112277` là số của
  `circus_fashioniq_release_v1`, tính bằng một transform tất định cho mọi input, và summary đã mang
  `reproduction_status: METRICS_MEASURED_MAPPING_UNVERIFIED`.
- **Cái thật sự thiếu:** summary **không nói** rằng đường composition dùng `preprocess_val` cho
  query trong khi đường replay dùng `preprocess_train`. Đó là **lỗ hổng ghi chú**, không phải lỗi
  hành vi.

### 5.2 Reference eligible trong ranking, thay cho mask `-10e10` của upstream

- Upstream mask reference ở 5 nhóm: `habit`/`airknow` `test*.py:151`, `conesep` `test.py:68`,
  `intent` `test.py:150`, `dcnet` `trainer_joint.py:217-218`, `hint`/`encoder`/`pair`/`tgcir`/
  `clvc_net` `test.py:66-83`, `limn` `test.py:80-82`. **quan sát được**
- Spec **cố ý không** kế thừa (`CIRCUS_COMPOSITION_SPEC.md` "Protocol and gallery"): "reference is
  eligible in ranking".
- Kiểm chứng không tin lời khai: runner gọi `adapter.measured_reference_eligibility()`, chạy
  `score`+`rank` thật trên gallery tổng hợp nơi reference là exact match, và đọc rank đo được
  (`adapters.py:424-470`, gọi ở `runner.py:884-890`). **quan sát được**
- Kết quả trên artifact thật: `hint`, `intent`, `conesep` mới đều có
  `{"measured": true, "reference_eligible": true, "reference_rank": 1}`; `habit` trước đó có
  `reference_eligibility` giống vậy. `limn` trả `measured: false` kèm lý do (cần `Mapping` gallery
  thật) — **trung thực**, không phải false-negative.
- **Chỗ dễ đọc sai:** artifact cũ (`497c25c5`, `6a528e…`, `ad477…`, `315dd…`, `10173…`) mang
  `reference_eligible: true` dạng **literal** ở `runner.py` thời điểm đó. Đừng trích nó như bằng
  chứng đo được. **quan sát được** (`BLOCKERS.vi.md` B7)

### 5.3 Ảnh đen thay vào **raw PIL** trước transform

- `runner._stage_query_path` mở file, `copy()` PIL raw, gọi `adapt_query_inputs`, và với
  `text_only_black_image` **ghi PIL đã biến đổi ra PNG** rồi trả path; worker mở và transform file đó.
  `inputs.black_image_like` trả `Image.new(image.mode, image.size, 0)` — **cùng size, cùng mode, số 0
  thô, trước mọi transform**. **quan sát được**
- Kiểm chứng phân biệt được với can thiệp *sai*: một ảnh đen 342×444 qua `preprocess_val` của ENCODER
  cho tensor hằng với các giá trị duy nhất `[-1.7923, -1.7521, -1.4802]`, mean **−1.6749**, std
  0.1386 — tức **màu đen sau normalize**, không phải 0. Một implementation "zero sau normalize" hoặc
  "zero hoá feature" sẽ cho mean 0. **quan sát được**
- Kiểm thêm mode: 4000 ảnh phát hành ngẫu nhiên là `{'RGB': 3998, 'L': 2}`; với 2 file không-RGB,
  `black_image_like` + `convert("RGB")` vẫn đúng `(0,0,0)`. **quan sát được**

### 5.4 Empty text = special token bình thường

- Chuỗi: `inputs.adapt_query_inputs` trả `""` (`inputs.py:26`) → `BlipCaptionProcessor` lowercase/strip
  → `""` → BERT tokenize.
- Đo: `''` → `ids [101, 102, 0,0,…]`, **2 non-pad**, decode `[CLS] [SEP]`. open_clip ViT-B-32:
  `[49406, 49407, 0×75]`, pad 75. **quan sát được**
- Vậy điều kiện này đúng nghĩa "bỏ edit text trước tokenization; special token bình thường được
  phép còn lại" — đúng như spec. **Không** sinh chuỗi toàn pad, **không** đổi pattern special token.
- Caveat phải ghi: `"   "` và `""` **không phân biệt được** vì processor strip. Nếu tương lai muốn
  ablation "rỗng nhưng không vắng", processor hiện tại không diễn đạt được.

### 5.5 Gallery CIRCUS 74,381 vs gallery category của upstream

- Upstream dùng gallery per-category: nhóm A 15,536 (`split.<cat>.val.json`), nhóm C 8,619 (pair
  union), TME **chia theo category** 3817/6346/5373. **quan sát được (executed trên split thật)**
- CIRCUS dùng **74,381** identity phát hành, giữ nguyên thứ tự.
- Vì vậy Recall upstream và Recall composition khác nhau **đồng thời** ở kích thước gallery, policy
  reference, cách pool cohort và bộ metric ⇒ **không so được theo cả hai chiều**. Không có số
  composition nào được đọc như reproduction upstream.
- Ghi chú về paper: spec nêu paper nói 18,853 ảnh nhưng artifact phát hành có 74,381; rank lớn nhất
  quan sát được trong artifact là 31,987 ở sample index 4435 ⇒ đây không phải metadata thừa.

### 5.6 nDCG / gap: đơn vị

- `Recall` là **PERCENT**; `nDCG` và `gap` là **FRACTION**; `mrr` là fraction. `metrics.py:73-76` mã
  hoá đúng điều này; gap là fraction (`:148`).
- `composition_gap_ndcg = 1 - max(I,T)/MM`, **tính sau khi lấy trung bình trên cùng cohort**, không
  phải trung bình của tỉ lệ từng query (`metrics.py:126-148`). Gap âm được **giữ**, không clamp.
- Query không chấm được **giảm coverage**, không vào mean, **không** tính là miss
  (`metrics.py:59,72`).
- `mrr_gap` là đại lượng **riêng**, không được gọi là `compgap`.

### 5.7 Query text không bị re-split theo "and"

`inputs.adapt_query_inputs` trả `query.query_txt` nguyên văn; `grep` không thấy chỗ nào split trong
`workbench/composition/`. Biện luận: text phát hành **đã là** caption đã ghép; tách ra rồi ghép lại
sẽ tạo ra cấu trúc mà bản phát hành không khẳng định. **quan sát được**

### 5.8 Tên condition đặt theo input bị **bỏ**, không theo tên file upstream

Spec ghi rõ: `ablation_configs/image_only_zero_image.json` chứa `{"name": "img_black"}` nhưng được
dùng cho **text-only**; `text_only_drop_text.json` chứa `{"name": "text_drop_all"}` nhưng dùng cho
**image-only**. Vì vậy workbench đặt tên theo nghĩa (`text_only_black_image`,
`image_only_empty_text`), không theo tên file. **quan sát được**

---

## 6. Sample ID thật, command, môi trường, precision, tolerance, so sánh trung gian, đường dẫn bằng chứng

### 6.1 Cohort và sample ID

| đại lượng | giá trị | nguồn |
|---|---|---|
| gallery | **74,381** identity, thứ tự cố định, `gallery_global.npy` shape `(74381, 32, 256)` cho BLIP-2 family | `CIRCUS_COMPOSITION_SPEC.md` §Release data contract; `cache_and_ties.json` |
| query full | **6,003** | M-BEIR `mbeir_fashioniq_task7_test.jsonl` |
| shortcut-free | **4,069** | `…_test_multimodal.jsonl` |
| validated solved / unsolved / union | **368 / 218 / 586**, giao = 0, `validated ⊆ sf ⊆ full` | spec |
| val query của FashionIQ (dùng cho gallery native) | 2017 / 2038 / 1961 = **6,016** | `PROTOCOL_MATRIX.md` §Counts |
| full val split | 3817 / 6346 / 5373 = **15,536** | executed |
| pair union | 2628 / 3089 / 2902 = **8,619** | executed |

Sample ID cụ thể, dùng làm ví dụ truy vết được:

| qid | condition | shipped rank | fast rank | delta score | ghi chú |
|---|---|---:|---:|---:|---|
| `7:574` | multimodal | 1628 | 1627 | −2.98e-08 | có tie chính xác ở shipped (multiplicity 2), fast multiplicity 1 |
| `7:666` | multimodal | 42456 | 42455 | — | tie ở shipped, không tie ở fast |
| `7:690` | multimodal | 63717 | 63716 | −4.47e-08 | multiplicity 1 cả hai; rounding quyết định |
| `7:2253` | multimodal | 17475 | 17474 | — | multiplicity 1 cả hai |
| `7:4437` | multimodal | — | — | 0.6810510158538818 | order-only, **shipped có tie chính xác** |
| `7:56` | image-only | — | — | 0.5519279837608337 vs 0.551927924156189 | order-only, **fast** thành bằng chính xác |
| `7:150`/`7:99`/`7:157`/`7:179`/`7:194` | image-only | — | — | — | tie census: multiplicity 7 / 2 / 2 / 2 / 2 |
| `7:2179` | — | — | — | — | 119 query ambiguous category (`{shirt, toptee}`) |

### 6.2 Command

```bash
# Anchor run của audit equivalence (habit / fiq_n02), gallery 74381x32x256
run_id = 497c25c57fbeba885ecfb9d1

# Đường production, decode-only + thread (KHÔNG đổi run_id):
workbench/envs/ctl312h/bin/python -u workbench/scripts/composition_eval.py \
  --config workbench/docs/reviews/20261108T113610-eval/circus_cohort.yaml \
  --models intent,hint,conesep --subsets full,sf,validated \
  --topk-depth 50 --keep-features --decode-workers 8

# Đo equivalence: fast path vs shipped, có verify anchor
workbench/envs/ctl312h/bin/python workbench/scripts/audit/equivalence_fastpath.py --verify-sample

# Reproduce protocol audit (clone 16 repo tại pin)
bash workbench/docs/reviews/20261108T155252-protocol-audit/raw/clone_upstream_pins.sh "$A" /tmp/pins.tsv

# Chạy thật 1 model native trên CPU (CLIP4Cir noft):
P=/home/heheboiz/data/cir/workbench/artifacts/checkpoint_selection/20261106T002617+0700/prepared/clip4cir/fashionIQ_dataset
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 \
WORKBENCH_EVALUATOR_ENV_PASSTHROUGH=OMP_NUM_THREADS:MKL_NUM_THREADS \
WORKBENCH_CLIP4CIR_PYTHON=$PWD/workbench/artifacts/environments/clip4cir-38/bin/python \
taskset -c 0-7 workbench/envs/ctl312h/bin/python workbench/scripts/evaluate_models.py \
  --model combiner_rn50x4_noft --dataset-root "$P" --canonical-dataset-root "$P" --top-k 200
```

### 6.3 Môi trường

| vai trò | interpreter | py | torch | torchvision | CUDA |
|---|---|---|---|---|---|
| control | `workbench/envs/ctl312h/bin/python` | 3.12.11 | 2.15.0+cu132 | 0.29.2+cu132 | 13.2 |
| BLIP-2 family (habit/intent/hint/conesep) | `workbench/envs/ilearn39-adapted/bin/python` | 3.9.25 | 2.7.1+cu128 | 0.22.1+cu128 | 12.8 |
| ENCODER | `workbench/envs/encoder310-adapted/bin/python` | 3.10.20 | 2.7.1+cu128 | 0.22.1+cu128 | 12.8 |
| LIMN | `workbench/envs/limn310-adapted/bin/python` | 3.10.20 | 2.7.1+cu128 | 0.22.1+cu128 | 12.8 |
| TME | `workbench/envs/tme-adapted/bin/python` | 3.10.22 | 2.7.1+cu128 | 0.22.1+cu128 | 12.8 |
| PTHA+MTST | `workbench/artifacts/environments/cir-005ef4e1b02999a5/bin/python` | 3.10.22 | 2.7.1+cu128 | 0.22.1+cu128 | 12.8 |
| CLIP4Cir (đúng pin) | `workbench/artifacts/environments/clip4cir-38/bin/python` | 3.8.20 | 1.11.0+cpu | 0.12.0+cpu | CPU-only |
| DCNet (đúng pin) | `workbench/artifacts/environments/dcnet-377/bin/python` | 3.7.7 | 1.4.0+cpu | 0.5.0+cpu | CPU-only |
| SPRC (đúng pin) | `workbench/artifacts/environments/sprc-39/bin/python` | 3.9.25 | 2.0.1+cpu | 0.15.2+cpu | CPU-only |

**9/17** biến `WORKBENCH_*_PYTHON` được set trong `workbench.env`; **8** model không có interpreter
resolve được, và fallback `workbench/artifacts/environment_assignment.json` **không tồn tại**.
**quan sát được** (đếm lại trong lượt này: 9 biến)

NumPy trong mọi phép đo equivalence: **1.26.4**. **quan sát được**

`WORKBENCH_RUNTIME_ADAPTATION=sm_120` là **bắt buộc** cho habit/intent/hint/conesep, nếu không thì
contract lịch sử (py 3.8.10 / torch 2.1.0) từ chối interpreter 3.9 với `PYTHON_VERSION_MISMATCH`.
Hệ quả: mọi run của 4 model đó là **ADAPTED RUNTIME**, không phải reproduction nguyên bản.

### 6.4 Tolerance — khai trước, không nới sau

| lớp | đối tượng | tolerance | lý do |
|---|---|---|---|
| A. identity chính xác | qid, vị trí gallery, rank, id Top-K, **thứ tự** Top-K, membership, denominator, bit hit/miss | **0 khác biệt** | `hit_at_10` đọc `r <= 10`; một đơn vị rank ở biên là một đơn vị metric |
| B. feature float | ma trận feature/score cho thay đổi decode-only | **bit equality**; báo `max_abs`, `max_rel`, `#bytes differing` | "bit-identical" kiểm được chính xác; tolerance sẽ chỉ che vi phạm |
| C. score float khi đổi thứ tự cộng | `optimize=True`, GEMM chunk, BLAS thread | **không có tolerance** là tiêu chí pass; phán quyết lấy từ lớp A trên rank suy ra | nới float không làm rank change chấp nhận được |
| D. metric tổng hợp | R@10, R@50, nDCG, gap | báo **chính xác**, delta ở full float precision | Recall percent, nDCG fraction; cấm trộn đơn vị |
| E. thời gian | q/s, wall clock | **không phải tiêu chí pass** | — |

Cụ thể cho batch: `atol = 1e-6`, `rtol = 0` cho score batch-1 vs batch-32; `atol = 0` cho repeat
cùng process cùng input; **discrete = exact equality**.

**Sai số đo được** (tất cả nằm trong tolerance đã khai, không cái nào bị nới):

| so sánh | max abs | số liệu |
|---|---:|---|
| batch 32 vs all-at-once (score) | `0.0` | exact equal: yes |
| batch 1 vs all-at-once (score) | `6.56e-07` | exact equal: no, trong 1e-6 |
| 200 query thật, batch 1 vs 32 | `2.38e-07` | **0 rank khác** |
| 25 nhóm duplicate lớn nhất, batch 1 vs 32 | `5.36e-07` | **0 rank flip** |
| repeat cùng process | `0.0` | `exact_equal: true` |
| metric aggregation, thứ tự thuận vs ngược | `0.0` | gap `0.388704727214254` **bit-identical** cả hai chiều, cohort 6003 |

### 6.5 So sánh trung gian (intermediate) — những gì đã thực sự so, ở đâu

| so sánh | phạm vi | kết quả | file |
|---|---|---|---|
| anchor correctness của harness | 12 query/condition × 3 condition | **0 mismatch** với artifact đã lưu | `fastpath_vs_stored.json` → `anchor_verification` |
| `_ranks_from_scores` vs brute-force stable argsort | 8 case | **8/8 identical** | `differential_stages.json` |
| decode habit, worker×batch | 10 dòng đo (batch 32 × worker 1/2/4/8; batch 1/64/256 × worker 1/8) | **tất cả `bit_identical: true`**, một sha256 `5fc8002d…` | `decode_habit.json` + `decode_habit_batch.json` |
| decode encoder, worker×batch | 16 dòng (4×4 grid) | **tất cả true**, một sha256 `4f6a14f9…` | `decode_encoder.json` |
| decode LIMN, 2/4/8 worker | 512 ảnh | `max abs delta 0.0` | `THROUGHPUT.vi.md` §4 |
| hai fast path với nhau (`opt` vs `gemm1024`) | mọi query đã đổi + mọi rank mới | `identical_changed_query_sets_and_new_ranks: true` cả 3 condition | `divergence_detail.json` |
| metric aggregate stored vs opt vs gemm | full 6,003 × 3 | xem §9 | `gap_and_axes.json`, `metric_gate*.json` |
| encoder train-vs-val transform | 64 ảnh thật | pixel max 3.8269808292388916; feature max 1.2528373748064041; cosine min 0.8261280059814453 | `encoder_transform.json` |
| encoder rank impact (PROXY) | 128 ảnh gallery, 64 query | 48/64 đổi, max |Δrank| 51 | `encoder_rank_impact.json` |
| empty-text tokenization | BERT + open_clip | `[101,102]` / `[49406,49407,0×75]` | `logs/empty_text_tokens.txt`, `logs/encoder_empty_text_tokens.txt` |
| black-image tensor | 1 ảnh 342×444 | unique `[-1.7923,-1.7521,-1.4802]`, mean −1.6749 | `FINDINGS.md` D4 |
| cache reuse dưới fingerprint khác | cache habit 2.3 GB thật | sidecar fingerprint `c8aa7f0fa359b357…`; load dưới fingerprint cũ `True`, dưới post-F1 `False` | `PRE_F1_ARTIFACTS_AND_CACHE_VERDICT.md` §3 |

### 6.6 Trạng thái artifact hiện tại (đọc trực tiếp trong lượt này)

`workbench/artifacts/composition/summary/*/full.json` — **quan sát được**:

| run dir | status | queries | MM R@10 | MM R@50 | gap | `seed_applied` | BLAS policy |
|---|---|---:|---:|---:|---:|---|---|
| `habit_fiq_n02_497c25c57fbeba885ecfb9d1` | COMPLETED | 6003 | 35.61552557054806 | 54.3728135932034 | 0.388704727214254 | `null` (pre-F1) | BLAS-neutral (naive einsum, Δ = 0.0) |
| `conesep_fiq_n02_30f6de423861cc1d6c78c361` | COMPLETED | 6003 | 35.48225887056472 | 54.13959686823255 | 0.3942551173824983 | **`false`** (post-F1) | **`unknown (predates the pin)`** — GEMM path |
| `intent_fiq_n02_37ee72aa298b276d1bfda0fa` | COMPLETED | 6003 | 34.39946693320007 | 54.00633016824921 | 0.37538250032494136 | `null` (pre-F1) | BLAS-neutral (naive einsum, Δ = 0.0) |
| `hint_fashioniq_4bc9893a977e501b82378e76` | COMPLETED | 6003 | 32.816924870897886 | 51.77411294352824 | 0.43870420817223343 | **`false`** (post-F1) | BLAS-neutral (naive einsum, Δ = 0.0) |
| `tme_fashioniq_0_ad477312aa57c1b9114f9e51` | COMPLETED | 6003 | 34.01632517074796 | 53.95635515575545 | 0.3586012581618244 | `null` | BLAS-neutral (naive einsum, Δ = 0.0) |
| `ptha_mtst_cirr_transfer_315dd590eeb03d2c78d7937a` | COMPLETED | 6003 | 19.473596535065802 | 34.44944194569382 | 0.24362219946484542 | `null` | **`unknown (predates the pin)`** — GEMM path |
| `encoder_fashioniq_6a528e877c328772e2c7bb28` | COMPLETED | 6003 | 31.01782442112277 | 49.5752123938031 | 0.47230210227676583 | `null` | **`unknown (predates the pin)`** — GEMM path |
| `limn_base_iter0_all_categories_10173dc45dea8c90fd2a284b` | **PARTIAL** | 6003 | 30.064581917063222 | 48.759347382732834 | 0.5417445849251903 | `null` | **`unknown (predates the pin)`** — GEMM path |
| `habit_fiq_n02_e0dd04a46ef7a58334b9e353` | **FAILED** | 6003 | — | — | — | `null` | BLAS-neutral (naive einsum, Δ = 0.0) |
| `encoder_fashioniq_eda30a37c0bc17ac43844f09` | COMPLETED | **8** | 12.5 | 25.0 | 0.5329370636334637 | `null` | **`unknown (predates the pin)`** — GEMM path, smoke |
| `limn_base_iter0_all_categories_3f86f58483b368377e2c151e` | COMPLETED | **8** | 25.0 | 37.5 | 0.5213859188416812 | `null` | **`unknown (predates the pin)`** — GEMM path, smoke |
| `released_audit_task7_mm_any_…` | COMPLETED | 6003 | 52.84024654339497 | 71.2976844910878 | 0.3667022194996058 | `null` | n/a — imported baseline, no model run |
| `SYNTHETIC_encoder_fashioniq_fixture001` | COMPLETED | 6003 | 12.5 | 25.0 | 0.5204 | `null` | n/a — fixture |
| `SYNTHETIC_habit_fiq_n02_fixture002` | COMPLETED | 6003 | 15.0 | 30.0 | 0.2856 | `null` | n/a — fixture |

**Cột BLAS policy là bắt buộc khi đọc bảng này.** Không một run nào ở trên chạy sau khi pin ra đời
(`bf6a448`), nên **không run nào mang bản ghi policy trong artifact** — `identity` của chúng có 61
khoá và **0** khoá thread/BLAS/OMP. Vì thế 5 model GEMM-path (`encoder`, `limn`, `csmcir`,
`ptha_mtst`, `conesep`) được ghi **`unknown (predates the pin)`**: chúng **không** so sánh trực tiếp
được với một run mới dưới policy hiện tại, và mọi ΔnDCG đo được của chúng chỉ nên đọc như **bậc độ
lượng** (1e-8 … 8e-5), không phải hằng số. 4 model naive-einsum (`habit`, `intent`, `hint`, `tme`)
**không** chịu ảnh hưởng của BLAS threading — contraction của chúng không có đường BLAS, đo phẳng ở
1/2/4/8/10 thread, delta **đúng 0.0** — nên nhãn của chúng giữ nguyên. Lý do và số đo:
`20261108T-throughput/BLAS-POLICY.vi.md`.

Hai dòng `SYNTHETIC_*` là fixture; hai dòng `query_count: 8` là smoke. **Cả bốn bị loại khỏi mọi
claim metric**, theo đúng yêu cầu của `MODEL_INVENTORY.md` §7.

Ba dòng **`intent`**, **`hint`**, **`conesep`** là run mới (timestamp file 16:38 / 17:07 / 17:42 cùng
ngày) với `topk_export.status: TOPK_EXPORT_COMPLETE`, `depth: 50`, `missing_query_count: 0`, và
`reference_eligibility` đo được — nhưng **không báo cáo nào trong corpus nguồn nhắc tới chúng**:
`RUNS.vi.md` ghi `intent` = QUEUED, `hint` và `conesep` = NOT_RUN. **quan sát được**. Đây là điều báo
cáo này ghi nhận như một cập nhật, không như một phép đo do nó thực hiện.

Lưu ý về `conesep`: nó là dòng duy nhất trong nhóm này có `checkpoint_training_noise_pct: 20`, nên
con số `35.48225887056472` **không** phải clean reproduction — nhãn noise phải đi cùng nó.

---

## 7. Defect đã xác nhận, cách sửa, kết quả cũ bị ảnh hưởng, cache invalidation, trạng thái chạy lại

### 7.1 F1 — `--seed` là trang trí (CONFIRMED_DEFECT, HIGH, disclosure-grade) → **ĐÃ SỬA**

- **Hiện tượng:** `--seed` đi vào `run_composition`, được hash vào `identity`
  (`runner.py:944`, `"seed": seed`) nên tạo `run_id` khác, nhưng **không chỗ nào trong đường
  composition áp nó**. `grep manual_seed|np.random.seed|random.seed|deterministic` trên
  `runner.py`, `adapters.py`, `composition_eval.py` → **0 lời gọi**, chỉ có comment. **quan sát được**
- **Hệ quả ngược:** `run_id` **không** bind RNG state thật sự đã tạo feature.
- **Chọn cách sửa (b) bỏ khỏi `identity`**, không chọn (a) seed thật: không có gì rút ngẫu nhiên ở
  encode time. Bằng chứng: mọi transform là `TargetPad → Resize(BICUBIC) → CenterCrop → Normalize`;
  mọi adapter `eval()` + `no_grad()`; `grep '\.generate(\|do_sample\|num_beams\|multinomial\|bernoulli'`
  trên `workbench/composition/` **rỗng**; các draw trong third-party đều `self.training`-gated
  (HINT `np.random` dưới `if self.training`; ConeSep `torch.randn` trong `forward()` ở loss path;
  LIMN `torch.bernoulli` bị comment; INTENT `import random` không dùng). Thứ duy nhất thật sự draw là
  `metrics.paired_bootstrap_gap` với `seed` là **tham số**, và runner **không gọi** nó. **quan sát được**
- **Đã áp:** bỏ `"seed": seed` khỏi `identity` kèm comment lý do; summary vẫn giữ `"seed"` cho
  provenance và thêm `"seed_applied": False`; help text CLI sửa từ "seed recorded for deterministic
  summaries" thành câu nói rõ đây là provenance, **không** đổi run id. Giữ tên flag `--seed` vì nó
  được tham chiếu ở `CIRCUS_COMPOSITION_SPEC.md`, `A38_FEATURE_MATRIX.csv`, baseline transcript.
- **Test:** `test_audit_fix_f1_seed_provenance.py`, 4 check. Revert-proof: **3 failed, 1 passed** khi
  gỡ fix; **4 passed** khi có fix.
- **Kết quả cũ bị ảnh hưởng: KHÔNG có số nào.** `seed` chưa bao giờ được đọc bởi computation, nên bỏ
  nó khỏi `identity` không thể đổi kết quả. Cái bị mất là **addressability**: run id cũ không tái
  tính được nữa. Đây là kết luận quan trọng nhất và nó đã được ghi rõ trong verdict gốc.
- **Cache invalidation: CÓ, hai khoá độc lập.** (1) đường dẫn `runs/<run_id>/work/<model>/features`;
  (2) sidecar `fingerprint` = `_digest(identity)`. Đo trên cache thật 2.3 GB của habit: load dưới
  fingerprint cũ `True`, dưới post-F1 `False`. Không có copy/symlink/`--resume` nào adopt được.
- **Trạng thái chạy lại:** **không bắt buộc cho tính đúng của số**; bắt buộc nếu muốn một cây
  artifact khoá thống nhất. Đánh dấu pre-F1 **không cần đổi tên**: mọi summary pre-F1 có
  `seed_applied: null`, post-F1 có `false` — marker tự mô tả.
- **Đo được:** 14 summary trên đĩa có đúng **2** cái `seed_applied: false` —
  `hint_fashioniq_4bc9893a…` và `conesep_fiq_n02_30f6de42…`; 12 cái còn lại `null` (pre-F1, gồm cả
  fixture và run FAILED). Tức gần như toàn bộ artifact hiện có là pre-F1. **quan sát được**

### 7.2 F3 — cache gallery nhận mà không kiểm shape (CONFIRMED_DEFECT, MEDIUM) → **ĐÃ SỬA**

- **Hiện tượng:** `_encode_gallery` gọi `_load_memmap(destination, fingerprint)` **không** shape, còn
  `_encode_queries` truyền `(len(queries), dimension)`; `_load_memmap` chỉ enforce shape khi được cho.
- **Cái audit gốc nói chưa đúng:** `.npy` bị truncate **đã** bị từ chối — NumPy raise
  `ValueError: mmap length is greater than file size`, `_load_memmap` bắt và biến thành cache miss.
  Đo lại trực tiếp:

  ```
  valid, không shape arg                -> ACCEPTED shape=(200, 4)
  truncated body, header intact         -> REFUSED(None)   <- NumPy, không phải ta
  complete 100 rows (gallery 200)       -> ACCEPTED shape=(100, 4)   <- lỗ thật
  complete 100 rows, shape (200,4)      -> REFUSED(None)
  ```
- **Lỗ thật còn lại:** cache **đọc được nhưng sai số hàng** — gallery ghi dở mà sidecar cũng bị ghi
  lại, hoặc cache copy giữa các run. Mọi hàng sau chỗ thiếu mô tả một ảnh khác, triệu chứng duy nhất
  là điểm xấu đi.
- **Đã áp:** `_memmap_metadata_shape(path)` + trong `_encode_gallery`:
  `expected_shape = (len(image_paths), *metadata_shape[1:])`. Số hàng lấy từ shape **đã ghi** của
  sidecar ghép với kích thước gallery của run này; độ rộng feature lấy từ sidecar vì chưa biết trước
  khi encode.
- **Test:** `test_audit_fix_f3_gallery_cache_shape.py`, 3 check (short-cache không được dùng lại;
  cache đúng shape vẫn được dùng lại; truncate thành cache miss chứ không raise).
- **Kết quả cũ bị ảnh hưởng:** không đổi số. Nhưng một artifact pre-fix có cache sai kích thước **có
  thể** đã được tính từ ma trận sai — đó là caveat về tính toàn vẹn dữ liệu của artifact đó, không
  phải thay đổi identity.
- **Cache invalidation: KHÔNG.** Fingerprint và identity không đổi; fix chỉ đổi việc có **tin** file
  cache hay không.
- **Trạng thái chạy lại:** không cần.

### 7.3 F4 — ENCODER hard-code `checkpoint_mapping_status` (CONFIRMED_DEFECT, MEDIUM) → **ĐÃ SỬA**

- **Hiện tượng:** `model_encoder.py:302-303` (worker, không phải parent) dựng checkpoint record với
  `"checkpoint_mapping_status": "VERIFIED_METADATA"` **literal**. Literal đó là **một trong bốn gate**
  của `_unused_checkpoint_keys` (`replay/encoder.py:73-102`), quyết định hai tensor
  `binding_decoder.fc.0.*` (đã chứng minh không được đọc, `max_abs_diff 0.0` khi perturb) có được bỏ
  qua im lặng hay không. ⇒ registry hạ status **không** làm adapter chặt hơn.
- **Đã áp:** 2 hunk — (1) `_run_job` payload thêm
  `"checkpoint_mapping_status": self._checkpoint.get(...)`; (2) worker đọc `job.get(...)` thay literal.
  Dùng `.get` chứ không `[...]`: job thiếu key → `None`, và `None != "VERIFIED_METADATA"` ⇒ **từ chối**.
  Fail-closed có chủ ý.
- **Kiểm chứng (không phải giả định):** registry thật sự có giá trị đó
  (`encoder/per_query_checkpoint → fashioniq → VERIFIED_METADATA`, sha `0f31c501…9173`); gate thật sự
  keyed trên field đó; file sau patch `ast.parse` ok và literal đã biến mất; hành vi đổi đúng:

  ```
  VERIFIED_METADATA        permitted=['backbone.binding_decoder.fc.0.bias', '...weight']  ACCEPTED
  downgraded UNVERIFIED    permitted=[]                                                    REFUSED
  absent (None)            permitted=[]                                                    REFUSED
  ```
- **Test:** 5 check, trong đó 3 check đường **downgrade** (happy-path-only sẽ pass dù literal còn).
  Bằng chứng không vacuous: khôi phục chỉ literal của worker → `REACHED_PAST_MAPPING_GATE`, tức
  worker **bỏ qua `UNVERIFIED`** đúng như defect. Với patch: 5 passed.
- **Kết quả cũ bị ảnh hưởng: không đổi số.** Registry status là `VERIFIED_METADATA`, cùng trọng số
  được nạp, cùng hai tensor được bỏ qua.
- **Cache invalidation cho ENCODER: CÓ, vì lý do khác** — `runner._module_source_hash` hash chính file
  adapter (`runner.py:802-808`) vào `identity.adapter_module_source_hash`. Đo ngay sau F4:
  `model_encoder.py` `3f4efc789448106478f8c73386b0669be4d85167bcde033e2c244da556d70687`, ở HEAD
  `1a8b0dd25827ca0ba90f504d81f2fa190cede1ee3fb3bfe335e70ab359005823`. **Sau F7 (cùng file) giá trị
  hiện tại là `bf60d855be3d72dd2ff0abc6d8b9dc12dbb5fad4a50aa87aaa774454f65a76ab`** — re-hash trong lượt
  viết báo cáo này. Vì hash tính từ nội dung file lúc chạy, F4 rồi F7 **chỉ mint một** `run_id` mới;
  trạng thái trung gian `3f4efc…` không bao giờ sinh run nào. Đây là hệ quả namespace của source hash,
  không phải hệ quả số học.
- **Ghi nhận bất nhất trong chính registry:** cùng một block vừa nói `checkpoint_mapping_status:
  VERIFIED_METADATA` vừa nói `reproduction_status: METRICS_MEASURED_MAPPING_UNVERIFIED` và "direct
  file URL and SHA remain unverified". Hai nhãn trung thực về hai thứ khác nhau; ai chỉ đọc nhãn đầu
  sẽ tin mapping quá mức. **quan sát được**

### 7.4 F6 — comment LIMN nêu lý do sai (CONFIRMED_DEFECT, LOW, tài liệu) → **ĐÃ SỬA**

- `replay/limn_observer.py:162-169` nói tie order tái lập được vì "argsort is stable in NumPy".
  **Sai**: `np.argsort` default là `'quicksort'` (introsort), **không stable**. Kiểm bằng chính
  interpreter dùng ở đây: numpy 1.26.4, `inspect.signature` là `(a, axis=-1, kind=None, order=None)`,
  docstring nói *"The default is 'quicksort.'"*. **quan sát được**
- **Hành vi vẫn đúng** vì `limn_observer.py:170` phát **cùng lời gọi trên cùng input** như native
  `test.py` (`np.argsort(-sims[i, :])`, `third_party/LIMN/LIMN/test.py:85,155,244`), và lời gọi đó
  deterministic với một dtype/kích thước cho trước. Chỉ **lý do** sai.
- **Đã áp:** comment thay bằng câu đúng, cộng cảnh báo rằng làm `kind` explicit — hoặc chỉ làm ở một
  phía — sẽ đổi tie order và phá chính reproduction mà comment đó tồn tại để biện minh. AST của file
  **byte-identical** trước và sau; 6 module test LIMN pass (67 test cùng các file F1/F3/F4).
- **Kết quả cũ bị ảnh hưởng:** không. **Cache invalidation: KHÔNG** — `limn_observer.py` không nằm
  trong `identity` (`grep limn_observer runner.py adapters.py` rỗng), nên run id của LIMN chỉ đổi vì F1.

### 7.5 F7 — ENCODER và PTHA+MTST **không chạy được từ HEAD** (blocker chức năng) → **ĐÃ SỬA**

Đây là phát hiện nặng nhất của cả chương trình kiểm toán, và nó **không** nằm trong danh sách defect
ban đầu — nó được tìm ra khi viết test cho F4.

- **Hiện tượng:** hai adapter gọi helper song song hoá decode mà **không import**, nên raise `NameError`
  **bên trong `_worker_main`**, và `except Exception` của chính worker biến nó thành
  `reply(False, error_type="NameError", ...)`. Run chết ở bước nạp model và **trông như lỗi model**.

  | adapter | call site | helper | import? |
  |---|---|---|---|
  | `model_encoder.py` | `:330` | `limit_torch_threads()` | **không** |
  | `model_encoder.py` | `:331` | `resolved_decode_workers()` | **không** |
  | `model_encoder.py` | `:353` | `prefetch_rows(...)` | **không** |
  | `model_ptha_mtst.py` | `:320` | `limit_torch_threads()` | **không** |
  | `model_ptha_mtst.py` | `:321` | `resolved_decode_workers()` | **không** |
  | `model_ptha_mtst.py` | `:345` | `prefetch_rows(...)` | **không** |

  Mọi số dòng trên được verify lại trong file trước khi sửa, không lấy từ báo cáo.
- **Nguyên nhân:** thay đổi decode song song thêm 3 helper vào nhiều adapter; 5 cái (`habit`, `intent`,
  `hint`, `conesep`, `tme`) nhận **cả call và import**; `limn` cũng vậy (import nhiều dòng `:45-47`);
  `encoder` và `ptha_mtst` chỉ nhận **call**.
- **Đã áp:** thêm 3 tên vào block `from workbench.composition.adapters import (...)` sẵn có ở cả hai file.
- **Audit tĩnh toàn bộ 9 adapter** (tập helper suy từ public callable của `adapters.py`, 15 tên):
  **đúng hai file hỏng; cả 9 giờ sạch.** `model_csmcir` không có decode pool nên không thuộc lớp lỗi này.
- **Bằng chứng runtime, không phải đọc:** trong `encoder310-adapted`, `limit_torch_threads() -> 10`,
  `resolved_decode_workers() -> 10`, `prefetch_rows` callable; và trong `_worker_main.__globals__` cả
  sáu cặp `(module, helper)` trả `in globals=True callable=True` và **`is adapters.X`** (tức dùng
  chung implementation, không phải bản sao).
- **Test:** `test_audit_fix_encoder_helpers_imported.py`, 23 check, tham số hoá trên 9 adapter, dùng
  phân tích `symtable` (walk AST cho false positive trên `reply`/`transform` vì chúng bound ở enclosing
  scope). Revert-proof: gỡ import encoder → **3 FAILED**; gỡ import ptha → **4 FAILED**; cả hai khôi
  phục → **23 passed**.
- **Vì sao test cũ không bắt được:** `test_circus_adapters_*.py` import module và construct `Adapter()`,
  **không bao giờ vào `_worker_main`**. `NameError` trong thân hàm là invisible cho tới khi dòng đó chạy.
  Phải kiểm tĩnh.
- **Kết quả cũ bị ảnh hưởng:** ENCODER (`31.01782442112277 / 49.5752123938031`) và PTHA+MTST
  (`19.473596535065802 / 34.44944194569382`) phải gắn nhãn **"pre-generated artifact, not runnable from
  HEAD"**. Số **không sai** — code tạo ra chúng đã đúng — nhưng tree hiện tại **không thể** sinh lại
  chúng, và không được trình bày như thể một run mới sẽ tái tạo được.
- **Cache invalidation:** cả hai module hash đổi ⇒ re-key.
  `model_encoder.py` hiện `bf60d855be3d72dd…` vs HEAD `1a8b0dd25827ca0b…`;
  `model_ptha_mtst.py` hiện `768bc3ae9a2bd0af…` vs HEAD `546d1a52d2b6ca58…`.
  Với ENCODER, F4 + F7 nằm **cùng một file** nên chỉ mint **một** run id mới (hash tính từ nội dung
  file lúc chạy); trạng thái trung gian không mint id nào vì không có run nào được tạo từ chúng.
- **Trạng thái chạy lại:** blocker đã gỡ, **chưa re-run**. Nhãn "pre-generated artifact" **vẫn còn
  hiệu lực** cho tới khi có run mới dưới id mới.

### 7.6 Các phát hiện equivalence (không phải defect của code sản phẩm)

**(a) `optimize=True` / chunked GEMM đổi 207/18,009 rank** — xem §9. Đây là **quyết định bị từ chối**,
không phải bug được sửa.

**(b) 851 nhóm gallery trùng byte** — `_ranks_from_scores` đếm `1 + #greater + #earlier-equal`
(`adapters.py:352-368`), nên rank phụ thuộc thứ tự gallery tại tie. Gallery thật có **851 nhóm trùng
byte-identical, phủ 2,386 vị trí, nhóm lớn nhất 54, 1,535 hàng dư**. Đã verify chúng là **file nguồn
giống hệt nhau dưới product ID khác nhau** (sha256 của `.png` bằng nhau 8/8 nhóm kiểm; pixel
correlation 1.0; mean-abs-diff 0.0). Một pass resolve độc lập: 74,381 identity → 74,381 file, 0
collision.

Tie census 200 query mỗi condition:

| condition | positive bị tie | rank phụ thuộc tie order | top-1 bị tie | query có duplicate score |
|---|---:|---:|---:|---:|
| multimodal | 2/200 | 2/200 | 0 | 200/200 |
| text_only_black_image | 2/200 | 2/200 | 0 | 200/200 |
| image_only_empty_text | 7/200 | 5/200 | 0 | 200/200 |

⇒ ước lượng ~60–150 rank/condition phụ thuộc tie order trên 6,003 query — khớp với việc 207 rank change
tập trung đúng ở các nhóm gần-trùng. **Đây là thuộc tính của dữ liệu phát hành**, không sửa được
nếu không đổi gallery đã pin. Nó làm **tăng** recall cho query nằm trong nhóm trùng, và **hiện không
được nêu ở bất kỳ summary nào**.

**(c) 7 input vắng khỏi cache identity** — xem §7.7. **Một phần đã đóng**: BLAS thread count và
`scoring_module_source_hash` nay là field identity; còn 5 vắng.

**(d) 19 cửa sổ Top-K đổi thứ tự** — xem §9.

### 7.7 Cache identity thiếu input (DEFECT, **đã đóng một phần**)

> **Trạng thái cập nhật.** Verdict gốc là "7 input vắng". Sau đó **hai** trong số đó đã đóng:
> `blas_threads_pinned` (`runner.py:1198`) và `scoring_module_source_hash`. **Năm vẫn vắng.** Đây vẫn
> là **DEFECT**, không phải PASS: một defect được đóng một phần không trở thành gate xanh.

`_load_memmap` chấp nhận cache khi **sidecar `.json`** có `fingerprint` bằng digest kỳ vọng và `.npy`
tồn tại đúng shape. Sidecar giữ **đúng** `dtype`, `fingerprint`, `shape`. **Không có gì hash nội dung
mảng.** ⇒ copy `.npy` rồi viết lại sidecar là được nhận, không kiểm tra số.

| input | trạng thái | ghi chú |
|---|---|---|
| **BLAS / OMP / MKL intra-op thread count** | **ĐÃ ĐÓNG** | `blas_threads_pinned: 1` trong `identity` (`runner.py:1198`), và pin thực thi trong `_rank_groups_concurrently`. Run cũ ghi `unknown (predates the pin)` — xem §6.6 |
| **`scoring_module_source_hash`** | **ĐÃ ĐÓNG (một phần)** | có trong `identity` của run mới; run `497c25c57fbeba885ecfb9d1` **vẫn vắng** field này ⇒ nay báo `measurement_identity_confidence: partial`, không còn bị gọi là `duplicate`. Vẫn chỉ hash **một** file (`adapters.py`) |
| tf32 / cudnn benchmark / deterministic flags | **VẪN VẮNG** | không ghi; có thể đổi bit forward |
| GPU compute capability / driver / CUDA version | **VẪN VẮNG** | chỉ **tên** device được ghi; capability quyết định kernel thì không |
| code và tham số thật của transform | **VẪN VẮNG** | `preprocessing.registry_values` ghi **nhãn** registry (`fashioniq_layout`, `image_size`, `transform`), không ghi source của transform bên thứ ba ⇒ sửa bên trong hàm transform không bị phát hiện |
| **nội dung mảng feature** (byte-level) | **VẪN VẮNG** | không có verify byte-level ở đâu; fingerprint là của identity, không phải của bytes. Sidecar chỉ kiểm fingerprint, **không bao giờ** kiểm bytes |
| decode worker count | **cố ý không ghi** | decode bit-identical (đã đo lại 3 họ transform) — hợp lý, nhưng nó vẫn là một input. Ghi nhận, không sửa |

Vẫn còn mở, và vẫn là defect: **851 hàng feature gallery trùng nhau trên 2386 vị trí** (nhóm lớn
nhất 54) làm tie mang tính hệ thống — xem G3c. Trùng ở mức bytes trong gallery đã pin thì rank phụ
thuộc thứ tự, và không field identity nào sửa được điều đó.

Hệ quả còn lại: (a) viết lại sidecar làm số cũ trông mới; (b) đổi `score()` trong `model_*.py` mint
run id mới, nhưng đổi một helper dùng chung **ngoài** `adapters.py` thì không; (c) ~~đổi BLAS thread
count đổi được điểm fast-path dưới một identity không đổi~~ — **đã đóng**.

---

## 8. Baseline vs optimized, đo được

### 8.1 Batch size — manh mối sai, bị bác bỏ bằng số

Slice thật 2,000 ảnh gallery, checkpoint `habit` `fiq_n02`:

| batch | img/s (fp32) | peak alloc | peak reserved |
|---:|---:|---:|---:|
| 64 | 17.6 | 3484 MiB | 3958 MiB |
| 128 | 18.0 | 4350 MiB | 5066 MiB |
| 256 | 17.9 | 6082 MiB | 7306 MiB |
| 512 | 18.0 | 9544 MiB | 11788 MiB |

`img/s` **không đổi** từ 64 đến 512 trong khi VRAM tăng 3484 → 9544 MiB. Forward thuần GPU:

| batch | forward | img/s | peak alloc |
|---:|---:|---:|---:|
| 32 | 381 ms | 83.9 | 3053 MiB |
| 64 | 763 ms | 83.9 | 3486 MiB |
| 128 | 1538 ms | 83.2 | 4354 MiB |
| 256 | 3028 ms | 84.6 | 6090 MiB |

⇒ **bão hoà từ batch 32**. VRAM dư ~11 GB lúc chạy thật không phải chỗ lãng phí cần lấp — đó là chỗ
trống vì GPU không có việc. **quan sát được**

bf16: 17.7 img/s (b64), 16.2 (b256) so với fp32 17.6 / 17.9 — **không nhanh hơn, chậm hơn ở batch lớn**.
Không áp dụng. **quan sát được**

### 8.2 Nút cổ chai thật: decode + thread

```
decode only  512: 0.30s    (~1700 img/s)
transform only 512: 19.56s (~26 img/s)      <- torch 20 thread
transform only 512 (1 thread): 0.38s (~1350 img/s)
```

Ảnh FashionIQ nhỏ (8–28 KB, ~342×407) nên mỗi op trong transform quá nhỏ để bù chi phí launch thread.
Song song hoá decode (giữ nguyên thứ tự lô):

| workers | wall | img/s |
| workers | wall (s) | img/s |
|---:|---:|---:|
| 1 | 1.66 | 618 |
| 2 | 1.31 | 780 |
| 4 | 0.61 | 1672 |
| 8 | 0.47 | 2192 |
| 10 | 0.53 | 1915 |

> **Bảng này là PROSE-ONLY, KHÔNG có raw file — không được trích như số đo.** Nó cũng không tái lập được về mặt số học: `wall × img/s` ở mỗi dòng cho ~1015–1030 ảnh, tức **~1024 ảnh/dòng**, khác với `vdecode.txt` (512 ảnh, 109.1 img/s serial, tối ưu **682.7 img/s ở 4 worker**). Hai thứ là hai thí nghiệm khác nhau, và `2192` **không** phải điểm tốt nhất mà `vdecode.txt` đo được. Số có raw để trích: **682.7 img/s ở 4 worker**.

A/B cuối, cùng 512 ảnh, cùng transform, cùng checkpoint, **chỉ decode+transform** (không forward):

| | img/s | quy ra 74,381 ảnh |
|---|---:|---:|
| **BEFORE** (1 worker, torch 20 thread) | 21.2 | 58.5 phút |
| **AFTER** (8 worker, torch 8 thread) | **2089.2** | **0.6 phút** |

⇒ **98× — nhưng đây là decode+transform, KHÔNG phải cả model.** Đây là con số dễ bị trích sai nhất
trong toàn bộ tài liệu này.

A/B riêng cho `intent` (transform của chính nó, checkpoint thật): BEFORE 30.3 img/s → AFTER **2291.9
img/s**, delta 0.0, cũng **decode+transform**.

### 8.3 End-to-end thật: ~5×, không phải 98×

- Pipeline đầy đủ (decode + transform + forward GPU): **17.6–18.0 img/s ở batch 64–512**, đo trên
  2,000 ảnh thật.
- Forward GPU bão hoà ~84 img/s ⇒ ≈886 s cho 74,381 ảnh ≈ **15 phút**, và đây là **sàn cứng**.
- Trước: **~73 phút** cho 74,381 ảnh. Sau: **~15 phút**. ⇒ **~5×**.

Đừng bấm giờ `intent` rồi thấy số khác mà tưởng tài liệu sai: **98× là decode, 5× là cả model**.

### 8.4 Bit-identity của decode — đo theo TỪNG họ transform, không suy từ HABIT

Delta 0.0 của HABIT **không** suy ra được cho model khác. Đo riêng, 512 ảnh thật, worker 2/4/8:

| họ | transform | max abs delta | bit-identical |
|---|---|---:|---|
| HABIT / INTENT / HINT / TME / CONESEP / PTHA | `targetpad_transform(1.25, 224)` (lavis) | **0.0** | ✅ |
| ENCODER | `open_clip` `preprocess_val` (`image.copy()`) | **0.0** | ✅ |
| LIMN | `open_clip.image_transform(224, …)` (`image.copy()`) | **0.0** | ✅ |

`CSMCIR` không đo được theo cách này vì nó nạp **cả gallery vào một tensor** trong **một** job (không
có vòng lặp lô để song song hoá). Bản thân `model_csmcir.py` **không được sửa** decode.

### 8.5 Song song theo trục query — ĐƯỢC CHẤP NHẬN

Khác bản chất với `optimize=True`: mỗi query là một tích vô hướng **độc lập** trên cùng gallery, nên
chạy nhiều query đồng thời **không thể** đổi thứ tự cộng dồn bên trong bất kỳ query nào.

**⚠️ PROSE-ONLY, KHÔNG CÓ RAW FILE.** Bảng dưới đây **không** có file raw kèm theo — xem §8.8. Con số
duy nhất trong bảng này có bằng chứng raw là **cột "rank identical"**, và nó đến từ
`20261108T-throughput/query.txt` (512 query, 0 mismatch). Các cột `wall` / `q/s` / `speedup` chỉ tồn
tại trong prose và **không được trích như số đo**. Phép đo authoritative cho `WORKBENCH_SCORE_THREADS`
là `test_query_parallel_ranking_is_identical_to_serial`, nó khẳng định tính đồng nhất chứ không đo
throughput.

Đo trên chính đường production (`rank_in_query_chunks` trong `runner.py`), 512 query thật, so với
**rank đã lưu của run thật**:

| threads | wall | q/s | speedup | 6003 query | rank identical |
|---:|---:|---:|---:|---:|---|
| 1 | 106.0 s | 4.8 | 1.00× | 20.7 phút | chuẩn |
| 2 | 58.7 s | 8.7 | 1.81× | 11.5 phút | ✅ |
| 4 | 32.6 s | 15.7 | 3.26× | 6.4 phút | ✅ |
| **8** | **18.7 s** | **27.4** | **5.67×** | **3.7 phút** | ✅ |

**0 mismatch trên 512 query** so với artifact đã publish — không phải so với bản tính lại.

Tương tác BLAS: hai cơ chế **không còn tranh nhau** — chúng đã được tách bằng code. `runner.py` ghim
`BLAS_THREADS_PINNED = 1` (`runner.py:597`) và `_rank_groups_concurrently` gọi
`_blas_thread_limit(BLAS_THREADS_PINNED)` **một lần, bao ngoài cả hai nhánh** (serial và concurrent,
`runner.py:775`, restore trong `finally`), ngay trước khi fan out nhóm query. Nên BLAS đa luồng bên
trong một group worker không xảy ra: song song chỉ có ở cấp nhóm. Việc đặt toggle **bên trong** worker
đã được đo là đổi rank ở 4 và 8 thread (một worker restore giá trị cũ trong khi worker khác còn đang
tính), nên toggle được hoist ra ngoài — đó là lý do của vị trí hiện tại.

Ghi chú cũ trong `composition_eval.py` nói 6.2 q/s là do "OpenBLAS baked in at import"; điều đó đã
được sửa. Cơ chế thật: `np.einsum("nd,gqd->ngq")` naive **không có đường BLAS** (vòng lặp numpy tự
viết), nên đổi `OPENBLAS_NUM_THREADS` 1→10 **không đổi** tốc độ (6.5/6.0/6.2/5.6/6.2 q/s — phẳng
trên cả 5 mức, đây là bằng chứng loại trừ lỗi pool-sizing muộn). Với họ GEMM-path thì ngược lại: số
thread BLAS đổi **kết quả**, không chỉ thời gian. Lý do đầy đủ và bảng theo từng họ adapter:
**`20261108T-throughput/BLAS-POLICY.vi.md`**.

Vị trí code: `rank_in_query_chunks`, `score_in_query_chunks` và `_rank_groups_concurrently` trong
`workbench/composition/runner.py`, **không sửa `model_*.py`**.

Về run identity: `runner.py` **vẫn không được hash toàn bộ** vào `identity` — cố ý, để mỗi lần sửa
một comment trong file đó không mint run id mới và vô hiệu hoá mọi cache. Nhưng **hai input số học
thực sự quan trọng thì nay là field identity tường minh**, nên câu "`run_id` không đổi" **không còn
đúng**:

| field | dòng | vì sao phải là identity |
|---|---|---|
| `blas_threads_pinned` | `runner.py:1198` | đổi policy BLAS đổi điểm score ⇒ đổi rank suy ra được. Không ghim thì `--resume` tái dùng rank cũ trong khi lần chạy mới sinh rank khác |
| `query_scope_resolution` | `runner.py:1170` | bản đồ scope theo query quyết định **ma trận gallery nào** mỗi query được chấm. `query_order_identity` hash `query.category` của loader, vốn `None` đúng ở những query mà resolver scope — nên thiếu digest này thì wire resolver xong `run_id` vẫn y nguyên và `--resume` tái dùng 119 null cũ |

Hệ quả phải nói rõ: bật/tắt hai thứ trên, đổi `batch_size` (hình dạng nhóm là một phần của số học),
hoặc đổi scope map **đều mint run id mới**. Artifact đã publish dưới policy cũ vì thế **không** so
sánh trực tiếp được với run mới mà không nêu nhãn policy — xem bảng ở §6.6. Nút
`WORKBENCH_SCORE_THREADS` thì ngược lại: nó chỉ đổi **concurrency**, không đổi kích thước nhóm, nên
không đổi kết quả và không đổi `run_id` (`test_group_size_does_not_depend_on_thread_count`).

### 8.6 `optimize=True` / chunked GEMM bị TỪ CHỐI — tới 17.8× nhưng đổi rank

Điểm số học trên CPU là nút cổ chai lớn thứ hai: log cohort cho thấy `cpu=+60s/min` = **đúng một core
trên 20** suốt pha scoring, trong khi GPU nằm ở 852 MiB.

| cách | q/s | 6003 query | max abs delta |
|---|---:|---:|---:|
| `einsum` hiện tại (production) | 5.6 | 18.0 phút | 0 (chuẩn) |
| `einsum(..., optimize=True)` | 25.7 | 3.9 phút | 3.576e-07 |
| chunked GEMM (chunk 4096) | **99.7** | **1.0 phút** | 3.576e-07 |

Đo ở `OPENBLAS_NUM_THREADS=8` (`ein_threads.txt`). Tức GEMM nhanh **17.8×**, `optimize=True` **4.6×**.
Con số **"18×"** xuất hiện trong yêu cầu và trong `THROUGHPUT.vi.md` §5 là **của chunked GEMM**
(`99.7 / 5.6 = 17.80`), **không phải** của `optimize=True` — `optimize=True` chỉ **4.59×**
(`25.7 / 5.6`). Cả hai đều bị từ chối vì cùng lý do, nhưng đừng gán 18× cho `optimize=True`.
Ngoài ra con số "27–43 q/s" trong `EQUIVALENCE.md:45-46` **không khớp**
`ein_threads.txt` ở 2 thread (22.1 q/s cho `opt`, 56.2 cho GEMM) — xem §8.8.

**Câu hỏi quyết định là METRIC, và câu trả lời là KHÔNG** — xem §9. Ba lý do cụ thể, không giáo điều:

1. **207 query đổi rank là artifact đã công bố.** Top-K candidate được trả cho UI; đổi rank nghĩa là
   người dùng thấy thứ tự khác, dù Recall tổng không đổi.
2. **0 query vượt biên K=10/K=50 ở cohort này — nhưng đó là quan sát, không phải bảo đảm.** Query nào
   có rank đúng 49/50/51 sẽ vượt biên chỉ vì 1 ULP; không có gì trong phép tính ngăn điều đó.
3. **`run_id` không đổi nếu chỉ sửa thân `score()`**, nên một rank bị lật sẽ nằm im trong artifact cũ
   — loại lỗi tệ nhất vì không ai truy được.

**Quyết định: giữ `np.einsum("nd,gqd->ngq").max(-1)` làm mặc định cho cả 9 adapter.** Đã áp rồi
revert; `grep -c optimize=True` = **0** ở habit/intent/hint/tme. Ba adapter còn lại
(conesep/csmcir/ptha_mtst) **vốn đã có** `optimize=True` từ trước — không phải thay đổi của lượt này,
và không được sửa.

### 8.7 Hai model không chạy song song được

`habit` encode đạt **8439 MiB** peak. Hai model ≈ 16.9 GB > 16.3 GB VRAM. Không vừa ⇒ tuần tự hoá là
đúng.

### 8.8 Chỗ số liệu throughput không truy vết được

Đây là phần **phải nói rõ**, vì nó là chỗ dễ trích sai nhất:

| con số | nằm ở đâu | vấn đề |
|---|---|---|
| `21.2 img/s` (BEFORE) | chỉ có trong prose `THROUGHPUT.vi.md` | **không có file raw**. `vdecode.txt` đo serial **109.1 img/s** (512 ảnh / 4.69s). Hai số chỉ tương thích nếu "21.2" là cấu hình 1 worker + torch **20 thread** còn "109.1" là cấu hình khác — nhưng không file nào ghi cấu hình của 109.1 |
| `2089.2 img/s` (AFTER) | chỉ prose | **không có raw file**. `vdecode.txt` (512 ảnh, decode+transform) cho tối đa **682.7 img/s ở 4 worker**; không có dòng 2089.2 ở bất kỳ file nào. `2192 img/s` cũng không có raw — nó chỉ nằm trong bảng prose ~1024 ảnh/dòng ở trên |
| bảng 4 thread (106.0 / 58.7 / 32.6 / 18.7 s) | chỉ prose | **không có file raw**, và `equivalence` audit nói thẳng là **nó không đo** `WORKBENCH_SCORE_THREADS` |
| "27–43 q/s `optimize`, 70–75 q/s GEMM **dưới `OPENBLAS_NUM_THREADS=2`**" | prose | `ein_threads.txt` ở 2 thread ghi `optimize` **22.1** và `gemm` **56.2**. Hai số không khớp |
| `seconds` trong `metric_gate*.json` (`opt` 66.5 s cho 6003 query multimodal) | JSON | 66.5 s cho 6,003 query = **~90 q/s**, trong khi `ein_threads` đo `optimize=True` ở 22–34 q/s. Một trong hai phép đo dùng cấu hình khác (khả năng cao: chunk/thread khác), nhưng **không file nào nói** |

⇒ Dùng các con số **có file raw** khi trích: `batch.txt`, `ein_threads.txt`, `vdecode.txt`,
`metric_gate.json`, `eq_fast.txt`. Các con số chỉ có trong prose phải ghi kèm "prose-only, không có
raw file". **quan sát được**

---

## 9. Full-evaluation equivalence — số mismatch chính xác

### 9.1 Cấu hình đo

| trường | giá trị |
|---|---|
| run anchor | `497c25c57fbeba885ecfb9d1` (habit / `fiq_n02`) |
| gallery | `74381 × 32 × 256` |
| query | `6003 × 256` × **3** condition = **18,009** phép so |
| tolerance | khai trong `TOLERANCES.md` **trước** khi đo; lớp A = 0 khác biệt |
| anchor verification | 12 query/condition × 3 = 36 phép; **0 mismatch** |
| NumPy | 1.26.4 |
| BLAS env khi đo | `OMP/MKL/OPENBLAS_NUM_THREADS = 2` |

### 9.2 Bảng quyết định (stored vs fast path, full 6,003 × 3)

| condition | rank đổi / 6003 | max \|Δrank\| | vượt K=10 | vượt K=50 | Top-K set khác | Top-K order khác | ΔR@10 | ΔR@50 | ΔnDCG |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| multimodal | **4** | 1 | **0** | **0** | 0 | 7 | 0.0 | 0.0 | +1.4044159724058147e-09 |
| text_only_black_image | **35** | 2 | **0** | **0** | 0 | 7 | 0.0 | 0.0 | +1.5525044860797976e-09 |
| image_only_empty_text | **168** | 5 | **0** | **0** | 0 | 5 | 0.0 | 0.0 | −1.5751937930693848e-08 |
| **tổng** | **207 / 18,009** | **5** | **0** | **0** | **0** | **19** | **0.0** | **0.0** | — |

`optimize=True` và `gemm1024` cho **đúng cùng bảng** — cùng delta, cùng 207 query, cùng max Δrank 5,
0 vượt biên. Chúng đồng ý với nhau trên **mọi** query đã đổi và **mọi** rank mới
(`divergence_detail.json → fastpath_agreement → identical_changed_query_sets_and_new_ranks: true`).
GEMM nhanh hơn 4× nhưng **không "sạch" hơn một chút nào** — cả hai chỉ khác `einsum` ở **thứ tự cộng
dồn float**.

### 9.3 Metric tổng hợp, full precision

| metric | stored | fast (`opt` == `gemm1024`) | delta |
|---|---:|---:|---:|
| MM nDCG | 0.32828280509551133 | 0.3282828064999273 | +1.4044159724058147e-09 |
| text nDCG | 0.2006777268917305 | 0.20067772844423498 | +1.5525044860797976e-09 |
| image nDCG | 0.11373588650412819 | 0.11373587075219026 | −1.5751937930693848e-08 |
| **composition gap nDCG** | 0.388704727214254 | 0.38870472510024856 | **−2.1140054151658205e-09** |
| MM R@10 (%) | 35.61552557054806 | 35.61552557054806 | **0.0** |
| MM R@50 (%) | 54.3728135932034 | 54.3728135932034 | **0.0** |
| text R@10 (%) | 15.292353823088456 | 15.292353823088456 | **0.0** |
| text R@50 (%) | 27.053140096618357 | 27.053140096618357 | **0.0** |
| image R@10 (%) | 3.315009162085624 | 3.315009162085624 | **0.0** |
| image R@50 (%) | 10.31151091121106 | 10.31151091121106 | **0.0** |
| MM MRR | 0.20496768403872728 | 0.20496768410229743 | +6.357014914470938e-11 |
| mrr_gap | 0.11678778274193043 | 0.11678778259623385 | −1.457e-10 |
| scored counts (mm/text/image) | 6003 / 6003 / 6003 | 6003 / 6003 / 6003 | **0** |

### 9.4 Cơ chế: 1-ULP giữa các gallery row gần-trùng

Mọi rank đổi nằm trên một cặp row gần-trùng trong không gian feature. **Hai cơ chế khác nhau, không
được gộp:**

1. **Tie chính xác bị tách bởi 1 ULP.** Shipped cho positive **bằng đúng** một row khác;
   `_ranks_from_scores` cộng `#earlier-equal`. Fast path đổi thứ tự tích luỹ ⇒ positive dịch ~3e-08 ⇒
   đẳng thức biến mất ⇒ rank dịch 1.
   Ví dụ `7:574`: shipped `0.3319489359855652`, multiplicity **2**, rank 1628; fast `0.3319489657878876`,
   multiplicity **1**, rank 1627; margin −2.98e-08.
2. **Near-tie thật sự được quyết định bởi bit cuối.** Cả hai multiplicity 1 nhưng hai hàng xóm lệch
   < 5e-08. Ví dụ `7:690`: shipped `0.08333621919155121` (rank 63717) vs fast `0.08333626389503479`
   (rank 63716); margin −4.47e-08.

Số tie interaction trong các rank đã đổi: **2/4** (multimodal), **16/35** (text-only), **59/120**
(image-only; sample cap là 120 của 168). **Max |score margin| quan sát: 8.940696716308594e-08.**

Độ sâu của rank đổi (histogram `min(old,new)`):

| condition | 1–10 | 11–50 | 51–100 | 101–1000 | 1001–5000 | 5001–20000 | 20001–74381 |
|---|---:|---:|---:|---:|---:|---:|---:|
| multimodal | **0** | **0** | **0** | 0 | 1 | 1 | 2 |
| text_only_black_image | **0** | **0** | **0** | 1 | 2 | 2 | 30 |
| image_only_empty_text | **0** | **0** | **0** | 1 | 3 | 4 | 160 |

⇒ Một rank change ở ~1k–74k **không thể** đổi R@10 hay R@50 (cả hai là boolean `r <= K`), và không cái
nào vượt biên. Nó **có** đổi nDCG vì nDCG đọc rank chính xác; mức đổi đo được 1e-9 … 1.6e-8.

### 9.5 19 cửa sổ Top-K đổi thứ tự, membership không đổi

`set_differs = 0` cho cả 18,009: **không query nào đổi tập id depth-50**. Nhưng 19 cửa sổ trình bày
cùng 50 id theo thứ tự khác: **7 / 7 / 5**.

Phân loại chính xác theo `kind` trong JSON (tự đếm lại):

| loại | số |
|---|---:|
| "NOT a shipped-path tie: a genuine score reordering" | **16** |
| "exact tie broken by earlier gallery position upstream" | **3** |
| tổng | **19** |

Và đếm theo cờ: `shipped_exactly_equal` true ở **3** row, `fast_exactly_equal` true ở **6** row, "một
trong hai phía bằng chính xác" ở **9** row.

⚠️ **`20261108T153045-audit-equivalence/FINDINGS.md` F2 nói "4 trong 19 là exact-tie artefact"**, còn
`divergence_detail.json` cho con số **3** (theo `shipped_exactly_equal`) và **9** (theo "một trong hai
phía"). Ba con số không thể cùng đúng. JSON là dữ liệu thô, nên **JSON đúng**; con số "4" trong prose
**không truy vết được**. Điều này không đổi phán quyết (FAIL về rank identity), nhưng nếu ai trích
"4/19" thì đang trích một con số không có nguồn.

### 9.6 Bị từ chối, và điều kiện để chấp nhận

**Không chấp nhận như một thay thế im lặng. Chỉ chấp nhận như một thay đổi scoring function có khai
báo, mint `run_id` mới và artifact được dán nhãn lại.**

Được giữ: R@10, R@50 (bit-identical), cohort và denominator, tập Top-K candidate của mọi query.

Không được giữ: 207 rank đích của query (tới 5 vị trí), 19 thứ tự Top-K, nDCG tới O(1e-8), gap tới
−2.11e-09, MRR tới O(1e-11).

Nếu chấp nhận, phải: (1) mint `run_id` mới — **đã xảy ra**, vì source của `score()` đi vào
`adapter_source_hashes`/`adapter_module_source_hash`; (2) re-run **mọi** hàng đã publish để cả bảng
dùng một scoring function — bảng trộn là **không hợp lệ khoa học**; (3) summary ghi rõ đường scoring
(`scoring_impl: einsum_optimize` / `gemm_chunk1024`), không chỉ source hash, kèm ghi chú rằng rank
từng query lệch tới 5 vị trí còn metric K-hit không đổi; (4) artifact nào giữ rank per-query xuyên
model **không** được trộn pre/post.

**Phạm vi của con số 207:** đây là **một model** (habit) với **một** layout token 32×256.
CONESEP/CSMCIR/PTHA dùng subscript `nmk` và **đã** ship `optimize=True` từ trước; ENCODER và LIMN
dùng `@` với chunk 8192 row ở controller. ⇒ **hiện không có một scoring function thống nhất trên 9
adapter**, và độ nhạy rank với đổi thứ tự cộng mới được đo cho **đúng một** trong số chúng. Áp fast
path đồng loạt mà không verify từng model là tổng quát hoá một phép đo lên sáu code path chưa đo.

### 9.7 Cái gì `PARTIALLY_VERIFIED`, cái gì `NOT_RUN`

| hạng mục | trạng thái |
|---|---|
| decode/transform bit-identity, 2 họ transform, grid worker×batch | **PASS**, đo được |
| anchor correctness của harness | **PASS**, 36/36 |
| rank-identity của fast path, full 6,003 × 3 | **FAIL**, đo được đầy đủ |
| metric aggregate của fast path | **đo được**, đầy đủ |
| `optimize=True` vs `gemm1024` đồng ý với nhau | **PASS**, đo được |
| **forward feature bit-identity (G2-fwd)** | **NOT_RUN** — GPU bận toàn bộ; pixel bằng nhau **không** suy ra feature bằng nhau |
| **cold vs warm cache (G4d)** | **NOT_RUN** — chỉ có đúng một instance warm của habit cache trên đĩa; tạo bản cold cần re-encode 74,381 ảnh trên GPU; mô phỏng sẽ đo script chứ không đo runner |
| **query-axis parallelism do audit equivalence đo** | **NOT_RUN** — nó nói thẳng "Not measured by me"; phép đo dương tính nằm ở `THROUGHPUT.vi.md` (§8.5), prose-only |
| G3 trên đường shipped, full 6,003 × 3 | **PARTIAL** — einsum shipped 5.6 q/s ⇒ ~55 phút/cấu hình; bất biến cũng được bảo đảm về cấu trúc (mỗi query một hàng dense độc lập) |
| batch 1 vs 32 qua **forward thật** | **NOT_RUN** |
| query count không chia hết 32 qua `_encode_queries` trên dữ liệu thật | **NOT_RUN** — review code: prefill NaN + ghi theo index làm mất/misalign hàng **bất khả thi về cấu trúc** |
| determinism qua **hai process** | **NOT_RUN** — repo ghi nhận hai process có thể khác feature hash dù seed cố định (`replay/encoder.py:297-302`) |

---

## 10. Blocker, vùng chưa xác minh, next steps cụ thể

### 10.1 Blocker đang sống

| # | blocker | ảnh hưởng | cách gỡ |
|---|---|---|---|
| B1 | `clip4cir_rn50x4_fullft`: `adapters/models.py:374-375` hard-code `combiner_state.pt`/`clip_state.pt`, trên đĩa chỉ có `fiq_comb_*.pt`/`fiq_clip_*.pt` | variant này không chạy được | sửa `CLIP4CirFullftAdapter.command` đọc `required_files` từ variant — **một dòng**, chờ chủ `workbench/backend/**` |
| B2 | `dcnet`: patch `weights_only=False` vs pin `torch==1.4.0` — mâu thuẫn hai chiều trên cp37 | không chạy được | chọn một: (a) nâng pin lên 1.13.1 (wheel cp37 có); (b) patch không dùng `weights_only`; (c) patch có điều kiện theo version. Tất cả đều sửa registry/patches |
| B3 | `sprc/sprc_fiq_vitl`: `BLOCKED_EXTERNAL_ASSET_PROVENANCE` cho BLIP-2 (file có, sha tác giả không công bố) | không chạy được | khai một sha upstream, hoặc chủ registry admit như `LOCALLY_RECORDED_ASSET` |
| B4 | `sprc`: `command_status: COMMAND_BLOCKED_MODEL_MAPPING` là **stale** (mô tả variant `sprc_fiq_unresolved`, không mô tả ViT-L đã `MAPPING_CONFIRMED` 628/628) | chặn cả variant chạy được | tách `command_status` theo variant, hoặc nâng lên `COMMAND_AUDITED` |
| B5 | `sprc`: `src/data_utils.py` hard-code root, **không có** `--dataset-root` flag | không gỡ được như CLIP4Cir | stage `fashionIQ_dataset` tại `FASHIONIQ_ROOT`, hoặc chấp nhận prepared root là canonical |
| B6 | `clvc_net`, `tgcir`: **4 `.cuda()` không guard** trên đường chấm; torch 1.6 chỉ ship kernel tới sm_86, host capability 12.0 | không chạy được trên host này | GPU + torch ≥ 2.7 có sm_120, **hoặc** bản `test.py` guard device |
| B7 | `pair`: MAPPING_CONTRADICTED (pickle cần `model_vssp` + `MHTransformer_dis` mà repo pin không ship) | không chạy được | cần đúng revision module, hoặc một revision PAIR khác có hình học đó |
| B8 | 8/17 model không có interpreter resolve được (`workbench.env` chỉ set 9 biến; `environment_assignment.json` không tồn tại) | airknow, pair, clvc_net, dcnet, combiner_noft, fullft, tgcir, sprc | set 8 biến (operator-owned, **không** được sửa bởi audit này) hoặc provision `environment_assignment.json` với `state: RUNTIME_VERIFIED` |
| B9 | `sprc_fiq_unresolved`: file không tồn tại (`find /` 0 kết quả) | không có bytes để đọc | tải file rồi chạy key-diff `strict=False` như đã làm cho ViT-L |
| B10 | `tme`: `full.json` **không** chứa `variant_provenance` | artifact đơn lẻ bị đọc như reproduction hợp lệ | một dòng trong `runner.py` ghi `variant_provenance` |
| B11 | 4 model native không thể xuất hiện ở `/api/runs` (chỉ index `ResultRun`; không model nào khai `per_query_observer`) | aggregate-only | viết observer thật, hoặc chấp nhận aggregate-only — **quyết định sản phẩm** |
| B12 | `preparation_contracts.yaml` thiếu package: `setuptools` (CLIP4Cir), `einops` (DCNet), `decord`/`pycocoevalcap`/`webdataset`/`einops`/`scikit-image` (SPRC) | `verify_environment` báo `DEPENDENCIES_OK` trên env **không import nổi evaluator** | bổ sung contract |

### 10.2 Vùng chưa xác minh

| # | vùng | vì sao |
|---|---|---|
| U1 | **forward-feature equivalence** cho **mọi** model | GPU thuộc cohort `intent` toàn bộ cửa sổ kiểm toán; chạy song song bị cấm và cũng không cho số so sánh được |
| U2 | **cold vs warm cache**, cache enabled vs disabled, end-to-end | chỉ có một instance warm của habit cache; bản cold cần re-encode 74,381 ảnh |
| U3 | **tác động của `preprocess_train` lên `ENCODER`/`LIMN` trên gallery 74,381** | không cache feature ENCODER nào sống trên host; chỉ có PROXY 128 ảnh |
| U4 | **`WORKBENCH_SCORE_THREADS`** (do audit equivalence) | nó tự nói không đo; phép đo dương tính nằm ở tài liệu throughput nhưng **prose-only** |
| U5 | **vocabulary `bert-base-uncased`** mà BLIP-2 family thực sự resolve | upstream hard-code `/root/bert-base-uncased`; path đó **absent/unreadable** ở đây (`ls` → Permission denied; `exists False`), trong khi có snapshot HF cache `models--bert-base-uncased/snapshots/86b5e093…`. `[DEC]` là bos nhưng một encode đơn phát ra `[CLS] … [SEP]` nên pattern special token không bị ảnh hưởng; **vocab khác sẽ đổi id và do đó đổi embedding**. Chưa execute |
| U6 | **cấu trúc bên trong** của `dcnet/trained_model.pth` (legacy non-zip pickle), 6 file `clvc_net`, `tgcir/Fashioniq.pt` | digest khớp nhưng ruột chưa đọc |
| U7 | **membership gallery TME lúc chạy** | cache `val_images.pt`/`val_name_to_idx.json` trên host là 1 byte / `{}`; không đọc được cache của tác giả |
| U8 | **`gallery_all=True` có bao giờ tạo ra số CLVC-Net đã publish không** | không artifact nào trong tree pin ghi flag này |
| U9 | **airknow/conesep/habit/intent: `test.py:test` hay một script offline** đã tạo bảng paper | các repo đó không ship scoring CLI lẫn result JSON |
| U10 | **compute capability thật của host**, đo độc lập | 547 bản ghi `gpu_capabilities`; 342 ghi `"8.9"` nhưng kèm `"Synthetic GPU"` + `"CUDA out of memory"` + interpreter `pytest-of-…` ⇒ **test fixture**, không phải đo host. 6 bản ghi `"12.0"` là thật và `capability_verified`, tên `torch 2.7.1+cu128`. Audit này không chạy GPU nên lấy từ 6 bản ghi đó |
| U11 | **nhãn "pre-generated artifact" đã được gỡ chưa cho ENCODER/PTHA** | blocker đã sửa nhưng **chưa re-run**; nhãn vẫn còn hiệu lực |
| U12 | **ba run mới `intent` / `hint` / `conesep`** | có trên đĩa, metric đầy đủ, `topk_export` COMPLETE, nhưng **không tài liệu nào chịu trách nhiệm** và không nằm trong bất kỳ audit nguồn nào. `intent` là pre-F1 (`seed_applied: null`), `hint` và `conesep` là post-F1 (`false`). Ba run này có `query_count: 6003` và `coverage: 1.0`, nên **không** thuộc lớp smoke/fixture bị loại |
| U13 | **input identity còn vắng** (sau khi G5 đóng một phần) | **Đã đóng, không còn ở danh sách này**: BLAS/OMP/MKL intra-op thread count (`blas_threads_pinned`, `runner.py:1198`) và `scoring_module_source_hash`. **Vẫn vắng**: tf32/cuDNN benchmark/deterministic flags; GPU compute capability + driver + CUDA version (chỉ **tên** device được ghi); source thật của hàm transform bên thứ ba (`preprocessing.registry_values` chỉ ghi nhãn registry); **byte-level verification của mảng feature** khi đọc cache (sidecar kiểm fingerprint, không bao giờ kiểm bytes); và 851 hàng feature gallery trùng trên 2386 vị trí làm tie hệ thống. Xem §7.7 và G5 |

### 10.3 Định nghĩa gate — câu hỏi chưa giải quyết

**G0** được đặt tên bởi chính tiêu đề `REPOSITORY_INVENTORY.md`. **G2/G3/G4** được định nghĩa bởi
`TOLERANCES.md`. Nhưng **G1, G5, G6, G7 không được định nghĩa ở bất kỳ file nào** trong
`20261108T15*`, `20261108T-throughput`, hay `20261108T113610-eval`.

Nguy hiểm cụ thể: các chữ **G1–G7 xuất hiện lại** trong `workbench/docs/reviews/a40/` và
`final_pc_audit/` với nghĩa **hoàn toàn khác** — G1 = artifact safety, G2 = browser E2E,
G3 = mobile timeout, G4 = pipeline interruption, G5 = ledger integrity, G6 = laptop handoff, và trong
`final_pc_audit` thì G7 = "plan with and without `WORKBENCH_RUNTIME_ADAPTATION=sm_120`". **Không được
trộn hai bộ.** Bất kỳ tài liệu nào viết "G7" phải nêu chủ đề của nó.

Báo cáo này dùng cách đọc rộng nhất có thể bảo vệ được bằng bằng chứng, và nói rõ cách đọc đó ở §2.2
và trong `GATE_MATRIX.md`. Nếu chương trình kiểm toán có một bản định nghĩa gate ở nơi khác, nó **phải
được dẫn ra** thì các nhãn G1/G5/G6/G7 mới có nghĩa. **câu hỏi chưa giải quyết**

### 10.4 Model mà con số công bố là "pre-generated artifact, not runnable from HEAD"

Trạng thái **tại thời điểm audit-fix**, và nhãn này **chưa được gỡ**:

| model | MM R@10 / R@50 | artifact run_id | runnable từ HEAD? | nhãn |
|---|---|---|---|---|
| **ENCODER** | 31.01782442112277 / 49.5752123938031 | `6a528e877c328772e2c7bb28`, `eda30a37c0bc17ac43844f09` (smoke) | **KHÔNG** — `NameError` ở `:330` (đã sửa) | **pre-generated artifact, not runnable from HEAD** |
| **PTHA+MTST** | 19.473596535065802 / 34.44944194569382 | `315dd590eeb03d2c78d7937a` | **KHÔNG** — `NameError` ở `:320` (đã sửa) | **pre-generated artifact, not runnable from HEAD** |
| LIMN | 30.064581917063222 / 48.759347382732834 | `10173dc45dea8c90fd2a284b` | có — import cả 3 ở `:45-47` | runs; id đổi vì F1/F6 |
| TME | 34.01632517074796 / 53.95635515575545 | `ad477312aa57c1b9114f9e51` | có — import cả 3 ở `:15` | runs |
| HABIT | 35.61552557054806 / 54.3728135932034 | `497c25c57fbeba885ecfb9d1`, `e0dd04a46ef7a58334b9e353` (FAILED) | có | runs |
| INTENT | 34.39946693320007 / 54.00633016824921 | `37ee72aa298b276d1bfda0fa` | có | runs |
| HINT | 32.816924870897886 / 51.77411294352824 | `4bc9893a977e501b82378e76` | có | runs, post-F1 |
| CONESEP | 35.48225887056472 / 54.13959686823255 | `30f6de423861cc1d6c78c361` | có | runs, post-F1, topk COMPLETE; nhãn `fiq_n02` = **train noise 20%** |
| CSMCIR | — | — | không, vì lý do đã ghi (`mode_supported` False mọi condition) | n/a |

Bốn model có số **đã publish** cần chạy lại vì fix decode đổi `adapter_module_source_hash`:
`model_encoder`, `model_limn`, `model_ptha_mtst`, `model_tme`. Cache 2.3 GB của chúng đã bị xoá. ⇒
**từ HEAD không còn đường tái lập đúng các id đã công bố** — đây là provenance break, và nó nặng hơn
decode chậm. Xử lý (không được làm ngầm): patch (xong) → verify bit-identity từng model (xong, cả ba
họ transform 0.0) → **chạy lại cả bốn** với `--topk-depth 50 --keep-features`, mỗi model ~15–25 phút
thay vì ~73 ⇒ cả bốn ≈ 1.5 h. Dòng mới là "lần thử thứ hai", phải ghi provenance; run id cũ đánh dấu
**pre-decode-change artifact**; **không thay thế im lặng** một con số đã công bố.

### 10.5 Next steps, theo thứ tự phụ thuộc

1. **Gỡ B1** (`adapters/models.py:374-375`) — một dòng, mở khoá `clip4cir_rn50x4_fullft`, mở khoá luôn
   khả năng đối chiếu cặp noft/fullft trên cùng protocol.
2. **Chạy lại 4 model có số đã publish** (encoder, limn, ptha_mtst, tme) dưới run id mới, có
   `--topk-depth 50 --keep-features`. Đây là bước duy nhất chuyển
   "pre-generated artifact, not runnable from HEAD" thành một artifact tái lập được.
3. **Đo G2-fwd** cho ít nhất 2 họ transform (một lavis, một open_clip) ngay khi GPU trống: encode
   cùng 512 ảnh hai lần ở hai cấu hình batch/worker khác nhau và so **feature matrix**, không so pixel.
   Đây là gate duy nhất còn trống ở giữa "decode đúng" và "số đúng".
4. **Đo cold vs warm cache** (G4d) trên một model có cache sống được (habit), sau khi (2) xong.
5. **Đóng nốt G5**: `scoring_module_source_hash` nay có trong `identity` cho run mới và
   `blas_threads_pinned` đã ghi BLAS thread count ⇒ **hai mục đã xong**. Còn lại: ghi tf32/cuDNN flags
   và GPU capability/driver/CUDA version vào `identity`, và — quan trọng nhất — verify byte của mảng
   feature khi đọc cache, không chỉ sidecar. G5 là **DEFECT đã đóng một phần**, không phải gate xanh.
6. **Quyết định scoring function**: hoặc giữ bất đối xứng hiện tại (4 model một thứ tự cộng, 3 model
   khác) và **ghi caveat rằng rank giữa hai nhóm không so được ở mức từng query**, hoặc chuyển đồng
   loạt sang `optimize=True`/GEMM với run id mới và re-run **toàn bộ**. Trạng thái hiện tại — hai thứ
   tự cộng trong một cohort đã publish — là lựa chọn **tệ hơn cả hai**.
7. **Đo `WORKBENCH_SCORE_THREADS` bằng một file raw** (cùng harness `equivalence_fastpath.py
   --verify-sample`, ~15 phút) để con số 5.67× có nguồn, hoặc ghi nó là prose-only trong mọi tài liệu
   release.
8. **Bổ sung package thiếu vào `preparation_contracts.yaml`** (B12) để `verify_environment` không báo
   `DEPENDENCIES_OK` trên env không import nổi evaluator.
9. **Ghi `variant_provenance` vào summary** (B10) để artifact TME sm_120 không bị đọc như reproduction.
10. **Quyết định** SPN4CIR: đăng ký thành biến thể thứ 18 hay xoá clone. Audit này **không** đưa ra quan
    điểm; nó chỉ ghi rằng "16" là **registered count**.
11. **Ghi lại định nghĩa G1/G5/G6/G7** nếu chương trình kiểm toán có chúng ở nơi khác (§10.3), nếu
    không thì bỏ các nhãn đó khỏi tài liệu release.

---

## Cách dùng nhanh

Các phép kiểm rẻ nhất, theo thứ tự, đều **không cần GPU**:

```bash
# 1) Đếm lại 16/17/29 và 21/29 staged — 2 giây
workbench/envs/ctl312h/bin/python - <<'PY'
import yaml, sys; sys.path.insert(0,'.')
from workbench.backend import registry
d = yaml.safe_load(open('workbench/registry/models.yaml'))
urls = {m['upstream_repo_url'] for m in d['models']}
vars_ = [(m['model_id'], v) for m in d['models'] for v in m['checkpoint_variants']]
present = sum(1 for mid, v in vars_ if registry.checkpoint_path(mid, v).exists())
print(len(d['models']), 'variants;', len(urls), 'repos;', len(vars_), 'checkpoints;', present, 'staged')
PY
# kỳ vọng: 17 variants; 16 repos; 29 checkpoints; 21 staged

# 2) Tìm một artifact là pre-F1 hay post-F1 — 1 giây
grep -o '"seed_applied": [a-z]*' workbench/artifacts/composition/summary/*/full.json
# null = pre-F1 (addressability đã mất); false = post-F1

# 3) Chứng minh run id không tái lập được cho artifact cũ — 1 giây
#    `"seed"` vẫn xuất hiện đúng 1 lần (dòng summary), nên đừng grep cả file:
awk '/identity = \{/,/^            \}/' workbench/composition/runner.py | grep -c '"seed": seed'
#   kỳ vọng: 0  → seed không còn trong identity, nên run id cũ không tái tính được
grep -n 'seed_applied' workbench/composition/runner.py    # phải thấy "seed_applied": False
grep -o '"seed_applied": [a-z]*' workbench/artifacts/composition/summary/*/full.json

# 4) Chứng minh TME pin không tồn tại trong clone teacher — 1 giây
git -C teacher/repos/TME cat-file -t 394aad79df2c104feb32a2c96132582012765062  # fatal
git -C teacher/repos/TME remote -v                                              # He-Changhao/2025-CVPR-TME.git

# 5) Chứng minh CSMCIR teacher tree ở trạng thái thứ ba — 1 giây
sha256sum teacher/repos/CSMCIR/src/lavis/models/__init__.py \
          workbench/third_party/CSMCIR/src/lavis/models/__init__.py
# 8e9cd6a8… (không phải base, không phải derived)  vs  0ec13443… (derived, đúng)

# 6) Đếm lại 19 order-only divergence và phân loại tie — 1 giây
cd workbench/docs/reviews/20261108T153045-audit-equivalence && python3 -c "
import json,collections
d=json.load(open('divergence_detail.json'))['order_only_divergences']
print(sum(v['count'] for v in d.values()), collections.Counter(r['kind'][:20] for v in d.values() for r in v['rows']))"
# 19  Counter({'NOT a shipped-path ti': 16, 'exact tie broken by': 3})

# 7) Xác nhận input vắng khỏi cache identity — 1 giây
# Đây là số ĐO TẠI LÚC AUDIT (7). Từ đó 2 đã đóng (blas_threads_pinned, scoring_module_source_hash):
# runner.py:1198 ghi blas_threads_pinned; xem BLAS-POLICY.vi.md và EQUIVALENCE.md G4e cho trạng thái hiện tại (5 còn vắng).
python3 -c "
import json; d=json.load(open('workbench/docs/reviews/20261108T153045-audit-equivalence/cache_and_ties.json'))['cache_identity']
print(len(d['inputs_not_in_identity']), 'missing |', 'scoring hash present:', d['identity_has_scoring_module_source_hash'],
      '| sidecar verifies bytes:', d['sidecar_verifies_array_bytes'])"
# 7 missing | False | False  ← số của lượt audit, không phải của HEAD hôm nay

# 8) Kiểm metric aggregate full precision, không làm tròn
python3 -c "
import json; d=json.load(open('workbench/docs/reviews/20261108T153045-audit-equivalence/gap_and_axes.json'))
print(d['gap']['stored']['composition_gap_ndcg'], d['gap']['opt']['composition_gap_ndcg'])"
# 0.388704727214254  0.38870472510024856

# 9) Kiểm hai model từng không chạy được từ HEAD đã có import chưa — 1 giây
grep -n 'limit_torch_threads\|prefetch_rows\|resolved_decode_workers' \
  workbench/composition/model_encoder.py workbench/composition/model_ptha_mtst.py | head
# mỗi file phải có cả 3 tên trong block import VÀ các call site

# 10) Đếm diff vendored thật (13 sửa + 2 thêm, 8 repo) thay vì tin prose
grep -c '^======== ' workbench/docs/reviews/20261108T155252-protocol-audit/raw/vendored_diffs.txt  # 15
```

Không có phép nào ở trên cần GPU, không cái nào sửa file, và tất cả đều chạy dưới 5 giây.
