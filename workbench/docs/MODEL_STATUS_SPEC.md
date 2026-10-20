# MODEL_STATUS_SPEC — trạng thái 16 model CIR

Tài liệu trạng thái chuẩn. Registry (`workbench/registry/models.yaml`) vẫn là **nguồn tham số thực thi**;
file này mô tả **tình trạng, bằng chứng và điều kiện chuyển trạng thái**. Không có backend status
service, database hay dashboard mới.

| | |
|---|---|
| Cập nhật | 2026-10-19 |
| Branch | `research/cir-failure-workbench` |
| Source SHA đã kiểm | `e46ad65aa792b01461cc5319b2b8511b4714df41` + các sửa của lượt R10 (ghi ở `workbench/docs/reviews/r10/R10_REMEDIATION.md`) |
| Môi trường kiểm | laptop: Linux x64, `.venv` Python 3.13 + torch CPU, RTX 3050 4 GiB; **không có checkpoint nào staged** |
| Phạm vi đã chạy | static + fixture + parser/unit + native constructor + native LAVIS processor + **native DCNet ConfigParser** + venv thật. **Không có inference nào chạy.** |
| Lượt R9 | 4 lỗi tái hiện được ở `914703e` đã sửa (F08/F09/F04/F05), F06 áp dụng |
| Lượt R10 | 6 finding độc lập đã tái hiện và sửa: R10-01/02 (DCNet binding + staging), R10-03 (lifecycle processor→checkpoint), R10-04 (source_blocker lỗi thời), R10-05 (profile dependency + scanner đọc sai chương trình), R10-06 (venv cô lập mượn package của base) |

---

## 1. Quy tắc đọc trạng thái

Các tầng **độc lập**; một tầng đạt không suy ra tầng sau:

| Tầng | Nghĩa | Cách chứng minh |
|---|---|---|
| `SOURCE_READY` | source vendored, pin + digest manifest khớp, không nested `.git` | `test_fresh_clone_delivery.py`, `test_vendored_source_integrity.py` |
| `ENTRYPOINT_EXISTS` | có file entrypoint thật (CLI hoặc wrapper) | đọc source, `--help` chạy được |
| `PREFLIGHT_PASS` | preflight chạy và **báo blocker đúng** trên input thiếu | test per-model + `test_component_boundaries.py` |
| `PREFLIGHT_IMPLEMENTED_BLOCKED` | có đường preflight thật nhưng model **đang** bị chặn đúng, nên chưa lần nào chạy hết | test preflight + blocker list hiện tại |
| `CHECKPOINT_MAPPING_VERIFIED` | file/biến thể được bind bởi metadata tác giả (revision + size + hash) | HF LFS oid, digest archive |
| `CHECKPOINT_LOADED` | checkpoint thật đã `load_state_dict`/unpickle thành công | **NOT_CHECKED toàn bộ** — không có file local |
| `CODE_TESTED` | đường code (adapter→argparse→serializer→parser) có test hồi quy | pytest |
| `GPU_SMOKE_VERIFIED` | 1 batch forward trên GPU thật | **NOT_CHECKED** |
| `FULL_VAL_VERIFIED` | đủ dress/shirt/toptee + aggregate đúng protocol | **NOT_CHECKED** |
| `AGGREGATE_PARSED` | parser đọc được output native thật | test parser với output shape thật |
| `PER_QUERY_EXPORT_VERIFIED` | có exporter + parity so với native | chỉ `limn` |
| `PAPER_PARITY_CHECKED` | đã so với số paper | **NOT_CHECKED** |

**Quy tắc:**

- Chưa kiểm ⇒ ghi `NOT_CHECKED`/`UNKNOWN`, không suy ra pass/fail.
- **Không dùng một chữ `COMPLETE`.** Một model có thể sạch bug vẫn `BLOCKED` vì weights.
- Phân biệt: không có checkpoint local ≠ mapping chưa rõ ≠ hash chưa kiểm ≠ load thất bại ≠
  thiếu parser ≠ chưa chạy inference ≠ tác giả không công bố weights.
- Preflight-only **không** chuyển thành inference-ready bằng cách đổi tên status.
- Smoke 1 category **không** phải full benchmark; full val cần đủ 3 category.

**3 protocol không được trộn:**

| protocol | gallery | reference |
|---|---|---|
| `fashioniq_original_split` | full split | **eligible** |
| `fashioniq_full_gallery_ref_excluded` | full split | excluded |
| `fashioniq_val_split` | pair-union (ordered) | excluded |

Metric schema là **%**; native có thể trả fraction (DCNet) — đổi đơn vị đúng **một lần**. Macro giữ
quy ước native đã xác minh (mean không trọng số theo category), không query-weighted.

---

## 2. Bảng tổng quan 16 model

Mức kiểm chứng cao nhất thực tế đạt được (không phải mức mong đợi):

| # | model_id | upstream | protocol native | checkpoint variant / noise | source+patch integrity | ckpt/data/env readiness | replay capability | parser | per-query | mức thực tế | blocker |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `limn` | iLearn-Lab/TPAMI24-LIMN | `fashioniq_val_split` | 3 ckpt category, clean eval, training noise unstated | vendored, pin OK | **3 ckpt staged, size+sha256 VERIFIED**; env provisioned (torch 1.12.1+cu113) | native wrapper | `limn_structured_json` | **có** exporter+observer+publish, parity 0 mismatch | **`FULL_VAL_VERIFIED` + `PAPER_PARITY_CHECKED`** | — (blocker engineering đã đóng) |
| 2 | `encoder` | iLearn-Lab/AAAI25-ENCODER | `fashioniq_val_split` | `fashioniq.pt` @rev `dd9b0865` | vendored + 2 local fix | **ckpt staged, size 631 879 365 + sha256 `0f31c501…` VERIFIED**; LAION aux staged; env CHƯA provision | wrapper | shared replay | không | `CHECKPOINT_MAPPING_VERIFIED` | `ENV_NOT_PROVISIONED` |
| 3 | `hint` | iLearn-Lab/ICASSP26-HINT | `fashioniq_val_split` | `fashioniq.pt` 2 724 451 182 B | vendored, pin OK | size/hash VERIFIED; chưa stage | wrapper + cache isolation | `hint_native_recall` | không | `AGGREGATE_PARSED` | `PUBLIC_ASSET_SETUP` |
| 4 | `pair` | iLearn-Lab/ICASSP25-PAIR | `fashioniq_val_split` | `pair-B1.pt`/`pair-B2.pt` | vendored + 1 local fix + `data/datasets.py` recovered | mapping **UNRESOLVED** | wrapper | shared replay | không | `CODE_TESTED` | `AUTHOR_ONLY_INFORMATION` + `SCIENTIFIC_VARIANT` |
| 5 | `clvc_net` | iLearn-Lab/SIGIR21-CLVC-Net | `fashioniq_val_split` | 6 whole models (2/category) | vendored + torch.hub fix | archive `CLVCNET.zip` interior chưa xác minh | wrapper | shared replay | không | `CODE_TESTED` | `AUTHOR_ONLY_INFORMATION` |
| 6 | `tgcir` | iLearn-Lab/MM23-TG-CIR | `fashioniq_val_split` | `TG-CIR.zip` (Drive) | vendored, pin OK | archive member/hash chưa xác minh; pin clip đã sửa | **preflight-only** | không | không | `PREFLIGHT_IMPLEMENTED_BLOCKED` | `AUTHOR_ONLY_INFORMATION` |
| 7 | `habit` | iLearn-Lab/AAAI26-HABIT | `fashioniq_full_gallery_ref_excluded` | N0.2/N0.5/N0.8, clean eval | vendored, pin OK | hash VERIFIED; chưa stage; **không có bản clean** | wrapper | shared replay | không | `CODE_TESTED` | `PUBLIC_ASSET_SETUP` + `SCIENTIFIC_VARIANT` |
| 8 | `intent` | iLearn-Lab/AAAI26-INTENT | `fashioniq_full_gallery_ref_excluded` | N0.2/N0.5/N0.8 | vendored, pin OK | hash VERIFIED; chưa stage | wrapper | shared replay | không | `CODE_TESTED` | `PUBLIC_ASSET_SETUP` |
| 9 | `conesep` | Lee-zixu/ConeSep (redirect) | `fashioniq_full_gallery_ref_excluded` | N0.2/N0.5/N0.8 | vendored, pin OK | hash **chưa có** (`expected_sha256: null` trong registry); chưa stage | wrapper | shared replay | không | `CODE_TESTED` | `PUBLIC_ASSET_SETUP` |
| 10 | `airknow` | iLearn-Lab/CVPR26-Air-Know | `fashioniq_full_gallery_ref_excluded` | 0.5/0.8, clean eval | vendored + 2 local fix | hash **chưa có** (`expected_sha256: null` trong registry); chưa stage | wrapper | shared replay | không | `CODE_TESTED` | `PUBLIC_ASSET_SETUP` + `SCIENTIFIC_VARIANT` (train mode) |
| 11 | `dcnet` | ozmig77/dcnet | `fashioniq_full_gallery_ref_excluded` | `trained_model.pth` (Drive) | vendored, pin OK | GloVe pkl chưa công bố; **data_dir binding kiểm trên cả launch path** | preflight + CLI | `dcnet_native_logger` (đọc thứ tự category) | không | `AGGREGATE_PARSED` | `AUTHOR_ONLY_INFORMATION` |
| 12 | `csmcir` | qzp2018/CSMCIR | `fashioniq_original_split` | `fashioniq_tuned_clip_best.pt` 2 720 615 859 B | vendored + **2 approved patches (cả hai bắt buộc)** | ckpt hash VERIFIED; CoT captions đã có; chưa stage | script CLI (`__main__`) | `csmcir_stdout_json` | không | `CODE_TESTED` | `PUBLIC_ASSET_SETUP` |
| 13 | `combiner_rn50x4_noft` | ABaldrati/CLIP4Cir | `fashioniq_original_split` | RN50x4 noft Combiner | vendored, pin OK | archive 5 parts, digest công bố, **chưa tải** | script CLI | `combiner_stdout_recall` | không | `SOURCE_READY` | `PUBLIC_ASSET_SETUP` |
| 14 | `clip4cir_rn50x4_fullft` | ABaldrati/CLIP4Cir | `fashioniq_original_split` | fullft CLIP + Combiner (cùng config) | vendored, pin OK | như trên; cần **cặp** khớp | script CLI | `combiner_stdout_recall` | không | `SOURCE_READY` | `PUBLIC_ASSET_SETUP` |
| 15 | `sprc` | chunmeifeng/SPRC | `fashioniq_original_split` | ViT-g (paper) / ViT-L (variant) | vendored, pin OK | ViT-g OneDrive 403; ViT-L reachable | script CLI | `csmcir_stdout_json` | không | `CODE_TESTED` | `PUBLIC_ASSET_SETUP` + `SCIENTIFIC_VARIANT` |
| 16 | `ptha_mtst` | zyinan99/MTST_PTHA | `fashioniq_original_split` | **không có FashionIQ ckpt** | vendored + 3 local fix | không có weights | preflight-only | không | không | `PREFLIGHT_IMPLEMENTED_BLOCKED` | `AUTHOR_ONLY_INFORMATION` |


> **Ảnh hưởng của R14** (21/10/2026): R14 chạy **thật** LIMN end-to-end và sửa **ba** lỗi chặn đường
> chạy, tất cả ở tầng hạ tầng — không đổi ranking, preprocessing, reference policy hay semantics
> khoa học của model nào. (1) `pc.sh` truyền `-T` cho `docker exec` trực tiếp; CLI thật từ chối
> (`unknown shorthand flag: 'T'`, exit 125), nên nhánh preparation (mọi model **trừ** `limn`/`csmcir`)
> và **verifier dùng chung** không bao giờ chạy được. (2) `runtime._exports_python_init` chạy con
> `dlopen` bằng `sys.executable` (Python của orchestrator); extension CPython gắn cứng ABI, nên
> `_C.cpython-310-*.so` báo `undefined symbol` và bị coi là **không load được** — khiến cổng GPU
> `capability` từ chối **mọi** model có interpreter riêng, và lỗi hiện ra như "CUDA unavailable".
> (3) `pinned_revision` chạy `git rev-parse HEAD` trong checkout đã bị gỡ `.git` lồng, nên Git đi
> ngược lên repo workbench và trả về **commit của workbench**; completion proof so nó với
> `upstream_commit_sha` của model rồi từ chối — không model vendored nào tạo được proof hợp lệ.
> Cả ba nay đã sửa và có regression bắt được khi revert.
>
> LIMN là model **đầu tiên** đạt `CHECKPOINT_LOADED`, `GPU_SMOKE_VERIFIED`, `FULL_VAL_VERIFIED` và
> `PAPER_PARITY_CHECKED`. `encoder` và `hint` vẫn dừng ở mức cũ: checkpoint của `encoder` đã stage và
> verify trên PC này, nhưng interpreter của nó **chưa** được provision, và cả hai **không** khai
> `per_query_observer` nên chưa có per-query để so cross-model.

> **Ảnh hưởng của R13** (21/10/2026): R13 sửa **năm** lỗi, tất cả ở tầng dependency/verifier, không đổi mức
> khoa học của model nào. (1) Verifier từ chối **nguồn đúng** vì khoảng trắng hợp lệ trước `git+` bị dính
> vào URL; (2) cùng lúc đó `runtime` **bỏ** `Requirement.url`, nên PyPI `clip==0.2.0` thoả một khai báo
> OpenAI CLIP và được `RUNTIME_READY` rồi được **reuse**; (3) hai entry CLIP4Cir vẫn khai `clip` trần;
> (4) `pip check` fail với stdout rỗng (interpreter không có pip) vẫn được coi là pass; (5) cùng một
> direct reference khai hai lần bị coi là hai nguồn khác nhau. Thêm hai giả thuyết đã kiểm và đóng:
> venv hợp lệ bị từ chối khi `bin/python` là symlink tới binary nền (nay so **environment identity**
> = `sys.prefix`, không so đường dẫn binary), và nguồn bị chặn không thể mượn readiness từ record GPU
> cũ (cache khoá theo contract digest, có `packages`).
> R13-02 chọn revision `a1d0717` cho CLIP4Cir: README upstream **không pin revision**, nên đây là
> **adaptation được ghi rõ**, không phải pin của tác giả.

> **Ảnh hưởng của R12** (20/10/2026): R12 sửa hai lỗi contract **không** đổi mức khoa học của model nào.
> (1) `scikit-learn` bị loại khỏi plan của `habit`, `intent`, `conesep`, `airknow` vì một census sai —
> census import `lavis.models`, import này **vỡ giữa đường** khi thiếu dependency, nên mọi module cần
> sau điểm vỡ đều không được đếm. Bốn class file native import `sklearn` ở module scope không guard,
> và INTENT import thêm `matplotlib.pyplot`; nay chúng được khai và plan cài chúng.
> (2) `clip @ git+https://github.com/openai/CLIP.git@a1d071...` của `tgcir` và `sprc` bị biến thành
> `clip` trần, nên installer cài `clip==0.2.0` — một package clipboard không có `load`/`tokenize`
> mà evaluator gọi. Nay source được giữ nguyên suốt parse → merge → grouping → lệnh cài → verify.
> Cả hai đều là lỗi **engineering**; không model nào được nâng lên `CHECKPOINT_LOADED`,
> `GPU_SMOKE_VERIFIED` hay `PAPER_PARITY_CHECKED`, vì mọi blocker còn lại vẫn là asset tác giả/GPU.

> **Ảnh hưởng của R11 lên bảng này** (20/10/2026): R11 sửa contract *môi trường* của `hint`, `habit`,
> `conesep`, `airknow` (Python 3.8 không cài được `pandas==2.2.0`/`scikit-image==0.22.0`) và của
> `sprc` (planner và verifier đọc hai nguồn version khác nhau). Các mức trong bảng **không** thay đổi
> vì chúng vẫn bị chặn bởi asset tác giả, không phải bởi environment: sửa R11 làm cho bước
> provisioning *có thể thành công*, chứ không tự sinh ra checkpoint hay GPU. Riêng cột
> `ckpt/data/env readiness` của 5 model LAVIS nay đúng hơn ở tầng dependency: profile đã cài được,
> và đường processor → checkpoint của cả 5 wrapper đã có bằng chứng chạy thật trong hai tiến trình
> riêng (`workbench/tests/test_r11_lavis_lifecycle.py`) — nhưng đó là **fixture**, không phải
> weight tác giả. Không model nào được nâng lên `CHECKPOINT_LOADED` hay `GPU_SMOKE_VERIFIED`.

**`limn` đạt `CHECKPOINT_LOADED`, `GPU_SMOKE_VERIFIED`, `FULL_VAL_VERIFIED` và
`PAPER_PARITY_CHECKED`** trên PC này (số đo ở §3.1). **15 model còn lại** chưa đạt mức nào trong bốn
mức đó: mọi số trong registry cho chúng vẫn là số paper, không phải số đo.

---

## 3. Chi tiết theo model

### 1. `limn`
- Upstream `iLearn-Lab/TPAMI24-LIMN` @ `7d7bc9b116f5`. Wrapper `workbench/replay/limn.py` (native
  `test.py:test()`), exporter `limn_export.py`, observer `limn_observer.py`, publish `limn_publish.py`.
- Checkpoint: 3 whole-model `0_{dress,shirt,toptee}_best_model.pt`, Hub LFS sha256 đã ghi trong registry.
  Training noise chưa nêu trong paper ⇒ `null` (không gán 0). Evaluation noise 0.
- Dataset: `fashioniq_ilearn_resized_training`, `.jpg`. Trên PC này dùng `data/fashionIQ_dataset`
  (`FashionIQ` là symlink cùng thư mục): `resized_image/{dress,shirt,toptee}` =
  3817 / 6346 / 5373 ảnh `.jpg`, khớp đúng layout native yêu cầu.
- Protocol `fashioniq_val_split`; ranking native `np.argsort(-sims)` + mask reference `-10e10`.
  Cache isolation: có. Per-query: **có**, parity so với native aggregate.
- **ĐÃ CHẠY THẬT (21/10/2026)** — checkpoint tác giả, dữ liệu tác giả, GPU thật (RTX 3050, sm_86):

| hạng mục | kết quả đo |
|---|---|
| checkpoint integrity | cả 3 file: size **OK**, sha256 **OK** |
| checkpoint load | `model.LIMN`, 440.26 M tham số, `cuda:0` |
| GPU probe | `cuda_available=true`, `computed=true`, cc **8.6** |
| **dress** | R1 **18.6911** · R10 **50.7685** · R50 **74.0208** |
| **shirt** | R1 **25.5152** · R10 **55.5447** · R50 **76.7910** |
| **toptee** | R1 **26.2111** · R10 **60.5813** · R50 **81.6930** |
| **macro** | R10 **55.6315** · R50 **77.5016** · mean **66.5665** |
| paper (làm tham chiếu) | R10 55.91 · R50 77.82 · mean 66.87 |

  Độ lệch so với paper: R10 **−0.28**, R50 **−0.32**, mean **−0.30** điểm. Sai số này nằm trong
  khoảng của một lần chạy lại hợp lệ (không cắt tỉa, không đổi ranking, không đổi reference policy).
- Per-query: exporter chạy thật trên cả 3 category, **6016 query**, `top_k=50`,
  `INSTRUMENTATION_PARITY_VERIFIED`, `instrumentation_mismatches: {}` — tức ranking do observer
  dựng lại **khớp tuyệt đối** với aggregate native. Artifact canonical đã publish:
  `workbench/artifacts/results/fashioniq_val_split/limn/20261021T040000000000Z_limn_base_iter0_all_categories_fashioniq_val_split.json`
- Hạ tầng: index rebuild → `/api/runs` trả run `data_kind: experiment` → `/api/queries` trả
  `total: 6016` kèm `top_results` → UI Sample Explorer hiển thị `1–20 of 6016` với ảnh thật.
- `WORKBENCH_LIMN_BATCH_SIZE` (mặc định 32 = thiết lập paper) là knob vận hành cho GPU nhỏ; nó chỉ
  đổi số ảnh mỗi lượt forward, không đổi thứ hạng (`test()` concat rồi chuẩn hoá sau). Lượt đo này
  dùng 4; provenance ghi rõ `batch_size_source: operator override`.
- **Còn thiếu để hoàn tất khoa học**: xác nhận trên GPU đích `sm_120` (RTX 5070 Ti). Lượt này chạy
  profile **historical pins** trên `sm_86`; profile `adapted_runtime: sm_120` vẫn chưa được đo.

### 2. `encoder`
- Upstream @ `29a2a31d6a56`. Local fix: `datasets1.py` + `import datasets as datasets1`.
- Checkpoint `fashioniq.pt`: **đã verify độc lập** — Hub revision `dd9b086569fe69bd9111b6114546201c97934b01`,
  size 631 879 365, LFS oid `0f31c50150a7bda3495e3378df13b4cec7924b52efd23b5e5dd9731459da9173`;
  Range read trả `PK..fashioniq/data.pkl` + `OrderedDict` bắt đầu `loss_T`.
- Loader: giữ `strict=False` (upstream) nhưng **không bỏ qua report** — missing/unexpected (ngoài
  `loss_T`) ⇒ raise. Blocker: backbone LAION ViT-B/32 + ckpt chưa stage.
- Đã chạy: `test_encoder_replay.py` (5), `test_verified_declarations.py` (F10).

### 3. `hint`
- Upstream @ `bec50b6c8c19`. Wrapper `hint.py` chạy trong `isolated_cache_root` (F03/B3):
  `fashion_iq_data.json` + `test_queries_*.pkl` là tensor **đã transform**, nên root tạm mới là bắt buộc.
- Parser `hint_native_recall` đọc **cả hai** shape thật: JSON của wrapper và print native
  `[('dress_r10', v), ...]` (`test.py:103-108`). Macro tính lại.
- Hub: `iLearn-Lab/HINT` (không phải `ICASSP26-HINT` — trả 401). `fashioniq.pt` 2 724 451 182 B,
  LFS `bf9c72d4…`.
- Đã chạy: `test_hint_replay.py` (4), `test_hint_metric_parser.py` (11),
  `test_r9_defect_regressions.py` (F05 train-inputs + F06 processor).
- **F05 đã sửa (lượt R9)**: `validate_dataset(root, categories=('dress',))` từng trả `[]` trên root
  chỉ có dress, nhưng constructor native (`datasets.py:35-47,59-73`) mở **cả ba** category bất kể
  `--category`, và `isolated_cache_root` luôn rỗng nên nhánh cold luôn chạy ⇒ checkpoint được load
  xong mới chết ở `cap.shirt.train.json`. Nay tập input của constructor là câu hỏi **riêng**, kiểm
  bằng `workbench/replay/common.py:constructor_input_blockers`, dùng chung cho `hint`, `habit`,
  `intent`, `conesep`, `airknow`, `pair`.
- **F06 đã sửa (lượt R9)**: `txt_processors` lấy từ config pinned qua
  `workbench/replay/lavis_processors.py`; không còn dựng model BLIP-2 thứ hai, không cần
  `/root/bert-base-uncased`. `--backbone pretrain_vitL` chọn đúng config ViT-L mà model class công bố.

### 4. `pair`
- Upstream @ `bfb0b98cdb4c`. Local fix: `models/MHTransformer_rev.py` =
  **`EXPERIMENTAL_RECONSTRUCTION`** (đã ghi rõ trong docstring: chỉ khác 2 dòng so với
  `MHTransformer.py` sau khi bỏ class ordinary-attention; `MultiHeadAttentionRev.forward` dùng
  `1 - attention.softmax(dim=2)`), và `data/datasets.py` (file nguồn **bắt buộc** từng bị `.gitignore`
  `data/` nuốt).
- Checkpoint: `pair-B1.pt` / `pair-B2.pt` công bố nhưng **không có nhãn dataset/variant** ⇒ mapping
  `UNVERIFIED`. Không chọn dataset/architecture thay tác giả.
- Đã chạy: `test_pair_replay.py` (5). Blocker: `AUTHOR_ONLY_INFORMATION`.
- **F05 đã sửa (lượt R9)**: `data/datasets.py:34-42` dựng cache và đọc cả ba category; validator
  từng chỉ kiểm category được chọn. Nay dùng chung `constructor_input_blockers`.

### 5. `clvc_net`
- Upstream @ `bd9b68894898`. Local fix `resnet.py` dùng `torch.hub.load_state_dict_from_url`
  (fallback torchvision cũ) — **cùng URL ResNet-50**.
- Transform **đã sửa (F03)**: native FashionIQ val = `Resize(256)→CenterCrop(224)→ToTensor→ImageNet
  Normalize` (`train.py:90-96`); wrapper trước dùng `Resize(224)` (nhánh fashion200k) — cùng shape
  nhưng tensor khác tới 4.24 đơn vị chuẩn hoá trên ảnh 300×500.
- Checkpoint: cần **6** whole model (local+global × 3 category). Tên trong wrapper là **giả định**,
  chưa phải member của archive tác giả ⇒ blocker.

### 6. `tgcir`
- Upstream @ `65fa78eaf8ca`. **F07 đã sửa**: pin `clip==0.2.0` (clipboard CLI) → OpenAI CLIP
  revision `a1d07173…` (verify: `clip.py` 200, có `load` + `tokenize`).
- Wrapper `tgcir.py` **preflight-only**, luôn blocked cho tới khi xác minh member/hash của
  `TG-CIR.zip`. Không đổi nhãn thành "implemented".
- Native `test.py:6` = `test(params, model, testset, category)`, gallery pair-union, mask reference
  `-10e10` ⇒ khớp `fashioniq_val_split`.

### 7–9. `habit`, `intent`, `conesep`
- Habit @ `6ddc2cd63eac`, Intent @ `2a75fc6f57d1`, ConeSep @ `bd4bf629ebe6` (redirect từ
  `iLearn-Lab/CVPR26-ConeSep` sang `Lee-zixu/ConeSep`, đã ghi provenance).
- Cả ba publish **chỉ** N0.2 / N0.5 / N0.8. Evaluate trên clean validation là hợp lệ **nhưng phải
  label**: `checkpoint_training_noise_pct` và `evaluation_noise_pct` giữ **tách biệt**; không mô tả
  checkpoint noisy như clean-trained.
- Adapter truyền `--variant <checkpoint_id>` ⇒ biến thể được chọn chính là biến thể load.
- Parser: shared replay (đọc `categories` — F02).
- **F05 + F06 đã sửa (lượt R9)**, cùng cách như `hint`: tập input của constructor native được kiểm
  độc lập với `--variant`/`--category`, và `txt_processors` lấy từ config pinned thay vì dựng model
  BLIP-2 thứ hai (3 model này từng chạm `/root/bert-base-uncased` qua `Blip2Base.init_tokenizer`).

### 10. `airknow`
- Upstream @ `6ab6fa95b746`. Local fix: guard 2 import vắng + `.png` fallback (không đổi hành vi khi
  `.jpg` có mặt).
- **F01 đã sửa**: adapter truyền `--variant` (trước là `--checkpoint-id`, không wrapper nào nhận ⇒
  argparse exit 2 trước khi chạm model). Test ép cả `fiq_n05` và `fiq_n08`.
- Native gọi `model.train()` ở `test_BLIP2.py:10` cho nhánh FashionIQ ⇒ giữ nguyên, ghi là
  `SCIENTIFIC_VARIANT`; `--eval-mode` là biến thể thí nghiệm được label riêng.
- **F05 + F06 đã sửa (lượt R9)**: như `hint`. `--variant` (F01) và `eval_mode` không đổi; chỉ đường
  lấy `txt_processors` và tập input preflight thay đổi.

### 11. `dcnet`
- Upstream @ `68c79d38569f`. CLI thật: `python test.py --resume <run_dir>`.
- **F08 đã sửa (lượt R9)**: evaluator đọc `config.json:data_loader.args.data_dir` và `CE_dataset`
  join literal đó vào **CWD của tiến trình** (không có `chdir` nào trong DCNet; `evaluate_models.py`
  launch với `cwd=source_cwd(...)` = checkout). Bản trước chỉ từ chối *relative* và chỉ nằm trong
  `dcnet.preflight` — một helper mà `DCNetAdapter` **không** gọi, nên official path không được bảo vệ.
  Nay: resolve theo CWD, so với `--dataset-root` cho **cả** absolute lẫn relative, **và** chạy trong
  `backend/preflight.preflight_blockers` — danh sách mà `doctor`, `pc_readiness`, `prepare_model` và
  `evaluate_models` đều đọc. Symlink canonical (dataset stage đúng cách) pass; thư mục thường trong
  checkout hoặc path absolute trỏ nơi khác ⇒ **refuse** kèm đường dẫn cụ thể.
  **ĐÃ BỎ (R10-01, xác nhận lại ở R11):** override `WORKBENCH_DCNET_DATA_DIR`. Evaluator native
  (`ConfigParser`) không đọc biến môi trường, nên override chỉ khiến validator trả lời một câu hỏi
  khác với consumer — tức là nó *gây ra* đúng loại lỗi mà validator tồn tại để chặn. Quyết định
  binding nay chỉ đọc `config.json` của run và CWD của evaluator. Đừng set biến này.
  **R11 bổ sung:** mọi cấp JSON của config nay được kiểm shape trước khi `.get()` — `[]`,
  `{"data_loader": ["x"]}`, `{"data_loader": {"args": null}}` đều trả blocker nêu tên field thay vì
  ném `AttributeError` ra khỏi readiness view.
  Run directory giờ chép `config.json` vào evidence dir để `verify_run` re-parse được stdout.
- **F09 đã sửa (lượt R9)**: parser `dcnet_native_logger`. `model/metric.py:5-21` trả **fraction 0..1**
  ⇒ đổi % đúng một lần. `test.py:74-84` in cả `recall` lẫn `comb` bằng **cùng** vòng
  `trainer.categories` (`base_trainer.py:32` = `config['data_loader']['args']['categories']`); sáu ô
  `comb:` không có tên, nên bản trước map theo tuple cứng `(dress, shirt, toptee)` và **transpose im
  lặng** khi config dùng thứ tự khác (`comb_avg` không phát hiện được vì tổng bất biến với hoán vị).
  Nay: thứ tự lấy từ chính stream (`[cat] comp corr r@10`), config là **nhân chứng thứ hai** — lệch
  nhau ⇒ từ chối; thiếu config ⇒ vẫn đúng, vì stream tự nó xác định thứ tự. composition/correction
  giữ **riêng**.
- Blocker: `cap.{cat}.glove.val.pkl` chưa công bố; OOV embedding chưa seed ⇒ reconstruction
  deterministic là **biến thể tuỳ chọn**, không phải mặc định.

### 12. `csmcir`
- Upstream @ `774f94e2076f`. **2 approved patches** (đều verify được, và edit sau patch bị từ chối):
  1. `src/lavis/models/__init__.py` — 98 import unguarded của module không tồn tại chặn registration.
  2. **`src/validate_blip_csmcir.py` (F04 mới)** — evaluator in metric JSON rồi **vô điều kiện** chạy
     `vis_fiq`, nơi `labels.append(data[1])` là target-id string nhưng bị xử như tensor ⇒
     AttributeError, exit ≠ 0 **sau khi metric đã tính**. Patch đặt dump sau cờ `--visualize` mặc định
     tắt; công cụ vẽ không bị đụng.
- **F04 đã sửa (lượt R9)**: `approved_patch_state` **return ngay trong vòng lặp**, nên chỉ cần patch
  #1 ⇒ `applied=True` trong khi patch #2 chưa áp; và không consumer nào phân biệt partial với complete.
  Nay hàm kiểm **mọi** patch đã khai báo, trả thêm `patches[]` (từng mục một), và `detail` nói rõ
  "*N* of *M* required patches are not applied". Một checkout nửa vời vì vậy bị báo là
  `source repair incomplete` ngay trong `model_requirements`, và `source_blocker` vẫn còn hiệu lực.
  Cả hai patch đều cần cho workflow benchmark mặc định; `--visualize` vẫn **off** khi chạy benchmark.
- CoT captions **có công bố** tại Hub `peng12138/CSMCIR` (`fashioniq_qwen_captions/qwen_captions/`),
  byte-identical với bản local (sha256 từng category đã verify). Blocker "captions unavailable" đã
  **rút lại**.
- Entrypoint `src/validate_blip_csmcir.py` có `__main__` thật. Data root `<checkout>/fashionIQ_dataset`,
  ảnh `.png`. Checkpoint key `Blip2QformerCirAlignPromptCSMCIR`.

### 13–14. `combiner_rn50x4_noft`, `clip4cir_rn50x4_fullft`
- Cùng checkout `CLIP4Cir` @ `dfed9f748a8a`; **hai entry riêng**, không trộn cặp.
- Dims adapter truyền `--projection-dim 2560 --hidden-dim 5120` = **đúng** cho RN50x4
  (`validate.py:269-270`, feature_dim 640 ×4/×8). Ví dụ CIRR trong README (4096/8192) là **typo đã
  được tác giả xác nhận** (issue #9) — không "sửa lại" theo README.
- Release `weights` có 5 part `pretrained_models.tar.gz` kèm sha256 (ghi ở registry/agent report);
  **chưa tải** (9.8 GB trống, archive ~8.55 GB). Member: `FashionIQ/RN50x4_noft/fiq_comb_RN50x4_noft.pt`,
  `FashionIQ/RN50x4_fullft/fiq_comb_RN50x4_fullft.pt`, `FashionIQ/RN50x4_fullft/fiq_clip_RN50x4_fullft.pt`.
- noft = CLIP RN50x4 gốc + Combiner; fullft = **cặp** CLIP fine-tuned + Combiner cùng config.
- Protocol: full gallery, reference **eligible** (`validate.py:43-56`, `127-132`) ⇒ `fashioniq_original_split`.

### 15. `sprc`
- Upstream @ `2935a5397732`. **F14 đã sửa**: `backbone` được **khai báo** theo variant trong registry,
  adapter đọc khai báo đó; selection không khai báo ⇒ **refuse** (không đoán từ filename, vì loader
  dùng `strict=False` sẽ im lặng đánh giá model khởi tạo một phần).
- Variant: `sprc_fiq_unresolved` (`backbone: pretrain` = ViT-g) và `sprc_fiq_vitl`
  (`backbone: pretrain_vitL`, `scientific_variant: true`, có `variant_provenance` + comparability).
  ViT-L reachable (200, 1 459 756 541 B); OneDrive ViT-g 403; Baidu cần mật khẩu.
- Adapter truyền `--blip-model-name blip2_cir_align_prompt` (default pinned
  `blip2_cir_rerank_learn` **không được register** ở đâu).

### 16. `ptha_mtst`
- Upstream @ `6a02ee4763fc`. Local fix: guard 1 import vắng; `base_path` đọc `MTST_DATA_ROOT` với
  literal upstream làm fallback.
- Hub `yinanzhou1/mtst_ptha` publish **chỉ** CIRR (`tuned_mtst_cirr.pt` 2 720 352 973 B,
  `cirr_modifier_generated_5p_completed_rev.json`, `cirr_train_modifier_rev_long.json`) — **không có**
  FashionIQ weight. Không thay bằng CIRR, không random, không tự train.
- Protocol nếu có weights: `fashioniq_original_split` (`validate_blip.py:44-51` không có reference mask).

---

## 4. Findings F01–F16

| # | Quyết định | Thay đổi / lý do | Bằng chứng | Blocker còn lại |
|---|---|---|---|---|
| F01 | `FIXED` | adapter `airknow` dùng `--variant` (trước `--checkpoint-id`) | `test_component_boundaries.py` (2 variant + sweep 15 adapter) | — |
| F02 | `FIXED` | parser đọc cả `categories` và `results`; cả hai mà mâu thuẫn ⇒ từ chối | `test_component_boundaries.py` (4+3 model + abuse cases) | — |
| F03 | `FIXED` | `clvc_net.py` dùng `Resize(256)` theo nhánh FashionIQ native | `test_component_boundaries.py` so tensor thật | — |
| F04 | `FIXED` | `--visualize` off-by-default, qua approved patch thứ 2 | `test_component_boundaries.py` (4 test patch + gate) | — |
| F05 | `FIXED` | validator HINT/ENCODER/PAIR yêu cầu train; layout → `ilearn_resized_training`. **Lượt R9**: tập input của constructor native được kiểm bằng `constructor_input_blockers` cho **cả bảy** wrapper (`hint`, `encoder`, `pair`, `habit`, `intent`, `conesep`, `airknow`) thay vì chỉ theo category được chọn; bốn khai báo `fashioniq_layout` của habit/intent/conesep/airknow được nâng lên `[..._training]` vì layout cũ bỏ `cap.*.train.json` mà constructor vẫn mở | `test_r9_defect_regressions.py` (D1–D3, 19 test) + `test_verified_declarations.py` | — |
| F06 | `UNFINISHED_ENGINEERING` | Thay `load_model_and_preprocess` bằng processor lấy trực tiếp từ config pinned: **đã áp** cho `hint`, `habit`, `intent`, `conesep`, `airknow`. `lavis.processors` trong repo là bản pinned; `BeatsAudioProcessor` (module import `torchaudio`) bị bỏ khỏi import graph vì torchaudio không có trong môi trường khai báo của **bất kỳ** model nào và không liên quan tới text preprocessing — đây là một dòng import chết, không phải một bản thay thế hành vi. Đường code `load_model_and_preprocess` không còn được gọi (assert bằng AST), và `BlipCaptionProcessor` thật trả `'  A dog, running!  '` → `'a dog, running'`, `'Dress.'` → `'dress'`. Vẫn **không** có output parity so với invocation native đầy đủ, vì bản native cần `model_cls.from_pretrained` → `/root/bert-base-uncased` + trọng số BLIP-2 gốc mà host này không có; biên đó ghi trong docstring `workbench/replay/lavis_processors.py` và trong provenance của mỗi run (`text_processor.loaded_via`). | `test_r9_defect_regressions.py` (E1–E4, 15 test) + `workbench/scripts/lavis_processor_parity.py` (chạy được) | `FIXED_WITH_RECORDED_BOUNDARY` |
| F07 | `FIXED` | pin OpenAI CLIP revision thay `clip==0.2.0` | `test_verified_declarations.py` | — |
| F08 | `FIXED` | preflight báo `data_dir` mismatch/relative/missing | `dcnet.validate_dataset_binding` + test | — |
| F09 | `FIXED` | parser DCNet, fraction→% một lần, nhánh `comb:`, tách comp/corr | `test_verified_declarations.py` (6 test) | — |
| F10 | `FIXED` | ENCODER `VERIFIED_METADATA` + loader báo missing/unexpected | `test_verified_declarations.py`; HF API + Range read | `PUBLIC_ASSET_SETUP` |
| F11 | `BLOCKED_EXTERNAL` | tên 6 member của `CLVCNET.zip` chưa xác minh; `torch.load` whole-model trên torch ≥2.6 cần `weights_only=False` sau khi verify hash — **forward-compat, không phải lỗi torch 1.6.0** | agent report | `AUTHOR_ONLY_INFORMATION` |
| F12 | `ALREADY_FIXED` | `MHTransformer_rev.py` đã label `EXPERIMENTAL_RECONSTRUCTION`; B1/B2 mapping giữ UNVERIFIED | agent diff + registry | `AUTHOR_ONLY_INFORMATION` |
| F13 | `ALREADY_FIXED` | dims 2560/5120 đúng; noft/fullft tách; digest 5 part ghi lại | agent report | `PUBLIC_ASSET_SETUP` |
| F14 | `FIXED` | `backbone` khai báo theo variant; không khai báo ⇒ refuse | `test_verified_declarations.py` (3 test) | `PUBLIC_ASSET_SETUP` + `SCIENTIFIC_VARIANT` |
| F15 | `BLOCKED_EXTERNAL` | tái xác minh Hub: vẫn **không có** FashionIQ weight | agent report + HF API | `AUTHOR_ONLY_INFORMATION` |
| F16 | `ALREADY_FIXED` | 176 test LIMN/export/replay/fresh-clone pass; không sửa code LIMN | agent report + coordinator run | `ENVIRONMENT_RESOURCE_LIMIT` |
| R10-01 | `FIXED` | Bỏ `WORKBENCH_DCNET_DATA_DIR` khỏi quyết định binding — evaluator native không đọc biến đó, nên nó chỉ làm validator tin sai. Quyết định nay chỉ đọc `config.json` + CWD evaluator | ConfigParser **native thật**: link→selected đọc selected; link→wrong đọc `['WRONG']`; override không cứu được | — |
| R10-02 | `FIXED` | Khai `canonical_dataset_link: dataset/fashioniq`; stager tạo thư mục cha khi ghi thật; `validate_source` nhận contract đầy đủ; preflight dùng CWD checkout | Bản sao checkout: verified `True` trước/sau staging, `unauthorized_untracked == []`; dry-run không tạo parent | — |
| R10-03 | `FIXED` | `prepare()` giữ context suốt đời process; `register_path` idempotent khi cùng giá trị (khác giá trị vẫn raise); `reset()` cho tool nhiều checkout | Fixture `lavis/__init__.py` + checkpoint `torch.save` thật: load thành công, digest sai vẫn bị refuse | — |
| R10-04 | `FIXED` | Xoá 3 `source_blocker` lỗi thời (`airknow`/`conesep`/`intent`) | `preflight_blockers` thật: 3→2, 1→0, 1→0; blocker asset/runtime còn nguyên | — |
| R10-05 | `FIXED` | Bỏ `salesforce-lavis` (không thể đồng thời thoả `timm==0.9.16` của vendored `eva_vit.py`); **thêm `Adapter.launcher`/`launcher_roots`** và cho `_evaluator_imports` đi theo chương trình chạy thật thay vì `script` | Wheel sha256 khớp PyPI; scanner `hint` từ `{numpy,torch,tqdm}` thành có cả `transformers`/`timm`/`omegaconf` | 6 môi trường cài thật chưa chạy (xem §2 của `R10_REMEDIATION.md`) |
| R10-06 | `FIXED` (đã đính chính ở R11) | `inspect_environment_distributions` chỉ đọc base inventory khi interpreter **thực sự kế thừa**: `pyvenv.cfg` có `include-system-site-packages = true`. Nhánh `conda-meta` **đã bị xoá** — đó chỉ là một tên thư mục, ai cũng tạo được, và Conda env vốn cô lập nên nó tái tạo đúng lỗi R10-06 ở dạng khác | Venv thật: cô lập ⇒ `packaging` biến khỏi inventory; `--system-site-packages` ⇒ còn; prefix chỉ có `conda-meta/` ⇒ inventory **rỗng**; version sai vẫn bị chặn | — |

---

## 5. Điều kiện cập nhật về sau

1. Sửa source / checkpoint / preprocessing / protocol / runtime ⇒ phải ghi **phạm vi bằng chứng cần
   kiểm lại**. Kết quả cũ **không** tự xác minh cho code/weights mới.
2. `PREFLIGHT_PASS` **không** thành inference-ready bằng đổi tên status.
3. Full FashionIQ val cần **đủ 3 category** + aggregate đúng protocol; smoke 1 category không phải
   full benchmark.
4. Metric % phải đổi đơn vị đúng **một lần**; macro giữ quy ước native đã xác minh.
5. Thiếu per-query exporter phải ghi **thiếu** — không dựng 15 observer mới cho đẹp checklist.
6. Không chạy được GPU ⇒ ghi `PENDING` + command PC cụ thể. Không đánh dấu "không còn bug".
7. Ba protocol giữ **riêng**; không trộn cohort trong một bảng so sánh.

---

## 6. Việc cần làm trên PC, theo thứ tự

```bash
# 0. Môi trường cho chính workbench (KHÁC môi trường của từng model ở bước 4).
#    Danh sách này là mức tối thiểu để chạy được các lệnh bên dưới: pytest cho bước 2,
#    pyyaml cho registry, omegaconf cho preflight/wrapper, torch cho loader/parser,
#    fastapi+uvicorn cho API, packaging cho runtime verifier. opencv-python-headless là
#    dependency của chính các evaluator LAVIS (không có nó thì import closure của chúng thiếu),
#    nên nó thuộc MÔI TRƯỜNG CỦA MODEL, và được khai trong registry — bước 4 cài nó.
python -m venv .venv && . .venv/bin/activate
pip install pytest pyyaml omegaconf "torch" torchvision fastapi uvicorn pydantic httpx packaging

# 1. Lấy source đã kiểm
git clone git@github.com:Le-Minh-Nhut/cir.git && cd cir
git checkout research/cir-failure-workbench      # hoặc fetch + checkout đúng SHA đã ghi ở đầu file

# 2. Xác nhận clone tự verify (không cần checkpoint)
python -m pytest workbench/tests/test_fresh_clone_delivery.py \
  workbench/tests/test_vendored_source_integrity.py -q

# 3. Xem chính xác cái gì thiếu TRƯỚC khi tải gì
python workbench/scripts/download_checkpoints.py --all --dry-run
python workbench/scripts/download_auxiliary_assets.py --all --dry-run

# 4. Stage assets (theo model đang làm)
python workbench/scripts/download_checkpoints.py --model limn
python workbench/scripts/download_auxiliary_assets.py --model csmcir

# 4b. DCNet: evaluator của nó KHÔNG có cờ dataset-root — nó đọc `dataset/fashioniq` tính từ CWD
#     của chính nó. Link đó phải được stage (và đã được khai trong registry nên integrity gate
#     chấp nhận). Lệnh này tự tạo thư mục cha khi cần; --check-only/--dry-run không ghi gì.
python workbench/scripts/prepare_dataset.py --model dcnet --dataset-root "$FASHIONIQ_ROOT"
#     rồi xác nhận binding khớp với cái evaluator thật sẽ mở:
python -c "import sys; sys.path.insert(0,'.'); from pathlib import Path; \
from workbench.replay.dcnet import binding_mismatch; \
print(binding_mismatch(Path('<run_dir>'), Path('$FASHIONIQ_ROOT'), Path('workbench/third_party/DCNet')) or 'BOUND OK')"

# 5. Preflight (không load model) — phải báo blocker đúng, không crash
#    Preflight chỉ đọc registry + config, không import model, nên venv workbench là đủ.
python workbench/replay/hint.py  --checkpoint-root <dir> --dataset-root <FIQ> \
  --source-root workbench/third_party/HINT --preflight-only
python workbench/replay/pair.py  --checkpoint-root <dir> --dataset-root <FIQ> \
  --source-root workbench/third_party/PAIR --preflight-only

# 6. Tạo/chọn MÔI TRƯỜNG CỦA MODEL — bước này KHÔNG được bỏ.
#
#    Bước 4b ở trên chỉ TẢI asset. Nó không cài dependency, và chạy model bằng python của
#    venv workbench là SAI: mỗi model khai một `python_env_var` riêng (WORKBENCH_<MODEL>_PYTHON)
#    trỏ tới interpreter của chính nó, và runtime verifier chỉ chấp nhận interpreter đó.
#    Venv workbench cố ý mỏng (xem bước 0); nó KHÔNG có torch/timm/transformers/diffusers/peft
#    mà checkpoint class của các model LAVIS cần để unpickle.
#
#    Dùng flow provisioning có sẵn — nó đọc profile đã kiểm trong registry, tạo môi trường
#    CÁCH LY, rồi verify bằng interpreter thật của môi trường đó:
python workbench/scripts/manage_environment.py --list                 # xem trạng thái mọi model
python workbench/scripts/manage_environment.py --model hint --create \
  --allow-env-install --allow-network --dry-run                      # xem trước, chưa cài
python workbench/scripts/manage_environment.py --model hint --create \
  --allow-env-install --allow-network                                # tạo thật

#    Sau khi tạo, xác nhận trạng thái trước khi chạy (in ra tier/status/reason thật):
python -c "import sys; sys.path.insert(0,'.'); \
from workbench.backend.registry import model_by_id; \
from workbench.backend.runtime import verify_environment; \
r = verify_environment(model_by_id('hint'), probe=True); \
print(r.tier, r.status, r.reason)"

#    Chỉ khi tự cài tay (đã có interpreter riêng), export biến tương ứng. Ghi bằng đường dẫn
#    TUYỆT ĐỐI tới interpreter của môi trường model, KHÔNG phải `python` của venv workbench:
export WORKBENCH_HINT_PYTHON=/path/to/hint-env/bin/python
export WORKBENCH_HABIT_PYTHON=/path/to/habit-env/bin/python
#    ... v.v. cho intent, conesep, airknow, sprc, limn, encoder, ...

# 7. Chạy thật. Repo KHÔNG có `workbench.cli` — dùng đúng entrypoint bên dưới.
#    Đặt các biến đường dẫn thật, có dấu ngoặc kép (đừng để placeholder `<...>` trong shell):
FIQ="/path/to/FashionIQ"                       # canonical FashionIQ root
CKPT="/path/to/checkpoints/hint"               # run directory có fashioniq.pt
SRC="$PWD/workbench/third_party/HINT"

# 7a. Kiểm trước, không load model — chạy được bằng venv workbench:
"$WORKBENCH_HINT_PYTHON" workbench/replay/hint.py --preflight-only \
  --checkpoint-root "$CKPT" --dataset-root "$FIQ" --source-root "$SRC"

# 7b. Chạy thật — bằng interpreter CỦA MODEL, đúng ba cờ bắt buộc của wrapper:
"$WORKBENCH_HINT_PYTHON" workbench/replay/hint.py \
  --checkpoint-root "$CKPT" --dataset-root "$FIQ" --source-root "$SRC"

#    Wrapper hỗ trợ thêm: --batch-size, --backbone, --device, --category (một category),
#    --preflight-only, --output. Chạy `--help` để xem đúng tham số của từng model:
"$WORKBENCH_HINT_PYTHON" workbench/replay/hint.py --help
python workbench/replay/limn.py --help     # LIMN dùng các cờ riêng của nó
```
> **Đã sửa ở R12:** bản trước ghi `python -m workbench.cli replay ...`; module đó **không tồn tại**
> trong repo, nên lệnh không chạy được. Câu giải thích kèm theo ("wrapper spawn interpreter đúng qua
> CLI này") cũng mô tả code không có. Wrapper đọc `python_env_var` qua `runtime.python_executable`
> khi nó tự spawn, nhưng bản thân nó phải được gọi bằng interpreter của model.

> **Đính chính (R11):** các đoạn cũ của tài liệu này mô tả `conda-meta` như bằng chứng một môi
> trường kế thừa package của base, và mô tả `WORKBENCH_DCNET_DATA_DIR` như một override còn hiệu
> lực. Cả hai đã bị bỏ: Conda env vốn cô lập, và override đó không được evaluator native đọc nên
> chỉ tạo ra một validator trả lời khác câu hỏi của consumer. Đừng cấu hình theo chúng.

### 6b. LIMN — quy trình ĐÃ CHẠY THẬT trên PC này

Dùng đúng chuỗi lệnh dưới đây; nó đã tạo ra số đo ở §3.1 và artifact per-query đã publish.

```bash
cd /path/to/cir
FIQ="$PWD/data/fashionIQ_dataset"          # `FashionIQ` là symlink cùng thư mục
CKPT="$PWD/workbench/artifacts/checkpoints/limn"

# (1) Interpreter của LIMN: pin lịch sử torch 1.12.1 + open_clip 2.20.0, KHÔNG dùng venv workbench.
uv venv --python 3.10 workbench/envs/limn-historical
uv pip install --python workbench/envs/limn-historical/bin/python \
  --index-url https://download.pytorch.org/whl/cu113 "torch==1.12.1+cu113" "torchvision==0.13.1+cu113"
uv pip install --python workbench/envs/limn-historical/bin/python \
  "open_clip_torch==2.20.0" "numpy<2" ftfy regex tqdm

# (2) PHẢI chạy bước này một lần: wheel torch 1.12.1 khai GNU_STACK RWE, kernel/loader hiện đại
#     từ chối và `import torch` chết với "cannot enable executable stack". Xoá cờ đó trên
#     libtorch_cpu.so. Bản đã sửa nằm trong workbench/envs/limn-historical, nên bước này chỉ cần
#     khi tạo lại môi trường từ đầu.
workbench/envs/limn-historical/bin/patchelf --clear-execstack \
  workbench/envs/limn-historical/lib/python3.10/site-packages/torch/lib/libtorch_cpu.so

export WORKBENCH_LIMN_PYTHON="$PWD/workbench/envs/limn-historical/bin/python"

# (3) Preflight (không load model) rồi chạy that qua entrypoint có audit.
"$WORKBENCH_LIMN_PYTHON" workbench/replay/limn.py --preflight-only \
  --checkpoint-root "$CKPT" --dataset-root "$FIQ" --source-root "$PWD/workbench/third_party/LIMN"

export WORKBENCH_LIMN_BATCH_SIZE=4                       # GPU 4 GiB; 32 là thiết lập paper
export WORKBENCH_EVALUATOR_ENV_PASSTHROUGH=WORKBENCH_LIMN_BATCH_SIZE
python -m workbench.scripts.evaluate_models --model limn \
  --run-id "$(date -u +%Y%m%dT%H%M%SZ)_limn_base_iter0_all_categories_fashioniq_val_split"
```

- `--run-id` phải **mới**: orchestrator từ chối ghi đè artifact của một run id đã tồn tại.
- `WORKBENCH_EVALUATOR_ENV_PASSTHROUGH` là cơ chế có sẵn để chuyển biến vận hành qua allowlist;
  đừng nới `_EVALUATOR_ENV_ALLOWLIST` và **đừng** đưa `WORKBENCH_LEDGER_KEY` vào đó.
- Sau khi chạy: rebuild index rồi xem UI.
```bash
python -c "import sys;sys.path.insert(0,'.');from workbench.backend.index import rebuild_index;print(rebuild_index())"
```
Trong UI: chọn protocol **FashionIQ - Val Split (Pair-Union Gallery, Reference Excluded)**,
vào **Sample Explorer** — 6016 query sẽ hiện kèm Top-K.

**Thứ tự chuẩn bị artifact:**

1. **LIMN** (3 ckpt, mapping VERIFIED, có exporter) — chạy được ngay, cho cả per-query parity.
2. **HINT** (`fashioniq.pt`) — mapping VERIFIED, parser sẵn.
3. **ENCODER** (backbone LAION ViT-B/32 + `fashioniq.pt`) — mapping VERIFIED.
4. **CSMCIR** — apply 2 patch, CoT captions đã có, ckpt 2.7 GB + BLIP-2/EVA-ViT-g caches.
5. **HABIT/INTENT/ConeSep/Air-Know** — checkpoint noisy, phải ghi nhãn noise khi báo cáo.
6. **CLIP4Cir ×2** — tải và xác minh 5 part trước khi giải nén.
7. **SPRC** — chỉ ViT-L public; ghi rõ đây là variant khác paper.
8. **PAIR / CLVC-Net / TG-CIR / DCNet / PTHA+MTST** — cần **tác giả** cung cấp mapping/weights.

**Artifact cần chuẩn bị (không commit vào Git):** FashionIQ canonical root (`captions/`,
`image_splits/`, `images/*.png`, `resized_image/{cat}/*.jpg`, `qwen_captions/`), checkpoint root theo
`workbench/artifacts/checkpoints/<model_id>/`, auxiliary caches (`~/.cache/lavis`,
`~/.cache/torch/hub/checkpoints`, `~/.cache/clip`).
