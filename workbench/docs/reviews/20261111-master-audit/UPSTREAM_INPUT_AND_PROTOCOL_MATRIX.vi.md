# Ma trận upstream, đầu vào và protocol — xác minh độc lập

Phạm vi: nửa "upstream" của audit tại `92ed66de4f359ef479da18bf8939723b6f9313e5`, nhánh
`research/cir-failure-workbench`. Mọi khẳng định dưới đây đến từ (a) clone sạch checkout detached
đúng SHA trong `workbench/registry/models.yaml`, (b) lần trực tiếp `data/FashionIQ`, hoặc (c) một lệnh
đã chạy mà output được dán lại. Chỗ nào không kiểm được ghi `UNVERIFIED` kèm lý do.

Máy chạy: `workbench/envs/ctl312h/bin/python`, `git`, 20 CPU. Không GPU, không chạy suite, không sửa
`workbench/registry/*.yaml`. Mọi clone nằm trong `mktemp -d` riêng.

---

## Bảng tổng hợp

| # | Upstream | URL trong master file | SHA pin | Gallery | Reference | Rank call | Metric | Config phục vụ |
|---|---|---|---|---|---|---|---|---|
| 1 | Air-Know | `iLearn-Lab/CVPR26-Air-Know` | `6ab6fa95…bce99b7` ✔ | full category | loại, scorer-mask | `np.argsort(-sims)`, maxpool | R@1/10/50, % | `airknow` |
| 2 | ConeSep | `iLearn-Lab/CVPR26-ConeSep` | `bd4bf629…5b` ✔ | full category | loại, scorer-mask | `np.argsort(-sims)`, maxpool | R@1/10/50 | `conesep` |
| 3 | HABIT | `iLearn-Lab/AAAI26-HABIT` | `6ddc2cd6…ee306` ✔ | full category | loại, scorer-mask | `np.argsort(-sims)`, maxpool | R@1/10/50 | `habit` |
| 4 | INTENT | `iLearn-Lab/AAAI26-INTENT` | `2a75fc6f…72f51f3` ✔ | full category | loại, scorer-mask | `np.argsort(-sims)`, maxpool | R@1/10/50 | `intent` |
| 5 | DCNet | `ozmig77/dcnet` | `68c79d38…cc5fb` ✔ | full category (`*_trg`) | loại, mask **hai** ma trận | **`torch.topk(dim=0)`** ×3 | R@10/50, 3 nhánh | `dcnet` |
| 6 | CLIP4Cir | `ABaldrati/CLIP4Cir` | `dfed9f74…bdf13` ✔ | full category (`classic`) | **giữ** | **`torch.argsort`** ↑ | R@10/50, `assert` | `clip4cir_rn50x4_fullft` |
| 6b | CLIP4Cir | (cùng repo) | (cùng SHA) | full category | **giữ** | **`torch.argsort`** ↑ | R@10/50 | `combiner_rn50x4_noft` |
| 7 | MTST+PTHA | `zyinan99/MTST_PTHA` | `6a02ee47…91d86` ✔ | full + `exists()` filter | giữ | **`torch.argsort`** ↑ | R@10/50 | `ptha_mtst` |
| 8 | TME | `li-shuxian/TME` | `394aad79…065062` ✔ | full category + cache | giữ (mask CIRR-only) | **`torch.argsort`** ↑ | R@1/5/10/50 | `tme` |
| 9 | HINT | `iLearn-Lab/ICASSP26-HINT` | `bec50b6c…ea5b2` ✔ | pair-union | loại, scorer-mask | `np.argsort(-sims)`, maxpool | R@1/10/50 | `hint` |
| 10 | ENCODER | `iLearn-Lab/AAAI25-ENCODER` | `29a2a31d…7018` ✔ | pair-union | loại, scorer-mask | `np.argsort(-sims)` | R@1/10/50 | `encoder` |
| 11 | PAIR | `iLearn-Lab/ICASSP25-PAIR` | `bfb0b98c…d888f` ✔ | pair-union | loại, scorer-mask | `np.argsort(-sims)` | R@1/10/50 | `pair` |
| 12 | CLVC-Net | `iLearn-Lab/SIGIR21-CLVC-Net` | `bd9b6889…279cb` ✔ | pair-union (`gallery_all=False`) | loại, scorer-mask | `np.argsort(-sims)` | R@1/10/50 | `clvc_net` |
| 13 | TG-CIR | `iLearn-Lab/MM23-TG-CIR` | `65fa78ea…91bdee` ✔ | pair-union | loại, scorer-mask | `np.argsort(-sims)` | R@1/10/50 | `tgcir` |
| 14 | LIMN | `iLearn-Lab/TPAMI24-LIMN` | `7d7bc9b1…d3c3e` ✔ | pair-union | loại, scorer-mask | `np.argsort(-sims)` | R@1/10/50 | `limn` |
| 15 | CSMCIR | `qzp2018/CSMCIR` | `774f94e2…1a44e` ✔ | full category | giữ (mask CIRR-only) | `torch.argsort` ↑ | R@10/50 | `csmcir` (gác) |
| 16 | SPRC | `chunmeifeng/SPRC` | `2935a539…e36f0f` ✔ | full category | giữ (mask CIRR-only) | `torch.argsort` ↑ | R@10/50 | `sprc` (gác) |

Cột "Rank call" là của **đường FashionIQ val** trong từng repo, không phải nhánh CIRR hay submission.
"↑" = sắp **tăng dần** trên `distances = 1 - cos` (không phải giảm dần trên similarity).

---

## A. Định danh source

### A.1 Cách làm

Với mỗi repo: `git init` trong một `mktemp -d` riêng, `git remote add origin <URL>`,
`git fetch --depth 1 origin <SHA>`, `git checkout --detach FETCH_HEAD`. Sau đó ghi
`git rev-parse HEAD`, `git rev-parse HEAD^{tree}`, `git show -s --format='%cI|%an <%ae>|%s'`.
Kết quả thô: `/tmp/aud/clone_status.tsv`, `/tmp/aud/redirects.tsv`, `/tmp/aud/sha_refs.tsv`,
`/tmp/aud/repo_meta.tsv`, `/tmp/aud/diff_<Repo>.json`.

**16/16 SHA pin giải mã được và HEAD của clone trùng khít full 40 ký tự với registry.** Không có
`FETCH_FAIL`, không `CHECKOUT_FAIL`, không repo nào cần fallback fetch đầy đủ.

### A.2 Bảng định danh đầy đủ

| Repo | Full SHA | Tree hash | Commit date (ISO) | Author | Subject |
|---|---|---|---|---|---|
| AirKnow | `6ab6fa95b746f85b08eea4cb66eb51448bce99b7` | `457a88158d45687d327dc2afc746640ed4e36c0a` | 2026-09-28T14:40:56+08:00 | Zhiheng Fu | Update README.md |
| ConeSep | `bd4bf629ebe68045c1a54dcc61e4a3ea65f8759b` | `657dacf55ea6171d2ba70c67760971ed2adf3455` | 2026-05-25T17:59:18+08:00 | Zixu Li (李子旭) | Update README.md |
| HABIT | `6ddc2cd63eac1f65693f4e821c34fb3b42aee306` | `66baa74e4e31cb3af79667d2828df7fe4f54ce52` | 2026-05-25T18:02:56+08:00 | Zixu Li (李子旭) | Update README.md |
| INTENT | `2a75fc6f57d1a036bf3167a7fe1d2cee072f51f3` | `5997ea1066ad813d8c8ba99eff7f87780bd45bb9` | 2026-05-25T18:15:51+08:00 | Zixu Li (李子旭) | Update README.md |
| DCNet | `68c79d38569f01ca39c3b4ea641b3b7eb2ccb5fb` | `d95bc5afa18ea0f2eb2d211050cee27f46bd09f4` | 2021-03-15T11:42:57+09:00 | jongseok kim | Readme typo |
| CLIP4Cir | `dfed9f748a8a4d05abf613164e06d59b22dbdf13` | `09656243029f0f57cbe28222e8c85200e1528854` | 2023-09-05T09:31:52+02:00 | Alberto Baldrati | Update README.md |
| MTST_PTHA | `6a02ee4763fc2260b6abfb18be78853facb91d86` | `8739e24bc033b06fb8de3ac2b0f68cbc77b33180` | 2025-02-09T18:52:27+08:00 | Yinan Zhou | Update contents |
| TME | `394aad79df2c104feb32a2c96132582012765062` | `4dc1d8a72b13d2def75e6fc0daea1ee5f88c0110` | 2026-10-19T10:12:30+08:00 | Li Shuxian | docs: add live Papers with Code CIRR leaderboard badge |
| HINT | `bec50b6c8c19111893b617979502b948d1cea5b2` | `5dedd588be5dc467ccfba61766b9e154970a405d` | 2026-04-09T16:38:14+08:00 | Mingyu Zhang | Create LICENSE |
| ENCODER | `29a2a31d6a56f677bf450c3be7cdaef423fb7018` | `243fd56fe31bff40d99281d916da6f221e9ee005` | 2026-05-25T18:00:29+08:00 | Zixu Li (李子旭) | Update README.md |
| PAIR | `bfb0b98cdb4c44f6b9190baaa84d783a77dd888f` | `8f33157b83ad62ed551a1cdc12b1de0ec8ecb12a` | 2026-04-05T15:58:53+08:00 | Zhiheng Fu | Update README.md |
| CLVCNet | `bd9b6889489806cb60676597880f91d918b279cb` | `f0a688840434bac342b20af9ff9c757451007a16` | 2026-04-05T18:54:18+08:00 | Haokun Wen | Update README.md |
| TGCIR | `65fa78eaf8cabe8197fcc22ccab42207c591bdee` | `7ba66b0937a4363a53d277660a240400b204c5e6` | 2026-04-09T16:03:25+08:00 | Haokun Wen | Update README.md |
| LIMN | `7d7bc9b116f594a65ac22457491edf28a88d3c3e` | `6535743a188c5d21a61c2cbf15e51181b593266c` | 2026-04-05T19:33:05+08:00 | Haokun Wen | Update README.md |
| CSMCIR | `774f94e2076ff17ea91703a6239d2a08f0e1a44e` | `2fff43c84c8e6253abd2c39d17e106007f7aa136` | 2026-04-27T16:30:50+08:00 | Zhipeng Qian | Update README.md |
| SPRC | `2935a5397732260d1db6fa577e5926f963e36f0f` | `4f959b47e1d808348597c793a5f0645182684af5` | 2024-04-16T21:29:29+08:00 | Yang Bai | Update README.md |

### A.3 Redirect và identity URL

`curl -sIL -o /dev/null -w '%{url_effective} %{http_code} %{num_redirects}'` cho **cả 16 URL**:
**0/16 redirect, 0/16 trả về URL khác, cả 16 là HTTP 200 tại đúng URL đã ghi.**

Kiểm thêm qua GitHub API (`/repos/<owner>/<name>`): **0/16 là fork** (`fork: false` ở cả 16), nên
không repo nào là bản sao lệch của một upstream khác.

**Kết luận A.3: 0/16 URL sai hoặc redirect.** Cảnh báo trong master file rằng ma trận cũ từng ghi sai
URL cho INTENT, HINT, PAIR, CLVC-Net, DCNet, TG-CIR và LIMN **không tái hiện trên HEAD hiện tại** —
bảy URL đó hiện đều 200 tại chỗ, không redirect, SHA pin giải mã được. Cái còn lại của cảnh báo là
**hình dạng tên**: thư mục trên đĩa (`source_dir`) khác tail URL ở **11/16** repo, nhưng đó là quy ước
đặt tên nội bộ trong `models.yaml`, không phải lỗi định danh:

| Repo | tail URL | `source_dir` | khớp? |
|---|---|---|---|
| AirKnow | `CVPR26-Air-Know` | `AirKnow` | khác |
| ConeSep | `CVPR26-ConeSep` | `ConeSep` | khác |
| HABIT | `AAAI26-HABIT` | `HABIT` | khác |
| INTENT | `AAAI26-INTENT` | `INTENT` | khác |
| DCNet | `dcnet` | `DCNet` | khác (hoa/thường) |
| HINT | `ICASSP26-HINT` | `HINT` | khác |
| ENCODER | `AAAI25-ENCODER` | `ENCODER` | khác |
| PAIR | `ICASSP25-PAIR` | `PAIR` | khác |
| CLVCNet | `SIGIR21-CLVC-Net` | `CLVCNet` | khác |
| TGCIR | `MM23-TG-CIR` | `TGCIR` | khác |
| LIMN | `TPAMI24-LIMN` | `LIMN` | khác |
| CLIP4Cir | `CLIP4Cir` | `CLIP4Cir` | khớp |
| MTST_PTHA | `MTST_PTHA` | `MTST_PTHA` | khớp |
| TME | `TME` | `TME` | khớp |
| CSMCIR | `CSMCIR` | `CSMCIR` | khớp |
| SPRC | `SPRC` | `SPRC` | khớp |

### A.4 Nhánh/ref chứa SHA pin — hai điểm cần biết

`git ls-remote` cho từng URL, lọc theo SHA pin:

- **CLIP4Cir: SHA `dfed9f74` chỉ nằm ở `refs/tags/weights`.** `refs/heads/master` là
  `56bb547909d57bd8b5d2ef7ba55149962f77d482`. Pin vào **tag** vẫn hợp lệ (commit bất biến) nhưng nghĩa
  là "trạng thái repo tại thời điểm tag `weights`", không phải master HEAD. Phải ghi rõ khi trích dẫn.
- **AirKnow: SHA `6ab6fa95` không nằm ở `HEAD`/`main` hiện tại.** Vẫn fetch được bằng SHA trực tiếp,
  nhưng `main` đã tiến lên sau pin; mọi trích dẫn phải kèm full SHA, không được nói "Air-Know main".

13 repo còn lại: SHA pin **là** `HEAD` và đầu nhánh mặc định (`main`, trừ TME là `master`).

### A.5 Đối chiếu clone ↔ `workbench/third_party/**`

Cách làm: Python walk cả hai cây (bỏ `.git`, `__pycache__`, `.ipynb_checkpoints`), so `sha256` từng
file. Chạy 4 tiến trình song song.

| Repo | file upstream | file vendor | chỉ upstream | chỉ vendor | file khác nội dung |
|---|---:|---:|---:|---:|---|
| AirKnow | 1219 | 1220 | 0 | 1 | **2** |
| CLIP4Cir | 14 | 15 | 0 | 1 | **0** |
| CLVCNet | 9 | 10 | 0 | 1 | **1** |
| CSMCIR | 357 | 362 | 0 | 5 | **2** |
| ConeSep | 1222 | 1223 | 0 | 1 | **0** |
| DCNet | 39 | 40 | 0 | 1 | **1** |
| ENCODER | 25 | 28 | 0 | 3 | **1** |
| HABIT | 1235 | 1236 | 0 | 1 | **0** |
| HINT | 1233 | 1227 | 7 | 1 | **0** |
| INTENT | 1229 | 1230 | 0 | 1 | **0** |
| LIMN | 33 | 34 | 0 | 1 | **0** |
| MTST_PTHA | 361 | 362 | 0 | 1 | **3** |
| PAIR | 21 | 23 | 0 | 2 | **0** |
| SPRC | 355 | 356 | 0 | 1 | **0** |
| TGCIR | 11 | 12 | 0 | 1 | **0** |
| TME | 352 | 357 | 1 | 6 | **4** |

(1 = `provenance.json` do workbench tự thêm, trừ các dòng ghi rõ.)

Tám repo có vendor **byte-identical** với clone ngoài `provenance.json`: CLIP4Cir, ConeSep, HABIT,
INTENT, LIMN, PAIR, TGCIR, SPRC.

| Repo | File khác/chỉ-vendor | Loại | Căn cứ |
|---|---|---|---|
| CLVCNet | `resnet.py` | **adaptation có chủ đích** | `provenance.json:local_fixes` (`origin: model_recovery`); `torchvision.models.utils` → `torch.hub` có fallback. Chỉ import. |
| DCNet | `base/base_trainer.py` | **adaptation có chủ đích** | `local_fixes`; thêm `map_location=self.device` và `weights_only=False` có điều kiện `inspect.signature`, kèm comment. Không đổi công thức. |
| AirKnow | `data_utils.py` | **adaptation có chủ đích** | `local_fixes`; thêm `_resolve_img_path()` fallback `.jpg`→`.png`. |
| AirKnow | `lavis/models/__init__.py` | **adaptation có chủ đích** | `local_fixes`; bọc import trong `try/except` cho 98 module tác giả không commit. |
| MTST_PTHA | `.gitattributes` | **adaptation có chủ đích** | `local_fixes` justification: `* text=auto` sẽ viết lại CRLF của 124 file đã vendor và phá digest; thay bằng `* -text -diff`. |
| MTST_PTHA | `MTST_PTHA/src/data_utils.py` | **adaptation có chủ đích** | `local_fixes`; `base_path` đọc `MTST_DATA_ROOT`. |
| MTST_PTHA | `MTST_PTHA/src/lavis/models/__init__.py` | **adaptation có chủ đích** | import-guard như AirKnow. |
| TME | `precompute.sh` | **adaptation có chủ đích** | `local_fixes`: bỏ `nohup` CIRR chạy nền. |
| TME | `src/utility.py` | **adaptation có chủ đích** | `local_fixes`: kiểm CUDA trước khi chọn device. |
| TME | `src/precompute.py`, `src/precompute_evaluation.py` | **adaptation có chủ đích** | cùng nhóm device/CUDA; không đổi rank/metric. |
| CSMCIR | `src/lavis/models/__init__.py`, `src/validate_blip_csmcir.py` | **adaptation có chủ đích** | provenance có patch/base/derived sha256 cho cả hai. |
| ENCODER | `evaluate_model.py` | **adaptation có chủ đích** | `local_fixes` `origin: model_recovery`; đổi `import datasets1` → `import datasets as datasets1`, phù hợp với shim 23 byte dưới đây. |
| ENCODER | `datasets1.py` | **shim 23 byte, không phải loader** | `wc -c` = 23; nội dung đúng một dòng `from datasets import *`. Đây là compatibility shim của tác giả; **không** phải workbench dựng loader khác. |
| ENCODER | `open_clip_pytorch_model.bin` | artifact, không phải source | trọng số, không nằm trong git upstream. |
| CSMCIR | `COT_ours2/fashioniq/*_cot_val.json` (3) | **input bổ sung, không có upstream** | không có trong clone; xem §15. |
| CSMCIR | `src/lavis/models/blip2_models/..._shiyan92.pyc` | artifact build | `.pyc`, không ảnh hưởng logic đọc được. |
| TME | `features/FashionIQ/val_images.pt`, `val_name_to_idx.json`, `log_TME/other/precomputation.log`, `weight/*.pth` | artifact/cache | cache precompute + trọng số. |
| HINT | 7 file `.idea/*` | **loại khi vendoring, không phải drift** | `provenance.json:excluded_by_vendoring` liệt kê đúng 7 đường dẫn. |
| TME | `blip2_instruct_vicuna7b copy.yaml` (chỉ upstream) | **loại khi vendoring** | `excluded_by_vendoring`. |

**Không có file nào là "unknown drift".** Mọi khác biệt hoặc khớp `provenance.json:local_fixes` /
`excluded_by_vendoring`, hoặc là artifact không phải source (trọng số, cache, `.pyc`), hoặc là
`provenance.json` do workbench tự sinh. Đối chiếu chéo: mọi mục `local_fixes` trong 16
`provenance.json` đều xuất hiện thực tế trong danh sách trên, và không mục nào trong danh sách trên
vắng mặt khỏi `provenance.json` (trừ artifact/cache/`provenance.json`).

### A.6 Đối chiếu `workbench/replay/**`

`replay/` không chứa bản sao source upstream; nó import từ `third_party` lúc chạy, nên đối tượng so
khớp là `third_party` (đã làm ở A.5). Kiểm chứng: 22 file Python trong `replay/` đều là wrapper, không
có `datasets.py`/`test.py` của tác giả. Ví dụ `workbench/replay/tgcir.py` chỉ 95 dòng và tự mô tả là
"Read-only preflight … never loads checkpoint data, constructs replacement weights, initializes CLIP,
or runs inference" (`replay/tgcir.py:1-6`). Vì vậy "so với replay" = "so third_party + kiểm wrapper gọi
đúng hàm native"; phần gọi-đúng-hàm thuộc mục C.

### A.7 Đối chiếu chéo registry ↔ master file ↔ clone

- **Registry ↔ clone:** 16/16 SHA và 16/16 URL khớp tuyệt đối (17 model entry, CLIP4Cir dùng chung cho
  2 config).
- **Registry ↔ `provenance.json`:** `upstream_commit` trong cả 16 provenance khớp registry; cùng
  `upstream_commit_date` khớp commit date đo từ `git show` (bảng A.2). Không lệch.
- **Master file ↔ clone:** 16/16 SHA viết tắt trong master file là tiền tố đúng của full SHA.
- **Khác biệt không phải lỗi:** registry ghi ConeSep là `https://github.com/Lee-zixu/ConeSep`
  (`models.yaml:251`), master file ghi mirror `iLearn-Lab/CVPR26-ConeSep`. Hai URL trỏ **cùng commit**
  `bd4bf629…` (kiểm `git ls-remote` cả hai). Registry có comment giải thích (`models.yaml:253`). Vẫn là
  **disagreement về URL cần ghi**: ai trích master file sẽ dẫn người đọc tới mirror, không tới repo tác
  giả.
- **Một pin "lỏng" thực sự:** Air-Know pin không còn trên `main` (§A.4). Registry không ghi điều này.

---

## B. Đếm dataset FashionIQ

### B.1 File và revision dùng

| Loại | Đường dẫn | sha256 (16 ký tự đầu) |
|---|---|---|
| Query val dress | `/home/heheboiz/data/cir/data/FashionIQ/captions/cap.dress.val.json` | `5e5117d45695df9c` |
| Query val shirt | `…/captions/cap.shirt.val.json` | `7b7ca3797b85dddf` |
| Query val toptee | `…/captions/cap.toptee.val.json` | `b4e09f428e6c255c` |
| Full gallery dress/shirt/toptee | `…/image_splits/split.{dress,shirt,toptee}.val.json` | (đếm ở B.3) |
| Correction dict | `…/captions/correction_dict_{dress,shirt,toptee}.json` | `9e4a59877098fe26` / `ca29e29e23c0a198` / `3dfbf15fa6d3a30d` |

`FASHIONIQ_ROOT=/home/heheboiz/data/cir/data/FashionIQ` (`workbench/config/workbench.env`). Hash đầy đủ
sinh bằng `hashlib.sha256` trên bytes thô. Dataset revision: **không có manifest/revision trong
`data/FashionIQ`** — không README, không tag, không `.git`. Vì vậy "revision" định danh bằng đúng bộ
sha256 ở trên. Nhánh nguồn khai báo là `XiaoxiaoGuo/fashion-iq`, nhưng **UNVERIFIED** rằng file local
khớp byte với một commit cụ thể của repo đó — không clone repo đó trong phạm vi này và không có digest
tham chiếu để so.

### B.2 Số query native val

Đếm `len(json.load(...))` trên từng `cap.<cat>.val.json`; mỗi phần tử có `target`, `candidate`,
`captions`.

| Category | Đo được | Kỳ vọng | Khớp? |
|---|---:|---:|---|
| dress | **2017** | 2017 | ✔ |
| shirt | **2038** | 2038 | ✔ |
| toptee | **1961** | 1961 | ✔ |
| **tổng** | **6016** | 6016 | ✔ |

Không phần tử nào thiếu khoá ở ba file val. `distinct(target)` = số query ở cả ba category, nên 6016
là số annotation, không phải số ảnh.

### B.3 Gallery full category

`len(json.load(...))` trên `image_splits/split.<cat>.val.json`, kèm `len(set(...))`.

| Category | Đo được | distinct | Kỳ vọng | Khớp? |
|---|---:|---:|---:|---|
| dress | **3817** | 3817 | 3817 | ✔ |
| shirt | **6346** | 6346 | 6346 | ✔ |
| toptee | **5373** | 5373 | 5373 | ✔ |

Không file nào có ID trùng, nên full gallery = đúng độ dài file. Đây là **list có thứ tự**:
`json.load` trả `list`, loader upstream dùng `images.index(...)` hoặc enumerate, nên thứ tự file là
thứ tự gallery. Không được sort lại.

### B.4 Gallery pair-union

Quy tắc tôi dùng, và **quy tắc này là quyết định**, vì thứ tự đổi rank:

```
seen = set(); order = []
for q in cap.<cat>.val.json:            # đúng thứ tự annotation trong file
    for key in ('candidate', 'target'): # reference TRƯỚC, target SAU
        v = q[key]
        if v not in seen:
            seen.add(v); order.append(v)
```

Đây là "first-seen, reference-then-target", đúng vòng lặp
`if i['source_img_id'] not in test_targets_id: test_targets_id.append(...)` rồi mới tới target, mà
`TGCIR/datasets.py:149-156`, `LIMN/LIMN/datasets.py`, `HINT.FaD8fy/datasets.py:154-159`,
`ENCODER.JZt48G/datasets.py:185-196`, `PAIR.jqTJq4/data/datasets.py:146-160`,
`CLVCNet.PhSSgX/datasets.py:239-246` dùng.

| Category | Đo được | Kỳ vọng | Khớp? |
|---|---:|---:|---|
| dress | **2628** | 2628 | ✔ |
| shirt | **3089** | 3089 | ✔ |
| toptee | **2902** | 2902 | ✔ |

Kiểm thêm: cả ba union đều là **tập con của full split** (`union ⊆ set(full)` = True ở cả ba).

### B.5 Thứ tự: pair-union KHÔNG phải "full split đã lọc"

So hai thứ tự trên cùng một tập ID:

- **A** = first-seen ref-then-target trên `cap.<cat>.val.json` (thứ tự upstream dùng).
- **B** = duyệt `split.<cat>.val.json`, giữ lại ID thuộc union.

| Category | n_union | A == B? | max lệch vị trí | trung bình lệch | số ID đổi vị trí |
|---|---:|---|---:|---:|---:|
| dress | 2628 | **False** | 2606 | 450.5 | 2625 / 2628 |
| shirt | 3089 | **False** | 3069 | 519.6 | 3086 / 3089 |
| toptee | 2902 | **False** | 2768 | 487.1 | 2899 / 2902 |

Nếu dựng pair-union bằng lọc full split, vị trí của gần như mọi ứng viên lệch trung bình ~450–520 bậc.
`np.argsort` sắp theo score nên vị trí gallery chỉ quyết định **thứ tự hoà điểm**, nhưng khi có hoà
điểm (embedding trùng, ảnh trùng, lượng tử hoá) thì rank đổi thật. Mọi run pair-union phải ghi
"first-seen ref-then-target", không chỉ "pair-union".

### B.6 TG-CIR: log cũ khớp pair-union hay full split?

- `TGCIR/datasets.py:127-159` (`get_test_data`) đọc **cả hai**: `split.<cat>.val.json` vào `images`
  (`:129-130`) để lấy **vị trí** (`images.index(candidate)`, `:143`), và `cap.<cat>.val.json` vào
  `ref_captions` (`:131-132`) để lấy **danh sách query**.
- Gallery dựng từ `test_queries`, **không** từ `images`:
  `TGCIR/datasets.py:149-156`
  ```python
  test_targets_id = []
  for i in test_queries:
      if i['source_img_id'] not in test_targets_id:
          test_targets_id.append(i['source_img_id'])
      if i['target_img_id'] not in test_targets_id:
          test_targets_id.append(i['target_img_id'])
  ```
  và `test_targets` dựng bằng `for i in test_targets_id:` (`:157-162`).
- `test_queries` lặp theo `ref_captions` (`:139`), mỗi phần tử đẩy **source trước, target sau** ⇒
  gallery là **first-seen ref-then-target union**, đúng B.4. Số đo 2628/3089/2902 khớp.
- Vị trí gallery **không** phải vị trí trong split: `source_img_id` là index trong split file (`:143`),
  nhưng `test_targets_id.index(...)` (`test.py:65,67`) là index trong **union**.

**Kết luận:** gallery TG-CIR là pair-union, 2628/3089/2902, thứ tự first-seen ref-then-target. Hồ sơ
cũ gán full split là **SAI**. Master file đúng ở điểm này. Lưu ý phụ: TG-CIR vẫn **phụ thuộc** file
full split để chạy (dùng `images.index()`), dù gallery không dùng nó — có thể là nguồn gốc của việc
gán nhầm.

### B.7 Số gallery mà `workbench/registry` khai

`native_protocol` khai **tên protocol**, không khai số gallery:

| `native_protocol` | Model | Gallery thực tế theo source |
|---|---|---|
| `fashioniq_full_gallery_ref_excluded` | airknow, conesep, habit, intent, dcnet | full category, mask reference ở scorer |
| `fashioniq_original_split` | ptha_mtst, combiner_rn50x4_noft, clip4cir_rn50x4_fullft, tme, csmcir, sprc | full category, reference eligible |
| `fashioniq_val_split` | hint, encoder, pair, clvc_net, tgcir, limn | pair-union, mask reference ở scorer |

`workbench/docs/MODEL_STATUS_SPEC.md:207` định nghĩa `fashioniq_val_split` = "pair-union (ordered) |
excluded". Từ "ordered" **không đủ**: không nói ordered *theo cái gì*, và B.5 chứng minh hai cách
ordered cho hai thứ tự khác nhau trên 99.9% ID. Đề nghị sửa thành "first-seen ref-then-target over
cap.<cat>.val.json" (đề xuất text, không sửa file).

---

## C. Protocol theo từng cấu hình

Quy ước: mọi `file:line` trỏ vào **clone pin**, không phải `third_party`, trừ khi ghi rõ. Đường dẫn
viết tắt: `$A` = `/tmp/aud/clone/AirKnow.IHVqGe`, v.v. (bảng A.2).

Cột "macro" trong mọi mục: FashionIQ native không có một hàm macro chung; ba category được đánh giá
độc lập rồi lấy trung bình ở **entrypoint**, không ở hàm metric. Với CLIP4Cir/PTHA/TME/CSMCIR/SPRC
entrypoint in `mean(average_recall10_list)`; với Air-Know/ConeSep/HABIT/INTENT/HINT/ENCODER/PAIR/
CLVC-Net/TG-CIR/LIMN entrypoint in ba số theo category và (nếu có) cộng dồn, nên "macro" là việc của
người đọc log. Ghi rõ ở từng mục.

---

### 1. Air-Know — cấu hình `airknow`

- **Entrypoint:** `$A/test_BLIP2.py:9` `def test(params, model, testset, category, txt_processors)`.
  Nhánh FashionIQ là `$A/test_BLIP2.py:82-88` (dress/shirt/toptee lấy `testset.test_queries_*`).
  Launcher `$A/train_BLIP2.py` gọi nó trong vòng eval; cấu hình dataset ở
  `$A/train_BLIP2.py` (`load_dataset` → `datasets.FashionIQ(..., transform=img_transform)`).
- **Loader / gallery:** `$A/datasets.py:181-222` `get_test_data`. Đọc
  `image_splits/split.<cat>.val.json` (`:183`) vào `images`, đọc
  `captions/cap.<cat>.val.json` (`:185`) vào `ref_captions`. `self.split = 'original-split'`
  hard-code ở `$A/datasets.py:31` (không có flag), nên nhánh `elif self.split == 'original-split':`
  (`:218-225`) luôn thắng: gallery = **toàn bộ** `images`, `test_targets_id.append(id)` cho mọi index.
  **Full category gallery, đúng file order.**
- **Reference:** **loại ở scorer-time**. `$A/test_BLIP2.py:151`
  `sims[i, test_targets_id.index(t['source_img_id'])] = -10e10` — chạy trước `argsort` ở `:154`.
  Không lọc ở load-time (gallery vẫn đủ 3817/6346/5373), không xoá ở metric-time.
- **Rank call:** `$A/test_BLIP2.py:154`
  `nn_result = [np.argsort(-sims[i, :])[:50] for i in range(sims.shape[0])]` —
  `np.argsort` trên **phủ định** similarity, tức giảm dần; quicksort không stable, **không có tie
  policy**. Trước đó `:144` `sims = sims.max(-1)` — điểm cuối cùng là **max-pool theo token/query
  token**, không phải dot thẳng.

- **Image pipeline:**
  - decode: `$A/datasets.py:155-160` (`get_img`) / `:166-171` (`get_test_img`):
    `PIL.Image.open(...)` rồi `.convert('RGB')`. Không xử lý EXIF, không grayscale/RGBA path riêng.
  - transform: `$A/datasets.py` dùng **một** `preprocess` cho cả query và gallery; nó được dựng ở
    launcher từ `targetpad_transform(target_ratio=1.25, dim=224)` / `squarepad_transform(224)`
    (`$A/data_utils.py:76-108`), chuỗi
    `TargetPad(1.25, 224) → Resize(224, BICUBIC) → CenterCrop(224) → ToTensor → Normalize(CLIP mean/std)`
    (`$A/data_utils.py:99-108`).
  - **Eval dùng val-transform, không train-transform.** Không có augmentation trong eval: transform
    là Compose tất định. `ToTensor` cho float32 CHW trên [0,1] rồi trừ mean/chia std.
- **Text:** `$A/datasets.py:129-133` `concat_text` = `"{correct_text(c0)} and {correct_text(c1)}"`;
  `correct_text` (`:122-127`) lowercase, xoá dấu câu bằng `str.maketrans({k:' ' for k in
  string.punctuation})`, rồi map qua `correction_dict_<cat>.json` (**file local trong
  `data/FashionIQ/captions/`, không phải file trong repo** — 3 file có mặt tại
  `data/FashionIQ/captions/correction_dict_{dress,shirt,toptee}.json`).
  Tokenizer: `txt_processors["eval"]` của LAVIS (`$A/test_BLIP2.py:112`), `BlipCaptionProcessor` với
  `max_words=50` (`$A/lavis/processors/blip_processors.py:30,65-66` — cắt cứng theo **từ**, không phải
  token), rồi vào `self.tokenizer(..., padding="max_length", truncation=True, max_length=
  self.max_txt_len)` với `max_txt_len=32` (`$A/lavis/models/blip2_models/blip2_cir.py:110`). BOS là
  `[CLS]` của BERT, EOS `[SEP]`, PAD `[PAD]`; token thứ 32 là query token (xem dưới).
  Pooling: `fusion_output.last_hidden_state[:, 32, :]` (`blip2_cir.py` trong
  `extract_retrieval_compose`) — vị trí 32 là **query token**, không phải mean-pool.
- **Checkpoint:** FashionIQ eval **không** load checkpoint trong `test_BLIP2.py`. Loader duy nhất là
  `$A/utils.py:139-140` `torch.load(checkpoint); model.load_state_dict(checkpoint['state_dict'])` —
  **strict mặc định** (`strict=True` không bị override). Tuy nhiên đường eval FashionIQ của repo này
  không đi qua `utils.load_checkpoint`; `train_BLIP2.py` huấn luyện rồi eval trong cùng process. Với
  workbench, đường "load rồi eval" là **ngoài source pin** và phải ghi là adaptation.
  Tensor sinh lúc construction: `self.query_tokens` (`blip2_cir.py`) và `learnable_relation` (nếu bật)
  — xem §10 (ENCODER) cho cùng vấn đề; ở Air-Know `learnable_relation` **không** xuất hiện.
- **Metric:** `$A/test_BLIP2.py:157-166`: `for k in [1,10,50]: r = 100*r/len(nn_result)`.
  Mẫu số = `len(nn_result)` = **số query đã chấm** = 2017/2038/1961. Không có query bị bỏ (không lọc
  load-time), nên coverage = 1.0. Macro: entrypoint không tự lấy trung bình; log in ba category.
- **Khác biệt với claim của workbench:** `MODEL_STATUS_SPEC.md`/`PROTOCOL_MATRIX.csv` ghi
  "reference masked with -10e10 before argsort" cho Air-Know — **đúng**. Nhưng CSV ghi
  `tie_policy=np.argsort(-sims)` chung, và bỏ mất `sims.max(-1)` ở `:144`. Đó là khác biệt toán học
  thật: score không phải dot product thuần.

**REFUTED (từ ma trận cũ):** dòng Air-Know trong `PROTOCOL_MATRIX.csv` ghi
`mask=sims[i, ref_index] = -10e10 at test_BLIP2.py:151` — **đúng**. Nhưng `image_pipeline=targetpad
1.25 / 224` bỏ sót `sims.max(-1)` tại `test_BLIP2.py:144`, tức bỏ mất một phép gộp điểm. Đồng thời
`score_orientation=higher cosine better` sai về bản chất: sau `max(-1)` giá trị là max của inner
product theo token, không phải cosine.

---

### 2. ConeSep — cấu hình `conesep`

- **Entrypoint:** `$C/test.py:9` `def test(params, model, testset, category, txt_processors)`.
  FashionIQ: `$C/test.py:12-85`.
- **Loader / gallery:** dùng **cùng** `get_test_data` pattern như Air-Know:
  `$C/datasets.py:108-... `; `self.split = 'original-split'` hard-code ở `$C/datasets.py:27`.
  Gallery = full `split.<cat>.val.json`, file order. `$C/datasets.py` là bản khác của Air-Know
  (md5 khác: `f9afeb7d…` vs `de3d8943…`) nhưng cùng logic split.
- **Reference:** scorer-time mask, `$C/test.py:68` `sims[i, test_targets_id.index(...)] = -10e10`,
  trước `argsort` ở `:71`.
- **Rank call:** `$C/test.py:71` (và lặp lại y hệt ở `:75` — dòng thứ hai **thừa**, tính lại cùng một
  `nn_result`) `np.argsort(-sims[i, :])[:50]`. Không tie policy.
- **Điểm gộp:** `$C/test.py:64` `sims = sims.max(-1)` — **giống Air-Know**, max-pool theo token.
  **Feature normalization bị comment out** ở `$C/test.py:124-127`; nghĩa là score là inner product
  của vector **chưa normalize**. Đây là khác biệt thật so với ENCODER/PAIR/CLVC/TG-CIR/LIMN (những repo
  normalize).
- **Image pipeline:** `$C/datasets.py` + `$C/data_utils.py`; `data_utils.py` md5 `8e43336d…` **giống
  hệt** Air-Know, HABIT, INTENT → cùng `TargetPad(1.25,224) → Resize(224,BICUBIC) → CenterCrop(224) →
  ToTensor → CLIP normalize`. Một transform cho cả query và gallery.
- **Text:** `$C/datasets.py:115-118` `concat_text` = `correct_text(c0) + " and " + correct_text(c1)`;
  `correct_text` ở `:108-113` giống Air-Know (lowercase, punctuation→space, correction dict).
  Tokenizer `txt_processors["eval"]` (`$C/test.py:35`), max_words=50, `max_txt_len=32`.
- **Checkpoint:** `$C/utils.py:139-140` strict. Đường eval FashionIQ không load; xem ghi chú §1.
- **Metric:** `$C/test.py:73-83`, `100*r/len(nn_result)`, `k∈{1,10,50}`, mẫu số = số query.
- **Refuted:** CSV cũ ghi `mask=sims[i, ref] = -10e10 at test.py:68` — đúng. Nhưng ghi
  `image_pipeline=targetpad 1.25 / 224` mà **không** nói normalization bị tắt
  (`test.py:124-127`) trong khi ENCODER/PAIR normalize. So sánh score tuyệt đối giữa ConeSep và
  ENCODER là không hợp lệ về thang đo.

---

### 3. HABIT — cấu hình `habit`

- **Entrypoint:** `$H/test.py:9` `def test(params, model, testset, category, txt_processors)`;
  FashionIQ `$H/test.py:105-171`. **`model.eval()` ở `$H/test.py:10`** — khác Air-Know.
- **Loader / gallery:** `$H/datasets.py:28` signature có `split='original-split'` **mặc định**;
  launcher `$H/train.py:108-112` không truyền `split`, nên nhánh gốc thắng. Gallery = full
  `split.<cat>.val.json` (`$H/datasets.py:206-217` đường `original-split`), file order.
- **Reference:** scorer-time mask `$H/test.py:151`, trước `argsort` `:154`.
- **Rank call:** `$H/test.py:154` `np.argsort(-sims[i, :])[:50]`.
- **Điểm gộp:** `$H/test.py:136-138` `sims = np.matmul(...); sims = sims.squeeze(); sims = sims.max(-1)`
  — max-pool. Normalize **bị comment out** ở `:133-136`.
- **Image pipeline:** `$H/data_utils.py` md5 `8e43336d…` = Air-Know/ConeSep/INTENT → TargetPad 1.25,
  224, BICUBIC, CenterCrop, CLIP normalize. `$H/datasets.py:151-159` `get_test_img` dùng
  `self.transform` (một transform duy nhất).
- **Text:** `$H/datasets.py:112-121` `correct_text`/`concat_text` giống hệt Air-Know. Tokenizer
  `txt_processors["eval"]` (`$H/test.py:111`). `max_txt_len=32`
  (`$H/lavis/models/blip2_models/HABIT.py:136`), pooling ở `last_hidden_state[:, 32, :]`
  (`HABIT.py:409-411`) — query token, không mean.
- **Checkpoint:** `$H/utils.py:139-140` strict. Không có load trong `test.py`.
- **Metric:** `$H/test.py:160-170`, `100*r/len(nn_result)`, k∈{1,10,50}.

---

### 4. INTENT — cấu hình `intent`

- **Entrypoint:** `$I/test.py:9` (file `test.py`), FashionIQ `$I/test.py:104-170`. `model.eval()` ở
  `$I/test.py:10`.
- **Loader / gallery:** `$I/datasets.py:24` `split='original-split'` mặc định. Gallery = full split,
  file order (`$I/datasets.py:200-210`).
- **Reference:** scorer-time mask `$I/test.py:150`, `argsort` `:153`.
- **Rank call:** `$I/test.py:153` `np.argsort(-sims[i, :])[:50]`.
- **Điểm gộp:** `$I/test.py:135-137` `np.matmul → squeeze → max(-1)`. Normalize comment out `:132-135`.
- **Image:** `$I/data_utils.py` md5 `8e43336d…`, giống 3 repo trên.
- **Text:** `$I/datasets.py` `correct_text`/`concat_text` cùng dạng; tokenizer `txt_processors["eval"]`
  `$I/test.py:111`.
- **Checkpoint:** launcher `$I/train_INTENT.py:347-384` chỉ `torch.save`; **không có đường load
  checkpoint** trong repo pin. `$I/utils.py:139-140` có helper strict nhưng `train_INTENT.py` không gọi.
  Vì vậy "eval từ checkpoint" cho INTENT là **UNVERIFIED trong source**; workbench phải tự viết
  loader.
- **Metric:** `$I/test.py:156-166`, `100*r/len(nn_result)`, k∈{1,10,50}.
- **Ghi chú refuted:** CSV cũ ghi mask `at test.py:150` — **đúng**. Không có claim sai khác đáng kể ở
  INTENT ngoài việc bỏ `max(-1)`.

---

### 5. DCNet — cấu hình `dcnet`

- **Entrypoint:** `$D/test.py:17` `def test(config)` → `$D/test.py:65`
  `metric = trainer._valid_epoch(save_textatt=True)` với `TrainerJoint`. FashionIQ val với
  `--mode val`.
- **Loader / gallery:** `$D/data_loader/CE_dataset.py:22-26`
  `self.no_target = (split == 'val_trg' or split == 'test_trg')`; khi đó
  `self.data = read_json(data_dir + f"/image_splits/split.{category}.{split.split('_')[0]}.json")`
  → **full category gallery** từ `split.<cat>.val.json`, file order. DataLoader chọn `val_trg` cho
  gallery ở `$D/data_loader/data_loaders.py:179-202`. Query từ `split='val'` (`:172`) đọc
  `cap.<cat>.glove.val.pkl` (`CE_dataset.py:31-33`) — **không** đọc JSON caption.
- **Reference:** **scorer-time mask, áp cho CẢ HAI** ma trận điểm:
  `$D/trainer/trainer_joint.py:217-218`
  ```python
  score_matrix_comp[ei, id2targetidx[elem['candidate']]] = -10e10
  score_matrix_corr[ei, id2targetidx[elem['candidate']]] = -10e10
  ```
  Đây là điểm khác biệt: mask phải áp hai lần, và nếu chỉ áp một lần thì nhánh còn lại vẫn trả về
  reference.
- **Rank call:** **`torch.topk`, không phải argsort**, và **ba lần cho ba score khác nhau**:
  `$D/trainer/trainer_joint.py:228` (score_comp), `:232` (score_corr), `:240` (score_comb) —
  ```python
  _, topk = score.topk(dim=0, k=50, largest=self.largest, sorted=True)
  ```
  `largest` = `True` từ config (`$D/configs/ce/fashioniq_dcnet.json:78` `"largest": true`),
  đọc ở `$D/base/base_trainer.py:30`. `torch.topk(sorted=True)` là sort ổn định theo index khi bằng
  điểm trong implementation CUDA/CPU hiện hành — nhưng **repo không khai báo policy**, nên tie policy
  vẫn phải coi là "implementation-defined".
  **`dim=0`** chứ không phải `dim=-1`: mỗi cột là một query, mỗi hàng là một gallery item. Ma trận
  score vì vậy có shape `(N_gallery, N_query)` — **ngược** với mọi repo khác trong 16. Đây là lỗi dễ
  mắc nhất khi port.
  Nhánh `comb` cộng **log-softmax chứ không cộng score thô**: `$D/trainer/trainer_joint.py:239`
  `score_comb = F.log_softmax(score_comp[j], dim=-1) + F.log_softmax(score_corr[j], dim=-1)`.
  Metric cuối báo cả ba: `metric['recall']` (comp, corr) và `metric['comb']`.
- **Image pipeline:**
  - decode: `$D/data_loader/CE_dataset.py:66-78` `Image.open(...).convert('RGB')`.
  - ảnh **đã resize sẵn** trong `resized_images/` (`CE_dataset.py:19`), nên eval transform là
    `CenterCrop(224) → ToTensor → Normalize(ImageNet mean/std)` — **không có Resize**
    (`$D/data_loader/data_loaders.py:66-70`, nhánh `split in ['val','val_trg','test','test_trg']`).
    Train transform khác: `RandomCrop(224) → RandomHorizontalFlip → ToTensor → Normalize`
    (`:26-32`).
  - **Khác biệt lớn:** normalize là **ImageNet**, không phải CLIP
    (`data_loaders.py:28-29,48-49,67-70`). Không TargetPad.
- **Text:** `$D/configs/ce/fashioniq_dcnet.json` `max_text_words: 30`, `text_feat: "glove"`,
  `text_dim: 300`. Vector từ đã **precompute** vào `cap.<cat>.glove.val.pkl`; eval chỉ đọc
  `self.data[index]['wv']` (`CE_dataset.py:54-59`). Không có tokenizer/BOS/EOS ở eval — text là
  ma trận GloVe. Correction dictionary **không** được áp ở eval.
- **Checkpoint:** `$D/base/base_trainer.py:193-...` (bản vendor đã patch) `torch.load(ckpt_path, ...)`.
  Source pin: `$D/base/base_trainer.py:192` `checkpoint = torch.load(ckpt_path)` — **không** có
  `map_location`, nên checkpoint lưu từ CUDA không load được trên CPU. Patch vendor thêm
  `map_location` + `weights_only=False` có điều kiện; **đây là adaptation cần thiết, không phải drift**
  (xem A.5). Load vào model qua `self.model.load_state_dict(...)` strict mặc định.
- **Metric:** `$D/model/metric.py:12-21` `compute_score`: mẫu số `n = len(solution)` = số query,
  `sum(r@10)/n`, `sum(r@50)/n`. Macro: `$D/trainer/trainer_joint.py:245-249`
  `metric['recall_avg'] = metric['recall'][:,:2].mean()`, `comb_avg = metric['comb'].mean()` — trung
  bình trên **category × metric**, không phải trên query.
- **Refuted:** CSV cũ ghi `gallery_kind=full split (val_trg loader)` — đúng. Ghi
  `reference_policy=reference masked in BOTH score matrices (trainer_joint.py:217-218)` — **đúng**.
  Nhưng ghi `tie_policy=np.argsort(-sims)` và `rank_definition=1 + #{scores greater}` — **sai**:
  DCNet dùng `torch.topk(dim=0, ...)` ba lần và còn cộng log-softmax cho nhánh comb. Thêm nữa CSV ghi
  `image_pipeline=images pre-resized to 256 then CenterCrop(224) -> ImageNet norm (no TargetPad)` —
  nửa đầu ("pre-resized to 256") là **suy đoán**: source chỉ đọc từ `resized_images/` và không ghi
  kích thước gốc; kích thước thật phụ thuộc artifact trên đĩa. Ghi `UNVERIFIED` cho 256.

---

### 6. CLIP4Cir — hai cấu hình riêng biệt

Hai cấu hình **dùng chung** `src/validate.py` nhưng khác **entrypoint huấn luyện** và khác
**`--combining_function`**. Không bao giờ gộp.

#### 6a. `clip4cir_rn50x4_fullft`

- **Entrypoint:** `$L/src/validate.py:...` hàm `main` in `average recall10/50`; eval
  `compute_fiq_val_metrics` ở `$L/src/validate.py:21-57`. Cấu hình full-FT được tạo bởi
  `$L/src/clip_fine_tune.py` (train CLIP), rồi eval bằng `validate.py` với `--combining-function sum`.
- **Gallery:** `$L/src/validate.py:127-128`
  ```python
  classic_val_dataset = FashionIQDataset('val', [dress_type], 'classic', preprocess)
  index_features, index_names = extract_index_features(classic_val_dataset, clip_model)
  ```
  `classic` mode đọc `image_splits/split.<cat>.val.json` (`$L/src/data_utils.py:144-147`) →
  **full category gallery, file order**, `index_names` giữ đúng thứ tự đó
  (`$L/src/utils.py:33-44`).
- **Reference:** **GIỮ.** Nhánh `relative` split `val` trả `(reference_name, target_name, captions)`
  (`$L/src/data_utils.py:165-167`) và gallery không bị lọc; `sims`/`distances` không có dòng mask nào
  trong `compute_fiq_val_metrics` (`$L/src/validate.py:21-57`). Mask duy nhất trong file là ở nhánh
  **CIRR** (`$L/src/validate.py:159`), không chạy cho FashionIQ.
- **Rank call:** **`torch.argsort`, không phải numpy**, và **tăng dần trên `distances = 1 - cos`**:
  `$L/src/validate.py:44-46`
  ```python
  distances = 1 - predicted_features @ index_features.T
  sorted_indices = torch.argsort(distances, dim=-1).cpu()
  ```
  `torch.argsort` mặc định **không stable**; không tie policy.
- **Image pipeline:** `$L/src/data_utils.py` `targetpad_transform(target_ratio, input_dim)` →
  `TargetPad(1.25, dim) → Resize(dim, BICUBIC) → CenterCrop(dim) → ToTensor → Normalize(CLIP mean/std)`
  với `dim = clip_model.visual.input_resolution` (`$L/src/validate.py:283-286`, `:276`).
  Với RN50x4 `input_resolution = 288`, nên landscape là **288**, không phải 224. Một transform cho
  query và gallery. **Eval dùng val transform** (không augmentation, không RandomResizedCrop).
- **Text:** `$L/src/validate.py:88-93`
  ```python
  flattened_captions: list = np.array(captions).T.flatten().tolist()
  input_captions = [f"{flattened_captions[i].strip('.?, ').capitalize()} and {flattened_captions[i+1].strip('.?, ')}" ...]
  text_inputs = clip.tokenize(input_captions, context_length=77)
  ```
  → **deterministic** (khác train, nơi `generate_randomized_fiq_caption` chọn ngẫu nhiên 1 trong 4
  dạng ở `$L/src/utils.py:57-78`). Tokenizer là **OpenAI CLIP** `clip.tokenize`, `context_length=77`,
  BOS/EOS tự thêm (`[SOT]`/`[EOT]`), PAD id 0, truncate mặc định `True`. **Không** dùng correction
  dictionary — CLIP4Cir pin không có `correct_text`.
- **Checkpoint:** `$L/src/validate.py:284-287`
  `saved_state_dict = torch.load(args.clip_model_path, ...); clip_model.load_state_dict(saved_state_dict["CLIP"])`
  — strict mặc định, key bọc trong `"CLIP"`. Với `--combining-function sum` không có checkpoint
  combiner.
- **Metric:** `$L/src/validate.py:48-55`
  ```python
  assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())
  recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
  ```
  Mẫu số = `len(labels)` = **số query trong batch cuối cùng đã collate**. `collate_fn`
  (`$L/src/utils.py:81-88`) **bỏ** phần tử `None`, nên nếu một ảnh không đọc được thì số query giảm và
  mẫu số giảm theo — đây là lý do master file nói "CLIP4Cir có thể bỏ sample lỗi ở collate". `assert`
  buộc **đúng một** positive khớp tên target; trùng ID sẽ làm assert nổ, không âm thầm đếm hai.
  Macro: `$L/src/validate.py:391-393` `mean(average_recall10_list)` trên **ba category**.

#### 6b. `combiner_rn50x4_noft`

- **Entrypoint:** cùng `$L/src/validate.py` nhưng `--combining-function combiner`; train ở
  `$L/src/combiner_train.py` (CLIP đóng băng, chỉ train Combiner).
- **Gallery / reference / rank / metric:** **giống hệt 6a** vì dùng chung
  `compute_fiq_val_metrics`. Không có nhánh mask cho FashionIQ.
- **Khác biệt thật:** `combining_function = combiner.combine_features`
  (`$L/src/validate.py:303-307`), và `combiner.load_state_dict(state_dict["Combiner"])`
  (`:307`) **strict** với key bọc `"Combiner"`. Kiến trúc `Combiner(feature_dim, projection_dim,
  hidden_dim)` (`$L/src/combiner.py`). `--hidden-dim` mặc định `640*8` (`validate.py:270`).
  `combiner.eval()` được gọi tường minh (`:306`) — quan trọng vì Combiner có dropout.
- **Vì sao phải tách:** hai cấu hình dùng **cùng** gallery/rank/metric nhưng **khác checkpoint và khác
  hàm fusion**. Gộp chúng thành một "CLIP4Cir" là gộp hai artifact khác nhau; gộp `FullFT` vào
  `Combiner` còn làm sai cả `--clip-model-path` (FullFT cần checkpoint CLIP đã fine-tune; NoFT dùng
  CLIP gốc + Combiner).

---

### 7. MTST + PTHA — cấu hình `ptha_mtst`

- **Entrypoint:** `$P/MTST_PTHA/src/validate_blip.py:210-229` `fashioniq_val_retrieval` →
  `compute_fiq_val_metrics` (`:24-57`).
- **Gallery:** `$P/MTST_PTHA/src/validate_blip.py:224-226`
  `FashionIQDataset('val', [dress_type], 'classic', preprocess)` →
  `extract_index_features` (`$P/MTST_PTHA/src/utils.py:20-44`). `classic` đọc
  `image_splits/split.<cat>.val.json` và **lọc tồn tại file**:
  `$P/MTST_PTHA/src/data_utils.py:168-172`
  ```python
  for image_name in self.image_names[:]:
      file_path = base_path / 'fashionIQ_dataset' / 'images' / f"{image_name}.png"
      if not file_path.exists():
          self.image_names.remove(image_name)
  ```
  → **full category gallery, file order, sau khi bỏ ảnh thiếu**. Trên host này **0 ảnh bị bỏ** vì
  `data/FashionIQ/images/` có 74381 file `.png` (đo: `ls … | wc -l` = 74381) và mọi ID trong 3 split
  đều giải được. Nhưng filter **có** chạy và sẽ đổi gallery nếu host khác thiếu ảnh — đây là điều
  kiện phải ghi theo run.
- **Reference:** **giữ**, không mask. `sims`/`distances` không có dòng gán `-10e10` trong
  `compute_fiq_val_metrics` (`:37-57`). Nhánh `val` của dataset trả `target_name` (không trả ảnh
  reference), nhưng reference vẫn **nằm trong index** vì index là toàn bộ `image_names`.
- **Rank call:** `$P/MTST_PTHA/src/validate_blip.py:44-46`
  `distances = 1 - pred_sim; sorted_indices = torch.argsort(distances, dim=-1).cpu()` —
  **`torch.argsort` tăng dần trên distance**.
- **Image pipeline:** `preprocess` được dựng ở entrypoint; dạng chuỗi giống CLIP4Cir
  `targetpad_transform(target_ratio, input_dim)`. **UNVERIFIED chính xác `input_dim`/`target_ratio`**
  trong `validate_blip.py` vì entrypoint đọc từ `argparse` ở file khác; cần đọc `validate_blip.py`'s
  `main` để chốt. Điều chắc chắn: một transform duy nhất cho query và gallery, dùng
  `BICUBIC` + `CenterCrop` + CLIP normalize theo `$P/MTST_PTHA/src/data_utils.py`.
- **Text:** `$P/MTST_PTHA/src/data_utils.py` dùng caption thô từ `cap.<cat>.val.json`; `txt_processors`
  truyền vào `generate_fiq_val_predictions`. Token: LAVIS BERT tokenizer,
  `max_txt_len` mặc định 32 (theo `lavis/models/blip2_models/…`, cùng họ với Air-Know). Caption
  **không** `correct_text` — PTHA pin không có correction dict (kiểm: `grep -rn correct_text` không
  có trong `MTST_PTHA/src/data_utils.py`).
- **Checkpoint:** không có `torch.load` trong `validate_blip.py`; checkpoint được nạp ở launcher.
  `strict` mặc định khi `load_state_dict` — **UNVERIFIED cụ thể dòng** vì launcher dùng
  `torch.load(...)` rồi load model; cần đọc `validate_blip.py`'s `main`.
- **Metric:** `$P/MTST_PTHA/src/validate_blip.py:49-55`, `assert` đúng một positive, `100*sum/len`,
  k∈{10,50}. Macro: `$P/MTST_PTHA/src/blip_validate.py:85-90`
  `average_recall_at10 = mean(recalls_at10)`, `average_recall_at50 = mean(recalls_at50)`,
  `average_recall = (mean(r50) + mean(r10)) / 2` — trung bình trên **ba category**.
- **Chốt lại transform (đã đóng UNVERIFIED):** `$P/MTST_PTHA/src/validate_blip.py:523` `input_dim = 224`;
  transform mặc định `targetpad` với `target_ratio=1.25` (`:498-500`, `:528-530`). Vậy chuỗi ảnh là
  `TargetPad(1.25, 224) → Resize(224, BICUBIC) → CenterCrop(224) → ToTensor → Normalize(CLIP)`,
  **224 chứ không phải 288** — khác CLIP4Cir RN50x4 (288).
- **Checkpoint — `strict=False`, đã đóng UNVERIFIED:** `$P/MTST_PTHA/src/validate_blip.py:521-522`
  ```python
  checkpoint = torch.load(blip_model_path, map_location=device)
  msg = blip_model.load_state_dict(checkpoint[blip_model.__class__.__name__], strict=False)
  ```
  **Non-strict**, key bọc theo `blip_model.__class__.__name__`, và code chỉ **in** `missing_keys` chứ
  không kiểm. Đây là khác biệt thật so với CLIP4Cir/CSMCIR (strict mặc định): một checkpoint thiếu
  layer vẫn "load thành công" ở PTHA và cho ra số liệu không cảnh báo.
- **Phát hiện quan trọng — entrypoint pin KHÔNG có nhánh FashionIQ:**
  `$P/MTST_PTHA/src/validate_blip.py:497-511` `main()` chỉ gọi `blip_validate_cirr` khi
  `args.dataset == 'CIRR'`; **không có nhánh `fashionIQ`**. Đường FashionIQ tiên quyết là
  `$P/MTST_PTHA/src/blip_fine_tune_2.py:400,422-427`
  ```python
  if args.dataset.lower() == 'cirr':   clip_finetune_cirr(**training_hyper_params)
  elif args.dataset.lower() == 'fashioniq':
      training_hyper_params.update({'train_dress_types': ['dress','toptee','shirt'],
                                    'val_dress_types': ['dress','toptee','shirt']})
      clip_finetune_fiq(**training_hyper_params)
  ```
  tức FashionIQ chỉ được đánh giá **bên trong vòng huấn luyện** (`clip_finetune_fiq` →
  `compute_fiq_val_metrics` qua `blip_validate.py:62-90`), **không** qua `validate_blip.py`.
  Hệ quả cho workbench: "eval PTHA từ checkpoint FashionIQ gốc" không có entrypoint sẵn; nó phải gọi
  thẳng `compute_fiq_val_metrics`/`fashioniq_val_retrieval`. Đây là lý do khớp với việc registry chỉ
  có `tuned_mtst_cirr.pt` (checkpoint CIRR) — không có checkpoint FashionIQ nào để eval. Ghi
  `REFUTED` cho bất kỳ claim nào nói PTHA có đường validate FashionIQ độc lập.

---

### 8. TME — cấu hình `tme`

- **Entrypoint:** `$T/src/precompute_evaluation.py:...` hàm chứa vòng ở `:38-...`; được gọi từ
  `$T/src/precompute.py`.
- **Gallery:** `$T/src/data.py:186-190`
  ```python
  for dress_type in self.dress_types:
      with open(self.path / 'image_splits' / f'split.{dress_type}.{split}.json') as f:
          self.image_type_names[dress_type] = json.load(f)
          self.image_names.extend(...)
  ```
  rồi `set_specific_dress_type` (`:190-195`) chọn `self.current_image_names =
  self.image_type_names[dress_type]` → **full category gallery, file order** (một category một lần).
  `precompute_evaluation.py:52-53` `index_names = dataset.current_image_names` giữ đúng thứ tự đó.
- **Reference:** **giữ** trên FashionIQ. Mask **chỉ** chạy cho CIRR:
  `$T/src/precompute_evaluation.py:85-87`
  ```python
  if category == 'CIRR':  # only for cirr, delete the reference image from current query.
      ...
      distances[...] = 10e10
  ```
- **Rank call:** `$T/src/precompute_evaluation.py:88`
  `sorted_indices = torch.argsort(distances, dim=-1).cpu()` với `distances = 1 - cos` (`:80-81`) →
  **`torch.argsort` tăng dần trên distance**. Không tie policy.
- **Image pipeline:** `$T/src/data.py:237` `image = self.preprocess(PIL.Image.open(image_path))`;
  preprocess dựng ở entrypoint (thường `targetpad_transform`). **Có cache precompute:**
  `features/FashionIQ/val_images.pt` và `val_name_to_idx.json` (artifact vendor, xem A.5) — nghĩa là
  run thật có thể đọc feature **đã tính trước** thay vì decode lại ảnh. Khi có cache, image pipeline
  của run là pipeline **lúc tạo cache**, không phải pipeline trong source hiện tại. Đây là rủi ro
  provenance thật và phải ghi theo digest cache.
- **Text:** `$T/src/precompute_evaluation.py:72-75`
  ```python
  captions = np.array(captions).flatten().tolist()
  captions = [f"{captions[i].strip('.?, ').capitalize()} and {captions[i+1].strip('.?, ')}" ...]
  captions = [text_preprocessor(item) for item in captions]
  ```
  → ghép **deterministic** `cap1 and cap2`, capitalize + strip dấu — **giống CLIP4Cir**, không dùng
  correction dict. `text_preprocessor` là LAVIS `txt_processors['eval']`.
- **Checkpoint:** ngoài phần eval; launcher load `weight/blip2_pretrained.pth`, `weight/eva_vit_g.pth`
  (artifact vendor). **UNVERIFIED strict/missing-key** cho model chính — cần đọc `src/precompute.py`
  và class model. Không được suy từ tên file.
- **Metric:** `$T/src/precompute_evaluation.py:90-107`: `labels` so tên, rồi
  `recall_at1/5/10/50 = 100*sum/len(labels)`, `acc = (r10+r50)/2`, `mean = mean([r10,r50])`.
  Mẫu số = `len(labels)` = số query đã chấm.
- **Refuted:** CSV cũ ghi `score_orientation=lower distance better (1 - cos)` — đúng; ghi
  `mask=none on the FIQ path` — đúng. Nhưng **bỏ sót precompute cache** (`val_images.pt`) là nguồn dữ
  liệu thật của feature gallery. Một run đọc cache không chứng minh được image pipeline hiện tại.

---

### 9. HINT — cấu hình `hint`

- **Entrypoint:** `$N/test.py:...` `def test(params, model, testset, category, txt_processors)`;
  FashionIQ ở `$N/test.py:118-170`.
- **Gallery:** `$N/datasets.py:131-166` `get_test_data`; `:154-159`
  ```python
  for i in test_queries:
      if i['source_img_id'] not in test_targets_id: test_targets_id.append(i['source_img_id'])
      if i['target_img_id']  not in test_targets_id: test_targets_id.append(i['target_img_id'])
  ```
  → **pair-union, first-seen ref-then-target**, 2628/3089/2902.
- **Reference:** scorer-time mask `$N/test.py:155-156`
  `sims[i, test_targets_id.index(t['source_img_id'])] = -10e10`, trước `argsort` `:158`.
- **Rank call:** `$N/test.py:158` `np.argsort(-sims[i, :])[:50]`.
- **Điểm gộp:** `$N/test.py:149-151` `np.matmul → squeeze → max(-1)`. Normalize **comment out**
  (`:140-143`).
- **Image:** `$N/data_utils.py` `targetpad_transform`; cùng họ ilearn.
- **Text:** `$N/datasets.py:75-82` `correct_text` + `concat_text` (kwarg `correction_dict`), caption
  `"{c0} and {c1}"` đã correct. Tokenizer `txt_processors["eval"]` (`$N/test.py:128`).
- **Checkpoint:** `$N/utils.py` cùng helper strict; eval không load.
- **Metric:** `$N/test.py:160-170`, `100*r/len(nn_result)`, k∈{1,10,50}.

---

### 10. ENCODER — cấu hình `encoder`

- **Entrypoint:** `$E/evaluate_model.py` với `--dataset fashioniq`; eval
  `$E/test.py:...` `def test(...)` → FashionIQ branch `$E/test.py:82-...`.
- **Loader / gallery:** `$E/datasets.py:64` `FashionIQ(path, transform=None, split='val-split')`.
  `--fashioniq_split` mặc định `val-split` (`$E/evaluate_model.py:15`), nên nhánh
  `$E/datasets.py:185-196` thắng: **pair-union first-seen ref-then-target**. `split.<cat>.val.json`
  chỉ được dùng để map ID→index (`$E/datasets.py:146-147`).
- **Reference:** scorer-time mask `$E/test.py:71`
  `sims[i, test_targets_id.index(t['source_img_id'])] = -10e10`, trước `argsort` `:71-...`.
- **Rank call:** `$E/test.py:71` `np.argsort(-sims[i, :])[:50]`.
- **Image pipeline — BẤT ĐỐI XỨNG, đúng như master file nói:**
  ```python
  $E/datasets.py:137  out['source_img_data'] = self.get_img(candidate, stage=0)   # TRAIN transform
  $E/datasets.py:146  out['source_img_data'] = ... stage=0                        # query
  $E/datasets.py:158  out['target_img_data'] = ... stage=1                        # gallery = VAL transform
  ```
  `get_img` (`:148-154`) chọn `self.transform[stage]`; transform list được dựng
  `[preprocess_train, preprocess_val]` ở `$E/evaluate_model.py:40,43`.
  → **query dùng `preprocess_train`** (có `RandomResizedCrop`/flip), **gallery dùng `preprocess_val`**.
  Đây là hành vi **của tác giả**, không phải lỗi workbench. Hệ quả: cùng một ảnh ở hai vai khác nhau
  cho ra hai tensor khác nhau, và mỗi lần chạy query sẽ random crop khác nhau nếu seed khác → không
  tất định.
- **Text:** `$E/datasets.py:117-126` `correct_text`/`concat_text`; tokenizer `open_clip` ViT-B-32
  (`$E/evaluate_model.py:40` `open_clip.create_model_and_transforms('ViT-B-32', ...)`), tức **không**
  dùng BERT của BLIP. `max_length` do tokenizer simple của open_clip (77) quản lý.
- **Checkpoint:** `$E/evaluate_model.py:32` `--ckpt_path`; load bằng `torch.load` + `load_state_dict`
  — **UNVERIFIED chính xác dòng strict** vì cần đọc `main` của `evaluate_model.py` (file này là
  `model_recovery`, bản vendor khác upstream, xem A.5). **`datasets1.py` chỉ là 23 byte**
  `from datasets import *` (đo bằng `wc -c` = 23, `od -c` xác nhận) — nghĩa là `datasets1` **không**
  phải loader riêng; nó re-export `datasets.py` của chính repo. Đây là shim tương thích của tác giả,
  không phải adaptation làm đổi protocol. Ghi chú này quan trọng: nó bác bỏ khả năng "workbench tự
  dựng dataset loader khác".
- **Metric:** `$E/test.py:73-83`, `100*r/len(nn_result)`, k∈{1,10,50}.

---

### 11. PAIR — cấu hình `pair`

- **Entrypoint:** `$R/test.py:6` `def test(params, model, testset, category)`; FashionIQ `:80-...`.
  **Không có launcher eval FashionIQ:** `$R/train.py:128-...` main chỉ train, không gọi `test.test`;
  `--dataset` mặc định là `shoes`-family (`$R/train.py:30-33`). Vì vậy "PAIR native eval" phải gọi
  `test.py` thủ công — **đây là đường không có sẵn trong repo**.
- **Gallery:** `$R/data/datasets.py:146-160`, cùng pattern
  `if i['source_img_id'] not in test_targets_id: …` → **pair-union first-seen ref-then-target**.
  `--fashioniq_split` mặc định `val-split` (`$R/train.py:29`).
- **Reference:** scorer-time mask `$R/test.py:...` (`sims[i, test_targets_id.index(...)] = -10e10`),
  trước `np.argsort(-sims[i,:])` ở `$R/test.py:71`. **Điểm gộp:** `$R/test.py` không thấy
  `sims.max(-1)`; normalize chạy (`np.linalg.norm`).
- **Image pipeline:** tương tự ENCODER — `transform[stage]`, query `stage=0` (train), gallery
  `stage=1` (val): `$R/data/datasets.py:138,140,158,165`, `get_img` `:106-113`.
- **Text:** `$R/data/datasets.py` `correct_text`/`concat_text`; tokenizer open_clip ViT-B-32.
- **Checkpoint:** `$R/train.py` load bằng `torch.load` + `load_state_dict`; **UNVERIFIED dòng**.
  Registry ghi `checkpoint_mapping_status: UNVERIFIED` cho cả `pair-B1.pt` và `pair-B2.pt`, khớp với
  việc `models/MHTransformer_rev.py` **chỉ có ở vendor**, không upstream (bảng A.5) — nghĩa là
  checkpoint cần class không có trong pin. Đây là blocker thật, không phải thiếu sót tài liệu.
- **Metric:** `$R/test.py:73-83`, `100*r/len(nn_result)`, k∈{1,10,50}.

---

### 12. CLVC-Net — cấu hình `clvc_net`

- **Entrypoint:** `$V/test.py:6` `def test(params, local_model, global_model, testset, dataname)`;
  FashionIQ nhánh `else` (`$V/test.py:80-...`).
- **Gallery:** `$V/datasets.py:230-246` `get_test_targets`:
  ```python
  if self.gallery_all:      # toàn bộ self.images
  else:                     # pair-union first-seen ref-then-target
  ```
  **Không caller nào trong repo pin truyền `gallery_all`** (`grep -rn gallery_all` chỉ ra
  `datasets.py`), nên **default `gallery_all=False`** (`:153`) → **pair-union**, 2628/3089/2902.
  Đây là điểm cần nói rõ: `gallery_all=True` tồn tại nhưng là đường chết.
- **Reference:** scorer-time mask `$V/test.py:...` `-10e10` trước `np.argsort(-sims[i,:])` ở
  `$V/test.py:138` (nhánh FashionIQ). **Điểm gộp:** `$V/test.py` normalize bằng
  `np.linalg.norm` rồi mới dot — **khác** nhóm BLIP (không max-pool).
- **Image pipeline:** `$V/train.py:76-95`
  - testset: `Resize(256) → CenterCrop(224) → ToTensor → Normalize(ImageNet mean/std)`.
  - trainset: `Resize(256) → RandomCrop(224) → RandomHorizontalFlip → ToTensor → Normalize(ImageNet)`.
  → eval dùng **val transform cho cả query và gallery**, normalize **ImageNet** (cùng họ DCNet, không
  phải CLIP). **Một transform cho cả hai**, không bất đối xứng.
- **Text:** `$V/datasets.py:173-175`
  ```python
  def concat_text(self, captions):
      text = "<BOS> {} <AND> {} <EOS>".format(captions[0], captions[1])
  ```
  → **thêm token literal `"<BOS>"`, `"<AND>"`, `"<EOS>"` vào chuỗi**, không phải special token của
  tokenizer. Tokenizer là `SimpleVocab` (xem `$V/text_model.py`), ascii, xoá dấu câu — **không** dùng
  correction dictionary và **không** lowercase ở eval.
- **Checkpoint — strict, đã đóng UNVERIFIED:** `$V/utils.py:125-141` `load_checkpoint`, thân là
  `:136-137`
  ```python
  checkpoint = torch.load(checkpoint)
  model.load_state_dict(checkpoint['state_dict'])
  ```
  **strict mặc định** (`strict` không bị override), key bọc `'state_dict'`. Không `map_location` ⇒
  checkpoint lưu từ CUDA không load được trên CPU (cùng lớp lỗi với DCNet, xem §5).
  `--restore_file` ở `$V/train.py:62-64`.
- **Metric:** `$V/test.py:...` `100*r/len(nn_result)`, k∈{1,10,50} (nhánh FashionIQ).

---

### 13. TG-CIR — cấu hình `tgcir`

- **Entrypoint:** `$G/train.py:200` `t = test.test(args, model, trainset, category)` cho ba category
  (`:199-203`); body ở `$G/test.py:6-82`.
- **Gallery:** `$G/datasets.py:127-162` `get_test_data`; `:149-156` first-seen ref-then-target →
  **pair-union**, 2628/3089/2902 (§B.6 đã chứng minh). `split.<cat>.val.json` chỉ dùng để
  `images.index(...)` (`:143,145`).
- **Reference:** scorer-time mask `$G/test.py:67`
  `sims[i, test_targets_id.index(t['source_img_id'])] = -10e10`, trước `argsort` `:70`.
- **Rank call:** `$G/test.py:70` `np.argsort(-sims[i, :])[:50]`. **Không** `sims.max(-1)`: TG-CIR dot
  thẳng sau khi tự normalize bằng `np.linalg.norm` (`$G/test.py:54-57`).
- **Image pipeline:** `$G/train.py:88` `clip, preprocess = clip.load(args.img_encoder, ...)`; dataset
  dùng `self.transform` cho cả `get_img` và `get_test_img` (`$G/datasets.py:113-126`). Ảnh đọc từ
  `resized_image/<cat>/<id>.jpg` (`:114,120,139`). **Một transform**, OpenAI CLIP preprocess của
  encoder đã chọn.
- **Text:** `$G/datasets.py:78-80`
  `concat_text = "{correct_text(c0)} and {correct_text(c1)}"`; `correct_text` (`:70-76`) lowercase,
  punctuation→space, correction dict **local trong repo**:
  `$G/correction_dict/correction_dict_{dress,shirt,toptee,shoes}.json` — tức TG-CIR mang dict riêng,
  không đọc từ `data/FashionIQ/captions/`. Workbench dùng `data/FashionIQ/captions/correction_dict_*`
  (`workbench/replay/tgcir.py:26-27`), nên phải so digest hai bộ này; **chưa kiểm trong lượt này**
  (`UNVERIFIED`).
  Tokenizer: `clip.tokenize` qua model wrapper, max 77.
- **Checkpoint:** `$G/train.py:74` khai `--restore_file` nhưng **`grep -rn restore_file` chỉ ra đúng
  một dòng khai báo** — tham số được parse và **không dùng ở đâu**. Xác nhận đúng như master file §M-05.
  Vì vậy "native eval từ checkpoint" cho TG-CIR **không có trong source pin**.
- **Metric:** `$G/test.py:73-80`, `100*r/len(nn_result)`, k∈{1,10,50}.

---

### 14. LIMN — cấu hình `limn`

- **Entrypoint:** `$M/LIMN/train.py` (launcher) gọi `test.test(...)`; body `$M/LIMN/test.py:...`.
- **Gallery:** `$M/LIMN/datasets.py:136-160` first-seen ref-then-target → **pair-union**.
- **Reference:** scorer-time mask `$M/LIMN/test.py:81-83`
  ```python
  if name != 'birds':
      for i, t in enumerate(test_queries):
          sims[i, test_targets_id.index(t['source_img_id'])] = -10e10
  ```
  Guard `!= 'birds'` là di sản dataset LaSCo/birds; với FashionIQ luôn chạy.
- **Rank call:** `$M/LIMN/test.py:85` `np.argsort(-sims[i, :])[:50]`.
- **Image pipeline:** `$M/LIMN/train.py:78-89`
  `[preprocess_train, preprocess_val] = open_clip.create_model_and_transforms('ViT-L-14', ...)`;
  dataset `get_test_img` dùng `self.transform[1]` = **preprocess_val** (`$M/LIMN/datasets.py:125`) cho
  **query và gallery**. `get_img` (train) dùng `transform[0]`. Eval **không** dùng train transform —
  tức LIMN sạch hơn ENCODER/PAIR ở điểm này.
- **Text:** `$M/LIMN/datasets.py` `correct_text`+`concat_text`; tokenizer là open_clip ViT-L-14
  tokenizer (`$M/LIMN/train.py:78`), không phải BERT/BLIP.
- **Checkpoint:** `$M/LIMN/train.py` `--restore_file`; **UNVERIFIED dòng strict**.
- **Metric:** `$M/LIMN/test.py:88-98`, `100*r/len(nn_result)`, k∈{1,10,50}.

---

### 15. CSMCIR — cấu hình `csmcir` (tạm gác)

- **Entrypoint:** `$S/src/validate_blip_csmcir.py:211-230` `fashioniq_val_retrieval` →
  `compute_fiq_val_metrics` (`:24-57`).
- **Gallery:** `$S/src/validate_blip_csmcir.py:225-226` `FashionIQDataset('val', [dress_type],
  'classic', preprocess)`; `classic` đọc `image_splits/split.<cat>.val.json` →
  **full category gallery, file order**, **không** lọc tồn tại.
- **Reference:** **giữ** trên FashionIQ; mask chỉ ở nhánh CIRR. Cùng khuôn với CLIP4Cir/PTHA/SPRC/TME.
- **Rank call:** `torch.argsort(distances, dim=-1)` (`:45`), `distances = 1 - pred_sim` (`:44`) →
  tăng dần trên distance.
- **Image pipeline:** `targetpad_transform` dựng ở entrypoint; **UNVERIFIED dim**.
- **Text:** caption thô + `txt_processors` của LAVIS; **UNVERIFIED** correction dict.
- **Checkpoint:** registry `VERIFIED_METADATA` với sha256 `c8812071de06…`; 6 digest local/global ở
  registry khớp source? **UNVERIFIED trong lượt này** (không mở bundle). Master file §M-04 nói CLI
  vẫn từ chối mapping vô điều kiện — không kiểm lại được nếu không chạy CLI, và tôi không chạy.
- **Metric:** `:49-57`, `assert` đúng một positive, `100*sum/len`, k∈{10,50}.
- **Blocker:** `COT_ours2/fashioniq/*_cot_val.json` là **vendor-only** (bảng A.5) — không có trong
  clone. Nếu eval cần CoT caption, source pin **không** chạy được; đây là blocker đầu vào, không phải
  lỗi code.

---

### 16. SPRC — cấu hình `sprc` (tạm gác)

- **Entrypoint:** `$X/src/validate_blip.py:210-229` (cùng khuôn PTHA/CSMCIR) →
  `compute_fiq_val_metrics` `:24-57`; biến thể rerank ở `$X/src/validate_blip_rerank.py`.
- **Gallery:** `classic` trên `split.<cat>.val.json` → full category, file order.
- **Reference:** giữ trên FashionIQ (mask CIRR-only).
- **Rank call:** `torch.argsort(distances, dim=-1)` (`validate_blip.py:45`; rerank thêm
  `torch.argsort` thứ hai ở `validate_blip_rerank.py:67` cho top-100).
- **Image/text/checkpoint/metric:** cùng khuôn BLIP như PTHA; **UNVERIFIED** chi tiết transform.
- **Blocker:** registry `sprc_fiq.pt` có `checkpoint_mapping_status: UNVERIFIED`, chỉ
  `sprc_fiq_vitl.pt` là `VERIFIED_METADATA`. Master file §M-10 nói tạm gác vì checkpoint — khớp.

---

## D. Các claim trong ma trận cũ mà source bác bỏ

Mỗi dòng: claim cũ → chỗ nó sai → `file:line` bác bỏ. Ma trận cũ =
`workbench/docs/reviews/20261110-e2e-repair/PROTOCOL_MATRIX.csv`; file tham chiếu thứ hai
`workbench/docs/MODEL_STATUS_SPEC.md`.

### D.1 Quy tắc `np.argsort(-sims)` phủ định toàn bộ

CSV cũ đặt `tie_policy=np.argsort(-sims) default (quicksort, not stable)` cho **cả 26 dòng** và
`rank_definition=1 + #{scores greater than target}` cho cả 26. Source bác bỏ ở 5/16 upstream:

1. **CLIP4Cir** — `src/validate.py:44-46`
   `distances = 1 - predicted_features @ index_features.T; torch.argsort(distances, dim=-1)`.
   Sắp **tăng dần trên distance**, `torch` không `numpy`. Comment trong file ghi đúng "Compute the
   distances and sort the results" (`:43`).
2. **CLIP4Cir Combiner** — cùng `validate.py`, cùng dòng.
3. **MTST+PTHA** — `MTST_PTHA/src/validate_blip.py:44-45`, y hệt CLIP4Cir.
4. **TME** — `src/precompute_evaluation.py:80-88`
   `distances = 1 - distances; sorted_indices = torch.argsort(distances, dim=-1)`.
5. **CSMCIR** — `src/validate_blip_csmcir.py:44-45`; **SPRC** — `src/validate_blip.py:44-45`.
6. **DCNet** — `trainer/trainer_joint.py:228,232,240`
   `score.topk(dim=0, k=50, largest=self.largest, sorted=True)`, `dim=0` chứ không `dim=-1`, gọi **ba
   lần** cho ba ma trận, và nhánh `comb` cộng **log-softmax** chứ không cộng score thô (`:239`).

Nghĩa là quy tắc chung `np.argsort(-sims)` sai cho **10/17 cấu hình** (5 cấu hình torch.argsort +
DCNet + cùng repo CLIP4Cir hai config).

### D.2 Claim "score = cosine / inner product" bỏ qua max-pool

CSV cũ ghi `score_orientation=higher cosine better` cho Air-Know, ConeSep, HABIT, INTENT, HINT. Source:

- Air-Know `test_BLIP2.py:144` `sims = sims.max(-1)`
- ConeSep `test.py:63-64` `sims = sims.squeeze(); sims = sims.max(-1)`
- HABIT `test.py:136-138` cùng dạng
- INTENT `test.py:135-137` cùng dạng
- HINT `test.py:149-151` cùng dạng

Điểm cuối là **max theo trục token**, không phải dot của hai vector. Đồng thời normalize bị comment
out ở ConeSep (`test.py:124-127`), HABIT (`:133-136`), INTENT (`:132-135`), HINT (`:140-143`) — nên
"cosine" còn sai cả về chuẩn hoá.

### D.3 Claim "ENCODER dùng val transform cho tất cả ảnh"

CSV cũ ghi `image_pipeline=preprocess_train (RandomResizedCrop) for ALL images incl. gallery` cho
ENCODER. Source ngược lại ở phần gallery:

- `ENCODER.JZt48G/datasets.py:137,146` query `get_img(..., stage=0)` = `preprocess_train`
- `ENCODER.JZt48G/datasets.py:158,165` gallery `get_img(..., stage=1)` = `preprocess_val`

Vậy bất đối xứng là **query=train, gallery=val**, không phải "train cho tất cả". Cùng lỗi ở PAIR
(`PAIR.jqTJq4/data/datasets.py:138,140` vs `:158,165`).

### D.4 Claim PAIR/ENCODER "cosine similarity, higher better" bỏ normalization

`ENCODER.JZt48G/test.py:56-58` và `PAIR.jqTJq4/test.py` normalize `all_queries`/`all_imgs` bằng
`np.linalg.norm` **trước** khi dot — ngược với nhóm Air-Know/ConeSep/HABIT/INTENT/HINT nơi normalize
bị comment. Hai nhóm không cùng thang điểm; so điểm thô giữa chúng là vô nghĩa.

### D.5 Claim CLIP4Cir FullFT và Combiner là một

CSV cũ tách hai dòng nhưng cùng `upstream_commit` và cùng mô tả `mask=none`, `rank_definition` —
đúng phần protocol, nhưng ghi `label_verdict=MISLABELED` cho `clip4cir_rn50x4_fullft` mà không nêu
khác biệt checkpoint. Source: hai config khác **checkpoint key** (`validate.py:286` `["CLIP"]` cho
FullFT; `:307` `["Combiner"]` cho Combiner) và khác **hàm fusion** (`element_wise_sum` vs
`combiner.combine_features`, `:297-307`). Gộp là sai; tách là đúng, và phải nêu checkpoint khác nhau.

### D.6 Claim PTHA có đường validate FashionIQ độc lập

CSV cũ ghi PTHA `protocol_label_validated=fashioniq_original_split`, `label_verdict=CONFIRMED`, ngụ ý
có entrypoint validate. Source: `MTST_PTHA/src/validate_blip.py:497-511` `main()` chỉ dispatch `CIRR`;
FashionIQ chỉ chạy **bên trong** `clip_finetune_fiq` (`MTST_PTHA/src/blip_fine_tune_2.py:25,400,422-427`).
Không có đường validate-only cho FashionIQ trong pin.

### D.7 Claim DCNet gallery "pre-resized to 256"

CSV cũ ghi `image_pipeline=images pre-resized to 256 then CenterCrop(224)`. Source chỉ đọc
`resized_images/` (`data_loader/CE_dataset.py:19`) và áp `CenterCrop(224)` (`data_loaders.py:67`);
**không** có dòng nào ghi 256. Kích thước gốc của artifact là dữ kiện của đĩa, không phải của source ⇒
phải ghi `UNVERIFIED` cho 256.

### D.8 Claim INTENT có load checkpoint strict

CSV cũ ghi `mask=sims[i, ref] = -10e10 at test.py:150` (đúng) nhưng toàn dòng ngụ ý có đường eval từ
checkpoint. Source: `INTENT.l6VkTd/train_INTENT.py:380-384` chỉ `torch.save`; **không** có
`torch.load`/`load_state_dict` nào trong entrypoint. Helper strict ở `utils.py:139-140` tồn tại nhưng
không được gọi. Đường eval-từ-checkpoint của INTENT là **UNVERIFIED trong source**.

### D.9 Claim gallery "pair-union (ordered)" là đủ để tái lập

`MODEL_STATUS_SPEC.md:207` ghi `fashioniq_val_split` = `pair-union (ordered) | excluded`. B.5 chứng
minh hai cách ordered khác nhau ở 99.9% vị trí (dress 2625/2628 ID đổi chỗ, lệch trung bình 450 bậc).
Không ghi rõ "first-seen ref-then-target" thì không tái lập được thứ tự hoà điểm.

### D.10 Claim TG-CIR gallery là full split (hồ sơ cũ)

Hồ sơ cũ gán TG-CIR `full split`. Source `TGCIR/datasets.py:149-156` dựng union từ `test_queries`
theo first-seen ref-then-target ⇒ **pair-union 2628/3089/2902**. Master file đúng khi nói log khớp
pair-union.

### D.11 Claim CLVC-Net có thể chạy `gallery_all=True`

CSV cũ ghi `gallery_kind=pair union (gallery_all=False)` — **đúng**. Nhưng không nêu rằng nhánh
`gallery_all=True` là **đường chết**: `grep -rn gallery_all` trên pin chỉ trả về `datasets.py:153,163,234`,
không caller nào truyền `True`. Giá trị `False` là default của signature nên nó luôn thắng.

### D.12 Claim cảnh báo URL sai cho 7 repo

Master file cảnh báo ma trận cũ từng ghi sai URL cho INTENT, HINT, PAIR, CLVC-Net, DCNet, TG-CIR,
LIMN. Trên HEAD: **0/16 redirect, 0/16 sai, 0/16 là fork** (A.3). Cảnh báo đó không còn tái hiện; chỉ
còn khác biệt `source_dir` ↔ tail URL (11/16) là quy ước đặt tên, không phải lỗi.

---

## E. Tổng hợp trạng thái kiểm chứng

| Hạng mục | Kết quả |
|---|---|
| SHA pin giải mã được | **16/16** |
| HEAD clone = registry SHA | **16/16** |
| URL redirect | **0/16** |
| URL sai / fork | **0/16** / **0/16** |
| Vendor khớp clone (ngoài `provenance.json`) | **8/16** byte-identical; 8/16 có patch đã phân loại |
| Drift không giải thích được | **0** |
| Query val (dress/shirt/toptee) | **2017 / 2038 / 1961 = 6016** ✔ |
| Full gallery | **3817 / 6346 / 5373** ✔ |
| Pair-union | **2628 / 3089 / 2902** ✔ |
| Upstream dùng `np.argsort(-sims)` | 10 cấu hình |
| Upstream dùng `torch.argsort` (tăng dần trên distance) | 6 cấu hình (CLIP4Cir ×2, PTHA, TME, CSMCIR, SPRC) |
| Upstream dùng `torch.topk` | 1 cấu hình (DCNet ×3 nhánh) |
| Claim cũ bị bác bỏ | **12 nhóm** (D.1–D.12) |

---

## F. Đóng ba mục `UNVERIFIED` (lượt hai)

### F.1 CSMCIR — sáu digest registry so với bundle thật

Cách làm: đọc `approved_source_patches` của `csmcir` trong `workbench/registry/models.yaml` (hai phần
tử, mỗi phần tử ba digest), rồi tự tính `hashlib.sha256` trên: file patch
(`workbench/patches/…`), file **base** trong clone pin (`/tmp/aud/clone/CSMCIR.sJVolI/<path>`), và file
**derived** trong vendor (`workbench/third_party/CSMCIR/<path>`). Kịch bản: cell Python ghi ra
`/tmp/aud/csmcir_verify.txt`.

| Path | Loại digest | Registry | Đo được | Kết quả |
|---|---|---|---|---|
| `src/lavis/models/__init__.py` | `patch_sha256` | `3f2bac48fe1a1775…` | `3f2bac48fe1a1775…` | **MATCH** |
| `src/lavis/models/__init__.py` | `base_sha256` | `9d8c914b551a7de6…` | `9d8c914b551a7de6…` | **MATCH** |
| `src/lavis/models/__init__.py` | `derived_sha256` | `0ec134431697af6f…` | `0ec134431697af6f…` | **MATCH** |
| `src/validate_blip_csmcir.py` | `patch_sha256` | `0fcff28bf1f9abd3…` | `0fcff28bf1f9abd3…` | **MATCH** |
| `src/validate_blip_csmcir.py` | `base_sha256` | `aa7a4395cdb700fd…` | `aa7a4395cdb700fd…` | **MATCH** |
| `src/validate_blip_csmcir.py` | `derived_sha256` | `a057e8f3ad7097ad…` | `a057e8f3ad7097ad…` | **MATCH** |

**6/6 MATCH.** Nghĩa là `VERIFIED_METADATA` của CSMCIR **không** phải claim metadata sai. Quan trọng
hơn: digest `base_sha256` khớp với file trong **clone pin độc lập**, chứng minh patch được áp lên đúng
bản source đã pin, không phải lên một bản khác rồi ghi lại digest của chính nó. Và `derived_sha256`
khớp file trong `third_party`, chứng minh vendor **đúng là** base + patch.

Vì vậy vấn đề của CSMCIR **không** thuộc lớp "registry khai VERIFIED_METADATA nhưng digest sai". Vấn
đề còn lại của nó là blocker đầu vào đã ghi ở §15: `COT_ours2/fashioniq/*_cot_val.json` là vendor-only,
không có trong clone pin — tức **artifact**, không phải metadata. Trạng thái "tạm gác" giữ nguyên, nhưng
lý do phải là blocker artifact, **không** phải nghi vấn digest.

### F.2 TG-CIR `correction_dict` so với `data/FashionIQ/captions/`

Ba bản: clone pin (`/tmp/aud/clone/TGCIR.LwsZVm/correction_dict/`), vendor
(`workbench/third_party/TGCIR/correction_dict/`), dataset
(`data/FashionIQ/captions/correction_dict_<cat>.json`). So bytes và so nội dung JSON.

| Category | clone | vendor | dataset | byte-identical? |
|---|---|---|---|---|
| dress | `9e4a59877098fe26` | `9e4a59877098fe26` | `9e4a59877098fe26` | **có** |
| shirt | `ca29e29e23c0a198` | `ca29e29e23c0a198` | `ca29e29e23c0a198` | **có** |
| toptee | `3dfbf15fa6d3a30d` | `3dfbf15fa6d3a30d` | `3dfbf15fa6d3a30d` | **có** |

So thêm nội dung: số entry và giá trị trùng khít — dress 363/363 (0 chỉ-TG-CIR, 0 chỉ-dataset, 0 giá
trị khác), shirt 360/360, toptee 391/391.

**Kết luận:** correction dictionary của TG-CIR **byte-identical** với bộ trong `data/FashionIQ/captions/`
ở cả ba category. TG-CIR **không** dùng bộ text khác; caption của nó là cùng một chuỗi sau
`correct_text` như các repo đọc từ `data/FashionIQ`. Lo ngại "TG-CIR có dict riêng nên caption khác" bị
bác bỏ. Hệ quả cho workbench: việc `workbench/replay/tgcir.py:26-27` trỏ vào
`data/FashionIQ/captions/correction_dict_*` là **đúng nguồn**, không phải thay thế tạm.

### F.3 Strict-load cho ENCODER, LIMN, PAIR, TG-CIR

| Repo | Vị trí | Kiểu load | `strict` | Missing key được xử lý thế nào |
|---|---|---|---|---|
| ENCODER | `evaluate_model.py:60,63` | `torch.load(args.ckpt_path)` → `load_state_dict(model, strict=False)` | **False** | **Không in, không kiểm** — `strict=False` trả `NamedTuple` nhưng code **bỏ qua giá trị trả về** |
| LIMN | `LIMN/utils.py:136-137` | `torch.load(checkpoint)` → `load_state_dict(checkpoint['state_dict'])` | **True** (mặc định) | Nổ `RuntimeError` nếu lệch key |
| PAIR | `utils/utils.py:136-137` | y hệt LIMN | **True** (mặc định) | Nổ `RuntimeError` |
| TG-CIR | `utils.py:136-137` | y hệt LIMN | **True** (mặc định) | Nổ `RuntimeError` |

Chi tiết cần ghi:

- **ENCODER là trường hợp tệ nhất trong bốn**, và tệ hơn PTHA: PTHA còn `print("Missing keys {}")`
  (`MTST_PTHA/src/validate_blip.py:521-522`), còn ENCODER `evaluate_model.py:63` là
  `Encoder_model.load_state_dict(model, strict=False)` — **nuốt im lặng** danh sách missing/unexpected.
  Một checkpoint thiếu layer vẫn "load thành công" và cho số liệu không cảnh báo. Đây là rủi ro
  provenance thật cho mọi run ENCODER.
- Ba repo LIMN/PAIR/TG-CIR dùng **cùng một** helper `load_checkpoint` (`utils.py:125-143`) byte-identical
  giữa ba repo, thân là `torch.load` + `load_state_dict(checkpoint['state_dict'])` **không** truyền
  `strict` ⇒ strict mặc định `True`. Tuy nhiên **cả ba đều không gọi helper đó từ entrypoint**:
  `grep -rn load_checkpoint` chỉ trả về đúng dòng định nghĩa. `--restore_file` được **khai báo**
  (`LIMN/train.py:67`, `TGCIR/train.py:74`) nhưng không được dùng — cùng dạng với TG-CIR.
  Không có `torch.load`/`load_state_dict` nào khác trong các entrypoint đó.
- Hệ quả: "eval từ checkpoint" cho LIMN/PAIR/TG-CIR **không có trong source pin**; workbench phải tự
  viết loader. Khi tự viết, phải chủ động chọn `strict=True` và **assert** kết quả, vì không có tiền lệ
  nào trong ba repo để dựa vào.
- PAIR có thêm đường `torch.load(os.path.join(model_dir, i))` trong `cirr_test_submission.py:103` —
  nhánh CIRR submission, **không** áp cho FashionIQ, và đó là pickle toàn model chứ không phải
  state_dict.

---

## G. Chỗ master file đúng / chỗ master file hẹp hơn source

### G.1 Master file đúng, đã kiểm lại bằng source

- **"DCNet có `topk` theo nhánh"** — đúng. `trainer/trainer_joint.py:228,232,240`, ba nhánh
  (comp, corr, comb), cộng thêm log-softmax ở nhánh comb (`:239`).
- **"CLIP4Cir/PTHA/TME dùng `torch.argsort`"** — đúng cả ba:
  `CLIP4Cir/src/validate.py:44-46`, `MTST_PTHA/src/validate_blip.py:44-45`,
  `TME/src/precompute_evaluation.py:80-88`. Cùng dạng còn có CSMCIR và SPRC.
- **"Không áp `np.argsort(-sims)` hoặc quy tắc tie chung từ ma trận cũ"** — đúng, và đây là cảnh báo
  có giá trị nhất trong cả file.
- **"CLIP4Cir có thể bỏ sample lỗi ở collate"** — đúng: `src/utils.py:81-88` lọc `None` trước
  `default_collate`, nên mẫu số metric co lại theo.
- **"PTHA lọc ảnh PNG gallery thiếu"** — đúng: `MTST_PTHA/src/data_utils.py:168-172` dùng
  `Path.exists()` trên `.png` rồi `remove()`. Trên host này lọc là no-op (0 ảnh thiếu), nhưng code
  **có** chạy.
- **"ENCODER: train transform cho query, val cho gallery"** — đúng, `datasets.py:137,146` vs
  `:158,165`.
- **"TG-CIR log khớp pair-union, hồ sơ cũ gán nhầm full split"** — đúng; hồ sơ cũ sai (§B.6).
- **"CLIP4Cir FullFT giữ ref"** — đúng; không có mask nào trong `compute_fiq_val_metrics`.
- **"TG-CIR `--restore_file` được parse nhưng không dùng"** — đúng: chỉ có dòng khai báo
  `train.py:74`, không nơi nào đọc.
- **"PAIR checkpoint đòi `model_vssp`/`MHTransformer_dis`"** — đúng ở mức cấu trúc: `MHTransformer_rev.py`
  chỉ có trong vendor, không upstream. Registry cũng đánh `UNVERIFIED` cho cả hai checkpoint.

### G.2 Master file hẹp hơn source (thiếu, không sai)

- **Cảnh báo "7 URL từng sai" không tái hiện.** Trên HEAD này: **0/16 URL sai, 0/16 redirect,
  0/16 fork** (A.3). Bảy repo được nêu (INTENT, HINT, PAIR, CLVC-Net, DCNet, TG-CIR, LIMN) đều 200 tại
  đúng URL, SHA pin giải mã được, không redirect. Người đọc không nên đi tìm một lỗi URL không còn tồn
  tại; cái còn lại chỉ là tên thư mục `source_dir` khác tail URL ở 11/16, là quy ước nội bộ.
- **"Order changes ranks" là đúng nhưng quá yếu.** Không nói ordered *theo cái gì*. Đo được: hai cách
  ordered lệch nhau ở 2625/2628, 3086/3089, 2899/2902 ID, trung bình 450–520 bậc (§B.5).
- **"DCNet ảnh resize sẵn"** — master file nói "ảnh resize sẵn", source chỉ đọc `resized_images/`; nói
  "256" là suy đoán, không có trong source (§D.7).
- **Không nêu `sims.max(-1)`** ở 5 upstream BLIP. Đây là phép gộp điểm, không phải chi tiết phụ; bỏ nó
  thì score không tái lập được (§D.2).
- **Không nêu ENCODER `strict=False` nuốt missing keys** (`evaluate_model.py:63`), trong khi đây là
  đường dễ cho ra số liệu sai nhất trong 16 repo.
- **Không nêu PTHA thiếu entrypoint validate FashionIQ** (§D.6).

Ba mục từng ghi `UNVERIFIED` đã đóng ở §F (CSMCIR 6/6 digest MATCH; TG-CIR correction dict
byte-identical với dataset ở cả ba category; strict-load đã chốt cho ENCODER/LIMN/PAIR/TG-CIR).

Còn `UNVERIFIED` sau lượt hai **không còn mục nào chặn kết luận protocol**. Những gì vẫn chưa kiểm
được, và lý do:

| Mục | Lý do chưa đóng |
|---|---|
| Revision gốc của `data/FashionIQ` so với `XiaoxiaoGuo/fashion-iq` | Không clone repo dataset trong phạm vi này; không có digest tham chiếu. Định danh bằng sha256 của chính các file đã ghi (B.1). |
| Kích thước gốc ảnh trong `resized_images/` của DCNet | Là dữ kiện của artifact trên đĩa, không phải của source; source chỉ đọc thư mục. |
| `input_dim`/`target_ratio` đã dùng trong run lịch sử (khác mặc định source) | Cần log run, không cần source. Source nói 224/1.25 cho PTHA, 288 cho CLIP4Cir RN50x4. |
| Số query thực sự được chấm trong các run đã lưu | Cần per-query artifact; việc này thuộc nửa workbench, không thuộc upstream. |
