# Hướng dẫn một lệnh duy nhất cho CIR Failure Analysis Workbench

> **Lưu ý:** Hướng dẫn Docker/`./pc.sh` dưới đây là tài liệu lịch sử, không còn là quy trình hiện hành. Không làm theo lệnh hoặc trạng thái model cũ. Dùng [AUTOPILOT_GUIDE.md](AUTOPILOT_GUIDE.md) làm quy trình chính, [FINAL_PC_READINESS.md](FINAL_PC_READINESS.md) cho đường setup/run/resume duy nhất, và [MODEL_STATUS_SPEC.md](MODEL_STATUS_SPEC.md) cho blocker hiện tại. TME weights: `python -m workbench.scripts.download_auxiliary_assets --model tme`; cache feature validation cần CUDA preparation riêng với `--allow-gpu-preparation` và ước tính khoảng 10.47 GiB CACHE/STORAGE, không phải bảo đảm VRAM; mức phù hợp VRAM phụ thuộc GPU và batch.

Tài liệu hướng dẫn dành cho nhà nghiên cứu Composed Image Retrieval (CIR). Toàn bộ quy trình từ chuẩn bị môi trường, kiểm chứng phần cứng, nạp dữ liệu đến đánh giá mô hình và trực quan hóa kết quả đều được tự động hóa thông qua một lệnh duy nhất. Bạn không cần can thiệp hay gỡ lỗi cấu hình Docker nội bộ.

---

## 1. Tổng quan

CIR Failure Analysis Workbench là hệ thống phục vụ tái lập khoa học và phân tích lỗi cho các mô hình Composed Image Retrieval trên tập dữ liệu benchmark FashionIQ.

- **Kho mã nguồn**: `https://github.com/Le-Minh-Nhut/cir`
- **Nhánh nghiên cứu**: `research/cir-failure-workbench`
- **Điểm vào duy nhất (Single Entrypoint)**: Tập lệnh `./pc.sh` nằm tại thư mục gốc của repository.

Dưới nền tảng, `./pc.sh` đóng vai trò điều phối trung tâm. Script tự động gọi các công cụ quản trị cấp thấp hơn gồm `workbench/docker/run.sh` (`up|doctor|prepare <model>|pilot <model>|status|logs|stop`) và pipeline bên trong container `workbench/scripts/pipeline.py`. Bạn chỉ cần tương tác trực tiếp với giao diện của `./pc.sh`.

---

## 2. Yêu cầu phần cứng tối thiểu

Hệ thống được thiết kế và kiểm chứng tối ưu cho cấu hình phần cứng:

- **GPU mục tiêu**: NVIDIA GeForce RTX 5070 Ti (kiến trúc Blackwell, Compute Capability `sm_120`, 16 GB VRAM).
- **Hệ điều hành**: Linux x86_64 (khuyến nghị Ubuntu 22.04 LTS hoặc Ubuntu 24.04 LTS).
- **Phần mềm máy chủ (Host)**:
  - NVIDIA Driver hỗ trợ CUDA 12.8 trở lên.
  - Docker Engine (khuyến nghị phiên bản 24.0 trở lên) và plugin Docker Compose v2 (`docker compose`).
  - `nvidia-container-toolkit` đã được cấu hình cho Docker runtime.
- **Bộ nhớ RAM**: Tối thiểu 16 GB (khuyến nghị 32 GB).
- **Dung lượng đĩa trống**: Tối thiểu 50 GB dung lượng khả dụng tại phân vùng làm việc để lưu trữ Docker image, dataset FashionIQ, checkpoints và artifacts.

---

## 3. Dữ liệu cần có sẵn

Bộ dữ liệu FashionIQ cần được chuẩn bị sẵn trên máy host trước khi chạy đánh giá. Đường dẫn mặc định được cấu hình là `HOST_FASHIONIQ_ROOT=../../data/FashionIQ` (tính tương đối từ thư mục `workbench/docker/`).

Cấu trúc thư mục dữ liệu yêu cầu:

```text
data/FashionIQ/
├── images/
│   ├── dress/
│   ├── shirt/
│   └── toptee/
├── image_splits/
│   ├── split.dress.val.json
│   ├── split.shirt.val.json
│   └── split.toptee.val.json
└── captions/
    ├── cap.dress.val.json
    ├── cap.shirt.val.json
    └── cap.toptee.val.json
```

**Bảo vệ dữ liệu chuẩn (DATA-02)**:
Hai thư mục dữ liệu gốc `images/` và `image_splits/` được mount vào container ở chế độ chỉ đọc (**ro** - Read-Only) ở cấp kernel:
- `/data/FashionIQ/images` (**ro**)
- `/data/FashionIQ/image_splits` (**ro**)

Mọi tiến trình trong container không thể can thiệp hay sửa đổi ảnh gốc. Các ảnh đã chuẩn hóa kích thước (resized) và dữ liệu dẫn xuất (`captions/correction_dict_*.json`) được lưu riêng tại `/data/FashionIQ/resized_image/` và thư mục `captions/`.

---

## 4. Chuẩn bị lần đầu

### Bước 1: Lấy mã nguồn

```bash
git clone -b research/cir-failure-workbench https://github.com/Le-Minh-Nhut/cir
cd cir
```

### Bước 2: Cấu hình môi trường

Tập tin cấu hình chính đặt tại `workbench/docker/.env`. Nếu tập tin này chưa tồn tại, `./pc.sh` sẽ tự động khởi tạo từ bản mẫu `workbench/docker/.env.example`.

Các biến cấu hình quan trọng trong `.env`:
- `HOST_FASHIONIQ_ROOT`: Đường dẫn tới dataset FashionIQ trên host (mặc định `../../data/FashionIQ`).
- `HOST_WORKBENCH_THIRD_PARTY_ROOT`: Thư mục lưu mã nguồn mô hình upstream (mặc định `../../workbench/third_party`).
- `HOST_WORKBENCH_CHECKPOINT_ROOT`: Thư mục lưu trọng số mô hình (mặc định `../../workbench/artifacts/checkpoints`).
- `HOST_WORKBENCH_ARTIFACTS_ROOT`: Thư mục lưu artifacts và reports (mặc định `../../workbench/artifacts`).
- `HOST_WORKBENCH_RESULTS_ROOT`: Thư mục lưu kết quả đánh giá (mặc định `../../workbench/artifacts/results`).
- `WORKBENCH_BACKEND_PORT`: Cổng dịch vụ Backend API trên host (mặc định `8000`).
- `WORKBENCH_FRONTEND_PORT`: Cổng giao diện Web UI trên host (mặc định `5173`).
- `WORKBENCH_UID` / `WORKBENCH_GID`: Định danh user và group trên host, bảo đảm các file kết quả ghi ra thuộc quyền sở hữu của bạn, không bị phân quyền root.
- `WORKBENCH_ENABLE_GPU`: Thiết lập `1` để bật GPU.
- `WORKBENCH_RUNTIME_ADAPTATION`: Thiết lập `sm_120` để kích hoạt runtime tương thích kiến trúc Blackwell cho các mô hình hỗ trợ.

---

## 5. MỘT LỆNH DUY NHẤT

Tại thư mục gốc của repository, bạn chỉ cần thực thi một lệnh duy nhất:

```bash
./pc.sh
```

Đây là lệnh hoàn chỉnh cho toàn bộ quy trình nghiên cứu. Script sẽ tự động:
1. Kiểm tra toàn bộ điều kiện tiên quyết phần cứng và phần mềm trên host.
2. Thiết lập cấu hình mặc định (mô hình `limn`, `WORKBENCH_ENABLE_GPU=1`, `WORKBENCH_RUNTIME_ADAPTATION=sm_120`).
3. Khởi chạy Docker stack trong chế độ nền và xác nhận dịch vụ sẵn sàng.
4. Tải và kiểm tra mã băm của toàn bộ trọng số checkpoint chính thức.
5. Kiểm chứng năng lực tính toán thật của GPU bên trong runtime mô hình.
6. Xếp hàng và thực thi đánh giá mô hình LIMN trên tập FashionIQ val split.
7. Đối chiếu kết quả với sổ cái bằng chứng và in báo cáo khoa học trực tiếp ra terminal.

Bạn không cần cài đặt Python virtualenv hay bất kỳ package pip nào trên máy host.

---

## 6. Các giai đoạn thực thi

Khi `./pc.sh` chạy, hệ thống sẽ tuần tự đi qua các giai đoạn nghiêm ngặt sau:

1. **Preflight Check**:
   - Thăm dò hệ điều hành Linux, driver NVIDIA và tên card qua `nvidia-smi`.
   - Kiểm tra kết nối tới Docker daemon, plugin `docker compose`, và runtime `nvidia` trong `docker info`.
   - Xác nhận dung lượng RAM và đĩa trống đạt ngưỡng an toàn.
   - Kiểm tra các thư mục host và quyền sở hữu file.
2. **Configuration Resolution**:
   - Phân tích cú pháp nghiêm ngặt tập tin `workbench/docker/.env`.
   - Khởi tạo giá trị mặc định cho mô hình LIMN nếu chưa có.
3. **Dataset Verification**:
   - Xác thực sự hiện diện và tính toàn vẹn của thư mục dataset FashionIQ cùng các thư mục con `images/`, `image_splits/`, `captions/`.
4. **Docker Stack Startup**:
   - Nạp cấu hình `docker-compose.yml` kết hợp override `docker-compose.gpu.yml`.
   - Khởi động container `cir-workbench`.
   - Thăm dò mức độ sẵn sàng thật (readiness polling) thông qua endpoint API `GET /api/models` và cổng frontend 5173 theo thời hạn định sẵn.
5. **Model Preparation (LIMN)**:
   - Đồng bộ mã nguồn upstream của LIMN vào thư mục `/third_party`.
   - Chỉ khi operator chọn rõ `WORKBENCH_RUNTIME_ADAPTATION=sm_120`, cài profile cu128 đã khai báo cho LIMN; khai báo profile hoặc solver thành công chưa chứng minh cài đặt, import hay thực thi GPU đích.
   - Chuẩn bị dữ liệu FashionIQ derived (ảnh resized và correction dicts).
   - Tải bộ trọng số checkpoint bundle `base_iter0_all_categories` (`0_dress_best_model.pt`, `0_shirt_best_model.pt`, `0_toptee_best_model.pt`) và đối chiếu mã băm SHA256 chính thức.
6. **GPU Capability Verification**:
   - Thực thi `probe_gpu_capability` bên trong container trực tiếp trên môi trường của LIMN.
   - Kiểm tra tính toán tensor trên GPU và các file compiled ELF. Cơ chế fail-closed đảm bảo dừng ngay nếu không thể tính toán trên GPU thật.
7. **Evaluation Execution**:
   - Đưa job vào hàng đợi thông qua API Backend: `model_id=limn`, `checkpoint_id=base_iter0_all_categories`, `protocol_id=fashioniq_val_split`, `top_k=200`.
   - Cơ chế phát hiện trùng lặp: nếu đã có job tương ứng đang chạy, script tự động theo dõi (attach) thay vì tạo job mới; nếu đã có kết quả hoàn tất có chứng thực, script tái sử dụng trừ khi có cờ `--force`.
   - Theo dõi trạng thái job cho tới khi hoàn tất (`completed`, `failed`, `blocked`, `cancelled`).
8. **Validation & Index Rebuild**:
   - Đối chiếu báo cáo tổng hợp với sổ cái bằng chứng `evidence_ledger.jsonl`.
   - So sánh các chỉ số với ngưỡng tham chiếu bài báo khoa học.
   - Yêu cầu backend tái xây dựng chỉ mục DuckDB (`/api/index/rebuild`).
   - In bảng kết quả tổng hợp bằng tiếng Việt.

---

## 7. Kết quả thành công mong đợi

Khi quá trình đánh giá hoàn thành thành công, `./pc.sh` sẽ in bảng tổng kết các chỉ số đo lường thực tế trên terminal:

- **Metrics đo lường**: Recall@10 và Recall@50 cho từng danh mục (`dress`, `shirt`, `toptee`) và điểm trung bình vĩ mô (Macro Aggregate).
- **Trạng thái thẩm định đối chiếu bài báo (Scientific Benchmark Gate)**:
  - `PASS_WITHIN_5_PERCENT`: Giá trị đo lường nằm trong phạm vi sai số 5% so với số liệu công bố trong bài báo gốc.
  - `PASS_WITHIN_5_PERCENT_VARIANT`: Tái lập thành công trong ngưỡng 5% của biến thể đã ghi nhận.
  - `PASS_WITH_METRICS_ONLY_MAPPING_UNVERIFIED`: Các chỉ số đạt ngưỡng nhưng liên kết mapping chưa được xác thực đầy đủ.
  - `ALERT_METRIC_DEVIATION`: Metric lệch khỏi ngưỡng công bố của paper. **Lưu ý**: Đây là một kết quả nghiên cứu khoa học (RESEARCH FINDING), không phải lỗi hệ thống kỹ thuật.
  - `NOT_RUN`: Thẩm định chưa được chạy.

**Vị trí lưu trữ kết quả**:
- Báo cáo tổng hợp trong container: `/artifacts/reports/<model>_<checkpoint>_<run_id>_aggregate.json`
- Báo cáo tổng hợp trên host: `${HOST_WORKBENCH_ARTIFACTS_ROOT}/reports/<model>_<checkpoint>_<run_id>_aggregate.json` (mặc định tại `workbench/artifacts/reports/`).
- Sổ cái bằng chứng xác thực: `/artifacts/evidence_ledger.jsonl` (trên host: `workbench/artifacts/evidence_ledger.jsonl`).

Một báo cáo chỉ được phục vụ qua API khi được sổ cái xác thực đồng thuận: mã băm báo cáo trùng khớp, trường định danh hợp lệ, `return_code == 0`, và `invocation_status == "INVOCATION_COMPLETED"`.

---

## 8. Xem kết quả nghiên cứu

Sau khi chạy xong, bạn có thể phân tích sâu kết quả qua hai phương thức:

### 1. Giao diện Web UI trực quan

Mở trình duyệt và truy cập:
```text
http://localhost:5173
```

Giao diện Web cung cấp:
- Danh sách mô hình, checkpoint và các lượt chạy (runs).
- Trực quan hóa kết quả tìm kiếm CIR tương tác: xem ảnh truy vấn, văn bản chỉnh sửa và top-k ảnh được truy xuất.
- Phân tích nhóm lỗi (Failure Cohorts) theo từng thuộc tính trang phục.
- Bảng xếp hạng benchmark đối chiếu trực tiếp với bài báo.

### 2. Backend REST API

Backend API lắng nghe tại:
```text
http://localhost:8000
```

Các endpoint chính:
- `http://localhost:8000/api/models`: Danh sách mô hình đã đăng ký và cấu hình checkpoint.
- `http://localhost:8000/api/runs`: Toàn bộ các lần chạy đánh giá đã được chứng thực.
- `http://localhost:8000/api/jobs`: Hàng đợi tác vụ và trạng thái xử lý.
- `http://localhost:8000/api/benchmark-runs`: Báo cáo so sánh đối chuẩn khoa học.

Bạn cũng có thể xem trực tiếp các file JSON thô trong thư mục `workbench/artifacts/reports/` trên host.

---

## 9. Chạy model khác

Mặc định `./pc.sh` chạy mô hình LIMN. Để đánh giá mô hình khác đã đăng ký, sử dụng cờ `--model`:

```bash
./pc.sh --model csmcir
```

**Sự khác biệt giữa LIMN và CSMCIR**:
- **LIMN**:
  - `model_id`: `limn`
  - `checkpoint_id`: `base_iter0_all_categories` (gồm 3 checkpoint con: `base_iter0_dress`, `base_iter0_shirt`, `base_iter0_toptee`).
  - `protocol_id`: `fashioniq_val_split`
  - `top_k`: `200`
  - Khai báo `adapted_runtime`: Sử dụng runtime thích nghi cho `sm_120`.
- **CSMCIR**:
  - `model_id`: `csmcir`
  - `checkpoint_id`: Biến thể đơn `fashioniq`.
  - `protocol_id`: `fashioniq_original_split` (giao thức gốc của CSMCIR).
  - Không có `adapted_runtime`: CSMCIR không khai báo runtime thích nghi và chạy trên môi trường nguyên bản được hỗ trợ.

**Các tùy chọn CLI của `./pc.sh`**:
- `./pc.sh`: Chạy luồng chuẩn mặc định cho LIMN.
- `./pc.sh --model <tên_model>`: Chọn mô hình cần đánh giá (ví dụ `csmcir`).
- `./pc.sh --doctor`: Chỉ thực hiện kiểm tra chẩn đoán hệ thống host và cấu hình, hoàn toàn không làm thay đổi hệ thống, không khởi động container.
- `./pc.sh --verify-only`: Chỉ thực hiện kiểm chứng năng lực GPU trên môi trường mô hình rồi dừng lại, không chạy đánh giá. Cờ này **KHÔNG** xác thực một lượt chạy đã có, **KHÔNG** dựng lại chỉ mục DuckDB và **KHÔNG** tải checkpoint. Muốn xác thực lượt đã có rồi dựng lại chỉ mục, chạy `./pc.sh` không kèm cờ (nó tái sử dụng kết quả đã chứng thực thay vì đánh giá lại). Xem mục 11 để có lệnh cụ thể từng bước.
- `./pc.sh --force`: Buộc chạy lại toàn bộ quy trình đánh giá ngay cả khi đã có kết quả hoàn tất hợp lệ từ trước.
- `./pc.sh --timeout <số_giây>`: Đặt giới hạn thời gian chạy cho tác vụ đánh giá (tính bằng giây).
- `./pc.sh --no-gpu`: Chạy ở chế độ không dùng GPU (chỉ dùng cho kiểm tra tích hợp hoặc môi trường không có GPU).
- `./pc.sh --dataset <đường_dẫn>`: Chỉ định đường dẫn tùy chỉnh đến thư mục FashionIQ trên host.
- `./pc.sh --checkpoint <id>`: Ghim đúng một checkpoint hoặc bundle id. Dùng để **chủ động** chạy một
  danh mục đơn lẻ (ví dụ `base_iter0_dress`) — đây **KHÔNG** phải tái lập đầy đủ. Bỏ cờ này thì script
  chạy bundle đầy đủ mà registry khai, và **từ chối nêu tên** nếu thiếu thành phần, chứ không tự thu
  hẹp phạm vi.
- `./pc.sh --help`: In thông tin hướng dẫn và toàn bộ các tùy chọn hỗ trợ.

---

## 10. Lỗi thường gặp

### 1. Thiếu NVIDIA driver (`NVIDIA driver missing`)
- **Hiện tượng**: `nvidia-smi` báo lỗi lệnh không tồn tại hoặc `NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver`.
- **Nguyên nhân**: Driver NVIDIA chưa được cài đặt trên máy host hoặc bị ngắt kết nối sau khi cập nhật nhân Linux.
- **Khắc phục**: Cài đặt NVIDIA driver tương thích trên host hỗ trợ CUDA 12.8 trở lên (khuyến nghị driver dòng 570 trở lên, ví dụ: `sudo apt update && sudo apt install nvidia-driver-570`) và khởi động lại máy tính.
  - **Quy tắc tương thích**: Driver máy host bắt buộc phải hỗ trợ CUDA 12.8 để vận hành vi kiến trúc Blackwell (GeForce RTX 5070 Ti / `sm_120`). Đầu ra lệnh `nvidia-smi` hiển thị phiên bản CUDA tối đa mà driver hỗ trợ ở góc trên bên phải (yêu cầu $\ge 12.8$). Môi trường tính toán LIMN đã được tương thích hóa (`WORKBENCH_RUNTIME_ADAPTATION=sm_120`) tự động tải các gói wheel biên dịch sẵn cho CUDA 12.8 (`cu128`), do đó driver dòng 550 cũ sẽ không thể kích hoạt CUDA context trên phần cứng Blackwell.

### 2. Thiếu nvidia-container-toolkit (`nvidia-container-toolkit missing`)
- **Hiện tượng**: Quá trình preflight cảnh báo thiếu NVIDIA runtime, hoặc Docker báo lỗi `docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]`.
- **Nguyên nhân**: Docker daemon trên host chưa được cấu hình với NVIDIA Container Toolkit.
- **Khắc phục**:
  1. Cài đặt toolkit theo tài liệu chính thức của NVIDIA.
  2. Cấu hình Docker runtime: `sudo nvidia-ctk runtime configure --runtime=docker`.
  3. Khởi động lại dịch vụ Docker: `sudo systemctl restart docker`.

### 3. GPU không khả dụng (`GPU unavailable`)
- **Hiện tượng**: `./pc.sh --doctor` báo trạng thái `GPU validation: PENDING` với lý do `WORKBENCH_ENABLE_GPU is not 1` hoặc container không nhìn thấy card đồ họa.
- **Nguyên nhân**: Biến `WORKBENCH_ENABLE_GPU` bị tắt trong môi trường hoặc trong file `.env`.
- **Khắc phục**: Đảm bảo khai báo `WORKBENCH_ENABLE_GPU=1` trong `workbench/docker/.env` và chạy lại `./pc.sh`.

### 4. Không tương thích CUDA (`CUDA incompatibility`)
- **Hiện tượng**: PyTorch ném lỗi không tìm thấy kernel CUDA tương thích cho kiến trúc `sm_120` (GeForce RTX 5070 Ti - Blackwell), hoặc lỗi khởi tạo CUDA context.
- **Nguyên nhân**: Thư viện PyTorch nguyên bản của mô hình (ví dụ PyTorch 1.12.1 của LIMN phát hành năm 2022) không hỗ trợ phần cứng kiến trúc mới.
Khắc phục: Không giả định runtime thích nghi đã được cài hoặc hoạt động. Chọn rõ `WORKBENCH_RUNTIME_ADAPTATION=sm_120` cho model có profile tương ứng; sau provision, xác minh import và capability trên GPU đích. Profile hiện được khai báo cho chín model được chọn, nhưng mọi thực thi `sm_120` vẫn `TARGET_GPU_PENDING`.

### 5. Xung đột phụ thuộc Python (`Python dependency conflict`)
- **Hiện tượng**: Quá trình chuẩn bị môi trường mô hình thất bại trong bước solver của conda/mamba hoặc cài đặt pip.
- **Nguyên nhân**: Xung đột phiên bản giữa các gói phụ thuộc bên ngoài khi cài đặt.
- **Khắc phục**: Workbench sử dụng `micromamba` tích hợp sẵn trong container với danh sách ràng buộc đã được khóa chặt (lockfile). Xem file log tại thư mục `workbench/artifacts/environments/` để xác định gói gây lỗi.

### 6. Hết dung lượng đĩa (`Disk space exhausted`)
- **Hiện tượng**: Docker build hoặc quá trình tải checkpoint báo lỗi `No space left on device` (ENOSPC).
- **Nguyên nhân**: Dataset FashionIQ (ảnh và captions), 3 file weights của LIMN (~1.8 GB mỗi file), và image Docker chiếm dung lượng đáng kể.
- **Khắc phục**: Đảm bảo phân vùng làm việc còn ít nhất 50 GB trống. Dọn dẹp các image hoặc build cache Docker cũ không dùng bằng lệnh `docker system prune -f`, sau đó kiểm tra lại bằng `df -h`.

### 7. Thiếu dataset (`Dataset missing`)
- **Hiện tượng**: `./pc.sh` dừng lại và thông báo không tìm thấy thư mục dataset hoặc thiếu các thư mục con `images/`, `image_splits/`.
- **Nguyên nhân**: Đường dẫn `HOST_FASHIONIQ_ROOT` trong `.env` chưa chính xác hoặc dữ liệu chưa được giải nén đúng cấu trúc.
- **Khắc phục**: Kiểm tra lại cấu trúc thư mục như mô tả ở Mục 3, sau đó truyền đường dẫn chính xác bằng cờ:
  ```bash
  ./pc.sh --dataset /duong/dan/that/toi/FashionIQ
  ```

### 8. Sai lệch mã băm checkpoint (`Checkpoint hash mismatch`)
- **Hiện tượng**: Quá trình chuẩn bị checkpoint dừng lại và báo lỗi hash mismatch đối với một trong các file `.pt`.
- **Nguyên nhân**: Quá trình tải file từ máy chủ từ xa bị đứt đoạn khiến file bị hỏng hoặc tải thiếu.
- **Khắc phục**: Xóa file checkpoint hỏng tương ứng trong `workbench/artifacts/checkpoints/<tên_model>/` và chạy lại `./pc.sh`. Hệ thống sẽ tự động tải lại và kiểm tra mã băm SHA256 chính thức.

### 9. Tràn bộ nhớ GPU (`CUDA out of memory`)
- **Hiện tượng**: Quá trình trích xuất đặc trưng hoặc đánh giá bị dừng với ngoại lệ `torch.cuda.OutOfMemoryError`.
- **Nguyên nhân**: Một tiến trình khác đang chiếm dụng bộ nhớ trên card RTX 5070 Ti, hoặc batch size vượt quá dung lượng 16 GB VRAM.
- **Khắc phục**: Kiểm tra các tiến trình khác trên host qua `nvidia-smi` và tắt các tác vụ nặng không cần thiết. Cấu hình mặc định của hệ thống đã được căn chỉnh an toàn cho 16 GB VRAM.

### 10. Quyền hạn Docker (`Docker permissions error`)
- **Hiện tượng**: Lệnh `./pc.sh` dừng ngay từ bước preflight với lỗi `permission denied while trying to connect to the Docker daemon socket`.
- **Nguyên nhân**: User hiện tại chưa được cấp quyền truy cập Docker socket mà không cần `sudo`.
- **Khắc phục**: Thêm người dùng vào nhóm docker trên host:
  ```bash
  sudo usermod -aG docker $USER
  ```
  Sau đó đăng xuất và đăng nhập lại hệ thống (hoặc chạy lệnh `newgrp docker`) rồi thử lại.

### 11. Thiếu đầu ra benchmark (`Benchmark output missing`)
- **Hiện tượng**: Tác vụ đánh giá kết thúc nhưng không có báo cáo tổng hợp trong thư mục reports hoặc API trả về thông tin rỗng.
- **Nguyên nhân**: Tiến trình đánh giá gặp ngoại lệ bất thường trước khi ghi nhận metrics đầy đủ.
- **Khắc phục**: Mở file log của job tương ứng tại `workbench/artifacts/jobs/<job_id>.log` để đọc traceback lỗi chi tiết, hoặc kiểm tra trạng thái hàng đợi bằng:
  ```bash
  workbench/docker/run.sh status
  ```

### 12. Sổ cái bằng chứng không xác minh được (`ledger does not verify`)
- **Hiện tượng**: `./pc.sh` báo xác thực thất bại với lý do liên quan tới evidence ledger, dù đánh giá đã chạy xong. API cũng trả về `400` với `error=ledger_key_unusable` cho `/api/benchmark-runs` (kèm cả hai đường dẫn khóa), thay vì trả danh sách rỗng `[]` không nêu lý do.
- **Nguyên nhân**: Sổ cái bằng chứng được ký bằng khóa HMAC. Khóa này nằm ở `$XDG_CONFIG_HOME/cir/workbench_ledger_key` — trong container là `/artifacts/config/cir/workbench_ledger_key`, tức **trên ổ đĩa host** (bind mount) nên bền vững và ghi được bởi uid vận hành. Nếu khóa bị mất, đổi, hoặc có **hai** khóa khác nhau cùng tồn tại (ví dụ một bản cũ ở gốc repo), mọi entry đã ký bằng khóa kia sẽ không xác minh được nữa.
- **Khắc phục**:
  ```bash
  # 1) Xem có mấy khóa đang tồn tại
  ls -la workbench/artifacts/config/cir/workbench_ledger_key
  ls -la .workbench_ledger_key          # bản cũ ở gốc repo (nếu có)
  ```
  - Nếu chỉ có **một** khóa: chạy lại `./pc.sh` — hệ thống tự dùng lại khóa đó.
  - Nếu có **hai** khóa khác nhau: hệ thống cố ý **từ chối** thay vì đoán sai. Giữ lại khóa đã ký sổ cái hiện có, xoá bản còn lại, hoặc đặt tường minh `WORKBENCH_LEDGER_KEY` trong `workbench/docker/.env`.
  - Không xoá sổ cái `evidence_ledger.jsonl`: đó là bằng chứng của các lần chạy trước.

---

## 11. Khôi phục

CIR Failure Analysis Workbench được xây dựng theo nguyên tắc bảo toàn trạng thái (Resumable) và bất biến (Idempotent):

- **Tự động khôi phục**: Nếu quá trình chạy `./pc.sh` bị ngắt quãng giữa chừng (do mất điện, nhấn Ctrl+C, hoặc sự cố mạng khi tải checkpoint), bạn chỉ cần chạy lại:
  ```bash
  ./pc.sh
  ```
- Script sẽ nhận biết các thành phần đã hoàn thành hợp lệ (checkpoint đã có mã băm đúng, môi trường mô hình đã tạo, dữ liệu đã chuẩn hóa) và tiếp tục thực hiện giai đoạn tiếp theo mà không lặp lại các công việc trước đó.
- **Tự động gắn vào job đang chạy (Job Attachment)**: Nếu Docker stack đã có một job đang được xếp hàng (`queued`) hoặc đang xử lý (`running`), `./pc.sh` sẽ tự động chuyển sang trạng thái theo dõi job đó thay vì tạo thêm job trùng lặp.
- **Tái sử dụng kết quả**: Nếu một lượt chạy cho cùng mô hình, cùng checkpoint bundle và giao thức đã hoàn tất với báo cáo được sổ cái chứng thực, `./pc.sh` sẽ hiển thị ngay kết quả đó. Nếu muốn ép buộc chạy đánh giá mới từ đầu, hãy sử dụng:
  ```bash
  ./pc.sh --force
  ```

- **Lệnh mặc định chạy ĐẦY ĐỦ hoặc TỪ CHỐI, không bao giờ thoái hoá**

  `./pc.sh` yêu cầu **bundle đầy đủ ba danh mục** mà registry của mô hình đã khai
  (`base_iter0_all_categories` với LIMN). Hai kết cục duy nhất:

  * **Chạy đủ** — cả ba trọng số đã tải, chạy được, mapping đã xác thực, sha khớp.
  * **Từ chối nêu tên** — thiếu một thành phần thì script **dừng** ở giai đoạn
    `CHECKPOINT_SELECTION`, in ra **đúng thành phần nào còn thiếu hoặc chưa chạy được** và
    **không xếp job nào**. Xem mục 6 để tải nốt rồi chạy lại.

  Script sẽ **không** âm thầm rút xuống một danh mục rồi trình bày như một lần tái lập đầy đủ. Một
  danh mục đơn lẻ làm sai lệch chính đại lượng mà bài báo công bố (macro trên ba danh mục), nên nó
  chỉ được chạy khi bạn **chủ động** yêu cầu:

  ```bash
  ./pc.sh --checkpoint base_iter0_dress     # CHỦ ĐỘNG chạy một danh mục
  ```

  Khi đó script nói rõ đây **KHÔNG** phải tái lập đầy đủ, và UI cũng gắn nhãn
  `Single-category (partial evaluation — NOT complete reproduction)`.

- **Lệnh xác thực và dựng lại chỉ mục (từng bước, chạy được thật)**

  ```bash
  # 1. Kiểm chứng năng lực GPU của môi trường mô hình, rồi DỪNG.
  #    KHÔNG xác thực lượt chạy, KHÔNG dựng lại chỉ mục, KHÔNG tải checkpoint.
  ./pc.sh --verify-only

  # 2. Xác thực lại một lượt chạy đã có (các cờ --model/--checkpoint/--protocol là BẮT BUỘC),
  #    chạy bên trong container vì đó là nơi có môi trường và khoá sổ cái:
  docker exec cir-workbench python3 workbench/scripts/verify_run.py \
    --run-id <run_id> --model limn \
    --checkpoint base_iter0_all_categories --protocol fashioniq_val_split --json

  # 3. Kiểm tra mọi artifact canonical theo schema:
  python -m workbench.scripts.validate_results --all --strict-real

  # 4. Dựng lại chỉ mục DuckDB mà không cần khởi động lại container:
  curl -sS -X POST http://127.0.0.1:8000/api/index/rebuild

  # 5. Xác nhận lượt chạy đã hiện trên API/UI:
  curl -sS http://127.0.0.1:8000/api/runs | python3 -c "import json,sys; print([r['run_id'] for r in json.load(sys.stdin)])"
  curl -sS http://127.0.0.1:8000/api/benchmark-runs | python3 -c "import json,sys; print([r['run_id'] for r in json.load(sys.stdin)])"
  ```

  `/api/runs` là **chỉ mục per-query** (chỉ LIMN có); `/api/benchmark-runs` là **báo cáo tổng hợp** (mọi
  model). Một lượt aggregate-only **không** có mặt ở endpoint thứ nhất — đó là đúng theo thiết kế.

- **Kết quả đã chứng thực thì không tự "cũ đi"**

  Nếu sau khi có kết quả chứng thực mà bạn (hoặc một tiến trình khác) sửa **một byte** của trọng số,
  đổi file `command.json` hay log của lượt chạy, hệ thống sẽ **rút chứng thực** ở lần đọc kế tiếp:
  lượt chạy vẫn hiển thị (đó là một sự kiện đã xảy ra) nhưng **không còn** được trình bày là bằng
  chứng nghiên cứu. Khôi phục lại byte gốc thì chứng thực trở lại. Đây là chủ ý: tem chứng thực là
  một phát biểu về **nội dung hiện tại** của artifact, không phải một thuộc tính vĩnh viễn của tên file.

---

## 12. Log

Hệ thống ghi nhận log minh bạch tại các vị trí xác định:

- **Thông báo lỗi tức thì**: Khi `./pc.sh` gặp lỗi, vị trí tệp log cụ thể sẽ được in trực tiếp ra terminal.
- **Log của Docker container**:
  Để theo dõi luồng log tổng thể của container Docker (backend và frontend), sử dụng công cụ nâng cao:
  ```bash
  workbench/docker/run.sh logs
  ```
- **Log của từng Job đánh giá**:
  Mỗi lần chạy evaluation được ghi lại tại:
  - Trong container: `/artifacts/jobs/<job_id>.log`
  - Trên host: `${HOST_WORKBENCH_ARTIFACTS_ROOT}/jobs/<job_id>.log` (mặc định `workbench/artifacts/jobs/`).
- **Sổ cái bằng chứng khoa học**:
  Mọi tác vụ hoàn thành đều được ghi một dòng JSON vào:
  - `workbench/artifacts/evidence_ledger.jsonl`

---

## 13. Dừng hệ thống

Khi hoàn tất công việc nghiên cứu hoặc cần giải phóng tài nguyên hệ thống, bạn sử dụng lệnh:

```bash
workbench/docker/run.sh stop
```

Lệnh này thực hiện `docker compose down` một cách an toàn.

**Nguyên tắc bảo toàn dữ liệu**:
Lệnh dừng hệ thống **TUYỆT ĐỐI KHÔNG** xóa kết quả đánh giá (`/results`), không xóa các tệp báo cáo (`/artifacts/reports`), không xóa checkpoints đã tải (`/checkpoints`), và không xóa chỉ mục dữ liệu DuckDB. Toàn bộ dữ liệu nằm an toàn trên máy host và sẵn sàng cho các phiên làm việc tiếp theo.

---

## 14. Tính toàn vẹn khoa học

Trong nghiên cứu thực nghiệm khoa học máy tính, việc hiểu đúng bản chất của kết quả là tối quan trọng. Workbench phân biệt rõ ràng 5 khái niệm sau:

1. **Chạy thành công (`Execution Success`)**:
   - Là việc tiến trình chạy từ đầu đến cuối với mã thoát 0 (`return_code == 0`) và trạng thái job là `completed`.
   - Ý nghĩa: Hệ thống kỹ thuật và phần mềm không bị crash. Điều này chưa nói lên độ chính xác về mặt khoa học của mô hình.

   **Quan trọng — con số đến từ output, không từ lời tự khai.** Trình xác thực
   (`verify_run.py`) **đọc lại chính `stdout.log`/`stderr.log` của lượt chạy** (hai file đã được
   `evidence_digest` ràng buộc vào completion proof) và **trích xuất lại** metric bằng đúng parser mà
   evaluator đã dùng, rồi **so với con số mà report khai**. Nếu output không chứa dòng metric nào,
   hoặc chứa số khác với report, lượt chạy **không** được coi là bằng chứng nghiên cứu.

   Lý do phải làm vậy: các trường như `extraction_status` hay `metric_extraction_available` do **chính
   lượt chạy tự viết về mình**. Một report có thể khai `EXTRACTED_VERIFIED` mang số của bài báo trong
   khi output không hề có metric — trước đây hệ thống đã chấp nhận đúng tình huống đó. Output là phần
   duy nhất mà report **không thể tự trả lời thay**.

2. **Model đã được xác thực GPU (`GPU-Verified Runtime Capability`)**:
   - Khác biệt hoàn toàn với việc chỉ chạy được trên CPU.
   - Ý nghĩa: Khẳng định rằng các phép toán tensor thực sự chạy trên nhân phần cứng của GPU NVIDIA (RTX 5070 Ti) với thư viện compiled ELF tương thích. Hệ thống hoạt động theo nguyên tắc fail-closed, tuyệt đối không bao giờ làm giả capability GPU.
3. **Đánh giá đầy đủ (`Complete Evaluation Coverage`)**:
   - Là việc quá trình đánh giá đã thực hiện trọn vẹn trên toàn bộ tập dữ liệu kiểm thử của cả 3 danh mục FashionIQ (`dress`, `shirt`, `toptee`) với ngưỡng xếp hạng `top_k=200`.
   - Ý nghĩa: Khác với các đợt chạy thử nghiệm rút gọn (smoke test) chỉ kiểm tra vài sample.
4. **Khớp số liệu paper (`Paper Reference Parity`)**:
   - So sánh trực tiếp giá trị Recall đo được với bảng số liệu đã công bố trong bài báo gốc thông qua Scientific Benchmark Gate (ví dụ trạng thái `PASS_WITHIN_5_PERCENT`).
   - **Nguyên tắc cốt lõi**: Nếu Recall đo được thấp hơn hoặc lệch khỏi bài báo (`ALERT_METRIC_DEVIATION`), đây là một **PHÁT HIỆN NGHIÊN CỨU (RESEARCH FINDING)** về khả năng tái lập trong thực tế khoa học. Đây **TUYỆT ĐỐI KHÔNG PHẢI** là lỗi hạ tầng hay sự cố kỹ thuật.
5. **Tái lập với runtime đã adapt (`Reproduction under Adapted Runtime`)**:
   - Phân biệt minh bạch giữa việc tái lập trên môi trường đóng băng lịch sử nguyên bản (legacy pinned runtime) và môi trường đã thích nghi phần cứng (`ADAPTED_RUNTIME`).
   - Ý nghĩa: Khi mô hình cổ điển được nâng cấp runtime (PyTorch >=2.7 để chạy được trên Blackwell sm_120), báo cáo được gắn cờ `ADAPTED_RUNTIME` trong provenance. Hệ thống không bao giờ gán nhãn đây là "tái tạo chính xác tuyệt đối" (exact historical reproduction), nhằm bảo đảm tính trung thực học thuật cao nhất.

### Trạng thái chứng thực của một báo cáo trên API

API trả về một đối tượng `verification` **tách biệt** bốn câu hỏi, để bạn không phải suy diễn:

| Trường | Ý nghĩa |
|---|---|
| `artifact_exists` | Có tệp báo cáo tổng hợp trên đĩa (kể cả báo cáo lịch sử / đã bị từ chối) |
| `execution_authenticated` | Byte của báo cáo đúng là byte sổ cái HMAC ghi nhận, đúng định danh, `return_code == 0`, `invocation_status == "INVOCATION_COMPLETED"` |
| `is_research_evidence` / `certified` | **Có phải bằng chứng nghiên cứu hợp lệ**: completion proof xác thực, metric extraction thực sự đã chạy, macro khớp category |
| `parity_status` / `benchmark_gate` | Phán quyết đối chiếu paper (mục 4 ở trên) |

Các trạng thái `state` có thể gặp:

- `certified` — bằng chứng nghiên cứu hợp lệ. Chỉ trạng thái này mới được trình bày như kết quả đã chứng thực.
- `proof_unavailable` — sổ cái xác thực được việc thực thi, nhưng tệp completion proof không còn (bị xoá/prune). **Không** phải `certified`; dùng để phân biệt với trường hợp dưới.
- `uncorroborated` — không có entry sổ cái khớp. **Không** phải `certified`.
- `refused` — proof có nhưng bị từ chối (chữ ký, định danh, hoặc số liệu không hợp lệ).

Giao diện Web hiển thị nhãn riêng cho từng trạng thái: một run chưa `certified` **không bao giờ** được vẽ như một lượt pass sạch.

### Từ vựng phán quyết của release verifier

`workbench/docker/release_verify.sh` không còn chỉ nhìn J1–J12: nó đánh giá **10 chiều độc lập**
(`STATIC_SOURCE`, `DEPLOYMENT`, `CONFIGURATION`, `MODEL_ENVIRONMENT`, `OPERATIONS`, `GPU_RUNTIME`,
`MODEL_INFERENCE`, `MODEL_EVALUATION`, `SCIENTIFIC_OUTPUT`, `TARGET_BINDING`) và chỉ đạt
`RELEASE_READY` khi **mọi** chiều bắt buộc PASS.

Hai điểm cần lưu ý về cách các chiều được quyết định:

- **`TARGET_BINDING`** trói mọi chiều gắn với mục tiêu vào **một danh tính phát hành đã khai**
  (`WORKBENCH_RELEASE_TARGET_MODEL/CHECKPOINT/PROTOCOL/INTERPRETER`, mặc định là artifact LIMN mà
  repo này phát hành). Bằng chứng của model/checkpoint/giao thức/runtime adaptation **khác** bị **nêu
  tên** trong `off_target_evidence_runs` và **không** được tính. Một mục tiêu không giải được → chiều
  này FAIL → `RELEASE_BLOCKED`, chứ không âm thầm nới thành "nhận bất kỳ".
- **Một chiều không thể PASS nếu check của chính nó không PASS.** `SCIENTIFIC_OUTPUT` cần cả check
  `J7` **và** bằng chứng; bằng chứng một mình không đủ để bật một chiều đang PENDING hay FAIL.

| Verdict | Nghĩa |
|---|---|
| `RELEASE_READY` | Mọi chiều bắt buộc PASS, gồm GPU thật, suy luận thật và đánh giá khoa học thật |
| `GPU_VALIDATION_PENDING` | Kỹ thuật đạt, nhưng chưa có bằng chứng GPU trên máy đích |
| `ENGINEERING_INCOMPLETE` | Có chiều bắt buộc PENDING vì lý do triển khai (ví dụ chạy `--offline`) |
| `RELEASE_BLOCKED` | Có chiều bắt buộc FAIL (ví dụ khóa sổ cái không dùng được) |

Trên chính máy phát hành này, verdict đúng là `ENGINEERING_INCOMPLETE` (chế độ `--offline`) hoặc
`RELEASE_BLOCKED` khi có xung đột khóa sổ cái — **không bao giờ** là `RELEASE_READY`, vì chưa có GPU
đích, suy luận thật hay đánh giá FashionIQ thật. Đó là hành vi trung thực, không phải lỗi.

---

## 15. Phân tích theo từng truy vấn (per-query export)

Sau khi có một lượt LIMN đầy đủ, bước tiếp theo của nghiên cứu là phân tích **theo từng truy vấn**:
truy vấn nào thất bại ở mọi mô hình, ảnh nhiễu nào bị lấy nhầm chung, mức độ trùng nhau của tập thất
bại. Đường đánh giá chính thức của LIMN (`./pc.sh`) **không** sinh ra dữ liệu đó — hàm `test()` gốc
tính ranking rồi **vứt đi**, chỉ giữ lại Recall tổng hợp.

Việc xuất artifact per-query là **tự động** bên trong `./pc.sh` cho LIMN — không có cờ nào để bật, và
**không cần chạy tay**. Nó là **hai tầng**, cả hai chạy dưới cùng một `run_id` do orchestrator sinh:

1. `python -m workbench.replay.limn_export` — chạy dưới **interpreter của model**; quan sát ranking và
   ghi ra **một payload JSON thuần** (không import schema canonical, vì môi trường model không có
   pydantic). Tham số bắt buộc: `--checkpoint-root`, `--dataset-root`, `--source-root`, `--results-root`.
2. `python -m workbench.replay.limn_publish --payload <payload>` — chạy dưới **interpreter workbench**;
   validate payload, kiểm chính sách admission, công bố artifact canonical atomic.

Hai tầng phải chạy cùng nhau: `limn_export` mới chỉ ghi payload trung gian, **chưa** tạo artifact
canonical. Chạy mỗi tầng 1 bằng tay sẽ để lại một payload chưa được công bố, và `/api/runs` sẽ không
thấy gì. Nếu buộc phải chạy tay để điều tra, chạy đúng cả hai như trên, với `--run-id` do bạn chọn.

Đây là một **entry point riêng**, cố ý tách khỏi `workbench/replay/limn.py`. Lý do kỹ thuật: bộ lập
kế hoạch môi trường truy vết import từ `limn.py` để quyết định interpreter của LIMN, nên mọi thứ
import trong file đó trở thành **yêu cầu môi trường đã khai**; để observer (numpy) và schema kết quả
canonical (pydantic) ở đó sẽ khiến LIMN đòi interpreter cho những gói đường chạy native không hề dùng.

**Hợp đồng — đọc kỹ trước khi tin kết quả:**

1. **Parity thiết bị đo (instrumentation parity) là điều kiện bắt buộc.** Sau khi quan sát ranking,
   hệ thống **tính lại Recall từ chính ranking đó** và so với Recall native cho R@1/R@10/R@50 **từng
   category**, ở dạng **số nguyên hit**. Lệch ⇒ exit code 1, artifact bị **giữ lại**, và lý do được in
   ra. Kết quả tổng hợp native **không** bị sửa để hai bên khớp nhau.
2. **Con số đến từ ranking, không từ report.** Mọi metric trong artifact được dẫn xuất từ ranking đã
   quan sát ở bước trên.
3. **Caption thô được giữ nguyên.** `raw_captions` là nguyên văn từ `cap.{category}.val.json`;
   `model_input_text` là văn bản **thực sự** đưa vào LIMN (đã lowercase, bỏ dấu câu, áp
   `correction_dict`, nối bằng `" and "`). Hai trường này **không** thay thế nhau và **không** được
   ghi đè lẫn nhau.
4. **Kích thước gallery theo từng category.** FashionIQ có gallery khác nhau cho dress/shirt/toptee;
   artifact lưu `category_gallery_sizes` và `category_eligible_counts`, và validator dùng đúng bound
   của từng category — rank 300 ở dress không được "tha" bởi gallery toptee lớn hơn.
5. **Chưa có xác thực trên GPU đích.** Trên máy dev, `QUERY_EXPORT_IMPLEMENTED` là đúng, nhưng
   `QUERY_EXPORT_NATIVE_PARITY_VERIFIED` **phải** vẫn là PENDING cho tới khi chạy thật trên PC. Không
   có chứng thư GPU nào được tạo ra ở đây.
6. **Artifact tổng hợp không bị thay thế.** File aggregate chính thức vẫn là kết quả đánh giá; artifact
   per-query chỉ là **bằng chứng bổ sung**.

---

## 16. Quy trình nghiên cứu sau lần đánh giá đầu tiên

Sau khi hoàn tất lần đánh giá đầu tiên với `./pc.sh`, bạn có thể tiến hành chu trình phân tích khoa học tiếp theo.

> **Giới hạn hiện tại — đọc trước khi dùng giao diện phân tích theo truy vấn**
>
> `./pc.sh` chạy đường đánh giá chính thức của LIMN, và đường đó sinh ra **Recall tổng hợp theo
> category** (R@1/R@10/R@50 cho dress/shirt/toptee) cùng macro aggregate — đúng những gì paper công bố.
> Nó **không** xuất xếp hạng Top-K theo từng truy vấn.
>
> Hệ quả: các màn hình phân tích *theo từng truy vấn* (`/api/queries`, so sánh cặp model trên cùng
> một truy vấn, Jaccard của tập thất bại) hiện **chưa vận hành được với kết quả LIMN thật**. Chúng
> chỉ chạy trên dữ liệu mock dùng cho phát triển. **Không có kết quả truy hồi nào bị tạo ra giả** ở
> bất kỳ đâu.
>
> Chi tiết kỹ thuật và phần triển khai tối thiểu còn thiếu: `workbench/docs/reviews/final_pc_audit/PER_QUERY_EXPORT_STATUS.md`.
>
> Điều **vẫn dùng được ngay** sau lần chạy đầu: Recall theo category, macro aggregate, thẩm định
> paper gate, và so sánh giữa các model ở mức category.

1. **Đọc kết quả ở mức category và macro**:
   - Truy cập giao diện Web UI tại `http://localhost:5173`.
   - Đối chiếu R@10/R@50 của dress/shirt/toptee với số liệu paper do `verify_run.py` và
     benchmark gate cung cấp: đây là mức mà paper công bố và là mức được thẩm định.
   - Sai lệch so với paper là **phát hiện nghiên cứu**, không phải lỗi hạ tầng.
2. **Phân tích theo Cohorts**:
   - Cohorts và nhãn annotation là công cụ gán nhãn thủ công của bạn: bạn tự tạo cohort rồi gắn nhãn lên các truy vấn mình chọn. Hệ thống lưu và xuất được chúng.
   - Việc hệ thống *tự động* phân loại truy vấn thất bại thành "nhầm màu sắc / sai kiểu dáng / sai hoa văn" cần dữ liệu xếp hạng theo từng truy vấn — xem khối giới hạn ở đầu mục này.
3. **Đánh giá so sánh đa mô hình**:
   - Chạy đánh giá mô hình thứ hai bằng lệnh:
     ```bash
     ./pc.sh --model csmcir
     ```
   - So sánh R@10/R@50 theo category giữa LIMN và CSMCIR ở mức số liệu paper công bố.
4. **Khai thác dữ liệu báo cáo chuyên sâu**:
   - Các file báo cáo tổng hợp JSON tại `workbench/artifacts/reports/` chứa metric theo category,
     macro aggregate, thẩm định gate, provenance checkpoint và metadata thực thi (commit nguồn,
     digest môi trường, runtime adaptation) — đủ để trích dẫn trong báo cáo nghiên cứu.
   - Ma trận nhầm lẫn và so sánh theo từng truy vấn **không** nằm trong báo cáo tổng hợp; chúng cần
     dữ liệu per-query như đã nêu ở khối giới hạn phía trên.
