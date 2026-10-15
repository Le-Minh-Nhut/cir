# PC Docker Quickstart — CIR Failure Analysis Workbench

Máy đích: PC với RTX 5070 Ti (sm_120 / Blackwell, 16 GB). Tất cả lệnh chạy từ thư mục
`workbench/docker/`.

## 1. Chuẩn bị

```bash
cd workbench/docker
cp .env.example .env          # sửa HOST_* nếu repo không nằm ở vị trí mặc định
```

Volumes mà compose mount (mặc định trỏ về thư mục repo, resolve từ `workbench/docker/`):

| Host (mặc định)                              | Trong container                    | Nội dung                       |
| -------------------------------------------- | ---------------------------------- | ------------------------------ |
| `../../data/FashionIQ`                       | `/data/FashionIQ`                  | Dataset FashionIQ              |
| `../../data/FashionIQ/images`                | `/data/FashionIQ/images` (**ro**)  | Ảnh canonical, bất khả biến    |
| `../../data/FashionIQ/image_splits`          | `/data/FashionIQ/image_splits` (**ro**) | Split canonical, bất khả biến |
| `../../workbench/third_party`                | `/third_party`                     | Source model đã sync           |
| `../../workbench/artifacts/checkpoints`      | `/checkpoints`                     | Weights đã tải                 |
| `../../workbench/artifacts`                  | `/artifacts`                       | DuckDB index, runs, reports    |
| `../../workbench/artifacts/results`          | `/results`                         | Kết quả evaluation             |

Dataset root writable: preparation ghi artifact derived (`resized_image/`,
`captions/correction_dict_*.json`) bên cạnh file canonical. Hai submount `images/` và
`image_splits/` là read-only ở cấp kernel — chuẩn DATA-02: không quá trình nào trong
container có thể sửa dữ liệu benchmark gốc.

## 2. Chạy workbench

```bash
./run.sh up       # build + start; doctor chạy ngay sau đó
./run.sh logs     # xem log
./run.sh stop
```

- Backend API: `http://127.0.0.1:8000` (`/api/models`, `/api/runs`, …)
- Frontend: `http://127.0.0.1:5173` (proxy `/api` về backend cùng container)

Lần chạy đầu, `/api/runs` trả về 400 `index_unavailable` cho tới khi DuckDB index được
rebuild — UI hiển thị lỗi này trung thực; các endpoint khác vẫn hoạt động.

## 3. Bật GPU (Blackwell)

Cần `nvidia-container-toolkit` trên host (quyền sudo). Sau khi cài:

```bash
WORKBENCH_ENABLE_GPU=1 ./run.sh up
```

`run.sh` nạp thêm `docker-compose.gpu.yml` (override yêu cầu tất cả GPU NVIDIA). Không có
toolkit, container từ chối start thay vì lặng lẽ chạy CPU-only (fail closed). Sau đó:

1. `./run.sh doctor` — phần "GPU visibility" chỉ báo gì container thấy thật.
2. Kiểm chứng runtime model: `verify_gpu_runtime()` chạy probe thật (tensor + compiled core
   ELF) bên trong environment của model; kết quả gắn với host/driver, hết hạn khi host đổi.

Nếu toolkit chưa cài: `doctor` báo **PENDING** kèm lý do — workbench không bao giờ giả lập
kết quả GPU.

## 4. Runtime thích nghi cho Blackwell (sm_120)

Pin lịch sử của LIMN (torch 1.12.1) không có kernel sm_120. Registry khai báo
`adapted_runtime` (torch >=2.7,<3 cu128, torchvision >=0.22,<1, open_clip_torch >=2.24).
Chỉ hoạt động khi operator chọn rõ ràng:

```bash
export WORKBENCH_RUNTIME_ADAPTATION=sm_120
```

Mọi run chạy dưới adapted runtime được đánh dấu `ADAPTED_RUNTIME` trong `variant_provenance`
của report — không bao giờ được dán nhãn tái tạo chính xác (exact reproduction) paper.

## 5. Pilot evaluations

```bash
./run.sh prepare limn    # sync source, tạo env, dataset prep, tải checkpoints, probe GPU
./run.sh pilot limn      # prepare (nếu thiếu) + xếp job evaluation qua API
./run.sh pilot csmcir    # ưu tiên 2
./run.sh status
```

`pilot` chọn **bundle** đã đăng ký trong registry (`base_iter0_all_categories` = dress +
shirt + toptee) — không bao giờ dừng ở một category. Job chỉ được xếp khi preflight pass;
gate paper 5% là phán quyết chuẩn, không thể bỏ qua.

## 6. Troubleshooting

- `runs: index_unavailable` → rebuild index (POST `/api/index/rebuild`) hoặc khởi tạo artifacts.
- `failed to discover GPU vendor from CDI` → toolkit chưa cài/cấu hình trên host.
- Job trạng thái lowercase (`queued`/`preparing`/`running`/`blocked`/`completed`/`failed`/`cancelled`) — đó là contract backend.
