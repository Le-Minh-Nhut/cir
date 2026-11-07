# Hai run exit 0 — mỗi run truy vết được độc lập

Hai lần chạy CSMCIR thành công trong buổi 20261107. Cả hai cho **cùng số đo đến 1e-13**.

| | run A | run B |
|---|---|---|
| run_id | `20261107T054343058424Z_csmcir_fashioniq_f564570e` | `20261107T054933277898Z_csmcir_fashioniq_f03887d2` |
| log dir | `workbench/artifacts/logs/csmcir__fashioniq__20261107T054343058424Z_csmcir_fashioniq_f564570e/` | `workbench/artifacts/logs/csmcir__fashioniq__20261107T054933277898Z_csmcir_fashioniq_f03887d2/` |
| bắt đầu (UTC) | 05:43:43.058459 | 05:49:33.277931 |
| kết thúc (UTC) | 05:48:00.255038 | 05:53:38.814318 |
| `return_code` | 0 | 0 |
| `invocation_status` | `INVOCATION_COMPLETED` | `INVOCATION_COMPLETED` |
| `verdict` | `PROOF_VALIDATED` | `PROOF_VALIDATED` |
| `reproduction_status` | `EXTRACTED_VERIFIED` | `EXTRACTED_VERIFIED` |
| MACRO R@10 | 57.05073873202006 | 57.05073873202006 |
| MACRO R@50 | 77.29859352111816 | 77.29859352111816 |
| `evaluator_environment_knobs` | **`null` — không ghi lại** | đủ bốn knob |
| provenance đầy đủ | **KHÔNG** | **CÓ** |
| `command.json` sha256 | `4ef74d2255913e84f962f973780c1a80c21bdb88d58a1b2640b5d585ba4381ff` | `5673efbf58bdf09fd87487ace6fe771b667661b8fcf58ab62ccd903a19f30dc6` |

Từng category, đo trên chính `aggregate_report.json` của mỗi run (giống nhau tuyệt đối):

| category | R@10 | R@50 |
|---|---|---|
| dress | 52.20624804496765 | 74.86366033554077 |
| shirt | 57.75269865989685 | 76.1040210723877 |
| toptee | 61.19326949119568 | 80.92809915542603 |
| **macro** | **57.05073873202006** | **77.29859352111816** |

## Vì sao run A thiếu provenance

`f564570e` chạy trước khi `command.json` bắt đầu ghi knob thật đã tới evaluator. Trường
`environment.evaluator_environment_knobs` là `null`, nên **không có bằng chứng trong artifact** rằng
`WORKBENCH_CSMCIR_QUERY_BS=8` / `WORKBENCH_CSMCIR_SAVE_MEMORY=1` / `PYTORCH_CUDA_ALLOC_CONF` đã được
đặt cho lần chạy đó. Đó là khác biệt duy nhất giữa hai lần chạy, và nó là khác biệt về **ghi chép**,
không phải về phép tính: `argv`, `cwd`, `command_digest` (`7aa7a6db…`), pin commit và
`checkpoint_local_sha256` giống nhau, và số đo trùng đến chữ số cuối.

Chỉ **run B (`f03887d2`) là run đủ provenance**. Mọi trích dẫn "hai lần chạy khớp bit" phải nêu rõ
điều này.

## Trong mỗi thư mục con

| file | nguồn |
|---|---|
| `command.json` | `artifacts/logs/<run>/command.json`, **bản sao byte-for-byte** |
| `invocation_outcome.json` | `artifacts/logs/<run>/invocation_outcome.json`, bản sao byte-for-byte |
| `aggregate_report.json` | `artifacts/reports/<run>_aggregate.json`, bản sao byte-for-byte |

`stdout.log` / `stderr.log` của run B nằm nguyên văn trong `../full_eval_pipeline.txt` (có nhãn nguồn
và sha256 từng file). Chúng không được chép lại ở đây để tránh hai bản có thể lệch nhau.

## Bốn run exit 1 (không dùng làm bằng chứng số đo)

`c340acd3` (05:39:16Z), `2d0369f2` (05:40:51Z), `a24ea590` (2026-11-06T19:23:13Z) — `return_code: 1`.
Thêm `6448321b` (04:25:14Z) và `985f54ad` (04:41:39Z) không có `command.json` (chết trước khi
provenance được ghi). Không run nào trong số này có metric JSON trong stdout.
