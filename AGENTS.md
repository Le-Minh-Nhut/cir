# AGENTS.md — quy ước vận hành repo này

## Chạy CPU: dùng hết máy, trừ khi có lý do

Máy này **20 core / 20 thread** (Intel Ultra 7 265KF). Mặc định **chạy full công suất** cho mọi việc
tốn CPU: test suite, build, script quét dữ liệu, tiền xử lý. Đừng để mặc định 1 luồng.

Ngoại lệ duy nhất: khi hai job cùng giành một tài nguyên độc quyền (GPU, một checkpoint, một port,
một thư mục output) thì chạy **tuần tự**, vì song song sẽ hỏng kết quả. Tài nguyên chia sẻ được thì
song song.

### Test suite

Interpreter bắt buộc: `workbench/envs/ctl312h/bin/python` (CPython 3.12.11, có `Python.h`).
`.venv` trỏ tới nó. Python hệ thống thiếu header → thêm ~41 failure giả.

```bash
# song song, mặc định
workbench/envs/ctl312h/bin/python -m pytest workbench/tests -q -p no:cacheprovider -n 20

# serial — dùng khi cần số liệu chuẩn để ghi vào tài liệu release, hoặc khi nghi flaky
workbench/envs/ctl312h/bin/python -m pytest workbench/tests -q -p no:cacheprovider
```

`pytest-xdist` đã cài trong env đó. Đo trên máy này (2243 test): serial **26:18**, `-n 20` **5:57** —
nhanh 4.4×, cùng kết quả 2195 passed / 1 failed / 47 skipped. Con số **được trích vào tài liệu release
phải lấy từ lượt serial** (`PC_PILOT_HANDOFF.md`, `FINAL_PC_RELEASE_VERDICT.md`, `tests/SUMMARY.json`) —
song song dùng để lặp nhanh trong lúc sửa, không phải nguồn số.

Một số test đụng filesystem/thư mục tạm dùng chung; nếu `-n 20` đỏ mà `serial` xanh, đó là vấn đề
isolation của test, phải sửa test chứ không hạ số worker.

## GPU: tuần tự, một model một lúc

Không bao giờ chạy hai job GPU đồng thời. Huỷ job GPU phải kill **cả cây process** — huỷ shell wrapper
để lại evaluator sống (đã gặp: giữ 5.68 GiB hơn 1 giờ).

## Đĩa: dọn scratch, đừng để đầy

`/tmp` từng đầy tới **100%** (174G, riêng `/tmp/pytest-of-heheboiz` 117G) và làm suite đỏ với
`OSError: [Errno 28] No space left on device` — đọc như lỗi test nhưng là hết đĩa. Sau khi dọn còn 174G.

- Test xdist dựng `/tmp/pytest-of-<user>/`; xoá định kỳ sau khi đo xong.
- Worktree tạm (`/tmp/wt*`, `/tmp/basewt`, `/tmp/rev_*`, `/tmp/revchk`) là scratch, xoá ngay khi xong.
- Nếu suite đỏ bất thường với `Errno 28`, chạy `df -h /` trước khi debug test.

## Shell trong session này

Output hay bị rút gọn/lẫn. Ghi ra file dưới `/tmp` rồi đọc lại, hoặc in từ trong Python. Đừng tin
output dài đọc trực tiếp.

## Không được làm

- Không `git reset --hard`, `git clean -fdx`, stash việc của người dùng.
- Không `sudo`, không đổi driver/kernel/CUDA, không reboot, không reset GPU.
- Không sửa `registry/*.yaml` hay thay đổi ngữ nghĩa đánh giá khi chưa được duyệt.
- Không commit checkpoint, dataset, cache, secret.
