#!/usr/bin/env bash
# CIR Failure Analysis Workbench — One-Command PC Entrypoint
# Orchestrates preflight, configuration, Docker stack, model preparation,
# GPU runtime verification, evaluation, validation, and Vietnamese summary.
set -euo pipefail

say() { printf '%s\n' "$*"; }

trim() {
    local var="$*"
    var="${var#"${var%%[![:space:]]*}"}"
    var="${var%"${var##*[![:space:]]}"}"
    printf '%s' "$var"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${CIR_REPO_ROOT:-$SCRIPT_DIR}"
WORKBENCH_DOCKER_DIR="${WORKBENCH_DOCKER_DIR:-${REPO_ROOT}/workbench/docker}"
RUN_SH="${WORKBENCH_DOCKER_DIR}/run.sh"

MODEL="limn"
DOCTOR_MODE=0
VERIFY_ONLY=0
FORCE=0
TIMEOUT=3600
NO_GPU=0
DATASET_PATH_OVERRIDE=""
CHECKPOINT_REQUEST=""
SELF_CHECK=0
FINAL_RUN_ID=""
FINAL_CHECKPOINT=""
FINAL_PROTOCOL=""

# Temporary files this run created, removed together on EVERY exit path.
#
# The launcher writes the API bodies and the verifier's verdict to files rather than carrying them
# through argv, so those files now have a lifetime the launcher owns. Removing them where they are
# created is not enough: `fail_stage` and a non-zero `sys.exit` from the summary both leave the shell
# immediately, and the SIGINT/SIGTERM handler below leaves it too -- so each of those paths leaked
# `cir_idx.*` (which holds run metadata), `cir_dedup.*`, `cir_verify_out.*`, `cir_verify_state.*` and
# `cir_verify_msg.*` under a shared temp directory. An EXIT trap covers all of them at once,
# including the paths nobody remembered.
LAUNCHER_TEMP_PATHS=()
_register_temp() {
    LAUNCHER_TEMP_PATHS+=("$1")
}
_cleanup_temp_paths() {
    local path
    for path in "${LAUNCHER_TEMP_PATHS[@]:-}"; do
        [[ -n "$path" ]] && rm -rf -- "$path"
    done
    LAUNCHER_TEMP_PATHS=()
}
# `EXIT` fires for normal return, `exit`, and an uncaught `set -e` abort. The signal handlers exit
# 130, which fires it as well, so no path needs its own cleanup call.
trap '_cleanup_temp_paths' EXIT

# Trap SIGINT/SIGTERM (Behaviour 10)
cleanup_on_signal() {
    local sig="$1"
    say ""
    say "================================================================================"
    say "ĐÃ NHẬN TÍN HIỆU DỪNG (${sig})"
    say "Bảo toàn nguyên vẹn artifacts, nhật ký và trạng thái hệ thống."
    say "Không tự động xoá kết quả hay dữ liệu đã tải."
    say "================================================================================"
    exit 130
}
trap 'cleanup_on_signal SIGINT' SIGINT
trap 'cleanup_on_signal SIGTERM' SIGTERM

# Behaviour 9 (Failure summary)
fail_stage() {
    local stage="$1"
    local error_msg="$2"
    local log_location="${3:-N/A}"
    local remedy="${4:-Xem hướng dẫn trong tài liệu PC_ONE_COMMAND_GUIDE.vi.md}"

    say ""
    say "================================================================================"
    say "LỖI THỰC THI — CIR FAILURE ANALYSIS WORKBENCH"
    say "================================================================================"
    say "Giai đoạn thất bại (Failed stage) : ${stage}"
    say "Chi tiết lỗi (Error details)      : ${error_msg}"
    say "Vị trí nhật ký (Log location)     : ${log_location}"
    say "Hành động khắc phục (Remedy)      : ${remedy}"
    say "================================================================================"
    exit 1
}

# Strict .env parsing (mirrors run.sh:32-54, preserves quotes and comments)
env_value() { # env_value KEY DEFAULT [FILE]
    local key="$1" default="$2" file="${3:-${WORKBENCH_DOCKER_DIR}/.env}" line k v
    if [[ -n "${!key:-}" ]]; then
        printf '%s\n' "${!key}"; return 0
    fi
    if [[ -f "$file" ]]; then
        while IFS= read -r line || [[ -n "$line" ]]; do
            # Ignore comments and non-assignments
            [[ "$line" =~ ^[[:space:]]*# ]] && continue
            [[ "$line" != *=* ]] && continue
            k="${line%%=*}"
            k="$(trim "$k")"
            if [[ "$k" == "$key" ]]; then
                v="${line#*=}"
                v="${v%$'\r'}"
                local first_char="${v:0:1}"
                if [[ "$first_char" == '"' ]]; then
                    local stripped="${v:1}"
                    v="${stripped%%\"*}"
                elif [[ "$first_char" == "'" ]]; then
                    local stripped="${v:1}"
                    v="${stripped%%\'*}"
                else
                    v="${v%%#*}"
                    v="$(trim "$v")"
                fi
                printf '%s\n' "$v"; return 0
            fi
        done < "$file"
    fi
    printf '%s\n' "$default"
}

validate_env() { # validate_env KEY VALUE
    case "$2" in
        ''|*[!0-9]*) fail_stage "CONFIG" "invalid ${1}=${2}: must be an integer port" "${WORKBENCH_DOCKER_DIR}/.env" "Cấu hình cổng hợp lệ từ 1 đến 65535";;
    esac
    if (( $2 < 1 || $2 > 65535 )); then
        fail_stage "CONFIG" "invalid ${1}=${2}: port out of range" "${WORKBENCH_DOCKER_DIR}/.env" "Cấu hình cổng hợp lệ từ 1 đến 65535"
    fi
}

validate_env_file() {
    local file="${WORKBENCH_DOCKER_DIR}/.env"
    if [[ -f "$file" ]]; then
        while IFS= read -r line || [[ -n "$line" ]]; do
            [[ "$line" =~ ^[[:space:]]*# ]] && continue
            [[ "$line" != *=* ]] && continue
            local k="${line%%=*}"
            local v="${line#*=}"
            k="$(trim "$k")"
            v="${v%$'\r'}"
            v="$(trim "$v")"
            # Comment first, then one layer of matching quotes -- the order compose and
            # `env_value` use. Checking the quotes first is what rejected a perfectly well-formed
            # .env: `PORT="8000" # default` kept its quotes (the closing one was no longer last),
            # and `validate_env` then refused `"8000"` as a non-integer.
            #
            # Validating the FILE's value directly, rather than asking `env_value`, is deliberate:
            # `env_value` prefers the process environment, so a malformed .env would be silently
            # bypassed by an exported variable and the file would never actually be validated.
            if [[ "$v" == \"* ]]; then
                local stripped="${v:1}"
                v="${stripped%%\"*}"
            elif [[ "$v" == \'* ]]; then
                local stripped="${v:1}"
                v="${stripped%%\'*}"
            else
                v="${v%%#*}"
                v="$(trim "$v")"
            fi
            if [[ "$k" == "WORKBENCH_BACKEND_PORT" || "$k" == "WORKBENCH_FRONTEND_PORT" ]]; then
                validate_env "$k" "$v"
            fi
        done < "$file"
    fi
}

resolve_path() {
    local base_dir="$1" p="$2"
    if [[ "$p" = /* ]]; then
        printf '%s\n' "$p"
    else
        printf '%s\n' "$(cd "$base_dir" 2>/dev/null && cd "$p" 2>/dev/null && pwd || printf '%s/%s' "$base_dir" "$p")"
    fi
}

# Sanitizes execution environment to avoid leaking credentials/secrets to child processes
sanitize_exec() {
    local unset_flags=()
    local var
    for var in $(env | cut -d= -f1 | grep -E '(TOKEN|KEY|SECRET|LEDGER)' || true); do
        unset_flags+=("-u" "$var")
    done
    if (( ${#unset_flags[@]} > 0 )); then
        env "${unset_flags[@]}" "$@"
    else
        "$@"
    fi
}

show_help() {
    cat <<'EOF'
Usage: ./pc.sh [OPTIONS]

One-command PC entrypoint for the CIR Failure Analysis Workbench.
Orchestrates preflight, configuration, Docker stack, model preparation,
GPU runtime verification, evaluation, validation, and Vietnamese summary.

Options:
  --model <id>       Model to evaluate (default: limn; e.g. csmcir)
  --doctor           Run host and runtime preflight checks only (read-only)
  --verify-only      Verify the model's GPU RUNTIME inside the container and stop. It does NOT
                     verify a recorded run, does NOT rebuild the DuckDB index, and does NOT
                     download checkpoints. To verify an existing run and refresh the index, run
                     ./pc.sh without this flag: it reuses a corroborated run instead of
                     re-evaluating.
  --force            Force re-evaluation even if completed corroborated run exists
  --timeout <sec>    Maximum wait time for evaluation job in seconds (default: 3600)
  --no-gpu           Disable GPU override (WORKBENCH_ENABLE_GPU=0)
  --dataset <path>   Override FashionIQ dataset path
  --checkpoint <id>  Pin one exact checkpoint or bundle id. Use to run an INTENTIONAL
                     single-category evaluation; omit it to run the model's declared bundle,
                     which refuses by name when any member is missing rather than silently
                     narrowing itself to one category.
  --self-check       Run internal self-tests for helper functions and exit
  -h, --help         Show this help message and exit
EOF
}

# CLI Argument Parsing
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            [[ $# -ge 2 ]] || fail_stage "CLI_PARSING" "Thiếu giá trị cho --model" "" "Truyền tên model, ví dụ: --model limn"
            MODEL="$2"; shift 2;;
        --doctor)
            DOCTOR_MODE=1; shift;;
        --verify-only)
            VERIFY_ONLY=1; shift;;
        --force)
            FORCE=1; shift;;
        --timeout)
            [[ $# -ge 2 ]] || fail_stage "CLI_PARSING" "Thiếu giá trị cho --timeout" "" "Truyền số giây, ví dụ: --timeout 3600"
            TIMEOUT="$2"; shift 2;;
        --no-gpu)
            NO_GPU=1; shift;;
        --dataset)
            [[ $# -ge 2 ]] || fail_stage "CLI_PARSING" "Thiếu giá trị cho --dataset" "" "Truyền đường dẫn FashionIQ, ví dụ: --dataset /path/to/FashionIQ"
            DATASET_PATH_OVERRIDE="$2"; shift 2;;
        --checkpoint)
            [[ $# -ge 2 ]] || fail_stage "CLI_PARSING" "Thiếu giá trị cho --checkpoint" "" "Truyền checkpoint id, ví dụ: --checkpoint base_iter0_all_categories"
            CHECKPOINT_REQUEST="$2"; shift 2;;
        --self-check)
            SELF_CHECK=1; shift;;
        -h|--help)
            show_help; exit 0;;
        *)
            fail_stage "CLI_PARSING" "Tùy chọn không hợp lệ: $1" "" "Xem hướng dẫn: ./pc.sh --help";;
    esac
done

# Internal Self-Check
if [[ "$SELF_CHECK" == "1" ]]; then
    tmp_env="$(mktemp)"
    printf 'FOO=bar\n# comment\nBAZ="qux # inside quotes"\nSINGLE='\''single # inside'\''\nPORT=8000\n' > "$tmp_env"
    val_foo="$(env_value FOO "" "$tmp_env")"
    [[ "$val_foo" == "bar" ]] || { echo "self-check fail: FOO"; rm -f "$tmp_env"; exit 1; }
    val_baz="$(env_value BAZ "" "$tmp_env")"
    [[ "$val_baz" == "qux # inside quotes" ]] || { echo "self-check fail: BAZ"; rm -f "$tmp_env"; exit 1; }
    val_single="$(env_value SINGLE "" "$tmp_env")"
    [[ "$val_single" == "single # inside" ]] || { echo "self-check fail: SINGLE"; rm -f "$tmp_env"; exit 1; }
    val_default="$(env_value NOT_FOUND "def" "$tmp_env")"
    [[ "$val_default" == "def" ]] || { echo "self-check fail: NOT_FOUND"; rm -f "$tmp_env"; exit 1; }
    rm -f "$tmp_env"

    resolved="$(resolve_path "/home" "sub/path")"
    [[ "$resolved" == *"/sub/path" ]] || { echo "self-check fail: resolve_path relative"; exit 1; }
    resolved_abs="$(resolve_path "/home" "/etc/passwd")"
    [[ "$resolved_abs" == "/etc/passwd" ]] || { echo "self-check fail: resolve_path absolute"; exit 1; }

    say "pc.sh self-check passed."
    exit 0
fi

# Behaviour 1: Preflight (Read-only when is_doctor == 1)
run_preflight() {
    local is_doctor="$1"
    local has_errors=0

    say "================================================================================"
    if (( is_doctor == 1 )); then
        say "CIR FAILURE ANALYSIS WORKBENCH — BÁO CÁO TIỀN KIỂM (PREFLIGHT REPORT)"
        say "Chế độ: --doctor (chỉ đọc, không can thiệp hệ thống, không bật container)"
    else
        say "CIR FAILURE ANALYSIS WORKBENCH — TIỀN KIỂM HỆ THỐNG (PREFLIGHT)"
    fi
    say "================================================================================"

    # 1. Linux Distro & Kernel
    if [[ -f /etc/os-release ]]; then
        local os_pretty
        os_pretty="$(grep -E '^PRETTY_NAME=' /etc/os-release | cut -d= -f2- | tr -d '"' || uname -s)"
        say "[OK] Hệ điều hành (OS)      : ${os_pretty} (kernel $(uname -r))"
    else
        say "[CẢNH BÁO] Không tìm thấy /etc/os-release (kernel $(uname -r))"
    fi

    # 2. Git Revision
    if command -v git >/dev/null 2>&1 && git -C "$REPO_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        local git_branch git_rev
        git_branch="$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
        git_rev="$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
        say "[OK] Git Repository         : branch ${git_branch}, commit ${git_rev}"
    else
        say "[CẢNH BÁO] Không kiểm tra được Git repository tại ${REPO_ROOT}"
    fi

    # 3. RAM
    if [[ -f /proc/meminfo ]]; then
        local mem_kb mem_gb mem_avail_kb mem_avail_gb
        mem_kb="$(awk '/MemTotal:/ {print $2}' /proc/meminfo)"
        mem_avail_kb="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo 2>/dev/null || echo '0')"
        mem_gb=$(( mem_kb / 1048576 ))
        mem_avail_gb=$(( mem_avail_kb / 1048576 ))
        if (( mem_gb >= 16 )); then
            say "[OK] Bộ nhớ RAM             : ${mem_gb} GB (khả dụng: ${mem_avail_gb} GB)"
        elif (( mem_gb >= 8 )); then
            say "[OK] Bộ nhớ RAM             : ${mem_gb} GB (khả dụng: ${mem_avail_gb} GB) — khuyến nghị >= 16 GB"
        else
            say "[FAIL] Bộ nhớ RAM           : ${mem_gb} GB (thấp hơn mức tối thiểu 8 GB)"
            say "       Khắc phục            : Nâng cấp RAM hệ thống (khuyến nghị 16 GB cho pipeline)"
            has_errors=1
        fi
    else
        say "[CẢNH BÁO] Không đọc được /proc/meminfo"
    fi

    # 4. Free Disk (Robust parsing for trailing G / M / units)
    local free_disk_raw free_disk_gb
    free_disk_raw="$(df -PB 1G "$REPO_ROOT" 2>/dev/null | awk 'NR==2 {print $4}' || true)"
    if [[ -z "$free_disk_raw" ]]; then
        free_disk_raw="$(df "$REPO_ROOT" 2>/dev/null | awk 'NR==2 {print $4}' || echo '0')"
    fi
    free_disk_gb="$(printf '%s' "$free_disk_raw" | sed 's/[^0-9]//g')"
    free_disk_gb="${free_disk_gb:-0}"
    # LIMN alone needs ~5.4 GB for its three checkpoints plus environment and cache headroom;
    # a run that passes preflight and then dies of ENOSPC mid-download wastes hours. The floor
    # is therefore the real working minimum, not an arbitrary small number.
    if (( free_disk_gb >= 50 )); then
        say "[OK] Dung lượng đĩa trống  : ${free_disk_gb} GB khả dụng tại ${REPO_ROOT}"
    elif (( free_disk_gb >= 25 )); then
        say "[OK] Dung lượng đĩa trống  : ${free_disk_gb} GB khả dụng — đủ cho LIMN; khuyến nghị >= 50 GB cho model khác và cache"
    else
        say "[FAIL] Dung lượng đĩa trống  : ${free_disk_gb} GB (thấp hơn mức tối thiểu 25 GB)"
        say "       Khắc phục            : giải phóng dung lượng (docker system prune; xoá checkpoint không dùng) rồi chạy lại ./pc.sh"
        has_errors=1
    fi

    # 5. Required Host Dirs & Ownership
    local cur_uid cur_user
    if command -v id >/dev/null 2>&1; then
        cur_uid="$(id -u)"
        cur_user="$(id -un)"
    else
        cur_uid="${UID:-1000}"
        cur_user="${USER:-${LOGNAME:-operator}}"
    fi

    local req_dirs=("${REPO_ROOT}/data" "${REPO_ROOT}/workbench/artifacts" "${REPO_ROOT}/workbench/third_party")
    local dirs_ok=1
    for d in "${req_dirs[@]}"; do
        if (( is_doctor == 0 )) && [[ ! -d "$d" ]]; then
            mkdir -p "$d" 2>/dev/null || true
        fi
        if [[ -d "$d" ]]; then
            local dir_uid="$cur_uid"
            if command -v stat >/dev/null 2>&1; then
                dir_uid="$(stat -c '%u' "$d" 2>/dev/null || echo "$cur_uid")"
            fi
            if [[ "$dir_uid" != "$cur_uid" ]]; then
                say "[FAIL] Quyền sở hữu thư mục  : ${d} thuộc uid ${dir_uid}, không phải user ${cur_user} (${cur_uid})"
                say "       Khắc phục            : sudo chown -R ${cur_user}:${cur_user} ${d}"
                dirs_ok=0
                has_errors=1
            fi
        fi
    done
    if (( dirs_ok == 1 )); then
        say "[OK] Thư mục & Quyền sở hữu : Đầy đủ (uid: ${cur_uid}, user: ${cur_user})"
    fi

    # 6. Docker Binary & Daemon
    if ! command -v docker >/dev/null 2>&1; then
        say "[FAIL] Docker Engine        : Lệnh docker không tồn tại trên PATH"
        say "       Khắc phục            : sudo apt-get update && sudo apt-get install -y docker.io"
        return 1
    else
        local docker_ver
        if docker_ver="$(docker version --format '{{.Server.Version}}' 2>/dev/null)"; then
            say "[OK] Docker Engine          : phiên bản ${docker_ver}, daemon phản hồi tốt"
        else
            say "[FAIL] Docker Engine        : daemon không phản hồi hoặc không có quyền truy cập socket"
            say "       Khắc phục            : sudo systemctl start docker && sudo usermod -aG docker ${cur_user}"
            return 1
        fi
    fi

    # 7. Docker Compose Plugin
    if docker compose version >/dev/null 2>&1; then
        local compose_ver
        compose_ver="$(docker compose version --short 2>/dev/null || echo 'v2')"
        say "[OK] Docker Compose         : plugin phiên bản ${compose_ver}"
    else
        say "[FAIL] Docker Compose       : Chưa cài plugin docker compose v2"
        say "       Khắc phục            : sudo apt-get update && sudo apt-get install -y docker-compose-v2"
        has_errors=1
    fi

    # 8. NVIDIA Driver & GPU Hardware
    if (( NO_GPU == 1 )); then
        say "[SKIP] NVIDIA Driver & GPU  : Bỏ qua do cờ --no-gpu được kích hoạt"
    else
        if command -v nvidia-smi >/dev/null 2>&1; then
            local smi_info
            if smi_info="$(nvidia-smi --query-gpu=driver_version,name,compute_cap --format=csv,noheader,nounits 2>/dev/null)"; then
                local driver_ver gpu_name compute_cap
                driver_ver="$(echo "$smi_info" | cut -d',' -f1 | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
                gpu_name="$(echo "$smi_info" | cut -d',' -f2 | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
                compute_cap="$(echo "$smi_info" | cut -d',' -f3 | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
                say "[OK] NVIDIA Driver & GPU    : Driver ${driver_ver} | Thiết bị: ${gpu_name} (Compute ${compute_cap})"
                if [[ "$compute_cap" == "12.0" || "$compute_cap" == "12."* ]]; then
                    say "     Kiến trúc              : Blackwell sm_120 (RTX 5070 Ti) được nhận diện chính xác"
                fi
            else
                if (( is_doctor == 1 )); then
                    say "[PENDING] NVIDIA Driver     : nvidia-smi gặp lỗi khi kết nối driver"
                    say "          Khắc phục         : Cài đặt nvidia-driver tương thích và khởi động lại máy"
                else
                    say "[FAIL] NVIDIA Driver        : nvidia-smi không thể giao tiếp với driver NVIDIA"
                    say "       Khắc phục            : Cài đặt nvidia-driver tương thích và khởi động lại máy"
                    has_errors=1
                fi
            fi
        else
            if (( is_doctor == 1 )); then
                say "[PENDING] NVIDIA Driver     : Lệnh nvidia-smi không tìm thấy trên PATH"
                say "          Khắc phục         : sudo apt-get install -y nvidia-driver-580 và reboot"
            else
                say "[FAIL] NVIDIA Driver        : Thiếu driver NVIDIA trên host"
                say "       Khắc phục            : sudo apt-get install -y nvidia-driver-580 và reboot"
                has_errors=1
            fi
        fi
    fi

    # 9. NVIDIA Container Toolkit in Docker
    if (( NO_GPU == 1 )); then
        say "[SKIP] NVIDIA Container RT  : Bỏ qua do cờ --no-gpu được kích hoạt"
    else
        local has_nvidia_rt=0
        # The runtime name must appear in the registered-runtimes object. Matching the word
        # "nvidia" anywhere in `docker info` also matches the GPU model, the driver version and
        # unrelated plugin strings, so an operator without the toolkit would be told the runtime
        # is configured.
        if docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"'; then
            has_nvidia_rt=1
        fi
        if (( has_nvidia_rt == 1 )); then
            say "[OK] NVIDIA Container Runtime: Đã cấu hình và đăng ký trong Docker daemon"
        else
            if (( is_doctor == 1 )); then
                say "[PENDING] NVIDIA Container Runtime: Chưa phát hiện runtime 'nvidia' trong Docker"
                say "          Khắc phục         : Cài đặt nvidia-container-toolkit (sudo apt-get install -y nvidia-container-toolkit)"
                say "                              sudo nvidia-ctk runtime configure --runtime=docker"
                say "                              sudo systemctl restart docker"
            else
                say "[FAIL] NVIDIA Container Runtime: Chưa cấu hình nvidia-container-toolkit"
                say "       Khắc phục            : sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit"
                say "                              sudo nvidia-ctk runtime configure --runtime=docker"
                say "                              sudo systemctl restart docker"
                if [[ -t 0 ]]; then
                    read -r -p "Bạn có muốn cài đặt nvidia-container-toolkit ngay bây giờ? [y/N]: " yn_install || true
                    case "$yn_install" in
                        [yY]|[yY][eE][sS])
                            say "Đang chạy cài đặt nvidia-container-toolkit..."
                            sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
                            sudo nvidia-ctk runtime configure --runtime=docker
                            sudo systemctl restart docker
                            if docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"'; then
                                say "[OK] Cài đặt và cấu hình nvidia-container-toolkit thành công!"
                                has_nvidia_rt=1
                            else
                                say "[FAIL] Cài đặt xong nhưng Docker daemon vẫn chưa nhận runtime 'nvidia'."
                                has_errors=1
                            fi
                            ;;
                        *)
                            say "Đã bỏ qua cài đặt tự động."
                            has_errors=1
                            ;;
                    esac
                else
                    has_errors=1
                fi
            fi
        fi
    fi

    # 10. Dataset Verification (Behaviour 3)
    local raw_ds_path="${DATASET_PATH_OVERRIDE:-$(env_value HOST_FASHIONIQ_ROOT ../../data/FashionIQ)}"
    local resolved_ds
    resolved_ds="$(resolve_path "${WORKBENCH_DOCKER_DIR}" "$raw_ds_path")"
    if [[ -d "${resolved_ds}/images" && -d "${resolved_ds}/image_splits" && -d "${resolved_ds}/captions" ]]; then
        say "[OK] Dataset FashionIQ      : Tồn tại và đầy đủ (images/, image_splits/, captions/) tại ${resolved_ds}"
    else
        if (( is_doctor == 1 )); then
            say "[CẢNH BÁO] Dataset FashionIQ: Chưa tìm thấy hoặc thiếu thư mục con tại ${resolved_ds}"
            say "          Khắc phục         : Thiết lập HOST_FASHIONIQ_ROOT trong .env hoặc dùng cờ --dataset <đường_dẫn>"
        else
            if [[ -t 0 ]]; then
                say "[CẢNH BÁO] Không tìm thấy dataset FashionIQ tại ${resolved_ds}"
                read -r -p "Nhập đường dẫn tuyệt đối tới FashionIQ dataset: " user_ds || true
                if [[ -n "$user_ds" && -d "${user_ds}/images" && -d "${user_ds}/image_splits" && -d "${user_ds}/captions" ]]; then
                    DATASET_PATH_OVERRIDE="$user_ds"
                    resolved_ds="$user_ds"
                    say "[OK] Đã xác nhận dataset FashionIQ tại ${resolved_ds}"
                    if [[ -f "${WORKBENCH_DOCKER_DIR}/.env" ]]; then
                        if grep -q '^HOST_FASHIONIQ_ROOT=' "${WORKBENCH_DOCKER_DIR}/.env"; then
                            sed -i "s|^HOST_FASHIONIQ_ROOT=.*|HOST_FASHIONIQ_ROOT=${user_ds}|" "${WORKBENCH_DOCKER_DIR}/.env"
                        else
                            printf 'HOST_FASHIONIQ_ROOT=%s\n' "${user_ds}" >> "${WORKBENCH_DOCKER_DIR}/.env"
                        fi
                    fi
                else
                    say "[FAIL] Đường dẫn nhập vào không hợp lệ hoặc thiếu dữ liệu FashionIQ"
                    has_errors=1
                fi
            else
                say "[FAIL] Không tìm thấy dataset FashionIQ tại ${resolved_ds}"
                say "       Khắc phục            : Thiết lập HOST_FASHIONIQ_ROOT trong workbench/docker/.env hoặc dùng --dataset <path>"
                has_errors=1
            fi
        fi
    fi

    # 11. Container Status Check (Read-only)
    if command -v docker >/dev/null 2>&1; then
        local c_id
        c_id="$(docker ps -q --filter name=^cir-workbench$ 2>/dev/null | head -n1 || true)"
        if [[ -n "$c_id" ]]; then
            say "[INFO] Container cir-workbench: Đang chạy (Container ID: ${c_id})"
        else
            if (( is_doctor == 1 )); then
                say "[INFO] Container cir-workbench: Chưa khởi chạy (ở chế độ --doctor, không tự ý bật container)"
            fi
        fi
    fi

    say "================================================================================"
    if (( is_doctor == 1 )); then
        say "BÁO CÁO TIỀN KIỂM HOÀN TẤT (READ-ONLY)"
        say "================================================================================"
        return 0
    fi

    if (( has_errors != 0 )); then
        fail_stage "PREFLIGHT" "Các điều kiện tiên quyết không thoả mãn (xem chi tiết [FAIL] phía trên)" "" "Cài đặt các gói phần mềm còn thiếu theo hướng dẫn trước khi tiếp tục"
    fi
}

# Behaviour 2: Config
configure_environment() {
    say "Cấu hình môi trường workbench/docker/.env..."
    local created_new_env=0
    if [[ ! -f "${WORKBENCH_DOCKER_DIR}/.env" ]]; then
        if [[ -f "${WORKBENCH_DOCKER_DIR}/.env.example" ]]; then
            say "Khởi tạo workbench/docker/.env từ .env.example..."
            cp "${WORKBENCH_DOCKER_DIR}/.env.example" "${WORKBENCH_DOCKER_DIR}/.env"
            created_new_env=1
        else
            fail_stage "CONFIG" "Thiếu file template ${WORKBENCH_DOCKER_DIR}/.env.example" "${WORKBENCH_DOCKER_DIR}" "Khôi phục lại .env.example từ repository"
        fi
    fi

    local target_adaptation=""
    target_adaptation="$(python3 -c "
import sys
model = sys.argv[1]
models_yaml = sys.argv[2]
try:
    import yaml
    with open(models_yaml, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    for m in data.get('models', []):
        if m.get('model_id') == model:
            env = m.get('environment') or {}
            adapted = env.get('adapted_runtime') or {}
            print(adapted.get('target_compute_capability') or '')
            sys.exit(0)
except Exception:
    pass
if model == 'limn':
    print('sm_120')
else:
    print('')
" "$MODEL" "${REPO_ROOT}/workbench/registry/models.yaml" 2>/dev/null || echo "sm_120")"

    if (( NO_GPU == 1 )); then
        export WORKBENCH_ENABLE_GPU="0"
    else
        export WORKBENCH_ENABLE_GPU="1"
    fi

    if [[ -n "$target_adaptation" ]]; then
        export WORKBENCH_RUNTIME_ADAPTATION="$target_adaptation"
        say "Kích hoạt thích nghi runtime cho ${MODEL}: WORKBENCH_RUNTIME_ADAPTATION=${target_adaptation}"
    else
        export WORKBENCH_RUNTIME_ADAPTATION=""
        say "Mô hình ${MODEL} không khai báo adapted_runtime: giữ nguyên runtime chuẩn"
    fi

    if (( created_new_env == 1 )); then
        if grep -q '^# *WORKBENCH_ENABLE_GPU=' "${WORKBENCH_DOCKER_DIR}/.env"; then
            sed -i "s/^# *WORKBENCH_ENABLE_GPU=.*/WORKBENCH_ENABLE_GPU=${WORKBENCH_ENABLE_GPU}/" "${WORKBENCH_DOCKER_DIR}/.env"
        elif ! grep -q '^WORKBENCH_ENABLE_GPU=' "${WORKBENCH_DOCKER_DIR}/.env"; then
            printf 'WORKBENCH_ENABLE_GPU=%s\n' "${WORKBENCH_ENABLE_GPU}" >> "${WORKBENCH_DOCKER_DIR}/.env"
        fi

        if [[ -n "$target_adaptation" ]]; then
            if grep -q '^# *WORKBENCH_RUNTIME_ADAPTATION=' "${WORKBENCH_DOCKER_DIR}/.env"; then
                sed -i "s/^# *WORKBENCH_RUNTIME_ADAPTATION=.*/WORKBENCH_RUNTIME_ADAPTATION=${target_adaptation}/" "${WORKBENCH_DOCKER_DIR}/.env"
            elif ! grep -q '^WORKBENCH_RUNTIME_ADAPTATION=' "${WORKBENCH_DOCKER_DIR}/.env"; then
                printf 'WORKBENCH_RUNTIME_ADAPTATION=%s\n' "${target_adaptation}" >> "${WORKBENCH_DOCKER_DIR}/.env"
            fi
        fi
    fi

    if [[ -n "$DATASET_PATH_OVERRIDE" ]]; then
        export HOST_FASHIONIQ_ROOT="$DATASET_PATH_OVERRIDE"
    fi

    if command -v id >/dev/null 2>&1; then
        export WORKBENCH_UID="$(id -u)"
        export WORKBENCH_GID="$(id -g)"
    else
        export WORKBENCH_UID="${UID:-1000}"
        export WORKBENCH_GID="1000"
    fi

    say "Môi trường đã sẵn sàng: WORKBENCH_ENABLE_GPU=${WORKBENCH_ENABLE_GPU}, MODEL=${MODEL}"
}

# The single source of truth for "does this container's effective configuration satisfy this
# run?". The inline check that used to live in the reuse decision and the correction path are
# the same check, so a restart is now followed by exactly the same inspection that decided it
# was needed: a restart that does not correct the configuration can no longer be followed by
# preparation and evaluation on a container that still has the wrong mount.
inspect_container_config() {
    local base="$1"
    inspect_config_satisfies=1
    inspect_mismatch_reasons=()
    inspect_mounts_known=1
    inspect_mounts_desc="ok"
    inspect_gpu_mismatch=0
    inspect_adaptation_mismatch=0
    inspect_mount_mismatch=0
    inspect_env_mismatch=0

    # 1. GPU override / in-container CUDA visibility
    if (( WORKBENCH_ENABLE_GPU == 1 )); then
        local gpu_ok=0
        local dev_req
        dev_req="$(docker inspect --format '{{json .HostConfig.DeviceRequests}}' cir-workbench 2>/dev/null || true)"
        if [[ "$dev_req" == *"gpu"* || "$dev_req" == *"nvidia"* ]]; then
            gpu_ok=1
        fi
        if docker exec cir-workbench python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" >/dev/null 2>&1; then
            gpu_ok=1
        elif (( gpu_ok == 0 )); then
            if docker exec cir-workbench nvidia-smi >/dev/null 2>&1; then
                gpu_ok=1
            fi
        fi
        if (( gpu_ok == 0 )); then
            inspect_config_satisfies=0
            inspect_gpu_mismatch=1
            inspect_mismatch_reasons+=("Container cir-workbench thiếu GPU passthrough (GPU override không có hiệu lực hoặc không có CUDA visibility)")
        fi
    fi

    # 2. Runtime adaptation value
    local expected_adaptation="${WORKBENCH_RUNTIME_ADAPTATION:-}"
    local cont_adapt
    cont_adapt="$(docker exec cir-workbench python -c "import os; print(os.environ.get('WORKBENCH_RUNTIME_ADAPTATION', ''))" 2>/dev/null || true)"
    if [[ -z "$cont_adapt" ]]; then
        cont_adapt="$(docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' cir-workbench 2>/dev/null | grep '^WORKBENCH_RUNTIME_ADAPTATION=' | head -n1 | cut -d= -f2- || true)"
    fi
    if [[ "$cont_adapt" != "$expected_adaptation" ]]; then
        inspect_config_satisfies=0
        inspect_adaptation_mismatch=1
        inspect_mismatch_reasons+=("Runtime adaptation không khớp (container='${cont_adapt}', yêu cầu='${expected_adaptation}')")
    fi

    # 2b. Image identity: the code inside the container must be THIS checkout's code.
    # `pc.sh` reuses a live container on a configuration match, and configuration says nothing about
    # which commit was baked into the image. After a `git pull` the old image's backend, queue,
    # verifier and evaluator would sign a report naming the new commit -- the report would be about
    # code that did not run (found by an independent audit). `run.sh up` stamps `cir.commit`; an
    # image built without the label, or one built from another commit, is a mismatch.
    local host_commit image_commit
    host_commit="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
    image_commit="$(docker inspect --format '{{index .Config.Labels "cir.commit"}}' cir-workbench 2>/dev/null || echo unknown)"
    if [[ -z "$image_commit" || "$image_commit" == "<no value>" ]]; then
        image_commit="unknown"
    fi
    if [[ "$host_commit" == "unknown" ]]; then
        # No checkout to compare against (a packaged deployment, or git absent). Saying so is the
        # honest outcome: the image identity is unverifiable here, not wrong.
        say "[CẢNH BÁO] Không xác định được commit của checkout hiện tại; bỏ qua kiểm tra danh tính image."
    elif [[ "$image_commit" == "unknown" ]]; then
        inspect_config_satisfies=0
        inspect_mismatch_reasons+=("Không xác định được commit của image đang chạy (nhãn cir.commit trống). Image có thể là code của lần checkout trước; dựng lại bằng 'workbench/docker/run.sh up'.")
    elif [[ "$image_commit" != "$host_commit" ]]; then
        inspect_config_satisfies=0
        inspect_mismatch_reasons+=("Image đang chạy được dựng từ commit ${image_commit:0:12} nhưng checkout hiện tại là ${host_commit:0:12}; lượt chạy sẽ ghi commit mới trong khi backend/queue/verifier chạy code cũ.")
    fi

    # 3. Mounts provenance (P1-3)
    local exp_ds_raw="${DATASET_PATH_OVERRIDE:-$(env_value HOST_FASHIONIQ_ROOT ../../data/FashionIQ)}"
    local expected_ds_mount="$(resolve_path "${WORKBENCH_DOCKER_DIR}" "$exp_ds_raw")"
    local exp_cp_raw="$(env_value HOST_WORKBENCH_CHECKPOINT_ROOT ../../workbench/artifacts/checkpoints)"
    local expected_cp_mount="$(resolve_path "${WORKBENCH_DOCKER_DIR}" "$exp_cp_raw")"
    local exp_art_raw="$(env_value HOST_WORKBENCH_ARTIFACTS_ROOT ../../workbench/artifacts)"
    local expected_art_mount="$(resolve_path "${WORKBENCH_DOCKER_DIR}" "$exp_art_raw")"

    local mounts_check_result
    mounts_check_result="$(docker inspect --format '{{json .Mounts}}' cir-workbench 2>/dev/null | python3 -c "
import json, os, sys

try:
    raw = sys.stdin.read().strip()
    if not raw:
        print('UNKNOWN empty-inspect')
        sys.exit(0)
    data = json.loads(raw)
except Exception:
    print('UNKNOWN unparseable-json')
    sys.exit(0)

if not isinstance(data, list):
    print('UNKNOWN not-a-list')
    sys.exit(0)

mounts = {}
for item in data:
    if isinstance(item, dict):
        dest = item.get('Destination')
        src = item.get('Source')
        if dest and src:
            mounts[dest] = src

expected = {
    '/data/FashionIQ': sys.argv[1],
    '/checkpoints': sys.argv[2],
    '/artifacts': sys.argv[3],
}

mismatches = []
for dest, exp_path in expected.items():
    if dest not in mounts:
        mismatches.append(f'MISSING:{dest}')
    else:
        act_real = os.path.realpath(mounts[dest])
        exp_real = os.path.realpath(exp_path)
        if act_real != exp_real:
            mismatches.append(f'MISMATCH:{dest}:{mounts[dest]}:{exp_path}')

if mismatches:
    print('REJECT ' + '; '.join(mismatches))
else:
    print('OK')
" "$expected_ds_mount" "$expected_cp_mount" "$expected_art_mount" 2>/dev/null || echo "UNKNOWN docker-inspect-failed")"

    if [[ "$mounts_check_result" == UNKNOWN* ]]; then
        inspect_mounts_known=0
        inspect_config_satisfies=0
        inspect_mount_mismatch=1
        inspect_mounts_desc="${mounts_check_result#UNKNOWN }"
        inspect_mismatch_reasons+=("Không thể xác định mount của container cir-workbench từ docker inspect (${inspect_mounts_desc})")
    elif [[ "$mounts_check_result" == REJECT* ]]; then
        inspect_config_satisfies=0
        inspect_mount_mismatch=1
        local mm_list="${mounts_check_result#REJECT }"
        local item
        IFS=';' read -ra mm_arr <<< "$mm_list"
        for item in "${mm_arr[@]}"; do
            item="$(trim "$item")"
            if [[ "$item" == MISSING:* ]]; then
                local missing_dest="${item#MISSING:}"
                inspect_mismatch_reasons+=("Mount ${missing_dest} bị thiếu trong container cir-workbench")
            elif [[ "$item" == MISMATCH:* ]]; then
                local mm_parts="${item#MISMATCH:}"
                local m_dest="${mm_parts%%:*}"
                local m_rest="${mm_parts#*:}"
                local m_act="${m_rest%%:*}"
                local m_exp="${m_rest#*:}"
                inspect_mismatch_reasons+=("Mount ${m_dest} nguồn không khớp (container='${m_act}', yêu cầu='${m_exp}')")
            fi
        done
    fi

    # The dataset must be readable where the evaluator reads it, not merely mounted.
    if ! docker exec cir-workbench test -d /data/FashionIQ/images 2>/dev/null; then
        inspect_config_satisfies=0
        inspect_mount_mismatch=1
        inspect_mismatch_reasons+=("Dataset mount /data/FashionIQ/images không tồn tại hoặc không thể truy cập")
    fi

    # 4. Runtime environment responds
    if ! docker exec cir-workbench python -c "import sys; sys.exit(0)" 2>/dev/null; then
        inspect_config_satisfies=0
        inspect_env_mismatch=1
        inspect_mismatch_reasons+=("Môi trường thực thi trong container không phản hồi")
    fi

}

# Fail closed on a container that still does not satisfy the requested configuration after a
# restart/recreation. Returns non-zero and names the exact remedy for each cause.
require_corrected_configuration() {
    local stage="$1"
    local base="$2"
    inspect_container_config "$base"
    if (( inspect_config_satisfies == 1 )); then
        say "[OK] Cấu hình container cir-workbench đã đúng yêu cầu sau khi khởi động lại."
        return 0
    fi
    local reasons_str
    reasons_str="$(IFS='; '; echo "${inspect_mismatch_reasons[*]}")"

    local remedies=()
    local log_cmd=""
    if (( inspect_gpu_mismatch == 1 )); then
        remedies+=("Cài đặt nvidia-container-toolkit trên host hoặc cấu hình WORKBENCH_ENABLE_GPU trong workbench/docker/.env (hoặc chạy với cờ --no-gpu nếu không dùng GPU)")
        log_cmd="docker inspect --format '{{json .HostConfig.DeviceRequests}}' cir-workbench"
    fi
    if (( inspect_adaptation_mismatch == 1 )); then
        remedies+=("Kiểm tra WORKBENCH_RUNTIME_ADAPTATION trong workbench/docker/.env cho khớp cấu hình yêu cầu")
        [[ -z "$log_cmd" ]] && log_cmd="docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' cir-workbench"
    fi
    if (( inspect_mount_mismatch == 1 )); then
        remedies+=("Sửa các khóa host path trong workbench/docker/.env (HOST_FASHIONIQ_ROOT / HOST_WORKBENCH_CHECKPOINT_ROOT / HOST_WORKBENCH_ARTIFACTS_ROOT) cho khớp yêu cầu của lượt chạy này, rồi chạy lại ./pc.sh")
        [[ -z "$log_cmd" ]] && log_cmd="docker inspect --format '{{json .Mounts}}' cir-workbench"
    fi
    if (( inspect_env_mismatch == 1 )); then
        remedies+=("Kiểm tra trạng thái container và docker daemon")
        [[ -z "$log_cmd" ]] && log_cmd="docker logs cir-workbench"
    fi
    if (( ${#remedies[@]} == 0 )); then
        remedies+=("Kiểm tra cấu hình container trong workbench/docker/.env")
        log_cmd="docker inspect cir-workbench"
    fi
    local remedy_str
    remedy_str="$(IFS='; '; echo "${remedies[*]}")"

    fail_stage "${stage}" "Container cir-workbench vẫn KHÔNG thỏa mãn cấu hình yêu cầu sau khi khởi động lại: ${reasons_str}" "${log_cmd}" "${remedy_str}"
}

# Behaviour 4: Docker Stack Startup & Readiness Polling
start_docker_and_wait_readiness() {
    local backend_port frontend_port base
    backend_port="$(env_value WORKBENCH_BACKEND_PORT 8000)"
    frontend_port="$(env_value WORKBENCH_FRONTEND_PORT 5173)"
    base="http://127.0.0.1:${backend_port}"

    validate_env WORKBENCH_BACKEND_PORT "$backend_port"
    validate_env WORKBENCH_FRONTEND_PORT "$frontend_port"

    local container_running=0
    local c_id
    c_id="$(docker ps -q --filter name=^cir-workbench$ 2>/dev/null | head -n1 || true)"
    if [[ -n "$c_id" ]]; then
        local code
        code="$(curl -s --max-time 5 -o /dev/null -w '%{http_code}' "${base}/api/jobs" 2>/dev/null || true)"
        # A container is running but its API does not answer. Whether it may be restarted is
        # decided by the SAME question the healthy path asks — is a job in flight? — answered
        # here from the persisted queue state, which is a bind mount the host can read. Replacing
        # a container on a non-200 response without that check would tear down a running
        # evaluation; the permissive answer is never taken on an unreadable state.
        local running_jobs_known=1 running_jobs_desc="none"
        if [[ "$code" != "200" ]]; then
            local rj_info
            rj_info="$(curl -s --max-time 5 "${base}/api/jobs" 2>/dev/null | python3 -c "
import json, sys
try:
    jobs = json.load(sys.stdin)
except Exception:
    print('UNKNOWN unreadable'); raise SystemExit(0)
if not isinstance(jobs, list):
    print('UNKNOWN not-a-list'); raise SystemExit(0)
active = [j for j in jobs if isinstance(j, dict) and j.get('status') in {'queued', 'preparing', 'running'}]
print((str(len(active)) + ' ' + ', '.join(f\"{j.get('job_id', 'job')}({j.get('status')})\" for j in active)) if active else '0 none')
")" || rj_info="UNKNOWN curl-failed"
            running_jobs_desc="${rj_info#* }"
            if [[ "${rj_info%% *}" == "UNKNOWN" ]]; then
                # The API is not the only record: the queue persists each job's state under the
                # artifacts root, a bind mount the host can read directly.
                local persisted_active
                persisted_active="$(python3 - "$WORKBENCH_DOCKER_DIR/.env" "$REPO_ROOT" <<'PYEOF' 2>/dev/null || echo "UNKNOWN"
import json, sys
from pathlib import Path
env_file, repo = Path(sys.argv[1]), Path(sys.argv[2])
artifacts = repo / "workbench" / "artifacts"
if env_file.is_file():
    for line in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip().startswith("HOST_WORKBENCH_ARTIFACTS_ROOT="):
            # Comment first (outside quotes), then one layer of matching quotes -- the same order
            # compose and `env_value` use. Stripping quotes first left a trailing comment inside
            # the value, so a commented path resolved to '/path" # comment' and the running-queue
            # probe reported no active job.
            raw = line.split("=", 1)[1]
            raw = raw.split("#", 1)[0].strip()
            if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "\"'":
                raw = raw[1:-1]
            artifacts = Path(raw) if raw.startswith("/") else (env_file.parent / raw).resolve()
            break
queue = artifacts / "queue_jobs"
if not queue.is_dir():
    print("0 none"); raise SystemExit(0)
active = []
for state in sorted(queue.glob("*/queue_state.json")):
    try:
        record = json.loads(state.read_text(encoding="utf-8"))
    except Exception:
        print("UNKNOWN unreadable-state"); raise SystemExit(0)
    if record.get("status") in {"queued", "preparing", "running"}:
        active.append(f"{record.get('job_id', 'job')}({record.get('status')})")
print((str(len(active)) + " " + ", ".join(active)) if active else "0 none")
PYEOF
)"
                if [[ "${persisted_active%% *}" == "UNKNOWN" ]]; then
                    running_jobs_known=0
                else
                    rj_info="$persisted_active"
                    running_jobs_desc="${persisted_active#* }"
                fi
            fi
            if (( running_jobs_known == 1 )) && [[ "${rj_info%% *}" != "0" ]]; then
                fail_stage "CONTAINER_REUSE" "Container cir-workbench đang chạy tác vụ (${running_jobs_desc}) nhưng API backend không trả lời (HTTP ${code}); không khởi động lại container khi đang có tác vụ hoạt động." "docker ps / ${base}/api/jobs" "Chờ tác vụ hoàn tất (workbench/docker/run.sh status) rồi chạy lại ./pc.sh."
            fi
            if (( running_jobs_known == 0 )); then
                fail_stage "CONTAINER_REUSE" "Container cir-workbench đang chạy nhưng KHÔNG xác nhận được nó có tác vụ đang hoạt động hay không (API HTTP ${code}, trạng thái hàng đợi không đọc được); từ chối khởi động lại để bảo toàn thí nghiệm đang chạy." "docker ps / ${base}/api/jobs" "Khôi phục backend rồi chạy lại ./pc.sh; hoặc dừng thủ công bằng 'workbench/docker/run.sh stop' nếu chắc chắn không có tác vụ."
            fi
        fi
        if [[ "$code" == "200" ]]; then
            # P1-5: Reusing a running container on HTTP 200 alone is insufficient.
            # Verify effective configuration before reusing.
            local config_satisfies=1
            local mismatch_reasons=()

            # Active jobs check on running container. An unreadable list must NOT read as "no
            # active job": the decision it guards is whether a running container may be
            # replaced, and guessing "idle" on an API failure restarts a container that may be
            # mid-experiment. Unknown is reported as unknown and handled fail-closed below.
            local active_jobs_info active_count active_desc jobs_known=1
            active_jobs_info="$(curl -s --max-time 5 "${base}/api/jobs" 2>/dev/null | python3 -c "
import json, sys
try:
    jobs = json.load(sys.stdin)
except Exception:
    print('UNKNOWN unreadable')
    raise SystemExit(0)
if not isinstance(jobs, list):
    print('UNKNOWN not-a-list')
    raise SystemExit(0)
active = [j for j in jobs if isinstance(j, dict) and j.get('status') in {'queued', 'preparing', 'running'}]
if active:
    print(len(active), ', '.join(f\"{j.get('job_id', 'job')}({j.get('status')})\" for j in active))
else:
    print('0 none')
")" || active_jobs_info="UNKNOWN curl-failed"
            active_count="${active_jobs_info%% *}"
            active_desc="${active_jobs_info#* }"
            if [[ "$active_count" == "UNKNOWN" ]]; then
                jobs_known=0
                mismatch_reasons+=("Không đọc được danh sách tác vụ từ ${base}/api/jobs (${active_desc}) — không thể xác nhận container đang rảnh")
            fi

            inspect_container_config "$base"
            local config_satisfies=$inspect_config_satisfies
            local mounts_known=$inspect_mounts_known mounts_desc="$inspect_mounts_desc"
            # `mismatch_reasons` is declared once, before the first append above: re-declaring it
            # here would silently discard the reasons already recorded (the unreadable job list),
            # and the fail-closed branch below would then report a failure without the reason that
            # caused it. The inspection's reasons are accumulated into the same array instead.
            if (( ${#inspect_mismatch_reasons[@]} > 0 )); then
                mismatch_reasons+=("${inspect_mismatch_reasons[@]}")
            fi
            if (( config_satisfies == 1 )); then
                say "Docker container cir-workbench đã đang chạy, cấu hình phù hợp và API sẵn sàng (${base})."
                container_running=1
            else
                local reasons_str
                reasons_str="$(IFS='; '; echo "${mismatch_reasons[*]}")"
                say "[CẢNH BÁO] Container cir-workbench đang chạy nhưng cấu hình không thỏa mãn yêu cầu:"
                say "  ${reasons_str}"

                if (( jobs_known == 0 )); then
                    fail_stage "CONTAINER_REUSE" "Container cir-workbench đang chạy nhưng KHÔNG xác nhận được nó có tác vụ đang hoạt động hay không (${active_desc}); cấu hình lại container lúc này có thể phá một thí nghiệm đang chạy. Chi tiết: ${reasons_str}" "docker ps / ${base}/api/jobs" "Khôi phục API backend rồi chạy lại ./pc.sh; hoặc dừng thủ công bằng 'workbench/docker/run.sh stop' nếu chắc chắn không có tác vụ nào."
                fi

                if (( active_count > 0 )); then
                    fail_stage "CONTAINER_REUSE" "Container cir-workbench đang chạy tác vụ (${active_desc}) nhưng cấu hình không thỏa mãn yêu cầu của lượt chạy này (${reasons_str}). Tuyệt đối không dừng hoặc khởi động lại container khi đang có tác vụ hoạt động." "docker ps / api/jobs" "Chờ tác vụ hiện tại hoàn tất hoặc hủy tác vụ trước khi cấu hình lại container."
                fi
                if (( mounts_known == 0 )); then
                    fail_stage "CONTAINER_REUSE" "Container cir-workbench đang chạy nhưng KHÔNG thể xác định cấu hình mount từ docker inspect (${mounts_desc}); từ chối tái sử dụng container để bảo đảm tính toàn vẹn dữ liệu. Chi tiết: ${reasons_str}" "docker inspect --format '{{json .Mounts}}' cir-workbench" "Kiểm tra docker daemon và container cir-workbench"
                fi

                # If container is missing GPU passthrough and the host cannot satisfy it (no nvidia runtime in docker info), fail immediately instead of restarting identically.
                if (( inspect_gpu_mismatch == 1 )); then
                    local host_has_toolkit=0
                    if docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"'; then
                        host_has_toolkit=1
                    fi
                    if (( host_has_toolkit == 0 )); then
                        fail_stage "CONTAINER_REUSE" "Container cir-workbench thiếu GPU passthrough nhưng host chưa cài đặt nvidia-container-toolkit; không thể cấu hình GPU trong container trên host này. Chi tiết: ${reasons_str}" "docker info" "Cài đặt nvidia-container-toolkit (sudo apt-get install -y nvidia-container-toolkit) và khởi động lại docker daemon; hoặc chạy pc.sh với cờ --no-gpu (WORKBENCH_ENABLE_GPU=0) để đánh giá bằng CPU."
                    fi
                fi

                say "Không có tác vụ đang chạy. Khởi chạy lại Docker stack qua run.sh up để cấu hình đúng..."
                "${RUN_SH}" up
                # A restart is requested, never assumed to have corrected anything: the same
                # inspection that refused the container is run again against the new one, and a
                # residual mismatch fails closed here instead of being discovered against the
                # wrong dataset hours into an evaluation.
                require_corrected_configuration "CONTAINER_RECONFIGURE" "$base"
                container_running=1
            fi
        else
            # The container exists but its API does not answer. It is not restarted blindly: the
            # active-job question was already answered above (from the API or from the persisted
            # queue), so reaching here means no job is in flight and no trajectory is at risk.
            # The restart is not trusted either: only a container that then satisfies the
            # requested configuration is used for this run (readiness is polled below).
            say "[CẢNH BÁO] Container cir-workbench đang chạy nhưng API backend chưa trả lời (HTTP ${code}); không có tác vụ đang chạy nên stack sẽ được khởi động lại với cấu hình hiện tại."
            "${RUN_SH}" up
            require_corrected_configuration "CONTAINER_RECONFIGURE" "$base"
            container_running=1
        fi
    fi

    if (( container_running == 0 )); then
        say "Khởi chạy Docker stack qua run.sh up..."
        "${RUN_SH}" up
    fi
    say "Thăm dò trạng thái sẵn sàng (readiness polling) của Backend và Frontend..."
    local ready_timeout="$TIMEOUT"
    if (( ready_timeout > 180 )); then
        ready_timeout=180
    fi
    if (( ready_timeout < 2 )); then
        ready_timeout=2
    fi

    # Readiness is "the backend answers", not "the whole registry has been resolved". /api/jobs
    # is a cheap in-memory listing, while /api/models builds the full per-model availability
    # view (registry, checkpoints, environment fingerprints) and can take tens of seconds on a
    # host with multi-GB checkpoints — polling it here would make readiness a function of how
    # much hashing happened to be needed. The /api/models check stays as a separate, single,
    # generously-timed validation below.
    local deadline=$(( SECONDS + ready_timeout ))
    local is_ready=0
    local be_code="" fe_code=""
    while (( SECONDS < deadline )); do
        be_code="$(curl -s --max-time 5 -o /dev/null -w '%{http_code}' "${base}/api/jobs" 2>/dev/null || true)"
        fe_code="$(curl -s --max-time 5 -o /dev/null -w '%{http_code}' "http://127.0.0.1:${frontend_port}" 2>/dev/null || true)"
        if [[ "$be_code" == "200" && ( "$fe_code" == "200" || "$fe_code" == "304" ) ]]; then
            is_ready=1
            break
        fi
        sleep 2
    done

    if (( is_ready != 1 )); then
        fail_stage "DOCKER_READINESS" "Dịch vụ unreachable / timeout sau ${ready_timeout} giây (Backend HTTP ${be_code}, Frontend HTTP ${fe_code})" "docker logs cir-workbench" "Kiểm tra log container bằng: docker logs cir-workbench"
    fi

    say "[OK] Docker stack đã sẵn sàng: Backend (${base}), Frontend (http://127.0.0.1:${frontend_port})"

    say "Kiểm tra mount dataset trong container (/data/FashionIQ/images)..."
    if ! docker exec cir-workbench test -d /data/FashionIQ/images; then
        fail_stage "DATASET_MOUNT" "Thư mục /data/FashionIQ/images không tồn tại hoặc không thể truy cập trong container cir-workbench" "workbench/docker/.env" "Kiểm tra HOST_FASHIONIQ_ROOT trong workbench/docker/.env"
    fi
}

# Behaviour 5: Prepare
run_model_preparation() {
    say "Chuẩn bị mô hình ${MODEL} qua pipeline guarded..."
    case "$MODEL" in
        limn|csmcir)
            sanitize_exec "${RUN_SH}" prepare "$MODEL"
            ;;
        *)
            sanitize_exec docker exec -T cir-workbench python workbench/scripts/pipeline.py setup \
                --model "${MODEL}" \
                --dataset-root "$(env_value WORKBENCH_DATASET_ROOT /data/FashionIQ)" \
                --sync-sources --download-checkpoints --apply \
                --allow-network --allow-large-downloads --allow-env-install --allow-preparation \
                --allow-gpu-eval --allow-index-write --resume
            ;;
    esac
}

# Behaviour 6: GPU Capability Verification
verify_gpu_in_container() {
    if (( NO_GPU == 1 )); then
        say "[SKIP] Bỏ qua kiểm chứng GPU do cờ --no-gpu"
        return 0
    fi

    say "Kiểm chứng runtime GPU bên trong container cho mô hình ${MODEL}..."
    local probe_out probe_rc=0
    probe_out="$(docker exec cir-workbench python -c "import sys, workbench.scripts.pipeline as p; sys.exit(p.probe_gpu_capability(sys.argv[1]))" "${MODEL}" 2>&1)" || probe_rc=$?

    if (( probe_rc != 0 )); then
        local diag_file="${REPO_ROOT}/workbench/artifacts/reports/gpu_diagnostic_${MODEL}_$(date +%Y%m%d_%H%M%S).txt"
        mkdir -p "$(dirname "$diag_file")"
        printf '%s\n' "$probe_out" > "$diag_file"
        fail_stage "GPU_VERIFICATION" "Kiểm chứng runtime GPU thất bại (exit code ${probe_rc}):\n${probe_out}" "$diag_file" "Kiểm tra tính tương thích CUDA/PyTorch và thiết lập driver GPU"
    fi

    say "$probe_out"
    say "[OK] Năng lực GPU cho ${MODEL} đã được kiểm chứng và ghi nhận thành công."
}
run_verifier_in_container() {
    local c_id
    c_id="$(docker ps -q --filter name=^cir-workbench$ 2>/dev/null | head -n1 || true)"
    if [[ -z "$c_id" ]]; then
        fail_stage "VERIFICATION" "Container cir-workbench không chạy. Xác thực bằng chứng thực thi khoa học phải chạy bên trong container execution environment." "docker ps" "Khởi động container bằng '${RUN_SH} up' hoặc chạy lại ./pc.sh"
    fi
    # `docker exec` starts a fresh process with the image's ENV plus the service's declared
    # `environment:` — it does NOT inherit whatever this shell has. The ledger key resolves from
    # XDG_CONFIG_HOME, so a bare exec could read a different key (or none) than the evaluation
    # wrote with, and every run would look uncorroborated. The values are read from the running
    # container's own configuration, not assumed from the host.
    local -a exec_env=()
    local declared
    while IFS= read -r declared; do
        [[ -n "$declared" ]] && exec_env+=(-e "$declared")
    done < <(docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' cir-workbench 2>/dev/null \
             | grep -E '^(XDG_CONFIG_HOME|WORKBENCH_[A-Z_]+|CIR_REPO_ROOT|CIR_DATA_ROOT|FASHIONIQ_ROOT)=' || true)
    docker exec -T "${exec_env[@]}" cir-workbench python3 workbench/scripts/verify_run.py "$@"
}

# Behaviour 7: Evaluation
execute_evaluation() {
    local backend_port base
    backend_port="$(env_value WORKBENCH_BACKEND_PORT 8000)"
    base="http://127.0.0.1:${backend_port}"

    say "Xác định protocol và checkpoint chính thức cho ${MODEL} từ API..."
    # Resolving the registry builds the per-model availability view (registry, checkpoint
    # presence and digests, environment fingerprints). On a host with multi-GB checkpoints that
    # is seconds, not milliseconds, so the timeout is generous and one retry is allowed — a
    # bare 30 s cap turns a slow-but-correct first call into a spurious failure.
    local protocol checkpoint bundle_status bundle_detail
    local model_info=""
    local attempt
    for attempt in 1 2; do
        local models_program
        models_program="$(cat <<'PYMODELS'

import json, sys

model_id = sys.argv[1]
requested = sys.argv[2] if len(sys.argv) > 2 else ''
try:
    models = json.load(sys.stdin)
except Exception as e:
    sys.exit(f'models endpoint error: {e}')

m = next((item for item in models if item.get('model_id') == model_id), None)
if not m:
    sys.exit(f'Mô hình {model_id} không tìm thấy trong /api/models')

protocol = m.get('native_protocol')
if not protocol:
    # No default. `ptha_mtst`'s protocol is unresolved in the registry, and substituting
    # `fashioniq_val_split` made the launcher evaluate a model under a protocol nobody had
    # established -- the one thing the protocol invariance rules forbid. A model whose native
    # protocol is not declared cannot be evaluated by this launcher, and saying so is the answer.
    sys.exit(f'Mô hình {model_id} không khai báo native_protocol trong registry; '
             'không thể xác định giao thức đánh giá (từ chối thay bằng mặc định).')
bundles = m.get('checkpoint_bundles') or []
checkpoints = m.get('checkpoints') or []

def _member_reasons(member):
    # The real /api/models contract. checkpoint_availability() reports presence as
    # checkpoint_downloaded and readiness as runnable; there has never been an 'available' field,
    # so testing for one made every member -- and every checkpoint -- look absent, which silently
    # downgraded the default full-bundle reproduction to a single category.
    reasons = []
    # String coercion, not concatenation of a value that may be null: a member whose
    # checkpoint_id is missing or null is exactly the malformed response this function exists to
    # report, and it must name that rather than raise TypeError inside the selection.
    cid = str(member.get('checkpoint_id') or '<unnamed member>')
    if not member.get('checkpoint_downloaded'):
        reasons.append(cid + ' chưa tải về (' + str(member.get('checkpoint_path') or '?') + ')')
    if not member.get('runnable'):
        blockers = member.get('blocking_reasons') or []
        reasons.append(cid + ' chưa sẵn sàng chạy' + (' — ' + str(blockers[0]) if blockers else ''))
    if not member.get('mapping_verified'):
        reasons.append(cid + ' chưa VERIFIED_METADATA')
    if member.get('official_sha256_known') and member.get('official_sha256_match') is False:
        reasons.append(cid + ' không khớp sha256 chính thức')
    return reasons

def _bundle_reasons(bundle):
    required = list(bundle.get('required_checkpoint_ids') or [])
    members = bundle.get('members') or []
    by_id = {mm.get('checkpoint_id'): mm for mm in members}
    required = [str(x) for x in required]
    reasons = []
    if not required:
        reasons.append('bundle không khai báo required_checkpoint_ids')
    absent = [cid for cid in required if cid not in by_id]
    if absent:
        reasons.append('API không báo cáo thành phần ' + ', '.join(absent))
    for member in members:
        reasons.extend(_member_reasons(member))
    return reasons

def _emit(selected, reasons, note):
    print(protocol, selected, 'complete' if not reasons else 'incomplete',
          ' | '.join(reasons) if reasons else note, sep='\t')

# No explicit selection: the model's declared bundle IS the target. This either runs the complete
# official experiment or refuses by name -- it never narrows itself to one category and presents
# that as the reproduction.
if not requested:
    if not bundles:
        if not checkpoints:
            sys.exit(model_id + ' không có checkpoint nào trong registry')
        ckpt = checkpoints[0]
        _emit(ckpt.get('checkpoint_id'), _member_reasons(ckpt), 'sẵn sàng')
        sys.exit(0)
    bundle = bundles[0]
    _emit(bundle.get('bundle_id'), _bundle_reasons(bundle), 'toàn bộ thành phần đã sẵn sàng')
    sys.exit(0)

# An explicit selection is the operator asking for that exact thing, including an intentional
# single-category run; it is still refused when the artifact it names is not actually usable.
bundle = next((b for b in bundles if b.get('bundle_id') == requested), None)
if bundle is not None:
    _emit(requested, _bundle_reasons(bundle), 'toàn bộ thành phần đã sẵn sàng')
    sys.exit(0)
ckpt = next((c for c in checkpoints if c.get('checkpoint_id') == requested), None)
if ckpt is None:
    sys.exit('checkpoint ' + requested + ' không có trong registry của ' + model_id)
_emit(requested, _member_reasons(ckpt), 'sẵn sàng')
PYMODELS
)"
        if model_info="$(curl -fsS --max-time 300 "${base}/api/models" 2>/dev/null | python3 -c "$models_program" "$MODEL" "$CHECKPOINT_REQUEST" 2>/dev/null)"; then
            if [[ -n "$model_info" ]]; then
                break
            fi
        fi
        say "  Lần thử ${attempt} thất bại; thử lại sau 5 giây..."
        sleep 5
    done
    if [[ -z "$model_info" ]]; then
        fail_stage "REGISTRY_RESOLUTION" "Không đọc được /api/models sau 2 lần thử" "docker logs cir-workbench" "Kiểm tra log container bằng: docker logs cir-workbench"
    fi

    IFS=$'\t' read -r protocol checkpoint bundle_status bundle_detail <<<"$model_info"
    FINAL_CHECKPOINT="$checkpoint"
    FINAL_PROTOCOL="$protocol"

    # A partial selection may only ever be the operator's explicit choice. The default command
    # asks for the model's declared bundle and gets either that complete experiment or a named
    # refusal -- never a single category wearing the label of a full reproduction.
    if [[ "$bundle_status" == "incomplete" ]]; then
        fail_stage "CHECKPOINT_SELECTION" \
            "Không thể chạy đánh giá đầy đủ cho ${MODEL} với checkpoint '${checkpoint}': ${bundle_detail}" \
            "workbench/artifacts/checkpoints/" \
            "Tải và xác thực checkpoint trước — '--verify-only' KHÔNG tải checkpoint; chạy giai đoạn chuẩn bị bằng ./pc.sh (hoặc ./pc.sh --checkpoint <id> để chủ động chạy một danh mục đơn lẻ, KHÔNG phải tái lập đầy đủ)"
    fi
    if [[ -n "$CHECKPOINT_REQUEST" && -n "$bundle_detail" ]]; then
        say "Lưu ý: đã chọn checkpoint '${checkpoint}' theo yêu cầu; đây KHÔNG phải tái lập đầy đủ nếu chỉ gồm một danh mục."
    fi
    say "Mô hình: ${MODEL} | Checkpoint: ${checkpoint} | Protocol: ${protocol} | top_k: 200"

    # Job Dedup: Check if a completed corroborated run exists in /api/runs or /api/benchmark-runs
    local target_run_id=""
    if (( FORCE != 1 )); then
        # The dedup reader must see BOTH artifact families: a per-query run in the DuckDB index and
        # an aggregate-only run beside it. Both bodies go to FILES for the same reason the index gate
        # does it -- a single argv element is capped at MAX_ARG_STRLEN (128 KiB) regardless of
        # ARG_MAX, and both endpoints grow with the deployment's run count, so passing them as
        # arguments would fail with E2BIG before python started. Only paths cross the boundary.
        #
        # A failed curl leaves an empty or truncated file; the `[ -s ]` test below replaces it with
        # `[]`, which is the semantics the old `|| echo "[]"` had, without holding the body in a
        # shell variable.
        local dedup_tmp_dir runs_json_file b_runs_json_file
        dedup_tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/cir_dedup.XXXXXX")" || dedup_tmp_dir=""
        [[ -n "$dedup_tmp_dir" ]] && _register_temp "$dedup_tmp_dir"
        local candidate_runs=""
        if [[ -n "$dedup_tmp_dir" ]]; then
            chmod 700 "$dedup_tmp_dir" 2>/dev/null || true
            runs_json_file="${dedup_tmp_dir}/runs.json"
            b_runs_json_file="${dedup_tmp_dir}/benchmark-runs.json"
            curl -s --max-time 10 "${base}/api/runs" >"$runs_json_file" 2>/dev/null || true
            curl -s --max-time 10 "${base}/api/benchmark-runs" >"$b_runs_json_file" 2>/dev/null || true
            # A failed call leaves an empty file, which the reader already treats as "no runs" --
            # the semantics the old `|| echo "[]"` had.
            [[ -s "$runs_json_file" ]] || printf '%s' "[]" >"$runs_json_file"
            [[ -s "$b_runs_json_file" ]] || printf '%s' "[]" >"$b_runs_json_file"

            local dedup_paths="${runs_json_file}
${b_runs_json_file}"
            candidate_runs="$(python3 -c "
import json, sys

model_id, checkpoint_id, protocol_id = sys.argv[1], sys.argv[2], sys.argv[3]
paths = sys.argv[4]

# Paths, not bodies: see the index gate for why an argv element cannot carry a response body.
def _load(path):
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as handle:
            data = json.load(handle)
    except Exception:
        return []
    return data if isinstance(data, list) else []

all_runs = []
for _path in paths.split('\n', 1):
    all_runs.extend(_load(_path))

seen = set()
for r in all_runs:
    if isinstance(r, dict):
        if (r.get('model_id') == model_id and
            r.get('checkpoint_id') == checkpoint_id and
            r.get('protocol_id') == protocol_id and
            r.get('run_id')):
            rid = str(r['run_id']).strip()
            if rid and rid not in seen:
                seen.add(rid)
                print(rid)
" "$MODEL" "$checkpoint" "$protocol" "$dedup_paths" 2>/dev/null || true)"

            if [[ -n "$candidate_runs" ]]; then
                while IFS= read -r cand_id; do
                    [[ -z "$cand_id" ]] && continue
                    local v_rc=0 v_out_file
                    v_out_file="${dedup_tmp_dir}/verifier.$$.${RANDOM}.json"
                    # The verifier's own JSON goes to a FILE too. It is small in the success case and
                    # unbounded in the refusal case, where every rejected reason is echoed back, and
                    # it used to be argv[1] of the reader below -- the same 128 KiB ceiling.
                    # A SUBSHELL, not a redirection group. `run_verifier_in_container` calls
                    # `fail_stage` when the container is down, and `fail_stage` exits the shell it
                    # runs in; the previous `$( ... )` captured that exit because a command
                    # substitution IS a subshell. `>file` alone kept the function in the current
                    # shell, so that exit terminated the whole launcher from the middle of the frame
                    # it was analyzing, and the refusal vanished into the redirection: the operator
                    # saw a silent exit 1 with no stage named. Parentheses restore the old
                    # containment while still landing the output in the file every reader expects.
                    ( run_verifier_in_container --run-id "$cand_id" --model "$MODEL" --checkpoint "$checkpoint" --protocol "$protocol" --json ) >"$v_out_file" 2>&1 || v_rc=$?
                    local check_rc=0
                    python3 - "$v_out_file" "$v_rc" "$cand_id" "$b_runs_json_file" <<'PYCHECK' || check_rc=$?
import json, sys

v_out_path = sys.argv[1]
v_rc = int(sys.argv[2])
cand_id = sys.argv[3]
b_runs_path = sys.argv[4]

try:
    with open(v_out_path, 'r', encoding='utf-8', errors='replace') as handle:
        v_out_str = handle.read()
except OSError:
    # A reader that cannot read the verdict knows nothing about the run, so it refuses rather than
    # falling through to the aggregate-answer path below.
    sys.exit(1)

try:
    with open(b_runs_path, 'r', encoding='utf-8', errors='replace') as handle:
        b_runs_str = handle.read()
except OSError:
    b_runs_str = "[]"

res = None
try:
    res = json.loads(v_out_str)
except Exception:
    for line in reversed(v_out_str.strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                res = json.loads(line)
                break
            except Exception:
                pass
if not isinstance(res, dict):
    sys.exit(1)
if v_rc == 0 and res.get("is_research_evidence"):
    sys.exit(0)

try:
    b_runs = json.loads(b_runs_str)
except Exception:
    b_runs = []

api_proof_unavailable = False
if isinstance(b_runs, list):
    for r in b_runs:
        if isinstance(r, dict) and str(r.get("run_id")).strip() == cand_id:
            v = r.get("verification")
            if isinstance(v, dict):
                if (v.get("state") == "proof_unavailable" or v.get("certification_mode") == "proof_unavailable") and v.get("execution_authenticated"):
                    api_proof_unavailable = True
                    break
            elif r.get("corroborated") or r.get("benchmark_gate"):
                api_proof_unavailable = True
                break

if not api_proof_unavailable:
    sys.exit(1)

reasons = list(res.get("reasons") or [])
if not reasons:
    sys.exit(1)

ledger_ok = all(
    ("proof" in r.lower() or "declared output missing" in r.lower()) and
    ("missing" in r.lower() or "does not exist" in r.lower() or "no such file" in r.lower()) and
    "evidence ledger" not in r.lower() and
    "ledger return_code" not in r.lower() and
    "ledger invocation_status" not in r.lower() and
    "ledger=" not in r.lower() and
    "report digest" not in r.lower() and
    "category metric" not in r.lower() and
    "metric extraction" not in r.lower() and
    "missing aggregate" not in r.lower() and
    "status=" not in r.lower() and
    "return_code=" not in r.lower() and
    "checksum" not in r.lower() and
    "mismatch" not in r.lower() and
    "did not complete" not in r.lower()
    for r in reasons
)

execution = res.get("execution") or {}
if execution.get("return_code") not in (None, 0):
    ledger_ok = False
if execution.get("invocation_status") not in (None, "INVOCATION_COMPLETED"):
    ledger_ok = False

if ledger_ok:
    sys.exit(2)

sys.exit(1)
PYCHECK
                    if (( check_rc == 0 )); then
                        target_run_id="$cand_id"
                        FINAL_RUN_PROOF_UNAVAILABLE=0
                        break
                    elif (( check_rc == 2 )); then
                        target_run_id="$cand_id"
                        FINAL_RUN_PROOF_UNAVAILABLE=1
                        break
                    fi
                done <<< "$candidate_runs"
            fi
            rm -rf "$dedup_tmp_dir"
        fi

        if [[ -n "$target_run_id" ]]; then
            say "Phát hiện kết quả đánh giá đã hoàn thành và được sổ cái chứng thực: run_id=${target_run_id}"
            say "Tái sử dụng kết quả có sẵn (sử dụng cờ --force nếu muốn buộc đánh giá lại)."
            FINAL_RUN_ID="$target_run_id"
            return 0
        fi
    fi

    # Check if a matching job is already queued or running
    local active_job_id=""
    active_job_id="$(curl -s --max-time 10 "${base}/api/jobs" 2>/dev/null | python3 -c "
import json, sys

model_id = sys.argv[1]
checkpoint_id = sys.argv[2]
protocol_id = sys.argv[3]

try:
    jobs = json.load(sys.stdin)
    if isinstance(jobs, list):
        for j in jobs:
            if (j.get('model_id') == model_id and
                j.get('checkpoint_id') == checkpoint_id and
                j.get('protocol_id') == protocol_id and
                j.get('status') in {'queued', 'preparing', 'running'}):
                print(j['job_id'])
                sys.exit(0)
except Exception:
    pass
" "$MODEL" "$checkpoint" "$protocol" 2>/dev/null || true)"

    local job_id=""
    if [[ -n "$active_job_id" ]]; then
        say "Phát hiện job tương ứng đang xử lý (job_id: ${active_job_id}). Tự động gắn (attach) theo dõi..."
        job_id="$active_job_id"
    else
        say "Xếp hàng đánh giá qua API POST /api/jobs..."
        local post_resp
        post_resp="$(curl -s -w '\n%{http_code}' --max-time 60 -X POST "${base}/api/jobs" \
            -H "Content-Type: application/json" \
            -d "{\"model_id\":\"${MODEL}\",\"checkpoint_id\":\"${checkpoint}\",\"protocol_id\":\"${protocol}\",\"top_k\":200}")"
        local post_code="${post_resp##*$'\n'}"
        local post_body="${post_resp%$'\n'*}"

        if [[ "$post_code" == "200" || "$post_code" == "201" ]]; then
            job_id="$(python3 -c "import json, sys; print(json.loads(sys.argv[1]).get('job_id', ''))" "$post_body")"
            say "Đã xếp job thành công: ${job_id}"
        elif [[ "$post_code" == "409" ]]; then
            job_id="$(python3 -c "import json, sys; print(json.loads(sys.argv[1]).get('details', {}).get('job_id', ''))" "$post_body")"
            if [[ -n "$job_id" ]]; then
                say "API thông báo job trùng lặp đang chạy: gắn vào job_id ${job_id}"
            else
                fail_stage "EVALUATION_QUEUE" "Lỗi xếp hàng (409): ${post_body}" "workbench/artifacts/jobs/" "Kiểm tra danh sách jobs tại ${base}/api/jobs"
            fi
        else
            fail_stage "EVALUATION_QUEUE" "Lỗi xếp hàng (HTTP ${post_code}): ${post_body}" "workbench/artifacts/jobs/" "Kiểm tra log backend"
        fi
    fi

    # Monitor until terminal state
    say "Đang theo dõi tiến trình thực thi job ${job_id} (thời hạn: ${TIMEOUT}s)..."
    local start_sec=$SECONDS
    local job_status="queued"
    local run_id_from_job=""
    while (( SECONDS - start_sec < TIMEOUT )); do
        # Query /api/jobs list first (canonical in run.sh/API), fallback to individual job endpoint
        # The job's OWN record, never a substitute. Falling back to another job's entry would
        # report that job's terminal status for a job that may never have run — a fabricated
        # success. Absent means unknown, and the loop keeps waiting.
        job_json="$(curl -s --max-time 15 "${base}/api/jobs" 2>/dev/null | python3 -c "
import json, sys
job_id = sys.argv[1]
try:
    jobs = json.load(sys.stdin)
    if isinstance(jobs, list):
        target = next((j for j in jobs if j.get('job_id') == job_id), None)
        if target:
            print(json.dumps(target))
except Exception:
    pass
" "$job_id" 2>/dev/null || true)"

        if [[ -z "$job_json" ]]; then
            job_json="$(curl -s --max-time 15 "${base}/api/jobs/${job_id}" 2>/dev/null | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    if isinstance(data, dict):
        print(json.dumps(data))
    elif isinstance(data, list) and data:
        print(json.dumps(data[0]))
except Exception:
    pass
" 2>/dev/null || true)"
        fi

        if [[ -n "$job_json" ]]; then
            job_status="$(python3 -c "import json, sys; print(json.loads(sys.argv[1]).get('status', ''))" "$job_json")"
            local elapsed=$(( SECONDS - start_sec ))
            say "[$(date +%T)] Job ${job_id}: trạng thái '${job_status}' (đã chạy ${elapsed}s)..."

            case "$job_status" in
                completed)
                    run_id_from_job="$(python3 -c "import json, sys; d = json.loads(sys.argv[1]); print(d.get('provenance', {}).get('run_id') or '')" "$job_json")"
                    break
                    ;;
                failed|blocked|cancelled)
                    local err_msg log_dir
                    err_msg="$(python3 -c "import json, sys; d = json.loads(sys.argv[1]); print(d.get('error') or d.get('stderr_tail') or 'Đánh giá bị ngắt quãng') " "$job_json")"
                    log_dir="$(python3 -c "import json, sys; d = json.loads(sys.argv[1]); print(d.get('log_directory') or '')" "$job_json")"
                    fail_stage "EVALUATION_EXECUTION" "Đánh giá thất bại với trạng thái '${job_status}' (GPU chưa kiểm chứng hoặc lỗi):\n${err_msg}" "${log_dir:-workbench/artifacts/jobs/}" "Kiểm tra log của job tại thư mục log hoặc docker logs cir-workbench"
                    ;;
            esac
        fi
        sleep 1
    done

    if [[ "$job_status" != "completed" ]]; then
        curl -s -X POST "${base}/api/jobs/${job_id}/cancel" >/dev/null 2>&1 || true
        fail_stage "EVALUATION_TIMEOUT" "Quá thời hạn chờ timeout (${TIMEOUT}s) cho job ${job_id}" "workbench/artifacts/jobs/" "Tăng thời gian timeout với cờ --timeout <số_giây>"
    fi

    if [[ -z "$run_id_from_job" ]]; then
        # There is deliberately NO fallback here.
        #
        # A search of /api/benchmark-runs by model+checkpoint+protocol was tried and removed. The
        # job that just completed already ran a commit with NO `cir.commit` label and no
        # `--run-id`; the currently checked-out source is a DIFFERENT commit, and the job just
        # completed against a source that has since changed. A search that selected *any*
        # corroborated report with those three fields would therefore bind the summary -- and so
        # the verification, the gate, the index gate and the success claim -- to whatever previous
        # execution happened to be newest, which is how a certificate about one artifact gets
        # printed for another. Refusing costs one re-run; inheriting costs the science.
        fail_stage "EVALUATION_IDENTITY" \
            "Job ${job_id} báo hoàn tất nhưng hồ sơ của chính nó KHÔNG ghi run_id, nên không có danh tính thực thi nào để xác thực." \
            "${base}/api/jobs" \
            "Không suy đoán lượt chạy này là lượt nào: một báo cáo cùng model/checkpoint/protocol có thể thuộc lần chạy KHÁC. Kiểm tra 'docker logs cir-workbench' và ${base}/api/jobs; nếu có lượt chạy hợp lệ đã chứng thực, xác nhận danh tính của nó rồi chạy lại ./pc.sh (script sẽ tái sử dụng kết quả đã chứng thực thay vì đánh giá lại)."
    fi

    FINAL_RUN_ID="$run_id_from_job"
}


# The DuckDB index is a CONSUMER of the verification, not a step beside it: `index.evidence_status`
# promotes a per-query artifact only by reading the receipt the API derives from the aggregate report.
# Rebuilding before the verifier ran therefore indexed a state that was already stale, and `|| true`
# hid every failure -- so a run could be fully evaluated and verified yet absent from the API and the UI.
rebuild_and_verify_index() { # rebuild_and_verify_index RUN_ID MODEL CHECKPOINT PROTOCOL
    local run_id="$1" model="$2" checkpoint="$3" protocol="$4"
    local backend_port base
    backend_port="$(env_value WORKBENCH_BACKEND_PORT 8000)"
    base="http://127.0.0.1:${backend_port}"

    say "Tái xây dựng chỉ mục DuckDB (/api/index/rebuild)..."
    local resp code body
    # `|| true` INSIDE the substitution: a curl that times out or cannot connect must not abort the
    # script here (that would skip the diagnosis), and its own error text is what the operator needs.
    resp="$(curl -sS -w '\n%{http_code}' --max-time 300 -X POST "${base}/api/index/rebuild" 2>&1 || true)"
    code="$(printf '%s' "$resp" | tail -n1)"
    body="$(printf '%s' "$resp" | sed '$d')"
    if [[ ! "$code" =~ ^2[0-9][0-9]$ ]]; then
        fail_stage "INDEX_REBUILD" \
            "Tái xây dựng chỉ mục DuckDB thất bại (HTTP ${code:-không có mã}): ${body:-không có phản hồi}" \
            "${base}/api/index/rebuild" \
            "Chỉ mục là thứ API và UI đọc; lượt chạy đã đánh giá vẫn còn trên đĩa nhưng CHƯA hiển thị được. Kiểm tra log backend (docker logs cir-workbench), thử lại POST ${base}/api/index/rebuild, rồi chạy lại ./pc.sh."
    fi

    # An HTTP 200 is not enough: the rebuild must actually contain THIS run with the evidence state
    # the verification just established. A stale index -- the previous model's run, or this run
    # before the verifier spoke -- returns 200 and would be reported as success.
    # The reader lives in a variable because a here-document and a pipe cannot both feed one
    # interpreter: the document wins and the JSON would be lost.
    local runs_check reports_check checked
    # The API bodies are written to FILES, never passed as arguments.
    #
    # A single argv element is capped at MAX_ARG_STRLEN -- 32 pages, i.e. 128 KiB -- independently of
    # the much larger ARG_MAX, and both endpoints grow with the number of runs the deployment has
    # accumulated. The gate used to pass both bodies (and the verifier's JSON) through argv, so past
    # that cap execve failed with E2BIG *before* python started and the launcher reported an index
    # failure for a run whose evidence was perfectly valid. `mktemp` names the files and only short
    # paths cross the argv boundary; the bodies themselves stay on disk.
    #
    # The redirection also replaces the command substitution: a curl failure leaves an EMPTY or
    # PARTIAL file rather than a shell variable, so the empty-body test below is the same test it
    # always was, and the `__API_UNREACHABLE__` sentinel is written explicitly instead of being
    # substituted. A PARTIAL body is the case that differs from a shell variable: `>` writes what
    # curl managed to send before it failed, and a truncated JSON document does not parse, so the
    # gate refuses it as malformed rather than reading it as an empty deployment.
    local idx_tmp_dir
    idx_tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/cir_idx.XXXXXX")" || \
        fail_stage "INDEX_VERIFY" "Không tạo được thư mục tạm để đọc API" "${TMPDIR:-/tmp}" \
            "Kiểm tra quyền ghi vào thư mục tạm rồi chạy lại ./pc.sh"
    _register_temp "$idx_tmp_dir"
    # The bodies can carry run metadata an operator would not want world-readable; the directory is
    # private to this user and is removed on every exit path below.
    chmod 700 "$idx_tmp_dir" 2>/dev/null || true
    local runs_file reports_file
    runs_file="${idx_tmp_dir}/runs.json"
    reports_file="${idx_tmp_dir}/benchmark-runs.json"
    if ! curl -fsS --max-time 60 "${base}/api/runs" >"$runs_file" 2>/dev/null; then
        printf '%s' "__API_UNREACHABLE__" >"$runs_file"
    fi
    # The aggregate view is the one that carries runs with no per-query export, so the gate has to
    # ask BOTH endpoints. An aggregate-only run is in the second and not the first.
    if ! curl -fsS --max-time 60 "${base}/api/benchmark-runs" >"$reports_file" 2>/dev/null; then
        printf '%s' "__API_UNREACHABLE__" >"$reports_file"
    fi
    local idx_program
    idx_program="$(cat <<'PYIDX'
import json, sys

# The success contract, read from the API the operator will actually look at.
#
# Two artifact families exist and they answer different questions. The DuckDB index holds the
# canonical PER-QUERY records, and only a model that declares an observer ever writes one;
# aggregate reports live beside it, are served by /api/benchmark-runs, and are the whole result
# for the other fifteen models. Requiring every model to appear in /api/runs therefore made a
# correct aggregate-only run unable to satisfy its own gate.
#
# What is required is now derived from the SAME run, by identity: its aggregate report must be
# present in /api/benchmark-runs and certified. Its per-query artifact must be recorded in
# /api/runs exactly when the report DECLARES an observer -- checked against the report's own
# `per_query_capability`, never assumed from the model id here.
#
# A per-query run whose artifact is missing or unverified keeps its report in the aggregate view
# (a real execution happened) but is refused, so this gate can never present one as full.
paths, run_id, model, checkpoint, protocol = sys.argv[1:6]
# The bodies arrive as PATHS. A single argv element is capped at MAX_ARG_STRLEN (128 KiB) no matter
# how generous ARG_MAX is, and both endpoints grow with the deployment's run count, so passing them
# as arguments failed with E2BIG before this program ever started. A path cannot grow with the data.
def _body(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except OSError as error:
        # An unreadable file is NOT an empty deployment: every message below that treats a missing
        # body as "the API is down" must stay reachable, so an I/O failure is reported as unreachable
        # rather than as a parse error about content nobody received.
        return "__API_UNREACHABLE__ (%s: %s)" % (type(error).__name__, error)

_runs_path, _reports_path = paths.split("\n", 1)
raw = _body(_runs_path)
reports_raw = _body(_reports_path)
if raw.strip().startswith("__API_UNREACHABLE__") or not raw.strip():
    sys.exit("không gọi được /api/runs: %s" % raw.strip().replace("__API_UNREACHABLE__", "").strip(" ()")
             if raw.strip().startswith("__API_UNREACHABLE__") else "không gọi được /api/runs")
try:
    runs = json.loads(raw)
except Exception as error:
    sys.exit("không đọc được /api/runs: %s" % type(error).__name__)
if not isinstance(runs, list):
    sys.exit("phản hồi /api/runs không phải danh sách")

if reports_raw.strip().startswith("__API_UNREACHABLE__") or not reports_raw.strip():
    sys.exit("không gọi được /api/benchmark-runs: %s"
             % reports_raw.strip().replace("__API_UNREACHABLE__", "").strip(" ()")
             if reports_raw.strip().startswith("__API_UNREACHABLE__") else "không gọi được /api/benchmark-runs")
try:
    reports = json.loads(reports_raw)
except Exception as error:
    sys.exit("không đọc được /api/benchmark-runs: %s" % type(error).__name__)
if not isinstance(reports, list):
    sys.exit("phản hồi /api/benchmark-runs không phải danh sách")

report = None
for candidate in reports:
    if isinstance(candidate, dict) and candidate.get("run_id") == run_id:
        report = candidate
        break
if report is None:
    sys.exit("báo cáo tổng hợp của lượt chạy %s KHÔNG được API phục vụ: chưa có trong sổ cái, "
             "danh tính không khớp, hoặc tệp không đọc được" % run_id)
for field, expected in (("model_id", model), ("checkpoint_id", checkpoint), ("protocol_id", protocol)):
    if report.get(field) != expected:
        sys.exit("báo cáo tổng hợp gắn %s=%r, không phải %r" % (field, report.get(field), expected))
receipt = report.get("verification") if isinstance(report.get("verification"), dict) else {}

# ONE policy is an honest not-a-pass rather than a refusal: the deployment's completion proof is
# gone, so the heavy verifier cannot run, and the API says so by name. The launcher's own verifier
# output decides whether that is the actual reason; this program only sees the API, so it carries
# the policy forward rather than re-deciding it.
proof_policy = (
    receipt.get("certification_mode") == "proof_unavailable"
    and receipt.get("execution_authenticated") is True
    and receipt.get("state") in ("proof_unavailable", "refused"))

# A DEVELOPMENT CERTIFICATE IS NOT A PRODUCTION CERTIFICATION.
#
# `_certification_mode` answers "development" only for a process that declared itself one AND opted
# in, and there is deliberately no completion proof to validate in that state -- which is why the
# API's development receipt can carry `certified=true` and `is_research_evidence=true` while nothing
# about the RUN was ever verified against a proof. Those booleans answer the verifier's metric rules;
# they are not a statement about the artifact. Reading them here without asking WHICH policy produced
# them is what would let a development receipt satisfy the production release gate, so the policy is
# checked first and by name. The frontend was already hardened for this; the launcher's own gate is
# the one that decides the operator's exit code, so it must refuse independently.
_certification_mode = receipt.get("certification_mode")
if receipt.get("development_mode") is True or _certification_mode == "development":
    sys.exit("API chứng thực lượt chạy này bằng CHỨNG THƯ PHÁT TRIỂN (development_mode=%r, "
             "certification_mode=%r). Đây là chế độ dành cho quy trình phát triển: completion proof "
             "đã KHÔNG được xác thực, nên lượt chạy không thể là bằng chứng nghiên cứu đã chứng thực "
             "của bản phát hành. Chạy lại trên một tiến trình triển khai (không đặt "
             "WORKBENCH_DEVELOPMENT_MODE / WORKBENCH_ALLOW_DEVELOPMENT_CERTIFICATES) để có chứng thực "
             "đầy đủ theo proof."
             % (receipt.get("development_mode"), _certification_mode))

if not (receipt.get("is_research_evidence") and receipt.get("certified")):
    if not proof_policy:
        reasons = "; ".join(str(r) for r in (receipt.get("evidence_reasons") or receipt.get("reasons") or []))
        sys.exit("API không công bố lượt chạy này là bằng chứng nghiên cứu đã chứng thực "
                 "(trạng thái %r)%s" % (receipt.get("state") or "không xác định",
                                        (": " + reasons) if reasons else ""))

# The receipt must name the bytes it is a verdict about. The row arrives from the API, so the
# digest is what ties the fields below to the file the ledger corroborated: a receipt without one
# means the capability that decides the whole success contract was read from a report nobody
# certified.
if not receipt.get("report_digest"):
    sys.exit("API không công bố digest của báo cáo cho lượt chạy %s, nên không thể gắn báo cáo "
             "với chứng thực" % run_id)

# The model's OWN declaration decides which contract applies, and it comes from the registry --
# the same declaration the evaluator read when it decided whether to run an observer. Reading it
# out of the report cannot work: a report is a file, and one without the field would downgrade an
# observer model to aggregate-only and skip its Top-K requirement entirely.
declaration = sys.argv[6].strip() if len(sys.argv) > 6 else ""
if declaration.startswith("REGISTRY_CONFIRMED_PER_QUERY:"):
    declared_observer = declaration.split(":", 1)[1].strip()
elif declaration == "REGISTRY_CONFIRMED_AGGREGATE_ONLY":
    declared_observer = ""
else:
    # Unknown is not absent. Without a read of the registry there is no authority for the claim that
    # this model needs no per-query evidence, and the contract cannot be decided from a report.
    sys.exit("không xác định được khai báo per-query của mô hình %s từ registry (%r)"
             % (model, declaration or "không có phản hồi"))
report_capability = report.get("per_query_capability")
if declared_observer and not report_capability:
    sys.exit("mô hình %s KHAI BÁO trình quan sát (%s) trong registry, nhưng báo cáo của lượt chạy "
             "%s không khai báo năng lực per-query: báo cáo và registry nói hai điều khác nhau "
             "về cùng một mô hình" % (model, declared_observer, run_id))
capability = report_capability or declared_observer
state = report.get("per_query_state")
if proof_policy:
    # Which axis the missing proof belongs to is a property of the model, so the policy name is
    # reported with it rather than instead of it.
    state = "PROOF_UNAVAILABLE_AND_" + (state or "CAPABILITY_ABSENT")
indexed = None
for candidate in runs:
    if isinstance(candidate, dict) and candidate.get("run_id") == run_id:
        indexed = candidate
        break

if not capability:
    # Aggregate-only by declaration. There is no per-query artifact to look for, and demanding
    # one would refuse a run that is complete for what its model claims to be able to do.
    #
    # The state is REPORTED, not folded into the token: the shell compares this value, and a
    # line with a parenthetical in it would compare unequal to every branch that looks for it.
    if state and state not in ("CAPABILITY_ABSENT",):
        print("ghi chú: báo cáo khai %r cho một mô hình không có trình quan sát" % state,
              file=sys.stderr)
    # EXECUTION AUTHENTICATION IS NOT SCIENTIFIC CERTIFICATION.
    #
    # `execution_authenticated` says the ledger corroborates that this invocation ran and wrote these
    # bytes at this identity. `certified` says the completion proof validated and the run's numbers
    # are reproducible from its own output. A missing proof satisfies the first and not the second,
    # and the two must never share a token: on the honest-policy branch the API is telling the caller
    # that it could NOT complete the verification, so an aggregate-only run in that state is
    # "executed and recorded", not "verified". Reporting AGGREGATE_ONLY_VERIFIED here would upgrade a
    # run that never passed verification into a verified research result.
    if proof_policy:
        print("AGGREGATE_ONLY_EXECUTION_AUTHENTICATED_PROOF_UNAVAILABLE")
    else:
        print("AGGREGATE_ONLY_VERIFIED")
    sys.exit(0)

if indexed is None:
    sys.exit("mô hình %s khai báo trình quan sát theo truy vấn, nhưng lượt chạy %s KHÔNG có trong "
             "chỉ mục /api/runs: artifact Top-K chưa được xuất, hoặc chưa được chứng thực"
             % (model, run_id))
for field, expected in (("model_id", model), ("checkpoint_id", checkpoint), ("protocol_id", protocol)):
    if indexed.get(field) != expected:
        sys.exit("chỉ mục gắn %s=%r, không phải %r" % (field, indexed.get(field), expected))
status = indexed.get("evidence_status")
if not isinstance(status, str) or not status:
    sys.exit("chỉ mục không công bố trạng thái bằng chứng cho lượt chạy này")
if status != "PER_QUERY_VERIFIED":
    sys.exit("mô hình %s khai báo trình quan sát theo truy vấn và artifact Top-K của nó đang ở "
             "trạng thái %r, không phải PER_QUERY_VERIFIED" % (model, status))
print(status)
PYIDX
)"
    # The reader is handed its inputs as ARGUMENTS, never on stdin: a here-document and a pipe
    # cannot both feed one interpreter, and this script has a here-document immediately below.
    # The registry's own declaration of the model's per-query capability is passed IN. It is the
    # same field the evaluator read when it decided whether to run an observer, and the launcher
    # already has it: the summary's own model_info did not carry it, so it is asked for here, once,
    # from the API that serves the registry.
    local declared_observer
    # THREE states, never two. "The registry declares no observer" and "I could not read the
    # registry" are different facts, and collapsing them -- which an empty string did -- turns an
    # API outage into an authoritative statement that the model needs no per-query evidence. The
    # program therefore prints a named state, and UNKNOWN is refused below.
    declared_observer="$(curl -fsS --max-time 30 "${base}/api/models" 2>/dev/null | python3 -c "
import json, sys
model_id = sys.argv[1]
try:
    models = json.load(sys.stdin)
except Exception:
    print('REGISTRY_UNKNOWN')
    raise SystemExit(0)
if not isinstance(models, list):
    print('REGISTRY_UNKNOWN')
    raise SystemExit(0)
for entry in models:
    if isinstance(entry, dict) and entry.get('model_id') == model_id:
        value = entry.get('per_query_observer')
        if isinstance(value, str) and value.strip():
            print('REGISTRY_CONFIRMED_PER_QUERY:' + value.strip())
        elif value is None:
            # The key is absent or null for every model that declares no observer: that IS the
            # registry's declaration, and it is the only way aggregate-only is established.
            print('REGISTRY_CONFIRMED_AGGREGATE_ONLY')
        else:
            # Present but not a string: a malformed declaration is not a declaration.
            print('REGISTRY_UNKNOWN')
        break
else:
    print('REGISTRY_UNKNOWN')
" "$model" 2>/dev/null || true)"
    if [[ -z "$declared_observer" || "$declared_observer" == "REGISTRY_UNKNOWN" ]]; then
        fail_stage "REGISTRY_CAPABILITY" \
            "Không xác định được khai báo per-query của mô hình ${model} từ /api/models (${declared_observer:-không có phản hồi})" \
            "${base}/api/models" \
            "Không tuyên bố hoàn tất khi chưa biết mô hình có trình quan sát hay không: một API không đọc được KHÔNG có nghĩa là mô hình không cần bằng chứng per-query. Kiểm tra backend (docker logs cir-workbench) rồi chạy lại ./pc.sh."
    fi
    local idx_paths="${runs_file}
${reports_file}"
    # STDOUT IS THE TOKEN, STDERR IS THE EXPLANATION, and they must never be merged.
    #
    # This reader is both a verdict and a sentence: it prints the evidence state on stdout, and it
    # also prints advisory notes -- a model that declares no observer, a state worth mentioning -- on
    # stderr precisely so they cannot contaminate the token. `2>&1` destroyed that: the note was
    # prepended to the status, `INDEX_EVIDENCE_STATUS` no longer equalled any token the summary
    # compares against, every branch below fell through to its `else`, and a proof-unavailable run
    # was reported "SẴN SÀNG CHO PILOT" with exit 0 instead of exit 7. The program already states
    # this contract in its own comment; the shell was violating it.
    #
    # The notes go to the operator's stderr, where they were always meant to be read.
    local idx_stderr_file="${idx_tmp_dir}/gate.stderr"
    checked="$(python3 -c "$idx_program" "$idx_paths" "$run_id" "$model" "$checkpoint" "$protocol" "${declared_observer:-}" 2>"$idx_stderr_file")" \
        || checked="__INDEX_CHECK_FAILED__: ${checked:-không xác định} $(cat "$idx_stderr_file" 2>/dev/null)"
    if [[ -s "$idx_stderr_file" ]]; then
        # Kept out of the token but not thrown away: a silent stderr would be its own defect.
        cat "$idx_stderr_file" >&2
    fi
    rm -rf "$idx_tmp_dir"
    if [[ "$checked" == __INDEX_CHECK_FAILED__* ]]; then
        fail_stage "INDEX_VERIFY" \
            "Lượt chạy không hiện diện trong chỉ mục với đúng danh tính và trạng thái bằng chứng: ${checked#__INDEX_CHECK_FAILED__: }" \
            "${base}/api/runs" \
            "Không tuyên bố hoàn tất: API phải phục vụ được lượt chạy này với đúng danh tính và đúng trạng thái bằng chứng mà nó khai. Xem workbench/docs/PC_ONE_COMMAND_GUIDE.vi.md (chỉ mục DuckDB)."
    fi
    INDEX_EVIDENCE_STATUS="$checked"
}

# The verdict is consumed exactly once. Two facts fix the order: an unverified run must not be
# indexed nor reported as finished, so the gate runs first; and the receipt the index promotes a
# per-query artifact from is derived from the aggregate report, so the rebuild must run after the
# verifier has spoken. This function therefore decides, writes the decision for the summary, and
# leaves the rebuild to `rebuild_and_verify_index`.
verify_gate() { # verify_gate VERIFY_OUT_PATH VERIFY_RC MODEL CHECKPOINT PROTOCOL RUN_ID PROOF_UNAVAILABLE REPORT_PATH HOST_ARTIFACTS
    # The first argument is a PATH, not the verdict text. A refusal that names every rejected reason
    # is unbounded, and a single argv element is capped at MAX_ARG_STRLEN (128 KiB) independently of
    # ARG_MAX, so passing the text here could fail with E2BIG before this reader started.
    local verify_out="$1" verify_rc="$2" model="$3" checkpoint="$4" protocol="$5" run_id="$6"
    # The caller owns the temp path and passes it in: it creates one with `mktemp`, so the file the
    # gate writes is the file the summary reads, and neither name is guessable in a shared /tmp.
    VERIFY_GATE_STATE="${VERIFY_STATE_FILE:-${TMPDIR:-/tmp}/cir_verify_state.$$}"
    python3 - "$verify_out" "$verify_rc" "$model" "$checkpoint" "$protocol" "$run_id" \
             "${7:-0}" "${8:-}" "$VERIFY_GATE_STATE" "${9:-}" <<'PYGATE'
#!/usr/bin/env python3
"""The one place the verifier's verdict is interpreted. Prints the refusal, writes the decision."""
import json, sys

# The verdict is READ FROM A FILE. Its size is data, and argv cannot carry data.
try:
    with open(sys.argv[1], "r", encoding="utf-8", errors="replace") as _handle:
        verify_out = _handle.read()
except OSError as _error:
    # An unreadable verdict is not an empty one: the refusal below is the honest answer, and it must
    # not be reachable as "result is None -> invalid JSON" with no indication that the file was gone.
    print(f"Lỗi: không đọc được kết quả xác thực: {type(_error).__name__}: {_error}")
    sys.exit(1)
verify_rc = int(sys.argv[2])
model_id = sys.argv[3]
checkpoint_id = sys.argv[4]
protocol_id = sys.argv[5]
run_id = sys.argv[6]
proof_unavailable = sys.argv[7] == "1"
report_path = sys.argv[8]
state_path = sys.argv[9]

result = None
for line in reversed(verify_out.strip().splitlines()):
    line = line.strip()
    if line.startswith("{") and line.endswith("}"):
        try:
            result = json.loads(line)
            break
        except Exception:
            pass
if result is None:
    try:
        result = json.loads(verify_out)
    except Exception:
        pass
if result is None:
    print(f"Lỗi: verify_run.py không trả về JSON hợp lệ (exit code {verify_rc}):\n{verify_out}")
    sys.exit(1)

verified = bool(result.get("is_research_evidence", False))

# The per-query axis is a SEPARATE achievement from the aggregate. A model that declares an observer
# is only fully pilot-ready when its Top-K artifact was exported; a model that declares none is
# aggregate-only BY DESIGN and must not be reported as a failure. Read from the aggregate report
# rather than guessed, because the report is what the run itself recorded.
per_query_capability = None
per_query_state = None
_per_query_reason = ""
# The verifier runs INSIDE the container, so the path it reports is the CONTAINER's
# (`/artifacts/reports/...`), which does not exist on the host.
_report_path = report_path
_host_artifacts = sys.argv[10] if len(sys.argv) > 10 else ""
if _report_path and _host_artifacts and _report_path.startswith("/artifacts/"):
    _report_path = _host_artifacts.rstrip("/") + "/" + _report_path[len("/artifacts/"):]
_report_read = False
if _report_path:
    try:
        with open(_report_path, "r", encoding="utf-8") as fh:
            _report = json.load(fh)
        # The path comes from the verifier's own JSON, but a report is a FILE: reading whichever
        # one a path happens to name, without checking whose it is, is how one run's per-query
        # state gets attributed to another. The identity is re-checked here rather than trusted.
        if str(_report.get("run_id") or "") != run_id:
            raise ValueError(
                "report %s describes run %r, not %r" % (_report_path, _report.get("run_id"), run_id))
        for _field, _expected in (("model_id", model_id), ("checkpoint_id", checkpoint_id),
                                  ("protocol_id", protocol_id)):
            if str(_report.get(_field) or "") != _expected:
                raise ValueError(
                    "report %s names %s=%r, not %r"
                    % (_report_path, _field, _report.get(_field), _expected))
        per_query_capability = _report.get("per_query_capability")
        per_query_state = _report.get("per_query_state")
        _per_query_reason = _report.get("per_query_export_reason") or ""
        _report_read = True
    except Exception as error:
        # A failure here is not "the run is aggregate-only": if the report could not be read, or it
        # belongs to a different run, nothing about this run's per-query axis is known, and the
        # summary refuses with code 5 rather than concluding either way.
        _per_query_reason = f"không đọc được report của lượt chạy: {type(error).__name__}: {error}"
        _report_read = False

# The verification the launcher just ran IS the corroboration the run could not report about itself.
if verified and per_query_state == "PER_QUERY_EXPORTED_UNVERIFIED" \
        and result.get("per_query_artifact_digest"):
    per_query_state = "PER_QUERY_VERIFIED"

reasons = list(result.get("reasons") or []) + list(result.get("evidence_reasons") or [])

if verify_rc != 0 or not verified:
    if proof_unavailable:
        proof_only = all(
            ("proof" in r.lower() or "declared output missing" in r.lower()) and
            ("missing" in r.lower() or "does not exist" in r.lower() or "no such file" in r.lower()) and
            "evidence ledger" not in r.lower() and
            "ledger return_code" not in r.lower() and
            "ledger invocation_status" not in r.lower() and
            "ledger=" not in r.lower() and
            "report digest" not in r.lower() and
            "category metric" not in r.lower() and
            "metric extraction" not in r.lower() and
            "missing aggregate" not in r.lower() and
            "status=" not in r.lower() and
            "return_code=" not in r.lower() and
            "checksum" not in r.lower() and
            "mismatch" not in r.lower() and
            "did not complete" not in r.lower()
            for r in reasons
        )
        if not proof_only:
            reasons_str = "; ".join(str(r) for r in reasons) if reasons else f"verify_run.py trả về mã lỗi {verify_rc}"
            print(f"Xác thực thất bại: Thiếu bằng chứng thực thi khoa học hoặc bằng chứng không hợp lệ!\n"
                  f"Chi tiết: {reasons_str}")
            sys.exit(1)
    else:
        reasons_str = "; ".join(str(r) for r in reasons) if reasons else f"verify_run.py trả về mã lỗi {verify_rc}"
        remedy = ""
        if any("ledger key" in str(r).lower() for r in reasons):
            remedy = ("\nKhắc phục: giữ ĐÚNG MỘT tệp khóa (xoá tệp còn lại), hoặc đặt tường minh "
                      "WORKBENCH_LEDGER_KEY trong workbench/docker/.env — xem "
                      "workbench/docs/PC_ONE_COMMAND_GUIDE.vi.md mục 12 (Sổ cái bằng chứng không xác minh được).")
        print(f"Xác thực thất bại: Thiếu bằng chứng thực thi khoa học hoặc bằng chứng không hợp lệ!\n"
              f"Chi tiết: {reasons_str}{remedy}")
        sys.exit(1)

with open(state_path, "w", encoding="utf-8") as handle:
    json.dump({
        "result": result,
        "per_query_capability": per_query_capability,
        "per_query_state": per_query_state,
        "per_query_reason": _per_query_reason,
        "report_read": _report_read,
        "proof_unavailable": proof_unavailable,
        "model_id": model_id,
        "checkpoint_id": checkpoint_id,
        "protocol_id": protocol_id,
        "run_id": run_id,
        "report_path": result.get("report_path") or "",
    }, handle)
PYGATE
}

# Behaviour 8 & 9: Validation & Vietnamese Summary
validate_and_summarize() {
    local run_id="$1"
    local model="${2:-$MODEL}"
    local checkpoint="${3:-${FINAL_CHECKPOINT:-base_iter0_all_categories}}"
    local protocol="${4:-${FINAL_PROTOCOL:-fashioniq_val_split}}"

    if [[ -z "$run_id" ]]; then
        fail_stage "VERIFICATION" "Không có run_id để xác thực" "" "Đảm bảo quá trình đánh giá đã tạo run_id hợp lệ"
    fi

    local backend_port base
    backend_port="$(env_value WORKBENCH_BACKEND_PORT 8000)"
    base="http://127.0.0.1:${backend_port}"

    say "Xác thực bằng chứng thực thi khoa học và đối chiếu sổ cái qua verify_run.py..."

    local verify_out verify_rc=0
    # Initialised on every path: this carried a value from a previous evaluation in any shell that
    # called the function twice, and the gate would then have read the wrong run's report. Not
    # `local` because the caller reads it after this function returns.
    FINAL_RUN_REPORT_PATH=""
    # The verdict goes to a FILE, never into argv: it is small when the run passes and unbounded when
    # it does not, because `2>&1` folds in whatever the container wrote, and a single argv element is
    # capped at MAX_ARG_STRLEN (128 KiB) regardless of ARG_MAX. A refusal long enough to be useful
    # would be the one that could not be read.
    local verify_out_file
    verify_out_file="$(mktemp "${TMPDIR:-/tmp}/cir_verify_out.XXXXXX")" || \
        fail_stage "VERIFICATION" "Không tạo được tệp tạm để đọc kết quả xác thực" "${TMPDIR:-/tmp}" \
            "Kiểm tra quyền ghi vào thư mục tạm rồi chạy lại ./pc.sh"
    _register_temp "$verify_out_file"
    # A subshell for the same reason the dedup reader uses one: this function calls `fail_stage`
    # when the container is down, and `fail_stage` exits the shell it runs in. The old
    # `$( ... )` contained that exit; a bare redirection does not, and the message would land in the
    # file while the launcher died without ever printing the stage.
    ( run_verifier_in_container --run-id "$run_id" --model "$model" --checkpoint "$checkpoint" --protocol "$protocol" --json ) >"$verify_out_file" 2>&1 || verify_rc=$?
    # The run's own report is the only place its per-query state is recorded, so the summary reads it
    # rather than inferring readiness from the aggregate verdict alone.
    FINAL_RUN_REPORT_PATH="$(python3 -c "
import json, sys
try:
    with open(sys.argv[1], 'r', encoding='utf-8', errors='replace') as _handle:
        out = _handle.read()
except OSError:
    # Nothing to read is not an empty verdict: leave the path empty so the summary's own
    # report-read check refuses rather than concluding this run has no per-query axis.
    out = ''
for line in reversed(out.strip().splitlines()):
    line = line.strip()
    if line.startswith('{') and line.endswith('}'):
        try:
            print((json.loads(line).get('report_path') or ''))
            raise SystemExit(0)
        except SystemExit:
            raise
        except Exception:
            pass
" "$verify_out_file" 2>/dev/null || true)"
    local host_artifacts
    host_artifacts="$(resolve_path "${WORKBENCH_DOCKER_DIR}" "$(env_value HOST_WORKBENCH_ARTIFACTS_ROOT ../../workbench/artifacts)")"
    local verify_state verify_message
    # `mktemp` rather than a name derived from `$$`: a PID-derived path in a shared temp directory is
    # predictable, so another process can pre-create it as a symlink and redirect either the refusal
    # text or the state this summary reads. In this deployment the container's /tmp is private and
    # the worst case is a confusing message, but the file is one line of shell to make unguessable and
    # there is no reason to leave a predictable name in place.
    verify_state="$(mktemp "${TMPDIR:-/tmp}/cir_verify_state.XXXXXX")"
    verify_message="$(mktemp "${TMPDIR:-/tmp}/cir_verify_msg.XXXXXX")"
    _register_temp "$verify_state"
    _register_temp "$verify_message"
    VERIFY_STATE_FILE="$verify_state"
    if ! verify_gate "$verify_out_file" "$verify_rc" "$model" "$checkpoint" "$protocol" "$run_id" \
            "${FINAL_RUN_PROOF_UNAVAILABLE:-0}" "${FINAL_RUN_REPORT_PATH:-}" "$host_artifacts" >"$verify_message"; then
        # The text is CAPTURED before the file is removed. `cat` the file, `rm` it, then `cat` it
        # again inside the fail_stage argument read an empty file: the operator got the refusal in
        # the stage block with NO reason attached, which is the one thing this branch exists to
        # deliver. Both the visible line and the failure detail come from the captured text.
        local verify_refusal
        verify_refusal="$(cat "$verify_message")"
        rm -f "$verify_state" "$verify_message" "$verify_out_file"
        printf '%s\n' "$verify_refusal"
        fail_stage "VERIFICATION" "$verify_refusal" "verify_run.py (trong container)" \
            "Đọc phần 'Chi tiết' ở trên, sửa nguyên nhân, rồi chạy lại ./pc.sh (dùng ./pc.sh --verify-only nếu bạn chỉ muốn kiểm chứng lại năng lực GPU trước khi tốn thời gian đánh giá)"
    fi
    rm -f "$verify_message"

    # The verifier has spoken, so the index can now reflect the FINAL verification state. The
    # rebuild used to run before this point and swallowed its own failure, so a run could be fully
    # evaluated and verified yet absent from the API and the UI.
    rebuild_and_verify_index "$run_id" "$model" "$checkpoint" "$protocol"

    python3 - "${INDEX_EVIDENCE_STATUS:-}" "$verify_state" <<'PYDOC'
import json, sys

# The state file is DATA. Passing its path as the program made python execute the JSON as a script.
with open(sys.argv[2], encoding="utf-8") as handle:
    _state = json.load(handle)

result = _state["result"] or {}
per_query_capability = _state["per_query_capability"]
per_query_state = _state["per_query_state"]
_per_query_reason = _state["per_query_reason"]
_report_read = _state["report_read"]
proof_unavailable = _state["proof_unavailable"]
model_id = _state["model_id"]
checkpoint_id = _state["checkpoint_id"]
protocol_id = _state["protocol_id"]
run_id = _state["run_id"]
_index_evidence_status = sys.argv[1] if len(sys.argv) > 1 else ""
# Extract verified details
report_path = result.get("report_path") or "N/A"
identity = result.get("identity") or {}
metrics = result.get("metrics") or {}
cat_metrics = metrics.get("category") or {}
obs_metrics = metrics.get("observed") or {}
gate = result.get("gate") or {}
gate_status = gate.get("status") or "NOT_EVALUATED"

commit = identity.get("git_commit") or "N/A"
adaptation = identity.get("runtime_adaptation") or "None"

# The two claims about one run are checked BEFORE anything is reported. Printing a conclusion and
# then contradicting it in the same output is a report that cannot be trusted about itself.
if per_query_state == "PER_QUERY_VERIFIED" and _index_evidence_status != "PER_QUERY_VERIFIED":
    print(f"LỖI: phần tổng hợp kết luận ĐÃ CHỨNG THỰC nhưng chỉ mục đang công bố "
          f"{_index_evidence_status!r} cho đúng lượt chạy này.")
    print("      API và UI đọc chỉ mục, nên hai nơi đang nói hai điều khác nhau về cùng một lượt chạy.")
    print("      Hành động: chạy lại './pc.sh' — thao tác đó XÁC THỰC rồi DỰNG LẠI chỉ mục.")
    print("      '--verify-only' chỉ kiểm chứng GPU: nó KHÔNG xác thực lượt chạy và KHÔNG dựng lại chỉ mục.")
    sys.exit(6)
if per_query_state in ("PER_QUERY_EXPORT_FAILED", "PER_QUERY_EXPORTED_UNVERIFIED") \
        and _index_evidence_status == "PER_QUERY_VERIFIED":
    print(f"LỖI: chỉ mục công bố ĐÃ CHỨNG THỰC trong khi phần tổng hợp kết luận "
          f"{per_query_state}. Một trong hai đã đọc sai trạng thái.")
    sys.exit(6)

def fmt_val(v):
    if v is None:
        return "N/A"
    try:
        return f"{float(v):.2f}"
    except Exception:
        return str(v)

print("")
print("=" * 80)
print("KẾT QUẢ ĐÁNH GIÁ CHÍNH THỨC — CIR FAILURE ANALYSIS WORKBENCH")
print("=" * 80)
print(f"Mô hình (Model)          : {model_id.upper()}")
print(f"Checkpoint Bundle        : {checkpoint_id}")
print(f"Protocol                 : {protocol_id}")
print(f"Run ID                   : {run_id}")
print(f"Git Commit               : {commit}")
print(f"Runtime Adaptation       : {adaptation}")
print(f"Thẩm định bài báo (Gate) : {gate_status}")
print(f"Trạng thái thực thi      : {(result.get('execution') or {}).get('execution_status') or 'EXECUTION_COMPLETED'} "
      f"(return_code={(result.get('execution') or {}).get('return_code', 0)})")
if proof_unavailable:
    print(f"Xác thực Sổ cái (Ledger) : HỢP LỆ SỔ CÁI (TỆP PROOF KHÔNG CÒN TỒN TẠI)")
else:
    print(f"Xác thực Sổ cái (Ledger) : HỢP LỆ (CORROBORATED) — verify_run.py: verified=True")
print("-" * 80)
print("--- KẾT QUẢ ĐO ĐẠC THỰC TẾ (ACTUAL MEASURED METRICS) ---")
for cat in ["dress", "shirt", "toptee"]:
    m = cat_metrics.get(cat) or {}
    r10 = fmt_val(m.get("r10"))
    r50 = fmt_val(m.get("r50"))
    print(f"{cat.capitalize():<8}: Recall@10 = {r10:>6} | Recall@50 = {r50:>6}")
print("-" * 80)
print("Chỉ số tổng hợp (Macro Aggregate):")
print(f"  Recall@10 : {fmt_val(obs_metrics.get('r10'))}")
print(f"  Recall@50 : {fmt_val(obs_metrics.get('r50'))}")
print(f"  Mean      : {fmt_val(obs_metrics.get('mean'))}")
print("-" * 80)
if not _report_read:
    # Unknown is not "aggregate-only". Stating the latter here would tell the operator a model has no
    # observer when the truth is that this launcher could not read what the run recorded.
    print(f"Phân tích theo truy vấn  : KHÔNG XÁC ĐỊNH ĐƯỢC — {_per_query_reason}")
elif per_query_state == "CAPABILITY_ABSENT" or per_query_capability is None:
    print(f"Phân tích theo truy vấn  : KHÔNG HỖ TRỢ CHO MÔ HÌNH NÀY (aggregate-only; không phải lỗi)")
elif per_query_state == "PER_QUERY_EXPORTED_UNVERIFIED":
    print(f"Phân tích theo truy vấn  : ĐÃ XUẤT Top-K — CHỜ TRÌNH XÁC THỰC chứng thực")
elif per_query_state == "PER_QUERY_VERIFIED":
    print(f"Phân tích theo truy vấn  : ĐÃ XUẤT + ĐÃ CHỨNG THỰC (Top-K + phân tích khả dụng)")
else:
    print(f"Phân tích theo truy vấn  : THẤT BẠI — {per_query_capability} được khai báo nhưng không")
    print(f"                           có artifact Top-K nào được xuất. Lý do: {_per_query_reason}")
    print(f"                           Kết quả TỔNG HỢP ở trên vẫn hợp lệ và đã được lưu.")
if _index_evidence_status == "AGGREGATE_ONLY_EXECUTION_AUTHENTICATED_PROOF_UNAVAILABLE":
    print("Trạng thái chỉ mục: ĐÃ XÁC THỰC LƯỢT GỌI — CHƯA CHỨNG THỰC KHOA HỌC (thiếu completion proof).")
    print("                   Sổ cái xác nhận LƯỢT GỌI này đã chạy và đã ghi đúng các byte này; nhưng")
    print("                   tệp completion proof không còn, nên xác thực đầy đủ KHÔNG chạy được và")
    print("                   lượt chạy KHÔNG được coi là bằng chứng nghiên cứu đã chứng thực.")
    print("                   Báo cáo tổng hợp vẫn được phục vụ (đó là ghi nhận một sự kiện đã xảy ra).")
elif _index_evidence_status.startswith("PROOF_UNAVAILABLE"):
    print(f"Trạng thái chỉ mục: {_index_evidence_status} — tệp completion proof không còn tồn tại, nên")
    print("                   xác thực đầy đủ không chạy được; API vẫn xác thực được LƯỢT GỌI")
    print("                   (execution_authenticated) và đã phục vụ báo cáo tổng hợp.")
elif _index_evidence_status == "AGGREGATE_ONLY_VERIFIED":
    print("Trạng thái chỉ mục: AGGREGATE_ONLY_VERIFIED — mô hình này không khai báo trình quan sát,")
    print("                   nên KHÔNG có bản ghi per-query nào để chỉ mục nắm giữ (đúng theo thiết kế).")
    print("                   Báo cáo tổng hợp đã được API chứng thực tại /api/benchmark-runs.")
else:
    print(f"Trạng thái chỉ mục: {_index_evidence_status or 'KHÔNG XÁC ĐỊNH ĐƯỢC'}")
print(f"File báo cáo     : {report_path}")
print("-" * 80)
if per_query_capability and per_query_state in (None, "PER_QUERY_EXPORT_FAILED"):
    print("KẾT LUẬN: TỔNG HỢP ĐẠT — PILOT ĐẦY ĐỦ CHƯA SẴN SÀNG.")
    print("  Đánh giá chính thức đã hoàn tất và được chứng thực, nhưng mô hình này KHAI BÁO có")
    print("  trình quan sát theo truy vấn và nó đã KHÔNG xuất được artifact Top-K. Phân tích")
    print("  thất bại theo từng truy vấn (Top-K, so sánh mô hình, Jaccard) CHƯA dùng được.")
elif per_query_capability and per_query_state == "PER_QUERY_EXPORTED_UNVERIFIED":
    print("KẾT LUẬN: TỔNG HỢP ĐẠT — PILOT ĐẦY ĐỦ chờ chứng thực artifact Top-K.")
else:
    if _index_evidence_status == "AGGREGATE_ONLY_EXECUTION_AUTHENTICATED_PROOF_UNAVAILABLE" \
            or _index_evidence_status.startswith("PROOF_UNAVAILABLE"):
        print("KẾT LUẬN: LƯỢT GỌI ĐÃ ĐƯỢC XÁC THỰC — CHƯA PHẢI BẰNG CHỨNG ĐÃ CHỨNG THỰC (tệp proof không còn).")
    else:
        print("KẾT LUẬN: SẴN SÀNG CHO PILOT (theo khả năng đã khai báo của mô hình).")
    # The evidence line must agree with the conclusion two lines above it. Printing "aggregate
    # evidence: certified" for a run whose proof is gone made the report contradict itself in the
    # same breath -- the defect class this summary has been fixed for before, and the reason the
    # conclusion is computed before anything is printed.
    if _index_evidence_status == "AGGREGATE_ONLY_EXECUTION_AUTHENTICATED_PROOF_UNAVAILABLE" \
            or _index_evidence_status.startswith("PROOF_UNAVAILABLE"):
        print("  Bằng chứng tổng hợp: LƯỢT GỌI đã được sổ cái xác thực; khoa học CHƯA được chứng thực.")
    else:
        print("  Bằng chứng tổng hợp: đã chứng thực và API phục vụ được lượt chạy này.")
    if _index_evidence_status == "AGGREGATE_ONLY_EXECUTION_AUTHENTICATED_PROOF_UNAVAILABLE":
        print("  Chỉ mục DuckDB: không có bản ghi per-query (mô hình không có trình quan sát).")
        print("  Xác thực khoa học: CHƯA đạt — không có completion proof để chạy xác thực đầy đủ.")
    elif _index_evidence_status.startswith("PROOF_UNAVAILABLE"):
        print("  Chỉ mục DuckDB: lượt chạy này KHÔNG được promote lên PER_QUERY_VERIFIED (tệp proof")
        print("  không còn nên không có receipt nào gắn được với artifact). Đây KHÔNG phải pilot đầy đủ.")
    elif _index_evidence_status == "AGGREGATE_ONLY_VERIFIED":
        # Saying "the DuckDB index holds this run" here would be false: the index holds canonical
        # PER-QUERY records, and this model declares no observer, so there is nothing of its kind
        # to hold. The aggregate view is where its evidence lives, and the summary must send the
        # operator there rather than to a page that will be empty.
        print("  Chỉ mục DuckDB: KHÔNG chứa lượt chạy này, và đúng là như vậy — xem báo cáo tổng hợp")
        print("  tại /api/benchmark-runs (tab Models của UI). Phân tích theo truy vấn không áp dụng")
        print("  cho mô hình này: không có Top-K nào được lưu.")
    else:
        print("  Chỉ mục DuckDB: đã chứa lượt chạy này (API và UI đọc được kết quả).")
        print("  Trạng thái bằng chứng per-query ở trên.")
print("Ghi chú khoa học :")
print("  Độ lệch số học so với bài báo gốc (nếu có) phản ánh kết quả đo đạc thực")
print("  nghiệm trên runtime của thiết bị, là phát hiện nghiên cứu (research finding).")
print("================================================================================")

if not _report_read:
    # The launcher cannot say a declared observer succeeded if it could not read the verdict.
    sys.exit(5)
# The index is what the API and the UI read, so the launcher's own conclusion and the index's
# published state must be the SAME claim. Two places disagreeing about one run means the operator is
# told one thing and the interface shows another.
if _index_evidence_status == "AGGREGATE_ONLY_EXECUTION_AUTHENTICATED_PROOF_UNAVAILABLE":
    # Not 0. The invocation is authenticated and the numbers are recorded, but nothing verified them,
    # and a zero here would present an unverified run as a passed pilot.
    sys.exit(7)
if per_query_capability and per_query_state in (None, "PER_QUERY_EXPORT_FAILED"):
    sys.exit(3)
if per_query_capability and per_query_state == "PER_QUERY_EXPORTED_UNVERIFIED":
    # 4 also covers the proof-unavailable run of an observer model: its artifact can never be
    # promoted, so the full pilot is not ready and the exit code says so instead of 0.
    sys.exit(4)
PYDOC
    # The summary has read it; nothing else consumes the state file or the captured verdict.
    rm -f "$verify_state" "$verify_out_file"
}

# Main Execution Flow
main() {
    validate_env_file

    # If not in doctor mode, initialize/validate configuration first
    if (( DOCTOR_MODE == 0 )); then
        configure_environment
        local b_port f_port
        b_port="$(env_value WORKBENCH_BACKEND_PORT 8000)"
        f_port="$(env_value WORKBENCH_FRONTEND_PORT 5173)"
        validate_env WORKBENCH_BACKEND_PORT "$b_port"
        validate_env WORKBENCH_FRONTEND_PORT "$f_port"
    else
        if [[ -f "${WORKBENCH_DOCKER_DIR}/.env" ]]; then
            local b_port f_port
            b_port="$(env_value WORKBENCH_BACKEND_PORT 8000)"
            f_port="$(env_value WORKBENCH_FRONTEND_PORT 5173)"
            validate_env WORKBENCH_BACKEND_PORT "$b_port"
            validate_env WORKBENCH_FRONTEND_PORT "$f_port"
        fi
    fi

    # 1. Preflight
    if ! run_preflight "$DOCTOR_MODE"; then
        exit 1
    fi
    if (( DOCTOR_MODE == 1 )); then
        exit 0
    fi

    # 2. Docker stack up & readiness
    start_docker_and_wait_readiness

    # 3. Preparation
    if (( VERIFY_ONLY == 0 )); then
        run_model_preparation
    fi

    # 4. GPU Verification
    verify_gpu_in_container
    if (( VERIFY_ONLY == 1 )); then
        say ""
        say "================================================================================"
        say "KIỂM CHỨNG GPU HOÀN TẤT THÀNH CÔNG (--verify-only)"
        say "Mô hình: ${MODEL}"
        say "Năng lực GPU đã được xác nhận và ghi nhận an toàn."
        say "================================================================================"
        exit 0
    fi

    # 5. Evaluation
    execute_evaluation

    # 6. Validation & Summary
    validate_and_summarize "$FINAL_RUN_ID" "$MODEL" "${FINAL_CHECKPOINT:-base_iter0_all_categories}" "${FINAL_PROTOCOL:-fashioniq_val_split}"
}

main "$@"
