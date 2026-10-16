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
SELF_CHECK=0
FINAL_RUN_ID=""
FINAL_CHECKPOINT=""
FINAL_PROTOCOL=""

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
            if [[ "$v" == \"*\" ]]; then
                v="${v:1:${#v}-2}"
            elif [[ "$v" == \'*\' ]]; then
                v="${v:1:${#v}-2}"
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
  --verify-only      Verify model GPU runtime in container and report (no evaluation)
  --force            Force re-evaluation even if completed corroborated run exists
  --timeout <sec>    Maximum wait time for evaluation job in seconds (default: 3600)
  --no-gpu           Disable GPU override (WORKBENCH_ENABLE_GPU=0)
  --dataset <path>   Override FashionIQ dataset path
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
                    config_satisfies=0
                    mismatch_reasons+=("GPU override không có hiệu lực hoặc container không có CUDA visibility")
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
                config_satisfies=0
                mismatch_reasons+=("Runtime adaptation không khớp (container='${cont_adapt}', yêu cầu='${expected_adaptation}')")
            fi

            # 3. Dataset mount
            if ! docker exec cir-workbench test -d /data/FashionIQ/images 2>/dev/null; then
                config_satisfies=0
                mismatch_reasons+=("Dataset mount /data/FashionIQ/images không tồn tại hoặc không thể truy cập")
            fi

            # 4. Model environment
            if ! docker exec cir-workbench python -c "import sys; sys.exit(0)" 2>/dev/null; then
                config_satisfies=0
                mismatch_reasons+=("Môi trường thực thi trong container không phản hồi")
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

                say "Không có tác vụ đang chạy. Khởi chạy lại Docker stack qua run.sh up để cấu hình đúng..."
                "${RUN_SH}" up
                container_running=0
            fi
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
    local protocol checkpoint
    local model_info=""
    local attempt
    for attempt in 1 2; do
        if model_info="$(curl -fsS --max-time 300 "${base}/api/models" 2>/dev/null | python3 -c "
import json, sys

model_id = sys.argv[1]
try:
    models = json.load(sys.stdin)
except Exception as e:
    sys.exit(f'models endpoint error: {e}')

m = next((item for item in models if item.get('model_id') == model_id), None)
if not m:
    sys.exit(f'Mô hình {model_id} không tìm thấy trong /api/models')

protocol = m.get('native_protocol') or 'fashioniq_val_split'
bundles = m.get('checkpoint_bundles') or []
if bundles and bundles[0].get('bundle_id'):
    checkpoint = bundles[0]['bundle_id']
elif m.get('checkpoints'):
    checkpoint = m['checkpoints'][0].get('checkpoint_id')
else:
    checkpoint = 'base_iter0_all_categories'
print(protocol, checkpoint)
" "$MODEL" 2>/dev/null)"; then
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

    read -r protocol checkpoint <<<"$model_info"
    FINAL_CHECKPOINT="$checkpoint"
    FINAL_PROTOCOL="$protocol"
    say "Mô hình: ${MODEL} | Checkpoint: ${checkpoint} | Protocol: ${protocol} | top_k: 200"

    # Job Dedup: Check if a completed corroborated run exists in /api/runs or /api/benchmark-runs
    local target_run_id=""
    if (( FORCE != 1 )); then
        local runs_json b_runs_json
        runs_json="$(curl -s --max-time 10 "${base}/api/runs" 2>/dev/null || echo "[]")"
        b_runs_json="$(curl -s --max-time 10 "${base}/api/benchmark-runs" 2>/dev/null || echo "[]")"

        local candidate_runs
        candidate_runs="$(python3 -c "
import json, sys

model_id = sys.argv[1]
checkpoint_id = sys.argv[2]
protocol_id = sys.argv[3]
runs_text = sys.argv[4]
b_runs_text = sys.argv[5]

all_runs = []
for t in (runs_text, b_runs_text):
    try:
        data = json.loads(t)
        if isinstance(data, list):
            all_runs.extend(data)
    except Exception:
        pass

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
" "$MODEL" "$checkpoint" "$protocol" "$runs_json" "$b_runs_json" 2>/dev/null || true)"

        local verify_script="${REPO_ROOT}/workbench/scripts/verify_run.py"
        if [[ ! -f "$verify_script" ]]; then
            verify_script="${SCRIPT_DIR}/workbench/scripts/verify_run.py"
        fi

        if [[ -n "$candidate_runs" && -f "$verify_script" ]]; then
            export CIR_REPO_ROOT="$REPO_ROOT"
            export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:$PYTHONPATH}"
            while IFS= read -r cand_id; do
                [[ -z "$cand_id" ]] && continue
                local v_rc=0 v_out
                v_out="$(python3 "$verify_script" --run-id "$cand_id" --model "$MODEL" --checkpoint "$checkpoint" --protocol "$protocol" --json 2>&1)" || v_rc=$?
                if (( v_rc == 0 )) && python3 -c "import json, sys; res = json.loads(sys.argv[1]); sys.exit(0 if res.get('verified') else 1)" "$v_out" 2>/dev/null; then
                    target_run_id="$cand_id"
                    break
                fi
            done <<< "$candidate_runs"
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
        run_id_from_job="$(curl -s --max-time 10 "${base}/api/benchmark-runs" 2>/dev/null | python3 -c "
import json, sys
try:
    runs = json.load(sys.stdin)
    if runs:
        print(runs[-1].get('run_id', ''))
except Exception:
    pass
" 2>/dev/null || true)"
    fi

    FINAL_RUN_ID="$run_id_from_job"
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

    say "Tái xây dựng chỉ mục DuckDB (/api/index/rebuild)..."
    curl -fsS --max-time 60 -X POST "${base}/api/index/rebuild" >/dev/null 2>&1 || true

    say "Xác thực bằng chứng thực thi khoa học và đối chiếu sổ cái qua verify_run.py..."

    local verify_script="${REPO_ROOT}/workbench/scripts/verify_run.py"
    if [[ ! -f "$verify_script" ]]; then
        verify_script="${SCRIPT_DIR}/workbench/scripts/verify_run.py"
    fi

    if [[ ! -f "$verify_script" ]]; then
        fail_stage "VERIFICATION" "Không tìm thấy công cụ xác thực độc lập verify_run.py tại ${verify_script}" "$verify_script" "Kiểm tra sự tồn tại của workbench/scripts/verify_run.py"
    fi

    export CIR_REPO_ROOT="$REPO_ROOT"
    export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:$PYTHONPATH}"

    local verify_out verify_rc=0
    verify_out="$(python3 "$verify_script" --run-id "$run_id" --model "$model" --checkpoint "$checkpoint" --protocol "$protocol" --json 2>&1)" || verify_rc=$?

    python3 - <<'PYDOC' "$verify_out" "$verify_rc" "$model" "$checkpoint" "$protocol" "$run_id"
import json, sys

verify_out = sys.argv[1]
verify_rc = int(sys.argv[2])
model_id = sys.argv[3]
checkpoint_id = sys.argv[4]
protocol_id = sys.argv[5]
run_id = sys.argv[6]

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
    sys.exit(f"Lỗi: verify_run.py không trả về JSON hợp lệ (exit code {verify_rc}):\n{verify_out}")

verified = bool(result.get("verified", False))
reasons = result.get("reasons") or []

if verify_rc != 0 or not verified:
    reasons_str = "; ".join(str(r) for r in reasons) if reasons else f"verify_run.py trả về mã lỗi {verify_rc}"
    err_msg = f"Xác thực thất bại: Thiếu bằng chứng thực thi khoa học hoặc bằng chứng không hợp lệ!\nChi tiết: {reasons_str}"
    print(err_msg)
    sys.exit(err_msg)

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
print(f"File báo cáo     : {report_path}")
print("Ghi chú khoa học :")
print("  Độ lệch số học so với bài báo gốc (nếu có) phản ánh kết quả đo đạc thực")
print("  nghiệm trên runtime của thiết bị, là phát hiện nghiên cứu (research finding).")
print("================================================================================")
PYDOC
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
