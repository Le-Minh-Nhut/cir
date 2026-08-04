from __future__ import annotations

import platform
import random
import subprocess
import sys
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    if seed < 0:
        raise ValueError(f"Seed must be non-negative, received: {seed}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_torch_runtime(*, deterministic: bool, benchmark: bool) -> None:

    # deterministic=True: ưu tiên kết quả ổn định.
    # benchmark=True: cuDNN thử nhiều thuật toán rồi chọn cái nhanh nhất.
    if deterministic and benchmark:
        raise ValueError(
            "runtime.deterministic and runtime.benchmark cannot both be true."
        )

    torch.use_deterministic_algorithms(
        deterministic,
        warn_only=True,
    )

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = benchmark


def resolve_device(device_name: str, accelerator_index: int = 0) -> torch.device:
    normalized_name = device_name.strip().lower()

    if accelerator_index < 0:
        raise ValueError(
            "accelerator_index must be non-negative, "
            f"received: {accelerator_index}"
        )

    if normalized_name == "auto":
        if torch.cuda.is_available():
            return torch.device(f"cuda:{accelerator_index}")

        if torch.backends.mps.is_available():
            return torch.device("mps")

        return torch.device("cpu")

    if normalized_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but no CUDA device is available."
            )

        device_count = torch.cuda.device_count()
        if accelerator_index >= device_count:
            raise ValueError(
                f"CUDA device index {accelerator_index} is invalid. "
                f"Available CUDA devices: {device_count}."
            )

        return torch.device(f"cuda:{accelerator_index}")

    if normalized_name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS was requested but is not available."
            )

        return torch.device("mps")

    if normalized_name == "cpu":
        return torch.device("cpu")

    raise ValueError(
        f"Unsupported device '{device_name}'. "
        "Expected one of: auto, cuda, mps, cpu."
    )

# lấy mã hash của commit để đưa cho checkpoint 
def get_git_commit() -> str | None:
    try:
        result = subprocess.run(
            [
                "git", # # chương trình cần chạy
                "rev-parse", # đối số thứ nhất
                "HEAD" # đối số thứ hai
            ],
            check=True, # Nếu lệnh terminal thất bại, Python sẽ ném lỗi
            capture_output=True, # Giữ lại output của lệnh thay vì in trực tiếp ra terminal
            text=True, # Yêu cầu kết quả trả về dưới dạng chuỗi Python str
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    commit = result.stdout.strip()
    return commit or None


def collect_environment_metadata() -> dict[str, Any]:
    metadata = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_available": torch.backends.cudnn.is_available(),
        "cudnn_version": torch.backends.cudnn.version(),
        "mps_available": torch.backends.mps.is_available(),
        "git_commit": get_git_commit(),
    }

    if torch.cuda.is_available():
        metadata["cuda_device_count"] = torch.cuda.device_count()
        metadata["cuda_devices"] = [ # lấy tên từng gpu
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]
    else:
        metadata["cuda_device_count"] = 0
        metadata["cuda_devices"] = []

    return metadata