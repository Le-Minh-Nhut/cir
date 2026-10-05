"""Resolve local operator paths and ports without changing process environment."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "workbench.env"
_DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]
_KEYS = {
    "CIR_REPO_ROOT",
    "CIR_DATA_ROOT",
    "FASHIONIQ_ROOT",
    "WORKBENCH_HOST",
    "WORKBENCH_BACKEND_PORT",
    "WORKBENCH_FRONTEND_PORT",
    "WORKBENCH_CHECKPOINT_ROOT",
    "WORKBENCH_RESULTS_ROOT",
    "WORKBENCH_THIRD_PARTY_ROOT",
}


@dataclass(frozen=True)
class WorkbenchConfig:
    CIR_REPO_ROOT: Path
    CIR_DATA_ROOT: Path
    FASHIONIQ_ROOT: Path
    WORKBENCH_HOST: str
    WORKBENCH_BACKEND_PORT: int
    WORKBENCH_FRONTEND_PORT: int
    WORKBENCH_CHECKPOINT_ROOT: Path
    WORKBENCH_RESULTS_ROOT: Path
    WORKBENCH_THIRD_PARTY_ROOT: Path


def _read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, separator, value = stripped.partition("=")
        if not separator or not key.strip() or any(char.isspace() for char in key):
            raise ValueError(f"malformed {path}:{line_number}: expected KEY=VALUE")
        if key not in _KEYS:
            raise ValueError(f"unknown setting {path}:{line_number}: {key}")
        values[key] = value.strip()
    return values


def _path(value: str, repo_root: Path) -> Path:
    path = Path(value).expanduser()
    return (repo_root / path if not path.is_absolute() else path).resolve()


def _port(name: str, value: str) -> int:
    try:
        port = int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer port") from error
    if not 1 <= port <= 65535:
        raise ValueError(f"{name} must be between 1 and 65535")
    return port


def resolve_config() -> WorkbenchConfig:
    """Return file settings overridden by matching environment variables."""
    values = _read_env(_CONFIG_PATH)
    values.update({key: os.environ[key] for key in _KEYS if key in os.environ})
    repo_root = _path(values.get("CIR_REPO_ROOT", str(_DEFAULT_REPO_ROOT)), _DEFAULT_REPO_ROOT)
    data_root = _path(values.get("CIR_DATA_ROOT", "data"), repo_root)
    workbench_root = repo_root / "workbench"
    return WorkbenchConfig(
        CIR_REPO_ROOT=repo_root,
        CIR_DATA_ROOT=data_root,
        FASHIONIQ_ROOT=_path(values.get("FASHIONIQ_ROOT", str(data_root / "FashionIQ")), repo_root),
        WORKBENCH_HOST=values.get("WORKBENCH_HOST", "127.0.0.1"),
        WORKBENCH_BACKEND_PORT=_port("WORKBENCH_BACKEND_PORT", values.get("WORKBENCH_BACKEND_PORT", "8000")),
        WORKBENCH_FRONTEND_PORT=_port("WORKBENCH_FRONTEND_PORT", values.get("WORKBENCH_FRONTEND_PORT", "5173")),
        WORKBENCH_CHECKPOINT_ROOT=_path(values.get("WORKBENCH_CHECKPOINT_ROOT", str(workbench_root / "artifacts" / "checkpoints")), repo_root),
        WORKBENCH_RESULTS_ROOT=_path(values.get("WORKBENCH_RESULTS_ROOT", str(workbench_root / "artifacts" / "results")), repo_root),
        WORKBENCH_THIRD_PARTY_ROOT=_path(values.get("WORKBENCH_THIRD_PARTY_ROOT", str(workbench_root / "third_party")), repo_root),
    )


_CONFIG = resolve_config()
CIR_REPO_ROOT = _CONFIG.CIR_REPO_ROOT
CIR_DATA_ROOT = _CONFIG.CIR_DATA_ROOT
FASHIONIQ_ROOT = _CONFIG.FASHIONIQ_ROOT
WORKBENCH_HOST = _CONFIG.WORKBENCH_HOST
WORKBENCH_BACKEND_PORT = _CONFIG.WORKBENCH_BACKEND_PORT
WORKBENCH_FRONTEND_PORT = _CONFIG.WORKBENCH_FRONTEND_PORT
WORKBENCH_CHECKPOINT_ROOT = _CONFIG.WORKBENCH_CHECKPOINT_ROOT
WORKBENCH_RESULTS_ROOT = _CONFIG.WORKBENCH_RESULTS_ROOT
WORKBENCH_THIRD_PARTY_ROOT = _CONFIG.WORKBENCH_THIRD_PARTY_ROOT
