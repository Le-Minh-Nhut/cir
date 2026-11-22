"""Shared training utilities: seeding, VRAM batch-size auto-probing, metric selection, logging."""

from __future__ import annotations

import gc
import json
import os
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from workbench.benchmarks.schema import DEFAULT_SPLIT_TAG, REPO_ROOT


def seed_everything(seed: int = 42) -> None:
    """Set deterministic seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "unknown"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def compute_val_selection_score(dataset: str, metrics: Dict[str, Any]) -> Tuple[str, float]:
    """Compute the canonical scalar validation score used for checkpoint selection & early stopping.

    - EDIR: ``macro_mean_R1_R5_R10`` = mean(macro R@1, macro R@5, macro R@10).
      Smooth on 497 val queries while weighting all 15 subcategories equally.
    - PinPoint: ``mAP@10`` (official primary ranking metric in PinPoint README).
    """
    ds = dataset.strip().lower()
    if ds == "edir":
        m = metrics["macro_subcategory"]
        score = float((m["Recall@1"] + m["Recall@5"] + m["Recall@10"]) / 3.0)
        return "macro_mean_R1_R5_R10", score
    if ds == "pinpoint":
        score = float(metrics["overall"]["mAP@10"])
        return "mAP@10", score
    raise ValueError(f"Unsupported dataset: {dataset}")


def probe_batch_size_and_accum(
    probe_step_fn: Callable[[int], None],
    default_effective_bs: int = 16,
    allow_upgrade_to_32: bool = True,
    max_micro_bs: int = 64,
    upgrade_vram_frac: float = 0.70,
    safe_vram_frac: float = 0.78,
) -> Tuple[int, int, List[Dict[str, Any]]]:
    """Probe real GPU forward+backward VRAM and pick the largest safe (micro_batch_size, grad_accum).

    Policy (leaving >=22-30% headroom for AdamW optimizer states & cuBLAS workspace):
      1. If ``allow_upgrade_to_32``, probes downward from ``max_micro_bs`` (e.g. 64 -> 32)
         and selects the largest batch size with peak VRAM <= ``upgrade_vram_frac`` (70%).
      2. Otherwise if ``bs=16`` fits safely (peak VRAM <= ``safe_vram_frac`` 78%), uses ``(16, 1)``.
      3. Otherwise falls back to smaller micro-batch (8 -> 4 -> 2) with gradient accumulation
         so ``micro_bs * grad_accum == default_effective_bs`` (16).
    """
    if not torch.cuda.is_available():
        return default_effective_bs, 1, [{"micro_bs": default_effective_bs, "status": "cpu_fallback"}]

    total_bytes = torch.cuda.get_device_properties(0).total_memory
    total_mib = total_bytes / (1024**2)

    candidates: List[Tuple[int, int, float]] = []
    if allow_upgrade_to_32:
        for bs in (128, 96, 64, 48, 40, 32, 24):
            if bs <= max_micro_bs:
                candidates.append((bs, 1, upgrade_vram_frac))
    candidates.append((16, 1, safe_vram_frac))
    candidates.append((8, max(1, default_effective_bs // 8), safe_vram_frac))
    candidates.append((4, max(1, default_effective_bs // 4), safe_vram_frac))
    candidates.append((2, max(1, default_effective_bs // 2), safe_vram_frac))

    probe_log: List[Dict[str, Any]] = []
    for micro_bs, grad_accum, max_frac in candidates:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        try:
            probe_step_fn(micro_bs)
            torch.cuda.synchronize()
            peak_bytes = torch.cuda.max_memory_allocated()
            peak_mib = peak_bytes / (1024**2)
            peak_frac = peak_bytes / float(total_bytes)
            dt = time.time() - t0
            ok = peak_frac <= max_frac
            entry = {
                "micro_bs": micro_bs,
                "grad_accum": grad_accum,
                "effective_bs": micro_bs * grad_accum,
                "peak_vram_mib": round(peak_mib, 1),
                "total_vram_mib": round(total_mib, 1),
                "peak_vram_frac": round(peak_frac, 4),
                "max_allowed_frac": max_frac,
                "probe_sec": round(dt, 2),
                "status": "SELECTED" if ok else "EXCEEDS_HEADROOM",
            }
            probe_log.append(entry)
            gc.collect()
            torch.cuda.empty_cache()
            if ok:
                return micro_bs, grad_accum, probe_log
        except torch.cuda.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
            probe_log.append(
                {
                    "micro_bs": micro_bs,
                    "grad_accum": grad_accum,
                    "effective_bs": micro_bs * grad_accum,
                    "status": "OOM",
                }
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                gc.collect()
                torch.cuda.empty_cache()
                probe_log.append(
                    {
                        "micro_bs": micro_bs,
                        "grad_accum": grad_accum,
                        "effective_bs": micro_bs * grad_accum,
                        "status": "OOM",
                    }
                )
            else:
                raise

    raise RuntimeError(f"All batch sizes OOMed during VRAM probe: {probe_log}")


def default_run_dir(dataset: str, model_id: str, split_tag: str = DEFAULT_SPLIT_TAG) -> Path:
    d = REPO_ROOT / "workbench" / "artifacts" / "benchmark_runs" / dataset.lower() / split_tag / model_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
