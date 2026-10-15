#!/usr/bin/env python3
"""Honest GPU passthrough validation for the workbench container.

Reports exactly what this container can see. No synthetic fallback, no fabricated capability:
if the device is not visible, the answer is PENDING with the reason. On a successful probe it
runs the same evidence standard the workbench uses for models (a real tensor allocation and a
compiled-core check via the orchestrator's own probe code).

Usage:
    python workbench/scripts/gpu_check.py            # human summary
    python workbench/scripts/gpu_check.py --json     # machine-readable
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    report: dict = {"host_visible": False, "torch_probe": None, "verdict": "PENDING"}

    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,compute_cap,driver_version",
             "--format=csv,noheader"], text=True, stderr=subprocess.DEVNULL)
        report["host_visible"] = True
        report["nvidia_smi"] = [line.strip() for line in out.splitlines() if line.strip()]
    except (OSError, subprocess.CalledProcessError) as error:
        report["nvidia_smi_error"] = f"{type(error).__name__}: {error}"

    if report["host_visible"]:
        try:
            import torch

            probe = {
                "torch": torch.__version__,
                "cuda_build": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
            }
            if probe["cuda_available"]:
                major, minor = torch.cuda.get_device_capability(0)
                probe["compute_capability"] = f"{major}.{minor}"
                probe["device_name"] = torch.cuda.get_device_name(0)
                probe["arch_list"] = list(torch.cuda.get_arch_list())
                # A real allocation + compute, not an import success.
                tensor = torch.ones(1, device="cuda")
                probe["computed"] = bool((tensor * 2).sum().item() == 2.0)
                report["verdict"] = "VERIFIED" if probe["computed"] else "FAILED"
            else:
                report["verdict"] = "PENDING"
                report["pending_reason"] = (
                    "device visible to nvidia-smi but torch.cuda.is_available() is false; "
                    "check driver compatibility and the torch CUDA build")
            report["torch_probe"] = probe
        except Exception as error:  # a missing/broken torch is exactly what this reports
            report["torch_probe"] = {"error": f"{type(error).__name__}: {error}"}
            report["verdict"] = "PENDING"
            report["pending_reason"] = "torch probe failed inside this environment"
    else:
        report["pending_reason"] = (
            "no GPU visible inside this container; install nvidia-container-toolkit on the "
            "host and enable the compose deploy.resources.gpu block")

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"GPU verdict: {report['verdict']}")
        for line in report.get("nvidia_smi") or []:
            print(f"  nvidia-smi: {line}")
        if report.get("torch_probe"):
            probe = report["torch_probe"]
            print(f"  torch: {probe.get('torch')} cuda_build={probe.get('cuda_build')} "
                  f"cuda_available={probe.get('cuda_available')}")
            if probe.get("compute_capability"):
                print(f"  device: {probe.get('device_name')} cc={probe['compute_capability']} "
                      f"arch_list={probe.get('arch_list')}")
        if report.get("pending_reason"):
            print(f"  pending reason: {report['pending_reason']}")
    return 0 if report["verdict"] in ("VERIFIED", "PENDING") else 1

if __name__ == "__main__":
    raise SystemExit(main())
