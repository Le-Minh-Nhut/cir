#!/usr/bin/env python3
"""Run pinned DCNet test.py with checkpoint bytes and val outputs isolated per evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-id", required=True)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    if not source.is_dir() or not (source / "test.py").is_file():
        raise ValueError(f"DCNet source/test.py missing: {source}")
    if not checkpoint.is_dir():
        raise ValueError(f"DCNet run-directory checkpoint missing: {checkpoint}")
    evaluation_root = Path(os.environ["evaluation_output_root"]).resolve(strict=True)
    evaluation_run_id = os.environ.get("evaluation_run_id")
    if not evaluation_run_id or any(part in evaluation_run_id for part in ("/", "\\", "..")):
        raise ValueError("evaluation_run_id missing or unsafe")
    checkpoint_id = args.checkpoint_id
    if not checkpoint_id or any(part in checkpoint_id for part in ("/", "\\", "..")):
        raise ValueError("checkpoint_id missing or unsafe")
    run_dir = evaluation_root / "logs" / f"dcnet__{checkpoint_id}__{evaluation_run_id}"
    if not run_dir.is_dir():
        raise ValueError(f"audited evaluator run directory missing: {run_dir}")
    resume = run_dir / "dcnet-native-resume"
    resume.mkdir(mode=0o700, parents=False, exist_ok=False)
    inputs = {}
    for name in ("config.json", "trained_model.pth"):
        original = checkpoint / name
        if not original.is_file() or original.is_symlink():
            raise ValueError(f"DCNet checkpoint member missing or not regular: {original}")
        staged = resume / name
        shutil.copyfile(original, staged)
        os.chmod(staged, 0o600)
        original_digest, staged_digest = _sha256(original), _sha256(staged)
        if original_digest != staged_digest:
            raise RuntimeError(f"DCNet staging digest mismatch for {name}")
        inputs[name] = {"original": str(original), "staged": str(staged),
                        "original_sha256": original_digest, "staged_sha256": staged_digest}
    original_att = checkpoint / "val" / "att_result.pt"
    original_att_sha = _sha256(original_att) if original_att.is_file() else None
    manifest = {"model_id": "dcnet", "checkpoint_id": args.checkpoint_id,
                "evaluation_run_id": evaluation_run_id,
                "output_only_override": {"resume_dir": str(resume),
                                          "native_eval_output_dir": str(resume / "val")},
                "inputs": inputs, "protected_original_att_result": {
                    "path": str(original_att), "sha256_before": original_att_sha}}
    manifest_path = run_dir / "dcnet-native-staging.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")

    process = subprocess.Popen([sys.executable, str(source / "test.py"),
                                "--resume", str(resume)], cwd=source)
    child_marker = run_dir / "dcnet-native-child.json"
    temporary = child_marker.with_suffix(".tmp")
    temporary.write_text(json.dumps({"pid": process.pid, "run_id": evaluation_run_id}) + "\n")
    temporary.replace(child_marker)
    returncode = process.wait()
    manifest["native_child"] = {"pid": process.pid, "returncode": returncode,
                                "run_id": evaluation_run_id}
    child_marker.unlink(missing_ok=True)
    after_att_sha = _sha256(original_att) if original_att.is_file() else None
    fresh_att = resume / "val" / "att_result.pt"
    manifest["protected_original_att_result"]["sha256_after"] = after_att_sha
    manifest["protected_original_att_result"]["unchanged"] = original_att_sha == after_att_sha
    manifest["staged_outputs"] = {"native_att_result": str(fresh_att),
                                  "native_att_result_exists": fresh_att.is_file(),
                                  "native_config": str(resume / "val" / "config.json"),
                                  "native_config_exists": (resume / "val" / "config.json").is_file()}
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    if original_att_sha != after_att_sha:
        raise RuntimeError("DCNet evaluator changed original checkpoint val/att_result.pt")
    if returncode == 0 and not fresh_att.is_file():
        raise RuntimeError(f"DCNet evaluation succeeded without staged att_result.pt: {fresh_att}")
    return returncode

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"DCNet isolated evaluator failed: {type(error).__name__}: {error}",
              file=sys.stderr, flush=True)
        raise SystemExit(1)
