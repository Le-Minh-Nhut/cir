from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from workbench.backend.operator_config import WorkbenchConfig


def load_module():
    spec = importlib.util.spec_from_file_location("download_auxiliary_assets", Path("workbench/scripts/download_auxiliary_assets.py"))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_dry_run_reports_canonical_qwen_destinations(monkeypatch, tmp_path: Path, capsys) -> None:
    module = load_module()
    config = WorkbenchConfig(tmp_path, tmp_path / "data", tmp_path / "FashionIQ", "127.0.0.1", 8000, 5173, tmp_path / "checkpoints", tmp_path / "results", tmp_path / "third_party")
    monkeypatch.setattr(module, "resolve_config", lambda: config)
    assert module.main(["--model", "csmcir", "--dry-run"]) == 0

    output = capsys.readouterr().out
    assert "https://huggingface.co/peng12138/CSMCIR" in output
    assert str(config.FASHIONIQ_ROOT / "qwen_captions" / "dress_cot_val.json") in output
    assert "exact file URLs are unverified" in output
