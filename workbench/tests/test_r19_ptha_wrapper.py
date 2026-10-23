"""R19 regression coverage for adapter-generated PTHA wrapper commands."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from workbench.backend import runtime
from workbench.backend.adapters.base import EvalRequest
from workbench.backend.adapters.models import ADAPTERS
from workbench.backend.registry import checkpoint_path, model_by_id


def _command(tmp_path: Path, checkpoint_id: str = "cirr_transfer") -> tuple[list[str], Path]:
    model = model_by_id("ptha_mtst")
    variant = next(item for item in model["checkpoint_variants"]
                   if item["checkpoint_id"] == "cirr_transfer")
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint = checkpoint_path("ptha_mtst", variant, checkpoint_root)
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"tiny checkpoint fixture")

    dataset_root = tmp_path / "fashioniq"
    for category in ("dress", "shirt", "toptee"):
        for relative in (f"captions/cap.{category}.val.json",
                         f"image_splits/split.{category}.val.json"):
            path = dataset_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("[]", encoding="utf-8")

    request = EvalRequest("ptha_mtst", checkpoint_id, model["native_protocol"],
                          dataset_root, tmp_path / "output.json")
    source = REPO / "workbench" / "third_party" / model["source_dir"]
    with patch.object(runtime, "python_executable", lambda _: sys.executable):
        command = ADAPTERS["ptha_mtst"]().command(source, checkpoint, request)
    return command, checkpoint


def test_adapter_command_reaches_checkpoint_at_member_root(tmp_path: Path) -> None:
    command, checkpoint = _command(tmp_path)
    model = model_by_id("ptha_mtst")
    variant = next(item for item in model["checkpoint_variants"]
                   if item["checkpoint_id"] == "cirr_transfer")
    checkpoint_root = Path(command[command.index("--checkpoint-root") + 1])

    assert checkpoint == checkpoint_root / variant["filename"]
    assert checkpoint_root == checkpoint.parent
    assert "ptha_mtst/ptha_mtst" not in str(checkpoint)
    assert command[command.index("--checkpoint-id") + 1] == "cirr_transfer"

    completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert "unrecognized arguments" not in completed.stderr, completed.stderr
    assert completed.returncode != 2, completed.stderr
    assert "PTHA+MTST CIRR checkpoint missing" not in completed.stderr, completed.stderr


def test_unknown_checkpoint_id_is_refused_with_valid_ids(tmp_path: Path) -> None:
    command, _checkpoint = _command(tmp_path)
    index = command.index("--checkpoint-id") + 1
    command[index] = "not_registered"

    completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert completed.returncode != 0
    assert "not_registered" in completed.stderr
    assert "cirr_transfer" in completed.stderr
