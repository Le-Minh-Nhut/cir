from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[2]
WRAPPER = REPO / "workbench" / "replay" / "dcnet_api.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_native_configparser_writes_only_into_private_checkpoint_stage(tmp_path: Path) -> None:
    source = REPO / "workbench" / "third_party" / "DCNet"
    checkpoint = tmp_path / "registered-checkpoint"
    (checkpoint / "val").mkdir(parents=True)
    config = {
        "trainer": {"save_dir": str(checkpoint)},
        "data_loader": {"args": {"mode": "train"}},
    }
    original_config = checkpoint / "config.json"
    original_config.write_text(json.dumps(config))
    original_weights = checkpoint / "trained_model.pth"
    original_weights.write_bytes(b"private staged weight bytes")
    protected_att = checkpoint / "val" / "att_result.pt"
    protected_att.write_bytes(b"pre-existing result")
    config_sha = _sha256(original_config)
    weights_sha = _sha256(original_weights)
    att_sha = _sha256(protected_att)

    run_id = "test-run-001"
    run_dir = tmp_path / "artifacts" / "logs" / f"dcnet__fashioniq_run_directory__{run_id}"
    run_dir.mkdir(parents=True)
    # Exercise the real pinned ConfigParser without loading the checkpoint or model.
    wrapper_source = tmp_path / "source"
    wrapper_source.mkdir()
    (wrapper_source / "logger").mkdir()
    (wrapper_source / "logger" / "logger_config.json").write_text(
        (source / "logger" / "logger_config.json").read_text())
    test_entry = tmp_path / "fixture-source" / "test.py"
    test_entry.parent.mkdir()
    test_entry.write_text(
        "import argparse, json, sys\n"
        "from pathlib import Path\n"
        "from parse_config import ConfigParser\n"
        "args = argparse.ArgumentParser()\n"
        "args.add_argument('--resume', required=True)\n"
        "args.add_argument('--device', default='')\n"
        "args.add_argument('--seeds', default='0')\n"
        "args.add_argument('--mode', default='val')\n"
        "config = ConfigParser(args, 'eval')\n"
        "(config.save_dir / 'att_result.pt').write_bytes(b'fresh isolated result')\n"
    )
    (wrapper_source / "test.py").write_text(test_entry.read_text())
    env = os.environ.copy()
    env.update({
        "evaluation_output_root": str(tmp_path / "artifacts"),
        "evaluation_run_id": run_id,
        "PYTHONPATH": os.pathsep.join((str(source), str(REPO))),
    })

    result = subprocess.run(
        [sys.executable, str(WRAPPER), "--source", str(wrapper_source),
         "--checkpoint", str(checkpoint), "--checkpoint-id", "fashioniq_run_directory"],
        cwd=source, env=env, capture_output=True, text=True, check=False,
    )
    manifest = json.loads((run_dir / "dcnet-native-staging.json").read_text())
    assert manifest["native_child"]["run_id"] == run_id
    assert manifest["native_child"]["pid"] > 0

    assert result.returncode == 0, result.stderr
    assert _sha256(original_config) == config_sha
    assert _sha256(original_weights) == weights_sha
    assert _sha256(protected_att) == att_sha
    manifest = json.loads((run_dir / "dcnet-native-staging.json").read_text())
    stage = run_dir / "dcnet-native-resume"
    assert manifest["inputs"]["config.json"]["original_sha256"] == config_sha
    assert manifest["inputs"]["config.json"]["staged_sha256"] == config_sha
    assert manifest["inputs"]["trained_model.pth"]["original_sha256"] == weights_sha
    assert manifest["inputs"]["trained_model.pth"]["staged_sha256"] == weights_sha
    assert manifest["protected_original_att_result"]["unchanged"] is True
    assert manifest["staged_outputs"]["native_att_result_exists"] is True
    assert (stage / "val" / "config.json").is_file()
    assert (stage / "val" / "info.log").is_file()
    assert (stage / "val" / "att_result.pt").read_bytes() == b"fresh isolated result"
