from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

from training.run_manifest import build_run_manifest
from test_taper_mag_training_contract import backbone


def test_run_manifest_is_stable_and_records_explicit_contract(tmp_path) -> None:
    config = yaml.safe_load(
        (Path(__file__).parents[1] / "conf" / "taper_mag_v4_base.yaml").read_text(
            encoding="utf-8"
        )
    )
    config["data"]["dataset_root"] = str(tmp_path / "missing-dataset")
    manifest = backbone().manifest()
    first = build_run_manifest(config, manifest, {"train_global": "abc"})
    second = build_run_manifest(config, manifest, {"train_global": "abc"})
    assert first == second
    assert first["backbone"]["revision"] == config["backbone"]["revision"]
    assert first["batching"]["effective_batch"] == 256
    assert first["dataset"]["annotation_hashes"]["status"] == "unavailable"
    assert first["teacher"]["hard_negatives"] == 64
    assert first["resume_contract"]["checkpoint"] == "epoch-boundary only"
    assert first["resume_contract"]["mid_epoch"] == "unsupported and rejected"
