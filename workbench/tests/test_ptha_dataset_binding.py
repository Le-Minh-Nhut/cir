"""Probe MTST's native FashionIQ constructor against canonical dataset roots."""
from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from PIL import Image

import pytest

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "workbench" / "third_party" / "MTST_PTHA" / "MTST_PTHA" / "src"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from workbench.replay.ptha_mtst import _fashioniq_dataset_binding


def _fixture(root: Path, marker: str) -> None:
    for directory in ("captions", "image_splits", "images"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    (root / "captions" / "cap.dress.val.json").write_text(
        json.dumps([{"candidate": marker + ".jpg", "target": marker + "_target.jpg", "captions": [marker]}]),
        encoding="utf-8",
    )
    (root / "image_splits" / "split.dress.val.json").write_text(
        json.dumps([marker]), encoding="utf-8"
    )
    image = Image.new("RGB", (1, 1), (255, 0, 0) if marker == "canonical" else (0, 0, 255))
    image.save(root / "images" / f"{marker}.png")

def test_native_constructor_binds_requested_root_and_ignores_decoy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(str(SRC))
    old_module = sys.modules.pop("data_utils", None)
    old_environment = os.environ.get("MTST_DATA_ROOT")
    try:
        data_utils = importlib.import_module("data_utils")
        for basename in ("FashionIQ", "fiq_stage"):
            parent = tmp_path / basename
            requested = parent / basename
            _fixture(requested, "canonical")
            decoy = parent / "fashionIQ_dataset"
            _fixture(decoy, "decoy")
            decoy_before = (decoy / "captions" / "cap.dress.val.json").read_bytes()

            original_env = os.environ.get("MTST_DATA_ROOT")
            data_utils.base_path = parent
            previous = data_utils.FashionIQDataset("val", ["dress"], "relative", lambda image: image)
            assert previous.triplets[0]["captions"] == ["decoy"]

            with _fashioniq_dataset_binding(requested) as native:
                staging_root = native.base_path
                relative = native.FashionIQDataset("val", ["dress"], "relative", lambda image: image)
                classic = native.FashionIQDataset("val", ["dress"], "classic", lambda image: image)
                assert relative.triplets[0]["captions"] == ["canonical"]
                assert relative.image_names == ["canonical"]
                assert classic.image_names == ["canonical"]
                assert (native.base_path / "fashionIQ_dataset").resolve() == requested.resolve()
                assert classic[0][1].getpixel((0, 0)) == (255, 0, 0)
            assert not staging_root.exists()
            assert data_utils.base_path == parent
            assert os.environ.get("MTST_DATA_ROOT") == original_env

            assert (decoy / "captions" / "cap.dress.val.json").read_bytes() == decoy_before
    finally:
        sys.modules.pop("data_utils", None)
        if old_module is not None:
            sys.modules["data_utils"] = old_module
        if old_environment is None:
            os.environ.pop("MTST_DATA_ROOT", None)
        else:
            os.environ["MTST_DATA_ROOT"] = old_environment
