from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from datasets.cirr import CIRRDataset, build_cirr_image_store, load_cirr_image_mapping


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_cirr_annotation_maps_to_common_sample(tmp_path: Path) -> None:
    record = {
        "pairid": 123,
        "reference": "ref",
        "target_hard": "target",
        "target_soft": {"a": 0.2},
        "caption": "make it red",
        "img_set": {
            "id": 1,
            "members": ["ref", "a", "b", "target", "c", "d"],
            "reference_rank": 0,
            "target_rank": 3,
        },
    }
    _write_json(tmp_path / "cap.rc2.val.json", [record])

    sample = CIRRDataset(tmp_path, "val")[0]

    assert sample.sample_id == "cirr:rc2:val:123"
    assert sample.benchmark_id == "123"
    assert sample.reference_id == "ref"
    assert sample.target_id == "target"
    assert sample.modification_text == "make it red"
    assert sample.category is None
    assert sample.group_members == ("ref", "a", "b", "target", "c", "d")
    assert sample.ground_truth_ids == ("target",)
    assert sample.metadata["group_id"] == "1"
    assert sample.metadata["target_soft"] == {"a": 0.2}


def test_cirr_test1_allows_missing_public_ground_truth(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "cap.rc2.test1.json",
        [{"pairid": 9, "reference": "ref", "caption": "brighter"}],
    )

    sample = CIRRDataset(tmp_path, "test1")[0]

    assert sample.target_id is None
    assert sample.ground_truth_ids == ()


def test_cirr_image_split_resolves_relative_paths(tmp_path: Path) -> None:
    split_root = tmp_path / "image_splits"
    image_root = tmp_path / "img_raw"
    image_path = image_root / "dev" / "ref.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (2, 2), color="red").save(image_path)
    _write_json(split_root / "split.rc2.val.json", {"ref": "dev/ref.png"})

    mapping = load_cirr_image_mapping(split_root, "val")
    store = build_cirr_image_store(image_root, split_root, "val")

    assert mapping == {"ref": "dev/ref.png"}
    assert store.path_for("ref") == image_path
    assert store.load("ref").mode == "RGB"
