from __future__ import annotations

import json

from evaluation.fashioniq import build_validation_datasets


def test_evaluation_dataset_uses_explicit_caption_policy(tmp_path) -> None:
    record = [
        {
            "candidate": "reference",
            "target": "target",
            "captions": ["make it red", "add long sleeves"],
        }
    ]
    (tmp_path / "cap.dress.val.json").write_text(json.dumps(record), encoding="utf-8")

    datasets = build_validation_datasets(
        tmp_path,
        ["dress"],
        caption_policy="normalized_ordered_and",
        seed=9,
        correction_dicts={"dress": {}},
    )

    assert datasets["dress"].caption_policy == "normalized_ordered_and"
    assert datasets["dress"][0].modification_text == "make it red and add long sleeves"
