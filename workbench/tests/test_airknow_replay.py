"""Tests for Air-Know replay wrapper and contract validation."""
from __future__ import annotations

from pathlib import Path
import pytest

from workbench.backend.fashioniq_layout import CATEGORIES
from workbench.replay.airknow import (
    CHECKPOINTS,
    DEFAULT_VARIANT,
    KNOWN_AUTHOR_SHA256,
    MODEL,
    PROTOCOL,
    preflight,
    validate_checkpoint,
    validate_dataset,
    validate_source,
)


def test_airknow_protocol_and_model_id():
    assert MODEL["model_id"] == "airknow"
    assert PROTOCOL == "fashioniq_full_gallery_ref_excluded"
    assert DEFAULT_VARIANT == "fiq_n05"


def test_airknow_checkpoint_variants():
    assert "fiq_n05" in CHECKPOINTS
    assert "fiq_n08" in CHECKPOINTS
    assert "fiq_n02" not in CHECKPOINTS
    assert len(KNOWN_AUTHOR_SHA256) == 2
    for var in CHECKPOINTS.values():
        assert var["checkpoint_training_noise_pct"] in (50, 80)
        assert var["evaluation_noise_pct"] == 0


def test_airknow_preflight_fails_on_empty_paths(tmp_path: Path):
    source_dir = tmp_path / "source"
    ckpt_dir = tmp_path / "checkpoints"
    data_dir = tmp_path / "dataset"
    source_dir.mkdir()
    ckpt_dir.mkdir()
    data_dir.mkdir()

    blockers = preflight(ckpt_dir, data_dir, source_dir)
    assert any("source" in b.lower() for b in blockers)
    assert any("checkpoint" in b.lower() for b in blockers)
    assert any("dataset" in b.lower() for b in blockers)


def test_airknow_validate_source_pinned():
    airknow_source = Path(__file__).resolve().parents[2] / "workbench" / "third_party" / "AirKnow"
    blockers = validate_source(airknow_source)
    assert blockers == []


def test_airknow_native_signature_and_train_mode():
    airknow_source = Path(__file__).resolve().parents[2] / "workbench" / "third_party" / "AirKnow"
    test_py = airknow_source / "test_BLIP2.py"
    assert test_py.is_file()
    lines = test_py.read_text().splitlines()
    found_sig = False
    for i, line in enumerate(lines):
        if line.startswith("def test("):
            assert line.strip() == "def test(params, model, testset, category, txt_processors):"
            found_sig = True
            # verify model.train() is called on the next line
            assert lines[i + 1].strip() == "model.train()"
            break
    assert found_sig, "def test(...) not found in test_BLIP2.py"


def test_native_argsort_capture_delegates_and_restores_numpy():
    from types import SimpleNamespace
    import numpy as np
    from workbench.replay.airknow import _native_test_with_score_capture

    original_numpy = SimpleNamespace(argsort=np.argsort, asarray=np.asarray)
    module = SimpleNamespace(np=original_numpy)

    def native_test(params, model, dataset, category, processors):
        order = module.np.argsort(np.array([3.0, 1.0]))
        np.testing.assert_array_equal(order, [1, 0])
        return [("r1", 1.0)]

    module.test = native_test
    captures = []
    result = _native_test_with_score_capture(
        module, None, None, object(), "dress", None,
        lambda dataset, category, observed: captures.append((category, observed)))
    assert result == {"r1": 1.0}
    assert module.np is original_numpy
    assert captures[0][0] == "dress"
    np.testing.assert_array_equal(captures[0][1][0][0], [3.0, 1.0])
    np.testing.assert_array_equal(captures[0][1][0][1], [1, 0])


def test_airknow_rows_bind_native_argsort_to_gallery_ids(tmp_path: Path):
    import json
    import numpy as np
    from types import SimpleNamespace
    from workbench.replay.airknow import _capture_airknow_native_rows

    (tmp_path / "split.dress.val.json").write_text(json.dumps(["ref", "target", "other"]))
    dataset = SimpleNamespace(
        split_dir=str(tmp_path),
        test_queries_dress=[{"source_img_id": 0, "target_img_id": 1, "mod": {"str": "edits"}}],
        test_targets_dress=[{"target_img_id": i} for i in range(3)],
    )
    masked_scores = np.array([1e11, -3., -2.])
    order = np.array([1, 2, 0])
    rows = _capture_airknow_native_rows(
        dataset, "dress", [(masked_scores, order), (masked_scores.copy(), order.copy())])
    assert rows[0]["query_id"] == "dress#row0:ref:target"
    assert rows[0]["reference_id"] == "dress_ref"
    assert rows[0]["target_id"] == "dress_target"
    assert rows[0]["text"] == "edits"
    np.testing.assert_array_equal(rows[0]["scores"], [-10e10, 3.0, 2.0])
    np.testing.assert_array_equal(rows[0]["source_order"], [1, 2, 0])
    with pytest.raises(ValueError, match="duplicate source sort rows differ"):
        _capture_airknow_native_rows(dataset, "dress", [(masked_scores, order), (masked_scores, order[::-1])])

def test_airknow_native_constructor_getter_population_is_exact() -> None:
    from workbench.replay.airknow import _airknow_image_count
    root = Path(__file__).resolve().parents[2] / "data" / "FashionIQ"
    assert _airknow_image_count(root) == 27_568
