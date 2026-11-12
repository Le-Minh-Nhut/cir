"""The named image-encoder-only condition routes to its own adapter method, and only there.

Two facts this pins:
* `clvc_image_encoder_only` must reach `encode_image_only_queries`, with NO text argument, and must
  never touch `encode_queries` (which would drive a text tower this ablation does not have);
* every protocol condition must still reach `encode_queries` with its texts, unchanged.

CPU only: fixture adapters that record what they were called with. No GPU, no checkpoint, no model.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from workbench.composition import inputs, runner

DIMENSION = 4
IMAGE_ONLY = "clvc_image_encoder_only"

class BothEncoders:
    """Declares both methods; records every call shape."""

    def __init__(self) -> None:
        self.query_calls: list[tuple[int, list[str], list]] = []
        self.image_only_calls: list[tuple[int, list]] = []

    def encode_queries(self, image_paths, texts, categories):
        self.query_calls.append((len(image_paths), list(texts), list(categories)))
        return np.ones((len(image_paths), DIMENSION), dtype=np.float32)

    def encode_image_only_queries(self, image_paths, categories):
        # Signature has NO texts parameter: a caller that passes three positional arguments here
        # fails with TypeError, which is the point.
        self.image_only_calls.append((len(image_paths), list(categories)))
        return np.ones((len(image_paths), DIMENSION), dtype=np.float32)

class QueriesOnly(BothEncoders):
    """The historical adapter shape: no image-only method at all."""

    encode_image_only_queries = None

def _queries(tmp_path: Path, count: int = 3):
    queries, paths = [], []
    for index in range(count):
        source = tmp_path / f"q{index}.png"
        Image.new("RGB", (3, 2), (index + 1, 1, 1)).save(source)
        queries.append(SimpleNamespace(qid=f"7:{index + 1}", query_txt=f"caption {index}",
                                       category="dress"))
        paths.append(source)
    return queries, paths

def _encode(adapter, tmp_path: Path, condition: str, queries, paths):
    return runner._encode_queries(adapter, queries, paths, list(range(len(queries))), condition,
                                  tmp_path, tmp_path / "features" / f"query_{condition}_dress.npy",
                                  "fingerprint", DIMENSION, 2)

def test_image_only_condition_routes_to_its_own_method_and_never_to_encode_queries(tmp_path: Path) -> None:
    queries, paths = _queries(tmp_path)
    adapter = BothEncoders()
    features, errors, category_errors = _encode(adapter, tmp_path, IMAGE_ONLY, queries, paths)
    assert adapter.image_only_calls == [(3, ["dress", "dress", "dress"])]
    assert adapter.query_calls == [], "the image-only condition must not reach the text path"
    assert np.isfinite(np.asarray(features)).all()
    assert errors == {} and category_errors == {}

@pytest.mark.parametrize("condition", inputs.CONDITIONS)
def test_protocol_conditions_still_route_to_encode_queries(tmp_path: Path, condition: str) -> None:
    queries, paths = _queries(tmp_path)
    adapter = BothEncoders()
    _encode(adapter, tmp_path, condition, queries, paths)
    assert adapter.image_only_calls == [], f"{condition} must not use the image-only encoder"
    assert len(adapter.query_calls) == 1
    count, texts, categories = adapter.query_calls[0]
    assert count == 3 and categories == ["dress"] * 3
    expected = "" if condition == "image_only_empty_text" else "caption 0"
    assert texts == ["" if condition == "image_only_empty_text" else f"caption {index}"
                     for index in range(3)], expected

def test_adapter_without_the_method_is_refused_by_name_and_protocol_conditions_still_work(tmp_path: Path) -> None:
    queries, paths = _queries(tmp_path)
    adapter = QueriesOnly()
    with pytest.raises(runner.AdapterEncoderMissing, match="adapter_encoder_missing"):
        _encode(adapter, tmp_path, IMAGE_ONLY, queries, paths)
    assert adapter.query_calls == [], "the refusal must happen BEFORE any encode"

    _encode(adapter, tmp_path, "multimodal", queries, paths)
    assert len(adapter.query_calls) == 1, "protocol conditions are untouched by the refusal"

def test_refusal_names_the_condition_the_required_method_and_the_alternative() -> None:
    with pytest.raises(runner.AdapterEncoderMissing) as raised:
        runner._condition_encoder(QueriesOnly(), IMAGE_ONLY)
    message = str(raised.value)
    assert IMAGE_ONLY in message
    assert "encode_image_only_queries" in message
    assert "encode_queries" in message

def test_the_encoder_map_does_not_widen_the_protocol_conditions() -> None:
    """Adding the ablation name must not add a fourth protocol condition anywhere."""
    assert inputs.CONDITIONS == ("multimodal", "text_only_black_image", "image_only_empty_text")
    assert inputs.ABLATION_CONDITIONS == (IMAGE_ONLY,)
    assert IMAGE_ONLY not in inputs.CONDITIONS
    assert inputs.encoder_method(IMAGE_ONLY) == "encode_image_only_queries"
    assert [inputs.encoder_method(name) for name in inputs.CONDITIONS] == ["encode_queries"] * 3
    assert inputs.condition_label(IMAGE_ONLY) == "Image only (image encoder)"

def test_adapt_query_inputs_leaves_the_image_untouched_and_drops_the_text() -> None:
    source = Image.new("RGB", (3, 2), (7, 8, 9))
    for condition in (IMAGE_ONLY, "image_only_empty_text"):
        adapted, text = inputs.adapt_query_inputs(source, "a caption", condition)
        assert text == ""
        assert adapted is source, f"{condition} must not stage a modified image"
    black, text = inputs.adapt_query_inputs(source, "a caption", "text_only_black_image")
    assert text == "a caption" and np.asarray(black).sum() == 0

def test_stage_query_path_keeps_the_original_file_for_the_image_only_condition(tmp_path: Path) -> None:
    source = tmp_path / "B005X4PL1G.png"
    Image.new("RGB", (3, 2), (1, 2, 3)).save(source)
    query = SimpleNamespace(qid="7:1", query_txt="a caption", category="dress")
    staged, text = runner._stage_query_path(source, query, IMAGE_ONLY, tmp_path)
    assert staged == source, "the image-only path must use the release image, not a staged copy"
    assert text == ""
