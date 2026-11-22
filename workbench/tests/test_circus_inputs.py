"""Fast contracts for CIRCUS input ablation and adapter dispatch seams."""
from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from workbench.composition import inputs


def test_black_image_preserves_size_mode_and_is_fresh() -> None:
    image = Image.new("RGB", (4, 3), (80, 20, 200))
    black = inputs.black_image_like(image)
    assert black.size == image.size
    assert black.mode == image.mode
    assert black is not image
    assert all(pixel == (0, 0, 0) for pixel in black.getdata())


def test_black_intervention_precedes_transform() -> None:
    image = Image.new("RGB", (2, 1), (100, 30, 220))
    transform = lambda value: np.asarray(value, dtype=np.float32) / 127.5 - 1.0
    seen: list[np.ndarray] = []

    def recording_transform(value: Image.Image) -> np.ndarray:
        seen.append(np.asarray(value).copy())
        return transform(value)

    adapted_image, adapted_text = inputs.adapt_query_inputs(image, "edit", "text_only_black_image")
    actual = recording_transform(adapted_image)
    assert adapted_text == "edit"
    assert np.all(seen[-1] == 0)
    normalized_original = transform(image)
    zeroed_after_transform = np.zeros_like(normalized_original)
    assert not np.array_equal(actual, zeroed_after_transform)


def test_staged_black_path_and_empty_text_cross_worker_seam(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from workbench.composition.runner import _stage_query_path

    source = tmp_path / "reference.png"
    Image.new("RGB", (5, 3), (50, 90, 120)).save(source)
    query = SimpleNamespace(qid="q-black", query_txt="edit shirt")
    black_path, black_text = _stage_query_path(source, query, "text_only_black_image", tmp_path)
    with Image.open(black_path) as black:
        assert black.size == (5, 3)
        assert all(pixel == (0, 0, 0) for pixel in black.getdata())
    assert black_text == "edit shirt"

    image_path, empty_text = _stage_query_path(source, query, "image_only_empty_text", tmp_path)
    assert image_path == source
    assert empty_text == ""


def test_image_only_empty_text_keeps_image_and_empty_string() -> None:
    image = Image.new("RGB", (3, 2), (12, 34, 56))
    adapted, text = inputs.adapt_query_inputs(image, "discard me", "image_only_empty_text")
    assert text == ""
    assert adapted is image
    assert np.array_equal(np.asarray(adapted), np.asarray(image))


def test_multimodal_keeps_inputs_unchanged() -> None:
    image = Image.new("RGB", (2, 2), (1, 2, 3))
    text = "same text"
    adapted, adapted_text = inputs.adapt_query_inputs(image, text, "multimodal")
    assert adapted is image
    assert adapted_text == text


def test_unknown_condition_raises() -> None:
    with pytest.raises(ValueError, match="unknown composition condition"):
        inputs.adapt_query_inputs(Image.new("RGB", (1, 1)), "text", "unknown")


def test_gallery_objects_are_invariant_to_query_condition() -> None:
    query = Image.new("RGB", (2, 2), (10, 20, 30))
    gallery = Image.new("RGB", (3, 2), (40, 50, 60))
    gallery_pixels = np.asarray(gallery).copy()
    original_gallery = gallery
    for condition in inputs.CONDITIONS:
        inputs.adapt_query_inputs(query, "edit", condition)
        assert gallery is original_gallery
        assert np.array_equal(np.asarray(gallery), gallery_pixels)


def test_inputs_module_has_no_runtime_torch_import() -> None:
    source = Path(inputs.__file__).read_text(encoding="utf-8")
    assert not any(line.strip().startswith(("import torch", "from torch"))
                   for line in source.splitlines())


def test_capability_reads_registry_and_names_missing_adapter_module(monkeypatch: pytest.MonkeyPatch) -> None:
    adapters = importlib.import_module("workbench.composition.adapters")
    absent = {"models": [{"model_id": "fixture", "checkpoint_variants": []}]}
    assert adapters.capability("fixture", absent) is None
    declared = {"models": [{"model_id": "encoder",
                            "composition_adapter": "workbench.composition.model_encoder"}]}
    assert adapters.capability("encoder", declared) == "workbench.composition.model_encoder"
    from workbench.composition.model_encoder import Adapter

    assert adapters.adapter_class("encoder", declared) is Adapter
    with pytest.raises(ImportError, match="module_does_not_exist"):
        adapters.adapter_class("fixture", {
            "models": [{"model_id": "fixture",
                        "composition_adapter": "workbench.composition.module_does_not_exist"}]
        })


def test_limn_refuses_query_without_resolved_category(tmp_path: Path) -> None:
    from workbench.composition.model_limn import Adapter, CategoryScopeError

    adapter = Adapter(config=None)
    adapter.work_root = tmp_path
    with pytest.raises(CategoryScopeError, match="category_scope_invalid") as caught:
        adapter.encode_queries([Image.new("RGB", (1, 1))], ["edit"], [None])
    assert caught.value.model_id == "limn"
    assert caught.value.batch_indices == (0,)
    assert caught.value.reasons_by_index == {0: "missing_category"}


def test_adapter_requires_explicit_work_root() -> None:
    from workbench.composition.adapters import AdapterWorkRootError
    from workbench.composition.model_limn import Adapter

    with pytest.raises(AdapterWorkRootError, match="work_root must be assigned"):
        Adapter(config=None).encode_queries(["unused.png"], ["edit"], ["dress"])


def test_default_rank_ties_are_stable() -> None:
    adapters = importlib.import_module("workbench.composition.adapters")

    class FixtureAdapter(adapters.CompositionAdapter):
        model_id = "fixture"

        def encode_queries(self, images, texts, categories):
            raise NotImplementedError

        def encode_targets(self, images, categories):
            raise NotImplementedError

    adapter = FixtureAdapter(config=None)
    queries = np.asarray([[1.0, 0.0]], dtype=np.float32)
    targets = np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    positions = [[1]]
    first = adapter.rank(queries, targets, target_positions=positions)
    second = adapter.rank(queries, targets, target_positions=positions)
    assert first == second == [2]


def test_runner_forwards_only_selected_capture_ids_in_worker_order() -> None:
    from workbench.composition.runner import _capture_kwargs

    class CapturingAdapter:
        capture_path = Path("capture.npz")

        def encode_targets(self, paths, categories, *, capture_ids=None, selected_capture_ids=None):
            pass

    ids = ["g3", "g1", "g2"]
    assert _capture_kwargs(CapturingAdapter(), CapturingAdapter.encode_targets, ids,
                           {"g1", "g3"}) == {
                               "capture_ids": ids, "selected_capture_ids": ["g3", "g1"]}


def test_gallery_capture_selection_reaches_real_encode_call_in_order(tmp_path: Path) -> None:
    from workbench.composition.runner import _encode_gallery

    image_paths = []
    for identity in ("g2", "g1", "g3"):
        path = tmp_path / f"{identity}.png"
        Image.new("RGB", (8, 8), "white").save(path)
        image_paths.append(path)
    observed: list[tuple[list[str], list[str], Path]] = []

    class Adapter:
        batch_size = 2
        capture_path: Path | None = None

        def encode_targets(self, paths, categories, *, capture_ids: list[str] | None = None,
                          selected_capture_ids: list[str] | None = None):
            assert self.capture_path is not None
            assert capture_ids is not None and selected_capture_ids is not None
            observed.append((list(capture_ids), list(selected_capture_ids), self.capture_path))
            np.savez(self.capture_path, ids=np.asarray(selected_capture_ids))
            return np.ones((len(paths), 2), dtype=np.float32)

    _encode_gallery(Adapter(), image_paths, "dress", tmp_path / "features.npy", "fingerprint",
                    2, 1, 1, 1, image_ids=("id-2", "id-1", "id-3"),
                    selected_capture_ids={"id-1", "id-3"})

    assert len(observed) == 1
    assert observed[0][:2] == (["id-2", "id-1", "id-3"], ["id-1", "id-3"])
    assert observed[0][2].is_file()
    with np.load(observed[0][2], allow_pickle=False) as capture:
        assert capture["ids"].tolist() == ["id-1", "id-3"]

