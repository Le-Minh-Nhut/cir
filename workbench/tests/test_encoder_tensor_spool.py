from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from workbench.replay.tensor_spool import disk_backed_images


def _dataset_root(root: Path) -> Path:
    for category in ("dress", "shirt", "toptee"):
        (root / "captions").mkdir(parents=True, exist_ok=True)
        (root / "image_splits").mkdir(parents=True, exist_ok=True)
        (root / "captions" / f"cap.{category}.val.json").write_text(json.dumps([
            {"candidate": f"{category}-a", "target": f"{category}-b"},
            {"candidate": f"{category}-b", "target": f"{category}-c"},
        ]))
        (root / "image_splits" / f"split.{category}.val.json").write_text(
            json.dumps([f"{category}-{suffix}" for suffix in "abc"]))
    return root


def test_disk_backed_images_preserves_transform_order_bytes_and_tensor_lifetime(tmp_path: Path) -> None:
    calls: list[tuple[str, int]] = []

    class FashionIQ:
        def get_img(self, name: str, stage: int = 0) -> torch.Tensor:
            calls.append((name, stage))
            return torch.arange(12, dtype=torch.float32).reshape(3, 2, 2) + len(calls)

    source = SimpleNamespace(FashionIQ=FashionIQ)
    cache = tmp_path / "cache"
    cache.mkdir()
    dataset_root = _dataset_root(tmp_path / "dataset")
    expected: list[tuple[str, int]] = []
    for category in ("dress", "shirt", "toptee"):
        expected.extend((f"{category}_{image}", 0) for image in ("a", "b", "b", "c"))
        expected.extend((f"{category}_{image}", 1) for image in "abc")
    original = FashionIQ.get_img
    tensors: list[torch.Tensor] = []

    with disk_backed_images(dataset_root, cache, source) as receipt:
        dataset = FashionIQ()
        for index, (name, stage) in enumerate(expected):
            actual = dataset.get_img(name, stage)
            expected_tensor = torch.arange(12, dtype=torch.float32).reshape(3, 2, 2) + index + 1
            assert actual.dtype == expected_tensor.dtype
            assert actual.shape == expected_tensor.shape
            assert actual.stride() == expected_tensor.stride()
            assert torch.equal(actual, expected_tensor)
            tensors.append(actual)
        assert receipt["stored_image_tensors"] == len(expected)
        assert Path(receipt["spool_path"]).stat().st_size == receipt["max_tensor_bytes"]
    assert calls == expected
    assert FashionIQ.get_img is original
    assert receipt["spool_file_removed"]
    assert not Path(receipt["spool_path"]).exists()
    assert torch.equal(tensors[-1], torch.arange(12, dtype=torch.float32).reshape(3, 2, 2) + len(expected))



def test_spool_receipt_provenance_is_json_serializable(tmp_path: Path) -> None:
    class FashionIQ:
        def get_img(self, name: str, stage: int = 0) -> torch.Tensor:
            return torch.ones((3, 2, 2), dtype=torch.float32)

    source = SimpleNamespace(FashionIQ=FashionIQ)
    cache = tmp_path / "cache"
    cache.mkdir()
    dataset_root = _dataset_root(tmp_path / "dataset")
    with disk_backed_images(dataset_root, cache, source) as receipt:
        dataset = FashionIQ()
        count = receipt["expected_image_tensors"]
        for index in range(count):
            dataset.get_img(f"image-{index}")
        receipt.evict()
        provenance = {"tensor_storage": receipt}
        encoded = json.dumps(provenance)
        assert "evict" not in encoded
        assert receipt["stored_image_tensors"] == count


def test_dual_getter_spool(tmp_path: Path) -> None:
    class FashionIQ:
        def get_test_img(self, name): return torch.full((3, 2, 2), 1.)
        def get_img(self, name): return torch.full((3, 2, 2), 2.)
    source = SimpleNamespace(FashionIQ=FashionIQ)
    cache = tmp_path / "cache"; cache.mkdir()
    with disk_backed_images(_dataset_root(tmp_path / "dataset"), cache, source,
                            method_name=("get_test_img", "get_img"), expected_count=2) as receipt:
        item = FashionIQ()
        assert item.get_test_img("q")[0, 0, 0] == 1
        assert item.get_img("g")[0, 0, 0] == 2
        assert receipt["getter_counts"] == {"get_test_img": 1, "get_img": 1}
def test_spool_enforces_exact_expected_count(tmp_path: Path) -> None:
    class FashionIQ:
        def get_test_img(self, name: str) -> torch.Tensor:
            return torch.ones((3, 2, 2), dtype=torch.float32)

    source = SimpleNamespace(FashionIQ=FashionIQ)
    cache = tmp_path / "cache"
    cache.mkdir()
    dataset_root = _dataset_root(tmp_path / "dataset")
    with disk_backed_images(dataset_root, cache, source, method_name="get_test_img",
                            expected_count=2) as receipt:
        FashionIQ().get_test_img("source")
        FashionIQ().get_test_img("target")
        assert receipt["stored_image_tensors"] == 2
    try:
        with disk_backed_images(dataset_root, cache, source, method_name="get_test_img",
                                expected_count=3) as receipt:
            FashionIQ().get_test_img("source")
            FashionIQ().get_test_img("target")
    except RuntimeError as error:
        assert str(error) == "FashionIQ spool population mismatch: 2 != 3"
    else:
        raise AssertionError("spool accepted an under-populated explicit count")
    try:
        with disk_backed_images(dataset_root, cache, source, method_name="get_test_img",
                                expected_count=1):
            FashionIQ().get_test_img("source")
            FashionIQ().get_test_img("target")
    except RuntimeError as error:
        assert str(error) == "FashionIQ getter produced more tensors than declared population"
    else:
        raise AssertionError("spool accepted an over-populated explicit count")


def test_spool_rejects_invalid_expected_count(tmp_path: Path) -> None:
    class FashionIQ:
        def get_img(self, name: str) -> torch.Tensor:
            return torch.ones((3, 2, 2), dtype=torch.float32)

    source = SimpleNamespace(FashionIQ=FashionIQ)
    cache = tmp_path / "cache"
    cache.mkdir()
    dataset_root = _dataset_root(tmp_path / "dataset")
    counts: list[Any] = [0, -1, True, 1.5]
    for count in counts:
        try:
            with disk_backed_images(dataset_root, cache, source, expected_count=count):
                raise AssertionError("spool accepted an invalid explicit count")
        except ValueError as error:
            assert str(error) == "FashionIQ expected_image_count must be a positive integer"

def test_spool_selects_native_test_getter(tmp_path: Path) -> None:
    calls = 0

    class FashionIQ:
        def get_test_img(self, name: str) -> torch.Tensor:
            nonlocal calls
            calls += 1
            return torch.ones((3, 2, 2), dtype=torch.float32) * calls

    source = SimpleNamespace(FashionIQ=FashionIQ)
    cache = tmp_path / "cache"
    cache.mkdir()
    dataset_root = _dataset_root(tmp_path / "dataset")
    with disk_backed_images(dataset_root, cache, source, method_name="get_test_img") as receipt:
        image = FashionIQ().get_test_img("dress_a")
        assert image.equal(torch.ones((3, 2, 2)))
        for index in range(2, receipt["expected_image_tensors"] + 1):
            assert FashionIQ().get_test_img(f"image-{index}").equal(torch.ones((3, 2, 2)) * index)
        assert receipt["stored_image_tensors"] == receipt["expected_image_tensors"]
    assert receipt["spool_file_removed"]

def test_airknow_source_scorer_reuses_spooled_views_after_context(tmp_path: Path, monkeypatch) -> None:
    import importlib.util

    source_file = Path(__file__).resolve().parents[1] / "third_party/AirKnow/test_BLIP2.py"
    spec = importlib.util.spec_from_file_location("airknow_spool_test_BLIP2", source_file)
    assert spec is not None and spec.loader is not None
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)

    class FashionIQ:
        def get_test_img(self, name: str) -> torch.Tensor:
            return torch.full((3, 2, 2), 1.0 if name == "q0" else 2.0)

        def get_img(self, name: str) -> torch.Tensor:
            return torch.full((3, 2, 2), float(int(name[-1]) + 1))

    source_modules = SimpleNamespace(FashionIQ=FashionIQ)
    cache = tmp_path / "cache"
    cache.mkdir()
    dataset_root = _dataset_root(tmp_path / "dataset")
    with disk_backed_images(dataset_root, cache, source_modules,
                            method_name=("get_test_img", "get_img"), expected_count=5) as receipt:
        queries = [
            {"source_img_id": 0, "target_img_id": 1, "source_img_data": FashionIQ().get_test_img("q0"), "mod": {"str": "red"}},
            {"source_img_id": 1, "target_img_id": 2, "source_img_data": FashionIQ().get_test_img("q1"), "mod": {"str": "blue"}},
        ]
        targets = [{"target_img_id": i, "target_img_data": FashionIQ().get_img(f"g{i}")} for i in range(3)]
        dataset = SimpleNamespace(test_queries_dress=queries, test_targets_dress=targets)
        assert receipt["getter_counts"] == {"get_test_img": 2, "get_img": 3}
    assert receipt["spool_file_removed"]

    class Model:
        def train(self):
            pass

        def extract_retrieval_compose(self, images, captions):
            return torch.ones((len(captions), 1, 1, 2))

        def extract_retrieval_target(self, images):
            return torch.ones((len(images), 2, 32))
    params = SimpleNamespace(batch_size=1, dataset="fashioniq")
    processors = {"eval": lambda text: text}
    expected = source.test(params, Model(), dataset, "dress", processors)
    observed = source.test(params, Model(), dataset, "dress", processors)
    assert expected == observed
    assert [name for name, _ in observed] == ["dress_r1", "dress_r10", "dress_r50"]
