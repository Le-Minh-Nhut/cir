from __future__ import annotations

import json
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


def _evict(tensor_map: Any, *, flush: bool = True) -> None:
    if flush:
        tensor_map.flush()
    tensor_map._mmap.madvise(__import__("mmap").MADV_DONTNEED)

def expected_image_count(dataset_root: Path, split: str = "val") -> int:
    """Count transformed query and endpoint images in FashionIQ's three native categories."""
    count = 0
    for category in ("dress", "shirt", "toptee"):
        captions = json.loads((dataset_root / "captions" / f"cap.{category}.{split}.json").read_text())
        images = json.loads((dataset_root / "image_splits" / f"split.{category}.{split}.json").read_text())
        positions = {image: i for i, image in enumerate(images)}
        endpoints = {positions[row[key]] for row in captions for key in ("candidate", "target")}
        count += 2 * len(captions) + len(endpoints)
    return count


class TensorSpoolReceipt(dict[str, Any]):
    _evict_callback: Any = None

    def evict(self) -> None:
        if self._evict_callback is None:
            raise RuntimeError("FashionIQ tensor spool is not initialized")
        self._evict_callback()
@contextmanager
def disk_backed_images(dataset_root: Path, cache_root: Path, datasets: Any, *,
                       method_name: str | tuple[str, ...] = "get_img", split: str = "val",
                       expected_count: int | None = None) -> Iterator[TensorSpoolReceipt]:
    """Run native transforms in order, returning tensor views backed by run-local disk storage.

    Multiple getters share one spool when native constructor calls distinct methods for the same
    materialized population.
    """
    import numpy as np
    import torch

    cls = datasets.FashionIQ
    method_names = (method_name,) if isinstance(method_name, str) else method_name
    if not method_names or len(set(method_names)) != len(method_names):
        raise ValueError("FashionIQ getter method names must be non-empty and unique")
    originals = {name: getattr(cls, name) for name in method_names}
    expected = expected_image_count(dataset_root, split) if expected_count is None else expected_count
    if type(expected) is not int or expected <= 0:
        raise ValueError("FashionIQ expected_image_count must be a positive integer")
    state: dict[str, Any] = {"map": None, "path": None, "index": 0, "shape": None, "dtype": None}
    method_counts = {name: 0 for name in method_names}
    receipt = TensorSpoolReceipt(storage="numpy-memmap", expected_image_tensors=expected,
                                 stored_image_tensors=0, getter_counts=method_counts,
                                 max_tensor_bytes=None, spool_file_removed=False)
    receipt._evict_callback = lambda: _evict(state["map"], flush=False)


    def wrap_getter(original: Any, method: str):
        def get_image(instance: Any, image_name: str, *args: Any, **kwargs: Any) -> Any:
            tensor = original(instance, image_name, *args, **kwargs)
            if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu" or not tensor.is_contiguous():
                raise TypeError("FashionIQ image getter must return a contiguous CPU tensor")
            array = tensor.numpy()
            if array.dtype != np.float32:
                raise TypeError(f"FashionIQ image getter returned {array.dtype}, expected float32")
            if state["map"] is None:
                size = expected * array.nbytes
                required = 2 * size + (1 << 30)
                if shutil.disk_usage(cache_root).free < required:
                    raise OSError(f"insufficient free disk for FashionIQ tensor spool and native cache: need {required} bytes")
                handle = tempfile.NamedTemporaryFile(prefix="fashioniq-image-tensors-", suffix=".f32",
                                                     dir=cache_root, delete=False)
                state["path"] = Path(handle.name)
                handle.truncate(size)
                handle.close()
                tensor_map = np.memmap(state["path"], mode="r+", dtype=array.dtype,
                                       shape=(expected, *array.shape))
                state.update(map=tensor_map, shape=tuple(array.shape), dtype=array.dtype)
                receipt.update(max_tensor_bytes=size, tensor_shape=list(array.shape), tensor_dtype=str(array.dtype),
                               spool_path=str(state["path"]))
                receipt._evict_callback = lambda: _evict(state["map"], flush=False)
            if tuple(array.shape) != state["shape"] or array.dtype != state["dtype"]:
                raise ValueError("FashionIQ transformed image shape or dtype changed during construction")
            index = state["index"]
            if index >= expected:
                raise RuntimeError("FashionIQ getter produced more tensors than declared population")
            tensor_map = state["map"]
            if tensor_map is None:
                raise RuntimeError("FashionIQ tensor spool failed to initialize")
            tensor_map[index] = array
            if (index + 1) % 64 == 0 or index + 1 == expected:
                _evict(tensor_map)
            state["index"] = index + 1
            method_counts[method] += 1
            receipt["stored_image_tensors"] = index + 1
            return torch.from_numpy(tensor_map[index])

        return get_image

    wrappers = {name: wrap_getter(originals[name], name) for name in method_names}
    try:
        for name, wrapper in wrappers.items():
            setattr(cls, name, wrapper)
        yield receipt
        if state["index"] != expected:
            raise RuntimeError(f"FashionIQ spool population mismatch: {state['index']} != {expected}")
        receipt["getter_counts"] = dict(method_counts)
    finally:
        for name, original in originals.items():
            setattr(cls, name, original)
        tensor_map = state["map"]
        if tensor_map is not None:
            tensor_map.flush()
            del tensor_map
        spool_path = state["path"]
        if spool_path is not None:
            spool_path.unlink(missing_ok=True)
            receipt["spool_file_removed"] = True
