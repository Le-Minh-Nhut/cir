"""Observe native CLIP4Cir FashionIQ score and sorter calls without editing upstream source."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Callable


class _ArgsortCapture:
    def __init__(self, argsort: Callable[..., Any], callback: Callable[[Any, Any, tuple[Any, ...], dict[str, Any]], None]):
        self._argsort = argsort
        self._callback = callback

    def __call__(self, values: Any, *args: Any, **kwargs: Any) -> Any:
        order = self._argsort(values, *args, **kwargs)
        self._callback(values, order, args, kwargs)
        return order


def bind_fashioniq_rows(dataset: Any, scores: Any, order: Any, gallery_ids: Any) -> list[dict[str, Any]]:
    """Bind source distance/order rows to exact FashionIQ query and gallery IDs."""
    import numpy as np

    score_rows = scores.detach().cpu().numpy()
    order_rows = order.detach().cpu().numpy()
    triplets = dataset.triplets
    gallery_ids = list(gallery_ids)
    if len(triplets) != len(score_rows) or len(order_rows) != len(triplets):
        raise ValueError("CLIP4Cir native query capture does not match FashionIQ triplets")
    if score_rows.shape[1] != len(gallery_ids) or order_rows.shape[1] != len(gallery_ids):
        raise ValueError("CLIP4Cir native score/order capture does not match gallery IDs")
    category = dataset.dress_types[0]
    rows = []
    for index, triplet in enumerate(triplets):
        target = triplet["target"]
        reference = triplet["candidate"]
        if target not in gallery_ids:
            raise ValueError(f"{category}: CLIP4Cir target missing from native gallery: {target}")
        rows.append({
            "query_id": f"{category}#row{index}:{reference}:{target}",
            "target_id": target,
            "reference_id": reference,
            "distances": np.array(score_rows[index], copy=True),
            "native_order": np.array(order_rows[index], dtype=np.int64, copy=True),
        })
    return rows
def run_native_capture(source_root: Path, argv: list[str], capture: Callable[..., None],
                       dataset_root: Path | None = None) -> None:
    """Run pinned validate.py main; record exact source argsort inputs and returned order."""
    source_root = source_root.resolve()
    source_dir = source_root / "src"
    sys.path.insert(0, str(source_dir))
    try:
        if dataset_root is not None:
            data_utils = importlib.import_module("data_utils")
            setattr(data_utils, "base_path", dataset_root.resolve())
        validate = importlib.import_module("validate")
        original_main = sys.argv
        original_argsort = validate.torch.argsort
        original_compute = validate.compute_fiq_val_metrics

        def observed_compute(dataset: Any, clip_model: Any, index_features: Any,
                             index_names: Any, combining_function: Any) -> Any:
            argsort = _ArgsortCapture(
                original_argsort,
                lambda scores, order, args, kwargs: capture(
                    dataset, scores, order, index_names, args, kwargs))
            validate.torch.argsort = argsort
            try:
                return original_compute(dataset, clip_model, index_features, index_names,
                                        combining_function)
            finally:
                validate.torch.argsort = original_argsort

        setattr(validate, "compute_fiq_val_metrics", observed_compute)
        sys.argv = [str(source_dir / "validate.py"), *argv]
        try:
            validate.main()
        finally:
            sys.argv = original_main
            setattr(validate, "compute_fiq_val_metrics", original_compute)
            validate.torch.argsort = original_argsort
            sys.modules.pop("validate", None)
    finally:
        sys.path.remove(str(source_dir))
