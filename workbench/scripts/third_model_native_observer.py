"""Capture native FashionIQ score/rank calls made by a pinned test evaluator.

Import ``observe_test_numpy`` around one real ``test_module.test`` call, then
call ``write_native_scores`` with the same ordered query/gallery IDs. The proxy
returns NumPy's original results; it records the actual post-mask sort inputs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


import os


def atomic_npz(output: Path, **arrays: Any) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("wb") as handle:
        import numpy as np
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)

def atomic_json(output: Path, value: Any) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)

def write_forward_control_diagnostics(output: Path, reference: dict[str, Any],
                                      comparisons: dict[str, Any]) -> dict[str, bool]:
    """Persist exact Adapter repeat/batch deltas before strict acceptance checks."""
    import numpy as np

    checks: dict[str, Any] = {}
    for name, candidate in comparisons.items():
        baseline = reference[name]
        row: dict[str, Any] = {"shape": list(baseline.shape), "candidate_shape": list(candidate.shape),
            "dtype": str(baseline.dtype), "candidate_dtype": str(candidate.dtype),
            "exact_equal": bool(np.array_equal(baseline, candidate))}
        if baseline.shape == candidate.shape and baseline.dtype.kind in "fiu" and candidate.dtype.kind in "fiu":
            maximum = total = 0.0
            different = elements = 0
            for start in range(0, len(baseline), 32):
                delta = np.abs(baseline[start:start + 32].astype(np.float64, copy=False)
                               - candidate[start:start + 32].astype(np.float64, copy=False))
                maximum = max(maximum, float(delta.max(initial=0)))
                total += float(delta.sum(dtype=np.float64))
                different += int(np.count_nonzero(delta))
                elements += delta.size
            row.update(max_abs_delta=maximum, mean_abs_delta=total / elements if elements else 0.0,
                       different_elements=different, compared_elements=elements)
        checks[name] = row
    flags = {name: bool(row["exact_equal"]) for name, row in checks.items()}
    atomic_json(output, {"schema": "adapter-forward-controls/v1", "checks": checks,
                         "passed": all(flags.values())})
    return flags



class _NumpyObserver:
    def __init__(self, numpy: Any, expected_queries: int, model_id: str) -> None:
        self.numpy = numpy
        self.model_id = model_id
        self.expected_queries = expected_queries
        self.matmul_calls = 0
        self.sort_inputs: list[Any] = []
        self.sort_orders: list[Any] = []
        self.argsort_calls = 0
        self.identical_repeats = True

    def __getattr__(self, name: str) -> Any:
        return getattr(self.numpy, name)

    def matmul(self, left: Any, right: Any) -> Any:
        result = self.numpy.matmul(left, right)
        self.matmul_calls += 1
        return result

    def argsort(self, values: Any, *args: Any, **kwargs: Any) -> Any:
        order = self.numpy.argsort(values, *args, **kwargs)
        index = self.argsort_calls
        self.argsort_calls += 1
        if index < self.expected_queries:
            self.sort_inputs.append(-values.copy())
            self.sort_orders.append(order.copy())
        else:
            index -= self.expected_queries
            if index >= self.expected_queries:
                raise ValueError("native evaluator made more than two argsort passes")
            self.identical_repeats &= self.numpy.array_equal(self.sort_inputs[index], -values)
            self.identical_repeats &= self.numpy.array_equal(self.sort_orders[index], order)
        return order


def observe_test_numpy(test_module: Any, expected_queries: int, model_id: str) -> _NumpyObserver:
    """Swap only evaluator module's NumPy binding; caller restores it in ``finally``."""
    import numpy as np

    original = getattr(test_module, "np", None)
    if original is not np:
        raise ValueError("native evaluator must bind the active NumPy module as test_module.np")
    if expected_queries < 1:
        raise ValueError("native evaluator query count must be positive")
    observer = _NumpyObserver(np, expected_queries, model_id)
    test_module.np = observer
    return observer


def write_native_scores(observer: _NumpyObserver, output: Path, *, model_id: str, category: str,
                        query_ids: list[str], gallery_ids: list[str],
                        reference_positions: list[int], target_positions: list[int]) -> dict[str, Any]:
    """Seal full post-mask native score rows and the evaluator's actual argsort outputs."""
    import numpy as np

    count = len(query_ids)
    if (observer.matmul_calls != 1 or observer.argsort_calls != 2 * count
            or len(observer.sort_inputs) != count or len(observer.sort_orders) != count
            or not observer.identical_repeats):
        raise ValueError("native evaluator score/rank calls are incomplete or repeated passes differ")
    if not (len(reference_positions) == len(target_positions) == count):
        raise ValueError("native query IDs and reference/target positions are misaligned")
    scores = np.stack(observer.sort_inputs)
    orders = np.stack(observer.sort_orders)
    if scores.shape != orders.shape or scores.shape != (count, len(gallery_ids)):
        raise ValueError("native score/rank arrays do not match declared query and gallery order")
    if not np.isfinite(scores).all() or not np.array_equal(np.sort(orders, axis=1),
                                                          np.broadcast_to(np.arange(len(gallery_ids)), orders.shape)):
        raise ValueError("native score/rank arrays contain non-finite scores or invalid permutations")
    for row, reference in enumerate(reference_positions):
        if not 0 <= reference < len(gallery_ids):
            raise ValueError(f"native reference position out of range at query {row}")
        if orders[row, -1] != reference:
            raise ValueError(f"native reference mask is not the final ranked gallery at query {row}")
    atomic_npz(output, scores=scores, orders=orders,
               query_ids=np.asarray(query_ids), gallery_ids=np.asarray(gallery_ids),
               reference_positions=np.asarray(reference_positions, dtype=np.int32),
               target_positions=np.asarray(target_positions, dtype=np.int32))
    return {
        "schema": "native-source-test-scores/v1", "model_id": model_id, "category": category,
        "source": "pinned test_module.test np.matmul operands and post-mask np.argsort",
        "matmul_calls": observer.matmul_calls, "argsort_queries": count,
        "identical_repeat_passes": observer.identical_repeats,
        "query_count": count, "gallery_count": len(gallery_ids),
        "query_order_sha256": hashlib.sha256(("\n".join(query_ids) + "\n").encode()).hexdigest(),
        "gallery_order_sha256": hashlib.sha256(("\n".join(gallery_ids) + "\n").encode()).hexdigest(),
        "scores_shape": list(scores.shape), "scores_dtype": str(scores.dtype),
        "orders_shape": list(orders.shape), "orders_dtype": str(orders.dtype),
        "artifact": output.name,
    }

def self_check() -> int:
    import numpy as np
    from types import SimpleNamespace

    evaluator = SimpleNamespace(np=np)
    observer = observe_test_numpy(evaluator, 2, "test")
    try:
        scores = evaluator.np.matmul(np.eye(2, dtype=np.float32),
                                     np.arange(6, dtype=np.float32).reshape(2, 3).T)
        scores[0, 0] = -10e10
        scores[1, 1] = -10e10
        orders = [evaluator.np.argsort(-scores[row]) for row in range(2)]
        for row in range(2):
            evaluator.np.argsort(-scores[row])
    finally:
        evaluator.np = np
    receipt = write_native_scores(observer, Path("/tmp/native-observer-self-check.npz"),
                                  model_id="test", category="test", query_ids=["q0", "q1"],
                                  gallery_ids=["g0", "g1", "g2"],
                                  reference_positions=[0, 1], target_positions=[2, 2])
    with np.load("/tmp/native-observer-self-check.npz") as artifact:
        assert artifact["orders"].tolist() == np.stack(orders).tolist()
        assert artifact["scores"][0, 0] == np.float32(-10e10)
    assert receipt["matmul_calls"] == 1 and receipt["argsort_queries"] == 2
    Path("/tmp/native-observer-self-check.npz").unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(self_check())

